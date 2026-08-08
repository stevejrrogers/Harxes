//! Multi-turn agent loop orchestrating repeated LLM turns, executing requested
//! tool calls (bash/read/write) and feeding results back until it stops.

use std::sync::Arc;

use harxes_core_domain::domain::value_objects::{Message, ProviderId, Role, ToolCall};
use harxes_core_domain::ports::{FileSystemPort, LlmError, LlmPort};

/// Guardrail configuration for the agent loop.
#[derive(Debug, Clone)]
pub struct LoopLimits {
    pub max_iterations: usize,
    pub max_total_tokens: u64,
    /// Per-call transcript cap used by the context manager to trim older turns.
    pub context_window_tokens: u64,
}

impl Default for LoopLimits {
    fn default() -> Self {
        Self {
            max_iterations: 25,
            max_total_tokens: 128_000,
            context_window_tokens: 96_000,
        }
    }
}

/// Outcome of an agent run.
#[derive(Debug)]
pub struct LoopOutcome {
    pub final_text: String,
    pub iterations: usize,
    pub usage_total_tokens: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub truncated_by_guardrail: bool,
}

/// Application service driving the tool-calling agent loop.
pub struct AgentLoop {
    llm: Arc<dyn LlmPort>,
    shell: Arc<dyn harxes_core_domain::ports::ShellPort>,
    fsys: Arc<dyn FileSystemPort>,
    policy: Option<harxes_core_domain::domain::value_objects::PermissionPolicy>,
    session: Option<(Arc<dyn harxes_core_domain::ports::SessionStorePort>, String)>,
    observer: Option<Arc<dyn harxes_core_domain::ports::ToolObserver>>,
}

impl AgentLoop {
    pub fn new(
        llm: Arc<dyn LlmPort>,
        shell: Arc<dyn harxes_core_domain::ports::ShellPort>,
        fsys: Arc<dyn FileSystemPort>,
    ) -> Self {
        Self {
            llm,
            shell,
            fsys,
            policy: None,
            session: None,
            observer: None,
        }
    }

    /// Attach a file-permission policy gating mutating tool operations.
    pub fn with_policy(
        mut self,
        p: harxes_core_domain::domain::value_objects::PermissionPolicy,
    ) -> Self {
        self.policy = Some(p);
        self
    }

    /// Attach a session store so transcripts are persisted on completion.
    pub fn with_session(
        mut self,
        store: Arc<dyn harxes_core_domain::ports::SessionStorePort>,
        id: impl Into<String>,
    ) -> Self {
        self.session = Some((store, id.into()));
        self
    }

    /// Attach an observer for live tool-execution feedback.
    pub fn with_observer(mut self, o: Arc<dyn harxes_core_domain::ports::ToolObserver>) -> Self {
        self.observer = Some(o);
        self
    }

    fn tool_specs() -> Vec<harxes_core_domain::domain::value_objects::ToolSpec> {
        vec![
            harxes_core_domain::domain::value_objects::ToolSpec::new(
                "Bash",
                "Run a shell command and return stdout/stderr/exit code.",
            ),
            harxes_core_domain::domain::value_objects::ToolSpec::new(
                "Read",
                "Read a file from disk and return its contents.",
            ),
            harxes_core_domain::domain::value_objects::ToolSpec::new(
                "Write",
                "Write content to a file on disk.",
            ),
        ]
    }

    async fn execute_call(&self, call: &ToolCall) -> String {
        use harxes_core_domain::domain::services::tool_protocol::{parse_args, ParsedArgs, ToolId};
        if let Some(obs) = &self.observer {
            obs.on_tool_start(
                call.name.as_str(),
                &Self::preview(call.arguments.as_str(), 80),
            );
        }
        let result = match ToolId::parse(call.name.as_str()) {
            Some(tool) => match parse_args(tool, &call.arguments) {
                ParsedArgs::Bash { command } => self.run_bash(&command).await,
                ParsedArgs::Read { path } => self.read_file(&path).await,
                ParsedArgs::Write { path, content } => self.write_file_parts(&path, &content).await,
            },
            None => format!("unknown tool '{}'", call.name),
        };
        if let Some(obs) = &self.observer {
            obs.on_tool_result(call.name.as_str(), &Self::preview(result.as_str(), 100));
        }
        result
    }

    fn preview(raw: &str, max: usize) -> String {
        let mut a = raw.chars().take(max).collect::<String>();
        if raw.len() > max {
            a.push_str("...");
        }
        a.replace('\n', " ")
    }

    async fn run_bash(&self, cmd: &str) -> String {
        use harxes_core_domain::ports::ShellExitStatus;
        match self.shell.run_command(".", cmd).await {
            Ok(o) => {
                let code = match o.exit_status {
                    ShellExitStatus::Success => 0,
                    ShellExitStatus::Failure(c) => c,
                };
                format!(
                    "exit={} stdout={} stderr={}",
                    code,
                    o.stdout.trim(),
                    o.stderr.trim()
                )
            }
            Err(e) => format!("shell error {e}"),
        }
    }

    async fn read_file(&self, path: &str) -> String {
        match self.fsys.read(path.trim()).await {
            Ok(c) => c,
            Err(e) => format!("read error {e}"),
        }
    }

    async fn write_file_parts(&self, path: &str, content: &str) -> String {
        if path.is_empty() {
            return "write error: empty path".to_string();
        }
        use harxes_core_domain::domain::value_objects::FsOp;
        if let Some(policy) = &self.policy {
            if !policy.permits(FsOp::Write, path) {
                return format!(
                    "permission required: writing {} is not in the allow list; no action taken",
                    path
                );
            }
        }
        match self.fsys.write(path, content).await {
            Ok(()) => format!("wrote {} bytes to {}", content.len(), path),
            Err(e) => format!("write error {e}"),
        }
    }

    /// Shared loop body: drive tool-calling turns over an existing transcript
    /// until a text-only turn or guardrails cut it off. Returns both the outcome
    /// and the final transcript so callers can carry conversation state.
    async fn run_loop(
        &self,
        provider_id: &ProviderId,
        model_id: &str,
        mut transcript: Vec<Message>,
        limits: &LoopLimits,
    ) -> Result<(LoopOutcome, Vec<Message>), LlmError> {
        let tools = Self::tool_specs();
        let mut iterations = 0usize;
        let mut total_tokens = 0u64;
        let mut total_input = 0u64;
        let mut total_output = 0u64;
        let mut truncated = false;

        loop {
            if iterations >= limits.max_iterations {
                truncated = true;
                break;
            }
            // Keep the transcript within the per-call window budget.
            use harxes_core_domain::domain::value_objects::trim_to_budget;
            if !transcript.is_empty() {
                transcript = trim_to_budget(
                    std::mem::take(&mut transcript),
                    limits.context_window_tokens,
                );
            }
            let resp = self
                .llm
                .generate(provider_id, model_id, &transcript, &tools, None)
                .await?;
            total_tokens += resp.usage.total_tokens;
            total_input += resp.usage.input_tokens;
            total_output += resp.usage.output_tokens;
            if total_tokens > limits.max_total_tokens {
                truncated = true;
                break;
            }
            iterations += 1;

            if resp.tool_calls.is_empty() {
                self.persist_session(&transcript);
                let out = LoopOutcome {
                    final_text: resp.content.clone(),
                    iterations,
                    usage_total_tokens: total_tokens,
                    input_tokens: total_input,
                    output_tokens: total_output,
                    truncated_by_guardrail: truncated,
                };
                return Ok((out, transcript));
            }

            // Feed tool calls + results back into the transcript.
            transcript.push(Message::assistant_with_tools(resp.tool_calls.clone()));
            for call in resp.tool_calls.iter() {
                let output = self.execute_call(call).await;
                transcript.push(Message::tool_result(call.id.clone(), output));
            }
        }

        self.persist_session(&transcript);
        Ok((
            LoopOutcome {
                final_text: "[stopped by guardrail]".to_string(),
                iterations,
                usage_total_tokens: total_tokens,
                input_tokens: total_input,
                output_tokens: total_output,
                truncated_by_guardrail: truncated,
            },
            transcript,
        ))
    }

    /// Run a fresh session from a system + user prompt.
    pub async fn run(
        &self,
        provider_id: &ProviderId,
        model_id: &str,
        system_prompt: &str,
        user_prompt: &str,
        limits: &LoopLimits,
    ) -> Result<LoopOutcome, LlmError> {
        let transcript = vec![
            Message::new(Role::System, system_prompt.to_string()),
            Message::new(Role::User, user_prompt.to_string()),
        ];
        self.run_loop(provider_id, model_id, transcript, limits)
            .await
            .map(|(o, _)| o)
    }

    fn persist_session(&self, transcript: &[Message]) {
        if let Some((store, id)) = &self.session {
            use harxes_core_domain::ports::SessionRecord;
            let rec = SessionRecord {
                id: id.clone(),
                created_at: String::new(),
                transcript: transcript.to_vec(),
            };
            let _ = store.save(&rec);
        }
    }
}

#[async_trait::async_trait]
impl crate::ports::AgentPort for AgentLoop {
    async fn run(
        &self,
        provider_id: &ProviderId,
        model_id: &str,
        system_prompt: &str,
        user_prompt: &str,
        limits: &LoopLimits,
    ) -> Result<LoopOutcome, LlmError> {
        self.run(provider_id, model_id, system_prompt, user_prompt, limits)
            .await
    }

    async fn continue_chat(
        &self,
        provider_id: &ProviderId,
        model_id: &str,
        history: &[Message],
        user_prompt: &str,
        limits: &LoopLimits,
    ) -> Result<crate::ports::ConversationResult, LlmError> {
        let mut transcript = history.to_vec();
        transcript.push(Message::new(Role::User, user_prompt.to_string()));
        let (outcome, final_transcript) = self
            .run_loop(provider_id, model_id, transcript, limits)
            .await?;
        Ok(crate::ports::ConversationResult {
            outcome,
            transcript: final_transcript,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use harxes_core_domain::domain::value_objects::{ToolCall, ToolSpec};
    use harxes_core_domain::ports::{
        AgentResponse, CommandOutput, FileSystemPort, FsError, ShellError, ShellExitStatus,
    };
    use std::sync::{Arc, Mutex};

    /// Scripted fake LLM: pops from a queue of AgentResponses.
    struct FakeLlmScript(Mutex<Vec<AgentResponse>>);

    #[async_trait::async_trait]
    impl LlmPort for FakeLlmScript {
        async fn generate(
            &self,
            _p: &ProviderId,
            _m: &str,
            messages: &[Message],
            tools: &[ToolSpec],
            _t: Option<f64>,
        ) -> Result<AgentResponse, LlmError> {
            let mut q = self.0.lock().unwrap();
            if q.is_empty() {
                return Ok(AgentResponse::text("done", Default::default()));
            }
            let resp = q.remove(0);
            // capture tool spec count into content? no—return as-is.
            let _ = tools;
            let _ = messages;
            Ok(resp)
        }
    }

    struct FakeShell;
    #[async_trait::async_trait]
    impl harxes_core_domain::ports::ShellPort for FakeShell {
        async fn run_command(&self, _wd: &str, cmd: &str) -> Result<CommandOutput, ShellError> {
            Ok(CommandOutput {
                stdout: format!("out:{cmd}"),
                stderr: String::new(),
                exit_status: ShellExitStatus::Success,
            })
        }
    }

    struct FakeFs;
    #[async_trait::async_trait]
    impl FileSystemPort for FakeFs {
        async fn read(&self, p: &str) -> Result<String, FsError> {
            Ok(format!("read:{p}"))
        }
        async fn write(&self, p: &str, c: &str) -> Result<(), FsError> {
            let _ = (p, c);
            Ok(())
        }
    }

    fn agent(script: Vec<AgentResponse>) -> AgentLoop {
        AgentLoop::new(
            Arc::new(FakeLlmScript(Mutex::new(script))),
            Arc::new(FakeShell),
            Arc::new(FakeFs),
        )
    }

    #[tokio::test]
    async fn loop_stops_on_text_only_first_turn() {
        let a = agent(vec![]);
        let pid = ProviderId::new("x").unwrap();
        let out = a
            .run(&pid, "m", "sys", "hi", &LoopLimits::default())
            .await
            .unwrap();
        assert_eq!(out.iterations, 1);
        assert!(!out.truncated_by_guardrail);
    }

    #[tokio::test]
    async fn loop_executes_tool_then_text() {
        // turn 1: request a Bash call; turn 2: plain text
        let bash = ToolCall {
            id: "c1".into(),
            name: "Bash".into(),
            arguments: "echo hi".into(),
        };
        let script = vec![
            AgentResponse {
                content: String::new(),
                usage: Default::default(),
                tool_calls: vec![bash],
            },
            AgentResponse::text("all done", Default::default()),
        ];
        let a = agent(script);
        let pid = ProviderId::new("x").unwrap();
        let out = a
            .run(
                &pid,
                "m",
                "sys",
                "do it",
                &LoopLimits {
                    max_iterations: 25,
                    max_total_tokens: 9999,
                    context_window_tokens: 5000,
                },
            )
            .await
            .unwrap();
        assert_eq!(out.iterations, 2);
        assert_eq!(out.final_text, "all done");
    }

    #[tokio::test]
    async fn loop_truncates_on_iteration_limit() {
        // always returns a tool call -> never stops by itself
        let tc = || ToolCall {
            id: "c1".into(),
            name: "Read".into(),
            arguments: "a.txt".into(),
        };
        let resp = AgentResponse {
            content: String::new(),
            usage: Default::default(),
            tool_calls: vec![tc()],
        };
        // script returns the same response regardless by being long; instead use empty queue => but empty returns text.
        // Force iteration limit by giving many copies then rely on limit.
        let mut script = Vec::new();
        for _ in 0..30 {
            script.push(resp.clone());
        }
        let a = agent(script);
        let pid = ProviderId::new("x").unwrap();
        let out = a
            .run(
                &pid,
                "m",
                "sys",
                "loop forever",
                &LoopLimits {
                    max_iterations: 5,
                    max_total_tokens: 9999,
                    context_window_tokens: 5000,
                },
            )
            .await
            .unwrap();
        assert!(out.truncated_by_guardrail);
        assert_eq!(out.iterations, 5);
    }

    #[tokio::test]
    async fn write_blocked_by_policy() {
        use harxes_core_domain::domain::value_objects::PermissionPolicy;
        let shell = Arc::new(FakeShell);
        let fsys = Arc::new(FakeFs);
        let llm = Arc::new(FakeLlmScript(Mutex::new(vec![])));
        let a = AgentLoop::new(llm, shell, fsys).with_policy(PermissionPolicy {
            allow_globs: vec!["/ok/**".to_string()],
            deny_globs: vec![],
        });

        let out = a.write_file_parts("bad.txt", "secret").await;
        assert!(out.contains("permission required"));

        let ok = a.write_file_parts("/ok/a.txt", "hi").await;
        assert!(ok.starts_with("wrote"));
    }

    #[tokio::test]
    async fn persists_session_on_completion() {
        use harxes_core_domain::ports::{SessionRecord, SessionStorePort};
        use std::sync::{Arc, Mutex};

        struct MemStore(Mutex<Vec<SessionRecord>>);
        impl SessionStorePort for MemStore {
            fn save(
                &self,
                s: &SessionRecord,
            ) -> Result<(), harxes_core_domain::ports::SessionStoreError> {
                self.0.lock().unwrap().push(s.clone());
                Ok(())
            }
            fn load(&self, _id: &str) -> Option<SessionRecord> {
                None
            }
        }

        let store = Arc::new(MemStore(Mutex::new(vec![])));
        let llm = Arc::new(FakeLlmScript(Mutex::new(vec![])));
        let a = AgentLoop::new(llm, Arc::new(FakeShell), Arc::new(FakeFs))
            .with_session(store.clone(), "sess-1");

        let pid = ProviderId::new("x").unwrap();
        a.run(&pid, "m", "sys", "hello", &LoopLimits::default())
            .await
            .unwrap();
        assert_eq!(store.0.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn continue_chat_appends_user_turn() {
        use crate::ports::AgentPort;
        let a = agent(vec![]);
        let pid = ProviderId::new("x").unwrap();
        let history = vec![
            Message::new(Role::System, "sys"),
            Message::new(Role::User, "first"),
        ];
        let res = a
            .continue_chat(&pid, "m", &history, "second", &LoopLimits::default())
            .await
            .unwrap();
        assert_eq!(res.outcome.iterations, 1);
        // transcript should include system + first user + second user + assistant reply
        assert!(res.transcript.iter().any(|m| m.content == "second"));
    }
}
