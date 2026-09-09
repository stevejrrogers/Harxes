//! # harxes-core — embeddable Harxes agent engine
//!
//! An in-process, headless agent runner with no CLI, TUI, or TTY dependency.
//! A host process (e.g. CoXAgent) links this crate, constructs a
//! [`HarxesEngine`] once, and calls [`HarxesEngine::run`] per task. Each run is:
//!
//! - **Multi-instance & `Send + Sync`** — no global state; run many engines and
//!   many concurrent runs in one process.
//! - **Cancel-safe** — dropping the run's driver future aborts the LLM request
//!   and SIGKILLs the process group of every in-flight Bash child (the shell
//!   adapter spawns children in their own group with kill-on-drop). See the
//!   `drop_mid_run_leaves_no_process` test.
//! - **Wall-clock bounded** — [`RunRequest::timeout`] caps the run; on expiry
//!   the driver future is dropped (killing children) and the outcome is
//!   [`EngineError::Timeout`].
//! - **Headless-safe permissions** — no interactive prompt ever; risky
//!   operations resolve against [`PermissionMode`] and fail closed by default.
//! - **Sandbox passthrough** — the host may inject its own confined
//!   [`ShellPort`]/[`FileSystemPort`] via [`EngineConfig`]; the engine never
//!   picks a sandbox policy itself.

use std::sync::Arc;
use std::time::Duration;

use harxes_app::usecases::agent_loop::{AgentLoop, LoopLimits, LoopOutcome};
use harxes_core_domain::domain::value_objects::{
    CommandPolicy, Message, ProviderId, Role,
};
use harxes_core_domain::ports::{
    FileSystemPort, LlmError, LlmPort, PermissionDecider, ShellPort, ToolObserver,
};

pub use harxes_core_domain::domain::value_objects::Message as ChatMessage;

/// Which model backend to talk to. Credentials are passed explicitly — the
/// engine never reads process environment or a keychain.
#[derive(Debug, Clone)]
pub enum ProviderSpec {
    /// Anthropic Messages API. `base_url` is the full endpoint.
    Anthropic { base_url: String, api_key: String },
    /// Any OpenAI-compatible Chat Completions endpoint (OpenAI, LiteLLM,
    /// local servers). `base_url` may be an API root (`.../v1`) or a full URL.
    OpenAiCompatible { base_url: String, api_key: String },
}

/// How the engine resolves an operation the command policy marks "ask". There
/// is no TTY, so it can never block on a human — it resolves immediately.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PermissionMode {
    /// Deny anything not explicitly allowed by policy (safest; the default).
    #[default]
    DenyUnlessAllowed,
    /// Allow operations the policy didn't explicitly deny (trust the sandbox
    /// to contain them). Use only when the host confines the shell/fs.
    AllowUnlessDenied,
}

/// Everything needed to build an engine. Reusable across many runs.
pub struct EngineConfig {
    pub provider: ProviderSpec,
    /// Default model id; overridable per-run via [`RunRequest::model`].
    pub model: String,
    pub limits: LoopLimits,
    /// Shell allow/deny rules applied to Bash commands.
    pub command_policy: CommandPolicy,
    pub permission: PermissionMode,
    /// Host-provided confined shell (sandbox passthrough). Defaults to a plain
    /// process-group shell with kill-on-drop when `None`.
    pub shell: Option<Arc<dyn ShellPort>>,
    /// Host-provided filesystem (e.g. a workdir jail). Defaults to the host FS.
    pub fs: Option<Arc<dyn FileSystemPort>>,
}

impl EngineConfig {
    /// Minimal config for an OpenAI-compatible endpoint with safe defaults.
    pub fn openai(base_url: impl Into<String>, api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            provider: ProviderSpec::OpenAiCompatible {
                base_url: base_url.into(),
                api_key: api_key.into(),
            },
            model: model.into(),
            limits: LoopLimits::default(),
            command_policy: CommandPolicy::default(),
            permission: PermissionMode::default(),
            shell: None,
            fs: None,
        }
    }
}

/// One task to run. Stateless with respect to the engine.
#[derive(Debug, Clone, Default)]
pub struct RunRequest {
    /// The user instruction for this run.
    pub prompt: String,
    /// System prompt. When `None`, the caller is expected to supply guidance in
    /// `history` or `prompt`; the engine adds none of its own.
    pub system_prompt: Option<String>,
    /// Prior conversation to continue from (e.g. a resumed ticket). The new
    /// `prompt` is appended as a user turn.
    pub history: Vec<Message>,
    /// Wall-clock budget for the whole run. `None` = no engine-imposed limit
    /// (the loop's own iteration/token guardrails still apply).
    pub timeout: Option<Duration>,
    /// Per-run model override; falls back to [`EngineConfig::model`].
    pub model: Option<String>,
}

/// A live event emitted while a run executes. Consumed off the event channel.
#[derive(Debug, Clone)]
pub enum RunEvent {
    /// A chunk of streamed assistant text.
    Text(String),
    /// A chunk of streamed hidden reasoning ("thinking").
    Reasoning(String),
    /// A tool is about to run. `summary` is a clean one-line label.
    ToolStart { name: String, summary: String },
    /// A tool finished; `summary` previews its result.
    ToolEnd { name: String, summary: String },
    /// A transient failure triggered a backoff before retrying.
    Retry { wait_secs: u64 },
}

/// Why a run stopped. Machine-readable for the host's classifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// The model finished normally.
    Done,
    /// A loop guardrail (iteration or token cap) cut the run short.
    Guardrail,
}

/// The structured result of a completed run.
#[derive(Debug, Clone)]
pub struct RunOutcome {
    pub final_text: String,
    /// Full machine-readable transcript (system/user/assistant/tool turns).
    pub transcript: Vec<Message>,
    pub iterations: usize,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
    pub stop_reason: StopReason,
}

impl RunOutcome {
    fn from_loop(o: LoopOutcome, transcript: Vec<Message>) -> Self {
        Self {
            final_text: o.final_text,
            transcript,
            iterations: o.iterations,
            input_tokens: o.input_tokens,
            output_tokens: o.output_tokens,
            total_tokens: o.usage_total_tokens,
            stop_reason: if o.truncated_by_guardrail {
                StopReason::Guardrail
            } else {
                StopReason::Done
            },
        }
    }
}

/// Typed failure taxonomy for a run. The host classifies infra faults against
/// these variants instead of grepping error strings.
#[derive(Debug, Clone)]
pub enum EngineError {
    /// The wall-clock budget was exceeded; children were killed.
    Timeout,
    /// The provider was unreachable or returned a transport/5xx/rate-limit
    /// failure after retries — retrying the whole task elsewhere is sane.
    ProviderUnavailable(String),
    /// Authentication failed for the provider — the credential is broken.
    AuthDead,
    /// The task itself failed (a run-level error that isn't infra).
    TaskFailed(String),
    /// The engine was misconfigured (bad base_url, empty model, etc.).
    Config(String),
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Timeout => write!(f, "run exceeded its wall-clock budget"),
            Self::ProviderUnavailable(m) => write!(f, "provider unavailable: {m}"),
            Self::AuthDead => write!(f, "provider authentication failed"),
            Self::TaskFailed(m) => write!(f, "task failed: {m}"),
            Self::Config(m) => write!(f, "engine misconfigured: {m}"),
        }
    }
}
impl std::error::Error for EngineError {}

impl EngineError {
    /// True when this is an infrastructure fault (worth retrying the task on
    /// different capacity), as opposed to a genuine task failure.
    pub fn is_infra_fault(&self) -> bool {
        matches!(
            self,
            EngineError::Timeout | EngineError::ProviderUnavailable(_) | EngineError::AuthDead
        )
    }

    fn from_llm(e: LlmError) -> Self {
        match e {
            LlmError::Auth { .. } => EngineError::AuthDead,
            LlmError::RateLimited { .. } => {
                EngineError::ProviderUnavailable("rate limited".into())
            }
            LlmError::Timeout => EngineError::ProviderUnavailable("provider timeout".into()),
            LlmError::Request(m) => EngineError::ProviderUnavailable(m),
        }
    }
}

/// Fail-closed / fail-open permission decider driven by [`PermissionMode`].
/// Never blocks — there is no interactive prompt in a headless engine.
struct HeadlessDecider {
    allow: bool,
}
impl PermissionDecider for HeadlessDecider {
    fn decide_write(&self, _path: &str) -> bool {
        self.allow
    }
    fn decide_bash(&self, _command: &str) -> bool {
        self.allow
    }
}

/// Bridges the loop's [`ToolObserver`] callbacks onto the run's event channel.
struct EventObserver {
    tx: tokio::sync::mpsc::UnboundedSender<RunEvent>,
}
impl ToolObserver for EventObserver {
    fn on_tool_start(&self, name: &str, args_preview: &str) {
        let _ = self.tx.send(RunEvent::ToolStart {
            name: name.to_string(),
            summary: args_preview.to_string(),
        });
    }
    fn on_tool_result(&self, name: &str, result_preview: &str) {
        let _ = self.tx.send(RunEvent::ToolEnd {
            name: name.to_string(),
            summary: result_preview.to_string(),
        });
    }
    fn on_retry(&self, wait_secs: u64) {
        let _ = self.tx.send(RunEvent::Retry { wait_secs });
    }
    fn on_reasoning(&self, text: &str) {
        let _ = self.tx.send(RunEvent::Reasoning(text.to_string()));
    }
    fn on_stream_delta(&self, text: &str) {
        let _ = self.tx.send(RunEvent::Text(text.to_string()));
    }
}

/// The cancel-safe future that drives a run to its outcome. Dropping it aborts
/// the run and kills in-flight child processes.
pub type RunDriver =
    std::pin::Pin<Box<dyn std::future::Future<Output = Result<RunOutcome, EngineError>> + Send>>;
/// Receiver for a run's live [`RunEvent`] stream.
pub type EventStream = tokio::sync::mpsc::UnboundedReceiver<RunEvent>;

/// A running task: an event stream plus a cancel-safe driver future.
///
/// Read [`RunHandle::events`] while polling the driver from [`RunHandle::split`]
/// with `tokio::select!`. **Dropping the driver future aborts the run** and
/// kills any in-flight child processes — that is the cancellation contract.
pub struct RunHandle {
    events: EventStream,
    driver: RunDriver,
}

impl RunHandle {
    /// Split into the event receiver and the driver future so the host can
    /// `select!` over both. Dropping the returned future cancels the run.
    pub fn split(self) -> (EventStream, RunDriver) {
        (self.events, self.driver)
    }

    /// Convenience: await the outcome, draining events in the background.
    /// Prefer [`RunHandle::split`] when you need the live event stream.
    pub async fn wait(self) -> Result<RunOutcome, EngineError> {
        let (mut events, driver) = self.split();
        tokio::pin!(driver);
        loop {
            tokio::select! {
                _ = events.recv() => {}
                out = &mut driver => return out,
            }
        }
    }
}

/// The embeddable engine. Cheap to clone the handle; build once, run many.
#[derive(Clone)]
pub struct HarxesEngine {
    llm: Arc<dyn LlmPort>,
    shell: Arc<dyn ShellPort>,
    fs: Arc<dyn FileSystemPort>,
    provider_id: ProviderId,
    model: String,
    limits: LoopLimits,
    command_policy: CommandPolicy,
    permission: PermissionMode,
}

impl HarxesEngine {
    /// Build an engine from config. Returns [`EngineError::Config`] on bad input.
    pub fn new(config: EngineConfig) -> Result<Self, EngineError> {
        if config.model.trim().is_empty() {
            return Err(EngineError::Config("model must not be empty".into()));
        }
        let (llm, provider_name): (Arc<dyn LlmPort>, &str) = match &config.provider {
            ProviderSpec::Anthropic { base_url, api_key } => {
                if base_url.trim().is_empty() {
                    return Err(EngineError::Config("empty base_url".into()));
                }
                (
                    Arc::new(harxes_infra_llm::providers::anthropic::AnthropicClient::new(
                        base_url.clone(),
                        api_key.clone(),
                    )),
                    "anthropic",
                )
            }
            ProviderSpec::OpenAiCompatible { base_url, api_key } => {
                if base_url.trim().is_empty() {
                    return Err(EngineError::Config("empty base_url".into()));
                }
                (
                    Arc::new(harxes_infra_llm::providers::openai::OpenAiClient::new(
                        base_url.clone(),
                        api_key.clone(),
                    )),
                    "openai",
                )
            }
        };
        let provider_id = ProviderId::new(provider_name)
            .map_err(|_| EngineError::Config("invalid provider id".into()))?;
        let shell = config
            .shell
            .unwrap_or_else(|| Arc::new(harxes_infra_shell::TokioCommandShell::new(600)));
        let fs = config
            .fs
            .unwrap_or_else(|| Arc::new(harxes_infra_fs::HostFileSystem));
        Ok(Self {
            llm,
            shell,
            fs,
            provider_id,
            model: config.model,
            limits: config.limits,
            command_policy: config.command_policy,
            permission: config.permission,
        })
    }

    /// Start a run. Returns immediately with a [`RunHandle`]; work happens as
    /// the driver future is polled.
    pub fn run(&self, req: RunRequest) -> RunHandle {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<RunEvent>();
        let decider = Arc::new(HeadlessDecider {
            allow: self.permission == PermissionMode::AllowUnlessDenied,
        });
        let agent = AgentLoop::new(self.llm.clone(), self.shell.clone(), self.fs.clone())
            .with_streaming(true)
            .with_observer(Arc::new(EventObserver { tx }))
            .with_decider(decider)
            .with_command_policy(self.command_policy.clone());

        let provider_id = self.provider_id.clone();
        let model = req.model.clone().unwrap_or_else(|| self.model.clone());
        let limits = self.limits.clone();
        let timeout = req.timeout;

        // Build the transcript: optional system + prior history + new user turn.
        let mut transcript = Vec::new();
        if let Some(sys) = req.system_prompt {
            transcript.push(Message::new(Role::System, sys));
        }
        transcript.extend(req.history);
        transcript.push(Message::new(Role::User, req.prompt));

        let driver = Box::pin(async move {
            let work = async move {
                let (outcome, final_transcript) = agent
                    .run_transcript(&provider_id, &model, transcript, &limits)
                    .await
                    .map_err(EngineError::from_llm)?;
                Ok(RunOutcome::from_loop(outcome, final_transcript))
            };
            match timeout {
                // On elapse the inner `work` future is dropped here, which drops
                // the AgentLoop run future and (via kill-on-drop) SIGKILLs any
                // in-flight Bash process group.
                Some(d) => match tokio::time::timeout(d, work).await {
                    Ok(r) => r,
                    Err(_) => Err(EngineError::Timeout),
                },
                None => work.await,
            }
        });

        RunHandle { events: rx, driver }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use harxes_core_domain::domain::value_objects::ToolSpec;
    use harxes_core_domain::ports::{AgentResponse, FsError, GlobOptions, GrepMatch};
    use std::sync::atomic::{AtomicUsize, Ordering};

    // ---- Minimal fakes -----------------------------------------------------
    struct EchoLlm;
    #[async_trait::async_trait]
    impl LlmPort for EchoLlm {
        async fn generate(
            &self,
            _p: &ProviderId,
            _m: &str,
            _msgs: &[Message],
            _t: &[ToolSpec],
            _temp: Option<f64>,
        ) -> Result<AgentResponse, LlmError> {
            Ok(AgentResponse::text("done", Default::default()))
        }
    }

    struct DeadFs;
    #[async_trait::async_trait]
    impl FileSystemPort for DeadFs {
        async fn read(&self, _p: &str) -> Result<String, FsError> {
            Ok(String::new())
        }
        async fn write(&self, _p: &str, _c: &str) -> Result<(), FsError> {
            Ok(())
        }
        async fn glob(&self, _pat: &str, _o: &GlobOptions) -> Vec<String> {
            vec![]
        }
        async fn grep(&self, _n: &str, _p: &str, _m: usize) -> Result<Vec<GrepMatch>, FsError> {
            Ok(vec![])
        }
        async fn replace(&self, _p: &str, _o: &str, _n: &str) -> Result<usize, FsError> {
            Ok(0)
        }
    }

    fn engine_with(llm: Arc<dyn LlmPort>, shell: Arc<dyn ShellPort>) -> HarxesEngine {
        HarxesEngine {
            llm,
            shell,
            fs: Arc::new(DeadFs),
            provider_id: ProviderId::new("x").unwrap(),
            model: "m".into(),
            limits: LoopLimits::default(),
            command_policy: CommandPolicy::default(),
            permission: PermissionMode::AllowUnlessDenied,
        }
    }

    #[tokio::test]
    async fn simple_run_completes_with_outcome() {
        let e = engine_with(Arc::new(EchoLlm), Arc::new(harxes_infra_shell::TokioCommandShell::new(5)));
        let out = e
            .run(RunRequest {
                prompt: "hi".into(),
                ..Default::default()
            })
            .wait()
            .await
            .unwrap();
        assert_eq!(out.final_text, "done");
        assert_eq!(out.stop_reason, StopReason::Done);
        // transcript: user + assistant (no system supplied).
        assert!(out.transcript.iter().any(|m| m.role == Role::User));
    }

    #[tokio::test]
    async fn timeout_returns_typed_timeout() {
        // An LLM that never returns starves the run past its budget.
        struct HangLlm;
        #[async_trait::async_trait]
        impl LlmPort for HangLlm {
            async fn generate(
                &self,
                _p: &ProviderId,
                _m: &str,
                _msgs: &[Message],
                _t: &[ToolSpec],
                _temp: Option<f64>,
            ) -> Result<AgentResponse, LlmError> {
                futures_never().await
            }
        }
        async fn futures_never() -> Result<AgentResponse, LlmError> {
            loop {
                tokio::time::sleep(Duration::from_secs(3600)).await;
            }
        }
        let e = engine_with(Arc::new(HangLlm), Arc::new(harxes_infra_shell::TokioCommandShell::new(5)));
        let err = e
            .run(RunRequest {
                prompt: "hi".into(),
                timeout: Some(Duration::from_millis(200)),
                ..Default::default()
            })
            .wait()
            .await
            .unwrap_err();
        assert!(matches!(err, EngineError::Timeout));
        assert!(err.is_infra_fault());
    }

    /// The core cancellation guarantee: an LLM that asks to run a long Bash
    /// sleep, then dropping the run mid-flight, must leave no child process.
    #[tokio::test]
    async fn drop_mid_run_leaves_no_process() {
        // LLM: first turn requests a long sleep via Bash; would never reach a
        // second turn because we drop the run before the sleep returns.
        struct SleepLlm(AtomicUsize);
        #[async_trait::async_trait]
        impl LlmPort for SleepLlm {
            async fn generate(
                &self,
                _p: &ProviderId,
                _m: &str,
                _msgs: &[Message],
                _t: &[ToolSpec],
                _temp: Option<f64>,
            ) -> Result<AgentResponse, LlmError> {
                use harxes_core_domain::domain::value_objects::ToolCall;
                let n = self.0.fetch_add(1, Ordering::SeqCst);
                if n == 0 {
                    Ok(AgentResponse {
                        content: String::new(),
                        usage: Default::default(),
                        tool_calls: vec![ToolCall {
                            id: "1".into(),
                            name: "Bash".into(),
                            // A uniquely-named marker so we can grep for it.
                            arguments: r#"{"command":"sleep 240 # harxes_cancel_probe_zzz"}"#
                                .into(),
                        }],
                    })
                } else {
                    Ok(AgentResponse::text("done", Default::default()))
                }
            }
        }
        let e = engine_with(
            Arc::new(SleepLlm(AtomicUsize::new(0))),
            Arc::new(harxes_infra_shell::TokioCommandShell::new(600)),
        );
        let handle = e.run(RunRequest {
            prompt: "run it".into(),
            ..Default::default()
        });
        let (_events, driver) = handle.split();
        // Let the Bash child actually spawn, then drop the driver (cancel).
        let driven = tokio::spawn(driver);
        tokio::time::sleep(Duration::from_millis(600)).await;
        driven.abort();
        let _ = driven.await;
        // Give the OS a moment to reap the killed group.
        tokio::time::sleep(Duration::from_millis(400)).await;
        // Assert no lingering process carries our unique marker.
        let out = std::process::Command::new("pgrep")
            .arg("-f")
            .arg("harxes_cancel_probe_zzz")
            .output()
            .expect("pgrep");
        let hits = String::from_utf8_lossy(&out.stdout);
        let live: Vec<&str> = hits.split_whitespace().collect();
        assert!(
            live.is_empty(),
            "cancelled run left process(es) alive: {live:?}"
        );
    }
}
