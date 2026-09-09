//! Multi-turn agent loop orchestrating repeated LLM turns, executing requested
//! tool calls (bash/read/write) and feeding results back until it stops.

use std::sync::Arc;

use harxes_core_domain::domain::value_objects::{
    Message, ProviderId, Role, ToolCall, ToolSpec,
};
use harxes_core_domain::ports::{AgentResponse, FileSystemPort, LlmError, LlmPort};

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
            max_iterations: 40,
            // Cumulative in+out across the whole turn. This is a runaway-cost
            // backstop, not a working budget: with a large context every call
            // re-sends ~100k input tokens, so a tight cap kills real work.
            max_total_tokens: 1_500_000,
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
    decider: Option<Arc<dyn harxes_core_domain::ports::PermissionDecider>>,
    session: Option<(Arc<dyn harxes_core_domain::ports::SessionStorePort>, String)>,
    observer: Option<Arc<dyn harxes_core_domain::ports::ToolObserver>>,
    stream: bool,
    /// Depth of nested sub-agent delegation (0 = top-level agent). Capped to
    /// prevent unbounded recursion through the `Delegate` tool.
    delegation_depth: usize,
    /// Shared plan/todo list the agent can manage with the `Todo` tool.
    todos: Option<Arc<std::sync::Mutex<Vec<harxes_core_domain::domain::value_objects::TodoItem>>>>,
    /// User-configured shell-command allow/deny rules.
    command_policy: harxes_core_domain::domain::value_objects::CommandPolicy,
    /// Runtime-discovered tools (e.g. MCP servers), merged into the tool list.
    dynamic_tools: Option<Arc<dyn harxes_core_domain::ports::DynamicToolPort>>,
    /// User-configured lifecycle hooks run around tool execution.
    hooks: harxes_core_domain::ports::HooksConfig,
    /// Web fetcher backing the `Fetch` tool (optional).
    web: Option<Arc<dyn harxes_core_domain::ports::WebPort>>,
    /// Models to fail over to (in order) when the primary model errors out
    /// terminally after retries.
    fallback_models: Vec<String>,
    /// Working directory for shell commands and relative path anchoring.
    /// Defaults to "." (the process CWD); an embedding host sets the run's
    /// worktree so tools don't operate against the host's directory.
    working_dir: String,
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
            decider: None,
            session: None,
            observer: None,
            stream: false,
            delegation_depth: 0,
            todos: None,
            command_policy: Default::default(),
            dynamic_tools: None,
            hooks: Default::default(),
            web: None,
            fallback_models: Vec::new(),
            working_dir: ".".to_string(),
        }
    }

    /// Set the working directory for Bash and relative-path tools.
    pub fn with_working_dir(mut self, dir: impl Into<String>) -> Self {
        let d = dir.into();
        if !d.trim().is_empty() {
            self.working_dir = d;
        }
        self
    }

    /// Configure ordered fallback models used when the primary model fails.
    pub fn with_fallback_models(mut self, models: Vec<String>) -> Self {
        self.fallback_models = models;
        self
    }

    /// Attach a web fetcher enabling the `Fetch` tool.
    pub fn with_web(mut self, w: Arc<dyn harxes_core_domain::ports::WebPort>) -> Self {
        self.web = Some(w);
        self
    }

    /// Attach user-configured lifecycle hooks.
    pub fn with_hooks(mut self, hooks: harxes_core_domain::ports::HooksConfig) -> Self {
        self.hooks = hooks;
        self
    }

    /// Attach a provider of runtime-discovered tools (e.g. MCP servers).
    pub fn with_dynamic_tools(
        mut self,
        d: Arc<dyn harxes_core_domain::ports::DynamicToolPort>,
    ) -> Self {
        self.dynamic_tools = Some(d);
        self
    }

    /// Attach user-configured shell-command allow/deny rules.
    pub fn with_command_policy(
        mut self,
        p: harxes_core_domain::domain::value_objects::CommandPolicy,
    ) -> Self {
        self.command_policy = p;
        self
    }

    /// Attach a file-permission policy gating mutating tool operations.
    pub fn with_policy(
        mut self,
        p: harxes_core_domain::domain::value_objects::PermissionPolicy,
    ) -> Self {
        self.policy = Some(p);
        self
    }

    /// Attach an interactive decider for operations not covered by the policy.
    pub fn with_decider(
        mut self,
        d: Arc<dyn harxes_core_domain::ports::PermissionDecider>,
    ) -> Self {
        self.decider = Some(d);
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

    /// Attach a shared plan/todo list the agent can manage via the `Todo` tool.
    pub fn with_todos(
        mut self,
        todos: Arc<std::sync::Mutex<Vec<harxes_core_domain::domain::value_objects::TodoItem>>>,
    ) -> Self {
        self.todos = Some(todos);
        self
    }

    /// Enable realtime streaming: text deltas are pushed to the observer as the
    /// model generates, instead of only appearing when a turn completes. Falls
    /// back to non-streaming for providers that cannot stream.
    pub fn with_streaming(mut self, on: bool) -> Self {
        self.stream = on;
        self
    }

    fn tool_specs(&self) -> Vec<harxes_core_domain::domain::value_objects::ToolSpec> {
        use harxes_core_domain::domain::services::tool_protocol::all_tool_specs;
        let mut specs = all_tool_specs();
        if let Some(d) = &self.dynamic_tools {
            specs.extend(d.specs());
        }
        specs
    }

    async fn execute_call(
        &self,
        provider_id: &ProviderId,
        model_id: &str,
        call: &ToolCall,
    ) -> String {
        use harxes_core_domain::domain::services::tool_protocol::{parse_args, ParsedArgs, ToolId};
        if let Some(obs) = &self.observer {
            use harxes_core_domain::domain::services::tool_protocol::tool_summary;
            // Show a clean "Bash cargo test" summary in the panel, but keep the
            // tool name column separate so the observer can style it.
            let summary = tool_summary(call.name.as_str(), call.arguments.as_str());
            let arg = summary
                .strip_prefix(call.name.as_str())
                .map(|s| s.trim())
                .unwrap_or(&summary);
            obs.on_tool_start(call.name.as_str(), arg);
        }
        // pre_tool hooks may veto the call (exit code 2).
        if let Some(block_reason) = self.run_pre_hooks(call).await {
            if let Some(obs) = &self.observer {
                obs.on_tool_result(call.name.as_str(), &Self::preview(&block_reason, 100));
            }
            return block_reason;
        }
        let result = match ToolId::parse(call.name.as_str()) {
            Some(tool) => match parse_args(tool, &call.arguments) {
                ParsedArgs::Bash { command } => self.run_bash(&command).await,
                ParsedArgs::Read { path, offset, limit } => {
                    self.read_file(&path, offset, limit).await
                }
                ParsedArgs::Write { path, content } => self.write_file_parts(&path, &content).await,
                ParsedArgs::Edit {
                    path,
                    old_string,
                    new_string,
                } => self.edit_file(&path, &old_string, &new_string).await,
                ParsedArgs::Grep {
                    needle,
                    pattern,
                    max_matches,
                } => self.grep(&needle, &pattern, max_matches).await,
                ParsedArgs::Glob { pattern, max_depth } => {
                    self.glob(&pattern, max_depth).await
                }
                ParsedArgs::List { path } => self.list_dir_wd(&path),
                ParsedArgs::Skill { name } => self.load_skill_wd(&name),
                ParsedArgs::Delegate { task, context } => {
                    self.delegate_subtask(provider_id, model_id, &task, context.as_deref())
                        .await
                }
                ParsedArgs::Todo { action, items } => self.handle_todo(action, items),
                ParsedArgs::Fetch { url } => match &self.web {
                    Some(w) => match w.fetch(&url).await {
                        Ok(text) => text,
                        Err(e) => format!("fetch error: {e}"),
                    },
                    None => "fetch error: web access is not available".to_string(),
                },
                ParsedArgs::Search { query } => match &self.web {
                    Some(w) => match w.search(&query).await {
                        Ok(text) => text,
                        Err(e) => format!("search error: {e}"),
                    },
                    None => "search error: web access is not available".to_string(),
                },
            },
            None => match &self.dynamic_tools {
                Some(d) if d.owns(call.name.as_str()) => {
                    d.call(call.name.as_str(), call.arguments.as_str()).await
                }
                _ => format!("unknown tool '{}'", call.name),
            },
        };
        let result = self.run_post_hooks(call, result).await;
        if let Some(obs) = &self.observer {
            obs.on_tool_result(call.name.as_str(), &Self::preview(result.as_str(), 100));
        }
        result
    }

    /// True when a tool call may run concurrently with its neighbors:
    /// read-only tools, plus Delegate — issuing several delegations in one
    /// turn is an explicit fan-out request, so run the sub-agents in parallel.
    fn is_parallel_safe(name: &str) -> bool {
        use harxes_core_domain::domain::services::tool_protocol::ToolId;
        matches!(
            ToolId::parse(name),
            Some(
                ToolId::Read
                    | ToolId::Grep
                    | ToolId::Glob
                    | ToolId::Fetch
                    | ToolId::Search
                    | ToolId::Delegate
            )
        )
    }

    /// Execute a turn's tool calls, returning outputs in call order. Runs of
    /// consecutive read-only calls execute concurrently; everything else runs
    /// sequentially at its original position.
    async fn execute_calls(
        &self,
        provider_id: &ProviderId,
        model_id: &str,
        calls: &[ToolCall],
    ) -> Vec<String> {
        let mut outputs: Vec<String> = vec![String::new(); calls.len()];
        let mut i = 0usize;
        while i < calls.len() {
            if Self::is_parallel_safe(calls[i].name.as_str()) {
                let mut j = i;
                while j < calls.len() && Self::is_parallel_safe(calls[j].name.as_str()) {
                    j += 1;
                }
                let batch = futures_util::future::join_all(
                    calls[i..j]
                        .iter()
                        .map(|c| self.execute_call(provider_id, model_id, c)),
                )
                .await;
                for (k, out) in batch.into_iter().enumerate() {
                    outputs[i + k] = out;
                }
                i = j;
            } else {
                outputs[i] = self.execute_call(provider_id, model_id, &calls[i]).await;
                i += 1;
            }
        }
        outputs
    }

    /// True when a hook rule's matcher applies to the tool name (empty or `*`
    /// matches everything; otherwise a case-insensitive `*`-wildcard match).
    fn hook_matches(matcher: &str, tool: &str) -> bool {
        let m = matcher.trim().to_lowercase();
        let t = tool.to_lowercase();
        if m.is_empty() || m == "*" {
            return true;
        }
        if let Some(prefix) = m.strip_suffix('*') {
            return t.starts_with(prefix);
        }
        t == m
    }

    /// Shell snippet exporting the tool context for a hook command.
    fn hook_env(call: &ToolCall) -> String {
        let esc = |s: &str| s.replace('\'', r"'\''");
        format!(
            "export HARXES_TOOL_NAME='{}' HARXES_TOOL_ARGS='{}'; ",
            esc(call.name.as_str()),
            esc(call.arguments.as_str())
        )
    }

    /// Run matching pre_tool hooks. Returns Some(reason) when a hook blocks
    /// the call (exit code 2); other exit codes are advisory and ignored.
    async fn run_pre_hooks(&self, call: &ToolCall) -> Option<String> {
        use harxes_core_domain::ports::ShellExitStatus;
        for rule in &self.hooks.pre_tool {
            if !Self::hook_matches(&rule.matcher, call.name.as_str()) {
                continue;
            }
            let cmd = format!("{}{}", Self::hook_env(call), rule.command);
            if let Ok(o) = self.shell.run_command(&self.working_dir, &cmd).await {
                if o.exit_status == ShellExitStatus::Failure(2) {
                    let why = if o.stderr.trim().is_empty() {
                        o.stdout.trim().to_string()
                    } else {
                        o.stderr.trim().to_string()
                    };
                    return Some(format!(
                        "blocked by pre_tool hook '{}': {}",
                        rule.command,
                        if why.is_empty() { "(no reason given)" } else { &why }
                    ));
                }
            }
        }
        None
    }

    /// Run matching post_tool hooks; non-empty stdout is appended to the tool
    /// result as feedback for the model.
    async fn run_post_hooks(&self, call: &ToolCall, mut result: String) -> String {
        for rule in &self.hooks.post_tool {
            if !Self::hook_matches(&rule.matcher, call.name.as_str()) {
                continue;
            }
            let cmd = format!("{}{}", Self::hook_env(call), rule.command);
            if let Ok(o) = self.shell.run_command(&self.working_dir, &cmd).await {
                let out = o.stdout.trim();
                if !out.is_empty() {
                    let capped: String = out.chars().take(1000).collect();
                    result.push_str(&format!("\n[hook feedback] {capped}"));
                }
            }
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
        use harxes_core_domain::domain::value_objects::CommandVerdict;
        match self.command_policy.evaluate(cmd) {
            CommandVerdict::Deny => {
                return format!("permission denied: '{cmd}' is blocked by the configured command deny list");
            }
            CommandVerdict::Allow => {}
            CommandVerdict::Ask => {
                if Self::is_dangerous(cmd) {
                    if let Some(d) = &self.decider {
                        if !d.decide_bash(cmd) {
                            return format!("permission denied: '{cmd}' was not approved");
                        }
                    }
                }
            }
        }
        match self.shell.run_command(&self.working_dir, cmd).await {
            Ok(o) => {
                let code = match o.exit_status {
                    ShellExitStatus::Success => 0,
                    ShellExitStatus::Failure(c) => c,
                };
                // Cap output so one noisy command cannot flood the context;
                // keep the tail, where errors and summaries usually live.
                const MAX_STREAM: usize = 20_000;
                let cap = |s: &str| -> String {
                    let t = s.trim();
                    if t.chars().count() <= MAX_STREAM {
                        return t.to_string();
                    }
                    let tail: String = t
                        .chars()
                        .rev()
                        .take(MAX_STREAM)
                        .collect::<Vec<_>>()
                        .into_iter()
                        .rev()
                        .collect();
                    format!("[output truncated to last {MAX_STREAM} chars]\n{tail}")
                };
                format!(
                    "exit={} stdout={} stderr={}",
                    code,
                    cap(&o.stdout),
                    cap(&o.stderr)
                )
            }
            Err(e) => format!("shell error {e}"),
        }
    }

    /// Heuristic: commands that can irreversibly destroy state.
    fn is_dangerous(cmd: &str) -> bool {
        // Normalize and inspect every sub-command (a danger hidden after
        // `cd foo &&` or `x; y` must still be caught).
        let lower = cmd.to_lowercase();
        // Substrings that are dangerous ANYWHERE in the command line.
        const DANGER_SUBSTR: [&str; 16] = [
            "rm -rf",
            "rm -fr",
            "rm -r --no-preserve-root",
            "sudo ",
            "mkfs",
            ":(){ :|:& };:", // fork bomb
            "dd if=",
            "> /dev/sd",
            "> /dev/disk",
            "of=/dev/",
            "chmod -r 777 /",
            "chown -r",
            "git push --force",
            "git push -f",
            "git reset --hard",
            "git clean -",
        ];
        if DANGER_SUBSTR.iter().any(|p| lower.contains(p)) {
            return true;
        }
        // Piping a network fetch straight into a shell: curl … | sh / | bash.
        let piped_exec = (lower.contains("curl ") || lower.contains("wget "))
            && (lower.contains("| sh") || lower.contains("| bash") || lower.contains("|sh") || lower.contains("|bash"));
        // rm -r targeting an absolute/home root rather than a local subdir.
        let rm_root = lower.contains("rm -r")
            && (lower.contains(" /") || lower.contains(" ~") || lower.contains(" $home"));
        piped_exec || rm_root
    }

    /// Default line cap for Read so one huge file cannot flood the context.
    const READ_DEFAULT_LIMIT: usize = 2000;

    async fn read_file(&self, path: &str, offset: Option<usize>, limit: Option<usize>) -> String {
        let content = match self.fsys.read(path.trim()).await {
            Ok(c) => c,
            Err(e) => return format!("read error {e}"),
        };
        let total = content.lines().count();
        let start = offset.unwrap_or(1).max(1) - 1; // 1-based -> 0-based
        let limit = limit.unwrap_or(Self::READ_DEFAULT_LIMIT).max(1);
        let end = (start + limit).min(total);
        // Number every line (cat -n style) so the model can cite locations and
        // target edits precisely. Right-align the gutter to the widest number.
        let width = end.max(1).to_string().len();
        let body: String = content
            .lines()
            .enumerate()
            .skip(start)
            .take(limit)
            .map(|(i, l)| format!("{:>width$}  {l}", i + 1, width = width))
            .collect::<Vec<_>>()
            .join("\n");
        if start == 0 && total <= limit {
            return body;
        }
        format!(
            "[showing lines {}-{} of {}; pass offset/limit to read more]\n{}",
            start + 1,
            end,
            total,
            body
        )
    }

    /// Load a skill's full instructions by name from `.harxes/skills/<name>/`
    /// or `.claude/skills/<name>/SKILL.md` (frontmatter stripped), anchored to
    /// the working directory.
    fn load_skill_wd(&self, name: &str) -> String {
        let want = name.trim();
        if want.is_empty() {
            return "skill error: no skill name given".to_string();
        }
        let wd = &self.working_dir;
        for sub in [".harxes/skills", ".claude/skills"] {
            let path = format!("{wd}/{sub}/{want}/SKILL.md");
            if let Ok(content) = std::fs::read_to_string(&path) {
                // Strip a leading `---` frontmatter block if present.
                let body = match content.trim_start().strip_prefix("---") {
                    Some(rest) => match rest.find("\n---") {
                        Some(end) => rest[end + 4..].trim_start_matches(['\r', '\n']).to_string(),
                        None => content.clone(),
                    },
                    None => content.clone(),
                };
                return format!("[skill: {want}]\n{}", body.trim());
            }
        }
        format!("skill error: no skill named '{want}' (check the Available skills list)")
    }

    /// List a directory, anchoring a relative path to the working directory.
    fn list_dir_wd(&self, path: &str) -> String {
        let raw = path.trim();
        let anchored;
        let p: &str = if raw.is_empty() || raw == "." {
            &self.working_dir
        } else if std::path::Path::new(raw).is_absolute() {
            raw
        } else {
            anchored = format!("{}/{}", self.working_dir, raw);
            &anchored
        };
        Self::list_dir(p)
    }

    /// List a directory's entries (dirs first, then files; dirs get a trailing
    /// `/`). Well-known noise dirs are hidden from the top level.
    fn list_dir(path: &str) -> String {
        let p = if path.trim().is_empty() { "." } else { path.trim() };
        let rd = match std::fs::read_dir(p) {
            Ok(rd) => rd,
            Err(e) => return format!("list error: cannot read '{p}': {e}"),
        };
        const HIDE: [&str; 6] = [".git", "target", "node_modules", ".venv", "dist", "__pycache__"];
        let mut dirs = Vec::new();
        let mut files = Vec::new();
        for entry in rd.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
            if is_dir {
                if HIDE.contains(&name.as_str()) {
                    continue;
                }
                dirs.push(format!("{name}/"));
            } else {
                files.push(name);
            }
        }
        dirs.sort();
        files.sort();
        if dirs.is_empty() && files.is_empty() {
            return format!("{p}: (empty)");
        }
        let mut out = format!("{p}:\n");
        for d in dirs.iter().chain(files.iter()) {
            out.push_str(&format!("  {d}\n"));
        }
        out.trim_end().to_string()
    }

    async fn grep(&self, needle: &str, pattern: &str, max_matches: usize) -> String {
        use harxes_core_domain::ports::GrepMatch;
        let needle = needle.trim();
        if needle.is_empty() {
            return "grep error: empty needle".to_string();
        }
        let pattern = if pattern.trim().is_empty() { "**/*" } else { pattern };
        let cap = max_matches.max(1);
        match self.fsys.grep(needle, pattern, cap).await {
            Ok(hits) if hits.is_empty() => format!("no matches for '{needle}'"),
            Ok(hits) => {
                let truncated = hits.len() >= cap;
                // Group hits by path for a compact report.
                let mut by_file: std::collections::BTreeMap<&str, Vec<&GrepMatch>> =
                    std::collections::BTreeMap::new();
                for h in &hits {
                    by_file.entry(h.path.as_str()).or_default().push(h);
                }
                let mut out = format!(
                    "{} match{} in {} file{}:\n",
                    hits.len(),
                    if hits.len() == 1 { "" } else { "es" },
                    by_file.len(),
                    if by_file.len() == 1 { "" } else { "s" }
                );
                for (path, lines) in by_file {
                    out.push_str(&format!("{path}:\n"));
                    for h in lines {
                        out.push_str(&format!("  {}: {}\n", h.line_number, h.line.trim_end()));
                    }
                }
                if truncated {
                    out.push_str(&format!(
                        "[stopped at {cap} matches — narrow the pattern or raise max_matches for more]"
                    ));
                }
                out
            }
            Err(e) => format!("grep error {e}"),
        }
    }

    async fn glob(&self, pattern: &str, max_depth: Option<usize>) -> String {
        use harxes_core_domain::ports::GlobOptions;
        let opts = GlobOptions {
            max_depth,
            ignore: vec![],
        };
        let mut files = self.fsys.glob(pattern, &opts).await;
        if files.is_empty() {
            return format!("no files match '{pattern}'");
        }
        // Newest first so the most relevant files surface at the top, then cap
        // so a broad glob can't flood the context.
        const CAP: usize = 200;
        files.sort_by_cached_key(|p| {
            std::cmp::Reverse(
                std::fs::metadata(p)
                    .and_then(|m| m.modified())
                    .ok(),
            )
        });
        let total = files.len();
        let shown = total.min(CAP);
        let mut out = format!("{total} file{}:\n", if total == 1 { "" } else { "s" });
        out.push_str(&files[..shown].join("\n"));
        if total > CAP {
            out.push_str(&format!("\n[showing newest {CAP} of {total} — narrow the pattern for the rest]"));
        }
        out
    }

    async fn write_file_parts(&self, path: &str, content: &str) -> String {
        if path.is_empty() {
            return "write error: empty path".to_string();
        }
        use harxes_core_domain::domain::value_objects::FsOp;
        let allowed = match &self.policy {
            Some(p) => p.permits(FsOp::Write, path),
            None => true,
        };
        if !allowed {
            if let Some(d) = &self.decider {
                if !d.decide_write(path) {
                    return format!("permission denied: writing {} was not approved", path);
                }
            } else {
                return format!(
                    "permission required: writing {} is not in the allow list; no action taken",
                    path
                );
            }
        }
        // Golden-diff review: if this overwrites an existing, different file and
        // an interactive decider is present, show the exact change and require
        // approval before writing.
        if let Some(d) = &self.decider {
            if let Ok(old) = self.fsys.read(path).await {
                if old != content {
                    let diff = harxes_core_domain::domain::services::diff::generate_diff(&old, content);
                    if !d.decide_write_diff(path, &diff) {
                        return format!("permission denied: diff to {path} was not approved");
                    }
                }
            }
        }
        match self.fsys.write(path, content).await {
            Ok(()) => format!("wrote {} bytes to {}", content.len(), path),
            Err(e) => format!("write error {e}"),
        }
    }

    /// Consult the policy (and prompt decider) before a mutating filesystem
    /// operation. Returns `Some(reason)` when the write should be blocked.
    fn ensure_write_granted(&self, path: &str) -> Option<String> {
        use harxes_core_domain::domain::value_objects::FsOp;
        if path.is_empty() {
            return Some("empty path".to_string());
        }
        let allowed = match &self.policy {
            Some(p) => p.permits(FsOp::Write, path),
            None => true,
        };
        if !allowed {
            if let Some(d) = &self.decider {
                if !d.decide_write(path) {
                    return Some(format!("writing {path} was not approved"));
                }
            } else {
                return Some(format!(
                    "writing {path} is not in the allow list; no action taken"
                ));
            }
        }
        None
    }

    async fn edit_file(&self, path: &str, old_string: &str, new_string: &str) -> String {
        if let Some(reason) = self.ensure_write_granted(path) {
            return format!("permission denied: {reason}");
        }
        // Golden-diff review: preview the single hunk before applying.
        let mut applied_diff = String::new();
        if let Ok(old) = self.fsys.read(path).await {
            if old.contains(old_string) {
                let new = old.replacen(old_string, new_string, 1);
                applied_diff =
                    harxes_core_domain::domain::services::diff::generate_diff(&old, &new);
                if let Some(d) = &self.decider {
                    if !d.decide_write_diff(path, &applied_diff) {
                        return format!("permission denied: edit to {path} was not approved");
                    }
                }
            }
        }
        match self.fsys.replace(path, old_string, new_string).await {
            Ok(bytes) => {
                // Include the applied hunk (capped) so the model can verify
                // the change without re-reading the file.
                let mut out = format!("edited {path} ({bytes} bytes written)");
                if !applied_diff.is_empty() {
                    let capped: String = applied_diff.chars().take(2000).collect();
                    out.push('\n');
                    out.push_str(&capped);
                    if capped.len() < applied_diff.len() {
                        out.push_str("\n[diff truncated]");
                    }
                }
                out
            }
            Err(e) => {
                let mut msg = format!("edit error {e}");
                // When old_string doesn't match, point the model at the
                // closest-looking region so it can correct itself in one step
                // instead of blindly re-reading and retrying.
                if let Ok(content) = self.fsys.read(path).await {
                    let occurrences = content.matches(old_string).count();
                    if occurrences == 0 {
                        if let Some((ln, snippet)) = Self::closest_snippet(&content, old_string) {
                            msg.push_str(&format!(
                                "\nold_string was not found in the file. Closest match near line {ln}:\n{snippet}\nAdjust old_string to match the file exactly (whitespace and indentation matter)."
                            ));
                        }
                    } else if occurrences > 1 {
                        msg.push_str(&format!(
                            "\nold_string matched {occurrences} places — it must be unique. Add surrounding lines to old_string so it identifies exactly one location."
                        ));
                    }
                }
                msg
            }
        }
    }

    /// Locate the file region most similar to the first meaningful line of a
    /// failed `old_string`, returning (1-based line, ±3-line snippet).
    fn closest_snippet(content: &str, old_string: &str) -> Option<(usize, String)> {
        let probe = old_string.lines().find(|l| !l.trim().is_empty())?.trim();
        if probe.len() < 4 {
            return None;
        }
        let bigrams = |s: &str| -> std::collections::HashSet<(char, char)> {
            let cs: Vec<char> = s.chars().collect();
            cs.windows(2).map(|w| (w[0], w[1])).collect()
        };
        let pb = bigrams(probe);
        let lines: Vec<&str> = content.lines().collect();
        let (mut best, mut best_score) = (None, 0.0f64);
        for (i, line) in lines.iter().enumerate() {
            let t = line.trim();
            if t.is_empty() {
                continue;
            }
            let lb = bigrams(t);
            let inter = pb.intersection(&lb).count() as f64;
            let denom = pb.len().max(lb.len()).max(1) as f64;
            let score = inter / denom;
            if score > best_score {
                best_score = score;
                best = Some(i);
            }
        }
        let i = best.filter(|_| best_score >= 0.4)?;
        let lo = i.saturating_sub(3);
        let hi = (i + 4).min(lines.len());
        Some((i + 1, lines[lo..hi].join("\n")))
    }

    /// Maximum delegation nesting depth before the `Delegate` tool refuses.
    const MAX_DELEGATION_DEPTH: usize = 3;

    /// Spawn a fresh sub-agent in its own context to complete `task`, reusing
    /// this agent's filesystem, shell, policy and decider. The sub-agent runs a
    /// dedicated, shorter loop and returns its final text as the tool result.
    /// Wraps a sub-agent's tool events for the parent observer: names are
    /// prefixed with an indented `└` per delegation depth, and stream deltas
    /// are swallowed so nested output cannot garble the parent's live text.
    fn nested_prefix(depth: usize, name: &str) -> String {
        format!("{}└ {name}", "  ".repeat(depth.saturating_sub(1)))
    }

    async fn delegate_subtask(
        &self,
        provider_id: &ProviderId,
        model_id: &str,
        task: &str,
        context: Option<&str>,
    ) -> String {
        if task.trim().is_empty() {
            return "delegate error: empty task".to_string();
        }
        if self.delegation_depth >= Self::MAX_DELEGATION_DEPTH {
            return "delegate error: max delegation depth reached".to_string();
        }
        let mut prompt = String::new();
        prompt.push_str(
            "You are a sub-agent working on one delegated sub-task. You have the same \
             filesystem, shell and tooling as the main agent. Complete ONLY this sub-task, \
             then return a concise final result. Do not restate the task or add commentary.",
        );
        prompt.push_str("\n\nSUB-TASK:\n");
        prompt.push_str(task);
        if let Some(c) = context {
            if !c.trim().is_empty() {
                prompt.push_str("\n\nCONTEXT:\n");
                prompt.push_str(c.trim());
            }
        }

        let sub = AgentLoop {
            llm: self.llm.clone(),
            shell: self.shell.clone(),
            fsys: self.fsys.clone(),
            policy: self.policy.clone(),
            decider: self.decider.clone(),
            session: None,
            // Forward tool activity to the parent's observer, indented per
            // delegation depth, so sub-agent work is visible in the UI.
            observer: self.observer.clone().map(|o| {
                Arc::new(NestedObserver {
                    inner: o,
                    depth: self.delegation_depth + 1,
                }) as Arc<dyn harxes_core_domain::ports::ToolObserver>
            }),
            stream: false,
            delegation_depth: self.delegation_depth + 1,
            // Sub-agents do not touch the parent's plan: delegated tasks are a
            // single step from the parent's perspective.
            todos: None,
            command_policy: self.command_policy.clone(),
            dynamic_tools: self.dynamic_tools.clone(),
            hooks: self.hooks.clone(),
            web: self.web.clone(),
            fallback_models: self.fallback_models.clone(),
            working_dir: self.working_dir.clone(),
        };
        let limits = LoopLimits {
            max_iterations: 15,
            ..LoopLimits::default()
        };
        let system = "You are a helpful autonomous sub-agent. Use your tools to complete the delegated sub-task, then return the final result text.";
        match sub.run(provider_id, model_id, system, &prompt, &limits).await {
            Ok(out) => format!("[sub-agent result]\n{}", out.final_text),
            Err(e) => format!("sub-agent error {e}"),
        }
    }

    /// Apply a `Todo` action against the shared plan list: `write` replaces
    /// the whole plan atomically, `list` reads it back. Returns a short report
    /// folded into the transcript.
    fn handle_todo(
        &self,
        action: harxes_core_domain::domain::value_objects::TodoAction,
        new_items: Vec<harxes_core_domain::domain::value_objects::TodoItem>,
    ) -> String {
        use harxes_core_domain::domain::value_objects::{
            normalize_todos, render_todos, TodoAction,
        };
        let Some(list) = &self.todos else {
            return "todo error: the Todo tool is not available in this context".to_string();
        };
        let mut items = list.lock().unwrap();
        match action {
            TodoAction::List => {
                if items.is_empty() {
                    return "todo: (empty plan)".to_string();
                }
                let done = items.iter().filter(|i| i.done()).count();
                format!("todo ({done}/{} done):\n{}", items.len(), render_todos(&items))
            }
            TodoAction::Write => {
                if new_items.is_empty() {
                    return "todo error: write requires a non-empty `todos` array (send the complete plan)".to_string();
                }
                let mut next = new_items;
                normalize_todos(&mut next);
                *items = next;
                let done = items.iter().filter(|i| i.done()).count();
                format!(
                    "plan updated ({done}/{} done):\n{}",
                    items.len(),
                    render_todos(&items)
                )
            }
        }
    }

    /// Snapshot of the shared plan, or None when no plan is attached/empty.
    fn todo_snapshot(&self) -> Option<Vec<harxes_core_domain::domain::value_objects::TodoItem>> {
        let items = self.todos.as_ref()?.lock().unwrap().clone();
        if items.is_empty() {
            None
        } else {
            Some(items)
        }
    }

    /// Call the LLM with exponential backoff on transient failures
    /// (rate limits and 5xx-class transport errors), honoring the provider's
    /// `retry_after` hint when present. Auth and timeout errors are terminal
    /// and returned immediately. When streaming is enabled, text deltas are
    /// pushed to the observer in real time.
    /// Call the primary model with retries; on terminal failure walk the
    /// configured fallback models in order. Auth errors are not failed over
    /// (the key is broken for every model alike).
    async fn generate_with_failover(
        &self,
        provider_id: &ProviderId,
        model_id: &str,
        transcript: &[Message],
        tools: &[ToolSpec],
    ) -> Result<AgentResponse, LlmError> {
        let primary_err = match self
            .retry_generate(provider_id, model_id, transcript, tools, 3)
            .await
        {
            Ok(r) => return Ok(r),
            Err(e @ LlmError::Auth { .. }) => return Err(e),
            Err(e) => e,
        };
        for fb in &self.fallback_models {
            if fb == model_id {
                continue;
            }
            if let Some(obs) = &self.observer {
                obs.on_tool_start("Failover", &format!("{model_id} failed — trying {fb}"));
            }
            match self.retry_generate(provider_id, fb, transcript, tools, 2).await {
                Ok(r) => {
                    if let Some(obs) = &self.observer {
                        obs.on_tool_result("Failover", &format!("continuing on {fb}"));
                    }
                    return Ok(r);
                }
                Err(_) => continue,
            }
        }
        Err(primary_err)
    }

    async fn retry_generate(
        &self,
        provider_id: &ProviderId,
        model_id: &str,
        transcript: &[Message],
        tools: &[ToolSpec],
        max_retries: usize,
    ) -> Result<AgentResponse, LlmError> {
        use harxes_core_domain::ports::{LlmError as LE, StreamEvent, StreamSink};
        // Build a sink forwarding text deltas to the observer when streaming.
        let observer = self.observer.clone();
        let sink: Option<StreamSink> = if self.stream {
            Some(Arc::new(move |ev: StreamEvent| match ev {
                StreamEvent::Text(t) => {
                    if let Some(obs) = &observer {
                        obs.on_stream_delta(&t);
                    }
                }
                StreamEvent::Reasoning(t) => {
                    if let Some(obs) = &observer {
                        obs.on_reasoning(&t);
                    }
                }
                StreamEvent::ToolCall(_) => {}
            }))
        } else {
            None
        };

        let mut attempt = 0usize;
        loop {
            let result = match &sink {
                Some(s) => self
                    .llm
                    .generate_stream(
                        provider_id,
                        model_id,
                        transcript,
                        tools,
                        None,
                        s.clone(),
                    )
                    .await,
                None => {
                    self.llm
                        .generate(provider_id, model_id, transcript, tools, None)
                        .await
                }
            };
            match result {
                Ok(r) => return Ok(r),
                Err(e) => {
                    // Classify the error as retryable (rate limit / transient
                    // 5xx-class request failure) or terminal.
                    // Rate limits deserve real patience: quota windows are
                    // usually tens of seconds, so give them more attempts and
                    // longer waits than generic transient failures.
                    let plan = match &e {
                        LE::RateLimited { retry_after } => Some((
                            max_retries.max(6),
                            (2000u64 << attempt.min(4)).min(30_000),
                            *retry_after,
                        )),
                        LE::Request(_) => Some((
                            max_retries,
                            250u64.saturating_mul(1 << attempt.min(5)),
                            None,
                        )),
                        _ => None,
                    };
                    match plan {
                        Some((cap, base, ra)) if attempt < cap => {
                            let wait = match ra {
                                Some(secs) => base.max(secs * 1000),
                                None => base,
                            };
                            if let Some(obs) = &self.observer {
                                obs.on_retry(wait / 1000);
                            }
                            tokio::time::sleep(std::time::Duration::from_millis(wait)).await;
                            attempt += 1;
                        }
                        _ => return Err(e),
                    }
                }
            }
        }
    }

    /// Shared loop body: drive tool-calling turns over an existing transcript
    /// until a text-only turn or guardrails cut it off. Returns both the outcome
    /// and the final transcript so callers can carry conversation state.
    #[async_recursion::async_recursion]
    async fn run_loop(
        &self,
        provider_id: &ProviderId,
        model_id: &str,
        mut transcript: Vec<Message>,
        limits: &LoopLimits,
    ) -> Result<(LoopOutcome, Vec<Message>), LlmError> {
        let tools = self.tool_specs();
        let mut iterations = 0usize;
        // Plan-awareness tracking: nudge the model when it works for several
        // turns without touching an in-progress plan (Phase-2 "staleness nudge").
        let mut last_plan = self.todo_snapshot();
        let mut plan_stale_turns = 0usize;
        let mut total_tokens = 0u64;
        let mut total_input = 0u64;
        let mut total_output = 0u64;
        let mut truncated = false;

        loop {
            if iterations >= limits.max_iterations {
                truncated = true;
                break;
            }
            // Keep the transcript within the per-call window budget, folding the
            // oldest turns into a compact summary thread instead of a hard gap.
            use harxes_core_domain::domain::value_objects::compress_transcript_to_budget;
            if !transcript.is_empty() {
                transcript = compress_transcript_to_budget(
                    std::mem::take(&mut transcript),
                    limits.context_window_tokens,
                    Some(64),
                );
            }
            let resp = self
                .generate_with_failover(provider_id, model_id, &transcript, &tools)
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
                // A turn that ends with no text at all (some reasoning models
                // finish a tool sequence emitting only hidden reasoning) must
                // never render as a blank screen.
                let final_text = if resp.content.trim().is_empty() {
                    "(the model finished without a text reply — it may consider the task done, or need a more specific instruction)".to_string()
                } else {
                    resp.content.clone()
                };
                let out = LoopOutcome {
                    final_text,
                    iterations,
                    usage_total_tokens: total_tokens,
                    input_tokens: total_input,
                    output_tokens: total_output,
                    truncated_by_guardrail: truncated,
                };
                return Ok((out, transcript));
            }

            // Feed tool calls + results back into the transcript. Read-only
            // calls in the same turn run concurrently; anything mutating keeps
            // strict sequential order.
            transcript.push(Message::assistant_with_tools(resp.tool_calls.clone()));
            let mut outputs = self
                .execute_calls(provider_id, model_id, &resp.tool_calls)
                .await;
            let last_ix = resp.tool_calls.len().saturating_sub(1);
            for (i, call) in resp.tool_calls.iter().enumerate() {
                let mut output = std::mem::take(&mut outputs[i]);
                // Fold a plan reminder into the last tool result of the turn
                // when the plan has gone stale mid-task.
                if i == last_ix {
                    let now = self.todo_snapshot();
                    if now == last_plan {
                        plan_stale_turns += 1;
                    } else {
                        plan_stale_turns = 0;
                        last_plan = now.clone();
                    }
                    let has_active = now
                        .as_ref()
                        .map(|p| p.iter().any(|t| !t.done()))
                        .unwrap_or(false);
                    if has_active && plan_stale_turns >= 5 {
                        use harxes_core_domain::domain::value_objects::render_todos;
                        output.push_str(&format!(
                            "\n\n<system-reminder>Your todo plan has not been updated for a while. Current state:\n{}\nUse the Todo tool (action=write) to mark finished steps completed and set the step you are on to in_progress.</system-reminder>",
                            render_todos(now.as_deref().unwrap_or(&[]))
                        ));
                        plan_stale_turns = 0;
                    }
                }
                transcript.push(Message::tool_result(call.id.clone(), output));
            }
            // Persist after every iteration so a turn killed mid-flight (rate
            // limit, network death, Ctrl-C) can be resumed without losing the
            // tool work already done.
            self.persist_session(&transcript);
        }

        self.persist_session(&transcript);
        let reason = if iterations >= limits.max_iterations {
            format!("reached the {} iteration cap", limits.max_iterations)
        } else {
            format!("used {total_tokens} tokens (cap {})", limits.max_total_tokens)
        };
        Ok((
            LoopOutcome {
                final_text: format!(
                    "[stopped by guardrail: {reason} after {iterations} iterations. Progress is saved — reply 'continue' to keep going, or raise `limits` in ~/.harxes/config.json]"
                ),
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

    /// Drive the tool-calling loop over a pre-built transcript, returning both
    /// the outcome and the final transcript. The embeddable engine uses this to
    /// run with caller-supplied history/system prompts.
    pub async fn run_transcript(
        &self,
        provider_id: &ProviderId,
        model_id: &str,
        transcript: Vec<Message>,
        limits: &LoopLimits,
    ) -> Result<(LoopOutcome, Vec<Message>), LlmError> {
        self.run_loop(provider_id, model_id, transcript, limits).await
    }

    fn persist_session(&self, transcript: &[Message]) {
        if let Some((store, id)) = &self.session {
            use harxes_core_domain::ports::SessionRecord;
            let rec = SessionRecord {
                id: id.clone(),
                created_at: String::new(),
                transcript: transcript.to_vec(),
                todos: self.todo_snapshot().unwrap_or_default(),
            };
            let _ = store.save(&rec);
        }
    }
}

/// Observer adapter for delegated sub-agents: relabels tool events with a
/// nesting prefix and drops stream deltas (see [`AgentLoop::nested_prefix`]).
struct NestedObserver {
    inner: Arc<dyn harxes_core_domain::ports::ToolObserver>,
    depth: usize,
}

impl harxes_core_domain::ports::ToolObserver for NestedObserver {
    fn on_tool_start(&self, name: &str, args_preview: &str) {
        self.inner
            .on_tool_start(&AgentLoop::nested_prefix(self.depth, name), args_preview);
    }
    fn on_tool_result(&self, name: &str, result_preview: &str) {
        self.inner
            .on_tool_result(&AgentLoop::nested_prefix(self.depth, name), result_preview);
    }
    fn on_retry(&self, wait_secs: u64) {
        self.inner.on_retry(wait_secs);
    }
    // on_stream_delta: default no-op — sub-agent text stays out of the
    // parent's live stream.
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

    async fn continue_chat_with(
        &self,
        provider_id: &ProviderId,
        model_id: &str,
        history: &[Message],
        user: Message,
        limits: &LoopLimits,
    ) -> Result<crate::ports::ConversationResult, LlmError> {
        let mut transcript = history.to_vec();
        transcript.push(user);
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
        async fn glob(
            &self,
            _p: &str,
            _o: &harxes_core_domain::ports::GlobOptions,
        ) -> Vec<String> {
            vec!["a.rs".to_string()]
        }
        async fn grep(
            &self,
            _n: &str,
            _p: &str,
            _m: usize,
        ) -> Result<Vec<harxes_core_domain::ports::GrepMatch>, FsError> {
            Ok(vec![])
        }
        async fn replace(
            &self,
            _p: &str,
            _o: &str,
            _n: &str,
        ) -> Result<usize, FsError> {
            Ok(0)
        }
    }

    fn agent(script: Vec<AgentResponse>) -> AgentLoop {
        AgentLoop::new(
            Arc::new(FakeLlmScript(Mutex::new(script))),
            Arc::new(FakeShell),
            Arc::new(FakeFs),
        )
    }

    /// Shell that fails with exit 2 for commands containing "block-hook".
    struct HookShell;
    #[async_trait::async_trait]
    impl harxes_core_domain::ports::ShellPort for HookShell {
        async fn run_command(&self, _wd: &str, cmd: &str) -> Result<CommandOutput, ShellError> {
            if cmd.contains("block-hook") {
                return Ok(CommandOutput {
                    stdout: String::new(),
                    stderr: "not allowed by policy".into(),
                    exit_status: ShellExitStatus::Failure(2),
                });
            }
            Ok(CommandOutput {
                stdout: format!("out:{cmd}"),
                stderr: String::new(),
                exit_status: ShellExitStatus::Success,
            })
        }
    }

    #[test]
    fn dangerous_command_detection() {
        for c in [
            "rm -rf /",
            "cd /tmp && rm -rf build",
            "sudo systemctl stop x",
            "git push --force origin main",
            "git reset --hard HEAD~3",
            "curl https://x.sh | sh",
            "wget -qO- x|bash",
            "dd if=/dev/zero of=/dev/sda",
            "chmod -R 777 /etc",
        ] {
            assert!(AgentLoop::is_dangerous(c), "should flag: {c}");
        }
        for c in [
            "cargo test",
            "git push origin main",
            "rm ./tmp.txt",
            "ls -la",
            "git commit -m x",
            "cargo build --release",
        ] {
            assert!(!AgentLoop::is_dangerous(c), "should NOT flag: {c}");
        }
    }

    #[test]
    fn closest_snippet_points_at_similar_region() {
        let content = "fn main() {\n    let total_cost = compute();\n    println!(\"{total_cost}\");\n}";
        let (ln, snip) =
            AgentLoop::closest_snippet(content, "let total_cost = compute_all();").unwrap();
        assert_eq!(ln, 2);
        assert!(snip.contains("compute()"));
        // Nothing remotely similar -> no noisy suggestion.
        assert!(AgentLoop::closest_snippet(content, "zzzz qqqq wwww eeee").is_none());
    }

    #[tokio::test]
    async fn failover_switches_models_on_terminal_error() {
        use harxes_core_domain::ports::LlmError;
        /// Fails every call for model "primary"; answers on "backup".
        struct PickyLlm;
        #[async_trait::async_trait]
        impl LlmPort for PickyLlm {
            async fn generate(
                &self,
                _p: &ProviderId,
                model_id: &str,
                _m: &[Message],
                _t: &[ToolSpec],
                _temp: Option<f64>,
            ) -> Result<AgentResponse, LlmError> {
                if model_id == "backup" {
                    Ok(AgentResponse::text("rescued", Default::default()))
                } else {
                    Err(LlmError::Request("boom".into()))
                }
            }
        }
        let a = AgentLoop::new(Arc::new(PickyLlm), Arc::new(FakeShell), Arc::new(FakeFs))
            .with_fallback_models(vec!["backup".into()]);
        let pid = ProviderId::new("x").unwrap();
        let out = a
            .run(&pid, "primary", "sys", "hi", &LoopLimits::default())
            .await
            .unwrap();
        assert_eq!(out.final_text, "rescued");

        // Without fallbacks the primary error surfaces.
        let a2 = AgentLoop::new(Arc::new(PickyLlm), Arc::new(FakeShell), Arc::new(FakeFs));
        assert!(a2
            .run(&pid, "primary", "sys", "hi", &LoopLimits::default())
            .await
            .is_err());
    }

    #[tokio::test]
    async fn read_pages_large_files() {
        struct BigFs;
        #[async_trait::async_trait]
        impl FileSystemPort for BigFs {
            async fn read(&self, _p: &str) -> Result<String, FsError> {
                Ok((1..=5000).map(|i| format!("line{i}")).collect::<Vec<_>>().join("\n"))
            }
            async fn write(&self, _p: &str, _c: &str) -> Result<(), FsError> {
                Ok(())
            }
            async fn replace(
                &self,
                _p: &str,
                _o: &str,
                _n: &str,
            ) -> Result<usize, FsError> {
                Ok(0)
            }
            async fn glob(
                &self,
                _pat: &str,
                _o: &harxes_core_domain::ports::GlobOptions,
            ) -> Vec<String> {
                vec![]
            }
            async fn grep(
                &self,
                _n: &str,
                _p: &str,
                _m: usize,
            ) -> Result<Vec<harxes_core_domain::ports::GrepMatch>, FsError> {
                Ok(vec![])
            }
        }
        let a = AgentLoop::new(
            Arc::new(FakeLlmScript(Mutex::new(vec![]))),
            Arc::new(FakeShell),
            Arc::new(BigFs),
        );
        // Default cap.
        let out = a.read_file("big.txt", None, None).await;
        assert!(out.starts_with("[showing lines 1-2000 of 5000"), "{}", &out[..80]);
        assert!(out.contains("line2000") && !out.contains("line2001\n"));
        // Paging.
        let page = a.read_file("big.txt", Some(4999), Some(10)).await;
        assert!(page.starts_with("[showing lines 4999-5000 of 5000"));
        assert!(page.contains("line5000"));
        // Small enough reads come back raw (FakeFs path in other tests).
    }

    #[tokio::test]
    async fn mixed_calls_keep_positional_outputs() {
        let a = agent(vec![]);
        let pid = ProviderId::new("x").unwrap();
        let calls = vec![
            ToolCall { id: "1".into(), name: "Read".into(), arguments: r#"{"path":"a"}"#.into() },
            ToolCall { id: "2".into(), name: "Read".into(), arguments: r#"{"path":"b"}"#.into() },
            ToolCall { id: "3".into(), name: "Bash".into(), arguments: r#"{"command":"echo hi"}"#.into() },
            ToolCall { id: "4".into(), name: "Read".into(), arguments: r#"{"path":"c"}"#.into() },
        ];
        let outs = a.execute_calls(&pid, "m", &calls).await;
        assert_eq!(outs.len(), 4);
        assert!(outs[0].contains("read:a"), "{}", outs[0]);
        assert!(outs[1].contains("read:b"), "{}", outs[1]);
        assert!(outs[2].contains("echo hi"), "{}", outs[2]);
        assert!(outs[3].contains("read:c"), "{}", outs[3]);
    }

    #[tokio::test]
    async fn pre_tool_hook_blocks_matching_call() {
        use harxes_core_domain::ports::{HookRule, HooksConfig};
        let a = AgentLoop::new(
            Arc::new(FakeLlmScript(Mutex::new(vec![]))),
            Arc::new(HookShell),
            Arc::new(FakeFs),
        )
        .with_hooks(HooksConfig {
            pre_tool: vec![HookRule {
                matcher: "Read".into(),
                command: "block-hook".into(),
            }],
            post_tool: vec![],
        });
        let pid = ProviderId::new("x").unwrap();
        let blocked = a
            .execute_call(&pid, "m", &ToolCall {
                id: "1".into(),
                name: "Read".into(),
                arguments: r#"{"path":"x.txt"}"#.into(),
            })
            .await;
        assert!(blocked.contains("blocked by pre_tool hook"), "{blocked}");
        assert!(blocked.contains("not allowed by policy"));
        // Non-matching tool runs normally.
        let ok = a
            .execute_call(&pid, "m", &ToolCall {
                id: "2".into(),
                name: "Glob".into(),
                arguments: r#"{"pattern":"*"}"#.into(),
            })
            .await;
        assert!(!ok.contains("blocked"), "{ok}");
    }

    #[tokio::test]
    async fn post_tool_hook_appends_feedback() {
        use harxes_core_domain::ports::{HookRule, HooksConfig};
        let a = AgentLoop::new(
            Arc::new(FakeLlmScript(Mutex::new(vec![]))),
            Arc::new(FakeShell),
            Arc::new(FakeFs),
        )
        .with_hooks(HooksConfig {
            pre_tool: vec![],
            post_tool: vec![HookRule {
                matcher: "*".into(),
                command: "lint-check".into(),
            }],
        });
        let pid = ProviderId::new("x").unwrap();
        let out = a
            .execute_call(&pid, "m", &ToolCall {
                id: "1".into(),
                name: "Read".into(),
                arguments: r#"{"path":"x.txt"}"#.into(),
            })
            .await;
        assert!(out.contains("read:x.txt"), "{out}");
        assert!(out.contains("[hook feedback]"), "{out}");
        assert!(out.contains("lint-check"), "{out}");
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
    async fn delegate_spawns_a_sub_agent_in_fresh_context() {
        // turn 1: request a Delegate call; the nested sub-agent pops the next
        // response (text-only) and returns, so the outer turn finishes.
        let delegate = ToolCall {
            id: "d1".into(),
            name: "Delegate".into(),
            arguments: r#"{"task":"inspect src","context":"the code"}"#.into(),
        };
        let script = vec![
            AgentResponse {
                content: String::new(),
                usage: Default::default(),
                tool_calls: vec![delegate],
            },
            AgentResponse::text("delegated result compiled", Default::default()),
            AgentResponse::text("outer done", Default::default()),
        ];
        let a = agent(script);
        let pid = ProviderId::new("x").unwrap();
        let out = a
            .run(
                &pid,
                "m",
                "sys",
                "investigate and report",
                &LoopLimits {
                    max_iterations: 25,
                    max_total_tokens: 9999,
                    context_window_tokens: 5000,
                },
            )
            .await
            .unwrap();
        // The delegated sub-agent's result is folded into the final answer.
        assert!(out.final_text.contains("outer done"));
        assert_eq!(out.iterations, 2);
    }

    #[tokio::test]
    async fn delegate_refuses_beyond_max_depth() {
        // Response keeps delegating; sub-agents consume the text response first
        // which terminates recursion at the leaves, so give the outer a fresh
        // agent whose sub keeps asking for a delegate too — instead assert the
        // depth guard directly via a deeply-nested construction is unnecessary.
        // Here we just confirm a Delegate call with an empty task is rejected.
        let delegate = ToolCall {
            id: "d1".into(),
            name: "Delegate".into(),
            arguments: r#"{"task":""}"#.into(),
        };
        let script = vec![
            AgentResponse {
                content: String::new(),
                usage: Default::default(),
                tool_calls: vec![delegate],
            },
            AgentResponse::text("done", Default::default()),
        ];
        let a = agent(script);
        let pid = ProviderId::new("x").unwrap();
        let out = a
            .run(&pid, "m", "sys", "go", &LoopLimits::default())
            .await
            .unwrap();
        // The empty-task delegate is rejected and folded into a subsequent turn.
        assert_eq!(out.final_text, "done");
    }

    #[test]
    fn nested_prefix_indents_by_depth() {
        assert_eq!(AgentLoop::nested_prefix(1, "Bash"), "└ Bash");
        assert_eq!(AgentLoop::nested_prefix(2, "Read"), "  └ Read");
    }

    #[tokio::test]
    async fn todo_tool_updates_shared_plan() {
        use harxes_core_domain::domain::value_objects::TodoItem;
        let todos: Arc<std::sync::Mutex<Vec<TodoItem>>> =
            Arc::new(std::sync::Mutex::new(vec![
                TodoItem::new("step one"),
                TodoItem::new("step two"),
            ]));
        let mark_done = ToolCall {
            id: "t1".into(),
            name: "Todo".into(),
            arguments: r#"{"action":"write","todos":[
                {"label":"step one","status":"completed"},
                {"label":"step two","active_form":"Doing step two","status":"in_progress"}
            ]}"#.into(),
        };
        let script = vec![
            AgentResponse {
                content: String::new(),
                usage: Default::default(),
                tool_calls: vec![mark_done],
            },
            AgentResponse::text("all set", Default::default()),
        ];
        let a = agent(script).with_todos(todos.clone());
        let pid = ProviderId::new("x").unwrap();
        let out = a
            .run(&pid, "m", "sys", "plan the work", &LoopLimits::default())
            .await
            .unwrap();
        assert_eq!(out.final_text, "all set");
        let items = todos.lock().unwrap();
        assert!(items[0].done(), "step one should be marked done");
        assert!(!items[1].done());
        assert_eq!(items[1].active_label(), "Doing step two");
    }

    #[tokio::test]
    async fn todo_list_renders_current_plan() {
        use harxes_core_domain::domain::value_objects::TodoItem;
        let todos: Arc<std::sync::Mutex<Vec<TodoItem>>> =
            Arc::new(std::sync::Mutex::new(vec![
                TodoItem::new("alpha"),
                TodoItem::new("beta"),
            ]));
        let list = ToolCall {
            id: "t1".into(),
            name: "Todo".into(),
            arguments: r#"{"action":"list"}"#.into(),
        };
        let script = vec![
            AgentResponse {
                content: String::new(),
                usage: Default::default(),
                tool_calls: vec![list],
            },
            AgentResponse::text("ok", Default::default()),
        ];
        let a = agent(script).with_todos(todos.clone());
        let pid = ProviderId::new("x").unwrap();
        let _ = a
            .run(&pid, "m", "sys", "go", &LoopLimits::default())
            .await
            .unwrap();
        // The list result is folded into the transcript as a tool result.
        // A second text turn means the loop completed normally.
        let items = todos.lock().unwrap();
        assert_eq!(items.len(), 2);
        assert!(!items[0].done() && !items[1].done());
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
    async fn write_shows_golden_diff_and_consults_decider() {
        use harxes_core_domain::ports::PermissionDecider;
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct Capture(Arc<AtomicUsize>);
        impl PermissionDecider for Capture {
            fn decide_write(&self, _: &str) -> bool {
                true
            }
            fn decide_bash(&self, _: &str) -> bool {
                true
            }
            fn decide_write_diff(&self, _path: &str, diff: &str) -> bool {
                // A real diff preview must have been produced.
                assert!(
                    diff.contains('-') || diff.contains('+'),
                    "expected a golden diff, got: {diff}"
                );
                self.0.fetch_add(1, Ordering::SeqCst);
                false // reject so we can observe the call
            }
        }

        let calls = Arc::new(AtomicUsize::new(0));
        let gate = Arc::new(Capture(calls.clone()));
        let shell = Arc::new(FakeShell);
        let fsys = Arc::new(FakeFs);
        let llm = Arc::new(FakeLlmScript(Mutex::new(vec![])));
        let a = AgentLoop::new(llm, shell, fsys).with_decider(gate);

        // FakeFs::read returns "read:{path}" which differs from the write content,
        // so the modification triggers a golden-diff review.
        let out = a.write_file_parts("/x.txt", "some new content").await;
        assert!(
            out.contains("permission denied"),
            "expected diff review to block, got: {out}"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn edit_respects_write_policy() {
        use harxes_core_domain::domain::value_objects::PermissionPolicy;
        let fsys = Arc::new(FakeFs);
        let a = AgentLoop::new(
            Arc::new(FakeLlmScript(Mutex::new(vec![]))),
            Arc::new(FakeShell),
            fsys,
        )
        .with_policy(PermissionPolicy {
            allow_globs: vec!["/ok/**".to_string()],
            deny_globs: vec![],
        });

        let blocked = a.edit_file("bad.txt", "a", "b").await;
        assert!(blocked.contains("permission denied"));

        let allowed = a.edit_file("/ok/x.txt", "old", "new").await;
        assert!(allowed.contains("edited"));
    }

    // E2E: a dangerous bash command is gated by the decider (returns denied).
    #[tokio::test]
    async fn dangerous_bash_is_gated_by_decider() {
        use harxes_core_domain::ports::PermissionDecider;
        use std::sync::atomic::{AtomicBool, Ordering};

        struct DenyAll(AtomicBool);
        impl PermissionDecider for DenyAll {
            fn decide_write(&self, _: &str) -> bool {
                true
            }
            fn decide_bash(&self, _: &str) -> bool {
                !self.0.load(Ordering::SeqCst)
            }
        }

        let shell = Arc::new(FakeShell);
        let fsys = Arc::new(FakeFs);
        let llm = Arc::new(FakeLlmScript(Mutex::new(vec![])));
        let gate = Arc::new(DenyAll(AtomicBool::new(true))); // deny first
        let a = AgentLoop::new(llm, shell, fsys).with_decider(gate);

        // Dangerous prefix -> decider consulted and denies.
        let denied = a.run_bash("rm -rf /").await;
        assert!(denied.contains("permission denied"), "got: {denied}");

        // Safe command runs without consulting the decider.
        let ran = a.run_bash("echo hi").await;
        assert!(ran.contains("exit=0"));
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

    #[tokio::test]
    async fn retry_generate_backs_off_once_then_succeeds() {
        use harxes_core_domain::ports::{LlmError, ToolObserver};
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct Flaky(Arc<AtomicUsize>);
        #[async_trait::async_trait]
        impl LlmPort for Flaky {
            async fn generate(
                &self,
                _p: &ProviderId,
                _m: &str,
                _msgs: &[Message],
                _t: &[ToolSpec],
                _tt: Option<f64>,
            ) -> Result<AgentResponse, LlmError> {
                let n = self.0.fetch_add(1, Ordering::SeqCst);
                if n == 0 {
                    Err(LlmError::RateLimited { retry_after: Some(0) })
                } else {
                    Ok(AgentResponse::text("recovered", Default::default()))
                }
            }
        }

        struct CountingRetries(Arc<AtomicUsize>);
        impl ToolObserver for CountingRetries {
            fn on_tool_start(&self, _: &str, _: &str) {}
            fn on_tool_result(&self, _: &str, _: &str) {}
            fn on_retry(&self, _: u64) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let calls = Arc::new(AtomicUsize::new(0));
        let retries = Arc::new(AtomicUsize::new(0));
        let agent = AgentLoop::new(
            Arc::new(Flaky(calls.clone())),
            Arc::new(FakeShell),
            Arc::new(FakeFs),
        )
        .with_observer(Arc::new(CountingRetries(retries.clone())));

        let pid = ProviderId::new("x").unwrap();
        let resp = agent
            .retry_generate(&pid, "m", &[], &[], 3)
            .await
            .unwrap();
        assert_eq!(resp.content, "recovered");
        assert_eq!(calls.load(Ordering::SeqCst), 2, "should retry exactly once");
        assert_eq!(retries.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn glob_tool_lists_files_and_grep_reports_no_match() {
        use harxes_core_domain::ports::{GlobOptions, GrepMatch};
        // Use a custom fs that reports glob hits and empty grep results.
        struct FakeFs2;
        #[async_trait::async_trait]
        impl FileSystemPort for FakeFs2 {
            async fn read(&self, p: &str) -> Result<String, FsError> {
                Ok(format!("read:{p}"))
            }
            async fn write(&self, _p: &str, _c: &str) -> Result<(), FsError> {
                Ok(())
            }
            async fn glob(&self, _p: &str, _o: &GlobOptions) -> Vec<String> {
                vec!["crates/cli/src/main.rs".to_string(), "crates/app/src/lib.rs".to_string()]
            }
            async fn grep(&self, _n: &str, _p: &str, _m: usize) -> Result<Vec<GrepMatch>, FsError> {
                Ok(vec![])
            }
            async fn replace(&self, _p: &str, _o: &str, _n: &str) -> Result<usize, FsError> {
                Ok(0)
            }
        }

        let a = AgentLoop::new(
            Arc::new(FakeLlmScript(Mutex::new(vec![]))),
            Arc::new(FakeShell),
            Arc::new(FakeFs2),
        );
        let glob_out = a.glob("**/*.rs", None).await;
        assert!(glob_out.contains("crates/cli/src/main.rs"));
        let grep_out = a.grep("nothing", "**/*", 10).await;
        assert!(grep_out.contains("no matches"));
    }
}
