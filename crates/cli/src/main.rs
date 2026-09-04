//! `harxes` CLI entrypoint.
//!
//! Composition root for the hexagonal application: wires real infrastructure
//! adapters into the application ports, then either runs a one-shot prompt or
//! an interactive REPL that carries conversation state across turns and can be
//! resumed later via a persisted session.

mod compose;
mod tui;
mod ui;

use clap::Parser;
use harxes_app::ports::agent::AgentPort;
use harxes_core_domain::domain::value_objects::{Message, ProviderId, Role};
use harxes_core_domain::ports::{SessionRecord, SessionStorePort};
use harxes_infra_fs::context::ContextStore;
use harxes_infra_session::{JsonConfigStore, JsonSessionStore};
use std::sync::Arc;

/// Harxes — a Rust coding agent harness.
#[derive(Debug, Parser)]
#[command(name = "harxes", version)]
struct Cli {
    /// One-shot prompt to run; omit for interactive REPL.
    #[arg(value_name = "PROMPT")]
    prompt: Option<String>,

    /// Provider id to use (e.g. anthropic|openai|litellm).
    #[arg(long)]
    provider: Option<String>,

    /// Model id override.
    #[arg(long)]
    model: Option<String>,

    /// Resume an existing session by its saved id.
    #[arg(long)]
    resume: Option<String>,

    /// List all saved sessions and exit.
    #[arg(long)]
    list: bool,

    /// List available providers and exit.
    #[arg(long)]
    providers: bool,

    /// Override the LLM base URL for the chosen provider.
    #[arg(long)]
    base_url: Option<String>,
}

fn main() {
    let cli = Cli::parse();
    // Cross-thread channel for live streaming text deltas (per-token typing in
    // the TUI). Only the interactive REPL drains the receiver.
    let (stream_delta_tx, stream_delta_rx) = std::sync::mpsc::channel::<String>();
    let wiring = match compose::assemble(
        cli.provider.as_deref(),
        cli.base_url.as_deref(),
        Some(stream_delta_tx),
    ) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("harxes : {e}");
            std::process::exit(1);
        }
    };

    // Create the standard agent-scoped project structure (AGENTS.md + .harxes/)
    // that is shared with other AI agents, before any work begins.
    compose::ensure_project_scaffold();

    if cli.providers {
        for p in compose::available_providers() {
            println!("{p}");
        }
        return;
    }
    if cli.list {
        let store = JsonSessionStore::new(compose::default_config_dir());
        let ids = store.list();
        if ids.is_empty() {
            println!("no saved sessions");
        } else {
            for id in ids {
                println!("{id}");
            }
        }
        return;
    }
    if let Some(prompt) = cli.prompt.clone() {
        run_one_shot(&wiring, &cli, &prompt);
        return;
    }
    run_repl(&wiring, &cli, stream_delta_rx);
}

fn resolve_model(wiring: &compose::Wiring, cli: &Cli) -> String {
    match compose::resolve_provider(&wiring.provider_id) {
        Ok(rp) => compose::effective_model(&rp, cli.model.as_deref()),
        Err(_) => "claude-sonnet-4-5".to_string(),
    }
}

fn make_pid(wiring: &compose::Wiring) -> ProviderId {
    ProviderId::new(&wiring.provider_id).unwrap_or_else(|_| ProviderId::new("anthropic").unwrap())
}

fn run_one_shot(wiring: &compose::Wiring, cli: &Cli, prompt: &str) {
    let pid = make_pid(wiring);
    let model = resolve_model(wiring, cli);
    // One-shot runs get the same workspace context (project guide, notes) as
    // REPL turns, folded into the system prompt.
    let ctx = ContextStore::new(std::path::PathBuf::from(".harxes"));
    let mut system = build_system_prompt();
    let block = ctx.build_context_block("one-shot");
    if !block.trim().is_empty() {
        system.push_str("\n\n");
        system.push_str(&block);
    }
    match tokio::runtime::Runtime::new() {
        Ok(rt) => rt.block_on(async {
            wiring.streamed.store(false, std::sync::atomic::Ordering::Relaxed);
            let history = vec![Message::new(Role::System, system.clone())];
            let user = harxes_core_domain::domain::value_objects::Message::user_with_images(
                expand_file_mentions(prompt),
                attach_images(prompt),
            );
            match wiring
                .agent
                .continue_chat_with(&pid, &model, &history, user, &wiring.limits)
                .await
            {
                Ok(res) => {
                    let out = res.outcome;
                    // If the live stream already printed the text, don't re-print it.
                    if !wiring.streamed.load(std::sync::atomic::Ordering::Relaxed) {
                        println!("{}", ui::render_assistant(&out.final_text));
                    }
                    if out.truncated_by_guardrail {
                        eprintln!(
                            "{}",
                            ui::C::yellow(&format!(
                                "⚠ stopped by guardrail after {} iterations",
                                out.iterations
                            ))
                        );
                    }
                    // Quiet, informative footer: tokens + estimated cost.
                    let cost = estimate_cost(&model, out.input_tokens, out.output_tokens);
                    eprintln!(
                        "{}",
                        ui::C::dim(&format!(
                            "· {} iterations · {} tokens (↑{} ↓{}) · ${:.4}",
                            out.iterations,
                            out.usage_total_tokens,
                            out.input_tokens,
                            out.output_tokens,
                            cost
                        ))
                    );
                }
                Err(e) => eprintln!("{}", ui::C::yellow(&format!("harxes error: {e}"))),
            }
        }),
        Err(e) => {
            eprintln!("harxes : failed to start runtime : {e}");
            std::process::exit(1);
        }
    }
}

/// Resolve --resume, mapping the special id "last" to the newest session.
fn resolve_resume_id(cli: &Cli) -> Option<String> {
    let id = cli.resume.as_ref()?;
    if id == "last" {
        JsonSessionStore::new(compose::default_config_dir())
            .list()
            .into_iter()
            .next()
    } else {
        Some(id.clone())
    }
}

/// Derive a short human slug from the first message, e.g.
/// "fix login bug in auth" -> "fix-login-bug-in".
fn session_slug(msg: &str) -> String {
    let words: Vec<String> = msg
        .split_whitespace()
        .take(4)
        .map(|w| {
            w.chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
                .to_lowercase()
        })
        .filter(|w| !w.is_empty())
        .collect();
    let slug: String = words.join("-").chars().take(28).collect();
    if slug.is_empty() {
        "sess".to_string()
    } else {
        slug
    }
}

/// Base64 (standard alphabet, padded) — tiny local encoder to avoid a dep.
fn base64_encode(data: &[u8]) -> String {
    const AB: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = u32::from(b[0]) << 16 | u32::from(b[1]) << 8 | u32::from(b[2]);
        out.push(AB[(n >> 18) as usize & 63] as char);
        out.push(AB[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 { AB[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if chunk.len() > 2 { AB[n as usize & 63] as char } else { '=' });
    }
    out
}

/// Expand `@path` mentions in a user message: for each `@relative/path` that
/// points at an existing readable text file, append its contents so the model
/// has them in context. The original text is preserved. Non-existent or
/// binary/huge files are left as-is (the model can still Read them).
fn expand_file_mentions(msg: &str) -> String {
    const MAX: u64 = 100_000;
    let mut appendix = String::new();
    let mut seen = std::collections::HashSet::new();
    for token in msg.split_whitespace() {
        let Some(raw) = token.strip_prefix('@') else { continue };
        let path = raw.trim_end_matches([',', '.', ';', ':', ')']);
        if path.is_empty() || !seen.insert(path.to_string()) {
            continue;
        }
        let Ok(meta) = std::fs::metadata(path) else { continue };
        if !meta.is_file() || meta.len() > MAX {
            continue;
        }
        if let Ok(content) = std::fs::read_to_string(path) {
            appendix.push_str(&format!("\n\n--- {path} ---\n{content}"));
        }
    }
    if appendix.is_empty() {
        msg.to_string()
    } else {
        format!("{msg}\n\n[Referenced files:]{appendix}")
    }
}

/// Scan a prompt for existing image-file paths and load them as attachments
/// (vision input). The prompt text is left untouched.
fn attach_images(prompt: &str) -> Vec<harxes_core_domain::domain::value_objects::ImageData> {
    const MAX_IMAGE_BYTES: u64 = 5_000_000;
    let mut out = Vec::new();
    for token in prompt.split_whitespace() {
        let path = token.trim_matches(['\'', '"', ',', ';']);
        let mime = match path.rsplit('.').next().map(|e| e.to_lowercase()).as_deref() {
            Some("png") => "image/png",
            Some("jpg") | Some("jpeg") => "image/jpeg",
            Some("gif") => "image/gif",
            Some("webp") => "image/webp",
            _ => continue,
        };
        let Ok(meta) = std::fs::metadata(path) else { continue };
        if !meta.is_file() || meta.len() > MAX_IMAGE_BYTES {
            continue;
        }
        if let Ok(bytes) = std::fs::read(path) {
            out.push(harxes_core_domain::domain::value_objects::ImageData {
                media_type: mime.to_string(),
                base64: base64_encode(&bytes),
            });
        }
    }
    out
}

/// The base system prompt: identity, environment, and working discipline.
/// Workspace context (project guide, notes) is appended by the caller.
/// The `/help` panel: every command, aligned, matching what's implemented.
const HELP_TEXT: &str = "\
Commands
  /help          show this help
  /clear         clear the screen
  /model <id>    switch model      · /models  list models on this provider
  /cost          tokens + estimated cost
  /compact       summarize the conversation to free context
  /undo          drop the last exchange
  /sessions      list saved sessions   · /resume <id>  load one (or --resume last)
  /plan <task>   break a task into a checklist
  /todo ...      list | done <n> | add <text> | clear
  /remember <x>  save a note to project memory (.harxes/agents/NOTES.md)
  /init          generate a HARXES.md project guide
  /mcp           list connected MCP servers and tools
  /theme <d|l>   dark or light
  /export <file> write the transcript to <file>.md
  /exit          quit  (Ctrl-C cancels a running turn)

Tip: mention @path/to/file to pull a file into the message.";

fn build_system_prompt() -> String {
    let cwd = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_default();
    let branch = {
        let head = std::path::Path::new(".git").join("HEAD");
        std::fs::read_to_string(head)
            .ok()
            .and_then(|c| c.strip_prefix("ref: refs/heads/").map(|b| b.trim().to_string()))
    };
    let mut env = format!("Working directory: {cwd}\nPlatform: {}", std::env::consts::OS);
    if let Some(b) = branch {
        env.push_str(&format!("\nGit branch: {b}"));
    }
    format!(
        "You are Harxes, a coding agent running in the user's terminal. You complete tasks \
         by calling tools; only your final text is shown to the user.\n\n\
         # Environment\n{env}\n\n\
         # Understand before you change\n\
         - Explore first. For anything non-trivial, use Grep/Glob/List/Read to find the \
         relevant files and understand how the feature is wired before editing. Don't guess \
         at file contents — read them.\n\
         - Follow the project's conventions: match the style, naming, and structure of \
         nearby code. Before using a library, confirm it's already a dependency (check the \
         manifest / imports); don't introduce new ones without reason.\n\
         - Check AGENTS.md / HARXES.md and any build/lint config for how this project \
         expects work to be done.\n\n\
         # Plan and execute\n\
         - For a task with more than one step, first write a plan with the Todo tool \
         (action=write), keep exactly one item in_progress, and mark items completed as you \
         finish them.\n\
         - Prefer the dedicated tools (Read, Edit, Write, Grep, Glob, List) over shell \
         equivalents (cat/grep/find/ls); use Bash for builds, tests, git, and package tools. \
         Bash resets to the project root each call — use absolute paths or chain with &&.\n\
         - Make surgical edits with Edit; use Write only for new files or full rewrites.\n\
         - Use Delegate for self-contained investigations that would bloat your context; \
         issuing several Delegate calls at once runs them in parallel.\n\n\
         # Verify and finish\n\
         - After changing code, run the project's build or tests before declaring success. \
         Report failures honestly — never claim something works if you haven't checked.\n\
         - When done, briefly state what you changed and why. Be concise: no preamble, no \
         restating the task, sparing markdown. Don't paste large file dumps back at the user.\n\
         - If the request is genuinely ambiguous, ask one clarifying question instead of \
         guessing; otherwise make a reasonable assumption, state it, and proceed.\n\n\
         # Safety\n\
         - Explain a side-effecting command before running it. Never run destructive \
         commands (rm -rf, force-push, hard reset, disk writes, piping downloads to a shell) \
         unless the user explicitly asked for that exact operation.\n\
         - Do not commit or push unless asked. Assist with defensive security and \
         legitimate work; decline to build malware or exfiltrate secrets."
    )
}

/// Build the initial todo store from plan labels: all pending except the
/// first step, which starts in_progress.
fn seed_todos(
    labels: Vec<String>,
) -> Vec<harxes_core_domain::domain::value_objects::TodoItem> {
    use harxes_core_domain::domain::value_objects::{TodoItem, TodoStatus};
    labels
        .into_iter()
        .enumerate()
        .map(|(i, l)| {
            let mut t = TodoItem::new(l);
            if i == 0 {
                t.status = TodoStatus::InProgress;
            }
            t
        })
        .collect()
}

fn run_repl(
    wiring: &compose::Wiring,
    cli: &Cli,
    stream_delta_rx: std::sync::mpsc::Receiver<String>,
) {
    let pid = make_pid(wiring);
    let model = resolve_model(wiring, cli);
    let session_id = match resolve_resume_id(cli) {
        Some(id) => id,
        None => format!(
            "sess-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0)
        ),
    };
    // The TUI polls the gate every frame, so approvals can be answered here.
    wiring.approval_gate.set_interactive(true);
    let mut state = tui::AppState::new(&wiring.provider_id, &model);
    // The tasks panel renders straight from the agent-managed todo store, so
    // Todo-tool writes show up live while the agent is working.
    state.todos = wiring.todos.clone();
    state.reasoning = wiring.reasoning.clone();
    state.active_tools = wiring.active_tools.clone();

    use harxes_core_domain::domain::value_objects::Role;
    // Seed chat pane from any resumed history.
    let (transcript, saved_todos) = load_history(cli);
    if !saved_todos.is_empty() {
        *wiring.todos.lock().unwrap() = saved_todos;
    }
    for m in &transcript {
        if m.role == Role::User {
            state.lines.push(tui::ChatLine::User(m.content.clone()));
        }
    }

    let rt = match tokio::runtime::Runtime::new() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("harxes : {e}");
            std::process::exit(1);
        }
    };

    ui::tui_mode(true);
    use std::cell::RefCell;
    let trx = RefCell::new(transcript);
    // Current active session id; mutable so /resume can switch mid-session.
    let sess_cell = RefCell::new(session_id.clone());

    use std::rc::Rc;
    let model_cell = Rc::new(RefCell::new(model));
    // Cross-thread event channel: background turn task -> TUI poll_events.
    let (tx, rx) =
        std::sync::mpsc::channel::<Result<(Vec<tui::ChatLine>, u64, u64, Vec<Message>), String>>();
    let config_dir = compose::default_config_dir();
    // Project-scoped agent context: agents write/re-read state under `<cwd>/.harxes`.
    let ctx_store = ContextStore::new(std::path::PathBuf::from(".harxes"));
    let handle = rt.handle().clone();
    // Handle of the in-flight turn task, so a Ctrl-C can abort it mid-turn.
    let turn_handle: std::sync::Arc<std::sync::Mutex<Option<tokio::task::JoinHandle<()>>>> =
        std::sync::Arc::new(std::sync::Mutex::new(None));

    state
        .lines
        .push(tui::ChatLine::Tool(format!("session {session_id}")));
    if state.lines.is_empty() {
        state.lines.push(tui::ChatLine::Agent(welcome_banner()));
    }
    for s in &wiring.mcp_status {
        state.lines.push(tui::ChatLine::Tool(s.clone()));
    }

    let result = tui::run(
        &mut state,
        |st, msg| {
            let t = msg.trim().to_string();
            if t.is_empty() {
                return;
            }
            // While an approval is pending, intercept y/n replies.
            if st.approval_prompt.is_some() {
                match t.to_lowercase().as_str() {
                    "y" | "yes" | "allow" => {
                        wiring.approval_gate.resolve(true);
                        st.lines.push(tui::ChatLine::Tool("allowed".into()));
                        return;
                    }
                    "n" | "no" | "deny" => {
                        wiring.approval_gate.resolve(false);
                        st.lines.push(tui::ChatLine::Tool("denied".into()));
                        return;
                    }
                    _ => {}
                }
            }
            // `/init` expands into a canned prompt and runs as a normal turn.
            let t = if t == "/init" {
                st.lines.push(tui::ChatLine::Tool(
                    "generating HARXES.md project guide…".into(),
                ));
                String::from(
                    "Explore this repository (use Glob/Read/Grep/Bash as needed) and write a \
                     concise HARXES.md project guide at the repo root using the Write tool. \
                     Cover: what the project is, how it is structured, how to build and test it, \
                     and any conventions a coding agent must follow. Keep it under 60 lines. \
                     If HARXES.md already exists, improve it instead of starting over.",
                )
            } else {
                t
            };
            // Slash commands handled here.
            match t.as_str() {
                "/help" => {
                    st.lines
                        .push(tui::ChatLine::Agent(String::from(HELP_TEXT)));
                    return;
                }
                "/cost" => {
                    let t = st.status.total_tokens;
                    let ti = st.status.total_input_tokens;
                    let to = st.status.total_output_tokens;
                    let mut out = format!("tokens: {t} (in {ti} / out {to})");
                    // Per-model breakdown (a session can switch models).
                    let mut rows: Vec<_> = st.status.per_model.iter().collect();
                    rows.sort_by(|a, b| a.0.cmp(b.0));
                    let mut total = 0.0;
                    for (model, (i, o)) in rows {
                        let c = estimate_cost(model, *i, *o);
                        total += c;
                        out.push_str(&format!(
                            "\n  {model}: {} (in {i} / out {o}) ~ ${c:.4}",
                            i + o
                        ));
                    }
                    if total == 0.0 {
                        total = estimate_cost(&st.status.model, ti, to);
                    }
                    out.push_str(&format!("\nestimated cost: ${total:.4}"));
                    st.lines.push(tui::ChatLine::Agent(out));
                    return;
                }
                "/mcp" => {
                    if wiring.mcp_status.is_empty() {
                        st.lines.push(tui::ChatLine::Tool(
                            "no MCP servers configured — add them under \"mcp\" in ~/.harxes/config.json".into(),
                        ));
                    } else {
                        for s in &wiring.mcp_status {
                            st.lines.push(tui::ChatLine::Tool(s.clone()));
                        }
                        for t in &wiring.mcp_tools {
                            st.lines.push(tui::ChatLine::Tool(format!("  · {t}")));
                        }
                    }
                    return;
                }
                "/models" => {
                    match rt.block_on(wiring.llm.list_models()) {
                        Ok(ms) if !ms.is_empty() => {
                            for m in ms.iter().take(30) {
                                st.lines.push(tui::ChatLine::Tool(format!("· {m}")));
                            }
                            // Pre-fill a pick-list: arrows choose, Enter fills
                            // "/model <id>", Enter again switches.
                            st.input = "/model ".to_string();
                            st.cursor_ix = st.input.len();
                            st.completions =
                                ms.iter().take(30).map(|m| format!("/model {m}")).collect();
                            st.completion_sel = 0;
                            st.lines.push(tui::ChatLine::Tool(
                                "pick a model with ↑/↓ then Enter".into(),
                            ));
                        }
                        Ok(_) => st.lines.push(tui::ChatLine::Tool(
                            "provider does not expose a model list".into(),
                        )),
                        Err(e) => st
                            .lines
                            .push(tui::ChatLine::Agent(format!("models error: {e}"))),
                    }
                    return;
                }
                "/sessions" => {
                    let store = JsonSessionStore::new(config_dir.clone());
                    let ids = store.list();
                    if ids.is_empty() {
                        st.lines
                            .push(tui::ChatLine::Agent(String::from("no saved sessions")));
                        return;
                    }
                    for id in ids.iter().take(15) {
                        let (n, first) = store
                            .load(id)
                            .map(|r| {
                                let f = r
                                    .transcript
                                    .iter()
                                    .find(|m| m.role == Role::User)
                                    .map(|m| {
                                        m.content.chars().take(48).collect::<String>()
                                    })
                                    .unwrap_or_default();
                                (r.transcript.len(), f)
                            })
                            .unwrap_or((0, String::new()));
                        st.lines.push(tui::ChatLine::Tool(format!(
                            "{id}  ({n} msgs)  {first}"
                        )));
                    }
                    st.lines.push(tui::ChatLine::Tool(
                        "resume with /resume <id> (or --resume last)".into(),
                    ));
                    return;
                }
                _ if t.starts_with("/resume ") => {
                    let id = t.trim_start_matches("/resume ").trim().to_string();
                    if id.is_empty() {
                        st.lines.push(tui::ChatLine::Agent(String::from(
                            "usage: /resume <session-id>",
                        )));
                        return;
                    }
                    let store = JsonSessionStore::new(config_dir.clone());
                    match store.load(&id) {
                        Some(rec) => {
                            *trx.borrow_mut() = rec.transcript.clone();
                            *sess_cell.borrow_mut() = id.clone();
                            st.lines.clear();
                            // Replay the whole conversation, not just the user
                            // side, so a resumed session reads like it did.
                            for m in &rec.transcript {
                                match m.role {
                                    Role::User => st
                                        .lines
                                        .push(tui::ChatLine::User(m.content.clone())),
                                    Role::Assistant if !m.content.trim().is_empty() => st
                                        .lines
                                        .push(tui::ChatLine::Agent(m.content.clone())),
                                    Role::Assistant => {
                                        for tc in &m.tool_calls {
                                            let d = harxes_core_domain::domain::services::tool_protocol::tool_summary(
                                                tc.name.as_str(),
                                                tc.arguments.as_str(),
                                            );
                                            st.lines.push(tui::ChatLine::Tool(d));
                                        }
                                    }
                                    _ => {}
                                }
                            }
                            *wiring.todos.lock().unwrap() = rec.todos.clone();
                            st.scroll = 0;
                            st.lines.push(tui::ChatLine::Tool(format!(
                                "resumed session {id} ({} messages)",
                                rec.transcript.len()
                            )));
                        }
                        None => st
                            .lines
                            .push(tui::ChatLine::Agent(format!("no session {id}"))),
                    }
                    return;
                }
                _ if t.starts_with("/plan") => {
                    let task = t.trim_start_matches("/plan").trim().to_string();
                    if task.is_empty() {
                        st.lines
                            .push(tui::ChatLine::Agent(String::from("usage: /plan <task>")));
                        return;
                    }
                    st.processing = true;
                    let cur_model = model_cell.borrow().clone();
                    let pid2 = make_pid(wiring);
                    let labels = rt.block_on(generate_plan(
                        wiring.agent.clone(),
                        &pid2,
                        &cur_model,
                        &task,
                    ));
                    if labels.is_empty() {
                        st.lines.push(tui::ChatLine::Agent(String::from(
                            "could not generate plan",
                        )));
                    } else {
                        st.set_plan(labels.clone());
                        // Seed the shared, agent-managed todo store; the first
                        // step starts in_progress.
                        *wiring.todos.lock().unwrap() = seed_todos(labels);
                        for (i, t) in st.plan.iter().enumerate() {
                            st.lines.push(tui::ChatLine::Tool(format!(
                                "{} [ ] {}",
                                i + 1,
                                t.label
                            )));
                        }
                        st.lines.push(tui::ChatLine::Tool(
                            "plan is active — the agent tracks progress via its Todo tool".into(),
                        ));
                    }
                    st.processing = false;
                    return;
                }

                _ if t.starts_with("/todo") => {
                    let arg = t.trim_start_matches("/todo").trim();
                   let verb = arg.split_whitespace().next();
                   match verb {
                       Some("done") => {
                           let n = arg
                               .split_whitespace()
                               .nth(1)
                               .and_then(|s| s.parse::<usize>().ok());
                           match n {
                               Some(n) if n >= 1 => {
                                   let mut store = wiring.todos.lock().unwrap();
                                   if n <= store.len() {
                                       store[n - 1].status =
                                           harxes_core_domain::domain::value_objects::TodoStatus::Completed;
                                       let label = store[n - 1].label.clone();
                                       if let Some(item) = st.plan.get_mut(n - 1) {
                                           item.done = true;
                                       }
                                       st.lines.push(tui::ChatLine::Tool(format!(
                                           "done: {label}"
                                       )));
                                   } else {
                                       st.lines.push(tui::ChatLine::Agent(format!(
                                           "no item {n} (plan has {})",
                                           store.len()
                                       )));
                                   }
                               }
                               _ => st.lines.push(tui::ChatLine::Agent(String::from(
                                   "usage: /todo done <n>",
                               ))),
                           }
                       }
                       Some("clear") => {
                           wiring.todos.lock().unwrap().clear();
                           st.plan.clear();
                           st.lines.push(tui::ChatLine::Tool("plan cleared".into()));
                       }
                       Some("add") => {
                           let label = arg
                               .strip_prefix("add")
                               .unwrap_or("")
                               .trim()
                               .to_string();
                           if label.is_empty() {
                               st.lines.push(tui::ChatLine::Agent(String::from(
                                   "usage: /todo add <text>",
                               )));
                           } else {
                               wiring.todos.lock().unwrap().push(
                                   harxes_core_domain::domain::value_objects::TodoItem::new(
                                       label.clone(),
                                   ),
                               );
                               st.plan.push(tui::TaskItem::new(label.clone()));
                               st.lines.push(tui::ChatLine::Tool(format!(
                                   "added todo: {label}"
                               )));
                           }
                       }
                       Some("list") | None => {
                           let store = wiring.todos.lock().unwrap();
                           st.plan = store
                               .iter()
                               .map(|i| {
                                   let mut ti = tui::TaskItem::new(i.label.clone());
                                   ti.done = i.done();
                                   ti
                               })
                               .collect();
                           if store.is_empty() {
                               st.lines.push(tui::ChatLine::Tool(
                                   "(empty plan — use /plan <task>)".into(),
                               ));
                           } else {
                               use harxes_core_domain::domain::value_objects::TodoStatus;
                               for (i, item) in store.iter().enumerate() {
                                   let m = match item.status {
                                       TodoStatus::Completed => "[✓]",
                                       TodoStatus::InProgress => "[▸]",
                                       TodoStatus::Pending => "[ ]",
                                   };
                                   st.lines.push(tui::ChatLine::Tool(format!(
                                       "{} {} {}",
                                       i + 1,
                                       m,
                                       item.label
                                   )));
                               }
                           }
                       }
                       _ => st.lines.push(tui::ChatLine::Agent(String::from(
                           "usage: /todo done <n> | /todo add <text> | /todo list | /todo clear",
                       ))),
                   }
                   return;
                }

                _ if t == "/theme" || t.starts_with("/theme ") => {
                    let arg = t.trim_start_matches("/theme").trim().to_lowercase();
                    match arg.as_str() {
                        "light" => {
                            st.theme = tui::PaneTheme::light();
                            st.lines.push(tui::ChatLine::Tool("theme: light".into()));
                        }
                        "dark" => {
                            st.theme = tui::PaneTheme::dark();
                            st.lines.push(tui::ChatLine::Tool("theme: dark".into()));
                        }
                        _ => st.lines.push(tui::ChatLine::Agent(String::from(
                            "usage: /theme dark|light",
                        ))),
                    }
                    return;
                }
                _ if t.starts_with("/export ") => {
                    let path = t.trim_start_matches("/export ").trim().to_string();
                    if path.is_empty() {
                        st.lines.push(tui::ChatLine::Agent(String::from(
                            "usage: /export <file.md|.html>",
                        )));
                        return;
                    }
                    let tb = trx.borrow();
                    let mut body = String::new();
                    for m in tb.iter() {
                        match m.role {
                            Role::User => body.push_str(&format!("## User\n{}\n\n", m.content)),
                            Role::System => body.push_str(&format!("_system:_ {}\n\n", m.content)),
                            _ => body.push_str(&format!("## Assistant\n{}\n\n", m.content)),
                        }
                        for tc in &m.tool_calls {
                            body.push_str(&format!(
                                "> tool: **{}**\n```json\n{}\n```\n",
                                tc.name, tc.arguments
                            ));
                        }
                    }
                    let sess = sess_cell.borrow().clone();
                    let meta = format!("**session:** {sess}\n\n**model:** {}  \n**tokens:** {} (in {}/out {})  \n**cost:** ${:.4}\n\n", st.status.model, st.status.total_tokens, st.status.total_input_tokens, st.status.total_output_tokens, st.status.total_cost);
                    let is_html = path.to_lowercase().ends_with(".html");
                    let content = if is_html {
                        html_page(&sess, &meta, &body)
                    } else {
                        format!("# Harxes session\n\n{}\n{}", meta, body)
                    };
                    match std::fs::write(&path, &content) {
                        Ok(_) => st.lines.push(tui::ChatLine::Tool(format!(
                            "exported {} to {path}",
                            if is_html { "html" } else { "markdown" }
                        ))),
                        Err(e) => st.lines.push(tui::ChatLine::Agent(format!("error: {e}"))),
                    }
                    drop(tb);
                    return;
                }
                _ if t.starts_with("/model ") => {
                    let newm = t.trim_start_matches("/model ").trim().to_string();
                    if newm.is_empty() {
                        st.lines
                            .push(tui::ChatLine::Agent(String::from("usage: /model <name>")));
                        return;
                    }
                    *model_cell.borrow_mut() = newm.clone();
                    st.status.model = newm.clone();
                    st.lines
                        .push(tui::ChatLine::Agent(format!("switched model to {newm}")));
                    return;
                }
                "/compact" => {
                    let mut tb = trx.borrow_mut();
                    if tb.is_empty() {
                        drop(tb);
                        st.lines
                            .push(tui::ChatLine::Agent(String::from("nothing to compact")));
                        return;
                    }
                    st.processing = true;
                    let cur_model = model_cell.borrow().clone();
                    let prompt = "Summarize this conversation into a concise recap that preserves all key facts, decisions, constraints and the current task. Return ONLY the summary text.";
                    match rt.block_on(wiring.agent.continue_chat(
                        &pid,
                        &cur_model,
                        &tb[..],
                        prompt,
                        &wiring.limits,
                    )) {
                        Ok(res) => {
                            let before = tb.len();
                            let summary = res.outcome.final_text.clone();
                            *tb = vec![Message::new(
                                Role::System,
                                format!("Previous conversation summary: {summary}"),
                            )];
                            // Reset the visible transcript to match, so screen
                            // and context don't desync; keep a short recap.
                            st.lines.clear();
                            st.scroll = 0;
                            st.auto_scroll = true;
                            st.lines.push(tui::ChatLine::Tool(format!(
                                "context compacted — {before} messages → 1 summary"
                            )));
                            st.lines.push(tui::ChatLine::Agent(summary));
                        }
                        Err(e) => st.lines.push(tui::ChatLine::Agent(format!("error: {e}"))),
                    }
                    drop(tb);
                    st.processing = false;
                    return;
                }
                "/undo" => {
                    let mut tb = trx.borrow_mut();
                    let mut cut = None;
                    for (i, m) in tb.iter().enumerate() {
                        if m.role == Role::User {
                            cut = Some(i);
                        }
                    }
                    if let Some(c) = cut {
                        tb.truncate(c);
                    }
                    drop(tb);
                    if let Some(pos) = st
                        .lines
                        .iter()
                        .rposition(|l| matches!(l, tui::ChatLine::User(_)))
                    {
                        st.lines.truncate(pos);
                    }
                    st.lines.push(tui::ChatLine::Tool("undid last turn".into()));
                    return;
                }
                _ if t.starts_with("/remember ") => {
                    let note = t.trim_start_matches("/remember ").trim().to_string();
                    if note.is_empty() {
                        st.lines.push(tui::ChatLine::Agent(String::from(
                            "usage: /remember <note>",
                        )));
                    } else {
                        let cur_sid = sess_cell.borrow().clone();
                        match ctx_store.remember_session(&cur_sid, &note) {
                            Ok(_) => {
                                let _ =
                                    ctx_store.remember_workspace(&format!("[{cur_sid}] {note}"));
                                st.lines
                                    .push(tui::ChatLine::Tool(format!("remembered: {note}")));
                            }
                            Err(e) => st
                                .lines
                                .push(tui::ChatLine::Agent(format!("error saving memory: {e}"))),
                        }
                    }
                    return;
                }
                _ => {}
            }

            if t.starts_with('/') {
                // Unrecognized slash command: tell the user instead of
                // silently swallowing it.
                let cmd = t.split_whitespace().next().unwrap_or("/");
                st.lines.push(tui::ChatLine::Agent(format!(
                    "unknown command {cmd} — type /help for the list"
                )));
                return;
            }
            // First message of a fresh session: bake a readable slug into the
            // session id so /sessions and --resume are human-friendly.
            if trx.borrow().is_empty() {
                let ts = sess_cell
                    .borrow()
                    .rsplit('-')
                    .next()
                    .unwrap_or("0")
                    .to_string();
                let named = format!("{}-{}", session_slug(&t), ts);
                *sess_cell.borrow_mut() = named.clone();
                st.lines.push(tui::ChatLine::Tool(format!("session {named}")));
            }
            st.lines.push(tui::ChatLine::User(msg.clone()));
            st.processing = true;
            // No auto-planning here: the agent plans for itself via the Todo
            // tool when a task warrants it (an upfront blocking generate_plan
            // call doubled latency and turned conversational messages into
            // garbage plans). Explicit /plan remains available.
            if trx.borrow().len() > 60 {
                st.lines.push(tui::ChatLine::Tool(
                    "context is getting long — consider /compact".into(),
                ));
            }
            let cur_model = model_cell.borrow().clone();
            // Snapshot owned values for the background turn task.
            let agent = wiring.agent.clone();
            let pid_str = wiring.provider_id.clone();
            let cfg_dir = config_dir.clone();
            let ctx_for_turn = ctx_store.clone();
            let sid = sess_cell.borrow().clone();
            let limits = wiring.limits.clone();
            let history_snapshot = trx.borrow().clone();
            let txc = tx.clone();
            let handle2 = handle.clone();
            let turn_slot = turn_handle.clone();
            let todos_for_turn = wiring.todos.clone();
            let jh = handle2.spawn(async move {
                let r = run_turn_owned(
                    agent,
                    pid_str,
                    cur_model,
                    cfg_dir,
                    ctx_for_turn,
                    sid,
                    history_snapshot,
                    t,
                    limits,
                    todos_for_turn,
                )
                .await;
                let _ = txc.send(r);
            });
            *turn_slot.lock().unwrap() = Some(jh);
        },
        |st| {
            // Drain live streaming deltas first so the typewriter reveals text
            // as the model emits it (per-token), before any completed-turn batch.
            while let Ok(delta) = stream_delta_rx.try_recv() {
                if !delta.is_empty() {
                    st.streaming_turn = true;
                    st.live_text.push_str(&delta);
                    st.typing_text = st.live_text.clone();
                    // Real text is arriving — thinking is done for this turn.
                    st.reasoning.lock().unwrap().clear();
                }
            }
            // Surface any outstanding approval request for the user to answer.
            st.approval_prompt = wiring.approval_gate.has_pending();
            while let Ok(ev) = rx.try_recv() {
                match ev {
                    Ok((lines, in_tok, out_tok, new_hist)) => {
                        for l in lines {
                            match l {
                                // If the live stream already surfaced the text,
                                // don't re-type it from the completed batch.
                                tui::ChatLine::Agent(_txt) if st.streaming_turn => {}
                                tui::ChatLine::Agent(txt) => {
                                    st.typing_text = txt;
                                    st.typing_shown = 0;
                                }
                                other => st.lines.push(other),
                            }
                        }
                        let tok = in_tok.saturating_add(out_tok);
                        st.status.total_input_tokens += in_tok;
                        st.status.total_output_tokens += out_tok;
                        st.status.total_tokens += tok;
                        let m = st.status.model.clone();
                        let e = st.status.per_model.entry(m).or_insert((0, 0));
                        e.0 += in_tok;
                        e.1 += out_tok;
                        st.status.total_cost = st
                            .status
                            .per_model
                            .iter()
                            .map(|(model, (i, o))| estimate_cost(model, *i, *o))
                            .sum();
                        *trx.borrow_mut() = new_hist;
                        // Auto-tick plan items whose key words appeared.
                        let combined: String = trx
                            .borrow()
                            .iter()
                            .filter(|m| m.role == Role::Assistant)
                            .map(|m| m.content.clone())
                            .collect::<Vec<_>>()
                            .join(" ");
                        st.plan_auto_tick(&combined);
                        st.processing = false;
                        // Turn complete: stop live-stream mode for the next turn.
                        st.streaming_turn = false;
                        st.live_text.clear();
                        st.active_tools.lock().unwrap().clear();
                        st.reasoning.lock().unwrap().clear();
                    }
                    Err(e) => {
                        st.active_tools.lock().unwrap().clear();
                        let sid = sess_cell.borrow().clone();
                        st.lines.push(tui::ChatLine::Agent(format!(
                            "error: {e}\nProgress so far is saved — retry here, or later run: harxes --resume {sid}"
                        )));
                        st.processing = false;
                    }
                }
            }
        },
        || {
            // At the end of a cancelled turn, ensure the approval gate isn't
            // left blocked (a worker may be parked on a pending request).
            let aborted = if let Ok(mut slot) = turn_handle.lock() {
                if let Some(jh) = slot.take() {
                    jh.abort();
                    true
                } else {
                    false
                }
            } else {
                false
            };
            if aborted {
                wiring.approval_gate.cancel_all();
            }
        },
    );
    if let Err(e) = result {
        eprintln!("harxes tui error: {e}");
    }
    // End-of-session usage report, per model.
    if state.status.total_tokens > 0 {
        let mut rows: Vec<_> = state.status.per_model.iter().collect();
        rows.sort_by(|a, b| a.0.cmp(b.0));
        let mut total = 0.0;
        println!(
            "session usage: {} tokens (in {} / out {})",
            state.status.total_tokens,
            state.status.total_input_tokens,
            state.status.total_output_tokens
        );
        for (model, (i, o)) in rows {
            let c = estimate_cost(model, *i, *o);
            total += c;
            println!("  {model}: {} (in {i} / out {o}) ~ ${c:.4}", i + o);
        }
        if total == 0.0 {
            total = estimate_cost(
                &state.status.model,
                state.status.total_input_tokens,
                state.status.total_output_tokens,
            );
        }
        println!("estimated cost: ${total:.4}");
    }
}

fn load_history(
    cli: &Cli,
) -> (
    Vec<Message>,
    Vec<harxes_core_domain::domain::value_objects::TodoItem>,
) {
    match resolve_resume_id(cli) {
        Some(id) => {
            let store = JsonSessionStore::new(compose::default_config_dir());
            match store.load(&id) {
                Some(rec) => (rec.transcript, rec.todos),
                None => {
                    eprintln!("harxes: no saved session '{id}', starting fresh");
                    (vec![], vec![])
                }
            }
        }
        None => (vec![], vec![]),
    }
}

/// Run one conversational turn from owned values (usable inside tokio::spawn).
/// Returns display lines, token usage, and the updated transcript.
#[allow(clippy::too_many_arguments)]
async fn run_turn_owned(
    agent: Arc<dyn AgentPort>,
    pid_str: String,
    model: String,
    config_dir: String,
    ctx_store: ContextStore,
    session_id: String,
    history: Vec<Message>,
    user_msg: String,
    limits: harxes_app::usecases::agent_loop::LoopLimits,
    todos: Arc<
        std::sync::Mutex<Vec<harxes_core_domain::domain::value_objects::TodoItem>>,
    >,
) -> Result<(Vec<tui::ChatLine>, u64, u64, Vec<Message>), String> {
    let pid = ProviderId::new(&pid_str).unwrap_or_else(|_| ProviderId::new("anthropic").unwrap());
    let store = JsonSessionStore::new(config_dir);
    // Agent started working: ensure its session context directory exists.
    let _ = ctx_store.ensure_session(&session_id);
    // Load persisted session + workspace memory into the transcript.
    let mut messages = history;
    // Fresh conversations get the full system prompt; resumed ones may already
    // carry one, so only insert when the transcript has no system message yet.
    let has_system = messages.iter().any(|m| m.role == Role::System);
    let mut system = build_system_prompt();
    let ctx_block = ctx_store.build_context_block(&session_id);
    if !ctx_block.trim().is_empty() {
        system.push_str("\n\n");
        system.push_str(&ctx_block);
    }
    if !has_system {
        messages.insert(0, Message::new(Role::System, system));
    }
    let images = attach_images(&user_msg);
    // Expand @path references by appending the referenced files' contents.
    let augmented = expand_file_mentions(&user_msg);
    let user = harxes_core_domain::domain::value_objects::Message::user_with_images(
        augmented,
        images,
    );
    match agent
        .continue_chat_with(&pid, &model, &messages, user, &limits)
        .await
    {
        Ok(res) => {
            let in_tok = res.outcome.input_tokens;
            let out_tok = res.outcome.output_tokens;
            let new_hist = res.transcript.clone();
            let _ = store.save(&SessionRecord {
                id: session_id,
                created_at: String::new(),
                transcript: new_hist.clone(),
                todos: todos.lock().unwrap().clone(),
            });
            // Emit what the agent did and said, in chronological order.
            let mut results: std::collections::HashMap<String, String> = Default::default();
            for m in &new_hist {
                if m.role == Role::Tool {
                    if let Some(id) = &m.tool_call_id {
                        results.insert(id.clone(), m.content.clone());
                    }
                }
            }
            let mut lines: Vec<tui::ChatLine> = Vec::new();
            for m in &new_hist {
                // Surface the agent’s spoken (non-empty) commentary.
                if m.role == Role::Assistant && !m.content.trim().is_empty() {
                    lines.push(tui::ChatLine::Agent(m.content.clone()));
                }
                // Pair each requested tool call with its result.
                for tc in &m.tool_calls {
                    let detail = harxes_core_domain::domain::services::tool_protocol::tool_summary(
                        tc.name.as_str(),
                        tc.arguments.as_str(),
                    );
                    match results.get(&tc.id) {
                        Some(out) => lines
                            .push(tui::ChatLine::Tool(format!("{detail}\n{}", pretty_result(out)))),
                        None => lines.push(tui::ChatLine::Tool(detail)),
                    }
                }
            }
            // Final answer last.
            lines.push(tui::ChatLine::Agent(res.outcome.final_text.clone()));

            Ok((lines, in_tok, out_tok, new_hist))
        }
        Err(e) => Err(e.to_string()),
    }
}

/// Format a tool result for the collapsed tool row: a short, readable summary
/// (exit code / line count) plus the first lines of output, indented.
fn pretty_result(out: &str) -> String {
    let t = out.trim();
    // Bash results look like "exit=0 stdout=... stderr=...": surface the code.
    let head = if let Some(rest) = t.strip_prefix("exit=") {
        let code = rest.split_whitespace().next().unwrap_or("?");
        let ok = code == "0";
        format!("    {} exit {code}", if ok { "✓" } else { "✗" })
    } else {
        let n = t.lines().count();
        format!("    ✓ {n} line{}", if n == 1 { "" } else { "s" })
    };
    // Add up to 3 preview lines of the body.
    let body: Vec<String> = t
        .lines()
        .filter(|l| !l.trim().is_empty())
        .take(3)
        .map(|l| {
            let s: String = l.chars().take(96).collect();
            format!("    {s}")
        })
        .collect();
    if body.is_empty() {
        head
    } else {
        format!("{head}\n{}", body.join("\n"))
    }
}

/// Ask the model to break a task into a checklist; returns parsed labels.
async fn generate_plan(
    agent: Arc<dyn AgentPort>,
    pid: &ProviderId,
    model: &str,
    task: &str,
) -> Vec<String> {
    let prompt = format!("Break this task into a concise checklist of 2-6 actionable subtasks. Return ONLY the subtasks, one per line, with no numbering, no markdown bullets, no intro or outro text.\n\nTask: {task}");
    match agent
        .run(
            pid,
            model,
            "You are a task planner.",
            &prompt,
            &harxes_app::usecases::agent_loop::LoopLimits::default(),
        )
        .await
    {
        Ok(out) => out
            .final_text
            .lines()
            .map(|l| {
                l.trim()
                    .trim_start_matches('-')
                    .trim_start_matches("*")
                    .trim()
                    .to_string()
            })
            .filter(|l| !l.is_empty())
            .collect(),
        Err(_) => vec![],
    }
}

/// Render the session as a self-contained dark-styled HTML page.
fn html_page(session: &str, meta: &str, body: &str) -> String {
    let esc_body = body
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;");
    let tmpl = r#"<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Harxes session</title>
<style>
body{background:#0d1117;color:#e6edf3;font-family:system-ui,sans-serif;max-width:820px;margin:40px auto;padding:0 20px}
h1{color:#58a6ff}h2{color:#79c0ff}pre{background:#161b22;padding:12px;border-radius:8px;overflow-x:auto}
code{background:#161b22;padding:2px 5px;border-radius:4px}blockquote{border-left:3px solid #30363d;margin-left:0;padding-left:14px}
</style></head><body>
<h1>Harxes session</h1>
<p><strong>session:</strong> {session}</p>
<hr>
<h2>Metadata</h2><div>{meta}</div>
<h2>Transcript</h2><pre>{esc_body}</pre>
</body></html>"#;
    tmpl.replace("{session}", session)
        .replace("{meta}", meta)
        .replace("{esc_body}", &esc_body)
}

/// Modern ASCII-art welcome banner shown when the TUI starts.
fn welcome_banner() -> String {
    let logo = r#"
       ___  __               
      / _ |/ /__  ____ ___   
     / __ / / _ \/ __ `__ \ 
    /_/ |_/_/\___/\_/ /_/ /_/
"#;
    format!("{}\n\n  Harxes v{} — a Rust coding agent that plans, runs tools and remembers.\n  \n  Type a task below and press Enter, or use commands:\n  /plan <task>   break a task into a checklist\n  /theme         switch light/dark\n  /export F      save session (markdown or HTML)\n  /resume <id>   load a past session\n  /help          all commands", logo, env!("CARGO_PKG_VERSION"))
}

/// (input, output) USD per million tokens for a model id. Config `pricing`
/// entries (substring-matched, loaded once) override the built-in table.
fn model_pricing(model: &str) -> (f64, f64) {
    use std::sync::OnceLock;
    static OVERRIDES: OnceLock<
        std::collections::BTreeMap<String, harxes_core_domain::ports::ModelPricing>,
    > = OnceLock::new();
    let overrides = OVERRIDES
        .get_or_init(|| {
            use harxes_core_domain::ports::ConfigStorePort;
            JsonConfigStore::new(compose::default_config_dir()).load().pricing
        });
    let m = model.to_lowercase();
    for (needle, p) in overrides {
        if m.contains(&needle.to_lowercase()) {
            return (p.input_per_mtok, p.output_per_mtok);
        }
    }
    match m.as_str() {
        s if s.contains("claude-opus") => (15.0, 75.0),
        s if s.contains("claude-sonnet") => (3.0, 15.0),
        s if s.contains("claude-haiku") => (0.8, 4.0),
        s if s.contains("gpt-4o-mini") => (0.15, 0.6),
        s if s.contains("gpt-4o") => (2.5, 10.0),
        s if s.contains("deepseek") => (0.3, 1.2),
        s if s.contains("glm") => (0.6, 2.2),
        s if s.contains("qwen") => (0.4, 1.2),
        _ => (0.5, 1.5),
    }
}

/// Cost estimate (USD) with separate input/output pricing.
fn estimate_cost(model: &str, in_tok: u64, out_tok: u64) -> f64 {
    let (i, o) = model_pricing(model);
    (in_tok as f64 * i + out_tok as f64 * o) / 1_000_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_mentions_expand_existing_files_only() {
        let dir = std::env::temp_dir().join(format!("hx-at-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("note.txt");
        std::fs::write(&f, "hello from file").unwrap();
        let msg = format!("look at @{} and @/nope/missing.txt", f.display());
        let out = expand_file_mentions(&msg);
        assert!(out.contains("hello from file"));
        assert!(out.contains("Referenced files"));
        // A message with no mentions is returned unchanged.
        assert_eq!(expand_file_mentions("just text"), "just text");
    }

    #[test]
    fn base64_encodes_rfc_vectors() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn attach_images_finds_existing_files_only() {
        let dir = std::env::temp_dir().join(format!("hx-img-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let img = dir.join("shot.png");
        std::fs::write(&img, [0x89u8, 0x50, 0x4E, 0x47]).unwrap();
        let prompt = format!("what is in {} and also missing.png?", img.display());
        let imgs = attach_images(&prompt);
        assert_eq!(imgs.len(), 1, "only the existing file attaches");
        assert_eq!(imgs[0].media_type, "image/png");
        assert_eq!(imgs[0].base64, base64_encode(&[0x89, 0x50, 0x4E, 0x47]));
    }

    #[test]
    fn tool_summary_extracts_key_arg() {
        use harxes_core_domain::domain::services::tool_protocol::tool_summary;
        assert_eq!(
            tool_summary("Bash", r#"{"command":"cargo test"}"#),
            "Bash cargo test"
        );
        assert_eq!(
            tool_summary("Read", r#"{"path":"src/main.rs"}"#),
            "Read src/main.rs"
        );
        assert!(pretty_result("exit=0 stdout=ok stderr=").contains("✓ exit 0"));
        assert!(pretty_result("exit=1 stdout= stderr=boom").contains("✗ exit 1"));
    }
}
