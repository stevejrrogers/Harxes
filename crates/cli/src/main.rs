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
use harxes_infra_session::JsonSessionStore;
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
    let wiring = match compose::assemble(cli.provider.as_deref(), cli.base_url.as_deref()) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("harxes : {e}");
            std::process::exit(1);
        }
    };

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
    run_repl(&wiring, &cli);
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
    match tokio::runtime::Runtime::new() {
        Ok(rt) => rt.block_on(async {
            match wiring
                .agent
                .run(&pid, &model, "You are Harxes.", prompt, &wiring.limits)
                .await
            {
                Ok(out) => {
                    println!("{}", ui::render_assistant(&out.final_text));
                    if out.truncated_by_guardrail {
                        eprintln!(
                            "{}",
                            ui::C::yellow(&format!(
                                "[stopped by guardrail after {} iterations]",
                                out.iterations
                            ))
                        );
                    }
                    println!(
                        "{} {}",
                        ui::C::dim("[iterations=]"),
                        ui::C::dim(&out.iterations.to_string())
                    );
                }
                Err(e) => eprintln!("harxes:{e}"),
            }
        }),
        Err(e) => {
            eprintln!("harxes : failed to start runtime : {e}");
            std::process::exit(1);
        }
    }
}

fn run_repl(wiring: &compose::Wiring, cli: &Cli) {
    let pid = make_pid(wiring);
    let model = resolve_model(wiring, cli);
    let session_id = match &cli.resume {
        Some(id) => id.clone(),
        None => format!(
            "sess-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0)
        ),
    };
    let mut state = tui::AppState::new(&wiring.provider_id, &model);
    state.active_tools = wiring.active_tools.clone();

    use harxes_core_domain::domain::value_objects::Role;
    // Seed chat pane from any resumed history.
    let transcript = load_history(cli);
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

    state
        .lines
        .push(tui::ChatLine::Tool(format!("session {session_id}")));
    if state.lines.is_empty() {
        state.lines.push(tui::ChatLine::Agent(String::from(
            "✦ Harxes ready. Type below and press Enter.",
        )));
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
            // Slash commands handled here.
            match t.as_str() {
                "/help" => {
                    st.lines.push(tui::ChatLine::Agent(String ::from("/help      this help\n/clear     clear the screen\n/cost      total tokens used\n/model X   switch model\n/compact   summarize context\n/sessions  list saved sessions\n/remember X save a note to agent memory\n/resume I  load saved session by id\n/plan T    break task T into a checklist\n/todo done N   tick item N on the plan\n/export F  write transcript to file F.md\n/cost      tokens + estimated cost\n/exit      quit")));
                    return;
                }
                "/cost" => {
                    let t = st.status.total_tokens;
                    let ti = st.status.total_input_tokens;
                    let to = st.status.total_output_tokens;
                    let model = st.status.model.clone();
                    let cost = estimate_cost(&model, t);
                    st.lines.push(tui::ChatLine::Agent(format!(
                        "tokens: {t} (in {ti} / out {to})\nestimated cost: ${cost:.4}",
                    )));
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
                    for id in &ids {
                        let n = store.load(id).map(|r| r.transcript.len()).unwrap_or(0);
                        st.lines
                            .push(tui::ChatLine::Tool(format!("{id}  ({n} msgs)")));
                    }
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
                            for m in &rec.transcript {
                                if m.role == Role::User {
                                    st.lines.push(tui::ChatLine::User(m.content.clone()));
                                }
                            }
                            st.scroll = 0;
                            st.lines
                                .push(tui::ChatLine::Tool(format!("resumed session {id}")));
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
                        st.set_plan(labels);
                        for (i, t) in st.plan.iter().enumerate() {
                            st.lines.push(tui::ChatLine::Tool(format!(
                                "{} [ ] {}",
                                i + 1,
                                t.label
                            )));
                        }
                    }
                    st.processing = false;
                    return;
                }

                _ if t.starts_with("/todo") => {
                    let arg = t.trim_start_matches("/todo").trim();
                    if let Some(rest) = arg.strip_prefix("done ") {
                        if let Ok(n) = rest.trim().parse::<usize>() {
                            if let Some(item) = st.plan.get_mut(n.saturating_sub(1)) {
                                item.done = true;
                                st.lines
                                    .push(tui::ChatLine::Tool(format!("done: {}", item.label)));
                            } else {
                                st.lines
                                    .push(tui::ChatLine::Agent(String::from("no such item")));
                            }
                        } else {
                            st.lines
                                .push(tui::ChatLine::Agent(String::from("usage: /todo done <n>")));
                        }
                    } else if arg == "list" || arg.is_empty() {
                        if st.plan.is_empty() {
                            st.lines.push(tui::ChatLine::Tool(
                                "(empty plan — use /plan <task>)".into(),
                            ));
                        } else {
                            for (i, t) in st.plan.iter().enumerate() {
                                let m = if t.done { "[✓]" } else { "[ ]" };
                                st.lines.push(tui::ChatLine::Tool(format!(
                                    "{} {} {}",
                                    i + 1,
                                    m,
                                    t.label
                                )));
                            }
                        }
                    } else {
                        st.lines.push(tui::ChatLine::Agent(String::from(
                            "usage: /todo done <n> | /todo list",
                        )));
                    }
                    return;
                }

                _ if t.starts_with("/export ") => {
                    let path = t.trim_start_matches("/export ").trim().to_string();
                    if path.is_empty() {
                        st.lines.push(tui::ChatLine::Agent(String::from(
                            "usage: /export <file.md>",
                        )));
                        return;
                    }
                    let tb = trx.borrow();
                    let mut md = String::from("# Harxes session\n\n");
                    for m in tb.iter() {
                        match m.role {
                            Role::User => md.push_str(&format!("## User\n{}\n\n", m.content)),
                            Role::System => md.push_str(&format!("_system:_ {}\n\n", m.content)),
                            _ => md.push_str(&format!("## Assistant\n{}\n\n", m.content)),
                        }
                        for tc in &m.tool_calls {
                            md.push_str(&format!(
                                "> tool: **{}**\n```json\n{}\n```\n",
                                tc.name, tc.arguments
                            ));
                        }
                    }
                    match std::fs::write(&path, &md) {
                        Ok(_) => st
                            .lines
                            .push(tui::ChatLine::Tool(format!("exported to {path}"))),
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
                            let summary = res.outcome.final_text.clone();
                            *tb = vec![Message::new(
                                Role::System,
                                format!("Previous conversation summary: {summary}"),
                            )];
                            st.lines
                                .push(tui::ChatLine::Tool("context compacted".into()));
                            st.lines
                                .push(tui::ChatLine::Agent(format!("summary: {summary}")));
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
                return;
            }
            st.lines.push(tui::ChatLine::User(msg.clone()));
            st.processing = true;
            // Auto-propose a plan for a new task (only when none is active yet).
            if st.plan.is_empty() && t.split_whitespace().count() > 2 {
                let cur_model2 = model_cell.borrow().clone();
                let pid3 = make_pid(wiring);
                let labels =
                    rt.block_on(generate_plan(wiring.agent.clone(), &pid3, &cur_model2, &t));
                if !labels.is_empty() {
                    st.set_plan(labels);
                    for (i, it) in st.plan.iter().enumerate() {
                        st.lines
                            .push(tui::ChatLine::Tool(format!("{} [ ] {}", i + 1, it.label)));
                    }
                }
            }
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
            handle2.spawn(async move {
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
                )
                .await;
                let _ = txc.send(r);
            });
        },
        |st| {
            // Surface any outstanding approval request for the user to answer.
            st.approval_prompt = wiring.approval_gate.has_pending();
            while let Ok(ev) = rx.try_recv() {
                match ev {
                    Ok((lines, in_tok, out_tok, new_hist)) => {
                        for l in lines {
                            match l {
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
                        st.status.total_cost = estimate_cost(&m, st.status.total_tokens);
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
                    }
                    Err(e) => {
                        st.lines.push(tui::ChatLine::Agent(format!("error: {e}")));
                        st.processing = false;
                    }
                }
            }
        },
    );
    if let Err(e) = result {
        eprintln!("harxes tui error: {e}");
    }
}

fn load_history(cli: &Cli) -> Vec<Message> {
    match &cli.resume {
        Some(id) => {
            let store = JsonSessionStore::new(compose::default_config_dir());
            match store.load(id) {
                Some(rec) => rec.transcript,
                None => {
                    eprintln!("harxes: no saved session '{id}', starting fresh");
                    vec![]
                }
            }
        }
        None => vec![],
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
) -> Result<(Vec<tui::ChatLine>, u64, u64, Vec<Message>), String> {
    let pid = ProviderId::new(&pid_str).unwrap_or_else(|_| ProviderId::new("anthropic").unwrap());
    let store = JsonSessionStore::new(config_dir);
    // Agent started working: ensure its session context directory exists.
    let _ = ctx_store.ensure_session(&session_id);
    // Load persisted session + workspace memory into the transcript.
    let mut messages = history;
    let ctx_block = ctx_store.build_context_block(&session_id);
    if !ctx_block.trim().is_empty() {
        messages.insert(0, Message::new(Role::System, ctx_block));
    }
    match agent
        .continue_chat(&pid, &model, &messages, &user_msg, &limits)
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
            });
            let mut lines = vec![tui::ChatLine::Agent(res.outcome.final_text.clone())];
            // Detailed per-tool blocks: pair each requested call with its result.
            let mut results: std::collections::HashMap<String, String> = Default::default();
            for m in &new_hist {
                if m.role == Role::Tool {
                    if let Some(id) = &m.tool_call_id {
                        results.insert(id.clone(), m.content.clone());
                    }
                }
            }
            for m in &new_hist {
                for tc in &m.tool_calls {
                    let detail = tool_preview(tc.name.as_str(), tc.arguments.as_str());
                    match results.get(&tc.id) {
                        Some(out) => {
                            lines.push(tui::ChatLine::Tool(format!("{detail}\n    → {out}")))
                        }
                        None => lines.push(tui::ChatLine::Tool(detail)),
                    }
                }
            }

            Ok((lines, in_tok, out_tok, new_hist))
        }
        Err(e) => Err(e.to_string()),
    }
}

/// Compact one-line label for a tool invocation, e.g. "Bash: echo hi".
fn tool_preview(name: &str, args: &str) -> String {
    let trimmed = args.trim();
    if trimmed.is_empty() {
        name.to_string()
    } else {
        format!("{name}: {}", &trimmed[..trimmed.len().min(80)])
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

/// Rough per-1k-token pricing estimate (USD) for common models.
fn estimate_cost(model: &str, tokens: u64) -> f64 {
    let per_1k = match model.to_lowercase().as_str() {
        m if m.contains("claude-sonnet") => 0.003,
        m if m.contains("claude-opus") => 0.015,
        m if m.contains("gpt-4o") => 0.0025,
        _ => 0.001,
    };
    (tokens as f64) / 1000.0 * per_1k
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_preview_formats() {
        assert_eq!(tool_preview("Bash", "echo hi"), "Bash: echo hi");
        assert_eq!(tool_preview("Read", ""), "Read");
        // long args truncated
        let long = "x".repeat(200);
        let p = tool_preview("Write", &long);
        assert!(p.len() < 100);
    }
}
