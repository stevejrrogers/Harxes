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
            // Slash commands handled here.
            match t.as_str() {
                "/help" => {
                    st.lines.push(tui::ChatLine::Agent(String ::from("/help      this help\n/clear     clear the screen\n/cost      total tokens used\n/model X   switch model\n/compact   summarize context\n/sessions  list saved sessions\n/remember X save a note to agent memory\n/cost      tokens + estimated cost\n/exit      quit")));
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
                        match ctx_store.remember_session(&session_id, &note) {
                            Ok(_) => {
                                let _ =
                                    ctx_store.remember_workspace(&format!("[{session_id}] {note}"));
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
            let sid = session_id.clone();
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
            let mut tool_names: Vec<String> = Vec::new();
            for m in &new_hist {
                for tc in &m.tool_calls {
                    if !tool_names.contains(&tc.name) {
                        tool_names.push(tc.name.clone());
                    }
                }
            }
            let mut lines = vec![tui::ChatLine::Agent(res.outcome.final_text.clone())];
            for tn in &tool_names {
                lines.push(tui::ChatLine::Tool(format!("ran {tn}")));
            }
            Ok((lines, in_tok, out_tok, new_hist))
        }
        Err(e) => Err(e.to_string()),
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
