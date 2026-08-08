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
use harxes_core_domain::domain::value_objects::{Message, ProviderId};
use harxes_core_domain::ports::SessionStorePort;
use harxes_infra_session::JsonSessionStore;

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
    let store = JsonSessionStore::new(compose::default_config_dir());
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

    state
        .lines
        .push(tui::ChatLine::Tool(format!("session {session_id}")));
    if state.lines.is_empty() {
        state.lines.push(tui::ChatLine::Agent(String::from(
            "✦ Harxes ready. Type below and press Enter.",
        )));
    }

    let result = tui::run(&mut state, |st, msg| {
        let t = msg.trim().to_string();
        if t.is_empty() {
            return;
        }
        // Slash commands handled here.
        match t.as_str() {
            "/help" => {
                st.lines.push(tui::ChatLine::Agent(String ::from("/help      this help\n/clear     clear the screen\n/cost      total tokens used\n/model X   switch model\n/compact   summarize context\n/exit      quit")));
                return;
            }
            "/cost" => {
                st.lines.push(tui::ChatLine::Agent(format!(
                    "total tokens used: {}",
                    st.status.total_tokens
                )));
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
            _ => {}
        }

        if t.starts_with('/') {
            return;
        }
        st.lines.push(tui::ChatLine::User(msg.clone()));
        st.processing = true;
        let cur_model = model_cell.borrow().clone();
        let mut tb = trx.borrow_mut();
        let out = rt.block_on(run_turn(
            wiring,
            &pid,
            &cur_model,
            &store,
            &session_id,
            &mut tb,
            &t,
        ));
        drop(tb);
        match out {
            Ok((lines, tokens)) => {
                for l in lines {
                    st.lines.push(l);
                }
                st.status.total_tokens += tokens;
            }
            Err(e) => st.lines.push(tui::ChatLine::Agent(format!("error: {e}"))),
        }
        st.processing = false;
    });
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

async fn run_turn(
    wiring: &compose::Wiring,
    pid: &ProviderId,
    model: &str,
    store: &JsonSessionStore,
    session_id: &str,
    transcript: &mut Vec<Message>,
    user_msg: &str,
) -> Result<(Vec<tui::ChatLine>, u64), String> {
    use harxes_core_domain::ports::SessionRecord;
    match wiring
        .agent
        .continue_chat(pid, model, transcript, user_msg, &wiring.limits)
        .await
    {
        Ok(res) => {
            *transcript = res.transcript.clone();
            let _ = store.save(&SessionRecord {
                id: session_id.to_string(),
                created_at: String::new(),
                transcript: res.transcript.clone(),
            });
            let tokens = res.outcome.usage_total_tokens;
            // Surface tool calls performed this turn into the chat.
            let mut tool_names: Vec<String> = Vec::new();
            for m in &res.transcript {
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
            Ok((lines, tokens))
        }
        Err(e) => Err(e.to_string()),
    }
}
