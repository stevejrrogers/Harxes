//! Full-screen TUI (ratatui) - opencode-style layout.
//! Title bar, scrollable chat pane with avatars and syntax highlighting,
//! a right-side status panel, and a separate input box at the bottom.

use crossterm::event::{self, Event, KeyCode, KeyModifiers};
use ratatui::{
    prelude::*,
    widgets::{Block, Borders, Paragraph},
};
use std::sync::OnceLock;
use std::time::Duration;
use syntect::easy::HighlightLines;
use syntect::highlighting::{Color as SynColor, FontStyle as SynFont};
use syntect::highlighting::{Theme, ThemeSet};
use syntect::parsing::{SyntaxReference, SyntaxSet};

/// Light/dark theme controlling pane backgrounds.
#[derive(Debug, Clone, Copy)]
pub struct PaneTheme {
    pub bg_status: Color,
    pub bg_input: Color,
}

impl PaneTheme {
    pub fn dark() -> Self {
        Self {
            bg_status: Color::Indexed(234),
            bg_input: Color::Indexed(236),
        }
    }
    pub fn light() -> Self {
        Self {
            bg_status: Color::White,
            bg_input: Color::Indexed(254),
        }
    }
}

/// A checklist item the agent breaks its task down into.
#[derive(Debug, Clone)]
pub struct TaskItem {
    pub label: String,
    pub done: bool,
}

impl TaskItem {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            done: false,
        }
    }
}

#[derive(Clone)]
pub enum ChatLine {
    User(String),
    Agent(String),
    Tool(String),
}

pub struct StatusInfo {
    pub provider: String,
    pub model: String,
    pub total_tokens: u64,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub tools_used: Vec<String>,
    pub total_cost: f64,
    /// Per-model usage: model id → (input tokens, output tokens). A session
    /// can span several models via /model, so cost is reported per model.
    pub per_model: std::collections::HashMap<String, (u64, u64)>,
    pub cwd: String,
    pub git_branch: Option<String>,
}

/// Short label for the current working directory (last 2 segments).
fn current_dir_label() -> String {
    let cwd = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_default();
    let home = std::env::var("HOME").unwrap_or_default();
    let shown = if !home.is_empty() && cwd.starts_with(&home) {
        format!("~{}", &cwd[home.len()..])
    } else {
        cwd
    };
    shown
}

/// Detect the current git branch by walking up from the CWD looking for
/// `.git/HEAD` (no subprocess).
fn current_git_branch() -> Option<String> {
    use std::fs;
    let mut dir = std::env::current_dir().ok()?;
    loop {
        let head = fs::read_to_string(dir.join(".git").join("HEAD"));
        if let Ok(content) = head {
            if let Some(branch) = content.strip_prefix("ref: refs/heads/") {
                return Some(branch.trim().to_string());
            }
        }
        if !dir.pop() {
            return None;
        }
    }
}

pub struct AppState {
    pub lines: Vec<ChatLine>,
    pub input: String,
    /// Byte offset of the edit cursor inside [`Self::input`].
    pub cursor_ix: usize,
    pub processing: bool,
    pub scroll: u16,
    pub status: StatusInfo,
    pub history: Vec<String>,
    pub hist_pos: Option<usize>,
    pub auto_scroll: bool,
    pub spinner: u32,
    pub typing_text: String,
    pub typing_shown: usize,
    /// Slash command autocomplete menu state.
    pub completions: Vec<String>,
    pub completion_sel: usize,
    /// Live view of tools currently executing (shared with the background task).
    pub active_tools: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    /// Outstanding approval request awaiting a y/n answer (if any).
    pub approval_prompt: Option<String>,
    /// Agent-proposed plan broken into subtasks with checkboxes.
    pub plan: Vec<TaskItem>,
    /// Live view of the agent-managed todo store (shared with the agent loop);
    /// when non-empty it supersedes `plan` in the tasks panel.
    pub todos: std::sync::Arc<
        std::sync::Mutex<Vec<harxes_core_domain::domain::value_objects::TodoItem>>,
    >,
    /// Pane background theme (dark or light).
    pub theme: PaneTheme,
    /// True while the current turn's text is arriving via live streaming.
    pub streaming_turn: bool,
    /// Accumulated streamed text of the current turn (authoritative final text).
    pub live_text: String,
    /// When the in-flight turn started (drives the elapsed-time display).
    pub turn_started: Option<std::time::Instant>,
}

impl AppState {
    pub fn new(provider: &str, model: &str) -> Self {
        Self {
            lines: vec![],
            input: String::new(),
            cursor_ix: 0,
            processing: false,
            scroll: 0,
            status: StatusInfo {
                provider: provider.to_string(),
                model: model.to_string(),
                total_tokens: 0,
                total_input_tokens: 0,
                total_output_tokens: 0,
                tools_used: vec![],
                total_cost: 0.0,
                per_model: Default::default(),
                cwd: current_dir_label(),
                git_branch: current_git_branch(),
            },
            history: vec![],
            hist_pos: None,
            auto_scroll: true,
            spinner: 0,
            typing_text: String::new(),
            typing_shown: 0,
            completions: vec![],
            completion_sel: 0,
            active_tools: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            approval_prompt: None,
            plan: vec![],
            todos: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            theme: PaneTheme::dark(),
            streaming_turn: false,
            live_text: String::new(),
            turn_started: None,
        }
    }

    /// Replace the whole plan with parsed subtasks (labels only, undone).
    pub fn set_plan(&mut self, labels: Vec<String>) {
        self.plan = labels
            .into_iter()
            .filter(|l| !l.trim().is_empty())
            .map(TaskItem::new)
            .collect();
    }

    /// Mark the first undone item whose label contains `needle` as done.
    #[allow(dead_code)]
    pub fn mark_done(&mut self, needle: &str) -> bool {
        let n = needle.trim().to_lowercase();
        if let Some(t) = self
            .plan
            .iter_mut()
            .find(|t| !t.done && t.label.to_lowercase().contains(&n))
        {
            t.done = true;
            true
        } else {
            false
        }
    }

    /// Auto-tick undone items whose key words all appear in `text`.
    pub fn plan_auto_tick(&mut self, text: &str) {
        let t = text.to_lowercase();
        const STOP: [&str; 12] = [
            "the", "a", "an", "to", "of", "and", "in", "for", "on", "with", "this", "that",
        ];
        for item in self.plan.iter_mut() {
            if item.done {
                continue;
            }
            let words: Vec<String> = item
                .label
                .to_lowercase()
                .split_whitespace()
                .map(|x| x.trim_matches([',', '(', ')']).to_string())
                .collect();
            let sig: Vec<&String> = words
                .iter()
                .filter(|x| !STOP.contains(&x.as_str()) && x.len() > 2)
                .collect();
            if sig.is_empty() {
                continue;
            }
            if sig.iter().all(|word| t.contains(word.as_str())) {
                item.done = true;
            }
        }
    }
}

/// Split a plain-text line into styled spans, honoring markdown bold
/// (**text**) so agent output reads like a rendered Copilot chat.
fn markdown_spans(line: &str) -> Vec<Span<'static>> {
    let mut spans = Vec::new();
    let mut rest = line;
    loop {
        // Heading: up to three # prefix -> bold colored line.
        let trimmed_head = rest.trim_start();
        let level = trimmed_head.chars().take_while(|&c| c == '#').count();
        if (1..=3).contains(&level)
            && trimmed_head
                .as_bytes()
                .get(level)
                .is_some_and(|b| b.is_ascii_whitespace())
        {
            let body = trimmed_head[level..].trim_start();
            let color = if level == 1 {
                Color::Magenta
            } else if level == 2 {
                Color::Cyan
            } else {
                Color::Blue
            };
            spans.push(Span::styled(
                body.to_string(),
                Style::new().fg(color).bold(),
            ));
            return spans;
        }
        // Bold: **text**
        match rest.find("**") {
            Some(start) => {
                if start > 0 {
                    spans.push(Span::raw(rest[..start].to_string()));
                }
                let after = &rest[start + 2..];
                match after.find("**") {
                    Some(end) => {
                        spans.push(Span::styled(after[..end].to_string(), Style::new().bold()));
                        rest = &after[end + 2..];
                    }
                    None => {
                        spans.push(Span::raw(rest.to_string()));
                        return spans;
                    }
                }
            }
            None => {
                spans.push(Span::raw(rest.to_string()));
                break;
            }
        }
    }
    spans
}

fn agent_lines(text: &str) -> Vec<Line<'static>> {
    if !text.contains("```") {
        // No code fences: render each line with inline markdown styling.
        let mut out = Vec::new();
        for ln in text.split('\n') {
            out.push(Line::from(markdown_spans(ln)));
        }
        return out;
    }
    let mut out = Vec::new();
    let mut rest = text;
    loop {
        match rest.find("```") {
            None => {
                for ln in rest.split('\n') {
                    out.push(Line::from(markdown_spans(ln)));
                }
                break;
            }
            Some(p) => {
                if p > 0 {
                    let seg = &rest[..p];
                    for ln in seg.split('\n') {
                        out.push(Line::from(markdown_spans(ln)));
                    }
                }
                rest = &rest[p + 3..];
                if let Some(e) = rest.find("```") {
                    let mut body = &rest[..e];
                    let mut lang_hint = String::new();
                    if let Some(nl) = body.find('\n') {
                        lang_hint = body[..nl].trim().to_string();
                        {
                            let label = if lang_hint.is_empty() {
                                "code".to_string()
                            } else {
                                lang_hint.clone()
                            };
                            out.push(Line::from(vec![
                                Span::styled("  ╭─ ", Style::new().fg(Color::DarkGray)),
                                Span::styled(label, Style::new().fg(Color::Yellow)),
                            ]));
                        }
                        body = &body[nl + 1..];
                    }
                    for l in code_block_lines(&lang_hint, body) {
                        let mut spans =
                            vec![Span::styled("  │ ", Style::new().fg(Color::DarkGray))];
                        spans.extend(l.spans);
                        out.push(Line::from(spans));
                    }
                    out.push(Line::from(vec![Span::styled(
                        "  ╰─",
                        Style::new().fg(Color::DarkGray),
                    )]));
                    rest = &rest[e + 3..];
                } else {
                    break;
                }
            }
        }
    }
    out
}

fn est_lines(state: &AppState, width: usize) -> usize {
    let w = width.max(10);
    let mut n = 0usize;
    for l in &state.lines {
        match l {
            ChatLine::User(t) => n += 1 + t.len() / w,
            ChatLine::Agent(t) => n += 2 + t.len() / w,
            ChatLine::Tool(_) => n += 1,
        }
    }
    n
}

fn chat_pane(frame: &mut Frame, state: &mut AppState, area: Rect) {
    if state.auto_scroll {
        let total = est_lines(state, area.width as usize);
        state.scroll = (total.saturating_sub(area.height as usize)) as u16;
    }
    let mut text = ratatui::text::Text::default();
    for line in &state.lines {
        match line {
            ChatLine::User(t) => {
                text.push_line(Line::from(vec![
                    Span::styled("  YOU", Style::new().fg(Color::Green).bold()),
                    Span::styled(" ──", Style::new().fg(Color::Green)),
                ]));
                for u in t.split('\n') {
                    text.push_line(Line::from(vec![Span::styled(
                        format!("  {}", u),
                        Style::new().fg(Color::Rgb(196, 210, 230)),
                    )]));
                }
                text.push_line(Line::raw(""));
            }
            ChatLine::Agent(t) => {
                text.push_line(Line::from(vec![
                    Span::styled("  HARXES", Style::new().fg(Color::Magenta).bold()),
                    Span::styled(" ✦", Style::new().fg(Color::Magenta)),
                ]));
                for l in agent_lines(t) {
                    text.push_line(l);
                }
                text.push_line(Line::raw(""));
            }
            ChatLine::Tool(t) => {
                text.push_line(Line::from(vec![
                    Span::styled("  ⏺ ", Style::new().fg(Color::Cyan)),
                    Span::styled(t.to_string(), Style::new().fg(Color::Gray)),
                ]));
            }
        }
    }
    // typewriter reveal OUTSIDE the per-line loop, once at the end.
    if !state.typing_text.is_empty() {
        let shown = &state.typing_text[..state.typing_shown.min(state.typing_text.len())];
        text.push_line(Line::from(vec![Span::styled(
            "  HARXES ✦",
            Style::new().fg(Color::Magenta).bold(),
        )]));
        text.push_line(Line::from(vec![
            Span::raw(shown.to_string()),
            if state.spinner.is_multiple_of(2) {
                Span::styled("▋", Style::new().fg(Color::Cyan))
            } else {
                Span::raw(" ")
            },
        ]));
    }

    let para = Paragraph::new(text)
        .style(Style::default())
        .wrap(ratatui::widgets::Wrap { trim: true })
        .scroll((state.scroll, 0));
    frame.render_widget(para, area);
}

/// Human-friendly token count: 950, 12.3k, 1.2M.
fn fmt_tokens(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}k", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

/// Truncate from the LEFT with a leading ellipsis so the tail (most specific
/// part of a path) stays visible.
fn truncate_left(s: &str, max: usize) -> String {
    let n = s.chars().count();
    if n <= max {
        return s.to_string();
    }
    let tail: String = s.chars().skip(n - max.saturating_sub(1)).collect();
    format!("…{tail}")
}

fn status_panel(frame: &mut Frame, state: &AppState, area: Rect) {
    let mut text = ratatui::text::Text::default();

    // Header: state dot + elapsed time while working.
    if state.processing {
        const SPIN: [char; 4] = ['⠋', '⠙', '⠸', '⠴'];
        let secs = state
            .turn_started
            .map(|t| t.elapsed().as_secs())
            .unwrap_or(0);
        text.push_line(Line::from(vec![Span::styled(
            format!(
                "  {} WORKING · {}s",
                SPIN[state.spinner as usize % SPIN.len()],
                secs
            ),
            Style::new().fg(Color::Yellow).bold(),
        )]));
    } else {
        text.push_line(Line::from(vec![Span::styled(
            "  ● READY",
            Style::new().fg(Color::Green).bold(),
        )]));
    }
    text.push_line(Line::from(vec![Span::styled(
        format!("   {}", state.status.model),
        Style::new().fg(Color::Blue).bold(),
    )]));

    text.push_line(Line::from(vec![Span::styled(
        format!("  {}", "─".repeat(26)),
        Style::new().fg(Color::DarkGray),
    )]));

    // cwd + git branch. Truncate the path from the left so the panel never
    // clips it mid-word without warning.
    let dir_room = (area.width as usize).saturating_sub(13).max(8);
    text.push_line(Line::from(vec![
        Span::styled("  dir      ", Style::new().fg(Color::DarkGray)),
        Span::raw(truncate_left(&state.status.cwd, dir_room)),
    ]));
    if let Some(b) = &state.status.git_branch {
        text.push_line(Line::from(vec![
            Span::styled("  branch   ", Style::new().fg(Color::DarkGray)),
            Span::styled(format!("⎋ {b}"), Style::new().fg(Color::Magenta)),
        ]));
    }

    // Provider + model rows (model moved to header).
    text.push_line(Line::from(vec![
        Span::styled("  provider ", Style::new().fg(Color::DarkGray)),
        Span::styled(state.status.provider.clone(), Style::new().fg(Color::Cyan)),
    ]));

    // Token usage on one compact line.
    text.push_line(Line::from(vec![
        Span::styled("  tokens   ", Style::new().fg(Color::DarkGray)),
        Span::styled(
            fmt_tokens(state.status.total_tokens),
            Style::new().fg(Color::Yellow).bold(),
        ),
        Span::styled(
            format!(
                " (↑{} ↓{})",
                fmt_tokens(state.status.total_input_tokens),
                fmt_tokens(state.status.total_output_tokens)
            ),
            Style::new().fg(Color::DarkGray),
        ),
    ]));
    text.push_line(Line::from(vec![
        Span::styled("  cost     ", Style::new().fg(Color::DarkGray)),
        Span::styled(
            format!("${:.4}", state.status.total_cost),
            Style::new().fg(Color::Green).bold(),
        ),
    ]));

    frame.render_widget(
        Paragraph::new(text).style(Style::default().bg(state.theme.bg_status).fg(Color::White)),
        area,
    );
}

fn tasks_panel(frame: &mut Frame, state: &AppState, area: Rect) {
    use harxes_core_domain::domain::value_objects::TodoStatus;
    // The agent-managed store is the live source of truth; the legacy
    // `state.plan` list is only a fallback for pre-todo flows.
    let todos = state.todos.lock().unwrap().clone();
    let has_plan = !todos.is_empty() || !state.plan.is_empty();
    if !state.processing
        && state.status.tools_used.is_empty()
        && state.approval_prompt.is_none()
        && !has_plan
    {
        return;
    }
    let mut text = ratatui::text::Text::default();
    let done = todos.iter().filter(|t| t.done()).count();
    let header = if todos.is_empty() {
        "  PLAN".to_string()
    } else {
        format!("  PLAN {done}/{}", todos.len())
    };
    text.push_line(Line::from(vec![Span::styled(
        header,
        Style::new().fg(Color::Cyan).bold(),
    )]));
    if !todos.is_empty() {
        const SPIN: [char; 4] = ['⠋', '⠙', '⠸', '⠴'];
        if done == todos.len() {
            text.push_line(Line::from(vec![Span::styled(
                format!("  ✓ all {done} steps done"),
                Style::new().fg(Color::Green),
            )]));
        } else {
            for t in &todos {
                let (mark, col, label) = match t.status {
                    TodoStatus::Completed => {
                        ("✓".to_string(), Color::DarkGray, t.label.clone())
                    }
                    TodoStatus::InProgress => (
                        SPIN[state.spinner as usize % SPIN.len()].to_string(),
                        Color::Cyan,
                        t.active_label().to_string(),
                    ),
                    TodoStatus::Pending => ("○".to_string(), Color::White, t.label.clone()),
                };
                let style = if t.status == TodoStatus::Completed {
                    Style::new().fg(col).add_modifier(Modifier::CROSSED_OUT)
                } else if t.status == TodoStatus::InProgress {
                    Style::new().fg(col).bold()
                } else {
                    Style::new().fg(col)
                };
                text.push_line(Line::from(vec![
                    Span::styled(format!("  {mark} "), Style::new().fg(col)),
                    Span::styled(label, style),
                ]));
            }
        }
    } else if !state.plan.is_empty() {
        for t in &state.plan {
            let box_mark = if t.done { "[✓]" } else { "[ ]" };
            let col = if t.done { Color::Green } else { Color::White };
            text.push_line(Line::from(vec![
                Span::styled(format!("  {box_mark} "), Style::new().fg(col)),
                Span::styled(t.label.clone(), Style::new().fg(col)),
            ]));
        }
    } else {
        text.push_line(Line::from(vec![Span::styled(
            "  (no plan yet)",
            Style::new().fg(Color::DarkGray),
        )]));
    }
    if let Some(p) = &state.approval_prompt {
        text.push_line(Line::raw(""));
        text.push_line(Line::from(vec![Span::styled(
            "  ⚠ allow? y/n",
            Style::new().fg(Color::Red).add_modifier(Modifier::BOLD),
        )]));
        for ln in p.split('\n') {
            // Diff-aware coloring so golden-diff approvals read like a diff.
            let style = if ln.starts_with('+') {
                Style::new().fg(Color::Green)
            } else if ln.starts_with('-') {
                Style::new().fg(Color::Red)
            } else {
                Style::new()
            };
            text.push_line(Line::from(vec![
                Span::styled("    ", Style::new().fg(Color::DarkGray)),
                Span::styled(ln.to_string(), style),
            ]));
        }
    }
    let live = {
        match state.active_tools.lock() {
            Ok(g) => g.clone(),
            Err(_) => vec![],
        }
    };
    const LIVE_SPIN: [char; 4] = ['⠋', '⠙', '⠸', '⠴'];
    let live_dot = LIVE_SPIN[state.spinner as usize % LIVE_SPIN.len()];
    if !live.is_empty() {
        for t in live.iter().take(4) {
            text.push_line(Line::from(vec![
                Span::styled(format!("  {live_dot} "), Style::new().fg(Color::Yellow)),
                Span::raw(t.to_string()),
            ]));
        }
    } else if state.processing {
        text.push_line(Line::from(vec![Span::styled(
            format!("  {live_dot} thinking…"),
            Style::new().fg(Color::Yellow),
        )]));
    }
    frame.render_widget(
        Paragraph::new(text).style(Style::default().bg(state.theme.bg_status).fg(Color::White)),
        area,
    );
}

fn input_pane(frame: &mut Frame, state: &AppState, area: Rect) {
    let prompt_style = Style::new().fg(Color::Green).bold();
    let cur = if state.spinner % 5 < 3 {
        Span::styled("▋", Style::new().fg(Color::Cyan))
    } else {
        Span::raw(" ")
    };
    let mut lines: Vec<Line> = Vec::new();
    if state.input.is_empty() && !state.processing {
        lines.push(Line::from(vec![
            Span::styled(
                "  Ask anything...  (/help for commands)",
                Style::new().fg(Color::DarkGray),
            ),
            cur,
        ]));
    } else {
        // Split into lines with their starting byte offsets.
        let mut starts: Vec<usize> = vec![0];
        for (i, b) in state.input.bytes().enumerate() {
            if b == b'\n' {
                starts.push(i + 1);
            }
        }
        // Find which line the cursor is on. Floor the index onto a character
        // boundary so slicing below never panics even if a code path ever left
        // the cursor mid-multibyte-character.
        let raw_cursor = state.cursor_ix.min(state.input.len());
        let cursor = state.input.floor_char_boundary(raw_cursor);
        let line_idx = starts.partition_point(|&s| s <= cursor).saturating_sub(1);
        let line_start = starts[line_idx];
        // Column in chars from line start to cursor.
        let col_chars = state.input[line_start..cursor].chars().count();
        for (i, part) in state.input.split('\n').enumerate() {
            let mut spans = vec![Span::styled("❯ ", prompt_style)];
            if i == line_idx {
                let nchars = part.chars().count();
                let col = col_chars.min(nchars);
                let before: String = part.chars().take(col).collect();
                let after: String = part.chars().skip(col).collect();
                spans.push(Span::raw(before));
                spans.push(cur.clone());
                if !after.is_empty() {
                    spans.push(Span::raw(after));
                }
            } else {
                spans.push(Span::raw(part.to_string()));
            }
            lines.push(Line::from(spans));
        }
    }
    // Copilot-style keybinding hint bar.
    if area.height >= 3 {
        lines.push(Line::raw(""));
        lines.push(Line::from(vec![Span::styled(
            "  Enter send · Alt+Enter newline · ←→ move · Up/Down history · Esc quit",
            Style::new().fg(Color::DarkGray),
        )]));
    }

    frame.render_widget(
        Paragraph::new(lines).style(Style::default().bg(state.theme.bg_input).fg(Color::White)),
        area,
    );
}

pub fn draw(frame: &mut Frame, state: &mut AppState) {
    // Track when the current turn began so the header can show elapsed time.
    if state.processing && state.turn_started.is_none() {
        state.turn_started = Some(std::time::Instant::now());
    } else if !state.processing {
        state.turn_started = None;
    }
    let area = frame.area();
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(68), Constraint::Percentage(32)])
        .split(area);
    let nlines = (state.input.matches('\n').count() as u16) + 1;
    let in_h = nlines.saturating_add(2).min(10);
    let left = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(in_h)])
        .split(cols[0]);
    // Right column: status sized to its content at top; tasks fill the rest.
    let r = cols[1];
    let sh = r.height.min(9);
    let status_area = Rect::new(r.x, r.y, r.width, sh);
    let ty = r.y.saturating_add(sh);
    let th = r.height.saturating_sub(sh);
    let tasks_area = Rect::new(r.x, ty, r.width, th);
    chat_pane(frame, state, left[0]);
    input_pane(frame, state, left[1]);
    status_panel(frame, state, status_area);
    tasks_panel(frame, state, tasks_area);
    // Float the menu just ABOVE the input pane so it never covers what the
    // user is typing or the hint bar.
    let menu_h = (state.completions.len().min(5) as u16) + 2;
    completion_menu(
        frame,
        &state.completions,
        state.completion_sel,
        left[1].y.saturating_sub(menu_h),
    );
}

pub fn run(
    state: &mut AppState,
    mut on_command: impl FnMut(&mut AppState, String),
    mut poll_events: impl FnMut(&mut AppState),
    mut cancel_turn: impl FnMut(),
) -> Result<(), Box<dyn std::error::Error>> {
    let mut terminal = match ratatui::try_init() {
        Ok(t) => t,
        Err(e) => return Err(format!("cannot init terminal: {e}").into()),
    };
    loop {
        // Advance the frame clock every loop tick so UI animations (blinking
        // cursors) run even when idle.
        state.spinner = state.spinner.wrapping_add(1);
        // Advance the typewriter reveal by a few chars per frame.
        if !state.typing_text.is_empty() && state.typing_shown < state.typing_text.len() {
            state.typing_shown = (state.typing_shown + 3).min(state.typing_text.len());
        } else if !state.typing_text.is_empty() && state.typing_shown >= state.typing_text.len() {
            // done revealing: finalize into a real Agent line.
            let done = std::mem::take(&mut state.typing_text);
            state.lines.push(ChatLine::Agent(done));
            state.auto_scroll = true;
        }
        poll_events(state);
        terminal.draw(|f| draw(f, state))?;
        if event::poll(Duration::from_millis(80))? {
            if let Event::Key(k) = event::read()? {
                if k.modifiers
                    .contains(crossterm::event::KeyModifiers::CONTROL)
                    && k.code == KeyCode::Char('c')
                {
                    if state.processing {
                        state.processing = false;
                        cancel_turn();
                        state.typing_text.clear();
                        state.typing_shown = 0;
                        state.streaming_turn = false;
                        state.live_text.clear();
                        state.scroll = 0;
                        state.auto_scroll = true;
                        state
                            .lines
                            .push(ChatLine::Tool(String::from("⛔ cancelled")));
                    } else {
                        break;
                    }
                    continue;
                }
                match k.code {
                    KeyCode::Char(c) => {
                        let ix = state.cursor_ix.min(state.input.len());
                        if !state.input.is_char_boundary(ix) {
                            continue;
                        }
                        state.input.insert(ix, c);
                        state.cursor_ix = ix + c.len_utf8();
                        update_completions(state);
                    }
                    KeyCode::Backspace => {
                        let cur = state.cursor_ix.min(state.input.len());
                        if cur > 0 && !state.input.is_empty() {
                            let sidx = state.input.floor_char_boundary(cur - 1);
                            if sidx < cur {
                                state.input.replace_range(sidx..cur, "");
                                state.cursor_ix = sidx;
                            }
                            update_completions(state);
                        }
                    }
                    KeyCode::Left => {
                        if !state.completions.is_empty() || state.hist_pos.is_some() {
                            continue;
                        }
                        let cur = state.cursor_ix.min(state.input.len());
                        if cur > 0 && !state.input.is_empty() {
                            state.cursor_ix = state.input.floor_char_boundary(cur - 1);
                        }
                    }
                    KeyCode::Right => {
                        if !state.completions.is_empty() || state.hist_pos.is_some() {
                            continue;
                        }
                        let raw = state.cursor_ix.min(state.input.len());
                        // Floor onto a boundary so the slice below never panics.
                        let cur = state.input.floor_char_boundary(raw);
                        if cur < state.input.len() {
                            if let Some(ch) = state.input[cur..].chars().next() {
                                state.cursor_ix = cur + ch.len_utf8();
                            }
                        }
                    }

                    KeyCode::Enter => {
                        if !state.completions.is_empty() {
                            if let Some(c) = state.completions.get(state.completion_sel) {
                                state.input = c.clone();
                                state.cursor_ix = state.input.len();
                            }
                            state.completions.clear();
                            continue;
                        }
                        let wants_newline = k.modifiers.contains(KeyModifiers::ALT)
                            || k.modifiers.contains(KeyModifiers::SHIFT);
                        // Multi-line composing: Alt or Shift + Enter inserts a new
                        // line; plain Enter always submits.
                        if wants_newline {
                            if !state.input.ends_with('\n') {
                                state.input.push('\n');
                            }
                            continue;
                        }
                        let msg = std::mem::take(&mut state.input);
                        let t = msg.trim();
                        if t == "/exit" {
                            break;
                        }
                        if t == "/clear" {
                            state.lines.clear();
                            state.scroll = 0;
                            continue;
                        }
                        if t.starts_with('/') {
                            on_command(state, msg);
                            state.scroll = 0;
                            continue;
                        }
                        state.auto_scroll = true;
                        if !t.is_empty() {
                            if state.history.last().map(String::as_str) != Some(t) {
                                state.history.push(t.to_string());
                            }
                            state.hist_pos = None;
                            on_command(state, msg);
                        }
                        state.scroll = 0;
                    }
                    // Up/Down: navigate completions when open, else input history.
                    KeyCode::Up => {
                        if !state.completions.is_empty() {
                            let len = state.completions.len();
                            state.completion_sel = if state.completion_sel == 0 {
                                len - 1
                            } else {
                                state.completion_sel - 1
                            };
                            continue;
                        }
                        let n = state.history.len();
                        if n == 0 {
                            continue;
                        }
                        let pos = match state.hist_pos {
                            Some(p) => p.saturating_add(1).min(n - 1),
                            None => 0,
                        };
                        if pos < n {
                            state.input = state.history[n - 1 - pos].clone();
                            state.cursor_ix = state.input.len();
                        }
                        state.hist_pos = Some(pos);
                    }
                    KeyCode::Down => {
                        if !state.completions.is_empty() {
                            let len = state.completions.len();
                            state.completion_sel = (state.completion_sel + 1) % len;
                            continue;
                        }
                        if let Some(p) = state.hist_pos {
                            if p == 0 {
                                state.input.clear();
                                state.hist_pos = None;
                            } else {
                                let np = p - 1;
                                let n = state.history.len();
                                if np < n {
                                    state.input = state.history[n - 1 - np].clone();
                                    state.cursor_ix = state.input.len();
                                };
                                state.hist_pos = Some(np);
                            }
                        }
                    }
                    // Chat scroll
                    KeyCode::PageDown => {
                        state.auto_scroll = false;
                        state.scroll = state.scroll.saturating_sub(5);
                    }
                    KeyCode::PageUp => {
                        state.auto_scroll = false;
                        state.scroll += 5;
                    }
                    KeyCode::Esc => {
                        if !state.completions.is_empty() {
                            state.completions.clear();
                        } else if state.input.contains('\n') {
                            // Exit multi-line compose: drop the trailing line.
                            if let Some(pos) = state.input.rfind('\n') {
                                state.input.truncate(pos);
                                state.cursor_ix = state.cursor_ix.min(state.input.len());
                            }
                        } else if !state.input.is_empty() || state.processing {
                            // Cancel current editing; don't quit while there is
                            // typed text or an active task (guards against a
                            // stray Esc breaking out of the app).
                            state.input.clear();
                            state.cursor_ix = 0;
                        } else {
                            break;
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    ratatui::restore();
    Ok(())
}

fn syn_ctx() -> (&'static SyntaxSet, &'static syntect::highlighting::Theme) {
    static SS: OnceLock<SyntaxSet> = OnceLock::new();
    static TH: OnceLock<syntect::highlighting::Theme> = OnceLock::new();
    let ss = SS.get_or_init(SyntaxSet::load_defaults_newlines);
    let th = TH.get_or_init(|| {
        let ts = ThemeSet::load_defaults();
        ts.themes["base16-ocean.dark"].clone()
    });
    (ss, th)
}

fn syn_color(c: SynColor) -> Color {
    Color::Rgb(c.r, c.g, c.b)
}

/// Highlight one code line using syntect, returning ratatui Spans.
fn hl_spans(
    ss: &SyntaxSet,
    theme: &Theme,
    syntax: &SyntaxReference,
    line: &str,
) -> Vec<Span<'static>> {
    let mut hl = HighlightLines::new(syntax, theme);
    let mut out = Vec::new();
    if let Ok(ranges) = hl.highlight_line(line, ss) {
        for (st, text) in ranges {
            let mut style = Style::default();
            style = style.fg(syn_color(st.foreground));
            if st.font_style.contains(SynFont::BOLD) {
                style = style.add_modifier(Modifier::BOLD);
            }
            if st.font_style.contains(SynFont::ITALIC) {
                style = style.add_modifier(Modifier::ITALIC);
            }
            out.push(Span::styled(text.to_string(), style));
        }
    }
    out
}

/// Render a fenced code block with real syntax highlighting.
fn code_block_lines(lang: &str, code: &str) -> Vec<Line<'static>> {
    let (ss, theme) = syn_ctx();
    let syntax = ss
        .find_syntax_by_token(lang)
        .or_else(|| ss.find_syntax_by_name(lang))
        .unwrap_or_else(|| ss.find_syntax_plain_text());
    let mut out = Vec::new();
    for codeline in code.split('\n') {
        out.push(Line::from(hl_spans(ss, theme, syntax, codeline)));
    }
    out
}

pub const SLASH_COMMANDS: [&str; 16] = [
    "/help",
    "/clear",
    "/cost",
    "/model ",
    "/compact",
    "/undo",
    "/sessions",
    "/resume ",
    "/remember ",
    "/plan ",
    "/todo",
    "/init",
    "/mcp",
    "/theme",
    "/export ",
    "/exit",
];

/// Recompute the autocomplete list based on the current input.
pub fn update_completions(state: &mut AppState) {
    if !state.input.starts_with("/") || state.input.contains(' ') {
        state.completions.clear();
        state.completion_sel = 0;
        return;
    }
    let prefix = &state.input;
    let mut matches: Vec<String> = SLASH_COMMANDS
        .iter()
        .filter(|c| c.starts_with(prefix))
        .map(|c| c.to_string())
        .collect();
    if matches.is_empty() {
        matches.push(prefix.to_string());
    }
    if state.completion_sel >= matches.len() {
        state.completion_sel = 0;
    }
    state.completions = matches;
}

fn completion_menu(frame: &mut Frame, completions: &[String], selected: usize, anchor_y: u16) {
    if completions.is_empty() {
        return;
    }
    let shown = completions.len().min(5);
    let w = 24u16;
    let mut h = (shown as u16) + 2;
    let x = 1u16;
    // Keep the menu within the terminal height.
    let max_h = frame
        .area()
        .height
        .saturating_sub(anchor_y)
        .saturating_sub(1)
        .max(3);
    if h > max_h {
        h = max_h;
    }
    let y = anchor_y.min(frame.area().height.saturating_sub(h));
    let popup = Rect::new(x, y, w, h);
    let mut text = ratatui::text::Text::default();
    for (i, c) in completions.iter().take(shown).enumerate() {
        let sel = i == selected % completions.len();
        let st = if sel {
            Style::new().fg(Color::Black).bg(Color::Cyan)
        } else {
            Style::new().fg(Color::White)
        };
        text.push_line(Line::from(vec![Span::styled(c.clone(), st)]));
    }
    frame.render_widget(
        Paragraph::new(text)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::new().fg(Color::Cyan)),
            )
            .bg(Color::Indexed(236)),
        popup,
    );
}

#[cfg(test)]
mod tests {

    #[test]
    fn pane_theme_dark_and_light_differ() {
        let d = PaneTheme::dark();
        let l = PaneTheme::light();
        assert_ne!(d.bg_status, l.bg_status);
        assert_ne!(d.bg_input, l.bg_input);
    }
    use super::*;
    use ratatui::{backend::TestBackend, Terminal};
    #[test]
    fn cursor_renders_at_position_in_input_pane() {
        use ratatui::{backend::TestBackend, Terminal};
        let mut st = AppState::new("litellm", "DeepSeek-V4-Flash");
        st.input = "hello world".to_string();
        // Place the cursor after 'hello' (index 5).
        st.cursor_ix = 5;
        st.spinner = 0; // ensure blinking block is shown
                        // Wide so the status column doesn't crowd out the input line.
        let backend = TestBackend::new(80, 10);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|f| draw(f, &mut st)).unwrap();
        let buf = terminal.backend().buffer();
        // Concatenate every row so we can find where the cursor block lands.
        let all: String = (0..buf.area.height)
            .flat_map(|y| {
                (0..buf.area.width)
                    .map(move |x| buf[(x, y)].symbol().chars().next().unwrap_or(' '))
                    .chain(std::iter::once('\n'))
            })
            .collect();
        assert!(
            all.contains("hello\u{258b}"),
            "cursor not positioned after 'hello': {all}"
        );
    }

    #[test]
    fn draw_is_stable_across_tricky_input_states() {
        use ratatui::{backend::TestBackend, Terminal};
        let mut st = AppState::new("litellm", "DeepSeek-V4-Flash");
        // Empty input + processing + stale cursor (non-zero) -> must not panic.
        st.processing = true;
        st.cursor_ix = 7;
        st.input.clear();
        let backend = TestBackend::new(80, 12);
        let mut t1 = Terminal::new(backend).unwrap();
        t1.draw(|f| draw(f, &mut st)).unwrap();

        // Multi-line with cursor placed between two lines.
        let mut st2 = AppState::new("litellm", "DeepSeek-V4-Flash");
        st2.input.push_str("line one\nline two");
        st2.cursor_ix = 5; // right after "line "
        let b2 = TestBackend::new(80, 12);
        let mut t2 = Terminal::new(b2).unwrap();
        t2.draw(|f| draw(f, &mut st2)).unwrap();

        // Emoji mid-line then cursor inside it: floor_char_boundary safety.
        let mut st3 = AppState::new("litellm", "DeepSeek-V4-Flash");
        st3.input.push_str("a\u{1F600}b"); // a😀b
        st3.cursor_ix = 4; // inside the emoji bytes
        let b3 = TestBackend::new(80, 12);
        let mut t3 = Terminal::new(b3).unwrap();
        t3.draw(|f| draw(f, &mut st3)).unwrap();
    }

    #[test]
    fn capture_frame() {
        let mut st = AppState::new("litellm", "DeepSeek-V4-Flash");
        st.lines.push(ChatLine ::User("this is a very long user message that should definitely wrap onto multiple lines within the chat pane so we can confirm paragraph wrapping works correctly".into()));
        st.lines.push(ChatLine::Agent(
            "## Plan\nI will **build** and run it.\n```rust\nfn main(){}\n```".into(),
        ));
        st.lines.push(ChatLine::Tool("Bash echo hi".into()));
        st.input = "line one\nline two".to_string();
        st.set_plan(vec![
            "write code".to_string(),
            "test it".to_string(),
            "commit".to_string(),
        ]);
        st.plan[0].done = true;
        st.processing = true;
        st.spinner = 2;
        st.typing_text = "streaming demo: This is revealed gradually.".to_string();
        st.typing_shown = 28;
        st.input = "/c".to_string();
        update_completions(&mut st);
        let backend = TestBackend::new(120, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|f| draw(f, &mut st)).unwrap();
        let buf = terminal.backend().buffer();
        let area = buf.area;
        let mut lines = Vec::new();
        for y in 0..area.height {
            let mut row = String::new();
            for x in 0..area.width {
                row.push(buf[(x, y)].symbol().chars().next().unwrap_or(' '));
            }
            lines.push(row.trim_end().to_string());
        }
        let out = lines.join("\n");
        std::fs::write("/tmp/harxes_tui_frame.txt", out).unwrap();
        assert!(lines.len() > 5);
    }
}
