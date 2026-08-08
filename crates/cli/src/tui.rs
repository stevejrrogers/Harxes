//! Full-screen TUI (ratatui) - opencode-style layout.
//! Title bar, scrollable chat pane with avatars and syntax highlighting,
//! a right-side status panel, and a separate input box at the bottom.

use crossterm::event::{self, Event, KeyCode};
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
    pub tools_used: Vec<String>,
}

pub struct AppState {
    pub lines: Vec<ChatLine>,
    pub input: String,
    pub processing: bool,
    pub scroll: u16,
    pub status: StatusInfo,
    pub history: Vec<String>,
    pub hist_pos: Option<usize>,
    pub auto_scroll: bool,
    pub spinner: u32,
}

impl AppState {
    pub fn new(provider: &str, model: &str) -> Self {
        Self {
            lines: vec![],
            input: String::new(),
            processing: false,
            scroll: 0,
            status: StatusInfo {
                provider: provider.to_string(),
                model: model.to_string(),
                total_tokens: 0,
                tools_used: vec![],
            },
            history: vec![],
            hist_pos: None,
            auto_scroll: true,
            spinner: 0,
        }
    }
}
/// Background tints for panes (opencode-style dark).
const BG_TITLE: Color = Color::Indexed(237);
const BG_CHAT: Color = Color::Indexed(235);
const BG_STATUS: Color = Color::Indexed(234);
const BG_INPUT: Color = Color::Indexed(236);

fn title_bar(frame: &mut Frame, state: &AppState, area: Rect) {
    let (badge, col) = if state.processing {
        ("THINKING", Color::Yellow)
    } else {
        ("READY", Color::Green)
    };
    let spin = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧"];
    let indicator = if state.processing {
        format!("{} {badge} ", spin[(state.spinner as usize) % spin.len()])
    } else {
        format!("● {badge} ")
    };
    let line = Line::from(vec![
        Span::styled("  ✦ Harxes ", Style::new().fg(Color::Magenta).bold()),
        Span::styled(indicator, Style::new().fg(col).bold()),
        Span::styled(&state.status.provider, Style::new().fg(Color::Cyan)),
        Span::raw(" / "),
        Span::styled(&state.status.model, Style::new().fg(Color::Blue)),
    ]);
    frame.render_widget(
        Paragraph::new(line).style(Style::default().bg(BG_TITLE).fg(Color::White)),
        area,
    );
}

fn agent_lines(text: &str) -> Vec<Line<'static>> {
    if !text.contains("```") {
        return vec![Line::from(vec![Span::raw(text.to_string())])];
    }
    let mut out = Vec::new();
    let mut rest = text;
    loop {
        match rest.find("```") {
            None => {
                out.push(Line::from(vec![Span::raw(rest.to_string())]));
                break;
            }
            Some(p) => {
                if p > 0 {
                    out.push(Line::from(vec![Span::raw(rest[..p].to_string())]));
                }
                rest = &rest[p + 3..];
                if let Some(e) = rest.find("```") {
                    let mut body = &rest[..e];
                    let mut lang_hint = String::new();
                    if let Some(nl) = body.find('\n') {
                        lang_hint = body[..nl].trim().to_string();
                        if !lang_hint.is_empty() {
                            out.push(Line::from(vec![Span::styled(
                                format!("[{}]", lang_hint),
                                Style::new().fg(Color::Yellow).bold(),
                            )]));
                        }
                        body = &body[nl + 1..];
                    }
                    for l in code_block_lines(&lang_hint, body) {
                        out.push(l);
                    }
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
                    Span::styled("❯ ", Style::new().fg(Color::Green).bold()),
                    Span::raw(t.to_string()),
                ]));
            }
            ChatLine::Agent(t) => {
                text.push_line(Line::from(vec![Span::styled(
                    "✦ ",
                    Style::new().fg(Color::Magenta).bold(),
                )]));
                for l in agent_lines(t) {
                    text.push_line(l);
                }
                text.push_line(Line::from(vec![Span::raw("")]));
            }
            ChatLine::Tool(t) => {
                text.push_line(Line::from(vec![
                    Span::styled("⏺ ", Style::new().fg(Color::Cyan)),
                    Span::raw(t.to_string()),
                ]));
            }
        }
    }
    let para = Paragraph::new(text)
        .style(Style::default().bg(BG_CHAT).fg(Color::White))
        .wrap(ratatui::widgets::Wrap { trim: true })
        .scroll((state.scroll, 0));
    frame.render_widget(para, area);
}

fn status_panel(frame: &mut Frame, state: &AppState, area: Rect) {
    let mut text = ratatui::text::Text::default();
    text.push_line(Line::from(vec![Span::styled(
        "  STATUS ",
        Style::new().fg(Color::Magenta).bold(),
    )]));
    text.push_line(Line::from(vec![Span::raw("")]));
    text.push_line(Line::from(vec![
        Span::styled("provider ", Style::new().fg(Color::Cyan).bold()),
        Span::raw(state.status.provider.clone()),
    ]));
    text.push_line(Line::from(vec![
        Span::styled("model    ", Style::new().fg(Color::Blue).bold()),
        Span::raw(state.status.model.clone()),
    ]));
    text.push_line(Line::from(vec![
        Span::styled("tokens   ", Style::new().fg(Color::Yellow).bold()),
        Span::raw(state.status.total_tokens.to_string()),
    ]));
    text.push_line(Line::from(vec![Span::raw("")]));
    text.push_line(Line::from(vec![Span::styled(
        "  tools used ",
        Style::new().fg(Color::Cyan).bold(),
    )]));
    for t in &state.status.tools_used {
        text.push_line(Line::from(vec![Span::raw(format!("   ⏺ {t}"))]));
    }

    frame.render_widget(
        Paragraph::new(text).style(Style::default().bg(BG_STATUS).fg(Color::White)),
        area,
    );
}

fn input_pane(frame: &mut Frame, state: &AppState, area: Rect) {
    let content = Line::from(vec![
        Span::styled("❯ ", Style::new().fg(Color::Green).bold()),
        Span::raw(state.input.clone()),
    ]);
    frame.render_widget(
        Paragraph::new(content)
            .style(Style::default().bg(BG_INPUT).fg(Color::White))
            .block(
                Block::default()
                    .borders(Borders::TOP)
                    .border_style(Style::new().fg(Color::DarkGray)),
            ),
        area,
    );
}

pub fn draw(frame: &mut Frame, state: &mut AppState) {
    let area = frame.area();
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Min(1),
            Constraint::Length(2),
        ])
        .split(area);
    title_bar(frame, state, outer[0]);
    let mid = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(78), Constraint::Percentage(22)])
        .split(outer[1]);
    chat_pane(frame, state, mid[0]);
    status_panel(frame, state, mid[1]);
    input_pane(frame, state, outer[2]);
}

pub fn run(
    state: &mut AppState,
    mut on_command: impl FnMut(&mut AppState, String),
    mut poll_events: impl FnMut(&mut AppState),
) -> Result<(), Box<dyn std::error::Error>> {
    let mut terminal = match ratatui::try_init() {
        Ok(t) => t,
        Err(e) => return Err(format!("cannot init terminal: {e}").into()),
    };
    loop {
        if state.processing {
            state.spinner = state.spinner.wrapping_add(1);
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
                    } else {
                        break;
                    }
                    continue;
                }
                match k.code {
                    KeyCode::Char(c) => state.input.push(c),
                    KeyCode::Backspace => {
                        state.input.pop();
                    }
                    KeyCode::Enter => {
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
                            // remember for input history
                            if state.history.last().map(String::as_str) != Some(t) {
                                state.history.push(t.to_string());
                            }
                            state.hist_pos = None;
                            on_command(state, msg);
                        }
                        state.scroll = 0;
                    }
                    // Input history navigation (Up/Down)
                    KeyCode::Up => {
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
                        }
                        state.hist_pos = Some(pos);
                    }
                    KeyCode::Down => {
                        if let Some(p) = state.hist_pos {
                            if p == 0 {
                                state.input.clear();
                                state.hist_pos = None;
                            } else {
                                let np = p - 1;
                                let n = state.history.len();
                                if np < n {
                                    state.input = state.history[n - 1 - np].clone();
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

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{backend::TestBackend, Terminal};
    #[test]
    fn capture_frame() {
        let mut st = AppState::new("litellm", "DeepSeek-V4-Flash");
        st.lines.push(ChatLine ::User("this is a very long user message that should definitely wrap onto multiple lines within the chat pane so we can confirm paragraph wrapping works correctly".into()));
        st.lines.push(ChatLine::Agent(
            "Here is code:\n```rust\nfn main(){}\n```".into(),
        ));
        st.lines.push(ChatLine::Tool("Bash echo hi".into()));
        st.input = "my input here".to_string();
        st.processing = true;
        st.spinner = 2;
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
