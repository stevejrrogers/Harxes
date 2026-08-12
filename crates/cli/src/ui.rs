//! Terminal UI helpers: live tool-feedback observer, ANSI styling, banners,
//! turn dividers and syntax highlighting for code blocks.

use harxes_core_domain::ports::ToolObserver;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, OnceLock};

/// ANSI color helper wrapping raw escape codes.
pub struct C;
impl C {
    pub fn bold(s: &str) -> String {
        format!("\x1b[1m{}\x1b[0m", s)
    }
    pub fn cyan(s: &str) -> String {
        format!("\x1b[36m{}\x1b[0m", s)
    }
    pub fn green(s: &str) -> String {
        format!("\x1b[32m{}\x1b[0m", s)
    }
    pub fn yellow(s: &str) -> String {
        format!("\x1b[33m{}\x1b[0m", s)
    }
    pub fn dim(s: &str) -> String {
        format!("\x1b[2m{}\x1b[0m", s)
    }
}

/// Prints live tool feedback lines as tools execute (e.g. "⏺ Bash(cmd)") and
/// mirrors the currently-executing tools into a shared buffer the TUI renders.
pub struct LiveToolObserver {
    pub active: Arc<Mutex<Vec<String>>>,
    /// Set to true once any streaming delta has been surfaced. The one-shot CLI
    /// uses this to avoid re-printing the fully-rendered answer after the live
    /// stream already showed the text.
    pub streamed: Arc<AtomicBool>,
    /// Optional sender for live text deltas. When set (TUI mode), each streaming
    /// token is forwarded for per-token typewriter rendering instead of being
    /// printed to stdout.
    pub deltas: Option<std::sync::mpsc::Sender<String>>,
}
impl ToolObserver for LiveToolObserver {
    fn on_tool_start(&self, name: &str, args_preview: &str) {
        if let Ok(mut a) = self.active.lock() {
            a.push(format!("{name} {args_preview}"));
        }
        if std::env::var("HARXES_TUI").is_ok() {
            return;
        }
        println!(
            "{} {} {}",
            C::cyan("⏺"),
            C::bold(name),
            C::dim(args_preview)
        );
    }
    fn on_tool_result(&self, _name: &str, _result_preview: &str) {}
    fn on_retry(&self, wait_secs: u64) {
        if std::env::var("HARXES_TUI").is_ok() {
            return;
        }
        println!(
            "{} transient error — retrying in {}s",
            C::yellow("⟳"),
            C::dim(&wait_secs.to_string())
        );
    }
    fn on_stream_delta(&self, text: &str) {
        self.streamed.store(true, std::sync::atomic::Ordering::Relaxed);
        if std::env::var("HARXES_TUI").is_ok() {
            // In TUI mode, forward the delta for live rendering.
            if let Some(tx) = &self.deltas {
                let _ = tx.send(text.to_string());
            }
            return;
        }
        use std::io::Write as _;
        let mut out = std::io::stdout().lock();
        let _ = out.write_all(text.as_bytes());
        let _ = out.flush();
    }
}

fn ctx() -> (
    &'static syntect::parsing::SyntaxSet,
    &'static syntect::highlighting::Theme,
) {
    static SS: OnceLock<syntect::parsing::SyntaxSet> = OnceLock::new();
    static TH: OnceLock<syntect::highlighting::Theme> = OnceLock::new();
    let ss = SS.get_or_init(syntect::parsing::SyntaxSet::load_defaults_newlines);
    let th = TH.get_or_init(|| {
        let ts = syntect::highlighting::ThemeSet::load_defaults();
        ts.themes["base16-ocean.dark"].clone()
    });
    (ss, th)
}

fn hl_line(
    ss: &syntect::parsing::SyntaxSet,
    theme: &syntect::highlighting::Theme,
    syntax: &syntect::parsing::SyntaxReference,
    line: &str,
) -> String {
    use std::fmt::Write as _;
    use syntect::easy::HighlightLines;
    use syntect::highlighting::Style;
    let mut h = HighlightLines::new(syntax, theme);
    let ranges = h.highlight_line(line, ss).unwrap_or_default();
    let mut out = String::new();
    for (st, txt) in ranges {
        if txt.is_empty() {
            continue;
        }
        if st == Style::default() {
            out.push_str(txt);
        } else {
            let fg = st.foreground;
            let _ = write!(out, "\x1b[38;2;{};{};{}m{}\x1b[0m", fg.r, fg.g, fg.b, txt);
        }
    }
    out
}

/// Render model text with fenced ```code``` blocks syntax-highlighted.
pub fn render_assistant(text: &str) -> String {
    if !text.contains("```") {
        return text.to_string();
    }
    let (ss, th) = ctx();
    let mut out = String::new();
    let mut rest = text;
    while let Some(pos) = rest.find("```") {
        out.push_str(&rest[..pos]);
        rest = &rest[pos + 3..];
        // language tag line
        let le = rest.find('\n').unwrap_or(rest.len());
        let lang = rest[..le].trim().to_string();
        rest = if le < rest.len() { &rest[le + 1..] } else { "" };
        match rest.find("```") {
            Some(end) => {
                if !lang.is_empty() {
                    if let Some(syn) = ss.find_syntax_by_token(&lang) {
                        for line in rest[..end].split('\n') {
                            out.push_str(&hl_line(ss, th, syn, line));
                            out.push('\n');
                        }
                    } else {
                        for line in rest[..end].split('\n') {
                            out.push_str(&C::green(line));
                            out.push('\n');
                        }
                    }
                } else {
                    for line in rest[..end].split('\n') {
                        out.push_str(&C::green(line));
                        out.push('\n');
                    }
                }
                rest = &rest[end + 3..];
            }
            None => break,
        }
    }
    out.push_str(rest);
    out
}

/// Toggle TUI mode: when true, suppress stdio tool feedback.
pub fn tui_mode(mode: bool) {
    if mode {
        std::env::set_var("HARXES_TUI", "1");
    } else {
        std::env::remove_var("HARXES_TUI");
    }
}
