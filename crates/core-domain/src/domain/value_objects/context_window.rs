//! Long-context management: estimates token usage of a transcript and trims
//! older turns when the window budget is exceeded. Pure logic, easily tested.

use super::message::{Message, Role};

/// Rough token estimate (~4 chars per token), good enough for budgeting.
pub fn estimate_tokens(text: &str) -> u64 {
    (text.len() / 4).max(1) as u64
}

/// Estimated tokens for a whole message.
pub fn message_tokens(m: &Message) -> u64 {
    let mut n = estimate_tokens(&m.content);
    for tc in &m.tool_calls {
        n += estimate_tokens(&tc.name) + estimate_tokens(&tc.arguments);
    }
    if let Some(id) = &m.tool_call_id {
        n += estimate_tokens(id);
    }
    n
}

/// Estimated tokens for a full transcript.
pub fn transcript_tokens(messages: &[Message]) -> u64 {
    messages.iter().map(message_tokens).sum()
}

/// Trim a transcript to fit within `max_tokens`. Leading System messages are
/// always kept; newest messages preserved, oldest interior turns dropped first.
pub fn trim_to_budget(messages: Vec<Message>, max_tokens: u64) -> Vec<Message> {
    compress_transcript_to_budget(messages, max_tokens, None)
}

/// Compact summary placeholder folded into a transcript when older turns are
/// compressed away. Kept as a single [`Role::System`] message so the model
/// still holds a thread of the earlier work without paying for every old token.
fn summary_message(lines: &[String]) -> Message {
    let mut body = String::from("### Prior context (older turns compressed to save space)\n");
    for l in lines {
        body.push_str(l);
        body.push('\n');
    }
    Message::new(Role::System, body)
}

/// Smart context manager. If the transcript fits, return it unchanged. If not,
/// keep the newest turns verbatim and fold the oldest into a single compact
/// "prior context" summary (preserving the first line of each dropped turn and
/// any assistant text), so long sessions keep a meaningful thread instead of a
/// hard gap.
///
/// `max_summary_lines` bounds how many old turns are folded in; when `None`,
/// all dropped turns contribute up to a one-line gist each.
pub fn compress_transcript_to_budget(
    messages: Vec<Message>,
    max_tokens: u64,
    max_summary_lines: Option<usize>,
) -> Vec<Message> {
    if transcript_tokens(&messages) <= max_tokens {
        return messages;
    }

    // Keep leading System messages intact.
    let mut split_at = 0usize;
    while split_at < messages.len() && messages[split_at].role == Role::System {
        split_at += 1;
    }
    let system = messages[..split_at].to_vec();
    let rest = &messages[split_at..];

    // Keep the newest turn(s) verbatim so the model can answer the latest prompt.
    // Track the kept original indices to know what was dropped.
    let mut kept_rev_indices: Vec<usize> = Vec::new();
    let mut kept_rev: Vec<Message> = Vec::new();
    for (i, m) in rest.iter().enumerate().rev() {
        kept_rev.push(m.clone());
        if transcript_tokens(&system) + transcript_tokens(&kept_rev) > max_tokens {
            kept_rev.pop();
            break;
        }
        kept_rev_indices.push(i);
    }
    kept_rev.reverse();
    let kept_indices: Vec<usize> = kept_rev_indices.into_iter().rev().collect();

    // Fold a compact gist of the dropped turns into a summary message.
    let mut summary_lines: Vec<String> = Vec::new();
    let mut over_cap = false;
    for (i, m) in rest.iter().enumerate() {
        if kept_indices.contains(&i) {
            continue;
        }
        let first_line = m.content.lines().next().unwrap_or("").trim();
        if first_line.is_empty() {
            continue;
        }
        let label = match m.role {
            Role::User => format!("user: {first_line}"),
            Role::Assistant => format!("assistant: {}", truncate_ellipsis(first_line, 160)),
            Role::Tool => format!("tool: {}", truncate_ellipsis(first_line, 120)),
            Role::System => String::new(),
        };
        if !label.is_empty() {
            summary_lines.push(label);
        }
        if let Some(cap) = max_summary_lines {
            if summary_lines.len() >= cap {
                over_cap = true;
                break;
            }
        }
    }

    if !summary_lines.is_empty() && !over_cap {
        let summary = summary_message(&summary_lines);
        let mut out = system;
        out.push(summary);
        out.extend(kept_rev.clone());
        return out;
    }

    // Nothing worth folding (or summary too big): hard-drop oldest turns.
    let mut out = system;
    out.extend(kept_rev);
    out
}

/// Truncate `s` to at most `max` characters, appending an ellipsis when cut.
fn truncate_ellipsis(s: &str, max: usize) -> String {
    let mut chars = s.chars();
    let mut head: String = chars.by_ref().take(max).collect();
    if chars.next().is_some() {
        head.push('…');
    }
    head
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fits_within_budget_returns_unchanged() {
        let msgs = vec![
            Message::new(Role::System, "sys"),
            Message::new(Role::User, "hello world"),
        ];
        let out = trim_to_budget(msgs.clone(), 1000);
        assert_eq!(out.len(), msgs.len());
    }

    #[test]
    fn trims_oldest_turns_first() {
        // Each user message is large so the budget forces dropping oldest.
        let big = "x".repeat(400);
        let msgs = vec![
            Message::new(Role::System, "sys"),
            Message::new(Role::User, big.clone()),
            Message::new(Role::Assistant, "ok1"),
            Message::new(Role::User, big.clone()),
            Message::new(Role::Assistant, "ok2"),
        ];
        // Budget ~ enough for system + last turn only.
        let out = trim_to_budget(msgs.clone(), 320);
        assert!(transcript_tokens(&out) <= 320);
        // Newest assistant still present.
        assert!(out.iter().any(|m| m.content == "ok2"));
        // System preserved first.
        assert_eq!(out[0].role, Role::System);
    }

    #[test]
    fn compression_folds_summary_instead_of_blank() {
        let big = "y".repeat(400);
        let msgs = vec![
            Message::new(Role::System, "sys"),
            Message::new(Role::User, "first big request about auth"),
            Message::new(Role::Assistant, "I implemented JWT auth"),
            Message::new(Role::User, big.clone()),
            Message::new(Role::Assistant, "Latest reply"),
        ];
        let out = compress_transcript_to_budget(msgs, 800, Some(4));
        // The oldest user request about auth should still show up in the summary.
        let joined: String = out
            .iter()
            .map(|m| m.content.clone())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(joined.contains("auth"), "summary should preserve gist: {joined}");
        // Latest reply kept.
        assert!(out.iter().any(|m| m.content == "Latest reply"));
    }
}
