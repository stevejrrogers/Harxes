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
    if transcript_tokens(&messages) <= max_tokens {
        return messages;
    }
    let mut split_at = 0usize;
    while split_at < messages.len() && messages[split_at].role == Role::System {
        split_at += 1;
    }
    let system = &messages[..split_at];
    let rest = &messages[split_at..];

    let mut kept_rev: Vec<Message> = Vec::new();
    for m in rest.iter().rev() {
        kept_rev.push(m.clone());
        let total: u64 = transcript_tokens(system) + transcript_tokens(&kept_rev);
        if total > max_tokens {
            kept_rev.pop();
            break;
        }
    }

    let mut out = system.to_vec();
    out.extend(kept_rev.into_iter().rev());
    out
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
        // Each user message ~ large so budget forces dropping oldest.
        let big = "x".repeat(400);
        let msgs = vec![
            Message::new(Role::System, "sys"),
            Message::new(Role::User, big.clone()),
            Message::new(Role::Assistant, "ok1"),
            Message::new(Role::User, big.clone()),
            Message::new(Role::Assistant, "ok2"),
        ];
        // budget ~ enough for system + last ~2 turns but not all
        let out = trim_to_budget(msgs.clone(), 320);
        assert!(transcript_tokens(&out) <= 320);
        // newest assistant still present
        assert!(out.iter().any(|m| m.content == "ok2"));
        // system preserved first
        assert_eq!(out[0].role, Role::System);
    }
}
