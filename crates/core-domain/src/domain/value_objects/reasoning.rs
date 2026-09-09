//! Reasoning-effort control: a provider-agnostic dial for how much hidden
//! "thinking" a model should spend. Maps to `reasoning_effort` on
//! OpenAI-compatible endpoints and to a thinking `budget_tokens` on Anthropic.
//! Providers that don't support it ignore the hint.

use serde::{Deserialize, Serialize};

/// How much reasoning a model should spend before answering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    Low,
    Medium,
    High,
}

impl ReasoningEffort {
    /// The OpenAI-compatible `reasoning_effort` string.
    pub fn as_str(&self) -> &'static str {
        match self {
            ReasoningEffort::Low => "low",
            ReasoningEffort::Medium => "medium",
            ReasoningEffort::High => "high",
        }
    }

    /// Parse from a string (case-insensitive); `None` for anything else.
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_lowercase().as_str() {
            "low" => Some(ReasoningEffort::Low),
            "medium" | "med" => Some(ReasoningEffort::Medium),
            "high" => Some(ReasoningEffort::High),
            _ => None,
        }
    }

    /// An Anthropic thinking `budget_tokens` value for this effort level.
    pub fn anthropic_budget_tokens(&self) -> u32 {
        match self {
            ReasoningEffort::Low => 1024,
            ReasoningEffort::Medium => 6_000,
            ReasoningEffort::High => 16_000,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_and_render_roundtrip() {
        assert_eq!(ReasoningEffort::parse("HIGH"), Some(ReasoningEffort::High));
        assert_eq!(ReasoningEffort::parse("med"), Some(ReasoningEffort::Medium));
        assert_eq!(ReasoningEffort::parse("nope"), None);
        assert_eq!(ReasoningEffort::Low.as_str(), "low");
        assert!(ReasoningEffort::High.anthropic_budget_tokens() > ReasoningEffort::Low.anthropic_budget_tokens());
    }
}
