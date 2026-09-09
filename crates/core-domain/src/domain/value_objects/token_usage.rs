use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
    /// Reasoning ("thinking") tokens, when the provider reports them separately
    /// (a subset of output_tokens). `None` = not reported.
    pub reasoning_tokens: Option<u64>,
}

impl TokenUsage {
    pub fn new(input_tokens: u64, output_tokens: u64) -> Self {
        Self {
            input_tokens,
            output_tokens,
            total_tokens: input_tokens + output_tokens,
            reasoning_tokens: None,
        }
    }

    /// Set the reasoning-token count (builder style).
    pub fn with_reasoning(mut self, reasoning: Option<u64>) -> Self {
        self.reasoning_tokens = reasoning;
        self
    }
}
