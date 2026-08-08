use async_trait::async_trait;

use crate::domain::value_objects::{Message, ProviderId, TokenUsage, ToolCall, ToolSpec};

#[derive(Debug, Clone)]
pub struct AgentResponse {
    pub content: String,
    pub usage: TokenUsage,
    /// Tool-call requests from the model (empty for a plain text turn).
    pub tool_calls: Vec<ToolCall>,
}

impl AgentResponse {
    pub fn text(content: impl Into<String>, usage: TokenUsage) -> Self {
        Self {
            content: content.into(),
            usage,
            tool_calls: vec![],
        }
    }
}

#[derive(Debug)]
pub enum LlmError {
    Request(String),
    Auth { provider: ProviderId },
    RateLimited { retry_after: Option<u64> },
    Timeout,
}

impl std::fmt::Display for LlmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Request(msg) => write!(f, "request error: {msg}"),
            Self::Auth { provider } => write!(
                f,
                "authentication failed for {}; check API key",
                provider.as_str()
            ),
            Self::RateLimited { retry_after } => match retry_after {
                Some(secs) => write!(f, "rate limited; retry after {secs}s"),
                None => write!(f, "rate limited"),
            },
            Self::Timeout => write!(f, "request timed out"),
        }
    }
}

impl std::error::Error for LlmError {}

/// Driven port (hexagonal): infrastructure implements this with a concrete
/// LLM provider HTTP client. Domain and application depend on this trait only.
#[async_trait]
pub trait LlmPort: Send + Sync {
    /// Generate one assistant turn from a chat transcript, offering the model
    /// the given tools to call. Implementations may populate
    /// [`AgentResponse::tool_calls`] when the model requests tool use.
    async fn generate(
        &self,
        provider: &ProviderId,
        model_id: &str,
        messages: &[Message],
        tools: &[ToolSpec],
        temperature: Option<f64>,
    ) -> Result<AgentResponse, LlmError>;
}
