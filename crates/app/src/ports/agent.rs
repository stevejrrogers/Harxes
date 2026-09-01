//! Application-layer ports (the "driving"/left side of hexagonal): interfaces
//! the outer layers (CLI) depend on to interact with the application without
//! touching concrete usecase structs.

use async_trait::async_trait;

use harxes_core_domain::domain::value_objects::{Message, ProviderId};
use harxes_core_domain::ports::LlmError;

use crate::usecases::{agent_loop::LoopLimits, agent_loop::LoopOutcome};

/// Outcome of one conversational turn: the visible text plus the updated
/// transcript so callers can carry conversation state across turns.
#[derive(Debug)]
pub struct ConversationResult {
    pub outcome: LoopOutcome,
    pub transcript: Vec<Message>,
}

/// Driving port: the primary entry point used by the CLI to run an agent.
#[async_trait]
pub trait AgentPort: Send + Sync {
    /// Run a full agent session (multi-turn tool loop) and return its outcome.
    async fn run(
        &self,
        provider_id: &ProviderId,
        model_id: &str,
        system_prompt: &str,
        user_prompt: &str,
        limits: &LoopLimits,
    ) -> Result<LoopOutcome, LlmError>;

    /// Continue a conversation from prior history with a new user turn.
    async fn continue_chat(
        &self,
        provider_id: &ProviderId,
        model_id: &str,
        history: &[Message],
        user_prompt: &str,
        limits: &LoopLimits,
    ) -> Result<ConversationResult, LlmError>;

    /// Like [`Self::continue_chat`] but takes a full user [`Message`], so
    /// callers can attach images (vision input). Default drops attachments.
    async fn continue_chat_with(
        &self,
        provider_id: &ProviderId,
        model_id: &str,
        history: &[Message],
        user: Message,
        limits: &LoopLimits,
    ) -> Result<ConversationResult, LlmError> {
        self.continue_chat(provider_id, model_id, history, &user.content, limits)
            .await
    }
}
