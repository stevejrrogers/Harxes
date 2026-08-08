use std::sync::Arc;

use harxes_core_domain::domain::value_objects::{Message, ProviderId, Role, TokenUsage};
use harxes_core_domain::ports::{LlmError, LlmPort};

/// Application service (hexagonal: use-case ring). Depends only on ports.
pub struct AgentRunner {
    llm: Arc<dyn LlmPort>,
}

#[derive(Debug)]
pub enum RunError {
    Llm(LlmError),
    EmptyResponse,
}

impl std::fmt::Display for RunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Llm(e) => write!(f, "llm error: {e}"),
            Self::EmptyResponse => write!(f, "model returned empty response"),
        }
    }
}

impl std::error::Error for RunError {}

impl From<LlmError> for RunError {
    fn from(e: LlmError) -> Self {
        Self::Llm(e)
    }
}

#[derive(Debug)]
pub struct TurnResult {
    pub content: String,
    pub usage: TokenUsage,
}

pub struct AgentRunnerBuilder {
    llm: Option<Arc<dyn LlmPort>>,
}

impl AgentRunnerBuilder {
    pub fn new() -> Self {
        Self { llm: None }
    }

    pub fn with_llm(mut self, llm: Arc<dyn LlmPort>) -> Self {
        self.llm = Some(llm);
        self
    }

    pub fn build(self) -> Result<AgentRunner, String> {
        let llm = self.llm.ok_or("missing llm port")?;
        Ok(AgentRunner { llm })
    }
}

impl Default for AgentRunnerBuilder {
    fn default() -> Self {
        Self::new()
    }
}
impl AgentRunner {
    /// Execute one agent turn given a transcript.
    #[allow(clippy::too_many_arguments)]
    pub async fn run_turn(
        &self,
        provider_id: &ProviderId,
        model_id_str: &str,
        system_prompt: &str,
        user_prompt: &str,
        prior_messages: &[Message],
        temperature: Option<f64>,
    ) -> Result<TurnResult, RunError> {
        let mut messages = Vec::with_capacity(prior_messages.len() + 2);
        messages.push(Message::new(Role::System, system_prompt.to_string()));
        messages.extend_from_slice(prior_messages);
        messages.push(Message::new(Role::User, user_prompt.to_string()));

        let resp = self
            .llm
            .generate(
                provider_id,
                model_id_str,
                messages.as_slice(),
                &[],
                temperature,
            )
            .await?;

        if resp.content.trim().is_empty() {
            return Err(RunError::EmptyResponse);
        }

        Ok(TurnResult {
            content: resp.content,
            usage: resp.usage,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use harxes_core_domain::domain::value_objects::ProviderId;
    use harxes_core_domain::ports::AgentResponse;
    use std::sync::{Arc, Mutex};

    struct FakeLlm {
        response: String,
        calls: Mutex<Vec<(String, usize)>>,
    }

    #[async_trait::async_trait]
    impl LlmPort for FakeLlm {
        async fn generate(
            &self,
            provider: &ProviderId,
            model_id: &str,
            messages: &[Message],
            _tools: &[harxes_core_domain::domain::value_objects::ToolSpec],
            _temperature: Option<f64>,
        ) -> Result<AgentResponse, LlmError> {
            self.calls
                .lock()
                .unwrap()
                .push((provider.as_str().to_string(), messages.len()));
            let _ = model_id;
            Ok(AgentResponse::text(
                self.response.clone(),
                TokenUsage::new(10, 5),
            ))
        }
    }

    fn runner_with(response: &str) -> (AgentRunner, Arc<FakeLlm>) {
        let llm = Arc::new(FakeLlm {
            response: response.to_string(),
            calls: Mutex::new(vec![]),
        });
        let runner = AgentRunnerBuilder::new()
            .with_llm(llm.clone())
            .build()
            .unwrap();
        (runner, llm)
    }

    #[tokio::test]
    async fn run_turn_builds_system_user_messages_and_returns_content() {
        let (runner, llm) = runner_with("Hello from model");
        let provider = ProviderId::new("anthropic").unwrap();

        let res = runner
            .run_turn(&provider, "claude-sonnet-4-5", "be brief", "hi", &[], None)
            .await
            .unwrap();

        assert_eq!(res.content, "Hello from model");
        assert_eq!(res.usage.total_tokens, 15);
        // one system + one user message
        let calls = llm.calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].1, 2);
    }

    #[tokio::test]
    async fn run_turn_appends_prior_messages() {
        let (runner, llm) = runner_with("ok");
        let provider = ProviderId::new("anthropic").unwrap();
        let prior = vec![Message::new(Role::Assistant, "earlier")];

        runner
            .run_turn(
                &provider,
                "claude-sonnet-4-5",
                "sys",
                "prompt",
                &prior,
                Some(0.7),
            )
            .await
            .unwrap();

        // system + prior + user = 3 messages
        assert_eq!(llm.calls.lock().unwrap()[0].1, 3);
    }

    #[tokio::test]
    async fn empty_response_is_error() {
        let (runner, _llm) = runner_with("   ");
        let provider = ProviderId::new("anthropic").unwrap();
        assert!(matches!(
            runner
                .run_turn(&provider, "claude-sonnet-4-5", "sys", "hi", &[], None)
                .await,
            Err(RunError::EmptyResponse)
        ));
    }
}
