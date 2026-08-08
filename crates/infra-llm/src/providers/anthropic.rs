use async_trait::async_trait;
use harxes_core_domain::domain::value_objects::{Message, ProviderId, Role, ToolSpec};
use harxes_core_domain::ports::{AgentResponse, LlmError, LlmPort};
use serde::Serialize;

#[derive(Serialize)]
struct ApiMessage {
    role: String,
    content: String,
}

impl From<&Message> for ApiMessage {
    fn from(m: &Message) -> Self {
        let role = match m.role {
            Role::System | Role::User | Role::Tool => "user".to_string(),
            Role::Assistant => "assistant".to_string(),
        };
        Self {
            role,
            content: m.content.clone(),
        }
    }
}

#[derive(Serialize)]
struct RequestBody<'a> {
    model: &'a str,
    max_tokens: u32,
    system: Option<&'a str>,
    messages: Vec<ApiMessage>,
    temperature: Option<f64>,
}

#[derive(serde::Deserialize)]
struct ResponseBody {
    content: Vec<ContentBlock>,
    usage: Usage,
}

#[derive(serde::Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    kind: String,
    text: Option<String>,
}

#[derive(serde::Deserialize)]
struct Usage {
    input_tokens: u64,
    output_tokens: u64,
}

const DEFAULT_MAX_TOKENS: u32 = 4096;

/// Anthropic Messages API driven adapter.
pub struct AnthropicClient {
    http: reqwest::Client,
    base_url: String,
    api_key: String,
}

impl AnthropicClient {
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            // `base_url` is a FULL endpoint URL (e.g. .../v1/messages).
            // Do NOT append any path here to avoid double-path bugs.
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key: api_key.into(),
        }
    }

    fn url(&self) -> String {
        self.base_url.clone()
    }
}

#[async_trait]
impl LlmPort for AnthropicClient {
    async fn generate(
        &self,
        provider: &ProviderId,
        model_id: &str,
        messages: &[Message],
        _tools: &[ToolSpec],
        temperature: Option<f64>,
    ) -> Result<AgentResponse, LlmError> {
        let system = messages
            .iter()
            .find(|m| m.role == Role::System)
            .map(|m| m.content.as_str());
        let api_messages: Vec<_> = messages
            .iter()
            .filter(|m| m.role != Role::System)
            .map(ApiMessage::from)
            .collect();

        let body = RequestBody {
            model: model_id,
            max_tokens: DEFAULT_MAX_TOKENS,
            system,
            messages: api_messages,
            temperature,
        };

        let resp = self
            .http
            .post(self.url())
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await;

        let resp = match resp {
            Ok(r) => r,
            Err(e) => return Err(LlmError::Request(format!("transport error: {e}"))),
        };

        if resp.status().is_server_error()
            || resp.status() == reqwest::StatusCode::TOO_MANY_REQUESTS
        {
            return Err(LlmError::RateLimited { retry_after: None });
        }
        if resp.status() == reqwest::StatusCode::UNAUTHORIZED
            || resp.status() == reqwest::StatusCode::FORBIDDEN
        {
            return Err(LlmError::Auth {
                provider: provider.clone(),
            });
        }
        if !resp.status().is_success() {
            return Err(LlmError::Request(format!("HTTP {}", resp.status())));
        }

        let parsed: Result<ResponseBody, _> = resp.json().await;
        let rb = match parsed {
            Ok(x) => x,
            Err(e) => return Err(LlmError::Request(format!("decode error: {e}"))),
        };

        let text = rb
            .content
            .into_iter()
            .filter(|c| c.kind == "text")
            .filter_map(|c| c.text)
            .collect::<Vec<_>>()
            .join("\n");

        Ok(AgentResponse::text(
            text,
            harxes_core_domain::domain::value_objects::TokenUsage::new(
                rb.usage.input_tokens,
                rb.usage.output_tokens,
            ),
        ))
    }
}
