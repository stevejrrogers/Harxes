//! OpenAI-compatible Chat Completions driven adapter with native tool-calling
//! support. Also used for LiteLLM and other OpenAI-compatible proxies.

use async_trait::async_trait;
use harxes_core_domain::domain::value_objects::{
    Message, ProviderId, Role, TokenUsage, ToolCall, ToolSpec,
};
use harxes_core_domain::ports::{AgentResponse, LlmError, LlmPort};
use serde::Serialize;

#[derive(Serialize)]
struct FunctionDef<'a> {
    name: &'a str,
    description: &'a str,
    parameters: serde_json::Value,
}

#[derive(Serialize)]
struct ToolDef<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    function: FunctionDef<'a>,
}

#[derive(Serialize)]
struct RequestToolCall<'a> {
    id: &'a str,
    #[serde(rename = "type")]
    kind: &'static str,
    function: RequestFunction<'a>,
}

#[derive(Serialize)]
struct RequestFunction<'a> {
    name: &'a str,
    arguments: &'a str,
}

#[derive(Serialize)]
struct ApiMessage<'a> {
    role: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<RequestToolCall<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<&'a str>,
}

/// Build the OpenAI wire representation of a domain message.
fn to_api_message(m: &Message) -> ApiMessage<'_> {
    match m.role {
        Role::System => ApiMessage {
            role: "system",
            content: Some(m.content.clone()),
            tool_calls: vec![],
            tool_call_id: None,
        },
        Role::User => ApiMessage {
            role: "user",
            content: Some(m.content.clone()),
            tool_calls: vec![],
            tool_call_id: None,
        },
        Role::Tool => ApiMessage {
            role: "tool",
            content: Some(m.content.clone()),
            tool_calls: vec![],
            tool_call_id: m.tool_call_id.as_deref(),
        },
        Role::Assistant if !m.tool_calls.is_empty() => {
            let calls = m
                .tool_calls
                .iter()
                .map(|c| RequestToolCall {
                    id: &c.id,
                    kind: "function",
                    function: RequestFunction {
                        name: &c.name,
                        arguments: &c.arguments,
                    },
                })
                .collect();
            // content may carry a preamble; keep it if present
            let content = if m.content.is_empty() {
                None
            } else {
                Some(m.content.clone())
            };
            ApiMessage {
                role: "assistant",
                content,
                tool_calls: calls,
                tool_call_id: None,
            }
        }
        Role::Assistant => ApiMessage {
            role: "assistant",
            content: Some(m.content.clone()),
            tool_calls: vec![],
            tool_call_id: None,
        },
    }
}

fn build_tools(tools: &[ToolSpec]) -> Vec<ToolDef<'_>> {
    tools.iter().map(|t|ToolDef{
          kind :"function",
           function :FunctionDef{
                name :&t.id ,
                 description :&t.description ,
                  parameters :serde_json::json!({"type":"object","properties":{},"additionalProperties":true}),
            },
      }).collect()
}

#[derive(Serialize)]
struct RequestBody<'a> {
    model: &'a str,
    messages: Vec<ApiMessage<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ToolDef<'a>>,
}

#[derive(serde::Deserialize)]
struct ResponseBody {
    choices: Vec<Choice>,
    usage: Usage,
}

#[derive(serde::Deserialize)]
struct Choice {
    message: ResponseMessage,
}

#[derive(serde::Deserialize)]
struct ResponseMessage {
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<ResponseToolCall>,
}

#[derive(serde::Deserialize)]
struct ResponseToolCall {
    id: String,
    function: ResponseFunction,
}
#[derive(serde::Deserialize)]
struct ResponseFunction {
    name: String,
    arguments: String,
}

#[derive(serde::Deserialize)]
struct Usage {
    prompt_tokens: u64,
    completion_tokens: u64,
}

/// OpenAI Chat Completions API driven adapter with native tool-calling.
pub struct OpenAiClient {
    http: reqwest::Client,
    base_url: String,
    api_key: String,
}

impl OpenAiClient {
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            // `base_url` is a FULL endpoint URL; do not append path.
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key: api_key.into(),
        }
    }
    fn url(&self) -> String {
        self.base_url.clone()
    }
}

#[async_trait]
impl LlmPort for OpenAiClient {
    async fn generate(
        &self,
        provider: &ProviderId,
        model_id: &str,
        messages: &[Message],
        tools: &[harxes_core_domain::domain::value_objects::ToolSpec],
        temperature: Option<f64>,
    ) -> Result<AgentResponse, LlmError> {
        let api_messages = messages.iter().map(to_api_message).collect();
        let body = RequestBody {
            model: model_id,
            messages: api_messages,
            temperature,
            tools: build_tools(tools),
        };
        let resp = match self
            .http
            .post(self.url())
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => return Err(LlmError::Request(format!("transport error : {e}"))),
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
        let rb: ResponseBody = resp
            .json()
            .await
            .map_err(|e| LlmError::Request(format!("decode error : {e}")))?;
        let first = rb.choices.into_iter().next();
        match first {
            Some(choice) => {
                let content = choice.message.content.unwrap_or_default();
                let tool_calls = choice
                    .message
                    .tool_calls
                    .into_iter()
                    .map(|tc| ToolCall {
                        id: tc.id,
                        name: tc.function.name,
                        arguments: tc.function.arguments,
                    })
                    .collect();
                Ok(AgentResponse {
                    content,
                    usage: TokenUsage::new(rb.usage.prompt_tokens, rb.usage.completion_tokens),
                    tool_calls,
                })
            }
            None => Ok(AgentResponse::text(String::new(), TokenUsage::default())),
        }
    }
}
