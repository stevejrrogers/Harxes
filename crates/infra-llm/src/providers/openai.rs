//! OpenAI-compatible Chat Completions driven adapter with native tool-calling
//! support. Also used for LiteLLM and other OpenAI-compatible proxies.

use async_trait::async_trait;
use harxes_core_domain::domain::value_objects::{
    Message, ProviderId, Role, TokenUsage, ToolCall, ToolSpec,
};
use harxes_core_domain::ports::{AgentResponse, LlmError, LlmPort, StreamSink};
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

    async fn generate_stream(
        &self,
        _provider: &ProviderId,
        model_id: &str,
        messages: &[Message],
        tools: &[harxes_core_domain::domain::value_objects::ToolSpec],
        temperature: Option<f64>,
        sink: StreamSink,
    ) -> Result<AgentResponse, LlmError> {
        use futures_util::StreamExt;
        use harxes_core_domain::ports::StreamEvent;

        let api_messages = messages.iter().map(to_api_message).collect::<Vec<_>>();
        let body = serde_json::json!({
            "model": model_id,
            "messages": api_messages,
            "temperature": temperature,
            "stream": true,
            "stream_options": {"include_usage": true},
            "tools": build_tools(tools),
        });

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
        if resp.status().is_client_error() {
            return Err(LlmError::Request(format!("HTTP {}", resp.status())));
        }

        let mut text = String::new();
        let mut prompt_tokens: u64 = 0;
        let mut completion_tokens: u64 = 0;
        // Accumulate tool calls by their stream index.
        let mut tool_calls: std::collections::BTreeMap<usize, ToolCall> =
            std::collections::BTreeMap::new();

        let mut stream = resp.bytes_stream();
        let mut buf = Vec::<u8>::new();
        while let Some(chunk) = stream.next().await {
            let chunk = match chunk {
                Ok(c) => c,
                Err(e) => return Err(LlmError::Request(format!("stream error : {e}"))),
            };
            buf.extend_from_slice(&chunk);
            while let Some(pos) = buf.windows(2).position(|w| w == b"\n\n") {
                let event: Vec<u8> = buf.drain(..=pos).collect();
                let text_evt = String::from_utf8_lossy(&event);
                for line in text_evt.lines() {
                    if let Some(data) = line.strip_prefix("data:") {
                        let payload = data.trim();
                        if payload.is_empty() || payload == "[DONE]" {
                            continue;
                        }
                        let v: serde_json::Value = match serde_json::from_str(payload) {
                            Ok(v) => v,
                            Err(_) => continue,
                        };
                        if let Some(u) = v.get("usage") {
                            prompt_tokens = u
                                .get("prompt_tokens")
                                .and_then(|x| x.as_u64())
                                .unwrap_or(0);
                            completion_tokens = u
                                .get("completion_tokens")
                                .and_then(|x| x.as_u64())
                                .unwrap_or(0);
                        }
                        let Some(choice) = v
                            .get("choices")
                            .and_then(|c| c.as_array())
                            .and_then(|a| a.first())
                        else {
                            continue;
                        };
                        let Some(delta) = choice.get("delta") else {
                            continue;
                        };
                        if let Some(t) = delta.get("content").and_then(|x| x.as_str()) {
                            if t.is_empty() {
                                continue;
                            }
                            text.push_str(t);
                            sink(StreamEvent::Text(t.to_string()));
                        }
                        if let Some(tcs) = delta.get("tool_calls").and_then(|x| x.as_array()) {
                            for tc in tcs {
                                let idx = tc.get("index").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
                                let entry = tool_calls
                                    .entry(idx)
                                    .or_insert_with(|| ToolCall {
                                        id: String::new(),
                                        name: String::new(),
                                        arguments: String::new(),
                                    });
                                if let Some(id) = tc.get("id").and_then(|x| x.as_str()) {
                                    entry.id = id.to_string();
                                }
                                if let Some(f) = tc.get("function") {
                                    if let Some(n) = f.get("name").and_then(|x| x.as_str()) {
                                        entry.name = n.to_string();
                                    }
                                    if let Some(a) = f.get("arguments").and_then(|x| x.as_str()) {
                                        entry.arguments.push_str(a);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let tool_calls: Vec<ToolCall> = tool_calls.into_values().collect();
        for tc in &tool_calls {
            sink(StreamEvent::ToolCall(tc.clone()));
        }
        Ok(AgentResponse {
            content: text,
            usage: TokenUsage::new(prompt_tokens, completion_tokens),
            tool_calls,
        })
    }
}
