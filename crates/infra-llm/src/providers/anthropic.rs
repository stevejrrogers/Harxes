use async_trait::async_trait;
use harxes_core_domain::domain::value_objects::{
    Message, ProviderId, Role, ToolCall, ToolSpec,
};
use harxes_core_domain::ports::{AgentResponse, LlmError, LlmPort, StreamSink};
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
        // Anthropic: the first System message maps to the dedicated `system`
        // field; any later System messages (e.g. a compressed context summary)
        // are kept in the message list as user-role content so they are not lost.
        let mut saw_system = false;
        let api_messages: Vec<_> = messages
            .iter()
            .filter(|m| {
                if m.role == Role::System {
                    if !saw_system {
                        saw_system = true;
                        return false;
                    }
                }
                true
            })
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

    async fn generate_stream(
        &self,
        _provider: &ProviderId,
        model_id: &str,
        messages: &[Message],
        _tools: &[ToolSpec],
        temperature: Option<f64>,
        sink: StreamSink,
    ) -> Result<AgentResponse, LlmError> {
        use futures_util::StreamExt;
        use harxes_core_domain::ports::StreamEvent;

        let system = messages
            .iter()
            .find(|m| m.role == Role::System)
            .map(|m| m.content.as_str());
        let mut saw_system = false;
        let api_messages: Vec<_> = messages
            .iter()
            .filter(|m| {
                if m.role == Role::System {
                    if !saw_system {
                        saw_system = true;
                        return false;
                    }
                }
                true
            })
            .map(ApiMessage::from)
            .collect();

        let body = serde_json::json!({
            "model": model_id,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "stream": true,
            "system": system,
            "messages": api_messages
                .iter()
                .map(|m| serde_json::json!({"role": m.role, "content": m.content}))
                .collect::<Vec<_>>(),
            "temperature": temperature,
        });

        let resp = match self
            .http
            .post(self.url())
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => return Err(LlmError::Request(format!("transport error: {e}"))),
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
        let mut input_tokens: u64 = 0;
        let mut output_tokens: u64 = 0;

        let mut stream = resp.bytes_stream();
        let mut buf = Vec::<u8>::new();
        let mut tool_blocks: Vec<(String, String, String)> = Vec::new(); // (id, name, json)
        let mut current_tool: Option<(String, String, String)> = None;

        while let Some(chunk) = stream.next().await {
            let chunk = match chunk {
                Ok(c) => c,
                Err(e) => return Err(LlmError::Request(format!("stream error: {e}"))),
            };
            buf.extend_from_slice(&chunk);
            // Process complete SSE events separated by blank lines.
            while let Some(pos) = buf.windows(2).position(|w| w == b"\n\n") {
                let event_bytes: Vec<u8> = buf.drain(..=pos).collect();
                let event_str = String::from_utf8_lossy(&event_bytes);
                for line in event_str.lines() {
                    if let Some(data) = line.strip_prefix("data:") {
                        let payload = data.trim();
                        if payload.is_empty() || payload == "[DONE]" {
                            continue;
                        }
                        let v: serde_json::Value = match serde_json::from_str(payload) {
                            Ok(v) => v,
                            Err(_) => continue,
                        };
                        let kind = v.get("type").and_then(|k| k.as_str()).unwrap_or("");
                        match kind {
                            "message_start" => {
                                input_tokens = v
                                    .pointer("/message/usage/input_tokens")
                                    .and_then(|x| x.as_u64())
                                    .unwrap_or(0);
                            }
                            "content_block_start" => {
                                if let Some(cb) = v.get("content_block") {
                                    if cb.get("type").and_then(|t| t.as_str()) == Some("tool_use")
                                    {
                                        let id = cb
                                            .get("id")
                                            .and_then(|x| x.as_str())
                                            .unwrap_or("")
                                            .to_string();
                                        let name = cb
                                            .get("name")
                                            .and_then(|x| x.as_str())
                                            .unwrap_or("")
                                            .to_string();
                                        current_tool = Some((id, name, String::new()));
                                    }
                                }
                            }
                            "content_block_delta" => {
                                if let Some(delta) = v.get("delta") {
                                    match delta.get("type").and_then(|t| t.as_str()) {
                                        Some("text_delta") => {
                                            if let Some(t) =
                                                delta.get("text").and_then(|x| x.as_str())
                                            {
                                                text.push_str(t);
                                                sink(StreamEvent::Text(t.to_string()));
                                            }
                                        }
                                        Some("input_json_delta") => {
                                            if let Some(pj) = delta
                                                .get("partial_json")
                                                .and_then(|x| x.as_str())
                                            {
                                                if let Some(t) = &mut current_tool {
                                                    t.2.push_str(pj);
                                                }
                                            }
                                        }
                                        _ => {}
                                    }
                                }
                            }
                            "content_block_stop" => {
                                if let Some(t) = current_tool.take() {
                                    let call = ToolCall {
                                        id: t.0,
                                        name: t.1,
                                        arguments: t.2,
                                    };
                                    sink(StreamEvent::ToolCall(call.clone()));
                                    tool_blocks.push((call.id, call.name, call.arguments));
                                }
                            }
                            "message_delta" => {
                                output_tokens = v
                                    .pointer("/usage/output_tokens")
                                    .and_then(|x| x.as_u64())
                                    .unwrap_or(0);
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        // Buffer may hold a trailing event without a blank line.
        if !buf.is_empty() {
            let event_str = String::from_utf8_lossy(&buf);
            for line in event_str.lines() {
                if let Some(data) = line.strip_prefix("data:") {
                    let payload = data.trim();
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(payload) {
                        if let Some(delta) = v.get("delta") {
                            if delta.get("type").and_then(|t| t.as_str()) == Some("text_delta") {
                                if let Some(t) = delta.get("text").and_then(|x| x.as_str()) {
                                    text.push_str(t);
                                }
                            }
                        }
                        if let Some(u) = v.get("usage") {
                            if let Some(ot) = u.get("output_tokens").and_then(|x| x.as_u64()) {
                                output_tokens = ot;
                            }
                        }
                    }
                }
            }
        }

        let tool_calls = tool_blocks
            .into_iter()
            .map(|(id, name, arguments)| ToolCall { id, name, arguments })
            .collect();
        Ok(AgentResponse {
            content: text,
            usage: harxes_core_domain::domain::value_objects::TokenUsage::new(
                input_tokens,
                output_tokens,
            ),
            tool_calls,
        })
    }
}
