use async_trait::async_trait;
use harxes_core_domain::domain::value_objects::{
    Message, ProviderId, Role, ToolCall, ToolSpec,
};
use harxes_core_domain::ports::{AgentResponse, LlmError, LlmPort, StreamSink};
use serde::Serialize;

#[derive(Serialize)]
struct RequestBody<'a> {
    model: &'a str,
    max_tokens: u32,
    system: Option<&'a str>,
    messages: serde_json::Value,
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<serde_json::Value>,
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
    /// Present for `tool_use` blocks.
    id: Option<String>,
    name: Option<String>,
    input: Option<serde_json::Value>,
}

#[derive(serde::Deserialize)]
struct Usage {
    input_tokens: u64,
    output_tokens: u64,
}

const DEFAULT_MAX_TOKENS: u32 = 4096;

/// Encode the transcript into the Anthropic `messages` payload. The first
/// System message is carried in the dedicated `system` field and skipped here;
/// later System messages (e.g. a compressed context summary) are kept as
/// user-role content so they are never dropped. Assistant tool requests become
/// `tool_use` content blocks and tool results become `tool_result` blocks.
fn encode_messages(messages: &[Message]) -> serde_json::Value {
    let mut saw_system = false;
    let arr: Vec<serde_json::Value> = messages
        .iter()
        .filter(|m| {
            if m.role == Role::System && !saw_system {
                saw_system = true;
                return false;
            }
            true
        })
        .map(|m| {
            let role = match m.role {
                Role::Assistant => "assistant",
                _ => "user",
            };
            let content = match m.role {
                Role::Assistant if !m.tool_calls.is_empty() => {
                    let mut blocks: Vec<serde_json::Value> = Vec::new();
                    if !m.content.is_empty() {
                        blocks.push(serde_json::json!({ "type": "text", "text": m.content }));
                    }
                    for tc in &m.tool_calls {
                        let input = serde_json::from_str(&tc.arguments)
                            .unwrap_or_else(|_| serde_json::json!({}));
                        blocks.push(serde_json::json!({
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": input,
                        }));
                    }
                    serde_json::Value::Array(blocks)
                }
                Role::Tool => serde_json::json!([{
                    "type": "tool_result",
                    "tool_use_id": m.tool_call_id.clone().unwrap_or_default(),
                    "content": m.content,
                }]),
                _ => serde_json::Value::String(m.content.clone()),
            };
            serde_json::json!({ "role": role, "content": content })
        })
        .collect();
    serde_json::Value::Array(arr)
}

/// Encode tool declarations into the Anthropic `tools` payload.
fn encode_tools(tools: &[ToolSpec]) -> Vec<serde_json::Value> {
    tools
        .iter()
        .map(|t| {
            serde_json::json!({
                "name": t.id,
                "description": t.description,
                "input_schema": if t.input_schema.is_null() {
                    serde_json::json!({ "type": "object" })
                } else {
                    t.input_schema.clone()
                },
            })
        })
        .collect()
}

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
        tools: &[ToolSpec],
        temperature: Option<f64>,
    ) -> Result<AgentResponse, LlmError> {
        let system = messages
            .iter()
            .find(|m| m.role == Role::System)
            .map(|m| m.content.as_str());
        let body = RequestBody {
            model: model_id,
            max_tokens: DEFAULT_MAX_TOKENS,
            system,
            messages: encode_messages(messages),
            temperature,
            tools: encode_tools(tools),
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
            .iter()
            .filter(|c| c.kind == "text")
            .filter_map(|c| c.text.as_ref())
            .cloned()
            .collect::<Vec<_>>()
            .join("\n");

        let tool_calls: Vec<ToolCall> = rb
            .content
            .iter()
            .filter(|c| c.kind == "tool_use")
            .filter_map(|c| {
                let id = c.id.clone()?;
                let name = c.name.clone()?;
                let input = c.input.clone().unwrap_or_else(|| serde_json::json!({}));
                Some(ToolCall {
                    id,
                    name,
                    arguments: input.to_string(),
                })
            })
            .collect();

        Ok(AgentResponse {
            content: text,
            usage: harxes_core_domain::domain::value_objects::TokenUsage::new(
                rb.usage.input_tokens,
                rb.usage.output_tokens,
            ),
            tool_calls,
        })
    }

    async fn generate_stream(
        &self,
        _provider: &ProviderId,
        model_id: &str,
        messages: &[Message],
        tools: &[ToolSpec],
        temperature: Option<f64>,
        sink: StreamSink,
    ) -> Result<AgentResponse, LlmError> {
        use futures_util::StreamExt;
        use harxes_core_domain::ports::StreamEvent;

        let system = messages
            .iter()
            .find(|m| m.role == Role::System)
            .map(|m| m.content.as_str());

        let body = serde_json::json!({
            "model": model_id,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "stream": true,
            "system": system,
            "messages": encode_messages(messages),
            "tools": encode_tools(tools),
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

#[cfg(test)]
mod tests {
    use super::*;
    use harxes_core_domain::domain::value_objects::ToolCall;
    use serde_json::json;

    fn simple_tools() -> Vec<ToolSpec> {
        vec![ToolSpec::with_schema(
            "Bash",
            "Run a shell command.",
            serde_json::json!({
                "type": "object",
                "properties": { "command": { "type": "string" } },
                "required": ["command"]
            }),
        )]
    }

    #[test]
    fn encode_tools_includes_name_description_and_schema() {
        let tools = encode_tools(&simple_tools());
        assert_eq!(tools.len(), 1);
        let t = &tools[0];
        assert_eq!(t["name"], "Bash");
        assert_eq!(t["description"], "Run a shell command.");
        assert_eq!(t["input_schema"]["required"][0], "command");
    }

    #[test]
    fn encode_messages_emits_tool_use_block_for_assistant() {
        let call = ToolCall {
            id: "toolu_1".into(),
            name: "Bash".into(),
            arguments: json!({"command": "ls"}).to_string(),
        };
        let m = Message::assistant_with_tools(vec![call]);
        let encoded = encode_messages(&[m]);
        let arr = encoded.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["role"], "assistant");
        let blocks = arr[0]["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0]["type"], "tool_use");
        assert_eq!(blocks[0]["id"], "toolu_1");
        assert_eq!(blocks[0]["name"], "Bash");
        assert_eq!(blocks[0]["input"]["command"], "ls");
    }

    #[test]
    fn encode_messages_emits_tool_result_block_for_tool_role() {
        let m = Message::tool_result("toolu_1", "ok");
        let encoded = encode_messages(&[m]);
        let arr = encoded.as_array().unwrap();
        assert_eq!(arr[0]["role"], "user");
        let blocks = arr[0]["content"].as_array().unwrap();
        assert_eq!(blocks[0]["type"], "tool_result");
        assert_eq!(blocks[0]["tool_use_id"], "toolu_1");
        assert_eq!(blocks[0]["content"], "ok");
    }

    #[test]
    fn first_system_message_goes_to_system_field_and_is_skipped() {
        let sys = Message::new(Role::System, "you are the agent");
        let user = Message::new(Role::User, "hi");
        let encoded = encode_messages(&[sys, user]);
        let arr = encoded.as_array().unwrap();
        // Only the user message remains; the first System was lifted to `system`.
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["role"], "user");
    }
}
