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
    content: Option<serde_json::Value>,
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
            content: Some(serde_json::Value::String(m.content.clone())),
            tool_calls: vec![],
            tool_call_id: None,
        },
        Role::User => ApiMessage {
            role: "user",
            content: Some(if m.images.is_empty() {
                serde_json::Value::String(m.content.clone())
            } else {
                // Multimodal: text part plus data-URI image parts.
                let mut parts =
                    vec![serde_json::json!({"type": "text", "text": m.content})];
                for img in &m.images {
                    parts.push(serde_json::json!({
                        "type": "image_url",
                        "image_url": {"url": format!("data:{};base64,{}", img.media_type, img.base64)}
                    }));
                }
                serde_json::Value::Array(parts)
            }),
            tool_calls: vec![],
            tool_call_id: None,
        },
        Role::Tool => ApiMessage {
            role: "tool",
            content: Some(serde_json::Value::String(m.content.clone())),
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
                Some(serde_json::Value::String(m.content.clone()))
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
            content: Some(serde_json::Value::String(m.content.clone())),
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
                  parameters : if t.input_schema.is_null() {
                    serde_json::json!({"type":"object","properties":{},"additionalProperties":true})
                  } else {
                    t.input_schema.clone()
                  },
            },
      }).collect()
}

/// Parse the `Retry-After` header (seconds form) into a value for backoff.
fn retry_after_secs(resp: &reqwest::Response) -> Option<u64> {
    resp.headers()
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
}

/// Consume a failed response into a readable error, surfacing the API's own
/// error message (JSON `error.message` or raw body, capped) so the user sees
/// *why* — e.g. "context length exceeded", "model not found".
async fn http_error(resp: reqwest::Response, hint: &str) -> LlmError {
    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    let detail = serde_json::from_str::<serde_json::Value>(&body)
        .ok()
        .and_then(|v| {
            v.get("error")
                .and_then(|e| e.get("message"))
                .and_then(|m| m.as_str())
                .map(|s| s.to_string())
                .or_else(|| v.get("message").and_then(|m| m.as_str()).map(|s| s.to_string()))
        })
        .unwrap_or_else(|| body.chars().take(300).collect());
    let detail = detail.trim();
    if detail.is_empty() {
        LlmError::Request(format!("HTTP {status}{hint}"))
    } else {
        LlmError::Request(format!("HTTP {status}{hint}: {detail}"))
    }
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
    /// Reasoning-model channel (DeepSeek, GLM, o-series via proxies). Only
    /// used as a fallback when `content` comes back empty.
    #[serde(default)]
    reasoning_content: Option<String>,
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
        // `base_url` is the FULL endpoint URL — except the common case of an
        // OpenAI-compatible API root (`.../v1` or a bare origin), which gets
        // `/chat/completions` appended so users can paste the base URL a
        // proxy like LiteLLM advertises.
        let mut url = base_url.into().trim_end_matches('/').to_string();
        let bare_origin = url
            .split_once("://")
            .map(|(_, rest)| !rest.contains('/'))
            .unwrap_or(false);
        if url.ends_with("/v1") || bare_origin {
            url.push_str("/chat/completions");
        }
        Self {
            http: reqwest::Client::builder()
                .connect_timeout(std::time::Duration::from_secs(20))
                .read_timeout(std::time::Duration::from_secs(120))
                .build()
                .unwrap_or_default(),
            base_url: url,
            api_key: api_key.into(),
        }
    }
    fn url(&self) -> String {
        self.base_url.clone()
    }
}

#[async_trait]
impl LlmPort for OpenAiClient {
    async fn list_models(&self) -> Result<Vec<String>, LlmError> {
        // Derive the /models endpoint from the chat completions URL.
        let url = self
            .base_url
            .trim_end_matches("/chat/completions")
            .trim_end_matches('/')
            .to_string()
            + "/models";
        let resp = self
            .http
            .get(&url)
            .bearer_auth(&self.api_key)
            .send()
            .await
            .map_err(|e| LlmError::Request(format!("transport error : {e}")))?;
        if !resp.status().is_success() {
            return Err(LlmError::Request(format!("HTTP {}", resp.status())));
        }
        let v: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| LlmError::Request(format!("decode error : {e}")))?;
        Ok(v
            .get("data")
            .and_then(|d| d.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|m| m.get("id").and_then(|x| x.as_str()))
                    .map(|s| s.to_string())
                    .collect()
            })
            .unwrap_or_default())
    }

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
            return Err(LlmError::RateLimited {
                retry_after: retry_after_secs(&resp),
            });
        }
        if resp.status() == reqwest::StatusCode::UNAUTHORIZED
            || resp.status() == reqwest::StatusCode::FORBIDDEN
        {
            return Err(LlmError::Auth {
                provider: provider.clone(),
            });
        }
        if !resp.status().is_success() {
            let hint = if resp.status() == reqwest::StatusCode::NOT_FOUND {
                format!(" at {} — check that --base-url points at an OpenAI-compatible chat completions endpoint", self.url())
            } else {
                String::new()
            };
            return Err(http_error(resp, &hint).await);
        }
        let rb: ResponseBody = resp
            .json()
            .await
            .map_err(|e| LlmError::Request(format!("decode error : {e}")))?;
        let first = rb.choices.into_iter().next();
        match first {
            Some(choice) => {
                let mut content = choice.message.content.unwrap_or_default();
                if content.trim().is_empty() && choice.message.tool_calls.is_empty() {
                    content = choice.message.reasoning_content.unwrap_or_default();
                }
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
        let mut reasoning = String::new();
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
                        // Reasoning-model channel: accumulate silently so the
                        // turn isn't empty when a model answers only there.
                        if let Some(t) =
                            delta.get("reasoning_content").and_then(|x| x.as_str())
                        {
                            if !t.is_empty() {
                                reasoning.push_str(t);
                                sink(StreamEvent::Reasoning(t.to_string()));
                            }
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
        if text.trim().is_empty() && tool_calls.is_empty() && !reasoning.trim().is_empty() {
            sink(StreamEvent::Text(reasoning.clone()));
            text = reasoning;
        }
        Ok(AgentResponse {
            content: text,
            usage: TokenUsage::new(prompt_tokens, completion_tokens),
            tool_calls,
        })
    }
}

#[cfg(test)]
mod vision_tests {
    use super::*;
    use harxes_core_domain::domain::value_objects::ImageData;

    #[test]
    fn user_message_with_images_becomes_multimodal_parts() {
        let m = Message::user_with_images(
            "what is this?",
            vec![ImageData { media_type: "image/png".into(), base64: "QUJD".into() }],
        );
        let api = to_api_message(&m);
        let v = serde_json::to_value(&api).unwrap();
        let parts = v["content"].as_array().expect("array content");
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[1]["type"], "image_url");
        assert_eq!(parts[1]["image_url"]["url"], "data:image/png;base64,QUJD");
        // Plain user message keeps string content.
        let msg = Message::new(Role::User, "hi");
        let plain = to_api_message(&msg);
        assert!(serde_json::to_value(&plain).unwrap()["content"].is_string());
    }
}

#[cfg(test)]
mod url_tests {
    use super::OpenAiClient;

    #[test]
    fn appends_chat_completions_to_api_roots() {
        let cases = [
            ("https://proxy.example/v1", "https://proxy.example/v1/chat/completions"),
            ("https://proxy.example/v1/", "https://proxy.example/v1/chat/completions"),
            ("https://proxy.example", "https://proxy.example/chat/completions"),
            (
                "https://proxy.example/v1/chat/completions",
                "https://proxy.example/v1/chat/completions",
            ),
            (
                "https://proxy.example/custom/endpoint",
                "https://proxy.example/custom/endpoint",
            ),
        ];
        for (input, want) in cases {
            let c = OpenAiClient::new(input, "k");
            assert_eq!(c.url(), want, "for input {input}");
        }
    }
}
