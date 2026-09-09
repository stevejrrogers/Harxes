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

/// Break a long run of reasoning that arrives without newlines into readable
/// segments: once the current line exceeds ~300 chars, insert a newline at the
/// next sentence boundary (or hard-wrap at ~500 chars). Keeps the token-delta
/// stream but stops a consumer from coalescing one giant line. `since_nl` is
/// the char count on the current line so far; returns (emitted, new_since_nl).
fn segment_reasoning(delta: &str, mut since_nl: usize) -> (String, usize) {
    let mut out = String::with_capacity(delta.len() + 2);
    for ch in delta.chars() {
        if ch == '\n' {
            out.push(ch);
            since_nl = 0;
            continue;
        }
        out.push(ch);
        since_nl += 1;
        if since_nl >= 300 && matches!(ch, '.' | '!' | '?') {
            out.push('\n');
            since_nl = 0;
        } else if since_nl >= 500 && ch == ' ' {
            out.pop();
            out.push('\n');
            since_nl = 0;
        }
    }
    (out, since_nl)
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
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<&'static str>,
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
    #[serde(default)]
    completion_tokens_details: Option<CompletionDetails>,
}

#[derive(serde::Deserialize)]
struct CompletionDetails {
    #[serde(default)]
    reasoning_tokens: Option<u64>,
}

/// How requests authenticate. `Static` is a fixed bearer (OpenAI, LiteLLM);
/// `Copilot` exchanges a GitHub OAuth token for a short-lived Copilot bearer
/// and refreshes it on expiry.
enum Auth {
    Static(String),
    Copilot(std::sync::Arc<CopilotAuth>),
}

/// GitHub Copilot token exchange: swaps a long-lived GitHub OAuth token for a
/// short-lived (~25 min) Copilot bearer, cached until just before it expires.
struct CopilotAuth {
    github_token: String,
    // (bearer, expires_at_unix_secs)
    cached: tokio::sync::Mutex<Option<(String, u64)>>,
}

impl CopilotAuth {
    async fn bearer(&self, http: &reqwest::Client) -> Result<String, LlmError> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        {
            let guard = self.cached.lock().await;
            if let Some((tok, exp)) = guard.as_ref() {
                if *exp > now + 60 {
                    return Ok(tok.clone());
                }
            }
        }
        // Refresh via the Copilot token endpoint.
        let resp = http
            .get("https://api.github.com/copilot_internal/v2/token")
            .header("Authorization", format!("token {}", self.github_token))
            .header("Editor-Version", "vscode/1.95.0")
            .header("User-Agent", "GitHubCopilotChat/0.22.0")
            .send()
            .await
            .map_err(|e| LlmError::Request(format!("copilot token exchange: {e}")))?;
        if resp.status() == reqwest::StatusCode::UNAUTHORIZED
            || resp.status() == reqwest::StatusCode::FORBIDDEN
        {
            return Err(LlmError::Auth {
                provider: ProviderId::new("copilot").unwrap(),
            });
        }
        if !resp.status().is_success() {
            return Err(LlmError::Request(format!(
                "copilot token exchange HTTP {}",
                resp.status()
            )));
        }
        let v: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| LlmError::Request(format!("copilot token decode: {e}")))?;
        let token = v
            .get("token")
            .and_then(|x| x.as_str())
            .ok_or_else(|| LlmError::Request("copilot token: no `token` field".into()))?
            .to_string();
        let exp = v.get("expires_at").and_then(|x| x.as_u64()).unwrap_or(now + 1500);
        *self.cached.lock().await = Some((token.clone(), exp));
        Ok(token)
    }
}

/// OpenAI-compatible Chat Completions adapter with native tool-calling. Also
/// backs LiteLLM proxies and (via [`OpenAiClient::copilot`]) GitHub Copilot.
pub struct OpenAiClient {
    http: reqwest::Client,
    base_url: String,
    auth: Auth,
    /// Extra headers sent on every chat request (Copilot integration headers).
    extra_headers: Vec<(&'static str, &'static str)>,
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
            http: Self::build_http(),
            base_url: url,
            auth: Auth::Static(api_key.into()),
            extra_headers: Vec::new(),
        }
    }

    /// A GitHub Copilot client: talks to the Copilot chat/completions endpoint
    /// (OpenAI-compatible body) using a GitHub OAuth token, which it exchanges
    /// for short-lived Copilot bearers and refreshes automatically.
    pub fn copilot(github_token: impl Into<String>) -> Self {
        Self {
            http: Self::build_http(),
            base_url: "https://api.githubcopilot.com/chat/completions".to_string(),
            auth: Auth::Copilot(std::sync::Arc::new(CopilotAuth {
                github_token: github_token.into(),
                cached: tokio::sync::Mutex::new(None),
            })),
            extra_headers: vec![
                ("Copilot-Integration-Id", "vscode-chat"),
                ("Editor-Version", "vscode/1.95.0"),
                ("Editor-Plugin-Version", "copilot-chat/0.22.0"),
            ],
        }
    }

    fn build_http() -> reqwest::Client {
        reqwest::Client::builder()
            .user_agent(concat!("harxes/", env!("CARGO_PKG_VERSION")))
            .connect_timeout(std::time::Duration::from_secs(20))
            .read_timeout(std::time::Duration::from_secs(120))
            .build()
            .unwrap_or_default()
    }

    /// Resolve the current bearer (refreshing a Copilot token if needed) and
    /// apply auth + integration headers to a request builder.
    async fn authorize(
        &self,
        req: reqwest::RequestBuilder,
    ) -> Result<reqwest::RequestBuilder, LlmError> {
        let bearer = match &self.auth {
            Auth::Static(k) => k.clone(),
            Auth::Copilot(c) => c.bearer(&self.http).await?,
        };
        let mut req = req.bearer_auth(bearer);
        for (k, v) in &self.extra_headers {
            req = req.header(*k, *v);
        }
        Ok(req)
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
        let req = self.authorize(self.http.get(&url)).await?;
        let resp = req
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
        reasoning_effort: Option<harxes_core_domain::domain::value_objects::ReasoningEffort>,
    ) -> Result<AgentResponse, LlmError> {
        let api_messages = messages.iter().map(to_api_message).collect();
        let body = RequestBody {
            model: model_id,
            messages: api_messages,
            temperature,
            reasoning_effort: reasoning_effort.map(|e| e.as_str()),
            tools: build_tools(tools),
        };
        let req = self.authorize(self.http.post(self.url()).json(&body)).await?;
        let resp = match req.send().await {
            Ok(r) => r,
            Err(e) => return Err(LlmError::Request(format!("transport error : {e}"))),
        };
        if resp.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
            return Err(LlmError::RateLimited {
                retry_after: retry_after_secs(&resp),
            });
        }
        if resp.status().is_server_error() {
            // 5xx: transient but the provider may be down — fail fast so the
            // caller can fail over, rather than waiting out a rate-limit budget.
            return Err(LlmError::Request(format!("HTTP {}", resp.status())));
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
                    usage: TokenUsage::new(rb.usage.prompt_tokens, rb.usage.completion_tokens)
                        .with_reasoning(
                            rb.usage
                                .completion_tokens_details
                                .and_then(|d| d.reasoning_tokens),
                        ),
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
        reasoning_effort: Option<harxes_core_domain::domain::value_objects::ReasoningEffort>,
        sink: StreamSink,
    ) -> Result<AgentResponse, LlmError> {
        use futures_util::StreamExt;
        use harxes_core_domain::ports::StreamEvent;

        let api_messages = messages.iter().map(to_api_message).collect::<Vec<_>>();
        let mut body = serde_json::json!({
            "model": model_id,
            "messages": api_messages,
            "temperature": temperature,
            "stream": true,
            "stream_options": {"include_usage": true},
            "tools": build_tools(tools),
        });
        if let Some(e) = reasoning_effort {
            body["reasoning_effort"] = serde_json::json!(e.as_str());
        }

        let req = self.authorize(self.http.post(self.url()).json(&body)).await?;
        let resp = match req.send().await {
            Ok(r) => r,
            Err(e) => return Err(LlmError::Request(format!("transport error : {e}"))),
        };
        if resp.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
            return Err(LlmError::RateLimited {
                retry_after: retry_after_secs(&resp),
            });
        }
        if resp.status().is_server_error() {
            return Err(LlmError::Request(format!("HTTP {}", resp.status())));
        }
        if resp.status().is_client_error() {
            return Err(LlmError::Request(format!("HTTP {}", resp.status())));
        }

        let mut text = String::new();
        let mut reasoning = String::new();
        let mut prompt_tokens: u64 = 0;
        let mut completion_tokens: u64 = 0;
        let mut reasoning_tokens: Option<u64> = None;
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
                            if let Some(rt) = u
                                .get("completion_tokens_details")
                                .and_then(|d| d.get("reasoning_tokens"))
                                .and_then(|x| x.as_u64())
                            {
                                reasoning_tokens = Some(rt);
                            }
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
                                let since = reasoning.len()
                                    - reasoning.rfind('\n').map(|i| i + 1).unwrap_or(0);
                                let (seg, _) = segment_reasoning(t, since);
                                reasoning.push_str(&seg);
                                sink(StreamEvent::Reasoning(seg));
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
            usage: TokenUsage::new(prompt_tokens, completion_tokens)
                .with_reasoning(reasoning_tokens),
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
    use super::{segment_reasoning, OpenAiClient, RequestBody};

    #[test]
    fn request_body_includes_reasoning_effort() {
        use harxes_core_domain::domain::value_objects::ReasoningEffort;
        let body = RequestBody {
            model: "m",
            messages: vec![],
            temperature: None,
            reasoning_effort: Some(ReasoningEffort::Low).map(|e| e.as_str()),
            tools: vec![],
        };
        let v = serde_json::to_value(&body).unwrap();
        assert_eq!(v["reasoning_effort"], "low");
        // Absent when None.
        let body2 = RequestBody { model: "m", messages: vec![], temperature: None, reasoning_effort: None, tools: vec![] };
        assert!(serde_json::to_value(&body2).unwrap().get("reasoning_effort").is_none());
    }

    #[test]
    fn reasoning_segmentation_breaks_long_runs() {
        // A ~400-char run with sentence boundaries and no newline gets broken.
        let long = "This is a sentence. ".repeat(25); // 500 chars, boundaries
        let (out, _) = segment_reasoning(&long, 0);
        assert!(out.contains('\n'), "expected a break inserted");
        // Every emitted line stays under the hard-wrap ceiling.
        assert!(out.lines().all(|l| l.chars().count() <= 520));
        // Short input passes through unchanged.
        let (short, n) = segment_reasoning("brief thought", 0);
        assert_eq!(short, "brief thought");
        assert_eq!(n, 13);
    }

    #[test]
    fn copilot_client_targets_copilot_endpoint() {
        let c = OpenAiClient::copilot("gho_faketoken");
        assert_eq!(c.url(), "https://api.githubcopilot.com/chat/completions");
        assert!(matches!(c.auth, super::Auth::Copilot(_)));
        assert!(c
            .extra_headers
            .iter()
            .any(|(k, v)| *k == "Copilot-Integration-Id" && *v == "vscode-chat"));
    }

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
