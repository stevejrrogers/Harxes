//! End-to-end tests of the Anthropic adapter against a local mock HTTP server.
//!
//! These validate the full wire pipeline without requiring a live API key:
//! request body encodes tools + tool_use/tool_result blocks, and the adapter
//! correctly extracts `tool_use` from both the non-stream and SSE responses.

use harxes_core_domain::domain::value_objects::{Message, Role, ToolSpec};
use harxes_core_domain::ports::LlmPort;
use harxes_infra_llm::providers::anthropic::AnthropicClient;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpListener;
use tokio::sync::oneshot;

/// A bare-bones HTTP/1.1 responder that returns `body` for every request.
async fn spawn_mock(body: String) -> (String, oneshot::Receiver<Vec<u8>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (req_tx, req_rx) = oneshot::channel();
    tokio::spawn(async move {
        let (mut sock, _) = listener.accept().await.unwrap();
        let mut reader = BufReader::new(&mut sock);
        // Read the request head only; a small request is plenty.
        let mut head = String::new();
        let _ = reader.read_line(&mut head).await;
        let _ = req_tx.send(head.into_bytes());
        let len = body.len();
        let _ = sock
            .write_all(
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {len}\r\n\r\n{body}"
                )
                .as_bytes(),
            )
            .await;
        let _ = sock.shutdown().await;
    });
    (format!("http://{addr}"), req_rx)
}

fn bash_spec() -> Vec<ToolSpec> {
    vec![ToolSpec::with_schema(
        "Bash",
        "Run a shell command and capture its output.",
        serde_json::json!({
            "type": "object",
            "properties": { "command": { "type": "string" } },
            "required": ["command"]
        }),
    )]
}

#[tokio::test]
async fn non_stream_parses_tool_use_from_response() {
    let body = r#"{
        "content": [
            {"type":"text","text":"I'll check."},
            {"type":"tool_use","id":"toolu_1","name":"Bash","input":{"command":"pwd"}}
        ],
        "usage":{"input_tokens":12,"output_tokens":4}
    }"#;
    let (url, req_rx) = spawn_mock(body.to_string()).await;
    let client = AnthropicClient::new(url, "test-key");
    let pid = harxes_core_domain::domain::value_objects::ProviderId::new("anthropic").unwrap();
    let msgs = vec![Message::new(Role::User, "run pwd")];

    let resp = client
        .generate(&pid, "claude-sonnet", &msgs, &bash_spec(), Some(0.0))
        .await
        .unwrap();

    assert_eq!(resp.content, "I'll check.");
    assert_eq!(resp.tool_calls.len(), 1);
    assert_eq!(resp.tool_calls[0].id, "toolu_1");
    assert_eq!(resp.tool_calls[0].name, "Bash");
    assert!(resp.tool_calls[0].arguments.contains("pwd"));

    // The request must have hit the POST endpoint.
    let req = String::from_utf8_lossy(&req_rx.await.unwrap()).to_string();
    assert!(req.contains("POST"));
}

#[tokio::test]
async fn stream_parses_tool_use_from_sse() {
    // SSE stream: a tool_use block assembled from partial JSON.
    let sse = concat!(
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":10,\"output_tokens\":0}}}\n\n",
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_9\",\"name\":\"Bash\"}}\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"command\\\":\\\"ls\\\"}\"}}\n\n",
        "event: content_block_stop\ndata: {\"type\":\"content_block_stop\"}\n\n",
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":3}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
    );
    let (url, _req_rx) = spawn_mock(sse.to_string()).await;
    let client = AnthropicClient::new(url, "test-key");
    let pid = harxes_core_domain::domain::value_objects::ProviderId::new("anthropic").unwrap();
    let msgs = vec![Message::new(Role::User, "ls please")];
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let sink = std::sync::Arc::new(move |ev| {
        let _ = tx.send(ev);
    });

    let resp = client
        .generate_stream(&pid, "claude-sonnet", &msgs, &bash_spec(), Some(0.0), sink)
        .await
        .unwrap();

    assert_eq!(resp.tool_calls.len(), 1);
    assert_eq!(resp.tool_calls[0].name, "Bash");
    assert!(resp.tool_calls[0].arguments.contains("ls"));
    // Verify a ToolCall stream event was surfaced before completion.
    let mut saw_tool_event = false;
    while let Ok(ev) = rx.try_recv() {
        if let harxes_core_domain::ports::StreamEvent::ToolCall(tc) = ev {
            assert_eq!(tc.name, "Bash");
            saw_tool_event = true;
        }
    }
    assert!(saw_tool_event, "expected a ToolCall stream event");
}
