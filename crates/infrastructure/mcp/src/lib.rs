//! Driven-port adapter (hexagonal infrastructure ring): MCP client over stdio.
//! Each configured server is spawned as a child process speaking JSON-RPC 2.0
//! (one message per line); its tools are exposed to the agent through
//! [`DynamicToolPort`] under names `mcp__<server>__<tool>`.

use std::collections::BTreeMap;
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use harxes_core_domain::domain::value_objects::ToolSpec;
use harxes_core_domain::ports::{DynamicToolPort, McpServerConfig};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout};
use tokio::sync::Mutex;

const PROTOCOL_VERSION: &str = "2024-11-05";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

/// One live stdio MCP server connection.
pub struct McpServer {
    name: String,
    /// stdin/stdout of the child, guarded together so requests serialize.
    io: Mutex<(ChildStdin, BufReader<ChildStdout>)>,
    /// Keep the child alive for the life of the connection.
    _child: Child,
    next_id: AtomicU64,
    tools: Vec<(String, ToolSpec)>,
}

impl McpServer {
    /// Spawn the server, run the MCP initialize handshake and list its tools.
    pub async fn connect(name: &str, cfg: &McpServerConfig) -> Result<Self, String> {
        let mut cmd = tokio::process::Command::new(&cfg.command);
        cmd.args(&cfg.args)
            .envs(cfg.env.iter())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true);
        let mut child = cmd
            .spawn()
            .map_err(|e| format!("mcp '{name}': spawn '{}' failed: {e}", cfg.command))?;
        let stdin = child.stdin.take().ok_or("mcp: no stdin")?;
        let stdout = BufReader::new(child.stdout.take().ok_or("mcp: no stdout")?);
        let mut server = Self {
            name: name.to_string(),
            io: Mutex::new((stdin, stdout)),
            _child: child,
            next_id: AtomicU64::new(1),
            tools: Vec::new(),
        };

        server
            .request(
                "initialize",
                serde_json::json!({
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "harxes", "version": env!("CARGO_PKG_VERSION")},
                }),
            )
            .await?;
        server.notify("notifications/initialized").await?;

        let listed = server.request("tools/list", serde_json::json!({})).await?;
        let mut tools = Vec::new();
        for t in listed
            .get("tools")
            .and_then(|x| x.as_array())
            .cloned()
            .unwrap_or_default()
        {
            let Some(tool_name) = t.get("name").and_then(|x| x.as_str()) else {
                continue;
            };
            let spec = ToolSpec {
                id: format!("mcp__{}__{}", name, tool_name),
                description: t
                    .get("description")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string(),
                input_schema: t
                    .get("inputSchema")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null),
            };
            tools.push((tool_name.to_string(), spec));
        }
        server.tools = tools;
        Ok(server)
    }

    /// The tool specs this server contributes (prefixed names).
    pub fn specs(&self) -> Vec<ToolSpec> {
        self.tools.iter().map(|(_, s)| s.clone()).collect()
    }

    /// Call one of this server's tools by its unprefixed name.
    pub async fn call_tool(&self, tool: &str, arguments: &str) -> String {
        let args: serde_json::Value =
            serde_json::from_str(arguments).unwrap_or(serde_json::json!({}));
        let res = self
            .request(
                "tools/call",
                serde_json::json!({"name": tool, "arguments": args}),
            )
            .await;
        match res {
            Ok(v) => render_tool_result(&v),
            Err(e) => format!("mcp tool error: {e}"),
        }
    }

    async fn notify(&self, method: &str) -> Result<(), String> {
        let msg = serde_json::json!({"jsonrpc": "2.0", "method": method});
        let mut io = self.io.lock().await;
        write_line(&mut io.0, &msg).await
    }

    /// Send a request and read frames until the matching response id arrives
    /// (server-initiated notifications are skipped).
    async fn request(
        &self,
        method: &str,
        params: serde_json::Value,
    ) -> Result<serde_json::Value, String> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let msg = serde_json::json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        let fut = async {
            let mut io = self.io.lock().await;
            write_line(&mut io.0, &msg).await?;
            let mut line = String::new();
            loop {
                line.clear();
                let n = io
                    .1
                    .read_line(&mut line)
                    .await
                    .map_err(|e| format!("read: {e}"))?;
                if n == 0 {
                    return Err("server closed the connection".to_string());
                }
                let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) else {
                    continue;
                };
                if v.get("id").and_then(|x| x.as_u64()) != Some(id) {
                    continue; // notification or unrelated frame
                }
                if let Some(err) = v.get("error") {
                    return Err(
                        err.get("message")
                            .and_then(|m| m.as_str())
                            .unwrap_or("unknown error")
                            .to_string(),
                    );
                }
                return Ok(v.get("result").cloned().unwrap_or(serde_json::Value::Null));
            }
        };
        tokio::time::timeout(REQUEST_TIMEOUT, fut)
            .await
            .map_err(|_| format!("mcp '{}': request '{method}' timed out", self.name))?
    }
}

async fn write_line(stdin: &mut ChildStdin, msg: &serde_json::Value) -> Result<(), String> {
    let mut raw = msg.to_string();
    raw.push('\n');
    stdin
        .write_all(raw.as_bytes())
        .await
        .map_err(|e| format!("write: {e}"))
}

/// Flatten an MCP `tools/call` result into transcript text: text content
/// blocks joined, anything else JSON-encoded.
fn render_tool_result(result: &serde_json::Value) -> String {
    if result
        .get("isError")
        .and_then(|x| x.as_bool())
        .unwrap_or(false)
    {
        // fall through: error details live in content
    }
    let blocks = result.get("content").and_then(|c| c.as_array());
    let Some(blocks) = blocks else {
        return result.to_string();
    };
    let mut out = Vec::new();
    for b in blocks {
        match b.get("type").and_then(|t| t.as_str()) {
            Some("text") => {
                out.push(b.get("text").and_then(|t| t.as_str()).unwrap_or("").to_string())
            }
            _ => out.push(b.to_string()),
        }
    }
    let text = out.join("\n");
    if result
        .get("isError")
        .and_then(|x| x.as_bool())
        .unwrap_or(false)
    {
        format!("mcp tool error: {text}")
    } else {
        text
    }
}

/// All connected MCP servers, presented to the agent as one dynamic tool hub.
pub struct McpToolHub {
    servers: Vec<McpServer>,
}

impl McpToolHub {
    /// Connect every configured server, skipping (and reporting) failures.
    /// Returns the hub plus human-readable status lines.
    pub async fn connect_all(
        configs: &BTreeMap<String, McpServerConfig>,
    ) -> (Self, Vec<String>) {
        let mut servers = Vec::new();
        let mut status = Vec::new();
        for (name, cfg) in configs {
            match McpServer::connect(name, cfg).await {
                Ok(s) => {
                    status.push(format!("mcp '{name}': {} tools", s.tools.len()));
                    servers.push(s);
                }
                Err(e) => status.push(format!("mcp '{name}': FAILED — {e}")),
            }
        }
        (Self { servers }, status)
    }

    pub fn is_empty(&self) -> bool {
        self.servers.is_empty()
    }
}

#[async_trait]
impl DynamicToolPort for McpToolHub {
    fn specs(&self) -> Vec<ToolSpec> {
        self.servers.iter().flat_map(|s| s.specs()).collect()
    }

    fn owns(&self, name: &str) -> bool {
        name.starts_with("mcp__")
    }

    async fn call(&self, name: &str, arguments: &str) -> String {
        // name is mcp__<server>__<tool>
        let Some(rest) = name.strip_prefix("mcp__") else {
            return format!("unknown tool '{name}'");
        };
        let Some((server_name, tool)) = rest.split_once("__") else {
            return format!("unknown tool '{name}'");
        };
        match self.servers.iter().find(|s| s.name == server_name) {
            Some(s) => s.call_tool(tool, arguments).await,
            None => format!("mcp error: no connected server named '{server_name}'"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal MCP server implemented as an inline shell script: replies to
    /// initialize, tools/list (one `echo` tool) and tools/call.
    const FAKE_SERVER: &str = r#"
import sys, json
for line in sys.stdin:
    try:
        msg = json.loads(line)
    except Exception:
        continue
    mid = msg.get("id")
    m = msg.get("method")
    if m == "initialize":
        print(json.dumps({"jsonrpc":"2.0","id":mid,"result":{"protocolVersion":"2024-11-05","capabilities":{},"serverInfo":{"name":"fake","version":"0"}}}), flush=True)
    elif m == "tools/list":
        print(json.dumps({"jsonrpc":"2.0","id":mid,"result":{"tools":[{"name":"echo","description":"Echo back the given text","inputSchema":{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}}]}}), flush=True)
    elif m == "tools/call":
        t = msg["params"]["arguments"].get("text","")
        print(json.dumps({"jsonrpc":"2.0","id":mid,"result":{"content":[{"type":"text","text":"echo: "+t}]}}), flush=True)
    elif mid is not None:
        print(json.dumps({"jsonrpc":"2.0","id":mid,"error":{"code":-32601,"message":"nope"}}), flush=True)
"#;

    fn fake_config() -> McpServerConfig {
        McpServerConfig {
            command: "python3".into(),
            args: vec!["-c".into(), FAKE_SERVER.into()],
            env: Default::default(),
        }
    }

    #[tokio::test]
    async fn handshake_list_and_call() {
        let server = McpServer::connect("fake", &fake_config())
            .await
            .expect("connect");
        let specs = server.specs();
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].id, "mcp__fake__echo");
        assert!(specs[0].input_schema.get("properties").is_some());

        let out = server.call_tool("echo", r#"{"text":"xin chao"}"#).await;
        assert_eq!(out, "echo: xin chao");
    }

    #[tokio::test]
    async fn hub_routes_by_server_name() {
        let mut cfgs = BTreeMap::new();
        cfgs.insert("fake".to_string(), fake_config());
        let (hub, status) = McpToolHub::connect_all(&cfgs).await;
        assert!(status[0].contains("1 tools"), "{status:?}");
        assert!(hub.owns("mcp__fake__echo"));
        let out = hub.call("mcp__fake__echo", r#"{"text":"hi"}"#).await;
        assert_eq!(out, "echo: hi");
        let missing = hub.call("mcp__ghost__echo", "{}").await;
        assert!(missing.contains("no connected server"));
    }

    #[tokio::test]
    async fn failed_spawn_is_reported_not_fatal() {
        let mut cfgs = BTreeMap::new();
        cfgs.insert(
            "broken".to_string(),
            McpServerConfig {
                command: "/nonexistent-mcp-binary".into(),
                args: vec![],
                env: Default::default(),
            },
        );
        let (hub, status) = McpToolHub::connect_all(&cfgs).await;
        assert!(hub.is_empty());
        assert!(status[0].contains("FAILED"));
    }
}
