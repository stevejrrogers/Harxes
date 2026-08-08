//! Domain service that defines the tool-calling protocol used by the agent
//! loop: which tools exist, how to parse their arguments and how to render a
//! result back into the transcript. Keeps model-facing conventions out of the
//! application usecase layer.

use crate::domain::value_objects::ToolSpec;

/// Identifiers of the tools the agent may invoke.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolId {
    Bash,
    Read,
    Write,
}

impl ToolId {
    pub fn as_str(&self) -> &'static str {
        match self {
            ToolId::Bash => "Bash",
            ToolId::Read => "Read",
            ToolId::Write => "Write",
        }
    }
    pub fn parse(name: &str) -> Option<ToolId> {
        match name {
            "Bash" | "bash" => Some(ToolId::Bash),
            "Read" | "read" => Some(ToolId::Read),
            "Write" | "write" => Some(ToolId::Write),
            _ => None,
        }
    }
}

/// Type-safe argument payloads for each tool.
#[derive(Debug)]
pub enum ParsedArgs {
    Bash { command: String },
    Read { path: String },
    Write { path: String, content: String },
}

/// The full set of tool specifications offered to the model.
pub fn all_tool_specs() -> Vec<ToolSpec> {
    vec![
        ToolSpec::new("Bash", "Run a shell command and capture its output."),
        ToolSpec::new("Read", "Read a file from disk by path."),
        ToolSpec::new("Write", "Write content to a file on disk."),
    ]
}

/// Parse raw tool-call arguments into a typed payload. Model arguments may be
/// a JSON object (OpenAI-style) or a plain string; JSON fields win.
pub fn parse_args(tool: ToolId, raw: &str) -> ParsedArgs {
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(raw) {
        if let Some(obj) = v.as_object() {
            match tool {
                ToolId::Bash => {
                    let command = obj.get("command").and_then(|x| x.as_str()).unwrap_or(raw);
                    return ParsedArgs::Bash {
                        command: command.to_string(),
                    };
                }
                ToolId::Read => {
                    let path = obj.get("path").and_then(|x| x.as_str()).unwrap_or(raw);
                    return ParsedArgs::Read {
                        path: path.to_string(),
                    };
                }
                ToolId::Write => {
                    let path = obj.get("path").and_then(|x| x.as_str()).unwrap_or(raw);
                    let content = obj.get("content").and_then(|x| x.as_str()).unwrap_or("");
                    return ParsedArgs::Write {
                        path: path.to_string(),
                        content: content.to_string(),
                    };
                }
            }
        }
    }

    match tool {
        ToolId::Bash => ParsedArgs::Bash {
            command: raw.trim().to_string(),
        },
        ToolId::Read => ParsedArgs::Read {
            path: raw.trim().to_string(),
        },
        ToolId::Write => {
            let (path, content) = match raw.split_once('\n') {
                Some((p, c)) => (p.trim().to_string(), c.to_string()),
                None => (raw.trim().to_string(), String::new()),
            };
            ParsedArgs::Write { path, content }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_json_bash_args() {
        match parse_args(ToolId::Bash, r#"{"command":"echo hi"}"#) {
            ParsedArgs::Bash { command } => assert_eq!(command, "echo hi"),
            _ => panic!("expected bash"),
        }
    }

    #[test]
    fn parse_json_write_args() {
        match parse_args(ToolId::Write, r#"{"path":"a.txt","content":"hello"}"#) {
            ParsedArgs::Write { path, content } => {
                assert_eq!(path, "a.txt");
                assert_eq!(content, "hello");
            }
            _ => panic!("expected write"),
        }
    }

    #[test]
    fn parse_plain_string_fallback() {
        match parse_args(ToolId::Read, "src/main.rs") {
            ParsedArgs::Read { path } => assert_eq!(path, "src/main.rs"),
            _ => panic!(),
        }
    }
}
