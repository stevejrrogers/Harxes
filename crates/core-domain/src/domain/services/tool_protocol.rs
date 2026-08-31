//! Domain service that defines the tool-calling protocol used by the agent
//! loop: which tools exist, how to parse their arguments and how to render a
//! result back into the transcript. Keeps model-facing conventions out of the
//! application usecase layer.

use crate::domain::value_objects::{TodoAction, TodoItem, TodoStatus, ToolSpec};

/// Identifiers of the tools the agent may invoke.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolId {
    Bash,
    Read,
    Write,
    Edit,
    Grep,
    Glob,
    Delegate,
    Todo,
}

impl ToolId {
    pub fn as_str(&self) -> &'static str {
        match self {
            ToolId::Bash => "Bash",
            ToolId::Read => "Read",
            ToolId::Write => "Write",
            ToolId::Edit => "Edit",
            ToolId::Grep => "Grep",
            ToolId::Glob => "Glob",
            ToolId::Delegate => "Delegate",
            ToolId::Todo => "Todo",
        }
    }
    pub fn parse(name: &str) -> Option<ToolId> {
        match name {
            "Bash" | "bash" => Some(ToolId::Bash),
            "Read" | "read" => Some(ToolId::Read),
            "Write" | "write" => Some(ToolId::Write),
            "Edit" | "edit" => Some(ToolId::Edit),
            "Grep" | "grep" => Some(ToolId::Grep),
            "Glob" | "glob" => Some(ToolId::Glob),
            "Delegate" | "delegate" => Some(ToolId::Delegate),
            "Todo" | "todo" => Some(ToolId::Todo),
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
    Edit { path: String, old_string: String, new_string: String },
    Grep { needle: String, pattern: String, max_matches: usize },
    Glob { pattern: String, max_depth: Option<usize> },
    Delegate { task: String, context: Option<String> },
    Todo { action: TodoAction, items: Vec<TodoItem> },
}

/// The full set of tool specifications offered to the model.
pub fn all_tool_specs() -> Vec<ToolSpec> {
    vec![
        ToolSpec::with_schema(
            "Bash",
            "Run a shell command and capture its output.",
            obj_schema(vec![("command", "string")]),
        ),
        ToolSpec::with_schema(
            "Read",
            "Read a file from disk by path.",
            obj_schema(vec![("path", "string")]),
        ),
        ToolSpec::with_schema(
            "Write",
            "Write content to a file on disk.",
            obj_schema(vec![("path", "string"), ("content", "string")]),
        ),
        ToolSpec::with_schema(
            "Edit",
            "Replace a unique old_string with new_string inside an existing file (surgical edit).",
            obj_schema(vec![
                ("path", "string"),
                ("old_string", "string"),
                ("new_string", "string"),
            ]),
        ),
        ToolSpec::with_schema(
            "Grep",
            "Search files for a pattern (regular expression or plain text) under a glob path.",
            obj_schema(vec![
                ("needle", "string"),
                ("pattern", "string"),
                ("max_matches", "integer"),
            ]),
        ),
        ToolSpec::with_schema(
            "Glob",
            "List files matching a glob pattern (e.g. **/*.rs).",
            obj_schema(vec![("pattern", "string"), ("max_depth", "integer")]),
        ),
        ToolSpec::with_schema(
            "Delegate",
            "Delegate an independent sub-task to a fresh sub-agent that runs in its own context (separate window, same filesystem/shell), then returns a concise final result. Use for parallelizable or self-contained investigation tasks. Provide a self-contained task; optionally include a snippet of relevant context.",
            {
                let mut properties = serde_json::Map::new();
                properties.insert(
                    "task".to_string(),
                    serde_json::json!({ "type": "string", "description": "Self-contained sub-task for the sub-agent" }),
                );
                properties.insert(
                    "context".to_string(),
                    serde_json::json!({ "type": "string", "description": "Optional relevant context snippet" }),
                );
                serde_json::json!({
                    "type": "object",
                    "properties": properties,
                    "required": ["task"],
                })
            },
        ),
        ToolSpec::with_schema(
            "Todo",
            "Manage your task plan. Use action=write to replace the entire plan with the `todos` array — send the complete updated list every time (never a partial one). Mark exactly one item in_progress before starting work on it, and mark it completed immediately when finished. Use for any multi-step task; action=list reads the current plan back.",
            {
                let mut properties = serde_json::Map::new();
                properties.insert(
                    "action".to_string(),
                    serde_json::json!({ "type": "string", "enum": ["write", "list"], "description": "write replaces the whole plan; list reads it" }),
                );
                properties.insert(
                    "todos".to_string(),
                    serde_json::json!({
                        "type": "array",
                        "description": "The complete plan (required for write)",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": { "type": "string", "description": "Imperative step description, e.g. 'Run tests'" },
                                "active_form": { "type": "string", "description": "Present-continuous form shown while working, e.g. 'Running tests'" },
                                "status": { "type": "string", "enum": ["pending", "in_progress", "completed"] }
                            },
                            "required": ["label", "status"]
                        }
                    }),
                );
                serde_json::json!({
                    "type": "object",
                    "properties": properties,
                    "required": ["action"],
                })
            },
        ),
    ]
}

/// Build a JSON-Schema object with properties of the given types, all required.
///
/// `props` is a list of `(name, jsonschema_type)` pairs (e.g. `"string"`,
/// `"integer"`). The schema declares every property as required.
fn obj_schema(props: Vec<(&'static str, &'static str)>) -> serde_json::Value {
    let mut properties = serde_json::Map::new();
    let mut required = Vec::new();
    for (name, ty) in props {
        required.push(serde_json::Value::String(name.to_string()));
        properties.insert(name.to_string(), serde_json::json!({ "type": ty }));
    }
    serde_json::json!({
        "type": "object",
        "properties": properties,
        "required": required,
    })
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
                ToolId::Edit => {
                    let path = obj.get("path").and_then(|x| x.as_str()).unwrap_or("");
                    let old_string = obj.get("old_string").and_then(|x| x.as_str()).unwrap_or("");
                    let new_string = obj.get("new_string").and_then(|x| x.as_str()).unwrap_or("");
                    return ParsedArgs::Edit {
                        path: path.to_string(),
                        old_string: old_string.to_string(),
                        new_string: new_string.to_string(),
                    };
                }
                ToolId::Grep => {
                    let needle = obj.get("needle").and_then(|x| x.as_str()).unwrap_or("");
                    let pattern = obj
                        .get("pattern")
                        .and_then(|x| x.as_str())
                        .unwrap_or("**/*");
                    let max_matches = obj
                        .get("max_matches")
                        .and_then(|x| x.as_u64())
                        .unwrap_or(100) as usize;
                    return ParsedArgs::Grep {
                        needle: needle.to_string(),
                        pattern: pattern.to_string(),
                        max_matches,
                    };
                }
                ToolId::Glob => {
                    let pattern = obj.get("pattern").and_then(|x| x.as_str()).unwrap_or(raw);
                    let max_depth = obj.get("max_depth").and_then(|x| x.as_u64()).map(|d| d as usize);
                    return ParsedArgs::Glob {
                        pattern: pattern.to_string(),
                        max_depth,
                    };
                }
                ToolId::Delegate => {
                    let task = obj.get("task").and_then(|x| x.as_str()).unwrap_or("");
                    let context = obj.get("context").and_then(|x| x.as_str());
                    return ParsedArgs::Delegate {
                        task: task.to_string(),
                        context: context.map(|s| s.to_string()),
                    };
                }
                ToolId::Todo => {
                    let action = obj
                        .get("action")
                        .and_then(|x| x.as_str())
                        .and_then(TodoAction::parse)
                        .unwrap_or(TodoAction::List);
                    let items = obj
                        .get("todos")
                        .and_then(|x| x.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|it| {
                                    let label = it.get("label")?.as_str()?.trim();
                                    if label.is_empty() {
                                        return None;
                                    }
                                    Some(TodoItem {
                                        label: label.to_string(),
                                        active_form: it
                                            .get("active_form")
                                            .and_then(|x| x.as_str())
                                            .unwrap_or("")
                                            .to_string(),
                                        status: it
                                            .get("status")
                                            .and_then(|x| x.as_str())
                                            .and_then(TodoStatus::parse)
                                            .unwrap_or(TodoStatus::Pending),
                                    })
                                })
                                .collect()
                        })
                        .unwrap_or_default();
                    return ParsedArgs::Todo { action, items };
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
        ToolId::Edit => {
            // Plain-string payloads are ambiguous; default to empty so the
            // model is told to pass JSON (path/old_string/new_string).
            ParsedArgs::Edit {
                path: String::new(),
                old_string: String::new(),
                new_string: String::new(),
            }
        }
        ToolId::Grep => ParsedArgs::Grep {
            needle: raw.trim().to_string(),
            pattern: "**/*".to_string(),
            max_matches: 100,
        },
        ToolId::Glob => ParsedArgs::Glob {
            pattern: raw.trim().to_string(),
            max_depth: None,
        },
        ToolId::Delegate => ParsedArgs::Delegate {
            task: raw.trim().to_string(),
            context: None,
        },
        ToolId::Todo => ParsedArgs::Todo {
            action: TodoAction::List,
            items: Vec::new(),
        },
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

    #[test]
    fn parse_json_delegate_args() {
        match parse_args(ToolId::Delegate, r#"{"task":"t","context":"c"}"#) {
            ParsedArgs::Delegate { task, context } => {
                assert_eq!(task, "t");
                assert_eq!(context.as_deref(), Some("c"));
            }
            _ => panic!("expected delegate"),
        }
        // context is optional
        match parse_args(ToolId::Delegate, r#"{"task":"only"}"#) {
            ParsedArgs::Delegate { task, context } => {
                assert_eq!(task, "only");
                assert!(context.is_none());
            }
            _ => panic!("expected delegate"),
        }
    }

    #[test]
    fn parse_json_todo_args() {
        let raw = r#"{"action":"write","todos":[
            {"label":"write tests","active_form":"Writing tests","status":"in_progress"},
            {"label":"ship it","status":"pending"},
            {"label":"","status":"pending"}
        ]}"#;
        match parse_args(ToolId::Todo, raw) {
            ParsedArgs::Todo { action, items } => {
                assert_eq!(action, TodoAction::Write);
                assert_eq!(items.len(), 2, "blank labels are dropped");
                assert_eq!(items[0].label, "write tests");
                assert_eq!(items[0].active_form, "Writing tests");
                assert_eq!(items[0].status, TodoStatus::InProgress);
                assert_eq!(items[1].status, TodoStatus::Pending);
            }
            _ => panic!("expected todo"),
        }
        // unknown action defaults to list
        match parse_args(ToolId::Todo, r#"{"action":"nope"}"#) {
            ParsedArgs::Todo { action, .. } => assert_eq!(action, TodoAction::List),
            _ => panic!("expected todo"),
        }
    }
}
