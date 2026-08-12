//! Tool-calling value objects used by the agent loop.

use serde::{Deserialize, Serialize};

/// A request from the model to invoke a tool.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    /// Canonical tool id/name as known to the runner (e.g. "Bash").
    pub name: String,
    /// JSON-encoded arguments for the tool.
    pub arguments: String,
}

/// The outcome of executing a [`ToolCall`].
#[derive(Debug, Clone)]
pub struct ToolResult {
    /// Matches [`ToolCall::id`].
    pub call_id: String,
    pub output: String,
}

/// Declaration of a tool available to the model.
#[derive(Debug, Clone)]
pub struct ToolSpec {
    pub id: String,
    pub description: String,
    /// JSON-Schema describing the tool's arguments. Providers serialize this
    /// into their tool-declaration format so the model knows what to send.
    pub input_schema: serde_json::Value,
}

impl ToolSpec {
    pub fn new(id: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            description: description.into(),
            input_schema: serde_json::Value::Null,
        }
    }

    /// Build a tool spec with an explicit argument schema.
    pub fn with_schema(
        id: impl Into<String>,
        description: impl Into<String>,
        input_schema: serde_json::Value,
    ) -> Self {
        Self {
            id: id.into(),
            description: description.into(),
            input_schema,
        }
    }

    /// Number of required argument names for this tool (from the schema's
    /// `required` list), if present. Used by providers that need it.
    pub fn required_args(&self) -> Vec<String> {
        self.input_schema
            .get("required")
            .and_then(|r| r.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default()
    }
}
