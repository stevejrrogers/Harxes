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
}

impl ToolSpec {
    pub fn new(id: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            description: description.into(),
        }
    }
}
