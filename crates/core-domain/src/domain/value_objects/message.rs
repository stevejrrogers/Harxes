use serde::{Deserialize, Serialize};

use super::tool_call::ToolCall;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// A chat message that may carry an assistant's requested [`ToolCall`]s or a
/// [`Role::Tool`] result referencing a specific call id.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
    /// Populated for [`Role::Assistant`] messages that request tool use.
    pub tool_calls: Vec<ToolCall>,
    /// Populated for [`Role::Tool`] messages linking back to a call id.
    pub tool_call_id: Option<String>,
}

impl Message {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            tool_calls: vec![],
            tool_call_id: None,
        }
    }

    /// Build an assistant message carrying one or more requested [`ToolCall`]s.
    pub fn assistant_with_tools(tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: Role::Assistant,
            content: String::new(),
            tool_calls,
            tool_call_id: None,
        }
    }

    /// Build a [`Role::Tool`] result message for a given call id.
    pub fn tool_result(call_id: impl Into<String>, output: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: output.into(),
            tool_calls: vec![],
            tool_call_id: Some(call_id.into()),
        }
    }
}
