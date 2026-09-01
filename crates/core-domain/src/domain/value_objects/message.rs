use serde::{Deserialize, Serialize};

use super::tool_call::ToolCall;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// An image attached to a user message, carried as base64 for the provider
/// adapters to encode into their multimodal wire formats.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageData {
    /// MIME type, e.g. `image/png`.
    pub media_type: String,
    /// Base64-encoded bytes (no data-URI prefix).
    pub base64: String,
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
    /// Images attached to a [`Role::User`] message (vision input).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<ImageData>,
}

impl Message {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            tool_calls: vec![],
            tool_call_id: None,
            images: vec![],
        }
    }

    /// Build a user message with attached images (vision input).
    pub fn user_with_images(content: impl Into<String>, images: Vec<ImageData>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
            tool_calls: vec![],
            tool_call_id: None,
            images,
        }
    }

    /// Build an assistant message carrying one or more requested [`ToolCall`]s.
    pub fn assistant_with_tools(tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: Role::Assistant,
            content: String::new(),
            tool_calls,
            tool_call_id: None,
            images: vec![],
        }
    }

    /// Build a [`Role::Tool`] result message for a given call id.
    pub fn tool_result(call_id: impl Into<String>, output: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: output.into(),
            tool_calls: vec![],
            tool_call_id: Some(call_id.into()),
            images: vec![],
        }
    }
}
