use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ToolKind {
    Bash,
    Read,
    Write,
    Glob,
}

#[derive(Debug, Clone)]
pub struct Tool {
    id: String,
    kind: ToolKind,
    description: String,
}

impl Tool {
    pub fn new(id: impl Into<String>, kind: ToolKind, description: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            kind,
            description: description.into(),
        }
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn kind(&self) -> &ToolKind {
        &self.kind
    }

    pub fn description(&self) -> &str {
        &self.description
    }
}
