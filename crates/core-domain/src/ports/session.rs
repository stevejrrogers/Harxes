//! Driven port (hexagonal): persistence of agent session transcripts so a
//! conversation can be resumed across restarts.

use serde::{Deserialize, Serialize};

use crate::domain::value_objects::{Message, TodoItem};

/// A serializable record of one agent session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionRecord {
    pub id: String,
    pub created_at: String,
    pub transcript: Vec<Message>,
    /// Snapshot of the agent-managed plan, restored on `--resume`.
    #[serde(default)]
    pub todos: Vec<TodoItem>,
}

/// Driven port for storing/loading [`SessionRecord`]s on disk.
pub trait SessionStorePort: Send + Sync {
    /// Persist a session under its id.
    fn save(&self, session: &SessionRecord) -> Result<(), SessionStoreError>;
    /// Load a previously saved session by id.
    fn load(&self, id: &str) -> Option<SessionRecord>;
}

#[derive(Debug)]
pub enum SessionStoreError {
    Io(String),
}

impl std::fmt::Display for SessionStoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(m) => write!(f, "session store io error : {m}"),
        }
    }
}
impl std::error::Error for SessionStoreError {}
