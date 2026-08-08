use crate::domain::value_objects::{ModelId, ProviderId};

#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub provider: ProviderId,
    pub model: ModelId,
}

#[derive(Debug, Clone)]
pub struct Agent {
    id: String,
    config: AgentConfig,
}

impl Agent {
    pub fn new(id: impl Into<String>, config: AgentConfig) -> Self {
        Self {
            id: id.into(),
            config,
        }
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn config(&self) -> &AgentConfig {
        &self.config
    }
}
