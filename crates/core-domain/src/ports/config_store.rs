use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub id: String,
    pub base_url: String,
    pub default_model: String,
    pub api_key_env: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HarxesConfig {
    pub active_provider: Option<String>,
    /// provider_id -> config
    #[serde(default)]
    pub providers: std::collections::BTreeMap<String, ProviderConfig>,
}

/// Driven port (hexagonal): persistence of user preferences (which provider is
/// active etc.). Concrete implementors store JSON in `~/.harxes/config.json`.
pub trait ConfigStorePort: Send + Sync {
    fn load(&self) -> HarxesConfig;
    fn save(&self, config: &HarxesConfig) -> Result<(), ConfigStoreError>;
}

#[derive(Debug)]
pub enum ConfigStoreError {
    Io(String),
}

impl std::fmt::Display for ConfigStoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(msg) => write!(f, "config store io error: {msg}"),
        }
    }
}

impl std::error::Error for ConfigStoreError {}
