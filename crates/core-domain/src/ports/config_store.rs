use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub id: String,
    pub base_url: String,
    pub default_model: String,
    pub api_key_env: String,
}

/// USD pricing per million tokens for one model (substring-matched by id).
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct ModelPricing {
    pub input_per_mtok: f64,
    pub output_per_mtok: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HarxesConfig {
    pub active_provider: Option<String>,
    /// provider_id -> config
    #[serde(default)]
    pub providers: std::collections::BTreeMap<String, ProviderConfig>,
    /// Shell-command allow/deny rules (see `CommandPolicy`): allow skips the
    /// approval prompt for risky commands, deny refuses outright.
    #[serde(default)]
    pub commands: crate::domain::value_objects::CommandPolicy,
    /// Cost-estimate overrides: model-id substring -> USD per million tokens.
    #[serde(default)]
    pub pricing: std::collections::BTreeMap<String, ModelPricing>,
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
