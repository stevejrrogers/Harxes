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
    /// MCP servers to launch: name -> stdio launch spec. Their tools appear
    /// to the model as `mcp__<name>__<tool>`.
    #[serde(default)]
    pub mcp: std::collections::BTreeMap<String, McpServerConfig>,
    /// Lifecycle hooks run around tool execution.
    #[serde(default)]
    pub hooks: HooksConfig,
    /// Ordered fallback models tried when the primary model fails terminally.
    #[serde(default)]
    pub fallback_models: Vec<String>,
    /// Agent-loop guardrail overrides (absent fields keep the defaults).
    #[serde(default)]
    pub limits: LimitsConfig,
    /// Default reasoning effort: "low" | "medium" | "high" (absent = provider default).
    #[serde(default)]
    pub reasoning_effort: Option<String>,
}

/// Optional guardrail overrides for the agent loop.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LimitsConfig {
    pub max_iterations: Option<usize>,
    pub max_total_tokens: Option<u64>,
    pub context_window_tokens: Option<u64>,
    /// Seconds to wait out a rate-limited (429) provider before failing.
    pub rate_limit_patience_secs: Option<u64>,
    /// Distinct recent tool outputs kept verbatim before aging (default 8).
    pub aging_keep_recent: Option<usize>,
}

/// One lifecycle hook: a shell command run around tool execution. `matcher`
/// filters by tool name (`*` wildcard supported; empty = every tool).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HookRule {
    #[serde(rename = "match", default)]
    pub matcher: String,
    pub command: String,
}

/// Hook sets by lifecycle event. A `pre_tool` hook exiting with code 2 blocks
/// the tool call (its output becomes the refusal reason shown to the model);
/// `post_tool` hook output is appended to the tool result as feedback.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HooksConfig {
    #[serde(default)]
    pub pre_tool: Vec<HookRule>,
    #[serde(default)]
    pub post_tool: Vec<HookRule>,
}

/// Launch spec for one stdio MCP server.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct McpServerConfig {
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    /// Extra environment variables for the server process.
    #[serde(default)]
    pub env: std::collections::BTreeMap<String, String>,
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
