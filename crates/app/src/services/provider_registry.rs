use serde::Serialize;

/// A statically-known provider descriptor. The registry acts as a Strategy
/// factory: the CLI picks one based on the active config and builds the
/// corresponding infrastructure adapter.
#[derive(Debug, Clone, Serialize)]
pub struct ProviderDescriptor {
    pub id: &'static str,
    pub display_name: &'static str,
    pub base_url: &'static str,
    pub default_model: &'static str,
    pub api_key_env_var: &'static str,
}

pub const ANTHROPIC: ProviderDescriptor = ProviderDescriptor {
    id: "anthropic",
    display_name: "Anthropic Claude",
    base_url: "https://api.anthropic.com/v1/messages",
    default_model: "claude-sonnet-4-5",
    api_key_env_var: "ANTHROPIC_API_KEY",
};

pub const OPENAI: ProviderDescriptor = ProviderDescriptor {
    id: "openai",
    display_name: "OpenAI GPT",
    base_url: "https://api.openai.com/v1/chat/completions",
    default_model: "gpt-4o",
    api_key_env_var: "OPENAI_API_KEY",
};

/// Registry / simple factory over known providers.
pub struct ProviderRegistry;

impl ProviderRegistry {
    pub fn all() -> Vec<ProviderDescriptor> {
        vec![ANTHROPIC.clone(), OPENAI.clone()]
    }

    pub fn find(id: &str) -> Option<ProviderDescriptor> {
        Self::all().into_iter().find(|p| p.id == id)
    }
}
