//! Composition root: wires infrastructure adapters into application ports,
//! resolving providers from both the static registry and on-disk config so
//! custom/OpenAI-compatible endpoints (e.g. LiteLLM proxy) are supported.

use std::sync::Arc;

use harxes_app::ports::AgentPort;
use harxes_app::services::provider_registry::{ProviderDescriptor, ProviderRegistry};
use harxes_app::usecases::agent_loop::{AgentLoop, LoopLimits};
use harxes_core_domain::ports::ProviderConfig;
use harxes_core_domain::ports::{ConfigStorePort, LlmPort, SecretsVaultPort, SessionStorePort};
use harxes_infra_auth::EnvSecretsVault;
use harxes_infra_fs::HostFileSystem;
use harxes_infra_llm::providers::{anthropic::AnthropicClient, openai::OpenAiClient};
use harxes_infra_session::{JsonConfigStore, JsonSessionStore};
use harxes_infra_shell::TokioCommandShell;

/// A resolved provider target: either a built-in one or one defined by user
/// configuration pointing at any OpenAI-compatible endpoint (LiteLLM etc.).
#[derive(Debug, Clone)]
pub enum ResolvedProvider {
    Native(ProviderDescriptor),
    Custom(ProviderConfig),
}

impl ResolvedProvider {
    pub fn id(&self) -> &str {
        match self {
            ResolvedProvider::Native(d) => d.id,
            ResolvedProvider::Custom(c) => &c.id,
        }
    }
}

/// Default location of the user configuration file directory.
pub fn default_config_dir() -> String {
    std::env::var("HARXES_HOME").unwrap_or_else(|_| format!("{}/.harxes", std::env!("HOME")))
}

/// Load providers defined on disk (`~/.harxes/config.json`), merging them with
/// the static built-in registry.
fn load_custom_providers() -> Vec<ResolvedProvider> {
    let store = JsonConfigStore::new(default_config_dir());
    let cfg = store.load();
    let mut out: Vec<ResolvedProvider> = vec![];
    for (_id, pc) in cfg.providers {
        out.push(ResolvedProvider::Custom(pc));
    }
    // built-ins last so CLI --provider can still find native ones by id.
    for d in ProviderRegistry::all() {
        if !out.iter().any(|r| r.id() == d.id) {
            out.push(ResolvedProvider::Native(d));
        }
    }
    out
}

/// Resolve a provider by id from config first, then the built-in registry.
pub fn resolve_provider(id: &str) -> Result<ResolvedProvider, String> {
    let all = load_custom_providers();
    all.into_iter()
        .find(|r| r.id() == id)
        .ok_or_else(|| format!("unknown provider: {id}"))
}

/// Build the concrete LLM adapter for a resolved provider, reading its API key
/// from the environment via [`EnvSecretsVault`].
fn override_url(rp: ResolvedProvider, url: &str) -> ResolvedProvider {
    match rp {
        ResolvedProvider::Native(d) => {
            let cfg = ProviderConfig {
                id: d.id.to_string(),
                base_url: url.to_string(),
                default_model: d.default_model.to_string(),
                api_key_env: d.api_key_env_var.to_string(),
            };
            ResolvedProvider::Custom(cfg)
        }
        ResolvedProvider::Custom(mut c) => {
            c.base_url = url.to_string();
            ResolvedProvider::Custom(c)
        }
    }
}

pub fn build_llm(rp: &ResolvedProvider) -> Result<Arc<dyn LlmPort>, String> {
    let vault = EnvSecretsVault;
    match rp {
        ResolvedProvider::Native(d) => {
            let key = vault
                .get_secret(d.api_key_env_var)
                .ok_or_else(|| format!("missing API key env var {}", d.api_key_env_var))?;
            match d.id {
                "anthropic" => Ok(Arc::new(AnthropicClient::new(d.base_url, key))),
                _ => Ok(Arc::new(OpenAiClient::new(d.base_url, key))),
            }
        }
        ResolvedProvider::Custom(c) => {
            // Any OpenAI-compatible endpoint (LiteLLM proxy, local model server).
            let key = vault.get_secret(&c.api_key_env).unwrap_or_default();
            Ok(Arc::new(OpenAiClient::new(c.base_url.clone(), key)))
        }
    }
}

/// Default provider id when none is given.
pub fn default_provider_id() -> String {
    let store = JsonConfigStore::new(default_config_dir());
    store
        .load()
        .active_provider
        .unwrap_or_else(|| "anthropic".to_string())
}

/// Result of assembly: a wired agent (as an abstraction) plus provider info.
pub struct Wiring {
    pub agent: Arc<dyn AgentPort>,
    pub provider_id: String,
    pub limits: LoopLimits,
}

/// Assemble a fully-wired agent for the given (optional) provider id. The CLI
/// receives the agent through the [`AgentPort`] abstraction, not a concrete type.
pub fn assemble(cli_provider: Option<&str>, cli_base_url: Option<&str>) -> Result<Wiring, String> {
    let pid = match cli_provider {
        Some(p) => p.to_string(),
        None => default_provider_id(),
    };
    let mut rp = resolve_provider(&pid)?;
    if let Some(url) = cli_base_url {
        rp = override_url(rp, url);
    }
    let llm = build_llm(&rp)?;
    let shell = Arc::new(TokioCommandShell::new(120));
    let fsys = Arc::new(HostFileSystem);
    let agent: Arc<dyn AgentPort> = Arc::new(
        AgentLoop::with_session(
            AgentLoop::new(llm, shell, fsys),
            Arc::new(JsonSessionStore::new(default_config_dir())) as Arc<dyn SessionStorePort>,
            pid.clone(),
        )
        .with_observer(Arc::new(crate::ui::ConsoleToolObserver)),
    );
    Ok(Wiring {
        agent,
        provider_id: pid.clone(),
        limits: LoopLimits::default(),
    })
}

/// Effective model for a resolved provider; CLI override wins, else descriptor default.
pub fn effective_model(rp: &ResolvedProvider, cli_model: Option<&str>) -> String {
    if let Some(m) = cli_model {
        return m.to_string();
    }
    match rp {
        ResolvedProvider::Native(d) => d.default_model.to_string(),
        ResolvedProvider::Custom(c) => c.default_model.clone(),
    }
}

/// Enumerate all provider ids available (native + custom from config).
pub fn available_providers() -> Vec<String> {
    load_custom_providers()
        .into_iter()
        .map(|r| r.id().to_string())
        .collect()
}
