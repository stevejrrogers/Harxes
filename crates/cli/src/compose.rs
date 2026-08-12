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

/// Default AGENTS.md content, written only when the project has none.
/// AGENTS.md is the cross-agent instruction file (Claude Code, opencode,
/// cursor, codex all auto-load it into context).
const AGENTS_MD_TEMPLATE: &str = r#"# Project agent guide

This file is read automatically by AI coding agents (Harxes, Claude Code,
opencode, cursor, codex) to learn how to work in this project.

## Conventions
- Prefer the language and frameworks already used in this repository.
- Keep changes minimal and focused; do not touch unrelated code.
- Add or update tests for any behavior you change.

## Commands
- Build:  cargo build --release
- Test:   cargo test --workspace
- Lint:   cargo clippy --all-targets
"#;

/// Seed content for shared workspace memory (.harxes/agents/NOTES.md).
const NOTES_TEMPLATE: &str = r#"# Shared agent memory

This file is shared across sessions. Use /remember to add notes that any
future session of Harxes (or another agent) should know about this project.
"#;

/// Create the standard agent-scoped project structure, shared with other
/// agents (Claude Code, opencode, cursor, codex all auto-read AGENTS.md).
/// Idempotent: existing files are never overwritten.
pub fn ensure_project_scaffold() {
    use std::path::PathBuf;
    // AGENTS.md lives at the project root (cwd), shared across all agents.
    let agents_md = PathBuf::from("AGENTS.md");
    if !agents_md.exists() {
        let _ = std::fs::write(&agents_md, AGENTS_MD_TEMPLATE);
    }
    // .harxes/ agent workspace: memory + per-session context.
    let harxes = PathBuf::from(".harxes");
    let _ = std::fs::create_dir_all(harxes.join("agents"));
    let _ = std::fs::create_dir_all(harxes.join("sessions"));
    // Seed shared workspace memory file with guidance if absent.
    let notes = harxes.join("agents").join("NOTES.md");
    if !notes.exists() {
        let _ = std::fs::write(&notes, NOTES_TEMPLATE);
    }
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
    /// Live view of tools currently executing (shared with the TUI).
    pub active_tools: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    /// Set to true once the observer delivered at least one streaming delta
    /// during a one-shot turn (the live stream already printed the text).
    pub streamed: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Interactive approval gate for dangerous operations.
    pub approval_gate: Arc<ApprovalGate>,
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
    let active_tools: std::sync::Arc<std::sync::Mutex<Vec<String>>> =
        std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let streamed = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let observer = crate::ui::LiveToolObserver {
        active: active_tools.clone(),
        streamed: streamed.clone(),
    };
    let approval_gate = Arc::new(ApprovalGate::default());
    let decider = Arc::new(GateDecider(approval_gate.clone()));
    let agent: Arc<dyn AgentPort> = Arc::new(
        AgentLoop::with_session(
            AgentLoop::new(llm, shell, fsys),
            Arc::new(JsonSessionStore::new(default_config_dir())) as Arc<dyn SessionStorePort>,
            pid.clone(),
        )
        .with_decider(decider)
        .with_streaming(true)
        .with_observer(Arc::new(observer)),
    );
    Ok(Wiring {
        agent,
        provider_id: pid.clone(),
        limits: LoopLimits::default(),
        active_tools,
        streamed,
        approval_gate,
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

/// Interactive approval gate: blocks an operation until the TUI user replies.
#[derive(Debug, Default)]
pub struct ApprovalGate {
    inner: std::sync::Mutex<Vec<Request>>,
    wake: std::sync::Condvar,
}

#[derive(Debug, Default)]
struct Request {
    description: String,
    allowed: Option<bool>,
}

impl ApprovalGate {
    /// Request approval for `description`; blocks the caller thread until a
    /// decision is made (or auto-denies if the process is shutting down).
    pub fn request(&self, description: &str) -> bool {
        let mut q = self.inner.lock().unwrap();
        q.push(Request {
            description: description.to_string(),
            allowed: None,
        });
        loop {
            if let Some(r) = q.last_mut() {
                if let Some(v) = r.allowed {
                    return v;
                }
            }
            q = self.wake.wait(q).unwrap();
        }
    }

    /// True when there is an outstanding request awaiting a decision.
    pub fn has_pending(&self) -> Option<String> {
        let q = self.inner.lock().unwrap();
        q.iter()
            .rev()
            .find(|r| r.allowed.is_none())
            .map(|r| r.description.clone())
    }

    /// Resolve the most recent pending request with `allowed`.
    pub fn resolve(&self, allowed: bool) -> bool {
        let mut q = self.inner.lock().unwrap();
        if let Some(r) = q.iter_mut().rev().find(|r| r.allowed.is_none()) {
            r.allowed = Some(allowed);
            self.wake.notify_all();
            true
        } else {
            false
        }
    }

    /// Drop any pending requests (auto-deny so a blocked task can exit).
    pub fn cancel_all(&self) {
        let mut q = self.inner.lock().unwrap();
        for r in q.iter_mut() {
            if r.allowed.is_none() {
                r.allowed = Some(false);
            }
        }
        self.wake.notify_all();
    }
}

/// Bridges [`ApprovalGate`] to the application's [`PermissionDecider`] port.
pub struct GateDecider(pub Arc<ApprovalGate>);

impl harxes_core_domain::ports::PermissionDecider for GateDecider {
    fn decide_write(&self, path: &str) -> bool {
        self.0.request(&format!("write {path}"))
    }
    fn decide_bash(&self, command: &str) -> bool {
        self.0.request(&format!("run bash \"{command}\""))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use harxes_core_domain::ports::PermissionDecider;

    #[test]
    fn approval_gate_blocks_then_resolves() {
        let gate = Arc::new(ApprovalGate::default());
        let g2 = gate.clone();
        let worker = std::thread::spawn(move || g2.request("write /tmp/x"));
        // Give the worker a moment to block on the request.
        std::thread::sleep(std::time::Duration::from_millis(100));
        assert!(gate.has_pending().is_some());
        assert!(gate.resolve(true));
        assert!(worker.join().unwrap());
    }

    #[test]
    fn decider_bridges_to_gate() {
        let gate = Arc::new(ApprovalGate::default());
        let decider = GateDecider(gate.clone());
        let g2 = gate.clone();
        let worker = std::thread::spawn(move || decider.decide_bash("rm -rf /"));
        std::thread::sleep(std::time::Duration::from_millis(100));
        g2.resolve(false);
        assert!(!worker.join().unwrap());
    }
}
