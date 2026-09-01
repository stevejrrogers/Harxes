#![allow(dead_code)]

mod config_store;
mod dynamic_tools;
mod file_system;
mod llm;
mod permission_decider;
mod secrets_vault;
mod session;
mod shell;
mod tool_observer;
mod web;

// Re-exports form the hexagonal "driven interface" surface that infrastructure
// adapters implement and application services consume.
pub use config_store::{
    ConfigStoreError, ConfigStorePort, HarxesConfig, HookRule, HooksConfig, McpServerConfig,
    ModelPricing, ProviderConfig,
};
pub use dynamic_tools::DynamicToolPort;
pub use file_system::{FileSystemPort, FsError, GlobOptions, GrepMatch};
pub use llm::{AgentResponse, LlmError, LlmPort, StreamEvent, StreamSink};
pub use permission_decider::PermissionDecider;
pub use secrets_vault::{keys, SecretsVaultPort};
pub use session::{SessionRecord, SessionStoreError, SessionStorePort};
pub use shell::{CommandOutput, ShellError, ShellExitStatus, ShellPort};
pub use tool_observer::ToolObserver;
pub use web::WebPort;
