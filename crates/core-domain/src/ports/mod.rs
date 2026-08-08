#![allow(dead_code)]

mod config_store;
mod file_system;
mod llm;
mod secrets_vault;
mod session;
mod shell;
mod tool_observer;

// Re-exports form the hexagonal "driven interface" surface that infrastructure
// adapters implement and application services consume.
pub use config_store::{ConfigStoreError, ConfigStorePort, HarxesConfig, ProviderConfig};
pub use file_system::{FileSystemPort, FsError};
pub use llm::{AgentResponse, LlmError, LlmPort};
pub use secrets_vault::{keys, SecretsVaultPort};
pub use session::{SessionRecord, SessionStoreError, SessionStorePort};
pub use shell::{CommandOutput, ShellError, ShellExitStatus, ShellPort};
pub use tool_observer::ToolObserver;
