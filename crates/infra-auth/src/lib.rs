//! Driven-port adapter (hexagonal infrastructure ring): provides secrets
//! (API keys) to the application from environment variables / keychain.
//!
//! Skeleton status: reads from process environment only. A future variant may
//! read from macOS Keychain or `~/.harxes/credentials.json`.

use std::env;

use harxes_core_domain::ports::{keys, SecretsVaultPort};

/// Reads secrets from environment variables such as `ANTHROPIC_API_KEY`.
#[derive(Debug, Clone, Default)]
pub struct EnvSecretsVault;

impl SecretsVaultPort for EnvSecretsVault {
    fn get_secret(&self, key: &str) -> Option<String> {
        env::var(key).ok()
    }
}

impl EnvSecretsVault {
    /// Convenience accessors for well-known keys.
    pub fn anthropic_key(&self) -> Option<String> {
        self.get_secret(keys::ANTHROPIC_API_KEY)
    }

    pub fn openai_key(&self) -> Option<String> {
        self.get_secret(keys::OPENAI_API_KEY)
    }
}
