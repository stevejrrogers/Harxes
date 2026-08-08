/// Driven port (hexagonal): secure storage for provider API keys. Concrete
/// implementors read from keychain / env `.env`, never hardcode secrets.
pub trait SecretsVaultPort: Send + Sync {
    fn get_secret(&self, key: &str) -> Option<String>;
}

/// Well-known secret keys.
pub mod keys {
    pub const ANTHROPIC_API_KEY: &str = "ANTHROPIC_API_KEY";
    pub const OPENAI_API_KEY: &str = "OPENAI_API_KEY";
}
