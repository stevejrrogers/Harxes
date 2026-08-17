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
        // Prefer the live environment first.
        if let Ok(v) = env::var(key) {
            if !v.is_empty() {
                return Some(v);
            }
        }
        // Fallback: read the value from a shell rc file so a freshly launched
        // shell that has not sourced ~/.zshrc still finds its API keys.
        Self::from_shell_rc(key)
    }
}

impl EnvSecretsVault {
    /// Look up an exported variable in the user's shell rc files.
    fn from_shell_rc(var: &str) -> Option<String> {
        for path in ["~/.zshenv", "~/.zprofile", "~/.zshrc"] {
            let expanded = shellexpand_home(path)?;
            if let Ok(content) = std::fs::read_to_string(&expanded) {
                for line in content.lines() {
                    let line = line.trim();
                    let pattern = format!("export {var}=");
                    if let Some(rest) = line.strip_prefix(&pattern) {
                        if let Some(val) = extract_quoted_value(rest).or_else(|| unquoted(rest)) {
                            if !val.is_empty() && !val.starts_with('#') {
                                return Some(val);
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Convenience accessors for well-known keys.
    pub fn anthropic_key(&self) -> Option<String> {
        self.get_secret(keys::ANTHROPIC_API_KEY)
    }

    pub fn openai_key(&self) -> Option<String> {
        self.get_secret(keys::OPENAI_API_KEY)
    }
}

/// Expand a leading `~` to the user's home directory.
fn shellexpand_home(path: &str) -> Option<std::path::PathBuf> {
    if let Some(rest) = path.strip_prefix("~/") {
        let home = env::var("HOME").ok()?;
        Some(std::path::PathBuf::from(home).join(rest))
    } else {
        Some(std::path::PathBuf::from(path))
    }
}

/// Pull a value out of `export VAR="..."` or `export VAR='...'`.
fn extract_quoted_value(rest: &str) -> Option<String> {
    let rest = rest.trim_start();
    for q in ['"', '\''] {
        if let Some(stripped) = rest.strip_prefix(q) {
            if let Some(idx) = stripped.find(q) {
                return Some(stripped[..idx].to_string());
            }
        }
    }
    None
}

/// Fallback: an unquoted value up to whitespace / trailing comment.
fn unquoted(rest: &str) -> Option<String> {
    let end = rest.find([' ', '\t', '#']).unwrap_or(rest.len());
    let v = rest[..end].trim();
    if v.is_empty() {
        None
    } else {
        Some(v.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quoted_value_extraction() {
        assert_eq!(
            extract_quoted_value("\"sk-abc\"").as_deref(),
            Some("sk-abc")
        );
        assert_eq!(extract_quoted_value("'xyz'").as_deref(), Some("xyz"));
        assert_eq!(unquoted("bare123 # comment").as_deref(), Some("bare123"));
        assert_eq!(unquoted("   "), None);
    }

    #[test]
    fn secret_falls_back_to_shell_rc() {
        // Point HOME at a fake dir with an rc file so we don't touch real env.
        let home = std::env::temp_dir().join(format!("harxes-secret-test-{}", std::process::id()));
        std::fs::create_dir_all(&home).unwrap();
        std::fs::write(home.join(".zshrc"), "export TEST_KEY=\"sk-from-rc\"\n").unwrap();
        let _guard = EnvGuard {
            old: env::var_os("HOME"),
        };
        env::set_var("HOME", &home);
        // Ensure not present in env.
        env::remove_var("TEST_KEY");
        let vault = EnvSecretsVault;
        assert_eq!(vault.get_secret("TEST_KEY").as_deref(), Some("sk-from-rc"));
    }

    struct EnvGuard {
        old: Option<std::ffi::OsString>,
    }
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.old {
                Some(v) => env::set_var("HOME", v),
                None => env::remove_var("HOME"),
            }
            env::remove_var("TEST_KEY");
        }
    }
}
