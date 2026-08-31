//! File-permission policy used to gate destructive tool operations (write,
//! delete) against a configured allow/deny list of path globs.

use serde::{Deserialize, Serialize};

/// A filesystem operation subject to permission checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FsOp {
    Read,
    Write,
}

/// White/black-list policy over path globs. Deny rules take precedence.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PermissionPolicy {
    #[serde(default)]
    pub allow_globs: Vec<String>,
    #[serde(default)]
    pub deny_globs: Vec<String>,
}

impl PermissionPolicy {
    /// Read ops are allowed unless denied; mutating ops (Write) need an allow
    /// glob match and no deny glob match.
    pub fn permits(&self, op: FsOp, path: &str) -> bool {
        let denied = self.deny_globs.iter().any(|g| matches_path(g, path));
        if denied {
            return false;
        }
        match op {
            FsOp::Read => true,
            FsOp::Write => self.allow_globs.iter().any(|g| matches_path(g, path)),
        }
    }
}

/// Minimal glob matcher supporting a trailing `/**` (whole subtree), a trailing
/// `/*` (direct children), a leading/trailing bare `*`, and exact/prefix match.
fn matches_path(glob: &str, path: &str) -> bool {
    if let Some(rest) = glob.strip_suffix("/**") {
        return path == rest || path.starts_with(&format!("{}/", rest));
    }
    if let Some(prefix) = glob.strip_suffix("/*") {
        if let Some(idx) = path.rfind('/') {
            return &path[..idx] == prefix;
        }
        return false;
    }
    if glob.contains('*') {
        return star(glob.as_bytes(), path.as_bytes());
    }
    glob == path || path.starts_with(&format!("{}/", glob))
}

/// Simple `*` wildcard DP matcher (matches any run of chars, may cross '/').
fn star(p: &[u8], t: &[u8]) -> bool {
    let np = p.len();
    let nt = t.len();
    let mut dp = vec![vec![false; nt + 1]; np + 1];
    dp[0][0] = true;
    for i in 1..=np {
        if p[i - 1] == b'*' {
            dp[i][0] = dp[i - 1][0];
        }
    }
    for i in 1..=np {
        for j in 1..=nt {
            if p[i - 1] == b'*' {
                dp[i][j] = dp[i - 1][j] || dp[i][j - 1];
            } else {
                dp[i][j] = dp[i - 1][j - 1] && (p[i - 1] == t[j - 1]);
            }
        }
    }
    dp[np][nt]
}

/// Verdict for a shell command against the configured [`CommandPolicy`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandVerdict {
    /// Explicitly allowed: run without prompting, even if heuristically risky.
    Allow,
    /// Explicitly denied: refuse without prompting.
    Deny,
    /// No rule matched: fall back to the default gating (prompt when risky).
    Ask,
}

/// User-configured allow/deny rules for shell commands, matched against the
/// full command string. Patterns support `*` wildcards (e.g. `cargo *`,
/// `git status`); a pattern without `*` also matches as a word-boundary
/// prefix (`git` matches `git log` but not `gitk`). Deny takes precedence.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CommandPolicy {
    #[serde(default)]
    pub allow: Vec<String>,
    #[serde(default)]
    pub deny: Vec<String>,
}

impl CommandPolicy {
    pub fn evaluate(&self, cmd: &str) -> CommandVerdict {
        let c = cmd.trim();
        if self.deny.iter().any(|p| matches_command(p, c)) {
            return CommandVerdict::Deny;
        }
        if self.allow.iter().any(|p| matches_command(p, c)) {
            return CommandVerdict::Allow;
        }
        CommandVerdict::Ask
    }
}

fn matches_command(pattern: &str, cmd: &str) -> bool {
    let p = pattern.trim();
    if p.is_empty() {
        return false;
    }
    if p.contains('*') {
        return star(p.as_bytes(), cmd.as_bytes());
    }
    cmd == p || cmd.starts_with(&format!("{p} "))
}

#[cfg(test)]
mod command_tests {
    use super::*;

    #[test]
    fn wildcard_and_prefix_matching() {
        let pol = CommandPolicy {
            allow: vec!["cargo *".into(), "git status".into(), "ls".into()],
            deny: vec!["rm -rf /*".into(), "sudo *".into()],
        };
        assert_eq!(pol.evaluate("cargo test --workspace"), CommandVerdict::Allow);
        assert_eq!(pol.evaluate("git status"), CommandVerdict::Allow);
        assert_eq!(pol.evaluate("ls -la"), CommandVerdict::Allow);
        assert_eq!(pol.evaluate("git push"), CommandVerdict::Ask);
        assert_eq!(pol.evaluate("sudo rm x"), CommandVerdict::Deny);
        assert_eq!(pol.evaluate("rm -rf /etc"), CommandVerdict::Deny);
        // word-boundary: "ls" must not match "lsof"
        assert_eq!(pol.evaluate("lsof -i"), CommandVerdict::Ask);
    }

    #[test]
    fn deny_beats_allow() {
        let pol = CommandPolicy {
            allow: vec!["git *".into()],
            deny: vec!["git push *".into(), "git push".into()],
        };
        assert_eq!(pol.evaluate("git log"), CommandVerdict::Allow);
        assert_eq!(pol.evaluate("git push origin main"), CommandVerdict::Deny);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_allowed_by_default() {
        let p = PermissionPolicy::default();
        assert!(p.permits(FsOp::Read, "/etc/passwd"));
    }

    #[test]
    fn write_denied_without_allow() {
        let p = PermissionPolicy::default();
        assert!(!p.permits(FsOp::Write, "/tmp/x.txt"));
    }

    #[test]
    fn write_allowed_in_dir_subtree() {
        let p = PermissionPolicy {
            allow_globs: vec!["/workspace/**".into()],
            deny_globs: vec![],
        };
        assert!(p.permits(FsOp::Write, "/workspace/a/b/c.rs"));
        assert!(!p.permits(FsOp::Write, "/etc/x"));
    }

    #[test]
    fn deny_overrides_allow() {
        let p = PermissionPolicy {
            allow_globs: vec!["/**".into()],
            deny_globs: vec!["**/*.env".into()],
        };
        assert!(p.permits(FsOp::Write, "/ok.txt"));
        assert!(!p.permits(FsOp::Write, "/secret.env"));
        // exact prefix match
        assert!(matches_path("/workspace", "/workspace/app/src/lib.rs"));
    }
}
