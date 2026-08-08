//! Driven port: lets an external caller decide whether a risky operation
//! (writing outside the allow list, or running a dangerous shell command)
//! may proceed. The CLI wires this to an interactive allow/deny prompt.

/// Returns whether an operation should be allowed.
pub trait PermissionDecider: Send + Sync {
    /// Called when a Write is not covered by the policy allow list.
    fn decide_write(&self, path: &str) -> bool;
    /// Called before running a command flagged as potentially dangerous.
    fn decide_bash(&self, command: &str) -> bool;
}
