//! Driven-port adapter (hexagonal infrastructure ring): executes shell
//! commands on the host OS via `/bin/sh -c`, with a configurable timeout.

use std::process::Stdio;
use std::time::Duration;

use async_trait::async_trait;
use harxes_core_domain::ports::{CommandOutput, ShellError, ShellExitStatus, ShellPort};

/// Adapter that spawns `/bin/sh -c <cmd>` subprocesses with a timeout.
pub struct TokioCommandShell {
    default_timeout: Duration,
}

impl TokioCommandShell {
    pub fn new(timeout_secs: u64) -> Self {
        Self {
            default_timeout: Duration::from_secs(timeout_secs),
        }
    }
}

#[async_trait]
impl ShellPort for TokioCommandShell {
    async fn run_command(&self, working_dir: &str, cmd: &str) -> Result<CommandOutput, ShellError> {
        run_shell(self.default_timeout, working_dir, cmd).await
    }
}

/// RAII guard that SIGKILLs a process group when it is dropped. Combined with a
/// dedicated process group per command, this ensures a cancelled or timed-out
/// turn also reaps any grandchildren the shell spawned — not just the shell
/// leader itself.
struct KillGroupOnDrop {
    pgid: i32,
}

impl KillGroupOnDrop {
    fn from_child(child: &tokio::process::Child) -> Self {
        // With process_group(0) the leader's pid equals its process-group id.
        Self {
            pgid: child.id().map(|p| p as i32).unwrap_or(-1),
        }
    }
    fn kill(&self) {
        if self.pgid > 0 {
            unsafe { libc::kill(-self.pgid, libc::SIGKILL) };
        }
    }
}

impl Drop for KillGroupOnDrop {
    fn drop(&mut self) {
        // Best-effort cleanup of any still-running process group (e.g. when the
        // awaiting future is cancelled mid-flight).
        self.kill();
    }
}

async fn run_shell(
    timeout: Duration,
    working_dir: &str,
    cmd: &str,
) -> Result<CommandOutput, ShellError> {
    let mut command = tokio::process::Command::new("/bin/sh");
    command
        .arg("-c")
        .arg(cmd)
        .current_dir(working_dir)
        // Give the command its own process group so we can reap it and any
        // descendants when the turn is cancelled or times out.
        .process_group(0)
        .kill_on_drop(true)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let child = command
        .spawn()
        .map_err(|e| ShellError::Spawn(e.to_string()))?;
    let guard = KillGroupOnDrop::from_child(&child);

    let output = match tokio::time::timeout(timeout, child.wait_with_output()).await {
        Ok(res) => {
            // Completed normally (or errored); the guard's Drop is now a no-op
            // because the child already exited, but kill() is harmless anyway.
            res.map_err(|e| ShellError::Io(e.to_string()))?
        }
        Err(_) => {
            // Timed out: kill the whole process group before reporting.
            guard.kill();
            return Err(ShellError::Io("command timed out".to_string()));
        }
    };

    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    let exit_status = match output.status.code() {
        Some(0) => ShellExitStatus::Success,
        Some(c) => ShellExitStatus::Failure(c),
        None => ShellExitStatus::Failure(-1),
    };
    Ok(CommandOutput {
        stdout,
        stderr,
        exit_status,
    })
}


#[cfg(test)]
mod tests {
    use super::*;
    use harxes_core_domain::ports::ShellExitStatus;

    #[tokio::test]
    async fn runs_command_and_captures_output() {
        let shell = TokioCommandShell::new(10);
        let out = shell.run_command(".", "echo hello && echo world >&2").await.unwrap();
        assert_eq!(out.stdout.trim(), "hello");
        assert_eq!(out.stderr.trim(), "world");
        assert_eq!(out.exit_status, ShellExitStatus::Success);
    }

    #[tokio::test]
    async fn times_out_and_reports_error() {
        let shell = TokioCommandShell::new(1);
        let res = shell.run_command(".", "sleep 30").await;
        assert!(matches!(res, Err(ShellError::Io(_))));
    }

    #[tokio::test]
    async fn explicit_timeout_reaps_grandchild_group() {
        // A short timeout guarantees a mid-flight group that must be reaped.
        let timeout = Duration::from_millis(200);
        let cwd = ".".to_string();
        let child = tokio::process::Command::new("/bin/sh")
            .arg("-c")
            .arg("sleep 30 & wait")
            .current_dir(cwd)
            .process_group(0)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .spawn()
            .unwrap();
        let pgid = child.id().unwrap() as i32;
        let guard = KillGroupOnDrop::from_child(&child);
        let _ = tokio::time::timeout(timeout, child.wait_with_output()).await;
        guard.kill();
        // Give the OS a moment to reap.
        tokio::time::sleep(Duration::from_millis(100)).await;
        // The whole group (leader + grandchild sleep) must be gone: kill(2) with
        // signal 0 probes existence and should now fail with ESRCH.
        let alive = unsafe { libc::kill(-pgid, 0) };
        assert_ne!(alive, 0, "process group {pgid} should have been reaped");
    }
}
