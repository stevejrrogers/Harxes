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

async fn run_shell(
    timeout: Duration,
    working_dir: &str,
    cmd: &str,
) -> Result<CommandOutput, ShellError> {
    let child = tokio::process::Command::new("/bin/sh")
        .arg("-c")
        .arg(cmd)
        .current_dir(working_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| ShellError::Spawn(e.to_string()))?;

    let output = match tokio::time::timeout(timeout, child.wait_with_output()).await {
        Ok(res) => res.map_err(|e| ShellError::Io(e.to_string()))?,
        Err(_) => return Err(ShellError::Io("command timed out".to_string())),
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
