use async_trait::async_trait;

/// Exit status of a shell command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShellExitStatus {
    Success,
    Failure(i32),
}

#[derive(Debug, Clone)]
pub struct CommandOutput {
    pub stdout: String,
    pub stderr: String,
    pub exit_status: ShellExitStatus,
}

impl CommandOutput {
    pub fn is_success(&self) -> bool {
        matches!(self.exit_status, ShellExitStatus::Success)
    }
}

#[derive(Debug)]
pub enum ShellError {
    Spawn(String),
    Io(String),
}

impl std::fmt::Display for ShellError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Spawn(msg) => write!(f, "failed to spawn command: {msg}"),
            Self::Io(msg) => write!(f, "io error: {msg}"),
        }
    }
}

impl std::error::Error for ShellError {}

/// Driven port (hexagonal): infrastructure implements this by spawning OS
/// processes. Domain/application never touch the process directly.
#[async_trait]
pub trait ShellPort: Send + Sync {
    async fn run_command(&self, working_dir: &str, cmd: &str) -> Result<CommandOutput, ShellError>;
}
