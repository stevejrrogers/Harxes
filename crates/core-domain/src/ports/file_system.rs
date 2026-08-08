use async_trait::async_trait;

#[derive(Debug)]
pub enum FsError {
    NotFound(String),
    PermissionDenied(String),
    Io(String),
}

impl std::fmt::Display for FsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(p) => write!(f, "not found: {p}"),
            Self::PermissionDenied(p) => write!(f, "permission denied: {p}"),
            Self::Io(msg) => write!(f, "io error: {msg}"),
        }
    }
}

impl std::error::Error for FsError {}

/// Driven port (hexagonal): filesystem abstraction so domain logic can be
/// tested with in-memory or virtual filesystems.
#[async_trait]
pub trait FileSystemPort: Send + Sync {
    async fn read(&self, path: &str) -> Result<String, FsError>;
    async fn write(&self, path: &str, content: &str) -> Result<(), FsError>;
}
