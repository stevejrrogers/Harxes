//! Driven-port adapter (hexagonal infrastructure ring): reads and writes files
//! on the host filesystem for the [`FileSystemPort`].

use async_trait::async_trait;
use harxes_core_domain::ports::{FileSystemPort, FsError};

/// Simple host-filesystem implementation of [`FileSystemPort`].
#[derive(Debug, Clone, Default)]
pub struct HostFileSystem;

#[async_trait]
impl FileSystemPort for HostFileSystem {
    async fn read(&self, path: &str) -> Result<String, FsError> {
        match tokio::fs::read_to_string(path).await {
            Ok(content) => Ok(content),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                Err(FsError::NotFound(path.to_string()))
            }
            Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
                Err(FsError::PermissionDenied(path.to_string()))
            }
            Err(e) => Err(FsError::Io(e.to_string())),
        }
    }

    async fn write(&self, path: &str, content: &str) -> Result<(), FsError> {
        tokio::fs::write(path, content)
            .await
            .map_err(|e| FsError::Io(e.to_string()))
    }
}
