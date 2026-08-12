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

/// A single matching line from a [`FileSystemPort::grep`] search, together
/// with the file it was found in so results can be grouped by path.
#[derive(Debug, Clone)]
pub struct GrepMatch {
    /// Path of the file containing the match.
    pub path: String,
    /// 1-based line number of the match.
    pub line_number: u64,
    /// The full text of the matching line.
    pub line: String,
}

/// Options controlling a directory scan performed by [`FileSystemPort::glob`].
#[derive(Debug, Clone, Default)]
pub struct GlobOptions {
    /// Only include files that are at or under this depth from the root (0 =
    /// unbounded).
    pub max_depth: Option<usize>,
    /// Comma-separated list of gitignore-style patterns to ignore (in addition
    /// to well-known dirs like `.git` and `target`).
    pub ignore: Vec<String>,
}

/// Driven port (hexagonal): filesystem abstraction so domain logic can be
/// tested with in-memory or virtual filesystems.
#[async_trait]
pub trait FileSystemPort: Send + Sync {
    async fn read(&self, path: &str) -> Result<String, FsError>;
    async fn write(&self, path: &str, content: &str) -> Result<(), FsError>;

    /// Return the project files contained in a directory scan whose paths match
    /// `pattern` (a glob such as `**/*.rs`). Implementations must skip the
    /// well-known build/git/vendor directories by default.
    async fn glob(&self, pattern: &str, options: &GlobOptions) -> Vec<String>;

    /// Search files under `pattern` for `needle`, returning matching lines
    /// (capped at `max_matches` to bound output). Implementations must skip the
    /// well-known build/git/vendor directories by default.
    async fn grep(
        &self,
        needle: &str,
        pattern: &str,
        max_matches: usize,
    ) -> Result<Vec<GrepMatch>, FsError>;

    /// Replace the first (and only) occurrence of `old_string` with
    /// `new_string` inside `path`. Fails with [`FsError::Io`] when `old_string`
    /// is absent or matches more than one distinct location, so the model gets
    /// clear feedback to refine its edit.
    async fn replace(
        &self,
        path: &str,
        old_string: &str,
        new_string: &str,
    ) -> Result<usize, FsError>;
}
