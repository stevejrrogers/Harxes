//! Driven-port adapter (hexagonal infrastructure ring): reads and writes files
//! on the host filesystem for the [`FileSystemPort`].

pub mod context;
pub mod skills;

use async_trait::async_trait;
use harxes_core_domain::ports::{FileSystemPort, FsError, GlobOptions, GrepMatch};
use std::path::{Path, PathBuf};

/// Directories always skipped when scanning for `glob`/`grep` results: version
/// control, build artifacts, dependencies and temp dirs.
const SKIP_DIRS: &[&str] = &[
    ".git",
    ".hg",
    ".svn",
    "target",
    "node_modules",
    "vendor",
    "dist",
    "build",
    ".next",
    ".cache",
];

/// Simple host-filesystem implementation of [`FileSystemPort`]. All relative
/// paths (reads, writes, and glob/grep roots) resolve against [`Self::root`],
/// which defaults to the process working directory. An embedding host anchors
/// a run to its worktree with [`HostFileSystem::rooted`] instead of relying on
/// the process CWD (which, in-process, is the host's, not the run's).
#[derive(Debug, Clone)]
pub struct HostFileSystem {
    root: PathBuf,
}

impl Default for HostFileSystem {
    fn default() -> Self {
        Self {
            root: PathBuf::from("."),
        }
    }
}

impl HostFileSystem {
    /// A filesystem rooted at `root`; relative paths resolve against it.
    pub fn rooted(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Resolve a caller path against the root (absolute paths pass through).
    fn resolve(&self, path: &str) -> PathBuf {
        let p = Path::new(path);
        if p.is_absolute() {
            p.to_path_buf()
        } else {
            self.root.join(p)
        }
    }

    /// Recursively walk `root` yielding every file whose `Pattern` matches,
    /// skipping hidden dirs and well-known build/vendor dirs.
    fn collect_matches(
        root: &Path,
        relative_parent: &Path,
        matcher: &dyn Fn(&str) -> bool,
        max_depth: Option<usize>,
        out: &mut Vec<String>,
    ) {
        let entries = match std::fs::read_dir(root) {
            Ok(e) => e,
            Err(_) => return,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') && !name.contains('.') {
                // Skip dot-directories (e.g. .git) and dot-files (.env etc.)
                continue;
            }
            let rel = relative_parent.join(&name);
            let rel_str = rel.to_string_lossy().replace('\\', "/");
            if path.is_dir() {
                // Skip build/vendor dirs.
                if SKIP_DIRS.contains(&name.as_str()) {
                    continue;
                }
                let depth = relative_parent.components().count();
                if let Some(max) = max_depth {
                    if depth >= max {
                        continue;
                    }
                }
                Self::collect_matches(&path, &rel, matcher, max_depth, out);
            } else if matcher(&rel_str) {
                out.push(rel_str);
            }
        }
    }
}

#[async_trait]
impl FileSystemPort for HostFileSystem {
    async fn read(&self, path: &str) -> Result<String, FsError> {
        match tokio::fs::read_to_string(self.resolve(path)).await {
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
        let full = self.resolve(path);
        if let Some(parent) = full.parent() {
            let _ = tokio::fs::create_dir_all(parent).await;
        }
        tokio::fs::write(full, content)
            .await
            .map_err(|e| FsError::Io(e.to_string()))
    }

    async fn glob(&self, pattern: &str, options: &GlobOptions) -> Vec<String> {
        // Compile the pattern; on failure return an empty list (caller formats).
        let matcher = match glob::Pattern::new(pattern) {
            Ok(p) => p,
            Err(_) => return Vec::new(),
        };
        // The `glob` crate matches against full paths. Restrict matching to the
        // path relative to the root only.
        let root = self.root.clone();
        let mut out = Vec::new();
        let ignore: Vec<String> = options.ignore.iter().flat_map(|s| s.split(',')).map(|s| s.trim().to_string()).collect();
        Self::collect_matches(
            &root,
            Path::new(""),
            &|rel: &str| {
                if ignore.iter().any(|pat| glob::Pattern::new(pat).map(|p| p.matches(rel)).unwrap_or(false)) {
                    return false;
                }
                matcher.matches(rel)
            },
            options.max_depth,
            &mut out,
        );
        out.sort();
        out
    }

    async fn grep(
        &self,
        needle: &str,
        pattern: &str,
        max_matches: usize,
    ) -> Result<Vec<GrepMatch>, FsError> {
        // If `needle` is a valid regex use it, else fall back to a literal
        // substring search (so plain words still work).
        let regex = regex::Regex::new(needle).ok();
        let matcher = match glob::Pattern::new(pattern) {
            Ok(p) => p,
            Err(_) => return Ok(Vec::new()),
        };
        let root = self.root.clone();
        let mut matches = Vec::new();
        let mut pending: Vec<PathBuf> = vec![root.clone()];
        let mut scanned: std::collections::HashSet<PathBuf> = std::collections::HashSet::new();
        while let Some(dir) = pending.pop() {
            if !scanned.insert(dir.clone()) {
                continue;
            }
            let entries = match std::fs::read_dir(&dir) {
                Ok(e) => e,
                Err(_) => continue,
            };
            for entry in entries.flatten() {
                let path = entry.path();
                let name = entry.file_name().to_string_lossy().into_owned();
                if name.starts_with('.') {
                    continue;
                }
                if path.is_dir() {
                    if !SKIP_DIRS.contains(&name.as_str()) {
                        pending.push(path);
                    }
                    continue;
                }
                let rel_path = path.strip_prefix(&root).unwrap_or(&path);
                let rel = rel_path.to_string_lossy().replace('\\', "/");
                let rel_trim = rel.trim_start_matches("./");
                if !matcher.matches(rel_trim) {
                    continue;
                }
                // Read and search the text file (skip binary-looking content).
                let content = match std::fs::read_to_string(&path) {
                    Ok(c) => c,
                    Err(_) => continue,
                };
                for (idx, line) in content.lines().enumerate() {
                    let hit = match &regex {
                        Some(re) => re.is_match(line),
                        None => line.contains(needle),
                    };
                    if hit {
                        matches.push(GrepMatch {
                            path: rel_trim.to_string(),
                            line_number: (idx + 1) as u64,
                            line: line.to_string(),
                        });
                        if matches.len() >= max_matches {
                            return Ok(matches);
                        }
                    }
                }
            }
        }
        Ok(matches)
    }

    async fn replace(
        &self,
        path: &str,
        old_string: &str,
        new_string: &str,
    ) -> Result<usize, FsError> {
        let full = self.resolve(path);
        let content = match std::fs::read_to_string(&full) {
            Ok(c) => c,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(FsError::NotFound(path.to_string()))
            }
            Err(e) => return Err(FsError::Io(e.to_string())),
        };
        if old_string.is_empty() {
            return Err(FsError::Io(
                "edit failed: old_string must not be empty".to_string(),
            ));
        }
        let occurrences = content.match_indices(old_string).count();
        if occurrences == 0 {
            return Err(FsError::Io(format!(
                "edit failed: old_string not found in {path}"
            )));
        }
        if occurrences > 1 {
            return Err(FsError::Io(format!(
                "edit failed: old_string matches {occurrences} times in {path}; \
                 include more surrounding context to make the edit unambiguous"
            )));
        }
        let start = content
            .find(old_string)
            .ok_or_else(|| FsError::Io("edit failed: locate error".to_string()))?;
        let mut out = String::with_capacity(content.len() + new_string.len());
        out.push_str(&content[..start]);
        out.push_str(new_string);
        out.push_str(&content[start + old_string.len()..]);
        std::fs::write(path, &out).map_err(|e| FsError::Io(e.to_string()))?;
        Ok(new_string.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("harxes-fs-test-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[tokio::test]
    async fn rooted_anchors_read_write_and_glob() {
        let dir = std::env::temp_dir().join(format!("hx-rooted-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::write(dir.join("src/main.rs"), "fn main() {}").unwrap();
        let fs = HostFileSystem::rooted(&dir);
        // Relative read resolves against root, not the process CWD.
        assert_eq!(fs.read("src/main.rs").await.unwrap(), "fn main() {}");
        // Relative write lands under root (creating parent dirs).
        fs.write("out/new.txt", "hi").await.unwrap();
        assert_eq!(std::fs::read_to_string(dir.join("out/new.txt")).unwrap(), "hi");
        // Glob walks from root and returns root-relative paths.
        let files = fs.glob("**/*.rs", &GlobOptions::default()).await;
        assert!(files.iter().any(|f| f == "src/main.rs"), "{files:?}");
        // Grep too.
        let hits = fs.grep("fn main", "**/*.rs", 10).await.unwrap();
        assert_eq!(hits[0].path, "src/main.rs");
    }

    #[tokio::test]
    async fn glob_finds_nested_sources_and_skips_target() {
        let root = tmpdir("glob");
        std::fs::create_dir_all(root.join("src/deep")).unwrap();
        std::fs::create_dir_all(root.join("target")).unwrap();
        std::fs::write(root.join("src/main.rs"), "").unwrap();
        std::fs::write(root.join("src/deep/util.rs"), "").unwrap();
        std::fs::write(root.join("target/hidden.rs"), "").unwrap();

        // Run from inside the temp dir.
        let prev = std::env::current_dir().unwrap();
        std::env::set_current_dir(&root).unwrap();
        let fs = HostFileSystem::default();
        let files = fs.glob("**/*.rs", &GlobOptions::default()).await;
        std::env::set_current_dir(prev).unwrap();

        assert!(files.iter().any(|f| f == "src/main.rs"));
        assert!(files.iter().any(|f| f == "src/deep/util.rs"));
        assert!(!files.iter().any(|f| f.contains("target")), "target must be skipped: {files:?}");
    }

    #[tokio::test]
    async fn replace_is_unique_and_preserves_rest() {
        let root = tmpdir("replace");
        let file = root.join("a.txt");
        std::fs::write(&file, "hello world hello").unwrap();
        let fs = HostFileSystem::default();
        // "hello" appears twice -> must fail.
        assert!(fs.replace(file.to_str().unwrap(), "hello", "bye").await.is_err());
        // unique substring -> succeeds.
        let n = fs.replace(file.to_str().unwrap(), "world", "Rust").await.unwrap();
        assert!(n > 0);
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "hello Rust hello");
    }
}
