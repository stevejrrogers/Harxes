//! Persistent agent context: a per-session directory plus shared workspace
//! memory that the agent can write and re-read across turns.

use std::fs;
use std::path::PathBuf;
/// Root of the workspace context. Everything lives under `<root>/agents`.
#[derive(Debug, Clone)]
pub struct ContextStore {
    root: PathBuf,
}
impl ContextStore {
    /// Create a store rooted at `root` (e.g. `<project>/.harxes`).
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }
    fn agents_dir(&self) -> PathBuf {
        self.root.join("agents")
    }
    fn session_dir(&self, id: &str) -> PathBuf {
        self.agents_dir().join(id)
    }
    fn session_ctx(&self, id: &str) -> PathBuf {
        self.session_dir(id).join("CONTEXT.md")
    }
    /// The shared workspace memory file (project-wide, not per-session).
    pub fn notes_path(&self) -> PathBuf {
        self.agents_dir().join("NOTES.md")
    }
    /// Ensure the agent context directory for a session exists. Created only
    /// when the agent actually starts working (a turn/tool runs).
    pub fn ensure_session(&self, id: &str) -> std::io::Result<PathBuf> {
        let dir = self.session_dir(id);
        fs::create_dir_all(&dir)?;
        Ok(dir)
    }
    /// Read the current session CONTEXT.md, or an empty string if absent.
    pub fn read_session_context(&self, id: &str) -> String {
        fs::read_to_string(self.session_ctx(id)).unwrap_or_default()
    }
    /// Append a line to the session CONTEXT.md, creating the dir if needed.
    pub fn remember_session(&self, id: &str, note: &str) -> std::io::Result<()> {
        self.ensure_session(id)?;
        let mut body = self.read_session_context(id);
        if !body.is_empty() && !body.ends_with('\n') {
            body.push('\n');
        }
        body.push_str(note);
        body.push('\n');
        fs::write(self.session_ctx(id), body)
    }
    /// Read shared workspace memory (NOTES.md), or empty string if absent.
    pub fn read_notes(&self) -> String {
        fs::read_to_string(self.notes_path()).unwrap_or_default()
    }
    /// Append a line to shared workspace memory.
    pub fn remember_workspace(&self, note: &str) -> std::io::Result<()> {
        self.ensure_agents_dir()?;
        let mut body = self.read_notes();
        if !body.is_empty() && !body.ends_with('\n') {
            body.push('\n');
        }
        body.push_str(note);
        body.push('\n');
        fs::write(self.notes_path(), body)
    }
    fn ensure_agents_dir(&self) -> std::io::Result<()> {
        fs::create_dir_all(self.agents_dir())
    }
    /// Read the project guide from the project root (parent of .harxes).
    /// First match wins: HARXES.md, AGENTS.md, CLAUDE.md.
    pub fn read_agents_md(&self) -> String {
        let Some(dir) = self.root.parent() else {
            return String::new();
        };
        for name in ["HARXES.md", "AGENTS.md", "CLAUDE.md"] {
            if let Ok(body) = std::fs::read_to_string(dir.join(name)) {
                if !body.trim().is_empty() {
                    return body;
                }
            }
        }
        String::new()
    }

    /// Read the user-global guide (`~/.harxes/HARXES.md`), applying to every
    /// project — the analogue of a user-level CLAUDE.md.
    pub fn read_global_guide(&self) -> String {
        let home = std::env::var("HOME").unwrap_or_default();
        if home.is_empty() {
            return String::new();
        }
        std::fs::read_to_string(PathBuf::from(home).join(".harxes").join("HARXES.md"))
            .unwrap_or_default()
    }

    /// Combine session + workspace context into a block for the system prompt.
    pub fn build_context_block(&self, id: &str) -> String {
        let mut out = String::new();
        let global = self.read_global_guide();
        if !global.trim().is_empty() {
            out.push_str("# User guide (~/.harxes/HARXES.md)\n");
            out.push_str(global.trim());
            out.push_str("\n\n");
        }
        let agents_md = self.read_agents_md();
        if !agents_md.trim().is_empty() {
            out.push_str("# Project guide\n");
            out.push_str(agents_md.trim());
            out.push_str("\n\n");
        }
        let sess = self.read_session_context(id);
        if !sess.trim().is_empty() {
            out.push_str("# Session memory\n");
            out.push_str(sess.trim());
            out.push_str("\n\n");
        }
        let notes = self.read_notes();
        if !notes.trim().is_empty() {
            out.push_str("# Workspace memory\n");
            out.push_str(notes.trim());
            out.push('\n');
        }
        out.trim_end().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_root(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("harxes-ctx-test-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn project_guide_prefers_harxes_md() {
        let project = tmp_root("guide");
        let root = project.join(".harxes");
        fs::create_dir_all(&root).unwrap();
        let store = ContextStore::new(&root);
        assert_eq!(store.read_agents_md(), "");

        fs::write(project.join("CLAUDE.md"), "claude guide").unwrap();
        assert_eq!(store.read_agents_md().trim(), "claude guide");
        fs::write(project.join("AGENTS.md"), "agents guide").unwrap();
        assert_eq!(store.read_agents_md().trim(), "agents guide");
        fs::write(project.join("HARXES.md"), "harxes guide").unwrap();
        assert_eq!(store.read_agents_md().trim(), "harxes guide");
        // Empty preferred file falls through to the next candidate.
        fs::write(project.join("HARXES.md"), "  \n").unwrap();
        assert_eq!(store.read_agents_md().trim(), "agents guide");
    }

    #[test]
    fn session_dir_is_lazy_and_memory_persists() {
        let root = tmp_root("session");
        let store = ContextStore::new(&root);
        // No dir until we start working.
        assert!(!store.session_dir("s1").exists());
        assert!(!store.notes_path().exists());

        store.remember_session("s1", "decided to use Rust").unwrap();
        assert!(store.session_dir("s1").is_dir());
        assert!(store.read_session_context("s1").contains("Rust"));

        // Second remember appends, does not clobber.
        store.remember_session("s1", "use axum for http").unwrap();
        let ctx = store.read_session_context("s1");
        assert!(ctx.contains("Rust"));
        assert!(ctx.contains("axum"));
    }

    #[test]
    fn workspace_notes_are_shared_across_sessions() {
        let root = tmp_root("workspace");
        let store = ContextStore::new(&root);
        store.remember_workspace("project goal: harness").unwrap();
        assert_eq!(store.read_notes(), "project goal: harness\n");
    }

    #[test]
    fn context_block_combines_both() {
        let root = tmp_root("block");
        let store = ContextStore::new(&root);
        store
            .remember_session("s9", "user prefers VN output")
            .unwrap();
        store.remember_workspace("deploy target is main").unwrap();
        let block = store.build_context_block("s9");
        assert!(block.contains("# Session memory"));
        assert!(block.contains("# Workspace memory"));
    }
}
