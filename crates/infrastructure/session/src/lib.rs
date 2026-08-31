//! Driven-port adapter (hexagonal infrastructure ring): persists user
//! preferences to a JSON file on disk (`~/.harxes/config.json` by default).

use std::fs;
use std::path::{Path, PathBuf};

use harxes_core_domain::ports::{
    ConfigStoreError, ConfigStorePort, HarxesConfig, SessionRecord, SessionStoreError,
    SessionStorePort,
};

/// Stores configuration as pretty-printed JSON at `<config_dir>/config.json`.
#[derive(Debug, Clone)]
pub struct JsonConfigStore {
    config_dir: PathBuf,
}

impl JsonConfigStore {
    pub fn new(config_dir: impl Into<PathBuf>) -> Self {
        Self {
            config_dir: config_dir.into(),
        }
    }

    fn config_path(&self) -> PathBuf {
        self.config_dir.join("config.json")
    }
}

impl ConfigStorePort for JsonConfigStore {
    fn load(&self) -> HarxesConfig {
        let path = self.config_path();
        match fs::read_to_string(&path) {
            Ok(raw) => serde_json::from_str(&raw).unwrap_or_default(),
            Err(_) => HarxesConfig::default(),
        }
    }

    fn save(&self, config: &HarxesConfig) -> Result<(), ConfigStoreError> {
        let path = self.config_path();
        if let Some(parent) = Path::new(&path).parent() {
            fs::create_dir_all(parent).map_err(|e| ConfigStoreError::Io(e.to_string()))?;
        }
        let raw = serde_json::to_string_pretty(config)
            .map_err(|e| ConfigStoreError::Io(e.to_string()))?;
        fs::write(&path, raw).map_err(|e| ConfigStoreError::Io(e.to_string()))
    }
}

/// Stores agent sessions as JSON files at `<config_dir>/sessions/<id>.json`.
#[derive(Debug, Clone)]
pub struct JsonSessionStore {
    config_dir: PathBuf,
}

impl JsonSessionStore {
    pub fn new(config_dir: impl Into<PathBuf>) -> Self {
        Self {
            config_dir: config_dir.into(),
        }
    }

    fn path_for(&self, id: &str) -> PathBuf {
        self.config_dir
            .join("sessions")
            .join(format!("{}.json", id))
    }

    /// List all saved session ids (sorted by newest mtime first).
    pub fn list(&self) -> Vec<String> {
        let dir = self.config_dir.join("sessions");
        let mut out = Vec::new();
        if let Ok(rd) = fs::read_dir(&dir) {
            for e in rd.flatten() {
                let name = e.file_name().to_string_lossy().into_owned();
                if let Some(id) = name.strip_suffix(".json") {
                    out.push(id.to_string());
                }
            }
        }
        out.sort_by(|a, b| {
            let pa = self.path_for(a);
            let pb = self.path_for(b);
            let mt_a = fs::metadata(&pa).and_then(|m| m.modified()).ok();
            let mt_b = fs::metadata(&pb).and_then(|m| m.modified()).ok();
            mt_b.cmp(&mt_a)
        });
        out
    }
}

impl SessionStorePort for JsonSessionStore {
    fn save(&self, session: &SessionRecord) -> Result<(), SessionStoreError> {
        let path = self.path_for(&session.id);
        if let Some(parent) = Path::new(&path).parent() {
            fs::create_dir_all(parent).map_err(|e| SessionStoreError::Io(e.to_string()))?;
        }
        let raw = serde_json::to_string_pretty(session)
            .map_err(|e| SessionStoreError::Io(e.to_string()))?;
        fs::write(&path, raw).map_err(|e| SessionStoreError::Io(e.to_string()))
    }
    fn load(&self, id: &str) -> Option<SessionRecord> {
        let raw = fs::read_to_string(self.path_for(id)).ok()?;
        serde_json::from_str(&raw).ok()
    }
}

#[cfg(test)]
mod session_tests {
    use super::*;
    use harxes_core_domain::domain::value_objects::{Message, Role};
    #[test]
    fn json_session_round_trip() {
        let dir = std::env::temp_dir().join(format!("hx-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let store = JsonSessionStore::new(dir);
        let rec = SessionRecord {
            id: "s1".to_string(),
            created_at: String::new(),
            transcript: vec![
                Message::new(Role::System, "sys"),
                Message::new(Role::User, "hi"),
            ],
            todos: vec![],
        };
        store.save(&rec).unwrap();
        let loaded = store.load("s1").expect("should load");
        assert_eq!(loaded.transcript.len(), 2);
    }
}
