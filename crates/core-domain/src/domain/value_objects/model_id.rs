use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelId(String);

impl ModelId {
    pub fn new(id: impl Into<String>) -> Result<Self, InvalidModelId> {
        let id = id.into();
        if id.trim().is_empty() {
            return Err(InvalidModelId);
        }
        Ok(Self(id))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone)]
pub struct InvalidModelId;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_valid_model_id() {
        let id = ModelId::new("claude-sonnet-4-5").unwrap();
        assert_eq!(id.as_str(), "claude-sonnet-4-5");
    }

    #[test]
    fn rejects_empty_model_id() {
        assert!(ModelId::new("").is_err());
        assert!(ModelId::new("   ").is_err());
    }

    #[test]
    fn trims_but_stores_original() {
        let id = ModelId::new("gpt-4o").unwrap();
        assert_eq!(id.as_str(), "gpt-4o");
    }
}
