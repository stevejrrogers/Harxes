use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProviderId(String);

impl ProviderId {
    pub fn new(id: impl Into<String>) -> Result<Self, InvalidProviderId> {
        let id = id.into();
        if id.trim().is_empty() {
            return Err(InvalidProviderId);
        }
        Ok(Self(id))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone)]
pub struct InvalidProviderId;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_valid_provider_id() {
        let id = ProviderId::new("anthropic").unwrap();
        assert_eq!(id.as_str(), "anthropic");
    }

    #[test]
    fn rejects_empty_provider_id() {
        assert!(ProviderId::new("").is_err());
        assert!(ProviderId::new("  ").is_err());
    }

    #[test]
    fn equality_and_hash() {
        let a = ProviderId::new("openai").unwrap();
        let b = ProviderId::new("openai").unwrap();
        assert_eq!(a, b);
    }
}
