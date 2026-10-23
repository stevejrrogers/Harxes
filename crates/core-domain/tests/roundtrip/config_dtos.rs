//! Round-trip tests for the configuration / session DTOs (CXA-B305a, group 2):
//! the Harxes config graph plus permission policy, model identity and session
//! record — everything serialized to/from `config.toml`-backed JSON surfaces.

use std::collections::BTreeMap;

use harxes_core_domain::domain::value_objects::{
    ModelId, PermissionPolicy, ProviderId,
};
use harxes_core_domain::ports::{HarxesConfig, SessionRecord};

use crate::expect_round_trip;

#[test]
fn roundtrip_permission_policy_default() {
    expect_round_trip("PermissionPolicy(default)", &PermissionPolicy::default());
}

#[test]
fn roundtrip_provider_and_model_ids() {
    if let Ok(p) = ProviderId::new("anthropic") {
        expect_round_trip("ProviderId", &p);
    }
    if let Ok(m) = ModelId::new("claude-sonnet-4-5") {
        expect_round_trip("ModelId", &m);
    }
}

#[test]
fn roundtrip_session_record_full() {
    let record = SessionRecord {
        id: "sess_0123456789abcdef".into(),
        created_at: "2025-06-01T10:00:00Z".into(),
        transcript: vec![],
        todos: vec![],
    };
    expect_round_trip("SessionRecord(empty)", &record);
}

/// `HarxesConfig` holds a pricing map; BTreeMap keeps byte order stable so the
/// decode→encode leg is deterministic even though JSON objects are unordered.
#[test]
fn roundtrip_harxes_config_with_pricing_map() {
    let mut pricing = BTreeMap::new();
    pricing.insert("claude-sonnet-4-5".to_string(), Default::default());
    let cfg = HarxesConfig { pricing, ..Default::default() };
    expect_round_trip("HarxesConfig(with-pricing)", &cfg);
}
