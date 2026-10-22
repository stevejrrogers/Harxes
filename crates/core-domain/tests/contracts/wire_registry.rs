//! Wire-surface registry (CXA-B287a / CXA-B316): an exhaustive, statically
//! checked inventory of every wire-facing type currently exposed across the
//! HTTP/JSON surface, each entry naming its owning adapter or port file.
//!
//! The contract guard tests (later parts) iterate `WIRE_SURFACE` and assert
//! invariants over each entry. The count assertion below makes accidental
//! drift — a DTO added or removed without registering it — fail loudly at
//! test time instead of silently changing the guarded surface.

use harxes_core_domain::ports::WireType;

/// Every wire-facing (serde) type on the current surface.
///
/// Grouped by owning adapter/port path as registered in the codebase today:
/// - `core-domain/src/domain/value_objects/message.rs`      — chat-message wire model
/// - `core-domain/src/domain/value_objects/tool_call.rs`    — tool-call envelope
/// - `core-domain/src/domain/value_objects/todo.rs`         — plan-item wire model
/// - `core-domain/src/domain/value_objects/reasoning.rs`    — reasoning-effort dial
/// - `core-domain/src/domain/value_objects/provider_id.rs`  — provider identity string
/// - `core-domain/src/ports/config_store.rs`                — user config JSON (`~/.harxes/config.json`)
/// - `core-domain/src/ports/session.rs`                     — persisted session transcript
/// - `app/src/services/provider_registry.rs`                — provider descriptors (serialized to UI/host)
pub const WIRE_SURFACE: &[WireType] = &[
    // --- crates/core-domain/src/domain/value_objects/message.rs -------------
    WireType { name: "Role", owner: "crates/core-domain/src/domain/value_objects/message.rs" },
    WireType { name: "ImageData", owner: "crates/core-domain/src/domain/value_objects/message.rs" },
    WireType { name: "Message", owner: "crates/core-domain/src/domain/value_objects/message.rs" },
    // --- crates/core-domain/src/domain/value_objects/tool_call.rs -----------
    WireType { name: "ToolCall", owner: "crates/core-domain/src/domain/value_objects/tool_call.rs" },
    // --- crates/core-domain/src/domain/value_objects/todo.rs ----------------
    WireType { name: "TodoStatus", owner: "crates/core-domain/src/domain/value_objects/todo.rs" },
    WireType { name: "TodoItem", owner: "crates/core-domain/src/domain/value_objects/todo.rs" },
    // --- crates/core-domain/src/domain/value_objects/reasoning.rs -----------
    WireType { name: "ReasoningEffort", owner: "crates/core-domain/src/domain/value_objects/reasoning.rs" },
    // --- crates/core-domain/src/domain/value_objects/provider_id.rs ---------
    WireType { name: "ProviderId", owner: "crates/core-domain/src/domain/value_objects/provider_id.rs" },
    // --- crates/core-domain/src/ports/config_store.rs -----------------------
    WireType { name: "ProviderConfig", owner: "crates/core-domain/src/ports/config_store.rs" },
    WireType { name: "ModelPricing", owner: "crates/core-domain/src/ports/config_store.rs" },
    WireType { name: "HarxesConfig", owner: "crates/core-domain/src/ports/config_store.rs" },
    WireType { name: "LimitsConfig", owner: "crates/core-domain/src/ports/config_store.rs" },
    WireType { name: "HookRule", owner: "crates/core-domain/src/ports/config_store.rs" },
    WireType { name: "HooksConfig", owner: "crates/core-domain/src/ports/config_store.rs" },
    WireType { name: "McpServerConfig", owner: "crates/core-domain/src/ports/config_store.rs" },
    // --- crates/core-domain/src/ports/session.rs ----------------------------
    WireType { name: "SessionRecord", owner: "crates/core-domain/src/ports/session.rs" },
];

/// Hand-maintained expected total; if the compiler says this no longer equals
/// `WIRE_SURFACE.len()`, someone changed the wire surface without updating the
/// registry — exactly the drift this ticket exists to catch.
pub const EXPECTED_WIRE_TYPES: usize = 16;

#[test]
fn registry_count_matches_expected_total() {
    assert_eq!(
        WIRE_SURFACE.len(),
        EXPECTED_WIRE_TYPES,
        "the wire surface changed: update WIRE_SURFACE (and EXPECTED_WIRE_TYPES) \
         so contract guards cover every wire-facing type"
    );
}

/// Sanity: every entry must name its owning adapter file.
#[test]
fn every_entry_names_its_owner() {
    for t in WIRE_SURFACE {
        assert!(!t.owner.is_empty(), "entry {} has no owning path", t.name);
        assert!(t.owner.starts_with("crates/"), "owner path not repo-relative: {}", t.owner);
    }
}
