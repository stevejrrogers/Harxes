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
/// - `infrastructure/llm/src/providers/{anthropic,openai}.rs` — provider request/response DTOs (CXA-B318; cross-checked against that crate's compile-time `wire_manifest`)
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
    // --- crates/infrastructure/llm/src/providers/anthropic.rs (CXA-B318) ----
    WireType { name: "RequestBody",     owner: "crates/infrastructure/llm/src/providers/anthropic.rs" },
    WireType { name: "ResponseBody",    owner: "crates/infrastructure/llm/src/providers/anthropic.rs" },
    WireType { name: "ContentBlock",    owner: "crates/infrastructure/llm/src/providers/anthropic.rs" },
    WireType { name: "Usage",           owner: "crates/infrastructure/llm/src/providers/anthropic.rs" },
    // --- crates/infrastructure/llm/src/providers/openai.rs (CXA-B318) -------
    WireType { name: "FunctionDef",      owner: "crates/infrastructure/llm/src/providers/openai.rs" },
    WireType { name: "ToolDef",          owner: "crates/infrastructure/llm/src/providers/openai.rs" },
    WireType { name: "RequestToolCall",  owner: "crates/infrastructure/llm/src/providers/openai.rs" },
    WireType { name: "RequestFunction",  owner: "crates/infrastructure/llm/src/providers/openai.rs" },
    WireType { name: "ApiMessage",       owner: "crates/infrastructure/llm/src/providers/openai.rs" },
    WireType { name: "RequestBody",      owner: "crates/infrastructure/llm/src/providers/openai.rs" },
    WireType { name: "ResponseBody",     owner: "crates/infrastructure/llm/src/providers/openai.rs" },
    WireType { name: "Choice",           owner: "crates/infrastructure/llm/src/providers/openai.rs" },
    WireType { name: "ResponseMessage",  owner: "crates/infrastructure/llm/src/providers/openai.rs" },
    WireType { name: "ResponseToolCall", owner: "crates/infrastructure/llm/src/providers/openai.rs" },
    WireType { name: "ResponseFunction", owner: "crates/infrastructure/llm/src/providers/openai.rs" },
    WireType { name: "Usage",            owner: "crates/infrastructure/llm/src/providers/openai.rs" },
    WireType { name: "CompletionDetails", owner: "crates/infrastructure/llm/src/providers/openai.rs" },
];

/// Hand-maintained expected total; if the compiler says this no longer equals
/// `WIRE_SURFACE.len()`, someone changed the wire surface without updating the
/// registry — exactly the drift this ticket exists to catch.
pub const EXPECTED_WIRE_TYPES: usize = 33;

// CXA-B287c / CXA-B318 — registry-completeness cross-check. The llm adapter
// crate owns a compile-time-proven manifest (`harxes_infra_llm::wire_manifest`)
// listing every serde type in its providers. Those entries must appear here:
// forgetting one is real drift (a wire type with no contract guards).
//
// Hermeticity note: these tests are pure data — no env vars, no network, no fs
// writes outside target/, and no ordering dependence between test binaries.
//
// Fingerprint workaround (one-liner per CXA-B318 AC4): a stale-fingerprint /
// timestamp fs error under target/ after this suite was added was cleared once
// with a TARGETED clean — `cargo clean -p harxes-infra-llm -p harxes-core-domain`
// — not a full `cargo clean`; root cause was editing a dependency crate while
// its compiled artifacts were mid-write during a parallel earlier run.

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

// ---------------------------------------------------------------------------
// CXA-B287c / CXA-B318 — the guard itself (assertion machinery, tested pure)
// ---------------------------------------------------------------------------

/// A missing registry entry, as reported by [`find_registry_gaps`].
#[derive(Debug, PartialEq, Eq)]
pub struct RegistryGap {
    pub name: &'static str,
    pub owner: &'static str,
}

/// Pure core of the drift guard: which adapter-manifest entries are absent
/// from `registered`. No IO, no env, deterministic — trivially testable both
/// green and red without editing committed code.
pub fn find_registry_gaps(
    registered: &[WireType],
    manifest: &[harxes_infra_llm::wire_manifest::AdapterWireType],
) -> Vec<RegistryGap> {
    let is_registered = |name: &str, owner: &str| {
        registered.iter().any(|w| w.name == name && w.owner.ends_with(owner))
    };
    manifest
        .iter()
        .filter(|a| !is_registered(a.name, a.owner))
        .map(|a| RegistryGap { name: a.name, owner: a.owner })
        .collect()
}

/// The live guard: every type in the llm adapter's compile-time-proven wire
/// manifest must be present in this registry. Removing a DTO's `WIRE_SURFACE`
/// line (drift) fails here with an actionable gap list naming the entry.
#[test]
fn registry_covers_adapter_wire_manifest() {
    use harxes_infra_llm::wire_manifest::ADAPTER_WIRE_TYPES;

    let gaps = find_registry_gaps(WIRE_SURFACE, ADAPTER_WIRE_TYPES);
    assert!(
        gaps.is_empty(),
        "wire surface drifted: adapter types missing from WIRE_SURFACE \
         (add one line each to the registry): {:?}",
        gaps
    );
}

/// Negative-behaviour proof of the guard MACHINERY (not of committed state):
/// simulate dropping one registry entry and confirm the detector reports
/// exactly that gap — proving red/green works without editing real code.
/// (Manually simulated end-to-end too: deleting the "Choice" line from
/// `WIRE_SURFACE` turns `registry_covers_adapter_wire_manifest` FAILED with
/// `[RegistryGap { name: "Choice", owner: "providers/openai.rs" }]`.)
#[test]
fn gap_detector_fires_when_an_entry_is_removed() {
    use harxes_infra_llm::wire_manifest::ADAPTER_WIRE_TYPES;

    // Drop the last openai entry ("CompletionDetails") from a simulated
    // registry — mirrors the manual red/green drill done for this ticket.
    let mut reduced: Vec<WireType> = WIRE_SURFACE.iter().cloned().collect();
    let removed = reduced.pop().unwrap();

    let gaps = find_registry_gaps(&reduced, ADAPTER_WIRE_TYPES);
    assert_eq!(gaps.len(), 1, "exactly one simulated gap expected");
    assert_eq!(gaps[0].name, removed.name);
    assert!(removed.owner.ends_with(gaps[0].owner));
}

/// Positive control on a fully-populated registry (green path of the same
/// machinery).
#[test]
fn gap_detector_silent_on_full_registry() {
    use harxes_infra_llm::wire_manifest::ADAPTER_WIRE_TYPES;

    assert!(find_registry_gaps(WIRE_SURFACE, ADAPTER_WIRE_TYPES).is_empty());
}
