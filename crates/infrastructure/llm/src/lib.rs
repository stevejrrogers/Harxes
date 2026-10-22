//! LLM provider adapters.
pub mod providers;

// CXA-B287c / CXA-B318: compile-time manifest of this adapter's serde
// (wire-surface) types, cross-checked by the contract suite in core-domain —
// see `wire_manifest.rs` for the drift-guard rationale.
pub mod wire_manifest;
