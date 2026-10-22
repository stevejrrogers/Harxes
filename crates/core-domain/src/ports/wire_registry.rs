//! Wire-surface type descriptors (CXA-B287a / CXA-B316): the registry data
//! model. A `WireType` names one serde-facing type on the HTTP/JSON surface
//! and the adapter/port file that owns it; `tests/contracts/wire_registry.rs`
//! holds the exhaustive instance list plus its drift-count assertion.

/// One wire-facing (serde) type exposed across an adapter boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WireType {
    /// Rust type name as declared in the owning module.
    pub name: &'static str,
    /// Repo-relative path of the module that owns the type.
    pub owner: &'static str,
}
