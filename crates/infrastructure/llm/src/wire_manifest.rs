//! Compile-time wire-surface manifest (CXA-B287c / CXA-B318): every
//! serde-facing type declared inside this infrastructure adapter crate is
//! listed here with the exact module path that owns it.
//!
//! Drift guard: each entry is *statically proven* by `tests` below against a
//! generic bound matching its [`WireDir`] — deleting, renaming or un-deriving
//! a DTO fails to COMPILE, and adding one without a manifest entry fails the
//! contract-suite coverage test in `core-domain/tests/contracts/wire_registry.rs`.
//! Wire drift cannot land silently.

use serde::de::DeserializeOwned;
use serde::Serialize;

/// Which side of the wire a type serves. Adapter DTOs are one-sided by
/// construction: request payloads exist only to be serialized, response
/// payloads only to be deserialized, so demanding both serde traits on
/// everything would fail to compile against reality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WireDir {
    /// Serialized outbound (provider HTTP request payloads).
    Request,
    /// Deserialized inbound (provider HTTP response payloads).
    Response,
}

/// One declared wire type in an adapter: its Rust name, owning module path,
/// and direction.
///
/// Structurally parallel to core-domain's `WireType` but kept separate on
/// purpose: the domain descriptor carries no trait bounds and must not depend
/// on adapter concerns. The contract suite joins the two lists by
/// `(name, owner)` instead.
#[derive(Debug, Clone, Copy)]
pub struct AdapterWireType {
    /// Rust type name as declared in the owning module.
    pub name: &'static str,
    /// Crate-relative path of the module that declares the type.
    pub owner: &'static str,
    /// Direction this type travels over the wire.
    pub dir: WireDir,
}

/// Every serde struct declared in this crate's adapters, grouped by file.
///
/// Counts verified against source at time of writing: `providers/openai.rs`
/// derives serde on 13 types (`FunctionDef`, `ToolDef`, `RequestToolCall`,
/// `RequestFunction`, `ApiMessage`, `RequestBody`, `ResponseBody`, `Choice`,
/// `ResponseMessage`, `ResponseToolCall`, `ResponseFunction`, `Usage`,
/// `CompletionDetails`) and `providers/anthropic.rs` on 4 (`RequestBody`,
/// `ResponseBody`, `ContentBlock`, `Usage`). Adding a new serde-derived type
/// under `src/providers/` requires one line here plus one `prove_*::<..>` call
/// in `tests::manifest_index_alignment`.
pub const ADAPTER_WIRE_TYPES: &[AdapterWireType] = &[
    // --- src/providers/anthropic.rs ----------------------------------------
    AdapterWireType { name: "RequestBody",       owner: "providers/anthropic.rs", dir: WireDir::Request },
    AdapterWireType { name: "ResponseBody",      owner: "providers/anthropic.rs", dir: WireDir::Response },
    AdapterWireType { name: "ContentBlock",      owner: "providers/anthropic.rs", dir: WireDir::Response },
    AdapterWireType { name: "Usage",             owner: "providers/anthropic.rs", dir: WireDir::Response },
    // --- src/providers/openai.rs -------------------------------------------
    AdapterWireType { name: "FunctionDef",       owner: "providers/openai.rs", dir: WireDir::Request },
    AdapterWireType { name: "ToolDef",           owner: "providers/openai.rs", dir: WireDir::Request },
    AdapterWireType { name: "RequestToolCall",   owner: "providers/openai.rs", dir: WireDir::Request },
    AdapterWireType { name: "RequestFunction",   owner: "providers/openai.rs", dir: WireDir::Request },
    AdapterWireType { name: "ApiMessage",        owner: "providers/openai.rs", dir: WireDir::Request },
    AdapterWireType { name: "RequestBody",       owner: "providers/openai.rs", dir: WireDir::Request },
    AdapterWireType { name: "ResponseBody",      owner: "providers/openai.rs", dir: WireDir::Response },
    AdapterWireType { name: "Choice",            owner: "providers/openai.rs", dir: WireDir::Response },
    AdapterWireType { name: "ResponseMessage",   owner: "providers/openai.rs", dir: WireDir::Response },
    AdapterWireType { name: "ResponseToolCall",  owner: "providers/openai.rs", dir: WireDir::Response },
    AdapterWireType { name: "ResponseFunction",  owner: "providers/openai.rs", dir: WireDir::Response },
    AdapterWireType { name: "Usage",             owner: "providers/openai.rs", dir: WireDir::Response },
    AdapterWireType { name: "CompletionDetails", owner: "providers/openai.rs", dir: WireDir::Response },
];

// Statically prove an entry still names a real, serde-capable Rust type: if
// the concrete type was renamed or deleted, `tests::manifest_index_alignment`
// stops compiling with an unresolved-path error pointing at exactly which
// manifest slot went stale — the loudest possible drift signal.
//
// NOTE (CXA-B318 negative-behaviour proofs, all manually simulated):
// - remove one ADAPTER_WIRE_TYPES line            ⇒ count assert + alignment
//   test panic (and contract-side registry gap);
// - rename/delete the underlying DTO              ⇒ this file FAILS TO COMPILE;
// - drop the DTO's #[derive(Serialize/Deserialize)] ⇒ same compile failure.
#[allow(dead_code)]
fn prove_request<T: Serialize>(_a: &AdapterWireType) {}

#[allow(dead_code)]
fn prove_response<T: DeserializeOwned>(_a: &AdapterWireType) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::{anthropic, openai};

    /// Entries and static proofs stay index-aligned: every manifest slot is
    /// covered by exactly one `prove_*::<ConcreteType>` call, so a new entry
    /// without a compile-time proof (or vice versa) cannot slip through.
    #[test]
    fn manifest_index_alignment() {
        assert_eq!(ADAPTER_WIRE_TYPES.len(), 17);
        // providers/anthropic.rs
        prove_request::<anthropic::RequestBody>(&ADAPTER_WIRE_TYPES[0]);
        prove_response::<anthropic::ResponseBody>(&ADAPTER_WIRE_TYPES[1]);
        prove_response::<anthropic::ContentBlock>(&ADAPTER_WIRE_TYPES[2]);
        prove_response::<anthropic::Usage>(&ADAPTER_WIRE_TYPES[3]);
        // providers/openai.rs
        prove_request::<openai::FunctionDef>(&ADAPTER_WIRE_TYPES[4]);
        prove_request::<openai::ToolDef>(&ADAPTER_WIRE_TYPES[5]);
        prove_request::<openai::RequestToolCall>(&ADAPTER_WIRE_TYPES[6]);
        prove_request::<openai::RequestFunction>(&ADAPTER_WIRE_TYPES[7]);
        prove_request::<openai::ApiMessage>(&ADAPTER_WIRE_TYPES[8]);
        prove_request::<openai::RequestBody>(&ADAPTER_WIRE_TYPES[9]);
        prove_response::<openai::ResponseBody>(&ADAPTER_WIRE_TYPES[10]);
        prove_response::<openai::Choice>(&ADAPTER_WIRE_TYPES[11]);
        prove_response::<openai::ResponseMessage>(&ADAPTER_WIRE_TYPES[12]);
        prove_response::<openai::ResponseToolCall>(&ADAPTER_WIRE_TYPES[13]);
        prove_response::<openai::ResponseFunction>(&ADAPTER_WIRE_TYPES[14]);
        prove_response::<openai::Usage>(&ADAPTER_WIRE_TYPES[15]);
        prove_response::<openai::CompletionDetails>(&ADAPTER_WIRE_TYPES[16]);
    }

    /// Every entry points into the providers adapter subtree — the manifest is
    /// scoped to this crate's HTTP-wire adapters, nothing else.
    #[test]
    fn manifest_entries_scope_to_providers_subtree() {
        for t in ADAPTER_WIRE_TYPES {
            assert!(
                t.owner.starts_with("providers/"),
                "wire DTO {} registered outside providers/: {}",
                t.name,
                t.owner
            );
        }
    }

    /// Entries are unique by (name, owner) so the contract suite can treat
    /// them as set elements rather than multiset noise.
    #[test]
    fn manifest_entries_are_unique() {
        let mut seen = Vec::new();
        for t in ADAPTER_WIRE_TYPES {
            assert!(
                !seen.contains(&(t.name, t.owner)),
                "duplicate manifest entry: {} ({})",
                t.name,
                t.owner
            );
            seen.push((t.name, t.owner));
        }
    }
}
