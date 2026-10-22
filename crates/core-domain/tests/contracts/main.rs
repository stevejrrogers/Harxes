//! Contract guard-test scaffolding (CXA-B287a / CXA-B316): module tree only.
//!
//! Part 1 of 3 — NO invariant test bodies here yet. `wire_registry.rs`
//! enumerates every wire-facing type with its owning adapter;
//! `mod_helpers.rs` builds samples purely from real domain types. Later
//! parts add invariant cases on top of these.
#![allow(dead_code)] // scaffolding: builders consumed by parts 2-3

mod mod_helpers;
mod wire_registry;

// CXA-B287b / CXA-B317: the invariant suites (one file per responsibility).
#[path = "requests_invariants.rs"]
mod requests_invariants;
#[path = "responses_invariants.rs"]
mod responses_invariants;
#[path = "error_envelope_invariants.rs"]
mod error_envelope_invariants;
