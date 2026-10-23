//! CXA-B305a [1/3] — round-trip test suite entry point.
//!
//! * `harness`            — reusable encode→decode / decode→encode assertion
//! * `dto_inventory.json` — machine-readable source-of-truth DTO inventory
//! * per-domain files     — conversation and config/policy groups

mod harness;
#[path = "transport_coverage.rs"]
mod transport_coverage;

pub use harness::{assert_round_trip, expect_round_trip, RoundTripOk};

mod config_dtos;
mod conversation_dtos;
