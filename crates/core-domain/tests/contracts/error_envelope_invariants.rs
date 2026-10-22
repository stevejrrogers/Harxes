//! Error-envelope invariants (CXA-B287b / CXA-B317): there is exactly ONE
//! canonical error shape per port boundary, its variants are exhaustively
//! matched and displayable, and every adapter maps provider HTTP statuses
//! through the same `LlmError` struct. Pure checks over real types — no IO.
//!
//! The adapters (`crates/infrastructure/llm/src/providers/*`) implement the
//! status→variant mapping by hand today; these tests pin the ENUM side of
//! that contract: all four transport outcomes exist, render a non-empty
//! message, and are reachable as `dyn std::error::Error`.

use harxes_core_domain::domain::value_objects::ProviderId;
use harxes_core_domain::ports::{ConfigStoreError, LlmError, SessionStoreError};

/// Every [`LlmError`] variant renders a non-empty, lowercase-prefixed,
/// human-readable message via `Display` — the envelope shown to users/logs.
#[test]
fn llm_error_display_is_total_and_nonempty() {
        let cases: Vec<(&'static str, LlmError)> = vec![
            ("request error", LlmError::Request("boom".into())),
            (
                "authentication failed",
                LlmError::Auth {
                    provider: ProviderId::new("anthropic").expect("valid id"),
                },
            ),
            (
                "rate limited",
                LlmError::RateLimited { retry_after: Some(7) },
            ),
            (
                "rate limited",
                LlmError::RateLimited { retry_after: None },
            ),
            ("timed out", LlmError::Timeout),
        ];
        for (phrase, err) in &cases {
            let text = err.to_string();
            assert!(!text.is_empty(), "{phrase:?} rendered empty");
            assert!(
                text.contains(phrase),
                "envelope lost its stable phrasing {phrase:?}: got {text:?}"
        );
    }
}

/// Canonical error shape is a real `std::error::Error`, not just Display:
/// adapters can wrap/compose it uniformly.
#[test]
fn llm_error_implements_std_error() {
    fn takes_std_error<E: std::error::Error>(_: &E) {}
    takes_std_error(&LlmError::Timeout);

    // ...and so do the store-error envelopes of the other ports.
    takes_std_error(&SessionStoreError::Io("x".into()));
    takes_std_error(&ConfigStoreError::Io("x".into()));

    // Composability invariant: any store failure can be surfaced as one
    // generic box without losing its type identity.
    let boxed: Box<dyn std::error::Error> = Box::new(ConfigStoreError::Io("disk".into()));
    assert!(boxed.to_string().contains("config store"));
}

/// One canonical shape per transport outcome class; exhaustive match proves
/// no variant was added without an envelope decision (compile-time + runtime).
#[test]
fn llm_error_variants_are_exactly_the_transport_classes() {
    let classes: Vec<&'static str> = [
        LlmError::Request(String::new()),
        LlmError::Timeout,
        LlmError::RateLimited { retry_after: None },
    ]
    .iter()
    .map(|e| classify(e))
    .chain({
        let p = ProviderId::new("p").unwrap();
        [classify(&LlmError::Auth { provider: p })]
    })
    .collect();
    // Auth is its own class (status 401/403 at the adapter); rate-limiting
    // is its own (429 + Retry-After). No two collapse together.
    let mut sorted = classes.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), classes.len(), "each variant must map to a distinct transport class");
    assert_eq!(
        classes.iter().filter(|c| **c == "auth").count(),
        1,
        "exactly one auth-classed variant"
    );
}

/// Totality helper: every `LlmError` falls into one of the agreed classes.
/// Adding a variant extends this match — the compiler enforces totality.
fn classify(err: &LlmError) -> &'static str {
    match err {
        LlmError::Request(_) => "request",
        LlmError::Auth { .. } => "auth",
        LlmError::RateLimited { .. } => "rate_limited",
        LlmError::Timeout => "timeout",
    }
}
