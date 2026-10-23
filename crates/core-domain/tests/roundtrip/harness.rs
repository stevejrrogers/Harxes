//! Reusable round-trip test harness for public transport DTOs (CXA-B305a).
//!
//! Pure functions only — no network, no server spin-up. Given a serde type and
//! a representative fixture it asserts both directions of the wire contract:
//!
//! * encode→decode: decoding the encoded bytes yields the same JSON value as
//!   encoding the input (normalized, so unordered maps / absent-vs-default
//!   fields never flake)
//! * decode→encode: re-encoding the decoded value reproduces the canonical
//!   bytes exactly.

use serde::de::DeserializeOwned;
use serde::Serialize;

/// A successfully verified round trip.
pub struct RoundTripOk {
    /// Bytes produced by the first encode (`serde_json::to_vec_pretty`).
    #[allow(dead_code)]
    pub canonical_bytes: Vec<u8>,
}

/// Failure detail for one leg of the round trip.
#[derive(Debug)]
pub enum RoundTripError {
    /// `encode` itself failed.
    Encode(String),
    /// `decode` of the encoded bytes failed.
    Decode { json: String, error: String },
    /// Decoded value differs from the input after normalization.
    NotEqual { expected: serde_json::Value, actual: serde_json::Value },
    /// Re-encoding the decoded value diverged from the canonical bytes.
    ByteDrift { canonical: String, reencoded: String },
}

impl std::fmt::Display for RoundTripError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Encode(e) => write!(f, "encode failed: {e}"),
            Self::Decode { error, .. } => write!(f, "decode failed: {error}"),
            Self::NotEqual { .. } => write!(f, "decoded value != original fixture"),
            Self::ByteDrift { .. } => write!(f, "re-encoded bytes != canonical bytes"),
        }
    }
}

impl std::error::Error for RoundTripError {}

fn to_json<T: Serialize>(value: &T) -> Result<serde_json::Value, RoundTripError> {
    serde_json::to_value(value).map_err(|e| RoundTripError::Encode(e.to_string()))
}

/// Full two-directional round trip over a real DTO instance. Equality is
/// checked on normalized `serde_json::Value`s (order-insensitive objects), not
/// on `PartialEq` of the type — many DTOs do not derive it.
///
/// The fixture must be buildable from production constructors (no fabricated data).
pub fn assert_round_trip<T>(fixture: &T) -> Result<RoundTripOk, RoundTripError>
where
    T: Serialize + DeserializeOwned,
{
    // Leg 1: encode → decode must equal the input (semantic, normalized).
    let bytes =
        serde_json::to_vec_pretty(fixture).map_err(|e| RoundTripError::Encode(e.to_string()))?;
    let decoded: T = serde_json::from_slice(&bytes).map_err(|e| RoundTripError::Decode {
        json: String::from_utf8_lossy(&bytes).into_owned(),
        error: e.to_string(),
    })?;

    let expected = to_json(fixture)?;
    let actual = to_json(&decoded)?;
    if expected != actual {
        return Err(RoundTripError::NotEqual { expected, actual });
    }

    // Leg 2: decode → encode must reproduce the canonical bytes exactly.
    let reencoded =
        serde_json::to_vec_pretty(&decoded).map_err(|e| RoundTripError::Encode(e.to_string()))?;
    if reencoded != bytes {
        return Err(RoundTripError::ByteDrift {
            canonical: String::from_utf8_lossy(&bytes).into_owned(),
            reencoded: String::from_utf8_lossy(&reencoded).into_owned(),
        });
    }

    Ok(RoundTripOk { canonical_bytes: bytes })
}

/// Convenience wrapper for use inside `#[test]`: panics with the failure detail.
#[track_caller]
pub fn expect_round_trip<T>(name: &'static str, fixture: &T) -> RoundTripOk
where
    T: Serialize + DeserializeOwned,
{
    match assert_round_trip(fixture) {
        Ok(ok) => ok,
        Err(err) => panic!("round-trip failed for {name}: {err}"),
    }
}
