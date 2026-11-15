//! Recorded-only workflow transition reducer for ADR-015 Phase 4.
//!
//! This crate deliberately has no resolver, provider, graph, lane, or Python
//! dependency.  A caller supplies a result that was already recorded by the
//! authoritative worker.  The reducer validates and folds only durable state.

use kogwistar_contracts::{
    NATIVE_SEAM_ATTESTATION_VERSION, attach_rust_native_executor_attestation, canonical_json,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, OnceLock};
use thiserror::Error;

include!("sections/native_operations.rs");
include!("sections/routing.rs");
include!("sections/worker_decisions.rs");
include!("sections/recorded_transitions.rs");

#[cfg(test)]
mod tests {
    use super::*;
    include!("tests.rs");
}
