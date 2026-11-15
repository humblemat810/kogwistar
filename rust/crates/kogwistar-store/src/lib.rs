//! Async-first storage contracts. Shadow handles expose reads only.
pub use kogwistar_contracts::GraphEntityKind;
use kogwistar_contracts::{
    ContextSnapshotV1, ConversationContractError, ConversationSummaryV1, ConversationTranscriptV1,
    ConversationTurnV1, EntityEventEnvelope, GraphModelClass, GraphModelEnvelope,
    PersistentGraphEntityV1, ToolAuditContractError, ToolAuditItemV1,
};
use serde::Serialize;
use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;
use thiserror::Error;

pub type StoreResult<T> = Result<T, StoreError>;
pub type EntityEvent = EntityEventEnvelope;

/// Backend-neutral helpers for ABI-preserving integer queue timestamps.
pub fn integer_timestamp(value: i64) -> Value {
    Value::from(value)
}
pub fn timestamp_i64(value: &Value) -> i64 {
    value.as_i64().unwrap_or_default()
}

#[derive(Clone, Debug, PartialEq)]
pub struct GraphRecord {
    pub id: String,
    pub document: Option<String>,
    pub metadata: Map<String, Value>,
    pub embedding: Option<Vec<f32>>,
}

include!("sections/graph_read_models.rs");
include!("sections/graph_traversal.rs");
include!("sections/graph_mutations.rs");
include!("sections/store_models.rs");
include!("sections/queue_store_contracts.rs");
include!("sections/run_graph_store_traits.rs");
include!("sections/event_projection_traits.rs");
include!("sections/auth_identity_store.rs");
include!("sections/runtime_projection.rs");
include!("sections/store_errors.rs");
include!("sections/conversation_list.rs");

#[cfg(test)]
mod tests {
    use super::*;

    include!("tests/part_01.rs");
    include!("tests/part_02.rs");
}
