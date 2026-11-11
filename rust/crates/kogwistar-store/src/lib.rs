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

/// Shared ADR-015 serving-projection shape. Both durable stores must emit the
/// same namespace, cursor metadata, and schema version for a reduced runtime.
pub const RUNTIME_CURRENT_STATE_NAMESPACE: &str = "workflow_runtime_current_state";
pub const RUNTIME_PROJECTION_SCHEMA_VERSION: i64 = 1;

pub fn runtime_checkpoint_namespace(conversation_id: &str) -> String {
    format!("{conversation_id}:workflow_checkpoint_latest")
}

pub fn runtime_status_namespace(conversation_id: &str) -> String {
    format!("{conversation_id}:workflow_run_status")
}

pub fn runtime_projection_write(
    payload: Map<String, Value>,
    seq: i64,
    materialization_status: impl Into<String>,
) -> NamedProjectionWrite {
    NamedProjectionWrite {
        payload,
        last_authoritative_seq: seq,
        last_materialized_seq: seq,
        projection_schema_version: RUNTIME_PROJECTION_SCHEMA_VERSION,
        materialization_status: materialization_status.into(),
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuthUser {
    pub user_id: String,
    pub email: String,
    pub display_name: Option<String>,
    pub is_active: bool,
    pub global_role: Option<String>,
    pub global_ns: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExternalIdentity {
    pub issuer: String,
    pub subject: String,
    pub user_id: String,
    pub email: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolveExternalIdentity {
    pub issuer: String,
    pub subject: String,
    pub email: String,
    pub display_name: Option<String>,
    pub new_user_id: String,
    pub default_role: String,
    pub default_ns: String,
}

/// Standalone auth-database contract. Auth storage deliberately remains
/// separate from engine metadata because Python configures it through
/// `AUTH_DB_URL` and defaults to an independent `auth.sqlite` database.
pub trait AuthIdentityStore: Send + Sync {
    fn auth_user(
        &self,
        user_id: &str,
    ) -> impl Future<Output = StoreResult<Option<AuthUser>>> + Send;
    fn external_identity(
        &self,
        issuer: &str,
        subject: &str,
    ) -> impl Future<Output = StoreResult<Option<ExternalIdentity>>> + Send;
    /// Read the Python-compatible per-workflow role from the shared auth DB.
    fn workflow_acl_role(
        &self,
        _workflow_id: &str,
        _user_id: &str,
    ) -> impl Future<Output = StoreResult<Option<String>>> + Send {
        async { Ok(None) }
    }
    fn resolve_external_identity(
        &self,
        request: ResolveExternalIdentity,
    ) -> impl Future<Output = StoreResult<AuthUser>> + Send;
}

/// Backend-neutral helpers for ABI-preserving integer queue timestamps.
pub fn integer_timestamp(value: i64) -> Value {
    Value::from(value)
}
pub fn timestamp_i64(value: &Value) -> i64 {
    value.as_i64().unwrap_or_default()
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum StoreError {
    #[error("namespace must not be empty")]
    EmptyNamespace,
    #[error("graph record id must not be empty")]
    EmptyRecordId,
    #[error("vector dimension mismatch: expected {expected}, got {actual}")]
    VectorDimensionMismatch { expected: usize, actual: usize },
    #[error("vector contains a non-finite value")]
    NonFiniteVector,
    #[error("cosine distance is undefined for a zero vector")]
    ZeroNormVector,
    #[error("replay cursor {cursor} exceeds latest event sequence {latest}")]
    CursorOutOfRange { cursor: i64, latest: i64 },
    #[error("replay cursor cannot regress from {current} to {requested}")]
    CursorRegresses { current: i64, requested: i64 },
    #[error(
        "restored event sequence for namespace {namespace:?} must be positive and strictly increasing; got {seq}"
    )]
    InvalidRestoredEventSequence { namespace: String, seq: i64 },
    #[error("restored event id {event_id:?} is duplicated in namespace {namespace:?}")]
    DuplicateRestoredEventId { namespace: String, event_id: String },
    #[error(
        "event id {event_id:?} belongs to namespace {existing_namespace:?}, not {requested_namespace:?}"
    )]
    EventIdNamespaceCollision {
        event_id: String,
        existing_namespace: String,
        requested_namespace: String,
    },
    #[error("restored named projection key {key:?} is duplicated in namespace {namespace:?}")]
    DuplicateRestoredProjection { namespace: String, key: String },
    #[error("{identifier} must not be empty or contain a NUL byte")]
    InvalidIdentifier { identifier: &'static str },
    #[error("recovery batch limit must be positive")]
    InvalidRecoveryBatchLimit,
    #[error("invalid entity event payload: {message}")]
    InvalidEntityEventPayload { message: String },
    #[error("unsupported entity event operation {op:?}")]
    UnsupportedEntityEventOperation { op: String },
    #[error("{backend} store failure: {message}")]
    Backend { backend: String, message: String },
}

#[derive(Clone, Debug, PartialEq)]
pub struct GraphRecord {
    pub id: String,
    pub document: Option<String>,
    pub metadata: Map<String, Value>,
    pub embedding: Option<Vec<f32>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConversationListEntry {
    pub id: String,
    pub start_node_id: String,
    pub status: String,
    pub turn_count: i64,
}

/// Incrementally projects conversation-list rows from already-authorized pages.
/// Retains only turn counts and matching start records, not every graph node.
#[derive(Clone, Debug, Default)]
pub struct ConversationListAccumulator {
    user_id: String,
    turn_counts: BTreeMap<String, i64>,
    starts: Vec<(String, String, String)>,
}

impl ConversationListAccumulator {
    pub fn new(user_id: impl Into<String>) -> Self {
        Self {
            user_id: user_id.into(),
            ..Self::default()
        }
    }

    /// Add one page after the caller has applied namespace and ACL filtering.
    pub fn extend_visible_records(&mut self, records: &[GraphRecord]) {
        for record in records
            .iter()
            .filter(|record| !is_tombstoned(&record.metadata))
        {
            match value_string(record, "entity_type").as_deref() {
                Some("conversation_turn" | "assistant_turn")
                    if record_field(record, "turn_index")
                        .and_then(|value| value.as_i64())
                        .is_some() =>
                {
                    if let Some(conversation_id) = value_string(record, "conversation_id") {
                        *self.turn_counts.entry(conversation_id).or_default() += 1;
                    }
                }
                Some("conversation_start")
                    if value_string(record, "user_id").as_deref()
                        == Some(self.user_id.as_str()) =>
                {
                    if let Some(conversation_id) = value_string(record, "conversation_id") {
                        let status =
                            value_string(record, "status").unwrap_or_else(|| "active".to_owned());
                        self.starts
                            .push((conversation_id, record.id.clone(), status));
                    }
                }
                _ => {}
            }
        }
    }

    pub fn finish(self) -> Vec<ConversationListEntry> {
        let mut entries = self
            .starts
            .into_iter()
            .map(
                |(conversation_id, start_node_id, status)| ConversationListEntry {
                    turn_count: self
                        .turn_counts
                        .get(&conversation_id)
                        .copied()
                        .unwrap_or_default(),
                    id: conversation_id,
                    start_node_id,
                    status,
                },
            )
            .collect::<Vec<_>>();
        entries.sort_by_key(|entry| std::cmp::Reverse(entry.turn_count));
        entries
    }
}

/// Project conversation-list source rows to the small field set needed after
/// graph scanning. Source documents and embeddings are deliberately discarded.
pub fn compact_conversation_list_candidates(records: &[GraphRecord]) -> Vec<GraphRecord> {
    const FIELDS: &[&str] = &[
        "entity_type",
        "turn_index",
        "conversation_id",
        "user_id",
        "status",
        "lifecycle_status",
    ];
    records
        .iter()
        .filter_map(|record| {
            let entity_type = value_string(record, "entity_type")?;
            if !matches!(
                entity_type.as_str(),
                "conversation_start" | "conversation_turn" | "assistant_turn"
            ) {
                return None;
            }
            let metadata = FIELDS
                .iter()
                .filter_map(|field| {
                    record_field(record, field).map(|value| ((*field).to_owned(), value))
                })
                .collect();
            Some(GraphRecord {
                id: record.id.clone(),
                document: None,
                metadata,
                embedding: None,
            })
        })
        .collect()
}

include!("sections/graph_read_models.rs");
include!("sections/graph_traversal.rs");
include!("sections/graph_mutations.rs");
include!("sections/store_models.rs");
include!("sections/queue_store_contracts.rs");
include!("sections/run_graph_store_traits.rs");
include!("sections/event_projection_traits.rs");

#[cfg(test)]
mod tests {
    use super::*;

    include!("tests/part_01.rs");
    include!("tests/part_02.rs");
}
