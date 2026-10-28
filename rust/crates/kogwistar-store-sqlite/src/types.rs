//! Public SQLite store errors and raw event/projection contracts.

use kogwistar_runtime::RecordedRuntimeError;
use kogwistar_store::{NamedProjectionWrite, StoreError};
use serde_json::Error as JsonError;
use thiserror::Error;

/// Errors for direct SQLite APIs. Trait methods retain `StoreError` for their
/// validation and cursor contract.
#[derive(Debug, Error)]
pub enum SqliteStoreError {
    #[error("SQLite operation failed: {0}")]
    Sql(#[from] rusqlite::Error),
    #[error("filesystem operation failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("SQLite database authority is held by another process for {path}: {message}")]
    AuthorityConflict { path: String, message: String },
    #[error("namespace must not be empty")]
    EmptyNamespace,
    #[error(
        "event_id {event_id:?} already belongs to namespace {existing_namespace:?}, not {requested_namespace:?}"
    )]
    EventIdNamespaceCollision {
        event_id: String,
        existing_namespace: String,
        requested_namespace: String,
    },
    #[error("event payload is not valid JSON: {0}")]
    InvalidPayload(#[from] JsonError),
    #[error("sequence value must not be negative: {value}")]
    NegativeSequenceValue { value: i64 },
    #[error("transaction aborted: {0}")]
    TransactionAborted(String),
    #[error("SQLite cached session belongs to another execution context")]
    ContextMismatch,
    #[error(transparent)]
    RecordedRuntime(#[from] RecordedRuntimeError),
    #[error("recorded runtime transition conflict: {0}")]
    RecordedRuntimeConflict(String),
    #[error(transparent)]
    Store(#[from] StoreError),
}

pub type SqliteStoreResult<T> = Result<T, SqliteStoreError>;

/// One compare-and-swap mutation for an atomic named-projection batch.
pub struct NamedProjectionCas {
    pub namespace: String,
    pub key: String,
    pub expected_last_authoritative_seq: Option<i64>,
    pub expected_last_materialized_seq: Option<i64>,
    pub projection: NamedProjectionWrite,
}

/// Event row with the raw SQLite JSON text retained byte-for-byte as stored.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawEntityEvent {
    pub namespace: String,
    pub seq: i64,
    pub event_id: String,
    pub entity_kind: String,
    pub entity_id: String,
    pub op: String,
    pub payload_json: String,
    pub created_at: i64,
}

/// New event preserving the caller's original `payload_json` text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NewRawEntityEvent {
    pub event_id: String,
    pub entity_kind: String,
    pub entity_id: String,
    pub op: String,
    pub payload_json: String,
}

/// An event imported from a portable archive with authoritative identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RestoredRawEntityEvent {
    pub seq: i64,
    pub event_id: String,
    pub entity_kind: String,
    pub entity_id: String,
    pub op: String,
    pub payload_json: String,
    pub created_at: i64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AppendedRawEvent {
    pub event: RawEntityEvent,
    pub inserted: bool,
}
