//! Durable SQLite implementation of the Phase-3 event-store slice.
//!
//! Schema names and event semantics intentionally match Python's
//! `EngineSQLite`; this crate does not own queues, projections, or graph data.

use kogwistar_contracts::EntityEventEnvelope;
use kogwistar_engine::EntityProjection;
use kogwistar_runtime::{
    PersistedRecordedTransition, RECORDED_RUNTIME_CONTRACT_VERSION, RecordedRuntimeError,
    RecordedRuntimeState, RecordedRuntimeTransition, RecordedTransitionResult,
    RecordedWorkerHandoff, RecordedWorkerSuccessEffect, RuntimeFrontier, RuntimeStepExecutePayload,
    RuntimeStepExecuteRequest, RuntimeWorkerEffectStatus, frontier_after_worker_resume,
    frontier_after_worker_success, frontier_after_worker_suspend, reduce_recorded_transition,
    transition_digest, worker_effect_digest,
};
use kogwistar_store::{
    AcceptedIndexJobResult, AppendedEvent, AuthIdentityStore, AuthUser, EntityEvent,
    EntityRebuildRequest, EntityRecoveryReport, EntityRecoveryRequest, EventPruneStore,
    EventReadStore, EventWriteStore, ExternalIdentity, IndexJob, IndexJobReadStore,
    IndexJobWriteStore, LaneMessageClaimFilter, LaneMessageFilter, LaneMessageReadStore,
    LaneMessageWriteStore, NamedProjection, NamedProjectionWrite, NewEntityEvent, NewIndexJob,
    NewProjectedLaneMessage, ProjectedLaneMessage, ProjectionReadStore, ProjectionWriteStore,
    RUNTIME_CURRENT_STATE_NAMESPACE, ReplayCursor, ResolveExternalIdentity, ServerRun,
    ServerRunCreate, ServerRunEvent, ServerRunReadStore, ServerRunUpdate, ServerRunWriteStore,
    StoreError, StoreResult, WorkflowDesignDelta, WorkflowDesignDeltaWrite,
    WorkflowDesignHistoryReadStore, WorkflowDesignHistoryWriteStore, WorkflowDesignSnapshot,
    WorkflowDesignSnapshotWrite, runtime_checkpoint_namespace, runtime_projection_write,
    runtime_status_namespace, validate_entity_rebuild_request, validate_entity_recovery_request,
};
use rusqlite::{Connection, OpenFlags, OptionalExtension, TransactionBehavior, params};
use serde_json::{Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::CStr;
use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;
use uuid::Uuid;

const BUSY_TIMEOUT_MS: u64 = 30_000;
// Bump this when initialize_schema gains a migration. The marker lets ordinary
// opens avoid taking a schema write lock after initialization, including while
// another connection has an active data transaction.
const STORE_SCHEMA_VERSION: i64 = 1;
static NEXT_CLAIM_TOKEN: AtomicU64 = AtomicU64::new(0);

/// Errors for direct SQLite APIs. Trait methods retain `kogwistar-store`'s
/// existing `StoreError` contract for its validation and cursor errors.
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
    InvalidPayload(#[from] serde_json::Error),
    #[error("sequence value must not be negative: {value}")]
    NegativeSequenceValue { value: i64 },
    #[error("transaction aborted: {0}")]
    TransactionAborted(String),
    #[error("SQLite cached session belongs to another execution context")]
    ContextMismatch,
    #[error("invalid recorded runtime transition: {0}")]
    RecordedRuntime(#[from] RecordedRuntimeError),
    #[error("recorded runtime transition conflict: {0}")]
    RecordedRuntimeConflict(String),
    #[error(transparent)]
    Store(#[from] StoreError),
}

/// One compare-and-swap mutation for an atomic named-projection batch.
pub struct NamedProjectionCas {
    pub namespace: String,
    pub key: String,
    pub expected_last_authoritative_seq: Option<i64>,
    pub expected_last_materialized_seq: Option<i64>,
    pub projection: NamedProjectionWrite,
}

pub type SqliteStoreResult<T> = Result<T, SqliteStoreError>;

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

/// New event for callers that must retain Python's original `payload_json` text.
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

/// Cloneable durable store handle. Clones share one serialized SQLite
/// connection; SQLite is a single-writer store and repeated open/WAL teardown
/// otherwise dominates short authoritative operations.
#[derive(Clone, Debug)]
pub struct SqliteStore {
    path: Arc<PathBuf>,
    connection: Arc<Mutex<Connection>>,
    _authority_lock: Arc<ProcessAuthorityLock>,
}

impl Drop for SqliteStore {
    fn drop(&mut self) {
        if std::env::var_os("KOGWISTAR_SQLITE_DIAGNOSTICS").as_deref()
            != Some(std::ffi::OsStr::new("1"))
            || Arc::strong_count(&self.connection) != 1
        {
            return;
        }
        let connection = self
            .connection
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        eprintln!(
            "[sqlite-rust-close] conn={:p} path={} {}",
            unsafe { connection.handle() },
            self.path.display(),
            sqlite_sidecars(self.path.as_ref())
        );
    }
}

/// Independent Python-compatible `AUTH_DB_URL=sqlite:///...` store. Opening
/// it creates only auth relations and never engine metadata relations.
#[derive(Clone, Debug)]
pub struct SqliteAuthStore {
    path: Arc<PathBuf>,
}

mod auth;
mod authority;
mod authority_identity;
mod connection_uow;
mod events_recovery;
mod index_jobs;
mod lane_messages;
mod projections;
mod recovery;
mod row_mapping;
mod schema;
mod store;
mod trait_impls;
mod workflow_runtime;

use authority::{ProcessAuthorityLock, database_initialization_lock, is_private_memory_database};
#[cfg(test)]
use authority_identity::ACTIVE_DATABASE_IDENTITIES;
use authority_identity::{register_active_database_identity, unregister_active_database_identity};
pub use connection_uow::SqliteUnitOfWork;
use connection_uow::*;
use events_recovery::*;
use index_jobs::*;
use lane_messages::*;
use projections::*;
use recovery::*;
use row_mapping::*;
use schema::*;
use workflow_runtime::*;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

fn unix_epoch_seconds() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is before Unix epoch")
        .as_secs() as i64
}

fn unix_epoch_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock precedes Unix epoch")
        .as_millis() as i64
}

fn require_trait_namespace(namespace: &str) -> StoreResult<()> {
    if namespace.is_empty() {
        Err(StoreError::EmptyNamespace)
    } else {
        Ok(())
    }
}

fn trait_error(error: SqliteStoreError) -> StoreError {
    match error {
        SqliteStoreError::Store(error) => error,
        SqliteStoreError::EmptyNamespace => StoreError::EmptyNamespace,
        SqliteStoreError::EventIdNamespaceCollision {
            event_id,
            existing_namespace,
            requested_namespace,
        } => StoreError::EventIdNamespaceCollision {
            event_id,
            existing_namespace,
            requested_namespace,
        },
        error => StoreError::Backend {
            backend: "sqlite".to_owned(),
            message: error.to_string(),
        },
    }
}
