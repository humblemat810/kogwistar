//! Durable SQLite implementation of the Phase-3 event-store slice.
//!
//! Schema names and event semantics intentionally match Python's
//! `EngineSQLite`; this crate does not own queues, projections, or graph data.

use kogwistar_contracts::EntityEventEnvelope;
use kogwistar_engine::EntityProjection;
use kogwistar_runtime::{
    PersistedRecordedTransition, RECORDED_RUNTIME_CONTRACT_VERSION, RecordedRuntimeError,
    RecordedRuntimeState, RecordedRuntimeTransition, RecordedTransitionResult,
    RecordedWorkerHandoff, RecordedWorkerSuccessEffect, RuntimeFrontier, RuntimeWorkerEffectStatus,
    frontier_after_worker_resume, frontier_after_worker_success, frontier_after_worker_suspend,
    reduce_recorded_transition, transition_digest, worker_effect_digest,
};
use kogwistar_store::{
    AppendedEvent, AuthIdentityStore, AuthUser, EntityEvent, EntityRebuildRequest,
    EntityRecoveryReport, EntityRecoveryRequest, EventPruneStore, EventReadStore, EventWriteStore,
    ExternalIdentity, IndexJob, IndexJobReadStore, IndexJobWriteStore, LaneMessageFilter,
    LaneMessageReadStore, LaneMessageWriteStore, NamedProjection, NamedProjectionWrite,
    NewEntityEvent, NewIndexJob, NewProjectedLaneMessage, ProjectedLaneMessage,
    ProjectionReadStore, ProjectionWriteStore, ReplayCursor, ResolveExternalIdentity, ServerRun,
    ServerRunCreate, ServerRunEvent, ServerRunReadStore, ServerRunUpdate, ServerRunWriteStore,
    StoreError, StoreResult, WorkflowDesignDelta, WorkflowDesignDeltaWrite,
    WorkflowDesignHistoryReadStore, WorkflowDesignHistoryWriteStore, WorkflowDesignSnapshot,
    WorkflowDesignSnapshotWrite, validate_entity_rebuild_request, validate_entity_recovery_request,
};
use rusqlite::{Connection, OpenFlags, OptionalExtension, TransactionBehavior, params};
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

const BUSY_TIMEOUT_MS: u64 = 30_000;
const RUNTIME_CURRENT_STATE_NAMESPACE: &str = "workflow_runtime_current_state";
static NEXT_CLAIM_TOKEN: AtomicU64 = AtomicU64::new(0);

/// Errors for direct SQLite APIs. Trait methods retain `kogwistar-store`'s
/// existing `StoreError` contract for its validation and cursor errors.
#[derive(Debug, Error)]
pub enum SqliteStoreError {
    #[error("SQLite operation failed: {0}")]
    Sql(#[from] rusqlite::Error),
    #[error("filesystem operation failed: {0}")]
    Io(#[from] std::io::Error),
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
    #[error("invalid recorded runtime transition: {0}")]
    RecordedRuntime(#[from] RecordedRuntimeError),
    #[error("recorded runtime transition conflict: {0}")]
    RecordedRuntimeConflict(String),
    #[error(transparent)]
    Store(#[from] StoreError),
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
}

/// Independent Python-compatible `AUTH_DB_URL=sqlite:///...` store. Opening
/// it creates only auth relations and never engine metadata relations.
#[derive(Clone, Debug)]
pub struct SqliteAuthStore {
    path: Arc<PathBuf>,
}

impl SqliteAuthStore {
    pub fn open(path: impl AsRef<Path>) -> SqliteStoreResult<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent().filter(|value| !value.as_os_str().is_empty()) {
            fs::create_dir_all(parent)?;
        }
        let store = Self {
            path: Arc::new(path),
        };
        let connection = store.connection()?;
        initialize_auth_schema(&connection)?;
        Ok(store)
    }

    pub fn path(&self) -> &Path {
        self.path.as_ref()
    }

    fn connection(&self) -> SqliteStoreResult<Connection> {
        let connection = Connection::open_with_flags(
            self.path.as_ref(),
            OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE,
        )?;
        connection.busy_timeout(std::time::Duration::from_millis(BUSY_TIMEOUT_MS))?;
        connection.execute_batch("PRAGMA foreign_keys = ON;")?;
        Ok(connection)
    }
}

fn sqlite_auth_user(row: &rusqlite::Row<'_>) -> rusqlite::Result<AuthUser> {
    Ok(AuthUser {
        user_id: row.get(0)?,
        email: row.get(1)?,
        display_name: row.get(2)?,
        is_active: row.get(3)?,
        global_role: row.get(4)?,
        global_ns: row.get(5)?,
    })
}

fn sqlite_store_error(error: SqliteStoreError) -> StoreError {
    StoreError::Backend {
        backend: "sqlite".to_owned(),
        message: error.to_string(),
    }
}

impl AuthIdentityStore for SqliteAuthStore {
    async fn auth_user(&self, user_id: &str) -> StoreResult<Option<AuthUser>> {
        let connection = self.connection().map_err(sqlite_store_error)?;
        connection
            .query_row(
                "SELECT user_id, email, display_name, is_active, global_role, global_ns FROM users WHERE user_id = ?1",
                [user_id],
                sqlite_auth_user,
            )
            .optional()
            .map_err(|error| sqlite_store_error(error.into()))
    }

    async fn external_identity(
        &self,
        issuer: &str,
        subject: &str,
    ) -> StoreResult<Option<ExternalIdentity>> {
        let connection = self.connection().map_err(sqlite_store_error)?;
        connection
            .query_row(
                "SELECT issuer, subject, user_id, email FROM external_identities WHERE issuer = ?1 AND subject = ?2",
                params![issuer, subject],
                |row| {
                    Ok(ExternalIdentity {
                        issuer: row.get(0)?,
                        subject: row.get(1)?,
                        user_id: row.get(2)?,
                        email: row.get(3)?,
                    })
                },
            )
            .optional()
            .map_err(|error| sqlite_store_error(error.into()))
    }

    async fn resolve_external_identity(
        &self,
        request: ResolveExternalIdentity,
    ) -> StoreResult<AuthUser> {
        let mut connection = self.connection().map_err(sqlite_store_error)?;
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|error| sqlite_store_error(error.into()))?;
        let existing_user_id = transaction
            .query_row(
                "SELECT user_id FROM external_identities WHERE issuer = ?1 AND subject = ?2",
                params![request.issuer, request.subject],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .map_err(|error| sqlite_store_error(error.into()))?;
        let user_id = if let Some(user_id) = existing_user_id {
            user_id
        } else {
            let email_user_id = transaction
                .query_row(
                    "SELECT user_id FROM users WHERE email = ?1",
                    [&request.email],
                    |row| row.get::<_, String>(0),
                )
                .optional()
                .map_err(|error| sqlite_store_error(error.into()))?;
            let user_id = email_user_id.unwrap_or_else(|| request.new_user_id.clone());
            transaction
                .execute(
                    "INSERT OR IGNORE INTO users (user_id, email, display_name, is_active, global_role, global_ns, created_at) VALUES (?1, ?2, ?3, 1, ?4, ?5, CURRENT_TIMESTAMP)",
                    params![user_id, request.email, request.display_name, request.default_role, request.default_ns],
                )
                .map_err(|error| sqlite_store_error(error.into()))?;
            transaction
                .execute(
                    "INSERT INTO external_identities (issuer, subject, user_id, email) VALUES (?1, ?2, ?3, ?4)",
                    params![request.issuer, request.subject, user_id, request.email],
                )
                .map_err(|error| sqlite_store_error(error.into()))?;
            user_id
        };
        transaction
            .execute(
                "UPDATE users SET last_login_at = CURRENT_TIMESTAMP WHERE user_id = ?1",
                [&user_id],
            )
            .map_err(|error| sqlite_store_error(error.into()))?;
        let user = transaction
            .query_row(
                "SELECT user_id, email, display_name, is_active, global_role, global_ns FROM users WHERE user_id = ?1",
                [&user_id],
                sqlite_auth_user,
            )
            .map_err(|error| sqlite_store_error(error.into()))?;
        transaction
            .commit()
            .map_err(|error| sqlite_store_error(error.into()))?;
        Ok(user)
    }
}

impl SqliteStore {
    /// Open (or create) a Python-compatible EngineSQLite database.
    ///
    /// `rusqlite` uses its `bundled` feature here so native SQLite behavior is
    /// reproducible instead of depending on a host-provided SQLite library.
    pub fn open(path: impl AsRef<Path>) -> SqliteStoreResult<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)?;
        }
        let conn = configured_connection(&path)?;
        initialize_schema(&conn)?;
        Ok(Self {
            path: Arc::new(path),
            connection: Arc::new(Mutex::new(conn)),
        })
    }

    pub fn path(&self) -> &Path {
        self.path.as_ref()
    }

    /// Run a short UoW under `BEGIN IMMEDIATE`. Returning an error rolls back
    /// every sequence, event, and cursor change performed through `uow`.
    pub fn immediate_transaction<T, F>(&self, operation: F) -> SqliteStoreResult<T>
    where
        F: FnOnce(&mut SqliteUnitOfWork<'_>) -> SqliteStoreResult<T>,
    {
        let mut conn = self
            .connection
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if conn.is_autocommit() {
            let transaction = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
            let result = {
                let mut uow = SqliteUnitOfWork {
                    transaction: &transaction,
                };
                operation(&mut uow)
            };
            match result {
                Ok(value) => {
                    transaction.commit()?;
                    Ok(value)
                }
                Err(error) => Err(error),
            }
        } else {
            let savepoint = conn.savepoint()?;
            let result = {
                let mut uow = SqliteUnitOfWork {
                    transaction: &savepoint,
                };
                operation(&mut uow)
            };
            match result {
                Ok(value) => {
                    savepoint.commit()?;
                    Ok(value)
                }
                Err(error) => Err(error),
            }
        }
    }

    /// Begin a Python-facade unit of work on the cached connection. Calls made
    /// before commit/rollback join it through per-operation savepoints.
    pub fn begin_external_transaction(&self) -> SqliteStoreResult<()> {
        let conn = self
            .connection
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if !conn.is_autocommit() {
            return Err(SqliteStoreError::TransactionAborted(
                "SQLite external transaction is already active".to_owned(),
            ));
        }
        conn.execute_batch("BEGIN IMMEDIATE")?;
        Ok(())
    }

    pub fn commit_external_transaction(&self) -> SqliteStoreResult<()> {
        let conn = self
            .connection
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if conn.is_autocommit() {
            return Err(SqliteStoreError::TransactionAborted(
                "SQLite external transaction is not active".to_owned(),
            ));
        }
        conn.execute_batch("COMMIT")?;
        Ok(())
    }

    pub fn rollback_external_transaction(&self) -> SqliteStoreResult<()> {
        let conn = self
            .connection
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if conn.is_autocommit() {
            return Err(SqliteStoreError::TransactionAborted(
                "SQLite external transaction is not active".to_owned(),
            ));
        }
        conn.execute_batch("ROLLBACK")?;
        Ok(())
    }

    /// Alias for `immediate_transaction`; all store UoWs are immediate.
    pub fn transaction<T, F>(&self, operation: F) -> SqliteStoreResult<T>
    where
        F: FnOnce(&mut SqliteUnitOfWork<'_>) -> SqliteStoreResult<T>,
    {
        self.immediate_transaction(operation)
    }

    pub fn next_global_seq(&self) -> SqliteStoreResult<i64> {
        self.immediate_transaction(|uow| uow.next_global_seq())
    }

    pub fn current_global_seq(&self) -> SqliteStoreResult<i64> {
        self.with_connection(current_global_seq)
    }

    pub fn next_user_seq(&self, user_id: &str) -> SqliteStoreResult<i64> {
        self.immediate_transaction(|uow| uow.next_user_seq(user_id))
    }

    pub fn current_user_seq(&self, user_id: &str) -> SqliteStoreResult<i64> {
        self.with_connection(|conn| current_user_seq(conn, user_id))
    }

    pub fn set_user_seq(&self, user_id: &str, value: i64) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| uow.set_user_seq(user_id, value))
    }

    /// Python-compatible alias for `next_user_seq`.
    pub fn next_scoped_seq(&self, scope_id: &str) -> SqliteStoreResult<i64> {
        self.next_user_seq(scope_id)
    }

    /// Python-compatible alias for `current_user_seq`.
    pub fn current_scoped_seq(&self, scope_id: &str) -> SqliteStoreResult<i64> {
        self.current_user_seq(scope_id)
    }

    /// Python-compatible alias for `set_user_seq`.
    pub fn set_scoped_seq(&self, scope_id: &str, value: i64) -> SqliteStoreResult<()> {
        self.set_user_seq(scope_id, value)
    }

    /// Allocate a namespace-local event sequence without appending an event.
    pub fn alloc_event_seq(&self, namespace: &str) -> SqliteStoreResult<i64> {
        self.immediate_transaction(|uow| uow.alloc_event_seq(namespace))
    }

    pub fn append_raw_entity_event(
        &self,
        namespace: &str,
        event: NewRawEntityEvent,
    ) -> SqliteStoreResult<AppendedRawEvent> {
        self.immediate_transaction(|uow| uow.append_raw_entity_event(namespace, event))
    }

    /// Raw replay is exclusive of `after_seq`, preserving stored JSON text.
    pub fn replay_raw_events(
        &self,
        namespace: &str,
        after_seq: i64,
        limit: usize,
    ) -> SqliteStoreResult<Vec<RawEntityEvent>> {
        self.with_connection(|conn| replay_raw_events(conn, namespace, after_seq, limit))
    }

    pub fn latest_retained_event_seq(&self, namespace: &str) -> SqliteStoreResult<i64> {
        self.with_connection(|conn| latest_retained_event_seq(conn, namespace))
    }

    /// Delete namespace events whose sequence is strictly above `to_seq`.
    pub fn prune_entity_events_after(
        &self,
        namespace: &str,
        to_seq: i64,
    ) -> SqliteStoreResult<u64> {
        self.immediate_transaction(|uow| uow.prune_entity_events_after(namespace, to_seq))
    }

    /// Strict cursor advance: bounded by retained events and monotonic.
    pub fn strict_advance_replay_cursor(
        &self,
        namespace: &str,
        consumer: &str,
        last_seq: i64,
    ) -> SqliteStoreResult<ReplayCursor> {
        self.immediate_transaction(|uow| {
            uow.strict_advance_replay_cursor(namespace, consumer, last_seq)
        })
    }

    /// Python `cursor_set` semantics: direct overwrite, including regression or
    /// a value beyond retained events. This is intentionally separate from the
    /// strict store trait for a future permissive Python adapter.
    pub fn set_replay_cursor_legacy(
        &self,
        namespace: &str,
        consumer: &str,
        last_seq: i64,
    ) -> SqliteStoreResult<ReplayCursor> {
        self.immediate_transaction(|uow| {
            uow.set_replay_cursor_legacy(namespace, consumer, last_seq)
        })
    }

    pub fn replay_cursor(
        &self,
        namespace: &str,
        consumer: &str,
    ) -> SqliteStoreResult<ReplayCursor> {
        self.with_connection(|conn| replay_cursor(conn, namespace, consumer))
    }

    pub fn index_applied_fingerprint(
        &self,
        namespace: &str,
        coalesce_key: &str,
    ) -> SqliteStoreResult<Option<String>> {
        self.with_connection(|conn| {
            conn.query_row(
                "SELECT applied_fingerprint FROM index_applied_state WHERE namespace=?1 AND coalesce_key=?2",
                params![namespace, coalesce_key],
                |row| row.get(0),
            )
            .optional()
            .map_err(Into::into)
        })
    }

    pub fn set_index_applied_fingerprint(
        &self,
        namespace: &str,
        coalesce_key: &str,
        applied_fingerprint: Option<&str>,
        last_job_id: Option<&str>,
    ) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| {
            uow.set_index_applied_fingerprint(
                namespace,
                coalesce_key,
                applied_fingerprint,
                last_job_id,
            )
        })
    }

    /// Bounded operator-triggered recovery. Event history remains authoritative;
    /// projection replacement and durable cursor advance share one SQLite UoW.
    pub fn recover_entity_projection(
        &self,
        request: EntityRecoveryRequest,
    ) -> SqliteStoreResult<EntityRecoveryReport> {
        validate_entity_recovery_request(&request)?;
        self.immediate_transaction(|uow| uow.recover_entity_projection(&request))
    }

    /// Explicit full rebuild. Ordinary store opening never calls this method.
    pub fn rebuild_entity_projection(
        &self,
        request: EntityRebuildRequest,
    ) -> SqliteStoreResult<EntityRecoveryReport> {
        validate_entity_rebuild_request(&request)?;
        self.immediate_transaction(|uow| uow.rebuild_entity_projection(&request))
    }

    pub fn get_named_projection(
        &self,
        namespace: &str,
        key: &str,
    ) -> SqliteStoreResult<Option<NamedProjection>> {
        self.with_connection(|conn| named_projection(conn, namespace, key))
    }

    pub fn list_named_projections(
        &self,
        namespace: &str,
    ) -> SqliteStoreResult<Vec<NamedProjection>> {
        self.with_connection(|conn| named_projections(conn, namespace))
    }

    pub fn replace_named_projection(
        &self,
        namespace: &str,
        key: &str,
        projection: NamedProjectionWrite,
    ) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| uow.replace_named_projection(namespace, key, projection))
    }

    pub fn compare_and_swap_named_projection(
        &self,
        namespace: &str,
        key: &str,
        expected_last_authoritative_seq: Option<i64>,
        expected_last_materialized_seq: Option<i64>,
        projection: NamedProjectionWrite,
    ) -> SqliteStoreResult<bool> {
        self.immediate_transaction(|uow| {
            uow.compare_and_swap_named_projection(
                namespace,
                key,
                expected_last_authoritative_seq,
                expected_last_materialized_seq,
                projection,
            )
        })
    }

    pub fn clear_named_projection(&self, namespace: &str, key: &str) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| uow.clear_named_projection(namespace, key))
    }

    pub fn clear_projection_namespace(&self, namespace: &str) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| uow.clear_projection_namespace(namespace))
    }

    pub fn put_workflow_design_snapshot(
        &self,
        workflow_id: &str,
        snapshot: WorkflowDesignSnapshotWrite,
    ) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| uow.put_workflow_design_snapshot(workflow_id, snapshot))
    }

    pub fn get_workflow_design_snapshot(
        &self,
        workflow_id: &str,
        max_version: i64,
        schema_version: i64,
    ) -> SqliteStoreResult<Option<WorkflowDesignSnapshot>> {
        self.with_connection(|conn| {
            workflow_design_snapshot(conn, workflow_id, max_version, schema_version)
        })
    }

    pub fn clear_workflow_design_snapshots(&self, workflow_id: &str) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| uow.clear_workflow_design_snapshots(workflow_id))
    }

    pub fn put_workflow_design_delta(
        &self,
        workflow_id: &str,
        delta: WorkflowDesignDeltaWrite,
    ) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| uow.put_workflow_design_delta(workflow_id, delta))
    }

    pub fn get_workflow_design_delta(
        &self,
        workflow_id: &str,
        version: i64,
        schema_version: i64,
    ) -> SqliteStoreResult<Option<WorkflowDesignDelta>> {
        self.with_connection(|conn| {
            workflow_design_delta(conn, workflow_id, version, schema_version)
        })
    }

    pub fn clear_workflow_design_deltas(&self, workflow_id: &str) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| uow.clear_workflow_design_deltas(workflow_id))
    }

    pub fn create_server_run(&self, run: ServerRunCreate) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| uow.create_server_run(run))
    }
    pub fn get_server_run(&self, run_id: &str) -> SqliteStoreResult<Option<ServerRun>> {
        self.with_connection(|conn| server_run(conn, run_id))
    }
    pub fn list_server_runs(
        &self,
        status: Option<&str>,
        workflow_id: Option<&str>,
        conversation_id: Option<&str>,
        limit: usize,
    ) -> SqliteStoreResult<Vec<ServerRun>> {
        self.with_connection(|conn| server_runs(conn, status, workflow_id, conversation_id, limit))
    }
    pub fn append_server_run_event(
        &self,
        run_id: &str,
        event_type: &str,
        payload_json: String,
    ) -> SqliteStoreResult<ServerRunEvent> {
        self.immediate_transaction(|uow| {
            uow.append_server_run_event(run_id, event_type, payload_json)
        })
    }
    pub fn list_server_run_events(
        &self,
        run_id: &str,
        after_seq: i64,
        limit: usize,
    ) -> SqliteStoreResult<Vec<ServerRunEvent>> {
        self.with_connection(|conn| server_run_events(conn, run_id, after_seq, limit))
    }
    pub fn update_server_run(
        &self,
        run_id: &str,
        update: ServerRunUpdate,
    ) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| uow.update_server_run(run_id, update))
    }
    pub fn request_server_run_cancel(&self, run_id: &str) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| uow.request_server_run_cancel(run_id))
    }

    /// Apply one already-recorded runtime result under one `BEGIN IMMEDIATE`
    /// transaction.  This operation never invokes a resolver, provider, tool,
    /// lane, graph, or Python callback.
    pub fn apply_recorded_runtime_transition(
        &self,
        transition: RecordedRuntimeTransition,
        abort_after_writes: bool,
    ) -> SqliteStoreResult<RecordedTransitionResult> {
        self.immediate_transaction(|uow| {
            uow.apply_recorded_runtime_transition(transition, abort_after_writes)
        })
    }

    /// Atomically apply a recorded worker result and acknowledge the exact
    /// claimed lane request.  No callback or dispatch runs inside this UoW.
    pub fn apply_claimed_recorded_runtime_transition(
        &self,
        handoff: RecordedWorkerHandoff,
        transition: RecordedRuntimeTransition,
        abort_after_writes: bool,
    ) -> SqliteStoreResult<RecordedTransitionResult> {
        self.immediate_transaction(|uow| {
            uow.apply_claimed_recorded_runtime_transition(handoff, transition, abort_after_writes)
        })
    }

    pub fn apply_claimed_recorded_worker_effect(
        &self,
        handoff: RecordedWorkerHandoff,
        effect: RecordedWorkerSuccessEffect,
    ) -> SqliteStoreResult<RecordedTransitionResult> {
        self.immediate_transaction(|uow| uow.apply_claimed_recorded_worker_effect(handoff, effect))
    }

    pub fn resume_recorded_runtime_token(
        &self,
        run_id: &str,
        workflow_id: &str,
        conversation_id: &str,
        node_id: &str,
        token_id: &str,
        resume_payload: Option<Value>,
    ) -> SqliteStoreResult<RecordedTransitionResult> {
        let run_id = run_id.to_owned();
        let workflow_id = workflow_id.to_owned();
        let conversation_id = conversation_id.to_owned();
        let node_id = node_id.to_owned();
        let token_id = token_id.to_owned();
        self.immediate_transaction(|uow| {
            let current = read_recorded_runtime_state(
                uow.transaction,
                &run_id,
                &workflow_id,
                &conversation_id,
            )?
            .ok_or_else(|| {
                SqliteStoreError::RecordedRuntimeConflict(format!(
                    "run {run_id:?} has no recorded runtime state"
                ))
            })?;
            let parent = current
                .frontier
                .suspended
                .iter()
                .find(|(node, _, token, _)| node == &node_id && token == &token_id)
                .and_then(|(_, _, _, parent)| parent.clone());
            let frontier = frontier_after_worker_resume(
                &current.frontier,
                &node_id,
                &token_id,
                parent.as_deref(),
            )?;
            let expected_event_seq = latest_server_run_event_seq(uow.transaction, &run_id)?;
            let result = uow.apply_recorded_runtime_transition(
                RecordedRuntimeTransition {
                    contract_version: RECORDED_RUNTIME_CONTRACT_VERSION,
                    transition_id: format!("resume-{run_id}-{token_id}-{expected_event_seq}"),
                    expected_event_seq,
                    kind: kogwistar_runtime::RecordedTransitionKind::ResumeResult,
                    run_id: run_id.clone(),
                    workflow_id: workflow_id.clone(),
                    conversation_id: conversation_id.clone(),
                    user_id: None,
                    user_turn_node_id: None,
                    step_seq: current.last_step_seq.saturating_add(1),
                    node_id: Some(node_id.clone()),
                    token_id: Some(token_id.clone()),
                    parent_token_id: parent.clone(),
                    initial_state: None,
                    state_update: Vec::new(),
                    update: None,
                    state_schema: Map::new(),
                    frontier: Some(frontier),
                    result: None,
                    wait_reason: None,
                    resume_payload,
                    errors: Vec::new(),
                },
                false,
            )?;
            let message_id = format!(
                "lane|{}",
                kogwistar_contracts::stable_id(
                    "runtime.worker.request",
                    &[
                        result.reduced.state.run_id.clone(),
                        token_id.clone(),
                        node_id.clone(),
                        result
                            .reduced
                            .state
                            .last_step_seq
                            .saturating_add(1)
                            .to_string(),
                    ],
                )
            );
            let now = unix_epoch_seconds();
            uow.project_lane_message(NewProjectedLaneMessage {
                message_id,
                namespace: "workflow".to_owned(),
                purpose: "system".to_owned(),
                inbox_id: "workflow-runtime".to_owned(),
                conversation_id: conversation_id.clone(),
                recipient_id: "python-worker".to_owned(),
                sender_id: "rust-scheduler".to_owned(),
                msg_type: "workflow.step.execute".to_owned(),
                status: "pending".to_owned(),
                created_at: now,
                available_at: now,
                run_id: Some(run_id.clone()),
                step_id: Some(node_id.clone()),
                correlation_id: Some(run_id.clone()),
                payload_json: Some(serde_json::to_string(&serde_json::json!({
                    "contract_version": 1,
                    "kind": "workflow.step.execute",
                    "run_id": run_id,
                    "workflow_id": workflow_id,
                    "conversation_id": conversation_id,
                    "node_id": node_id,
                    "join_mask": result.reduced.state.frontier.pending
                        .iter()
                        .find(|(_, _, token, _)| token == &token_id)
                        .map(|(_, mask, _, _)| *mask)
                        .unwrap_or(0),
                    "token_id": token_id,
                    "parent_token_id": parent,
                    "step_seq": result.reduced.state.last_step_seq.saturating_add(1),
                    "expected_event_seq": result.event_seq,
                    "state": result.reduced.state.state,
                    "resume_payload": result.reduced.state.resume_payload,
                }))?),
                error_json: None,
            })?;
            Ok(result)
        })
    }

    /// Reopen/read path for recorded runtime recovery.  Pending tokens are
    /// already normalized to Python's pending-plus-inflight recovery frontier;
    /// suspended tokens remain separately parked in `frontier.suspended`.
    pub fn read_recorded_runtime_state(
        &self,
        run_id: &str,
        workflow_id: &str,
        conversation_id: &str,
    ) -> SqliteStoreResult<Option<RecordedRuntimeState>> {
        self.with_connection(|conn| {
            read_recorded_runtime_state(conn, run_id, workflow_id, conversation_id)
        })
    }

    pub fn enqueue_index_job(&self, job: NewIndexJob) -> SqliteStoreResult<String> {
        self.immediate_transaction(|uow| uow.enqueue_index_job(job))
    }
    pub fn claim_index_jobs(
        &self,
        limit: usize,
        lease_seconds: i64,
        namespace: Option<&str>,
    ) -> SqliteStoreResult<Vec<IndexJob>> {
        self.immediate_transaction(|uow| uow.claim_index_jobs(limit, lease_seconds, namespace))
    }
    pub fn mark_index_job_done(
        &self,
        job_id: &str,
        claim_token: Option<&str>,
    ) -> SqliteStoreResult<bool> {
        self.immediate_transaction(|uow| uow.mark_index_job_done(job_id, claim_token))
    }
    pub fn mark_index_job_failed(
        &self,
        job_id: &str,
        error: &str,
        final_: bool,
        claim_token: Option<&str>,
    ) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| {
            uow.mark_index_job_failed(job_id, error, final_, claim_token)
        })
    }
    pub fn bump_retry_and_requeue(
        &self,
        job_id: &str,
        error: &str,
        delay_seconds: i64,
        claim_token: Option<&str>,
    ) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| {
            uow.bump_retry_and_requeue(job_id, error, delay_seconds, claim_token)
        })
    }
    pub fn renew_index_job_lease(
        &self,
        job_id: &str,
        claim_token: &str,
        lease_seconds: i64,
    ) -> SqliteStoreResult<bool> {
        self.immediate_transaction(|uow| {
            uow.renew_index_job_lease(job_id, claim_token, lease_seconds)
        })
    }
    pub fn requeue_index_job_at_tail(
        &self,
        job_id: &str,
        payload_json: String,
        delay_seconds: i64,
        claim_token: Option<&str>,
    ) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| {
            uow.requeue_index_job_at_tail(job_id, payload_json, delay_seconds, claim_token)
        })
    }
    pub fn list_index_jobs(
        &self,
        namespace: Option<&str>,
        status: Option<&str>,
        entity_kind: Option<&str>,
        entity_id: Option<&str>,
        index_kind: Option<&str>,
        limit: usize,
    ) -> SqliteStoreResult<Vec<IndexJob>> {
        self.with_connection(|conn| {
            list_index_jobs(
                conn,
                namespace,
                status,
                entity_kind,
                entity_id,
                index_kind,
                limit,
            )
        })
    }

    pub fn project_lane_message(&self, row: NewProjectedLaneMessage) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| uow.project_lane_message(row))
    }
    pub fn get_projected_lane_message(
        &self,
        message_id: &str,
    ) -> SqliteStoreResult<Option<ProjectedLaneMessage>> {
        self.with_connection(|conn| projected_lane_message(conn, message_id))
    }
    pub fn list_projected_lane_messages(
        &self,
        filter: LaneMessageFilter,
    ) -> SqliteStoreResult<Vec<ProjectedLaneMessage>> {
        self.with_connection(|conn| list_projected_lane_messages(conn, &filter))
    }
    pub fn update_projected_lane_message_status(
        &self,
        message_id: &str,
        status: &str,
        error_json: Option<String>,
    ) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| {
            uow.update_projected_lane_message_status(message_id, status, error_json)
        })
    }
    pub fn update_projected_lane_message_links(
        &self,
        message_id: &str,
        prev: Option<String>,
        next: Option<String>,
        inbox_tail: Option<String>,
        conversation_tail: Option<String>,
    ) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| {
            uow.update_projected_lane_message_links(
                message_id,
                prev,
                next,
                inbox_tail,
                conversation_tail,
            )
        })
    }
    pub fn clear_projected_lane_messages(&self, namespace: &str) -> SqliteStoreResult<u64> {
        self.immediate_transaction(|uow| uow.clear_projected_lane_messages(namespace))
    }
    pub fn claim_projected_lane_messages(
        &self,
        namespace: &str,
        inbox_id: &str,
        claimed_by: &str,
        limit: usize,
        lease_seconds: i64,
    ) -> SqliteStoreResult<Vec<ProjectedLaneMessage>> {
        self.immediate_transaction(|uow| {
            uow.claim_projected_lane_messages(namespace, inbox_id, claimed_by, limit, lease_seconds)
        })
    }
    pub fn ack_projected_lane_message(
        &self,
        message_id: &str,
        claimed_by: &str,
    ) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| uow.ack_projected_lane_message(message_id, claimed_by))
    }
    pub fn requeue_projected_lane_message(
        &self,
        message_id: &str,
        claimed_by: &str,
        error_json: Option<String>,
        delay_seconds: i64,
    ) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| {
            uow.requeue_projected_lane_message(message_id, claimed_by, error_json, delay_seconds)
        })
    }
    pub fn dead_letter_projected_lane_message(
        &self,
        message_id: &str,
        claimed_by: &str,
        error_json: Option<String>,
    ) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| {
            uow.dead_letter_projected_lane_message(message_id, claimed_by, error_json)
        })
    }
    pub fn repair_orphaned_claimed_lane_messages(
        &self,
        namespace: &str,
        inbox_id: Option<&str>,
        limit: usize,
    ) -> SqliteStoreResult<Vec<String>> {
        self.immediate_transaction(|uow| {
            uow.repair_orphaned_claimed_lane_messages(namespace, inbox_id, limit)
        })
    }

    fn with_connection<T>(
        &self,
        operation: impl FnOnce(&Connection) -> SqliteStoreResult<T>,
    ) -> SqliteStoreResult<T> {
        let conn = self
            .connection
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        operation(&conn)
    }
}

fn configured_connection(path: &Path) -> SqliteStoreResult<Connection> {
    let conn = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE,
    )?;
    conn.busy_timeout(std::time::Duration::from_millis(BUSY_TIMEOUT_MS))?;
    conn.execute_batch(
        "PRAGMA foreign_keys = ON; PRAGMA journal_mode = WAL; PRAGMA synchronous = NORMAL;",
    )?;
    Ok(conn)
}

/// Operations sharing one `BEGIN IMMEDIATE` transaction.
pub struct SqliteUnitOfWork<'connection> {
    transaction: &'connection Connection,
}

impl SqliteUnitOfWork<'_> {
    pub fn set_index_applied_fingerprint(
        &mut self,
        namespace: &str,
        coalesce_key: &str,
        applied_fingerprint: Option<&str>,
        last_job_id: Option<&str>,
    ) -> SqliteStoreResult<()> {
        self.transaction.execute(
            "INSERT INTO index_applied_state(namespace,coalesce_key,applied_fingerprint,applied_at,last_job_id) VALUES(?1,?2,?3,?4,?5) ON CONFLICT(namespace,coalesce_key) DO UPDATE SET applied_fingerprint=excluded.applied_fingerprint,applied_at=excluded.applied_at,last_job_id=excluded.last_job_id",
            params![namespace, coalesce_key, applied_fingerprint, unix_epoch_seconds(), last_job_id],
        )?;
        Ok(())
    }

    pub fn project_lane_message(&mut self, row: NewProjectedLaneMessage) -> SqliteStoreResult<()> {
        project_lane_message(self.transaction, row)
    }
    pub fn get_projected_lane_message(
        &self,
        message_id: &str,
    ) -> SqliteStoreResult<Option<ProjectedLaneMessage>> {
        projected_lane_message(self.transaction, message_id)
    }
    pub fn get_server_run(&self, run_id: &str) -> SqliteStoreResult<Option<ServerRun>> {
        server_run(self.transaction, run_id)
    }
    pub fn list_projected_lane_messages(
        &self,
        filter: &LaneMessageFilter,
    ) -> SqliteStoreResult<Vec<ProjectedLaneMessage>> {
        list_projected_lane_messages(self.transaction, filter)
    }
    pub fn update_projected_lane_message_status(
        &mut self,
        message_id: &str,
        status: &str,
        error_json: Option<String>,
    ) -> SqliteStoreResult<()> {
        update_projected_lane_message_status(self.transaction, message_id, status, error_json)
    }
    pub fn update_projected_lane_message_links(
        &mut self,
        message_id: &str,
        prev: Option<String>,
        next: Option<String>,
        inbox_tail: Option<String>,
        conversation_tail: Option<String>,
    ) -> SqliteStoreResult<()> {
        update_projected_lane_message_links(
            self.transaction,
            message_id,
            prev,
            next,
            inbox_tail,
            conversation_tail,
        )
    }
    pub fn update_projected_lane_message_payload(
        &mut self,
        message_id: &str,
        payload_json: String,
    ) -> SqliteStoreResult<()> {
        let changed = self.transaction.execute(
            "UPDATE projected_lane_messages SET payload_json=?1 WHERE message_id=?2",
            params![payload_json, message_id],
        )?;
        if changed != 1 {
            return Err(SqliteStoreError::RecordedRuntimeConflict(format!(
                "lane message {message_id:?} payload update matched {changed} rows"
            )));
        }
        Ok(())
    }
    pub fn clear_projected_lane_messages(&mut self, namespace: &str) -> SqliteStoreResult<u64> {
        clear_projected_lane_messages(self.transaction, namespace)
    }
    pub fn claim_projected_lane_messages(
        &mut self,
        namespace: &str,
        inbox_id: &str,
        claimed_by: &str,
        limit: usize,
        lease_seconds: i64,
    ) -> SqliteStoreResult<Vec<ProjectedLaneMessage>> {
        claim_projected_lane_messages(
            self.transaction,
            namespace,
            inbox_id,
            claimed_by,
            limit,
            lease_seconds,
        )
    }
    pub fn ack_projected_lane_message(
        &mut self,
        message_id: &str,
        claimed_by: &str,
    ) -> SqliteStoreResult<()> {
        ack_projected_lane_message(self.transaction, message_id, claimed_by)
    }
    pub fn requeue_projected_lane_message(
        &mut self,
        message_id: &str,
        claimed_by: &str,
        error_json: Option<String>,
        delay_seconds: i64,
    ) -> SqliteStoreResult<()> {
        requeue_projected_lane_message(
            self.transaction,
            message_id,
            claimed_by,
            error_json,
            delay_seconds,
        )
    }
    pub fn dead_letter_projected_lane_message(
        &mut self,
        message_id: &str,
        claimed_by: &str,
        error_json: Option<String>,
    ) -> SqliteStoreResult<()> {
        dead_letter_projected_lane_message(self.transaction, message_id, claimed_by, error_json)
    }
    pub fn repair_orphaned_claimed_lane_messages(
        &mut self,
        namespace: &str,
        inbox_id: Option<&str>,
        limit: usize,
    ) -> SqliteStoreResult<Vec<String>> {
        repair_orphaned_claimed_lane_messages(self.transaction, namespace, inbox_id, limit)
    }
    pub fn enqueue_index_job(&mut self, job: NewIndexJob) -> SqliteStoreResult<String> {
        enqueue_index_job(self.transaction, job)
    }
    pub fn claim_index_jobs(
        &mut self,
        limit: usize,
        lease_seconds: i64,
        namespace: Option<&str>,
    ) -> SqliteStoreResult<Vec<IndexJob>> {
        claim_index_jobs(self.transaction, limit, lease_seconds, namespace)
    }
    pub fn mark_index_job_done(
        &mut self,
        job_id: &str,
        claim_token: Option<&str>,
    ) -> SqliteStoreResult<bool> {
        mark_index_job_done(self.transaction, job_id, claim_token)
    }
    pub fn mark_index_job_failed(
        &mut self,
        job_id: &str,
        error: &str,
        final_: bool,
        claim_token: Option<&str>,
    ) -> SqliteStoreResult<()> {
        mark_index_job_failed(self.transaction, job_id, error, final_, claim_token)
    }
    pub fn bump_retry_and_requeue(
        &mut self,
        job_id: &str,
        error: &str,
        delay_seconds: i64,
        claim_token: Option<&str>,
    ) -> SqliteStoreResult<()> {
        bump_retry_and_requeue(self.transaction, job_id, error, delay_seconds, claim_token)
    }
    pub fn renew_index_job_lease(
        &mut self,
        job_id: &str,
        claim_token: &str,
        lease_seconds: i64,
    ) -> SqliteStoreResult<bool> {
        renew_index_job_lease(self.transaction, job_id, claim_token, lease_seconds)
    }
    pub fn requeue_index_job_at_tail(
        &mut self,
        job_id: &str,
        payload_json: String,
        delay_seconds: i64,
        claim_token: Option<&str>,
    ) -> SqliteStoreResult<()> {
        requeue_index_job_at_tail(
            self.transaction,
            job_id,
            payload_json,
            delay_seconds,
            claim_token,
        )
    }
    pub fn next_global_seq(&mut self) -> SqliteStoreResult<i64> {
        self.transaction
            .query_row(
                "UPDATE global_seq SET value = value + 1 RETURNING value",
                [],
                |row| row.get(0),
            )
            .map_err(Into::into)
    }

    pub fn current_global_seq(&self) -> SqliteStoreResult<i64> {
        current_global_seq(self.transaction)
    }

    pub fn next_user_seq(&mut self, user_id: &str) -> SqliteStoreResult<i64> {
        self.transaction
            .query_row(
                "INSERT INTO user_seq(user_id, value) VALUES (?1, 1) \
                 ON CONFLICT(user_id) DO UPDATE SET value = user_seq.value + 1 \
                 RETURNING value",
                [user_id],
                |row| row.get(0),
            )
            .map_err(Into::into)
    }

    pub fn current_user_seq(&self, user_id: &str) -> SqliteStoreResult<i64> {
        current_user_seq(self.transaction, user_id)
    }

    pub fn set_user_seq(&mut self, user_id: &str, value: i64) -> SqliteStoreResult<()> {
        if value < 0 {
            return Err(SqliteStoreError::NegativeSequenceValue { value });
        }
        self.transaction.execute(
            "INSERT INTO user_seq(user_id, value) VALUES (?1, ?2) \
             ON CONFLICT(user_id) DO UPDATE SET value = excluded.value",
            params![user_id, value],
        )?;
        Ok(())
    }

    pub fn next_scoped_seq(&mut self, scope_id: &str) -> SqliteStoreResult<i64> {
        self.next_user_seq(scope_id)
    }

    pub fn current_scoped_seq(&self, scope_id: &str) -> SqliteStoreResult<i64> {
        self.current_user_seq(scope_id)
    }

    pub fn set_scoped_seq(&mut self, scope_id: &str, value: i64) -> SqliteStoreResult<()> {
        self.set_user_seq(scope_id, value)
    }

    pub fn alloc_event_seq(&mut self, namespace: &str) -> SqliteStoreResult<i64> {
        self.transaction
            .query_row(
                "INSERT INTO namespace_seq(namespace, next_seq) VALUES (?1, 2) \
                 ON CONFLICT(namespace) DO UPDATE SET next_seq = namespace_seq.next_seq + 1 \
                 RETURNING next_seq - 1",
                [namespace],
                |row| row.get(0),
            )
            .map_err(Into::into)
    }

    pub fn append_raw_entity_event(
        &mut self,
        namespace: &str,
        event: NewRawEntityEvent,
    ) -> SqliteStoreResult<AppendedRawEvent> {
        append_raw_entity_event(self.transaction, namespace, event)
    }

    pub fn replay_raw_events(
        &self,
        namespace: &str,
        after_seq: i64,
        limit: usize,
    ) -> SqliteStoreResult<Vec<RawEntityEvent>> {
        replay_raw_events(self.transaction, namespace, after_seq, limit)
    }

    pub fn latest_retained_event_seq(&self, namespace: &str) -> SqliteStoreResult<i64> {
        latest_retained_event_seq(self.transaction, namespace)
    }

    pub fn prune_entity_events_after(
        &mut self,
        namespace: &str,
        to_seq: i64,
    ) -> SqliteStoreResult<u64> {
        prune_entity_events_after(self.transaction, namespace, to_seq)
    }

    pub fn replay_cursor(
        &self,
        namespace: &str,
        consumer: &str,
    ) -> SqliteStoreResult<ReplayCursor> {
        replay_cursor(self.transaction, namespace, consumer)
    }

    pub fn recover_entity_projection(
        &mut self,
        request: &EntityRecoveryRequest,
    ) -> SqliteStoreResult<EntityRecoveryReport> {
        recover_entity_projection_uow(self, request)
    }

    pub fn rebuild_entity_projection(
        &mut self,
        request: &EntityRebuildRequest,
    ) -> SqliteStoreResult<EntityRecoveryReport> {
        rebuild_entity_projection_uow(self, request)
    }

    pub fn strict_advance_replay_cursor(
        &mut self,
        namespace: &str,
        consumer: &str,
        last_seq: i64,
    ) -> SqliteStoreResult<ReplayCursor> {
        strict_advance_replay_cursor(self.transaction, namespace, consumer, last_seq)
    }

    pub fn set_replay_cursor_legacy(
        &mut self,
        namespace: &str,
        consumer: &str,
        last_seq: i64,
    ) -> SqliteStoreResult<ReplayCursor> {
        set_replay_cursor_legacy(self.transaction, namespace, consumer, last_seq)
    }

    pub fn replace_named_projection(
        &mut self,
        namespace: &str,
        key: &str,
        projection: NamedProjectionWrite,
    ) -> SqliteStoreResult<()> {
        replace_named_projection(self.transaction, namespace, key, projection)
    }

    pub fn compare_and_swap_named_projection(
        &mut self,
        namespace: &str,
        key: &str,
        expected_last_authoritative_seq: Option<i64>,
        expected_last_materialized_seq: Option<i64>,
        projection: NamedProjectionWrite,
    ) -> SqliteStoreResult<bool> {
        compare_and_swap_named_projection(
            self.transaction,
            namespace,
            key,
            expected_last_authoritative_seq,
            expected_last_materialized_seq,
            projection,
        )
    }

    pub fn clear_named_projection(&mut self, namespace: &str, key: &str) -> SqliteStoreResult<()> {
        clear_named_projection(self.transaction, namespace, key)
    }

    pub fn clear_projection_namespace(&mut self, namespace: &str) -> SqliteStoreResult<()> {
        clear_projection_namespace(self.transaction, namespace)
    }

    pub fn put_workflow_design_snapshot(
        &mut self,
        workflow_id: &str,
        snapshot: WorkflowDesignSnapshotWrite,
    ) -> SqliteStoreResult<()> {
        put_workflow_design_snapshot(self.transaction, workflow_id, snapshot)
    }

    pub fn get_workflow_design_snapshot(
        &self,
        workflow_id: &str,
        max_version: i64,
        schema_version: i64,
    ) -> SqliteStoreResult<Option<WorkflowDesignSnapshot>> {
        workflow_design_snapshot(self.transaction, workflow_id, max_version, schema_version)
    }

    pub fn clear_workflow_design_snapshots(&mut self, workflow_id: &str) -> SqliteStoreResult<()> {
        clear_workflow_design_snapshots(self.transaction, workflow_id)
    }

    pub fn put_workflow_design_delta(
        &mut self,
        workflow_id: &str,
        delta: WorkflowDesignDeltaWrite,
    ) -> SqliteStoreResult<()> {
        put_workflow_design_delta(self.transaction, workflow_id, delta)
    }

    pub fn get_workflow_design_delta(
        &self,
        workflow_id: &str,
        version: i64,
        schema_version: i64,
    ) -> SqliteStoreResult<Option<WorkflowDesignDelta>> {
        workflow_design_delta(self.transaction, workflow_id, version, schema_version)
    }

    pub fn clear_workflow_design_deltas(&mut self, workflow_id: &str) -> SqliteStoreResult<()> {
        clear_workflow_design_deltas(self.transaction, workflow_id)
    }

    pub fn create_server_run(&mut self, run: ServerRunCreate) -> SqliteStoreResult<()> {
        create_server_run(self.transaction, run)
    }
    pub fn append_server_run_event(
        &mut self,
        run_id: &str,
        event_type: &str,
        payload_json: String,
    ) -> SqliteStoreResult<ServerRunEvent> {
        append_server_run_event(self.transaction, run_id, event_type, payload_json)
    }
    pub fn update_server_run(
        &mut self,
        run_id: &str,
        update: ServerRunUpdate,
    ) -> SqliteStoreResult<()> {
        update_server_run(self.transaction, run_id, update)
    }
    pub fn request_server_run_cancel(&mut self, run_id: &str) -> SqliteStoreResult<()> {
        request_server_run_cancel(self.transaction, run_id)
    }

    pub fn apply_recorded_runtime_transition(
        &mut self,
        transition: RecordedRuntimeTransition,
        abort_after_writes: bool,
    ) -> SqliteStoreResult<RecordedTransitionResult> {
        apply_recorded_runtime_transition(self.transaction, transition, abort_after_writes)
    }

    pub fn apply_claimed_recorded_runtime_transition(
        &mut self,
        handoff: RecordedWorkerHandoff,
        transition: RecordedRuntimeTransition,
        abort_after_writes: bool,
    ) -> SqliteStoreResult<RecordedTransitionResult> {
        apply_claimed_recorded_runtime_transition(
            self.transaction,
            handoff,
            transition,
            abort_after_writes,
        )
    }

    pub fn apply_claimed_recorded_worker_effect(
        &mut self,
        handoff: RecordedWorkerHandoff,
        effect: RecordedWorkerSuccessEffect,
    ) -> SqliteStoreResult<RecordedTransitionResult> {
        apply_claimed_recorded_worker_effect(self.transaction, handoff, effect)
    }
}

impl ServerRunReadStore for SqliteStore {
    async fn server_run(&self, run_id: &str) -> StoreResult<Option<ServerRun>> {
        SqliteStore::get_server_run(self, run_id).map_err(trait_error)
    }

    async fn server_runs(
        &self,
        status: Option<&str>,
        workflow_id: Option<&str>,
        conversation_id: Option<&str>,
        limit: usize,
    ) -> StoreResult<Vec<ServerRun>> {
        SqliteStore::list_server_runs(self, status, workflow_id, conversation_id, limit)
            .map_err(trait_error)
    }

    async fn server_run_events(
        &self,
        run_id: &str,
        after_seq: i64,
        limit: usize,
    ) -> StoreResult<Vec<ServerRunEvent>> {
        SqliteStore::list_server_run_events(self, run_id, after_seq, limit).map_err(trait_error)
    }
}

impl ServerRunWriteStore for SqliteStore {
    async fn create_server_run(&self, run: ServerRunCreate) -> StoreResult<()> {
        SqliteStore::create_server_run(self, run).map_err(trait_error)
    }

    async fn append_server_run_event(
        &self,
        run_id: &str,
        event_type: &str,
        payload_json: String,
    ) -> StoreResult<ServerRunEvent> {
        SqliteStore::append_server_run_event(self, run_id, event_type, payload_json)
            .map_err(trait_error)
    }

    async fn update_server_run(&self, run_id: &str, update: ServerRunUpdate) -> StoreResult<()> {
        SqliteStore::update_server_run(self, run_id, update).map_err(trait_error)
    }

    async fn request_server_run_cancel(&self, run_id: &str) -> StoreResult<()> {
        SqliteStore::request_server_run_cancel(self, run_id).map_err(trait_error)
    }
}

impl IndexJobReadStore for SqliteStore {
    async fn index_jobs(
        &self,
        namespace: Option<&str>,
        status: Option<&str>,
        entity_kind: Option<&str>,
        entity_id: Option<&str>,
        index_kind: Option<&str>,
        limit: usize,
    ) -> StoreResult<Vec<IndexJob>> {
        self.list_index_jobs(namespace, status, entity_kind, entity_id, index_kind, limit)
            .map_err(trait_error)
    }
}
impl IndexJobWriteStore for SqliteStore {
    async fn enqueue_index_job(&self, job: NewIndexJob) -> StoreResult<String> {
        SqliteStore::enqueue_index_job(self, job).map_err(trait_error)
    }
    async fn claim_index_jobs(
        &self,
        limit: usize,
        lease_seconds: i64,
        namespace: Option<&str>,
    ) -> StoreResult<Vec<IndexJob>> {
        SqliteStore::claim_index_jobs(self, limit, lease_seconds, namespace).map_err(trait_error)
    }
    async fn mark_index_job_done(
        &self,
        job_id: &str,
        claim_token: Option<&str>,
    ) -> StoreResult<bool> {
        SqliteStore::mark_index_job_done(self, job_id, claim_token).map_err(trait_error)
    }
    async fn mark_index_job_failed(
        &self,
        job_id: &str,
        error: &str,
        final_: bool,
        claim_token: Option<&str>,
    ) -> StoreResult<()> {
        SqliteStore::mark_index_job_failed(self, job_id, error, final_, claim_token)
            .map_err(trait_error)
    }
    async fn bump_retry_and_requeue(
        &self,
        job_id: &str,
        error: &str,
        delay: i64,
        claim_token: Option<&str>,
    ) -> StoreResult<()> {
        SqliteStore::bump_retry_and_requeue(self, job_id, error, delay, claim_token)
            .map_err(trait_error)
    }
    async fn renew_index_job_lease(
        &self,
        job_id: &str,
        claim_token: &str,
        lease_seconds: i64,
    ) -> StoreResult<bool> {
        SqliteStore::renew_index_job_lease(self, job_id, claim_token, lease_seconds)
            .map_err(trait_error)
    }
    async fn requeue_index_job_at_tail(
        &self,
        job_id: &str,
        payload_json: String,
        delay: i64,
        claim_token: Option<&str>,
    ) -> StoreResult<()> {
        SqliteStore::requeue_index_job_at_tail(self, job_id, payload_json, delay, claim_token)
            .map_err(trait_error)
    }
}

impl LaneMessageReadStore for SqliteStore {
    async fn projected_lane_message(
        &self,
        message_id: &str,
    ) -> StoreResult<Option<ProjectedLaneMessage>> {
        self.get_projected_lane_message(message_id)
            .map_err(trait_error)
    }
    async fn projected_lane_messages(
        &self,
        filter: LaneMessageFilter,
    ) -> StoreResult<Vec<ProjectedLaneMessage>> {
        self.list_projected_lane_messages(filter)
            .map_err(trait_error)
    }
}
impl LaneMessageWriteStore for SqliteStore {
    async fn project_lane_message(&self, row: NewProjectedLaneMessage) -> StoreResult<()> {
        SqliteStore::project_lane_message(self, row).map_err(trait_error)
    }
    async fn update_projected_lane_message_status(
        &self,
        id: &str,
        status: &str,
        error: Option<String>,
    ) -> StoreResult<()> {
        SqliteStore::update_projected_lane_message_status(self, id, status, error)
            .map_err(trait_error)
    }
    async fn update_projected_lane_message_links(
        &self,
        id: &str,
        prev: Option<String>,
        next: Option<String>,
        inbox_tail: Option<String>,
        conversation_tail: Option<String>,
    ) -> StoreResult<()> {
        SqliteStore::update_projected_lane_message_links(
            self,
            id,
            prev,
            next,
            inbox_tail,
            conversation_tail,
        )
        .map_err(trait_error)
    }
    async fn clear_projected_lane_messages(&self, namespace: &str) -> StoreResult<u64> {
        SqliteStore::clear_projected_lane_messages(self, namespace).map_err(trait_error)
    }
    async fn claim_projected_lane_messages(
        &self,
        namespace: &str,
        inbox: &str,
        owner: &str,
        limit: usize,
        lease: i64,
    ) -> StoreResult<Vec<ProjectedLaneMessage>> {
        SqliteStore::claim_projected_lane_messages(self, namespace, inbox, owner, limit, lease)
            .map_err(trait_error)
    }
    async fn ack_projected_lane_message(&self, id: &str, owner: &str) -> StoreResult<()> {
        SqliteStore::ack_projected_lane_message(self, id, owner).map_err(trait_error)
    }
    async fn requeue_projected_lane_message(
        &self,
        id: &str,
        owner: &str,
        error: Option<String>,
        delay: i64,
    ) -> StoreResult<()> {
        SqliteStore::requeue_projected_lane_message(self, id, owner, error, delay)
            .map_err(trait_error)
    }
    async fn dead_letter_projected_lane_message(
        &self,
        id: &str,
        owner: &str,
        error: Option<String>,
    ) -> StoreResult<()> {
        SqliteStore::dead_letter_projected_lane_message(self, id, owner, error).map_err(trait_error)
    }
}

impl EventReadStore for SqliteStore {
    async fn replay_events(
        &self,
        namespace: &str,
        after_seq: i64,
        limit: usize,
    ) -> StoreResult<Vec<EntityEvent>> {
        require_trait_namespace(namespace)?;
        self.replay_raw_events(namespace, after_seq, limit)
            .map_err(trait_error)?
            .into_iter()
            .map(raw_to_entity_event)
            .collect::<SqliteStoreResult<Vec<_>>>()
            .map_err(trait_error)
    }

    async fn replay_cursor(&self, namespace: &str, consumer: &str) -> StoreResult<ReplayCursor> {
        require_trait_namespace(namespace)?;
        SqliteStore::replay_cursor(self, namespace, consumer).map_err(trait_error)
    }

    async fn latest_event_seq(&self, namespace: &str) -> StoreResult<i64> {
        require_trait_namespace(namespace)?;
        self.latest_retained_event_seq(namespace)
            .map_err(trait_error)
    }
}

impl EventWriteStore for SqliteStore {
    async fn append_entity_event(
        &self,
        namespace: &str,
        event: NewEntityEvent,
    ) -> StoreResult<AppendedEvent> {
        require_trait_namespace(namespace)?;
        let raw = self
            .append_raw_entity_event(
                namespace,
                NewRawEntityEvent {
                    event_id: event.event_id,
                    entity_kind: event.entity_kind,
                    entity_id: event.entity_id,
                    op: event.op,
                    payload_json: serde_json::to_string(&event.payload)
                        .expect("serde_json::Value serialization cannot fail"),
                },
            )
            .map_err(trait_error)?;
        Ok(AppendedEvent {
            event: raw_to_entity_event(raw.event).map_err(trait_error)?,
            inserted: raw.inserted,
        })
    }

    async fn advance_replay_cursor(
        &self,
        namespace: &str,
        consumer: &str,
        last_seq: i64,
    ) -> StoreResult<ReplayCursor> {
        require_trait_namespace(namespace)?;
        self.strict_advance_replay_cursor(namespace, consumer, last_seq)
            .map_err(trait_error)
    }
}

impl EventPruneStore for SqliteStore {
    async fn prune_entity_events_after(&self, namespace: &str, to_seq: i64) -> StoreResult<u64> {
        require_trait_namespace(namespace)?;
        SqliteStore::prune_entity_events_after(self, namespace, to_seq).map_err(trait_error)
    }
}

impl ProjectionReadStore for SqliteStore {
    async fn named_projection(
        &self,
        namespace: &str,
        key: &str,
    ) -> StoreResult<Option<NamedProjection>> {
        require_trait_namespace(namespace)?;
        self.get_named_projection(namespace, key)
            .map_err(trait_error)
    }

    async fn named_projections(&self, namespace: &str) -> StoreResult<Vec<NamedProjection>> {
        require_trait_namespace(namespace)?;
        self.list_named_projections(namespace).map_err(trait_error)
    }
}

impl ProjectionWriteStore for SqliteStore {
    async fn replace_named_projection(
        &self,
        namespace: &str,
        key: &str,
        projection: NamedProjectionWrite,
    ) -> StoreResult<()> {
        require_trait_namespace(namespace)?;
        SqliteStore::replace_named_projection(self, namespace, key, projection).map_err(trait_error)
    }

    async fn compare_and_swap_named_projection(
        &self,
        namespace: &str,
        key: &str,
        expected_last_authoritative_seq: Option<i64>,
        expected_last_materialized_seq: Option<i64>,
        projection: NamedProjectionWrite,
    ) -> StoreResult<bool> {
        require_trait_namespace(namespace)?;
        SqliteStore::compare_and_swap_named_projection(
            self,
            namespace,
            key,
            expected_last_authoritative_seq,
            expected_last_materialized_seq,
            projection,
        )
        .map_err(trait_error)
    }

    async fn clear_named_projection(&self, namespace: &str, key: &str) -> StoreResult<()> {
        require_trait_namespace(namespace)?;
        SqliteStore::clear_named_projection(self, namespace, key).map_err(trait_error)
    }

    async fn clear_projection_namespace(&self, namespace: &str) -> StoreResult<()> {
        require_trait_namespace(namespace)?;
        SqliteStore::clear_projection_namespace(self, namespace).map_err(trait_error)
    }
}

impl WorkflowDesignHistoryReadStore for SqliteStore {
    async fn workflow_design_snapshot(
        &self,
        workflow_id: &str,
        max_version: i64,
        schema_version: i64,
    ) -> StoreResult<Option<WorkflowDesignSnapshot>> {
        SqliteStore::get_workflow_design_snapshot(self, workflow_id, max_version, schema_version)
            .map_err(trait_error)
    }

    async fn workflow_design_delta(
        &self,
        workflow_id: &str,
        version: i64,
        schema_version: i64,
    ) -> StoreResult<Option<WorkflowDesignDelta>> {
        SqliteStore::get_workflow_design_delta(self, workflow_id, version, schema_version)
            .map_err(trait_error)
    }
}

impl WorkflowDesignHistoryWriteStore for SqliteStore {
    async fn put_workflow_design_snapshot(
        &self,
        workflow_id: &str,
        snapshot: WorkflowDesignSnapshotWrite,
    ) -> StoreResult<()> {
        SqliteStore::put_workflow_design_snapshot(self, workflow_id, snapshot).map_err(trait_error)
    }

    async fn clear_workflow_design_snapshots(&self, workflow_id: &str) -> StoreResult<()> {
        SqliteStore::clear_workflow_design_snapshots(self, workflow_id).map_err(trait_error)
    }

    async fn put_workflow_design_delta(
        &self,
        workflow_id: &str,
        delta: WorkflowDesignDeltaWrite,
    ) -> StoreResult<()> {
        SqliteStore::put_workflow_design_delta(self, workflow_id, delta).map_err(trait_error)
    }

    async fn clear_workflow_design_deltas(&self, workflow_id: &str) -> StoreResult<()> {
        SqliteStore::clear_workflow_design_deltas(self, workflow_id).map_err(trait_error)
    }
}

fn initialize_schema(conn: &Connection) -> SqliteStoreResult<()> {
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS global_seq (
            value INTEGER NOT NULL
        );
        INSERT OR IGNORE INTO global_seq(rowid, value) VALUES (1, 0);
        CREATE TABLE IF NOT EXISTS user_seq (
            user_id TEXT PRIMARY KEY,
            value   INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS namespace_seq (
            namespace TEXT PRIMARY KEY,
            next_seq   INTEGER NOT NULL
        );
        INSERT OR IGNORE INTO namespace_seq(namespace, next_seq) VALUES ('default', 1);
        CREATE TABLE IF NOT EXISTS entity_events (
            namespace    TEXT NOT NULL DEFAULT 'default',
            seq          INTEGER NOT NULL,
            event_id     TEXT NOT NULL,
            entity_kind  TEXT NOT NULL,
            entity_id    TEXT NOT NULL,
            op           TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at   INTEGER NOT NULL,
            PRIMARY KEY(namespace, seq),
            UNIQUE(event_id)
        );
        CREATE INDEX IF NOT EXISTS idx_entity_events_aggregate
        ON entity_events(namespace, entity_kind, entity_id, seq);
        CREATE TABLE IF NOT EXISTS replay_cursors (
            namespace  TEXT NOT NULL DEFAULT 'default',
            consumer   TEXT NOT NULL,
            last_seq   INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            PRIMARY KEY(namespace, consumer)
        );
        CREATE TABLE IF NOT EXISTS named_projections (
            namespace TEXT NOT NULL,
            key TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            last_authoritative_seq INTEGER NOT NULL,
            last_materialized_seq INTEGER NOT NULL,
            projection_schema_version INTEGER NOT NULL,
            materialization_status TEXT NOT NULL,
            updated_at_ms INTEGER NOT NULL,
            PRIMARY KEY(namespace, key)
        );
        CREATE INDEX IF NOT EXISTS idx_named_projections_namespace
        ON named_projections(namespace, updated_at_ms);
        CREATE TABLE IF NOT EXISTS workflow_design_snapshots (
            workflow_id TEXT NOT NULL,
            version INTEGER NOT NULL,
            seq INTEGER NOT NULL,
            payload_json TEXT NOT NULL,
            schema_version INTEGER NOT NULL,
            created_at_ms INTEGER NOT NULL,
            PRIMARY KEY(workflow_id, version)
        );
        CREATE TABLE IF NOT EXISTS workflow_design_version_deltas (
            workflow_id TEXT NOT NULL,
            version INTEGER NOT NULL,
            prev_version INTEGER NOT NULL,
            target_seq INTEGER NOT NULL,
            forward_json TEXT NOT NULL,
            inverse_json TEXT NOT NULL,
            schema_version INTEGER NOT NULL,
            created_at_ms INTEGER NOT NULL,
            PRIMARY KEY(workflow_id, version)
        );
        CREATE TABLE IF NOT EXISTS server_runs (
            run_id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            workflow_id TEXT NOT NULL,
            user_id TEXT,
            user_turn_node_id TEXT,
            assistant_turn_node_id TEXT,
            status TEXT NOT NULL,
            cancel_requested INTEGER NOT NULL DEFAULT 0,
            result_json TEXT,
            error_json TEXT,
            created_at_ms INTEGER NOT NULL,
            updated_at_ms INTEGER NOT NULL,
            started_at_ms INTEGER,
            finished_at_ms INTEGER
        );
        CREATE TABLE IF NOT EXISTS server_run_events (
            seq INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at_ms INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_server_runs_status ON server_runs(status, updated_at_ms);
        CREATE INDEX IF NOT EXISTS idx_server_run_events_run_seq ON server_run_events(run_id, seq);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_server_run_events_recorded_transition_id
        ON server_run_events(json_extract(payload_json, '$.transition_id'))
        WHERE event_type = 'workflow.recorded_transition.v1';
        CREATE TABLE IF NOT EXISTS index_jobs (
            job_id TEXT PRIMARY KEY,
            namespace TEXT NOT NULL DEFAULT 'default',
            entity_kind TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            index_kind TEXT NOT NULL,
            coalesce_key TEXT NOT NULL,
            op TEXT NOT NULL,
            status TEXT NOT NULL,
            lease_until INTEGER,
            next_run_at INTEGER,
            max_retries INTEGER NOT NULL DEFAULT 10,
            retry_count INTEGER NOT NULL DEFAULT 0,
            last_error TEXT,
            payload_json TEXT,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            claim_token TEXT
        );
        CREATE TABLE IF NOT EXISTS index_applied_state (
            namespace TEXT NOT NULL DEFAULT 'default',
            coalesce_key TEXT NOT NULL,
            applied_fingerprint TEXT,
            applied_at INTEGER NOT NULL,
            last_job_id TEXT,
            PRIMARY KEY(namespace, coalesce_key)
        );
        CREATE INDEX IF NOT EXISTS idx_index_applied_state_key
        ON index_applied_state(coalesce_key);
        CREATE TABLE IF NOT EXISTS projected_lane_messages (
            message_id TEXT PRIMARY KEY, namespace TEXT NOT NULL DEFAULT 'default', purpose TEXT NOT NULL DEFAULT 'user_visible',
            inbox_id TEXT NOT NULL, conversation_id TEXT NOT NULL, recipient_id TEXT NOT NULL, sender_id TEXT NOT NULL,
            msg_type TEXT NOT NULL, status TEXT NOT NULL, seq INTEGER NOT NULL, conversation_seq INTEGER NOT NULL,
            claimed_by TEXT, lease_until INTEGER, retry_count INTEGER NOT NULL DEFAULT 0, created_at INTEGER NOT NULL,
            available_at INTEGER NOT NULL, run_id TEXT, step_id TEXT, correlation_id TEXT, payload_json TEXT, error_json TEXT,
            prev_message_id TEXT, next_message_id TEXT, inbox_tail_message_id TEXT, conversation_tail_message_id TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_lane_messages_namespace_inbox_seq ON projected_lane_messages(namespace, inbox_id, seq);
        CREATE INDEX IF NOT EXISTS idx_lane_messages_claim ON projected_lane_messages(namespace, inbox_id, status, available_at, lease_until);
        CREATE INDEX IF NOT EXISTS idx_lane_messages_conversation_seq ON projected_lane_messages(namespace, conversation_id, conversation_seq);
        ",
    )?;
    // Python-created databases may predate queue coalescing, scheduling, and
    // lease ownership. Add columns before creating indexes that reference
    // them so every historical Python schema remains openable by Rust.
    let index_job_columns = conn
        .prepare("PRAGMA table_info(index_jobs)")?
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<Result<Vec<_>, _>>()?;
    if !index_job_columns.iter().any(|column| column == "namespace") {
        conn.execute(
            "ALTER TABLE index_jobs ADD COLUMN namespace TEXT NOT NULL DEFAULT 'default'",
            [],
        )?;
    }
    if !index_job_columns
        .iter()
        .any(|column| column == "coalesce_key")
    {
        conn.execute(
            "ALTER TABLE index_jobs ADD COLUMN coalesce_key TEXT NOT NULL DEFAULT ''",
            [],
        )?;
    }
    if !index_job_columns
        .iter()
        .any(|column| column == "next_run_at")
    {
        conn.execute("ALTER TABLE index_jobs ADD COLUMN next_run_at INTEGER", [])?;
    }
    if !index_job_columns
        .iter()
        .any(|column| column == "max_retries")
    {
        conn.execute(
            "ALTER TABLE index_jobs ADD COLUMN max_retries INTEGER NOT NULL DEFAULT 10",
            [],
        )?;
    }
    if !index_job_columns
        .iter()
        .any(|column| column == "claim_token")
    {
        conn.execute("ALTER TABLE index_jobs ADD COLUMN claim_token TEXT", [])?;
    }
    let lane_columns = conn
        .prepare("PRAGMA table_info(projected_lane_messages)")?
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<Result<Vec<_>, _>>()?;
    if !lane_columns.iter().any(|column| column == "purpose") {
        conn.execute(
            "ALTER TABLE projected_lane_messages ADD COLUMN purpose TEXT NOT NULL DEFAULT 'user_visible'",
            [],
        )?;
    }
    conn.execute_batch(
        "CREATE INDEX IF NOT EXISTS idx_index_jobs_status_lease ON index_jobs(status, lease_until);
         CREATE INDEX IF NOT EXISTS idx_index_jobs_entity ON index_jobs(entity_kind, entity_id, index_kind);
         CREATE INDEX IF NOT EXISTS idx_index_jobs_namespace ON index_jobs(namespace);
         CREATE UNIQUE INDEX IF NOT EXISTS uq_index_jobs_pending_ns_coalesce
         ON index_jobs(namespace, coalesce_key) WHERE status='PENDING';",
    )?;
    Ok(())
}

fn initialize_auth_schema(conn: &Connection) -> SqliteStoreResult<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS users (
            user_id VARCHAR(64) NOT NULL PRIMARY KEY,
            email VARCHAR(255) NOT NULL UNIQUE,
            display_name VARCHAR(255),
            is_active BOOLEAN NOT NULL,
            global_role VARCHAR(32),
            global_ns VARCHAR(255),
            created_at DATETIME NOT NULL,
            last_login_at DATETIME
        );
        CREATE INDEX IF NOT EXISTS ix_users_email ON users (email);
        CREATE TABLE IF NOT EXISTS external_identities (
            issuer VARCHAR(255) NOT NULL,
            subject VARCHAR(255) NOT NULL,
            user_id VARCHAR(64) NOT NULL,
            email VARCHAR(255),
            PRIMARY KEY (issuer, subject),
            FOREIGN KEY(user_id) REFERENCES users (user_id)
        );
        CREATE INDEX IF NOT EXISTS ix_external_identities_user_id ON external_identities (user_id);
        CREATE TABLE IF NOT EXISTS workflow_acl (
            workflow_id VARCHAR(255) NOT NULL,
            user_id VARCHAR(64) NOT NULL,
            role VARCHAR(32) NOT NULL,
            PRIMARY KEY (workflow_id, user_id),
            FOREIGN KEY(user_id) REFERENCES users (user_id)
        );",
    )?;
    Ok(())
}

fn enqueue_index_job(conn: &Connection, job: NewIndexJob) -> SqliteStoreResult<String> {
    require_queue_namespace(&job.namespace)?;
    let now = unix_epoch_seconds();
    let key = job.coalesce_key();
    if let Some((job_id, existing_op)) = conn.query_row(
        "SELECT job_id, op FROM index_jobs WHERE namespace=?1 AND coalesce_key=?2 AND status='PENDING' ORDER BY created_at ASC LIMIT 1",
        params![job.namespace, key], |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
    ).optional()? {
        let op = if job.op == "DELETE" || existing_op == "DELETE" { "DELETE" } else { &job.op };
        conn.execute("UPDATE index_jobs SET op=?1,payload_json=?2,updated_at=?3 WHERE job_id=?4", params![op, job.payload_json, now, job_id])?;
        return Ok(job_id);
    }
    conn.execute(
        "INSERT OR IGNORE INTO index_jobs(job_id,namespace,entity_kind,entity_id,index_kind,coalesce_key,op,status,lease_until,next_run_at,max_retries,retry_count,last_error,payload_json,created_at,updated_at,claim_token) VALUES (?1,?2,?3,?4,?5,?6,?7,'PENDING',NULL,NULL,?8,0,NULL,?9,?10,?10,NULL)",
        params![job.job_id, job.namespace, job.entity_kind, job.entity_id, job.index_kind, key, job.op, job.max_retries, job.payload_json, now],
    )?;
    Ok(job.job_id)
}

fn claim_index_jobs(
    conn: &Connection,
    limit: usize,
    lease_seconds: i64,
    namespace: Option<&str>,
) -> SqliteStoreResult<Vec<IndexJob>> {
    if limit == 0 {
        return Ok(Vec::new());
    }
    let now = unix_epoch_seconds();
    let lease_until = now + lease_seconds;
    let token = format!(
        "sqlite-{}-{}-{}",
        now,
        std::process::id(),
        NEXT_CLAIM_TOKEN.fetch_add(1, Ordering::Relaxed)
    );
    let namespace_sql = if namespace.is_some() {
        "AND namespace=?3"
    } else {
        ""
    };
    let sql = format!(
        "WITH candidates AS (SELECT job_id FROM index_jobs WHERE ((status='PENDING' AND (next_run_at IS NULL OR next_run_at <= ?1)) OR (status='DOING' AND lease_until IS NOT NULL AND lease_until < ?1)) {namespace_sql} ORDER BY created_at ASC,job_id ASC LIMIT ?2) UPDATE index_jobs SET status='DOING',lease_until=?4,claim_token=?5,updated_at=?1 WHERE job_id IN (SELECT job_id FROM candidates) RETURNING job_id,namespace,entity_kind,entity_id,index_kind,coalesce_key,op,status,lease_until,next_run_at,max_retries,retry_count,last_error,payload_json,created_at,updated_at,claim_token"
    );
    let mut statement = conn.prepare(&sql)?;
    let rows = if let Some(namespace) = namespace {
        statement.query_map(
            params![
                now,
                i64::try_from(limit).unwrap_or(i64::MAX),
                namespace,
                lease_until,
                token
            ],
            index_job_from_row,
        )?
    } else {
        statement.query_map(
            params![
                now,
                i64::try_from(limit).unwrap_or(i64::MAX),
                Option::<String>::None,
                lease_until,
                token
            ],
            index_job_from_row,
        )?
    };
    rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
}

fn mark_index_job_done(
    conn: &Connection,
    job_id: &str,
    claim_token: Option<&str>,
) -> SqliteStoreResult<bool> {
    Ok(conn.execute("UPDATE index_jobs SET status='DONE',lease_until=NULL,claim_token=NULL,updated_at=?1 WHERE job_id=?2 AND status='DOING' AND (?3 IS NULL OR claim_token=?3)", params![unix_epoch_seconds(), job_id, claim_token])? != 0)
}
fn mark_index_job_failed(
    conn: &Connection,
    job_id: &str,
    error: &str,
    final_: bool,
    claim_token: Option<&str>,
) -> SqliteStoreResult<()> {
    conn.execute("UPDATE index_jobs SET status=CASE WHEN ?1 THEN 'FAILED' ELSE status END,lease_until=NULL,claim_token=NULL,last_error=?2,updated_at=?3 WHERE job_id=?4 AND status='DOING' AND (?5 IS NULL OR claim_token=?5)", params![final_, truncate_error(error), unix_epoch_seconds(), job_id, claim_token])?;
    Ok(())
}
fn bump_retry_and_requeue(
    conn: &Connection,
    job_id: &str,
    error: &str,
    delay_seconds: i64,
    claim_token: Option<&str>,
) -> SqliteStoreResult<()> {
    let now = unix_epoch_seconds();
    conn.execute("UPDATE index_jobs SET retry_count=retry_count+1,last_error=?1,status=CASE WHEN retry_count+1>=max_retries THEN 'FAILED' ELSE 'PENDING' END,lease_until=NULL,claim_token=NULL,next_run_at=CASE WHEN retry_count+1>=max_retries THEN NULL ELSE ?2 END,updated_at=?3 WHERE job_id=?4 AND status='DOING' AND (?5 IS NULL OR claim_token=?5)", params![truncate_error(error), now+delay_seconds.max(0), now, job_id, claim_token])?;
    Ok(())
}
fn renew_index_job_lease(
    conn: &Connection,
    job_id: &str,
    claim_token: &str,
    lease_seconds: i64,
) -> SqliteStoreResult<bool> {
    let now = unix_epoch_seconds();
    Ok(conn.execute("UPDATE index_jobs SET lease_until=?1,updated_at=?2 WHERE job_id=?3 AND status='DOING' AND claim_token=?4 AND (lease_until IS NULL OR lease_until>=?2)", params![now+lease_seconds,now,job_id,claim_token])? != 0)
}
fn requeue_index_job_at_tail(
    conn: &Connection,
    job_id: &str,
    payload_json: String,
    delay_seconds: i64,
    claim_token: Option<&str>,
) -> SqliteStoreResult<()> {
    let now = unix_epoch_seconds();
    let tail: i64 = conn.query_row(
        "SELECT COALESCE(MAX(created_at),?1)+1 FROM index_jobs",
        [now],
        |row| row.get(0),
    )?;
    conn.execute("UPDATE index_jobs SET status='PENDING',lease_until=NULL,claim_token=NULL,next_run_at=?1,payload_json=?2,created_at=?3,updated_at=?4 WHERE job_id=?5 AND status='DOING' AND (?6 IS NULL OR claim_token=?6)", params![now+delay_seconds.max(0),payload_json,tail,now,job_id,claim_token])?;
    Ok(())
}
fn list_index_jobs(
    conn: &Connection,
    namespace: Option<&str>,
    status: Option<&str>,
    entity_kind: Option<&str>,
    entity_id: Option<&str>,
    index_kind: Option<&str>,
    limit: usize,
) -> SqliteStoreResult<Vec<IndexJob>> {
    let query = "SELECT job_id,namespace,entity_kind,entity_id,index_kind,coalesce_key,op,status,lease_until,next_run_at,max_retries,retry_count,last_error,payload_json,created_at,updated_at,claim_token FROM index_jobs WHERE (?1 IS NULL OR namespace=?1) AND (?2 IS NULL OR status=?2) AND (?3 IS NULL OR entity_kind=?3) AND (?4 IS NULL OR entity_id=?4) AND (?5 IS NULL OR index_kind=?5) ORDER BY created_at ASC,job_id ASC LIMIT ?6";
    let mut statement = conn.prepare(query)?;
    let rows = statement.query_map(
        params![
            namespace,
            status,
            entity_kind,
            entity_id,
            index_kind,
            i64::try_from(limit).unwrap_or(i64::MAX)
        ],
        index_job_from_row,
    )?;
    rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
}
fn index_job_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<IndexJob> {
    Ok(IndexJob {
        job_id: row.get(0)?,
        namespace: row.get(1)?,
        entity_kind: row.get(2)?,
        entity_id: row.get(3)?,
        index_kind: row.get(4)?,
        coalesce_key: row.get(5)?,
        op: row.get(6)?,
        status: row.get(7)?,
        lease_until: row.get::<_, Option<i64>>(8)?.map(serde_json::Value::from),
        next_run_at: row.get::<_, Option<i64>>(9)?.map(serde_json::Value::from),
        max_retries: row.get(10)?,
        retry_count: row.get(11)?,
        last_error: row.get(12)?,
        payload_json: row.get(13)?,
        created_at: serde_json::Value::from(row.get::<_, i64>(14)?),
        updated_at: serde_json::Value::from(row.get::<_, i64>(15)?),
        claim_token: row.get(16)?,
        claim_attempts: 0,
    })
}
fn require_queue_namespace(namespace: &str) -> SqliteStoreResult<()> {
    if namespace.is_empty() {
        Err(SqliteStoreError::EmptyNamespace)
    } else {
        Ok(())
    }
}

const LANE_SELECT: &str = "message_id,namespace,purpose,inbox_id,conversation_id,recipient_id,sender_id,msg_type,status,seq,conversation_seq,claimed_by,lease_until,retry_count,created_at,available_at,run_id,step_id,correlation_id,payload_json,error_json,prev_message_id,next_message_id,inbox_tail_message_id,conversation_tail_message_id";
fn lane_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<ProjectedLaneMessage> {
    Ok(ProjectedLaneMessage {
        message_id: row.get(0)?,
        namespace: row.get(1)?,
        purpose: row.get(2)?,
        inbox_id: row.get(3)?,
        conversation_id: row.get(4)?,
        recipient_id: row.get(5)?,
        sender_id: row.get(6)?,
        msg_type: row.get(7)?,
        status: row.get(8)?,
        seq: row.get(9)?,
        conversation_seq: row.get(10)?,
        claimed_by: row.get(11)?,
        lease_until: row.get::<_, Option<i64>>(12)?.map(serde_json::Value::from),
        retry_count: row.get(13)?,
        created_at: row.get(14)?,
        available_at: row.get(15)?,
        run_id: row.get(16)?,
        step_id: row.get(17)?,
        correlation_id: row.get(18)?,
        payload_json: row.get(19)?,
        error_json: row.get(20)?,
        prev_message_id: row.get(21)?,
        next_message_id: row.get(22)?,
        inbox_tail_message_id: row.get(23)?,
        conversation_tail_message_id: row.get(24)?,
    })
}
fn projected_lane_message(
    conn: &Connection,
    id: &str,
) -> SqliteStoreResult<Option<ProjectedLaneMessage>> {
    conn.query_row(
        &format!("SELECT {LANE_SELECT} FROM projected_lane_messages WHERE message_id=?1"),
        [id],
        lane_from_row,
    )
    .optional()
    .map_err(Into::into)
}
fn project_lane_message(conn: &Connection, row: NewProjectedLaneMessage) -> SqliteStoreResult<()> {
    require_queue_namespace(&row.namespace)?;
    let seq:i64=conn.query_row("SELECT COALESCE(MAX(seq),0)+1 FROM projected_lane_messages WHERE namespace=?1 AND inbox_id=?2",params![row.namespace,row.inbox_id],|r|r.get(0))?;
    let conversation_seq:i64=conn.query_row("SELECT COALESCE(MAX(conversation_seq),0)+1 FROM projected_lane_messages WHERE namespace=?1 AND conversation_id=?2",params![row.namespace,row.conversation_id],|r|r.get(0))?;
    let prev:Option<String>=conn.query_row("SELECT message_id FROM projected_lane_messages WHERE namespace=?1 AND inbox_id=?2 ORDER BY seq DESC,created_at DESC LIMIT 1",params![row.namespace,row.inbox_id],|r|r.get(0)).optional()?;
    conn.execute("INSERT OR IGNORE INTO projected_lane_messages(message_id,namespace,purpose,inbox_id,conversation_id,recipient_id,sender_id,msg_type,status,seq,conversation_seq,claimed_by,lease_until,retry_count,created_at,available_at,run_id,step_id,correlation_id,payload_json,error_json,prev_message_id,next_message_id,inbox_tail_message_id,conversation_tail_message_id) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,NULL,NULL,0,?12,?13,?14,?15,?16,?17,?18,?19,NULL,?1,?1)",params![row.message_id,row.namespace,if row.purpose.is_empty(){"user_visible".to_owned()}else{row.purpose},row.inbox_id,row.conversation_id,row.recipient_id,row.sender_id,row.msg_type,row.status,seq,conversation_seq,row.created_at,row.available_at,row.run_id,row.step_id,row.correlation_id,row.payload_json,row.error_json,prev])?;
    Ok(())
}
fn update_projected_lane_message_status(
    conn: &Connection,
    id: &str,
    status: &str,
    error: Option<String>,
) -> SqliteStoreResult<()> {
    conn.execute("UPDATE projected_lane_messages SET status=?1,error_json=COALESCE(?2,error_json),claimed_by=CASE WHEN ?1 IN ('completed','failed','cancelled') THEN NULL ELSE claimed_by END,lease_until=CASE WHEN ?1 IN ('completed','failed','cancelled') THEN NULL ELSE lease_until END WHERE message_id=?3",params![status,error,id])?;
    Ok(())
}
fn update_projected_lane_message_links(
    conn: &Connection,
    id: &str,
    prev: Option<String>,
    next: Option<String>,
    inbox_tail: Option<String>,
    conversation_tail: Option<String>,
) -> SqliteStoreResult<()> {
    conn.execute("UPDATE projected_lane_messages SET prev_message_id=?1,next_message_id=?2,inbox_tail_message_id=?3,conversation_tail_message_id=?4 WHERE message_id=?5",params![prev,next,inbox_tail,conversation_tail,id])?;
    Ok(())
}
fn clear_projected_lane_messages(conn: &Connection, namespace: &str) -> SqliteStoreResult<u64> {
    Ok(conn.execute(
        "DELETE FROM projected_lane_messages WHERE namespace=?1",
        [namespace],
    )? as u64)
}
fn claim_projected_lane_messages(
    conn: &Connection,
    namespace: &str,
    inbox: &str,
    owner: &str,
    limit: usize,
    lease: i64,
) -> SqliteStoreResult<Vec<ProjectedLaneMessage>> {
    if limit == 0 {
        return Ok(vec![]);
    }
    let now = unix_epoch_seconds();
    let until = now + lease;
    let mut stmt=conn.prepare("SELECT message_id FROM projected_lane_messages WHERE namespace=?1 AND inbox_id=?2 AND ((status='pending' AND available_at<=?3) OR (status='claimed' AND lease_until IS NOT NULL AND lease_until<?3)) ORDER BY seq ASC,created_at ASC LIMIT ?4")?;
    let ids = stmt
        .query_map(
            params![
                namespace,
                inbox,
                now,
                i64::try_from(limit).unwrap_or(i64::MAX)
            ],
            |r| r.get::<_, String>(0),
        )?
        .collect::<Result<Vec<_>, _>>()?;
    for id in &ids {
        conn.execute("UPDATE projected_lane_messages SET status='claimed',claimed_by=?1,lease_until=?2 WHERE message_id=?3",params![owner,until,id])?;
    }
    let mut out = Vec::with_capacity(ids.len());
    for id in ids {
        if let Some(row) = projected_lane_message(conn, &id)? {
            out.push(row)
        }
    }
    Ok(out)
}
fn ack_projected_lane_message(conn: &Connection, id: &str, owner: &str) -> SqliteStoreResult<()> {
    conn.execute("UPDATE projected_lane_messages SET status='completed',claimed_by=NULL,lease_until=NULL WHERE message_id=?1 AND status NOT IN ('completed','failed','cancelled','dead-letter') AND (claimed_by IS NULL OR claimed_by=?2)",params![id,owner])?;
    Ok(())
}
fn requeue_projected_lane_message(
    conn: &Connection,
    id: &str,
    owner: &str,
    error: Option<String>,
    delay: i64,
) -> SqliteStoreResult<()> {
    conn.execute("UPDATE projected_lane_messages SET status='pending',claimed_by=NULL,lease_until=NULL,retry_count=retry_count+1,available_at=?1,error_json=COALESCE(?2,error_json) WHERE message_id=?3 AND status NOT IN ('completed','failed','cancelled','dead-letter') AND (claimed_by IS NULL OR claimed_by=?4)",params![unix_epoch_seconds()+delay.max(0),error,id,owner])?;
    Ok(())
}
fn dead_letter_projected_lane_message(
    conn: &Connection,
    id: &str,
    owner: &str,
    error: Option<String>,
) -> SqliteStoreResult<()> {
    conn.execute("UPDATE projected_lane_messages SET status='dead-letter',claimed_by=NULL,lease_until=NULL,error_json=COALESCE(?1,error_json) WHERE message_id=?2 AND status NOT IN ('completed','failed','cancelled','dead-letter') AND (claimed_by IS NULL OR claimed_by=?3)",params![error,id,owner])?;
    Ok(())
}
fn repair_orphaned_claimed_lane_messages(
    conn: &Connection,
    namespace: &str,
    inbox_id: Option<&str>,
    limit: usize,
) -> SqliteStoreResult<Vec<String>> {
    if limit == 0 {
        return Ok(Vec::new());
    }
    let now = unix_epoch_seconds();
    let mut statement = conn.prepare(
        "SELECT message_id FROM projected_lane_messages \
         WHERE namespace=?1 AND (?2 IS NULL OR inbox_id=?2) AND status='claimed' \
         AND (lease_until IS NULL OR lease_until<=?3) \
         ORDER BY seq ASC, created_at ASC LIMIT ?4",
    )?;
    let ids = statement
        .query_map(
            params![
                namespace,
                inbox_id,
                now,
                i64::try_from(limit).unwrap_or(i64::MAX)
            ],
            |row| row.get::<_, String>(0),
        )?
        .collect::<Result<Vec<_>, _>>()?;
    drop(statement);
    for id in &ids {
        conn.execute(
            "UPDATE projected_lane_messages SET status='pending', claimed_by=NULL, \
             lease_until=NULL WHERE message_id=?1 AND status='claimed' \
             AND (lease_until IS NULL OR lease_until<=?2)",
            params![id, now],
        )?;
    }
    Ok(ids)
}
fn list_projected_lane_messages(
    conn: &Connection,
    f: &LaneMessageFilter,
) -> SqliteStoreResult<Vec<ProjectedLaneMessage>> {
    let mut rows = Vec::new();
    let mut stmt = conn.prepare(&format!(
        "SELECT {LANE_SELECT} FROM projected_lane_messages"
    ))?;
    for item in stmt.query_map([], lane_from_row)? {
        let row = item?;
        let eq = |x: &Option<String>, v: &str| x.as_ref().is_none_or(|x| x == v);
        if eq(&f.namespace, &row.namespace)
            && eq(&f.purpose, &row.purpose)
            && eq(&f.inbox_id, &row.inbox_id)
            && eq(&f.conversation_id, &row.conversation_id)
            && eq(&f.status, &row.status)
            && eq(&f.msg_type, &row.msg_type)
            && eq(&f.sender_id, &row.sender_id)
            && eq(&f.recipient_id, &row.recipient_id)
            && f.correlation_id
                .as_ref()
                .is_none_or(|x| row.correlation_id.as_ref() == Some(x))
            && f.created_at_gte.is_none_or(|x| row.created_at >= x)
            && f.created_at_lte.is_none_or(|x| row.created_at <= x)
            && f.available_at_gte.is_none_or(|x| row.available_at >= x)
            && f.available_at_lte.is_none_or(|x| row.available_at <= x)
        {
            rows.push(row)
        }
    }
    rows.sort_by(|a, b| {
        (a.created_at, a.seq, &a.message_id).cmp(&(b.created_at, b.seq, &b.message_id))
    });
    if f.newest_first {
        rows.reverse()
    }
    rows.truncate(f.limit);
    Ok(rows)
}
fn truncate_error(error: &str) -> String {
    error.chars().take(2000).collect()
}

fn projection_payload_json(
    payload: &serde_json::Map<String, serde_json::Value>,
) -> SqliteStoreResult<String> {
    // `Map` preserves input order under this workspace feature. Rebuild through
    // BTreeMap to exactly match Python json.dumps(sort_keys=True, separators=...).
    let compact = serde_json::to_string(&BTreeMap::<_, _>::from_iter(payload.iter()))
        .map_err(|error| SqliteStoreError::TransactionAborted(error.to_string()))?;
    Ok(python_ascii_json(&compact))
}

fn python_ascii_json(compact_json: &str) -> String {
    compact_json
        .chars()
        .flat_map(|character| match character as u32 {
            0..=0x7f => character.to_string().chars().collect::<Vec<_>>(),
            value @ 0x80..=0xffff => format!("\\u{value:04x}").chars().collect(),
            value => {
                let offset = value - 0x1_0000;
                let high = 0xd800 + (offset >> 10);
                let low = 0xdc00 + (offset & 0x3ff);
                format!("\\u{high:04x}\\u{low:04x}").chars().collect()
            }
        })
        .collect()
}

fn projection_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<NamedProjection> {
    let payload_json: String = row.get(2)?;
    let payload = serde_json::from_str::<serde_json::Value>(&payload_json)
        .ok()
        .and_then(|value| value.as_object().cloned())
        .ok_or(rusqlite::Error::InvalidQuery)?;
    Ok(NamedProjection {
        namespace: row.get(0)?,
        key: row.get(1)?,
        payload,
        last_authoritative_seq: row.get(3)?,
        last_materialized_seq: row.get(4)?,
        projection_schema_version: row.get(5)?,
        materialization_status: row.get(6)?,
        updated_at_ms: row.get(7)?,
    })
}

fn named_projection(
    conn: &Connection,
    namespace: &str,
    key: &str,
) -> SqliteStoreResult<Option<NamedProjection>> {
    conn.query_row(
        "SELECT namespace, key, payload_json, last_authoritative_seq, last_materialized_seq, projection_schema_version, materialization_status, updated_at_ms FROM named_projections WHERE namespace = ?1 AND key = ?2",
        params![namespace, key],
        projection_from_row,
    )
    .optional()
    .map_err(Into::into)
}

fn named_projections(
    conn: &Connection,
    namespace: &str,
) -> SqliteStoreResult<Vec<NamedProjection>> {
    let mut statement = conn.prepare(
        "SELECT namespace, key, payload_json, last_authoritative_seq, last_materialized_seq, projection_schema_version, materialization_status, updated_at_ms FROM named_projections WHERE namespace = ?1 ORDER BY key ASC",
    )?;
    statement
        .query_map([namespace], projection_from_row)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(Into::into)
}

fn replace_named_projection(
    conn: &Connection,
    namespace: &str,
    key: &str,
    projection: NamedProjectionWrite,
) -> SqliteStoreResult<()> {
    let payload_json = projection_payload_json(&projection.payload)?;
    conn.execute(
        "INSERT INTO named_projections(namespace, key, payload_json, last_authoritative_seq, last_materialized_seq, projection_schema_version, materialization_status, updated_at_ms) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8) ON CONFLICT(namespace, key) DO UPDATE SET payload_json = excluded.payload_json, last_authoritative_seq = excluded.last_authoritative_seq, last_materialized_seq = excluded.last_materialized_seq, projection_schema_version = excluded.projection_schema_version, materialization_status = excluded.materialization_status, updated_at_ms = excluded.updated_at_ms",
        params![namespace, key, payload_json, projection.last_authoritative_seq, projection.last_materialized_seq, projection.projection_schema_version, projection.materialization_status, unix_epoch_millis()],
    )?;
    Ok(())
}

fn compare_and_swap_named_projection(
    conn: &Connection,
    namespace: &str,
    key: &str,
    expected_last_authoritative_seq: Option<i64>,
    expected_last_materialized_seq: Option<i64>,
    projection: NamedProjectionWrite,
) -> SqliteStoreResult<bool> {
    let payload_json = projection_payload_json(&projection.payload)?;
    let updated_at_ms = unix_epoch_millis();
    let changes = match (expected_last_authoritative_seq, expected_last_materialized_seq) {
        (None, None) => conn.execute(
            "INSERT INTO named_projections(namespace, key, payload_json, last_authoritative_seq, last_materialized_seq, projection_schema_version, materialization_status, updated_at_ms) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8) ON CONFLICT(namespace, key) DO NOTHING",
            params![namespace, key, payload_json, projection.last_authoritative_seq, projection.last_materialized_seq, projection.projection_schema_version, projection.materialization_status, updated_at_ms],
        )?,
        (Some(expected_authoritative), Some(expected_materialized)) => conn.execute(
            "UPDATE named_projections SET payload_json = ?1, last_authoritative_seq = ?2, last_materialized_seq = ?3, projection_schema_version = ?4, materialization_status = ?5, updated_at_ms = ?6 WHERE namespace = ?7 AND key = ?8 AND last_authoritative_seq = ?9 AND last_materialized_seq = ?10",
            params![payload_json, projection.last_authoritative_seq, projection.last_materialized_seq, projection.projection_schema_version, projection.materialization_status, updated_at_ms, namespace, key, expected_authoritative, expected_materialized],
        )?,
        _ => 0,
    };
    Ok(changes == 1)
}

fn clear_named_projection(conn: &Connection, namespace: &str, key: &str) -> SqliteStoreResult<()> {
    conn.execute(
        "DELETE FROM named_projections WHERE namespace = ?1 AND key = ?2",
        params![namespace, key],
    )?;
    Ok(())
}

fn clear_projection_namespace(conn: &Connection, namespace: &str) -> SqliteStoreResult<()> {
    conn.execute(
        "DELETE FROM named_projections WHERE namespace = ?1",
        [namespace],
    )?;
    Ok(())
}

fn current_global_seq(conn: &Connection) -> SqliteStoreResult<i64> {
    Ok(conn
        .query_row("SELECT value FROM global_seq WHERE rowid = 1", [], |row| {
            row.get(0)
        })
        .optional()?
        .unwrap_or(0))
}

fn current_user_seq(conn: &Connection, user_id: &str) -> SqliteStoreResult<i64> {
    Ok(conn
        .query_row(
            "SELECT value FROM user_seq WHERE user_id = ?1",
            [user_id],
            |row| row.get(0),
        )
        .optional()?
        .unwrap_or(0))
}

fn append_raw_entity_event(
    conn: &Connection,
    namespace: &str,
    event: NewRawEntityEvent,
) -> SqliteStoreResult<AppendedRawEvent> {
    if let Some(existing) = event_by_id(conn, &event.event_id)? {
        if existing.namespace != namespace {
            return Err(SqliteStoreError::EventIdNamespaceCollision {
                event_id: event.event_id,
                existing_namespace: existing.namespace,
                requested_namespace: namespace.to_owned(),
            });
        }
        return Ok(AppendedRawEvent {
            event: existing,
            inserted: false,
        });
    }
    let seq = conn.query_row(
        "INSERT INTO namespace_seq(namespace, next_seq) VALUES (?1, 2) \
         ON CONFLICT(namespace) DO UPDATE SET next_seq = namespace_seq.next_seq + 1 \
         RETURNING next_seq - 1",
        [namespace],
        |row| row.get(0),
    )?;
    let stored = RawEntityEvent {
        namespace: namespace.to_owned(),
        seq,
        event_id: event.event_id,
        entity_kind: event.entity_kind,
        entity_id: event.entity_id,
        op: event.op,
        payload_json: event.payload_json,
        created_at: unix_epoch_seconds(),
    };
    conn.execute(
        "INSERT INTO entity_events(namespace, seq, event_id, entity_kind, entity_id, op, payload_json, created_at) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        params![
            stored.namespace,
            stored.seq,
            stored.event_id,
            stored.entity_kind,
            stored.entity_id,
            stored.op,
            stored.payload_json,
            stored.created_at,
        ],
    )?;
    Ok(AppendedRawEvent {
        event: stored,
        inserted: true,
    })
}

fn event_by_id(conn: &Connection, event_id: &str) -> SqliteStoreResult<Option<RawEntityEvent>> {
    conn.query_row(
        "SELECT namespace, seq, event_id, entity_kind, entity_id, op, payload_json, created_at \
         FROM entity_events WHERE event_id = ?1",
        [event_id],
        raw_event_from_row,
    )
    .optional()
    .map_err(Into::into)
}

fn replay_raw_events(
    conn: &Connection,
    namespace: &str,
    after_seq: i64,
    limit: usize,
) -> SqliteStoreResult<Vec<RawEntityEvent>> {
    let sql_limit = i64::try_from(limit).unwrap_or(i64::MAX);
    let mut statement = conn.prepare(
        "SELECT namespace, seq, event_id, entity_kind, entity_id, op, payload_json, created_at \
         FROM entity_events WHERE namespace = ?1 AND seq > ?2 ORDER BY seq ASC LIMIT ?3",
    )?;
    let rows = statement.query_map(params![namespace, after_seq, sql_limit], raw_event_from_row)?;
    rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
}

fn recovery_projection_write(
    projection: &EntityProjection,
    latest_authoritative_seq: i64,
) -> NamedProjectionWrite {
    NamedProjectionWrite {
        payload: projection.payload(),
        last_authoritative_seq: latest_authoritative_seq,
        last_materialized_seq: projection.last_seq(),
        projection_schema_version: 1,
        materialization_status: "ready".to_owned(),
    }
}

fn recovery_report(
    projection: &EntityProjection,
    processed_count: usize,
    prior_cursor: i64,
    latest_authoritative_seq: i64,
) -> EntityRecoveryReport {
    EntityRecoveryReport {
        processed_count,
        prior_cursor,
        new_cursor: projection.last_seq(),
        latest_authoritative_seq,
        caught_up: projection.last_seq() >= latest_authoritative_seq,
        canonical_payload: projection.canonical_payload(),
        digest: projection.digest(),
    }
}

fn recover_entity_projection_uow(
    uow: &mut SqliteUnitOfWork<'_>,
    request: &EntityRecoveryRequest,
) -> SqliteStoreResult<EntityRecoveryReport> {
    validate_entity_recovery_request(request)?;
    let prior_cursor = uow
        .replay_cursor(&request.namespace, &request.consumer)?
        .last_seq;
    let current = named_projection(
        uow.transaction,
        &request.projection_namespace,
        &request.projection_key,
    )?;
    let mut projection = EntityProjection::from_payload(current.as_ref().map(|row| &row.payload))?;
    if projection.last_seq() != prior_cursor {
        return Err(StoreError::InvalidEntityEventPayload {
            message: "recovery projection cursor does not match durable replay cursor".to_owned(),
        }
        .into());
    }
    let events = replay_raw_events(
        uow.transaction,
        &request.namespace,
        prior_cursor,
        request.batch_limit,
    )?;
    let latest_authoritative_seq = latest_retained_event_seq(uow.transaction, &request.namespace)?;
    for raw in &events {
        projection.fold(&raw_to_entity_event(raw.clone())?)?;
    }
    let projection_changed = !events.is_empty() || current.is_none();
    if projection_changed {
        replace_named_projection(
            uow.transaction,
            &request.projection_namespace,
            &request.projection_key,
            recovery_projection_write(&projection, latest_authoritative_seq),
        )?;
    }
    if request.abort_after_projection {
        return Err(SqliteStoreError::TransactionAborted(
            "requested after entity recovery projection write".to_owned(),
        ));
    }
    if projection_changed || projection.last_seq() != prior_cursor {
        uow.strict_advance_replay_cursor(
            &request.namespace,
            &request.consumer,
            projection.last_seq(),
        )?;
    }
    Ok(recovery_report(
        &projection,
        events.len(),
        prior_cursor,
        latest_authoritative_seq,
    ))
}

fn rebuild_entity_projection_uow(
    uow: &mut SqliteUnitOfWork<'_>,
    request: &EntityRebuildRequest,
) -> SqliteStoreResult<EntityRecoveryReport> {
    validate_entity_rebuild_request(request)?;
    let prior_cursor = uow
        .replay_cursor(&request.namespace, &request.consumer)?
        .last_seq;
    let events = replay_raw_events(uow.transaction, &request.namespace, 0, usize::MAX)?;
    let latest_authoritative_seq = latest_retained_event_seq(uow.transaction, &request.namespace)?;
    let mut projection = EntityProjection::empty();
    for raw in &events {
        projection.fold(&raw_to_entity_event(raw.clone())?)?;
    }
    replace_named_projection(
        uow.transaction,
        &request.projection_namespace,
        &request.projection_key,
        recovery_projection_write(&projection, latest_authoritative_seq),
    )?;
    if request.abort_after_projection {
        return Err(SqliteStoreError::TransactionAborted(
            "requested after entity rebuild projection write".to_owned(),
        ));
    }
    // Full rebuild is explicitly allowed to reset a stale consumer cursor to
    // the rebuilt corpus boundary. It uses legacy set only after projection
    // write, in the same UoW, because strict monotonic advance rejects reset.
    uow.set_replay_cursor_legacy(&request.namespace, &request.consumer, projection.last_seq())?;
    Ok(recovery_report(
        &projection,
        events.len(),
        prior_cursor,
        latest_authoritative_seq,
    ))
}

fn latest_retained_event_seq(conn: &Connection, namespace: &str) -> SqliteStoreResult<i64> {
    conn.query_row(
        "SELECT COALESCE(MAX(seq), 0) FROM entity_events WHERE namespace = ?1",
        [namespace],
        |row| row.get(0),
    )
    .map_err(Into::into)
}

fn prune_entity_events_after(
    conn: &Connection,
    namespace: &str,
    to_seq: i64,
) -> SqliteStoreResult<u64> {
    let deleted = conn.execute(
        "DELETE FROM entity_events WHERE namespace = ?1 AND seq > ?2",
        params![namespace, to_seq],
    )?;
    u64::try_from(deleted).map_err(|error| SqliteStoreError::TransactionAborted(error.to_string()))
}

fn put_workflow_design_snapshot(
    conn: &Connection,
    workflow_id: &str,
    snapshot: WorkflowDesignSnapshotWrite,
) -> SqliteStoreResult<()> {
    conn.execute(
        "INSERT INTO workflow_design_snapshots(workflow_id, version, seq, payload_json, schema_version, created_at_ms) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6) \
         ON CONFLICT(workflow_id, version) DO UPDATE SET \
         seq = excluded.seq, payload_json = excluded.payload_json, \
         schema_version = excluded.schema_version, created_at_ms = excluded.created_at_ms",
        params![
            workflow_id,
            snapshot.version,
            snapshot.seq,
            snapshot.payload_json,
            snapshot.schema_version,
            unix_epoch_millis(),
        ],
    )?;
    Ok(())
}

fn workflow_design_snapshot(
    conn: &Connection,
    workflow_id: &str,
    max_version: i64,
    schema_version: i64,
) -> SqliteStoreResult<Option<WorkflowDesignSnapshot>> {
    conn.query_row(
        "SELECT workflow_id, version, seq, payload_json, schema_version, created_at_ms \
         FROM workflow_design_snapshots \
         WHERE workflow_id = ?1 AND version <= ?2 AND schema_version = ?3 \
         ORDER BY version DESC LIMIT 1",
        params![workflow_id, max_version, schema_version],
        workflow_design_snapshot_from_row,
    )
    .optional()
    .map_err(Into::into)
}

fn clear_workflow_design_snapshots(conn: &Connection, workflow_id: &str) -> SqliteStoreResult<()> {
    conn.execute(
        "DELETE FROM workflow_design_snapshots WHERE workflow_id = ?1",
        [workflow_id],
    )?;
    Ok(())
}

fn put_workflow_design_delta(
    conn: &Connection,
    workflow_id: &str,
    delta: WorkflowDesignDeltaWrite,
) -> SqliteStoreResult<()> {
    conn.execute(
        "INSERT INTO workflow_design_version_deltas(workflow_id, version, prev_version, target_seq, forward_json, inverse_json, schema_version, created_at_ms) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8) \
         ON CONFLICT(workflow_id, version) DO UPDATE SET \
         prev_version = excluded.prev_version, target_seq = excluded.target_seq, \
         forward_json = excluded.forward_json, inverse_json = excluded.inverse_json, \
         schema_version = excluded.schema_version, created_at_ms = excluded.created_at_ms",
        params![
            workflow_id,
            delta.version,
            delta.prev_version,
            delta.target_seq,
            delta.forward_json,
            delta.inverse_json,
            delta.schema_version,
            unix_epoch_millis(),
        ],
    )?;
    Ok(())
}

fn workflow_design_delta(
    conn: &Connection,
    workflow_id: &str,
    version: i64,
    schema_version: i64,
) -> SqliteStoreResult<Option<WorkflowDesignDelta>> {
    conn.query_row(
        "SELECT workflow_id, version, prev_version, target_seq, forward_json, inverse_json, schema_version, created_at_ms \
         FROM workflow_design_version_deltas \
         WHERE workflow_id = ?1 AND version = ?2 AND schema_version = ?3",
        params![workflow_id, version, schema_version],
        workflow_design_delta_from_row,
    )
    .optional()
    .map_err(Into::into)
}

fn clear_workflow_design_deltas(conn: &Connection, workflow_id: &str) -> SqliteStoreResult<()> {
    conn.execute(
        "DELETE FROM workflow_design_version_deltas WHERE workflow_id = ?1",
        [workflow_id],
    )?;
    Ok(())
}

fn create_server_run(conn: &Connection, run: ServerRunCreate) -> SqliteStoreResult<()> {
    let now = unix_epoch_millis();
    conn.execute(
        "INSERT INTO server_runs(run_id, conversation_id, workflow_id, user_id, user_turn_node_id, assistant_turn_node_id, status, cancel_requested, result_json, error_json, created_at_ms, updated_at_ms, started_at_ms, finished_at_ms) VALUES (?1, ?2, ?3, ?4, ?5, NULL, ?6, 0, NULL, NULL, ?7, ?7, NULL, NULL)",
        params![run.run_id, run.conversation_id, run.workflow_id, run.user_id, run.user_turn_node_id, run.status, now],
    )?;
    Ok(())
}

fn server_run(conn: &Connection, run_id: &str) -> SqliteStoreResult<Option<ServerRun>> {
    conn.query_row("SELECT run_id, conversation_id, workflow_id, user_id, user_turn_node_id, assistant_turn_node_id, status, cancel_requested, result_json, error_json, created_at_ms, updated_at_ms, started_at_ms, finished_at_ms FROM server_runs WHERE run_id = ?1", [run_id], server_run_from_row).optional().map_err(Into::into)
}

fn server_runs(
    conn: &Connection,
    status: Option<&str>,
    workflow_id: Option<&str>,
    conversation_id: Option<&str>,
    limit: usize,
) -> SqliteStoreResult<Vec<ServerRun>> {
    let mut sql = "SELECT run_id, conversation_id, workflow_id, user_id, user_turn_node_id, assistant_turn_node_id, status, cancel_requested, result_json, error_json, created_at_ms, updated_at_ms, started_at_ms, finished_at_ms FROM server_runs".to_owned();
    let mut predicates = Vec::new();
    if status.is_some() {
        predicates.push("status = ?");
    }
    if workflow_id.is_some() {
        predicates.push("workflow_id = ?");
    }
    if conversation_id.is_some() {
        predicates.push("conversation_id = ?");
    }
    if !predicates.is_empty() {
        sql.push_str(" WHERE ");
        sql.push_str(&predicates.join(" AND "));
    }
    sql.push_str(" ORDER BY created_at_ms DESC, run_id DESC LIMIT ?");
    let mut bindings: Vec<rusqlite::types::Value> = Vec::new();
    if let Some(value) = status {
        bindings.push(value.to_owned().into());
    }
    if let Some(value) = workflow_id {
        bindings.push(value.to_owned().into());
    }
    if let Some(value) = conversation_id {
        bindings.push(value.to_owned().into());
    }
    bindings.push((limit as i64).into());
    let mut statement = conn.prepare(&sql)?;
    statement
        .query_map(rusqlite::params_from_iter(bindings), server_run_from_row)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(Into::into)
}

fn append_server_run_event(
    conn: &Connection,
    run_id: &str,
    event_type: &str,
    payload_json: String,
) -> SqliteStoreResult<ServerRunEvent> {
    let created_at_ms = unix_epoch_millis();
    conn.execute("INSERT INTO server_run_events(run_id, event_type, payload_json, created_at_ms) VALUES (?1, ?2, ?3, ?4)", params![run_id, event_type, payload_json, created_at_ms])?;
    Ok(ServerRunEvent {
        seq: conn.last_insert_rowid(),
        run_id: run_id.to_owned(),
        event_type: event_type.to_owned(),
        payload_json,
        created_at_ms,
    })
}

fn server_run_events(
    conn: &Connection,
    run_id: &str,
    after_seq: i64,
    limit: usize,
) -> SqliteStoreResult<Vec<ServerRunEvent>> {
    let mut statement=conn.prepare("SELECT seq, run_id, event_type, payload_json, created_at_ms FROM server_run_events WHERE run_id = ?1 AND seq > ?2 ORDER BY seq ASC LIMIT ?3")?;
    statement
        .query_map(
            params![run_id, after_seq, limit as i64],
            server_run_event_from_row,
        )?
        .collect::<Result<Vec<_>, _>>()
        .map_err(Into::into)
}

fn update_server_run(
    conn: &Connection,
    run_id: &str,
    update: ServerRunUpdate,
) -> SqliteStoreResult<()> {
    conn.execute("UPDATE server_runs SET status=?1, assistant_turn_node_id=?2, result_json=?3, error_json=?4, started_at_ms=?5, finished_at_ms=?6, cancel_requested=COALESCE(?7,cancel_requested), updated_at_ms=?8 WHERE run_id=?9", params![update.status, update.assistant_turn_node_id, update.result_json, update.error_json, update.started_at_ms, update.finished_at_ms, update.cancel_requested.map(i64::from), unix_epoch_millis(), run_id])?;
    Ok(())
}

fn request_server_run_cancel(conn: &Connection, run_id: &str) -> SqliteStoreResult<()> {
    let Some(run) = server_run(conn, run_id)? else {
        return Ok(());
    };
    let mut current =
        read_recorded_runtime_state(conn, run_id, &run.workflow_id, &run.conversation_id)?;
    if current.is_none() {
        let payload_json = conn
            .query_row(
                "SELECT payload_json FROM projected_lane_messages \
                 WHERE run_id=?1 AND msg_type='workflow.run.execute' ORDER BY created_at LIMIT 1",
                [run_id],
                |row| row.get::<_, Option<String>>(0),
            )
            .optional()?
            .flatten();
        if let Some(payload_json) = payload_json {
            let payload: Value = serde_json::from_str(&payload_json)?;
            let expected_event_seq = latest_server_run_event_seq(conn, run_id)?;
            apply_recorded_runtime_transition_inner(
                conn,
                RecordedRuntimeTransition {
                    contract_version: RECORDED_RUNTIME_CONTRACT_VERSION,
                    transition_id: format!("start-{run_id}"),
                    expected_event_seq,
                    kind: kogwistar_runtime::RecordedTransitionKind::Start,
                    run_id: run_id.to_owned(),
                    workflow_id: run.workflow_id.clone(),
                    conversation_id: run.conversation_id.clone(),
                    user_id: payload["user_id"].as_str().map(str::to_owned),
                    user_turn_node_id: run
                        .user_turn_node_id
                        .clone()
                        .or_else(|| payload["turn_node_id"].as_str().map(str::to_owned)),
                    step_seq: 0,
                    node_id: Some("start".to_owned()),
                    token_id: Some(run_id.to_owned()),
                    parent_token_id: None,
                    initial_state: Some(
                        payload["initial_state"]
                            .as_object()
                            .cloned()
                            .unwrap_or_default(),
                    ),
                    state_update: Vec::new(),
                    update: None,
                    state_schema: Map::new(),
                    frontier: Some(RuntimeFrontier {
                        pending: vec![("start".to_owned(), 0, run_id.to_owned(), None)],
                        ..RuntimeFrontier::default()
                    }),
                    result: None,
                    wait_reason: None,
                    resume_payload: None,
                    errors: Vec::new(),
                },
                None,
                false,
            )?;
            current =
                read_recorded_runtime_state(conn, run_id, &run.workflow_id, &run.conversation_id)?;
        }
    }
    if let Some(state) = current
        && !state.status.is_terminal()
    {
        let expected_event_seq = latest_server_run_event_seq(conn, run_id)?;
        apply_recorded_runtime_transition_inner(
            conn,
            RecordedRuntimeTransition {
                contract_version: RECORDED_RUNTIME_CONTRACT_VERSION,
                transition_id: format!("cancel-{run_id}-{expected_event_seq}"),
                expected_event_seq,
                kind: kogwistar_runtime::RecordedTransitionKind::Cancel,
                run_id: run_id.to_owned(),
                workflow_id: run.workflow_id.clone(),
                conversation_id: run.conversation_id.clone(),
                user_id: None,
                user_turn_node_id: None,
                step_seq: state.last_step_seq.max(0),
                node_id: state.last_node_id,
                token_id: state.last_token_id,
                parent_token_id: state.last_parent_token_id,
                initial_state: None,
                state_update: Vec::new(),
                update: None,
                state_schema: Map::new(),
                frontier: None,
                result: None,
                wait_reason: None,
                resume_payload: None,
                errors: Vec::new(),
            },
            None,
            false,
        )?;
    } else {
        conn.execute("UPDATE server_runs SET cancel_requested=1, status=CASE WHEN status IN ('cancelled','failed','succeeded') THEN status ELSE 'cancelling' END, updated_at_ms=?1 WHERE run_id=?2", params![unix_epoch_millis(), run_id])?;
    }
    conn.execute(
        "UPDATE projected_lane_messages SET status='cancelled',claimed_by=NULL,lease_until=NULL \
         WHERE run_id=?1 AND status NOT IN ('completed','failed','cancelled','dead-letter')",
        [run_id],
    )?;
    Ok(())
}

const RECORDED_RUNTIME_EVENT_TYPE: &str = "workflow.recorded_transition.v1";
const RUNTIME_PROJECTION_SCHEMA_VERSION: i64 = 1;

fn runtime_checkpoint_namespace(conversation_id: &str) -> String {
    format!("{conversation_id}:workflow_checkpoint_latest")
}

fn runtime_status_namespace(conversation_id: &str) -> String {
    format!("{conversation_id}:workflow_run_status")
}

fn runtime_event_payload(persisted: &PersistedRecordedTransition) -> SqliteStoreResult<String> {
    serde_json::to_string(persisted).map_err(|error| {
        SqliteStoreError::TransactionAborted(format!(
            "cannot encode recorded runtime transition payload: {error}"
        ))
    })
}

fn parse_runtime_event_payload(
    payload_json: &str,
) -> SqliteStoreResult<Option<PersistedRecordedTransition>> {
    match serde_json::from_str::<PersistedRecordedTransition>(payload_json) {
        Ok(value) if value.contract_version == RECORDED_RUNTIME_CONTRACT_VERSION => Ok(Some(value)),
        Ok(_) => Ok(None),
        Err(_) => Ok(None),
    }
}

fn recorded_runtime_events(
    conn: &Connection,
    run_id: &str,
) -> SqliteStoreResult<Vec<ServerRunEvent>> {
    let mut statement = conn.prepare(
        "SELECT seq, run_id, event_type, payload_json, created_at_ms FROM server_run_events \
         WHERE run_id = ?1 AND event_type = ?2 ORDER BY seq ASC",
    )?;
    statement
        .query_map(
            params![run_id, RECORDED_RUNTIME_EVENT_TYPE],
            server_run_event_from_row,
        )?
        .collect::<Result<Vec<_>, _>>()
        .map_err(Into::into)
}

fn latest_server_run_event_seq(conn: &Connection, run_id: &str) -> SqliteStoreResult<i64> {
    conn.query_row(
        "SELECT COALESCE(MAX(seq), 0) FROM server_run_events WHERE run_id = ?1",
        [run_id],
        |row| row.get(0),
    )
    .map_err(Into::into)
}

fn recorded_runtime_projection(
    payload: serde_json::Map<String, serde_json::Value>,
    seq: i64,
    materialization_status: String,
) -> NamedProjectionWrite {
    NamedProjectionWrite {
        payload,
        last_authoritative_seq: seq,
        last_materialized_seq: seq,
        projection_schema_version: RUNTIME_PROJECTION_SCHEMA_VERSION,
        materialization_status,
    }
}

fn runtime_json_object<T: serde::Serialize>(value: &T) -> SqliteStoreResult<Map<String, Value>> {
    serde_json::to_value(value)?
        .as_object()
        .cloned()
        .ok_or_else(|| {
            SqliteStoreError::TransactionAborted(
                "runtime projection payload is not an object".to_owned(),
            )
        })
}

fn store_runtime_serving_projections(
    conn: &Connection,
    persisted: &PersistedRecordedTransition,
    event_seq: i64,
) -> SqliteStoreResult<()> {
    replace_named_projection(
        conn,
        RUNTIME_CURRENT_STATE_NAMESPACE,
        &persisted.reduced.state.run_id,
        recorded_runtime_projection(
            runtime_json_object(&persisted.reduced.state)?,
            event_seq,
            "ready".to_owned(),
        ),
    )
}

fn recorded_transition_by_id(
    conn: &Connection,
    transition_id: &str,
) -> SqliteStoreResult<Option<(PersistedRecordedTransition, i64)>> {
    conn.query_row(
        "SELECT payload_json, seq FROM server_run_events \
         WHERE event_type = ?1 AND json_extract(payload_json, '$.transition_id') = ?2 LIMIT 1",
        params![RECORDED_RUNTIME_EVENT_TYPE, transition_id],
        |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?)),
    )
    .optional()?
    .map(|(payload_json, seq)| {
        let payload = serde_json::from_str::<PersistedRecordedTransition>(&payload_json)?;
        Ok((payload, seq))
    })
    .transpose()
}

fn read_recorded_runtime_state(
    conn: &Connection,
    run_id: &str,
    workflow_id: &str,
    conversation_id: &str,
) -> SqliteStoreResult<Option<RecordedRuntimeState>> {
    let Some(run) = server_run(conn, run_id)? else {
        return Ok(None);
    };
    if run.workflow_id != workflow_id || run.conversation_id != conversation_id {
        return Err(SqliteStoreError::RecordedRuntimeConflict(format!(
            "run identity differs for {run_id:?}"
        )));
    }
    if let Some(row) = named_projection(conn, RUNTIME_CURRENT_STATE_NAMESPACE, run_id)?
        && row.materialization_status == "ready"
    {
        let has_newer_recorded_event: bool = conn.query_row(
            "SELECT EXISTS(
                SELECT 1 FROM server_run_events
                WHERE run_id = ?1 AND event_type = ?2 AND seq > ?3
             )",
            params![
                run_id,
                RECORDED_RUNTIME_EVENT_TYPE,
                row.last_authoritative_seq
            ],
            |result| result.get(0),
        )?;
        if !has_newer_recorded_event
            && let Ok(state) =
                serde_json::from_value::<RecordedRuntimeState>(Value::Object(row.payload))
            && state.run_id == run_id
            && state.workflow_id == workflow_id
            && state.conversation_id == conversation_id
        {
            return Ok(Some(state));
        }
    }
    let mut latest: Option<RecordedRuntimeState> = None;
    for event in recorded_runtime_events(conn, run_id)? {
        if let Some(payload) = parse_runtime_event_payload(&event.payload_json)?
            && payload.reduced.state.run_id == run_id
            && payload.reduced.state.workflow_id == workflow_id
            && payload.reduced.state.conversation_id == conversation_id
        {
            latest = Some(payload.reduced.state);
        }
    }
    Ok(latest)
}

fn apply_recorded_runtime_transition(
    conn: &Connection,
    transition: RecordedRuntimeTransition,
    abort_after_writes: bool,
) -> SqliteStoreResult<RecordedTransitionResult> {
    apply_recorded_runtime_transition_inner(conn, transition, None, abort_after_writes)
}

fn apply_claimed_recorded_runtime_transition(
    conn: &Connection,
    handoff: RecordedWorkerHandoff,
    transition: RecordedRuntimeTransition,
    abort_after_writes: bool,
) -> SqliteStoreResult<RecordedTransitionResult> {
    validate_worker_handoff(&handoff)?;
    let lane = projected_lane_message(conn, &handoff.message_id)?.ok_or_else(|| {
        SqliteStoreError::RecordedRuntimeConflict(format!(
            "worker lane message {:?} does not exist",
            handoff.message_id
        ))
    })?;
    validate_worker_lane_identity(&lane, &handoff, &transition)?;

    if lane.status == "completed" {
        return retry_completed_worker_handoff(conn, &handoff, &transition);
    }
    if lane.status != "claimed"
        || lane.claimed_by.as_deref() != Some(handoff.claimed_by.as_str())
        || lane
            .lease_until
            .as_ref()
            .and_then(|value| value.as_i64())
            .is_none_or(|lease_until| lease_until < unix_epoch_seconds())
    {
        return Err(SqliteStoreError::RecordedRuntimeConflict(format!(
            "worker lane message {:?} is not actively claimed by {:?}",
            handoff.message_id, handoff.claimed_by
        )));
    }

    let result =
        apply_recorded_runtime_transition_inner(conn, transition, Some(handoff.clone()), false)?;
    let acknowledged = conn.execute(
        "UPDATE projected_lane_messages SET status='completed',claimed_by=NULL,lease_until=NULL \
         WHERE message_id=?1 AND status='claimed' AND claimed_by=?2 AND lease_until>=?3",
        params![handoff.message_id, handoff.claimed_by, unix_epoch_seconds()],
    )?;
    if acknowledged != 1 {
        return Err(SqliteStoreError::RecordedRuntimeConflict(format!(
            "worker lane message {:?} claim changed before acknowledgement",
            handoff.message_id
        )));
    }
    if abort_after_writes {
        return Err(SqliteStoreError::TransactionAborted(
            "requested after recorded runtime worker handoff writes".to_owned(),
        ));
    }
    Ok(result)
}

fn apply_claimed_recorded_worker_effect(
    conn: &Connection,
    handoff: RecordedWorkerHandoff,
    effect: RecordedWorkerSuccessEffect,
) -> SqliteStoreResult<RecordedTransitionResult> {
    validate_worker_handoff(&handoff)?;
    if effect.contract_version != RECORDED_RUNTIME_CONTRACT_VERSION || effect.effect_id.is_empty() {
        return Err(SqliteStoreError::RecordedRuntimeConflict(
            "worker effect contract_version/effect_id is invalid".to_owned(),
        ));
    }
    let effect_digest = worker_effect_digest(&effect)?;
    let lane = projected_lane_message(conn, &handoff.message_id)?.ok_or_else(|| {
        SqliteStoreError::RecordedRuntimeConflict(format!(
            "worker lane message {:?} does not exist",
            handoff.message_id
        ))
    })?;
    if lane.run_id.as_deref() != Some(handoff.run_id.as_str())
        || lane.step_id.as_deref() != Some(handoff.step_id.as_str())
        || lane.correlation_id.as_deref() != Some(handoff.correlation_id.as_str())
    {
        return Err(SqliteStoreError::RecordedRuntimeConflict(format!(
            "worker lane message {:?} run/step/correlation identity differs",
            handoff.message_id
        )));
    }
    if server_run(conn, &handoff.run_id)?.is_some_and(|run| run.cancel_requested) {
        return Err(SqliteStoreError::RecordedRuntimeConflict(format!(
            "worker lane message {:?} belongs to a cancel-requested run",
            handoff.message_id
        )));
    }
    if lane.status == "completed" {
        if let Some((payload, event_seq)) = recorded_transition_by_id(conn, &effect.effect_id)?
            && (payload.worker_handoff.as_ref() == Some(&handoff)
                || payload
                    .worker_handoff
                    .as_ref()
                    .is_some_and(|stored| stored.message_id == handoff.message_id))
        {
            if payload.worker_effect_digest.as_deref() != Some(effect_digest.as_str()) {
                return Err(SqliteStoreError::RecordedRuntimeConflict(format!(
                    "worker handoff {:?} retried with different effect",
                    handoff.message_id
                )));
            }
            return Ok(payload.result(event_seq, true));
        }
        return Err(SqliteStoreError::RecordedRuntimeConflict(format!(
            "completed worker lane message {:?} has no matching recorded effect",
            handoff.message_id
        )));
    }
    if lane.status != "claimed"
        || lane.claimed_by.as_deref() != Some(handoff.claimed_by.as_str())
        || lane
            .lease_until
            .as_ref()
            .and_then(|value| value.as_i64())
            .is_none_or(|lease_until| lease_until < unix_epoch_seconds())
    {
        return Err(SqliteStoreError::RecordedRuntimeConflict(format!(
            "worker lane message {:?} is not actively claimed by {:?}",
            handoff.message_id, handoff.claimed_by
        )));
    }
    let payload: serde_json::Value =
        serde_json::from_str(lane.payload_json.as_deref().unwrap_or("{}"))?;
    let workflow_id = payload["workflow_id"].as_str().unwrap_or_default();
    let conversation_id = payload["conversation_id"].as_str().unwrap_or_default();
    let token_id = payload["token_id"]
        .as_str()
        .unwrap_or(handoff.run_id.as_str());
    let parent_token_id = payload["parent_token_id"].as_str();
    let current = read_recorded_runtime_state(conn, &handoff.run_id, workflow_id, conversation_id)?
        .ok_or_else(|| {
            SqliteStoreError::RecordedRuntimeConflict(format!(
                "worker lane message {:?} has no recorded runtime state",
                handoff.message_id
            ))
        })?;
    if current
        .frontier
        .pending
        .first()
        .is_none_or(|(node, _, token, parent)| {
            node != &handoff.step_id || token != token_id || parent.as_deref() != parent_token_id
        })
    {
        return Err(SqliteStoreError::RecordedRuntimeConflict(format!(
            "worker lane message {:?} result is not next in canonical frontier order",
            handoff.message_id
        )));
    }
    let expected_event_seq = latest_server_run_event_seq(conn, &handoff.run_id)?;
    let step_seq = current.last_step_seq.saturating_add(1);
    let authoritative_successors = kogwistar_runtime::authoritative_runtime_successors(
        &current.static_routes,
        &handoff.step_id,
        &effect.successors,
    )?;
    let next_frontier = match effect.status {
        RuntimeWorkerEffectStatus::Success => frontier_after_worker_success(
            &current.frontier,
            &handoff.step_id,
            token_id,
            parent_token_id,
            step_seq,
            &kogwistar_runtime::RuntimeWorkerSuccessEffect {
                successors: authoritative_successors,
            },
        )?,
        RuntimeWorkerEffectStatus::Suspended => frontier_after_worker_suspend(
            &current.frontier,
            &handoff.step_id,
            token_id,
            parent_token_id,
        )?,
    };
    let transition = RecordedRuntimeTransition {
        contract_version: RECORDED_RUNTIME_CONTRACT_VERSION,
        transition_id: effect.effect_id.clone(),
        expected_event_seq,
        kind: match effect.status {
            RuntimeWorkerEffectStatus::Success => {
                kogwistar_runtime::RecordedTransitionKind::RecordedStepSuccess
            }
            RuntimeWorkerEffectStatus::Suspended => {
                kogwistar_runtime::RecordedTransitionKind::Suspend
            }
        },
        run_id: handoff.run_id.clone(),
        workflow_id: workflow_id.to_owned(),
        conversation_id: conversation_id.to_owned(),
        user_id: None,
        user_turn_node_id: None,
        step_seq,
        node_id: Some(handoff.step_id.clone()),
        token_id: Some(token_id.to_owned()),
        parent_token_id: parent_token_id.map(str::to_owned),
        initial_state: None,
        state_update: effect.state_update,
        update: effect.update,
        state_schema: effect.state_schema,
        frontier: Some(next_frontier),
        result: effect.result,
        wait_reason: effect.wait_reason,
        resume_payload: effect.resume_payload,
        errors: effect.errors,
    };
    let reduced = reduce_recorded_transition(Some(&current), &transition)?;
    let request_digest = transition_digest(&transition)?;
    let persisted = PersistedRecordedTransition {
        contract_version: RECORDED_RUNTIME_CONTRACT_VERSION,
        transition_id: transition.transition_id.clone(),
        request_digest,
        worker_handoff: Some(handoff.clone()),
        worker_effect_digest: Some(effect_digest),
        reduced: reduced.clone(),
    };
    let event = append_server_run_event(
        conn,
        &transition.run_id,
        RECORDED_RUNTIME_EVENT_TYPE,
        runtime_event_payload(&persisted)?,
    )?;
    store_runtime_serving_projections(conn, &persisted, event.seq)?;
    replace_named_projection(
        conn,
        &runtime_checkpoint_namespace(&transition.conversation_id),
        &transition.run_id,
        recorded_runtime_projection(
            reduced.checkpoint.clone(),
            event.seq,
            reduced.state.status.server_status().to_owned(),
        ),
    )?;
    replace_named_projection(
        conn,
        &runtime_status_namespace(&transition.conversation_id),
        &transition.run_id,
        recorded_runtime_projection(reduced.run_status.clone(), event.seq, "running".to_owned()),
    )?;
    let prior_run = server_run(conn, &transition.run_id)?;
    update_server_run(
        conn,
        &transition.run_id,
        ServerRunUpdate {
            status: reduced.server_status.clone(),
            assistant_turn_node_id: prior_run
                .as_ref()
                .and_then(|run| run.assistant_turn_node_id.clone()),
            result_json: reduced
                .result
                .as_ref()
                .map(serde_json::to_string)
                .transpose()?,
            error_json: if reduced.errors.is_empty() {
                None
            } else {
                Some(serde_json::to_string(&reduced.errors)?)
            },
            started_at_ms: prior_run.as_ref().and_then(|run| run.started_at_ms),
            finished_at_ms: None,
            cancel_requested: Some(false),
        },
    )?;
    let acknowledged = conn.execute(
        "UPDATE projected_lane_messages SET status='completed',claimed_by=NULL,lease_until=NULL \
         WHERE message_id=?1 AND status='claimed' AND claimed_by=?2 AND lease_until>=?3",
        params![handoff.message_id, handoff.claimed_by, unix_epoch_seconds()],
    )?;
    if acknowledged != 1 {
        return Err(SqliteStoreError::RecordedRuntimeConflict(format!(
            "worker lane message {:?} claim changed before acknowledgement",
            handoff.message_id
        )));
    }
    Ok(persisted.result(event.seq, false))
}

fn validate_worker_handoff(handoff: &RecordedWorkerHandoff) -> SqliteStoreResult<()> {
    for (field, value) in [
        ("message_id", handoff.message_id.as_str()),
        ("claimed_by", handoff.claimed_by.as_str()),
        ("run_id", handoff.run_id.as_str()),
        ("step_id", handoff.step_id.as_str()),
        ("correlation_id", handoff.correlation_id.as_str()),
    ] {
        if value.is_empty() {
            return Err(SqliteStoreError::RecordedRuntimeConflict(format!(
                "worker handoff {field} must not be empty"
            )));
        }
    }
    Ok(())
}

fn validate_worker_lane_identity(
    lane: &ProjectedLaneMessage,
    handoff: &RecordedWorkerHandoff,
    transition: &RecordedRuntimeTransition,
) -> SqliteStoreResult<()> {
    if handoff.run_id != transition.run_id
        || transition.node_id.as_deref() != Some(handoff.step_id.as_str())
        || lane.run_id.as_deref() != Some(handoff.run_id.as_str())
        || lane.step_id.as_deref() != Some(handoff.step_id.as_str())
        || lane.correlation_id.as_deref() != Some(handoff.correlation_id.as_str())
    {
        return Err(SqliteStoreError::RecordedRuntimeConflict(format!(
            "worker lane message {:?} run/step/correlation identity differs",
            handoff.message_id
        )));
    }
    Ok(())
}

fn retry_completed_worker_handoff(
    conn: &Connection,
    handoff: &RecordedWorkerHandoff,
    transition: &RecordedRuntimeTransition,
) -> SqliteStoreResult<RecordedTransitionResult> {
    let request_digest = transition_digest(transition)?;
    if let Some((payload, event_seq)) = recorded_transition_by_id(conn, &transition.transition_id)?
    {
        if payload.request_digest != request_digest
            || payload.worker_handoff.as_ref() != Some(handoff)
        {
            return Err(SqliteStoreError::RecordedRuntimeConflict(format!(
                "worker handoff {:?} retried with different result",
                handoff.message_id
            )));
        }
        return Ok(payload.result(event_seq, true));
    }
    Err(SqliteStoreError::RecordedRuntimeConflict(format!(
        "completed worker lane message {:?} has no matching recorded result",
        handoff.message_id
    )))
}

fn apply_recorded_runtime_transition_inner(
    conn: &Connection,
    transition: RecordedRuntimeTransition,
    worker_handoff: Option<RecordedWorkerHandoff>,
    abort_after_writes: bool,
) -> SqliteStoreResult<RecordedTransitionResult> {
    let request_digest = transition_digest(&transition)?;

    // Idempotency lookup occurs before sequence/CAS checks. Exact retry returns
    // the original response; reusing an id for different immutable input fails.
    if let Some((payload, event_seq)) = recorded_transition_by_id(conn, &transition.transition_id)?
    {
        if payload.request_digest != request_digest || payload.worker_handoff != worker_handoff {
            return Err(SqliteStoreError::RecordedRuntimeConflict(format!(
                "transition_id {:?} reused with different payload",
                transition.transition_id
            )));
        }
        return Ok(payload.result(event_seq, true));
    }

    let existing_run = server_run(conn, &transition.run_id)?;
    if let Some(run) = &existing_run
        && (run.workflow_id != transition.workflow_id
            || run.conversation_id != transition.conversation_id)
    {
        return Err(SqliteStoreError::RecordedRuntimeConflict(format!(
            "run identity differs for {:?}",
            transition.run_id
        )));
    }
    let current = read_recorded_runtime_state(
        conn,
        &transition.run_id,
        &transition.workflow_id,
        &transition.conversation_id,
    )?;
    let current_event_seq = latest_server_run_event_seq(conn, &transition.run_id)?;
    if current_event_seq != transition.expected_event_seq {
        return Err(SqliteStoreError::RecordedRuntimeConflict(format!(
            "expected_event_seq {}, current {}",
            transition.expected_event_seq, current_event_seq
        )));
    }
    let reduced = reduce_recorded_transition(current.as_ref(), &transition)?;

    if current.is_none() && existing_run.is_none() {
        let user_turn_node_id = transition.user_turn_node_id.clone().ok_or_else(|| {
            SqliteStoreError::RecordedRuntimeConflict(
                "start transition requires user_turn_node_id".to_owned(),
            )
        })?;
        create_server_run(
            conn,
            ServerRunCreate {
                run_id: transition.run_id.clone(),
                conversation_id: transition.conversation_id.clone(),
                workflow_id: transition.workflow_id.clone(),
                user_id: transition.user_id.clone(),
                user_turn_node_id,
                status: reduced.server_status.clone(),
            },
        )?;
    }

    let persisted = PersistedRecordedTransition {
        contract_version: RECORDED_RUNTIME_CONTRACT_VERSION,
        transition_id: transition.transition_id.clone(),
        request_digest,
        worker_handoff,
        worker_effect_digest: None,
        reduced: reduced.clone(),
    };
    let event = append_server_run_event(
        conn,
        &transition.run_id,
        RECORDED_RUNTIME_EVENT_TYPE,
        runtime_event_payload(&persisted)?,
    )?;
    let event_seq = event.seq;
    store_runtime_serving_projections(conn, &persisted, event_seq)?;
    replace_named_projection(
        conn,
        &runtime_checkpoint_namespace(&transition.conversation_id),
        &transition.run_id,
        recorded_runtime_projection(
            reduced.checkpoint.clone(),
            event_seq,
            reduced.state.status.server_status().to_owned(),
        ),
    )?;
    let status_materialization = match reduced.state.status {
        kogwistar_runtime::RecordedRunStatus::Completed => "completed",
        kogwistar_runtime::RecordedRunStatus::Failed => "failed",
        kogwistar_runtime::RecordedRunStatus::Cancelled => "cancelled",
        kogwistar_runtime::RecordedRunStatus::Suspended => "suspended",
        kogwistar_runtime::RecordedRunStatus::Running => "running",
    }
    .to_owned();
    replace_named_projection(
        conn,
        &runtime_status_namespace(&transition.conversation_id),
        &transition.run_id,
        recorded_runtime_projection(
            reduced.run_status.clone(),
            event_seq,
            status_materialization,
        ),
    )?;
    let prior_run = existing_run.as_ref();
    update_server_run(
        conn,
        &transition.run_id,
        ServerRunUpdate {
            status: reduced.server_status.clone(),
            assistant_turn_node_id: prior_run.and_then(|run| run.assistant_turn_node_id.clone()),
            result_json: reduced
                .result
                .as_ref()
                .map(serde_json::to_string)
                .transpose()?
                .or_else(|| prior_run.and_then(|run| run.result_json.clone())),
            error_json: if reduced.errors.is_empty() {
                prior_run.and_then(|run| run.error_json.clone())
            } else {
                Some(serde_json::to_string(&reduced.errors)?)
            },
            started_at_ms: prior_run.and_then(|run| run.started_at_ms).or_else(|| {
                (transition.kind == kogwistar_runtime::RecordedTransitionKind::Start)
                    .then(unix_epoch_millis)
            }),
            finished_at_ms: if reduced.state.status.is_terminal() {
                Some(unix_epoch_millis())
            } else {
                prior_run.and_then(|run| run.finished_at_ms)
            },
            cancel_requested: Some(matches!(
                reduced.state.status,
                kogwistar_runtime::RecordedRunStatus::Cancelled
            )),
        },
    )?;
    if abort_after_writes {
        return Err(SqliteStoreError::TransactionAborted(
            "requested after recorded runtime transition writes".to_owned(),
        ));
    }
    Ok(persisted.result(event_seq, false))
}

fn replay_cursor(
    conn: &Connection,
    namespace: &str,
    consumer: &str,
) -> SqliteStoreResult<ReplayCursor> {
    let last_seq = conn
        .query_row(
            "SELECT last_seq FROM replay_cursors WHERE namespace = ?1 AND consumer = ?2",
            params![namespace, consumer],
            |row| row.get(0),
        )
        .optional()?
        .unwrap_or(0);
    Ok(ReplayCursor {
        namespace: namespace.to_owned(),
        consumer: consumer.to_owned(),
        last_seq,
    })
}

fn strict_advance_replay_cursor(
    conn: &Connection,
    namespace: &str,
    consumer: &str,
    last_seq: i64,
) -> SqliteStoreResult<ReplayCursor> {
    let latest = latest_retained_event_seq(conn, namespace)?;
    if last_seq > latest {
        return Err(StoreError::CursorOutOfRange {
            cursor: last_seq,
            latest,
        }
        .into());
    }
    let current = replay_cursor(conn, namespace, consumer)?.last_seq;
    if last_seq < current {
        return Err(StoreError::CursorRegresses {
            current,
            requested: last_seq,
        }
        .into());
    }
    set_replay_cursor_legacy(conn, namespace, consumer, last_seq)
}

fn set_replay_cursor_legacy(
    conn: &Connection,
    namespace: &str,
    consumer: &str,
    last_seq: i64,
) -> SqliteStoreResult<ReplayCursor> {
    conn.execute(
        "INSERT INTO replay_cursors(namespace, consumer, last_seq, updated_at) VALUES (?1, ?2, ?3, ?4) \
         ON CONFLICT(namespace, consumer) DO UPDATE SET last_seq = excluded.last_seq, updated_at = excluded.updated_at",
        params![namespace, consumer, last_seq, unix_epoch_seconds()],
    )?;
    Ok(ReplayCursor {
        namespace: namespace.to_owned(),
        consumer: consumer.to_owned(),
        last_seq,
    })
}

fn raw_event_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<RawEntityEvent> {
    Ok(RawEntityEvent {
        namespace: row.get(0)?,
        seq: row.get(1)?,
        event_id: row.get(2)?,
        entity_kind: row.get(3)?,
        entity_id: row.get(4)?,
        op: row.get(5)?,
        payload_json: row.get(6)?,
        created_at: row.get(7)?,
    })
}

fn workflow_design_snapshot_from_row(
    row: &rusqlite::Row<'_>,
) -> rusqlite::Result<WorkflowDesignSnapshot> {
    Ok(WorkflowDesignSnapshot {
        workflow_id: row.get(0)?,
        version: row.get(1)?,
        seq: row.get(2)?,
        payload_json: row.get(3)?,
        schema_version: row.get(4)?,
        created_at_ms: row.get(5)?,
    })
}

fn workflow_design_delta_from_row(
    row: &rusqlite::Row<'_>,
) -> rusqlite::Result<WorkflowDesignDelta> {
    Ok(WorkflowDesignDelta {
        workflow_id: row.get(0)?,
        version: row.get(1)?,
        prev_version: row.get(2)?,
        target_seq: row.get(3)?,
        forward_json: row.get(4)?,
        inverse_json: row.get(5)?,
        schema_version: row.get(6)?,
        created_at_ms: row.get(7)?,
    })
}

fn server_run_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<ServerRun> {
    Ok(ServerRun {
        run_id: row.get(0)?,
        conversation_id: row.get(1)?,
        workflow_id: row.get(2)?,
        user_id: row.get(3)?,
        user_turn_node_id: row.get(4)?,
        assistant_turn_node_id: row.get(5)?,
        status: row.get(6)?,
        cancel_requested: row.get::<_, i64>(7)? != 0,
        result_json: row.get(8)?,
        error_json: row.get(9)?,
        created_at_ms: row.get(10)?,
        updated_at_ms: row.get(11)?,
        started_at_ms: row.get(12)?,
        finished_at_ms: row.get(13)?,
    })
}
fn server_run_event_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<ServerRunEvent> {
    Ok(ServerRunEvent {
        seq: row.get(0)?,
        run_id: row.get(1)?,
        event_type: row.get(2)?,
        payload_json: row.get(3)?,
        created_at_ms: row.get(4)?,
    })
}

fn raw_to_entity_event(raw: RawEntityEvent) -> SqliteStoreResult<EntityEvent> {
    Ok(EntityEventEnvelope {
        namespace: raw.namespace,
        seq: raw.seq,
        event_id: raw.event_id,
        entity_kind: raw.entity_kind,
        entity_id: raw.entity_id,
        op: raw.op,
        payload: serde_json::from_str(&raw.payload_json)?,
    })
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
        // `kogwistar-store` deliberately has no persistence-failure variant.
        // Direct APIs expose precise SQLite errors; the legacy trait error
        // surface is closed, so retain an existing identity-conflict variant
        // rather than panic across an async store boundary.
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::future::Future;
    use std::pin::pin;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};
    use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
    use std::thread;

    static NEXT_DB: AtomicUsize = AtomicUsize::new(0);

    fn database_path(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "kogwistar-store-sqlite-{label}-{}-{}.db",
            std::process::id(),
            NEXT_DB.fetch_add(1, Ordering::Relaxed),
        ))
    }

    fn store(label: &str) -> (SqliteStore, PathBuf) {
        let path = database_path(label);
        (SqliteStore::open(&path).unwrap(), path)
    }

    fn event(event_id: &str, entity_id: &str) -> NewEntityEvent {
        NewEntityEvent {
            event_id: event_id.to_owned(),
            entity_kind: "node".to_owned(),
            entity_id: entity_id.to_owned(),
            op: "UPSERT".to_owned(),
            payload: serde_json::json!({"id": entity_id}),
        }
    }

    fn raw_event(event_id: &str, payload_json: &str) -> NewRawEntityEvent {
        NewRawEntityEvent {
            event_id: event_id.to_owned(),
            entity_kind: "node".to_owned(),
            entity_id: "node-1".to_owned(),
            op: "UPSERT".to_owned(),
            payload_json: payload_json.to_owned(),
        }
    }

    fn runtime_start(run_id: &str, transition_id: &str) -> RecordedRuntimeTransition {
        serde_json::from_value(serde_json::json!({
            "contract_version": 1,
            "transition_id": transition_id,
            "expected_event_seq": 0,
            "kind": "start",
            "run_id": run_id,
            "workflow_id": "wf-1",
            "conversation_id": "conv-1",
            "user_id": "user-1",
            "user_turn_node_id": "turn-1",
            "step_seq": 0,
            "node_id": "step-1",
            "token_id": "token-1",
            "initial_state": {"answer": "seed"},
            "frontier": {
                "pending": [["step-1", 0, "token-1", null]],
                "suspended": [], "join_node_ids": [], "join_outstanding": [],
                "join_waiters": {}
            }
        }))
        .unwrap()
    }

    fn runtime_result(
        run_id: &str,
        transition_id: &str,
        expected_event_seq: i64,
        answer: &str,
    ) -> RecordedRuntimeTransition {
        serde_json::from_value(serde_json::json!({
            "contract_version": 1,
            "transition_id": transition_id,
            "expected_event_seq": expected_event_seq,
            "kind": "recorded_step_success",
            "run_id": run_id,
            "workflow_id": "wf-1",
            "conversation_id": "conv-1",
            "step_seq": 0,
            "node_id": "step-1",
            "token_id": "token-1",
            "state_update": [["u", {"answer": answer}]],
            "frontier": {
                "pending": [], "suspended": [], "join_node_ids": [],
                "join_outstanding": [], "join_waiters": {}
            },
            "result": {"answer": answer}
        }))
        .unwrap()
    }

    #[test]
    fn auth_store_opens_python_schema_and_resolves_identity_atomically() {
        let path = database_path("auth");
        let store = SqliteAuthStore::open(&path).unwrap();
        let request = ResolveExternalIdentity {
            issuer: "https://issuer.example".to_owned(),
            subject: "subject-1".to_owned(),
            email: "alice@example.com".to_owned(),
            display_name: Some("Alice".to_owned()),
            new_user_id: "generated-1".to_owned(),
            default_role: "ro".to_owned(),
            default_ns: "docs".to_owned(),
        };
        let created = block_on(store.resolve_external_identity(request.clone())).unwrap();
        assert_eq!(created.user_id, "generated-1");
        assert_eq!(created.email, "alice@example.com");
        assert_eq!(created.display_name.as_deref(), Some("Alice"));
        assert_eq!(created.global_role.as_deref(), Some("ro"));
        let retried = block_on(store.resolve_external_identity(ResolveExternalIdentity {
            new_user_id: "must-not-win".to_owned(),
            email: "changed@example.com".to_owned(),
            ..request
        }))
        .unwrap();
        assert_eq!(retried.user_id, "generated-1");
        let identity = block_on(store.external_identity("https://issuer.example", "subject-1"))
            .unwrap()
            .unwrap();
        assert_eq!(identity.user_id, "generated-1");
        assert_eq!(identity.email.as_deref(), Some("alice@example.com"));
        let connection = Connection::open(&path).unwrap();
        let table_names = connection
            .prepare("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")
            .unwrap()
            .query_map([], |row| row.get::<_, String>(0))
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(
            table_names,
            ["external_identities", "users", "workflow_acl"]
        );
        drop(connection);
        drop(store);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn auth_store_links_existing_python_user_by_email() {
        let path = database_path("auth-python-user");
        let store = SqliteAuthStore::open(&path).unwrap();
        let connection = Connection::open(&path).unwrap();
        connection
            .execute(
                "INSERT INTO users (user_id, email, display_name, is_active, global_role, global_ns, created_at) VALUES ('python-user', 'alice@example.com', 'Python Alice', 1, 'rw', 'docs,workflow', CURRENT_TIMESTAMP)",
                [],
            )
            .unwrap();
        drop(connection);
        let user = block_on(store.resolve_external_identity(ResolveExternalIdentity {
            issuer: "issuer".to_owned(),
            subject: "subject".to_owned(),
            email: "alice@example.com".to_owned(),
            display_name: Some("Ignored".to_owned()),
            new_user_id: "rust-user".to_owned(),
            default_role: "ro".to_owned(),
            default_ns: "docs".to_owned(),
        }))
        .unwrap();
        assert_eq!(user.user_id, "python-user");
        assert_eq!(user.display_name.as_deref(), Some("Python Alice"));
        assert_eq!(user.global_role.as_deref(), Some("rw"));
        assert_eq!(
            block_on(store.external_identity("issuer", "subject"))
                .unwrap()
                .unwrap()
                .user_id,
            "python-user"
        );
        drop(store);
        fs::remove_file(path).unwrap();
    }

    fn block_on<T>(future: impl Future<Output = T>) -> T {
        fn no_op(_: *const ()) {}
        fn clone(_: *const ()) -> RawWaker {
            RawWaker::new(std::ptr::null(), &VTABLE)
        }
        static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, no_op, no_op, no_op);
        let raw = RawWaker::new(std::ptr::null(), &VTABLE);
        let waker = unsafe { Waker::from_raw(raw) };
        let mut context = Context::from_waker(&waker);
        let mut future = pin!(future);
        loop {
            match future.as_mut().poll(&mut context) {
                Poll::Ready(value) => return value,
                Poll::Pending => thread::yield_now(),
            }
        }
    }

    #[test]
    fn python_legacy_shaped_database_is_opened_read_and_written_without_reformatting_json() {
        let path = database_path("legacy");
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch(
            "
            CREATE TABLE global_seq (value INTEGER NOT NULL);
            INSERT INTO global_seq(rowid, value) VALUES (1, 4);
            CREATE TABLE user_seq (user_id TEXT PRIMARY KEY, value INTEGER NOT NULL);
            INSERT INTO user_seq VALUES ('alice', 8);
            CREATE TABLE namespace_seq (namespace TEXT PRIMARY KEY, next_seq INTEGER NOT NULL);
            INSERT INTO namespace_seq VALUES ('legacy', 7);
            CREATE TABLE entity_events (
                namespace TEXT NOT NULL DEFAULT 'default', seq INTEGER NOT NULL, event_id TEXT NOT NULL,
                entity_kind TEXT NOT NULL, entity_id TEXT NOT NULL, op TEXT NOT NULL,
                payload_json TEXT NOT NULL, created_at INTEGER NOT NULL,
                PRIMARY KEY(namespace, seq), UNIQUE(event_id)
            );
            INSERT INTO entity_events VALUES ('legacy', 6, 'old-id', 'node', 'old', 'UPSERT', '{\"spaced\": [ 1, 2 ]}', 123);
            CREATE TABLE replay_cursors (
                namespace TEXT NOT NULL DEFAULT 'default', consumer TEXT NOT NULL,
                last_seq INTEGER NOT NULL, updated_at INTEGER NOT NULL,
                PRIMARY KEY(namespace, consumer)
            );
            INSERT INTO replay_cursors VALUES ('legacy', 'reader', 6, 124);
            ",
        )
        .unwrap();
        drop(conn);

        let store = SqliteStore::open(&path).unwrap();
        assert_eq!(store.current_global_seq().unwrap(), 4);
        assert_eq!(store.current_user_seq("alice").unwrap(), 8);
        assert_eq!(
            store.replay_raw_events("legacy", 0, 10).unwrap()[0].payload_json,
            "{\"spaced\": [ 1, 2 ]}"
        );
        assert_eq!(store.replay_cursor("legacy", "reader").unwrap().last_seq, 6);
        assert_eq!(
            store
                .append_raw_entity_event("legacy", raw_event("new-id", "{\"original\": true }"))
                .unwrap()
                .event
                .seq,
            7
        );
        drop(store);

        let conn = Connection::open(&path).unwrap();
        let row: (i64, String) = conn
            .query_row(
                "SELECT seq, payload_json FROM entity_events WHERE namespace = 'legacy' AND event_id = 'new-id'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(row, (7, "{\"original\": true }".to_owned()));
        drop(conn);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn python_legacy_queue_schema_gets_all_additive_columns_before_indexes() {
        let path = database_path("legacy-queue-migration");
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch(
            "CREATE TABLE index_jobs (
                job_id TEXT PRIMARY KEY, entity_kind TEXT NOT NULL, entity_id TEXT NOT NULL,
                index_kind TEXT NOT NULL, op TEXT NOT NULL, status TEXT NOT NULL,
                lease_until INTEGER, retry_count INTEGER NOT NULL DEFAULT 0,
                last_error TEXT, payload_json TEXT, created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
             );
             CREATE TABLE projected_lane_messages (
                message_id TEXT PRIMARY KEY, namespace TEXT NOT NULL DEFAULT 'default',
                inbox_id TEXT NOT NULL, conversation_id TEXT NOT NULL, recipient_id TEXT NOT NULL,
                sender_id TEXT NOT NULL, msg_type TEXT NOT NULL, status TEXT NOT NULL,
                seq INTEGER NOT NULL, conversation_seq INTEGER NOT NULL, claimed_by TEXT,
                lease_until INTEGER, retry_count INTEGER NOT NULL DEFAULT 0, created_at INTEGER NOT NULL,
                available_at INTEGER NOT NULL, run_id TEXT, step_id TEXT, correlation_id TEXT,
                payload_json TEXT, error_json TEXT, prev_message_id TEXT, next_message_id TEXT,
                inbox_tail_message_id TEXT, conversation_tail_message_id TEXT
             );",
        )
        .unwrap();
        drop(conn);

        let store = SqliteStore::open(&path).unwrap();
        let conn = Connection::open(&path).unwrap();
        let index_job_columns = conn
            .prepare("PRAGMA table_info(index_jobs)")
            .unwrap()
            .query_map([], |row| row.get::<_, String>(1))
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        for expected in [
            "namespace",
            "coalesce_key",
            "next_run_at",
            "max_retries",
            "claim_token",
        ] {
            assert!(index_job_columns.iter().any(|column| column == expected));
        }
        let purpose_exists = conn
            .prepare("PRAGMA table_info(projected_lane_messages)")
            .unwrap()
            .query_map([], |row| row.get::<_, String>(1))
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
            .iter()
            .any(|column| column == "purpose");
        assert!(purpose_exists);
        let coalesce_index_exists: bool = conn
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='index' AND name='uq_index_jobs_pending_ns_coalesce')",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(coalesce_index_exists);
        drop(conn);
        drop(store);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rust_created_database_has_python_legacy_schema_and_sql_semantics() {
        let (store, path) = store("schema");
        store.next_global_seq().unwrap();
        store.next_user_seq("alice").unwrap();
        store
            .append_raw_entity_event("default", raw_event("event-1", "{\"x\":1}"))
            .unwrap();
        drop(store);

        let conn = Connection::open(&path).unwrap();
        let tables: Vec<String> = conn
            .prepare("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(
            tables,
            vec![
                "entity_events",
                "global_seq",
                "index_applied_state",
                "index_jobs",
                "named_projections",
                "namespace_seq",
                "projected_lane_messages",
                "replay_cursors",
                "server_run_events",
                "server_runs",
                "sqlite_sequence",
                "user_seq",
                "workflow_design_snapshots",
                "workflow_design_version_deltas",
            ]
        );
        let global: i64 = conn
            .query_row("SELECT value FROM global_seq WHERE rowid = 1", [], |row| {
                row.get(0)
            })
            .unwrap();
        let user: i64 = conn
            .query_row(
                "SELECT value FROM user_seq WHERE user_id = 'alice'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        let event: (String, i64, String) = conn
            .query_row(
                "SELECT namespace, seq, payload_json FROM entity_events WHERE event_id = 'event-1'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(
            (global, user, event),
            (1, 1, ("default".to_owned(), 1, "{\"x\":1}".to_owned()))
        );
        drop(conn);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn recorded_transition_lookup_uses_json_expression_index() {
        let (store, path) = store("runtime-transition-index");
        drop(store);
        let conn = Connection::open(&path).unwrap();
        let plan = conn
            .prepare(
                "EXPLAIN QUERY PLAN SELECT payload_json, seq FROM server_run_events \
                 WHERE event_type = ?1 AND json_extract(payload_json, '$.transition_id') = ?2 LIMIT 1",
            )
            .unwrap()
            .query_map(params![RECORDED_RUNTIME_EVENT_TYPE, "transition-1"], |row| {
                row.get::<_, String>(3)
            })
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
            .join(" ");
        assert!(
            plan.contains("idx_server_run_events_recorded_transition_id"),
            "{plan}"
        );
        drop(conn);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn runtime_current_state_projection_is_disposable_fresh_and_atomic() {
        let (store, path) = store("runtime-current-state");
        let start_transition = runtime_start("run-1", "start-1");
        let start = store
            .apply_recorded_runtime_transition(start_transition.clone(), false)
            .unwrap();
        let exact_retry = store
            .apply_recorded_runtime_transition(start_transition.clone(), false)
            .unwrap();
        assert!(exact_retry.idempotent);
        assert_eq!(start.event_seq, exact_retry.event_seq);
        let mut conflicting_start = start_transition;
        conflicting_start.initial_state = Some(Map::from_iter([(
            "answer".to_owned(),
            Value::String("changed".to_owned()),
        )]));
        assert!(matches!(
            store.apply_recorded_runtime_transition(conflicting_start, false),
            Err(SqliteStoreError::RecordedRuntimeConflict(_))
        ));

        let start_projection = store
            .get_named_projection(RUNTIME_CURRENT_STATE_NAMESPACE, "run-1")
            .unwrap()
            .unwrap();
        assert_eq!(start_projection.last_authoritative_seq, start.event_seq);
        assert_eq!(start_projection.payload["state"]["answer"], "seed");

        let result_transition = runtime_result("run-1", "result-1", start.event_seq, "worker");
        assert!(matches!(
            store.apply_recorded_runtime_transition(result_transition.clone(), true),
            Err(SqliteStoreError::TransactionAborted(_))
        ));
        assert_eq!(
            store.list_server_run_events("run-1", 0, 10).unwrap().len(),
            1
        );
        assert_eq!(
            store
                .get_named_projection(RUNTIME_CURRENT_STATE_NAMESPACE, "run-1")
                .unwrap()
                .unwrap()
                .last_authoritative_seq,
            start.event_seq
        );

        let result = store
            .apply_recorded_runtime_transition(result_transition, false)
            .unwrap();
        assert_eq!(
            store
                .read_recorded_runtime_state("run-1", "wf-1", "conv-1")
                .unwrap()
                .unwrap()
                .state["answer"],
            "worker"
        );

        let conn = Connection::open(&path).unwrap();
        conn.execute(
            "UPDATE named_projections SET payload_json='{}', last_authoritative_seq=?1 \
             WHERE namespace=?2 AND key='run-1'",
            params![start.event_seq, RUNTIME_CURRENT_STATE_NAMESPACE],
        )
        .unwrap();
        drop(conn);
        let replayed = store
            .read_recorded_runtime_state("run-1", "wf-1", "conv-1")
            .unwrap()
            .unwrap();
        assert_eq!(replayed.state["answer"], "worker");
        assert_eq!(result.reduced.state, replayed);

        let conn = Connection::open(&path).unwrap();
        conn.execute(
            "DELETE FROM named_projections WHERE namespace=?1 AND key='run-1'",
            [RUNTIME_CURRENT_STATE_NAMESPACE],
        )
        .unwrap();
        drop(conn);
        assert_eq!(
            store
                .read_recorded_runtime_state("run-1", "wf-1", "conv-1")
                .unwrap()
                .unwrap(),
            result.reduced.state
        );
        drop(store);
        fs::remove_file(path).unwrap();
    }

    /// Manual code-smell probe. Absolute timings vary by host; growth ratio is
    /// the useful signal. Both indexed retry lookup and warm current-state read
    /// should remain near-flat as unrelated recorded history grows.
    #[test]
    #[ignore = "manual runtime serving-path scale probe"]
    fn runtime_serving_lookup_scale_probe() {
        use std::time::Instant;

        fn measure(history_size: usize) -> (u128, u128) {
            let (store, path) = store(&format!("runtime-scale-{history_size}"));
            let start_transition = runtime_start("target-run", "target-transition");
            store
                .apply_recorded_runtime_transition(start_transition.clone(), false)
                .unwrap();
            let persisted_json = store
                .list_server_run_events("target-run", 0, 1)
                .unwrap()
                .pop()
                .unwrap()
                .payload_json;
            let mut persisted: Value = serde_json::from_str(&persisted_json).unwrap();
            let conn = Connection::open(&path).unwrap();
            let transaction = conn.unchecked_transaction().unwrap();
            for index in 0..history_size {
                persisted["transition_id"] = Value::String(format!("noise-{index}"));
                transaction
                    .execute(
                        "INSERT INTO server_run_events(run_id,event_type,payload_json,created_at_ms) \
                         VALUES (?1,?2,?3,0)",
                        params![
                            format!("noise-run-{index}"),
                            RECORDED_RUNTIME_EVENT_TYPE,
                            serde_json::to_string(&persisted).unwrap()
                        ],
                    )
                    .unwrap();
            }
            transaction.commit().unwrap();
            drop(conn);

            let retry_started = Instant::now();
            for _ in 0..200 {
                assert!(
                    store
                        .apply_recorded_runtime_transition(start_transition.clone(), false)
                        .unwrap()
                        .idempotent
                );
            }
            let retry_ns = retry_started.elapsed().as_nanos();
            let state_started = Instant::now();
            for _ in 0..200 {
                assert!(
                    store
                        .read_recorded_runtime_state("target-run", "wf-1", "conv-1")
                        .unwrap()
                        .is_some()
                );
            }
            let state_ns = state_started.elapsed().as_nanos();
            drop(store);
            fs::remove_file(path).unwrap();
            (retry_ns, state_ns)
        }

        let small = measure(100);
        let large = measure(10_000);
        eprintln!(
            "runtime-serving-scale small_100={{retry_ns:{},state_ns:{}}} \
             large_10000={{retry_ns:{},state_ns:{}}} ratios={{retry:{:.2},state:{:.2}}}",
            small.0,
            small.1,
            large.0,
            large.1,
            large.0 as f64 / small.0.max(1) as f64,
            large.1 as f64 / small.1.max(1) as f64,
        );
    }

    #[test]
    fn global_user_and_scoped_sequences_are_isolated_and_current() {
        let (store, path) = store("sequences");
        assert_eq!(store.current_global_seq().unwrap(), 0);
        assert_eq!(store.next_global_seq().unwrap(), 1);
        assert_eq!(store.next_global_seq().unwrap(), 2);
        assert_eq!(store.current_user_seq("alice").unwrap(), 0);
        assert_eq!(store.next_user_seq("alice").unwrap(), 1);
        assert_eq!(store.next_user_seq("alice").unwrap(), 2);
        assert_eq!(store.next_scoped_seq("project-7").unwrap(), 1);
        assert_eq!(store.current_scoped_seq("project-7").unwrap(), 1);
        assert_eq!(store.current_user_seq("alice").unwrap(), 2);
        store.set_scoped_seq("project-7", 11).unwrap();
        assert_eq!(store.next_user_seq("project-7").unwrap(), 12);
        assert!(matches!(
            store.set_user_seq("alice", -1),
            Err(SqliteStoreError::NegativeSequenceValue { value: -1 })
        ));
        drop(store);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn append_replay_idempotency_and_cross_namespace_collision_follow_contract() {
        let (store, path) = store("events");
        let first = block_on(EventWriteStore::append_entity_event(
            &store,
            "one",
            event("same", "one"),
        ))
        .unwrap();
        let second = block_on(EventWriteStore::append_entity_event(
            &store,
            "one",
            event("second", "two"),
        ))
        .unwrap();
        let retry = block_on(EventWriteStore::append_entity_event(
            &store,
            "one",
            event("same", "changed"),
        ))
        .unwrap();
        assert_eq!((first.event.seq, second.event.seq), (1, 2));
        assert!(first.inserted);
        assert!(!retry.inserted);
        assert_eq!(retry.event.entity_id, "one");
        assert_eq!(
            block_on(EventReadStore::replay_events(&store, "one", 1, 10))
                .unwrap()
                .iter()
                .map(|event| event.seq)
                .collect::<Vec<_>>(),
            [2]
        );
        assert!(matches!(
            store.append_raw_entity_event("two", raw_event("same", "{}")),
            Err(SqliteStoreError::EventIdNamespaceCollision { ref existing_namespace, .. }) if existing_namespace == "one"
        ));
        assert_eq!(
            block_on(EventWriteStore::append_entity_event(
                &store,
                "two",
                event("same", "changed"),
            )),
            Err(StoreError::EventIdNamespaceCollision {
                event_id: "same".to_owned(),
                existing_namespace: "one".to_owned(),
                requested_namespace: "two".to_owned(),
            })
        );
        assert_eq!(store.alloc_event_seq("gap").unwrap(), 1);
        assert_eq!(
            block_on(EventReadStore::latest_event_seq(&store, "gap")).unwrap(),
            0
        );
        drop(store);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn trait_surfaces_sqlite_failures_as_backend_errors() {
        let (store, path) = store("trait-backend-error");
        let conn = Connection::open(&path).unwrap();
        conn.execute("DROP TABLE entity_events", []).unwrap();
        drop(conn);

        assert!(matches!(
            block_on(EventReadStore::replay_events(&store, "ns", 0, 10)),
            Err(StoreError::Backend { ref backend, ref message })
                if backend == "sqlite" && message.contains("entity_events")
        ));
        drop(store);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn cursor_trait_is_strict_while_legacy_set_is_permissive() {
        let (store, path) = store("cursors");
        block_on(EventWriteStore::append_entity_event(
            &store,
            "ns",
            event("one", "one"),
        ))
        .unwrap();
        assert_eq!(
            block_on(EventWriteStore::advance_replay_cursor(
                &store, "ns", "sink", 1
            ))
            .unwrap()
            .last_seq,
            1
        );
        assert_eq!(
            block_on(EventWriteStore::advance_replay_cursor(
                &store, "ns", "sink", 0
            )),
            Err(StoreError::CursorRegresses {
                current: 1,
                requested: 0
            })
        );
        assert_eq!(
            block_on(EventWriteStore::advance_replay_cursor(
                &store, "ns", "sink", 2
            )),
            Err(StoreError::CursorOutOfRange {
                cursor: 2,
                latest: 1
            })
        );
        assert_eq!(
            store
                .set_replay_cursor_legacy("ns", "sink", 99)
                .unwrap()
                .last_seq,
            99
        );
        assert_eq!(
            store
                .set_replay_cursor_legacy("ns", "sink", -7)
                .unwrap()
                .last_seq,
            -7
        );
        drop(store);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn immediate_uow_commits_and_rolls_back_sequences_events_and_cursors_together() {
        let (store, path) = store("uow");
        store
            .immediate_transaction(|uow| {
                assert_eq!(uow.next_global_seq()?, 1);
                assert_eq!(
                    uow.append_raw_entity_event("ns", raw_event("committed", "{}"))?
                        .event
                        .seq,
                    1
                );
                uow.strict_advance_replay_cursor("ns", "sink", 1)?;
                Ok(())
            })
            .unwrap();
        let rollback: SqliteStoreResult<()> = store.immediate_transaction(|uow| {
            assert_eq!(uow.next_global_seq()?, 2);
            assert_eq!(
                uow.append_raw_entity_event("ns", raw_event("rolled-back", "{}"))?
                    .event
                    .seq,
                2
            );
            uow.strict_advance_replay_cursor("ns", "sink", 2)?;
            Err(SqliteStoreError::TransactionAborted(
                "test rollback".to_owned(),
            ))
        });
        assert!(matches!(
            rollback,
            Err(SqliteStoreError::TransactionAborted(_))
        ));
        assert_eq!(store.current_global_seq().unwrap(), 1);
        assert_eq!(store.latest_retained_event_seq("ns").unwrap(), 1);
        assert_eq!(store.replay_cursor("ns", "sink").unwrap().last_seq, 1);
        assert!(
            store
                .replay_raw_events("ns", 0, 10)
                .unwrap()
                .iter()
                .all(|event| event.event_id != "rolled-back")
        );
        drop(store);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn external_transaction_joins_operations_and_commits_or_rolls_back() {
        let (store, path) = store("external-uow");
        store.begin_external_transaction().unwrap();
        assert_eq!(store.next_global_seq().unwrap(), 1);
        store
            .append_raw_entity_event("ns", raw_event("commit", "{}"))
            .unwrap();
        store.commit_external_transaction().unwrap();
        assert_eq!(store.current_global_seq().unwrap(), 1);
        assert_eq!(store.replay_raw_events("ns", 0, 10).unwrap().len(), 1);

        store.begin_external_transaction().unwrap();
        assert_eq!(store.next_global_seq().unwrap(), 2);
        store
            .append_raw_entity_event("ns", raw_event("rollback", "{}"))
            .unwrap();
        store.rollback_external_transaction().unwrap();
        assert_eq!(store.current_global_seq().unwrap(), 1);
        assert_eq!(store.replay_raw_events("ns", 0, 10).unwrap().len(), 1);

        drop(store);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn reopen_is_durable_and_concurrent_global_allocations_are_serialized() {
        let (store, path) = store("durable");
        store.next_global_seq().unwrap();
        store
            .append_raw_entity_event("ns", raw_event("durable", "{\"kept\":true}"))
            .unwrap();
        drop(store);
        let reopened = SqliteStore::open(&path).unwrap();
        assert_eq!(reopened.current_global_seq().unwrap(), 1);
        assert_eq!(
            reopened.replay_raw_events("ns", 0, 1).unwrap()[0].payload_json,
            "{\"kept\":true}"
        );

        let workers = 6;
        let barrier = Arc::new(Barrier::new(workers));
        let mut handles = Vec::new();
        for _ in 0..workers {
            let store = reopened.clone();
            let barrier = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                barrier.wait();
                store.next_global_seq().unwrap()
            }));
        }
        let mut issued = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>();
        issued.sort_unstable();
        assert_eq!(issued, [2, 3, 4, 5, 6, 7]);
        drop(reopened);
        fs::remove_file(path).unwrap();
    }
}
