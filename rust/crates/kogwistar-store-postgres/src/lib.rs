//! PostgreSQL event-store core for ADR-015 Phase 3.
//!
//! This crate intentionally owns only `namespace_seq`, `entity_events`, and
//! `replay_cursors`. Every relation is schema-qualified; correctness never
//! depends on PostgreSQL's `search_path`.

use deadpool_postgres::{Manager, Pool};
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
    AcceptedIndexJobResult, AppendedEvent, AppliedGraphMutation, AuthIdentityStore, AuthUser, EntityEvent,
    EntityRebuildRequest, EntityRecoveryReport, EntityRecoveryRequest, EventPruneStore,
    EventReadStore, EventWriteStore, ExternalIdentity, GraphMutation, GraphMutationStore,
    GraphProjectionRead, GraphProjectionVectorQuery, GraphRecord, IndexJob, IndexJobReadStore,
    IndexJobWriteStore, LaneMessageFilter, LaneMessageReadStore, LaneMessageWriteStore,
    NamedProjection, NamedProjectionWrite, NewEntityEvent, NewIndexJob, NewProjectedLaneMessage,
    ProjectedLaneMessage, ProjectionReadStore, ProjectionWriteStore,
    RUNTIME_CURRENT_STATE_NAMESPACE, ReplayCursor, ResolveExternalIdentity, ServerRun,
    ServerRunCreate, ServerRunEvent, ServerRunReadStore, ServerRunUpdate, ServerRunWriteStore,
    StoreError, StoreResult, VectorMatch, WorkflowDesignDelta, WorkflowDesignDeltaWrite,
    WorkflowDesignHistoryReadStore, WorkflowDesignHistoryWriteStore, WorkflowDesignSnapshot,
    WorkflowDesignSnapshotWrite, runtime_checkpoint_namespace, runtime_projection_write,
    runtime_status_namespace, validate_entity_rebuild_request, validate_entity_recovery_request,
};
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;
use tokio_postgres::{Config, GenericClient, NoTls, Row, Transaction};

/// Future accepted by [`PostgresStore::transaction`].  The lifetime is tied
/// to the UoW borrow, so a pooled client cannot escape its transaction.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

#[derive(Debug, Error)]
pub enum PostgresStoreError {
    #[error("invalid PostgreSQL schema identifier {schema:?}")]
    InvalidSchema { schema: String },
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
    #[error("transaction aborted: {0}")]
    TransactionAborted(String),
    #[error(transparent)]
    RecordedRuntime(#[from] RecordedRuntimeError),
    #[error("recorded runtime conflict: {0}")]
    RecordedRuntimeConflict(String),
    #[error("PostgreSQL operation failed: {0}")]
    Backend(String),
    #[error(transparent)]
    Store(#[from] StoreError),
}

pub type PostgresStoreResult<T> = Result<T, PostgresStoreError>;

/// Event row retaining exact `payload_json` bytes supplied by a caller.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawEntityEvent {
    pub namespace: String,
    pub seq: i64,
    pub event_id: String,
    pub entity_kind: String,
    pub entity_id: String,
    pub op: String,
    pub payload_json: String,
    /// PostgreSQL's textual `TIMESTAMPTZ` representation. It is deliberately
    /// not parsed so raw event APIs add no time-library dependency.
    pub created_at: String,
}

/// Raw append input. Unlike trait input, JSON is not reparsed or reformatted.
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

/// Existing Python pgvector collection names accepted by the explicit graph
/// schema action.  Every identifier is validated and quoted before SQL build.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphTableNames {
    pub nodes: String,
    pub edges: String,
    pub documents: String,
    pub domains: String,
}

/// One atomic metadata patch for an existing Python pgvector graph row.
///
/// Lifecycle updates use merge semantics in Python: keys absent from `patch`
/// survive in both the serialized document and the projection metadata.  This
/// request preserves that contract while joining event append to the same
/// PostgreSQL transaction.
#[derive(Clone, Debug, PartialEq)]
pub struct GraphMetadataPatchMutation {
    pub scope: kogwistar_store::GraphScope,
    pub table: String,
    pub entity_kind: String,
    pub event_id: String,
    pub op: String,
    pub entity_id: String,
    pub document: Option<String>,
    pub metadata_patch: Map<String, Value>,
    pub payload: Value,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GraphDeleteMutation {
    pub scope: kogwistar_store::GraphScope,
    pub table: String,
    pub entity_kind: String,
    pub event_id: String,
    pub entity_id: String,
    pub payload: Value,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GraphProjectionUpsert {
    pub scope: kogwistar_store::GraphScope,
    pub table: String,
    pub record: GraphRecord,
    pub embedding_dim: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GraphProjectionMetadataPatch {
    pub scope: kogwistar_store::GraphScope,
    pub table: String,
    pub entity_id: String,
    pub document: Option<String>,
    pub metadata_patch: Map<String, Value>,
    pub patch_document_metadata: bool,
}

impl Default for GraphTableNames {
    fn default() -> Self {
        Self {
            nodes: "gke_nodes".to_owned(),
            edges: "gke_edges".to_owned(),
            documents: "gke_documents".to_owned(),
            domains: "gke_domains".to_owned(),
        }
    }
}

#[derive(Clone, Debug)]
struct Tables {
    schema: String,
    quoted_schema: String,
    global_seq: String,
    user_seq: String,
    index_applied_state: String,
    index_applied_state_key_index: String,
    namespace_seq: String,
    entity_events: String,
    replay_cursors: String,
    aggregate_index: String,
    named_projections: String,
    named_projections_namespace_index: String,
    workflow_design_snapshots: String,
    workflow_design_version_deltas: String,
    server_runs: String,
    server_run_events: String,
    server_runs_status_index: String,
    server_run_events_run_seq_index: String,
    server_run_events_transition_id_index: String,
    index_jobs: String,
    index_jobs_status_lease_index: String,
    index_jobs_status_next_run_index: String,
    index_jobs_entity_index: String,
    index_jobs_namespace_index: String,
    index_jobs_pending_index: String,
    projected_lane_messages: String,
    lane_messages_namespace_inbox_seq_index: String,
    lane_messages_claim_index: String,
    lane_messages_conversation_seq_index: String,
}

impl Tables {
    fn new(schema: &str) -> PostgresStoreResult<Self> {
        let quoted_schema = quote_identifier(schema)?;
        Ok(Self {
            schema: schema.to_owned(),
            quoted_schema: quoted_schema.clone(),
            global_seq: qualified(&quoted_schema, "global_seq")?,
            user_seq: qualified(&quoted_schema, "user_seq")?,
            index_applied_state: qualified(&quoted_schema, "index_applied_state")?,
            index_applied_state_key_index: quote_identifier("idx_index_applied_state_key")?,
            namespace_seq: qualified(&quoted_schema, "namespace_seq")?,
            entity_events: qualified(&quoted_schema, "entity_events")?,
            replay_cursors: qualified(&quoted_schema, "replay_cursors")?,
            // PostgreSQL creates an index in the indexed table's schema and
            // rejects a schema-qualified index name in CREATE INDEX syntax.
            aggregate_index: quote_identifier("idx_entity_events_aggregate")?,
            named_projections: qualified(&quoted_schema, "named_projections")?,
            named_projections_namespace_index: quote_identifier("idx_named_projections_namespace")?,
            workflow_design_snapshots: qualified(&quoted_schema, "workflow_design_snapshots")?,
            workflow_design_version_deltas: qualified(
                &quoted_schema,
                "workflow_design_version_deltas",
            )?,
            server_runs: qualified(&quoted_schema, "server_runs")?,
            server_run_events: qualified(&quoted_schema, "server_run_events")?,
            server_runs_status_index: quote_identifier("idx_server_runs_status")?,
            server_run_events_run_seq_index: quote_identifier("idx_server_run_events_run_seq")?,
            server_run_events_transition_id_index: quote_identifier(
                "idx_server_run_events_recorded_transition_id",
            )?,
            index_jobs: qualified(&quoted_schema, "index_jobs")?,
            index_jobs_status_lease_index: quote_identifier("idx_index_jobs_status_lease")?,
            index_jobs_status_next_run_index: quote_identifier("idx_index_jobs_status_next_run")?,
            index_jobs_entity_index: quote_identifier("idx_index_jobs_entity")?,
            index_jobs_namespace_index: quote_identifier("idx_index_jobs_namespace")?,
            index_jobs_pending_index: quote_identifier("uq_index_jobs_pending_ns_ck")?,
            projected_lane_messages: qualified(&quoted_schema, "projected_lane_messages")?,
            lane_messages_namespace_inbox_seq_index: quote_identifier(
                "idx_lane_messages_namespace_inbox_seq",
            )?,
            lane_messages_claim_index: quote_identifier("idx_lane_messages_claim")?,
            lane_messages_conversation_seq_index: quote_identifier(
                "idx_lane_messages_conversation_seq",
            )?,
        })
    }
}

/// Cloneable async store backed by `deadpool-postgres` and `tokio-postgres`.
#[derive(Clone)]
pub struct PostgresStore {
    pool: Pool,
    tables: Tables,
}

/// Independent Python-compatible PostgreSQL `AUTH_DB_URL` store. Relations
/// remain unqualified so PostgreSQL DSN `search_path` keeps SQLAlchemy's
/// existing deployment behavior.
#[derive(Clone)]
pub struct PostgresAuthStore {
    pool: Pool,
}

impl std::fmt::Debug for PostgresAuthStore {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PostgresAuthStore")
            .finish_non_exhaustive()
    }
}

impl PostgresAuthStore {
    pub fn from_config(config: Config) -> PostgresStoreResult<Self> {
        let manager = Manager::new(config, NoTls);
        let pool = Pool::builder(manager)
            .build()
            .map_err(|error| PostgresStoreError::Backend(error.to_string()))?;
        Ok(Self { pool })
    }

    pub fn from_dsn(dsn: &str) -> PostgresStoreResult<Self> {
        let config = dsn
            .parse::<Config>()
            .map_err(|error| PostgresStoreError::Backend(error.to_string()))?;
        Self::from_config(config)
    }

    pub async fn ensure_schema(&self) -> PostgresStoreResult<()> {
        let client = self.client().await?;
        client
            .batch_execute(
                "CREATE TABLE IF NOT EXISTS users (
                    user_id VARCHAR(64) NOT NULL PRIMARY KEY,
                    email VARCHAR(255) NOT NULL UNIQUE,
                    display_name VARCHAR(255),
                    is_active BOOLEAN NOT NULL,
                    global_role VARCHAR(32),
                    global_ns VARCHAR(255),
                    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
                    last_login_at TIMESTAMP WITHOUT TIME ZONE
                );
                CREATE INDEX IF NOT EXISTS ix_users_email ON users (email);
                CREATE TABLE IF NOT EXISTS external_identities (
                    issuer VARCHAR(255) NOT NULL,
                    subject VARCHAR(255) NOT NULL,
                    user_id VARCHAR(64) NOT NULL REFERENCES users(user_id),
                    email VARCHAR(255),
                    PRIMARY KEY (issuer, subject)
                );
                CREATE INDEX IF NOT EXISTS ix_external_identities_user_id ON external_identities (user_id);
                CREATE TABLE IF NOT EXISTS workflow_acl (
                    workflow_id VARCHAR(255) NOT NULL,
                    user_id VARCHAR(64) NOT NULL REFERENCES users(user_id),
                    role VARCHAR(32) NOT NULL,
                    PRIMARY KEY (workflow_id, user_id)
                );",
            )
            .await
            .map_err(backend)
    }

    async fn client(&self) -> PostgresStoreResult<deadpool_postgres::Object> {
        self.pool
            .get()
            .await
            .map_err(|error| PostgresStoreError::Backend(error.to_string()))
    }
}

fn postgres_auth_user(row: &Row) -> AuthUser {
    AuthUser {
        user_id: row.get(0),
        email: row.get(1),
        display_name: row.get(2),
        is_active: row.get(3),
        global_role: row.get(4),
        global_ns: row.get(5),
    }
}

fn postgres_auth_store_error(error: PostgresStoreError) -> StoreError {
    StoreError::Backend {
        backend: "postgres".to_owned(),
        message: error.to_string(),
    }
}

impl AuthIdentityStore for PostgresAuthStore {
    async fn auth_user(&self, user_id: &str) -> StoreResult<Option<AuthUser>> {
        let client = self.client().await.map_err(postgres_auth_store_error)?;
        client
            .query_opt(
                "SELECT user_id, email, display_name, is_active, global_role, global_ns FROM users WHERE user_id = $1",
                &[&user_id],
            )
            .await
            .map(|row| row.as_ref().map(postgres_auth_user))
            .map_err(|error| postgres_auth_store_error(backend(error)))
    }

    async fn external_identity(
        &self,
        issuer: &str,
        subject: &str,
    ) -> StoreResult<Option<ExternalIdentity>> {
        let client = self.client().await.map_err(postgres_auth_store_error)?;
        client
            .query_opt(
                "SELECT issuer, subject, user_id, email FROM external_identities WHERE issuer = $1 AND subject = $2",
                &[&issuer, &subject],
            )
            .await
            .map(|row| {
                row.map(|row| ExternalIdentity {
                    issuer: row.get(0),
                    subject: row.get(1),
                    user_id: row.get(2),
                    email: row.get(3),
                })
            })
            .map_err(|error| postgres_auth_store_error(backend(error)))
    }

    async fn resolve_external_identity(
        &self,
        request: ResolveExternalIdentity,
    ) -> StoreResult<AuthUser> {
        let mut client = self.client().await.map_err(postgres_auth_store_error)?;
        let transaction = client
            .transaction()
            .await
            .map_err(|error| postgres_auth_store_error(backend(error)))?;
        transaction
            .query_one(
                "SELECT pg_advisory_xact_lock(hashtextextended($1::text, hashtextextended($2::text, 0)))",
                &[&request.issuer, &request.subject],
            )
            .await
            .map_err(|error| postgres_auth_store_error(backend(error)))?;
        let existing_identity = transaction
            .query_opt(
                "SELECT user_id FROM external_identities WHERE issuer = $1 AND subject = $2 FOR UPDATE",
                &[&request.issuer, &request.subject],
            )
            .await
            .map_err(|error| postgres_auth_store_error(backend(error)))?;
        let user_id = if let Some(row) = existing_identity {
            row.get::<_, String>(0)
        } else {
            transaction
                .execute(
                    "INSERT INTO users (user_id, email, display_name, is_active, global_role, global_ns, created_at) VALUES ($1, $2, $3, TRUE, $4, $5, CURRENT_TIMESTAMP) ON CONFLICT (email) DO NOTHING",
                    &[&request.new_user_id, &request.email, &request.display_name, &request.default_role, &request.default_ns],
                )
                .await
                .map_err(|error| postgres_auth_store_error(backend(error)))?;
            let user_id = transaction
                .query_one(
                    "SELECT user_id FROM users WHERE email = $1",
                    &[&request.email],
                )
                .await
                .map_err(|error| postgres_auth_store_error(backend(error)))?
                .get::<_, String>(0);
            transaction
                .execute(
                    "INSERT INTO external_identities (issuer, subject, user_id, email) VALUES ($1, $2, $3, $4) ON CONFLICT (issuer, subject) DO NOTHING",
                    &[&request.issuer, &request.subject, &user_id, &request.email],
                )
                .await
                .map_err(|error| postgres_auth_store_error(backend(error)))?;
            transaction
                .query_one(
                    "SELECT user_id FROM external_identities WHERE issuer = $1 AND subject = $2",
                    &[&request.issuer, &request.subject],
                )
                .await
                .map_err(|error| postgres_auth_store_error(backend(error)))?
                .get::<_, String>(0)
        };
        let row = transaction
            .query_one(
                "UPDATE users SET last_login_at = CURRENT_TIMESTAMP WHERE user_id = $1 RETURNING user_id, email, display_name, is_active, global_role, global_ns",
                &[&user_id],
            )
            .await
            .map_err(|error| postgres_auth_store_error(backend(error)))?;
        let user = postgres_auth_user(&row);
        transaction
            .commit()
            .await
            .map_err(|error| postgres_auth_store_error(backend(error)))?;
        Ok(user)
    }
}

impl std::fmt::Debug for PostgresStore {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PostgresStore")
            .field("schema", &self.tables.schema)
            .finish_non_exhaustive()
    }
}

impl PostgresStore {
    /// Build a store from a parsed `tokio_postgres::Config`.
    pub fn from_config(config: Config, schema: impl AsRef<str>) -> PostgresStoreResult<Self> {
        let tables = Tables::new(schema.as_ref())?;
        let manager = Manager::new(config, NoTls);
        let pool = Pool::builder(manager)
            .build()
            .map_err(|error| PostgresStoreError::Backend(error.to_string()))?;
        Ok(Self { pool, tables })
    }

    /// Build a store from a libpq-style DSN accepted by `tokio-postgres`.
    pub fn from_dsn(dsn: &str, schema: impl AsRef<str>) -> PostgresStoreResult<Self> {
        let config = dsn
            .parse::<Config>()
            .map_err(|error| PostgresStoreError::Backend(error.to_string()))?;
        Self::from_config(config, schema)
    }

    /// Build from an existing pool. This is useful when app wiring owns pool
    /// sizing or connection recycling policy.
    pub fn from_pool(pool: Pool, schema: impl AsRef<str>) -> PostgresStoreResult<Self> {
        Ok(Self {
            pool,
            tables: Tables::new(schema.as_ref())?,
        })
    }

    pub fn schema(&self) -> &str {
        &self.tables.schema
    }

    /// Create the exact Python `EnginePostgresMetaStore` Phase-2b tables.
    pub async fn ensure_schema(&self) -> PostgresStoreResult<()> {
        let client = self.client().await?;
        client
            .batch_execute(&schema_sql(&self.tables))
            .await
            .map_err(backend)
    }

    /// Explicit operator action: create only Python-compatible pgvector graph
    /// collections.  Ordinary `ensure_schema` deliberately never enables an
    /// extension or creates graph projection tables.
    pub async fn create_graph_schema(
        &self,
        embedding_dim: usize,
        names: GraphTableNames,
    ) -> PostgresStoreResult<()> {
        let client = self.client().await?;
        create_graph_schema(&**client, &self.tables, embedding_dim, names).await
    }

    /// Run append/cursor work in one RAII PostgreSQL transaction.
    ///
    /// The pooled client stays local to this method. `PostgresUnitOfWork`
    /// owns its `Transaction<'_>` borrow, so the client cannot escape and a
    /// cancelled future, panic, or dropped UoW lets `Transaction::drop`
    /// perform PostgreSQL's rollback protection. Returning an error also
    /// explicitly rolls back every UoW operation.
    pub async fn transaction<T, F>(&self, operation: F) -> PostgresStoreResult<T>
    where
        F: for<'a> FnOnce(&'a mut PostgresUnitOfWork) -> BoxFuture<'a, PostgresStoreResult<T>>,
    {
        let mut pooled_client = self.client().await?;
        // `deadpool_postgres::Object::transaction` returns deadpool's wrapper.
        // Bind its underlying tokio client first, so UoW owns the native RAII
        // `tokio_postgres::Transaction`, not a pool wrapper or raw SQL.
        let client: &mut tokio_postgres::Client = &mut pooled_client;
        let transaction = client.transaction().await.map_err(backend)?;
        let mut uow = PostgresUnitOfWork {
            transaction,
            tables: self.tables.clone(),
        };
        match operation(&mut uow).await {
            Ok(value) => {
                uow.transaction.commit().await.map_err(backend)?;
                Ok(value)
            }
            Err(error) => {
                if let Err(rollback) = uow.transaction.rollback().await {
                    return Err(PostgresStoreError::Backend(format!(
                        "{error}; rollback failed: {rollback}"
                    )));
                }
                Err(error)
            }
        }
    }

    pub async fn alloc_event_seq(&self, namespace: &str) -> PostgresStoreResult<i64> {
        require_namespace(namespace)?;
        let namespace = namespace.to_owned();
        self.transaction(move |uow| Box::pin(async move { uow.alloc_event_seq(&namespace).await }))
            .await
    }

    pub async fn next_global_seq(&self) -> PostgresStoreResult<i64> {
        self.transaction(|uow| Box::pin(async move { uow.next_global_seq().await }))
            .await
    }

    pub async fn current_global_seq(&self) -> PostgresStoreResult<i64> {
        let client = self.client().await?;
        current_global_seq(&**client, &self.tables).await
    }

    pub async fn next_user_seq(&self, user_id: &str) -> PostgresStoreResult<i64> {
        let user_id = user_id.to_owned();
        self.transaction(move |uow| Box::pin(async move { uow.next_user_seq(&user_id).await }))
            .await
    }

    pub async fn current_user_seq(&self, user_id: &str) -> PostgresStoreResult<i64> {
        let client = self.client().await?;
        current_user_seq(&**client, &self.tables, user_id).await
    }

    pub async fn set_user_seq(&self, user_id: &str, value: i64) -> PostgresStoreResult<()> {
        let user_id = user_id.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move { uow.set_user_seq(&user_id, value).await })
        })
        .await
    }

    pub async fn get_index_applied_fingerprint(
        &self,
        namespace: &str,
        coalesce_key: &str,
    ) -> PostgresStoreResult<Option<String>> {
        let client = self.client().await?;
        get_index_applied_fingerprint(&**client, &self.tables, namespace, coalesce_key).await
    }

    pub async fn set_index_applied_fingerprint(
        &self,
        namespace: &str,
        coalesce_key: &str,
        applied_fingerprint: Option<String>,
        last_job_id: Option<String>,
    ) -> PostgresStoreResult<()> {
        let namespace = namespace.to_owned();
        let coalesce_key = coalesce_key.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.set_index_applied_fingerprint(
                    &namespace,
                    &coalesce_key,
                    applied_fingerprint,
                    last_job_id,
                )
                .await
            })
        })
        .await
    }

    /// Apply graph projection row and authoritative event in one PostgreSQL
    /// transaction.  `gke_*` remains Python's schema; this merely joins it.
    pub async fn apply_graph_mutation(
        &self,
        mutation: GraphMutation,
    ) -> PostgresStoreResult<AppliedGraphMutation> {
        self.transaction(move |uow| {
            Box::pin(async move { uow.apply_graph_mutation(mutation).await })
        })
        .await
    }

    pub async fn apply_graph_metadata_patch_mutation(
        &self,
        mutation: GraphMetadataPatchMutation,
    ) -> PostgresStoreResult<Option<AppliedGraphMutation>> {
        self.transaction(move |uow| {
            Box::pin(async move { uow.apply_graph_metadata_patch_mutation(mutation).await })
        })
        .await
    }

    pub async fn apply_graph_delete_mutation(
        &self,
        mutation: GraphDeleteMutation,
    ) -> PostgresStoreResult<Option<AppliedGraphMutation>> {
        self.transaction(move |uow| {
            Box::pin(async move { uow.apply_graph_delete_mutation(mutation).await })
        })
        .await
    }

    pub async fn upsert_graph_projection(
        &self,
        write: GraphProjectionUpsert,
    ) -> PostgresStoreResult<()> {
        self.transaction(move |uow| {
            Box::pin(async move { uow.upsert_graph_projection(write).await })
        })
        .await
    }

    pub async fn patch_graph_projection_metadata(
        &self,
        patch: GraphProjectionMetadataPatch,
    ) -> PostgresStoreResult<bool> {
        self.transaction(move |uow| {
            Box::pin(async move { uow.patch_graph_projection_metadata(patch).await })
        })
        .await
    }

    pub async fn graph_projection_records(
        &self,
        read: GraphProjectionRead,
    ) -> PostgresStoreResult<Vec<GraphRecord>> {
        let client = self.client().await?;
        graph_projection_records(&**client, &self.tables, read).await
    }

    pub async fn graph_projection_vector_query(
        &self,
        query: GraphProjectionVectorQuery,
    ) -> PostgresStoreResult<Vec<VectorMatch>> {
        let client = self.client().await?;
        graph_projection_vector_query(&**client, &self.tables, query).await
    }

    pub async fn append_raw_entity_event(
        &self,
        namespace: &str,
        event: NewRawEntityEvent,
    ) -> PostgresStoreResult<AppendedRawEvent> {
        require_namespace(namespace)?;
        let namespace = namespace.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move { uow.append_raw_entity_event(&namespace, event).await })
        })
        .await
    }

    /// Raw replay is exclusive of `after_seq`, ordered ascending, and bounded.
    pub async fn replay_raw_events(
        &self,
        namespace: &str,
        after_seq: i64,
        limit: usize,
    ) -> PostgresStoreResult<Vec<RawEntityEvent>> {
        require_namespace(namespace)?;
        let client = self.client().await?;
        replay_raw_events(&**client, &self.tables, namespace, after_seq, limit).await
    }

    /// `MAX(seq)` over retained rows, not allocation state. Gaps are legal.
    pub async fn latest_retained_event_seq(&self, namespace: &str) -> PostgresStoreResult<i64> {
        require_namespace(namespace)?;
        let client = self.client().await?;
        latest_retained_event_seq(&**client, &self.tables, namespace).await
    }

    pub async fn prune_entity_events_after(
        &self,
        namespace: &str,
        to_seq: i64,
    ) -> PostgresStoreResult<u64> {
        require_namespace(namespace)?;
        let namespace = namespace.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move { uow.prune_entity_events_after(&namespace, to_seq).await })
        })
        .await
    }

    pub async fn replay_cursor(
        &self,
        namespace: &str,
        consumer: &str,
    ) -> PostgresStoreResult<ReplayCursor> {
        require_namespace(namespace)?;
        let client = self.client().await?;
        replay_cursor(&**client, &self.tables, namespace, consumer).await
    }

    /// Bounded operator-triggered recovery. Projection replacement and cursor
    /// movement are committed only together with the authoritative replay fold.
    pub async fn recover_entity_projection(
        &self,
        request: EntityRecoveryRequest,
    ) -> PostgresStoreResult<EntityRecoveryReport> {
        validate_entity_recovery_request(&request)?;
        self.transaction(move |uow| {
            Box::pin(async move { uow.recover_entity_projection(&request).await })
        })
        .await
    }

    /// Explicit full rebuild. `ensure_schema` and opening a store never call it.
    pub async fn rebuild_entity_projection(
        &self,
        request: EntityRebuildRequest,
    ) -> PostgresStoreResult<EntityRecoveryReport> {
        validate_entity_rebuild_request(&request)?;
        self.transaction(move |uow| {
            Box::pin(async move { uow.rebuild_entity_projection(&request).await })
        })
        .await
    }

    pub async fn get_named_projection(
        &self,
        namespace: &str,
        key: &str,
    ) -> PostgresStoreResult<Option<NamedProjection>> {
        require_namespace(namespace)?;
        let client = self.client().await?;
        named_projection(&**client, &self.tables, namespace, key).await
    }

    pub async fn list_named_projections(
        &self,
        namespace: &str,
    ) -> PostgresStoreResult<Vec<NamedProjection>> {
        require_namespace(namespace)?;
        let client = self.client().await?;
        named_projections(&**client, &self.tables, namespace).await
    }

    pub async fn replace_named_projection(
        &self,
        namespace: &str,
        key: &str,
        projection: NamedProjectionWrite,
    ) -> PostgresStoreResult<()> {
        require_namespace(namespace)?;
        let namespace = namespace.to_owned();
        let key = key.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.replace_named_projection(&namespace, &key, projection)
                    .await
            })
        })
        .await
    }

    pub async fn compare_and_swap_named_projection(
        &self,
        namespace: &str,
        key: &str,
        expected_last_authoritative_seq: Option<i64>,
        expected_last_materialized_seq: Option<i64>,
        projection: NamedProjectionWrite,
    ) -> PostgresStoreResult<bool> {
        require_namespace(namespace)?;
        let namespace = namespace.to_owned();
        let key = key.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.compare_and_swap_named_projection(
                    &namespace,
                    &key,
                    expected_last_authoritative_seq,
                    expected_last_materialized_seq,
                    projection,
                )
                .await
            })
        })
        .await
    }

    pub async fn clear_named_projection(
        &self,
        namespace: &str,
        key: &str,
    ) -> PostgresStoreResult<()> {
        require_namespace(namespace)?;
        let namespace = namespace.to_owned();
        let key = key.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move { uow.clear_named_projection(&namespace, &key).await })
        })
        .await
    }

    pub async fn clear_projection_namespace(&self, namespace: &str) -> PostgresStoreResult<()> {
        require_namespace(namespace)?;
        let namespace = namespace.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move { uow.clear_projection_namespace(&namespace).await })
        })
        .await
    }

    pub async fn put_workflow_design_snapshot(
        &self,
        workflow_id: &str,
        snapshot: WorkflowDesignSnapshotWrite,
    ) -> PostgresStoreResult<()> {
        let workflow_id = workflow_id.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.put_workflow_design_snapshot(&workflow_id, snapshot)
                    .await
            })
        })
        .await
    }

    pub async fn get_workflow_design_snapshot(
        &self,
        workflow_id: &str,
        max_version: i64,
        schema_version: i64,
    ) -> PostgresStoreResult<Option<WorkflowDesignSnapshot>> {
        let client = self.client().await?;
        workflow_design_snapshot(
            &**client,
            &self.tables,
            workflow_id,
            max_version,
            schema_version,
        )
        .await
    }

    pub async fn clear_workflow_design_snapshots(
        &self,
        workflow_id: &str,
    ) -> PostgresStoreResult<()> {
        let workflow_id = workflow_id.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move { uow.clear_workflow_design_snapshots(&workflow_id).await })
        })
        .await
    }

    pub async fn put_workflow_design_delta(
        &self,
        workflow_id: &str,
        delta: WorkflowDesignDeltaWrite,
    ) -> PostgresStoreResult<()> {
        let workflow_id = workflow_id.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move { uow.put_workflow_design_delta(&workflow_id, delta).await })
        })
        .await
    }

    pub async fn get_workflow_design_delta(
        &self,
        workflow_id: &str,
        version: i64,
        schema_version: i64,
    ) -> PostgresStoreResult<Option<WorkflowDesignDelta>> {
        let client = self.client().await?;
        workflow_design_delta(
            &**client,
            &self.tables,
            workflow_id,
            version,
            schema_version,
        )
        .await
    }

    pub async fn clear_workflow_design_deltas(&self, workflow_id: &str) -> PostgresStoreResult<()> {
        let workflow_id = workflow_id.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move { uow.clear_workflow_design_deltas(&workflow_id).await })
        })
        .await
    }

    pub async fn create_server_run(&self, run: ServerRunCreate) -> PostgresStoreResult<()> {
        self.transaction(move |uow| Box::pin(async move { uow.create_server_run(run).await }))
            .await
    }
    pub async fn get_server_run(&self, run_id: &str) -> PostgresStoreResult<Option<ServerRun>> {
        server_run(&**self.client().await?, &self.tables, run_id).await
    }
    pub async fn list_server_runs(
        &self,
        status: Option<&str>,
        workflow_id: Option<&str>,
        conversation_id: Option<&str>,
        limit: usize,
    ) -> PostgresStoreResult<Vec<ServerRun>> {
        server_runs(
            &**self.client().await?,
            &self.tables,
            status,
            workflow_id,
            conversation_id,
            limit,
        )
        .await
    }
    pub async fn append_server_run_event(
        &self,
        run_id: &str,
        event_type: &str,
        payload_json: String,
    ) -> PostgresStoreResult<ServerRunEvent> {
        let run_id = run_id.to_owned();
        let event_type = event_type.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.append_server_run_event(&run_id, &event_type, payload_json)
                    .await
            })
        })
        .await
    }
    pub async fn list_server_run_events(
        &self,
        run_id: &str,
        after_seq: i64,
        limit: usize,
    ) -> PostgresStoreResult<Vec<ServerRunEvent>> {
        server_run_events(
            &**self.client().await?,
            &self.tables,
            run_id,
            after_seq,
            limit,
        )
        .await
    }
    pub async fn update_server_run(
        &self,
        run_id: &str,
        update: ServerRunUpdate,
    ) -> PostgresStoreResult<()> {
        let run_id = run_id.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move { uow.update_server_run(&run_id, update).await })
        })
        .await
    }
    pub async fn request_server_run_cancel(&self, run_id: &str) -> PostgresStoreResult<()> {
        let run_id = run_id.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move { uow.request_server_run_cancel(&run_id).await })
        })
        .await
    }

    pub async fn apply_recorded_runtime_transition(
        &self,
        transition: RecordedRuntimeTransition,
        abort_after_writes: bool,
    ) -> PostgresStoreResult<RecordedTransitionResult> {
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.apply_recorded_runtime_transition(transition, abort_after_writes)
                    .await
            })
        })
        .await
    }

    pub async fn apply_claimed_recorded_runtime_transition(
        &self,
        handoff: RecordedWorkerHandoff,
        transition: RecordedRuntimeTransition,
        abort_after_writes: bool,
    ) -> PostgresStoreResult<RecordedTransitionResult> {
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.apply_claimed_recorded_runtime_transition(
                    handoff,
                    transition,
                    abort_after_writes,
                )
                .await
            })
        })
        .await
    }

    pub async fn apply_claimed_recorded_worker_effect(
        &self,
        handoff: RecordedWorkerHandoff,
        effect: RecordedWorkerSuccessEffect,
    ) -> PostgresStoreResult<RecordedTransitionResult> {
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.apply_claimed_recorded_worker_effect(handoff, effect)
                    .await
            })
        })
        .await
    }

    pub async fn resume_recorded_runtime_token(
        &self,
        run_id: &str,
        workflow_id: &str,
        conversation_id: &str,
        node_id: &str,
        token_id: &str,
        resume_payload: Option<Value>,
    ) -> PostgresStoreResult<RecordedTransitionResult> {
        let run_id = run_id.to_owned();
        let workflow_id = workflow_id.to_owned();
        let conversation_id = conversation_id.to_owned();
        let node_id = node_id.to_owned();
        let token_id = token_id.to_owned();
        self.transaction(|uow| {
            Box::pin(async move {
                let current = read_recorded_runtime_state(
                    &uow.transaction,
                    &uow.tables,
                    &run_id,
                    &workflow_id,
                    &conversation_id,
                )
                .await?
                .ok_or_else(|| {
                    PostgresStoreError::RecordedRuntimeConflict(format!(
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
                let expected_event_seq =
                    latest_server_run_event_seq(&uow.transaction, &uow.tables, &run_id).await?;
                let result = uow
                    .apply_recorded_runtime_transition(
                        RecordedRuntimeTransition {
                            contract_version: RECORDED_RUNTIME_CONTRACT_VERSION,
                            transition_id: format!(
                                "resume-{run_id}-{token_id}-{expected_event_seq}"
                            ),
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
                    )
                    .await?;
                let message_id = format!(
                    "lane|{}",
                    kogwistar_contracts::stable_id(
                        "runtime.worker.request",
                        &[
                            run_id.clone(),
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
                let now = unix_epoch_millis() / 1000;
                let join_mask = result
                    .reduced
                    .state
                    .frontier
                    .pending
                    .iter()
                    .find(|(_, _, token, _)| token == &token_id)
                    .map(|(_, mask, _, _)| *mask)
                    .unwrap_or(0);
                let op = result
                    .reduced
                    .state
                    .state
                    .get("_rt_node_ops")
                    .and_then(Value::as_object)
                    .and_then(|ops| ops.get(&node_id))
                    .cloned()
                    .unwrap_or_else(|| Value::String("noop".to_owned()));
                let payload = RuntimeStepExecutePayload::from_recorded_state(
                    &result.reduced.state,
                    RuntimeStepExecuteRequest {
                        node_id: node_id.clone(),
                        op,
                        join_mask,
                        token_id: token_id.clone(),
                        parent_token_id: parent.clone(),
                        step_seq: result.reduced.state.last_step_seq.saturating_add(1),
                        expected_event_seq: result.event_seq,
                        resume_effect: result.reduced.state.resume_payload.clone(),
                    },
                );
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
                    payload_json: Some(serde_json::to_string(&payload)?),
                    error_json: None,
                })
                .await?;
                Ok(result)
            })
        })
        .await
    }

    pub async fn read_recorded_runtime_state(
        &self,
        run_id: &str,
        workflow_id: &str,
        conversation_id: &str,
    ) -> PostgresStoreResult<Option<RecordedRuntimeState>> {
        read_recorded_runtime_state(
            &**self.client().await?,
            &self.tables,
            run_id,
            workflow_id,
            conversation_id,
        )
        .await
    }

    pub async fn enqueue_index_job(&self, job: NewIndexJob) -> PostgresStoreResult<String> {
        let namespace = job.namespace.clone();
        require_namespace(&namespace)?;
        self.transaction(move |uow| Box::pin(async move { uow.enqueue_index_job(job).await }))
            .await
    }
    pub async fn claim_index_jobs(
        &self,
        limit: usize,
        lease_seconds: i64,
        namespace: Option<&str>,
    ) -> PostgresStoreResult<Vec<IndexJob>> {
        let namespace = namespace.map(ToOwned::to_owned);
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.claim_index_jobs(limit, lease_seconds, namespace.as_deref())
                    .await
            })
        })
        .await
    }
    pub async fn mark_index_job_done(
        &self,
        job_id: &str,
        claim_token: Option<&str>,
    ) -> PostgresStoreResult<bool> {
        let job_id = job_id.to_owned();
        let claim_token = claim_token.map(ToOwned::to_owned);
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.mark_index_job_done(&job_id, claim_token.as_deref())
                    .await
            })
        })
        .await
    }
    pub async fn accept_index_job_result(
        &self,
        job_id: &str,
        claim_token: &str,
        result_json: &str,
        result_sha256: &str,
    ) -> PostgresStoreResult<AcceptedIndexJobResult> {
        let job_id = job_id.to_owned();
        let claim_token = claim_token.to_owned();
        let result_json = result_json.to_owned();
        let result_sha256 = result_sha256.to_owned();
        self.transaction(move |uow| Box::pin(async move {
            uow.accept_index_job_result(&job_id, &claim_token, &result_json, &result_sha256).await
        })).await
    }
    pub async fn index_job_result(&self, job_id: &str) -> PostgresStoreResult<Option<AcceptedIndexJobResult>> {
        index_job_result(&**self.client().await?, &self.tables, job_id).await
    }
    pub async fn mark_index_job_failed(
        &self,
        job_id: &str,
        error: &str,
        final_: bool,
        claim_token: Option<&str>,
    ) -> PostgresStoreResult<()> {
        let job_id = job_id.to_owned();
        let error = error.to_owned();
        let claim_token = claim_token.map(ToOwned::to_owned);
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.mark_index_job_failed(&job_id, &error, final_, claim_token.as_deref())
                    .await
            })
        })
        .await
    }
    pub async fn bump_retry_and_requeue(
        &self,
        job_id: &str,
        error: &str,
        delay: i64,
        claim_token: Option<&str>,
    ) -> PostgresStoreResult<()> {
        let job_id = job_id.to_owned();
        let error = error.to_owned();
        let claim_token = claim_token.map(ToOwned::to_owned);
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.bump_retry_and_requeue(&job_id, &error, delay, claim_token.as_deref())
                    .await
            })
        })
        .await
    }
    pub async fn renew_index_job_lease(
        &self,
        job_id: &str,
        claim_token: &str,
        lease_seconds: i64,
    ) -> PostgresStoreResult<bool> {
        let job_id = job_id.to_owned();
        let claim_token = claim_token.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.renew_index_job_lease(&job_id, &claim_token, lease_seconds)
                    .await
            })
        })
        .await
    }
    pub async fn requeue_index_job_at_tail(
        &self,
        job_id: &str,
        payload_json: String,
        delay: i64,
        claim_token: Option<&str>,
    ) -> PostgresStoreResult<()> {
        let job_id = job_id.to_owned();
        let claim_token = claim_token.map(ToOwned::to_owned);
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.requeue_index_job_at_tail(&job_id, payload_json, delay, claim_token.as_deref())
                    .await
            })
        })
        .await
    }
    pub async fn list_index_jobs(
        &self,
        namespace: Option<&str>,
        status: Option<&str>,
        entity_kind: Option<&str>,
        entity_id: Option<&str>,
        index_kind: Option<&str>,
        limit: usize,
    ) -> PostgresStoreResult<Vec<IndexJob>> {
        let client = self.client().await?;
        list_index_jobs(
            &**client,
            &self.tables,
            namespace,
            status,
            entity_kind,
            entity_id,
            index_kind,
            limit,
        )
        .await
    }

    pub async fn project_lane_message(
        &self,
        row: NewProjectedLaneMessage,
    ) -> PostgresStoreResult<()> {
        self.transaction(move |uow| Box::pin(async move { uow.project_lane_message(row).await }))
            .await
    }
    pub async fn get_projected_lane_message(
        &self,
        id: &str,
    ) -> PostgresStoreResult<Option<ProjectedLaneMessage>> {
        projected_lane_message(&**self.client().await?, &self.tables, id).await
    }
    pub async fn list_projected_lane_messages(
        &self,
        filter: LaneMessageFilter,
    ) -> PostgresStoreResult<Vec<ProjectedLaneMessage>> {
        list_projected_lane_messages(&**self.client().await?, &self.tables, &filter).await
    }
    pub async fn update_projected_lane_message_status(
        &self,
        id: &str,
        status: &str,
        error: Option<String>,
    ) -> PostgresStoreResult<()> {
        let id = id.to_owned();
        let status = status.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.update_projected_lane_message_status(&id, &status, error)
                    .await
            })
        })
        .await
    }
    pub async fn update_projected_lane_message_links(
        &self,
        id: &str,
        prev: Option<String>,
        next: Option<String>,
        inbox_tail: Option<String>,
        conversation_tail: Option<String>,
    ) -> PostgresStoreResult<()> {
        let id = id.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.update_projected_lane_message_links(
                    &id,
                    prev,
                    next,
                    inbox_tail,
                    conversation_tail,
                )
                .await
            })
        })
        .await
    }
    pub async fn clear_projected_lane_messages(&self, namespace: &str) -> PostgresStoreResult<u64> {
        let namespace = namespace.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move { uow.clear_projected_lane_messages(&namespace).await })
        })
        .await
    }
    pub async fn claim_projected_lane_messages(
        &self,
        namespace: &str,
        inbox: &str,
        owner: &str,
        limit: usize,
        lease: i64,
    ) -> PostgresStoreResult<Vec<ProjectedLaneMessage>> {
        let namespace = namespace.to_owned();
        let inbox = inbox.to_owned();
        let owner = owner.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.claim_projected_lane_messages(&namespace, &inbox, &owner, limit, lease)
                    .await
            })
        })
        .await
    }
    pub async fn ack_projected_lane_message(
        &self,
        id: &str,
        owner: &str,
    ) -> PostgresStoreResult<()> {
        let id = id.to_owned();
        let owner = owner.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move { uow.ack_projected_lane_message(&id, &owner).await })
        })
        .await
    }
    pub async fn requeue_projected_lane_message(
        &self,
        id: &str,
        owner: &str,
        error: Option<String>,
        delay: i64,
    ) -> PostgresStoreResult<()> {
        let id = id.to_owned();
        let owner = owner.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.requeue_projected_lane_message(&id, &owner, error, delay)
                    .await
            })
        })
        .await
    }
    pub async fn dead_letter_projected_lane_message(
        &self,
        id: &str,
        owner: &str,
        error: Option<String>,
    ) -> PostgresStoreResult<()> {
        let id = id.to_owned();
        let owner = owner.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.dead_letter_projected_lane_message(&id, &owner, error)
                    .await
            })
        })
        .await
    }
    pub async fn repair_orphaned_claimed_lane_messages(
        &self,
        namespace: &str,
        inbox_id: Option<&str>,
        limit: usize,
    ) -> PostgresStoreResult<Vec<String>> {
        let namespace = namespace.to_owned();
        let inbox_id = inbox_id.map(str::to_owned);
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.repair_orphaned_claimed_lane_messages(&namespace, inbox_id.as_deref(), limit)
                    .await
            })
        })
        .await
    }

    /// Strict trait-compatible advance: monotonic and at most retained latest.
    pub async fn strict_advance_replay_cursor(
        &self,
        namespace: &str,
        consumer: &str,
        last_seq: i64,
    ) -> PostgresStoreResult<ReplayCursor> {
        require_namespace(namespace)?;
        let namespace = namespace.to_owned();
        let consumer = consumer.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.strict_advance_replay_cursor(&namespace, &consumer, last_seq)
                    .await
            })
        })
        .await
    }

    /// Python legacy `cursor_set`: overwrite without monotonic/range checks.
    pub async fn set_replay_cursor_legacy(
        &self,
        namespace: &str,
        consumer: &str,
        last_seq: i64,
    ) -> PostgresStoreResult<ReplayCursor> {
        require_namespace(namespace)?;
        let namespace = namespace.to_owned();
        let consumer = consumer.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.set_replay_cursor_legacy(&namespace, &consumer, last_seq)
                    .await
            })
        })
        .await
    }

    async fn client(&self) -> PostgresStoreResult<deadpool_postgres::Object> {
        self.pool.get().await.map_err(backend)
    }
}

/// Operations sharing one RAII PostgreSQL transaction.
pub struct PostgresUnitOfWork<'transaction> {
    transaction: Transaction<'transaction>,
    tables: Tables,
}

impl PostgresUnitOfWork<'_> {
    /// Serialize runtime queue admission checks within this database schema.
    /// The lock is transaction-scoped and released on commit or rollback.
    pub async fn lock_runtime_queue_admission(&self) -> PostgresStoreResult<()> {
        advisory_lock(
            &self.transaction,
            &format!(
                "runtime-queue-admission\u{1f}{}",
                self.tables.projected_lane_messages
            ),
            73,
        )
        .await
    }

    pub async fn apply_graph_mutation(
        &mut self,
        mutation: GraphMutation,
    ) -> PostgresStoreResult<AppliedGraphMutation> {
        apply_graph_mutation(&self.transaction, &self.tables, mutation).await
    }

    pub async fn apply_graph_metadata_patch_mutation(
        &mut self,
        mutation: GraphMetadataPatchMutation,
    ) -> PostgresStoreResult<Option<AppliedGraphMutation>> {
        apply_graph_metadata_patch_mutation(&self.transaction, &self.tables, mutation).await
    }

    pub async fn apply_graph_delete_mutation(
        &mut self,
        mutation: GraphDeleteMutation,
    ) -> PostgresStoreResult<Option<AppliedGraphMutation>> {
        apply_graph_delete_mutation(&self.transaction, &self.tables, mutation).await
    }

    pub async fn upsert_graph_projection(
        &mut self,
        write: GraphProjectionUpsert,
    ) -> PostgresStoreResult<()> {
        upsert_graph_projection(&self.transaction, &self.tables, write).await
    }

    pub async fn patch_graph_projection_metadata(
        &mut self,
        patch: GraphProjectionMetadataPatch,
    ) -> PostgresStoreResult<bool> {
        patch_graph_projection_metadata(&self.transaction, &self.tables, patch).await
    }

    pub async fn graph_projection_records(
        &self,
        read: GraphProjectionRead,
    ) -> PostgresStoreResult<Vec<GraphRecord>> {
        graph_projection_records(&self.transaction, &self.tables, read).await
    }

    pub async fn graph_projection_vector_query(
        &self,
        query: GraphProjectionVectorQuery,
    ) -> PostgresStoreResult<Vec<VectorMatch>> {
        graph_projection_vector_query(&self.transaction, &self.tables, query).await
    }

    pub async fn project_lane_message(
        &mut self,
        row: NewProjectedLaneMessage,
    ) -> PostgresStoreResult<()> {
        project_lane_message(&self.transaction, &self.tables, row).await
    }
    pub async fn get_projected_lane_message(
        &self,
        id: &str,
    ) -> PostgresStoreResult<Option<ProjectedLaneMessage>> {
        projected_lane_message(&self.transaction, &self.tables, id).await
    }
    pub async fn get_server_run(&self, run_id: &str) -> PostgresStoreResult<Option<ServerRun>> {
        server_run(&self.transaction, &self.tables, run_id).await
    }
    pub async fn list_projected_lane_messages(
        &self,
        filter: &LaneMessageFilter,
    ) -> PostgresStoreResult<Vec<ProjectedLaneMessage>> {
        list_projected_lane_messages(&self.transaction, &self.tables, filter).await
    }
    pub async fn update_projected_lane_message_status(
        &mut self,
        id: &str,
        status: &str,
        error: Option<String>,
    ) -> PostgresStoreResult<()> {
        update_projected_lane_message_status(&self.transaction, &self.tables, id, status, error)
            .await
    }
    pub async fn update_projected_lane_message_payload(
        &mut self,
        id: &str,
        payload_json: String,
    ) -> PostgresStoreResult<()> {
        let changed = self
            .transaction
            .execute(
                &format!(
                    "UPDATE {} SET payload_json=$1 WHERE message_id=$2",
                    self.tables.projected_lane_messages
                ),
                &[&payload_json, &id],
            )
            .await
            .map_err(backend)?;
        if changed != 1 {
            return Err(PostgresStoreError::RecordedRuntimeConflict(format!(
                "lane message {id:?} payload update matched {changed} rows"
            )));
        }
        Ok(())
    }
    pub async fn update_projected_lane_message_links(
        &mut self,
        id: &str,
        prev: Option<String>,
        next: Option<String>,
        inbox_tail: Option<String>,
        conversation_tail: Option<String>,
    ) -> PostgresStoreResult<()> {
        update_projected_lane_message_links(
            &self.transaction,
            &self.tables,
            id,
            prev,
            next,
            inbox_tail,
            conversation_tail,
        )
        .await
    }
    pub async fn clear_projected_lane_messages(
        &mut self,
        namespace: &str,
    ) -> PostgresStoreResult<u64> {
        clear_projected_lane_messages(&self.transaction, &self.tables, namespace).await
    }
    pub async fn claim_projected_lane_messages(
        &mut self,
        namespace: &str,
        inbox: &str,
        owner: &str,
        limit: usize,
        lease: i64,
    ) -> PostgresStoreResult<Vec<ProjectedLaneMessage>> {
        claim_projected_lane_messages(
            &self.transaction,
            &self.tables,
            namespace,
            inbox,
            owner,
            limit,
            lease,
        )
        .await
    }
    pub async fn claim_projected_lane_messages_for_run(
        &mut self,
        namespace: &str,
        inbox: &str,
        run_id: &str,
        owner: &str,
        limit: usize,
        lease: i64,
    ) -> PostgresStoreResult<Vec<ProjectedLaneMessage>> {
        claim_projected_lane_messages_for_run(
            &self.transaction,
            &self.tables,
            namespace,
            inbox,
            run_id,
            owner,
            limit,
            lease,
        )
        .await
    }
    pub async fn ack_projected_lane_message(
        &mut self,
        id: &str,
        owner: &str,
    ) -> PostgresStoreResult<()> {
        ack_projected_lane_message(&self.transaction, &self.tables, id, owner).await
    }
    pub async fn requeue_projected_lane_message(
        &mut self,
        id: &str,
        owner: &str,
        error: Option<String>,
        delay: i64,
    ) -> PostgresStoreResult<()> {
        requeue_projected_lane_message(&self.transaction, &self.tables, id, owner, error, delay)
            .await
    }
    pub async fn dead_letter_projected_lane_message(
        &mut self,
        id: &str,
        owner: &str,
        error: Option<String>,
    ) -> PostgresStoreResult<()> {
        dead_letter_projected_lane_message(&self.transaction, &self.tables, id, owner, error).await
    }
    pub async fn repair_orphaned_claimed_lane_messages(
        &mut self,
        namespace: &str,
        inbox_id: Option<&str>,
        limit: usize,
    ) -> PostgresStoreResult<Vec<String>> {
        repair_orphaned_claimed_lane_messages(
            &self.transaction,
            &self.tables,
            namespace,
            inbox_id,
            limit,
        )
        .await
    }
    pub async fn enqueue_index_job(&mut self, job: NewIndexJob) -> PostgresStoreResult<String> {
        require_namespace(&job.namespace)?;
        advisory_lock(
            &self.transaction,
            &format!(
                "index-job\u{1f}{}\u{1f}{}",
                job.namespace,
                job.coalesce_key()
            ),
            2,
        )
        .await?;
        enqueue_index_job(&self.transaction, &self.tables, job).await
    }
    pub async fn claim_index_jobs(
        &mut self,
        limit: usize,
        lease_seconds: i64,
        namespace: Option<&str>,
    ) -> PostgresStoreResult<Vec<IndexJob>> {
        claim_index_jobs(
            &self.transaction,
            &self.tables,
            limit,
            lease_seconds,
            namespace,
        )
        .await
    }
    pub async fn mark_index_job_done(
        &mut self,
        job_id: &str,
        claim_token: Option<&str>,
    ) -> PostgresStoreResult<bool> {
        mark_index_job_done(&self.transaction, &self.tables, job_id, claim_token).await
    }
    pub async fn accept_index_job_result(
        &mut self,
        job_id: &str,
        claim_token: &str,
        result_json: &str,
        result_sha256: &str,
    ) -> PostgresStoreResult<AcceptedIndexJobResult> {
        accept_index_job_result(&self.transaction, &self.tables, job_id, claim_token, result_json, result_sha256).await
    }
    pub async fn mark_index_job_failed(
        &mut self,
        job_id: &str,
        error: &str,
        final_: bool,
        claim_token: Option<&str>,
    ) -> PostgresStoreResult<()> {
        mark_index_job_failed(
            &self.transaction,
            &self.tables,
            job_id,
            error,
            final_,
            claim_token,
        )
        .await
    }
    pub async fn bump_retry_and_requeue(
        &mut self,
        job_id: &str,
        error: &str,
        delay: i64,
        claim_token: Option<&str>,
    ) -> PostgresStoreResult<()> {
        bump_retry_and_requeue(
            &self.transaction,
            &self.tables,
            job_id,
            error,
            delay,
            claim_token,
        )
        .await
    }
    pub async fn renew_index_job_lease(
        &mut self,
        job_id: &str,
        claim_token: &str,
        lease_seconds: i64,
    ) -> PostgresStoreResult<bool> {
        renew_index_job_lease(
            &self.transaction,
            &self.tables,
            job_id,
            claim_token,
            lease_seconds,
        )
        .await
    }
    pub async fn requeue_index_job_at_tail(
        &mut self,
        job_id: &str,
        payload_json: String,
        delay: i64,
        claim_token: Option<&str>,
    ) -> PostgresStoreResult<()> {
        requeue_index_job_at_tail(
            &self.transaction,
            &self.tables,
            job_id,
            payload_json,
            delay,
            claim_token,
        )
        .await
    }
    pub async fn alloc_event_seq(&mut self, namespace: &str) -> PostgresStoreResult<i64> {
        require_namespace(namespace)?;
        allocate_event_seq(&self.transaction, &self.tables, namespace).await
    }

    pub async fn next_global_seq(&mut self) -> PostgresStoreResult<i64> {
        next_global_seq(&self.transaction, &self.tables).await
    }

    pub async fn current_global_seq(&self) -> PostgresStoreResult<i64> {
        current_global_seq(&self.transaction, &self.tables).await
    }

    pub async fn next_user_seq(&mut self, user_id: &str) -> PostgresStoreResult<i64> {
        next_user_seq(&self.transaction, &self.tables, user_id).await
    }

    pub async fn current_user_seq(&self, user_id: &str) -> PostgresStoreResult<i64> {
        current_user_seq(&self.transaction, &self.tables, user_id).await
    }

    pub async fn set_user_seq(&mut self, user_id: &str, value: i64) -> PostgresStoreResult<()> {
        set_user_seq(&self.transaction, &self.tables, user_id, value).await
    }

    pub async fn get_index_applied_fingerprint(
        &self,
        namespace: &str,
        coalesce_key: &str,
    ) -> PostgresStoreResult<Option<String>> {
        get_index_applied_fingerprint(&self.transaction, &self.tables, namespace, coalesce_key)
            .await
    }

    pub async fn set_index_applied_fingerprint(
        &mut self,
        namespace: &str,
        coalesce_key: &str,
        applied_fingerprint: Option<String>,
        last_job_id: Option<String>,
    ) -> PostgresStoreResult<()> {
        set_index_applied_fingerprint(
            &self.transaction,
            &self.tables,
            namespace,
            coalesce_key,
            applied_fingerprint,
            last_job_id,
        )
        .await
    }

    pub async fn append_raw_entity_event(
        &mut self,
        namespace: &str,
        event: NewRawEntityEvent,
    ) -> PostgresStoreResult<AppendedRawEvent> {
        require_namespace(namespace)?;
        append_raw_entity_event(&self.transaction, &self.tables, namespace, event).await
    }

    pub async fn replay_raw_events(
        &self,
        namespace: &str,
        after_seq: i64,
        limit: usize,
    ) -> PostgresStoreResult<Vec<RawEntityEvent>> {
        require_namespace(namespace)?;
        replay_raw_events(&self.transaction, &self.tables, namespace, after_seq, limit).await
    }

    pub async fn latest_retained_event_seq(&self, namespace: &str) -> PostgresStoreResult<i64> {
        require_namespace(namespace)?;
        latest_retained_event_seq(&self.transaction, &self.tables, namespace).await
    }

    pub async fn prune_entity_events_after(
        &mut self,
        namespace: &str,
        to_seq: i64,
    ) -> PostgresStoreResult<u64> {
        require_namespace(namespace)?;
        prune_entity_events_after(&self.transaction, &self.tables, namespace, to_seq).await
    }

    pub async fn replay_cursor(
        &self,
        namespace: &str,
        consumer: &str,
    ) -> PostgresStoreResult<ReplayCursor> {
        require_namespace(namespace)?;
        replay_cursor(&self.transaction, &self.tables, namespace, consumer).await
    }

    pub async fn recover_entity_projection(
        &mut self,
        request: &EntityRecoveryRequest,
    ) -> PostgresStoreResult<EntityRecoveryReport> {
        recover_entity_projection_uow(self, request).await
    }

    pub async fn rebuild_entity_projection(
        &mut self,
        request: &EntityRebuildRequest,
    ) -> PostgresStoreResult<EntityRecoveryReport> {
        rebuild_entity_projection_uow(self, request).await
    }

    pub async fn strict_advance_replay_cursor(
        &mut self,
        namespace: &str,
        consumer: &str,
        last_seq: i64,
    ) -> PostgresStoreResult<ReplayCursor> {
        require_namespace(namespace)?;
        advisory_lock(&self.transaction, &cursor_lock_key(namespace, consumer), 1).await?;
        let latest = latest_retained_event_seq(&self.transaction, &self.tables, namespace).await?;
        if last_seq > latest {
            return Err(StoreError::CursorOutOfRange {
                cursor: last_seq,
                latest,
            }
            .into());
        }
        let current = replay_cursor(&self.transaction, &self.tables, namespace, consumer)
            .await?
            .last_seq;
        if last_seq < current {
            return Err(StoreError::CursorRegresses {
                current,
                requested: last_seq,
            }
            .into());
        }
        set_replay_cursor_legacy(
            &self.transaction,
            &self.tables,
            namespace,
            consumer,
            last_seq,
        )
        .await
    }

    pub async fn set_replay_cursor_legacy(
        &mut self,
        namespace: &str,
        consumer: &str,
        last_seq: i64,
    ) -> PostgresStoreResult<ReplayCursor> {
        require_namespace(namespace)?;
        set_replay_cursor_legacy(
            &self.transaction,
            &self.tables,
            namespace,
            consumer,
            last_seq,
        )
        .await
    }

    pub async fn replace_named_projection(
        &mut self,
        namespace: &str,
        key: &str,
        projection: NamedProjectionWrite,
    ) -> PostgresStoreResult<()> {
        replace_named_projection(&self.transaction, &self.tables, namespace, key, projection).await
    }

    pub async fn get_named_projection(
        &mut self,
        namespace: &str,
        key: &str,
    ) -> PostgresStoreResult<Option<NamedProjection>> {
        named_projection(&self.transaction, &self.tables, namespace, key).await
    }

    pub async fn lock_named_projection(
        &mut self,
        namespace: &str,
        key: &str,
    ) -> PostgresStoreResult<Option<NamedProjection>> {
        named_projection_for_update(&self.transaction, &self.tables, namespace, key).await
    }

    pub async fn compare_and_swap_named_projection(
        &mut self,
        namespace: &str,
        key: &str,
        expected_last_authoritative_seq: Option<i64>,
        expected_last_materialized_seq: Option<i64>,
        projection: NamedProjectionWrite,
    ) -> PostgresStoreResult<bool> {
        compare_and_swap_named_projection(
            &self.transaction,
            &self.tables,
            namespace,
            key,
            expected_last_authoritative_seq,
            expected_last_materialized_seq,
            projection,
        )
        .await
    }

    pub async fn clear_named_projection(
        &mut self,
        namespace: &str,
        key: &str,
    ) -> PostgresStoreResult<()> {
        clear_named_projection(&self.transaction, &self.tables, namespace, key).await
    }

    pub async fn clear_projection_namespace(&mut self, namespace: &str) -> PostgresStoreResult<()> {
        clear_projection_namespace(&self.transaction, &self.tables, namespace).await
    }

    pub async fn put_workflow_design_snapshot(
        &mut self,
        workflow_id: &str,
        snapshot: WorkflowDesignSnapshotWrite,
    ) -> PostgresStoreResult<()> {
        put_workflow_design_snapshot(&self.transaction, &self.tables, workflow_id, snapshot).await
    }

    pub async fn get_workflow_design_snapshot(
        &self,
        workflow_id: &str,
        max_version: i64,
        schema_version: i64,
    ) -> PostgresStoreResult<Option<WorkflowDesignSnapshot>> {
        workflow_design_snapshot(
            &self.transaction,
            &self.tables,
            workflow_id,
            max_version,
            schema_version,
        )
        .await
    }

    pub async fn clear_workflow_design_snapshots(
        &mut self,
        workflow_id: &str,
    ) -> PostgresStoreResult<()> {
        clear_workflow_design_snapshots(&self.transaction, &self.tables, workflow_id).await
    }

    pub async fn put_workflow_design_delta(
        &mut self,
        workflow_id: &str,
        delta: WorkflowDesignDeltaWrite,
    ) -> PostgresStoreResult<()> {
        put_workflow_design_delta(&self.transaction, &self.tables, workflow_id, delta).await
    }

    pub async fn get_workflow_design_delta(
        &self,
        workflow_id: &str,
        version: i64,
        schema_version: i64,
    ) -> PostgresStoreResult<Option<WorkflowDesignDelta>> {
        workflow_design_delta(
            &self.transaction,
            &self.tables,
            workflow_id,
            version,
            schema_version,
        )
        .await
    }

    pub async fn clear_workflow_design_deltas(
        &mut self,
        workflow_id: &str,
    ) -> PostgresStoreResult<()> {
        clear_workflow_design_deltas(&self.transaction, &self.tables, workflow_id).await
    }

    pub async fn create_server_run(&mut self, run: ServerRunCreate) -> PostgresStoreResult<()> {
        create_server_run(&self.transaction, &self.tables, run).await
    }
    pub async fn append_server_run_event(
        &mut self,
        run_id: &str,
        event_type: &str,
        payload_json: String,
    ) -> PostgresStoreResult<ServerRunEvent> {
        append_server_run_event(
            &self.transaction,
            &self.tables,
            run_id,
            event_type,
            payload_json,
        )
        .await
    }
    pub async fn update_server_run(
        &mut self,
        run_id: &str,
        update: ServerRunUpdate,
    ) -> PostgresStoreResult<()> {
        update_server_run(&self.transaction, &self.tables, run_id, update).await
    }
    pub async fn request_server_run_cancel(&mut self, run_id: &str) -> PostgresStoreResult<()> {
        request_server_run_cancel(&self.transaction, &self.tables, run_id).await
    }

    pub async fn apply_recorded_runtime_transition(
        &mut self,
        transition: RecordedRuntimeTransition,
        abort_after_writes: bool,
    ) -> PostgresStoreResult<RecordedTransitionResult> {
        apply_recorded_runtime_transition(
            &self.transaction,
            &self.tables,
            transition,
            abort_after_writes,
        )
        .await
    }

    pub async fn apply_claimed_recorded_runtime_transition(
        &mut self,
        handoff: RecordedWorkerHandoff,
        transition: RecordedRuntimeTransition,
        abort_after_writes: bool,
    ) -> PostgresStoreResult<RecordedTransitionResult> {
        apply_claimed_recorded_runtime_transition(
            &self.transaction,
            &self.tables,
            handoff,
            transition,
            abort_after_writes,
        )
        .await
    }

    pub async fn apply_claimed_recorded_worker_effect(
        &mut self,
        handoff: RecordedWorkerHandoff,
        effect: RecordedWorkerSuccessEffect,
    ) -> PostgresStoreResult<RecordedTransitionResult> {
        apply_claimed_recorded_worker_effect(&self.transaction, &self.tables, handoff, effect).await
    }
}

impl ServerRunReadStore for PostgresStore {
    async fn server_run(&self, run_id: &str) -> StoreResult<Option<ServerRun>> {
        PostgresStore::get_server_run(self, run_id)
            .await
            .map_err(trait_error)
    }

    async fn server_runs(
        &self,
        status: Option<&str>,
        workflow_id: Option<&str>,
        conversation_id: Option<&str>,
        limit: usize,
    ) -> StoreResult<Vec<ServerRun>> {
        PostgresStore::list_server_runs(self, status, workflow_id, conversation_id, limit)
            .await
            .map_err(trait_error)
    }

    async fn server_run_events(
        &self,
        run_id: &str,
        after_seq: i64,
        limit: usize,
    ) -> StoreResult<Vec<ServerRunEvent>> {
        PostgresStore::list_server_run_events(self, run_id, after_seq, limit)
            .await
            .map_err(trait_error)
    }
}

impl ServerRunWriteStore for PostgresStore {
    async fn create_server_run(&self, run: ServerRunCreate) -> StoreResult<()> {
        PostgresStore::create_server_run(self, run)
            .await
            .map_err(trait_error)
    }

    async fn append_server_run_event(
        &self,
        run_id: &str,
        event_type: &str,
        payload_json: String,
    ) -> StoreResult<ServerRunEvent> {
        PostgresStore::append_server_run_event(self, run_id, event_type, payload_json)
            .await
            .map_err(trait_error)
    }

    async fn update_server_run(&self, run_id: &str, update: ServerRunUpdate) -> StoreResult<()> {
        PostgresStore::update_server_run(self, run_id, update)
            .await
            .map_err(trait_error)
    }

    async fn request_server_run_cancel(&self, run_id: &str) -> StoreResult<()> {
        PostgresStore::request_server_run_cancel(self, run_id)
            .await
            .map_err(trait_error)
    }
}

impl IndexJobReadStore for PostgresStore {
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
            .await
            .map_err(trait_error)
    }
}
impl IndexJobWriteStore for PostgresStore {
    async fn enqueue_index_job(&self, job: NewIndexJob) -> StoreResult<String> {
        PostgresStore::enqueue_index_job(self, job)
            .await
            .map_err(trait_error)
    }
    async fn claim_index_jobs(
        &self,
        limit: usize,
        lease_seconds: i64,
        namespace: Option<&str>,
    ) -> StoreResult<Vec<IndexJob>> {
        PostgresStore::claim_index_jobs(self, limit, lease_seconds, namespace)
            .await
            .map_err(trait_error)
    }
    async fn mark_index_job_done(
        &self,
        job_id: &str,
        claim_token: Option<&str>,
    ) -> StoreResult<bool> {
        PostgresStore::mark_index_job_done(self, job_id, claim_token)
            .await
            .map_err(trait_error)
    }
    async fn accept_index_job_result(
        &self,
        job_id: &str,
        claim_token: &str,
        result_json: &str,
        result_sha256: &str,
    ) -> StoreResult<AcceptedIndexJobResult> {
        PostgresStore::accept_index_job_result(self, job_id, claim_token, result_json, result_sha256)
            .await
            .map_err(trait_error)
    }
    async fn index_job_result(&self, job_id: &str) -> StoreResult<Option<AcceptedIndexJobResult>> {
        PostgresStore::index_job_result(self, job_id).await.map_err(trait_error)
    }
    async fn mark_index_job_failed(
        &self,
        job_id: &str,
        error: &str,
        final_: bool,
        claim_token: Option<&str>,
    ) -> StoreResult<()> {
        PostgresStore::mark_index_job_failed(self, job_id, error, final_, claim_token)
            .await
            .map_err(trait_error)
    }
    async fn bump_retry_and_requeue(
        &self,
        job_id: &str,
        error: &str,
        delay: i64,
        claim_token: Option<&str>,
    ) -> StoreResult<()> {
        PostgresStore::bump_retry_and_requeue(self, job_id, error, delay, claim_token)
            .await
            .map_err(trait_error)
    }
    async fn renew_index_job_lease(
        &self,
        job_id: &str,
        claim_token: &str,
        lease_seconds: i64,
    ) -> StoreResult<bool> {
        PostgresStore::renew_index_job_lease(self, job_id, claim_token, lease_seconds)
            .await
            .map_err(trait_error)
    }
    async fn requeue_index_job_at_tail(
        &self,
        job_id: &str,
        payload_json: String,
        delay: i64,
        claim_token: Option<&str>,
    ) -> StoreResult<()> {
        PostgresStore::requeue_index_job_at_tail(self, job_id, payload_json, delay, claim_token)
            .await
            .map_err(trait_error)
    }
}

impl LaneMessageReadStore for PostgresStore {
    async fn projected_lane_message(&self, id: &str) -> StoreResult<Option<ProjectedLaneMessage>> {
        self.get_projected_lane_message(id)
            .await
            .map_err(trait_error)
    }
    async fn projected_lane_messages(
        &self,
        filter: LaneMessageFilter,
    ) -> StoreResult<Vec<ProjectedLaneMessage>> {
        self.list_projected_lane_messages(filter)
            .await
            .map_err(trait_error)
    }
}
impl LaneMessageWriteStore for PostgresStore {
    async fn project_lane_message(&self, row: NewProjectedLaneMessage) -> StoreResult<()> {
        PostgresStore::project_lane_message(self, row)
            .await
            .map_err(trait_error)
    }
    async fn update_projected_lane_message_status(
        &self,
        id: &str,
        status: &str,
        error: Option<String>,
    ) -> StoreResult<()> {
        PostgresStore::update_projected_lane_message_status(self, id, status, error)
            .await
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
        PostgresStore::update_projected_lane_message_links(
            self,
            id,
            prev,
            next,
            inbox_tail,
            conversation_tail,
        )
        .await
        .map_err(trait_error)
    }
    async fn clear_projected_lane_messages(&self, namespace: &str) -> StoreResult<u64> {
        PostgresStore::clear_projected_lane_messages(self, namespace)
            .await
            .map_err(trait_error)
    }
    async fn claim_projected_lane_messages(
        &self,
        namespace: &str,
        inbox: &str,
        owner: &str,
        limit: usize,
        lease: i64,
    ) -> StoreResult<Vec<ProjectedLaneMessage>> {
        PostgresStore::claim_projected_lane_messages(self, namespace, inbox, owner, limit, lease)
            .await
            .map_err(trait_error)
    }
    async fn ack_projected_lane_message(&self, id: &str, owner: &str) -> StoreResult<()> {
        PostgresStore::ack_projected_lane_message(self, id, owner)
            .await
            .map_err(trait_error)
    }
    async fn requeue_projected_lane_message(
        &self,
        id: &str,
        owner: &str,
        error: Option<String>,
        delay: i64,
    ) -> StoreResult<()> {
        PostgresStore::requeue_projected_lane_message(self, id, owner, error, delay)
            .await
            .map_err(trait_error)
    }
    async fn dead_letter_projected_lane_message(
        &self,
        id: &str,
        owner: &str,
        error: Option<String>,
    ) -> StoreResult<()> {
        PostgresStore::dead_letter_projected_lane_message(self, id, owner, error)
            .await
            .map_err(trait_error)
    }
}

impl EventReadStore for PostgresStore {
    async fn replay_events(
        &self,
        namespace: &str,
        after_seq: i64,
        limit: usize,
    ) -> StoreResult<Vec<EntityEvent>> {
        require_trait_namespace(namespace)?;
        self.replay_raw_events(namespace, after_seq, limit)
            .await
            .map_err(trait_error)?
            .into_iter()
            .map(raw_to_entity_event)
            .collect::<PostgresStoreResult<Vec<_>>>()
            .map_err(trait_error)
    }

    async fn replay_cursor(&self, namespace: &str, consumer: &str) -> StoreResult<ReplayCursor> {
        require_trait_namespace(namespace)?;
        PostgresStore::replay_cursor(self, namespace, consumer)
            .await
            .map_err(trait_error)
    }

    async fn latest_event_seq(&self, namespace: &str) -> StoreResult<i64> {
        require_trait_namespace(namespace)?;
        self.latest_retained_event_seq(namespace)
            .await
            .map_err(trait_error)
    }
}

impl EventWriteStore for PostgresStore {
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
            .await
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
            .await
            .map_err(trait_error)
    }
}

impl EventPruneStore for PostgresStore {
    async fn prune_entity_events_after(&self, namespace: &str, to_seq: i64) -> StoreResult<u64> {
        require_trait_namespace(namespace)?;
        PostgresStore::prune_entity_events_after(self, namespace, to_seq)
            .await
            .map_err(trait_error)
    }
}

impl GraphMutationStore for PostgresStore {
    async fn apply_graph_mutation(
        &self,
        mutation: GraphMutation,
    ) -> StoreResult<AppliedGraphMutation> {
        PostgresStore::apply_graph_mutation(self, mutation)
            .await
            .map_err(trait_error)
    }

    async fn graph_projection_records(
        &self,
        read: GraphProjectionRead,
    ) -> StoreResult<Vec<GraphRecord>> {
        PostgresStore::graph_projection_records(self, read)
            .await
            .map_err(trait_error)
    }

    async fn graph_projection_vector_query(
        &self,
        query: GraphProjectionVectorQuery,
    ) -> StoreResult<Vec<VectorMatch>> {
        PostgresStore::graph_projection_vector_query(self, query)
            .await
            .map_err(trait_error)
    }
}

impl ProjectionReadStore for PostgresStore {
    async fn named_projection(
        &self,
        namespace: &str,
        key: &str,
    ) -> StoreResult<Option<NamedProjection>> {
        require_trait_namespace(namespace)?;
        self.get_named_projection(namespace, key)
            .await
            .map_err(trait_error)
    }

    async fn named_projections(&self, namespace: &str) -> StoreResult<Vec<NamedProjection>> {
        require_trait_namespace(namespace)?;
        self.list_named_projections(namespace)
            .await
            .map_err(trait_error)
    }
}

impl ProjectionWriteStore for PostgresStore {
    async fn replace_named_projection(
        &self,
        namespace: &str,
        key: &str,
        projection: NamedProjectionWrite,
    ) -> StoreResult<()> {
        require_trait_namespace(namespace)?;
        PostgresStore::replace_named_projection(self, namespace, key, projection)
            .await
            .map_err(trait_error)
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
        PostgresStore::compare_and_swap_named_projection(
            self,
            namespace,
            key,
            expected_last_authoritative_seq,
            expected_last_materialized_seq,
            projection,
        )
        .await
        .map_err(trait_error)
    }

    async fn clear_named_projection(&self, namespace: &str, key: &str) -> StoreResult<()> {
        require_trait_namespace(namespace)?;
        PostgresStore::clear_named_projection(self, namespace, key)
            .await
            .map_err(trait_error)
    }

    async fn clear_projection_namespace(&self, namespace: &str) -> StoreResult<()> {
        require_trait_namespace(namespace)?;
        PostgresStore::clear_projection_namespace(self, namespace)
            .await
            .map_err(trait_error)
    }
}

impl WorkflowDesignHistoryReadStore for PostgresStore {
    async fn workflow_design_snapshot(
        &self,
        workflow_id: &str,
        max_version: i64,
        schema_version: i64,
    ) -> StoreResult<Option<WorkflowDesignSnapshot>> {
        PostgresStore::get_workflow_design_snapshot(self, workflow_id, max_version, schema_version)
            .await
            .map_err(trait_error)
    }

    async fn workflow_design_delta(
        &self,
        workflow_id: &str,
        version: i64,
        schema_version: i64,
    ) -> StoreResult<Option<WorkflowDesignDelta>> {
        PostgresStore::get_workflow_design_delta(self, workflow_id, version, schema_version)
            .await
            .map_err(trait_error)
    }
}

impl WorkflowDesignHistoryWriteStore for PostgresStore {
    async fn put_workflow_design_snapshot(
        &self,
        workflow_id: &str,
        snapshot: WorkflowDesignSnapshotWrite,
    ) -> StoreResult<()> {
        PostgresStore::put_workflow_design_snapshot(self, workflow_id, snapshot)
            .await
            .map_err(trait_error)
    }

    async fn clear_workflow_design_snapshots(&self, workflow_id: &str) -> StoreResult<()> {
        PostgresStore::clear_workflow_design_snapshots(self, workflow_id)
            .await
            .map_err(trait_error)
    }

    async fn put_workflow_design_delta(
        &self,
        workflow_id: &str,
        delta: WorkflowDesignDeltaWrite,
    ) -> StoreResult<()> {
        PostgresStore::put_workflow_design_delta(self, workflow_id, delta)
            .await
            .map_err(trait_error)
    }

    async fn clear_workflow_design_deltas(&self, workflow_id: &str) -> StoreResult<()> {
        PostgresStore::clear_workflow_design_deltas(self, workflow_id)
            .await
            .map_err(trait_error)
    }
}

async fn allocate_event_seq<C>(
    client: &C,
    tables: &Tables,
    namespace: &str,
) -> PostgresStoreResult<i64>
where
    C: GenericClient + Sync,
{
    let row = client
        .query_one(&allocation_sql(tables), &[&namespace])
        .await
        .map_err(backend)?;
    row.try_get(0).map_err(backend)
}

async fn next_global_seq<C>(client: &C, tables: &Tables) -> PostgresStoreResult<i64>
where
    C: GenericClient + Sync,
{
    let row = client
        .query_one(
            &format!(
                "UPDATE {} SET value=value+1 RETURNING value",
                tables.global_seq
            ),
            &[],
        )
        .await
        .map_err(backend)?;
    row.try_get(0).map_err(backend)
}

async fn current_global_seq<C>(client: &C, tables: &Tables) -> PostgresStoreResult<i64>
where
    C: GenericClient + Sync,
{
    client
        .query_opt(
            &format!("SELECT value FROM {} LIMIT 1", tables.global_seq),
            &[],
        )
        .await
        .map_err(backend)?
        .map(|row| row.try_get(0).map_err(backend))
        .unwrap_or(Ok(0))
}

async fn next_user_seq<C>(client: &C, tables: &Tables, user_id: &str) -> PostgresStoreResult<i64>
where
    C: GenericClient + Sync,
{
    let row = client
        .query_one(
            &format!(
                "INSERT INTO {}(user_id,value) VALUES($1,1) ON CONFLICT(user_id) DO UPDATE SET value={}.value+1 RETURNING value",
                tables.user_seq, tables.user_seq
            ),
            &[&user_id],
        )
        .await
        .map_err(backend)?;
    row.try_get(0).map_err(backend)
}

async fn current_user_seq<C>(client: &C, tables: &Tables, user_id: &str) -> PostgresStoreResult<i64>
where
    C: GenericClient + Sync,
{
    client
        .query_opt(
            &format!("SELECT value FROM {} WHERE user_id=$1", tables.user_seq),
            &[&user_id],
        )
        .await
        .map_err(backend)?
        .map(|row| row.try_get(0).map_err(backend))
        .unwrap_or(Ok(0))
}

async fn set_user_seq<C>(
    client: &C,
    tables: &Tables,
    user_id: &str,
    value: i64,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    if value < 0 {
        return Err(PostgresStoreError::Backend(
            "sequence value must be non-negative".to_owned(),
        ));
    }
    client
        .execute(
            &format!(
                "INSERT INTO {}(user_id,value) VALUES($1,$2) ON CONFLICT(user_id) DO UPDATE SET value=EXCLUDED.value",
                tables.user_seq
            ),
            &[&user_id, &value],
        )
        .await
        .map_err(backend)?;
    Ok(())
}

async fn get_index_applied_fingerprint<C>(
    client: &C,
    tables: &Tables,
    namespace: &str,
    coalesce_key: &str,
) -> PostgresStoreResult<Option<String>>
where
    C: GenericClient + Sync,
{
    client
        .query_opt(
            &format!(
                "SELECT applied_fingerprint FROM {} WHERE namespace=$1 AND coalesce_key=$2",
                tables.index_applied_state
            ),
            &[&namespace, &coalesce_key],
        )
        .await
        .map_err(backend)?
        .map(|row| row.try_get(0).map_err(backend))
        .transpose()
}

async fn set_index_applied_fingerprint<C>(
    client: &C,
    tables: &Tables,
    namespace: &str,
    coalesce_key: &str,
    applied_fingerprint: Option<String>,
    last_job_id: Option<String>,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    client
        .execute(
            &format!(
                "INSERT INTO {}(namespace,coalesce_key,applied_fingerprint,applied_at,last_job_id) VALUES($1,$2,$3,NOW(),$4) ON CONFLICT(namespace,coalesce_key) DO UPDATE SET applied_fingerprint=EXCLUDED.applied_fingerprint,applied_at=NOW(),last_job_id=EXCLUDED.last_job_id",
                tables.index_applied_state
            ),
            &[&namespace, &coalesce_key, &applied_fingerprint, &last_job_id],
        )
        .await
        .map_err(backend)?;
    Ok(())
}

fn projection_payload_json(
    payload: &serde_json::Map<String, serde_json::Value>,
) -> PostgresStoreResult<String> {
    let compact = serde_json::to_string(&BTreeMap::<_, _>::from_iter(payload.iter()))
        .map_err(|error| PostgresStoreError::TransactionAborted(error.to_string()))?;
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

fn projection_from_row(row: &Row) -> PostgresStoreResult<NamedProjection> {
    let payload_json: String = row.try_get(2).map_err(backend)?;
    let payload = serde_json::from_str::<serde_json::Value>(&payload_json)
        .map_err(PostgresStoreError::InvalidPayload)?
        .as_object()
        .cloned()
        .ok_or_else(|| {
            PostgresStoreError::Backend(
                "named projection payload must deserialize to an object".to_owned(),
            )
        })?;
    Ok(NamedProjection {
        namespace: row.try_get(0).map_err(backend)?,
        key: row.try_get(1).map_err(backend)?,
        payload,
        last_authoritative_seq: row.try_get(3).map_err(backend)?,
        last_materialized_seq: row.try_get(4).map_err(backend)?,
        projection_schema_version: i64::from(row.try_get::<_, i32>(5).map_err(backend)?),
        materialization_status: row.try_get(6).map_err(backend)?,
        updated_at_ms: row.try_get(7).map_err(backend)?,
    })
}

async fn named_projection<C>(
    client: &C,
    tables: &Tables,
    namespace: &str,
    key: &str,
) -> PostgresStoreResult<Option<NamedProjection>>
where
    C: GenericClient + Sync,
{
    client.query_opt(
        &format!("SELECT namespace, key, payload_json, last_authoritative_seq, last_materialized_seq, projection_schema_version, materialization_status, updated_at_ms FROM {} WHERE namespace = $1 AND key = $2", tables.named_projections),
        &[&namespace, &key],
    ).await.map_err(backend)?.map(|row| projection_from_row(&row)).transpose()
}

async fn named_projection_for_update<C>(
    client: &C,
    tables: &Tables,
    namespace: &str,
    key: &str,
) -> PostgresStoreResult<Option<NamedProjection>>
where
    C: GenericClient + Sync,
{
    client.query_opt(
        &format!("SELECT namespace, key, payload_json, last_authoritative_seq, last_materialized_seq, projection_schema_version, materialization_status, updated_at_ms FROM {} WHERE namespace = $1 AND key = $2 FOR UPDATE", tables.named_projections),
        &[&namespace, &key],
    ).await.map_err(backend)?.map(|row| projection_from_row(&row)).transpose()
}

async fn named_projections<C>(
    client: &C,
    tables: &Tables,
    namespace: &str,
) -> PostgresStoreResult<Vec<NamedProjection>>
where
    C: GenericClient + Sync,
{
    client.query(
        &format!("SELECT namespace, key, payload_json, last_authoritative_seq, last_materialized_seq, projection_schema_version, materialization_status, updated_at_ms FROM {} WHERE namespace = $1 ORDER BY key ASC", tables.named_projections),
        &[&namespace],
    ).await.map_err(backend)?.iter().map(projection_from_row).collect()
}

async fn replace_named_projection<C>(
    client: &C,
    tables: &Tables,
    namespace: &str,
    key: &str,
    projection: NamedProjectionWrite,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    let payload_json = projection_payload_json(&projection.payload)?;
    let updated_at_ms = unix_epoch_millis();
    let projection_schema_version =
        i32::try_from(projection.projection_schema_version).map_err(|_| {
            PostgresStoreError::Backend(
                "projection schema version exceeds PostgreSQL INTEGER".to_owned(),
            )
        })?;
    client.execute(
        &format!("INSERT INTO {}(namespace, key, payload_json, last_authoritative_seq, last_materialized_seq, projection_schema_version, materialization_status, updated_at_ms) VALUES ($1,$2,$3,$4,$5,$6,$7,$8) ON CONFLICT(namespace,key) DO UPDATE SET payload_json=EXCLUDED.payload_json,last_authoritative_seq=EXCLUDED.last_authoritative_seq,last_materialized_seq=EXCLUDED.last_materialized_seq,projection_schema_version=EXCLUDED.projection_schema_version,materialization_status=EXCLUDED.materialization_status,updated_at_ms=EXCLUDED.updated_at_ms", tables.named_projections),
        &[&namespace, &key, &payload_json, &projection.last_authoritative_seq, &projection.last_materialized_seq, &projection_schema_version, &projection.materialization_status, &updated_at_ms],
    ).await.map_err(backend)?;
    Ok(())
}

async fn compare_and_swap_named_projection<C>(
    client: &C,
    tables: &Tables,
    namespace: &str,
    key: &str,
    expected_last_authoritative_seq: Option<i64>,
    expected_last_materialized_seq: Option<i64>,
    projection: NamedProjectionWrite,
) -> PostgresStoreResult<bool>
where
    C: GenericClient + Sync,
{
    let payload_json = projection_payload_json(&projection.payload)?;
    let updated_at_ms = unix_epoch_millis();
    let projection_schema_version =
        i32::try_from(projection.projection_schema_version).map_err(|_| {
            PostgresStoreError::Backend(
                "projection schema version exceeds PostgreSQL INTEGER".to_owned(),
            )
        })?;
    let changes = match (expected_last_authoritative_seq, expected_last_materialized_seq) {
        (None, None) => client.execute(
            &format!("INSERT INTO {}(namespace, key, payload_json, last_authoritative_seq, last_materialized_seq, projection_schema_version, materialization_status, updated_at_ms) VALUES ($1,$2,$3,$4,$5,$6,$7,$8) ON CONFLICT(namespace,key) DO NOTHING", tables.named_projections),
            &[&namespace, &key, &payload_json, &projection.last_authoritative_seq, &projection.last_materialized_seq, &projection_schema_version, &projection.materialization_status, &updated_at_ms],
        ).await.map_err(backend)?,
        (Some(expected_authoritative), Some(expected_materialized)) => client.execute(
            &format!("UPDATE {} SET payload_json=$1,last_authoritative_seq=$2,last_materialized_seq=$3,projection_schema_version=$4,materialization_status=$5,updated_at_ms=$6 WHERE namespace=$7 AND key=$8 AND last_authoritative_seq=$9 AND last_materialized_seq=$10", tables.named_projections),
            &[&payload_json, &projection.last_authoritative_seq, &projection.last_materialized_seq, &projection_schema_version, &projection.materialization_status, &updated_at_ms, &namespace, &key, &expected_authoritative, &expected_materialized],
        ).await.map_err(backend)?,
        _ => 0,
    };
    Ok(changes == 1)
}

async fn clear_named_projection<C>(
    client: &C,
    tables: &Tables,
    namespace: &str,
    key: &str,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    client
        .execute(
            &format!(
                "DELETE FROM {} WHERE namespace = $1 AND key = $2",
                tables.named_projections
            ),
            &[&namespace, &key],
        )
        .await
        .map_err(|error| PostgresStoreError::Backend(format!("{error:?}")))?;
    Ok(())
}

async fn clear_projection_namespace<C>(
    client: &C,
    tables: &Tables,
    namespace: &str,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    client
        .execute(
            &format!(
                "DELETE FROM {} WHERE namespace = $1",
                tables.named_projections
            ),
            &[&namespace],
        )
        .await
        .map_err(backend)?;
    Ok(())
}

fn unix_epoch_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock precedes Unix epoch")
        .as_millis() as i64
}

fn allocation_sql(tables: &Tables) -> String {
    format!(
        "INSERT INTO {} AS target(namespace, next_seq) VALUES ($1, 2) \
         ON CONFLICT(namespace) DO UPDATE SET next_seq = target.next_seq + 1 \
         RETURNING next_seq - 1",
        tables.namespace_seq
    )
}

async fn append_raw_entity_event<C>(
    client: &C,
    tables: &Tables,
    namespace: &str,
    event: NewRawEntityEvent,
) -> PostgresStoreResult<AppendedRawEvent>
where
    C: GenericClient + Sync,
{
    // A per-event advisory xact lock closes the absent-row unique-key race:
    // same-id retries observe the committed row before allocating a sequence.
    advisory_lock(client, &event.event_id, 0).await?;
    if let Some(existing) = event_by_id(client, tables, &event.event_id).await? {
        if existing.namespace != namespace {
            return Err(PostgresStoreError::EventIdNamespaceCollision {
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
    let seq = allocate_event_seq(client, tables, namespace).await?;
    let row = client
        .query_one(
            &format!(
                "INSERT INTO {}(namespace, seq, event_id, entity_kind, entity_id, op, payload_json) \
                 VALUES ($1, $2, $3, $4, $5, $6, $7) \
                 RETURNING namespace, seq, event_id, entity_kind, entity_id, op, payload_json, created_at::TEXT",
                tables.entity_events
            ),
            &[
                &namespace,
                &seq,
                &event.event_id,
                &event.entity_kind,
                &event.entity_id,
                &event.op,
                &event.payload_json,
            ],
        )
        .await
        .map_err(backend)?;
    Ok(AppendedRawEvent {
        event: raw_event_from_row(&row)?,
        inserted: true,
    })
}

fn graph_table(tables: &Tables, table: &str) -> PostgresStoreResult<String> {
    quote_identifier(table)?;
    qualified(&tables.quoted_schema, table)
}

async fn create_graph_schema<C>(
    client: &C,
    tables: &Tables,
    embedding_dim: usize,
    names: GraphTableNames,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    if embedding_dim == 0 {
        return Err(PostgresStoreError::Backend(
            "graph embedding dimension must be positive".to_owned(),
        ));
    }
    let dim = i32::try_from(embedding_dim).map_err(|_| {
        PostgresStoreError::Backend("graph embedding dimension exceeds INTEGER".to_owned())
    })?;
    let relations = [names.nodes, names.edges, names.documents, names.domains]
        .into_iter()
        .map(|name| qualified(&tables.quoted_schema, &name))
        .collect::<PostgresStoreResult<Vec<_>>>()?;
    // Relation/type/dimension are validated constants. Data remains bind-bound
    // everywhere possible; pgvector requires explicit `TEXT::vector` casts.
    let sql = format!(
        "{} \
         CREATE EXTENSION IF NOT EXISTS vector; \
         CREATE SCHEMA IF NOT EXISTS {}; \
         CREATE TABLE IF NOT EXISTS {} (id VARCHAR PRIMARY KEY, document TEXT NULL, metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb, embedding vector({}) NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()); \
         CREATE TABLE IF NOT EXISTS {} (id VARCHAR PRIMARY KEY, document TEXT NULL, metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb, embedding vector({}) NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()); \
         CREATE TABLE IF NOT EXISTS {} (id VARCHAR PRIMARY KEY, document TEXT NULL, metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb, embedding vector({}) NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()); \
         CREATE TABLE IF NOT EXISTS {} (id VARCHAR PRIMARY KEY, document TEXT NULL, metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb, embedding vector({}) NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW());",
        schema_sql(tables),
        tables.quoted_schema,
        relations[0],
        dim,
        relations[1],
        dim,
        relations[2],
        dim,
        relations[3],
        dim,
    );
    client.batch_execute(&sql).await.map_err(backend)
}

fn graph_scope_pairs(
    scope: &kogwistar_store::GraphScope,
) -> PostgresStoreResult<Vec<(String, Value)>> {
    require_namespace(&scope.namespace)?;
    let mut pairs = vec![(
        "namespace".to_owned(),
        Value::String(scope.namespace.clone()),
    )];
    if let Some(workspace_id) = &scope.workspace_id {
        pairs.push((
            "workspace_id".to_owned(),
            Value::String(workspace_id.clone()),
        ));
    }
    if let Some(graph_space) = &scope.graph_space {
        pairs.push(("graph_space".to_owned(), Value::String(graph_space.clone())));
    }
    Ok(pairs)
}

fn graph_metadata(
    scope: &kogwistar_store::GraphScope,
    metadata: &Map<String, Value>,
) -> PostgresStoreResult<Map<String, Value>> {
    let scope_pairs = graph_scope_pairs(scope)?;
    for (key, expected) in &scope_pairs {
        if let Some(actual) = metadata.get(key)
            && actual != expected
        {
            return Err(PostgresStoreError::Backend(format!(
                "graph metadata scope {key:?} conflicts with request scope"
            )));
        }
    }
    let mut materialized = metadata.clone();
    for (key, value) in scope_pairs {
        materialized.insert(key, value);
    }
    Ok(materialized)
}

fn graph_scope_matches_or_legacy_default(
    scope: &kogwistar_store::GraphScope,
    metadata: &Map<String, Value>,
) -> PostgresStoreResult<bool> {
    let expected = graph_metadata(scope, &Map::new())?;
    if expected
        .iter()
        .all(|(key, value)| metadata.get(key) == Some(value))
    {
        return Ok(true);
    }
    Ok(scope.namespace == "default"
        && scope.workspace_id.is_none()
        && scope.graph_space.is_none()
        && !metadata.contains_key("namespace")
        && !metadata.contains_key("workspace_id")
        && !metadata.contains_key("graph_space"))
}

fn finite_vector(vector: &[f32]) -> PostgresStoreResult<()> {
    if vector.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(StoreError::NonFiniteVector.into())
    }
}

fn vector_literal(vector: &[f32]) -> String {
    format!(
        "[{}]",
        vector
            .iter()
            .map(f32::to_string)
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn sql_text_literal(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

fn graph_record_from_row(row: &Row) -> PostgresStoreResult<GraphRecord> {
    let metadata_json: String = row.try_get(2).map_err(backend)?;
    let metadata: Map<String, Value> = serde_json::from_str(&metadata_json)?;
    let embedding_json: Option<String> = row.try_get(3).map_err(backend)?;
    let embedding = embedding_json
        .as_deref()
        .map(serde_json::from_str)
        .transpose()?;
    Ok(GraphRecord {
        id: row.try_get(0).map_err(backend)?,
        document: row.try_get(1).map_err(backend)?,
        metadata,
        embedding,
    })
}

fn graph_filter_sql(
    scope: &kogwistar_store::GraphScope,
    filter: &kogwistar_store::MetadataFilter,
    start: usize,
) -> PostgresStoreResult<(String, Vec<String>)> {
    let scope_pairs = graph_scope_pairs(scope)?;
    let legacy_default =
        scope.namespace == "default" && scope.workspace_id.is_none() && scope.graph_space.is_none();
    let filter_pairs = filter
        .equals
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()));
    let mut clauses = Vec::with_capacity(scope_pairs.len() + filter.equals.len());
    let mut values = Vec::with_capacity(scope_pairs.len() + filter.equals.len());
    for (index, (key, value)) in scope_pairs.into_iter().chain(filter_pairs).enumerate() {
        let key_index = start + index;
        let value_json = serde_json::to_string(&value).expect("JSON value serializes");
        let exact = format!(
            "metadata -> ${key_index}::TEXT = {}::JSONB",
            sql_text_literal(&value_json)
        );
        if legacy_default && index == 0 && key == "namespace" {
            clauses.push(format!(
                "({exact} OR (NOT (metadata ? 'namespace') AND NOT (metadata ? 'workspace_id') AND NOT (metadata ? 'graph_space')))"
            ));
        } else {
            clauses.push(exact);
        }
        values.push(key);
    }
    Ok((clauses.join(" AND "), values))
}

async fn apply_graph_mutation<C>(
    client: &C,
    tables: &Tables,
    mutation: GraphMutation,
) -> PostgresStoreResult<AppliedGraphMutation>
where
    C: GenericClient + Sync,
{
    require_namespace(&mutation.scope.namespace)?;
    if mutation.record.id.is_empty() {
        return Err(StoreError::EmptyRecordId.into());
    }
    if mutation.event_id.is_empty() {
        return Err(PostgresStoreError::Backend(
            "graph mutation event id must not be empty".to_owned(),
        ));
    }
    if !matches!(mutation.op.as_str(), "ADD" | "REPLACE" | "TOMBSTONE") {
        return Err(PostgresStoreError::Backend(
            "graph mutation op must be ADD, REPLACE, or TOMBSTONE".to_owned(),
        ));
    }
    let metadata = graph_metadata(&mutation.scope, &mutation.record.metadata)?;
    if let Some(embedding) = &mutation.record.embedding {
        finite_vector(embedding)?;
        if embedding.len() != mutation.embedding_dim {
            return Err(StoreError::VectorDimensionMismatch {
                expected: mutation.embedding_dim,
                actual: embedding.len(),
            }
            .into());
        }
    }
    let table = graph_table(tables, &mutation.table)?;
    let payload_json = serde_json::to_string(&mutation.payload)?;
    let appended = append_raw_entity_event(
        client,
        tables,
        &mutation.scope.namespace,
        NewRawEntityEvent {
            event_id: mutation.event_id,
            entity_kind: mutation.entity_kind,
            entity_id: mutation.record.id.clone(),
            op: mutation.op,
            payload_json,
        },
    )
    .await?;
    let event = raw_to_entity_event(appended.event)?;
    if !appended.inserted {
        return Ok(AppliedGraphMutation {
            event,
            inserted: false,
            mutated: false,
        });
    }

    // `id` is globally unique in the Python collection DDL.  Never let one
    // namespace/workspace silently overwrite another's graph row.
    if let Some(row) = client
        .query_opt(
            &format!("SELECT metadata::TEXT FROM {table} WHERE id = $1 FOR UPDATE"),
            &[&mutation.record.id],
        )
        .await
        .map_err(backend)?
    {
        let current_json: String = row.try_get(0).map_err(backend)?;
        let current: Map<String, Value> = serde_json::from_str(&current_json)?;
        if !graph_scope_matches_or_legacy_default(&mutation.scope, &current)? {
            return Err(PostgresStoreError::Backend(
                "graph record id belongs to a different scope".to_owned(),
            ));
        }
    }
    let metadata_json = serde_json::to_string(&metadata)?;
    let embedding_sql = mutation
        .record
        .embedding
        .as_deref()
        .map(|vector| format!("{}::vector", sql_text_literal(&vector_literal(vector))))
        .unwrap_or_else(|| "NULL".to_owned());
    client
        .execute(
            &format!(
                "INSERT INTO {table}(id, document, metadata, embedding) VALUES ($1, $2, {}::JSONB, {embedding_sql}) \
                 ON CONFLICT(id) DO UPDATE SET document = EXCLUDED.document, metadata = EXCLUDED.metadata, \
                 embedding = EXCLUDED.embedding, updated_at = NOW()",
                sql_text_literal(&metadata_json),
            ),
            &[&mutation.record.id, &mutation.record.document],
        )
        .await
        .map_err(|error| PostgresStoreError::Backend(format!("graph projection upsert failed: {error:?}")))?;
    Ok(AppliedGraphMutation {
        event,
        inserted: true,
        mutated: true,
    })
}

async fn apply_graph_metadata_patch_mutation<C>(
    client: &C,
    tables: &Tables,
    mutation: GraphMetadataPatchMutation,
) -> PostgresStoreResult<Option<AppliedGraphMutation>>
where
    C: GenericClient + Sync,
{
    require_namespace(&mutation.scope.namespace)?;
    if mutation.entity_id.is_empty() {
        return Err(StoreError::EmptyRecordId.into());
    }
    if mutation.event_id.is_empty() {
        return Err(PostgresStoreError::Backend(
            "graph metadata patch event id must not be empty".to_owned(),
        ));
    }
    if !matches!(mutation.op.as_str(), "REPLACE" | "TOMBSTONE") {
        return Err(PostgresStoreError::Backend(
            "graph metadata patch op must be REPLACE or TOMBSTONE".to_owned(),
        ));
    }
    let table = graph_table(tables, &mutation.table)?;
    let Some(row) = client
        .query_opt(
            &format!("SELECT document, metadata::TEXT FROM {table} WHERE id = $1 FOR UPDATE"),
            &[&mutation.entity_id],
        )
        .await
        .map_err(backend)?
    else {
        return Ok(None);
    };

    let document: Option<String> = row.try_get(0).map_err(backend)?;
    let current_metadata_json: String = row.try_get(1).map_err(backend)?;
    let mut current_metadata: Map<String, Value> = serde_json::from_str(&current_metadata_json)?;
    if !graph_scope_matches_or_legacy_default(&mutation.scope, &current_metadata)? {
        return Err(PostgresStoreError::Backend(
            "graph record id belongs to a different scope".to_owned(),
        ));
    }

    let payload_json = serde_json::to_string(&mutation.payload)?;
    let appended = append_raw_entity_event(
        client,
        tables,
        &mutation.scope.namespace,
        NewRawEntityEvent {
            event_id: mutation.event_id,
            entity_kind: mutation.entity_kind,
            entity_id: mutation.entity_id.clone(),
            op: mutation.op,
            payload_json,
        },
    )
    .await?;
    let event = raw_to_entity_event(appended.event)?;
    if !appended.inserted {
        return Ok(Some(AppliedGraphMutation {
            event,
            inserted: false,
            mutated: false,
        }));
    }

    current_metadata = graph_metadata(&mutation.scope, &current_metadata)?;
    let validated_patch = graph_metadata(&mutation.scope, &mutation.metadata_patch)?;
    current_metadata.extend(validated_patch);
    let patched_document = if let Some(replacement) = mutation.document {
        replacement
    } else {
        let mut document_value = document
            .as_deref()
            .and_then(|value| serde_json::from_str::<Value>(value).ok())
            .and_then(|value| value.as_object().cloned())
            .unwrap_or_default();
        let mut document_metadata = document_value
            .remove("metadata")
            .and_then(|value| value.as_object().cloned())
            .unwrap_or_default();
        document_metadata.extend(mutation.metadata_patch);
        document_value.insert("metadata".to_owned(), Value::Object(document_metadata));
        serde_json::to_string(&Value::Object(document_value))?
    };
    let patched_metadata = serde_json::to_string(&current_metadata)?;
    client
        .execute(
            &format!(
                "UPDATE {table} SET document = $1, metadata = {}::JSONB, updated_at = NOW() WHERE id = $2",
                sql_text_literal(&patched_metadata)
            ),
            &[&patched_document, &mutation.entity_id],
        )
        .await
        .map_err(|error| {
            PostgresStoreError::Backend(format!("graph metadata patch failed: {error:?}"))
        })?;
    Ok(Some(AppliedGraphMutation {
        event,
        inserted: true,
        mutated: true,
    }))
}

async fn apply_graph_delete_mutation<C>(
    client: &C,
    tables: &Tables,
    mutation: GraphDeleteMutation,
) -> PostgresStoreResult<Option<AppliedGraphMutation>>
where
    C: GenericClient + Sync,
{
    require_namespace(&mutation.scope.namespace)?;
    if mutation.entity_id.is_empty() {
        return Err(StoreError::EmptyRecordId.into());
    }
    if mutation.event_id.is_empty() {
        return Err(PostgresStoreError::Backend(
            "graph delete event id must not be empty".to_owned(),
        ));
    }
    advisory_lock(client, &mutation.event_id, 0).await?;
    if let Some(existing) = event_by_id(client, tables, &mutation.event_id).await? {
        if existing.namespace != mutation.scope.namespace {
            return Err(PostgresStoreError::EventIdNamespaceCollision {
                event_id: mutation.event_id,
                existing_namespace: existing.namespace,
                requested_namespace: mutation.scope.namespace,
            });
        }
        return Ok(Some(AppliedGraphMutation {
            event: raw_to_entity_event(existing)?,
            inserted: false,
            mutated: false,
        }));
    }

    let table = graph_table(tables, &mutation.table)?;
    let Some(row) = client
        .query_opt(
            &format!("SELECT metadata::TEXT FROM {table} WHERE id = $1 FOR UPDATE"),
            &[&mutation.entity_id],
        )
        .await
        .map_err(backend)?
    else {
        return Ok(None);
    };
    let current_metadata_json: String = row.try_get(0).map_err(backend)?;
    let current_metadata: Map<String, Value> = serde_json::from_str(&current_metadata_json)?;
    if !graph_scope_matches_or_legacy_default(&mutation.scope, &current_metadata)? {
        return Err(PostgresStoreError::Backend(
            "graph record id belongs to a different scope".to_owned(),
        ));
    }

    let appended = append_raw_entity_event(
        client,
        tables,
        &mutation.scope.namespace,
        NewRawEntityEvent {
            event_id: mutation.event_id,
            entity_kind: mutation.entity_kind,
            entity_id: mutation.entity_id.clone(),
            op: "DELETE".to_owned(),
            payload_json: serde_json::to_string(&mutation.payload)?,
        },
    )
    .await?;
    let event = raw_to_entity_event(appended.event)?;
    client
        .execute(
            &format!("DELETE FROM {table} WHERE id = $1"),
            &[&mutation.entity_id],
        )
        .await
        .map_err(|error| {
            PostgresStoreError::Backend(format!("graph projection delete failed: {error:?}"))
        })?;
    Ok(Some(AppliedGraphMutation {
        event,
        inserted: true,
        mutated: true,
    }))
}

async fn upsert_graph_projection<C>(
    client: &C,
    tables: &Tables,
    write: GraphProjectionUpsert,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    require_namespace(&write.scope.namespace)?;
    if write.record.id.is_empty() {
        return Err(StoreError::EmptyRecordId.into());
    }
    let metadata = graph_metadata(&write.scope, &write.record.metadata)?;
    if let Some(embedding) = &write.record.embedding {
        finite_vector(embedding)?;
        if embedding.len() != write.embedding_dim {
            return Err(StoreError::VectorDimensionMismatch {
                expected: write.embedding_dim,
                actual: embedding.len(),
            }
            .into());
        }
    }
    let table = graph_table(tables, &write.table)?;
    if let Some(row) = client
        .query_opt(
            &format!("SELECT metadata::TEXT FROM {table} WHERE id = $1 FOR UPDATE"),
            &[&write.record.id],
        )
        .await
        .map_err(backend)?
    {
        let current_json: String = row.try_get(0).map_err(backend)?;
        let current: Map<String, Value> = serde_json::from_str(&current_json)?;
        if !graph_scope_matches_or_legacy_default(&write.scope, &current)? {
            return Err(PostgresStoreError::Backend(
                "graph record id belongs to a different scope".to_owned(),
            ));
        }
    }
    let metadata_json = serde_json::to_string(&metadata)?;
    let embedding_sql = write
        .record
        .embedding
        .as_deref()
        .map(|vector| format!("{}::vector", sql_text_literal(&vector_literal(vector))))
        .unwrap_or_else(|| "NULL".to_owned());
    client
        .execute(
            &format!(
                "INSERT INTO {table}(id, document, metadata, embedding) VALUES ($1, $2, {}::JSONB, {embedding_sql}) \
                 ON CONFLICT(id) DO UPDATE SET document = EXCLUDED.document, metadata = EXCLUDED.metadata, \
                 embedding = EXCLUDED.embedding, updated_at = NOW()",
                sql_text_literal(&metadata_json),
            ),
            &[&write.record.id, &write.record.document],
        )
        .await
        .map_err(|error| {
            PostgresStoreError::Backend(format!("graph projection upsert failed: {error:?}"))
        })?;
    Ok(())
}

async fn patch_graph_projection_metadata<C>(
    client: &C,
    tables: &Tables,
    patch: GraphProjectionMetadataPatch,
) -> PostgresStoreResult<bool>
where
    C: GenericClient + Sync,
{
    require_namespace(&patch.scope.namespace)?;
    if patch.entity_id.is_empty() {
        return Err(StoreError::EmptyRecordId.into());
    }
    let table = graph_table(tables, &patch.table)?;
    let Some(row) = client
        .query_opt(
            &format!("SELECT document, metadata::TEXT FROM {table} WHERE id = $1 FOR UPDATE"),
            &[&patch.entity_id],
        )
        .await
        .map_err(backend)?
    else {
        return Ok(false);
    };
    let document: Option<String> = row.try_get(0).map_err(backend)?;
    let current_metadata_json: String = row.try_get(1).map_err(backend)?;
    let mut current_metadata: Map<String, Value> = serde_json::from_str(&current_metadata_json)?;
    if !graph_scope_matches_or_legacy_default(&patch.scope, &current_metadata)? {
        return Err(PostgresStoreError::Backend(
            "graph record id belongs to a different scope".to_owned(),
        ));
    }
    current_metadata = graph_metadata(&patch.scope, &current_metadata)?;
    let validated_patch = graph_metadata(&patch.scope, &patch.metadata_patch)?;
    current_metadata.extend(validated_patch);
    let patched_document = if let Some(replacement) = patch.document {
        replacement
    } else if !patch.patch_document_metadata {
        document.unwrap_or_default()
    } else {
        let mut document_value = document
            .as_deref()
            .and_then(|value| serde_json::from_str::<Value>(value).ok())
            .and_then(|value| value.as_object().cloned())
            .unwrap_or_default();
        let mut document_metadata = document_value
            .remove("metadata")
            .and_then(|value| value.as_object().cloned())
            .unwrap_or_default();
        document_metadata.extend(patch.metadata_patch);
        document_value.insert("metadata".to_owned(), Value::Object(document_metadata));
        serde_json::to_string(&Value::Object(document_value))?
    };
    let metadata_json = serde_json::to_string(&current_metadata)?;
    client
        .execute(
            &format!(
                "UPDATE {table} SET document = $1, metadata = {}::JSONB, updated_at = NOW() WHERE id = $2",
                sql_text_literal(&metadata_json)
            ),
            &[&patched_document, &patch.entity_id],
        )
        .await
        .map_err(|error| {
            PostgresStoreError::Backend(format!("graph projection patch failed: {error:?}"))
        })?;
    Ok(true)
}

async fn graph_projection_records<C>(
    client: &C,
    tables: &Tables,
    read: GraphProjectionRead,
) -> PostgresStoreResult<Vec<GraphRecord>>
where
    C: GenericClient + Sync,
{
    let table = graph_table(tables, &read.table)?;
    let (filter_sql, mut values) = graph_filter_sql(&read.scope, &read.metadata, 1)?;
    let mut clauses = vec![filter_sql];
    if let Some(ids) = read.ids {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let start = values.len() + 1;
        clauses.push(format!(
            "id IN ({})",
            (0..ids.len())
                .map(|index| format!("${}", start + index))
                .collect::<Vec<_>>()
                .join(", ")
        ));
        values.extend(ids);
    }
    let limit = i64::try_from(read.limit).unwrap_or(i64::MAX).to_string();
    values.push(limit);
    let params: Vec<&(dyn tokio_postgres::types::ToSql + Sync)> = values
        .iter()
        .map(|value| value as &(dyn tokio_postgres::types::ToSql + Sync))
        .collect();
    let rows = client
        .query(
            &format!(
                "SELECT id, document, metadata::TEXT, embedding::TEXT FROM {table} WHERE {} ORDER BY id ASC LIMIT ${}::TEXT::BIGINT",
                clauses.join(" AND "),
                values.len()
            ),
            &params,
        )
        .await
        .map_err(backend)?;
    rows.iter().map(graph_record_from_row).collect()
}

async fn graph_projection_vector_query<C>(
    client: &C,
    tables: &Tables,
    query: GraphProjectionVectorQuery,
) -> PostgresStoreResult<Vec<VectorMatch>>
where
    C: GenericClient + Sync,
{
    if query.query.embedding.len() != query.embedding_dim {
        return Err(StoreError::VectorDimensionMismatch {
            expected: query.embedding_dim,
            actual: query.query.embedding.len(),
        }
        .into());
    }
    finite_vector(&query.query.embedding)?;
    let table = graph_table(tables, &query.table)?;
    let (filter_sql, mut values) = graph_filter_sql(&query.scope, &query.query.metadata, 1)?;
    let vector_sql = format!(
        "{}::vector",
        sql_text_literal(&vector_literal(&query.query.embedding))
    );
    let limit_index = values.len() + 1;
    values.push(
        i64::try_from(query.query.limit)
            .unwrap_or(i64::MAX)
            .to_string(),
    );
    let params: Vec<&(dyn tokio_postgres::types::ToSql + Sync)> = values
        .iter()
        .map(|value| value as &(dyn tokio_postgres::types::ToSql + Sync))
        .collect();
    let operator = match query.query.metric {
        kogwistar_store::DistanceMetric::Cosine => "<=>",
        kogwistar_store::DistanceMetric::L2 => "<->",
        kogwistar_store::DistanceMetric::InnerProduct => "<#>",
    };
    let rows = client
        .query(
            &format!(
                "SELECT id, document, metadata::TEXT, embedding::TEXT, embedding {operator} {vector_sql} AS distance \
                 FROM {table} WHERE embedding IS NOT NULL AND {filter_sql} \
                 ORDER BY distance ASC, id ASC LIMIT ${limit_index}::TEXT::BIGINT"
            ),
            &params,
        )
        .await
        .map_err(backend)?;
    rows.iter()
        .map(|row| {
            Ok(VectorMatch {
                record: graph_record_from_row(row)?,
                distance: row.try_get(4).map_err(backend)?,
            })
        })
        .collect()
}

async fn event_by_id<C>(
    client: &C,
    tables: &Tables,
    event_id: &str,
) -> PostgresStoreResult<Option<RawEntityEvent>>
where
    C: GenericClient + Sync,
{
    client
        .query_opt(
            &format!(
                "SELECT namespace, seq, event_id, entity_kind, entity_id, op, payload_json, created_at::TEXT \
                 FROM {} WHERE event_id = $1",
                tables.entity_events
            ),
            &[&event_id],
        )
        .await
        .map_err(backend)?
        .as_ref()
        .map(raw_event_from_row)
        .transpose()
}

async fn replay_raw_events<C>(
    client: &C,
    tables: &Tables,
    namespace: &str,
    after_seq: i64,
    limit: usize,
) -> PostgresStoreResult<Vec<RawEntityEvent>>
where
    C: GenericClient + Sync,
{
    let limit = i64::try_from(limit).unwrap_or(i64::MAX);
    client
        .query(
            &format!(
                "SELECT namespace, seq, event_id, entity_kind, entity_id, op, payload_json, created_at::TEXT \
                 FROM {} WHERE namespace = $1 AND seq > $2 ORDER BY seq ASC LIMIT $3",
                tables.entity_events
            ),
            &[&namespace, &after_seq, &limit],
        )
        .await
        .map_err(backend)?
        .iter()
        .map(raw_event_from_row)
        .collect()
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

async fn recover_entity_projection_uow(
    uow: &mut PostgresUnitOfWork<'_>,
    request: &EntityRecoveryRequest,
) -> PostgresStoreResult<EntityRecoveryReport> {
    validate_entity_recovery_request(request)?;
    // Serialize each consumer cursor and its dependent projection replacement.
    advisory_lock(
        &uow.transaction,
        &cursor_lock_key(&request.namespace, &request.consumer),
        2,
    )
    .await?;
    advisory_lock(
        &uow.transaction,
        &projection_lock_key(&request.projection_namespace, &request.projection_key),
        3,
    )
    .await?;
    let prior_cursor = uow
        .replay_cursor(&request.namespace, &request.consumer)
        .await?
        .last_seq;
    let current = named_projection(
        &uow.transaction,
        &uow.tables,
        &request.projection_namespace,
        &request.projection_key,
    )
    .await?;
    let mut projection = EntityProjection::from_payload(current.as_ref().map(|row| &row.payload))?;
    if projection.last_seq() != prior_cursor {
        return Err(StoreError::InvalidEntityEventPayload {
            message: "recovery projection cursor does not match durable replay cursor".to_owned(),
        }
        .into());
    }
    let events = uow
        .replay_raw_events(&request.namespace, prior_cursor, request.batch_limit)
        .await?;
    let latest_authoritative_seq = uow.latest_retained_event_seq(&request.namespace).await?;
    for raw in &events {
        projection.fold(&raw_to_entity_event(raw.clone())?)?;
    }
    let projection_changed = !events.is_empty() || current.is_none();
    if projection_changed {
        uow.replace_named_projection(
            &request.projection_namespace,
            &request.projection_key,
            recovery_projection_write(&projection, latest_authoritative_seq),
        )
        .await?;
    }
    if request.abort_after_projection {
        return Err(PostgresStoreError::TransactionAborted(
            "requested after entity recovery projection write".to_owned(),
        ));
    }
    if projection_changed || projection.last_seq() != prior_cursor {
        uow.strict_advance_replay_cursor(
            &request.namespace,
            &request.consumer,
            projection.last_seq(),
        )
        .await?;
    }
    Ok(recovery_report(
        &projection,
        events.len(),
        prior_cursor,
        latest_authoritative_seq,
    ))
}

async fn rebuild_entity_projection_uow(
    uow: &mut PostgresUnitOfWork<'_>,
    request: &EntityRebuildRequest,
) -> PostgresStoreResult<EntityRecoveryReport> {
    validate_entity_rebuild_request(request)?;
    advisory_lock(
        &uow.transaction,
        &cursor_lock_key(&request.namespace, &request.consumer),
        2,
    )
    .await?;
    advisory_lock(
        &uow.transaction,
        &projection_lock_key(&request.projection_namespace, &request.projection_key),
        3,
    )
    .await?;
    let prior_cursor = uow
        .replay_cursor(&request.namespace, &request.consumer)
        .await?
        .last_seq;
    let events = uow
        .replay_raw_events(&request.namespace, 0, usize::MAX)
        .await?;
    let latest_authoritative_seq = uow.latest_retained_event_seq(&request.namespace).await?;
    let mut projection = EntityProjection::empty();
    for raw in &events {
        projection.fold(&raw_to_entity_event(raw.clone())?)?;
    }
    uow.replace_named_projection(
        &request.projection_namespace,
        &request.projection_key,
        recovery_projection_write(&projection, latest_authoritative_seq),
    )
    .await?;
    if request.abort_after_projection {
        return Err(PostgresStoreError::TransactionAborted(
            "requested after entity rebuild projection write".to_owned(),
        ));
    }
    // Reset is explicit operator semantics; complete projection and cursor are
    // still one UoW. Incremental recovery remains strict and monotonic.
    uow.set_replay_cursor_legacy(&request.namespace, &request.consumer, projection.last_seq())
        .await?;
    Ok(recovery_report(
        &projection,
        events.len(),
        prior_cursor,
        latest_authoritative_seq,
    ))
}

async fn latest_retained_event_seq<C>(
    client: &C,
    tables: &Tables,
    namespace: &str,
) -> PostgresStoreResult<i64>
where
    C: GenericClient + Sync,
{
    let row = client
        .query_one(
            &format!(
                "SELECT COALESCE(MAX(seq), 0) FROM {} WHERE namespace = $1",
                tables.entity_events
            ),
            &[&namespace],
        )
        .await
        .map_err(backend)?;
    row.try_get(0).map_err(backend)
}

async fn prune_entity_events_after<C>(
    client: &C,
    tables: &Tables,
    namespace: &str,
    to_seq: i64,
) -> PostgresStoreResult<u64>
where
    C: GenericClient + Sync,
{
    client
        .execute(
            &format!(
                "DELETE FROM {} WHERE namespace = $1 AND seq > $2",
                tables.entity_events
            ),
            &[&namespace, &to_seq],
        )
        .await
        .map_err(backend)
}

fn pg_schema_version(value: i64) -> PostgresStoreResult<i32> {
    i32::try_from(value).map_err(|_| {
        PostgresStoreError::Backend(
            "workflow design schema version exceeds PostgreSQL INTEGER".to_owned(),
        )
    })
}

async fn put_workflow_design_snapshot<C>(
    client: &C,
    tables: &Tables,
    workflow_id: &str,
    snapshot: WorkflowDesignSnapshotWrite,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    let schema_version = pg_schema_version(snapshot.schema_version)?;
    let created_at_ms = unix_epoch_millis();
    client
        .execute(
            &format!(
                "INSERT INTO {}(workflow_id, version, seq, payload_json, schema_version, created_at_ms) \
                 VALUES ($1,$2,$3,$4,$5,$6) \
                 ON CONFLICT(workflow_id,version) DO UPDATE SET \
                 seq=EXCLUDED.seq,payload_json=EXCLUDED.payload_json, \
                 schema_version=EXCLUDED.schema_version,created_at_ms=EXCLUDED.created_at_ms",
                tables.workflow_design_snapshots
            ),
            &[
                &workflow_id,
                &snapshot.version,
                &snapshot.seq,
                &snapshot.payload_json,
                &schema_version,
                &created_at_ms,
            ],
        )
        .await
        .map_err(backend)?;
    Ok(())
}

async fn workflow_design_snapshot<C>(
    client: &C,
    tables: &Tables,
    workflow_id: &str,
    max_version: i64,
    schema_version: i64,
) -> PostgresStoreResult<Option<WorkflowDesignSnapshot>>
where
    C: GenericClient + Sync,
{
    let schema_version = pg_schema_version(schema_version)?;
    client
        .query_opt(
            &format!(
                "SELECT workflow_id,version,seq,payload_json,schema_version,created_at_ms \
                 FROM {} WHERE workflow_id=$1 AND version <= $2 AND schema_version=$3 \
                 ORDER BY version DESC LIMIT 1",
                tables.workflow_design_snapshots
            ),
            &[&workflow_id, &max_version, &schema_version],
        )
        .await
        .map_err(backend)?
        .as_ref()
        .map(workflow_design_snapshot_from_row)
        .transpose()
}

async fn clear_workflow_design_snapshots<C>(
    client: &C,
    tables: &Tables,
    workflow_id: &str,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    client
        .execute(
            &format!(
                "DELETE FROM {} WHERE workflow_id = $1",
                tables.workflow_design_snapshots
            ),
            &[&workflow_id],
        )
        .await
        .map_err(backend)?;
    Ok(())
}

async fn put_workflow_design_delta<C>(
    client: &C,
    tables: &Tables,
    workflow_id: &str,
    delta: WorkflowDesignDeltaWrite,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    let schema_version = pg_schema_version(delta.schema_version)?;
    let created_at_ms = unix_epoch_millis();
    client
        .execute(
            &format!(
                "INSERT INTO {}(workflow_id,version,prev_version,target_seq,forward_json,inverse_json,schema_version,created_at_ms) \
                 VALUES ($1,$2,$3,$4,$5,$6,$7,$8) \
                 ON CONFLICT(workflow_id,version) DO UPDATE SET \
                 prev_version=EXCLUDED.prev_version,target_seq=EXCLUDED.target_seq, \
                 forward_json=EXCLUDED.forward_json,inverse_json=EXCLUDED.inverse_json, \
                 schema_version=EXCLUDED.schema_version,created_at_ms=EXCLUDED.created_at_ms",
                tables.workflow_design_version_deltas
            ),
            &[
                &workflow_id,
                &delta.version,
                &delta.prev_version,
                &delta.target_seq,
                &delta.forward_json,
                &delta.inverse_json,
                &schema_version,
                &created_at_ms,
            ],
        )
        .await
        .map_err(backend)?;
    Ok(())
}

async fn workflow_design_delta<C>(
    client: &C,
    tables: &Tables,
    workflow_id: &str,
    version: i64,
    schema_version: i64,
) -> PostgresStoreResult<Option<WorkflowDesignDelta>>
where
    C: GenericClient + Sync,
{
    let schema_version = pg_schema_version(schema_version)?;
    client
        .query_opt(
            &format!(
                "SELECT workflow_id,version,prev_version,target_seq,forward_json,inverse_json,schema_version,created_at_ms \
                 FROM {} WHERE workflow_id=$1 AND version=$2 AND schema_version=$3",
                tables.workflow_design_version_deltas
            ),
            &[&workflow_id, &version, &schema_version],
        )
        .await
        .map_err(backend)?
        .as_ref()
        .map(workflow_design_delta_from_row)
        .transpose()
}

async fn clear_workflow_design_deltas<C>(
    client: &C,
    tables: &Tables,
    workflow_id: &str,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    client
        .execute(
            &format!(
                "DELETE FROM {} WHERE workflow_id = $1",
                tables.workflow_design_version_deltas
            ),
            &[&workflow_id],
        )
        .await
        .map_err(backend)?;
    Ok(())
}

async fn create_server_run<C>(
    client: &C,
    tables: &Tables,
    run: ServerRunCreate,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    let now = unix_epoch_millis();
    client.execute(&format!("INSERT INTO {}(run_id,conversation_id,workflow_id,user_id,user_turn_node_id,assistant_turn_node_id,status,cancel_requested,result_json,error_json,created_at_ms,updated_at_ms,started_at_ms,finished_at_ms) VALUES ($1,$2,$3,$4,$5,NULL,$6,0,NULL,NULL,$7,$7,NULL,NULL)", tables.server_runs), &[&run.run_id,&run.conversation_id,&run.workflow_id,&run.user_id,&run.user_turn_node_id,&run.status,&now]).await.map_err(backend)?;
    Ok(())
}
async fn server_run<C>(
    client: &C,
    tables: &Tables,
    run_id: &str,
) -> PostgresStoreResult<Option<ServerRun>>
where
    C: GenericClient + Sync,
{
    client.query_opt(&format!("SELECT run_id,conversation_id,workflow_id,user_id,user_turn_node_id,assistant_turn_node_id,status,cancel_requested,result_json,error_json,created_at_ms,updated_at_ms,started_at_ms,finished_at_ms FROM {} WHERE run_id=$1", tables.server_runs), &[&run_id]).await.map_err(backend)?.as_ref().map(server_run_from_row).transpose()
}
async fn server_runs<C>(
    client: &C,
    tables: &Tables,
    status: Option<&str>,
    workflow_id: Option<&str>,
    conversation_id: Option<&str>,
    limit: usize,
) -> PostgresStoreResult<Vec<ServerRun>>
where
    C: GenericClient + Sync,
{
    let limit = i64::try_from(limit).unwrap_or(i64::MAX);
    client.query(&format!("SELECT run_id,conversation_id,workflow_id,user_id,user_turn_node_id,assistant_turn_node_id,status,cancel_requested,result_json,error_json,created_at_ms,updated_at_ms,started_at_ms,finished_at_ms FROM {} WHERE ($1::TEXT IS NULL OR status=$1) AND ($2::TEXT IS NULL OR workflow_id=$2) AND ($3::TEXT IS NULL OR conversation_id=$3) ORDER BY created_at_ms DESC,run_id DESC LIMIT $4", tables.server_runs), &[&status,&workflow_id,&conversation_id,&limit]).await.map_err(backend)?.iter().map(server_run_from_row).collect()
}
async fn append_server_run_event<C>(
    client: &C,
    tables: &Tables,
    run_id: &str,
    event_type: &str,
    payload_json: String,
) -> PostgresStoreResult<ServerRunEvent>
where
    C: GenericClient + Sync,
{
    let created_at_ms = unix_epoch_millis();
    let row=client.query_one(&format!("INSERT INTO {}(run_id,event_type,payload_json,created_at_ms) VALUES ($1,$2,$3,$4) RETURNING seq", tables.server_run_events), &[&run_id,&event_type,&payload_json,&created_at_ms]).await.map_err(backend)?;
    Ok(ServerRunEvent {
        seq: row.try_get(0).map_err(backend)?,
        run_id: run_id.to_owned(),
        event_type: event_type.to_owned(),
        payload_json,
        created_at_ms,
    })
}
async fn server_run_events<C>(
    client: &C,
    tables: &Tables,
    run_id: &str,
    after_seq: i64,
    limit: usize,
) -> PostgresStoreResult<Vec<ServerRunEvent>>
where
    C: GenericClient + Sync,
{
    let limit = i64::try_from(limit).unwrap_or(i64::MAX);
    client.query(&format!("SELECT seq,run_id,event_type,payload_json,created_at_ms FROM {} WHERE run_id=$1 AND seq>$2 ORDER BY seq ASC LIMIT $3", tables.server_run_events), &[&run_id,&after_seq,&limit]).await.map_err(backend)?.iter().map(server_run_event_from_row).collect()
}
async fn update_server_run<C>(
    client: &C,
    tables: &Tables,
    run_id: &str,
    update: ServerRunUpdate,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    let now = unix_epoch_millis();
    let cancel_requested = update.cancel_requested.map(i32::from);
    client.execute(&format!("UPDATE {} SET status=$1,assistant_turn_node_id=$2,result_json=$3,error_json=$4,started_at_ms=$5,finished_at_ms=$6,cancel_requested=COALESCE($7,cancel_requested),updated_at_ms=$8 WHERE run_id=$9", tables.server_runs), &[&update.status,&update.assistant_turn_node_id,&update.result_json,&update.error_json,&update.started_at_ms,&update.finished_at_ms,&cancel_requested,&now,&run_id]).await.map_err(backend)?;
    Ok(())
}
async fn request_server_run_cancel<C>(
    client: &C,
    tables: &Tables,
    run_id: &str,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    advisory_lock(client, &format!("recorded-run\u{1f}{run_id}"), 42).await?;
    let Some(run) = server_run(client, tables, run_id).await? else {
        return Ok(());
    };
    let mut current = read_recorded_runtime_state(
        client,
        tables,
        run_id,
        &run.workflow_id,
        &run.conversation_id,
    )
    .await?;
    if current.is_none() {
        let payload_json = client
            .query_opt(
                &format!(
                    "SELECT payload_json FROM {} WHERE run_id=$1 AND msg_type='workflow.run.execute' \
                     ORDER BY created_at LIMIT 1",
                    tables.projected_lane_messages
                ),
                &[&run_id],
            )
            .await
            .map_err(backend)?
            .and_then(|row| row.try_get::<_, Option<String>>(0).ok().flatten());
        if let Some(payload_json) = payload_json {
            let payload: Value = serde_json::from_str(&payload_json)?;
            let expected_event_seq = latest_server_run_event_seq(client, tables, run_id).await?;
            apply_recorded_runtime_transition_inner(
                client,
                tables,
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
                None,
                false,
            )
            .await?;
            current = read_recorded_runtime_state(
                client,
                tables,
                run_id,
                &run.workflow_id,
                &run.conversation_id,
            )
            .await?;
        }
    }
    if let Some(state) = current
        && !state.status.is_terminal()
    {
        let expected_event_seq = latest_server_run_event_seq(client, tables, run_id).await?;
        apply_recorded_runtime_transition_inner(
            client,
            tables,
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
            None,
            false,
        )
        .await?;
    } else {
        let now = unix_epoch_millis();
        client.execute(&format!("UPDATE {} SET cancel_requested=1,status=CASE WHEN status IN ('cancelled','failed','succeeded') THEN status ELSE 'cancelling' END,updated_at_ms=$1 WHERE run_id=$2", tables.server_runs), &[&now,&run_id]).await.map_err(backend)?;
    }
    client
        .execute(
            &format!(
                "UPDATE {} SET status='cancelled',claimed_by=NULL,lease_until=NULL \
                 WHERE run_id=$1 AND status NOT IN ('completed','failed','cancelled','dead-letter')",
                tables.projected_lane_messages
            ),
            &[&run_id],
        )
        .await
        .map_err(backend)?;
    Ok(())
}

const RECORDED_RUNTIME_EVENT_TYPE: &str = "workflow.recorded_transition.v1";

fn parse_runtime_event_payload(payload_json: &str) -> Option<PersistedRecordedTransition> {
    serde_json::from_str::<PersistedRecordedTransition>(payload_json)
        .ok()
        .filter(|value| value.contract_version == RECORDED_RUNTIME_CONTRACT_VERSION)
}

async fn recorded_runtime_events<C>(
    client: &C,
    tables: &Tables,
    run_id: Option<&str>,
) -> PostgresStoreResult<Vec<ServerRunEvent>>
where
    C: GenericClient + Sync,
{
    let rows = client
        .query(
            &format!(
                "SELECT seq,run_id,event_type,payload_json,created_at_ms FROM {} \
                 WHERE event_type=$1 AND ($2::TEXT IS NULL OR run_id=$2) ORDER BY seq ASC",
                tables.server_run_events
            ),
            &[&RECORDED_RUNTIME_EVENT_TYPE, &run_id],
        )
        .await
        .map_err(backend)?;
    rows.iter().map(server_run_event_from_row).collect()
}

async fn latest_server_run_event_seq<C>(
    client: &C,
    tables: &Tables,
    run_id: &str,
) -> PostgresStoreResult<i64>
where
    C: GenericClient + Sync,
{
    client
        .query_one(
            &format!(
                "SELECT COALESCE(MAX(seq),0) FROM {} WHERE run_id=$1",
                tables.server_run_events
            ),
            &[&run_id],
        )
        .await
        .map_err(backend)?
        .try_get(0)
        .map_err(backend)
}

fn runtime_json_object<T: serde::Serialize>(value: &T) -> PostgresStoreResult<Map<String, Value>> {
    serde_json::to_value(value)?
        .as_object()
        .cloned()
        .ok_or_else(|| {
            PostgresStoreError::TransactionAborted(
                "runtime projection payload is not an object".to_owned(),
            )
        })
}

async fn store_runtime_current_state<C>(
    client: &C,
    tables: &Tables,
    persisted: &PersistedRecordedTransition,
    event_seq: i64,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    replace_named_projection(
        client,
        tables,
        RUNTIME_CURRENT_STATE_NAMESPACE,
        &persisted.reduced.state.run_id,
        runtime_projection_write(
            runtime_json_object(&persisted.reduced.state)?,
            event_seq,
            "ready".to_owned(),
        ),
    )
    .await
}

async fn recorded_transition_by_id<C>(
    client: &C,
    tables: &Tables,
    transition_id: &str,
) -> PostgresStoreResult<Option<(PersistedRecordedTransition, i64)>>
where
    C: GenericClient + Sync,
{
    client
        .query_opt(
            &format!(
                "SELECT payload_json, seq FROM {} \
                 WHERE event_type='workflow.recorded_transition.v1' \
                 AND payload_json::jsonb->>'transition_id'=$1 LIMIT 1",
                tables.server_run_events
            ),
            &[&transition_id],
        )
        .await
        .map_err(backend)?
        .map(|row| {
            let payload_json: String = row.try_get(0).map_err(backend)?;
            let seq: i64 = row.try_get(1).map_err(backend)?;
            Ok((serde_json::from_str(&payload_json)?, seq))
        })
        .transpose()
}

async fn read_recorded_runtime_state<C>(
    client: &C,
    tables: &Tables,
    run_id: &str,
    workflow_id: &str,
    conversation_id: &str,
) -> PostgresStoreResult<Option<RecordedRuntimeState>>
where
    C: GenericClient + Sync,
{
    let Some(run) = server_run(client, tables, run_id).await? else {
        return Ok(None);
    };
    if run.workflow_id != workflow_id || run.conversation_id != conversation_id {
        return Err(PostgresStoreError::RecordedRuntimeConflict(format!(
            "run identity differs for {run_id:?}"
        )));
    }
    if let Some(row) =
        named_projection(client, tables, RUNTIME_CURRENT_STATE_NAMESPACE, run_id).await?
        && row.materialization_status == "ready"
    {
        let has_newer_recorded_event: bool = client
            .query_one(
                &format!(
                    "SELECT EXISTS(
                        SELECT 1 FROM {} WHERE run_id=$1 AND event_type=$2 AND seq>$3
                     )",
                    tables.server_run_events
                ),
                &[
                    &run_id,
                    &RECORDED_RUNTIME_EVENT_TYPE,
                    &row.last_authoritative_seq,
                ],
            )
            .await
            .map_err(backend)?
            .try_get(0)
            .map_err(backend)?;
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
    let mut latest = None;
    for event in recorded_runtime_events(client, tables, Some(run_id)).await? {
        if let Some(payload) = parse_runtime_event_payload(&event.payload_json)
            && payload.reduced.state.run_id == run_id
            && payload.reduced.state.workflow_id == workflow_id
            && payload.reduced.state.conversation_id == conversation_id
        {
            latest = Some(payload.reduced.state);
        }
    }
    Ok(latest)
}

async fn apply_recorded_runtime_transition<C>(
    client: &C,
    tables: &Tables,
    transition: RecordedRuntimeTransition,
    abort_after_writes: bool,
) -> PostgresStoreResult<RecordedTransitionResult>
where
    C: GenericClient + Sync,
{
    apply_recorded_runtime_transition_inner(
        client,
        tables,
        transition,
        None,
        None,
        abort_after_writes,
    )
    .await
}

async fn apply_claimed_recorded_runtime_transition<C>(
    client: &C,
    tables: &Tables,
    handoff: RecordedWorkerHandoff,
    transition: RecordedRuntimeTransition,
    abort_after_writes: bool,
) -> PostgresStoreResult<RecordedTransitionResult>
where
    C: GenericClient + Sync,
{
    validate_worker_handoff(&handoff)?;
    let row = client
        .query_opt(
            &format!(
                "SELECT {LANE_SELECT},lease_until>=NOW() FROM {} WHERE message_id=$1 FOR UPDATE",
                tables.projected_lane_messages
            ),
            &[&handoff.message_id],
        )
        .await
        .map_err(backend)?
        .ok_or_else(|| {
            PostgresStoreError::RecordedRuntimeConflict(format!(
                "worker lane message {:?} does not exist",
                handoff.message_id
            ))
        })?;
    let lane = lane_from_row(&row)?;
    let lease_active: Option<bool> = row.try_get(25).map_err(backend)?;
    validate_worker_lane_identity(&lane, &handoff, &transition)?;

    if lane.status == "completed" {
        return retry_completed_worker_handoff(client, tables, &handoff, &transition).await;
    }
    if lane.status != "claimed"
        || lane.claimed_by.as_deref() != Some(handoff.claimed_by.as_str())
        || lease_active != Some(true)
    {
        return Err(PostgresStoreError::RecordedRuntimeConflict(format!(
            "worker lane message {:?} is not actively claimed by {:?}",
            handoff.message_id, handoff.claimed_by
        )));
    }

    let result = apply_recorded_runtime_transition_inner(
        client,
        tables,
        transition,
        Some(handoff.clone()),
        None,
        false,
    )
    .await?;
    let acknowledged = client
        .execute(
            &format!(
                "UPDATE {} SET status='completed',claimed_by=NULL,lease_until=NULL \
                 WHERE message_id=$1 AND status='claimed' AND claimed_by=$2 AND lease_until>=NOW()",
                tables.projected_lane_messages
            ),
            &[&handoff.message_id, &handoff.claimed_by],
        )
        .await
        .map_err(backend)?;
    if acknowledged != 1 {
        return Err(PostgresStoreError::RecordedRuntimeConflict(format!(
            "worker lane message {:?} claim changed before acknowledgement",
            handoff.message_id
        )));
    }
    if abort_after_writes {
        return Err(PostgresStoreError::TransactionAborted(
            "requested after recorded runtime worker handoff writes".to_owned(),
        ));
    }
    Ok(result)
}

fn validate_worker_handoff(handoff: &RecordedWorkerHandoff) -> PostgresStoreResult<()> {
    for (field, value) in [
        ("message_id", handoff.message_id.as_str()),
        ("claimed_by", handoff.claimed_by.as_str()),
        ("run_id", handoff.run_id.as_str()),
        ("step_id", handoff.step_id.as_str()),
        ("correlation_id", handoff.correlation_id.as_str()),
    ] {
        if value.is_empty() {
            return Err(PostgresStoreError::RecordedRuntimeConflict(format!(
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
) -> PostgresStoreResult<()> {
    if handoff.run_id != transition.run_id
        || transition.node_id.as_deref() != Some(handoff.step_id.as_str())
        || lane.run_id.as_deref() != Some(handoff.run_id.as_str())
        || lane.step_id.as_deref() != Some(handoff.step_id.as_str())
        || lane.correlation_id.as_deref() != Some(handoff.correlation_id.as_str())
    {
        return Err(PostgresStoreError::RecordedRuntimeConflict(format!(
            "worker lane message {:?} run/step/correlation identity differs",
            handoff.message_id
        )));
    }
    Ok(())
}

async fn retry_completed_worker_handoff<C>(
    client: &C,
    tables: &Tables,
    handoff: &RecordedWorkerHandoff,
    transition: &RecordedRuntimeTransition,
) -> PostgresStoreResult<RecordedTransitionResult>
where
    C: GenericClient + Sync,
{
    let request_digest = transition_digest(transition)?;
    if let Some((payload, event_seq)) =
        recorded_transition_by_id(client, tables, &transition.transition_id).await?
    {
        if payload.request_digest != request_digest
            || payload.worker_handoff.as_ref() != Some(handoff)
        {
            return Err(PostgresStoreError::RecordedRuntimeConflict(format!(
                "worker handoff {:?} retried with different result",
                handoff.message_id
            )));
        }
        return Ok(payload.result(event_seq, true));
    }
    Err(PostgresStoreError::RecordedRuntimeConflict(format!(
        "completed worker lane message {:?} has no matching recorded result",
        handoff.message_id
    )))
}

async fn apply_claimed_recorded_worker_effect<C>(
    client: &C,
    tables: &Tables,
    handoff: RecordedWorkerHandoff,
    effect: RecordedWorkerSuccessEffect,
) -> PostgresStoreResult<RecordedTransitionResult>
where
    C: GenericClient + Sync,
{
    validate_worker_handoff(&handoff)?;
    if effect.contract_version != RECORDED_RUNTIME_CONTRACT_VERSION || effect.effect_id.is_empty() {
        return Err(PostgresStoreError::RecordedRuntimeConflict(
            "worker effect contract_version/effect_id is invalid".to_owned(),
        ));
    }
    let effect_digest = worker_effect_digest(&effect)?;
    advisory_lock(client, &format!("recorded-run\u{1f}{}", handoff.run_id), 42).await?;
    let row = client
        .query_opt(
            &format!(
                "SELECT {LANE_SELECT},lease_until>=NOW() FROM {} WHERE message_id=$1 FOR UPDATE",
                tables.projected_lane_messages
            ),
            &[&handoff.message_id],
        )
        .await
        .map_err(backend)?
        .ok_or_else(|| {
            PostgresStoreError::RecordedRuntimeConflict(format!(
                "worker lane message {:?} does not exist",
                handoff.message_id
            ))
        })?;
    let lane = lane_from_row(&row)?;
    let lease_active: Option<bool> = row.try_get(25).map_err(backend)?;
    if lane.run_id.as_deref() != Some(handoff.run_id.as_str())
        || lane.step_id.as_deref() != Some(handoff.step_id.as_str())
        || lane.correlation_id.as_deref() != Some(handoff.correlation_id.as_str())
    {
        return Err(PostgresStoreError::RecordedRuntimeConflict(format!(
            "worker lane message {:?} run/step/correlation identity differs",
            handoff.message_id
        )));
    }
    if server_run(client, tables, &handoff.run_id)
        .await?
        .is_some_and(|run| run.cancel_requested)
    {
        return Err(PostgresStoreError::RecordedRuntimeConflict(format!(
            "worker lane message {:?} belongs to a cancel-requested run",
            handoff.message_id
        )));
    }
    if lane.status == "completed" {
        if let Some((payload, event_seq)) =
            recorded_transition_by_id(client, tables, &effect.effect_id).await?
            && payload
                .worker_handoff
                .as_ref()
                .is_some_and(|stored| stored.message_id == handoff.message_id)
        {
            if payload.worker_effect_digest.as_deref() != Some(effect_digest.as_str()) {
                return Err(PostgresStoreError::RecordedRuntimeConflict(format!(
                    "worker handoff {:?} retried with different effect",
                    handoff.message_id
                )));
            }
            return Ok(payload.result(event_seq, true));
        }
        return Err(PostgresStoreError::RecordedRuntimeConflict(format!(
            "completed worker lane message {:?} has no matching recorded effect",
            handoff.message_id
        )));
    }
    if lane.status != "claimed"
        || lane.claimed_by.as_deref() != Some(handoff.claimed_by.as_str())
        || lease_active != Some(true)
    {
        return Err(PostgresStoreError::RecordedRuntimeConflict(format!(
            "worker lane message {:?} is not actively claimed by {:?}",
            handoff.message_id, handoff.claimed_by
        )));
    }
    let payload: Value = serde_json::from_str(lane.payload_json.as_deref().unwrap_or("{}"))?;
    let workflow_id = payload["workflow_id"].as_str().unwrap_or_default();
    let conversation_id = payload["conversation_id"].as_str().unwrap_or_default();
    let token_id = payload["token_id"]
        .as_str()
        .unwrap_or(handoff.run_id.as_str());
    let parent_token_id = payload["parent_token_id"].as_str();
    let current = read_recorded_runtime_state(
        client,
        tables,
        &handoff.run_id,
        workflow_id,
        conversation_id,
    )
    .await?
    .ok_or_else(|| {
        PostgresStoreError::RecordedRuntimeConflict(format!(
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
        return Err(PostgresStoreError::RecordedRuntimeConflict(format!(
            "worker lane message {:?} result is not next in canonical frontier order",
            handoff.message_id
        )));
    }
    let expected_event_seq = latest_server_run_event_seq(client, tables, &handoff.run_id).await?;
    let step_seq = current.last_step_seq.saturating_add(1);
    let authoritative_successors = match effect.status {
        RuntimeWorkerEffectStatus::Success => {
            kogwistar_runtime::authoritative_runtime_successors_for_result(
                &current.static_routes,
                &handoff.step_id,
                &effect.successors,
                &effect.route_next,
                false,
            )?
        }
        RuntimeWorkerEffectStatus::Suspended => Vec::new(),
        RuntimeWorkerEffectStatus::Failed => {
            kogwistar_runtime::authoritative_runtime_successors_for_result(
                &current.static_routes,
                &handoff.step_id,
                &effect.successors,
                &effect.route_next,
                true,
            )?
        }
    };
    let terminal_failure =
        effect.status == RuntimeWorkerEffectStatus::Failed && authoritative_successors.is_empty();
    let frontier = match effect.status {
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
        RuntimeWorkerEffectStatus::Failed if terminal_failure => RuntimeFrontier::default(),
        RuntimeWorkerEffectStatus::Failed => frontier_after_worker_success(
            &current.frontier,
            &handoff.step_id,
            token_id,
            parent_token_id,
            step_seq,
            &kogwistar_runtime::RuntimeWorkerSuccessEffect {
                successors: authoritative_successors,
            },
        )?,
    };
    let transition = RecordedRuntimeTransition {
        contract_version: RECORDED_RUNTIME_CONTRACT_VERSION,
        transition_id: effect.effect_id,
        expected_event_seq,
        kind: match effect.status {
            RuntimeWorkerEffectStatus::Success => {
                kogwistar_runtime::RecordedTransitionKind::RecordedStepSuccess
            }
            RuntimeWorkerEffectStatus::Suspended => {
                kogwistar_runtime::RecordedTransitionKind::Suspend
            }
            RuntimeWorkerEffectStatus::Failed if terminal_failure => {
                kogwistar_runtime::RecordedTransitionKind::Fail
            }
            RuntimeWorkerEffectStatus::Failed => {
                kogwistar_runtime::RecordedTransitionKind::RecordedStepSuccess
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
        frontier: Some(frontier),
        result: effect.result,
        wait_reason: effect.wait_reason,
        resume_payload: effect.resume_payload,
        errors: effect.errors,
    };
    let result = apply_recorded_runtime_transition_inner(
        client,
        tables,
        transition,
        Some(handoff.clone()),
        Some(effect_digest),
        false,
    )
    .await?;
    let acknowledged = client
        .execute(
            &format!(
                "UPDATE {} SET status='completed',claimed_by=NULL,lease_until=NULL \
                 WHERE message_id=$1 AND status='claimed' AND claimed_by=$2 AND lease_until>=NOW()",
                tables.projected_lane_messages
            ),
            &[&handoff.message_id, &handoff.claimed_by],
        )
        .await
        .map_err(backend)?;
    if acknowledged != 1 {
        return Err(PostgresStoreError::RecordedRuntimeConflict(format!(
            "worker lane message {:?} claim changed before acknowledgement",
            handoff.message_id
        )));
    }
    Ok(result)
}

async fn apply_recorded_runtime_transition_inner<C>(
    client: &C,
    tables: &Tables,
    transition: RecordedRuntimeTransition,
    worker_handoff: Option<RecordedWorkerHandoff>,
    persisted_worker_effect_digest: Option<String>,
    abort_after_writes: bool,
) -> PostgresStoreResult<RecordedTransitionResult>
where
    C: GenericClient + Sync,
{
    let request_digest = transition_digest(&transition)?;
    // Serialize global transition-id reuse and per-run sequence/CAS checks.
    advisory_lock(
        client,
        &format!("recorded-transition\u{1f}{}", transition.transition_id),
        41,
    )
    .await?;
    advisory_lock(
        client,
        &format!("recorded-run\u{1f}{}", transition.run_id),
        42,
    )
    .await?;

    if let Some((payload, event_seq)) =
        recorded_transition_by_id(client, tables, &transition.transition_id).await?
    {
        if payload.request_digest != request_digest || payload.worker_handoff != worker_handoff {
            return Err(PostgresStoreError::RecordedRuntimeConflict(format!(
                "transition_id {:?} reused with different payload",
                transition.transition_id
            )));
        }
        return Ok(payload.result(event_seq, true));
    }

    let existing_run = server_run(client, tables, &transition.run_id).await?;
    if let Some(run) = &existing_run
        && (run.workflow_id != transition.workflow_id
            || run.conversation_id != transition.conversation_id)
    {
        return Err(PostgresStoreError::RecordedRuntimeConflict(format!(
            "run identity differs for {:?}",
            transition.run_id
        )));
    }
    let current = read_recorded_runtime_state(
        client,
        tables,
        &transition.run_id,
        &transition.workflow_id,
        &transition.conversation_id,
    )
    .await?;
    let current_event_seq = latest_server_run_event_seq(client, tables, &transition.run_id).await?;
    if current_event_seq != transition.expected_event_seq {
        return Err(PostgresStoreError::RecordedRuntimeConflict(format!(
            "expected_event_seq {}, current {}",
            transition.expected_event_seq, current_event_seq
        )));
    }
    let reduced = reduce_recorded_transition(current.as_ref(), &transition)?;

    if current.is_none() && existing_run.is_none() {
        let user_turn_node_id = transition.user_turn_node_id.clone().ok_or_else(|| {
            PostgresStoreError::RecordedRuntimeConflict(
                "start transition requires user_turn_node_id".to_owned(),
            )
        })?;
        create_server_run(
            client,
            tables,
            ServerRunCreate {
                run_id: transition.run_id.clone(),
                conversation_id: transition.conversation_id.clone(),
                workflow_id: transition.workflow_id.clone(),
                user_id: transition.user_id.clone(),
                user_turn_node_id,
                status: reduced.server_status.clone(),
            },
        )
        .await?;
    }

    let persisted = PersistedRecordedTransition {
        contract_version: RECORDED_RUNTIME_CONTRACT_VERSION,
        transition_id: transition.transition_id.clone(),
        request_digest,
        worker_handoff,
        worker_effect_digest: persisted_worker_effect_digest,
        reduced: reduced.clone(),
    };
    let event = append_server_run_event(
        client,
        tables,
        &transition.run_id,
        RECORDED_RUNTIME_EVENT_TYPE,
        serde_json::to_string(&persisted)?,
    )
    .await?;
    let event_seq = event.seq;
    store_runtime_current_state(client, tables, &persisted, event_seq).await?;
    replace_named_projection(
        client,
        tables,
        &runtime_checkpoint_namespace(&transition.conversation_id),
        &transition.run_id,
        runtime_projection_write(
            reduced.checkpoint.clone(),
            event_seq,
            reduced.state.status.server_status().to_owned(),
        ),
    )
    .await?;
    let status_materialization = match reduced.state.status {
        kogwistar_runtime::RecordedRunStatus::Completed => "completed",
        kogwistar_runtime::RecordedRunStatus::Failed => "failed",
        kogwistar_runtime::RecordedRunStatus::Cancelled => "cancelled",
        kogwistar_runtime::RecordedRunStatus::Suspended => "suspended",
        kogwistar_runtime::RecordedRunStatus::Running => "running",
    }
    .to_owned();
    replace_named_projection(
        client,
        tables,
        &runtime_status_namespace(&transition.conversation_id),
        &transition.run_id,
        runtime_projection_write(
            reduced.run_status.clone(),
            event_seq,
            status_materialization,
        ),
    )
    .await?;
    let prior_run = existing_run.as_ref();
    update_server_run(
        client,
        tables,
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
    )
    .await?;
    if abort_after_writes {
        return Err(PostgresStoreError::TransactionAborted(
            "requested after recorded runtime transition writes".to_owned(),
        ));
    }
    Ok(persisted.result(event_seq, false))
}

async fn replay_cursor<C>(
    client: &C,
    tables: &Tables,
    namespace: &str,
    consumer: &str,
) -> PostgresStoreResult<ReplayCursor>
where
    C: GenericClient + Sync,
{
    let row = client
        .query_opt(
            &format!(
                "SELECT last_seq FROM {} WHERE namespace = $1 AND consumer = $2",
                tables.replay_cursors
            ),
            &[&namespace, &consumer],
        )
        .await
        .map_err(backend)?;
    Ok(ReplayCursor {
        namespace: namespace.to_owned(),
        consumer: consumer.to_owned(),
        last_seq: row
            .as_ref()
            .map(|row| row.try_get(0).map_err(backend))
            .transpose()?
            .unwrap_or(0),
    })
}

async fn set_replay_cursor_legacy<C>(
    client: &C,
    tables: &Tables,
    namespace: &str,
    consumer: &str,
    last_seq: i64,
) -> PostgresStoreResult<ReplayCursor>
where
    C: GenericClient + Sync,
{
    client
        .execute(
            &format!(
                "INSERT INTO {}(namespace, consumer, last_seq, updated_at) VALUES ($1, $2, $3, NOW()) \
                 ON CONFLICT(namespace, consumer) DO UPDATE \
                 SET last_seq = EXCLUDED.last_seq, updated_at = NOW()",
                tables.replay_cursors
            ),
            &[&namespace, &consumer, &last_seq],
        )
        .await
        .map_err(backend)?;
    Ok(ReplayCursor {
        namespace: namespace.to_owned(),
        consumer: consumer.to_owned(),
        last_seq,
    })
}

async fn advisory_lock<C>(client: &C, key: &str, seed: i64) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    client
        .query_one(
            "SELECT pg_advisory_xact_lock(hashtextextended($1, $2))",
            &[&key, &seed],
        )
        .await
        .map(|_| ())
        .map_err(backend)
}

fn cursor_lock_key(namespace: &str, consumer: &str) -> String {
    format!("{namespace}\u{1f}{consumer}")
}

fn projection_lock_key(namespace: &str, key: &str) -> String {
    format!("{namespace}\u{1f}{key}")
}

fn raw_event_from_row(row: &Row) -> PostgresStoreResult<RawEntityEvent> {
    Ok(RawEntityEvent {
        namespace: row.try_get(0).map_err(backend)?,
        seq: row.try_get(1).map_err(backend)?,
        event_id: row.try_get(2).map_err(backend)?,
        entity_kind: row.try_get(3).map_err(backend)?,
        entity_id: row.try_get(4).map_err(backend)?,
        op: row.try_get(5).map_err(backend)?,
        payload_json: row.try_get(6).map_err(backend)?,
        created_at: row.try_get(7).map_err(backend)?,
    })
}

async fn enqueue_index_job<C>(
    client: &C,
    tables: &Tables,
    job: NewIndexJob,
) -> PostgresStoreResult<String>
where
    C: GenericClient + Sync,
{
    require_namespace(&job.namespace)?;
    let key = job.coalesce_key();
    let max_retries = i32::try_from(job.max_retries).map_err(|_| {
        PostgresStoreError::Backend("max_retries exceeds PostgreSQL INTEGER".to_owned())
    })?;
    let row=client.query_one(&format!("INSERT INTO {}(job_id,namespace,entity_kind,entity_id,index_kind,coalesce_key,op,status,lease_until,next_run_at,max_retries,retry_count,last_error,payload_json,created_at,updated_at,claim_token,claim_attempts) VALUES ($1,$2,$3,$4,$5,$6,$7,'PENDING',NULL,NULL,$8,0,NULL,$9,NOW(),NOW(),NULL,0) ON CONFLICT(namespace,coalesce_key) WHERE status='PENDING' DO UPDATE SET op=CASE WHEN EXCLUDED.op='DELETE' OR {}.op='DELETE' THEN 'DELETE' ELSE EXCLUDED.op END,payload_json=EXCLUDED.payload_json,updated_at=NOW() RETURNING job_id",tables.index_jobs,tables.index_jobs), &[&job.job_id,&job.namespace,&job.entity_kind,&job.entity_id,&job.index_kind,&key,&job.op,&max_retries,&job.payload_json]).await.map_err(backend)?;
    row.try_get(0).map_err(backend)
}
async fn claim_index_jobs<C>(
    client: &C,
    tables: &Tables,
    limit: usize,
    lease_seconds: i64,
    namespace: Option<&str>,
) -> PostgresStoreResult<Vec<IndexJob>>
where
    C: GenericClient + Sync,
{
    if limit == 0 {
        return Ok(Vec::new());
    };
    let limit = i64::try_from(limit).unwrap_or(i64::MAX);
    let lease_seconds = lease_seconds.to_string();
    client.execute(
        &format!("UPDATE {} SET status='FAILED',last_error='lease ownership exceeded',lease_until=NULL,claim_token=NULL,updated_at=NOW() WHERE status='DOING' AND lease_until<NOW() AND claim_attempts>=$1", tables.index_jobs),
        &[&3_i32],
    ).await.map_err(backend)?;
    let rows=client.query(&format!("WITH candidates AS (SELECT job_id FROM {} WHERE ((status='PENDING' AND (next_run_at IS NULL OR next_run_at<=NOW())) OR (status='DOING' AND lease_until IS NOT NULL AND lease_until<NOW())) AND ($1::TEXT IS NULL OR namespace=$1) ORDER BY created_at ASC,job_id ASC LIMIT $2 FOR UPDATE SKIP LOCKED) UPDATE {} j SET status='DOING',lease_until=NOW()+($3::TEXT||' seconds')::interval,claim_token=md5(random()::TEXT||clock_timestamp()::TEXT||j.job_id),claim_attempts=j.claim_attempts+CASE WHEN j.status='DOING' THEN 1 ELSE 0 END,updated_at=NOW() FROM candidates c WHERE j.job_id=c.job_id RETURNING j.job_id,j.namespace,j.entity_kind,j.entity_id,j.index_kind,j.coalesce_key,j.op,j.status,j.lease_until::TEXT,j.next_run_at::TEXT,j.max_retries,j.retry_count,j.last_error,j.payload_json,j.created_at::TEXT,j.updated_at::TEXT,j.claim_token,j.claim_attempts,j.accepted_result_json,j.accepted_result_sha256,j.accepted_at::TEXT",tables.index_jobs,tables.index_jobs), &[&namespace,&limit,&lease_seconds]).await.map_err(backend)?;
    rows.iter().map(index_job_from_row).collect()
}
async fn mark_index_job_done<C>(
    client: &C,
    tables: &Tables,
    job_id: &str,
    claim_token: Option<&str>,
) -> PostgresStoreResult<bool>
where
    C: GenericClient + Sync,
{
    Ok(client.execute(&format!("UPDATE {} SET status='DONE',lease_until=NULL,claim_token=NULL,updated_at=NOW() WHERE job_id=$1 AND status='DOING' AND ($2::TEXT IS NULL OR claim_token=$2)",tables.index_jobs), &[&job_id,&claim_token]).await.map_err(backend)?!=0)
}
async fn index_job_result<C>(client: &C, tables: &Tables, job_id: &str) -> PostgresStoreResult<Option<AcceptedIndexJobResult>>
where C: GenericClient + Sync {
    let row = client.query_opt(&format!("SELECT accepted_result_json, accepted_result_sha256, accepted_at::TEXT FROM {} WHERE job_id=$1", tables.index_jobs), &[&job_id]).await.map_err(backend)?;
    Ok(row.map(|row| AcceptedIndexJobResult {
        status: "existing".to_owned(),
        result_json: row.get(0),
        result_sha256: row.get(1),
        accepted_at: row.get::<_, Option<String>>(2).map(serde_json::Value::String),
    }))
}
async fn accept_index_job_result<C>(client: &C, tables: &Tables, job_id: &str, claim_token: &str, result_json: &str, result_sha256: &str) -> PostgresStoreResult<AcceptedIndexJobResult>
where C: GenericClient + Sync {
    if let Some(existing) = index_job_result(client, tables, job_id).await? {
        if existing.result_json.is_some() { return Ok(existing); }
    }
    let row = client.query_opt(&format!("UPDATE {} SET accepted_result_json=$1,accepted_result_sha256=$2,accepted_at=NOW() WHERE job_id=$3 AND status='DOING' AND claim_token=$4 AND (lease_until IS NULL OR lease_until>=NOW()) AND accepted_result_json IS NULL RETURNING accepted_at::TEXT", tables.index_jobs), &[&result_json, &result_sha256, &job_id, &claim_token]).await.map_err(backend)?;
    if let Some(row) = row {
        return Ok(AcceptedIndexJobResult { status: "accepted".to_owned(), result_json: Some(result_json.to_owned()), result_sha256: Some(result_sha256.to_owned()), accepted_at: row.get::<_, Option<String>>(0).map(serde_json::Value::String) });
    }
    Ok(AcceptedIndexJobResult { status: "rejected".to_owned(), result_json: None, result_sha256: None, accepted_at: None })
}
async fn mark_index_job_failed<C>(
    client: &C,
    tables: &Tables,
    job_id: &str,
    error: &str,
    final_: bool,
    claim_token: Option<&str>,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    client.execute(&format!("UPDATE {} SET status=CASE WHEN $1 THEN 'FAILED' ELSE status END,lease_until=NULL,claim_token=NULL,last_error=$2,updated_at=NOW() WHERE job_id=$3 AND status='DOING' AND ($4::TEXT IS NULL OR claim_token=$4)",tables.index_jobs), &[&final_,&truncate_error(error),&job_id,&claim_token]).await.map_err(backend)?;
    Ok(())
}
async fn bump_retry_and_requeue<C>(
    client: &C,
    tables: &Tables,
    job_id: &str,
    error: &str,
    delay: i64,
    claim_token: Option<&str>,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    let delay = delay.max(0).to_string();
    client.execute(&format!("UPDATE {} SET retry_count=retry_count+1,last_error=$1,status=CASE WHEN retry_count+1>=max_retries THEN 'FAILED' ELSE 'PENDING' END,lease_until=NULL,claim_token=NULL,next_run_at=CASE WHEN retry_count+1>=max_retries THEN NULL ELSE NOW()+($2::TEXT||' seconds')::interval END,updated_at=NOW() WHERE job_id=$3 AND status='DOING' AND ($4::TEXT IS NULL OR claim_token=$4)",tables.index_jobs), &[&truncate_error(error),&delay,&job_id,&claim_token]).await.map_err(backend)?;
    Ok(())
}
async fn renew_index_job_lease<C>(
    client: &C,
    tables: &Tables,
    job_id: &str,
    claim_token: &str,
    lease_seconds: i64,
) -> PostgresStoreResult<bool>
where
    C: GenericClient + Sync,
{
    let lease_seconds = lease_seconds.to_string();
    Ok(client.execute(&format!("UPDATE {} SET lease_until=NOW()+($1::TEXT||' seconds')::interval,updated_at=NOW() WHERE job_id=$2 AND status='DOING' AND claim_token=$3 AND (lease_until IS NULL OR lease_until>=NOW())",tables.index_jobs), &[&lease_seconds,&job_id,&claim_token]).await.map_err(backend)?!=0)
}
async fn requeue_index_job_at_tail<C>(
    client: &C,
    tables: &Tables,
    job_id: &str,
    payload_json: String,
    delay: i64,
    claim_token: Option<&str>,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    let delay = delay.max(0).to_string();
    client.execute(&format!("UPDATE {} SET status='PENDING',lease_until=NULL,claim_token=NULL,next_run_at=NOW()+($1::TEXT||' seconds')::interval,payload_json=$2,created_at=(SELECT COALESCE(MAX(created_at),NOW())+INTERVAL '1 microsecond' FROM {}),updated_at=NOW() WHERE job_id=$3 AND status='DOING' AND ($4::TEXT IS NULL OR claim_token=$4)",tables.index_jobs,tables.index_jobs), &[&delay,&payload_json,&job_id,&claim_token]).await.map_err(backend)?;
    Ok(())
}
#[allow(clippy::too_many_arguments)]
async fn list_index_jobs<C>(
    client: &C,
    tables: &Tables,
    namespace: Option<&str>,
    status: Option<&str>,
    entity_kind: Option<&str>,
    entity_id: Option<&str>,
    index_kind: Option<&str>,
    limit: usize,
) -> PostgresStoreResult<Vec<IndexJob>>
where
    C: GenericClient + Sync,
{
    let limit = i64::try_from(limit).unwrap_or(i64::MAX);
    let rows=client.query(&format!("SELECT job_id,namespace,entity_kind,entity_id,index_kind,coalesce_key,op,status,lease_until::TEXT,next_run_at::TEXT,max_retries,retry_count,last_error,payload_json,created_at::TEXT,updated_at::TEXT,claim_token,claim_attempts,accepted_result_json,accepted_result_sha256,accepted_at::TEXT FROM {} WHERE ($1::TEXT IS NULL OR namespace=$1) AND ($2::TEXT IS NULL OR status=$2) AND ($3::TEXT IS NULL OR entity_kind=$3) AND ($4::TEXT IS NULL OR entity_id=$4) AND ($5::TEXT IS NULL OR index_kind=$5) ORDER BY created_at ASC,job_id ASC LIMIT $6",tables.index_jobs), &[&namespace,&status,&entity_kind,&entity_id,&index_kind,&limit]).await.map_err(backend)?;
    rows.iter().map(index_job_from_row).collect()
}
fn index_job_from_row(row: &Row) -> PostgresStoreResult<IndexJob> {
    Ok(IndexJob {
        job_id: row.try_get(0).map_err(backend)?,
        namespace: row.try_get(1).map_err(backend)?,
        entity_kind: row.try_get(2).map_err(backend)?,
        entity_id: row.try_get(3).map_err(backend)?,
        index_kind: row.try_get(4).map_err(backend)?,
        coalesce_key: row.try_get(5).map_err(backend)?,
        op: row.try_get(6).map_err(backend)?,
        status: row.try_get(7).map_err(backend)?,
        lease_until: row
            .try_get::<_, Option<String>>(8)
            .map_err(backend)?
            .map(serde_json::Value::String),
        next_run_at: row
            .try_get::<_, Option<String>>(9)
            .map_err(backend)?
            .map(serde_json::Value::String),
        max_retries: i64::from(row.try_get::<_, i32>(10).map_err(backend)?),
        retry_count: i64::from(row.try_get::<_, i32>(11).map_err(backend)?),
        last_error: row.try_get(12).map_err(backend)?,
        payload_json: row.try_get(13).map_err(backend)?,
        created_at: serde_json::Value::String(row.try_get(14).map_err(backend)?),
        updated_at: serde_json::Value::String(row.try_get(15).map_err(backend)?),
        claim_token: row.try_get(16).map_err(backend)?,
        claim_attempts: i64::from(row.try_get::<_, i32>(17).map_err(backend)?),
        accepted_result_json: row.try_get(18).map_err(backend)?,
        accepted_result_sha256: row.try_get(19).map_err(backend)?,
        accepted_at: row.try_get::<_, Option<String>>(20).map_err(backend)?.map(serde_json::Value::String),
    })
}
fn truncate_error(error: &str) -> String {
    error.chars().take(2000).collect()
}

const LANE_SELECT: &str = "message_id,namespace,purpose,inbox_id,conversation_id,recipient_id,sender_id,msg_type,status,seq,conversation_seq,claimed_by,lease_until::TEXT,retry_count,created_at,available_at,run_id,step_id,correlation_id,payload_json,error_json,prev_message_id,next_message_id,inbox_tail_message_id,conversation_tail_message_id";
const LANE_RETURNING: &str = "x.message_id,x.namespace,x.purpose,x.inbox_id,x.conversation_id,x.recipient_id,x.sender_id,x.msg_type,x.status,x.seq,x.conversation_seq,x.claimed_by,x.lease_until::TEXT,x.retry_count,x.created_at,x.available_at,x.run_id,x.step_id,x.correlation_id,x.payload_json,x.error_json,x.prev_message_id,x.next_message_id,x.inbox_tail_message_id,x.conversation_tail_message_id";
fn lane_from_row(row: &Row) -> PostgresStoreResult<ProjectedLaneMessage> {
    Ok(ProjectedLaneMessage {
        message_id: row.try_get(0).map_err(backend)?,
        namespace: row.try_get(1).map_err(backend)?,
        purpose: row.try_get(2).map_err(backend)?,
        inbox_id: row.try_get(3).map_err(backend)?,
        conversation_id: row.try_get(4).map_err(backend)?,
        recipient_id: row.try_get(5).map_err(backend)?,
        sender_id: row.try_get(6).map_err(backend)?,
        msg_type: row.try_get(7).map_err(backend)?,
        status: row.try_get(8).map_err(backend)?,
        seq: row.try_get(9).map_err(backend)?,
        conversation_seq: row.try_get(10).map_err(backend)?,
        claimed_by: row.try_get(11).map_err(backend)?,
        lease_until: row
            .try_get::<_, Option<String>>(12)
            .map_err(backend)?
            .map(serde_json::Value::String),
        retry_count: i64::from(row.try_get::<_, i32>(13).map_err(backend)?),
        created_at: row.try_get(14).map_err(backend)?,
        available_at: row.try_get(15).map_err(backend)?,
        run_id: row.try_get(16).map_err(backend)?,
        step_id: row.try_get(17).map_err(backend)?,
        correlation_id: row.try_get(18).map_err(backend)?,
        payload_json: row.try_get(19).map_err(backend)?,
        error_json: row.try_get(20).map_err(backend)?,
        prev_message_id: row.try_get(21).map_err(backend)?,
        next_message_id: row.try_get(22).map_err(backend)?,
        inbox_tail_message_id: row.try_get(23).map_err(backend)?,
        conversation_tail_message_id: row.try_get(24).map_err(backend)?,
    })
}
async fn project_lane_message<C>(
    c: &C,
    t: &Tables,
    row: NewProjectedLaneMessage,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    require_namespace(&row.namespace)?;
    advisory_lock(
        c,
        &format!("lane-inbox\u{1f}{}\u{1f}{}", row.namespace, row.inbox_id),
        31,
    )
    .await?;
    advisory_lock(
        c,
        &format!(
            "lane-conversation\u{1f}{}\u{1f}{}",
            row.namespace, row.conversation_id
        ),
        32,
    )
    .await?;
    c.execute(&format!("INSERT INTO {}(message_id,namespace,purpose,inbox_id,conversation_id,recipient_id,sender_id,msg_type,status,seq,conversation_seq,claimed_by,lease_until,retry_count,created_at,available_at,run_id,step_id,correlation_id,payload_json,error_json,prev_message_id,next_message_id,inbox_tail_message_id,conversation_tail_message_id) SELECT $1,$2,$3,$4,$5,$6,$7,$8,$9,COALESCE((SELECT MAX(seq)+1 FROM {} WHERE namespace=$2 AND inbox_id=$4),1),COALESCE((SELECT MAX(conversation_seq)+1 FROM {} WHERE namespace=$2 AND conversation_id=$5),1),NULL,NULL,0,$10,$11,$12,$13,$14,$15,$16,(SELECT message_id FROM {} WHERE namespace=$2 AND inbox_id=$4 ORDER BY seq DESC,created_at DESC LIMIT 1),NULL,$1,$1 ON CONFLICT(message_id) DO NOTHING",t.projected_lane_messages,t.projected_lane_messages,t.projected_lane_messages,t.projected_lane_messages),&[&row.message_id,&row.namespace,&if row.purpose.is_empty(){"user_visible".to_owned()}else{row.purpose},&row.inbox_id,&row.conversation_id,&row.recipient_id,&row.sender_id,&row.msg_type,&row.status,&row.created_at,&row.available_at,&row.run_id,&row.step_id,&row.correlation_id,&row.payload_json,&row.error_json]).await.map_err(backend)?;
    Ok(())
}
async fn projected_lane_message<C>(
    c: &C,
    t: &Tables,
    id: &str,
) -> PostgresStoreResult<Option<ProjectedLaneMessage>>
where
    C: GenericClient + Sync,
{
    c.query_opt(
        &format!(
            "SELECT {LANE_SELECT} FROM {} WHERE message_id=$1",
            t.projected_lane_messages
        ),
        &[&id],
    )
    .await
    .map_err(backend)?
    .map(|r| lane_from_row(&r))
    .transpose()
}
async fn update_projected_lane_message_status<C>(
    c: &C,
    t: &Tables,
    id: &str,
    status: &str,
    error: Option<String>,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    c.execute(&format!("UPDATE {} SET status=$1,error_json=COALESCE($2,error_json),claimed_by=CASE WHEN $1 IN ('completed','failed','cancelled') THEN NULL ELSE claimed_by END,lease_until=CASE WHEN $1 IN ('completed','failed','cancelled') THEN NULL ELSE lease_until END WHERE message_id=$3",t.projected_lane_messages),&[&status,&error,&id]).await.map_err(backend)?;
    Ok(())
}
async fn update_projected_lane_message_links<C>(
    c: &C,
    t: &Tables,
    id: &str,
    prev: Option<String>,
    next: Option<String>,
    inbox_tail: Option<String>,
    conversation_tail: Option<String>,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    c.execute(&format!("UPDATE {} SET prev_message_id=$1,next_message_id=$2,inbox_tail_message_id=$3,conversation_tail_message_id=$4 WHERE message_id=$5",t.projected_lane_messages),&[&prev,&next,&inbox_tail,&conversation_tail,&id]).await.map_err(backend)?;
    Ok(())
}
async fn clear_projected_lane_messages<C>(
    c: &C,
    t: &Tables,
    namespace: &str,
) -> PostgresStoreResult<u64>
where
    C: GenericClient + Sync,
{
    c.execute(
        &format!(
            "DELETE FROM {} WHERE namespace=$1",
            t.projected_lane_messages
        ),
        &[&namespace],
    )
    .await
    .map_err(backend)
}
async fn claim_projected_lane_messages<C>(
    c: &C,
    t: &Tables,
    namespace: &str,
    inbox: &str,
    owner: &str,
    limit: usize,
    lease: i64,
) -> PostgresStoreResult<Vec<ProjectedLaneMessage>>
where
    C: GenericClient + Sync,
{
    if limit == 0 {
        return Ok(vec![]);
    }
    let limit = i64::try_from(limit).unwrap_or(i64::MAX);
    let lease = lease.to_string();
    let rows=c.query(&format!("WITH picked AS (SELECT message_id FROM {} WHERE namespace=$1 AND inbox_id=$2 AND ((status='pending' AND available_at<=EXTRACT(EPOCH FROM NOW())::BIGINT) OR (status='claimed' AND lease_until IS NOT NULL AND lease_until<NOW())) ORDER BY seq ASC,created_at ASC LIMIT $3 FOR UPDATE SKIP LOCKED) UPDATE {} x SET status='claimed',claimed_by=$4,lease_until=NOW()+($5::TEXT||' seconds')::interval FROM picked WHERE x.message_id=picked.message_id RETURNING {}",t.projected_lane_messages,t.projected_lane_messages,LANE_RETURNING),&[&namespace,&inbox,&limit,&owner,&lease]).await.map_err(backend)?;
    rows.iter().map(lane_from_row).collect()
}
#[allow(clippy::too_many_arguments)]
async fn claim_projected_lane_messages_for_run<C>(
    c: &C,
    t: &Tables,
    namespace: &str,
    inbox: &str,
    run_id: &str,
    owner: &str,
    limit: usize,
    lease: i64,
) -> PostgresStoreResult<Vec<ProjectedLaneMessage>>
where
    C: GenericClient + Sync,
{
    if limit == 0 {
        return Ok(vec![]);
    }
    let limit = i64::try_from(limit).unwrap_or(i64::MAX);
    let lease = lease.to_string();
    let rows=c.query(&format!("WITH picked AS (SELECT message_id FROM {} WHERE namespace=$1 AND inbox_id=$2 AND run_id=$3 AND ((status='pending' AND available_at<=EXTRACT(EPOCH FROM NOW())::BIGINT) OR (status='claimed' AND lease_until IS NOT NULL AND lease_until<NOW())) ORDER BY seq ASC,created_at ASC LIMIT $4 FOR UPDATE SKIP LOCKED) UPDATE {} x SET status='claimed',claimed_by=$5,lease_until=NOW()+($6::TEXT||' seconds')::interval FROM picked WHERE x.message_id=picked.message_id RETURNING {}",t.projected_lane_messages,t.projected_lane_messages,LANE_RETURNING),&[&namespace,&inbox,&run_id,&limit,&owner,&lease]).await.map_err(backend)?;
    rows.iter().map(lane_from_row).collect()
}
async fn ack_projected_lane_message<C>(
    c: &C,
    t: &Tables,
    id: &str,
    owner: &str,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    c.execute(&format!("UPDATE {} SET status='completed',claimed_by=NULL,lease_until=NULL WHERE message_id=$1 AND status NOT IN ('completed','failed','cancelled','dead-letter') AND (claimed_by IS NULL OR claimed_by=$2)",t.projected_lane_messages),&[&id,&owner]).await.map_err(backend)?;
    Ok(())
}
async fn requeue_projected_lane_message<C>(
    c: &C,
    t: &Tables,
    id: &str,
    owner: &str,
    error: Option<String>,
    delay: i64,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    let delay = delay.max(0).to_string();
    c.execute(&format!("UPDATE {} SET status='pending',claimed_by=NULL,lease_until=NULL,retry_count=retry_count+1,available_at=EXTRACT(EPOCH FROM(NOW()+($1::TEXT||' seconds')::interval))::BIGINT,error_json=COALESCE($2,error_json) WHERE message_id=$3 AND status NOT IN ('completed','failed','cancelled','dead-letter') AND (claimed_by IS NULL OR claimed_by=$4)",t.projected_lane_messages),&[&delay,&error,&id,&owner]).await.map_err(backend)?;
    Ok(())
}
async fn dead_letter_projected_lane_message<C>(
    c: &C,
    t: &Tables,
    id: &str,
    owner: &str,
    error: Option<String>,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    c.execute(&format!("UPDATE {} SET status='dead-letter',claimed_by=NULL,lease_until=NULL,error_json=COALESCE($1,error_json) WHERE message_id=$2 AND status NOT IN ('completed','failed','cancelled','dead-letter') AND (claimed_by IS NULL OR claimed_by=$3)",t.projected_lane_messages),&[&error,&id,&owner]).await.map_err(backend)?;
    Ok(())
}
async fn repair_orphaned_claimed_lane_messages<C>(
    client: &C,
    tables: &Tables,
    namespace: &str,
    inbox_id: Option<&str>,
    limit: usize,
) -> PostgresStoreResult<Vec<String>>
where
    C: GenericClient + Sync,
{
    if limit == 0 {
        return Ok(Vec::new());
    }
    let limit = i64::try_from(limit).unwrap_or(i64::MAX);
    let rows = client
        .query(
            &format!(
                "SELECT message_id FROM {} WHERE namespace=$1 \
                 AND ($2::TEXT IS NULL OR inbox_id=$2) AND status='claimed' \
                 AND (lease_until IS NULL OR lease_until<=NOW()) \
                 ORDER BY seq ASC, created_at ASC LIMIT $3 FOR UPDATE SKIP LOCKED",
                tables.projected_lane_messages
            ),
            &[&namespace, &inbox_id, &limit],
        )
        .await
        .map_err(backend)?;
    let ids = rows
        .iter()
        .map(|row| row.try_get::<_, String>(0).map_err(backend))
        .collect::<PostgresStoreResult<Vec<_>>>()?;
    for id in &ids {
        client
            .execute(
                &format!(
                    "UPDATE {} SET status='pending', claimed_by=NULL, lease_until=NULL \
                     WHERE message_id=$1 AND status='claimed' \
                     AND (lease_until IS NULL OR lease_until<=NOW())",
                    tables.projected_lane_messages
                ),
                &[id],
            )
            .await
            .map_err(backend)?;
    }
    Ok(ids)
}
async fn list_projected_lane_messages<C>(
    c: &C,
    t: &Tables,
    f: &LaneMessageFilter,
) -> PostgresStoreResult<Vec<ProjectedLaneMessage>>
where
    C: GenericClient + Sync,
{
    let limit = i64::try_from(f.limit).unwrap_or(i64::MAX);
    let order = if f.newest_first { "DESC" } else { "ASC" };
    let rows=c.query(&format!("SELECT {LANE_SELECT} FROM {} WHERE ($1::TEXT IS NULL OR namespace=$1) AND ($2::TEXT IS NULL OR purpose=$2) AND ($3::TEXT IS NULL OR inbox_id=$3) AND ($4::TEXT IS NULL OR conversation_id=$4) AND ($5::TEXT IS NULL OR status=$5) AND ($6::TEXT IS NULL OR msg_type=$6) AND ($7::TEXT IS NULL OR sender_id=$7) AND ($8::TEXT IS NULL OR recipient_id=$8) AND ($9::TEXT IS NULL OR correlation_id=$9) AND ($10::BIGINT IS NULL OR created_at>=$10) AND ($11::BIGINT IS NULL OR created_at<=$11) AND ($12::BIGINT IS NULL OR available_at>=$12) AND ($13::BIGINT IS NULL OR available_at<=$13) ORDER BY created_at {order},seq {order},message_id {order} LIMIT $14",t.projected_lane_messages),&[&f.namespace,&f.purpose,&f.inbox_id,&f.conversation_id,&f.status,&f.msg_type,&f.sender_id,&f.recipient_id,&f.correlation_id,&f.created_at_gte,&f.created_at_lte,&f.available_at_gte,&f.available_at_lte,&limit]).await.map_err(backend)?;
    rows.iter()
        .map(lane_from_row)
        .collect::<PostgresStoreResult<Vec<_>>>()
}

fn workflow_design_snapshot_from_row(row: &Row) -> PostgresStoreResult<WorkflowDesignSnapshot> {
    Ok(WorkflowDesignSnapshot {
        workflow_id: row.try_get(0).map_err(backend)?,
        version: row.try_get(1).map_err(backend)?,
        seq: row.try_get(2).map_err(backend)?,
        payload_json: row.try_get(3).map_err(backend)?,
        schema_version: i64::from(row.try_get::<_, i32>(4).map_err(backend)?),
        created_at_ms: row.try_get(5).map_err(backend)?,
    })
}

fn workflow_design_delta_from_row(row: &Row) -> PostgresStoreResult<WorkflowDesignDelta> {
    Ok(WorkflowDesignDelta {
        workflow_id: row.try_get(0).map_err(backend)?,
        version: row.try_get(1).map_err(backend)?,
        prev_version: row.try_get(2).map_err(backend)?,
        target_seq: row.try_get(3).map_err(backend)?,
        forward_json: row.try_get(4).map_err(backend)?,
        inverse_json: row.try_get(5).map_err(backend)?,
        schema_version: i64::from(row.try_get::<_, i32>(6).map_err(backend)?),
        created_at_ms: row.try_get(7).map_err(backend)?,
    })
}
fn server_run_from_row(row: &Row) -> PostgresStoreResult<ServerRun> {
    Ok(ServerRun {
        run_id: row.try_get(0).map_err(backend)?,
        conversation_id: row.try_get(1).map_err(backend)?,
        workflow_id: row.try_get(2).map_err(backend)?,
        user_id: row.try_get(3).map_err(backend)?,
        user_turn_node_id: row.try_get(4).map_err(backend)?,
        assistant_turn_node_id: row.try_get(5).map_err(backend)?,
        status: row.try_get(6).map_err(backend)?,
        cancel_requested: row.try_get::<_, i32>(7).map_err(backend)? != 0,
        result_json: row.try_get(8).map_err(backend)?,
        error_json: row.try_get(9).map_err(backend)?,
        created_at_ms: row.try_get(10).map_err(backend)?,
        updated_at_ms: row.try_get(11).map_err(backend)?,
        started_at_ms: row.try_get(12).map_err(backend)?,
        finished_at_ms: row.try_get(13).map_err(backend)?,
    })
}
fn server_run_event_from_row(row: &Row) -> PostgresStoreResult<ServerRunEvent> {
    Ok(ServerRunEvent {
        seq: row.try_get(0).map_err(backend)?,
        run_id: row.try_get(1).map_err(backend)?,
        event_type: row.try_get(2).map_err(backend)?,
        payload_json: row.try_get(3).map_err(backend)?,
        created_at_ms: row.try_get(4).map_err(backend)?,
    })
}

fn raw_to_entity_event(raw: RawEntityEvent) -> PostgresStoreResult<EntityEvent> {
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

fn require_namespace(namespace: &str) -> PostgresStoreResult<()> {
    if namespace.is_empty() {
        Err(PostgresStoreError::EmptyNamespace)
    } else {
        Ok(())
    }
}

fn require_trait_namespace(namespace: &str) -> StoreResult<()> {
    if namespace.is_empty() {
        Err(StoreError::EmptyNamespace)
    } else {
        Ok(())
    }
}

fn trait_error(error: PostgresStoreError) -> StoreError {
    match error {
        PostgresStoreError::Store(error) => error,
        PostgresStoreError::EmptyNamespace => StoreError::EmptyNamespace,
        PostgresStoreError::EventIdNamespaceCollision {
            event_id,
            existing_namespace,
            requested_namespace,
        } => StoreError::EventIdNamespaceCollision {
            event_id,
            existing_namespace,
            requested_namespace,
        },
        error => StoreError::Backend {
            backend: "postgres".to_owned(),
            message: error.to_string(),
        },
    }
}

fn backend(error: impl std::fmt::Display) -> PostgresStoreError {
    PostgresStoreError::Backend(error.to_string())
}

/// Strict ASCII PostgreSQL identifier validation, then always quote it.
pub fn quote_identifier(identifier: &str) -> PostgresStoreResult<String> {
    let mut bytes = identifier.bytes();
    let Some(first) = bytes.next() else {
        return Err(PostgresStoreError::InvalidSchema {
            schema: identifier.to_owned(),
        });
    };
    if !(first.is_ascii_alphabetic() || first == b'_')
        || !bytes.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
    {
        return Err(PostgresStoreError::InvalidSchema {
            schema: identifier.to_owned(),
        });
    }
    Ok(format!("\"{identifier}\""))
}

fn qualified(schema: &str, relation: &str) -> PostgresStoreResult<String> {
    Ok(format!("{schema}.{}", quote_identifier(relation)?))
}

fn schema_sql(tables: &Tables) -> String {
    format!(
        "CREATE SCHEMA IF NOT EXISTS {schema};\
         CREATE TABLE IF NOT EXISTS {global_seq} (value BIGINT NOT NULL);\
         INSERT INTO {global_seq}(value) SELECT 0 WHERE NOT EXISTS (SELECT 1 FROM {global_seq});\
         CREATE TABLE IF NOT EXISTS {user_seq} (user_id TEXT PRIMARY KEY, value BIGINT NOT NULL);\
         CREATE TABLE IF NOT EXISTS {index_applied_state} (\
            namespace TEXT NOT NULL DEFAULT 'default',coalesce_key TEXT NOT NULL,\
            applied_fingerprint TEXT NULL,applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),\
            last_job_id TEXT NULL,PRIMARY KEY(namespace, coalesce_key)\
         );\
         CREATE INDEX IF NOT EXISTS {index_applied_state_key_index}\
            ON {index_applied_state}(coalesce_key);\
         CREATE TABLE IF NOT EXISTS {namespace_seq} (\
            namespace TEXT PRIMARY KEY, next_seq BIGINT NOT NULL\
         );\
         INSERT INTO {namespace_seq}(namespace, next_seq) VALUES ('default', 1)\
            ON CONFLICT(namespace) DO NOTHING;\
         CREATE TABLE IF NOT EXISTS {entity_events} (\
            namespace TEXT NOT NULL DEFAULT 'default',\
            seq BIGINT NOT NULL,\
            event_id TEXT NOT NULL,\
            entity_kind TEXT NOT NULL,\
            entity_id TEXT NOT NULL,\
            op TEXT NOT NULL,\
            payload_json TEXT NOT NULL,\
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),\
            PRIMARY KEY(namespace, seq),\
            UNIQUE(event_id)\
         );\
         CREATE INDEX IF NOT EXISTS {aggregate_index}\
            ON {entity_events}(namespace, entity_kind, entity_id, seq);\
         CREATE TABLE IF NOT EXISTS {replay_cursors} (\
            namespace TEXT NOT NULL DEFAULT 'default',\
            consumer TEXT NOT NULL,\
            last_seq BIGINT NOT NULL,\
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),\
            PRIMARY KEY(namespace, consumer)\
         );\
         CREATE TABLE IF NOT EXISTS {named_projections} (\
            namespace TEXT NOT NULL,\
            key TEXT NOT NULL,\
            payload_json TEXT NOT NULL,\
            last_authoritative_seq BIGINT NOT NULL,\
            last_materialized_seq BIGINT NOT NULL,\
            projection_schema_version INTEGER NOT NULL,\
            materialization_status TEXT NOT NULL,\
            updated_at_ms BIGINT NOT NULL,\
            PRIMARY KEY(namespace, key)\
         );\
         CREATE INDEX IF NOT EXISTS {named_projections_namespace_index}\
            ON {named_projections}(namespace, updated_at_ms);\
         CREATE TABLE IF NOT EXISTS {workflow_design_snapshots} (\
            workflow_id TEXT NOT NULL,\
            version BIGINT NOT NULL,\
            seq BIGINT NOT NULL,\
            payload_json TEXT NOT NULL,\
            schema_version INTEGER NOT NULL,\
            created_at_ms BIGINT NOT NULL,\
            PRIMARY KEY(workflow_id, version)\
         );\
         CREATE TABLE IF NOT EXISTS {workflow_design_version_deltas} (\
            workflow_id TEXT NOT NULL,\
            version BIGINT NOT NULL,\
            prev_version BIGINT NOT NULL,\
            target_seq BIGINT NOT NULL,\
            forward_json TEXT NOT NULL,\
            inverse_json TEXT NOT NULL,\
            schema_version INTEGER NOT NULL,\
            created_at_ms BIGINT NOT NULL,\
            PRIMARY KEY(workflow_id, version)\
         );\
         CREATE TABLE IF NOT EXISTS {server_runs} (\
            run_id TEXT PRIMARY KEY,conversation_id TEXT NOT NULL,workflow_id TEXT NOT NULL,\
            user_id TEXT NULL,user_turn_node_id TEXT NULL,assistant_turn_node_id TEXT NULL,\
            status TEXT NOT NULL,cancel_requested INTEGER NOT NULL DEFAULT 0,\
            result_json TEXT NULL,error_json TEXT NULL,created_at_ms BIGINT NOT NULL,\
            updated_at_ms BIGINT NOT NULL,started_at_ms BIGINT NULL,finished_at_ms BIGINT NULL\
         );\
         CREATE TABLE IF NOT EXISTS {server_run_events} (\
            seq BIGSERIAL PRIMARY KEY,run_id TEXT NOT NULL,event_type TEXT NOT NULL,\
            payload_json TEXT NOT NULL,created_at_ms BIGINT NOT NULL\
         );\
         CREATE INDEX IF NOT EXISTS {server_runs_status_index} ON {server_runs}(status, updated_at_ms);\
         CREATE INDEX IF NOT EXISTS {server_run_events_run_seq_index} ON {server_run_events}(run_id, seq);\
         CREATE UNIQUE INDEX IF NOT EXISTS {server_run_events_transition_id_index}\
            ON {server_run_events} ((payload_json::jsonb->>'transition_id'))\
            WHERE event_type='workflow.recorded_transition.v1';\
         CREATE TABLE IF NOT EXISTS {index_jobs} (\
            job_id TEXT PRIMARY KEY,namespace TEXT NOT NULL DEFAULT 'default',entity_kind TEXT NOT NULL,entity_id TEXT NOT NULL,index_kind TEXT NOT NULL,coalesce_key TEXT NOT NULL,op TEXT NOT NULL,status TEXT NOT NULL,lease_until TIMESTAMPTZ NULL,next_run_at TIMESTAMPTZ NULL,max_retries INTEGER NOT NULL DEFAULT 10,retry_count INTEGER NOT NULL DEFAULT 0,last_error TEXT NULL,payload_json TEXT NULL,created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),claim_token TEXT NULL,claim_attempts INTEGER NOT NULL DEFAULT 0,accepted_result_json TEXT NULL,accepted_result_sha256 TEXT NULL,accepted_at TIMESTAMPTZ NULL\
         );\
         ALTER TABLE {index_jobs} ADD COLUMN IF NOT EXISTS claim_token TEXT NULL;\
         ALTER TABLE {index_jobs} ADD COLUMN IF NOT EXISTS claim_attempts INTEGER NOT NULL DEFAULT 0;\
         ALTER TABLE {index_jobs} ADD COLUMN IF NOT EXISTS accepted_result_json TEXT NULL;\
         ALTER TABLE {index_jobs} ADD COLUMN IF NOT EXISTS accepted_result_sha256 TEXT NULL;\
         ALTER TABLE {index_jobs} ADD COLUMN IF NOT EXISTS accepted_at TIMESTAMPTZ NULL;\
         CREATE INDEX IF NOT EXISTS {index_jobs_status_lease_index} ON {index_jobs}(status, lease_until);\
         CREATE INDEX IF NOT EXISTS {index_jobs_status_next_run_index} ON {index_jobs}(status, next_run_at);\
         CREATE INDEX IF NOT EXISTS {index_jobs_entity_index} ON {index_jobs}(entity_kind, entity_id, index_kind);\
         CREATE INDEX IF NOT EXISTS {index_jobs_namespace_index} ON {index_jobs}(namespace);\
         CREATE UNIQUE INDEX IF NOT EXISTS {index_jobs_pending_index} ON {index_jobs}(namespace, coalesce_key) WHERE status='PENDING';\
         CREATE TABLE IF NOT EXISTS {projected_lane_messages} (\
            message_id TEXT PRIMARY KEY,namespace TEXT NOT NULL DEFAULT 'default',purpose TEXT NOT NULL DEFAULT 'user_visible',\
            inbox_id TEXT NOT NULL,conversation_id TEXT NOT NULL,recipient_id TEXT NOT NULL,sender_id TEXT NOT NULL,\
            msg_type TEXT NOT NULL,status TEXT NOT NULL,seq BIGINT NOT NULL,conversation_seq BIGINT NOT NULL,\
            claimed_by TEXT NULL,lease_until TIMESTAMPTZ NULL,retry_count INTEGER NOT NULL DEFAULT 0,created_at BIGINT NOT NULL,\
            available_at BIGINT NOT NULL,run_id TEXT NULL,step_id TEXT NULL,correlation_id TEXT NULL,payload_json TEXT NULL,error_json TEXT NULL,\
            prev_message_id TEXT NULL,next_message_id TEXT NULL,inbox_tail_message_id TEXT NULL,conversation_tail_message_id TEXT NULL\
         );\
         CREATE INDEX IF NOT EXISTS {lane_messages_namespace_inbox_seq_index} ON {projected_lane_messages}(namespace,inbox_id,seq);\
         CREATE INDEX IF NOT EXISTS {lane_messages_claim_index} ON {projected_lane_messages}(namespace,inbox_id,status,available_at,lease_until);\
         CREATE INDEX IF NOT EXISTS {lane_messages_conversation_seq_index} ON {projected_lane_messages}(namespace,conversation_id,conversation_seq);",
        schema = tables.quoted_schema,
        global_seq = tables.global_seq,
        user_seq = tables.user_seq,
        index_applied_state = tables.index_applied_state,
        index_applied_state_key_index = tables.index_applied_state_key_index,
        namespace_seq = tables.namespace_seq,
        entity_events = tables.entity_events,
        aggregate_index = tables.aggregate_index,
        replay_cursors = tables.replay_cursors,
        named_projections = tables.named_projections,
        named_projections_namespace_index = tables.named_projections_namespace_index,
        workflow_design_snapshots = tables.workflow_design_snapshots,
        workflow_design_version_deltas = tables.workflow_design_version_deltas,
        server_runs = tables.server_runs,
        server_run_events = tables.server_run_events,
        server_runs_status_index = tables.server_runs_status_index,
        server_run_events_run_seq_index = tables.server_run_events_run_seq_index,
        server_run_events_transition_id_index = tables.server_run_events_transition_id_index,
        index_jobs = tables.index_jobs,
        index_jobs_status_lease_index = tables.index_jobs_status_lease_index,
        index_jobs_status_next_run_index = tables.index_jobs_status_next_run_index,
        index_jobs_entity_index = tables.index_jobs_entity_index,
        index_jobs_namespace_index = tables.index_jobs_namespace_index,
        index_jobs_pending_index = tables.index_jobs_pending_index,
        projected_lane_messages = tables.projected_lane_messages,
        lane_messages_namespace_inbox_seq_index = tables.lane_messages_namespace_inbox_seq_index,
        lane_messages_claim_index = tables.lane_messages_claim_index,
        lane_messages_conversation_seq_index = tables.lane_messages_conversation_seq_index,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use kogwistar_store::{EventReadStore, EventWriteStore};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::sync::Notify;

    static NEXT_SCHEMA: AtomicUsize = AtomicUsize::new(0);

    fn test_schema(label: &str) -> String {
        format!(
            "kogwistar_phase3_{}_{}_{}",
            label,
            std::process::id(),
            NEXT_SCHEMA.fetch_add(1, Ordering::Relaxed)
        )
    }

    fn raw(id: &str, payload_json: &str) -> NewRawEntityEvent {
        NewRawEntityEvent {
            event_id: id.to_owned(),
            entity_kind: "node".to_owned(),
            entity_id: "n".to_owned(),
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
            "parent_token_id": null,
            "initial_state": {"answer": "seed"},
            "frontier": {
                "pending": [["step-1", 0, "token-1", null]],
                "suspended": [],
                "join_node_ids": [],
                "join_outstanding": [],
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
            "parent_token_id": null,
            "state_update": [["u", {"answer": answer}]],
            "frontier": {
                "pending": [],
                "suspended": [],
                "join_node_ids": [],
                "join_outstanding": [],
                "join_waiters": {}
            },
            "result": {"answer": answer}
        }))
        .unwrap()
    }

    fn runtime_failure_start(run_id: &str, transition_id: &str) -> RecordedRuntimeTransition {
        let mut start = runtime_start(run_id, transition_id);
        start.initial_state.as_mut().unwrap().insert(
            "_rt_routes".to_owned(),
            serde_json::json!([{
                "edge_id": "recover-edge",
                "source_node_id": "step-1",
                "target_node_id": "recover",
                "aliases": ["recover"],
                "join_mask": 0,
                "predicate": "on_failure",
                "multiplicity": "one",
                "is_default": false,
                "priority": 100,
                "source_fanout": false
            }]),
        );
        start
    }

    fn runtime_lane(run_id: &str, message_id: &str) -> NewProjectedLaneMessage {
        NewProjectedLaneMessage {
            message_id: message_id.to_owned(),
            namespace: "runtime".to_owned(),
            purpose: "user_visible".to_owned(),
            inbox_id: "python-workers".to_owned(),
            conversation_id: "conv-1".to_owned(),
            recipient_id: "python-worker".to_owned(),
            sender_id: "rust-runtime".to_owned(),
            msg_type: "workflow.worker.request.v1".to_owned(),
            status: "pending".to_owned(),
            created_at: 1,
            available_at: 0,
            run_id: Some(run_id.to_owned()),
            step_id: Some("step-1".to_owned()),
            correlation_id: Some("corr-1".to_owned()),
            payload_json: Some("{}".to_owned()),
            error_json: None,
        }
    }

    fn runtime_handoff(run_id: &str, message_id: &str, owner: &str) -> RecordedWorkerHandoff {
        RecordedWorkerHandoff {
            message_id: message_id.to_owned(),
            claimed_by: owner.to_owned(),
            run_id: run_id.to_owned(),
            step_id: "step-1".to_owned(),
            correlation_id: "corr-1".to_owned(),
        }
    }

    fn runtime_worker_lane(run_id: &str, message_id: &str) -> NewProjectedLaneMessage {
        let mut lane = runtime_lane(run_id, message_id);
        lane.correlation_id = Some(run_id.to_owned());
        lane.payload_json = Some(
            serde_json::json!({
                "workflow_id": "wf-1",
                "conversation_id": "conv-1",
                "token_id": "token-1",
                "parent_token_id": null
            })
            .to_string(),
        );
        lane
    }

    fn runtime_worker_handoff(
        run_id: &str,
        message_id: &str,
        owner: &str,
    ) -> RecordedWorkerHandoff {
        let mut handoff = runtime_handoff(run_id, message_id, owner);
        handoff.correlation_id = run_id.to_owned();
        handoff
    }

    fn runtime_failure_effect(
        effect_id: &str,
        route_next: &[&str],
        successors: Value,
    ) -> RecordedWorkerSuccessEffect {
        serde_json::from_value(serde_json::json!({
            "contract_version": 1,
            "effect_id": effect_id,
            "status": "failed",
            "state_update": [["u", {"attempted": true}]],
            "successors": successors,
            "route_next": route_next,
            "result": {"workflow_status": "failed"},
            "errors": ["boom"]
        }))
        .unwrap()
    }

    #[test]
    fn identifier_validation_and_qualification_are_strict() {
        assert_eq!(quote_identifier("Mixed_Name9").unwrap(), "\"Mixed_Name9\"");
        for invalid in ["", "9name", "has-dash", "has space", "public;DROP"] {
            assert!(matches!(
                quote_identifier(invalid),
                Err(PostgresStoreError::InvalidSchema { .. })
            ));
        }
        let tables = Tables::new("CamelCase").unwrap();
        assert_eq!(tables.entity_events, "\"CamelCase\".\"entity_events\"");
        assert!(!tables.entity_events.contains("search_path"));
    }

    #[test]
    fn graph_read_filter_accepts_only_legacy_default_scope_rows() {
        let default_scope = kogwistar_store::GraphScope {
            namespace: "default".to_owned(),
            workspace_id: None,
            graph_space: None,
        };
        let (sql, values) = graph_filter_sql(
            &default_scope,
            &kogwistar_store::MetadataFilter::default(),
            1,
        )
        .unwrap();
        assert_eq!(values, vec!["namespace"]);
        assert!(sql.contains("metadata -> $1::TEXT"));
        assert!(sql.contains("NOT (metadata ? 'namespace')"));

        let scoped = kogwistar_store::GraphScope {
            namespace: "default".to_owned(),
            workspace_id: Some("workspace".to_owned()),
            graph_space: None,
        };
        let (sql, values) =
            graph_filter_sql(&scoped, &kogwistar_store::MetadataFilter::default(), 1).unwrap();
        assert_eq!(values, vec!["namespace", "workspace_id"]);
        assert!(!sql.contains("NOT (metadata"));
    }

    #[test]
    fn sql_objects_are_schema_qualified_and_python_shaped() {
        let tables = Tables::new("events").unwrap();
        assert_eq!(tables.namespace_seq, "\"events\".\"namespace_seq\"");
        assert_eq!(tables.replay_cursors, "\"events\".\"replay_cursors\"");
        assert_eq!(tables.aggregate_index, "\"idx_entity_events_aggregate\"");
        assert_eq!(cursor_lock_key("a", "b"), "a\u{1f}b");
        let sql = schema_sql(&tables);
        assert!(sql.contains("CREATE SCHEMA IF NOT EXISTS \"events\""));
        assert!(sql.contains("CREATE TABLE IF NOT EXISTS \"events\".\"namespace_seq\""));
        assert!(sql.contains("event_id TEXT NOT NULL"));
        assert!(sql.contains("UNIQUE(event_id)"));
        assert!(sql.contains("created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()"));
        assert!(sql.contains("PRIMARY KEY(namespace, consumer)"));
        assert!(!sql.contains("search_path"));
        assert_eq!(
            allocation_sql(&tables),
            "INSERT INTO \"events\".\"namespace_seq\" AS target(namespace, next_seq) VALUES ($1, 2) ON CONFLICT(namespace) DO UPDATE SET next_seq = target.next_seq + 1 RETURNING next_seq - 1"
        );
    }

    #[tokio::test]
    async fn postgres_integration_event_store_core() -> PostgresStoreResult<()> {
        let Some(dsn) = std::env::var("KOGWISTAR_TEST_PG_DSN").ok() else {
            eprintln!("KOGWISTAR_TEST_PG_DSN absent; PostgreSQL integration skipped");
            return Ok(());
        };
        let schema = test_schema("core");
        let store = PostgresStore::from_dsn(&dsn, &schema)?;
        store.ensure_schema().await?;

        // Rust-created schema can be queried through Python-compatible DDL.
        let client = store.client().await?;
        let tables = Tables::new(&schema)?;
        let columns = client
            .query_one(
                "SELECT count(*) FROM information_schema.columns WHERE table_schema = $1 AND table_name = 'entity_events'",
                &[&schema],
            )
            .await
            .map_err(backend)?;
        assert_eq!(columns.try_get::<_, i64>(0).map_err(backend)?, 8);
        drop(client);

        let first = store
            .append_raw_entity_event("one", raw("same", "{\"spaced\": [ 1, 2 ]}"))
            .await?;
        assert_eq!(first.event.seq, 1);
        assert_eq!(first.event.payload_json, "{\"spaced\": [ 1, 2 ]}");
        let retry = store
            .append_raw_entity_event("one", raw("same", "{}"))
            .await?;
        assert!(!retry.inserted);
        assert_eq!(retry.event.seq, 1);
        assert!(matches!(
            store.append_raw_entity_event("two", raw("same", "{}")).await,
            Err(PostgresStoreError::EventIdNamespaceCollision { ref existing_namespace, ref requested_namespace, .. })
                if existing_namespace == "one" && requested_namespace == "two"
        ));

        for id in ["two", "three"] {
            store.append_raw_entity_event("one", raw(id, "{}")).await?;
        }
        assert_eq!(store.replay_raw_events("one", 1, 1).await?.len(), 1);
        assert_eq!(store.latest_retained_event_seq("one").await?, 3);
        assert_eq!(
            EventWriteStore::advance_replay_cursor(&store, "one", "sink", 3)
                .await?
                .last_seq,
            3
        );
        assert!(matches!(
            EventWriteStore::advance_replay_cursor(&store, "one", "sink", 2).await,
            Err(StoreError::CursorRegresses {
                current: 3,
                requested: 2
            })
        ));
        assert!(matches!(
            EventWriteStore::advance_replay_cursor(&store, "one", "sink", 4).await,
            Err(StoreError::CursorOutOfRange {
                cursor: 4,
                latest: 3
            })
        ));
        assert_eq!(
            store
                .set_replay_cursor_legacy("one", "sink", 99)
                .await?
                .last_seq,
            99
        );

        let rollback: PostgresStoreResult<()> = store
            .transaction(|uow| {
                Box::pin(async move {
                    uow.append_raw_entity_event("rollback", raw("rollback-id", "{}"))
                        .await?;
                    uow.set_replay_cursor_legacy("rollback", "sink", 1).await?;
                    Err(PostgresStoreError::TransactionAborted("test".to_owned()))
                })
            })
            .await;
        assert!(matches!(
            rollback,
            Err(PostgresStoreError::TransactionAborted(_))
        ));
        assert_eq!(store.latest_retained_event_seq("rollback").await?, 0);
        assert_eq!(store.replay_cursor("rollback", "sink").await?.last_seq, 0);

        // Cancelling after a write drops the native Transaction and must not
        // return an open transaction to the pool or retain partial state.
        let write_reached = Arc::new(Notify::new());
        let task_store = store.clone();
        let task_reached = Arc::clone(&write_reached);
        let cancelled = tokio::spawn(async move {
            task_store
                .transaction(|uow| {
                    Box::pin(async move {
                        uow.append_raw_entity_event("cancelled", raw("cancelled-id", "{}"))
                            .await?;
                        task_reached.notify_one();
                        std::future::pending::<()>().await;
                        #[allow(unreachable_code)]
                        Ok(())
                    })
                })
                .await
        });
        write_reached.notified().await;
        cancelled.abort();
        assert!(
            cancelled
                .await
                .expect_err("cancelled transaction completed")
                .is_cancelled()
        );
        assert_eq!(store.latest_retained_event_seq("cancelled").await?, 0);

        let concurrent = (0..8).map(|n| {
            let store = store.clone();
            tokio::spawn(async move {
                store
                    .append_raw_entity_event("concurrent", raw(&format!("e-{n}"), "{}"))
                    .await
                    .map(|row| row.event.seq)
            })
        });
        let mut sequences = Vec::new();
        for task in concurrent {
            sequences.push(task.await.expect("task panicked")?);
        }
        sequences.sort_unstable();
        assert_eq!(sequences, (1..=8).collect::<Vec<_>>());

        // Search path deliberately points elsewhere; all store queries remain isolated.
        let client = store.client().await?;
        client
            .batch_execute("SET search_path TO pg_catalog")
            .await
            .map_err(backend)?;
        drop(client);
        assert_eq!(EventReadStore::latest_event_seq(&store, "one").await?, 3);

        let isolated_schema = test_schema("isolated");
        let isolated = PostgresStore::from_dsn(&dsn, &isolated_schema)?;
        isolated.ensure_schema().await?;
        assert_eq!(isolated.latest_retained_event_seq("one").await?, 0);
        isolated
            .append_raw_entity_event("one", raw("isolated-id", "{}"))
            .await?;
        assert_eq!(isolated.latest_retained_event_seq("one").await?, 1);
        assert_eq!(store.latest_retained_event_seq("one").await?, 3);

        let cleaner = store.client().await?;
        cleaner
            .batch_execute(&format!("DROP SCHEMA {} CASCADE", tables.quoted_schema))
            .await
            .map_err(backend)?;
        let isolated_tables = Tables::new(&isolated_schema)?;
        let cleaner = store.client().await?;
        cleaner
            .batch_execute(&format!(
                "DROP SCHEMA {} CASCADE",
                isolated_tables.quoted_schema
            ))
            .await
            .map_err(backend)?;
        Ok(())
    }

    #[tokio::test]
    async fn recorded_runtime_and_claimed_handoff_are_atomic_when_dsn_available()
    -> PostgresStoreResult<()> {
        let Some(dsn) = std::env::var("KOGWISTAR_TEST_PG_DSN").ok() else {
            return Ok(());
        };
        let schema = test_schema("recorded_runtime");
        let store = PostgresStore::from_dsn(&dsn, &schema)?;
        let tables = Tables::new(&schema)?;
        store.ensure_schema().await?;

        let planner = store.client().await?;
        planner
            .batch_execute("SET enable_seqscan=off")
            .await
            .map_err(backend)?;
        let plan = planner
            .query(
                &format!(
                    "EXPLAIN SELECT payload_json, seq FROM {} \
                     WHERE event_type='workflow.recorded_transition.v1' \
                     AND payload_json::jsonb->>'transition_id'=$1 LIMIT 1",
                    tables.server_run_events
                ),
                &[&"index-plan-probe"],
            )
            .await
            .map_err(backend)?
            .iter()
            .map(|row| row.get::<_, String>(0))
            .collect::<Vec<_>>()
            .join(" ");
        assert!(
            plan.contains("idx_server_run_events_recorded_transition_id"),
            "{plan}"
        );

        let start_1 = store
            .apply_recorded_runtime_transition(runtime_start("run-1", "start-1"), false)
            .await?;
        store
            .project_lane_message(runtime_lane("run-1", "request-1"))
            .await?;
        let claimed = store
            .claim_projected_lane_messages("runtime", "python-workers", "worker-1", 1, 30)
            .await?;
        assert_eq!(claimed.len(), 1);
        let transition = runtime_result("run-1", "result-1", start_1.event_seq, "worker");
        let handoff = runtime_handoff("run-1", "request-1", "worker-1");
        let first = store
            .apply_claimed_recorded_runtime_transition(handoff.clone(), transition.clone(), false)
            .await?;
        let retry = store
            .apply_claimed_recorded_runtime_transition(handoff.clone(), transition.clone(), false)
            .await?;
        assert!(!first.idempotent && retry.idempotent);
        assert_eq!(first.event_seq, retry.event_seq);
        assert_eq!(
            store
                .get_projected_lane_message("request-1")
                .await?
                .unwrap()
                .status,
            "completed"
        );
        assert_eq!(
            store
                .read_recorded_runtime_state("run-1", "wf-1", "conv-1")
                .await?
                .unwrap()
                .state["answer"],
            serde_json::json!("worker")
        );
        assert!(matches!(
            store
                .apply_claimed_recorded_runtime_transition(
                    handoff,
                    runtime_result("run-1", "result-1", start_1.event_seq, "changed"),
                    false,
                )
                .await,
            Err(PostgresStoreError::RecordedRuntimeConflict(_))
        ));

        let start_2 = store
            .apply_recorded_runtime_transition(runtime_start("run-2", "start-2"), false)
            .await?;
        store
            .project_lane_message(runtime_lane("run-2", "request-2"))
            .await?;
        store
            .claim_projected_lane_messages("runtime", "python-workers", "worker-2", 1, 30)
            .await?;
        assert!(matches!(
            store
                .apply_claimed_recorded_runtime_transition(
                    runtime_handoff("run-2", "request-2", "wrong-worker"),
                    runtime_result("run-2", "result-2", start_2.event_seq, "worker"),
                    false,
                )
                .await,
            Err(PostgresStoreError::RecordedRuntimeConflict(_))
        ));

        let start_3 = store
            .apply_recorded_runtime_transition(runtime_start("run-3", "start-3"), false)
            .await?;
        store
            .project_lane_message(runtime_lane("run-3", "request-3"))
            .await?;
        store
            .claim_projected_lane_messages("runtime", "python-workers", "dead-worker", 1, -1)
            .await?;
        store
            .claim_projected_lane_messages("runtime", "python-workers", "new-worker", 1, 30)
            .await?;
        assert!(matches!(
            store
                .apply_claimed_recorded_runtime_transition(
                    runtime_handoff("run-3", "request-3", "dead-worker"),
                    runtime_result("run-3", "result-3", start_3.event_seq, "worker"),
                    false,
                )
                .await,
            Err(PostgresStoreError::RecordedRuntimeConflict(_))
        ));
        let reclaimed = store
            .apply_claimed_recorded_runtime_transition(
                runtime_handoff("run-3", "request-3", "new-worker"),
                runtime_result("run-3", "result-3", start_3.event_seq, "worker"),
                false,
            )
            .await?;
        assert!(!reclaimed.idempotent);

        let start_4 = store
            .apply_recorded_runtime_transition(runtime_start("run-4", "start-4"), false)
            .await?;
        let before = store
            .list_server_run_events("run-4", 0, usize::MAX)
            .await?
            .len();
        assert!(matches!(
            store
                .apply_recorded_runtime_transition(
                    runtime_result("run-4", "result-4", start_4.event_seq, "rolled"),
                    true,
                )
                .await,
            Err(PostgresStoreError::TransactionAborted(_))
        ));
        assert_eq!(
            store
                .list_server_run_events("run-4", 0, usize::MAX)
                .await?
                .len(),
            before
        );

        let start_5 = store
            .apply_recorded_runtime_transition(runtime_start("run-5", "start-5"), false)
            .await?;
        store
            .project_lane_message(runtime_lane("run-5", "request-5"))
            .await?;
        store
            .claim_projected_lane_messages("runtime", "python-workers", "worker-5", 1, 30)
            .await?;
        store.request_server_run_cancel("run-5").await?;
        assert_eq!(
            store.get_server_run("run-5").await?.unwrap().status,
            "cancelled"
        );
        assert_eq!(
            store
                .read_recorded_runtime_state("run-5", "wf-1", "conv-1")
                .await?
                .unwrap()
                .status,
            kogwistar_runtime::RecordedRunStatus::Cancelled
        );
        assert_eq!(
            store
                .get_projected_lane_message("request-5")
                .await?
                .unwrap()
                .status,
            "cancelled"
        );
        assert!(matches!(
            store
                .apply_claimed_recorded_runtime_transition(
                    runtime_handoff("run-5", "request-5", "worker-5"),
                    runtime_result("run-5", "result-5", start_5.event_seq, "stale"),
                    false,
                )
                .await,
            Err(PostgresStoreError::RecordedRuntimeConflict(_))
        ));

        for (run_id, message_id, owner) in [
            ("handled", "handled-request", "handled-worker"),
            ("unhandled", "unhandled-request", "unhandled-worker"),
            ("forged", "forged-request", "forged-worker"),
        ] {
            store
                .apply_recorded_runtime_transition(
                    runtime_failure_start(run_id, &format!("start-{run_id}")),
                    false,
                )
                .await?;
            store
                .project_lane_message(runtime_worker_lane(run_id, message_id))
                .await?;
            assert_eq!(
                store
                    .claim_projected_lane_messages("runtime", "python-workers", owner, 1, 30,)
                    .await?
                    .len(),
                1
            );
        }

        let handled_handoff =
            runtime_worker_handoff("handled", "handled-request", "handled-worker");
        let handled_effect =
            runtime_failure_effect("handled-effect", &["recover"], serde_json::json!([]));
        let handled = store
            .apply_claimed_recorded_worker_effect(handled_handoff.clone(), handled_effect.clone())
            .await?;
        let handled_retry = store
            .apply_claimed_recorded_worker_effect(handled_handoff, handled_effect)
            .await?;
        assert!(!handled.idempotent && handled_retry.idempotent);
        assert_eq!(handled.event_seq, handled_retry.event_seq);
        assert_eq!(
            handled.reduced.state.status,
            kogwistar_runtime::RecordedRunStatus::Running
        );
        assert_eq!(handled.reduced.state.frontier.pending[0].0, "recover");
        assert_eq!(
            store
                .get_projected_lane_message("handled-request")
                .await?
                .unwrap()
                .status,
            "completed"
        );

        let unhandled = store
            .apply_claimed_recorded_worker_effect(
                runtime_worker_handoff("unhandled", "unhandled-request", "unhandled-worker"),
                runtime_failure_effect("unhandled-effect", &[], serde_json::json!([])),
            )
            .await?;
        assert_eq!(
            unhandled.reduced.state.status,
            kogwistar_runtime::RecordedRunStatus::Failed
        );
        assert!(unhandled.reduced.state.frontier.pending.is_empty());
        assert_eq!(
            store.get_server_run("unhandled").await?.unwrap().status,
            "failed"
        );

        assert!(matches!(
            store
                .apply_claimed_recorded_worker_effect(
                    runtime_worker_handoff("forged", "forged-request", "forged-worker"),
                    runtime_failure_effect(
                        "forged-effect",
                        &[],
                        serde_json::json!([{"node_id": "forged", "join_mask": 0}]),
                    ),
                )
                .await,
            Err(PostgresStoreError::RecordedRuntime(_))
        ));
        assert_eq!(
            store
                .list_server_run_events("forged", 0, usize::MAX)
                .await?
                .len(),
            1
        );
        assert_eq!(
            store
                .get_projected_lane_message("forged-request")
                .await?
                .unwrap()
                .status,
            "claimed"
        );

        store
            .client()
            .await?
            .batch_execute(&format!("DROP SCHEMA {} CASCADE", tables.quoted_schema))
            .await
            .map_err(backend)
    }

    #[tokio::test]
    async fn python_created_schema_is_read_and_written_when_dsn_available()
    -> PostgresStoreResult<()> {
        let Some(dsn) = std::env::var("KOGWISTAR_TEST_PG_DSN").ok() else {
            return Ok(());
        };
        let schema = test_schema("python");
        let store = PostgresStore::from_dsn(&dsn, &schema)?;
        let tables = Tables::new(&schema)?;
        let client = store.client().await?;
        client
            .batch_execute(&format!(
                "CREATE SCHEMA {schema};\
                 CREATE TABLE {ns}(namespace TEXT PRIMARY KEY, next_seq BIGINT NOT NULL);\
                 INSERT INTO {ns} VALUES ('legacy', 7);\
                 CREATE TABLE {events}(namespace TEXT NOT NULL DEFAULT 'default', seq BIGINT NOT NULL, event_id TEXT NOT NULL, entity_kind TEXT NOT NULL, entity_id TEXT NOT NULL, op TEXT NOT NULL, payload_json TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), PRIMARY KEY(namespace, seq), UNIQUE(event_id));\
                 INSERT INTO {events}(namespace, seq, event_id, entity_kind, entity_id, op, payload_json) VALUES ('legacy', 6, 'old', 'node', 'old', 'UPSERT', '{{\"original\": true }}');\
                 CREATE TABLE {cursors}(namespace TEXT NOT NULL DEFAULT 'default', consumer TEXT NOT NULL, last_seq BIGINT NOT NULL, updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), PRIMARY KEY(namespace, consumer));\
                 INSERT INTO {cursors}(namespace, consumer, last_seq) VALUES ('legacy', 'reader', 6);",
                schema = tables.quoted_schema,
                ns = tables.namespace_seq,
                events = tables.entity_events,
                cursors = tables.replay_cursors,
            ))
            .await
            .map_err(backend)?;
        drop(client);
        let row = store
            .append_raw_entity_event("legacy", raw("new", "{\"kept\": true }"))
            .await?;
        assert_eq!(row.event.seq, 7);
        assert_eq!(row.event.payload_json, "{\"kept\": true }");
        assert_eq!(store.replay_cursor("legacy", "reader").await?.last_seq, 6);
        let cleaner = store.client().await?;
        cleaner
            .batch_execute(&format!("DROP SCHEMA {} CASCADE", tables.quoted_schema))
            .await
            .map_err(backend)
    }

    #[tokio::test]
    async fn auth_store_links_python_user_and_retries_identity_when_dsn_available()
    -> PostgresStoreResult<()> {
        let Some(dsn) = std::env::var("KOGWISTAR_TEST_PG_DSN").ok() else {
            return Ok(());
        };
        let schema = test_schema("auth_identity");
        let quoted_schema = quote_identifier(&schema)?;
        let base_config = dsn
            .parse::<Config>()
            .map_err(|error| PostgresStoreError::Backend(error.to_string()))?;
        let (client, connection) = base_config.connect(NoTls).await.map_err(backend)?;
        tokio::spawn(async move {
            let _ = connection.await;
        });
        client
            .batch_execute(&format!("CREATE SCHEMA {quoted_schema}"))
            .await
            .map_err(backend)?;
        let mut auth_config = base_config;
        auth_config.options(format!("-c search_path={schema}"));
        let store = PostgresAuthStore::from_config(auth_config)?;
        store.ensure_schema().await?;
        client
            .execute(
                &format!(
                    "INSERT INTO {quoted_schema}.users (user_id, email, display_name, is_active, global_role, global_ns, created_at) VALUES ('python-user', 'alice@example.com', 'Python Alice', TRUE, 'rw', 'docs,workflow', CURRENT_TIMESTAMP)"
                ),
                &[],
            )
            .await
            .map_err(backend)?;
        let request = ResolveExternalIdentity {
            issuer: "https://issuer.example".to_owned(),
            subject: "subject-1".to_owned(),
            email: "alice@example.com".to_owned(),
            display_name: Some("Ignored".to_owned()),
            new_user_id: "rust-user".to_owned(),
            default_role: "ro".to_owned(),
            default_ns: "docs".to_owned(),
        };
        let linked = store.resolve_external_identity(request.clone()).await?;
        assert_eq!(linked.user_id, "python-user");
        assert_eq!(linked.global_role.as_deref(), Some("rw"));
        let retried = store
            .resolve_external_identity(ResolveExternalIdentity {
                new_user_id: "must-not-win".to_owned(),
                email: "changed@example.com".to_owned(),
                ..request
            })
            .await?;
        assert_eq!(retried.user_id, "python-user");
        assert_eq!(
            store
                .external_identity("https://issuer.example", "subject-1")
                .await?
                .unwrap()
                .email
                .as_deref(),
            Some("alice@example.com")
        );
        client
            .batch_execute(&format!("DROP SCHEMA {quoted_schema} CASCADE"))
            .await
            .map_err(backend)?;
        Ok(())
    }
}
