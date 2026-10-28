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
    AcceptedIndexJobResult, AppendedEvent, AppliedGraphMutation, AuthIdentityStore, AuthUser,
    DocumentAuthorityCapability, DocumentProjectionUpsert, DocumentStore, EntityEvent,
    EntityRebuildRequest, EntityRecoveryReport, EntityRecoveryRequest, EventPruneStore,
    EventReadStore, EventWriteStore, ExternalIdentity, GraphAuthorityCapability, GraphMutation,
    GraphMutationBatch, GraphMutationStore, GraphProjectionPageRead, GraphProjectionRead,
    GraphProjectionVectorQuery, GraphProjectionWriteStore, GraphRecord, GraphScope, IndexJob,
    IndexJobClaimFilter, IndexJobReadStore, IndexJobWriteStore, LaneMessageClaimFilter,
    LaneMessageFilter, LaneMessageReadStore, LaneMessageWriteStore, NamedProjection,
    NamedProjectionWrite, NewEntityEvent, NewIndexJob, NewProjectedLaneMessage,
    ProjectedLaneMessage, ProjectionReadStore, ProjectionWriteStore,
    RUNTIME_CURRENT_STATE_NAMESPACE, ReplayCursor, ResolveExternalIdentity, ServerRun,
    ServerRunCreate, ServerRunEvent, ServerRunReadStore, ServerRunUpdate, ServerRunWriteStore,
    StoreError, StoreResult, VectorMatch, WorkflowDesignDelta, WorkflowDesignDeltaWrite,
    WorkflowDesignHistoryReadStore, WorkflowDesignHistoryWriteStore, WorkflowDesignSnapshot,
    WorkflowDesignSnapshotWrite, runtime_checkpoint_namespace, runtime_projection_write,
    runtime_status_namespace, validate_entity_rebuild_request, validate_entity_recovery_request,
    validate_graph_lifecycle_target, validate_graph_mutation, validate_graph_mutation_batch,
    validate_graph_projection_target,
};
use serde_json::{Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;
use std::pin::Pin;
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;
use tokio_postgres::{Config, GenericClient, NoTls, Row, Transaction};

// Must match EnginePostgresMetaStore's stable startup DDL lock. Python and Rust
// processes can bootstrap the same schema concurrently, so this key is part of
// the cross-language storage protocol.
const POSTGRES_BOOTSTRAP_ADVISORY_LOCK_KEY: i64 = 748_219_503;
const EDGE_ENDPOINTS_TABLE: &str = "gke_edge_endpoints";
const NODE_DOCS_TABLE: &str = "gke_node_docs";
const NODE_REFS_TABLE: &str = "gke_node_refs";
const EDGE_REFS_TABLE: &str = "gke_edge_refs";

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

/// One compare-and-swap item for an atomic named-projection batch.
///
/// All expected values are checked while their rows are locked before any
/// replacement is applied. This keeps a multi-projection materialization from
/// exposing a partial artifact/graph/catalog commit.
#[derive(Clone, Debug, PartialEq)]
pub struct NamedProjectionCas {
    pub namespace: String,
    pub key: String,
    pub expected_last_authoritative_seq: Option<i64>,
    pub expected_last_materialized_seq: Option<i64>,
    pub projection: NamedProjectionWrite,
}

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
    /// PostgreSQL's textual `TIMESTAMPTZ` embedding. It is deliberately
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

/// Backend projection names needed when the native Rust store drains its
/// deterministic, metadata-only graph join-index jobs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphIndexTableNames {
    pub nodes: String,
    pub edges: String,
    pub node_docs: String,
    pub node_refs: String,
    pub edge_refs: String,
    pub edge_endpoints: String,
}

impl Default for GraphIndexTableNames {
    fn default() -> Self {
        Self {
            nodes: "gke_nodes".to_owned(),
            edges: "gke_edges".to_owned(),
            node_docs: NODE_DOCS_TABLE.to_owned(),
            node_refs: NODE_REFS_TABLE.to_owned(),
            edge_refs: EDGE_REFS_TABLE.to_owned(),
            edge_endpoints: EDGE_ENDPOINTS_TABLE.to_owned(),
        }
    }
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

/// Physical projection purge after canonical lifecycle admission.
///
/// This is not the user-facing delete lifecycle. Node and edge callers must
/// persist a canonical tombstone before this operation is accepted; the
/// tombstone remains the durable source of deletion history even when the
/// materialized row is later purged.
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
    edge_endpoints: String,
    node_docs: String,
    node_refs: String,
    edge_refs: String,
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
            edge_endpoints: qualified(&quoted_schema, EDGE_ENDPOINTS_TABLE)?,
            node_docs: qualified(&quoted_schema, NODE_DOCS_TABLE)?,
            node_refs: qualified(&quoted_schema, NODE_REFS_TABLE)?,
            edge_refs: qualified(&quoted_schema, EDGE_REFS_TABLE)?,
        })
    }
}

/// Cloneable async store backed by `deadpool-postgres` and `tokio-postgres`.
#[derive(Clone)]
pub struct PostgresStore {
    pool: Pool,
    tables: Tables,
}

mod auth;
mod event_projection;
mod graph;
mod index_job_results;
mod index_jobs;
mod lane_messages;
mod named_projections;
mod recovery;
mod row_mapping;
mod store;
mod trait_impls;
mod unit_of_work;
mod workflow_runtime;

pub use auth::PostgresAuthStore;
use event_projection::*;
use graph::*;
use index_job_results::*;
use index_jobs::*;
use lane_messages::*;
use named_projections::*;
use recovery::*;
pub use row_mapping::quote_identifier;
use row_mapping::*;
pub use unit_of_work::PostgresUnitOfWork;
use workflow_runtime::*;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
