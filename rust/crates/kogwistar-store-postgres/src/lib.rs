//! PostgreSQL event-store core for ADR-015 Phase 3.
//!
//! This crate intentionally owns only `namespace_seq`, `entity_events`, and
//! `replay_cursors`. Every relation is schema-qualified; correctness never
//! depends on PostgreSQL's `search_path`.

use deadpool_postgres::{Manager, Pool};
use kogwistar_contracts::EntityEventEnvelope;
use kogwistar_engine::EntityProjection;
use kogwistar_runtime::{
    PersistedRecordedTransition, RECORDED_RUNTIME_CONTRACT_VERSION, RecordedRuntimeState,
    RecordedRuntimeTransition, RecordedTransitionResult, RecordedWorkerHandoff,
    RecordedWorkerSuccessEffect, RuntimeFrontier, RuntimeStepExecutePayload,
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
use tokio_postgres::{Config, GenericClient, NoTls, Row, Transaction};

// Must match EnginePostgresMetaStore's stable startup DDL lock. Python and Rust
// processes can bootstrap the same schema concurrently, so this key is part of
// the cross-language storage protocol.
const POSTGRES_BOOTSTRAP_ADVISORY_LOCK_KEY: i64 = 748_219_503;
const EDGE_ENDPOINTS_TABLE: &str = "gke_edge_endpoints";
const NODE_DOCS_TABLE: &str = "gke_node_docs";
const NODE_REFS_TABLE: &str = "gke_node_refs";
const EDGE_REFS_TABLE: &str = "gke_edge_refs";

/// Future accepted by [`PostgresStore::transaction`]. The lifetime is tied to
/// the UoW borrow, so a pooled client cannot escape its transaction.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

mod types;
pub use types::{
    AppendedRawEvent, GraphDeleteMutation, GraphIndexTableNames, GraphMetadataPatchMutation,
    GraphProjectionMetadataPatch, GraphProjectionUpsert, GraphTableNames, NamedProjectionCas,
    NewRawEntityEvent, PostgresStoreError, PostgresStoreResult, RawEntityEvent,
};

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
mod table_names;
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
use table_names::Tables;
pub use unit_of_work::PostgresUnitOfWork;
use workflow_runtime::*;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
