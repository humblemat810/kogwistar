//! Public request, response, and error types for the PostgreSQL store.

use kogwistar_runtime::RecordedRuntimeError;
use kogwistar_store::{GraphRecord, GraphScope, NamedProjectionWrite, StoreError};
use serde_json::{Map, Value};
use thiserror::Error;

/// Errors returned by direct PostgreSQL store APIs.
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
    /// PostgreSQL's textual `TIMESTAMPTZ` embedding.
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

/// Existing Python pgvector collection names accepted by the explicit graph schema action.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphTableNames {
    pub nodes: String,
    pub edges: String,
    pub documents: String,
    pub domains: String,
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

/// Backend projection names used by deterministic graph join-index jobs.
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
            node_docs: "gke_node_docs".to_owned(),
            node_refs: "gke_node_refs".to_owned(),
            edge_refs: "gke_edge_refs".to_owned(),
            edge_endpoints: "gke_edge_endpoints".to_owned(),
        }
    }
}

/// Atomic metadata patch for an existing Python pgvector graph row.
#[derive(Clone, Debug, PartialEq)]
pub struct GraphMetadataPatchMutation {
    pub scope: GraphScope,
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
#[derive(Clone, Debug, PartialEq)]
pub struct GraphDeleteMutation {
    pub scope: GraphScope,
    pub table: String,
    pub entity_kind: String,
    pub event_id: String,
    pub entity_id: String,
    pub payload: Value,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GraphProjectionUpsert {
    pub scope: GraphScope,
    pub table: String,
    pub record: GraphRecord,
    pub embedding_dim: usize,
}

/// PostgreSQL Stage-1 row shared with Python's two-stage projection adapter.
#[derive(Clone, Debug, PartialEq)]
pub struct Stage1GraphProjection {
    pub namespace: String,
    pub entity_kind: String,
    pub entity_id: String,
    pub document: String,
    pub metadata: Map<String, Value>,
    pub source_fingerprint: String,
    pub revision: i64,
    pub materialization_status: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GraphProjectionMetadataPatch {
    pub scope: GraphScope,
    pub table: String,
    pub entity_id: String,
    pub document: Option<String>,
    pub metadata_patch: Map<String, Value>,
    pub patch_document_metadata: bool,
}
