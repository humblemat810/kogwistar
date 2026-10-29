//! Transport-neutral Phase-5 API contracts.
//!
//! Network transports adapt these typed decisions. They do not own domain
//! policy, authentication role checks, SSE framing, or JSON-RPC envelopes.

use base64::Engine as _;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use jsonwebtoken::{
    Algorithm, DecodingKey, EncodingKey, Header, Validation, decode, decode_header, encode,
};
use kogwistar_contracts::{
    ContextSnapshotV1, ConversationTranscriptV1, DelegationAuthorityV1, DocumentUpsertV1,
    GraphMutationBatchV1 as GraphMutationBatchInput, GraphMutationV1 as GraphMutationInput,
    GraphNeighborsRequestV1, ToolAuditResponseV1, attach_rust_native_seam_attestation,
    attenuate_capabilities, graph_neighbors_request_from_str,
};
use kogwistar_engine::embedding::EmbeddingProvider;
use kogwistar_runtime::{
    RECORDED_RUNTIME_CONTRACT_VERSION, RecordedRuntimeState, RecordedRuntimeTransition,
    RecordedTransitionKind, RecordedWorkerHandoff, RecordedWorkerSuccessEffect, RuntimeFrontier,
    RuntimeStepExecutePayload, RuntimeStepExecuteRequest,
};
use kogwistar_store::{
    AuthIdentityStore, AuthUser, ConversationListAccumulator, DistanceMetric,
    DocumentAuthorityCapability, DocumentProjectionUpsert, DocumentStore, EntityEvent,
    GraphAclIndex, GraphAclPrincipal, GraphAuthorityCapability, GraphEntityKind,
    GraphFindEdgesQuery, GraphKHopQuery, GraphMutation, GraphMutationBatch, GraphMutationStore,
    GraphProjectionRead, GraphProjectionVectorQuery, GraphReadStore, GraphRecord, GraphScope,
    GraphShortestPathQuery, IndexJobClaimFilter, LaneMessageClaimFilter, LaneMessageFilter,
    MetadataFilter, NamedProjection, NamedProjectionWrite, NewIndexJob, NewProjectedLaneMessage,
    ProjectedLaneMessage, ResolveExternalIdentity, ServerRun, ServerRunCreate, ServerRunEvent,
    ServerRunReadStore, VectorQuery, WorkflowDesignDeltaWrite, WorkflowDesignSnapshot,
    WorkflowDesignSnapshotWrite, compact_conversation_list_candidates,
    conversation_summary_from_visible_records, conversation_transcript_from_visible_records,
    for_each_graph_projection_page, graph_find_edges_from_records, graph_k_hop_from_records,
    graph_neighbors_from_records, graph_projection_records_all, graph_records_visible_by_acl,
    graph_records_visible_by_acl_with_index, graph_shortest_path_from_records,
    tool_audit_items_from_visible_records, validate_typed_graph_mutation,
};
use kogwistar_store_postgres::{
    NewRawEntityEvent as PostgresNewRawEntityEvent, PostgresAuthStore, PostgresStore,
};
use kogwistar_store_sqlite::{
    NewRawEntityEvent as SqliteNewRawEntityEvent, RawEntityEvent as SqliteRawEntityEvent,
    SqliteAuthStore, SqliteStore,
};

mod cutover;
mod native_answer_persist;
mod native_memory_pin;
mod runtime_contract;
pub use cutover::{
    ImplementationSnapshot, PENDING_SERVER_CUTOVER_ROUTES, PENDING_SYSCALL_CUTOVER_OPS,
    TRANSPORT_OWNED_HTTP_ROUTES,
};
use cutover::{is_pending_server_cutover_route, is_transport_owned_http_route};
use runtime_contract::{
    default_runtime_kind, runtime_implementation_attestation, runtime_worker_recipient,
    worker_protocol_for_runtime_kind,
};

use axum::{
    Json, Router,
    body::{Body, Bytes},
    extract::{OriginalUri, Query, State},
    http::{HeaderMap, HeaderValue, Method, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{MethodFilter, get, on, post},
};

include!(concat!(env!("OUT_DIR"), "/frozen_routes.rs"));
include!("sections/contracts_and_runtime.rs");
include!("sections/response_policy.rs");
include!("sections/graph_acl_effects.rs");
include!("sections/document_graph.rs");
include!("sections/workflow_design.rs");
include!("sections/sqlite_service.rs");
include!("sections/postgres_contracts.rs");
include!("sections/postgres_graph_service.rs");
include!("sections/postgres_automation_service.rs");
include!("sections/postgres_runtime_service.rs");
include!("sections/application_services.rs");
include!("sections/auth_state.rs");
include!("sections/auth_handlers.rs");
include!("sections/http_transport.rs");

#[cfg(test)]
mod tests {
    use super::*;

    include!("tests/part_01.rs");
    include!("tests/part_02.rs");
    include!("tests/part_03.rs");
    include!("tests/part_04.rs");
    include!("tests/part_05.rs");
    include!("tests/part_06.rs");
}
