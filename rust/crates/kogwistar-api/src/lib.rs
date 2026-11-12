//! Transport-neutral Phase-5 API contracts.
//!
//! Network transports adapt these typed decisions. They do not own domain
//! policy, authentication role checks, SSE framing, or JSON-RPC envelopes.

use base64::Engine as _;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use jsonwebtoken::{
    Algorithm, DecodingKey, EncodingKey, Header, Validation, decode, decode_header, encode,
};
use kogwistar_contracts::{
    ContextSnapshotV1, ConversationTranscriptV1, DelegationAuthorityV1, DocumentUpsertV1,
    GraphExtractionRequestV1, GraphIndexJobV1 as GraphIndexJobInput, GraphModelClass,
    GraphModelEnvelope, GraphMutationBatchV1 as GraphMutationBatchInput,
    GraphMutationV1 as GraphMutationInput, GraphNeighborsRequestV1, GraphNeighborsResolveModeV1,
    ToolAuditResponseV1, attach_rust_native_seam_attestation, attenuate_capabilities,
    graph_neighbors_request_from_str, worker_identity_attestation_from_str,
};
use kogwistar_engine::canonical_revision::read_canonical_entity_revision;
use kogwistar_engine::embedding::EmbeddingProvider;
use kogwistar_runtime::{
    RECORDED_RUNTIME_CONTRACT_VERSION, RecordedRuntimeState, RecordedRuntimeTransition,
    RecordedTransitionKind, RecordedWorkerHandoff, RecordedWorkerSuccessEffect, RuntimeFrontier,
    RuntimeStepExecutePayload, RuntimeStepExecuteRequest,
};
use kogwistar_store::{
    AuthIdentityStore, AuthUser, ConversationListAccumulator, DistanceMetric,
    DocumentAuthorityCapability, DocumentProjectionUpsert, DocumentStore, EntityEvent,
    GraphAclIndex, GraphAclPrincipal, GraphAclProjection, GraphAuthorityCapability,
    GraphEntityKind, GraphFindEdgesQuery, GraphKHopQuery, GraphMutation, GraphMutationBatch,
    GraphMutationStore, GraphProjectionRead, GraphProjectionVectorQuery, GraphReadStore,
    GraphRecord, GraphScope, GraphShortestPathQuery, IndexJobClaimFilter, LaneMessageClaimFilter,
    LaneMessageFilter, MetadataFilter, NamedProjection, NamedProjectionWrite, NewIndexJob,
    NewProjectedLaneMessage, ProjectedLaneMessage, ResolveExternalIdentity, ServerRun,
    ServerRunCreate, ServerRunEvent, ServerRunReadStore, StoreError, VectorQuery,
    WorkflowDesignDeltaWrite, WorkflowDesignSnapshot, WorkflowDesignSnapshotWrite,
    compact_conversation_list_candidates, conversation_summary_from_visible_records,
    conversation_transcript_from_visible_records, for_each_graph_projection_page,
    graph_find_edges_from_records, graph_k_hop_from_records, graph_neighbors_from_records,
    graph_projection_records_all, graph_records_visible_by_acl,
    graph_records_visible_by_acl_with_index, graph_shortest_path_from_records,
    graph_viz_cytoscape_from_records, graph_viz_d3_from_records, resolve_graph_edge_redirect_id,
    resolve_graph_node_redirect_id, tool_audit_items_from_visible_records,
    validate_typed_graph_mutation,
};
use kogwistar_store_postgres::{
    GraphIndexTableNames, GraphProjectionUpsert as PostgresGraphProjectionUpsert,
    NewRawEntityEvent as PostgresNewRawEntityEvent, PostgresAuthStore, PostgresStore,
};
use kogwistar_store_sqlite::{
    NewRawEntityEvent as SqliteNewRawEntityEvent, RawEntityEvent as SqliteRawEntityEvent,
    SqliteAuthStore, SqliteStore, SqliteStoreError, SqliteUnitOfWork,
};

mod cutover;
mod native_answer_generation;
mod native_answer_lifecycle;
mod native_answer_link;
mod native_answer_materialize;
mod native_answer_persist;
mod native_answer_prepare;
mod native_answer_project;
mod native_answer_state;
mod native_answer_view;
mod native_event_stream;
mod native_memory_pin;
mod runtime_contract;
pub mod search_index_fts;
pub mod short_id_store;
mod transient_deltas;
pub use cutover::{ImplementationSnapshot, TRANSPORT_OWNED_HTTP_ROUTES};
use cutover::{is_pending_server_cutover_route, is_transport_owned_http_route};
use kogwistar_store_memory::InMemoryStore;
use runtime_contract::{
    default_runtime_kind, runtime_implementation_attestation, runtime_worker_identity_recipient,
    runtime_worker_recipient, worker_protocol_for_runtime_kind,
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
include!("sections/wisdom_graph.rs");
include!("sections/wisdom_retrieval.rs");
include!("sections/workflow_design.rs");
include!("sections/sqlite_service.rs");
include!("sections/sqlite_native_answer_context.rs");
include!("sections/sqlite_revision_fenced_projection.rs");
include!("sections/sqlite_chroma_write_effects.rs");
include!("sections/sqlite_chroma_conversation_reads.rs");
include!("sections/sqlite_graph_reads.rs");
include!("sections/sqlite_run_lifecycle.rs");
include!("sections/sqlite_native_runtime_execution.rs");
include!("sections/postgres_contracts.rs");
include!("sections/postgres_graph_service.rs");
include!("sections/postgres_automation_service.rs");
include!("sections/postgres_runtime_service.rs");
include!("sections/postgres_native_conversation.rs");
include!("sections/postgres_runtime_execution.rs");
include!("sections/postgres_graph_effects.rs");
include!("sections/postgres_public_graph_effects.rs");
include!("sections/provider_event_publication.rs");
include!("sections/application_services.rs");
include!("sections/sqlite_application_service.rs");
include!("sections/sqlite_automation_services.rs");
include!("sections/in_memory_service.rs");
include!("sections/auth_state.rs");
include!("sections/auth_handlers.rs");
include!("sections/http_transport.rs");

#[cfg(test)]
mod tests {
    use super::*;

    include!("tests/graph_scope_contracts.rs");
    include!("tests/server_routes.rs");
    include!("tests/conversation_read_authorization.rs");
    include!("tests/postgres_conversation_reads.rs");
    include!("tests/conversation_create.rs");
    include!("tests/server_readiness.rs");
    include!("tests/graph_query_contracts.rs");
    include!("tests/graph_read_routes.rs");
    include!("tests/runtime_graph_contracts.rs");
    include!("tests/native_runtime_effects.rs");
    include!("tests/answer_workflow_admission.rs");
    include!("tests/auth.rs");
    include!("tests/mcp.rs");
    include!("tests/server_transport.rs");
    include!("tests/runtime.rs");
    include!("tests/sqlite_visibility_acl.rs");
    include!("tests/workflow_design_history.rs");
    include!("tests/postgres_graph_mutations.rs");
    include!("tests/postgres_graph_reads.rs");
    include!("tests/postgres_document_projection.rs");
    include!("tests/postgres_design_history.rs");
    include!("tests/postgres_service_events.rs");
    include!("tests/postgres_configuration.rs");
    include!("tests/sqlite_service_events.rs");
    include!("tests/document_validation.rs");
    include!("tests/graph_mutation_contracts.rs");
    include!("tests/visualization_routes.rs");
    include!("tests/runtime_wire_contracts.rs");
    include!("tests/embedding_provider_contracts.rs");
    include!("tests/sqlite_stage1_recovery.rs");
    include!("tests/postgres_index_projection.rs");
    include!("tests/sqlite_projection_lifecycle.rs");
    include!("tests/embedding_configuration.rs");
    include!("tests/inline_runtime.rs");
    include!("tests/mcp_run_observability.rs");
    include!("tests/mcp_workflow_contracts.rs");
    include!("tests/workflow_run_authorization.rs");
    include!("tests/workflow_mutation_authorization.rs");
    include!("tests/conversation_tool_authorization.rs");
    include!("tests/workflow_service_mcp.rs");
    include!("tests/conversation_mcp.rs");
    include!("tests/sqlite_run_service.rs");
    include!("tests/document_id_tools.rs");
    include!("tests/stage2_projection.rs");
    include!("tests/run_owner_authority.rs");
    include!("tests/postgres_worker_authority.rs");
    include!("tests/postgres_graph_effects.rs");
    include!("tests/postgres_inline_effects.rs");
    include!("tests/conversation_admission.rs");
    include!("tests/sqlite_runtime.rs");
    include!("tests/tcp_transport.rs");
    include!("tests/sqlite_graph_topology_reads.rs");
    include!("tests/sqlite_transcript_reads.rs");
    include!("tests/postgres_agentic_answer.rs");
    include!("tests/native_answer_views.rs");
    include!("tests/workflow_run_access.rs");
    include!("tests/workflow_monitoring.rs");
    include!("tests/workflow_run_routes.rs");
    include!("tests/wisdom_graph_authority.rs");
    include!("tests/provider_event_contracts.rs");
    include!("tests/sqlite_graph_writes.rs");
    include!("tests/sqlite_conversation_admission.rs");
    include!("tests/sqlite_runtime_effects.rs");
    include!("tests/sqlite_provider_events.rs");
    include!("tests/sqlite_wisdom_retrieval.rs");
    include!("tests/postgres_identity.rs");
    include!("tests/in_memory_conversations.rs");
    include!("tests/in_memory_run_admission.rs");
    include!("tests/in_memory_provider_events.rs");
    include!("tests/in_memory_semantic_retrieval.rs");
    include!("tests/inline_owner_authority.rs");
    include!("tests/in_memory_graph_routes.rs");
    include!("tests/postgres_visualization.rs");
    include!("tests/vector_query_acl.rs");
    include!("tests/postgres_graph_types.rs");
    include!("tests/in_memory_mcp_traversal.rs");
    include!("tests/postgres_hybrid_search.rs");
    include!("tests/postgres_workflow_run_payload_parity.rs");
}
