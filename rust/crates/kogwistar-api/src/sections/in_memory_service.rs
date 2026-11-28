// PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/in_memory_service/graph_operations.rs#graph_read_effect; python=kogwistar/graph_query.py#GraphQuery.neighbors; fixture=contracts/golden/graph-acl-traversal-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/in_memory_mcp_traversal.rs#in_memory_mcp_acl_traversal_matches_shared_python_contract; python_test=tests/mcp/test_chat_mcp_graph_reads.py#test_mcp_neighbors_preserve_acl_visibility_and_hide_topology
use super::*;

#[derive(Clone)]
pub(super) struct InMemoryGraphApplicationService {
    pub(super) store: InMemoryStore,
    pub(super) capabilities: SharedCapabilityState,
    pub(super) native_operations: kogwistar_runtime::NativeOperationRegistry,
    pub(super) embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
    pub(super) chat_completion_provider:
        Option<crate::native_answer_generation::OpenAiCompatibleChatConfig>,
    pub(super) transient_run_deltas: crate::transient_deltas::TransientRunDeltas,
    pub(super) search_index_fts: Option<Arc<Mutex<rusqlite::Connection>>>,
    pub(super) owner_auth_store: Option<AuthStore>,
}

#[derive(Clone)]
pub(super) struct InMemoryConversationAdmission {
    pub(super) graph_batch: GraphMutationBatch,
    pub(super) expected_projections: Vec<kogwistar_store_memory::GraphProjectionExpectation>,
    pub(super) request_fingerprint: Option<String>,
    pub(super) actor_user_id: String,
}

impl Default for InMemoryGraphApplicationService {
    fn default() -> Self {
        Self {
            store: InMemoryStore::default(),
            capabilities: Arc::new(Mutex::new(CapabilityState::default())),
            // PYTHON_REFERENCE_V1: rust=rust/crates/kogwistar-api/src/sections/in_memory_service.rs#default; python=kogwistar/conversation/agentic_answering.py#AgenticAnsweringAgent; rust_test=rust/crates/kogwistar-api/src/tests/answer_workflow_admission.rs#answer_workflow_admission_supports_per_node_python_and_registered_rust_executors; python_test=tests/workflow/test_agentic_answering_workflow_design_spec.py#test_answer_designs_match_shared_runtime_admission_contract
            native_operations: crate::native_answer_persist::registry()
                .expect("built-in Rust-native answer operation registry must be valid"),
            embedding_provider: None,
            chat_completion_provider: None,
            transient_run_deltas: crate::transient_deltas::TransientRunDeltas::default(),
            search_index_fts: None,
            owner_auth_store: None,
        }
    }
}

#[path = "in_memory_service/application_service.rs"]
mod application_service;
#[path = "in_memory_service/conversations.rs"]
mod conversations;
#[path = "in_memory_service/graph_operations.rs"]
pub(super) mod graph_operations;
#[path = "in_memory_service/native_worker.rs"]
mod native_worker;
#[path = "in_memory_service/run_lifecycle.rs"]
mod run_lifecycle;
