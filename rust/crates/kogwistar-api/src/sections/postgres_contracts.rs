#[derive(Clone)]
pub struct PostgresRunApplicationService {
    store: PostgresStore,
    max_queue: usize,
    capabilities: SharedCapabilityState,
    native_operations: kogwistar_runtime::NativeOperationRegistry,
    owner_auth_store: Option<Arc<AuthStore>>,
    embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
    embedding_storage_scope: Result<String, String>,
    chat_completion_provider: Option<crate::native_answer_generation::OpenAiCompatibleChatConfig>,
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
struct RuntimeOwnerBindingV1 {
    version: u32,
    user_id: String,
    admitted_capabilities: BTreeSet<String>,
    admitted_scope: Value,
}

#[derive(Clone)]
struct PreparedConversationAdmission {
    conversation_id: String,
    owner_id: String,
    expected_tail_id: Option<String>,
    scope: GraphScope,
    acl_principal: GraphAclPrincipal,
    mutations: Vec<(GraphMutation, kogwistar_contracts::GraphModelEnvelope)>,
    index_jobs: Vec<NewIndexJob>,
    actor_user_id: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RuntimeAdmissionOutcome {
    Queued,
    QueueFull,
    ConversationChanged,
}

fn parse_graph_neighbors_request(body: &[u8]) -> Result<GraphNeighborsRequestV1, String> {
    let payload = std::str::from_utf8(body).map_err(|error| error.to_string())?;
    let input = graph_neighbors_request_from_str(payload).map_err(|error| error.to_string())?;
    input.validate().map_err(str::to_owned)?;
    Ok(input)
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GraphVectorQueryWireRequest {
    entity_kind: String,
    query_embeddings: Vec<Vec<f32>>,
    metric: String,
    #[serde(default = "default_graph_truth_graph")]
    truth_graph: String,
    #[serde(default = "default_graph_vector_n_results")]
    n_results: usize,
    #[serde(default, rename = "where")]
    where_filter: Map<String, Value>,
    #[serde(default)]
    include: Option<Vec<String>>,
}

struct GraphVectorQueryRequest {
    entity_kind: String,
    query_embeddings: Vec<Vec<f32>>,
    metric: DistanceMetric,
    truth_graph: String,
    n_results: usize,
    where_filter: MetadataFilter,
    include: Vec<String>,
}

fn default_graph_vector_n_results() -> usize {
    10
}

fn default_graph_truth_graph() -> String {
    "knowledge".to_owned()
}

fn parse_graph_vector_query_request(body: &[u8]) -> Result<GraphVectorQueryRequest, String> {
    let wire: GraphVectorQueryWireRequest =
        serde_json::from_slice(body).map_err(|error| error.to_string())?;
    if !matches!(wire.entity_kind.as_str(), "node" | "edge") {
        return Err("entity_kind must be 'node' or 'edge'".to_owned());
    }
    if wire.truth_graph.trim().is_empty() || wire.truth_graph.len() > 255 {
        return Err("truth_graph must contain 1 to 255 characters".to_owned());
    }
    if wire.query_embeddings.is_empty() || wire.query_embeddings.len() > 32 {
        return Err("query_embeddings must contain 1 to 32 vectors".to_owned());
    }
    if wire
        .query_embeddings
        .iter()
        .any(|embedding| embedding.is_empty() || embedding.len() > 8192)
    {
        return Err("each query embedding must contain 1 to 8192 values".to_owned());
    }
    if !(1..=1000).contains(&wire.n_results) {
        return Err("n_results must be between 1 and 1000".to_owned());
    }
    if wire
        .where_filter
        .iter()
        .any(|(key, value)| key.starts_with('$') || value.is_object())
    {
        return Err("where supports exact scalar/array equality only".to_owned());
    }
    let metric = match wire.metric.trim().to_ascii_lowercase().as_str() {
        "cosine" => DistanceMetric::Cosine,
        "l2" => DistanceMetric::L2,
        "ip" | "inner" | "inner_product" | "innerproduct" => {
            DistanceMetric::InnerProduct
        }
        _ => return Err("metric must be cosine, l2, or ip".to_owned()),
    };
    let include = wire
        .include
        .unwrap_or_else(|| vec!["documents".into(), "embeddings".into(), "metadatas".into()]);
    if include.iter().any(|field| {
        !matches!(field.as_str(), "documents" | "embeddings" | "metadatas" | "distances")
    }) {
        return Err("include contains an unsupported field".to_owned());
    }
    Ok(GraphVectorQueryRequest {
        entity_kind: wire.entity_kind,
        query_embeddings: wire.query_embeddings,
        metric,
        truth_graph: wire.truth_graph,
        n_results: wire.n_results,
        where_filter: MetadataFilter {
            equals: wire.where_filter.into_iter().collect(),
        },
        include,
    })
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GraphFindEdgesRequest {
    relation: Option<String>,
    src_label_contains: Option<String>,
    tgt_label_contains: Option<String>,
    doc_id: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GraphShortestPathRequest {
    src_id: String,
    dst_id: String,
    doc_id: Option<String>,
    #[serde(default = "default_graph_max_depth")]
    max_depth: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GraphKHopRequest {
    start_ids: Vec<String>,
    #[serde(default = "default_graph_k")]
    k: usize,
    doc_id: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct InternalConversationTranscriptRequest {
    conversation_id: String,
    #[serde(default = "default_conversation_limit")]
    limit: usize,
}

struct ConversationTranscriptAuthorization {
    principal: Value,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct InternalContextSnapshotRequest {
    conversation_id: String,
    #[serde(default)]
    run_id: Option<String>,
    #[serde(default)]
    stage: Option<String>,
}

struct ContextSnapshotAuthorization {
    principal: Value,
}

fn default_conversation_limit() -> usize {
    10_000
}

fn default_graph_max_depth() -> usize {
    8
}

fn default_graph_k() -> usize {
    1
}

fn parse_graph_find_edges_request(body: &[u8]) -> Result<GraphFindEdgesRequest, String> {
    serde_json::from_slice(body).map_err(|error| error.to_string())
}

fn parse_internal_conversation_transcript_request(
    body: &[u8],
) -> Result<InternalConversationTranscriptRequest, String> {
    let input: InternalConversationTranscriptRequest =
        serde_json::from_slice(body).map_err(|error| error.to_string())?;
    validate_internal_conversation_transcript_request(&input)?;
    Ok(input)
}

fn validate_internal_conversation_transcript_request(
    input: &InternalConversationTranscriptRequest,
) -> Result<(), String> {
    if input.conversation_id.trim().is_empty() || input.limit > 10_000 {
        return Err("conversation_id is required and limit must be <= 10000".to_owned());
    }
    Ok(())
}

fn parse_internal_context_snapshot_request(
    body: &[u8],
) -> Result<InternalContextSnapshotRequest, String> {
    let input: InternalContextSnapshotRequest =
        serde_json::from_slice(body).map_err(|error| error.to_string())?;
    validate_internal_context_snapshot_request(&input)?;
    Ok(input)
}

fn validate_internal_context_snapshot_request(
    input: &InternalContextSnapshotRequest,
) -> Result<(), String> {
    if input.conversation_id.trim().is_empty()
        || input.run_id.as_deref().is_some_and(str::is_empty)
        || input.stage.as_deref().is_some_and(str::is_empty)
    {
        return Err("conversation_id is required and run_id/stage must not be empty".to_owned());
    }
    Ok(())
}

fn context_snapshot_syscall_field(
    args: &Map<String, Value>,
    field: &str,
) -> Result<Option<String>, String> {
    match args.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value.clone())),
        Some(_) => Err(format!("{field} must be a string or null")),
    }
}
