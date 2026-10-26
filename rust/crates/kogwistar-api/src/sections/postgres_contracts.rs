#[derive(Clone)]
pub struct PostgresRunApplicationService {
    store: PostgresStore,
    max_queue: usize,
    capabilities: SharedCapabilityState,
    native_operations: kogwistar_runtime::NativeOperationRegistry,
}

fn parse_graph_neighbors_request(body: &[u8]) -> Result<GraphNeighborsRequestV1, String> {
    let payload = std::str::from_utf8(body).map_err(|error| error.to_string())?;
    let input = graph_neighbors_request_from_str(payload).map_err(|error| error.to_string())?;
    input.validate().map_err(str::to_owned)?;
    Ok(input)
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

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct InternalContextSnapshotRequest {
    conversation_id: String,
    #[serde(default)]
    run_id: Option<String>,
    #[serde(default)]
    stage: Option<String>,
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
    if input.conversation_id.trim().is_empty() || input.limit > 10_000 {
        return Err("conversation_id is required and limit must be <= 10000".to_owned());
    }
    Ok(input)
}

fn parse_internal_context_snapshot_request(
    body: &[u8],
) -> Result<InternalContextSnapshotRequest, String> {
    let input: InternalContextSnapshotRequest =
        serde_json::from_slice(body).map_err(|error| error.to_string())?;
    if input.conversation_id.trim().is_empty()
        || input.run_id.as_deref().is_some_and(str::is_empty)
        || input.stage.as_deref().is_some_and(str::is_empty)
    {
        return Err("conversation_id is required and run_id/stage must not be empty".to_owned());
    }
    Ok(input)
}
