// PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/in_memory_service.rs#graph_read_effect; python=kogwistar/graph_query.py#GraphQuery.neighbors; fixture=contracts/golden/graph-acl-traversal-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/part_08.rs#in_memory_mcp_acl_traversal_matches_shared_python_contract; python_test=tests/mcp/test_chat_mcp_tools.py#test_mcp_neighbors_preserve_acl_visibility_and_hide_topology
#[derive(Clone)]
struct InMemoryGraphApplicationService {
    store: InMemoryStore,
    capabilities: SharedCapabilityState,
    embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
    search_index_fts: Option<Arc<Mutex<rusqlite::Connection>>>,
}

impl Default for InMemoryGraphApplicationService {
    fn default() -> Self {
        Self {
            store: InMemoryStore::default(),
            capabilities: Arc::new(Mutex::new(CapabilityState::default())),
            embedding_provider: None,
            search_index_fts: None,
        }
    }
}

impl InMemoryGraphApplicationService {
    // PYTHON_REFERENCE_V1: rust=rust/crates/kogwistar-api/src/sections/in_memory_service.rs#execute_native_step_once; python=kogwistar/runtime/rust_worker.py#RustRuntimeWorker; fixture=contracts/golden/runtime-native-noop-v1.json,contracts/golden/runtime-native-increment-counter-v1.json,contracts/golden/runtime-native-emit-result-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/part_08.rs#in_memory_inline_native_result_is_persisted_and_completes_run; python_test=tests/runtime/test_rust_runtime_python_worker.py#test_python_worker_increment_counter_matches_shared_native_effect_golden; note=Rust-native operations stay on Rust path; Python worker explicitly rejects native-target operations.
    async fn dead_letter_native_lane(
        &self,
        message_id: &str,
        claimed_by: &str,
        reason: &str,
    ) -> Result<(), String> {
        kogwistar_store::LaneMessageWriteStore::dead_letter_projected_lane_message(
            &self.store,
            message_id,
            claimed_by,
            Some(json!({"error": reason}).to_string()),
        )
        .await
        .map_err(|error| error.to_string())
    }

    async fn execute_native_step_once(&self) -> Result<bool, String> {
        let claimed_by = format!("rust-native-worker#{}", uuid::Uuid::new_v4());
        let mut claimed =
            kogwistar_store::LaneMessageWriteStore::claim_projected_lane_messages_filtered(
                &self.store,
                LaneMessageClaimFilter {
                    namespace: "workflow".to_owned(),
                    inbox_id: "workflow-runtime".to_owned(),
                    claimed_by: claimed_by.clone(),
                    message_ids: None,
                    run_id: None,
                    msg_type: Some("workflow.step.execute".to_owned()),
                    recipient_id: Some("rust-native-worker".to_owned()),
                    limit: 1,
                    lease_seconds: 60,
                },
            )
            .await
            .map_err(|error| error.to_string())?;
        let Some(lane) = claimed.pop() else {
            return Ok(false);
        };
        let message_id = lane.message_id.clone();
        let run_id = lane.run_id.clone().unwrap_or_default();
        let cancel_requested =
            kogwistar_store::ServerRunReadStore::server_run(&self.store, &run_id)
                .await
                .map_err(|error| error.to_string())?
                .is_some_and(|run| run.cancel_requested);
        if cancel_requested {
            kogwistar_store::LaneMessageWriteStore::update_projected_lane_message_status(
                &self.store,
                &message_id,
                "cancelled",
                None,
            )
            .await
            .map_err(|error| error.to_string())?;
            return Ok(true);
        }
        let payload: Value = match lane
            .payload_json
            .as_deref()
            .map(serde_json::from_str)
            .transpose()
        {
            Ok(Some(payload)) => payload,
            Ok(None) => json!({}),
            Err(error) => {
                self.dead_letter_native_lane(
                    &message_id,
                    &claimed_by,
                    "invalid native step payload",
                )
                .await?;
                return Err(format!("invalid native step payload: {error}"));
            }
        };
        let operation = payload["op"].as_str().and_then(|operation| {
            kogwistar_runtime::native_operation_for_target(
                operation,
                payload["executor_target"].as_str(),
            )
        });
        if !matches!(
            operation,
            Some("noop" | "increment-counter" | "emit-result")
        ) {
            self.dead_letter_native_lane(
                &message_id,
                &claimed_by,
                "in-memory inline runtime does not support this native operation",
            )
            .await?;
            return Ok(true);
        }
        let expected_event_seq = payload["expected_event_seq"].as_i64().unwrap_or_default();
        let work = runtime_claimed_work(lane, &claimed_by, run_id, payload, expected_event_seq);
        let (handoff, native_request) = match native_request_for_claimed_work(&work) {
            Ok(value) => value,
            Err(error) => {
                self.dead_letter_native_lane(
                    &message_id,
                    &claimed_by,
                    "native step request validation failed",
                )
                .await?;
                return Err(error);
            }
        };
        let result = match native_result_from_request(
            handoff.clone(),
            native_request,
            &kogwistar_runtime::NativeOperationRegistry::default(),
        ) {
            Ok(result) => result,
            Err(error) => {
                self.dead_letter_native_lane(
                    &message_id,
                    &claimed_by,
                    "native noop execution failed",
                )
                .await?;
                return Err(error);
            }
        };
        let Some(effect) = result.effect else {
            self.dead_letter_native_lane(
                &message_id,
                &claimed_by,
                "native noop did not produce a worker effect",
            )
            .await?;
            return Err("native noop did not produce a worker effect".to_owned());
        };
        if let Err(error) = self
            .store
            .apply_claimed_recorded_worker_effect(handoff, effect)
        {
            self.dead_letter_native_lane(
                &message_id,
                &claimed_by,
                "in-memory worker effect could not be committed",
            )
            .await?;
            return Err(error.to_string());
        }
        Ok(true)
    }

    fn supports_graph_tool(name: &str) -> bool {
        matches!(
            name,
            "kg_find_edges"
                | "kg_k_hop"
                | "kg_load_persisted"
                | "kg_neighbors"
                | "kg_propose_bruteforce"
                | "kg_shortest_path"
                | "kg_semantic_seed_then_expand_text"
                | "kg_viz_cytoscape_json"
                | "kg_viz_d3_json"
                | "propose_vector"
        )
    }

    fn with_embedding_provider(mut self, provider: Arc<dyn EmbeddingProvider>) -> Self {
        self.embedding_provider = Some(provider);
        self
    }

    fn with_search_index_path(mut self, path: impl AsRef<std::path::Path>) -> Result<Self, String> {
        let connection = crate::search_index_fts::open_index(path)
            .map_err(|error| format!("could not open search-index FTS database: {error}"))?;
        self.search_index_fts = Some(Arc::new(Mutex::new(connection)));
        Ok(self)
    }

    fn authorized_graph_scope(
        &self,
        request: &ApiEffectRequest,
    ) -> Result<GraphScope, ApiEffectResponse> {
        if !capability_allowed(request, &self.capabilities, "read_graph", &["read_graph"]) {
            return Err(effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Graph read requires read_graph capability",
            ));
        }
        let namespaces = principal_strings(&request.principal, "ns");
        if !namespaces
            .iter()
            .any(|namespace| matches!(namespace.as_str(), "*" | "docs"))
        {
            return Err(effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_NAMESPACE_FORBIDDEN",
                "Graph reads require docs namespace",
            ));
        }
        let scope_value = security_scope_value(&request.principal);
        Ok(GraphScope {
            namespace: scope_value["storage_namespace"]
                .as_str()
                .unwrap_or("default")
                .to_owned(),
            workspace_id: request
                .principal
                .get("workspace_id")
                .or_else(|| request.principal.get("workspace"))
                .and_then(Value::as_str)
                .map(str::to_owned),
            graph_space: request
                .principal
                .get("graph_space")
                .and_then(Value::as_str)
                .map(str::to_owned),
        })
    }

    async fn graph_snapshot(
        &self,
        scope: GraphScope,
    ) -> Result<(Vec<GraphRecord>, Vec<GraphRecord>), String> {
        let (nodes, edges) = self
            .store
            .graph_node_edge_snapshot(&scope, 10_001)
            .map_err(|error| error.to_string())?;
        if nodes.len() > 10_000 || edges.len() > 10_000 {
            return Err("in-memory graph snapshot exceeds the 10000-record bound".to_owned());
        }
        Ok((nodes, edges))
    }

    async fn graph_read_effect(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        if request.path_and_query == INTERNAL_GRAPH_LOAD_PERSISTED_ROUTE {
            return self.graph_load_persisted_effect(request).await;
        }
        if request.path_and_query == INTERNAL_GRAPH_PROPOSE_BRUTE_FORCE_ROUTE {
            return self.graph_propose_bruteforce_effect(request).await;
        }
        if request.path_and_query == INTERNAL_GRAPH_VECTOR_QUERY_ROUTE {
            return self.graph_vector_query_effect(request).await;
        }
        if request.path_and_query == GRAPH_SEMANTIC_SEED_EXPAND_ROUTE {
            return self.graph_semantic_seed_expand_effect(request).await;
        }
        if request.path_and_query == INTERNAL_GRAPH_PROPOSE_VECTOR_ROUTE {
            return self.graph_vector_proposal_effect(request).await;
        }
        let scope = match self.authorized_graph_scope(request) {
            Ok(scope) => scope,
            Err(response) => return response,
        };
        let truth_graph = if request.path_and_query == INTERNAL_GRAPH_NEIGHBORS_ROUTE {
            match parse_graph_neighbors_request(&request.body) {
                Ok(input) => input.truth_graph,
                Err(error) => return invalid_graph_request(error),
            }
        } else if matches!(
            request.path_and_query.as_str(),
            INTERNAL_GRAPH_VIZ_D3_ROUTE | INTERNAL_GRAPH_VIZ_CYTOSCAPE_ROUTE
        ) {
            let input: GraphVisualizationRequest = match serde_json::from_slice(&request.body) {
                Ok(input) => input,
                Err(error) => return invalid_graph_request(error.to_string()),
            };
            input.truth_graph().to_owned()
        } else {
            "knowledge".to_owned()
        };
        let (nodes, edges) = match self.graph_snapshot(scope).await {
            Ok(records) => records,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error,
                );
            }
        };
        let scope_value = security_scope_value(&request.principal);
        let principal = graph_acl_principal_from_claims(
            &request.principal,
            scope_value["security_scope"].as_str(),
        );
        let (nodes, edges) =
            graph_records_visible_by_acl(&nodes, &edges, truth_graph.trim(), &principal);
        let result = match request.path_and_query.as_str() {
            INTERNAL_GRAPH_NEIGHBORS_ROUTE => {
                let input = match parse_graph_neighbors_request(&request.body) {
                    Ok(input) => input,
                    Err(error) => return invalid_graph_request(error),
                };
                let neighbors = graph_neighbors_from_records(
                    input.rid.trim(),
                    &nodes,
                    &edges,
                    input.doc_id.as_deref(),
                );
                json!({"nodes": neighbors.nodes, "edges": neighbors.edges})
            }
            INTERNAL_GRAPH_FIND_EDGES_ROUTE => {
                let input = match parse_graph_find_edges_request(&request.body) {
                    Ok(input) => input,
                    Err(error) => return invalid_graph_request(error),
                };
                let query = GraphFindEdgesQuery {
                    relation: input.relation,
                    src_label_contains: input.src_label_contains,
                    tgt_label_contains: input.tgt_label_contains,
                    doc_id: input.doc_id,
                };
                json!({"edges": graph_find_edges_from_records(&query, &nodes, &edges)})
            }
            INTERNAL_GRAPH_SHORTEST_PATH_ROUTE => {
                let input: GraphShortestPathRequest = match serde_json::from_slice(&request.body) {
                    Ok(input) => input,
                    Err(error) => return invalid_graph_request(error.to_string()),
                };
                if input.src_id.trim().is_empty()
                    || input.dst_id.trim().is_empty()
                    || input.max_depth > 64
                {
                    return invalid_graph_request(
                        "src_id and dst_id are required and max_depth must be <= 64".to_owned(),
                    );
                }
                let query = GraphShortestPathQuery {
                    src_id: input.src_id,
                    dst_id: input.dst_id,
                    doc_id: input.doc_id,
                    max_depth: input.max_depth,
                };
                json!({"path": graph_shortest_path_from_records(&query, &nodes, &edges)})
            }
            INTERNAL_GRAPH_K_HOP_ROUTE => {
                let input: GraphKHopRequest = match serde_json::from_slice(&request.body) {
                    Ok(input) => input,
                    Err(error) => return invalid_graph_request(error.to_string()),
                };
                if input.start_ids.is_empty()
                    || input.start_ids.len() > 100
                    || input.start_ids.iter().any(|id| id.trim().is_empty())
                    || input.k > 8
                {
                    return invalid_graph_request(
                        "start_ids must contain 1 to 100 IDs and k must be <= 8".to_owned(),
                    );
                }
                let query = GraphKHopQuery {
                    start_ids: input.start_ids,
                    k: input.k,
                    doc_id: input.doc_id,
                };
                let layers = graph_k_hop_from_records(&query, &nodes, &edges)
                    .into_iter()
                    .map(|layer| json!({"nodes": layer.nodes, "edges": layer.edges}))
                    .collect::<Vec<_>>();
                json!({"layers": layers})
            }
            INTERNAL_GRAPH_VIZ_D3_ROUTE | INTERNAL_GRAPH_VIZ_CYTOSCAPE_ROUTE => {
                let input: GraphVisualizationRequest = match serde_json::from_slice(&request.body) {
                    Ok(input) => input,
                    Err(error) => return invalid_graph_request(error.to_string()),
                };
                if request.path_and_query == INTERNAL_GRAPH_VIZ_CYTOSCAPE_ROUTE {
                    graph_viz_cytoscape_from_records(
                        &nodes,
                        &edges,
                        input.doc_id.as_deref(),
                        &input.mode,
                        input.insertion_method.as_deref(),
                    )
                } else {
                    graph_viz_d3_from_records(
                        &nodes,
                        &edges,
                        input.doc_id.as_deref(),
                        &input.mode,
                        input.insertion_method.as_deref(),
                    )
                }
            }
            _ => return unavailable_effect(request.clone()),
        };
        json_effect(StatusCode::OK, rust_native_attested(result))
    }

    // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/in_memory_service.rs#graph_semantic_seed_expand_effect; python=kogwistar/graph_query.py#GraphQuery.semantic_seed_then_expand_text; fixture=contracts/golden/semantic-seed-expand-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/part_08.rs#in_memory_semantic_seed_expand_is_available_through_authenticated_http_mcp; python_test=tests/primitives/test_graph_query.py#test_semantic_seed_then_expand_text
    async fn graph_semantic_seed_expand_effect(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        let scope = match self.authorized_graph_scope(request) {
            Ok(scope) => scope,
            Err(response) => return response,
        };
        let input: SemanticSeedExpandRequest = match serde_json::from_slice(&request.body) {
            Ok(input) => input,
            Err(error) => return invalid_graph_request(error.to_string()),
        };
        let query_text = input.text.trim();
        let top_k = input.top_k.unwrap_or(5);
        let hops = input.hops.unwrap_or(1);
        if query_text.is_empty()
            || query_text.chars().count() > 16_384
            || !(1..=1000).contains(&top_k)
            || hops > 32
        {
            return invalid_graph_request(
                "text must contain 1 to 16384 characters, top_k must be 1 to 1000, and hops must be at most 32".to_owned(),
            );
        }
        let Some(provider) = self.embedding_provider.as_ref() else {
            return effect_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "KOGWISTAR_EMBEDDING_UNAVAILABLE",
                "Semantic graph retrieval requires a configured embedding provider",
            );
        };
        let (embedding, _embedding_dim, metric) =
            match semantic_query_embedding(provider.as_ref(), query_text).await {
                Ok(result) => result,
                Err(response) => return response,
            };
        let mut metadata = BTreeMap::new();
        if let Some(workspace_id) = scope.workspace_id.as_ref() {
            metadata.insert("workspace_id".to_owned(), json!(workspace_id));
        }
        if let Some(graph_space) = scope.graph_space.as_ref() {
            metadata.insert("graph_space".to_owned(), json!(graph_space));
        }
        if let Some(doc_id) = input.doc_ids.as_ref() {
            metadata.insert("doc_id".to_owned(), json!(doc_id));
        }
        let query = VectorQuery {
            embedding,
            limit: 10_000,
            metadata: MetadataFilter { equals: metadata },
            metric,
        };
        let (all_nodes, all_edges, mut query_matches) = match self
            .store
            .graph_node_edge_vector_snapshot(&scope, 10_001, "gke_nodes", &[query])
        {
            Ok(snapshot) => snapshot,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_VECTOR_QUERY_ERROR",
                    error.to_string(),
                );
            }
        };
        if all_nodes.len() > 10_000 || all_edges.len() > 10_000 {
            return effect_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "KOGWISTAR_STORE_ERROR",
                "in-memory graph snapshot exceeds the 10000-record bound",
            );
        }
        let scope_value = security_scope_value(&request.principal);
        let principal = graph_acl_principal_from_claims(
            &request.principal,
            scope_value["security_scope"].as_str(),
        );
        let (visible_nodes, visible_edges) =
            graph_records_visible_by_acl(&all_nodes, &all_edges, "knowledge", &principal);
        let visible_node_ids = visible_nodes
            .iter()
            .map(|record| record.id.as_str())
            .collect::<BTreeSet<_>>();
        let matches = query_matches.pop().unwrap_or_default();
        let visible_seeds = matches
            .into_iter()
            .filter(|matched| {
                matched
                    .record
                    .metadata
                    .get("entity_type")
                    .and_then(Value::as_str)
                    != Some("acl_record")
                    && !matched
                        .record
                        .metadata
                        .get("lifecycle_status")
                        .and_then(Value::as_str)
                        .is_some_and(|status| matches!(status, "tombstoned" | "deleted"))
            })
            .take(top_k)
            .map(|matched| matched.record)
            .filter(|record| visible_node_ids.contains(record.id.as_str()))
            .collect::<Vec<_>>();
        let seed_ids = visible_seeds
            .iter()
            .map(|record| record.id.clone())
            .collect::<Vec<_>>();
        let seeds = visible_seeds
            .iter()
            .filter_map(graph_record_document_json)
            .collect::<Vec<_>>();
        let layers = graph_k_hop_from_records(
            &GraphKHopQuery {
                start_ids: seed_ids,
                k: hops,
                doc_id: None,
            },
            &visible_nodes,
            &visible_edges,
        )
        .into_iter()
        .map(|layer| {
            let nodes = layer
                .nodes
                .iter()
                .filter_map(|id| visible_nodes.iter().find(|record| &record.id == id))
                .filter_map(graph_record_document_json)
                .collect::<BTreeSet<_>>();
            let edges = layer
                .edges
                .iter()
                .filter_map(|id| visible_edges.iter().find(|record| &record.id == id))
                .filter_map(graph_record_document_json)
                .collect::<BTreeSet<_>>();
            json!({
                "nodes": nodes.into_iter().collect::<Vec<_>>(),
                "edges": edges.into_iter().collect::<Vec<_>>(),
            })
        })
        .collect::<Vec<_>>();
        json_effect(
            StatusCode::OK,
            rust_native_attested(json!({"seeds": seeds, "layers": layers})),
        )
    }

    // PYTHON_PARITY: kogwistar/server/mcp_tools.py::_load_persisted_graph
    async fn graph_load_persisted_effect(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        let scope = match self.authorized_graph_scope(request) {
            Ok(scope) => scope,
            Err(response) => return response,
        };
        let input = match serde_json::from_slice::<LoadPersistedGraphRequest>(&request.body) {
            Ok(input) => input,
            Err(error) => {
                return invalid_graph_request(format!(
                    "Invalid persisted graph lookup request: {error}"
                ));
            }
        };
        if input.doc_ids.is_empty()
            || input.doc_ids.len() > 1000
            || input.doc_ids.iter().any(|doc| doc.trim().is_empty())
            || input
                .insertion_method
                .as_ref()
                .is_some_and(|method| method.trim().is_empty())
        {
            return invalid_graph_request(
                "doc_ids must contain 1 to 1000 non-empty IDs and insertion_method must be non-empty when provided"
                    .to_owned(),
            );
        }
        let (all_nodes, all_edges) = match self.graph_snapshot(scope).await {
            Ok(records) => records,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error,
                );
            }
        };
        let scope_value = security_scope_value(&request.principal);
        let principal = graph_acl_principal_from_claims(
            &request.principal,
            scope_value["security_scope"].as_str(),
        );
        let (nodes, edges) =
            graph_records_visible_by_acl(&all_nodes, &all_edges, "knowledge", &principal);
        let (node_ids, edge_ids) = graph_load_persisted_ids_from_records(&nodes, &edges, &input);
        json_effect(
            StatusCode::OK,
            rust_native_attested(json!({"node_ids": node_ids, "edge_ids": edge_ids})),
        )
    }

    // PYTHON_PARITY: kogwistar/server/mcp_tools.py::kg_propose_bruteforce
    async fn graph_propose_bruteforce_effect(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        let scope = match self.authorized_graph_scope(request) {
            Ok(scope) => scope,
            Err(response) => return response,
        };
        let input = match serde_json::from_slice::<ProposeBruteForceRequest>(&request.body) {
            Ok(input) => input,
            Err(error) => {
                return invalid_graph_request(format!(
                    "Invalid brute-force proposal request: {error}"
                ));
            }
        };
        if brute_force_pair_kinds(&input.pair_kind).is_none()
            || input.allowed_docs.as_ref().is_some_and(|docs| {
                docs.len() > 1000 || docs.iter().any(|doc| doc.trim().is_empty())
            })
            || input
                .anchor_doc_id
                .as_ref()
                .is_some_and(|doc| doc.trim().is_empty())
            || input.limit_per_bucket.is_some_and(|limit| limit > 5000)
            || input
                .where_filter
                .as_ref()
                .is_some_and(|value| !value.is_object())
        {
            return invalid_graph_request(
                "pair_kind, document IDs, where, or limit_per_bucket are invalid or exceed their bounds"
                    .to_owned(),
            );
        }
        if let Some(where_filter) = input.where_filter.as_ref() {
            if let Err(error) = kogwistar_contracts::metadata_filter_matches_from_str(
                &json!({"metadata": {}, "where": where_filter}).to_string(),
            ) {
                return invalid_graph_request(error.to_string());
            }
        }
        let (all_nodes, all_edges) = match self.graph_snapshot(scope).await {
            Ok(records) => records,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error,
                );
            }
        };
        let scope_value = security_scope_value(&request.principal);
        let principal = graph_acl_principal_from_claims(
            &request.principal,
            scope_value["security_scope"].as_str(),
        );
        let (nodes, edges) =
            graph_records_visible_by_acl(&all_nodes, &all_edges, "knowledge", &principal);
        let (node_ids_by_doc, edge_ids_by_doc) =
            brute_force_document_indexes_from_records(&nodes, &edges);
        let mut docs = match input.allowed_docs.as_ref().filter(|docs| !docs.is_empty()) {
            Some(docs) => docs
                .iter()
                .cloned()
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>(),
            None => node_ids_by_doc
                .keys()
                .chain(edge_ids_by_doc.keys())
                .cloned()
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect(),
        };
        if let Some(anchor) = input.anchor_doc_id.as_ref()
            && !docs.contains(anchor)
        {
            docs.insert(0, anchor.clone());
        }
        if docs.len() > 1000 {
            return invalid_graph_request(
                "brute-force proposal is limited to 1000 documents".to_owned(),
            );
        }
        let nodes = nodes
            .into_iter()
            .map(|record| (record.id.clone(), record))
            .collect::<BTreeMap<_, _>>();
        let edges = edges
            .into_iter()
            .map(|record| (record.id.clone(), record))
            .collect::<BTreeMap<_, _>>();
        let pairs = match brute_force_proposal_pairs(
            &docs,
            &node_ids_by_doc,
            &edge_ids_by_doc,
            &nodes,
            &edges,
            &input,
        ) {
            Ok(pairs) => pairs,
            Err(error) => return invalid_graph_request(error),
        };
        json_effect(
            StatusCode::OK,
            rust_native_attested(json!({"pairs": pairs})),
        )
    }

    // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/in_memory_service.rs#graph_vector_proposal_effect; python=kogwistar/server/mcp_tools.py#propose_vector; fixture=contracts/golden/graph-vector-proposal-acl-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/part_08.rs#in_memory_backend_routes_authenticated_graph_write_and_mcp_read; python_test=tests/mcp/test_chat_mcp_tools.py#test_propose_vector_returns_only_acl_visible_pairs
    // Mirrors Python's propose_vector contract for the in-memory backend only;
    // PostgreSQL and SQLite+Chroma remain unadvertised until their snapshots
    // can provide the same pre-top-K filtering and ACL guarantees.
    async fn graph_vector_proposal_effect(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        let scope = match self.authorized_graph_scope(request) {
            Ok(scope) => scope,
            Err(response) => return response,
        };
        let input = match GraphVectorProposalRequest::parse(&request.body) {
            Ok(input) => input,
            Err(error) => return invalid_graph_request(error),
        };
        let where_filter = input
            .where_filter
            .as_ref()
            .filter(|value| value.is_object());
        let validation = json!({"metadata": {}, "where": where_filter});
        if let Err(error) =
            kogwistar_contracts::metadata_filter_matches_from_str(&validation.to_string())
        {
            return invalid_graph_request(error.to_string());
        }
        let (all_nodes, all_edges) = match self.graph_snapshot(scope.clone()).await {
            Ok(records) => records,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error,
                );
            }
        };
        let scope_value = security_scope_value(&request.principal);
        let principal = graph_acl_principal_from_claims(
            &request.principal,
            scope_value["security_scope"].as_str(),
        );
        let (visible_nodes, visible_edges) =
            graph_records_visible_by_acl(&all_nodes, &all_edges, "knowledge", &principal);
        let visible_node_ids = visible_nodes
            .iter()
            .map(|record| record.id.as_str())
            .collect::<BTreeSet<_>>();
        let visible_edge_ids = visible_edges
            .iter()
            .map(|record| record.id.as_str())
            .collect::<BTreeSet<_>>();
        let node_seeds = selected_proposal_seeds(&all_nodes, input.new_node_ids.as_deref());
        let edge_seeds = selected_proposal_seeds(&all_edges, input.new_edge_ids.as_deref());
        let mut pairs = Vec::new();
        let mut seen = BTreeSet::new();
        for (seed, seed_kind) in node_seeds
            .into_iter()
            .map(|record| (record, "node"))
            .chain(edge_seeds.into_iter().map(|record| (record, "edge")))
        {
            if seed.metadata.get("entity_type").and_then(Value::as_str) == Some("acl_record") {
                continue;
            }
            if !(if seed_kind == "node" {
                visible_node_ids.contains(seed.id.as_str())
            } else {
                visible_edge_ids.contains(seed.id.as_str())
            }) {
                continue;
            }
            let Some(query) = seed.embedding.as_deref() else {
                continue;
            };
            let seed_doc = proposal_doc_id(seed);
            for (candidates, candidate_kind) in [
                (all_nodes.as_slice(), "node"),
                (all_edges.as_slice(), "edge"),
            ] {
                if candidate_kind == "edge" && !input.include_edges {
                    continue;
                }
                let mut ranked = Vec::new();
                for candidate in candidates {
                    if candidate
                        .metadata
                        .get("entity_type")
                        .and_then(Value::as_str)
                        == Some("acl_record")
                    {
                        continue;
                    }
                    if candidate_kind == "node" && candidate.id == seed.id {
                        continue;
                    }
                    if candidate
                        .metadata
                        .get("lifecycle_status")
                        .and_then(Value::as_str)
                        .is_some_and(|status| matches!(status, "tombstoned" | "deleted"))
                    {
                        continue;
                    }
                    let where_matches = match proposal_where_matches(candidate, where_filter) {
                        Ok(matches) => matches,
                        Err(error) => return invalid_graph_request(error),
                    };
                    if !where_matches {
                        continue;
                    }
                    let candidate_doc = proposal_doc_id(candidate);
                    if input.allowed_docs.as_ref().is_some_and(|docs| {
                        !docs.is_empty()
                            && candidate_doc.is_none_or(|doc| !docs.iter().any(|d| d == doc))
                    }) {
                        continue;
                    }
                    if input.cross_doc_only && seed_doc.is_some() && seed_doc == candidate_doc {
                        continue;
                    }
                    if input.anchor_doc_id.as_ref().is_some_and(|anchor| {
                        input.anchor_only
                            && seed_doc != Some(anchor.as_str())
                            && candidate_doc != Some(anchor.as_str())
                    }) {
                        continue;
                    }
                    let Some(distance) =
                        proposal_cosine_distance(query, candidate.embedding.as_deref())
                    else {
                        continue;
                    };
                    let in_threshold = if input.score_mode == "distance" {
                        distance <= input.max_distance
                    } else {
                        distance >= input.min_similarity
                    };
                    if in_threshold {
                        ranked.push((candidate, distance));
                    }
                }
                ranked.sort_by(|left, right| left.1.total_cmp(&right.1));
                for (candidate, _) in ranked.into_iter().take(input.top_k) {
                    let candidate_visible = if candidate_kind == "node" {
                        visible_node_ids.contains(candidate.id.as_str())
                    } else {
                        visible_edge_ids.contains(candidate.id.as_str())
                    };
                    if !candidate_visible {
                        continue;
                    }
                    let (left, right, left_kind, right_kind) = if seed.id <= candidate.id {
                        (&seed.id, &candidate.id, seed_kind, candidate_kind)
                    } else {
                        (&candidate.id, &seed.id, candidate_kind, seed_kind)
                    };
                    let key = (left.clone(), right.clone());
                    if seen.insert(key) {
                        pairs.push(json!({
                            "left_id": left,
                            "left_kind": left_kind,
                            "right_id": right,
                            "right_kind": right_kind,
                        }));
                    }
                }
            }
        }
        json_effect(
            StatusCode::OK,
            rust_native_attested(json!({"pairs": pairs})),
        )
    }

    async fn graph_vector_query_effect(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        let scope = match self.authorized_graph_scope(request) {
            Ok(scope) => scope,
            Err(response) => return response,
        };
        let input = match parse_graph_vector_query_request(&request.body) {
            Ok(input) => input,
            Err(error) => return invalid_graph_request(error),
        };
        let table = if input.entity_kind == "node" {
            "gke_nodes"
        } else {
            "gke_edges"
        };
        let scope_value = security_scope_value(&request.principal);
        let acl_principal = graph_acl_principal_from_claims(
            &request.principal,
            scope_value["security_scope"].as_str(),
        );
        let queries = input
            .query_embeddings
            .iter()
            .map(|embedding| VectorQuery {
                embedding: embedding.clone(),
                limit: 10_000,
                metadata: input.where_filter.clone(),
                metric: input.metric,
            })
            .collect::<Vec<_>>();
        let (nodes, edges, vector_matches) = match self
            .store
            .graph_node_edge_vector_snapshot(&scope, 10_001, table, &queries)
        {
            Ok(records) => records,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_VECTOR_QUERY_ERROR",
                    error.to_string(),
                );
            }
        };
        if nodes.len() > 10_000 || edges.len() > 10_000 {
            return effect_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "KOGWISTAR_STORE_ERROR",
                "in-memory graph snapshot exceeds the 10000-record bound",
            );
        }
        let (visible_nodes, visible_edges) =
            graph_records_visible_by_acl(&nodes, &edges, &input.truth_graph, &acl_principal);
        let visible_ids = if input.entity_kind == "node" {
            visible_nodes
                .iter()
                .map(|record| record.id.as_str())
                .collect::<BTreeSet<_>>()
        } else {
            visible_edges
                .iter()
                .map(|record| record.id.as_str())
                .collect::<BTreeSet<_>>()
        };
        let mut ids = Vec::with_capacity(input.query_embeddings.len());
        let mut documents = Vec::with_capacity(input.query_embeddings.len());
        let mut metadatas = Vec::with_capacity(input.query_embeddings.len());
        let mut embeddings = Vec::with_capacity(input.query_embeddings.len());
        let mut distances = Vec::with_capacity(input.query_embeddings.len());
        for matches in vector_matches {
            let matches = matches
                .into_iter()
                .filter(|matched| {
                    matched
                        .record
                        .metadata
                        .get("entity_type")
                        .and_then(Value::as_str)
                        != Some("acl_record")
                        && !matched
                            .record
                            .metadata
                            .get("lifecycle_status")
                            .and_then(Value::as_str)
                            .is_some_and(|status| matches!(status, "tombstoned" | "deleted"))
                })
                .take(input.n_results)
                .filter(|matched| visible_ids.contains(matched.record.id.as_str()))
                .collect::<Vec<_>>();
            ids.push(
                matches
                    .iter()
                    .map(|matched| json!(matched.record.id))
                    .collect::<Vec<_>>(),
            );
            if input.include.iter().any(|field| field == "documents") {
                documents.push(
                    matches
                        .iter()
                        .map(|matched| json!(matched.record.document))
                        .collect::<Vec<_>>(),
                );
            }
            if input.include.iter().any(|field| field == "metadatas") {
                metadatas.push(
                    matches
                        .iter()
                        .map(|matched| json!(matched.record.metadata))
                        .collect::<Vec<_>>(),
                );
            }
            if input.include.iter().any(|field| field == "embeddings") {
                embeddings.push(
                    matches
                        .iter()
                        .map(|matched| json!(matched.record.embedding))
                        .collect::<Vec<_>>(),
                );
            }
            if input.include.iter().any(|field| field == "distances") {
                distances.push(
                    matches
                        .iter()
                        .map(|matched| json!(matched.distance))
                        .collect::<Vec<_>>(),
                );
            }
        }
        let mut response = serde_json::Map::new();
        response.insert("ids".to_owned(), json!(ids));
        if input.include.iter().any(|field| field == "documents") {
            response.insert("documents".to_owned(), json!(documents));
        }
        if input.include.iter().any(|field| field == "metadatas") {
            response.insert("metadatas".to_owned(), json!(metadatas));
        }
        if input.include.iter().any(|field| field == "embeddings") {
            response.insert("embeddings".to_owned(), json!(embeddings));
        }
        if input.include.iter().any(|field| field == "distances") {
            response.insert("distances".to_owned(), json!(distances));
        }
        json_effect(
            StatusCode::OK,
            rust_native_attested(Value::Object(response)),
        )
    }

    async fn graph_mutation_effect(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        if !capability_allowed(request, &self.capabilities, "write_graph", &["write_graph"]) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Graph mutation requires write_graph capability",
            );
        }
        if !principal_strings(&request.principal, "ns")
            .iter()
            .any(|namespace| matches!(namespace.as_str(), "*" | "docs"))
        {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_NAMESPACE_FORBIDDEN",
                "Graph mutations require docs namespace",
            );
        }
        let input: GraphMutationBatchInput = match serde_json::from_slice(&request.body) {
            Ok(input) => input,
            Err(error) => return invalid_graph_request(error.to_string()),
        };
        if let Err(error) = input.validate() {
            return invalid_graph_request(error.to_string());
        }
        let mutations = match graph_mutation_batch_value(input.clone()) {
            Ok(mutations) => mutations,
            Err(error) => return invalid_graph_request(error),
        };
        if let Err(error) = validate_graph_mutation_envelopes(&input, &mutations) {
            return invalid_graph_request(error);
        }
        let security_scope = security_scope_value(&request.principal);
        if mutations
            .iter()
            .any(|mutation| !graph_mutation_scope_allowed(&mutation.scope, &security_scope))
        {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_SCOPE_FORBIDDEN",
                "Graph mutation scope is outside the principal storage scope",
            );
        }
        let applied = if input.index_jobs.is_empty() {
            self.store.apply_graph_mutations_atomic(mutations).await
        } else {
            let index_jobs = match graph_index_jobs_value(&input, &mutations) {
                Ok(jobs) => jobs,
                Err(error) => return invalid_graph_request(error),
            };
            self.store
                .apply_graph_mutations_with_index_jobs_atomic(GraphMutationBatch {
                    mutations,
                    index_jobs,
                })
                .await
        };
        match applied {
            Ok(results) => json_effect(
                StatusCode::OK,
                rust_native_attested(json!({
                    "contract_version": 1,
                    "atomic": true,
                    "results": results.into_iter().map(|result| json!({
                        "event": {
                            "namespace": result.event.namespace,
                            "seq": result.event.seq,
                            "event_id": result.event.event_id,
                            "entity_kind": result.event.entity_kind,
                            "entity_id": result.event.entity_id,
                            "op": result.event.op,
                            "payload": result.event.payload,
                        },
                        "inserted": result.inserted,
                        "mutated": result.mutated,
                    })).collect::<Vec<_>>(),
                })),
            ),
            Err(error) => effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_GRAPH_MUTATION_CONFLICT",
                error.to_string(),
            ),
        }
    }
}

fn invalid_graph_request(message: String) -> ApiEffectResponse {
    effect_error(
        StatusCode::UNPROCESSABLE_ENTITY,
        "KOGWISTAR_INVALID_REQUEST",
        message,
    )
}

fn selected_proposal_seeds<'a>(
    records: &'a [GraphRecord],
    selected_ids: Option<&[String]>,
) -> Vec<&'a GraphRecord> {
    let by_id = records
        .iter()
        .map(|record| (record.id.as_str(), record))
        .collect::<BTreeMap<_, _>>();
    match selected_ids {
        Some(ids) => ids
            .iter()
            .filter_map(|id| by_id.get(id.as_str()).copied())
            .collect(),
        None => records.iter().collect(),
    }
}

fn proposal_doc_id(record: &GraphRecord) -> Option<&str> {
    record
        .metadata
        .get("doc_id")
        .and_then(Value::as_str)
        .or_else(|| {
            record
                .metadata
                .get("doc_ids")
                .and_then(Value::as_array)
                .and_then(|ids| ids.first())
                .and_then(Value::as_str)
        })
}

fn proposal_where_matches(
    record: &GraphRecord,
    where_filter: Option<&Value>,
) -> Result<bool, String> {
    let payload = json!({
        "metadata": record.metadata,
        "where": where_filter,
    });
    kogwistar_contracts::metadata_filter_matches_from_str(&payload.to_string())
        .map_err(|error| error.to_string())
}

fn proposal_cosine_distance(query: &[f32], candidate: Option<&[f32]>) -> Option<f64> {
    let candidate = candidate?;
    if query.len() != candidate.len()
        || query
            .iter()
            .chain(candidate)
            .any(|value| !value.is_finite())
    {
        return None;
    }
    let (dot, query_norm, candidate_norm) = query.iter().zip(candidate).fold(
        (0.0, 0.0, 0.0),
        |(dot, query_norm, candidate_norm), (left, right)| {
            let left = f64::from(*left);
            let right = f64::from(*right);
            (
                dot + left * right,
                query_norm + left * left,
                candidate_norm + right * right,
            )
        },
    );
    if query_norm == 0.0 || candidate_norm == 0.0 {
        Some(2.0)
    } else {
        Some(1.0 - dot / (query_norm.sqrt() * candidate_norm.sqrt()))
    }
}

impl ApplicationService for InMemoryGraphApplicationService {
    fn execute_inline_native_work_once(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<bool, String>> + Send + '_>> {
        Box::pin(self.execute_native_step_once())
    }

    fn graph_authority_capability(&self) -> GraphAuthorityCapability {
        self.store.graph_authority_capability()
    }

    fn supports_mcp_tool(&self, surface: &str, name: &str) -> bool {
        surface == "root"
            && Self::supports_graph_tool(name)
            && (name != "kg_semantic_seed_then_expand_text" || self.embedding_provider.is_some())
    }

    fn supports_mcp_tool_for_principal(
        &self,
        surface: &str,
        name: &str,
        principal: &Value,
    ) -> bool {
        if !self.supports_mcp_tool(surface, name)
            || !matches!(principal_role(principal).as_str(), "ro" | "rw")
            || !principal_strings(principal, "ns")
                .iter()
                .any(|namespace| namespace == "docs" || namespace == "*")
        {
            return false;
        }
        let state = self
            .capabilities
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        effective_capabilities(principal, &state).contains("read_graph")
    }

    fn supports_http_operation(&self, method: &str, path: &str) -> bool {
        let route = path.split('?').next();
        if method == "GET" && route == Some("/api/search_index_hybrid") {
            return self.embedding_provider.is_some() && self.search_index_fts.is_some();
        }
        if method == "POST" && route == Some("/api/add_index_entries") {
            return self.embedding_provider.is_some()
                && self.search_index_fts.is_some()
                && !PENDING_SERVER_CUTOVER_ROUTES.contains(&(method, "/api/add_index_entries"));
        }
        method == "POST"
            && (route != Some(GRAPH_SEMANTIC_SEED_EXPAND_ROUTE)
                || self.embedding_provider.is_some())
            && matches!(
                route,
                Some(INTERNAL_GRAPH_MUTATIONS_ROUTE)
                    | Some(INTERNAL_GRAPH_LOAD_PERSISTED_ROUTE)
                    | Some(INTERNAL_GRAPH_NEIGHBORS_ROUTE)
                    | Some(INTERNAL_GRAPH_PROPOSE_BRUTE_FORCE_ROUTE)
                    | Some(INTERNAL_GRAPH_VECTOR_QUERY_ROUTE)
                    | Some(GRAPH_SEMANTIC_SEED_EXPAND_ROUTE)
                    | Some(INTERNAL_GRAPH_PROPOSE_VECTOR_ROUTE)
                    | Some(INTERNAL_GRAPH_FIND_EDGES_ROUTE)
                    | Some(INTERNAL_GRAPH_SHORTEST_PATH_ROUTE)
                    | Some(INTERNAL_GRAPH_K_HOP_ROUTE)
                    | Some(INTERNAL_GRAPH_VIZ_D3_ROUTE)
                    | Some(INTERNAL_GRAPH_VIZ_CYTOSCAPE_ROUTE)
            )
    }

    fn execute(
        &self,
        request: ApiEffectRequest,
    ) -> Pin<Box<dyn Future<Output = ApiEffectResponse> + Send + '_>> {
        Box::pin(async move {
            if request.method == "GET"
                && request.path_and_query.split('?').next() == Some("/api/search_index_hybrid")
                && self.supports_http_operation("GET", "/api/search_index_hybrid")
            {
                return self.search_index_hybrid_effect(&request).await;
            }
            if request.method == "POST"
                && request.path_and_query.split('?').next() == Some("/api/add_index_entries")
                && self.supports_http_operation("POST", "/api/add_index_entries")
            {
                let (Some(fts), Some(provider)) = (
                    self.search_index_fts.as_deref(),
                    self.embedding_provider.as_deref(),
                ) else {
                    return effect_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "KOGWISTAR_SEARCH_INDEX_UNAVAILABLE",
                        "Search-index event, FTS, vector, or embedding services are not configured",
                    );
                };
                return crate::search_index_fts::search_index_upsert_effect(
                    &request,
                    &self.capabilities,
                    &self.store,
                    &self.store,
                    "nodes_index",
                    Some(fts),
                    Some(provider),
                )
                .await;
            }
            if request.method != "POST" {
                return unavailable_effect(request);
            }
            if request.path_and_query == INTERNAL_GRAPH_MUTATIONS_ROUTE {
                self.graph_mutation_effect(&request).await
            } else if matches!(
                request.path_and_query.as_str(),
                INTERNAL_GRAPH_LOAD_PERSISTED_ROUTE
                    | INTERNAL_GRAPH_NEIGHBORS_ROUTE
                    | INTERNAL_GRAPH_PROPOSE_BRUTE_FORCE_ROUTE
                    | INTERNAL_GRAPH_VECTOR_QUERY_ROUTE
                    | GRAPH_SEMANTIC_SEED_EXPAND_ROUTE
                    | INTERNAL_GRAPH_PROPOSE_VECTOR_ROUTE
                    | INTERNAL_GRAPH_FIND_EDGES_ROUTE
                    | INTERNAL_GRAPH_SHORTEST_PATH_ROUTE
                    | INTERNAL_GRAPH_K_HOP_ROUTE
                    | INTERNAL_GRAPH_VIZ_D3_ROUTE
                    | INTERNAL_GRAPH_VIZ_CYTOSCAPE_ROUTE
            ) {
                self.graph_read_effect(&request).await
            } else {
                unavailable_effect(request)
            }
        })
    }
}

impl InMemoryGraphApplicationService {
    // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/in_memory_service.rs#search_index_hybrid_effect; python=kogwistar/server_mcp_with_admin.py#search_index_hybrid; fixture=contracts/golden/search-index-hybrid-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/part_08.rs#in_memory_search_index_hybrid_is_scoped_and_acl_filtered; python_test=tests/core/test_search_index_contract.py#test_hybrid_search_fusion_matches_shared_rust_golden_fixture
    async fn search_index_hybrid_effect(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        if !capability_allowed(request, &self.capabilities, "read_graph", &["read_graph"]) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Search-index query requires read_graph capability",
            );
        }
        let (query, limit, resolve_node) =
            match crate::search_index_fts::parse_hybrid_search_query(&request.path_and_query) {
                Ok(parsed) => parsed,
                Err(error) => {
                    return effect_error(
                        StatusCode::UNPROCESSABLE_ENTITY,
                        "KOGWISTAR_INVALID_REQUEST",
                        error,
                    );
                }
            };
        let (Some(provider), Some(fts)) = (
            self.embedding_provider.as_ref(),
            self.search_index_fts.as_ref(),
        ) else {
            return effect_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "KOGWISTAR_SEARCH_INDEX_UNAVAILABLE",
                "Search-index embedding or FTS storage is not configured",
            );
        };
        let hits = match crate::search_index_fts::hybrid_search(
            fts,
            &self.store,
            "nodes_index",
            provider.as_ref(),
            &query,
            limit,
        )
        .await
        {
            Ok(hits) => hits,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_SEARCH_INDEX_ERROR",
                    error,
                );
            }
        };
        let scope = match self.authorized_graph_scope(request) {
            Ok(scope) => scope,
            Err(response) => return response,
        };
        let (nodes, _) = match self.graph_snapshot(scope).await {
            Ok(records) => records,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error,
                );
            }
        };
        let scope_value = security_scope_value(&request.principal);
        let principal = graph_acl_principal_from_claims(
            &request.principal,
            scope_value["security_scope"].as_str(),
        );
        let (visible_nodes, _) = graph_records_visible_by_acl(&nodes, &[], "knowledge", &principal);
        let candidate_ids = hits
            .iter()
            .map(|hit| hit.node_id.as_str())
            .collect::<BTreeSet<_>>();
        let canonical_nodes = visible_nodes
            .into_iter()
            .filter(|record| candidate_ids.contains(record.id.as_str()))
            .collect::<Vec<_>>();
        json_effect(
            StatusCode::OK,
            crate::search_index_fts::project_hybrid_search_result(
                &query,
                &hits,
                &canonical_nodes,
                resolve_node,
                limit,
            ),
        )
    }
}
