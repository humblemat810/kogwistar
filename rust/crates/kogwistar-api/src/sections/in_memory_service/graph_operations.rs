use super::*;
use serde_json::json;

impl InMemoryGraphApplicationService {
    pub(crate) fn supports_graph_tool(name: &str) -> bool {
        matches!(
            name,
            "kg_find_edges"
                | "kg_k_hop"
                | "kg_load_persisted"
                | "kg_neighbors"
                | "kg_propose_bruteforce"
                | "kg_shortest_path"
                | "kg_semantic_seed_then_expand_text"
                | "wisdom.semantic_seed_then_expand"
                | "kg_viz_cytoscape_json"
                | "kg_viz_d3_json"
                | "propose_vector"
        )
    }

    pub(crate) fn with_embedding_provider(mut self, provider: Arc<dyn EmbeddingProvider>) -> Self {
        self.embedding_provider = Some(provider);
        self
    }

    pub(crate) fn with_chat_completion_provider(
        mut self,
        provider: crate::native_answer_generation::OpenAiCompatibleChatConfig,
    ) -> Self {
        self.chat_completion_provider = Some(provider);
        self
    }

    pub(crate) async fn with_owner_auth_config(
        mut self,
        config: &AuthConfig,
    ) -> Result<Self, String> {
        self.owner_auth_store = Some(AuthStore::from_config(config).await?);
        Ok(self)
    }

    pub(crate) fn with_search_index_path(
        mut self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<Self, String> {
        let connection = crate::search_index_fts::open_index(path)
            .map_err(|error| format!("could not open search-index FTS database: {error}"))?;
        self.search_index_fts = Some(Arc::new(Mutex::new(connection)));
        Ok(self)
    }

    pub(crate) fn authorized_graph_scope(
        &self,
        request: &ApiEffectRequest,
    ) -> Result<GraphScope, ApiEffectResponse> {
        self.authorized_graph_scope_for_namespace(request, "docs")
    }

    pub(crate) fn authorized_graph_scope_for_namespace(
        &self,
        request: &ApiEffectRequest,
        required_namespace: &str,
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
            .any(|namespace| namespace == "*" || namespace == required_namespace)
        {
            return Err(effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_NAMESPACE_FORBIDDEN",
                format!("Graph reads require {required_namespace} namespace"),
            ));
        }
        let scope_value = security_scope_value(&request.principal);
        let scope = GraphScope {
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
        };
        if required_namespace == "wisdom" && scope.namespace != required_namespace {
            return Err(effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_SCOPE_FORBIDDEN",
                format!("Graph reads require storage namespace {required_namespace}"),
            ));
        }
        Ok(scope)
    }

    pub(crate) async fn graph_snapshot(
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

    pub(crate) async fn graph_read_effect(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
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
        if request.path_and_query == WISDOM_SEMANTIC_SEED_EXPAND_ROUTE {
            return self.wisdom_semantic_seed_expand_effect(request).await;
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
        let redirect_rid = if request.path_and_query == INTERNAL_GRAPH_NEIGHBORS_ROUTE {
            match parse_graph_neighbors_request(&request.body) {
                Ok(input) => match input.resolve_mode {
                    GraphNeighborsResolveModeV1::ActiveOnly => Some(input.rid.trim().to_owned()),
                    GraphNeighborsResolveModeV1::Redirect => {
                        if nodes.iter().any(|node| node.id == input.rid.trim()) {
                            resolve_graph_node_redirect_id(input.rid.trim(), &nodes)
                        } else {
                            resolve_graph_edge_redirect_id(input.rid.trim(), &edges)
                        }
                    }
                },
                Err(_) => None,
            }
        } else {
            None
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
                    redirect_rid.as_deref().unwrap_or_default(),
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

    // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/in_memory_service/graph_operations.rs#graph_semantic_seed_expand_effect; python=kogwistar/graph_query.py#GraphQuery.semantic_seed_then_expand_text; fixture=contracts/golden/semantic-seed-expand-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/in_memory_semantic_retrieval.rs#in_memory_semantic_seed_expand_is_available_through_authenticated_http_mcp; python_test=tests/primitives/test_graph_query.py#test_semantic_seed_then_expand_text
    pub(crate) async fn graph_semantic_seed_expand_effect(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        self.semantic_seed_expand_effect(request, "knowledge", "docs", 5, 1)
            .await
    }

    pub(crate) async fn wisdom_semantic_seed_expand_effect(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        self.semantic_seed_expand_effect(request, "wisdom", "wisdom", 10, 2)
            .await
    }

    pub(crate) async fn semantic_seed_expand_effect(
        &self,
        request: &ApiEffectRequest,
        truth_graph: &str,
        required_namespace: &str,
        default_top_k: usize,
        default_hops: usize,
    ) -> ApiEffectResponse {
        let scope = match self.authorized_graph_scope_for_namespace(request, required_namespace) {
            Ok(scope) => scope,
            Err(response) => return response,
        };
        let input: SemanticSeedExpandRequest = match serde_json::from_slice(&request.body) {
            Ok(input) => input,
            Err(error) => return invalid_graph_request(error.to_string()),
        };
        let query_text = input.text.trim();
        let top_k = input.top_k.unwrap_or(default_top_k);
        let hops = input.hops.unwrap_or(default_hops);
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
            graph_records_visible_by_acl(&all_nodes, &all_edges, truth_graph, &principal);
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
    pub(crate) async fn graph_load_persisted_effect(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
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
    pub(crate) async fn graph_propose_bruteforce_effect(
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
        if let Some(where_filter) = input.where_filter.as_ref()
            && let Err(error) = kogwistar_contracts::metadata_filter_matches_from_str(
                &json!({"metadata": {}, "where": where_filter}).to_string(),
            )
        {
            return invalid_graph_request(error.to_string());
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

    // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/in_memory_service/graph_operations.rs#graph_vector_proposal_effect; python=kogwistar/server/mcp_tools.py#propose_vector; fixture=contracts/golden/graph-vector-proposal-acl-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/in_memory_graph_routes.rs#in_memory_backend_routes_authenticated_graph_write_and_mcp_read; python_test=tests/mcp/test_chat_mcp_graph_reads.py#test_propose_vector_returns_only_acl_visible_pairs
    pub(crate) async fn graph_vector_proposal_effect(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
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
        let pairs = match graph_vector_proposal_pairs(
            &all_nodes,
            &all_edges,
            &visible_nodes,
            &visible_edges,
            &input,
            where_filter,
        ) {
            Ok(pairs) => pairs,
            Err(error) => return invalid_graph_request(error),
        };
        json_effect(
            StatusCode::OK,
            rust_native_attested(json!({"pairs": pairs})),
        )
    }

    // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/in_memory_service/graph_operations.rs#graph_vector_query_effect; python=kogwistar/engine_core/subsystems/acl.py#ACLAwareReadSubsystem.query_nodes; fixture=contracts/golden/graph-vector-source-acl-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/vector_query_acl.rs#in_memory_vector_query_requires_exact_source_span_acl_reference; python_test=tests/mcp/test_chat_mcp_graph_reads.py#test_vector_query_requires_exact_source_span_acl_reference
    pub(crate) async fn graph_vector_query_effect(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
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

    // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/in_memory_service/graph_operations.rs#document_tree_upsert_effect; python=kogwistar/server_mcp_with_admin.py#document_upsert_tree; fixture=contracts/golden/graph-extraction-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/in_memory_graph_routes.rs#in_memory_document_tree_upsert_matches_graph_extraction_contract; python_test=tests/server/test_chat_server_api_documents.py#test_document_upsert_tree_accepts_nonempty_shared_graph_extraction
    pub(crate) async fn document_tree_upsert_effect(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        if principal_role(&request.principal) != "rw" {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Graph upsert requires rw role",
            );
        }
        if authenticated_run_owner_id(&request.principal).is_none() {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_GRAPH_OWNER_REQUIRED",
                "Graph upsert requires an authenticated owner identity",
            );
        }
        if !principal_strings(&request.principal, "ns")
            .iter()
            .any(|namespace| matches!(namespace.as_str(), "*" | "docs"))
        {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_NAMESPACE_FORBIDDEN",
                "Graph upsert requires docs namespace",
            );
        }
        let (graph_request, inserted_nodes, inserted_edges) =
            match document_tree_upsert_graph_request(request) {
                Ok(parsed) => parsed,
                Err(response) => return response,
            };
        let input = match graph_upsert_request(&graph_request) {
            Ok(input) => input,
            Err(response) => return response,
        };
        if inserted_nodes == 0 && inserted_edges == 0 {
            return json_effect(
                StatusCode::OK,
                json!({
                    "status": "ok",
                    "inserted_nodes": 0,
                    "inserted_edges": 0,
                    "engine_result": {
                        "document_id": input.doc_id,
                        "node_ids": [],
                        "edge_ids": [],
                        "nodes_added": 0,
                        "edges_added": 0
                    }
                }),
            );
        }

        let security_scope = security_scope_value(&request.principal);
        let scope = GraphScope {
            namespace: security_scope["storage_namespace"]
                .as_str()
                .unwrap_or("docs")
                .to_owned(),
            workspace_id: security_scope["workspace"].as_str().map(str::to_owned),
            graph_space: security_scope["graph_space"].as_str().map(str::to_owned),
        };
        let documents = match self
            .store
            .graph_projection_records(GraphProjectionRead {
                scope: scope.clone(),
                table: "gke_documents".to_owned(),
                ids: Some(vec![input.doc_id.clone()]),
                metadata: MetadataFilter::default(),
                limit: 1,
            })
            .await
        {
            Ok(records) => records,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_DOCUMENT_READ_FAILED",
                    error.to_string(),
                );
            }
        };
        let Some(source_text) = documents.first().and_then(|record| record.document.clone()) else {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_SOURCE_DOCUMENT_REQUIRED",
                "Graph upsert requires an existing source document in the request scope",
            );
        };

        let external = match graph_upsert_external_endpoints(&input) {
            Ok(endpoints) => endpoints,
            Err(error) => return invalid_graph_request(error),
        };
        let mut external_nodes = external.node_ids;
        let mut pending_edges = external.edge_ids;
        let mut external_edges = BTreeMap::<String, GraphRecord>::new();
        while !pending_edges.is_empty() {
            let pending = pending_edges
                .iter()
                .filter(|id| !external_edges.contains_key(*id))
                .cloned()
                .collect::<Vec<_>>();
            pending_edges.clear();
            if pending.is_empty() {
                break;
            }
            let records = match self
                .store
                .graph_projection_records(GraphProjectionRead {
                    scope: scope.clone(),
                    table: "gke_edges".to_owned(),
                    ids: Some(pending.clone()),
                    metadata: MetadataFilter::default(),
                    limit: pending.len(),
                })
                .await
            {
                Ok(records) => records,
                Err(error) => {
                    return effect_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "KOGWISTAR_GRAPH_ENDPOINT_READ_FAILED",
                        error.to_string(),
                    );
                }
            };
            if records.len() != pending.len() {
                return invalid_graph_request(
                    "one or more external hyperedge endpoints are missing from the request scope"
                        .to_owned(),
                );
            }
            for record in records {
                external_nodes.extend(
                    ["source_ids", "target_ids"]
                        .into_iter()
                        .flat_map(|field| graph_record_string_ids(&record, field)),
                );
                pending_edges.extend(
                    ["source_edge_ids", "target_edge_ids"]
                        .into_iter()
                        .flat_map(|field| graph_record_string_ids(&record, field)),
                );
                external_edges.insert(record.id.clone(), record);
            }
            if external_nodes.len().saturating_add(external_edges.len()) > 10_000 {
                return invalid_graph_request(
                    "graph endpoint closure exceeds 10000 entities".to_owned(),
                );
            }
        }
        let external_node_records = if external_nodes.is_empty() {
            Vec::new()
        } else {
            match self
                .store
                .graph_projection_records(GraphProjectionRead {
                    scope: scope.clone(),
                    table: "gke_nodes".to_owned(),
                    ids: Some(external_nodes.iter().cloned().collect()),
                    metadata: MetadataFilter::default(),
                    limit: external_nodes.len(),
                })
                .await
            {
                Ok(records) => records,
                Err(error) => {
                    return effect_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "KOGWISTAR_GRAPH_ENDPOINT_READ_FAILED",
                        error.to_string(),
                    );
                }
            }
        };
        if external_node_records.len() != external_nodes.len() {
            return invalid_graph_request(
                "one or more external node endpoints are missing from the request scope".to_owned(),
            );
        }
        let endpoint_records = external_node_records
            .iter()
            .chain(external_edges.values())
            .cloned()
            .collect::<Vec<_>>();
        if endpoint_records.iter().any(wisdom_record_is_tombstoned) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_GRAPH_ENDPOINT_FORBIDDEN",
                "Graph upsert endpoint is tombstoned",
            );
        }
        let external = GraphUpsertExternalEndpoints {
            node_ids: external_nodes,
            edge_ids: external_edges.keys().cloned().collect(),
        };
        let (batch_input, node_ids, edge_ids) = match graph_upsert_mutation_batch(
            &input,
            &source_text,
            &security_scope,
            &external,
            "knowledge",
        ) {
            Ok(batch) => batch,
            Err(error) => return invalid_graph_request(error),
        };
        let mutations = match graph_mutation_batch_value(batch_input.clone()) {
            Ok(mutations) => mutations,
            Err(error) => return invalid_graph_request(error),
        };
        let targets = acl_targets_for_graph_mutations(&mutations).0;
        let mut existing_acl = Vec::new();
        for target_id in acl_target_entity_ids(&targets) {
            let mut filter = MetadataFilter::default();
            filter
                .equals
                .insert("entity_type".to_owned(), json!("acl_record"));
            filter
                .equals
                .insert("acl_truth_graph".to_owned(), json!("knowledge"));
            filter
                .equals
                .insert("acl_target_entity_id".to_owned(), json!(target_id));
            match self
                .store
                .graph_projection_records(GraphProjectionRead {
                    scope: scope.clone(),
                    table: "gke_nodes".to_owned(),
                    ids: None,
                    metadata: filter,
                    limit: 10_000,
                })
                .await
            {
                Ok(records) => existing_acl.extend(records),
                Err(error) => {
                    return effect_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "KOGWISTAR_ACL_READ_FAILED",
                        error.to_string(),
                    );
                }
            }
        }
        let effect_id = uuid::Uuid::new_v4().to_string();
        let (acl_mutations, acl_jobs) = match derive_default_acl_graph_effects_from_records(
            &scope,
            &mutations,
            targets,
            existing_acl,
            AclDefaultsContext {
                truth_graph: "knowledge",
                owner_id: &graph_acl_principal_id(&request.principal),
                security_scope: security_scope["security_scope"].as_str(),
                effect_id: &effect_id,
            },
        ) {
            Ok(result) => result,
            Err(error) => return invalid_graph_request(error),
        };
        let mut all_mutations = mutations.clone();
        all_mutations.extend(acl_mutations);
        let mut index_jobs = match graph_index_jobs_value(&batch_input, &mutations) {
            Ok(jobs) => jobs,
            Err(error) => return invalid_graph_request(error),
        };
        index_jobs.extend(acl_jobs);
        match self
            .store
            .apply_graph_mutations_with_index_jobs_atomic(GraphMutationBatch {
                mutations: all_mutations,
                index_jobs,
            })
            .await
        {
            Ok(_) => json_effect(
                StatusCode::OK,
                json!({
                    "status": "ok",
                    "inserted_nodes": inserted_nodes,
                    "inserted_edges": inserted_edges,
                    "engine_result": {
                        "document_id": input.doc_id,
                        "node_ids": node_ids,
                        "edge_ids": edge_ids,
                        "nodes_added": node_ids.len(),
                        "edges_added": edge_ids.len()
                    }
                }),
            ),
            Err(error) => effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_GRAPH_MUTATION_CONFLICT",
                error.to_string(),
            ),
        }
    }

    pub(crate) async fn graph_mutation_effect(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
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

pub(crate) fn invalid_graph_request(message: String) -> ApiEffectResponse {
    effect_error(
        StatusCode::UNPROCESSABLE_ENTITY,
        "KOGWISTAR_INVALID_REQUEST",
        message,
    )
}

pub(crate) fn selected_proposal_seeds<'a>(
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

pub(crate) fn proposal_doc_id(record: &GraphRecord) -> Option<&str> {
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

pub(crate) fn proposal_where_matches(
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

pub(crate) fn proposal_cosine_distance(query: &[f32], candidate: Option<&[f32]>) -> Option<f64> {
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

// Shared by in-memory and persistent graph services so ranking, filters, and
// ACL-safe result projection stay one Python-parity implementation.
pub(crate) fn graph_vector_proposal_pairs(
    all_nodes: &[GraphRecord],
    all_edges: &[GraphRecord],
    visible_nodes: &[GraphRecord],
    visible_edges: &[GraphRecord],
    input: &GraphVectorProposalRequest,
    where_filter: Option<&Value>,
) -> Result<Vec<Value>, String> {
    let visible_node_ids = visible_nodes
        .iter()
        .map(|record| record.id.as_str())
        .collect::<BTreeSet<_>>();
    let visible_edge_ids = visible_edges
        .iter()
        .map(|record| record.id.as_str())
        .collect::<BTreeSet<_>>();
    let node_seeds = selected_proposal_seeds(all_nodes, input.new_node_ids.as_deref());
    let edge_seeds = selected_proposal_seeds(all_edges, input.new_edge_ids.as_deref());
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
        for (candidates, candidate_kind) in [(all_nodes, "node"), (all_edges, "edge")] {
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
                if !proposal_where_matches(candidate, where_filter)? {
                    continue;
                }
                let candidate_doc = proposal_doc_id(candidate);
                if input.allowed_docs.as_ref().is_some_and(|docs| {
                    !docs.is_empty()
                        && candidate_doc.is_none_or(|doc| !docs.iter().any(|item| item == doc))
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
            let visible_ids = if candidate_kind == "node" {
                &visible_node_ids
            } else {
                &visible_edge_ids
            };
            for (candidate, _) in ranked.into_iter().take(input.top_k) {
                if !visible_ids.contains(candidate.id.as_str()) {
                    continue;
                }
                let (left, right, left_kind, right_kind) = if seed.id <= candidate.id {
                    (&seed.id, &candidate.id, seed_kind, candidate_kind)
                } else {
                    (&candidate.id, &seed.id, candidate_kind, seed_kind)
                };
                if seen.insert((left.clone(), right.clone())) {
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
    Ok(pairs)
}
