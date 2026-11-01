// PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/in_memory_service.rs#InMemoryGraphApplicationService; python=kogwistar/engine_core/in_memory_backend.py#build_in_memory_backend; rust_test=rust/crates/kogwistar-api/src/tests/part_08.rs#in_memory_backend_routes_authenticated_graph_write_and_mcp_read
#[derive(Clone)]
struct InMemoryGraphApplicationService {
    store: InMemoryStore,
    capabilities: SharedCapabilityState,
}

impl Default for InMemoryGraphApplicationService {
    fn default() -> Self {
        Self {
            store: InMemoryStore::default(),
            capabilities: Arc::new(Mutex::new(CapabilityState::default())),
        }
    }
}

impl InMemoryGraphApplicationService {
    fn supports_graph_tool(name: &str) -> bool {
        matches!(
            name,
            "kg_find_edges" | "kg_k_hop" | "kg_neighbors" | "kg_shortest_path"
        )
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
        let read = |table: &str| GraphProjectionRead {
            scope: scope.clone(),
            table: table.to_owned(),
            ids: None,
            metadata: MetadataFilter::default(),
            limit: 10_001,
        };
        let nodes = self
            .store
            .graph_projection_records(read("gke_nodes"))
            .await
            .map_err(|error| error.to_string())?;
        let edges = self
            .store
            .graph_projection_records(read("gke_edges"))
            .await
            .map_err(|error| error.to_string())?;
        if nodes.len() > 10_000 || edges.len() > 10_000 {
            return Err("in-memory graph snapshot exceeds the 10000-record bound".to_owned());
        }
        Ok((nodes, edges))
    }

    async fn graph_read_effect(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        let scope = match self.authorized_graph_scope(request) {
            Ok(scope) => scope,
            Err(response) => return response,
        };
        let truth_graph = if request.path_and_query == INTERNAL_GRAPH_NEIGHBORS_ROUTE {
            match parse_graph_neighbors_request(&request.body) {
                Ok(input) => input.truth_graph,
                Err(error) => return invalid_graph_request(error),
            }
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
            _ => return unavailable_effect(request.clone()),
        };
        json_effect(StatusCode::OK, rust_native_attested(result))
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

impl ApplicationService for InMemoryGraphApplicationService {
    fn graph_authority_capability(&self) -> GraphAuthorityCapability {
        self.store.graph_authority_capability()
    }

    fn supports_mcp_tool(&self, surface: &str, name: &str) -> bool {
        surface == "root" && Self::supports_graph_tool(name)
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
        method == "POST"
            && matches!(
                path.split('?').next(),
                Some(INTERNAL_GRAPH_MUTATIONS_ROUTE)
                    | Some(INTERNAL_GRAPH_NEIGHBORS_ROUTE)
                    | Some(INTERNAL_GRAPH_FIND_EDGES_ROUTE)
                    | Some(INTERNAL_GRAPH_SHORTEST_PATH_ROUTE)
                    | Some(INTERNAL_GRAPH_K_HOP_ROUTE)
            )
    }

    fn execute(
        &self,
        request: ApiEffectRequest,
    ) -> Pin<Box<dyn Future<Output = ApiEffectResponse> + Send + '_>> {
        Box::pin(async move {
            if request.method != "POST" {
                return unavailable_effect(request);
            }
            if request.path_and_query == INTERNAL_GRAPH_MUTATIONS_ROUTE {
                self.graph_mutation_effect(&request).await
            } else if matches!(
                request.path_and_query.as_str(),
                INTERNAL_GRAPH_NEIGHBORS_ROUTE
                    | INTERNAL_GRAPH_FIND_EDGES_ROUTE
                    | INTERNAL_GRAPH_SHORTEST_PATH_ROUTE
                    | INTERNAL_GRAPH_K_HOP_ROUTE
            ) {
                self.graph_read_effect(&request).await
            } else {
                unavailable_effect(request)
            }
        })
    }
}
