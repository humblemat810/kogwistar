// SQLite-authoritative graph reads over the eventual Chroma projection.
// Python parity: kogwistar/engine_core/subsystems/read.py and kogwistar/graph_query.py.
impl SqliteRunApplicationService {
    async fn chroma_graph_load_persisted_effect(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        if !capability_allowed(request, &self.capabilities, "read_graph", &["read_graph"]) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Persisted graph lookup requires read_graph capability",
            );
        }
        let input = match serde_json::from_slice::<LoadPersistedGraphRequest>(&request.body) {
            Ok(input) => input,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    format!("Invalid persisted graph lookup request: {error}"),
                );
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
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                "doc_ids must contain 1 to 1000 non-empty IDs and insertion_method must be non-empty when provided",
            );
        }
        let graph_snapshot = match self.chroma_graph_snapshot(request, "knowledge").await {
            Ok(snapshot) => snapshot,
            Err(response) => return response,
        };
        let (node_ids, edge_ids) = graph_load_persisted_ids_from_records(
            graph_snapshot.nodes(),
            graph_snapshot.edges(),
            &input,
        );
        json_effect(
            StatusCode::OK,
            rust_native_attested(json!({"node_ids": node_ids, "edge_ids": edge_ids})),
        )
    }

    // PYTHON_PARITY: kogwistar/server/mcp_tools.py::kg_propose_bruteforce
    async fn chroma_graph_propose_bruteforce_effect(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        if !capability_allowed(request, &self.capabilities, "read_graph", &["read_graph"]) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Brute-force graph proposal requires read_graph capability",
            );
        }
        let input = match serde_json::from_slice::<ProposeBruteForceRequest>(&request.body) {
            Ok(input) => input,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    format!("Invalid brute-force proposal request: {error}"),
                );
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
            || input
                .limit_per_bucket
                .is_some_and(|limit| limit > 5000)
            || input
                .where_filter
                .as_ref()
                .is_some_and(|value| !value.is_object())
        {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                "pair_kind, document IDs, where, or limit_per_bucket are invalid or exceed their bounds",
            );
        }
        if let Some(where_filter) = input.where_filter.as_ref() {
            if let Err(error) = kogwistar_contracts::metadata_filter_matches_from_str(
                &json!({"metadata": {}, "where": where_filter}).to_string(),
            ) {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error.to_string(),
                );
            }
        }
        let graph_snapshot = match self.chroma_graph_snapshot(request, "knowledge").await {
            Ok(snapshot) => snapshot,
            Err(response) => return response,
        };
        let nodes = graph_snapshot.nodes();
        let edges = graph_snapshot.edges();
        let (node_ids_by_doc, edge_ids_by_doc) =
            brute_force_document_indexes_from_records(nodes, edges);
        let mut docs = match input.allowed_docs.as_ref().filter(|docs| !docs.is_empty()) {
            Some(docs) => docs
                .iter()
                .cloned()
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>(),
            None => node_ids_by_doc
                .keys()
                .chain(edge_ids_by_doc.keys())
                .cloned()
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect(),
        };
        if let Some(anchor) = input.anchor_doc_id.as_ref()
            && !docs.contains(anchor)
        {
            docs.insert(0, anchor.clone());
        }
        if docs.len() > 1000 {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                "brute-force proposal is limited to 1000 documents",
            );
        }
        let nodes = nodes
            .iter()
            .cloned()
            .map(|record| (record.id.clone(), record))
            .collect::<BTreeMap<_, _>>();
        let edges = edges
            .iter()
            .cloned()
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
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error,
                );
            }
        };
        json_effect(
            StatusCode::OK,
            rust_native_attested(json!({"pairs": pairs})),
        )
    }

    async fn chroma_graph_neighbors_effect(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        if !capability_allowed(request, &self.capabilities, "read_graph", &["read_graph"]) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Graph read requires read_graph capability",
            );
        }
        let input = match parse_graph_neighbors_request(&request.body) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error.to_string(),
                );
            }
        };
        let graph_snapshot = match self
            .chroma_graph_snapshot(request, input.truth_graph.trim())
            .await
        {
            Ok(records) => records,
            Err(response) => return response,
        };
        let nodes = graph_snapshot.nodes();
        let edges = graph_snapshot.edges();
        let rid = match input.resolve_mode {
            GraphNeighborsResolveModeV1::ActiveOnly => Some(input.rid.trim().to_owned()),
            GraphNeighborsResolveModeV1::Redirect => match self
                .resolve_chroma_redirect_target(
                    request,
                    input.rid.trim(),
                    GraphEntityKind::Node,
                    &graph_snapshot,
                )
                .await
            {
                Ok((true, target)) => target,
                Ok((false, _)) => match self
                    .resolve_chroma_redirect_target(
                        request,
                        input.rid.trim(),
                        GraphEntityKind::Edge,
                        &graph_snapshot,
                    )
                    .await
                {
                    Ok((_, target)) => target,
                    Err(response) => return response,
                },
                Err(response) => return response,
            },
        };
        let neighbors = graph_neighbors_from_records(
            rid.as_deref().unwrap_or_default(),
            nodes,
            edges,
            input.doc_id.as_deref(),
        );
        json_effect(
            StatusCode::OK,
            rust_native_attested(json!({
                "nodes": neighbors.nodes.into_iter().collect::<Vec<_>>(),
                "edges": neighbors.edges.into_iter().collect::<Vec<_>>(),
            })),
        )
    }

    async fn resolve_chroma_redirect_target(
        &self,
        request: &ApiEffectRequest,
        entity_id: &str,
        entity_kind: GraphEntityKind,
        visible_graph: &GraphAclProjection,
    ) -> Result<(bool, Option<String>), ApiEffectResponse> {
        const MAX_REDIRECT_HOPS: usize = 10_000;
        let scope = security_scope_value(&request.principal);
        let namespace = scope["storage_namespace"].as_str().unwrap_or("default");
        let graph = self.chroma_graph.as_ref().ok_or_else(|| {
            effect_error(
                StatusCode::NOT_IMPLEMENTED,
                "KOGWISTAR_GRAPH_AUTHORITY_UNSUPPORTED",
                "The SQLite service has no configured graph projection",
            )
        })?;
        let mut chain = Vec::new();
        let mut visited = std::collections::BTreeSet::new();
        let mut current = entity_id.to_owned();

        for _ in 0..MAX_REDIRECT_HOPS {
            if !visited.insert(current.clone()) {
                return Ok((!chain.is_empty(), None));
            }
            let record = match kogwistar_store::GraphReadStore::graph_record_of_kind(
                graph,
                namespace,
                entity_kind,
                &current,
            )
            .await
            {
                Ok(Some(record)) => record,
                Ok(None) => return Ok((!chain.is_empty(), None)),
                Err(error) => {
                    return Err(effect_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "KOGWISTAR_STORE_ERROR",
                        error.to_string(),
                    ));
                }
            };
            if !chroma_record_matches_principal_scope(&record, &request.principal) {
                return Ok((true, None));
            }
            let redirect_to = record
                .metadata
                .get("redirect_to_id")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|target| !target.is_empty())
                .map(str::to_owned);
            chain.push(record);
            let Some(target) = redirect_to else {
                let resolved = match entity_kind {
                    GraphEntityKind::Node => resolve_graph_node_redirect_id(entity_id, &chain),
                    GraphEntityKind::Edge => resolve_graph_edge_redirect_id(entity_id, &chain),
                };
                let visible = match entity_kind {
                    GraphEntityKind::Node => visible_graph.nodes(),
                    GraphEntityKind::Edge => visible_graph.edges(),
                };
                return Ok((true, resolved.filter(|target| {
                    visible.iter().any(|record| &record.id == target)
                })));
            };
            current = target;
        }
        Ok((!chain.is_empty(), None))
    }

    async fn chroma_graph_find_edges_effect(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        if !capability_allowed(request, &self.capabilities, "read_graph", &["read_graph"]) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Graph read requires read_graph capability",
            );
        }
        let input = match parse_graph_find_edges_request(&request.body) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error,
                );
            }
        };
        let graph_snapshot = match self.chroma_graph_snapshot(request, "knowledge").await {
            Ok(records) => records,
            Err(response) => return response,
        };
        let nodes = graph_snapshot.nodes();
        let edges = graph_snapshot.edges();
        let query = GraphFindEdgesQuery {
            relation: input.relation,
            src_label_contains: input.src_label_contains,
            tgt_label_contains: input.tgt_label_contains,
            doc_id: input.doc_id,
        };
        json_effect(
            StatusCode::OK,
            rust_native_attested(json!({
                "edges": graph_find_edges_from_records(&query, &nodes, &edges),
            })),
        )
    }

    async fn chroma_graph_shortest_path_effect(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        if !capability_allowed(request, &self.capabilities, "read_graph", &["read_graph"]) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Graph read requires read_graph capability",
            );
        }
        let input: GraphShortestPathRequest = match serde_json::from_slice(&request.body) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error.to_string(),
                );
            }
        };
        if input.src_id.trim().is_empty() || input.dst_id.trim().is_empty() || input.max_depth > 128
        {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                "src_id and dst_id are required and max_depth must be <= 128",
            );
        }
        let graph_snapshot = match self.chroma_graph_snapshot(request, "knowledge").await {
            Ok(records) => records,
            Err(response) => return response,
        };
        let nodes = graph_snapshot.nodes();
        let edges = graph_snapshot.edges();
        let query = GraphShortestPathQuery {
            src_id: input.src_id,
            dst_id: input.dst_id,
            doc_id: input.doc_id,
            max_depth: input.max_depth,
        };
        json_effect(
            StatusCode::OK,
            rust_native_attested(json!({
                "path": graph_shortest_path_from_records(&query, &nodes, &edges),
            })),
        )
    }

    async fn chroma_graph_k_hop_effect(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        if !capability_allowed(request, &self.capabilities, "read_graph", &["read_graph"]) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Graph read requires read_graph capability",
            );
        }
        let input: GraphKHopRequest = match serde_json::from_slice(&request.body) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error.to_string(),
                );
            }
        };
        if input.start_ids.is_empty() || input.start_ids.iter().any(|id| id.trim().is_empty()) {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                "start_ids must contain at least one non-empty ID",
            );
        }
        if input.k > 32 {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                "k must be <= 32",
            );
        }
        let graph_snapshot = match self.chroma_graph_snapshot(request, "knowledge").await {
            Ok(records) => records,
            Err(response) => return response,
        };
        let nodes = graph_snapshot.nodes();
        let edges = graph_snapshot.edges();
        let query = GraphKHopQuery {
            start_ids: input.start_ids,
            k: input.k,
            doc_id: input.doc_id,
        };
        let layers = graph_k_hop_from_records(&query, &nodes, &edges)
            .into_iter()
            .map(|layer| {
                json!({
                    "nodes": layer.nodes.into_iter().collect::<Vec<_>>(),
                    "edges": layer.edges.into_iter().collect::<Vec<_>>(),
                })
            })
            .collect::<Vec<_>>();
        json_effect(
            StatusCode::OK,
            rust_native_attested(json!({"layers": layers})),
        )
    }

    async fn chroma_graph_visualization_effect(
        &self,
        request: &ApiEffectRequest,
        cytoscape: bool,
    ) -> ApiEffectResponse {
        if !capability_allowed(request, &self.capabilities, "read_graph", &["read_graph"]) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Graph read requires read_graph capability",
            );
        }
        let input: GraphVisualizationRequest = match serde_json::from_slice(&request.body) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error.to_string(),
                );
            }
        };
        let graph_snapshot = match self
            .chroma_graph_snapshot(request, input.truth_graph())
            .await
        {
            Ok(records) => records,
            Err(response) => return response,
        };
        let nodes = graph_snapshot.nodes();
        let edges = graph_snapshot.edges();
        let result = if cytoscape {
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
        };
        json_effect(StatusCode::OK, rust_native_attested(result))
    }

    // Python reference: GraphQuery.semantic_seed_then_expand_text and
    // server.mcp_tools.kg_semantic_seed_then_expand_text. Keep seed ranking in
    // Chroma and apply host-owned scope/ACL/lifecycle checks before expansion.
    async fn chroma_graph_semantic_seed_expand_effect(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        if !capability_allowed(request, &self.capabilities, "read_graph", &["read_graph"]) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Semantic graph retrieval requires read_graph capability",
            );
        }
        let input: SemanticSeedExpandRequest = match serde_json::from_slice(&request.body) {
            Ok(input) => input,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error.to_string(),
                );
            }
        };
        let top_k = input.top_k.unwrap_or(5);
        let hops = input.hops.unwrap_or(1);
        let query_text = input.text.trim();
        if query_text.is_empty() || query_text.chars().count() > 16_384 {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                "text must contain 1 to 16384 characters, top_k must be 1 to 1000, and hops must be at most 32",
            );
        }
        if !(1..=1000).contains(&top_k) || hops > 32 {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                "text must contain 1 to 16384 characters, top_k must be 1 to 1000, and hops must be at most 32",
            );
        }
        let Some(provider) = self.embedding_provider.as_ref() else {
            return effect_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "KOGWISTAR_EMBEDDING_UNAVAILABLE",
                "Semantic graph retrieval requires a configured embedding provider",
            );
        };
        let (vector, _embedding_dim, metric) =
            match semantic_query_embedding(provider.as_ref(), query_text).await {
                Ok(result) => result,
                Err(response) => return response,
            };
        let scope_value = security_scope_value(&request.principal);
        let namespace = scope_value["storage_namespace"]
            .as_str()
            .unwrap_or("default");
        let mut metadata = MetadataFilter::default();
        if let Some(workspace) = request
            .principal
            .get("workspace_id")
            .or_else(|| request.principal.get("workspace"))
            .and_then(Value::as_str)
        {
            metadata
                .equals
                .insert("workspace_id".to_owned(), json!(workspace));
        }
        if let Some(graph_space) = request.principal.get("graph_space").and_then(Value::as_str) {
            metadata
                .equals
                .insert("graph_space".to_owned(), json!(graph_space));
        }
        let doc_id = input
            .doc_ids
            .as_deref()
            .filter(|value| !value.trim().is_empty());
        if let Some(doc_id) = doc_id {
            metadata
                .equals
                .insert("doc_id".to_owned(), json!(doc_id));
        }
        let Some(graph) = self.chroma_graph.as_ref() else {
            return effect_error(
                StatusCode::NOT_IMPLEMENTED,
                "KOGWISTAR_GRAPH_AUTHORITY_UNSUPPORTED",
                "The SQLite service has no configured graph projection",
            );
        };
        let matched = match graph
            .vector_query_of_kind_excluding_metadata(
                namespace,
                GraphEntityKind::Node,
                &VectorQuery {
                    embedding: vector,
                    limit: top_k,
                    metadata,
                    metric,
                },
                &[
                    ("entity_type".to_owned(), json!("acl_record")),
                    ("lifecycle_status".to_owned(), json!("tombstoned")),
                    ("lifecycle_status".to_owned(), json!("deleted")),
                ],
            )
            .await
        {
            Ok(matches) => matches,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_VECTOR_QUERY_ERROR",
                    error.to_string(),
                );
            }
        };
        let snapshot = match self.chroma_graph_snapshot(request, &self.graph_role).await {
            Ok(snapshot) => snapshot,
            Err(response) => return response,
        };
        let visible_node_ids = snapshot
            .nodes()
            .iter()
            .map(|record| record.id.as_str())
            .collect::<BTreeSet<_>>();
        let visible_seeds = matched
            .into_iter()
            .map(|hit| hit.record)
            .filter(|record| visible_node_ids.contains(record.id.as_str()))
            .collect::<Vec<_>>();
        let seed_ids = visible_seeds
            .iter()
            .map(|record| record.id.clone())
            .collect::<Vec<_>>();
        let seed_documents = visible_seeds
            .iter()
            .filter_map(graph_record_document_json)
            .collect::<Vec<_>>();
        let layers = graph_k_hop_from_records(
            &GraphKHopQuery {
                start_ids: seed_ids,
                k: hops,
                doc_id: None,
            },
            snapshot.nodes(),
            snapshot.edges(),
        )
        .into_iter()
        .map(|layer| {
            let node_documents = layer
                .nodes
                .iter()
                .filter_map(|id| snapshot.nodes().iter().find(|record| &record.id == id))
                .filter_map(graph_record_document_json)
                .collect::<BTreeSet<_>>();
            let edge_documents = layer
                .edges
                .iter()
                .filter_map(|id| snapshot.edges().iter().find(|record| &record.id == id))
                .filter_map(graph_record_document_json)
                .collect::<BTreeSet<_>>();
            json!({
                "nodes": node_documents.into_iter().collect::<Vec<_>>(),
                "edges": edge_documents.into_iter().collect::<Vec<_>>(),
            })
        })
        .collect::<Vec<_>>();
        json_effect(
            StatusCode::OK,
            rust_native_attested(json!({"seeds": seed_documents, "layers": layers})),
        )
    }

    async fn chroma_graph_vector_query_effect(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        if !capability_allowed(request, &self.capabilities, "read_graph", &["read_graph"]) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Graph vector query requires read_graph capability",
            );
        }
        let input = match parse_graph_vector_query_request(&request.body) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error,
                );
            }
        };

        let kind = if input.entity_kind == "node" {
            GraphEntityKind::Node
        } else {
            GraphEntityKind::Edge
        };
        let scope_value = security_scope_value(&request.principal);
        let namespace = scope_value["storage_namespace"]
            .as_str()
            .unwrap_or("default");
        let mut metadata = input.where_filter.clone();
        if let Some(workspace) = request
            .principal
            .get("workspace_id")
            .or_else(|| request.principal.get("workspace"))
            .and_then(Value::as_str)
        {
            metadata.equals.insert(
                "workspace_id".to_owned(),
                Value::String(workspace.to_owned()),
            );
        }
        if let Some(graph_space) = request.principal.get("graph_space").and_then(Value::as_str) {
            metadata.equals.insert(
                "graph_space".to_owned(),
                Value::String(graph_space.to_owned()),
            );
        }

        let Some(graph) = &self.chroma_graph else {
            return effect_error(
                StatusCode::NOT_IMPLEMENTED,
                "KOGWISTAR_GRAPH_AUTHORITY_UNSUPPORTED",
                "The SQLite service has no configured graph projection",
            );
        };
        let mut matches_by_query = Vec::with_capacity(input.query_embeddings.len());
        for embedding in &input.query_embeddings {
            match graph
                .vector_query_of_kind_excluding_metadata(
                    namespace,
                    kind,
                    &VectorQuery {
                        embedding: embedding.clone(),
                        limit: input.n_results,
                        metadata: metadata.clone(),
                        metric: input.metric,
                    },
                    &[
                        ("entity_type".to_owned(), json!("acl_record")),
                        ("lifecycle_status".to_owned(), json!("tombstoned")),
                        ("lifecycle_status".to_owned(), json!("deleted")),
                    ],
                )
                .await
            {
                Ok(matches) => matches_by_query.push(matches),
                Err(error) => {
                    return effect_error(
                        StatusCode::UNPROCESSABLE_ENTITY,
                        "KOGWISTAR_VECTOR_QUERY_ERROR",
                        error.to_string(),
                    );
                }
            }
        }
        for matches in &mut matches_by_query {
            matches.retain(|matched| {
                chroma_record_matches_principal_scope(&matched.record, &request.principal)
            });
        }

        let projected_hits = matches_by_query
            .iter()
            .flatten()
            .map(|matched| matched.record.clone())
            .collect::<Vec<_>>();
        let (mut hit_nodes, mut hit_edges) = if kind == GraphEntityKind::Node {
            (projected_hits, Vec::new())
        } else {
            (Vec::new(), projected_hits)
        };
        if let Err(error) =
            self.filter_current_chroma_stage2(namespace, &mut hit_nodes, &mut hit_edges)
        {
            return effect_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "KOGWISTAR_STORE_ERROR",
                error,
            );
        }
        let current_hit_ids = hit_nodes
            .iter()
            .chain(hit_edges.iter())
            .map(|record| record.id.as_str())
            .collect::<BTreeSet<_>>();
        for matches in &mut matches_by_query {
            matches.retain(|matched| current_hit_ids.contains(matched.record.id.as_str()));
        }

        let mut candidate_by_id = BTreeMap::new();
        for matched in matches_by_query.iter().flatten() {
            candidate_by_id
                .entry(matched.record.id.clone())
                .or_insert_with(|| matched.record.clone());
        }
        if candidate_by_id.len() > 10_000 {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_VECTOR_QUERY_ERROR",
                "vector candidate ACL set exceeds the 10000-record limit",
            );
        }
        let candidate_records = candidate_by_id.into_values().collect::<Vec<_>>();
        let (candidate_nodes, candidate_edges) = if kind == GraphEntityKind::Node {
            (candidate_records, Vec::new())
        } else {
            let mut edges = candidate_records;
            let mut known_edge_ids = edges
                .iter()
                .map(|record| record.id.clone())
                .collect::<BTreeSet<_>>();
            let mut frontier = edges
                .iter()
                .flat_map(|record| {
                    graph_record_string_ids(record, "source_edge_ids")
                        .into_iter()
                        .chain(graph_record_string_ids(record, "target_edge_ids"))
                })
                .collect::<BTreeSet<_>>();
            while !frontier.is_empty() {
                let pending = frontier
                    .iter()
                    .filter(|edge_id| !known_edge_ids.contains(*edge_id))
                    .cloned()
                    .collect::<Vec<_>>();
                frontier.clear();
                if known_edge_ids.len().saturating_add(pending.len()) > 10_000 {
                    return effect_error(
                        StatusCode::UNPROCESSABLE_ENTITY,
                        "KOGWISTAR_VECTOR_QUERY_ERROR",
                        "edge ACL dependency closure exceeds 10000 records",
                    );
                }
                let referenced = match graph
                    .graph_records_by_ids_of_kind(namespace, GraphEntityKind::Edge, &pending)
                    .await
                {
                    Ok(records) => records,
                    Err(error) => {
                        return effect_error(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "KOGWISTAR_STORE_ERROR",
                            error.to_string(),
                        );
                    }
                };
                for record in referenced {
                    if !chroma_record_matches_principal_scope(&record, &request.principal) {
                        continue;
                    }
                    if !known_edge_ids.insert(record.id.clone()) {
                        continue;
                    }
                    frontier.extend(graph_record_string_ids(&record, "source_edge_ids"));
                    frontier.extend(graph_record_string_ids(&record, "target_edge_ids"));
                    edges.push(record);
                }
            }
            let node_ids = edges
                .iter()
                .flat_map(|record| {
                    graph_record_string_ids(record, "source_ids")
                        .into_iter()
                        .chain(graph_record_string_ids(record, "target_ids"))
                })
                .collect::<BTreeSet<_>>();
            let node_ids = node_ids.into_iter().collect::<Vec<_>>();
            if node_ids.len() > 10_000 {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_VECTOR_QUERY_ERROR",
                    "edge endpoint ACL set exceeds the 10000-record limit",
                );
            }
            let mut nodes = match graph
                .graph_records_by_ids_of_kind(namespace, GraphEntityKind::Node, &node_ids)
                .await
            {
                Ok(records) => records,
                Err(error) => {
                    return effect_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "KOGWISTAR_STORE_ERROR",
                        error.to_string(),
                    );
                }
            };
            nodes
                .retain(|record| chroma_record_matches_principal_scope(record, &request.principal));
            (nodes, edges)
        };
        // SQLite owns canonical ACL events; the eventual Chroma projection is
        // never authoritative for a visibility decision.
        let acl_entity_ids = candidate_nodes
            .iter()
            .chain(candidate_edges.iter())
            .map(|record| record.id.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        if acl_entity_ids.len() > 10_000 {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_VECTOR_QUERY_ERROR",
                "vector candidate ACL set exceeds the 10000-record limit",
            );
        }
        let acl_scope = kogwistar_store::GraphScope {
            namespace: namespace.to_owned(),
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
        let mut acl_records = match self.store.immediate_transaction(|uow| {
            uow.graph_acl_records_for_entities(&acl_scope, &input.truth_graph, &acl_entity_ids)
        }) {
            Ok(records) => records,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error.to_string(),
                );
            }
        };
        acl_records
            .retain(|record| chroma_record_matches_principal_scope(record, &request.principal));
        let mut acl_index = GraphAclIndex::default();
        acl_index.extend_from_records(&acl_records, &input.truth_graph);
        let principal = graph_acl_principal_from_claims(
            &request.principal,
            scope_value["security_scope"].as_str(),
        );
        let (visible_nodes, visible_edges) = graph_records_visible_by_acl_with_index(
            &candidate_nodes,
            &candidate_edges,
            &acl_index,
            &principal,
        );
        let visible_ids = visible_nodes
            .iter()
            .chain(visible_edges.iter())
            .map(|record| record.id.as_str())
            .collect::<BTreeSet<_>>();

        let mut ids = Vec::with_capacity(matches_by_query.len());
        let mut documents = Vec::with_capacity(matches_by_query.len());
        let mut metadatas = Vec::with_capacity(matches_by_query.len());
        let mut embeddings = Vec::with_capacity(matches_by_query.len());
        let mut distances = Vec::with_capacity(matches_by_query.len());
        for matches in matches_by_query {
            let matches = matches
                .into_iter()
                .filter(|matched| {
                    visible_ids.contains(matched.record.id.as_str())
                        && !matched
                            .record
                            .metadata
                            .get("lifecycle_status")
                            .and_then(Value::as_str)
                            .is_some_and(|status| matches!(status, "tombstoned" | "deleted"))
                })
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

}
