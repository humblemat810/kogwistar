fn graph_record_string_ids(record: &GraphRecord, field: &str) -> Vec<String> {
    let document = record
        .document
        .as_deref()
        .and_then(|document| serde_json::from_str::<Value>(document).ok());
    let field_value = record
        .metadata
        .get(field)
        .or_else(|| document.as_ref().and_then(|document| document.get(field)));
    let decoded_field = field_value.and_then(|value| match value {
        Value::String(encoded) => serde_json::from_str::<Value>(encoded).ok(),
        value => Some(value.clone()),
    });
    decoded_field
        .as_ref()
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .collect()
}

impl PostgresRunApplicationService {
    fn authorize_context_snapshot(
        &self,
        request: &ApiEffectRequest,
    ) -> Result<ContextSnapshotAuthorization, ApiEffectResponse> {
        if !capability_allowed(request, &self.capabilities, "project_view", &["project_view"]) {
            return Err(effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Context snapshot reads require project_view capability",
            ));
        }
        Ok(ContextSnapshotAuthorization {
            principal: request.principal.clone(),
        })
    }

    fn authorize_conversation_transcript(
        &self,
        request: &ApiEffectRequest,
    ) -> Result<ConversationTranscriptAuthorization, ApiEffectResponse> {
        if !capability_allowed(request, &self.capabilities, "read_graph", &["read_graph"]) {
            return Err(effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Conversation transcript reads require read_graph capability",
            ));
        }
        Ok(ConversationTranscriptAuthorization {
            principal: request.principal.clone(),
        })
    }

    async fn authorized_conversation_nodes(
        &self,
        request_principal: &Value,
    ) -> Result<Vec<GraphRecord>, String> {
        let (scope, acl_principal) = conversation_graph_scope_and_acl_principal(request_principal);
        let records = graph_projection_records_all(
            &self.store,
            GraphProjectionRead {
                scope,
                table: "gke_nodes".to_owned(),
                ids: None,
                metadata: MetadataFilter::default(),
                limit: 10_000,
            },
        )
        .await
        .map_err(|error| error.to_string())?;
        let (visible_nodes, _) =
            graph_records_visible_by_acl(&records, &[], "conversation", &acl_principal);
        Ok(visible_nodes)
    }

    async fn authorized_tool_audit_nodes(
        &self,
        request_principal: &Value,
        conversation_id: Option<&str>,
        limit: usize,
    ) -> Result<Vec<GraphRecord>, String> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let (scope, acl_principal) = conversation_graph_scope_and_acl_principal(request_principal);
        let mut candidates = Vec::new();
        for entity_type in ["tool_call", "tool_result"] {
            let mut equals = BTreeMap::new();
            equals.insert("entity_type".to_owned(), json!(entity_type));
            if let Some(conversation_id) = conversation_id.filter(|value| !value.is_empty()) {
                equals.insert("conversation_id".to_owned(), json!(conversation_id));
            }
            candidates.extend(
                graph_projection_records_all(
                    &self.store,
                    GraphProjectionRead {
                        scope: scope.clone(),
                        table: "gke_nodes".to_owned(),
                        ids: None,
                        metadata: MetadataFilter { equals },
                        limit,
                    },
                )
                .await
                .map_err(|error| error.to_string())?,
            );
        }
        let acl_records = graph_projection_records_all(
            &self.store,
            GraphProjectionRead {
                scope,
                table: "gke_nodes".to_owned(),
                ids: None,
                metadata: MetadataFilter {
                    equals: BTreeMap::from([("entity_type".to_owned(), json!("acl_record"))]),
                },
                limit: 10_000,
            },
        )
        .await
        .map_err(|error| error.to_string())?;
        let mut acl_index = GraphAclIndex::default();
        acl_index.extend_from_records(&acl_records, "conversation");
        let (visible_nodes, _) =
            graph_records_visible_by_acl_with_index(&candidates, &[], &acl_index, &acl_principal);
        Ok(visible_nodes)
    }

    async fn tool_audit_effect(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        let namespaces = principal_strings(&request.principal, "ns");
        let namespaces = if namespaces.is_empty() {
            vec!["docs".to_owned()]
        } else {
            namespaces
        };
        if !namespaces
            .iter()
            .any(|namespace| matches!(namespace.as_str(), "*" | "conversation" | "workflow"))
        {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_NAMESPACE_FORBIDDEN",
                "Tool audit reads require conversation or workflow namespace",
            );
        }
        if !capability_allowed(
            request,
            &self.capabilities,
            "project_view",
            &["project_view", "read_graph"],
        ) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Reading tool audit requires project_view capability",
            );
        }
        let (conversation_id, limit) = match tool_audit_query(&request.path_and_query) {
            Ok(query) => query,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error,
                );
            }
        };
        let visible_nodes = match self
            .authorized_tool_audit_nodes(&request.principal, conversation_id.as_deref(), limit)
            .await
        {
            Ok(nodes) => nodes,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error,
                );
            }
        };
        let items = match tool_audit_items_from_visible_records(
            &visible_nodes,
            conversation_id.as_deref(),
            limit,
        ) {
            Ok(items) => items,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_TOOL_AUDIT_INVALID",
                    error.to_string(),
                );
            }
        };
        let response = ToolAuditResponseV1 {
            schema_version: kogwistar_contracts::TOOL_AUDIT_WIRE_VERSION,
            conversation_id,
            items,
        };
        if let Err(error) = response.validate() {
            return effect_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "KOGWISTAR_TOOL_AUDIT_INVALID",
                error.to_string(),
            );
        }
        let mut payload =
            serde_json::to_value(response).expect("validated tool audit response serializes");
        if let Some(object) = payload.as_object_mut() {
            object.remove("schema_version");
            if let Some(items) = object.get_mut("items").and_then(Value::as_array_mut) {
                for item in items {
                    if let Some(item) = item.as_object_mut() {
                        item.remove("schema_version");
                    }
                }
            }
        }
        json_effect(StatusCode::OK, payload)
    }

    async fn conversation_transcript_projection(
        &self,
        request: &ApiEffectRequest,
        input: InternalConversationTranscriptRequest,
        authorization: ConversationTranscriptAuthorization,
    ) -> Result<ConversationTranscriptV1, ApiEffectResponse> {
        if authorization.principal != request.principal {
            return Err(effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Conversation transcript authorization does not match the request principal",
            ));
        }
        if let Err(error) = validate_internal_conversation_transcript_request(&input) {
            return Err(effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                error,
            ));
        }
        let visible_nodes = self
            .authorized_conversation_nodes(&request.principal)
            .await
            .map_err(|error| {
                effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error,
                )
            })?;
        conversation_transcript_from_visible_records(
            &visible_nodes,
            &input.conversation_id,
            input.limit,
        )
        .map_err(|error| {
            effect_error(
                StatusCode::NOT_FOUND,
                "KOGWISTAR_CONVERSATION_NOT_FOUND",
                error.to_string(),
            )
        })
    }

    async fn internal_conversation_transcript_effect(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        let authorization = match self.authorize_conversation_transcript(request) {
            Ok(value) => value,
            Err(response) => return response,
        };
        let input = match parse_internal_conversation_transcript_request(&request.body) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error,
                );
            }
        };
        match self
            .conversation_transcript_projection(request, input, authorization)
            .await
        {
            Ok(transcript) => json_effect(
                StatusCode::OK,
                rust_native_attested(
                    serde_json::to_value(transcript).expect("transcript serializes"),
                ),
            ),
            Err(response) => response,
        }
    }

    async fn context_snapshot_projection(
        &self,
        request: &ApiEffectRequest,
        input: InternalContextSnapshotRequest,
        authorization: ContextSnapshotAuthorization,
    ) -> Result<ContextSnapshotV1, ApiEffectResponse> {
        if authorization.principal != request.principal {
            return Err(effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Context snapshot authorization does not match the request principal",
            ));
        }
        if let Err(error) = validate_internal_context_snapshot_request(&input) {
            return Err(effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                error,
            ));
        }
        let visible_nodes = self
            .authorized_conversation_nodes(&request.principal)
            .await
            .map_err(|error| {
                effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error,
                )
            })?;
        kogwistar_store::context_snapshot_from_visible_records(
            &visible_nodes,
            &input.conversation_id,
            input.run_id.as_deref(),
            input.stage.as_deref(),
        )
        .map_err(|error| {
            effect_error(
                    StatusCode::NOT_FOUND,
                    "KOGWISTAR_CONTEXT_SNAPSHOT_NOT_FOUND",
                    error.to_string(),
                )
        })
    }

    async fn internal_context_snapshot_effect(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        let authorization = match self.authorize_context_snapshot(request) {
            Ok(value) => value,
            Err(response) => return response,
        };
        let input = match parse_internal_context_snapshot_request(&request.body) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error,
                );
            }
        };
        match self
            .context_snapshot_projection(request, input, authorization)
            .await
        {
            Ok(snapshot) => json_effect(
                StatusCode::OK,
                rust_native_attested(
                    serde_json::to_value(snapshot).expect("snapshot serializes"),
                ),
            ),
            Err(response) => response,
        }
    }

    async fn latest_snapshot_effect(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        let namespaces = principal_strings(&request.principal, "ns");
        let namespaces = if namespaces.is_empty() {
            vec!["docs".to_owned()]
        } else {
            namespaces
        };
        if !namespaces
            .iter()
            .any(|namespace| matches!(namespace.as_str(), "*" | "conversation" | "workflow"))
        {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_NAMESPACE_FORBIDDEN",
                "Context snapshot reads require conversation or workflow namespace",
            );
        }
        let Some(conversation_id) = latest_snapshot_conversation_id(&request.path_and_query) else {
            return effect_error(
                StatusCode::NOT_FOUND,
                "KOGWISTAR_ROUTE_NOT_FOUND",
                "Conversation snapshot route was not found",
            );
        };
        let (run_id, stage) = context_snapshot_query(&request.path_and_query);
        let authorization = match self.authorize_context_snapshot(request) {
            Ok(value) => value,
            Err(response) => return response,
        };
        match self
            .context_snapshot_projection(
                request,
                InternalContextSnapshotRequest {
                    conversation_id,
                    run_id,
                    stage,
                },
                authorization,
            )
            .await
        {
            Ok(snapshot) => json_effect(StatusCode::OK, public_context_snapshot_value(snapshot)),
            Err(response) => response,
        }
    }

    async fn conversation_list_effect(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        let namespaces = principal_strings(&request.principal, "ns");
        let namespaces = if namespaces.is_empty() {
            vec!["docs".to_owned()]
        } else {
            namespaces
        };
        if !namespaces
            .iter()
            .any(|namespace| matches!(namespace.as_str(), "*" | "conversation"))
        {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_NAMESPACE_FORBIDDEN",
                "Conversation reads require conversation namespace",
            );
        }
        if !capability_allowed(request, &self.capabilities, "read_graph", &["read_graph"]) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Conversation reads require read_graph capability",
            );
        }
        let user_id = request
            .principal
            .get("sub")
            .or_else(|| request.principal.get("user_id"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let Some(user_id) = user_id else {
            return effect_error(
                StatusCode::UNAUTHORIZED,
                "KOGWISTAR_USER_IDENTITY_REQUIRED",
                "User Identity required",
            );
        };
        let (scope, acl_principal) = conversation_graph_scope_and_acl_principal(&request.principal);
        let mut acl_index = GraphAclIndex::default();
        let mut candidates = Vec::new();
        let read_result = for_each_graph_projection_page(
            &self.store,
            GraphProjectionRead {
                scope,
                table: "gke_nodes".to_owned(),
                ids: None,
                metadata: MetadataFilter::default(),
                limit: 1024,
            },
            |page| {
                acl_index.extend_from_records(page, "conversation");
                candidates.extend(compact_conversation_list_candidates(page));
                Ok(())
            },
        )
        .await;
        if let Err(error) = read_result {
            return effect_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "KOGWISTAR_STORE_ERROR",
                error.to_string(),
            );
        }
        let (visible_nodes, _) =
            graph_records_visible_by_acl_with_index(&candidates, &[], &acl_index, &acl_principal);
        let mut conversations = ConversationListAccumulator::new(user_id);
        conversations.extend_visible_records(&visible_nodes);
        let conversations = conversations
            .finish()
            .into_iter()
            .map(|entry| {
                json!({
                    "id": entry.id,
                    "start_node_id": entry.start_node_id,
                    "status": entry.status,
                    "turn_count": entry.turn_count,
                })
            })
            .collect::<Vec<_>>();
        json_effect(StatusCode::OK, json!({"conversations": conversations}))
    }

    async fn conversation_detail_effect(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        let namespaces = principal_strings(&request.principal, "ns");
        let namespaces = if namespaces.is_empty() {
            vec!["docs".to_owned()]
        } else {
            namespaces
        };
        if !namespaces
            .iter()
            .any(|namespace| matches!(namespace.as_str(), "*" | "conversation"))
        {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_NAMESPACE_FORBIDDEN",
                "Conversation reads require conversation namespace",
            );
        }
        if !capability_allowed(request, &self.capabilities, "read_graph", &["read_graph"]) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Conversation reads require read_graph capability",
            );
        }
        let Some(conversation_id) = conversation_detail_id(&request.path_and_query) else {
            return effect_error(
                StatusCode::NOT_FOUND,
                "KOGWISTAR_ROUTE_NOT_FOUND",
                "Conversation route was not found",
            );
        };
        let visible_nodes = match self.authorized_conversation_nodes(&request.principal).await {
            Ok(nodes) => nodes,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error,
                );
            }
        };
        let summary =
            match conversation_summary_from_visible_records(&visible_nodes, &conversation_id) {
                Ok(summary) => summary,
                Err(error) => {
                    return effect_error(
                        StatusCode::NOT_FOUND,
                        "KOGWISTAR_CONVERSATION_NOT_FOUND",
                        error.to_string(),
                    );
                }
            };
        let mut payload = rust_native_attested(
            serde_json::to_value(summary).expect("conversation summary serializes"),
        );
        if let Some(object) = payload.as_object_mut() {
            object.remove("schema_version");
            object.remove("implementation_authority");
            object.remove("seam_attestation_version");
            object.remove("seam_crossings");
        }
        json_effect(StatusCode::OK, payload)
    }

    async fn conversation_transcript_effect(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        let namespaces = principal_strings(&request.principal, "ns");
        let namespaces = if namespaces.is_empty() {
            vec!["docs".to_owned()]
        } else {
            namespaces
        };
        if !namespaces
            .iter()
            .any(|namespace| matches!(namespace.as_str(), "*" | "conversation"))
        {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_NAMESPACE_FORBIDDEN",
                "Conversation reads require conversation namespace",
            );
        }
        let Some(conversation_id) = conversation_transcript_id(&request.path_and_query) else {
            return effect_error(
                StatusCode::NOT_FOUND,
                "KOGWISTAR_ROUTE_NOT_FOUND",
                "Conversation transcript route was not found",
            );
        };
        let authorization = match self.authorize_conversation_transcript(request) {
            Ok(value) => value,
            Err(response) => return response,
        };
        match self
            .conversation_transcript_projection(
                request,
                InternalConversationTranscriptRequest {
                    conversation_id,
                    limit: 10_000,
                },
                authorization,
            )
            .await
        {
            Ok(transcript) => {
                let mut payload =
                    serde_json::to_value(transcript).expect("transcript serializes");
                if let Some(object) = payload.as_object_mut() {
                    object.remove("schema_version");
                    if let Some(turns) = object.get_mut("turns").and_then(Value::as_array_mut) {
                        for turn in turns {
                            if let Some(turn) = turn.as_object_mut() {
                                turn.remove("schema_version");
                            }
                        }
                    }
                }
                json_effect(StatusCode::OK, payload)
            }
            Err(response) => response,
        }
    }

    async fn graph_neighbors_effect(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
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
        let rid = input.rid.trim();
        let doc_id = input.doc_id.as_deref();
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
        let read = |table: &'static str| GraphProjectionRead {
            scope: scope.clone(),
            table: table.to_owned(),
            ids: None,
            metadata: MetadataFilter::default(),
            limit: 10_000,
        };
        let nodes = match self.store.graph_projection_records(read("gke_nodes")).await {
            Ok(records) => records,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error.to_string(),
                );
            }
        };
        let edges = match self.store.graph_projection_records(read("gke_edges")).await {
            Ok(records) => records,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error.to_string(),
                );
            }
        };
        let principal = graph_acl_principal_from_claims(
            &request.principal,
            scope_value["security_scope"].as_str(),
        );
        let (nodes, edges) =
            graph_records_visible_by_acl(&nodes, &edges, input.truth_graph.trim(), &principal);
        let neighbors = graph_neighbors_from_records(rid, &nodes, &edges, doc_id);
        json_effect(
            StatusCode::OK,
            rust_native_attested(json!({
                "nodes": neighbors.nodes.into_iter().collect::<Vec<_>>(),
                "edges": neighbors.edges.into_iter().collect::<Vec<_>>(),
            })),
        )
    }

    async fn authorized_vector_candidate_ids(
        &self,
        scope: &GraphScope,
        truth_graph: &str,
        entity_kind: &str,
        candidates_by_query: &[Vec<GraphRecord>],
        principal: &GraphAclPrincipal,
    ) -> Result<BTreeSet<String>, String> {
        let mut seen_candidate_ids = BTreeSet::new();
        let candidates = candidates_by_query
            .iter()
            .flatten()
            .filter(|record| seen_candidate_ids.insert(record.id.clone()))
            .cloned()
            .collect::<Vec<_>>();
        let (nodes, edges) = if entity_kind == "node" {
            (candidates, Vec::new())
        } else {
            let mut edges = candidates;
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
                if pending.is_empty() {
                    break;
                }
                if known_edge_ids.len().saturating_add(pending.len()) > 10_000 {
                    return Err("edge ACL dependency closure exceeds 10000 records".to_owned());
                }
                let referenced = graph_projection_records_all(
                    &self.store,
                    GraphProjectionRead {
                        scope: scope.clone(),
                        table: "gke_edges".to_owned(),
                        ids: Some(pending),
                        metadata: MetadataFilter::default(),
                        limit: 10_000,
                    },
                )
                .await
                .map_err(|error| error.to_string())?;
                for record in referenced {
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
            let nodes = if node_ids.is_empty() {
                Vec::new()
            } else {
                graph_projection_records_all(
                    &self.store,
                    GraphProjectionRead {
                        scope: scope.clone(),
                        table: "gke_nodes".to_owned(),
                        ids: Some(node_ids.into_iter().collect()),
                        metadata: MetadataFilter::default(),
                        limit: 10_000,
                    },
                )
                .await
                .map_err(|error| error.to_string())?
            };
            (nodes, edges)
        };
        let entity_ids = nodes
            .iter()
            .chain(edges.iter())
            .map(|record| record.id.clone())
            .collect::<BTreeSet<_>>();
        let acl_records = self
            .store
            .graph_acl_records_for_entities(scope, truth_graph, &entity_ids.into_iter().collect::<Vec<_>>())
            .await
            .map_err(|error| error.to_string())?;
        let mut acl_index = GraphAclIndex::default();
        acl_index.extend_from_records(&acl_records, truth_graph);
        let (visible_nodes, visible_edges) =
            graph_records_visible_by_acl_with_index(&nodes, &edges, &acl_index, principal);
        Ok(visible_nodes
            .iter()
            .chain(visible_edges.iter())
            .map(|record| record.id.clone())
            .collect())
    }

    async fn graph_vector_query_effect(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
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
        let table = if input.entity_kind == "node" {
            "gke_nodes"
        } else {
            "gke_edges"
        };
        let principal = graph_acl_principal_from_claims(
            &request.principal,
            scope_value["security_scope"].as_str(),
        );

        let mut all_ids = Vec::with_capacity(input.query_embeddings.len());
        let mut all_documents = Vec::with_capacity(input.query_embeddings.len());
        let mut all_metadatas = Vec::with_capacity(input.query_embeddings.len());
        let mut all_embeddings = Vec::with_capacity(input.query_embeddings.len());
        let mut all_distances = Vec::with_capacity(input.query_embeddings.len());
        let queries = input
            .query_embeddings
            .iter()
            .map(|embedding| GraphProjectionVectorQuery {
                    scope: scope.clone(),
                    table: table.to_owned(),
                    query: VectorQuery {
                        embedding: embedding.clone(),
                        limit: input.n_results,
                        metadata: input.where_filter.clone(),
                        metric: input.metric,
                    },
                    embedding_dim: embedding.len(),
                })
            .collect();
        let matches_by_query = match self
            .store
            .graph_projection_vector_query_batch(queries)
            .await
        {
            Ok(matches) => matches,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_VECTOR_QUERY_ERROR",
                    error.to_string(),
                );
            }
        };
        let candidate_rows = matches_by_query
            .iter()
            .map(|matches| {
                matches
                    .iter()
                    .map(|matched| matched.record.clone())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let visible_ids = match self
            .authorized_vector_candidate_ids(
                &scope,
                &input.truth_graph,
                &input.entity_kind,
                &candidate_rows,
                &principal,
            )
            .await
        {
            Ok(ids) => ids,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_ACL_QUERY_ERROR",
                    error,
                );
            }
        };
        for matches in matches_by_query {
            let visible_matches = matches
                .into_iter()
                .filter(|matched| {
                    !matched
                        .record
                        .metadata
                        .get("lifecycle_status")
                        .and_then(Value::as_str)
                        .is_some_and(|status| matches!(status, "tombstoned" | "deleted"))
                        && visible_ids.contains(&matched.record.id)
                })
                .collect::<Vec<_>>();
            all_ids.push(
                visible_matches
                    .iter()
                    .map(|matched| json!(matched.record.id))
                    .collect::<Vec<_>>(),
            );
            if input.include.iter().any(|field| field == "documents") {
                all_documents.push(
                    visible_matches
                        .iter()
                        .map(|matched| json!(matched.record.document))
                        .collect::<Vec<_>>(),
                );
            }
            if input.include.iter().any(|field| field == "metadatas") {
                all_metadatas.push(
                    visible_matches
                        .iter()
                        .map(|matched| json!(matched.record.metadata))
                        .collect::<Vec<_>>(),
                );
            }
            if input.include.iter().any(|field| field == "embeddings") {
                all_embeddings.push(
                    visible_matches
                        .iter()
                        .map(|matched| json!(matched.record.embedding))
                        .collect::<Vec<_>>(),
                );
            }
            if input.include.iter().any(|field| field == "distances") {
                all_distances.push(
                    visible_matches
                        .iter()
                        .map(|matched| json!(matched.distance))
                        .collect::<Vec<_>>(),
                );
            }
        }
        let mut response = serde_json::Map::new();
        response.insert("ids".to_owned(), json!(all_ids));
        if input.include.iter().any(|field| field == "documents") {
            response.insert("documents".to_owned(), json!(all_documents));
        }
        if input.include.iter().any(|field| field == "metadatas") {
            response.insert("metadatas".to_owned(), json!(all_metadatas));
        }
        if input.include.iter().any(|field| field == "embeddings") {
            response.insert("embeddings".to_owned(), json!(all_embeddings));
        }
        if input.include.iter().any(|field| field == "distances") {
            response.insert("distances".to_owned(), json!(all_distances));
        }
        json_effect(StatusCode::OK, rust_native_attested(Value::Object(response)))
    }

    async fn graph_find_edges_effect(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
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
        let read = |table: &'static str| GraphProjectionRead {
            scope: scope.clone(),
            table: table.to_owned(),
            ids: None,
            metadata: MetadataFilter::default(),
            limit: 10_000,
        };
        let nodes = match self.store.graph_projection_records(read("gke_nodes")).await {
            Ok(records) => records,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error.to_string(),
                );
            }
        };
        let edges = match self.store.graph_projection_records(read("gke_edges")).await {
            Ok(records) => records,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error.to_string(),
                );
            }
        };
        let principal = graph_acl_principal_from_claims(
            &request.principal,
            scope_value["security_scope"].as_str(),
        );
        let (nodes, edges) = graph_records_visible_by_acl(&nodes, &edges, "knowledge", &principal);
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

    async fn graph_shortest_path_effect(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
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
        let read = |table: &'static str| GraphProjectionRead {
            scope: scope.clone(),
            table: table.to_owned(),
            ids: None,
            metadata: MetadataFilter::default(),
            limit: 10_000,
        };
        let nodes = match self.store.graph_projection_records(read("gke_nodes")).await {
            Ok(records) => records,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error.to_string(),
                );
            }
        };
        let edges = match self.store.graph_projection_records(read("gke_edges")).await {
            Ok(records) => records,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error.to_string(),
                );
            }
        };
        let principal = graph_acl_principal_from_claims(
            &request.principal,
            scope_value["security_scope"].as_str(),
        );
        let (nodes, edges) = graph_records_visible_by_acl(&nodes, &edges, "knowledge", &principal);
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

    async fn graph_k_hop_effect(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
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
        let read = |table: &'static str| GraphProjectionRead {
            scope: scope.clone(),
            table: table.to_owned(),
            ids: None,
            metadata: MetadataFilter::default(),
            limit: 10_000,
        };
        let nodes = match self.store.graph_projection_records(read("gke_nodes")).await {
            Ok(records) => records,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error.to_string(),
                );
            }
        };
        let edges = match self.store.graph_projection_records(read("gke_edges")).await {
            Ok(records) => records,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error.to_string(),
                );
            }
        };
        let principal = graph_acl_principal_from_claims(
            &request.principal,
            scope_value["security_scope"].as_str(),
        );
        let (nodes, edges) = graph_records_visible_by_acl(&nodes, &edges, "knowledge", &principal);
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
            rust_native_attested(json!({
                "layers": layers,
            })),
        )
    }

    async fn syscall_effect(&self, request: &ApiEffectRequest, op: &str) -> ApiEffectResponse {
        let input = match syscall_input(request, op) {
            Ok(input) => input,
            Err(response) => return response,
        };
        let response = match op {
            "spawn_process" => {
                if !syscall_allowed(
                    request,
                    &self.capabilities,
                    op,
                    true,
                    &["spawn_process", "workflow.run.write"],
                ) {
                    syscall_forbidden(op)
                } else {
                    let delegated_args = match prepare_delegated_run_args(
                        &Value::Object(input.args.clone()),
                        &request.principal,
                        &self.capabilities,
                    ) {
                        Ok(value) => value,
                        Err(error) => {
                            return effect_error(
                                StatusCode::UNPROCESSABLE_ENTITY,
                                "KOGWISTAR_INVALID_DELEGATION",
                                error,
                            );
                        }
                    };
                    let nested = ApiEffectRequest {
                        contract_version: request.contract_version,
                        method: "POST".to_owned(),
                        path_and_query: "/api/workflow/runs".to_owned(),
                        body: serde_json::to_vec(&delegated_args)
                            .expect("delegated syscall args serialize"),
                        principal: delegated_child_principal(&request.principal, &delegated_args),
                    };
                    syscall_result(&input.version, op, self.submit_run(&nested).await)
                }
            }
            "terminate_process" => {
                if !syscall_allowed(
                    request,
                    &self.capabilities,
                    op,
                    true,
                    &["workflow.run.write"],
                ) {
                    syscall_forbidden(op)
                } else if let Some(run_id) = input.args["run_id"].as_str() {
                    match self.store.request_server_run_cancel(run_id).await {
                        Ok(()) => match self.store.get_server_run(run_id).await {
                            Ok(Some(run)) => syscall_result(
                                &input.version,
                                op,
                                json_effect(StatusCode::ACCEPTED, server_run_value(run)),
                            ),
                            Ok(None) => effect_error(
                                StatusCode::NOT_FOUND,
                                "KOGWISTAR_RUN_NOT_FOUND",
                                format!("Unknown run_id: {run_id}"),
                            ),
                            Err(error) => effect_error(
                                StatusCode::INTERNAL_SERVER_ERROR,
                                "KOGWISTAR_STORE_ERROR",
                                error.to_string(),
                            ),
                        },
                        Err(error) => effect_error(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "KOGWISTAR_STORE_ERROR",
                            error.to_string(),
                        ),
                    }
                } else {
                    effect_error(
                        StatusCode::BAD_REQUEST,
                        "KOGWISTAR_INVALID_REQUEST",
                        "run_id is required",
                    )
                }
            }
            "checkpoint" => {
                if !syscall_allowed(
                    request,
                    &self.capabilities,
                    op,
                    false,
                    &["workflow.run.read"],
                ) {
                    syscall_forbidden(op)
                } else if let (Some(run_id), Some(step_seq)) = (
                    input.args["run_id"].as_str(),
                    input.args["step_seq"].as_i64(),
                ) {
                    match self.store.list_server_run_events(run_id, 0, 100_000).await {
                        Ok(events) => syscall_result(
                            &input.version,
                            op,
                            runtime_inspection_effect(
                                request,
                                run_id,
                                &["checkpoints", &step_seq.to_string()],
                                events,
                            ),
                        ),
                        Err(error) => effect_error(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "KOGWISTAR_STORE_ERROR",
                            error.to_string(),
                        ),
                    }
                } else {
                    effect_error(
                        StatusCode::BAD_REQUEST,
                        "KOGWISTAR_INVALID_REQUEST",
                        "run_id and integer step_seq are required",
                    )
                }
            }
            "receive_message" => {
                if !syscall_allowed_in_namespace(
                    request,
                    &self.capabilities,
                    op,
                    false,
                    &["read_graph"],
                    "conversation",
                ) {
                    syscall_forbidden(op)
                } else if let Some(conversation_id) = input.args["conversation_id"].as_str() {
                    let authorization = match self.authorize_conversation_transcript(request) {
                        Ok(value) => value,
                        Err(response) => return syscall_result(&input.version, op, response),
                    };
                    match self
                        .conversation_transcript_projection(
                            request,
                            InternalConversationTranscriptRequest {
                                conversation_id: conversation_id.to_owned(),
                                limit: 10_000,
                            },
                            authorization,
                        )
                        .await
                    {
                        Ok(transcript) => syscall_result(
                            &input.version,
                            op,
                            json_effect(
                                StatusCode::OK,
                                legacy_receive_message_value(transcript),
                            ),
                        ),
                        Err(response) => syscall_result(&input.version, op, response),
                    }
                } else {
                    effect_error(
                        StatusCode::BAD_REQUEST,
                        "KOGWISTAR_INVALID_REQUEST",
                        "conversation_id is required",
                    )
                }
            }
            "mount_memory" => {
                if !syscall_allowed_in_namespace(
                    request,
                    &self.capabilities,
                    op,
                    false,
                    &["project_view"],
                    "conversation",
                ) {
                    syscall_forbidden(op)
                } else if let Some(conversation_id) = input.args["conversation_id"].as_str() {
                    let authorization = match self.authorize_context_snapshot(request) {
                        Ok(value) => value,
                        Err(response) => return syscall_result(&input.version, op, response),
                    };
                    let run_id = match context_snapshot_syscall_field(&input.args, "run_id") {
                        Ok(value) => value,
                        Err(error) => {
                            return syscall_result(
                                &input.version,
                                op,
                                effect_error(
                                    StatusCode::UNPROCESSABLE_ENTITY,
                                    "KOGWISTAR_INVALID_REQUEST",
                                    error,
                                ),
                            );
                        }
                    };
                    let stage = match context_snapshot_syscall_field(&input.args, "stage") {
                        Ok(value) => value,
                        Err(error) => {
                            return syscall_result(
                                &input.version,
                                op,
                                effect_error(
                                    StatusCode::UNPROCESSABLE_ENTITY,
                                    "KOGWISTAR_INVALID_REQUEST",
                                    error,
                                ),
                            );
                        }
                    };
                    match self
                        .context_snapshot_projection(
                            request,
                            InternalContextSnapshotRequest {
                                conversation_id: conversation_id.to_owned(),
                                run_id,
                                stage,
                            },
                            authorization,
                        )
                        .await
                    {
                        Ok(snapshot) => syscall_result(
                            &input.version,
                            op,
                            json_effect(StatusCode::OK, public_context_snapshot_value(snapshot)),
                        ),
                        Err(response) => syscall_result(&input.version, op, response),
                    }
                } else {
                    effect_error(
                        StatusCode::BAD_REQUEST,
                        "KOGWISTAR_INVALID_REQUEST",
                        "conversation_id is required",
                    )
                }
            }
            "project_view" => {
                if !syscall_allowed(
                    request,
                    &self.capabilities,
                    op,
                    false,
                    &["read_security_scope", "project_view"],
                ) {
                    syscall_forbidden(op)
                } else {
                    let runs = match self.store.list_server_runs(None, None, None, 10_000).await {
                        Ok(runs) => runs,
                        Err(error) => {
                            return effect_error(
                                StatusCode::INTERNAL_SERVER_ERROR,
                                "KOGWISTAR_STORE_ERROR",
                                error.to_string(),
                            );
                        }
                    };
                    let lanes = match self
                        .store
                        .list_projected_lane_messages(LaneMessageFilter {
                            namespace: Some("workflow".to_owned()),
                            limit: 10_000,
                            ..LaneMessageFilter::default()
                        })
                        .await
                    {
                        Ok(lanes) => lanes,
                        Err(error) => {
                            return effect_error(
                                StatusCode::INTERNAL_SERVER_ERROR,
                                "KOGWISTAR_STORE_ERROR",
                                error.to_string(),
                            );
                        }
                    };
                    let mut events = Vec::new();
                    for run in &runs {
                        match self.store.list_server_run_events(&run.run_id, 0, 200).await {
                            Ok(mut run_events) => events.append(&mut run_events),
                            Err(error) => {
                                return effect_error(
                                    StatusCode::INTERNAL_SERVER_ERROR,
                                    "KOGWISTAR_STORE_ERROR",
                                    error.to_string(),
                                );
                            }
                        }
                    }
                    syscall_result(
                        &input.version,
                        op,
                        operational_effect(request, "/api/workflow/resources", runs, lanes, events),
                    )
                }
            }
            "resume" => {
                if !syscall_allowed(
                    request,
                    &self.capabilities,
                    op,
                    true,
                    &["workflow.run.write"],
                ) {
                    syscall_forbidden(op)
                } else if let Some(run_id) = input.args["run_id"].as_str() {
                    let nested = ApiEffectRequest {
                        contract_version: request.contract_version,
                        method: "POST".to_owned(),
                        path_and_query: format!("/api/workflow/runs/{run_id}/resume"),
                        body: serde_json::to_vec(&input.args).expect("syscall args serialize"),
                        principal: request.principal.clone(),
                    };
                    syscall_result(
                        &input.version,
                        op,
                        self.resume_runtime_run(run_id, &nested).await,
                    )
                } else {
                    effect_error(
                        StatusCode::BAD_REQUEST,
                        "KOGWISTAR_INVALID_REQUEST",
                        "run_id is required",
                    )
                }
            }
            "request_approval" => {
                if !syscall_allowed(request, &self.capabilities, op, true, &["approve_action"]) {
                    syscall_forbidden(op)
                } else {
                    request_approval_syscall(request, &self.capabilities, &input)
                }
            }
            _ => unavailable_effect(request.clone()),
        };
        syscall_audit(
            &self.capabilities,
            &input.version,
            op,
            syscall_audit_status(&response),
        );
        response
    }
}
