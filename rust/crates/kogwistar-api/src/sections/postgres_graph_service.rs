impl PostgresRunApplicationService {
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

    async fn internal_conversation_transcript_effect(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        if !capability_allowed(request, &self.capabilities, "read_graph", &["read_graph"]) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Conversation transcript reads require read_graph capability",
            );
        }
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
        let transcript = match conversation_transcript_from_visible_records(
            &visible_nodes,
            &input.conversation_id,
            input.limit,
        ) {
            Ok(transcript) => transcript,
            Err(error) => {
                return effect_error(
                    StatusCode::NOT_FOUND,
                    "KOGWISTAR_CONVERSATION_NOT_FOUND",
                    error.to_string(),
                );
            }
        };
        json_effect(
            StatusCode::OK,
            rust_native_attested(serde_json::to_value(transcript).expect("transcript serializes")),
        )
    }

    async fn internal_context_snapshot_effect(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        if !capability_allowed(
            request,
            &self.capabilities,
            "project_view",
            &["project_view"],
        ) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Context snapshot reads require project_view capability",
            );
        }
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
        let snapshot = match kogwistar_store::context_snapshot_from_visible_records(
            &visible_nodes,
            &input.conversation_id,
            input.run_id.as_deref(),
            input.stage.as_deref(),
        ) {
            Ok(snapshot) => snapshot,
            Err(error) => {
                return effect_error(
                    StatusCode::NOT_FOUND,
                    "KOGWISTAR_CONTEXT_SNAPSHOT_NOT_FOUND",
                    error.to_string(),
                );
            }
        };
        json_effect(
            StatusCode::OK,
            rust_native_attested(serde_json::to_value(snapshot).expect("snapshot serializes")),
        )
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
        let body = serde_json::to_vec(&json!({
            "conversation_id": conversation_id,
            "run_id": run_id,
            "stage": stage,
        }))
        .expect("context snapshot request serializes");
        let internal = ApiEffectRequest {
            contract_version: request.contract_version,
            method: "POST".to_owned(),
            path_and_query: "/internal/conversation/context-snapshot".to_owned(),
            body,
            principal: request.principal.clone(),
        };
        let mut response = self.internal_context_snapshot_effect(&internal).await;
        // Preserve the Python-compatible response shape; Rust projection and
        // seam metadata are implementation details at this boundary.
        strip_snapshot_implementation_fields(&mut response);
        response
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
        let internal = ApiEffectRequest {
            contract_version: request.contract_version,
            method: "POST".to_owned(),
            path_and_query: "/internal/conversation/transcript".to_owned(),
            body: serde_json::to_vec(&json!({
                "conversation_id": conversation_id,
                "limit": 10_000,
            }))
            .expect("conversation transcript request serializes"),
            principal: request.principal.clone(),
        };
        let mut response = self
            .internal_conversation_transcript_effect(&internal)
            .await;
        if (200..300).contains(&response.status)
            && let Ok(mut payload) = serde_json::from_slice::<Value>(&response.body)
            && let Some(object) = payload.as_object_mut()
        {
            object.remove("schema_version");
            object.remove("implementation_authority");
            object.remove("seam_attestation_version");
            object.remove("seam_crossings");
            if let Some(turns) = object.get_mut("turns").and_then(Value::as_array_mut) {
                for turn in turns {
                    if let Some(turn) = turn.as_object_mut() {
                        turn.remove("schema_version");
                    }
                }
            }
            response.body = serde_json::to_vec(&payload).expect("transcript response serializes");
        }
        response
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
        let mut groups = principal_strings(&request.principal, "groups");
        groups.extend(principal_strings(&request.principal, "principal_groups"));
        groups.sort();
        groups.dedup();
        let principal = GraphAclPrincipal {
            principal_id: principal_subject(&request.principal),
            groups: groups.into_iter().collect(),
            security_scope: scope_value["security_scope"].as_str().map(str::to_owned),
        };
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
        let mut groups = principal_strings(&request.principal, "groups");
        groups.extend(principal_strings(&request.principal, "principal_groups"));
        groups.sort();
        groups.dedup();
        let principal = GraphAclPrincipal {
            principal_id: principal_subject(&request.principal),
            groups: groups.into_iter().collect(),
            security_scope: scope_value["security_scope"].as_str().map(str::to_owned),
        };
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
        let mut groups = principal_strings(&request.principal, "groups");
        groups.extend(principal_strings(&request.principal, "principal_groups"));
        groups.sort();
        groups.dedup();
        let principal = GraphAclPrincipal {
            principal_id: principal_subject(&request.principal),
            groups: groups.into_iter().collect(),
            security_scope: scope_value["security_scope"].as_str().map(str::to_owned),
        };
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
        let mut groups = principal_strings(&request.principal, "groups");
        groups.extend(principal_strings(&request.principal, "principal_groups"));
        groups.sort();
        groups.dedup();
        let principal = GraphAclPrincipal {
            principal_id: principal_subject(&request.principal),
            groups: groups.into_iter().collect(),
            security_scope: scope_value["security_scope"].as_str().map(str::to_owned),
        };
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
                    let nested = ApiEffectRequest {
                        contract_version: request.contract_version,
                        method: "POST".to_owned(),
                        path_and_query: "/internal/conversation/transcript".to_owned(),
                        body: serde_json::to_vec(&json!({
                            "conversation_id": conversation_id,
                            "limit": 10_000,
                        }))
                        .expect("transcript syscall args serialize"),
                        principal: request.principal.clone(),
                    };
                    let mut transcript_response =
                        self.internal_conversation_transcript_effect(&nested).await;
                    legacy_receive_message_payload(&mut transcript_response);
                    syscall_result(&input.version, op, transcript_response)
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
                } else if input.args["conversation_id"].as_str().is_some() {
                    let nested = ApiEffectRequest {
                        contract_version: request.contract_version,
                        method: "POST".to_owned(),
                        path_and_query: "/internal/conversation/context-snapshot".to_owned(),
                        body: serde_json::to_vec(&json!({
                            "conversation_id": input.args["conversation_id"],
                            "run_id": input.args.get("run_id"),
                            "stage": input.args.get("stage"),
                        }))
                        .expect("snapshot syscall args serialize"),
                        principal: request.principal.clone(),
                    };
                    let mut snapshot_response =
                        self.internal_context_snapshot_effect(&nested).await;
                    strip_snapshot_implementation_fields(&mut snapshot_response);
                    syscall_result(&input.version, op, snapshot_response)
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
