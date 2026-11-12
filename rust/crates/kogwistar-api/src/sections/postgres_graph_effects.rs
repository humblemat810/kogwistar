// Internal graph mutations, indexing, document projection, and effect dispatch.
// Python counterparts: engine_core.subsystems.write/embed and ChatRunService
// effect dispatch; public upsert operations live in the adjacent section.
impl PostgresRunApplicationService {
    async fn apply_internal_graph_mutations(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        let graph_store = self.graph_store_for_type(&self.graph_role);
        if !graph_store
            .graph_authority_capability()
            .supports_atomic_index_jobs()
        {
            return effect_error(
                StatusCode::NOT_IMPLEMENTED,
                "KOGWISTAR_GRAPH_AUTHORITY_UNSUPPORTED",
                "This service does not provide atomic graph authority with index jobs",
            );
        }
        if !capability_allowed(request, &self.capabilities, "write_graph", &["write_graph"]) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Graph mutation requires write_graph capability",
            );
        }
        let mut input: GraphMutationBatchInput = match serde_json::from_slice(&request.body) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error.to_string(),
                );
            }
        };
        if let Err(error) = input.validate() {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                error.to_string(),
            );
        }
        let mutations = match graph_mutation_batch_value(input.clone()) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error,
                );
            }
        };
        if let Err(error) = validate_graph_mutation_envelopes(&input, &mutations) {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                error,
            );
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
        if let Err(error) = self.embed_missing_graph_vectors(&mut input).await {
            return effect_error(StatusCode::BAD_GATEWAY, "KOGWISTAR_EMBEDDING_FAILED", error);
        }
        let mutations = match graph_mutation_batch_value(input.clone()) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error,
                );
            }
        };
        if let Err(error) = validate_graph_mutation_envelopes(&input, &mutations) {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                error,
            );
        }
        let applied = if input.index_jobs.is_empty() {
            graph_store.apply_graph_mutations_atomic(mutations).await
        } else {
            let index_jobs = match graph_index_jobs_value(&input, &mutations) {
                Ok(value) => value,
                Err(error) => {
                    return effect_error(
                        StatusCode::UNPROCESSABLE_ENTITY,
                        "KOGWISTAR_INVALID_REQUEST",
                        error,
                    );
                }
            };
            graph_store
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

    async fn embed_missing_graph_vectors(
        &self,
        input: &mut GraphMutationBatchInput,
    ) -> Result<(), String> {
        // A batch with index intents intentionally remains Stage 1 and is
        // embedded by the durable worker, never synchronously promoted here.
        if !input.index_jobs.is_empty() {
            return Ok(());
        }
        let Some(provider) = self.embedding_provider.as_ref() else {
            return Ok(());
        };
        let profile = provider.profile();
        profile.validate().map_err(|error| error.to_string())?;

        let mut positions = Vec::new();
        let mut texts = Vec::new();
        for (position, mutation) in input.mutations.iter_mut().enumerate() {
            if !matches!(mutation.op.as_str(), "ADD" | "REPLACE")
                || mutation.record.embedding.is_some()
                || mutation
                    .record
                    .metadata
                    .get("entity_type")
                    .and_then(Value::as_str)
                    == Some("acl_record")
            {
                continue;
            }
            if mutation.embedding_dim != 0 && mutation.embedding_dim != profile.dimension {
                return Err("embedding profile dimension conflicts with graph mutation".to_owned());
            }
            let text = mutation
                .record
                .document
                .as_ref()
                .ok_or_else(|| "missing graph document for embedding generation".to_owned())?;
            positions.push(position);
            texts.push(text.clone());
        }
        if texts.is_empty() {
            return Ok(());
        }

        let vectors = provider
            .embed(&texts)
            .await
            .map_err(|_| "embedding provider failed".to_owned())?;
        if vectors.len() != positions.len() {
            return Err("embedding provider returned an unexpected vector count".to_owned());
        }
        for (position, vector) in positions.into_iter().zip(vectors) {
            if vector.len() != profile.dimension || vector.iter().any(|value| !value.is_finite()) {
                return Err("embedding provider returned an invalid vector".to_owned());
            }
            let vector = vector
                .into_iter()
                .map(|value| value as f32)
                .collect::<Vec<_>>();
            if vector.iter().any(|value| !value.is_finite()) {
                return Err("embedding provider vector exceeds graph storage range".to_owned());
            }
            input.mutations[position].record.embedding = Some(vector);
            input.mutations[position].embedding_dim = profile.dimension;
        }
        input.validate().map_err(|error| error.to_string())
    }

    async fn document_projection_effect(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        let write = match document_projection_write(request, &self.capabilities) {
            Ok(write) => write,
            Err(response) => return response,
        };
        if !self
            .document_authority_capability()
            .supports_projection_upsert()
        {
            return effect_error(
                StatusCode::NOT_IMPLEMENTED,
                "KOGWISTAR_DOCUMENT_AUTHORITY_UNSUPPORTED",
                "The selected Rust store does not own document projections",
            );
        }
        let write = match embed_document_projection(write, self.embedding_provider.as_deref()).await
        {
            Ok(write) => write,
            Err(()) => {
                return effect_error(
                    StatusCode::BAD_GATEWAY,
                    "KOGWISTAR_EMBEDDING_FAILED",
                    "The configured embedding provider returned an invalid result or failed",
                );
            }
        };
        match self.store.upsert_document_projection(write).await {
            Ok(()) => json_effect(StatusCode::OK, json!({"ok": true})),
            Err(error) => effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_DOCUMENT_PROJECTION_FAILED",
                error.to_string(),
            ),
        }
    }

    async fn execute_async(&self, request: ApiEffectRequest) -> ApiEffectResponse {
        if request.method == "POST"
            && let Some((run_id, tail)) = runtime_run_route(&request.path_and_query)
            && tail.iter().map(String::as_str).collect::<Vec<_>>() == ["resume"]
        {
            return self.resume_runtime_run_and_wait(&run_id, &request).await;
        }
        if is_pending_server_cutover_route(&request.method, &request.path_and_query) {
            return effect_error(
                StatusCode::NOT_IMPLEMENTED,
                "KOGWISTAR_ROUTE_DEFERRED",
                "This route remains deferred to the Python application service",
            );
        }
        if request.method == "POST" && is_conversation_list_route(&request.path_and_query) {
            let input: CreateConversationHttpInput = match serde_json::from_slice(&request.body) {
                Ok(input) => input,
                Err(error) => {
                    return effect_error(
                        StatusCode::UNPROCESSABLE_ENTITY,
                        "KOGWISTAR_INVALID_REQUEST",
                        error.to_string(),
                    );
                }
            };
            let mut args =
                serde_json::Map::from_iter([("user_id".to_owned(), json!(input.user_id))]);
            if let Some(conversation_id) = input.conversation_id {
                args.insert("conversation_id".to_owned(), json!(conversation_id));
            }
            if let Some(start_node_id) = input.start_node_id {
                args.insert("start_node_id".to_owned(), json!(start_node_id));
            }
            let syscall_request = ApiEffectRequest {
                contract_version: request.contract_version,
                method: "POST".to_owned(),
                path_and_query: "/api/syscall/v1/create_conversation".to_owned(),
                body: serde_json::to_vec(&json!({
                    "version":"v1",
                    "op":"create_conversation",
                    "args":args,
                }))
                .expect("typed conversation-create input serializes"),
                principal: request.principal,
            };
            let response = self
                .syscall_effect(&syscall_request, "create_conversation")
                .await;
            if !(200..300).contains(&response.status) {
                return response;
            }
            let envelope: Value = match serde_json::from_slice(&response.body) {
                Ok(envelope) => envelope,
                Err(error) => {
                    return effect_error(
                        StatusCode::BAD_GATEWAY,
                        "KOGWISTAR_INVALID_SYSCALL_RESPONSE",
                        error.to_string(),
                    );
                }
            };
            return match envelope.get("result") {
                Some(result) => json_effect(StatusCode::OK, result.clone()),
                None => effect_error(
                    StatusCode::BAD_GATEWAY,
                    "KOGWISTAR_INVALID_SYSCALL_RESPONSE",
                    "conversation-create syscall returned no result",
                ),
            };
        }
        if request.method == "POST"
            && let Some(conversation_id) = conversation_answer_id(&request.path_and_query)
        {
            let input: SubmitConversationAnswerHttpInput =
                match serde_json::from_slice(&request.body) {
                    Ok(input) => input,
                    Err(error) => {
                        return effect_error(
                            StatusCode::UNPROCESSABLE_ENTITY,
                            "KOGWISTAR_INVALID_REQUEST",
                            error.to_string(),
                        );
                    }
                };
            if input.text.is_empty() {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    "text must contain at least one character",
                );
            }
            // Match the Python route: an authenticated identity takes precedence
            // over the optional compatibility field in the request body.
            let effective_user_id = authenticated_run_owner_id(&request.principal)
                .map(str::to_owned)
                .or(input.user_id);
            let mut args = serde_json::Map::from_iter([
                ("conversation_id".to_owned(), json!(conversation_id)),
                ("text".to_owned(), json!(input.text)),
                ("workflow_id".to_owned(), json!(input.workflow_id)),
            ]);
            if let Some(user_id) = effective_user_id {
                args.insert("user_id".to_owned(), json!(user_id));
            }
            let syscall_request = ApiEffectRequest {
                contract_version: request.contract_version,
                method: "POST".to_owned(),
                path_and_query: "/api/syscall/v1/send_message".to_owned(),
                body: serde_json::to_vec(&json!({
                    "version":"v1",
                    "op":"send_message",
                    "args":args,
                }))
                .expect("typed answer-submission input serializes"),
                principal: request.principal,
            };
            let response = self.syscall_effect(&syscall_request, "send_message").await;
            if !(200..300).contains(&response.status) {
                return response;
            }
            let envelope: Value = match serde_json::from_slice(&response.body) {
                Ok(envelope) => envelope,
                Err(error) => {
                    return effect_error(
                        StatusCode::BAD_GATEWAY,
                        "KOGWISTAR_INVALID_SYSCALL_RESPONSE",
                        error.to_string(),
                    );
                }
            };
            let Some(result) = envelope.get("result") else {
                return effect_error(
                    StatusCode::BAD_GATEWAY,
                    "KOGWISTAR_INVALID_SYSCALL_RESPONSE",
                    "send-message syscall returned no result",
                );
            };
            let Some(run_id) = result.get("run_id").and_then(Value::as_str) else {
                return effect_error(
                    StatusCode::BAD_GATEWAY,
                    "KOGWISTAR_INVALID_SYSCALL_RESPONSE",
                    "send-message syscall result omitted run_id",
                );
            };
            let Some(turn_node_id) = result.get("turn_node_id").and_then(Value::as_str) else {
                return effect_error(
                    StatusCode::BAD_GATEWAY,
                    "KOGWISTAR_INVALID_SYSCALL_RESPONSE",
                    "send-message syscall result omitted turn_node_id",
                );
            };
            let workflow_id = result
                .get("workflow_id")
                .and_then(Value::as_str)
                .unwrap_or("debug.rag.v1");
            return json_effect(
                StatusCode::ACCEPTED,
                json!({
                    "run_id": run_id,
                    "conversation_id": conversation_id,
                    "workflow_id": workflow_id,
                    "status": "queued",
                    "user_turn_node_id": turn_node_id,
                }),
            );
        }
        if request.method == "GET"
            && request.path_and_query.split('?').next() == Some("/api/workflow/operator/inbox")
        {
            if let Some(error) = operator_inbox_authorization_error(&request, &self.capabilities) {
                return error;
            }
            let scope = security_scope_value(&request.principal);
            let namespace = scope["storage_namespace"].as_str().unwrap_or("default");
            let inbox_id = decoded_query_string(&request.path_and_query, "inbox_id");
            let status = decoded_query_string(&request.path_and_query, "status");
            let lanes = match self
                .store
                .list_projected_lane_messages(LaneMessageFilter {
                    namespace: Some(namespace.to_owned()),
                    inbox_id,
                    status,
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
            let nodes = match self.authorized_conversation_nodes(&request.principal).await {
                Ok(nodes) => nodes,
                Err(error) => {
                    return effect_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "KOGWISTAR_STORE_ERROR",
                        error,
                    );
                }
            };
            return operator_inbox_effect(&request, &lanes, &nodes);
        }
        if request.method == "GET"
            && request.path_and_query.split('?').next() == Some("/api/workflow/tools/audit")
        {
            return self.tool_audit_effect(&request).await;
        }
        if request.method == "GET" && is_conversation_list_route(&request.path_and_query) {
            return self.conversation_list_effect(&request).await;
        }
        if request.method == "GET" && conversation_transcript_id(&request.path_and_query).is_some()
        {
            return self.conversation_transcript_effect(&request).await;
        }
        if request.method == "GET" && conversation_detail_id(&request.path_and_query).is_some() {
            return self.conversation_detail_effect(&request).await;
        }
        if request.method == "GET"
            && latest_snapshot_conversation_id(&request.path_and_query).is_some()
        {
            return self.latest_snapshot_effect(&request).await;
        }
        if request.method == "POST" && request.path_and_query == INTERNAL_DOCUMENT_LOOKUP_ROUTE {
            return document_id_lookup_effect(&request, &self.store, &self.capabilities).await;
        }
        if is_document_projection_route(&request.method, &request.path_and_query) {
            let public_upsert =
                request.path_and_query.split('?').next() == Some(PUBLIC_DOCUMENT_UPSERT_ROUTE);
            let response = self.document_projection_effect(&request).await;
            if public_upsert && response.status == StatusCode::OK {
                return json_effect(StatusCode::OK, json!({"status": "ok"}));
            }
            return response;
        }
        if is_public_document_tree_upsert_route(&request.method, &request.path_and_query) {
            return self.public_document_tree_upsert_effect(&request).await;
        }
        if is_public_graph_upsert_route(&request.method, &request.path_and_query) {
            return self.public_graph_upsert_effect(&request).await;
        }
        if is_public_wisdom_graph_upsert_route(&request.method, &request.path_and_query) {
            return self.public_wisdom_graph_upsert_effect(&request).await;
        }
        if request.method == "POST" && request.path_and_query == WISDOM_SEMANTIC_SEED_EXPAND_ROUTE {
            return self.wisdom_semantic_seed_expand_effect(&request).await;
        }
        if request.method == "POST" && request.path_and_query == GRAPH_SEMANTIC_SEED_EXPAND_ROUTE {
            return self.graph_semantic_seed_expand_effect(&request).await;
        }
        if let Some(response) = visualization_shell_effect(&request) {
            return response;
        }
        if let Some(response) = document_graph_validation_effect(&request) {
            return response;
        }
        if let Some(response) = capability_effect(&request, &self.capabilities) {
            return response;
        }
        if let Some(response) = syscall_read_effect(&request, &self.capabilities) {
            return response;
        }
        if request.method == "POST"
            && let Some(op) = syscall_operation(&request.path_and_query)
        {
            return self.syscall_effect(&request, &op).await;
        }
        if let Some(response) = designer_capabilities_effect(&request, &self.capabilities) {
            return response;
        }
        if let Some((workflow_id, route)) = workflow_design_route(&request.path_and_query) {
            return self
                .workflow_design_effect(&request, &workflow_id, route)
                .await;
        }
        if let Some(response) = self.service_read(&request).await {
            return response;
        }
        if request.method == "POST"
            && request.path_and_query.split('?').next() == Some("/api/workflow/services")
        {
            return self.declare_service(&request).await;
        }
        if request.method == "POST"
            && let Some(service_id) = service_trigger_route(&request.path_and_query)
        {
            return self.trigger_service(&request, &service_id).await;
        }
        if request.method == "POST"
            && let Some(service_id) = service_repair_route(&request.path_and_query)
        {
            return self
                .repair_service_projections(&request, service_id.as_deref())
                .await;
        }
        if request.method == "POST"
            && let Some((service_id, enabled)) = service_enabled_route(&request.path_and_query)
        {
            return self
                .set_service_enabled(&request, &service_id, enabled)
                .await;
        }
        if request.method == "POST"
            && let Some((service_id, action)) = request
                .path_and_query
                .split('?')
                .next()
                .and_then(|path| path.strip_prefix("/api/workflow/services/"))
                .and_then(|rest| rest.split_once('/'))
            && action == "heartbeat"
        {
            return self.record_service_heartbeat(&request, service_id).await;
        }
        if request.method == "POST"
            && let Some(run_id) = dead_letter_replay_run_id(&request.path_and_query)
        {
            return self.replay_dead_letter(&request, &run_id).await;
        }
        if request.method == "POST"
            && request.path_and_query.split('?').next()
                == Some("/api/workflow/messages/repair-orphans")
        {
            return self.repair_orphaned_messages(&request).await;
        }
        if request.method == "POST" && request.path_and_query == INTERNAL_RUNTIME_CLAIM_ROUTE {
            return self.claim_runtime_work(&request).await;
        }
        if request.method == "POST" && request.path_and_query == INTERNAL_RUNTIME_RESULTS_ROUTE {
            return self.apply_runtime_result(&request).await;
        }
        if request.method == "POST"
            && request.path_and_query == INTERNAL_RUNTIME_NATIVE_EXECUTE_ROUTE
        {
            return self.execute_native_runtime_step(&request).await;
        }
        if request.method == "POST"
            && request.path_and_query == INTERNAL_CONVERSATION_TRANSCRIPT_ROUTE
        {
            return self.internal_conversation_transcript_effect(&request).await;
        }
        if request.method == "POST"
            && request.path_and_query == INTERNAL_CONVERSATION_CONTEXT_SNAPSHOT_ROUTE
        {
            return self.internal_context_snapshot_effect(&request).await;
        }
        if request.method == "POST" && request.path_and_query == INTERNAL_GRAPH_MUTATIONS_ROUTE {
            return self.apply_internal_graph_mutations(&request).await;
        }
        if request.method == "POST" && request.path_and_query == INTERNAL_GRAPH_LOAD_PERSISTED_ROUTE
        {
            return self.graph_load_persisted_effect(&request).await;
        }
        if request.method == "POST"
            && request.path_and_query == INTERNAL_GRAPH_PROPOSE_BRUTE_FORCE_ROUTE
        {
            return self.graph_propose_bruteforce_effect(&request).await;
        }
        if request.method == "POST" && request.path_and_query == INTERNAL_GRAPH_NEIGHBORS_ROUTE {
            return self.graph_neighbors_effect(&request).await;
        }
        if request.method == "POST" && request.path_and_query == INTERNAL_GRAPH_VECTOR_QUERY_ROUTE {
            return self.graph_vector_query_effect(&request).await;
        }
        if request.method == "POST"
            && request.path_and_query.split('?').next() == Some("/api/add_index_entries")
        {
            return crate::search_index_fts::search_index_upsert_effect(
                &request,
                &self.capabilities,
                &self.store,
                &self.store,
                "gke_node_index",
                self.search_index_fts.as_deref(),
                self.embedding_provider.as_deref(),
            )
            .await;
        }
        if request.method == "GET"
            && request.path_and_query.split('?').next() == Some("/api/search_index_hybrid")
        {
            return self.search_index_hybrid_effect(&request).await;
        }
        if request.method == "POST" && request.path_and_query == INTERNAL_GRAPH_FIND_EDGES_ROUTE {
            return self.graph_find_edges_effect(&request).await;
        }
        if request.method == "POST" && request.path_and_query == INTERNAL_GRAPH_SHORTEST_PATH_ROUTE
        {
            return self.graph_shortest_path_effect(&request).await;
        }
        if request.method == "POST" && request.path_and_query == INTERNAL_GRAPH_K_HOP_ROUTE {
            return self.graph_k_hop_effect(&request).await;
        }
        if request.method == "POST" && request.path_and_query == INTERNAL_GRAPH_VIZ_D3_ROUTE {
            return self.graph_visualization_effect(&request, false).await;
        }
        if request.method == "POST" && request.path_and_query == INTERNAL_GRAPH_VIZ_CYTOSCAPE_ROUTE
        {
            return self.graph_visualization_effect(&request, true).await;
        }
        if request.method == "POST"
            && request.path_and_query.split('?').next() == Some("/api/workflow/runs")
        {
            if !capability_allowed(
                &request,
                &self.capabilities,
                "spawn_process",
                &["spawn_process", "workflow.run.write"],
            ) {
                return effect_error(
                    StatusCode::FORBIDDEN,
                    "KOGWISTAR_CAPABILITY_FORBIDDEN",
                    "Submitting workflow run requires spawn_process and workflow.run.write capabilities",
                );
            }
            return self.submit_run(&request).await;
        }
        if request.method == "GET"
            && let Some(path) = operational_path(&request.path_and_query)
        {
            let process_table = path == "/api/workflow/process-table";
            let process_listing = process_table || path == "/api/workflow/blocked-runs";
            if path == "/api/workflow/dead-letters"
                && (principal_role(&request.principal) != "ro"
                    && principal_role(&request.principal) != "rw"
                    || !principal_strings(&request.principal, "ns")
                        .iter()
                        .any(|namespace| namespace == "workflow" || namespace == "*"))
            {
                return effect_error(
                    StatusCode::FORBIDDEN,
                    "KOGWISTAR_SCOPE_FORBIDDEN",
                    "Reading dead letters requires ro/rw role and workflow namespace",
                );
            }
            if path == "/api/workflow/dead-letters"
                && !capability_allowed(
                    &request,
                    &self.capabilities,
                    "project_view",
                    &["project_view"],
                )
            {
                return effect_error(
                    StatusCode::FORBIDDEN,
                    "KOGWISTAR_CAPABILITY_FORBIDDEN",
                    "Reading dead letters requires project_view capability",
                );
            }
            if path == "/api/workflow/resources"
                && !capability_allowed(
                    &request,
                    &self.capabilities,
                    "read_security_scope",
                    &["read_security_scope", "project_view"],
                )
            {
                return effect_error(
                    StatusCode::FORBIDDEN,
                    "KOGWISTAR_CAPABILITY_FORBIDDEN",
                    "Reading resource snapshot requires read_security_scope capability",
                );
            }
            if matches!(
                path,
                "/api/workflow/budget" | "/api/workflow/budget/history"
            ) && !capability_allowed(
                &request,
                &self.capabilities,
                "project_view",
                &["project_view", "read_security_scope"],
            ) {
                return effect_error(
                    StatusCode::FORBIDDEN,
                    "KOGWISTAR_CAPABILITY_FORBIDDEN",
                    "Reading budget data requires project_view and read_security_scope capabilities",
                );
            }
            if path == "/api/workflow/operator/dashboard"
                && let Some(error) =
                    operator_inbox_authorization_error(&request, &self.capabilities)
            {
                return error;
            }
            if path == "/api/workflow/operator/dashboard"
                && !capability_allowed(
                    &request,
                    &self.capabilities,
                    "project_view",
                    &["project_view", "workflow.run.read", "read_security_scope"],
                )
            {
                return effect_error(
                    StatusCode::FORBIDDEN,
                    "KOGWISTAR_CAPABILITY_FORBIDDEN",
                    "Reading operator dashboard requires project_view, workflow.run.read, and read_security_scope capabilities",
                );
            }
            if process_listing
                && !capability_any_allowed(
                    &request,
                    &self.capabilities,
                    "project_view",
                    &["project_view", "workflow.run.read"],
                )
            {
                return effect_error(
                    StatusCode::FORBIDDEN,
                    "KOGWISTAR_CAPABILITY_FORBIDDEN",
                    "Reading process table requires project_view or workflow.run.read capability",
                );
            }
            let limit = query_integer(
                &request.path_and_query,
                "limit",
                if process_listing { 100 } else { 200 },
            )
            .clamp(1, 10_000) as usize;
            let lane_namespace = security_scope_value(&request.principal)["storage_namespace"]
                .as_str()
                .unwrap_or("default")
                .to_owned();
            let runs = match self
                .store
                .list_server_runs(
                    process_table
                        .then(|| decoded_query_string(&request.path_and_query, "status"))
                        .flatten()
                        .as_deref(),
                    process_table
                        .then(|| decoded_query_string(&request.path_and_query, "workflow_id"))
                        .flatten()
                        .as_deref(),
                    process_table
                        .then(|| decoded_query_string(&request.path_and_query, "conversation_id"))
                        .flatten()
                        .as_deref(),
                    if process_listing { limit } else { 10_000 },
                )
                .await
            {
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
                    namespace: Some(lane_namespace),
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
            let dashboard_inbox = if path == "/api/workflow/operator/dashboard" {
                let nodes = match self.authorized_conversation_nodes(&request.principal).await {
                    Ok(nodes) => nodes,
                    Err(error) => {
                        return effect_error(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "KOGWISTAR_STORE_ERROR",
                            error,
                        );
                    }
                };
                let inbox = operator_inbox_effect(&request, &lanes, &nodes);
                let inbox: Value =
                    serde_json::from_slice(&inbox.body).unwrap_or_else(|_| json!({}));
                inbox["messages"].as_array().cloned().unwrap_or_default()
            } else {
                Vec::new()
            };
            let mut events = Vec::new();
            for run in &runs {
                match self
                    .store
                    .list_server_run_events(
                        &run.run_id,
                        0,
                        if process_listing { 50 } else { limit },
                    )
                    .await
                {
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
            let service_rows = match self.store.list_named_projections("service_registry").await {
                Ok(rows) => rows,
                Err(error) => {
                    return effect_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "KOGWISTAR_STORE_ERROR",
                        error.to_string(),
                    );
                }
            };
            let services = service_rows
                .into_iter()
                .map(|row| Value::Object(row.payload))
                .collect::<Vec<_>>();
            let service_event_rows = if process_listing {
                match self
                    .store
                    .replay_raw_events("workflow", 0, usize::MAX)
                    .await
                {
                    Ok(rows) => rows
                        .into_iter()
                        .map(|event| {
                            (
                                event.seq,
                                event.entity_kind,
                                event.entity_id,
                                event.op,
                                event.payload_json,
                            )
                        })
                        .collect(),
                    Err(error) => {
                        return effect_error(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "KOGWISTAR_STORE_ERROR",
                            error.to_string(),
                        );
                    }
                }
            } else {
                Vec::new()
            };
            let response = operational_effect(
                &request,
                path,
                runs,
                lanes,
                events,
                services,
                service_event_rows,
            );
            if path == "/api/workflow/operator/dashboard" {
                let response = operator_dashboard_with_capabilities(response, {
                    let capability_state = self
                        .capabilities
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    capability_snapshot(&request.principal, &capability_state)
                });
                return operator_dashboard_with_inbox(response, dashboard_inbox);
            }
            return response;
        }
        let Some((run_id, tail)) = runtime_run_route(&request.path_and_query) else {
            return unavailable_effect(request);
        };
        let tail = tail.iter().map(String::as_str).collect::<Vec<_>>();
        if request.method == "GET"
            && (tail.is_empty()
                || (tail.as_slice() == ["events", "poll"]
                    && request.path_and_query.starts_with("/api/workflow/runs/"))
                || (tail.as_slice() == ["steps"]
                    && request.path_and_query.starts_with("/api/workflow/runs/"))
                || is_runtime_inspection_tail(&tail))
        {
            // PYTHON_PARITY_V1: checkpoint/replay/step reads require workflow scope;
            // python=kogwistar/server/chat_api.py#get_checkpoint and #replay_run
            let required_namespace = if is_runtime_inspection_tail(&tail) {
                "workflow"
            } else {
                runtime_run_namespace(&request.path_and_query).unwrap_or("workflow")
            };
            if !principal_strings(&request.principal, "ns")
                .iter()
                .any(|namespace| namespace == "*" || namespace == required_namespace)
            {
                return effect_error(
                    StatusCode::FORBIDDEN,
                    "KOGWISTAR_NAMESPACE_FORBIDDEN",
                    format!("Run reads require {required_namespace} namespace"),
                );
            }
            if !capability_allowed(
                &request,
                &self.capabilities,
                "workflow.run.read",
                &["workflow.run.read"],
            ) {
                return effect_error(
                    StatusCode::FORBIDDEN,
                    "KOGWISTAR_CAPABILITY_FORBIDDEN",
                    "Reading workflow runs requires workflow.run.read capability",
                );
            }
        }
        // PYTHON_PARITY_V1: authorize workflow cancel scope before lookup; python=kogwistar/server/runtime_api.py#create_runtime_router.cancel_workflow_run
        if request.method == "POST" && tail == ["cancel"] {
            let required_namespace =
                runtime_run_namespace(&request.path_and_query).unwrap_or("workflow");
            if !principal_strings(&request.principal, "ns")
                .iter()
                .any(|namespace| namespace == "*" || namespace == required_namespace)
            {
                return effect_error(
                    StatusCode::FORBIDDEN,
                    "KOGWISTAR_NAMESPACE_FORBIDDEN",
                    format!("Cancelling runs requires {required_namespace} namespace"),
                );
            }
        }
        if request.method == "GET" && tail == ["evidence"] {
            let namespaces = principal_strings(&request.principal, "ns");
            if !namespaces
                .iter()
                .any(|namespace| matches!(namespace.as_str(), "*" | "workflow"))
            {
                return effect_error(
                    StatusCode::FORBIDDEN,
                    "KOGWISTAR_NAMESPACE_FORBIDDEN",
                    "Run evidence reads require workflow namespace",
                );
            }
            for (action, message) in [
                (
                    "workflow.run.read",
                    "Reading run evidence requires workflow.run.read capability",
                ),
                (
                    "project_view",
                    "Reading run evidence snapshots requires project_view capability",
                ),
            ] {
                if !capability_allowed(&request, &self.capabilities, action, &[action]) {
                    return effect_error(
                        StatusCode::FORBIDDEN,
                        "KOGWISTAR_CAPABILITY_FORBIDDEN",
                        message,
                    );
                }
            }
        }
        if request.method == "GET" && tail == ["lineage"] {
            let namespaces = principal_strings(&request.principal, "ns");
            if !namespaces
                .iter()
                .any(|namespace| matches!(namespace.as_str(), "*" | "workflow"))
            {
                return effect_error(
                    StatusCode::FORBIDDEN,
                    "KOGWISTAR_NAMESPACE_FORBIDDEN",
                    "Workflow lineage reads require workflow namespace",
                );
            }
            if !capability_allowed(
                &request,
                &self.capabilities,
                "workflow.run.read",
                &["workflow.run.read"],
            ) {
                return effect_error(
                    StatusCode::FORBIDDEN,
                    "KOGWISTAR_CAPABILITY_FORBIDDEN",
                    "Reading workflow run lineage requires workflow.run.read capability",
                );
            }
        }
        // PYTHON_PARITY_V1: resume-contract reads require workflow.run.read before run lookup; python=kogwistar/server/chat_service.py#ChatRunService.resume_contract
        if request.method == "GET" && tail == ["resume-contract"] {
            if !principal_strings(&request.principal, "ns")
                .iter()
                .any(|namespace| namespace == "*" || namespace == "workflow")
            {
                return effect_error(
                    StatusCode::FORBIDDEN,
                    "KOGWISTAR_NAMESPACE_FORBIDDEN",
                    "Resume-contract reads require workflow namespace",
                );
            }
            if !capability_allowed(
                &request,
                &self.capabilities,
                "workflow.run.read",
                &["workflow.run.read"],
            ) {
                return effect_error(
                    StatusCode::FORBIDDEN,
                    "KOGWISTAR_CAPABILITY_FORBIDDEN",
                    "Reading resume contract requires workflow.run.read capability",
                );
            }
        }
        let run = match self.store.get_server_run(&run_id).await {
            Ok(Some(run)) => run,
            Ok(None) => {
                return effect_error(
                    StatusCode::NOT_FOUND,
                    "KOGWISTAR_RUN_NOT_FOUND",
                    format!("Unknown run_id: {run_id}"),
                );
            }
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error.to_string(),
                );
            }
        };
        // PYTHON_PARITY_V1: non-terminal cancellation requires write capability; python=kogwistar/server/chat_service.py#ChatRunService.cancel_run
        if request.method == "POST"
            && tail == ["cancel"]
            && !matches!(run.status.as_str(), "succeeded" | "failed" | "cancelled")
            && !capability_allowed(
                &request,
                &self.capabilities,
                "workflow.run.write",
                &["workflow.run.write"],
            )
        {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Cancelling a non-terminal run requires workflow.run.write capability",
            );
        }
        if request.method == "GET"
            && (tail.is_empty()
                || (tail.as_slice() == ["steps"]
                    && request.path_and_query.starts_with("/api/workflow/runs/")))
        {
            if let (Some(auth_store), Some(user_id)) = (
                self.owner_auth_store.as_ref(),
                authenticated_run_owner_id(&request.principal),
            ) {
                match auth_store
                    .workflow_access_allowed(&run.workflow_id, user_id, "ro")
                    .await
                {
                    Ok(true) => {}
                    Ok(false) => {
                        return effect_error(
                            StatusCode::FORBIDDEN,
                            "KOGWISTAR_WORKFLOW_FORBIDDEN",
                            "User does not have read access to this workflow",
                        );
                    }
                    Err(error) => {
                        return effect_error(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "KOGWISTAR_AUTH_STORE_ERROR",
                            error,
                        );
                    }
                }
            }
        }
        if request.method == "GET" && tail == ["lineage"] {
            let user_id = request
                .principal
                .get("user_id")
                .or_else(|| request.principal.get("sub"))
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty() && !value.eq_ignore_ascii_case("anonymous"));
            if let Some(user_id) = user_id {
                let Some(auth_store) = self.owner_auth_store.as_ref() else {
                    return effect_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "KOGWISTAR_AUTH_UNAVAILABLE",
                        "Workflow ACL authority is not configured",
                    );
                };
                match auth_store
                    .workflow_access_allowed(&run.workflow_id, user_id, "ro")
                    .await
                {
                    Ok(true) => {}
                    Ok(false) => {
                        return effect_error(
                            StatusCode::FORBIDDEN,
                            "KOGWISTAR_WORKFLOW_FORBIDDEN",
                            "User does not have read access to this workflow",
                        );
                    }
                    Err(error) => {
                        return effect_error(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "KOGWISTAR_AUTH_STORE_ERROR",
                            error,
                        );
                    }
                }
            }
            let (scope, acl_principal) =
                conversation_graph_scope_and_acl_principal(&request.principal);
            let mut lineage = Vec::new();
            let mut current = run_id.clone();
            let mut seen = BTreeSet::new();
            while !current.is_empty() && seen.insert(current.clone()) {
                let node_id = format!("wf_run|{current}");
                let records = match graph_projection_records_all(
                    &self.store,
                    GraphProjectionRead {
                        scope: scope.clone(),
                        table: "gke_nodes".to_owned(),
                        ids: Some(vec![node_id.clone()]),
                        metadata: MetadataFilter::default(),
                        limit: 1,
                    },
                )
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
                let acl_records = match graph_projection_records_all(
                    &self.store,
                    GraphProjectionRead {
                        scope: scope.clone(),
                        table: "gke_nodes".to_owned(),
                        ids: None,
                        metadata: MetadataFilter {
                            equals: BTreeMap::from([
                                ("entity_type".to_owned(), json!("acl_record")),
                                ("acl_truth_graph".to_owned(), json!("conversation")),
                                ("acl_target_grain".to_owned(), json!("node")),
                                ("acl_target_entity_id".to_owned(), json!(node_id)),
                            ]),
                        },
                        limit: 10_000,
                    },
                )
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
                let mut acl_index_records = records;
                acl_index_records.extend(acl_records);
                let (visible_records, _) = graph_records_visible_by_acl(
                    &acl_index_records,
                    &[],
                    "conversation",
                    &acl_principal,
                );
                let Some(record) = visible_records.first() else {
                    if lineage.is_empty() {
                        return effect_error(
                            StatusCode::NOT_FOUND,
                            "KOGWISTAR_RUN_NOT_FOUND",
                            format!("Workflow run not found: {run_id}"),
                        );
                    }
                    break;
                };
                if matches!(
                    conversation_record_string(record, "lifecycle_status").as_deref(),
                    Some("tombstoned" | "deleted")
                ) {
                    if lineage.is_empty() {
                        return effect_error(
                            StatusCode::NOT_FOUND,
                            "KOGWISTAR_RUN_NOT_FOUND",
                            format!("Workflow run not found: {run_id}"),
                        );
                    }
                    break;
                }
                let workflow_id =
                    conversation_record_string(record, "workflow_id").unwrap_or_default();
                if let Some(user_id) = user_id
                    && !workflow_id.is_empty()
                {
                    let Some(auth_store) = self.owner_auth_store.as_ref() else {
                        return effect_error(
                            StatusCode::SERVICE_UNAVAILABLE,
                            "KOGWISTAR_AUTH_UNAVAILABLE",
                            "Workflow ACL authority is not configured",
                        );
                    };
                    match auth_store
                        .workflow_access_allowed(&workflow_id, user_id, "ro")
                        .await
                    {
                        Ok(true) => {}
                        Ok(false) => {
                            return effect_error(
                                StatusCode::FORBIDDEN,
                                "KOGWISTAR_WORKFLOW_FORBIDDEN",
                                "User does not have read access to every workflow in this lineage",
                            );
                        }
                        Err(error) => {
                            return effect_error(
                                StatusCode::INTERNAL_SERVER_ERROR,
                                "KOGWISTAR_AUTH_STORE_ERROR",
                                error,
                            );
                        }
                    }
                }
                let parent = conversation_record_value(record, "parent_run_id");
                lineage.push(json!({
                    "run_id": current,
                    "workflow_id": workflow_id,
                    "parent_run_id": parent,
                    "status": conversation_record_string(record, "status").unwrap_or_default(),
                    "wf_invoked": conversation_record_value(record, "wf_invoked"),
                }));
                current = parent
                    .and_then(|value| value.as_str().map(str::to_owned))
                    .unwrap_or_default();
            }
            if query_parameter_present(&request.path_and_query, "limit") {
                let limit =
                    query_integer(&request.path_and_query, "limit", 200).clamp(1, 1000) as usize;
                let truncated = lineage.len() > limit;
                lineage.truncate(limit);
                return json_effect(
                    StatusCode::OK,
                    json!({"run_id": run_id, "lineage": lineage, "truncated": truncated}),
                );
            }
            return json_effect(
                StatusCode::OK,
                json!({"run_id": run_id, "lineage": lineage}),
            );
        }
        if request.method == "GET" && tail == ["evidence"] {
            let authorization = ContextSnapshotAuthorization {
                principal: request.principal.clone(),
            };
            let snapshot = match self
                .context_snapshot_projection(
                    &request,
                    InternalContextSnapshotRequest {
                        conversation_id: run.conversation_id,
                        run_id: Some(run_id.clone()),
                        stage: None,
                    },
                    authorization,
                )
                .await
            {
                Ok(snapshot) => snapshot,
                Err(response) => return response,
            };
            return json_effect(StatusCode::OK, public_run_evidence_value(&run_id, snapshot));
        }
        if request.method == "GET" && is_runtime_inspection_tail(&tail) {
            return match self.store.list_server_run_events(&run_id, 0, 200_000).await {
                Ok(events) => runtime_inspection_effect(&request, &run_id, &tail, events),
                Err(error) => effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error.to_string(),
                ),
            };
        }
        match (request.method.as_str(), tail.as_slice()) {
            ("GET", []) => json_effect(StatusCode::OK, server_run_value(run)),
            ("GET", ["events", "poll"]) => {
                let after_seq = query_integer(&request.path_and_query, "after_seq", 0);
                let limit = query_integer(&request.path_and_query, "limit", 500).clamp(1, 10_000);
                match self
                    .store
                    .list_server_run_events(&run_id, after_seq, limit as usize)
                    .await
                {
                    Ok(events) => {
                        let mut body = json!({
                            "run_id": run_id,
                            "events": events.into_iter().map(server_run_event_value).collect::<Vec<_>>(),
                        });
                        if query_parameter_present(&request.path_and_query, "after_client_sequence")
                        {
                            let caller_id = authenticated_run_owner_id(&request.principal);
                            if caller_id.is_none() || run.user_id.as_deref() != caller_id {
                                return effect_error(
                                    StatusCode::FORBIDDEN,
                                    "KOGWISTAR_RUN_OWNER_FORBIDDEN",
                                    "transient run deltas are visible only to the run owner",
                                );
                            }
                            let after_client_sequence =
                                query_integer(&request.path_and_query, "after_client_sequence", 0)
                                    .max(0);
                            let client_deltas = match self.transient_run_deltas.list(
                                &run_id,
                                after_client_sequence,
                                limit as usize,
                            ) {
                                Ok(deltas) => deltas,
                                Err(error) => {
                                    return effect_error(
                                        StatusCode::INTERNAL_SERVER_ERROR,
                                        "KOGWISTAR_TRANSIENT_EVENT_ERROR",
                                        error,
                                    );
                                }
                            };
                            body["client_deltas"] = json!(client_deltas);
                        }
                        json_effect(StatusCode::OK, body)
                    }
                    Err(error) => effect_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "KOGWISTAR_STORE_ERROR",
                        error.to_string(),
                    ),
                }
            }
            ("GET", ["events"]) => {
                let after_seq = query_integer(&request.path_and_query, "after_seq", 0);
                match self
                    .store
                    .list_server_run_events(&run_id, after_seq, 10_000)
                    .await
                {
                    Ok(events) => {
                        let mut body = String::new();
                        for event in events {
                            let payload = serde_json::from_str::<Value>(&event.payload_json)
                                .unwrap_or_else(|_| json!({}));
                            let mut event_payload = json!({
                                "run_id": event.run_id,
                                "event_type": event.event_type,
                                "created_at_ms": event.created_at_ms,
                            });
                            if let (Some(target), Some(source)) =
                                (event_payload.as_object_mut(), payload.as_object())
                            {
                                target.extend(source.clone());
                            }
                            let event_type =
                                event_payload["event_type"].as_str().unwrap_or("run.event");
                            body.push_str(&sse_frame(
                                event_type,
                                &event_payload,
                                Some(&event.seq.to_string()),
                            ));
                        }
                        if query_parameter_present(&request.path_and_query, "after_client_sequence")
                        {
                            let caller_id = authenticated_run_owner_id(&request.principal);
                            if caller_id.is_none() || run.user_id.as_deref() != caller_id {
                                return effect_error(
                                    StatusCode::FORBIDDEN,
                                    "KOGWISTAR_RUN_OWNER_FORBIDDEN",
                                    "transient run deltas are visible only to the run owner",
                                );
                            }
                            let after_client_sequence =
                                query_integer(&request.path_and_query, "after_client_sequence", 0)
                                    .max(0);
                            let deltas = match self.transient_run_deltas.list(
                                &run_id,
                                after_client_sequence,
                                MAX_TRANSIENT_DELTA_PAGE,
                            ) {
                                Ok(deltas) => deltas,
                                Err(error) => {
                                    return effect_error(
                                        StatusCode::INTERNAL_SERVER_ERROR,
                                        "KOGWISTAR_TRANSIENT_EVENT_ERROR",
                                        error,
                                    );
                                }
                            };
                            for mut delta in deltas {
                                let event_type = delta["event_type"]
                                    .as_str()
                                    .unwrap_or("response.text.delta")
                                    .to_owned();
                                let sequence =
                                    delta["client_sse_sequence"].as_i64().unwrap_or_default();
                                delta["run_id"] = json!(run_id);
                                let event_id = format!("delta-{sequence}");
                                body.push_str(&sse_frame(&event_type, &delta, Some(&event_id)));
                            }
                        }
                        ApiEffectResponse {
                            status: StatusCode::OK.as_u16(),
                            content_type: "text/event-stream; charset=utf-8".to_owned(),
                            body: body.into_bytes(),
                        }
                    }
                    Err(error) => effect_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "KOGWISTAR_STORE_ERROR",
                        error.to_string(),
                    ),
                }
            }
            ("GET", ["resume-contract"]) => match self
                .store
                .read_recorded_runtime_state(&run_id, &run.workflow_id, &run.conversation_id)
                .await
            {
                Ok(state) => json_effect(StatusCode::OK, resume_contract_value(state, &run_id)),
                Err(error) => effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error.to_string(),
                ),
            },
            ("POST", ["resume"]) => self.resume_runtime_run(&run_id, &request).await,
            ("POST", ["cancel"])
                if matches!(run.status.as_str(), "succeeded" | "failed" | "cancelled") =>
            {
                json_effect(StatusCode::ACCEPTED, server_run_value(run))
            }
            ("POST", ["cancel"]) => match self.store.request_server_run_cancel(&run_id).await {
                Ok(()) => match self.store.get_server_run(&run_id).await {
                    Ok(Some(updated)) => {
                        json_effect(StatusCode::ACCEPTED, server_run_value(updated))
                    }
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
            },
            _ => unavailable_effect(request),
        }
    }
}
