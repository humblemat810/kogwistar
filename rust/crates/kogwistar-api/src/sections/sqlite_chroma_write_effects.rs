// SQLite/Chroma graph and conversation write effects, retaining SQLite authority.
// Python parity: kogwistar/server/chat_api.py, chat_service.py, and engine_core/subsystems/write.py.
impl SqliteRunApplicationService {
    async fn sqlite_chroma_invoke_builtin_tool_effect(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        let input = match syscall_input(request, "invoke_tool") {
            Ok(input) => input,
            Err(response) => return response,
        };
        if !syscall_allowed_in_namespace(
            request,
            &self.capabilities,
            "invoke_tool",
            true,
            &[],
            "conversation",
        ) {
            return syscall_forbidden("invoke_tool");
        }
        let Some(tool_name) = input
            .args
            .get("tool_name")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .map(str::to_ascii_lowercase)
        else {
            return effect_error(
                StatusCode::BAD_REQUEST,
                "KOGWISTAR_INVALID_REQUEST",
                "tool_name is required",
            );
        };

        let result = match tool_name.as_str() {
            "list_transcript" | "latest_snapshot" => {
                let Some(conversation_id) = input
                    .args
                    .get("conversation_id")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                else {
                    return effect_error(
                        StatusCode::BAD_REQUEST,
                        "KOGWISTAR_INVALID_REQUEST",
                        format!("conversation_id is required for {tool_name}"),
                    );
                };
                let operation = if tool_name == "list_transcript" {
                    "receive_message"
                } else {
                    "mount_memory"
                };
                let mut args = serde_json::Map::new();
                args.insert(
                    "conversation_id".to_owned(),
                    Value::String(conversation_id.to_owned()),
                );
                if tool_name == "latest_snapshot" {
                    for field in ["run_id", "stage"] {
                        if let Some(value) = input.args.get(field) {
                            args.insert(field.to_owned(), value.clone());
                        }
                    }
                }
                let nested_request = ApiEffectRequest {
                    contract_version: request.contract_version,
                    method: "POST".to_owned(),
                    path_and_query: format!("/api/syscall/v1/{operation}"),
                    body: serde_json::to_vec(&json!({
                        "version": input.version,
                        "args": args,
                    }))
                    .expect("nested syscall args serialize"),
                    principal: request.principal.clone(),
                };
                let response = if tool_name == "list_transcript" {
                    self.sqlite_chroma_receive_message_effect(&nested_request)
                        .await
                } else {
                    self.sqlite_chroma_mount_memory_effect(&nested_request)
                        .await
                };
                if !(200..300).contains(&response.status) {
                    return response;
                }
                let payload: Value = match serde_json::from_slice(&response.body) {
                    Ok(payload) => payload,
                    Err(error) => {
                        return effect_error(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "KOGWISTAR_INVALID_INTERNAL_RESPONSE",
                            error.to_string(),
                        );
                    }
                };
                let Some(mut value) = payload.get("result").cloned() else {
                    return effect_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "KOGWISTAR_INVALID_INTERNAL_RESPONSE",
                        "built-in graph tool returned no result",
                    );
                };
                if tool_name == "list_transcript" {
                    value["tool_name"] = Value::String(tool_name.clone());
                    value
                } else {
                    json!({
                        "tool_name": tool_name,
                        "conversation_id": conversation_id,
                        "snapshot": value,
                    })
                }
            }
            "capability_snapshot" => {
                if !capability_allowed(
                    request,
                    &self.capabilities,
                    "project_view",
                    &["project_view", "read_security_scope"],
                ) {
                    return effect_error(
                        StatusCode::FORBIDDEN,
                        "KOGWISTAR_CAPABILITY_FORBIDDEN",
                        "built-in tool \"capability_snapshot\" requires project_view and read_security_scope",
                    );
                }
                let state = self
                    .capabilities
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                json!({
                    "tool_name": tool_name,
                    "snapshot": capability_snapshot(&request.principal, &state),
                })
            }
            "resource_snapshot" => {
                if !capability_allowed(
                    request,
                    &self.capabilities,
                    "project_view",
                    &["project_view", "read_security_scope"],
                ) {
                    return effect_error(
                        StatusCode::FORBIDDEN,
                        "KOGWISTAR_CAPABILITY_FORBIDDEN",
                        "built-in tool \"resource_snapshot\" requires project_view and read_security_scope",
                    );
                }
                let runs = match self.store.list_server_runs(None, None, None, 10_000) {
                    Ok(rows) => rows,
                    Err(error) => {
                        return effect_error(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "KOGWISTAR_STORE_ERROR",
                            error.to_string(),
                        );
                    }
                };
                let lanes = match self.store.list_projected_lane_messages(LaneMessageFilter {
                    namespace: Some("workflow".to_owned()),
                    limit: 10_000,
                    ..LaneMessageFilter::default()
                }) {
                    Ok(rows) => rows,
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
                    match self.store.list_server_run_events(&run.run_id, 0, 200) {
                        Ok(mut rows) => events.append(&mut rows),
                        Err(error) => {
                            return effect_error(
                                StatusCode::INTERNAL_SERVER_ERROR,
                                "KOGWISTAR_STORE_ERROR",
                                error.to_string(),
                            );
                        }
                    }
                }
                let service_rows = match self.store.list_named_projections("service_registry") {
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
                let usage = usage_history(&events, 10_000);
                let mut snapshot = resource_value(
                    &runs,
                    &lanes,
                    &usage,
                    &events,
                    configured_storage_usage_bytes(),
                );
                snapshot["services"] = resource_service_value(&services);
                snapshot
                    .as_object_mut()
                    .expect("resource snapshot is an object")
                    .remove("migration");
                json!({"tool_name": tool_name, "snapshot": snapshot})
            }
            _ => {
                return effect_error(
                    StatusCode::NOT_FOUND,
                    "KOGWISTAR_TOOL_NOT_FOUND",
                    format!("unknown built-in tool: {tool_name}"),
                );
            }
        };
        syscall_result(
            &input.version,
            "invoke_tool",
            json_effect(StatusCode::OK, result),
        )
    }

    async fn sqlite_document_projection_effect(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        if !self
            .document_authority_capability()
            .supports_projection_upsert()
        {
            return effect_error(
                StatusCode::NOT_IMPLEMENTED,
                "KOGWISTAR_DOCUMENT_AUTHORITY_UNSUPPORTED",
                "The selected SQLite+Chroma service does not own source-document projections",
            );
        }
        let write = match document_projection_write(request, &self.capabilities) {
            Ok(write) => write,
            Err(response) => return response,
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

    async fn sqlite_chroma_graph_mutation_effect(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        if self.chroma_graph.is_none() {
            return effect_error(
                StatusCode::NOT_IMPLEMENTED,
                "KOGWISTAR_GRAPH_AUTHORITY_UNSUPPORTED",
                "SQLite graph mutations require the configured eventual Chroma projection",
            );
        }
        if !capability_allowed_for_principal(
            &request.principal,
            &self.capabilities,
            "write_graph",
            &["write_graph"],
        ) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Graph mutation requires write_graph capability",
            );
        }
        let input: GraphMutationBatchInput = match serde_json::from_slice(&request.body) {
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
            Ok(mutations) => mutations,
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
        let index_jobs = match graph_index_jobs_value(&input, &mutations) {
            Ok(jobs) => jobs,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error,
                );
            }
        };
        let committed = self.store.immediate_transaction(|uow| {
            uow.apply_canonical_graph_mutation_batch(GraphMutationBatch {
                mutations: mutations.clone(),
                index_jobs: index_jobs.clone(),
            })
        });
        match committed {
            Ok(results) => json_effect(
                StatusCode::OK,
                rust_native_attested(json!({
                    "contract_version": 1,
                    "atomic": true,
                    "projection_consistency": "eventual",
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

    // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/sqlite_service.rs#sqlite_chroma_create_conversation_effect; python=kogwistar/server/syscall_api.py#create_syscall_router.dispatch; fixture=contracts/golden/conversation-create-syscall-result-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/part_07.rs#sqlite_chroma_conversation_create_and_turn_admission_are_canonical_and_atomic; python_test=tests/server/test_syscall_api.py#test_syscall_create_conversation_uses_authenticated_owner
    async fn sqlite_chroma_create_conversation_effect(
        &self,
        request: &ApiEffectRequest,
        input: &SyscallInput,
    ) -> ApiEffectResponse {
        if !syscall_allowed_in_namespace(
            request,
            &self.capabilities,
            "create_conversation",
            true,
            &["spawn_process"],
            "conversation",
        ) {
            return syscall_forbidden("create_conversation");
        }
        let Some(owner_id) = authenticated_run_owner_id(&request.principal) else {
            return effect_error(
                StatusCode::UNAUTHORIZED,
                "KOGWISTAR_AUTHENTICATION_REQUIRED",
                "conversation creation requires an authenticated principal",
            );
        };
        let Some(requested_user_id) = input
            .args
            .get("user_id")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        else {
            return effect_error(
                StatusCode::BAD_REQUEST,
                "KOGWISTAR_INVALID_REQUEST",
                "user_id is required",
            );
        };
        if !requested_user_id.eq_ignore_ascii_case(owner_id) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_RUN_OWNER_FORBIDDEN",
                "user_id must match the authenticated principal",
            );
        }
        let optional_id = |key: &str| -> Result<Option<String>, ApiEffectResponse> {
            match input.args.get(key) {
                None | Some(Value::Null) => Ok(None),
                Some(Value::String(value)) if !value.trim().is_empty() => {
                    Ok(Some(value.trim().to_owned()))
                }
                _ => Err(effect_error(
                    StatusCode::BAD_REQUEST,
                    "KOGWISTAR_INVALID_REQUEST",
                    format!("{key} must be a non-empty string or null"),
                )),
            }
        };
        let conversation_id = match optional_id("conversation_id") {
            Ok(Some(value)) => value,
            Ok(None) => uuid::Uuid::new_v4().to_string(),
            Err(response) => return response,
        };
        let requested_start_id = match optional_id("start_node_id") {
            Ok(value) => value,
            Err(response) => return response,
        };
        let user_id = owner_id.to_owned();
        let start_id = requested_start_id.unwrap_or_else(|| {
            kogwistar_contracts::stable_id(
                "conversation.node",
                &[
                    user_id.clone(),
                    conversation_id.clone(),
                    "Start of conversation".to_owned(),
                    "-1".to_owned(),
                    "system".to_owned(),
                    "conversation_summary".to_owned(),
                    "True".to_owned(),
                ],
            )
            .to_string()
        });
        let scope_values = security_scope_value(&request.principal);
        let scope = GraphScope {
            namespace: scope_values["storage_namespace"]
                .as_str()
                .unwrap_or("default")
                .to_owned(),
            workspace_id: scope_values["workspace"].as_str().map(str::to_owned),
            graph_space: Some("conversation".to_owned()),
        };
        if scope.namespace != "conversation" || !graph_mutation_scope_allowed(&scope, &scope_values)
        {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_SCOPE_FORBIDDEN",
                "conversation creation requires the authenticated conversation storage scope",
            );
        }

        let metadata = json!({
            "level_from_root": 0,
            "entity_type": "conversation_start",
            "turn_index": -1,
            "in_conversation_chain": true,
            "in_ui_chain": true,
            "lifecycle_status": "active",
            "redirect_to_id": null,
            "role": "system",
            "conversation_id": conversation_id,
            "user_id": user_id,
        });
        let mentions = json!([{
            "spans": [{
                "collection_page_url": "conversation/_conv:_dummy",
                "document_page_url": "conversation/_conv:_dummy",
                "doc_id": "_conv:_dummy",
                "insertion_method": "system",
                "page_number": 1,
                "start_char": 0,
                "end_char": 1,
                "excerpt": "",
                "context_before": "",
                "context_after": "",
                "source_cluster_id": null,
                "verification": {"method": "system", "is_verified": true, "score": 1.0, "notes": ""}
            }],
            "multimodal_spans": []
        }]);
        let payload = json!({
            "id": start_id,
            "label": "conversation start",
            "type": "entity",
            "summary": "Start of conversation",
            "domain_id": null,
            "canonical_entity_id": null,
            "properties": {"status": "active"},
            "mentions": mentions,
            "embedding": null,
            "doc_id": null,
            "metadata": metadata,
            "level_from_root": 0,
            "role": "system",
            "turn_index": -1,
            "conversation_id": conversation_id,
            "user_id": user_id,
        });
        let envelope = kogwistar_contracts::GraphModelEnvelope::new(
            kogwistar_contracts::GraphModelClass::ConversationNode,
            payload.clone(),
        );
        if let Err(error) = envelope.validate_family_v1() {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_GRAPH_MODEL",
                error.to_string(),
            );
        }
        let document = {
            let mut document_value = payload.clone();
            if let Some(object) = document_value.as_object_mut() {
                object.remove("metadata");
                object.remove("embedding");
            }
            match serde_json::to_string(&document_value) {
                Ok(document) => document,
                Err(error) => {
                    return effect_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "KOGWISTAR_SERIALIZATION_FAILED",
                        error.to_string(),
                    );
                }
            }
        };
        let mut record_metadata = metadata.as_object().cloned().unwrap_or_default();
        record_metadata.insert("namespace".to_owned(), json!(scope.namespace));
        record_metadata.insert("_class_name".to_owned(), json!("ConversationNode"));
        record_metadata.insert("label".to_owned(), json!("conversation start"));
        record_metadata.insert("type".to_owned(), json!("entity"));
        record_metadata.insert("summary".to_owned(), json!("Start of conversation"));
        record_metadata.insert("properties".to_owned(), json!("{\"status\":\"active\"}"));
        record_metadata.insert("mentions".to_owned(), json!(mentions.to_string()));
        if let Some(graph_space) = &scope.graph_space {
            record_metadata.insert("graph_space".to_owned(), json!(graph_space));
        }
        if let Some(workspace_id) = &scope.workspace_id {
            record_metadata.insert("workspace_id".to_owned(), json!(workspace_id));
        }
        let event_id = kogwistar_contracts::stable_id(
            "conversation.create.event",
            &[conversation_id.clone(), start_id.clone()],
        )
        .to_string();
        let mutation = GraphMutation {
            scope: scope.clone(),
            table: "gke_nodes".to_owned(),
            entity_kind: "node".to_owned(),
            event_id: event_id.clone(),
            op: "ADD".to_owned(),
            payload: payload.clone(),
            record: GraphRecord {
                id: start_id.clone(),
                document: Some(document),
                metadata: record_metadata,
                embedding: None,
            },
            embedding_dim: 0,
        };
        let owner_acl_id = user_id.to_ascii_lowercase();
        let acl_scope = scope_values["security_scope"]
            .as_str()
            .filter(|value| !value.trim().is_empty())
            .map(str::to_ascii_lowercase);
        let response_conversation_id = conversation_id.clone();
        let response_start_id = start_id.clone();
        let transaction = self.store.immediate_transaction(|uow| {
            if !uow
                .latest_entity_events_by_metadata(
                    &scope.namespace,
                    "node",
                    "conversation_id",
                    &conversation_id,
                )?
                .is_empty()
            {
                return Err(SqliteStoreError::TransactionAborted(
                    "conversation_id already exists".to_owned(),
                ));
            }
            if uow
                .latest_entity_events(&scope.namespace, "node", std::slice::from_ref(&start_id))?
                .contains_key(&start_id)
            {
                return Err(SqliteStoreError::TransactionAborted(
                    "start_node_id already exists".to_owned(),
                ));
            }
            let targets = acl_targets_for_graph_mutations(std::slice::from_ref(&mutation)).0;
            let target_ids = acl_target_entity_ids(&targets);
            let existing =
                uow.graph_acl_records_for_entities(&scope, "conversation", &target_ids)?;
            let (acl_mutations, mut index_jobs) = derive_default_acl_graph_effects_from_records(
                &scope,
                std::slice::from_ref(&mutation),
                targets,
                existing,
                "conversation",
                &owner_acl_id,
                acl_scope.as_deref(),
                &event_id,
            )
            .map_err(SqliteStoreError::TransactionAborted)?;
            index_jobs.push(NewIndexJob {
                job_id: kogwistar_contracts::stable_id(
                    "conversation.create.index_job",
                    &[event_id.clone(), start_id.clone()],
                )
                .to_string(),
                namespace: scope.namespace.clone(),
                entity_kind: "node".to_owned(),
                entity_id: start_id.clone(),
                index_kind: "node_embedding".to_owned(),
                op: "UPSERT".to_owned(),
                payload_json: None,
                max_retries: 10,
            });
            let mut mutations = vec![mutation.clone()];
            mutations.extend(acl_mutations);
            uow.apply_canonical_graph_mutation_batch(GraphMutationBatch {
                mutations,
                index_jobs,
            })?;
            Ok(())
        });
        match transaction {
            Ok(()) => syscall_result(
                &input.version,
                "create_conversation",
                json_effect(
                    StatusCode::CREATED,
                    json!({
                        "conversation_id": response_conversation_id,
                        "user_id": user_id,
                        "status": "active",
                        "start_node_id": response_start_id,
                        "tail_node_id": response_start_id,
                        "turn_count": 0,
                    }),
                ),
            ),
            Err(error) => effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_CONVERSATION_CREATE_CONFLICT",
                error.to_string(),
            ),
        }
    }

    async fn sqlite_chroma_send_message_effect(
        &self,
        request: &ApiEffectRequest,
        input: &SyscallInput,
    ) -> ApiEffectResponse {
        if !syscall_allowed_in_namespace(
            request,
            &self.capabilities,
            "send_message",
            true,
            &["send_message"],
            "conversation",
        ) {
            return syscall_forbidden("send_message");
        }
        let conversation_id = input
            .args
            .get("conversation_id")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let Some(conversation_id) = conversation_id else {
            return effect_error(
                StatusCode::BAD_REQUEST,
                "KOGWISTAR_INVALID_REQUEST",
                "conversation_id is required",
            );
        };
        let text = input
            .args
            .get("text")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let Some(text) = text else {
            return effect_error(
                StatusCode::BAD_REQUEST,
                "KOGWISTAR_INVALID_REQUEST",
                "text must be non-empty",
            );
        };
        let workflow_id = input
            .args
            .get("workflow_id")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or("agentic_answering.v2");
        let requested_user_id = match input.args.get("user_id") {
            None | Some(Value::Null) => None,
            Some(Value::String(value)) if value.trim().is_empty() => None,
            Some(Value::String(value)) => Some(value.trim()),
            Some(_) => {
                return effect_error(
                    StatusCode::BAD_REQUEST,
                    "KOGWISTAR_INVALID_REQUEST",
                    "user_id must be a string when provided",
                );
            }
        };
        let (scope, acl_principal) = conversation_graph_scope_and_acl_principal(&request.principal);
        if scope.namespace != "conversation" {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_SCOPE_FORBIDDEN",
                "conversation message admission requires conversation storage scope",
            );
        }
        let graph_snapshot = match self.chroma_graph_snapshot(request, "conversation").await {
            Ok(records) => records,
            Err(response) => return response,
        };
        // Keep the typed projection alive while selecting this conversation;
        // it carries the ACL index derived from the complete snapshot.
        let conversation_nodes = graph_snapshot
            .nodes()
            .iter()
            .filter(|record| {
                conversation_record_string(record, "conversation_id").as_deref()
                    == Some(conversation_id)
                    && !matches!(
                        conversation_record_string(record, "lifecycle_status").as_deref(),
                        Some("tombstoned" | "deleted")
                    )
            })
            .collect::<Vec<_>>();
        let starts = conversation_nodes
            .iter()
            .filter(|record| {
                conversation_record_string(record, "entity_type").as_deref()
                    == Some("conversation_start")
            })
            .collect::<Vec<_>>();
        let Some(start) = starts.first() else {
            return effect_error(
                StatusCode::NOT_FOUND,
                "KOGWISTAR_CONVERSATION_NOT_FOUND",
                format!("Unknown conversation_id: {conversation_id}"),
            );
        };
        if starts.len() != 1 {
            return effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_CONVERSATION_INVALID",
                "conversation must have exactly one visible conversation_start node",
            );
        }
        let Some(owner_id) =
            conversation_record_string(start, "user_id").filter(|value| !value.trim().is_empty())
        else {
            return effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_CONVERSATION_INVALID",
                "conversation owner is missing",
            );
        };
        let actor_id = authenticated_run_owner_id(&request.principal);
        if requested_user_id.or(actor_id) != Some(owner_id.as_str()) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_RUN_OWNER_FORBIDDEN",
                "user_id must match the owner of the target conversation",
            );
        }
        let Some(auth_store) = self.owner_auth_store.as_ref() else {
            return effect_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "KOGWISTAR_OWNER_REVALIDATION_UNAVAILABLE",
                "conversation owner revalidation is not configured",
            );
        };
        let owner_user = match auth_store.auth_user(&owner_id).await {
            Ok(Some(user)) if user.is_active => user,
            Ok(_) => {
                return effect_error(
                    StatusCode::FORBIDDEN,
                    "KOGWISTAR_RUN_OWNER_FORBIDDEN",
                    "conversation owner is missing or inactive",
                );
            }
            Err(error) => {
                return effect_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "KOGWISTAR_OWNER_REVALIDATION_UNAVAILABLE",
                    error,
                );
            }
        };
        let mut owner_principal = runtime_principal_from_auth_user(&owner_user);
        if !principal_strings(&owner_principal, "ns")
            .iter()
            .any(|namespace| namespace == &scope.namespace || namespace == "*")
        {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_SCOPE_FORBIDDEN",
                "conversation owner is not authorized for the conversation namespace",
            );
        }
        owner_principal["ns"] = json!([scope.namespace]);
        owner_principal["storage_ns"] = json!(scope.namespace);
        if let Some(workspace_id) = &scope.workspace_id {
            owner_principal["workspace"] = json!(workspace_id);
        }
        if let Some(graph_space) = &scope.graph_space {
            owner_principal["graph_space"] = json!(graph_space);
        }
        if let Some(security_scope) = &acl_principal.security_scope {
            owner_principal["security_scope"] = json!(security_scope);
        }
        let Some(owner_binding) = runtime_owner_binding_v1(&owner_principal, &self.capabilities)
        else {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_RUN_OWNER_FORBIDDEN",
                "conversation owner has no authenticated run identity",
            );
        };
        owner_principal = match revalidated_runtime_owner_principal(
            &owner_binding,
            &owner_user,
            &self.capabilities,
            "write_graph",
        ) {
            Ok(principal) => principal,
            Err(error) => {
                return effect_error(
                    StatusCode::FORBIDDEN,
                    "KOGWISTAR_RUN_OWNER_FORBIDDEN",
                    error,
                );
            }
        };
        let Some(provider) = self.embedding_provider.as_ref() else {
            return effect_error(
                StatusCode::BAD_GATEWAY,
                "KOGWISTAR_EMBEDDING_FAILED",
                "conversation message admission requires a configured embedding provider",
            );
        };
        let profile = provider.profile();
        if let Err(error) = profile.validate() {
            return effect_error(
                StatusCode::BAD_GATEWAY,
                "KOGWISTAR_EMBEDDING_FAILED",
                error.to_string(),
            );
        }
        let turn_id = uuid::Uuid::new_v4().to_string();
        let run_id = uuid::Uuid::new_v4().to_string();
        let mut tail = conversation_nodes
            .iter()
            .filter(|record| {
                matches!(
                    conversation_record_string(record, "entity_type").as_deref(),
                    Some(
                        "conversation_start"
                            | "conversation_turn"
                            | "conversation_summary"
                            | "assistant_turn"
                    )
                ) && conversation_record_value(record, "in_conversation_chain")
                    .and_then(|value| value.as_bool())
                    .unwrap_or(true)
            })
            .cloned()
            .collect::<Vec<_>>();
        tail.sort_by_key(|record| {
            (
                conversation_record_i64(record, "turn_index").unwrap_or(-1),
                record.id.clone(),
            )
        });
        let previous = tail.last();
        let turn_index = previous
            .and_then(|record| conversation_record_i64(record, "turn_index"))
            .map(|index| index + 1)
            .unwrap_or(0);
        let step_seq = previous
            .and_then(|record| conversation_record_i64(record, "run_step_seq"))
            .unwrap_or(0)
            + 1;
        let node_metadata = json!({
            "entity_type": "conversation_turn",
            "turn_index": turn_index,
            "level_from_root": 0,
            "in_conversation_chain": true,
            "in_ui_chain": true,
            "run_id": conversation_id,
            "run_step_seq": step_seq,
            "attempt_seq": 0,
            "lifecycle_status": "active",
            "redirect_to_id": null,
            "role": "user",
            "conversation_id": conversation_id,
            "user_id": owner_id,
        });
        let node_mentions = json!([{"spans": [conversation_span(conversation_id, &turn_id, text)], "multimodal_spans": []}]);
        let node_payload = json!({
            "id": turn_id,
            "label": format!("Turn {turn_index} (user)"),
            "type": "entity",
            "summary": text,
            "domain_id": null,
            "canonical_entity_id": null,
            "properties": {},
            "mentions": node_mentions,
            "doc_id": turn_id,
            "metadata": node_metadata,
            "level_from_root": 0,
            "role": "user",
            "turn_index": turn_index,
            "conversation_id": conversation_id,
            "user_id": owner_id,
        });
        let node_envelope = kogwistar_contracts::GraphModelEnvelope::new(
            kogwistar_contracts::GraphModelClass::ConversationNode,
            node_payload.clone(),
        );
        if let Err(error) = node_envelope.validate_family_v1() {
            return effect_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "KOGWISTAR_CONVERSATION_MODEL_INVALID",
                error.to_string(),
            );
        }
        let mut node_document_payload = node_payload.clone();
        node_document_payload
            .as_object_mut()
            .expect("conversation turn payload is an object")
            .remove("metadata");
        let node_document = serde_json::to_string(&node_document_payload)
            .expect("conversation turn document serializes");
        let mut node_projection = node_metadata.as_object().unwrap().clone();
        node_projection.extend([
            ("doc_id".to_owned(), json!(turn_id)),
            (
                "label".to_owned(),
                json!(format!("Turn {turn_index} (user)")),
            ),
            ("type".to_owned(), json!("entity")),
            ("summary".to_owned(), json!(text)),
            ("properties".to_owned(), json!("{}")),
            ("mentions".to_owned(), json!(node_mentions.to_string())),
        ]);
        if let Some(graph_space) = &scope.graph_space {
            node_projection.insert("graph_space".to_owned(), json!(graph_space));
        }
        if let Some(workspace_id) = &scope.workspace_id {
            node_projection.insert("workspace_id".to_owned(), json!(workspace_id));
        }
        let mut mutations = vec![(
            GraphMutation {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                entity_kind: "node".to_owned(),
                event_id: uuid::Uuid::new_v4().to_string(),
                op: "ADD".to_owned(),
                payload: node_payload,
                record: GraphRecord {
                    id: turn_id.clone(),
                    document: Some(node_document),
                    metadata: node_projection,
                    embedding: None,
                },
                embedding_dim: profile.dimension,
            },
            node_envelope,
        )];
        let mut index_jobs = conversation_index_job_kinds("node")
            .expect("node conversation indexing contract exists")
            .iter()
            .map(|kind| conversation_index_job(&scope.namespace, "node", &turn_id, kind))
            .collect::<Vec<_>>();
        let mut edge_payload_and_document = None;
        if let Some(previous) = previous {
            let previous_id = previous.id.clone();
            let (char_distance, turn_distance, tail_turn_index) =
                conversation_next_turn_distances(previous, text, turn_index);
            let edge_id = kogwistar_contracts::stable_id(
                "conversation.edge",
                &[
                    owner_id.clone(),
                    conversation_id.to_owned(),
                    "next_turn".to_owned(),
                    (turn_index + 1).to_string(),
                    format!("['{previous_id}']"),
                    format!("['{turn_id}']"),
                    "[]".to_owned(),
                    "[]".to_owned(),
                    "conversation_edge".to_owned(),
                ],
            )
            .to_string();
            let edge_payload = json!({
                "id": edge_id,
                "source_ids": [previous_id],
                "target_ids": [turn_id],
                "source_edge_ids": [],
                "target_edge_ids": [],
                "relation": "next_turn",
                "label": "next_turn",
                "type": "relationship",
                "summary": "Sequential flow",
                "doc_id": format!("conv:{conversation_id}"),
                "mentions": node_mentions,
                "domain_id": null,
                "canonical_entity_id": null,
                "properties": {"entity_type": "conversation_edge"},
                "metadata": {
                    "relation": "next_turn",
                    "target_id": turn_id,
                    "char_distance_from_last_summary": char_distance,
                    "turn_distance_from_last_summary": turn_distance,
                    "tail_turn_index": tail_turn_index,
                    "causal_type": "chain",
                    "run_id": conversation_id,
                    "run_step_seq": step_seq,
                    "attempt_seq": 0,
                    "lifecycle_status": "active",
                    "redirect_to_id": null,
                },
            });
            let edge_envelope = kogwistar_contracts::GraphModelEnvelope::new(
                kogwistar_contracts::GraphModelClass::ConversationEdge,
                edge_payload.clone(),
            );
            if let Err(error) = edge_envelope.validate_family_v1() {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_CONVERSATION_MODEL_INVALID",
                    error.to_string(),
                );
            }
            let edge_document = serde_json::to_string(&edge_payload)
                .expect("conversation edge document serializes");
            edge_payload_and_document = Some((edge_payload, edge_document, edge_envelope));
        }
        let embedding_texts = edge_payload_and_document
            .as_ref()
            .map(|(_, document, _)| vec![format!("user: {text}"), document.clone()])
            .unwrap_or_else(|| vec![format!("user: {text}")]);
        let embeddings = match provider.embed(&embedding_texts).await {
            Ok(vectors)
                if vectors.len() == embedding_texts.len()
                    && vectors.iter().all(|vector| {
                        vector.len() == profile.dimension && vector.iter().all(|v| v.is_finite())
                    }) =>
            {
                vectors
            }
            Ok(_) => {
                return effect_error(
                    StatusCode::BAD_GATEWAY,
                    "KOGWISTAR_EMBEDDING_FAILED",
                    "embedding provider returned an invalid conversation vector",
                );
            }
            Err(error) => {
                return effect_error(
                    StatusCode::BAD_GATEWAY,
                    "KOGWISTAR_EMBEDDING_FAILED",
                    error.to_string(),
                );
            }
        };
        mutations[0].0.record.embedding =
            Some(embeddings[0].iter().map(|value| *value as f32).collect());
        if let Some((edge_payload, edge_document, edge_envelope)) = edge_payload_and_document {
            let edge_id = edge_payload["id"].as_str().unwrap().to_owned();
            let mut metadata = edge_payload["metadata"].as_object().unwrap().clone();
            metadata.extend([
                ("doc_id".to_owned(), edge_payload["doc_id"].clone()),
                ("relation".to_owned(), json!("next_turn")),
                (
                    "source_ids".to_owned(),
                    json!(edge_payload["source_ids"].to_string()),
                ),
                (
                    "target_ids".to_owned(),
                    json!(edge_payload["target_ids"].to_string()),
                ),
                ("source_edge_ids".to_owned(), json!("[]")),
                ("target_edge_ids".to_owned(), json!("[]")),
                ("type".to_owned(), json!("relationship")),
                ("summary".to_owned(), json!("Sequential flow")),
                (
                    "properties".to_owned(),
                    json!(edge_payload["properties"].to_string()),
                ),
                (
                    "mentions".to_owned(),
                    json!(edge_payload["mentions"].to_string()),
                ),
                ("node_endpoint_count".to_owned(), json!(2)),
                ("edge_endpoint_count".to_owned(), json!(0)),
                ("total_endpoint_count".to_owned(), json!(2)),
            ]);
            if let Some(graph_space) = &scope.graph_space {
                metadata.insert("graph_space".to_owned(), json!(graph_space));
            }
            if let Some(workspace_id) = &scope.workspace_id {
                metadata.insert("workspace_id".to_owned(), json!(workspace_id));
            }
            mutations.push((
                GraphMutation {
                    scope: scope.clone(),
                    table: "gke_edges".to_owned(),
                    entity_kind: "edge".to_owned(),
                    event_id: uuid::Uuid::new_v4().to_string(),
                    op: "ADD".to_owned(),
                    payload: edge_payload,
                    record: GraphRecord {
                        id: edge_id.clone(),
                        document: Some(edge_document),
                        metadata,
                        embedding: Some(embeddings[1].iter().map(|value| *value as f32).collect()),
                    },
                    embedding_dim: profile.dimension,
                },
                edge_envelope,
            ));
            for kind in conversation_index_job_kinds("edge")
                .expect("edge conversation indexing contract exists")
            {
                index_jobs.push(conversation_index_job(
                    &scope.namespace,
                    "edge",
                    &edge_id,
                    kind,
                ));
            }
        }
        let prepared = PreparedConversationAdmission {
            conversation_id: conversation_id.to_owned(),
            owner_id: owner_id.clone(),
            expected_tail_id: previous.map(|record| record.id.clone()),
            scope: scope.clone(),
            acl_principal,
            mutations,
            index_jobs,
            actor_user_id: actor_id.unwrap_or(&owner_id).to_owned(),
        };
        let initial_state = serde_json::Map::from_iter([
            ("conversation_id".to_owned(), json!(conversation_id)),
            ("user_text".to_owned(), json!(text)),
            ("turn_node_id".to_owned(), json!(turn_id)),
            ("turn_index".to_owned(), json!(turn_index)),
            ("user_id".to_owned(), json!(owner_id)),
            ("max_retrieval_level".to_owned(), json!(4)),
        ]);
        let nested = ApiEffectRequest {
            contract_version: request.contract_version,
            method: "POST".to_owned(),
            path_and_query: "/api/workflow/runs".to_owned(),
            body: serde_json::to_vec(&json!({
                "run_id": run_id,
                "workflow_id": workflow_id,
                "conversation_id": conversation_id,
                "turn_node_id": turn_id,
                "user_id": owner_id,
                "initial_state": initial_state,
            }))
            .expect("conversation message workflow request serializes"),
            principal: owner_principal,
        };
        let response = self.submit_run_with_conversation_admission(&nested, Some(prepared));
        syscall_result(&input.version, "send_message", response)
    }

}
