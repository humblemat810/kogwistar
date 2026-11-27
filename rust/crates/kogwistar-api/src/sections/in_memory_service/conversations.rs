use super::*;
use serde_json::json;

impl InMemoryGraphApplicationService {
    // PYTHON_PARITY_V1: Preserve Python conversation read scope and visibility projections. rust=rust/crates/kogwistar-api/src/sections/in_memory_service/conversations.rs#in_memory_conversation_read_effect; python=kogwistar/server/chat_service_conversation_queries.py#_ConversationQueryService; fixture=contracts/golden/conversation-list-http-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/in_memory_conversations.rs#conversation_read_acl_scope_matrix_matches_python; python_test=kogwistar/tests/server/test_conversation_graph_scope.py#test_conversation_read_acl_scope_matrix
    pub(crate) async fn in_memory_conversation_read_effect(
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

        let is_list = is_conversation_list_route(&request.path_and_query);
        let transcript_id = conversation_transcript_id(&request.path_and_query);
        let is_transcript = transcript_id.is_some();
        let conversation_id = if is_list {
            None
        } else if let Some(conversation_id) = transcript_id {
            Some(conversation_id)
        } else {
            conversation_detail_id(&request.path_and_query)
        };
        if !is_list && conversation_id.is_none() {
            return effect_error(
                StatusCode::NOT_FOUND,
                "KOGWISTAR_ROUTE_NOT_FOUND",
                "Conversation route was not found",
            );
        }
        let user_id = request
            .principal
            .get("user_id")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty());
        if is_list && user_id.is_none() {
            return effect_error(
                StatusCode::UNAUTHORIZED,
                "KOGWISTAR_USER_IDENTITY_REQUIRED",
                "User Identity required",
            );
        }

        let (scope, acl_principal) = conversation_graph_scope_and_acl_principal(&request.principal);
        let workspace_id = scope.workspace_id.clone();
        let graph_space = scope.graph_space.clone();
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
        let (mut visible_nodes, _) =
            graph_records_visible_by_acl(&nodes, &edges, "conversation", &acl_principal);
        visible_nodes.retain(|record| {
            (workspace_id.is_none()
                || record.metadata.get("workspace_id").and_then(Value::as_str)
                    == workspace_id.as_deref())
                && (graph_space.is_none()
                    || record.metadata.get("graph_space").and_then(Value::as_str)
                        == graph_space.as_deref())
        });

        if is_list {
            let user_id = user_id.expect("list identity validated above");
            let candidates = compact_conversation_list_candidates(&visible_nodes);
            let mut accumulator = ConversationListAccumulator::new(user_id);
            accumulator.extend_visible_records(&candidates);
            let conversations = accumulator
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
            return json_effect(StatusCode::OK, json!({"conversations": conversations}));
        }

        let conversation_id = conversation_id.expect("non-list route parsed above");
        let exists = visible_nodes.iter().any(|record| {
            conversation_record_string(record, "conversation_id").as_deref()
                == Some(conversation_id.as_str())
                && !matches!(
                    conversation_record_string(record, "lifecycle_status").as_deref(),
                    Some("tombstoned" | "deleted")
                )
        });
        if !exists {
            return effect_error(
                StatusCode::NOT_FOUND,
                "KOGWISTAR_CONVERSATION_NOT_FOUND",
                format!("Unknown conversation_id: {conversation_id}"),
            );
        }

        if is_transcript {
            let transcript = match conversation_transcript_from_visible_records(
                &visible_nodes,
                &conversation_id,
                usize::MAX,
            ) {
                Ok(transcript) => transcript,
                Err(error) => {
                    return effect_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "KOGWISTAR_INVALID_TRANSCRIPT",
                        error.to_string(),
                    );
                }
            };
            let mut payload = serde_json::to_value(transcript).expect("transcript serializes");
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
            return json_effect(StatusCode::OK, payload);
        }

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
        let mut payload = serde_json::to_value(summary).expect("conversation summary serializes");
        if let Some(object) = payload.as_object_mut() {
            object.remove("schema_version");
        }
        json_effect(StatusCode::OK, payload)
    }

    // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/in_memory_service/conversations.rs#in_memory_mount_memory_effect; python=kogwistar/server/syscall_api.py#mount_memory; fixture=contracts/golden/mount-memory-result-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/in_memory_conversations.rs#in_memory_mount_memory_matches_shared_contract_and_authority; python_test=tests/server/test_mount_memory_wire_contract.py#test_python_mount_memory_result_matches_shared_wire_fixture
    pub(crate) async fn in_memory_mount_memory_effect(
        &self,
        request: &ApiEffectRequest,
        input: &SyscallInput,
    ) -> ApiEffectResponse {
        if !syscall_allowed_in_namespace(
            request,
            &self.capabilities,
            "mount_memory",
            false,
            &["project_view"],
            "conversation",
        ) {
            return syscall_forbidden("mount_memory");
        }
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
                "conversation_id is required",
            );
        };
        let run_id = match context_snapshot_syscall_field(&input.args, "run_id") {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error,
                );
            }
        };
        let stage = match context_snapshot_syscall_field(&input.args, "stage") {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error,
                );
            }
        };
        let (scope, acl_principal) = conversation_graph_scope_and_acl_principal(&request.principal);
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
        let (visible_nodes, _) =
            graph_records_visible_by_acl(&nodes, &edges, "conversation", &acl_principal);
        let snapshot = match kogwistar_store::context_snapshot_from_visible_records(
            &visible_nodes,
            conversation_id,
            run_id.as_deref(),
            stage.as_deref(),
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
        syscall_result(
            &input.version,
            "mount_memory",
            json_effect(StatusCode::OK, public_context_snapshot_value(snapshot)),
        )
    }

    // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/in_memory_service/conversations.rs#in_memory_create_conversation_effect; python=kogwistar/server/syscall_api.py#create_syscall_router.dispatch; fixture=contracts/golden/conversation-create-syscall-result-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/in_memory_conversations.rs#in_memory_create_conversation_is_atomic_and_matches_shared_contract; python_test=tests/server/test_syscall_api.py#test_syscall_create_conversation_uses_authenticated_owner
    pub(crate) async fn in_memory_create_conversation_effect(
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
            Ok(Some(id)) => id,
            Ok(None) => uuid::Uuid::new_v4().to_string(),
            Err(response) => return response,
        };
        let requested_start_id = match optional_id("start_node_id") {
            Ok(id) => id,
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
        let mut document_payload = payload.clone();
        if let Some(object) = document_payload.as_object_mut() {
            object.remove("metadata");
            object.remove("embedding");
        }
        let document = match serde_json::to_string(&document_payload) {
            Ok(document) => document,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_SERIALIZATION_FAILED",
                    error.to_string(),
                );
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
            payload,
            record: GraphRecord {
                id: start_id.clone(),
                document: Some(document),
                metadata: record_metadata,
                embedding: None,
            },
            embedding_dim: 0,
        };
        let targets = acl_targets_for_graph_mutations(std::slice::from_ref(&mutation)).0;
        let mut acl_filter = MetadataFilter::default();
        acl_filter
            .equals
            .insert("entity_type".to_owned(), json!("acl_record"));
        acl_filter
            .equals
            .insert("acl_truth_graph".to_owned(), json!("conversation"));
        let existing_acl = match self
            .store
            .graph_projection_records(GraphProjectionRead {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                ids: None,
                metadata: acl_filter,
                limit: 10_000,
            })
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
        let owner_acl_id = user_id.to_ascii_lowercase();
        let acl_scope = scope_values["security_scope"]
            .as_str()
            .filter(|value| !value.trim().is_empty())
            .map(str::to_ascii_lowercase);
        let (mut acl_mutations, mut index_jobs) =
            match derive_default_acl_graph_effects_from_records(
                &scope,
                std::slice::from_ref(&mutation),
                targets,
                existing_acl,
                AclDefaultsContext {
                    truth_graph: "conversation",
                    owner_id: &owner_acl_id,
                    security_scope: acl_scope.as_deref(),
                    effect_id: &event_id,
                },
            ) {
                Ok(effects) => effects,
                Err(error) => {
                    return effect_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "KOGWISTAR_ACL_EFFECT_FAILED",
                        error,
                    );
                }
            };
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
        let mut mutations = vec![mutation];
        mutations.append(&mut acl_mutations);
        let mut conversation_filter = MetadataFilter::default();
        conversation_filter
            .equals
            .insert("entity_type".to_owned(), json!("conversation_start"));
        conversation_filter
            .equals
            .insert("conversation_id".to_owned(), json!(conversation_id));
        let batch = GraphMutationBatch {
            mutations,
            index_jobs,
        };
        let commit = self
            .store
            .apply_graph_mutations_with_index_jobs_if_absent(
                batch,
                &[(scope.clone(), "gke_nodes".to_owned(), start_id.clone())],
                &[GraphProjectionRead {
                    scope,
                    table: "gke_nodes".to_owned(),
                    ids: None,
                    metadata: conversation_filter,
                    limit: 1,
                }],
            )
            .await;
        match commit {
            Ok(_) => syscall_result(
                &input.version,
                "create_conversation",
                json_effect(
                    StatusCode::CREATED,
                    json!({
                        "conversation_id": conversation_id,
                        "user_id": user_id,
                        "status": "active",
                        "start_node_id": start_id,
                        "tail_node_id": start_id,
                        "turn_count": 0,
                    }),
                ),
            ),
            Err(StoreError::Backend { backend, message })
                if backend == "memory-graph-write" && message.starts_with("unique graph") =>
            {
                effect_error(
                    StatusCode::CONFLICT,
                    "KOGWISTAR_CONVERSATION_CREATE_CONFLICT",
                    message,
                )
            }
            Err(error) => effect_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "KOGWISTAR_STORE_ERROR",
                error.to_string(),
            ),
        }
    }

    // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/in_memory_service/conversations.rs#in_memory_send_message_effect; python=kogwistar/server/chat_service_run_execution.py#submit_turn_for_answer; fixture=contracts/golden/answer-workflow-runtime-contract-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/in_memory_conversations.rs#in_memory_send_message_admits_turn_and_run_atomically; python_test=tests/server/test_capability_kernel_contract.py#test_submit_turn_for_answer_replays_idempotency_key_and_rejects_payload_change
    pub(crate) async fn in_memory_send_message_effect(
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
                "conversation_id is required",
            );
        };
        let Some(text) = input
            .args
            .get("text")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        else {
            return json_effect(
                StatusCode::BAD_REQUEST,
                json!({"detail":"text must be non-empty"}),
            );
        };
        let workflow_id = input
            .args
            .get("workflow_id")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or("agentic_answering.v2");
        let idempotency_key = match conversation_idempotency_key(&input.args) {
            Ok(key) => key,
            Err(error) => {
                return json_effect(StatusCode::BAD_REQUEST, json!({"detail":error}));
            }
        };
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
        let actor_id = authenticated_run_owner_id(&request.principal);
        let (scope, acl_principal) = conversation_graph_scope_and_acl_principal(&request.principal);
        if scope.namespace != "conversation" {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_SCOPE_FORBIDDEN",
                "conversation message admission requires conversation storage scope",
            );
        }

        let mut conversation_filter = MetadataFilter::default();
        conversation_filter
            .equals
            .insert("conversation_id".to_owned(), json!(conversation_id));
        let conversation_read = GraphProjectionRead {
            scope: scope.clone(),
            table: "gke_nodes".to_owned(),
            ids: None,
            metadata: conversation_filter,
            limit: usize::MAX,
        };
        let conversation_nodes = match self
            .store
            .graph_projection_records(conversation_read.clone())
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
        let mut acl_filter = MetadataFilter::default();
        acl_filter
            .equals
            .insert("entity_type".to_owned(), json!("acl_record"));
        acl_filter
            .equals
            .insert("acl_truth_graph".to_owned(), json!("conversation"));
        let acl_read = GraphProjectionRead {
            scope: scope.clone(),
            table: "gke_nodes".to_owned(),
            ids: None,
            metadata: acl_filter,
            limit: usize::MAX,
        };
        let acl_records = match self.store.graph_projection_records(acl_read.clone()).await {
            Ok(records) => records,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error.to_string(),
                );
            }
        };
        let conversation_snapshot = conversation_nodes.clone();
        let acl_snapshot = acl_records.clone();
        let mut acl_input = conversation_nodes.clone();
        acl_input.extend(acl_records.clone());
        let (visible_nodes, _) =
            graph_records_visible_by_acl(&acl_input, &[], "conversation", &acl_principal);
        let conversation_nodes = visible_nodes
            .into_iter()
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
        if requested_user_id.or(actor_id) != Some(owner_id.as_str()) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_RUN_OWNER_FORBIDDEN",
                "user_id must match the owner of the target conversation",
            );
        }

        let request_fingerprint = idempotency_key.as_deref().map(|_| {
            conversation_send_message_fingerprint(&owner_id, conversation_id, workflow_id, text)
        });
        let turn_id = idempotency_key
            .as_deref()
            .map(|key| {
                conversation_send_message_id(
                    "conversation.send_message.turn",
                    &owner_id,
                    conversation_id,
                    key,
                )
            })
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let run_id = idempotency_key
            .as_deref()
            .map(|key| {
                conversation_send_message_id(
                    "conversation.send_message.run",
                    &owner_id,
                    conversation_id,
                    key,
                )
            })
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        if let Some(fingerprint) = request_fingerprint.as_deref() {
            match kogwistar_store::ServerRunReadStore::server_run(&self.store, &run_id).await {
                Ok(Some(existing)) => {
                    let events = match kogwistar_store::ServerRunReadStore::server_run_events(
                        &self.store,
                        &run_id,
                        0,
                        100,
                    )
                    .await
                    {
                        Ok(events) => events,
                        Err(error) => {
                            return effect_error(
                                StatusCode::SERVICE_UNAVAILABLE,
                                "KOGWISTAR_STORAGE_UNAVAILABLE",
                                error.to_string(),
                            );
                        }
                    };
                    return match conversation_idempotency_replay_result(
                        &existing,
                        &events,
                        ConversationIdempotencyExpectation {
                            run_id: &run_id,
                            conversation_id,
                            owner_id: &owner_id,
                            workflow_id,
                            turn_id: &turn_id,
                            request_fingerprint: fingerprint,
                        },
                    ) {
                        Ok(result) => syscall_result(
                            &input.version,
                            "send_message",
                            json_effect(StatusCode::OK, result),
                        ),
                        Err(error) => effect_error(
                            StatusCode::CONFLICT,
                            "KOGWISTAR_IDEMPOTENCY_CONFLICT",
                            error,
                        ),
                    };
                }
                Ok(None) => {}
                Err(error) => {
                    return effect_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "KOGWISTAR_STORAGE_UNAVAILABLE",
                        error.to_string(),
                    );
                }
            }
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
        let Some(previous) = tail.last() else {
            return effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_CONVERSATION_INVALID",
                "conversation has no visible tail node",
            );
        };
        let turn_index = conversation_record_i64(previous, "turn_index")
            .map(|index| index + 1)
            .unwrap_or(0);
        let step_seq = conversation_record_i64(previous, "run_step_seq").unwrap_or(0) + 1;
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
        let mentions = json!([{
            "spans": [conversation_span(conversation_id, &turn_id, text)],
            "multimodal_spans": []
        }]);
        let node_payload = json!({
            "id": turn_id,
            "label": format!("Turn {turn_index} (user)"),
            "type": "entity",
            "summary": text,
            "domain_id": null,
            "canonical_entity_id": null,
            "properties": {},
            "mentions": mentions,
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
            ("mentions".to_owned(), json!(mentions.to_string())),
        ]);
        if let Some(graph_space) = &scope.graph_space {
            node_projection.insert("graph_space".to_owned(), json!(graph_space));
        }
        if let Some(workspace_id) = &scope.workspace_id {
            node_projection.insert("workspace_id".to_owned(), json!(workspace_id));
        }
        let node_event_id = conversation_send_message_id(
            "conversation.send_message.node_event",
            &owner_id,
            conversation_id,
            &turn_id,
        );
        let mut mutations = vec![(
            GraphMutation {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                entity_kind: "node".to_owned(),
                event_id: node_event_id,
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
            .expect("conversation node index contract exists")
            .iter()
            .map(|kind| conversation_index_job(&scope.namespace, "node", &turn_id, kind))
            .collect::<Vec<_>>();
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
            "mentions": mentions,
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
        let edge_document =
            serde_json::to_string(&edge_payload).expect("conversation edge document serializes");
        let embedding_texts = vec![format!("user: {text}"), edge_document.clone()];
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
        mutations[0].0.record.embedding = Some(embeddings[0].iter().map(|v| *v as f32).collect());
        let mut edge_metadata = edge_payload["metadata"].as_object().unwrap().clone();
        edge_metadata.extend([
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
            ("mentions".to_owned(), json!(mentions.to_string())),
            ("node_endpoint_count".to_owned(), json!(2)),
            ("edge_endpoint_count".to_owned(), json!(0)),
            ("total_endpoint_count".to_owned(), json!(2)),
        ]);
        if let Some(graph_space) = &scope.graph_space {
            edge_metadata.insert("graph_space".to_owned(), json!(graph_space));
        }
        if let Some(workspace_id) = &scope.workspace_id {
            edge_metadata.insert("workspace_id".to_owned(), json!(workspace_id));
        }
        let edge_event_id = conversation_send_message_id(
            "conversation.send_message.edge_event",
            &owner_id,
            conversation_id,
            &turn_id,
        );
        mutations.push((
            GraphMutation {
                scope: scope.clone(),
                table: "gke_edges".to_owned(),
                entity_kind: "edge".to_owned(),
                event_id: edge_event_id,
                op: "ADD".to_owned(),
                payload: edge_payload,
                record: GraphRecord {
                    id: edge_id.clone(),
                    document: Some(edge_document),
                    metadata: edge_metadata,
                    embedding: Some(embeddings[1].iter().map(|v| *v as f32).collect()),
                },
                embedding_dim: profile.dimension,
            },
            edge_envelope,
        ));
        for kind in
            conversation_index_job_kinds("edge").expect("conversation edge index contract exists")
        {
            index_jobs.push(conversation_index_job(
                &scope.namespace,
                "edge",
                &edge_id,
                kind,
            ));
        }
        let graph_mutations = mutations
            .iter()
            .map(|(mutation, _)| mutation.clone())
            .collect::<Vec<_>>();
        let acl_targets = acl_targets_for_graph_mutations(&graph_mutations).0;
        let (acl_mutations, acl_jobs) = match derive_default_acl_graph_effects_from_records(
            &scope,
            &graph_mutations,
            acl_targets,
            acl_records,
            AclDefaultsContext {
                truth_graph: "conversation",
                owner_id: &owner_id.to_ascii_lowercase(),
                security_scope: acl_principal.security_scope.as_deref(),
                effect_id: &run_id,
            },
        ) {
            Ok(effects) => effects,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_ACL_EFFECT_FAILED",
                    error,
                );
            }
        };
        for (mutation, envelope) in &mutations {
            if let Err(error) = validate_typed_graph_mutation(mutation, envelope) {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_GRAPH_MODEL",
                    error.to_string(),
                );
            }
        }
        let mut all_mutations = graph_mutations;
        all_mutations.extend(acl_mutations);
        index_jobs.extend(acl_jobs);
        let graph_batch = GraphMutationBatch {
            mutations: all_mutations,
            index_jobs,
        };
        let expected_projections = vec![
            kogwistar_store_memory::GraphProjectionExpectation {
                read: conversation_read,
                records: conversation_snapshot,
            },
            kogwistar_store_memory::GraphProjectionExpectation {
                read: acl_read,
                records: acl_snapshot,
            },
        ];
        let admission = InMemoryConversationAdmission {
            graph_batch,
            expected_projections,
            request_fingerprint,
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
        let response = self.submit_run_effect(&nested, Some(admission)).await;
        syscall_result(&input.version, "send_message", response)
    }

    // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/in_memory_service/conversations.rs#in_memory_receive_message_effect; python=kogwistar/server/chat_service.py#ChatRunService.list_transcript; fixture=contracts/golden/receive-message-result-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/in_memory_conversations.rs#in_memory_receive_message_matches_shared_python_contract; python_test=tests/server/test_receive_message_wire_contract.py#test_python_receive_message_result_matches_shared_wire_fixture
    pub(crate) async fn in_memory_receive_message_effect(
        &self,
        request: &ApiEffectRequest,
        input: &SyscallInput,
    ) -> ApiEffectResponse {
        if !syscall_allowed_in_namespace(
            request,
            &self.capabilities,
            "receive_message",
            false,
            &["read_graph"],
            "conversation",
        ) {
            return syscall_forbidden("receive_message");
        }
        let Some(conversation_id) = input
            .args
            .get("conversation_id")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        else {
            return json_effect(
                StatusCode::NOT_FOUND,
                json!({"detail": "'conversation_id'"}),
            );
        };
        let security_scope = security_scope_value(&request.principal);
        let scope = GraphScope {
            namespace: security_scope["storage_namespace"]
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
        let acl_principal = graph_acl_principal_from_claims(
            &request.principal,
            security_scope["security_scope"].as_str(),
        );
        let (visible_nodes, _) =
            graph_records_visible_by_acl(&nodes, &edges, "conversation", &acl_principal);
        let conversation_exists = visible_nodes.iter().any(|record| {
            conversation_record_string(record, "conversation_id").as_deref()
                == Some(conversation_id)
                && !matches!(
                    conversation_record_string(record, "lifecycle_status").as_deref(),
                    Some("tombstoned" | "deleted")
                )
        });
        if !conversation_exists {
            return effect_error(
                StatusCode::NOT_FOUND,
                "KOGWISTAR_CONVERSATION_NOT_FOUND",
                format!("Unknown conversation_id: {conversation_id}"),
            );
        }
        let transcript = match conversation_transcript_from_visible_records(
            &visible_nodes,
            conversation_id,
            10_000,
        ) {
            Ok(transcript) => transcript,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_INVALID_TRANSCRIPT",
                    error.to_string(),
                );
            }
        };
        let result = legacy_receive_message_value(transcript);
        syscall_result(
            &input.version,
            "receive_message",
            json_effect(StatusCode::OK, result),
        )
    }

    // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/in_memory_service/conversations.rs#in_memory_invoke_builtin_tool_effect; python=kogwistar/server/syscall_api.py#_invoke_builtin_tool; fixture=contracts/golden/invoke-tool-results-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/in_memory_conversations.rs#in_memory_invoke_builtin_tools_match_shared_contract_and_authority; python_test=tests/server/test_syscall_api.py#test_invoke_builtin_tool_results_match_shared_contract_fixture
    pub(crate) async fn in_memory_invoke_builtin_tool_effect(
        &self,
        request: &ApiEffectRequest,
        input: &SyscallInput,
    ) -> ApiEffectResponse {
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
            return json_effect(
                StatusCode::BAD_REQUEST,
                json!({"detail":"tool_name is required"}),
            );
        };

        let result = match tool_name.as_str() {
            "list_transcript" => {
                let response = self.in_memory_receive_message_effect(request, input).await;
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
                let Some(mut result) = payload.get("result").cloned() else {
                    return effect_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "KOGWISTAR_INVALID_INTERNAL_RESPONSE",
                        "built-in graph tool returned no result",
                    );
                };
                result["tool_name"] = Value::String(tool_name.clone());
                result
            }
            "latest_snapshot" => {
                if !capability_allowed(
                    request,
                    &self.capabilities,
                    "project_view",
                    &["project_view"],
                ) {
                    return effect_error(
                        StatusCode::FORBIDDEN,
                        "KOGWISTAR_CAPABILITY_FORBIDDEN",
                        "built-in tool \"latest_snapshot\" requires project_view",
                    );
                }
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
                        "conversation_id is required for latest_snapshot",
                    );
                };
                let (scope, acl_principal) =
                    conversation_graph_scope_and_acl_principal(&request.principal);
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
                let (visible_nodes, _) =
                    graph_records_visible_by_acl(&nodes, &edges, "conversation", &acl_principal);
                let run_id = input.args.get("run_id").and_then(Value::as_str);
                let stage = input.args.get("stage").and_then(Value::as_str);
                let snapshot = match kogwistar_store::context_snapshot_from_visible_records(
                    &visible_nodes,
                    conversation_id,
                    run_id,
                    stage,
                ) {
                    Ok(snapshot) => snapshot,
                    Err(_) => {
                        return effect_error(
                            StatusCode::NOT_FOUND,
                            "KOGWISTAR_NOT_FOUND",
                            "no authorized context snapshot found",
                        );
                    }
                };
                json!({
                    "tool_name": tool_name,
                    "conversation_id": conversation_id,
                    "snapshot": public_context_snapshot_value(snapshot),
                })
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
                    "read_security_scope",
                    &["read_security_scope", "project_view"],
                ) {
                    return effect_error(
                        StatusCode::FORBIDDEN,
                        "KOGWISTAR_CAPABILITY_FORBIDDEN",
                        "built-in tool \"resource_snapshot\" requires read_security_scope and project_view",
                    );
                }
                let runs = match self.store.server_runs(None, None, None, 10_000).await {
                    Ok(rows) => rows,
                    Err(error) => {
                        return effect_error(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "KOGWISTAR_STORE_ERROR",
                            error.to_string(),
                        );
                    }
                };
                let lanes = match kogwistar_store::LaneMessageReadStore::projected_lane_messages(
                    &self.store,
                    LaneMessageFilter {
                        namespace: Some("workflow".to_owned()),
                        limit: 10_000,
                        ..LaneMessageFilter::default()
                    },
                )
                .await
                {
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
                    match self.store.server_run_events(&run.run_id, 0, 200).await {
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
                let usage = usage_history(&events, 10_000);
                let mut snapshot = resource_value(&runs, &lanes, &usage, &events, 0);
                snapshot
                    .as_object_mut()
                    .expect("resource snapshot is an object")
                    .remove("migration");
                json!({"tool_name": tool_name, "snapshot": snapshot})
            }
            _ => {
                return json_effect(
                    StatusCode::NOT_FOUND,
                    json!({"detail":format!("'unknown tool: {tool_name}'")}),
                );
            }
        };
        syscall_result(
            &input.version,
            "invoke_tool",
            json_effect(StatusCode::OK, result),
        )
    }
}
