// Chroma-backed conversation read projections and transcript/list assembly.
// Python parity: kogwistar/server/chat_service_conversation_queries.py.
impl SqliteRunApplicationService {
    async fn sqlite_chroma_receive_message_effect(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        let input = match syscall_input(request, "receive_message") {
            Ok(input) => input,
            Err(response) => return response,
        };
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
            .filter(|value| !value.trim().is_empty())
        else {
            return json_effect(
                StatusCode::NOT_FOUND,
                json!({"detail": "'conversation_id'"}),
            );
        };
        let graph_snapshot = match self.chroma_graph_snapshot(request, "conversation").await {
            Ok(records) => records,
            Err(response) => return response,
        };
        let nodes = graph_snapshot.nodes();
        let transcript =
            match conversation_transcript_from_visible_records(&nodes, conversation_id, 10_000) {
                Ok(transcript) => transcript,
                Err(error) => {
                    return effect_error(
                        StatusCode::NOT_FOUND,
                        "KOGWISTAR_CONVERSATION_NOT_FOUND",
                        error.to_string(),
                    );
                }
            };
        syscall_result(
            &input.version,
            "receive_message",
            json_effect(StatusCode::OK, legacy_receive_message_value(transcript)),
        )
    }

    async fn sqlite_chroma_mount_memory_effect(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        let input = match syscall_input(request, "mount_memory") {
            Ok(input) => input,
            Err(response) => return response,
        };
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
            .filter(|value| !value.trim().is_empty())
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
        let graph_snapshot = match self.chroma_graph_snapshot(request, "conversation").await {
            Ok(records) => records,
            Err(response) => return response,
        };
        let nodes = graph_snapshot.nodes();
        let snapshot = match kogwistar_store::context_snapshot_from_visible_records(
            &nodes,
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

    async fn sqlite_chroma_conversation_transcript_effect(
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
        if !capability_allowed(request, &self.capabilities, "read_graph", &["read_graph"]) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Conversation transcript reads require read_graph capability",
            );
        }
        let graph_snapshot = match self.chroma_graph_snapshot(request, "conversation").await {
            Ok(records) => records,
            Err(response) => return response,
        };
        let nodes = graph_snapshot.nodes();
        let transcript =
            match conversation_transcript_from_visible_records(&nodes, &conversation_id, 10_000) {
                Ok(transcript) => transcript,
                Err(error) => {
                    return effect_error(
                        StatusCode::NOT_FOUND,
                        "KOGWISTAR_CONVERSATION_NOT_FOUND",
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
        json_effect(StatusCode::OK, payload)
    }

    async fn sqlite_chroma_conversation_detail_effect(
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
        let graph_snapshot = match self.chroma_graph_snapshot(request, "conversation").await {
            Ok(records) => records,
            Err(response) => return response,
        };
        let nodes = graph_snapshot.nodes();
        let summary = match conversation_summary_from_visible_records(&nodes, &conversation_id) {
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

    async fn sqlite_chroma_conversation_list_effect(
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
        let graph_snapshot = match self.chroma_graph_snapshot(request, "conversation").await {
            Ok(records) => records,
            Err(response) => return response,
        };
        let nodes = graph_snapshot.nodes();
        let candidates = compact_conversation_list_candidates(&nodes);
        let mut conversations = ConversationListAccumulator::new(user_id);
        conversations.extend_visible_records(&candidates);
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

    // PYTHON_PARITY: kogwistar/server/mcp_tools.py::_load_persisted_graph
}
