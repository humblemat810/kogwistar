use super::*;

impl ApplicationService for InMemoryGraphApplicationService {
    fn execute_inline_native_work_once(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<bool, String>> + Send + '_>> {
        Box::pin(self.execute_native_step_once())
    }

    fn graph_authority_capability(&self) -> GraphAuthorityCapability {
        self.store.graph_authority_capability()
    }

    fn workflow_run_submit_allowed(
        &self,
        workflow_id: String,
        user_id: Option<String>,
    ) -> Pin<Box<dyn Future<Output = Result<bool, String>> + Send + '_>> {
        Box::pin(async move {
            let Some(auth_store) = self.owner_auth_store.as_ref() else {
                return Ok(true);
            };
            let Some(user_id) = user_id else {
                return Ok(false);
            };
            auth_store
                .workflow_access_allowed(&workflow_id, &user_id, "rw")
                .await
        })
    }

    fn workflow_access_allowed(
        &self,
        workflow_id: String,
        user_id: String,
        required_role: String,
    ) -> Pin<Box<dyn Future<Output = Result<bool, String>> + Send + '_>> {
        Box::pin(async move {
            let Some(auth_store) = self.owner_auth_store.as_ref() else {
                return Ok(true);
            };
            auth_store
                .workflow_access_allowed(&workflow_id, &user_id, &required_role)
                .await
        })
    }

    fn supports_mcp_tool(&self, surface: &str, name: &str) -> bool {
        if surface == "workflow"
            && matches!(
                name,
                "workflow.run_status"
                    | "workflow.run_events"
                    | "workflow.run_lineage"
                    | "workflow.run_steps"
                    | "workflow.run_evidence"
                    | "workflow.run_resume"
                    | "workflow.run_cancel"
                    | "workflow.run_submit"
                    | "workflow.dead_letters"
                    | "workflow.dead_letter_replay"
            )
        {
            return true;
        }
        surface == "root"
            && Self::supports_graph_tool(name)
            && (!matches!(
                name,
                "kg_semantic_seed_then_expand_text" | "wisdom.semantic_seed_then_expand"
            ) || self.embedding_provider.is_some())
    }

    fn supports_mcp_tool_for_principal(
        &self,
        surface: &str,
        name: &str,
        principal: &Value,
    ) -> bool {
        // PYTHON_PARITY_NOTE_V1: status tool admission checks role + workflow namespace; get_run enforces workflow.run.read. rust=rust/crates/kogwistar-api/src/sections/in_memory_service/application_service.rs#supports_mcp_tool_for_principal; python=kogwistar/server/chat_mcp.py#build_workflow_mcp.workflow_run_status+kogwistar/server/chat_service.py#ChatRunService.get_run; rust_test=kogwistar/rust/crates/kogwistar-api/src/tests/workflow_run_authorization.rs#workflow_run_status_mcp_admission_matches_python_role_namespace_policy_for_all_stores; python_test=kogwistar/tests/mcp/test_chat_mcp_workflow_run_access.py#test_mcp_workflow_run_status_uses_capability_without_workflow_acl
        if name == "workflow.run_status" {
            return surface == "workflow"
                && matches!(principal_role(principal).as_str(), "ro" | "rw")
                && principal_strings(principal, "ns")
                    .iter()
                    .any(|namespace| namespace == "workflow" || namespace == "*")
                && self.supports_mcp_tool(surface, name);
        }
        // PYTHON_PARITY_NOTE_V1: run-event reads require ro/rw, workflow scope, and workflow.run.read; rust=rust/crates/kogwistar-api/src/sections/in_memory_service/application_service.rs#supports_mcp_tool_for_principal; python=kogwistar/server/chat_mcp.py#build_workflow_mcp.workflow_run_events; rust_test=rust/crates/kogwistar-api/src/tests/workflow_run_authorization.rs#workflow_run_events_mcp_admission_matches_python_policy_for_all_stores
        if name == "workflow.run_events" {
            return surface == "workflow"
                && matches!(principal_role(principal).as_str(), "ro" | "rw")
                && principal_strings(principal, "ns")
                    .iter()
                    .any(|namespace| namespace == "workflow" || namespace == "*")
                && effective_capabilities(
                    principal,
                    &self
                        .capabilities
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner()),
                )
                .contains("workflow.run.read")
                && self.supports_mcp_tool(surface, name);
        }
        // PYTHON_PARITY_NOTE_V1: lineage MCP admission checks role and workflow namespace; execution enforces workflow.run.read, owner/workflow authority, and graph ACL. Python=kogwistar/server/chat_mcp.py#build_workflow_mcp.workflow_run_lineage; Rust test=rust/crates/kogwistar-api/src/tests/workflow_run_access.rs#in_memory_workflow_lineage_route_matches_shared_contract; Python test=kogwistar/tests/mcp/test_chat_mcp_workflow_run_access.py#test_mcp_workflow_run_lineage_matches_shared_contract+test_mcp_workflow_run_lineage_denies_scope_capability_and_child_acl.
        if name == "workflow.run_lineage" {
            return surface == "workflow"
                && matches!(principal_role(principal).as_str(), "ro" | "rw")
                && principal_strings(principal, "ns")
                    .iter()
                    .any(|namespace| namespace == "workflow" || namespace == "*")
                && self.supports_mcp_tool(surface, name);
        }
        // PYTHON_PARITY_NOTE_V1: workflow.run_steps MCP admission uses role + workflow namespace; service execution enforces workflow.run.read, owner, and configured workflow ACL. rust=rust/crates/kogwistar-api/src/sections/in_memory_service/application_service.rs#supports_mcp_tool_for_principal; python=kogwistar/server/chat_mcp.py#build_workflow_mcp.workflow_run_steps+kogwistar/server/chat_service.py#ChatRunService.list_steps; rust_test=rust/crates/kogwistar-api/src/tests/workflow_run_authorization.rs#workflow_run_steps_mcp_admission_matches_python_role_namespace_policy_for_all_stores; python_test=tests/mcp/test_chat_mcp_workflow_run_access.py#test_chat_service_steps_still_require_read_capability_after_mcp_admission
        if name == "workflow.run_steps" {
            return surface == "workflow"
                && matches!(principal_role(principal).as_str(), "ro" | "rw")
                && principal_strings(principal, "ns")
                    .iter()
                    .any(|namespace| namespace == "workflow" || namespace == "*")
                && self.supports_mcp_tool(surface, name);
        }
        // PYTHON_PARITY_NOTE_V1: workflow.run_evidence MCP admission requires role + workflow namespace; execution requires workflow.run.read + project_view and owner-scoped ACL-filtered snapshot. Python=kogwistar/server/chat_mcp.py#build_workflow_mcp.workflow_run_evidence+kogwistar/server/chat_service.py#ChatRunService.run_evidence; rust=rust/crates/kogwistar-api/src/sections/in_memory_service/application_service.rs#supports_mcp_tool_for_principal; fixture=contracts/golden/run-evidence-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/workflow_run_access.rs#in_memory_run_evidence_route_matches_shared_contract; python_test=tests/mcp/test_chat_mcp_workflow_run_access.py#test_mcp_workflow_run_evidence_matches_shared_contract+test_mcp_workflow_run_evidence_denies_scope_and_missing_capabilities
        if name == "workflow.run_evidence" {
            let effective = effective_capabilities(
                principal,
                &self
                    .capabilities
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner()),
            );
            return surface == "workflow"
                && matches!(principal_role(principal).as_str(), "ro" | "rw")
                && principal_strings(principal, "ns")
                    .iter()
                    .any(|namespace| namespace == "workflow" || namespace == "*")
                && effective.contains("workflow.run.read")
                && effective.contains("project_view")
                && self.supports_mcp_tool(surface, name);
        }
        if name == "workflow.run_resume" {
            let effective = effective_capabilities(
                principal,
                &self
                    .capabilities
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner()),
            );
            return surface == "workflow"
                && matches!(principal_role(principal).as_str(), "rw")
                && principal_strings(principal, "ns")
                    .iter()
                    .any(|namespace| namespace == "workflow" || namespace == "*")
                && (effective.contains("workflow.run.write")
                    || effective.contains("approve_action"))
                && self.supports_mcp_tool(surface, name);
        }
        if name == "workflow.run_cancel" {
            return surface == "workflow"
                && principal_role(principal) == "rw"
                && principal_strings(principal, "ns")
                    .iter()
                    .any(|namespace| namespace == "workflow" || namespace == "*")
                && self.supports_mcp_tool(surface, name);
        }
        if name == "workflow.run_submit" {
            let effective = effective_capabilities(
                principal,
                &self
                    .capabilities
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner()),
            );
            return surface == "workflow"
                && principal_role(principal) == "rw"
                && principal_strings(principal, "ns")
                    .iter()
                    .any(|namespace| namespace == "workflow" || namespace == "*")
                && effective.contains("spawn_process")
                && effective.contains("workflow.run.write")
                && self.supports_mcp_tool(surface, name);
        }
        if name == "wisdom.semantic_seed_then_expand" {
            return self.supports_mcp_tool(surface, name)
                && matches!(principal_role(principal).as_str(), "ro" | "rw")
                && principal_strings(principal, "ns")
                    .iter()
                    .any(|namespace| namespace == "wisdom" || namespace == "*")
                && effective_capabilities(
                    principal,
                    &self
                        .capabilities
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner()),
                )
                .contains("read_graph");
        }
        if name == "workflow.dead_letter_replay" {
            let effective = effective_capabilities(
                principal,
                &self
                    .capabilities
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner()),
            );
            return surface == "workflow"
                && principal_role(principal) == "rw"
                && principal_strings(principal, "ns")
                    .iter()
                    .any(|namespace| namespace == "workflow" || namespace == "*")
                && effective.contains("service.manage")
                && self.supports_mcp_tool(surface, name);
        }
        if name == "workflow.dead_letters" {
            let effective = effective_capabilities(
                principal,
                &self
                    .capabilities
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner()),
            );
            return surface == "workflow"
                && matches!(principal_role(principal).as_str(), "ro" | "rw")
                && principal_strings(principal, "ns")
                    .iter()
                    .any(|namespace| namespace == "workflow" || namespace == "*")
                && effective.contains("project_view")
                && self.supports_mcp_tool(surface, name);
        }
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
        // PYTHON_PARITY_NOTE_V1: Public dead-letter listing is an implemented HTTP operation, not a deferred route; rust=rust/crates/kogwistar-api/src/sections/in_memory_service/application_service.rs#supports_http_operation; python=kogwistar/server/runtime_api.py#create_runtime_router.dead_letters; fixture=contracts/golden/workflow-dead-letters-http-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/workflow_run_authorization.rs#in_memory_dead_letters_http_matches_shared_python_contract; python_test=tests/server/test_chat_server_api_workflow_design.py#test_workflow_dead_letters_matches_shared_contract
        if method == "GET" && path.split('?').next() == Some("/api/workflow/dead-letters") {
            return true;
        }
        // PYTHON_PARITY_NOTE_V1: Public replay endpoint uses the same lane CAS as MCP; rust=rust/crates/kogwistar-api/src/sections/in_memory_service/application_service.rs#supports_http_operation; python=kogwistar/server/runtime_api.py#create_runtime_router.replay_dead_letter; fixture=contracts/golden/workflow-dead-letter-replay-missing-http-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/workflow_run_authorization.rs#in_memory_dead_letter_replay_http_matches_missing_run_contract; python_test=tests/server/test_chat_server_api_workflow_design.py#test_workflow_dead_letter_replay_missing_run_matches_contract
        if method == "POST" && dead_letter_replay_run_id(path).is_some() {
            return true;
        }
        if method == "GET"
            && (is_conversation_list_route(path)
                || conversation_detail_id(path).is_some()
                || conversation_transcript_id(path).is_some())
        {
            return true;
        }
        if method == "DELETE" && admin_document_delete_id(path).is_some() {
            return self.store.supports_document_rollback();
        }
        let route = path.split('?').next();
        if method == "POST"
            && matches!(
                route,
                Some("/api/syscall/v1/send_message")
                    | Some("/api/syscall/v1/receive_message")
                    | Some("/api/syscall/v1/create_conversation")
                    | Some("/api/syscall/v1/mount_memory")
                    | Some("/api/syscall/v1/invoke_tool")
                    | Some("/api/syscall/v1/spawn_process")
                    | Some("/api/syscall/v1/terminate_process")
                    | Some("/api/syscall/v1/resume")
                    | Some("/api/syscall/v1/checkpoint")
                    | Some("/api/syscall/v1/project_view")
                    | Some("/api/syscall/v1/request_approval")
            )
        {
            return true;
        }
        if method == "POST" && route == Some("/api/workflow/runs") {
            return true;
        }
        if method == "POST" && route == Some("/api/document.upsert_tree") {
            return true;
        }
        if method == "POST"
            && runtime_run_route(path).is_some_and(|(_, tail)| tail.as_slice() == ["resume"])
        {
            return true;
        }
        if method == "POST"
            && runtime_run_route(path).is_some_and(|(_, tail)| tail.as_slice() == ["cancel"])
        {
            return true;
        }
        if method == "GET"
            && runtime_run_route(path).is_some_and(|(_, tail)| {
                let tail_refs = tail.iter().map(String::as_str).collect::<Vec<_>>();
                tail.is_empty()
                    || is_runtime_inspection_tail(&tail_refs)
                    || tail.as_slice() == ["lineage"]
                    || tail.as_slice() == ["events"]
                    || tail.as_slice() == ["events", "poll"]
                    || tail.as_slice() == ["evidence"]
            })
        {
            return true;
        }
        if method == "GET" && route == Some("/api/search_index_hybrid") {
            return self.embedding_provider.is_some() && self.search_index_fts.is_some();
        }
        if method == "POST" && route == Some("/api/add_index_entries") {
            return self.embedding_provider.is_some()
                && self.search_index_fts.is_some()
                && !PENDING_SERVER_CUTOVER_ROUTES.contains(&(method, "/api/add_index_entries"));
        }
        method == "POST"
            && (!matches!(
                route,
                Some(GRAPH_SEMANTIC_SEED_EXPAND_ROUTE) | Some(WISDOM_SEMANTIC_SEED_EXPAND_ROUTE)
            ) || self.embedding_provider.is_some())
            && matches!(
                route,
                Some(INTERNAL_GRAPH_MUTATIONS_ROUTE)
                    | Some(INTERNAL_GRAPH_LOAD_PERSISTED_ROUTE)
                    | Some(INTERNAL_GRAPH_NEIGHBORS_ROUTE)
                    | Some(INTERNAL_GRAPH_PROPOSE_BRUTE_FORCE_ROUTE)
                    | Some(INTERNAL_GRAPH_VECTOR_QUERY_ROUTE)
                    | Some(GRAPH_SEMANTIC_SEED_EXPAND_ROUTE)
                    | Some(WISDOM_SEMANTIC_SEED_EXPAND_ROUTE)
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
            if request.method == "DELETE"
                && let Some(document_id) = admin_document_delete_id(&request.path_and_query)
            {
                return admin_document_delete_effect(&request, &self.store, document_id).await;
            }
            if request.method == "POST"
                && request.path_and_query.split('?').next() == Some("/api/workflow/runs")
            {
                return self.submit_run_effect(&request, None).await;
            }
            if is_public_document_tree_upsert_route(&request.method, &request.path_and_query) {
                return self.document_tree_upsert_effect(&request).await;
            }
            if request.method == "POST"
                && let Some((run_id, tail)) = runtime_run_route(&request.path_and_query)
                && tail.iter().map(String::as_str).collect::<Vec<_>>() == ["resume"]
            {
                return self.resume_runtime_run_effect(&request, &run_id).await;
            }
            if request.method == "POST"
                && let Some((run_id, tail)) = runtime_run_route(&request.path_and_query)
                && tail.iter().map(String::as_str).collect::<Vec<_>>() == ["cancel"]
            {
                return self.cancel_runtime_run_effect(&request, &run_id).await;
            }
            if request.method == "POST"
                && let Some(run_id) = dead_letter_replay_run_id(&request.path_and_query)
            {
                return self.replay_dead_letter(&request, &run_id).await;
            }
            if request.method == "GET"
                && request.path_and_query.split('?').next() == Some("/api/workflow/dead-letters")
            {
                return self.dead_letters_snapshot(&request).await;
            }
            if request.method == "POST"
                && request.path_and_query.split('?').next() == Some("/api/syscall/v1/send_message")
            {
                let input = match syscall_input(&request, "send_message") {
                    Ok(input) => input,
                    Err(response) => return response,
                };
                let response = self.in_memory_send_message_effect(&request, &input).await;
                syscall_audit(
                    &self.capabilities,
                    &input.version,
                    "send_message",
                    syscall_audit_status(&response),
                );
                return response;
            }
            if request.method == "POST"
                && request.path_and_query.split('?').next()
                    == Some("/api/syscall/v1/receive_message")
            {
                let input = match syscall_input(&request, "receive_message") {
                    Ok(input) => input,
                    Err(response) => return response,
                };
                let response = self
                    .in_memory_receive_message_effect(&request, &input)
                    .await;
                syscall_audit(
                    &self.capabilities,
                    &input.version,
                    "receive_message",
                    syscall_audit_status(&response),
                );
                return response;
            }
            if request.method == "GET"
                && (is_conversation_list_route(&request.path_and_query)
                    || conversation_detail_id(&request.path_and_query).is_some()
                    || conversation_transcript_id(&request.path_and_query).is_some())
            {
                return self.in_memory_conversation_read_effect(&request).await;
            }
            if request.method == "POST"
                && request.path_and_query.split('?').next()
                    == Some("/api/syscall/v1/create_conversation")
            {
                let input = match syscall_input(&request, "create_conversation") {
                    Ok(input) => input,
                    Err(response) => return response,
                };
                let response = self
                    .in_memory_create_conversation_effect(&request, &input)
                    .await;
                syscall_audit(
                    &self.capabilities,
                    &input.version,
                    "create_conversation",
                    syscall_audit_status(&response),
                );
                return response;
            }
            if request.method == "POST"
                && request.path_and_query.split('?').next() == Some("/api/syscall/v1/mount_memory")
            {
                let input = match syscall_input(&request, "mount_memory") {
                    Ok(input) => input,
                    Err(response) => return response,
                };
                let response = self.in_memory_mount_memory_effect(&request, &input).await;
                syscall_audit(
                    &self.capabilities,
                    &input.version,
                    "mount_memory",
                    syscall_audit_status(&response),
                );
                return response;
            }
            if request.method == "POST"
                && request.path_and_query.split('?').next() == Some("/api/syscall/v1/invoke_tool")
            {
                let input = match syscall_input(&request, "invoke_tool") {
                    Ok(input) => input,
                    Err(response) => return response,
                };
                let response = self
                    .in_memory_invoke_builtin_tool_effect(&request, &input)
                    .await;
                syscall_audit(
                    &self.capabilities,
                    &input.version,
                    "invoke_tool",
                    syscall_audit_status(&response),
                );
                return response;
            }
            if request.method == "POST"
                && matches!(
                    request.path_and_query.split('?').next(),
                    Some("/api/syscall/v1/spawn_process")
                        | Some("/api/syscall/v1/terminate_process")
                        | Some("/api/syscall/v1/resume")
                        | Some("/api/syscall/v1/checkpoint")
                        | Some("/api/syscall/v1/project_view")
                        | Some("/api/syscall/v1/request_approval")
                )
            {
                let op = syscall_operation(&request.path_and_query)
                    .expect("matched workflow syscall route has operation");
                let input = match syscall_input(&request, &op) {
                    Ok(input) => input,
                    Err(response) => return response,
                };
                let response = self
                    .in_memory_workflow_syscall_effect(&request, &input)
                    .await;
                syscall_audit(
                    &self.capabilities,
                    &input.version,
                    &op,
                    syscall_audit_status(&response),
                );
                return response;
            }
            if request.method == "GET" && runtime_run_route(&request.path_and_query).is_some() {
                return self.read_runtime_run_effect(&request).await;
            }
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
                    | WISDOM_SEMANTIC_SEED_EXPAND_ROUTE
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
