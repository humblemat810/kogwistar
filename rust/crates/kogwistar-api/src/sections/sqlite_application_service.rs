// SQLite ApplicationService adapter and runtime dispatch.
// Python counterparts: ChatRunService and runtime_api operation dispatch.
impl ApplicationService for SqliteRunApplicationService {
    fn execute_inline_embedding_work_once(
        &self,
        limit: usize,
        lease_seconds: i64,
    ) -> Pin<Box<dyn Future<Output = Result<bool, String>> + Send + '_>> {
        Box::pin(async move {
            self.run_embedding_projection_jobs_once(limit, lease_seconds, None)
                .await
                .map(|summary| summary.claimed == limit)
        })
    }

    fn reconcile_eventual_projection_once(
        &self,
        namespace: String,
        page_size: usize,
    ) -> Pin<Box<dyn Future<Output = Result<usize, String>> + Send + '_>> {
        Box::pin(async move {
            self.reconcile_chroma_projection(&namespace, page_size)
                .await
        })
    }

    fn execute_inline_native_work_once(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<bool, String>> + Send + '_>> {
        Box::pin(async move {
            let input = RuntimeClaimRequest {
                claimed_by: "rust-native-worker".to_owned(),
                limit: 1,
                lease_seconds: 60,
                run_id: None,
                recipient_ids: vec!["rust-native-worker".to_owned()],
            };
            let recipients = effective_runtime_claim_recipients(&input.recipient_ids)?;
            let mut claimed = self.claim_runtime_work_items(input, recipients)?;
            let Some(work) = claimed.pop() else {
                return Ok(false);
            };
            let operation = work
                .payload
                .get("op")
                .and_then(Value::as_str)
                .map(|operation| {
                    kogwistar_runtime::native_operation_for_target(
                        operation,
                        work.payload.get("executor_target").and_then(Value::as_str),
                    )
                    .unwrap_or(operation)
                });
            let mut result = if operation == Some("agent.event.request") {
                let (handoff, native_request) = native_request_for_claimed_work(&work)?;
                self.execute_sqlite_native_provider_event_request(handoff, native_request)
                    .await?
            } else if operation == Some("agent.event.receive") {
                let (handoff, mut native_request) = native_request_for_claimed_work(&work)?;
                let principal = self
                    .resolve_inline_native_owner_principal(&handoff.run_id, "read_graph")
                    .await
                    .map_err(|_| {
                        "provider-event receive requires the run owner's current read_graph authority"
                            .to_owned()
                    })?;
                let Some(context) = self
                    .claim_native_provider_event_context(&native_request, &principal)
                    .await?
                else {
                    return Err("no authorized provider event is currently claimable".to_owned());
                };
                native_request.ephemeral_context.insert(
                    "authorized_provider_event".to_owned(),
                    context["event"].clone(),
                );
                native_request.ephemeral_context.insert(
                    "authorized_provider_event_claim".to_owned(),
                    context["claim"].clone(),
                );
                native_result_from_request(handoff, native_request, &self.native_operations)?
            } else if operation == Some("aa_get_view_and_question") {
                let (handoff, mut native_request) = native_request_for_claimed_work(&work)?;
                let view = self
                    .native_answer_view_context(&native_request, None)
                    .await
                    .map_err(|error| match error {
                        NativeEvidenceContextError::OwnerAuthorization => {
                            "native answer view requires the run owner's current read_graph authority"
                                .to_owned()
                        }
                        NativeEvidenceContextError::GraphReadRejected => {
                            "native answer view requires an active user turn in the admitted conversation"
                                .to_owned()
                        }
                        NativeEvidenceContextError::GraphReadUnavailable => {
                            "native answer-view storage is temporarily unavailable".to_owned()
                        }
                        NativeEvidenceContextError::CallerNotRunOwner => unreachable!(
                            "inline native work has no external caller principal"
                        ),
                    })?;
                native_request
                    .ephemeral_context
                    .insert("authorized_answer_view".to_owned(), view);
                native_result_from_request(handoff, native_request, &self.native_operations)?
            } else if operation == Some("aa_retrieve_candidates") {
                let (handoff, mut native_request) = native_request_for_claimed_work(&work)?;
                let records = self
                    .native_candidate_context(&native_request, None)
                    .await
                    .map_err(|error| match error {
                        NativeEvidenceContextError::OwnerAuthorization => {
                            "native candidate retrieval requires the run owner's current read_graph authority"
                                .to_owned()
                        }
                        NativeEvidenceContextError::GraphReadRejected => {
                            "native candidate retrieval input is invalid".to_owned()
                        }
                        NativeEvidenceContextError::GraphReadUnavailable => {
                            "native candidate retrieval is temporarily unavailable".to_owned()
                        }
                        NativeEvidenceContextError::CallerNotRunOwner => unreachable!(
                            "inline native work has no external caller principal"
                        ),
                    })?;
                native_request
                    .ephemeral_context
                    .insert("authorized_candidate_records".to_owned(), records);
                native_result_from_request(handoff, native_request, &self.native_operations)?
            } else if operation == Some("aa_materialize_evidence_pack") {
                let (handoff, mut native_request) = native_request_for_claimed_work(&work)?;
                let records = self
                    .native_evidence_context(&native_request, None)
                    .await
                    .map_err(|error| match error {
                        NativeEvidenceContextError::OwnerAuthorization => {
                            "native evidence materialization requires the run owner's current read_graph authority"
                                .to_owned()
                        }
                        NativeEvidenceContextError::GraphReadRejected => {
                            "native evidence materialization could not read the selected knowledge records"
                                .to_owned()
                        }
                        NativeEvidenceContextError::GraphReadUnavailable => {
                            "native evidence storage is temporarily unavailable".to_owned()
                        }
                        NativeEvidenceContextError::CallerNotRunOwner => unreachable!(
                            "inline native work has no external caller principal"
                        ),
                    })?;
                native_request
                    .ephemeral_context
                    .insert("authorized_evidence_records".to_owned(), records);
                native_result_from_request(handoff, native_request, &self.native_operations)?
            } else if operation == Some("aa_project_pointers") {
                let (handoff, mut native_request) = native_request_for_claimed_work(&work)?;
                let records = self
                    .native_pointer_context(&native_request, None)
                    .await
                    .map_err(|error| match error {
                        NativeEvidenceContextError::OwnerAuthorization => {
                            "native pointer projection requires the run owner's current read_graph authority"
                                .to_owned()
                        }
                        NativeEvidenceContextError::GraphReadRejected => {
                            "native pointer projection input or evidence is not authorized"
                                .to_owned()
                        }
                        NativeEvidenceContextError::GraphReadUnavailable => {
                            "native pointer projection storage is temporarily unavailable"
                                .to_owned()
                        }
                        NativeEvidenceContextError::CallerNotRunOwner => unreachable!(
                            "inline native work has no external caller principal"
                        ),
                    })?;
                native_request
                    .ephemeral_context
                    .insert("authorized_pointer_records".to_owned(), records);
                native_result_from_request(handoff, native_request, &self.native_operations)?
            } else if operation == Some("link_assistant_turn") {
                let (handoff, mut native_request) = native_request_for_claimed_work(&work)?;
                let records = self
                    .native_turn_link_context(&native_request, None)
                    .await
                    .map_err(|error| match error {
                        NativeEvidenceContextError::OwnerAuthorization => {
                            "native turn linking requires the run owner's current read_graph authority"
                                .to_owned()
                        }
                        NativeEvidenceContextError::GraphReadRejected => {
                            "native turn linking requires visible, active turns in this conversation"
                                .to_owned()
                        }
                        NativeEvidenceContextError::GraphReadUnavailable => {
                            "native turn storage is temporarily unavailable".to_owned()
                        }
                        NativeEvidenceContextError::CallerNotRunOwner => unreachable!(
                            "inline native work has no external caller principal"
                        ),
                    })?;
                native_request
                    .ephemeral_context
                    .insert("authorized_turn_records".to_owned(), records);
                native_result_from_request(handoff, native_request, &self.native_operations)?
            } else if operation == Some("aa_generate_answer_with_citations") {
                let (handoff, native_request) = native_request_for_claimed_work(&work)?;
                self.execute_sqlite_native_answer_generation(handoff, native_request)
                    .await?
            } else if operation == Some("aa_select_used_evidence") {
                let (handoff, native_request) = native_request_for_claimed_work(&work)?;
                if crate::native_answer_generation::selection_snapshot_is_enabled(&native_request) {
                    self.persist_sqlite_native_selection_context_snapshot(&native_request)
                        .await?;
                }
                native_result_from_request(handoff, native_request, &self.native_operations)?
            } else {
                native_result_for_claimed_work(work, &self.native_operations)?
            };
            let mut inline_principal = if result.effect.as_ref().is_some_and(|effect| {
                effect.graph_mutation_batch.is_some()
                    || effect.graph_mutation_batch_v2.is_some()
                    || !effect.provider_event_acknowledgements.is_empty()
            }) {
                match self
                    .resolve_inline_native_owner_principal(&result.handoff.run_id, "write_graph")
                    .await
                {
                    Ok(principal) => Some(principal),
                    Err(InlineOwnerPrincipalError::Denied(_)) => {
                        let effect = result
                            .effect
                            .as_mut()
                            .expect("graph effect is present in this branch");
                        fail_inline_graph_effect(
                            effect,
                            "KOGWISTAR_GRAPH_AUTHORIZATION_DENIED",
                            "inline graph effects failed current owner authorization",
                        );
                        None
                    }
                    Err(InlineOwnerPrincipalError::Unavailable(error)) => return Err(error),
                }
            } else {
                None
            };
            let scope_denied = inline_principal
                .as_ref()
                .zip(
                    result
                        .effect
                        .as_ref()
                        .and_then(recorded_effect_graph_mutations),
                )
                .is_some_and(|(principal, mutations)| {
                    inline_graph_mutations_outside_owner_scope(principal, mutations)
                });
            if scope_denied {
                let effect = result
                    .effect
                    .as_mut()
                    .expect("graph batch is present in this branch");
                fail_inline_graph_effect(
                    effect,
                    "KOGWISTAR_SCOPE_FORBIDDEN",
                    "inline graph effect scope is outside the authenticated run owner scope",
                );
                inline_principal = None;
            }
            let applied = self.apply_runtime_result_input(
                result,
                kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET,
                inline_principal.as_ref(),
            );
            if !(200..300).contains(&applied.status) {
                return Err(format!(
                    "native runtime result application returned HTTP {}",
                    applied.status
                ));
            }
            Ok(true)
        })
    }

    fn supports_mcp_tool(&self, surface: &str, name: &str) -> bool {
        if surface == "root" && name == "document_id_from_file_name" {
            return DocumentStore::supports_document_projection_lookup(&self.store);
        }
        if surface == "root" && name == "kg_load_persisted" {
            return self.chroma_graph.is_some();
        }
        if surface == "root" && name == "kg_propose_bruteforce" {
            return self.chroma_graph.is_some();
        }
        if name == "workflow.operator_inbox" && self.chroma_graph.is_none() {
            return false;
        }
        if surface == "root" {
            return self.chroma_graph.is_some()
                && (matches!(
                    name,
                    "kg_find_edges"
                        | "kg_load_persisted"
                        | "kg_propose_bruteforce"
                        | "propose_vector"
                        | "kg_k_hop"
                        | "kg_neighbors"
                        | "kg_shortest_path"
                        | "kg_semantic_seed_then_expand_text"
                        | "wisdom.semantic_seed_then_expand"
                        | "kg_viz_cytoscape_json"
                        | "kg_viz_d3_json"
                ) || (name == "wisdom.kg_upsert_graph" && self.embedding_provider.is_some()))
                && (!matches!(
                    name,
                    "kg_semantic_seed_then_expand_text" | "wisdom.semantic_seed_then_expand"
                )
                    || self.embedding_provider.is_some());
        }
        name != "workflow.tool_audit"
            && name != "workflow.run_evidence"
            && name != "workflow.run_lineage"
            && (name != "conversation.create" || self.chroma_graph.is_some())
            && (name != "conversation.ask"
                || (self.chroma_graph.is_some()
                    && self.embedding_provider.is_some()
                    && self.owner_auth_store.is_some()))
            && (name != "conversation.get_transcript" || self.chroma_graph.is_some())
            && rust_mcp_tool_supported(surface, name)
    }

    fn graph_authority_capability(&self) -> GraphAuthorityCapability {
        if self.chroma_graph.is_some() {
            GraphAuthorityCapability::AtomicBatchWithIndexJobs
        } else {
            GraphAuthorityCapability::Unsupported
        }
    }

    fn document_authority_capability(&self) -> DocumentAuthorityCapability {
        if self.chroma_graph.is_some() {
            self.store.document_authority_capability()
        } else {
            DocumentAuthorityCapability::Unsupported
        }
    }

    fn supports_mcp_tool_for_principal(
        &self,
        surface: &str,
        name: &str,
        principal: &Value,
    ) -> bool {
        // PYTHON_PARITY_V1: service declaration MCP admission requires rw/workflow; the effect independently checks service.manage. rust=rust/crates/kogwistar-api/src/sections/sqlite_application_service.rs#supports_mcp_tool_for_principal; python=kogwistar/server/chat_mcp.py#build_workflow_mcp.workflow_service_declare+kogwistar/server/chat_service.py#ChatRunService.declare_service; fixture=contracts/golden/workflow-service-declare-mcp-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/workflow_mutation_authorization.rs#workflow_service_declare_mcp_admission_matches_python_policy_for_all_stores; python_test=tests/mcp/test_chat_mcp_tools.py#test_mcp_workflow_service_declare_denies_unauthorized_effects
        if name == "workflow.service_declare" {
            return surface == "workflow"
                && principal_role(principal) == "rw"
                && principal_strings(principal, "ns")
                    .iter()
                    .any(|namespace| namespace == "workflow" || namespace == "*")
                && self.supports_mcp_tool(surface, name);
        }
        // PYTHON_PARITY_V1: service lifecycle MCP admission requires rw/workflow; effects independently check service.manage. rust=rust/crates/kogwistar-api/src/sections/sqlite_application_service.rs#supports_mcp_tool_for_principal; python=kogwistar/server/chat_mcp.py#build_workflow_mcp.workflow_service_enable+workflow_service_disable+kogwistar/server/chat_service.py#ChatRunService.enable_service+disable_service; fixtures=contracts/golden/workflow-service-enable-mcp-v1.json+workflow-service-disable-mcp-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/workflow_mutation_authorization.rs#workflow_service_lifecycle_mcp_admission_matches_python_policy_for_all_stores; python_test=tests/mcp/test_chat_mcp_tools.py#test_mcp_workflow_service_lifecycle_denies_unauthorized_effects
        if matches!(name, "workflow.service_enable" | "workflow.service_disable") {
            return surface == "workflow"
                && principal_role(principal) == "rw"
                && principal_strings(principal, "ns")
                    .iter()
                    .any(|namespace| namespace == "workflow" || namespace == "*")
                && self.supports_mcp_tool(surface, name);
        }
        // PYTHON_PARITY_V1: heartbeat MCP admission requires rw/workflow; its effect requires both service.heartbeat and service.manage. rust=rust/crates/kogwistar-api/src/sections/sqlite_application_service.rs#supports_mcp_tool_for_principal; python=kogwistar/server/chat_mcp.py#build_workflow_mcp.workflow_service_heartbeat+kogwistar/server/chat_service.py#ChatRunService.record_service_heartbeat; fixture=contracts/golden/workflow-service-heartbeat-mcp-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/workflow_mutation_authorization.rs#workflow_service_heartbeat_mcp_admission_matches_python_policy_for_all_stores; python_test=tests/mcp/test_chat_mcp_tools.py#test_mcp_workflow_service_heartbeat_denies_unauthorized_effects
        if name == "workflow.service_heartbeat" {
            return surface == "workflow"
                && principal_role(principal) == "rw"
                && principal_strings(principal, "ns")
                    .iter()
                    .any(|namespace| namespace == "workflow" || namespace == "*")
                && self.supports_mcp_tool(surface, name);
        }
        // PYTHON_PARITY_V1: trigger MCP admission requires rw/workflow; its effect independently checks service.manage. rust=rust/crates/kogwistar-api/src/sections/sqlite_application_service.rs#supports_mcp_tool_for_principal; python=kogwistar/server/chat_mcp.py#build_workflow_mcp.workflow_service_trigger+kogwistar/server/chat_service.py#ChatRunService.trigger_service; fixture=contracts/golden/workflow-service-trigger-mcp-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/workflow_mutation_authorization.rs#workflow_service_trigger_mcp_admission_matches_python_policy_for_all_stores; python_test=tests/mcp/test_chat_mcp_tools.py#test_mcp_workflow_service_trigger_denies_unauthorized_effects
        if name == "workflow.service_trigger" {
            return surface == "workflow"
                && principal_role(principal) == "rw"
                && principal_strings(principal, "ns")
                    .iter()
                    .any(|namespace| namespace == "workflow" || namespace == "*")
                && self.supports_mcp_tool(surface, name);
        }
        // PYTHON_PARITY_V1: repair MCP admission requires rw/workflow; its effect independently checks service.manage. rust=rust/crates/kogwistar-api/src/sections/sqlite_application_service.rs#supports_mcp_tool_for_principal; python=kogwistar/server/chat_mcp.py#build_workflow_mcp.workflow_service_repair+kogwistar/server/chat_service.py#ChatRunService.repair_service_projection; fixture=contracts/golden/workflow-service-repair-mcp-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/workflow_mutation_authorization.rs#workflow_service_repair_mcp_admission_matches_python_policy_for_all_stores; python_test=tests/mcp/test_chat_mcp_tools.py#test_mcp_workflow_service_repair_denies_unauthorized_effects
        if name == "workflow.service_repair" {
            return surface == "workflow"
                && principal_role(principal) == "rw"
                && principal_strings(principal, "ns")
                    .iter()
                    .any(|namespace| namespace == "workflow" || namespace == "*")
                && self.supports_mcp_tool(surface, name);
        }
        // PYTHON_PARITY_V1: bulk repair MCP admission requires rw/workflow; its effect independently checks service.manage. rust=rust/crates/kogwistar-api/src/sections/sqlite_application_service.rs#supports_mcp_tool_for_principal; python=kogwistar/server/chat_mcp.py#build_workflow_mcp.workflow_services_repair+kogwistar/server/chat_service.py#ChatRunService.repair_service_projections; fixture=contracts/golden/workflow-services-repair-mcp-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/workflow_mutation_authorization.rs#workflow_services_repair_mcp_admission_matches_python_policy_for_all_stores; python_test=tests/mcp/test_chat_mcp_tools.py#test_mcp_workflow_services_repair_denies_unauthorized_effects
        if name == "workflow.services_repair" {
            return surface == "workflow"
                && principal_role(principal) == "rw"
                && principal_strings(principal, "ns")
                    .iter()
                    .any(|namespace| namespace == "workflow" || namespace == "*")
                && self.supports_mcp_tool(surface, name);
        }
        if name == "wisdom.semantic_seed_then_expand" {
            if surface != "root"
                || !matches!(principal_role(principal).as_str(), "ro" | "rw")
                || !principal_strings(principal, "ns")
                    .iter()
                    .any(|namespace| namespace == "wisdom" || namespace == "*")
            {
                return false;
            }
            let state = self
                .capabilities
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            return effective_capabilities(principal, &state).contains("read_graph")
                && self.supports_mcp_tool(surface, name);
        }
        // PYTHON_PARITY_V1: transcript reads require ro/rw, conversation scope, and read_graph; rust=rust/crates/kogwistar-api/src/sections/sqlite_application_service.rs#supports_mcp_tool_for_principal; python=kogwistar/server/mcp_tools.py#TOOL_CAPABILITIES; rust_test=rust/crates/kogwistar-api/src/tests/conversation_tool_authorization.rs#conversation_transcript_mcp_admission_matches_python_policy_for_all_stores; python_test=tests/mcp/test_chat_mcp_tools.py#test_mcp_conversation_get_transcript_denies_missing_capability_and_namespace
        if name == "conversation.get_transcript" {
            if surface != "conversation"
                || !matches!(principal_role(principal).as_str(), "ro" | "rw")
                || !principal_strings(principal, "ns")
                    .iter()
                    .any(|namespace| namespace == "conversation" || namespace == "*")
            {
                return false;
            }
            let state = self
                .capabilities
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            return effective_capabilities(principal, &state).contains("read_graph")
                && self.supports_mcp_tool(surface, name);
        }
        // PYTHON_PARITY_V1: conversation status admission requires ro/rw and conversation namespace; rust=rust/crates/kogwistar-api/src/sections/sqlite_application_service.rs#supports_mcp_tool_for_principal; python=kogwistar/server/chat_mcp.py#build_conversation_mcp.conversation_run_status; rust_test=rust/crates/kogwistar-api/src/tests/conversation_tool_authorization.rs#conversation_run_status_mcp_admission_matches_python_policy_for_all_stores; python_test=tests/mcp/test_chat_mcp_tools.py#test_mcp_conversation_run_status_denies_unauthorized_effects
        if name == "conversation.run_status" {
            return surface == "conversation"
                && matches!(principal_role(principal).as_str(), "ro" | "rw")
                && principal_strings(principal, "ns")
                    .iter()
                    .any(|namespace| namespace == "conversation" || namespace == "*")
                && self.supports_mcp_tool(surface, name);
        }
        // PYTHON_PARITY_V1: cancel admission requires rw/conversation; rust=rust/crates/kogwistar-api/src/sections/sqlite_application_service.rs#supports_mcp_tool_for_principal; python=kogwistar/server/chat_mcp.py#build_conversation_mcp.conversation_cancel_run; rust_test=rust/crates/kogwistar-api/src/tests/conversation_tool_authorization.rs#conversation_cancel_mcp_admission_matches_python_policy_for_all_stores; python_test=tests/mcp/test_chat_mcp_tools.py#test_mcp_conversation_cancel_run_denies_unauthorized_effects
        // Capability remains status-aware in the downstream cancel service.
        if name == "conversation.cancel_run" {
            return surface == "conversation"
                && principal_role(principal) == "rw"
                && principal_strings(principal, "ns")
                    .iter()
                    .any(|namespace| namespace == "conversation" || namespace == "*")
                && self.supports_mcp_tool(surface, name);
        }
        // PYTHON_PARITY_V1: explicit read_graph + docs scope gate for these root graph tools; python=kogwistar/server/mcp_tools.py#TOOL_CAPABILITIES; rust_test=kogwistar/rust/crates/kogwistar-api/src/tests/sqlite_visibility_acl.rs#sqlite_visibility_and_capabilities_enforce_claims_across_rest_and_mcp; python_test=kogwistar/tests/mcp/test_chat_mcp_tools.py#test_root_graph_mcp_honors_explicit_capability_claims
        if matches!(
            name,
            "kg_load_persisted"
                | "kg_propose_bruteforce"
                | "propose_vector"
                | "kg_semantic_seed_then_expand_text"
                | "kg_viz_d3_json"
                | "kg_viz_cytoscape_json"
        ) {
            if surface != "root"
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
            return effective_capabilities(principal, &state).contains("read_graph")
                && self.supports_mcp_tool(surface, name);
        }
        // PYTHON_PARITY_V1: require read_graph for explicit capability claims; legacy role defaults remain intact; rust=rust/crates/kogwistar-api/src/sections/sqlite_application_service.rs#supports_mcp_tool_for_principal; python=kogwistar/server/mcp_tools.py#TOOL_CAPABILITIES; rust_test=rust/crates/kogwistar-api/src/tests/document_id_tools.rs#document_id_lookup_mcp_admission_matches_python_policy_for_all_stores; python_test=tests/mcp/test_chat_mcp_tools.py#test_document_id_from_file_name_denies_scope_and_capability_before_lookup
        if name == "document_id_from_file_name" {
            if surface != "root"
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
            return effective_capabilities(principal, &state).contains("read_graph")
                && self.supports_mcp_tool(surface, name);
        }
        // PYTHON_PARITY_V1: run-event reads require ro/rw, workflow scope, and workflow.run.read before event-store access; rust=kogwistar/rust/crates/kogwistar-api/src/sections/sqlite_application_service.rs#supports_mcp_tool_for_principal; python=kogwistar/server/chat_mcp.py#build_workflow_mcp.workflow_run_events+kogwistar/server/chat_service.py#ChatRunService.list_run_events; rust_test=kogwistar/rust/crates/kogwistar-api/src/tests/workflow_run_authorization.rs#workflow_run_events_mcp_admission_matches_python_policy_for_all_stores; python_test=kogwistar/tests/mcp/test_chat_mcp_tools.py#test_mcp_workflow_run_events_denies_missing_read_capability_before_read
        if name == "workflow.run_events" {
            if surface != "workflow"
                || !matches!(principal_role(principal).as_str(), "ro" | "rw")
                || !principal_strings(principal, "ns")
                    .iter()
                    .any(|namespace| namespace == "workflow" || namespace == "*")
            {
                return false;
            }
            let state = self
                .capabilities
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            return effective_capabilities(principal, &state).contains("workflow.run.read")
                && self.supports_mcp_tool(surface, name);
        }
        // PYTHON_PARITY_V1: dashboard preflights nested read capabilities before MCP dispatch; rust=kogwistar/rust/crates/kogwistar-api/src/sections/sqlite_application_service.rs#supports_mcp_tool_for_principal; python=kogwistar/server/chat_mcp.py#build_workflow_mcp.workflow_operator_dashboard+kogwistar/server/chat_service.py#ChatRunService.operator_dashboard; fixture=contracts/golden/operator-dashboard-v1.json; rust_test=kogwistar/rust/crates/kogwistar-api/src/tests/workflow_mutation_authorization.rs#workflow_operator_dashboard_mcp_admission_matches_python_policy_for_all_stores; python_test=kogwistar/tests/mcp/test_chat_mcp_tools.py#test_mcp_workflow_operator_dashboard_denies_before_reads
        if name == "workflow.operator_dashboard" {
            if surface != "workflow"
                || !matches!(principal_role(principal).as_str(), "ro" | "rw")
                || !principal_strings(principal, "ns")
                    .iter()
                    .any(|namespace| namespace == "workflow" || namespace == "*")
            {
                return false;
            }
            let state = self
                .capabilities
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let capabilities = effective_capabilities(principal, &state);
            return ["project_view", "workflow.run.read", "read_security_scope"]
                .iter()
                .all(|required| capabilities.contains(*required))
                && self.supports_mcp_tool(surface, name);
        }
        if name == "workflow.operator_inbox" {
            if surface != "workflow"
                || !matches!(principal_role(principal).as_str(), "ro" | "rw")
                || !principal_strings(principal, "ns")
                    .iter()
                    .any(|namespace| namespace == "workflow" || namespace == "*")
            {
                return false;
            }
            let state = self
                .capabilities
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            return effective_capabilities(principal, &state).contains("project_view")
                && self.supports_mcp_tool(surface, name);
        }
        // PYTHON_PARITY_V1: orphan repair requires rw/workflow plus project_view before dispatch; rust=rust/crates/kogwistar-api/src/sections/sqlite_application_service.rs#supports_mcp_tool_for_principal; python=kogwistar/server/chat_mcp.py#build_workflow_mcp.workflow_message_orphans_repair; python_service=kogwistar/server/chat_service.py#ChatRunService.repair_orphaned_claimed_messages; fixture=contracts/golden/workflow-repair-orphans-http-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/workflow_mutation_authorization.rs#workflow_message_orphans_repair_authorization_matches_python_policy_for_all_stores; python_test=tests/mcp/test_chat_mcp_tools.py#test_mcp_workflow_message_orphans_repair_denies_before_dispatch
        if name == "workflow.message_orphans_repair" {
            if surface != "workflow"
                || principal_role(principal) != "rw"
                || !principal_strings(principal, "ns")
                    .iter()
                    .any(|namespace| namespace == "workflow" || namespace == "*")
            {
                return false;
            }
            let state = self
                .capabilities
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            return effective_capabilities(principal, &state).contains("project_view")
                && self.supports_mcp_tool(surface, name);
        }
        // PYTHON_PARITY_V1: node upsert requires rw/workflow plus workflow.design.write and write_graph; rust=rust/crates/kogwistar-api/src/sections/sqlite_application_service.rs#supports_mcp_tool_for_principal; python=kogwistar/server/chat_mcp.py#build_workflow_mcp.workflow_design_node_upsert; python_service=kogwistar/server/chat_service.py#ChatRunService.workflow_design_upsert_node; fixture=contracts/golden/workflow-design-node-upsert-mcp-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/workflow_mutation_authorization.rs#workflow_design_node_upsert_authorization_matches_python_policy_for_all_stores; python_test=tests/mcp/test_chat_mcp_tools.py#test_mcp_workflow_design_node_upsert_denies_before_dispatch
        if name == "workflow.design_node_upsert" {
            if surface != "workflow"
                || principal_role(principal) != "rw"
                || !principal_strings(principal, "ns")
                    .iter()
                    .any(|namespace| namespace == "workflow" || namespace == "*")
            {
                return false;
            }
            let state = self
                .capabilities
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let capabilities = effective_capabilities(principal, &state);
            return ["workflow.design.write", "write_graph"]
                .iter()
                .all(|required| capabilities.contains(*required))
                && self.supports_mcp_tool(surface, name);
        }
        // PYTHON_PARITY_V1: undo/redo require rw/workflow plus workflow.design.write and approve_action; rust=rust/crates/kogwistar-api/src/sections/sqlite_application_service.rs#supports_mcp_tool_for_principal; python=kogwistar/server/chat_mcp.py#build_workflow_mcp.workflow_design_redo; python_service=kogwistar/server/chat_service.py#ChatRunService.workflow_design_redo; fixture=contracts/golden/workflow-design-redo-mcp-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/workflow_mutation_authorization.rs#workflow_design_redo_authorization_matches_python_policy_for_all_stores; python_test=tests/mcp/test_chat_mcp_tools.py#test_mcp_workflow_design_redo_denies_before_dispatch
        if matches!(name, "workflow.design_undo" | "workflow.design_redo") {
            if surface != "workflow"
                || principal_role(principal) != "rw"
                || !principal_strings(principal, "ns")
                    .iter()
                    .any(|namespace| namespace == "workflow" || namespace == "*")
            {
                return false;
            }
            let state = self
                .capabilities
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let capabilities = effective_capabilities(principal, &state);
            return ["workflow.design.write", "approve_action"]
                .iter()
                .all(|required| capabilities.contains(*required))
                && self.supports_mcp_tool(surface, name);
        }
        // PYTHON_PARITY_V1: node deletion requires rw/workflow plus workflow.design.write and write_graph; rust=rust/crates/kogwistar-api/src/sections/sqlite_application_service.rs#supports_mcp_tool_for_principal; python=kogwistar/server/chat_mcp.py#build_workflow_mcp.workflow_design_node_delete; python_service=kogwistar/server/chat_service.py#ChatRunService.workflow_design_delete_node; fixture=contracts/golden/workflow-design-node-delete-mcp-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/workflow_mutation_authorization.rs#workflow_design_node_delete_mcp_admission_matches_python_policy_for_all_stores; python_test=tests/mcp/test_chat_mcp_tools.py#test_mcp_workflow_design_node_delete_denies_before_dispatch
        if name == "workflow.design_node_delete" {
            if surface != "workflow"
                || principal_role(principal) != "rw"
                || !principal_strings(principal, "ns")
                    .iter()
                    .any(|namespace| namespace == "workflow" || namespace == "*")
            {
                return false;
            }
            let state = self
                .capabilities
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let capabilities = effective_capabilities(principal, &state);
            return ["workflow.design.write", "write_graph"]
                .iter()
                .all(|required| capabilities.contains(*required))
                && self.supports_mcp_tool(surface, name);
        }
        // PYTHON_PARITY_V1: edge deletion requires rw/workflow plus workflow.design.write and write_graph; rust=rust/crates/kogwistar-api/src/sections/sqlite_application_service.rs#supports_mcp_tool_for_principal; python=kogwistar/server/chat_mcp.py#build_workflow_mcp.workflow_design_edge_delete; python_service=kogwistar/server/chat_service.py#ChatRunService.workflow_design_delete_edge; fixture=contracts/golden/workflow-design-edge-delete-mcp-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/workflow_run_authorization.rs#workflow_design_edge_delete_mcp_admission_matches_python_policy_for_all_stores; python_test=tests/mcp/test_chat_mcp_tools.py#test_mcp_workflow_design_edge_delete_denies_before_dispatch
        if name == "workflow.design_edge_delete" {
            if surface != "workflow"
                || principal_role(principal) != "rw"
                || !principal_strings(principal, "ns")
                    .iter()
                    .any(|namespace| namespace == "workflow" || namespace == "*")
            {
                return false;
            }
            let state = self
                .capabilities
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let capabilities = effective_capabilities(principal, &state);
            return ["workflow.design.write", "write_graph"]
                .iter()
                .all(|required| capabilities.contains(*required))
                && self.supports_mcp_tool(surface, name);
        }
        // PYTHON_PARITY_V1: edge upsert requires rw/workflow plus workflow.design.write and write_graph; rust=rust/crates/kogwistar-api/src/sections/sqlite_application_service.rs#supports_mcp_tool_for_principal; python=kogwistar/server/chat_mcp.py#build_workflow_mcp.workflow_design_edge_upsert; python_service=kogwistar/server/chat_service.py#ChatRunService.workflow_design_upsert_edge; fixture=contracts/golden/workflow-design-edge-upsert-mcp-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/workflow_run_authorization.rs#workflow_design_edge_upsert_mcp_admission_matches_python_policy_for_all_stores; python_test=tests/mcp/test_chat_mcp_tools.py#test_mcp_workflow_design_edge_upsert_denies_before_dispatch
        if name == "workflow.design_edge_upsert" {
            if surface != "workflow"
                || principal_role(principal) != "rw"
                || !principal_strings(principal, "ns")
                    .iter()
                    .any(|namespace| namespace == "workflow" || namespace == "*")
            {
                return false;
            }
            let state = self
                .capabilities
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let capabilities = effective_capabilities(principal, &state);
            return ["workflow.design.write", "write_graph"]
                .iter()
                .all(|required| capabilities.contains(*required))
                && self.supports_mcp_tool(surface, name);
        }
        // PYTHON_PARITY_V1: design history reads require ro/rw, workflow namespace, and workflow.design.inspect; rust=rust/crates/kogwistar-api/src/sections/sqlite_application_service.rs#supports_mcp_tool_for_principal; python=kogwistar/server/chat_mcp.py#build_workflow_mcp.workflow_design_history; python_service=kogwistar/server/chat_service.py#ChatRunService.workflow_design_history; fixture=contracts/golden/workflow-design-history-mcp-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/workflow_run_authorization.rs#workflow_design_history_mcp_admission_matches_python_policy_for_all_stores; python_test=tests/mcp/test_chat_mcp_tools.py#test_mcp_workflow_design_history_denies_before_read
        if name == "workflow.design_history" {
            if surface != "workflow"
                || !matches!(principal.get("role").and_then(Value::as_str), Some("ro" | "rw"))
                || !principal_strings(principal, "ns")
                    .iter()
                    .any(|namespace| namespace == "workflow" || namespace == "*")
            {
                return false;
            }
            let state = self
                .capabilities
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            return effective_capabilities(principal, &state).contains("workflow.design.inspect")
                && self.supports_mcp_tool(surface, name);
        }
        // PYTHON_PARITY_V1: dead-letter snapshots require ro/rw, workflow namespace, and project_view; rust=rust/crates/kogwistar-api/src/sections/sqlite_application_service.rs#supports_mcp_tool_for_principal; python=kogwistar/server/chat_mcp.py#build_workflow_mcp.workflow_dead_letters; python_service=kogwistar/server/chat_service.py#ChatRunService.dead_letter_snapshot; fixture=contracts/golden/workflow-dead-letters-mcp-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/workflow_run_authorization.rs#workflow_dead_letters_mcp_admission_matches_python_policy_for_all_stores; python_test=tests/mcp/test_chat_mcp_tools.py#test_mcp_workflow_dead_letters_denies_before_read
        if name == "workflow.dead_letters" {
            if surface != "workflow"
                || !matches!(principal_role(principal).as_str(), "ro" | "rw")
                || !principal_strings(principal, "ns")
                    .iter()
                    .any(|namespace| namespace == "workflow" || namespace == "*")
            {
                return false;
            }
            let state = self
                .capabilities
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            return effective_capabilities(principal, &state).contains("project_view")
                && self.supports_mcp_tool(surface, name);
        }
        // PYTHON_PARITY_V1: dead-letter replay requires rw/workflow plus service.manage; rust=rust/crates/kogwistar-api/src/sections/sqlite_application_service.rs#supports_mcp_tool_for_principal; python=kogwistar/server/chat_mcp.py#build_workflow_mcp.workflow_dead_letter_replay; python_service=kogwistar/server/chat_service.py#ChatRunService.replay_dead_letter; fixture=contracts/golden/workflow-dead-letter-replay-mcp-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/workflow_run_authorization.rs#workflow_dead_letter_replay_mcp_admission_matches_python_policy_for_all_stores; python_test=tests/mcp/test_chat_mcp_tools.py#test_mcp_workflow_dead_letter_replay_denies_without_dispatch
        if name == "workflow.dead_letter_replay" {
            if surface != "workflow"
                || principal_role(principal) != "rw"
                || !principal_strings(principal, "ns")
                    .iter()
                    .any(|namespace| namespace == "workflow" || namespace == "*")
            {
                return false;
            }
            let state = self
                .capabilities
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            return effective_capabilities(principal, &state).contains("service.manage")
                && self.supports_mcp_tool(surface, name);
        }
        // PYTHON_PARITY_V1: approval/revocation require rw/workflow plus approve_action, project_view, and read_security_scope before dispatch; rust=rust/crates/kogwistar-api/src/sections/sqlite_application_service.rs#supports_mcp_tool_for_principal; python=kogwistar/server/chat_mcp.py#build_workflow_mcp.workflow_capability_revoke; python_service=kogwistar/server/chat_service.py#ChatRunService.capability_revoke; fixture=contracts/golden/workflow-capability-revoke-mcp-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/conversation_tool_authorization.rs#workflow_capability_approve_mcp_admission_matches_python_policy_for_all_stores; python_test=tests/mcp/test_chat_mcp_tools.py#test_mcp_workflow_capability_revoke_denies_without_mutation
        if matches!(name, "workflow.capability_approve" | "workflow.capability_revoke") {
            if surface != "workflow"
                || principal_role(principal) != "rw"
                || !principal_strings(principal, "ns")
                    .iter()
                    .any(|namespace| namespace == "workflow" || namespace == "*")
            {
                return false;
            }
            let state = self
                .capabilities
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let capabilities = effective_capabilities(principal, &state);
            return ["approve_action", "project_view", "read_security_scope"]
                .iter()
                .all(|required| capabilities.contains(*required))
                && self.supports_mcp_tool(surface, name);
        }
        // PYTHON_PARITY_V1: both read snapshots require project_view + read_security_scope; rust=rust/crates/kogwistar-api/src/sections/sqlite_application_service.rs#supports_mcp_tool_for_principal; python=kogwistar/server/chat_mcp.py#build_workflow_mcp.workflow_capabilities_snapshot; python_service=kogwistar/server/chat_service.py#capability_snapshot; fixture=contracts/golden/workflow-capabilities-snapshot-mcp-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/workflow_run_authorization.rs#workflow_capabilities_snapshot_mcp_admission_matches_python_policy_for_all_stores; python_test=tests/mcp/test_chat_mcp_tools.py#test_mcp_workflow_capabilities_snapshot_denies_missing_capability_before_snapshot
        if matches!(name, "workflow.visibility_snapshot" | "workflow.capabilities_snapshot") {
            if surface != "workflow"
                || !matches!(principal_role(principal).as_str(), "ro" | "rw")
                || !principal_strings(principal, "ns")
                    .iter()
                    .any(|namespace| namespace == "workflow" || namespace == "*")
            {
                return false;
            }
            let state = self
                .capabilities
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let capabilities = effective_capabilities(principal, &state);
            return capabilities.contains("read_security_scope")
                && capabilities.contains("project_view")
                && self.supports_mcp_tool(surface, name);
        }
        // PYTHON_PARITY_V1: service read MCP operations require ro/rw, workflow namespace, and both service.inspect and project_view; rust=rust/crates/kogwistar-api/src/sections/sqlite_application_service.rs#supports_mcp_tool_for_principal; python=kogwistar/server/chat_mcp.py#build_workflow_mcp.workflow_service_get+workflow_service_list+workflow_service_events+kogwistar/server/chat_service.py#ChatRunService.get_service+list_services+list_service_events; rust_test=rust/crates/kogwistar-api/src/tests/workflow_mutation_authorization.rs#workflow_service_read_mcp_admission_matches_python_policy_for_all_stores+workflow_service_events_mcp_admission_matches_python_policy_for_all_stores; python_test=tests/mcp/test_chat_mcp_tools.py#test_mcp_workflow_service_reads_deny_before_lookup+test_mcp_workflow_service_events_deny_before_read
        if matches!(name, "workflow.service_get" | "workflow.service_list" | "workflow.service_events") {
            if surface != "workflow"
                || !matches!(principal_role(principal).as_str(), "ro" | "rw")
                || !principal_strings(principal, "ns")
                    .iter()
                    .any(|namespace| namespace == "workflow" || namespace == "*")
            {
                return false;
            }
            let state = self
                .capabilities
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let capabilities = effective_capabilities(principal, &state);
            return ["service.inspect", "project_view"]
                .iter()
                .all(|required| capabilities.contains(*required))
                && self.supports_mcp_tool(surface, name);
        }
        if matches!(name, "workflow.process_table" | "workflow.blocked_runs") {
            if surface != "workflow"
                || !matches!(principal_role(principal).as_str(), "ro" | "rw")
                || !principal_strings(principal, "ns")
                    .iter()
                    .any(|namespace| namespace == "workflow" || namespace == "*")
            {
                return false;
            }
            let state = self
                .capabilities
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let capabilities = effective_capabilities(principal, &state);
            return (capabilities.contains("project_view")
                || capabilities.contains("workflow.run.read"))
                && self.supports_mcp_tool(surface, name);
        }
        if name != "wisdom.kg_upsert_graph" {
            return self.supports_mcp_tool(surface, name);
        }
        if surface != "root"
            || principal_role(principal) != "rw"
            || !principal_strings(principal, "ns")
                .iter()
                .any(|namespace| namespace == "wisdom" || namespace == "*")
        {
            return false;
        }
        let state = self
            .capabilities
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        effective_capabilities(principal, &state).contains("write_graph")
            && self.supports_mcp_tool(surface, name)
    }

    fn supports_http_operation(&self, method: &str, path: &str) -> bool {
        let is_composed_search_index = method == "GET"
            && path == "/api/search_index_hybrid"
            && self.search_index_fts.is_some()
            && self.embedding_provider.is_some()
            && self.chroma_graph.is_some()
            && self.chroma_node_index.is_some();
        let is_composed_search_index_admission = method == "POST"
            && path == "/api/add_index_entries"
            && self.search_index_fts.is_some()
            && self.embedding_provider.is_some()
            && self.chroma_graph.is_some()
            && self.chroma_node_index.is_some();
        if method == "GET" && path == "/api/search_index_hybrid" && !is_composed_search_index {
            return false;
        }
        let is_composed_graph_read = self.chroma_graph.is_some()
            && method == "POST"
            && matches!(
                path.split('?').next(),
                Some(INTERNAL_GRAPH_NEIGHBORS_ROUTE)
                    | Some(INTERNAL_GRAPH_LOAD_PERSISTED_ROUTE)
                    | Some(INTERNAL_GRAPH_PROPOSE_BRUTE_FORCE_ROUTE)
                    | Some(INTERNAL_GRAPH_VECTOR_QUERY_ROUTE)
                    | Some(INTERNAL_GRAPH_FIND_EDGES_ROUTE)
                    | Some(INTERNAL_GRAPH_SHORTEST_PATH_ROUTE)
                    | Some(INTERNAL_GRAPH_K_HOP_ROUTE)
                    | Some(INTERNAL_GRAPH_VIZ_D3_ROUTE)
                    | Some(INTERNAL_GRAPH_VIZ_CYTOSCAPE_ROUTE)
            );
        let is_composed_wisdom_semantic_read = self.chroma_graph.is_some()
            && self.embedding_provider.is_some()
            && method == "POST"
            && path.split('?').next() == Some(WISDOM_SEMANTIC_SEED_EXPAND_ROUTE);
        let is_composed_graph_mutation = self.chroma_graph.is_some()
            && method == "POST"
            && path.split('?').next() == Some(INTERNAL_GRAPH_MUTATIONS_ROUTE);
        let is_composed_document_projection =
            self.chroma_graph.is_some() && is_document_projection_route(method, path);
        let is_composed_wisdom_graph_upsert = self.chroma_graph.is_some()
            && self.embedding_provider.is_some()
            && is_public_wisdom_graph_upsert_route(method, path);
        let is_composed_graph_upsert =
            self.chroma_graph.is_some() && is_public_graph_upsert_route(method, path);
        let is_composed_document_tree_upsert = self.chroma_graph.is_some()
            && is_public_document_tree_upsert_route(method, path);
        let is_composed_conversation_transcript = self.chroma_graph.is_some()
            && method == "GET"
            && conversation_transcript_id(path).is_some();
        let is_composed_conversation_detail = self.chroma_graph.is_some()
            && method == "GET"
            && conversation_detail_id(path).is_some();
        let is_composed_conversation_list =
            self.chroma_graph.is_some() && method == "GET" && is_conversation_list_route(path);
        let is_composed_conversation_create = self.chroma_graph.is_some()
            && method == "POST"
            && path.split('?').next() == Some("/api/conversations");
        let is_composed_conversation_answer = self.chroma_graph.is_some()
            && method == "POST"
            && conversation_answer_id(path).is_some();
        (FROZEN_OPENAPI_ROUTES.contains(&(method, path))
            || is_composed_graph_read
            || is_composed_wisdom_semantic_read
            || is_composed_graph_mutation
            || is_composed_document_projection
            || is_composed_wisdom_graph_upsert
            || is_composed_graph_upsert
            || is_composed_document_tree_upsert
            || is_composed_conversation_transcript
            || is_composed_conversation_detail
            || is_composed_conversation_list
            || is_composed_conversation_create
            || is_composed_conversation_answer
            || is_composed_search_index
            || is_composed_search_index_admission)
            && !is_transport_owned_http_route(method, path)
            && !PENDING_SERVER_CUTOVER_ROUTES.contains(&(method, path))
            && (path != "/api/workflow/runs/{run_id}/lineage"
                || self.graph_authority_capability().supports_single_mutation())
            && (!SQLITE_UNSUPPORTED_HTTP_ROUTES.contains(&(method, path))
                || is_composed_graph_read
                || is_composed_wisdom_semantic_read
                || is_composed_graph_mutation
                || is_composed_document_projection
                || is_composed_wisdom_graph_upsert
                || is_composed_graph_upsert
                || is_composed_document_tree_upsert
                || is_composed_conversation_transcript
                || is_composed_conversation_detail
                || is_composed_conversation_list
                || is_composed_conversation_create
                || is_composed_conversation_answer
                || is_composed_search_index
                || is_composed_search_index_admission)
    }

    fn execute(
        &self,
        request: ApiEffectRequest,
    ) -> Pin<Box<dyn Future<Output = ApiEffectResponse> + Send + '_>> {
        Box::pin(async move {
            if request.method == "POST"
                && let Some((run_id, tail)) = runtime_run_route(&request.path_and_query)
                && tail.iter().map(String::as_str).collect::<Vec<_>>() == ["resume"]
            {
                if !principal_strings(&request.principal, "ns")
                    .iter()
                    .any(|namespace| namespace == "*" || namespace == "workflow")
                {
                    return effect_error(
                        StatusCode::FORBIDDEN,
                        "KOGWISTAR_NAMESPACE_FORBIDDEN",
                        "Resuming workflow runs requires workflow namespace",
                    );
                }
                if !capability_any_allowed(
                    &request,
                    &self.capabilities,
                    "workflow.run.write",
                    &["workflow.run.write", "approve_action"],
                ) {
                    return effect_error(
                        StatusCode::FORBIDDEN,
                        "KOGWISTAR_CAPABILITY_FORBIDDEN",
                        "Resuming workflow runs requires workflow.run.write capability",
                    );
                }
                let input: RuntimeResumeRequest = match serde_json::from_slice(&request.body) {
                    Ok(input) => input,
                    Err(error) => {
                        return effect_error(
                            StatusCode::UNPROCESSABLE_ENTITY,
                            "KOGWISTAR_INVALID_REQUEST",
                            error.to_string(),
                        );
                    }
                };
                let mut handoff = request.clone();
                handoff.path_and_query = format!("/api/workflow/runs/{run_id}/resume");
                let admitted = self.execute_sync(handoff);
                if !(200..300).contains(&admitted.status) {
                    return admitted;
                }
                loop {
                    let run = match self.store.get_server_run(&run_id) {
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
                    if matches!(
                        run.status.as_str(),
                        "succeeded" | "failed" | "cancelled" | "suspended"
                    ) {
                        let state = match self.store.read_recorded_runtime_state(
                            &run_id,
                            &input.workflow_id,
                            &input.conversation_id,
                        ) {
                            Ok(state) => state,
                            Err(error) => {
                                return effect_error(
                                    StatusCode::INTERNAL_SERVER_ERROR,
                                    "KOGWISTAR_STORE_ERROR",
                                    error.to_string(),
                                );
                            }
                        };
                        return json_effect(StatusCode::OK, resumed_run_result(run, state));
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(25)).await;
                }
            }
            if request.method == "POST"
                && request.path_and_query.split('?').next() == Some("/api/add_index_entries")
                && self.supports_http_operation("POST", "/api/add_index_entries")
            {
                let (Some(index_store), Some(fts), Some(provider)) = (
                    self.chroma_node_index.as_ref(),
                    self.search_index_fts.as_ref(),
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
                    index_store,
                    "nodes_index",
                    Some(fts),
                    Some(provider),
                )
                .await;
            }
            if request.method == "GET"
                && request.path_and_query.split('?').next() == Some("/api/search_index_hybrid")
                && self.supports_http_operation("GET", "/api/search_index_hybrid")
            {
                return self.sqlite_search_index_hybrid_effect(&request).await;
            }
            if request.method == "POST" && request.path_and_query == INTERNAL_DOCUMENT_LOOKUP_ROUTE
            {
                return document_id_lookup_effect(&request, &self.store, &self.capabilities).await;
            }
            if request.method == "POST"
                && self.chroma_graph.is_some()
                && is_document_projection_route(&request.method, &request.path_and_query)
            {
                return self.sqlite_document_projection_effect(&request).await;
            }
            if request.method == "POST" && request.path_and_query == INTERNAL_RUNTIME_RESULTS_ROUTE
            {
                if request.body.len() > MAX_RUNTIME_WORKER_RESULT_BYTES {
                    return effect_error(
                        StatusCode::PAYLOAD_TOO_LARGE,
                        "KOGWISTAR_REQUEST_TOO_LARGE",
                        "Runtime worker result exceeds the 2 MiB request limit",
                    );
                }
                if let Ok(input) =
                    serde_json::from_slice::<RuntimeWorkerResultRequest>(&request.body)
                    && input
                        .effect
                        .as_ref()
                        .is_some_and(|effect| !effect.provider_event_acknowledgements.is_empty())
                {
                    let owner_principal = match self
                        .resolve_inline_native_owner_principal(&input.handoff.run_id, "write_graph")
                        .await
                    {
                        Ok(principal) => principal,
                        Err(InlineOwnerPrincipalError::Denied(_)) => {
                            return effect_error(
                                StatusCode::FORBIDDEN,
                                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                                "provider event acknowledgement requires the run owner's current write_graph authority",
                            );
                        }
                        Err(InlineOwnerPrincipalError::Unavailable(_)) => {
                            return effect_error(
                                StatusCode::SERVICE_UNAVAILABLE,
                                "KOGWISTAR_RUN_OWNER_UNAVAILABLE",
                                "provider event acknowledgement could not revalidate the run owner",
                            );
                        }
                    };
                    return self.apply_runtime_result_with_principal(&request, &owner_principal);
                }
            }
            if request.method == "GET"
                && self.chroma_graph.is_some()
                && conversation_transcript_id(&request.path_and_query).is_some()
            {
                return self
                    .sqlite_chroma_conversation_transcript_effect(&request)
                    .await;
            }
            if request.method == "GET" && self.chroma_graph.is_some() {
                if conversation_detail_id(&request.path_and_query).is_some() {
                    return self
                        .sqlite_chroma_conversation_detail_effect(&request)
                        .await;
                }
                if is_conversation_list_route(&request.path_and_query) {
                    return self.sqlite_chroma_conversation_list_effect(&request).await;
                }
            }
            if request.method == "POST" && self.chroma_graph.is_some() {
                match request.path_and_query.split('?').next() {
                    Some(PUBLIC_GRAPH_UPSERT_ROUTE) => {
                        return self.sqlite_public_graph_upsert_effect(&request).await;
                    }
                    Some(PUBLIC_DOCUMENT_TREE_UPSERT_ROUTE) => {
                        return self
                            .sqlite_public_document_tree_upsert_effect(&request)
                            .await;
                    }
                    Some(PUBLIC_WISDOM_GRAPH_UPSERT_ROUTE) if self.embedding_provider.is_some() => {
                        return self
                            .sqlite_public_wisdom_graph_upsert_effect(&request)
                            .await;
                    }
                    Some(INTERNAL_GRAPH_MUTATIONS_ROUTE) => {
                        return self.sqlite_chroma_graph_mutation_effect(&request).await;
                    }
                    Some("/api/syscall/v1/receive_message") => {
                        let response = self.sqlite_chroma_receive_message_effect(&request).await;
                        if let Ok(input) = syscall_input(&request, "receive_message") {
                            syscall_audit(
                                &self.capabilities,
                                &input.version,
                                "receive_message",
                                syscall_audit_status(&response),
                            );
                        }
                        return response;
                    }
                    Some("/api/syscall/v1/mount_memory") => {
                        let response = self.sqlite_chroma_mount_memory_effect(&request).await;
                        if let Ok(input) = syscall_input(&request, "mount_memory") {
                            syscall_audit(
                                &self.capabilities,
                                &input.version,
                                "mount_memory",
                                syscall_audit_status(&response),
                            );
                        }
                        return response;
                    }
                    Some("/api/syscall/v1/invoke_tool") => {
                        let response = self
                            .sqlite_chroma_invoke_builtin_tool_effect(&request)
                            .await;
                        if let Ok(input) = syscall_input(&request, "invoke_tool") {
                            syscall_audit(
                                &self.capabilities,
                                &input.version,
                                "invoke_tool",
                                syscall_audit_status(&response),
                            );
                        }
                        return response;
                    }
                    Some(INTERNAL_GRAPH_LOAD_PERSISTED_ROUTE) => {
                        return self.chroma_graph_load_persisted_effect(&request).await;
                    }
                    Some(INTERNAL_GRAPH_NEIGHBORS_ROUTE) => {
                        return self.chroma_graph_neighbors_effect(&request).await;
                    }
                    Some(INTERNAL_GRAPH_PROPOSE_BRUTE_FORCE_ROUTE) => {
                        return self.chroma_graph_propose_bruteforce_effect(&request).await;
                    }
                    Some(INTERNAL_GRAPH_VECTOR_QUERY_ROUTE) => {
                        return self.chroma_graph_vector_query_effect(&request).await;
                    }
                    Some(INTERNAL_GRAPH_PROPOSE_VECTOR_ROUTE) => {
                        return self.chroma_graph_propose_vector_effect(&request).await;
                    }
                    Some(GRAPH_SEMANTIC_SEED_EXPAND_ROUTE) if self.embedding_provider.is_some() => {
                        return self
                            .chroma_graph_semantic_seed_expand_effect(&request)
                            .await;
                    }
                    Some(WISDOM_SEMANTIC_SEED_EXPAND_ROUTE)
                        if self.embedding_provider.is_some() =>
                    {
                        return self
                            .chroma_wisdom_semantic_seed_expand_effect(&request)
                            .await;
                    }
                    Some(INTERNAL_GRAPH_FIND_EDGES_ROUTE) => {
                        return self.chroma_graph_find_edges_effect(&request).await;
                    }
                    Some(INTERNAL_GRAPH_SHORTEST_PATH_ROUTE) => {
                        return self.chroma_graph_shortest_path_effect(&request).await;
                    }
                    Some(INTERNAL_GRAPH_K_HOP_ROUTE) => {
                        return self.chroma_graph_k_hop_effect(&request).await;
                    }
                    Some(INTERNAL_GRAPH_VIZ_D3_ROUTE) => {
                        return self
                            .chroma_graph_visualization_effect(&request, false)
                            .await;
                    }
                    Some(INTERNAL_GRAPH_VIZ_CYTOSCAPE_ROUTE) => {
                        return self.chroma_graph_visualization_effect(&request, true).await;
                    }
                    _ => {}
                }
            }
            if request.method == "POST"
                && request.path_and_query == INTERNAL_RUNTIME_NATIVE_EXECUTE_ROUTE
            {
                return self.execute_native_runtime_step_async(&request).await;
            }
            if request.method == "POST"
                && request.path_and_query.split('?').next()
                    == Some("/api/syscall/v1/create_conversation")
                && self.chroma_graph.is_some()
            {
                let input = match syscall_input(&request, "create_conversation") {
                    Ok(input) => input,
                    Err(response) => return response,
                };
                return self
                    .sqlite_chroma_create_conversation_effect(&request, &input)
                    .await;
            }
            if request.method == "POST"
                && request.path_and_query.split('?').next() == Some("/api/syscall/v1/send_message")
                && self.chroma_graph.is_some()
            {
                let input = match syscall_input(&request, "send_message") {
                    Ok(input) => input,
                    Err(response) => return response,
                };
                return self
                    .sqlite_chroma_send_message_effect(&request, &input)
                    .await;
            }
            if request.method == "POST"
                && request.path_and_query.split('?').next() == Some("/api/conversations")
                && self.chroma_graph.is_some()
            {
                let input: CreateConversationHttpInput = match serde_json::from_slice(&request.body)
                {
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
                let input = match syscall_input(&syscall_request, "create_conversation") {
                    Ok(input) => input,
                    Err(response) => return response,
                };
                let response = self
                    .sqlite_chroma_create_conversation_effect(&syscall_request, &input)
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
                && self.chroma_graph.is_some()
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
                if input.text.trim().is_empty() {
                    return effect_error(
                        StatusCode::BAD_REQUEST,
                        "KOGWISTAR_INVALID_REQUEST",
                        "text must be non-empty",
                    );
                }
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
                let syscall_input = match syscall_input(&syscall_request, "send_message") {
                    Ok(input) => input,
                    Err(response) => return response,
                };
                let syscall_response = self
                    .sqlite_chroma_send_message_effect(&syscall_request, &syscall_input)
                    .await;
                if !(200..300).contains(&syscall_response.status) {
                    return syscall_response;
                }
                let envelope: Value = match serde_json::from_slice(&syscall_response.body) {
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
                let (Some(run_id), Some(turn_node_id)) = (
                    result.get("run_id").and_then(Value::as_str),
                    result.get("turn_node_id").and_then(Value::as_str),
                ) else {
                    return effect_error(
                        StatusCode::BAD_GATEWAY,
                        "KOGWISTAR_INVALID_SYSCALL_RESPONSE",
                        "send-message syscall result omitted run_id or turn_node_id",
                    );
                };
                let workflow_id = result
                    .get("workflow_id")
                    .and_then(Value::as_str)
                    .unwrap_or("debug.rag.v1");
                return json_effect(
                    StatusCode::ACCEPTED,
                    json!({
                        "run_id":run_id,
                        "conversation_id":conversation_id,
                        "workflow_id":workflow_id,
                        "status":"queued",
                        "user_turn_node_id":turn_node_id,
                    }),
                );
            }
            if request.method == "GET"
                && request.path_and_query.split('?').next()
                    == Some("/api/workflow/operator/dashboard")
            {
                if let Some(error) =
                    operator_inbox_authorization_error(&request, &self.capabilities)
                {
                    return error;
                }
                if !capability_allowed(
                    &request,
                    &self.capabilities,
                    "project_view",
                    &["project_view", "workflow.run.read", "read_security_scope"],
                ) {
                    return effect_error(
                        StatusCode::FORBIDDEN,
                        "KOGWISTAR_CAPABILITY_FORBIDDEN",
                        "Reading operator dashboard requires project_view, workflow.run.read, and read_security_scope capabilities",
                    );
                }
                let dashboard = self.execute_sync(request.clone());
                if dashboard.status != StatusCode::OK.as_u16() {
                    return dashboard;
                }
                let dashboard = operator_dashboard_with_capabilities(dashboard, {
                    let capability_state = self
                        .capabilities
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    capability_snapshot(&request.principal, &capability_state)
                });
                if self.chroma_graph.is_none() {
                    return dashboard;
                }
                let scope = security_scope_value(&request.principal);
                let namespace = scope["storage_namespace"].as_str().unwrap_or("default");
                let lanes = match self.store.list_projected_lane_messages(LaneMessageFilter {
                    namespace: Some(namespace.to_owned()),
                    limit: 10_000,
                    ..LaneMessageFilter::default()
                }) {
                    Ok(lanes) => lanes,
                    Err(error) => {
                        return effect_error(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "KOGWISTAR_STORE_ERROR",
                            error.to_string(),
                        );
                    }
                };
                let graph = match self.chroma_graph_snapshot(&request, "conversation").await {
                    Ok(graph) => graph,
                    Err(error) => return error,
                };
                let inbox = operator_inbox_effect(&request, &lanes, graph.nodes());
                let inbox: Value =
                    serde_json::from_slice(&inbox.body).unwrap_or_else(|_| json!({}));
                let messages = inbox["messages"].as_array().cloned().unwrap_or_default();
                return operator_dashboard_with_inbox(dashboard, messages);
            }
            if request.method == "GET"
                && request.path_and_query.split('?').next() == Some("/api/workflow/operator/inbox")
            {
                // PYTHON_PARITY_V1: authorize before checking optional graph projection availability; rust=rust/crates/kogwistar-api/src/sections/sqlite_application_service.rs#execute_sync; python=kogwistar/server/chat_mcp.py#build_workflow_mcp.workflow_operator_inbox+kogwistar/server/chat_service.py#ChatRunService.list_operator_inbox; rust_test=rust/crates/kogwistar-api/src/tests/workflow_mutation_authorization.rs#workflow_operator_inbox_authorization_denies_before_backend_access; python_test=tests/mcp/test_chat_mcp_tools.py#test_mcp_workflow_operator_inbox_denies_before_reads
                if let Some(error) =
                    operator_inbox_authorization_error(&request, &self.capabilities)
                {
                    return error;
                }
                if self.chroma_graph.is_none() {
                    return effect_error(
                        StatusCode::NOT_IMPLEMENTED,
                        "KOGWISTAR_GRAPH_AUTHORITY_UNSUPPORTED",
                        "Operator inbox requires a configured conversation graph projection",
                    );
                }
                let scope = security_scope_value(&request.principal);
                let namespace = scope["storage_namespace"].as_str().unwrap_or("default");
                let lanes = match self.store.list_projected_lane_messages(LaneMessageFilter {
                    namespace: Some(namespace.to_owned()),
                    inbox_id: decoded_query_string(&request.path_and_query, "inbox_id"),
                    status: decoded_query_string(&request.path_and_query, "status"),
                    limit: 10_000,
                    ..LaneMessageFilter::default()
                }) {
                    Ok(lanes) => lanes,
                    Err(error) => {
                        return effect_error(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "KOGWISTAR_STORE_ERROR",
                            error.to_string(),
                        );
                    }
                };
                let graph = match self.chroma_graph_snapshot(&request, "conversation").await {
                    Ok(graph) => graph,
                    Err(error) => return error,
                };
                return operator_inbox_effect(&request, &lanes, graph.nodes());
            }
            if request.method == "GET"
                && request
                    .path_and_query
                    .split('?')
                    .next()
                    .is_some_and(|path| {
                        path.starts_with("/api/workflow/runs/") && path.ends_with("/steps")
                    })
                && let (Some(auth_store), Some(user_id)) = (
                    self.owner_auth_store.as_ref(),
                    authenticated_run_owner_id(&request.principal),
                )
            {
                let Some((run_id, _)) = runtime_run_route(&request.path_and_query) else {
                    return self.execute_sync(request);
                };
                if !principal_strings(&request.principal, "ns")
                    .iter()
                    .any(|namespace| namespace == "*" || namespace == "workflow")
                {
                    return effect_error(
                        StatusCode::FORBIDDEN,
                        "KOGWISTAR_NAMESPACE_FORBIDDEN",
                        "Run reads require workflow namespace",
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
                if let Ok(Some(run)) = self.get_run(&run_id) {
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
            self.execute_sync(request)
        })
    }
}
