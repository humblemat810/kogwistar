use super::*;
use kogwistar_store::{LaneMessageReadStore, LaneMessageWriteStore};
use serde_json::json;

impl InMemoryGraphApplicationService {
    pub(crate) async fn submit_run_effect(
        &self,
        request: &ApiEffectRequest,
        conversation_admission: Option<InMemoryConversationAdmission>,
    ) -> ApiEffectResponse {
        if conversation_admission.is_none()
            && !principal_strings(&request.principal, "ns")
                .iter()
                .any(|namespace| matches!(namespace.as_str(), "*" | "workflow"))
        {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_NAMESPACE_FORBIDDEN",
                "Workflow run admission requires workflow namespace",
            );
        }
        if !capability_allowed(
            request,
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
        let mut input: RuntimeSubmitRun = match serde_json::from_slice(&request.body) {
            Ok(input) => input,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error.to_string(),
                );
            }
        };
        let (Some(owner_id), workflow_id, conversation_id) = (
            authenticated_run_owner_id(&request.principal),
            input.workflow_id.trim(),
            input.conversation_id.trim(),
        ) else {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_RUN_OWNER_FORBIDDEN",
                "Run admission requires an authenticated owner",
            );
        };
        if workflow_id.is_empty() || conversation_id.is_empty() {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                "workflow_id and conversation_id are required",
            );
        }
        if input
            .user_id
            .as_deref()
            .is_some_and(|requested| requested.trim() != owner_id)
            || !expected_run_owner_matches(input.expected_run_owner_id.as_deref(), Some(owner_id))
        {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_RUN_OWNER_FORBIDDEN",
                "run user_id must match the authenticated principal",
            );
        }
        let graph_projection = match kogwistar_store::ProjectionReadStore::named_projection(
            &self.store,
            "workflow_design",
            workflow_id,
        )
        .await
        {
            Ok(projection) => projection,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error.to_string(),
                );
            }
        };
        let graph_snapshot =
            match kogwistar_store::WorkflowDesignHistoryReadStore::workflow_design_snapshot(
                &self.store,
                workflow_id,
                i64::MAX,
                1,
            )
            .await
            {
                Ok(snapshot) => snapshot,
                Err(error) => {
                    return effect_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "KOGWISTAR_STORE_ERROR",
                        error.to_string(),
                    );
                }
            };
        let graph_plan = exact_runtime_graph_plan(graph_projection, graph_snapshot.clone());
        if graph_plan.is_none()
            && runtime_snapshot_declares_worker_identity(graph_snapshot.as_ref())
        {
            return effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_INVALID_WORKER_IDENTITY",
                "workflow worker identity pin is invalid or its authoritative plan is unavailable",
            );
        }
        if let Some(plan) = graph_plan.as_ref() {
            if input.start_node_id.is_none() {
                input.start_node_id = Some(plan.start_node_id.clone());
            }
            if input.node_ops.is_empty() {
                input.node_ops = plan.node_ops.clone();
            }
            if input.node_executor_targets.is_empty() {
                input.node_executor_targets = plan.node_executor_targets.clone();
            }
            if submitted_runtime_graph_conflicts(plan, &input) {
                return effect_error(
                    StatusCode::CONFLICT,
                    "KOGWISTAR_RUNTIME_GRAPH_CONFLICT",
                    "submitted frozen runtime graph differs from authoritative workflow design",
                );
            }
        }
        let effective_routes = graph_plan
            .as_ref()
            .map(|plan| plan.routes.clone())
            .unwrap_or_else(|| input.runtime_routes.clone());
        let effective_join_node_ids = graph_plan
            .as_ref()
            .map(|plan| plan.join_node_ids.clone())
            .unwrap_or_else(|| input.join_node_ids.clone());
        let effective_start_join_mask = graph_plan
            .as_ref()
            .map(|plan| plan.start_join_mask)
            .unwrap_or(input.start_join_mask);
        let effective_start_node_id = graph_plan
            .as_ref()
            .map(|plan| plan.start_node_id.clone())
            .or_else(|| {
                input
                    .start_node_id
                    .as_deref()
                    .map(str::trim)
                    .filter(|node_id| !node_id.is_empty())
                    .map(str::to_owned)
            })
            .unwrap_or_else(|| "start".to_owned());
        let effective_node_ops = graph_plan
            .as_ref()
            .map(|plan| plan.node_ops.clone())
            .unwrap_or_else(|| {
                if input.node_ops.is_empty() {
                    BTreeMap::from([(effective_start_node_id.clone(), "noop".to_owned())])
                } else {
                    input.node_ops.clone()
                }
            });
        let effective_node_versions = graph_plan
            .as_ref()
            .map(|plan| plan.node_versions.clone())
            .unwrap_or_else(|| {
                if input.node_versions.is_empty() {
                    effective_node_ops
                        .keys()
                        .map(|node_id| (node_id.clone(), "v1".to_owned()))
                        .collect()
                } else {
                    input.node_versions.clone()
                }
            });
        let effective_node_environment_ids = graph_plan
            .as_ref()
            .map(|plan| plan.node_environment_ids.clone())
            .unwrap_or_else(|| input.node_environment_ids.clone());
        let effective_node_worker_identities = graph_plan
            .as_ref()
            .map(|plan| plan.node_worker_identities.clone())
            .unwrap_or_else(|| input.node_worker_identities.clone());
        let mut effective_node_executor_targets = graph_plan
            .as_ref()
            .map(|plan| plan.node_executor_targets.clone())
            .unwrap_or_else(|| input.node_executor_targets.clone());
        if graph_plan.is_none() && effective_node_executor_targets.is_empty() {
            effective_node_executor_targets.extend(effective_node_ops.keys().map(|node_id| {
                (
                    node_id.clone(),
                    kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET.to_owned(),
                )
            }));
        }
        let registered_native_operations = self.native_operations.registered_operations();
        if effective_node_ops.is_empty()
            || !effective_node_ops.contains_key(&effective_start_node_id)
            || !effective_node_environment_ids.is_empty()
            || !effective_node_worker_identities.is_empty()
            || effective_node_ops.iter().any(|(node_id, operation)| {
                !matches!(
                    effective_node_executor_targets
                        .get(node_id)
                        .map(String::as_str),
                    Some(kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET)
                ) || !kogwistar_runtime::native_operation_for_target(
                    operation,
                    effective_node_executor_targets
                        .get(node_id)
                        .map(String::as_str),
                )
                .is_some_and(|name| {
                    registered_native_operations
                        .iter()
                        .any(|(registered, version)| {
                            registered == name
                                && effective_node_versions.get(node_id) == Some(version)
                        })
                })
            })
            || effective_node_executor_targets
                .keys()
                .any(|node_id| !effective_node_ops.contains_key(node_id))
            || effective_node_versions
                .keys()
                .any(|node_id| !effective_node_ops.contains_key(node_id))
        {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_RUNTIME_PLAN_UNSUPPORTED",
                "in-memory execution requires a valid graph whose every node uses a supported v1 Rust-native operation",
            );
        }
        let run_id = input
            .run_id
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let turn_node_id = input
            .turn_node_id
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
            .unwrap_or_else(|| format!("wf_turn|{run_id}"));
        let start_node_id = effective_start_node_id.as_str();
        let operation = effective_node_ops
            .get(start_node_id)
            .map(String::as_str)
            .unwrap_or("noop");
        let op_version = effective_node_versions
            .get(start_node_id)
            .map(String::as_str)
            .unwrap_or("v1");
        let executor_target = effective_node_executor_targets
            .get(start_node_id)
            .map(String::as_str)
            .unwrap_or(kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_secs() as i64)
            .unwrap_or_default();
        let mut initial_state = input.initial_state;
        for reserved in [
            "_rt_graph_scope",
            "_rt_run_owner_id",
            "_rt_node_ops",
            "_rt_node_versions",
            "_rt_node_environment_ids",
            "_rt_node_worker_identities",
            "_rt_node_executor_targets",
            "_rt_routes",
            "_rt_worker_protocol",
        ] {
            initial_state.remove(reserved);
        }
        if let Err(message) =
            install_runtime_trace_context(&mut initial_state, input.trace_context.as_ref())
        {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_TRACE_CONTEXT",
                message,
            );
        }
        initial_state.insert(
            "_rt_graph_scope".to_owned(),
            security_scope_value(&request.principal),
        );
        initial_state.insert("_rt_run_owner_id".to_owned(), json!(owner_id));
        install_authoritative_node_executor_targets(
            &mut initial_state,
            &effective_node_executor_targets,
        );
        inject_authoritative_capability_state(
            &mut initial_state,
            &request.principal,
            &self.capabilities,
        );
        if !effective_routes.is_empty() {
            initial_state.insert(
                "_rt_routes".to_owned(),
                serde_json::to_value(&effective_routes).expect("runtime routes serialize"),
            );
        }
        initial_state.insert(
            "_rt_node_ops".to_owned(),
            serde_json::to_value(&effective_node_ops).expect("runtime node ops serialize"),
        );
        if effective_node_versions
            .values()
            .any(|version| version != "v1")
        {
            initial_state.insert(
                "_rt_node_versions".to_owned(),
                serde_json::to_value(&effective_node_versions)
                    .expect("runtime node versions serialize"),
            );
        }
        let runtime_kind = if input.runtime_kind.trim().is_empty() {
            default_runtime_kind()
        } else {
            input.runtime_kind
        };
        let mut worker_payload = json!({
            "contract_version": 1,
            "kind": "workflow.step.execute",
            "run_id": run_id,
            "workflow_id": workflow_id,
            "conversation_id": conversation_id,
            "turn_node_id": turn_node_id,
            "worker_protocol": worker_protocol_for_runtime_kind(&runtime_kind),
            "node_id": start_node_id,
            "op": operation,
            "op_version": op_version,
            "executor_target": executor_target,
            "join_mask": effective_start_join_mask,
            "token_id": run_id,
            "step_seq": 0,
            "expected_event_seq": 2,
            "runtime_routes": effective_routes,
            "state": initial_state,
        });
        add_executor_target(&mut worker_payload, operation, Some(executor_target));
        let mut transition_payload = worker_payload.clone();
        transition_payload["initial_state"] = worker_payload["state"].clone();
        transition_payload["start_node_id"] = json!(start_node_id);
        transition_payload["join_node_ids"] = json!(effective_join_node_ids);
        transition_payload["start_join_mask"] = json!(effective_start_join_mask);
        let mut transition = runtime_start_transition(&transition_payload, &run_id);
        transition.expected_event_seq = 1;
        transition.user_id = Some(owner_id.to_owned());
        let mut admission_payload = json!({
            "run_id": run_id,
            "run_kind": "workflow_runtime",
            "conversation_id": conversation_id,
            "workflow_id": workflow_id,
            "status": "queued",
            "turn_node_id": turn_node_id,
            "implementation_attestation": runtime_implementation_attestation(
                &runtime_kind
            ),
        });
        if let Some(binding) = runtime_owner_binding_v1(&request.principal, &self.capabilities) {
            admission_payload["runtime_owner_binding"] = json!(binding);
        }
        if let Some(admission) = conversation_admission.as_ref() {
            admission_payload["admission_actor"] = json!({
                "user_id": admission.actor_user_id,
                "operation": "send_message",
            });
            if let Some(fingerprint) = &admission.request_fingerprint {
                admission_payload["request_fingerprint"] = json!(fingerprint);
            }
        }
        let run = ServerRunCreate {
            run_id: run_id.clone(),
            conversation_id: conversation_id.to_owned(),
            workflow_id: workflow_id.to_owned(),
            user_id: Some(owner_id.to_owned()),
            user_turn_node_id: turn_node_id.clone(),
            status: "queued".to_owned(),
        };
        let message_id = format!("lane|runtime-start|{run_id}");
        let lane = NewProjectedLaneMessage {
            message_id,
            namespace: "workflow".to_owned(),
            purpose: "system".to_owned(),
            inbox_id: "workflow-runtime".to_owned(),
            conversation_id: conversation_id.to_owned(),
            recipient_id: "rust-native-worker".to_owned(),
            sender_id: "rust-scheduler".to_owned(),
            msg_type: "workflow.step.execute".to_owned(),
            status: "pending".to_owned(),
            created_at: now,
            available_at: now,
            run_id: Some(run_id.clone()),
            step_id: Some(start_node_id.to_owned()),
            correlation_id: Some(run_id.clone()),
            payload_json: Some(worker_payload.to_string()),
            error_json: None,
        };
        let admission_result = match conversation_admission {
            Some(admission) => self
                .store
                .admit_server_run_with_graph_and_start_if_unchanged(
                    admission.graph_batch,
                    &admission.expected_projections,
                    run,
                    admission_payload,
                    transition,
                    lane,
                ),
            None => {
                self.store
                    .admit_server_run_with_start(run, admission_payload, transition, lane)
            }
        };
        match admission_result {
            Ok(_) => json_effect(
                StatusCode::ACCEPTED,
                json!({
                    "run_id": run_id,
                    "conversation_id": conversation_id,
                    "workflow_id": workflow_id,
                    "turn_node_id": turn_node_id,
                    "status": "queued",
                    "priority_class": input.priority_class,
                    "token_budget": input.token_budget,
                    "time_budget_ms": input.time_budget_ms,
                    "admission": "accepted",
                    "user_turn_node_id": turn_node_id,
                }),
            ),
            Err(error) => effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_RUNTIME_ADMISSION_CONFLICT",
                error.to_string(),
            ),
        }
    }

    // PYTHON_PARITY_NOTE_V1: Rust maps Python workflow syscalls to existing in-memory run, checkpoint, resource, and approval handlers; authorization and mutation semantics stay centralized. rust=rust/crates/kogwistar-api/src/sections/in_memory_service/run_lifecycle.rs#in_memory_workflow_syscall_effect; python=kogwistar/server/syscall_api.py#create_syscall_router; fixture=contracts/golden/workflow-run-resume-result-v1.json,contracts/golden/run-cancel-v1.json,contracts/golden/operator-dashboard-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/in_memory_run_admission.rs#in_memory_workflow_syscalls_use_native_run_handlers+in_memory_resume_syscall_preserves_resume_envelope_and_result+in_memory_project_view_syscall_is_reachable_through_authenticated_http; python_test=tests/server/test_syscall_api.py#test_syscall_spawn_process_returns_ok_contract+test_syscall_request_approval_is_blocked_contract+test_syscall_checkpoint_accepts_conversation_namespace_and_numeric_string_step_seq,tests/server/test_chat_server_api_run_inspection.py#test_resume_route_uses_python_defaults_and_checks_namespace+test_checkpoint_detail_uses_python_defaults_and_checks_namespace+test_runtime_resource_snapshot_reports_pressure
    pub(crate) async fn in_memory_workflow_syscall_effect(
        &self,
        request: &ApiEffectRequest,
        input: &SyscallInput,
    ) -> ApiEffectResponse {
        if input.op == "request_approval" {
            return request_approval_syscall_effect(request, &self.capabilities, input);
        }
        let required = match input.op.as_str() {
            "spawn_process" => &["spawn_process", "workflow.run.write"][..],
            "terminate_process" | "resume" => &["workflow.run.write"][..],
            "checkpoint" => &["workflow.run.read"][..],
            "project_view" => &["read_security_scope", "project_view"][..],
            _ => return unavailable_effect(request.clone()),
        };
        let write = matches!(
            input.op.as_str(),
            "spawn_process" | "terminate_process" | "resume"
        );
        if !syscall_allowed(request, &self.capabilities, &input.op, write, required) {
            return syscall_forbidden(&input.op);
        }
        match input.op.as_str() {
            "spawn_process" => {
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
                syscall_result(
                    &input.version,
                    &input.op,
                    self.submit_run_effect(&nested, None).await,
                )
            }
            "terminate_process" => {
                let Some(run_id) = input.args.get("run_id").and_then(Value::as_str) else {
                    return effect_error(
                        StatusCode::NOT_FOUND,
                        "KOGWISTAR_INVALID_REQUEST",
                        "run_id is required",
                    );
                };
                let nested = ApiEffectRequest {
                    contract_version: request.contract_version,
                    method: "POST".to_owned(),
                    path_and_query: format!("/api/workflow/runs/{run_id}/cancel"),
                    body: Vec::new(),
                    principal: request.principal.clone(),
                };
                syscall_result(
                    &input.version,
                    &input.op,
                    self.cancel_runtime_run_effect(&nested, run_id).await,
                )
            }
            "resume" => {
                let Some(run_id) = input.args.get("run_id").and_then(Value::as_str) else {
                    return effect_error(
                        StatusCode::NOT_FOUND,
                        "KOGWISTAR_INVALID_REQUEST",
                        "run_id is required",
                    );
                };
                let mut resume_args = input.args.clone();
                resume_args.remove("run_id");
                let nested = ApiEffectRequest {
                    contract_version: request.contract_version,
                    method: "POST".to_owned(),
                    path_and_query: format!("/api/workflow/runs/{run_id}/resume"),
                    body: serde_json::to_vec(&resume_args).expect("syscall args serialize"),
                    principal: request.principal.clone(),
                };
                syscall_result(
                    &input.version,
                    &input.op,
                    self.resume_runtime_run_effect(&nested, run_id).await,
                )
            }
            "checkpoint" => {
                let step_seq = input.args.get("step_seq").and_then(syscall_step_seq);
                let (Some(run_id), Some(step_seq)) =
                    (input.args.get("run_id").and_then(Value::as_str), step_seq)
                else {
                    return effect_error(
                        StatusCode::BAD_REQUEST,
                        "KOGWISTAR_INVALID_REQUEST",
                        "run_id and integer step_seq are required",
                    );
                };
                match kogwistar_store::ServerRunReadStore::server_run_events(
                    &self.store,
                    run_id,
                    0,
                    100_000,
                )
                .await
                {
                    Ok(events) => syscall_result(
                        &input.version,
                        &input.op,
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
            }
            "project_view" => {
                let runs = match kogwistar_store::ServerRunReadStore::server_runs(
                    &self.store,
                    None,
                    None,
                    None,
                    10_000,
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
                    match kogwistar_store::ServerRunReadStore::server_run_events(
                        &self.store,
                        &run.run_id,
                        0,
                        200,
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
                let services = match kogwistar_store::ProjectionReadStore::named_projections(
                    &self.store,
                    "service_registry",
                )
                .await
                {
                    Ok(rows) => rows
                        .into_iter()
                        .map(|row| Value::Object(row.payload))
                        .collect(),
                    Err(error) => {
                        return effect_error(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "KOGWISTAR_STORE_ERROR",
                            error.to_string(),
                        );
                    }
                };
                syscall_result(
                    &input.version,
                    &input.op,
                    operational_effect(
                        request,
                        "/api/workflow/resources",
                        runs,
                        lanes,
                        events,
                        services,
                        Vec::new(),
                    ),
                )
            }
            "request_approval" => request_approval_syscall(request, &self.capabilities, input),
            _ => unreachable!("operation validated before dispatch"),
        }
    }

    // PYTHON_PARITY_NOTE_V1: Rust mirrors the workflow-scoped REST/MCP cancel contract; terminal runs return unchanged, active runs require workflow.run.write, and the volatile store atomically records cancellation plus lane cleanup. rust=rust/crates/kogwistar-api/src/sections/in_memory_service/run_lifecycle.rs#cancel_runtime_run_effect; python=kogwistar/server/chat_service.py#cancel_run,kogwistar/server/chat_mcp.py#build_workflow_mcp.workflow_run_cancel; fixture=contracts/golden/run-cancel-v1.json,contracts/golden/workflow-run-cancel-mcp-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/in_memory_run_admission.rs#in_memory_run_cancel_matches_shared_contract_and_authority; python_test=tests/mcp/test_chat_mcp_workflow_run_access.py#test_mcp_workflow_run_cancel_matches_shared_contract,tests/core/test_in_memory_meta_store.py#test_in_memory_meta_retry_and_run_cancel_contract
    pub(crate) async fn cancel_runtime_run_effect(
        &self,
        request: &ApiEffectRequest,
        run_id: &str,
    ) -> ApiEffectResponse {
        if principal_role(&request.principal) != "rw"
            || !principal_strings(&request.principal, "ns")
                .iter()
                .any(|namespace| namespace == "*" || namespace == "workflow")
        {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_NAMESPACE_FORBIDDEN",
                "Cancelling workflow runs requires workflow namespace and write role",
            );
        }
        let run = match kogwistar_store::ServerRunReadStore::server_run(&self.store, run_id).await {
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
        if matches!(run.status.as_str(), "succeeded" | "failed" | "cancelled") {
            return json_effect(StatusCode::ACCEPTED, server_run_value(run));
        }
        if !capability_allowed(
            request,
            &self.capabilities,
            "workflow.run.write",
            &["workflow.run.write"],
        ) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Cancelling workflow runs requires workflow.run.write capability",
            );
        }
        let event = kogwistar_store::ServerRunWriteStore::append_server_run_event(
            &self.store,
            run_id,
            "run.cancelling",
            json!({"run_id": run_id, "status": "cancelling"}).to_string(),
        )
        .await;
        match event {
            Ok(_) => {
                match kogwistar_store::ServerRunWriteStore::request_server_run_cancel(
                    &self.store,
                    run_id,
                )
                .await
                {
                    Ok(()) => {
                        match kogwistar_store::ServerRunReadStore::server_run(&self.store, run_id)
                            .await
                        {
                            Ok(Some(run)) => {
                                json_effect(StatusCode::ACCEPTED, server_run_value(run))
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
                        }
                    }
                    Err(error) => effect_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "KOGWISTAR_STORE_ERROR",
                        error.to_string(),
                    ),
                }
            }
            Err(error) => effect_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "KOGWISTAR_STORE_ERROR",
                error.to_string(),
            ),
        }
    }

    // PYTHON_PARITY_NOTE_V1: Rust mirrors ChatRunService.resume_run and the synchronous public resume response; the volatile store commits resume transition plus continuation lane atomically, without claiming durability across process loss.
    pub(crate) async fn resume_runtime_run_effect(
        &self,
        request: &ApiEffectRequest,
        run_id: &str,
    ) -> ApiEffectResponse {
        if principal_role(&request.principal) != "rw"
            || !principal_strings(&request.principal, "ns")
                .iter()
                .any(|namespace| namespace == "*" || namespace == "workflow")
        {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_NAMESPACE_FORBIDDEN",
                "Resuming workflow runs requires workflow namespace and write role",
            );
        }
        if !capability_any_allowed(
            request,
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
        match kogwistar_store::ServerRunReadStore::server_run(&self.store, run_id).await {
            Ok(Some(_)) => {}
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
        }
        match self.store.resume_recorded_runtime_token(
            run_id,
            &input.workflow_id,
            &input.conversation_id,
            &input.suspended_node_id,
            &input.suspended_token_id,
            Some(input.client_result),
        ) {
            Err(error) => {
                return effect_error(
                    StatusCode::CONFLICT,
                    "KOGWISTAR_RUNTIME_RESUME_CONFLICT",
                    error.to_string(),
                );
            }
            Ok(result) if !query_boolean(&request.path_and_query, "wait_for_completion", true) => {
                return json_effect(
                    StatusCode::OK,
                    serde_json::to_value(result).expect("recorded runtime result serializes"),
                );
            }
            Ok(_) => {}
        }

        loop {
            let run =
                match kogwistar_store::ServerRunReadStore::server_run(&self.store, run_id).await {
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
                let state = match kogwistar_store::ProjectionReadStore::named_projection(
                    &self.store,
                    kogwistar_store::RUNTIME_CURRENT_STATE_NAMESPACE,
                    run_id,
                )
                .await
                {
                    Ok(Some(projection)) => {
                        serde_json::from_value::<kogwistar_runtime::RecordedRuntimeState>(
                            Value::Object(projection.payload),
                        )
                        .ok()
                    }
                    Ok(None) => None,
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
            match self.execute_native_step_once().await {
                Ok(true) => continue,
                Ok(false) => tokio::time::sleep(std::time::Duration::from_millis(25)).await,
                Err(error) => {
                    return effect_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "KOGWISTAR_RUNTIME_WORKER_FAILED",
                        error,
                    );
                }
            }
        }
    }

    // PYTHON_PARITY_NOTE_V1: in-memory run inspection and lineage use Python-equivalent owner, workflow ACL, graph ACL, and namespace checks. Python=kogwistar/server/chat_service_run_inspection.py#ChatRunService.workflow_run_lineage+kogwistar/server/runtime_api.py#create_runtime_router.get_workflow_run_lineage; Rust test=rust/crates/kogwistar-api/src/tests/workflow_run_access.rs#in_memory_workflow_lineage_route_matches_shared_contract; Python test=kogwistar/tests/server/test_workflow_lineage_api.py#test_python_lineage_stops_at_parent_hidden_by_graph_acl.
    pub(crate) async fn read_runtime_run_effect(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        let Some((run_id, tail)) = runtime_run_route(&request.path_and_query) else {
            return unavailable_effect(request.clone());
        };
        let tail_refs = tail.iter().map(String::as_str).collect::<Vec<_>>();
        let required_namespace =
            runtime_run_namespace(&request.path_and_query).unwrap_or("workflow");
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
            request,
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
        let run_evidence_read = tail.as_slice() == ["evidence"];
        if run_evidence_read
            && !capability_allowed(
                request,
                &self.capabilities,
                "project_view",
                &["project_view"],
            )
        {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Reading run evidence requires project_view capability",
            );
        }
        let Some(owner_id) = authenticated_run_owner_id(&request.principal) else {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_RUN_OWNER_FORBIDDEN",
                "Run inspection requires an authenticated owner",
            );
        };
        let workflow_run_read = required_namespace == "workflow"
            && (tail.is_empty()
                || is_runtime_inspection_tail(&tail_refs)
                || tail.as_slice() == ["lineage"]
                || tail.as_slice() == ["events"]
                || tail.as_slice() == ["events", "poll"]);
        let delegated_workflow_read = workflow_run_read
            && (self.owner_auth_store.is_some() || is_mcp_workflow_status_read(request));
        let run = match kogwistar_store::ServerRunReadStore::server_run(&self.store, &run_id).await
        {
            Ok(Some(run))
                if run.user_id.as_deref() == Some(owner_id) || delegated_workflow_read =>
            {
                run
            }
            Ok(Some(_)) => {
                return effect_error(
                    StatusCode::FORBIDDEN,
                    "KOGWISTAR_RUN_OWNER_FORBIDDEN",
                    "Run inspection is restricted to its owner",
                );
            }
            Ok(None) => {
                return effect_error(
                    StatusCode::NOT_FOUND,
                    "KOGWISTAR_RUN_NOT_FOUND",
                    "Unknown run_id",
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
        if (tail.is_empty()
            || is_runtime_inspection_tail(&tail_refs)
            || tail.as_slice() == ["lineage"])
            && !is_mcp_workflow_status_read(request)
            && let Some(auth_store) = self.owner_auth_store.as_ref()
        {
            match auth_store
                .workflow_access_allowed(&run.workflow_id, owner_id, "ro")
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
        if tail.is_empty() {
            return json_effect(StatusCode::OK, server_run_value(run));
        }
        if run_evidence_read {
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
            let snapshot = match kogwistar_store::context_snapshot_from_visible_records(
                &visible_nodes,
                &run.conversation_id,
                Some(&run_id),
                None,
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
            return json_effect(StatusCode::OK, public_run_evidence_value(&run_id, snapshot));
        }
        if tail.as_slice() == ["lineage"] {
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
            let authenticated_user = authenticated_run_owner_id(&request.principal);
            let mut lineage = Vec::new();
            let mut current = run_id.clone();
            let mut seen = BTreeSet::new();
            while !current.is_empty() && seen.insert(current.clone()) {
                let node_id = format!("wf_run|{current}");
                let record = visible_nodes.iter().find(|record| record.id == node_id);
                let Some(record) = record.filter(|record| {
                    !matches!(
                        record
                            .metadata
                            .get("lifecycle_status")
                            .and_then(Value::as_str),
                        Some("tombstoned" | "deleted")
                    )
                }) else {
                    if lineage.is_empty() {
                        return effect_error(
                            StatusCode::NOT_FOUND,
                            "KOGWISTAR_RUN_NOT_FOUND",
                            "Workflow run not found",
                        );
                    }
                    break;
                };
                let workflow_id = record
                    .metadata
                    .get("workflow_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if let (Some(user_id), Some(auth_store)) =
                    (authenticated_user, self.owner_auth_store.as_ref())
                    && !workflow_id.is_empty()
                {
                    match auth_store
                        .workflow_access_allowed(workflow_id, user_id, "ro")
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
                let parent = record
                    .metadata
                    .get("parent_run_id")
                    .cloned()
                    .unwrap_or(Value::Null);
                lineage.push(json!({
                    "run_id": current,
                    "workflow_id": workflow_id,
                    "parent_run_id": parent,
                    "status": record.metadata.get("status").and_then(Value::as_str).unwrap_or_default(),
                    "wf_invoked": record.metadata.get("wf_invoked").cloned().unwrap_or(Value::Null),
                }));
                current = parent.as_str().unwrap_or_default().to_owned();
            }
            if query_parameter_present(&request.path_and_query, "limit") {
                let limit =
                    query_integer(&request.path_and_query, "limit", 200).clamp(1, 1000) as usize;
                let truncated = lineage.len() > limit;
                lineage.truncate(limit);
                return json_effect(
                    StatusCode::OK,
                    json!({"run_id":run_id,"lineage":lineage,"truncated":truncated}),
                );
            }
            return json_effect(StatusCode::OK, json!({"run_id":run_id,"lineage":lineage}));
        }
        let after_seq = query_integer(&request.path_and_query, "after_seq", 0);
        let limit = query_integer(&request.path_and_query, "limit", 500).clamp(1, 10_000);
        match kogwistar_store::ServerRunReadStore::server_run_events(
            &self.store,
            &run_id,
            after_seq,
            limit as usize,
        )
        .await
        {
            Ok(events) if is_runtime_inspection_tail(&tail_refs) => {
                runtime_inspection_effect(request, &run_id, &tail_refs, events)
            }
            Ok(events)
                if tail.as_slice() == ["events"] || tail.as_slice() == ["events", "poll"] =>
            {
                let events = events
                    .into_iter()
                    .map(server_run_event_value)
                    .collect::<Vec<_>>();
                json_effect(StatusCode::OK, json!({"run_id": run_id, "events": events}))
            }
            Ok(_) => unavailable_effect(request.clone()),
            Err(error) => effect_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "KOGWISTAR_STORE_ERROR",
                error.to_string(),
            ),
        }
    }

    // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/in_memory_service/run_lifecycle.rs#search_index_hybrid_effect; python=kogwistar/server_mcp_with_admin.py#search_index_hybrid; fixture=contracts/golden/search-index-hybrid-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/in_memory_semantic_retrieval.rs#in_memory_search_index_hybrid_is_scoped_and_acl_filtered; python_test=tests/core/test_search_index_contract.py#test_hybrid_search_fusion_matches_shared_rust_golden_fixture
    pub(crate) async fn search_index_hybrid_effect(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        if !capability_allowed(request, &self.capabilities, "read_graph", &["read_graph"]) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Search-index query requires read_graph capability",
            );
        }
        let (query, limit, resolve_node) =
            match crate::search_index_fts::parse_hybrid_search_query(&request.path_and_query) {
                Ok(parsed) => parsed,
                Err(error) => {
                    return effect_error(
                        StatusCode::UNPROCESSABLE_ENTITY,
                        "KOGWISTAR_INVALID_REQUEST",
                        error,
                    );
                }
            };
        let (Some(provider), Some(fts)) = (
            self.embedding_provider.as_ref(),
            self.search_index_fts.as_ref(),
        ) else {
            return effect_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "KOGWISTAR_SEARCH_INDEX_UNAVAILABLE",
                "Search-index embedding or FTS storage is not configured",
            );
        };
        let hits = match crate::search_index_fts::hybrid_search(
            fts,
            &self.store,
            "nodes_index",
            provider.as_ref(),
            &query,
            limit,
        )
        .await
        {
            Ok(hits) => hits,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_SEARCH_INDEX_ERROR",
                    error,
                );
            }
        };
        let scope = match self.authorized_graph_scope(request) {
            Ok(scope) => scope,
            Err(response) => return response,
        };
        let (nodes, _) = match self.graph_snapshot(scope).await {
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
        let (visible_nodes, _) = graph_records_visible_by_acl(&nodes, &[], "knowledge", &principal);
        let candidate_ids = hits
            .iter()
            .map(|hit| hit.node_id.as_str())
            .collect::<BTreeSet<_>>();
        let canonical_nodes = visible_nodes
            .into_iter()
            .filter(|record| candidate_ids.contains(record.id.as_str()))
            .collect::<Vec<_>>();
        json_effect(
            StatusCode::OK,
            crate::search_index_fts::project_hybrid_search_result(
                &query,
                &hits,
                &canonical_nodes,
                resolve_node,
                limit,
            ),
        )
    }

    // PYTHON_PARITY_NOTE_V1: Replay one workflow dead-letter with store-level CAS so concurrent MCP retries yield at most one success. rust=rust/crates/kogwistar-api/src/sections/in_memory_service/run_lifecycle.rs#replay_dead_letter; python=kogwistar/server/chat_mcp.py#build_workflow_mcp.workflow_dead_letter_replay+kogwistar/server/chat_service.py#ChatRunService.replay_dead_letter; fixture=contracts/golden/workflow-dead-letter-replay-mcp-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/workflow_run_authorization.rs#in_memory_dead_letter_replay_mcp_requeues_once_under_concurrency; python_test=tests/mcp/test_chat_mcp_workflow_authority.py#test_mcp_workflow_dead_letter_replay_dispatches_effect_once
    pub(crate) async fn replay_dead_letter(
        &self,
        request: &ApiEffectRequest,
        run_id: &str,
    ) -> ApiEffectResponse {
        if principal_role(&request.principal) != "rw"
            || !principal_strings(&request.principal, "ns")
                .iter()
                .any(|namespace| namespace == "workflow" || namespace == "*")
        {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_SCOPE_FORBIDDEN",
                "Replaying dead letters requires rw role and workflow namespace",
            );
        }
        if !capability_allowed(
            request,
            &self.capabilities,
            "service.manage",
            &["service.manage"],
        ) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Replaying dead letters requires service.manage capability",
            );
        }
        let lanes = match self
            .store
            .projected_lane_messages(kogwistar_store::LaneMessageFilter {
                namespace: Some("workflow".to_owned()),
                status: Some("dead-letter".to_owned()),
                limit: 10_000,
                ..kogwistar_store::LaneMessageFilter::default()
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
        let Some(lane) = lanes
            .into_iter()
            .find(|lane| lane.run_id.as_deref() == Some(run_id))
        else {
            return json_effect(StatusCode::OK, json!({"run_id": run_id, "replayed": false}));
        };
        match self
            .store
            .compare_and_set_projected_lane_message_status(
                &lane.message_id,
                "dead-letter",
                "pending",
                None,
            )
            .await
        {
            Ok(true) => json_effect(
                StatusCode::OK,
                json!({
                    "run_id": run_id,
                    "replayed": true,
                    "dead_letter": lane_progress_value(&lane),
                }),
            ),
            Ok(false) => json_effect(StatusCode::OK, json!({"run_id": run_id, "replayed": false})),
            Err(error) => effect_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "KOGWISTAR_STORE_ERROR",
                error.to_string(),
            ),
        }
    }
}
