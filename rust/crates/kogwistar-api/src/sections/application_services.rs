fn fail_inline_graph_effect(
    effect: &mut kogwistar_runtime::RecordedWorkerSuccessEffect,
    code: &str,
    message: &str,
) {
    effect.graph_mutation_batch = None;
    effect.status = kogwistar_runtime::RuntimeWorkerEffectStatus::Failed;
    effect.state_update.clear();
    effect.update = None;
    effect.state_schema.clear();
    effect.successors.clear();
    effect.route_next.clear();
    effect.result = Some(serde_json::json!({"code": code}));
    effect.errors = vec![message.to_owned()];
    effect.wait_reason = None;
    effect.resume_payload = None;
    effect.usage = None;
    effect.trace_events.clear();
}

impl PostgresRunApplicationService {
    async fn execute_native_runtime_step(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        let input: RuntimeNativeExecuteRequest = match serde_json::from_slice(&request.body) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error.to_string(),
                );
            }
        };
        let native_request = match native_step_request_from_payload(&input.handoff, input.payload) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::CONFLICT,
                    "KOGWISTAR_NATIVE_HANDOFF_CONFLICT",
                    error,
                );
            }
        };
        let effect = match kogwistar_runtime::execute_native_step_with_registry(
            &native_request,
            &self.native_operations,
        ) {
            Ok(effect) => effect,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_NATIVE_EXECUTION_REJECTED",
                    error.to_string(),
                );
            }
        };
        let result = RuntimeWorkerResultRequest {
            handoff: input.handoff,
            transition: None,
            effect: Some(effect),
        };
        self.apply_runtime_result_input(
            result,
            Some(request),
            None,
            kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET,
        )
            .await
    }
}

impl ApplicationService for PostgresRunApplicationService {
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
            let mut claimed = self.claim_runtime_work_items(input, recipients).await?;
            let Some(work) = claimed.pop() else {
                return Ok(false);
            };
            let mut result = native_result_for_claimed_work(work, &self.native_operations)?;
            let run_id = result.handoff.run_id.clone();
            let mut inline_principal = if result
                .effect
                .as_ref()
                .and_then(|effect| effect.graph_mutation_batch.as_ref())
                .is_some()
            {
                match self.resolve_inline_native_owner_principal(&run_id).await {
                    Ok(principal) => Some(principal),
                    Err(_) => {
                        let effect = result
                            .effect
                            .as_mut()
                            .expect("graph batch is present in this branch");
                        fail_inline_graph_effect(
                            effect,
                            "KOGWISTAR_GRAPH_AUTHORIZATION_DENIED",
                            "inline graph effects failed current owner authorization",
                        );
                        None
                    }
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
                        .and_then(|effect| effect.graph_mutation_batch.as_ref()),
                )
                .is_some_and(|(principal, batch)| {
                    let security_scope = security_scope_value(principal);
                    batch.mutations.iter().any(|mutation| {
                        !graph_mutation_scope_allowed(
                            &GraphScope {
                                namespace: mutation.scope.namespace.clone(),
                                workspace_id: mutation.scope.workspace_id.clone(),
                                graph_space: mutation.scope.graph_space.clone(),
                            },
                            &security_scope,
                        )
                    })
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
            let applied = self
                .apply_runtime_result_input(
                    result,
                    None,
                    inline_principal,
                    kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET,
                )
                .await;
            if !(200..300).contains(&applied.status) {
                return Err(format!(
                    "native runtime result application returned HTTP {}",
                    applied.status
                ));
            }
            Ok(true)
        })
    }

    fn graph_authority_capability(&self) -> GraphAuthorityCapability {
        self.store.graph_authority_capability()
    }

    fn document_authority_capability(&self) -> DocumentAuthorityCapability {
        self.store.document_authority_capability()
    }

    fn supports_mcp_tool(&self, surface: &str, name: &str) -> bool {
        rust_mcp_tool_supported(surface, name)
            && (surface != "root" || self.graph_authority_capability().supports_single_mutation())
            && (name != "workflow.tool_audit"
                || self.graph_authority_capability().supports_single_mutation())
    }

    fn supports_http_operation(&self, method: &str, path: &str) -> bool {
        FROZEN_OPENAPI_ROUTES.contains(&(method, path))
            && !is_transport_owned_http_route(method, path)
            && !PENDING_SERVER_CUTOVER_ROUTES.contains(&(method, path))
    }

    fn execute(
        &self,
        request: ApiEffectRequest,
    ) -> Pin<Box<dyn Future<Output = ApiEffectResponse> + Send + '_>> {
        Box::pin(async move { self.execute_async(request).await })
    }
}

impl ApplicationService for SqliteRunApplicationService {
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
            let result = native_result_for_claimed_work(work, &self.native_operations)?;
            let applied = self.apply_runtime_result_input(
                result,
                kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET,
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
        if surface == "root" {
            return self.chroma_graph.is_some()
                && matches!(
                    name,
                    "kg_find_edges" | "kg_k_hop" | "kg_neighbors" | "kg_shortest_path"
                );
        }
        name != "workflow.tool_audit" && rust_mcp_tool_supported(surface, name)
    }

    fn supports_http_operation(&self, method: &str, path: &str) -> bool {
        let is_composed_graph_read = self.chroma_graph.is_some()
            && method == "POST"
            && matches!(
                path.split('?').next(),
                Some(INTERNAL_GRAPH_NEIGHBORS_ROUTE)
                    | Some(INTERNAL_GRAPH_VECTOR_QUERY_ROUTE)
                    | Some(INTERNAL_GRAPH_FIND_EDGES_ROUTE)
                    | Some(INTERNAL_GRAPH_SHORTEST_PATH_ROUTE)
                    | Some(INTERNAL_GRAPH_K_HOP_ROUTE)
            );
        (FROZEN_OPENAPI_ROUTES.contains(&(method, path)) || is_composed_graph_read)
            && !is_transport_owned_http_route(method, path)
            && !PENDING_SERVER_CUTOVER_ROUTES.contains(&(method, path))
            && (!SQLITE_UNSUPPORTED_HTTP_ROUTES.contains(&(method, path))
                || is_composed_graph_read)
    }

    fn execute(
        &self,
        request: ApiEffectRequest,
    ) -> Pin<Box<dyn Future<Output = ApiEffectResponse> + Send + '_>> {
        Box::pin(async move {
            if request.method == "POST" && self.chroma_graph.is_some() {
                match request.path_and_query.split('?').next() {
                    Some(INTERNAL_GRAPH_NEIGHBORS_ROUTE) => {
                        return self.chroma_graph_neighbors_effect(&request).await;
                    }
                    Some(INTERNAL_GRAPH_VECTOR_QUERY_ROUTE) => {
                        return self.chroma_graph_vector_query_effect(&request).await;
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
                    _ => {}
                }
            }
            self.execute_sync(request)
        })
    }
}

impl SqliteRunApplicationService {
    fn syscall_effect(&self, request: &ApiEffectRequest, op: &str) -> ApiEffectResponse {
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
                    syscall_result(&input.version, op, self.submit_run(&nested))
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
                    match self.store.request_server_run_cancel(run_id) {
                        Ok(()) => match self.store.get_server_run(run_id) {
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
                    match self.store.list_server_run_events(run_id, 0, 100_000) {
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
                } else {
                    unavailable_effect(request.clone())
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
                } else {
                    unavailable_effect(request.clone())
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
                    let runs = match self.store.list_server_runs(None, None, None, 10_000) {
                        Ok(runs) => runs,
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
                        match self.store.list_server_run_events(&run.run_id, 0, 200) {
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
                    syscall_result(&input.version, op, self.resume_runtime_run(run_id, &nested))
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

    fn trigger_service(&self, request: &ApiEffectRequest, service_id: &str) -> ApiEffectResponse {
        if !capability_allowed(
            request,
            &self.capabilities,
            "service.manage",
            &["service.manage"],
        ) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Triggering service requires service.manage capability",
            );
        }
        let mut input: ServiceTriggerInput = match serde_json::from_slice(&request.body) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error.to_string(),
                );
            }
        };
        if let Err(error) = validate_service_trigger(&mut input) {
            return effect_error(StatusCode::BAD_REQUEST, "KOGWISTAR_INVALID_REQUEST", error);
        }
        let timestamp = now_ms();
        let now_seconds = (timestamp / 1000) as i64;
        let max_queue = self.max_queue;
        let outcome = self.store.transaction(|uow| {
            let current = uow
                .get_named_projection("service_registry", service_id)?
                .ok_or_else(|| {
                    kogwistar_store_sqlite::SqliteStoreError::TransactionAborted(
                        "KOGWISTAR_SERVICE_NOT_FOUND".to_owned(),
                    )
                })?;
            let current_value = Value::Object(current.payload);
            if let Some(run_id) = current_value["current_child_run_id"]
                .as_str()
                .filter(|value| !value.is_empty())
                && let Some(run) = uow.get_server_run(run_id)?
                && !matches!(run.status.as_str(), "succeeded" | "failed" | "cancelled")
            {
                return Ok(Some(current_value));
            }
            let history = uow.replay_raw_events("workflow", 0, usize::MAX)?;
            let definition = latest_service_definition(
                history.into_iter().map(|event| {
                    (
                        event.seq,
                        event.entity_kind,
                        event.entity_id,
                        event.op,
                        event.payload_json,
                    )
                }),
                service_id,
            )
            .ok_or_else(|| {
                kogwistar_store_sqlite::SqliteStoreError::TransactionAborted(
                    "KOGWISTAR_SERVICE_NOT_FOUND".to_owned(),
                )
            })?;
            if service_trigger_is_suppressed(
                &definition,
                &current_value,
                &input.trigger_type,
                timestamp,
            ) {
                return Ok(Some(current_value));
            }
            let (target_kind, target_config) =
                service_definition_runtime(&definition).map_err(|message| {
                    kogwistar_store_sqlite::SqliteStoreError::TransactionAborted(message.to_owned())
                })?;
            let target_ref = definition["properties"]["target_ref"]
                .as_str()
                .unwrap_or_default()
                .to_owned();
            let mut run_id = None;
            if target_kind == "workflow" {
                let active_count = ["pending", "claimed"]
                    .into_iter()
                    .map(|status| {
                        uow.list_projected_lane_messages(&LaneMessageFilter {
                            namespace: Some("workflow".to_owned()),
                            inbox_id: Some("workflow-runtime".to_owned()),
                            status: Some(status.to_owned()),
                            limit: max_queue,
                            ..LaneMessageFilter::default()
                        })
                        .map(|lanes| lanes.len())
                    })
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .sum::<usize>();
                if active_count >= max_queue {
                    return Ok(None);
                }
                let target = target_config.as_object().cloned().unwrap_or_default();
                let conversation_id = input
                    .payload
                    .get("conversation_id")
                    .and_then(Value::as_str)
                    .or_else(|| target.get("conversation_id").and_then(Value::as_str))
                    .map(str::to_owned)
                    .unwrap_or_else(|| format!("svc:{service_id}"));
                let turn_node_id = input
                    .payload
                    .get("turn_node_id")
                    .and_then(Value::as_str)
                    .or_else(|| target.get("turn_node_id").and_then(Value::as_str))
                    .map(str::to_owned)
                    .unwrap_or_else(|| format!("wf_turn|{}", uuid::Uuid::new_v4()));
                let user_id = input
                    .payload
                    .get("user_id")
                    .and_then(Value::as_str)
                    .or_else(|| target.get("user_id").and_then(Value::as_str))
                    .map(str::to_owned);
                let mut initial_state = target
                    .get("initial_state")
                    .and_then(Value::as_object)
                    .cloned()
                    .unwrap_or_default();
                if let Some(overrides) = input
                    .payload
                    .get("initial_state")
                    .and_then(Value::as_object)
                {
                    initial_state.extend(overrides.clone());
                }
                let created_run_id = uuid::Uuid::new_v4().to_string();
                let message_id = format!("lane|{}", uuid::Uuid::new_v4());
                let graph_plan = exact_runtime_graph_plan(
                    uow.get_named_projection("workflow_design", &target_ref)?,
                    uow.get_workflow_design_snapshot(&target_ref, i64::MAX, 1)?,
                );
                let runtime_routes = graph_plan
                    .as_ref()
                    .map(|plan| plan.routes.clone())
                    .unwrap_or_default();
                let join_node_ids = graph_plan
                    .as_ref()
                    .map(|plan| plan.join_node_ids.clone())
                    .unwrap_or_default();
                let start_join_mask = graph_plan
                    .as_ref()
                    .map(|plan| plan.start_join_mask)
                    .unwrap_or_default();
                let start_node_id = graph_plan
                    .as_ref()
                    .map(|plan| plan.start_node_id.clone())
                    .unwrap_or_else(|| "start".to_owned());
                let node_ops = graph_plan
                    .as_ref()
                    .map(|plan| plan.node_ops.clone())
                    .unwrap_or_default();
                let node_versions = graph_plan
                    .as_ref()
                    .map(|plan| plan.node_versions.clone())
                    .unwrap_or_default();
                let node_environment_ids = graph_plan
                    .as_ref()
                    .map(|plan| plan.node_environment_ids.clone())
                    .unwrap_or_default();
                let node_executor_targets = graph_plan
                    .as_ref()
                    .map(|plan| plan.node_executor_targets.clone())
                    .unwrap_or_default();
                let start_op = node_ops.get(&start_node_id).cloned();
                let start_op_version = node_versions.get(&start_node_id).cloned();
                let start_environment_id = node_environment_ids.get(&start_node_id).cloned();
                let start_executor_target = node_executor_targets.get(&start_node_id).cloned();
                if !runtime_routes.is_empty() {
                    initial_state.insert(
                        "_rt_routes".to_owned(),
                        serde_json::to_value(&runtime_routes)
                            .expect("static runtime routes serialize"),
                    );
                }
                if !node_ops.is_empty() {
                    initial_state.insert(
                        "_rt_node_ops".to_owned(),
                        serde_json::to_value(&node_ops).expect("runtime node ops serialize"),
                    );
                }
                if node_versions.values().any(|version| version != "v1") {
                    initial_state.insert(
                        "_rt_node_versions".to_owned(),
                        serde_json::to_value(&node_versions)
                            .expect("runtime node versions serialize"),
                    );
                }
                if !node_environment_ids.is_empty() {
                    initial_state.insert(
                        "_rt_node_environment_ids".to_owned(),
                        serde_json::to_value(&node_environment_ids)
                            .expect("runtime node environment ids serialize"),
                    );
                }
                install_authoritative_node_executor_targets(
                    &mut initial_state,
                    &node_executor_targets,
                );
                let runtime_kind = target
                    .get("runtime_kind")
                    .and_then(Value::as_str)
                    .unwrap_or("sync");
                let worker_protocol = worker_protocol_for_runtime_kind(runtime_kind);
                initial_state.insert(
                    "_rt_worker_protocol".to_owned(),
                    Value::String(worker_protocol.to_owned()),
                );
                uow.create_server_run(ServerRunCreate {
                    run_id: created_run_id.clone(),
                    conversation_id: conversation_id.clone(),
                    workflow_id: target_ref.clone(),
                    user_id: user_id.clone(),
                    user_turn_node_id: turn_node_id.clone(),
                    status: "queued".to_owned(),
                })?;
                let created = uow.append_server_run_event(
                    &created_run_id,
                    "run.created",
                    json!({
                        "run_id": created_run_id,
                        "run_kind": "workflow_runtime",
                        "conversation_id": conversation_id,
                        "workflow_id": target_ref,
                        "status": "queued",
                        "turn_node_id": turn_node_id,
                        "implementation_attestation": runtime_implementation_attestation(runtime_kind),
                    })
                    .to_string(),
                )?;
                let priority_class = target
                    .get("priority_class")
                    .and_then(Value::as_str)
                    .unwrap_or("background");
                let mut payload = json!({
                    "contract_version": 1,
                    "kind": "workflow.run.execute",
                    "run_id": created_run_id,
                    "workflow_id": target_ref,
                    "conversation_id": conversation_id,
                    "turn_node_id": turn_node_id,
                    "user_id": user_id,
                    "initial_state": initial_state,
                    "priority_class": priority_class,
                    "token_budget": target.get("token_budget"),
                    "time_budget_ms": target.get("time_budget_ms"),
                    "runtime_kind": runtime_kind,
                    "worker_protocol": worker_protocol,
                    "join_node_ids": join_node_ids,
                    "start_join_mask": start_join_mask,
                    "start_node_id": start_node_id,
                    "op": start_op,
                    "runtime_routes": runtime_routes,
                    "expected_event_seq": created.seq,
                });
                if let Some(version) = start_op_version.filter(|version| version != "v1") {
                    payload["op_version"] = json!(version);
                }
                if let Some(environment_id) = start_environment_id.as_ref() {
                    payload["op_environment_id"] = json!(environment_id);
                }
                add_executor_target(
                    &mut payload,
                    start_op.as_deref().unwrap_or("noop"),
                    start_executor_target.as_deref(),
                );
                uow.project_lane_message(NewProjectedLaneMessage {
                    message_id,
                    namespace: "workflow".to_owned(),
                    purpose: "system".to_owned(),
                    inbox_id: "workflow-runtime".to_owned(),
                    conversation_id,
                    recipient_id: runtime_worker_recipient_for_operation(
                        start_op.as_deref().unwrap_or("noop"),
                        start_environment_id.as_deref(),
                        start_executor_target.as_deref(),
                    ),
                    sender_id: "rust-scheduler".to_owned(),
                    msg_type: "workflow.run.execute".to_owned(),
                    status: "pending".to_owned(),
                    created_at: now_seconds,
                    available_at: now_seconds,
                    run_id: Some(created_run_id.clone()),
                    step_id: Some(start_node_id),
                    correlation_id: Some(created_run_id.clone()),
                    payload_json: Some(payload.to_string()),
                    error_json: None,
                })?;
                run_id = Some(created_run_id);
            }
            let mut triggered_payload = input.payload.clone();
            triggered_payload.insert("trigger_type".to_owned(), json!(input.trigger_type));
            for (event_type, payload) in [
                (
                    "service.starting",
                    json!({"trigger_type": input.trigger_type}),
                ),
                ("service.triggered", Value::Object(triggered_payload)),
            ] {
                let (entity_id, event) =
                    service_event_node(service_id, event_type, &payload, timestamp);
                uow.append_raw_entity_event(
                    "workflow",
                    SqliteNewRawEntityEvent {
                        event_id: uuid::Uuid::new_v4().to_string(),
                        entity_kind: "node".to_owned(),
                        entity_id,
                        op: "ADD".to_owned(),
                        payload_json: event.to_string(),
                    },
                )?;
            }
            if let Some(run_id) = &run_id {
                let (entity_id, event) = service_event_node(
                    service_id,
                    "service.run_spawned",
                    &json!({
                        "trigger_type": input.trigger_type,
                        "run_id": run_id,
                        "workflow_id": target_ref,
                    }),
                    timestamp,
                );
                uow.append_raw_entity_event(
                    "workflow",
                    SqliteNewRawEntityEvent {
                        event_id: uuid::Uuid::new_v4().to_string(),
                        entity_kind: "node".to_owned(),
                        entity_id,
                        op: "ADD".to_owned(),
                        payload_json: event.to_string(),
                    },
                )?;
            }
            let history = uow.replay_raw_events("workflow", 0, usize::MAX)?;
            let projections = service_projection_values(history.into_iter().map(|event| {
                (
                    event.seq,
                    event.entity_kind,
                    event.entity_id,
                    event.op,
                    event.payload_json,
                )
            }))
            .map_err(kogwistar_store_sqlite::SqliteStoreError::TransactionAborted)?;
            let (last_seq, projection) = projections.get(service_id).ok_or_else(|| {
                kogwistar_store_sqlite::SqliteStoreError::TransactionAborted(
                    "KOGWISTAR_SERVICE_TRIGGER_INVALID".to_owned(),
                )
            })?;
            uow.replace_named_projection(
                "service_registry",
                service_id,
                service_projection_write(*last_seq, projection),
            )?;
            Ok(Some(projection.clone()))
        });
        match outcome {
            Ok(Some(projection)) => json_effect(StatusCode::OK, projection),
            Ok(None) => effect_error(
                StatusCode::TOO_MANY_REQUESTS,
                "KOGWISTAR_QUEUE_FULL",
                "workflow runtime queue is full",
            ),
            Err(error) if error.to_string().contains("KOGWISTAR_SERVICE_NOT_FOUND") => {
                effect_error(
                    StatusCode::NOT_FOUND,
                    "KOGWISTAR_SERVICE_NOT_FOUND",
                    format!("Unknown service_id: {service_id}"),
                )
            }
            Err(error) => effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_SERVICE_TRIGGER_CONFLICT",
                error.to_string(),
            ),
        }
    }

    fn declare_service(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        if !capability_allowed(
            request,
            &self.capabilities,
            "service.manage",
            &["service.manage"],
        ) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Declaring service requires service.manage capability",
            );
        }
        let mut input: ServiceDeclareInput = match serde_json::from_slice(&request.body) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error.to_string(),
                );
            }
        };
        if let Err(error) = validate_service_declare(&mut input) {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                error,
            );
        }
        let timestamp = now_ms();
        let service_id = input.service_id.clone();
        let outcome = self.store.transaction(|uow| {
            let (definition_id, definition) = service_definition_node(&input, timestamp);
            uow.append_raw_entity_event(
                "workflow",
                SqliteNewRawEntityEvent {
                    event_id: uuid::Uuid::new_v4().to_string(),
                    entity_kind: "node".to_owned(),
                    entity_id: definition_id,
                    op: "ADD".to_owned(),
                    payload_json: definition.to_string(),
                },
            )?;
            let lifecycle_type = if input.enabled {
                "service.enabled"
            } else {
                "service.stopped"
            };
            let (event_id, event) = service_event_node(
                &service_id,
                lifecycle_type,
                &json!({"enabled": input.enabled, "autostart": input.autostart}),
                timestamp,
            );
            uow.append_raw_entity_event(
                "workflow",
                SqliteNewRawEntityEvent {
                    event_id: uuid::Uuid::new_v4().to_string(),
                    entity_kind: "node".to_owned(),
                    entity_id: event_id,
                    op: "ADD".to_owned(),
                    payload_json: event.to_string(),
                },
            )?;
            if input.enabled {
                let (event_id, event) = service_event_node(
                    &service_id,
                    "service.degraded",
                    &json!({"health_status": "degraded"}),
                    timestamp,
                );
                uow.append_raw_entity_event(
                    "workflow",
                    SqliteNewRawEntityEvent {
                        event_id: uuid::Uuid::new_v4().to_string(),
                        entity_kind: "node".to_owned(),
                        entity_id: event_id,
                        op: "ADD".to_owned(),
                        payload_json: event.to_string(),
                    },
                )?;
            }
            let history = uow.replay_raw_events("workflow", 0, usize::MAX)?;
            let projections = service_projection_values(history.into_iter().map(|event| {
                (
                    event.seq,
                    event.entity_kind,
                    event.entity_id,
                    event.op,
                    event.payload_json,
                )
            }))
            .map_err(kogwistar_store_sqlite::SqliteStoreError::TransactionAborted)?;
            let (last_seq, projection) = projections.get(&service_id).ok_or_else(|| {
                kogwistar_store_sqlite::SqliteStoreError::TransactionAborted(
                    "KOGWISTAR_SERVICE_DECLARATION_INVALID".to_owned(),
                )
            })?;
            uow.replace_named_projection(
                "service_registry",
                &service_id,
                service_projection_write(*last_seq, projection),
            )?;
            Ok(projection.clone())
        });
        match outcome {
            Ok(projection) => json_effect(StatusCode::OK, projection),
            Err(error) => effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_SERVICE_DECLARATION_CONFLICT",
                error.to_string(),
            ),
        }
    }

    fn record_service_heartbeat(
        &self,
        request: &ApiEffectRequest,
        service_id: &str,
    ) -> ApiEffectResponse {
        #[derive(Deserialize)]
        struct HeartbeatInput {
            instance_id: String,
            #[serde(default)]
            payload: serde_json::Map<String, Value>,
        }
        if !capability_allowed(
            request,
            &self.capabilities,
            "service.heartbeat",
            &["service.heartbeat", "service.manage"],
        ) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Recording service heartbeat requires service.heartbeat capability",
            );
        }
        let input: HeartbeatInput = match serde_json::from_slice::<HeartbeatInput>(&request.body) {
            Ok(value) if !value.instance_id.trim().is_empty() => value,
            Ok(_) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    "instance_id must not be empty",
                );
            }
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error.to_string(),
                );
            }
        };
        let timestamp = now_ms();
        let outcome = self.store.transaction(|uow| {
            let current = uow
                .get_named_projection("service_registry", service_id)?
                .ok_or_else(|| {
                    kogwistar_store_sqlite::SqliteStoreError::TransactionAborted(
                        "KOGWISTAR_SERVICE_NOT_FOUND".to_owned(),
                    )
                })?;
            let mut projection = Value::Object(current.payload);
            let previous_health = projection["health_status"]
                .as_str()
                .unwrap_or_default()
                .to_owned();
            let previous_error = projection["last_error"].clone();
            let next_error = input
                .payload
                .get("last_error")
                .cloned()
                .unwrap_or(Value::Null);
            projection["instance_id"] = json!(input.instance_id);
            projection["last_heartbeat_ms"] = json!(timestamp);
            projection["lifecycle_status"] = json!("healthy");
            projection["health_status"] = json!("healthy");
            projection["updated_at_ms"] = json!(timestamp);
            projection["last_error"] = if next_error.is_null() {
                Value::Null
            } else {
                json!(next_error.as_str().unwrap_or_default())
            };
            let mut last_seq = current.last_authoritative_seq;
            if matches!(previous_health.as_str(), "degraded" | "failed" | "stopped") {
                let (entity_id, event) = service_event_node(
                    service_id,
                    "service.healthy",
                    &json!({
                        "health_status": "healthy",
                        "instance_id": input.instance_id,
                    }),
                    timestamp,
                );
                last_seq = uow
                    .append_raw_entity_event(
                        "workflow",
                        SqliteNewRawEntityEvent {
                            event_id: uuid::Uuid::new_v4().to_string(),
                            entity_kind: "node".to_owned(),
                            entity_id,
                            op: "ADD".to_owned(),
                            payload_json: event.to_string(),
                        },
                    )?
                    .event
                    .seq;
            }
            if previous_error != projection["last_error"] && !projection["last_error"].is_null() {
                let (entity_id, event) = service_event_node(
                    service_id,
                    "service.error_changed",
                    &json!({
                        "instance_id": input.instance_id,
                        "last_error": projection["last_error"],
                    }),
                    timestamp,
                );
                last_seq = uow
                    .append_raw_entity_event(
                        "workflow",
                        SqliteNewRawEntityEvent {
                            event_id: uuid::Uuid::new_v4().to_string(),
                            entity_kind: "node".to_owned(),
                            entity_id,
                            op: "ADD".to_owned(),
                            payload_json: event.to_string(),
                        },
                    )?
                    .event
                    .seq;
            }
            uow.replace_named_projection(
                "service_registry",
                service_id,
                service_projection_write(last_seq, &projection),
            )?;
            Ok(projection)
        });
        match outcome {
            Ok(projection) => json_effect(StatusCode::OK, projection),
            Err(error) if error.to_string().contains("KOGWISTAR_SERVICE_NOT_FOUND") => {
                effect_error(
                    StatusCode::NOT_FOUND,
                    "KOGWISTAR_SERVICE_NOT_FOUND",
                    format!("Unknown service_id: {service_id}"),
                )
            }
            Err(error) => effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_SERVICE_HEARTBEAT_CONFLICT",
                error.to_string(),
            ),
        }
    }

    fn set_service_enabled(
        &self,
        request: &ApiEffectRequest,
        service_id: &str,
        enabled: bool,
    ) -> ApiEffectResponse {
        if !capability_allowed(
            request,
            &self.capabilities,
            "service.manage",
            &["service.manage"],
        ) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Changing service lifecycle requires service.manage capability",
            );
        }
        let timestamp = now_ms();
        let outcome = self.store.transaction(|uow| {
            let history = uow.replay_raw_events("workflow", 0, usize::MAX)?;
            let definition = latest_service_definition(
                history.into_iter().map(|event| {
                    (
                        event.seq,
                        event.entity_kind,
                        event.entity_id,
                        event.op,
                        event.payload_json,
                    )
                }),
                service_id,
            )
            .ok_or_else(|| {
                kogwistar_store_sqlite::SqliteStoreError::TransactionAborted(
                    "KOGWISTAR_SERVICE_NOT_FOUND".to_owned(),
                )
            })?;
            let (definition_id, definition, event_id, event) =
                service_enabled_nodes(definition, service_id, enabled, timestamp)
                    .map_err(kogwistar_store_sqlite::SqliteStoreError::TransactionAborted)?;
            uow.append_raw_entity_event(
                "workflow",
                SqliteNewRawEntityEvent {
                    event_id: uuid::Uuid::new_v4().to_string(),
                    entity_kind: "node".to_owned(),
                    entity_id: definition_id,
                    op: "ADD".to_owned(),
                    payload_json: definition.to_string(),
                },
            )?;
            uow.append_raw_entity_event(
                "workflow",
                SqliteNewRawEntityEvent {
                    event_id: uuid::Uuid::new_v4().to_string(),
                    entity_kind: "node".to_owned(),
                    entity_id: event_id,
                    op: "ADD".to_owned(),
                    payload_json: event.to_string(),
                },
            )?;
            let history = uow.replay_raw_events("workflow", 0, usize::MAX)?;
            let projections = service_projection_values(history.into_iter().map(|event| {
                (
                    event.seq,
                    event.entity_kind,
                    event.entity_id,
                    event.op,
                    event.payload_json,
                )
            }))
            .map_err(kogwistar_store_sqlite::SqliteStoreError::TransactionAborted)?;
            let (last_seq, projection) = projections.get(service_id).ok_or_else(|| {
                kogwistar_store_sqlite::SqliteStoreError::TransactionAborted(
                    "KOGWISTAR_SERVICE_NOT_FOUND".to_owned(),
                )
            })?;
            uow.replace_named_projection(
                "service_registry",
                service_id,
                service_projection_write(*last_seq, projection),
            )?;
            Ok(projection.clone())
        });
        match outcome {
            Ok(projection) => json_effect(StatusCode::OK, projection),
            Err(error) if error.to_string().contains("KOGWISTAR_SERVICE_NOT_FOUND") => {
                effect_error(
                    StatusCode::NOT_FOUND,
                    "KOGWISTAR_SERVICE_NOT_FOUND",
                    format!("Unknown service_id: {service_id}"),
                )
            }
            Err(error) => effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_SERVICE_MUTATION_CONFLICT",
                error.to_string(),
            ),
        }
    }

    fn repair_service_projections(
        &self,
        request: &ApiEffectRequest,
        service_id: Option<&str>,
    ) -> ApiEffectResponse {
        if !capability_allowed(
            request,
            &self.capabilities,
            "service.manage",
            &["service.manage"],
        ) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Repairing service projections requires service.manage capability",
            );
        }
        let events = match self.store.replay_raw_events("workflow", 0, usize::MAX) {
            Ok(events) => events,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error.to_string(),
                );
            }
        };
        let projections = match service_projection_values(events.into_iter().map(|event| {
            (
                event.seq,
                event.entity_kind,
                event.entity_id,
                event.op,
                event.payload_json,
            )
        })) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_SERVICE_HISTORY_INVALID",
                    error,
                );
            }
        };
        if let Some(service_id) = service_id {
            let Some((last_seq, projection)) = projections.get(service_id) else {
                return effect_error(
                    StatusCode::NOT_FOUND,
                    "KOGWISTAR_SERVICE_NOT_FOUND",
                    format!("Unknown service_id: {service_id}"),
                );
            };
            return match self.store.replace_named_projection(
                "service_registry",
                service_id,
                service_projection_write(*last_seq, projection),
            ) {
                Ok(()) => json_effect(
                    StatusCode::OK,
                    json!({"service_id": service_id, "projection": projection}),
                ),
                Err(error) => effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error.to_string(),
                ),
            };
        }
        let limit = query_integer(&request.path_and_query, "limit", 10_000).clamp(0, 10_000);
        let selected = projections
            .into_iter()
            .take(limit as usize)
            .collect::<Vec<_>>();
        let outcome = self.store.transaction(|uow| {
            for (service_id, (last_seq, projection)) in &selected {
                uow.replace_named_projection(
                    "service_registry",
                    service_id,
                    service_projection_write(*last_seq, projection),
                )?;
            }
            Ok(())
        });
        match outcome {
            Ok(()) => json_effect(
                StatusCode::OK,
                json!({
                    "repaired_service_ids": selected
                        .into_iter()
                        .map(|(service_id, _)| service_id)
                        .collect::<Vec<_>>(),
                }),
            ),
            Err(error) => effect_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "KOGWISTAR_STORE_ERROR",
                error.to_string(),
            ),
        }
    }

    fn execute_sync(&self, request: ApiEffectRequest) -> ApiEffectResponse {
        if is_pending_server_cutover_route(&request.method, &request.path_and_query) {
            return effect_error(
                StatusCode::NOT_IMPLEMENTED,
                "KOGWISTAR_ROUTE_DEFERRED",
                "This route remains deferred to the Python application service",
            );
        }
        if request.method == "GET"
            && request.path_and_query.split('?').next() == Some("/api/workflow/tools/audit")
        {
            return effect_error(
                StatusCode::NOT_IMPLEMENTED,
                "KOGWISTAR_GRAPH_AUTHORITY_UNSUPPORTED",
                "The SQLite application service does not provide Rust graph authority",
            );
        }
        if request.method == "GET" && conversation_transcript_id(&request.path_and_query).is_some()
        {
            return effect_error(
                StatusCode::NOT_IMPLEMENTED,
                "KOGWISTAR_GRAPH_AUTHORITY_UNSUPPORTED",
                "The SQLite application service does not provide Rust graph authority",
            );
        }
        if request.method == "GET" && conversation_detail_id(&request.path_and_query).is_some() {
            return effect_error(
                StatusCode::NOT_IMPLEMENTED,
                "KOGWISTAR_GRAPH_AUTHORITY_UNSUPPORTED",
                "The SQLite application service does not provide Rust graph authority",
            );
        }
        if request.method == "GET"
            && latest_snapshot_conversation_id(&request.path_and_query).is_some()
        {
            return effect_error(
                StatusCode::NOT_IMPLEMENTED,
                "KOGWISTAR_GRAPH_AUTHORITY_UNSUPPORTED",
                "The SQLite application service does not provide Rust graph authority",
            );
        }
        if let Some(response) = visualization_shell_effect(&request) {
            return response;
        }
        if request.method == "POST" && request.path_and_query == INTERNAL_DOCUMENT_PROJECTION_ROUTE
        {
            return effect_error(
                StatusCode::NOT_IMPLEMENTED,
                "KOGWISTAR_DOCUMENT_AUTHORITY_UNSUPPORTED",
                "The SQLite application service does not provide document projection authority",
            );
        }
        if request.method == "POST"
            && matches!(
                request.path_and_query.split('?').next(),
                Some(INTERNAL_GRAPH_FIND_EDGES_ROUTE)
                    | Some(INTERNAL_GRAPH_NEIGHBORS_ROUTE)
                    | Some(INTERNAL_GRAPH_VECTOR_QUERY_ROUTE)
                    | Some(INTERNAL_GRAPH_SHORTEST_PATH_ROUTE)
                    | Some(INTERNAL_GRAPH_K_HOP_ROUTE)
                    | Some(INTERNAL_CONVERSATION_TRANSCRIPT_ROUTE)
                    | Some(INTERNAL_CONVERSATION_CONTEXT_SNAPSHOT_ROUTE)
                    | Some(INTERNAL_GRAPH_MUTATIONS_ROUTE)
            )
        {
            return effect_error(
                StatusCode::NOT_IMPLEMENTED,
                "KOGWISTAR_GRAPH_AUTHORITY_UNSUPPORTED",
                "The SQLite application service does not provide Rust graph authority",
            );
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
            return self.syscall_effect(&request, &op);
        }
        if let Some(response) = designer_capabilities_effect(&request, &self.capabilities) {
            return response;
        }
        if let Some((workflow_id, route)) = workflow_design_route(&request.path_and_query) {
            return self.workflow_design_effect(&request, &workflow_id, route);
        }
        if let Some(response) = self.service_read(&request) {
            return response;
        }
        if request.method == "POST"
            && request.path_and_query.split('?').next() == Some("/api/workflow/services")
        {
            return self.declare_service(&request);
        }
        if request.method == "POST"
            && let Some(service_id) = service_trigger_route(&request.path_and_query)
        {
            return self.trigger_service(&request, &service_id);
        }
        if request.method == "POST"
            && let Some(service_id) = service_repair_route(&request.path_and_query)
        {
            return self.repair_service_projections(&request, service_id.as_deref());
        }
        if request.method == "POST"
            && let Some((service_id, enabled)) = service_enabled_route(&request.path_and_query)
        {
            return self.set_service_enabled(&request, &service_id, enabled);
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
            return self.record_service_heartbeat(&request, service_id);
        }
        if request.method == "POST"
            && let Some(run_id) = dead_letter_replay_run_id(&request.path_and_query)
        {
            return self.replay_dead_letter(&request, &run_id);
        }
        if request.method == "POST"
            && request.path_and_query.split('?').next()
                == Some("/api/workflow/messages/repair-orphans")
        {
            return self.repair_orphaned_messages(&request);
        }
        if request.method == "POST" && request.path_and_query == INTERNAL_RUNTIME_CLAIM_ROUTE {
            return self.claim_runtime_work(&request);
        }
        if request.method == "POST" && request.path_and_query == INTERNAL_RUNTIME_RESULTS_ROUTE {
            return self.apply_runtime_result(&request);
        }
        if request.method == "POST"
            && request.path_and_query == INTERNAL_RUNTIME_NATIVE_EXECUTE_ROUTE
        {
            return self.execute_native_runtime_step(&request);
        }
        if request.method == "POST"
            && request.path_and_query.split('?').next() == Some("/api/workflow/runs")
        {
            return self.submit_run(&request);
        }
        if request.method == "GET"
            && let Some(path) = operational_path(&request.path_and_query)
        {
            let limit =
                query_integer(&request.path_and_query, "limit", 200).clamp(1, 10_000) as usize;
            let runs = match self.store.list_server_runs(None, None, None, 10_000) {
                Ok(runs) => runs,
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
                match self.store.list_server_run_events(&run.run_id, 0, limit) {
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
            return operational_effect(&request, path, runs, lanes, events);
        }
        let Some((run_id, tail)) = runtime_run_route(&request.path_and_query) else {
            return unavailable_effect(request);
        };
        let run = match self.get_run(&run_id) {
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
                    error,
                );
            }
        };
        let tail = tail.iter().map(String::as_str).collect::<Vec<_>>();
        if request.method == "GET" && is_runtime_inspection_tail(&tail) {
            return match self.store.list_server_run_events(&run_id, 0, 200_000) {
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
                {
                    Ok(events) => json_effect(
                        StatusCode::OK,
                        json!({
                            "run_id": run_id,
                            "events": events.into_iter().map(server_run_event_value).collect::<Vec<_>>(),
                        }),
                    ),
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
            ("GET", ["resume-contract"]) => match self.store.read_recorded_runtime_state(
                &run_id,
                &run.workflow_id,
                &run.conversation_id,
            ) {
                Ok(Some(state)) if !state.frontier.suspended.is_empty() => {
                    json_effect(StatusCode::OK, resume_contract_value(state))
                }
                Ok(Some(_)) => effect_error(
                    StatusCode::CONFLICT,
                    "KOGWISTAR_RUNTIME_NOT_SUSPENDED",
                    "run has no suspended token",
                ),
                Ok(None) => effect_error(
                    StatusCode::NOT_FOUND,
                    "KOGWISTAR_RUNTIME_STATE_NOT_FOUND",
                    format!("No recorded runtime state for {run_id}"),
                ),
                Err(error) => effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error.to_string(),
                ),
            },
            ("POST", ["resume"]) => self.resume_runtime_run(&run_id, &request),
            ("POST", ["cancel"]) => match self.store.request_server_run_cancel(&run_id) {
                Ok(()) => match self.get_run(&run_id) {
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
                        error,
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
