impl PostgresRunApplicationService {
    pub fn from_dsn(dsn: &str, schema: &str) -> Result<Self, String> {
        let dsn = dsn
            .strip_prefix("postgresql+psycopg://")
            .map(|rest| format!("postgresql://{rest}"))
            .or_else(|| {
                dsn.strip_prefix("postgresql+psycopg2://")
                    .map(|rest| format!("postgresql://{rest}"))
            })
            .unwrap_or_else(|| dsn.to_owned());
        PostgresStore::from_dsn(&dsn, schema)
            .map(|store| Self {
                store,
                max_queue: runtime_max_queue(),
                capabilities: Arc::new(Mutex::new(CapabilityState::default())),
                native_operations: kogwistar_runtime::NativeOperationRegistry::default(),
            })
            .map_err(|error| error.to_string())
    }

    pub async fn ensure_schema(&self) -> Result<(), String> {
        self.store
            .ensure_schema()
            .await
            .map_err(|error| error.to_string())
    }

    /// Compose trusted application-owned Rust operations into the opt-in native
    /// workflow lane. The default registry remains unchanged for existing users.
    pub fn with_native_operations(
        mut self,
        native_operations: kogwistar_runtime::NativeOperationRegistry,
    ) -> Self {
        self.native_operations = native_operations;
        self
    }

    pub fn with_max_queue(mut self, max_queue: usize) -> Self {
        self.max_queue = max_queue.max(1);
        self
    }

    async fn submit_run(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        let input: RuntimeSubmitRun = match serde_json::from_slice(&request.body) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error.to_string(),
                );
            }
        };
        let workflow_id = input.workflow_id.trim().to_owned();
        let conversation_id = input.conversation_id.trim().to_owned();
        if workflow_id.is_empty() || conversation_id.is_empty() {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                "workflow_id and conversation_id are required",
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
        let message_id = format!("lane|{}", uuid::Uuid::new_v4());
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|value| value.as_secs() as i64)
            .unwrap_or(0);
        let priority_class = if input.priority_class.is_empty() {
            "foreground".to_owned()
        } else {
            input.priority_class.clone()
        };
        let runtime_kind = if input.runtime_kind.is_empty() {
            "sync".to_owned()
        } else {
            input.runtime_kind.clone()
        };
        let worker_protocol = worker_protocol_for_runtime_kind(&runtime_kind);
        let graph_plan = match (
            self.store
                .get_named_projection("workflow_design", &workflow_id)
                .await,
            self.store
                .get_workflow_design_snapshot(&workflow_id, i64::MAX, 1)
                .await,
        ) {
            (Ok(projection), Ok(snapshot)) => exact_runtime_graph_plan(projection, snapshot),
            _ => None,
        };
        if graph_plan
            .as_ref()
            .is_some_and(|plan| submitted_runtime_graph_conflicts(plan, &input))
        {
            return effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_RUNTIME_GRAPH_CONFLICT",
                "submitted frozen runtime graph differs from authoritative workflow design",
            );
        }
        let effective_routes = graph_plan
            .as_ref()
            .map(|plan| plan.routes.clone())
            .unwrap_or(input.runtime_routes);
        let effective_join_node_ids = graph_plan
            .as_ref()
            .map(|plan| plan.join_node_ids.clone())
            .unwrap_or(input.join_node_ids);
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
                    .filter(|value| !value.is_empty())
                    .map(str::to_owned)
            })
            .unwrap_or_else(|| "start".to_owned());
        let effective_node_ops = graph_plan
            .as_ref()
            .map(|plan| plan.node_ops.clone())
            .unwrap_or(input.node_ops);
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
                    input.node_versions
                }
            });
        let effective_node_environment_ids = graph_plan
            .as_ref()
            .map(|plan| plan.node_environment_ids.clone())
            .unwrap_or(input.node_environment_ids);
        let effective_node_executor_targets = graph_plan
            .as_ref()
            .map(|plan| plan.node_executor_targets.clone())
            .unwrap_or(input.node_executor_targets);
        // Keep PostgreSQL admission wire-identical to SQLite: the worker must
        // receive a frozen operation, never a null value requiring a node-id
        // resolver guess.
        let effective_start_op = effective_node_ops
            .get(&effective_start_node_id)
            .cloned()
            .unwrap_or_else(|| "noop".to_owned());
        let effective_start_op_version = effective_node_versions
            .get(&effective_start_node_id)
            .cloned()
            .unwrap_or_else(|| "v1".to_owned());
        let effective_start_environment_id = effective_node_environment_ids
            .get(&effective_start_node_id)
            .cloned();
        let effective_start_executor_target = effective_node_executor_targets
            .get(&effective_start_node_id)
            .cloned();
        let mut initial_state = input.initial_state;
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
                serde_json::to_value(&effective_routes).expect("static runtime routes serialize"),
            );
        }
        if !effective_node_ops.is_empty() {
            initial_state.insert(
                "_rt_node_ops".to_owned(),
                serde_json::to_value(&effective_node_ops).expect("runtime node ops serialize"),
            );
        }
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
        if !effective_node_environment_ids.is_empty() {
            initial_state.insert(
                "_rt_node_environment_ids".to_owned(),
                serde_json::to_value(&effective_node_environment_ids)
                    .expect("runtime node environment ids serialize"),
            );
        }
        initial_state.insert(
            "_rt_worker_protocol".to_owned(),
            Value::String(worker_protocol.to_owned()),
        );
        if let Some(value) = input.token_budget {
            initial_state.insert("token_budget".to_owned(), json!(value));
        }
        if let Some(value) = input.time_budget_ms {
            initial_state.insert("time_budget_ms".to_owned(), json!(value));
        }
        let response = json!({
            "run_id": run_id,
            "conversation_id": conversation_id,
            "workflow_id": workflow_id,
            "turn_node_id": turn_node_id,
            "status": "queued",
            "priority_class": priority_class,
            "token_budget": input.token_budget,
            "time_budget_ms": input.time_budget_ms,
            "admission": "accepted",
            "lane_message_id": message_id,
        });
        if let Ok(Some(existing)) = self.store.get_server_run(&run_id).await {
            let lanes = self
                .store
                .list_projected_lane_messages(LaneMessageFilter {
                    namespace: Some("workflow".to_owned()),
                    inbox_id: Some("workflow-runtime".to_owned()),
                    correlation_id: Some(run_id.clone()),
                    limit: 10,
                    ..LaneMessageFilter::default()
                })
                .await
                .unwrap_or_default();
            let mut expected_worker_payload = json!({
                "contract_version": 1,
                "kind": "workflow.run.execute",
                "run_id": run_id,
                "workflow_id": workflow_id,
                "conversation_id": conversation_id,
                "turn_node_id": turn_node_id,
                "user_id": input.user_id,
                "initial_state": initial_state,
                "priority_class": priority_class,
                "token_budget": input.token_budget,
                "time_budget_ms": input.time_budget_ms,
                "runtime_kind": runtime_kind,
                "worker_protocol": worker_protocol,
                "join_node_ids": effective_join_node_ids,
                "start_join_mask": effective_start_join_mask,
                "start_node_id": effective_start_node_id,
                "op": effective_start_op,
                "runtime_routes": effective_routes,
            });
            if effective_start_op_version != "v1" {
                expected_worker_payload["op_version"] = json!(effective_start_op_version);
            }
            if let Some(environment_id) = effective_start_environment_id.as_ref() {
                expected_worker_payload["op_environment_id"] = json!(environment_id);
            }
            add_executor_target(
                &mut expected_worker_payload,
                &effective_start_op,
                effective_start_executor_target.as_deref(),
            );
            return match runtime_submit_retry_response(&existing, &lanes, &expected_worker_payload)
            {
                Ok(Some(value)) => json_effect(StatusCode::ACCEPTED, value),
                Ok(None) | Err(_) => effect_error(
                    StatusCode::CONFLICT,
                    "KOGWISTAR_RUNTIME_ADMISSION_CONFLICT",
                    format!("run_id {run_id:?} already exists with different admission state"),
                ),
            };
        }
        if let Err((code, message)) = validate_native_operation_registry(
            &effective_node_ops,
            &effective_node_versions,
            &effective_node_executor_targets,
            &self.native_operations,
        ) {
            return effect_error(StatusCode::UNPROCESSABLE_ENTITY, code, message);
        }
        let max_queue = self.max_queue;
        let queued_priority_class = priority_class.clone();
        let queue_full_admission = if matches!(priority_class.as_str(), "background" | "batch") {
            "rejected"
        } else {
            "deferred"
        };
        let outcome = self
            .store
            .transaction(|uow| {
                Box::pin(async move {
                    uow.lock_runtime_queue_admission().await?;
                    let pending = uow
                        .list_projected_lane_messages(&LaneMessageFilter {
                            namespace: Some("workflow".to_owned()),
                            inbox_id: Some("workflow-runtime".to_owned()),
                            status: Some("pending".to_owned()),
                            limit: max_queue,
                            ..LaneMessageFilter::default()
                        })
                        .await?;
                    let claimed = uow
                        .list_projected_lane_messages(&LaneMessageFilter {
                            namespace: Some("workflow".to_owned()),
                            inbox_id: Some("workflow-runtime".to_owned()),
                            status: Some("claimed".to_owned()),
                            limit: max_queue,
                            ..LaneMessageFilter::default()
                        })
                        .await?;
                    if pending.len() + claimed.len() >= max_queue {
                        return Ok(false);
                    }
                    uow.create_server_run(ServerRunCreate {
                        run_id: run_id.clone(),
                        conversation_id: conversation_id.clone(),
                        workflow_id: workflow_id.clone(),
                        user_id: input.user_id.clone(),
                        user_turn_node_id: turn_node_id.clone(),
                        status: "queued".to_owned(),
                    })
                    .await?;
                    let created = uow
                        .append_server_run_event(
                            &run_id,
                            "run.created",
                            serde_json::to_string(&json!({
                                "run_id": run_id,
                                "run_kind": "workflow_runtime",
                                "conversation_id": conversation_id,
                                "workflow_id": workflow_id,
                                "status": "queued",
                                "turn_node_id": turn_node_id,
                                "implementation_attestation": runtime_implementation_attestation(&runtime_kind),
                            }))?,
                        )
                        .await?;
                    let mut payload = json!({
                        "contract_version": 1,
                        "kind": "workflow.run.execute",
                        "run_id": run_id,
                        "workflow_id": workflow_id,
                        "conversation_id": conversation_id,
                        "turn_node_id": turn_node_id,
                        "user_id": input.user_id,
                        "initial_state": initial_state,
                        "priority_class": queued_priority_class,
                        "token_budget": input.token_budget,
                        "time_budget_ms": input.time_budget_ms,
                        "runtime_kind": runtime_kind,
                        "worker_protocol": worker_protocol,
                        "join_node_ids": effective_join_node_ids,
                        "start_join_mask": effective_start_join_mask,
                        "start_node_id": effective_start_node_id,
                        "op": effective_start_op,
                        "runtime_routes": effective_routes,
                        "expected_event_seq": created.seq,
                    });
                    if effective_start_op_version != "v1" {
                        payload["op_version"] = json!(effective_start_op_version);
                    }
                    if let Some(environment_id) = effective_start_environment_id.as_ref() {
                        payload["op_environment_id"] = json!(environment_id);
                    }
                    add_executor_target(
                        &mut payload,
                        &effective_start_op,
                        effective_start_executor_target.as_deref(),
                    );
                    uow.project_lane_message(NewProjectedLaneMessage {
                        message_id,
                        namespace: "workflow".to_owned(),
                        purpose: "system".to_owned(),
                        inbox_id: "workflow-runtime".to_owned(),
                        conversation_id,
                        recipient_id: runtime_worker_recipient_for_operation(
                            &effective_start_op,
                            effective_start_environment_id.as_deref(),
                            effective_start_executor_target.as_deref(),
                        ),
                        sender_id: "rust-scheduler".to_owned(),
                        msg_type: "workflow.run.execute".to_owned(),
                        status: "pending".to_owned(),
                        created_at: now,
                        available_at: now,
                        run_id: Some(run_id.clone()),
                        step_id: Some(effective_start_node_id),
                        correlation_id: Some(run_id),
                        payload_json: Some(serde_json::to_string(&payload)?),
                        error_json: None,
                    })
                    .await?;
                    Ok(true)
                })
            })
            .await;
        match outcome {
            Ok(true) => json_effect(StatusCode::ACCEPTED, response),
            Ok(false) => json_effect(
                StatusCode::TOO_MANY_REQUESTS,
                json!({
                    "admission": queue_full_admission,
                    "reason": "queue_full",
                    "max_queue": max_queue,
                }),
            ),
            Err(error) => effect_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "KOGWISTAR_STORE_ERROR",
                error.to_string(),
            ),
        }
    }

    async fn claim_runtime_work(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        let input: RuntimeClaimRequest = match serde_json::from_slice(&request.body) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error.to_string(),
                );
            }
        };
        if input.claimed_by.trim().is_empty() || input.limit == 0 || input.lease_seconds <= 0 {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                "claimed_by, positive limit, and positive lease_seconds are required",
            );
        }
        let recipient_ids = match effective_runtime_claim_recipients(&input.recipient_ids) {
            Ok(values) => values,
            Err(message) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    message,
                );
            }
        };
        match self.claim_runtime_work_items(input, recipient_ids).await {
            Ok(work) => json_effect(StatusCode::OK, json!({"work": work})),
            Err(error) => effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_RUNTIME_CLAIM_CONFLICT",
                error,
            ),
        }
    }

    async fn claim_runtime_work_items(
        &self,
        input: RuntimeClaimRequest,
        recipient_ids: Vec<String>,
    ) -> Result<Vec<RuntimeClaimedWork>, String> {
        let claim_owner = format!("{}#{}", input.claimed_by, uuid::Uuid::new_v4());
        self.store
            .transaction(|uow| {
                Box::pin(async move {
                    let mut claimed = Vec::new();
                    for recipient_id in &recipient_ids {
                        let remaining = input.limit.saturating_sub(claimed.len());
                        if remaining == 0 {
                            break;
                        }
                        claimed.extend(
                            uow.claim_projected_lane_messages_filtered(LaneMessageClaimFilter {
                                namespace: "workflow".to_owned(),
                                inbox_id: "workflow-runtime".to_owned(),
                                claimed_by: claim_owner.clone(),
                                message_ids: None,
                                run_id: input.run_id.clone(),
                                msg_type: None,
                                recipient_id: Some(recipient_id.clone()),
                                limit: remaining,
                                lease_seconds: input.lease_seconds,
                            })
                            .await?,
                        );
                    }
                    let mut values = Vec::new();
                    for lane in claimed {
                        let mut payload: Value =
                            serde_json::from_str(lane.payload_json.as_deref().unwrap_or("{}"))?;
                        let run_id = lane.run_id.clone().unwrap_or_default();
                        if uow
                            .get_server_run(&run_id)
                            .await?
                            .is_some_and(|run| run.cancel_requested)
                        {
                            uow.update_projected_lane_message_status(
                                &lane.message_id,
                                "cancelled",
                                None,
                            )
                            .await?;
                            continue;
                        }
                        let starts_runtime = lane.msg_type == "workflow.run.execute"
                            && payload["runtime_started"].as_bool() != Some(true);
                        let expected_event_seq = if starts_runtime {
                            uow.apply_recorded_runtime_transition(
                                runtime_start_transition(&payload, &run_id),
                                false,
                            )
                            .await?
                            .event_seq
                        } else {
                            payload["expected_event_seq"].as_i64().ok_or_else(|| {
                                kogwistar_store_postgres::PostgresStoreError::RecordedRuntimeConflict(
                                    format!(
                                        "continuation lane message {:?} lacks expected_event_seq",
                                        lane.message_id
                                    ),
                                )
                            })?
                        };
                        payload["expected_event_seq"] = json!(expected_event_seq);
                        if lane.msg_type == "workflow.run.execute" {
                            payload["runtime_started"] = json!(true);
                        }
                        uow.update_projected_lane_message_payload(
                            &lane.message_id,
                            serde_json::to_string(&payload)?,
                        )
                        .await?;
                        values.push(runtime_claimed_work(
                            lane,
                            &claim_owner,
                            run_id,
                            payload,
                            expected_event_seq,
                        ));
                    }
                    Ok(values)
                })
            })
            .await
            .map_err(|error| error.to_string())
    }

    async fn apply_runtime_result(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        let input: RuntimeWorkerResultRequest = match serde_json::from_slice(&request.body) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error.to_string(),
                );
            }
        };
        self.apply_runtime_result_input(input).await
    }

    async fn apply_runtime_result_input(
        &self,
        input: RuntimeWorkerResultRequest,
    ) -> ApiEffectResponse {
        if input.transition.is_some() == input.effect.is_some() {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                "exactly one of transition or effect is required",
            );
        }
        if let Some(effect) = input.effect.as_ref()
            && let Err(message) =
                validate_python_worker_trace_binding(&effect.trace_events, &input.handoff.claimed_by)
        {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_WORKER_TRACE",
                message,
            );
        }
        let outcome = self
            .store
            .transaction(|uow| {
                Box::pin(async move {
                    let handoff = input.handoff;
                    let (applied, effect_id, terminal_result, usage, trace_events) =
                        if let Some(transition) = input.transition {
                            let effect_id = transition.transition_id.clone();
                            let terminal_result = transition.result.clone();
                            let applied = uow
                                .apply_claimed_recorded_runtime_transition(
                                    handoff, transition, false,
                                )
                                .await?;
                            (applied, effect_id, terminal_result, None, Vec::new())
                        } else {
                            let effect = input.effect.expect("effect checked above");
                            let effect_id = effect.effect_id.clone();
                            let terminal_result = effect.result.clone();
                            let usage = effect.usage.clone();
                            let trace_events = effect.trace_events.clone();
                            let applied = uow
                                .apply_claimed_recorded_worker_effect(handoff, effect)
                                .await?;
                            (applied, effect_id, terminal_result, usage, trace_events)
                        };
                    let state = applied.reduced.state.clone();
                    let frontier = &state.frontier;
                    let unfinished = !frontier.pending.is_empty()
                        || !frontier.suspended.is_empty()
                        || frontier.join_outstanding.iter().any(|value| *value != 0)
                        || frontier
                            .join_waiters
                            .values()
                            .any(|waiters| !waiters.is_empty());
                    let applied_idempotent = applied.idempotent;
                    let result = if unfinished || state.status.is_terminal() {
                        for (node_id, join_mask, token_id, parent_token_id) in &frontier.pending {
                            let existing_lanes = uow
                                .list_projected_lane_messages(&LaneMessageFilter {
                                    namespace: Some("workflow".to_owned()),
                                    inbox_id: Some("workflow-runtime".to_owned()),
                                    correlation_id: Some(state.run_id.clone()),
                                    ..LaneMessageFilter::default()
                                })
                                .await?;
                            if active_runtime_lane_covers_token(&existing_lanes, token_id) {
                                continue;
                            }
                            let next_step_seq = state.last_step_seq.saturating_add(1);
                            let message_id = format!(
                                "lane|{}",
                                kogwistar_contracts::stable_id(
                                    "runtime.worker.request",
                                    &[
                                        state.run_id.clone(),
                                        token_id.clone(),
                                        node_id.clone(),
                                        next_step_seq.to_string(),
                                    ],
                                )
                            );
                            let now = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .map(|value| value.as_secs() as i64)
                                .unwrap_or(0);
                            if uow.get_projected_lane_message(&message_id).await?.is_some() {
                                continue;
                            }
                            let op = state
                                .state
                                .get("_rt_node_ops")
                                .and_then(Value::as_object)
                                .and_then(|ops| ops.get(node_id))
                                .cloned()
                                .unwrap_or_else(|| Value::String("noop".to_owned()));
                            let payload = RuntimeStepExecutePayload::from_recorded_state(
                                &state,
                                RuntimeStepExecuteRequest {
                                    node_id: node_id.clone(),
                                    op,
                                    join_mask: *join_mask,
                                    token_id: token_id.clone(),
                                    parent_token_id: parent_token_id.clone(),
                                    step_seq: next_step_seq,
                                    expected_event_seq: applied.event_seq,
                                    resume_effect: None,
                                },
                            );
                            uow.project_lane_message(NewProjectedLaneMessage {
                                message_id,
                                namespace: "workflow".to_owned(),
                                purpose: "system".to_owned(),
                                inbox_id: "workflow-runtime".to_owned(),
                                conversation_id: state.conversation_id.clone(),
                                recipient_id: runtime_worker_recipient_for_state(&state, node_id),
                                sender_id: "rust-scheduler".to_owned(),
                                msg_type: "workflow.step.execute".to_owned(),
                                status: "pending".to_owned(),
                                created_at: now,
                                available_at: now,
                                run_id: Some(state.run_id.clone()),
                                step_id: Some(node_id.clone()),
                                correlation_id: Some(state.run_id.clone()),
                                payload_json: Some(serde_json::to_string(&payload)?),
                                error_json: None,
                            })
                            .await?;
                        }
                        applied
                    } else {
                        uow.apply_recorded_runtime_transition(
                            RecordedRuntimeTransition {
                                contract_version: RECORDED_RUNTIME_CONTRACT_VERSION,
                                transition_id: format!("complete-{effect_id}"),
                                expected_event_seq: applied.event_seq,
                                kind: RecordedTransitionKind::Complete,
                                run_id: state.run_id,
                                workflow_id: state.workflow_id,
                                conversation_id: state.conversation_id,
                                user_id: None,
                                user_turn_node_id: None,
                                step_seq: state.last_step_seq,
                                node_id: state.last_node_id,
                                token_id: state.last_token_id,
                                parent_token_id: state.last_parent_token_id,
                                initial_state: None,
                                state_update: Vec::new(),
                                update: None,
                                state_schema: serde_json::Map::new(),
                                frontier: Some(RuntimeFrontier::default()),
                                result: terminal_result,
                                wait_reason: None,
                                resume_payload: None,
                                errors: Vec::new(),
                            },
                            false,
                        )
                        .await?
                    };
                    if !applied_idempotent {
                        if let Some(usage) = usage {
                            uow.append_server_run_event(
                                &result.reduced.state.run_id,
                                "workflow.usage.v1",
                                serde_json::to_string(&json!({
                                    "run_id": result.reduced.state.run_id,
                                    "step_id": result.reduced.state.last_node_id,
                                    "token_id": result.reduced.state.last_token_id,
                                    "effect_id": effect_id,
                                    "usage": usage,
                                }))?,
                            )
                            .await?;
                        }
                        for (trace_index, trace) in trace_events.into_iter().enumerate() {
                            uow.append_server_run_event(
                                &result.reduced.state.run_id,
                                "workflow.trace.v1",
                                serde_json::to_string(&json!({
                                    "run_id": result.reduced.state.run_id,
                                    "step_id": result.reduced.state.last_node_id,
                                    "token_id": result.reduced.state.last_token_id,
                                    "effect_id": effect_id,
                                    "trace_index": trace_index,
                                    "trace": trace,
                                }))?,
                            )
                            .await?;
                        }
                    }
                    Ok(result)
                })
            })
            .await;
        match outcome {
            Ok(result) => json_effect(
                StatusCode::OK,
                serde_json::to_value(result).expect("recorded runtime result serializes"),
            ),
            Err(error) => effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_RUNTIME_RESULT_CONFLICT",
                error.to_string(),
            ),
        }
    }

    async fn resume_runtime_run(
        &self,
        run_id: &str,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        let input: RuntimeResumeRequest = match serde_json::from_slice(&request.body) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error.to_string(),
                );
            }
        };
        match self
            .store
            .resume_recorded_runtime_token(
                run_id,
                &input.workflow_id,
                &input.conversation_id,
                &input.suspended_node_id,
                &input.suspended_token_id,
                Some(input.client_result),
            )
            .await
        {
            Ok(result) => json_effect(
                StatusCode::OK,
                serde_json::to_value(result).expect("recorded runtime result serializes"),
            ),
            Err(error) => effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_RUNTIME_RESUME_CONFLICT",
                error.to_string(),
            ),
        }
    }

    async fn apply_internal_graph_mutations(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        if !self
            .store
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
        let applied = if input.index_jobs.is_empty() {
            self.store.apply_graph_mutations_atomic(mutations).await
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
            self.store
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
        if request.method == "POST" && request.path_and_query == INTERNAL_DOCUMENT_PROJECTION_ROUTE
        {
            return self.document_projection_effect(&request).await;
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
        if request.method == "POST" && request.path_and_query == INTERNAL_GRAPH_NEIGHBORS_ROUTE {
            return self.graph_neighbors_effect(&request).await;
        }
        if request.method == "POST" && request.path_and_query == INTERNAL_GRAPH_VECTOR_QUERY_ROUTE {
            return self.graph_vector_query_effect(&request).await;
        }
        if request.method == "POST" && request.path_and_query == INTERNAL_GRAPH_FIND_EDGES_ROUTE {
            return self.graph_find_edges_effect(&request).await;
        }
        if request.method == "POST" && request.path_and_query == INTERNAL_GRAPH_SHORTEST_PATH_ROUTE {
            return self.graph_shortest_path_effect(&request).await;
        }
        if request.method == "POST" && request.path_and_query == INTERNAL_GRAPH_K_HOP_ROUTE {
            return self.graph_k_hop_effect(&request).await;
        }
        if request.method == "POST"
            && request.path_and_query.split('?').next() == Some("/api/workflow/runs")
        {
            return self.submit_run(&request).await;
        }
        if request.method == "GET"
            && let Some(path) = operational_path(&request.path_and_query)
        {
            let limit =
                query_integer(&request.path_and_query, "limit", 200).clamp(1, 10_000) as usize;
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
                match self
                    .store
                    .list_server_run_events(&run.run_id, 0, limit)
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
            return operational_effect(&request, path, runs, lanes, events);
        }
        let Some((run_id, tail)) = runtime_run_route(&request.path_and_query) else {
            return unavailable_effect(request);
        };
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
        let tail = tail.iter().map(String::as_str).collect::<Vec<_>>();
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
            ("POST", ["resume"]) => self.resume_runtime_run(&run_id, &request).await,
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
