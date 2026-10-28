impl PostgresRunApplicationService {
    fn service_run_owner_binding(
        requested_user_id: Option<&str>,
        trigger_binding: Option<&RuntimeOwnerBindingV1>,
    ) -> Result<(Option<String>, Option<RuntimeOwnerBindingV1>), &'static str> {
        let Some(requested_user_id) = requested_user_id
            .map(str::trim)
            .filter(|user_id| !user_id.is_empty())
        else {
            return Ok((None, None));
        };
        let Some(binding) = trigger_binding.filter(|binding| binding.user_id == requested_user_id)
        else {
            return Err("KOGWISTAR_RUN_OWNER_FORBIDDEN");
        };
        Ok((Some(binding.user_id.clone()), Some(binding.clone())))
    }

    async fn trigger_service(
        &self,
        request: &ApiEffectRequest,
        service_id: &str,
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
        let service_id_owned = service_id.to_owned();
        let owner_binding = runtime_owner_binding_v1(&request.principal, &self.capabilities);
        let outcome = self
            .store
            .transaction(move |uow| {
                Box::pin(async move {
                    let current = uow
                        .lock_named_projection("service_registry", &service_id_owned)
                        .await?
                        .ok_or_else(|| {
                            kogwistar_store_postgres::PostgresStoreError::TransactionAborted(
                                "KOGWISTAR_SERVICE_NOT_FOUND".to_owned(),
                            )
                        })?;
                    let current_value = Value::Object(current.payload);
                    if let Some(run_id) = current_value["current_child_run_id"]
                        .as_str()
                        .filter(|value| !value.is_empty())
                        && let Some(run) = uow.get_server_run(run_id).await?
                        && !matches!(run.status.as_str(), "succeeded" | "failed" | "cancelled")
                    {
                        return Ok(Some(current_value));
                    }
                    let history = uow.replay_raw_events("workflow", 0, usize::MAX).await?;
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
                        &service_id_owned,
                    )
                    .ok_or_else(|| {
                        kogwistar_store_postgres::PostgresStoreError::TransactionAborted(
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
                    let (target_kind, target_config) = service_definition_runtime(&definition)
                        .map_err(|message| {
                            kogwistar_store_postgres::PostgresStoreError::TransactionAborted(
                                message.to_owned(),
                            )
                        })?;
                    let target_ref = definition["properties"]["target_ref"]
                        .as_str()
                        .unwrap_or_default()
                        .to_owned();
                    let mut run_id = None;
                    if target_kind == "workflow" {
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
                            return Ok(None);
                        }
                        let target = target_config.as_object().cloned().unwrap_or_default();
                        let conversation_id = input
                            .payload
                            .get("conversation_id")
                            .and_then(Value::as_str)
                            .or_else(|| target.get("conversation_id").and_then(Value::as_str))
                            .map(str::to_owned)
                            .unwrap_or_else(|| format!("svc:{service_id_owned}"));
                        let turn_node_id = input
                            .payload
                            .get("turn_node_id")
                            .and_then(Value::as_str)
                            .or_else(|| target.get("turn_node_id").and_then(Value::as_str))
                            .map(str::to_owned)
                            .unwrap_or_else(|| format!("wf_turn|{}", uuid::Uuid::new_v4()));
                        let requested_user_id = input
                            .payload
                            .get("user_id")
                            .and_then(Value::as_str)
                            .map(str::trim)
                            .filter(|user_id| !user_id.is_empty())
                            .or_else(|| {
                                target
                                    .get("user_id")
                                    .and_then(Value::as_str)
                                    .map(str::trim)
                                    .filter(|user_id| !user_id.is_empty())
                            })
                            .map(str::to_owned);
                        let (user_id, run_owner_binding) =
                            Self::service_run_owner_binding(
                                requested_user_id.as_deref(),
                                owner_binding.as_ref(),
                            )
                            .map_err(|code| {
                                kogwistar_store_postgres::PostgresStoreError::TransactionAborted(
                                    code.to_owned(),
                                )
                            })?;
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
                            uow.get_named_projection("workflow_design", &target_ref)
                                .await?,
                            uow.get_workflow_design_snapshot(&target_ref, i64::MAX, 1)
                                .await?,
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
                        let start_environment_id =
                            node_environment_ids.get(&start_node_id).cloned();
                        let start_executor_target =
                            node_executor_targets.get(&start_node_id).cloned();
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
                                serde_json::to_value(&node_ops)
                                    .expect("runtime node ops serialize"),
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
                        })
                        .await?;
                        let created = uow
                            .append_server_run_event(
                                &created_run_id,
                                "run.created",
                                json!({
                                    "run_id": created_run_id,
                                    "run_kind": "workflow_runtime",
                                    "conversation_id": conversation_id,
                                    "workflow_id": target_ref,
                                    "status": "queued",
                                    "turn_node_id": turn_node_id,
                                    "runtime_owner_binding": run_owner_binding,
                                    "implementation_attestation": runtime_implementation_attestation(runtime_kind),
                                })
                                .to_string(),
                            )
                            .await?;
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
                        })
                        .await?;
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
                            service_event_node(&service_id_owned, event_type, &payload, timestamp);
                        uow.append_raw_entity_event(
                            "workflow",
                            PostgresNewRawEntityEvent {
                                event_id: uuid::Uuid::new_v4().to_string(),
                                entity_kind: "node".to_owned(),
                                entity_id,
                                op: "ADD".to_owned(),
                                payload_json: event.to_string(),
                            },
                        )
                        .await?;
                    }
                    if let Some(run_id) = &run_id {
                        let (entity_id, event) = service_event_node(
                            &service_id_owned,
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
                            PostgresNewRawEntityEvent {
                                event_id: uuid::Uuid::new_v4().to_string(),
                                entity_kind: "node".to_owned(),
                                entity_id,
                                op: "ADD".to_owned(),
                                payload_json: event.to_string(),
                            },
                        )
                        .await?;
                    }
                    let history = uow.replay_raw_events("workflow", 0, usize::MAX).await?;
                    let projections = service_projection_values(history.into_iter().map(|event| {
                        (
                            event.seq,
                            event.entity_kind,
                            event.entity_id,
                            event.op,
                            event.payload_json,
                        )
                    }))
                    .map_err(kogwistar_store_postgres::PostgresStoreError::TransactionAborted)?;
                    let (last_seq, projection) =
                        projections.get(&service_id_owned).ok_or_else(|| {
                            kogwistar_store_postgres::PostgresStoreError::TransactionAborted(
                                "KOGWISTAR_SERVICE_TRIGGER_INVALID".to_owned(),
                            )
                        })?;
                    uow.replace_named_projection(
                        "service_registry",
                        &service_id_owned,
                        service_projection_write(*last_seq, projection),
                    )
                    .await?;
                    Ok(Some(projection.clone()))
                })
            })
            .await;
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
            Err(error) if error.to_string().contains("KOGWISTAR_RUN_OWNER_FORBIDDEN") => {
                effect_error(
                    StatusCode::FORBIDDEN,
                    "KOGWISTAR_RUN_OWNER_FORBIDDEN",
                    "A service-triggered run owner must match the authenticated trigger principal",
                )
            }
            Err(error) => effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_SERVICE_TRIGGER_CONFLICT",
                error.to_string(),
            ),
        }
    }

    async fn declare_service(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
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
        let outcome = self
            .store
            .transaction(move |uow| {
                Box::pin(async move {
                    let (definition_id, definition) = service_definition_node(&input, timestamp);
                    uow.append_raw_entity_event(
                        "workflow",
                        PostgresNewRawEntityEvent {
                            event_id: uuid::Uuid::new_v4().to_string(),
                            entity_kind: "node".to_owned(),
                            entity_id: definition_id,
                            op: "ADD".to_owned(),
                            payload_json: definition.to_string(),
                        },
                    )
                    .await?;
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
                        PostgresNewRawEntityEvent {
                            event_id: uuid::Uuid::new_v4().to_string(),
                            entity_kind: "node".to_owned(),
                            entity_id: event_id,
                            op: "ADD".to_owned(),
                            payload_json: event.to_string(),
                        },
                    )
                    .await?;
                    if input.enabled {
                        let (event_id, event) = service_event_node(
                            &service_id,
                            "service.degraded",
                            &json!({"health_status": "degraded"}),
                            timestamp,
                        );
                        uow.append_raw_entity_event(
                            "workflow",
                            PostgresNewRawEntityEvent {
                                event_id: uuid::Uuid::new_v4().to_string(),
                                entity_kind: "node".to_owned(),
                                entity_id: event_id,
                                op: "ADD".to_owned(),
                                payload_json: event.to_string(),
                            },
                        )
                        .await?;
                    }
                    let history = uow.replay_raw_events("workflow", 0, usize::MAX).await?;
                    let projections = service_projection_values(history.into_iter().map(|event| {
                        (
                            event.seq,
                            event.entity_kind,
                            event.entity_id,
                            event.op,
                            event.payload_json,
                        )
                    }))
                    .map_err(kogwistar_store_postgres::PostgresStoreError::TransactionAborted)?;
                    let (last_seq, projection) = projections.get(&service_id).ok_or_else(|| {
                        kogwistar_store_postgres::PostgresStoreError::TransactionAborted(
                            "KOGWISTAR_SERVICE_DECLARATION_INVALID".to_owned(),
                        )
                    })?;
                    uow.replace_named_projection(
                        "service_registry",
                        &service_id,
                        service_projection_write(*last_seq, projection),
                    )
                    .await?;
                    Ok(projection.clone())
                })
            })
            .await;
        match outcome {
            Ok(projection) => json_effect(StatusCode::OK, projection),
            Err(error) => effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_SERVICE_DECLARATION_CONFLICT",
                error.to_string(),
            ),
        }
    }

    async fn record_service_heartbeat(
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
        let service_id_owned = service_id.to_owned();
        let outcome = self
            .store
            .transaction(move |uow| {
                Box::pin(async move {
                    let current = uow
                        .get_named_projection("service_registry", &service_id_owned)
                        .await?
                        .ok_or_else(|| {
                            kogwistar_store_postgres::PostgresStoreError::TransactionAborted(
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
                            &service_id_owned,
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
                                PostgresNewRawEntityEvent {
                                    event_id: uuid::Uuid::new_v4().to_string(),
                                    entity_kind: "node".to_owned(),
                                    entity_id,
                                    op: "ADD".to_owned(),
                                    payload_json: event.to_string(),
                                },
                            )
                            .await?
                            .event
                            .seq;
                    }
                    if previous_error != projection["last_error"]
                        && !projection["last_error"].is_null()
                    {
                        let (entity_id, event) = service_event_node(
                            &service_id_owned,
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
                                PostgresNewRawEntityEvent {
                                    event_id: uuid::Uuid::new_v4().to_string(),
                                    entity_kind: "node".to_owned(),
                                    entity_id,
                                    op: "ADD".to_owned(),
                                    payload_json: event.to_string(),
                                },
                            )
                            .await?
                            .event
                            .seq;
                    }
                    uow.replace_named_projection(
                        "service_registry",
                        &service_id_owned,
                        service_projection_write(last_seq, &projection),
                    )
                    .await?;
                    Ok(projection)
                })
            })
            .await;
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

    async fn set_service_enabled(
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
        let service_id_owned = service_id.to_owned();
        let outcome = self
            .store
            .transaction(move |uow| {
                Box::pin(async move {
                    let history = uow.replay_raw_events("workflow", 0, usize::MAX).await?;
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
                        &service_id_owned,
                    )
                    .ok_or_else(|| {
                        kogwistar_store_postgres::PostgresStoreError::TransactionAborted(
                            "KOGWISTAR_SERVICE_NOT_FOUND".to_owned(),
                        )
                    })?;
                    let (definition_id, definition, event_id, event) =
                        service_enabled_nodes(definition, &service_id_owned, enabled, timestamp)
                            .map_err(
                                kogwistar_store_postgres::PostgresStoreError::TransactionAborted,
                            )?;
                    uow.append_raw_entity_event(
                        "workflow",
                        PostgresNewRawEntityEvent {
                            event_id: uuid::Uuid::new_v4().to_string(),
                            entity_kind: "node".to_owned(),
                            entity_id: definition_id,
                            op: "ADD".to_owned(),
                            payload_json: definition.to_string(),
                        },
                    )
                    .await?;
                    uow.append_raw_entity_event(
                        "workflow",
                        PostgresNewRawEntityEvent {
                            event_id: uuid::Uuid::new_v4().to_string(),
                            entity_kind: "node".to_owned(),
                            entity_id: event_id,
                            op: "ADD".to_owned(),
                            payload_json: event.to_string(),
                        },
                    )
                    .await?;
                    let history = uow.replay_raw_events("workflow", 0, usize::MAX).await?;
                    let projections = service_projection_values(history.into_iter().map(|event| {
                        (
                            event.seq,
                            event.entity_kind,
                            event.entity_id,
                            event.op,
                            event.payload_json,
                        )
                    }))
                    .map_err(kogwistar_store_postgres::PostgresStoreError::TransactionAborted)?;
                    let (last_seq, projection) =
                        projections.get(&service_id_owned).ok_or_else(|| {
                            kogwistar_store_postgres::PostgresStoreError::TransactionAborted(
                                "KOGWISTAR_SERVICE_NOT_FOUND".to_owned(),
                            )
                        })?;
                    uow.replace_named_projection(
                        "service_registry",
                        &service_id_owned,
                        service_projection_write(*last_seq, projection),
                    )
                    .await?;
                    Ok(projection.clone())
                })
            })
            .await;
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

    async fn repair_service_projections(
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
        let events = match self
            .store
            .replay_raw_events("workflow", 0, usize::MAX)
            .await
        {
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
            return match self
                .store
                .replace_named_projection(
                    "service_registry",
                    service_id,
                    service_projection_write(*last_seq, projection),
                )
                .await
            {
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
        let writes = selected.clone();
        let outcome = self
            .store
            .transaction(move |uow| {
                Box::pin(async move {
                    for (service_id, (last_seq, projection)) in &writes {
                        uow.replace_named_projection(
                            "service_registry",
                            service_id,
                            service_projection_write(*last_seq, projection),
                        )
                        .await?;
                    }
                    Ok(())
                })
            })
            .await;
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

    async fn design_events(&self, workflow_id: &str) -> Result<Vec<EntityEvent>, String> {
        let namespace = format!("wf_design:{workflow_id}");
        self.store
            .replay_raw_events(&namespace, 0, usize::MAX)
            .await
            .map(|events| {
                events
                    .into_iter()
                    .map(|event| EntityEvent {
                        namespace: event.namespace,
                        seq: event.seq,
                        event_id: event.event_id,
                        entity_kind: event.entity_kind,
                        entity_id: event.entity_id,
                        op: event.op,
                        payload: serde_json::from_str(&event.payload_json).unwrap_or(Value::Null),
                    })
                    .collect()
            })
            .map_err(|error| error.to_string())
    }

    async fn workflow_design_effect(
        &self,
        request: &ApiEffectRequest,
        workflow_id: &str,
        route: DesignRoute,
    ) -> ApiEffectResponse {
        let required = if matches!(route, DesignRoute::Graph | DesignRoute::History) {
            "workflow.design.inspect"
        } else {
            "workflow.design.write"
        };
        if !capability_allowed(request, &self.capabilities, required, &[required]) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                format!("Workflow design action requires {required} capability"),
            );
        }
        let events = match self.design_events(workflow_id).await {
            Ok(events) => events,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error,
                );
            }
        };
        if route == DesignRoute::History {
            return self.workflow_design_history(request, workflow_id).await;
        }
        if route == DesignRoute::Graph {
            return match workflow_design_graph_value(workflow_id, &events) {
                Ok(value) => json_effect(StatusCode::OK, value),
                Err(error) => effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_WORKFLOW_DESIGN_HISTORY_INVALID",
                    error,
                ),
            };
        }
        let input = match parse_design_input(request) {
            Ok(input) => input,
            Err(response) => return response,
        };
        let designer = match designer_id(request, &input) {
            Ok(value) => value,
            Err(response) => return response,
        };
        let history = match workflow_design_history_value(workflow_id, events.clone(), "ready") {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_WORKFLOW_DESIGN_HISTORY_INVALID",
                    error,
                );
            }
        };
        let current = history["current_version"].as_i64().unwrap_or(0);
        let active_tip = history["active_tip_version"].as_i64().unwrap_or(0);
        let namespace = format!("wf_design:{workflow_id}");
        let timestamp = now_ms();
        let workflow_id_owned = workflow_id.to_owned();
        let outcome = self.store.transaction(move |uow| Box::pin(async move {
            match route {
                DesignRoute::Undo | DesignRoute::Redo => {
                    let active = history["versions"].as_array().cloned().unwrap_or_default();
                    let ids = active.iter().filter_map(|item| item["version"].as_i64()).collect::<Vec<_>>();
                    let target = if route == DesignRoute::Undo {
                        history["selected_versions"].as_array().and_then(|items| items.iter().find(|item| item["version"].as_i64() == Some(current)))
                            .and_then(|item| item["prev_version"].as_i64()).unwrap_or(0)
                    } else { ids.iter().position(|value| *value == current).and_then(|index| ids.get(index + 1)).copied().unwrap_or(current) };
                    if target == current || (route == DesignRoute::Undo && current == 0) { let mut value = history.clone(); value["status"] = json!("noop"); return Ok(value); }
                    let op = if route == DesignRoute::Undo { "UNDO_APPLIED" } else { "REDO_APPLIED" };
                    uow.append_raw_entity_event(&namespace, PostgresNewRawEntityEvent {
                        event_id: uuid::Uuid::new_v4().to_string(), entity_kind: "design_control".to_owned(), entity_id: workflow_id_owned.clone(), op: op.to_owned(),
                        payload_json: json!({"designer_id": designer, "source":"rest", "ts_ms":timestamp, "from_version":current, "to_version":target,
                            "target_seq":history["commits"][target.to_string()]["target_seq"].as_i64().unwrap_or(0)}).to_string(),
                    }).await?;
                    let folded_events = uow.replay_raw_events(&namespace, 0, usize::MAX).await?.into_iter().map(|event| EntityEvent {
                        namespace:event.namespace,seq:event.seq,event_id:event.event_id,entity_kind:event.entity_kind,entity_id:event.entity_id,op:event.op,
                        payload:serde_json::from_str(&event.payload_json).unwrap_or(Value::Null),
                    }).collect::<Vec<_>>();
                    let mut folded = workflow_design_history_value(&workflow_id_owned, folded_events, "ready")
                        .map_err(kogwistar_store_postgres::PostgresStoreError::TransactionAborted)?;
                    uow.replace_named_projection("workflow_design", &workflow_id_owned, workflow_design_projection_write(&folded)).await?;
                    folded["status"] = json!("ok"); Ok(folded)
                }
                DesignRoute::NodeUpsert | DesignRoute::NodeDelete(_) | DesignRoute::EdgeUpsert | DesignRoute::EdgeDelete(_) => {
                    let before = graph_at_version(&events, current);
                    let mut graph = before.clone();
                    let (action, entity_id, deleted) = mutate_design_graph(&workflow_id_owned, &route, &input, &mut graph)
                        .map_err(|response| kogwistar_store_postgres::PostgresStoreError::TransactionAborted(String::from_utf8_lossy(&response.body).into_owned()))?;
                    let branch = current < active_tip;
                    if branch { uow.append_raw_entity_event(&namespace, PostgresNewRawEntityEvent {
                        event_id:uuid::Uuid::new_v4().to_string(),entity_kind:"design_control".to_owned(),entity_id:workflow_id_owned.clone(),op:"BRANCH_DROPPED".to_owned(),
                        payload_json:json!({"designer_id":designer,"source":"rest","ts_ms":timestamp,"drop_from_version":current+1,"drop_to_version":active_tip,
                            "drop_from_seq":history["versions"].as_array().and_then(|items|items.iter().find(|item|item["version"].as_i64().unwrap_or(0)>current)).and_then(|item|item["seq"].as_i64()).unwrap_or(0),
                            "drop_to_seq":history["versions"].as_array().and_then(|items|items.last()).and_then(|item|item["seq"].as_i64()).unwrap_or(0)}).to_string(),
                    }).await?; }
                    let version = history["allocated_max_version"].as_i64().unwrap_or(0)+1;
                    let entity = design_entity_event(&action, &entity_id, &before, &graph);
                    let data_event = uow.append_raw_entity_event(&namespace, PostgresNewRawEntityEvent {
                        event_id:entity.event_id,entity_kind:entity.entity_kind,entity_id:entity.entity_id,op:entity.op,payload_json:entity.payload.to_string(),
                    }).await?;
                    uow.append_raw_entity_event(&namespace, PostgresNewRawEntityEvent {
                        event_id:uuid::Uuid::new_v4().to_string(),entity_kind:"design_control".to_owned(),entity_id:workflow_id_owned.clone(),op:"MUTATION_COMMITTED".to_owned(),
                        payload_json:json!({"designer_id":designer,"source":"rest","ts_ms":timestamp,"action":action,"entity_id":entity_id,"version":version,
                            "prev_version":current,"target_seq":data_event.event.seq}).to_string(),
                    }).await?;
                    uow.put_workflow_design_delta(&workflow_id_owned, WorkflowDesignDeltaWrite {
                        version,prev_version:current,target_seq:data_event.event.seq,
                        forward_json:visible_delta(&before,&graph).to_string(),inverse_json:visible_delta(&graph,&before).to_string(),schema_version:1,
                    }).await?;
                    if version % 50 == 0 { uow.put_workflow_design_snapshot(&workflow_id_owned, WorkflowDesignSnapshotWrite {
                        version,seq:data_event.event.seq,payload_json:graph.to_string(),schema_version:1,
                    }).await?; }
                    let folded_events = uow.replay_raw_events(&namespace,0,usize::MAX).await?.into_iter().map(|event|EntityEvent{
                        namespace:event.namespace,seq:event.seq,event_id:event.event_id,entity_kind:event.entity_kind,entity_id:event.entity_id,op:event.op,
                        payload:serde_json::from_str(&event.payload_json).unwrap_or(Value::Null),}).collect::<Vec<_>>();
                    let folded=workflow_design_history_value(&workflow_id_owned,folded_events,"ready").map_err(kogwistar_store_postgres::PostgresStoreError::TransactionAborted)?;
                    uow.replace_named_projection("workflow_design",&workflow_id_owned,workflow_design_projection_write(&folded)).await?;
                    Ok(json!({"workflow_id":workflow_id_owned,"namespace":namespace,
                        if deleted {if action=="node_delete"{"node_id"}else{"edge_id"}}else if action=="node_upsert"{"node_id"}else{"edge_id"}:entity_id,
                        "designer_id":designer,"deleted":deleted,"version":folded["current_version"],"seq":data_event.event.seq,"can_undo":folded["can_undo"],"can_redo":folded["can_redo"]}))
                }
                _ => unreachable!(),
            }
        })).await;
        match outcome {
            Ok(value) => json_effect(StatusCode::OK, value),
            Err(error)
                if error.to_string().contains("KOGWISTAR_NODE_NOT_FOUND")
                    || error.to_string().contains("KOGWISTAR_EDGE_NOT_FOUND") =>
            {
                effect_error(
                    StatusCode::NOT_FOUND,
                    "KOGWISTAR_DESIGN_ENTITY_NOT_FOUND",
                    error.to_string(),
                )
            }
            Err(error) => effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_WORKFLOW_DESIGN_CONFLICT",
                error.to_string(),
            ),
        }
    }

    async fn workflow_design_history(
        &self,
        request: &ApiEffectRequest,
        workflow_id: &str,
    ) -> ApiEffectResponse {
        if !capability_allowed(
            request,
            &self.capabilities,
            "workflow.design.inspect",
            &["workflow.design.inspect"],
        ) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Inspecting workflow design requires workflow.design.inspect capability",
            );
        }
        let namespace = format!("wf_design:{workflow_id}");
        let events = match self
            .store
            .replay_raw_events(&namespace, 0, usize::MAX)
            .await
        {
            Ok(events) => events
                .into_iter()
                .map(|event| EntityEvent {
                    namespace: event.namespace,
                    seq: event.seq,
                    event_id: event.event_id,
                    entity_kind: event.entity_kind,
                    entity_id: event.entity_id,
                    op: event.op,
                    payload: serde_json::from_str(&event.payload_json).unwrap_or(Value::Null),
                })
                .collect(),
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error.to_string(),
                );
            }
        };
        let history = match workflow_design_history_value(workflow_id, events, "ready") {
            Ok(history) => history,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_WORKFLOW_DESIGN_HISTORY_INVALID",
                    error,
                );
            }
        };
        match self
            .store
            .replace_named_projection(
                "workflow_design",
                workflow_id,
                workflow_design_projection_write(&history),
            )
            .await
        {
            Ok(()) => json_effect(StatusCode::OK, history),
            Err(error) => effect_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "KOGWISTAR_STORE_ERROR",
                error.to_string(),
            ),
        }
    }

    async fn service_read(&self, request: &ApiEffectRequest) -> Option<ApiEffectResponse> {
        if request.method != "GET" {
            return None;
        }
        let path = request
            .path_and_query
            .split('?')
            .next()
            .unwrap_or(&request.path_and_query);
        if path != "/api/workflow/services" && service_route(&request.path_and_query).is_none() {
            return None;
        }
        if !capability_allowed(
            request,
            &self.capabilities,
            "service.inspect",
            &["service.inspect", "project_view"],
        ) {
            return Some(effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Inspecting service requires service.inspect capability",
            ));
        }
        if path == "/api/workflow/services" {
            let limit =
                query_integer(&request.path_and_query, "limit", 200).clamp(0, 10_000) as usize;
            return Some(
                match self.store.list_named_projections("service_registry").await {
                    Ok(rows) => {
                        let mut services = rows
                            .into_iter()
                            .map(|row| Value::Object(row.payload))
                            .collect::<Vec<_>>();
                        services.sort_by(|left, right| {
                            left["service_id"]
                                .as_str()
                                .cmp(&right["service_id"].as_str())
                        });
                        services.truncate(limit);
                        json_effect(StatusCode::OK, json!({"services": services}))
                    }
                    Err(error) => effect_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "KOGWISTAR_STORE_ERROR",
                        error.to_string(),
                    ),
                },
            );
        }
        let (service_id, route) =
            service_route(&request.path_and_query).expect("validated service route");
        if route == ServiceRoute::Events {
            let limit =
                query_integer(&request.path_and_query, "limit", 500).clamp(1, 10_000) as usize;
            return Some(
                match self
                    .store
                    .replay_raw_events("workflow", 0, usize::MAX)
                    .await
                {
                    Ok(events) => json_effect(
                        StatusCode::OK,
                        json!({
                            "service_id": service_id,
                            "events": service_event_values(
                                events.into_iter().map(|event| (
                                    event.seq,
                                    event.entity_kind,
                                    event.entity_id,
                                    event.op,
                                    event.payload_json,
                                )),
                                &service_id,
                                limit,
                            ),
                        }),
                    ),
                    Err(error) => effect_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "KOGWISTAR_STORE_ERROR",
                        error.to_string(),
                    ),
                },
            );
        }
        Some(
            match self
                .store
                .get_named_projection("service_registry", &service_id)
                .await
            {
                Ok(Some(row)) => json_effect(StatusCode::OK, Value::Object(row.payload)),
                Ok(None) => effect_error(
                    StatusCode::NOT_FOUND,
                    "KOGWISTAR_SERVICE_NOT_FOUND",
                    format!("Unknown service_id: {service_id}"),
                ),
                Err(error) => effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error.to_string(),
                ),
            },
        )
    }

    async fn repair_orphaned_messages(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        if !capability_allowed(
            request,
            &self.capabilities,
            "project_view",
            &["project_view"],
        ) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Repairing claimed messages requires project_view capability",
            );
        }
        let inbox_id = query_string(&request.path_and_query, "inbox_id");
        let limit = query_integer(&request.path_and_query, "limit", 100).clamp(0, 10_000) as usize;
        match self
            .store
            .repair_orphaned_claimed_lane_messages("workflow", inbox_id.as_deref(), limit)
            .await
        {
            Ok(repaired) => json_effect(StatusCode::OK, json!({"repaired_message_ids": repaired})),
            Err(error) => effect_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "KOGWISTAR_STORE_ERROR",
                error.to_string(),
            ),
        }
    }

    async fn replay_dead_letter(
        &self,
        request: &ApiEffectRequest,
        run_id: &str,
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
                "Replaying dead letters requires service.manage capability",
            );
        }
        let lanes = match self
            .store
            .list_projected_lane_messages(LaneMessageFilter {
                namespace: Some("workflow".to_owned()),
                status: Some("dead-letter".to_owned()),
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
        let Some(lane) = lanes
            .into_iter()
            .find(|lane| lane.run_id.as_deref() == Some(run_id))
        else {
            return json_effect(StatusCode::OK, json!({"run_id": run_id, "replayed": false}));
        };
        match self
            .store
            .update_projected_lane_message_status(&lane.message_id, "pending", None)
            .await
        {
            Ok(()) => json_effect(
                StatusCode::OK,
                json!({
                    "run_id": run_id,
                    "replayed": true,
                    "dead_letter": lane_progress_value(&lane),
                }),
            ),
            Err(error) => effect_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "KOGWISTAR_STORE_ERROR",
                error.to_string(),
            ),
        }
    }
}
