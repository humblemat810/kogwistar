// Native run admission, worker results, and resume operations.
// Python counterpart: kogwistar.server.runtime_api routes backed by
// ChatRunService/runtime execution helpers.
impl PostgresRunApplicationService {
    async fn submit_run(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        self.submit_run_with_conversation_admission(request, None)
            .await
    }

    async fn submit_run_with_conversation_admission(
        &self,
        request: &ApiEffectRequest,
        conversation_admission: Option<PreparedConversationAdmission>,
    ) -> ApiEffectResponse {
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
        let authenticated_owner = authenticated_run_owner_id(&request.principal);
        if input.user_id.as_deref().is_some_and(|requested| {
            authenticated_owner.is_none_or(|owner| owner != requested.trim())
        }) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_RUN_OWNER_FORBIDDEN",
                "run user_id must match the authenticated principal",
            );
        }
        let run_user_id = authenticated_owner.map(str::to_owned);
        let owner_binding = runtime_owner_binding_v1(&request.principal, &self.capabilities);
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
        let graph_projection = self
            .store
            .get_named_projection("workflow_design", &workflow_id)
            .await
            .ok()
            .flatten();
        let graph_snapshot = self
            .store
            .get_workflow_design_snapshot(&workflow_id, i64::MAX, 1)
            .await
            .ok()
            .flatten();
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
        if graph_plan.is_none()
            && !runtime_worker_identity_map_is_valid(
                &input.node_worker_identities,
                &input.node_environment_ids,
                &input.node_executor_targets,
            )
        {
            return effect_error(
                StatusCode::BAD_REQUEST,
                "KOGWISTAR_INVALID_WORKER_IDENTITY",
                "submitted worker identity pins must be valid and match Python executor environments",
            );
        }
        if !answer_workflow_admission_ready(
            &workflow_id,
            graph_plan.as_ref(),
            &self.native_operations,
        ) {
            return effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_ANSWER_WORKFLOW_NOT_READY",
                "the selected answer workflow has no complete authoritative plan with a compatible executor",
            );
        }
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
        let effective_node_worker_identities = graph_plan
            .as_ref()
            .map(|plan| plan.node_worker_identities.clone())
            .unwrap_or(input.node_worker_identities);
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
        let effective_start_worker_identity = effective_node_worker_identities
            .get(&effective_start_node_id)
            .cloned();
        let effective_start_executor_target = effective_node_executor_targets
            .get(&effective_start_node_id)
            .cloned();
        let mut initial_state = input.initial_state;
        initial_state.remove("_rt_graph_scope");
        initial_state.remove("_rt_run_owner_id");
        initial_state.remove("_rt_node_worker_identities");
        initial_state.insert(
            "_rt_graph_scope".to_owned(),
            security_scope_value(&request.principal),
        );
        if let Some(owner_id) = run_user_id.as_deref() {
            initial_state.insert("_rt_run_owner_id".to_owned(), json!(owner_id));
        }
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
        if !effective_node_worker_identities.is_empty() {
            initial_state.insert(
                "_rt_node_worker_identities".to_owned(),
                serde_json::to_value(&effective_node_worker_identities)
                    .expect("runtime node worker identities serialize"),
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
                "user_id": run_user_id,
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
            if let Some(identity) = effective_start_worker_identity.as_ref() {
                expected_worker_payload["required_worker_identity"] = identity.clone();
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
        let admission_actor_user_id = conversation_admission
            .as_ref()
            .map(|admission| admission.actor_user_id.clone());
        let conversation_schema = self.conversation_store.schema().to_owned();
        let outcome = self
            .store
            .transaction_with_graph_schemas(&[&conversation_schema], |uow| {
                let conversation_schema = conversation_schema.clone();
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
                        return Ok(RuntimeAdmissionOutcome::QueueFull);
                    }
                    if let Some(admission) = conversation_admission.as_ref() {
                        uow.lock_entity_identity(
                            &admission.scope.namespace,
                            "conversation",
                            &admission.conversation_id,
                        )
                        .await?;
                        let current_nodes = uow
                            .graph_projection_records_for_update_in_schema(
                                &conversation_schema,
                                GraphProjectionRead {
                                scope: admission.scope.clone(),
                                table: "gke_nodes".to_owned(),
                                ids: None,
                                metadata: MetadataFilter {
                                    equals: BTreeMap::from([(
                                        "conversation_id".to_owned(),
                                        json!(admission.conversation_id),
                                    )]),
                                },
                                limit: 10_000,
                                },
                            )
                            .await?;
                        let acl_nodes = uow
                            .graph_projection_records_in_schema(
                                &conversation_schema,
                                GraphProjectionRead {
                                scope: admission.scope.clone(),
                                table: "gke_nodes".to_owned(),
                                ids: None,
                                metadata: MetadataFilter {
                                    equals: BTreeMap::from([(
                                        "entity_type".to_owned(),
                                        json!("acl_record"),
                                    )]),
                                },
                                limit: 10_000,
                                },
                            )
                            .await?;
                        let mut acl_input = current_nodes;
                        acl_input.extend(acl_nodes);
                        let (visible_nodes, _) = graph_records_visible_by_acl(
                            &acl_input,
                            &[],
                            "conversation",
                            &admission.acl_principal,
                        );
                        let owner_matches = visible_nodes.iter().any(|record| {
                            conversation_record_string(record, "entity_type").as_deref()
                                == Some("conversation_start")
                                && conversation_record_string(record, "user_id").as_deref()
                                    == Some(admission.owner_id.as_str())
                        });
                        let mut current_tail = visible_nodes
                            .iter()
                            .filter(|record| {
                                matches!(
                                    conversation_record_string(record, "entity_type").as_deref(),
                                    Some("conversation_start" | "conversation_turn" | "conversation_summary" | "assistant_turn")
                                ) && conversation_record_value(record, "in_conversation_chain")
                                    .and_then(|value| value.as_bool())
                                    .unwrap_or(true)
                                    && !matches!(
                                        conversation_record_string(record, "lifecycle_status")
                                            .as_deref(),
                                        Some("tombstoned" | "deleted")
                                    )
                            })
                            .collect::<Vec<_>>();
                        current_tail.sort_by_key(|record| {
                            (
                                conversation_record_i64(record, "turn_index").unwrap_or(-1),
                                record.id.clone(),
                            )
                        });
                        let actual_tail_id = current_tail.last().map(|record| record.id.as_str());
                        if !owner_matches || actual_tail_id != admission.expected_tail_id.as_deref()
                        {
                            return Ok(RuntimeAdmissionOutcome::ConversationChanged);
                        }
                    }
                    uow.create_server_run(ServerRunCreate {
                        run_id: run_id.clone(),
                        conversation_id: conversation_id.clone(),
                        workflow_id: workflow_id.clone(),
                        user_id: run_user_id.clone(),
                        user_turn_node_id: turn_node_id.clone(),
                        status: "queued".to_owned(),
                    })
                    .await?;
                    if let Some(admission) = conversation_admission.as_ref() {
                        let graph_mutations = admission
                            .mutations
                            .iter()
                            .map(|(mutation, _)| mutation.clone())
                            .collect::<Vec<_>>();
                        let (acl_mutations, acl_index_jobs) =
                            resolve_default_acl_graph_effects_in_schema(
                                uow,
                                &conversation_schema,
                                &admission.scope,
                                &graph_mutations,
                                "conversation",
                                &admission.owner_id,
                                admission.acl_principal.security_scope.as_deref(),
                                &run_id,
                            )
                            .await?;
                        for (mutation, envelope) in &admission.mutations {
                            kogwistar_store::validate_typed_graph_mutation(mutation, envelope)?;
                            uow.apply_graph_mutation_in_schema(
                                &conversation_schema,
                                mutation.clone(),
                            )
                            .await?;
                        }
                        for mutation in acl_mutations {
                            uow.apply_graph_mutation_in_schema(&conversation_schema, mutation)
                                .await?;
                        }
                        for job in admission
                            .index_jobs
                            .iter()
                            .chain(acl_index_jobs.iter())
                        {
                            uow.enqueue_index_job_in_schema(&conversation_schema, job.clone())
                                .await?;
                        }
                    }
                    let mut created_payload = json!({
                        "run_id": run_id,
                        "run_kind": "workflow_runtime",
                        "conversation_id": conversation_id,
                        "workflow_id": workflow_id,
                        "status": "queued",
                        "turn_node_id": turn_node_id,
                        "runtime_owner_binding": owner_binding,
                        "implementation_attestation": runtime_implementation_attestation(&runtime_kind),
                    });
                    if let Some(actor_user_id) = admission_actor_user_id.as_deref() {
                        created_payload["admission_actor"] = json!({
                            "user_id": actor_user_id,
                            "operation": "send_message",
                        });
                    }
                    let created = uow
                        .append_server_run_event(
                            &run_id,
                            "run.created",
                            serde_json::to_string(&created_payload)?,
                        )
                        .await?;
                    let mut payload = json!({
                        "contract_version": 1,
                        "kind": "workflow.run.execute",
                        "run_id": run_id,
                        "workflow_id": workflow_id,
                        "conversation_id": conversation_id,
                        "turn_node_id": turn_node_id,
                        "user_id": run_user_id,
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
                    if let Some(identity) = effective_start_worker_identity.as_ref() {
                        payload["required_worker_identity"] = identity.clone();
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
                            effective_start_worker_identity.as_ref(),
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
                    Ok(RuntimeAdmissionOutcome::Queued)
                })
            })
            .await;
        match outcome {
            Ok(RuntimeAdmissionOutcome::Queued) => json_effect(StatusCode::ACCEPTED, response),
            Ok(RuntimeAdmissionOutcome::QueueFull) => json_effect(
                StatusCode::TOO_MANY_REQUESTS,
                json!({
                    "admission": queue_full_admission,
                    "reason": "queue_full",
                    "max_queue": max_queue,
                }),
            ),
            Ok(RuntimeAdmissionOutcome::ConversationChanged) => effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_CONVERSATION_CHANGED",
                "conversation owner, ACL visibility, or tail changed while preparing the message",
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
        if runtime_claim_identity_registration_matches(&request.principal, &recipient_ids).is_err()
        {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_WORKER_IDENTITY_FORBIDDEN",
                "identity-specific work requires a matching host-registered worker identity",
            );
        }
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
        let supports_graph_mutation_batch = self
            .store
            .graph_authority_capability()
            .supports_atomic_index_jobs();
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
                        let mut work = runtime_claimed_work(
                            lane,
                            &claim_owner,
                            run_id,
                            payload,
                            expected_event_seq,
                        );
                        if supports_graph_mutation_batch {
                            work.worker_effect_capabilities
                                .push("graph_mutation_batch_v1".to_owned());
                            work.worker_effect_capabilities
                                .push("graph_mutation_batch_v2".to_owned());
                        }
                        values.push(work);
                    }
                    Ok(values)
                })
            })
            .await
            .map_err(|error| error.to_string())
    }

    async fn apply_runtime_result(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        let input = match runtime_worker_result_from_body(&request.body) {
            Ok(value) => value,
            Err(response) => return response,
        };
        self.apply_runtime_result_input(input, Some(request), None, "python-worker")
            .await
    }

    async fn apply_runtime_result_input(
        &self,
        input: RuntimeWorkerResultRequest,
        request: Option<&ApiEffectRequest>,
        inline_principal: Option<Value>,
        executor_target: &str,
    ) -> ApiEffectResponse {
        if input.transition.is_some() == input.effect.is_some() {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                "exactly one of transition or effect is required",
            );
        }
        if let Some(effect) = input.effect.as_ref()
            && let Err(message) = validate_python_worker_submission_trace_binding(
                &effect.trace_events,
                &input.handoff,
            )
        {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_WORKER_TRACE",
                message,
            );
        }
        if let Some(effect) = input.effect.as_ref()
            && !effect.provider_event_publications.is_empty()
            && (effect.graph_mutation_batch_v2.is_none()
                || effect
                    .graph_mutation_batch_v2
                    .as_ref()
                    .is_some_and(|batch| batch.default_acl.truth_graph != "conversation"))
        {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                "provider event publications require an ACL-aware conversation graph batch",
            );
        }
        let effect_batch: Option<Result<(RuntimeGraphMutationBatch<'_>, Option<&str>), String>> =
            input.effect.as_ref().and_then(|effect| {
                match (
                    effect.graph_mutation_batch.as_ref(),
                    effect.graph_mutation_batch_v2.as_ref(),
                ) {
                    (Some(_), Some(_)) => Some(Err(
                        "worker effect cannot carry both graph batch versions".to_owned(),
                    )),
                    (Some(v1), None) => Some(Ok((RuntimeGraphMutationBatch::V1(v1), None))),
                    (None, Some(v2)) => Some(Ok((
                        RuntimeGraphMutationBatch::V2(v2),
                        Some(v2.default_acl.truth_graph.as_str()),
                    ))),
                    (None, None) => None,
                }
            });
        let graph_effects = if let Some(batch_result) = effect_batch {
            let (batch, default_acl_truth_graph) = match batch_result {
                Ok(value) => value,
                Err(error) => {
                    return effect_error(
                        StatusCode::UNPROCESSABLE_ENTITY,
                        "KOGWISTAR_INVALID_REQUEST",
                        error,
                    );
                }
            };
            if !self
                .store
                .graph_authority_capability()
                .supports_atomic_index_jobs()
            {
                return effect_error(
                    StatusCode::NOT_IMPLEMENTED,
                    "KOGWISTAR_GRAPH_AUTHORITY_UNSUPPORTED",
                    "Worker graph effects require atomic graph authority with index jobs",
                );
            }
            let principal = request
                .map(|request| &request.principal)
                .or(inline_principal.as_ref());
            let Some(principal) = principal else {
                return effect_error(
                    StatusCode::FORBIDDEN,
                    "KOGWISTAR_CAPABILITY_FORBIDDEN",
                    "Worker graph effects require an authenticated write_graph principal",
                );
            };
            if !capability_allowed_for_principal(
                principal,
                &self.capabilities,
                "write_graph",
                &["write_graph"],
            ) {
                return effect_error(
                    StatusCode::FORBIDDEN,
                    "KOGWISTAR_CAPABILITY_FORBIDDEN",
                    "Worker graph effects require write_graph capability",
                );
            }
            if inline_principal.is_some()
                && !principal_strings(principal, "capabilities")
                    .iter()
                    .any(|capability| capability == "write_graph")
            {
                return effect_error(
                    StatusCode::FORBIDDEN,
                    "KOGWISTAR_CAPABILITY_FORBIDDEN",
                    "Inline graph effects exceed the run's admitted capability ceiling",
                );
            }
            let Some(run_owner_id) = authenticated_run_owner_id(principal) else {
                return effect_error(
                    StatusCode::FORBIDDEN,
                    "KOGWISTAR_RUN_OWNER_FORBIDDEN",
                    "Worker graph effects require an authenticated run owner",
                );
            };
            let (mutations, index_jobs) = match batch {
                RuntimeGraphMutationBatch::V1(batch) => {
                    if let Err(error) = batch.validate() {
                        return effect_error(
                            StatusCode::UNPROCESSABLE_ENTITY,
                            "KOGWISTAR_INVALID_REQUEST",
                            error.to_string(),
                        );
                    }
                    let input = batch.clone();
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
                    (mutations, index_jobs)
                }
                RuntimeGraphMutationBatch::V2(batch) => {
                    match graph_mutation_batch_v2_value((*batch).clone()) {
                        Ok(value) => {
                            if let Some(effect) = input.effect.as_ref()
                                && let Err(error) = validate_provider_event_publications(
                                    &effect.provider_event_publications,
                                    &value.0,
                                    &input.handoff.run_id,
                                )
                            {
                                return effect_error(
                                    StatusCode::UNPROCESSABLE_ENTITY,
                                    "KOGWISTAR_INVALID_REQUEST",
                                    error,
                                );
                            }
                            value
                        }
                        Err(error) => {
                            return effect_error(
                                StatusCode::UNPROCESSABLE_ENTITY,
                                "KOGWISTAR_INVALID_REQUEST",
                                error,
                            );
                        }
                    }
                }
            };
            if let Err(error) = validate_runtime_graph_mutations(&mutations) {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error,
                );
            }
            let security_scope = security_scope_value(principal);
            if mutations
                .iter()
                .any(|mutation| !graph_mutation_scope_allowed(&mutation.scope, &security_scope))
            {
                return effect_error(
                    StatusCode::FORBIDDEN,
                    "KOGWISTAR_SCOPE_FORBIDDEN",
                    "Worker graph effect scope is outside the principal storage scope",
                );
            }
            Some((
                mutations,
                index_jobs,
                default_acl_truth_graph.map(str::to_owned),
                run_owner_id.to_owned(),
                principal.clone(),
                inline_principal.is_some(),
                input
                    .effect
                    .as_ref()
                    .map(|effect| effect.effect_id.clone())
                    .unwrap_or_default(),
            ))
        } else {
            None
        };
        let provider_event_publications = input
            .effect
            .as_ref()
            .map(|effect| effect.provider_event_publications.clone())
            .unwrap_or_default();
        let provider_event_acknowledgements = input
            .effect
            .as_ref()
            .map(|effect| effect.provider_event_acknowledgements.clone())
            .unwrap_or_default();
        let runtime_run_id = input.handoff.run_id.clone();
        if validate_provider_event_acknowledgements(&provider_event_acknowledgements).is_err() {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                "provider event acknowledgements must be unique, complete, and limited to 100",
            );
        }
        if !provider_event_acknowledgements.is_empty()
            && input.effect.as_ref().is_none_or(|effect| {
                effect.status != kogwistar_runtime::RuntimeWorkerEffectStatus::Success
            })
        {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                "provider events can only be acknowledged with a successful durable handling step",
            );
        }
        if provider_event_acknowledgements.iter().any(|ack| {
            provider_event_publications
                .iter()
                .any(|publication| publication.event.event_id == ack.event_id)
        }) {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                "one worker effect cannot publish and acknowledge the same provider event",
            );
        }
        let provider_event_ack_scope = if provider_event_acknowledgements.is_empty() {
            None
        } else {
            let principal = match self
                .resolve_inline_native_owner_principal(&runtime_run_id, "write_graph")
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
            let owner_id = match authenticated_run_owner_id(&principal) {
                Some(owner_id) => owner_id.to_owned(),
                None => {
                    return effect_error(
                        StatusCode::FORBIDDEN,
                        "KOGWISTAR_RUN_OWNER_FORBIDDEN",
                        "provider event acknowledgement requires an authenticated run owner",
                    );
                }
            };
            let namespace = match security_scope_value(&principal)["storage_namespace"]
                .as_str()
                .map(str::to_owned)
            {
                Some(namespace) => namespace,
                None => {
                    return effect_error(
                        StatusCode::FORBIDDEN,
                        "KOGWISTAR_SCOPE_FORBIDDEN",
                        "provider event acknowledgement has no admitted storage namespace",
                    );
                }
            };
            let admission = match self
                .store
                .list_server_run_events(&runtime_run_id, 0, 1)
                .await
            {
                Ok(events) => events
                    .into_iter()
                    .find(|event| event.event_type == "run.created"),
                Err(_) => {
                    return effect_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "KOGWISTAR_RUN_OWNER_UNAVAILABLE",
                        "provider event run admission could not be revalidated",
                    );
                }
            };
            let Some(admission) = admission else {
                return effect_error(
                    StatusCode::FORBIDDEN,
                    "KOGWISTAR_RUN_OWNER_FORBIDDEN",
                    "provider event run admission binding is missing",
                );
            };
            let binding = serde_json::from_str::<Value>(&admission.payload_json)
                .ok()
                .and_then(|payload| payload.get("runtime_owner_binding").cloned())
                .and_then(|binding| serde_json::from_value::<RuntimeOwnerBindingV1>(binding).ok());
            let Some(binding) = binding.filter(|binding| binding.user_id == owner_id) else {
                return effect_error(
                    StatusCode::FORBIDDEN,
                    "KOGWISTAR_RUN_OWNER_FORBIDDEN",
                    "provider event run owner does not match its admission binding",
                );
            };
            let Some(auth_store) = self.owner_auth_store.clone() else {
                return effect_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "KOGWISTAR_RUN_OWNER_UNAVAILABLE",
                    "provider event run owner revalidation is not configured",
                );
            };
            Some((namespace, owner_id, binding, auth_store))
        };
        let commit_capabilities = self.capabilities.clone();
        let expected_executor_target = executor_target.to_owned();
        let conversation_schema = self.conversation_store.schema().to_owned();
        let workflow_schema = self.workflow_store.schema().to_owned();
        let wisdom_schema = self.wisdom_store.schema().to_owned();
        let knowledge_schema = self.store.schema().to_owned();
        let graph_role = self.graph_role.clone();
        let graph_schemas = [
            conversation_schema.as_str(),
            workflow_schema.as_str(),
            wisdom_schema.as_str(),
        ];
        let outcome = self
            .store
            .transaction_with_graph_schemas(&graph_schemas, |uow| {
                let provider_event_ack_scope = provider_event_ack_scope.clone();
                let commit_capabilities = commit_capabilities.clone();
                let conversation_schema = conversation_schema.clone();
                let workflow_schema = workflow_schema.clone();
                let wisdom_schema = wisdom_schema.clone();
                let knowledge_schema = knowledge_schema.clone();
                let graph_role = graph_role.clone();
                Box::pin(async move {
                    let lane = uow
                        .get_projected_lane_message(&input.handoff.message_id)
                        .await?
                        .ok_or_else(|| {
                            kogwistar_store_postgres::PostgresStoreError::RecordedRuntimeConflict(
                                "claimed runtime lane does not exist".to_owned(),
                            )
                        })?;
                    validate_runtime_lane_executor(&lane, &expected_executor_target).map_err(|message| {
                        kogwistar_store_postgres::PostgresStoreError::RecordedRuntimeConflict(
                            message.to_owned(),
                        )
                    })?;
                    if let Some((mutations, _, _, run_owner_id, principal, inline, _)) = graph_effects.as_ref() {
                        if !capability_allowed_for_principal(
                            principal,
                            &commit_capabilities,
                            "write_graph.commit",
                            &["write_graph"],
                        ) || (*inline
                            && !principal_strings(principal, "capabilities")
                                .iter()
                                .any(|capability| capability == "write_graph"))
                        {
                            return Ok(Err(RuntimeGraphCommitDenial::Capability));
                        }
                        let commit_scope = security_scope_value(principal);
                        if mutations.iter().any(|mutation| {
                            !graph_mutation_scope_allowed(&mutation.scope, &commit_scope)
                        }) {
                            return Ok(Err(RuntimeGraphCommitDenial::Scope));
                        }
                        let run = uow.get_server_run(&input.handoff.run_id).await?;
                        if !run.as_ref().is_some_and(|run| {
                            run.user_id.as_deref() == Some(run_owner_id.as_str())
                        }) {
                            return Ok(Err(RuntimeGraphCommitDenial::Owner));
                        }
                        if provider_event_publications.iter().any(|publication| {
                            run.as_ref().is_none_or(|run| {
                                run.conversation_id != publication.conversation_id
                            })
                        }) {
                            return Err(
                                kogwistar_store_postgres::PostgresStoreError::RecordedRuntimeConflict(
                                    "provider event conversation does not match its durable run".to_owned(),
                                ),
                            );
                        }
                    }
                    if let Some((mutations, mut index_jobs, default_acl_truth_graph, _, principal, _, effect_id)) = graph_effects {
                        let mut mutations = mutations;
                        let graph_type = default_acl_truth_graph
                            .as_deref()
                            .or_else(|| mutations.first().and_then(|mutation| mutation.scope.graph_space.as_deref()))
                            .unwrap_or(&graph_role)
                            .to_owned();
                        if mutations.iter().any(|mutation| {
                            mutation.scope.graph_space.as_deref().is_some_and(|scope_type| {
                                scope_type != graph_type
                            })
                        }) {
                            return Err(kogwistar_store_postgres::PostgresStoreError::RecordedRuntimeConflict(
                                "one worker graph batch cannot span graph types".to_owned(),
                            ));
                        }
                        let graph_schema = match graph_type.as_str() {
                            "conversation" => &conversation_schema,
                            "workflow" => &workflow_schema,
                            "wisdom" => &wisdom_schema,
                            _ => &knowledge_schema,
                        };
                        if let Some(truth_graph) = default_acl_truth_graph {
                            let scope = mutations.first().map(|mutation| mutation.scope.clone()).ok_or_else(|| {
                                kogwistar_store_postgres::PostgresStoreError::RecordedRuntimeConflict(
                                    "ACL-aware graph batch has no mutations".to_owned(),
                                )
                            })?;
                            let owner_id = authenticated_run_owner_id(&principal).ok_or_else(|| {
                                kogwistar_store_postgres::PostgresStoreError::RecordedRuntimeConflict(
                                    "ACL-aware graph batch has no authenticated run owner".to_owned(),
                                )
                            })?.to_ascii_lowercase();
                            let security_scope = principal.get("security_scope").and_then(Value::as_str).map(str::trim).filter(|value| !value.is_empty()).map(str::to_ascii_lowercase);
                            let (acl_mutations, acl_jobs) = resolve_default_acl_graph_effects_in_schema(
                                uow,
                                graph_schema,
                                &scope,
                                &mutations,
                                &truth_graph,
                                &owner_id,
                                security_scope.as_deref(),
                                &effect_id,
                            ).await?;
                            mutations.extend(acl_mutations);
                            index_jobs.extend(acl_jobs);
                        }
                        for mutation in mutations {
                            uow.apply_graph_mutation_in_schema(graph_schema, mutation).await?;
                        }
                        for index_job in index_jobs {
                            uow.enqueue_index_job_in_schema(graph_schema, index_job).await?;
                        }
                    }
                    for publication in &provider_event_publications {
                        let event = &publication.event;
                        let message_id = format!("msg:{}", event.event_id);
                        let payload_json = serde_json::to_string(event)?;
                        let now = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map_err(|error| {
                                kogwistar_store_postgres::PostgresStoreError::Backend(
                                    format!("system clock is before Unix epoch: {error}"),
                                )
                            })?
                            .as_secs() as i64;
                        if let Some(existing) = uow.get_projected_lane_message(&message_id).await? {
                            let matches = existing.namespace == publication.namespace
                                && existing.purpose == "agent_event"
                                && existing.inbox_id == "agent-provider-events"
                                && existing.conversation_id == publication.conversation_id
                                && existing.recipient_id == "agent-event-consumer"
                                && existing.sender_id == "agent-runtime"
                                && existing.msg_type == "agent.provider_event"
                                && existing.run_id.as_deref() == Some(input.handoff.run_id.as_str())
                                && existing.step_id.as_deref() == Some(event.trace.step_id.as_str())
                                && existing.correlation_id.as_deref() == Some(event.event_id.as_str())
                                && existing.payload_json.as_deref() == Some(payload_json.as_str());
                            if !matches {
                                return Err(
                                    kogwistar_store_postgres::PostgresStoreError::RecordedRuntimeConflict(
                                        "provider event lane projection conflicts with the existing event".to_owned(),
                                    ),
                                );
                            }
                        } else {
                            uow.project_lane_message(NewProjectedLaneMessage {
                                message_id,
                                namespace: publication.namespace.clone(),
                                purpose: "agent_event".to_owned(),
                                inbox_id: "agent-provider-events".to_owned(),
                                conversation_id: publication.conversation_id.clone(),
                                recipient_id: "agent-event-consumer".to_owned(),
                                sender_id: "agent-runtime".to_owned(),
                                msg_type: "agent.provider_event".to_owned(),
                                status: "pending".to_owned(),
                                created_at: now,
                                available_at: now,
                                run_id: Some(input.handoff.run_id.clone()),
                                step_id: Some(event.trace.step_id.clone()),
                                correlation_id: Some(event.event_id.clone()),
                                payload_json: Some(payload_json),
                                error_json: None,
                            })
                            .await?;
                        }
                    }
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
                    if !applied.idempotent {
                        let mut linked_step_nodes = BTreeSet::new();
                        for acknowledgement in &provider_event_acknowledgements {
                            if acknowledgement.message_id
                                != format!("msg:{}", acknowledgement.event_id)
                            {
                                return Err(
                                    kogwistar_store_postgres::PostgresStoreError::RecordedRuntimeConflict(
                                        "provider event acknowledgement message ID does not match its event".to_owned(),
                                    ),
                                );
                            }
                            let lane = uow
                                .get_projected_lane_message(&acknowledgement.message_id)
                                .await?
                                .ok_or_else(|| {
                                    kogwistar_store_postgres::PostgresStoreError::RecordedRuntimeConflict(
                                        "provider event acknowledgement lane does not exist".to_owned(),
                                    )
                                })?;
                            let Some((ack_namespace, ack_owner_id, binding, auth_store)) =
                                provider_event_ack_scope.as_ref()
                            else {
                                return Err(
                                    kogwistar_store_postgres::PostgresStoreError::RecordedRuntimeConflict(
                                        "provider event acknowledgement lacks a validated run-owner scope".to_owned(),
                                    ),
                                );
                            };
                            let event = lane
                                .payload_json
                                .as_deref()
                                .and_then(|payload| {
                                    serde_json::from_str::<kogwistar_contracts::DurableProviderEventV1>(payload).ok()
                                });
                            let run = uow.get_server_run(&runtime_run_id).await?;
                            if !run.as_ref().is_some_and(|run| {
                                run.user_id.as_deref() == Some(ack_owner_id.as_str())
                            }) {
                                return Err(
                                    kogwistar_store_postgres::PostgresStoreError::RecordedRuntimeConflict(
                                        "provider event run owner changed before acknowledgement".to_owned(),
                                    ),
                                );
                            }
                            let current_user = auth_store
                                .auth_user(ack_owner_id)
                                .await
                                .map_err(|_| kogwistar_store_postgres::PostgresStoreError::RecordedRuntimeConflict(
                                    "provider event run owner could not be revalidated".to_owned(),
                                ))?
                                .filter(|user| user.is_active)
                                .ok_or_else(|| kogwistar_store_postgres::PostgresStoreError::RecordedRuntimeConflict(
                                    "provider event run owner is missing or inactive".to_owned(),
                                ))?;
                            let current_principal = revalidated_runtime_owner_principal(
                                binding,
                                &current_user,
                                &commit_capabilities,
                                "write_graph",
                            )
                            .map_err(|_| kogwistar_store_postgres::PostgresStoreError::RecordedRuntimeConflict(
                                "provider event run owner no longer has write_graph authority".to_owned(),
                            ))?;
                            let current_scope = security_scope_value(&current_principal);
                            if current_scope["storage_namespace"].as_str()
                                != Some(ack_namespace.as_str())
                            {
                                return Err(
                                    kogwistar_store_postgres::PostgresStoreError::RecordedRuntimeConflict(
                                        "provider event acknowledgement namespace is outside the current run-owner scope".to_owned(),
                                    ),
                                );
                            }
                            if lane.namespace.trim().is_empty()
                                || lane.namespace != *ack_namespace
                                || lane.purpose != "agent_event"
                                || lane.inbox_id != "agent-provider-events"
                                || lane.msg_type != "agent.provider_event"
                                || lane.recipient_id != "agent-event-consumer"
                                || lane.status != "claimed"
                                || lane.claimed_by.as_deref()
                                    != Some(acknowledgement.claimed_by.as_str())
                                || lane.run_id.as_deref() != Some(runtime_run_id.as_str())
                                || lane.conversation_id
                                    != run.as_ref().map(|run| run.conversation_id.as_str()).unwrap_or_default()
                                || lane.correlation_id.as_deref()
                                    != Some(acknowledgement.event_id.as_str())
                                || event.as_ref().is_none_or(|event| {
                                    event.event_id != acknowledgement.event_id
                                        || event.trace.run_id != runtime_run_id
                                })
                            {
                                return Err(
                                    kogwistar_store_postgres::PostgresStoreError::RecordedRuntimeConflict(
                                        "provider event acknowledgement is not bound to the active run claim".to_owned(),
                                    ),
                                );
                            }
                            let event = event.ok_or_else(|| {
                                kogwistar_store_postgres::PostgresStoreError::RecordedRuntimeConflict(
                                    "provider event payload is unavailable for causal linkage".to_owned(),
                                )
                            })?;
                            let handler_step_seq = applied
                                .reduced
                                .state
                                .state
                                .get("agent_event_handling_step_seq")
                                .and_then(Value::as_i64)
                                .filter(|step_seq| *step_seq >= 0)
                                .ok_or_else(|| {
                                    kogwistar_store_postgres::PostgresStoreError::RecordedRuntimeConflict(
                                        "provider event has no valid durable handler step sequence".to_owned(),
                                    )
                                })?;
                            let mut referenced_step_seqs = vec![handler_step_seq];
                            if let Some(producer_step_seq) = event.trace.producer_step_seq {
                                if producer_step_seq < 0 {
                                    return Err(
                                        kogwistar_store_postgres::PostgresStoreError::RecordedRuntimeConflict(
                                            "provider event has an invalid producer step sequence".to_owned(),
                                        ),
                                    );
                                }
                                referenced_step_seqs.push(producer_step_seq);
                            }
                            referenced_step_seqs.sort_unstable();
                            referenced_step_seqs.dedup();
                            let recorded_steps = uow
                                .recorded_runtime_steps(&runtime_run_id, &referenced_step_seqs)
                                .await?;
                            let step_snapshots = provider_event_step_snapshots(
                                recorded_steps,
                                &runtime_run_id,
                                &lane.conversation_id,
                                &applied.reduced.state.workflow_id,
                                &referenced_step_seqs,
                            )
                            .map_err(
                                kogwistar_store_postgres::PostgresStoreError::RecordedRuntimeConflict,
                            )?;
                            let workflow_node_id = applied
                                .reduced
                                .state
                                .last_node_id
                                .as_deref()
                                .filter(|value| !value.trim().is_empty())
                                .ok_or_else(|| {
                                    kogwistar_store_postgres::PostgresStoreError::RecordedRuntimeConflict(
                                        "provider event handling step has no workflow node identity".to_owned(),
                                    )
                                })?;
                            let step_seq = applied.reduced.state.last_step_seq;
                            if step_seq < 0 {
                                return Err(
                                    kogwistar_store_postgres::PostgresStoreError::RecordedRuntimeConflict(
                                        "provider event handling step has an invalid sequence".to_owned(),
                                    ),
                                );
                            }
                            if handler_step_seq >= step_seq
                                || event
                                    .trace
                                    .producer_step_seq
                                    .is_some_and(|producer_step_seq| producer_step_seq >= step_seq)
                            {
                                return Err(
                                    kogwistar_store_postgres::PostgresStoreError::RecordedRuntimeConflict(
                                        "provider event causal steps must precede the acknowledgement step".to_owned(),
                                    ),
                                );
                            }
                            let graph_scope = GraphScope {
                                namespace: ack_namespace.clone(),
                                workspace_id: current_scope
                                    .get("workspace")
                                    .and_then(Value::as_str)
                                    .filter(|value| !value.is_empty())
                                    .map(str::to_owned),
                                graph_space: Some("conversation".to_owned()),
                            };
                            if !graph_mutation_scope_allowed(&graph_scope, &current_scope) {
                                return Err(
                                    kogwistar_store_postgres::PostgresStoreError::RecordedRuntimeConflict(
                                        "provider event handling graph scope changed before commit".to_owned(),
                                    ),
                                );
                            }
                            let mut causal_mutations = provider_event_ack_graph_mutations(
                                &acknowledgement.event_id,
                                &runtime_run_id,
                                &lane.conversation_id,
                                &applied.reduced.state.workflow_id,
                                workflow_node_id,
                                step_seq,
                                &effect_id,
                                &graph_scope,
                            );
                            causal_mutations.extend(provider_event_step_link_mutations(
                                &acknowledgement.event_id,
                                &runtime_run_id,
                                &lane.conversation_id,
                                event.trace.producer_step_seq,
                                handler_step_seq,
                                &step_snapshots,
                                &graph_scope,
                            ));
                            causal_mutations.retain(|mutation| {
                                mutation.entity_kind != "node"
                                    || linked_step_nodes.insert(mutation.record.id.clone())
                            });
                            let (acl_mutations, acl_jobs) = resolve_default_acl_graph_effects_in_schema(
                                uow,
                                &conversation_schema,
                                &graph_scope,
                                &causal_mutations,
                                "conversation",
                                &graph_acl_principal_id(&current_principal),
                                current_scope["security_scope"].as_str(),
                                &acknowledgement.event_id,
                            )
                            .await?;
                            for mutation in causal_mutations {
                                uow.apply_graph_mutation_in_schema(&conversation_schema, mutation)
                                    .await?;
                            }
                            for mutation in acl_mutations {
                                uow.apply_graph_mutation_in_schema(&conversation_schema, mutation)
                                    .await?;
                            }
                            for job in acl_jobs {
                                uow.enqueue_index_job_in_schema(&conversation_schema, job)
                                    .await?;
                            }
                            uow.ack_claimed_projected_lane_message(
                                &acknowledgement.message_id,
                                &acknowledgement.claimed_by,
                            )
                            .await?;
                            uow.append_server_run_event(
                                &runtime_run_id,
                                "workflow.provider_event.handled.v1",
                                serde_json::to_string(&json!({
                                    "event_id": acknowledgement.event_id,
                                    "message_id": acknowledgement.message_id,
                                    "workflow_node_id": applied.reduced.state.last_node_id,
                                    "step_seq": applied.reduced.state.last_step_seq,
                                    "effect_id": effect_id,
                                }))?,
                            )
                            .await?;
                        }
                    }
                    if applied.idempotent {
                        return Ok(Ok(applied));
                    }
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
                    // The response identifies this worker result, not a later
                    // completion event appended as a consequence of it.
                    let worker_result = applied.clone();
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
                                    expected_event_seq: uow
                                        .latest_server_run_event_seq(&state.run_id)
                                        .await?,
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
                                expected_event_seq: uow
                                    .latest_server_run_event_seq(&state.run_id)
                                    .await?,
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
                    Ok(Ok(worker_result))
                })
            })
            .await;
        match outcome {
            Ok(Err(RuntimeGraphCommitDenial::Owner)) => effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_RUN_OWNER_FORBIDDEN",
                "Authenticated principal does not own the durable run",
            ),
            Ok(Err(RuntimeGraphCommitDenial::Capability)) => effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "write_graph capability was revoked before graph commit",
            ),
            Ok(Err(RuntimeGraphCommitDenial::Scope)) => effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_SCOPE_FORBIDDEN",
                "Worker graph effect scope is outside the principal storage scope at commit",
            ),
            Ok(Ok(result)) => json_effect(
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

    async fn resume_runtime_run_and_wait(
        &self,
        run_id: &str,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
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
        let mut handoff = request.clone();
        handoff.path_and_query = format!("/api/workflow/runs/{run_id}/resume");
        let admitted = self.resume_runtime_run(run_id, &handoff).await;
        if !(200..300).contains(&admitted.status) {
            return admitted;
        }
        loop {
            let run = match self.store.get_server_run(run_id).await {
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
                let state = match self
                    .store
                    .read_recorded_runtime_state(run_id, &input.workflow_id, &input.conversation_id)
                    .await
                {
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
}
