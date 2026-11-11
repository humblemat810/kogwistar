// Durable run admission, queue lifecycle, claims, and worker-result handling.
// Python parity: kogwistar/server/runtime_api.py and kogwistar/runtime/runtime.py.
impl SqliteRunApplicationService {
    fn get_run(&self, run_id: &str) -> Result<Option<ServerRun>, String> {
        self.store
            .get_server_run(run_id)
            .map_err(|error| error.to_string())
    }

    fn submit_run(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        self.submit_run_with_conversation_admission(request, None)
    }

    fn submit_run_with_conversation_admission(
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
        let workflow_id = input.workflow_id.trim();
        let conversation_id = input.conversation_id.trim();
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
        let run_user_id = authenticated_owner
            .map(str::to_owned)
            .or_else(|| input.user_id.clone());
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
            default_runtime_priority_class()
        } else {
            input.priority_class.clone()
        };
        let runtime_kind = if input.runtime_kind.is_empty() {
            default_runtime_kind()
        } else {
            input.runtime_kind.clone()
        };
        let worker_protocol = worker_protocol_for_runtime_kind(&runtime_kind);
        let mut event_payload = json!({
            "run_id": run_id,
            "run_kind": "workflow_runtime",
            "conversation_id": conversation_id,
            "workflow_id": workflow_id,
            "status": "queued",
            "turn_node_id": turn_node_id,
            "implementation_attestation": runtime_implementation_attestation(&runtime_kind),
        });
        if let Some(binding) = owner_binding {
            event_payload["runtime_owner_binding"] = json!(binding);
        }
        let graph_projection = self
            .store
            .get_named_projection("workflow_design", workflow_id)
            .ok()
            .flatten();
        let graph_snapshot = self
            .store
            .get_workflow_design_snapshot(workflow_id, i64::MAX, 1)
            .ok()
            .flatten();
        let graph_plan = exact_runtime_graph_plan(graph_projection, graph_snapshot.clone());
        if graph_plan.is_none() && runtime_snapshot_declares_worker_identity(graph_snapshot.as_ref()) {
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
        if is_answer_workflow_id(workflow_id)
            && (self.chroma_graph.is_none()
                || !answer_workflow_admission_ready(
                    workflow_id,
                    graph_plan.as_ref(),
                    &self.native_operations,
                ))
        {
            return effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_ANSWER_WORKFLOW_NOT_READY",
                "SQLite answer workflows require configured graph authority and a complete plan with compatible executors",
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
        // Even a bare transport-level submission must hand the Python worker a
        // frozen operation.  Later continuation lanes already use this same
        // no-op fallback; leaving the admission lane as JSON null made the
        // worker correctly reject an otherwise valid durable run.
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
        let mut initial_state = input.initial_state.clone();
        // Runtime scope and owner are server-derived authority, never caller state.
        // Keep SQLite admission aligned with PostgreSQL before scheduling any step.
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
        if let Some(token_budget) = input.token_budget {
            initial_state.insert("token_budget".to_owned(), json!(token_budget));
        }
        if let Some(time_budget_ms) = input.time_budget_ms {
            initial_state.insert("time_budget_ms".to_owned(), json!(time_budget_ms));
        }
        let worker_payload = json!({
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
        let mut worker_payload = worker_payload;
        if effective_start_op_version != "v1" {
            worker_payload["op_version"] = json!(effective_start_op_version);
        }
        if let Some(environment_id) = effective_start_environment_id.as_ref() {
            worker_payload["op_environment_id"] = json!(environment_id);
        }
        if let Some(identity) = effective_start_worker_identity.as_ref() {
            worker_payload["required_worker_identity"] = identity.clone();
        }
        add_executor_target(
            &mut worker_payload,
            &effective_start_op,
            effective_start_executor_target.as_deref(),
        );
        if let Ok(Some(existing)) = self.store.get_server_run(&run_id) {
            let lanes = self
                .store
                .list_projected_lane_messages(LaneMessageFilter {
                    namespace: Some("workflow".to_owned()),
                    inbox_id: Some("workflow-runtime".to_owned()),
                    correlation_id: Some(run_id.clone()),
                    limit: 10,
                    ..LaneMessageFilter::default()
                })
                .unwrap_or_default();
            return match runtime_submit_retry_response(&existing, &lanes, &worker_payload) {
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
        let admission_actor_user_id = conversation_admission
            .as_ref()
            .map(|admission| admission.actor_user_id.clone());
        let outcome = self.store.immediate_transaction(|uow| {
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
                return Ok(RuntimeAdmissionOutcome::QueueFull);
            }
            if let Some(admission) = conversation_admission.as_ref() {
                let current_events = uow.latest_entity_events_by_metadata(
                    &admission.scope.namespace,
                    "node",
                    "conversation_id",
                    &admission.conversation_id,
                )?;
                let current_nodes = current_events
                    .iter()
                    .filter(|event| !event.op.eq_ignore_ascii_case("DELETE"))
                    .map(conversation_record_from_raw_event)
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .filter(|record| {
                        !matches!(
                            conversation_record_string(record, "lifecycle_status").as_deref(),
                            Some("tombstoned" | "deleted")
                        )
                    })
                    .collect::<Vec<_>>();
                let current_ids = current_nodes
                    .iter()
                    .map(|record| record.id.clone())
                    .collect::<Vec<_>>();
                let current_acl = uow.graph_acl_records_for_entities(
                    &admission.scope,
                    "conversation",
                    &current_ids,
                )?;
                let mut acl_inputs = current_nodes;
                acl_inputs.extend(current_acl);
                let (visible_nodes, _) = graph_records_visible_by_acl(
                    &acl_inputs,
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
                    .collect::<Vec<_>>();
                current_tail.sort_by_key(|record| {
                    (
                        conversation_record_i64(record, "turn_index").unwrap_or(-1),
                        record.id.clone(),
                    )
                });
                let actual_tail_id = current_tail.last().map(|record| record.id.as_str());
                if !owner_matches || actual_tail_id != admission.expected_tail_id.as_deref() {
                    return Ok(RuntimeAdmissionOutcome::ConversationChanged);
                }
            }
            uow.create_server_run(ServerRunCreate {
                run_id: run_id.clone(),
                conversation_id: conversation_id.to_owned(),
                workflow_id: workflow_id.to_owned(),
                user_id: run_user_id.clone(),
                user_turn_node_id: turn_node_id.clone(),
                status: "queued".to_owned(),
            })?;
            if let Some(admission) = conversation_admission.as_ref() {
                let graph_mutations = admission
                    .mutations
                    .iter()
                    .map(|(mutation, _)| mutation.clone())
                    .collect::<Vec<_>>();
                let acl_targets = acl_targets_for_graph_mutations(&graph_mutations).0;
                let target_ids = acl_target_entity_ids(&acl_targets);
                let existing_acl = uow.graph_acl_records_for_entities(
                    &admission.scope,
                    "conversation",
                    &target_ids,
                )?;
                let (acl_mutations, acl_jobs) = derive_default_acl_graph_effects_from_records(
                    &admission.scope,
                    &graph_mutations,
                    acl_targets,
                    existing_acl,
                    "conversation",
                    &admission.owner_id.to_ascii_lowercase(),
                    admission.acl_principal.security_scope.as_deref(),
                    &run_id,
                )
                .map_err(SqliteStoreError::TransactionAborted)?;
                for (mutation, envelope) in &admission.mutations {
                    validate_typed_graph_mutation(mutation, envelope)?;
                }
                let mut mutations = graph_mutations;
                mutations.extend(acl_mutations);
                let mut index_jobs = admission.index_jobs.clone();
                index_jobs.extend(acl_jobs);
                uow.apply_canonical_graph_mutation_batch(GraphMutationBatch {
                    mutations,
                    index_jobs,
                })?;
            }
            let created_event = uow.append_server_run_event(&run_id, "run.created", {
                let mut payload = event_payload.clone();
                if let Some(actor_user_id) = admission_actor_user_id.as_deref() {
                    payload["admission_actor"] = json!({
                        "user_id": actor_user_id,
                        "operation": "send_message",
                    });
                }
                serde_json::to_string(&payload)?
            })?;
            let mut persisted_worker_payload = worker_payload.clone();
            persisted_worker_payload["expected_event_seq"] = json!(created_event.seq);
            uow.project_lane_message(NewProjectedLaneMessage {
                message_id: message_id.clone(),
                namespace: "workflow".to_owned(),
                purpose: "system".to_owned(),
                inbox_id: "workflow-runtime".to_owned(),
                conversation_id: conversation_id.to_owned(),
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
                correlation_id: Some(run_id.clone()),
                payload_json: Some(serde_json::to_string(&persisted_worker_payload)?),
                error_json: None,
            })?;
            Ok(RuntimeAdmissionOutcome::Queued)
        });
        match outcome {
            Ok(RuntimeAdmissionOutcome::QueueFull) => {
                return json_effect(
                    StatusCode::TOO_MANY_REQUESTS,
                    json!({
                        "admission": if matches!(priority_class.as_str(), "background" | "batch") { "rejected" } else { "deferred" },
                        "reason": "queue_full",
                        "max_queue": max_queue,
                    }),
                );
            }
            Ok(RuntimeAdmissionOutcome::ConversationChanged) => {
                return effect_error(
                    StatusCode::CONFLICT,
                    "KOGWISTAR_CONVERSATION_CHANGED",
                    "conversation owner or tail changed while the turn was being admitted",
                );
            }
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error.to_string(),
                );
            }
            Ok(RuntimeAdmissionOutcome::Queued) => {}
        }
        json_effect(
            StatusCode::ACCEPTED,
            json!({
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
            }),
        )
    }

    fn claim_runtime_work(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
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
        if runtime_claim_identity_registration_matches(&request.principal, &recipient_ids).is_err() {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_WORKER_IDENTITY_FORBIDDEN",
                "identity-specific work requires a matching host-registered worker identity",
            );
        }
        match self.claim_runtime_work_items(input, recipient_ids) {
            Ok(work) => json_effect(StatusCode::OK, json!({"work": work})),
            Err(error) => effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_RUNTIME_CLAIM_CONFLICT",
                error,
            ),
        }
    }

    fn claim_runtime_work_items(
        &self,
        input: RuntimeClaimRequest,
        recipient_ids: Vec<String>,
    ) -> Result<Vec<RuntimeClaimedWork>, String> {
        let claim_owner = format!("{}#{}", input.claimed_by, uuid::Uuid::new_v4());
        let supports_graph_mutation_batch = self.chroma_graph.is_some();
        self.store
            .immediate_transaction(|uow| {
                let mut claimed = Vec::new();
                for recipient_id in &recipient_ids {
                    let remaining = input.limit.saturating_sub(claimed.len());
                    if remaining == 0 {
                        break;
                    }
                    claimed.extend(uow.claim_projected_lane_messages_filtered(
                        LaneMessageClaimFilter {
                            namespace: "workflow".to_owned(),
                            inbox_id: "workflow-runtime".to_owned(),
                            claimed_by: claim_owner.clone(),
                            message_ids: None,
                            run_id: input.run_id.clone(),
                            msg_type: None,
                            recipient_id: Some(recipient_id.clone()),
                            limit: remaining,
                            lease_seconds: input.lease_seconds,
                        },
                    )?);
                }
                let mut values = Vec::new();
                for lane in claimed {
                    let payload: Value =
                        serde_json::from_str(lane.payload_json.as_deref().unwrap_or("{}"))?;
                    let run_id = lane.run_id.clone().unwrap_or_default();
                    if uow
                        .get_server_run(&run_id)?
                        .is_some_and(|run| run.cancel_requested)
                    {
                        uow.update_projected_lane_message_status(
                            &lane.message_id,
                            "cancelled",
                            None,
                        )?;
                        continue;
                    }
                    let starts_runtime = lane.msg_type == "workflow.run.execute"
                        && payload["runtime_started"].as_bool() != Some(true);
                    let expected_event_seq = if starts_runtime {
                        uow.apply_recorded_runtime_transition(
                            runtime_start_transition(&payload, &run_id),
                            false,
                        )?
                        .event_seq
                    } else {
                        payload["expected_event_seq"].as_i64().ok_or_else(|| {
                            kogwistar_store_sqlite::SqliteStoreError::RecordedRuntimeConflict(
                                format!(
                                    "continuation lane message {:?} lacks expected_event_seq",
                                    lane.message_id
                                ),
                            )
                        })?
                    };
                    let mut claimed_payload = payload.clone();
                    claimed_payload["expected_event_seq"] = json!(expected_event_seq);
                    if lane.msg_type == "workflow.run.execute" {
                        claimed_payload["runtime_started"] = json!(true);
                    }
                    uow.update_projected_lane_message_payload(
                        &lane.message_id,
                        serde_json::to_string(&claimed_payload)?,
                    )?;
                    let mut work = runtime_claimed_work(
                        lane,
                        &claim_owner,
                        run_id,
                        claimed_payload,
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
            .map_err(|error| error.to_string())
    }

    fn apply_runtime_result(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        self.apply_runtime_result_with_principal(request, &request.principal)
    }

    fn apply_runtime_result_with_principal(
        &self,
        request: &ApiEffectRequest,
        principal: &Value,
    ) -> ApiEffectResponse {
        let input = match runtime_worker_result_from_body(&request.body) {
            Ok(value) => value,
            Err(response) => return response,
        };
        self.apply_runtime_result_input(input, "python-worker", Some(principal))
    }

    fn apply_runtime_result_input(
        &self,
        input: RuntimeWorkerResultRequest,
        executor_target: &str,
        principal: Option<&Value>,
    ) -> ApiEffectResponse {
        if input.transition.is_some() == input.effect.is_some() {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                "exactly one of transition or effect is required",
            );
        }
        if input.effect.as_ref().is_some_and(|effect| {
            effect.graph_mutation_batch.is_some() && effect.graph_mutation_batch_v2.is_some()
        }) {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                "worker effect cannot carry both graph batch versions",
            );
        }
        let provider_event_acknowledgements = input
            .effect
            .as_ref()
            .map(|effect| effect.provider_event_acknowledgements.clone())
            .unwrap_or_default();
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
        if !provider_event_acknowledgements.is_empty() && self.chroma_graph.is_none() {
            return effect_error(
                StatusCode::NOT_IMPLEMENTED,
                "KOGWISTAR_GRAPH_AUTHORITY_UNSUPPORTED",
                "SQLite provider-event acknowledgement requires the configured Chroma graph projection",
            );
        }
        let provider_event_publications = input
            .effect
            .as_ref()
            .map(|effect| effect.provider_event_publications.clone())
            .unwrap_or_default();
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
            let Some(principal) = principal else {
                return effect_error(
                    StatusCode::FORBIDDEN,
                    "KOGWISTAR_CAPABILITY_FORBIDDEN",
                    "provider event acknowledgement requires the run owner's current write_graph authority",
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
                    "provider event acknowledgement requires the run owner's current write_graph authority",
                );
            }
            let Some(owner_id) = authenticated_run_owner_id(principal).map(str::to_owned) else {
                return effect_error(
                    StatusCode::FORBIDDEN,
                    "KOGWISTAR_RUN_OWNER_FORBIDDEN",
                    "provider event acknowledgement requires an authenticated run owner",
                );
            };
            let security_scope = security_scope_value(principal);
            let Some(namespace) = security_scope["storage_namespace"]
                .as_str()
                .filter(|value| !value.trim().is_empty())
                .map(str::to_owned)
            else {
                return effect_error(
                    StatusCode::FORBIDDEN,
                    "KOGWISTAR_SCOPE_FORBIDDEN",
                    "provider event acknowledgement has no admitted storage namespace",
                );
            };
            let scope = GraphScope {
                namespace,
                workspace_id: security_scope
                    .get("workspace")
                    .and_then(Value::as_str)
                    .filter(|value| !value.is_empty())
                    .map(str::to_owned),
                graph_space: Some("conversation".to_owned()),
            };
            if !graph_mutation_scope_allowed(&scope, &security_scope) {
                return effect_error(
                    StatusCode::FORBIDDEN,
                    "KOGWISTAR_SCOPE_FORBIDDEN",
                    "provider event acknowledgement scope is outside the run owner's admitted scope",
                );
            }
            Some((scope, owner_id, principal.clone()))
        };
        let graph_batch = input.effect.as_ref().and_then(|effect| {
            match (
                effect.graph_mutation_batch.as_ref(),
                effect.graph_mutation_batch_v2.as_ref(),
            ) {
                (Some(_), Some(_)) => Some(Err(
                    "worker effect cannot carry both graph batch versions".to_owned(),
                )),
                (Some(v1), None) => Some((|| {
                    if !provider_event_publications.is_empty() {
                        return Err(
                            "provider event publications require an ACL-aware graph batch V2"
                                .to_owned(),
                        );
                    }
                    let mutations = graph_mutation_batch_value(v1.clone())?;
                    validate_graph_mutation_envelopes(v1, &mutations)?;
                    validate_runtime_graph_mutations(&mutations)?;
                    let index_jobs = graph_index_jobs_value(v1, &mutations)?;
                    Ok((
                        GraphMutationBatch {
                            mutations,
                            index_jobs,
                        },
                        None,
                    ))
                })()),
                (None, Some(v2)) => Some((|| {
                    let (mutations, index_jobs) = graph_mutation_batch_v2_value(v2.clone())?;
                    validate_runtime_graph_mutations(&mutations)?;
                    if !provider_event_publications.is_empty() {
                        if v2.default_acl.truth_graph != "conversation" {
                            return Err(
                                "provider event publications require conversation ACL context"
                                    .to_owned(),
                            );
                        }
                        validate_provider_event_publications(
                            &provider_event_publications,
                            &mutations,
                            &input.handoff.run_id,
                        )?;
                    }
                    Ok((
                        GraphMutationBatch {
                            mutations,
                            index_jobs,
                        },
                        Some(v2.default_acl.truth_graph.clone()),
                    ))
                })()),
                (None, None) => None,
            }
        });
        let graph_batch = match graph_batch {
            Some(Ok((batch, default_acl_truth_graph))) => {
                if self.chroma_graph.is_none() {
                    return effect_error(
                        StatusCode::NOT_IMPLEMENTED,
                        "KOGWISTAR_GRAPH_AUTHORITY_UNSUPPORTED",
                        "runtime graph effects require SQLite canonical events with configured Chroma projection",
                    );
                }
                let Some(principal) = principal else {
                    return effect_error(
                        StatusCode::FORBIDDEN,
                        "KOGWISTAR_CAPABILITY_FORBIDDEN",
                        "runtime graph effects require an authenticated write_graph principal",
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
                        "runtime graph effects require write_graph capability",
                    );
                }
                let Some(owner_id) = authenticated_run_owner_id(principal).map(str::to_owned)
                else {
                    return effect_error(
                        StatusCode::FORBIDDEN,
                        "KOGWISTAR_RUN_OWNER_FORBIDDEN",
                        "runtime graph effects require an authenticated run owner",
                    );
                };
                let scope = security_scope_value(principal);
                if batch
                    .mutations
                    .iter()
                    .any(|mutation| !graph_mutation_scope_allowed(&mutation.scope, &scope))
                {
                    return effect_error(
                        StatusCode::FORBIDDEN,
                        "KOGWISTAR_SCOPE_FORBIDDEN",
                        "runtime graph effect scope is outside the authenticated owner scope",
                    );
                }
                let effect_id = input
                    .effect
                    .as_ref()
                    .map(|effect| effect.effect_id.clone())
                    .unwrap_or_default();
                Some((
                    batch,
                    owner_id,
                    principal.clone(),
                    default_acl_truth_graph,
                    effect_id,
                    provider_event_publications.clone(),
                ))
            }
            Some(Err(error)) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error,
                );
            }
            None => None,
        };
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
        let runtime_run_id = input.handoff.run_id.clone();
        let outcome = self.store.immediate_transaction(|uow| {
            if let Some((batch, owner_id, principal, default_acl_truth_graph, effect_id, publications)) = graph_batch.as_ref() {
                if !capability_allowed_for_principal(
                    principal,
                    &self.capabilities,
                    "write_graph.commit",
                    &["write_graph"],
                ) {
                    return Err(kogwistar_store_sqlite::SqliteStoreError::RecordedRuntimeConflict(
                        "runtime graph owner no longer has write_graph capability".to_owned(),
                    ));
                }
                let run = uow.get_server_run(&input.handoff.run_id)?;
                if !run
                    .as_ref()
                    .is_some_and(|run| run.user_id.as_deref() == Some(owner_id.as_str()))
                {
                    return Err(kogwistar_store_sqlite::SqliteStoreError::RecordedRuntimeConflict(
                        "runtime graph effect owner does not match the durable run owner".to_owned(),
                    ));
                }
                if publications.iter().any(|publication| {
                    run.as_ref().is_none_or(|run| {
                        run.conversation_id != publication.conversation_id
                    })
                }) {
                    return Err(kogwistar_store_sqlite::SqliteStoreError::RecordedRuntimeConflict(
                        "provider event conversation does not match its durable run".to_owned(),
                    ));
                }
                let scope = security_scope_value(principal);
                if batch
                    .mutations
                    .iter()
                    .any(|mutation| !graph_mutation_scope_allowed(&mutation.scope, &scope))
                {
                    return Err(kogwistar_store_sqlite::SqliteStoreError::RecordedRuntimeConflict(
                        "runtime graph effect scope is outside the current owner scope".to_owned(),
                    ));
                }
                let mut batch = batch.clone();
                if let Some(truth_graph) = default_acl_truth_graph {
                    let scope = batch.mutations.first().map(|mutation| mutation.scope.clone()).ok_or_else(|| {
                        kogwistar_store_sqlite::SqliteStoreError::TransactionAborted(
                            "ACL-aware graph batch must not be empty".to_owned(),
                        )
                    })?;
                    let targets = acl_targets_for_graph_mutations(&batch.mutations).0;
                    let entity_ids = acl_target_entity_ids(&targets);
                    let existing = uow.graph_acl_records_for_entities(&scope, truth_graph, &entity_ids)?;
                    let security_scope = principal
                        .get("security_scope")
                        .and_then(Value::as_str)
                        .map(str::trim)
                        .filter(|value| !value.is_empty())
                        .map(str::to_ascii_lowercase);
                    let (acl_mutations, acl_jobs) = derive_default_acl_graph_effects_from_records(
                        &scope,
                        &batch.mutations,
                        targets,
                        existing,
                        truth_graph,
                        &owner_id.to_ascii_lowercase(),
                        security_scope.as_deref(),
                        effect_id,
                    )
                    .map_err(kogwistar_store_sqlite::SqliteStoreError::TransactionAborted)?;
                    batch.mutations.extend(acl_mutations);
                    batch.index_jobs.extend(acl_jobs);
                }
                uow.apply_canonical_graph_mutation_batch(batch)?;
                for publication in publications {
                    let event = &publication.event;
                    let message_id = format!("msg:{}", event.event_id);
                    let payload_json = serde_json::to_string(event).map_err(|error| {
                        kogwistar_store_sqlite::SqliteStoreError::TransactionAborted(
                            format!("provider event serialization failed: {error}"),
                        )
                    })?;
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map_err(|error| {
                            kogwistar_store_sqlite::SqliteStoreError::TransactionAborted(
                                format!("system clock is before Unix epoch: {error}"),
                            )
                        })?
                        .as_secs() as i64;
                    if let Some(existing) = uow.get_projected_lane_message(&message_id)? {
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
                            return Err(kogwistar_store_sqlite::SqliteStoreError::RecordedRuntimeConflict(
                                "provider event lane projection conflicts with the existing event".to_owned(),
                            ));
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
                        })?;
                    }
                }
            }
            let handoff = input.handoff;
            let lane = uow
                .get_projected_lane_message(&handoff.message_id)?
                .ok_or_else(|| {
                    kogwistar_store_sqlite::SqliteStoreError::RecordedRuntimeConflict(
                        "claimed runtime lane does not exist".to_owned(),
                    )
                })?;
            validate_runtime_lane_executor(&lane, executor_target).map_err(|message| {
                kogwistar_store_sqlite::SqliteStoreError::RecordedRuntimeConflict(
                    message.to_owned(),
                )
            })?;
            let usage = input
                .effect
                .as_ref()
                .and_then(|effect| effect.usage.clone());
            let trace_events = input
                .effect
                .as_ref()
                .map(|effect| effect.trace_events.clone())
                .unwrap_or_default();
            let (applied, transition) = if let Some(transition) = input.transition {
                let applied = uow.apply_claimed_recorded_runtime_transition(
                    handoff,
                    transition.clone(),
                    false,
                )?;
                (applied, transition)
            } else {
                let effect = input.effect.expect("effect checked above");
                let result = effect.result.clone();
                let effect_id = effect.effect_id.clone();
                let applied = uow.apply_claimed_recorded_worker_effect(handoff.clone(), effect)?;
                let state = applied.reduced.state.clone();
                (
                    applied,
                    RecordedRuntimeTransition {
                        contract_version: RECORDED_RUNTIME_CONTRACT_VERSION,
                        transition_id: effect_id,
                        expected_event_seq: 0,
                        kind: RecordedTransitionKind::RecordedStepSuccess,
                        run_id: handoff.run_id,
                        workflow_id: state.workflow_id.clone(),
                        conversation_id: state.conversation_id.clone(),
                        user_id: None,
                        user_turn_node_id: None,
                        step_seq: state.last_step_seq,
                        node_id: state.last_node_id.clone(),
                        token_id: state.last_token_id.clone(),
                        parent_token_id: state.last_parent_token_id.clone(),
                        initial_state: None,
                        state_update: Vec::new(),
                        update: None,
                        state_schema: serde_json::Map::new(),
                        frontier: None,
                        result,
                        wait_reason: None,
                        resume_payload: None,
                        errors: Vec::new(),
                    },
                )
            };
            if !applied.idempotent {
                if let Some((scope, owner_id, principal)) = provider_event_ack_scope.as_ref() {
                    if !capability_allowed_for_principal(
                        principal,
                        &self.capabilities,
                        "write_graph.commit",
                        &["write_graph"],
                    ) {
                        return Err(kogwistar_store_sqlite::SqliteStoreError::RecordedRuntimeConflict(
                            "provider event run owner no longer has write_graph authority".to_owned(),
                        ));
                    }
                    let run = uow.get_server_run(&runtime_run_id)?;
                    if !run.as_ref().is_some_and(|run| {
                        run.user_id.as_deref() == Some(owner_id.as_str())
                    }) {
                        return Err(kogwistar_store_sqlite::SqliteStoreError::RecordedRuntimeConflict(
                            "provider event run owner changed before acknowledgement".to_owned(),
                        ));
                    }
                    for acknowledgement in &provider_event_acknowledgements {
                        apply_sqlite_provider_event_acknowledgement(
                            uow,
                            acknowledgement,
                            &runtime_run_id,
                            &applied.reduced.state,
                            &transition.transition_id,
                            scope,
                            owner_id,
                            principal,
                        )?;
                    }
                }
            }
            if applied.idempotent {
                return Ok(applied);
            }
            let frontier = &applied.reduced.state.frontier;
            let telemetry_run_id = transition.run_id.clone();
            let telemetry_step_id = transition.node_id.clone();
            let telemetry_token_id = transition.token_id.clone();
            let telemetry_effect_id = transition.transition_id.clone();
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
            let _completion_result = if unfinished || applied.reduced.state.status.is_terminal() {
                for (node_id, join_mask, token_id, parent_token_id) in &frontier.pending {
                    let existing_lanes = uow.list_projected_lane_messages(&LaneMessageFilter {
                        namespace: Some("workflow".to_owned()),
                        inbox_id: Some("workflow-runtime".to_owned()),
                        correlation_id: Some(transition.run_id.clone()),
                        ..LaneMessageFilter::default()
                    })?;
                    if active_runtime_lane_covers_token(&existing_lanes, token_id) {
                        continue;
                    }
                    let next_step_seq = applied.reduced.state.last_step_seq.saturating_add(1);
                    let message_id = format!(
                        "lane|{}",
                        kogwistar_contracts::stable_id(
                            "runtime.worker.request",
                            &[
                                transition.run_id.clone(),
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
                    if uow.get_projected_lane_message(&message_id)?.is_some() {
                        continue;
                    }
                    let op = applied
                        .reduced
                        .state
                        .state
                        .get("_rt_node_ops")
                        .and_then(Value::as_object)
                        .and_then(|ops| ops.get(node_id))
                        .cloned()
                        .unwrap_or_else(|| Value::String("noop".to_owned()));
                    let payload = RuntimeStepExecutePayload::from_recorded_state(
                        &applied.reduced.state,
                        RuntimeStepExecuteRequest {
                            node_id: node_id.clone(),
                            op,
                            join_mask: *join_mask,
                            token_id: token_id.clone(),
                            parent_token_id: parent_token_id.clone(),
                            step_seq: next_step_seq,
                            expected_event_seq: uow
                                .latest_server_run_event_seq(&transition.run_id)?,
                            resume_effect: None,
                        },
                    );
                    uow.project_lane_message(NewProjectedLaneMessage {
                        message_id,
                        namespace: "workflow".to_owned(),
                        purpose: "system".to_owned(),
                        inbox_id: "workflow-runtime".to_owned(),
                        conversation_id: transition.conversation_id.clone(),
                        recipient_id: runtime_worker_recipient_for_state(
                            &applied.reduced.state,
                            node_id,
                        ),
                        sender_id: "rust-scheduler".to_owned(),
                        msg_type: "workflow.step.execute".to_owned(),
                        status: "pending".to_owned(),
                        created_at: now,
                        available_at: now,
                        run_id: Some(transition.run_id.clone()),
                        step_id: Some(node_id.clone()),
                        correlation_id: Some(transition.run_id.clone()),
                        payload_json: Some(serde_json::to_string(&payload)?),
                        error_json: None,
                    })?;
                }
                applied
            } else {
                uow.apply_recorded_runtime_transition(
                    RecordedRuntimeTransition {
                        contract_version: RECORDED_RUNTIME_CONTRACT_VERSION,
                        transition_id: format!("complete-{}", transition.transition_id),
                        expected_event_seq: uow
                            .latest_server_run_event_seq(&transition.run_id)?,
                        kind: RecordedTransitionKind::Complete,
                        run_id: transition.run_id,
                        workflow_id: transition.workflow_id,
                        conversation_id: transition.conversation_id,
                        user_id: None,
                        user_turn_node_id: None,
                        step_seq: transition.step_seq,
                        node_id: transition.node_id,
                        token_id: transition.token_id,
                        parent_token_id: transition.parent_token_id,
                        initial_state: None,
                        state_update: Vec::new(),
                        update: None,
                        state_schema: serde_json::Map::new(),
                        frontier: Some(RuntimeFrontier::default()),
                        result: transition.result,
                        wait_reason: None,
                        resume_payload: None,
                        errors: Vec::new(),
                    },
                    false,
                )?
            };
            if !applied_idempotent {
                if let Some(usage) = usage {
                    uow.append_server_run_event(
                        &telemetry_run_id,
                        "workflow.usage.v1",
                        serde_json::to_string(&json!({
                            "run_id": telemetry_run_id,
                            "step_id": telemetry_step_id,
                            "token_id": telemetry_token_id,
                            "effect_id": telemetry_effect_id,
                            "usage": usage,
                        }))?,
                    )?;
                }
                for (trace_index, trace) in trace_events.into_iter().enumerate() {
                    uow.append_server_run_event(
                        &telemetry_run_id,
                        "workflow.trace.v1",
                        serde_json::to_string(&json!({
                            "run_id": telemetry_run_id,
                            "step_id": telemetry_step_id,
                            "token_id": telemetry_token_id,
                            "effect_id": telemetry_effect_id,
                            "trace_index": trace_index,
                            "trace": trace,
                        }))?,
                    )?;
                }
            }
            Ok(worker_result)
        });
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

}
