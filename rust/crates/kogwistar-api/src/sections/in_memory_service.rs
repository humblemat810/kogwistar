// PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/in_memory_service.rs#graph_read_effect; python=kogwistar/graph_query.py#GraphQuery.neighbors; fixture=contracts/golden/graph-acl-traversal-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/in_memory_mcp_traversal.rs#in_memory_mcp_acl_traversal_matches_shared_python_contract; python_test=tests/mcp/test_chat_mcp_tools.py#test_mcp_neighbors_preserve_acl_visibility_and_hide_topology
#[derive(Clone)]
struct InMemoryGraphApplicationService {
    store: InMemoryStore,
    capabilities: SharedCapabilityState,
    native_operations: kogwistar_runtime::NativeOperationRegistry,
    embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
    chat_completion_provider: Option<crate::native_answer_generation::OpenAiCompatibleChatConfig>,
    transient_run_deltas: crate::transient_deltas::TransientRunDeltas,
    search_index_fts: Option<Arc<Mutex<rusqlite::Connection>>>,
    owner_auth_store: Option<AuthStore>,
}

#[derive(Clone)]
struct InMemoryConversationAdmission {
    graph_batch: GraphMutationBatch,
    expected_projections: Vec<kogwistar_store_memory::GraphProjectionExpectation>,
    request_fingerprint: Option<String>,
    actor_user_id: String,
}

impl Default for InMemoryGraphApplicationService {
    fn default() -> Self {
        Self {
            store: InMemoryStore::default(),
            capabilities: Arc::new(Mutex::new(CapabilityState::default())),
            // PYTHON_REFERENCE_V1: rust=rust/crates/kogwistar-api/src/sections/in_memory_service.rs#default; python=kogwistar/conversation/agentic_answering.py#AgenticAnsweringAgent; rust_test=rust/crates/kogwistar-api/src/tests/answer_workflow_admission.rs#answer_workflow_admission_supports_per_node_python_and_registered_rust_executors; python_test=tests/workflow/test_agentic_answering_workflow_design_spec.py#test_answer_designs_match_shared_runtime_admission_contract
            native_operations: crate::native_answer_persist::registry()
                .expect("built-in Rust-native answer operation registry must be valid"),
            embedding_provider: None,
            chat_completion_provider: None,
            transient_run_deltas: crate::transient_deltas::TransientRunDeltas::default(),
            search_index_fts: None,
            owner_auth_store: None,
        }
    }
}

impl InMemoryGraphApplicationService {
    /// Compose trusted Rust-native application operations for the in-memory
    /// backend, matching the persistent runtime service extension point.
    #[cfg(test)]
    fn with_native_operations(
        mut self,
        native_operations: kogwistar_runtime::NativeOperationRegistry,
    ) -> Self {
        self.native_operations = native_operations;
        self
    }

    async fn resolve_inline_native_owner_principal(
        &self,
        run_id: &str,
        required_capability: &str,
    ) -> Result<Value, InlineOwnerPrincipalError> {
        let run = kogwistar_store::ServerRunReadStore::server_run(&self.store, run_id)
            .await
            .map_err(|error| InlineOwnerPrincipalError::Unavailable(error.to_string()))?
            .ok_or_else(|| {
                InlineOwnerPrincipalError::Denied("native worker run is missing".to_owned())
            })?;
        let owner_id = run
            .user_id
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| {
                InlineOwnerPrincipalError::Denied(
                    "native worker run has no authenticated owner".to_owned(),
                )
            })?;
        let created =
            kogwistar_store::ServerRunReadStore::server_run_events(&self.store, run_id, 0, 1)
                .await
                .map_err(|error| InlineOwnerPrincipalError::Unavailable(error.to_string()))?
                .into_iter()
                .find(|event| event.event_type == "run.created")
                .ok_or_else(|| {
                    InlineOwnerPrincipalError::Denied(
                        "native worker run has no admission event".to_owned(),
                    )
                })?;
        let created_payload: Value = serde_json::from_str(&created.payload_json).map_err(|_| {
            InlineOwnerPrincipalError::Denied("native worker admission event is invalid".to_owned())
        })?;
        let binding: RuntimeOwnerBindingV1 = serde_json::from_value(
            created_payload
                .get("runtime_owner_binding")
                .cloned()
                .ok_or_else(|| {
                    InlineOwnerPrincipalError::Denied(
                        "native worker run has no trusted owner binding".to_owned(),
                    )
                })?,
        )
        .map_err(|_| {
            InlineOwnerPrincipalError::Denied("native worker owner binding is invalid".to_owned())
        })?;
        if binding.user_id != owner_id {
            return Err(InlineOwnerPrincipalError::Denied(
                "native worker owner binding does not match the durable run owner".to_owned(),
            ));
        }
        let auth_store = self.owner_auth_store.as_ref().ok_or_else(|| {
            InlineOwnerPrincipalError::Unavailable(
                "native worker owner revalidation is not configured".to_owned(),
            )
        })?;
        let user = auth_store
            .auth_user(owner_id)
            .await
            .map_err(InlineOwnerPrincipalError::Unavailable)?
            .filter(|user| user.is_active)
            .ok_or_else(|| {
                InlineOwnerPrincipalError::Denied(
                    "native worker run owner is missing or inactive".to_owned(),
                )
            })?;
        revalidated_runtime_owner_principal(
            &binding,
            &user,
            &self.capabilities,
            required_capability,
        )
        .map_err(InlineOwnerPrincipalError::Denied)
    }

    // PYTHON_REFERENCE_V1: rust=rust/crates/kogwistar-api/src/sections/in_memory_service.rs#execute_native_step_once; python=kogwistar/runtime/rust_worker.py#RustRuntimeWorker; fixture=contracts/golden/runtime-native-noop-v1.json,contracts/golden/runtime-native-increment-counter-v1.json,contracts/golden/runtime-native-emit-result-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/in_memory_provider_events.rs#in_memory_inline_native_result_is_persisted_and_completes_run; python_test=tests/runtime/test_rust_runtime_python_worker.py#test_python_worker_increment_counter_matches_shared_native_effect_golden; note=Rust-native operations stay on Rust path; Python worker explicitly rejects native-target operations.
    async fn dead_letter_native_lane(
        &self,
        message_id: &str,
        claimed_by: &str,
        reason: &str,
    ) -> Result<(), String> {
        kogwistar_store::LaneMessageWriteStore::dead_letter_projected_lane_message(
            &self.store,
            message_id,
            claimed_by,
            Some(json!({"error": reason}).to_string()),
        )
        .await
        .map_err(|error| error.to_string())
    }

    async fn claim_in_memory_native_provider_event(
        &self,
        request: &kogwistar_runtime::RuntimeNativeStepRequest,
    ) -> Result<Option<Value>, String> {
        let principal = self
            .resolve_inline_native_owner_principal(&request.run_id, "read_graph")
            .await
            .map_err(|_| "provider-event owner authorization is unavailable".to_owned())?;
        let security_scope = security_scope_value(&principal);
        let namespace = security_scope["storage_namespace"]
            .as_str()
            .ok_or_else(|| "provider-event owner has no storage namespace".to_owned())?;
        let scope = GraphScope {
            namespace: namespace.to_owned(),
            workspace_id: security_scope["workspace"]
                .as_str()
                .filter(|value| !value.is_empty())
                .map(str::to_owned),
            graph_space: Some("conversation".to_owned()),
        };
        let acl_principal =
            graph_acl_principal_from_claims(&principal, security_scope["security_scope"].as_str());
        let claimed_by = format!("rust-native-agent-event:{}", uuid::Uuid::new_v4());
        let mut claimed = self
            .store
            .claim_visible_provider_event(
                &scope,
                &request.conversation_id,
                &acl_principal,
                LaneMessageClaimFilter {
                    namespace: scope.namespace.clone(),
                    inbox_id: "agent-provider-events".to_owned(),
                    claimed_by: claimed_by.clone(),
                    message_ids: None,
                    run_id: Some(request.run_id.clone()),
                    msg_type: Some("agent.provider_event".to_owned()),
                    recipient_id: Some("agent-event-consumer".to_owned()),
                    limit: 1,
                    lease_seconds: 60,
                },
            )
            .map_err(|_| "provider-event claim failed closed".to_owned())?;
        let Some(lane) = claimed.pop() else {
            return Ok(None);
        };
        let event: kogwistar_contracts::DurableProviderEventV1 =
            serde_json::from_str(lane.payload_json.as_deref().unwrap_or(""))
                .map_err(|_| "claimed provider event payload is invalid".to_owned())?;
        if lane.namespace != namespace
            || lane.purpose != "agent_event"
            || lane.inbox_id != "agent-provider-events"
            || lane.conversation_id != request.conversation_id
            || lane.recipient_id != "agent-event-consumer"
            || lane.sender_id != "agent-runtime"
            || lane.msg_type != "agent.provider_event"
            || lane.status != "claimed"
            || lane.claimed_by.as_deref() != Some(claimed_by.as_str())
            || lane.run_id.as_deref() != Some(request.run_id.as_str())
            || lane.correlation_id.as_deref() != Some(event.event_id.as_str())
            || lane.message_id != format!("msg:{}", event.event_id)
            || event.trace.run_id != request.run_id
        {
            return Err(
                "claimed provider event does not match its authorized run and lane".to_owned(),
            );
        }
        Ok(Some(json!({
            "event": event,
            "claim": {
                "message_id": lane.message_id,
                "claimed_by": claimed_by,
                "event_id": event.event_id,
            }
        })))
    }

    async fn execute_native_step_once(&self) -> Result<bool, String> {
        let claimed_by = format!("rust-native-worker#{}", uuid::Uuid::new_v4());
        let mut claimed =
            kogwistar_store::LaneMessageWriteStore::claim_projected_lane_messages_filtered(
                &self.store,
                LaneMessageClaimFilter {
                    namespace: "workflow".to_owned(),
                    inbox_id: "workflow-runtime".to_owned(),
                    claimed_by: claimed_by.clone(),
                    message_ids: None,
                    run_id: None,
                    msg_type: Some("workflow.step.execute".to_owned()),
                    recipient_id: Some("rust-native-worker".to_owned()),
                    limit: 1,
                    lease_seconds: 60,
                },
            )
            .await
            .map_err(|error| error.to_string())?;
        let Some(lane) = claimed.pop() else {
            return Ok(false);
        };
        let claim_message_id = lane.message_id.clone();
        let claim_owner = claimed_by.clone();
        let operation = async move {
            let message_id = lane.message_id.clone();
            let run_id = lane.run_id.clone().unwrap_or_default();
            let cancel_requested =
                kogwistar_store::ServerRunReadStore::server_run(&self.store, &run_id)
                    .await
                    .map_err(|error| error.to_string())?
                    .is_some_and(|run| run.cancel_requested);
            if cancel_requested {
                kogwistar_store::LaneMessageWriteStore::update_projected_lane_message_status(
                    &self.store,
                    &message_id,
                    "cancelled",
                    None,
                )
                .await
                .map_err(|error| error.to_string())?;
                return Ok(true);
            }
            let payload: Value = match lane
                .payload_json
                .as_deref()
                .map(serde_json::from_str)
                .transpose()
            {
                Ok(Some(payload)) => payload,
                Ok(None) => json!({}),
                Err(error) => {
                    self.dead_letter_native_lane(
                        &message_id,
                        &claimed_by,
                        "invalid native step payload",
                    )
                    .await?;
                    return Err(format!("invalid native step payload: {error}"));
                }
            };
            let operation = payload["op"]
                .as_str()
                .map(str::to_owned)
                .and_then(|operation| {
                    kogwistar_runtime::native_operation_for_target(
                        &operation,
                        payload["executor_target"].as_str(),
                    )
                    .map(str::to_owned)
                });
            let expected_event_seq = payload["expected_event_seq"].as_i64().unwrap_or_default();
            let work = runtime_claimed_work(lane, &claimed_by, run_id, payload, expected_event_seq);
            let (handoff, mut native_request) = match native_request_for_claimed_work(&work) {
                Ok(value) => value,
                Err(error) => {
                    self.dead_letter_native_lane(
                        &message_id,
                        &claimed_by,
                        "native step request validation failed",
                    )
                    .await?;
                    return Err(error);
                }
            };
            if !self
                .native_operations
                .supports(&native_request.operation, &native_request.op_version)
            {
                self.dead_letter_native_lane(
                    &message_id,
                    &claimed_by,
                    "native operation is not registered",
                )
                .await?;
                return Ok(true);
            }
            let result = match if operation.as_deref() == Some("agent.event.request") {
                self.execute_in_memory_native_provider_event_request(
                    handoff.clone(),
                    native_request,
                )
                .await
            } else {
                if operation.as_deref() == Some("agent.event.receive") {
                    match self
                        .claim_in_memory_native_provider_event(&native_request)
                        .await
                    {
                        Ok(Some(context)) => {
                            native_request.ephemeral_context.insert(
                                "authorized_provider_event".to_owned(),
                                context["event"].clone(),
                            );
                            native_request.ephemeral_context.insert(
                                "authorized_provider_event_claim".to_owned(),
                                context["claim"].clone(),
                            );
                        }
                        Ok(None) => {
                            kogwistar_store::LaneMessageWriteStore::requeue_projected_lane_message(
                                &self.store,
                                &message_id,
                                &claimed_by,
                                None,
                                1,
                            )
                            .await
                            .map_err(|error| error.to_string())?;
                            return Ok(true);
                        }
                        Err(error) => return Err(error),
                    }
                }
                native_result_from_request(handoff.clone(), native_request, &self.native_operations)
            } {
                Ok(result) => result,
                Err(error) => {
                    self.dead_letter_native_lane(
                        &message_id,
                        &claimed_by,
                        "native noop execution failed",
                    )
                    .await?;
                    return Err(error);
                }
            };
            let Some(effect) = result.effect else {
                self.dead_letter_native_lane(
                    &message_id,
                    &claimed_by,
                    "native noop did not produce a worker effect",
                )
                .await?;
                return Err("native noop did not produce a worker effect".to_owned());
            };
            let commit = if effect.graph_mutation_batch.is_some()
                || effect.graph_mutation_batch_v2.is_some()
            {
                if !effect.provider_event_acknowledgements.is_empty() {
                    self.dead_letter_native_lane(
                    &message_id,
                    &claimed_by,
                    "combined graph effects and provider-event acknowledgements are unsupported",
                )
                .await?;
                    return Err("combined graph effects and provider-event acknowledgements are unsupported".to_owned());
                }
                match self
                    .in_memory_native_graph_effect_batch(&handoff, &effect)
                    .await
                {
                    Ok((batch, acl_snapshot)) => self
                        .store
                        .apply_claimed_recorded_worker_effect_with_graph_effects(
                            handoff,
                            effect,
                            batch,
                            acl_snapshot,
                        ),
                    Err(error) => {
                        self.dead_letter_native_lane(
                            &message_id,
                            &claimed_by,
                            "native graph effect preparation failed",
                        )
                        .await?;
                        return Err(error);
                    }
                }
            } else if effect.provider_event_acknowledgements.is_empty() {
                self.store
                    .apply_claimed_recorded_worker_effect(handoff, effect)
            } else {
                match self
                    .in_memory_provider_event_ack_graph_batch(&handoff, &effect)
                    .await
                {
                    Ok(batch) => self
                        .store
                        .apply_claimed_recorded_worker_effect_with_graph(handoff, effect, batch),
                    Err(error) => {
                        self.dead_letter_native_lane(
                            &message_id,
                            &claimed_by,
                            "provider event acknowledgement graph preparation failed",
                        )
                        .await?;
                        return Err(error);
                    }
                }
            };
            if let Err(error) = commit {
                self.dead_letter_native_lane(
                    &message_id,
                    &claimed_by,
                    "in-memory worker effect could not be committed",
                )
                .await?;
                return Err(error.to_string());
            }
            Ok::<bool, String>(true)
        };
        match kogwistar_engine::with_lane_claim_renewal(
            &self.store,
            &claim_message_id,
            &claim_owner,
            60,
            operation,
        )
        .await
        {
            Ok(Some(result)) => result,
            Ok(None) => {
                Err("native runtime lane lease was lost; result was not committed".to_owned())
            }
            Err(error) => Err(format!("native runtime lane lease renewal failed: {error}")),
        }
    }

    async fn in_memory_native_graph_effect_batch(
        &self,
        handoff: &RecordedWorkerHandoff,
        effect: &RecordedWorkerSuccessEffect,
    ) -> Result<
        (
            GraphMutationBatch,
            Option<kogwistar_store_memory::GraphAclSnapshot>,
        ),
        String,
    > {
        let (mutations, index_jobs, truth_graph) = match (
            effect.graph_mutation_batch.as_ref(),
            effect.graph_mutation_batch_v2.as_ref(),
        ) {
            (Some(_), Some(_)) => {
                return Err("worker effect cannot carry both graph batch versions".to_owned());
            }
            (Some(batch), None) => {
                let input: GraphMutationBatchInput = batch.clone();
                let mutations = graph_mutation_batch_value(input.clone())?;
                validate_graph_mutation_envelopes(&input, &mutations)?;
                validate_runtime_graph_mutations(&mutations)?;
                let index_jobs = graph_index_jobs_value(&input, &mutations)?;
                (mutations, index_jobs, None)
            }
            (None, Some(batch)) => {
                let (mutations, index_jobs) = graph_mutation_batch_v2_value(batch.clone())?;
                validate_runtime_graph_mutations(&mutations)?;
                (
                    mutations,
                    index_jobs,
                    Some(batch.default_acl.truth_graph.clone()),
                )
            }
            (None, None) => return Err("worker graph effect is missing".to_owned()),
        };
        if effect.status != kogwistar_runtime::RuntimeWorkerEffectStatus::Success {
            return Err("graph effects require a successful worker result".to_owned());
        }

        let principal = self
            .resolve_inline_native_owner_principal(&handoff.run_id, "write_graph")
            .await
            .map_err(|_| "native graph effect owner authorization failed closed".to_owned())?;
        if !capability_allowed_for_principal(
            &principal,
            &self.capabilities,
            "write_graph",
            &["write_graph"],
        ) || !principal_strings(&principal, "capabilities")
            .iter()
            .any(|capability| capability == "write_graph")
        {
            return Err("native graph effect requires current write_graph authority".to_owned());
        }
        let run = kogwistar_store::ServerRunReadStore::server_run(&self.store, &handoff.run_id)
            .await
            .map_err(|_| "native graph effect run lookup failed".to_owned())?
            .ok_or_else(|| "native graph effect run is missing".to_owned())?;
        let owner_id = authenticated_run_owner_id(&principal)
            .filter(|owner| Some(*owner) == run.user_id.as_deref())
            .ok_or_else(|| "native graph effect owner differs from durable run owner".to_owned())?
            .to_ascii_lowercase();
        let security_scope = security_scope_value(&principal);
        if mutations
            .iter()
            .any(|mutation| !graph_mutation_scope_allowed(&mutation.scope, &security_scope))
        {
            return Err("native graph effect scope exceeds run-owner scope".to_owned());
        }

        let mut graph_batch = GraphMutationBatch {
            mutations,
            index_jobs,
        };
        let acl_snapshot = if let Some(truth_graph) = truth_graph {
            let scope = graph_batch
                .mutations
                .first()
                .map(|mutation| mutation.scope.clone())
                .ok_or_else(|| "ACL-aware graph batch must not be empty".to_owned())?;
            let targets = acl_targets_for_graph_mutations(&graph_batch.mutations).0;
            let target_ids = acl_target_entity_ids(&targets);
            let mut existing_acl = Vec::new();
            for target_id in &target_ids {
                let mut filter = MetadataFilter::default();
                filter
                    .equals
                    .insert("entity_type".to_owned(), json!("acl_record"));
                filter
                    .equals
                    .insert("acl_truth_graph".to_owned(), json!(truth_graph));
                filter
                    .equals
                    .insert("acl_target_entity_id".to_owned(), json!(target_id));
                existing_acl.extend(
                    self.store
                        .graph_projection_records(GraphProjectionRead {
                            scope: scope.clone(),
                            table: "gke_nodes".to_owned(),
                            ids: None,
                            metadata: filter,
                            limit: 10000,
                        })
                        .await
                        .map_err(|_| "native graph ACL lookup failed closed".to_owned())?,
                );
            }
            let (acl_mutations, acl_jobs) = derive_default_acl_graph_effects_from_records(
                &scope,
                &graph_batch.mutations,
                targets,
                existing_acl.clone(),
                AclDefaultsContext {
                    truth_graph: &truth_graph,
                    owner_id: &graph_acl_principal_id(&principal),
                    security_scope: security_scope["security_scope"].as_str(),
                    effect_id: &effect.effect_id,
                },
            )?;
            graph_batch.mutations.extend(acl_mutations);
            graph_batch.index_jobs.extend(acl_jobs);
            Some(kogwistar_store_memory::GraphAclSnapshot {
                scope,
                truth_graph,
                target_entity_ids: target_ids,
                records: existing_acl,
            })
        } else {
            None
        };

        let current = self
            .resolve_inline_native_owner_principal(&handoff.run_id, "write_graph")
            .await
            .map_err(|_| "native graph owner authorization changed before commit".to_owned())?;
        if authenticated_run_owner_id(&current) != Some(owner_id.as_str())
            || security_scope_value(&current) != security_scope
            || !principal_strings(&current, "capabilities")
                .iter()
                .any(|capability| capability == "write_graph")
        {
            return Err("native graph owner capability or scope changed before commit".to_owned());
        }
        Ok((graph_batch, acl_snapshot))
    }

    async fn in_memory_provider_event_ack_graph_batch(
        &self,
        handoff: &RecordedWorkerHandoff,
        effect: &RecordedWorkerSuccessEffect,
    ) -> Result<GraphMutationBatch, String> {
        let acknowledgement = effect
            .provider_event_acknowledgements
            .first()
            .filter(|_| effect.provider_event_acknowledgements.len() == 1)
            .ok_or_else(|| "provider event acknowledgement count is invalid".to_owned())?;
        let principal = self
            .resolve_inline_native_owner_principal(&handoff.run_id, "write_graph")
            .await
            .map_err(|_| {
                "provider-event acknowledgement owner authorization failed closed".to_owned()
            })?;
        let run = kogwistar_store::ServerRunReadStore::server_run(&self.store, &handoff.run_id)
            .await
            .map_err(|_| "provider-event run lookup failed".to_owned())?
            .ok_or_else(|| "provider-event run is missing".to_owned())?;
        let owner_id = authenticated_run_owner_id(&principal)
            .filter(|owner| Some(*owner) == run.user_id.as_deref())
            .ok_or_else(|| "provider-event run owner changed before acknowledgement".to_owned())?
            .to_ascii_lowercase();

        let mut events = Vec::new();
        let mut after_seq = 0;
        loop {
            let page = kogwistar_store::ServerRunReadStore::server_run_events(
                &self.store,
                &handoff.run_id,
                after_seq,
                1000,
            )
            .await
            .map_err(|_| "provider-event run history lookup failed".to_owned())?;
            let full_page = page.len() == 1000;
            if let Some(last) = page.last() {
                after_seq = last.seq;
            }
            events.extend(page);
            if !full_page {
                break;
            }
        }
        let transitions = events
            .iter()
            .filter(|event| event.event_type == "workflow.recorded_transition.v1")
            .cloned()
            .collect::<Vec<_>>();
        let latest_state = transitions
            .iter()
            .rev()
            .find_map(|event| {
                serde_json::from_str::<Value>(&event.payload_json)
                    .ok()?
                    .get("reduced")?
                    .get("state")
                    .cloned()
            })
            .ok_or_else(|| "provider-event run has no durable runtime state".to_owned())?;
        let workflow_id = latest_state["workflow_id"]
            .as_str()
            .filter(|value| !value.is_empty())
            .ok_or_else(|| "provider-event workflow identity is missing".to_owned())?;
        let conversation_id = latest_state["conversation_id"]
            .as_str()
            .filter(|value| !value.is_empty())
            .ok_or_else(|| "provider-event conversation identity is missing".to_owned())?;
        if latest_state["run_id"].as_str() != Some(handoff.run_id.as_str()) {
            return Err("provider-event runtime state belongs to another run".to_owned());
        }
        let handler_step_seq = latest_state["state"]["agent_event_handling_step_seq"]
            .as_i64()
            .filter(|seq| *seq >= 0)
            .ok_or_else(|| "provider-event handler step sequence is invalid".to_owned())?;
        let step_seq = latest_state["last_step_seq"]
            .as_i64()
            .filter(|seq| *seq >= 0)
            .ok_or_else(|| "provider-event acknowledgement step sequence is invalid".to_owned())?
            .saturating_add(1);
        let event_lane = kogwistar_store::LaneMessageReadStore::projected_lane_message(
            &self.store,
            &acknowledgement.message_id,
        )
        .await
        .map_err(|_| "provider-event lane lookup failed".to_owned())?
        .ok_or_else(|| "provider-event lane is missing".to_owned())?;
        let event: kogwistar_contracts::DurableProviderEventV1 = event_lane
            .payload_json
            .as_deref()
            .and_then(|payload| serde_json::from_str(payload).ok())
            .ok_or_else(|| "provider-event payload is invalid".to_owned())?;
        if event.event_id != acknowledgement.event_id
            || event.trace.run_id != handoff.run_id
            || event_lane.conversation_id != conversation_id
        {
            return Err("provider-event identity or conversation changed".to_owned());
        }
        let mut referenced_step_seqs = vec![handler_step_seq];
        if let Some(producer_step_seq) = event.trace.producer_step_seq {
            if producer_step_seq < 0 {
                return Err("provider-event producer step sequence is invalid".to_owned());
            }
            referenced_step_seqs.push(producer_step_seq);
        }
        referenced_step_seqs.sort_unstable();
        referenced_step_seqs.dedup();
        let snapshots = provider_event_step_snapshots(
            transitions,
            &handoff.run_id,
            conversation_id,
            workflow_id,
            &referenced_step_seqs,
        )?;
        if handler_step_seq >= step_seq
            || event
                .trace
                .producer_step_seq
                .is_some_and(|producer_step_seq| producer_step_seq >= step_seq)
        {
            return Err("provider-event causal steps must precede acknowledgement".to_owned());
        }
        let security_scope = security_scope_value(&principal);
        let storage_namespace = latest_state["state"]["_rt_graph_scope"]["storage_namespace"]
            .as_str()
            .filter(|value| !value.is_empty())
            .ok_or_else(|| "provider-event graph namespace is missing".to_owned())?;
        let scope = GraphScope {
            namespace: storage_namespace.to_owned(),
            workspace_id: security_scope["workspace"]
                .as_str()
                .filter(|value| !value.is_empty())
                .map(str::to_owned),
            graph_space: Some("conversation".to_owned()),
        };
        if !graph_mutation_scope_allowed(&scope, &security_scope) {
            return Err("provider-event graph scope changed before acknowledgement".to_owned());
        }
        let workflow_node_id = handoff.step_id.as_str();
        let mut mutations = provider_event_ack_graph_mutations(ProviderEventAckGraphInput {
            event_id: &acknowledgement.event_id,
            run_id: &handoff.run_id,
            conversation_id,
            workflow_id,
            workflow_node_id,
            step_seq,
            effect_id: &effect.effect_id,
            scope: &scope,
        });
        mutations.extend(provider_event_step_link_mutations(
            &acknowledgement.event_id,
            &handoff.run_id,
            conversation_id,
            event.trace.producer_step_seq,
            handler_step_seq,
            &snapshots,
            &scope,
        ));
        let mut linked_step_nodes = BTreeSet::new();
        mutations.retain(|mutation| {
            mutation.entity_kind != "node" || linked_step_nodes.insert(mutation.record.id.clone())
        });
        let (targets, _) = acl_targets_for_graph_mutations(&mutations);
        let mut existing_acl = Vec::new();
        for target_id in acl_target_entity_ids(&targets) {
            let mut acl_filter = MetadataFilter::default();
            acl_filter
                .equals
                .insert("entity_type".to_owned(), json!("acl_record"));
            acl_filter
                .equals
                .insert("acl_truth_graph".to_owned(), json!("conversation"));
            acl_filter
                .equals
                .insert("acl_target_entity_id".to_owned(), json!(target_id));
            existing_acl.extend(
                self.store
                    .graph_projection_records(GraphProjectionRead {
                        scope: scope.clone(),
                        table: "gke_nodes".to_owned(),
                        ids: None,
                        metadata: acl_filter,
                        limit: 10000,
                    })
                    .await
                    .map_err(|_| "provider-event ACL projection lookup failed".to_owned())?,
            );
        }
        let (acl_mutations, acl_jobs) = derive_default_acl_graph_effects_from_records(
            &scope,
            &mutations,
            targets,
            existing_acl,
            AclDefaultsContext {
                truth_graph: "conversation",
                owner_id: &graph_acl_principal_id(&principal),
                security_scope: security_scope["security_scope"].as_str(),
                effect_id: &acknowledgement.event_id,
            },
        )?;
        mutations.extend(acl_mutations);

        let current_principal = self
            .resolve_inline_native_owner_principal(&handoff.run_id, "write_graph")
            .await
            .map_err(|_| "provider-event owner authorization changed before commit".to_owned())?;
        if authenticated_run_owner_id(&current_principal) != Some(owner_id.as_str())
            || security_scope_value(&current_principal) != security_scope
            || !principal_strings(&current_principal, "capabilities")
                .iter()
                .any(|capability| capability == "write_graph")
        {
            return Err(
                "provider-event owner capability or scope changed before commit".to_owned(),
            );
        }
        Ok(GraphMutationBatch {
            mutations,
            index_jobs: acl_jobs,
        })
    }

    async fn execute_in_memory_native_provider_event_request(
        &self,
        handoff: RecordedWorkerHandoff,
        mut request: kogwistar_runtime::RuntimeNativeStepRequest,
    ) -> Result<RuntimeWorkerResultRequest, String> {
        let Some(provider) = self.chat_completion_provider.as_ref() else {
            return Ok(failed_native_result(
                handoff,
                &request,
                "KOGWISTAR_CHAT_PROVIDER_NOT_CONFIGURED",
                "Rust-native provider-event streaming requires a configured chat provider",
            ));
        };
        let provider_request: kogwistar_contracts::ProviderRequestV1 = serde_json::from_value(
            request
                .state
                .get("provider_request")
                .cloned()
                .ok_or_else(|| "agent.event.request requires provider_request state".to_owned())?,
        )
        .map_err(|_| "provider_request does not match the v1 contract".to_owned())?;
        provider_request
            .validate()
            .map_err(|_| "provider_request does not match the v1 contract".to_owned())?;
        if provider_request.model != provider.model() {
            return Err("provider_request model does not match the configured model".to_owned());
        }
        let principal = self
            .resolve_inline_native_owner_principal(&request.run_id, "write_graph")
            .await
            .map_err(|_| "provider-event request owner authorization is unavailable".to_owned())?;
        if !capability_allowed_for_principal(
            &principal,
            &self.capabilities,
            "write_graph.provider_event.request",
            &["write_graph"],
        ) || !principal_strings(&principal, "capabilities")
            .iter()
            .any(|capability| capability == "write_graph")
        {
            return Err(
                "provider-event request requires current run-owner write_graph authority"
                    .to_owned(),
            );
        }
        let run = kogwistar_store::ServerRunReadStore::server_run(&self.store, &request.run_id)
            .await
            .map_err(|_| "provider-event run could not be verified".to_owned())?
            .ok_or_else(|| "provider-event run is missing".to_owned())?;
        if run.conversation_id != request.conversation_id {
            return Err("provider-event conversation does not match the admitted run".to_owned());
        }

        use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
        let client_sequence = Arc::new(AtomicI64::new(
            request
                .state
                .get("agent_event_client_sse_sequence")
                .and_then(Value::as_i64)
                .filter(|sequence| *sequence >= 0)
                .unwrap_or(0),
        ));
        let semantic_count = Arc::new(AtomicI64::new(0));
        let provider_sequence = Arc::new(AtomicI64::new(0));
        let saw_failure = Arc::new(AtomicBool::new(false));
        let summaries = Arc::new(Mutex::new(Vec::<Value>::new()));
        let sink_error = Arc::new(Mutex::new(None::<String>));
        let provider_id = "openai_compatible_chat_completions";
        let model = provider.model().to_owned();
        let stream_result = provider
            .stream_request_events(&provider_request, |event| {
                let client_sequence = Arc::clone(&client_sequence);
                let semantic_count = Arc::clone(&semantic_count);
                let provider_sequence = Arc::clone(&provider_sequence);
                let saw_failure = Arc::clone(&saw_failure);
                let summaries = Arc::clone(&summaries);
                let sink_error = Arc::clone(&sink_error);
                let request = request.clone();
                let model = model.clone();
                async move {
                    if let kogwistar_contracts::ProviderEventV1::TextDelta { delta, .. } = &event {
                        if delta.chars().count() > 4096 {
                            return Err(
                                "transient text delta exceeds the configured size limit".to_owned()
                            );
                        }
                        let sequence = client_sequence
                            .fetch_add(1, Ordering::Relaxed)
                            .saturating_add(1);
                        let _ = self
                            .transient_run_deltas
                            .publish(&request.run_id, sequence, delta);
                        return Ok(());
                    }
                    let event_value = serde_json::to_value(&event)
                        .map_err(|_| "provider event could not be encoded".to_owned())?;
                    provider_sequence.fetch_max(
                        event_value
                            .get("provider_sequence")
                            .and_then(Value::as_i64)
                            .unwrap_or(0),
                        Ordering::Relaxed,
                    );
                    if matches!(
                        event,
                        kogwistar_contracts::ProviderEventV1::ResponseFailed { .. }
                    ) {
                        saw_failure.store(true, Ordering::Relaxed);
                    }
                    match self
                        .persist_in_memory_native_stream_semantic_event(
                            &request,
                            event,
                            provider_id,
                            &model,
                            client_sequence.load(Ordering::Relaxed),
                        )
                        .await
                    {
                        Ok(summary) => {
                            semantic_count.fetch_add(1, Ordering::Relaxed);
                            if let Ok(mut values) = summaries.lock() {
                                values.push(summary);
                            }
                            Ok(())
                        }
                        Err(error) => {
                            if let Ok(mut saved) = sink_error.lock() {
                                *saved = Some(error.clone());
                            }
                            Err(error)
                        }
                    }
                }
            })
            .await;
        if let Some(error) = sink_error.lock().ok().and_then(|mut saved| saved.take()) {
            return Err(error);
        }
        if (stream_result.is_err() || semantic_count.load(Ordering::Relaxed) == 0)
            && !saw_failure.load(Ordering::Relaxed)
        {
            let failure = kogwistar_contracts::ProviderEventV1::ResponseFailed {
                contract_version: kogwistar_contracts::PROVIDER_EVENT_CONTRACT_VERSION.to_owned(),
                provider_event_id: None,
                provider_sequence: Some(provider_sequence.load(Ordering::Relaxed)),
                response_id: None,
                error_code: if stream_result.is_err() {
                    "provider_stream_error"
                } else {
                    "empty_provider_stream"
                }
                .to_owned(),
                message: "provider request did not complete with semantic output".to_owned(),
                retryable: false,
            };
            let summary = self
                .persist_in_memory_native_stream_semantic_event(
                    &request,
                    failure,
                    provider_id,
                    &model,
                    client_sequence.load(Ordering::Relaxed),
                )
                .await?;
            semantic_count.fetch_add(1, Ordering::Relaxed);
            summaries
                .lock()
                .map_err(|_| "provider trace summary is unavailable".to_owned())?
                .push(summary);
        }
        request.ephemeral_context.insert(
            "provider_event_request_result".to_owned(),
            json!({
                "provider_sequence": provider_sequence.load(Ordering::Relaxed),
                "client_sse_sequence": client_sequence.load(Ordering::Relaxed),
                "semantic_event_count": semantic_count.load(Ordering::Relaxed),
            }),
        );
        request.ephemeral_context.insert(
            "_provider_event_trace_summaries".to_owned(),
            Value::Array(
                summaries
                    .lock()
                    .map_err(|_| "provider trace summary is unavailable".to_owned())?
                    .clone(),
            ),
        );
        request
            .ephemeral_context
            .insert("provider_model".to_owned(), json!(model));
        native_result_from_request(handoff, request, &self.native_operations)
    }

    async fn persist_in_memory_native_stream_semantic_event(
        &self,
        request: &kogwistar_runtime::RuntimeNativeStepRequest,
        event: kogwistar_contracts::ProviderEventV1,
        provider_id: &str,
        model: &str,
        client_sse_sequence: i64,
    ) -> Result<Value, String> {
        let summary = crate::native_answer_generation::semantic_event_trace_summary(&event)
            .ok_or_else(|| "provider emitted a non-semantic event".to_owned())?;
        let event_value = serde_json::to_value(&event)
            .map_err(|_| "provider event could not be encoded".to_owned())?;
        let trace_context = kogwistar_runtime::native_trace_context_fields(request);
        let attempt = trace_context
            .get("attempt")
            .and_then(Value::as_i64)
            .filter(|value| *value > 0)
            .unwrap_or(1);
        let trace = kogwistar_contracts::EventTraceMetadataV1 {
            run_id: request.run_id.clone(),
            turn_id: request
                .turn_node_id
                .clone()
                .or_else(|| {
                    request
                        .state
                        .get("turn_node_id")
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                })
                .unwrap_or_default(),
            workflow_node_id: request.node_id.clone(),
            step_id: request.step_seq.to_string(),
            attempt,
            producer_step_seq: Some(request.step_seq),
            producer_attempt: Some(attempt),
            trace_id: trace_context
                .get("trace_id")
                .and_then(Value::as_str)
                .map(str::to_owned),
            span_id: trace_context
                .get("span_id")
                .and_then(Value::as_str)
                .map(str::to_owned),
            parent_span_id: trace_context
                .get("parent_span_id")
                .and_then(Value::as_str)
                .map(str::to_owned),
            provider_id: provider_id.to_owned(),
            provider_response_id: event_value
                .get("response_id")
                .and_then(Value::as_str)
                .map(str::to_owned),
            provider_sequence: event_value.get("provider_sequence").and_then(Value::as_i64),
            client_sse_sequence: Some(client_sse_sequence),
        };
        let durable = kogwistar_contracts::durable_provider_event(&trace, event);
        let principal = self
            .resolve_inline_native_owner_principal(&request.run_id, "write_graph")
            .await
            .map_err(|_| {
                "provider-event owner authorization is unavailable at commit".to_owned()
            })?;
        let security_scope = security_scope_value(&principal);
        if !capability_allowed_for_principal(
            &principal,
            &self.capabilities,
            "write_graph.provider_event.commit",
            &["write_graph"],
        ) || !principal_strings(&principal, "capabilities")
            .iter()
            .any(|value| value == "write_graph")
        {
            return Err(
                "provider-event write_graph capability was revoked before commit".to_owned(),
            );
        }
        let scope = GraphScope {
            namespace: security_scope["storage_namespace"]
                .as_str()
                .filter(|value| !value.trim().is_empty())
                .ok_or_else(|| "provider-event owner has no storage namespace".to_owned())?
                .to_owned(),
            workspace_id: security_scope["workspace"]
                .as_str()
                .filter(|value| !value.is_empty())
                .map(str::to_owned),
            graph_space: Some("conversation".to_owned()),
        };
        let publication = kogwistar_contracts::DurableProviderEventPublicationV1 {
            namespace: scope.namespace.clone(),
            conversation_id: request.conversation_id.clone(),
            event: durable.clone(),
        };
        let mut mutations = provider_event_graph_mutations(
            &publication,
            &scope,
            security_scope["security_scope"]
                .as_str()
                .unwrap_or_default(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|_| "system clock is before Unix epoch".to_owned())?
                .as_secs() as i64,
        );
        validate_provider_event_publications(
            std::slice::from_ref(&publication),
            &mutations,
            &request.run_id,
        )?;
        if mutations
            .iter()
            .any(|mutation| !graph_mutation_scope_allowed(&mutation.scope, &security_scope))
        {
            return Err(
                "provider event publication is outside the run owner's admitted scope".to_owned(),
            );
        }
        let owner_id = authenticated_run_owner_id(&principal)
            .ok_or_else(|| "provider event requires authenticated owner".to_owned())?
            .to_owned();
        let message_id = format!("msg:{}", durable.event_id);
        let mut metadata = MetadataFilter::default();
        metadata
            .equals
            .insert("entity_type".to_owned(), json!("acl_record"));
        metadata
            .equals
            .insert("acl_truth_graph".to_owned(), json!("conversation"));
        metadata
            .equals
            .insert("acl_target_entity_id".to_owned(), json!(message_id));
        let existing_acl = self
            .store
            .graph_projection_records(GraphProjectionRead {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                ids: None,
                metadata,
                limit: 10000,
            })
            .await
            .map_err(|_| "provider-event ACL projection lookup failed".to_owned())?;
        let commit_principal = self
            .resolve_inline_native_owner_principal(&request.run_id, "write_graph")
            .await
            .map_err(|_| "provider-event owner authorization changed before commit".to_owned())?;
        let commit_scope = security_scope_value(&commit_principal);
        if authenticated_run_owner_id(&commit_principal) != Some(owner_id.as_str())
            || commit_scope != security_scope
            || !capability_allowed_for_principal(
                &commit_principal,
                &self.capabilities,
                "write_graph.provider_event.commit",
                &["write_graph"],
            )
            || !principal_strings(&commit_principal, "capabilities")
                .iter()
                .any(|capability| capability == "write_graph")
        {
            return Err(
                "provider-event owner capability or scope changed before commit".to_owned(),
            );
        }
        let (acl_mutations, acl_jobs) = derive_default_acl_graph_effects_from_records(
            &scope,
            &mutations,
            acl_targets_for_graph_mutations(&mutations).0,
            existing_acl,
            AclDefaultsContext {
                truth_graph: "conversation",
                owner_id: &graph_acl_principal_id(&commit_principal),
                security_scope: security_scope["security_scope"].as_str(),
                effect_id: &durable.event_id,
            },
        )?;
        mutations.extend(acl_mutations);
        let event_payload = serde_json::to_string(&durable)
            .map_err(|_| "provider event could not be serialized".to_owned())?;
        let mut trace_payload = json!({
            "run_id": request.run_id, "trace_id": request.run_id,
            "workflow_id": request.workflow_id, "conversation_id": request.conversation_id,
            "turn_node_id": request.turn_node_id, "step_id": request.node_id,
            "step_seq": request.step_seq, "token_id": request.token_id,
            "parent_token_id": request.parent_token_id, "executor_target": "rust-native",
            "provider_id": provider_id, "provider_model": model,
            "client_sse_sequence": client_sse_sequence, "provider_event": summary,
            "event_id": durable.event_id,
        });
        if let Some(fields) = trace_payload.as_object_mut() {
            fields.extend(trace_context);
        }
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|_| "system clock is before Unix epoch".to_owned())?
            .as_secs() as i64;
        let lane = NewProjectedLaneMessage {
            message_id,
            namespace: scope.namespace.clone(),
            purpose: "agent_event".to_owned(),
            inbox_id: "agent-provider-events".to_owned(),
            conversation_id: request.conversation_id.clone(),
            recipient_id: "agent-event-consumer".to_owned(),
            sender_id: "agent-runtime".to_owned(),
            msg_type: "agent.provider_event".to_owned(),
            status: "pending".to_owned(),
            created_at: now,
            available_at: now,
            run_id: Some(request.run_id.clone()),
            step_id: Some(durable.trace.step_id.clone()),
            correlation_id: Some(durable.event_id.clone()),
            payload_json: Some(event_payload.clone()),
            error_json: None,
        };
        self.store
            .persist_provider_event_publication(
                kogwistar_store_memory::ProviderEventPublicationCommit {
                    run_id: request.run_id.clone(),
                    owner_id,
                    conversation_id: request.conversation_id.clone(),
                    namespace: scope.namespace,
                    provider_event_json: event_payload,
                    graph_mutations: mutations,
                    index_jobs: acl_jobs,
                    lane,
                    run_event_type: "workflow.provider_event.v1".to_owned(),
                    run_event_payload_json: serde_json::to_string(&trace_payload)
                        .map_err(|_| "provider event trace could not be serialized".to_owned())?,
                },
            )
            .map_err(|_| "provider event in-memory commit failed".to_owned())?;
        Ok(summary)
    }

    fn supports_graph_tool(name: &str) -> bool {
        matches!(
            name,
            "kg_find_edges"
                | "kg_k_hop"
                | "kg_load_persisted"
                | "kg_neighbors"
                | "kg_propose_bruteforce"
                | "kg_shortest_path"
                | "kg_semantic_seed_then_expand_text"
                | "wisdom.semantic_seed_then_expand"
                | "kg_viz_cytoscape_json"
                | "kg_viz_d3_json"
                | "propose_vector"
        )
    }

    fn with_embedding_provider(mut self, provider: Arc<dyn EmbeddingProvider>) -> Self {
        self.embedding_provider = Some(provider);
        self
    }

    fn with_chat_completion_provider(
        mut self,
        provider: crate::native_answer_generation::OpenAiCompatibleChatConfig,
    ) -> Self {
        self.chat_completion_provider = Some(provider);
        self
    }

    async fn with_owner_auth_config(mut self, config: &AuthConfig) -> Result<Self, String> {
        self.owner_auth_store = Some(AuthStore::from_config(config).await?);
        Ok(self)
    }

    fn with_search_index_path(mut self, path: impl AsRef<std::path::Path>) -> Result<Self, String> {
        let connection = crate::search_index_fts::open_index(path)
            .map_err(|error| format!("could not open search-index FTS database: {error}"))?;
        self.search_index_fts = Some(Arc::new(Mutex::new(connection)));
        Ok(self)
    }

    fn authorized_graph_scope(
        &self,
        request: &ApiEffectRequest,
    ) -> Result<GraphScope, ApiEffectResponse> {
        self.authorized_graph_scope_for_namespace(request, "docs")
    }

    fn authorized_graph_scope_for_namespace(
        &self,
        request: &ApiEffectRequest,
        required_namespace: &str,
    ) -> Result<GraphScope, ApiEffectResponse> {
        if !capability_allowed(request, &self.capabilities, "read_graph", &["read_graph"]) {
            return Err(effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Graph read requires read_graph capability",
            ));
        }
        let namespaces = principal_strings(&request.principal, "ns");
        if !namespaces
            .iter()
            .any(|namespace| namespace == "*" || namespace == required_namespace)
        {
            return Err(effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_NAMESPACE_FORBIDDEN",
                format!("Graph reads require {required_namespace} namespace"),
            ));
        }
        let scope_value = security_scope_value(&request.principal);
        let scope = GraphScope {
            namespace: scope_value["storage_namespace"]
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
        if required_namespace == "wisdom" && scope.namespace != required_namespace {
            return Err(effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_SCOPE_FORBIDDEN",
                format!("Graph reads require storage namespace {required_namespace}"),
            ));
        }
        Ok(scope)
    }

    async fn graph_snapshot(
        &self,
        scope: GraphScope,
    ) -> Result<(Vec<GraphRecord>, Vec<GraphRecord>), String> {
        let (nodes, edges) = self
            .store
            .graph_node_edge_snapshot(&scope, 10_001)
            .map_err(|error| error.to_string())?;
        if nodes.len() > 10_000 || edges.len() > 10_000 {
            return Err("in-memory graph snapshot exceeds the 10000-record bound".to_owned());
        }
        Ok((nodes, edges))
    }

    async fn graph_read_effect(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        if request.path_and_query == INTERNAL_GRAPH_LOAD_PERSISTED_ROUTE {
            return self.graph_load_persisted_effect(request).await;
        }
        if request.path_and_query == INTERNAL_GRAPH_PROPOSE_BRUTE_FORCE_ROUTE {
            return self.graph_propose_bruteforce_effect(request).await;
        }
        if request.path_and_query == INTERNAL_GRAPH_VECTOR_QUERY_ROUTE {
            return self.graph_vector_query_effect(request).await;
        }
        if request.path_and_query == GRAPH_SEMANTIC_SEED_EXPAND_ROUTE {
            return self.graph_semantic_seed_expand_effect(request).await;
        }
        if request.path_and_query == WISDOM_SEMANTIC_SEED_EXPAND_ROUTE {
            return self.wisdom_semantic_seed_expand_effect(request).await;
        }
        if request.path_and_query == INTERNAL_GRAPH_PROPOSE_VECTOR_ROUTE {
            return self.graph_vector_proposal_effect(request).await;
        }
        let scope = match self.authorized_graph_scope(request) {
            Ok(scope) => scope,
            Err(response) => return response,
        };
        let truth_graph = if request.path_and_query == INTERNAL_GRAPH_NEIGHBORS_ROUTE {
            match parse_graph_neighbors_request(&request.body) {
                Ok(input) => input.truth_graph,
                Err(error) => return invalid_graph_request(error),
            }
        } else if matches!(
            request.path_and_query.as_str(),
            INTERNAL_GRAPH_VIZ_D3_ROUTE | INTERNAL_GRAPH_VIZ_CYTOSCAPE_ROUTE
        ) {
            let input: GraphVisualizationRequest = match serde_json::from_slice(&request.body) {
                Ok(input) => input,
                Err(error) => return invalid_graph_request(error.to_string()),
            };
            input.truth_graph().to_owned()
        } else {
            "knowledge".to_owned()
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
        let redirect_rid = if request.path_and_query == INTERNAL_GRAPH_NEIGHBORS_ROUTE {
            match parse_graph_neighbors_request(&request.body) {
                Ok(input) => match input.resolve_mode {
                    GraphNeighborsResolveModeV1::ActiveOnly => Some(input.rid.trim().to_owned()),
                    GraphNeighborsResolveModeV1::Redirect => {
                        if nodes.iter().any(|node| node.id == input.rid.trim()) {
                            resolve_graph_node_redirect_id(input.rid.trim(), &nodes)
                        } else {
                            resolve_graph_edge_redirect_id(input.rid.trim(), &edges)
                        }
                    }
                },
                Err(_) => None,
            }
        } else {
            None
        };
        let scope_value = security_scope_value(&request.principal);
        let principal = graph_acl_principal_from_claims(
            &request.principal,
            scope_value["security_scope"].as_str(),
        );
        let (nodes, edges) =
            graph_records_visible_by_acl(&nodes, &edges, truth_graph.trim(), &principal);
        let result = match request.path_and_query.as_str() {
            INTERNAL_GRAPH_NEIGHBORS_ROUTE => {
                let input = match parse_graph_neighbors_request(&request.body) {
                    Ok(input) => input,
                    Err(error) => return invalid_graph_request(error),
                };
                let neighbors = graph_neighbors_from_records(
                    redirect_rid.as_deref().unwrap_or_default(),
                    &nodes,
                    &edges,
                    input.doc_id.as_deref(),
                );
                json!({"nodes": neighbors.nodes, "edges": neighbors.edges})
            }
            INTERNAL_GRAPH_FIND_EDGES_ROUTE => {
                let input = match parse_graph_find_edges_request(&request.body) {
                    Ok(input) => input,
                    Err(error) => return invalid_graph_request(error),
                };
                let query = GraphFindEdgesQuery {
                    relation: input.relation,
                    src_label_contains: input.src_label_contains,
                    tgt_label_contains: input.tgt_label_contains,
                    doc_id: input.doc_id,
                };
                json!({"edges": graph_find_edges_from_records(&query, &nodes, &edges)})
            }
            INTERNAL_GRAPH_SHORTEST_PATH_ROUTE => {
                let input: GraphShortestPathRequest = match serde_json::from_slice(&request.body) {
                    Ok(input) => input,
                    Err(error) => return invalid_graph_request(error.to_string()),
                };
                if input.src_id.trim().is_empty()
                    || input.dst_id.trim().is_empty()
                    || input.max_depth > 64
                {
                    return invalid_graph_request(
                        "src_id and dst_id are required and max_depth must be <= 64".to_owned(),
                    );
                }
                let query = GraphShortestPathQuery {
                    src_id: input.src_id,
                    dst_id: input.dst_id,
                    doc_id: input.doc_id,
                    max_depth: input.max_depth,
                };
                json!({"path": graph_shortest_path_from_records(&query, &nodes, &edges)})
            }
            INTERNAL_GRAPH_K_HOP_ROUTE => {
                let input: GraphKHopRequest = match serde_json::from_slice(&request.body) {
                    Ok(input) => input,
                    Err(error) => return invalid_graph_request(error.to_string()),
                };
                if input.start_ids.is_empty()
                    || input.start_ids.len() > 100
                    || input.start_ids.iter().any(|id| id.trim().is_empty())
                    || input.k > 8
                {
                    return invalid_graph_request(
                        "start_ids must contain 1 to 100 IDs and k must be <= 8".to_owned(),
                    );
                }
                let query = GraphKHopQuery {
                    start_ids: input.start_ids,
                    k: input.k,
                    doc_id: input.doc_id,
                };
                let layers = graph_k_hop_from_records(&query, &nodes, &edges)
                    .into_iter()
                    .map(|layer| json!({"nodes": layer.nodes, "edges": layer.edges}))
                    .collect::<Vec<_>>();
                json!({"layers": layers})
            }
            INTERNAL_GRAPH_VIZ_D3_ROUTE | INTERNAL_GRAPH_VIZ_CYTOSCAPE_ROUTE => {
                let input: GraphVisualizationRequest = match serde_json::from_slice(&request.body) {
                    Ok(input) => input,
                    Err(error) => return invalid_graph_request(error.to_string()),
                };
                if request.path_and_query == INTERNAL_GRAPH_VIZ_CYTOSCAPE_ROUTE {
                    graph_viz_cytoscape_from_records(
                        &nodes,
                        &edges,
                        input.doc_id.as_deref(),
                        &input.mode,
                        input.insertion_method.as_deref(),
                    )
                } else {
                    graph_viz_d3_from_records(
                        &nodes,
                        &edges,
                        input.doc_id.as_deref(),
                        &input.mode,
                        input.insertion_method.as_deref(),
                    )
                }
            }
            _ => return unavailable_effect(request.clone()),
        };
        json_effect(StatusCode::OK, rust_native_attested(result))
    }

    // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/in_memory_service.rs#graph_semantic_seed_expand_effect; python=kogwistar/graph_query.py#GraphQuery.semantic_seed_then_expand_text; fixture=contracts/golden/semantic-seed-expand-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/in_memory_semantic_retrieval.rs#in_memory_semantic_seed_expand_is_available_through_authenticated_http_mcp; python_test=tests/primitives/test_graph_query.py#test_semantic_seed_then_expand_text
    async fn graph_semantic_seed_expand_effect(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        self.semantic_seed_expand_effect(request, "knowledge", "docs", 5, 1)
            .await
    }

    async fn wisdom_semantic_seed_expand_effect(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        self.semantic_seed_expand_effect(request, "wisdom", "wisdom", 10, 2)
            .await
    }

    async fn semantic_seed_expand_effect(
        &self,
        request: &ApiEffectRequest,
        truth_graph: &str,
        required_namespace: &str,
        default_top_k: usize,
        default_hops: usize,
    ) -> ApiEffectResponse {
        let scope = match self.authorized_graph_scope_for_namespace(request, required_namespace) {
            Ok(scope) => scope,
            Err(response) => return response,
        };
        let input: SemanticSeedExpandRequest = match serde_json::from_slice(&request.body) {
            Ok(input) => input,
            Err(error) => return invalid_graph_request(error.to_string()),
        };
        let query_text = input.text.trim();
        let top_k = input.top_k.unwrap_or(default_top_k);
        let hops = input.hops.unwrap_or(default_hops);
        if query_text.is_empty()
            || query_text.chars().count() > 16_384
            || !(1..=1000).contains(&top_k)
            || hops > 32
        {
            return invalid_graph_request(
                "text must contain 1 to 16384 characters, top_k must be 1 to 1000, and hops must be at most 32".to_owned(),
            );
        }
        let Some(provider) = self.embedding_provider.as_ref() else {
            return effect_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "KOGWISTAR_EMBEDDING_UNAVAILABLE",
                "Semantic graph retrieval requires a configured embedding provider",
            );
        };
        let (embedding, _embedding_dim, metric) =
            match semantic_query_embedding(provider.as_ref(), query_text).await {
                Ok(result) => result,
                Err(response) => return response,
            };
        let mut metadata = BTreeMap::new();
        if let Some(workspace_id) = scope.workspace_id.as_ref() {
            metadata.insert("workspace_id".to_owned(), json!(workspace_id));
        }
        if let Some(graph_space) = scope.graph_space.as_ref() {
            metadata.insert("graph_space".to_owned(), json!(graph_space));
        }
        if let Some(doc_id) = input.doc_ids.as_ref() {
            metadata.insert("doc_id".to_owned(), json!(doc_id));
        }
        let query = VectorQuery {
            embedding,
            limit: 10_000,
            metadata: MetadataFilter { equals: metadata },
            metric,
        };
        let (all_nodes, all_edges, mut query_matches) = match self
            .store
            .graph_node_edge_vector_snapshot(&scope, 10_001, "gke_nodes", &[query])
        {
            Ok(snapshot) => snapshot,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_VECTOR_QUERY_ERROR",
                    error.to_string(),
                );
            }
        };
        if all_nodes.len() > 10_000 || all_edges.len() > 10_000 {
            return effect_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "KOGWISTAR_STORE_ERROR",
                "in-memory graph snapshot exceeds the 10000-record bound",
            );
        }
        let scope_value = security_scope_value(&request.principal);
        let principal = graph_acl_principal_from_claims(
            &request.principal,
            scope_value["security_scope"].as_str(),
        );
        let (visible_nodes, visible_edges) =
            graph_records_visible_by_acl(&all_nodes, &all_edges, truth_graph, &principal);
        let visible_node_ids = visible_nodes
            .iter()
            .map(|record| record.id.as_str())
            .collect::<BTreeSet<_>>();
        let matches = query_matches.pop().unwrap_or_default();
        let visible_seeds = matches
            .into_iter()
            .filter(|matched| {
                matched
                    .record
                    .metadata
                    .get("entity_type")
                    .and_then(Value::as_str)
                    != Some("acl_record")
                    && !matched
                        .record
                        .metadata
                        .get("lifecycle_status")
                        .and_then(Value::as_str)
                        .is_some_and(|status| matches!(status, "tombstoned" | "deleted"))
            })
            .take(top_k)
            .map(|matched| matched.record)
            .filter(|record| visible_node_ids.contains(record.id.as_str()))
            .collect::<Vec<_>>();
        let seed_ids = visible_seeds
            .iter()
            .map(|record| record.id.clone())
            .collect::<Vec<_>>();
        let seeds = visible_seeds
            .iter()
            .filter_map(graph_record_document_json)
            .collect::<Vec<_>>();
        let layers = graph_k_hop_from_records(
            &GraphKHopQuery {
                start_ids: seed_ids,
                k: hops,
                doc_id: None,
            },
            &visible_nodes,
            &visible_edges,
        )
        .into_iter()
        .map(|layer| {
            let nodes = layer
                .nodes
                .iter()
                .filter_map(|id| visible_nodes.iter().find(|record| &record.id == id))
                .filter_map(graph_record_document_json)
                .collect::<BTreeSet<_>>();
            let edges = layer
                .edges
                .iter()
                .filter_map(|id| visible_edges.iter().find(|record| &record.id == id))
                .filter_map(graph_record_document_json)
                .collect::<BTreeSet<_>>();
            json!({
                "nodes": nodes.into_iter().collect::<Vec<_>>(),
                "edges": edges.into_iter().collect::<Vec<_>>(),
            })
        })
        .collect::<Vec<_>>();
        json_effect(
            StatusCode::OK,
            rust_native_attested(json!({"seeds": seeds, "layers": layers})),
        )
    }

    // PYTHON_PARITY: kogwistar/server/mcp_tools.py::_load_persisted_graph
    async fn graph_load_persisted_effect(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        let scope = match self.authorized_graph_scope(request) {
            Ok(scope) => scope,
            Err(response) => return response,
        };
        let input = match serde_json::from_slice::<LoadPersistedGraphRequest>(&request.body) {
            Ok(input) => input,
            Err(error) => {
                return invalid_graph_request(format!(
                    "Invalid persisted graph lookup request: {error}"
                ));
            }
        };
        if input.doc_ids.is_empty()
            || input.doc_ids.len() > 1000
            || input.doc_ids.iter().any(|doc| doc.trim().is_empty())
            || input
                .insertion_method
                .as_ref()
                .is_some_and(|method| method.trim().is_empty())
        {
            return invalid_graph_request(
                "doc_ids must contain 1 to 1000 non-empty IDs and insertion_method must be non-empty when provided"
                    .to_owned(),
            );
        }
        let (all_nodes, all_edges) = match self.graph_snapshot(scope).await {
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
        let (nodes, edges) =
            graph_records_visible_by_acl(&all_nodes, &all_edges, "knowledge", &principal);
        let (node_ids, edge_ids) = graph_load_persisted_ids_from_records(&nodes, &edges, &input);
        json_effect(
            StatusCode::OK,
            rust_native_attested(json!({"node_ids": node_ids, "edge_ids": edge_ids})),
        )
    }

    // PYTHON_PARITY: kogwistar/server/mcp_tools.py::kg_propose_bruteforce
    async fn graph_propose_bruteforce_effect(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        let scope = match self.authorized_graph_scope(request) {
            Ok(scope) => scope,
            Err(response) => return response,
        };
        let input = match serde_json::from_slice::<ProposeBruteForceRequest>(&request.body) {
            Ok(input) => input,
            Err(error) => {
                return invalid_graph_request(format!(
                    "Invalid brute-force proposal request: {error}"
                ));
            }
        };
        if brute_force_pair_kinds(&input.pair_kind).is_none()
            || input.allowed_docs.as_ref().is_some_and(|docs| {
                docs.len() > 1000 || docs.iter().any(|doc| doc.trim().is_empty())
            })
            || input
                .anchor_doc_id
                .as_ref()
                .is_some_and(|doc| doc.trim().is_empty())
            || input.limit_per_bucket.is_some_and(|limit| limit > 5000)
            || input
                .where_filter
                .as_ref()
                .is_some_and(|value| !value.is_object())
        {
            return invalid_graph_request(
                "pair_kind, document IDs, where, or limit_per_bucket are invalid or exceed their bounds"
                    .to_owned(),
            );
        }
        if let Some(where_filter) = input.where_filter.as_ref()
            && let Err(error) = kogwistar_contracts::metadata_filter_matches_from_str(
                &json!({"metadata": {}, "where": where_filter}).to_string(),
            )
        {
            return invalid_graph_request(error.to_string());
        }
        let (all_nodes, all_edges) = match self.graph_snapshot(scope).await {
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
        let (nodes, edges) =
            graph_records_visible_by_acl(&all_nodes, &all_edges, "knowledge", &principal);
        let (node_ids_by_doc, edge_ids_by_doc) =
            brute_force_document_indexes_from_records(&nodes, &edges);
        let mut docs = match input.allowed_docs.as_ref().filter(|docs| !docs.is_empty()) {
            Some(docs) => docs
                .iter()
                .cloned()
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>(),
            None => node_ids_by_doc
                .keys()
                .chain(edge_ids_by_doc.keys())
                .cloned()
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect(),
        };
        if let Some(anchor) = input.anchor_doc_id.as_ref()
            && !docs.contains(anchor)
        {
            docs.insert(0, anchor.clone());
        }
        if docs.len() > 1000 {
            return invalid_graph_request(
                "brute-force proposal is limited to 1000 documents".to_owned(),
            );
        }
        let nodes = nodes
            .into_iter()
            .map(|record| (record.id.clone(), record))
            .collect::<BTreeMap<_, _>>();
        let edges = edges
            .into_iter()
            .map(|record| (record.id.clone(), record))
            .collect::<BTreeMap<_, _>>();
        let pairs = match brute_force_proposal_pairs(
            &docs,
            &node_ids_by_doc,
            &edge_ids_by_doc,
            &nodes,
            &edges,
            &input,
        ) {
            Ok(pairs) => pairs,
            Err(error) => return invalid_graph_request(error),
        };
        json_effect(
            StatusCode::OK,
            rust_native_attested(json!({"pairs": pairs})),
        )
    }

    // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/in_memory_service.rs#graph_vector_proposal_effect; python=kogwistar/server/mcp_tools.py#propose_vector; fixture=contracts/golden/graph-vector-proposal-acl-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/in_memory_graph_routes.rs#in_memory_backend_routes_authenticated_graph_write_and_mcp_read; python_test=tests/mcp/test_chat_mcp_tools.py#test_propose_vector_returns_only_acl_visible_pairs
    async fn graph_vector_proposal_effect(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        let scope = match self.authorized_graph_scope(request) {
            Ok(scope) => scope,
            Err(response) => return response,
        };
        let input = match GraphVectorProposalRequest::parse(&request.body) {
            Ok(input) => input,
            Err(error) => return invalid_graph_request(error),
        };
        let where_filter = input
            .where_filter
            .as_ref()
            .filter(|value| value.is_object());
        let validation = json!({"metadata": {}, "where": where_filter});
        if let Err(error) =
            kogwistar_contracts::metadata_filter_matches_from_str(&validation.to_string())
        {
            return invalid_graph_request(error.to_string());
        }
        let (all_nodes, all_edges) = match self.graph_snapshot(scope.clone()).await {
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
        let (visible_nodes, visible_edges) =
            graph_records_visible_by_acl(&all_nodes, &all_edges, "knowledge", &principal);
        let pairs = match graph_vector_proposal_pairs(
            &all_nodes,
            &all_edges,
            &visible_nodes,
            &visible_edges,
            &input,
            where_filter,
        ) {
            Ok(pairs) => pairs,
            Err(error) => return invalid_graph_request(error),
        };
        json_effect(
            StatusCode::OK,
            rust_native_attested(json!({"pairs": pairs})),
        )
    }

    // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/in_memory_service.rs#graph_vector_query_effect; python=kogwistar/engine_core/subsystems/acl.py#ACLAwareReadSubsystem.query_nodes; fixture=contracts/golden/graph-vector-source-acl-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/vector_query_acl.rs#in_memory_vector_query_requires_exact_source_span_acl_reference; python_test=tests/mcp/test_chat_mcp_tools.py#test_vector_query_requires_exact_source_span_acl_reference
    async fn graph_vector_query_effect(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        let scope = match self.authorized_graph_scope(request) {
            Ok(scope) => scope,
            Err(response) => return response,
        };
        let input = match parse_graph_vector_query_request(&request.body) {
            Ok(input) => input,
            Err(error) => return invalid_graph_request(error),
        };
        let table = if input.entity_kind == "node" {
            "gke_nodes"
        } else {
            "gke_edges"
        };
        let scope_value = security_scope_value(&request.principal);
        let acl_principal = graph_acl_principal_from_claims(
            &request.principal,
            scope_value["security_scope"].as_str(),
        );
        let queries = input
            .query_embeddings
            .iter()
            .map(|embedding| VectorQuery {
                embedding: embedding.clone(),
                limit: 10_000,
                metadata: input.where_filter.clone(),
                metric: input.metric,
            })
            .collect::<Vec<_>>();
        let (nodes, edges, vector_matches) = match self
            .store
            .graph_node_edge_vector_snapshot(&scope, 10_001, table, &queries)
        {
            Ok(records) => records,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_VECTOR_QUERY_ERROR",
                    error.to_string(),
                );
            }
        };
        if nodes.len() > 10_000 || edges.len() > 10_000 {
            return effect_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "KOGWISTAR_STORE_ERROR",
                "in-memory graph snapshot exceeds the 10000-record bound",
            );
        }
        let (visible_nodes, visible_edges) =
            graph_records_visible_by_acl(&nodes, &edges, &input.truth_graph, &acl_principal);
        let visible_ids = if input.entity_kind == "node" {
            visible_nodes
                .iter()
                .map(|record| record.id.as_str())
                .collect::<BTreeSet<_>>()
        } else {
            visible_edges
                .iter()
                .map(|record| record.id.as_str())
                .collect::<BTreeSet<_>>()
        };
        let mut ids = Vec::with_capacity(input.query_embeddings.len());
        let mut documents = Vec::with_capacity(input.query_embeddings.len());
        let mut metadatas = Vec::with_capacity(input.query_embeddings.len());
        let mut embeddings = Vec::with_capacity(input.query_embeddings.len());
        let mut distances = Vec::with_capacity(input.query_embeddings.len());
        for matches in vector_matches {
            let matches = matches
                .into_iter()
                .filter(|matched| {
                    matched
                        .record
                        .metadata
                        .get("entity_type")
                        .and_then(Value::as_str)
                        != Some("acl_record")
                        && !matched
                            .record
                            .metadata
                            .get("lifecycle_status")
                            .and_then(Value::as_str)
                            .is_some_and(|status| matches!(status, "tombstoned" | "deleted"))
                })
                .take(input.n_results)
                .filter(|matched| visible_ids.contains(matched.record.id.as_str()))
                .collect::<Vec<_>>();
            ids.push(
                matches
                    .iter()
                    .map(|matched| json!(matched.record.id))
                    .collect::<Vec<_>>(),
            );
            if input.include.iter().any(|field| field == "documents") {
                documents.push(
                    matches
                        .iter()
                        .map(|matched| json!(matched.record.document))
                        .collect::<Vec<_>>(),
                );
            }
            if input.include.iter().any(|field| field == "metadatas") {
                metadatas.push(
                    matches
                        .iter()
                        .map(|matched| json!(matched.record.metadata))
                        .collect::<Vec<_>>(),
                );
            }
            if input.include.iter().any(|field| field == "embeddings") {
                embeddings.push(
                    matches
                        .iter()
                        .map(|matched| json!(matched.record.embedding))
                        .collect::<Vec<_>>(),
                );
            }
            if input.include.iter().any(|field| field == "distances") {
                distances.push(
                    matches
                        .iter()
                        .map(|matched| json!(matched.distance))
                        .collect::<Vec<_>>(),
                );
            }
        }
        let mut response = serde_json::Map::new();
        response.insert("ids".to_owned(), json!(ids));
        if input.include.iter().any(|field| field == "documents") {
            response.insert("documents".to_owned(), json!(documents));
        }
        if input.include.iter().any(|field| field == "metadatas") {
            response.insert("metadatas".to_owned(), json!(metadatas));
        }
        if input.include.iter().any(|field| field == "embeddings") {
            response.insert("embeddings".to_owned(), json!(embeddings));
        }
        if input.include.iter().any(|field| field == "distances") {
            response.insert("distances".to_owned(), json!(distances));
        }
        json_effect(
            StatusCode::OK,
            rust_native_attested(Value::Object(response)),
        )
    }

    async fn graph_mutation_effect(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        if !capability_allowed(request, &self.capabilities, "write_graph", &["write_graph"]) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Graph mutation requires write_graph capability",
            );
        }
        if !principal_strings(&request.principal, "ns")
            .iter()
            .any(|namespace| matches!(namespace.as_str(), "*" | "docs"))
        {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_NAMESPACE_FORBIDDEN",
                "Graph mutations require docs namespace",
            );
        }
        let input: GraphMutationBatchInput = match serde_json::from_slice(&request.body) {
            Ok(input) => input,
            Err(error) => return invalid_graph_request(error.to_string()),
        };
        if let Err(error) = input.validate() {
            return invalid_graph_request(error.to_string());
        }
        let mutations = match graph_mutation_batch_value(input.clone()) {
            Ok(mutations) => mutations,
            Err(error) => return invalid_graph_request(error),
        };
        if let Err(error) = validate_graph_mutation_envelopes(&input, &mutations) {
            return invalid_graph_request(error);
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
                Ok(jobs) => jobs,
                Err(error) => return invalid_graph_request(error),
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
}

fn invalid_graph_request(message: String) -> ApiEffectResponse {
    effect_error(
        StatusCode::UNPROCESSABLE_ENTITY,
        "KOGWISTAR_INVALID_REQUEST",
        message,
    )
}

fn selected_proposal_seeds<'a>(
    records: &'a [GraphRecord],
    selected_ids: Option<&[String]>,
) -> Vec<&'a GraphRecord> {
    let by_id = records
        .iter()
        .map(|record| (record.id.as_str(), record))
        .collect::<BTreeMap<_, _>>();
    match selected_ids {
        Some(ids) => ids
            .iter()
            .filter_map(|id| by_id.get(id.as_str()).copied())
            .collect(),
        None => records.iter().collect(),
    }
}

fn proposal_doc_id(record: &GraphRecord) -> Option<&str> {
    record
        .metadata
        .get("doc_id")
        .and_then(Value::as_str)
        .or_else(|| {
            record
                .metadata
                .get("doc_ids")
                .and_then(Value::as_array)
                .and_then(|ids| ids.first())
                .and_then(Value::as_str)
        })
}

fn proposal_where_matches(
    record: &GraphRecord,
    where_filter: Option<&Value>,
) -> Result<bool, String> {
    let payload = json!({
        "metadata": record.metadata,
        "where": where_filter,
    });
    kogwistar_contracts::metadata_filter_matches_from_str(&payload.to_string())
        .map_err(|error| error.to_string())
}

fn proposal_cosine_distance(query: &[f32], candidate: Option<&[f32]>) -> Option<f64> {
    let candidate = candidate?;
    if query.len() != candidate.len()
        || query
            .iter()
            .chain(candidate)
            .any(|value| !value.is_finite())
    {
        return None;
    }
    let (dot, query_norm, candidate_norm) = query.iter().zip(candidate).fold(
        (0.0, 0.0, 0.0),
        |(dot, query_norm, candidate_norm), (left, right)| {
            let left = f64::from(*left);
            let right = f64::from(*right);
            (
                dot + left * right,
                query_norm + left * left,
                candidate_norm + right * right,
            )
        },
    );
    if query_norm == 0.0 || candidate_norm == 0.0 {
        Some(2.0)
    } else {
        Some(1.0 - dot / (query_norm.sqrt() * candidate_norm.sqrt()))
    }
}

// Shared by in-memory and persistent graph services so ranking, filters, and
// ACL-safe result projection stay one Python-parity implementation.
fn graph_vector_proposal_pairs(
    all_nodes: &[GraphRecord],
    all_edges: &[GraphRecord],
    visible_nodes: &[GraphRecord],
    visible_edges: &[GraphRecord],
    input: &GraphVectorProposalRequest,
    where_filter: Option<&Value>,
) -> Result<Vec<Value>, String> {
    let visible_node_ids = visible_nodes
        .iter()
        .map(|record| record.id.as_str())
        .collect::<BTreeSet<_>>();
    let visible_edge_ids = visible_edges
        .iter()
        .map(|record| record.id.as_str())
        .collect::<BTreeSet<_>>();
    let node_seeds = selected_proposal_seeds(all_nodes, input.new_node_ids.as_deref());
    let edge_seeds = selected_proposal_seeds(all_edges, input.new_edge_ids.as_deref());
    let mut pairs = Vec::new();
    let mut seen = BTreeSet::new();
    for (seed, seed_kind) in node_seeds
        .into_iter()
        .map(|record| (record, "node"))
        .chain(edge_seeds.into_iter().map(|record| (record, "edge")))
    {
        if seed.metadata.get("entity_type").and_then(Value::as_str) == Some("acl_record") {
            continue;
        }
        if !(if seed_kind == "node" {
            visible_node_ids.contains(seed.id.as_str())
        } else {
            visible_edge_ids.contains(seed.id.as_str())
        }) {
            continue;
        }
        let Some(query) = seed.embedding.as_deref() else {
            continue;
        };
        let seed_doc = proposal_doc_id(seed);
        for (candidates, candidate_kind) in [(all_nodes, "node"), (all_edges, "edge")] {
            if candidate_kind == "edge" && !input.include_edges {
                continue;
            }
            let mut ranked = Vec::new();
            for candidate in candidates {
                if candidate
                    .metadata
                    .get("entity_type")
                    .and_then(Value::as_str)
                    == Some("acl_record")
                {
                    continue;
                }
                if candidate_kind == "node" && candidate.id == seed.id {
                    continue;
                }
                if candidate
                    .metadata
                    .get("lifecycle_status")
                    .and_then(Value::as_str)
                    .is_some_and(|status| matches!(status, "tombstoned" | "deleted"))
                {
                    continue;
                }
                if !proposal_where_matches(candidate, where_filter)? {
                    continue;
                }
                let candidate_doc = proposal_doc_id(candidate);
                if input.allowed_docs.as_ref().is_some_and(|docs| {
                    !docs.is_empty()
                        && candidate_doc.is_none_or(|doc| !docs.iter().any(|item| item == doc))
                }) {
                    continue;
                }
                if input.cross_doc_only && seed_doc.is_some() && seed_doc == candidate_doc {
                    continue;
                }
                if input.anchor_doc_id.as_ref().is_some_and(|anchor| {
                    input.anchor_only
                        && seed_doc != Some(anchor.as_str())
                        && candidate_doc != Some(anchor.as_str())
                }) {
                    continue;
                }
                let Some(distance) =
                    proposal_cosine_distance(query, candidate.embedding.as_deref())
                else {
                    continue;
                };
                let in_threshold = if input.score_mode == "distance" {
                    distance <= input.max_distance
                } else {
                    distance >= input.min_similarity
                };
                if in_threshold {
                    ranked.push((candidate, distance));
                }
            }
            ranked.sort_by(|left, right| left.1.total_cmp(&right.1));
            let visible_ids = if candidate_kind == "node" {
                &visible_node_ids
            } else {
                &visible_edge_ids
            };
            for (candidate, _) in ranked.into_iter().take(input.top_k) {
                if !visible_ids.contains(candidate.id.as_str()) {
                    continue;
                }
                let (left, right, left_kind, right_kind) = if seed.id <= candidate.id {
                    (&seed.id, &candidate.id, seed_kind, candidate_kind)
                } else {
                    (&candidate.id, &seed.id, candidate_kind, seed_kind)
                };
                if seen.insert((left.clone(), right.clone())) {
                    pairs.push(json!({
                        "left_id": left,
                        "left_kind": left_kind,
                        "right_id": right,
                        "right_kind": right_kind,
                    }));
                }
            }
        }
    }
    Ok(pairs)
}

impl ApplicationService for InMemoryGraphApplicationService {
    fn execute_inline_native_work_once(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<bool, String>> + Send + '_>> {
        Box::pin(self.execute_native_step_once())
    }

    fn graph_authority_capability(&self) -> GraphAuthorityCapability {
        self.store.graph_authority_capability()
    }

    fn supports_mcp_tool(&self, surface: &str, name: &str) -> bool {
        if surface == "workflow"
            && matches!(
                name,
                "workflow.run_events" | "workflow.run_resume" | "workflow.run_cancel"
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
        // PYTHON_PARITY_NOTE_V1: run-event reads require ro/rw, workflow scope, and workflow.run.read; rust=rust/crates/kogwistar-api/src/sections/in_memory_service.rs#supports_mcp_tool_for_principal; python=kogwistar/server/chat_mcp.py#build_workflow_mcp.workflow_run_events; rust_test=rust/crates/kogwistar-api/src/tests/workflow_run_authorization.rs#workflow_run_events_mcp_admission_matches_python_policy_for_all_stores
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
        if method == "POST"
            && runtime_run_route(path)
                .is_some_and(|(_, tail)| tail.as_slice() == ["resume"])
        {
            return true;
        }
        if method == "POST"
            && runtime_run_route(path)
                .is_some_and(|(_, tail)| tail.as_slice() == ["cancel"])
        {
            return true;
        }
        if method == "GET"
            && runtime_run_route(path).is_some_and(|(_, tail)| {
                tail.is_empty()
                    || tail.as_slice() == ["events"]
                    || tail.as_slice() == ["events", "poll"]
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
                && request.path_and_query.split('?').next()
                    == Some("/api/syscall/v1/send_message")
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
                && request.path_and_query.split('?').next()
                    == Some("/api/syscall/v1/mount_memory")
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
                let response = self.in_memory_workflow_syscall_effect(&request, &input).await;
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

impl InMemoryGraphApplicationService {
    // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/in_memory_service.rs#in_memory_mount_memory_effect; python=kogwistar/server/syscall_api.py#mount_memory; fixture=contracts/golden/mount-memory-result-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/in_memory_conversations.rs#in_memory_mount_memory_matches_shared_contract_and_authority; python_test=tests/server/test_mount_memory_wire_contract.py#test_python_mount_memory_result_matches_shared_wire_fixture
    async fn in_memory_mount_memory_effect(
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

    // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/in_memory_service.rs#in_memory_create_conversation_effect; python=kogwistar/server/syscall_api.py#create_syscall_router.dispatch; fixture=contracts/golden/conversation-create-syscall-result-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/in_memory_conversations.rs#in_memory_create_conversation_is_atomic_and_matches_shared_contract; python_test=tests/server/test_syscall_api.py#test_syscall_create_conversation_uses_authenticated_owner
    async fn in_memory_create_conversation_effect(
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

    // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/in_memory_service.rs#in_memory_send_message_effect; python=kogwistar/server/chat_service_run_execution.py#submit_turn_for_answer; fixture=contracts/golden/answer-workflow-runtime-contract-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/in_memory_conversations.rs#in_memory_send_message_admits_turn_and_run_atomically; python_test=tests/server/test_capability_kernel_contract.py#test_submit_turn_for_answer_replays_idempotency_key_and_rejects_payload_change
    async fn in_memory_send_message_effect(
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
                return json_effect(
                    StatusCode::BAD_REQUEST,
                    json!({"detail":error}),
                );
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
            ("label".to_owned(), json!(format!("Turn {turn_index} (user)"))),
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
        let edge_document = serde_json::to_string(&edge_payload)
            .expect("conversation edge document serializes");
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
        let mut edge_metadata = edge_payload["metadata"]
            .as_object()
            .unwrap()
            .clone();
        edge_metadata.extend([
            ("doc_id".to_owned(), edge_payload["doc_id"].clone()),
            ("relation".to_owned(), json!("next_turn")),
            ("source_ids".to_owned(), json!(edge_payload["source_ids"].to_string())),
            ("target_ids".to_owned(), json!(edge_payload["target_ids"].to_string())),
            ("source_edge_ids".to_owned(), json!("[]")),
            ("target_edge_ids".to_owned(), json!("[]")),
            ("type".to_owned(), json!("relationship")),
            ("summary".to_owned(), json!("Sequential flow")),
            ("properties".to_owned(), json!(edge_payload["properties"].to_string())),
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
        for kind in conversation_index_job_kinds("edge")
            .expect("conversation edge index contract exists")
        {
            index_jobs.push(conversation_index_job(&scope.namespace, "edge", &edge_id, kind));
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

    // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/in_memory_service.rs#in_memory_receive_message_effect; python=kogwistar/server/chat_service.py#ChatRunService.list_transcript; fixture=contracts/golden/receive-message-result-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/in_memory_conversations.rs#in_memory_receive_message_matches_shared_python_contract; python_test=tests/server/test_receive_message_wire_contract.py#test_python_receive_message_result_matches_shared_wire_fixture
    async fn in_memory_receive_message_effect(
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

    // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/in_memory_service.rs#in_memory_invoke_builtin_tool_effect; python=kogwistar/server/syscall_api.py#_invoke_builtin_tool; fixture=contracts/golden/invoke-tool-results-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/in_memory_conversations.rs#in_memory_invoke_builtin_tools_match_shared_contract_and_authority; python_test=tests/server/test_syscall_api.py#test_invoke_builtin_tool_results_match_shared_contract_fixture
    async fn in_memory_invoke_builtin_tool_effect(
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

    // PYTHON_REFERENCE_V1: rust=rust/crates/kogwistar-api/src/sections/in_memory_service.rs#submit_run_effect; python=kogwistar/engine_core/in_memory_meta.py#create_server_run+append_server_run_event,kogwistar/messaging/service.py#project_lane_message; fixture=contracts/golden/runtime-native-noop-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/in_memory_run_admission.rs#in_memory_native_run_admission_is_authenticated_and_atomic+in_memory_native_admission_uses_authoritative_design_and_schedules_successors; python_test=tests/core/test_in_memory_meta_store.py#test_in_memory_meta_retry_and_run_cancel_contract; note=In-memory public admission validates an exact ready workflow-design snapshot when present; all steps are supported v1 Rust-native operations, and successor lanes commit atomically with worker effects without language fallback.
    async fn submit_run_effect(
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
            None => self
                .store
                .admit_server_run_with_start(run, admission_payload, transition, lane),
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

    // PYTHON_PARITY_NOTE_V1: Rust maps Python workflow syscalls to existing in-memory run, checkpoint, resource, and approval handlers; authorization and mutation semantics stay centralized. rust=rust/crates/kogwistar-api/src/sections/in_memory_service.rs#in_memory_workflow_syscall_effect; python=kogwistar/server/syscall_api.py#create_syscall_router; fixture=contracts/golden/workflow-run-resume-result-v1.json,contracts/golden/run-cancel-v1.json,contracts/golden/operator-dashboard-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/in_memory_run_admission.rs#in_memory_workflow_syscalls_use_native_run_handlers+in_memory_resume_syscall_preserves_resume_envelope_and_result+in_memory_project_view_syscall_is_reachable_through_authenticated_http; python_test=tests/server/test_syscall_api.py#test_syscall_spawn_process_returns_ok_contract+test_syscall_request_approval_is_blocked_contract+test_syscall_checkpoint_accepts_conversation_namespace_and_numeric_string_step_seq,tests/server/test_chat_server_api.py#test_resume_route_uses_python_defaults_and_checks_namespace+test_checkpoint_detail_uses_python_defaults_and_checks_namespace+test_runtime_resource_snapshot_reports_pressure
    async fn in_memory_workflow_syscall_effect(
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
        let write = matches!(input.op.as_str(), "spawn_process" | "terminate_process" | "resume");
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
                let (Some(run_id), Some(step_seq)) = (
                    input.args.get("run_id").and_then(Value::as_str),
                    step_seq,
                ) else {
                    return effect_error(
                        StatusCode::BAD_REQUEST,
                        "KOGWISTAR_INVALID_REQUEST",
                        "run_id and integer step_seq are required",
                    );
                };
                match kogwistar_store::ServerRunReadStore::server_run_events(
                    &self.store, run_id, 0, 100_000,
                ).await {
                    Ok(events) => syscall_result(
                        &input.version,
                        &input.op,
                        runtime_inspection_effect(
                            request, run_id, &["checkpoints", &step_seq.to_string()], events,
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
                    &self.store, None, None, None, 10_000,
                ).await {
                    Ok(runs) => runs,
                    Err(error) => return effect_error(StatusCode::INTERNAL_SERVER_ERROR, "KOGWISTAR_STORE_ERROR", error.to_string()),
                };
                let lanes = match kogwistar_store::LaneMessageReadStore::projected_lane_messages(
                    &self.store,
                    LaneMessageFilter { namespace: Some("workflow".to_owned()), limit: 10_000, ..LaneMessageFilter::default() },
                ).await {
                    Ok(lanes) => lanes,
                    Err(error) => return effect_error(StatusCode::INTERNAL_SERVER_ERROR, "KOGWISTAR_STORE_ERROR", error.to_string()),
                };
                let mut events = Vec::new();
                for run in &runs {
                    match kogwistar_store::ServerRunReadStore::server_run_events(
                        &self.store, &run.run_id, 0, 200,
                    ).await {
                        Ok(mut run_events) => events.append(&mut run_events),
                        Err(error) => return effect_error(StatusCode::INTERNAL_SERVER_ERROR, "KOGWISTAR_STORE_ERROR", error.to_string()),
                    }
                }
                let services = match kogwistar_store::ProjectionReadStore::named_projections(
                    &self.store, "service_registry",
                ).await {
                    Ok(rows) => rows.into_iter().map(|row| Value::Object(row.payload)).collect(),
                    Err(error) => return effect_error(StatusCode::INTERNAL_SERVER_ERROR, "KOGWISTAR_STORE_ERROR", error.to_string()),
                };
                syscall_result(
                    &input.version,
                    &input.op,
                    operational_effect(request, "/api/workflow/resources", runs, lanes, events, services, Vec::new()),
                )
            }
            "request_approval" => request_approval_syscall(request, &self.capabilities, input),
            _ => unreachable!("operation validated before dispatch"),
        }
    }

    // PYTHON_PARITY_NOTE_V1: Rust mirrors the workflow-scoped REST/MCP cancel contract; terminal runs return unchanged, active runs require workflow.run.write, and the volatile store atomically records cancellation plus lane cleanup. rust=rust/crates/kogwistar-api/src/sections/in_memory_service.rs#cancel_runtime_run_effect; python=kogwistar/server/chat_service.py#cancel_run,kogwistar/server/chat_mcp.py#build_workflow_mcp.workflow_run_cancel; fixture=contracts/golden/run-cancel-v1.json,contracts/golden/workflow-run-cancel-mcp-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/in_memory_run_admission.rs#in_memory_run_cancel_matches_shared_contract_and_authority; python_test=tests/mcp/test_chat_mcp_tools.py#test_mcp_workflow_run_cancel_matches_shared_contract,tests/core/test_in_memory_meta_store.py#test_in_memory_meta_retry_and_run_cancel_contract
    async fn cancel_runtime_run_effect(
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
        let run = match kogwistar_store::ServerRunReadStore::server_run(&self.store, run_id).await
        {
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
        match kogwistar_store::ServerRunWriteStore::request_server_run_cancel(&self.store, run_id)
            .await
        {
            Ok(()) => match kogwistar_store::ServerRunReadStore::server_run(&self.store, run_id)
                .await
            {
                Ok(Some(run)) => json_effect(StatusCode::ACCEPTED, server_run_value(run)),
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
    }

    // PYTHON_PARITY_NOTE_V1: Rust mirrors ChatRunService.resume_run and the synchronous public resume response; the volatile store commits resume transition plus continuation lane atomically, without claiming durability across process loss.
    async fn resume_runtime_run_effect(
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
            let run = match kogwistar_store::ServerRunReadStore::server_run(&self.store, run_id).await
            {
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
                    Ok(Some(projection)) => serde_json::from_value::<
                        kogwistar_runtime::RecordedRuntimeState,
                    >(Value::Object(projection.payload))
                    .ok(),
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

    async fn read_runtime_run_effect(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        let Some((run_id, tail)) = runtime_run_route(&request.path_and_query) else {
            return unavailable_effect(request.clone());
        };
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
        let Some(owner_id) = authenticated_run_owner_id(&request.principal) else {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_RUN_OWNER_FORBIDDEN",
                "Run inspection requires an authenticated owner",
            );
        };
        let run = match kogwistar_store::ServerRunReadStore::server_run(&self.store, &run_id).await
        {
            Ok(Some(run)) if run.user_id.as_deref() == Some(owner_id) => run,
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
        if tail.is_empty() {
            return json_effect(StatusCode::OK, server_run_value(run));
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

    // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/in_memory_service.rs#search_index_hybrid_effect; python=kogwistar/server_mcp_with_admin.py#search_index_hybrid; fixture=contracts/golden/search-index-hybrid-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/in_memory_semantic_retrieval.rs#in_memory_search_index_hybrid_is_scoped_and_acl_filtered; python_test=tests/core/test_search_index_contract.py#test_hybrid_search_fusion_matches_shared_rust_golden_fixture
    async fn search_index_hybrid_effect(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
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
}
