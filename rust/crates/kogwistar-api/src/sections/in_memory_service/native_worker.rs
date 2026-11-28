use super::*;
use serde_json::json;

impl InMemoryGraphApplicationService {
    /// Compose trusted Rust-native application operations for the in-memory
    /// backend, matching the persistent runtime service extension point.
    #[cfg(test)]
    pub(crate) fn with_native_operations(
        mut self,
        native_operations: kogwistar_runtime::NativeOperationRegistry,
    ) -> Self {
        self.native_operations = native_operations;
        self
    }

    pub(crate) async fn resolve_inline_native_owner_principal(
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

    // PYTHON_REFERENCE_V1: rust=rust/crates/kogwistar-api/src/sections/in_memory_service/native_worker.rs#execute_native_step_once; python=kogwistar/runtime/rust_worker.py#RustRuntimeWorker; fixture=contracts/golden/runtime-native-noop-v1.json,contracts/golden/runtime-native-increment-counter-v1.json,contracts/golden/runtime-native-emit-result-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/in_memory_provider_events.rs#in_memory_inline_native_result_is_persisted_and_completes_run; python_test=tests/runtime/test_rust_runtime_python_worker.py#test_python_worker_increment_counter_matches_shared_native_effect_golden; note=Rust-native operations stay on Rust path; Python worker explicitly rejects native-target operations.
    pub(crate) async fn dead_letter_native_lane(
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

    pub(crate) async fn claim_in_memory_native_provider_event(
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

    pub(crate) async fn execute_native_step_once(&self) -> Result<bool, String> {
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

    pub(crate) async fn in_memory_native_graph_effect_batch(
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

    pub(crate) async fn in_memory_provider_event_ack_graph_batch(
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

    pub(crate) async fn execute_in_memory_native_provider_event_request(
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

    pub(crate) async fn persist_in_memory_native_stream_semantic_event(
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
}
