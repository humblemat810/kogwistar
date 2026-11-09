// PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/in_memory_service.rs#graph_read_effect; python=kogwistar/graph_query.py#GraphQuery.neighbors; fixture=contracts/golden/graph-acl-traversal-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/part_08.rs#in_memory_mcp_acl_traversal_matches_shared_python_contract; python_test=tests/mcp/test_chat_mcp_tools.py#test_mcp_neighbors_preserve_acl_visibility_and_hide_topology
#[derive(Clone)]
struct InMemoryGraphApplicationService {
    store: InMemoryStore,
    capabilities: SharedCapabilityState,
    embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
    chat_completion_provider: Option<crate::native_answer_generation::OpenAiCompatibleChatConfig>,
    transient_run_deltas: crate::transient_deltas::TransientRunDeltas,
    search_index_fts: Option<Arc<Mutex<rusqlite::Connection>>>,
    owner_auth_store: Option<AuthStore>,
}

impl Default for InMemoryGraphApplicationService {
    fn default() -> Self {
        Self {
            store: InMemoryStore::default(),
            capabilities: Arc::new(Mutex::new(CapabilityState::default())),
            embedding_provider: None,
            chat_completion_provider: None,
            transient_run_deltas: crate::transient_deltas::TransientRunDeltas::default(),
            search_index_fts: None,
            owner_auth_store: None,
        }
    }
}

impl InMemoryGraphApplicationService {
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

    // PYTHON_REFERENCE_V1: rust=rust/crates/kogwistar-api/src/sections/in_memory_service.rs#execute_native_step_once; python=kogwistar/runtime/rust_worker.py#RustRuntimeWorker; fixture=contracts/golden/runtime-native-noop-v1.json,contracts/golden/runtime-native-increment-counter-v1.json,contracts/golden/runtime-native-emit-result-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/part_08.rs#in_memory_inline_native_result_is_persisted_and_completes_run; python_test=tests/runtime/test_rust_runtime_python_worker.py#test_python_worker_increment_counter_matches_shared_native_effect_golden; note=Rust-native operations stay on Rust path; Python worker explicitly rejects native-target operations.
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
        if !matches!(
            operation.as_deref(),
            Some(
                "noop"
                    | "increment-counter"
                    | "emit-result"
                    | "agent.event.request"
                    | "agent.event.receive"
                    | "agent.event.ack"
            )
        ) {
            self.dead_letter_native_lane(
                &message_id,
                &claimed_by,
                "in-memory inline runtime does not support this native operation",
            )
            .await?;
            return Ok(true);
        }
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
        let result = match if operation.as_deref() == Some("agent.event.request") {
            self.execute_in_memory_native_provider_event_request(handoff.clone(), native_request)
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
            native_result_from_request(
                handoff.clone(),
                native_request,
                &kogwistar_runtime::NativeOperationRegistry::default(),
            )
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
        if !effect.provider_event_acknowledgements.is_empty() {
            self.resolve_inline_native_owner_principal(&handoff.run_id, "write_graph")
                .await
                .map_err(|_| {
                    "provider-event acknowledgement owner authorization failed closed".to_owned()
                })?;
        }
        if let Err(error) = self
            .store
            .apply_claimed_recorded_worker_effect(handoff, effect)
        {
            self.dead_letter_native_lane(
                &message_id,
                &claimed_by,
                "in-memory worker effect could not be committed",
            )
            .await?;
            return Err(error.to_string());
        }
        Ok(true)
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
        native_result_from_request(
            handoff,
            request,
            &kogwistar_runtime::NativeOperationRegistry::default(),
        )
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
            "conversation",
            &owner_id,
            security_scope["security_scope"].as_str(),
            &durable.event_id,
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
            .any(|namespace| matches!(namespace.as_str(), "*" | "docs"))
        {
            return Err(effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_NAMESPACE_FORBIDDEN",
                "Graph reads require docs namespace",
            ));
        }
        let scope_value = security_scope_value(&request.principal);
        Ok(GraphScope {
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
        })
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
                    input.rid.trim(),
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

    // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/in_memory_service.rs#graph_semantic_seed_expand_effect; python=kogwistar/graph_query.py#GraphQuery.semantic_seed_then_expand_text; fixture=contracts/golden/semantic-seed-expand-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/part_08.rs#in_memory_semantic_seed_expand_is_available_through_authenticated_http_mcp; python_test=tests/primitives/test_graph_query.py#test_semantic_seed_then_expand_text
    async fn graph_semantic_seed_expand_effect(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        let scope = match self.authorized_graph_scope(request) {
            Ok(scope) => scope,
            Err(response) => return response,
        };
        let input: SemanticSeedExpandRequest = match serde_json::from_slice(&request.body) {
            Ok(input) => input,
            Err(error) => return invalid_graph_request(error.to_string()),
        };
        let query_text = input.text.trim();
        let top_k = input.top_k.unwrap_or(5);
        let hops = input.hops.unwrap_or(1);
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
            graph_records_visible_by_acl(&all_nodes, &all_edges, "knowledge", &principal);
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
        if let Some(where_filter) = input.where_filter.as_ref() {
            if let Err(error) = kogwistar_contracts::metadata_filter_matches_from_str(
                &json!({"metadata": {}, "where": where_filter}).to_string(),
            ) {
                return invalid_graph_request(error.to_string());
            }
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

    // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/in_memory_service.rs#graph_vector_proposal_effect; python=kogwistar/server/mcp_tools.py#propose_vector; fixture=contracts/golden/graph-vector-proposal-acl-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/part_08.rs#in_memory_backend_routes_authenticated_graph_write_and_mcp_read; python_test=tests/mcp/test_chat_mcp_tools.py#test_propose_vector_returns_only_acl_visible_pairs
    // Mirrors Python's propose_vector contract for the in-memory backend only;
    // PostgreSQL and SQLite+Chroma remain unadvertised until their snapshots
    // can provide the same pre-top-K filtering and ACL guarantees.
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
        let visible_node_ids = visible_nodes
            .iter()
            .map(|record| record.id.as_str())
            .collect::<BTreeSet<_>>();
        let visible_edge_ids = visible_edges
            .iter()
            .map(|record| record.id.as_str())
            .collect::<BTreeSet<_>>();
        let node_seeds = selected_proposal_seeds(&all_nodes, input.new_node_ids.as_deref());
        let edge_seeds = selected_proposal_seeds(&all_edges, input.new_edge_ids.as_deref());
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
            for (candidates, candidate_kind) in [
                (all_nodes.as_slice(), "node"),
                (all_edges.as_slice(), "edge"),
            ] {
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
                    let where_matches = match proposal_where_matches(candidate, where_filter) {
                        Ok(matches) => matches,
                        Err(error) => return invalid_graph_request(error),
                    };
                    if !where_matches {
                        continue;
                    }
                    let candidate_doc = proposal_doc_id(candidate);
                    if input.allowed_docs.as_ref().is_some_and(|docs| {
                        !docs.is_empty()
                            && candidate_doc.is_none_or(|doc| !docs.iter().any(|d| d == doc))
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
                for (candidate, _) in ranked.into_iter().take(input.top_k) {
                    let candidate_visible = if candidate_kind == "node" {
                        visible_node_ids.contains(candidate.id.as_str())
                    } else {
                        visible_edge_ids.contains(candidate.id.as_str())
                    };
                    if !candidate_visible {
                        continue;
                    }
                    let (left, right, left_kind, right_kind) = if seed.id <= candidate.id {
                        (&seed.id, &candidate.id, seed_kind, candidate_kind)
                    } else {
                        (&candidate.id, &seed.id, candidate_kind, seed_kind)
                    };
                    let key = (left.clone(), right.clone());
                    if seen.insert(key) {
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
        json_effect(
            StatusCode::OK,
            rust_native_attested(json!({"pairs": pairs})),
        )
    }

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
        surface == "root"
            && Self::supports_graph_tool(name)
            && (name != "kg_semantic_seed_then_expand_text" || self.embedding_provider.is_some())
    }

    fn supports_mcp_tool_for_principal(
        &self,
        surface: &str,
        name: &str,
        principal: &Value,
    ) -> bool {
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
        let route = path.split('?').next();
        if method == "POST" && route == Some("/api/workflow/runs") {
            return true;
        }
        if method == "GET"
            && runtime_run_route(path)
                .is_some_and(|(_, tail)| tail.is_empty() || tail.as_slice() == ["events"])
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
            && (route != Some(GRAPH_SEMANTIC_SEED_EXPAND_ROUTE)
                || self.embedding_provider.is_some())
            && matches!(
                route,
                Some(INTERNAL_GRAPH_MUTATIONS_ROUTE)
                    | Some(INTERNAL_GRAPH_LOAD_PERSISTED_ROUTE)
                    | Some(INTERNAL_GRAPH_NEIGHBORS_ROUTE)
                    | Some(INTERNAL_GRAPH_PROPOSE_BRUTE_FORCE_ROUTE)
                    | Some(INTERNAL_GRAPH_VECTOR_QUERY_ROUTE)
                    | Some(GRAPH_SEMANTIC_SEED_EXPAND_ROUTE)
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
            if request.method == "POST"
                && request.path_and_query.split('?').next() == Some("/api/workflow/runs")
            {
                return self.submit_run_effect(&request).await;
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
    // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/in_memory_service.rs#submit_run_effect; python=kogwistar/engine_core/in_memory_meta.py#create_server_run+append_server_run_event,kogwistar/messaging/service.py#project_lane_message; fixture=contracts/golden/runtime-native-noop-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/part_08.rs#in_memory_native_run_admission_is_authenticated_and_atomic, rust/crates/kogwistar-api/src/tests/part_08.rs#in_memory_native_admission_uses_authoritative_design_and_rejects_conflicts; python_test=tests/core/test_in_memory_meta_store.py#test_in_memory_meta_retry_and_run_cancel_contract; note=In-memory public admission validates an exact ready workflow-design snapshot when present, accepts only one unjoined Rust-native v1 step, and atomically admits run/event/transition/lane without language fallback.
    async fn submit_run_effect(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        if !principal_strings(&request.principal, "ns")
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
            "workflow.run.write",
            &["workflow.run.write"],
        ) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Workflow run admission requires workflow.run.write capability",
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
            let node_id = plan.start_node_id.as_str();
            let operation = plan.node_ops.get(node_id).map(String::as_str);
            let version = plan.node_versions.get(node_id).map(String::as_str);
            let executor_target = plan.node_executor_targets.get(node_id).map(String::as_str);
            if plan.node_ops.len() != 1
                || plan.node_versions.len() != 1
                || !plan.join_node_ids.is_empty()
                || plan.start_join_mask != 0
                || !plan.routes.is_empty()
                || !plan.node_environment_ids.is_empty()
                || !plan.node_worker_identities.is_empty()
                || operation.is_none_or(|operation| {
                    !matches!(
                        operation,
                        "noop"
                            | "increment-counter"
                            | "emit-result"
                            | "agent.event.request"
                            | "agent.event.receive"
                            | "agent.event.ack"
                    )
                })
                || version != Some("v1")
                || executor_target != Some(kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET)
            {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_RUNTIME_PLAN_UNSUPPORTED",
                    "in-memory native admission supports one unjoined v1 Rust-native start step only",
                );
            }
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
        if !input.join_node_ids.is_empty()
            || input.start_join_mask != 0
            || !input.runtime_routes.is_empty()
            || input.node_ops.len() > 1
            || input
                .node_ops
                .keys()
                .any(|node_id| node_id != input.start_node_id.as_deref().unwrap_or("start"))
            || !input.node_environment_ids.is_empty()
            || !input.node_worker_identities.is_empty()
        {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_RUNTIME_PLAN_UNSUPPORTED",
                "in-memory native admission currently accepts one unjoined native start step only",
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
        let start_node_id = input
            .start_node_id
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or("start");
        let operation = input
            .node_ops
            .get(start_node_id)
            .map(String::as_str)
            .unwrap_or("noop");
        let executor_target = input
            .node_executor_targets
            .get(start_node_id)
            .map(String::as_str)
            .unwrap_or(kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET);
        let native_operation =
            kogwistar_runtime::native_operation_for_target(operation, Some(executor_target));
        if executor_target != kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET
            || !matches!(
                native_operation,
                Some(
                    "noop"
                        | "increment-counter"
                        | "emit-result"
                        | "agent.event.request"
                        | "agent.event.receive"
                        | "agent.event.ack"
                )
            )
            || input
                .node_executor_targets
                .keys()
                .any(|node_id| node_id != start_node_id)
            || input
                .node_versions
                .get(start_node_id)
                .is_some_and(|version| version != "v1")
            || input
                .node_versions
                .keys()
                .any(|node_id| node_id != start_node_id)
        {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_RUNTIME_PLAN_UNSUPPORTED",
                "in-memory admission requires a supported v1 Rust-native operation",
            );
        }
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
        initial_state.insert(
            "_rt_graph_scope".to_owned(),
            security_scope_value(&request.principal),
        );
        initial_state.insert("_rt_run_owner_id".to_owned(), json!(owner_id));
        install_authoritative_node_executor_targets(
            &mut initial_state,
            &BTreeMap::from([(start_node_id.to_owned(), executor_target.to_owned())]),
        );
        inject_authoritative_capability_state(
            &mut initial_state,
            &request.principal,
            &self.capabilities,
        );
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
            "op_version": "v1",
            "executor_target": executor_target,
            "join_mask": 0,
            "token_id": run_id,
            "step_seq": 0,
            "expected_event_seq": 2,
            "state": initial_state,
        });
        add_executor_target(&mut worker_payload, operation, Some(executor_target));
        let mut transition_payload = worker_payload.clone();
        transition_payload["initial_state"] = worker_payload["state"].clone();
        transition_payload["start_node_id"] = json!(start_node_id);
        transition_payload["join_node_ids"] = json!([]);
        transition_payload["start_join_mask"] = json!(0);
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
        match self
            .store
            .admit_server_run_with_start(run, admission_payload, transition, lane)
        {
            Ok(_) => json_effect(
                StatusCode::ACCEPTED,
                json!({
                    "run_id": run_id,
                    "conversation_id": conversation_id,
                    "workflow_id": workflow_id,
                    "status": "queued",
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

    async fn read_runtime_run_effect(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        let Some((run_id, tail)) = runtime_run_route(&request.path_and_query) else {
            return unavailable_effect(request.clone());
        };
        if !principal_strings(&request.principal, "ns")
            .iter()
            .any(|namespace| matches!(namespace.as_str(), "*" | "workflow"))
        {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_NAMESPACE_FORBIDDEN",
                "Workflow run reads require workflow namespace",
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
            Ok(events) if tail.as_slice() == ["events"] => {
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

    // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/in_memory_service.rs#search_index_hybrid_effect; python=kogwistar/server_mcp_with_admin.py#search_index_hybrid; fixture=contracts/golden/search-index-hybrid-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/part_08.rs#in_memory_search_index_hybrid_is_scoped_and_acl_filtered; python_test=tests/core/test_search_index_contract.py#test_hybrid_search_fusion_matches_shared_rust_golden_fixture
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
