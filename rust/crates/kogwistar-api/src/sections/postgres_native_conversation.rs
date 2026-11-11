// Native conversation, messaging, and answer-context operations.
// Python counterparts: ConversationService, LaneMessagingService, and
// ChatRunService, intentionally split here by domain within one Rust adapter.
impl PostgresRunApplicationService {
    pub fn with_max_queue(mut self, max_queue: usize) -> Self {
        self.max_queue = max_queue.max(1);
        self
    }

    /// Opt in to server-side embedding for synchronous graph mutation batches.
    /// Batches carrying index jobs retain their asynchronous two-stage path.
    pub fn with_embedding_provider(mut self, provider: Arc<dyn EmbeddingProvider>) -> Self {
        self.embedding_provider = Some(provider);
        self
    }

    /// Configure the local FTS sidecar used by the hybrid search index.
    /// Vector data remains in the selected graph backend's node-index store.
    ///
    pub fn with_search_index_path(
        mut self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<Self, String> {
        let connection = crate::search_index_fts::open_index(path)
            .map_err(|error| format!("could not open search-index FTS database: {error}"))?;
        self.search_index_fts = Some(Arc::new(Mutex::new(connection)));
        Ok(self)
    }

    /// Configure a Rust-native OpenAI-compatible structured-answer provider.
    /// No Python worker fallback is used when this provider is missing.
    pub(crate) fn with_chat_completion_provider(
        mut self,
        provider: crate::native_answer_generation::OpenAiCompatibleChatConfig,
    ) -> Self {
        self.chat_completion_provider = Some(provider);
        self
    }

    async fn persist_native_stream_semantic_event(
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
        let attempt = kogwistar_runtime::native_trace_context_fields(request)
            .get("attempt")
            .and_then(Value::as_i64)
            .filter(|attempt| *attempt > 0)
            .unwrap_or(1);
        let trace_context = kogwistar_runtime::native_trace_context_fields(request);
        let event_trace = kogwistar_contracts::EventTraceMetadataV1 {
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
        let durable_event = kogwistar_contracts::durable_provider_event(&event_trace, event);
        let mut trace_payload = json!({
            "run_id": request.run_id,
            "trace_id": request.run_id,
            "workflow_id": request.workflow_id,
            "conversation_id": request.conversation_id,
            "turn_node_id": request.turn_node_id,
            "step_id": request.node_id,
            "step_seq": request.step_seq,
            "token_id": request.token_id,
            "parent_token_id": request.parent_token_id,
            "executor_target": "rust-native",
            "provider_id": provider_id,
            "provider_model": model,
            "client_sse_sequence": client_sse_sequence,
            "provider_event": summary,
        });
        if let Some(fields) = trace_payload.as_object_mut() {
            fields.extend(trace_context);
        }
        self.persist_native_provider_event(
            durable_event,
            request.conversation_id.clone(),
            trace_payload,
        )
        .await?;
        Ok(summary)
    }

    async fn execute_native_provider_event_request(
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
            .await?;
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
        if !self
            .store
            .graph_authority_capability()
            .supports_atomic_index_jobs()
        {
            return Err("provider-event request requires atomic graph authority".to_owned());
        }
        let run = self
            .store
            .server_run(&request.run_id)
            .await
            .map_err(|_| "provider-event run could not be verified".to_owned())?
            .ok_or_else(|| "provider-event run is missing".to_owned())?;
        if run.conversation_id != request.conversation_id {
            return Err("provider-event conversation does not match the admitted run".to_owned());
        }

        use std::sync::atomic::{AtomicI64, Ordering};
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
        let saw_failure = Arc::new(std::sync::atomic::AtomicBool::new(false));
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
                let service = self;
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
                        let _ =
                            service
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
                    let sequence = client_sequence.load(Ordering::Relaxed);
                    match service
                        .persist_native_stream_semantic_event(
                            &request,
                            event,
                            provider_id,
                            &model,
                            sequence,
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
                    "provider_stream_error".to_owned()
                } else {
                    "empty_provider_stream".to_owned()
                },
                message: "provider request did not complete with semantic output".to_owned(),
                retryable: false,
            };
            let summary = self
                .persist_native_stream_semantic_event(
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

    async fn native_generate_answer(
        &self,
        request: &kogwistar_runtime::RuntimeNativeStepRequest,
    ) -> Result<(Value, Vec<Value>), String> {
        let provider = self
            .chat_completion_provider
            .as_ref()
            .ok_or_else(|| "Rust-native structured-answer provider is not configured".to_owned())?;
        let runtime = request
            .state
            .get("_rt")
            .and_then(Value::as_object)
            .ok_or_else(|| "native answer generation requires runtime context".to_owned())?;
        let evidence_pack = runtime.get("evidence_pack").ok_or_else(|| {
            "native answer generation requires a materialized evidence pack".to_owned()
        })?;
        let question = request
            .state
            .get("question")
            .and_then(Value::as_str)
            .ok_or_else(|| "native answer generation requires a question".to_owned())?;
        let system_prompt = request
            .state
            .get("system_prompt")
            .and_then(Value::as_str)
            .ok_or_else(|| "native answer generation requires a system prompt".to_owned())?;
        let used_node_ids = request
            .state
            .get("used_node_ids")
            .and_then(Value::as_array)
            .ok_or_else(|| "native answer generation requires selected evidence IDs".to_owned())?
            .iter()
            .map(|id| {
                id.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| "selected evidence IDs must be strings".to_owned())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let service = self;
        let run_id = request.run_id.clone();
        let workflow_id = request.workflow_id.clone();
        let conversation_id = request.conversation_id.clone();
        let node_id = request.node_id.clone();
        let step_seq = request.step_seq;
        let token_id = request.token_id.clone();
        let parent_token_id = request.parent_token_id.clone();
        let turn_node_id = request
            .turn_node_id
            .clone()
            .or_else(|| {
                request
                    .state
                    .get("turn_node_id")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            })
            .ok_or_else(|| "native answer generation requires a turn node ID".to_owned())?;
        let trace_context = kogwistar_runtime::native_trace_context_fields(request);
        let model = provider.model().to_owned();
        provider
            .answer_with_event_sink(
                system_prompt,
                question,
                evidence_pack,
                &used_node_ids,
                move |event| {
                    let service = service;
                    let run_id = run_id.clone();
                    let workflow_id = workflow_id.clone();
                    let conversation_id = conversation_id.clone();
                    let node_id = node_id.clone();
                    let token_id = token_id.clone();
                    let parent_token_id = parent_token_id.clone();
                    let turn_node_id = turn_node_id.clone();
                    let trace_context = trace_context.clone();
                    let model = model.clone();
                    async move {
                        if let kogwistar_contracts::ProviderEventV1::TextDelta { delta, .. } =
                            &event
                        {
                            if delta.chars().count() > 4096 {
                                return Err(
                                    "transient text delta exceeds the configured size limit"
                                        .to_owned(),
                                );
                            }
                            // Client delivery is observational; bounded-buffer failure must not fail the workflow.
                            let _ = service.transient_run_deltas.publish_next(&run_id, delta);
                            return Ok(());
                        }
                        let Some(provider_event) =
                            crate::native_answer_generation::semantic_event_trace_summary(&event)
                        else {
                            return Ok(());
                        };
                        let event_value = serde_json::to_value(&event)
                            .map_err(|_| "could not encode provider event".to_owned())?;
                        let response_id = event_value
                            .get("response_id")
                            .and_then(Value::as_str)
                            .map(str::to_owned);
                        let provider_sequence =
                            event_value.get("provider_sequence").and_then(Value::as_i64);
                        let client_sse_sequence = service
                            .transient_run_deltas
                            .last_sequence(&run_id)
                            .unwrap_or_default();
                        let attempt = trace_context
                            .get("attempt")
                            .and_then(Value::as_i64)
                            .filter(|attempt| *attempt > 0)
                            .unwrap_or(1);
                        let event_trace = kogwistar_contracts::EventTraceMetadataV1 {
                            run_id: run_id.clone(),
                            turn_id: turn_node_id.clone(),
                            workflow_node_id: node_id.clone(),
                            step_id: format!(
                                "provider-response:{}",
                                response_id.as_deref().unwrap_or("unknown")
                            ),
                            attempt,
                            producer_step_seq: Some(step_seq),
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
                            provider_id: "openai_compatible_chat_completions".to_owned(),
                            provider_response_id: response_id,
                            provider_sequence,
                            client_sse_sequence: Some(client_sse_sequence),
                        };
                        let event_id =
                            kogwistar_contracts::durable_provider_event_id(&event_trace, &event);
                        let durable_event = kogwistar_contracts::DurableProviderEventV1 {
                            event_id,
                            trace: event_trace,
                            event,
                        };
                        let mut payload = json!({
                            "run_id": run_id,
                            "trace_id": run_id,
                            "workflow_id": workflow_id,
                            "conversation_id": conversation_id,
                            "turn_node_id": turn_node_id,
                            "step_id": node_id,
                            "step_seq": step_seq,
                            "token_id": token_id,
                            "parent_token_id": parent_token_id,
                            "executor_target": "rust-native",
                            "provider_id": "openai_compatible_chat_completions",
                            "provider_model": model,
                            "client_sse_sequence": client_sse_sequence,
                            "provider_event": provider_event,
                        });
                        if let Some(fields) = payload.as_object_mut() {
                            fields.extend(trace_context);
                        }
                        service
                            .persist_native_provider_event(durable_event, conversation_id, payload)
                            .await?;
                        Ok(())
                    }
                },
            )
            .await
    }

    async fn persist_native_answer_context_snapshot(
        &self,
        request: &kogwistar_runtime::RuntimeNativeStepRequest,
    ) -> Result<(), String> {
        let model = self
            .chat_completion_provider
            .as_ref()
            .ok_or_else(|| "Rust-native structured-answer provider is not configured".to_owned())?
            .model()
            .to_owned();
        let batch =
            crate::native_answer_generation::answer_context_snapshot_batch(request, &model)?;
        self.persist_native_context_snapshot_batch(request, batch)
            .await
    }

    async fn persist_native_selection_context_snapshot(
        &self,
        request: &kogwistar_runtime::RuntimeNativeStepRequest,
    ) -> Result<(), String> {
        kogwistar_runtime::validate_native_step_registration(request, &self.native_operations)
            .map_err(|_| {
                "native evidence selection is not registered for this operation version".to_owned()
            })?;
        let batch = crate::native_answer_generation::selection_context_snapshot_batch(request)?;
        self.persist_native_context_snapshot_batch(request, batch)
            .await
    }

    async fn persist_native_context_snapshot_batch(
        &self,
        request: &kogwistar_runtime::RuntimeNativeStepRequest,
        batch: kogwistar_contracts::GraphMutationBatchV2,
    ) -> Result<(), String> {
        let principal = self
            .resolve_inline_native_owner_principal(&request.run_id, "write_graph")
            .await?;
        if !capability_allowed_for_principal(
            &principal,
            &self.capabilities,
            "write_graph",
            &["write_graph"],
        ) || !principal_strings(&principal, "capabilities")
            .iter()
            .any(|capability| capability == "write_graph")
        {
            return Err(
                "native context snapshot requires the run owner's current write_graph capability"
                    .to_owned(),
            );
        }
        if !self
            .store
            .graph_authority_capability()
            .supports_atomic_index_jobs()
        {
            return Err("native context snapshot requires atomic graph authority".to_owned());
        }
        let (mutations, index_jobs) = graph_mutation_batch_v2_value(batch)
            .map_err(|_| "native context snapshot graph mutation is invalid".to_owned())?;
        let security_scope = security_scope_value(&principal);
        if mutations
            .iter()
            .any(|mutation| !graph_mutation_scope_allowed(&mutation.scope, &security_scope))
        {
            return Err(
                "native context snapshot scope is outside the run owner's admitted scope"
                    .to_owned(),
            );
        }
        let run_id = request.run_id.clone();
        let expected_owner = authenticated_run_owner_id(&principal)
            .ok_or_else(|| {
                "native context snapshot requires an authenticated run owner".to_owned()
            })?
            .to_owned();
        let owner_id = expected_owner.to_ascii_lowercase();
        let acl_scope = principal
            .get("security_scope")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_ascii_lowercase);
        let effect_id = format!(
            "native-context-snapshot|{}|{}|{}",
            request.run_id, request.step_seq, request.token_id
        );
        let commit_capabilities = self.capabilities.clone();
        let conversation_schema = self.conversation_store.schema().to_owned();
        self.store
            .transaction_with_graph_schemas(&[conversation_schema.as_str()], |uow| {
                let mutations = mutations.clone();
                let index_jobs = index_jobs.clone();
                let principal = principal.clone();
                let run_id = run_id.clone();
                let expected_owner = expected_owner.clone();
                let owner_id = owner_id.clone();
                let acl_scope = acl_scope.clone();
                let effect_id = effect_id.clone();
                let conversation_schema = conversation_schema.clone();
                let commit_capabilities = commit_capabilities.clone();
                Box::pin(async move {
                    let run = uow.get_server_run(&run_id).await?;
                    if !run.is_some_and(|run| {
                        run.user_id.as_deref() == Some(expected_owner.as_str())
                    }) {
                        return Err(kogwistar_store_postgres::PostgresStoreError::RecordedRuntimeConflict(
                            "native context snapshot run owner changed before commit".to_owned(),
                        ));
                    }
                    if !capability_allowed_for_principal(
                        &principal,
                        &commit_capabilities,
                        "write_graph.commit",
                        &["write_graph"],
                    ) || !principal_strings(&principal, "capabilities")
                        .iter()
                        .any(|capability| capability == "write_graph")
                    {
                        return Err(kogwistar_store_postgres::PostgresStoreError::RecordedRuntimeConflict(
                            "native context snapshot write_graph capability was revoked before commit".to_owned(),
                        ));
                    }
                    let commit_scope = security_scope_value(&principal);
                    if mutations.iter().any(|mutation| {
                        !graph_mutation_scope_allowed(&mutation.scope, &commit_scope)
                    }) {
                        return Err(kogwistar_store_postgres::PostgresStoreError::RecordedRuntimeConflict(
                            "native context snapshot scope changed before commit".to_owned(),
                        ));
                    }
                    let scope = mutations.first().map(|mutation| mutation.scope.clone()).ok_or_else(|| {
                        kogwistar_store_postgres::PostgresStoreError::RecordedRuntimeConflict(
                            "native context snapshot has no graph mutation".to_owned(),
                        )
                    })?;
                    let (acl_mutations, acl_jobs) = resolve_default_acl_graph_effects_in_schema(
                        uow,
                        &conversation_schema,
                        &scope,
                        &mutations,
                        "conversation",
                        &owner_id,
                        acl_scope.as_deref(),
                        &effect_id,
                    )
                    .await?;
                    for mutation in mutations {
                        uow.apply_graph_mutation_in_schema(&conversation_schema, mutation)
                            .await?;
                    }
                    for mutation in acl_mutations {
                        uow.apply_graph_mutation_in_schema(&conversation_schema, mutation)
                            .await?;
                    }
                    for job in index_jobs.into_iter().chain(acl_jobs) {
                        uow.enqueue_index_job_in_schema(&conversation_schema, job)
                            .await?;
                    }
                    Ok(())
                })
            })
            .await
            .map_err(|_| "native answer context snapshot could not be committed".to_owned())
    }

    async fn execute_native_answer_generation(
        &self,
        handoff: RecordedWorkerHandoff,
        mut request: kogwistar_runtime::RuntimeNativeStepRequest,
    ) -> Result<RuntimeWorkerResultRequest, String> {
        kogwistar_runtime::validate_native_step_registration(&request, &self.native_operations)
            .map_err(|error| format!("native execution rejected: {error}"))?;
        if self.chat_completion_provider.is_none() {
            return Ok(failed_native_result(
                handoff,
                &request,
                "KOGWISTAR_CHAT_PROVIDER_NOT_CONFIGURED",
                "Rust-native answer generation requires a configured structured chat provider",
            ));
        }
        self.persist_native_answer_context_snapshot(&request)
            .await?;
        let (answer, provider_event_traces) = self.native_generate_answer(&request).await?;
        request
            .ephemeral_context
            .insert("provider_answer".to_owned(), answer);
        request.ephemeral_context.insert(
            "_provider_event_trace_summaries".to_owned(),
            Value::Array(provider_event_traces),
        );
        request.ephemeral_context.insert(
            "provider_model".to_owned(),
            json!(
                self.chat_completion_provider
                    .as_ref()
                    .expect("provider availability checked above")
                    .model()
            ),
        );
        native_result_from_request(handoff, request, &self.native_operations)
    }

    async fn execute_native_citation_repair(
        &self,
        handoff: RecordedWorkerHandoff,
        mut request: kogwistar_runtime::RuntimeNativeStepRequest,
    ) -> Result<RuntimeWorkerResultRequest, String> {
        let needs_repair = crate::native_answer_generation::answer_needs_citation_repair(&request)
            .map_err(|_| "native answer citations could not be validated".to_owned())?;
        if needs_repair && let Some(provider) = self.chat_completion_provider.as_ref() {
            let runtime = request
                .state
                .get("_rt")
                .and_then(Value::as_object)
                .ok_or_else(|| "native citation repair requires runtime context".to_owned())?;
            let evidence_pack = runtime
                .get("evidence_pack")
                .ok_or_else(|| "native citation repair requires evidence".to_owned())?;
            let answer = request
                .state
                .get("answer")
                .ok_or_else(|| "native citation repair requires an answer".to_owned())?;
            let question = request
                .state
                .get("question")
                .and_then(Value::as_str)
                .ok_or_else(|| "native citation repair requires a question".to_owned())?;
            let system_prompt = request
                .state
                .get("system_prompt")
                .and_then(Value::as_str)
                .ok_or_else(|| "native citation repair requires a system prompt".to_owned())?;
            let used_node_ids = request
                .state
                .get("used_node_ids")
                .and_then(Value::as_array)
                .ok_or_else(|| "native citation repair requires selected evidence IDs".to_owned())?
                .iter()
                .map(|id| {
                    id.as_str()
                        .map(str::to_owned)
                        .ok_or_else(|| "selected evidence IDs must be strings".to_owned())
                })
                .collect::<Result<Vec<_>, _>>()?;
            let repaired = provider
                .repair(
                    system_prompt,
                    question,
                    evidence_pack,
                    &used_node_ids,
                    answer,
                )
                .await?;
            request
                .ephemeral_context
                .insert("repaired_answer".to_owned(), repaired);
            request
                .ephemeral_context
                .insert("provider_model".to_owned(), json!(provider.model()));
        }
        native_result_from_request(handoff, request, &self.native_operations)
    }

    /// Run one bounded batch of durable embedding jobs entirely through the
    /// Rust-owned PostgreSQL queue, canonical projection reads, and promotion
    /// transaction. Other index-job kinds remain unclaimed.
    pub async fn run_embedding_projection_jobs_once(
        &self,
        limit: usize,
        lease_seconds: i64,
        namespace: Option<&str>,
    ) -> Result<kogwistar_engine::projection_worker::ProjectionWorkerSummary, String> {
        if limit == 0 || limit > 1000 || lease_seconds <= 0 {
            return Err("limit must be 1..=1000 and lease_seconds must be positive".to_owned());
        }
        if namespace.is_some_and(|value| value.trim().is_empty()) {
            return Err("namespace must not be empty".to_owned());
        }
        let provider = self
            .embedding_provider
            .as_ref()
            .cloned()
            .ok_or_else(|| "Rust embedding provider is not configured".to_owned())?;
        provider
            .profile()
            .validate()
            .map_err(|error| error.to_string())?;
        self.ensure_graph_type_schemas().await?;

        let store = self.graph_store_for_type(&self.graph_role).clone();
        let provider_store = store.clone();
        let index_filter = IndexJobClaimFilter {
            namespace: namespace.map(str::to_owned),
            index_kinds: vec!["node_embedding".to_owned()],
        };
        kogwistar_engine::projection_worker::run_graph_projection_jobs_once_filtered_with_batch_provider_input(
            &store,
            &store,
            limit,
            lease_seconds,
            index_filter,
            move |inputs| {
                let provider = provider.clone();
                let store = provider_store.clone();
                async move {
                    let mut results = Vec::with_capacity(inputs.len());
                    let mut prepared = Vec::new();
                    for input in inputs {
                        let job_id = input.job_id.clone();
                        let result = async {
                            if input.schema_version != 1
                                || input.index_kind != "node_embedding"
                                || !matches!(input.entity_kind.as_str(), "node" | "edge")
                            {
                                return Err(kogwistar_store::StoreError::Backend {
                                    backend: "postgres-embedding-worker".to_owned(),
                                    message: "unsupported embedding job contract".to_owned(),
                                });
                            }
                            let record = if let Some(staged) = store
                                .stage1_graph_projection(
                                    &input.namespace,
                                    &input.entity_kind,
                                    &input.entity_id,
                                )
                                .await
                                .map_err(|_| kogwistar_store::StoreError::Backend {
                                    backend: "postgres-embedding-worker".to_owned(),
                                    message: "PostgreSQL Stage-1 read failed".to_owned(),
                                })?
                            {
                                let expected: serde_json::Value = serde_json::from_str(
                                    input.payload_json.as_deref().unwrap_or("{}"),
                                )
                                .map_err(|_| kogwistar_store::StoreError::Backend {
                                    backend: "postgres-embedding-worker".to_owned(),
                                    message: "embedding job revision payload is invalid".to_owned(),
                                })?;
                                let expected_fingerprint = expected
                                    .get("source_fingerprint")
                                    .and_then(serde_json::Value::as_str)
                                    .filter(|value| !value.is_empty())
                                    .ok_or_else(|| kogwistar_store::StoreError::Backend {
                                        backend: "postgres-embedding-worker".to_owned(),
                                        message: "embedding job has no source fingerprint".to_owned(),
                                    })?;
                                let expected_revision = expected
                                    .get("canonical_revision")
                                    .and_then(serde_json::Value::as_i64)
                                    .ok_or_else(|| kogwistar_store::StoreError::Backend {
                                        backend: "postgres-embedding-worker".to_owned(),
                                        message: "embedding job has no canonical revision".to_owned(),
                                    })?;
                                let mut metadata = staged.metadata;
                                metadata.insert(
                                    "_kogwistar_stage1_source_fingerprint".to_owned(),
                                    serde_json::Value::String(staged.source_fingerprint),
                                );
                                metadata.insert(
                                    "_kogwistar_stage1_expected_fingerprint".to_owned(),
                                    serde_json::Value::String(expected_fingerprint.to_owned()),
                                );
                                metadata.insert(
                                    "_kogwistar_stage1_revision".to_owned(),
                                    serde_json::Value::from(staged.revision),
                                );
                                metadata.insert(
                                    "_kogwistar_stage1_expected_revision".to_owned(),
                                    serde_json::Value::from(expected_revision),
                                );
                                kogwistar_store::GraphRecord {
                                    id: staged.entity_id,
                                    document: Some(staged.document),
                                    metadata,
                                    embedding: None,
                                }
                            } else {
                                let table =
                                    kogwistar_store::graph_projection_table(&input.entity_kind)?;
                                let mut records = store
                                    .graph_projection_records(GraphProjectionRead {
                                        scope: GraphScope {
                                            namespace: input.namespace.clone(),
                                            workspace_id: None,
                                            graph_space: None,
                                        },
                                        table: table.to_owned(),
                                        ids: Some(vec![input.entity_id.clone()]),
                                        metadata: MetadataFilter::default(),
                                        limit: 1,
                                    })
                                    .await
                                    .map_err(|_| kogwistar_store::StoreError::Backend {
                                        backend: "postgres-embedding-worker".to_owned(),
                                        message: "canonical graph projection read failed".to_owned(),
                                    })?;
                                let Some(record) = records.pop() else {
                                    return Err(kogwistar_store::StoreError::Backend {
                                        backend: "postgres-embedding-worker".to_owned(),
                                        message: "canonical graph projection is missing".to_owned(),
                                    });
                                };
                                record
                            };
                            if record.id != input.entity_id {
                                return Err(kogwistar_store::StoreError::Backend {
                                    backend: "postgres-embedding-worker".to_owned(),
                                    message: "canonical graph projection identity mismatch".to_owned(),
                                });
                            }
                            let document = record.document.clone().ok_or_else(|| {
                                kogwistar_store::StoreError::Backend {
                                    backend: "postgres-embedding-worker".to_owned(),
                                    message: "canonical graph document is missing".to_owned(),
                                }
                            })?;
                            Ok((record, document))
                        }
                        .await;
                        match result {
                            Ok((record, document)) => prepared.push((job_id, record, document)),
                            Err(error) => results.push((job_id, Err(error))),
                        }
                    }

                    if prepared.is_empty() {
                        return Ok(results);
                    }
                    let documents = prepared
                        .iter()
                        .map(|(_, _, document)| document.clone())
                        .collect::<Vec<_>>();
                    let vectors = match provider.embed(&documents).await {
                        Ok(vectors) if vectors.len() == prepared.len() => vectors,
                        Ok(_) => {
                            results.extend(prepared.into_iter().map(|(id, _, _)| {
                                (id, Err(kogwistar_store::StoreError::Backend {
                                    backend: "postgres-embedding-worker".to_owned(),
                                    message: "embedding provider returned wrong batch length".to_owned(),
                                }))
                            }));
                            return Ok(results);
                        }
                        Err(_) => {
                            results.extend(prepared.into_iter().map(|(id, _, _)| {
                                (id, Err(kogwistar_store::StoreError::Backend {
                                    backend: "postgres-embedding-worker".to_owned(),
                                    message: "embedding provider request failed".to_owned(),
                                }))
                            }));
                            return Ok(results);
                        }
                    };
                    for ((job_id, mut record, _), vector) in prepared.into_iter().zip(vectors) {
                        if vector.len() != provider.profile().dimension
                            || vector.iter().any(|value| !value.is_finite())
                        {
                            results.push((job_id, Err(kogwistar_store::StoreError::Backend {
                                backend: "postgres-embedding-worker".to_owned(),
                                message: "embedding provider returned an invalid vector".to_owned(),
                            })));
                            continue;
                        }
                        let vector = vector.iter().map(|value| *value as f32).collect::<Vec<_>>();
                        if vector.iter().any(|value| !value.is_finite()) {
                            results.push((job_id, Err(kogwistar_store::StoreError::NonFiniteVector)));
                            continue;
                        }
                        record.embedding = Some(vector);
                        results.push((job_id, Ok(record)));
                    }
                    Ok(results)
                }
            },
        )
        .await
        .map_err(|error| error.to_string())
    }

    async fn create_conversation_syscall(
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
        let Some(actor_id) = authenticated_run_owner_id(&request.principal) else {
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
        if !requested_user_id.eq_ignore_ascii_case(actor_id) {
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
            Ok(Some(value)) => value,
            Ok(None) => uuid::Uuid::new_v4().to_string(),
            Err(response) => return response,
        };
        let start_node_id = match optional_id("start_node_id") {
            Ok(value) => value,
            Err(response) => return response,
        };
        let user_id = actor_id.to_owned();
        let start_id = start_node_id.unwrap_or_else(|| {
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
        let (scope, _) = conversation_graph_scope_and_acl_principal(&request.principal);
        if scope.namespace != "conversation" {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_SCOPE_FORBIDDEN",
                "conversation creation requires conversation storage scope",
            );
        }
        let metadata = serde_json::Map::from_iter([
            ("namespace".to_owned(), json!(scope.namespace)),
            ("level_from_root".to_owned(), json!(0)),
            ("entity_type".to_owned(), json!("conversation_start")),
            ("turn_index".to_owned(), json!(-1)),
            ("in_conversation_chain".to_owned(), json!(true)),
            ("in_ui_chain".to_owned(), json!(true)),
            ("lifecycle_status".to_owned(), json!("active")),
            ("redirect_to_id".to_owned(), Value::Null),
            ("role".to_owned(), json!("system")),
            ("conversation_id".to_owned(), json!(conversation_id)),
            ("user_id".to_owned(), json!(user_id)),
        ]);
        let mut event_metadata = metadata.clone();
        event_metadata.remove("namespace");
        let properties = json!({"status": "active"});
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
        let payload = json!({
            "id": start_id,
            "label": "conversation start",
            "type": "entity",
            "summary": "Start of conversation",
            "domain_id": null,
            "canonical_entity_id": null,
            "properties": properties,
            "mentions": mentions,
            "embedding": null,
            "doc_id": null,
            "metadata": event_metadata,
            "level_from_root": 0,
            "role": "system",
            "turn_index": -1,
            "conversation_id": conversation_id,
            "user_id": user_id,
        });
        let envelope = GraphModelEnvelope::new(GraphModelClass::ConversationNode, payload.clone());
        if let Err(error) = envelope.validate_family_v1() {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_GRAPH_MODEL",
                error.to_string(),
            );
        }
        let mut record_metadata = payload["metadata"].as_object().cloned().unwrap_or_default();
        record_metadata.insert("namespace".to_owned(), json!(scope.namespace));
        record_metadata.insert("_class_name".to_owned(), json!("ConversationNode"));
        record_metadata.insert("label".to_owned(), json!("conversation start"));
        record_metadata.insert("type".to_owned(), json!("entity"));
        record_metadata.insert("summary".to_owned(), json!("Start of conversation"));
        record_metadata.insert("properties".to_owned(), json!(properties.to_string()));
        record_metadata.insert("mentions".to_owned(), json!(mentions.to_string()));
        let mut document_payload = payload.clone();
        if let Some(object) = document_payload.as_object_mut() {
            object.remove("metadata");
            object.remove("embedding");
        }
        let document = match serde_json::to_string(&document_payload) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_SERIALIZATION_FAILED",
                    error.to_string(),
                );
            }
        };
        let (embedding, embedding_dim) = if let Some(provider) = &self.embedding_provider {
            let profile = provider.profile();
            if let Err(error) = profile.validate() {
                return effect_error(
                    StatusCode::BAD_GATEWAY,
                    "KOGWISTAR_EMBEDDING_FAILED",
                    error.to_string(),
                );
            }
            let texts = [document.clone()];
            let vectors = match provider.embed(&texts).await {
                Ok(vectors) if vectors.len() == 1 => vectors,
                Ok(_) => {
                    return effect_error(
                        StatusCode::BAD_GATEWAY,
                        "KOGWISTAR_EMBEDDING_FAILED",
                        "embedding provider returned an unexpected vector count",
                    );
                }
                Err(_) => {
                    return effect_error(
                        StatusCode::BAD_GATEWAY,
                        "KOGWISTAR_EMBEDDING_FAILED",
                        "embedding provider failed",
                    );
                }
            };
            let vector = &vectors[0];
            if vector.len() != profile.dimension || vector.iter().any(|value| !value.is_finite()) {
                return effect_error(
                    StatusCode::BAD_GATEWAY,
                    "KOGWISTAR_EMBEDDING_FAILED",
                    "embedding provider returned an invalid vector",
                );
            }
            let vector = vector.iter().map(|value| *value as f32).collect::<Vec<_>>();
            if vector.iter().any(|value| !value.is_finite()) {
                return effect_error(
                    StatusCode::BAD_GATEWAY,
                    "KOGWISTAR_EMBEDDING_FAILED",
                    "embedding provider returned an invalid vector",
                );
            }
            (Some(vector), profile.dimension)
        } else {
            (None, 0)
        };
        let event_id = kogwistar_contracts::stable_id(
            "conversation.create.event",
            &[conversation_id.clone(), start_id.clone()],
        )
        .to_string();
        let mut event_payload = payload.clone();
        if let Some(object) = event_payload.as_object_mut() {
            object.remove("embedding");
        }
        let mutation = GraphMutation {
            scope: scope.clone(),
            table: "gke_nodes".to_owned(),
            entity_kind: "node".to_owned(),
            event_id: event_id.clone(),
            op: "ADD".to_owned(),
            payload: event_payload,
            record: GraphRecord {
                id: start_id.clone(),
                document: Some(document),
                metadata: record_metadata,
                embedding,
            },
            embedding_dim,
        };
        let owner_id = user_id.to_ascii_lowercase();
        let security_scope = security_scope_value(&request.principal)
            .get("security_scope")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_ascii_lowercase);
        let response_conversation_id = conversation_id.clone();
        let response_start_id = start_id.clone();
        let transaction = self
            .conversation_store
            .transaction(|uow| {
                Box::pin(async move {
                    uow.lock_entity_identity(&scope.namespace, "conversation", &conversation_id)
                        .await?;
                    let existing = uow
                        .graph_projection_records_for_update(GraphProjectionRead {
                            scope: scope.clone(),
                            table: "gke_nodes".to_owned(),
                            ids: None,
                            metadata: MetadataFilter {
                                equals: BTreeMap::from([(
                                    "conversation_id".to_owned(),
                                    json!(conversation_id),
                                )]),
                            },
                            limit: 10_000,
                        })
                        .await?;
                    if existing.iter().any(|record| {
                        record.metadata.get("entity_type").and_then(Value::as_str)
                            == Some("conversation_start")
                    }) {
                        return Err(
                            kogwistar_store_postgres::PostgresStoreError::RecordedRuntimeConflict(
                                "conversation_id already exists".to_owned(),
                            ),
                        );
                    }
                    uow.lock_entity_identity(&scope.namespace, "node", &start_id)
                        .await?;
                    let node_collision = uow
                        .graph_projection_records_for_update(GraphProjectionRead {
                            scope: scope.clone(),
                            table: "gke_nodes".to_owned(),
                            ids: Some(vec![start_id.clone()]),
                            metadata: MetadataFilter::default(),
                            limit: 1,
                        })
                        .await?;
                    if !node_collision.is_empty() {
                        return Err(
                            kogwistar_store_postgres::PostgresStoreError::RecordedRuntimeConflict(
                                "start_node_id already exists".to_owned(),
                            ),
                        );
                    }
                    let (acl_mutations, index_jobs) = resolve_default_acl_graph_effects(
                        uow,
                        &scope,
                        std::slice::from_ref(&mutation),
                        "conversation",
                        &owner_id,
                        security_scope.as_deref(),
                        &event_id,
                    )
                    .await?;
                    uow.apply_graph_mutation(mutation).await?;
                    for acl_mutation in acl_mutations {
                        uow.apply_graph_mutation(acl_mutation).await?;
                    }
                    for index_job in index_jobs {
                        uow.enqueue_index_job(index_job).await?;
                    }
                    Ok(())
                })
            })
            .await;
        match transaction {
            Ok(()) => syscall_result(
                &input.version,
                "create_conversation",
                json_effect(
                    StatusCode::CREATED,
                    json!({
                        "conversation_id": response_conversation_id,
                        "user_id": user_id,
                        "status": "active",
                        "start_node_id": response_start_id,
                        "tail_node_id": response_start_id,
                        "turn_count": 0,
                    }),
                ),
            ),
            Err(error) => effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_CONVERSATION_CREATE_CONFLICT",
                error.to_string(),
            ),
        }
    }

    async fn send_message_syscall(
        &self,
        request: &ApiEffectRequest,
        input: &SyscallInput,
    ) -> ApiEffectResponse {
        let conversation_id = input
            .args
            .get("conversation_id")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let Some(conversation_id) = conversation_id else {
            return effect_error(
                StatusCode::BAD_REQUEST,
                "KOGWISTAR_INVALID_REQUEST",
                "conversation_id is required",
            );
        };
        let text = input
            .args
            .get("text")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let Some(text) = text else {
            return effect_error(
                StatusCode::BAD_REQUEST,
                "KOGWISTAR_INVALID_REQUEST",
                "text must be non-empty",
            );
        };
        let workflow_id = input.args["workflow_id"]
            .as_str()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or("agentic_answering.v2");
        let (scope, acl_principal) = conversation_graph_scope_and_acl_principal(&request.principal);
        let nodes = match self.authorized_conversation_nodes(&request.principal).await {
            Ok(nodes) => nodes,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error,
                );
            }
        };
        let conversation_nodes = nodes
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
        let mut starts = conversation_nodes
            .iter()
            .filter(|record| {
                conversation_record_string(record, "entity_type").as_deref()
                    == Some("conversation_start")
            })
            .collect::<Vec<_>>();
        starts.sort_by_key(|record| record.id.clone());
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
        let requested_user_id = match input.args.get("user_id") {
            None | Some(Value::Null) => None,
            Some(Value::String(requested)) if requested.trim().is_empty() => None,
            Some(Value::String(requested)) => Some(requested.as_str()),
            Some(_) => {
                return effect_error(
                    StatusCode::BAD_REQUEST,
                    "KOGWISTAR_INVALID_REQUEST",
                    "user_id must be a string when provided",
                );
            }
        };
        let effective_user_id =
            requested_user_id.or_else(|| authenticated_run_owner_id(&request.principal));
        if effective_user_id != Some(owner_id.as_str()) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_RUN_OWNER_FORBIDDEN",
                "user_id must match the owner of the target conversation",
            );
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
        if let Some(workspace) = scope.workspace_id.as_deref() {
            owner_principal["workspace"] = json!(workspace);
        }
        if let Some(graph_space) = scope.graph_space.as_deref() {
            owner_principal["graph_space"] = json!(graph_space);
        }
        if let Some(security_scope) = acl_principal.security_scope.as_deref() {
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
        let turn_id = uuid::Uuid::new_v4().to_string();
        let run_id = uuid::Uuid::new_v4().to_string();
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
        let previous = tail.last();
        let turn_index = previous
            .and_then(|record| conversation_record_i64(record, "turn_index"))
            .map(|index| index + 1)
            .unwrap_or(0);
        let step_seq = previous
            .and_then(|record| conversation_record_i64(record, "run_step_seq"))
            .unwrap_or(0)
            + 1;
        let span = conversation_span(conversation_id, &turn_id, text);
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
        let node_mentions = json!([{"spans": [span], "multimodal_spans": []}]);
        let node_payload = json!({
            "id": turn_id,
            "label": format!("Turn {turn_index} (user)"),
            "type": "entity",
            "summary": text,
            "domain_id": null,
            "canonical_entity_id": null,
            "properties": {},
            "mentions": node_mentions,
            "doc_id": turn_id,
            "metadata": node_metadata,
            "level_from_root": 0,
            "role": "user",
            "turn_index": turn_index,
            "conversation_id": conversation_id,
            "user_id": owner_id,
        });
        if let Err(error) = kogwistar_contracts::GraphModelEnvelope::new(
            kogwistar_contracts::GraphModelClass::ConversationNode,
            node_payload.clone(),
        )
        .validate_family_v1()
        {
            return effect_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "KOGWISTAR_CONVERSATION_MODEL_INVALID",
                error.to_string(),
            );
        }
        let node_document = {
            let mut value = node_payload.clone();
            value.as_object_mut().unwrap().remove("metadata");
            serde_json::to_string(&value).expect("conversation node payload serializes")
        };
        let mut node_metadata_projection = node_metadata.as_object().unwrap().clone();
        node_metadata_projection.extend([
            ("doc_id".to_owned(), json!(turn_id)),
            (
                "label".to_owned(),
                json!(format!("Turn {turn_index} (user)")),
            ),
            ("type".to_owned(), json!("entity")),
            ("summary".to_owned(), json!(text)),
            ("properties".to_owned(), json!("{}")),
            (
                "mentions".to_owned(),
                json!(serde_json::to_string(&node_mentions).unwrap()),
            ),
        ]);
        if let Some(graph_space) = &scope.graph_space {
            node_metadata_projection.insert("graph_space".to_owned(), json!(graph_space));
        }
        if let Some(workspace_id) = &scope.workspace_id {
            node_metadata_projection.insert("workspace_id".to_owned(), json!(workspace_id));
        }

        let mut mutations = Vec::new();
        let Some(node_index_kinds) = conversation_index_job_kinds("node") else {
            return effect_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "KOGWISTAR_INDEX_CONTRACT_INVALID",
                "conversation node index contract is unavailable",
            );
        };
        let mut index_jobs = node_index_kinds
            .iter()
            .map(|kind| conversation_index_job(&scope.namespace, "node", &turn_id, kind))
            .collect::<Vec<_>>();
        let mut edge_payload_and_document = None;
        if let Some(previous) = previous {
            let previous_id = previous.id.clone();
            let (char_distance, turn_distance, tail_turn_index) =
                conversation_next_turn_distances(previous, text, turn_index);
            let edge_index = turn_index + 1;
            let edge_id = kogwistar_contracts::stable_id(
                "conversation.edge",
                &[
                    owner_id.clone(),
                    conversation_id.to_owned(),
                    "next_turn".to_owned(),
                    edge_index.to_string(),
                    format!("['{previous_id}']"),
                    format!("['{turn_id}']"),
                    "[]".to_owned(),
                    "[]".to_owned(),
                    "conversation_edge".to_owned(),
                ],
            )
            .to_string();
            let edge_metadata = json!({
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
            });
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
                "mentions": node_mentions,
                "domain_id": null,
                "canonical_entity_id": null,
                "properties": {"entity_type": "conversation_edge"},
                "metadata": edge_metadata,
            });
            if let Err(error) = kogwistar_contracts::GraphModelEnvelope::new(
                kogwistar_contracts::GraphModelClass::ConversationEdge,
                edge_payload.clone(),
            )
            .validate_family_v1()
            {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_CONVERSATION_MODEL_INVALID",
                    error.to_string(),
                );
            }
            let edge_document =
                serde_json::to_string(&edge_payload).expect("conversation edge payload serializes");
            edge_payload_and_document = Some((edge_payload, edge_document, edge_id));
        }
        let embedding_texts = edge_payload_and_document
            .as_ref()
            .map(|(_, edge_document, _)| vec![format!("user: {text}"), edge_document.clone()])
            .unwrap_or_else(|| vec![format!("user: {text}")]);
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
        let node_mutation = GraphMutation {
            scope: scope.clone(),
            table: "gke_nodes".to_owned(),
            entity_kind: "node".to_owned(),
            event_id: uuid::Uuid::new_v4().to_string(),
            op: "ADD".to_owned(),
            payload: node_payload.clone(),
            record: GraphRecord {
                id: turn_id.clone(),
                document: Some(node_document),
                metadata: node_metadata_projection,
                embedding: Some(embeddings[0].iter().map(|v| *v as f32).collect()),
            },
            embedding_dim: profile.dimension,
        };
        mutations.push((
            node_mutation,
            kogwistar_contracts::GraphModelEnvelope::new(
                kogwistar_contracts::GraphModelClass::ConversationNode,
                node_payload,
            ),
        ));
        if let Some((edge_payload, edge_document, edge_id)) = edge_payload_and_document {
            let edge_metadata = edge_payload["metadata"].as_object().unwrap().clone();
            let source_ids = serde_json::to_string(&edge_payload["source_ids"])
                .expect("edge source IDs serialize");
            let target_ids = serde_json::to_string(&edge_payload["target_ids"])
                .expect("edge target IDs serialize");
            let source_edge_ids = serde_json::to_string(&edge_payload["source_edge_ids"])
                .expect("edge source-edge IDs serialize");
            let target_edge_ids = serde_json::to_string(&edge_payload["target_edge_ids"])
                .expect("edge target-edge IDs serialize");
            let properties = serde_json::to_string(&edge_payload["properties"])
                .expect("edge properties serialize");
            let mentions =
                serde_json::to_string(&edge_payload["mentions"]).expect("edge mentions serialize");
            let mut metadata = edge_metadata;
            metadata.extend([
                ("doc_id".to_owned(), edge_payload["doc_id"].clone()),
                ("relation".to_owned(), json!("next_turn")),
                ("source_ids".to_owned(), json!(source_ids)),
                ("target_ids".to_owned(), json!(target_ids)),
                ("source_edge_ids".to_owned(), json!(source_edge_ids)),
                ("target_edge_ids".to_owned(), json!(target_edge_ids)),
                ("type".to_owned(), json!("relationship")),
                ("summary".to_owned(), json!("Sequential flow")),
                ("properties".to_owned(), json!(properties)),
                ("mentions".to_owned(), json!(mentions)),
                ("node_endpoint_count".to_owned(), json!(2)),
                ("edge_endpoint_count".to_owned(), json!(0)),
                ("total_endpoint_count".to_owned(), json!(2)),
            ]);
            if let Some(graph_space) = &scope.graph_space {
                metadata.insert("graph_space".to_owned(), json!(graph_space));
            }
            if let Some(workspace_id) = &scope.workspace_id {
                metadata.insert("workspace_id".to_owned(), json!(workspace_id));
            }
            mutations.push((
                GraphMutation {
                    scope: scope.clone(),
                    table: "gke_edges".to_owned(),
                    entity_kind: "edge".to_owned(),
                    event_id: uuid::Uuid::new_v4().to_string(),
                    op: "ADD".to_owned(),
                    payload: edge_payload.clone(),
                    record: GraphRecord {
                        id: edge_id.clone(),
                        document: Some(edge_document),
                        metadata,
                        embedding: Some(embeddings[1].iter().map(|v| *v as f32).collect()),
                    },
                    embedding_dim: profile.dimension,
                },
                kogwistar_contracts::GraphModelEnvelope::new(
                    kogwistar_contracts::GraphModelClass::ConversationEdge,
                    edge_payload,
                ),
            ));
            let Some(edge_index_kinds) = conversation_index_job_kinds("edge") else {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_INDEX_CONTRACT_INVALID",
                    "conversation edge index contract is unavailable",
                );
            };
            index_jobs.extend(
                edge_index_kinds
                    .iter()
                    .map(|kind| conversation_index_job(&scope.namespace, "edge", &edge_id, kind)),
            );
        }
        let prepared = PreparedConversationAdmission {
            conversation_id: conversation_id.to_owned(),
            owner_id: owner_id.clone(),
            expected_tail_id: previous.map(|record| record.id.clone()),
            scope: scope.clone(),
            acl_principal,
            mutations,
            index_jobs,
            actor_user_id: authenticated_run_owner_id(&request.principal)
                .map(str::to_owned)
                .unwrap_or_else(|| principal_subject(&request.principal)),
        };
        let mut initial_state = serde_json::Map::new();
        initial_state.insert("conversation_id".to_owned(), json!(conversation_id));
        initial_state.insert("user_text".to_owned(), json!(text));
        initial_state.insert("turn_node_id".to_owned(), json!(turn_id));
        initial_state.insert("user_id".to_owned(), json!(owner_id));
        initial_state.insert("turn_index".to_owned(), json!(turn_index));
        initial_state.insert("max_retrieval_level".to_owned(), json!(4));
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
            .expect("message workflow request serializes"),
            principal: owner_principal,
        };
        self.submit_run_with_conversation_admission(&nested, Some(prepared))
            .await
    }
}
