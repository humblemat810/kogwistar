// Native execution of claimed workflow steps and native operations.
// Python parity: kogwistar/runtime/runtime.py (step execution and transition outcomes).
impl SqliteRunApplicationService {
    fn execute_native_runtime_step(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
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
        if native_request.operation == "aa_get_view_and_question" {
            return effect_error(
                StatusCode::NOT_IMPLEMENTED,
                "KOGWISTAR_NATIVE_ANSWER_VIEW_ASYNC_REQUIRED",
                "native answer-view execution requires the asynchronous application-service path",
            );
        }
        if native_request.operation == "agent.event.request" {
            return effect_error(
                StatusCode::NOT_IMPLEMENTED,
                "KOGWISTAR_PROVIDER_EVENT_ASYNC_REQUIRED",
                "native provider-event requests require the asynchronous application-service path",
            );
        }
        if native_request.operation == "aa_generate_answer_with_citations" {
            return effect_error(
                StatusCode::NOT_IMPLEMENTED,
                "KOGWISTAR_NATIVE_ANSWER_GENERATION_ASYNC_REQUIRED",
                "native answer generation requires the asynchronous application-service path",
            );
        }
        if native_request.operation == "aa_select_used_evidence"
            && crate::native_answer_generation::selection_snapshot_is_enabled(&native_request)
        {
            return effect_error(
                StatusCode::NOT_IMPLEMENTED,
                "KOGWISTAR_NATIVE_SELECTION_SNAPSHOT_ASYNC_REQUIRED",
                "default native evidence selection requires the asynchronous snapshot transaction path",
            );
        }
        self.execute_native_request(input.handoff, native_request)
    }

    async fn execute_native_runtime_step_async(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
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
        let mut native_request =
            match native_step_request_from_payload(&input.handoff, input.payload) {
                Ok(value) => value,
                Err(error) => {
                    return effect_error(
                        StatusCode::CONFLICT,
                        "KOGWISTAR_NATIVE_HANDOFF_CONFLICT",
                        error,
                    );
                }
            };
        if native_request.operation == "agent.event.request" {
            let result = match self
                .execute_sqlite_native_provider_event_request(input.handoff.clone(), native_request)
                .await
            {
                Ok(result) => result,
                Err(_) => {
                    return effect_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "KOGWISTAR_PROVIDER_EVENT_UNAVAILABLE",
                        "Rust-native provider event request is temporarily unavailable; the work remains unacknowledged",
                    );
                }
            };
            return self.apply_runtime_result_input(
                result,
                kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET,
                None,
            );
        } else if native_request.operation == "aa_generate_answer_with_citations" {
            let result = match self
                .execute_sqlite_native_answer_generation(input.handoff.clone(), native_request)
                .await
            {
                Ok(result) => result,
                Err(_) => {
                    return effect_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "KOGWISTAR_CHAT_PROVIDER_UNAVAILABLE",
                        "Rust-native structured chat provider is temporarily unavailable; the work remains unacknowledged",
                    );
                }
            };
            return self.apply_runtime_result_input(
                result,
                kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET,
                None,
            );
        } else if native_request.operation == "aa_get_view_and_question" {
            match self
                .native_answer_view_context(&native_request, Some(&request.principal))
                .await
            {
                Ok(view) => {
                    native_request
                        .ephemeral_context
                        .insert("authorized_answer_view".to_owned(), view);
                }
                Err(NativeEvidenceContextError::CallerNotRunOwner) => {
                    return effect_error(
                        StatusCode::FORBIDDEN,
                        "KOGWISTAR_RUN_OWNER_FORBIDDEN",
                        "native answer view requires the authenticated run owner",
                    );
                }
                Err(NativeEvidenceContextError::OwnerAuthorization) => {
                    return self.apply_runtime_result_input(
                        failed_native_result(
                            input.handoff,
                            &native_request,
                            "KOGWISTAR_CAPABILITY_FORBIDDEN",
                            "native answer view requires the run owner's current read_graph capability and admitted scope",
                        ),
                        kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET,
                        None,
                    );
                }
                Err(NativeEvidenceContextError::GraphReadRejected) => {
                    return self.apply_runtime_result_input(
                        failed_native_result(
                            input.handoff,
                            &native_request,
                            "KOGWISTAR_ANSWER_VIEW_REJECTED",
                            "native answer view requires an active user turn in the admitted conversation",
                        ),
                        kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET,
                        None,
                    );
                }
                Err(NativeEvidenceContextError::GraphReadUnavailable) => {
                    return effect_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "KOGWISTAR_ANSWER_VIEW_UNAVAILABLE",
                        "native answer-view storage is temporarily unavailable; the work remains unacknowledged",
                    );
                }
            }
        } else if native_request.operation == "aa_retrieve_candidates" {
            match self
                .native_candidate_context(&native_request, Some(&request.principal))
                .await
            {
                Ok(records) => {
                    native_request
                        .ephemeral_context
                        .insert("authorized_candidate_records".to_owned(), records);
                }
                Err(NativeEvidenceContextError::CallerNotRunOwner) => {
                    return effect_error(
                        StatusCode::FORBIDDEN,
                        "KOGWISTAR_RUN_OWNER_FORBIDDEN",
                        "native graph reads require the authenticated run owner",
                    );
                }
                Err(NativeEvidenceContextError::OwnerAuthorization) => {
                    return self.apply_runtime_result_input(
                        failed_native_result(
                            input.handoff,
                            &native_request,
                            "KOGWISTAR_CAPABILITY_FORBIDDEN",
                            "native candidate retrieval requires the run owner's current read_graph capability and admitted scope",
                        ),
                        kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET,
                        None,
                    );
                }
                Err(NativeEvidenceContextError::GraphReadRejected) => {
                    return self.apply_runtime_result_input(
                        failed_native_result(
                            input.handoff,
                            &native_request,
                            "KOGWISTAR_CANDIDATE_RETRIEVAL_REJECTED",
                            "native candidate retrieval configuration or input is invalid",
                        ),
                        kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET,
                        None,
                    );
                }
                Err(NativeEvidenceContextError::GraphReadUnavailable) => {
                    return effect_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "KOGWISTAR_CANDIDATE_RETRIEVAL_UNAVAILABLE",
                        "native candidate retrieval is temporarily unavailable; the work remains unacknowledged",
                    );
                }
            }
        } else if native_request.operation == "aa_materialize_evidence_pack" {
            match self
                .native_evidence_context(&native_request, Some(&request.principal))
                .await
            {
                Ok(records) => {
                    native_request
                        .ephemeral_context
                        .insert("authorized_evidence_records".to_owned(), records);
                }
                Err(NativeEvidenceContextError::CallerNotRunOwner) => {
                    return effect_error(
                        StatusCode::FORBIDDEN,
                        "KOGWISTAR_RUN_OWNER_FORBIDDEN",
                        "native evidence reads require the authenticated run owner",
                    );
                }
                Err(NativeEvidenceContextError::OwnerAuthorization) => {
                    return self.apply_runtime_result_input(
                        failed_native_result(
                            input.handoff,
                            &native_request,
                            "KOGWISTAR_CAPABILITY_FORBIDDEN",
                            "native evidence materialization requires the run owner's current read_graph capability and admitted scope",
                        ),
                        kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET,
                        None,
                    );
                }
                Err(NativeEvidenceContextError::GraphReadRejected) => {
                    return self.apply_runtime_result_input(
                        failed_native_result(
                            input.handoff,
                            &native_request,
                            "KOGWISTAR_EVIDENCE_READ_FAILED",
                            "native evidence materialization could not read the selected knowledge records",
                        ),
                        kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET,
                        None,
                    );
                }
                Err(NativeEvidenceContextError::GraphReadUnavailable) => {
                    return effect_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "KOGWISTAR_EVIDENCE_READ_UNAVAILABLE",
                        "native evidence storage is temporarily unavailable; the work remains unacknowledged",
                    );
                }
            }
        }
        if native_request.operation == "aa_select_used_evidence"
            && crate::native_answer_generation::selection_snapshot_is_enabled(&native_request)
        {
            if self
                .persist_sqlite_native_selection_context_snapshot(&native_request)
                .await
                .is_err()
            {
                return effect_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "KOGWISTAR_CONTEXT_SNAPSHOT_UNAVAILABLE",
                    "native evidence-selection context snapshot could not be committed; the work remains unacknowledged",
                );
            }
            let result = match native_result_from_request(
                input.handoff.clone(),
                native_request,
                &self.native_operations,
            ) {
                Ok(result) => result,
                Err(_) => {
                    return effect_error(
                        StatusCode::UNPROCESSABLE_ENTITY,
                        "KOGWISTAR_NATIVE_EXECUTION_REJECTED",
                        "native evidence selection rejected its bounded input",
                    );
                }
            };
            return self.apply_runtime_result_input(
                result,
                kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET,
                None,
            );
        }
        self.execute_native_request_async(input.handoff, native_request)
            .await
    }

    fn execute_native_request(
        &self,
        handoff: RecordedWorkerHandoff,
        native_request: kogwistar_runtime::RuntimeNativeStepRequest,
    ) -> ApiEffectResponse {
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
            handoff,
            transition: None,
            effect: Some(effect),
        };
        self.apply_runtime_result_input(
            result,
            kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET,
            None,
        )
    }

    async fn execute_native_request_async(
        &self,
        handoff: RecordedWorkerHandoff,
        native_request: kogwistar_runtime::RuntimeNativeStepRequest,
    ) -> ApiEffectResponse {
        let mut result = match kogwistar_runtime::execute_native_step_with_registry(
            &native_request,
            &self.native_operations,
        ) {
            Ok(effect) => RuntimeWorkerResultRequest {
                handoff,
                transition: None,
                effect: Some(effect),
            },
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_NATIVE_EXECUTION_REJECTED",
                    error.to_string(),
                );
            }
        };
        let needs_graph_authority = result.effect.as_ref().is_some_and(|effect| {
            effect.graph_mutation_batch.is_some()
                || effect.graph_mutation_batch_v2.is_some()
                || !effect.provider_event_acknowledgements.is_empty()
        });
        let principal = if needs_graph_authority {
            match self
                .resolve_inline_native_owner_principal(&result.handoff.run_id, "write_graph")
                .await
            {
                Ok(principal) => Some(principal),
                Err(InlineOwnerPrincipalError::Denied(_)) => {
                    let effect = result
                        .effect
                        .as_mut()
                        .expect("graph authority was requested for an effect");
                    fail_inline_graph_effect(
                        effect,
                        "KOGWISTAR_GRAPH_AUTHORIZATION_DENIED",
                        "inline graph effects failed current owner authorization",
                    );
                    None
                }
                Err(InlineOwnerPrincipalError::Unavailable(_)) => {
                    return effect_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "KOGWISTAR_RUN_OWNER_UNAVAILABLE",
                        "inline graph effect could not revalidate the run owner",
                    );
                }
            }
        } else {
            None
        };
        self.apply_runtime_result_input(
            result,
            kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET,
            principal.as_ref(),
        )
    }
}

fn sqlite_native_answer_view_from_visible_records(
    nodes: &[GraphRecord],
    edges: &[GraphRecord],
    conversation_id: &str,
    turn_id: &str,
) -> Result<Value, NativeEvidenceContextError> {
    fn record_document(record: &GraphRecord) -> Option<Value> {
        record
            .document
            .as_deref()
            .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
    }

    fn field<'a>(record: &'a GraphRecord, document: &'a Value, key: &str) -> Option<&'a Value> {
        let metadata = document.get("metadata").and_then(Value::as_object);
        document
            .get(key)
            .or_else(|| metadata.and_then(|metadata| metadata.get(key)))
            .or_else(|| record.metadata.get(key))
    }

    let Some(turn) = nodes.iter().find(|record| record.id == turn_id) else {
        return Err(NativeEvidenceContextError::GraphReadRejected);
    };
    let Some(turn_document) = record_document(turn) else {
        return Err(NativeEvidenceContextError::GraphReadRejected);
    };
    if field(turn, &turn_document, "entity_type").and_then(Value::as_str)
        != Some("conversation_turn")
        || field(turn, &turn_document, "conversation_id").and_then(Value::as_str)
            != Some(conversation_id)
        || field(turn, &turn_document, "role").and_then(Value::as_str) != Some("user")
        || field(turn, &turn_document, "lifecycle_status")
            .and_then(Value::as_str)
            .is_some_and(|status| matches!(status, "tombstoned" | "deleted"))
    {
        return Err(NativeEvidenceContextError::GraphReadRejected);
    }

    let record_value = |record: &GraphRecord| {
        json!({
            "id": record.id,
            "document": record.document,
            "metadata": record.metadata
        })
    };
    let context_nodes = nodes
        .iter()
        .filter(|record| {
            record_document(record).is_some_and(|document| {
                field(record, &document, "conversation_id").and_then(Value::as_str)
                    == Some(conversation_id)
            })
        })
        .map(record_value)
        .collect::<Vec<_>>();
    let conversation_document_id = format!("conv:{conversation_id}");
    let context_edges = edges
        .iter()
        .filter(|record| {
            record.metadata.get("doc_id").and_then(Value::as_str)
                == Some(conversation_document_id.as_str())
                || record
                    .metadata
                    .get("conversation_id")
                    .and_then(Value::as_str)
                    == Some(conversation_id)
        })
        .map(record_value)
        .collect::<Vec<_>>();

    Ok(json!({
        "turn": record_value(turn),
        "context_nodes": context_nodes,
        "context_edges": context_edges
    }))
}

fn chroma_record_matches_principal_scope(record: &GraphRecord, principal: &Value) -> bool {
    let workspace = principal
        .get("workspace_id")
        .or_else(|| principal.get("workspace"))
        .and_then(Value::as_str);
    if workspace.is_some_and(|workspace| {
        record.metadata.get("workspace_id").and_then(Value::as_str) != Some(workspace)
    }) {
        return false;
    }
    let graph_space = principal.get("graph_space").and_then(Value::as_str);
    if graph_space.is_some_and(|graph_space| {
        record.metadata.get("graph_space").and_then(Value::as_str) != Some(graph_space)
    }) {
        return false;
    }
    true
}

fn conversation_record_from_raw_event(
    event: &SqliteRawEntityEvent,
) -> Result<GraphRecord, SqliteStoreError> {
    let payload = serde_json::from_str::<Value>(&event.payload_json).map_err(|error| {
        SqliteStoreError::TransactionAborted(format!(
            "conversation event {:?} has invalid canonical JSON: {error}",
            event.event_id
        ))
    })?;
    let metadata = payload
        .get("metadata")
        .and_then(Value::as_object)
        .cloned()
        .ok_or_else(|| {
            SqliteStoreError::TransactionAborted(format!(
                "conversation event {:?} has no metadata object",
                event.event_id
            ))
        })?;
    Ok(GraphRecord {
        id: event.entity_id.clone(),
        document: None,
        metadata,
        embedding: None,
    })
}

fn sqlite_native_candidate_values(
    candidates: &[GraphRecord],
    acl_records: &[GraphRecord],
    truth_graph: &str,
    principal: &GraphAclPrincipal,
) -> Value {
    let mut acl_inputs = candidates
        .iter()
        .filter(|record| {
            !record
                .metadata
                .get("lifecycle_status")
                .and_then(Value::as_str)
                .is_some_and(|status| matches!(status, "tombstoned" | "deleted"))
        })
        .cloned()
        .collect::<Vec<_>>();
    acl_inputs.extend_from_slice(acl_records);
    let (visible_candidates, _) =
        graph_records_visible_by_acl(&acl_inputs, &[], truth_graph, principal);
    let mut projected = visible_candidates
        .into_iter()
        .map(|record| {
            json!({
                "id": record.id,
                "label": record.metadata.get("label").cloned().unwrap_or(Value::Null),
                "summary": record.metadata.get("summary").cloned().unwrap_or(Value::Null),
                "doc": record.document.unwrap_or_default(),
            })
        })
        .collect::<Vec<_>>();
    projected.sort_by(|left, right| {
        left.get("id")
            .and_then(Value::as_str)
            .cmp(&right.get("id").and_then(Value::as_str))
    });
    Value::Array(projected)
}

fn apply_sqlite_provider_event_acknowledgement(
    uow: &mut SqliteUnitOfWork<'_>,
    acknowledgement: &kogwistar_runtime::ProviderEventAcknowledgementV1,
    run_id: &str,
    state: &RecordedRuntimeState,
    effect_id: &str,
    scope: &GraphScope,
    owner_id: &str,
    principal: &Value,
) -> Result<(), kogwistar_store_sqlite::SqliteStoreError> {
    let lane = uow
        .get_projected_lane_message(&acknowledgement.message_id)?
        .ok_or_else(|| {
            kogwistar_store_sqlite::SqliteStoreError::RecordedRuntimeConflict(
                "provider event acknowledgement lane does not exist".to_owned(),
            )
        })?;
    let run = uow.get_server_run(run_id)?;
    if !run
        .as_ref()
        .is_some_and(|run| run.user_id.as_deref() == Some(owner_id))
    {
        return Err(
            kogwistar_store_sqlite::SqliteStoreError::RecordedRuntimeConflict(
                "provider event run owner changed before acknowledgement".to_owned(),
            ),
        );
    }
    let event: kogwistar_contracts::DurableProviderEventV1 = lane
        .payload_json
        .as_deref()
        .and_then(|payload| serde_json::from_str(payload).ok())
        .ok_or_else(|| {
            kogwistar_store_sqlite::SqliteStoreError::RecordedRuntimeConflict(
                "provider event payload is unavailable for causal linkage".to_owned(),
            )
        })?;
    if acknowledgement.message_id != format!("msg:{}", acknowledgement.event_id)
        || lane.namespace != scope.namespace
        || lane.purpose != "agent_event"
        || lane.inbox_id != "agent-provider-events"
        || lane.msg_type != "agent.provider_event"
        || lane.recipient_id != "agent-event-consumer"
        || lane.status != "claimed"
        || lane.claimed_by.as_deref() != Some(acknowledgement.claimed_by.as_str())
        || lane.run_id.as_deref() != Some(run_id)
        || lane.conversation_id
            != run
                .as_ref()
                .map(|run| run.conversation_id.as_str())
                .unwrap_or_default()
        || lane.correlation_id.as_deref() != Some(acknowledgement.event_id.as_str())
        || event.event_id != acknowledgement.event_id
        || event.trace.run_id != run_id
        || state.conversation_id != lane.conversation_id
        || state.run_id != run_id
    {
        return Err(
            kogwistar_store_sqlite::SqliteStoreError::RecordedRuntimeConflict(
                "provider event acknowledgement is not bound to the active run claim".to_owned(),
            ),
        );
    }
    let handler_step_seq = state
        .state
        .get("agent_event_handling_step_seq")
        .and_then(Value::as_i64)
        .filter(|step_seq| *step_seq >= 0)
        .ok_or_else(|| {
            kogwistar_store_sqlite::SqliteStoreError::RecordedRuntimeConflict(
                "provider event has no valid durable handler step sequence".to_owned(),
            )
        })?;
    let mut referenced_step_seqs = vec![handler_step_seq];
    if let Some(producer_step_seq) = event.trace.producer_step_seq {
        if producer_step_seq < 0 {
            return Err(
                kogwistar_store_sqlite::SqliteStoreError::RecordedRuntimeConflict(
                    "provider event has an invalid producer step sequence".to_owned(),
                ),
            );
        }
        referenced_step_seqs.push(producer_step_seq);
    }
    referenced_step_seqs.sort_unstable();
    referenced_step_seqs.dedup();
    let recorded_steps = uow.recorded_runtime_steps(run_id, &referenced_step_seqs)?;
    let snapshots = provider_event_step_snapshots(
        recorded_steps,
        run_id,
        &lane.conversation_id,
        &state.workflow_id,
        &referenced_step_seqs,
    )
    .map_err(kogwistar_store_sqlite::SqliteStoreError::RecordedRuntimeConflict)?;
    let workflow_node_id = state
        .last_node_id
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| {
            kogwistar_store_sqlite::SqliteStoreError::RecordedRuntimeConflict(
                "provider event handling step has no workflow node identity".to_owned(),
            )
        })?;
    let step_seq = state.last_step_seq;
    if step_seq < 0
        || handler_step_seq >= step_seq
        || event
            .trace
            .producer_step_seq
            .is_some_and(|producer_step_seq| producer_step_seq >= step_seq)
    {
        return Err(
            kogwistar_store_sqlite::SqliteStoreError::RecordedRuntimeConflict(
                "provider event causal steps must precede the acknowledgement step".to_owned(),
            ),
        );
    }
    let security_scope = security_scope_value(principal);
    if !graph_mutation_scope_allowed(scope, &security_scope) {
        return Err(
            kogwistar_store_sqlite::SqliteStoreError::RecordedRuntimeConflict(
                "provider event handling graph scope changed before commit".to_owned(),
            ),
        );
    }
    let mut causal_mutations = provider_event_ack_graph_mutations(
        &acknowledgement.event_id,
        run_id,
        &lane.conversation_id,
        &state.workflow_id,
        workflow_node_id,
        step_seq,
        effect_id,
        scope,
    );
    causal_mutations.extend(provider_event_step_link_mutations(
        &acknowledgement.event_id,
        run_id,
        &lane.conversation_id,
        event.trace.producer_step_seq,
        handler_step_seq,
        &snapshots,
        scope,
    ));
    let mut linked_step_nodes = BTreeSet::new();
    causal_mutations.retain(|mutation| {
        mutation.entity_kind != "node" || linked_step_nodes.insert(mutation.record.id.clone())
    });
    let (targets, _) = acl_targets_for_graph_mutations(&causal_mutations);
    let entity_ids = acl_target_entity_ids(&targets);
    let existing = uow.graph_acl_records_for_entities(scope, "conversation", &entity_ids)?;
    let (acl_mutations, acl_jobs) = derive_default_acl_graph_effects_from_records(
        scope,
        &causal_mutations,
        targets,
        existing,
        "conversation",
        &graph_acl_principal_id(principal),
        security_scope["security_scope"].as_str(),
        &acknowledgement.event_id,
    )
    .map_err(kogwistar_store_sqlite::SqliteStoreError::TransactionAborted)?;
    causal_mutations.extend(acl_mutations);
    uow.apply_canonical_graph_mutation_batch(GraphMutationBatch {
        mutations: causal_mutations,
        index_jobs: acl_jobs,
    })?;
    uow.ack_claimed_projected_lane_message(
        &acknowledgement.message_id,
        &acknowledgement.claimed_by,
    )?;
    uow.append_server_run_event(
        run_id,
        "workflow.provider_event.handled.v1",
        serde_json::to_string(&json!({
            "event_id": acknowledgement.event_id,
            "message_id": acknowledgement.message_id,
            "workflow_node_id": workflow_node_id,
            "step_seq": step_seq,
            "effect_id": effect_id,
        }))?,
    )?;
    Ok(())}
