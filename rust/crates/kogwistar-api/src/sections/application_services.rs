// PostgreSQL ApplicationService adapter and shared capability/route contract.
// Python counterparts: ChatRunService and runtime_api operation dispatch.
fn fail_inline_graph_effect(
    effect: &mut kogwistar_runtime::RecordedWorkerSuccessEffect,
    code: &str,
    message: &str,
) {
    effect.graph_mutation_batch = None;
    effect.graph_mutation_batch_v2 = None;
    effect.provider_event_acknowledgements.clear();
    effect.provider_event_publications.clear();
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
fn recorded_effect_has_graph_batch(
    effect: &kogwistar_runtime::RecordedWorkerSuccessEffect,
) -> bool {
    effect.graph_mutation_batch.is_some() || effect.graph_mutation_batch_v2.is_some()
}

fn recorded_effect_graph_mutations(
    effect: &kogwistar_runtime::RecordedWorkerSuccessEffect,
) -> Option<&[kogwistar_contracts::GraphMutationV1]> {
    effect
        .graph_mutation_batch_v2
        .as_ref()
        .map(|batch| batch.mutations.as_slice())
        .or_else(|| {
            effect
                .graph_mutation_batch
                .as_ref()
                .map(|batch| batch.mutations.as_slice())
        })
}

fn inline_graph_mutations_outside_owner_scope(
    principal: &Value,
    mutations: &[kogwistar_contracts::GraphMutationV1],
) -> bool {
    let security_scope = security_scope_value(principal);
    mutations.iter().any(|mutation| {
        !graph_mutation_scope_allowed(
            &GraphScope {
                namespace: mutation.scope.namespace.clone(),
                workspace_id: mutation.scope.workspace_id.clone(),
                graph_space: mutation.scope.graph_space.clone(),
            },
            &security_scope,
        )
    })
}

impl PostgresRunApplicationService {
    async fn completed_native_handoff_response(
        &self,
        handoff: &kogwistar_runtime::RecordedWorkerHandoff,
        principal: &Value,
    ) -> Result<Option<ApiEffectResponse>, ApiEffectResponse> {
        let lane = self
            .store
            .get_projected_lane_message(&handoff.message_id)
            .await
            .map_err(|_| {
                effect_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "KOGWISTAR_RUNTIME_HANDOFF_UNAVAILABLE",
                    "could not verify native handoff status",
                )
            })?;
        let Some(lane) = lane.filter(|lane| lane.status == "completed") else {
            return Ok(None);
        };
        if lane.run_id.as_deref() != Some(handoff.run_id.as_str())
            || lane.step_id.as_deref() != Some(handoff.step_id.as_str())
            || lane.correlation_id.as_deref() != Some(handoff.correlation_id.as_str())
        {
            return Err(effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_RUNTIME_HANDOFF_CONFLICT",
                "completed native handoff does not match run, step, and correlation identity",
            ));
        }
        let caller_id = authenticated_run_owner_id(principal).ok_or_else(|| {
            effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_RUN_OWNER_FORBIDDEN",
                "completed native handoff replay requires an authenticated run owner",
            )
        })?;
        let run = self.store.server_run(&handoff.run_id).await.map_err(|_| {
            effect_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "KOGWISTAR_RUNTIME_HANDOFF_UNAVAILABLE",
                "could not verify native handoff owner",
            )
        })?;
        if !run.is_some_and(|run| run.user_id.as_deref() == Some(caller_id)) {
            return Err(effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_RUN_OWNER_FORBIDDEN",
                "completed native handoff replay is restricted to the durable run owner",
            ));
        }
        Ok(Some(json_effect(
            StatusCode::OK,
            json!({
                "status": "completed",
                "idempotent_replay": true,
                "run_id": handoff.run_id,
                "step_id": handoff.step_id
            }),
        )))
    }

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
        let handoff = input.handoff;
        match self
            .completed_native_handoff_response(&handoff, &request.principal)
            .await
        {
            Ok(Some(response)) => return response,
            Ok(None) => {}
            Err(response) => return response,
        }
        let result = if native_request.operation == "agent.event.request" {
            match self
                .execute_native_provider_event_request(handoff.clone(), native_request.clone())
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
            }
        } else if native_request.operation == "aa_validate_or_repair_citations" {
            match self
                .execute_native_citation_repair(handoff.clone(), native_request.clone())
                .await
            {
                Ok(result) => result,
                Err(_) => {
                    return effect_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "KOGWISTAR_CHAT_PROVIDER_UNAVAILABLE",
                        "Rust-native citation repair provider is temporarily unavailable; the work remains unacknowledged",
                    );
                }
            }
        } else if native_request.operation == "aa_generate_answer_with_citations" {
            match self
                .execute_native_answer_generation(handoff.clone(), native_request.clone())
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
            }
        } else if native_request.operation == "aa_select_used_evidence"
            && crate::native_answer_generation::selection_snapshot_is_enabled(&native_request)
        {
            if self
                .persist_native_selection_context_snapshot(&native_request)
                .await
                .is_err()
            {
                return effect_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "KOGWISTAR_CONTEXT_SNAPSHOT_UNAVAILABLE",
                    "native evidence-selection context snapshot could not be committed; the work remains unacknowledged",
                );
            }
            match native_result_from_request(
                handoff.clone(),
                native_request.clone(),
                &self.native_operations,
            ) {
                Ok(result) => result,
                Err(_) => failed_native_result(
                    handoff,
                    &native_request,
                    "KOGWISTAR_NATIVE_EXECUTION_REJECTED",
                    "native evidence selection rejected its bounded input",
                ),
            }
        } else if native_request.operation == "aa_get_view_and_question" {
            match self
                .native_answer_view_context(&native_request, Some(&request.principal))
                .await
            {
                Ok(view) => {
                    native_request
                        .ephemeral_context
                        .insert("authorized_answer_view".to_owned(), view);
                    match native_result_from_request(
                        handoff.clone(),
                        native_request.clone(),
                        &self.native_operations,
                    ) {
                        Ok(result) => result,
                        Err(_) => failed_native_result(
                            handoff,
                            &native_request,
                            "KOGWISTAR_NATIVE_EXECUTION_REJECTED",
                            "native answer-view construction rejected its bounded input",
                        ),
                    }
                }
                Err(NativeEvidenceContextError::CallerNotRunOwner) => {
                    return effect_error(
                        StatusCode::FORBIDDEN,
                        "KOGWISTAR_RUN_OWNER_FORBIDDEN",
                        "native answer view requires the authenticated run owner",
                    );
                }
                Err(NativeEvidenceContextError::OwnerAuthorization) => failed_native_result(
                    handoff,
                    &native_request,
                    "KOGWISTAR_CAPABILITY_FORBIDDEN",
                    "native answer view requires the run owner's current read_graph capability and admitted scope",
                ),
                Err(NativeEvidenceContextError::GraphReadRejected) => failed_native_result(
                    handoff,
                    &native_request,
                    "KOGWISTAR_ANSWER_VIEW_REJECTED",
                    "native answer view requires an active user turn in the admitted conversation",
                ),
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
                    match native_result_from_request(
                        handoff.clone(),
                        native_request.clone(),
                        &self.native_operations,
                    ) {
                        Ok(result) => result,
                        Err(_) => failed_native_result(
                            handoff,
                            &native_request,
                            "KOGWISTAR_NATIVE_EXECUTION_REJECTED",
                            "native candidate retrieval rejected its bounded input",
                        ),
                    }
                }
                Err(NativeEvidenceContextError::CallerNotRunOwner) => {
                    return effect_error(
                        StatusCode::FORBIDDEN,
                        "KOGWISTAR_RUN_OWNER_FORBIDDEN",
                        "native graph reads require the authenticated run owner",
                    );
                }
                Err(NativeEvidenceContextError::OwnerAuthorization) => failed_native_result(
                    handoff,
                    &native_request,
                    "KOGWISTAR_CAPABILITY_FORBIDDEN",
                    "native candidate retrieval requires the run owner's current read_graph capability and admitted scope",
                ),
                Err(NativeEvidenceContextError::GraphReadRejected) => failed_native_result(
                    handoff,
                    &native_request,
                    "KOGWISTAR_CANDIDATE_RETRIEVAL_REJECTED",
                    "native candidate retrieval configuration or input is invalid",
                ),
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
                    match native_result_from_request(
                        handoff.clone(),
                        native_request.clone(),
                        &self.native_operations,
                    ) {
                        Ok(result) => result,
                        Err(_) => failed_native_result(
                            handoff,
                            &native_request,
                            "KOGWISTAR_NATIVE_EXECUTION_REJECTED",
                            "native evidence materialization rejected its bounded input",
                        ),
                    }
                }
                Err(NativeEvidenceContextError::CallerNotRunOwner) => {
                    return effect_error(
                        StatusCode::FORBIDDEN,
                        "KOGWISTAR_RUN_OWNER_FORBIDDEN",
                        "native evidence reads require the authenticated run owner",
                    );
                }
                Err(NativeEvidenceContextError::OwnerAuthorization) => failed_native_result(
                    handoff,
                    &native_request,
                    "KOGWISTAR_CAPABILITY_FORBIDDEN",
                    "native evidence materialization requires the run owner's current read_graph capability and admitted scope",
                ),
                Err(NativeEvidenceContextError::GraphReadRejected) => failed_native_result(
                    handoff,
                    &native_request,
                    "KOGWISTAR_EVIDENCE_READ_FAILED",
                    "native evidence materialization could not read the selected knowledge records",
                ),
                Err(NativeEvidenceContextError::GraphReadUnavailable) => {
                    return effect_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "KOGWISTAR_EVIDENCE_READ_UNAVAILABLE",
                        "native evidence storage is temporarily unavailable; the work remains unacknowledged",
                    );
                }
            }
        } else if native_request.operation == "aa_project_pointers" {
            match self
                .native_pointer_context(&native_request, Some(&request.principal))
                .await
            {
                Ok(records) => {
                    native_request
                        .ephemeral_context
                        .insert("authorized_pointer_records".to_owned(), records);
                    match native_result_from_request(
                        handoff.clone(),
                        native_request.clone(),
                        &self.native_operations,
                    ) {
                        Ok(result) => result,
                        Err(_) => failed_native_result(
                            handoff,
                            &native_request,
                            "KOGWISTAR_NATIVE_EXECUTION_REJECTED",
                            "native pointer projection rejected its bounded input",
                        ),
                    }
                }
                Err(NativeEvidenceContextError::CallerNotRunOwner) => {
                    return effect_error(
                        StatusCode::FORBIDDEN,
                        "KOGWISTAR_RUN_OWNER_FORBIDDEN",
                        "native pointer projection requires the authenticated run owner",
                    );
                }
                Err(NativeEvidenceContextError::OwnerAuthorization) => failed_native_result(
                    handoff,
                    &native_request,
                    "KOGWISTAR_CAPABILITY_FORBIDDEN",
                    "native pointer projection requires the run owner's current read_graph capability and admitted scope",
                ),
                Err(NativeEvidenceContextError::GraphReadRejected) => failed_native_result(
                    handoff,
                    &native_request,
                    "KOGWISTAR_POINTER_READ_FAILED",
                    "native pointer projection requires visible, active sources and run context",
                ),
                Err(NativeEvidenceContextError::GraphReadUnavailable) => {
                    return effect_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "KOGWISTAR_POINTER_READ_UNAVAILABLE",
                        "native pointer storage is temporarily unavailable; the work remains unacknowledged",
                    );
                }
            }
        } else if native_request.operation == "link_assistant_turn" {
            match self
                .native_turn_link_context(&native_request, Some(&request.principal))
                .await
            {
                Ok(records) => {
                    native_request
                        .ephemeral_context
                        .insert("authorized_turn_records".to_owned(), records);
                    match native_result_from_request(
                        handoff.clone(),
                        native_request.clone(),
                        &self.native_operations,
                    ) {
                        Ok(result) => result,
                        Err(_) => failed_native_result(
                            handoff,
                            &native_request,
                            "KOGWISTAR_NATIVE_EXECUTION_REJECTED",
                            "native assistant-turn linking rejected its bounded input",
                        ),
                    }
                }
                Err(NativeEvidenceContextError::CallerNotRunOwner) => {
                    return effect_error(
                        StatusCode::FORBIDDEN,
                        "KOGWISTAR_RUN_OWNER_FORBIDDEN",
                        "native turn linking requires the authenticated run owner",
                    );
                }
                Err(NativeEvidenceContextError::OwnerAuthorization) => failed_native_result(
                    handoff,
                    &native_request,
                    "KOGWISTAR_CAPABILITY_FORBIDDEN",
                    "native turn linking requires the run owner's current read_graph capability and admitted scope",
                ),
                Err(NativeEvidenceContextError::GraphReadRejected) => failed_native_result(
                    handoff,
                    &native_request,
                    "KOGWISTAR_TURN_READ_FAILED",
                    "native turn linking requires two visible, active turns in the same conversation",
                ),
                Err(NativeEvidenceContextError::GraphReadUnavailable) => {
                    return effect_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "KOGWISTAR_TURN_READ_UNAVAILABLE",
                        "native turn storage is temporarily unavailable; the work remains unacknowledged",
                    );
                }
            }
        } else {
            match native_result_from_request(handoff, native_request, &self.native_operations) {
                Ok(result) => result,
                Err(error) => {
                    return effect_error(
                        StatusCode::UNPROCESSABLE_ENTITY,
                        "KOGWISTAR_NATIVE_EXECUTION_REJECTED",
                        error,
                    );
                }
            }
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
    fn execute_inline_embedding_work_once(
        &self,
        limit: usize,
        lease_seconds: i64,
    ) -> Pin<Box<dyn Future<Output = Result<bool, String>> + Send + '_>> {
        Box::pin(async move {
            self.run_embedding_projection_jobs_once(limit, lease_seconds, None)
                .await
                .map(|summary| summary.claimed == limit)
        })
    }

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
            let (handoff, mut native_request) = native_request_for_claimed_work(&work)?;
            let mut result = if native_request.operation == "agent.event.request" {
                self.execute_native_provider_event_request(handoff.clone(), native_request.clone())
                    .await?
            } else if native_request.operation == "aa_validate_or_repair_citations" {
                self.execute_native_citation_repair(handoff.clone(), native_request.clone())
                    .await?
            } else if native_request.operation == "aa_generate_answer_with_citations" {
                self.execute_native_answer_generation(handoff.clone(), native_request.clone())
                    .await?
            } else if native_request.operation == "aa_select_used_evidence"
                && crate::native_answer_generation::selection_snapshot_is_enabled(&native_request)
            {
                self.persist_native_selection_context_snapshot(&native_request)
                    .await?;
                native_result_from_request(
                    handoff.clone(),
                    native_request.clone(),
                    &self.native_operations,
                )?
            } else if native_request.operation == "aa_get_view_and_question" {
                match self.native_answer_view_context(&native_request, None).await {
                    Ok(view) => {
                        native_request
                            .ephemeral_context
                            .insert("authorized_answer_view".to_owned(), view);
                        match native_result_from_request(
                            handoff.clone(),
                            native_request.clone(),
                            &self.native_operations,
                        ) {
                            Ok(result) => result,
                            Err(_) => failed_native_result(
                                handoff,
                                &native_request,
                                "KOGWISTAR_NATIVE_EXECUTION_REJECTED",
                                "native answer-view construction rejected its bounded input",
                            ),
                        }
                    }
                    Err(NativeEvidenceContextError::OwnerAuthorization) => failed_native_result(
                        handoff,
                        &native_request,
                        "KOGWISTAR_CAPABILITY_FORBIDDEN",
                        "native answer view requires the run owner's current read_graph capability and admitted scope",
                    ),
                    Err(NativeEvidenceContextError::GraphReadRejected) => failed_native_result(
                        handoff,
                        &native_request,
                        "KOGWISTAR_ANSWER_VIEW_REJECTED",
                        "native answer view requires an active user turn in the admitted conversation",
                    ),
                    Err(NativeEvidenceContextError::GraphReadUnavailable) => {
                        return Err(
                            "native answer-view storage is temporarily unavailable".to_owned()
                        );
                    }
                    Err(NativeEvidenceContextError::CallerNotRunOwner) => {
                        unreachable!("inline native execution has no external caller principal")
                    }
                }
            } else if native_request.operation == "aa_retrieve_candidates" {
                match self.native_candidate_context(&native_request, None).await {
                    Ok(records) => {
                        native_request
                            .ephemeral_context
                            .insert("authorized_candidate_records".to_owned(), records);
                        match native_result_from_request(
                            handoff.clone(),
                            native_request.clone(),
                            &self.native_operations,
                        ) {
                            Ok(result) => result,
                            Err(_) => failed_native_result(
                                handoff,
                                &native_request,
                                "KOGWISTAR_NATIVE_EXECUTION_REJECTED",
                                "native candidate retrieval rejected its bounded input",
                            ),
                        }
                    }
                    Err(NativeEvidenceContextError::OwnerAuthorization) => failed_native_result(
                        handoff,
                        &native_request,
                        "KOGWISTAR_CAPABILITY_FORBIDDEN",
                        "native candidate retrieval requires the run owner's current read_graph capability and admitted scope",
                    ),
                    Err(NativeEvidenceContextError::GraphReadRejected) => failed_native_result(
                        handoff,
                        &native_request,
                        "KOGWISTAR_CANDIDATE_RETRIEVAL_REJECTED",
                        "native candidate retrieval configuration or input is invalid",
                    ),
                    Err(NativeEvidenceContextError::GraphReadUnavailable) => {
                        return Err(
                            "native candidate retrieval is temporarily unavailable".to_owned()
                        );
                    }
                    Err(NativeEvidenceContextError::CallerNotRunOwner) => {
                        unreachable!("inline native execution has no external caller principal")
                    }
                }
            } else if native_request.operation == "aa_materialize_evidence_pack" {
                match self.native_evidence_context(&native_request, None).await {
                    Ok(records) => {
                        native_request
                            .ephemeral_context
                            .insert("authorized_evidence_records".to_owned(), records);
                        match native_result_from_request(
                            handoff.clone(),
                            native_request.clone(),
                            &self.native_operations,
                        ) {
                            Ok(result) => result,
                            Err(_) => failed_native_result(
                                handoff,
                                &native_request,
                                "KOGWISTAR_NATIVE_EXECUTION_REJECTED",
                                "native evidence materialization rejected its bounded input",
                            ),
                        }
                    }
                    Err(NativeEvidenceContextError::OwnerAuthorization) => failed_native_result(
                        handoff,
                        &native_request,
                        "KOGWISTAR_CAPABILITY_FORBIDDEN",
                        "native evidence materialization requires the run owner's current read_graph capability and admitted scope",
                    ),
                    Err(NativeEvidenceContextError::GraphReadRejected) => failed_native_result(
                        handoff,
                        &native_request,
                        "KOGWISTAR_EVIDENCE_READ_FAILED",
                        "native evidence materialization could not read the selected knowledge records",
                    ),
                    Err(NativeEvidenceContextError::GraphReadUnavailable) => {
                        return Err("native evidence storage is temporarily unavailable".to_owned());
                    }
                    Err(NativeEvidenceContextError::CallerNotRunOwner) => {
                        unreachable!("inline native execution has no external caller principal")
                    }
                }
            } else if native_request.operation == "aa_project_pointers" {
                match self.native_pointer_context(&native_request, None).await {
                    Ok(records) => {
                        native_request
                            .ephemeral_context
                            .insert("authorized_pointer_records".to_owned(), records);
                        match native_result_from_request(
                            handoff.clone(),
                            native_request.clone(),
                            &self.native_operations,
                        ) {
                            Ok(result) => result,
                            Err(_) => failed_native_result(
                                handoff,
                                &native_request,
                                "KOGWISTAR_NATIVE_EXECUTION_REJECTED",
                                "native pointer projection rejected its bounded input",
                            ),
                        }
                    }
                    Err(NativeEvidenceContextError::OwnerAuthorization) => failed_native_result(
                        handoff,
                        &native_request,
                        "KOGWISTAR_CAPABILITY_FORBIDDEN",
                        "native pointer projection requires the run owner's current read_graph capability and admitted scope",
                    ),
                    Err(NativeEvidenceContextError::GraphReadRejected) => failed_native_result(
                        handoff,
                        &native_request,
                        "KOGWISTAR_POINTER_READ_FAILED",
                        "native pointer projection requires visible, active sources and run context",
                    ),
                    Err(NativeEvidenceContextError::GraphReadUnavailable) => {
                        return Err("native pointer storage is temporarily unavailable".to_owned());
                    }
                    Err(NativeEvidenceContextError::CallerNotRunOwner) => {
                        unreachable!("inline native execution has no external caller principal")
                    }
                }
            } else if native_request.operation == "link_assistant_turn" {
                match self.native_turn_link_context(&native_request, None).await {
                    Ok(records) => {
                        native_request
                            .ephemeral_context
                            .insert("authorized_turn_records".to_owned(), records);
                        match native_result_from_request(
                            handoff.clone(),
                            native_request.clone(),
                            &self.native_operations,
                        ) {
                            Ok(result) => result,
                            Err(_) => failed_native_result(
                                handoff,
                                &native_request,
                                "KOGWISTAR_NATIVE_EXECUTION_REJECTED",
                                "native assistant-turn linking rejected its bounded input",
                            ),
                        }
                    }
                    Err(NativeEvidenceContextError::OwnerAuthorization) => failed_native_result(
                        handoff,
                        &native_request,
                        "KOGWISTAR_CAPABILITY_FORBIDDEN",
                        "native turn linking requires the run owner's current read_graph capability and admitted scope",
                    ),
                    Err(NativeEvidenceContextError::GraphReadRejected) => failed_native_result(
                        handoff,
                        &native_request,
                        "KOGWISTAR_TURN_READ_FAILED",
                        "native turn linking requires two visible, active turns in the same conversation",
                    ),
                    Err(NativeEvidenceContextError::GraphReadUnavailable) => {
                        return Err("native turn storage is temporarily unavailable".to_owned());
                    }
                    Err(NativeEvidenceContextError::CallerNotRunOwner) => {
                        unreachable!("inline native execution has no external caller principal")
                    }
                }
            } else if native_request.operation == "agent.event.receive" {
                match self.native_read_principal(&native_request, None).await {
                    Ok(principal) => {
                        let Some(context) = self
                            .claim_native_provider_event_context(&native_request, &principal)
                            .await?
                        else {
                            return Err(
                                "no authorized provider event is currently claimable".to_owned()
                            );
                        };
                        native_request.ephemeral_context.insert(
                            "authorized_provider_event".to_owned(),
                            context["event"].clone(),
                        );
                        native_request.ephemeral_context.insert(
                            "authorized_provider_event_claim".to_owned(),
                            context["claim"].clone(),
                        );
                        native_result_from_request(
                            handoff,
                            native_request,
                            &self.native_operations,
                        )?
                    }
                    Err(_) => failed_native_result(
                        handoff,
                        &native_request,
                        "KOGWISTAR_CAPABILITY_FORBIDDEN",
                        "provider-event receive requires the run owner's current read_graph capability and admitted scope",
                    ),
                }
            } else {
                native_result_from_request(handoff, native_request, &self.native_operations)?
            };
            let run_id = result.handoff.run_id.clone();
            let mut inline_principal = if result
                .effect
                .as_ref()
                .is_some_and(recorded_effect_has_graph_batch)
            {
                match self
                    .resolve_inline_native_owner_principal(&run_id, "write_graph")
                    .await
                {
                    Ok(principal) => Some(principal),
                    Err(InlineOwnerPrincipalError::Denied(_)) => {
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
                    Err(InlineOwnerPrincipalError::Unavailable(error)) => return Err(error),
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
                        .and_then(recorded_effect_graph_mutations),
                )
                .is_some_and(|(principal, mutations)| {
                    inline_graph_mutations_outside_owner_scope(principal, mutations)
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
        if surface == "root" && name == "document_id_from_file_name" {
            return DocumentStore::supports_document_projection_lookup(&self.store);
        }
        if surface == "root" && name == "kg_load_persisted" {
            return self.graph_authority_capability().supports_single_mutation();
        }
        if surface == "root" && name == "kg_propose_bruteforce" {
            return self.graph_authority_capability().supports_single_mutation();
        }
        rust_mcp_tool_supported(surface, name)
            && name != "propose_vector"
            && (surface != "root" || self.graph_authority_capability().supports_single_mutation())
            && (name != "wisdom.kg_upsert_graph"
                || (self.embedding_provider.is_some()
                    && self
                        .wisdom_store
                        .graph_authority_capability()
                        .supports_atomic_index_jobs()))
            && (name != "wisdom.semantic_seed_then_expand"
                || (self.embedding_provider.is_some()
                    && self
                        .wisdom_store
                        .graph_authority_capability()
                        .supports_single_mutation()))
            && (name != "kg_semantic_seed_then_expand_text"
                || (self.embedding_provider.is_some()
                    && self.graph_authority_capability().supports_single_mutation()))
            && (name != "workflow.tool_audit"
                || self.graph_authority_capability().supports_single_mutation())
    }

    fn supports_mcp_tool_for_principal(
        &self,
        surface: &str,
        name: &str,
        principal: &Value,
    ) -> bool {
        if name == "document_id_from_file_name" {
            if surface != "root"
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
            return effective_capabilities(principal, &state).contains("read_graph")
                && self.supports_mcp_tool(surface, name);
        }
        if name == "kg_load_persisted" {
            if surface != "root"
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
            return effective_capabilities(principal, &state).contains("read_graph")
                && self.supports_mcp_tool(surface, name);
        }
        if name == "kg_propose_bruteforce" {
            if surface != "root"
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
            return effective_capabilities(principal, &state).contains("read_graph")
                && self.supports_mcp_tool(surface, name);
        }
        if name == "workflow.operator_inbox" {
            if surface != "workflow"
                || !matches!(principal_role(principal).as_str(), "ro" | "rw")
                || !principal_strings(principal, "ns")
                    .iter()
                    .any(|namespace| namespace == "workflow" || namespace == "*")
            {
                return false;
            }
            let state = self
                .capabilities
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            return effective_capabilities(principal, &state).contains("project_view")
                && self.supports_mcp_tool(surface, name);
        }
        // Python counterpart: chat_mcp.workflow_visibility_snapshot's role/namespace decorators
        // and ChatService.visibility_snapshot's read_security_scope + project_view requirement.
        if name == "workflow.visibility_snapshot" {
            if surface != "workflow"
                || !matches!(principal_role(principal).as_str(), "ro" | "rw")
                || !principal_strings(principal, "ns")
                    .iter()
                    .any(|namespace| namespace == "workflow" || namespace == "*")
            {
                return false;
            }
            let state = self
                .capabilities
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let capabilities = effective_capabilities(principal, &state);
            return capabilities.contains("read_security_scope")
                && capabilities.contains("project_view")
                && self.supports_mcp_tool(surface, name);
        }
        if matches!(name, "workflow.process_table" | "workflow.blocked_runs") {
            if surface != "workflow"
                || !matches!(principal_role(principal).as_str(), "ro" | "rw")
                || !principal_strings(principal, "ns")
                    .iter()
                    .any(|namespace| namespace == "workflow" || namespace == "*")
            {
                return false;
            }
            let state = self
                .capabilities
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let capabilities = effective_capabilities(principal, &state);
            return (capabilities.contains("project_view")
                || capabilities.contains("workflow.run.read"))
                && self.supports_mcp_tool(surface, name);
        }
        let graph_read_namespace = match name {
            "kg_find_edges"
            | "kg_load_persisted"
            | "kg_propose_bruteforce"
            | "kg_k_hop"
            | "kg_neighbors"
            | "kg_shortest_path"
            | "kg_semantic_seed_then_expand_text"
            | "kg_viz_cytoscape_json"
            | "kg_viz_d3_json"
            | "propose_vector" => Some("docs"),
            "wisdom.semantic_seed_then_expand" => Some("wisdom"),
            _ => None,
        };
        if let Some(namespace) = graph_read_namespace {
            if surface != "root"
                || !matches!(principal_role(principal).as_str(), "ro" | "rw")
                || !principal_strings(principal, "ns")
                    .iter()
                    .any(|actual| actual == namespace || actual == "*")
            {
                return false;
            }
            let state = self
                .capabilities
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let authorized = effective_capabilities(principal, &state).contains("read_graph");
            return authorized && self.supports_mcp_tool(surface, name);
        }
        if name != "wisdom.kg_upsert_graph" {
            return self.supports_mcp_tool(surface, name);
        }
        if surface != "root"
            || principal_role(principal) != "rw"
            || !principal_strings(principal, "ns")
                .iter()
                .any(|namespace| namespace == "wisdom" || namespace == "*")
        {
            return false;
        }
        let state = self
            .capabilities
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        effective_capabilities(principal, &state).contains("write_graph")
            && self.supports_mcp_tool(surface, name)
    }

    fn supports_http_operation(&self, method: &str, path: &str) -> bool {
        if method == "POST" && path == "/api/add_index_entries" {
            return self.embedding_provider.is_some()
                && self.search_index_fts.is_some()
                && !PENDING_SERVER_CUTOVER_ROUTES.contains(&(method, path));
        }
        if method == "GET" && path == "/api/search_index_hybrid" {
            return self.embedding_provider.is_some()
                && self.search_index_fts.is_some()
                && !PENDING_SERVER_CUTOVER_ROUTES.contains(&(method, path));
        }
        FROZEN_OPENAPI_ROUTES.contains(&(method, path))
            && !is_transport_owned_http_route(method, path)
            && !PENDING_SERVER_CUTOVER_ROUTES.contains(&(method, path))
            && (method != "POST"
                || path != "/api/graph/upsert" && path != "/api/document.upsert_tree"
                || self
                    .graph_authority_capability()
                    .supports_atomic_index_jobs()
                    && self
                        .document_authority_capability()
                        .supports_projection_upsert())
    }

    fn execute(
        &self,
        request: ApiEffectRequest,
    ) -> Pin<Box<dyn Future<Output = ApiEffectResponse> + Send + '_>> {
        Box::pin(async move { self.execute_async(request).await })
    }
}
