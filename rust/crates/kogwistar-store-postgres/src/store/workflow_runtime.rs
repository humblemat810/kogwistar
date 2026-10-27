use super::super::*;

impl PostgresStore {
    pub async fn put_workflow_design_snapshot(
        &self,
        workflow_id: &str,
        snapshot: WorkflowDesignSnapshotWrite,
    ) -> PostgresStoreResult<()> {
        let workflow_id = workflow_id.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.put_workflow_design_snapshot(&workflow_id, snapshot)
                    .await
            })
        })
        .await
    }

    pub async fn get_workflow_design_snapshot(
        &self,
        workflow_id: &str,
        max_version: i64,
        schema_version: i64,
    ) -> PostgresStoreResult<Option<WorkflowDesignSnapshot>> {
        let client = self.client().await?;
        workflow_design_snapshot(
            &**client,
            &self.tables,
            workflow_id,
            max_version,
            schema_version,
        )
        .await
    }

    pub async fn clear_workflow_design_snapshots(
        &self,
        workflow_id: &str,
    ) -> PostgresStoreResult<()> {
        let workflow_id = workflow_id.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move { uow.clear_workflow_design_snapshots(&workflow_id).await })
        })
        .await
    }

    pub async fn put_workflow_design_delta(
        &self,
        workflow_id: &str,
        delta: WorkflowDesignDeltaWrite,
    ) -> PostgresStoreResult<()> {
        let workflow_id = workflow_id.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move { uow.put_workflow_design_delta(&workflow_id, delta).await })
        })
        .await
    }

    pub async fn get_workflow_design_delta(
        &self,
        workflow_id: &str,
        version: i64,
        schema_version: i64,
    ) -> PostgresStoreResult<Option<WorkflowDesignDelta>> {
        let client = self.client().await?;
        workflow_design_delta(
            &**client,
            &self.tables,
            workflow_id,
            version,
            schema_version,
        )
        .await
    }

    pub async fn clear_workflow_design_deltas(&self, workflow_id: &str) -> PostgresStoreResult<()> {
        let workflow_id = workflow_id.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move { uow.clear_workflow_design_deltas(&workflow_id).await })
        })
        .await
    }

    pub async fn create_server_run(&self, run: ServerRunCreate) -> PostgresStoreResult<()> {
        self.transaction(move |uow| Box::pin(async move { uow.create_server_run(run).await }))
            .await
    }
    pub async fn get_server_run(&self, run_id: &str) -> PostgresStoreResult<Option<ServerRun>> {
        server_run(&**self.client().await?, &self.tables, run_id).await
    }
    pub async fn list_server_runs(
        &self,
        status: Option<&str>,
        workflow_id: Option<&str>,
        conversation_id: Option<&str>,
        limit: usize,
    ) -> PostgresStoreResult<Vec<ServerRun>> {
        server_runs(
            &**self.client().await?,
            &self.tables,
            status,
            workflow_id,
            conversation_id,
            limit,
        )
        .await
    }
    pub async fn append_server_run_event(
        &self,
        run_id: &str,
        event_type: &str,
        payload_json: String,
    ) -> PostgresStoreResult<ServerRunEvent> {
        let run_id = run_id.to_owned();
        let event_type = event_type.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.append_server_run_event(&run_id, &event_type, payload_json)
                    .await
            })
        })
        .await
    }
    pub async fn list_server_run_events(
        &self,
        run_id: &str,
        after_seq: i64,
        limit: usize,
    ) -> PostgresStoreResult<Vec<ServerRunEvent>> {
        server_run_events(
            &**self.client().await?,
            &self.tables,
            run_id,
            after_seq,
            limit,
        )
        .await
    }
    pub async fn update_server_run(
        &self,
        run_id: &str,
        update: ServerRunUpdate,
    ) -> PostgresStoreResult<()> {
        let run_id = run_id.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move { uow.update_server_run(&run_id, update).await })
        })
        .await
    }
    pub async fn request_server_run_cancel(&self, run_id: &str) -> PostgresStoreResult<()> {
        let run_id = run_id.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move { uow.request_server_run_cancel(&run_id).await })
        })
        .await
    }

    pub async fn apply_recorded_runtime_transition(
        &self,
        transition: RecordedRuntimeTransition,
        abort_after_writes: bool,
    ) -> PostgresStoreResult<RecordedTransitionResult> {
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.apply_recorded_runtime_transition(transition, abort_after_writes)
                    .await
            })
        })
        .await
    }

    pub async fn apply_claimed_recorded_runtime_transition(
        &self,
        handoff: RecordedWorkerHandoff,
        transition: RecordedRuntimeTransition,
        abort_after_writes: bool,
    ) -> PostgresStoreResult<RecordedTransitionResult> {
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.apply_claimed_recorded_runtime_transition(
                    handoff,
                    transition,
                    abort_after_writes,
                )
                .await
            })
        })
        .await
    }

    pub async fn apply_claimed_recorded_worker_effect(
        &self,
        handoff: RecordedWorkerHandoff,
        effect: RecordedWorkerSuccessEffect,
    ) -> PostgresStoreResult<RecordedTransitionResult> {
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.apply_claimed_recorded_worker_effect(handoff, effect)
                    .await
            })
        })
        .await
    }

    pub async fn resume_recorded_runtime_token(
        &self,
        run_id: &str,
        workflow_id: &str,
        conversation_id: &str,
        node_id: &str,
        token_id: &str,
        resume_payload: Option<Value>,
    ) -> PostgresStoreResult<RecordedTransitionResult> {
        let run_id = run_id.to_owned();
        let workflow_id = workflow_id.to_owned();
        let conversation_id = conversation_id.to_owned();
        let node_id = node_id.to_owned();
        let token_id = token_id.to_owned();
        self.transaction(|uow| {
            Box::pin(async move {
                let current = read_recorded_runtime_state(
                    &uow.transaction,
                    &uow.tables,
                    &run_id,
                    &workflow_id,
                    &conversation_id,
                )
                .await?
                .ok_or_else(|| {
                    PostgresStoreError::RecordedRuntimeConflict(format!(
                        "run {run_id:?} has no recorded runtime state"
                    ))
                })?;
                let parent = current
                    .frontier
                    .suspended
                    .iter()
                    .find(|(node, _, token, _)| node == &node_id && token == &token_id)
                    .and_then(|(_, _, _, parent)| parent.clone());
                let frontier = frontier_after_worker_resume(
                    &current.frontier,
                    &node_id,
                    &token_id,
                    parent.as_deref(),
                )?;
                let expected_event_seq =
                    latest_server_run_event_seq(&uow.transaction, &uow.tables, &run_id).await?;
                let result = uow
                    .apply_recorded_runtime_transition(
                        RecordedRuntimeTransition {
                            contract_version: RECORDED_RUNTIME_CONTRACT_VERSION,
                            transition_id: format!(
                                "resume-{run_id}-{token_id}-{expected_event_seq}"
                            ),
                            expected_event_seq,
                            kind: kogwistar_runtime::RecordedTransitionKind::ResumeResult,
                            run_id: run_id.clone(),
                            workflow_id: workflow_id.clone(),
                            conversation_id: conversation_id.clone(),
                            user_id: None,
                            user_turn_node_id: None,
                            step_seq: current.last_step_seq.saturating_add(1),
                            node_id: Some(node_id.clone()),
                            token_id: Some(token_id.clone()),
                            parent_token_id: parent.clone(),
                            initial_state: None,
                            state_update: Vec::new(),
                            update: None,
                            state_schema: Map::new(),
                            frontier: Some(frontier),
                            result: None,
                            wait_reason: None,
                            resume_payload,
                            errors: Vec::new(),
                        },
                        false,
                    )
                    .await?;
                let message_id = format!(
                    "lane|{}",
                    kogwistar_contracts::stable_id(
                        "runtime.worker.request",
                        &[
                            run_id.clone(),
                            token_id.clone(),
                            node_id.clone(),
                            result
                                .reduced
                                .state
                                .last_step_seq
                                .saturating_add(1)
                                .to_string(),
                        ],
                    )
                );
                let now = unix_epoch_millis() / 1000;
                let join_mask = result
                    .reduced
                    .state
                    .frontier
                    .pending
                    .iter()
                    .find(|(_, _, token, _)| token == &token_id)
                    .map(|(_, mask, _, _)| *mask)
                    .unwrap_or(0);
                let op = result
                    .reduced
                    .state
                    .state
                    .get("_rt_node_ops")
                    .and_then(Value::as_object)
                    .and_then(|ops| ops.get(&node_id))
                    .cloned()
                    .unwrap_or_else(|| Value::String("noop".to_owned()));
                let payload = RuntimeStepExecutePayload::from_recorded_state(
                    &result.reduced.state,
                    RuntimeStepExecuteRequest {
                        node_id: node_id.clone(),
                        op,
                        join_mask,
                        token_id: token_id.clone(),
                        parent_token_id: parent.clone(),
                        step_seq: result.reduced.state.last_step_seq.saturating_add(1),
                        expected_event_seq: result.event_seq,
                        resume_effect: result.reduced.state.resume_payload.clone(),
                    },
                );
                uow.project_lane_message(NewProjectedLaneMessage {
                    message_id,
                    namespace: "workflow".to_owned(),
                    purpose: "system".to_owned(),
                    inbox_id: "workflow-runtime".to_owned(),
                    conversation_id: conversation_id.clone(),
                    recipient_id: "python-worker".to_owned(),
                    sender_id: "rust-scheduler".to_owned(),
                    msg_type: "workflow.step.execute".to_owned(),
                    status: "pending".to_owned(),
                    created_at: now,
                    available_at: now,
                    run_id: Some(run_id.clone()),
                    step_id: Some(node_id.clone()),
                    correlation_id: Some(run_id.clone()),
                    payload_json: Some(serde_json::to_string(&payload)?),
                    error_json: None,
                })
                .await?;
                Ok(result)
            })
        })
        .await
    }

    pub async fn read_recorded_runtime_state(
        &self,
        run_id: &str,
        workflow_id: &str,
        conversation_id: &str,
    ) -> PostgresStoreResult<Option<RecordedRuntimeState>> {
        read_recorded_runtime_state(
            &**self.client().await?,
            &self.tables,
            run_id,
            workflow_id,
            conversation_id,
        )
        .await
    }
}
