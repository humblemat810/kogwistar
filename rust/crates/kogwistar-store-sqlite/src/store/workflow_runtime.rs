use super::super::*;

impl SqliteStore {
    pub fn put_workflow_design_snapshot(
        &self,
        workflow_id: &str,
        snapshot: WorkflowDesignSnapshotWrite,
    ) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| uow.put_workflow_design_snapshot(workflow_id, snapshot))
    }

    pub fn get_workflow_design_snapshot(
        &self,
        workflow_id: &str,
        max_version: i64,
        schema_version: i64,
    ) -> SqliteStoreResult<Option<WorkflowDesignSnapshot>> {
        self.with_connection(|conn| {
            workflow_design_snapshot(conn, workflow_id, max_version, schema_version)
        })
    }

    pub fn clear_workflow_design_snapshots(&self, workflow_id: &str) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| uow.clear_workflow_design_snapshots(workflow_id))
    }

    pub fn put_workflow_design_delta(
        &self,
        workflow_id: &str,
        delta: WorkflowDesignDeltaWrite,
    ) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| uow.put_workflow_design_delta(workflow_id, delta))
    }

    pub fn get_workflow_design_delta(
        &self,
        workflow_id: &str,
        version: i64,
        schema_version: i64,
    ) -> SqliteStoreResult<Option<WorkflowDesignDelta>> {
        self.with_connection(|conn| {
            workflow_design_delta(conn, workflow_id, version, schema_version)
        })
    }

    pub fn clear_workflow_design_deltas(&self, workflow_id: &str) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| uow.clear_workflow_design_deltas(workflow_id))
    }

    pub fn create_server_run(&self, run: ServerRunCreate) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| uow.create_server_run(run))
    }
    pub fn get_server_run(&self, run_id: &str) -> SqliteStoreResult<Option<ServerRun>> {
        self.with_connection(|conn| server_run(conn, run_id))
    }
    pub fn list_server_runs(
        &self,
        status: Option<&str>,
        workflow_id: Option<&str>,
        conversation_id: Option<&str>,
        limit: usize,
    ) -> SqliteStoreResult<Vec<ServerRun>> {
        self.with_connection(|conn| server_runs(conn, status, workflow_id, conversation_id, limit))
    }
    pub fn append_server_run_event(
        &self,
        run_id: &str,
        event_type: &str,
        payload_json: String,
    ) -> SqliteStoreResult<ServerRunEvent> {
        self.immediate_transaction(|uow| {
            uow.append_server_run_event(run_id, event_type, payload_json)
        })
    }
    pub fn list_server_run_events(
        &self,
        run_id: &str,
        after_seq: i64,
        limit: usize,
    ) -> SqliteStoreResult<Vec<ServerRunEvent>> {
        self.with_connection(|conn| server_run_events(conn, run_id, after_seq, limit))
    }
    pub fn update_server_run(
        &self,
        run_id: &str,
        update: ServerRunUpdate,
    ) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| uow.update_server_run(run_id, update))
    }
    pub fn request_server_run_cancel(&self, run_id: &str) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| uow.request_server_run_cancel(run_id))
    }

    /// Apply one already-recorded runtime result under one `BEGIN IMMEDIATE`
    /// transaction.  This operation never invokes a resolver, provider, tool,
    /// lane, graph, or Python callback.
    pub fn apply_recorded_runtime_transition(
        &self,
        transition: RecordedRuntimeTransition,
        abort_after_writes: bool,
    ) -> SqliteStoreResult<RecordedTransitionResult> {
        self.immediate_transaction(|uow| {
            uow.apply_recorded_runtime_transition(transition, abort_after_writes)
        })
    }

    /// Atomically apply a recorded worker result and acknowledge the exact
    /// claimed lane request.  No callback or dispatch runs inside this UoW.
    pub fn apply_claimed_recorded_runtime_transition(
        &self,
        handoff: RecordedWorkerHandoff,
        transition: RecordedRuntimeTransition,
        abort_after_writes: bool,
    ) -> SqliteStoreResult<RecordedTransitionResult> {
        self.immediate_transaction(|uow| {
            uow.apply_claimed_recorded_runtime_transition(handoff, transition, abort_after_writes)
        })
    }

    pub fn apply_claimed_recorded_worker_effect(
        &self,
        handoff: RecordedWorkerHandoff,
        effect: RecordedWorkerSuccessEffect,
    ) -> SqliteStoreResult<RecordedTransitionResult> {
        self.immediate_transaction(|uow| uow.apply_claimed_recorded_worker_effect(handoff, effect))
    }

    pub fn resume_recorded_runtime_token(
        &self,
        run_id: &str,
        workflow_id: &str,
        conversation_id: &str,
        node_id: &str,
        token_id: &str,
        resume_payload: Option<Value>,
    ) -> SqliteStoreResult<RecordedTransitionResult> {
        let run_id = run_id.to_owned();
        let workflow_id = workflow_id.to_owned();
        let conversation_id = conversation_id.to_owned();
        let node_id = node_id.to_owned();
        let token_id = token_id.to_owned();
        self.immediate_transaction(|uow| {
            let current = read_recorded_runtime_state(
                uow.transaction,
                &run_id,
                &workflow_id,
                &conversation_id,
            )?
            .ok_or_else(|| {
                SqliteStoreError::RecordedRuntimeConflict(format!(
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
            let expected_event_seq = latest_server_run_event_seq(uow.transaction, &run_id)?;
            let result = uow.apply_recorded_runtime_transition(
                RecordedRuntimeTransition {
                    contract_version: RECORDED_RUNTIME_CONTRACT_VERSION,
                    transition_id: format!("resume-{run_id}-{token_id}-{expected_event_seq}"),
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
            )?;
            let message_id = format!(
                "lane|{}",
                kogwistar_contracts::stable_id(
                    "runtime.worker.request",
                    &[
                        result.reduced.state.run_id.clone(),
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
            let now = unix_epoch_seconds();
            let op = result
                .reduced
                .state
                .state
                .get("_rt_node_ops")
                .and_then(Value::as_object)
                .and_then(|ops| ops.get(&node_id))
                .cloned()
                .unwrap_or_else(|| Value::String("noop".to_owned()));
            let join_mask = result
                .reduced
                .state
                .frontier
                .pending
                .iter()
                .find(|(_, _, token, _)| token == &token_id)
                .map(|(_, mask, _, _)| *mask)
                .unwrap_or(0);
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
            })?;
            Ok(result)
        })
    }

    /// Reopen/read path for recorded runtime recovery.  Pending tokens are
    /// already normalized to Python's pending-plus-inflight recovery frontier;
    /// suspended tokens remain separately parked in `frontier.suspended`.
    pub fn read_recorded_runtime_state(
        &self,
        run_id: &str,
        workflow_id: &str,
        conversation_id: &str,
    ) -> SqliteStoreResult<Option<RecordedRuntimeState>> {
        self.with_connection(|conn| {
            read_recorded_runtime_state(conn, run_id, workflow_id, conversation_id)
        })
    }
}
