use super::*;

impl SqliteStore {
    /// Open (or create) a Python-compatible EngineSQLite database.
    ///
    /// `rusqlite` uses its `bundled` feature here so native SQLite behavior is
    /// reproducible instead of depending on a host-provided SQLite library.
    pub fn open(path: impl AsRef<Path>) -> SqliteStoreResult<Self> {
        let path = path.as_ref().to_path_buf();
        let initialization_lock = database_initialization_lock(&path);
        let _initialization_guard = initialization_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let private_memory = is_private_memory_database(&path);
        if !private_memory
            && let Some(parent) = path
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)?;
        }
        let mut authority_lock = ProcessAuthorityLock::acquire(&path)?;
        if !private_memory {
            OpenOptions::new()
                .create(true)
                .truncate(false)
                .read(true)
                .write(true)
                .open(&path)?;
            authority_lock.attach_file_identity(&path)?;
        }
        let conn = configured_connection(&path)?;
        let authority_lock = Arc::new(authority_lock);
        let schema_version: i64 = conn.query_row("PRAGMA user_version", [], |row| row.get(0))?;
        if schema_version < STORE_SCHEMA_VERSION {
            initialize_schema(&conn)?;
            conn.execute_batch(&format!("PRAGMA user_version = {STORE_SCHEMA_VERSION};"))?;
        }
        Ok(Self {
            path: Arc::new(path),
            connection: Arc::new(Mutex::new(conn)),
            _authority_lock: authority_lock,
        })
    }

    pub fn path(&self) -> &Path {
        self.path.as_ref()
    }

    /// Run a short UoW under `BEGIN IMMEDIATE`. Returning an error rolls back
    /// every sequence, event, and cursor change performed through `uow`.
    pub fn immediate_transaction<T, F>(&self, operation: F) -> SqliteStoreResult<T>
    where
        F: FnOnce(&mut SqliteUnitOfWork<'_>) -> SqliteStoreResult<T>,
    {
        let mut conn = self
            .connection
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if conn.is_autocommit() && !is_private_memory_database(self.path.as_ref()) {
            // Python EngineSQLite may commit between Rust calls. Do not start
            // an immediate transaction from the cached handle, whose read
            // snapshot can predate that external commit. Keep the store mutex
            // held while refreshing so native calls remain serialized.
            let mut fresh = configured_connection(self.path.as_ref())?;
            let transaction = fresh.transaction_with_behavior(TransactionBehavior::Immediate)?;
            let result = {
                let mut uow = SqliteUnitOfWork {
                    transaction: &transaction,
                };
                operation(&mut uow)
            };
            match result {
                Ok(value) => {
                    transaction.commit()?;
                    Ok(value)
                }
                Err(error) => Err(error),
            }
        } else {
            let savepoint = conn.savepoint()?;
            let result = {
                let mut uow = SqliteUnitOfWork {
                    transaction: &savepoint,
                };
                operation(&mut uow)
            };
            match result {
                Ok(value) => {
                    savepoint.commit()?;
                    Ok(value)
                }
                Err(error) => Err(error),
            }
        }
    }

    /// Begin a Python-facade unit of work on the cached connection. Calls made
    /// before commit/rollback join it through per-operation savepoints.
    pub fn begin_external_transaction(&self) -> SqliteStoreResult<()> {
        let conn = self
            .connection
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if !conn.is_autocommit() {
            return Err(SqliteStoreError::TransactionAborted(
                "SQLite external transaction is already active".to_owned(),
            ));
        }
        conn.execute_batch("BEGIN IMMEDIATE")?;
        sqlite_lifecycle_diagnostic("begin", self.path.as_ref());
        Ok(())
    }

    pub fn commit_external_transaction(&self) -> SqliteStoreResult<()> {
        let conn = self
            .connection
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if conn.is_autocommit() {
            return Err(SqliteStoreError::TransactionAborted(
                "SQLite external transaction is not active".to_owned(),
            ));
        }
        conn.execute_batch("COMMIT")?;
        sqlite_lifecycle_diagnostic("commit", self.path.as_ref());
        Ok(())
    }

    pub fn rollback_external_transaction(&self) -> SqliteStoreResult<()> {
        let conn = self
            .connection
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if conn.is_autocommit() {
            return Err(SqliteStoreError::TransactionAborted(
                "SQLite external transaction is not active".to_owned(),
            ));
        }
        conn.execute_batch("ROLLBACK")?;
        sqlite_lifecycle_diagnostic("rollback", self.path.as_ref());
        Ok(())
    }

    /// Alias for `immediate_transaction`; all store UoWs are immediate.
    pub fn transaction<T, F>(&self, operation: F) -> SqliteStoreResult<T>
    where
        F: FnOnce(&mut SqliteUnitOfWork<'_>) -> SqliteStoreResult<T>,
    {
        self.immediate_transaction(operation)
    }

    pub fn next_global_seq(&self) -> SqliteStoreResult<i64> {
        self.immediate_transaction(|uow| uow.next_global_seq())
    }

    pub fn current_global_seq(&self) -> SqliteStoreResult<i64> {
        self.with_connection(current_global_seq)
    }

    pub fn next_user_seq(&self, user_id: &str) -> SqliteStoreResult<i64> {
        self.immediate_transaction(|uow| uow.next_user_seq(user_id))
    }

    pub fn current_user_seq(&self, user_id: &str) -> SqliteStoreResult<i64> {
        self.with_connection(|conn| current_user_seq(conn, user_id))
    }

    pub fn set_user_seq(&self, user_id: &str, value: i64) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| uow.set_user_seq(user_id, value))
    }

    /// Python-compatible alias for `next_user_seq`.
    pub fn next_scoped_seq(&self, scope_id: &str) -> SqliteStoreResult<i64> {
        self.next_user_seq(scope_id)
    }

    /// Python-compatible alias for `current_user_seq`.
    pub fn current_scoped_seq(&self, scope_id: &str) -> SqliteStoreResult<i64> {
        self.current_user_seq(scope_id)
    }

    /// Python-compatible alias for `set_user_seq`.
    pub fn set_scoped_seq(&self, scope_id: &str, value: i64) -> SqliteStoreResult<()> {
        self.set_user_seq(scope_id, value)
    }

    /// Allocate a namespace-local event sequence without appending an event.
    pub fn alloc_event_seq(&self, namespace: &str) -> SqliteStoreResult<i64> {
        self.immediate_transaction(|uow| uow.alloc_event_seq(namespace))
    }

    pub fn append_raw_entity_event(
        &self,
        namespace: &str,
        event: NewRawEntityEvent,
    ) -> SqliteStoreResult<AppendedRawEvent> {
        self.immediate_transaction(|uow| uow.append_raw_entity_event(namespace, event))
    }

    pub fn append_restored_raw_entity_event(
        &self,
        namespace: &str,
        event: RestoredRawEntityEvent,
    ) -> SqliteStoreResult<AppendedRawEvent> {
        self.immediate_transaction(|uow| uow.append_restored_raw_entity_event(namespace, event))
    }

    /// Raw replay is exclusive of `after_seq`, preserving stored JSON text.
    pub fn replay_raw_events(
        &self,
        namespace: &str,
        after_seq: i64,
        limit: usize,
    ) -> SqliteStoreResult<Vec<RawEntityEvent>> {
        self.with_connection(|conn| replay_raw_events(conn, namespace, after_seq, limit))
    }

    pub fn latest_retained_event_seq(&self, namespace: &str) -> SqliteStoreResult<i64> {
        self.with_connection(|conn| latest_retained_event_seq(conn, namespace))
    }

    /// Delete namespace events whose sequence is strictly above `to_seq`.
    pub fn prune_entity_events_after(
        &self,
        namespace: &str,
        to_seq: i64,
    ) -> SqliteStoreResult<u64> {
        self.immediate_transaction(|uow| uow.prune_entity_events_after(namespace, to_seq))
    }

    /// Strict cursor advance: bounded by retained events and monotonic.
    pub fn strict_advance_replay_cursor(
        &self,
        namespace: &str,
        consumer: &str,
        last_seq: i64,
    ) -> SqliteStoreResult<ReplayCursor> {
        self.immediate_transaction(|uow| {
            uow.strict_advance_replay_cursor(namespace, consumer, last_seq)
        })
    }

    /// Python `cursor_set` semantics: direct overwrite, including regression or
    /// a value beyond retained events. This is intentionally separate from the
    /// strict store trait for a future permissive Python adapter.
    pub fn set_replay_cursor_legacy(
        &self,
        namespace: &str,
        consumer: &str,
        last_seq: i64,
    ) -> SqliteStoreResult<ReplayCursor> {
        self.immediate_transaction(|uow| {
            uow.set_replay_cursor_legacy(namespace, consumer, last_seq)
        })
    }

    pub fn replay_cursor(
        &self,
        namespace: &str,
        consumer: &str,
    ) -> SqliteStoreResult<ReplayCursor> {
        self.with_connection(|conn| replay_cursor(conn, namespace, consumer))
    }

    pub fn index_applied_fingerprint(
        &self,
        namespace: &str,
        coalesce_key: &str,
    ) -> SqliteStoreResult<Option<String>> {
        self.with_connection(|conn| {
            conn.query_row(
                "SELECT applied_fingerprint FROM index_applied_state WHERE namespace=?1 AND coalesce_key=?2",
                params![namespace, coalesce_key],
                |row| row.get(0),
            )
            .optional()
            .map_err(Into::into)
        })
    }

    pub fn set_index_applied_fingerprint(
        &self,
        namespace: &str,
        coalesce_key: &str,
        applied_fingerprint: Option<&str>,
        last_job_id: Option<&str>,
    ) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| {
            uow.set_index_applied_fingerprint(
                namespace,
                coalesce_key,
                applied_fingerprint,
                last_job_id,
            )
        })
    }

    /// Bounded operator-triggered recovery. Event history remains authoritative;
    /// projection replacement and durable cursor advance share one SQLite UoW.
    pub fn recover_entity_projection(
        &self,
        request: EntityRecoveryRequest,
    ) -> SqliteStoreResult<EntityRecoveryReport> {
        validate_entity_recovery_request(&request)?;
        self.immediate_transaction(|uow| uow.recover_entity_projection(&request))
    }

    /// Explicit full rebuild. Ordinary store opening never calls this method.
    pub fn rebuild_entity_projection(
        &self,
        request: EntityRebuildRequest,
    ) -> SqliteStoreResult<EntityRecoveryReport> {
        validate_entity_rebuild_request(&request)?;
        self.immediate_transaction(|uow| uow.rebuild_entity_projection(&request))
    }

    pub fn get_named_projection(
        &self,
        namespace: &str,
        key: &str,
    ) -> SqliteStoreResult<Option<NamedProjection>> {
        self.with_connection(|conn| named_projection(conn, namespace, key))
    }

    pub fn list_named_projections(
        &self,
        namespace: &str,
    ) -> SqliteStoreResult<Vec<NamedProjection>> {
        self.with_connection(|conn| named_projections(conn, namespace))
    }

    pub fn get_stage1_node_projection(
        &self,
        namespace: &str,
        key: &str,
    ) -> SqliteStoreResult<Option<NamedProjection>> {
        self.with_connection(|conn| stage1_node_projection(conn, namespace, key))
    }

    pub fn list_stage1_node_projections(
        &self,
        namespace: &str,
    ) -> SqliteStoreResult<Vec<NamedProjection>> {
        self.with_connection(|conn| stage1_node_projections(conn, namespace))
    }

    pub fn replace_stage1_node_projection(
        &self,
        namespace: &str,
        key: &str,
        projection: NamedProjectionWrite,
    ) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| {
            uow.replace_stage1_node_projection(namespace, key, projection)
        })
    }

    pub fn clear_stage1_node_projection(
        &self,
        namespace: &str,
        key: &str,
    ) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| uow.clear_stage1_node_projection(namespace, key))
    }

    pub fn replace_named_projection(
        &self,
        namespace: &str,
        key: &str,
        projection: NamedProjectionWrite,
    ) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| uow.replace_named_projection(namespace, key, projection))
    }

    pub fn compare_and_swap_named_projection(
        &self,
        namespace: &str,
        key: &str,
        expected_last_authoritative_seq: Option<i64>,
        expected_last_materialized_seq: Option<i64>,
        projection: NamedProjectionWrite,
    ) -> SqliteStoreResult<bool> {
        self.immediate_transaction(|uow| {
            uow.compare_and_swap_named_projection(
                namespace,
                key,
                expected_last_authoritative_seq,
                expected_last_materialized_seq,
                projection,
            )
        })
    }

    pub fn compare_and_swap_named_projections(
        &self,
        updates: Vec<NamedProjectionCas>,
    ) -> SqliteStoreResult<bool> {
        self.immediate_transaction(|uow| uow.compare_and_swap_named_projections(updates))
    }

    pub fn clear_named_projection(&self, namespace: &str, key: &str) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| uow.clear_named_projection(namespace, key))
    }

    pub fn clear_projection_namespace(&self, namespace: &str) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| uow.clear_projection_namespace(namespace))
    }

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

    pub fn enqueue_index_job(&self, job: NewIndexJob) -> SqliteStoreResult<String> {
        self.immediate_transaction(|uow| uow.enqueue_index_job(job))
    }
    pub fn claim_index_jobs(
        &self,
        limit: usize,
        lease_seconds: i64,
        namespace: Option<&str>,
    ) -> SqliteStoreResult<Vec<IndexJob>> {
        self.immediate_transaction(|uow| uow.claim_index_jobs(limit, lease_seconds, namespace))
    }
    pub fn mark_index_job_done(
        &self,
        job_id: &str,
        claim_token: Option<&str>,
    ) -> SqliteStoreResult<bool> {
        self.immediate_transaction(|uow| uow.mark_index_job_done(job_id, claim_token))
    }
    pub fn accept_index_job_result(
        &self,
        job_id: &str,
        claim_token: &str,
        result_json: &str,
        result_sha256: &str,
    ) -> SqliteStoreResult<AcceptedIndexJobResult> {
        self.immediate_transaction(|uow| {
            uow.accept_index_job_result(job_id, claim_token, result_json, result_sha256)
        })
    }
    pub fn index_job_result(
        &self,
        job_id: &str,
    ) -> SqliteStoreResult<Option<AcceptedIndexJobResult>> {
        self.with_connection(|conn| index_job_result(conn, job_id))
    }
    pub fn mark_index_job_failed(
        &self,
        job_id: &str,
        error: &str,
        final_: bool,
        claim_token: Option<&str>,
    ) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| {
            uow.mark_index_job_failed(job_id, error, final_, claim_token)
        })
    }
    pub fn bump_retry_and_requeue(
        &self,
        job_id: &str,
        error: &str,
        delay_seconds: i64,
        claim_token: Option<&str>,
    ) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| {
            uow.bump_retry_and_requeue(job_id, error, delay_seconds, claim_token)
        })
    }
    pub fn renew_index_job_lease(
        &self,
        job_id: &str,
        claim_token: &str,
        lease_seconds: i64,
    ) -> SqliteStoreResult<bool> {
        self.immediate_transaction(|uow| {
            uow.renew_index_job_lease(job_id, claim_token, lease_seconds)
        })
    }
    pub fn requeue_index_job_at_tail(
        &self,
        job_id: &str,
        payload_json: String,
        delay_seconds: i64,
        claim_token: Option<&str>,
    ) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| {
            uow.requeue_index_job_at_tail(job_id, payload_json, delay_seconds, claim_token)
        })
    }
    pub fn list_index_jobs(
        &self,
        namespace: Option<&str>,
        status: Option<&str>,
        entity_kind: Option<&str>,
        entity_id: Option<&str>,
        index_kind: Option<&str>,
        limit: usize,
    ) -> SqliteStoreResult<Vec<IndexJob>> {
        self.with_connection(|conn| {
            list_index_jobs(
                conn,
                namespace,
                status,
                entity_kind,
                entity_id,
                index_kind,
                limit,
            )
        })
    }

    pub fn project_lane_message(&self, row: NewProjectedLaneMessage) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| uow.project_lane_message(row))
    }
    pub fn get_projected_lane_message(
        &self,
        message_id: &str,
    ) -> SqliteStoreResult<Option<ProjectedLaneMessage>> {
        self.with_connection(|conn| projected_lane_message(conn, message_id))
    }
    pub fn list_projected_lane_messages(
        &self,
        filter: LaneMessageFilter,
    ) -> SqliteStoreResult<Vec<ProjectedLaneMessage>> {
        self.with_connection(|conn| list_projected_lane_messages(conn, &filter))
    }
    pub fn update_projected_lane_message_status(
        &self,
        message_id: &str,
        status: &str,
        error_json: Option<String>,
    ) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| {
            uow.update_projected_lane_message_status(message_id, status, error_json)
        })
    }
    pub fn update_projected_lane_message_links(
        &self,
        message_id: &str,
        prev: Option<String>,
        next: Option<String>,
        inbox_tail: Option<String>,
        conversation_tail: Option<String>,
    ) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| {
            uow.update_projected_lane_message_links(
                message_id,
                prev,
                next,
                inbox_tail,
                conversation_tail,
            )
        })
    }
    pub fn clear_projected_lane_messages(&self, namespace: &str) -> SqliteStoreResult<u64> {
        self.immediate_transaction(|uow| uow.clear_projected_lane_messages(namespace))
    }
    pub fn claim_projected_lane_messages(
        &self,
        namespace: &str,
        inbox_id: &str,
        claimed_by: &str,
        limit: usize,
        lease_seconds: i64,
    ) -> SqliteStoreResult<Vec<ProjectedLaneMessage>> {
        self.immediate_transaction(|uow| {
            uow.claim_projected_lane_messages(namespace, inbox_id, claimed_by, limit, lease_seconds)
        })
    }
    pub fn claim_projected_lane_messages_filtered(
        &self,
        filter: LaneMessageClaimFilter,
    ) -> SqliteStoreResult<Vec<ProjectedLaneMessage>> {
        self.immediate_transaction(|uow| uow.claim_projected_lane_messages_filtered(filter))
    }
    pub fn ack_projected_lane_message(
        &self,
        message_id: &str,
        claimed_by: &str,
    ) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| uow.ack_projected_lane_message(message_id, claimed_by))
    }
    pub fn requeue_projected_lane_message(
        &self,
        message_id: &str,
        claimed_by: &str,
        error_json: Option<String>,
        delay_seconds: i64,
    ) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| {
            uow.requeue_projected_lane_message(message_id, claimed_by, error_json, delay_seconds)
        })
    }
    pub fn dead_letter_projected_lane_message(
        &self,
        message_id: &str,
        claimed_by: &str,
        error_json: Option<String>,
    ) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| {
            uow.dead_letter_projected_lane_message(message_id, claimed_by, error_json)
        })
    }
    pub fn repair_orphaned_claimed_lane_messages(
        &self,
        namespace: &str,
        inbox_id: Option<&str>,
        limit: usize,
    ) -> SqliteStoreResult<Vec<String>> {
        self.immediate_transaction(|uow| {
            uow.repair_orphaned_claimed_lane_messages(namespace, inbox_id, limit)
        })
    }

    pub(super) fn with_connection<T>(
        &self,
        operation: impl FnOnce(&Connection) -> SqliteStoreResult<T>,
    ) -> SqliteStoreResult<T> {
        let conn = self
            .connection
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if conn.is_autocommit() && !is_private_memory_database(self.path.as_ref()) {
            drop(conn);
            // A Python EngineSQLite connection may commit between native
            // calls. A fresh read handle prevents an autocommit snapshot from
            // hiding those external commits while transaction reads still
            // use the Rust-owned connection above.
            let fresh = configured_connection(self.path.as_ref())?;
            operation(&fresh)
        } else {
            operation(&conn)
        }
    }
}
