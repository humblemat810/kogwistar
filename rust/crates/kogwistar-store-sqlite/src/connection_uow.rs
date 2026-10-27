use super::*;

pub(super) fn configured_connection(path: &Path) -> SqliteStoreResult<Connection> {
    let conn = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE,
    )?;
    conn.busy_timeout(std::time::Duration::from_millis(BUSY_TIMEOUT_MS))?;
    conn.execute_batch("PRAGMA foreign_keys = ON; PRAGMA synchronous = NORMAL;")?;
    let mut journal_mode: String = conn.query_row("PRAGMA journal_mode", [], |row| row.get(0))?;
    if !journal_mode.eq_ignore_ascii_case("wal") {
        conn.execute_batch("PRAGMA journal_mode = WAL;")?;
        journal_mode = conn.query_row("PRAGMA journal_mode", [], |row| row.get(0))?;
    }
    if std::env::var_os("KOGWISTAR_SQLITE_DIAGNOSTICS").as_deref()
        == Some(std::ffi::OsStr::new("1"))
    {
        let version = unsafe {
            CStr::from_ptr(rusqlite::ffi::sqlite3_libversion())
                .to_string_lossy()
                .into_owned()
        };
        let source_id = unsafe {
            CStr::from_ptr(rusqlite::ffi::sqlite3_sourceid())
                .to_string_lossy()
                .into_owned()
        };
        eprintln!(
            "[sqlite-rust-open] conn={:p} path={} version={} source_id={} journal_mode={} {}",
            unsafe { conn.handle() },
            path.display(),
            version,
            source_id,
            journal_mode,
            sqlite_sidecars(path)
        );
    }
    Ok(conn)
}

pub(super) fn sqlite_lifecycle_diagnostic(event: &str, path: &Path) {
    if std::env::var_os("KOGWISTAR_SQLITE_DIAGNOSTICS").as_deref()
        != Some(std::ffi::OsStr::new("1"))
    {
        return;
    }
    eprintln!(
        "[sqlite-rust-{event}] path={} {}",
        path.display(),
        sqlite_sidecars(path)
    );
}

pub(super) fn sqlite_sidecars(path: &Path) -> String {
    [
        path.to_path_buf(),
        PathBuf::from(format!("{}-wal", path.display())),
        PathBuf::from(format!("{}-shm", path.display())),
    ]
    .iter()
    .map(|candidate| match fs::metadata(candidate) {
        Ok(metadata) => format!(
            "{}=size:{}",
            candidate.file_name().unwrap_or_default().to_string_lossy(),
            metadata.len()
        ),
        Err(_) => format!(
            "{}=absent",
            candidate.file_name().unwrap_or_default().to_string_lossy()
        ),
    })
    .collect::<Vec<_>>()
    .join(" ")
}

/// Operations sharing one `BEGIN IMMEDIATE` transaction.
pub struct SqliteUnitOfWork<'connection> {
    pub(super) transaction: &'connection Connection,
}

impl SqliteUnitOfWork<'_> {
    pub fn set_index_applied_fingerprint(
        &mut self,
        namespace: &str,
        coalesce_key: &str,
        applied_fingerprint: Option<&str>,
        last_job_id: Option<&str>,
    ) -> SqliteStoreResult<()> {
        self.transaction.execute(
            "INSERT INTO index_applied_state(namespace,coalesce_key,applied_fingerprint,applied_at,last_job_id) VALUES(?1,?2,?3,?4,?5) ON CONFLICT(namespace,coalesce_key) DO UPDATE SET applied_fingerprint=excluded.applied_fingerprint,applied_at=excluded.applied_at,last_job_id=excluded.last_job_id",
            params![namespace, coalesce_key, applied_fingerprint, unix_epoch_seconds(), last_job_id],
        )?;
        Ok(())
    }

    pub fn project_lane_message(&mut self, row: NewProjectedLaneMessage) -> SqliteStoreResult<()> {
        project_lane_message(self.transaction, row)
    }
    pub fn get_projected_lane_message(
        &self,
        message_id: &str,
    ) -> SqliteStoreResult<Option<ProjectedLaneMessage>> {
        projected_lane_message(self.transaction, message_id)
    }
    pub fn get_server_run(&self, run_id: &str) -> SqliteStoreResult<Option<ServerRun>> {
        server_run(self.transaction, run_id)
    }
    pub fn list_projected_lane_messages(
        &self,
        filter: &LaneMessageFilter,
    ) -> SqliteStoreResult<Vec<ProjectedLaneMessage>> {
        list_projected_lane_messages(self.transaction, filter)
    }
    pub fn update_projected_lane_message_status(
        &mut self,
        message_id: &str,
        status: &str,
        error_json: Option<String>,
    ) -> SqliteStoreResult<()> {
        update_projected_lane_message_status(self.transaction, message_id, status, error_json)
    }
    pub fn update_projected_lane_message_links(
        &mut self,
        message_id: &str,
        prev: Option<String>,
        next: Option<String>,
        inbox_tail: Option<String>,
        conversation_tail: Option<String>,
    ) -> SqliteStoreResult<()> {
        update_projected_lane_message_links(
            self.transaction,
            message_id,
            prev,
            next,
            inbox_tail,
            conversation_tail,
        )
    }
    pub fn claim_projected_lane_messages_for_run(
        &mut self,
        namespace: &str,
        inbox_id: &str,
        run_id: &str,
        claimed_by: &str,
        limit: usize,
        lease_seconds: i64,
    ) -> SqliteStoreResult<Vec<ProjectedLaneMessage>> {
        claim_projected_lane_messages_for_run(
            self.transaction,
            namespace,
            inbox_id,
            run_id,
            claimed_by,
            limit,
            lease_seconds,
        )
    }
    pub fn update_projected_lane_message_payload(
        &mut self,
        message_id: &str,
        payload_json: String,
    ) -> SqliteStoreResult<()> {
        let changed = self.transaction.execute(
            "UPDATE projected_lane_messages SET payload_json=?1 WHERE message_id=?2",
            params![payload_json, message_id],
        )?;
        if changed != 1 {
            return Err(SqliteStoreError::RecordedRuntimeConflict(format!(
                "lane message {message_id:?} payload update matched {changed} rows"
            )));
        }
        Ok(())
    }
    pub fn clear_projected_lane_messages(&mut self, namespace: &str) -> SqliteStoreResult<u64> {
        clear_projected_lane_messages(self.transaction, namespace)
    }
    pub fn claim_projected_lane_messages(
        &mut self,
        namespace: &str,
        inbox_id: &str,
        claimed_by: &str,
        limit: usize,
        lease_seconds: i64,
    ) -> SqliteStoreResult<Vec<ProjectedLaneMessage>> {
        claim_projected_lane_messages(
            self.transaction,
            namespace,
            inbox_id,
            claimed_by,
            limit,
            lease_seconds,
        )
    }
    pub fn claim_projected_lane_messages_filtered(
        &mut self,
        filter: LaneMessageClaimFilter,
    ) -> SqliteStoreResult<Vec<ProjectedLaneMessage>> {
        claim_projected_lane_messages_filtered(self.transaction, &filter)
    }
    pub fn ack_projected_lane_message(
        &mut self,
        message_id: &str,
        claimed_by: &str,
    ) -> SqliteStoreResult<()> {
        ack_projected_lane_message(self.transaction, message_id, claimed_by)
    }
    pub fn requeue_projected_lane_message(
        &mut self,
        message_id: &str,
        claimed_by: &str,
        error_json: Option<String>,
        delay_seconds: i64,
    ) -> SqliteStoreResult<()> {
        requeue_projected_lane_message(
            self.transaction,
            message_id,
            claimed_by,
            error_json,
            delay_seconds,
        )
    }
    pub fn dead_letter_projected_lane_message(
        &mut self,
        message_id: &str,
        claimed_by: &str,
        error_json: Option<String>,
    ) -> SqliteStoreResult<()> {
        dead_letter_projected_lane_message(self.transaction, message_id, claimed_by, error_json)
    }
    pub fn repair_orphaned_claimed_lane_messages(
        &mut self,
        namespace: &str,
        inbox_id: Option<&str>,
        limit: usize,
    ) -> SqliteStoreResult<Vec<String>> {
        repair_orphaned_claimed_lane_messages(self.transaction, namespace, inbox_id, limit)
    }
    pub fn enqueue_index_job(&mut self, job: NewIndexJob) -> SqliteStoreResult<String> {
        enqueue_index_job(self.transaction, job)
    }
    pub fn claim_index_jobs(
        &mut self,
        limit: usize,
        lease_seconds: i64,
        namespace: Option<&str>,
    ) -> SqliteStoreResult<Vec<IndexJob>> {
        claim_index_jobs(self.transaction, limit, lease_seconds, namespace)
    }
    pub fn mark_index_job_done(
        &mut self,
        job_id: &str,
        claim_token: Option<&str>,
    ) -> SqliteStoreResult<bool> {
        mark_index_job_done(self.transaction, job_id, claim_token)
    }
    pub fn accept_index_job_result(
        &mut self,
        job_id: &str,
        claim_token: &str,
        result_json: &str,
        result_sha256: &str,
    ) -> SqliteStoreResult<AcceptedIndexJobResult> {
        accept_index_job_result(
            self.transaction,
            job_id,
            claim_token,
            result_json,
            result_sha256,
        )
    }
    pub fn mark_index_job_failed(
        &mut self,
        job_id: &str,
        error: &str,
        final_: bool,
        claim_token: Option<&str>,
    ) -> SqliteStoreResult<()> {
        mark_index_job_failed(self.transaction, job_id, error, final_, claim_token)
    }
    pub fn bump_retry_and_requeue(
        &mut self,
        job_id: &str,
        error: &str,
        delay_seconds: i64,
        claim_token: Option<&str>,
    ) -> SqliteStoreResult<()> {
        bump_retry_and_requeue(self.transaction, job_id, error, delay_seconds, claim_token)
    }
    pub fn renew_index_job_lease(
        &mut self,
        job_id: &str,
        claim_token: &str,
        lease_seconds: i64,
    ) -> SqliteStoreResult<bool> {
        renew_index_job_lease(self.transaction, job_id, claim_token, lease_seconds)
    }
    pub fn requeue_index_job_at_tail(
        &mut self,
        job_id: &str,
        payload_json: String,
        delay_seconds: i64,
        claim_token: Option<&str>,
    ) -> SqliteStoreResult<()> {
        requeue_index_job_at_tail(
            self.transaction,
            job_id,
            payload_json,
            delay_seconds,
            claim_token,
        )
    }
    pub fn next_global_seq(&mut self) -> SqliteStoreResult<i64> {
        self.transaction
            .query_row(
                "UPDATE global_seq SET value = value + 1 RETURNING value",
                [],
                |row| row.get(0),
            )
            .map_err(Into::into)
    }

    pub fn current_global_seq(&self) -> SqliteStoreResult<i64> {
        current_global_seq(self.transaction)
    }

    pub fn next_user_seq(&mut self, user_id: &str) -> SqliteStoreResult<i64> {
        self.transaction
            .query_row(
                "INSERT INTO user_seq(user_id, value) VALUES (?1, 1) \
                 ON CONFLICT(user_id) DO UPDATE SET value = user_seq.value + 1 \
                 RETURNING value",
                [user_id],
                |row| row.get(0),
            )
            .map_err(Into::into)
    }

    pub fn current_user_seq(&self, user_id: &str) -> SqliteStoreResult<i64> {
        current_user_seq(self.transaction, user_id)
    }

    pub fn set_user_seq(&mut self, user_id: &str, value: i64) -> SqliteStoreResult<()> {
        if value < 0 {
            return Err(SqliteStoreError::NegativeSequenceValue { value });
        }
        self.transaction.execute(
            "INSERT INTO user_seq(user_id, value) VALUES (?1, ?2) \
             ON CONFLICT(user_id) DO UPDATE SET value = excluded.value",
            params![user_id, value],
        )?;
        Ok(())
    }

    pub fn next_scoped_seq(&mut self, scope_id: &str) -> SqliteStoreResult<i64> {
        self.next_user_seq(scope_id)
    }

    pub fn current_scoped_seq(&self, scope_id: &str) -> SqliteStoreResult<i64> {
        self.current_user_seq(scope_id)
    }

    pub fn set_scoped_seq(&mut self, scope_id: &str, value: i64) -> SqliteStoreResult<()> {
        self.set_user_seq(scope_id, value)
    }

    pub fn alloc_event_seq(&mut self, namespace: &str) -> SqliteStoreResult<i64> {
        self.transaction
            .query_row(
                "INSERT INTO namespace_seq(namespace, next_seq) VALUES (?1, 2) \
                 ON CONFLICT(namespace) DO UPDATE SET next_seq = namespace_seq.next_seq + 1 \
                 RETURNING next_seq - 1",
                [namespace],
                |row| row.get(0),
            )
            .map_err(Into::into)
    }

    pub fn append_raw_entity_event(
        &mut self,
        namespace: &str,
        event: NewRawEntityEvent,
    ) -> SqliteStoreResult<AppendedRawEvent> {
        append_raw_entity_event(self.transaction, namespace, event)
    }

    pub fn append_restored_raw_entity_event(
        &mut self,
        namespace: &str,
        event: RestoredRawEntityEvent,
    ) -> SqliteStoreResult<AppendedRawEvent> {
        append_restored_raw_entity_event(self.transaction, namespace, event)
    }

    pub fn replay_raw_events(
        &self,
        namespace: &str,
        after_seq: i64,
        limit: usize,
    ) -> SqliteStoreResult<Vec<RawEntityEvent>> {
        replay_raw_events(self.transaction, namespace, after_seq, limit)
    }

    pub fn latest_retained_event_seq(&self, namespace: &str) -> SqliteStoreResult<i64> {
        latest_retained_event_seq(self.transaction, namespace)
    }

    pub fn prune_entity_events_after(
        &mut self,
        namespace: &str,
        to_seq: i64,
    ) -> SqliteStoreResult<u64> {
        prune_entity_events_after(self.transaction, namespace, to_seq)
    }

    pub fn replay_cursor(
        &self,
        namespace: &str,
        consumer: &str,
    ) -> SqliteStoreResult<ReplayCursor> {
        replay_cursor(self.transaction, namespace, consumer)
    }

    pub fn recover_entity_projection(
        &mut self,
        request: &EntityRecoveryRequest,
    ) -> SqliteStoreResult<EntityRecoveryReport> {
        recover_entity_projection_uow(self, request)
    }

    pub fn rebuild_entity_projection(
        &mut self,
        request: &EntityRebuildRequest,
    ) -> SqliteStoreResult<EntityRecoveryReport> {
        rebuild_entity_projection_uow(self, request)
    }

    pub fn strict_advance_replay_cursor(
        &mut self,
        namespace: &str,
        consumer: &str,
        last_seq: i64,
    ) -> SqliteStoreResult<ReplayCursor> {
        strict_advance_replay_cursor(self.transaction, namespace, consumer, last_seq)
    }

    pub fn set_replay_cursor_legacy(
        &mut self,
        namespace: &str,
        consumer: &str,
        last_seq: i64,
    ) -> SqliteStoreResult<ReplayCursor> {
        set_replay_cursor_legacy(self.transaction, namespace, consumer, last_seq)
    }

    pub fn replace_named_projection(
        &mut self,
        namespace: &str,
        key: &str,
        projection: NamedProjectionWrite,
    ) -> SqliteStoreResult<()> {
        replace_named_projection(self.transaction, namespace, key, projection)
    }

    pub fn replace_stage1_node_projection(
        &mut self,
        namespace: &str,
        key: &str,
        projection: NamedProjectionWrite,
    ) -> SqliteStoreResult<()> {
        replace_stage1_node_projection(self.transaction, namespace, key, projection)
    }

    pub fn get_named_projection(
        &mut self,
        namespace: &str,
        key: &str,
    ) -> SqliteStoreResult<Option<NamedProjection>> {
        named_projection(self.transaction, namespace, key)
    }

    pub fn compare_and_swap_named_projection(
        &mut self,
        namespace: &str,
        key: &str,
        expected_last_authoritative_seq: Option<i64>,
        expected_last_materialized_seq: Option<i64>,
        projection: NamedProjectionWrite,
    ) -> SqliteStoreResult<bool> {
        compare_and_swap_named_projection(
            self.transaction,
            namespace,
            key,
            expected_last_authoritative_seq,
            expected_last_materialized_seq,
            projection,
        )
    }

    pub fn compare_and_swap_named_projections(
        &mut self,
        updates: Vec<NamedProjectionCas>,
    ) -> SqliteStoreResult<bool> {
        compare_and_swap_named_projections(self.transaction, updates)
    }

    pub fn clear_named_projection(&mut self, namespace: &str, key: &str) -> SqliteStoreResult<()> {
        clear_named_projection(self.transaction, namespace, key)
    }

    pub fn clear_stage1_node_projection(
        &mut self,
        namespace: &str,
        key: &str,
    ) -> SqliteStoreResult<()> {
        clear_stage1_node_projection(self.transaction, namespace, key)
    }

    pub fn clear_projection_namespace(&mut self, namespace: &str) -> SqliteStoreResult<()> {
        clear_projection_namespace(self.transaction, namespace)
    }

    pub fn put_workflow_design_snapshot(
        &mut self,
        workflow_id: &str,
        snapshot: WorkflowDesignSnapshotWrite,
    ) -> SqliteStoreResult<()> {
        put_workflow_design_snapshot(self.transaction, workflow_id, snapshot)
    }

    pub fn get_workflow_design_snapshot(
        &self,
        workflow_id: &str,
        max_version: i64,
        schema_version: i64,
    ) -> SqliteStoreResult<Option<WorkflowDesignSnapshot>> {
        workflow_design_snapshot(self.transaction, workflow_id, max_version, schema_version)
    }

    pub fn clear_workflow_design_snapshots(&mut self, workflow_id: &str) -> SqliteStoreResult<()> {
        clear_workflow_design_snapshots(self.transaction, workflow_id)
    }

    pub fn put_workflow_design_delta(
        &mut self,
        workflow_id: &str,
        delta: WorkflowDesignDeltaWrite,
    ) -> SqliteStoreResult<()> {
        put_workflow_design_delta(self.transaction, workflow_id, delta)
    }

    pub fn get_workflow_design_delta(
        &self,
        workflow_id: &str,
        version: i64,
        schema_version: i64,
    ) -> SqliteStoreResult<Option<WorkflowDesignDelta>> {
        workflow_design_delta(self.transaction, workflow_id, version, schema_version)
    }

    pub fn clear_workflow_design_deltas(&mut self, workflow_id: &str) -> SqliteStoreResult<()> {
        clear_workflow_design_deltas(self.transaction, workflow_id)
    }

    pub fn create_server_run(&mut self, run: ServerRunCreate) -> SqliteStoreResult<()> {
        create_server_run(self.transaction, run)
    }
    pub fn append_server_run_event(
        &mut self,
        run_id: &str,
        event_type: &str,
        payload_json: String,
    ) -> SqliteStoreResult<ServerRunEvent> {
        append_server_run_event(self.transaction, run_id, event_type, payload_json)
    }
    pub fn update_server_run(
        &mut self,
        run_id: &str,
        update: ServerRunUpdate,
    ) -> SqliteStoreResult<()> {
        update_server_run(self.transaction, run_id, update)
    }
    pub fn request_server_run_cancel(&mut self, run_id: &str) -> SqliteStoreResult<()> {
        request_server_run_cancel(self.transaction, run_id)
    }

    pub fn apply_recorded_runtime_transition(
        &mut self,
        transition: RecordedRuntimeTransition,
        abort_after_writes: bool,
    ) -> SqliteStoreResult<RecordedTransitionResult> {
        apply_recorded_runtime_transition(self.transaction, transition, abort_after_writes)
    }

    pub fn apply_claimed_recorded_runtime_transition(
        &mut self,
        handoff: RecordedWorkerHandoff,
        transition: RecordedRuntimeTransition,
        abort_after_writes: bool,
    ) -> SqliteStoreResult<RecordedTransitionResult> {
        apply_claimed_recorded_runtime_transition(
            self.transaction,
            handoff,
            transition,
            abort_after_writes,
        )
    }

    pub fn apply_claimed_recorded_worker_effect(
        &mut self,
        handoff: RecordedWorkerHandoff,
        effect: RecordedWorkerSuccessEffect,
    ) -> SqliteStoreResult<RecordedTransitionResult> {
        apply_claimed_recorded_worker_effect(self.transaction, handoff, effect)
    }
}
