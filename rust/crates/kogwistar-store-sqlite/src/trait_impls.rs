use super::*;

impl ServerRunReadStore for SqliteStore {
    async fn server_run(&self, run_id: &str) -> StoreResult<Option<ServerRun>> {
        SqliteStore::get_server_run(self, run_id).map_err(trait_error)
    }

    async fn server_runs(
        &self,
        status: Option<&str>,
        workflow_id: Option<&str>,
        conversation_id: Option<&str>,
        limit: usize,
    ) -> StoreResult<Vec<ServerRun>> {
        SqliteStore::list_server_runs(self, status, workflow_id, conversation_id, limit)
            .map_err(trait_error)
    }

    async fn server_run_events(
        &self,
        run_id: &str,
        after_seq: i64,
        limit: usize,
    ) -> StoreResult<Vec<ServerRunEvent>> {
        SqliteStore::list_server_run_events(self, run_id, after_seq, limit).map_err(trait_error)
    }
}

impl ServerRunWriteStore for SqliteStore {
    async fn create_server_run(&self, run: ServerRunCreate) -> StoreResult<()> {
        SqliteStore::create_server_run(self, run).map_err(trait_error)
    }

    async fn append_server_run_event(
        &self,
        run_id: &str,
        event_type: &str,
        payload_json: String,
    ) -> StoreResult<ServerRunEvent> {
        SqliteStore::append_server_run_event(self, run_id, event_type, payload_json)
            .map_err(trait_error)
    }

    async fn update_server_run(&self, run_id: &str, update: ServerRunUpdate) -> StoreResult<()> {
        SqliteStore::update_server_run(self, run_id, update).map_err(trait_error)
    }

    async fn request_server_run_cancel(&self, run_id: &str) -> StoreResult<()> {
        SqliteStore::request_server_run_cancel(self, run_id).map_err(trait_error)
    }
}

impl IndexJobReadStore for SqliteStore {
    async fn index_jobs(
        &self,
        namespace: Option<&str>,
        status: Option<&str>,
        entity_kind: Option<&str>,
        entity_id: Option<&str>,
        index_kind: Option<&str>,
        limit: usize,
    ) -> StoreResult<Vec<IndexJob>> {
        self.list_index_jobs(namespace, status, entity_kind, entity_id, index_kind, limit)
            .map_err(trait_error)
    }
}
impl IndexJobWriteStore for SqliteStore {
    async fn enqueue_index_job(&self, job: NewIndexJob) -> StoreResult<String> {
        SqliteStore::enqueue_index_job(self, job).map_err(trait_error)
    }
    async fn claim_index_jobs(
        &self,
        limit: usize,
        lease_seconds: i64,
        namespace: Option<&str>,
    ) -> StoreResult<Vec<IndexJob>> {
        SqliteStore::claim_index_jobs(self, limit, lease_seconds, namespace).map_err(trait_error)
    }
    async fn mark_index_job_done(
        &self,
        job_id: &str,
        claim_token: Option<&str>,
    ) -> StoreResult<bool> {
        SqliteStore::mark_index_job_done(self, job_id, claim_token).map_err(trait_error)
    }
    async fn accept_index_job_result(
        &self,
        job_id: &str,
        claim_token: &str,
        result_json: &str,
        result_sha256: &str,
    ) -> StoreResult<AcceptedIndexJobResult> {
        SqliteStore::accept_index_job_result(self, job_id, claim_token, result_json, result_sha256)
            .map_err(trait_error)
    }
    async fn index_job_result(&self, job_id: &str) -> StoreResult<Option<AcceptedIndexJobResult>> {
        SqliteStore::index_job_result(self, job_id).map_err(trait_error)
    }
    async fn mark_index_job_failed(
        &self,
        job_id: &str,
        error: &str,
        final_: bool,
        claim_token: Option<&str>,
    ) -> StoreResult<()> {
        SqliteStore::mark_index_job_failed(self, job_id, error, final_, claim_token)
            .map_err(trait_error)
    }
    async fn bump_retry_and_requeue(
        &self,
        job_id: &str,
        error: &str,
        delay: i64,
        claim_token: Option<&str>,
    ) -> StoreResult<()> {
        SqliteStore::bump_retry_and_requeue(self, job_id, error, delay, claim_token)
            .map_err(trait_error)
    }
    async fn renew_index_job_lease(
        &self,
        job_id: &str,
        claim_token: &str,
        lease_seconds: i64,
    ) -> StoreResult<bool> {
        SqliteStore::renew_index_job_lease(self, job_id, claim_token, lease_seconds)
            .map_err(trait_error)
    }
    async fn requeue_index_job_at_tail(
        &self,
        job_id: &str,
        payload_json: String,
        delay: i64,
        claim_token: Option<&str>,
    ) -> StoreResult<()> {
        SqliteStore::requeue_index_job_at_tail(self, job_id, payload_json, delay, claim_token)
            .map_err(trait_error)
    }
}

impl LaneMessageReadStore for SqliteStore {
    async fn projected_lane_message(
        &self,
        message_id: &str,
    ) -> StoreResult<Option<ProjectedLaneMessage>> {
        self.get_projected_lane_message(message_id)
            .map_err(trait_error)
    }
    async fn projected_lane_messages(
        &self,
        filter: LaneMessageFilter,
    ) -> StoreResult<Vec<ProjectedLaneMessage>> {
        self.list_projected_lane_messages(filter)
            .map_err(trait_error)
    }
}
impl LaneMessageWriteStore for SqliteStore {
    async fn project_lane_message(&self, row: NewProjectedLaneMessage) -> StoreResult<()> {
        SqliteStore::project_lane_message(self, row).map_err(trait_error)
    }
    async fn update_projected_lane_message_status(
        &self,
        id: &str,
        status: &str,
        error: Option<String>,
    ) -> StoreResult<()> {
        SqliteStore::update_projected_lane_message_status(self, id, status, error)
            .map_err(trait_error)
    }
    async fn update_projected_lane_message_links(
        &self,
        id: &str,
        prev: Option<String>,
        next: Option<String>,
        inbox_tail: Option<String>,
        conversation_tail: Option<String>,
    ) -> StoreResult<()> {
        SqliteStore::update_projected_lane_message_links(
            self,
            id,
            prev,
            next,
            inbox_tail,
            conversation_tail,
        )
        .map_err(trait_error)
    }
    async fn clear_projected_lane_messages(&self, namespace: &str) -> StoreResult<u64> {
        SqliteStore::clear_projected_lane_messages(self, namespace).map_err(trait_error)
    }
    async fn claim_projected_lane_messages(
        &self,
        namespace: &str,
        inbox: &str,
        owner: &str,
        limit: usize,
        lease: i64,
    ) -> StoreResult<Vec<ProjectedLaneMessage>> {
        SqliteStore::claim_projected_lane_messages(self, namespace, inbox, owner, limit, lease)
            .map_err(trait_error)
    }
    async fn claim_projected_lane_messages_filtered(
        &self,
        filter: LaneMessageClaimFilter,
    ) -> StoreResult<Vec<ProjectedLaneMessage>> {
        SqliteStore::claim_projected_lane_messages_filtered(self, filter).map_err(trait_error)
    }
    async fn ack_projected_lane_message(&self, id: &str, owner: &str) -> StoreResult<()> {
        SqliteStore::ack_projected_lane_message(self, id, owner).map_err(trait_error)
    }
    async fn requeue_projected_lane_message(
        &self,
        id: &str,
        owner: &str,
        error: Option<String>,
        delay: i64,
    ) -> StoreResult<()> {
        SqliteStore::requeue_projected_lane_message(self, id, owner, error, delay)
            .map_err(trait_error)
    }
    async fn dead_letter_projected_lane_message(
        &self,
        id: &str,
        owner: &str,
        error: Option<String>,
    ) -> StoreResult<()> {
        SqliteStore::dead_letter_projected_lane_message(self, id, owner, error).map_err(trait_error)
    }
}

impl EventReadStore for SqliteStore {
    async fn replay_events(
        &self,
        namespace: &str,
        after_seq: i64,
        limit: usize,
    ) -> StoreResult<Vec<EntityEvent>> {
        require_trait_namespace(namespace)?;
        self.replay_raw_events(namespace, after_seq, limit)
            .map_err(trait_error)?
            .into_iter()
            .map(raw_to_entity_event)
            .collect::<SqliteStoreResult<Vec<_>>>()
            .map_err(trait_error)
    }

    async fn replay_cursor(&self, namespace: &str, consumer: &str) -> StoreResult<ReplayCursor> {
        require_trait_namespace(namespace)?;
        SqliteStore::replay_cursor(self, namespace, consumer).map_err(trait_error)
    }

    async fn latest_event_seq(&self, namespace: &str) -> StoreResult<i64> {
        require_trait_namespace(namespace)?;
        self.latest_retained_event_seq(namespace)
            .map_err(trait_error)
    }
}

impl EventWriteStore for SqliteStore {
    async fn append_entity_event(
        &self,
        namespace: &str,
        event: NewEntityEvent,
    ) -> StoreResult<AppendedEvent> {
        require_trait_namespace(namespace)?;
        let raw = self
            .append_raw_entity_event(
                namespace,
                NewRawEntityEvent {
                    event_id: event.event_id,
                    entity_kind: event.entity_kind,
                    entity_id: event.entity_id,
                    op: event.op,
                    payload_json: serde_json::to_string(&event.payload)
                        .expect("serde_json::Value serialization cannot fail"),
                },
            )
            .map_err(trait_error)?;
        Ok(AppendedEvent {
            event: raw_to_entity_event(raw.event).map_err(trait_error)?,
            inserted: raw.inserted,
        })
    }

    async fn advance_replay_cursor(
        &self,
        namespace: &str,
        consumer: &str,
        last_seq: i64,
    ) -> StoreResult<ReplayCursor> {
        require_trait_namespace(namespace)?;
        self.strict_advance_replay_cursor(namespace, consumer, last_seq)
            .map_err(trait_error)
    }
}

impl EventPruneStore for SqliteStore {
    async fn prune_entity_events_after(&self, namespace: &str, to_seq: i64) -> StoreResult<u64> {
        require_trait_namespace(namespace)?;
        SqliteStore::prune_entity_events_after(self, namespace, to_seq).map_err(trait_error)
    }
}

impl ProjectionReadStore for SqliteStore {
    async fn named_projection(
        &self,
        namespace: &str,
        key: &str,
    ) -> StoreResult<Option<NamedProjection>> {
        require_trait_namespace(namespace)?;
        self.get_named_projection(namespace, key)
            .map_err(trait_error)
    }

    async fn named_projections(&self, namespace: &str) -> StoreResult<Vec<NamedProjection>> {
        require_trait_namespace(namespace)?;
        self.list_named_projections(namespace).map_err(trait_error)
    }
}

impl ProjectionWriteStore for SqliteStore {
    async fn replace_named_projection(
        &self,
        namespace: &str,
        key: &str,
        projection: NamedProjectionWrite,
    ) -> StoreResult<()> {
        require_trait_namespace(namespace)?;
        SqliteStore::replace_named_projection(self, namespace, key, projection).map_err(trait_error)
    }

    async fn compare_and_swap_named_projection(
        &self,
        namespace: &str,
        key: &str,
        expected_last_authoritative_seq: Option<i64>,
        expected_last_materialized_seq: Option<i64>,
        projection: NamedProjectionWrite,
    ) -> StoreResult<bool> {
        require_trait_namespace(namespace)?;
        SqliteStore::compare_and_swap_named_projection(
            self,
            namespace,
            key,
            expected_last_authoritative_seq,
            expected_last_materialized_seq,
            projection,
        )
        .map_err(trait_error)
    }

    async fn clear_named_projection(&self, namespace: &str, key: &str) -> StoreResult<()> {
        require_trait_namespace(namespace)?;
        SqliteStore::clear_named_projection(self, namespace, key).map_err(trait_error)
    }

    async fn clear_projection_namespace(&self, namespace: &str) -> StoreResult<()> {
        require_trait_namespace(namespace)?;
        SqliteStore::clear_projection_namespace(self, namespace).map_err(trait_error)
    }
}

impl WorkflowDesignHistoryReadStore for SqliteStore {
    async fn workflow_design_snapshot(
        &self,
        workflow_id: &str,
        max_version: i64,
        schema_version: i64,
    ) -> StoreResult<Option<WorkflowDesignSnapshot>> {
        SqliteStore::get_workflow_design_snapshot(self, workflow_id, max_version, schema_version)
            .map_err(trait_error)
    }

    async fn workflow_design_delta(
        &self,
        workflow_id: &str,
        version: i64,
        schema_version: i64,
    ) -> StoreResult<Option<WorkflowDesignDelta>> {
        SqliteStore::get_workflow_design_delta(self, workflow_id, version, schema_version)
            .map_err(trait_error)
    }
}

impl WorkflowDesignHistoryWriteStore for SqliteStore {
    async fn put_workflow_design_snapshot(
        &self,
        workflow_id: &str,
        snapshot: WorkflowDesignSnapshotWrite,
    ) -> StoreResult<()> {
        SqliteStore::put_workflow_design_snapshot(self, workflow_id, snapshot).map_err(trait_error)
    }

    async fn clear_workflow_design_snapshots(&self, workflow_id: &str) -> StoreResult<()> {
        SqliteStore::clear_workflow_design_snapshots(self, workflow_id).map_err(trait_error)
    }

    async fn put_workflow_design_delta(
        &self,
        workflow_id: &str,
        delta: WorkflowDesignDeltaWrite,
    ) -> StoreResult<()> {
        SqliteStore::put_workflow_design_delta(self, workflow_id, delta).map_err(trait_error)
    }

    async fn clear_workflow_design_deltas(&self, workflow_id: &str) -> StoreResult<()> {
        SqliteStore::clear_workflow_design_deltas(self, workflow_id).map_err(trait_error)
    }
}
