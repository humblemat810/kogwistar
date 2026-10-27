use super::*;

impl ServerRunReadStore for PostgresStore {
    async fn server_run(&self, run_id: &str) -> StoreResult<Option<ServerRun>> {
        PostgresStore::get_server_run(self, run_id)
            .await
            .map_err(trait_error)
    }

    async fn server_runs(
        &self,
        status: Option<&str>,
        workflow_id: Option<&str>,
        conversation_id: Option<&str>,
        limit: usize,
    ) -> StoreResult<Vec<ServerRun>> {
        PostgresStore::list_server_runs(self, status, workflow_id, conversation_id, limit)
            .await
            .map_err(trait_error)
    }

    async fn server_run_events(
        &self,
        run_id: &str,
        after_seq: i64,
        limit: usize,
    ) -> StoreResult<Vec<ServerRunEvent>> {
        PostgresStore::list_server_run_events(self, run_id, after_seq, limit)
            .await
            .map_err(trait_error)
    }
}

impl ServerRunWriteStore for PostgresStore {
    async fn create_server_run(&self, run: ServerRunCreate) -> StoreResult<()> {
        PostgresStore::create_server_run(self, run)
            .await
            .map_err(trait_error)
    }

    async fn append_server_run_event(
        &self,
        run_id: &str,
        event_type: &str,
        payload_json: String,
    ) -> StoreResult<ServerRunEvent> {
        PostgresStore::append_server_run_event(self, run_id, event_type, payload_json)
            .await
            .map_err(trait_error)
    }

    async fn update_server_run(&self, run_id: &str, update: ServerRunUpdate) -> StoreResult<()> {
        PostgresStore::update_server_run(self, run_id, update)
            .await
            .map_err(trait_error)
    }

    async fn request_server_run_cancel(&self, run_id: &str) -> StoreResult<()> {
        PostgresStore::request_server_run_cancel(self, run_id)
            .await
            .map_err(trait_error)
    }
}

impl IndexJobReadStore for PostgresStore {
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
            .await
            .map_err(trait_error)
    }
}
impl IndexJobWriteStore for PostgresStore {
    async fn enqueue_index_job(&self, job: NewIndexJob) -> StoreResult<String> {
        PostgresStore::enqueue_index_job(self, job)
            .await
            .map_err(trait_error)
    }
    async fn claim_index_jobs(
        &self,
        limit: usize,
        lease_seconds: i64,
        namespace: Option<&str>,
    ) -> StoreResult<Vec<IndexJob>> {
        PostgresStore::claim_index_jobs(self, limit, lease_seconds, namespace)
            .await
            .map_err(trait_error)
    }
    async fn claim_index_jobs_filtered(
        &self,
        limit: usize,
        lease_seconds: i64,
        filter: IndexJobClaimFilter,
    ) -> StoreResult<Vec<IndexJob>> {
        PostgresStore::claim_index_jobs_filtered(self, limit, lease_seconds, filter)
            .await
            .map_err(trait_error)
    }
    async fn mark_index_job_done(
        &self,
        job_id: &str,
        claim_token: Option<&str>,
    ) -> StoreResult<bool> {
        PostgresStore::mark_index_job_done(self, job_id, claim_token)
            .await
            .map_err(trait_error)
    }
    async fn accept_index_job_result(
        &self,
        job_id: &str,
        claim_token: &str,
        result_json: &str,
        result_sha256: &str,
    ) -> StoreResult<AcceptedIndexJobResult> {
        PostgresStore::accept_index_job_result(
            self,
            job_id,
            claim_token,
            result_json,
            result_sha256,
        )
        .await
        .map_err(trait_error)
    }
    async fn index_job_result(&self, job_id: &str) -> StoreResult<Option<AcceptedIndexJobResult>> {
        PostgresStore::index_job_result(self, job_id)
            .await
            .map_err(trait_error)
    }
    async fn mark_index_job_failed(
        &self,
        job_id: &str,
        error: &str,
        final_: bool,
        claim_token: Option<&str>,
    ) -> StoreResult<()> {
        PostgresStore::mark_index_job_failed(self, job_id, error, final_, claim_token)
            .await
            .map_err(trait_error)
    }
    async fn bump_retry_and_requeue(
        &self,
        job_id: &str,
        error: &str,
        delay: i64,
        claim_token: Option<&str>,
    ) -> StoreResult<()> {
        PostgresStore::bump_retry_and_requeue(self, job_id, error, delay, claim_token)
            .await
            .map_err(trait_error)
    }
    async fn renew_index_job_lease(
        &self,
        job_id: &str,
        claim_token: &str,
        lease_seconds: i64,
    ) -> StoreResult<bool> {
        PostgresStore::renew_index_job_lease(self, job_id, claim_token, lease_seconds)
            .await
            .map_err(trait_error)
    }
    async fn requeue_index_job_at_tail(
        &self,
        job_id: &str,
        payload_json: String,
        delay: i64,
        claim_token: Option<&str>,
    ) -> StoreResult<()> {
        PostgresStore::requeue_index_job_at_tail(self, job_id, payload_json, delay, claim_token)
            .await
            .map_err(trait_error)
    }
}

impl LaneMessageReadStore for PostgresStore {
    async fn projected_lane_message(&self, id: &str) -> StoreResult<Option<ProjectedLaneMessage>> {
        self.get_projected_lane_message(id)
            .await
            .map_err(trait_error)
    }
    async fn projected_lane_messages(
        &self,
        filter: LaneMessageFilter,
    ) -> StoreResult<Vec<ProjectedLaneMessage>> {
        self.list_projected_lane_messages(filter)
            .await
            .map_err(trait_error)
    }
}
impl LaneMessageWriteStore for PostgresStore {
    async fn project_lane_message(&self, row: NewProjectedLaneMessage) -> StoreResult<()> {
        PostgresStore::project_lane_message(self, row)
            .await
            .map_err(trait_error)
    }
    async fn update_projected_lane_message_status(
        &self,
        id: &str,
        status: &str,
        error: Option<String>,
    ) -> StoreResult<()> {
        PostgresStore::update_projected_lane_message_status(self, id, status, error)
            .await
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
        PostgresStore::update_projected_lane_message_links(
            self,
            id,
            prev,
            next,
            inbox_tail,
            conversation_tail,
        )
        .await
        .map_err(trait_error)
    }
    async fn clear_projected_lane_messages(&self, namespace: &str) -> StoreResult<u64> {
        PostgresStore::clear_projected_lane_messages(self, namespace)
            .await
            .map_err(trait_error)
    }
    async fn claim_projected_lane_messages(
        &self,
        namespace: &str,
        inbox: &str,
        owner: &str,
        limit: usize,
        lease: i64,
    ) -> StoreResult<Vec<ProjectedLaneMessage>> {
        PostgresStore::claim_projected_lane_messages(self, namespace, inbox, owner, limit, lease)
            .await
            .map_err(trait_error)
    }
    async fn claim_projected_lane_messages_filtered(
        &self,
        filter: LaneMessageClaimFilter,
    ) -> StoreResult<Vec<ProjectedLaneMessage>> {
        PostgresStore::claim_projected_lane_messages_filtered(self, filter)
            .await
            .map_err(trait_error)
    }
    async fn ack_projected_lane_message(&self, id: &str, owner: &str) -> StoreResult<()> {
        PostgresStore::ack_projected_lane_message(self, id, owner)
            .await
            .map_err(trait_error)
    }
    async fn requeue_projected_lane_message(
        &self,
        id: &str,
        owner: &str,
        error: Option<String>,
        delay: i64,
    ) -> StoreResult<()> {
        PostgresStore::requeue_projected_lane_message(self, id, owner, error, delay)
            .await
            .map_err(trait_error)
    }
    async fn dead_letter_projected_lane_message(
        &self,
        id: &str,
        owner: &str,
        error: Option<String>,
    ) -> StoreResult<()> {
        PostgresStore::dead_letter_projected_lane_message(self, id, owner, error)
            .await
            .map_err(trait_error)
    }
}

impl EventReadStore for PostgresStore {
    async fn replay_events(
        &self,
        namespace: &str,
        after_seq: i64,
        limit: usize,
    ) -> StoreResult<Vec<EntityEvent>> {
        require_trait_namespace(namespace)?;
        self.replay_raw_events(namespace, after_seq, limit)
            .await
            .map_err(trait_error)?
            .into_iter()
            .map(raw_to_entity_event)
            .collect::<PostgresStoreResult<Vec<_>>>()
            .map_err(trait_error)
    }

    async fn replay_cursor(&self, namespace: &str, consumer: &str) -> StoreResult<ReplayCursor> {
        require_trait_namespace(namespace)?;
        PostgresStore::replay_cursor(self, namespace, consumer)
            .await
            .map_err(trait_error)
    }

    async fn latest_event_seq(&self, namespace: &str) -> StoreResult<i64> {
        require_trait_namespace(namespace)?;
        self.latest_retained_event_seq(namespace)
            .await
            .map_err(trait_error)
    }
}

impl EventWriteStore for PostgresStore {
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
            .await
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
            .await
            .map_err(trait_error)
    }
}

impl EventPruneStore for PostgresStore {
    async fn prune_entity_events_after(&self, namespace: &str, to_seq: i64) -> StoreResult<u64> {
        require_trait_namespace(namespace)?;
        PostgresStore::prune_entity_events_after(self, namespace, to_seq)
            .await
            .map_err(trait_error)
    }
}

impl GraphMutationStore for PostgresStore {
    fn graph_authority_capability(&self) -> GraphAuthorityCapability {
        GraphAuthorityCapability::AtomicBatchWithIndexJobs
    }

    async fn apply_graph_mutation(
        &self,
        mutation: GraphMutation,
    ) -> StoreResult<AppliedGraphMutation> {
        PostgresStore::apply_graph_mutation(self, mutation)
            .await
            .map_err(trait_error)
    }

    async fn apply_graph_mutations_atomic(
        &self,
        mutations: Vec<GraphMutation>,
    ) -> StoreResult<Vec<AppliedGraphMutation>> {
        PostgresStore::apply_graph_mutations_atomic(self, mutations)
            .await
            .map_err(trait_error)
    }

    async fn apply_graph_mutations_with_index_jobs_atomic(
        &self,
        batch: GraphMutationBatch,
    ) -> StoreResult<Vec<AppliedGraphMutation>> {
        PostgresStore::apply_graph_mutations_with_index_jobs_atomic(self, batch)
            .await
            .map_err(trait_error)
    }

    async fn graph_projection_records(
        &self,
        read: GraphProjectionRead,
    ) -> StoreResult<Vec<GraphRecord>> {
        PostgresStore::graph_projection_records(self, read)
            .await
            .map_err(trait_error)
    }

    async fn graph_projection_page(
        &self,
        page: GraphProjectionPageRead,
    ) -> StoreResult<Vec<GraphRecord>> {
        PostgresStore::graph_projection_page(self, page)
            .await
            .map_err(trait_error)
    }

    async fn graph_projection_vector_query(
        &self,
        query: GraphProjectionVectorQuery,
    ) -> StoreResult<Vec<VectorMatch>> {
        PostgresStore::graph_projection_vector_query(self, query)
            .await
            .map_err(trait_error)
    }
}

impl DocumentStore for PostgresStore {
    fn document_authority_capability(&self) -> DocumentAuthorityCapability {
        DocumentAuthorityCapability::ProjectionUpsert
    }

    async fn upsert_document_projection(&self, write: DocumentProjectionUpsert) -> StoreResult<()> {
        self.upsert_graph_projection(GraphProjectionUpsert {
            scope: write.scope,
            table: "gke_documents".to_owned(),
            record: write.record,
            embedding_dim: write.embedding_dim,
        })
        .await
        .map_err(trait_error)
    }
}

impl ProjectionReadStore for PostgresStore {
    async fn named_projection(
        &self,
        namespace: &str,
        key: &str,
    ) -> StoreResult<Option<NamedProjection>> {
        require_trait_namespace(namespace)?;
        self.get_named_projection(namespace, key)
            .await
            .map_err(trait_error)
    }

    async fn named_projections(&self, namespace: &str) -> StoreResult<Vec<NamedProjection>> {
        require_trait_namespace(namespace)?;
        self.list_named_projections(namespace)
            .await
            .map_err(trait_error)
    }
}

impl ProjectionWriteStore for PostgresStore {
    async fn replace_named_projection(
        &self,
        namespace: &str,
        key: &str,
        projection: NamedProjectionWrite,
    ) -> StoreResult<()> {
        require_trait_namespace(namespace)?;
        PostgresStore::replace_named_projection(self, namespace, key, projection)
            .await
            .map_err(trait_error)
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
        PostgresStore::compare_and_swap_named_projection(
            self,
            namespace,
            key,
            expected_last_authoritative_seq,
            expected_last_materialized_seq,
            projection,
        )
        .await
        .map_err(trait_error)
    }

    async fn clear_named_projection(&self, namespace: &str, key: &str) -> StoreResult<()> {
        require_trait_namespace(namespace)?;
        PostgresStore::clear_named_projection(self, namespace, key)
            .await
            .map_err(trait_error)
    }

    async fn clear_projection_namespace(&self, namespace: &str) -> StoreResult<()> {
        require_trait_namespace(namespace)?;
        PostgresStore::clear_projection_namespace(self, namespace)
            .await
            .map_err(trait_error)
    }
}

impl WorkflowDesignHistoryReadStore for PostgresStore {
    async fn workflow_design_snapshot(
        &self,
        workflow_id: &str,
        max_version: i64,
        schema_version: i64,
    ) -> StoreResult<Option<WorkflowDesignSnapshot>> {
        PostgresStore::get_workflow_design_snapshot(self, workflow_id, max_version, schema_version)
            .await
            .map_err(trait_error)
    }

    async fn workflow_design_delta(
        &self,
        workflow_id: &str,
        version: i64,
        schema_version: i64,
    ) -> StoreResult<Option<WorkflowDesignDelta>> {
        PostgresStore::get_workflow_design_delta(self, workflow_id, version, schema_version)
            .await
            .map_err(trait_error)
    }
}

impl WorkflowDesignHistoryWriteStore for PostgresStore {
    async fn put_workflow_design_snapshot(
        &self,
        workflow_id: &str,
        snapshot: WorkflowDesignSnapshotWrite,
    ) -> StoreResult<()> {
        PostgresStore::put_workflow_design_snapshot(self, workflow_id, snapshot)
            .await
            .map_err(trait_error)
    }

    async fn clear_workflow_design_snapshots(&self, workflow_id: &str) -> StoreResult<()> {
        PostgresStore::clear_workflow_design_snapshots(self, workflow_id)
            .await
            .map_err(trait_error)
    }

    async fn put_workflow_design_delta(
        &self,
        workflow_id: &str,
        delta: WorkflowDesignDeltaWrite,
    ) -> StoreResult<()> {
        PostgresStore::put_workflow_design_delta(self, workflow_id, delta)
            .await
            .map_err(trait_error)
    }

    async fn clear_workflow_design_deltas(&self, workflow_id: &str) -> StoreResult<()> {
        PostgresStore::clear_workflow_design_deltas(self, workflow_id)
            .await
            .map_err(trait_error)
    }
}
