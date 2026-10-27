use super::*;

/// Operations sharing one RAII PostgreSQL transaction.
pub struct PostgresUnitOfWork<'transaction> {
    pub(super) transaction: Transaction<'transaction>,
    pub(super) tables: Tables,
}

impl PostgresUnitOfWork<'_> {
    /// Serialize runtime queue admission checks within this database schema.
    /// The lock is transaction-scoped and released on commit or rollback.
    pub async fn lock_runtime_queue_admission(&self) -> PostgresStoreResult<()> {
        advisory_lock(
            &self.transaction,
            &format!(
                "runtime-queue-admission\u{1f}{}",
                self.tables.projected_lane_messages
            ),
            73,
        )
        .await
    }

    pub async fn apply_graph_mutation(
        &mut self,
        mutation: GraphMutation,
    ) -> PostgresStoreResult<AppliedGraphMutation> {
        validate_graph_mutation(&mutation)?;
        apply_graph_mutation(&self.transaction, &self.tables, mutation).await
    }

    pub async fn apply_graph_metadata_patch_mutation(
        &mut self,
        mutation: GraphMetadataPatchMutation,
    ) -> PostgresStoreResult<Option<AppliedGraphMutation>> {
        apply_graph_metadata_patch_mutation(&self.transaction, &self.tables, mutation).await
    }

    pub async fn apply_graph_delete_mutation(
        &mut self,
        mutation: GraphDeleteMutation,
    ) -> PostgresStoreResult<Option<AppliedGraphMutation>> {
        apply_graph_delete_mutation(&self.transaction, &self.tables, mutation).await
    }

    pub async fn upsert_graph_projection(
        &mut self,
        write: GraphProjectionUpsert,
    ) -> PostgresStoreResult<()> {
        upsert_graph_projection(&self.transaction, &self.tables, write).await
    }

    pub async fn promote_graph_embedding_projection(
        &mut self,
        namespace: &str,
        entity_kind: &str,
        mut candidate: GraphRecord,
    ) -> PostgresStoreResult<bool> {
        require_namespace(namespace)?;
        let table = match entity_kind {
            "node" => "gke_nodes",
            "edge" => "gke_edges",
            _ => {
                return Err(StoreError::InvalidEntityEventPayload {
                    message: "embedding promotion supports node or edge only".to_owned(),
                }
                .into());
            }
        };
        let embedding_dim = candidate
            .embedding
            .as_ref()
            .map(Vec::len)
            .filter(|dimension| *dimension > 0)
            .ok_or_else(|| {
                PostgresStoreError::Backend(
                    "embedding promotion requires a non-empty vector".to_owned(),
                )
            })?;
        let scope = GraphScope {
            namespace: namespace.to_owned(),
            workspace_id: None,
            graph_space: None,
        };
        let current = self
            .graph_projection_records_for_update(GraphProjectionRead {
                scope: scope.clone(),
                table: table.to_owned(),
                ids: Some(vec![candidate.id.clone()]),
                metadata: Default::default(),
                limit: 1,
            })
            .await?
            .into_iter()
            .next();
        let Some(mut current) = current else {
            return Ok(false);
        };
        if current
            .metadata
            .get("lifecycle_status")
            .and_then(Value::as_str)
            .is_some_and(|status| matches!(status, "tombstoned" | "deleted"))
        {
            return Ok(false);
        }
        let expected_metadata = graph_metadata(&scope, &candidate.metadata)?;
        if current.document != candidate.document || current.metadata != expected_metadata {
            return Ok(false);
        }

        // The provider is allowed to supply the vector only. Preserve the
        // canonical row values read and locked above.
        current.embedding = candidate.embedding.take();
        self.upsert_graph_projection(GraphProjectionUpsert {
            scope,
            table: table.to_owned(),
            record: current,
            embedding_dim,
        })
        .await?;
        Ok(true)
    }

    pub async fn patch_graph_projection_metadata(
        &mut self,
        patch: GraphProjectionMetadataPatch,
    ) -> PostgresStoreResult<bool> {
        patch_graph_projection_metadata(&self.transaction, &self.tables, patch).await
    }

    pub async fn graph_projection_records(
        &self,
        read: GraphProjectionRead,
    ) -> PostgresStoreResult<Vec<GraphRecord>> {
        graph_projection_records(&self.transaction, &self.tables, read, None, false).await
    }

    pub async fn graph_projection_records_for_update(
        &self,
        read: GraphProjectionRead,
    ) -> PostgresStoreResult<Vec<GraphRecord>> {
        graph_projection_records_with_lock(&self.transaction, &self.tables, read, None, false, true)
            .await
    }

    pub async fn graph_projection_vector_query(
        &self,
        query: GraphProjectionVectorQuery,
    ) -> PostgresStoreResult<Vec<VectorMatch>> {
        graph_projection_vector_query(&self.transaction, &self.tables, query, &BTreeMap::new())
            .await
    }

    pub async fn project_lane_message(
        &mut self,
        row: NewProjectedLaneMessage,
    ) -> PostgresStoreResult<()> {
        project_lane_message(&self.transaction, &self.tables, row).await
    }
    pub async fn get_projected_lane_message(
        &self,
        id: &str,
    ) -> PostgresStoreResult<Option<ProjectedLaneMessage>> {
        projected_lane_message(&self.transaction, &self.tables, id).await
    }
    pub async fn get_server_run(&self, run_id: &str) -> PostgresStoreResult<Option<ServerRun>> {
        server_run(&self.transaction, &self.tables, run_id).await
    }
    pub async fn list_projected_lane_messages(
        &self,
        filter: &LaneMessageFilter,
    ) -> PostgresStoreResult<Vec<ProjectedLaneMessage>> {
        list_projected_lane_messages(&self.transaction, &self.tables, filter).await
    }
    pub async fn update_projected_lane_message_status(
        &mut self,
        id: &str,
        status: &str,
        error: Option<String>,
    ) -> PostgresStoreResult<()> {
        update_projected_lane_message_status(&self.transaction, &self.tables, id, status, error)
            .await
    }
    pub async fn update_projected_lane_message_payload(
        &mut self,
        id: &str,
        payload_json: String,
    ) -> PostgresStoreResult<()> {
        let changed = self
            .transaction
            .execute(
                &format!(
                    "UPDATE {} SET payload_json=$1 WHERE message_id=$2",
                    self.tables.projected_lane_messages
                ),
                &[&payload_json, &id],
            )
            .await
            .map_err(backend)?;
        if changed != 1 {
            return Err(PostgresStoreError::RecordedRuntimeConflict(format!(
                "lane message {id:?} payload update matched {changed} rows"
            )));
        }
        Ok(())
    }
    pub async fn update_projected_lane_message_links(
        &mut self,
        id: &str,
        prev: Option<String>,
        next: Option<String>,
        inbox_tail: Option<String>,
        conversation_tail: Option<String>,
    ) -> PostgresStoreResult<()> {
        update_projected_lane_message_links(
            &self.transaction,
            &self.tables,
            id,
            prev,
            next,
            inbox_tail,
            conversation_tail,
        )
        .await
    }
    pub async fn clear_projected_lane_messages(
        &mut self,
        namespace: &str,
    ) -> PostgresStoreResult<u64> {
        clear_projected_lane_messages(&self.transaction, &self.tables, namespace).await
    }
    pub async fn claim_projected_lane_messages(
        &mut self,
        namespace: &str,
        inbox: &str,
        owner: &str,
        limit: usize,
        lease: i64,
    ) -> PostgresStoreResult<Vec<ProjectedLaneMessage>> {
        claim_projected_lane_messages(
            &self.transaction,
            &self.tables,
            namespace,
            inbox,
            owner,
            limit,
            lease,
        )
        .await
    }
    pub async fn claim_projected_lane_messages_filtered(
        &mut self,
        filter: LaneMessageClaimFilter,
    ) -> PostgresStoreResult<Vec<ProjectedLaneMessage>> {
        claim_projected_lane_messages_filtered(&self.transaction, &self.tables, &filter).await
    }
    pub async fn claim_projected_lane_messages_for_run(
        &mut self,
        namespace: &str,
        inbox: &str,
        run_id: &str,
        owner: &str,
        limit: usize,
        lease: i64,
    ) -> PostgresStoreResult<Vec<ProjectedLaneMessage>> {
        claim_projected_lane_messages_for_run(
            &self.transaction,
            &self.tables,
            namespace,
            inbox,
            run_id,
            owner,
            limit,
            lease,
        )
        .await
    }
    pub async fn ack_projected_lane_message(
        &mut self,
        id: &str,
        owner: &str,
    ) -> PostgresStoreResult<()> {
        ack_projected_lane_message(&self.transaction, &self.tables, id, owner).await
    }
    pub async fn requeue_projected_lane_message(
        &mut self,
        id: &str,
        owner: &str,
        error: Option<String>,
        delay: i64,
    ) -> PostgresStoreResult<()> {
        requeue_projected_lane_message(&self.transaction, &self.tables, id, owner, error, delay)
            .await
    }
    pub async fn dead_letter_projected_lane_message(
        &mut self,
        id: &str,
        owner: &str,
        error: Option<String>,
    ) -> PostgresStoreResult<()> {
        dead_letter_projected_lane_message(&self.transaction, &self.tables, id, owner, error).await
    }
    pub async fn repair_orphaned_claimed_lane_messages(
        &mut self,
        namespace: &str,
        inbox_id: Option<&str>,
        limit: usize,
    ) -> PostgresStoreResult<Vec<String>> {
        repair_orphaned_claimed_lane_messages(
            &self.transaction,
            &self.tables,
            namespace,
            inbox_id,
            limit,
        )
        .await
    }
    pub async fn enqueue_index_job(&mut self, job: NewIndexJob) -> PostgresStoreResult<String> {
        require_namespace(&job.namespace)?;
        advisory_lock(
            &self.transaction,
            &format!(
                "index-job\u{1f}{}\u{1f}{}",
                job.namespace,
                job.coalesce_key()
            ),
            2,
        )
        .await?;
        enqueue_index_job(&self.transaction, &self.tables, job).await
    }
    pub async fn claim_index_jobs(
        &mut self,
        limit: usize,
        lease_seconds: i64,
        namespace: Option<&str>,
    ) -> PostgresStoreResult<Vec<IndexJob>> {
        claim_index_jobs(
            &self.transaction,
            &self.tables,
            limit,
            lease_seconds,
            namespace,
            None,
        )
        .await
    }
    pub async fn claim_index_jobs_filtered(
        &mut self,
        limit: usize,
        lease_seconds: i64,
        namespace: Option<&str>,
        index_kinds: &[String],
    ) -> PostgresStoreResult<Vec<IndexJob>> {
        if index_kinds.is_empty() || index_kinds.iter().any(|kind| kind.trim().is_empty()) {
            return Err(PostgresStoreError::Backend(
                "filtered index-job claim requires non-empty index kinds".to_owned(),
            ));
        }
        claim_index_jobs(
            &self.transaction,
            &self.tables,
            limit,
            lease_seconds,
            namespace,
            Some(index_kinds),
        )
        .await
    }
    pub async fn reconcile_join_index_jobs(
        &mut self,
        limit: usize,
        lease_seconds: i64,
        namespace: Option<&str>,
        names: GraphIndexTableNames,
    ) -> PostgresStoreResult<usize> {
        reconcile_join_index_jobs(
            &self.transaction,
            &self.tables,
            limit,
            lease_seconds,
            namespace,
            names,
        )
        .await
    }
    pub async fn mark_index_job_done(
        &mut self,
        job_id: &str,
        claim_token: Option<&str>,
    ) -> PostgresStoreResult<bool> {
        mark_index_job_done(&self.transaction, &self.tables, job_id, claim_token).await
    }
    pub async fn accept_index_job_result(
        &mut self,
        job_id: &str,
        claim_token: &str,
        result_json: &str,
        result_sha256: &str,
    ) -> PostgresStoreResult<AcceptedIndexJobResult> {
        accept_index_job_result(
            &self.transaction,
            &self.tables,
            job_id,
            claim_token,
            result_json,
            result_sha256,
        )
        .await
    }
    pub async fn mark_index_job_failed(
        &mut self,
        job_id: &str,
        error: &str,
        final_: bool,
        claim_token: Option<&str>,
    ) -> PostgresStoreResult<()> {
        mark_index_job_failed(
            &self.transaction,
            &self.tables,
            job_id,
            error,
            final_,
            claim_token,
        )
        .await
    }
    pub async fn bump_retry_and_requeue(
        &mut self,
        job_id: &str,
        error: &str,
        delay: i64,
        claim_token: Option<&str>,
    ) -> PostgresStoreResult<()> {
        bump_retry_and_requeue(
            &self.transaction,
            &self.tables,
            job_id,
            error,
            delay,
            claim_token,
        )
        .await
    }
    pub async fn renew_index_job_lease(
        &mut self,
        job_id: &str,
        claim_token: &str,
        lease_seconds: i64,
    ) -> PostgresStoreResult<bool> {
        renew_index_job_lease(
            &self.transaction,
            &self.tables,
            job_id,
            claim_token,
            lease_seconds,
        )
        .await
    }
    pub async fn requeue_index_job_at_tail(
        &mut self,
        job_id: &str,
        payload_json: String,
        delay: i64,
        claim_token: Option<&str>,
    ) -> PostgresStoreResult<()> {
        requeue_index_job_at_tail(
            &self.transaction,
            &self.tables,
            job_id,
            payload_json,
            delay,
            claim_token,
        )
        .await
    }
    pub async fn alloc_event_seq(&mut self, namespace: &str) -> PostgresStoreResult<i64> {
        require_namespace(namespace)?;
        allocate_event_seq(&self.transaction, &self.tables, namespace).await
    }

    pub async fn next_global_seq(&mut self) -> PostgresStoreResult<i64> {
        next_global_seq(&self.transaction, &self.tables).await
    }

    pub async fn current_global_seq(&self) -> PostgresStoreResult<i64> {
        current_global_seq(&self.transaction, &self.tables).await
    }

    pub async fn next_user_seq(&mut self, user_id: &str) -> PostgresStoreResult<i64> {
        next_user_seq(&self.transaction, &self.tables, user_id).await
    }

    pub async fn current_user_seq(&self, user_id: &str) -> PostgresStoreResult<i64> {
        current_user_seq(&self.transaction, &self.tables, user_id).await
    }

    pub async fn set_user_seq(&mut self, user_id: &str, value: i64) -> PostgresStoreResult<()> {
        set_user_seq(&self.transaction, &self.tables, user_id, value).await
    }

    pub async fn get_index_applied_fingerprint(
        &self,
        namespace: &str,
        coalesce_key: &str,
    ) -> PostgresStoreResult<Option<String>> {
        get_index_applied_fingerprint(&self.transaction, &self.tables, namespace, coalesce_key)
            .await
    }

    pub async fn set_index_applied_fingerprint(
        &mut self,
        namespace: &str,
        coalesce_key: &str,
        applied_fingerprint: Option<String>,
        last_job_id: Option<String>,
    ) -> PostgresStoreResult<()> {
        set_index_applied_fingerprint(
            &self.transaction,
            &self.tables,
            namespace,
            coalesce_key,
            applied_fingerprint,
            last_job_id,
        )
        .await
    }

    pub async fn append_raw_entity_event(
        &mut self,
        namespace: &str,
        event: NewRawEntityEvent,
    ) -> PostgresStoreResult<AppendedRawEvent> {
        require_namespace(namespace)?;
        append_raw_entity_event(&self.transaction, &self.tables, namespace, event).await
    }

    pub async fn replay_raw_events(
        &self,
        namespace: &str,
        after_seq: i64,
        limit: usize,
    ) -> PostgresStoreResult<Vec<RawEntityEvent>> {
        require_namespace(namespace)?;
        replay_raw_events(&self.transaction, &self.tables, namespace, after_seq, limit).await
    }

    pub async fn latest_retained_event_seq(&self, namespace: &str) -> PostgresStoreResult<i64> {
        require_namespace(namespace)?;
        latest_retained_event_seq(&self.transaction, &self.tables, namespace).await
    }

    pub async fn prune_entity_events_after(
        &mut self,
        namespace: &str,
        to_seq: i64,
    ) -> PostgresStoreResult<u64> {
        require_namespace(namespace)?;
        prune_entity_events_after(&self.transaction, &self.tables, namespace, to_seq).await
    }

    pub async fn replay_cursor(
        &self,
        namespace: &str,
        consumer: &str,
    ) -> PostgresStoreResult<ReplayCursor> {
        require_namespace(namespace)?;
        replay_cursor(&self.transaction, &self.tables, namespace, consumer).await
    }

    pub async fn recover_entity_projection(
        &mut self,
        request: &EntityRecoveryRequest,
    ) -> PostgresStoreResult<EntityRecoveryReport> {
        recover_entity_projection_uow(self, request).await
    }

    pub async fn rebuild_entity_projection(
        &mut self,
        request: &EntityRebuildRequest,
    ) -> PostgresStoreResult<EntityRecoveryReport> {
        rebuild_entity_projection_uow(self, request).await
    }

    pub async fn strict_advance_replay_cursor(
        &mut self,
        namespace: &str,
        consumer: &str,
        last_seq: i64,
    ) -> PostgresStoreResult<ReplayCursor> {
        require_namespace(namespace)?;
        advisory_lock(&self.transaction, &cursor_lock_key(namespace, consumer), 1).await?;
        let latest = latest_retained_event_seq(&self.transaction, &self.tables, namespace).await?;
        if last_seq > latest {
            return Err(StoreError::CursorOutOfRange {
                cursor: last_seq,
                latest,
            }
            .into());
        }
        let current = replay_cursor(&self.transaction, &self.tables, namespace, consumer)
            .await?
            .last_seq;
        if last_seq < current {
            return Err(StoreError::CursorRegresses {
                current,
                requested: last_seq,
            }
            .into());
        }
        set_replay_cursor_legacy(
            &self.transaction,
            &self.tables,
            namespace,
            consumer,
            last_seq,
        )
        .await
    }

    pub async fn set_replay_cursor_legacy(
        &mut self,
        namespace: &str,
        consumer: &str,
        last_seq: i64,
    ) -> PostgresStoreResult<ReplayCursor> {
        require_namespace(namespace)?;
        set_replay_cursor_legacy(
            &self.transaction,
            &self.tables,
            namespace,
            consumer,
            last_seq,
        )
        .await
    }

    pub async fn replace_named_projection(
        &mut self,
        namespace: &str,
        key: &str,
        projection: NamedProjectionWrite,
    ) -> PostgresStoreResult<()> {
        replace_named_projection(&self.transaction, &self.tables, namespace, key, projection).await
    }

    pub async fn get_named_projection(
        &mut self,
        namespace: &str,
        key: &str,
    ) -> PostgresStoreResult<Option<NamedProjection>> {
        named_projection(&self.transaction, &self.tables, namespace, key).await
    }

    pub async fn lock_named_projection(
        &mut self,
        namespace: &str,
        key: &str,
    ) -> PostgresStoreResult<Option<NamedProjection>> {
        named_projection_for_update(&self.transaction, &self.tables, namespace, key).await
    }

    pub async fn compare_and_swap_named_projection(
        &mut self,
        namespace: &str,
        key: &str,
        expected_last_authoritative_seq: Option<i64>,
        expected_last_materialized_seq: Option<i64>,
        projection: NamedProjectionWrite,
    ) -> PostgresStoreResult<bool> {
        compare_and_swap_named_projection(
            &self.transaction,
            &self.tables,
            namespace,
            key,
            expected_last_authoritative_seq,
            expected_last_materialized_seq,
            projection,
        )
        .await
    }

    /// Atomically compare-and-swaps a set of named projections.
    pub async fn compare_and_swap_named_projections(
        &mut self,
        updates: Vec<NamedProjectionCas>,
    ) -> PostgresStoreResult<bool> {
        compare_and_swap_named_projections(&self.transaction, &self.tables, updates).await
    }

    pub async fn clear_named_projection(
        &mut self,
        namespace: &str,
        key: &str,
    ) -> PostgresStoreResult<()> {
        clear_named_projection(&self.transaction, &self.tables, namespace, key).await
    }

    pub async fn clear_projection_namespace(&mut self, namespace: &str) -> PostgresStoreResult<()> {
        clear_projection_namespace(&self.transaction, &self.tables, namespace).await
    }

    pub async fn put_workflow_design_snapshot(
        &mut self,
        workflow_id: &str,
        snapshot: WorkflowDesignSnapshotWrite,
    ) -> PostgresStoreResult<()> {
        put_workflow_design_snapshot(&self.transaction, &self.tables, workflow_id, snapshot).await
    }

    pub async fn get_workflow_design_snapshot(
        &self,
        workflow_id: &str,
        max_version: i64,
        schema_version: i64,
    ) -> PostgresStoreResult<Option<WorkflowDesignSnapshot>> {
        workflow_design_snapshot(
            &self.transaction,
            &self.tables,
            workflow_id,
            max_version,
            schema_version,
        )
        .await
    }

    pub async fn clear_workflow_design_snapshots(
        &mut self,
        workflow_id: &str,
    ) -> PostgresStoreResult<()> {
        clear_workflow_design_snapshots(&self.transaction, &self.tables, workflow_id).await
    }

    pub async fn put_workflow_design_delta(
        &mut self,
        workflow_id: &str,
        delta: WorkflowDesignDeltaWrite,
    ) -> PostgresStoreResult<()> {
        put_workflow_design_delta(&self.transaction, &self.tables, workflow_id, delta).await
    }

    pub async fn get_workflow_design_delta(
        &self,
        workflow_id: &str,
        version: i64,
        schema_version: i64,
    ) -> PostgresStoreResult<Option<WorkflowDesignDelta>> {
        workflow_design_delta(
            &self.transaction,
            &self.tables,
            workflow_id,
            version,
            schema_version,
        )
        .await
    }

    pub async fn clear_workflow_design_deltas(
        &mut self,
        workflow_id: &str,
    ) -> PostgresStoreResult<()> {
        clear_workflow_design_deltas(&self.transaction, &self.tables, workflow_id).await
    }

    pub async fn create_server_run(&mut self, run: ServerRunCreate) -> PostgresStoreResult<()> {
        create_server_run(&self.transaction, &self.tables, run).await
    }
    pub async fn append_server_run_event(
        &mut self,
        run_id: &str,
        event_type: &str,
        payload_json: String,
    ) -> PostgresStoreResult<ServerRunEvent> {
        append_server_run_event(
            &self.transaction,
            &self.tables,
            run_id,
            event_type,
            payload_json,
        )
        .await
    }
    pub async fn update_server_run(
        &mut self,
        run_id: &str,
        update: ServerRunUpdate,
    ) -> PostgresStoreResult<()> {
        update_server_run(&self.transaction, &self.tables, run_id, update).await
    }
    pub async fn request_server_run_cancel(&mut self, run_id: &str) -> PostgresStoreResult<()> {
        request_server_run_cancel(&self.transaction, &self.tables, run_id).await
    }

    pub async fn apply_recorded_runtime_transition(
        &mut self,
        transition: RecordedRuntimeTransition,
        abort_after_writes: bool,
    ) -> PostgresStoreResult<RecordedTransitionResult> {
        apply_recorded_runtime_transition(
            &self.transaction,
            &self.tables,
            transition,
            abort_after_writes,
        )
        .await
    }

    pub async fn apply_claimed_recorded_runtime_transition(
        &mut self,
        handoff: RecordedWorkerHandoff,
        transition: RecordedRuntimeTransition,
        abort_after_writes: bool,
    ) -> PostgresStoreResult<RecordedTransitionResult> {
        apply_claimed_recorded_runtime_transition(
            &self.transaction,
            &self.tables,
            handoff,
            transition,
            abort_after_writes,
        )
        .await
    }

    pub async fn apply_claimed_recorded_worker_effect(
        &mut self,
        handoff: RecordedWorkerHandoff,
        effect: RecordedWorkerSuccessEffect,
    ) -> PostgresStoreResult<RecordedTransitionResult> {
        apply_claimed_recorded_worker_effect(&self.transaction, &self.tables, handoff, effect).await
    }
}
