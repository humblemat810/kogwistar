use super::*;

impl PostgresUnitOfWork<'_> {
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
}
