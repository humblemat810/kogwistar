use super::*;

impl PostgresUnitOfWork<'_> {
    /// Read the latest canonical event for each requested entity in this UoW.
    pub async fn latest_entity_events(
        &self,
        namespace: &str,
        entity_kind: &str,
        entity_ids: &[String],
    ) -> PostgresStoreResult<std::collections::BTreeMap<String, RawEntityEvent>> {
        require_namespace(namespace)?;
        if !matches!(entity_kind, "node" | "edge") {
            return Err(PostgresStoreError::Backend(
                "latest entity event lookup requires node or edge kind".to_owned(),
            ));
        }
        if entity_ids.is_empty() {
            return Ok(std::collections::BTreeMap::new());
        }
        if entity_ids.iter().any(|id| id.trim().is_empty()) {
            return Err(PostgresStoreError::Backend(
                "latest entity event lookup requires non-empty entity IDs".to_owned(),
            ));
        }
        let rows = self
            .transaction
            .query(
                &format!(
                    "SELECT namespace, seq, event_id, entity_kind, entity_id, op, \
                            payload_json, created_at::text \
                     FROM {} WHERE namespace=$1 AND entity_kind=$2 AND entity_id=ANY($3) \
                     ORDER BY entity_id ASC, seq DESC",
                    self.tables.entity_events
                ),
                &[&namespace, &entity_kind, &entity_ids],
            )
            .await
            .map_err(backend)?;
        let mut latest = std::collections::BTreeMap::new();
        for row in rows {
            let entity_id: String = row.try_get(4).map_err(backend)?;
            if latest.contains_key(&entity_id) {
                continue;
            }
            latest.insert(
                entity_id.clone(),
                RawEntityEvent {
                    namespace: row.try_get(0).map_err(backend)?,
                    seq: row.try_get(1).map_err(backend)?,
                    event_id: row.try_get(2).map_err(backend)?,
                    entity_kind: row.try_get(3).map_err(backend)?,
                    entity_id,
                    op: row.try_get(5).map_err(backend)?,
                    payload_json: row.try_get(6).map_err(backend)?,
                    created_at: row.try_get(7).map_err(backend)?,
                },
            );
        }
        Ok(latest)
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

    pub async fn append_restored_raw_entity_event(
        &mut self,
        namespace: &str,
        event: RestoredRawEntityEvent,
    ) -> PostgresStoreResult<AppendedRawEvent> {
        append_restored_raw_entity_event(&self.transaction, &self.tables, namespace, event).await
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
