use super::*;

impl SqliteUnitOfWork<'_> {
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
}
