use super::super::*;

impl SqliteStore {
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
}
