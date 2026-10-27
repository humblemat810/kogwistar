use super::super::*;

impl PostgresStore {
    pub async fn alloc_event_seq(&self, namespace: &str) -> PostgresStoreResult<i64> {
        require_namespace(namespace)?;
        let namespace = namespace.to_owned();
        self.transaction(move |uow| Box::pin(async move { uow.alloc_event_seq(&namespace).await }))
            .await
    }

    pub async fn next_global_seq(&self) -> PostgresStoreResult<i64> {
        self.transaction(|uow| Box::pin(async move { uow.next_global_seq().await }))
            .await
    }

    pub async fn current_global_seq(&self) -> PostgresStoreResult<i64> {
        let client = self.client().await?;
        current_global_seq(&**client, &self.tables).await
    }

    pub async fn next_user_seq(&self, user_id: &str) -> PostgresStoreResult<i64> {
        let user_id = user_id.to_owned();
        self.transaction(move |uow| Box::pin(async move { uow.next_user_seq(&user_id).await }))
            .await
    }

    pub async fn current_user_seq(&self, user_id: &str) -> PostgresStoreResult<i64> {
        let client = self.client().await?;
        current_user_seq(&**client, &self.tables, user_id).await
    }

    pub async fn set_user_seq(&self, user_id: &str, value: i64) -> PostgresStoreResult<()> {
        let user_id = user_id.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move { uow.set_user_seq(&user_id, value).await })
        })
        .await
    }

    pub async fn append_raw_entity_event(
        &self,
        namespace: &str,
        event: NewRawEntityEvent,
    ) -> PostgresStoreResult<AppendedRawEvent> {
        require_namespace(namespace)?;
        let namespace = namespace.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move { uow.append_raw_entity_event(&namespace, event).await })
        })
        .await
    }

    /// Raw replay is exclusive of `after_seq`, ordered ascending, and bounded.
    pub async fn replay_raw_events(
        &self,
        namespace: &str,
        after_seq: i64,
        limit: usize,
    ) -> PostgresStoreResult<Vec<RawEntityEvent>> {
        require_namespace(namespace)?;
        let client = self.client().await?;
        replay_raw_events(&**client, &self.tables, namespace, after_seq, limit).await
    }

    /// `MAX(seq)` over retained rows, not allocation state. Gaps are legal.
    pub async fn latest_retained_event_seq(&self, namespace: &str) -> PostgresStoreResult<i64> {
        require_namespace(namespace)?;
        let client = self.client().await?;
        latest_retained_event_seq(&**client, &self.tables, namespace).await
    }

    pub async fn prune_entity_events_after(
        &self,
        namespace: &str,
        to_seq: i64,
    ) -> PostgresStoreResult<u64> {
        require_namespace(namespace)?;
        let namespace = namespace.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move { uow.prune_entity_events_after(&namespace, to_seq).await })
        })
        .await
    }

    pub async fn replay_cursor(
        &self,
        namespace: &str,
        consumer: &str,
    ) -> PostgresStoreResult<ReplayCursor> {
        require_namespace(namespace)?;
        let client = self.client().await?;
        replay_cursor(&**client, &self.tables, namespace, consumer).await
    }

    /// Strict trait-compatible advance: monotonic and at most retained latest.
    pub async fn strict_advance_replay_cursor(
        &self,
        namespace: &str,
        consumer: &str,
        last_seq: i64,
    ) -> PostgresStoreResult<ReplayCursor> {
        require_namespace(namespace)?;
        let namespace = namespace.to_owned();
        let consumer = consumer.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.strict_advance_replay_cursor(&namespace, &consumer, last_seq)
                    .await
            })
        })
        .await
    }

    /// Python legacy `cursor_set`: overwrite without monotonic/range checks.
    pub async fn set_replay_cursor_legacy(
        &self,
        namespace: &str,
        consumer: &str,
        last_seq: i64,
    ) -> PostgresStoreResult<ReplayCursor> {
        require_namespace(namespace)?;
        let namespace = namespace.to_owned();
        let consumer = consumer.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.set_replay_cursor_legacy(&namespace, &consumer, last_seq)
                    .await
            })
        })
        .await
    }
}
