use super::super::*;

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
