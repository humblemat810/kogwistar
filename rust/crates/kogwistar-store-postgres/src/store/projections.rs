use super::super::*;

impl PostgresStore {
    pub async fn get_index_applied_fingerprint(
        &self,
        namespace: &str,
        coalesce_key: &str,
    ) -> PostgresStoreResult<Option<String>> {
        let client = self.client().await?;
        get_index_applied_fingerprint(&**client, &self.tables, namespace, coalesce_key).await
    }

    pub async fn set_index_applied_fingerprint(
        &self,
        namespace: &str,
        coalesce_key: &str,
        applied_fingerprint: Option<String>,
        last_job_id: Option<String>,
    ) -> PostgresStoreResult<()> {
        let namespace = namespace.to_owned();
        let coalesce_key = coalesce_key.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.set_index_applied_fingerprint(
                    &namespace,
                    &coalesce_key,
                    applied_fingerprint,
                    last_job_id,
                )
                .await
            })
        })
        .await
    }

    /// Bounded operator-triggered recovery. Projection replacement and cursor
    /// movement are committed only together with the authoritative replay fold.
    pub async fn recover_entity_projection(
        &self,
        request: EntityRecoveryRequest,
    ) -> PostgresStoreResult<EntityRecoveryReport> {
        validate_entity_recovery_request(&request)?;
        self.transaction(move |uow| {
            Box::pin(async move { uow.recover_entity_projection(&request).await })
        })
        .await
    }

    /// Explicit full rebuild. `ensure_schema` and opening a store never call it.
    pub async fn rebuild_entity_projection(
        &self,
        request: EntityRebuildRequest,
    ) -> PostgresStoreResult<EntityRecoveryReport> {
        validate_entity_rebuild_request(&request)?;
        self.transaction(move |uow| {
            Box::pin(async move { uow.rebuild_entity_projection(&request).await })
        })
        .await
    }

    pub async fn get_named_projection(
        &self,
        namespace: &str,
        key: &str,
    ) -> PostgresStoreResult<Option<NamedProjection>> {
        require_namespace(namespace)?;
        let client = self.client().await?;
        named_projection(&**client, &self.tables, namespace, key).await
    }

    pub async fn list_named_projections(
        &self,
        namespace: &str,
    ) -> PostgresStoreResult<Vec<NamedProjection>> {
        require_namespace(namespace)?;
        let client = self.client().await?;
        named_projections(&**client, &self.tables, namespace).await
    }

    pub async fn replace_named_projection(
        &self,
        namespace: &str,
        key: &str,
        projection: NamedProjectionWrite,
    ) -> PostgresStoreResult<()> {
        require_namespace(namespace)?;
        let namespace = namespace.to_owned();
        let key = key.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.replace_named_projection(&namespace, &key, projection)
                    .await
            })
        })
        .await
    }

    pub async fn compare_and_swap_named_projection(
        &self,
        namespace: &str,
        key: &str,
        expected_last_authoritative_seq: Option<i64>,
        expected_last_materialized_seq: Option<i64>,
        projection: NamedProjectionWrite,
    ) -> PostgresStoreResult<bool> {
        require_namespace(namespace)?;
        let namespace = namespace.to_owned();
        let key = key.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.compare_and_swap_named_projection(
                    &namespace,
                    &key,
                    expected_last_authoritative_seq,
                    expected_last_materialized_seq,
                    projection,
                )
                .await
            })
        })
        .await
    }

    /// Atomically compare-and-swaps a set of named projections.
    pub async fn compare_and_swap_named_projections(
        &self,
        updates: Vec<NamedProjectionCas>,
    ) -> PostgresStoreResult<bool> {
        self.transaction(move |uow| {
            Box::pin(async move { uow.compare_and_swap_named_projections(updates).await })
        })
        .await
    }

    pub async fn clear_named_projection(
        &self,
        namespace: &str,
        key: &str,
    ) -> PostgresStoreResult<()> {
        require_namespace(namespace)?;
        let namespace = namespace.to_owned();
        let key = key.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move { uow.clear_named_projection(&namespace, &key).await })
        })
        .await
    }

    pub async fn clear_projection_namespace(&self, namespace: &str) -> PostgresStoreResult<()> {
        require_namespace(namespace)?;
        let namespace = namespace.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move { uow.clear_projection_namespace(&namespace).await })
        })
        .await
    }
}
