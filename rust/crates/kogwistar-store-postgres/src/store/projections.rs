use super::super::*;

impl PostgresStore {
    /// Read Python-compatible Stage-1 content for native PostgreSQL workers.
    pub async fn stage1_graph_projection(
        &self,
        namespace: &str,
        entity_kind: &str,
        entity_id: &str,
    ) -> PostgresStoreResult<Option<Stage1GraphProjection>> {
        require_namespace(namespace)?;
        if !matches!(entity_kind, "node" | "edge") || entity_id.trim().is_empty() {
            return Err(PostgresStoreError::Backend(
                "Stage-1 lookup requires node/edge kind and non-empty entity id".to_owned(),
            ));
        }
        let client = self.client().await?;
        let row = client
            .query_opt(
                &format!(
                    "SELECT namespace, entity_kind, entity_id, document, metadata::TEXT, \
                            source_fingerprint, revision, materialization_status \
                     FROM {} WHERE namespace=$1 AND entity_kind=$2 AND entity_id=$3",
                    self.tables.stage1_projections
                ),
                &[&namespace, &entity_kind, &entity_id],
            )
            .await
            .map_err(backend)?;
        row.map(|row| {
            let metadata_json: String = row.try_get(4).map_err(backend)?;
            Ok(Stage1GraphProjection {
                namespace: row.try_get(0).map_err(backend)?,
                entity_kind: row.try_get(1).map_err(backend)?,
                entity_id: row.try_get(2).map_err(backend)?,
                document: row.try_get(3).map_err(backend)?,
                metadata: serde_json::from_str(&metadata_json)?,
                source_fingerprint: row.try_get(5).map_err(backend)?,
                revision: row.try_get(6).map_err(backend)?,
                materialization_status: row.try_get(7).map_err(backend)?,
            })
        })
        .transpose()
    }

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
