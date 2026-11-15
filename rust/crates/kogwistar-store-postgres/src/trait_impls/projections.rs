use super::super::*;

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
