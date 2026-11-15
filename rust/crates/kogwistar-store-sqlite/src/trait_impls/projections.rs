use super::super::*;

impl ProjectionReadStore for SqliteStore {
    async fn named_projection(
        &self,
        namespace: &str,
        key: &str,
    ) -> StoreResult<Option<NamedProjection>> {
        require_trait_namespace(namespace)?;
        self.get_named_projection(namespace, key)
            .map_err(trait_error)
    }

    async fn named_projections(&self, namespace: &str) -> StoreResult<Vec<NamedProjection>> {
        require_trait_namespace(namespace)?;
        self.list_named_projections(namespace).map_err(trait_error)
    }
}

impl ProjectionWriteStore for SqliteStore {
    async fn replace_named_projection(
        &self,
        namespace: &str,
        key: &str,
        projection: NamedProjectionWrite,
    ) -> StoreResult<()> {
        require_trait_namespace(namespace)?;
        SqliteStore::replace_named_projection(self, namespace, key, projection).map_err(trait_error)
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
        SqliteStore::compare_and_swap_named_projection(
            self,
            namespace,
            key,
            expected_last_authoritative_seq,
            expected_last_materialized_seq,
            projection,
        )
        .map_err(trait_error)
    }

    async fn clear_named_projection(&self, namespace: &str, key: &str) -> StoreResult<()> {
        require_trait_namespace(namespace)?;
        SqliteStore::clear_named_projection(self, namespace, key).map_err(trait_error)
    }

    async fn clear_projection_namespace(&self, namespace: &str) -> StoreResult<()> {
        require_trait_namespace(namespace)?;
        SqliteStore::clear_projection_namespace(self, namespace).map_err(trait_error)
    }
}
