use super::*;

impl PostgresUnitOfWork<'_> {
    pub async fn replace_named_projection(
        &mut self,
        namespace: &str,
        key: &str,
        projection: NamedProjectionWrite,
    ) -> PostgresStoreResult<()> {
        replace_named_projection(&self.transaction, &self.tables, namespace, key, projection).await
    }

    pub async fn get_named_projection(
        &mut self,
        namespace: &str,
        key: &str,
    ) -> PostgresStoreResult<Option<NamedProjection>> {
        named_projection(&self.transaction, &self.tables, namespace, key).await
    }

    pub async fn lock_named_projection(
        &mut self,
        namespace: &str,
        key: &str,
    ) -> PostgresStoreResult<Option<NamedProjection>> {
        named_projection_for_update(&self.transaction, &self.tables, namespace, key).await
    }

    pub async fn compare_and_swap_named_projection(
        &mut self,
        namespace: &str,
        key: &str,
        expected_last_authoritative_seq: Option<i64>,
        expected_last_materialized_seq: Option<i64>,
        projection: NamedProjectionWrite,
    ) -> PostgresStoreResult<bool> {
        compare_and_swap_named_projection(
            &self.transaction,
            &self.tables,
            namespace,
            key,
            expected_last_authoritative_seq,
            expected_last_materialized_seq,
            projection,
        )
        .await
    }

    /// Atomically compare-and-swaps a set of named projections.
    pub async fn compare_and_swap_named_projections(
        &mut self,
        updates: Vec<NamedProjectionCas>,
    ) -> PostgresStoreResult<bool> {
        compare_and_swap_named_projections(&self.transaction, &self.tables, updates).await
    }

    pub async fn clear_named_projection(
        &mut self,
        namespace: &str,
        key: &str,
    ) -> PostgresStoreResult<()> {
        clear_named_projection(&self.transaction, &self.tables, namespace, key).await
    }

    pub async fn clear_projection_namespace(&mut self, namespace: &str) -> PostgresStoreResult<()> {
        clear_projection_namespace(&self.transaction, &self.tables, namespace).await
    }
}
