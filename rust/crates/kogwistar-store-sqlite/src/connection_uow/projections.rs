use super::*;

impl SqliteUnitOfWork<'_> {
    pub fn replace_named_projection(
        &mut self,
        namespace: &str,
        key: &str,
        projection: NamedProjectionWrite,
    ) -> SqliteStoreResult<()> {
        replace_named_projection(self.transaction, namespace, key, projection)
    }

    pub fn replace_stage1_node_projection(
        &mut self,
        namespace: &str,
        key: &str,
        projection: NamedProjectionWrite,
    ) -> SqliteStoreResult<()> {
        replace_stage1_node_projection(self.transaction, namespace, key, projection)
    }

    pub fn get_named_projection(
        &mut self,
        namespace: &str,
        key: &str,
    ) -> SqliteStoreResult<Option<NamedProjection>> {
        named_projection(self.transaction, namespace, key)
    }

    pub fn compare_and_swap_named_projection(
        &mut self,
        namespace: &str,
        key: &str,
        expected_last_authoritative_seq: Option<i64>,
        expected_last_materialized_seq: Option<i64>,
        projection: NamedProjectionWrite,
    ) -> SqliteStoreResult<bool> {
        compare_and_swap_named_projection(
            self.transaction,
            namespace,
            key,
            expected_last_authoritative_seq,
            expected_last_materialized_seq,
            projection,
        )
    }

    pub fn compare_and_swap_named_projections(
        &mut self,
        updates: Vec<NamedProjectionCas>,
    ) -> SqliteStoreResult<bool> {
        compare_and_swap_named_projections(self.transaction, updates)
    }

    pub fn clear_named_projection(&mut self, namespace: &str, key: &str) -> SqliteStoreResult<()> {
        clear_named_projection(self.transaction, namespace, key)
    }

    pub fn clear_stage1_node_projection(
        &mut self,
        namespace: &str,
        key: &str,
    ) -> SqliteStoreResult<()> {
        clear_stage1_node_projection(self.transaction, namespace, key)
    }

    pub fn clear_projection_namespace(&mut self, namespace: &str) -> SqliteStoreResult<()> {
        clear_projection_namespace(self.transaction, namespace)
    }
}
