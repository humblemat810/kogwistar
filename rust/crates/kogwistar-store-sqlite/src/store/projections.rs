use super::super::*;

impl SqliteStore {
    pub fn get_named_projection(
        &self,
        namespace: &str,
        key: &str,
    ) -> SqliteStoreResult<Option<NamedProjection>> {
        self.with_connection(|conn| named_projection(conn, namespace, key))
    }

    pub fn list_named_projections(
        &self,
        namespace: &str,
    ) -> SqliteStoreResult<Vec<NamedProjection>> {
        self.with_connection(|conn| named_projections(conn, namespace))
    }

    pub fn get_stage1_node_projection(
        &self,
        namespace: &str,
        key: &str,
    ) -> SqliteStoreResult<Option<NamedProjection>> {
        self.with_connection(|conn| stage1_node_projection(conn, namespace, key))
    }

    pub fn list_stage1_node_projections(
        &self,
        namespace: &str,
    ) -> SqliteStoreResult<Vec<NamedProjection>> {
        self.with_connection(|conn| stage1_node_projections(conn, namespace))
    }

    pub fn replace_stage1_node_projection(
        &self,
        namespace: &str,
        key: &str,
        projection: NamedProjectionWrite,
    ) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| {
            uow.replace_stage1_node_projection(namespace, key, projection)
        })
    }

    pub fn clear_stage1_node_projection(
        &self,
        namespace: &str,
        key: &str,
    ) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| uow.clear_stage1_node_projection(namespace, key))
    }

    pub fn clear_stage1_node_projection_if_current(
        &self,
        namespace: &str,
        key: &str,
        expected_last_authoritative_seq: i64,
    ) -> SqliteStoreResult<bool> {
        self.immediate_transaction(|uow| {
            uow.clear_stage1_node_projection_if_current(
                namespace,
                key,
                expected_last_authoritative_seq,
            )
        })
    }

    pub fn replace_named_projection(
        &self,
        namespace: &str,
        key: &str,
        projection: NamedProjectionWrite,
    ) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| uow.replace_named_projection(namespace, key, projection))
    }

    pub fn compare_and_swap_named_projection(
        &self,
        namespace: &str,
        key: &str,
        expected_last_authoritative_seq: Option<i64>,
        expected_last_materialized_seq: Option<i64>,
        projection: NamedProjectionWrite,
    ) -> SqliteStoreResult<bool> {
        self.immediate_transaction(|uow| {
            uow.compare_and_swap_named_projection(
                namespace,
                key,
                expected_last_authoritative_seq,
                expected_last_materialized_seq,
                projection,
            )
        })
    }

    pub fn compare_and_swap_named_projections(
        &self,
        updates: Vec<NamedProjectionCas>,
    ) -> SqliteStoreResult<bool> {
        self.immediate_transaction(|uow| uow.compare_and_swap_named_projections(updates))
    }

    pub fn clear_named_projection(&self, namespace: &str, key: &str) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| uow.clear_named_projection(namespace, key))
    }

    pub fn clear_projection_namespace(&self, namespace: &str) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| uow.clear_projection_namespace(namespace))
    }
}
