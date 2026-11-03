use super::super::*;

impl NodeIndexStore for PostgresStore {
    async fn upsert_node_index(&self, index: &str, records: Vec<GraphRecord>) -> StoreResult<()> {
        PostgresStore::upsert_node_index(self, index, records)
            .await
            .map_err(trait_error)
    }

    async fn get_node_index(&self, index: &str, ids: &[String]) -> StoreResult<Vec<GraphRecord>> {
        PostgresStore::get_node_index(self, index, ids)
            .await
            .map_err(trait_error)
    }

    async fn query_node_index(
        &self,
        index: &str,
        query: &VectorQuery,
    ) -> StoreResult<Vec<VectorMatch>> {
        PostgresStore::query_node_index(self, index, query)
            .await
            .map_err(trait_error)
    }
}
