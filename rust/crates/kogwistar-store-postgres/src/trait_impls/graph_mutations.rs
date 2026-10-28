use super::super::*;

impl GraphMutationStore for PostgresStore {
    fn graph_authority_capability(&self) -> GraphAuthorityCapability {
        GraphAuthorityCapability::AtomicBatchWithIndexJobs
    }

    async fn apply_graph_mutation(
        &self,
        mutation: GraphMutation,
    ) -> StoreResult<AppliedGraphMutation> {
        PostgresStore::apply_graph_mutation(self, mutation)
            .await
            .map_err(trait_error)
    }

    async fn apply_graph_mutations_atomic(
        &self,
        mutations: Vec<GraphMutation>,
    ) -> StoreResult<Vec<AppliedGraphMutation>> {
        PostgresStore::apply_graph_mutations_atomic(self, mutations)
            .await
            .map_err(trait_error)
    }

    async fn apply_graph_mutations_with_index_jobs_atomic(
        &self,
        batch: GraphMutationBatch,
    ) -> StoreResult<Vec<AppliedGraphMutation>> {
        PostgresStore::apply_graph_mutations_with_index_jobs_atomic(self, batch)
            .await
            .map_err(trait_error)
    }

    async fn graph_projection_records(
        &self,
        read: GraphProjectionRead,
    ) -> StoreResult<Vec<GraphRecord>> {
        PostgresStore::graph_projection_records(self, read)
            .await
            .map_err(trait_error)
    }

    async fn graph_projection_page(
        &self,
        page: GraphProjectionPageRead,
    ) -> StoreResult<Vec<GraphRecord>> {
        PostgresStore::graph_projection_page(self, page)
            .await
            .map_err(trait_error)
    }

    async fn graph_projection_vector_query(
        &self,
        query: GraphProjectionVectorQuery,
    ) -> StoreResult<Vec<VectorMatch>> {
        PostgresStore::graph_projection_vector_query(self, query)
            .await
            .map_err(trait_error)
    }
}

impl DocumentStore for PostgresStore {
    fn document_authority_capability(&self) -> DocumentAuthorityCapability {
        DocumentAuthorityCapability::ProjectionUpsert
    }

    async fn upsert_document_projection(&self, write: DocumentProjectionUpsert) -> StoreResult<()> {
        self.upsert_graph_projection(GraphProjectionUpsert {
            scope: write.scope,
            table: "gke_documents".to_owned(),
            record: write.record,
            embedding_dim: write.embedding_dim,
        })
        .await
        .map_err(trait_error)
    }
}
