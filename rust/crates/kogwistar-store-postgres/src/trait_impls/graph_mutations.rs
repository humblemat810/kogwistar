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

    async fn graph_incident_edges(
        &self,
        read: GraphIncidentEdgeRead,
    ) -> StoreResult<Vec<GraphRecord>> {
        PostgresStore::graph_incident_edges(self, read)
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

    async fn graph_projection_vector_query_nodes_as_of(
        &self,
        query: GraphProjectionVectorQuery,
        as_of_ts: &str,
        follow_redirects: bool,
        max_redirect_hops: usize,
    ) -> StoreResult<Vec<VectorMatch>> {
        let candidates =
            PostgresStore::graph_projection_vector_query_including_tombstones(self, query.clone())
                .await
                .map_err(trait_error)?;
        kogwistar_store::resolve_graph_projection_vector_candidates_as_of(
            self,
            &query,
            candidates,
            as_of_ts,
            follow_redirects,
            max_redirect_hops,
        )
        .await
    }
}

impl DocumentStore for PostgresStore {
    fn supports_document_projection_lookup(&self) -> bool {
        true
    }

    fn supports_document_projection_delete(&self) -> bool {
        true
    }

    fn supports_document_rollback(&self) -> bool {
        true
    }

    async fn document_projection_ids(
        &self,
        scope: GraphScope,
        ids: Vec<String>,
    ) -> StoreResult<Vec<String>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let limit = ids.len();
        let rows = self
            .graph_projection_records(GraphProjectionRead {
                scope,
                table: "gke_documents".to_owned(),
                ids: Some(ids),
                metadata: kogwistar_store::MetadataFilter::default(),
                limit,
            })
            .await
            .map_err(trait_error)?;
        Ok(rows.into_iter().map(|row| row.id).collect())
    }

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

    async fn delete_document_projection(
        &self,
        scope: GraphScope,
        document_id: String,
    ) -> StoreResult<bool> {
        PostgresStore::delete_document_projection(self, scope, document_id)
            .await
            .map_err(trait_error)
    }

    async fn delete_document_projections_by_source_id(
        &self,
        scope: GraphScope,
        source_document_id: String,
    ) -> StoreResult<Vec<String>> {
        PostgresStore::delete_document_projections_by_source_id(self, scope, source_document_id)
            .await
            .map_err(trait_error)
    }

    async fn rollback_document(
        &self,
        request: DocumentRollbackRequest,
    ) -> StoreResult<AppliedDocumentRollback> {
        PostgresStore::rollback_document(
            self,
            request.scope,
            request.document_id,
            request.operation_id,
            request.deleted_at,
            request.page_size,
        )
        .await
        .map_err(trait_error)
    }
}
