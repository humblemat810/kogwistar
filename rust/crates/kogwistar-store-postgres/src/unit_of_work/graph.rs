use super::*;

impl PostgresUnitOfWork<'_> {
    pub async fn lock_entity_identity(
        &self,
        namespace: &str,
        entity_kind: &str,
        entity_id: &str,
    ) -> PostgresStoreResult<()> {
        let key = format!("{namespace}\u{1f}{entity_kind}\u{1f}{entity_id}");
        self.transaction
            .query_one(
                "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                &[&key],
            )
            .await
            .map_err(backend)?;
        Ok(())
    }

    pub async fn lock_runtime_queue_admission(&self) -> PostgresStoreResult<()> {
        advisory_lock(
            &self.transaction,
            &format!(
                "runtime-queue-admission\u{1f}{}",
                self.tables.projected_lane_messages
            ),
            73,
        )
        .await
    }

    pub async fn apply_graph_mutation(
        &mut self,
        mutation: GraphMutation,
    ) -> PostgresStoreResult<AppliedGraphMutation> {
        validate_graph_mutation(&mutation)?;
        apply_graph_mutation(&self.transaction, &self.tables, mutation).await
    }

    pub async fn apply_graph_metadata_patch_mutation(
        &mut self,
        mutation: GraphMetadataPatchMutation,
    ) -> PostgresStoreResult<Option<AppliedGraphMutation>> {
        apply_graph_metadata_patch_mutation(&self.transaction, &self.tables, mutation).await
    }

    pub async fn apply_graph_delete_mutation(
        &mut self,
        mutation: GraphDeleteMutation,
    ) -> PostgresStoreResult<Option<AppliedGraphMutation>> {
        apply_graph_delete_mutation(&self.transaction, &self.tables, mutation).await
    }

    pub async fn upsert_graph_projection(
        &mut self,
        write: GraphProjectionUpsert,
    ) -> PostgresStoreResult<()> {
        upsert_graph_projection(&self.transaction, &self.tables, write).await
    }

    pub async fn promote_graph_embedding_projection(
        &mut self,
        namespace: &str,
        entity_kind: &str,
        mut candidate: GraphRecord,
    ) -> PostgresStoreResult<bool> {
        require_namespace(namespace)?;
        let table = match entity_kind {
            "node" => "gke_nodes",
            "edge" => "gke_edges",
            _ => {
                return Err(StoreError::InvalidEntityEventPayload {
                    message: "embedding promotion supports node or edge only".to_owned(),
                }
                .into());
            }
        };
        let embedding_dim = candidate
            .embedding
            .as_ref()
            .map(Vec::len)
            .filter(|dimension| *dimension > 0)
            .ok_or_else(|| {
                PostgresStoreError::Backend(
                    "embedding promotion requires a non-empty vector".to_owned(),
                )
            })?;
        let scope = GraphScope {
            namespace: namespace.to_owned(),
            workspace_id: None,
            graph_space: None,
        };
        let current = self
            .graph_projection_records_for_update(GraphProjectionRead {
                scope: scope.clone(),
                table: table.to_owned(),
                ids: Some(vec![candidate.id.clone()]),
                metadata: Default::default(),
                limit: 1,
            })
            .await?
            .into_iter()
            .next();
        let Some(mut current) = current else {
            return Ok(false);
        };
        if current
            .metadata
            .get("lifecycle_status")
            .and_then(Value::as_str)
            .is_some_and(|status| matches!(status, "tombstoned" | "deleted"))
        {
            return Ok(false);
        }
        let expected_metadata = graph_metadata(&scope, &candidate.metadata)?;
        if current.document != candidate.document || current.metadata != expected_metadata {
            return Ok(false);
        }

        // The provider is allowed to supply the vector only. Preserve the
        // canonical row values read and locked above.
        current.embedding = candidate.embedding.take();
        self.upsert_graph_projection(GraphProjectionUpsert {
            scope,
            table: table.to_owned(),
            record: current,
            embedding_dim,
        })
        .await?;
        Ok(true)
    }

    pub async fn patch_graph_projection_metadata(
        &mut self,
        patch: GraphProjectionMetadataPatch,
    ) -> PostgresStoreResult<bool> {
        patch_graph_projection_metadata(&self.transaction, &self.tables, patch).await
    }

    pub async fn graph_projection_records(
        &self,
        read: GraphProjectionRead,
    ) -> PostgresStoreResult<Vec<GraphRecord>> {
        graph_projection_records(&self.transaction, &self.tables, read, None, false).await
    }

    pub async fn graph_projection_records_for_update(
        &self,
        read: GraphProjectionRead,
    ) -> PostgresStoreResult<Vec<GraphRecord>> {
        graph_projection_records_with_lock(&self.transaction, &self.tables, read, None, false, true)
            .await
    }

    /// Read ACL truth through this unit of work's transaction. Callers that
    /// derive graph effects from ACL state must not use the store-level read
    /// helper, which may observe a different committed snapshot.
    pub async fn graph_acl_records_for_entities(
        &self,
        scope: &GraphScope,
        truth_graph: &str,
        entity_ids: &[String],
    ) -> PostgresStoreResult<Vec<GraphRecord>> {
        graph_acl_records_for_entities(
            &self.transaction,
            &self.tables,
            scope,
            truth_graph,
            entity_ids,
        )
        .await
    }

    pub async fn graph_projection_vector_query(
        &self,
        query: GraphProjectionVectorQuery,
    ) -> PostgresStoreResult<Vec<VectorMatch>> {
        graph_projection_vector_query(&self.transaction, &self.tables, query, &BTreeMap::new())
            .await
    }
}
