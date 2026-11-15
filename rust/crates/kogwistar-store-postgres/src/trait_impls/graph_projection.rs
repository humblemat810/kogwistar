use super::super::*;

impl GraphProjectionWriteStore for PostgresStore {
    async fn upsert_graph_projection_record(
        &self,
        namespace: &str,
        entity_kind: &str,
        record: GraphRecord,
    ) -> StoreResult<()> {
        require_trait_namespace(namespace)?;
        match entity_kind {
            "node" | "edge" => {}
            _ => {
                return Err(StoreError::InvalidEntityEventPayload {
                    message: "projection upsert supports node or edge only".to_owned(),
                });
            }
        }
        self.promote_graph_embedding_projection(
            namespace.to_owned(),
            entity_kind.to_owned(),
            record,
        )
        .await
        .map(|_| ())
        .map_err(trait_error)
    }

    async fn delete_graph_projection_record(
        &self,
        namespace: &str,
        entity_kind: &str,
        id: &str,
    ) -> StoreResult<bool> {
        self.delete_graph_projection_if_tombstoned(
            GraphScope {
                namespace: namespace.to_owned(),
                workspace_id: None,
                graph_space: None,
            },
            entity_kind.to_owned(),
            id.to_owned(),
        )
        .await
        .map_err(trait_error)
    }
}
