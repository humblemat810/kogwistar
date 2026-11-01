use super::super::*;

impl DocumentStore for SqliteStore {
    fn document_authority_capability(&self) -> DocumentAuthorityCapability {
        DocumentAuthorityCapability::ProjectionUpsert
    }

    async fn upsert_document_projection(&self, write: DocumentProjectionUpsert) -> StoreResult<()> {
        if write.scope.namespace.trim().is_empty() || write.record.id.trim().is_empty() {
            return Err(StoreError::Backend {
                backend: "sqlite-document".to_owned(),
                message: "document namespace and ID must be non-empty".to_owned(),
            });
        }
        if write.record.document.is_none() {
            return Err(StoreError::Backend {
                backend: "sqlite-document".to_owned(),
                message: "source document text is required".to_owned(),
            });
        }
        SqliteStore::upsert_source_document_projection(self, write).map_err(trait_error)
    }
}
