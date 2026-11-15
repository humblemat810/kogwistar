use super::super::*;

impl DocumentStore for SqliteStore {
    fn supports_document_projection_lookup(&self) -> bool {
        true
    }

    async fn document_projection_ids(
        &self,
        scope: GraphScope,
        ids: Vec<String>,
    ) -> StoreResult<Vec<String>> {
        let mut found = Vec::new();
        for id in ids {
            if SqliteStore::source_document_projection(self, &scope, &id)
                .map_err(trait_error)?
                .is_some()
            {
                found.push(id);
            }
        }
        Ok(found)
    }

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
