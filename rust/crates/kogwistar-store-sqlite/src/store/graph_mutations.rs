use super::super::*;

impl SqliteStore {
    /// Roll back a source document and its graph contributions in one SQLite
    /// transaction. Python parity: `RollbackSubsystem.rollback_document`.
    pub fn rollback_document(
        &self,
        scope: GraphScope,
        document_id: String,
        operation_id: String,
        deleted_at: String,
        page_size: usize,
    ) -> SqliteStoreResult<AppliedDocumentRollback> {
        self.immediate_transaction(|uow| {
            uow.apply_document_rollback(&scope, &document_id, &operation_id, &deleted_at, page_size)
        })
    }

    pub fn apply_canonical_graph_mutation_batch(
        &self,
        batch: GraphMutationBatch,
    ) -> SqliteStoreResult<Vec<AppliedGraphMutation>> {
        self.immediate_transaction(|uow| uow.apply_canonical_graph_mutation_batch(batch))
    }
}
