use super::super::*;

impl SqliteStore {
    pub fn apply_canonical_graph_mutation_batch(
        &self,
        batch: GraphMutationBatch,
    ) -> SqliteStoreResult<Vec<AppliedGraphMutation>> {
        self.immediate_transaction(|uow| uow.apply_canonical_graph_mutation_batch(batch))
    }
}
