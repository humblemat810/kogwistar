use super::*;

impl SqliteUnitOfWork<'_> {
    /// Read ACL records from the canonical event journal inside this UOW.
    pub fn graph_acl_records_for_entities(
        &self,
        scope: &kogwistar_store::GraphScope,
        truth_graph: &str,
        entity_ids: &[String],
    ) -> SqliteStoreResult<Vec<GraphRecord>> {
        crate::acl_records::graph_acl_records_for_entities(
            self.transaction,
            scope,
            truth_graph,
            entity_ids,
        )
    }
}
