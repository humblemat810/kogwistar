use super::super::*;

impl WorkflowDesignHistoryReadStore for PostgresStore {
    async fn workflow_design_snapshot(
        &self,
        workflow_id: &str,
        max_version: i64,
        schema_version: i64,
    ) -> StoreResult<Option<WorkflowDesignSnapshot>> {
        PostgresStore::get_workflow_design_snapshot(self, workflow_id, max_version, schema_version)
            .await
            .map_err(trait_error)
    }

    async fn workflow_design_delta(
        &self,
        workflow_id: &str,
        version: i64,
        schema_version: i64,
    ) -> StoreResult<Option<WorkflowDesignDelta>> {
        PostgresStore::get_workflow_design_delta(self, workflow_id, version, schema_version)
            .await
            .map_err(trait_error)
    }
}

impl WorkflowDesignHistoryWriteStore for PostgresStore {
    async fn put_workflow_design_snapshot(
        &self,
        workflow_id: &str,
        snapshot: WorkflowDesignSnapshotWrite,
    ) -> StoreResult<()> {
        PostgresStore::put_workflow_design_snapshot(self, workflow_id, snapshot)
            .await
            .map_err(trait_error)
    }

    async fn clear_workflow_design_snapshots(&self, workflow_id: &str) -> StoreResult<()> {
        PostgresStore::clear_workflow_design_snapshots(self, workflow_id)
            .await
            .map_err(trait_error)
    }

    async fn put_workflow_design_delta(
        &self,
        workflow_id: &str,
        delta: WorkflowDesignDeltaWrite,
    ) -> StoreResult<()> {
        PostgresStore::put_workflow_design_delta(self, workflow_id, delta)
            .await
            .map_err(trait_error)
    }

    async fn clear_workflow_design_deltas(&self, workflow_id: &str) -> StoreResult<()> {
        PostgresStore::clear_workflow_design_deltas(self, workflow_id)
            .await
            .map_err(trait_error)
    }
}
