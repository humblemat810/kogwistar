use super::super::*;

impl WorkflowDesignHistoryReadStore for SqliteStore {
    async fn workflow_design_snapshot(
        &self,
        workflow_id: &str,
        max_version: i64,
        schema_version: i64,
    ) -> StoreResult<Option<WorkflowDesignSnapshot>> {
        SqliteStore::get_workflow_design_snapshot(self, workflow_id, max_version, schema_version)
            .map_err(trait_error)
    }

    async fn workflow_design_delta(
        &self,
        workflow_id: &str,
        version: i64,
        schema_version: i64,
    ) -> StoreResult<Option<WorkflowDesignDelta>> {
        SqliteStore::get_workflow_design_delta(self, workflow_id, version, schema_version)
            .map_err(trait_error)
    }
}

impl WorkflowDesignHistoryWriteStore for SqliteStore {
    async fn put_workflow_design_snapshot(
        &self,
        workflow_id: &str,
        snapshot: WorkflowDesignSnapshotWrite,
    ) -> StoreResult<()> {
        SqliteStore::put_workflow_design_snapshot(self, workflow_id, snapshot).map_err(trait_error)
    }

    async fn clear_workflow_design_snapshots(&self, workflow_id: &str) -> StoreResult<()> {
        SqliteStore::clear_workflow_design_snapshots(self, workflow_id).map_err(trait_error)
    }

    async fn put_workflow_design_delta(
        &self,
        workflow_id: &str,
        delta: WorkflowDesignDeltaWrite,
    ) -> StoreResult<()> {
        SqliteStore::put_workflow_design_delta(self, workflow_id, delta).map_err(trait_error)
    }

    async fn clear_workflow_design_deltas(&self, workflow_id: &str) -> StoreResult<()> {
        SqliteStore::clear_workflow_design_deltas(self, workflow_id).map_err(trait_error)
    }
}
