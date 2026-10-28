use super::*;

impl PostgresUnitOfWork<'_> {
    pub async fn put_workflow_design_snapshot(
        &mut self,
        workflow_id: &str,
        snapshot: WorkflowDesignSnapshotWrite,
    ) -> PostgresStoreResult<()> {
        put_workflow_design_snapshot(&self.transaction, &self.tables, workflow_id, snapshot).await
    }

    pub async fn get_workflow_design_snapshot(
        &self,
        workflow_id: &str,
        max_version: i64,
        schema_version: i64,
    ) -> PostgresStoreResult<Option<WorkflowDesignSnapshot>> {
        workflow_design_snapshot(
            &self.transaction,
            &self.tables,
            workflow_id,
            max_version,
            schema_version,
        )
        .await
    }

    pub async fn clear_workflow_design_snapshots(
        &mut self,
        workflow_id: &str,
    ) -> PostgresStoreResult<()> {
        clear_workflow_design_snapshots(&self.transaction, &self.tables, workflow_id).await
    }

    pub async fn put_workflow_design_delta(
        &mut self,
        workflow_id: &str,
        delta: WorkflowDesignDeltaWrite,
    ) -> PostgresStoreResult<()> {
        put_workflow_design_delta(&self.transaction, &self.tables, workflow_id, delta).await
    }

    pub async fn get_workflow_design_delta(
        &self,
        workflow_id: &str,
        version: i64,
        schema_version: i64,
    ) -> PostgresStoreResult<Option<WorkflowDesignDelta>> {
        workflow_design_delta(
            &self.transaction,
            &self.tables,
            workflow_id,
            version,
            schema_version,
        )
        .await
    }

    pub async fn clear_workflow_design_deltas(
        &mut self,
        workflow_id: &str,
    ) -> PostgresStoreResult<()> {
        clear_workflow_design_deltas(&self.transaction, &self.tables, workflow_id).await
    }
}
