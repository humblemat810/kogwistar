use super::*;

impl SqliteUnitOfWork<'_> {
    pub fn put_workflow_design_snapshot(
        &mut self,
        workflow_id: &str,
        snapshot: WorkflowDesignSnapshotWrite,
    ) -> SqliteStoreResult<()> {
        put_workflow_design_snapshot(self.transaction, workflow_id, snapshot)
    }

    pub fn get_workflow_design_snapshot(
        &self,
        workflow_id: &str,
        max_version: i64,
        schema_version: i64,
    ) -> SqliteStoreResult<Option<WorkflowDesignSnapshot>> {
        workflow_design_snapshot(self.transaction, workflow_id, max_version, schema_version)
    }

    pub fn clear_workflow_design_snapshots(&mut self, workflow_id: &str) -> SqliteStoreResult<()> {
        clear_workflow_design_snapshots(self.transaction, workflow_id)
    }

    pub fn put_workflow_design_delta(
        &mut self,
        workflow_id: &str,
        delta: WorkflowDesignDeltaWrite,
    ) -> SqliteStoreResult<()> {
        put_workflow_design_delta(self.transaction, workflow_id, delta)
    }

    pub fn get_workflow_design_delta(
        &self,
        workflow_id: &str,
        version: i64,
        schema_version: i64,
    ) -> SqliteStoreResult<Option<WorkflowDesignDelta>> {
        workflow_design_delta(self.transaction, workflow_id, version, schema_version)
    }

    pub fn clear_workflow_design_deltas(&mut self, workflow_id: &str) -> SqliteStoreResult<()> {
        clear_workflow_design_deltas(self.transaction, workflow_id)
    }

    pub fn create_server_run(&mut self, run: ServerRunCreate) -> SqliteStoreResult<()> {
        create_server_run(self.transaction, run)
    }
    pub fn append_server_run_event(
        &mut self,
        run_id: &str,
        event_type: &str,
        payload_json: String,
    ) -> SqliteStoreResult<ServerRunEvent> {
        append_server_run_event(self.transaction, run_id, event_type, payload_json)
    }
    pub fn update_server_run(
        &mut self,
        run_id: &str,
        update: ServerRunUpdate,
    ) -> SqliteStoreResult<()> {
        update_server_run(self.transaction, run_id, update)
    }
    pub fn request_server_run_cancel(&mut self, run_id: &str) -> SqliteStoreResult<()> {
        request_server_run_cancel(self.transaction, run_id)
    }
}
