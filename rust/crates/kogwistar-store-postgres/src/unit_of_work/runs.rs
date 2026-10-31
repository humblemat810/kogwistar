use super::*;

impl PostgresUnitOfWork<'_> {
    pub async fn recorded_runtime_steps(
        &self,
        run_id: &str,
        step_seqs: &[i64],
    ) -> PostgresStoreResult<Vec<ServerRunEvent>> {
        recorded_runtime_steps(&self.transaction, &self.tables, run_id, step_seqs).await
    }

    pub async fn get_server_run(&self, run_id: &str) -> PostgresStoreResult<Option<ServerRun>> {
        server_run(&self.transaction, &self.tables, run_id).await
    }

    pub async fn create_server_run(&mut self, run: ServerRunCreate) -> PostgresStoreResult<()> {
        create_server_run(&self.transaction, &self.tables, run).await
    }
    pub async fn append_server_run_event(
        &mut self,
        run_id: &str,
        event_type: &str,
        payload_json: String,
    ) -> PostgresStoreResult<ServerRunEvent> {
        append_server_run_event(
            &self.transaction,
            &self.tables,
            run_id,
            event_type,
            payload_json,
        )
        .await
    }
    pub async fn update_server_run(
        &mut self,
        run_id: &str,
        update: ServerRunUpdate,
    ) -> PostgresStoreResult<()> {
        update_server_run(&self.transaction, &self.tables, run_id, update).await
    }
    pub async fn request_server_run_cancel(&mut self, run_id: &str) -> PostgresStoreResult<()> {
        request_server_run_cancel(&self.transaction, &self.tables, run_id).await
    }
}
