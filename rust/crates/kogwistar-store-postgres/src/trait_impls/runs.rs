use super::super::*;

impl ServerRunReadStore for PostgresStore {
    async fn server_run(&self, run_id: &str) -> StoreResult<Option<ServerRun>> {
        PostgresStore::get_server_run(self, run_id)
            .await
            .map_err(trait_error)
    }

    async fn server_runs(
        &self,
        status: Option<&str>,
        workflow_id: Option<&str>,
        conversation_id: Option<&str>,
        limit: usize,
    ) -> StoreResult<Vec<ServerRun>> {
        PostgresStore::list_server_runs(self, status, workflow_id, conversation_id, limit)
            .await
            .map_err(trait_error)
    }

    async fn server_run_events(
        &self,
        run_id: &str,
        after_seq: i64,
        limit: usize,
    ) -> StoreResult<Vec<ServerRunEvent>> {
        PostgresStore::list_server_run_events(self, run_id, after_seq, limit)
            .await
            .map_err(trait_error)
    }
}

impl ServerRunWriteStore for PostgresStore {
    async fn create_server_run(&self, run: ServerRunCreate) -> StoreResult<()> {
        PostgresStore::create_server_run(self, run)
            .await
            .map_err(trait_error)
    }

    async fn append_server_run_event(
        &self,
        run_id: &str,
        event_type: &str,
        payload_json: String,
    ) -> StoreResult<ServerRunEvent> {
        PostgresStore::append_server_run_event(self, run_id, event_type, payload_json)
            .await
            .map_err(trait_error)
    }

    async fn update_server_run(&self, run_id: &str, update: ServerRunUpdate) -> StoreResult<()> {
        PostgresStore::update_server_run(self, run_id, update)
            .await
            .map_err(trait_error)
    }

    async fn request_server_run_cancel(&self, run_id: &str) -> StoreResult<()> {
        PostgresStore::request_server_run_cancel(self, run_id)
            .await
            .map_err(trait_error)
    }
}
