use super::super::*;

impl ServerRunReadStore for SqliteStore {
    async fn server_run(&self, run_id: &str) -> StoreResult<Option<ServerRun>> {
        SqliteStore::get_server_run(self, run_id).map_err(trait_error)
    }

    async fn server_runs(
        &self,
        status: Option<&str>,
        workflow_id: Option<&str>,
        conversation_id: Option<&str>,
        limit: usize,
    ) -> StoreResult<Vec<ServerRun>> {
        SqliteStore::list_server_runs(self, status, workflow_id, conversation_id, limit)
            .map_err(trait_error)
    }

    async fn server_run_events(
        &self,
        run_id: &str,
        after_seq: i64,
        limit: usize,
    ) -> StoreResult<Vec<ServerRunEvent>> {
        SqliteStore::list_server_run_events(self, run_id, after_seq, limit).map_err(trait_error)
    }
}

impl ServerRunWriteStore for SqliteStore {
    async fn create_server_run(&self, run: ServerRunCreate) -> StoreResult<()> {
        SqliteStore::create_server_run(self, run).map_err(trait_error)
    }

    async fn append_server_run_event(
        &self,
        run_id: &str,
        event_type: &str,
        payload_json: String,
    ) -> StoreResult<ServerRunEvent> {
        SqliteStore::append_server_run_event(self, run_id, event_type, payload_json)
            .map_err(trait_error)
    }

    async fn update_server_run(&self, run_id: &str, update: ServerRunUpdate) -> StoreResult<()> {
        SqliteStore::update_server_run(self, run_id, update).map_err(trait_error)
    }

    async fn request_server_run_cancel(&self, run_id: &str) -> StoreResult<()> {
        SqliteStore::request_server_run_cancel(self, run_id).map_err(trait_error)
    }
}
