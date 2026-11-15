use super::super::*;

impl IndexJobReadStore for PostgresStore {
    async fn index_jobs(
        &self,
        namespace: Option<&str>,
        status: Option<&str>,
        entity_kind: Option<&str>,
        entity_id: Option<&str>,
        index_kind: Option<&str>,
        limit: usize,
    ) -> StoreResult<Vec<IndexJob>> {
        self.list_index_jobs(namespace, status, entity_kind, entity_id, index_kind, limit)
            .await
            .map_err(trait_error)
    }
}
impl IndexJobWriteStore for PostgresStore {
    async fn enqueue_index_job(&self, job: NewIndexJob) -> StoreResult<String> {
        PostgresStore::enqueue_index_job(self, job)
            .await
            .map_err(trait_error)
    }
    async fn claim_index_jobs(
        &self,
        limit: usize,
        lease_seconds: i64,
        namespace: Option<&str>,
    ) -> StoreResult<Vec<IndexJob>> {
        PostgresStore::claim_index_jobs(self, limit, lease_seconds, namespace)
            .await
            .map_err(trait_error)
    }
    async fn claim_index_jobs_filtered(
        &self,
        limit: usize,
        lease_seconds: i64,
        filter: IndexJobClaimFilter,
    ) -> StoreResult<Vec<IndexJob>> {
        PostgresStore::claim_index_jobs_filtered(self, limit, lease_seconds, filter)
            .await
            .map_err(trait_error)
    }
    async fn mark_index_job_done(
        &self,
        job_id: &str,
        claim_token: Option<&str>,
    ) -> StoreResult<bool> {
        PostgresStore::mark_index_job_done(self, job_id, claim_token)
            .await
            .map_err(trait_error)
    }
    async fn accept_index_job_result(
        &self,
        job_id: &str,
        claim_token: &str,
        result_json: &str,
        result_sha256: &str,
    ) -> StoreResult<AcceptedIndexJobResult> {
        PostgresStore::accept_index_job_result(
            self,
            job_id,
            claim_token,
            result_json,
            result_sha256,
        )
        .await
        .map_err(trait_error)
    }
    async fn index_job_result(&self, job_id: &str) -> StoreResult<Option<AcceptedIndexJobResult>> {
        PostgresStore::index_job_result(self, job_id)
            .await
            .map_err(trait_error)
    }
    async fn mark_index_job_failed(
        &self,
        job_id: &str,
        error: &str,
        final_: bool,
        claim_token: Option<&str>,
    ) -> StoreResult<()> {
        PostgresStore::mark_index_job_failed(self, job_id, error, final_, claim_token)
            .await
            .map_err(trait_error)
    }
    async fn bump_retry_and_requeue(
        &self,
        job_id: &str,
        error: &str,
        delay: i64,
        claim_token: Option<&str>,
    ) -> StoreResult<()> {
        PostgresStore::bump_retry_and_requeue(self, job_id, error, delay, claim_token)
            .await
            .map_err(trait_error)
    }
    async fn renew_index_job_lease(
        &self,
        job_id: &str,
        claim_token: &str,
        lease_seconds: i64,
    ) -> StoreResult<bool> {
        PostgresStore::renew_index_job_lease(self, job_id, claim_token, lease_seconds)
            .await
            .map_err(trait_error)
    }
    async fn requeue_index_job_at_tail(
        &self,
        job_id: &str,
        payload_json: String,
        delay: i64,
        claim_token: Option<&str>,
    ) -> StoreResult<()> {
        PostgresStore::requeue_index_job_at_tail(self, job_id, payload_json, delay, claim_token)
            .await
            .map_err(trait_error)
    }
}
