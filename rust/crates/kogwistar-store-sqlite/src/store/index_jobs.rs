use super::super::*;

impl SqliteStore {
    pub fn enqueue_index_job(&self, job: NewIndexJob) -> SqliteStoreResult<String> {
        self.immediate_transaction(|uow| uow.enqueue_index_job(job))
    }
    pub fn claim_index_jobs(
        &self,
        limit: usize,
        lease_seconds: i64,
        namespace: Option<&str>,
    ) -> SqliteStoreResult<Vec<IndexJob>> {
        self.immediate_transaction(|uow| uow.claim_index_jobs(limit, lease_seconds, namespace))
    }
    pub fn claim_index_jobs_filtered(
        &self,
        limit: usize,
        lease_seconds: i64,
        filter: IndexJobClaimFilter,
    ) -> SqliteStoreResult<Vec<IndexJob>> {
        self.immediate_transaction(|uow| {
            uow.claim_index_jobs_filtered(
                limit,
                lease_seconds,
                filter.namespace.as_deref(),
                &filter.index_kinds,
            )
        })
    }
    pub fn mark_index_job_done(
        &self,
        job_id: &str,
        claim_token: Option<&str>,
    ) -> SqliteStoreResult<bool> {
        self.immediate_transaction(|uow| uow.mark_index_job_done(job_id, claim_token))
    }
    pub fn accept_index_job_result(
        &self,
        job_id: &str,
        claim_token: &str,
        result_json: &str,
        result_sha256: &str,
    ) -> SqliteStoreResult<AcceptedIndexJobResult> {
        self.immediate_transaction(|uow| {
            uow.accept_index_job_result(job_id, claim_token, result_json, result_sha256)
        })
    }
    pub fn index_job_result(
        &self,
        job_id: &str,
    ) -> SqliteStoreResult<Option<AcceptedIndexJobResult>> {
        self.with_connection(|conn| index_job_result(conn, job_id))
    }
    pub fn mark_index_job_failed(
        &self,
        job_id: &str,
        error: &str,
        final_: bool,
        claim_token: Option<&str>,
    ) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| {
            uow.mark_index_job_failed(job_id, error, final_, claim_token)
        })
    }
    pub fn bump_retry_and_requeue(
        &self,
        job_id: &str,
        error: &str,
        delay_seconds: i64,
        claim_token: Option<&str>,
    ) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| {
            uow.bump_retry_and_requeue(job_id, error, delay_seconds, claim_token)
        })
    }
    pub fn renew_index_job_lease(
        &self,
        job_id: &str,
        claim_token: &str,
        lease_seconds: i64,
    ) -> SqliteStoreResult<bool> {
        self.immediate_transaction(|uow| {
            uow.renew_index_job_lease(job_id, claim_token, lease_seconds)
        })
    }
    pub fn requeue_index_job_at_tail(
        &self,
        job_id: &str,
        payload_json: String,
        delay_seconds: i64,
        claim_token: Option<&str>,
    ) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| {
            uow.requeue_index_job_at_tail(job_id, payload_json, delay_seconds, claim_token)
        })
    }
    pub fn list_index_jobs(
        &self,
        namespace: Option<&str>,
        status: Option<&str>,
        entity_kind: Option<&str>,
        entity_id: Option<&str>,
        index_kind: Option<&str>,
        limit: usize,
    ) -> SqliteStoreResult<Vec<IndexJob>> {
        self.with_connection(|conn| {
            list_index_jobs(
                conn,
                namespace,
                status,
                entity_kind,
                entity_id,
                index_kind,
                limit,
            )
        })
    }
}
