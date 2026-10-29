use super::*;

impl SqliteUnitOfWork<'_> {
    pub fn set_index_applied_fingerprint(
        &mut self,
        namespace: &str,
        coalesce_key: &str,
        applied_fingerprint: Option<&str>,
        last_job_id: Option<&str>,
    ) -> SqliteStoreResult<()> {
        self.transaction.execute(
            "INSERT INTO index_applied_state(namespace,coalesce_key,applied_fingerprint,applied_at,last_job_id) VALUES(?1,?2,?3,?4,?5) ON CONFLICT(namespace,coalesce_key) DO UPDATE SET applied_fingerprint=excluded.applied_fingerprint,applied_at=excluded.applied_at,last_job_id=excluded.last_job_id",
            params![namespace, coalesce_key, applied_fingerprint, unix_epoch_seconds(), last_job_id],
        )?;
        Ok(())
    }

    pub fn enqueue_index_job(&mut self, job: NewIndexJob) -> SqliteStoreResult<String> {
        enqueue_index_job(self.transaction, job)
    }
    pub fn claim_index_jobs(
        &mut self,
        limit: usize,
        lease_seconds: i64,
        namespace: Option<&str>,
    ) -> SqliteStoreResult<Vec<IndexJob>> {
        claim_index_jobs(self.transaction, limit, lease_seconds, namespace)
    }
    pub fn claim_index_jobs_filtered(
        &mut self,
        limit: usize,
        lease_seconds: i64,
        namespace: Option<&str>,
        index_kinds: &[String],
    ) -> SqliteStoreResult<Vec<IndexJob>> {
        claim_index_jobs_filtered(
            self.transaction,
            limit,
            lease_seconds,
            namespace,
            index_kinds,
        )
    }
    pub fn mark_index_job_done(
        &mut self,
        job_id: &str,
        claim_token: Option<&str>,
    ) -> SqliteStoreResult<bool> {
        mark_index_job_done(self.transaction, job_id, claim_token)
    }
    pub fn accept_index_job_result(
        &mut self,
        job_id: &str,
        claim_token: &str,
        result_json: &str,
        result_sha256: &str,
    ) -> SqliteStoreResult<AcceptedIndexJobResult> {
        accept_index_job_result(
            self.transaction,
            job_id,
            claim_token,
            result_json,
            result_sha256,
        )
    }
    pub fn mark_index_job_failed(
        &mut self,
        job_id: &str,
        error: &str,
        final_: bool,
        claim_token: Option<&str>,
    ) -> SqliteStoreResult<()> {
        mark_index_job_failed(self.transaction, job_id, error, final_, claim_token)
    }
    pub fn bump_retry_and_requeue(
        &mut self,
        job_id: &str,
        error: &str,
        delay_seconds: i64,
        claim_token: Option<&str>,
    ) -> SqliteStoreResult<()> {
        bump_retry_and_requeue(self.transaction, job_id, error, delay_seconds, claim_token)
    }
    pub fn renew_index_job_lease(
        &mut self,
        job_id: &str,
        claim_token: &str,
        lease_seconds: i64,
    ) -> SqliteStoreResult<bool> {
        renew_index_job_lease(self.transaction, job_id, claim_token, lease_seconds)
    }
    pub fn requeue_index_job_at_tail(
        &mut self,
        job_id: &str,
        payload_json: String,
        delay_seconds: i64,
        claim_token: Option<&str>,
    ) -> SqliteStoreResult<()> {
        requeue_index_job_at_tail(
            self.transaction,
            job_id,
            payload_json,
            delay_seconds,
            claim_token,
        )
    }
}
