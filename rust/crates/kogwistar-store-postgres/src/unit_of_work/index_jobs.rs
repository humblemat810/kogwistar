use super::*;

impl PostgresUnitOfWork<'_> {
    pub async fn enqueue_index_job(&mut self, job: NewIndexJob) -> PostgresStoreResult<String> {
        require_namespace(&job.namespace)?;
        advisory_lock(
            &self.transaction,
            &format!(
                "index-job\u{1f}{}\u{1f}{}",
                job.namespace,
                job.coalesce_key()
            ),
            2,
        )
        .await?;
        enqueue_index_job(&self.transaction, &self.tables, job).await
    }
    pub async fn claim_index_jobs(
        &mut self,
        limit: usize,
        lease_seconds: i64,
        namespace: Option<&str>,
    ) -> PostgresStoreResult<Vec<IndexJob>> {
        claim_index_jobs(
            &self.transaction,
            &self.tables,
            limit,
            lease_seconds,
            namespace,
            None,
        )
        .await
    }
    pub async fn claim_index_jobs_filtered(
        &mut self,
        limit: usize,
        lease_seconds: i64,
        namespace: Option<&str>,
        index_kinds: &[String],
    ) -> PostgresStoreResult<Vec<IndexJob>> {
        if index_kinds.is_empty() || index_kinds.iter().any(|kind| kind.trim().is_empty()) {
            return Err(PostgresStoreError::Backend(
                "filtered index-job claim requires non-empty index kinds".to_owned(),
            ));
        }
        claim_index_jobs(
            &self.transaction,
            &self.tables,
            limit,
            lease_seconds,
            namespace,
            Some(index_kinds),
        )
        .await
    }
    pub async fn reconcile_join_index_jobs(
        &mut self,
        limit: usize,
        lease_seconds: i64,
        namespace: Option<&str>,
        names: GraphIndexTableNames,
    ) -> PostgresStoreResult<usize> {
        reconcile_join_index_jobs(
            &self.transaction,
            &self.tables,
            limit,
            lease_seconds,
            namespace,
            names,
        )
        .await
    }
    pub async fn mark_index_job_done(
        &mut self,
        job_id: &str,
        claim_token: Option<&str>,
    ) -> PostgresStoreResult<bool> {
        mark_index_job_done(&self.transaction, &self.tables, job_id, claim_token).await
    }
    pub async fn accept_index_job_result(
        &mut self,
        job_id: &str,
        claim_token: &str,
        result_json: &str,
        result_sha256: &str,
    ) -> PostgresStoreResult<AcceptedIndexJobResult> {
        accept_index_job_result(
            &self.transaction,
            &self.tables,
            job_id,
            claim_token,
            result_json,
            result_sha256,
        )
        .await
    }
    pub async fn mark_index_job_failed(
        &mut self,
        job_id: &str,
        error: &str,
        final_: bool,
        claim_token: Option<&str>,
    ) -> PostgresStoreResult<()> {
        mark_index_job_failed(
            &self.transaction,
            &self.tables,
            job_id,
            error,
            final_,
            claim_token,
        )
        .await
    }
    pub async fn bump_retry_and_requeue(
        &mut self,
        job_id: &str,
        error: &str,
        delay: i64,
        claim_token: Option<&str>,
    ) -> PostgresStoreResult<()> {
        bump_retry_and_requeue(
            &self.transaction,
            &self.tables,
            job_id,
            error,
            delay,
            claim_token,
        )
        .await
    }
    pub async fn renew_index_job_lease(
        &mut self,
        job_id: &str,
        claim_token: &str,
        lease_seconds: i64,
    ) -> PostgresStoreResult<bool> {
        renew_index_job_lease(
            &self.transaction,
            &self.tables,
            job_id,
            claim_token,
            lease_seconds,
        )
        .await
    }
    pub async fn requeue_index_job_at_tail(
        &mut self,
        job_id: &str,
        payload_json: String,
        delay: i64,
        claim_token: Option<&str>,
    ) -> PostgresStoreResult<()> {
        requeue_index_job_at_tail(
            &self.transaction,
            &self.tables,
            job_id,
            payload_json,
            delay,
            claim_token,
        )
        .await
    }
}
