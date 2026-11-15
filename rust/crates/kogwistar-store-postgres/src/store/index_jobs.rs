use super::super::*;

impl PostgresStore {
    pub async fn enqueue_index_job(&self, job: NewIndexJob) -> PostgresStoreResult<String> {
        let namespace = job.namespace.clone();
        require_namespace(&namespace)?;
        self.transaction(move |uow| Box::pin(async move { uow.enqueue_index_job(job).await }))
            .await
    }
    pub async fn claim_index_jobs(
        &self,
        limit: usize,
        lease_seconds: i64,
        namespace: Option<&str>,
    ) -> PostgresStoreResult<Vec<IndexJob>> {
        let namespace = namespace.map(ToOwned::to_owned);
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.claim_index_jobs(limit, lease_seconds, namespace.as_deref())
                    .await
            })
        })
        .await
    }
    pub async fn claim_index_jobs_filtered(
        &self,
        limit: usize,
        lease_seconds: i64,
        filter: IndexJobClaimFilter,
    ) -> PostgresStoreResult<Vec<IndexJob>> {
        if filter.index_kinds.is_empty()
            || filter.index_kinds.iter().any(|kind| kind.trim().is_empty())
        {
            return Err(PostgresStoreError::Backend(
                "filtered index-job claim requires non-empty index kinds".to_owned(),
            ));
        }
        let namespace = filter.namespace;
        let index_kinds = filter.index_kinds;
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.claim_index_jobs_filtered(
                    limit,
                    lease_seconds,
                    namespace.as_deref(),
                    &index_kinds,
                )
                .await
            })
        })
        .await
    }
    pub async fn reconcile_join_index_jobs(
        &self,
        limit: usize,
        lease_seconds: i64,
        namespace: Option<&str>,
        names: GraphIndexTableNames,
    ) -> PostgresStoreResult<usize> {
        let namespace = namespace.map(ToOwned::to_owned);
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.reconcile_join_index_jobs(limit, lease_seconds, namespace.as_deref(), names)
                    .await
            })
        })
        .await
    }
    pub async fn mark_index_job_done(
        &self,
        job_id: &str,
        claim_token: Option<&str>,
    ) -> PostgresStoreResult<bool> {
        let job_id = job_id.to_owned();
        let claim_token = claim_token.map(ToOwned::to_owned);
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.mark_index_job_done(&job_id, claim_token.as_deref())
                    .await
            })
        })
        .await
    }
    pub async fn accept_index_job_result(
        &self,
        job_id: &str,
        claim_token: &str,
        result_json: &str,
        result_sha256: &str,
    ) -> PostgresStoreResult<AcceptedIndexJobResult> {
        let job_id = job_id.to_owned();
        let claim_token = claim_token.to_owned();
        let result_json = result_json.to_owned();
        let result_sha256 = result_sha256.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.accept_index_job_result(&job_id, &claim_token, &result_json, &result_sha256)
                    .await
            })
        })
        .await
    }
    pub async fn index_job_result(
        &self,
        job_id: &str,
    ) -> PostgresStoreResult<Option<AcceptedIndexJobResult>> {
        index_job_result(&**self.client().await?, &self.tables, job_id).await
    }
    pub async fn mark_index_job_failed(
        &self,
        job_id: &str,
        error: &str,
        final_: bool,
        claim_token: Option<&str>,
    ) -> PostgresStoreResult<()> {
        let job_id = job_id.to_owned();
        let error = error.to_owned();
        let claim_token = claim_token.map(ToOwned::to_owned);
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.mark_index_job_failed(&job_id, &error, final_, claim_token.as_deref())
                    .await
            })
        })
        .await
    }
    pub async fn bump_retry_and_requeue(
        &self,
        job_id: &str,
        error: &str,
        delay: i64,
        claim_token: Option<&str>,
    ) -> PostgresStoreResult<()> {
        let job_id = job_id.to_owned();
        let error = error.to_owned();
        let claim_token = claim_token.map(ToOwned::to_owned);
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.bump_retry_and_requeue(&job_id, &error, delay, claim_token.as_deref())
                    .await
            })
        })
        .await
    }
    pub async fn renew_index_job_lease(
        &self,
        job_id: &str,
        claim_token: &str,
        lease_seconds: i64,
    ) -> PostgresStoreResult<bool> {
        let job_id = job_id.to_owned();
        let claim_token = claim_token.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.renew_index_job_lease(&job_id, &claim_token, lease_seconds)
                    .await
            })
        })
        .await
    }
    pub async fn requeue_index_job_at_tail(
        &self,
        job_id: &str,
        payload_json: String,
        delay: i64,
        claim_token: Option<&str>,
    ) -> PostgresStoreResult<()> {
        let job_id = job_id.to_owned();
        let claim_token = claim_token.map(ToOwned::to_owned);
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.requeue_index_job_at_tail(&job_id, payload_json, delay, claim_token.as_deref())
                    .await
            })
        })
        .await
    }
    pub async fn list_index_jobs(
        &self,
        namespace: Option<&str>,
        status: Option<&str>,
        entity_kind: Option<&str>,
        entity_id: Option<&str>,
        index_kind: Option<&str>,
        limit: usize,
    ) -> PostgresStoreResult<Vec<IndexJob>> {
        let client = self.client().await?;
        list_index_jobs(
            &**client,
            &self.tables,
            namespace,
            status,
            entity_kind,
            entity_id,
            index_kind,
            limit,
        )
        .await
    }
}
