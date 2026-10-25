//! Backend-neutral Rust index-job result admission.
//!
//! This loop deliberately stops at durable result admission. Embedding and
//! vector projection remain caller-owned capabilities, so Python and Rust
//! providers can use the same lease, retry, and idempotency contract.

use kogwistar_store::{IndexJob, IndexJobWriteStore, StoreError, StoreResult};
use sha2::{Digest, Sha256};
use std::future::Future;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IndexWorkerSummary {
    pub claimed: usize,
    pub accepted: usize,
    pub existing: usize,
    pub claim_lost: usize,
    pub retried: usize,
    pub failed: usize,
}

fn worker_error(error: impl ToString) -> StoreError {
    StoreError::Backend {
        backend: "index-job-worker".to_owned(),
        message: error.to_string(),
    }
}

fn result_digest(result_json: &str) -> String {
    hex::encode(Sha256::digest(result_json.as_bytes()))
}

/// Claim and process one bounded batch of jobs using a caller-supplied result
/// provider. The provider is never given queue mutation authority.
pub async fn run_index_jobs_once<S, F, Fut>(
    store: &S,
    limit: usize,
    lease_seconds: i64,
    namespace: Option<&str>,
    mut provider: F,
) -> StoreResult<IndexWorkerSummary>
where
    S: IndexJobWriteStore,
    F: FnMut(IndexJob) -> Fut,
    Fut: Future<Output = StoreResult<String>>,
{
    let jobs = store
        .claim_index_jobs(limit, lease_seconds, namespace)
        .await?;
    let mut summary = IndexWorkerSummary {
        claimed: jobs.len(),
        ..IndexWorkerSummary::default()
    };

    for job in jobs {
        let Some(claim_token) = job.claim_token.as_deref() else {
            return Err(worker_error("claimed index job has no claim token"));
        };
        // A result may have been admitted before a worker lost the final
        // acknowledgement.  Replaying the provider in that state would waste
        // inference and could repeat provider-side effects.  The durable
        // result is the idempotency record; only finalize the queue row.
        if job.accepted_result_json.is_some() {
            if store
                .mark_index_job_done(&job.job_id, Some(claim_token))
                .await?
            {
                summary.existing += 1;
            } else {
                summary.claim_lost += 1;
            }
            continue;
        }
        let retry = job.retry_count + 1;
        match provider(job.clone()).await {
            Ok(result_json) => {
                let admission = store
                    .accept_index_job_result(
                        &job.job_id,
                        claim_token,
                        &result_json,
                        &result_digest(&result_json),
                    )
                    .await?;
                match admission.status.as_str() {
                    "accepted" => summary.accepted += 1,
                    "existing" => summary.existing += 1,
                    "rejected" => {
                        summary.claim_lost += 1;
                        continue;
                    }
                    status => {
                        return Err(worker_error(format!("unknown result status {status:?}")));
                    }
                }
                if !store
                    .mark_index_job_done(&job.job_id, Some(claim_token))
                    .await?
                {
                    summary.claim_lost += 1;
                }
            }
            Err(error) => {
                let message = error.to_string();
                if retry < job.max_retries {
                    store
                        .bump_retry_and_requeue(
                            &job.job_id,
                            &message,
                            2_i64.pow(retry.min(8) as u32 - 1),
                            Some(claim_token),
                        )
                        .await?;
                    summary.retried += 1;
                } else {
                    store
                        .mark_index_job_failed(&job.job_id, &message, true, Some(claim_token))
                        .await?;
                    summary.failed += 1;
                }
            }
        }
    }
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kogwistar_store::{IndexJobReadStore, NewIndexJob};
    use kogwistar_store_memory::InMemoryStore;

    async fn enqueue(store: &InMemoryStore, max_retries: i64) {
        store
            .enqueue_index_job(NewIndexJob {
                job_id: "job-1".to_owned(),
                namespace: "ns".to_owned(),
                entity_kind: "node".to_owned(),
                entity_id: "node-1".to_owned(),
                index_kind: "node_embedding".to_owned(),
                op: "UPSERT".to_owned(),
                payload_json: Some("{}".to_owned()),
                max_retries,
            })
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn admits_provider_result_and_marks_claim_done() {
        let store = InMemoryStore::new();
        enqueue(&store, 3).await;
        let summary = run_index_jobs_once(&store, 1, 30, Some("ns"), |_job| async {
            Ok("{\"embedding\":[1,0]}".to_owned())
        })
        .await
        .unwrap();
        assert_eq!(summary.accepted, 1);
        assert_eq!(summary.claim_lost, 0);
        assert_eq!(
            store
                .index_job_result("job-1")
                .await
                .unwrap()
                .unwrap()
                .status,
            "existing"
        );
    }

    #[tokio::test]
    async fn provider_failure_requeues_without_admitting_a_result() {
        let store = InMemoryStore::new();
        enqueue(&store, 3).await;
        let summary = run_index_jobs_once(&store, 1, 30, Some("ns"), |_job| async {
            Err(StoreError::Backend {
                backend: "provider".to_owned(),
                message: "offline".to_owned(),
            })
        })
        .await
        .unwrap();
        assert_eq!(summary.retried, 1);
        assert!(store.index_job_result("job-1").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn cancelled_provider_leaves_claim_for_lease_recovery() {
        let store = InMemoryStore::new();
        enqueue(&store, 3).await;
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let task_store = store.clone();
        let task = tokio::spawn(async move {
            let mut started_tx = Some(started_tx);
            run_index_jobs_once(&task_store, 1, 0, Some("ns"), move |_job| {
                let signal = started_tx.take();
                async move {
                    if let Some(signal) = signal {
                        let _ = signal.send(());
                    }
                    std::future::pending::<StoreResult<String>>().await
                }
            })
            .await
        });

        started_rx.await.unwrap();
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());

        let claimed = store
            .index_jobs(Some("ns"), Some("DOING"), None, None, None, 10)
            .await
            .unwrap();
        assert_eq!(claimed.len(), 1);
        let original_claim_token = claimed[0].claim_token.as_deref().unwrap();
        assert_eq!(claimed[0].retry_count, 0);
        assert!(claimed[0].accepted_result_json.is_none());

        std::thread::sleep(std::time::Duration::from_secs(1));
        let reclaimed = store.claim_index_jobs(1, 30, Some("ns")).await.unwrap();
        assert_eq!(reclaimed.len(), 1);
        assert_eq!(reclaimed[0].job_id, "job-1");
        assert_ne!(
            reclaimed[0].claim_token.as_deref(),
            Some(original_claim_token)
        );
    }

    #[tokio::test]
    async fn accepted_result_is_reused_after_lost_ack_and_lease_reclaim() {
        let store = InMemoryStore::new();
        enqueue(&store, 3).await;

        // Model a worker that durably admitted the provider result but lost
        // the acknowledgement before marking the queue row done.
        let claimed = store
            .claim_index_jobs(1, 0, Some("ns"))
            .await
            .unwrap()
            .pop()
            .unwrap();
        let claim_token = claimed.claim_token.clone().unwrap();
        let first_result = "{\"embedding\":[1,0]}";
        assert_eq!(
            store
                .accept_index_job_result(
                    "job-1",
                    &claim_token,
                    first_result,
                    &result_digest(first_result),
                )
                .await
                .unwrap()
                .status,
            "accepted"
        );

        // The in-memory clock is second-granularity, so wait for the lease to
        // become reclaimable rather than relying on a timing race.
        std::thread::sleep(std::time::Duration::from_secs(1));
        let mut provider_calls = 0;
        let summary = run_index_jobs_once(&store, 1, 30, Some("ns"), |_job| {
            provider_calls += 1;
            async { Ok("{\"embedding\":[0,1]}".to_owned()) }
        })
        .await
        .unwrap();

        assert_eq!(provider_calls, 0);
        assert_eq!(summary.accepted, 0);
        assert_eq!(summary.existing, 1);
        assert_eq!(summary.claim_lost, 0);
        assert_eq!(
            store
                .index_job_result("job-1")
                .await
                .unwrap()
                .unwrap()
                .result_json
                .as_deref(),
            Some(first_result)
        );
    }
}
