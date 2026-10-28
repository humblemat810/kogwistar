use kogwistar_store::{IndexJobWriteStore, StoreResult};
use std::future::Future;
use std::time::Duration;

pub(crate) async fn with_lease_renewal<Q, T, Fut>(
    queue: &Q,
    job_id: &str,
    claim_token: &str,
    lease_seconds: i64,
    operation: Fut,
) -> StoreResult<Option<StoreResult<T>>>
where
    Q: IndexJobWriteStore,
    Fut: Future<Output = StoreResult<T>>,
{
    let interval_ms = (lease_seconds.max(1) as u64)
        .saturating_mul(1_000)
        .saturating_div(3)
        .max(100);
    let mut heartbeat = tokio::time::interval(Duration::from_millis(interval_ms));
    heartbeat.tick().await;
    tokio::pin!(operation);
    loop {
        tokio::select! {
            result = &mut operation => return Ok(Some(result)),
            _ = heartbeat.tick() => {
                if !queue.renew_index_job_lease(job_id, claim_token, lease_seconds).await? {
                    return Ok(None);
                }
            }
        }
    }
}
