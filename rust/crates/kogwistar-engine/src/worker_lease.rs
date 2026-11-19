use kogwistar_store::{IndexJobWriteStore, LaneMessageWriteStore, StoreResult};
use std::collections::HashSet;
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

/// Keeps a runtime lane claim live while executing one native worker step.
/// A lost lease cancels the operation so it cannot submit a stale result.
pub async fn with_lane_claim_renewal<Q, T, E, Fut>(
    queue: &Q,
    message_id: &str,
    claimed_by: &str,
    lease_seconds: i64,
    operation: Fut,
) -> StoreResult<Option<Result<T, E>>>
where
    Q: LaneMessageWriteStore,
    T: Send,
    E: Send,
    Fut: Future<Output = Result<T, E>> + Send,
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
                if !queue
                    .renew_projected_lane_message_claim(message_id, claimed_by, lease_seconds)
                    .await?
                {
                    return Ok(None);
                }
            }
        }
    }
}

pub(crate) async fn with_batch_lease_renewal<Q, T, Fut>(
    queue: &Q,
    claims: &[(&str, &str)],
    lease_seconds: i64,
    operation: Fut,
) -> StoreResult<(T, HashSet<String>)>
where
    Q: IndexJobWriteStore,
    Fut: Future<Output = T>,
{
    let interval_ms = (lease_seconds.max(1) as u64)
        .saturating_mul(1_000)
        .saturating_div(3)
        .max(100);
    let mut heartbeat = tokio::time::interval(Duration::from_millis(interval_ms));
    heartbeat.tick().await;
    let mut lost = HashSet::new();
    tokio::pin!(operation);
    loop {
        tokio::select! {
            result = &mut operation => return Ok((result, lost)),
            _ = heartbeat.tick() => {
                for (job_id, claim_token) in claims {
                    if !lost.contains(*job_id)
                        && !queue.renew_index_job_lease(job_id, claim_token, lease_seconds).await?
                    {
                        lost.insert((*job_id).to_owned());
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::with_lane_claim_renewal;
    use kogwistar_store::{
        LaneMessageClaimFilter, LaneMessageReadStore, LaneMessageWriteStore,
        NewProjectedLaneMessage, StoreError, StoreResult,
    };
    use kogwistar_store_memory::InMemoryStore;
    use std::future;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    fn lane(message_id: &str) -> NewProjectedLaneMessage {
        NewProjectedLaneMessage {
            message_id: message_id.to_owned(),
            namespace: "workflow".to_owned(),
            purpose: "system".to_owned(),
            inbox_id: "workflow-runtime".to_owned(),
            conversation_id: "conversation".to_owned(),
            recipient_id: "rust-native-worker".to_owned(),
            sender_id: "scheduler".to_owned(),
            msg_type: "workflow.step.execute".to_owned(),
            status: "pending".to_owned(),
            created_at: 1,
            available_at: 0,
            run_id: Some("run".to_owned()),
            step_id: Some("step".to_owned()),
            correlation_id: Some("run".to_owned()),
            payload_json: Some("{}".to_owned()),
            error_json: None,
        }
    }

    async fn claim(
        store: &InMemoryStore,
        owner: &str,
    ) -> Vec<kogwistar_store::ProjectedLaneMessage> {
        store
            .claim_projected_lane_messages_filtered(LaneMessageClaimFilter {
                namespace: "workflow".to_owned(),
                inbox_id: "workflow-runtime".to_owned(),
                claimed_by: owner.to_owned(),
                message_ids: Some(vec!["lane-heartbeat".to_owned()]),
                run_id: Some("run".to_owned()),
                msg_type: Some("workflow.step.execute".to_owned()),
                recipient_id: Some("rust-native-worker".to_owned()),
                limit: 1,
                lease_seconds: 1,
            })
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn lane_heartbeat_prevents_reclaim_during_long_step() {
        let store = InMemoryStore::new();
        store
            .project_lane_message(lane("lane-heartbeat"))
            .await
            .unwrap();
        assert_eq!(claim(&store, "worker-1").await.len(), 1);

        let operation = async {
            tokio::time::sleep(Duration::from_millis(1_350)).await;
            Ok::<_, StoreError>(())
        };
        let competing_claim = async {
            tokio::time::sleep(Duration::from_millis(1_100)).await;
            claim(&store, "worker-2").await
        };
        let (result, competing_claim) = tokio::join!(
            with_lane_claim_renewal(&store, "lane-heartbeat", "worker-1", 1, operation),
            competing_claim
        );

        assert!(matches!(result, Ok(Some(Ok(())))));
        assert!(competing_claim.is_empty());
        let lane = store
            .projected_lane_message("lane-heartbeat")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(lane.claimed_by.as_deref(), Some("worker-1"));
    }

    #[tokio::test]
    async fn lost_lane_claim_cancels_step_future() {
        struct DropSignal(Arc<AtomicBool>);
        impl Drop for DropSignal {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }

        let store = InMemoryStore::new();
        store
            .project_lane_message(lane("lane-heartbeat"))
            .await
            .unwrap();
        assert_eq!(claim(&store, "worker-1").await.len(), 1);
        store
            .ack_projected_lane_message("lane-heartbeat", "worker-1")
            .await
            .unwrap();

        let dropped = Arc::new(AtomicBool::new(false));
        let signal = DropSignal(Arc::clone(&dropped));
        let operation = async move {
            let _signal = signal;
            future::pending::<StoreResult<()>>().await
        };
        let result = with_lane_claim_renewal(&store, "lane-heartbeat", "worker-1", 1, operation)
            .await
            .unwrap();

        assert!(result.is_none());
        assert!(dropped.load(Ordering::SeqCst));
    }
}
