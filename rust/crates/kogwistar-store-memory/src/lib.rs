//! Deterministic in-memory Phase 2 storage.
use kogwistar_contracts::EntityEventEnvelope;
use kogwistar_store::{
    AcceptedIndexJobResult, AppendedEvent, AppliedGraphMutation, DistanceMetric, EntityEvent,
    EventPruneStore, EventReadStore, EventWriteStore, GraphAuthorityCapability, GraphEntityKind,
    GraphMutation, GraphMutationBatch, GraphMutationStore, GraphProjectionPageRead,
    GraphProjectionRead, GraphProjectionVectorQuery, GraphReadStore, GraphRecord, GraphScope,
    GraphWriteStore, IndexJob, IndexJobReadStore, IndexJobWriteStore, LaneMessageClaimFilter,
    LaneMessageFilter, LaneMessageReadStore, LaneMessageWriteStore, MetadataFilter,
    NamedProjection, NamedProjectionWrite, NewEntityEvent, NewIndexJob, NewProjectedLaneMessage,
    NodeIndexStore, ProjectedLaneMessage, ProjectionReadStore, ProjectionWriteStore, ReplayCursor,
    ServerRun, ServerRunCreate, ServerRunEvent, ServerRunReadStore, ServerRunUpdate,
    ServerRunWriteStore, StoreError, StoreResult, VectorCollectionStore, VectorMatch, VectorQuery,
    WorkflowDesignDelta, WorkflowDesignDeltaWrite, WorkflowDesignHistoryReadStore,
    WorkflowDesignHistoryWriteStore, WorkflowDesignSnapshot, WorkflowDesignSnapshotWrite,
    integer_timestamp, timestamp_i64, validate_graph_mutation, validate_graph_mutation_batch,
    validate_graph_mutation_replay,
};
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};

mod recorded_runtime;
pub use recorded_runtime::ProviderEventPublicationCommit;

/// Cloned handles share one process-local writer lock and volatile state. A new
/// store starts empty; state is not persisted across process or store lifetimes.
/// This backend provides no cross-process coordination or durable transactions.
#[derive(Clone, Debug, Default)]
pub struct InMemoryStore {
    state: Arc<RwLock<State>>,
}
#[derive(Debug, Default)]
struct State {
    namespaces: BTreeMap<String, NamespaceState>,
    vector_collections: BTreeMap<(String, String), Vec<GraphRecord>>,
    graph_projections: BTreeMap<GraphScope, GraphProjectionState>,
    workflow_design_snapshots: BTreeMap<(String, i64), WorkflowDesignSnapshot>,
    workflow_design_deltas: BTreeMap<(String, i64), WorkflowDesignDelta>,
    server_runs: BTreeMap<String, ServerRun>,
    server_run_events: BTreeMap<String, Vec<ServerRunEvent>>,
    index_jobs: BTreeMap<String, IndexJob>,
    lane_messages: BTreeMap<String, ProjectedLaneMessage>,
    next_claim_token: i64,
}

fn enqueue_index_job_into_state(state: &mut State, job: NewIndexJob) -> String {
    let now = now_seconds();
    let key = job.coalesce_key();
    if let Some(existing) = state.index_jobs.values_mut().find(|row| {
        row.namespace == job.namespace && row.coalesce_key == key && row.status == "PENDING"
    }) {
        if job.op == "DELETE" || existing.op == "DELETE" {
            existing.op = "DELETE".to_owned();
        } else {
            existing.op = job.op;
        }
        existing.payload_json = job.payload_json;
        existing.updated_at = integer_timestamp(now);
        return existing.job_id.clone();
    }
    let job_id = job.job_id.clone();
    state.index_jobs.insert(
        job_id.clone(),
        IndexJob {
            job_id: job.job_id,
            namespace: job.namespace,
            entity_kind: job.entity_kind,
            entity_id: job.entity_id,
            index_kind: job.index_kind,
            coalesce_key: key,
            op: job.op,
            status: "PENDING".to_owned(),
            lease_until: None,
            next_run_at: None,
            max_retries: job.max_retries,
            retry_count: 0,
            last_error: None,
            payload_json: job.payload_json,
            created_at: integer_timestamp(now),
            updated_at: integer_timestamp(now),
            claim_token: None,
            claim_attempts: 0,
            accepted_result_json: None,
            accepted_result_sha256: None,
            accepted_at: None,
        },
    );
    job_id
}

impl IndexJobReadStore for InMemoryStore {
    async fn index_jobs(
        &self,
        namespace: Option<&str>,
        status: Option<&str>,
        entity_kind: Option<&str>,
        entity_id: Option<&str>,
        index_kind: Option<&str>,
        limit: usize,
    ) -> StoreResult<Vec<IndexJob>> {
        let mut rows = self
            .state
            .read()
            .expect("in-memory store lock poisoned")
            .index_jobs
            .values()
            .filter(|row| namespace.is_none_or(|value| row.namespace == value))
            .filter(|row| status.is_none_or(|value| row.status == value))
            .filter(|row| entity_kind.is_none_or(|value| row.entity_kind == value))
            .filter(|row| entity_id.is_none_or(|value| row.entity_id == value))
            .filter(|row| index_kind.is_none_or(|value| row.index_kind == value))
            .cloned()
            .collect::<Vec<_>>();
        rows.sort_by(|left, right| {
            (timestamp_i64(&left.created_at), &left.job_id)
                .cmp(&(timestamp_i64(&right.created_at), &right.job_id))
        });
        rows.truncate(limit);
        Ok(rows)
    }
}

impl LaneMessageReadStore for InMemoryStore {
    async fn projected_lane_message(
        &self,
        message_id: &str,
    ) -> StoreResult<Option<ProjectedLaneMessage>> {
        Ok(self
            .state
            .read()
            .expect("in-memory store lock poisoned")
            .lane_messages
            .get(message_id)
            .cloned())
    }

    async fn projected_lane_messages(
        &self,
        filter: LaneMessageFilter,
    ) -> StoreResult<Vec<ProjectedLaneMessage>> {
        let mut rows = self
            .state
            .read()
            .expect("in-memory store lock poisoned")
            .lane_messages
            .values()
            .filter(|row| {
                filter
                    .namespace
                    .as_ref()
                    .is_none_or(|v| row.namespace == *v)
            })
            .filter(|row| filter.purpose.as_ref().is_none_or(|v| row.purpose == *v))
            .filter(|row| filter.inbox_id.as_ref().is_none_or(|v| row.inbox_id == *v))
            .filter(|row| {
                filter
                    .conversation_id
                    .as_ref()
                    .is_none_or(|v| row.conversation_id == *v)
            })
            .filter(|row| filter.status.as_ref().is_none_or(|v| row.status == *v))
            .filter(|row| filter.msg_type.as_ref().is_none_or(|v| row.msg_type == *v))
            .filter(|row| {
                filter
                    .sender_id
                    .as_ref()
                    .is_none_or(|v| row.sender_id == *v)
            })
            .filter(|row| {
                filter
                    .recipient_id
                    .as_ref()
                    .is_none_or(|v| row.recipient_id == *v)
            })
            .filter(|row| {
                filter
                    .correlation_id
                    .as_ref()
                    .is_none_or(|v| row.correlation_id.as_ref() == Some(v))
            })
            .filter(|row| filter.created_at_gte.is_none_or(|v| row.created_at >= v))
            .filter(|row| filter.created_at_lte.is_none_or(|v| row.created_at <= v))
            .filter(|row| {
                filter
                    .available_at_gte
                    .is_none_or(|v| row.available_at >= v)
            })
            .filter(|row| {
                filter
                    .available_at_lte
                    .is_none_or(|v| row.available_at <= v)
            })
            .cloned()
            .collect::<Vec<_>>();
        rows.sort_by(|a, b| {
            (a.created_at, a.seq, &a.message_id).cmp(&(b.created_at, b.seq, &b.message_id))
        });
        if filter.newest_first {
            rows.reverse();
        }
        rows.truncate(filter.limit);
        Ok(rows)
    }
}

impl LaneMessageWriteStore for InMemoryStore {
    async fn project_lane_message(&self, row: NewProjectedLaneMessage) -> StoreResult<()> {
        namespace(&row.namespace)?;
        let mut state = self.state.write().expect("in-memory store lock poisoned");
        if state.lane_messages.contains_key(&row.message_id) {
            return Ok(());
        }
        let inbox_tail = state
            .lane_messages
            .values()
            .filter(|v| v.namespace == row.namespace && v.inbox_id == row.inbox_id)
            .max_by_key(|v| (v.seq, v.created_at))
            .cloned();
        let conversation_tail = state
            .lane_messages
            .values()
            .filter(|v| v.namespace == row.namespace && v.conversation_id == row.conversation_id)
            .max_by_key(|v| (v.conversation_seq, v.created_at))
            .cloned();
        let seq = inbox_tail.as_ref().map_or(1, |v| v.seq + 1);
        let conversation_seq = conversation_tail
            .as_ref()
            .map_or(1, |v| v.conversation_seq + 1);
        state.lane_messages.insert(
            row.message_id.clone(),
            ProjectedLaneMessage {
                message_id: row.message_id.clone(),
                namespace: row.namespace,
                purpose: if row.purpose.is_empty() {
                    "user_visible".to_owned()
                } else {
                    row.purpose
                },
                inbox_id: row.inbox_id,
                conversation_id: row.conversation_id,
                recipient_id: row.recipient_id,
                sender_id: row.sender_id,
                msg_type: row.msg_type,
                status: row.status,
                seq,
                conversation_seq,
                claimed_by: None,
                lease_until: None,
                retry_count: 0,
                created_at: row.created_at,
                available_at: row.available_at,
                run_id: row.run_id,
                step_id: row.step_id,
                correlation_id: row.correlation_id,
                payload_json: row.payload_json,
                error_json: row.error_json,
                prev_message_id: inbox_tail.map(|v| v.message_id),
                next_message_id: None,
                inbox_tail_message_id: Some(row.message_id.clone()),
                conversation_tail_message_id: Some(row.message_id),
            },
        );
        Ok(())
    }
    async fn update_projected_lane_message_status(
        &self,
        message_id: &str,
        status: &str,
        error_json: Option<String>,
    ) -> StoreResult<()> {
        if let Some(row) = self
            .state
            .write()
            .expect("in-memory store lock poisoned")
            .lane_messages
            .get_mut(message_id)
        {
            row.status = status.to_owned();
            if error_json.is_some() {
                row.error_json = error_json;
            }
            if matches!(status, "completed" | "failed" | "cancelled") {
                row.claimed_by = None;
                row.lease_until = None;
            }
        }
        Ok(())
    }
    async fn update_projected_lane_message_links(
        &self,
        message_id: &str,
        prev: Option<String>,
        next: Option<String>,
        inbox_tail: Option<String>,
        conversation_tail: Option<String>,
    ) -> StoreResult<()> {
        if let Some(row) = self
            .state
            .write()
            .expect("in-memory store lock poisoned")
            .lane_messages
            .get_mut(message_id)
        {
            row.prev_message_id = prev;
            row.next_message_id = next;
            row.inbox_tail_message_id = inbox_tail;
            row.conversation_tail_message_id = conversation_tail;
        }
        Ok(())
    }
    async fn clear_projected_lane_messages(&self, namespace_value: &str) -> StoreResult<u64> {
        let mut state = self.state.write().expect("in-memory store lock poisoned");
        let before = state.lane_messages.len();
        state
            .lane_messages
            .retain(|_, row| row.namespace != namespace_value);
        Ok((before - state.lane_messages.len()) as u64)
    }
    async fn claim_projected_lane_messages(
        &self,
        namespace_value: &str,
        inbox_id: &str,
        claimed_by: &str,
        limit: usize,
        lease_seconds: i64,
    ) -> StoreResult<Vec<ProjectedLaneMessage>> {
        if limit == 0 {
            return Ok(vec![]);
        }
        let now = now_seconds();
        let mut state = self.state.write().expect("in-memory store lock poisoned");
        let mut ids = state
            .lane_messages
            .values()
            .filter(|row| {
                row.namespace == namespace_value
                    && row.inbox_id == inbox_id
                    && ((row.status == "pending" && row.available_at <= now)
                        || (row.status == "claimed"
                            && row
                                .lease_until
                                .as_ref()
                                .is_some_and(|v| timestamp_i64(v) < now)))
            })
            .map(|row| row.message_id.clone())
            .collect::<Vec<_>>();
        ids.sort_by_key(|id| {
            let r = &state.lane_messages[id];
            (r.seq, r.created_at)
        });
        ids.truncate(limit);
        for id in &ids {
            let row = state
                .lane_messages
                .get_mut(id)
                .expect("selected lane row missing");
            row.status = "claimed".to_owned();
            row.claimed_by = Some(claimed_by.to_owned());
            row.lease_until = Some(integer_timestamp(now + lease_seconds));
        }
        Ok(ids
            .into_iter()
            .map(|id| state.lane_messages[&id].clone())
            .collect())
    }
    async fn claim_projected_lane_messages_filtered(
        &self,
        filter: LaneMessageClaimFilter,
    ) -> StoreResult<Vec<ProjectedLaneMessage>> {
        if filter.limit == 0 {
            return Ok(vec![]);
        }
        let now = now_seconds();
        let allowed_ids = filter
            .message_ids
            .as_ref()
            .map(|ids| ids.iter().collect::<std::collections::HashSet<_>>());
        let mut state = self.state.write().expect("in-memory store lock poisoned");
        let mut ids = state
            .lane_messages
            .values()
            .filter(|row| {
                row.namespace == filter.namespace
                    && row.inbox_id == filter.inbox_id
                    && allowed_ids
                        .as_ref()
                        .is_none_or(|ids| ids.contains(&row.message_id))
                    && filter
                        .run_id
                        .as_ref()
                        .is_none_or(|value| row.run_id.as_deref() == Some(value))
                    && filter
                        .msg_type
                        .as_ref()
                        .is_none_or(|value| row.msg_type == *value)
                    && filter
                        .recipient_id
                        .as_ref()
                        .is_none_or(|value| row.recipient_id == *value)
                    && ((row.status == "pending" && row.available_at <= now)
                        || (row.status == "claimed"
                            && row
                                .lease_until
                                .as_ref()
                                .is_some_and(|value| timestamp_i64(value) < now)))
            })
            .map(|row| row.message_id.clone())
            .collect::<Vec<_>>();
        ids.sort_by_key(|id| {
            let row = &state.lane_messages[id];
            (row.seq, row.created_at)
        });
        ids.truncate(filter.limit);
        for id in &ids {
            let row = state
                .lane_messages
                .get_mut(id)
                .expect("selected lane row missing");
            row.status = "claimed".to_owned();
            row.claimed_by = Some(filter.claimed_by.clone());
            row.lease_until = Some(integer_timestamp(now + filter.lease_seconds));
        }
        Ok(ids
            .into_iter()
            .map(|id| state.lane_messages[&id].clone())
            .collect())
    }
    async fn ack_projected_lane_message(
        &self,
        message_id: &str,
        claimed_by: &str,
    ) -> StoreResult<()> {
        if let Some(row) = self
            .state
            .write()
            .expect("in-memory store lock poisoned")
            .lane_messages
            .get_mut(message_id)
            && !matches!(
                row.status.as_str(),
                "completed" | "failed" | "cancelled" | "dead-letter"
            )
            && row.claimed_by.as_deref().is_none_or(|v| {
                row.status == "claimed"
                    && v == claimed_by
                    && row
                        .lease_until
                        .as_ref()
                        .is_some_and(|until| timestamp_i64(until) >= now_seconds())
            })
        {
            row.status = "completed".to_owned();
            row.claimed_by = None;
            row.lease_until = None;
        }
        Ok(())
    }
    async fn requeue_projected_lane_message(
        &self,
        message_id: &str,
        claimed_by: &str,
        error_json: Option<String>,
        delay: i64,
    ) -> StoreResult<()> {
        let now = now_seconds();
        if let Some(row) = self
            .state
            .write()
            .expect("in-memory store lock poisoned")
            .lane_messages
            .get_mut(message_id)
            && !matches!(
                row.status.as_str(),
                "completed" | "failed" | "cancelled" | "dead-letter"
            )
            && row.claimed_by.as_deref().is_none_or(|v| {
                row.status == "claimed"
                    && v == claimed_by
                    && row
                        .lease_until
                        .as_ref()
                        .is_some_and(|until| timestamp_i64(until) >= now)
            })
        {
            row.status = "pending".to_owned();
            row.claimed_by = None;
            row.lease_until = None;
            row.retry_count += 1;
            row.available_at = now + delay.max(0);
            if error_json.is_some() {
                row.error_json = error_json;
            }
        }
        Ok(())
    }
    async fn dead_letter_projected_lane_message(
        &self,
        message_id: &str,
        claimed_by: &str,
        error_json: Option<String>,
    ) -> StoreResult<()> {
        if let Some(row) = self
            .state
            .write()
            .expect("in-memory store lock poisoned")
            .lane_messages
            .get_mut(message_id)
            && !matches!(
                row.status.as_str(),
                "completed" | "failed" | "cancelled" | "dead-letter"
            )
            && row.claimed_by.as_deref().is_none_or(|v| {
                row.status == "claimed"
                    && v == claimed_by
                    && row
                        .lease_until
                        .as_ref()
                        .is_some_and(|until| timestamp_i64(until) >= now_seconds())
            })
        {
            row.status = "dead-letter".to_owned();
            row.claimed_by = None;
            row.lease_until = None;
            if error_json.is_some() {
                row.error_json = error_json;
            }
        }
        Ok(())
    }
}

impl IndexJobWriteStore for InMemoryStore {
    async fn enqueue_index_job(&self, job: NewIndexJob) -> StoreResult<String> {
        namespace(&job.namespace)?;
        let mut state = self.state.write().expect("in-memory store lock poisoned");
        Ok(enqueue_index_job_into_state(&mut state, job))
    }

    async fn claim_index_jobs(
        &self,
        limit: usize,
        lease_seconds: i64,
        namespace_filter: Option<&str>,
    ) -> StoreResult<Vec<IndexJob>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let now = now_seconds();
        let mut state = self.state.write().expect("in-memory store lock poisoned");
        let mut ids = state
            .index_jobs
            .values()
            .filter(|row| namespace_filter.is_none_or(|value| row.namespace == value))
            .filter(|row| {
                (row.status == "PENDING"
                    && row
                        .next_run_at
                        .as_ref()
                        .is_none_or(|value| timestamp_i64(value) <= now))
                    || (row.status == "DOING"
                        && row
                            .lease_until
                            .as_ref()
                            .is_some_and(|value| timestamp_i64(value) < now))
            })
            .map(|row| row.job_id.clone())
            .collect::<Vec<_>>();
        ids.sort_by(|left, right| {
            let left = &state.index_jobs[left];
            let right = &state.index_jobs[right];
            (timestamp_i64(&left.created_at), &left.job_id)
                .cmp(&(timestamp_i64(&right.created_at), &right.job_id))
        });
        ids.truncate(limit);
        let mut claimed = Vec::with_capacity(ids.len());
        for id in ids {
            state.next_claim_token += 1;
            let token = format!("memory-claim-{}", state.next_claim_token);
            let row = state.index_jobs.get_mut(&id).expect("selected job missing");
            row.status = "DOING".to_owned();
            row.lease_until = Some(integer_timestamp(now + lease_seconds));
            row.claim_token = Some(token);
            row.updated_at = integer_timestamp(now);
            claimed.push(row.clone());
        }
        Ok(claimed)
    }

    async fn mark_index_job_done(
        &self,
        job_id: &str,
        claim_token: Option<&str>,
    ) -> StoreResult<bool> {
        let now = now_seconds();
        let mut state = self.state.write().expect("in-memory store lock poisoned");
        let Some(row) = state.index_jobs.get_mut(job_id) else {
            return Ok(false);
        };
        if row.status != "DOING"
            || claim_token.is_some_and(|token| {
                row.claim_token.as_deref() != Some(token)
                    || row
                        .lease_until
                        .as_ref()
                        .is_none_or(|lease_until| timestamp_i64(lease_until) < now)
            })
        {
            return Ok(false);
        }
        row.status = "DONE".to_owned();
        row.lease_until = None;
        row.claim_token = None;
        row.updated_at = integer_timestamp(now);
        Ok(true)
    }

    async fn accept_index_job_result(
        &self,
        job_id: &str,
        claim_token: &str,
        result_json: &str,
        result_sha256: &str,
    ) -> StoreResult<AcceptedIndexJobResult> {
        let now = now_seconds();
        let mut state = self.state.write().expect("in-memory store lock poisoned");
        let Some(row) = state.index_jobs.get_mut(job_id) else {
            return Ok(AcceptedIndexJobResult {
                status: "rejected".to_owned(),
                result_json: None,
                result_sha256: None,
                accepted_at: None,
            });
        };
        if let Some(existing) = row.accepted_result_json.as_ref() {
            return Ok(AcceptedIndexJobResult {
                status: "existing".to_owned(),
                result_json: Some(existing.clone()),
                result_sha256: row.accepted_result_sha256.clone(),
                accepted_at: row.accepted_at.clone(),
            });
        }
        if row.status != "DOING"
            || row.claim_token.as_deref() != Some(claim_token)
            || row
                .lease_until
                .as_ref()
                .is_some_and(|value| timestamp_i64(value) < now)
        {
            return Ok(AcceptedIndexJobResult {
                status: "rejected".to_owned(),
                result_json: None,
                result_sha256: None,
                accepted_at: None,
            });
        }
        row.accepted_result_json = Some(result_json.to_owned());
        row.accepted_result_sha256 = Some(result_sha256.to_owned());
        row.accepted_at = Some(integer_timestamp(now));
        Ok(AcceptedIndexJobResult {
            status: "accepted".to_owned(),
            result_json: Some(result_json.to_owned()),
            result_sha256: Some(result_sha256.to_owned()),
            accepted_at: row.accepted_at.clone(),
        })
    }

    async fn index_job_result(&self, job_id: &str) -> StoreResult<Option<AcceptedIndexJobResult>> {
        let state = self.state.read().expect("in-memory store lock poisoned");
        Ok(state.index_jobs.get(job_id).and_then(|row| {
            row.accepted_result_json
                .as_ref()
                .map(|result_json| AcceptedIndexJobResult {
                    status: "existing".to_owned(),
                    result_json: Some(result_json.clone()),
                    result_sha256: row.accepted_result_sha256.clone(),
                    accepted_at: row.accepted_at.clone(),
                })
        }))
    }

    async fn mark_index_job_failed(
        &self,
        job_id: &str,
        error: &str,
        final_: bool,
        claim_token: Option<&str>,
    ) -> StoreResult<()> {
        let now = now_seconds();
        if let Some(row) = self
            .state
            .write()
            .expect("in-memory store lock poisoned")
            .index_jobs
            .get_mut(job_id)
            && row.status == "DOING"
            && claim_token.is_none_or(|token| {
                row.claim_token.as_deref() == Some(token)
                    && row
                        .lease_until
                        .as_ref()
                        .is_some_and(|lease_until| timestamp_i64(lease_until) >= now)
            })
        {
            if final_ {
                row.status = "FAILED".to_owned();
            }
            row.lease_until = None;
            row.last_error = Some(error.chars().take(2000).collect());
            row.updated_at = integer_timestamp(now);
        }
        Ok(())
    }

    async fn bump_retry_and_requeue(
        &self,
        job_id: &str,
        error: &str,
        next_run_at_seconds: i64,
        claim_token: Option<&str>,
    ) -> StoreResult<()> {
        let now = now_seconds();
        if let Some(row) = self
            .state
            .write()
            .expect("in-memory store lock poisoned")
            .index_jobs
            .get_mut(job_id)
            && row.status == "DOING"
            && claim_token.is_none_or(|token| {
                row.claim_token.as_deref() == Some(token)
                    && row
                        .lease_until
                        .as_ref()
                        .is_some_and(|lease_until| timestamp_i64(lease_until) >= now)
            })
        {
            row.retry_count += 1;
            row.last_error = Some(error.chars().take(2000).collect());
            row.lease_until = None;
            row.updated_at = integer_timestamp(now);
            if row.retry_count >= row.max_retries {
                row.status = "FAILED".to_owned();
                row.next_run_at = None;
            } else {
                row.status = "PENDING".to_owned();
                row.next_run_at = Some(integer_timestamp(now + next_run_at_seconds.max(0)));
            }
        }
        Ok(())
    }

    async fn renew_index_job_lease(
        &self,
        job_id: &str,
        claim_token: &str,
        lease_seconds: i64,
    ) -> StoreResult<bool> {
        let now = now_seconds();
        let mut state = self.state.write().expect("in-memory store lock poisoned");
        let Some(row) = state.index_jobs.get_mut(job_id) else {
            return Ok(false);
        };
        if row.status != "DOING"
            || row.claim_token.as_deref() != Some(claim_token)
            || row
                .lease_until
                .as_ref()
                .is_some_and(|value| timestamp_i64(value) < now)
        {
            return Ok(false);
        }
        row.lease_until = Some(integer_timestamp(now + lease_seconds));
        row.updated_at = integer_timestamp(now);
        Ok(true)
    }

    async fn requeue_index_job_at_tail(
        &self,
        job_id: &str,
        payload_json: String,
        delay_seconds: i64,
        claim_token: Option<&str>,
    ) -> StoreResult<()> {
        let now = now_seconds();
        let mut state = self.state.write().expect("in-memory store lock poisoned");
        let tail = state
            .index_jobs
            .values()
            .map(|row| timestamp_i64(&row.created_at))
            .max()
            .unwrap_or(now)
            + 1;
        if let Some(row) = state.index_jobs.get_mut(job_id)
            && row.status == "DOING"
            && claim_token.is_none_or(|token| {
                row.claim_token.as_deref() == Some(token)
                    && row
                        .lease_until
                        .as_ref()
                        .is_some_and(|lease_until| timestamp_i64(lease_until) >= now)
            })
        {
            row.status = "PENDING".to_owned();
            row.lease_until = None;
            row.claim_token = None;
            row.next_run_at = Some(integer_timestamp(now + delay_seconds.max(0)));
            row.payload_json = Some(payload_json);
            row.created_at = integer_timestamp(tail);
            row.updated_at = integer_timestamp(now);
        }
        Ok(())
    }
}

#[derive(Debug, Default)]
struct NamespaceState {
    records: BTreeMap<(String, String), GraphRecord>,
    record_order: Vec<(String, String)>,
    events: Vec<EntityEvent>,
    event_seq_by_id: BTreeMap<String, i64>,
    cursors: BTreeMap<String, i64>,
    next_event_seq: i64,
    projections: BTreeMap<String, NamedProjection>,
}

/// Materialized graph rows are separately keyed by all graph scope fields.
/// Events deliberately remain in `NamespaceState`, matching the authoritative
/// namespace-local event log and idempotency contract.
#[derive(Debug, Default)]
struct GraphProjectionState {
    records: BTreeMap<(String, String), GraphRecord>,
    record_order: Vec<(String, String)>,
}

impl InMemoryStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Atomic in-memory analogue of the Phase-3 graph mutation UoW.  It is
    /// intentionally a capability method rather than a new general backend:
    /// SQLite remains event/meta only until its own projection slice lands.
    pub async fn apply_graph_mutation(
        &self,
        mutation: GraphMutation,
    ) -> StoreResult<AppliedGraphMutation> {
        validate_graph_mutation(&mutation)?;
        namespace(&mutation.scope.namespace)?;
        let record = mutation.record;
        if record.id.is_empty() {
            return Err(StoreError::EmptyRecordId);
        }
        if mutation.event_id.is_empty() {
            return Err(StoreError::Backend {
                backend: "memory".to_owned(),
                message: "graph mutation event id must not be empty".to_owned(),
            });
        }
        if mutation.op != "ADD" && mutation.op != "REPLACE" && mutation.op != "TOMBSTONE" {
            return Err(StoreError::Backend {
                backend: "memory".to_owned(),
                message: "graph mutation op must be ADD, REPLACE, or TOMBSTONE".to_owned(),
            });
        }
        if let Some(embedding) = &record.embedding {
            finite(embedding)?;
            if embedding.len() != mutation.embedding_dim {
                return Err(StoreError::VectorDimensionMismatch {
                    expected: mutation.embedding_dim,
                    actual: embedding.len(),
                });
            }
        }
        let mut state = self.state.write().expect("in-memory store lock poisoned");
        let existing_event = state
            .namespaces
            .get(&mutation.scope.namespace)
            .and_then(|scope| {
                scope
                    .event_seq_by_id
                    .get(&mutation.event_id)
                    .and_then(|seq| scope.events.iter().find(|event| event.seq == *seq).cloned())
            });
        if let Some(event) = existing_event {
            return Ok(AppliedGraphMutation {
                event,
                inserted: false,
                mutated: false,
            });
        }
        let namespace_state = Self::scope_mut(&mut state, &mutation.scope.namespace);
        let seq = namespace_state.next_event_seq + 1;
        let event = EntityEvent {
            namespace: mutation.scope.namespace.clone(),
            seq,
            event_id: mutation.event_id,
            entity_kind: mutation.entity_kind,
            entity_id: record.id.clone(),
            op: mutation.op,
            payload: mutation.payload,
        };
        namespace_state.next_event_seq = seq;
        namespace_state
            .event_seq_by_id
            .insert(event.event_id.clone(), seq);
        namespace_state.events.push(event.clone());
        let projection = state.graph_projections.entry(mutation.scope).or_default();
        let record_key = (mutation.table, record.id.clone());
        if !projection.records.contains_key(&record_key) {
            projection.record_order.push(record_key.clone());
        }
        projection.records.insert(record_key, record);
        Ok(AppliedGraphMutation {
            event,
            inserted: true,
            mutated: true,
        })
    }

    /// Apply a graph mutation group and its explicit indexing intents under
    /// one in-memory write lock. This is atomic within this process, not durable.
    pub async fn apply_graph_mutations_with_index_jobs_atomic(
        &self,
        batch: GraphMutationBatch,
    ) -> StoreResult<Vec<AppliedGraphMutation>> {
        validate_graph_mutation_batch(&batch)?;
        let namespace_name = batch.mutations[0].scope.namespace.clone();
        let mut state = self.state.write().expect("in-memory store lock poisoned");
        let mut next_seq = state
            .namespaces
            .get(&namespace_name)
            .map_or(0, |scope| scope.next_event_seq);
        let mut seen = BTreeMap::<String, EntityEvent>::new();
        let mut pending = Vec::<(GraphMutation, EntityEvent)>::new();
        let mut results = Vec::with_capacity(batch.mutations.len());

        for mutation in batch.mutations {
            let existing = state
                .namespaces
                .get(&namespace_name)
                .and_then(|scope| {
                    scope
                        .event_seq_by_id
                        .get(&mutation.event_id)
                        .and_then(|seq| scope.events.iter().find(|event| event.seq == *seq))
                })
                .or_else(|| seen.get(&mutation.event_id));
            if let Some(event) = existing {
                validate_graph_mutation_replay(&mutation, event)?;
                results.push(AppliedGraphMutation {
                    event: event.clone(),
                    inserted: false,
                    mutated: false,
                });
                continue;
            }

            next_seq += 1;
            let record = mutation.record.clone();
            let event = EntityEvent {
                namespace: namespace_name.clone(),
                seq: next_seq,
                event_id: mutation.event_id.clone(),
                entity_kind: mutation.entity_kind.clone(),
                entity_id: record.id.clone(),
                op: mutation.op.clone(),
                payload: mutation.payload.clone(),
            };
            seen.insert(event.event_id.clone(), event.clone());
            pending.push((mutation, event.clone()));
            results.push(AppliedGraphMutation {
                event,
                inserted: true,
                mutated: true,
            });
        }

        for (mutation, event) in pending {
            let namespace_state = Self::scope_mut(&mut state, &namespace_name);
            namespace_state.next_event_seq = event.seq;
            namespace_state
                .event_seq_by_id
                .insert(event.event_id.clone(), event.seq);
            namespace_state.events.push(event);

            let projection = state.graph_projections.entry(mutation.scope).or_default();
            let record_key = (mutation.table, mutation.record.id.clone());
            if !projection.records.contains_key(&record_key) {
                projection.record_order.push(record_key.clone());
            }
            projection.records.insert(record_key, mutation.record);
        }

        for job in batch.index_jobs {
            enqueue_index_job_into_state(&mut state, job);
        }
        Ok(results)
    }

    /// Restore an isolated event snapshot without exposing restore semantics on
    /// authoritative store traits. Retained namespace-local sequences may start
    /// above one and contain gaps, but must be positive and strictly increasing.
    /// Cursor values are restored verbatim, including values beyond retained
    /// events; ordinary cursor advancement remains bounded by its write trait.
    pub fn restore_event_snapshot(
        &self,
        events: Vec<EntityEvent>,
        cursors: Vec<ReplayCursor>,
    ) -> StoreResult<()> {
        let mut restored = BTreeMap::<String, NamespaceState>::new();
        for event in events {
            namespace(&event.namespace)?;
            let scope = restored.entry(event.namespace.clone()).or_default();
            if event.seq <= 0
                || scope
                    .events
                    .last()
                    .is_some_and(|last| last.seq >= event.seq)
            {
                return Err(StoreError::InvalidRestoredEventSequence {
                    namespace: event.namespace,
                    seq: event.seq,
                });
            }
            if scope.event_seq_by_id.contains_key(&event.event_id) {
                return Err(StoreError::DuplicateRestoredEventId {
                    namespace: event.namespace,
                    event_id: event.event_id,
                });
            }
            scope.next_event_seq = event.seq;
            scope
                .event_seq_by_id
                .insert(event.event_id.clone(), event.seq);
            scope.events.push(event);
        }
        for cursor in cursors {
            namespace(&cursor.namespace)?;
            restored
                .entry(cursor.namespace)
                .or_default()
                .cursors
                .insert(cursor.consumer, cursor.last_seq);
        }

        let mut state = self.state.write().expect("in-memory store lock poisoned");
        for scope in state.namespaces.values_mut() {
            scope.events.clear();
            scope.event_seq_by_id.clear();
            scope.cursors.clear();
            scope.next_event_seq = 0;
        }
        for (namespace, restored_scope) in restored {
            let scope = Self::scope_mut(&mut state, &namespace);
            scope.events = restored_scope.events;
            scope.event_seq_by_id = restored_scope.event_seq_by_id;
            scope.cursors = restored_scope.cursors;
            scope.next_event_seq = restored_scope.next_event_seq;
        }
        Ok(())
    }

    /// Restore isolated named-projection rows for read inspection only. This
    /// inherent API intentionally does not add projection mutation to store traits.
    pub fn restore_projection_snapshot(
        &self,
        projections: Vec<NamedProjection>,
    ) -> StoreResult<()> {
        let mut restored = BTreeMap::<String, BTreeMap<String, NamedProjection>>::new();
        for projection in projections {
            namespace(&projection.namespace)?;
            let namespace = projection.namespace.clone();
            let key = projection.key.clone();
            if restored
                .entry(namespace.clone())
                .or_default()
                .insert(key.clone(), projection)
                .is_some()
            {
                return Err(StoreError::DuplicateRestoredProjection { namespace, key });
            }
        }
        let mut state = self.state.write().expect("in-memory store lock poisoned");
        for scope in state.namespaces.values_mut() {
            scope.projections.clear();
        }
        for (namespace, projections) in restored {
            Self::scope_mut(&mut state, &namespace).projections = projections;
        }
        Ok(())
    }

    /// Read node and edge projections under one lock so readers cannot combine
    /// opposite sides of an atomic graph-mutation batch from different states.
    pub fn graph_node_edge_snapshot(
        &self,
        scope: &GraphScope,
        limit: usize,
    ) -> StoreResult<(Vec<GraphRecord>, Vec<GraphRecord>)> {
        namespace(&scope.namespace)?;
        let state = self.state.read().expect("in-memory store lock poisoned");
        let projection = state.graph_projections.get(scope);
        let rows_for_table = |table: &str| {
            projection
                .map(|projection| {
                    projection
                        .record_order
                        .iter()
                        .filter(|(record_table, _)| record_table == table)
                        .filter_map(|key| projection.records.get(key))
                        .take(limit)
                        .cloned()
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
        };
        Ok((rows_for_table("gke_nodes"), rows_for_table("gke_edges")))
    }

    // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-store-memory/src/lib.rs#claim_visible_provider_event; python=kogwistar/agent/event_journal.py#claim_one,kogwistar/messaging/service.py#claim_pending_attempt; fixture=contracts/golden/agent-durable-provider-event-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/part_08.rs#in_memory_native_receive_claims_visible_event_and_routes_it; python_test=tests/agent/test_agent_event_journal.py#test_event_journal_persists_semantic_event_then_acknowledges_after_handler; note=ACL visibility and queue claim share one in-memory writer lock.
    /// Claim one provider-event lane only while holding the same lock used by
    /// graph projection writes, so ACL revocation cannot race visibility and
    /// claim in this process-local backend.
    pub fn claim_visible_provider_event(
        &self,
        scope: &GraphScope,
        conversation_id: &str,
        principal: &kogwistar_store::GraphAclPrincipal,
        filter: LaneMessageClaimFilter,
    ) -> StoreResult<Vec<ProjectedLaneMessage>> {
        if filter.namespace != scope.namespace || filter.limit == 0 {
            return Ok(Vec::new());
        }
        let now = now_seconds();
        let allowed_ids = filter
            .message_ids
            .as_ref()
            .map(|ids| ids.iter().collect::<std::collections::HashSet<_>>());
        let mut state = self.state.write().expect("in-memory store lock poisoned");
        let projection = state.graph_projections.get(scope);
        let graph_nodes = projection
            .into_iter()
            .flat_map(|projection| projection.record_order.iter())
            .filter(|(table, _)| table == "gke_nodes")
            .filter_map(|key| projection.and_then(|projection| projection.records.get(key)))
            .cloned()
            .collect::<Vec<_>>();
        let graph_edges = projection
            .into_iter()
            .flat_map(|projection| projection.record_order.iter())
            .filter(|(table, _)| table == "gke_edges")
            .filter_map(|key| projection.and_then(|projection| projection.records.get(key)))
            .cloned()
            .collect::<Vec<_>>();
        let eligible = |row: &&ProjectedLaneMessage| {
            row.namespace == filter.namespace
                && row.purpose == "agent_event"
                && row.inbox_id == filter.inbox_id
                && row.conversation_id == conversation_id
                && row.sender_id == "agent-runtime"
                && allowed_ids
                    .as_ref()
                    .is_none_or(|ids| ids.contains(&row.message_id))
                && filter
                    .run_id
                    .as_ref()
                    .is_none_or(|value| row.run_id.as_deref() == Some(value))
                && filter
                    .msg_type
                    .as_ref()
                    .is_none_or(|value| row.msg_type == *value)
                && filter
                    .recipient_id
                    .as_ref()
                    .is_none_or(|value| row.recipient_id == *value)
                && ((row.status == "pending" && row.available_at <= now)
                    || (row.status == "claimed"
                        && row
                            .lease_until
                            .as_ref()
                            .is_some_and(|value| timestamp_i64(value) < now)))
        };
        let candidate_lanes = state
            .lane_messages
            .values()
            .filter(eligible)
            .cloned()
            .collect::<Vec<_>>();
        if candidate_lanes.len() > 20_000 {
            return Err(StoreError::Backend {
                backend: "memory-provider-event-claim".to_owned(),
                message: "provider-event receive exceeded its 20000-candidate safety bound"
                    .to_owned(),
            });
        }
        let candidate_ids = candidate_lanes
            .iter()
            .map(|lane| lane.message_id.clone())
            .collect::<std::collections::BTreeSet<_>>();
        let mut acl_nodes = graph_nodes.clone();
        acl_nodes.extend(candidate_lanes.iter().map(|lane| {
            let mut candidate = GraphRecord::new(lane.message_id.clone());
            candidate.metadata = serde_json::json!({
                "entity_type": "lane_message",
                "mentions": []
            })
            .as_object()
            .expect("lane candidate metadata is an object")
            .clone();
            candidate
        }));
        let (visible_nodes, _) = kogwistar_store::graph_records_visible_by_acl(
            &acl_nodes,
            &graph_edges,
            "conversation",
            principal,
        );
        let visible_ids = visible_nodes
            .into_iter()
            .filter(|node| candidate_ids.contains(&node.id))
            .map(|node| node.id)
            .collect::<std::collections::BTreeSet<_>>();
        let mut ids = state
            .lane_messages
            .values()
            .filter(eligible)
            .filter(|row| visible_ids.contains(&row.message_id))
            .map(|row| row.message_id.clone())
            .collect::<Vec<_>>();
        ids.sort_by_key(|id| {
            let row = &state.lane_messages[id];
            (row.seq, row.created_at)
        });
        ids.truncate(filter.limit);
        for id in &ids {
            let row = state
                .lane_messages
                .get_mut(id)
                .expect("selected provider event lane disappeared");
            row.status = "claimed".to_owned();
            row.claimed_by = Some(filter.claimed_by.clone());
            row.lease_until = Some(integer_timestamp(now + filter.lease_seconds));
        }
        Ok(ids
            .into_iter()
            .map(|id| state.lane_messages[&id].clone())
            .collect())
    }

    /// Read graph rows and their vector candidates from one projection revision.
    /// ACL filtering can then use the returned rows without racing a concurrent
    /// atomic graph mutation between topology and vector-index reads.
    pub fn graph_node_edge_vector_snapshot(
        &self,
        scope: &GraphScope,
        limit: usize,
        table: &str,
        queries: &[VectorQuery],
    ) -> StoreResult<(Vec<GraphRecord>, Vec<GraphRecord>, Vec<Vec<VectorMatch>>)> {
        if !matches!(table, "gke_nodes" | "gke_edges") {
            return Err(StoreError::Backend {
                backend: "memory-graph-snapshot".to_owned(),
                message: "vector snapshot table must be gke_nodes or gke_edges".to_owned(),
            });
        }
        namespace(&scope.namespace)?;
        for query in queries {
            finite(&query.embedding)?;
        }
        let state = self.state.read().expect("in-memory store lock poisoned");
        let projection = state.graph_projections.get(scope);
        let rows_for_table = |table: &str| {
            projection
                .map(|projection| {
                    projection
                        .record_order
                        .iter()
                        .filter(|(record_table, _)| record_table == table)
                        .filter_map(|key| projection.records.get(key))
                        .take(limit)
                        .cloned()
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
        };
        let nodes = rows_for_table("gke_nodes");
        let edges = rows_for_table("gke_edges");
        let matches = queries
            .iter()
            .map(|query| graph_projection_vector_matches(projection, table, query))
            .collect::<StoreResult<Vec<_>>>()?;
        Ok((nodes, edges, matches))
    }

    fn graph_projection_rows(&self, read: &GraphProjectionRead) -> StoreResult<Vec<GraphRecord>> {
        namespace(&read.scope.namespace)?;
        let state = self.state.read().expect("in-memory store lock poisoned");
        let mut rows = state
            .graph_projections
            .get(&read.scope)
            .map(|projection| {
                projection
                    .record_order
                    .iter()
                    .filter(|(table, _)| table == &read.table)
                    .filter_map(|key| projection.records.get(key))
                    .filter(|record| read.metadata.matches(&record.metadata))
                    .cloned()
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        if let Some(ids) = &read.ids {
            rows.retain(|record| ids.contains(&record.id));
        }
        Ok(rows)
    }

    fn scope<'a>(state: &'a State, ns: &str) -> Option<&'a NamespaceState> {
        state.namespaces.get(ns)
    }
    fn scope_mut<'a>(state: &'a mut State, ns: &str) -> &'a mut NamespaceState {
        state.namespaces.entry(ns.to_owned()).or_default()
    }
}

impl ServerRunReadStore for InMemoryStore {
    async fn server_run(&self, run_id: &str) -> StoreResult<Option<ServerRun>> {
        Ok(self
            .state
            .read()
            .expect("in-memory store lock poisoned")
            .server_runs
            .get(run_id)
            .cloned())
    }

    async fn server_runs(
        &self,
        status: Option<&str>,
        workflow_id: Option<&str>,
        conversation_id: Option<&str>,
        limit: usize,
    ) -> StoreResult<Vec<ServerRun>> {
        let mut rows = self
            .state
            .read()
            .expect("in-memory store lock poisoned")
            .server_runs
            .values()
            .filter(|row| status.is_none_or(|value| row.status == value))
            .filter(|row| workflow_id.is_none_or(|value| row.workflow_id == value))
            .filter(|row| conversation_id.is_none_or(|value| row.conversation_id == value))
            .cloned()
            .collect::<Vec<_>>();
        // Python in-memory meta store is ascending, unlike SQL backends.
        rows.sort_by(|a, b| (a.created_at_ms, &a.run_id).cmp(&(b.created_at_ms, &b.run_id)));
        rows.truncate(limit);
        Ok(rows)
    }

    async fn server_run_events(
        &self,
        run_id: &str,
        after_seq: i64,
        limit: usize,
    ) -> StoreResult<Vec<ServerRunEvent>> {
        let mut rows = self
            .state
            .read()
            .expect("in-memory store lock poisoned")
            .server_run_events
            .get(run_id)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .filter(|row| row.seq > after_seq)
            .collect::<Vec<_>>();
        rows.sort_by_key(|row| row.seq);
        rows.truncate(limit);
        Ok(rows)
    }
}

impl ServerRunWriteStore for InMemoryStore {
    async fn create_server_run(&self, run: ServerRunCreate) -> StoreResult<()> {
        let now = now_ms();
        // Python memory behavior: duplicate id replaces prior row.
        self.state
            .write()
            .expect("in-memory store lock poisoned")
            .server_runs
            .insert(
                run.run_id.clone(),
                ServerRun {
                    run_id: run.run_id,
                    conversation_id: run.conversation_id,
                    workflow_id: run.workflow_id,
                    user_id: run.user_id,
                    user_turn_node_id: Some(run.user_turn_node_id),
                    assistant_turn_node_id: None,
                    status: run.status,
                    cancel_requested: false,
                    result_json: None,
                    error_json: None,
                    created_at_ms: now,
                    updated_at_ms: now,
                    started_at_ms: None,
                    finished_at_ms: None,
                },
            );
        Ok(())
    }

    async fn append_server_run_event(
        &self,
        run_id: &str,
        event_type: &str,
        payload_json: String,
    ) -> StoreResult<ServerRunEvent> {
        let now = now_ms();
        let mut state = self.state.write().expect("in-memory store lock poisoned");
        let events = state
            .server_run_events
            .entry(run_id.to_owned())
            .or_default();
        let event = ServerRunEvent {
            seq: events.last().map_or(1, |row| row.seq + 1),
            run_id: run_id.to_owned(),
            event_type: event_type.to_owned(),
            payload_json,
            created_at_ms: now,
        };
        events.push(event.clone());
        Ok(event)
    }

    async fn update_server_run(&self, run_id: &str, update: ServerRunUpdate) -> StoreResult<()> {
        let mut state = self.state.write().expect("in-memory store lock poisoned");
        if let Some(row) = state.server_runs.get_mut(run_id) {
            row.status = update.status;
            row.assistant_turn_node_id = update.assistant_turn_node_id;
            row.result_json = update.result_json;
            row.error_json = update.error_json;
            row.started_at_ms = update.started_at_ms;
            row.finished_at_ms = update.finished_at_ms;
            if let Some(value) = update.cancel_requested {
                row.cancel_requested = value;
            }
            row.updated_at_ms = now_ms();
        }
        Ok(())
    }

    async fn request_server_run_cancel(&self, run_id: &str) -> StoreResult<()> {
        let mut state = self.state.write().expect("in-memory store lock poisoned");
        if let Some(row) = state.server_runs.get_mut(run_id) {
            row.cancel_requested = true;
            if !matches!(row.status.as_str(), "cancelled" | "failed" | "succeeded") {
                row.status = "cancelling".to_owned();
            }
            row.updated_at_ms = now_ms();
        }
        Ok(())
    }
}
fn namespace(ns: &str) -> StoreResult<()> {
    if ns.is_empty() {
        Err(StoreError::EmptyNamespace)
    } else {
        Ok(())
    }
}
fn collection_name(collection: &str) -> StoreResult<()> {
    if collection.trim().is_empty() || collection.contains('\0') {
        Err(StoreError::InvalidIdentifier {
            identifier: "vector collection name",
        })
    } else {
        Ok(())
    }
}
fn graph_kind_name(kind: GraphEntityKind) -> &'static str {
    match kind {
        GraphEntityKind::Node => "node",
        GraphEntityKind::Edge => "edge",
    }
}
fn now_seconds() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_secs() as i64
}
fn finite(vector: &[f32]) -> StoreResult<()> {
    if vector.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(StoreError::NonFiniteVector)
    }
}
fn distance(metric: DistanceMetric, query: &[f32], candidate: &[f32]) -> StoreResult<f64> {
    if query.len() != candidate.len() {
        return Err(StoreError::VectorDimensionMismatch {
            expected: query.len(),
            actual: candidate.len(),
        });
    }
    finite(query)?;
    finite(candidate)?;
    match metric {
        DistanceMetric::L2 => Ok(query
            .iter()
            .zip(candidate)
            .map(|(a, b)| {
                let d = f64::from(*a) - f64::from(*b);
                d * d
            })
            .sum::<f64>()
            .sqrt()),
        DistanceMetric::Cosine => {
            let (dot, qn, cn) =
                query
                    .iter()
                    .zip(candidate)
                    .fold((0.0, 0.0, 0.0), |(dot, qn, cn), (a, b)| {
                        let a = f64::from(*a);
                        let b = f64::from(*b);
                        (dot + a * b, qn + a * a, cn + b * b)
                    });
            if qn == 0.0 || cn == 0.0 {
                Err(StoreError::ZeroNormVector)
            } else {
                Ok(1.0 - dot / (qn.sqrt() * cn.sqrt()))
            }
        }
        DistanceMetric::InnerProduct => Ok(-query
            .iter()
            .zip(candidate)
            .map(|(left, right)| f64::from(*left) * f64::from(*right))
            .sum::<f64>()),
    }
}

fn cosine_memory_distance(query: &[f32], candidate: Option<&[f32]>) -> f64 {
    let Some(candidate) = candidate else {
        return 2.0;
    };
    if query.len() != candidate.len() {
        return 2.0;
    }
    let (dot, query_norm, candidate_norm) = query.iter().zip(candidate).fold(
        (0.0, 0.0, 0.0),
        |(dot, query_norm, candidate_norm), (left, right)| {
            let left = f64::from(*left);
            let right = f64::from(*right);
            (
                dot + left * right,
                query_norm + left * left,
                candidate_norm + right * right,
            )
        },
    );
    if query_norm == 0.0 || candidate_norm == 0.0 {
        2.0
    } else {
        1.0 - dot / (query_norm.sqrt() * candidate_norm.sqrt())
    }
}

fn graph_projection_vector_matches(
    projection: Option<&GraphProjectionState>,
    table: &str,
    query: &VectorQuery,
) -> StoreResult<Vec<VectorMatch>> {
    let records = projection
        .into_iter()
        .flat_map(|projection| {
            projection
                .record_order
                .iter()
                .filter(|(record_table, _)| record_table == table)
                .filter_map(|key| projection.records.get(key))
        })
        .filter(|record| query.metadata.matches(&record.metadata))
        .filter(|record| record.embedding.is_some());
    let mut results = match query.metric {
        DistanceMetric::Cosine => records
            .map(|record| VectorMatch {
                record: record.clone(),
                distance: cosine_memory_distance(&query.embedding, record.embedding.as_deref()),
            })
            .collect(),
        DistanceMetric::L2 | DistanceMetric::InnerProduct => records
            .map(|record| {
                let vector = record.embedding.as_deref().expect("filtered embedding");
                distance(query.metric, &query.embedding, vector).map(|distance| VectorMatch {
                    record: record.clone(),
                    distance,
                })
            })
            .collect::<StoreResult<Vec<_>>>()?,
    };
    results.sort_by(|left, right| left.distance.total_cmp(&right.distance));
    results.truncate(query.limit);
    Ok(results)
}

impl VectorCollectionStore for InMemoryStore {
    async fn upsert_vector_collection(
        &self,
        ns: &str,
        collection: &str,
        records: Vec<GraphRecord>,
    ) -> StoreResult<()> {
        namespace(ns)?;
        collection_name(collection)?;
        if records.len() > 10_000 {
            return Err(StoreError::Backend {
                backend: "memory-vector-collection".to_owned(),
                message: "upsert batch exceeds the 10000-record limit".to_owned(),
            });
        }
        for record in &records {
            if record.id.trim().is_empty() || record.id.contains('\0') {
                return Err(StoreError::InvalidIdentifier {
                    identifier: "vector collection record id",
                });
            }
            if let Some(embedding) = &record.embedding {
                finite(embedding)?;
            }
        }
        let mut state = self.state.write().expect("in-memory store lock poisoned");
        let collection_records = state
            .vector_collections
            .entry((ns.to_owned(), collection.to_owned()))
            .or_default();
        for record in records {
            if let Some(existing) = collection_records
                .iter_mut()
                .find(|existing| existing.id == record.id)
            {
                *existing = record;
            } else {
                collection_records.push(record);
            }
        }
        Ok(())
    }

    async fn delete_vector_collection_records(
        &self,
        ns: &str,
        collection: &str,
        ids: Vec<String>,
    ) -> StoreResult<()> {
        namespace(ns)?;
        collection_name(collection)?;
        if ids.len() > 10_000 {
            return Err(StoreError::Backend {
                backend: "memory-vector-collection".to_owned(),
                message: "delete batch exceeds the 10000-record limit".to_owned(),
            });
        }
        if ids
            .iter()
            .any(|id| id.trim().is_empty() || id.contains('\0'))
        {
            return Err(StoreError::InvalidIdentifier {
                identifier: "vector collection record id",
            });
        }
        let ids = ids.into_iter().collect::<std::collections::BTreeSet<_>>();
        let mut state = self.state.write().expect("in-memory store lock poisoned");
        if let Some(records) = state
            .vector_collections
            .get_mut(&(ns.to_owned(), collection.to_owned()))
        {
            records.retain(|record| !ids.contains(&record.id));
        }
        Ok(())
    }

    async fn query_vector_collection(
        &self,
        ns: &str,
        collection: &str,
        query: &VectorQuery,
    ) -> StoreResult<Vec<VectorMatch>> {
        namespace(ns)?;
        collection_name(collection)?;
        finite(&query.embedding)?;
        if query.limit > 10_000 {
            return Err(StoreError::Backend {
                backend: "memory-vector-collection".to_owned(),
                message: "query limit exceeds the 10000-record limit".to_owned(),
            });
        }
        let state = self.state.read().expect("in-memory store lock poisoned");
        let Some(records) = state
            .vector_collections
            .get(&(ns.to_owned(), collection.to_owned()))
        else {
            return Ok(Vec::new());
        };
        let mut matches = records
            .iter()
            .filter(|record| query.metadata.matches(&record.metadata))
            .map(|record| {
                let distance = match query.metric {
                    DistanceMetric::Cosine => {
                        cosine_memory_distance(&query.embedding, record.embedding.as_deref())
                    }
                    metric => match record.embedding.as_deref() {
                        Some(embedding) => distance(metric, &query.embedding, embedding)?,
                        None => 2.0,
                    },
                };
                Ok(VectorMatch {
                    record: record.clone(),
                    distance,
                })
            })
            .collect::<StoreResult<Vec<_>>>()?;
        matches.sort_by(|left, right| left.distance.total_cmp(&right.distance));
        matches.truncate(query.limit);
        Ok(matches)
    }
}

impl NodeIndexStore for InMemoryStore {
    async fn upsert_node_index(&self, index: &str, records: Vec<GraphRecord>) -> StoreResult<()> {
        self.upsert_vector_collection("__node_index__", index, records)
            .await
    }

    async fn get_node_index(&self, index: &str, ids: &[String]) -> StoreResult<Vec<GraphRecord>> {
        collection_name(index)?;
        if ids.len() > 10_000 {
            return Err(StoreError::Backend {
                backend: "memory-node-index".to_owned(),
                message: "lookup exceeds the 10000-ID limit".to_owned(),
            });
        }
        let state = self.state.read().expect("in-memory store lock poisoned");
        let records = state
            .vector_collections
            .get(&("__node_index__".to_owned(), index.to_owned()));
        let Some(records) = records else {
            return Ok(Vec::new());
        };
        let by_id = records
            .iter()
            .map(|record| (record.id.as_str(), record))
            .collect::<BTreeMap<_, _>>();
        Ok(ids
            .iter()
            .filter_map(|id| by_id.get(id.as_str()).map(|record| (*record).clone()))
            .collect())
    }

    async fn query_node_index(
        &self,
        index: &str,
        query: &VectorQuery,
    ) -> StoreResult<Vec<VectorMatch>> {
        self.query_vector_collection("__node_index__", index, query)
            .await
    }
}

impl GraphReadStore for InMemoryStore {
    async fn graph_record(&self, ns: &str, id: &str) -> StoreResult<Option<GraphRecord>> {
        namespace(ns)?;
        let state = self.state.read().expect("in-memory store lock poisoned");
        let matches = Self::scope(&state, ns)
            .into_iter()
            .flat_map(|scope| scope.record_order.iter())
            .filter(|(_, record_id)| record_id == id)
            .filter_map(|key| Self::scope(&state, ns)?.records.get(key))
            .cloned()
            .collect::<Vec<_>>();
        match matches.as_slice() {
            [] => Ok(None),
            [record] => Ok(Some(record.clone())),
            _ => Err(StoreError::Backend {
                backend: "memory-graph-read".to_owned(),
                message: "record ID is ambiguous across entity kinds; use a kind-aware read"
                    .to_owned(),
            }),
        }
    }

    async fn graph_record_of_kind(
        &self,
        ns: &str,
        kind: GraphEntityKind,
        id: &str,
    ) -> StoreResult<Option<GraphRecord>> {
        namespace(ns)?;
        let state = self.state.read().expect("in-memory store lock poisoned");
        let key = (graph_kind_name(kind).to_owned(), id.to_owned());
        Ok(Self::scope(&state, ns)
            .and_then(|scope| scope.records.get(&key))
            .cloned())
    }
    async fn graph_records(
        &self,
        ns: &str,
        filter: &MetadataFilter,
    ) -> StoreResult<Vec<GraphRecord>> {
        namespace(ns)?;
        let state = self.state.read().expect("in-memory store lock poisoned");
        Ok(Self::scope(&state, ns)
            .map(|scope| {
                scope
                    .record_order
                    .iter()
                    .filter_map(|key| scope.records.get(key))
                    .filter(|record| filter.matches(&record.metadata))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default())
    }
    async fn graph_records_of_kind(
        &self,
        ns: &str,
        kind: GraphEntityKind,
        filter: &MetadataFilter,
    ) -> StoreResult<Vec<GraphRecord>> {
        namespace(ns)?;
        let state = self.state.read().expect("in-memory store lock poisoned");
        Ok(Self::scope(&state, ns)
            .map(|scope| {
                scope
                    .record_order
                    .iter()
                    .filter(|(record_kind, _)| record_kind == graph_kind_name(kind))
                    .filter_map(|key| scope.records.get(key))
                    .filter(|record| filter.matches(&record.metadata))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default())
    }
    async fn vector_query(&self, ns: &str, query: &VectorQuery) -> StoreResult<Vec<VectorMatch>> {
        namespace(ns)?;
        finite(&query.embedding)?;
        let state = self.state.read().expect("in-memory store lock poisoned");
        let mut results = match query.metric {
            DistanceMetric::Cosine => Self::scope(&state, ns)
                .into_iter()
                .flat_map(|scope| {
                    scope
                        .record_order
                        .iter()
                        .filter_map(|key| scope.records.get(key))
                })
                .filter(|record| query.metadata.matches(&record.metadata))
                .filter(|record| record.embedding.is_some())
                .map(|record| VectorMatch {
                    record: record.clone(),
                    distance: cosine_memory_distance(&query.embedding, record.embedding.as_deref()),
                })
                .collect(),
            DistanceMetric::L2 | DistanceMetric::InnerProduct => Self::scope(&state, ns)
                .into_iter()
                .flat_map(|scope| {
                    scope
                        .record_order
                        .iter()
                        .filter_map(|key| scope.records.get(key))
                })
                .filter(|record| query.metadata.matches(&record.metadata))
                .filter_map(|record| {
                    record.embedding.as_deref().map(|vector| {
                        distance(query.metric, &query.embedding, vector).map(|distance| {
                            VectorMatch {
                                record: record.clone(),
                                distance,
                            }
                        })
                    })
                })
                .collect::<StoreResult<Vec<_>>>()?,
        };
        // `sort_by` is stable, so equal distances retain namespace insertion order.
        results.sort_by(|left, right| left.distance.total_cmp(&right.distance));
        results.truncate(query.limit);
        Ok(results)
    }

    async fn vector_query_of_kind(
        &self,
        ns: &str,
        kind: GraphEntityKind,
        query: &VectorQuery,
    ) -> StoreResult<Vec<VectorMatch>> {
        namespace(ns)?;
        finite(&query.embedding)?;
        let state = self.state.read().expect("in-memory store lock poisoned");
        let records = Self::scope(&state, ns)
            .into_iter()
            .flat_map(|scope| {
                scope
                    .record_order
                    .iter()
                    .filter(|(record_kind, _)| record_kind == graph_kind_name(kind))
                    .filter_map(|key| scope.records.get(key))
            })
            .filter(|record| query.metadata.matches(&record.metadata));
        let mut results = match query.metric {
            DistanceMetric::Cosine => records
                .filter(|record| record.embedding.is_some())
                .map(|record| VectorMatch {
                    record: record.clone(),
                    distance: cosine_memory_distance(&query.embedding, record.embedding.as_deref()),
                })
                .collect(),
            DistanceMetric::L2 | DistanceMetric::InnerProduct => records
                .filter_map(|record| {
                    record.embedding.as_deref().map(|vector| {
                        distance(query.metric, &query.embedding, vector).map(|distance| {
                            VectorMatch {
                                record: record.clone(),
                                distance,
                            }
                        })
                    })
                })
                .collect::<StoreResult<Vec<_>>>()?,
        };
        results.sort_by(|left, right| left.distance.total_cmp(&right.distance));
        results.truncate(query.limit);
        Ok(results)
    }

    async fn vector_query_of_kind_numeric_lte(
        &self,
        ns: &str,
        kind: GraphEntityKind,
        query: &VectorQuery,
        field: &str,
        maximum: f64,
    ) -> StoreResult<Vec<VectorMatch>> {
        if field.trim().is_empty() || !maximum.is_finite() {
            return Err(StoreError::Backend {
                backend: "memory-graph-read".to_owned(),
                message: "invalid numeric metadata bound".to_owned(),
            });
        }
        let mut bounded_query = query.clone();
        bounded_query.limit = usize::MAX;
        let mut matches = self.vector_query_of_kind(ns, kind, &bounded_query).await?;
        matches.retain(|matched| {
            matched
                .record
                .metadata
                .get(field)
                .and_then(Value::as_f64)
                .is_some_and(|value| value <= maximum)
        });
        matches.truncate(query.limit);
        Ok(matches)
    }
}

impl GraphWriteStore for InMemoryStore {
    async fn upsert_graph_record(&self, ns: &str, record: GraphRecord) -> StoreResult<()> {
        namespace(ns)?;
        if record.id.is_empty() {
            return Err(StoreError::EmptyRecordId);
        }
        if let Some(vector) = &record.embedding {
            finite(vector)?;
        }
        let mut state = self.state.write().expect("in-memory store lock poisoned");
        let scope = Self::scope_mut(&mut state, ns);
        let key = (String::new(), record.id.clone());
        if !scope.records.contains_key(&key) {
            scope.record_order.push(key.clone());
        }
        scope.records.insert(key, record);
        Ok(())
    }
    async fn delete_graph_record(&self, ns: &str, id: &str) -> StoreResult<bool> {
        namespace(ns)?;
        let mut state = self.state.write().expect("in-memory store lock poisoned");
        let scope = Self::scope_mut(&mut state, ns);
        let matching_keys = scope
            .record_order
            .iter()
            .filter(|(_, record_id)| record_id == id)
            .cloned()
            .collect::<Vec<_>>();
        if matching_keys.len() > 1 {
            return Err(StoreError::Backend {
                backend: "memory-graph-write".to_owned(),
                message: "record ID is ambiguous across entity kinds; use a kind-aware delete"
                    .to_owned(),
            });
        }
        let Some(key) = matching_keys.first() else {
            return Ok(false);
        };
        scope.record_order.retain(|record_key| record_key != key);
        Ok(scope.records.remove(key).is_some())
    }

    async fn upsert_graph_record_of_kind(
        &self,
        ns: &str,
        kind: GraphEntityKind,
        record: GraphRecord,
    ) -> StoreResult<()> {
        namespace(ns)?;
        if record.id.is_empty() {
            return Err(StoreError::EmptyRecordId);
        }
        if let Some(vector) = &record.embedding {
            finite(vector)?;
        }
        let mut state = self.state.write().expect("in-memory store lock poisoned");
        let scope = Self::scope_mut(&mut state, ns);
        let key = (graph_kind_name(kind).to_owned(), record.id.clone());
        if !scope.records.contains_key(&key) {
            scope.record_order.push(key.clone());
        }
        scope.records.insert(key, record);
        Ok(())
    }

    async fn delete_graph_record_of_kind(
        &self,
        ns: &str,
        kind: GraphEntityKind,
        id: &str,
    ) -> StoreResult<bool> {
        namespace(ns)?;
        let mut state = self.state.write().expect("in-memory store lock poisoned");
        let scope = Self::scope_mut(&mut state, ns);
        let key = (graph_kind_name(kind).to_owned(), id.to_owned());
        scope.record_order.retain(|record_key| record_key != &key);
        Ok(scope.records.remove(&key).is_some())
    }
}

impl GraphMutationStore for InMemoryStore {
    fn graph_authority_capability(&self) -> GraphAuthorityCapability {
        GraphAuthorityCapability::AtomicBatchWithIndexJobs
    }

    async fn apply_graph_mutation(
        &self,
        mutation: GraphMutation,
    ) -> StoreResult<AppliedGraphMutation> {
        InMemoryStore::apply_graph_mutation(self, mutation).await
    }

    async fn apply_graph_mutations_atomic(
        &self,
        mutations: Vec<GraphMutation>,
    ) -> StoreResult<Vec<AppliedGraphMutation>> {
        self.apply_graph_mutations_with_index_jobs_atomic(GraphMutationBatch {
            mutations,
            index_jobs: Vec::new(),
        })
        .await
    }

    async fn apply_graph_mutations_with_index_jobs_atomic(
        &self,
        batch: GraphMutationBatch,
    ) -> StoreResult<Vec<AppliedGraphMutation>> {
        InMemoryStore::apply_graph_mutations_with_index_jobs_atomic(self, batch).await
    }

    async fn graph_projection_records(
        &self,
        read: GraphProjectionRead,
    ) -> StoreResult<Vec<GraphRecord>> {
        let mut rows = self.graph_projection_rows(&read)?;
        rows.truncate(read.limit);
        Ok(rows)
    }

    async fn graph_projection_page(
        &self,
        page: GraphProjectionPageRead,
    ) -> StoreResult<Vec<GraphRecord>> {
        let GraphProjectionPageRead { read, after_id } = page;
        let mut rows = self.graph_projection_rows(&read)?;
        if let Some(after_id) = after_id {
            rows.retain(|record| record.id > after_id);
        }
        rows.sort_by(|left, right| left.id.cmp(&right.id));
        rows.truncate(read.limit);
        Ok(rows)
    }

    async fn graph_projection_vector_query(
        &self,
        query: GraphProjectionVectorQuery,
    ) -> StoreResult<Vec<VectorMatch>> {
        if query.query.embedding.len() != query.embedding_dim {
            return Err(StoreError::VectorDimensionMismatch {
                expected: query.embedding_dim,
                actual: query.query.embedding.len(),
            });
        }
        namespace(&query.scope.namespace)?;
        finite(&query.query.embedding)?;
        let state = self.state.read().expect("in-memory store lock poisoned");
        let projection = state.graph_projections.get(&query.scope);
        graph_projection_vector_matches(projection, &query.table, &query.query)
    }
}

impl EventReadStore for InMemoryStore {
    async fn replay_events(
        &self,
        ns: &str,
        after: i64,
        limit: usize,
    ) -> StoreResult<Vec<EntityEvent>> {
        namespace(ns)?;
        let state = self.state.read().expect("in-memory store lock poisoned");
        Ok(Self::scope(&state, ns)
            .map(|scope| {
                scope
                    .events
                    .iter()
                    .filter(|event| event.seq > after)
                    .take(limit)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default())
    }
    async fn replay_cursor(&self, ns: &str, consumer: &str) -> StoreResult<ReplayCursor> {
        namespace(ns)?;
        let state = self.state.read().expect("in-memory store lock poisoned");
        let last_seq = Self::scope(&state, ns)
            .and_then(|scope| scope.cursors.get(consumer))
            .copied()
            .unwrap_or(0);
        Ok(ReplayCursor {
            namespace: ns.to_owned(),
            consumer: consumer.to_owned(),
            last_seq,
        })
    }
    async fn latest_event_seq(&self, ns: &str) -> StoreResult<i64> {
        namespace(ns)?;
        let state = self.state.read().expect("in-memory store lock poisoned");
        Ok(Self::scope(&state, ns)
            .and_then(|scope| scope.events.last().map(|event| event.seq))
            .unwrap_or(0))
    }
}

impl EventPruneStore for InMemoryStore {
    async fn prune_entity_events_after(&self, ns: &str, to_seq: i64) -> StoreResult<u64> {
        namespace(ns)?;
        let mut state = self.state.write().expect("in-memory store lock poisoned");
        let Some(scope) = state.namespaces.get_mut(ns) else {
            return Ok(0);
        };
        let first_removed = scope.events.partition_point(|event| event.seq <= to_seq);
        let deleted = scope.events.len() - first_removed;
        if deleted == 0 {
            return Ok(0);
        }
        for event in scope.events.drain(first_removed..) {
            scope.event_seq_by_id.remove(&event.event_id);
        }
        Ok(u64::try_from(deleted).expect("collection length exceeds u64"))
    }
}

impl WorkflowDesignHistoryReadStore for InMemoryStore {
    async fn workflow_design_snapshot(
        &self,
        workflow_id: &str,
        max_version: i64,
        schema_version: i64,
    ) -> StoreResult<Option<WorkflowDesignSnapshot>> {
        let state = self.state.read().expect("in-memory store lock poisoned");
        Ok(state
            .workflow_design_snapshots
            .range((workflow_id.to_owned(), i64::MIN)..=(workflow_id.to_owned(), max_version))
            .rev()
            .find(|(_, snapshot)| snapshot.schema_version == schema_version)
            .map(|(_, snapshot)| snapshot.clone()))
    }

    async fn workflow_design_delta(
        &self,
        workflow_id: &str,
        version: i64,
        schema_version: i64,
    ) -> StoreResult<Option<WorkflowDesignDelta>> {
        let state = self.state.read().expect("in-memory store lock poisoned");
        Ok(state
            .workflow_design_deltas
            .get(&(workflow_id.to_owned(), version))
            .filter(|delta| delta.schema_version == schema_version)
            .cloned())
    }
}

impl WorkflowDesignHistoryWriteStore for InMemoryStore {
    async fn put_workflow_design_snapshot(
        &self,
        workflow_id: &str,
        snapshot: WorkflowDesignSnapshotWrite,
    ) -> StoreResult<()> {
        let mut state = self.state.write().expect("in-memory store lock poisoned");
        state.workflow_design_snapshots.insert(
            (workflow_id.to_owned(), snapshot.version),
            WorkflowDesignSnapshot {
                workflow_id: workflow_id.to_owned(),
                version: snapshot.version,
                seq: snapshot.seq,
                payload_json: snapshot.payload_json,
                schema_version: snapshot.schema_version,
                created_at_ms: now_ms(),
            },
        );
        Ok(())
    }

    async fn clear_workflow_design_snapshots(&self, workflow_id: &str) -> StoreResult<()> {
        let mut state = self.state.write().expect("in-memory store lock poisoned");
        state
            .workflow_design_snapshots
            .retain(|(stored_workflow_id, _), _| stored_workflow_id != workflow_id);
        Ok(())
    }

    async fn put_workflow_design_delta(
        &self,
        workflow_id: &str,
        delta: WorkflowDesignDeltaWrite,
    ) -> StoreResult<()> {
        let mut state = self.state.write().expect("in-memory store lock poisoned");
        state.workflow_design_deltas.insert(
            (workflow_id.to_owned(), delta.version),
            WorkflowDesignDelta {
                workflow_id: workflow_id.to_owned(),
                version: delta.version,
                prev_version: delta.prev_version,
                target_seq: delta.target_seq,
                forward_json: delta.forward_json,
                inverse_json: delta.inverse_json,
                schema_version: delta.schema_version,
                created_at_ms: now_ms(),
            },
        );
        Ok(())
    }

    async fn clear_workflow_design_deltas(&self, workflow_id: &str) -> StoreResult<()> {
        let mut state = self.state.write().expect("in-memory store lock poisoned");
        state
            .workflow_design_deltas
            .retain(|(stored_workflow_id, _), _| stored_workflow_id != workflow_id);
        Ok(())
    }
}

impl EventWriteStore for InMemoryStore {
    async fn append_entity_event(
        &self,
        ns: &str,
        event: NewEntityEvent,
    ) -> StoreResult<AppendedEvent> {
        namespace(ns)?;
        let mut state = self.state.write().expect("in-memory store lock poisoned");
        let scope = Self::scope_mut(&mut state, ns);
        if let Some(seq) = scope.event_seq_by_id.get(&event.event_id) {
            let existing = scope
                .events
                .iter()
                .find(|current| current.seq == *seq)
                .expect("event identity index must reference an event")
                .clone();
            return Ok(AppendedEvent {
                event: existing,
                inserted: false,
            });
        }
        let seq = scope.next_event_seq + 1;
        scope.next_event_seq = seq;
        let envelope = EntityEventEnvelope {
            namespace: ns.to_owned(),
            seq,
            event_id: event.event_id.clone(),
            entity_kind: event.entity_kind,
            entity_id: event.entity_id,
            op: event.op,
            payload: event.payload,
        };
        scope.event_seq_by_id.insert(event.event_id, seq);
        scope.events.push(envelope.clone());
        Ok(AppendedEvent {
            event: envelope,
            inserted: true,
        })
    }
    async fn advance_replay_cursor(
        &self,
        ns: &str,
        consumer: &str,
        last_seq: i64,
    ) -> StoreResult<ReplayCursor> {
        namespace(ns)?;
        let mut state = self.state.write().expect("in-memory store lock poisoned");
        let scope = Self::scope_mut(&mut state, ns);
        if last_seq > scope.next_event_seq {
            return Err(StoreError::CursorOutOfRange {
                cursor: last_seq,
                latest: scope.next_event_seq,
            });
        }
        let current = scope.cursors.get(consumer).copied().unwrap_or(0);
        if last_seq < current {
            return Err(StoreError::CursorRegresses {
                current,
                requested: last_seq,
            });
        }
        scope.cursors.insert(consumer.to_owned(), last_seq);
        Ok(ReplayCursor {
            namespace: ns.to_owned(),
            consumer: consumer.to_owned(),
            last_seq,
        })
    }
}

impl ProjectionReadStore for InMemoryStore {
    async fn named_projection(&self, ns: &str, key: &str) -> StoreResult<Option<NamedProjection>> {
        namespace(ns)?;
        let state = self.state.read().expect("in-memory store lock poisoned");
        Ok(Self::scope(&state, ns)
            .and_then(|scope| scope.projections.get(key))
            .cloned())
    }

    async fn named_projections(&self, ns: &str) -> StoreResult<Vec<NamedProjection>> {
        namespace(ns)?;
        let state = self.state.read().expect("in-memory store lock poisoned");
        Ok(Self::scope(&state, ns)
            .map(|scope| scope.projections.values().cloned().collect())
            .unwrap_or_default())
    }
}

impl ProjectionWriteStore for InMemoryStore {
    async fn replace_named_projection(
        &self,
        ns: &str,
        key: &str,
        projection: NamedProjectionWrite,
    ) -> StoreResult<()> {
        namespace(ns)?;
        let mut state = self.state.write().expect("in-memory store lock poisoned");
        Self::scope_mut(&mut state, ns).projections.insert(
            key.to_owned(),
            NamedProjection {
                namespace: ns.to_owned(),
                key: key.to_owned(),
                payload: projection.payload,
                last_authoritative_seq: projection.last_authoritative_seq,
                last_materialized_seq: projection.last_materialized_seq,
                projection_schema_version: projection.projection_schema_version,
                materialization_status: projection.materialization_status,
                updated_at_ms: now_ms(),
            },
        );
        Ok(())
    }

    async fn compare_and_swap_named_projection(
        &self,
        ns: &str,
        key: &str,
        expected_last_authoritative_seq: Option<i64>,
        expected_last_materialized_seq: Option<i64>,
        projection: NamedProjectionWrite,
    ) -> StoreResult<bool> {
        namespace(ns)?;
        let mut state = self.state.write().expect("in-memory store lock poisoned");
        let existing = Self::scope(&state, ns)
            .and_then(|scope| scope.projections.get(key))
            .cloned();
        let matches = match (
            expected_last_authoritative_seq,
            expected_last_materialized_seq,
            existing.as_ref(),
        ) {
            (None, None, None) => true,
            (Some(authoritative), Some(materialized), Some(row)) => {
                row.last_authoritative_seq == authoritative
                    && row.last_materialized_seq == materialized
            }
            _ => false,
        };
        if !matches {
            return Ok(false);
        }
        Self::scope_mut(&mut state, ns).projections.insert(
            key.to_owned(),
            NamedProjection {
                namespace: ns.to_owned(),
                key: key.to_owned(),
                payload: projection.payload,
                last_authoritative_seq: projection.last_authoritative_seq,
                last_materialized_seq: projection.last_materialized_seq,
                projection_schema_version: projection.projection_schema_version,
                materialization_status: projection.materialization_status,
                updated_at_ms: now_ms(),
            },
        );
        Ok(true)
    }

    async fn clear_named_projection(&self, ns: &str, key: &str) -> StoreResult<()> {
        namespace(ns)?;
        let mut state = self.state.write().expect("in-memory store lock poisoned");
        if let Some(scope) = state.namespaces.get_mut(ns) {
            scope.projections.remove(key);
        }
        Ok(())
    }

    async fn clear_projection_namespace(&self, ns: &str) -> StoreResult<()> {
        namespace(ns)?;
        let mut state = self.state.write().expect("in-memory store lock poisoned");
        if let Some(scope) = state.namespaces.get_mut(ns) {
            scope.projections.clear();
        }
        Ok(())
    }
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock precedes Unix epoch")
        .as_millis() as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    use kogwistar_store::{
        GraphAuthorityCapability, GraphMutation, GraphMutationStore, GraphProjectionRead,
        GraphProjectionVectorQuery, GraphRecord, GraphScope, NewProjectedLaneMessage,
        NodeIndexStore, ProjectionWriteStore, ServerRunReadStore, ServerRunWriteStore,
        ShadowInspectable, VectorCollectionStore,
    };
    use serde_json::{Value, json};
    use std::future::Future;
    use std::pin::pin;
    use std::task::{Context, Poll, Waker};

    fn block_on<T>(future: impl Future<Output = T>) -> T {
        let waker = Waker::noop();
        let mut context = Context::from_waker(waker);
        let mut future = pin!(future);
        loop {
            match future.as_mut().poll(&mut context) {
                Poll::Ready(value) => return value,
                Poll::Pending => std::thread::yield_now(),
            }
        }
    }

    #[test]
    fn graph_authority_capability_declares_atomic_batches_and_index_jobs() {
        let capability = InMemoryStore::new().graph_authority_capability();
        assert_eq!(
            capability,
            GraphAuthorityCapability::AtomicBatchWithIndexJobs
        );
        assert!(capability.supports_single_mutation());
        assert!(capability.supports_atomic_index_jobs());
    }

    #[test]
    fn cloned_handles_share_volatile_state_but_new_store_starts_empty() {
        let store = InMemoryStore::new();
        let shared_handle = store.clone();
        let scope = GraphScope {
            namespace: "volatile-contract".to_owned(),
            workspace_id: None,
            graph_space: None,
        };
        block_on(store.apply_graph_mutation(graph_mutation(
            scope.clone(),
            "volatile-event",
            "volatile-node",
            &[1.0, 0.0],
        )))
        .expect("write to shared in-memory state");

        assert_eq!(
            block_on(shared_handle.replay_events(&scope.namespace, 0, 10))
                .expect("cloned handle reads shared state")
                .len(),
            1
        );
        assert!(
            block_on(InMemoryStore::new().replay_events(&scope.namespace, 0, 10))
                .expect("fresh store has independent volatile state")
                .is_empty()
        );
    }

    #[test]
    fn named_vector_collection_isolated_from_graph_and_other_namespaces() {
        let store = InMemoryStore::new();
        let record = |id: &str, embedding: Vec<f32>| GraphRecord {
            id: id.to_owned(),
            document: Some(format!("index entry {id}")),
            metadata: serde_json::Map::from_iter([("entry".to_owned(), json!(true))]),
            embedding: Some(embedding),
        };
        block_on(store.upsert_vector_collection(
            "knowledge",
            "node_index",
            vec![
                record("idx:one", vec![1.0, 0.0]),
                record("idx:two", vec![0.0, 1.0]),
            ],
        ))
        .expect("named vector entries upsert");
        block_on(store.upsert_vector_collection(
            "other",
            "node_index",
            vec![record("idx:private", vec![1.0, 0.0])],
        ))
        .expect("other namespace entry upsert");

        let query = VectorQuery {
            embedding: vec![1.0, 0.0],
            limit: 10,
            metadata: MetadataFilter::default(),
            metric: DistanceMetric::Cosine,
        };
        let results = block_on(store.query_vector_collection("knowledge", "node_index", &query))
            .expect("query named vector collection");
        assert_eq!(
            results
                .iter()
                .map(|item| item.record.id.as_str())
                .collect::<Vec<_>>(),
            ["idx:one", "idx:two"]
        );
        assert_eq!(results[0].distance, 0.0);
        assert!(
            block_on(store.graph_record("knowledge", "idx:one"))
                .expect("graph read")
                .is_none()
        );
        assert!(
            block_on(store.query_vector_collection("missing", "node_index", &query))
                .expect("isolated namespace query")
                .is_empty()
        );

        block_on(store.delete_vector_collection_records(
            "knowledge",
            "node_index",
            vec!["idx:one".to_owned()],
        ))
        .expect("delete from only the selected collection");
        let remaining = block_on(store.query_vector_collection("knowledge", "node_index", &query))
            .expect("query after delete");
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].record.id, "idx:two");
    }

    #[test]
    fn dedicated_node_index_store_preserves_lookup_order_and_collection_isolation() {
        let store = InMemoryStore::new();
        let make_record = |id: &str, embedding: Vec<f32>| GraphRecord {
            id: id.to_owned(),
            document: Some(format!("entry {id}")),
            metadata: serde_json::Map::new(),
            embedding: Some(embedding),
        };
        block_on(store.upsert_node_index(
            "graph-index",
            vec![
                make_record("idx:a", vec![1.0, 0.0]),
                make_record("idx:b", vec![0.0, 1.0]),
            ],
        ))
        .expect("index upsert");

        let ids = vec!["idx:b".to_owned(), "missing".to_owned(), "idx:a".to_owned()];
        let records = block_on(store.get_node_index("graph-index", &ids)).expect("index read");
        assert_eq!(
            records
                .iter()
                .map(|record| record.id.as_str())
                .collect::<Vec<_>>(),
            ["idx:b", "idx:a"]
        );
        assert!(
            block_on(store.get_node_index("other-index", &ids))
                .expect("isolated index")
                .is_empty()
        );

        let matches = block_on(store.query_node_index(
            "graph-index",
            &VectorQuery {
                embedding: vec![1.0, 0.0],
                limit: 1,
                metadata: MetadataFilter::default(),
                metric: DistanceMetric::Cosine,
            },
        ))
        .expect("index query");
        assert_eq!(matches[0].record.id, "idx:a");
    }

    #[test]
    fn graph_mutation_batch_commits_events_projection_and_index_jobs_atomically() {
        let store = InMemoryStore::new();
        let scope = GraphScope {
            namespace: "memory-atomic".to_owned(),
            workspace_id: Some("workspace".to_owned()),
            graph_space: Some("knowledge".to_owned()),
        };
        let first = graph_mutation(scope.clone(), "first-event", "first", &[1.0, 0.0]);
        let second = graph_mutation(scope.clone(), "second-event", "second", &[0.0, 1.0]);
        let job = NewIndexJob {
            job_id: "first-index-job".to_owned(),
            namespace: scope.namespace.clone(),
            entity_kind: "node".to_owned(),
            entity_id: "first".to_owned(),
            index_kind: "embedding".to_owned(),
            op: "UPSERT".to_owned(),
            payload_json: Some("{}".to_owned()),
            max_retries: 3,
        };
        let batch = GraphMutationBatch {
            mutations: vec![first.clone(), second],
            index_jobs: vec![job],
        };

        let applied = block_on(store.apply_graph_mutations_with_index_jobs_atomic(batch.clone()))
            .expect("valid batch commits");
        assert_eq!(applied.len(), 2);
        assert!(applied.iter().all(|item| item.inserted && item.mutated));
        assert_eq!(
            block_on(store.replay_events(&scope.namespace, 0, 10))
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            block_on(store.graph_projection_records(GraphProjectionRead {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                ids: None,
                metadata: MetadataFilter::default(),
                limit: 10,
            }))
            .unwrap()
            .len(),
            2
        );
        assert_eq!(
            block_on(store.index_jobs(
                Some(&scope.namespace),
                Some("PENDING"),
                None,
                None,
                None,
                10,
            ))
            .unwrap()
            .len(),
            1
        );

        let replay = block_on(store.apply_graph_mutations_with_index_jobs_atomic(batch))
            .expect("exact batch replay is idempotent");
        assert!(replay.iter().all(|item| !item.inserted && !item.mutated));
        assert_eq!(
            block_on(store.replay_events(&scope.namespace, 0, 10))
                .unwrap()
                .len(),
            2
        );
    }

    #[test]
    fn graph_node_edge_snapshot_never_mixes_atomic_batch_revisions() {
        fn graph_batch(scope: &GraphScope, generation: i64) -> GraphMutationBatch {
            let mutation = |table: &str, kind: &str, id: &str| GraphMutation {
                scope: scope.clone(),
                table: table.to_owned(),
                entity_kind: kind.to_owned(),
                event_id: format!("{kind}-{generation}"),
                op: if generation == 0 { "ADD" } else { "REPLACE" }.to_owned(),
                payload: json!({"id": id, "generation": generation}),
                record: GraphRecord {
                    id: id.to_owned(),
                    document: None,
                    metadata: json!({"generation": generation})
                        .as_object()
                        .unwrap()
                        .clone(),
                    embedding: None,
                },
                embedding_dim: 0,
            };
            GraphMutationBatch {
                mutations: vec![
                    mutation("gke_nodes", "node", "node-1"),
                    mutation("gke_edges", "edge", "edge-1"),
                ],
                index_jobs: vec![],
            }
        }

        let store = InMemoryStore::new();
        let scope = GraphScope {
            namespace: "memory-snapshot".to_owned(),
            workspace_id: Some("workspace".to_owned()),
            graph_space: Some("knowledge".to_owned()),
        };
        block_on(store.apply_graph_mutations_with_index_jobs_atomic(graph_batch(&scope, 0)))
            .expect("initial graph pair is valid");

        let barrier = Arc::new(std::sync::Barrier::new(2));
        let writer_store = store.clone();
        let writer_scope = scope.clone();
        let writer_barrier = barrier.clone();
        let writer =
            std::thread::spawn(move || {
                writer_barrier.wait();
                for generation in 1..=500 {
                    block_on(writer_store.apply_graph_mutations_with_index_jobs_atomic(
                        graph_batch(&writer_scope, generation),
                    ))
                    .expect("replacement graph pair is atomic");
                }
            });
        barrier.wait();
        for _ in 0..1_000 {
            let (nodes, edges) = store
                .graph_node_edge_snapshot(&scope, 10)
                .expect("snapshot read succeeds");
            let node_generation = nodes[0].metadata["generation"].as_i64().unwrap();
            let edge_generation = edges[0].metadata["generation"].as_i64().unwrap();
            assert_eq!(node_generation, edge_generation);
        }
        writer.join().expect("writer thread completes");
    }

    #[test]
    fn conflicting_graph_writes_are_serialized_within_process() {
        let store = InMemoryStore::new();
        let scope = GraphScope {
            namespace: "memory-conflicting-writes".to_owned(),
            workspace_id: Some("workspace".to_owned()),
            graph_space: Some("knowledge".to_owned()),
        };
        let barrier = Arc::new(std::sync::Barrier::new(3));
        let mut writers = Vec::new();

        for writer_id in ["writer-a", "writer-b"] {
            let writer_store = store.clone();
            let writer_scope = scope.clone();
            let writer_barrier = barrier.clone();
            writers.push(std::thread::spawn(move || {
                let mut mutation = graph_mutation(
                    writer_scope,
                    &format!("event-{writer_id}"),
                    "shared-node",
                    &[1.0, 0.0],
                );
                mutation.op = "REPLACE".to_owned();
                mutation.payload = json!({"id": "shared-node", "writer": writer_id});
                mutation
                    .record
                    .metadata
                    .insert("writer".to_owned(), json!(writer_id));

                writer_barrier.wait();
                block_on(writer_store.apply_graph_mutation(mutation))
                    .expect("serialized conflicting write succeeds")
            }));
        }
        barrier.wait();
        for writer in writers {
            writer.join().expect("writer thread completes");
        }

        let events =
            block_on(store.replay_events(&scope.namespace, 0, 10)).expect("event replay succeeds");
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].seq + 1, events[1].seq);

        let records = block_on(store.graph_projection_records(GraphProjectionRead {
            scope,
            table: "gke_nodes".to_owned(),
            ids: None,
            metadata: MetadataFilter::default(),
            limit: 10,
        }))
        .expect("materialized graph read succeeds");
        assert_eq!(records.len(), 1);
        assert_eq!(
            records[0].metadata.get("writer"),
            events[1].payload.get("writer")
        );
    }

    #[test]
    fn graph_vector_snapshot_never_mixes_atomic_graph_revisions() {
        fn graph_batch(scope: &GraphScope, generation: i64) -> GraphMutationBatch {
            let vectors = if generation % 2 == 0 {
                [("north", [1.0, 0.0]), ("east", [0.0, 1.0])]
            } else {
                [("north", [0.0, 1.0]), ("east", [1.0, 0.0])]
            };
            let mutations = vectors
                .into_iter()
                .map(|(id, embedding)| GraphMutation {
                    scope: scope.clone(),
                    table: "gke_nodes".to_owned(),
                    entity_kind: "node".to_owned(),
                    event_id: format!("{id}-{generation}"),
                    op: if generation == 0 { "ADD" } else { "REPLACE" }.to_owned(),
                    payload: json!({"id": id, "generation": generation}),
                    record: GraphRecord {
                        id: id.to_owned(),
                        document: Some(generation.to_string()),
                        metadata: json!({"generation": generation})
                            .as_object()
                            .unwrap()
                            .clone(),
                        embedding: Some(embedding.to_vec()),
                    },
                    embedding_dim: 2,
                })
                .collect();
            GraphMutationBatch {
                mutations,
                index_jobs: Vec::new(),
            }
        }

        let store = InMemoryStore::new();
        let scope = GraphScope {
            namespace: "memory-vector-snapshot".to_owned(),
            workspace_id: Some("workspace".to_owned()),
            graph_space: Some("knowledge".to_owned()),
        };
        block_on(store.apply_graph_mutations_with_index_jobs_atomic(graph_batch(&scope, 0)))
            .expect("initial graph batch is valid");

        let barrier = Arc::new(std::sync::Barrier::new(2));
        let writer_store = store.clone();
        let writer_scope = scope.clone();
        let writer_barrier = barrier.clone();
        let writer =
            std::thread::spawn(move || {
                writer_barrier.wait();
                for generation in 1..=500 {
                    block_on(writer_store.apply_graph_mutations_with_index_jobs_atomic(
                        graph_batch(&writer_scope, generation),
                    ))
                    .expect("replacement graph batch is atomic");
                }
            });
        barrier.wait();
        let query = VectorQuery {
            embedding: vec![1.0, 0.0],
            limit: 10,
            metadata: MetadataFilter::default(),
            metric: DistanceMetric::Cosine,
        };
        for _ in 0..1_000 {
            let (nodes, edges, matches) = store
                .graph_node_edge_vector_snapshot(&scope, 10, "gke_nodes", &[query.clone()])
                .expect("consistent graph and vector snapshot succeeds");
            assert!(edges.is_empty());
            let generation = nodes[0].metadata["generation"].as_i64().unwrap();
            assert!(
                nodes
                    .iter()
                    .all(|node| node.metadata["generation"].as_i64() == Some(generation))
            );
            let best = matches[0].first().expect("vector candidate exists");
            assert_eq!(
                best.record.metadata["generation"].as_i64(),
                Some(generation)
            );
            assert_eq!(
                best.record.id,
                if generation % 2 == 0 { "north" } else { "east" }
            );
        }
        writer.join().expect("writer thread completes");
    }

    #[test]
    fn invalid_graph_mutation_batch_has_no_partial_memory_effects() {
        let store = InMemoryStore::new();
        let scope = GraphScope {
            namespace: "memory-rollback".to_owned(),
            workspace_id: None,
            graph_space: None,
        };
        let first = graph_mutation(scope.clone(), "valid-event", "valid", &[1.0, 0.0]);
        let mut invalid = graph_mutation(scope.clone(), "invalid-event", "invalid", &[0.0, 1.0]);
        invalid.payload = serde_json::json!({"id": 4});
        let result = block_on(store.apply_graph_mutations_with_index_jobs_atomic(
            GraphMutationBatch {
                mutations: vec![first, invalid],
                index_jobs: vec![NewIndexJob {
                    job_id: "must-not-appear".to_owned(),
                    namespace: scope.namespace.clone(),
                    entity_kind: "node".to_owned(),
                    entity_id: "valid".to_owned(),
                    index_kind: "embedding".to_owned(),
                    op: "UPSERT".to_owned(),
                    payload_json: None,
                    max_retries: 3,
                }],
            },
        ));
        assert!(result.is_err());
        assert!(
            block_on(store.replay_events(&scope.namespace, 0, 10))
                .unwrap()
                .is_empty()
        );
        assert!(
            block_on(store.index_jobs(Some(&scope.namespace), None, None, None, None, 10,))
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn direct_graph_records_keep_node_and_edge_identity_disjoint() {
        let store = InMemoryStore::new();
        block_on(store.upsert_graph_record_of_kind(
            "typed-direct",
            GraphEntityKind::Node,
            record("shared-id", "node", &[1.0, 0.0]),
        ))
        .unwrap();
        block_on(store.upsert_graph_record_of_kind(
            "typed-direct",
            GraphEntityKind::Edge,
            record("shared-id", "edge", &[0.0, 1.0]),
        ))
        .unwrap();

        let node = block_on(store.graph_record_of_kind(
            "typed-direct",
            GraphEntityKind::Node,
            "shared-id",
        ))
        .unwrap()
        .unwrap();
        let edge = block_on(store.graph_record_of_kind(
            "typed-direct",
            GraphEntityKind::Edge,
            "shared-id",
        ))
        .unwrap()
        .unwrap();
        assert_eq!(node.metadata["team"], "node");
        assert_eq!(edge.metadata["team"], "edge");
        assert_eq!(
            block_on(store.graph_records_of_kind(
                "typed-direct",
                GraphEntityKind::Node,
                &MetadataFilter::default(),
            ))
            .unwrap()
            .len(),
            1
        );
        assert_eq!(
            block_on(store.graph_records_of_kind(
                "typed-direct",
                GraphEntityKind::Edge,
                &MetadataFilter::default(),
            ))
            .unwrap()
            .len(),
            1
        );

        let query = VectorQuery {
            embedding: vec![1.0, 0.0],
            limit: 5,
            metadata: MetadataFilter::default(),
            metric: DistanceMetric::Cosine,
        };
        let node_hits =
            block_on(store.vector_query_of_kind("typed-direct", GraphEntityKind::Node, &query))
                .unwrap();
        let edge_hits =
            block_on(store.vector_query_of_kind("typed-direct", GraphEntityKind::Edge, &query))
                .unwrap();
        assert_eq!(node_hits[0].distance, 0.0);
        assert_eq!(edge_hits[0].distance, 1.0);
        assert!(block_on(store.graph_record("typed-direct", "shared-id")).is_err());
        assert!(block_on(store.delete_graph_record("typed-direct", "shared-id")).is_err());
        assert!(
            block_on(store.delete_graph_record_of_kind(
                "typed-direct",
                GraphEntityKind::Node,
                "shared-id",
            ))
            .unwrap()
        );
        assert!(
            block_on(store.graph_record_of_kind(
                "typed-direct",
                GraphEntityKind::Node,
                "shared-id",
            ))
            .unwrap()
            .is_none()
        );
        assert!(
            block_on(store.graph_record_of_kind(
                "typed-direct",
                GraphEntityKind::Edge,
                "shared-id",
            ))
            .unwrap()
            .is_some()
        );
    }

    #[test]
    fn numeric_lte_vector_filter_is_applied_before_top_n() {
        let store = InMemoryStore::new();
        for (id, level, embedding) in [
            ("nearest-deep", 4, vec![1.0_f32, 0.0]),
            ("next-shallow", 1, vec![0.9, 0.1]),
            ("last-shallow", 2, vec![0.8, 0.2]),
        ] {
            let mut metadata = serde_json::Map::new();
            metadata.insert("level_from_root".to_owned(), json!(level));
            block_on(store.upsert_graph_record_of_kind(
                "level-filter",
                GraphEntityKind::Node,
                GraphRecord {
                    id: id.to_owned(),
                    document: None,
                    metadata,
                    embedding: Some(embedding),
                },
            ))
            .unwrap();
        }
        let query = VectorQuery {
            embedding: vec![1.0, 0.0],
            limit: 1,
            metadata: MetadataFilter::default(),
            metric: DistanceMetric::Cosine,
        };
        let matches = block_on(store.vector_query_of_kind_numeric_lte(
            "level-filter",
            GraphEntityKind::Node,
            &query,
            "level_from_root",
            2.0,
        ))
        .unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].record.id, "next-shallow");
        let shadow = store.shadow_inspection();
        let shadow_matches = block_on(shadow.vector_query_of_kind_numeric_lte(
            "level-filter",
            GraphEntityKind::Node,
            &query,
            "level_from_root",
            2.0,
        ))
        .unwrap();
        assert_eq!(shadow_matches, matches);
    }

    #[test]
    fn expired_claim_token_cannot_mutate_queue_state() {
        let store = InMemoryStore::new();
        for job_id in ["done", "failed", "retried", "requeued"] {
            block_on(store.enqueue_index_job(NewIndexJob {
                job_id: job_id.to_owned(),
                namespace: "ns".to_owned(),
                entity_kind: "node".to_owned(),
                entity_id: job_id.to_owned(),
                index_kind: "test".to_owned(),
                op: "UPSERT".to_owned(),
                payload_json: None,
                max_retries: 3,
            }))
            .unwrap();
        }
        let claims = block_on(store.claim_index_jobs(4, -1, Some("ns"))).unwrap();
        let tokens = claims
            .iter()
            .map(|claim| (claim.job_id.as_str(), claim.claim_token.as_deref().unwrap()))
            .collect::<BTreeMap<_, _>>();

        assert!(!block_on(store.mark_index_job_done("done", Some(tokens["done"]))).unwrap());
        block_on(store.mark_index_job_failed("failed", "late", true, Some(tokens["failed"])))
            .unwrap();
        block_on(store.bump_retry_and_requeue("retried", "late", 0, Some(tokens["retried"])))
            .unwrap();
        block_on(store.requeue_index_job_at_tail(
            "requeued",
            "{}".to_owned(),
            0,
            Some(tokens["requeued"]),
        ))
        .unwrap();

        let reclaimed = block_on(store.claim_index_jobs(4, 30, Some("ns"))).unwrap();
        assert_eq!(reclaimed.len(), 4);
        assert!(reclaimed.iter().all(|claim| claim.retry_count == 0));
        assert!(
            reclaimed.iter().all(|claim| {
                claim.claim_token.as_deref() != Some(tokens[claim.job_id.as_str()])
            })
        );
        assert!(block_on(store.mark_index_job_done("done", None)).unwrap());
    }

    #[test]
    fn expired_lane_claim_cannot_be_finalized_before_reclaim() {
        let store = InMemoryStore::new();
        for (message_id, operation) in [
            ("expired-ack", "ack"),
            ("expired-requeue", "requeue"),
            ("expired-dead-letter", "dead-letter"),
        ] {
            block_on(store.project_lane_message(NewProjectedLaneMessage {
                message_id: message_id.to_owned(),
                namespace: "runtime".to_owned(),
                purpose: "user_visible".to_owned(),
                inbox_id: "python-workers".to_owned(),
                conversation_id: "conversation".to_owned(),
                recipient_id: "python-worker".to_owned(),
                sender_id: "agent".to_owned(),
                msg_type: "workflow.worker.request.v1".to_owned(),
                status: "pending".to_owned(),
                created_at: 1,
                available_at: 0,
                run_id: Some("run-target".to_owned()),
                step_id: Some("step".to_owned()),
                correlation_id: Some("correlation".to_owned()),
                payload_json: Some("{}".to_owned()),
                error_json: None,
            }))
            .unwrap();
            let claimed = block_on(store.claim_projected_lane_messages_filtered(
                LaneMessageClaimFilter {
                    namespace: "runtime".to_owned(),
                    inbox_id: "python-workers".to_owned(),
                    claimed_by: "attempt-1".to_owned(),
                    message_ids: Some(vec![message_id.to_owned()]),
                    run_id: Some("run-target".to_owned()),
                    msg_type: Some("workflow.worker.request.v1".to_owned()),
                    recipient_id: Some("python-worker".to_owned()),
                    limit: 1,
                    lease_seconds: -5,
                },
            ))
            .unwrap();
            assert_eq!(claimed.len(), 1);
            assert!(timestamp_i64(claimed[0].lease_until.as_ref().unwrap()) < now_seconds());

            match operation {
                "ack" => block_on(store.ack_projected_lane_message(message_id, "attempt-1")),
                "requeue" => {
                    block_on(store.requeue_projected_lane_message(message_id, "attempt-1", None, 0))
                }
                "dead-letter" => block_on(store.dead_letter_projected_lane_message(
                    message_id,
                    "attempt-1",
                    None,
                )),
                _ => unreachable!(),
            }
            .unwrap();
            let state = store.state.read().unwrap();
            let row = &state.lane_messages[message_id];
            assert_eq!(row.status, "claimed", "{operation}");
            assert_eq!(row.claimed_by.as_deref(), Some("attempt-1"), "{operation}");
            assert_eq!(row.retry_count, 0, "{operation}");
        }
    }

    fn event(event_id: &str, entity_id: &str) -> NewEntityEvent {
        NewEntityEvent {
            event_id: event_id.to_owned(),
            entity_kind: "node".to_owned(),
            entity_id: entity_id.to_owned(),
            op: "upsert".to_owned(),
            payload: json!({"id": entity_id}),
        }
    }

    fn graph_mutation(
        scope: GraphScope,
        event_id: &str,
        id: &str,
        embedding: &[f32],
    ) -> GraphMutation {
        GraphMutation {
            scope,
            table: "gke_nodes".to_owned(),
            entity_kind: "node".to_owned(),
            event_id: event_id.to_owned(),
            op: "ADD".to_owned(),
            payload: json!({"id": id}),
            record: record(id, "red", embedding),
            embedding_dim: embedding.len(),
        }
    }

    #[test]
    fn typed_graph_mutation_guard_is_atomic_at_the_memory_store_seam() {
        let store = InMemoryStore::new();
        let scope = GraphScope {
            namespace: "typed".to_owned(),
            workspace_id: Some("ws".to_owned()),
            graph_space: Some("facts".to_owned()),
        };
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/python-graph-model-payloads-v1.json"
        )))
        .expect("valid graph model fixture");
        let payload = fixture["cases"][0]["payload"].clone();
        let envelope = kogwistar_contracts::GraphModelEnvelope::new(
            kogwistar_contracts::GraphModelClass::Node,
            payload.clone(),
        );
        let mutation = GraphMutation {
            scope: scope.clone(),
            table: "gke_nodes".to_owned(),
            entity_kind: "node".to_owned(),
            event_id: "typed-event".to_owned(),
            op: "ADD".to_owned(),
            payload: payload.clone(),
            record: GraphRecord {
                id: "parity-node".to_owned(),
                document: None,
                metadata: payload["metadata"].as_object().cloned().unwrap_or_default(),
                embedding: Some(vec![1.0, 0.0, 0.0]),
            },
            embedding_dim: 3,
        };
        let applied = block_on(store.apply_typed_graph_mutation(mutation.clone(), envelope));
        assert!(applied.is_ok());

        let mut invalid = mutation;
        invalid.event_id = "invalid-event".to_owned();
        invalid.record.id = "different-id".to_owned();
        let invalid_envelope = kogwistar_contracts::GraphModelEnvelope::new(
            kogwistar_contracts::GraphModelClass::Node,
            payload,
        );
        assert!(block_on(store.apply_typed_graph_mutation(invalid, invalid_envelope)).is_err());
        block_on(store.apply_graph_mutation(graph_mutation(
            scope.clone(),
            "typed-event-2",
            "a-parity-node",
            &[0.0, 1.0, 0.0],
        )))
        .expect("insert second record for cursor paging");
        let legacy_records = block_on(store.graph_projection_records(GraphProjectionRead {
            scope: scope.clone(),
            table: "gke_nodes".to_owned(),
            ids: None,
            metadata: Default::default(),
            limit: 10,
        }))
        .expect("legacy read preserves insertion order");
        assert_eq!(
            legacy_records
                .iter()
                .map(|record| record.id.as_str())
                .collect::<Vec<_>>(),
            vec!["parity-node", "a-parity-node"]
        );
        let records = block_on(kogwistar_store::graph_projection_records_all(
            &store,
            GraphProjectionRead {
                scope,
                table: "gke_nodes".to_owned(),
                ids: None,
                metadata: Default::default(),
                limit: 1,
            },
        ))
        .expect("read all memory projection pages");
        assert_eq!(
            records
                .iter()
                .map(|record| record.id.as_str())
                .collect::<Vec<_>>(),
            vec!["a-parity-node", "parity-node"]
        );
        let mut visited = Vec::new();
        let mut page_count = 0;
        block_on(kogwistar_store::for_each_graph_projection_page(
            &store,
            GraphProjectionRead {
                scope: GraphScope {
                    namespace: "typed".to_owned(),
                    workspace_id: Some("ws".to_owned()),
                    graph_space: Some("facts".to_owned()),
                },
                table: "gke_nodes".to_owned(),
                ids: None,
                metadata: Default::default(),
                limit: 1,
            },
            |page| {
                page_count += 1;
                visited.extend(page.iter().map(|record| record.id.clone()));
                Ok(())
            },
        ))
        .expect("visit pages without collecting projection records");
        assert_eq!(page_count, 2);
        assert_eq!(visited, vec!["a-parity-node", "parity-node"]);
    }

    #[test]
    fn graph_projection_keeps_same_id_separate_by_table() {
        let store = InMemoryStore::new();
        let scope = GraphScope {
            namespace: "same-id-projection".to_owned(),
            workspace_id: Some("ws".to_owned()),
            graph_space: Some("universal".to_owned()),
        };
        let node = graph_mutation(scope.clone(), "node-event", "shared-id", &[1.0, 0.0]);
        block_on(store.apply_graph_mutation(node)).unwrap();

        let mut edge = graph_mutation(scope.clone(), "edge-event", "shared-id", &[0.0, 1.0]);
        edge.table = "gke_edges".to_owned();
        edge.entity_kind = "edge".to_owned();
        block_on(store.apply_graph_mutation(edge)).unwrap();

        for (table, expected_embedding) in
            [("gke_nodes", vec![1.0, 0.0]), ("gke_edges", vec![0.0, 1.0])]
        {
            let records = block_on(store.graph_projection_records(GraphProjectionRead {
                scope: scope.clone(),
                table: table.to_owned(),
                ids: None,
                metadata: MetadataFilter::default(),
                limit: 10,
            }))
            .unwrap();
            assert_eq!(records.len(), 1, "{table}");
            assert_eq!(records[0].id, "shared-id");
            assert_eq!(records[0].embedding, Some(expected_embedding.clone()));

            let hits = block_on(
                store.graph_projection_vector_query(GraphProjectionVectorQuery {
                    scope: scope.clone(),
                    table: table.to_owned(),
                    query: VectorQuery {
                        embedding: vec![1.0, 0.0],
                        limit: 10,
                        metadata: MetadataFilter::default(),
                        metric: DistanceMetric::Cosine,
                    },
                    embedding_dim: 2,
                }),
            )
            .unwrap();
            assert_eq!(hits.len(), 1, "{table}");
            assert_eq!(hits[0].record.embedding, Some(expected_embedding));
        }
    }

    #[test]
    fn atomic_graph_mutation_batch_is_supported() {
        let store = InMemoryStore::new();
        let scope = GraphScope {
            namespace: "atomic".to_owned(),
            workspace_id: Some("ws".to_owned()),
            graph_space: Some("facts".to_owned()),
        };
        let mutation = graph_mutation(scope, "atomic-event", "atomic-node", &[1.0, 0.0]);

        let applied = block_on(store.apply_graph_mutations_atomic(vec![mutation]))
            .expect("in-memory store supports atomic graph batches");
        assert_eq!(applied.len(), 1);
        assert!(applied[0].inserted && applied[0].mutated);
    }

    #[test]
    fn graph_mutations_are_full_scope_isolated_and_share_namespace_event_sequence() {
        block_on(async {
            let store = InMemoryStore::new();
            let left = GraphScope {
                namespace: "ns".to_owned(),
                workspace_id: Some("one".to_owned()),
                graph_space: Some("facts".to_owned()),
            };
            let right = GraphScope {
                namespace: "ns".to_owned(),
                workspace_id: Some("two".to_owned()),
                graph_space: Some("facts".to_owned()),
            };
            let other_space = GraphScope {
                namespace: "ns".to_owned(),
                workspace_id: Some("one".to_owned()),
                graph_space: Some("other".to_owned()),
            };
            assert_eq!(
                store
                    .append_entity_event("ns", event("normal", "normal"))
                    .await
                    .unwrap()
                    .event
                    .seq,
                1
            );
            assert_eq!(
                store
                    .apply_graph_mutation(graph_mutation(left.clone(), "left", "same", &[1.0, 0.0]))
                    .await
                    .unwrap()
                    .event
                    .seq,
                2
            );
            assert_eq!(
                store
                    .apply_graph_mutation(graph_mutation(
                        right.clone(),
                        "right",
                        "same",
                        &[0.0, 1.0]
                    ))
                    .await
                    .unwrap()
                    .event
                    .seq,
                3
            );
            assert_eq!(
                store
                    .apply_graph_mutation(graph_mutation(
                        other_space.clone(),
                        "space",
                        "same",
                        &[0.5, 0.5],
                    ))
                    .await
                    .unwrap()
                    .event
                    .seq,
                4
            );
            let retry = store
                .apply_graph_mutation(graph_mutation(left.clone(), "left", "changed", &[1.0, 0.0]))
                .await
                .unwrap();
            assert!(!retry.inserted && !retry.mutated && retry.event.entity_id == "same");
            let read = |scope| GraphProjectionRead {
                scope,
                table: "gke_nodes".to_owned(),
                ids: None,
                metadata: MetadataFilter::default(),
                limit: 10,
            };
            assert_eq!(
                store
                    .graph_projection_records(read(left.clone()))
                    .await
                    .unwrap()[0]
                    .embedding,
                Some(vec![1.0, 0.0])
            );
            assert_eq!(
                store
                    .graph_projection_records(read(right.clone()))
                    .await
                    .unwrap()[0]
                    .embedding,
                Some(vec![0.0, 1.0])
            );
            assert_eq!(
                store
                    .graph_projection_records(read(other_space))
                    .await
                    .unwrap()[0]
                    .embedding,
                Some(vec![0.5, 0.5])
            );
            let matches = store
                .graph_projection_vector_query(GraphProjectionVectorQuery {
                    scope: left,
                    table: "gke_nodes".to_owned(),
                    query: VectorQuery {
                        embedding: vec![1.0, 0.0],
                        limit: 10,
                        metadata: MetadataFilter::default(),
                        metric: DistanceMetric::InnerProduct,
                    },
                    embedding_dim: 2,
                })
                .await
                .unwrap();
            assert_eq!(matches[0].record.id, "same");
            assert_eq!(matches[0].distance, -1.0);
        });
    }
    fn record(id: &str, team: &str, embedding: &[f32]) -> GraphRecord {
        let mut result = GraphRecord::new(id);
        result.document = Some(format!("document-{id}"));
        result.metadata.insert("team".to_owned(), json!(team));
        result.embedding = Some(embedding.to_vec());
        result
    }

    #[test]
    fn server_run_memory_keeps_python_divergences() {
        let store = InMemoryStore::new();
        block_on(store.create_server_run(ServerRunCreate {
            run_id: "z".into(),
            conversation_id: "c".into(),
            workflow_id: "w".into(),
            user_id: None,
            user_turn_node_id: "u".into(),
            status: "queued".into(),
        }))
        .unwrap();
        block_on(store.create_server_run(ServerRunCreate {
            run_id: "a".into(),
            conversation_id: "c".into(),
            workflow_id: "w".into(),
            user_id: Some("new".into()),
            user_turn_node_id: "u2".into(),
            status: "running".into(),
        }))
        .unwrap();
        // Explicitly overwrite a duplicate, Python memory semantics.
        block_on(store.create_server_run(ServerRunCreate {
            run_id: "z".into(),
            conversation_id: "c2".into(),
            workflow_id: "w2".into(),
            user_id: None,
            user_turn_node_id: "u3".into(),
            status: "queued".into(),
        }))
        .unwrap();
        assert_eq!(
            block_on(store.server_run("z"))
                .unwrap()
                .unwrap()
                .conversation_id,
            "c2"
        );
        let first = block_on(store.append_server_run_event("z", "one", "{}".into())).unwrap();
        let other = block_on(store.append_server_run_event("a", "one", "{}".into())).unwrap();
        let second = block_on(store.append_server_run_event("z", "two", "{}".into())).unwrap();
        assert_eq!((first.seq, other.seq, second.seq), (1, 1, 2));
        let rows = block_on(store.server_runs(None, None, None, 10)).unwrap();
        assert!(
            rows.windows(2)
                .all(|items| (items[0].created_at_ms, &items[0].run_id)
                    <= (items[1].created_at_ms, &items[1].run_id))
        );
    }

    #[test]
    fn graph_reads_are_namespace_scoped_and_insertion_ordered() {
        block_on(async {
            let store = InMemoryStore::new();
            store
                .upsert_graph_record("alpha", record("z", "red", &[1.0, 0.0]))
                .await
                .unwrap();
            store
                .upsert_graph_record("alpha", record("a", "blue", &[0.0, 1.0]))
                .await
                .unwrap();
            store
                .upsert_graph_record("beta", record("a", "red", &[1.0, 1.0]))
                .await
                .unwrap();
            let all = store
                .graph_records("alpha", &MetadataFilter::default())
                .await
                .unwrap();
            assert_eq!(
                all.iter().map(|item| item.id.as_str()).collect::<Vec<_>>(),
                ["z", "a"]
            );
            assert_eq!(store.graph_record("beta", "z").await.unwrap(), None);
            assert_eq!(
                store
                    .graph_record("beta", "a")
                    .await
                    .unwrap()
                    .unwrap()
                    .metadata["team"],
                Value::String("red".to_owned())
            );
        });
    }

    #[test]
    fn record_upsert_keeps_order_and_delete_reinsert_appends() {
        block_on(async {
            let store = InMemoryStore::new();
            store
                .upsert_graph_record("ns", record("first", "red", &[1.0, 0.0]))
                .await
                .unwrap();
            store
                .upsert_graph_record("ns", record("second", "red", &[0.0, 1.0]))
                .await
                .unwrap();
            store
                .upsert_graph_record("ns", record("first", "blue", &[1.0, 1.0]))
                .await
                .unwrap();
            assert_eq!(
                store
                    .graph_records("ns", &MetadataFilter::default())
                    .await
                    .unwrap()
                    .iter()
                    .map(|record| record.id.as_str())
                    .collect::<Vec<_>>(),
                ["first", "second"]
            );
            assert!(store.delete_graph_record("ns", "first").await.unwrap());
            store
                .upsert_graph_record("ns", record("first", "blue", &[1.0, 1.0]))
                .await
                .unwrap();
            assert_eq!(
                store
                    .graph_records("ns", &MetadataFilter::default())
                    .await
                    .unwrap()
                    .iter()
                    .map(|record| record.id.as_str())
                    .collect::<Vec<_>>(),
                ["second", "first"]
            );
        });
    }

    #[test]
    fn metadata_filter_requires_every_exact_predicate() {
        block_on(async {
            let store = InMemoryStore::new();
            let mut first = record("first", "red", &[1.0, 0.0]);
            first.metadata.insert("active".to_owned(), json!(true));
            store.upsert_graph_record("ns", first).await.unwrap();
            store
                .upsert_graph_record("ns", record("second", "red", &[0.0, 1.0]))
                .await
                .unwrap();
            let filter = MetadataFilter {
                equals: BTreeMap::from([
                    ("team".to_owned(), json!("red")),
                    ("active".to_owned(), json!(true)),
                ]),
            };
            assert_eq!(
                store
                    .graph_records("ns", &filter)
                    .await
                    .unwrap()
                    .iter()
                    .map(|item| item.id.as_str())
                    .collect::<Vec<_>>(),
                ["first"]
            );
        });
    }

    #[test]
    fn event_identity_is_idempotent_and_replay_is_ordered() {
        block_on(async {
            let store = InMemoryStore::new();
            let first = store
                .append_entity_event("ns", event("evt-1", "one"))
                .await
                .unwrap();
            let second = store
                .append_entity_event("ns", event("evt-2", "two"))
                .await
                .unwrap();
            let retry = store
                .append_entity_event("ns", event("evt-1", "changed"))
                .await
                .unwrap();
            let other = store
                .append_entity_event("other", event("evt-1", "one"))
                .await
                .unwrap();
            assert_eq!(
                (first.event.seq, second.event.seq, other.event.seq),
                (1, 2, 1)
            );
            assert!(!retry.inserted);
            assert_eq!(retry.event.entity_id, "one");
            assert_eq!(
                store
                    .replay_events("ns", 0, 10)
                    .await
                    .unwrap()
                    .iter()
                    .map(|item| item.seq)
                    .collect::<Vec<_>>(),
                [1, 2]
            );
            assert_eq!(store.replay_events("ns", 1, 1).await.unwrap()[0].seq, 2);
        });
    }

    #[test]
    fn restore_event_snapshot_preserves_retained_sequences_cursors_and_identity() {
        block_on(async {
            let store = InMemoryStore::new();
            store
                .restore_event_snapshot(
                    vec![
                        EntityEvent {
                            namespace: "ns".to_owned(),
                            seq: 3,
                            event_id: "evt-3".to_owned(),
                            entity_kind: "node".to_owned(),
                            entity_id: "three".to_owned(),
                            op: "UPSERT".to_owned(),
                            payload: json!({"id": "three"}),
                        },
                        EntityEvent {
                            namespace: "ns".to_owned(),
                            seq: 5,
                            event_id: "evt-5".to_owned(),
                            entity_kind: "node".to_owned(),
                            entity_id: "five".to_owned(),
                            op: "UPSERT".to_owned(),
                            payload: json!({"id": "five"}),
                        },
                    ],
                    vec![
                        ReplayCursor {
                            namespace: "ns".to_owned(),
                            consumer: "replay".to_owned(),
                            last_seq: 9,
                        },
                        ReplayCursor {
                            namespace: "empty".to_owned(),
                            consumer: "replay".to_owned(),
                            last_seq: 12,
                        },
                    ],
                )
                .unwrap();
            assert_eq!(store.latest_event_seq("ns").await.unwrap(), 5);
            assert_eq!(
                store
                    .replay_events("ns", 0, 10)
                    .await
                    .unwrap()
                    .iter()
                    .map(|event| event.seq)
                    .collect::<Vec<_>>(),
                [3, 5]
            );
            assert_eq!(
                store.replay_cursor("ns", "replay").await.unwrap().last_seq,
                9
            );
            assert_eq!(store.latest_event_seq("empty").await.unwrap(), 0);
            assert_eq!(
                store
                    .replay_cursor("empty", "replay")
                    .await
                    .unwrap()
                    .last_seq,
                12
            );
            assert_eq!(
                store
                    .append_entity_event("ns", event("evt-6", "six"))
                    .await
                    .unwrap()
                    .event
                    .seq,
                6
            );
            assert!(
                !store
                    .append_entity_event("ns", event("evt-3", "changed"))
                    .await
                    .unwrap()
                    .inserted
            );
            assert_eq!(
                store
                    .advance_replay_cursor("ns", "new", 7)
                    .await
                    .unwrap_err(),
                StoreError::CursorOutOfRange {
                    cursor: 7,
                    latest: 6
                }
            );
        });
    }

    #[test]
    fn restore_event_snapshot_rejects_invalid_event_shapes() {
        let store = InMemoryStore::new();
        let valid = |seq: i64, event_id: &str| EntityEvent {
            namespace: "ns".to_owned(),
            seq,
            event_id: event_id.to_owned(),
            entity_kind: "node".to_owned(),
            entity_id: event_id.to_owned(),
            op: "UPSERT".to_owned(),
            payload: json!({}),
        };
        assert_eq!(
            store
                .restore_event_snapshot(
                    vec![EntityEvent {
                        namespace: String::new(),
                        ..valid(1, "evt")
                    }],
                    vec![]
                )
                .unwrap_err(),
            StoreError::EmptyNamespace
        );
        assert_eq!(
            store
                .restore_event_snapshot(vec![valid(0, "evt")], vec![])
                .unwrap_err(),
            StoreError::InvalidRestoredEventSequence {
                namespace: "ns".to_owned(),
                seq: 0
            }
        );
        assert_eq!(
            store
                .restore_event_snapshot(vec![valid(3, "first"), valid(3, "second")], vec![])
                .unwrap_err(),
            StoreError::InvalidRestoredEventSequence {
                namespace: "ns".to_owned(),
                seq: 3
            }
        );
        assert_eq!(
            store
                .restore_event_snapshot(vec![valid(3, "duplicate"), valid(5, "duplicate")], vec![])
                .unwrap_err(),
            StoreError::DuplicateRestoredEventId {
                namespace: "ns".to_owned(),
                event_id: "duplicate".to_owned()
            }
        );
        assert_eq!(
            store
                .restore_event_snapshot(
                    vec![],
                    vec![ReplayCursor {
                        namespace: String::new(),
                        consumer: "replay".to_owned(),
                        last_seq: 1
                    }]
                )
                .unwrap_err(),
            StoreError::EmptyNamespace
        );
    }

    #[test]
    fn restore_projection_snapshot_is_key_ordered_and_shadow_read_only() {
        block_on(async {
            let store = InMemoryStore::new();
            let projection = |namespace: &str, key: &str, payload: Value| NamedProjection {
                namespace: namespace.to_owned(),
                key: key.to_owned(),
                payload: payload.as_object().unwrap().clone(),
                last_authoritative_seq: 8,
                last_materialized_seq: 7,
                projection_schema_version: 2,
                materialization_status: "ready".to_owned(),
                updated_at_ms: 1234,
            };
            store
                .restore_projection_snapshot(vec![
                    projection("bridge", "z", json!({"nested": {"v": [1, true]}})),
                    projection("bridge", "a", json!({"name": "first"})),
                    projection("other", "a", json!({"name": "isolated"})),
                ])
                .unwrap();
            let shadow = store.shadow_inspection();
            assert_eq!(
                shadow
                    .named_projections("bridge")
                    .await
                    .unwrap()
                    .iter()
                    .map(|projection| projection.key.as_str())
                    .collect::<Vec<_>>(),
                ["a", "z"]
            );
            assert_eq!(
                shadow
                    .named_projection("bridge", "z")
                    .await
                    .unwrap()
                    .unwrap()
                    .payload,
                json!({"nested": {"v": [1, true]}})
                    .as_object()
                    .unwrap()
                    .clone()
            );
            assert_eq!(
                store
                    .named_projection("other", "a")
                    .await
                    .unwrap()
                    .unwrap()
                    .payload["name"],
                Value::String("isolated".to_owned())
            );
        });
    }

    #[test]
    fn restore_projection_snapshot_rejects_duplicate_namespace_key() {
        let store = InMemoryStore::new();
        let projection = |payload: Value| NamedProjection {
            namespace: "bridge".to_owned(),
            key: "same".to_owned(),
            payload: payload.as_object().unwrap().clone(),
            last_authoritative_seq: 1,
            last_materialized_seq: 1,
            projection_schema_version: 1,
            materialization_status: "ready".to_owned(),
            updated_at_ms: 1,
        };
        assert_eq!(
            store
                .restore_projection_snapshot(vec![
                    projection(json!({"v": 1})),
                    projection(json!({"v": 2})),
                ])
                .unwrap_err(),
            StoreError::DuplicateRestoredProjection {
                namespace: "bridge".to_owned(),
                key: "same".to_owned()
            }
        );
    }

    #[test]
    fn named_projection_writes_are_cas_safe_key_ordered_and_namespace_isolated() {
        block_on(async {
            let store = InMemoryStore::new();
            let write = |value: i64| NamedProjectionWrite {
                payload: json!({"z": "雪", "a": value}).as_object().unwrap().clone(),
                last_authoritative_seq: value,
                last_materialized_seq: value,
                projection_schema_version: 1,
                materialization_status: "ready".to_owned(),
            };
            assert!(
                store
                    .compare_and_swap_named_projection("ns", "z", None, None, write(1))
                    .await
                    .unwrap()
            );
            assert!(
                !store
                    .compare_and_swap_named_projection("ns", "z", None, None, write(2))
                    .await
                    .unwrap()
            );
            assert!(
                !store
                    .compare_and_swap_named_projection("ns", "z", Some(0), Some(0), write(2))
                    .await
                    .unwrap()
            );
            assert!(
                store
                    .compare_and_swap_named_projection("ns", "z", Some(1), Some(1), write(2))
                    .await
                    .unwrap()
            );
            store
                .replace_named_projection("ns", "a", write(3))
                .await
                .unwrap();
            store
                .replace_named_projection("other", "a", write(4))
                .await
                .unwrap();
            assert_eq!(
                store
                    .named_projections("ns")
                    .await
                    .unwrap()
                    .iter()
                    .map(|row| row.key.as_str())
                    .collect::<Vec<_>>(),
                ["a", "z"]
            );
            store.clear_projection_namespace("ns").await.unwrap();
            assert!(store.named_projections("ns").await.unwrap().is_empty());
            assert!(
                store
                    .named_projection("other", "a")
                    .await
                    .unwrap()
                    .is_some()
            );
        });
    }

    #[test]
    fn replay_cursor_is_per_consumer_monotonic_and_bounded() {
        block_on(async {
            let store = InMemoryStore::new();
            store
                .append_entity_event("ns", event("evt-1", "one"))
                .await
                .unwrap();
            store
                .append_entity_event("ns", event("evt-2", "two"))
                .await
                .unwrap();
            assert_eq!(store.replay_cursor("ns", "sink").await.unwrap().last_seq, 0);
            assert_eq!(
                store
                    .advance_replay_cursor("ns", "sink", 1)
                    .await
                    .unwrap()
                    .last_seq,
                1
            );
            assert_eq!(
                store.replay_cursor("ns", "other").await.unwrap().last_seq,
                0
            );
            assert_eq!(
                store
                    .advance_replay_cursor("ns", "sink", 0)
                    .await
                    .unwrap_err(),
                StoreError::CursorRegresses {
                    current: 1,
                    requested: 0
                }
            );
            assert_eq!(
                store
                    .advance_replay_cursor("ns", "sink", 3)
                    .await
                    .unwrap_err(),
                StoreError::CursorOutOfRange {
                    cursor: 3,
                    latest: 2
                }
            );
        });
    }

    #[test]
    fn cosine_vector_query_matches_python_memory_candidates_and_ties() {
        block_on(async {
            let store = InMemoryStore::new();
            store
                .upsert_graph_record("ns", record("z", "keep", &[1.0, 0.0]))
                .await
                .unwrap();
            store
                .upsert_graph_record("ns", record("a", "keep", &[0.0, 1.0]))
                .await
                .unwrap();
            let mut missing = GraphRecord::new("missing");
            missing.metadata.insert("team".to_owned(), json!("keep"));
            store.upsert_graph_record("ns", missing).await.unwrap();
            store
                .upsert_graph_record("ns", record("mismatch", "keep", &[1.0, 0.0, 0.0]))
                .await
                .unwrap();
            store
                .upsert_graph_record("ns", record("zero", "keep", &[0.0, 0.0]))
                .await
                .unwrap();
            store
                .upsert_graph_record("ns", record("skip", "skip", &[1.0, 0.0]))
                .await
                .unwrap();
            let query = VectorQuery {
                embedding: vec![1.0, 1.0],
                limit: 5,
                metadata: MetadataFilter {
                    equals: BTreeMap::from([("team".to_owned(), json!("keep"))]),
                },
                metric: DistanceMetric::Cosine,
            };
            let results = store.vector_query("ns", &query).await.unwrap();
            assert_eq!(
                results
                    .iter()
                    .map(|item| item.record.id.as_str())
                    .collect::<Vec<_>>(),
                ["z", "a", "mismatch", "zero"]
            );
            assert_eq!(results[0].distance, results[1].distance);
            assert_eq!(results[2].distance, 2.0);
            assert_eq!(results[3].distance, 2.0);

            let zero_query = VectorQuery {
                embedding: vec![0.0, 0.0],
                limit: 5,
                metadata: MetadataFilter::default(),
                metric: DistanceMetric::Cosine,
            };
            assert!(
                store
                    .vector_query("ns", &zero_query)
                    .await
                    .unwrap()
                    .iter()
                    .all(|match_| match_.distance == 2.0)
            );

            let strict_l2 = VectorQuery {
                embedding: vec![1.0],
                limit: 5,
                metadata: MetadataFilter::default(),
                metric: DistanceMetric::L2,
            };
            assert_eq!(
                store.vector_query("ns", &strict_l2).await.unwrap_err(),
                StoreError::VectorDimensionMismatch {
                    expected: 1,
                    actual: 2
                }
            );
        });
    }

    #[test]
    fn shadow_inspection_is_read_only_and_does_not_mutate_state() {
        block_on(async {
            let store = InMemoryStore::new();
            store
                .upsert_graph_record("ns", record("one", "red", &[1.0, 0.0]))
                .await
                .unwrap();
            store
                .append_entity_event("ns", event("evt-1", "one"))
                .await
                .unwrap();
            store.advance_replay_cursor("ns", "sink", 1).await.unwrap();
            let shadow = store.shadow_inspection();
            assert_eq!(
                shadow
                    .graph_records("ns", &MetadataFilter::default())
                    .await
                    .unwrap()
                    .len(),
                1
            );
            assert_eq!(shadow.replay_events("ns", 0, 10).await.unwrap().len(), 1);
            assert_eq!(
                shadow.replay_cursor("ns", "sink").await.unwrap().last_seq,
                1
            );
            assert_eq!(
                store
                    .graph_records("ns", &MetadataFilter::default())
                    .await
                    .unwrap()
                    .len(),
                1
            );
            assert_eq!(store.latest_event_seq("ns").await.unwrap(), 1);
            assert_eq!(store.replay_cursor("ns", "sink").await.unwrap().last_seq, 1);
        });
    }
}
