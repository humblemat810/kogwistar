//! Deterministic in-memory Phase 2 storage.
use kogwistar_contracts::EntityEventEnvelope;
use kogwistar_store::{
    AppendedEvent, AppliedGraphMutation, DistanceMetric, EntityEvent, EventPruneStore,
    EventReadStore, EventWriteStore, GraphMutation, GraphMutationStore, GraphProjectionRead,
    GraphProjectionVectorQuery, GraphReadStore, GraphRecord, GraphScope, GraphWriteStore, IndexJob,
    IndexJobReadStore, IndexJobWriteStore, LaneMessageFilter, LaneMessageReadStore,
    LaneMessageWriteStore, MetadataFilter, NamedProjection, NamedProjectionWrite, NewEntityEvent,
    NewIndexJob, NewProjectedLaneMessage, ProjectedLaneMessage, ProjectionReadStore,
    ProjectionWriteStore, ReplayCursor, ServerRun, ServerRunCreate, ServerRunEvent,
    ServerRunReadStore, ServerRunUpdate, ServerRunWriteStore, StoreError, StoreResult, VectorMatch,
    VectorQuery, WorkflowDesignDelta, WorkflowDesignDeltaWrite, WorkflowDesignHistoryReadStore,
    WorkflowDesignHistoryWriteStore, WorkflowDesignSnapshot, WorkflowDesignSnapshotWrite,
    integer_timestamp, timestamp_i64,
};
use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};

#[derive(Clone, Debug, Default)]
pub struct InMemoryStore {
    state: Arc<RwLock<State>>,
}
#[derive(Debug, Default)]
struct State {
    namespaces: BTreeMap<String, NamespaceState>,
    graph_projections: BTreeMap<GraphScope, GraphProjectionState>,
    workflow_design_snapshots: BTreeMap<(String, i64), WorkflowDesignSnapshot>,
    workflow_design_deltas: BTreeMap<(String, i64), WorkflowDesignDelta>,
    server_runs: BTreeMap<String, ServerRun>,
    server_run_events: BTreeMap<String, Vec<ServerRunEvent>>,
    index_jobs: BTreeMap<String, IndexJob>,
    lane_messages: BTreeMap<String, ProjectedLaneMessage>,
    next_claim_token: i64,
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
            && row.claimed_by.as_deref().is_none_or(|v| v == claimed_by)
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
            && row.claimed_by.as_deref().is_none_or(|v| v == claimed_by)
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
            && row.claimed_by.as_deref().is_none_or(|v| v == claimed_by)
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
        let now = now_seconds();
        let key = job.coalesce_key();
        let mut state = self.state.write().expect("in-memory store lock poisoned");
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
            return Ok(existing.job_id.clone());
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
            },
        );
        Ok(job_id)
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
            || claim_token.is_some_and(|token| row.claim_token.as_deref() != Some(token))
        {
            return Ok(false);
        }
        row.status = "DONE".to_owned();
        row.lease_until = None;
        row.claim_token = None;
        row.updated_at = integer_timestamp(now);
        Ok(true)
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
            && claim_token.is_none_or(|token| row.claim_token.as_deref() == Some(token))
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
            && claim_token.is_none_or(|token| row.claim_token.as_deref() == Some(token))
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
            && claim_token.is_none_or(|token| row.claim_token.as_deref() == Some(token))
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
    records: BTreeMap<String, GraphRecord>,
    record_order: Vec<String>,
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
    records: BTreeMap<String, GraphRecord>,
    record_order: Vec<String>,
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
        if !projection.records.contains_key(&record.id) {
            projection.record_order.push(record.id.clone());
        }
        projection.records.insert(record.id.clone(), record);
        Ok(AppliedGraphMutation {
            event,
            inserted: true,
            mutated: true,
        })
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

impl GraphReadStore for InMemoryStore {
    async fn graph_record(&self, ns: &str, id: &str) -> StoreResult<Option<GraphRecord>> {
        namespace(ns)?;
        let state = self.state.read().expect("in-memory store lock poisoned");
        Ok(Self::scope(&state, ns)
            .and_then(|scope| scope.records.get(id))
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
                    .filter_map(|id| scope.records.get(id))
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
                        .filter_map(|id| scope.records.get(id))
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
                        .filter_map(|id| scope.records.get(id))
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
        if !scope.records.contains_key(&record.id) {
            scope.record_order.push(record.id.clone());
        }
        scope.records.insert(record.id.clone(), record);
        Ok(())
    }
    async fn delete_graph_record(&self, ns: &str, id: &str) -> StoreResult<bool> {
        namespace(ns)?;
        let mut state = self.state.write().expect("in-memory store lock poisoned");
        let scope = Self::scope_mut(&mut state, ns);
        let deleted = scope.records.remove(id).is_some();
        if deleted {
            scope.record_order.retain(|record_id| record_id != id);
        }
        Ok(deleted)
    }
}

impl GraphMutationStore for InMemoryStore {
    async fn apply_graph_mutation(
        &self,
        mutation: GraphMutation,
    ) -> StoreResult<AppliedGraphMutation> {
        InMemoryStore::apply_graph_mutation(self, mutation).await
    }

    async fn graph_projection_records(
        &self,
        read: GraphProjectionRead,
    ) -> StoreResult<Vec<GraphRecord>> {
        namespace(&read.scope.namespace)?;
        let state = self.state.read().expect("in-memory store lock poisoned");
        let mut rows = state
            .graph_projections
            .get(&read.scope)
            .map(|projection| {
                projection
                    .record_order
                    .iter()
                    .filter_map(|id| projection.records.get(id))
                    .filter(|record| read.metadata.matches(&record.metadata))
                    .cloned()
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        if let Some(ids) = read.ids {
            rows.retain(|record| ids.contains(&record.id));
        }
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
        let mut results = match query.query.metric {
            DistanceMetric::Cosine => projection
                .into_iter()
                .flat_map(|projection| {
                    projection
                        .record_order
                        .iter()
                        .filter_map(|id| projection.records.get(id))
                })
                .filter(|record| query.query.metadata.matches(&record.metadata))
                .filter(|record| record.embedding.is_some())
                .map(|record| VectorMatch {
                    record: record.clone(),
                    distance: cosine_memory_distance(
                        &query.query.embedding,
                        record.embedding.as_deref(),
                    ),
                })
                .collect(),
            DistanceMetric::L2 | DistanceMetric::InnerProduct => projection
                .into_iter()
                .flat_map(|projection| {
                    projection
                        .record_order
                        .iter()
                        .filter_map(|id| projection.records.get(id))
                })
                .filter(|record| query.query.metadata.matches(&record.metadata))
                .filter_map(|record| {
                    record.embedding.as_deref().map(|vector| {
                        distance(query.query.metric, &query.query.embedding, vector).map(
                            |distance| VectorMatch {
                                record: record.clone(),
                                distance,
                            },
                        )
                    })
                })
                .collect::<StoreResult<Vec<_>>>()?,
        };
        results.sort_by(|left, right| left.distance.total_cmp(&right.distance));
        results.truncate(query.query.limit);
        Ok(results)
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
        GraphMutation, GraphMutationStore, GraphProjectionRead, GraphProjectionVectorQuery,
        GraphScope, ProjectionWriteStore, ServerRunReadStore, ServerRunWriteStore,
        ShadowInspectable,
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
