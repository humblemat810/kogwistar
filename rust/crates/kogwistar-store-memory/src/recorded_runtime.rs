use super::{
    InMemoryStore, NamedProjection, ServerRun, ServerRunEvent, State, now_ms, now_seconds,
};
use kogwistar_runtime::{
    PersistedRecordedTransition, RECORDED_RUNTIME_CONTRACT_VERSION, RecordedRunStatus,
    RecordedRuntimeState, RecordedRuntimeTransition, RecordedTransitionKind,
    RecordedTransitionResult, RecordedWorkerHandoff, RecordedWorkerSuccessEffect,
    RuntimeWorkerEffectStatus, RuntimeWorkerSuccessEffect,
    authoritative_runtime_successors_for_result, frontier_after_worker_success,
    frontier_after_worker_suspend, reduce_recorded_transition, transition_digest,
    worker_effect_digest,
};
use kogwistar_store::{
    RUNTIME_CURRENT_STATE_NAMESPACE, StoreError, StoreResult, runtime_checkpoint_namespace,
    runtime_projection_write, runtime_status_namespace, timestamp_i64,
};
use serde_json::Value;

const RECORDED_RUNTIME_EVENT_TYPE: &str = "workflow.recorded_transition.v1";

// PYTHON_REFERENCE_V1: rust=rust/crates/kogwistar-store-memory/src/recorded_runtime.rs#apply_recorded_runtime_transition; python=kogwistar/engine_core/in_memory_meta.py#append_server_run_event+update_server_run; rust_test=rust/crates/kogwistar-store-memory/src/recorded_runtime.rs#recorded_transition_is_atomic_replayable_and_projected_in_memory; python_test=tests/core/test_recorded_transition_identity.py#test_metadata_stores_reject_duplicate_recorded_transition_ids; note=Rust adds an atomic reducer transaction; Python remains primitive append/update and rejects duplicate IDs.
impl InMemoryStore {
    /// Atomically records a Rust/Python-compatible runtime transition in the
    /// process-local store. One write lock is the transaction boundary; no
    /// resolver, provider, or graph callback runs while that lock is held.
    pub fn apply_recorded_runtime_transition(
        &self,
        transition: RecordedRuntimeTransition,
        abort_after_writes: bool,
    ) -> StoreResult<RecordedTransitionResult> {
        let mut state = self.state.write().expect("in-memory store lock poisoned");
        apply_transition(&mut state, transition, None, abort_after_writes)
    }

    /// Applies one claimed worker effect and acknowledges its lane in the same
    /// write-lock transaction. Retries require the original handoff and digest.
    pub fn apply_claimed_recorded_worker_effect(
        &self,
        handoff: RecordedWorkerHandoff,
        effect: RecordedWorkerSuccessEffect,
    ) -> StoreResult<RecordedTransitionResult> {
        validate_handoff(&handoff)?;
        if effect.contract_version != RECORDED_RUNTIME_CONTRACT_VERSION
            || effect.effect_id.is_empty()
        {
            return Err(runtime_conflict(
                "worker effect contract_version/effect_id is invalid",
            ));
        }
        if effect.graph_mutation_batch.is_some()
            || effect.graph_mutation_batch_v2.is_some()
            || !effect.provider_event_publications.is_empty()
            || !effect.provider_event_acknowledgements.is_empty()
        {
            return Err(runtime_conflict(
                "in-memory worker result does not support graph or provider-event effects",
            ));
        }
        let effect_digest = worker_effect_digest(&effect).map_err(runtime_error)?;
        let trace_events = effect.trace_events.clone();
        let usage = effect.usage.clone();
        let mut state = self.state.write().expect("in-memory store lock poisoned");
        let lane = state
            .lane_messages
            .get(&handoff.message_id)
            .cloned()
            .ok_or_else(|| runtime_conflict("worker lane message does not exist"))?;
        if lane.run_id.as_deref() != Some(handoff.run_id.as_str())
            || lane.step_id.as_deref() != Some(handoff.step_id.as_str())
            || lane.correlation_id.as_deref() != Some(handoff.correlation_id.as_str())
        {
            return Err(runtime_conflict(
                "worker lane run/step/correlation identity differs",
            ));
        }
        if state
            .server_runs
            .get(&handoff.run_id)
            .is_some_and(|run| run.cancel_requested)
        {
            return Err(runtime_conflict(
                "worker lane belongs to a cancel-requested run",
            ));
        }
        if lane.status == "completed" {
            if let Some((persisted, event_seq)) =
                recorded_transition_by_id(&state, &effect.effect_id)?
                && persisted.worker_handoff.as_ref().is_some_and(|stored| {
                    stored == &handoff || stored.message_id == handoff.message_id
                })
            {
                if persisted.worker_effect_digest.as_deref() != Some(effect_digest.as_str()) {
                    return Err(runtime_conflict(
                        "worker handoff retried with different effect",
                    ));
                }
                if let Some((completed, completed_seq)) =
                    recorded_transition_by_id(&state, &format!("complete-{}", effect.effect_id))?
                    && completed.reduced.state.run_id == handoff.run_id
                {
                    return Ok(completed.result(completed_seq, true));
                }
                return Ok(persisted.result(event_seq, true));
            }
            return Err(runtime_conflict(
                "completed worker lane has no matching recorded effect",
            ));
        }
        if lane.status != "claimed"
            || lane.claimed_by.as_deref() != Some(handoff.claimed_by.as_str())
            || lane
                .lease_until
                .as_ref()
                .is_none_or(|lease_until| timestamp_i64(lease_until) < now_seconds())
        {
            return Err(runtime_conflict(
                "worker lane is not actively claimed by this handoff",
            ));
        }

        let payload: Value = serde_json::from_str(lane.payload_json.as_deref().unwrap_or("{}"))
            .map_err(json_error)?;
        let workflow_id = payload["workflow_id"].as_str().unwrap_or_default();
        let conversation_id = payload["conversation_id"].as_str().unwrap_or_default();
        let token_id = payload["token_id"]
            .as_str()
            .unwrap_or(handoff.run_id.as_str());
        let parent_token_id = payload["parent_token_id"].as_str();
        let current = recorded_state_for(&state, &handoff.run_id, workflow_id, conversation_id)?
            .ok_or_else(|| runtime_conflict("worker lane has no recorded runtime state"))?;
        if current
            .frontier
            .pending
            .first()
            .is_none_or(|(node, _, token, parent)| {
                node != &handoff.step_id
                    || token != token_id
                    || parent.as_deref() != parent_token_id
            })
        {
            return Err(runtime_conflict(
                "worker result is not next in canonical frontier order",
            ));
        }
        let step_seq = current.last_step_seq.saturating_add(1);
        let successors = match effect.status {
            RuntimeWorkerEffectStatus::Success => authoritative_runtime_successors_for_result(
                &current.static_routes,
                &handoff.step_id,
                &effect.successors,
                &effect.route_next,
                false,
            ),
            RuntimeWorkerEffectStatus::Suspended => Ok(Vec::new()),
            RuntimeWorkerEffectStatus::Failed => authoritative_runtime_successors_for_result(
                &current.static_routes,
                &handoff.step_id,
                &effect.successors,
                &effect.route_next,
                true,
            ),
        }
        .map_err(runtime_error)?;
        let terminal_failure =
            effect.status == RuntimeWorkerEffectStatus::Failed && successors.is_empty();
        let frontier = match effect.status {
            RuntimeWorkerEffectStatus::Success => frontier_after_worker_success(
                &current.frontier,
                &handoff.step_id,
                token_id,
                parent_token_id,
                step_seq,
                &RuntimeWorkerSuccessEffect { successors },
            ),
            RuntimeWorkerEffectStatus::Suspended => frontier_after_worker_suspend(
                &current.frontier,
                &handoff.step_id,
                token_id,
                parent_token_id,
            ),
            RuntimeWorkerEffectStatus::Failed if terminal_failure => {
                Ok(kogwistar_runtime::RuntimeFrontier::default())
            }
            RuntimeWorkerEffectStatus::Failed => frontier_after_worker_success(
                &current.frontier,
                &handoff.step_id,
                token_id,
                parent_token_id,
                step_seq,
                &RuntimeWorkerSuccessEffect { successors },
            ),
        }
        .map_err(runtime_error)?;
        let transition = RecordedRuntimeTransition {
            contract_version: RECORDED_RUNTIME_CONTRACT_VERSION,
            transition_id: effect.effect_id.clone(),
            expected_event_seq: state
                .server_run_events
                .get(&handoff.run_id)
                .and_then(|events| events.last())
                .map_or(0, |event| event.seq),
            kind: match effect.status {
                RuntimeWorkerEffectStatus::Success => RecordedTransitionKind::RecordedStepSuccess,
                RuntimeWorkerEffectStatus::Suspended => RecordedTransitionKind::Suspend,
                RuntimeWorkerEffectStatus::Failed if terminal_failure => {
                    RecordedTransitionKind::Fail
                }
                RuntimeWorkerEffectStatus::Failed => RecordedTransitionKind::RecordedStepSuccess,
            },
            run_id: handoff.run_id.clone(),
            workflow_id: workflow_id.to_owned(),
            conversation_id: conversation_id.to_owned(),
            user_id: None,
            user_turn_node_id: None,
            step_seq,
            node_id: Some(handoff.step_id.clone()),
            token_id: Some(token_id.to_owned()),
            parent_token_id: parent_token_id.map(str::to_owned),
            initial_state: None,
            state_update: effect.state_update,
            update: effect.update,
            state_schema: effect.state_schema,
            frontier: Some(frontier),
            result: effect.result.clone(),
            wait_reason: effect.wait_reason,
            resume_payload: effect.resume_payload,
            errors: effect.errors,
        };
        let mut telemetry_events = Vec::new();
        if let Some(usage) = usage {
            telemetry_events.push((
                "workflow.usage.v1",
                serde_json::to_string(&serde_json::json!({
                    "run_id": handoff.run_id,
                    "step_id": handoff.step_id,
                    "token_id": token_id,
                    "effect_id": effect.effect_id,
                    "usage": usage,
                }))
                .map_err(json_error)?,
            ));
        }
        for (trace_index, trace) in trace_events.into_iter().enumerate() {
            telemetry_events.push((
                "workflow.trace.v1",
                serde_json::to_string(&serde_json::json!({
                    "run_id": handoff.run_id,
                    "step_id": handoff.step_id,
                    "token_id": token_id,
                    "effect_id": effect.effect_id,
                    "trace_index": trace_index,
                    "trace": trace,
                }))
                .map_err(json_error)?,
            ));
        }
        let applied = apply_transition(
            &mut state,
            transition,
            Some((handoff.clone(), effect_digest)),
            false,
        )?;
        let current_frontier = &applied.reduced.state.frontier;
        let unfinished = !current_frontier.pending.is_empty()
            || !current_frontier.suspended.is_empty()
            || current_frontier
                .join_outstanding
                .iter()
                .any(|value| *value != 0)
            || current_frontier
                .join_waiters
                .values()
                .any(|waiters| !waiters.is_empty());
        let result = if unfinished || applied.reduced.state.status.is_terminal() {
            applied
        } else {
            let completed_state = &applied.reduced.state;
            let expected_event_seq = state
                .server_run_events
                .get(&handoff.run_id)
                .and_then(|events| events.last())
                .map_or(0, |event| event.seq);
            apply_transition(
                &mut state,
                RecordedRuntimeTransition {
                    contract_version: RECORDED_RUNTIME_CONTRACT_VERSION,
                    transition_id: format!("complete-{}", effect.effect_id),
                    expected_event_seq,
                    kind: RecordedTransitionKind::Complete,
                    run_id: handoff.run_id.clone(),
                    workflow_id: completed_state.workflow_id.clone(),
                    conversation_id: completed_state.conversation_id.clone(),
                    user_id: None,
                    user_turn_node_id: None,
                    step_seq: completed_state.last_step_seq,
                    node_id: completed_state.last_node_id.clone(),
                    token_id: completed_state.last_token_id.clone(),
                    parent_token_id: completed_state.last_parent_token_id.clone(),
                    initial_state: None,
                    state_update: Vec::new(),
                    update: None,
                    state_schema: Default::default(),
                    frontier: Some(kogwistar_runtime::RuntimeFrontier::default()),
                    result: effect.result.clone(),
                    wait_reason: None,
                    resume_payload: None,
                    errors: Vec::new(),
                },
                None,
                false,
            )?
        };
        let mut next_seq = result.event_seq;
        for (event_type, payload_json) in telemetry_events {
            next_seq += 1;
            state
                .server_run_events
                .entry(handoff.run_id.clone())
                .or_default()
                .push(ServerRunEvent {
                    seq: next_seq,
                    run_id: handoff.run_id.clone(),
                    event_type: event_type.to_owned(),
                    payload_json,
                    created_at_ms: now_ms(),
                });
        }
        let lane = state
            .lane_messages
            .get_mut(&handoff.message_id)
            .expect("validated worker lane remains present under write lock");
        lane.status = "completed".to_owned();
        lane.claimed_by = None;
        lane.lease_until = None;
        Ok(result)
    }
}

fn apply_transition(
    state: &mut State,
    transition: RecordedRuntimeTransition,
    worker: Option<(RecordedWorkerHandoff, String)>,
    abort_after_writes: bool,
) -> StoreResult<RecordedTransitionResult> {
    let request_digest = transition_digest(&transition).map_err(runtime_error)?;

    if let Some((persisted, event_seq)) =
        recorded_transition_by_id(state, &transition.transition_id)?
    {
        if persisted.request_digest != request_digest
            || match &worker {
                Some((handoff, digest)) => {
                    persisted.worker_handoff.as_ref() != Some(handoff)
                        || persisted.worker_effect_digest.as_deref() != Some(digest.as_str())
                }
                None => persisted.worker_handoff.is_some(),
            }
        {
            return Err(runtime_conflict(format!(
                "transition_id {:?} reused with different payload",
                transition.transition_id
            )));
        }
        return Ok(persisted.result(event_seq, true));
    }

    let prior_run = state.server_runs.get(&transition.run_id).cloned();
    if let Some(run) = &prior_run
        && (run.workflow_id != transition.workflow_id
            || run.conversation_id != transition.conversation_id)
    {
        return Err(runtime_conflict(format!(
            "run identity differs for {:?}",
            transition.run_id
        )));
    }

    let current = recorded_state(state, &transition)?;
    let current_event_seq = state
        .server_run_events
        .get(&transition.run_id)
        .and_then(|events| events.last())
        .map_or(0, |event| event.seq);
    if current_event_seq != transition.expected_event_seq {
        return Err(runtime_conflict(format!(
            "expected_event_seq {}, current {}",
            transition.expected_event_seq, current_event_seq
        )));
    }

    let reduced =
        reduce_recorded_transition(current.as_ref(), &transition).map_err(runtime_error)?;
    let new_user_turn_node_id = if current.is_none() && prior_run.is_none() {
        Some(
            transition
                .user_turn_node_id
                .clone()
                .ok_or_else(|| runtime_conflict("start transition requires user_turn_node_id"))?,
        )
    } else {
        None
    };
    let event_seq = current_event_seq + 1;
    let persisted = PersistedRecordedTransition {
        contract_version: RECORDED_RUNTIME_CONTRACT_VERSION,
        transition_id: transition.transition_id.clone(),
        request_digest,
        worker_handoff: worker.as_ref().map(|(handoff, _)| handoff.clone()),
        worker_effect_digest: worker.map(|(_, digest)| digest),
        reduced: reduced.clone(),
    };
    let payload_json = serde_json::to_string(&persisted).map_err(json_error)?;
    let result_json = reduced
        .result
        .as_ref()
        .map(serde_json::to_string)
        .transpose()
        .map_err(json_error)?;
    let error_json = if reduced.errors.is_empty() {
        prior_run.as_ref().and_then(|run| run.error_json.clone())
    } else {
        Some(serde_json::to_string(&reduced.errors).map_err(json_error)?)
    };
    let state_payload = serde_json::to_value(&reduced.state)
        .map_err(json_error)?
        .as_object()
        .cloned()
        .ok_or_else(|| runtime_conflict("runtime state projection is not an object"))?;
    let server_status = reduced.server_status.clone();
    let status_materialization = match reduced.state.status {
        RecordedRunStatus::Completed => "completed",
        RecordedRunStatus::Failed => "failed",
        RecordedRunStatus::Cancelled => "cancelled",
        RecordedRunStatus::Suspended => "suspended",
        RecordedRunStatus::Running => "running",
    };
    let assistant_turn_node_id = (reduced.state.status == RecordedRunStatus::Completed)
        .then(|| {
            reduced
                .state
                .state
                .get("agentic_answering_result")
                .and_then(|result| result.get("assistant_turn_node_id"))
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .flatten()
        .or_else(|| {
            prior_run
                .as_ref()
                .and_then(|run| run.assistant_turn_node_id.clone())
        });
    let now = now_ms();

    if abort_after_writes {
        return Err(runtime_conflict(
            "in-memory runtime transition aborted before atomic commit",
        ));
    }

    if let Some(user_turn_node_id) = new_user_turn_node_id {
        state.server_runs.insert(
            transition.run_id.clone(),
            ServerRun {
                run_id: transition.run_id.clone(),
                conversation_id: transition.conversation_id.clone(),
                workflow_id: transition.workflow_id.clone(),
                user_id: transition.user_id.clone(),
                user_turn_node_id: Some(user_turn_node_id),
                assistant_turn_node_id: None,
                status: server_status.clone(),
                cancel_requested: false,
                result_json: None,
                error_json: None,
                created_at_ms: now,
                updated_at_ms: now,
                started_at_ms: None,
                finished_at_ms: None,
            },
        );
    }

    state
        .server_run_events
        .entry(transition.run_id.clone())
        .or_default()
        .push(ServerRunEvent {
            seq: event_seq,
            run_id: transition.run_id.clone(),
            event_type: RECORDED_RUNTIME_EVENT_TYPE.to_owned(),
            payload_json,
            created_at_ms: now,
        });

    put_projection(
        state,
        RUNTIME_CURRENT_STATE_NAMESPACE,
        &transition.run_id,
        runtime_projection_write(state_payload, event_seq, "ready"),
        now,
    );
    put_projection(
        state,
        &runtime_checkpoint_namespace(&transition.conversation_id),
        &transition.run_id,
        runtime_projection_write(reduced.checkpoint.clone(), event_seq, server_status.clone()),
        now,
    );
    put_projection(
        state,
        &runtime_status_namespace(&transition.conversation_id),
        &transition.run_id,
        runtime_projection_write(
            reduced.run_status.clone(),
            event_seq,
            status_materialization,
        ),
        now,
    );

    let run = state
        .server_runs
        .entry(transition.run_id.clone())
        .or_insert_with(|| {
            prior_run
                .clone()
                .expect("existing run checked before reduction")
        });
    run.status = server_status;
    run.assistant_turn_node_id = assistant_turn_node_id;
    run.result_json =
        result_json.or_else(|| prior_run.as_ref().and_then(|row| row.result_json.clone()));
    run.error_json = error_json;
    run.started_at_ms = prior_run
        .as_ref()
        .and_then(|row| row.started_at_ms)
        .or_else(|| (transition.kind == RecordedTransitionKind::Start).then_some(now));
    run.finished_at_ms = if reduced.state.status.is_terminal() {
        Some(now)
    } else {
        prior_run.as_ref().and_then(|row| row.finished_at_ms)
    };
    run.cancel_requested = reduced.state.status == RecordedRunStatus::Cancelled;
    run.updated_at_ms = now;

    Ok(persisted.result(event_seq, false))
}

fn recorded_transition_by_id(
    state: &State,
    transition_id: &str,
) -> StoreResult<Option<(PersistedRecordedTransition, i64)>> {
    for events in state.server_run_events.values() {
        for event in events {
            if event.event_type != RECORDED_RUNTIME_EVENT_TYPE {
                continue;
            }
            let Ok(persisted) =
                serde_json::from_str::<PersistedRecordedTransition>(&event.payload_json)
            else {
                continue;
            };
            if persisted.transition_id == transition_id {
                return Ok(Some((persisted, event.seq)));
            }
        }
    }
    Ok(None)
}

fn recorded_state(
    state: &State,
    transition: &RecordedRuntimeTransition,
) -> StoreResult<Option<RecordedRuntimeState>> {
    recorded_state_for(
        state,
        &transition.run_id,
        &transition.workflow_id,
        &transition.conversation_id,
    )
}

fn recorded_state_for(
    state: &State,
    run_id: &str,
    workflow_id: &str,
    conversation_id: &str,
) -> StoreResult<Option<RecordedRuntimeState>> {
    let mut latest = None;
    if let Some(events) = state.server_run_events.get(run_id) {
        for event in events {
            if event.event_type != RECORDED_RUNTIME_EVENT_TYPE {
                continue;
            }
            let Ok(persisted) =
                serde_json::from_str::<PersistedRecordedTransition>(&event.payload_json)
            else {
                continue;
            };
            if persisted.contract_version != RECORDED_RUNTIME_CONTRACT_VERSION {
                continue;
            }
            let runtime = persisted.reduced.state;
            if runtime.run_id == run_id
                && runtime.workflow_id == workflow_id
                && runtime.conversation_id == conversation_id
            {
                latest = Some(runtime);
            }
        }
    }
    Ok(latest)
}

fn validate_handoff(handoff: &RecordedWorkerHandoff) -> StoreResult<()> {
    for (field, value) in [
        ("message_id", handoff.message_id.as_str()),
        ("claimed_by", handoff.claimed_by.as_str()),
        ("run_id", handoff.run_id.as_str()),
        ("step_id", handoff.step_id.as_str()),
        ("correlation_id", handoff.correlation_id.as_str()),
    ] {
        if value.is_empty() {
            return Err(runtime_conflict(format!(
                "worker handoff {field} must not be empty"
            )));
        }
    }
    Ok(())
}

fn put_projection(
    state: &mut State,
    namespace: &str,
    key: &str,
    projection: kogwistar_store::NamedProjectionWrite,
    updated_at_ms: i64,
) {
    InMemoryStore::scope_mut(state, namespace)
        .projections
        .insert(
            key.to_owned(),
            NamedProjection {
                namespace: namespace.to_owned(),
                key: key.to_owned(),
                payload: projection.payload,
                last_authoritative_seq: projection.last_authoritative_seq,
                last_materialized_seq: projection.last_materialized_seq,
                projection_schema_version: projection.projection_schema_version,
                materialization_status: projection.materialization_status,
                updated_at_ms,
            },
        );
}

fn runtime_error(error: impl std::fmt::Display) -> StoreError {
    runtime_conflict(error.to_string())
}

fn json_error(error: serde_json::Error) -> StoreError {
    runtime_conflict(format!(
        "cannot encode recorded runtime transition: {error}"
    ))
}

fn runtime_conflict(message: impl Into<String>) -> StoreError {
    StoreError::Backend {
        backend: "in-memory runtime".to_owned(),
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kogwistar_store::{
        LaneMessageClaimFilter, LaneMessageReadStore, LaneMessageWriteStore,
        NewProjectedLaneMessage, ProjectionReadStore, ServerRunReadStore,
    };
    use serde_json::json;
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

    fn start(transition_id: &str) -> RecordedRuntimeTransition {
        serde_json::from_value(json!({
            "contract_version": 1,
            "transition_id": transition_id,
            "expected_event_seq": 0,
            "kind": "start",
            "run_id": "memory-run",
            "workflow_id": "memory-workflow",
            "conversation_id": "memory-conversation",
            "user_id": "user-1",
            "user_turn_node_id": "turn-1",
            "step_seq": 0,
            "initial_state": {"seed": 1},
            "frontier": {"pending": [], "suspended": [], "join_node_ids": [], "join_outstanding": [], "join_waiters": {}}
        }))
        .unwrap()
    }

    fn pending_start() -> RecordedRuntimeTransition {
        let mut transition = start("start-worker-run");
        transition.frontier = Some(
            serde_json::from_value(json!({
                "pending": [["noop", 0, "token-1", null]],
                "suspended": [],
                "join_node_ids": [],
                "join_outstanding": [],
                "join_waiters": {}
            }))
            .unwrap(),
        );
        transition
    }

    fn noop_effect() -> RecordedWorkerSuccessEffect {
        serde_json::from_value(json!({
            "contract_version": 1,
            "effect_id": "native-noop-effect",
            "status": "success",
            "state_update": [],
            "successors": [],
            "route_next": [],
            "result": {"workflow_status": "succeeded", "final_state": {"seed": 1}},
            "trace_events": [{"event": "runtime.native.step.completed", "operation": "noop"}]
        }))
        .unwrap()
    }

    fn worker_lane() -> NewProjectedLaneMessage {
        NewProjectedLaneMessage {
            message_id: "lane-noop-1".to_owned(),
            namespace: "workflow".to_owned(),
            purpose: "system".to_owned(),
            inbox_id: "workflow-runtime".to_owned(),
            conversation_id: "memory-conversation".to_owned(),
            recipient_id: "rust-native-worker".to_owned(),
            sender_id: "rust-scheduler".to_owned(),
            msg_type: "workflow.step.execute".to_owned(),
            status: "pending".to_owned(),
            created_at: now_seconds(),
            available_at: now_seconds(),
            run_id: Some("memory-run".to_owned()),
            step_id: Some("noop".to_owned()),
            correlation_id: Some("memory-run".to_owned()),
            payload_json: Some(
                json!({
                    "workflow_id": "memory-workflow",
                    "conversation_id": "memory-conversation",
                    "token_id": "token-1"
                })
                .to_string(),
            ),
            error_json: None,
        }
    }

    fn handoff(claimed_by: &str) -> RecordedWorkerHandoff {
        RecordedWorkerHandoff {
            message_id: "lane-noop-1".to_owned(),
            claimed_by: claimed_by.to_owned(),
            run_id: "memory-run".to_owned(),
            step_id: "noop".to_owned(),
            correlation_id: "memory-run".to_owned(),
        }
    }

    #[test]
    fn recorded_transition_is_atomic_replayable_and_projected_in_memory() {
        let store = InMemoryStore::new();
        let transition = start("start-1");
        let aborted = store
            .apply_recorded_runtime_transition(transition.clone(), true)
            .expect_err("injected abort must not commit staged writes");
        assert!(aborted.to_string().contains("aborted before atomic commit"));
        assert!(block_on(store.server_run("memory-run")).unwrap().is_none());
        assert!(
            block_on(store.server_run_events("memory-run", 0, 10))
                .unwrap()
                .is_empty()
        );
        assert!(
            block_on(store.named_projection("workflow_runtime_current_state", "memory-run"))
                .unwrap()
                .is_none()
        );

        let first = store
            .apply_recorded_runtime_transition(transition.clone(), false)
            .unwrap();
        assert_eq!(first.event_seq, 1);
        assert!(!first.idempotent);
        let replay = store
            .apply_recorded_runtime_transition(transition, false)
            .unwrap();
        assert!(replay.idempotent);
        assert_eq!(replay.event_seq, first.event_seq);
        assert_eq!(replay.reduced.state_digest, first.reduced.state_digest);

        let stale = store
            .apply_recorded_runtime_transition(start("stale-start"), false)
            .expect_err("stale event sequence must fail closed");
        assert!(
            stale
                .to_string()
                .contains("expected_event_seq 0, current 1")
        );

        let run = block_on(store.server_run("memory-run"))
            .unwrap()
            .expect("accepted start creates run registry row");
        assert_eq!(run.status, "running");
        let events = block_on(store.server_run_events("memory-run", 0, 10)).unwrap();
        assert_eq!(
            events.len(),
            1,
            "exact replay must not append another event"
        );
        let current =
            block_on(store.named_projection(RUNTIME_CURRENT_STATE_NAMESPACE, "memory-run"))
                .unwrap()
                .expect("accepted transition materializes current state");
        assert_eq!(current.payload["state"]["seed"], 1);
        assert_eq!(current.last_authoritative_seq, 1);
    }

    #[test]
    fn claimed_noop_effect_and_lane_ack_are_atomic_and_idempotent() {
        let store = InMemoryStore::new();
        store
            .apply_recorded_runtime_transition(pending_start(), false)
            .unwrap();
        block_on(store.project_lane_message(worker_lane())).unwrap();
        let claimed = block_on(store.claim_projected_lane_messages_filtered(
            LaneMessageClaimFilter {
                namespace: "workflow".to_owned(),
                inbox_id: "workflow-runtime".to_owned(),
                claimed_by: "rust-native-worker#claim-1".to_owned(),
                message_ids: Some(vec!["lane-noop-1".to_owned()]),
                run_id: Some("memory-run".to_owned()),
                msg_type: Some("workflow.step.execute".to_owned()),
                recipient_id: Some("rust-native-worker".to_owned()),
                limit: 1,
                lease_seconds: 60,
            },
        ))
        .unwrap();
        assert_eq!(claimed.len(), 1);

        let denied = store
            .apply_claimed_recorded_worker_effect(
                handoff("rust-native-worker#stale"),
                noop_effect(),
            )
            .expect_err("a different claimant must not commit an effect");
        assert!(denied.to_string().contains("not actively claimed"));
        assert_eq!(
            block_on(store.projected_lane_message("lane-noop-1"))
                .unwrap()
                .unwrap()
                .status,
            "claimed"
        );

        let effect = noop_effect();
        let applied = store
            .apply_claimed_recorded_worker_effect(
                handoff("rust-native-worker#claim-1"),
                effect.clone(),
            )
            .unwrap();
        assert!(!applied.idempotent);
        assert_eq!(applied.reduced.state.status, RecordedRunStatus::Completed);
        assert_eq!(applied.reduced.state.state["seed"], 1);
        let lane = block_on(store.projected_lane_message("lane-noop-1"))
            .unwrap()
            .unwrap();
        assert_eq!(lane.status, "completed");
        assert!(lane.claimed_by.is_none());

        let replay = store
            .apply_claimed_recorded_worker_effect(handoff("rust-native-worker#claim-1"), effect)
            .unwrap();
        assert!(replay.idempotent);
        assert_eq!(replay.event_seq, applied.event_seq);
        let events = block_on(store.server_run_events("memory-run", 0, 10)).unwrap();
        assert_eq!(events.len(), 4, "start, worker step, completion, and trace");
        assert_eq!(events[3].event_type, "workflow.trace.v1");
        assert_eq!(
            block_on(store.server_run("memory-run"))
                .unwrap()
                .unwrap()
                .status,
            "succeeded"
        );

        let mut changed = noop_effect();
        changed.result = Some(json!({"different": true}));
        assert!(
            store
                .apply_claimed_recorded_worker_effect(
                    handoff("rust-native-worker#claim-1"),
                    changed
                )
                .is_err()
        );
        assert_eq!(
            block_on(store.server_run_events("memory-run", 0, 10))
                .unwrap()
                .len(),
            4
        );
    }
}
