use super::{InMemoryStore, NamedProjection, ServerRun, ServerRunEvent, State, now_ms};
use kogwistar_runtime::{
    PersistedRecordedTransition, RECORDED_RUNTIME_CONTRACT_VERSION, RecordedRunStatus,
    RecordedRuntimeState, RecordedRuntimeTransition, RecordedTransitionKind,
    RecordedTransitionResult, reduce_recorded_transition, transition_digest,
};
use kogwistar_store::{
    RUNTIME_CURRENT_STATE_NAMESPACE, StoreError, StoreResult, runtime_checkpoint_namespace,
    runtime_projection_write, runtime_status_namespace,
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
        apply_transition(&mut state, transition, abort_after_writes)
    }
}

fn apply_transition(
    state: &mut State,
    transition: RecordedRuntimeTransition,
    abort_after_writes: bool,
) -> StoreResult<RecordedTransitionResult> {
    let request_digest = transition_digest(&transition).map_err(runtime_error)?;

    if let Some((persisted, event_seq)) =
        recorded_transition_by_id(state, &transition.transition_id)?
    {
        if persisted.request_digest != request_digest || persisted.worker_handoff.is_some() {
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
        worker_handoff: None,
        worker_effect_digest: None,
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
    let mut latest = None;
    if let Some(events) = state.server_run_events.get(&transition.run_id) {
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
            if runtime.run_id == transition.run_id
                && runtime.workflow_id == transition.workflow_id
                && runtime.conversation_id == transition.conversation_id
            {
                latest = Some(runtime);
            }
        }
    }
    Ok(latest)
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
    use kogwistar_store::{ProjectionReadStore, ServerRunReadStore};
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
}
