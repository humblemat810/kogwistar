use super::*;
use kogwistar_store::{EventReadStore, EventWriteStore};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::Notify;

static NEXT_SCHEMA: AtomicUsize = AtomicUsize::new(0);

fn test_schema(label: &str) -> String {
    format!(
        "kogwistar_phase3_{}_{}_{}",
        label,
        std::process::id(),
        NEXT_SCHEMA.fetch_add(1, Ordering::Relaxed)
    )
}

fn raw(id: &str, payload_json: &str) -> NewRawEntityEvent {
    NewRawEntityEvent {
        event_id: id.to_owned(),
        entity_kind: "node".to_owned(),
        entity_id: "n".to_owned(),
        op: "UPSERT".to_owned(),
        payload_json: payload_json.to_owned(),
    }
}

fn runtime_start(run_id: &str, transition_id: &str) -> RecordedRuntimeTransition {
    serde_json::from_value(serde_json::json!({
        "contract_version": 1,
        "transition_id": transition_id,
        "expected_event_seq": 0,
        "kind": "start",
        "run_id": run_id,
        "workflow_id": "wf-1",
        "conversation_id": "conv-1",
        "user_id": "user-1",
        "user_turn_node_id": "turn-1",
        "step_seq": 0,
        "node_id": "step-1",
        "token_id": "token-1",
        "parent_token_id": null,
        "initial_state": {"answer": "seed"},
        "frontier": {
            "pending": [["step-1", 0, "token-1", null]],
            "suspended": [],
            "join_node_ids": [],
            "join_outstanding": [],
            "join_waiters": {}
        }
    }))
    .unwrap()
}

fn runtime_result(
    run_id: &str,
    transition_id: &str,
    expected_event_seq: i64,
    answer: &str,
) -> RecordedRuntimeTransition {
    serde_json::from_value(serde_json::json!({
        "contract_version": 1,
        "transition_id": transition_id,
        "expected_event_seq": expected_event_seq,
        "kind": "recorded_step_success",
        "run_id": run_id,
        "workflow_id": "wf-1",
        "conversation_id": "conv-1",
        "step_seq": 0,
        "node_id": "step-1",
        "token_id": "token-1",
        "parent_token_id": null,
        "state_update": [["u", {"answer": answer}]],
        "frontier": {
            "pending": [],
            "suspended": [],
            "join_node_ids": [],
            "join_outstanding": [],
            "join_waiters": {}
        },
        "result": {"answer": answer}
    }))
    .unwrap()
}

fn runtime_failure_start(run_id: &str, transition_id: &str) -> RecordedRuntimeTransition {
    let mut start = runtime_start(run_id, transition_id);
    start.initial_state.as_mut().unwrap().insert(
        "_rt_routes".to_owned(),
        serde_json::json!([{
            "edge_id": "recover-edge",
            "source_node_id": "step-1",
            "target_node_id": "recover",
            "aliases": ["recover"],
            "join_mask": 0,
            "predicate": "on_failure",
            "multiplicity": "one",
            "is_default": false,
            "priority": 100,
            "source_fanout": false
        }]),
    );
    start
}

fn runtime_lane(run_id: &str, message_id: &str) -> NewProjectedLaneMessage {
    NewProjectedLaneMessage {
        message_id: message_id.to_owned(),
        namespace: "runtime".to_owned(),
        purpose: "user_visible".to_owned(),
        inbox_id: "python-workers".to_owned(),
        conversation_id: "conv-1".to_owned(),
        recipient_id: "python-worker".to_owned(),
        sender_id: "rust-runtime".to_owned(),
        msg_type: "workflow.worker.request.v1".to_owned(),
        status: "pending".to_owned(),
        created_at: 1,
        available_at: 0,
        run_id: Some(run_id.to_owned()),
        step_id: Some("step-1".to_owned()),
        correlation_id: Some("corr-1".to_owned()),
        payload_json: Some("{}".to_owned()),
        error_json: None,
    }
}

fn runtime_handoff(run_id: &str, message_id: &str, owner: &str) -> RecordedWorkerHandoff {
    RecordedWorkerHandoff {
        message_id: message_id.to_owned(),
        claimed_by: owner.to_owned(),
        run_id: run_id.to_owned(),
        step_id: "step-1".to_owned(),
        correlation_id: "corr-1".to_owned(),
    }
}

fn runtime_worker_lane(run_id: &str, message_id: &str) -> NewProjectedLaneMessage {
    let mut lane = runtime_lane(run_id, message_id);
    lane.correlation_id = Some(run_id.to_owned());
    lane.payload_json = Some(
        serde_json::json!({
            "workflow_id": "wf-1",
            "conversation_id": "conv-1",
            "token_id": "token-1",
            "parent_token_id": null
        })
        .to_string(),
    );
    lane
}

fn runtime_worker_handoff(run_id: &str, message_id: &str, owner: &str) -> RecordedWorkerHandoff {
    let mut handoff = runtime_handoff(run_id, message_id, owner);
    handoff.correlation_id = run_id.to_owned();
    handoff
}

fn runtime_failure_effect(
    effect_id: &str,
    route_next: &[&str],
    successors: Value,
) -> RecordedWorkerSuccessEffect {
    serde_json::from_value(serde_json::json!({
        "contract_version": 1,
        "effect_id": effect_id,
        "status": "failed",
        "state_update": [["u", {"attempted": true}]],
        "successors": successors,
        "route_next": route_next,
        "result": {"workflow_status": "failed"},
        "errors": ["boom"]
    }))
    .unwrap()
}

#[path = "tests/auth.rs"]
mod auth;
#[path = "tests/events.rs"]
mod events;
#[path = "tests/graph.rs"]
mod graph;
#[path = "tests/projections.rs"]
mod projections;
#[path = "tests/runtime.rs"]
mod runtime;
#[path = "tests/schema_compat.rs"]
mod schema_compat;
