// Domain-focused tests; shared fixtures live in the parent test module.
use super::*;

#[test]
fn recorded_transition_lookup_uses_json_expression_index() {
    let (store, path) = store("runtime-transition-index");
    drop(store);
    let conn = Connection::open(&path).unwrap();
    let plan = conn
            .prepare(
                "EXPLAIN QUERY PLAN SELECT payload_json, seq FROM server_run_events \
                 WHERE event_type = ?1 AND json_extract(payload_json, '$.transition_id') = ?2 LIMIT 1",
            )
            .unwrap()
            .query_map(params![RECORDED_RUNTIME_EVENT_TYPE, "transition-1"], |row| {
                row.get::<_, String>(3)
            })
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
            .join(" ");
    assert!(
        plan.contains("idx_server_run_events_recorded_transition_id"),
        "{plan}"
    );
    drop(conn);
    fs::remove_file(path).unwrap();
}
#[test]
fn runtime_current_state_projection_is_disposable_fresh_and_atomic() {
    let (store, path) = store("runtime-current-state");
    let start_transition = runtime_start("run-1", "start-1");
    let start = store
        .apply_recorded_runtime_transition(start_transition.clone(), false)
        .unwrap();
    let exact_retry = store
        .apply_recorded_runtime_transition(start_transition.clone(), false)
        .unwrap();
    assert!(exact_retry.idempotent);
    assert_eq!(start.event_seq, exact_retry.event_seq);
    let mut conflicting_start = start_transition;
    conflicting_start.initial_state = Some(Map::from_iter([(
        "answer".to_owned(),
        Value::String("changed".to_owned()),
    )]));
    assert!(matches!(
        store.apply_recorded_runtime_transition(conflicting_start, false),
        Err(SqliteStoreError::RecordedRuntimeConflict(_))
    ));

    let start_projection = store
        .get_named_projection(RUNTIME_CURRENT_STATE_NAMESPACE, "run-1")
        .unwrap()
        .unwrap();
    assert_eq!(start_projection.last_authoritative_seq, start.event_seq);
    assert_eq!(start_projection.payload["state"]["answer"], "seed");

    let result_transition = runtime_result("run-1", "result-1", start.event_seq, "worker");
    assert!(matches!(
        store.apply_recorded_runtime_transition(result_transition.clone(), true),
        Err(SqliteStoreError::TransactionAborted(_))
    ));
    assert_eq!(
        store.list_server_run_events("run-1", 0, 10).unwrap().len(),
        1
    );
    assert_eq!(
        store
            .get_named_projection(RUNTIME_CURRENT_STATE_NAMESPACE, "run-1")
            .unwrap()
            .unwrap()
            .last_authoritative_seq,
        start.event_seq
    );

    let result = store
        .apply_recorded_runtime_transition(result_transition, false)
        .unwrap();
    assert_eq!(
        store
            .read_recorded_runtime_state("run-1", "wf-1", "conv-1")
            .unwrap()
            .unwrap()
            .state["answer"],
        "worker"
    );

    let conn = Connection::open(&path).unwrap();
    conn.execute(
        "UPDATE named_projections SET payload_json='{}', last_authoritative_seq=?1 \
             WHERE namespace=?2 AND key='run-1'",
        params![start.event_seq, RUNTIME_CURRENT_STATE_NAMESPACE],
    )
    .unwrap();
    drop(conn);
    let replayed = store
        .read_recorded_runtime_state("run-1", "wf-1", "conv-1")
        .unwrap()
        .unwrap();
    assert_eq!(replayed.state["answer"], "worker");
    assert_eq!(result.reduced.state, replayed);

    let conn = Connection::open(&path).unwrap();
    conn.execute(
        "DELETE FROM named_projections WHERE namespace=?1 AND key='run-1'",
        [RUNTIME_CURRENT_STATE_NAMESPACE],
    )
    .unwrap();
    drop(conn);
    assert_eq!(
        store
            .read_recorded_runtime_state("run-1", "wf-1", "conv-1")
            .unwrap()
            .unwrap(),
        result.reduced.state
    );
    drop(store);
    fs::remove_file(path).unwrap();
}

#[test]
fn completed_answer_transition_materializes_assistant_turn_id() {
    let (store, path) = store("runtime-assistant-turn-id");
    let start = store
        .apply_recorded_runtime_transition(runtime_start("run-answer", "start-answer"), false)
        .unwrap();
    let mut result_transition =
        runtime_result("run-answer", "complete-answer", start.event_seq, "answer");
    result_transition.kind = kogwistar_runtime::RecordedTransitionKind::Complete;
    result_transition
        .state_update
        .push(kogwistar_runtime::RecordedStateUpdate::Pair((
            "u".to_owned(),
            Map::from_iter([(
                "agentic_answering_result".to_owned(),
                serde_json::json!({"assistant_turn_node_id": "assistant-turn-1"}),
            )]),
        )));

    let completed = store
        .apply_recorded_runtime_transition(result_transition, false)
        .unwrap();
    assert_eq!(
        completed.reduced.state.status,
        kogwistar_runtime::RecordedRunStatus::Completed
    );
    assert_eq!(
        store
            .get_server_run("run-answer")
            .unwrap()
            .unwrap()
            .assistant_turn_node_id
            .as_deref(),
        Some("assistant-turn-1")
    );

    drop(store);
    fs::remove_file(path).unwrap();
}
#[test]
fn worker_failure_routing_is_atomic_handled_terminal_and_retry_safe() {
    let (store, path) = store("runtime-worker-failure");
    let setup = |run_id: &str, message_id: &str, owner: &str| {
        store
            .apply_recorded_runtime_transition(
                runtime_failure_start(run_id, &format!("start-{run_id}")),
                false,
            )
            .unwrap();
        store
            .project_lane_message(runtime_worker_lane(run_id, message_id))
            .unwrap();
        assert_eq!(
            store
                .claim_projected_lane_messages("runtime", "python-workers", owner, 1, 30,)
                .unwrap()
                .len(),
            1
        );
    };

    setup("handled", "handled-request", "handled-worker");
    let handled_handoff = runtime_worker_handoff("handled", "handled-request", "handled-worker");
    let handled_effect =
        runtime_failure_effect("handled-effect", &["recover"], serde_json::json!([]));
    let handled = store
        .apply_claimed_recorded_worker_effect(handled_handoff.clone(), handled_effect.clone())
        .unwrap();
    let handled_retry = store
        .apply_claimed_recorded_worker_effect(handled_handoff, handled_effect)
        .unwrap();
    assert!(!handled.idempotent && handled_retry.idempotent);
    assert_eq!(handled.event_seq, handled_retry.event_seq);
    assert_eq!(
        handled.reduced.state.status,
        kogwistar_runtime::RecordedRunStatus::Running
    );
    assert_eq!(handled.reduced.state.frontier.pending[0].0, "recover");
    assert_eq!(
        store
            .get_projected_lane_message("handled-request")
            .unwrap()
            .unwrap()
            .status,
        "completed"
    );
    assert_eq!(
        store
            .list_server_run_events("handled", 0, 20)
            .unwrap()
            .len(),
        2
    );

    setup("unhandled", "unhandled-request", "unhandled-worker");
    let unhandled = store
        .apply_claimed_recorded_worker_effect(
            runtime_worker_handoff("unhandled", "unhandled-request", "unhandled-worker"),
            runtime_failure_effect("unhandled-effect", &[], serde_json::json!([])),
        )
        .unwrap();
    assert_eq!(
        unhandled.reduced.state.status,
        kogwistar_runtime::RecordedRunStatus::Failed
    );
    assert!(unhandled.reduced.state.frontier.pending.is_empty());
    assert_eq!(
        store.get_server_run("unhandled").unwrap().unwrap().status,
        "failed"
    );

    setup("forged", "forged-request", "forged-worker");
    assert!(matches!(
        store.apply_claimed_recorded_worker_effect(
            runtime_worker_handoff("forged", "forged-request", "forged-worker"),
            runtime_failure_effect(
                "forged-effect",
                &[],
                serde_json::json!([{"node_id": "forged", "join_mask": 0}]),
            ),
        ),
        Err(SqliteStoreError::RecordedRuntime(_))
    ));
    assert_eq!(
        store.list_server_run_events("forged", 0, 20).unwrap().len(),
        1
    );
    assert_eq!(
        store
            .get_projected_lane_message("forged-request")
            .unwrap()
            .unwrap()
            .status,
        "claimed"
    );
    assert_eq!(
        store
            .read_recorded_runtime_state("forged", "wf-1", "conv-1")
            .unwrap()
            .unwrap()
            .state["answer"],
        "seed"
    );

    drop(store);
    fs::remove_file(path).unwrap();
}
/// Manual code-smell probe. Absolute timings vary by host; growth ratio is
/// the useful signal. Both indexed retry lookup and warm current-state read
/// should remain near-flat as unrelated recorded history grows.
#[test]
#[ignore = "manual runtime serving-path scale probe"]
fn runtime_serving_lookup_scale_probe() {
    use std::time::Instant;

    fn measure(history_size: usize) -> (u128, u128) {
        let (store, path) = store(&format!("runtime-scale-{history_size}"));
        let start_transition = runtime_start("target-run", "target-transition");
        store
            .apply_recorded_runtime_transition(start_transition.clone(), false)
            .unwrap();
        let persisted_json = store
            .list_server_run_events("target-run", 0, 1)
            .unwrap()
            .pop()
            .unwrap()
            .payload_json;
        let mut persisted: Value = serde_json::from_str(&persisted_json).unwrap();
        let conn = Connection::open(&path).unwrap();
        let transaction = conn.unchecked_transaction().unwrap();
        for index in 0..history_size {
            persisted["transition_id"] = Value::String(format!("noise-{index}"));
            transaction
                .execute(
                    "INSERT INTO server_run_events(run_id,event_type,payload_json,created_at_ms) \
                         VALUES (?1,?2,?3,0)",
                    params![
                        format!("noise-run-{index}"),
                        RECORDED_RUNTIME_EVENT_TYPE,
                        serde_json::to_string(&persisted).unwrap()
                    ],
                )
                .unwrap();
        }
        transaction.commit().unwrap();
        drop(conn);

        let retry_started = Instant::now();
        for _ in 0..200 {
            assert!(
                store
                    .apply_recorded_runtime_transition(start_transition.clone(), false)
                    .unwrap()
                    .idempotent
            );
        }
        let retry_ns = retry_started.elapsed().as_nanos();
        let state_started = Instant::now();
        for _ in 0..200 {
            assert!(
                store
                    .read_recorded_runtime_state("target-run", "wf-1", "conv-1")
                    .unwrap()
                    .is_some()
            );
        }
        let state_ns = state_started.elapsed().as_nanos();
        drop(store);
        fs::remove_file(path).unwrap();
        (retry_ns, state_ns)
    }

    let small = measure(100);
    let large = measure(10_000);
    eprintln!(
        "runtime-serving-scale small_100={{retry_ns:{},state_ns:{}}} \
             large_10000={{retry_ns:{},state_ns:{}}} ratios={{retry:{:.2},state:{:.2}}}",
        small.0,
        small.1,
        large.0,
        large.1,
        large.0 as f64 / small.0.max(1) as f64,
        large.1 as f64 / small.1.max(1) as f64,
    );
}
