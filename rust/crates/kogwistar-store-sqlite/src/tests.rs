use super::*;
use std::future::Future;
use std::pin::pin;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
use std::thread;

static NEXT_DB: AtomicUsize = AtomicUsize::new(0);

fn database_path(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "kogwistar-store-sqlite-{label}-{}-{}.db",
        std::process::id(),
        NEXT_DB.fetch_add(1, Ordering::Relaxed),
    ))
}

fn store(label: &str) -> (SqliteStore, PathBuf) {
    let path = database_path(label);
    (SqliteStore::open(&path).unwrap(), path)
}

#[test]
fn private_memory_database_preserves_sqlite_memory_semantics() {
    let first = SqliteStore::open(":memory:").unwrap();
    let second = SqliteStore::open(":memory:").unwrap();

    assert_eq!(first.next_global_seq().unwrap(), 1);
    assert_eq!(first.current_global_seq().unwrap(), 1);
    assert_eq!(second.current_global_seq().unwrap(), 0);

    let first_connection = first.connection.lock().unwrap();
    let first_filename: String = first_connection
        .query_row("PRAGMA database_list", [], |row| row.get(2))
        .unwrap();
    assert!(first_filename.is_empty());
    first_connection
        .execute("CREATE TABLE private_memory_probe (value INTEGER)", [])
        .unwrap();
    drop(first_connection);

    let second_connection = second.connection.lock().unwrap();
    let second_filename: String = second_connection
        .query_row("PRAGMA database_list", [], |row| row.get(2))
        .unwrap();
    assert!(second_filename.is_empty());
    let table_count: i64 = second_connection
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'private_memory_probe'",
                [],
                |row| row.get(0),
            )
            .unwrap();
    assert_eq!(table_count, 0);
}

#[test]
fn expired_claim_token_cannot_mutate_queue_state() {
    let (store, path) = store("expired-index-claim");
    for job_id in ["done", "failed", "retried", "requeued"] {
        store
            .enqueue_index_job(NewIndexJob {
                job_id: job_id.to_owned(),
                namespace: "ns".to_owned(),
                entity_kind: "node".to_owned(),
                entity_id: job_id.to_owned(),
                index_kind: "test".to_owned(),
                op: "UPSERT".to_owned(),
                payload_json: None,
                max_retries: 3,
            })
            .unwrap();
    }
    let claims = store.claim_index_jobs(4, -1, Some("ns")).unwrap();
    let tokens = claims
        .iter()
        .map(|claim| (claim.job_id.as_str(), claim.claim_token.as_deref().unwrap()))
        .collect::<BTreeMap<_, _>>();

    assert!(
        !store
            .mark_index_job_done("done", Some(tokens["done"]))
            .unwrap()
    );
    store
        .mark_index_job_failed("failed", "late", true, Some(tokens["failed"]))
        .unwrap();
    store
        .bump_retry_and_requeue("retried", "late", 0, Some(tokens["retried"]))
        .unwrap();
    store
        .requeue_index_job_at_tail("requeued", "{}".to_owned(), 0, Some(tokens["requeued"]))
        .unwrap();

    let reclaimed = store.claim_index_jobs(4, 30, Some("ns")).unwrap();
    assert_eq!(reclaimed.len(), 4);
    assert!(reclaimed.iter().all(|claim| claim.retry_count == 0));
    assert!(
        reclaimed
            .iter()
            .all(|claim| { claim.claim_token.as_deref() != Some(tokens[claim.job_id.as_str()]) })
    );
    assert!(store.mark_index_job_done("done", None).unwrap());
    drop(store);
    fs::remove_file(path).unwrap();
}

#[test]
fn active_path_identity_is_shared_and_replacement_fails_closed() {
    let path = database_path("identity-registry");
    register_active_database_identity(path.clone(), "file-identity-a", &path).unwrap();
    register_active_database_identity(path.clone(), "file-identity-a", &path).unwrap();
    assert!(matches!(
        register_active_database_identity(path.clone(), "file-identity-b", &path),
        Err(SqliteStoreError::AuthorityConflict { .. })
    ));

    unregister_active_database_identity(&path, "file-identity-a");
    assert!(matches!(
        register_active_database_identity(path.clone(), "file-identity-b", &path),
        Err(SqliteStoreError::AuthorityConflict { .. })
    ));
    unregister_active_database_identity(&path, "file-identity-a");
    register_active_database_identity(path.clone(), "file-identity-b", &path).unwrap();
    unregister_active_database_identity(&path, "file-identity-b");
}

#[test]
fn concurrent_file_identity_replacement_has_one_active_winner() {
    let path = database_path("identity-race");
    let barrier = Arc::new(Barrier::new(3));
    let workers = ["file-identity-a", "file-identity-b"].map(|identity| {
        let path = path.clone();
        let barrier = Arc::clone(&barrier);
        thread::spawn(move || {
            barrier.wait();
            let result = register_active_database_identity(path.clone(), identity, &path);
            (identity, result.is_ok())
        })
    });
    barrier.wait();
    let outcomes = workers.map(|worker| worker.join().unwrap());
    let winners: Vec<_> = outcomes
        .iter()
        .filter_map(|(identity, accepted)| accepted.then_some(*identity))
        .collect();
    assert_eq!(winners.len(), 1);

    let active = ACTIVE_DATABASE_IDENTITIES.get().unwrap().lock().unwrap();
    assert_eq!(active.get(&path).map(|(_, count)| *count), Some(1));
    drop(active);
    unregister_active_database_identity(&path, winners[0]);
}

#[test]
fn standalone_rust_process_cannot_open_an_authority_owned_database() {
    let Ok(path) = std::env::var("KOGWISTAR_SQLITE_LOCK_CHILD_PATH") else {
        let path = database_path("cross-process-authority");
        let ready_path = PathBuf::from(format!("{}.ready", path.display()));
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args([
                "tests::standalone_rust_process_cannot_open_an_authority_owned_database",
                "--nocapture",
            ])
            .env("KOGWISTAR_SQLITE_LOCK_CHILD_PATH", &path)
            .env("KOGWISTAR_SQLITE_LOCK_CHILD_READY", &ready_path)
            .stdin(Stdio::piped())
            .spawn()
            .unwrap();
        for _ in 0..50 {
            if ready_path.exists() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        assert!(
            ready_path.exists(),
            "child did not acquire SQLite authority"
        );
        let error = SqliteStore::open(&path).unwrap_err();
        assert!(matches!(error, SqliteStoreError::AuthorityConflict { .. }));
        drop(child.stdin.take());
        assert!(child.wait().unwrap().success());
        let _ = fs::remove_file(&path);
        let _ = fs::remove_file(&ready_path);
        let _ = fs::remove_file(format!("{}.kogwistar-authority.lock", path.display()));
        return;
    };

    let path = PathBuf::from(path);
    let expect_conflict = std::env::var_os("KOGWISTAR_SQLITE_LOCK_EXPECT_CONFLICT").is_some();
    if expect_conflict {
        assert!(matches!(
            SqliteStore::open(&path),
            Err(SqliteStoreError::AuthorityConflict { .. })
        ));
    } else {
        let store = SqliteStore::open(&path).unwrap();
        std::hint::black_box(&store);
        fs::write(
            std::env::var("KOGWISTAR_SQLITE_LOCK_CHILD_READY").unwrap(),
            b"ready",
        )
        .unwrap();
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();
        drop(store);
        return;
    }
    fs::write(
        std::env::var("KOGWISTAR_SQLITE_LOCK_CHILD_READY").unwrap(),
        b"ready",
    )
    .unwrap();
    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();
}

#[cfg(unix)]
#[test]
fn active_rust_authority_rejects_database_path_replacement() {
    let (store, path) = store("active-path-replacement");
    let original = path.with_extension("original.db");
    let replacement = path.with_extension("replacement.db");
    let replacement_connection = Connection::open(&replacement).unwrap();
    replacement_connection
        .execute_batch("CREATE TABLE replacement_probe (value INTEGER);")
        .unwrap();
    drop(replacement_connection);

    fs::rename(&path, &original).unwrap();
    fs::rename(&replacement, &path).unwrap();
    let error = SqliteStore::open(&path).unwrap_err();
    assert!(matches!(error, SqliteStoreError::AuthorityConflict { .. }));

    drop(store);
    fs::remove_file(original).unwrap();
    fs::remove_file(path).unwrap();
}

#[test]
fn restored_raw_event_preserves_identity_and_is_idempotent() {
    let (store, path) = store("restored-raw-event");
    let event = RestoredRawEntityEvent {
        seq: 1,
        event_id: "archive-event".to_owned(),
        entity_kind: "node".to_owned(),
        entity_id: "node-1".to_owned(),
        op: "UPSERT".to_owned(),
        payload_json: "{\"value\":1}".to_owned(),
        created_at: 123,
    };
    assert!(
        store
            .append_restored_raw_entity_event("archive", event.clone())
            .unwrap()
            .inserted
    );
    assert!(
        !store
            .append_restored_raw_entity_event("archive", event.clone())
            .unwrap()
            .inserted
    );
    assert_eq!(
        store.replay_raw_events("archive", 0, 10).unwrap()[0].created_at,
        123
    );
    assert!(
            store
                .append_restored_raw_entity_event(
                    "archive",
                    RestoredRawEntityEvent { seq: 3, ..event },
                )
                .is_err()
        );
    drop(store);
    fs::remove_file(path).unwrap();
}

#[test]
fn stage1_node_projection_isolated_crud() {
    let (store, path) = store("stage1-crud");
    let projection = NamedProjectionWrite {
        payload: serde_json::json!({
            "id": "node-1",
            "metadata": {"kind": "person"}
        })
        .as_object()
        .unwrap()
        .clone(),
        last_authoritative_seq: 7,
        last_materialized_seq: 7,
        projection_schema_version: 1,
        materialization_status: "pending".to_owned(),
    };
    store
        .replace_stage1_node_projection("tenant-a", "node-1", projection)
        .unwrap();
    assert_eq!(
        store
            .get_stage1_node_projection("tenant-a", "node-1")
            .unwrap()
            .unwrap()
            .payload["metadata"]["kind"],
        "person"
    );
    assert_eq!(
        store
            .list_stage1_node_projections("tenant-a")
            .unwrap()
            .len(),
        1
    );
    assert!(
        store
            .get_named_projection("tenant-a", "node-1")
            .unwrap()
            .is_none()
    );
    store
        .clear_stage1_node_projection("tenant-a", "node-1")
        .unwrap();
    assert!(
        store
            .get_stage1_node_projection("tenant-a", "node-1")
            .unwrap()
            .is_none()
    );
    drop(store);
    fs::remove_file(path).unwrap();
}

#[test]
fn named_projection_batch_cas_is_all_or_nothing() {
    let (store, path) = store("named-projection-batch-cas");
    let projection = |value: i64| NamedProjectionWrite {
        payload: serde_json::json!({"value": value})
            .as_object()
            .unwrap()
            .clone(),
        last_authoritative_seq: value,
        last_materialized_seq: value,
        projection_schema_version: 1,
        materialization_status: "ready".to_owned(),
    };

    assert!(
        !store
            .compare_and_swap_named_projections(vec![
                NamedProjectionCas {
                    namespace: "skills".to_owned(),
                    key: "provider".to_owned(),
                    expected_last_authoritative_seq: None,
                    expected_last_materialized_seq: None,
                    projection: projection(1),
                },
                NamedProjectionCas {
                    namespace: "catalog".to_owned(),
                    key: "provider".to_owned(),
                    expected_last_authoritative_seq: Some(99),
                    expected_last_materialized_seq: Some(99),
                    projection: projection(1),
                },
            ])
            .unwrap()
    );
    assert!(
        store
            .get_named_projection("skills", "provider")
            .unwrap()
            .is_none()
    );

    assert!(
        store
            .compare_and_swap_named_projections(vec![
                NamedProjectionCas {
                    namespace: "skills".to_owned(),
                    key: "provider".to_owned(),
                    expected_last_authoritative_seq: None,
                    expected_last_materialized_seq: None,
                    projection: projection(1),
                },
                NamedProjectionCas {
                    namespace: "catalog".to_owned(),
                    key: "provider".to_owned(),
                    expected_last_authoritative_seq: None,
                    expected_last_materialized_seq: None,
                    projection: projection(1),
                },
            ])
            .unwrap()
    );
    assert_eq!(
        store
            .get_named_projection("skills", "provider")
            .unwrap()
            .unwrap()
            .payload["value"],
        1
    );
    assert_eq!(
        store
            .get_named_projection("catalog", "provider")
            .unwrap()
            .unwrap()
            .payload["value"],
        1
    );
    drop(store);
    fs::remove_file(path).unwrap();
}

fn event(event_id: &str, entity_id: &str) -> NewEntityEvent {
    NewEntityEvent {
        event_id: event_id.to_owned(),
        entity_kind: "node".to_owned(),
        entity_id: entity_id.to_owned(),
        op: "UPSERT".to_owned(),
        payload: serde_json::json!({"id": entity_id}),
    }
}

fn raw_event(event_id: &str, payload_json: &str) -> NewRawEntityEvent {
    NewRawEntityEvent {
        event_id: event_id.to_owned(),
        entity_kind: "node".to_owned(),
        entity_id: "node-1".to_owned(),
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
        "initial_state": {"answer": "seed"},
        "frontier": {
            "pending": [["step-1", 0, "token-1", null]],
            "suspended": [], "join_node_ids": [], "join_outstanding": [],
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
        "state_update": [["u", {"answer": answer}]],
        "frontier": {
            "pending": [], "suspended": [], "join_node_ids": [],
            "join_outstanding": [], "join_waiters": {}
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

fn runtime_worker_lane(run_id: &str, message_id: &str) -> NewProjectedLaneMessage {
    NewProjectedLaneMessage {
        message_id: message_id.to_owned(),
        namespace: "runtime".to_owned(),
        purpose: "system".to_owned(),
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
        correlation_id: Some(run_id.to_owned()),
        payload_json: Some(
            serde_json::json!({
                "workflow_id": "wf-1",
                "conversation_id": "conv-1",
                "token_id": "token-1",
                "parent_token_id": null
            })
            .to_string(),
        ),
        error_json: None,
    }
}

fn runtime_worker_handoff(run_id: &str, message_id: &str, owner: &str) -> RecordedWorkerHandoff {
    RecordedWorkerHandoff {
        message_id: message_id.to_owned(),
        claimed_by: owner.to_owned(),
        run_id: run_id.to_owned(),
        step_id: "step-1".to_owned(),
        correlation_id: run_id.to_owned(),
    }
}

#[test]
fn targeted_lane_claim_does_not_take_earlier_other_run() {
    let (store, path) = store("targeted-runtime-claim");
    store
        .project_lane_message(runtime_worker_lane("run-decoy", "lane-decoy"))
        .unwrap();
    store
        .project_lane_message(runtime_worker_lane("run-target", "lane-target"))
        .unwrap();

    let claimed = store
        .immediate_transaction(|uow| {
            uow.claim_projected_lane_messages_for_run(
                "runtime",
                "python-workers",
                "run-target",
                "target-worker",
                1,
                60,
            )
        })
        .unwrap();

    assert_eq!(claimed.len(), 1);
    assert_eq!(claimed[0].message_id, "lane-target");
    assert_eq!(
        store
            .get_projected_lane_message("lane-decoy")
            .unwrap()
            .unwrap()
            .status,
        "pending"
    );
    drop(store);
    fs::remove_file(path).unwrap();
}

#[test]
fn filtered_lane_claim_is_atomic_for_run_kind_recipient_and_ids() {
    let (store, path) = store("filtered-control-claim");
    store
        .project_lane_message(runtime_worker_lane("run-target", "lane-good"))
        .unwrap();
    let mut wrong_kind = runtime_worker_lane("run-target", "lane-kind");
    wrong_kind.msg_type = "agent.queue".to_owned();
    store.project_lane_message(wrong_kind).unwrap();
    let mut wrong_recipient = runtime_worker_lane("run-target", "lane-recipient");
    wrong_recipient.recipient_id = "other-agent".to_owned();
    store.project_lane_message(wrong_recipient).unwrap();
    let wrong_run = runtime_worker_lane("run-other", "lane-run");
    store.project_lane_message(wrong_run).unwrap();

    let claimed = store
        .claim_projected_lane_messages_filtered(LaneMessageClaimFilter {
            namespace: "runtime".to_owned(),
            inbox_id: "python-workers".to_owned(),
            claimed_by: "target-worker".to_owned(),
            message_ids: Some(vec![
                "lane-good".to_owned(),
                "lane-kind".to_owned(),
                "lane-recipient".to_owned(),
                "lane-run".to_owned(),
            ]),
            run_id: Some("run-target".to_owned()),
            msg_type: Some("workflow.worker.request.v1".to_owned()),
            recipient_id: Some("python-worker".to_owned()),
            limit: 10,
            lease_seconds: 60,
        })
        .unwrap();

    assert_eq!(
        claimed
            .iter()
            .map(|row| row.message_id.as_str())
            .collect::<Vec<_>>(),
        ["lane-good"]
    );
    for message_id in ["lane-kind", "lane-recipient", "lane-run"] {
        assert_eq!(
            store
                .get_projected_lane_message(message_id)
                .unwrap()
                .unwrap()
                .status,
            "pending"
        );
    }
    drop(store);
    fs::remove_file(path).unwrap();
}

#[test]
fn expired_lane_claim_cannot_be_finalized_before_reclaim() {
    let (store, path) = store("expired-lane-claim");
    for (message_id, operation) in [
        ("expired-ack", "ack"),
        ("expired-requeue", "requeue"),
        ("expired-dead-letter", "dead-letter"),
    ] {
        let mut row = runtime_worker_lane("run-target", message_id);
        row.available_at = 0;
        store.project_lane_message(row).unwrap();
        let claimed = store
            .claim_projected_lane_messages_filtered(LaneMessageClaimFilter {
                namespace: "runtime".to_owned(),
                inbox_id: "python-workers".to_owned(),
                claimed_by: "attempt-1".to_owned(),
                message_ids: Some(vec![message_id.to_owned()]),
                run_id: Some("run-target".to_owned()),
                msg_type: Some("workflow.worker.request.v1".to_owned()),
                recipient_id: Some("python-worker".to_owned()),
                limit: 1,
                lease_seconds: -1,
            })
            .unwrap();
        assert_eq!(claimed.len(), 1);
        assert!(claimed[0].lease_until.as_ref().unwrap().as_i64().unwrap() < unix_epoch_seconds());

        match operation {
            "ack" => store
                .ack_projected_lane_message(message_id, "attempt-1")
                .unwrap(),
            "requeue" => store
                .requeue_projected_lane_message(message_id, "attempt-1", None, 0)
                .unwrap(),
            "dead-letter" => store
                .dead_letter_projected_lane_message(message_id, "attempt-1", None)
                .unwrap(),
            _ => unreachable!(),
        }
        let row = store
            .get_projected_lane_message(message_id)
            .unwrap()
            .unwrap();
        assert_eq!(row.status, "claimed", "{operation}");
        assert_eq!(row.claimed_by.as_deref(), Some("attempt-1"), "{operation}");
        assert_eq!(row.retry_count, 0, "{operation}");
    }
    drop(store);
    fs::remove_file(path).unwrap();
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

#[test]
fn auth_store_opens_python_schema_and_resolves_identity_atomically() {
    let path = database_path("auth");
    let store = SqliteAuthStore::open(&path).unwrap();
    let request = ResolveExternalIdentity {
        issuer: "https://issuer.example".to_owned(),
        subject: "subject-1".to_owned(),
        email: "alice@example.com".to_owned(),
        display_name: Some("Alice".to_owned()),
        new_user_id: "generated-1".to_owned(),
        default_role: "ro".to_owned(),
        default_ns: "docs".to_owned(),
    };
    let created = block_on(store.resolve_external_identity(request.clone())).unwrap();
    assert_eq!(created.user_id, "generated-1");
    assert_eq!(created.email, "alice@example.com");
    assert_eq!(created.display_name.as_deref(), Some("Alice"));
    assert_eq!(created.global_role.as_deref(), Some("ro"));
    let retried = block_on(store.resolve_external_identity(ResolveExternalIdentity {
        new_user_id: "must-not-win".to_owned(),
        email: "changed@example.com".to_owned(),
        ..request
    }))
    .unwrap();
    assert_eq!(retried.user_id, "generated-1");
    let identity = block_on(store.external_identity("https://issuer.example", "subject-1"))
        .unwrap()
        .unwrap();
    assert_eq!(identity.user_id, "generated-1");
    assert_eq!(identity.email.as_deref(), Some("alice@example.com"));
    let connection = Connection::open(&path).unwrap();
    let table_names = connection
        .prepare("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")
        .unwrap()
        .query_map([], |row| row.get::<_, String>(0))
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(
        table_names,
        ["external_identities", "users", "workflow_acl"]
    );
    drop(connection);
    drop(store);
    fs::remove_file(path).unwrap();
}

#[test]
fn auth_store_links_existing_python_user_by_email() {
    let path = database_path("auth-python-user");
    let store = SqliteAuthStore::open(&path).unwrap();
    let connection = Connection::open(&path).unwrap();
    connection
            .execute(
                "INSERT INTO users (user_id, email, display_name, is_active, global_role, global_ns, created_at) VALUES ('python-user', 'alice@example.com', 'Python Alice', 1, 'rw', 'docs,workflow', CURRENT_TIMESTAMP)",
                [],
            )
            .unwrap();
    drop(connection);
    let user = block_on(store.resolve_external_identity(ResolveExternalIdentity {
        issuer: "issuer".to_owned(),
        subject: "subject".to_owned(),
        email: "alice@example.com".to_owned(),
        display_name: Some("Ignored".to_owned()),
        new_user_id: "rust-user".to_owned(),
        default_role: "ro".to_owned(),
        default_ns: "docs".to_owned(),
    }))
    .unwrap();
    assert_eq!(user.user_id, "python-user");
    assert_eq!(user.display_name.as_deref(), Some("Python Alice"));
    assert_eq!(user.global_role.as_deref(), Some("rw"));
    assert_eq!(
        block_on(store.external_identity("issuer", "subject"))
            .unwrap()
            .unwrap()
            .user_id,
        "python-user"
    );
    drop(store);
    fs::remove_file(path).unwrap();
}

fn block_on<T>(future: impl Future<Output = T>) -> T {
    fn no_op(_: *const ()) {}
    fn clone(_: *const ()) -> RawWaker {
        RawWaker::new(std::ptr::null(), &VTABLE)
    }
    static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, no_op, no_op, no_op);
    let raw = RawWaker::new(std::ptr::null(), &VTABLE);
    let waker = unsafe { Waker::from_raw(raw) };
    let mut context = Context::from_waker(&waker);
    let mut future = pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => thread::yield_now(),
        }
    }
}

#[test]
fn python_legacy_shaped_database_is_opened_read_and_written_without_reformatting_json() {
    let path = database_path("legacy");
    let conn = Connection::open(&path).unwrap();
    conn.execute_batch(
            "
            CREATE TABLE global_seq (value INTEGER NOT NULL);
            INSERT INTO global_seq(rowid, value) VALUES (1, 4);
            CREATE TABLE user_seq (user_id TEXT PRIMARY KEY, value INTEGER NOT NULL);
            INSERT INTO user_seq VALUES ('alice', 8);
            CREATE TABLE namespace_seq (namespace TEXT PRIMARY KEY, next_seq INTEGER NOT NULL);
            INSERT INTO namespace_seq VALUES ('legacy', 7);
            CREATE TABLE entity_events (
                namespace TEXT NOT NULL DEFAULT 'default', seq INTEGER NOT NULL, event_id TEXT NOT NULL,
                entity_kind TEXT NOT NULL, entity_id TEXT NOT NULL, op TEXT NOT NULL,
                payload_json TEXT NOT NULL, created_at INTEGER NOT NULL,
                PRIMARY KEY(namespace, seq), UNIQUE(event_id)
            );
            INSERT INTO entity_events VALUES ('legacy', 6, 'old-id', 'node', 'old', 'UPSERT', '{\"spaced\": [ 1, 2 ]}', 123);
            CREATE TABLE replay_cursors (
                namespace TEXT NOT NULL DEFAULT 'default', consumer TEXT NOT NULL,
                last_seq INTEGER NOT NULL, updated_at INTEGER NOT NULL,
                PRIMARY KEY(namespace, consumer)
            );
            INSERT INTO replay_cursors VALUES ('legacy', 'reader', 6, 124);
            ",
        )
        .unwrap();
    drop(conn);

    let store = SqliteStore::open(&path).unwrap();
    assert_eq!(store.current_global_seq().unwrap(), 4);
    assert_eq!(store.current_user_seq("alice").unwrap(), 8);
    assert_eq!(
        store.replay_raw_events("legacy", 0, 10).unwrap()[0].payload_json,
        "{\"spaced\": [ 1, 2 ]}"
    );
    assert_eq!(store.replay_cursor("legacy", "reader").unwrap().last_seq, 6);
    assert_eq!(
        store
            .append_raw_entity_event("legacy", raw_event("new-id", "{\"original\": true }"))
            .unwrap()
            .event
            .seq,
        7
    );
    drop(store);

    let conn = Connection::open(&path).unwrap();
    let row: (i64, String) = conn
            .query_row(
                "SELECT seq, payload_json FROM entity_events WHERE namespace = 'legacy' AND event_id = 'new-id'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
    assert_eq!(row, (7, "{\"original\": true }".to_owned()));
    drop(conn);
    fs::remove_file(path).unwrap();
}

#[test]
fn autocommit_reads_observe_commits_from_an_external_connection() {
    let (store, path) = store("external-read");
    let external = Connection::open(&path).unwrap();
    external
        .execute("UPDATE global_seq SET value = 1 WHERE rowid = 1", [])
        .unwrap();
    assert_eq!(store.current_global_seq().unwrap(), 1);
    drop(external);
    drop(store);
    fs::remove_file(path).unwrap();
}

#[test]
fn python_legacy_queue_schema_gets_all_additive_columns_before_indexes() {
    let path = database_path("legacy-queue-migration");
    let conn = Connection::open(&path).unwrap();
    conn.execute_batch(
            "CREATE TABLE index_jobs (
                job_id TEXT PRIMARY KEY, entity_kind TEXT NOT NULL, entity_id TEXT NOT NULL,
                index_kind TEXT NOT NULL, op TEXT NOT NULL, status TEXT NOT NULL,
                lease_until INTEGER, retry_count INTEGER NOT NULL DEFAULT 0,
                last_error TEXT, payload_json TEXT, created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
             );
             CREATE TABLE projected_lane_messages (
                message_id TEXT PRIMARY KEY, namespace TEXT NOT NULL DEFAULT 'default',
                inbox_id TEXT NOT NULL, conversation_id TEXT NOT NULL, recipient_id TEXT NOT NULL,
                sender_id TEXT NOT NULL, msg_type TEXT NOT NULL, status TEXT NOT NULL,
                seq INTEGER NOT NULL, conversation_seq INTEGER NOT NULL, claimed_by TEXT,
                lease_until INTEGER, retry_count INTEGER NOT NULL DEFAULT 0, created_at INTEGER NOT NULL,
                available_at INTEGER NOT NULL, run_id TEXT, step_id TEXT, correlation_id TEXT,
                payload_json TEXT, error_json TEXT, prev_message_id TEXT, next_message_id TEXT,
                inbox_tail_message_id TEXT, conversation_tail_message_id TEXT
             );",
        )
        .unwrap();
    drop(conn);

    let store = SqliteStore::open(&path).unwrap();
    let conn = Connection::open(&path).unwrap();
    let index_job_columns = conn
        .prepare("PRAGMA table_info(index_jobs)")
        .unwrap()
        .query_map([], |row| row.get::<_, String>(1))
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    for expected in [
        "namespace",
        "coalesce_key",
        "next_run_at",
        "max_retries",
        "claim_token",
    ] {
        assert!(index_job_columns.iter().any(|column| column == expected));
    }
    let purpose_exists = conn
        .prepare("PRAGMA table_info(projected_lane_messages)")
        .unwrap()
        .query_map([], |row| row.get::<_, String>(1))
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
        .iter()
        .any(|column| column == "purpose");
    assert!(purpose_exists);
    let coalesce_index_exists: bool = conn
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='index' AND name='uq_index_jobs_pending_ns_coalesce')",
                [],
                |row| row.get(0),
            )
            .unwrap();
    assert!(coalesce_index_exists);
    drop(conn);
    drop(store);
    fs::remove_file(path).unwrap();
}

#[test]
fn rust_created_database_has_python_legacy_schema_and_sql_semantics() {
    let (store, path) = store("schema");
    store.next_global_seq().unwrap();
    store.next_user_seq("alice").unwrap();
    store
        .append_raw_entity_event("default", raw_event("event-1", "{\"x\":1}"))
        .unwrap();
    drop(store);

    let conn = Connection::open(&path).unwrap();
    let tables: Vec<String> = conn
        .prepare("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")
        .unwrap()
        .query_map([], |row| row.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(
        tables,
        vec![
            "entity_events",
            "global_seq",
            "index_applied_state",
            "index_jobs",
            "named_projections",
            "namespace_seq",
            "projected_lane_messages",
            "replay_cursors",
            "server_run_events",
            "server_runs",
            "sqlite_sequence",
            "stage1_node_projections",
            "user_seq",
            "workflow_design_snapshots",
            "workflow_design_version_deltas",
        ]
    );
    let global: i64 = conn
        .query_row("SELECT value FROM global_seq WHERE rowid = 1", [], |row| {
            row.get(0)
        })
        .unwrap();
    let user: i64 = conn
        .query_row(
            "SELECT value FROM user_seq WHERE user_id = 'alice'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    let event: (String, i64, String) = conn
        .query_row(
            "SELECT namespace, seq, payload_json FROM entity_events WHERE event_id = 'event-1'",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .unwrap();
    assert_eq!(
        (global, user, event),
        (1, 1, ("default".to_owned(), 1, "{\"x\":1}".to_owned()))
    );
    drop(conn);
    fs::remove_file(path).unwrap();
}

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

#[test]
fn global_user_and_scoped_sequences_are_isolated_and_current() {
    let (store, path) = store("sequences");
    assert_eq!(store.current_global_seq().unwrap(), 0);
    assert_eq!(store.next_global_seq().unwrap(), 1);
    assert_eq!(store.next_global_seq().unwrap(), 2);
    assert_eq!(store.current_user_seq("alice").unwrap(), 0);
    assert_eq!(store.next_user_seq("alice").unwrap(), 1);
    assert_eq!(store.next_user_seq("alice").unwrap(), 2);
    assert_eq!(store.next_scoped_seq("project-7").unwrap(), 1);
    assert_eq!(store.current_scoped_seq("project-7").unwrap(), 1);
    assert_eq!(store.current_user_seq("alice").unwrap(), 2);
    store.set_scoped_seq("project-7", 11).unwrap();
    assert_eq!(store.next_user_seq("project-7").unwrap(), 12);
    assert!(matches!(
        store.set_user_seq("alice", -1),
        Err(SqliteStoreError::NegativeSequenceValue { value: -1 })
    ));
    drop(store);
    fs::remove_file(path).unwrap();
}

#[test]
fn append_replay_idempotency_and_cross_namespace_collision_follow_contract() {
    let (store, path) = store("events");
    let first = block_on(EventWriteStore::append_entity_event(
        &store,
        "one",
        event("same", "one"),
    ))
    .unwrap();
    let second = block_on(EventWriteStore::append_entity_event(
        &store,
        "one",
        event("second", "two"),
    ))
    .unwrap();
    let retry = block_on(EventWriteStore::append_entity_event(
        &store,
        "one",
        event("same", "changed"),
    ))
    .unwrap();
    assert_eq!((first.event.seq, second.event.seq), (1, 2));
    assert!(first.inserted);
    assert!(!retry.inserted);
    assert_eq!(retry.event.entity_id, "one");
    assert_eq!(
        block_on(EventReadStore::replay_events(&store, "one", 1, 10))
            .unwrap()
            .iter()
            .map(|event| event.seq)
            .collect::<Vec<_>>(),
        [2]
    );
    assert!(matches!(
        store.append_raw_entity_event("two", raw_event("same", "{}")),
        Err(SqliteStoreError::EventIdNamespaceCollision { ref existing_namespace, .. }) if existing_namespace == "one"
    ));
    assert_eq!(
        block_on(EventWriteStore::append_entity_event(
            &store,
            "two",
            event("same", "changed"),
        )),
        Err(StoreError::EventIdNamespaceCollision {
            event_id: "same".to_owned(),
            existing_namespace: "one".to_owned(),
            requested_namespace: "two".to_owned(),
        })
    );
    assert_eq!(store.alloc_event_seq("gap").unwrap(), 1);
    assert_eq!(
        block_on(EventReadStore::latest_event_seq(&store, "gap")).unwrap(),
        0
    );
    drop(store);
    fs::remove_file(path).unwrap();
}

#[test]
fn trait_surfaces_sqlite_failures_as_backend_errors() {
    let (store, path) = store("trait-backend-error");
    let conn = Connection::open(&path).unwrap();
    conn.execute("DROP TABLE entity_events", []).unwrap();
    drop(conn);

    assert!(matches!(
        block_on(EventReadStore::replay_events(&store, "ns", 0, 10)),
        Err(StoreError::Backend { ref backend, ref message })
            if backend == "sqlite" && message.contains("entity_events")
    ));
    drop(store);
    fs::remove_file(path).unwrap();
}

#[test]
fn cursor_trait_is_strict_while_legacy_set_is_permissive() {
    let (store, path) = store("cursors");
    block_on(EventWriteStore::append_entity_event(
        &store,
        "ns",
        event("one", "one"),
    ))
    .unwrap();
    assert_eq!(
        block_on(EventWriteStore::advance_replay_cursor(
            &store, "ns", "sink", 1
        ))
        .unwrap()
        .last_seq,
        1
    );
    assert_eq!(
        block_on(EventWriteStore::advance_replay_cursor(
            &store, "ns", "sink", 0
        )),
        Err(StoreError::CursorRegresses {
            current: 1,
            requested: 0
        })
    );
    assert_eq!(
        block_on(EventWriteStore::advance_replay_cursor(
            &store, "ns", "sink", 2
        )),
        Err(StoreError::CursorOutOfRange {
            cursor: 2,
            latest: 1
        })
    );
    assert_eq!(
        store
            .set_replay_cursor_legacy("ns", "sink", 99)
            .unwrap()
            .last_seq,
        99
    );
    assert_eq!(
        store
            .set_replay_cursor_legacy("ns", "sink", -7)
            .unwrap()
            .last_seq,
        -7
    );
    drop(store);
    fs::remove_file(path).unwrap();
}

#[test]
fn immediate_uow_commits_and_rolls_back_sequences_events_and_cursors_together() {
    let (store, path) = store("uow");
    store
        .immediate_transaction(|uow| {
            assert_eq!(uow.next_global_seq()?, 1);
            assert_eq!(
                uow.append_raw_entity_event("ns", raw_event("committed", "{}"))?
                    .event
                    .seq,
                1
            );
            uow.strict_advance_replay_cursor("ns", "sink", 1)?;
            Ok(())
        })
        .unwrap();
    let rollback: SqliteStoreResult<()> = store.immediate_transaction(|uow| {
        assert_eq!(uow.next_global_seq()?, 2);
        assert_eq!(
            uow.append_raw_entity_event("ns", raw_event("rolled-back", "{}"))?
                .event
                .seq,
            2
        );
        uow.strict_advance_replay_cursor("ns", "sink", 2)?;
        Err(SqliteStoreError::TransactionAborted(
            "test rollback".to_owned(),
        ))
    });
    assert!(matches!(
        rollback,
        Err(SqliteStoreError::TransactionAborted(_))
    ));
    assert_eq!(store.current_global_seq().unwrap(), 1);
    assert_eq!(store.latest_retained_event_seq("ns").unwrap(), 1);
    assert_eq!(store.replay_cursor("ns", "sink").unwrap().last_seq, 1);
    assert!(
        store
            .replay_raw_events("ns", 0, 10)
            .unwrap()
            .iter()
            .all(|event| event.event_id != "rolled-back")
    );
    drop(store);
    fs::remove_file(path).unwrap();
}

#[test]
fn external_transaction_joins_operations_and_commits_or_rolls_back() {
    let (store, path) = store("external-uow");
    store.begin_external_transaction().unwrap();
    assert_eq!(store.next_global_seq().unwrap(), 1);
    store
        .append_raw_entity_event("ns", raw_event("commit", "{}"))
        .unwrap();
    store.commit_external_transaction().unwrap();
    assert_eq!(store.current_global_seq().unwrap(), 1);
    assert_eq!(store.replay_raw_events("ns", 0, 10).unwrap().len(), 1);

    store.begin_external_transaction().unwrap();
    assert_eq!(store.next_global_seq().unwrap(), 2);
    store
        .append_raw_entity_event("ns", raw_event("rollback", "{}"))
        .unwrap();
    store.rollback_external_transaction().unwrap();
    assert_eq!(store.current_global_seq().unwrap(), 1);
    assert_eq!(store.replay_raw_events("ns", 0, 10).unwrap().len(), 1);

    drop(store);
    fs::remove_file(path).unwrap();
}

#[test]
fn reopen_is_durable_and_concurrent_global_allocations_are_serialized() {
    let (store, path) = store("durable");
    store.next_global_seq().unwrap();
    store
        .append_raw_entity_event("ns", raw_event("durable", "{\"kept\":true}"))
        .unwrap();
    drop(store);
    let reopened = SqliteStore::open(&path).unwrap();
    assert_eq!(reopened.current_global_seq().unwrap(), 1);
    assert_eq!(
        reopened.replay_raw_events("ns", 0, 1).unwrap()[0].payload_json,
        "{\"kept\":true}"
    );

    let workers = 6;
    let barrier = Arc::new(Barrier::new(workers));
    let mut handles = Vec::new();
    for _ in 0..workers {
        let store = reopened.clone();
        let barrier = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            barrier.wait();
            store.next_global_seq().unwrap()
        }));
    }
    let mut issued = handles
        .into_iter()
        .map(|handle| handle.join().unwrap())
        .collect::<Vec<_>>();
    issued.sort_unstable();
    assert_eq!(issued, [2, 3, 4, 5, 6, 7]);
    drop(reopened);
    fs::remove_file(path).unwrap();
}

#[test]
fn arbitrary_precision_event_payload_survives_reopen_and_typed_replay() {
    let (store, path) = store("arbitrary-precision-event");
    let integer = "1606938044258990275541962092341162602522202993782792835301376";
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../contracts/golden/python-graph-model-payloads-v1.json"
    ))
    .unwrap();
    let payload = fixture["cases"][0]["canonical_json"]
        .as_str()
        .expect("shared Python node fixture")
        .replace(
            "\"level_from_root\":0",
            &format!("\"level_from_root\":{integer}"),
        );
    store
        .append_raw_entity_event("ns", raw_event("large-level", &payload))
        .unwrap();
    drop(store);

    let reopened = SqliteStore::open(&path).unwrap();
    let raw = reopened.replay_raw_events("ns", 0, 1).unwrap();
    assert_eq!(raw[0].payload_json, payload);

    let replayed = block_on(EventReadStore::replay_events(&reopened, "ns", 0, 1)).unwrap();
    assert_eq!(
        serde_json::to_string(&replayed[0].payload).unwrap(),
        payload
    );
    let envelope_wire =
        format!(r#"{{"schema_version":1,"model_class":"node","payload":{payload}}}"#);
    let decoded = kogwistar_contracts::RawGraphModelEnvelopeV1::from_json(&envelope_wire)
        .unwrap()
        .decode_persistent_v2()
        .expect("replayed Python-shaped node retains its exact typed hierarchy level");
    assert_eq!(
        decoded
            .level_from_root
            .as_ref()
            .map(|level| level.as_decimal_str()),
        Some(integer)
    );
    drop(reopened);
    fs::remove_file(path).unwrap();
}

#[test]
fn accepted_index_job_result_is_first_live_claim_wins() {
    let (store, path) = store("accepted-result");
    store
        .enqueue_index_job(NewIndexJob {
            job_id: "candidate-job".to_owned(),
            namespace: "maintenance".to_owned(),
            entity_kind: "maintenance".to_owned(),
            entity_id: "doc-1".to_owned(),
            index_kind: "parse".to_owned(),
            op: "UPSERT".to_owned(),
            payload_json: Some("{}".to_owned()),
            max_retries: 3,
        })
        .unwrap();
    let claim = store
        .claim_index_jobs(1, 30, Some("maintenance"))
        .unwrap()
        .remove(0);
    let accepted = store
        .accept_index_job_result(
            &claim.job_id,
            claim.claim_token.as_deref().unwrap(),
            "{\"winner\":1}",
            "digest-1",
        )
        .unwrap();
    assert_eq!(accepted.status, "accepted");
    let existing = store
        .accept_index_job_result(&claim.job_id, "stale-worker", "{\"winner\":2}", "digest-2")
        .unwrap();
    assert_eq!(existing.status, "existing");
    assert_eq!(existing.result_json.as_deref(), Some("{\"winner\":1}"));
    drop(store);
    fs::remove_file(path).unwrap();
}
