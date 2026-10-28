// Domain-focused tests; shared fixtures live in the parent test module.
use super::*;

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
