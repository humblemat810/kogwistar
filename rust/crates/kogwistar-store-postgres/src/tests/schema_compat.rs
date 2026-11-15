// Domain-focused tests; shared fixtures live in the parent test module.
use super::*;

#[test]
fn identifier_validation_and_qualification_are_strict() {
    assert_eq!(quote_identifier("Mixed_Name9").unwrap(), "\"Mixed_Name9\"");
    for invalid in ["", "9name", "has-dash", "has space", "public;DROP"] {
        assert!(matches!(
            quote_identifier(invalid),
            Err(PostgresStoreError::InvalidSchema { .. })
        ));
    }
    let tables = Tables::new("CamelCase").unwrap();
    assert_eq!(tables.entity_events, "\"CamelCase\".\"entity_events\"");
    assert!(!tables.entity_events.contains("search_path"));
}
#[test]
fn sql_objects_are_schema_qualified_and_python_shaped() {
    let tables = Tables::new("events").unwrap();
    assert_eq!(tables.namespace_seq, "\"events\".\"namespace_seq\"");
    assert_eq!(tables.replay_cursors, "\"events\".\"replay_cursors\"");
    assert_eq!(tables.aggregate_index, "\"idx_entity_events_aggregate\"");
    assert_eq!(cursor_lock_key("a", "b"), "a\u{1f}b");
    let sql = schema_sql(&tables);
    assert!(sql.contains("CREATE SCHEMA IF NOT EXISTS \"events\""));
    assert!(sql.contains("CREATE TABLE IF NOT EXISTS \"events\".\"namespace_seq\""));
    assert!(sql.contains("event_id TEXT NOT NULL"));
    assert!(sql.contains("UNIQUE(event_id)"));
    assert!(sql.contains("created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()"));
    assert!(sql.contains("PRIMARY KEY(namespace, consumer)"));
    assert!(!sql.contains("search_path"));
    assert_eq!(
        allocation_sql(&tables),
        "INSERT INTO \"events\".\"namespace_seq\" AS target(namespace, next_seq) VALUES ($1, 2) ON CONFLICT(namespace) DO UPDATE SET next_seq = target.next_seq + 1 RETURNING next_seq - 1"
    );
}
#[tokio::test]
#[ignore = "requires KOGWISTAR_TEST_PG_DSN; run explicitly with --include-ignored"]
async fn python_created_schema_is_read_and_written_when_dsn_available() -> PostgresStoreResult<()> {
    let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
        .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
    let schema = test_schema("python");
    let store = PostgresStore::from_dsn(&dsn, &schema)?;
    let tables = Tables::new(&schema)?;
    let client = store.client().await?;
    client
            .batch_execute(&format!(
                "CREATE SCHEMA {schema};\
                 CREATE TABLE {ns}(namespace TEXT PRIMARY KEY, next_seq BIGINT NOT NULL);\
                 INSERT INTO {ns} VALUES ('legacy', 7);\
                 CREATE TABLE {events}(namespace TEXT NOT NULL DEFAULT 'default', seq BIGINT NOT NULL, event_id TEXT NOT NULL, entity_kind TEXT NOT NULL, entity_id TEXT NOT NULL, op TEXT NOT NULL, payload_json TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), PRIMARY KEY(namespace, seq), UNIQUE(event_id));\
                 INSERT INTO {events}(namespace, seq, event_id, entity_kind, entity_id, op, payload_json) VALUES ('legacy', 6, 'old', 'node', 'old', 'UPSERT', '{{\"original\": true }}');\
                 CREATE TABLE {cursors}(namespace TEXT NOT NULL DEFAULT 'default', consumer TEXT NOT NULL, last_seq BIGINT NOT NULL, updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), PRIMARY KEY(namespace, consumer));\
                 INSERT INTO {cursors}(namespace, consumer, last_seq) VALUES ('legacy', 'reader', 6);",
                schema = tables.quoted_schema,
                ns = tables.namespace_seq,
                events = tables.entity_events,
                cursors = tables.replay_cursors,
            ))
            .await
            .map_err(backend)?;
    drop(client);
    let row = store
        .append_raw_entity_event("legacy", raw("new", "{\"kept\": true }"))
        .await?;
    assert_eq!(row.event.seq, 7);
    assert_eq!(row.event.payload_json, "{\"kept\": true }");
    assert_eq!(store.replay_cursor("legacy", "reader").await?.last_seq, 6);
    let cleaner = store.client().await?;
    cleaner
        .batch_execute(&format!("DROP SCHEMA {} CASCADE", tables.quoted_schema))
        .await
        .map_err(backend)
}
