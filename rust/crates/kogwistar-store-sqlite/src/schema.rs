use super::*;

pub(super) fn initialize_schema(conn: &Connection) -> SqliteStoreResult<()> {
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS global_seq (
            value INTEGER NOT NULL
        );
        INSERT OR IGNORE INTO global_seq(rowid, value) VALUES (1, 0);
        CREATE TABLE IF NOT EXISTS user_seq (
            user_id TEXT PRIMARY KEY,
            value   INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS namespace_seq (
            namespace TEXT PRIMARY KEY,
            next_seq   INTEGER NOT NULL
        );
        INSERT OR IGNORE INTO namespace_seq(namespace, next_seq) VALUES ('default', 1);
        CREATE TABLE IF NOT EXISTS entity_events (
            namespace    TEXT NOT NULL DEFAULT 'default',
            seq          INTEGER NOT NULL,
            event_id     TEXT NOT NULL,
            entity_kind  TEXT NOT NULL,
            entity_id    TEXT NOT NULL,
            op           TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at   INTEGER NOT NULL,
            PRIMARY KEY(namespace, seq),
            UNIQUE(event_id)
        );
        CREATE INDEX IF NOT EXISTS idx_entity_events_aggregate
        ON entity_events(namespace, entity_kind, entity_id, seq);
        CREATE TABLE IF NOT EXISTS replay_cursors (
            namespace  TEXT NOT NULL DEFAULT 'default',
            consumer   TEXT NOT NULL,
            last_seq   INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            PRIMARY KEY(namespace, consumer)
        );
        CREATE TABLE IF NOT EXISTS named_projections (
            namespace TEXT NOT NULL,
            key TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            last_authoritative_seq INTEGER NOT NULL,
            last_materialized_seq INTEGER NOT NULL,
            projection_schema_version INTEGER NOT NULL,
            materialization_status TEXT NOT NULL,
            updated_at_ms INTEGER NOT NULL,
            PRIMARY KEY(namespace, key)
        );
        CREATE INDEX IF NOT EXISTS idx_named_projections_namespace
        ON named_projections(namespace, updated_at_ms);
        CREATE TABLE IF NOT EXISTS stage1_node_projections (
            namespace TEXT NOT NULL,
            key TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            last_authoritative_seq INTEGER NOT NULL,
            last_materialized_seq INTEGER NOT NULL,
            projection_schema_version INTEGER NOT NULL,
            materialization_status TEXT NOT NULL,
            updated_at_ms INTEGER NOT NULL,
            PRIMARY KEY(namespace, key)
        );
        CREATE INDEX IF NOT EXISTS idx_stage1_node_projections_namespace
        ON stage1_node_projections(namespace, updated_at_ms);
        CREATE TABLE IF NOT EXISTS workflow_design_snapshots (
            workflow_id TEXT NOT NULL,
            version INTEGER NOT NULL,
            seq INTEGER NOT NULL,
            payload_json TEXT NOT NULL,
            schema_version INTEGER NOT NULL,
            created_at_ms INTEGER NOT NULL,
            PRIMARY KEY(workflow_id, version)
        );
        CREATE TABLE IF NOT EXISTS workflow_design_version_deltas (
            workflow_id TEXT NOT NULL,
            version INTEGER NOT NULL,
            prev_version INTEGER NOT NULL,
            target_seq INTEGER NOT NULL,
            forward_json TEXT NOT NULL,
            inverse_json TEXT NOT NULL,
            schema_version INTEGER NOT NULL,
            created_at_ms INTEGER NOT NULL,
            PRIMARY KEY(workflow_id, version)
        );
        CREATE TABLE IF NOT EXISTS server_runs (
            run_id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            workflow_id TEXT NOT NULL,
            user_id TEXT,
            user_turn_node_id TEXT,
            assistant_turn_node_id TEXT,
            status TEXT NOT NULL,
            cancel_requested INTEGER NOT NULL DEFAULT 0,
            result_json TEXT,
            error_json TEXT,
            created_at_ms INTEGER NOT NULL,
            updated_at_ms INTEGER NOT NULL,
            started_at_ms INTEGER,
            finished_at_ms INTEGER
        );
        CREATE TABLE IF NOT EXISTS server_run_events (
            seq INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at_ms INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_server_runs_status ON server_runs(status, updated_at_ms);
        CREATE INDEX IF NOT EXISTS idx_server_run_events_run_seq ON server_run_events(run_id, seq);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_server_run_events_recorded_transition_id
        ON server_run_events(json_extract(payload_json, '$.transition_id'))
        WHERE event_type = 'workflow.recorded_transition.v1';
        CREATE TABLE IF NOT EXISTS index_jobs (
            job_id TEXT PRIMARY KEY,
            namespace TEXT NOT NULL DEFAULT 'default',
            entity_kind TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            index_kind TEXT NOT NULL,
            coalesce_key TEXT NOT NULL,
            op TEXT NOT NULL,
            status TEXT NOT NULL,
            lease_until INTEGER,
            next_run_at INTEGER,
            max_retries INTEGER NOT NULL DEFAULT 10,
            retry_count INTEGER NOT NULL DEFAULT 0,
            last_error TEXT,
            payload_json TEXT,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            claim_token TEXT,
            accepted_result_json TEXT,
            accepted_result_sha256 TEXT,
            accepted_at INTEGER
        );
        CREATE TABLE IF NOT EXISTS index_applied_state (
            namespace TEXT NOT NULL DEFAULT 'default',
            coalesce_key TEXT NOT NULL,
            applied_fingerprint TEXT,
            applied_at INTEGER NOT NULL,
            last_job_id TEXT,
            PRIMARY KEY(namespace, coalesce_key)
        );
        CREATE INDEX IF NOT EXISTS idx_index_applied_state_key
        ON index_applied_state(coalesce_key);
        CREATE TABLE IF NOT EXISTS projected_lane_messages (
            message_id TEXT PRIMARY KEY, namespace TEXT NOT NULL DEFAULT 'default', purpose TEXT NOT NULL DEFAULT 'user_visible',
            inbox_id TEXT NOT NULL, conversation_id TEXT NOT NULL, recipient_id TEXT NOT NULL, sender_id TEXT NOT NULL,
            msg_type TEXT NOT NULL, status TEXT NOT NULL, seq INTEGER NOT NULL, conversation_seq INTEGER NOT NULL,
            claimed_by TEXT, lease_until INTEGER, retry_count INTEGER NOT NULL DEFAULT 0, created_at INTEGER NOT NULL,
            available_at INTEGER NOT NULL, run_id TEXT, step_id TEXT, correlation_id TEXT, payload_json TEXT, error_json TEXT,
            prev_message_id TEXT, next_message_id TEXT, inbox_tail_message_id TEXT, conversation_tail_message_id TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_lane_messages_namespace_inbox_seq ON projected_lane_messages(namespace, inbox_id, seq);
        CREATE INDEX IF NOT EXISTS idx_lane_messages_claim ON projected_lane_messages(namespace, inbox_id, status, available_at, lease_until);
        CREATE INDEX IF NOT EXISTS idx_lane_messages_conversation_seq ON projected_lane_messages(namespace, conversation_id, conversation_seq);
        ",
    )?;
    // Python-created databases may predate queue coalescing, scheduling, and
    // lease ownership. Add columns before creating indexes that reference
    // them so every historical Python schema remains openable by Rust.
    let index_job_columns = conn
        .prepare("PRAGMA table_info(index_jobs)")?
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<Result<Vec<_>, _>>()?;
    if !index_job_columns.iter().any(|column| column == "namespace") {
        conn.execute(
            "ALTER TABLE index_jobs ADD COLUMN namespace TEXT NOT NULL DEFAULT 'default'",
            [],
        )?;
    }
    if !index_job_columns
        .iter()
        .any(|column| column == "coalesce_key")
    {
        conn.execute(
            "ALTER TABLE index_jobs ADD COLUMN coalesce_key TEXT NOT NULL DEFAULT ''",
            [],
        )?;
    }
    if !index_job_columns
        .iter()
        .any(|column| column == "next_run_at")
    {
        conn.execute("ALTER TABLE index_jobs ADD COLUMN next_run_at INTEGER", [])?;
    }
    if !index_job_columns
        .iter()
        .any(|column| column == "max_retries")
    {
        conn.execute(
            "ALTER TABLE index_jobs ADD COLUMN max_retries INTEGER NOT NULL DEFAULT 10",
            [],
        )?;
    }
    if !index_job_columns
        .iter()
        .any(|column| column == "claim_token")
    {
        conn.execute("ALTER TABLE index_jobs ADD COLUMN claim_token TEXT", [])?;
    }
    if !index_job_columns
        .iter()
        .any(|column| column == "accepted_result_json")
    {
        conn.execute(
            "ALTER TABLE index_jobs ADD COLUMN accepted_result_json TEXT",
            [],
        )?;
    }
    if !index_job_columns
        .iter()
        .any(|column| column == "accepted_result_sha256")
    {
        conn.execute(
            "ALTER TABLE index_jobs ADD COLUMN accepted_result_sha256 TEXT",
            [],
        )?;
    }
    if !index_job_columns
        .iter()
        .any(|column| column == "accepted_at")
    {
        conn.execute("ALTER TABLE index_jobs ADD COLUMN accepted_at INTEGER", [])?;
    }
    let lane_columns = conn
        .prepare("PRAGMA table_info(projected_lane_messages)")?
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<Result<Vec<_>, _>>()?;
    if !lane_columns.iter().any(|column| column == "purpose") {
        conn.execute(
            "ALTER TABLE projected_lane_messages ADD COLUMN purpose TEXT NOT NULL DEFAULT 'user_visible'",
            [],
        )?;
    }
    conn.execute_batch(
        "CREATE INDEX IF NOT EXISTS idx_index_jobs_status_lease ON index_jobs(status, lease_until);
         CREATE INDEX IF NOT EXISTS idx_index_jobs_entity ON index_jobs(entity_kind, entity_id, index_kind);
         CREATE INDEX IF NOT EXISTS idx_index_jobs_namespace ON index_jobs(namespace);
         CREATE UNIQUE INDEX IF NOT EXISTS uq_index_jobs_pending_ns_coalesce
         ON index_jobs(namespace, coalesce_key) WHERE status='PENDING';",
    )?;
    Ok(())
}
