use super::*;

/// Strict ASCII PostgreSQL identifier validation, then always quote it.
pub fn quote_identifier(identifier: &str) -> PostgresStoreResult<String> {
    let mut bytes = identifier.bytes();
    let Some(first) = bytes.next() else {
        return Err(PostgresStoreError::InvalidSchema {
            schema: identifier.to_owned(),
        });
    };
    if !(first.is_ascii_alphabetic() || first == b'_')
        || !bytes.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
    {
        return Err(PostgresStoreError::InvalidSchema {
            schema: identifier.to_owned(),
        });
    }
    Ok(format!("\"{identifier}\""))
}

pub(super) fn qualified(schema: &str, relation: &str) -> PostgresStoreResult<String> {
    Ok(format!("{schema}.{}", quote_identifier(relation)?))
}

pub(super) fn schema_sql(tables: &Tables) -> String {
    format!(
        "CREATE SCHEMA IF NOT EXISTS {schema};\
         CREATE TABLE IF NOT EXISTS {global_seq} (value BIGINT NOT NULL);\
         INSERT INTO {global_seq}(value) SELECT 0 WHERE NOT EXISTS (SELECT 1 FROM {global_seq});\
         CREATE TABLE IF NOT EXISTS {user_seq} (user_id TEXT PRIMARY KEY, value BIGINT NOT NULL);\
         CREATE TABLE IF NOT EXISTS {index_applied_state} (\
            namespace TEXT NOT NULL DEFAULT 'default',coalesce_key TEXT NOT NULL,\
            applied_fingerprint TEXT NULL,applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),\
            last_job_id TEXT NULL,PRIMARY KEY(namespace, coalesce_key)\
         );\
         CREATE INDEX IF NOT EXISTS {index_applied_state_key_index}\
            ON {index_applied_state}(coalesce_key);\
         CREATE TABLE IF NOT EXISTS {namespace_seq} (namespace TEXT PRIMARY KEY, next_seq BIGINT NOT NULL);\
         INSERT INTO {namespace_seq}(namespace, next_seq) VALUES ('default', 1) ON CONFLICT(namespace) DO NOTHING;\
         CREATE TABLE IF NOT EXISTS {entity_events} (\
            namespace TEXT NOT NULL DEFAULT 'default',seq BIGINT NOT NULL,event_id TEXT NOT NULL,\
            entity_kind TEXT NOT NULL,entity_id TEXT NOT NULL,op TEXT NOT NULL,payload_json TEXT NOT NULL,\
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),PRIMARY KEY(namespace, seq),UNIQUE(event_id)\
         );\
         CREATE INDEX IF NOT EXISTS {aggregate_index} ON {entity_events}(namespace, entity_kind, entity_id, seq);\
         CREATE TABLE IF NOT EXISTS {replay_cursors} (\
            namespace TEXT NOT NULL DEFAULT 'default',consumer TEXT NOT NULL,last_seq BIGINT NOT NULL,\
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),PRIMARY KEY(namespace, consumer)\
         );\
         CREATE TABLE IF NOT EXISTS {named_projections} (\
            namespace TEXT NOT NULL,key TEXT NOT NULL,payload_json TEXT NOT NULL,\
            last_authoritative_seq BIGINT NOT NULL,last_materialized_seq BIGINT NOT NULL,\
            projection_schema_version INTEGER NOT NULL,materialization_status TEXT NOT NULL,\
            updated_at_ms BIGINT NOT NULL,PRIMARY KEY(namespace, key)\
         );\
         CREATE INDEX IF NOT EXISTS {named_projections_namespace_index} ON {named_projections}(namespace, updated_at_ms);\
         CREATE TABLE IF NOT EXISTS {workflow_design_snapshots} (\
            workflow_id TEXT NOT NULL,version BIGINT NOT NULL,seq BIGINT NOT NULL,payload_json TEXT NOT NULL,\
            schema_version INTEGER NOT NULL,created_at_ms BIGINT NOT NULL,PRIMARY KEY(workflow_id, version)\
         );\
         CREATE TABLE IF NOT EXISTS {workflow_design_version_deltas} (\
            workflow_id TEXT NOT NULL,version BIGINT NOT NULL,prev_version BIGINT NOT NULL,target_seq BIGINT NOT NULL,\
            forward_json TEXT NOT NULL,inverse_json TEXT NOT NULL,schema_version INTEGER NOT NULL,\
            created_at_ms BIGINT NOT NULL,PRIMARY KEY(workflow_id, version)\
         );\
         CREATE TABLE IF NOT EXISTS {server_runs} (\
            run_id TEXT PRIMARY KEY,conversation_id TEXT NOT NULL,workflow_id TEXT NOT NULL,user_id TEXT NULL,\
            user_turn_node_id TEXT NULL,assistant_turn_node_id TEXT NULL,status TEXT NOT NULL,\
            cancel_requested INTEGER NOT NULL DEFAULT 0,result_json TEXT NULL,error_json TEXT NULL,\
            created_at_ms BIGINT NOT NULL,updated_at_ms BIGINT NOT NULL,started_at_ms BIGINT NULL,finished_at_ms BIGINT NULL\
         );\
         CREATE TABLE IF NOT EXISTS {server_run_events} (\
            seq BIGSERIAL PRIMARY KEY,run_id TEXT NOT NULL,event_type TEXT NOT NULL,\
            payload_json TEXT NOT NULL,created_at_ms BIGINT NOT NULL\
         );\
         CREATE INDEX IF NOT EXISTS {server_runs_status_index} ON {server_runs}(status, updated_at_ms);\
         CREATE INDEX IF NOT EXISTS {server_run_events_run_seq_index} ON {server_run_events}(run_id, seq);\
         CREATE UNIQUE INDEX IF NOT EXISTS {server_run_events_transition_id_index}\
            ON {server_run_events} ((payload_json::jsonb->>'transition_id')) WHERE event_type='workflow.recorded_transition.v1';\
         CREATE TABLE IF NOT EXISTS {index_jobs} (\
            job_id TEXT PRIMARY KEY,namespace TEXT NOT NULL DEFAULT 'default',entity_kind TEXT NOT NULL,entity_id TEXT NOT NULL,\
            index_kind TEXT NOT NULL,coalesce_key TEXT NOT NULL,op TEXT NOT NULL,status TEXT NOT NULL,lease_until TIMESTAMPTZ NULL,\
            next_run_at TIMESTAMPTZ NULL,max_retries INTEGER NOT NULL DEFAULT 10,retry_count INTEGER NOT NULL DEFAULT 0,\
            last_error TEXT NULL,payload_json TEXT NULL,created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),\
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),claim_token TEXT NULL,claim_attempts INTEGER NOT NULL DEFAULT 0,\
            accepted_result_json TEXT NULL,accepted_result_sha256 TEXT NULL,accepted_at TIMESTAMPTZ NULL\
         );\
         ALTER TABLE {index_jobs} ADD COLUMN IF NOT EXISTS claim_token TEXT NULL;\
         ALTER TABLE {index_jobs} ADD COLUMN IF NOT EXISTS claim_attempts INTEGER NOT NULL DEFAULT 0;\
         ALTER TABLE {index_jobs} ADD COLUMN IF NOT EXISTS accepted_result_json TEXT NULL;\
         ALTER TABLE {index_jobs} ADD COLUMN IF NOT EXISTS accepted_result_sha256 TEXT NULL;\
         ALTER TABLE {index_jobs} ADD COLUMN IF NOT EXISTS accepted_at TIMESTAMPTZ NULL;\
         CREATE INDEX IF NOT EXISTS {index_jobs_status_lease_index} ON {index_jobs}(status, lease_until);\
         CREATE INDEX IF NOT EXISTS {index_jobs_status_next_run_index} ON {index_jobs}(status, next_run_at);\
         CREATE INDEX IF NOT EXISTS {index_jobs_entity_index} ON {index_jobs}(entity_kind, entity_id, index_kind);\
         CREATE INDEX IF NOT EXISTS {index_jobs_namespace_index} ON {index_jobs}(namespace);\
         CREATE UNIQUE INDEX IF NOT EXISTS {index_jobs_pending_index} ON {index_jobs}(namespace, coalesce_key) WHERE status='PENDING';\
         CREATE TABLE IF NOT EXISTS {projected_lane_messages} (\
            message_id TEXT PRIMARY KEY,namespace TEXT NOT NULL DEFAULT 'default',purpose TEXT NOT NULL DEFAULT 'user_visible',\
            inbox_id TEXT NOT NULL,conversation_id TEXT NOT NULL,recipient_id TEXT NOT NULL,sender_id TEXT NOT NULL,\
            msg_type TEXT NOT NULL,status TEXT NOT NULL,seq BIGINT NOT NULL,conversation_seq BIGINT NOT NULL,\
            claimed_by TEXT NULL,lease_until TIMESTAMPTZ NULL,retry_count INTEGER NOT NULL DEFAULT 0,created_at BIGINT NOT NULL,\
            available_at BIGINT NOT NULL,run_id TEXT NULL,step_id TEXT NULL,correlation_id TEXT NULL,payload_json TEXT NULL,error_json TEXT NULL,\
            prev_message_id TEXT NULL,next_message_id TEXT NULL,inbox_tail_message_id TEXT NULL,conversation_tail_message_id TEXT NULL\
         );\
         CREATE INDEX IF NOT EXISTS {lane_messages_namespace_inbox_seq_index} ON {projected_lane_messages}(namespace,inbox_id,seq);\
         CREATE INDEX IF NOT EXISTS {lane_messages_claim_index} ON {projected_lane_messages}(namespace,inbox_id,status,available_at,lease_until);\
         CREATE INDEX IF NOT EXISTS {lane_messages_conversation_seq_index} ON {projected_lane_messages}(namespace,conversation_id,conversation_seq);",
        schema = tables.quoted_schema,
        global_seq = tables.global_seq,
        user_seq = tables.user_seq,
        index_applied_state = tables.index_applied_state,
        index_applied_state_key_index = tables.index_applied_state_key_index,
        namespace_seq = tables.namespace_seq,
        entity_events = tables.entity_events,
        aggregate_index = tables.aggregate_index,
        replay_cursors = tables.replay_cursors,
        named_projections = tables.named_projections,
        named_projections_namespace_index = tables.named_projections_namespace_index,
        workflow_design_snapshots = tables.workflow_design_snapshots,
        workflow_design_version_deltas = tables.workflow_design_version_deltas,
        server_runs = tables.server_runs,
        server_run_events = tables.server_run_events,
        server_runs_status_index = tables.server_runs_status_index,
        server_run_events_run_seq_index = tables.server_run_events_run_seq_index,
        server_run_events_transition_id_index = tables.server_run_events_transition_id_index,
        index_jobs = tables.index_jobs,
        index_jobs_status_lease_index = tables.index_jobs_status_lease_index,
        index_jobs_status_next_run_index = tables.index_jobs_status_next_run_index,
        index_jobs_entity_index = tables.index_jobs_entity_index,
        index_jobs_namespace_index = tables.index_jobs_namespace_index,
        index_jobs_pending_index = tables.index_jobs_pending_index,
        projected_lane_messages = tables.projected_lane_messages,
        lane_messages_namespace_inbox_seq_index = tables.lane_messages_namespace_inbox_seq_index,
        lane_messages_claim_index = tables.lane_messages_claim_index,
        lane_messages_conversation_seq_index = tables.lane_messages_conversation_seq_index,
    )
}
