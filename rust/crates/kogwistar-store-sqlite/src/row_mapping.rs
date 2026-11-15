use super::*;

pub(super) fn workflow_design_snapshot_from_row(
    row: &rusqlite::Row<'_>,
) -> rusqlite::Result<WorkflowDesignSnapshot> {
    Ok(WorkflowDesignSnapshot {
        workflow_id: row.get(0)?,
        version: row.get(1)?,
        seq: row.get(2)?,
        payload_json: row.get(3)?,
        schema_version: row.get(4)?,
        created_at_ms: row.get(5)?,
    })
}

pub(super) fn workflow_design_delta_from_row(
    row: &rusqlite::Row<'_>,
) -> rusqlite::Result<WorkflowDesignDelta> {
    Ok(WorkflowDesignDelta {
        workflow_id: row.get(0)?,
        version: row.get(1)?,
        prev_version: row.get(2)?,
        target_seq: row.get(3)?,
        forward_json: row.get(4)?,
        inverse_json: row.get(5)?,
        schema_version: row.get(6)?,
        created_at_ms: row.get(7)?,
    })
}

pub(super) fn server_run_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<ServerRun> {
    Ok(ServerRun {
        run_id: row.get(0)?,
        conversation_id: row.get(1)?,
        workflow_id: row.get(2)?,
        user_id: row.get(3)?,
        user_turn_node_id: row.get(4)?,
        assistant_turn_node_id: row.get(5)?,
        status: row.get(6)?,
        cancel_requested: row.get::<_, i64>(7)? != 0,
        result_json: row.get(8)?,
        error_json: row.get(9)?,
        created_at_ms: row.get(10)?,
        updated_at_ms: row.get(11)?,
        started_at_ms: row.get(12)?,
        finished_at_ms: row.get(13)?,
    })
}

pub(super) fn server_run_event_from_row(
    row: &rusqlite::Row<'_>,
) -> rusqlite::Result<ServerRunEvent> {
    Ok(ServerRunEvent {
        seq: row.get(0)?,
        run_id: row.get(1)?,
        event_type: row.get(2)?,
        payload_json: row.get(3)?,
        created_at_ms: row.get(4)?,
    })
}

pub(super) fn raw_to_entity_event(raw: RawEntityEvent) -> SqliteStoreResult<EntityEvent> {
    Ok(EntityEventEnvelope {
        namespace: raw.namespace,
        seq: raw.seq,
        event_id: raw.event_id,
        entity_kind: raw.entity_kind,
        entity_id: raw.entity_id,
        op: raw.op,
        payload: serde_json::from_str(&raw.payload_json)?,
    })
}

pub(super) fn replay_cursor(
    conn: &Connection,
    namespace: &str,
    consumer: &str,
) -> SqliteStoreResult<ReplayCursor> {
    let last_seq = conn
        .query_row(
            "SELECT last_seq FROM replay_cursors WHERE namespace = ?1 AND consumer = ?2",
            params![namespace, consumer],
            |row| row.get(0),
        )
        .optional()?
        .unwrap_or(0);
    Ok(ReplayCursor {
        namespace: namespace.to_owned(),
        consumer: consumer.to_owned(),
        last_seq,
    })
}

pub(super) fn strict_advance_replay_cursor(
    conn: &Connection,
    namespace: &str,
    consumer: &str,
    last_seq: i64,
) -> SqliteStoreResult<ReplayCursor> {
    let latest = latest_retained_event_seq(conn, namespace)?;
    if last_seq > latest {
        return Err(StoreError::CursorOutOfRange {
            cursor: last_seq,
            latest,
        }
        .into());
    }
    let current = replay_cursor(conn, namespace, consumer)?.last_seq;
    if last_seq < current {
        return Err(StoreError::CursorRegresses {
            current,
            requested: last_seq,
        }
        .into());
    }
    set_replay_cursor_legacy(conn, namespace, consumer, last_seq)
}

pub(super) fn set_replay_cursor_legacy(
    conn: &Connection,
    namespace: &str,
    consumer: &str,
    last_seq: i64,
) -> SqliteStoreResult<ReplayCursor> {
    conn.execute(
        "INSERT INTO replay_cursors(namespace, consumer, last_seq, updated_at) VALUES (?1, ?2, ?3, ?4) \
         ON CONFLICT(namespace, consumer) DO UPDATE SET last_seq = excluded.last_seq, updated_at = excluded.updated_at",
        params![namespace, consumer, last_seq, unix_epoch_seconds()],
    )?;
    Ok(ReplayCursor {
        namespace: namespace.to_owned(),
        consumer: consumer.to_owned(),
        last_seq,
    })
}

pub(super) fn raw_event_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<RawEntityEvent> {
    Ok(RawEntityEvent {
        namespace: row.get(0)?,
        seq: row.get(1)?,
        event_id: row.get(2)?,
        entity_kind: row.get(3)?,
        entity_id: row.get(4)?,
        op: row.get(5)?,
        payload_json: row.get(6)?,
        created_at: row.get(7)?,
    })
}
