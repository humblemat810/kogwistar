use super::super::*;

pub(crate) fn runtime_event_payload(
    persisted: &PersistedRecordedTransition,
) -> SqliteStoreResult<String> {
    serde_json::to_string(persisted).map_err(|error| {
        SqliteStoreError::TransactionAborted(format!(
            "cannot encode recorded runtime transition payload: {error}"
        ))
    })
}

pub(crate) fn parse_runtime_event_payload(
    payload_json: &str,
) -> SqliteStoreResult<Option<PersistedRecordedTransition>> {
    match serde_json::from_str::<PersistedRecordedTransition>(payload_json) {
        Ok(value) if value.contract_version == RECORDED_RUNTIME_CONTRACT_VERSION => Ok(Some(value)),
        Ok(_) => Ok(None),
        Err(_) => Ok(None),
    }
}

pub(crate) fn recorded_runtime_events(
    conn: &Connection,
    run_id: &str,
) -> SqliteStoreResult<Vec<ServerRunEvent>> {
    let mut statement = conn.prepare(
        "SELECT seq, run_id, event_type, payload_json, created_at_ms FROM server_run_events \
         WHERE run_id = ?1 AND event_type = ?2 ORDER BY seq ASC",
    )?;
    statement
        .query_map(
            params![run_id, RECORDED_RUNTIME_EVENT_TYPE],
            server_run_event_from_row,
        )?
        .collect::<Result<Vec<_>, _>>()
        .map_err(Into::into)
}

pub(crate) fn latest_server_run_event_seq(
    conn: &Connection,
    run_id: &str,
) -> SqliteStoreResult<i64> {
    conn.query_row(
        "SELECT COALESCE(MAX(seq), 0) FROM server_run_events WHERE run_id = ?1",
        [run_id],
        |row| row.get(0),
    )
    .map_err(Into::into)
}

pub(crate) fn runtime_json_object<T: serde::Serialize>(
    value: &T,
) -> SqliteStoreResult<Map<String, Value>> {
    serde_json::to_value(value)?
        .as_object()
        .cloned()
        .ok_or_else(|| {
            SqliteStoreError::TransactionAborted(
                "runtime projection payload is not an object".to_owned(),
            )
        })
}

pub(crate) fn store_runtime_serving_projections(
    conn: &Connection,
    persisted: &PersistedRecordedTransition,
    event_seq: i64,
) -> SqliteStoreResult<()> {
    replace_named_projection(
        conn,
        RUNTIME_CURRENT_STATE_NAMESPACE,
        &persisted.reduced.state.run_id,
        runtime_projection_write(
            runtime_json_object(&persisted.reduced.state)?,
            event_seq,
            "ready".to_owned(),
        ),
    )
}

pub(crate) fn recorded_transition_by_id(
    conn: &Connection,
    transition_id: &str,
) -> SqliteStoreResult<Option<(PersistedRecordedTransition, i64)>> {
    conn.query_row(
        "SELECT payload_json, seq FROM server_run_events \
         WHERE event_type = ?1 AND json_extract(payload_json, '$.transition_id') = ?2 LIMIT 1",
        params![RECORDED_RUNTIME_EVENT_TYPE, transition_id],
        |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?)),
    )
    .optional()?
    .map(|(payload_json, seq)| {
        let payload = serde_json::from_str::<PersistedRecordedTransition>(&payload_json)?;
        Ok((payload, seq))
    })
    .transpose()
}

pub(crate) fn read_recorded_runtime_state(
    conn: &Connection,
    run_id: &str,
    workflow_id: &str,
    conversation_id: &str,
) -> SqliteStoreResult<Option<RecordedRuntimeState>> {
    let Some(run) = server_run(conn, run_id)? else {
        return Ok(None);
    };
    if run.workflow_id != workflow_id || run.conversation_id != conversation_id {
        return Err(SqliteStoreError::RecordedRuntimeConflict(format!(
            "run identity differs for {run_id:?}"
        )));
    }
    if let Some(row) = named_projection(conn, RUNTIME_CURRENT_STATE_NAMESPACE, run_id)?
        && row.materialization_status == "ready"
    {
        let has_newer_recorded_event: bool = conn.query_row(
            "SELECT EXISTS(
                SELECT 1 FROM server_run_events
                WHERE run_id = ?1 AND event_type = ?2 AND seq > ?3
             )",
            params![
                run_id,
                RECORDED_RUNTIME_EVENT_TYPE,
                row.last_authoritative_seq
            ],
            |result| result.get(0),
        )?;
        if !has_newer_recorded_event
            && let Ok(state) =
                serde_json::from_value::<RecordedRuntimeState>(Value::Object(row.payload))
            && state.run_id == run_id
            && state.workflow_id == workflow_id
            && state.conversation_id == conversation_id
        {
            return Ok(Some(state));
        }
    }
    let mut latest: Option<RecordedRuntimeState> = None;
    for event in recorded_runtime_events(conn, run_id)? {
        if let Some(payload) = parse_runtime_event_payload(&event.payload_json)?
            && payload.reduced.state.run_id == run_id
            && payload.reduced.state.workflow_id == workflow_id
            && payload.reduced.state.conversation_id == conversation_id
        {
            latest = Some(payload.reduced.state);
        }
    }
    Ok(latest)
}
