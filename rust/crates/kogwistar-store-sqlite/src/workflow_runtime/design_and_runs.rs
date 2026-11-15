use super::super::*;

pub(crate) fn put_workflow_design_snapshot(
    conn: &Connection,
    workflow_id: &str,
    snapshot: WorkflowDesignSnapshotWrite,
) -> SqliteStoreResult<()> {
    conn.execute(
        "INSERT INTO workflow_design_snapshots(workflow_id, version, seq, payload_json, schema_version, created_at_ms) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6) \
         ON CONFLICT(workflow_id, version) DO UPDATE SET \
         seq = excluded.seq, payload_json = excluded.payload_json, \
         schema_version = excluded.schema_version, created_at_ms = excluded.created_at_ms",
        params![
            workflow_id,
            snapshot.version,
            snapshot.seq,
            snapshot.payload_json,
            snapshot.schema_version,
            unix_epoch_millis(),
        ],
    )?;
    Ok(())
}

pub(crate) fn workflow_design_snapshot(
    conn: &Connection,
    workflow_id: &str,
    max_version: i64,
    schema_version: i64,
) -> SqliteStoreResult<Option<WorkflowDesignSnapshot>> {
    conn.query_row(
        "SELECT workflow_id, version, seq, payload_json, schema_version, created_at_ms \
         FROM workflow_design_snapshots \
         WHERE workflow_id = ?1 AND version <= ?2 AND schema_version = ?3 \
         ORDER BY version DESC LIMIT 1",
        params![workflow_id, max_version, schema_version],
        workflow_design_snapshot_from_row,
    )
    .optional()
    .map_err(Into::into)
}

pub(crate) fn clear_workflow_design_snapshots(
    conn: &Connection,
    workflow_id: &str,
) -> SqliteStoreResult<()> {
    conn.execute(
        "DELETE FROM workflow_design_snapshots WHERE workflow_id = ?1",
        [workflow_id],
    )?;
    Ok(())
}

pub(crate) fn put_workflow_design_delta(
    conn: &Connection,
    workflow_id: &str,
    delta: WorkflowDesignDeltaWrite,
) -> SqliteStoreResult<()> {
    conn.execute(
        "INSERT INTO workflow_design_version_deltas(workflow_id, version, prev_version, target_seq, forward_json, inverse_json, schema_version, created_at_ms) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8) \
         ON CONFLICT(workflow_id, version) DO UPDATE SET \
         prev_version = excluded.prev_version, target_seq = excluded.target_seq, \
         forward_json = excluded.forward_json, inverse_json = excluded.inverse_json, \
         schema_version = excluded.schema_version, created_at_ms = excluded.created_at_ms",
        params![
            workflow_id,
            delta.version,
            delta.prev_version,
            delta.target_seq,
            delta.forward_json,
            delta.inverse_json,
            delta.schema_version,
            unix_epoch_millis(),
        ],
    )?;
    Ok(())
}

pub(crate) fn workflow_design_delta(
    conn: &Connection,
    workflow_id: &str,
    version: i64,
    schema_version: i64,
) -> SqliteStoreResult<Option<WorkflowDesignDelta>> {
    conn.query_row(
        "SELECT workflow_id, version, prev_version, target_seq, forward_json, inverse_json, schema_version, created_at_ms \
         FROM workflow_design_version_deltas \
         WHERE workflow_id = ?1 AND version = ?2 AND schema_version = ?3",
        params![workflow_id, version, schema_version],
        workflow_design_delta_from_row,
    )
    .optional()
    .map_err(Into::into)
}

pub(crate) fn clear_workflow_design_deltas(
    conn: &Connection,
    workflow_id: &str,
) -> SqliteStoreResult<()> {
    conn.execute(
        "DELETE FROM workflow_design_version_deltas WHERE workflow_id = ?1",
        [workflow_id],
    )?;
    Ok(())
}

pub(crate) fn create_server_run(conn: &Connection, run: ServerRunCreate) -> SqliteStoreResult<()> {
    let now = unix_epoch_millis();
    conn.execute(
        "INSERT INTO server_runs(run_id, conversation_id, workflow_id, user_id, user_turn_node_id, assistant_turn_node_id, status, cancel_requested, result_json, error_json, created_at_ms, updated_at_ms, started_at_ms, finished_at_ms) VALUES (?1, ?2, ?3, ?4, ?5, NULL, ?6, 0, NULL, NULL, ?7, ?7, NULL, NULL)",
        params![run.run_id, run.conversation_id, run.workflow_id, run.user_id, run.user_turn_node_id, run.status, now],
    )?;
    Ok(())
}

pub(crate) fn server_run(conn: &Connection, run_id: &str) -> SqliteStoreResult<Option<ServerRun>> {
    conn.query_row("SELECT run_id, conversation_id, workflow_id, user_id, user_turn_node_id, assistant_turn_node_id, status, cancel_requested, result_json, error_json, created_at_ms, updated_at_ms, started_at_ms, finished_at_ms FROM server_runs WHERE run_id = ?1", [run_id], server_run_from_row).optional().map_err(Into::into)
}

pub(crate) fn server_runs(
    conn: &Connection,
    status: Option<&str>,
    workflow_id: Option<&str>,
    conversation_id: Option<&str>,
    limit: usize,
) -> SqliteStoreResult<Vec<ServerRun>> {
    let mut sql = "SELECT run_id, conversation_id, workflow_id, user_id, user_turn_node_id, assistant_turn_node_id, status, cancel_requested, result_json, error_json, created_at_ms, updated_at_ms, started_at_ms, finished_at_ms FROM server_runs".to_owned();
    let mut predicates = Vec::new();
    if status.is_some() {
        predicates.push("status = ?");
    }
    if workflow_id.is_some() {
        predicates.push("workflow_id = ?");
    }
    if conversation_id.is_some() {
        predicates.push("conversation_id = ?");
    }
    if !predicates.is_empty() {
        sql.push_str(" WHERE ");
        sql.push_str(&predicates.join(" AND "));
    }
    sql.push_str(" ORDER BY created_at_ms DESC, run_id DESC LIMIT ?");
    let mut bindings: Vec<rusqlite::types::Value> = Vec::new();
    if let Some(value) = status {
        bindings.push(value.to_owned().into());
    }
    if let Some(value) = workflow_id {
        bindings.push(value.to_owned().into());
    }
    if let Some(value) = conversation_id {
        bindings.push(value.to_owned().into());
    }
    bindings.push((limit as i64).into());
    let mut statement = conn.prepare(&sql)?;
    statement
        .query_map(rusqlite::params_from_iter(bindings), server_run_from_row)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(Into::into)
}

pub(crate) fn append_server_run_event(
    conn: &Connection,
    run_id: &str,
    event_type: &str,
    payload_json: String,
) -> SqliteStoreResult<ServerRunEvent> {
    let created_at_ms = unix_epoch_millis();
    conn.execute("INSERT INTO server_run_events(run_id, event_type, payload_json, created_at_ms) VALUES (?1, ?2, ?3, ?4)", params![run_id, event_type, payload_json, created_at_ms])?;
    Ok(ServerRunEvent {
        seq: conn.last_insert_rowid(),
        run_id: run_id.to_owned(),
        event_type: event_type.to_owned(),
        payload_json,
        created_at_ms,
    })
}

pub(crate) fn server_run_events(
    conn: &Connection,
    run_id: &str,
    after_seq: i64,
    limit: usize,
) -> SqliteStoreResult<Vec<ServerRunEvent>> {
    let mut statement=conn.prepare("SELECT seq, run_id, event_type, payload_json, created_at_ms FROM server_run_events WHERE run_id = ?1 AND seq > ?2 ORDER BY seq ASC LIMIT ?3")?;
    statement
        .query_map(
            params![run_id, after_seq, limit as i64],
            server_run_event_from_row,
        )?
        .collect::<Result<Vec<_>, _>>()
        .map_err(Into::into)
}

pub(crate) fn update_server_run(
    conn: &Connection,
    run_id: &str,
    update: ServerRunUpdate,
) -> SqliteStoreResult<()> {
    conn.execute("UPDATE server_runs SET status=?1, assistant_turn_node_id=?2, result_json=?3, error_json=?4, started_at_ms=?5, finished_at_ms=?6, cancel_requested=COALESCE(?7,cancel_requested), updated_at_ms=?8 WHERE run_id=?9", params![update.status, update.assistant_turn_node_id, update.result_json, update.error_json, update.started_at_ms, update.finished_at_ms, update.cancel_requested.map(i64::from), unix_epoch_millis(), run_id])?;
    Ok(())
}

pub(crate) fn request_server_run_cancel(conn: &Connection, run_id: &str) -> SqliteStoreResult<()> {
    let Some(run) = server_run(conn, run_id)? else {
        return Ok(());
    };
    let mut current =
        read_recorded_runtime_state(conn, run_id, &run.workflow_id, &run.conversation_id)?;
    if current.is_none() {
        let payload_json = conn
            .query_row(
                "SELECT payload_json FROM projected_lane_messages \
                 WHERE run_id=?1 AND msg_type='workflow.run.execute' ORDER BY created_at LIMIT 1",
                [run_id],
                |row| row.get::<_, Option<String>>(0),
            )
            .optional()?
            .flatten();
        if let Some(payload_json) = payload_json {
            let payload: Value = serde_json::from_str(&payload_json)?;
            let expected_event_seq = latest_server_run_event_seq(conn, run_id)?;
            apply_recorded_runtime_transition_inner(
                conn,
                RecordedRuntimeTransition {
                    contract_version: RECORDED_RUNTIME_CONTRACT_VERSION,
                    transition_id: format!("start-{run_id}"),
                    expected_event_seq,
                    kind: kogwistar_runtime::RecordedTransitionKind::Start,
                    run_id: run_id.to_owned(),
                    workflow_id: run.workflow_id.clone(),
                    conversation_id: run.conversation_id.clone(),
                    user_id: payload["user_id"].as_str().map(str::to_owned),
                    user_turn_node_id: run
                        .user_turn_node_id
                        .clone()
                        .or_else(|| payload["turn_node_id"].as_str().map(str::to_owned)),
                    step_seq: 0,
                    node_id: Some("start".to_owned()),
                    token_id: Some(run_id.to_owned()),
                    parent_token_id: None,
                    initial_state: Some(
                        payload["initial_state"]
                            .as_object()
                            .cloned()
                            .unwrap_or_default(),
                    ),
                    state_update: Vec::new(),
                    update: None,
                    state_schema: Map::new(),
                    frontier: Some(RuntimeFrontier {
                        pending: vec![("start".to_owned(), 0, run_id.to_owned(), None)],
                        ..RuntimeFrontier::default()
                    }),
                    result: None,
                    wait_reason: None,
                    resume_payload: None,
                    errors: Vec::new(),
                },
                None,
                false,
            )?;
            current =
                read_recorded_runtime_state(conn, run_id, &run.workflow_id, &run.conversation_id)?;
        }
    }
    if let Some(state) = current
        && !state.status.is_terminal()
    {
        let expected_event_seq = latest_server_run_event_seq(conn, run_id)?;
        apply_recorded_runtime_transition_inner(
            conn,
            RecordedRuntimeTransition {
                contract_version: RECORDED_RUNTIME_CONTRACT_VERSION,
                transition_id: format!("cancel-{run_id}-{expected_event_seq}"),
                expected_event_seq,
                kind: kogwistar_runtime::RecordedTransitionKind::Cancel,
                run_id: run_id.to_owned(),
                workflow_id: run.workflow_id.clone(),
                conversation_id: run.conversation_id.clone(),
                user_id: None,
                user_turn_node_id: None,
                step_seq: state.last_step_seq.max(0),
                node_id: state.last_node_id,
                token_id: state.last_token_id,
                parent_token_id: state.last_parent_token_id,
                initial_state: None,
                state_update: Vec::new(),
                update: None,
                state_schema: Map::new(),
                frontier: None,
                result: None,
                wait_reason: None,
                resume_payload: None,
                errors: Vec::new(),
            },
            None,
            false,
        )?;
    } else {
        conn.execute("UPDATE server_runs SET cancel_requested=1, status=CASE WHEN status IN ('cancelled','failed','succeeded') THEN status ELSE 'cancelling' END, updated_at_ms=?1 WHERE run_id=?2", params![unix_epoch_millis(), run_id])?;
    }
    conn.execute(
        "UPDATE projected_lane_messages SET status='cancelled',claimed_by=NULL,lease_until=NULL \
         WHERE run_id=?1 AND status NOT IN ('completed','failed','cancelled','dead-letter')",
        [run_id],
    )?;
    Ok(())
}

pub(crate) const RECORDED_RUNTIME_EVENT_TYPE: &str = "workflow.recorded_transition.v1";
