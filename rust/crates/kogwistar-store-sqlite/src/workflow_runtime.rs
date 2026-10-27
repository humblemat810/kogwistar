use super::*;

pub(super) fn put_workflow_design_snapshot(
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

pub(super) fn workflow_design_snapshot(
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

pub(super) fn clear_workflow_design_snapshots(
    conn: &Connection,
    workflow_id: &str,
) -> SqliteStoreResult<()> {
    conn.execute(
        "DELETE FROM workflow_design_snapshots WHERE workflow_id = ?1",
        [workflow_id],
    )?;
    Ok(())
}

pub(super) fn put_workflow_design_delta(
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

pub(super) fn workflow_design_delta(
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

pub(super) fn clear_workflow_design_deltas(
    conn: &Connection,
    workflow_id: &str,
) -> SqliteStoreResult<()> {
    conn.execute(
        "DELETE FROM workflow_design_version_deltas WHERE workflow_id = ?1",
        [workflow_id],
    )?;
    Ok(())
}

pub(super) fn create_server_run(conn: &Connection, run: ServerRunCreate) -> SqliteStoreResult<()> {
    let now = unix_epoch_millis();
    conn.execute(
        "INSERT INTO server_runs(run_id, conversation_id, workflow_id, user_id, user_turn_node_id, assistant_turn_node_id, status, cancel_requested, result_json, error_json, created_at_ms, updated_at_ms, started_at_ms, finished_at_ms) VALUES (?1, ?2, ?3, ?4, ?5, NULL, ?6, 0, NULL, NULL, ?7, ?7, NULL, NULL)",
        params![run.run_id, run.conversation_id, run.workflow_id, run.user_id, run.user_turn_node_id, run.status, now],
    )?;
    Ok(())
}

pub(super) fn server_run(conn: &Connection, run_id: &str) -> SqliteStoreResult<Option<ServerRun>> {
    conn.query_row("SELECT run_id, conversation_id, workflow_id, user_id, user_turn_node_id, assistant_turn_node_id, status, cancel_requested, result_json, error_json, created_at_ms, updated_at_ms, started_at_ms, finished_at_ms FROM server_runs WHERE run_id = ?1", [run_id], server_run_from_row).optional().map_err(Into::into)
}

pub(super) fn server_runs(
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

pub(super) fn append_server_run_event(
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

pub(super) fn server_run_events(
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

pub(super) fn update_server_run(
    conn: &Connection,
    run_id: &str,
    update: ServerRunUpdate,
) -> SqliteStoreResult<()> {
    conn.execute("UPDATE server_runs SET status=?1, assistant_turn_node_id=?2, result_json=?3, error_json=?4, started_at_ms=?5, finished_at_ms=?6, cancel_requested=COALESCE(?7,cancel_requested), updated_at_ms=?8 WHERE run_id=?9", params![update.status, update.assistant_turn_node_id, update.result_json, update.error_json, update.started_at_ms, update.finished_at_ms, update.cancel_requested.map(i64::from), unix_epoch_millis(), run_id])?;
    Ok(())
}

pub(super) fn request_server_run_cancel(conn: &Connection, run_id: &str) -> SqliteStoreResult<()> {
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

pub(super) const RECORDED_RUNTIME_EVENT_TYPE: &str = "workflow.recorded_transition.v1";

pub(super) fn runtime_event_payload(
    persisted: &PersistedRecordedTransition,
) -> SqliteStoreResult<String> {
    serde_json::to_string(persisted).map_err(|error| {
        SqliteStoreError::TransactionAborted(format!(
            "cannot encode recorded runtime transition payload: {error}"
        ))
    })
}

pub(super) fn parse_runtime_event_payload(
    payload_json: &str,
) -> SqliteStoreResult<Option<PersistedRecordedTransition>> {
    match serde_json::from_str::<PersistedRecordedTransition>(payload_json) {
        Ok(value) if value.contract_version == RECORDED_RUNTIME_CONTRACT_VERSION => Ok(Some(value)),
        Ok(_) => Ok(None),
        Err(_) => Ok(None),
    }
}

pub(super) fn recorded_runtime_events(
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

pub(super) fn latest_server_run_event_seq(
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

pub(super) fn runtime_json_object<T: serde::Serialize>(
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

pub(super) fn store_runtime_serving_projections(
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

pub(super) fn recorded_transition_by_id(
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

pub(super) fn read_recorded_runtime_state(
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

pub(super) fn apply_recorded_runtime_transition(
    conn: &Connection,
    transition: RecordedRuntimeTransition,
    abort_after_writes: bool,
) -> SqliteStoreResult<RecordedTransitionResult> {
    apply_recorded_runtime_transition_inner(conn, transition, None, abort_after_writes)
}

pub(super) fn apply_claimed_recorded_runtime_transition(
    conn: &Connection,
    handoff: RecordedWorkerHandoff,
    transition: RecordedRuntimeTransition,
    abort_after_writes: bool,
) -> SqliteStoreResult<RecordedTransitionResult> {
    validate_worker_handoff(&handoff)?;
    let lane = projected_lane_message(conn, &handoff.message_id)?.ok_or_else(|| {
        SqliteStoreError::RecordedRuntimeConflict(format!(
            "worker lane message {:?} does not exist",
            handoff.message_id
        ))
    })?;
    validate_worker_lane_identity(&lane, &handoff, &transition)?;

    if lane.status == "completed" {
        return retry_completed_worker_handoff(conn, &handoff, &transition);
    }
    if lane.status != "claimed"
        || lane.claimed_by.as_deref() != Some(handoff.claimed_by.as_str())
        || lane
            .lease_until
            .as_ref()
            .and_then(|value| value.as_i64())
            .is_none_or(|lease_until| lease_until < unix_epoch_seconds())
    {
        return Err(SqliteStoreError::RecordedRuntimeConflict(format!(
            "worker lane message {:?} is not actively claimed by {:?}",
            handoff.message_id, handoff.claimed_by
        )));
    }

    let result =
        apply_recorded_runtime_transition_inner(conn, transition, Some(handoff.clone()), false)?;
    let acknowledged = conn.execute(
        "UPDATE projected_lane_messages SET status='completed',claimed_by=NULL,lease_until=NULL \
         WHERE message_id=?1 AND status='claimed' AND claimed_by=?2 AND lease_until>=?3",
        params![handoff.message_id, handoff.claimed_by, unix_epoch_seconds()],
    )?;
    if acknowledged != 1 {
        return Err(SqliteStoreError::RecordedRuntimeConflict(format!(
            "worker lane message {:?} claim changed before acknowledgement",
            handoff.message_id
        )));
    }
    if abort_after_writes {
        return Err(SqliteStoreError::TransactionAborted(
            "requested after recorded runtime worker handoff writes".to_owned(),
        ));
    }
    Ok(result)
}

pub(super) fn apply_claimed_recorded_worker_effect(
    conn: &Connection,
    handoff: RecordedWorkerHandoff,
    effect: RecordedWorkerSuccessEffect,
) -> SqliteStoreResult<RecordedTransitionResult> {
    validate_worker_handoff(&handoff)?;
    if effect.contract_version != RECORDED_RUNTIME_CONTRACT_VERSION || effect.effect_id.is_empty() {
        return Err(SqliteStoreError::RecordedRuntimeConflict(
            "worker effect contract_version/effect_id is invalid".to_owned(),
        ));
    }
    let effect_digest = worker_effect_digest(&effect)?;
    let lane = projected_lane_message(conn, &handoff.message_id)?.ok_or_else(|| {
        SqliteStoreError::RecordedRuntimeConflict(format!(
            "worker lane message {:?} does not exist",
            handoff.message_id
        ))
    })?;
    if lane.run_id.as_deref() != Some(handoff.run_id.as_str())
        || lane.step_id.as_deref() != Some(handoff.step_id.as_str())
        || lane.correlation_id.as_deref() != Some(handoff.correlation_id.as_str())
    {
        return Err(SqliteStoreError::RecordedRuntimeConflict(format!(
            "worker lane message {:?} run/step/correlation identity differs",
            handoff.message_id
        )));
    }
    if server_run(conn, &handoff.run_id)?.is_some_and(|run| run.cancel_requested) {
        return Err(SqliteStoreError::RecordedRuntimeConflict(format!(
            "worker lane message {:?} belongs to a cancel-requested run",
            handoff.message_id
        )));
    }
    if lane.status == "completed" {
        if let Some((payload, event_seq)) = recorded_transition_by_id(conn, &effect.effect_id)?
            && (payload.worker_handoff.as_ref() == Some(&handoff)
                || payload
                    .worker_handoff
                    .as_ref()
                    .is_some_and(|stored| stored.message_id == handoff.message_id))
        {
            if payload.worker_effect_digest.as_deref() != Some(effect_digest.as_str()) {
                return Err(SqliteStoreError::RecordedRuntimeConflict(format!(
                    "worker handoff {:?} retried with different effect",
                    handoff.message_id
                )));
            }
            return Ok(payload.result(event_seq, true));
        }
        return Err(SqliteStoreError::RecordedRuntimeConflict(format!(
            "completed worker lane message {:?} has no matching recorded effect",
            handoff.message_id
        )));
    }
    if lane.status != "claimed"
        || lane.claimed_by.as_deref() != Some(handoff.claimed_by.as_str())
        || lane
            .lease_until
            .as_ref()
            .and_then(|value| value.as_i64())
            .is_none_or(|lease_until| lease_until < unix_epoch_seconds())
    {
        return Err(SqliteStoreError::RecordedRuntimeConflict(format!(
            "worker lane message {:?} is not actively claimed by {:?}",
            handoff.message_id, handoff.claimed_by
        )));
    }
    let payload: serde_json::Value =
        serde_json::from_str(lane.payload_json.as_deref().unwrap_or("{}"))?;
    let workflow_id = payload["workflow_id"].as_str().unwrap_or_default();
    let conversation_id = payload["conversation_id"].as_str().unwrap_or_default();
    let token_id = payload["token_id"]
        .as_str()
        .unwrap_or(handoff.run_id.as_str());
    let parent_token_id = payload["parent_token_id"].as_str();
    let current = read_recorded_runtime_state(conn, &handoff.run_id, workflow_id, conversation_id)?
        .ok_or_else(|| {
            SqliteStoreError::RecordedRuntimeConflict(format!(
                "worker lane message {:?} has no recorded runtime state",
                handoff.message_id
            ))
        })?;
    if current
        .frontier
        .pending
        .first()
        .is_none_or(|(node, _, token, parent)| {
            node != &handoff.step_id || token != token_id || parent.as_deref() != parent_token_id
        })
    {
        return Err(SqliteStoreError::RecordedRuntimeConflict(format!(
            "worker lane message {:?} result is not next in canonical frontier order",
            handoff.message_id
        )));
    }
    let expected_event_seq = latest_server_run_event_seq(conn, &handoff.run_id)?;
    let step_seq = current.last_step_seq.saturating_add(1);
    let authoritative_successors = match effect.status {
        RuntimeWorkerEffectStatus::Success => {
            kogwistar_runtime::authoritative_runtime_successors_for_result(
                &current.static_routes,
                &handoff.step_id,
                &effect.successors,
                &effect.route_next,
                false,
            )?
        }
        RuntimeWorkerEffectStatus::Suspended => Vec::new(),
        RuntimeWorkerEffectStatus::Failed => {
            kogwistar_runtime::authoritative_runtime_successors_for_result(
                &current.static_routes,
                &handoff.step_id,
                &effect.successors,
                &effect.route_next,
                true,
            )?
        }
    };
    let terminal_failure =
        effect.status == RuntimeWorkerEffectStatus::Failed && authoritative_successors.is_empty();
    let next_frontier = match effect.status {
        RuntimeWorkerEffectStatus::Success => frontier_after_worker_success(
            &current.frontier,
            &handoff.step_id,
            token_id,
            parent_token_id,
            step_seq,
            &kogwistar_runtime::RuntimeWorkerSuccessEffect {
                successors: authoritative_successors,
            },
        )?,
        RuntimeWorkerEffectStatus::Suspended => frontier_after_worker_suspend(
            &current.frontier,
            &handoff.step_id,
            token_id,
            parent_token_id,
        )?,
        RuntimeWorkerEffectStatus::Failed if terminal_failure => RuntimeFrontier::default(),
        RuntimeWorkerEffectStatus::Failed => frontier_after_worker_success(
            &current.frontier,
            &handoff.step_id,
            token_id,
            parent_token_id,
            step_seq,
            &kogwistar_runtime::RuntimeWorkerSuccessEffect {
                successors: authoritative_successors,
            },
        )?,
    };
    let transition = RecordedRuntimeTransition {
        contract_version: RECORDED_RUNTIME_CONTRACT_VERSION,
        transition_id: effect.effect_id.clone(),
        expected_event_seq,
        kind: match effect.status {
            RuntimeWorkerEffectStatus::Success => {
                kogwistar_runtime::RecordedTransitionKind::RecordedStepSuccess
            }
            RuntimeWorkerEffectStatus::Suspended => {
                kogwistar_runtime::RecordedTransitionKind::Suspend
            }
            RuntimeWorkerEffectStatus::Failed if terminal_failure => {
                kogwistar_runtime::RecordedTransitionKind::Fail
            }
            RuntimeWorkerEffectStatus::Failed => {
                kogwistar_runtime::RecordedTransitionKind::RecordedStepSuccess
            }
        },
        run_id: handoff.run_id.clone(),
        workflow_id: workflow_id.to_owned(),
        conversation_id: conversation_id.to_owned(),
        user_id: None,
        user_turn_node_id: None,
        step_seq,
        node_id: Some(handoff.step_id.clone()),
        token_id: Some(token_id.to_owned()),
        parent_token_id: parent_token_id.map(str::to_owned),
        initial_state: None,
        state_update: effect.state_update,
        update: effect.update,
        state_schema: effect.state_schema,
        frontier: Some(next_frontier),
        result: effect.result,
        wait_reason: effect.wait_reason,
        resume_payload: effect.resume_payload,
        errors: effect.errors,
    };
    let reduced = reduce_recorded_transition(Some(&current), &transition)?;
    let request_digest = transition_digest(&transition)?;
    let persisted = PersistedRecordedTransition {
        contract_version: RECORDED_RUNTIME_CONTRACT_VERSION,
        transition_id: transition.transition_id.clone(),
        request_digest,
        worker_handoff: Some(handoff.clone()),
        worker_effect_digest: Some(effect_digest),
        reduced: reduced.clone(),
    };
    let event = append_server_run_event(
        conn,
        &transition.run_id,
        RECORDED_RUNTIME_EVENT_TYPE,
        runtime_event_payload(&persisted)?,
    )?;
    store_runtime_serving_projections(conn, &persisted, event.seq)?;
    replace_named_projection(
        conn,
        &runtime_checkpoint_namespace(&transition.conversation_id),
        &transition.run_id,
        runtime_projection_write(
            reduced.checkpoint.clone(),
            event.seq,
            reduced.state.status.server_status().to_owned(),
        ),
    )?;
    replace_named_projection(
        conn,
        &runtime_status_namespace(&transition.conversation_id),
        &transition.run_id,
        runtime_projection_write(reduced.run_status.clone(), event.seq, "running"),
    )?;
    let prior_run = server_run(conn, &transition.run_id)?;
    update_server_run(
        conn,
        &transition.run_id,
        ServerRunUpdate {
            status: reduced.server_status.clone(),
            assistant_turn_node_id: prior_run
                .as_ref()
                .and_then(|run| run.assistant_turn_node_id.clone()),
            result_json: reduced
                .result
                .as_ref()
                .map(serde_json::to_string)
                .transpose()?,
            error_json: if reduced.errors.is_empty() {
                None
            } else {
                Some(serde_json::to_string(&reduced.errors)?)
            },
            started_at_ms: prior_run.as_ref().and_then(|run| run.started_at_ms),
            finished_at_ms: None,
            cancel_requested: Some(false),
        },
    )?;
    let acknowledged = conn.execute(
        "UPDATE projected_lane_messages SET status='completed',claimed_by=NULL,lease_until=NULL \
         WHERE message_id=?1 AND status='claimed' AND claimed_by=?2 AND lease_until>=?3",
        params![handoff.message_id, handoff.claimed_by, unix_epoch_seconds()],
    )?;
    if acknowledged != 1 {
        return Err(SqliteStoreError::RecordedRuntimeConflict(format!(
            "worker lane message {:?} claim changed before acknowledgement",
            handoff.message_id
        )));
    }
    Ok(persisted.result(event.seq, false))
}

pub(super) fn validate_worker_handoff(handoff: &RecordedWorkerHandoff) -> SqliteStoreResult<()> {
    for (field, value) in [
        ("message_id", handoff.message_id.as_str()),
        ("claimed_by", handoff.claimed_by.as_str()),
        ("run_id", handoff.run_id.as_str()),
        ("step_id", handoff.step_id.as_str()),
        ("correlation_id", handoff.correlation_id.as_str()),
    ] {
        if value.is_empty() {
            return Err(SqliteStoreError::RecordedRuntimeConflict(format!(
                "worker handoff {field} must not be empty"
            )));
        }
    }
    Ok(())
}

pub(super) fn validate_worker_lane_identity(
    lane: &ProjectedLaneMessage,
    handoff: &RecordedWorkerHandoff,
    transition: &RecordedRuntimeTransition,
) -> SqliteStoreResult<()> {
    if handoff.run_id != transition.run_id
        || transition.node_id.as_deref() != Some(handoff.step_id.as_str())
        || lane.run_id.as_deref() != Some(handoff.run_id.as_str())
        || lane.step_id.as_deref() != Some(handoff.step_id.as_str())
        || lane.correlation_id.as_deref() != Some(handoff.correlation_id.as_str())
    {
        return Err(SqliteStoreError::RecordedRuntimeConflict(format!(
            "worker lane message {:?} run/step/correlation identity differs",
            handoff.message_id
        )));
    }
    Ok(())
}

pub(super) fn retry_completed_worker_handoff(
    conn: &Connection,
    handoff: &RecordedWorkerHandoff,
    transition: &RecordedRuntimeTransition,
) -> SqliteStoreResult<RecordedTransitionResult> {
    let request_digest = transition_digest(transition)?;
    if let Some((payload, event_seq)) = recorded_transition_by_id(conn, &transition.transition_id)?
    {
        if payload.request_digest != request_digest
            || payload.worker_handoff.as_ref() != Some(handoff)
        {
            return Err(SqliteStoreError::RecordedRuntimeConflict(format!(
                "worker handoff {:?} retried with different result",
                handoff.message_id
            )));
        }
        return Ok(payload.result(event_seq, true));
    }
    Err(SqliteStoreError::RecordedRuntimeConflict(format!(
        "completed worker lane message {:?} has no matching recorded result",
        handoff.message_id
    )))
}

pub(super) fn apply_recorded_runtime_transition_inner(
    conn: &Connection,
    transition: RecordedRuntimeTransition,
    worker_handoff: Option<RecordedWorkerHandoff>,
    abort_after_writes: bool,
) -> SqliteStoreResult<RecordedTransitionResult> {
    let request_digest = transition_digest(&transition)?;

    // Idempotency lookup occurs before sequence/CAS checks. Exact retry returns
    // the original response; reusing an id for different immutable input fails.
    if let Some((payload, event_seq)) = recorded_transition_by_id(conn, &transition.transition_id)?
    {
        if payload.request_digest != request_digest || payload.worker_handoff != worker_handoff {
            return Err(SqliteStoreError::RecordedRuntimeConflict(format!(
                "transition_id {:?} reused with different payload",
                transition.transition_id
            )));
        }
        return Ok(payload.result(event_seq, true));
    }

    let existing_run = server_run(conn, &transition.run_id)?;
    if let Some(run) = &existing_run
        && (run.workflow_id != transition.workflow_id
            || run.conversation_id != transition.conversation_id)
    {
        return Err(SqliteStoreError::RecordedRuntimeConflict(format!(
            "run identity differs for {:?}",
            transition.run_id
        )));
    }
    let current = read_recorded_runtime_state(
        conn,
        &transition.run_id,
        &transition.workflow_id,
        &transition.conversation_id,
    )?;
    let current_event_seq = latest_server_run_event_seq(conn, &transition.run_id)?;
    if current_event_seq != transition.expected_event_seq {
        return Err(SqliteStoreError::RecordedRuntimeConflict(format!(
            "expected_event_seq {}, current {}",
            transition.expected_event_seq, current_event_seq
        )));
    }
    let reduced = reduce_recorded_transition(current.as_ref(), &transition)?;

    if current.is_none() && existing_run.is_none() {
        let user_turn_node_id = transition.user_turn_node_id.clone().ok_or_else(|| {
            SqliteStoreError::RecordedRuntimeConflict(
                "start transition requires user_turn_node_id".to_owned(),
            )
        })?;
        create_server_run(
            conn,
            ServerRunCreate {
                run_id: transition.run_id.clone(),
                conversation_id: transition.conversation_id.clone(),
                workflow_id: transition.workflow_id.clone(),
                user_id: transition.user_id.clone(),
                user_turn_node_id,
                status: reduced.server_status.clone(),
            },
        )?;
    }

    let persisted = PersistedRecordedTransition {
        contract_version: RECORDED_RUNTIME_CONTRACT_VERSION,
        transition_id: transition.transition_id.clone(),
        request_digest,
        worker_handoff,
        worker_effect_digest: None,
        reduced: reduced.clone(),
    };
    let event = append_server_run_event(
        conn,
        &transition.run_id,
        RECORDED_RUNTIME_EVENT_TYPE,
        runtime_event_payload(&persisted)?,
    )?;
    let event_seq = event.seq;
    store_runtime_serving_projections(conn, &persisted, event_seq)?;
    replace_named_projection(
        conn,
        &runtime_checkpoint_namespace(&transition.conversation_id),
        &transition.run_id,
        runtime_projection_write(
            reduced.checkpoint.clone(),
            event_seq,
            reduced.state.status.server_status().to_owned(),
        ),
    )?;
    let status_materialization = match reduced.state.status {
        kogwistar_runtime::RecordedRunStatus::Completed => "completed",
        kogwistar_runtime::RecordedRunStatus::Failed => "failed",
        kogwistar_runtime::RecordedRunStatus::Cancelled => "cancelled",
        kogwistar_runtime::RecordedRunStatus::Suspended => "suspended",
        kogwistar_runtime::RecordedRunStatus::Running => "running",
    }
    .to_owned();
    replace_named_projection(
        conn,
        &runtime_status_namespace(&transition.conversation_id),
        &transition.run_id,
        runtime_projection_write(
            reduced.run_status.clone(),
            event_seq,
            status_materialization,
        ),
    )?;
    let prior_run = existing_run.as_ref();
    update_server_run(
        conn,
        &transition.run_id,
        ServerRunUpdate {
            status: reduced.server_status.clone(),
            assistant_turn_node_id: prior_run.and_then(|run| run.assistant_turn_node_id.clone()),
            result_json: reduced
                .result
                .as_ref()
                .map(serde_json::to_string)
                .transpose()?
                .or_else(|| prior_run.and_then(|run| run.result_json.clone())),
            error_json: if reduced.errors.is_empty() {
                prior_run.and_then(|run| run.error_json.clone())
            } else {
                Some(serde_json::to_string(&reduced.errors)?)
            },
            started_at_ms: prior_run.and_then(|run| run.started_at_ms).or_else(|| {
                (transition.kind == kogwistar_runtime::RecordedTransitionKind::Start)
                    .then(unix_epoch_millis)
            }),
            finished_at_ms: if reduced.state.status.is_terminal() {
                Some(unix_epoch_millis())
            } else {
                prior_run.and_then(|run| run.finished_at_ms)
            },
            cancel_requested: Some(matches!(
                reduced.state.status,
                kogwistar_runtime::RecordedRunStatus::Cancelled
            )),
        },
    )?;
    if abort_after_writes {
        return Err(SqliteStoreError::TransactionAborted(
            "requested after recorded runtime transition writes".to_owned(),
        ));
    }
    Ok(persisted.result(event_seq, false))
}
