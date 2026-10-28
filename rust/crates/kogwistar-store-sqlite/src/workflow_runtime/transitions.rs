use super::super::*;

pub(crate) fn apply_recorded_runtime_transition(
    conn: &Connection,
    transition: RecordedRuntimeTransition,
    abort_after_writes: bool,
) -> SqliteStoreResult<RecordedTransitionResult> {
    apply_recorded_runtime_transition_inner(conn, transition, None, abort_after_writes)
}

pub(crate) fn apply_claimed_recorded_runtime_transition(
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

pub(crate) fn apply_recorded_runtime_transition_inner(
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
