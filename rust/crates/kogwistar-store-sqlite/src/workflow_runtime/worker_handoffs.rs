use super::super::*;

pub(crate) fn apply_claimed_recorded_worker_effect(
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

pub(crate) fn validate_worker_handoff(handoff: &RecordedWorkerHandoff) -> SqliteStoreResult<()> {
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

pub(crate) fn validate_worker_lane_identity(
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

pub(crate) fn retry_completed_worker_handoff(
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
