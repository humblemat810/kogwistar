use super::super::*;

pub(crate) async fn apply_claimed_recorded_worker_effect<C>(
    client: &C,
    tables: &Tables,
    handoff: RecordedWorkerHandoff,
    effect: RecordedWorkerSuccessEffect,
) -> PostgresStoreResult<RecordedTransitionResult>
where
    C: GenericClient + Sync,
{
    validate_worker_handoff(&handoff)?;
    if effect.contract_version != RECORDED_RUNTIME_CONTRACT_VERSION || effect.effect_id.is_empty() {
        return Err(PostgresStoreError::RecordedRuntimeConflict(
            "worker effect contract_version/effect_id is invalid".to_owned(),
        ));
    }
    let effect_digest = worker_effect_digest(&effect)?;
    advisory_lock(client, &format!("recorded-run\u{1f}{}", handoff.run_id), 42).await?;
    let row = client
        .query_opt(
            &format!(
                "SELECT {LANE_SELECT},lease_until>=NOW() FROM {} WHERE message_id=$1 FOR UPDATE",
                tables.projected_lane_messages
            ),
            &[&handoff.message_id],
        )
        .await
        .map_err(backend)?
        .ok_or_else(|| {
            PostgresStoreError::RecordedRuntimeConflict(format!(
                "worker lane message {:?} does not exist",
                handoff.message_id
            ))
        })?;
    let lane = lane_from_row(&row)?;
    let lease_active: Option<bool> = row.try_get(25).map_err(backend)?;
    if lane.run_id.as_deref() != Some(handoff.run_id.as_str())
        || lane.step_id.as_deref() != Some(handoff.step_id.as_str())
        || lane.correlation_id.as_deref() != Some(handoff.correlation_id.as_str())
    {
        return Err(PostgresStoreError::RecordedRuntimeConflict(format!(
            "worker lane message {:?} run/step/correlation identity differs",
            handoff.message_id
        )));
    }
    if server_run(client, tables, &handoff.run_id)
        .await?
        .is_some_and(|run| run.cancel_requested)
    {
        return Err(PostgresStoreError::RecordedRuntimeConflict(format!(
            "worker lane message {:?} belongs to a cancel-requested run",
            handoff.message_id
        )));
    }
    if lane.status == "completed" {
        if let Some((payload, event_seq)) =
            recorded_transition_by_id(client, tables, &effect.effect_id).await?
            && payload
                .worker_handoff
                .as_ref()
                .is_some_and(|stored| stored.message_id == handoff.message_id)
        {
            if payload.worker_effect_digest.as_deref() != Some(effect_digest.as_str()) {
                return Err(PostgresStoreError::RecordedRuntimeConflict(format!(
                    "worker handoff {:?} retried with different effect",
                    handoff.message_id
                )));
            }
            return Ok(payload.result(event_seq, true));
        }
        return Err(PostgresStoreError::RecordedRuntimeConflict(format!(
            "completed worker lane message {:?} has no matching recorded effect",
            handoff.message_id
        )));
    }
    if lane.status != "claimed"
        || lane.claimed_by.as_deref() != Some(handoff.claimed_by.as_str())
        || lease_active != Some(true)
    {
        return Err(PostgresStoreError::RecordedRuntimeConflict(format!(
            "worker lane message {:?} is not actively claimed by {:?}",
            handoff.message_id, handoff.claimed_by
        )));
    }
    let payload: Value = serde_json::from_str(lane.payload_json.as_deref().unwrap_or("{}"))?;
    let workflow_id = payload["workflow_id"].as_str().unwrap_or_default();
    let conversation_id = payload["conversation_id"].as_str().unwrap_or_default();
    let token_id = payload["token_id"]
        .as_str()
        .unwrap_or(handoff.run_id.as_str());
    let parent_token_id = payload["parent_token_id"].as_str();
    let current = read_recorded_runtime_state(
        client,
        tables,
        &handoff.run_id,
        workflow_id,
        conversation_id,
    )
    .await?
    .ok_or_else(|| {
        PostgresStoreError::RecordedRuntimeConflict(format!(
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
        return Err(PostgresStoreError::RecordedRuntimeConflict(format!(
            "worker lane message {:?} result is not next in canonical frontier order",
            handoff.message_id
        )));
    }
    let expected_event_seq = latest_server_run_event_seq(client, tables, &handoff.run_id).await?;
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
    let frontier = match effect.status {
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
        transition_id: effect.effect_id,
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
        frontier: Some(frontier),
        result: effect.result,
        wait_reason: effect.wait_reason,
        resume_payload: effect.resume_payload,
        errors: effect.errors,
    };
    let result = apply_recorded_runtime_transition_inner(
        client,
        tables,
        transition,
        Some(handoff.clone()),
        Some(effect_digest),
        false,
    )
    .await?;
    let acknowledged = client
        .execute(
            &format!(
                "UPDATE {} SET status='completed',claimed_by=NULL,lease_until=NULL \
                 WHERE message_id=$1 AND status='claimed' AND claimed_by=$2 AND lease_until>=NOW()",
                tables.projected_lane_messages
            ),
            &[&handoff.message_id, &handoff.claimed_by],
        )
        .await
        .map_err(backend)?;
    if acknowledged != 1 {
        return Err(PostgresStoreError::RecordedRuntimeConflict(format!(
            "worker lane message {:?} claim changed before acknowledgement",
            handoff.message_id
        )));
    }
    Ok(result)
}
