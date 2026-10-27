use super::super::*;

pub(crate) fn parse_runtime_event_payload(
    payload_json: &str,
) -> Option<PersistedRecordedTransition> {
    serde_json::from_str::<PersistedRecordedTransition>(payload_json)
        .ok()
        .filter(|value| value.contract_version == RECORDED_RUNTIME_CONTRACT_VERSION)
}

pub(crate) async fn recorded_runtime_events<C>(
    client: &C,
    tables: &Tables,
    run_id: Option<&str>,
) -> PostgresStoreResult<Vec<ServerRunEvent>>
where
    C: GenericClient + Sync,
{
    let rows = client
        .query(
            &format!(
                "SELECT seq,run_id,event_type,payload_json,created_at_ms FROM {} \
                 WHERE event_type=$1 AND ($2::TEXT IS NULL OR run_id=$2) ORDER BY seq ASC",
                tables.server_run_events
            ),
            &[&RECORDED_RUNTIME_EVENT_TYPE, &run_id],
        )
        .await
        .map_err(backend)?;
    rows.iter().map(server_run_event_from_row).collect()
}

pub(crate) async fn latest_server_run_event_seq<C>(
    client: &C,
    tables: &Tables,
    run_id: &str,
) -> PostgresStoreResult<i64>
where
    C: GenericClient + Sync,
{
    client
        .query_one(
            &format!(
                "SELECT COALESCE(MAX(seq),0) FROM {} WHERE run_id=$1",
                tables.server_run_events
            ),
            &[&run_id],
        )
        .await
        .map_err(backend)?
        .try_get(0)
        .map_err(backend)
}

pub(crate) fn runtime_json_object<T: serde::Serialize>(
    value: &T,
) -> PostgresStoreResult<Map<String, Value>> {
    serde_json::to_value(value)?
        .as_object()
        .cloned()
        .ok_or_else(|| {
            PostgresStoreError::TransactionAborted(
                "runtime projection payload is not an object".to_owned(),
            )
        })
}

pub(crate) async fn store_runtime_current_state<C>(
    client: &C,
    tables: &Tables,
    persisted: &PersistedRecordedTransition,
    event_seq: i64,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    replace_named_projection(
        client,
        tables,
        RUNTIME_CURRENT_STATE_NAMESPACE,
        &persisted.reduced.state.run_id,
        runtime_projection_write(
            runtime_json_object(&persisted.reduced.state)?,
            event_seq,
            "ready".to_owned(),
        ),
    )
    .await
}

pub(crate) async fn recorded_transition_by_id<C>(
    client: &C,
    tables: &Tables,
    transition_id: &str,
) -> PostgresStoreResult<Option<(PersistedRecordedTransition, i64)>>
where
    C: GenericClient + Sync,
{
    client
        .query_opt(
            &format!(
                "SELECT payload_json, seq FROM {} \
                 WHERE event_type='workflow.recorded_transition.v1' \
                 AND payload_json::jsonb->>'transition_id'=$1 LIMIT 1",
                tables.server_run_events
            ),
            &[&transition_id],
        )
        .await
        .map_err(backend)?
        .map(|row| {
            let payload_json: String = row.try_get(0).map_err(backend)?;
            let seq: i64 = row.try_get(1).map_err(backend)?;
            Ok((serde_json::from_str(&payload_json)?, seq))
        })
        .transpose()
}

pub(crate) async fn read_recorded_runtime_state<C>(
    client: &C,
    tables: &Tables,
    run_id: &str,
    workflow_id: &str,
    conversation_id: &str,
) -> PostgresStoreResult<Option<RecordedRuntimeState>>
where
    C: GenericClient + Sync,
{
    let Some(run) = server_run(client, tables, run_id).await? else {
        return Ok(None);
    };
    if run.workflow_id != workflow_id || run.conversation_id != conversation_id {
        return Err(PostgresStoreError::RecordedRuntimeConflict(format!(
            "run identity differs for {run_id:?}"
        )));
    }
    if let Some(row) =
        named_projection(client, tables, RUNTIME_CURRENT_STATE_NAMESPACE, run_id).await?
        && row.materialization_status == "ready"
    {
        let has_newer_recorded_event: bool = client
            .query_one(
                &format!(
                    "SELECT EXISTS(
                        SELECT 1 FROM {} WHERE run_id=$1 AND event_type=$2 AND seq>$3
                     )",
                    tables.server_run_events
                ),
                &[
                    &run_id,
                    &RECORDED_RUNTIME_EVENT_TYPE,
                    &row.last_authoritative_seq,
                ],
            )
            .await
            .map_err(backend)?
            .try_get(0)
            .map_err(backend)?;
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
    let mut latest = None;
    for event in recorded_runtime_events(client, tables, Some(run_id)).await? {
        if let Some(payload) = parse_runtime_event_payload(&event.payload_json)
            && payload.reduced.state.run_id == run_id
            && payload.reduced.state.workflow_id == workflow_id
            && payload.reduced.state.conversation_id == conversation_id
        {
            latest = Some(payload.reduced.state);
        }
    }
    Ok(latest)
}

pub(crate) async fn apply_recorded_runtime_transition<C>(
    client: &C,
    tables: &Tables,
    transition: RecordedRuntimeTransition,
    abort_after_writes: bool,
) -> PostgresStoreResult<RecordedTransitionResult>
where
    C: GenericClient + Sync,
{
    apply_recorded_runtime_transition_inner(
        client,
        tables,
        transition,
        None,
        None,
        abort_after_writes,
    )
    .await
}

pub(crate) async fn apply_claimed_recorded_runtime_transition<C>(
    client: &C,
    tables: &Tables,
    handoff: RecordedWorkerHandoff,
    transition: RecordedRuntimeTransition,
    abort_after_writes: bool,
) -> PostgresStoreResult<RecordedTransitionResult>
where
    C: GenericClient + Sync,
{
    validate_worker_handoff(&handoff)?;
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
    validate_worker_lane_identity(&lane, &handoff, &transition)?;

    if lane.status == "completed" {
        return retry_completed_worker_handoff(client, tables, &handoff, &transition).await;
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

    let result = apply_recorded_runtime_transition_inner(
        client,
        tables,
        transition,
        Some(handoff.clone()),
        None,
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
    if abort_after_writes {
        return Err(PostgresStoreError::TransactionAborted(
            "requested after recorded runtime worker handoff writes".to_owned(),
        ));
    }
    Ok(result)
}

pub(crate) fn validate_worker_handoff(handoff: &RecordedWorkerHandoff) -> PostgresStoreResult<()> {
    for (field, value) in [
        ("message_id", handoff.message_id.as_str()),
        ("claimed_by", handoff.claimed_by.as_str()),
        ("run_id", handoff.run_id.as_str()),
        ("step_id", handoff.step_id.as_str()),
        ("correlation_id", handoff.correlation_id.as_str()),
    ] {
        if value.is_empty() {
            return Err(PostgresStoreError::RecordedRuntimeConflict(format!(
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
) -> PostgresStoreResult<()> {
    if handoff.run_id != transition.run_id
        || transition.node_id.as_deref() != Some(handoff.step_id.as_str())
        || lane.run_id.as_deref() != Some(handoff.run_id.as_str())
        || lane.step_id.as_deref() != Some(handoff.step_id.as_str())
        || lane.correlation_id.as_deref() != Some(handoff.correlation_id.as_str())
    {
        return Err(PostgresStoreError::RecordedRuntimeConflict(format!(
            "worker lane message {:?} run/step/correlation identity differs",
            handoff.message_id
        )));
    }
    Ok(())
}

pub(crate) async fn retry_completed_worker_handoff<C>(
    client: &C,
    tables: &Tables,
    handoff: &RecordedWorkerHandoff,
    transition: &RecordedRuntimeTransition,
) -> PostgresStoreResult<RecordedTransitionResult>
where
    C: GenericClient + Sync,
{
    let request_digest = transition_digest(transition)?;
    if let Some((payload, event_seq)) =
        recorded_transition_by_id(client, tables, &transition.transition_id).await?
    {
        if payload.request_digest != request_digest
            || payload.worker_handoff.as_ref() != Some(handoff)
        {
            return Err(PostgresStoreError::RecordedRuntimeConflict(format!(
                "worker handoff {:?} retried with different result",
                handoff.message_id
            )));
        }
        return Ok(payload.result(event_seq, true));
    }
    Err(PostgresStoreError::RecordedRuntimeConflict(format!(
        "completed worker lane message {:?} has no matching recorded result",
        handoff.message_id
    )))
}
