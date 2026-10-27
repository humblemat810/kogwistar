use super::*;

pub(super) async fn put_workflow_design_snapshot<C>(
    client: &C,
    tables: &Tables,
    workflow_id: &str,
    snapshot: WorkflowDesignSnapshotWrite,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    let schema_version = pg_schema_version(snapshot.schema_version)?;
    let created_at_ms = unix_epoch_millis();
    client
        .execute(
            &format!(
                "INSERT INTO {}(workflow_id, version, seq, payload_json, schema_version, created_at_ms) \
                 VALUES ($1,$2,$3,$4,$5,$6) \
                 ON CONFLICT(workflow_id,version) DO UPDATE SET \
                 seq=EXCLUDED.seq,payload_json=EXCLUDED.payload_json, \
                 schema_version=EXCLUDED.schema_version,created_at_ms=EXCLUDED.created_at_ms",
                tables.workflow_design_snapshots
            ),
            &[
                &workflow_id,
                &snapshot.version,
                &snapshot.seq,
                &snapshot.payload_json,
                &schema_version,
                &created_at_ms,
            ],
        )
        .await
        .map_err(backend)?;
    Ok(())
}

pub(super) async fn workflow_design_snapshot<C>(
    client: &C,
    tables: &Tables,
    workflow_id: &str,
    max_version: i64,
    schema_version: i64,
) -> PostgresStoreResult<Option<WorkflowDesignSnapshot>>
where
    C: GenericClient + Sync,
{
    let schema_version = pg_schema_version(schema_version)?;
    client
        .query_opt(
            &format!(
                "SELECT workflow_id,version,seq,payload_json,schema_version,created_at_ms \
                 FROM {} WHERE workflow_id=$1 AND version <= $2 AND schema_version=$3 \
                 ORDER BY version DESC LIMIT 1",
                tables.workflow_design_snapshots
            ),
            &[&workflow_id, &max_version, &schema_version],
        )
        .await
        .map_err(backend)?
        .as_ref()
        .map(workflow_design_snapshot_from_row)
        .transpose()
}

pub(super) async fn clear_workflow_design_snapshots<C>(
    client: &C,
    tables: &Tables,
    workflow_id: &str,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    client
        .execute(
            &format!(
                "DELETE FROM {} WHERE workflow_id = $1",
                tables.workflow_design_snapshots
            ),
            &[&workflow_id],
        )
        .await
        .map_err(backend)?;
    Ok(())
}

pub(super) async fn put_workflow_design_delta<C>(
    client: &C,
    tables: &Tables,
    workflow_id: &str,
    delta: WorkflowDesignDeltaWrite,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    let schema_version = pg_schema_version(delta.schema_version)?;
    let created_at_ms = unix_epoch_millis();
    client
        .execute(
            &format!(
                "INSERT INTO {}(workflow_id,version,prev_version,target_seq,forward_json,inverse_json,schema_version,created_at_ms) \
                 VALUES ($1,$2,$3,$4,$5,$6,$7,$8) \
                 ON CONFLICT(workflow_id,version) DO UPDATE SET \
                 prev_version=EXCLUDED.prev_version,target_seq=EXCLUDED.target_seq, \
                 forward_json=EXCLUDED.forward_json,inverse_json=EXCLUDED.inverse_json, \
                 schema_version=EXCLUDED.schema_version,created_at_ms=EXCLUDED.created_at_ms",
                tables.workflow_design_version_deltas
            ),
            &[
                &workflow_id,
                &delta.version,
                &delta.prev_version,
                &delta.target_seq,
                &delta.forward_json,
                &delta.inverse_json,
                &schema_version,
                &created_at_ms,
            ],
        )
        .await
        .map_err(backend)?;
    Ok(())
}

pub(super) async fn workflow_design_delta<C>(
    client: &C,
    tables: &Tables,
    workflow_id: &str,
    version: i64,
    schema_version: i64,
) -> PostgresStoreResult<Option<WorkflowDesignDelta>>
where
    C: GenericClient + Sync,
{
    let schema_version = pg_schema_version(schema_version)?;
    client
        .query_opt(
            &format!(
                "SELECT workflow_id,version,prev_version,target_seq,forward_json,inverse_json,schema_version,created_at_ms \
                 FROM {} WHERE workflow_id=$1 AND version=$2 AND schema_version=$3",
                tables.workflow_design_version_deltas
            ),
            &[&workflow_id, &version, &schema_version],
        )
        .await
        .map_err(backend)?
        .as_ref()
        .map(workflow_design_delta_from_row)
        .transpose()
}

pub(super) async fn clear_workflow_design_deltas<C>(
    client: &C,
    tables: &Tables,
    workflow_id: &str,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    client
        .execute(
            &format!(
                "DELETE FROM {} WHERE workflow_id = $1",
                tables.workflow_design_version_deltas
            ),
            &[&workflow_id],
        )
        .await
        .map_err(backend)?;
    Ok(())
}

pub(super) async fn create_server_run<C>(
    client: &C,
    tables: &Tables,
    run: ServerRunCreate,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    let now = unix_epoch_millis();
    client.execute(&format!("INSERT INTO {}(run_id,conversation_id,workflow_id,user_id,user_turn_node_id,assistant_turn_node_id,status,cancel_requested,result_json,error_json,created_at_ms,updated_at_ms,started_at_ms,finished_at_ms) VALUES ($1,$2,$3,$4,$5,NULL,$6,0,NULL,NULL,$7,$7,NULL,NULL)", tables.server_runs), &[&run.run_id,&run.conversation_id,&run.workflow_id,&run.user_id,&run.user_turn_node_id,&run.status,&now]).await.map_err(backend)?;
    Ok(())
}
pub(super) async fn server_run<C>(
    client: &C,
    tables: &Tables,
    run_id: &str,
) -> PostgresStoreResult<Option<ServerRun>>
where
    C: GenericClient + Sync,
{
    client.query_opt(&format!("SELECT run_id,conversation_id,workflow_id,user_id,user_turn_node_id,assistant_turn_node_id,status,cancel_requested,result_json,error_json,created_at_ms,updated_at_ms,started_at_ms,finished_at_ms FROM {} WHERE run_id=$1", tables.server_runs), &[&run_id]).await.map_err(backend)?.as_ref().map(server_run_from_row).transpose()
}
pub(super) async fn server_runs<C>(
    client: &C,
    tables: &Tables,
    status: Option<&str>,
    workflow_id: Option<&str>,
    conversation_id: Option<&str>,
    limit: usize,
) -> PostgresStoreResult<Vec<ServerRun>>
where
    C: GenericClient + Sync,
{
    let limit = i64::try_from(limit).unwrap_or(i64::MAX);
    client.query(&format!("SELECT run_id,conversation_id,workflow_id,user_id,user_turn_node_id,assistant_turn_node_id,status,cancel_requested,result_json,error_json,created_at_ms,updated_at_ms,started_at_ms,finished_at_ms FROM {} WHERE ($1::TEXT IS NULL OR status=$1) AND ($2::TEXT IS NULL OR workflow_id=$2) AND ($3::TEXT IS NULL OR conversation_id=$3) ORDER BY created_at_ms DESC,run_id DESC LIMIT $4", tables.server_runs), &[&status,&workflow_id,&conversation_id,&limit]).await.map_err(backend)?.iter().map(server_run_from_row).collect()
}
pub(super) async fn append_server_run_event<C>(
    client: &C,
    tables: &Tables,
    run_id: &str,
    event_type: &str,
    payload_json: String,
) -> PostgresStoreResult<ServerRunEvent>
where
    C: GenericClient + Sync,
{
    let created_at_ms = unix_epoch_millis();
    let row=client.query_one(&format!("INSERT INTO {}(run_id,event_type,payload_json,created_at_ms) VALUES ($1,$2,$3,$4) RETURNING seq", tables.server_run_events), &[&run_id,&event_type,&payload_json,&created_at_ms]).await.map_err(backend)?;
    Ok(ServerRunEvent {
        seq: row.try_get(0).map_err(backend)?,
        run_id: run_id.to_owned(),
        event_type: event_type.to_owned(),
        payload_json,
        created_at_ms,
    })
}
pub(super) async fn server_run_events<C>(
    client: &C,
    tables: &Tables,
    run_id: &str,
    after_seq: i64,
    limit: usize,
) -> PostgresStoreResult<Vec<ServerRunEvent>>
where
    C: GenericClient + Sync,
{
    let limit = i64::try_from(limit).unwrap_or(i64::MAX);
    client.query(&format!("SELECT seq,run_id,event_type,payload_json,created_at_ms FROM {} WHERE run_id=$1 AND seq>$2 ORDER BY seq ASC LIMIT $3", tables.server_run_events), &[&run_id,&after_seq,&limit]).await.map_err(backend)?.iter().map(server_run_event_from_row).collect()
}
pub(super) async fn update_server_run<C>(
    client: &C,
    tables: &Tables,
    run_id: &str,
    update: ServerRunUpdate,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    let now = unix_epoch_millis();
    let cancel_requested = update.cancel_requested.map(i32::from);
    client.execute(&format!("UPDATE {} SET status=$1,assistant_turn_node_id=$2,result_json=$3,error_json=$4,started_at_ms=$5,finished_at_ms=$6,cancel_requested=COALESCE($7,cancel_requested),updated_at_ms=$8 WHERE run_id=$9", tables.server_runs), &[&update.status,&update.assistant_turn_node_id,&update.result_json,&update.error_json,&update.started_at_ms,&update.finished_at_ms,&cancel_requested,&now,&run_id]).await.map_err(backend)?;
    Ok(())
}
pub(super) async fn request_server_run_cancel<C>(
    client: &C,
    tables: &Tables,
    run_id: &str,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    advisory_lock(client, &format!("recorded-run\u{1f}{run_id}"), 42).await?;
    let Some(run) = server_run(client, tables, run_id).await? else {
        return Ok(());
    };
    let mut current = read_recorded_runtime_state(
        client,
        tables,
        run_id,
        &run.workflow_id,
        &run.conversation_id,
    )
    .await?;
    if current.is_none() {
        let payload_json = client
            .query_opt(
                &format!(
                    "SELECT payload_json FROM {} WHERE run_id=$1 AND msg_type='workflow.run.execute' \
                     ORDER BY created_at LIMIT 1",
                    tables.projected_lane_messages
                ),
                &[&run_id],
            )
            .await
            .map_err(backend)?
            .and_then(|row| row.try_get::<_, Option<String>>(0).ok().flatten());
        if let Some(payload_json) = payload_json {
            let payload: Value = serde_json::from_str(&payload_json)?;
            let expected_event_seq = latest_server_run_event_seq(client, tables, run_id).await?;
            apply_recorded_runtime_transition_inner(
                client,
                tables,
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
                None,
                false,
            )
            .await?;
            current = read_recorded_runtime_state(
                client,
                tables,
                run_id,
                &run.workflow_id,
                &run.conversation_id,
            )
            .await?;
        }
    }
    if let Some(state) = current
        && !state.status.is_terminal()
    {
        let expected_event_seq = latest_server_run_event_seq(client, tables, run_id).await?;
        apply_recorded_runtime_transition_inner(
            client,
            tables,
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
            None,
            false,
        )
        .await?;
    } else {
        let now = unix_epoch_millis();
        client.execute(&format!("UPDATE {} SET cancel_requested=1,status=CASE WHEN status IN ('cancelled','failed','succeeded') THEN status ELSE 'cancelling' END,updated_at_ms=$1 WHERE run_id=$2", tables.server_runs), &[&now,&run_id]).await.map_err(backend)?;
    }
    client
        .execute(
            &format!(
                "UPDATE {} SET status='cancelled',claimed_by=NULL,lease_until=NULL \
                 WHERE run_id=$1 AND status NOT IN ('completed','failed','cancelled','dead-letter')",
                tables.projected_lane_messages
            ),
            &[&run_id],
        )
        .await
        .map_err(backend)?;
    Ok(())
}

const RECORDED_RUNTIME_EVENT_TYPE: &str = "workflow.recorded_transition.v1";

pub(super) fn parse_runtime_event_payload(
    payload_json: &str,
) -> Option<PersistedRecordedTransition> {
    serde_json::from_str::<PersistedRecordedTransition>(payload_json)
        .ok()
        .filter(|value| value.contract_version == RECORDED_RUNTIME_CONTRACT_VERSION)
}

pub(super) async fn recorded_runtime_events<C>(
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

pub(super) async fn latest_server_run_event_seq<C>(
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

pub(super) fn runtime_json_object<T: serde::Serialize>(
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

pub(super) async fn store_runtime_current_state<C>(
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

pub(super) async fn recorded_transition_by_id<C>(
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

pub(super) async fn read_recorded_runtime_state<C>(
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

pub(super) async fn apply_recorded_runtime_transition<C>(
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

pub(super) async fn apply_claimed_recorded_runtime_transition<C>(
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

pub(super) fn validate_worker_handoff(handoff: &RecordedWorkerHandoff) -> PostgresStoreResult<()> {
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

pub(super) fn validate_worker_lane_identity(
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

pub(super) async fn retry_completed_worker_handoff<C>(
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

pub(super) async fn apply_claimed_recorded_worker_effect<C>(
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

pub(super) async fn apply_recorded_runtime_transition_inner<C>(
    client: &C,
    tables: &Tables,
    transition: RecordedRuntimeTransition,
    worker_handoff: Option<RecordedWorkerHandoff>,
    persisted_worker_effect_digest: Option<String>,
    abort_after_writes: bool,
) -> PostgresStoreResult<RecordedTransitionResult>
where
    C: GenericClient + Sync,
{
    let request_digest = transition_digest(&transition)?;
    // Serialize global transition-id reuse and per-run sequence/CAS checks.
    advisory_lock(
        client,
        &format!("recorded-transition\u{1f}{}", transition.transition_id),
        41,
    )
    .await?;
    advisory_lock(
        client,
        &format!("recorded-run\u{1f}{}", transition.run_id),
        42,
    )
    .await?;

    if let Some((payload, event_seq)) =
        recorded_transition_by_id(client, tables, &transition.transition_id).await?
    {
        if payload.request_digest != request_digest || payload.worker_handoff != worker_handoff {
            return Err(PostgresStoreError::RecordedRuntimeConflict(format!(
                "transition_id {:?} reused with different payload",
                transition.transition_id
            )));
        }
        return Ok(payload.result(event_seq, true));
    }

    let existing_run = server_run(client, tables, &transition.run_id).await?;
    if let Some(run) = &existing_run
        && (run.workflow_id != transition.workflow_id
            || run.conversation_id != transition.conversation_id)
    {
        return Err(PostgresStoreError::RecordedRuntimeConflict(format!(
            "run identity differs for {:?}",
            transition.run_id
        )));
    }
    let current = read_recorded_runtime_state(
        client,
        tables,
        &transition.run_id,
        &transition.workflow_id,
        &transition.conversation_id,
    )
    .await?;
    let current_event_seq = latest_server_run_event_seq(client, tables, &transition.run_id).await?;
    if current_event_seq != transition.expected_event_seq {
        return Err(PostgresStoreError::RecordedRuntimeConflict(format!(
            "expected_event_seq {}, current {}",
            transition.expected_event_seq, current_event_seq
        )));
    }
    let reduced = reduce_recorded_transition(current.as_ref(), &transition)?;

    if current.is_none() && existing_run.is_none() {
        let user_turn_node_id = transition.user_turn_node_id.clone().ok_or_else(|| {
            PostgresStoreError::RecordedRuntimeConflict(
                "start transition requires user_turn_node_id".to_owned(),
            )
        })?;
        create_server_run(
            client,
            tables,
            ServerRunCreate {
                run_id: transition.run_id.clone(),
                conversation_id: transition.conversation_id.clone(),
                workflow_id: transition.workflow_id.clone(),
                user_id: transition.user_id.clone(),
                user_turn_node_id,
                status: reduced.server_status.clone(),
            },
        )
        .await?;
    }

    let persisted = PersistedRecordedTransition {
        contract_version: RECORDED_RUNTIME_CONTRACT_VERSION,
        transition_id: transition.transition_id.clone(),
        request_digest,
        worker_handoff,
        worker_effect_digest: persisted_worker_effect_digest,
        reduced: reduced.clone(),
    };
    let event = append_server_run_event(
        client,
        tables,
        &transition.run_id,
        RECORDED_RUNTIME_EVENT_TYPE,
        serde_json::to_string(&persisted)?,
    )
    .await?;
    let event_seq = event.seq;
    store_runtime_current_state(client, tables, &persisted, event_seq).await?;
    replace_named_projection(
        client,
        tables,
        &runtime_checkpoint_namespace(&transition.conversation_id),
        &transition.run_id,
        runtime_projection_write(
            reduced.checkpoint.clone(),
            event_seq,
            reduced.state.status.server_status().to_owned(),
        ),
    )
    .await?;
    let status_materialization = match reduced.state.status {
        kogwistar_runtime::RecordedRunStatus::Completed => "completed",
        kogwistar_runtime::RecordedRunStatus::Failed => "failed",
        kogwistar_runtime::RecordedRunStatus::Cancelled => "cancelled",
        kogwistar_runtime::RecordedRunStatus::Suspended => "suspended",
        kogwistar_runtime::RecordedRunStatus::Running => "running",
    }
    .to_owned();
    replace_named_projection(
        client,
        tables,
        &runtime_status_namespace(&transition.conversation_id),
        &transition.run_id,
        runtime_projection_write(
            reduced.run_status.clone(),
            event_seq,
            status_materialization,
        ),
    )
    .await?;
    let prior_run = existing_run.as_ref();
    update_server_run(
        client,
        tables,
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
    )
    .await?;
    if abort_after_writes {
        return Err(PostgresStoreError::TransactionAborted(
            "requested after recorded runtime transition writes".to_owned(),
        ));
    }
    Ok(persisted.result(event_seq, false))
}

pub(super) async fn replay_cursor<C>(
    client: &C,
    tables: &Tables,
    namespace: &str,
    consumer: &str,
) -> PostgresStoreResult<ReplayCursor>
where
    C: GenericClient + Sync,
{
    let row = client
        .query_opt(
            &format!(
                "SELECT last_seq FROM {} WHERE namespace = $1 AND consumer = $2",
                tables.replay_cursors
            ),
            &[&namespace, &consumer],
        )
        .await
        .map_err(backend)?;
    Ok(ReplayCursor {
        namespace: namespace.to_owned(),
        consumer: consumer.to_owned(),
        last_seq: row
            .as_ref()
            .map(|row| row.try_get(0).map_err(backend))
            .transpose()?
            .unwrap_or(0),
    })
}

pub(super) async fn set_replay_cursor_legacy<C>(
    client: &C,
    tables: &Tables,
    namespace: &str,
    consumer: &str,
    last_seq: i64,
) -> PostgresStoreResult<ReplayCursor>
where
    C: GenericClient + Sync,
{
    client
        .execute(
            &format!(
                "INSERT INTO {}(namespace, consumer, last_seq, updated_at) VALUES ($1, $2, $3, NOW()) \
                 ON CONFLICT(namespace, consumer) DO UPDATE \
                 SET last_seq = EXCLUDED.last_seq, updated_at = NOW()",
                tables.replay_cursors
            ),
            &[&namespace, &consumer, &last_seq],
        )
        .await
        .map_err(backend)?;
    Ok(ReplayCursor {
        namespace: namespace.to_owned(),
        consumer: consumer.to_owned(),
        last_seq,
    })
}

pub(super) async fn advisory_lock<C>(client: &C, key: &str, seed: i64) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    client
        .query_one(
            "SELECT pg_advisory_xact_lock(hashtextextended($1, $2))",
            &[&key, &seed],
        )
        .await
        .map(|_| ())
        .map_err(backend)
}

pub(super) fn cursor_lock_key(namespace: &str, consumer: &str) -> String {
    format!("{namespace}\u{1f}{consumer}")
}

pub(super) fn projection_lock_key(namespace: &str, key: &str) -> String {
    format!("{namespace}\u{1f}{key}")
}

pub(super) fn raw_event_from_row(row: &Row) -> PostgresStoreResult<RawEntityEvent> {
    Ok(RawEntityEvent {
        namespace: row.try_get(0).map_err(backend)?,
        seq: row.try_get(1).map_err(backend)?,
        event_id: row.try_get(2).map_err(backend)?,
        entity_kind: row.try_get(3).map_err(backend)?,
        entity_id: row.try_get(4).map_err(backend)?,
        op: row.try_get(5).map_err(backend)?,
        payload_json: row.try_get(6).map_err(backend)?,
        created_at: row.try_get(7).map_err(backend)?,
    })
}
