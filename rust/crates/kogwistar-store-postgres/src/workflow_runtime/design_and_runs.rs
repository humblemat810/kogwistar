use super::super::*;

pub(crate) async fn put_workflow_design_snapshot<C>(
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

pub(crate) async fn workflow_design_snapshot<C>(
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

pub(crate) async fn clear_workflow_design_snapshots<C>(
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

pub(crate) async fn put_workflow_design_delta<C>(
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

pub(crate) async fn workflow_design_delta<C>(
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

pub(crate) async fn clear_workflow_design_deltas<C>(
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

pub(crate) async fn create_server_run<C>(
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
pub(crate) async fn server_run<C>(
    client: &C,
    tables: &Tables,
    run_id: &str,
) -> PostgresStoreResult<Option<ServerRun>>
where
    C: GenericClient + Sync,
{
    client.query_opt(&format!("SELECT run_id,conversation_id,workflow_id,user_id,user_turn_node_id,assistant_turn_node_id,status,cancel_requested,result_json,error_json,created_at_ms,updated_at_ms,started_at_ms,finished_at_ms FROM {} WHERE run_id=$1", tables.server_runs), &[&run_id]).await.map_err(backend)?.as_ref().map(server_run_from_row).transpose()
}
pub(crate) async fn server_runs<C>(
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
pub(crate) async fn append_server_run_event<C>(
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
pub(crate) async fn server_run_events<C>(
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
pub(crate) async fn update_server_run<C>(
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
pub(crate) async fn request_server_run_cancel<C>(
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

pub(crate) const RECORDED_RUNTIME_EVENT_TYPE: &str = "workflow.recorded_transition.v1";
