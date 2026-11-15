use super::super::*;

pub(crate) async fn apply_recorded_runtime_transition_inner<C>(
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
    let assistant_turn_node_id = (reduced.state.status
        == kogwistar_runtime::RecordedRunStatus::Completed)
        .then(|| {
            reduced
                .state
                .state
                .get("agentic_answering_result")
                .and_then(|result| result.get("assistant_turn_node_id"))
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned)
        })
        .flatten()
        .or_else(|| prior_run.and_then(|run| run.assistant_turn_node_id.clone()));
    update_server_run(
        client,
        tables,
        &transition.run_id,
        ServerRunUpdate {
            status: reduced.server_status.clone(),
            assistant_turn_node_id,
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

pub(crate) async fn replay_cursor<C>(
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

pub(crate) async fn set_replay_cursor_legacy<C>(
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

pub(crate) async fn advisory_lock<C>(client: &C, key: &str, seed: i64) -> PostgresStoreResult<()>
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

pub(crate) fn cursor_lock_key(namespace: &str, consumer: &str) -> String {
    format!("{namespace}\u{1f}{consumer}")
}

pub(crate) fn projection_lock_key(namespace: &str, key: &str) -> String {
    format!("{namespace}\u{1f}{key}")
}

pub(crate) fn raw_event_from_row(row: &Row) -> PostgresStoreResult<RawEntityEvent> {
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
