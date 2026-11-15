use super::*;

pub(super) async fn event_by_id<C>(
    client: &C,
    tables: &Tables,
    event_id: &str,
) -> PostgresStoreResult<Option<RawEntityEvent>>
where
    C: GenericClient + Sync,
{
    client
        .query_opt(
            &format!(
                "SELECT namespace, seq, event_id, entity_kind, entity_id, op, payload_json, created_at::TEXT \
                 FROM {} WHERE event_id = $1",
                tables.entity_events
            ),
            &[&event_id],
        )
        .await
        .map_err(backend)?
        .as_ref()
        .map(raw_event_from_row)
        .transpose()
}

pub(super) async fn replay_raw_events<C>(
    client: &C,
    tables: &Tables,
    namespace: &str,
    after_seq: i64,
    limit: usize,
) -> PostgresStoreResult<Vec<RawEntityEvent>>
where
    C: GenericClient + Sync,
{
    let limit = i64::try_from(limit).unwrap_or(i64::MAX);
    client
        .query(
            &format!(
                "SELECT namespace, seq, event_id, entity_kind, entity_id, op, payload_json, created_at::TEXT \
                 FROM {} WHERE namespace = $1 AND seq > $2 ORDER BY seq ASC LIMIT $3",
                tables.entity_events
            ),
            &[&namespace, &after_seq, &limit],
        )
        .await
        .map_err(backend)?
        .iter()
        .map(raw_event_from_row)
        .collect()
}

pub(super) fn recovery_projection_write(
    projection: &EntityProjection,
    latest_authoritative_seq: i64,
) -> NamedProjectionWrite {
    NamedProjectionWrite {
        payload: projection.payload(),
        last_authoritative_seq: latest_authoritative_seq,
        last_materialized_seq: projection.last_seq(),
        projection_schema_version: 1,
        materialization_status: "ready".to_owned(),
    }
}

pub(super) fn recovery_report(
    projection: &EntityProjection,
    processed_count: usize,
    prior_cursor: i64,
    latest_authoritative_seq: i64,
) -> EntityRecoveryReport {
    EntityRecoveryReport {
        processed_count,
        prior_cursor,
        new_cursor: projection.last_seq(),
        latest_authoritative_seq,
        caught_up: projection.last_seq() >= latest_authoritative_seq,
        canonical_payload: projection.canonical_payload(),
        digest: projection.digest(),
    }
}

pub(super) async fn recover_entity_projection_uow(
    uow: &mut PostgresUnitOfWork<'_>,
    request: &EntityRecoveryRequest,
) -> PostgresStoreResult<EntityRecoveryReport> {
    validate_entity_recovery_request(request)?;
    // Serialize each consumer cursor and its dependent projection replacement.
    advisory_lock(
        &uow.transaction,
        &cursor_lock_key(&request.namespace, &request.consumer),
        2,
    )
    .await?;
    advisory_lock(
        &uow.transaction,
        &projection_lock_key(&request.projection_namespace, &request.projection_key),
        3,
    )
    .await?;
    let prior_cursor = uow
        .replay_cursor(&request.namespace, &request.consumer)
        .await?
        .last_seq;
    let current = named_projection(
        &uow.transaction,
        &uow.tables,
        &request.projection_namespace,
        &request.projection_key,
    )
    .await?;
    let mut projection = EntityProjection::from_payload(current.as_ref().map(|row| &row.payload))?;
    if projection.last_seq() != prior_cursor {
        return Err(StoreError::InvalidEntityEventPayload {
            message: "recovery projection cursor does not match durable replay cursor".to_owned(),
        }
        .into());
    }
    let events = uow
        .replay_raw_events(&request.namespace, prior_cursor, request.batch_limit)
        .await?;
    let latest_authoritative_seq = uow.latest_retained_event_seq(&request.namespace).await?;
    for raw in &events {
        projection.fold(&raw_to_entity_event(raw.clone())?)?;
    }
    let projection_changed = !events.is_empty() || current.is_none();
    if projection_changed {
        uow.replace_named_projection(
            &request.projection_namespace,
            &request.projection_key,
            recovery_projection_write(&projection, latest_authoritative_seq),
        )
        .await?;
    }
    if request.abort_after_projection {
        return Err(PostgresStoreError::TransactionAborted(
            "requested after entity recovery projection write".to_owned(),
        ));
    }
    if projection_changed || projection.last_seq() != prior_cursor {
        uow.strict_advance_replay_cursor(
            &request.namespace,
            &request.consumer,
            projection.last_seq(),
        )
        .await?;
    }
    Ok(recovery_report(
        &projection,
        events.len(),
        prior_cursor,
        latest_authoritative_seq,
    ))
}

pub(super) async fn rebuild_entity_projection_uow(
    uow: &mut PostgresUnitOfWork<'_>,
    request: &EntityRebuildRequest,
) -> PostgresStoreResult<EntityRecoveryReport> {
    validate_entity_rebuild_request(request)?;
    advisory_lock(
        &uow.transaction,
        &cursor_lock_key(&request.namespace, &request.consumer),
        2,
    )
    .await?;
    advisory_lock(
        &uow.transaction,
        &projection_lock_key(&request.projection_namespace, &request.projection_key),
        3,
    )
    .await?;
    let prior_cursor = uow
        .replay_cursor(&request.namespace, &request.consumer)
        .await?
        .last_seq;
    let events = uow
        .replay_raw_events(&request.namespace, 0, usize::MAX)
        .await?;
    let latest_authoritative_seq = uow.latest_retained_event_seq(&request.namespace).await?;
    let mut projection = EntityProjection::empty();
    for raw in &events {
        projection.fold(&raw_to_entity_event(raw.clone())?)?;
    }
    uow.replace_named_projection(
        &request.projection_namespace,
        &request.projection_key,
        recovery_projection_write(&projection, latest_authoritative_seq),
    )
    .await?;
    if request.abort_after_projection {
        return Err(PostgresStoreError::TransactionAborted(
            "requested after entity rebuild projection write".to_owned(),
        ));
    }
    // Reset is explicit operator semantics; complete projection and cursor are
    // still one UoW. Incremental recovery remains strict and monotonic.
    uow.set_replay_cursor_legacy(&request.namespace, &request.consumer, projection.last_seq())
        .await?;
    Ok(recovery_report(
        &projection,
        events.len(),
        prior_cursor,
        latest_authoritative_seq,
    ))
}

pub(super) async fn latest_retained_event_seq<C>(
    client: &C,
    tables: &Tables,
    namespace: &str,
) -> PostgresStoreResult<i64>
where
    C: GenericClient + Sync,
{
    let row = client
        .query_one(
            &format!(
                "SELECT COALESCE(MAX(seq), 0) FROM {} WHERE namespace = $1",
                tables.entity_events
            ),
            &[&namespace],
        )
        .await
        .map_err(backend)?;
    row.try_get(0).map_err(backend)
}

pub(super) async fn prune_entity_events_after<C>(
    client: &C,
    tables: &Tables,
    namespace: &str,
    to_seq: i64,
) -> PostgresStoreResult<u64>
where
    C: GenericClient + Sync,
{
    client
        .execute(
            &format!(
                "DELETE FROM {} WHERE namespace = $1 AND seq > $2",
                tables.entity_events
            ),
            &[&namespace, &to_seq],
        )
        .await
        .map_err(backend)
}

pub(super) fn pg_schema_version(value: i64) -> PostgresStoreResult<i32> {
    i32::try_from(value).map_err(|_| {
        PostgresStoreError::Backend(
            "workflow design schema version exceeds PostgreSQL INTEGER".to_owned(),
        )
    })
}
