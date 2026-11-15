use super::*;

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

pub(super) fn recover_entity_projection_uow(
    uow: &mut SqliteUnitOfWork<'_>,
    request: &EntityRecoveryRequest,
) -> SqliteStoreResult<EntityRecoveryReport> {
    validate_entity_recovery_request(request)?;
    let prior_cursor = uow
        .replay_cursor(&request.namespace, &request.consumer)?
        .last_seq;
    let current = named_projection(
        uow.transaction,
        &request.projection_namespace,
        &request.projection_key,
    )?;
    let mut projection = EntityProjection::from_payload(current.as_ref().map(|row| &row.payload))?;
    if projection.last_seq() != prior_cursor {
        return Err(StoreError::InvalidEntityEventPayload {
            message: "recovery projection cursor does not match durable replay cursor".to_owned(),
        }
        .into());
    }
    let events = replay_raw_events(
        uow.transaction,
        &request.namespace,
        prior_cursor,
        request.batch_limit,
    )?;
    let latest_authoritative_seq = latest_retained_event_seq(uow.transaction, &request.namespace)?;
    for raw in &events {
        projection.fold(&raw_to_entity_event(raw.clone())?)?;
    }
    let projection_changed = !events.is_empty() || current.is_none();
    if projection_changed {
        replace_named_projection(
            uow.transaction,
            &request.projection_namespace,
            &request.projection_key,
            recovery_projection_write(&projection, latest_authoritative_seq),
        )?;
    }
    if request.abort_after_projection {
        return Err(SqliteStoreError::TransactionAborted(
            "requested after entity recovery projection write".to_owned(),
        ));
    }
    if projection_changed || projection.last_seq() != prior_cursor {
        uow.strict_advance_replay_cursor(
            &request.namespace,
            &request.consumer,
            projection.last_seq(),
        )?;
    }
    Ok(recovery_report(
        &projection,
        events.len(),
        prior_cursor,
        latest_authoritative_seq,
    ))
}

pub(super) fn rebuild_entity_projection_uow(
    uow: &mut SqliteUnitOfWork<'_>,
    request: &EntityRebuildRequest,
) -> SqliteStoreResult<EntityRecoveryReport> {
    validate_entity_rebuild_request(request)?;
    let prior_cursor = uow
        .replay_cursor(&request.namespace, &request.consumer)?
        .last_seq;
    let events = replay_raw_events(uow.transaction, &request.namespace, 0, usize::MAX)?;
    let latest_authoritative_seq = latest_retained_event_seq(uow.transaction, &request.namespace)?;
    let mut projection = EntityProjection::empty();
    for raw in &events {
        projection.fold(&raw_to_entity_event(raw.clone())?)?;
    }
    replace_named_projection(
        uow.transaction,
        &request.projection_namespace,
        &request.projection_key,
        recovery_projection_write(&projection, latest_authoritative_seq),
    )?;
    if request.abort_after_projection {
        return Err(SqliteStoreError::TransactionAborted(
            "requested after entity rebuild projection write".to_owned(),
        ));
    }
    // Full rebuild may reset a stale consumer cursor to the rebuilt boundary.
    uow.set_replay_cursor_legacy(&request.namespace, &request.consumer, projection.last_seq())?;
    Ok(recovery_report(
        &projection,
        events.len(),
        prior_cursor,
        latest_authoritative_seq,
    ))
}

pub(super) fn latest_retained_event_seq(
    conn: &Connection,
    namespace: &str,
) -> SqliteStoreResult<i64> {
    conn.query_row(
        "SELECT COALESCE(MAX(seq), 0) FROM entity_events WHERE namespace = ?1",
        [namespace],
        |row| row.get(0),
    )
    .map_err(Into::into)
}

pub(super) fn prune_entity_events_after(
    conn: &Connection,
    namespace: &str,
    to_seq: i64,
) -> SqliteStoreResult<u64> {
    let deleted = conn.execute(
        "DELETE FROM entity_events WHERE namespace = ?1 AND seq > ?2",
        params![namespace, to_seq],
    )?;
    u64::try_from(deleted).map_err(|error| SqliteStoreError::TransactionAborted(error.to_string()))
}
