use super::*;

pub(super) fn current_global_seq(conn: &Connection) -> SqliteStoreResult<i64> {
    Ok(conn
        .query_row("SELECT value FROM global_seq WHERE rowid = 1", [], |row| {
            row.get(0)
        })
        .optional()?
        .unwrap_or(0))
}

pub(super) fn current_user_seq(conn: &Connection, user_id: &str) -> SqliteStoreResult<i64> {
    Ok(conn
        .query_row(
            "SELECT value FROM user_seq WHERE user_id = ?1",
            [user_id],
            |row| row.get(0),
        )
        .optional()?
        .unwrap_or(0))
}

pub(super) fn append_raw_entity_event(
    conn: &Connection,
    namespace: &str,
    event: NewRawEntityEvent,
) -> SqliteStoreResult<AppendedRawEvent> {
    if let Some(existing) = event_by_id(conn, &event.event_id)? {
        if existing.namespace != namespace {
            return Err(SqliteStoreError::EventIdNamespaceCollision {
                event_id: event.event_id,
                existing_namespace: existing.namespace,
                requested_namespace: namespace.to_owned(),
            });
        }
        return Ok(AppendedRawEvent {
            event: existing,
            inserted: false,
        });
    }
    let seq = conn.query_row(
        "INSERT INTO namespace_seq(namespace, next_seq) VALUES (?1, 2) \
         ON CONFLICT(namespace) DO UPDATE SET next_seq = namespace_seq.next_seq + 1 \
         RETURNING next_seq - 1",
        [namespace],
        |row| row.get(0),
    )?;
    let stored = RawEntityEvent {
        namespace: namespace.to_owned(),
        seq,
        event_id: event.event_id,
        entity_kind: event.entity_kind,
        entity_id: event.entity_id,
        op: event.op,
        payload_json: event.payload_json,
        created_at: unix_epoch_seconds(),
    };
    conn.execute(
        "INSERT INTO entity_events(namespace, seq, event_id, entity_kind, entity_id, op, payload_json, created_at) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        params![
            stored.namespace,
            stored.seq,
            stored.event_id,
            stored.entity_kind,
            stored.entity_id,
            stored.op,
            stored.payload_json,
            stored.created_at,
        ],
    )?;
    Ok(AppendedRawEvent {
        event: stored,
        inserted: true,
    })
}

pub(super) fn append_restored_raw_entity_event(
    conn: &Connection,
    namespace: &str,
    event: RestoredRawEntityEvent,
) -> SqliteStoreResult<AppendedRawEvent> {
    if namespace.is_empty() {
        return Err(SqliteStoreError::EmptyNamespace);
    }
    if event.seq < 1 {
        return Err(SqliteStoreError::NegativeSequenceValue { value: event.seq });
    }
    serde_json::from_str::<Value>(&event.payload_json)?;
    if let Some(existing) = event_by_id(conn, &event.event_id)? {
        if existing.namespace != namespace {
            return Err(SqliteStoreError::EventIdNamespaceCollision {
                event_id: event.event_id,
                existing_namespace: existing.namespace,
                requested_namespace: namespace.to_owned(),
            });
        }
        let same = existing.seq == event.seq
            && existing.entity_kind == event.entity_kind
            && existing.entity_id == event.entity_id
            && existing.op == event.op
            && existing.payload_json == event.payload_json
            && existing.created_at == event.created_at;
        if !same {
            return Err(SqliteStoreError::TransactionAborted(format!(
                "restored event_id {:?} conflicts with stored event",
                event.event_id
            )));
        }
        return Ok(AppendedRawEvent {
            event: existing,
            inserted: false,
        });
    }
    let latest = latest_retained_event_seq(conn, namespace)?;
    if event.seq != latest + 1 {
        return Err(SqliteStoreError::TransactionAborted(format!(
            "restored event sequence for {:?} must be {}, got {}",
            namespace,
            latest + 1,
            event.seq
        )));
    }
    let stored = RawEntityEvent {
        namespace: namespace.to_owned(),
        seq: event.seq,
        event_id: event.event_id,
        entity_kind: event.entity_kind,
        entity_id: event.entity_id,
        op: event.op,
        payload_json: event.payload_json,
        created_at: event.created_at,
    };
    conn.execute(
        "INSERT INTO entity_events(namespace, seq, event_id, entity_kind, entity_id, op, payload_json, created_at) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        params![
            stored.namespace,
            stored.seq,
            stored.event_id,
            stored.entity_kind,
            stored.entity_id,
            stored.op,
            stored.payload_json,
            stored.created_at,
        ],
    )?;
    conn.execute(
        "INSERT INTO namespace_seq(namespace, next_seq) VALUES (?1, ?2) \
         ON CONFLICT(namespace) DO UPDATE SET next_seq = excluded.next_seq",
        params![namespace, event.seq + 1],
    )?;
    Ok(AppendedRawEvent {
        event: stored,
        inserted: true,
    })
}

pub(super) fn event_by_id(
    conn: &Connection,
    event_id: &str,
) -> SqliteStoreResult<Option<RawEntityEvent>> {
    conn.query_row(
        "SELECT namespace, seq, event_id, entity_kind, entity_id, op, payload_json, created_at \
         FROM entity_events WHERE event_id = ?1",
        [event_id],
        raw_event_from_row,
    )
    .optional()
    .map_err(Into::into)
}

pub(super) fn replay_raw_events(
    conn: &Connection,
    namespace: &str,
    after_seq: i64,
    limit: usize,
) -> SqliteStoreResult<Vec<RawEntityEvent>> {
    let sql_limit = i64::try_from(limit).unwrap_or(i64::MAX);
    let mut statement = conn.prepare(
        "SELECT namespace, seq, event_id, entity_kind, entity_id, op, payload_json, created_at \
         FROM entity_events WHERE namespace = ?1 AND seq > ?2 ORDER BY seq ASC LIMIT ?3",
    )?;
    let rows = statement.query_map(params![namespace, after_seq, sql_limit], raw_event_from_row)?;
    rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
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
    // Full rebuild is explicitly allowed to reset a stale consumer cursor to
    // the rebuilt corpus boundary. It uses legacy set only after projection
    // write, in the same UoW, because strict monotonic advance rejects reset.
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
