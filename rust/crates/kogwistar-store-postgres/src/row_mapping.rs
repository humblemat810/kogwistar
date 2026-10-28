use super::*;

pub(super) fn workflow_design_snapshot_from_row(
    row: &Row,
) -> PostgresStoreResult<WorkflowDesignSnapshot> {
    Ok(WorkflowDesignSnapshot {
        workflow_id: row.try_get(0).map_err(backend)?,
        version: row.try_get(1).map_err(backend)?,
        seq: row.try_get(2).map_err(backend)?,
        payload_json: row.try_get(3).map_err(backend)?,
        schema_version: i64::from(row.try_get::<_, i32>(4).map_err(backend)?),
        created_at_ms: row.try_get(5).map_err(backend)?,
    })
}

pub(super) fn workflow_design_delta_from_row(
    row: &Row,
) -> PostgresStoreResult<WorkflowDesignDelta> {
    Ok(WorkflowDesignDelta {
        workflow_id: row.try_get(0).map_err(backend)?,
        version: row.try_get(1).map_err(backend)?,
        prev_version: row.try_get(2).map_err(backend)?,
        target_seq: row.try_get(3).map_err(backend)?,
        forward_json: row.try_get(4).map_err(backend)?,
        inverse_json: row.try_get(5).map_err(backend)?,
        schema_version: i64::from(row.try_get::<_, i32>(6).map_err(backend)?),
        created_at_ms: row.try_get(7).map_err(backend)?,
    })
}
pub(super) fn server_run_from_row(row: &Row) -> PostgresStoreResult<ServerRun> {
    Ok(ServerRun {
        run_id: row.try_get(0).map_err(backend)?,
        conversation_id: row.try_get(1).map_err(backend)?,
        workflow_id: row.try_get(2).map_err(backend)?,
        user_id: row.try_get(3).map_err(backend)?,
        user_turn_node_id: row.try_get(4).map_err(backend)?,
        assistant_turn_node_id: row.try_get(5).map_err(backend)?,
        status: row.try_get(6).map_err(backend)?,
        cancel_requested: row.try_get::<_, i32>(7).map_err(backend)? != 0,
        result_json: row.try_get(8).map_err(backend)?,
        error_json: row.try_get(9).map_err(backend)?,
        created_at_ms: row.try_get(10).map_err(backend)?,
        updated_at_ms: row.try_get(11).map_err(backend)?,
        started_at_ms: row.try_get(12).map_err(backend)?,
        finished_at_ms: row.try_get(13).map_err(backend)?,
    })
}
pub(super) fn server_run_event_from_row(row: &Row) -> PostgresStoreResult<ServerRunEvent> {
    Ok(ServerRunEvent {
        seq: row.try_get(0).map_err(backend)?,
        run_id: row.try_get(1).map_err(backend)?,
        event_type: row.try_get(2).map_err(backend)?,
        payload_json: row.try_get(3).map_err(backend)?,
        created_at_ms: row.try_get(4).map_err(backend)?,
    })
}

pub(super) fn raw_to_entity_event(raw: RawEntityEvent) -> PostgresStoreResult<EntityEvent> {
    Ok(EntityEventEnvelope {
        namespace: raw.namespace,
        seq: raw.seq,
        event_id: raw.event_id,
        entity_kind: raw.entity_kind,
        entity_id: raw.entity_id,
        op: raw.op,
        payload: serde_json::from_str(&raw.payload_json)?,
    })
}

pub(super) fn require_namespace(namespace: &str) -> PostgresStoreResult<()> {
    if namespace.is_empty() {
        Err(PostgresStoreError::EmptyNamespace)
    } else {
        Ok(())
    }
}

pub(super) fn require_trait_namespace(namespace: &str) -> StoreResult<()> {
    if namespace.is_empty() {
        Err(StoreError::EmptyNamespace)
    } else {
        Ok(())
    }
}

pub(super) fn trait_error(error: PostgresStoreError) -> StoreError {
    match error {
        PostgresStoreError::Store(error) => error,
        PostgresStoreError::EmptyNamespace => StoreError::EmptyNamespace,
        PostgresStoreError::EventIdNamespaceCollision {
            event_id,
            existing_namespace,
            requested_namespace,
        } => StoreError::EventIdNamespaceCollision {
            event_id,
            existing_namespace,
            requested_namespace,
        },
        error => StoreError::Backend {
            backend: "postgres".to_owned(),
            message: error.to_string(),
        },
    }
}

pub(super) fn backend(error: impl std::fmt::Display) -> PostgresStoreError {
    PostgresStoreError::Backend(error.to_string())
}
