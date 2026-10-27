use super::*;

pub(super) fn projection_payload_json(
    payload: &serde_json::Map<String, serde_json::Value>,
) -> SqliteStoreResult<String> {
    // `Map` preserves input order under this workspace feature. Rebuild through
    // BTreeMap to exactly match Python json.dumps(sort_keys=True, separators=...).
    let compact = serde_json::to_string(&BTreeMap::<_, _>::from_iter(payload.iter()))
        .map_err(|error| SqliteStoreError::TransactionAborted(error.to_string()))?;
    Ok(python_ascii_json(&compact))
}

pub(super) fn python_ascii_json(compact_json: &str) -> String {
    compact_json
        .chars()
        .flat_map(|character| match character as u32 {
            0..=0x7f => character.to_string().chars().collect::<Vec<_>>(),
            value @ 0x80..=0xffff => format!("\\u{value:04x}").chars().collect(),
            value => {
                let offset = value - 0x1_0000;
                let high = 0xd800 + (offset >> 10);
                let low = 0xdc00 + (offset & 0x3ff);
                format!("\\u{high:04x}\\u{low:04x}").chars().collect()
            }
        })
        .collect()
}

pub(super) fn projection_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<NamedProjection> {
    let payload_json: String = row.get(2)?;
    let payload = serde_json::from_str::<serde_json::Value>(&payload_json)
        .ok()
        .and_then(|value| value.as_object().cloned())
        .ok_or(rusqlite::Error::InvalidQuery)?;
    Ok(NamedProjection {
        namespace: row.get(0)?,
        key: row.get(1)?,
        payload,
        last_authoritative_seq: row.get(3)?,
        last_materialized_seq: row.get(4)?,
        projection_schema_version: row.get(5)?,
        materialization_status: row.get(6)?,
        updated_at_ms: row.get(7)?,
    })
}

pub(super) fn named_projection(
    conn: &Connection,
    namespace: &str,
    key: &str,
) -> SqliteStoreResult<Option<NamedProjection>> {
    conn.query_row(
        "SELECT namespace, key, payload_json, last_authoritative_seq, last_materialized_seq, projection_schema_version, materialization_status, updated_at_ms FROM named_projections WHERE namespace = ?1 AND key = ?2",
        params![namespace, key],
        projection_from_row,
    )
    .optional()
    .map_err(Into::into)
}

pub(super) fn named_projections(
    conn: &Connection,
    namespace: &str,
) -> SqliteStoreResult<Vec<NamedProjection>> {
    let mut statement = conn.prepare(
        "SELECT namespace, key, payload_json, last_authoritative_seq, last_materialized_seq, projection_schema_version, materialization_status, updated_at_ms FROM named_projections WHERE namespace = ?1 ORDER BY key ASC",
    )?;
    statement
        .query_map([namespace], projection_from_row)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(Into::into)
}

const STAGE1_NODE_PROJECTIONS_TABLE: &str = "stage1_node_projections";

pub(super) fn stage1_node_projection(
    conn: &Connection,
    namespace: &str,
    key: &str,
) -> SqliteStoreResult<Option<NamedProjection>> {
    conn.query_row(
        &format!("SELECT namespace, key, payload_json, last_authoritative_seq, last_materialized_seq, projection_schema_version, materialization_status, updated_at_ms FROM {STAGE1_NODE_PROJECTIONS_TABLE} WHERE namespace = ?1 AND key = ?2"),
        params![namespace, key],
        projection_from_row,
    )
    .optional()
    .map_err(Into::into)
}

pub(super) fn stage1_node_projections(
    conn: &Connection,
    namespace: &str,
) -> SqliteStoreResult<Vec<NamedProjection>> {
    let mut statement = conn.prepare(&format!(
        "SELECT namespace, key, payload_json, last_authoritative_seq, last_materialized_seq, projection_schema_version, materialization_status, updated_at_ms FROM {STAGE1_NODE_PROJECTIONS_TABLE} WHERE namespace = ?1 ORDER BY key ASC"
    ))?;
    statement
        .query_map([namespace], projection_from_row)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(Into::into)
}

pub(super) fn replace_named_projection(
    conn: &Connection,
    namespace: &str,
    key: &str,
    projection: NamedProjectionWrite,
) -> SqliteStoreResult<()> {
    let payload_json = projection_payload_json(&projection.payload)?;
    conn.execute(
        "INSERT INTO named_projections(namespace, key, payload_json, last_authoritative_seq, last_materialized_seq, projection_schema_version, materialization_status, updated_at_ms) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8) ON CONFLICT(namespace, key) DO UPDATE SET payload_json = excluded.payload_json, last_authoritative_seq = excluded.last_authoritative_seq, last_materialized_seq = excluded.last_materialized_seq, projection_schema_version = excluded.projection_schema_version, materialization_status = excluded.materialization_status, updated_at_ms = excluded.updated_at_ms",
        params![namespace, key, payload_json, projection.last_authoritative_seq, projection.last_materialized_seq, projection.projection_schema_version, projection.materialization_status, unix_epoch_millis()],
    )?;
    Ok(())
}

pub(super) fn replace_stage1_node_projection(
    conn: &Connection,
    namespace: &str,
    key: &str,
    projection: NamedProjectionWrite,
) -> SqliteStoreResult<()> {
    let payload_json = projection_payload_json(&projection.payload)?;
    conn.execute(
        &format!("INSERT INTO {STAGE1_NODE_PROJECTIONS_TABLE}(namespace, key, payload_json, last_authoritative_seq, last_materialized_seq, projection_schema_version, materialization_status, updated_at_ms) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8) ON CONFLICT(namespace, key) DO UPDATE SET payload_json = excluded.payload_json, last_authoritative_seq = excluded.last_authoritative_seq, last_materialized_seq = excluded.last_materialized_seq, projection_schema_version = excluded.projection_schema_version, materialization_status = excluded.materialization_status, updated_at_ms = excluded.updated_at_ms"),
        params![namespace, key, payload_json, projection.last_authoritative_seq, projection.last_materialized_seq, projection.projection_schema_version, projection.materialization_status, unix_epoch_millis()],
    )?;
    Ok(())
}

pub(super) fn compare_and_swap_named_projection(
    conn: &Connection,
    namespace: &str,
    key: &str,
    expected_last_authoritative_seq: Option<i64>,
    expected_last_materialized_seq: Option<i64>,
    projection: NamedProjectionWrite,
) -> SqliteStoreResult<bool> {
    let payload_json = projection_payload_json(&projection.payload)?;
    let updated_at_ms = unix_epoch_millis();
    let changes = match (expected_last_authoritative_seq, expected_last_materialized_seq) {
        (None, None) => conn.execute(
            "INSERT INTO named_projections(namespace, key, payload_json, last_authoritative_seq, last_materialized_seq, projection_schema_version, materialization_status, updated_at_ms) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8) ON CONFLICT(namespace, key) DO NOTHING",
            params![namespace, key, payload_json, projection.last_authoritative_seq, projection.last_materialized_seq, projection.projection_schema_version, projection.materialization_status, updated_at_ms],
        )?,
        (Some(expected_authoritative), Some(expected_materialized)) => conn.execute(
            "UPDATE named_projections SET payload_json = ?1, last_authoritative_seq = ?2, last_materialized_seq = ?3, projection_schema_version = ?4, materialization_status = ?5, updated_at_ms = ?6 WHERE namespace = ?7 AND key = ?8 AND last_authoritative_seq = ?9 AND last_materialized_seq = ?10",
            params![payload_json, projection.last_authoritative_seq, projection.last_materialized_seq, projection.projection_schema_version, projection.materialization_status, updated_at_ms, namespace, key, expected_authoritative, expected_materialized],
        )?,
        _ => 0,
    };
    Ok(changes == 1)
}

pub(super) fn compare_and_swap_named_projections(
    conn: &Connection,
    mut updates: Vec<NamedProjectionCas>,
) -> SqliteStoreResult<bool> {
    updates
        .sort_by(|left, right| (&left.namespace, &left.key).cmp(&(&right.namespace, &right.key)));
    let mut identities = BTreeSet::new();
    for update in &updates {
        if !identities.insert((update.namespace.as_str(), update.key.as_str())) {
            return Err(SqliteStoreError::TransactionAborted(
                "duplicate named projection key".to_owned(),
            ));
        }
        let current = conn
            .query_row(
                "SELECT last_authoritative_seq,last_materialized_seq FROM named_projections WHERE namespace=?1 AND key=?2",
                params![update.namespace, update.key],
                |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)),
            )
            .optional()?;
        match (
            update.expected_last_authoritative_seq,
            update.expected_last_materialized_seq,
            current,
        ) {
            (None, None, None) => {}
            (Some(expected_authoritative), Some(expected_materialized), Some(actual))
                if actual == (expected_authoritative, expected_materialized) => {}
            _ => return Ok(false),
        }
    }
    for update in updates {
        let payload_json = projection_payload_json(&update.projection.payload)?;
        let updated_at_ms = unix_epoch_millis();
        let result = match (
            update.expected_last_authoritative_seq,
            update.expected_last_materialized_seq,
        ) {
            (None, None) => conn.execute(
                "INSERT INTO named_projections(namespace,key,payload_json,last_authoritative_seq,last_materialized_seq,projection_schema_version,materialization_status,updated_at_ms) VALUES (?1,?2,?3,?4,?5,?6,?7,?8)",
                params![
                    update.namespace,
                    update.key,
                    payload_json,
                    update.projection.last_authoritative_seq,
                    update.projection.last_materialized_seq,
                    update.projection.projection_schema_version,
                    update.projection.materialization_status,
                    updated_at_ms,
                ],
            )?,
            (Some(expected_authoritative), Some(expected_materialized)) => conn.execute(
                "UPDATE named_projections SET payload_json=?1,last_authoritative_seq=?2,last_materialized_seq=?3,projection_schema_version=?4,materialization_status=?5,updated_at_ms=?6 WHERE namespace=?7 AND key=?8 AND last_authoritative_seq=?9 AND last_materialized_seq=?10",
                params![
                    payload_json,
                    update.projection.last_authoritative_seq,
                    update.projection.last_materialized_seq,
                    update.projection.projection_schema_version,
                    update.projection.materialization_status,
                    updated_at_ms,
                    update.namespace,
                    update.key,
                    expected_authoritative,
                    expected_materialized,
                ],
            )?,
            _ => unreachable!("mixed named projection CAS expectations validated above"),
        };
        if result != 1 {
            return Err(SqliteStoreError::TransactionAborted(
                "named projection CAS changed during batch".to_owned(),
            ));
        }
    }
    Ok(true)
}

pub(super) fn clear_named_projection(
    conn: &Connection,
    namespace: &str,
    key: &str,
) -> SqliteStoreResult<()> {
    conn.execute(
        "DELETE FROM named_projections WHERE namespace = ?1 AND key = ?2",
        params![namespace, key],
    )?;
    Ok(())
}

pub(super) fn clear_stage1_node_projection(
    conn: &Connection,
    namespace: &str,
    key: &str,
) -> SqliteStoreResult<()> {
    conn.execute(
        &format!("DELETE FROM {STAGE1_NODE_PROJECTIONS_TABLE} WHERE namespace = ?1 AND key = ?2"),
        params![namespace, key],
    )?;
    Ok(())
}

pub(super) fn clear_projection_namespace(
    conn: &Connection,
    namespace: &str,
) -> SqliteStoreResult<()> {
    conn.execute(
        "DELETE FROM named_projections WHERE namespace = ?1",
        [namespace],
    )?;
    Ok(())
}
