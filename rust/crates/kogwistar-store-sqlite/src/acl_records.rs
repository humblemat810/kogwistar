use super::*;

pub(super) fn graph_acl_records_for_entities(
    conn: &Connection,
    scope: &kogwistar_store::GraphScope,
    truth_graph: &str,
    entity_ids: &[String],
) -> SqliteStoreResult<Vec<GraphRecord>> {
    if scope.namespace.is_empty() || truth_graph.trim().is_empty() {
        return Err(SqliteStoreError::TransactionAborted(
            "ACL lookup requires a namespace and graph role".to_owned(),
        ));
    }
    if entity_ids.is_empty() {
        return Ok(Vec::new());
    }
    if entity_ids.iter().any(String::is_empty) {
        return Err(SqliteStoreError::TransactionAborted(
            "ACL lookup entity IDs must not be empty".to_owned(),
        ));
    }
    let mut scope_predicates = Vec::new();
    let mut params = vec![scope.namespace.clone(), truth_graph.to_owned()];
    for (key, value) in [
        ("workspace_id", scope.workspace_id.as_deref()),
        ("graph_space", scope.graph_space.as_deref()),
    ] {
        if let Some(value) = value {
            params.push(value.to_owned());
            scope_predicates.push(format!(
                "json_extract(payload_json, '$.metadata.{key}') = ?{}",
                params.len()
            ));
        }
    }
    let scope_predicates = if scope_predicates.is_empty() {
        String::new()
    } else {
        format!(" AND {}", scope_predicates.join(" AND "))
    };
    let id_start = params.len() + 1;
    let sql = format!(
        "SELECT entity_id, op, payload_json FROM entity_events \
         WHERE namespace = ?1 AND entity_kind = 'node' \
         AND json_extract(payload_json, '$.metadata.entity_type') = 'acl_record' \
         AND json_extract(payload_json, '$.metadata.acl_truth_graph') = ?2 \
         {scope_predicates} \
         AND json_extract(payload_json, '$.metadata.acl_target_entity_id') IN ({}) \
         ORDER BY seq ASC",
        (id_start..id_start + entity_ids.len())
            .map(|index| format!("?{index}"))
            .collect::<Vec<_>>()
            .join(",")
    );
    params.extend(entity_ids.iter().cloned());
    let mut statement = conn.prepare(&sql)?;
    let rows = statement.query_map(params_from_iter(params.iter()), |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
        ))
    })?;
    let mut records = Vec::new();
    for row in rows {
        let (id, op, payload_json) = row?;
        let payload: Value = serde_json::from_str(&payload_json)?;
        let Some(payload_object) = payload.as_object() else {
            return Err(SqliteStoreError::TransactionAborted(
                "canonical ACL event payload must be an object".to_owned(),
            ));
        };
        let mut metadata = payload_object
            .get("metadata")
            .and_then(Value::as_object)
            .cloned()
            .ok_or_else(|| {
                SqliteStoreError::TransactionAborted(
                    "canonical ACL event metadata must be an object".to_owned(),
                )
            })?;
        if let Some(properties) = payload_object.get("properties").and_then(Value::as_object) {
            metadata.extend(properties.clone());
        }
        if matches!(op.to_ascii_uppercase().as_str(), "TOMBSTONE" | "DELETE") {
            metadata.insert("tombstoned".to_owned(), Value::Bool(true));
        }
        records.push(GraphRecord {
            id,
            document: None,
            metadata,
            embedding: None,
        });
    }
    Ok(records)
}
