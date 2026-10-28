use super::super::*;

pub(crate) fn sql_text_literal(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

pub(crate) fn graph_record_from_row(row: &Row) -> PostgresStoreResult<GraphRecord> {
    let metadata_json: String = row.try_get(2).map_err(backend)?;
    let metadata: Map<String, Value> = serde_json::from_str(&metadata_json)?;
    let embedding_json: Option<String> = row.try_get(3).map_err(backend)?;
    let embedding = embedding_json
        .as_deref()
        .map(serde_json::from_str)
        .transpose()?;
    Ok(GraphRecord {
        id: row.try_get(0).map_err(backend)?,
        document: row.try_get(1).map_err(backend)?,
        metadata,
        embedding,
    })
}

pub(crate) fn graph_filter_sql(
    scope: &kogwistar_store::GraphScope,
    filter: &kogwistar_store::MetadataFilter,
    start: usize,
) -> PostgresStoreResult<(String, Vec<String>)> {
    graph_filter_sql_with_exclusions(scope, filter, &BTreeMap::new(), start)
}

pub(crate) fn graph_filter_sql_with_exclusions(
    scope: &kogwistar_store::GraphScope,
    filter: &kogwistar_store::MetadataFilter,
    metadata_not_equals: &BTreeMap<String, Value>,
    start: usize,
) -> PostgresStoreResult<(String, Vec<String>)> {
    let scope_pairs = graph_scope_pairs(scope)?;
    let legacy_default =
        scope.namespace == "default" && scope.workspace_id.is_none() && scope.graph_space.is_none();
    let filter_pairs = filter
        .equals
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()));
    let mut clauses =
        Vec::with_capacity(scope_pairs.len() + filter.equals.len() + metadata_not_equals.len());
    let mut values =
        Vec::with_capacity(scope_pairs.len() + filter.equals.len() + metadata_not_equals.len());
    for (index, (key, value)) in scope_pairs.into_iter().chain(filter_pairs).enumerate() {
        let key_index = start + index;
        let value_json = serde_json::to_string(&value).expect("JSON value serializes");
        let exact = format!(
            "metadata -> ${key_index}::TEXT = {}::JSONB",
            sql_text_literal(&value_json)
        );
        if legacy_default && index == 0 && key == "namespace" {
            clauses.push(format!(
                "({exact} OR (NOT (metadata ? 'namespace') AND NOT (metadata ? 'workspace_id') AND NOT (metadata ? 'graph_space')))"
            ));
        } else {
            clauses.push(exact);
        }
        values.push(key);
    }
    for (key, value) in metadata_not_equals {
        let key_index = start + values.len();
        let value_json = serde_json::to_string(value).expect("JSON value serializes");
        clauses.push(format!(
            "metadata -> ${key_index}::TEXT IS DISTINCT FROM {}::JSONB",
            sql_text_literal(&value_json)
        ));
        values.push(key.clone());
    }
    Ok((clauses.join(" AND "), values))
}
