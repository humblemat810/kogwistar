use super::super::*;

pub(crate) fn graph_scope_pairs(
    scope: &kogwistar_store::GraphScope,
) -> PostgresStoreResult<Vec<(String, Value)>> {
    require_namespace(&scope.namespace)?;
    let mut pairs = vec![(
        "namespace".to_owned(),
        Value::String(scope.namespace.clone()),
    )];
    if let Some(workspace_id) = &scope.workspace_id {
        pairs.push((
            "workspace_id".to_owned(),
            Value::String(workspace_id.clone()),
        ));
    }
    if let Some(graph_space) = &scope.graph_space {
        pairs.push(("graph_space".to_owned(), Value::String(graph_space.clone())));
    }
    Ok(pairs)
}

pub(crate) fn graph_metadata(
    scope: &kogwistar_store::GraphScope,
    metadata: &Map<String, Value>,
) -> PostgresStoreResult<Map<String, Value>> {
    let scope_pairs = graph_scope_pairs(scope)?;
    for (key, expected) in &scope_pairs {
        if let Some(actual) = metadata.get(key)
            && actual != expected
        {
            return Err(PostgresStoreError::Backend(format!(
                "graph metadata scope {key:?} conflicts with request scope"
            )));
        }
    }
    let mut materialized = metadata.clone();
    for (key, value) in scope_pairs {
        materialized.insert(key, value);
    }
    Ok(materialized)
}

pub(crate) fn graph_scope_matches_or_legacy_default(
    scope: &kogwistar_store::GraphScope,
    metadata: &Map<String, Value>,
) -> PostgresStoreResult<bool> {
    let expected = graph_metadata(scope, &Map::new())?;
    if expected
        .iter()
        .all(|(key, value)| metadata.get(key) == Some(value))
    {
        return Ok(true);
    }
    Ok(scope.namespace == "default"
        && scope.workspace_id.is_none()
        && scope.graph_space.is_none()
        && !metadata.contains_key("namespace")
        && !metadata.contains_key("workspace_id")
        && !metadata.contains_key("graph_space"))
}

pub(crate) fn finite_vector(vector: &[f32]) -> PostgresStoreResult<()> {
    if vector.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(StoreError::NonFiniteVector.into())
    }
}

pub(crate) fn vector_literal(vector: &[f32]) -> String {
    format!(
        "[{}]",
        vector
            .iter()
            .map(f32::to_string)
            .collect::<Vec<_>>()
            .join(",")
    )
}

pub(crate) fn graph_edge_endpoint_ids(
    payload: &Value,
    field: &str,
) -> PostgresStoreResult<Vec<String>> {
    match payload.get(field) {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::Array(values)) => values
            .iter()
            .map(|value| {
                value.as_str().map(str::to_owned).ok_or_else(|| {
                    PostgresStoreError::Backend(format!(
                        "edge mutation payload field {field:?} must contain only string ids"
                    ))
                })
            })
            .collect(),
        Some(_) => Err(PostgresStoreError::Backend(format!(
            "edge mutation payload field {field:?} must be an array of string ids"
        ))),
    }
}

pub(crate) async fn sync_graph_edge_endpoints<C>(
    client: &C,
    tables: &Tables,
    mutation: &GraphMutation,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    if mutation.entity_kind != "edge" {
        return Ok(());
    }

    client
        .execute(
            &format!(
                "DELETE FROM {} WHERE metadata ->> 'edge_id' = $1",
                tables.edge_endpoints
            ),
            &[&mutation.record.id],
        )
        .await
        .map_err(backend)?;

    if mutation.op == "TOMBSTONE" {
        return Ok(());
    }
    let payload = &mutation.payload;
    let object = payload.as_object().ok_or_else(|| {
        PostgresStoreError::Backend("edge mutation payload must be a JSON object".to_owned())
    })?;
    let relation = object.get("relation").filter(|value| !value.is_null());
    let doc_id = object.get("doc_id").filter(|value| !value.is_null());
    let causal_type = object
        .get("metadata")
        .and_then(Value::as_object)
        .and_then(|metadata| metadata.get("causal_type"))
        .filter(|value| !value.is_null());

    for (role, field, endpoint_type) in [
        ("src", "source_ids", "node"),
        ("tgt", "target_ids", "node"),
        ("src", "source_edge_ids", "edge"),
        ("tgt", "target_edge_ids", "edge"),
    ] {
        for endpoint_id in graph_edge_endpoint_ids(payload, field)? {
            let row_id = format!(
                "{}::{role}::{endpoint_type}::{endpoint_id}",
                mutation.record.id
            );
            let mut row = Map::new();
            row.insert("id".to_owned(), Value::String(row_id.clone()));
            row.insert(
                "edge_id".to_owned(),
                Value::String(mutation.record.id.clone()),
            );
            row.insert("endpoint_id".to_owned(), Value::String(endpoint_id));
            row.insert(
                "endpoint_type".to_owned(),
                Value::String(endpoint_type.to_owned()),
            );
            row.insert("role".to_owned(), Value::String(role.to_owned()));
            if let Some(value) = relation {
                row.insert("relation".to_owned(), value.clone());
            }
            if let Some(value) = doc_id {
                row.insert("doc_id".to_owned(), value.clone());
            }
            if let Some(value) = causal_type {
                row.insert("causal_type".to_owned(), value.clone());
            }

            let row_document = serde_json::to_string(&row)?;
            let metadata = serde_json::to_string(&graph_metadata(&mutation.scope, &row)?)?;
            client
                .execute(
                    &format!(
                        "INSERT INTO {} (id, document, metadata) VALUES ($1, $2, $3::TEXT::JSONB) \
                         ON CONFLICT (id) DO UPDATE SET document = EXCLUDED.document, \
                         metadata = EXCLUDED.metadata, updated_at = NOW()",
                        tables.edge_endpoints
                    ),
                    &[&row_id, &row_document, &metadata],
                )
                .await
                .map_err(backend)?;
        }
    }
    Ok(())
}

mod references;
pub(crate) use references::*;
