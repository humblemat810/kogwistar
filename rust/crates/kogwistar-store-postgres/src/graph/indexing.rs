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
    document: &Value,
    field: &str,
) -> PostgresStoreResult<Vec<String>> {
    match document.get(field) {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::Array(values)) => values
            .iter()
            .map(|value| {
                value.as_str().map(str::to_owned).ok_or_else(|| {
                    PostgresStoreError::Backend(format!(
                        "edge document field {field:?} must contain only string ids"
                    ))
                })
            })
            .collect(),
        Some(_) => Err(PostgresStoreError::Backend(format!(
            "edge document field {field:?} must be an array of string ids"
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
    let Some(document_json) = mutation.record.document.as_deref() else {
        return Ok(());
    };
    let document: Value = serde_json::from_str(document_json)?;
    let object = document.as_object().ok_or_else(|| {
        PostgresStoreError::Backend("edge document must be a JSON object".to_owned())
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
        for endpoint_id in graph_edge_endpoint_ids(&document, field)? {
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

pub(crate) fn node_document_ids(document: &Value) -> BTreeSet<String> {
    let mut ids = BTreeSet::new();
    let Some(mentions) = document.get("mentions").and_then(Value::as_array) else {
        return ids;
    };
    for mention in mentions {
        let spans = mention
            .get("spans")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or(std::slice::from_ref(mention));
        for span in spans {
            if let Some(doc_id) = span
                .get("doc_id")
                .and_then(Value::as_str)
                .filter(|doc_id| !doc_id.is_empty())
            {
                ids.insert(doc_id.to_owned());
                continue;
            }
            if let Some(url) = span.get("document_page_url").and_then(Value::as_str)
                && let Some(path) = url.split_once("document/").map(|(_, path)| path)
            {
                let doc_id = path
                    .chars()
                    .take_while(|character| character.is_ascii_alphanumeric() || *character == '-')
                    .collect::<String>();
                if !doc_id.is_empty() {
                    ids.insert(doc_id);
                }
            }
        }
    }
    ids
}

pub(crate) async fn sync_graph_node_docs<C>(
    client: &C,
    tables: &Tables,
    mutation: &GraphMutation,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    if mutation.entity_kind != "node" {
        return Ok(());
    }
    client
        .execute(
            &format!(
                "DELETE FROM {} WHERE metadata ->> 'node_id' = $1",
                tables.node_docs
            ),
            &[&mutation.record.id],
        )
        .await
        .map_err(backend)?;
    if mutation.op == "TOMBSTONE" {
        return Ok(());
    }

    let Some(document_json) = mutation.record.document.as_deref() else {
        return Ok(());
    };
    let Ok(document) = serde_json::from_str::<Value>(document_json) else {
        // GraphMutation also supports opaque documents. Only model-shaped JSON
        // records participate in the Python-compatible node/document index.
        return Ok(());
    };
    if !document.is_object() {
        return Ok(());
    }

    let doc_ids = node_document_ids(&document);
    for doc_id in &doc_ids {
        let row_id = format!("{}::{doc_id}", mutation.record.id);
        let row = serde_json::json!({
            "id": row_id,
            "node_id": mutation.record.id,
            "doc_id": doc_id,
            "mention_count": 1,
        });
        let row_json = serde_json::to_string(&row)?;
        let metadata_json = serde_json::to_string(&graph_metadata(
            &mutation.scope,
            row.as_object().expect("row is an object"),
        )?)?;
        client
            .execute(
                &format!(
                    "INSERT INTO {} (id, document, metadata) VALUES ($1, $2, $3::TEXT::JSONB) \
                     ON CONFLICT (id) DO UPDATE SET document = EXCLUDED.document, \
                     metadata = EXCLUDED.metadata, updated_at = NOW()",
                    tables.node_docs
                ),
                &[&row_id, &row_json, &metadata_json],
            )
            .await
            .map_err(backend)?;
    }

    let doc_ids_json = serde_json::to_string(&doc_ids.into_iter().collect::<Vec<_>>())?;
    let graph_table = graph_table(tables, &mutation.table)?;
    client
        .execute(
            &format!(
                "UPDATE {graph_table} SET metadata = metadata || jsonb_build_object('doc_ids', $1::TEXT), updated_at = NOW() WHERE id = $2"
            ),
            &[&doc_ids_json, &mutation.record.id],
        )
        .await
        .map_err(backend)?;
    Ok(())
}

pub(crate) fn graph_reference_rows(
    entity_kind: &str,
    entity_id: &str,
    document: &Value,
) -> Vec<Map<String, Value>> {
    let (owner_key, entity_doc_id) = match entity_kind {
        "node" => ("node_id", document.get("doc_id")),
        "edge" => ("edge_id", document.get("doc_id")),
        _ => return Vec::new(),
    };
    let Some(mentions) = document.get("mentions").and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut rows = Vec::new();
    for (mention_index, mention) in mentions.iter().enumerate() {
        let spans = mention
            .get("spans")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .or_else(|| {
                (mention.get("doc_id").is_some() || mention.get("document_page_url").is_some())
                    .then_some(std::slice::from_ref(mention))
            })
            .unwrap_or(&[]);
        for (span_index, span) in spans.iter().enumerate() {
            let Some(span) = span.as_object() else {
                continue;
            };
            let mut row = Map::new();
            let row_id = format!("{entity_id}::mention::{mention_index}::span::{span_index}");
            row.insert("id".to_owned(), Value::String(row_id));
            row.insert(owner_key.to_owned(), Value::String(entity_id.to_owned()));
            for field in [
                "insertion_method",
                "page_number",
                "excerpt",
                "start_char",
                "end_char",
            ] {
                if let Some(value) = span.get(field).filter(|value| !value.is_null()) {
                    row.insert(field.to_owned(), value.clone());
                }
            }
            if let Some(doc_id) = span
                .get("doc_id")
                .filter(|value| value.as_str().is_some_and(|value| !value.is_empty()))
                .or_else(|| entity_doc_id.filter(|value| !value.is_null()))
            {
                row.insert("doc_id".to_owned(), doc_id.clone());
            }
            if let Some(verification) = span.get("verification").and_then(Value::as_object) {
                for (source, target) in [
                    ("method", "verification_method"),
                    ("is_verified", "is_verified"),
                    ("score", "verificication_score"),
                ] {
                    if let Some(value) = verification.get(source).filter(|value| !value.is_null()) {
                        row.insert(target.to_owned(), value.clone());
                    }
                }
            }
            rows.push(row);
        }
    }
    rows
}

pub(crate) async fn sync_graph_refs<C>(
    client: &C,
    tables: &Tables,
    mutation: &GraphMutation,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    let (table, owner_key) = match mutation.entity_kind.as_str() {
        "node" => (&tables.node_refs, "node_id"),
        "edge" => (&tables.edge_refs, "edge_id"),
        _ => return Ok(()),
    };
    client
        .execute(
            &format!("DELETE FROM {table} WHERE metadata ->> '{owner_key}' = $1"),
            &[&mutation.record.id],
        )
        .await
        .map_err(backend)?;
    if mutation.op == "TOMBSTONE" {
        return Ok(());
    }
    let Some(document_json) = mutation.record.document.as_deref() else {
        return Ok(());
    };
    let Ok(document) = serde_json::from_str::<Value>(document_json) else {
        return Ok(());
    };
    if !document.is_object() {
        return Ok(());
    }
    for row in graph_reference_rows(&mutation.entity_kind, &mutation.record.id, &document) {
        let row_id = row
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| PostgresStoreError::Backend("reference row is missing id".to_owned()))?;
        let row_json = serde_json::to_string(&row)?;
        let metadata_json = serde_json::to_string(&graph_metadata(&mutation.scope, &row)?)?;
        client
            .execute(
                &format!(
                    "INSERT INTO {table} (id, document, metadata) VALUES ($1, $2, $3::TEXT::JSONB) \
                     ON CONFLICT (id) DO UPDATE SET document = EXCLUDED.document, \
                     metadata = EXCLUDED.metadata, updated_at = NOW()"
                ),
                &[&row_id, &row_json, &metadata_json],
            )
            .await
            .map_err(backend)?;
    }
    Ok(())
}
