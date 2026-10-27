use super::*;

pub(super) fn graph_scope_pairs(
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

pub(super) fn graph_metadata(
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

pub(super) fn graph_scope_matches_or_legacy_default(
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

pub(super) fn finite_vector(vector: &[f32]) -> PostgresStoreResult<()> {
    if vector.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(StoreError::NonFiniteVector.into())
    }
}

pub(super) fn vector_literal(vector: &[f32]) -> String {
    format!(
        "[{}]",
        vector
            .iter()
            .map(f32::to_string)
            .collect::<Vec<_>>()
            .join(",")
    )
}

pub(super) fn graph_edge_endpoint_ids(
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

pub(super) async fn sync_graph_edge_endpoints<C>(
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

pub(super) fn node_document_ids(document: &Value) -> BTreeSet<String> {
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

pub(super) async fn sync_graph_node_docs<C>(
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

pub(super) fn graph_reference_rows(
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

pub(super) async fn sync_graph_refs<C>(
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

pub(super) fn sql_text_literal(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

pub(super) fn graph_record_from_row(row: &Row) -> PostgresStoreResult<GraphRecord> {
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

pub(super) fn graph_filter_sql(
    scope: &kogwistar_store::GraphScope,
    filter: &kogwistar_store::MetadataFilter,
    start: usize,
) -> PostgresStoreResult<(String, Vec<String>)> {
    graph_filter_sql_with_exclusions(scope, filter, &BTreeMap::new(), start)
}

pub(super) fn graph_filter_sql_with_exclusions(
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

pub(super) async fn apply_graph_mutation<C>(
    client: &C,
    tables: &Tables,
    mutation: GraphMutation,
) -> PostgresStoreResult<AppliedGraphMutation>
where
    C: GenericClient + Sync,
{
    require_namespace(&mutation.scope.namespace)?;
    if mutation.record.id.is_empty() {
        return Err(StoreError::EmptyRecordId.into());
    }
    if mutation.event_id.is_empty() {
        return Err(PostgresStoreError::Backend(
            "graph mutation event id must not be empty".to_owned(),
        ));
    }
    if !matches!(mutation.op.as_str(), "ADD" | "REPLACE" | "TOMBSTONE") {
        return Err(PostgresStoreError::Backend(
            "graph mutation op must be ADD, REPLACE, or TOMBSTONE".to_owned(),
        ));
    }
    validate_graph_projection_target(&mutation.table, &mutation.entity_kind)?;
    let metadata = graph_metadata(&mutation.scope, &mutation.record.metadata)?;
    if let Some(embedding) = &mutation.record.embedding {
        finite_vector(embedding)?;
        if embedding.len() != mutation.embedding_dim {
            return Err(StoreError::VectorDimensionMismatch {
                expected: mutation.embedding_dim,
                actual: embedding.len(),
            }
            .into());
        }
    }
    let table = graph_table(tables, &mutation.table)?;
    let payload_json = serde_json::to_string(&mutation.payload)?;
    let appended = append_raw_entity_event(
        client,
        tables,
        &mutation.scope.namespace,
        NewRawEntityEvent {
            event_id: mutation.event_id.clone(),
            entity_kind: mutation.entity_kind.clone(),
            entity_id: mutation.record.id.clone(),
            op: mutation.op.clone(),
            payload_json,
        },
    )
    .await?;
    if !appended.inserted {
        let existing_payload: Value = serde_json::from_str(&appended.event.payload_json)
            .map_err(|error| PostgresStoreError::Backend(error.to_string()))?;
        if appended.event.entity_kind != mutation.entity_kind
            || appended.event.entity_id != mutation.record.id
            || appended.event.op != mutation.op
            || existing_payload != mutation.payload
        {
            return Err(PostgresStoreError::Backend(format!(
                "graph mutation event id {:?} was reused with different input",
                mutation.event_id
            )));
        }
    }
    let event = raw_to_entity_event(appended.event)?;
    if !appended.inserted {
        return Ok(AppliedGraphMutation {
            event,
            inserted: false,
            mutated: false,
        });
    }

    // `id` is globally unique in the Python collection DDL.  Never let one
    // namespace/workspace silently overwrite another's graph row.
    if let Some(row) = client
        .query_opt(
            &format!("SELECT metadata::TEXT FROM {table} WHERE id = $1 FOR UPDATE"),
            &[&mutation.record.id],
        )
        .await
        .map_err(backend)?
    {
        let current_json: String = row.try_get(0).map_err(backend)?;
        let current: Map<String, Value> = serde_json::from_str(&current_json)?;
        if !graph_scope_matches_or_legacy_default(&mutation.scope, &current)? {
            return Err(PostgresStoreError::Backend(
                "graph record id belongs to a different scope".to_owned(),
            ));
        }
    }
    let metadata_json = serde_json::to_string(&metadata)?;
    let embedding_sql = mutation
        .record
        .embedding
        .as_deref()
        .map(|vector| format!("{}::vector", sql_text_literal(&vector_literal(vector))))
        .unwrap_or_else(|| "NULL".to_owned());
    client
        .execute(
            &format!(
                "INSERT INTO {table}(id, document, metadata, embedding) VALUES ($1, $2, {}::JSONB, {embedding_sql}) \
                 ON CONFLICT(id) DO UPDATE SET document = EXCLUDED.document, metadata = EXCLUDED.metadata, \
                 embedding = EXCLUDED.embedding, updated_at = NOW()",
                sql_text_literal(&metadata_json),
            ),
            &[&mutation.record.id, &mutation.record.document],
        )
        .await
        .map_err(|error| PostgresStoreError::Backend(format!("graph projection upsert failed: {error:?}")))?;
    sync_graph_edge_endpoints(client, tables, &mutation).await?;
    sync_graph_node_docs(client, tables, &mutation).await?;
    sync_graph_refs(client, tables, &mutation).await?;
    Ok(AppliedGraphMutation {
        event,
        inserted: true,
        mutated: true,
    })
}

pub(super) async fn apply_graph_metadata_patch_mutation<C>(
    client: &C,
    tables: &Tables,
    mutation: GraphMetadataPatchMutation,
) -> PostgresStoreResult<Option<AppliedGraphMutation>>
where
    C: GenericClient + Sync,
{
    require_namespace(&mutation.scope.namespace)?;
    if mutation.entity_id.is_empty() {
        return Err(StoreError::EmptyRecordId.into());
    }
    if mutation.event_id.is_empty() {
        return Err(PostgresStoreError::Backend(
            "graph metadata patch event id must not be empty".to_owned(),
        ));
    }
    if !matches!(mutation.op.as_str(), "REPLACE" | "TOMBSTONE") {
        return Err(PostgresStoreError::Backend(
            "graph metadata patch op must be REPLACE or TOMBSTONE".to_owned(),
        ));
    }
    validate_graph_lifecycle_target(&mutation.table, &mutation.entity_kind)?;
    let table = graph_table(tables, &mutation.table)?;
    let Some(row) = client
        .query_opt(
            &format!("SELECT document, metadata::TEXT FROM {table} WHERE id = $1 FOR UPDATE"),
            &[&mutation.entity_id],
        )
        .await
        .map_err(backend)?
    else {
        return Ok(None);
    };

    let document: Option<String> = row.try_get(0).map_err(backend)?;
    let current_metadata_json: String = row.try_get(1).map_err(backend)?;
    let mut current_metadata: Map<String, Value> = serde_json::from_str(&current_metadata_json)?;
    if !graph_scope_matches_or_legacy_default(&mutation.scope, &current_metadata)? {
        return Err(PostgresStoreError::Backend(
            "graph record id belongs to a different scope".to_owned(),
        ));
    }

    let payload_json = serde_json::to_string(&mutation.payload)?;
    let appended = append_raw_entity_event(
        client,
        tables,
        &mutation.scope.namespace,
        NewRawEntityEvent {
            event_id: mutation.event_id.clone(),
            entity_kind: mutation.entity_kind.clone(),
            entity_id: mutation.entity_id.clone(),
            op: mutation.op.clone(),
            payload_json,
        },
    )
    .await?;
    let event = raw_to_entity_event(appended.event)?;
    if !appended.inserted {
        return Ok(Some(AppliedGraphMutation {
            event,
            inserted: false,
            mutated: false,
        }));
    }

    current_metadata = graph_metadata(&mutation.scope, &current_metadata)?;
    let validated_patch = graph_metadata(&mutation.scope, &mutation.metadata_patch)?;
    current_metadata.extend(validated_patch);
    let patched_document = if let Some(replacement) = mutation.document {
        replacement
    } else {
        let mut document_value = document
            .as_deref()
            .and_then(|value| serde_json::from_str::<Value>(value).ok())
            .and_then(|value| value.as_object().cloned())
            .unwrap_or_default();
        let mut document_metadata = document_value
            .remove("metadata")
            .and_then(|value| value.as_object().cloned())
            .unwrap_or_default();
        document_metadata.extend(mutation.metadata_patch);
        document_value.insert("metadata".to_owned(), Value::Object(document_metadata));
        serde_json::to_string(&Value::Object(document_value))?
    };
    let patched_metadata = serde_json::to_string(&current_metadata)?;
    client
        .execute(
            &format!(
                "UPDATE {table} SET document = $1, metadata = {}::JSONB, updated_at = NOW() WHERE id = $2",
                sql_text_literal(&patched_metadata)
            ),
            &[&patched_document, &mutation.entity_id],
        )
        .await
        .map_err(|error| {
            PostgresStoreError::Backend(format!("graph metadata patch failed: {error:?}"))
        })?;
    if matches!(mutation.entity_kind.as_str(), "node" | "edge") {
        let derived_mutation = GraphMutation {
            scope: mutation.scope,
            table: mutation.table,
            entity_kind: mutation.entity_kind,
            event_id: mutation.event_id,
            op: mutation.op,
            payload: mutation.payload,
            record: GraphRecord {
                id: mutation.entity_id,
                document: Some(patched_document),
                metadata: current_metadata,
                embedding: None,
            },
            embedding_dim: 0,
        };
        sync_graph_edge_endpoints(client, tables, &derived_mutation).await?;
        sync_graph_node_docs(client, tables, &derived_mutation).await?;
        sync_graph_refs(client, tables, &derived_mutation).await?;
    }
    Ok(Some(AppliedGraphMutation {
        event,
        inserted: true,
        mutated: true,
    }))
}

pub(super) async fn apply_graph_delete_mutation<C>(
    client: &C,
    tables: &Tables,
    mutation: GraphDeleteMutation,
) -> PostgresStoreResult<Option<AppliedGraphMutation>>
where
    C: GenericClient + Sync,
{
    require_namespace(&mutation.scope.namespace)?;
    if mutation.entity_id.is_empty() {
        return Err(StoreError::EmptyRecordId.into());
    }
    if mutation.event_id.is_empty() {
        return Err(PostgresStoreError::Backend(
            "graph delete event id must not be empty".to_owned(),
        ));
    }
    validate_graph_projection_target(&mutation.table, &mutation.entity_kind)?;
    advisory_lock(client, &mutation.event_id, 0).await?;
    if let Some(existing) = event_by_id(client, tables, &mutation.event_id).await? {
        if existing.namespace != mutation.scope.namespace {
            return Err(PostgresStoreError::EventIdNamespaceCollision {
                event_id: mutation.event_id,
                existing_namespace: existing.namespace,
                requested_namespace: mutation.scope.namespace,
            });
        }
        return Ok(Some(AppliedGraphMutation {
            event: raw_to_entity_event(existing)?,
            inserted: false,
            mutated: false,
        }));
    }

    let table = graph_table(tables, &mutation.table)?;
    let Some(row) = client
        .query_opt(
            &format!("SELECT metadata::TEXT FROM {table} WHERE id = $1 FOR UPDATE"),
            &[&mutation.entity_id],
        )
        .await
        .map_err(backend)?
    else {
        return Ok(None);
    };
    let current_metadata_json: String = row.try_get(0).map_err(backend)?;
    let current_metadata: Map<String, Value> = serde_json::from_str(&current_metadata_json)?;
    if !graph_scope_matches_or_legacy_default(&mutation.scope, &current_metadata)? {
        return Err(PostgresStoreError::Backend(
            "graph record id belongs to a different scope".to_owned(),
        ));
    }
    if matches!(mutation.entity_kind.as_str(), "node" | "edge")
        && !canonical_tombstone_present(&current_metadata)
    {
        return Err(PostgresStoreError::Backend(
            "physical graph purge requires a canonical tombstone".to_owned(),
        ));
    }

    let appended = append_raw_entity_event(
        client,
        tables,
        &mutation.scope.namespace,
        NewRawEntityEvent {
            event_id: mutation.event_id,
            entity_kind: mutation.entity_kind.clone(),
            entity_id: mutation.entity_id.clone(),
            op: "DELETE".to_owned(),
            payload_json: serde_json::to_string(&mutation.payload)?,
        },
    )
    .await?;
    let event = raw_to_entity_event(appended.event)?;
    client
        .execute(
            &format!("DELETE FROM {table} WHERE id = $1"),
            &[&mutation.entity_id],
        )
        .await
        .map_err(|error| {
            PostgresStoreError::Backend(format!("graph projection delete failed: {error:?}"))
        })?;
    if mutation.entity_kind == "edge" {
        client
            .execute(
                &format!(
                    "DELETE FROM {} WHERE metadata ->> 'edge_id' = $1",
                    tables.edge_endpoints
                ),
                &[&mutation.entity_id],
            )
            .await
            .map_err(backend)?;
    } else if mutation.entity_kind == "node" {
        client
            .execute(
                &format!(
                    "DELETE FROM {} WHERE metadata ->> 'node_id' = $1",
                    tables.node_docs
                ),
                &[&mutation.entity_id],
            )
            .await
            .map_err(backend)?;
        client
            .execute(
                &format!(
                    "DELETE FROM {} WHERE metadata ->> 'node_id' = $1",
                    tables.node_refs
                ),
                &[&mutation.entity_id],
            )
            .await
            .map_err(backend)?;
    } else if mutation.entity_kind == "edge" {
        client
            .execute(
                &format!(
                    "DELETE FROM {} WHERE metadata ->> 'edge_id' = $1",
                    tables.edge_refs
                ),
                &[&mutation.entity_id],
            )
            .await
            .map_err(backend)?;
    }
    Ok(Some(AppliedGraphMutation {
        event,
        inserted: true,
        mutated: true,
    }))
}

pub(super) fn canonical_tombstone_present(metadata: &Map<String, Value>) -> bool {
    matches!(
        metadata.get("lifecycle_status").and_then(Value::as_str),
        Some("tombstoned") | Some("deleted")
    )
}

pub(super) async fn upsert_graph_projection<C>(
    client: &C,
    tables: &Tables,
    write: GraphProjectionUpsert,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    require_namespace(&write.scope.namespace)?;
    if write.record.id.is_empty() {
        return Err(StoreError::EmptyRecordId.into());
    }
    let metadata = graph_metadata(&write.scope, &write.record.metadata)?;
    if let Some(embedding) = &write.record.embedding {
        finite_vector(embedding)?;
        if embedding.len() != write.embedding_dim {
            return Err(StoreError::VectorDimensionMismatch {
                expected: write.embedding_dim,
                actual: embedding.len(),
            }
            .into());
        }
    }
    let table = graph_table(tables, &write.table)?;
    if let Some(row) = client
        .query_opt(
            &format!("SELECT metadata::TEXT FROM {table} WHERE id = $1 FOR UPDATE"),
            &[&write.record.id],
        )
        .await
        .map_err(backend)?
    {
        let current_json: String = row.try_get(0).map_err(backend)?;
        let current: Map<String, Value> = serde_json::from_str(&current_json)?;
        if !graph_scope_matches_or_legacy_default(&write.scope, &current)? {
            return Err(PostgresStoreError::Backend(
                "graph record id belongs to a different scope".to_owned(),
            ));
        }
    }
    let metadata_json = serde_json::to_string(&metadata)?;
    let embedding_sql = write
        .record
        .embedding
        .as_deref()
        .map(|vector| format!("{}::vector", sql_text_literal(&vector_literal(vector))))
        .unwrap_or_else(|| "NULL".to_owned());
    client
        .execute(
            &format!(
                "INSERT INTO {table}(id, document, metadata, embedding) VALUES ($1, $2, {}::JSONB, {embedding_sql}) \
                 ON CONFLICT(id) DO UPDATE SET document = EXCLUDED.document, metadata = EXCLUDED.metadata, \
                 embedding = EXCLUDED.embedding, updated_at = NOW()",
                sql_text_literal(&metadata_json),
            ),
            &[&write.record.id, &write.record.document],
        )
        .await
        .map_err(|error| {
            PostgresStoreError::Backend(format!("graph projection upsert failed: {error:?}"))
        })?;
    Ok(())
}

pub(super) async fn delete_graph_projection_if_tombstoned<C>(
    client: &C,
    tables: &Tables,
    scope: &GraphScope,
    entity_kind: &str,
    entity_id: &str,
) -> PostgresStoreResult<bool>
where
    C: GenericClient + Sync,
{
    require_namespace(&scope.namespace)?;
    if entity_id.is_empty() {
        return Err(StoreError::EmptyRecordId.into());
    }
    let (table_name, derived_key) = match entity_kind {
        "node" => ("gke_nodes", "node_id"),
        "edge" => ("gke_edges", "edge_id"),
        _ => {
            return Err(StoreError::InvalidEntityEventPayload {
                message: "projection deletion supports node or edge only".to_owned(),
            }
            .into());
        }
    };
    let table = graph_table(tables, table_name)?;
    let Some(row) = client
        .query_opt(
            &format!("SELECT metadata::TEXT FROM {table} WHERE id = $1 FOR UPDATE"),
            &[&entity_id],
        )
        .await
        .map_err(backend)?
    else {
        return Ok(false);
    };
    let metadata_json: String = row.try_get(0).map_err(backend)?;
    let metadata: Map<String, Value> = serde_json::from_str(&metadata_json)?;
    if !graph_scope_matches_or_legacy_default(scope, &metadata)? {
        return Err(PostgresStoreError::Backend(
            "graph record id belongs to a different scope".to_owned(),
        ));
    }
    // A stale DELETE intent must never erase an active or replacement record.
    if !canonical_tombstone_present(&metadata) {
        return Ok(false);
    }

    client
        .execute(&format!("DELETE FROM {table} WHERE id = $1"), &[&entity_id])
        .await
        .map_err(backend)?;
    if entity_kind == "node" {
        for derived in [&tables.node_docs, &tables.node_refs] {
            client
                .execute(
                    &format!("DELETE FROM {derived} WHERE metadata ->> $1 = $2"),
                    &[&derived_key, &entity_id],
                )
                .await
                .map_err(backend)?;
        }
    } else {
        client
            .execute(
                &format!(
                    "DELETE FROM {} WHERE metadata ->> 'edge_id' = $1",
                    tables.edge_endpoints
                ),
                &[&entity_id],
            )
            .await
            .map_err(backend)?;
        client
            .execute(
                &format!(
                    "DELETE FROM {} WHERE metadata ->> 'edge_id' = $1",
                    tables.edge_refs
                ),
                &[&entity_id],
            )
            .await
            .map_err(backend)?;
    }
    Ok(true)
}

pub(super) async fn patch_graph_projection_metadata<C>(
    client: &C,
    tables: &Tables,
    patch: GraphProjectionMetadataPatch,
) -> PostgresStoreResult<bool>
where
    C: GenericClient + Sync,
{
    require_namespace(&patch.scope.namespace)?;
    if patch.entity_id.is_empty() {
        return Err(StoreError::EmptyRecordId.into());
    }
    let table = graph_table(tables, &patch.table)?;
    let Some(row) = client
        .query_opt(
            &format!("SELECT document, metadata::TEXT FROM {table} WHERE id = $1 FOR UPDATE"),
            &[&patch.entity_id],
        )
        .await
        .map_err(backend)?
    else {
        return Ok(false);
    };
    let document: Option<String> = row.try_get(0).map_err(backend)?;
    let current_metadata_json: String = row.try_get(1).map_err(backend)?;
    let mut current_metadata: Map<String, Value> = serde_json::from_str(&current_metadata_json)?;
    if !graph_scope_matches_or_legacy_default(&patch.scope, &current_metadata)? {
        return Err(PostgresStoreError::Backend(
            "graph record id belongs to a different scope".to_owned(),
        ));
    }
    current_metadata = graph_metadata(&patch.scope, &current_metadata)?;
    let validated_patch = graph_metadata(&patch.scope, &patch.metadata_patch)?;
    current_metadata.extend(validated_patch);
    let patched_document = if let Some(replacement) = patch.document {
        replacement
    } else if !patch.patch_document_metadata {
        document.unwrap_or_default()
    } else {
        let mut document_value = document
            .as_deref()
            .and_then(|value| serde_json::from_str::<Value>(value).ok())
            .and_then(|value| value.as_object().cloned())
            .unwrap_or_default();
        let mut document_metadata = document_value
            .remove("metadata")
            .and_then(|value| value.as_object().cloned())
            .unwrap_or_default();
        document_metadata.extend(patch.metadata_patch);
        document_value.insert("metadata".to_owned(), Value::Object(document_metadata));
        serde_json::to_string(&Value::Object(document_value))?
    };
    let metadata_json = serde_json::to_string(&current_metadata)?;
    client
        .execute(
            &format!(
                "UPDATE {table} SET document = $1, metadata = {}::JSONB, updated_at = NOW() WHERE id = $2",
                sql_text_literal(&metadata_json)
            ),
            &[&patched_document, &patch.entity_id],
        )
        .await
        .map_err(|error| {
            PostgresStoreError::Backend(format!("graph projection patch failed: {error:?}"))
        })?;
    Ok(true)
}

pub(super) async fn graph_projection_records<C>(
    client: &C,
    tables: &Tables,
    read: GraphProjectionRead,
    after_id: Option<String>,
    keyset_page: bool,
) -> PostgresStoreResult<Vec<GraphRecord>>
where
    C: GenericClient + Sync,
{
    graph_projection_records_with_lock(client, tables, read, after_id, keyset_page, false).await
}

pub(super) async fn graph_projection_records_with_lock<C>(
    client: &C,
    tables: &Tables,
    read: GraphProjectionRead,
    after_id: Option<String>,
    keyset_page: bool,
    for_update: bool,
) -> PostgresStoreResult<Vec<GraphRecord>>
where
    C: GenericClient + Sync,
{
    let table = graph_table(tables, &read.table)?;
    let (filter_sql, mut values) = graph_filter_sql(&read.scope, &read.metadata, 1)?;
    let mut clauses = vec![filter_sql];
    if let Some(ids) = read.ids {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let start = values.len() + 1;
        clauses.push(format!(
            "id IN ({})",
            (0..ids.len())
                .map(|index| format!("${}", start + index))
                .collect::<Vec<_>>()
                .join(", ")
        ));
        values.extend(ids);
    }
    if let Some(after_id) = after_id {
        let cursor_index = values.len() + 1;
        clauses.push(format!(
            "id COLLATE \"C\" > ${cursor_index}::TEXT COLLATE \"C\""
        ));
        values.push(after_id);
    }
    let limit = i64::try_from(read.limit).unwrap_or(i64::MAX).to_string();
    values.push(limit);
    let params: Vec<&(dyn tokio_postgres::types::ToSql + Sync)> = values
        .iter()
        .map(|value| value as &(dyn tokio_postgres::types::ToSql + Sync))
        .collect();
    let ordering = if keyset_page {
        "id COLLATE \"C\" ASC"
    } else {
        "id ASC"
    };
    let locking = if for_update { " FOR UPDATE" } else { "" };
    let rows = client
        .query(
            &format!(
                "SELECT id, document, metadata::TEXT, embedding::TEXT FROM {table} WHERE {} ORDER BY {ordering} LIMIT ${}::TEXT::BIGINT{locking}",
                clauses.join(" AND "),
                values.len()
            ),
            &params,
        )
        .await
        .map_err(backend)?;
    rows.iter().map(graph_record_from_row).collect()
}

pub(super) async fn graph_projection_vector_query<C>(
    client: &C,
    tables: &Tables,
    query: GraphProjectionVectorQuery,
    metadata_not_equals: &BTreeMap<String, Value>,
) -> PostgresStoreResult<Vec<VectorMatch>>
where
    C: GenericClient + Sync,
{
    if query.query.embedding.len() != query.embedding_dim {
        return Err(StoreError::VectorDimensionMismatch {
            expected: query.embedding_dim,
            actual: query.query.embedding.len(),
        }
        .into());
    }
    finite_vector(&query.query.embedding)?;
    let table = graph_table(tables, &query.table)?;
    let (filter_sql, mut values) = graph_filter_sql_with_exclusions(
        &query.scope,
        &query.query.metadata,
        metadata_not_equals,
        1,
    )?;
    let vector_sql = format!(
        "{}::vector",
        sql_text_literal(&vector_literal(&query.query.embedding))
    );
    let limit_index = values.len() + 1;
    values.push(
        i64::try_from(query.query.limit)
            .unwrap_or(i64::MAX)
            .to_string(),
    );
    let params: Vec<&(dyn tokio_postgres::types::ToSql + Sync)> = values
        .iter()
        .map(|value| value as &(dyn tokio_postgres::types::ToSql + Sync))
        .collect();
    let operator = match query.query.metric {
        kogwistar_store::DistanceMetric::Cosine => "<=>",
        kogwistar_store::DistanceMetric::L2 => "<->",
        kogwistar_store::DistanceMetric::InnerProduct => "<#>",
    };
    let rows = client
        .query(
            &format!(
                "SELECT id, document, metadata::TEXT, embedding::TEXT, embedding {operator} {vector_sql} AS distance \
                 FROM {table} WHERE embedding IS NOT NULL AND {filter_sql} \
                 ORDER BY distance ASC, id ASC LIMIT ${limit_index}::TEXT::BIGINT"
            ),
            &params,
        )
        .await
        .map_err(backend)?;
    rows.iter()
        .map(|row| {
            Ok(VectorMatch {
                record: graph_record_from_row(row)?,
                distance: row.try_get(4).map_err(backend)?,
            })
        })
        .collect()
}

pub(super) async fn graph_projection_vector_query_batch<C>(
    client: &C,
    tables: &Tables,
    queries: Vec<GraphProjectionVectorQuery>,
    metadata_not_equals: &BTreeMap<String, Value>,
) -> PostgresStoreResult<Vec<Vec<VectorMatch>>>
where
    C: GenericClient + Sync,
{
    let Some(first) = queries.first() else {
        return Ok(Vec::new());
    };
    require_namespace(&first.scope.namespace)?;
    let table = graph_table(tables, &first.table)?;
    let metric = first.query.metric;
    let mut query_vectors = Vec::with_capacity(queries.len());
    for (index, query) in queries.iter().enumerate() {
        if query.scope != first.scope
            || query.table != first.table
            || query.query.limit != first.query.limit
            || query.query.metadata != first.query.metadata
            || query.query.metric != metric
            || query.embedding_dim != first.embedding_dim
        {
            return Err(PostgresStoreError::Backend(
                "batched graph vector queries must share scope and search options".to_owned(),
            ));
        }
        if query.query.embedding.len() != query.embedding_dim {
            return Err(StoreError::VectorDimensionMismatch {
                expected: query.embedding_dim,
                actual: query.query.embedding.len(),
            }
            .into());
        }
        finite_vector(&query.query.embedding)?;
        let index = i64::try_from(index).map_err(|_| {
            PostgresStoreError::Backend("vector query batch exceeds PostgreSQL limits".to_owned())
        })?;
        query_vectors.push(format!(
            "({index}::BIGINT, {}::vector)",
            sql_text_literal(&vector_literal(&query.query.embedding))
        ));
    }
    let (filter_sql, mut values) = graph_filter_sql_with_exclusions(
        &first.scope,
        &first.query.metadata,
        metadata_not_equals,
        1,
    )?;
    let limit_index = values.len() + 1;
    values.push(
        i64::try_from(first.query.limit)
            .unwrap_or(i64::MAX)
            .to_string(),
    );
    let params: Vec<&(dyn tokio_postgres::types::ToSql + Sync)> = values
        .iter()
        .map(|value| value as &(dyn tokio_postgres::types::ToSql + Sync))
        .collect();
    let operator = match metric {
        kogwistar_store::DistanceMetric::Cosine => "<=>",
        kogwistar_store::DistanceMetric::L2 => "<->",
        kogwistar_store::DistanceMetric::InnerProduct => "<#>",
    };
    let rows = client
        .query(
            &format!(
                "WITH query_vectors(query_index, query_embedding) AS (VALUES {}) \
                 SELECT query_vectors.query_index, matches.id, matches.document, matches.metadata::TEXT, \
                        matches.embedding::TEXT, matches.distance \
                 FROM query_vectors CROSS JOIN LATERAL ( \
                     SELECT id, document, metadata, embedding, embedding {operator} query_vectors.query_embedding AS distance \
                     FROM {table} WHERE embedding IS NOT NULL AND {filter_sql} \
                     ORDER BY distance ASC, id ASC LIMIT ${limit_index}::TEXT::BIGINT \
                 ) AS matches ORDER BY query_vectors.query_index ASC, matches.distance ASC, matches.id ASC",
                query_vectors.join(", ")
            ),
            &params,
        )
        .await
        .map_err(backend)?;
    let mut result: Vec<Vec<VectorMatch>> = (0..queries.len()).map(|_| Vec::new()).collect();
    for row in &rows {
        let query_index: i64 = row.try_get(0).map_err(backend)?;
        let query_index = usize::try_from(query_index).map_err(|_| {
            PostgresStoreError::Backend("database returned invalid vector query index".to_owned())
        })?;
        let Some(matches) = result.get_mut(query_index) else {
            return Err(PostgresStoreError::Backend(
                "database returned out-of-range vector query index".to_owned(),
            ));
        };
        let metadata_json: String = row.try_get(3).map_err(backend)?;
        let metadata = serde_json::from_str(&metadata_json)?;
        let embedding_json: Option<String> = row.try_get(4).map_err(backend)?;
        let embedding = embedding_json
            .as_deref()
            .map(serde_json::from_str)
            .transpose()?;
        matches.push(VectorMatch {
            record: GraphRecord {
                id: row.try_get(1).map_err(backend)?,
                document: row.try_get(2).map_err(backend)?,
                metadata,
                embedding,
            },
            distance: row.try_get(5).map_err(backend)?,
        });
    }
    Ok(result)
}

pub(super) async fn graph_acl_records_for_entities<C>(
    client: &C,
    tables: &Tables,
    scope: &kogwistar_store::GraphScope,
    truth_graph: &str,
    entity_ids: &[String],
) -> PostgresStoreResult<Vec<GraphRecord>>
where
    C: GenericClient + Sync,
{
    if entity_ids.is_empty() {
        return Ok(Vec::new());
    }
    if entity_ids.len() > 10_000 {
        return Err(PostgresStoreError::Backend(
            "ACL entity lookup exceeds 10000 IDs".to_owned(),
        ));
    }
    let table = graph_table(tables, "gke_nodes")?;
    let filter = kogwistar_store::MetadataFilter {
        equals: [
            (
                "entity_type".to_owned(),
                Value::String("acl_record".to_owned()),
            ),
            (
                "acl_truth_graph".to_owned(),
                Value::String(truth_graph.to_owned()),
            ),
        ]
        .into_iter()
        .collect(),
    };
    let (scope_filter, values) = graph_filter_sql(scope, &filter, 1)?;
    let targets = entity_ids
        .iter()
        .map(|entity_id| {
            let json_id = serde_json::to_string(entity_id)?;
            Ok(format!(
                "metadata -> 'acl_target_entity_id' = {}::JSONB",
                sql_text_literal(&json_id)
            ))
        })
        .collect::<PostgresStoreResult<Vec<String>>>()?
        .join(" OR ");
    let rows = client
        .query(
            &format!(
                "SELECT id, document, metadata::TEXT, embedding::TEXT FROM {table} WHERE {scope_filter} AND ({targets}) ORDER BY id"
            ),
            &values
                .iter()
                .map(|value| value as &(dyn tokio_postgres::types::ToSql + Sync))
                .collect::<Vec<_>>(),
        )
        .await
        .map_err(backend)?;
    rows.iter().map(graph_record_from_row).collect()
}
