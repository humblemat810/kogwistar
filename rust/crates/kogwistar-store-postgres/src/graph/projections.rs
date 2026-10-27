use super::super::*;

pub(crate) fn canonical_tombstone_present(metadata: &Map<String, Value>) -> bool {
    matches!(
        metadata.get("lifecycle_status").and_then(Value::as_str),
        Some("tombstoned") | Some("deleted")
    )
}

pub(crate) async fn upsert_graph_projection<C>(
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

pub(crate) async fn delete_graph_projection_if_tombstoned<C>(
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

pub(crate) async fn patch_graph_projection_metadata<C>(
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
