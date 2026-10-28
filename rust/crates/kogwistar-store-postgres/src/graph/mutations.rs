use super::super::*;

pub(crate) async fn apply_graph_mutation<C>(
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

pub(crate) async fn apply_graph_metadata_patch_mutation<C>(
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

pub(crate) async fn apply_graph_delete_mutation<C>(
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
