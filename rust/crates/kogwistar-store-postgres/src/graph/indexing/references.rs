use super::*;

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
