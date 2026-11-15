use kogwistar_runtime::{
    NativeOperationOutput, NativeOperationRegistrationError, NativeOperationRegistry,
    RecordedStateUpdate, RuntimeNativeExecutionError, RuntimeNativeStepRequest,
};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::collections::BTreeSet;

const AUTHORIZED_RECORDS_KEY: &str = "authorized_evidence_records";
const MAX_EVIDENCE_IDS: usize = 100;
const MAX_CHARS_PER_ITEM: usize = 100_000;
const MAX_TOTAL_CHARS: usize = 1_000_000;

pub(crate) fn register(
    registry: &mut NativeOperationRegistry,
) -> Result<(), NativeOperationRegistrationError> {
    registry.register(
        "aa_materialize_evidence_pack",
        "v1",
        execute_materialize_evidence_pack,
    )
}

fn invalid(field: &'static str, reason: &'static str) -> RuntimeNativeExecutionError {
    RuntimeNativeExecutionError::InvalidState { field, reason }
}

fn ids(
    state: &Map<String, Value>,
    field: &'static str,
) -> Result<Vec<String>, RuntimeNativeExecutionError> {
    let Some(value) = state.get(field) else {
        return Ok(Vec::new());
    };
    let Some(values) = value.as_array() else {
        return Err(invalid(field, "expected an array of non-empty string IDs"));
    };
    if values.len() > MAX_EVIDENCE_IDS {
        return Err(invalid(field, "evidence ID count exceeds the native limit"));
    }
    let ids = values
        .iter()
        .map(|value| {
            value
                .as_str()
                .filter(|id| !id.trim().is_empty())
                .map(str::to_owned)
                .ok_or_else(|| invalid(field, "expected an array of non-empty string IDs"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if ids.iter().collect::<BTreeSet<_>>().len() != ids.len() {
        return Err(invalid(field, "duplicate evidence IDs are not allowed"));
    }
    Ok(ids)
}

fn bounded_limit(
    state: &Map<String, Value>,
    field: &'static str,
    default: usize,
    hard_limit: usize,
) -> Result<usize, RuntimeNativeExecutionError> {
    let Some(value) = state.get(field) else {
        return Ok(default);
    };
    let value = value
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| invalid(field, "expected a non-negative integer"))?;
    if value > hard_limit {
        return Err(invalid(
            field,
            "configured value exceeds the native safety limit",
        ));
    }
    Ok(value)
}

fn text(value: Option<&Value>) -> &str {
    value.and_then(Value::as_str).unwrap_or_default()
}

fn take_chars(value: &str, limit: usize) -> String {
    value.chars().take(limit).collect()
}

fn metadata_record_map(
    records: &Value,
    collection: &'static str,
) -> Result<BTreeMap<String, Map<String, Value>>, RuntimeNativeExecutionError> {
    let values = records
        .get(collection)
        .and_then(Value::as_array)
        .ok_or_else(|| {
            invalid(
                AUTHORIZED_RECORDS_KEY,
                "authorized record collection is missing",
            )
        })?;
    let mut result = BTreeMap::new();
    for value in values {
        let record = value.as_object().ok_or_else(|| {
            invalid(
                AUTHORIZED_RECORDS_KEY,
                "authorized record must be an object",
            )
        })?;
        let id = record
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
            .ok_or_else(|| invalid(AUTHORIZED_RECORDS_KEY, "authorized record ID is missing"))?;
        let metadata = record
            .get("metadata")
            .and_then(Value::as_object)
            .cloned()
            .ok_or_else(|| {
                invalid(
                    AUTHORIZED_RECORDS_KEY,
                    "authorized record metadata is missing",
                )
            })?;
        if result.insert(id.to_owned(), metadata).is_some() {
            return Err(invalid(
                AUTHORIZED_RECORDS_KEY,
                "duplicate authorized record ID",
            ));
        }
    }
    Ok(result)
}

fn record_document<'a>(records: &'a Value, collection: &str, id: &str) -> Option<&'a str> {
    records
        .get(collection)?
        .as_array()?
        .iter()
        .find(|record| record.get("id").and_then(Value::as_str) == Some(id))?
        .get("document")
        .and_then(Value::as_str)
}

fn normalized_mentions(metadata: &Map<String, Value>, text: &str, limit: usize) -> Value {
    let mentions = metadata.get("mentions").and_then(Value::as_array);
    if let Some(mentions) = mentions {
        let normalized = mentions
            .iter()
            .filter_map(|mention| {
                let spans = mention.get("spans")?.as_array()?;
                let spans = spans
                    .iter()
                    .filter_map(|span| {
                        let span = span.as_object()?;
                        let excerpt = span
                            .get("excerpt")
                            .or_else(|| span.get("verbatim_text"))
                            .or_else(|| span.get("text"))
                            .and_then(Value::as_str)
                            .unwrap_or_default();
                        let mut projected = Map::new();
                        projected.insert("excerpt".to_owned(), json!(take_chars(excerpt, limit)));
                        for (target, sources) in [
                            ("doc_id", &["doc_id"][..]),
                            ("page_number", &["page_number", "page"][..]),
                            ("document_page_url", &["document_page_url"][..]),
                        ] {
                            if let Some(value) = sources.iter().find_map(|source| {
                                span.get(*source).filter(|value| !value.is_null()).cloned()
                            }) {
                                projected.insert(target.to_owned(), value);
                            }
                        }
                        Some(Value::Object(projected))
                    })
                    .collect::<Vec<_>>();
                (!spans.is_empty()).then(|| json!({"spans": spans}))
            })
            .collect::<Vec<_>>();
        if !normalized.is_empty() {
            return Value::Array(normalized);
        }
    }
    json!([{
        "spans": [{
            "excerpt": take_chars(text, limit),
            "doc_id": metadata.get("doc_id").cloned().unwrap_or(Value::Null),
            "page_number": metadata.get("page_number").cloned().unwrap_or(Value::Null),
            "document_page_url": metadata.get("document_page_url").cloned().unwrap_or(Value::Null)
        }]
    }])
}

fn execute_materialize_evidence_pack(
    request: &RuntimeNativeStepRequest,
) -> Result<NativeOperationOutput, RuntimeNativeExecutionError> {
    let state = &request.state;
    let node_ids = ids(state, "used_node_ids")?;
    let edge_ids = ids(state, "used_edge_ids")?;
    let records = request
        .ephemeral_context
        .get(AUTHORIZED_RECORDS_KEY)
        .filter(|value| value.is_object())
        .ok_or_else(|| {
            invalid(
                AUTHORIZED_RECORDS_KEY,
                "host-authorized graph read is required",
            )
        })?;
    let nodes = metadata_record_map(records, "nodes")?;
    let edges = metadata_record_map(records, "edges")?;
    if node_ids.iter().any(|id| !nodes.contains_key(id))
        || edge_ids.iter().any(|id| !edges.contains_key(id))
    {
        return Err(invalid(
            AUTHORIZED_RECORDS_KEY,
            "authorized graph read does not cover every selected evidence ID",
        ));
    }

    let depth = state
        .get("materialize_depth")
        .and_then(Value::as_str)
        .filter(|depth| !depth.is_empty())
        .unwrap_or("shallow");
    if !matches!(depth, "shallow" | "deep") {
        return Err(invalid("materialize_depth", "expected shallow or deep"));
    }
    let max_item = bounded_limit(state, "max_chars_per_item", 900, MAX_CHARS_PER_ITEM)?;
    let max_total = bounded_limit(state, "max_total_chars", 12_000, MAX_TOTAL_CHARS)?;
    let mut total = 0usize;
    let mut packed_nodes = Vec::new();
    for id in &node_ids {
        let metadata = &nodes[id];
        let doc = record_document(records, "nodes", id).unwrap_or_default();
        let label = metadata
            .get("label")
            .or_else(|| metadata.get("title"))
            .and_then(Value::as_str)
            .unwrap_or_default();
        let summary = text(metadata.get("summary"));
        let source = if depth == "deep" && !doc.trim().is_empty() {
            doc.trim()
        } else if !summary.trim().is_empty() {
            summary.trim()
        } else {
            doc.trim()
        };
        let item_text = take_chars(source, max_item);
        if total.saturating_add(item_text.chars().count()) > max_total {
            break;
        }
        total += item_text.chars().count();
        packed_nodes.push(json!({
            "node_id": id,
            "label": label,
            "summary": take_chars(summary, 300),
            "text": item_text,
            "mentions": normalized_mentions(metadata, &item_text, max_item)
        }));
    }

    let packed_edges = edge_ids
        .iter()
        .map(|id| {
            let metadata = &edges[id];
            json!({
                "id": id,
                "label": metadata.get("label").cloned().unwrap_or(Value::Null),
                "relation": metadata.get("relation").or_else(|| metadata.get("type")).cloned().unwrap_or(Value::Null),
                "summary": metadata.get("summary").cloned().unwrap_or(Value::Null)
            })
        })
        .collect::<Vec<_>>();
    let pack = json!({"nodes": packed_nodes, "edges": packed_edges});
    let canonical_pack = serde_json::to_vec(&pack).map_err(|_| {
        invalid(
            "evidence_pack",
            "could not serialize the canonical evidence pack",
        )
    })?;
    let hash = format!("{:x}", Sha256::digest(canonical_pack));
    let digest = json!({
        "node_ids": node_ids,
        "edge_ids": edge_ids,
        "depth": depth,
        "max_chars_per_item": max_item,
        "max_total_chars": max_total,
        "evidence_pack_hash": hash
    });
    let mut update = Map::new();
    update.insert("evidence_pack_digest".to_owned(), digest);
    let log_entry = "aa_materialize_evidence_pack";
    let mut op_log = state
        .get("op_log")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    op_log.push(json!(log_entry));
    let mut auxiliary = Map::new();
    auxiliary.insert("op_log".to_owned(), json!(log_entry));
    let mut final_state = state.clone();
    final_state.remove(AUTHORIZED_RECORDS_KEY);
    let mut runtime_state = state
        .get("_rt")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    runtime_state.insert("evidence_pack".to_owned(), pack);
    update.insert("_rt".to_owned(), Value::Object(runtime_state));
    final_state.extend(update.clone());
    final_state.insert("op_log".to_owned(), Value::Array(op_log));
    Ok(NativeOperationOutput {
        state_update: vec![
            RecordedStateUpdate::Pair(("u".to_owned(), update)),
            RecordedStateUpdate::Pair(("a".to_owned(), auxiliary)),
        ],
        result: Some(json!({
            "workflow_status": "succeeded",
            "final_state": final_state,
            "conversation_node_id": null
        })),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(state: Value) -> RuntimeNativeStepRequest {
        let mut state = state;
        let records = state
            .as_object_mut()
            .and_then(|state| state.remove("_rt_authoritative_evidence_records"));
        let ephemeral_context = records
            .map(|records| Map::from_iter([(AUTHORIZED_RECORDS_KEY.to_owned(), records)]))
            .unwrap_or_default();
        RuntimeNativeStepRequest {
            executor_target: kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET.to_owned(),
            operation: "aa_materialize_evidence_pack".to_owned(),
            op_version: "v1".to_owned(),
            run_id: "run-1".to_owned(),
            workflow_id: "agentic_answering.v2".to_owned(),
            node_id: "materialize".to_owned(),
            conversation_id: "conv-1".to_owned(),
            turn_node_id: None,
            token_id: "token-1".to_owned(),
            parent_token_id: None,
            step_seq: 3,
            state: serde_json::from_value(state).unwrap(),
            runtime_routes: Vec::new(),
            ephemeral_context,
        }
    }

    fn state() -> Value {
        json!({
            "used_node_ids": ["node-1"],
            "used_edge_ids": ["edge-1"],
            "materialize_depth": "deep",
            "_rt_authoritative_evidence_records": {
                "nodes": [{
                    "id": "node-1",
                    "document": "A longer source document.",
                    "metadata": {
                        "label": "Example",
                        "summary": "Short summary",
                        "doc_id": "doc-1",
                        "mentions": [{"spans": [{"excerpt": "Verified source", "doc_id": "doc-1", "page_number": 2}]}]
                    }
                }],
                "edges": [{
                    "id": "edge-1",
                    "document": null,
                    "metadata": {"label": "relates", "relation": "supports", "summary": "Edge summary"}
                }]
            }
        })
    }

    #[test]
    fn materializer_projects_authorized_records_and_drops_ephemeral_context() {
        let mut registry = NativeOperationRegistry::default();
        register(&mut registry).unwrap();
        let effect =
            kogwistar_runtime::execute_native_step_with_registry(&request(state()), &registry)
                .unwrap();
        let final_state = &effect.result.as_ref().unwrap()["final_state"];
        assert_eq!(
            final_state["_rt"]["evidence_pack"]["nodes"][0]["text"],
            "A longer source document."
        );
        assert_eq!(
            final_state["_rt"]["evidence_pack"]["nodes"][0]["mentions"][0]["spans"][0]["excerpt"],
            "Verified source"
        );
        assert_eq!(
            final_state["_rt"]["evidence_pack"]["edges"][0]["relation"],
            "supports"
        );
        assert!(
            final_state
                .get("_rt_authoritative_evidence_records")
                .is_none()
        );
        assert!(
            final_state["evidence_pack_digest"]["evidence_pack_hash"]
                .as_str()
                .is_some()
        );
    }

    #[test]
    fn materializer_requires_complete_host_authorized_record_set() {
        let mut registry = NativeOperationRegistry::default();
        register(&mut registry).unwrap();
        let mut input = state();
        input[AUTHORIZED_RECORDS_KEY]["nodes"] = json!([]);
        assert!(
            kogwistar_runtime::execute_native_step_with_registry(&request(input), &registry)
                .is_err()
        );
    }

    #[test]
    fn materializer_fails_closed_without_host_read_context_and_bounds_limits() {
        let mut registry = NativeOperationRegistry::default();
        register(&mut registry).unwrap();
        let mut input = state();
        input
            .as_object_mut()
            .unwrap()
            .remove("_rt_authoritative_evidence_records");
        assert!(
            kogwistar_runtime::execute_native_step_with_registry(&request(input), &registry)
                .is_err()
        );
        let mut input = state();
        input["max_total_chars"] = json!(MAX_TOTAL_CHARS + 1);
        assert!(
            kogwistar_runtime::execute_native_step_with_registry(&request(input), &registry)
                .is_err()
        );
    }
}
