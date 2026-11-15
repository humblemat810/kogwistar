use kogwistar_store::{GraphRecord, StoreError};
use serde_json::{Map, Value, json};

pub(crate) const NAMESPACE_KEY: &str = "_kogwistar_namespace";
const ORIGINAL_METADATA_KEY: &str = "_kogwistar_original_metadata";
const CHROMA_JSON_METADATA_PREFIX: &str = "__kogwistar_json_v1__:";

pub(crate) fn node_index_metadata(
    metadata: &serde_json::Map<String, Value>,
) -> Result<Value, StoreError> {
    let mut safe = serde_json::Map::new();
    for (key, value) in metadata {
        if value.is_null() {
            continue;
        }
        let has_nested_value = value.is_object()
            || value
                .as_array()
                .is_some_and(|items| items.iter().any(|item| item.is_array() || item.is_object()));
        let encoded = if has_nested_value {
            Value::String(format!(
                "{CHROMA_JSON_METADATA_PREFIX}{}",
                serde_json::to_string(value).map_err(|error| StoreError::Backend {
                    backend: "chroma-node-index".to_owned(),
                    message: error.to_string(),
                })?
            ))
        } else if value.as_array().is_some_and(Vec::is_empty) {
            Value::String("[]".to_owned())
        } else {
            value.clone()
        };
        safe.insert(key.clone(), encoded);
    }
    Ok(Value::Object(safe))
}

pub(crate) fn restore_node_index_metadata(
    value: Option<&Value>,
) -> Result<serde_json::Map<String, Value>, StoreError> {
    let mut metadata = value
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    for item in metadata.values_mut() {
        if let Some(encoded) = item.as_str()
            && let Some(json) = encoded.strip_prefix(CHROMA_JSON_METADATA_PREFIX)
        {
            *item = serde_json::from_str(json).map_err(|error| StoreError::Backend {
                backend: "chroma-node-index".to_owned(),
                message: format!("stored compound metadata is invalid: {error}"),
            })?;
        }
    }
    Ok(metadata)
}

pub(crate) fn chroma_metadata(namespace: &str, record: &GraphRecord) -> Result<Value, StoreError> {
    if record.metadata.contains_key(NAMESPACE_KEY)
        || record.metadata.contains_key(ORIGINAL_METADATA_KEY)
    {
        return Err(StoreError::Backend {
            backend: "chroma".to_owned(),
            message: format!("metadata key {NAMESPACE_KEY:?} is reserved"),
        });
    }
    if record
        .metadata
        .get("namespace")
        .is_some_and(|value| value != namespace)
    {
        return Err(StoreError::Backend {
            backend: "chroma".to_owned(),
            message: "graph metadata namespace conflicts with the store namespace".to_owned(),
        });
    }
    let mut value = record.metadata.clone();
    value.insert(
        NAMESPACE_KEY.to_owned(),
        Value::String(namespace.to_owned()),
    );
    value.insert(
        ORIGINAL_METADATA_KEY.to_owned(),
        Value::String(Value::Object(record.metadata.clone()).to_string()),
    );
    let value = value
        .into_iter()
        .map(|(key, value)| (key, chroma_value(value)))
        .collect::<Map<String, Value>>();
    Ok(Value::Object(value))
}

pub(crate) fn query_where(namespace: &str, filter: &kogwistar_store::MetadataFilter) -> Value {
    // PYTHON_PARITY: kogwistar/engine_core/subsystems/write.py::WriteSubsystem._attach_graph_namespace
    // Python projections carry their graph namespace in metadata; retain the
    // reserved key alternative for records written by this Rust projection.
    query_where_with_exclusions(namespace, filter, &[], None)
}

pub(crate) fn query_where_with_exclusions(
    namespace: &str,
    filter: &kogwistar_store::MetadataFilter,
    metadata_not_equals: &[(String, Value)],
    numeric_lte: Option<(&str, f64)>,
) -> Value {
    let mut clauses = vec![json!({
        "$or": [
            {NAMESPACE_KEY: {"$eq": namespace}},
            {"namespace": {"$eq": namespace}}
        ]
    })];
    clauses.extend(
        filter
            .equals
            .iter()
            .map(|(key, value)| json!({key: {"$eq": chroma_filter_value(value)}})),
    );
    clauses.extend(
        metadata_not_equals
            .iter()
            .map(|(key, value)| json!({key: {"$ne": chroma_filter_value(value)}})),
    );
    if let Some((field, maximum)) = numeric_lte {
        clauses.push(json!({field: {"$lte": maximum}}));
    }
    if clauses.len() == 1 {
        clauses.remove(0)
    } else {
        json!({"$and": clauses})
    }
}

pub(crate) fn restore_record(
    id: String,
    document: Option<String>,
    metadata: Option<Value>,
    embedding: Option<Vec<f32>>,
) -> Result<GraphRecord, StoreError> {
    let mut metadata: Map<String, Value> = metadata
        .map(serde_json::to_value)
        .transpose()
        .map_err(|error| StoreError::Backend {
            backend: "chroma".to_owned(),
            message: error.to_string(),
        })?
        .and_then(|value| value.as_object().cloned())
        .unwrap_or_default();
    if let Some(original) = metadata.remove(ORIGINAL_METADATA_KEY) {
        let text = original.as_str().ok_or_else(|| StoreError::Backend {
            backend: "chroma".to_owned(),
            message: "stored canonical metadata payload is not a string".to_owned(),
        })?;
        metadata = serde_json::from_str::<Value>(text)
            .map_err(|error| StoreError::Backend {
                backend: "chroma".to_owned(),
                message: format!("stored canonical metadata payload is invalid: {error}"),
            })?
            .as_object()
            .cloned()
            .ok_or_else(|| StoreError::Backend {
                backend: "chroma".to_owned(),
                message: "stored canonical metadata payload is not an object".to_owned(),
            })?;
    } else {
        metadata.remove(NAMESPACE_KEY);
    }
    Ok(GraphRecord {
        id,
        document,
        metadata,
        embedding,
    })
}

pub(crate) fn validate_response_namespace(
    metadata: Option<&Value>,
    expected_namespace: &str,
) -> Result<(), StoreError> {
    let metadata = metadata
        .and_then(Value::as_object)
        .ok_or_else(|| StoreError::Backend {
            backend: "chroma".to_owned(),
            message: "returned record has no namespace metadata".to_owned(),
        })?;
    let reserved = metadata.get(NAMESPACE_KEY);
    let legacy = metadata.get("namespace");
    if reserved.is_none() && legacy.is_none() {
        return Err(StoreError::Backend {
            backend: "chroma".to_owned(),
            message: "returned record has no namespace metadata".to_owned(),
        });
    }
    for value in [reserved, legacy].into_iter().flatten() {
        if value.as_str() != Some(expected_namespace) {
            return Err(StoreError::Backend {
                backend: "chroma".to_owned(),
                message: "Chroma returned a record outside the requested namespace".to_owned(),
            });
        }
    }
    Ok(())
}

fn chroma_filter_value(value: &Value) -> Value {
    if value.is_null() || value.is_array() || value.is_object() {
        Value::String(value.to_string())
    } else {
        value.clone()
    }
}

fn chroma_value(value: Value) -> Value {
    if value.is_null() || value.is_array() || value.is_object() {
        Value::String(value.to_string())
    } else {
        value
    }
}

pub(crate) fn validate_namespace(namespace: &str) -> Result<(), StoreError> {
    if namespace.trim().is_empty() {
        Err(StoreError::EmptyNamespace)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_metadata_round_trips_nested_json_and_keeps_filter_fields() {
        let record = GraphRecord {
            id: "n1".to_owned(),
            document: Some("source text".to_owned()),
            metadata: serde_json::from_value(json!({
                "kind": "node",
                "nested": {"revision": 2},
                "labels": ["a", "b"]
            }))
            .expect("metadata object"),
            embedding: Some(vec![0.1, 0.2]),
        };
        let chroma = chroma_metadata("workspace-a", &record).expect("Chroma metadata");
        assert_eq!(chroma["kind"], "node");
        assert_eq!(chroma["nested"], "{\"revision\":2}");
        assert_eq!(chroma[NAMESPACE_KEY], "workspace-a");
        let restored = restore_record(
            record.id.clone(),
            record.document.clone(),
            Some(chroma),
            record.embedding.clone(),
        )
        .expect("restored record");
        assert_eq!(restored, record);
    }

    #[test]
    fn node_index_metadata_matches_python_chroma_encoding() {
        let metadata = serde_json::from_value(json!({
            "target_node_id": "node-1",
            "keywords": ["water", "safety"],
            "nested": {"revision": 2},
            "omitted": null,
            "empty": []
        }))
        .expect("metadata object");
        let encoded = node_index_metadata(&metadata).expect("encode metadata");
        assert_eq!(encoded["target_node_id"], "node-1");
        assert_eq!(encoded["keywords"], json!(["water", "safety"]));
        assert_eq!(encoded["nested"], "__kogwistar_json_v1__:{\"revision\":2}");
        assert_eq!(encoded["empty"], "[]");
        assert!(encoded.get("omitted").is_none());
        let restored = restore_node_index_metadata(Some(&encoded)).expect("restore metadata");
        let mut expected = metadata;
        expected.remove("omitted");
        expected.insert("empty".to_owned(), Value::String("[]".to_owned()));
        assert_eq!(restored, expected);
    }

    #[test]
    fn metadata_filter_is_namespace_scoped_and_preserves_exact_values() {
        let filter = kogwistar_store::MetadataFilter {
            equals: [("kind".to_owned(), json!("node"))].into(),
        };
        let query = query_where_with_exclusions(
            "workspace-a",
            &filter,
            &[
                ("entity_type".to_owned(), json!("acl_record")),
                ("lifecycle_status".to_owned(), json!("tombstoned")),
                ("lifecycle_status".to_owned(), json!("deleted")),
            ],
            None,
        );
        assert_eq!(
            query["$and"][0]["$or"][0][NAMESPACE_KEY]["$eq"],
            "workspace-a"
        );
        assert_eq!(
            query["$and"][0]["$or"][1]["namespace"]["$eq"],
            "workspace-a"
        );
        assert_eq!(query["$and"][1]["kind"]["$eq"], "node");
        assert_eq!(query["$and"][2]["entity_type"]["$ne"], "acl_record");
        assert_eq!(query["$and"][3]["lifecycle_status"]["$ne"], "tombstoned");
        assert_eq!(query["$and"][4]["lifecycle_status"]["$ne"], "deleted");
    }

    #[test]
    fn response_namespace_validation_accepts_rust_and_python_metadata() {
        assert!(
            validate_response_namespace(Some(&json!({NAMESPACE_KEY:"workspace-a"})), "workspace-a")
                .is_ok()
        );
        assert!(
            validate_response_namespace(Some(&json!({"namespace":"workspace-a"})), "workspace-a")
                .is_ok()
        );
        assert!(
            validate_response_namespace(
                Some(&json!({NAMESPACE_KEY:"workspace-a","namespace":"workspace-b"})),
                "workspace-a"
            )
            .is_err()
        );
        assert!(
            validate_response_namespace(Some(&json!({NAMESPACE_KEY:"workspace-b"})), "workspace-a")
                .is_err()
        );
        assert!(validate_response_namespace(Some(&json!({})), "workspace-a").is_err());
        assert!(validate_response_namespace(None, "workspace-a").is_err());
    }

    #[test]
    fn numeric_lte_vector_filter_is_combined_with_namespace_and_exact_filters() {
        let filter = kogwistar_store::MetadataFilter {
            equals: [("kind".to_owned(), json!("node"))].into(),
        };
        let query = query_where_with_exclusions(
            "workspace-a",
            &filter,
            &[],
            Some(("level_from_root", 2.0)),
        );
        assert_eq!(query["$and"][1]["kind"]["$eq"], "node");
        assert_eq!(query["$and"][2]["level_from_root"]["$lte"], 2.0);
    }

    #[test]
    fn graph_metadata_cannot_claim_another_namespace() {
        let record = GraphRecord {
            id: "x".to_owned(),
            document: None,
            metadata: serde_json::from_value(json!({"namespace": "other"}))
                .expect("metadata object"),
            embedding: None,
        };
        assert!(chroma_metadata("workspace-a", &record).is_err());
    }

    #[test]
    fn reserved_namespace_metadata_key_is_rejected() {
        let record = GraphRecord {
            id: "x".to_owned(),
            document: None,
            metadata: serde_json::from_value(json!({NAMESPACE_KEY: "spoofed"}))
                .expect("metadata object"),
            embedding: None,
        };
        assert!(chroma_metadata("real", &record).is_err());
    }
}
