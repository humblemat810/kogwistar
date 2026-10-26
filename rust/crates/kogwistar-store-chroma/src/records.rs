use kogwistar_store::{GraphRecord, StoreError};
use serde_json::{Map, Value, json};

pub(crate) const NAMESPACE_KEY: &str = "_kogwistar_namespace";
const ORIGINAL_METADATA_KEY: &str = "_kogwistar_original_metadata";

pub(crate) fn chroma_metadata(namespace: &str, record: &GraphRecord) -> Result<Value, StoreError> {
    if record.metadata.contains_key(NAMESPACE_KEY)
        || record.metadata.contains_key(ORIGINAL_METADATA_KEY)
    {
        return Err(StoreError::Backend {
            backend: "chroma".to_owned(),
            message: format!("metadata key {NAMESPACE_KEY:?} is reserved"),
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
    let mut clauses = vec![json!({NAMESPACE_KEY: {"$eq": namespace}})];
    clauses.extend(
        filter
            .equals
            .iter()
            .map(|(key, value)| json!({key: {"$eq": chroma_filter_value(value)}})),
    );
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
    fn metadata_filter_is_namespace_scoped_and_preserves_exact_values() {
        let filter = kogwistar_store::MetadataFilter {
            equals: [("kind".to_owned(), json!("node"))].into(),
        };
        let query = query_where("workspace-a", &filter);
        assert_eq!(query["$and"][0][NAMESPACE_KEY]["$eq"], "workspace-a");
        assert_eq!(query["$and"][1]["kind"]["$eq"], "node");
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
