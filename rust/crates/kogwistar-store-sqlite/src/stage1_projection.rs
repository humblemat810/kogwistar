//! Canonical graph event to SQLite Stage-1 graph projection conversion.

use kogwistar_contracts::{GraphModelClass, GraphModelEnvelope, canonical_json};
use kogwistar_store::{NamedProjectionWrite, StoreError};
use serde_json::{Map, Value};

pub const STAGE1_PROJECTION_SCHEMA_VERSION: i64 = 2;

pub fn stage1_projection_from_canonical_event(
    namespace: &str,
    entity_kind: &str,
    entity_id: &str,
    revision: i64,
    payload: &Value,
    fingerprint: &str,
) -> Result<NamedProjectionWrite, StoreError> {
    let model_class = match entity_kind {
        "node" => GraphModelClass::Node,
        "edge" => GraphModelClass::Edge,
        _ => {
            return Err(invalid(
                "Stage-1 projection supports only Node and Edge events",
            ));
        }
    };
    GraphModelEnvelope::new(model_class, payload.clone())
        .decode_persistent_v1()
        .map_err(|error| {
            invalid(format!(
                "canonical graph model cannot be projected: {error}"
            ))
        })?;

    let mut document_payload = payload.clone();
    let Some(entity) = document_payload.as_object_mut() else {
        return Err(invalid("canonical graph model payload must be an object"));
    };
    if entity.get("id").and_then(Value::as_str) != Some(entity_id) {
        return Err(invalid(
            "canonical graph model ID does not match the event identity",
        ));
    }
    entity.remove("embedding");
    if entity_kind == "node" {
        entity.remove("metadata");
    }
    let document = canonical_json(&document_payload, false);

    let source_metadata = payload.get("metadata").and_then(Value::as_object);
    let mut metadata = source_metadata.cloned().unwrap_or_default();
    if metadata
        .get("namespace")
        .is_some_and(|value| !value.is_null() && value.as_str() != Some(namespace))
    {
        return Err(invalid(
            "canonical graph metadata namespace conflicts with its event scope",
        ));
    }
    metadata.insert("namespace".to_owned(), Value::String(namespace.to_owned()));
    if entity_kind == "node" {
        metadata.retain(|_, value| !value.is_null());
        for field in [
            "doc_id",
            "label",
            "type",
            "summary",
            "domain_id",
            "canonical_entity_id",
        ] {
            copy_non_null(payload, field, &mut metadata);
        }
        for field in ["properties", "mentions"] {
            if let Some(value) = payload.get(field).filter(|value| !value.is_null()) {
                metadata.insert(
                    field.to_owned(),
                    Value::String(canonical_json(value, false)),
                );
            }
        }
    } else {
        for field in [
            "doc_id",
            "relation",
            "type",
            "summary",
            "domain_id",
            "canonical_entity_id",
        ] {
            copy_non_null(payload, field, &mut metadata);
        }
        for field in [
            "source_ids",
            "target_ids",
            "source_edge_ids",
            "target_edge_ids",
            "properties",
        ] {
            if let Some(value) = payload.get(field).filter(|value| !value.is_null()) {
                metadata.insert(
                    field.to_owned(),
                    Value::String(canonical_json(value, false)),
                );
            }
        }
        if let Some(value) = payload.get("mentions").filter(|value| !value.is_null()) {
            metadata.insert(
                "references".to_owned(),
                Value::String(canonical_json(value, false)),
            );
        }
        for (target, sources) in [
            ("wf_priority", ["wf_priority", "priority"]),
            ("wf_is_default", ["wf_is_default", "is_default"]),
            ("wf_predicate", ["wf_predicate", "predicate"]),
            ("wf_multiplicity", ["wf_multiplicity", "multiplicity"]),
        ] {
            if let Some(value) = sources
                .iter()
                .filter_map(|key| payload["metadata"].get(*key))
                .find(|value| json_truthy(value))
            {
                metadata.insert(target.to_owned(), value.clone());
            }
        }
        let node_endpoints = ["source_ids", "target_ids"]
            .iter()
            .filter_map(|key| payload.get(*key).and_then(Value::as_array))
            .map(Vec::len)
            .sum::<usize>();
        let edge_endpoints = ["source_edge_ids", "target_edge_ids"]
            .iter()
            .filter_map(|key| payload.get(*key).and_then(Value::as_array))
            .map(Vec::len)
            .sum::<usize>();
        metadata.insert(
            "node_endpoint_count".to_owned(),
            Value::from(node_endpoints),
        );
        metadata.insert(
            "edge_endpoint_count".to_owned(),
            Value::from(edge_endpoints),
        );
        metadata.insert(
            "total_endpoint_count".to_owned(),
            Value::from(node_endpoints + edge_endpoints),
        );
    }
    metadata
        .entry("lifecycle_status".to_owned())
        .or_insert_with(|| Value::String("active".to_owned()));
    if entity_kind == "edge" {
        metadata
            .entry("redirect_to_id".to_owned())
            .or_insert(Value::Null);
    }

    let projection_payload = serde_json::json!({
        "id": entity_id,
        "entity_kind": entity_kind,
        "document": document,
        "metadata": metadata,
        "source_fingerprint": fingerprint,
    });
    Ok(NamedProjectionWrite {
        payload: projection_payload.as_object().cloned().unwrap_or_default(),
        last_authoritative_seq: revision,
        last_materialized_seq: revision,
        projection_schema_version: STAGE1_PROJECTION_SCHEMA_VERSION,
        materialization_status: "pending".to_owned(),
    })
}

fn copy_non_null(payload: &Value, field: &str, metadata: &mut Map<String, Value>) {
    if let Some(value) = payload.get(field).filter(|value| !value.is_null()) {
        metadata.insert(field.to_owned(), value.clone());
    }
}

fn json_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(value) => value.as_f64().is_some_and(|number| number != 0.0),
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
    }
}

fn invalid(message: impl Into<String>) -> StoreError {
    StoreError::InvalidEntityEventPayload {
        message: message.into(),
    }
}
