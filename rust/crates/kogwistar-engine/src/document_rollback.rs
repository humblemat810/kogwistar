//! Pure source-grounding edits used by document rollback.
//!
//! Python parity: `kogwistar/engine_core/subsystems/rollback.py`
//! (`RollbackSubsystem._filter_mentions_for_document`).

use kogwistar_contracts::{canonical_json, stable_id};
use serde_json::{Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum GroundingRollbackError {
    #[error("grounding entry must be an object")]
    InvalidGrounding,
    #[error("grounding spans must be an array")]
    InvalidSpans,
    #[error("grounding span must be an object")]
    InvalidSpan,
    #[error("rollback snapshot must be an active canonical node")]
    UnsupportedSnapshot,
    #[error("canonical node payload must be an object with object metadata")]
    InvalidNodePayload,
    #[error("canonical graph payload mentions must be an array or null")]
    InvalidMentions,
    #[error("rollback snapshot must be an active canonical edge")]
    UnsupportedEdgeSnapshot,
    #[error("canonical edge payload identity, metadata, or endpoint fields are invalid")]
    InvalidEdgePayload,
}

/// Pure node-level decision used by backend transaction owners.
#[derive(Clone, Debug, PartialEq)]
pub enum NodeRollbackDecision {
    Tombstone {
        entity_id: String,
        revision: i64,
    },
    Replace {
        entity_id: String,
        revision: i64,
        replacement_id: String,
        replacement_payload: Value,
    },
}

/// Edge rewrite/tombstone decision. Caller owns endpoint discovery and writes.
#[derive(Clone, Debug, PartialEq)]
pub enum EdgeRollbackDecision {
    Tombstone {
        entity_id: String,
        revision: i64,
    },
    Replace {
        entity_id: String,
        revision: i64,
        replacement_id: String,
        replacement_payload: Value,
    },
}

/// Plan one canonical edge using Python rollback's node/edge redirect maps.
/// `canonical_entity_ids` mirrors `AdjudicationSubsystem.choose_anchor`: choose
/// first surviving canonical entity in endpoint order, otherwise lexicographic ID.
pub fn plan_edge_rollback(
    snapshot: &crate::canonical_revision::CanonicalGraphEntitySnapshot,
    document_id: &str,
    node_redirects: &BTreeMap<String, String>,
    removed_node_ids: &BTreeSet<String>,
    edge_redirects: &BTreeMap<String, String>,
    removed_edge_ids: &BTreeSet<String>,
    canonical_entity_ids: &BTreeMap<String, String>,
) -> Result<Option<EdgeRollbackDecision>, GroundingRollbackError> {
    if snapshot.entity_kind != "edge" || snapshot.state != "active" {
        return Err(GroundingRollbackError::UnsupportedEdgeSnapshot);
    }
    let Some(payload) = snapshot.payload.as_ref().and_then(Value::as_object) else {
        return Err(GroundingRollbackError::InvalidEdgePayload);
    };
    if payload.get("id").and_then(Value::as_str) != Some(snapshot.entity_id.as_str())
        || !payload.get("metadata").is_some_and(Value::is_object)
    {
        return Err(GroundingRollbackError::InvalidEdgePayload);
    }
    let mentions = match payload.get("mentions") {
        None | Some(Value::Null) => &[][..],
        Some(Value::Array(mentions)) => mentions.as_slice(),
        Some(_) => return Err(GroundingRollbackError::InvalidMentions),
    };
    let (surviving_mentions, mentions_changed) =
        filter_mentions_for_document(mentions, document_id)?;

    let original_source_ids = string_array(payload, "source_ids")?;
    let original_target_ids = string_array(payload, "target_ids")?;
    let original_source_edge_ids = string_array(payload, "source_edge_ids")?;
    let original_target_edge_ids = string_array(payload, "target_edge_ids")?;
    let mapped_source_ids = original_source_ids
        .iter()
        .filter(|id| !removed_node_ids.contains(*id))
        .map(|id| node_redirects.get(id).unwrap_or(id).clone())
        .collect::<Vec<_>>();
    let mapped_target_ids = original_target_ids
        .iter()
        .filter(|id| !removed_node_ids.contains(*id))
        .map(|id| node_redirects.get(id).unwrap_or(id).clone())
        .collect::<Vec<_>>();
    let mapped_source_edge_ids =
        redirected_edge_ids(&original_source_edge_ids, edge_redirects, removed_edge_ids);
    let mapped_target_edge_ids =
        redirected_edge_ids(&original_target_edge_ids, edge_redirects, removed_edge_ids);
    let mut source_ids = mapped_source_ids.clone();
    let mut target_ids = mapped_target_ids.clone();
    let endpoints_changed = mapped_source_ids != original_source_ids
        || mapped_target_ids != original_target_ids
        || mapped_source_edge_ids != original_source_edge_ids
        || mapped_target_edge_ids != original_target_edge_ids;

    if payload.get("relation").and_then(Value::as_str) == Some("same_as")
        && (mapped_source_ids != original_source_ids || mapped_target_ids != original_target_ids)
    {
        let mut seen = BTreeSet::new();
        let remain = mapped_source_ids
            .iter()
            .chain(&mapped_target_ids)
            .filter(|id| seen.insert((*id).clone()))
            .cloned()
            .collect::<Vec<_>>();
        if remain.len() >= 2 {
            let anchor = remain
                .iter()
                .find(|id| {
                    canonical_entity_ids
                        .get(*id)
                        .is_some_and(|canonical_id| !canonical_id.is_empty())
                })
                .cloned()
                .unwrap_or_else(|| remain.iter().min().expect("at least two IDs").clone());
            source_ids = vec![anchor.clone()];
            target_ids = remain.into_iter().filter(|id| id != &anchor).collect();
        } else {
            source_ids = remain.iter().take(1).cloned().collect();
            target_ids = remain.into_iter().skip(1).collect();
        }
    }

    if !mentions_changed && !endpoints_changed {
        return Ok(None);
    }

    let has_source = !source_ids.is_empty() || !mapped_source_edge_ids.is_empty();
    let has_target = !target_ids.is_empty() || !mapped_target_edge_ids.is_empty();
    if surviving_mentions.is_empty() || !has_source || !has_target {
        return Ok(Some(EdgeRollbackDecision::Tombstone {
            entity_id: snapshot.entity_id.clone(),
            revision: snapshot.revision,
        }));
    }
    let mut replacement_payload = payload.clone();
    let metadata = replacement_payload
        .get_mut("metadata")
        .and_then(Value::as_object_mut)
        .ok_or(GroundingRollbackError::InvalidEdgePayload)?;
    clean_lifecycle_metadata(metadata);
    let doc_ids = surviving_document_ids(&surviving_mentions);
    replacement_payload.insert("mentions".to_owned(), Value::Array(surviving_mentions));
    replacement_payload.insert(
        "doc_id".to_owned(),
        if doc_ids.len() == 1 {
            Value::String(doc_ids[0].clone())
        } else {
            Value::Null
        },
    );
    replacement_payload.insert(
        "source_ids".to_owned(),
        Value::Array(string_values(source_ids)),
    );
    replacement_payload.insert(
        "target_ids".to_owned(),
        Value::Array(string_values(target_ids)),
    );
    replacement_payload.insert(
        "source_edge_ids".to_owned(),
        Value::Array(string_values(mapped_source_edge_ids)),
    );
    replacement_payload.insert(
        "target_edge_ids".to_owned(),
        Value::Array(string_values(mapped_target_edge_ids)),
    );
    replacement_payload.remove("id");
    replacement_payload.remove("embedding");
    let stable_payload = canonical_json(&Value::Object(replacement_payload.clone()), false);
    let replacement_id = stable_id(
        "rollback.edge",
        &[
            snapshot.entity_id.clone(),
            document_id.to_owned(),
            stable_payload,
        ],
    )
    .to_string();
    replacement_payload.insert("id".to_owned(), Value::String(replacement_id.clone()));

    Ok(Some(EdgeRollbackDecision::Replace {
        entity_id: snapshot.entity_id.clone(),
        revision: snapshot.revision,
        replacement_id,
        replacement_payload: Value::Object(replacement_payload),
    }))
}

fn string_array(
    payload: &Map<String, Value>,
    key: &str,
) -> Result<Vec<String>, GroundingRollbackError> {
    match payload.get(key) {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::Array(values)) => values
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .map(str::to_owned)
                    .ok_or(GroundingRollbackError::InvalidEdgePayload)
            })
            .collect(),
        Some(_) => Err(GroundingRollbackError::InvalidEdgePayload),
    }
}

fn redirected_edge_ids(
    values: &[String],
    redirects: &BTreeMap<String, String>,
    removed: &BTreeSet<String>,
) -> Vec<String> {
    values
        .iter()
        .filter(|id| !removed.contains(*id))
        .map(|id| redirects.get(id).unwrap_or(id).clone())
        .collect()
}

fn string_values(values: Vec<String>) -> Vec<Value> {
    values.into_iter().map(Value::String).collect()
}

fn clean_lifecycle_metadata(metadata: &mut Map<String, Value>) {
    for key in [
        "lifecycle_status",
        "redirect_to_id",
        "deleted_at",
        "delete_reason",
        "deleted_by",
    ] {
        metadata.remove(key);
    }
}

fn surviving_document_ids(mentions: &[Value]) -> Vec<String> {
    mentions
        .iter()
        .filter_map(Value::as_object)
        .filter_map(|mention| mention.get("spans").and_then(Value::as_array))
        .flatten()
        .filter_map(Value::as_object)
        .filter_map(span_document_id)
        .map(str::to_owned)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

/// Apply Python's `RollbackSubsystem.rollback_document` node decision to one
/// canonical event snapshot. Storage, ACL, and transaction ownership stay with
/// the caller; this function only computes deterministic model changes.
pub fn plan_node_rollback(
    snapshot: &crate::canonical_revision::CanonicalGraphEntitySnapshot,
    document_id: &str,
) -> Result<Option<NodeRollbackDecision>, GroundingRollbackError> {
    if snapshot.entity_kind != "node" || snapshot.state != "active" {
        return Err(GroundingRollbackError::UnsupportedSnapshot);
    }
    let Some(payload) = snapshot.payload.as_ref().and_then(Value::as_object) else {
        return Err(GroundingRollbackError::InvalidNodePayload);
    };
    if payload.get("id").and_then(Value::as_str) != Some(snapshot.entity_id.as_str())
        || !payload.get("metadata").is_some_and(Value::is_object)
    {
        return Err(GroundingRollbackError::InvalidNodePayload);
    }
    let mentions = match payload.get("mentions") {
        None | Some(Value::Null) => &[][..],
        Some(Value::Array(mentions)) => mentions.as_slice(),
        Some(_) => return Err(GroundingRollbackError::InvalidMentions),
    };
    let (surviving_mentions, changed) = filter_mentions_for_document(mentions, document_id)?;
    if !changed {
        return Ok(None);
    }
    if surviving_mentions.is_empty() {
        return Ok(Some(NodeRollbackDecision::Tombstone {
            entity_id: snapshot.entity_id.clone(),
            revision: snapshot.revision,
        }));
    }

    let mut replacement_payload = payload.clone();
    let metadata = replacement_payload
        .get_mut("metadata")
        .and_then(Value::as_object_mut)
        .ok_or(GroundingRollbackError::InvalidNodePayload)?;
    for key in [
        "lifecycle_status",
        "redirect_to_id",
        "deleted_at",
        "delete_reason",
        "deleted_by",
    ] {
        metadata.remove(key);
    }
    let doc_ids = surviving_mentions
        .iter()
        .filter_map(Value::as_object)
        .filter_map(|mention| mention.get("spans").and_then(Value::as_array))
        .flatten()
        .filter_map(Value::as_object)
        .filter_map(span_document_id)
        .map(str::to_owned)
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();

    replacement_payload.insert("mentions".to_owned(), Value::Array(surviving_mentions));
    replacement_payload.insert(
        "doc_id".to_owned(),
        if doc_ids.len() == 1 {
            Value::String(doc_ids[0].to_owned())
        } else {
            Value::Null
        },
    );
    replacement_payload.remove("id");
    replacement_payload.remove("embedding");
    let stable_payload = canonical_json(&Value::Object(replacement_payload.clone()), false);
    let replacement_id = stable_id(
        "rollback.node",
        &[
            snapshot.entity_id.clone(),
            document_id.to_owned(),
            stable_payload,
        ],
    )
    .to_string();
    replacement_payload.insert("id".to_owned(), Value::String(replacement_id.clone()));

    Ok(Some(NodeRollbackDecision::Replace {
        entity_id: snapshot.entity_id.clone(),
        revision: snapshot.revision,
        replacement_id,
        replacement_payload: Value::Object(replacement_payload),
    }))
}

/// Remove spans belonging to one source document, preserving all other
/// grounding fields and span order. Empty grounding records are discarded.
///
/// The input is the persisted model payload, not Chroma metadata: source
/// mentions in the canonical graph record remain the rollback authority.
// PYTHON_PARITY_V1: rust=rust/crates/kogwistar-engine/src/document_rollback.rs#filter_mentions_for_document; python=kogwistar/engine_core/subsystems/rollback.py#RollbackSubsystem._filter_mentions_for_document; fixture=contracts/golden/document-rollback-span-filter-v1.json; rust_test=rust/crates/kogwistar-engine/src/document_rollback.rs#span_filter_matches_shared_python_fixture; python_test=tests/primitives/test_document_rollback.py#test_rollback_span_filter_matches_shared_python_fixture
pub fn filter_mentions_for_document(
    mentions: &[Value],
    document_id: &str,
) -> Result<(Vec<Value>, bool), GroundingRollbackError> {
    let mut kept_mentions = Vec::with_capacity(mentions.len());
    let mut changed = false;

    for mention in mentions {
        let mut mention = mention
            .as_object()
            .cloned()
            .ok_or(GroundingRollbackError::InvalidGrounding)?;
        let spans = mention
            .get_mut("spans")
            .ok_or(GroundingRollbackError::InvalidSpans)?
            .as_array_mut()
            .ok_or(GroundingRollbackError::InvalidSpans)?;
        let original_len = spans.len();
        let mut kept_spans = Vec::with_capacity(original_len);
        for span in spans.drain(..) {
            let span_object = span
                .as_object()
                .ok_or(GroundingRollbackError::InvalidSpan)?;
            if span_document_id(span_object) == Some(document_id) {
                changed = true;
            } else {
                kept_spans.push(span);
            }
        }
        if kept_spans.len() != original_len {
            changed = true;
        }
        if kept_spans.is_empty() {
            if original_len > 0 {
                changed = true;
            }
            continue;
        }
        mention.insert("spans".to_owned(), Value::Array(kept_spans));
        kept_mentions.push(Value::Object(mention));
    }

    Ok((kept_mentions, changed))
}

fn span_document_id(span: &Map<String, Value>) -> Option<&str> {
    span.get("doc_id")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .or_else(|| {
            let url = span.get("document_page_url")?.as_str()?;
            let (_, suffix) = url.split_once("document/")?;
            let id = suffix
                .split(|character: char| !character.is_ascii_alphanumeric() && character != '-')
                .next()?;
            (!id.is_empty()).then_some(id)
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn node_snapshot(payload: Value) -> crate::canonical_revision::CanonicalGraphEntitySnapshot {
        crate::canonical_revision::CanonicalGraphEntitySnapshot {
            namespace: "scope".to_owned(),
            entity_kind: "node".to_owned(),
            entity_id: "old-node".to_owned(),
            revision: 7,
            state: "active".to_owned(),
            payload: Some(payload),
        }
    }

    fn edge_snapshot(payload: Value) -> crate::canonical_revision::CanonicalGraphEntitySnapshot {
        crate::canonical_revision::CanonicalGraphEntitySnapshot {
            namespace: "scope".to_owned(),
            entity_kind: "edge".to_owned(),
            entity_id: "old-edge".to_owned(),
            revision: 11,
            state: "active".to_owned(),
            payload: Some(payload),
        }
    }

    fn edge_fixture() -> Value {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../../contracts/golden/python-graph-model-payloads-v1.json"
        ))
        .unwrap();
        fixture["cases"][1]["payload"].clone()
    }

    #[test]
    fn span_filter_matches_shared_python_fixture() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../../contracts/golden/document-rollback-span-filter-v1.json"
        ))
        .expect("shared rollback span fixture parses");
        let mentions = fixture["mentions"].as_array().expect("mentions array");

        let (actual, changed) = filter_mentions_for_document(
            mentions,
            fixture["document_id"].as_str().expect("document ID"),
        )
        .expect("fixture is valid");

        let actual_doc_ids: Vec<Vec<String>> = actual
            .iter()
            .map(|mention| {
                mention["spans"]
                    .as_array()
                    .expect("surviving span array")
                    .iter()
                    .map(|span| {
                        span["doc_id"]
                            .as_str()
                            .expect("span document ID")
                            .to_owned()
                    })
                    .collect()
            })
            .collect();
        assert_eq!(
            serde_json::to_value(actual_doc_ids).expect("serialize span IDs"),
            fixture["expected_span_doc_ids"]
        );
        assert_eq!(changed, fixture["changed"].as_bool().expect("changed flag"));
    }

    #[test]
    fn removes_only_matching_source_spans_and_preserves_other_grounding_data() {
        let mentions = vec![json!({
            "kind": "support",
            "spans": [
                {"doc_id": "doc-a", "start_char": 1},
                {"doc_id": "doc-b", "start_char": 2}
            ]
        })];

        let (kept, changed) = filter_mentions_for_document(&mentions, "doc-a").unwrap();

        assert!(changed);
        assert_eq!(
            kept,
            vec![json!({
                "kind": "support",
                "spans": [{"doc_id": "doc-b", "start_char": 2}]
            })]
        );
        assert_eq!(mentions[0]["spans"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn infers_source_from_document_url_and_drops_empty_groundings() {
        let mentions = vec![
            json!({"spans": [{"document_page_url": "https://host/document/doc-a?page=1"}]}),
            json!({"spans": [{"doc_id": "doc-b"}]}),
        ];

        let (kept, changed) = filter_mentions_for_document(&mentions, "doc-a").unwrap();

        assert!(changed);
        assert_eq!(kept, vec![json!({"spans": [{"doc_id": "doc-b"}]})]);
    }

    #[test]
    fn malformed_grounding_fails_closed() {
        assert_eq!(
            filter_mentions_for_document(&[json!({"spans": null})], "doc-a"),
            Err(GroundingRollbackError::InvalidSpans)
        );
    }

    #[test]
    fn surviving_node_support_produces_python_compatible_replacement() {
        let snapshot = node_snapshot(json!({
            "id": "old-node",
            "embedding": [0.1],
            "name": "Alpha",
            "doc_id": "doc-remove",
            "mentions": [{
                "kind": "support",
                "spans": [
                    {"doc_id": "doc-remove", "start_char": 1},
                    {"doc_id": "doc-keep", "start_char": 9}
                ]
            }],
            "metadata": {
                "lifecycle_status": "active",
                "redirect_to_id": null,
                "deleted_at": "old-time",
                "delete_reason": "old-reason",
                "deleted_by": "old-user",
                "source": "manual"
            }
        }));

        let decision = plan_node_rollback(&snapshot, "doc-remove").unwrap();

        let Some(NodeRollbackDecision::Replace {
            entity_id,
            revision,
            replacement_id,
            replacement_payload,
        }) = decision
        else {
            panic!("expected replacement decision");
        };
        assert_eq!(entity_id, "old-node");
        assert_eq!(revision, 7);
        assert_eq!(replacement_id, "93406411-bdb9-5116-b4de-0426d58409c9");
        assert_eq!(replacement_payload["id"], replacement_id);
        assert_eq!(replacement_payload["doc_id"], "doc-keep");
        assert_eq!(
            replacement_payload["mentions"][0]["spans"],
            json!([{"doc_id": "doc-keep", "start_char": 9}])
        );
        assert_eq!(replacement_payload["metadata"], json!({"source": "manual"}));
        assert!(replacement_payload.get("embedding").is_none());
    }

    #[test]
    fn node_without_surviving_support_is_tombstoned_and_unaffected_node_is_skipped() {
        let affected = node_snapshot(json!({
            "id": "old-node",
            "mentions": [{"spans": [{"doc_id": "doc-remove"}]}],
            "metadata": {"lifecycle_status": "active"}
        }));
        assert_eq!(
            plan_node_rollback(&affected, "doc-remove").unwrap(),
            Some(NodeRollbackDecision::Tombstone {
                entity_id: "old-node".to_owned(),
                revision: 7,
            })
        );

        let unaffected = node_snapshot(json!({
            "id": "old-node",
            "mentions": [{"spans": [{"doc_id": "other-doc"}]}],
            "metadata": {"lifecycle_status": "active"}
        }));
        assert_eq!(plan_node_rollback(&unaffected, "doc-remove").unwrap(), None);
    }

    #[test]
    fn malformed_node_payload_fails_closed_before_rollback_decision() {
        let malformed_mentions = node_snapshot(json!({
            "id": "old-node",
            "mentions": "not-an-array",
            "metadata": {}
        }));
        assert_eq!(
            plan_node_rollback(&malformed_mentions, "doc-remove"),
            Err(GroundingRollbackError::InvalidMentions)
        );

        let mismatched_id = node_snapshot(json!({
            "id": "other-node",
            "mentions": [],
            "metadata": {}
        }));
        assert_eq!(
            plan_node_rollback(&mismatched_id, "doc-remove"),
            Err(GroundingRollbackError::InvalidNodePayload)
        );
    }

    #[test]
    fn edge_rollback_rewrites_both_endpoint_kinds_and_surviving_mentions() {
        let mut payload = edge_fixture();
        payload["id"] = json!("old-edge");
        payload["doc_id"] = json!("doc-remove");
        payload["source_ids"] = json!(["node-old"]);
        payload["target_ids"] = json!(["node-target"]);
        payload["source_edge_ids"] = json!(["edge-old"]);
        payload["target_edge_ids"] = json!(["edge-target"]);
        payload["mentions"][0]["spans"][0]["doc_id"] = json!("doc-remove");
        let mut keep_span = payload["mentions"][0]["spans"][0].clone();
        keep_span["doc_id"] = json!("doc-keep");
        payload["mentions"][0]["spans"]
            .as_array_mut()
            .unwrap()
            .push(keep_span);
        let snapshot = edge_snapshot(payload);
        let node_redirects = BTreeMap::from([("node-old".to_owned(), "node-new".to_owned())]);
        let edge_redirects = BTreeMap::from([("edge-old".to_owned(), "edge-new".to_owned())]);

        let decision = plan_edge_rollback(
            &snapshot,
            "doc-remove",
            &node_redirects,
            &BTreeSet::new(),
            &edge_redirects,
            &BTreeSet::new(),
            &BTreeMap::new(),
        )
        .unwrap();

        let Some(EdgeRollbackDecision::Replace {
            entity_id,
            revision,
            replacement_id,
            replacement_payload,
        }) = decision
        else {
            panic!("expected edge replacement");
        };
        assert_eq!(entity_id, "old-edge");
        assert_eq!(revision, 11);
        assert_eq!(replacement_id, "2955bc14-1a65-5c1c-9677-b915265c26cf");
        assert_eq!(replacement_payload["id"], replacement_id);
        assert_eq!(replacement_payload["doc_id"], "doc-keep");
        assert_eq!(replacement_payload["source_ids"], json!(["node-new"]));
        assert_eq!(replacement_payload["target_ids"], json!(["node-target"]));
        assert_eq!(replacement_payload["source_edge_ids"], json!(["edge-new"]));
        assert_eq!(
            replacement_payload["target_edge_ids"],
            json!(["edge-target"])
        );
        assert_eq!(replacement_payload["metadata"]["workspace_id"], "ws");
        assert!(
            replacement_payload["metadata"]
                .get("lifecycle_status")
                .is_none()
        );
        assert_eq!(
            replacement_payload["mentions"][0]["spans"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
        assert!(replacement_payload.get("embedding").is_none());
    }

    #[test]
    fn same_as_anchor_matches_python_canonical_entity_preference() {
        let mut payload = edge_fixture();
        payload["id"] = json!("old-edge");
        payload["relation"] = json!("same_as");
        payload["source_ids"] = json!(["node-old", "node-b"]);
        payload["target_ids"] = json!(["node-c"]);
        payload["mentions"][0]["spans"][0]["doc_id"] = json!("doc-keep");
        let snapshot = edge_snapshot(payload);
        let canonical_ids = BTreeMap::from([
            ("node-new".to_owned(), "canonical-new".to_owned()),
            ("node-c".to_owned(), "canonical-c".to_owned()),
        ]);

        let decision = plan_edge_rollback(
            &snapshot,
            "doc-remove",
            &BTreeMap::from([("node-old".to_owned(), "node-new".to_owned())]),
            &BTreeSet::new(),
            &BTreeMap::new(),
            &BTreeSet::new(),
            &canonical_ids,
        )
        .unwrap();

        let Some(EdgeRollbackDecision::Replace {
            replacement_payload,
            ..
        }) = decision
        else {
            panic!("expected same_as edge replacement");
        };
        assert_eq!(replacement_payload["source_ids"], json!(["node-new"]));
        assert_eq!(
            replacement_payload["target_ids"],
            json!(["node-b", "node-c"])
        );

        let fallback = plan_edge_rollback(
            &snapshot,
            "doc-remove",
            &BTreeMap::from([("node-old".to_owned(), "node-new".to_owned())]),
            &BTreeSet::new(),
            &BTreeMap::new(),
            &BTreeSet::new(),
            &BTreeMap::new(),
        )
        .unwrap();
        let Some(EdgeRollbackDecision::Replace {
            replacement_payload,
            ..
        }) = fallback
        else {
            panic!("expected same_as fallback replacement");
        };
        assert_eq!(replacement_payload["source_ids"], json!(["node-b"]));
        assert_eq!(
            replacement_payload["target_ids"],
            json!(["node-new", "node-c"])
        );
    }

    #[test]
    fn edge_rollback_tombstones_when_removal_leaves_source_side_empty() {
        let mut payload = edge_fixture();
        payload["id"] = json!("old-edge");
        payload["source_ids"] = json!([]);
        payload["target_ids"] = json!(["node-target"]);
        payload["source_edge_ids"] = json!(["edge-removed"]);
        payload["target_edge_ids"] = json!([]);
        payload["mentions"][0]["spans"][0]["doc_id"] = json!("doc-keep");
        let snapshot = edge_snapshot(payload);

        assert_eq!(
            plan_edge_rollback(
                &snapshot,
                "doc-remove",
                &BTreeMap::new(),
                &BTreeSet::new(),
                &BTreeMap::new(),
                &BTreeSet::from(["edge-removed".to_owned()]),
                &BTreeMap::new(),
            )
            .unwrap(),
            Some(EdgeRollbackDecision::Tombstone {
                entity_id: "old-edge".to_owned(),
                revision: 11,
            })
        );
    }

    #[test]
    fn unrelated_unmentioned_edge_is_not_tombstoned_by_full_graph_scan() {
        let mut payload = edge_fixture();
        payload["id"] = json!("old-edge");
        payload["mentions"] = json!([]);
        let snapshot = edge_snapshot(payload);

        assert_eq!(
            plan_edge_rollback(
                &snapshot,
                "doc-remove",
                &BTreeMap::new(),
                &BTreeSet::new(),
                &BTreeMap::new(),
                &BTreeSet::new(),
                &BTreeMap::new(),
            )
            .unwrap(),
            None
        );
    }
}
