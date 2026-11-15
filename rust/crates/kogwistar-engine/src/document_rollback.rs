//! Pure source-grounding edits used by document rollback.
//!
//! Python parity: `kogwistar/engine_core/subsystems/rollback.py`
//! (`RollbackSubsystem._filter_mentions_for_document`).

use kogwistar_contracts::{canonical_json, stable_id};
use serde_json::{Map, Value};
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
    #[error("canonical node payload mentions must be an array or null")]
    InvalidNodeMentions,
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
        Some(_) => return Err(GroundingRollbackError::InvalidNodeMentions),
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
            Err(GroundingRollbackError::InvalidNodeMentions)
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
}
