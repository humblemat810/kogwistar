//! Pure source-grounding edits used by document rollback.
//!
//! Python parity: `kogwistar/engine_core/subsystems/rollback.py`
//! (`RollbackSubsystem._filter_mentions_for_document`).

use serde_json::Value;
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum GroundingRollbackError {
    #[error("grounding entry must be an object")]
    InvalidGrounding,
    #[error("grounding spans must be an array")]
    InvalidSpans,
    #[error("grounding span must be an object")]
    InvalidSpan,
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

fn span_document_id(span: &serde_json::Map<String, Value>) -> Option<&str> {
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
}
