//! Versioned input contract for document-scoped LLM graph extraction.
//!
//! Entity objects remain JSON maps at this boundary because the Python
//! rehydrator accepts both legacy `references` and canonical `mentions` forms.
//! The effect owner must normalize and
//! validate them before any persistent mutation.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use thiserror::Error;

pub const GRAPH_EXTRACTION_WIRE_VERSION: u32 = 1;

fn default_contract_version() -> u32 {
    GRAPH_EXTRACTION_WIRE_VERSION
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GraphExtractionRequestV1 {
    #[serde(default = "default_contract_version")]
    pub contract_version: u32,
    pub doc_id: String,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default = "default_doc_type")]
    pub doc_type: String,
    #[serde(default = "default_insertion_method")]
    pub insertion_method: String,
    #[serde(default)]
    pub nodes: Vec<Map<String, Value>>,
    #[serde(default)]
    pub edges: Vec<Map<String, Value>>,
}

fn default_doc_type() -> String {
    "text".to_owned()
}

fn default_insertion_method() -> String {
    "api_upsert".to_owned()
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum GraphExtractionValidationError {
    #[error("graph extraction contract_version must be 1")]
    ContractVersion,
    #[error("graph extraction document id must not be empty")]
    DocumentId,
    #[error("graph extraction document type must be text or ocr")]
    DocumentType,
    #[error("graph extraction insertion method must not be empty")]
    InsertionMethod,
    #[error("graph extraction references unknown flattened span {0:?}")]
    UnknownSpanReference(String),
    #[error("flattened span references are unsupported by this graph extraction request contract")]
    FlattenedSpansUnsupported,
    #[error("authoritative source text is unavailable for span document {0:?}")]
    SourceTextUnavailable(String),
    #[error("graph extraction span bounds or excerpt do not match source document {0:?}")]
    SpanMismatch(String),
}

impl GraphExtractionRequestV1 {
    pub fn validate(&self) -> Result<(), GraphExtractionValidationError> {
        if self.contract_version != GRAPH_EXTRACTION_WIRE_VERSION {
            return Err(GraphExtractionValidationError::ContractVersion);
        }
        if self.doc_id.trim().is_empty() {
            return Err(GraphExtractionValidationError::DocumentId);
        }
        if !matches!(self.doc_type.as_str(), "text" | "ocr") {
            return Err(GraphExtractionValidationError::DocumentType);
        }
        if self.insertion_method.trim().is_empty() {
            return Err(GraphExtractionValidationError::InsertionMethod);
        }
        Ok(())
    }

    /// Verify every textual extraction span against host-supplied source text.
    ///
    /// Offsets count Unicode scalar values, matching Python string indexing.
    /// The caller must supply authoritative text for every referenced document;
    /// this method never treats model-provided excerpts as source authority.
    pub fn validate_source_spans(
        &self,
        source_texts: &BTreeMap<String, String>,
    ) -> Result<(), GraphExtractionValidationError> {
        let mut spans_to_check = Vec::new();
        for entity in self.nodes.iter().chain(self.edges.iter()) {
            collect_entity_spans(entity, &mut spans_to_check)?;
        }
        for span in spans_to_check {
            validate_span_against_sources(span, &self.doc_id, source_texts)?;
        }
        Ok(())
    }
}

fn collect_entity_spans<'a>(
    entity: &'a Map<String, Value>,
    output: &mut Vec<&'a Value>,
) -> Result<(), GraphExtractionValidationError> {
    if let Some(references) = entity.get("references").and_then(Value::as_array) {
        output.extend(references);
    }
    if let Some(mentions) = entity
        .get("mentions")
        .or_else(|| entity.get("groundings"))
        .and_then(Value::as_array)
    {
        for mention in mentions {
            if let Some(spans) = mention.get("spans").and_then(Value::as_array) {
                output.extend(spans);
            } else if mention.get("span_ids").is_some() {
                return Err(GraphExtractionValidationError::FlattenedSpansUnsupported);
            } else if mention.get("start_char").is_some() {
                output.push(mention);
            }
        }
    }
    Ok(())
}

fn validate_span_against_sources(
    span: &Value,
    default_doc_id: &str,
    source_texts: &BTreeMap<String, String>,
) -> Result<(), GraphExtractionValidationError> {
    let doc_id = span
        .get("doc_id")
        .and_then(Value::as_str)
        .unwrap_or(default_doc_id);
    let source = source_texts
        .get(doc_id)
        .ok_or_else(|| GraphExtractionValidationError::SourceTextUnavailable(doc_id.to_owned()))?;
    let start = span
        .get("start_char")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| GraphExtractionValidationError::SpanMismatch(doc_id.to_owned()))?;
    let end_value = span
        .get("end_char")
        .and_then(Value::as_i64)
        .ok_or_else(|| GraphExtractionValidationError::SpanMismatch(doc_id.to_owned()))?;
    let excerpt = span
        .get("excerpt")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let end = if end_value == -1 {
        start.saturating_add(excerpt.chars().count())
    } else {
        usize::try_from(end_value)
            .map_err(|_| GraphExtractionValidationError::SpanMismatch(doc_id.to_owned()))?
    };
    let characters = source.chars().collect::<Vec<_>>();
    if end <= start
        || end > characters.len()
        || characters[start..end].iter().collect::<String>() != excerpt
    {
        return Err(GraphExtractionValidationError::SpanMismatch(
            doc_id.to_owned(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_graph_extraction_fixture_round_trips() {
        let raw = include_str!("../../../../contracts/golden/graph-extraction-v1.json");
        let request: GraphExtractionRequestV1 = serde_json::from_str(raw).unwrap();
        request.validate().unwrap();
        assert_eq!(request.doc_id, "doc-1");
        assert_eq!(request.nodes.len(), 2);
        assert_eq!(request.edges.len(), 1);
        assert_eq!(request.nodes[0]["local_id"], "nn:first");
        assert_eq!(request.edges[0]["source_ids"][0], "nn:first");
        assert_eq!(
            serde_json::from_str::<GraphExtractionRequestV1>(raw).unwrap(),
            request
        );
    }

    #[test]
    fn graph_extraction_rejects_invalid_header_and_unknown_root_fields() {
        let base = include_str!("../../../../contracts/golden/graph-extraction-v1.json");
        let mut value: Value = serde_json::from_str(base).unwrap();
        value["contract_version"] = Value::from(2);
        let request: GraphExtractionRequestV1 = serde_json::from_value(value).unwrap();
        assert_eq!(
            request.validate(),
            Err(GraphExtractionValidationError::ContractVersion)
        );

        let mut value: Value = serde_json::from_str(base).unwrap();
        value["unexpected"] = Value::Bool(true);
        assert!(serde_json::from_value::<GraphExtractionRequestV1>(value).is_err());
    }

    #[test]
    fn legacy_graph_upsert_defaults_remain_accepted() {
        let request: GraphExtractionRequestV1 =
            serde_json::from_str(r#"{"doc_id":"legacy-doc"}"#).unwrap();
        request.validate().unwrap();
        assert_eq!(request.contract_version, 1);
        assert_eq!(request.doc_type, "text");
        assert_eq!(request.insertion_method, "api_upsert");
        assert!(request.content.is_none());
        assert!(request.nodes.is_empty());
        assert!(request.edges.is_empty());
    }

    #[test]
    fn graph_extraction_checks_exact_unicode_spans_and_fails_closed() {
        let raw = include_str!("../../../../contracts/golden/graph-extraction-v1.json");
        let request: GraphExtractionRequestV1 = serde_json::from_str(raw).unwrap();
        let source = BTreeMap::from([("doc-1".to_owned(), "Alpha connects to Beta.".to_owned())]);
        request.validate_source_spans(&source).unwrap();

        let unicode_request = GraphExtractionRequestV1 {
            doc_id: "unicode".to_owned(),
            content: Some("naïve猫".to_owned()),
            doc_type: "text".to_owned(),
            insertion_method: "api_upsert".to_owned(),
            nodes: vec![
                serde_json::from_value(serde_json::json!({
                    "local_id":"nn:unicode", "label":"猫", "type":"entity", "summary":"猫",
                    "references":[{"start_char":5,"end_char":6,"excerpt":"猫"}]
                }))
                .unwrap(),
            ],
            edges: Vec::new(),
            contract_version: 1,
        };
        unicode_request
            .validate_source_spans(&BTreeMap::from([(
                "unicode".to_owned(),
                "naïve猫".to_owned(),
            )]))
            .unwrap();

        let mut altered = request.clone();
        altered.nodes[0].get_mut("references").unwrap()[0]["excerpt"] =
            Value::String("wrong".to_owned());
        assert!(matches!(
            altered.validate_source_spans(&source),
            Err(GraphExtractionValidationError::SpanMismatch(_))
        ));
        assert!(matches!(
            request.validate_source_spans(&BTreeMap::new()),
            Err(GraphExtractionValidationError::SourceTextUnavailable(_))
        ));

        let mut flattened = request;
        flattened.nodes[0].insert(
            "mentions".to_owned(),
            serde_json::json!([{"span_ids":["span-1"]}]),
        );
        flattened.nodes[0].remove("references");
        assert_eq!(
            flattened.validate_source_spans(&source),
            Err(GraphExtractionValidationError::FlattenedSpansUnsupported)
        );
    }
}
