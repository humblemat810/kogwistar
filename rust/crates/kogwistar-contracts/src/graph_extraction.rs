//! Versioned input contract for document-scoped LLM graph extraction.
//!
//! Entity objects remain JSON maps at this boundary because the Python
//! rehydrator accepts several compatible layouts (`mentions`, `references`,
//! and flattened grounding forms). The effect owner must normalize and
//! validate them before any persistent mutation.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use thiserror::Error;

pub const GRAPH_EXTRACTION_WIRE_VERSION: u32 = 1;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GraphExtractionRequestV1 {
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
}
