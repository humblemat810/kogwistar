//! Versioned document-ingestion DTOs shared across language boundaries.
//!
//! This is a wire contract only. Storage ownership, ACL checks, embedding
//! policy, and document lifecycle remain responsibilities of the effect
//! owner.

use serde::Deserialize;
use serde_json::{Map, Value};
use thiserror::Error;

pub const DOCUMENT_UPSERT_WIRE_VERSION: u32 = 1;

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DocumentUpsertV1 {
    #[serde(default = "default_document_upsert_version")]
    pub contract_version: u32,
    pub doc_id: String,
    #[serde(default = "default_document_type")]
    pub doc_type: String,
    #[serde(default = "default_insertion_method")]
    pub insertion_method: String,
    pub content: String,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

fn default_document_upsert_version() -> u32 {
    DOCUMENT_UPSERT_WIRE_VERSION
}

fn default_document_type() -> String {
    "text".to_owned()
}

fn default_insertion_method() -> String {
    "document_parser_v1".to_owned()
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum DocumentUpsertValidationError {
    #[error("document upsert contract_version must be 1")]
    ContractVersion,
    #[error("document id must not be empty")]
    DocumentId,
    #[error("document type must be text or ocr")]
    DocumentType,
    #[error("insertion method must not be empty")]
    InsertionMethod,
}

impl DocumentUpsertV1 {
    pub fn validate(&self) -> Result<(), DocumentUpsertValidationError> {
        if self.contract_version != DOCUMENT_UPSERT_WIRE_VERSION {
            return Err(DocumentUpsertValidationError::ContractVersion);
        }
        if self.doc_id.trim().is_empty() {
            return Err(DocumentUpsertValidationError::DocumentId);
        }
        if !matches!(self.doc_type.as_str(), "text" | "ocr") {
            return Err(DocumentUpsertValidationError::DocumentType);
        }
        if self.insertion_method.trim().is_empty() {
            return Err(DocumentUpsertValidationError::InsertionMethod);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_document_fixture_round_trips_with_legacy_defaults() {
        let input = include_str!("../../../../contracts/golden/document-upsert-v1.json");
        let document: DocumentUpsertV1 = serde_json::from_str(input).unwrap();
        document.validate().unwrap();
        assert_eq!(document.contract_version, 1);
        assert_eq!(document.doc_type, "text");
        assert_eq!(document.insertion_method, "api_upsert");
        assert_eq!(document.metadata["source"], "fixture");
    }

    #[test]
    fn invalid_document_upsert_fails_closed() {
        let document: DocumentUpsertV1 =
            serde_json::from_str(r#"{"doc_id":"doc","doc_type":"binary","content":"x"}"#).unwrap();
        assert_eq!(
            document.validate(),
            Err(DocumentUpsertValidationError::DocumentType)
        );
        assert!(
            serde_json::from_str::<DocumentUpsertV1>(
                r#"{"doc_id":"doc","content":"x","unexpected":true}"#
            )
            .is_err()
        );
    }
}
