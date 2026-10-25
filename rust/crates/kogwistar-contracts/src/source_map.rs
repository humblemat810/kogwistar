//! Versioned source-map chunk entries shared by document implementations.
//!
//! The Python `Document.source_map` remains an implementation detail.  This
//! contract carries only the deterministic, source-preserving chunk records
//! needed when a document pipeline crosses into Rust.

use serde::Deserialize;
use thiserror::Error;

pub const SOURCE_MAP_WIRE_VERSION: u32 = 1;

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SourceMapChunkV1 {
    pub chunk_id: String,
    pub doc_id: String,
    pub start_char: u64,
    pub end_char: u64,
    pub text: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SourceMapV1 {
    #[serde(default = "default_source_map_version")]
    pub contract_version: u32,
    pub doc_id: String,
    #[serde(default)]
    pub chunks: Vec<SourceMapChunkV1>,
}

fn default_source_map_version() -> u32 {
    SOURCE_MAP_WIRE_VERSION
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum SourceMapValidationError {
    #[error("source map contract_version must be 1")]
    ContractVersion,
    #[error("source map document id must not be empty")]
    DocumentId,
    #[error("source map chunk id must not be empty")]
    ChunkId,
    #[error("source map chunk document id must match the map document")]
    ChunkDocumentId,
    #[error("source map chunk end_char must be greater than start_char")]
    ChunkBounds,
    #[error("source map chunk ids must be unique")]
    DuplicateChunkId,
    #[error("source map chunks must be ordered by end_char")]
    ChunkOrder,
    #[error("source map chunk text does not match the authoritative document")]
    ChunkTextMismatch,
}

impl SourceMapV1 {
    pub fn validate(&self) -> Result<(), SourceMapValidationError> {
        if self.contract_version != SOURCE_MAP_WIRE_VERSION {
            return Err(SourceMapValidationError::ContractVersion);
        }
        if self.doc_id.trim().is_empty() {
            return Err(SourceMapValidationError::DocumentId);
        }

        let mut seen = std::collections::HashSet::with_capacity(self.chunks.len());
        let mut previous_end = 0;
        for chunk in &self.chunks {
            if chunk.chunk_id.trim().is_empty() {
                return Err(SourceMapValidationError::ChunkId);
            }
            if chunk.doc_id != self.doc_id {
                return Err(SourceMapValidationError::ChunkDocumentId);
            }
            if chunk.end_char <= chunk.start_char {
                return Err(SourceMapValidationError::ChunkBounds);
            }
            if !seen.insert(&chunk.chunk_id) {
                return Err(SourceMapValidationError::DuplicateChunkId);
            }
            if chunk.end_char <= previous_end {
                return Err(SourceMapValidationError::ChunkOrder);
            }
            previous_end = chunk.end_char;
        }
        Ok(())
    }

    /// Validate chunk excerpts using Python-compatible character offsets.
    ///
    /// Python's `str` slicing counts Unicode scalar values, while Rust string
    /// byte slicing does not.  Iterating over `char`s keeps this seam stable
    /// for non-ASCII documents without exposing a language-specific encoding
    /// detail in the wire contract.
    pub fn validate_against_source(
        &self,
        source_text: &str,
    ) -> Result<(), SourceMapValidationError> {
        self.validate()?;
        let characters: Vec<char> = source_text.chars().collect();
        for chunk in &self.chunks {
            let start = usize::try_from(chunk.start_char)
                .map_err(|_| SourceMapValidationError::ChunkTextMismatch)?;
            let end = usize::try_from(chunk.end_char)
                .map_err(|_| SourceMapValidationError::ChunkTextMismatch)?;
            if end > characters.len()
                || characters[start..end].iter().collect::<String>() != chunk.text
            {
                return Err(SourceMapValidationError::ChunkTextMismatch);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_source_map_fixture_round_trips() {
        let input = include_str!("../../../../contracts/golden/source-map-v1.json");
        let source_map: SourceMapV1 = serde_json::from_str(input).unwrap();
        source_map.validate().unwrap();
        source_map.validate_against_source("alpha beta").unwrap();
        assert_eq!(source_map.contract_version, SOURCE_MAP_WIRE_VERSION);
        assert_eq!(source_map.chunks[0].text, "alpha");
    }

    #[test]
    fn source_map_excerpt_validation_uses_unicode_character_offsets() {
        let source_map = SourceMapV1 {
            contract_version: SOURCE_MAP_WIRE_VERSION,
            doc_id: "doc-unicode".to_owned(),
            chunks: vec![SourceMapChunkV1 {
                chunk_id: "chunk-0".to_owned(),
                doc_id: "doc-unicode".to_owned(),
                start_char: 0,
                end_char: 3,
                text: "猫🙂a".to_owned(),
            }],
        };
        source_map.validate_against_source("猫🙂abc").unwrap();

        let mut altered = source_map;
        altered.chunks[0].text = "猫🙂b".to_owned();
        assert_eq!(
            altered.validate_against_source("猫🙂abc"),
            Err(SourceMapValidationError::ChunkTextMismatch)
        );
    }

    #[test]
    fn source_map_rejects_cross_document_duplicate_and_out_of_order_chunks() {
        let mut source_map = SourceMapV1 {
            contract_version: SOURCE_MAP_WIRE_VERSION,
            doc_id: "doc".to_owned(),
            chunks: vec![
                SourceMapChunkV1 {
                    chunk_id: "chunk-0".to_owned(),
                    doc_id: "doc".to_owned(),
                    start_char: 0,
                    end_char: 5,
                    text: "alpha".to_owned(),
                },
                SourceMapChunkV1 {
                    chunk_id: "chunk-0".to_owned(),
                    doc_id: "other".to_owned(),
                    start_char: 5,
                    end_char: 10,
                    text: "beta".to_owned(),
                },
            ],
        };
        assert_eq!(
            source_map.validate(),
            Err(SourceMapValidationError::ChunkDocumentId)
        );
        source_map.chunks[1].doc_id = "doc".to_owned();
        assert_eq!(
            source_map.validate(),
            Err(SourceMapValidationError::DuplicateChunkId)
        );
        source_map.chunks[1].chunk_id = "chunk-1".to_owned();
        source_map.chunks[1].end_char = 4;
        assert_eq!(
            source_map.validate(),
            Err(SourceMapValidationError::ChunkBounds)
        );
    }
}
