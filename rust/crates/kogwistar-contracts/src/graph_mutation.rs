//! Versioned graph-mutation wire DTOs shared by Rust API callers.
//!
//! These types deliberately contain no storage or authorization behavior.
//! Effect owners validate scope and capabilities before applying them.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use thiserror::Error;

use crate::GraphModelEnvelope;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GraphMutationBatchV1 {
    pub contract_version: u32,
    pub atomic: bool,
    pub mutations: Vec<GraphMutationV1>,
    #[serde(default)]
    pub index_jobs: Vec<GraphIndexJobV1>,
}

/// A host-resolved default-ACL request for a version-2 graph batch.
///
/// This is intent only: it carries no principal, owner, ACL version, or
/// authorization result. The transaction owner derives those from the
/// authenticated run and current ACL truth at commit time.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GraphDefaultAclIntentV1 {
    pub truth_graph: String,
}

/// ACL-aware extension of the graph mutation contract. Version 1 remains
/// unchanged; consumers that do not understand this contract must reject it.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GraphMutationBatchV2 {
    pub contract_version: u32,
    pub atomic: bool,
    pub mutations: Vec<GraphMutationV1>,
    #[serde(default)]
    pub index_jobs: Vec<GraphIndexJobV1>,
    pub default_acl: GraphDefaultAclIntentV1,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GraphIndexJobV1 {
    pub job_id: String,
    pub namespace: String,
    pub entity_kind: String,
    pub entity_id: String,
    pub index_kind: String,
    pub op: String,
    #[serde(default)]
    pub payload_json: Option<String>,
    #[serde(default = "default_index_job_max_retries")]
    pub max_retries: i64,
}

fn default_index_job_max_retries() -> i64 {
    3
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GraphMutationV1 {
    pub scope: GraphScopeV1,
    pub table: String,
    pub entity_kind: String,
    pub event_id: String,
    pub op: String,
    pub payload: Value,
    pub record: GraphRecordV1,
    pub embedding_dim: usize,
    #[serde(default)]
    pub model_envelope: Option<GraphModelEnvelope>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GraphScopeV1 {
    pub namespace: String,
    #[serde(default)]
    pub workspace_id: Option<String>,
    #[serde(default)]
    pub graph_space: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GraphRecordV1 {
    pub id: String,
    #[serde(default)]
    pub document: Option<String>,
    #[serde(default)]
    pub metadata: Map<String, Value>,
    #[serde(default)]
    pub embedding: Option<Vec<f32>>,
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum GraphMutationValidationError {
    #[error("graph mutation contract_version must be 1")]
    ContractVersion,
    #[error("ACL-aware graph mutation contract_version must be 2")]
    AclBatchContractVersion,
    #[error("default ACL truth_graph must not be empty")]
    AclTruthGraphEmpty,
    #[error("graph mutation batch must be atomic")]
    AtomicRequired,
    #[error("graph mutation batch must contain at least one mutation")]
    MutationsEmpty,
    #[error("graph mutation namespace must not be empty")]
    NamespaceEmpty,
    #[error("graph mutation scopes must match")]
    ScopeMismatch,
    #[error("index job namespace must match graph mutation namespace")]
    IndexJobNamespaceMismatch,
    #[error("graph mutation identifiers must not be empty")]
    IdentifierEmpty,
    #[error("graph mutation table and entity kind do not match")]
    TableEntityKindMismatch,
    #[error("unsupported graph mutation operation")]
    MutationOperation,
    #[error("graph mutation payload must be a JSON object")]
    PayloadObject,
    #[error("graph mutation embedding dimension does not match vector length")]
    EmbeddingDimension,
    #[error("graph mutation embedding values must be finite")]
    EmbeddingNonFinite,
    #[error("index job identifiers must not be empty")]
    IndexJobIdentifierEmpty,
    #[error("index job operation must be UPSERT or DELETE")]
    IndexJobOperation,
    #[error("index job max_retries must fit a non-negative PostgreSQL INTEGER")]
    IndexJobRetries,
    #[error("graph mutation payload id must match the projection record id")]
    PayloadRecordIdMismatch,
}

const MAX_POSTGRES_INTEGER: i64 = i32::MAX as i64;

impl GraphMutationBatchV1 {
    /// Validate semantic invariants shared with Python's Pydantic DTO.
    ///
    /// Deserialization only proves that a payload has the right broad JSON
    /// types. Callers at every language seam must invoke this method before
    /// converting the DTO into an effect or handing it to a store.
    pub fn validate(&self) -> Result<(), GraphMutationValidationError> {
        if self.contract_version != 1 {
            return Err(GraphMutationValidationError::ContractVersion);
        }
        if !self.atomic {
            return Err(GraphMutationValidationError::AtomicRequired);
        }
        let first = self
            .mutations
            .first()
            .ok_or(GraphMutationValidationError::MutationsEmpty)?;
        if first.scope.namespace.is_empty() {
            return Err(GraphMutationValidationError::NamespaceEmpty);
        }
        for mutation in &self.mutations {
            if mutation.scope.namespace.is_empty() {
                return Err(GraphMutationValidationError::NamespaceEmpty);
            }
            if mutation.scope != first.scope {
                return Err(GraphMutationValidationError::ScopeMismatch);
            }
            if mutation.event_id.is_empty() || mutation.record.id.is_empty() {
                return Err(GraphMutationValidationError::IdentifierEmpty);
            }
            let expected_table = match mutation.entity_kind.as_str() {
                "node" => "gke_nodes",
                "edge" => "gke_edges",
                _ => return Err(GraphMutationValidationError::TableEntityKindMismatch),
            };
            if mutation.table != expected_table {
                return Err(GraphMutationValidationError::TableEntityKindMismatch);
            }
            if !matches!(mutation.op.as_str(), "ADD" | "REPLACE" | "TOMBSTONE") {
                return Err(GraphMutationValidationError::MutationOperation);
            }
            if !mutation.payload.is_object() {
                return Err(GraphMutationValidationError::PayloadObject);
            }
            if mutation.payload.get("id") != Some(&Value::String(mutation.record.id.clone())) {
                return Err(GraphMutationValidationError::PayloadRecordIdMismatch);
            }
            if mutation
                .record
                .embedding
                .as_ref()
                .is_some_and(|embedding| embedding.iter().any(|value| !value.is_finite()))
            {
                return Err(GraphMutationValidationError::EmbeddingNonFinite);
            }
            if mutation
                .record
                .embedding
                .as_ref()
                .is_some_and(|embedding| embedding.len() != mutation.embedding_dim)
            {
                return Err(GraphMutationValidationError::EmbeddingDimension);
            }
        }
        for job in &self.index_jobs {
            if job.namespace != first.scope.namespace {
                return Err(GraphMutationValidationError::IndexJobNamespaceMismatch);
            }
            if job.job_id.is_empty()
                || job.namespace.is_empty()
                || job.entity_kind.is_empty()
                || job.entity_id.is_empty()
                || job.index_kind.is_empty()
            {
                return Err(GraphMutationValidationError::IndexJobIdentifierEmpty);
            }
            if !matches!(job.op.as_str(), "UPSERT" | "DELETE") {
                return Err(GraphMutationValidationError::IndexJobOperation);
            }
            if !(0..=MAX_POSTGRES_INTEGER).contains(&job.max_retries) {
                return Err(GraphMutationValidationError::IndexJobRetries);
            }
        }
        Ok(())
    }
}

impl GraphMutationBatchV2 {
    /// Validate the graph batch using V1 invariants plus the mandatory ACL intent.
    pub fn validate(&self) -> Result<(), GraphMutationValidationError> {
        if self.contract_version != 2 {
            return Err(GraphMutationValidationError::AclBatchContractVersion);
        }
        if self.default_acl.truth_graph.trim().is_empty() {
            return Err(GraphMutationValidationError::AclTruthGraphEmpty);
        }
        GraphMutationBatchV1 {
            contract_version: 1,
            atomic: self.atomic,
            mutations: self.mutations.clone(),
            index_jobs: self.index_jobs.clone(),
        }
        .validate()
    }
}

#[cfg(test)]
mod tests {
    use super::{GraphMutationBatchV1, GraphMutationBatchV2, GraphMutationValidationError};
    use serde_json::Value;

    #[test]
    fn graph_mutation_golden_fixture_uses_shared_dto() {
        let batch: GraphMutationBatchV1 = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/graph-mutation-batch-v1.json"
        )))
        .unwrap();
        assert_eq!(batch.contract_version, 1);
        assert!(batch.atomic);
        assert_eq!(batch.mutations.len(), 1);
        assert_eq!(batch.mutations[0].entity_kind, "node");
    }

    #[test]
    fn graph_mutation_index_job_retry_default_is_backward_compatible() {
        let mut value: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/graph-mutation-batch-v1.json"
        )))
        .unwrap();
        value["index_jobs"] = serde_json::json!([{
            "job_id": "job-1",
            "namespace": "tenant-a",
            "entity_kind": "node",
            "entity_id": "node-1",
            "index_kind": "node_embedding",
            "op": "UPSERT"
        }]);
        let batch: GraphMutationBatchV1 = serde_json::from_value(value).unwrap();
        assert_eq!(batch.index_jobs[0].max_retries, 3);
    }

    #[test]
    fn graph_mutation_shared_dto_rejects_unknown_fields() {
        let invalid = r#"{
            "contract_version": 1,
            "atomic": true,
            "mutations": [],
            "unexpected": true
        }"#;
        assert!(serde_json::from_str::<GraphMutationBatchV1>(invalid).is_err());
    }

    #[test]
    fn arbitrary_precision_json_preserves_graph_payload_and_projection_metadata() {
        let mut value: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/graph-mutation-batch-v1.json"
        )))
        .unwrap();
        value["mutations"][0]["payload"]["level_from_root"] = serde_json::json!(1);
        value["mutations"][0]["record"]["metadata"]["level_from_root"] = serde_json::json!(1);
        let safe_json = serde_json::to_string(&value).unwrap();
        let huge = "1606938044258990275541962092341162602522202993782792835301376";
        let raw_json = safe_json.replace(
            "\"level_from_root\":1",
            &format!("\"level_from_root\":{huge}"),
        );
        let batch: GraphMutationBatchV1 = serde_json::from_str(&raw_json).unwrap();
        let mutation = &batch.mutations[0];

        assert!(
            serde_json::to_string(&mutation.payload)
                .unwrap()
                .contains(huge)
        );
        assert!(
            serde_json::to_string(&mutation.record.metadata)
                .unwrap()
                .contains(huge)
        );
    }

    #[test]
    fn graph_mutation_semantic_validation_matches_python_contract() {
        let fixture = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/graph-mutation-batch-v1.json"
        ));
        let valid: GraphMutationBatchV1 = serde_json::from_str(fixture).unwrap();
        assert_eq!(valid.validate(), Ok(()));

        let mut empty = valid.clone();
        empty.mutations.clear();
        assert_eq!(
            empty.validate(),
            Err(GraphMutationValidationError::MutationsEmpty)
        );

        let mut wrong_table = valid.clone();
        wrong_table.mutations[0].table = "gke_edges".to_owned();
        assert_eq!(
            wrong_table.validate(),
            Err(GraphMutationValidationError::TableEntityKindMismatch)
        );

        let mut wrong_embedding = valid.clone();
        wrong_embedding.mutations[0].embedding_dim = 3;
        assert_eq!(
            wrong_embedding.validate(),
            Err(GraphMutationValidationError::EmbeddingDimension)
        );

        let mut physical_delete = valid;
        physical_delete.mutations[0].op = "DELETE".to_owned();
        assert_eq!(
            physical_delete.validate(),
            Err(GraphMutationValidationError::MutationOperation)
        );
    }

    #[test]
    fn acl_aware_graph_batch_v2_is_explicit_and_v1_stays_strict() {
        let fixture = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/graph-mutation-batch-v2.json"
        ));
        let batch: GraphMutationBatchV2 = serde_json::from_str(fixture).unwrap();
        assert_eq!(batch.validate(), Ok(()));
        assert_eq!(batch.default_acl.truth_graph, "conversation");

        let v1_error = serde_json::from_str::<GraphMutationBatchV1>(fixture);
        assert!(
            v1_error.is_err(),
            "V1 must not silently accept V2 semantics"
        );
        let mut invalid = batch;
        invalid.default_acl.truth_graph.clear();
        assert_eq!(
            invalid.validate(),
            Err(GraphMutationValidationError::AclTruthGraphEmpty)
        );
    }

    #[test]
    fn graph_mutation_invalid_cases_match_shared_python_fixture() {
        let baseline = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/graph-mutation-batch-v1.json"
        ));
        let invalid: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/graph-mutation-invalid-v1.json"
        )))
        .unwrap();
        assert_eq!(invalid["schema_version"], 1);

        for case in invalid["cases"].as_array().unwrap() {
            let case_id = case.as_str().unwrap();
            if case_id == "embedding_non_finite" {
                let mut batch: GraphMutationBatchV1 = serde_json::from_str(baseline).unwrap();
                batch.mutations[0].record.embedding = Some(vec![f32::NAN, 0.0]);
                assert_eq!(
                    batch.validate(),
                    Err(GraphMutationValidationError::EmbeddingNonFinite)
                );
                continue;
            }
            let mut value: Value = serde_json::from_str(baseline).unwrap();
            match case_id {
                "payload_record_id_mismatch" => {
                    value["mutations"][0]["record"]["id"] =
                        Value::String("different-node".to_owned());
                }
                "embedding_boolean" => {
                    value["mutations"][0]["record"]["embedding"] = serde_json::json!([true, 0]);
                }
                "embedding_numeric_string" => {
                    value["mutations"][0]["record"]["embedding"] = serde_json::json!(["1", 0]);
                }
                "embedding_dim_boolean" => {
                    value["mutations"][0]["embedding_dim"] = Value::Bool(true);
                }
                "index_retries_negative" | "index_retries_above_postgres_integer" => {
                    let max_retries = if case_id == "index_retries_negative" {
                        -1
                    } else {
                        i64::from(i32::MAX) + 1
                    };
                    value["index_jobs"] = serde_json::json!([{
                        "job_id": "index-node-1",
                        "namespace": "tenant-a",
                        "entity_kind": "node",
                        "entity_id": "node-1",
                        "index_kind": "node_embedding",
                        "op": "UPSERT",
                        "max_retries": max_retries
                    }]);
                }
                "index_retries_boolean" => {
                    value["index_jobs"] = serde_json::json!([{
                        "job_id": "index-node-1",
                        "namespace": "tenant-a",
                        "entity_kind": "node",
                        "entity_id": "node-1",
                        "index_kind": "node_embedding",
                        "op": "UPSERT",
                        "max_retries": true
                    }]);
                }
                _ => panic!("unknown graph mutation invalid case: {case_id}"),
            }

            let batch = match serde_json::from_value::<GraphMutationBatchV1>(value) {
                Ok(batch) => batch,
                Err(_) => continue,
            };
            let error = batch.validate().unwrap_err();
            match case_id {
                "payload_record_id_mismatch" => {
                    assert_eq!(error, GraphMutationValidationError::PayloadRecordIdMismatch)
                }
                "index_retries_negative" | "index_retries_above_postgres_integer" => {
                    assert_eq!(error, GraphMutationValidationError::IndexJobRetries)
                }
                "embedding_boolean"
                | "embedding_numeric_string"
                | "embedding_dim_boolean"
                | "index_retries_boolean" => {
                    panic!("Rust decoder unexpectedly accepted malformed case: {case_id}")
                }
                _ => unreachable!(),
            }
        }
    }
}
