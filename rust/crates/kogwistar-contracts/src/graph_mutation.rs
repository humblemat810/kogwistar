//! Versioned graph-mutation wire DTOs shared by Rust API callers.
//!
//! These types deliberately contain no storage or authorization behavior.
//! Effect owners validate scope and capabilities before applying them.

use serde::Deserialize;
use serde_json::{Map, Value};

use crate::GraphModelEnvelope;

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GraphMutationBatchV1 {
    pub contract_version: u32,
    pub atomic: bool,
    pub mutations: Vec<GraphMutationV1>,
    #[serde(default)]
    pub index_jobs: Vec<GraphIndexJobV1>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
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

#[derive(Clone, Debug, Deserialize, PartialEq)]
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

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GraphScopeV1 {
    pub namespace: String,
    #[serde(default)]
    pub workspace_id: Option<String>,
    #[serde(default)]
    pub graph_space: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
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

#[cfg(test)]
mod tests {
    use super::GraphMutationBatchV1;
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
}
