//! Versioned checkpoint metadata shared by Python and Rust projectors.

use serde::Deserialize;
use thiserror::Error;

pub const PROJECTION_CHECKPOINT_WIRE_VERSION: u32 = 1;

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ProjectionCheckpointV1 {
    #[serde(default = "default_checkpoint_version")]
    pub contract_version: u32,
    pub projection_id: String,
    pub workspace_id: String,
    pub projection_schema_version: u32,
    pub source_namespace: String,
    pub source_from_seq: u64,
    pub source_to_seq: u64,
    pub last_authoritative_seq: u64,
    pub last_materialized_seq: u64,
    pub projected_at_ms: Option<i64>,
    pub last_source_event_ts_ms: Option<i64>,
    pub snapshot_id: String,
    pub raw_event_count: u64,
    pub materialization_status: String,
    pub rebuild_reason: Option<String>,
    #[serde(default)]
    pub processed_event_ids: Vec<String>,
}

fn default_checkpoint_version() -> u32 {
    PROJECTION_CHECKPOINT_WIRE_VERSION
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum ProjectionCheckpointValidationError {
    #[error("projection checkpoint contract_version must be 1")]
    ContractVersion,
    #[error("projection checkpoint projection_id must not be empty")]
    ProjectionId,
    #[error("projection checkpoint workspace_id must not be empty")]
    WorkspaceId,
    #[error("projection checkpoint source_namespace must not be empty")]
    SourceNamespace,
    #[error("projection checkpoint snapshot_id must not be empty")]
    SnapshotId,
    #[error("projection checkpoint materialization_status must not be empty")]
    MaterializationStatus,
    #[error("projection checkpoint event IDs must be unique and non-empty")]
    EventIds,
}

impl ProjectionCheckpointV1 {
    pub fn validate(&self) -> Result<(), ProjectionCheckpointValidationError> {
        if self.contract_version != PROJECTION_CHECKPOINT_WIRE_VERSION {
            return Err(ProjectionCheckpointValidationError::ContractVersion);
        }
        if self.projection_id.trim().is_empty() {
            return Err(ProjectionCheckpointValidationError::ProjectionId);
        }
        if self.workspace_id.trim().is_empty() {
            return Err(ProjectionCheckpointValidationError::WorkspaceId);
        }
        if self.source_namespace.trim().is_empty() {
            return Err(ProjectionCheckpointValidationError::SourceNamespace);
        }
        if self.snapshot_id.trim().is_empty() {
            return Err(ProjectionCheckpointValidationError::SnapshotId);
        }
        if self.materialization_status.trim().is_empty() {
            return Err(ProjectionCheckpointValidationError::MaterializationStatus);
        }
        let mut event_ids =
            std::collections::HashSet::with_capacity(self.processed_event_ids.len());
        if self
            .processed_event_ids
            .iter()
            .any(|event_id| event_id.trim().is_empty() || !event_ids.insert(event_id))
        {
            return Err(ProjectionCheckpointValidationError::EventIds);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_projection_checkpoint_fixture_round_trips() {
        let input = include_str!("../../../../contracts/golden/projection-checkpoint-v1.json");
        let checkpoint: ProjectionCheckpointV1 = serde_json::from_str(input).unwrap();
        checkpoint.validate().unwrap();
        assert_eq!(checkpoint.contract_version, 1);
        assert_eq!(checkpoint.processed_event_ids, vec!["event-1", "event-2"]);
    }

    #[test]
    fn projection_checkpoint_rejects_duplicate_event_ids() {
        let checkpoint = ProjectionCheckpointV1 {
            contract_version: 1,
            projection_id: "projection".to_owned(),
            workspace_id: "workspace".to_owned(),
            projection_schema_version: 1,
            source_namespace: "events".to_owned(),
            source_from_seq: 1,
            source_to_seq: 2,
            last_authoritative_seq: 2,
            last_materialized_seq: 2,
            projected_at_ms: None,
            last_source_event_ts_ms: None,
            snapshot_id: "snapshot".to_owned(),
            raw_event_count: 2,
            materialization_status: "materialized".to_owned(),
            rebuild_reason: None,
            processed_event_ids: vec!["event-1".to_owned(), "event-1".to_owned()],
        };
        assert_eq!(
            checkpoint.validate(),
            Err(ProjectionCheckpointValidationError::EventIds)
        );
    }
}
