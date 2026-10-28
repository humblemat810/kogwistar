//! Schema-qualified PostgreSQL relation and index names.

use super::{
    EDGE_ENDPOINTS_TABLE, EDGE_REFS_TABLE, NODE_DOCS_TABLE, NODE_REFS_TABLE, PostgresStoreResult,
    qualified, quote_identifier,
};

#[derive(Clone, Debug)]
pub(super) struct Tables {
    pub(super) schema: String,
    pub(super) quoted_schema: String,
    pub(super) global_seq: String,
    pub(super) user_seq: String,
    pub(super) index_applied_state: String,
    pub(super) index_applied_state_key_index: String,
    pub(super) namespace_seq: String,
    pub(super) entity_events: String,
    pub(super) replay_cursors: String,
    pub(super) aggregate_index: String,
    pub(super) named_projections: String,
    pub(super) named_projections_namespace_index: String,
    pub(super) workflow_design_snapshots: String,
    pub(super) workflow_design_version_deltas: String,
    pub(super) server_runs: String,
    pub(super) server_run_events: String,
    pub(super) server_runs_status_index: String,
    pub(super) server_run_events_run_seq_index: String,
    pub(super) server_run_events_transition_id_index: String,
    pub(super) index_jobs: String,
    pub(super) index_jobs_status_lease_index: String,
    pub(super) index_jobs_status_next_run_index: String,
    pub(super) index_jobs_entity_index: String,
    pub(super) index_jobs_namespace_index: String,
    pub(super) index_jobs_pending_index: String,
    pub(super) projected_lane_messages: String,
    pub(super) lane_messages_namespace_inbox_seq_index: String,
    pub(super) lane_messages_claim_index: String,
    pub(super) lane_messages_conversation_seq_index: String,
    pub(super) edge_endpoints: String,
    pub(super) node_docs: String,
    pub(super) node_refs: String,
    pub(super) edge_refs: String,
}

impl Tables {
    pub(super) fn new(schema: &str) -> PostgresStoreResult<Self> {
        let quoted_schema = quote_identifier(schema)?;
        Ok(Self {
            schema: schema.to_owned(),
            quoted_schema: quoted_schema.clone(),
            global_seq: qualified(&quoted_schema, "global_seq")?,
            user_seq: qualified(&quoted_schema, "user_seq")?,
            index_applied_state: qualified(&quoted_schema, "index_applied_state")?,
            index_applied_state_key_index: quote_identifier("idx_index_applied_state_key")?,
            namespace_seq: qualified(&quoted_schema, "namespace_seq")?,
            entity_events: qualified(&quoted_schema, "entity_events")?,
            replay_cursors: qualified(&quoted_schema, "replay_cursors")?,
            aggregate_index: quote_identifier("idx_entity_events_aggregate")?,
            named_projections: qualified(&quoted_schema, "named_projections")?,
            named_projections_namespace_index: quote_identifier("idx_named_projections_namespace")?,
            workflow_design_snapshots: qualified(&quoted_schema, "workflow_design_snapshots")?,
            workflow_design_version_deltas: qualified(
                &quoted_schema,
                "workflow_design_version_deltas",
            )?,
            server_runs: qualified(&quoted_schema, "server_runs")?,
            server_run_events: qualified(&quoted_schema, "server_run_events")?,
            server_runs_status_index: quote_identifier("idx_server_runs_status")?,
            server_run_events_run_seq_index: quote_identifier("idx_server_run_events_run_seq")?,
            server_run_events_transition_id_index: quote_identifier(
                "idx_server_run_events_recorded_transition_id",
            )?,
            index_jobs: qualified(&quoted_schema, "index_jobs")?,
            index_jobs_status_lease_index: quote_identifier("idx_index_jobs_status_lease")?,
            index_jobs_status_next_run_index: quote_identifier("idx_index_jobs_status_next_run")?,
            index_jobs_entity_index: quote_identifier("idx_index_jobs_entity")?,
            index_jobs_namespace_index: quote_identifier("idx_index_jobs_namespace")?,
            index_jobs_pending_index: quote_identifier("uq_index_jobs_pending_ns_ck")?,
            projected_lane_messages: qualified(&quoted_schema, "projected_lane_messages")?,
            lane_messages_namespace_inbox_seq_index: quote_identifier(
                "idx_lane_messages_namespace_inbox_seq",
            )?,
            lane_messages_claim_index: quote_identifier("idx_lane_messages_claim")?,
            lane_messages_conversation_seq_index: quote_identifier(
                "idx_lane_messages_conversation_seq",
            )?,
            edge_endpoints: qualified(&quoted_schema, EDGE_ENDPOINTS_TABLE)?,
            node_docs: qualified(&quoted_schema, NODE_DOCS_TABLE)?,
            node_refs: qualified(&quoted_schema, NODE_REFS_TABLE)?,
            edge_refs: qualified(&quoted_schema, EDGE_REFS_TABLE)?,
        })
    }
}
