use super::*;
use kogwistar_contracts::stable_id;
use kogwistar_engine::document_rollback::{NodeRollbackDecision, plan_node_rollback};
use serde_json::{Map, json};

/// Result of one canonical-node rollback page. `scanned_through` advances even
/// when this graph scope has no affected nodes, so callers can safely continue.
#[derive(Clone, Debug, PartialEq)]
pub struct AppliedDocumentNodeRollbackPage {
    pub scanned_through: Option<String>,
    pub has_more: bool,
    pub mutations: Vec<AppliedGraphMutation>,
}

impl SqliteUnitOfWork<'_> {
    /// Apply one bounded document rollback page inside the current SQLite
    /// transaction. Python parity: `RollbackSubsystem.rollback_document` node
    /// phase; edges and document-projection cleanup remain separate phases.
    pub fn apply_document_node_rollback_page(
        &mut self,
        scope: &GraphScope,
        document_id: &str,
        operation_id: &str,
        deleted_at: &str,
        after_entity_id: Option<&str>,
        limit: usize,
    ) -> SqliteStoreResult<AppliedDocumentNodeRollbackPage> {
        if scope.namespace.is_empty()
            || document_id.is_empty()
            || operation_id.is_empty()
            || deleted_at.is_empty()
        {
            return Err(SqliteStoreError::TransactionAborted(
                "document rollback scope, document, operation, and timestamp must be non-empty"
                    .to_owned(),
            ));
        }

        let snapshots =
            self.canonical_graph_entities_page(&scope.namespace, "node", after_entity_id, limit)?;
        let scanned_through = snapshots.last().map(|snapshot| snapshot.entity_id.clone());
        let has_more = snapshots.len() == limit;
        let mut graph_mutations = Vec::new();

        for snapshot in snapshots {
            if snapshot.state != "active" {
                continue;
            }
            let payload = snapshot.payload.as_ref().ok_or_else(|| {
                SqliteStoreError::TransactionAborted(format!(
                    "active canonical node {} has no payload",
                    snapshot.entity_id
                ))
            })?;
            let metadata = payload
                .get("metadata")
                .and_then(Value::as_object)
                .ok_or_else(|| {
                    SqliteStoreError::TransactionAborted(format!(
                        "canonical node {} has invalid metadata",
                        snapshot.entity_id
                    ))
                })?;
            if !metadata_matches_scope(metadata, scope) {
                continue;
            }

            let decision = plan_node_rollback(&snapshot, document_id)
                .map_err(|error| SqliteStoreError::TransactionAborted(error.to_string()))?;
            match decision {
                None => {}
                Some(NodeRollbackDecision::Tombstone {
                    entity_id,
                    revision,
                }) => {
                    let mut tombstone_payload = payload.clone();
                    tombstone_payload["lifecycle_patch"] = json!({
                        "lifecycle_status": "tombstoned",
                        "redirect_to_id": null,
                        "deleted_at": deleted_at,
                        "delete_reason": format!("rollback_document:{document_id}")
                    });
                    graph_mutations.push(graph_mutation(
                        scope,
                        entity_id.clone(),
                        event_id(operation_id, &entity_id, revision, "tombstone"),
                        "TOMBSTONE",
                        tombstone_payload,
                    )?);
                }
                Some(NodeRollbackDecision::Replace {
                    entity_id,
                    revision,
                    replacement_id,
                    replacement_payload,
                }) => {
                    graph_mutations.push(graph_mutation(
                        scope,
                        replacement_id.clone(),
                        event_id(operation_id, &replacement_id, revision, "replacement"),
                        "ADD",
                        replacement_payload,
                    )?);

                    let mut redirect_payload = payload.clone();
                    redirect_payload["lifecycle_patch"] = json!({
                        "lifecycle_status": "tombstoned",
                        "redirect_to_id": replacement_id,
                        "deleted_at": deleted_at,
                        "delete_reason": format!("rollback_document:{document_id}")
                    });
                    graph_mutations.push(graph_mutation(
                        scope,
                        entity_id.clone(),
                        event_id(operation_id, &entity_id, revision, "redirect"),
                        "REPLACE",
                        redirect_payload,
                    )?);
                }
            }
        }

        let mutations = if graph_mutations.is_empty() {
            Vec::new()
        } else {
            self.apply_canonical_graph_mutation_batch(GraphMutationBatch {
                mutations: graph_mutations,
                index_jobs: Vec::new(),
            })?
        };
        Ok(AppliedDocumentNodeRollbackPage {
            scanned_through,
            has_more,
            mutations,
        })
    }
}

fn metadata_matches_scope(metadata: &Map<String, Value>, scope: &GraphScope) -> bool {
    [
        ("workspace_id", scope.workspace_id.as_deref()),
        ("graph_space", scope.graph_space.as_deref()),
    ]
    .into_iter()
    .all(|(key, expected)| match expected {
        Some(expected) => metadata.get(key).and_then(Value::as_str) == Some(expected),
        None => metadata.get(key).is_none_or(Value::is_null),
    })
}

fn event_id(operation_id: &str, entity_id: &str, revision: i64, effect: &str) -> String {
    stable_id(
        "rollback.node.event",
        &[
            operation_id.to_owned(),
            entity_id.to_owned(),
            revision.to_string(),
            effect.to_owned(),
        ],
    )
    .to_string()
}

fn graph_mutation(
    scope: &GraphScope,
    entity_id: String,
    event_id: String,
    op: &str,
    payload: Value,
) -> SqliteStoreResult<GraphMutation> {
    let metadata = payload
        .get("metadata")
        .and_then(Value::as_object)
        .cloned()
        .ok_or_else(|| {
            SqliteStoreError::TransactionAborted(format!(
                "rollback mutation for {entity_id} has invalid metadata"
            ))
        })?;
    Ok(GraphMutation {
        scope: scope.clone(),
        table: "gke_nodes".to_owned(),
        entity_kind: "node".to_owned(),
        event_id,
        op: op.to_owned(),
        record: GraphRecord {
            id: entity_id,
            document: Some(payload.to_string()),
            metadata,
            embedding: None,
        },
        payload,
        embedding_dim: 0,
    })
}
