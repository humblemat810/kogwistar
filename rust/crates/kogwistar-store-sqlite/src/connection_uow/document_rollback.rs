use super::*;
use kogwistar_contracts::stable_id;
use kogwistar_engine::document_rollback::{
    EdgeRollbackDecision, NodeRollbackDecision, plan_edge_rollback, plan_node_rollback,
};
pub use kogwistar_store::AppliedDocumentRollback;
use serde_json::{Map, json};
use std::collections::{BTreeMap, BTreeSet};

/// Result of one canonical-node rollback page. `scanned_through` advances even
/// when this graph scope has no affected nodes, so callers can safely continue.
#[derive(Clone, Debug, PartialEq)]
pub struct AppliedDocumentNodeRollbackPage {
    pub scanned_through: Option<String>,
    pub has_more: bool,
    pub mutations: Vec<AppliedGraphMutation>,
    pub node_redirects: BTreeMap<String, String>,
    pub removed_node_ids: BTreeSet<String>,
    pub canonical_entity_ids: BTreeMap<String, String>,
}

/// Edge maps produced by one rollback page feed later passes until endpoint
/// references converge across the edge graph.
#[derive(Clone, Debug, PartialEq)]
pub struct AppliedDocumentEdgeRollbackPage {
    pub scanned_through: Option<String>,
    pub has_more: bool,
    pub mutations: Vec<AppliedGraphMutation>,
    pub edge_redirects: BTreeMap<String, String>,
    pub removed_edge_ids: BTreeSet<String>,
}

const MAX_DOCUMENT_ROLLBACK_EDGE_PASSES: usize = 4096;

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
        let mut node_redirects = BTreeMap::new();
        let mut removed_node_ids = BTreeSet::new();
        let mut canonical_entity_ids = BTreeMap::new();

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
            if let Some(canonical_id) = payload
                .get("canonical_entity_id")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
            {
                canonical_entity_ids.insert(snapshot.entity_id.clone(), canonical_id.to_owned());
            }

            let decision = plan_node_rollback(&snapshot, document_id)
                .map_err(|error| SqliteStoreError::TransactionAborted(error.to_string()))?;
            match decision {
                None => {}
                Some(NodeRollbackDecision::Tombstone {
                    entity_id,
                    revision,
                }) => {
                    removed_node_ids.insert(entity_id.clone());
                    let mut tombstone_payload = payload.clone();
                    tombstone_payload["lifecycle_patch"] = json!({
                        "lifecycle_status": "tombstoned",
                        "redirect_to_id": null,
                        "deleted_at": deleted_at,
                        "delete_reason": format!("rollback_document:{document_id}")
                    });
                    graph_mutations.push(graph_mutation(
                        scope,
                        "node",
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
                    node_redirects.insert(entity_id.clone(), replacement_id.clone());
                    if let Some(canonical_id) = replacement_payload
                        .get("canonical_entity_id")
                        .and_then(Value::as_str)
                        .filter(|value| !value.is_empty())
                    {
                        canonical_entity_ids
                            .insert(replacement_id.clone(), canonical_id.to_owned());
                    }
                    graph_mutations.push(graph_mutation(
                        scope,
                        "node",
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
                        "node",
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
            node_redirects,
            removed_node_ids,
            canonical_entity_ids,
        })
    }

    /// Roll back graph contributions and remove the matching source projection
    /// in one SQLite transaction. Python parity: `RollbackSubsystem.rollback_document`.
    /// Edge passes repeat because a dependent edge can sort before the edge whose
    /// redirect makes its endpoint repair possible.
    pub fn apply_document_rollback(
        &mut self,
        scope: &GraphScope,
        document_id: &str,
        operation_id: &str,
        deleted_at: &str,
        page_size: usize,
    ) -> SqliteStoreResult<AppliedDocumentRollback> {
        if page_size == 0 {
            return Err(SqliteStoreError::TransactionAborted(
                "document rollback page size must be greater than zero".to_owned(),
            ));
        }

        let mut node_redirects = BTreeMap::new();
        let mut removed_node_ids = BTreeSet::new();
        let mut canonical_entity_ids = BTreeMap::new();
        let mut cursor = None;
        loop {
            let page = self.apply_document_node_rollback_page(
                scope,
                document_id,
                operation_id,
                deleted_at,
                cursor.as_deref(),
                page_size,
            )?;
            node_redirects.extend(page.node_redirects);
            removed_node_ids.extend(page.removed_node_ids);
            canonical_entity_ids.extend(page.canonical_entity_ids);
            if !page.has_more {
                break;
            }
            cursor = page.scanned_through;
        }

        let mut edge_redirects = BTreeMap::new();
        let mut removed_edge_ids = BTreeSet::new();
        let mut edge_passes = 0;
        loop {
            if edge_passes == MAX_DOCUMENT_ROLLBACK_EDGE_PASSES {
                return Err(SqliteStoreError::TransactionAborted(format!(
                    "document rollback did not converge within {MAX_DOCUMENT_ROLLBACK_EDGE_PASSES} edge passes"
                )));
            }
            edge_passes += 1;
            let previous_redirects = edge_redirects.clone();
            let previous_removed = removed_edge_ids.clone();
            let mut cursor = None;
            loop {
                let page = self.apply_document_edge_rollback_page(
                    scope,
                    document_id,
                    operation_id,
                    deleted_at,
                    cursor.as_deref(),
                    page_size,
                    &node_redirects,
                    &removed_node_ids,
                    &edge_redirects,
                    &removed_edge_ids,
                    &canonical_entity_ids,
                )?;
                edge_redirects.extend(page.edge_redirects);
                removed_edge_ids.extend(page.removed_edge_ids);
                if !page.has_more {
                    break;
                }
                cursor = page.scanned_through;
            }
            if edge_redirects == previous_redirects && removed_edge_ids == previous_removed {
                break;
            }
        }

        let deleted_document_projection =
            self.delete_source_document_projection(scope, document_id)?;
        Ok(AppliedDocumentRollback {
            node_redirects,
            removed_node_ids,
            edge_redirects,
            removed_edge_ids,
            deleted_document_projection,
            edge_passes,
        })
    }

    /// Apply one scoped edge rollback page using node outcomes and edge maps
    /// already discovered by the caller. Repeat pages/passes until no new maps
    /// emerge to resolve edge-to-edge dependencies.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_document_edge_rollback_page(
        &mut self,
        scope: &GraphScope,
        document_id: &str,
        operation_id: &str,
        deleted_at: &str,
        after_entity_id: Option<&str>,
        limit: usize,
        node_redirects: &BTreeMap<String, String>,
        removed_node_ids: &BTreeSet<String>,
        edge_redirects: &BTreeMap<String, String>,
        removed_edge_ids: &BTreeSet<String>,
        canonical_entity_ids: &BTreeMap<String, String>,
    ) -> SqliteStoreResult<AppliedDocumentEdgeRollbackPage> {
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
            self.canonical_graph_entities_page(&scope.namespace, "edge", after_entity_id, limit)?;
        let scanned_through = snapshots.last().map(|snapshot| snapshot.entity_id.clone());
        let has_more = snapshots.len() == limit;
        let mut page_edge_redirects = BTreeMap::new();
        let mut page_removed_edge_ids = BTreeSet::new();
        let mut graph_mutations = Vec::new();

        for snapshot in snapshots {
            if snapshot.state != "active" {
                continue;
            }
            let payload = snapshot.payload.as_ref().ok_or_else(|| {
                SqliteStoreError::TransactionAborted(format!(
                    "active canonical edge {} has no payload",
                    snapshot.entity_id
                ))
            })?;
            let metadata = payload
                .get("metadata")
                .and_then(Value::as_object)
                .ok_or_else(|| {
                    SqliteStoreError::TransactionAborted(format!(
                        "canonical edge {} has invalid metadata",
                        snapshot.entity_id
                    ))
                })?;
            if !metadata_matches_scope(metadata, scope) {
                continue;
            }

            let mut current_redirects = edge_redirects.clone();
            current_redirects.extend(page_edge_redirects.clone());
            let current_removed = removed_edge_ids
                .union(&page_removed_edge_ids)
                .cloned()
                .collect::<BTreeSet<_>>();
            let decision = plan_edge_rollback(
                &snapshot,
                document_id,
                node_redirects,
                removed_node_ids,
                &current_redirects,
                &current_removed,
                canonical_entity_ids,
            )
            .map_err(|error| SqliteStoreError::TransactionAborted(error.to_string()))?;

            match decision {
                None => {}
                Some(EdgeRollbackDecision::Tombstone {
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
                        "edge",
                        entity_id.clone(),
                        edge_event_id(operation_id, &entity_id, revision, "tombstone"),
                        "TOMBSTONE",
                        tombstone_payload,
                    )?);
                    page_removed_edge_ids.insert(entity_id);
                }
                Some(EdgeRollbackDecision::Replace {
                    entity_id,
                    revision,
                    replacement_id,
                    replacement_payload,
                }) => {
                    graph_mutations.push(graph_mutation(
                        scope,
                        "edge",
                        replacement_id.clone(),
                        edge_event_id(operation_id, &replacement_id, revision, "replacement"),
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
                        "edge",
                        entity_id.clone(),
                        edge_event_id(operation_id, &entity_id, revision, "redirect"),
                        "REPLACE",
                        redirect_payload,
                    )?);
                    page_edge_redirects.insert(entity_id, replacement_id);
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
        Ok(AppliedDocumentEdgeRollbackPage {
            scanned_through,
            has_more,
            mutations,
            edge_redirects: page_edge_redirects,
            removed_edge_ids: page_removed_edge_ids,
        })
    }
}

pub(super) fn metadata_matches_scope(metadata: &Map<String, Value>, scope: &GraphScope) -> bool {
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

fn edge_event_id(operation_id: &str, entity_id: &str, revision: i64, effect: &str) -> String {
    stable_id(
        "rollback.edge.event",
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
    entity_kind: &str,
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
        table: match entity_kind {
            "node" => "gke_nodes",
            "edge" => "gke_edges",
            _ => {
                return Err(SqliteStoreError::TransactionAborted(format!(
                    "unsupported rollback entity kind {entity_kind:?}"
                )));
            }
        }
        .to_owned(),
        entity_kind: entity_kind.to_owned(),
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
