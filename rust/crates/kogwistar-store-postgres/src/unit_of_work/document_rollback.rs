use super::*;
use kogwistar_engine::canonical_revision::{
    CanonicalGraphEntitySnapshot, reconstruct_canonical_graph_entity,
};
use kogwistar_engine::document_rollback::{
    EdgeRollbackDecision, NodeRollbackDecision, plan_edge_rollback, plan_node_rollback,
};
use serde_json::{Map, json};

const PAGE_SIZE_MAX: usize = 4096;
const MAX_EDGE_PASSES: usize = 4096;

impl PostgresUnitOfWork<'_> {
    /// Roll back source-document contributions and delete its projection in the
    /// caller's transaction. Python parity: `RollbackSubsystem.rollback_document`.
    // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-store-postgres/src/unit_of_work/document_rollback.rs#apply_document_rollback; python=kogwistar/engine_core/subsystems/rollback.py#RollbackSubsystem.rollback_document; fixture=contracts/golden/python-graph-model-payloads-v1.json
    pub async fn apply_document_rollback(
        &mut self,
        scope: &GraphScope,
        document_id: &str,
        operation_id: &str,
        deleted_at: &str,
        page_size: usize,
    ) -> PostgresStoreResult<AppliedDocumentRollback> {
        if scope.namespace.trim().is_empty()
            || document_id.trim().is_empty()
            || operation_id.trim().is_empty()
            || deleted_at.trim().is_empty()
            || page_size == 0
            || page_size > PAGE_SIZE_MAX
        {
            return Err(PostgresStoreError::TransactionAborted(
                "document rollback requires non-empty identifiers and page_size in 1..=4096"
                    .to_owned(),
            ));
        }

        let mut result = AppliedDocumentRollback::default();
        let mut canonical_entity_ids = std::collections::BTreeMap::new();
        let mut cursor: Option<String> = None;
        loop {
            let snapshots = self
                .canonical_graph_entities_page(
                    &scope.namespace,
                    "node",
                    cursor.as_deref(),
                    page_size,
                )
                .await?;
            let has_more = snapshots.len() == page_size;
            cursor = snapshots.last().map(|item| item.entity_id.clone());
            for snapshot in snapshots {
                let Some(payload) = active_scoped_payload(&snapshot, scope)? else {
                    continue;
                };
                if let Some(canonical_id) = payload
                    .get("canonical_entity_id")
                    .and_then(Value::as_str)
                    .filter(|id| !id.is_empty())
                {
                    canonical_entity_ids
                        .insert(snapshot.entity_id.clone(), canonical_id.to_owned());
                }
                match plan_node_rollback(&snapshot, document_id)
                    .map_err(|error| PostgresStoreError::TransactionAborted(error.to_string()))?
                {
                    None => {}
                    Some(NodeRollbackDecision::Tombstone {
                        entity_id,
                        revision,
                    }) => {
                        result.removed_node_ids.insert(entity_id.clone());
                        let mut tombstone = Value::Object(payload.clone());
                        tombstone["lifecycle_patch"] = json!({
                            "lifecycle_status": "tombstoned",
                            "redirect_to_id": null,
                            "deleted_at": deleted_at,
                            "delete_reason": format!("rollback_document:{document_id}")
                        });
                        self.apply_graph_mutation(rollback_mutation(
                            scope,
                            "node",
                            entity_id.clone(),
                            rollback_event_id(
                                operation_id,
                                &entity_id,
                                revision,
                                "tombstone",
                                "node",
                            ),
                            "TOMBSTONE",
                            tombstone,
                        )?)
                        .await?;
                    }
                    Some(NodeRollbackDecision::Replace {
                        entity_id,
                        revision,
                        replacement_id,
                        replacement_payload,
                    }) => {
                        result
                            .node_redirects
                            .insert(entity_id.clone(), replacement_id.clone());
                        if let Some(canonical_id) = replacement_payload
                            .get("canonical_entity_id")
                            .and_then(Value::as_str)
                            .filter(|id| !id.is_empty())
                        {
                            canonical_entity_ids
                                .insert(replacement_id.clone(), canonical_id.to_owned());
                        }
                        self.apply_graph_mutation(rollback_mutation(
                            scope,
                            "node",
                            replacement_id.clone(),
                            rollback_event_id(
                                operation_id,
                                &replacement_id,
                                revision,
                                "replacement",
                                "node",
                            ),
                            "ADD",
                            replacement_payload,
                        )?)
                        .await?;

                        let mut redirect = Value::Object(payload.clone());
                        redirect["lifecycle_patch"] = json!({
                            "lifecycle_status": "tombstoned",
                            "redirect_to_id": replacement_id,
                            "deleted_at": deleted_at,
                            "delete_reason": format!("rollback_document:{document_id}")
                        });
                        self.apply_graph_mutation(rollback_mutation(
                            scope,
                            "node",
                            entity_id.clone(),
                            rollback_event_id(
                                operation_id,
                                &entity_id,
                                revision,
                                "redirect",
                                "node",
                            ),
                            "REPLACE",
                            redirect,
                        )?)
                        .await?;
                    }
                }
            }
            if !has_more {
                break;
            }
        }

        loop {
            if result.edge_passes == MAX_EDGE_PASSES {
                return Err(PostgresStoreError::TransactionAborted(format!(
                    "document rollback did not converge within {MAX_EDGE_PASSES} edge passes"
                )));
            }
            result.edge_passes += 1;
            let prior_redirects = result.edge_redirects.clone();
            let prior_removed = result.removed_edge_ids.clone();
            let mut cursor: Option<String> = None;
            loop {
                let snapshots = self
                    .canonical_graph_entities_page(
                        &scope.namespace,
                        "edge",
                        cursor.as_deref(),
                        page_size,
                    )
                    .await?;
                let has_more = snapshots.len() == page_size;
                cursor = snapshots.last().map(|item| item.entity_id.clone());
                for snapshot in snapshots {
                    let Some(payload) = active_scoped_payload(&snapshot, scope)? else {
                        continue;
                    };
                    let decision = plan_edge_rollback(
                        &snapshot,
                        document_id,
                        &result.node_redirects,
                        &result.removed_node_ids,
                        &result.edge_redirects,
                        &result.removed_edge_ids,
                        &canonical_entity_ids,
                    )
                    .map_err(|error| PostgresStoreError::TransactionAborted(error.to_string()))?;
                    match decision {
                        None => {}
                        Some(EdgeRollbackDecision::Tombstone {
                            entity_id,
                            revision,
                        }) => {
                            let mut tombstone = Value::Object(payload.clone());
                            tombstone["lifecycle_patch"] = json!({
                                "lifecycle_status": "tombstoned",
                                "redirect_to_id": null,
                                "deleted_at": deleted_at,
                                "delete_reason": format!("rollback_document:{document_id}")
                            });
                            self.apply_graph_mutation(rollback_mutation(
                                scope,
                                "edge",
                                entity_id.clone(),
                                rollback_event_id(
                                    operation_id,
                                    &entity_id,
                                    revision,
                                    "tombstone",
                                    "edge",
                                ),
                                "TOMBSTONE",
                                tombstone,
                            )?)
                            .await?;
                            result.removed_edge_ids.insert(entity_id);
                        }
                        Some(EdgeRollbackDecision::Replace {
                            entity_id,
                            revision,
                            replacement_id,
                            replacement_payload,
                        }) => {
                            self.apply_graph_mutation(rollback_mutation(
                                scope,
                                "edge",
                                replacement_id.clone(),
                                rollback_event_id(
                                    operation_id,
                                    &replacement_id,
                                    revision,
                                    "replacement",
                                    "edge",
                                ),
                                "ADD",
                                replacement_payload,
                            )?)
                            .await?;
                            let mut redirect = Value::Object(payload.clone());
                            redirect["lifecycle_patch"] = json!({
                                "lifecycle_status": "tombstoned",
                                "redirect_to_id": replacement_id,
                                "deleted_at": deleted_at,
                                "delete_reason": format!("rollback_document:{document_id}")
                            });
                            self.apply_graph_mutation(rollback_mutation(
                                scope,
                                "edge",
                                entity_id.clone(),
                                rollback_event_id(
                                    operation_id,
                                    &entity_id,
                                    revision,
                                    "redirect",
                                    "edge",
                                ),
                                "REPLACE",
                                redirect,
                            )?)
                            .await?;
                            result.edge_redirects.insert(entity_id, replacement_id);
                        }
                    }
                }
                if !has_more {
                    break;
                }
            }
            if result.edge_redirects == prior_redirects && result.removed_edge_ids == prior_removed
            {
                break;
            }
        }

        result.deleted_document_projection = crate::graph::delete_source_document_projection(
            &self.transaction,
            &self.tables,
            scope,
            document_id,
        )
        .await?;
        Ok(result)
    }

    async fn canonical_graph_entities_page(
        &self,
        namespace: &str,
        entity_kind: &str,
        after_entity_id: Option<&str>,
        limit: usize,
    ) -> PostgresStoreResult<Vec<CanonicalGraphEntitySnapshot>> {
        require_namespace(namespace)?;
        if !matches!(entity_kind, "node" | "edge") || limit == 0 || limit > PAGE_SIZE_MAX {
            return Err(PostgresStoreError::TransactionAborted(
                "canonical graph page requires node/edge kind and limit in 1..=4096".to_owned(),
            ));
        }
        let limit = i64::try_from(limit).map_err(|_| {
            PostgresStoreError::TransactionAborted("canonical graph page limit overflow".to_owned())
        })?;
        let rows = self
            .transaction
            .query(
                &format!(
                    "SELECT entity_id FROM {} WHERE namespace=$1 AND entity_kind=$2 \
                     AND ($3::TEXT IS NULL OR entity_id > $3) \
                     GROUP BY entity_id ORDER BY entity_id LIMIT $4",
                    self.tables.entity_events
                ),
                &[&namespace, &entity_kind, &after_entity_id, &limit],
            )
            .await
            .map_err(backend)?;
        let ids = rows
            .iter()
            .map(|row| row.try_get::<_, String>(0).map_err(backend))
            .collect::<PostgresStoreResult<Vec<_>>>()?;
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        for id in &ids {
            self.lock_entity_identity(namespace, entity_kind, id)
                .await?;
        }
        let rows = self
            .transaction
            .query(
                &format!(
                    "SELECT namespace, seq, event_id, entity_kind, entity_id, op, payload_json, created_at::text \
                     FROM {} WHERE namespace=$1 AND entity_kind=$2 AND entity_id=ANY($3) \
                     ORDER BY entity_id ASC, seq ASC",
                    self.tables.entity_events
                ),
                &[&namespace, &entity_kind, &ids],
            )
            .await
            .map_err(backend)?;
        let mut events_by_id: std::collections::BTreeMap<String, Vec<EntityEvent>> =
            std::collections::BTreeMap::new();
        for row in rows {
            let raw = RawEntityEvent {
                namespace: row.try_get(0).map_err(backend)?,
                seq: row.try_get(1).map_err(backend)?,
                event_id: row.try_get(2).map_err(backend)?,
                entity_kind: row.try_get(3).map_err(backend)?,
                entity_id: row.try_get(4).map_err(backend)?,
                op: row.try_get(5).map_err(backend)?,
                payload_json: row.try_get(6).map_err(backend)?,
                created_at: row.try_get(7).map_err(backend)?,
            };
            let id = raw.entity_id.clone();
            events_by_id
                .entry(id)
                .or_default()
                .push(raw_to_entity_event(raw)?);
        }
        ids.into_iter()
            .map(|id| {
                let events = events_by_id.remove(&id).unwrap_or_default();
                reconstruct_canonical_graph_entity(&events, namespace, entity_kind, &id)
                    .map_err(PostgresStoreError::Store)?
                    .ok_or_else(|| {
                        PostgresStoreError::TransactionAborted(format!(
                            "canonical history missing for {entity_kind}:{id}"
                        ))
                    })
            })
            .collect()
    }
}

fn active_scoped_payload(
    snapshot: &CanonicalGraphEntitySnapshot,
    scope: &GraphScope,
) -> PostgresStoreResult<Option<Map<String, Value>>> {
    if snapshot.state != "active" {
        return Ok(None);
    }
    let payload = snapshot
        .payload
        .as_ref()
        .and_then(Value::as_object)
        .ok_or_else(|| {
            PostgresStoreError::TransactionAborted(format!(
                "active canonical {} {} has no object payload",
                snapshot.entity_kind, snapshot.entity_id
            ))
        })?;
    let metadata = payload
        .get("metadata")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            PostgresStoreError::TransactionAborted(format!(
                "canonical {} {} has invalid metadata",
                snapshot.entity_kind, snapshot.entity_id
            ))
        })?;
    let matches = [
        ("workspace_id", scope.workspace_id.as_deref()),
        ("graph_space", scope.graph_space.as_deref()),
    ]
    .into_iter()
    .all(|(key, expected)| match expected {
        Some(expected) => metadata.get(key).and_then(Value::as_str) == Some(expected),
        None => metadata.get(key).is_none_or(Value::is_null),
    });
    Ok(matches.then(|| payload.clone()))
}

fn rollback_mutation(
    scope: &GraphScope,
    entity_kind: &str,
    entity_id: String,
    event_id: String,
    op: &str,
    payload: Value,
) -> PostgresStoreResult<GraphMutation> {
    let payload_object = payload.as_object().ok_or_else(|| {
        PostgresStoreError::TransactionAborted("rollback payload must be an object".to_owned())
    })?;
    if payload_object.get("id").and_then(Value::as_str) != Some(entity_id.as_str()) {
        return Err(PostgresStoreError::TransactionAborted(format!(
            "rollback payload ID differs from event identity for {entity_kind}:{entity_id}"
        )));
    }
    let mut metadata = payload_object
        .get("metadata")
        .and_then(Value::as_object)
        .cloned()
        .ok_or_else(|| {
            PostgresStoreError::TransactionAborted(format!(
                "rollback payload metadata missing for {entity_kind}:{entity_id}"
            ))
        })?;
    if let Some(patch) = payload_object
        .get("lifecycle_patch")
        .and_then(Value::as_object)
    {
        metadata.extend(patch.clone());
    }
    Ok(GraphMutation {
        scope: scope.clone(),
        table: if entity_kind == "node" {
            "gke_nodes"
        } else {
            "gke_edges"
        }
        .to_owned(),
        entity_kind: entity_kind.to_owned(),
        event_id,
        op: op.to_owned(),
        payload: payload.clone(),
        record: GraphRecord {
            id: entity_id,
            document: Some(payload.to_string()),
            metadata,
            embedding: None,
        },
        embedding_dim: 0,
    })
}

fn rollback_event_id(
    operation_id: &str,
    entity_id: &str,
    revision: i64,
    effect: &str,
    entity_kind: &str,
) -> String {
    kogwistar_contracts::stable_id(
        if entity_kind == "node" {
            "rollback.node.event"
        } else {
            "rollback.edge.event"
        },
        &[
            operation_id.to_owned(),
            entity_id.to_owned(),
            revision.to_string(),
            effect.to_owned(),
        ],
    )
    .to_string()
}
