//! Volatile source-document projection support. Python parity: in-memory graph backend document lookup/upsert/delete.
use super::{InMemoryStore, namespace};
use kogwistar_contracts::stable_id;
use kogwistar_engine::{
    canonical_revision::{CanonicalGraphEntitySnapshot, reconstruct_canonical_graph_entity},
    document_rollback::{
        EdgeRollbackDecision, NodeRollbackDecision, plan_edge_rollback, plan_node_rollback,
    },
};
use kogwistar_store::{
    AppliedDocumentRollback, DocumentAuthorityCapability, DocumentProjectionUpsert,
    DocumentRollbackRequest, DocumentStore, GraphMutation, GraphMutationBatch, GraphMutationStore,
    GraphProjectionRead, GraphRecord, GraphScope, MetadataFilter, StoreError, StoreResult,
};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

const DOCUMENT_TABLE: &str = "gke_documents";
const MAX_EDGE_ROLLBACK_PASSES: usize = 4096;

fn rollback_mutation(
    scope: &GraphScope,
    kind: &str,
    entity_id: String,
    operation_id: &str,
    revision: i64,
    effect: &str,
    op: &str,
    payload: Value,
) -> StoreResult<GraphMutation> {
    let metadata = payload
        .get("metadata")
        .and_then(Value::as_object)
        .cloned()
        .ok_or_else(|| StoreError::Backend {
            backend: "memory-document".to_owned(),
            message: format!("rollback payload for {entity_id} has invalid metadata"),
        })?;
    let table = match kind {
        "node" => "gke_nodes",
        "edge" => "gke_edges",
        _ => {
            return Err(StoreError::Backend {
                backend: "memory-document".to_owned(),
                message: format!("unsupported rollback entity kind {kind:?}"),
            });
        }
    };
    let event_namespace = if kind == "node" {
        "rollback.node.event"
    } else {
        "rollback.edge.event"
    };
    let event_id = stable_id(
        event_namespace,
        &[
            operation_id.to_owned(),
            entity_id.clone(),
            revision.to_string(),
            effect.to_owned(),
        ],
    )
    .to_string();
    let mut projection_metadata = metadata.clone();
    if let Some(patch) = payload.get("lifecycle_patch").and_then(Value::as_object) {
        projection_metadata.extend(patch.clone());
    }
    Ok(GraphMutation {
        scope: scope.clone(),
        table: table.to_owned(),
        entity_kind: kind.to_owned(),
        event_id,
        op: op.to_owned(),
        record: GraphRecord {
            id: entity_id,
            document: Some(payload.to_string()),
            metadata: projection_metadata,
            embedding: None,
        },
        payload,
        embedding_dim: 0,
    })
}

fn scoped_entity_ids(state: &super::State, scope: &GraphScope, table: &str) -> Vec<String> {
    state
        .graph_projections
        .get(scope)
        .into_iter()
        .flat_map(|projection| projection.records.keys())
        .filter(|(record_table, _)| record_table == table)
        .map(|(_, id)| id.clone())
        .collect()
}

fn graph_snapshots(
    state: &super::State,
    scope: &GraphScope,
    kind: &str,
) -> StoreResult<Vec<CanonicalGraphEntitySnapshot>> {
    let Some(namespace) = state.namespaces.get(&scope.namespace) else {
        return Ok(Vec::new());
    };
    let table = if kind == "node" {
        "gke_nodes"
    } else {
        "gke_edges"
    };
    scoped_entity_ids(state, scope, table)
        .into_iter()
        .filter_map(|id| {
            match reconstruct_canonical_graph_entity(&namespace.events, &scope.namespace, kind, &id)
            {
                Ok(snapshot) => snapshot,
                Err(error) => return Some(Err(error)),
            }
            .map(Ok)
        })
        .collect()
}

fn metadata_matches_scope(metadata: &serde_json::Map<String, Value>, scope: &GraphScope) -> bool {
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

// PYTHON_PARITY_V1: rust=rust/crates/kogwistar-store-memory/src/documents.rs#apply_document_rollback; python=kogwistar/engine_core/subsystems/rollback.py#RollbackSubsystem.rollback_document; fixture=contracts/golden/python-graph-model-payloads-v1.json
fn apply_document_rollback(
    state: &mut super::State,
    request: &DocumentRollbackRequest,
) -> StoreResult<AppliedDocumentRollback> {
    if request.scope.namespace.trim().is_empty()
        || request.document_id.trim().is_empty()
        || request.operation_id.trim().is_empty()
        || request.deleted_at.trim().is_empty()
        || request.page_size == 0
    {
        return Err(StoreError::Backend {
            backend: "memory-document".to_owned(),
            message: "rollback scope, document, operation, timestamp, and page size are required"
                .to_owned(),
        });
    }
    namespace(&request.scope.namespace)?;

    // Work on a private state copy; publish once all planners and mutations succeed.
    let mut next = state.clone();
    let mut node_redirects = BTreeMap::new();
    let mut removed_node_ids = BTreeSet::new();
    let mut canonical_entity_ids = BTreeMap::new();
    let node_snapshots = graph_snapshots(&next, &request.scope, "node")?;
    for page in node_snapshots.chunks(request.page_size) {
        let mut node_mutations = Vec::new();
        for snapshot in page {
            if snapshot.state != "active" {
                continue;
            }
            let Some(payload) = snapshot.payload.as_ref() else {
                continue;
            };
            if !payload
                .get("metadata")
                .and_then(Value::as_object)
                .is_some_and(|metadata| metadata_matches_scope(metadata, &request.scope))
            {
                continue;
            }
            if let Some(canonical_id) = payload
                .get("canonical_entity_id")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
            {
                canonical_entity_ids.insert(snapshot.entity_id.clone(), canonical_id.to_owned());
            }
            match plan_node_rollback(snapshot, &request.document_id).map_err(|error| {
                StoreError::Backend {
                    backend: "memory-document".to_owned(),
                    message: error.to_string(),
                }
            })? {
                None => {}
                Some(NodeRollbackDecision::Tombstone {
                    entity_id,
                    revision,
                }) => {
                    removed_node_ids.insert(entity_id.clone());
                    let mut tombstone = payload.clone();
                    tombstone["lifecycle_patch"] = serde_json::json!({
                        "lifecycle_status": "tombstoned",
                        "redirect_to_id": null,
                        "deleted_at": request.deleted_at,
                        "delete_reason": format!("rollback_document:{}", request.document_id)
                    });
                    node_mutations.push(rollback_mutation(
                        &request.scope,
                        "node",
                        entity_id,
                        &request.operation_id,
                        revision,
                        "tombstone",
                        "TOMBSTONE",
                        tombstone,
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
                    node_mutations.push(rollback_mutation(
                        &request.scope,
                        "node",
                        replacement_id.clone(),
                        &request.operation_id,
                        revision,
                        "replacement",
                        "ADD",
                        replacement_payload,
                    )?);
                    let mut redirect = payload.clone();
                    redirect["lifecycle_patch"] = serde_json::json!({
                        "lifecycle_status": "tombstoned",
                        "redirect_to_id": replacement_id,
                        "deleted_at": request.deleted_at,
                        "delete_reason": format!("rollback_document:{}", request.document_id)
                    });
                    node_mutations.push(rollback_mutation(
                        &request.scope,
                        "node",
                        entity_id,
                        &request.operation_id,
                        revision,
                        "redirect",
                        "REPLACE",
                        redirect,
                    )?);
                }
            }
        }
        if !node_mutations.is_empty() {
            super::apply_graph_mutation_batch_to_state(
                &mut next,
                GraphMutationBatch {
                    mutations: node_mutations,
                    index_jobs: Vec::new(),
                },
            )?;
        }
    }

    let mut edge_redirects = BTreeMap::new();
    let mut removed_edge_ids = BTreeSet::new();
    let mut edge_passes = 0;
    loop {
        if edge_passes == MAX_EDGE_ROLLBACK_PASSES {
            return Err(StoreError::Backend {
                backend: "memory-document".to_owned(),
                message: format!("rollback exceeded {MAX_EDGE_ROLLBACK_PASSES} edge passes"),
            });
        }
        edge_passes += 1;
        let old_redirects = edge_redirects.clone();
        let old_removed = removed_edge_ids.clone();
        let edge_snapshots = graph_snapshots(&next, &request.scope, "edge")?;
        for page in edge_snapshots.chunks(request.page_size) {
            let mut mutations = Vec::new();
            for snapshot in page {
                if snapshot.state != "active" {
                    continue;
                }
                let Some(payload) = snapshot.payload.as_ref() else {
                    continue;
                };
                if !payload
                    .get("metadata")
                    .and_then(Value::as_object)
                    .is_some_and(|metadata| metadata_matches_scope(metadata, &request.scope))
                {
                    continue;
                }
                match plan_edge_rollback(
                    snapshot,
                    &request.document_id,
                    &node_redirects,
                    &removed_node_ids,
                    &edge_redirects,
                    &removed_edge_ids,
                    &canonical_entity_ids,
                )
                .map_err(|error| StoreError::Backend {
                    backend: "memory-document".to_owned(),
                    message: error.to_string(),
                })? {
                    None => {}
                    Some(EdgeRollbackDecision::Tombstone {
                        entity_id,
                        revision,
                    }) => {
                        removed_edge_ids.insert(entity_id.clone());
                        let mut tombstone = payload.clone();
                        tombstone["lifecycle_patch"] = serde_json::json!({
                            "lifecycle_status": "tombstoned",
                            "redirect_to_id": null,
                            "deleted_at": request.deleted_at,
                            "delete_reason": format!("rollback_document:{}", request.document_id)
                        });
                        mutations.push(rollback_mutation(
                            &request.scope,
                            "edge",
                            entity_id,
                            &request.operation_id,
                            revision,
                            "tombstone",
                            "TOMBSTONE",
                            tombstone,
                        )?);
                    }
                    Some(EdgeRollbackDecision::Replace {
                        entity_id,
                        revision,
                        replacement_id,
                        replacement_payload,
                    }) => {
                        edge_redirects.insert(entity_id.clone(), replacement_id.clone());
                        mutations.push(rollback_mutation(
                            &request.scope,
                            "edge",
                            replacement_id.clone(),
                            &request.operation_id,
                            revision,
                            "replacement",
                            "ADD",
                            replacement_payload,
                        )?);
                        let mut redirect = payload.clone();
                        redirect["lifecycle_patch"] = serde_json::json!({
                            "lifecycle_status": "tombstoned",
                            "redirect_to_id": replacement_id,
                            "deleted_at": request.deleted_at,
                            "delete_reason": format!("rollback_document:{}", request.document_id)
                        });
                        mutations.push(rollback_mutation(
                            &request.scope,
                            "edge",
                            entity_id,
                            &request.operation_id,
                            revision,
                            "redirect",
                            "REPLACE",
                            redirect,
                        )?);
                    }
                }
            }
            if !mutations.is_empty() {
                super::apply_graph_mutation_batch_to_state(
                    &mut next,
                    GraphMutationBatch {
                        mutations,
                        index_jobs: Vec::new(),
                    },
                )?;
            }
        }
        if edge_redirects == old_redirects && removed_edge_ids == old_removed {
            break;
        }
    }

    let deleted_document_projection = next
        .graph_projections
        .get_mut(&request.scope)
        .map(|projection| {
            let ids = projection
                .records
                .iter()
                .filter(|((table, id), record)| {
                    table == DOCUMENT_TABLE
                        && (id == &request.document_id
                            || record.metadata.get("doc_id").and_then(Value::as_str)
                                == Some(request.document_id.as_str()))
                })
                .map(|((_, id), _)| id.clone())
                .collect::<Vec<_>>();
            for id in &ids {
                projection
                    .records
                    .remove(&(DOCUMENT_TABLE.to_owned(), id.clone()));
            }
            projection
                .record_order
                .retain(|(table, id)| table != DOCUMENT_TABLE || !ids.contains(id));
            !ids.is_empty()
        })
        .unwrap_or(false);

    *state = next;
    Ok(AppliedDocumentRollback {
        node_redirects,
        removed_node_ids,
        edge_redirects,
        removed_edge_ids,
        deleted_document_projection,
        edge_passes,
    })
}

impl DocumentStore for InMemoryStore {
    fn supports_document_projection_lookup(&self) -> bool {
        true
    }

    fn supports_document_projection_delete(&self) -> bool {
        true
    }

    fn supports_document_rollback(&self) -> bool {
        true
    }

    fn document_projection_ids(
        &self,
        scope: GraphScope,
        ids: Vec<String>,
    ) -> impl std::future::Future<Output = StoreResult<Vec<String>>> + Send {
        async move {
            if ids.is_empty() {
                return Ok(Vec::new());
            }
            let rows = self
                .graph_projection_records(GraphProjectionRead {
                    scope,
                    table: DOCUMENT_TABLE.to_owned(),
                    ids: Some(ids),
                    metadata: MetadataFilter::default(),
                    limit: usize::MAX,
                })
                .await?;
            Ok(rows.into_iter().map(|row| row.id).collect())
        }
    }

    fn rollback_document(
        &self,
        request: DocumentRollbackRequest,
    ) -> impl std::future::Future<Output = StoreResult<AppliedDocumentRollback>> + Send {
        async move {
            let mut state = self.state.write().expect("in-memory store lock poisoned");
            apply_document_rollback(&mut state, &request)
        }
    }

    fn upsert_document_projection(
        &self,
        write: DocumentProjectionUpsert,
    ) -> impl std::future::Future<Output = StoreResult<()>> + Send {
        async move {
            if write.scope.namespace.trim().is_empty() || write.record.id.trim().is_empty() {
                return Err(StoreError::Backend {
                    backend: "memory-document".to_owned(),
                    message: "document namespace and ID must be non-empty".to_owned(),
                });
            }
            if write.record.document.is_none() {
                return Err(StoreError::Backend {
                    backend: "memory-document".to_owned(),
                    message: "source document text is required".to_owned(),
                });
            }
            namespace(&write.scope.namespace)?;
            write
                .scope
                .validate_metadata_scope(&write.record.metadata)?;
            if let Some(embedding) = &write.record.embedding {
                if embedding.iter().any(|value| !value.is_finite()) {
                    return Err(StoreError::NonFiniteVector);
                }
                if embedding.len() != write.embedding_dim {
                    return Err(StoreError::VectorDimensionMismatch {
                        expected: write.embedding_dim,
                        actual: embedding.len(),
                    });
                }
            }

            let mut state = self.state.write().expect("in-memory store lock poisoned");
            let projection = state.graph_projections.entry(write.scope).or_default();
            let key = (DOCUMENT_TABLE.to_owned(), write.record.id.clone());
            if !projection.records.contains_key(&key) {
                projection.record_order.push(key.clone());
            }
            projection.records.insert(key, write.record);
            Ok(())
        }
    }

    fn delete_document_projection(
        &self,
        scope: GraphScope,
        document_id: String,
    ) -> impl std::future::Future<Output = StoreResult<bool>> + Send {
        async move {
            namespace(&scope.namespace)?;
            if document_id.trim().is_empty() {
                return Err(StoreError::EmptyRecordId);
            }
            let mut state = self.state.write().expect("in-memory store lock poisoned");
            let Some(projection) = state.graph_projections.get_mut(&scope) else {
                return Ok(false);
            };
            let key = (DOCUMENT_TABLE.to_owned(), document_id.to_owned());
            projection.record_order.retain(|item| item != &key);
            Ok(projection.records.remove(&key).is_some())
        }
    }

    fn delete_document_projections_by_source_id(
        &self,
        scope: GraphScope,
        source_document_id: String,
    ) -> impl std::future::Future<Output = StoreResult<Vec<String>>> + Send {
        async move {
            namespace(&scope.namespace)?;
            if source_document_id.trim().is_empty() {
                return Err(StoreError::EmptyRecordId);
            }
            let mut state = self.state.write().expect("in-memory store lock poisoned");
            let Some(projection) = state.graph_projections.get_mut(&scope) else {
                return Ok(Vec::new());
            };
            let ids = projection
                .records
                .iter()
                .filter(|((table, id), record)| {
                    table == DOCUMENT_TABLE
                        && (id == &source_document_id
                            || record.metadata.get("doc_id").and_then(Value::as_str)
                                == Some(source_document_id.as_str()))
                })
                .map(|((_, id), _)| id.clone())
                .collect::<Vec<_>>();
            for id in &ids {
                let key = (DOCUMENT_TABLE.to_owned(), id.clone());
                projection.records.remove(&key);
                projection.record_order.retain(|item| item != &key);
            }
            Ok(ids)
        }
    }

    fn document_authority_capability(&self) -> DocumentAuthorityCapability {
        DocumentAuthorityCapability::ProjectionUpsert
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kogwistar_store::GraphRecord;
    use std::future::Future;
    use std::pin::pin;
    use std::task::{Context, Poll, Waker};

    fn block_on<T>(future: impl Future<Output = T>) -> T {
        let waker = Waker::noop();
        let mut context = Context::from_waker(waker);
        let mut future = pin!(future);
        loop {
            match future.as_mut().poll(&mut context) {
                Poll::Ready(value) => return value,
                Poll::Pending => std::thread::yield_now(),
            }
        }
    }

    fn scope(namespace: &str, workspace_id: Option<&str>) -> GraphScope {
        GraphScope {
            namespace: namespace.to_owned(),
            workspace_id: workspace_id.map(str::to_owned),
            graph_space: None,
        }
    }

    fn write(scope: GraphScope, id: &str, source_id: &str) -> DocumentProjectionUpsert {
        DocumentProjectionUpsert {
            scope,
            record: GraphRecord {
                id: id.to_owned(),
                document: Some(format!("source text for {id}")),
                metadata: serde_json::Map::from_iter([("doc_id".to_owned(), source_id.into())]),
                embedding: Some(vec![1.0, 0.0]),
            },
            embedding_dim: 2,
        }
    }

    fn graph_mutation(
        scope: &GraphScope,
        kind: &str,
        id: &str,
        mut payload: Value,
    ) -> GraphMutation {
        payload["id"] = Value::String(id.to_owned());
        payload["metadata"]["workspace_id"] = Value::String("tenant-a".to_owned());
        payload["metadata"]["graph_space"] = Value::String("knowledge".to_owned());
        let metadata = payload["metadata"]
            .as_object()
            .expect("fixture metadata")
            .clone();
        GraphMutation {
            scope: scope.clone(),
            table: if kind == "node" {
                "gke_nodes"
            } else {
                "gke_edges"
            }
            .to_owned(),
            entity_kind: kind.to_owned(),
            event_id: format!("seed-{kind}-{id}"),
            op: "ADD".to_owned(),
            record: GraphRecord {
                id: id.to_owned(),
                document: Some(payload.to_string()),
                metadata,
                embedding: None,
            },
            payload,
            embedding_dim: 0,
        }
    }

    fn parity_fixture(kind: &str) -> Value {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../../contracts/golden/python-graph-model-payloads-v1.json"
        ))
        .expect("Python graph-model fixture parses");
        fixture["cases"]
            .as_array()
            .expect("fixture cases")
            .iter()
            .find(|case| case["kind"] == kind)
            .expect("fixture kind")["payload"]
            .clone()
    }

    #[test]
    fn document_projection_is_upserted_looked_up_and_deleted_with_exact_scope() {
        let store = InMemoryStore::new();
        let tenant = scope("docs", Some("tenant-a"));
        let other_tenant = scope("docs", Some("tenant-b"));
        assert_eq!(
            store.document_authority_capability(),
            DocumentAuthorityCapability::ProjectionUpsert
        );
        assert!(DocumentStore::supports_document_projection_delete(&store));
        assert!(DocumentStore::supports_document_rollback(&store));
        block_on(store.upsert_document_projection(write(tenant.clone(), "chunk-1", "source-1")))
            .expect("upsert source projection");
        block_on(store.upsert_document_projection(write(
            other_tenant.clone(),
            "chunk-1",
            "source-1",
        )))
        .expect("same ID in another scope is independent");

        assert_eq!(
            block_on(store.document_projection_ids(tenant.clone(), vec!["chunk-1".into()]))
                .expect("lookup scoped document"),
            ["chunk-1"]
        );
        assert!(
            block_on(store.delete_document_projection(other_tenant, "chunk-1".into()))
                .expect("delete other scope")
        );
        assert_eq!(
            block_on(store.document_projection_ids(tenant.clone(), vec!["chunk-1".into()]))
                .expect("tenant record remains"),
            ["chunk-1"]
        );
        assert!(
            block_on(store.delete_document_projection(tenant.clone(), "chunk-1".into()))
                .expect("delete matching scope")
        );
        assert!(
            !block_on(store.delete_document_projection(tenant, "chunk-1".into()))
                .expect("repeat delete is idempotent")
        );
    }

    #[test]
    fn source_document_delete_matches_record_id_or_doc_id_only_in_scope() {
        let store = InMemoryStore::new();
        let tenant = scope("docs-source", Some("tenant-a"));
        let isolated = scope("docs-source", Some("tenant-b"));
        for (scoped, id, source) in [
            (tenant.clone(), "source-1", "other-source"),
            (tenant.clone(), "chunk-1", "source-1"),
            (tenant.clone(), "chunk-2", "source-1"),
            (isolated, "foreign-chunk", "source-1"),
        ] {
            block_on(store.upsert_document_projection(write(scoped, id, source)))
                .expect("write source projection");
        }

        let deleted = block_on(
            store.delete_document_projections_by_source_id(tenant.clone(), "source-1".into()),
        )
        .expect("delete source document group");
        assert_eq!(deleted, ["chunk-1", "chunk-2", "source-1"]);
        assert_eq!(
            block_on(store.document_projection_ids(
                tenant,
                vec!["source-1".into(), "chunk-1".into(), "chunk-2".into()],
            ))
            .expect("deleted records absent"),
            Vec::<String>::new()
        );
        assert_eq!(
            block_on(store.document_projection_ids(
                scope("docs-source", Some("tenant-b")),
                vec!["foreign-chunk".into()],
            ))
            .expect("other scope is untouched"),
            ["foreign-chunk"]
        );
    }

    #[test]
    fn rollback_rewrites_surviving_node_and_edge_and_deletes_source_projection() {
        let store = InMemoryStore::new();
        let graph_scope = GraphScope {
            namespace: "rollback-memory".to_owned(),
            workspace_id: Some("tenant-a".to_owned()),
            graph_space: Some("knowledge".to_owned()),
        };
        let mut node = parity_fixture("node");
        node["mentions"][0]["spans"][0]["doc_id"] = Value::String("doc-remove".to_owned());
        let mut kept_node_span = node["mentions"][0]["spans"][0].clone();
        kept_node_span["doc_id"] = Value::String("doc-keep".to_owned());
        node["mentions"][0]["spans"]
            .as_array_mut()
            .unwrap()
            .push(kept_node_span);
        let mut edge = parity_fixture("edge");
        edge["id"] = Value::String("edge-old".to_owned());
        edge["source_ids"] = serde_json::json!(["node-old"]);
        edge["mentions"][0]["spans"][0]["doc_id"] = Value::String("doc-remove".to_owned());
        let mut kept_edge_span = edge["mentions"][0]["spans"][0].clone();
        kept_edge_span["doc_id"] = Value::String("doc-keep".to_owned());
        edge["mentions"][0]["spans"]
            .as_array_mut()
            .unwrap()
            .push(kept_edge_span);

        block_on(store.apply_graph_mutation(graph_mutation(
            &graph_scope,
            "node",
            "node-old",
            node,
        )))
        .expect("seed node contribution");
        block_on(store.apply_graph_mutation(graph_mutation(
            &graph_scope,
            "edge",
            "edge-old",
            edge,
        )))
        .expect("seed edge contribution");
        block_on(store.upsert_document_projection(write(
            graph_scope.clone(),
            "doc-remove",
            "doc-remove",
        )))
        .expect("seed source projection");

        let result = block_on(store.rollback_document(DocumentRollbackRequest {
            scope: graph_scope.clone(),
            document_id: "doc-remove".to_owned(),
            operation_id: "rollback-1".to_owned(),
            deleted_at: "2026-11-16T00:00:00Z".to_owned(),
            page_size: 1,
        }))
        .expect("rollback succeeds");
        assert!(result.node_redirects.contains_key("node-old"));
        assert!(result.edge_redirects.contains_key("edge-old"));
        assert!(result.deleted_document_projection);

        let nodes = block_on(store.graph_projection_records(GraphProjectionRead {
            scope: graph_scope.clone(),
            table: "gke_nodes".to_owned(),
            ids: None,
            metadata: MetadataFilter::default(),
            limit: 100,
        }))
        .expect("read nodes");
        let old_node = nodes.iter().find(|record| record.id == "node-old").unwrap();
        assert_eq!(old_node.metadata["lifecycle_status"], "tombstoned");
        let edges = block_on(store.graph_projection_records(GraphProjectionRead {
            scope: graph_scope.clone(),
            table: "gke_edges".to_owned(),
            ids: None,
            metadata: MetadataFilter::default(),
            limit: 100,
        }))
        .expect("read edges");
        let old_edge = edges.iter().find(|record| record.id == "edge-old").unwrap();
        assert_eq!(old_edge.metadata["lifecycle_status"], "tombstoned");
        let replacement_edge_id = &result.edge_redirects["edge-old"];
        let replacement_edge = edges
            .iter()
            .find(|record| &record.id == replacement_edge_id)
            .expect("replacement edge is projected");
        let replacement_edge_payload: Value =
            serde_json::from_str(replacement_edge.document.as_deref().unwrap())
                .expect("replacement edge payload parses");
        assert_eq!(
            replacement_edge_payload["source_ids"][0],
            result.node_redirects["node-old"]
        );
        assert_eq!(
            block_on(store.document_projection_ids(graph_scope.clone(), vec!["doc-remove".into()]))
                .expect("read documents"),
            Vec::<String>::new()
        );

        let events_before_retry = store.state.read().unwrap().namespaces["rollback-memory"]
            .events
            .len();
        let retry = block_on(store.rollback_document(DocumentRollbackRequest {
            scope: graph_scope.clone(),
            document_id: "doc-remove".to_owned(),
            operation_id: "rollback-1".to_owned(),
            deleted_at: "2026-11-16T00:00:00Z".to_owned(),
            page_size: 1,
        }))
        .expect("retry is safe");
        assert!(retry.node_redirects.is_empty());
        assert!(retry.edge_redirects.is_empty());
        assert_eq!(
            store.state.read().unwrap().namespaces["rollback-memory"]
                .events
                .len(),
            events_before_retry
        );
    }

    #[test]
    fn rollback_requires_exact_scope_metadata_and_keeps_foreign_entity_active() {
        let store = InMemoryStore::new();
        let graph_scope = GraphScope {
            namespace: "rollback-scope".to_owned(),
            workspace_id: Some("tenant-a".to_owned()),
            graph_space: Some("knowledge".to_owned()),
        };
        let mut payload = parity_fixture("node");
        payload["doc_id"] = Value::String("doc-remove".to_owned());
        payload["mentions"][0]["spans"][0]["doc_id"] = Value::String("doc-remove".to_owned());
        payload["metadata"]
            .as_object_mut()
            .unwrap()
            .remove("workspace_id");
        let mut foreign = graph_mutation(&graph_scope, "node", "foreign-node", payload);
        foreign.payload["metadata"]
            .as_object_mut()
            .unwrap()
            .remove("workspace_id");
        foreign.record.metadata.remove("workspace_id");
        block_on(store.apply_graph_mutation(foreign)).expect("seed foreign-scope entity");

        let result = block_on(store.rollback_document(DocumentRollbackRequest {
            scope: graph_scope.clone(),
            document_id: "doc-remove".to_owned(),
            operation_id: "rollback-scope-1".to_owned(),
            deleted_at: "2026-11-16T00:00:00Z".to_owned(),
            page_size: 1,
        }))
        .expect("scoped rollback succeeds without touching foreign metadata");
        assert!(result.removed_node_ids.is_empty());
        let nodes = block_on(store.graph_projection_records(GraphProjectionRead {
            scope: graph_scope,
            table: "gke_nodes".to_owned(),
            ids: Some(vec!["foreign-node".to_owned()]),
            metadata: MetadataFilter::default(),
            limit: 10,
        }))
        .expect("read foreign node");
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].metadata["lifecycle_status"], "active");
    }

    #[test]
    fn malformed_entity_aborts_rollback_without_partial_memory_state() {
        let store = InMemoryStore::new();
        let graph_scope = GraphScope {
            namespace: "rollback-atomic".to_owned(),
            workspace_id: Some("tenant-a".to_owned()),
            graph_space: Some("knowledge".to_owned()),
        };
        let mut valid = parity_fixture("node");
        valid["mentions"][0]["spans"][0]["doc_id"] = Value::String("doc-remove".to_owned());
        let mut malformed = valid.clone();
        malformed["id"] = Value::String("z-malformed".to_owned());
        malformed["mentions"] = Value::String("not-an-array".to_owned());
        for (id, payload) in [("a-valid", valid), ("z-malformed", malformed)] {
            block_on(store.apply_graph_mutation(graph_mutation(&graph_scope, "node", id, payload)))
                .expect("seed graph entity");
        }
        block_on(store.upsert_document_projection(write(
            graph_scope.clone(),
            "doc-remove",
            "doc-remove",
        )))
        .expect("seed source projection");
        let event_count_before = store.state.read().unwrap().namespaces["rollback-atomic"]
            .events
            .len();

        let outcome = block_on(store.rollback_document(DocumentRollbackRequest {
            scope: graph_scope.clone(),
            document_id: "doc-remove".to_owned(),
            operation_id: "rollback-invalid".to_owned(),
            deleted_at: "2026-11-16T00:00:00Z".to_owned(),
            page_size: 1,
        }));
        assert!(outcome.is_err());
        assert_eq!(
            store.state.read().unwrap().namespaces["rollback-atomic"]
                .events
                .len(),
            event_count_before
        );
        let source_ids =
            block_on(store.document_projection_ids(graph_scope, vec!["doc-remove".to_owned()]))
                .expect("source projection remains after failed transaction");
        assert_eq!(source_ids, ["doc-remove"]);
    }

    #[test]
    fn rollback_revisits_earlier_sorted_edges_after_upstream_tombstone() {
        let store = InMemoryStore::new();
        let graph_scope = GraphScope {
            namespace: "rollback-edge-cascade".to_owned(),
            workspace_id: Some("tenant-a".to_owned()),
            graph_space: Some("knowledge".to_owned()),
        };
        let mut node = parity_fixture("node");
        node["mentions"][0]["spans"][0]["doc_id"] = Value::String("doc-remove".to_owned());
        let mut upstream = parity_fixture("edge");
        upstream["id"] = Value::String("z-upstream".to_owned());
        upstream["source_ids"] = serde_json::json!(["node-old"]);
        upstream["target_ids"] = serde_json::json!(["external:target"]);
        upstream["mentions"] = serde_json::json!([]);
        let mut downstream = parity_fixture("edge");
        downstream["id"] = Value::String("a-downstream".to_owned());
        downstream["source_ids"] = serde_json::json!([]);
        downstream["target_ids"] = serde_json::json!(["external:target"]);
        downstream["source_edge_ids"] = serde_json::json!(["z-upstream"]);
        downstream["mentions"] = serde_json::json!([]);

        for (kind, id, payload) in [
            ("node", "node-old", node),
            ("edge", "a-downstream", downstream),
            ("edge", "z-upstream", upstream),
        ] {
            block_on(store.apply_graph_mutation(graph_mutation(&graph_scope, kind, id, payload)))
                .expect("seed graph entity");
        }

        let result = block_on(store.rollback_document(DocumentRollbackRequest {
            scope: graph_scope.clone(),
            document_id: "doc-remove".to_owned(),
            operation_id: "rollback-edge-cascade-1".to_owned(),
            deleted_at: "2026-11-16T00:00:00Z".to_owned(),
            page_size: 1,
        }))
        .expect("edge cascade converges");
        assert!(result.removed_edge_ids.contains("z-upstream"));
        assert!(result.removed_edge_ids.contains("a-downstream"));
        assert!(result.edge_passes >= 2);

        let records = block_on(store.graph_projection_records(GraphProjectionRead {
            scope: graph_scope,
            table: "gke_edges".to_owned(),
            ids: Some(vec!["a-downstream".to_owned(), "z-upstream".to_owned()]),
            metadata: MetadataFilter::default(),
            limit: 10,
        }))
        .expect("read cascaded edges");
        assert!(
            records
                .iter()
                .all(|record| { record.metadata["lifecycle_status"] == "tombstoned" })
        );
    }

    #[test]
    fn invalid_document_projection_does_not_create_scope_state() {
        let store = InMemoryStore::new();
        let mut invalid = write(scope("docs-invalid", None), "bad", "bad");
        invalid.record.embedding = Some(vec![f32::NAN]);
        assert!(matches!(
            block_on(store.upsert_document_projection(invalid)),
            Err(StoreError::NonFiniteVector)
        ));
        assert!(
            block_on(
                store.document_projection_ids(scope("docs-invalid", None), vec!["bad".into()],)
            )
            .expect("invalid write leaves backend usable")
            .is_empty()
        );
    }
}
