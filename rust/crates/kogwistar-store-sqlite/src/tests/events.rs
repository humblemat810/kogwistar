// Domain-focused tests; shared fixtures live in the parent test module.
use super::*;

#[test]
fn restored_raw_event_preserves_identity_and_is_idempotent() {
    let (store, path) = store("restored-raw-event");
    let event = RestoredRawEntityEvent {
        seq: 1,
        event_id: "archive-event".to_owned(),
        entity_kind: "node".to_owned(),
        entity_id: "node-1".to_owned(),
        op: "UPSERT".to_owned(),
        payload_json: "{\"value\":1}".to_owned(),
        created_at: 123,
    };
    assert!(
        store
            .append_restored_raw_entity_event("archive", event.clone())
            .unwrap()
            .inserted
    );
    assert!(
        !store
            .append_restored_raw_entity_event("archive", event.clone())
            .unwrap()
            .inserted
    );
    assert_eq!(
        store.replay_raw_events("archive", 0, 10).unwrap()[0].created_at,
        123
    );
    assert!(
            store
                .append_restored_raw_entity_event(
                    "archive",
                    RestoredRawEntityEvent { seq: 3, ..event },
                )
                .is_err()
        );
    drop(store);
    fs::remove_file(path).unwrap();
}

#[test]
fn latest_entity_events_are_batched_and_scoped_by_namespace_and_kind() {
    let (store, path) = store("latest-entity-events");
    for (namespace, kind, id, event_id, op) in [
        ("scope", "node", "n1", "n1-v1", "ADD"),
        ("scope", "edge", "n1", "e-n1-v2", "REPLACE"),
        ("scope", "node", "n1", "n1-v3", "REPLACE"),
        ("other", "node", "n1", "other-n1", "ADD"),
    ] {
        store
            .append_raw_entity_event(
                namespace,
                NewRawEntityEvent {
                    event_id: event_id.to_owned(),
                    entity_kind: kind.to_owned(),
                    entity_id: id.to_owned(),
                    op: op.to_owned(),
                    payload_json: "{}".to_owned(),
                },
            )
            .unwrap();
    }
    let latest = store
        .latest_entity_events("scope", "node", &["n1".to_owned(), "missing".to_owned()])
        .unwrap();
    assert_eq!(latest.len(), 1);
    assert_eq!(latest["n1"].seq, 3);
    assert_eq!(latest["n1"].event_id, "n1-v3");
    assert!(
        store
            .latest_entity_events("scope", "node", &[])
            .unwrap()
            .is_empty()
    );
    drop(store);
    fs::remove_file(path).unwrap();
}

#[test]
fn latest_entity_events_page_is_keyset_ordered_and_returns_current_revisions() {
    let (store, path) = store("latest-entity-events-page");
    for (namespace, kind, id, event_id, op, revision) in [
        ("scope", "node", "n2", "n2-v1", "ADD", 1),
        ("scope", "node", "n1", "n1-v1", "ADD", 1),
        ("scope", "node", "n1", "n1-v2", "REPLACE", 2),
        ("scope", "edge", "e1", "e1-v1", "ADD", 1),
        ("other", "node", "n0", "other-v1", "ADD", 1),
    ] {
        store
            .append_raw_entity_event(
                namespace,
                NewRawEntityEvent {
                    event_id: event_id.to_owned(),
                    entity_kind: kind.to_owned(),
                    entity_id: id.to_owned(),
                    op: op.to_owned(),
                    payload_json: serde_json::json!({"revision": revision}).to_string(),
                },
            )
            .unwrap();
    }

    store
        .immediate_transaction(|uow| {
            let first = uow.latest_entity_events_page("scope", "node", None, 1)?;
            assert_eq!(first.len(), 1);
            assert_eq!(first[0].entity_id, "n1");
            assert_eq!(first[0].event_id, "n1-v2");
            let second = uow.latest_entity_events_page("scope", "node", Some("n1"), 1)?;
            assert_eq!(second.len(), 1);
            assert_eq!(second[0].entity_id, "n2");
            assert_eq!(second[0].event_id, "n2-v1");
            assert!(
                uow.latest_entity_events_page("scope", "node", Some("n2"), 1)?
                    .is_empty()
            );
            assert!(
                uow.latest_entity_events_page("scope", "node", None, 0)
                    .is_err()
            );
            Ok(())
        })
        .unwrap();

    drop(store);
    fs::remove_file(path).unwrap();
}

#[test]
fn canonical_graph_entity_page_reconstructs_lifecycle_delta_from_event_history() {
    let (store, path) = store("canonical-graph-entity-page");
    store
        .immediate_transaction(|uow| {
            for (seq, event_id, op, payload) in [
                (
                    1,
                    "node-add",
                    "ADD",
                    serde_json::json!({
                        "id": "n1",
                        "label": "entity",
                        "metadata": {"lifecycle_status": "active", "keep": true},
                        "mentions": [{"spans": [{"doc_id": "doc-a"}]}]
                    }),
                ),
                (
                    2,
                    "node-tombstone",
                    "TOMBSTONE",
                    serde_json::json!({
                        "entity_id": "n1",
                        "lifecycle_patch": {
                            "lifecycle_status": "tombstoned",
                            "redirect_to_id": null
                        }
                    }),
                ),
            ] {
                uow.append_restored_raw_entity_event(
                    "scope",
                    RestoredRawEntityEvent {
                        seq,
                        event_id: event_id.to_owned(),
                        entity_kind: "node".to_owned(),
                        entity_id: "n1".to_owned(),
                        op: op.to_owned(),
                        payload_json: payload.to_string(),
                        created_at: seq,
                    },
                )?;
            }
            let snapshots = uow.canonical_graph_entities_page("scope", "node", None, 10)?;
            assert_eq!(snapshots.len(), 1);
            assert_eq!(snapshots[0].revision, 2);
            assert_eq!(snapshots[0].state, "deleted");
            let payload = snapshots[0].payload.as_ref().unwrap();
            assert_eq!(payload["label"], "entity");
            assert_eq!(payload["metadata"]["keep"], true);
            assert_eq!(payload["metadata"]["lifecycle_status"], "tombstoned");
            assert!(payload["mentions"].is_array());
            Ok(())
        })
        .unwrap();

    drop(store);
    fs::remove_file(path).unwrap();
}

#[test]
fn canonical_graph_entity_pages_are_keyset_stable_and_scope_isolated() {
    let (store, path) = store("canonical-graph-entity-pages");
    store
        .immediate_transaction(|uow| {
            for (namespace, seq, entity_kind, entity_id) in [
                ("scope", 1, "node", "n-a"),
                ("scope", 2, "node", "n-b"),
                ("scope", 3, "node", "n-c"),
                ("scope", 4, "edge", "e-a"),
                ("other", 1, "node", "n-b"),
            ] {
                let payload = if entity_kind == "node" {
                    serde_json::json!({
                        "id": entity_id,
                        "label": format!("label-{entity_id}"),
                        "metadata": {"lifecycle_status": "active"}
                    })
                } else {
                    serde_json::json!({
                        "id": entity_id,
                        "relation": "links",
                        "metadata": {"lifecycle_status": "active"}
                    })
                };
                uow.append_restored_raw_entity_event(
                    namespace,
                    RestoredRawEntityEvent {
                        seq,
                        event_id: format!("{namespace}-{entity_id}"),
                        entity_kind: entity_kind.to_owned(),
                        entity_id: entity_id.to_owned(),
                        op: "ADD".to_owned(),
                        payload_json: payload.to_string(),
                        created_at: seq,
                    },
                )?;
            }

            let first = uow.canonical_graph_entities_page("scope", "node", None, 2)?;
            assert_eq!(
                first
                    .iter()
                    .map(|snapshot| snapshot.entity_id.as_str())
                    .collect::<Vec<_>>(),
                ["n-a", "n-b"]
            );
            assert_eq!(first[0].payload.as_ref().unwrap()["label"], "label-n-a");

            let second =
                uow.canonical_graph_entities_page("scope", "node", Some(&first[1].entity_id), 2)?;
            assert_eq!(second.len(), 1);
            assert_eq!(second[0].entity_id, "n-c");
            assert!(
                uow.canonical_graph_entities_page("scope", "edge", None, 10)?
                    .iter()
                    .all(|snapshot| snapshot.entity_id == "e-a")
            );
            assert_eq!(
                uow.canonical_graph_entities_page("other", "node", None, 10)?[0].entity_id,
                "n-b"
            );
            Ok(())
        })
        .unwrap();

    drop(store);
    fs::remove_file(path).unwrap();
}

#[test]
fn document_node_rollback_page_is_atomic_scoped_and_retry_safe() {
    let (store, path) = store("document-node-rollback-page");
    let scope = GraphScope {
        namespace: "scope".to_owned(),
        workspace_id: Some("workspace-a".to_owned()),
        graph_space: Some("knowledge".to_owned()),
    };
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../../contracts/golden/python-graph-model-payloads-v1.json"
    ))
    .unwrap();
    store
        .immediate_transaction(|uow| {
            for (seq, entity_id, workspace_id, mentions) in [
                (
                    1,
                    "a-replace",
                    "workspace-a",
                    serde_json::json!([{
                        "multimodal_spans": [],
                        "spans": [
                            {"collection_page_url":"https://example.test/source","context_after":"","context_before":"","doc_id":"doc-remove","document_page_url":"https://example.test/source#span","end_char":13,"excerpt":"Node evidence","insertion_method":"contract-test","page_number":1,"source_cluster_id":null,"start_char":0,"verification":{"is_verified":false,"method":"heuristic","notes":"no explicit verification from LLM","score":null}},
                            {"collection_page_url":"https://example.test/source","context_after":"","context_before":"","doc_id":"doc-keep","document_page_url":"https://example.test/source#span","end_char":13,"excerpt":"Node evidence","insertion_method":"contract-test","page_number":1,"source_cluster_id":null,"start_char":0,"verification":{"is_verified":false,"method":"heuristic","notes":"no explicit verification from LLM","score":null}}
                        ]
                    }]),
                ),
                (
                    2,
                    "b-tombstone",
                    "workspace-a",
                    serde_json::json!([{
                        "multimodal_spans": [],
                        "spans": [{"collection_page_url":"https://example.test/source","context_after":"","context_before":"","doc_id":"doc-remove","document_page_url":"https://example.test/source#span","end_char":13,"excerpt":"Node evidence","insertion_method":"contract-test","page_number":1,"source_cluster_id":null,"start_char":0,"verification":{"is_verified":false,"method":"heuristic","notes":"no explicit verification from LLM","score":null}}]
                    }]),
                ),
                (
                    3,
                    "c-other-workspace",
                    "workspace-b",
                    serde_json::json!([{
                        "multimodal_spans": [],
                        "spans": [{"collection_page_url":"https://example.test/source","context_after":"","context_before":"","doc_id":"doc-remove","document_page_url":"https://example.test/source#span","end_char":13,"excerpt":"Node evidence","insertion_method":"contract-test","page_number":1,"source_cluster_id":null,"start_char":0,"verification":{"is_verified":false,"method":"heuristic","notes":"no explicit verification from LLM","score":null}}]
                    }]),
                ),
            ] {
                let mut payload = fixture["cases"][0]["payload"].clone();
                payload["id"] = serde_json::json!(entity_id);
                payload["label"] = serde_json::json!(entity_id);
                payload["doc_id"] = serde_json::json!("doc-remove");
                payload["mentions"] = mentions;
                payload["metadata"]["workspace_id"] = serde_json::json!(workspace_id);
                payload["metadata"]["graph_space"] = serde_json::json!("knowledge");
                payload["metadata"]["lifecycle_status"] = serde_json::json!("active");
                uow.append_restored_raw_entity_event(
                    "scope",
                    RestoredRawEntityEvent {
                        seq,
                        event_id: format!("seed-{entity_id}"),
                        entity_kind: "node".to_owned(),
                        entity_id: entity_id.to_owned(),
                        op: "ADD".to_owned(),
                        payload_json: payload.to_string(),
                        created_at: seq,
                    },
                )?;
            }

            let first = uow.apply_document_node_rollback_page(
                &scope,
                "doc-remove",
                "rollback-op-1",
                "2026-11-15T00:00:00Z",
                None,
                10,
            )?;
            assert_eq!(first.scanned_through.as_deref(), Some("c-other-workspace"));
            assert!(!first.has_more);
            assert_eq!(first.mutations.len(), 3);
            assert!(first.mutations.iter().all(|mutation| mutation.inserted));

            let retry = uow.apply_document_node_rollback_page(
                &scope,
                "doc-remove",
                "rollback-op-1",
                "2026-11-15T00:00:00Z",
                None,
                10,
            )?;
            assert!(retry.mutations.is_empty());

            let nodes = uow.canonical_graph_entities_page("scope", "node", None, 10)?;
            let by_id = nodes
                .iter()
                .map(|snapshot| (snapshot.entity_id.as_str(), snapshot))
                .collect::<std::collections::BTreeMap<_, _>>();
            assert_eq!(by_id["a-replace"].state, "deleted");
            assert_eq!(
                by_id["a-replace"].payload.as_ref().unwrap()["metadata"]["redirect_to_id"],
                first.mutations[0].event.payload["id"]
            );
            assert_eq!(by_id["b-tombstone"].state, "deleted");
            assert_eq!(by_id["c-other-workspace"].state, "active");
            assert_eq!(
                nodes
                    .iter()
                    .filter(|snapshot| {
                        snapshot.state == "active"
                            && snapshot
                                .payload
                                .as_ref()
                                .and_then(|payload| payload["mentions"].as_array())
                                .is_some_and(|mentions| {
                                    mentions.iter().any(|mention| {
                                        mention["spans"].as_array().is_some_and(|spans| {
                                            spans.iter().any(|span| span["doc_id"] == "doc-keep")
                                        })
                                    })
                                })
                    })
                    .count(),
                1
            );
            Ok(())
        })
        .unwrap();

    drop(store);
    fs::remove_file(path).unwrap();
}

#[test]
fn document_edge_rollback_page_is_atomic_scoped_and_retry_safe() {
    let (store, path) = store("document-edge-rollback-page");
    let scope = GraphScope {
        namespace: "scope".to_owned(),
        workspace_id: Some("workspace-a".to_owned()),
        graph_space: Some("knowledge".to_owned()),
    };
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../../contracts/golden/python-graph-model-payloads-v1.json"
    ))
    .unwrap();
    store
        .immediate_transaction(|uow| {
            for (seq, edge_id, workspace_id, remove_source_endpoint, remove_document_span) in [
                (1, "a-replace", "workspace-a", false, true),
                (2, "b-tombstone", "workspace-a", true, false),
                (3, "c-other-workspace", "workspace-b", true, false),
            ] {
                let mut payload = fixture["cases"][1]["payload"].clone();
                payload["id"] = serde_json::json!(edge_id);
                payload["doc_id"] = serde_json::json!("doc-remove");
                payload["source_ids"] = if remove_source_endpoint {
                    serde_json::json!([])
                } else {
                    serde_json::json!(["node-old"])
                };
                payload["target_ids"] = serde_json::json!(["node-target"]);
                payload["source_edge_ids"] = if remove_source_endpoint {
                    serde_json::json!(["edge-removed"])
                } else {
                    serde_json::json!([])
                };
                payload["target_edge_ids"] = serde_json::json!([]);
                if remove_document_span {
                    let mut keep_span = payload["mentions"][0]["spans"][0].clone();
                    payload["mentions"][0]["spans"][0]["doc_id"] = serde_json::json!("doc-remove");
                    keep_span["doc_id"] = serde_json::json!("doc-keep");
                    payload["mentions"][0]["spans"]
                        .as_array_mut()
                        .unwrap()
                        .push(keep_span);
                } else {
                    payload["mentions"][0]["spans"][0]["doc_id"] = serde_json::json!("doc-keep");
                }
                payload["metadata"]["workspace_id"] = serde_json::json!(workspace_id);
                payload["metadata"]["graph_space"] = serde_json::json!("knowledge");
                payload["metadata"]["lifecycle_status"] = serde_json::json!("active");
                uow.append_restored_raw_entity_event(
                    "scope",
                    RestoredRawEntityEvent {
                        seq,
                        event_id: format!("seed-{edge_id}"),
                        entity_kind: "edge".to_owned(),
                        entity_id: edge_id.to_owned(),
                        op: "ADD".to_owned(),
                        payload_json: payload.to_string(),
                        created_at: seq,
                    },
                )?;
            }

            let node_redirects =
                std::collections::BTreeMap::from([("node-old".to_owned(), "node-new".to_owned())]);
            let removed_edge_ids = std::collections::BTreeSet::from(["edge-removed".to_owned()]);
            let first = uow.apply_document_edge_rollback_page(
                &scope,
                "doc-remove",
                "rollback-edge-op-1",
                "2026-11-15T00:00:00Z",
                None,
                10,
                &node_redirects,
                &std::collections::BTreeSet::new(),
                &std::collections::BTreeMap::new(),
                &removed_edge_ids,
                &std::collections::BTreeMap::new(),
            )?;
            assert_eq!(first.scanned_through.as_deref(), Some("c-other-workspace"));
            assert!(!first.has_more);
            assert_eq!(first.mutations.len(), 3);
            assert_eq!(
                first.edge_redirects.get("a-replace").map(String::as_str),
                first.mutations[0].event.payload["id"].as_str()
            );
            assert!(first.removed_edge_ids.contains("b-tombstone"));

            let retry = uow.apply_document_edge_rollback_page(
                &scope,
                "doc-remove",
                "rollback-edge-op-1",
                "2026-11-15T00:00:00Z",
                None,
                10,
                &node_redirects,
                &std::collections::BTreeSet::new(),
                &first.edge_redirects,
                &first.removed_edge_ids,
                &std::collections::BTreeMap::new(),
            )?;
            assert!(retry.mutations.is_empty());

            let edges = uow.canonical_graph_entities_page("scope", "edge", None, 10)?;
            let by_id = edges
                .iter()
                .map(|snapshot| (snapshot.entity_id.as_str(), snapshot))
                .collect::<std::collections::BTreeMap<_, _>>();
            assert_eq!(by_id["a-replace"].state, "deleted");
            assert_eq!(by_id["b-tombstone"].state, "deleted");
            assert_eq!(by_id["c-other-workspace"].state, "active");
            let replacement = by_id[first.edge_redirects["a-replace"].as_str()];
            assert_eq!(replacement.state, "active");
            assert_eq!(
                replacement.payload.as_ref().unwrap()["source_ids"],
                serde_json::json!(["node-new"])
            );
            assert_eq!(
                replacement.payload.as_ref().unwrap()["mentions"][0]["spans"]
                    .as_array()
                    .unwrap()
                    .len(),
                1
            );
            Ok(())
        })
        .unwrap();

    drop(store);
    fs::remove_file(path).unwrap();
}

#[test]
fn document_rollback_reaches_edge_dependency_fixed_point_before_projection_delete() {
    let (store, path) = store("document-rollback-fixed-point");
    let scope = GraphScope {
        namespace: "scope".to_owned(),
        workspace_id: Some("workspace-a".to_owned()),
        graph_space: Some("knowledge".to_owned()),
    };
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../../contracts/golden/python-graph-model-payloads-v1.json"
    ))
    .unwrap();
    store
        .immediate_transaction(|uow| {
            uow.upsert_source_document_projection(DocumentProjectionUpsert {
                scope: scope.clone(),
                record: GraphRecord {
                    id: "doc-remove".to_owned(),
                    document: Some("source document".to_owned()),
                    metadata: serde_json::json!({"doc_type": "text"})
                        .as_object()
                        .unwrap()
                        .clone(),
                    embedding: None,
                },
                embedding_dim: 0,
            })?;

            for (seq, edge_id) in [(1, "a-dependent"), (2, "z-support"), (3, "unrelated-edge")] {
                let mut payload = fixture["cases"][1]["payload"].clone();
                payload["id"] = serde_json::json!(edge_id);
                payload["metadata"]["workspace_id"] = serde_json::json!("workspace-a");
                payload["metadata"]["graph_space"] = serde_json::json!("knowledge");
                payload["metadata"]["lifecycle_status"] = serde_json::json!("active");
                payload["source_ids"] = if edge_id == "z-support" {
                    serde_json::json!(["node-source"])
                } else {
                    serde_json::json!([])
                };
                payload["target_ids"] = serde_json::json!(["node-target"]);
                payload["source_edge_ids"] = if edge_id == "a-dependent" {
                    serde_json::json!(["z-support"])
                } else {
                    serde_json::json!([])
                };
                payload["target_edge_ids"] = serde_json::json!([]);
                match edge_id {
                    "a-dependent" => {
                        payload["mentions"][0]["spans"][0]["doc_id"] =
                            serde_json::json!("doc-keep");
                    }
                    "z-support" => {
                        payload["doc_id"] = serde_json::json!("doc-remove");
                        let mut keep_span = payload["mentions"][0]["spans"][0].clone();
                        payload["mentions"][0]["spans"][0]["doc_id"] =
                            serde_json::json!("doc-remove");
                        keep_span["doc_id"] = serde_json::json!("doc-keep");
                        payload["mentions"][0]["spans"]
                            .as_array_mut()
                            .unwrap()
                            .push(keep_span);
                    }
                    _ => payload["mentions"] = serde_json::json!([]),
                }
                uow.append_restored_raw_entity_event(
                    "scope",
                    RestoredRawEntityEvent {
                        seq,
                        event_id: format!("seed-{edge_id}"),
                        entity_kind: "edge".to_owned(),
                        entity_id: edge_id.to_owned(),
                        op: "ADD".to_owned(),
                        payload_json: payload.to_string(),
                        created_at: seq,
                    },
                )?;
            }

            let result = uow.apply_document_rollback(
                &scope,
                "doc-remove",
                "rollback-fixed-point-1",
                "2026-11-15T00:00:00Z",
                1,
            )?;
            assert!(result.edge_passes >= 2);
            let dependent_id = result.edge_redirects.get("a-dependent").unwrap();
            let support_id = result.edge_redirects.get("z-support").unwrap();
            assert!(result.deleted_document_projection);
            assert!(
                uow.source_document_projection(&scope, "doc-remove")?
                    .is_none()
            );

            let snapshots = uow.canonical_graph_entities_page("scope", "edge", None, 20)?;
            let by_id = snapshots
                .iter()
                .map(|snapshot| (snapshot.entity_id.as_str(), snapshot))
                .collect::<BTreeMap<_, _>>();
            assert_eq!(by_id["unrelated-edge"].state, "active");
            assert_eq!(by_id[dependent_id.as_str()].state, "active");
            assert_eq!(by_id[support_id.as_str()].state, "active");
            assert_eq!(
                by_id[dependent_id.as_str()].payload.as_ref().unwrap()["source_edge_ids"],
                serde_json::json!([support_id])
            );
            Ok(())
        })
        .unwrap();

    drop(store);
    fs::remove_file(path).unwrap();
}

#[test]
fn sqlite_store_document_rollback_wrapper_uses_one_transaction() {
    let (store, path) = store("document-rollback-store-wrapper");
    let scope = GraphScope {
        namespace: "scope".to_owned(),
        workspace_id: Some("workspace-a".to_owned()),
        graph_space: Some("knowledge".to_owned()),
    };
    store
        .immediate_transaction(|uow| {
            uow.upsert_source_document_projection(DocumentProjectionUpsert {
                scope: scope.clone(),
                record: GraphRecord {
                    id: "doc-remove".to_owned(),
                    document: Some("source document".to_owned()),
                    metadata: Map::new(),
                    embedding: None,
                },
                embedding_dim: 0,
            })?;
            uow.append_restored_raw_entity_event(
                "scope",
                RestoredRawEntityEvent {
                    seq: 1,
                    event_id: "seed-node-remove".to_owned(),
                    entity_kind: "node".to_owned(),
                    entity_id: "node-remove".to_owned(),
                    op: "ADD".to_owned(),
                    payload_json: serde_json::json!({
                        "id": "node-remove",
                        "mentions": [{"spans": [{"doc_id": "doc-remove"}]}],
                        "metadata": {
                            "workspace_id": "workspace-a",
                            "graph_space": "knowledge",
                            "lifecycle_status": "active"
                        }
                    })
                    .to_string(),
                    created_at: 1,
                },
            )?;
            Ok(())
        })
        .unwrap();

    assert!(DocumentStore::supports_document_rollback(&store));
    let result = block_on(DocumentStore::rollback_document(
        &store,
        DocumentRollbackRequest {
            scope: scope.clone(),
            document_id: "doc-remove".to_owned(),
            operation_id: "rollback-store-wrapper".to_owned(),
            deleted_at: "2026-11-15T00:00:00Z".to_owned(),
            page_size: 1,
        },
    ))
    .unwrap();
    assert!(result.removed_node_ids.contains("node-remove"));
    assert!(result.deleted_document_projection);
    store
        .immediate_transaction(|uow| {
            let snapshots = uow.canonical_graph_entities_page("scope", "node", None, 10)?;
            assert_eq!(snapshots[0].state, "deleted");
            assert!(
                uow.source_document_projection(&scope, "doc-remove")?
                    .is_none()
            );
            Ok(())
        })
        .unwrap();

    drop(store);
    fs::remove_file(path).unwrap();
}

#[test]
fn latest_entity_events_are_visible_inside_the_canonical_write_transaction() {
    let (store, path) = store("latest-events-uow");
    store
        .immediate_transaction(|uow| {
            uow.append_raw_entity_event(
                "wisdom",
                NewRawEntityEvent {
                    event_id: "wisdom-node-v1".to_owned(),
                    entity_kind: "node".to_owned(),
                    entity_id: "node-1".to_owned(),
                    op: "ADD".to_owned(),
                    payload_json: r#"{"revision":1}"#.to_owned(),
                },
            )?;
            uow.append_raw_entity_event(
                "wisdom",
                NewRawEntityEvent {
                    event_id: "wisdom-node-v2".to_owned(),
                    entity_kind: "node".to_owned(),
                    entity_id: "node-1".to_owned(),
                    op: "REPLACE".to_owned(),
                    payload_json: r#"{"revision":2}"#.to_owned(),
                },
            )?;
            let latest = uow.latest_entity_events(
                "wisdom",
                "node",
                &["node-1".to_owned(), "missing".to_owned()],
            )?;
            assert_eq!(latest.len(), 1);
            assert_eq!(latest["node-1"].event_id, "wisdom-node-v2");
            assert_eq!(latest["node-1"].payload_json, r#"{"revision":2}"#);
            Ok(())
        })
        .unwrap();
    drop(store);
    fs::remove_file(path).unwrap();
}

#[test]
fn latest_entity_events_by_metadata_uses_current_canonical_revisions() {
    let (store, path) = store("latest-events-metadata");
    store
        .immediate_transaction(|uow| {
            for (event_id, entity_id, op, conversation_id) in [
                ("start-v1", "start", "ADD", "conversation-a"),
                ("start-v2", "start", "REPLACE", "conversation-b"),
                ("other-start", "other", "ADD", "conversation-a"),
            ] {
                uow.append_raw_entity_event(
                    "conversation",
                    NewRawEntityEvent {
                        event_id: event_id.to_owned(),
                        entity_kind: "node".to_owned(),
                        entity_id: entity_id.to_owned(),
                        op: op.to_owned(),
                        payload_json: serde_json::json!({
                            "metadata": {"conversation_id": conversation_id}
                        })
                        .to_string(),
                    },
                )?;
            }
            let matching = uow.latest_entity_events_by_metadata(
                "conversation",
                "node",
                "conversation_id",
                "conversation-a",
            )?;
            assert_eq!(matching.len(), 1);
            assert_eq!(matching[0].entity_id, "other");
            assert_eq!(matching[0].event_id, "other-start");
            Ok(())
        })
        .unwrap();
    drop(store);
    fs::remove_file(path).unwrap();
}

#[test]
fn global_user_and_scoped_sequences_are_isolated_and_current() {
    let (store, path) = store("sequences");
    assert_eq!(store.current_global_seq().unwrap(), 0);
    assert_eq!(store.next_global_seq().unwrap(), 1);
    assert_eq!(store.next_global_seq().unwrap(), 2);
    assert_eq!(store.current_user_seq("alice").unwrap(), 0);
    assert_eq!(store.next_user_seq("alice").unwrap(), 1);
    assert_eq!(store.next_user_seq("alice").unwrap(), 2);
    assert_eq!(store.next_scoped_seq("project-7").unwrap(), 1);
    assert_eq!(store.current_scoped_seq("project-7").unwrap(), 1);
    assert_eq!(store.current_user_seq("alice").unwrap(), 2);
    store.set_scoped_seq("project-7", 11).unwrap();
    assert_eq!(store.next_user_seq("project-7").unwrap(), 12);
    assert!(matches!(
        store.set_user_seq("alice", -1),
        Err(SqliteStoreError::NegativeSequenceValue { value: -1 })
    ));
    drop(store);
    fs::remove_file(path).unwrap();
}
#[test]
fn append_replay_idempotency_and_cross_namespace_collision_follow_contract() {
    let (store, path) = store("events");
    let first = block_on(EventWriteStore::append_entity_event(
        &store,
        "one",
        event("same", "one"),
    ))
    .unwrap();
    let second = block_on(EventWriteStore::append_entity_event(
        &store,
        "one",
        event("second", "two"),
    ))
    .unwrap();
    let retry = block_on(EventWriteStore::append_entity_event(
        &store,
        "one",
        event("same", "changed"),
    ))
    .unwrap();
    assert_eq!((first.event.seq, second.event.seq), (1, 2));
    assert!(first.inserted);
    assert!(!retry.inserted);
    assert_eq!(retry.event.entity_id, "one");
    assert_eq!(
        block_on(EventReadStore::replay_events(&store, "one", 1, 10))
            .unwrap()
            .iter()
            .map(|event| event.seq)
            .collect::<Vec<_>>(),
        [2]
    );
    assert!(matches!(
        store.append_raw_entity_event("two", raw_event("same", "{}")),
        Err(SqliteStoreError::EventIdNamespaceCollision { ref existing_namespace, .. }) if existing_namespace == "one"
    ));
    assert_eq!(
        block_on(EventWriteStore::append_entity_event(
            &store,
            "two",
            event("same", "changed"),
        )),
        Err(StoreError::EventIdNamespaceCollision {
            event_id: "same".to_owned(),
            existing_namespace: "one".to_owned(),
            requested_namespace: "two".to_owned(),
        })
    );
    assert_eq!(store.alloc_event_seq("gap").unwrap(), 1);
    assert_eq!(
        block_on(EventReadStore::latest_event_seq(&store, "gap")).unwrap(),
        0
    );
    drop(store);
    fs::remove_file(path).unwrap();
}
#[test]
fn cursor_trait_is_strict_while_legacy_set_is_permissive() {
    let (store, path) = store("cursors");
    block_on(EventWriteStore::append_entity_event(
        &store,
        "ns",
        event("one", "one"),
    ))
    .unwrap();
    assert_eq!(
        block_on(EventWriteStore::advance_replay_cursor(
            &store, "ns", "sink", 1
        ))
        .unwrap()
        .last_seq,
        1
    );
    assert_eq!(
        block_on(EventWriteStore::advance_replay_cursor(
            &store, "ns", "sink", 0
        )),
        Err(StoreError::CursorRegresses {
            current: 1,
            requested: 0
        })
    );
    assert_eq!(
        block_on(EventWriteStore::advance_replay_cursor(
            &store, "ns", "sink", 2
        )),
        Err(StoreError::CursorOutOfRange {
            cursor: 2,
            latest: 1
        })
    );
    assert_eq!(
        store
            .set_replay_cursor_legacy("ns", "sink", 99)
            .unwrap()
            .last_seq,
        99
    );
    assert_eq!(
        store
            .set_replay_cursor_legacy("ns", "sink", -7)
            .unwrap()
            .last_seq,
        -7
    );
    drop(store);
    fs::remove_file(path).unwrap();
}
#[test]
fn arbitrary_precision_event_payload_survives_reopen_and_typed_replay() {
    let (store, path) = store("arbitrary-precision-event");
    let integer = "1606938044258990275541962092341162602522202993782792835301376";
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../../contracts/golden/python-graph-model-payloads-v1.json"
    ))
    .unwrap();
    let payload = fixture["cases"][0]["canonical_json"]
        .as_str()
        .expect("shared Python node fixture")
        .replace(
            "\"level_from_root\":0",
            &format!("\"level_from_root\":{integer}"),
        );
    store
        .append_raw_entity_event("ns", raw_event("large-level", &payload))
        .unwrap();
    drop(store);

    let reopened = SqliteStore::open(&path).unwrap();
    let raw = reopened.replay_raw_events("ns", 0, 1).unwrap();
    assert_eq!(raw[0].payload_json, payload);

    let replayed = block_on(EventReadStore::replay_events(&reopened, "ns", 0, 1)).unwrap();
    assert_eq!(
        serde_json::to_string(&replayed[0].payload).unwrap(),
        payload
    );
    let envelope_wire =
        format!(r#"{{"schema_version":1,"model_class":"node","payload":{payload}}}"#);
    let decoded = kogwistar_contracts::RawGraphModelEnvelopeV1::from_json(&envelope_wire)
        .unwrap()
        .decode_persistent_v2()
        .expect("replayed Python-shaped node retains its exact typed hierarchy level");
    assert_eq!(
        decoded
            .level_from_root
            .as_ref()
            .map(|level| level.as_decimal_str()),
        Some(integer)
    );
    drop(reopened);
    fs::remove_file(path).unwrap();
}
