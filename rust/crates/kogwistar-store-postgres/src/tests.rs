use super::*;
use kogwistar_store::{EventReadStore, EventWriteStore};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::Notify;

static NEXT_SCHEMA: AtomicUsize = AtomicUsize::new(0);

fn test_schema(label: &str) -> String {
    format!(
        "kogwistar_phase3_{}_{}_{}",
        label,
        std::process::id(),
        NEXT_SCHEMA.fetch_add(1, Ordering::Relaxed)
    )
}

#[test]
fn node_document_ids_match_python_sorted_unique_projection() {
    let document = serde_json::json!({
        "mentions": [
            {"spans": [
                {"doc_id": "doc-b"},
                {"document_page_url": "https://example.test/document/doc-a/page"},
                {"doc_id": "", "document_page_url": "https://example.test/document/doc-c"}
            ]},
            {"doc_id": "doc-b"}
        ]
    });
    assert_eq!(
        node_document_ids(&document).into_iter().collect::<Vec<_>>(),
        vec!["doc-a".to_owned(), "doc-b".to_owned(), "doc-c".to_owned()]
    );
}

#[test]
fn graph_reference_rows_match_python_reference_projection_contract() {
    let document = serde_json::json!({
        "doc_id": "doc-fallback",
        "mentions": [{"spans": [
            {
                "doc_id": "doc-explicit",
                "insertion_method": "parser",
                "page_number": 4,
                "excerpt": "evidence",
                "start_char": 8,
                "end_char": 16,
                "verification": {
                    "method": "human",
                    "is_verified": true,
                    "score": 0.9,
                    "notes": "not projected"
                }
            },
            {"doc_id": null, "page_number": 5}
        ]}]
    });
    let rows = graph_reference_rows("node", "n-1", &document);
    assert_eq!(
        rows,
        vec![
            serde_json::json!({
                "id": "n-1::mention::0::span::0",
                "node_id": "n-1",
                "doc_id": "doc-explicit",
                "insertion_method": "parser",
                "verification_method": "human",
                "is_verified": true,
                "verificication_score": 0.9,
                "page_number": 4,
                "excerpt": "evidence",
                "start_char": 8,
                "end_char": 16
            })
            .as_object()
            .unwrap()
            .clone(),
            serde_json::json!({
                "id": "n-1::mention::0::span::1",
                "node_id": "n-1",
                "doc_id": "doc-fallback",
                "page_number": 5
            })
            .as_object()
            .unwrap()
            .clone()
        ]
    );
    assert!(graph_reference_rows("document", "d-1", &document).is_empty());
}

#[tokio::test]
#[ignore = "requires KOGWISTAR_TEST_PG_DSN; run explicitly with --include-ignored"]
async fn graph_mutations_materialize_node_docs_in_the_same_transaction() -> PostgresStoreResult<()>
{
    let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
        .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
    let schema = test_schema("node_docs_projection");
    let store = PostgresStore::from_dsn(&dsn, &schema)?;
    let tables = Tables::new(&schema)?;
    store.ensure_schema().await?;
    store
        .create_graph_schema(3, GraphTableNames::default())
        .await?;
    let scope = kogwistar_store::GraphScope {
        namespace: format!("node-docs-{}", std::process::id()),
        workspace_id: None,
        graph_space: None,
    };
    let node_id = "node-docs-native";
    let mutation = |event_id: &str, op: &str, mention_ids: &[&str]| {
        let document = serde_json::json!({
            "id": node_id,
            "mentions": mention_ids.iter().map(|doc_id| {
                serde_json::json!({"spans": [{"doc_id": doc_id}]})
            }).collect::<Vec<_>>()
        });
        GraphMutation {
            scope: scope.clone(),
            table: "gke_nodes".to_owned(),
            entity_kind: "node".to_owned(),
            event_id: event_id.to_owned(),
            op: op.to_owned(),
            payload: serde_json::json!({"id": node_id}),
            record: GraphRecord {
                id: node_id.to_owned(),
                document: Some(document.to_string()),
                metadata: Map::new(),
                embedding: None,
            },
            embedding_dim: 3,
        }
    };

    store
        .apply_graph_mutation(mutation(
            "node-docs-add",
            "ADD",
            &["doc-b", "doc-a", "doc-b"],
        ))
        .await?;
    let client = store.client().await?;
    let rows = client
            .query(
                &format!(
                    "SELECT id, metadata::TEXT FROM {} WHERE metadata->>'node_id'=$1 ORDER BY metadata->>'doc_id'",
                    tables.node_docs
                ),
                &[&node_id],
            )
            .await
            .map_err(backend)?;
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].get::<_, String>(0), "node-docs-native::doc-a");
    assert_eq!(rows[1].get::<_, String>(0), "node-docs-native::doc-b");
    let projection = store
        .graph_projection_records(GraphProjectionRead {
            scope: scope.clone(),
            table: "gke_nodes".to_owned(),
            ids: Some(vec![node_id.to_owned()]),
            metadata: kogwistar_store::MetadataFilter::default(),
            limit: 1,
        })
        .await?;
    assert_eq!(
        projection[0].metadata.get("doc_ids"),
        Some(&Value::String("[\"doc-a\",\"doc-b\"]".to_owned()))
    );

    store
        .apply_graph_mutation(mutation("node-docs-replace", "REPLACE", &["doc-c"]))
        .await?;
    let rows = client
        .query(
            &format!(
                "SELECT id FROM {} WHERE metadata->>'node_id'=$1",
                tables.node_docs
            ),
            &[&node_id],
        )
        .await
        .map_err(backend)?;
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get::<_, String>(0), "node-docs-native::doc-c");

    store
        .apply_graph_mutation(mutation("node-docs-tombstone", "TOMBSTONE", &["doc-c"]))
        .await?;
    let rows = client
        .query(
            &format!(
                "SELECT id FROM {} WHERE metadata->>'node_id'=$1",
                tables.node_docs
            ),
            &[&node_id],
        )
        .await
        .map_err(backend)?;
    assert!(rows.is_empty());

    client
        .batch_execute(&format!("DROP SCHEMA {} CASCADE", tables.quoted_schema))
        .await
        .map_err(backend)?;
    Ok(())
}

#[tokio::test]
#[ignore = "requires KOGWISTAR_TEST_PG_DSN; run explicitly with --include-ignored"]
async fn rust_projection_worker_uses_postgres_native_writes_and_tombstone_gate()
-> PostgresStoreResult<()> {
    let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
        .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
    let schema = test_schema("native_projection_worker");
    let store = PostgresStore::from_dsn(&dsn, &schema)?;
    store.ensure_schema().await?;
    store
        .create_graph_schema(3, GraphTableNames::default())
        .await?;
    let namespace = format!("native-projection-worker-{}", std::process::id());
    let scope = GraphScope {
        namespace: namespace.clone(),
        workspace_id: None,
        graph_space: None,
    };
    let node_id = "native-worker-node";
    let graph_mutation = |event_id: &str, op: &str| {
        let mut metadata = Map::new();
        if op == "TOMBSTONE" {
            metadata.insert(
                "lifecycle_status".to_owned(),
                Value::String("tombstoned".to_owned()),
            );
        }
        GraphMutation {
            scope: scope.clone(),
            table: "gke_nodes".to_owned(),
            entity_kind: "node".to_owned(),
            event_id: event_id.to_owned(),
            op: op.to_owned(),
            payload: serde_json::json!({"id": node_id}),
            record: GraphRecord {
                id: node_id.to_owned(),
                document: Some("native projection worker text".to_owned()),
                metadata,
                embedding: None,
            },
            embedding_dim: 3,
        }
    };
    store
        .apply_graph_mutation(graph_mutation("native-worker-add", "ADD"))
        .await?;
    store
        .enqueue_index_job(NewIndexJob {
            job_id: "native-worker-upsert".to_owned(),
            namespace: namespace.clone(),
            entity_kind: "node".to_owned(),
            entity_id: node_id.to_owned(),
            index_kind: "node_embedding".to_owned(),
            op: "UPSERT".to_owned(),
            payload_json: None,
            max_retries: 3,
        })
        .await?;

    let provider_calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let calls = provider_calls.clone();
    let summary =
        kogwistar_engine::projection_worker::run_graph_projection_jobs_once_with_provider_input(
            &store,
            &store,
            1,
            30,
            Some(&namespace),
            move |input| {
                let calls = calls.clone();
                async move {
                    calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    Ok(GraphRecord {
                        id: input.entity_id,
                        document: Some("native projection worker text".to_owned()),
                        metadata: Map::new(),
                        embedding: Some(vec![0.0, 1.0, 0.0]),
                    })
                }
            },
        )
        .await?;
    assert_eq!(summary.projected, 1);
    assert_eq!(provider_calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(
        store
            .index_jobs(Some(&namespace), Some("DONE"), None, None, None, 10)
            .await?
            .len(),
        1
    );
    let projected = store
        .graph_projection_records(GraphProjectionRead {
            scope: scope.clone(),
            table: "gke_nodes".to_owned(),
            ids: Some(vec![node_id.to_owned()]),
            metadata: kogwistar_store::MetadataFilter::default(),
            limit: 1,
        })
        .await?;
    assert_eq!(projected[0].embedding, Some(vec![0.0, 1.0, 0.0]));

    store
        .enqueue_index_job(NewIndexJob {
            job_id: "native-worker-stale-provider".to_owned(),
            namespace: namespace.clone(),
            entity_kind: "node".to_owned(),
            entity_id: node_id.to_owned(),
            index_kind: "node_embedding".to_owned(),
            op: "UPSERT".to_owned(),
            payload_json: None,
            max_retries: 3,
        })
        .await?;
    let stale_summary =
        kogwistar_engine::projection_worker::run_graph_projection_jobs_once_with_provider_input(
            &store,
            &store,
            1,
            30,
            Some(&namespace),
            move |input| async move {
                Ok(GraphRecord {
                    id: input.entity_id,
                    document: Some("stale source text".to_owned()),
                    metadata: Map::new(),
                    embedding: Some(vec![1.0, 0.0, 0.0]),
                })
            },
        )
        .await?;
    assert_eq!(stale_summary.projected, 1);
    let after_stale_result = store
        .graph_projection_records(GraphProjectionRead {
            scope: scope.clone(),
            table: "gke_nodes".to_owned(),
            ids: Some(vec![node_id.to_owned()]),
            metadata: kogwistar_store::MetadataFilter::default(),
            limit: 1,
        })
        .await?;
    assert_eq!(
        after_stale_result[0].document.as_deref(),
        Some("native projection worker text")
    );
    assert_eq!(after_stale_result[0].embedding, Some(vec![0.0, 1.0, 0.0]));

    // A stale DELETE must not remove a live replacement.
    store
        .enqueue_index_job(NewIndexJob {
            job_id: "native-worker-stale-delete".to_owned(),
            namespace: namespace.clone(),
            entity_kind: "node".to_owned(),
            entity_id: node_id.to_owned(),
            index_kind: "node_embedding".to_owned(),
            op: "DELETE".to_owned(),
            payload_json: None,
            max_retries: 3,
        })
        .await?;
    let delete_summary =
        kogwistar_engine::projection_worker::run_graph_projection_jobs_once_with_provider_input(
            &store,
            &store,
            1,
            30,
            Some(&namespace),
            |_input| async { panic!("DELETE must not invoke the embedding provider") },
        )
        .await?;
    assert_eq!(delete_summary.projected, 1);
    assert_eq!(
        store
            .graph_projection_records(GraphProjectionRead {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                ids: Some(vec![node_id.to_owned()]),
                metadata: kogwistar_store::MetadataFilter::default(),
                limit: 1,
            })
            .await?
            .len(),
        1
    );

    store
        .apply_graph_mutation(graph_mutation("native-worker-tombstone", "TOMBSTONE"))
        .await?;
    store
        .enqueue_index_job(NewIndexJob {
            job_id: "native-worker-tombstone-delete".to_owned(),
            namespace: namespace.clone(),
            entity_kind: "node".to_owned(),
            entity_id: node_id.to_owned(),
            index_kind: "node_embedding".to_owned(),
            op: "DELETE".to_owned(),
            payload_json: None,
            max_retries: 3,
        })
        .await?;
    let tombstone_summary =
        kogwistar_engine::projection_worker::run_graph_projection_jobs_once_with_provider_input(
            &store,
            &store,
            1,
            30,
            Some(&namespace),
            |_input| async { panic!("DELETE must not invoke the embedding provider") },
        )
        .await?;
    assert_eq!(tombstone_summary.projected, 1);
    assert!(
        store
            .graph_projection_records(GraphProjectionRead {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                ids: Some(vec![node_id.to_owned()]),
                metadata: kogwistar_store::MetadataFilter::default(),
                limit: 1,
            })
            .await?
            .is_empty()
    );

    let client = store.client().await?;
    client
        .batch_execute(&format!(
            "DROP SCHEMA {} CASCADE",
            Tables::new(&schema)?.quoted_schema
        ))
        .await
        .map_err(backend)?;
    Ok(())
}

#[tokio::test]
#[ignore = "requires KOGWISTAR_TEST_PG_DSN; run explicitly with --include-ignored"]
async fn graph_mutation_retry_requires_identical_effect_when_dsn_available()
-> PostgresStoreResult<()> {
    let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
        .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
    let schema = test_schema("graph_mutation_retry");
    let store = PostgresStore::from_dsn(&dsn, &schema)?;
    store.ensure_schema().await?;
    store
        .create_graph_schema(3, GraphTableNames::default())
        .await?;

    let namespace = format!("graph-mutation-retry-{}", std::process::id());
    let mutation = kogwistar_store::GraphMutation {
        scope: kogwistar_store::GraphScope {
            namespace: namespace.clone(),
            workspace_id: None,
            graph_space: None,
        },
        table: "gke_nodes".to_owned(),
        entity_kind: "node".to_owned(),
        event_id: "stable-node-write".to_owned(),
        op: "ADD".to_owned(),
        payload: serde_json::json!({"id": "retry-node", "text": "original"}),
        record: kogwistar_store::GraphRecord {
            id: "retry-node".to_owned(),
            document: Some("original".to_owned()),
            metadata: Map::new(),
            embedding: None,
        },
        embedding_dim: 3,
    };

    let batch = |mutation| kogwistar_store::GraphMutationBatch {
        mutations: vec![mutation],
        index_jobs: vec![kogwistar_store::NewIndexJob {
            job_id: "stable-node-index".to_owned(),
            namespace: namespace.clone(),
            entity_kind: "node".to_owned(),
            entity_id: "retry-node".to_owned(),
            index_kind: "node_docs".to_owned(),
            op: "UPSERT".to_owned(),
            payload_json: None,
            max_retries: 3,
        }],
    };
    let first = store
        .apply_graph_mutations_with_index_jobs_atomic(batch(mutation.clone()))
        .await?;
    let retry = store
        .apply_graph_mutations_with_index_jobs_atomic(batch(mutation.clone()))
        .await?;
    assert!(first[0].inserted && first[0].mutated);
    assert!(!retry[0].inserted && !retry[0].mutated);
    assert_eq!(retry[0].event.seq, first[0].event.seq);

    let mut conflicting = mutation;
    conflicting.payload["text"] = serde_json::json!("different");
    assert!(matches!(
        store
            .apply_graph_mutations_with_index_jobs_atomic(batch(conflicting))
            .await,
        Err(PostgresStoreError::Backend(message))
            if message.contains("reused with different input")
    ));

    let events = store.replay_raw_events(&namespace, 0, 10).await?;
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].event_id, "stable-node-write");
    let jobs = store
        .list_index_jobs(Some(&namespace), None, None, None, None, 10)
        .await?;
    assert_eq!(jobs.len(), 1);
    assert_eq!(jobs[0].job_id, "stable-node-index");
    let rows = store
        .graph_projection_records(GraphProjectionRead {
            scope: kogwistar_store::GraphScope {
                namespace: namespace.clone(),
                workspace_id: None,
                graph_space: None,
            },
            table: "gke_nodes".to_owned(),
            ids: Some(vec!["retry-node".to_owned()]),
            metadata: kogwistar_store::MetadataFilter::default(),
            limit: 1,
        })
        .await?;
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].document.as_deref(), Some("original"));

    let cleanup = store.client().await?;
    cleanup
        .batch_execute(&format!(
            "DROP SCHEMA {} CASCADE",
            Tables::new(&schema)?.quoted_schema
        ))
        .await
        .map_err(backend)?;
    Ok(())
}

#[test]
fn physical_graph_purge_requires_canonical_tombstone() {
    let active = Map::new();
    assert!(!canonical_tombstone_present(&active));

    let mut tombstoned = Map::new();
    tombstoned.insert(
        "lifecycle_status".to_owned(),
        Value::String("tombstoned".to_owned()),
    );
    assert!(canonical_tombstone_present(&tombstoned));

    tombstoned.insert(
        "lifecycle_status".to_owned(),
        Value::String("deleted".to_owned()),
    );
    assert!(canonical_tombstone_present(&tombstoned));
}

#[test]
fn lifecycle_mutations_accept_only_nodes_and_edges() {
    assert!(validate_graph_lifecycle_target("gke_nodes", "node").is_ok());
    assert!(validate_graph_lifecycle_target("gke_edges", "edge").is_ok());
    assert!(validate_graph_lifecycle_target("gke_nodes", "document").is_err());
    assert!(validate_graph_lifecycle_target("gke_edges", "node").is_err());
    assert!(validate_graph_lifecycle_target("gke_documents", "document").is_err());
    assert!(validate_graph_lifecycle_target("gke_domains", "domain").is_err());
}

#[test]
fn graph_mutations_map_each_entity_kind_to_exactly_one_table() {
    let targets = [
        ("node", "gke_nodes"),
        ("edge", "gke_edges"),
        ("document", "gke_documents"),
        ("domain", "gke_domains"),
    ];
    for (entity_kind, expected_table) in targets {
        assert!(validate_graph_projection_target(expected_table, entity_kind).is_ok());
        for (_, other_table) in targets {
            if other_table != expected_table {
                assert!(validate_graph_projection_target(other_table, entity_kind).is_err());
            }
        }
    }
    assert!(validate_graph_projection_target("gke_nodes", "unknown").is_err());
}

fn raw(id: &str, payload_json: &str) -> NewRawEntityEvent {
    NewRawEntityEvent {
        event_id: id.to_owned(),
        entity_kind: "node".to_owned(),
        entity_id: "n".to_owned(),
        op: "UPSERT".to_owned(),
        payload_json: payload_json.to_owned(),
    }
}

fn runtime_start(run_id: &str, transition_id: &str) -> RecordedRuntimeTransition {
    serde_json::from_value(serde_json::json!({
        "contract_version": 1,
        "transition_id": transition_id,
        "expected_event_seq": 0,
        "kind": "start",
        "run_id": run_id,
        "workflow_id": "wf-1",
        "conversation_id": "conv-1",
        "user_id": "user-1",
        "user_turn_node_id": "turn-1",
        "step_seq": 0,
        "node_id": "step-1",
        "token_id": "token-1",
        "parent_token_id": null,
        "initial_state": {"answer": "seed"},
        "frontier": {
            "pending": [["step-1", 0, "token-1", null]],
            "suspended": [],
            "join_node_ids": [],
            "join_outstanding": [],
            "join_waiters": {}
        }
    }))
    .unwrap()
}

fn runtime_result(
    run_id: &str,
    transition_id: &str,
    expected_event_seq: i64,
    answer: &str,
) -> RecordedRuntimeTransition {
    serde_json::from_value(serde_json::json!({
        "contract_version": 1,
        "transition_id": transition_id,
        "expected_event_seq": expected_event_seq,
        "kind": "recorded_step_success",
        "run_id": run_id,
        "workflow_id": "wf-1",
        "conversation_id": "conv-1",
        "step_seq": 0,
        "node_id": "step-1",
        "token_id": "token-1",
        "parent_token_id": null,
        "state_update": [["u", {"answer": answer}]],
        "frontier": {
            "pending": [],
            "suspended": [],
            "join_node_ids": [],
            "join_outstanding": [],
            "join_waiters": {}
        },
        "result": {"answer": answer}
    }))
    .unwrap()
}

fn runtime_failure_start(run_id: &str, transition_id: &str) -> RecordedRuntimeTransition {
    let mut start = runtime_start(run_id, transition_id);
    start.initial_state.as_mut().unwrap().insert(
        "_rt_routes".to_owned(),
        serde_json::json!([{
            "edge_id": "recover-edge",
            "source_node_id": "step-1",
            "target_node_id": "recover",
            "aliases": ["recover"],
            "join_mask": 0,
            "predicate": "on_failure",
            "multiplicity": "one",
            "is_default": false,
            "priority": 100,
            "source_fanout": false
        }]),
    );
    start
}

fn runtime_lane(run_id: &str, message_id: &str) -> NewProjectedLaneMessage {
    NewProjectedLaneMessage {
        message_id: message_id.to_owned(),
        namespace: "runtime".to_owned(),
        purpose: "user_visible".to_owned(),
        inbox_id: "python-workers".to_owned(),
        conversation_id: "conv-1".to_owned(),
        recipient_id: "python-worker".to_owned(),
        sender_id: "rust-runtime".to_owned(),
        msg_type: "workflow.worker.request.v1".to_owned(),
        status: "pending".to_owned(),
        created_at: 1,
        available_at: 0,
        run_id: Some(run_id.to_owned()),
        step_id: Some("step-1".to_owned()),
        correlation_id: Some("corr-1".to_owned()),
        payload_json: Some("{}".to_owned()),
        error_json: None,
    }
}

fn runtime_handoff(run_id: &str, message_id: &str, owner: &str) -> RecordedWorkerHandoff {
    RecordedWorkerHandoff {
        message_id: message_id.to_owned(),
        claimed_by: owner.to_owned(),
        run_id: run_id.to_owned(),
        step_id: "step-1".to_owned(),
        correlation_id: "corr-1".to_owned(),
    }
}

fn runtime_worker_lane(run_id: &str, message_id: &str) -> NewProjectedLaneMessage {
    let mut lane = runtime_lane(run_id, message_id);
    lane.correlation_id = Some(run_id.to_owned());
    lane.payload_json = Some(
        serde_json::json!({
            "workflow_id": "wf-1",
            "conversation_id": "conv-1",
            "token_id": "token-1",
            "parent_token_id": null
        })
        .to_string(),
    );
    lane
}

fn runtime_worker_handoff(run_id: &str, message_id: &str, owner: &str) -> RecordedWorkerHandoff {
    let mut handoff = runtime_handoff(run_id, message_id, owner);
    handoff.correlation_id = run_id.to_owned();
    handoff
}

fn runtime_failure_effect(
    effect_id: &str,
    route_next: &[&str],
    successors: Value,
) -> RecordedWorkerSuccessEffect {
    serde_json::from_value(serde_json::json!({
        "contract_version": 1,
        "effect_id": effect_id,
        "status": "failed",
        "state_update": [["u", {"attempted": true}]],
        "successors": successors,
        "route_next": route_next,
        "result": {"workflow_status": "failed"},
        "errors": ["boom"]
    }))
    .unwrap()
}

#[test]
fn identifier_validation_and_qualification_are_strict() {
    assert_eq!(quote_identifier("Mixed_Name9").unwrap(), "\"Mixed_Name9\"");
    for invalid in ["", "9name", "has-dash", "has space", "public;DROP"] {
        assert!(matches!(
            quote_identifier(invalid),
            Err(PostgresStoreError::InvalidSchema { .. })
        ));
    }
    let tables = Tables::new("CamelCase").unwrap();
    assert_eq!(tables.entity_events, "\"CamelCase\".\"entity_events\"");
    assert!(!tables.entity_events.contains("search_path"));
}

#[test]
fn graph_read_filter_accepts_only_legacy_default_scope_rows() {
    let default_scope = kogwistar_store::GraphScope {
        namespace: "default".to_owned(),
        workspace_id: None,
        graph_space: None,
    };
    let (sql, values) = graph_filter_sql(
        &default_scope,
        &kogwistar_store::MetadataFilter::default(),
        1,
    )
    .unwrap();
    assert_eq!(values, vec!["namespace"]);
    assert!(sql.contains("metadata -> $1::TEXT"));
    assert!(sql.contains("NOT (metadata ? 'namespace')"));

    let scoped = kogwistar_store::GraphScope {
        namespace: "default".to_owned(),
        workspace_id: Some("workspace".to_owned()),
        graph_space: None,
    };
    let (sql, values) =
        graph_filter_sql(&scoped, &kogwistar_store::MetadataFilter::default(), 1).unwrap();
    assert_eq!(values, vec!["namespace", "workspace_id"]);
    assert!(!sql.contains("NOT (metadata"));
}

#[test]
fn sql_objects_are_schema_qualified_and_python_shaped() {
    let tables = Tables::new("events").unwrap();
    assert_eq!(tables.namespace_seq, "\"events\".\"namespace_seq\"");
    assert_eq!(tables.replay_cursors, "\"events\".\"replay_cursors\"");
    assert_eq!(tables.aggregate_index, "\"idx_entity_events_aggregate\"");
    assert_eq!(cursor_lock_key("a", "b"), "a\u{1f}b");
    let sql = schema_sql(&tables);
    assert!(sql.contains("CREATE SCHEMA IF NOT EXISTS \"events\""));
    assert!(sql.contains("CREATE TABLE IF NOT EXISTS \"events\".\"namespace_seq\""));
    assert!(sql.contains("event_id TEXT NOT NULL"));
    assert!(sql.contains("UNIQUE(event_id)"));
    assert!(sql.contains("created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()"));
    assert!(sql.contains("PRIMARY KEY(namespace, consumer)"));
    assert!(!sql.contains("search_path"));
    assert_eq!(
        allocation_sql(&tables),
        "INSERT INTO \"events\".\"namespace_seq\" AS target(namespace, next_seq) VALUES ($1, 2) ON CONFLICT(namespace) DO UPDATE SET next_seq = target.next_seq + 1 RETURNING next_seq - 1"
    );
}

#[tokio::test]
#[ignore = "requires KOGWISTAR_TEST_PG_DSN; run explicitly with --include-ignored"]
async fn postgres_integration_event_store_core() -> PostgresStoreResult<()> {
    let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
        .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
    let schema = test_schema("core");
    let store = PostgresStore::from_dsn(&dsn, &schema)?;
    store.ensure_schema().await?;

    // Rust-created schema can be queried through Python-compatible DDL.
    let client = store.client().await?;
    let tables = Tables::new(&schema)?;
    let columns = client
            .query_one(
                "SELECT count(*) FROM information_schema.columns WHERE table_schema = $1 AND table_name = 'entity_events'",
                &[&schema],
            )
            .await
            .map_err(backend)?;
    assert_eq!(columns.try_get::<_, i64>(0).map_err(backend)?, 8);
    drop(client);

    let first = store
        .append_raw_entity_event("one", raw("same", "{\"spaced\": [ 1, 2 ]}"))
        .await?;
    assert_eq!(first.event.seq, 1);
    assert_eq!(first.event.payload_json, "{\"spaced\": [ 1, 2 ]}");
    let retry = store
        .append_raw_entity_event("one", raw("same", "{}"))
        .await?;
    assert!(!retry.inserted);
    assert_eq!(retry.event.seq, 1);
    assert!(matches!(
        store.append_raw_entity_event("two", raw("same", "{}")).await,
        Err(PostgresStoreError::EventIdNamespaceCollision { ref existing_namespace, ref requested_namespace, .. })
            if existing_namespace == "one" && requested_namespace == "two"
    ));

    for id in ["two", "three"] {
        store.append_raw_entity_event("one", raw(id, "{}")).await?;
    }
    assert_eq!(store.replay_raw_events("one", 1, 1).await?.len(), 1);
    assert_eq!(store.latest_retained_event_seq("one").await?, 3);
    assert_eq!(
        EventWriteStore::advance_replay_cursor(&store, "one", "sink", 3)
            .await?
            .last_seq,
        3
    );
    assert!(matches!(
        EventWriteStore::advance_replay_cursor(&store, "one", "sink", 2).await,
        Err(StoreError::CursorRegresses {
            current: 3,
            requested: 2
        })
    ));
    assert!(matches!(
        EventWriteStore::advance_replay_cursor(&store, "one", "sink", 4).await,
        Err(StoreError::CursorOutOfRange {
            cursor: 4,
            latest: 3
        })
    ));
    assert_eq!(
        store
            .set_replay_cursor_legacy("one", "sink", 99)
            .await?
            .last_seq,
        99
    );

    let rollback: PostgresStoreResult<()> = store
        .transaction(|uow| {
            Box::pin(async move {
                uow.append_raw_entity_event("rollback", raw("rollback-id", "{}"))
                    .await?;
                uow.set_replay_cursor_legacy("rollback", "sink", 1).await?;
                Err(PostgresStoreError::TransactionAborted("test".to_owned()))
            })
        })
        .await;
    assert!(matches!(
        rollback,
        Err(PostgresStoreError::TransactionAborted(_))
    ));
    assert_eq!(store.latest_retained_event_seq("rollback").await?, 0);
    assert_eq!(store.replay_cursor("rollback", "sink").await?.last_seq, 0);

    // Cancelling after a write drops the native Transaction and must not
    // return an open transaction to the pool or retain partial state.
    let write_reached = Arc::new(Notify::new());
    let task_store = store.clone();
    let task_reached = Arc::clone(&write_reached);
    let cancelled = tokio::spawn(async move {
        task_store
            .transaction(|uow| {
                Box::pin(async move {
                    uow.append_raw_entity_event("cancelled", raw("cancelled-id", "{}"))
                        .await?;
                    task_reached.notify_one();
                    std::future::pending::<()>().await;
                    #[allow(unreachable_code)]
                    Ok(())
                })
            })
            .await
    });
    write_reached.notified().await;
    cancelled.abort();
    assert!(
        cancelled
            .await
            .expect_err("cancelled transaction completed")
            .is_cancelled()
    );
    assert_eq!(store.latest_retained_event_seq("cancelled").await?, 0);

    let concurrent = (0..8).map(|n| {
        let store = store.clone();
        tokio::spawn(async move {
            store
                .append_raw_entity_event("concurrent", raw(&format!("e-{n}"), "{}"))
                .await
                .map(|row| row.event.seq)
        })
    });
    let mut sequences = Vec::new();
    for task in concurrent {
        sequences.push(task.await.expect("task panicked")?);
    }
    sequences.sort_unstable();
    assert_eq!(sequences, (1..=8).collect::<Vec<_>>());

    // Search path deliberately points elsewhere; all store queries remain isolated.
    let client = store.client().await?;
    client
        .batch_execute("SET search_path TO pg_catalog")
        .await
        .map_err(backend)?;
    drop(client);
    assert_eq!(EventReadStore::latest_event_seq(&store, "one").await?, 3);

    let isolated_schema = test_schema("isolated");
    let isolated = PostgresStore::from_dsn(&dsn, &isolated_schema)?;
    isolated.ensure_schema().await?;
    assert_eq!(isolated.latest_retained_event_seq("one").await?, 0);
    isolated
        .append_raw_entity_event("one", raw("isolated-id", "{}"))
        .await?;
    assert_eq!(isolated.latest_retained_event_seq("one").await?, 1);
    assert_eq!(store.latest_retained_event_seq("one").await?, 3);

    let cleaner = store.client().await?;
    cleaner
        .batch_execute(&format!("DROP SCHEMA {} CASCADE", tables.quoted_schema))
        .await
        .map_err(backend)?;
    let isolated_tables = Tables::new(&isolated_schema)?;
    let cleaner = store.client().await?;
    cleaner
        .batch_execute(&format!(
            "DROP SCHEMA {} CASCADE",
            isolated_tables.quoted_schema
        ))
        .await
        .map_err(backend)?;
    Ok(())
}

#[tokio::test]
#[ignore = "requires KOGWISTAR_TEST_PG_DSN; run explicitly with --include-ignored"]
async fn document_projection_upsert_reopens_and_retries_idempotently_when_dsn_available()
-> PostgresStoreResult<()> {
    let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
        .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
    let schema = test_schema("document_projection");
    let store = PostgresStore::from_dsn(&dsn, &schema)?;
    store.ensure_schema().await?;
    store
        .create_graph_schema(3, GraphTableNames::default())
        .await?;

    let scope = kogwistar_store::GraphScope {
        namespace: "docs".to_owned(),
        workspace_id: Some("workspace-a".to_owned()),
        graph_space: Some("knowledge".to_owned()),
    };
    let mut metadata = Map::new();
    metadata.insert("doc_type".to_owned(), Value::String("text".to_owned()));
    metadata.insert(
        "insertion_method".to_owned(),
        Value::String("integration-test".to_owned()),
    );
    let write = DocumentProjectionUpsert {
        scope: scope.clone(),
        record: GraphRecord {
            id: "document-1".to_owned(),
            document: Some("first body".to_owned()),
            metadata,
            embedding: None,
        },
        embedding_dim: 0,
    };
    DocumentStore::upsert_document_projection(&store, write.clone()).await?;

    // A retry with the same identity replaces the projection rather than
    // creating a duplicate row, while preserving the scoped contract.
    let mut retry = write;
    retry.record.document = Some("updated body".to_owned());
    DocumentStore::upsert_document_projection(&store, retry).await?;

    let read = GraphProjectionRead {
        scope: scope.clone(),
        table: "gke_documents".to_owned(),
        ids: Some(vec!["document-1".to_owned()]),
        metadata: kogwistar_store::MetadataFilter::default(),
        limit: 10,
    };
    let rows = store.graph_projection_records(read.clone()).await?;
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].document.as_deref(), Some("updated body"));
    assert_eq!(rows[0].metadata["namespace"], "docs");
    assert_eq!(rows[0].metadata["workspace_id"], "workspace-a");
    assert_eq!(rows[0].metadata["graph_space"], "knowledge");
    assert_eq!(rows[0].metadata["doc_type"], "text");
    assert_eq!(rows[0].metadata["insertion_method"], "integration-test");

    let next_page = store
        .graph_projection_page(GraphProjectionPageRead {
            read: read.clone(),
            after_id: Some("document-0".to_owned()),
        })
        .await?;
    assert_eq!(next_page, rows);
    let exhausted_page = store
        .graph_projection_page(GraphProjectionPageRead {
            read: read.clone(),
            after_id: Some("document-1".to_owned()),
        })
        .await?;
    assert!(exhausted_page.is_empty());

    let reopened = PostgresStore::from_dsn(&dsn, &schema)?;
    let reopened_rows = reopened.graph_projection_records(read).await?;
    assert_eq!(reopened_rows, rows);

    let cleanup = reopened.client().await?;
    cleanup
        .batch_execute(&format!(
            "DROP SCHEMA {} CASCADE",
            Tables::new(&schema)?.quoted_schema
        ))
        .await
        .map_err(backend)?;
    Ok(())
}

#[tokio::test]
#[ignore = "requires KOGWISTAR_TEST_PG_DSN; run explicitly with --include-ignored"]
async fn arbitrary_precision_graph_mutation_survives_postgres_reopen_when_dsn_available()
-> PostgresStoreResult<()> {
    let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
        .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
    let schema = test_schema("arbitrary_precision");
    let store = PostgresStore::from_dsn(&dsn, &schema)?;
    store.ensure_schema().await?;
    store
        .create_graph_schema(3, GraphTableNames::default())
        .await?;

    let integer = "1606938044258990275541962092341162602522202993782792835301376";
    let entity_id = "large-level-node";
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../../contracts/golden/python-graph-model-payloads-v1.json"
    ))?;
    let payload_json = fixture["cases"][0]["canonical_json"]
        .as_str()
        .expect("shared Python node fixture")
        .replace("\"id\":\"parity-node\"", &format!("\"id\":\"{entity_id}\""))
        .replace(
            "\"level_from_root\":0",
            &format!("\"level_from_root\":{integer}"),
        );
    let payload: Value = serde_json::from_str(&payload_json)?;
    let metadata: Map<String, Value> =
        serde_json::from_str(&format!(r#"{{"level_from_root":{integer}}}"#))?;
    let namespace = format!("large-integer-{}", std::process::id());
    let scope = kogwistar_store::GraphScope {
        namespace: namespace.clone(),
        workspace_id: None,
        graph_space: None,
    };
    store
        .apply_graph_mutation(kogwistar_store::GraphMutation {
            scope: scope.clone(),
            table: "gke_nodes".to_owned(),
            entity_kind: "node".to_owned(),
            event_id: format!("event-{entity_id}"),
            op: "ADD".to_owned(),
            payload,
            record: kogwistar_store::GraphRecord {
                id: entity_id.to_owned(),
                document: Some("arbitrary precision integration probe".to_owned()),
                metadata,
                embedding: None,
            },
            embedding_dim: 3,
        })
        .await?;

    let read = GraphProjectionRead {
        scope,
        table: "gke_nodes".to_owned(),
        ids: Some(vec![entity_id.to_owned()]),
        metadata: kogwistar_store::MetadataFilter::default(),
        limit: 1,
    };
    let reopened = PostgresStore::from_dsn(&dsn, &schema)?;
    let rows = reopened.graph_projection_records(read).await?;
    assert_eq!(rows.len(), 1);
    assert_eq!(
        serde_json::to_string(&rows[0].metadata["level_from_root"])?,
        integer
    );
    let events = reopened.replay_raw_events(&namespace, 0, 10).await?;
    assert_eq!(events.len(), 1);
    assert!(events[0].payload_json.contains(integer));
    let event_payload: Value = serde_json::from_str(&events[0].payload_json)?;
    assert_eq!(event_payload["id"], entity_id);
    assert_eq!(
        serde_json::to_string(&event_payload["level_from_root"])?,
        integer
    );
    let envelope_wire = format!(
        r#"{{"schema_version":1,"model_class":"node","payload":{}}}"#,
        events[0].payload_json
    );
    let decoded = kogwistar_contracts::RawGraphModelEnvelopeV1::from_json(&envelope_wire)
        .expect("persisted Python-shaped node parses")
        .decode_persistent_v2()
        .expect("persisted node retains its exact typed hierarchy level");
    assert_eq!(
        decoded
            .level_from_root
            .as_ref()
            .map(|level| level.as_decimal_str()),
        Some(integer)
    );
    assert!(decoded.raw_payload.get().contains(integer));

    let cleanup = reopened.client().await?;
    cleanup
        .batch_execute(&format!(
            "DROP SCHEMA {} CASCADE",
            Tables::new(&schema)?.quoted_schema
        ))
        .await
        .map_err(backend)?;
    Ok(())
}

#[tokio::test]
#[ignore = "requires KOGWISTAR_TEST_PG_DSN; run explicitly with --include-ignored"]
async fn graph_mutation_batch_rolls_back_when_index_intent_fails_when_dsn_available()
-> PostgresStoreResult<()> {
    let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
        .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
    let schema = test_schema("graph_batch_rollback");
    let store = PostgresStore::from_dsn(&dsn, &schema)?;
    store.ensure_schema().await?;
    store
        .create_graph_schema(3, GraphTableNames::default())
        .await?;
    let tables = Tables::new(&schema)?;
    store
        .client()
        .await?
        .batch_execute(&format!(
            "CREATE FUNCTION {}.reject_batch_index_job() RETURNS trigger \
                 LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'injected index job failure'; \
                 END; $$; CREATE TRIGGER reject_batch_index_job BEFORE INSERT ON {} \
                 FOR EACH ROW EXECUTE FUNCTION {}.reject_batch_index_job();",
            tables.quoted_schema, tables.index_jobs, tables.quoted_schema
        ))
        .await
        .map_err(backend)?;

    let namespace = format!("batch-{}", std::process::id());
    let entity_id = "batch-rollback-node";
    let mutation = kogwistar_store::GraphMutation {
        scope: kogwistar_store::GraphScope {
            namespace: namespace.clone(),
            workspace_id: None,
            graph_space: None,
        },
        table: "gke_nodes".to_owned(),
        entity_kind: "node".to_owned(),
        event_id: format!("event-{entity_id}"),
        op: "ADD".to_owned(),
        payload: serde_json::json!({"id": entity_id, "entity_type": "node"}),
        record: kogwistar_store::GraphRecord {
            id: entity_id.to_owned(),
            document: Some("atomic batch rollback probe".to_owned()),
            metadata: Map::new(),
            embedding: None,
        },
        embedding_dim: 3,
    };
    let edge_id = "batch-rollback-edge";
    let edge_mutation = kogwistar_store::GraphMutation {
        scope: kogwistar_store::GraphScope {
            namespace: namespace.clone(),
            workspace_id: None,
            graph_space: None,
        },
        table: "gke_edges".to_owned(),
        entity_kind: "edge".to_owned(),
        event_id: format!("event-{edge_id}"),
        op: "ADD".to_owned(),
        payload: serde_json::json!({
            "id": edge_id,
            "source_ids": ["batch-source"],
            "target_ids": ["batch-target"]
        }),
        record: kogwistar_store::GraphRecord {
            id: edge_id.to_owned(),
            document: Some(format!(
                r#"{{"id":"{edge_id}","source_ids":["batch-source"],"target_ids":["batch-target"]}}"#
            )),
            metadata: Map::new(),
            embedding: None,
        },
        embedding_dim: 3,
    };
    let batch = kogwistar_store::GraphMutationBatch {
        mutations: vec![mutation, edge_mutation],
        index_jobs: vec![kogwistar_store::NewIndexJob {
            job_id: format!("job-{entity_id}"),
            namespace: namespace.clone(),
            entity_kind: "node".to_owned(),
            entity_id: entity_id.to_owned(),
            index_kind: "node_docs".to_owned(),
            op: "UPSERT".to_owned(),
            payload_json: None,
            max_retries: 3,
        }],
    };

    assert!(matches!(
        store
            .apply_graph_mutations_with_index_jobs_atomic(batch)
            .await,
        Err(PostgresStoreError::Backend(_))
    ));
    assert!(store.replay_raw_events(&namespace, 0, 10).await?.is_empty());
    assert!(
        store
            .graph_projection_records(GraphProjectionRead {
                scope: kogwistar_store::GraphScope {
                    namespace: namespace.clone(),
                    workspace_id: None,
                    graph_space: None,
                },
                table: "gke_nodes".to_owned(),
                ids: Some(vec![entity_id.to_owned()]),
                metadata: kogwistar_store::MetadataFilter::default(),
                limit: 10,
            })
            .await?
            .is_empty()
    );
    assert!(
        store
            .graph_projection_records(GraphProjectionRead {
                scope: kogwistar_store::GraphScope {
                    namespace: namespace.clone(),
                    workspace_id: None,
                    graph_space: None,
                },
                table: "gke_edges".to_owned(),
                ids: Some(vec![edge_id.to_owned()]),
                metadata: kogwistar_store::MetadataFilter::default(),
                limit: 10,
            })
            .await?
            .is_empty()
    );
    let endpoint_row_count: i64 = store
        .client()
        .await?
        .query_one(
            &format!(
                "SELECT COUNT(*) FROM {} WHERE metadata ->> 'edge_id' = $1",
                tables.edge_endpoints
            ),
            &[&edge_id],
        )
        .await
        .map_err(backend)?
        .try_get(0)
        .map_err(backend)?;
    assert_eq!(endpoint_row_count, 0);
    assert!(
        store
            .claim_index_jobs(10, 30, Some(&namespace))
            .await?
            .is_empty()
    );

    let cleanup = store.client().await?;
    cleanup
        .batch_execute(&format!(
            "DROP SCHEMA {} CASCADE",
            Tables::new(&schema)?.quoted_schema
        ))
        .await
        .map_err(backend)?;
    Ok(())
}

#[tokio::test]
#[ignore = "requires KOGWISTAR_TEST_PG_DSN; run explicitly with --include-ignored"]
async fn recorded_runtime_and_claimed_handoff_are_atomic_when_dsn_available()
-> PostgresStoreResult<()> {
    let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
        .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
    let schema = test_schema("recorded_runtime");
    let store = PostgresStore::from_dsn(&dsn, &schema)?;
    let tables = Tables::new(&schema)?;
    store.ensure_schema().await?;

    let planner = store.client().await?;
    planner
        .batch_execute("SET enable_seqscan=off")
        .await
        .map_err(backend)?;
    let plan = planner
        .query(
            &format!(
                "EXPLAIN SELECT payload_json, seq FROM {} \
                     WHERE event_type='workflow.recorded_transition.v1' \
                     AND payload_json::jsonb->>'transition_id'=$1 LIMIT 1",
                tables.server_run_events
            ),
            &[&"index-plan-probe"],
        )
        .await
        .map_err(backend)?
        .iter()
        .map(|row| row.get::<_, String>(0))
        .collect::<Vec<_>>()
        .join(" ");
    assert!(
        plan.contains("idx_server_run_events_recorded_transition_id"),
        "{plan}"
    );

    let start_1 = store
        .apply_recorded_runtime_transition(runtime_start("run-1", "start-1"), false)
        .await?;
    store
        .project_lane_message(runtime_lane("run-1", "request-1"))
        .await?;
    let claimed = store
        .claim_projected_lane_messages("runtime", "python-workers", "worker-1", 1, 30)
        .await?;
    assert_eq!(claimed.len(), 1);
    let transition = runtime_result("run-1", "result-1", start_1.event_seq, "worker");
    let handoff = runtime_handoff("run-1", "request-1", "worker-1");
    let first = store
        .apply_claimed_recorded_runtime_transition(handoff.clone(), transition.clone(), false)
        .await?;
    let retry = store
        .apply_claimed_recorded_runtime_transition(handoff.clone(), transition.clone(), false)
        .await?;
    assert!(!first.idempotent && retry.idempotent);
    assert_eq!(first.event_seq, retry.event_seq);
    assert_eq!(
        store
            .get_projected_lane_message("request-1")
            .await?
            .unwrap()
            .status,
        "completed"
    );
    assert_eq!(
        store
            .read_recorded_runtime_state("run-1", "wf-1", "conv-1")
            .await?
            .unwrap()
            .state["answer"],
        serde_json::json!("worker")
    );
    assert!(matches!(
        store
            .apply_claimed_recorded_runtime_transition(
                handoff,
                runtime_result("run-1", "result-1", start_1.event_seq, "changed"),
                false,
            )
            .await,
        Err(PostgresStoreError::RecordedRuntimeConflict(_))
    ));

    let start_2 = store
        .apply_recorded_runtime_transition(runtime_start("run-2", "start-2"), false)
        .await?;
    store
        .project_lane_message(runtime_lane("run-2", "request-2"))
        .await?;
    store
        .claim_projected_lane_messages("runtime", "python-workers", "worker-2", 1, 30)
        .await?;
    assert!(matches!(
        store
            .apply_claimed_recorded_runtime_transition(
                runtime_handoff("run-2", "request-2", "wrong-worker"),
                runtime_result("run-2", "result-2", start_2.event_seq, "worker"),
                false,
            )
            .await,
        Err(PostgresStoreError::RecordedRuntimeConflict(_))
    ));

    let start_3 = store
        .apply_recorded_runtime_transition(runtime_start("run-3", "start-3"), false)
        .await?;
    store
        .project_lane_message(runtime_lane("run-3", "request-3"))
        .await?;
    store
        .claim_projected_lane_messages("runtime", "python-workers", "dead-worker", 1, -1)
        .await?;
    store
        .claim_projected_lane_messages("runtime", "python-workers", "new-worker", 1, 30)
        .await?;
    assert!(matches!(
        store
            .apply_claimed_recorded_runtime_transition(
                runtime_handoff("run-3", "request-3", "dead-worker"),
                runtime_result("run-3", "result-3", start_3.event_seq, "worker"),
                false,
            )
            .await,
        Err(PostgresStoreError::RecordedRuntimeConflict(_))
    ));
    let reclaimed = store
        .apply_claimed_recorded_runtime_transition(
            runtime_handoff("run-3", "request-3", "new-worker"),
            runtime_result("run-3", "result-3", start_3.event_seq, "worker"),
            false,
        )
        .await?;
    assert!(!reclaimed.idempotent);

    let start_4 = store
        .apply_recorded_runtime_transition(runtime_start("run-4", "start-4"), false)
        .await?;
    let before = store
        .list_server_run_events("run-4", 0, usize::MAX)
        .await?
        .len();
    assert!(matches!(
        store
            .apply_recorded_runtime_transition(
                runtime_result("run-4", "result-4", start_4.event_seq, "rolled"),
                true,
            )
            .await,
        Err(PostgresStoreError::TransactionAborted(_))
    ));
    assert_eq!(
        store
            .list_server_run_events("run-4", 0, usize::MAX)
            .await?
            .len(),
        before
    );

    let start_5 = store
        .apply_recorded_runtime_transition(runtime_start("run-5", "start-5"), false)
        .await?;
    store
        .project_lane_message(runtime_lane("run-5", "request-5"))
        .await?;
    store
        .claim_projected_lane_messages("runtime", "python-workers", "worker-5", 1, 30)
        .await?;
    store.request_server_run_cancel("run-5").await?;
    assert_eq!(
        store.get_server_run("run-5").await?.unwrap().status,
        "cancelled"
    );
    assert_eq!(
        store
            .read_recorded_runtime_state("run-5", "wf-1", "conv-1")
            .await?
            .unwrap()
            .status,
        kogwistar_runtime::RecordedRunStatus::Cancelled
    );
    assert_eq!(
        store
            .get_projected_lane_message("request-5")
            .await?
            .unwrap()
            .status,
        "cancelled"
    );
    assert!(matches!(
        store
            .apply_claimed_recorded_runtime_transition(
                runtime_handoff("run-5", "request-5", "worker-5"),
                runtime_result("run-5", "result-5", start_5.event_seq, "stale"),
                false,
            )
            .await,
        Err(PostgresStoreError::RecordedRuntimeConflict(_))
    ));

    for (run_id, message_id, owner) in [
        ("handled", "handled-request", "handled-worker"),
        ("unhandled", "unhandled-request", "unhandled-worker"),
        ("forged", "forged-request", "forged-worker"),
    ] {
        store
            .apply_recorded_runtime_transition(
                runtime_failure_start(run_id, &format!("start-{run_id}")),
                false,
            )
            .await?;
        store
            .project_lane_message(runtime_worker_lane(run_id, message_id))
            .await?;
        assert_eq!(
            store
                .claim_projected_lane_messages("runtime", "python-workers", owner, 1, 30,)
                .await?
                .len(),
            1
        );
    }

    let handled_handoff = runtime_worker_handoff("handled", "handled-request", "handled-worker");
    let handled_effect =
        runtime_failure_effect("handled-effect", &["recover"], serde_json::json!([]));
    let handled = store
        .apply_claimed_recorded_worker_effect(handled_handoff.clone(), handled_effect.clone())
        .await?;
    let handled_retry = store
        .apply_claimed_recorded_worker_effect(handled_handoff, handled_effect)
        .await?;
    assert!(!handled.idempotent && handled_retry.idempotent);
    assert_eq!(handled.event_seq, handled_retry.event_seq);
    assert_eq!(
        handled.reduced.state.status,
        kogwistar_runtime::RecordedRunStatus::Running
    );
    assert_eq!(handled.reduced.state.frontier.pending[0].0, "recover");
    assert_eq!(
        store
            .get_projected_lane_message("handled-request")
            .await?
            .unwrap()
            .status,
        "completed"
    );

    let unhandled = store
        .apply_claimed_recorded_worker_effect(
            runtime_worker_handoff("unhandled", "unhandled-request", "unhandled-worker"),
            runtime_failure_effect("unhandled-effect", &[], serde_json::json!([])),
        )
        .await?;
    assert_eq!(
        unhandled.reduced.state.status,
        kogwistar_runtime::RecordedRunStatus::Failed
    );
    assert!(unhandled.reduced.state.frontier.pending.is_empty());
    assert_eq!(
        store.get_server_run("unhandled").await?.unwrap().status,
        "failed"
    );

    assert!(matches!(
        store
            .apply_claimed_recorded_worker_effect(
                runtime_worker_handoff("forged", "forged-request", "forged-worker"),
                runtime_failure_effect(
                    "forged-effect",
                    &[],
                    serde_json::json!([{"node_id": "forged", "join_mask": 0}]),
                ),
            )
            .await,
        Err(PostgresStoreError::RecordedRuntime(_))
    ));
    assert_eq!(
        store
            .list_server_run_events("forged", 0, usize::MAX)
            .await?
            .len(),
        1
    );
    assert_eq!(
        store
            .get_projected_lane_message("forged-request")
            .await?
            .unwrap()
            .status,
        "claimed"
    );

    store
        .client()
        .await?
        .batch_execute(&format!("DROP SCHEMA {} CASCADE", tables.quoted_schema))
        .await
        .map_err(backend)
}

#[tokio::test]
#[ignore = "requires KOGWISTAR_TEST_PG_DSN; run explicitly with --include-ignored"]
async fn python_created_schema_is_read_and_written_when_dsn_available() -> PostgresStoreResult<()> {
    let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
        .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
    let schema = test_schema("python");
    let store = PostgresStore::from_dsn(&dsn, &schema)?;
    let tables = Tables::new(&schema)?;
    let client = store.client().await?;
    client
            .batch_execute(&format!(
                "CREATE SCHEMA {schema};\
                 CREATE TABLE {ns}(namespace TEXT PRIMARY KEY, next_seq BIGINT NOT NULL);\
                 INSERT INTO {ns} VALUES ('legacy', 7);\
                 CREATE TABLE {events}(namespace TEXT NOT NULL DEFAULT 'default', seq BIGINT NOT NULL, event_id TEXT NOT NULL, entity_kind TEXT NOT NULL, entity_id TEXT NOT NULL, op TEXT NOT NULL, payload_json TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), PRIMARY KEY(namespace, seq), UNIQUE(event_id));\
                 INSERT INTO {events}(namespace, seq, event_id, entity_kind, entity_id, op, payload_json) VALUES ('legacy', 6, 'old', 'node', 'old', 'UPSERT', '{{\"original\": true }}');\
                 CREATE TABLE {cursors}(namespace TEXT NOT NULL DEFAULT 'default', consumer TEXT NOT NULL, last_seq BIGINT NOT NULL, updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), PRIMARY KEY(namespace, consumer));\
                 INSERT INTO {cursors}(namespace, consumer, last_seq) VALUES ('legacy', 'reader', 6);",
                schema = tables.quoted_schema,
                ns = tables.namespace_seq,
                events = tables.entity_events,
                cursors = tables.replay_cursors,
            ))
            .await
            .map_err(backend)?;
    drop(client);
    let row = store
        .append_raw_entity_event("legacy", raw("new", "{\"kept\": true }"))
        .await?;
    assert_eq!(row.event.seq, 7);
    assert_eq!(row.event.payload_json, "{\"kept\": true }");
    assert_eq!(store.replay_cursor("legacy", "reader").await?.last_seq, 6);
    let cleaner = store.client().await?;
    cleaner
        .batch_execute(&format!("DROP SCHEMA {} CASCADE", tables.quoted_schema))
        .await
        .map_err(backend)
}

#[tokio::test]
#[ignore = "requires KOGWISTAR_TEST_PG_DSN; run explicitly with --include-ignored"]
async fn auth_store_links_python_user_and_retries_identity_when_dsn_available()
-> PostgresStoreResult<()> {
    let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
        .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
    let schema = test_schema("auth_identity");
    let quoted_schema = quote_identifier(&schema)?;
    let base_config = dsn
        .parse::<Config>()
        .map_err(|error| PostgresStoreError::Backend(error.to_string()))?;
    let (client, connection) = base_config.connect(NoTls).await.map_err(backend)?;
    tokio::spawn(async move {
        let _ = connection.await;
    });
    client
        .batch_execute(&format!("CREATE SCHEMA {quoted_schema}"))
        .await
        .map_err(backend)?;
    let mut auth_config = base_config;
    auth_config.options(format!("-c search_path={schema}"));
    let store = PostgresAuthStore::from_config(auth_config)?;
    store.ensure_schema().await?;
    client
            .execute(
                &format!(
                    "INSERT INTO {quoted_schema}.users (user_id, email, display_name, is_active, global_role, global_ns, created_at) VALUES ('python-user', 'alice@example.com', 'Python Alice', TRUE, 'rw', 'docs,workflow', CURRENT_TIMESTAMP)"
                ),
                &[],
            )
            .await
            .map_err(backend)?;
    let request = ResolveExternalIdentity {
        issuer: "https://issuer.example".to_owned(),
        subject: "subject-1".to_owned(),
        email: "alice@example.com".to_owned(),
        display_name: Some("Ignored".to_owned()),
        new_user_id: "rust-user".to_owned(),
        default_role: "ro".to_owned(),
        default_ns: "docs".to_owned(),
    };
    let linked = store.resolve_external_identity(request.clone()).await?;
    assert_eq!(linked.user_id, "python-user");
    assert_eq!(linked.global_role.as_deref(), Some("rw"));
    let retried = store
        .resolve_external_identity(ResolveExternalIdentity {
            new_user_id: "must-not-win".to_owned(),
            email: "changed@example.com".to_owned(),
            ..request
        })
        .await?;
    assert_eq!(retried.user_id, "python-user");
    assert_eq!(
        store
            .external_identity("https://issuer.example", "subject-1")
            .await?
            .unwrap()
            .email
            .as_deref(),
        Some("alice@example.com")
    );
    client
        .batch_execute(&format!("DROP SCHEMA {quoted_schema} CASCADE"))
        .await
        .map_err(backend)?;
    Ok(())
}
