// Domain-focused tests; shared fixtures live in the parent test module.
use super::*;

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
async fn concurrent_graph_schema_bootstrap_serializes_vector_extension_creation()
-> PostgresStoreResult<()> {
    let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
        .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
    let first_schema = test_schema("concurrent_graph_bootstrap_a");
    let second_schema = test_schema("concurrent_graph_bootstrap_b");
    let first = PostgresStore::from_dsn(&dsn, &first_schema)?;
    let second = PostgresStore::from_dsn(&dsn, &second_schema)?;

    let (first_result, second_result) = tokio::join!(
        first.create_graph_schema(3, GraphTableNames::default()),
        second.create_graph_schema(3, GraphTableNames::default()),
    );
    first_result?;
    second_result?;

    let client = first.client().await?;
    client
        .batch_execute(&format!(
            "DROP SCHEMA \"{first_schema}\" CASCADE; DROP SCHEMA \"{second_schema}\" CASCADE;"
        ))
        .await
        .map_err(backend)?;
    Ok(())
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
        "../../../../../contracts/golden/python-graph-model-payloads-v1.json"
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
