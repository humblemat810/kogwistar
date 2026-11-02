// Domain-focused tests; shared fixtures live in the parent test module.
use super::*;

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
async fn graph_projection_identity_is_scoped_and_legacy_rows_survive_schema_upgrade()
-> PostgresStoreResult<()> {
    let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
        .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../../../contracts/golden/graph-acl-v1.json"
    ))?;
    let schema = test_schema("graph_scope_identity");
    let store = PostgresStore::from_dsn(&dsn, &schema)?;
    store.ensure_schema().await?;
    store
        .create_graph_schema(3, GraphTableNames::default())
        .await
        .map_err(|error| PostgresStoreError::Backend(format!("initial schema: {error:?}")))?;

    let base_scope = GraphScope {
        namespace: "graph-type-isolation".to_owned(),
        workspace_id: None,
        graph_space: None,
    };
    let legacy_id = "legacy-row-before-scope-upgrade";
    store
        .upsert_graph_projection(GraphProjectionUpsert {
            scope: base_scope.clone(),
            table: "gke_nodes".to_owned(),
            record: GraphRecord {
                id: legacy_id.to_owned(),
                document: Some(serde_json::json!({"id":legacy_id,"label":"legacy"}).to_string()),
                metadata: Map::new(),
                embedding: None,
            },
            embedding_dim: 3,
        })
        .await
        .map_err(|error| PostgresStoreError::Backend(format!("legacy-row upsert: {error:?}")))?;

    let tables = Tables::new(&schema)?;
    let nodes = qualified(&tables.quoted_schema, "gke_nodes")?;
    let quoted_schema = &tables.quoted_schema;
    let client = store.client().await?;
    client
        .batch_execute(&format!(
            "ALTER TABLE {} DROP CONSTRAINT IF EXISTS gke_nodes_pkey; \
             ALTER TABLE {} ADD PRIMARY KEY (id); \
             DROP INDEX IF EXISTS {}.uq_graph_scope_identity_0; \
             DROP INDEX IF EXISTS {}.uq_graph_scope_identity_1; \
             DROP INDEX IF EXISTS {}.uq_graph_scope_identity_2; \
             DROP INDEX IF EXISTS {}.uq_graph_scope_identity_3; \
             DROP INDEX IF EXISTS {}.uq_graph_scope_identity_4; \
             DROP INDEX IF EXISTS {}.uq_graph_scope_identity_5; \
             DROP INDEX IF EXISTS {}.uq_graph_scope_identity_6; \
             DROP INDEX IF EXISTS {}.uq_graph_scope_identity_7",
            nodes,
            nodes,
            quoted_schema,
            quoted_schema,
            quoted_schema,
            quoted_schema,
            quoted_schema,
            quoted_schema,
            quoted_schema,
            quoted_schema,
        ))
        .await
        .map_err(|error| PostgresStoreError::Backend(format!("legacy PK setup: {error:?}")))?;
    drop(client);
    store
        .create_graph_schema(3, GraphTableNames::default())
        .await
        .map_err(|error| PostgresStoreError::Backend(format!("schema upgrade: {error:?}")))?;

    let isolation = &fixture["graph_type_isolation"];
    let entity_id = isolation["entity_id"].as_str().unwrap();
    let cases = isolation["cases"].as_array().unwrap();
    for case in cases {
        let truth_graph = case["truth_graph"].as_str().unwrap();
        let scope = GraphScope {
            graph_space: Some(truth_graph.to_owned()),
            ..base_scope.clone()
        };
        let document = serde_json::json!({"id":entity_id,"label":truth_graph});
        store
            .apply_graph_mutations_atomic(vec![GraphMutation {
                scope,
                table: "gke_nodes".to_owned(),
                entity_kind: "node".to_owned(),
                event_id: format!("graph-type-isolation-{truth_graph}"),
                op: "ADD".to_owned(),
                payload: serde_json::json!({"id":entity_id,"entity_type":"node"}),
                record: GraphRecord {
                    id: entity_id.to_owned(),
                    document: Some(document.to_string()),
                    metadata: Map::new(),
                    embedding: None,
                },
                embedding_dim: 3,
            }])
            .await
            .map_err(|error| {
                PostgresStoreError::Backend(format!("scoped mutation for {truth_graph}: {error:?}"))
            })?;
    }

    let legacy_rows = store
        .graph_projection_records(GraphProjectionRead {
            scope: base_scope.clone(),
            table: "gke_nodes".to_owned(),
            ids: Some(vec![legacy_id.to_owned()]),
            metadata: kogwistar_store::MetadataFilter::default(),
            limit: 10,
        })
        .await?;
    assert_eq!(legacy_rows.len(), 1);
    assert_eq!(
        legacy_rows[0].document.as_deref(),
        Some("{\"id\":\"legacy-row-before-scope-upgrade\",\"label\":\"legacy\"}")
    );

    for case in cases {
        let truth_graph = case["truth_graph"].as_str().unwrap();
        let scope = GraphScope {
            graph_space: Some(truth_graph.to_owned()),
            ..base_scope.clone()
        };
        let rows = store
            .graph_projection_records(GraphProjectionRead {
                scope,
                table: "gke_nodes".to_owned(),
                ids: Some(vec![entity_id.to_owned()]),
                metadata: kogwistar_store::MetadataFilter::default(),
                limit: 10,
            })
            .await?;
        assert_eq!(rows.len(), 1, "graph={truth_graph}");
        let document: Value = serde_json::from_str(rows[0].document.as_deref().unwrap())?;
        assert_eq!(document["label"], truth_graph);
    }

    drop(store);
    let reopened = PostgresStore::from_dsn(&dsn, &schema)?;
    reopened
        .client()
        .await?
        .batch_execute(&format!("DROP SCHEMA {} CASCADE", tables.quoted_schema))
        .await
        .map_err(backend)?;
    Ok(())
}
