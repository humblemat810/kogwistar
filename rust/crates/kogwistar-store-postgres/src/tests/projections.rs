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
