use super::*;

fn source_document(scope: GraphScope, id: &str, text: &str) -> DocumentProjectionUpsert {
    DocumentProjectionUpsert {
        scope,
        record: GraphRecord {
            id: id.to_owned(),
            document: Some(text.to_owned()),
            metadata: serde_json::json!({"doc_type": "text"})
                .as_object()
                .unwrap()
                .clone(),
            embedding: None,
        },
        embedding_dim: 0,
    }
}

#[test]
fn source_documents_round_trip_and_remain_storage_scope_isolated() {
    let (store, path) = store("source-document-scope");
    let base_scope = GraphScope {
        namespace: "tenant-a".to_owned(),
        workspace_id: Some("workspace-a".to_owned()),
        graph_space: Some("knowledge".to_owned()),
    };
    store
        .upsert_source_document_projection(source_document(
            base_scope.clone(),
            "doc-1",
            "authoritative text",
        ))
        .unwrap();

    let record = store
        .source_document_projection(&base_scope, "doc-1")
        .unwrap()
        .unwrap();
    assert_eq!(record.id, "doc-1");
    assert_eq!(record.document.as_deref(), Some("authoritative text"));
    assert_eq!(record.metadata["doc_type"], "text");
    assert!(record.embedding.is_none());

    let mut other_scope = base_scope.clone();
    other_scope.graph_space = Some("wisdom".to_owned());
    assert!(
        store
            .source_document_projection(&other_scope, "doc-1")
            .unwrap()
            .is_none()
    );

    drop(store);
    let reopened = SqliteStore::open(&path).unwrap();
    assert_eq!(
        reopened
            .source_document_projection(&base_scope, "doc-1")
            .unwrap()
            .unwrap()
            .document
            .as_deref(),
        Some("authoritative text")
    );
    drop(reopened);
    fs::remove_file(path).unwrap();
}

#[test]
fn source_document_write_participates_in_the_sqlite_unit_of_work() {
    let (store, path) = store("source-document-uow");
    let scope = GraphScope {
        namespace: "tenant-a".to_owned(),
        workspace_id: None,
        graph_space: None,
    };
    let result: SqliteStoreResult<()> = store.immediate_transaction(|uow| {
        uow.upsert_source_document_projection(source_document(
            scope.clone(),
            "doc-rollback",
            "must roll back",
        ))?;
        Err(SqliteStoreError::TransactionAborted(
            "injected graph batch failure".to_owned(),
        ))
    });
    assert!(result.is_err());
    assert!(
        store
            .source_document_projection(&scope, "doc-rollback")
            .unwrap()
            .is_none()
    );

    drop(store);
    fs::remove_file(path).unwrap();
}

#[test]
fn document_projection_lookup_is_scoped_and_preserves_requested_order() {
    let (store, path) = store("source-document-lookup");
    let scope = GraphScope {
        namespace: "tenant-a".to_owned(),
        workspace_id: Some("workspace-a".to_owned()),
        graph_space: Some("knowledge".to_owned()),
    };
    for id in ["doc-b", "doc-a"] {
        store
            .upsert_source_document_projection(source_document(
                scope.clone(),
                id,
                "authoritative text",
            ))
            .unwrap();
    }

    assert!(DocumentStore::supports_document_projection_lookup(&store));
    let found = block_on(DocumentStore::document_projection_ids(
        &store,
        scope.clone(),
        vec!["doc-a".to_owned(), "missing".to_owned(), "doc-b".to_owned()],
    ))
    .unwrap();
    assert_eq!(found, ["doc-a", "doc-b"]);

    let other_scope = GraphScope {
        namespace: "tenant-b".to_owned(),
        ..scope
    };
    assert!(
        block_on(DocumentStore::document_projection_ids(
            &store,
            other_scope,
            vec!["doc-a".to_owned()],
        ))
        .unwrap()
        .is_empty()
    );

    drop(store);
    fs::remove_file(path).unwrap();
}
