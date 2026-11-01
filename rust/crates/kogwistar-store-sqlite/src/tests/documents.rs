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
