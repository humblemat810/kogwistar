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

fn document_vector_job() -> NewIndexJob {
    NewIndexJob {
        job_id: "document-vector-doc-1".to_owned(),
        namespace: "tenant-a".to_owned(),
        entity_kind: "document".to_owned(),
        entity_id: "doc-1".to_owned(),
        index_kind: "document_embedding".to_owned(),
        op: "UPSERT".to_owned(),
        payload_json: Some(r#"{"schema_version":1}"#.to_owned()),
        max_retries: 5,
    }
}

#[test]
fn source_document_and_vector_job_commit_and_reopen_together() {
    let (store, path) = store("source-document-vector-job");
    let scope = GraphScope {
        namespace: "tenant-a".to_owned(),
        workspace_id: Some("workspace-a".to_owned()),
        graph_space: Some("knowledge".to_owned()),
    };
    let mut write = source_document(scope.clone(), "doc-1", "OCR text");
    write.record.embedding = Some(vec![0.6, 0.8, 0.0]);
    write.embedding_dim = 3;

    store
        .upsert_source_document_projection_with_index_job(write, document_vector_job())
        .unwrap();
    let jobs = block_on(IndexJobReadStore::index_jobs(
        &store,
        Some("tenant-a"),
        Some("PENDING"),
        Some("document"),
        Some("doc-1"),
        Some("document_embedding"),
        10,
    ))
    .unwrap();
    assert_eq!(jobs.len(), 1);
    assert_eq!(jobs[0].job_id, "document-vector-doc-1");
    assert_eq!(
        jobs[0].payload_json.as_deref(),
        Some(r#"{"schema_version":1}"#)
    );

    drop(store);
    let reopened = SqliteStore::open(&path).unwrap();
    let record = reopened
        .source_document_projection(&scope, "doc-1")
        .unwrap()
        .unwrap();
    assert_eq!(record.embedding.as_deref(), Some(&[0.6, 0.8, 0.0][..]));
    let reopened_jobs = block_on(IndexJobReadStore::index_jobs(
        &reopened,
        Some("tenant-a"),
        Some("PENDING"),
        Some("document"),
        Some("doc-1"),
        Some("document_embedding"),
        10,
    ))
    .unwrap();
    assert_eq!(reopened_jobs.len(), 1);

    drop(reopened);
    fs::remove_file(path).unwrap();
}

#[test]
fn failed_document_vector_job_enqueue_rolls_back_source_projection() {
    let (store, path) = store("source-document-vector-job-rollback");
    let scope = GraphScope {
        namespace: "tenant-a".to_owned(),
        workspace_id: Some("workspace-a".to_owned()),
        graph_space: Some("knowledge".to_owned()),
    };
    let connection = rusqlite::Connection::open(&path).unwrap();
    connection
        .execute_batch(
            "CREATE TRIGGER reject_document_vector_job BEFORE INSERT ON index_jobs
             WHEN NEW.index_kind = 'document_embedding'
             BEGIN SELECT RAISE(ABORT, 'synthetic document index queue failure'); END;",
        )
        .unwrap();
    drop(connection);

    let result = store.upsert_source_document_projection_with_index_job(
        source_document(scope.clone(), "doc-1", "OCR text"),
        document_vector_job(),
    );
    assert!(result.is_err());
    assert!(
        store
            .source_document_projection(&scope, "doc-1")
            .unwrap()
            .is_none()
    );
    assert!(
        block_on(IndexJobReadStore::index_jobs(
            &store,
            Some("tenant-a"),
            None,
            Some("document"),
            Some("doc-1"),
            Some("document_embedding"),
            10,
        ))
        .unwrap()
        .is_empty()
    );

    drop(store);
    fs::remove_file(path).unwrap();
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
fn source_document_projection_delete_is_scoped_and_idempotent() {
    let (store, path) = store("source-document-delete");
    let scope = GraphScope {
        namespace: "tenant-a".to_owned(),
        workspace_id: Some("workspace-a".to_owned()),
        graph_space: Some("knowledge".to_owned()),
    };
    let other_scope = GraphScope {
        graph_space: Some("wisdom".to_owned()),
        ..scope.clone()
    };
    store
        .upsert_source_document_projection(source_document(
            scope.clone(),
            "doc-1",
            "knowledge text",
        ))
        .unwrap();
    store
        .upsert_source_document_projection(source_document(
            other_scope.clone(),
            "doc-1",
            "wisdom text",
        ))
        .unwrap();

    assert!(DocumentStore::supports_document_projection_delete(&store));
    assert!(
        !block_on(DocumentStore::delete_document_projection(
            &store,
            GraphScope {
                namespace: "tenant-b".to_owned(),
                ..scope.clone()
            },
            "doc-1".to_owned(),
        ))
        .unwrap()
    );
    assert!(
        block_on(DocumentStore::delete_document_projection(
            &store,
            scope.clone(),
            "doc-1".to_owned(),
        ))
        .unwrap()
    );
    assert!(
        !block_on(DocumentStore::delete_document_projection(
            &store,
            scope.clone(),
            "doc-1".to_owned(),
        ))
        .unwrap()
    );
    assert!(
        store
            .source_document_projection(&scope, "doc-1")
            .unwrap()
            .is_none()
    );
    assert_eq!(
        store
            .source_document_projection(&other_scope, "doc-1")
            .unwrap()
            .unwrap()
            .document
            .as_deref(),
        Some("wisdom text")
    );

    drop(store);
    fs::remove_file(path).unwrap();
}

#[test]
fn source_document_group_delete_removes_all_matching_chunks_within_scope() {
    let (store, path) = store("source-document-group-delete");
    let scope = GraphScope {
        namespace: "tenant-a".to_owned(),
        workspace_id: Some("workspace-a".to_owned()),
        graph_space: Some("knowledge".to_owned()),
    };
    let other_scope = GraphScope {
        graph_space: Some("wisdom".to_owned()),
        ..scope.clone()
    };
    for id in ["source-doc", "source-doc::chunk-1", "source-doc::chunk-2"] {
        let mut projection = source_document(scope.clone(), id, "chunk text");
        projection
            .record
            .metadata
            .insert("doc_id".to_owned(), serde_json::json!("source-doc"));
        store.upsert_source_document_projection(projection).unwrap();
    }
    store
        .upsert_source_document_projection(source_document(
            scope.clone(),
            "other-source::chunk-1",
            "unrelated text",
        ))
        .unwrap();
    let mut other_scope_row = source_document(
        other_scope.clone(),
        "other-scope::chunk-1",
        "other scope text",
    );
    other_scope_row
        .record
        .metadata
        .insert("doc_id".to_owned(), serde_json::json!("source-doc"));
    store
        .upsert_source_document_projection(other_scope_row)
        .unwrap();

    let deleted = block_on(DocumentStore::delete_document_projections_by_source_id(
        &store,
        scope.clone(),
        "source-doc".to_owned(),
    ))
    .unwrap();
    assert_eq!(
        deleted,
        ["source-doc", "source-doc::chunk-1", "source-doc::chunk-2"]
    );
    assert!(
        block_on(DocumentStore::delete_document_projections_by_source_id(
            &store,
            scope.clone(),
            "source-doc".to_owned(),
        ))
        .unwrap()
        .is_empty()
    );
    assert!(
        store
            .source_document_projection(&scope, "source-doc::chunk-1")
            .unwrap()
            .is_none()
    );
    assert!(
        store
            .source_document_projection(&scope, "other-source::chunk-1")
            .unwrap()
            .is_some()
    );
    assert!(
        store
            .source_document_projection(&other_scope, "other-scope::chunk-1")
            .unwrap()
            .is_some()
    );

    drop(store);
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
