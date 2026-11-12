// Native-path derived graph payload and recovery parity tests.

async fn read_workflow_run_parity_event(
    service: &PostgresRunApplicationService,
    entity_id: &str,
) -> kogwistar_store_postgres::RawEntityEvent {
    service
        .workflow_store
        .latest_entity_events("tenant-a", "node", &[entity_id.to_owned()])
        .await
        .unwrap()
        .remove(entity_id)
        .expect("native write must persist a canonical entity event")
}

async fn read_workflow_run_parity_projection(
    service: &PostgresRunApplicationService,
    entity_id: &str,
) -> Vec<GraphRecord> {
    service
        .workflow_store
        .graph_projection_records(GraphProjectionRead {
            scope: GraphScope {
                namespace: "tenant-a".to_owned(),
                workspace_id: None,
                graph_space: None,
            },
            table: "gke_nodes".to_owned(),
            ids: Some(vec![entity_id.to_owned()]),
            metadata: MetadataFilter::default(),
            limit: 1,
        })
        .await
        .unwrap()
}

#[tokio::test]
#[ignore = "requires KOGWISTAR_TEST_PG_DSN; uses disposable PostgreSQL schemas"]
async fn postgres_native_workflow_run_node_preserves_python_payload_and_digest_after_reopen() {
    use sha2::Digest;

    let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
        .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
    let base = format!("rust_workflow_run_parity_{}", uuid::Uuid::new_v4().simple());
    let schemas = [
        format!("{base}_knowledge"),
        format!("{base}_conversation"),
        format!("{base}_workflow"),
        format!("{base}_wisdom"),
    ];
    let mut service = PostgresRunApplicationService::from_dsn_with_graph_schemas(
        &dsn,
        &schemas[0],
        &schemas[1],
        &schemas[2],
        &schemas[3],
    )
    .unwrap()
    .with_graph_role("workflow")
    .unwrap();
    service.ensure_schema().await.unwrap();
    for store in [
        &service.store,
        &service.conversation_store,
        &service.workflow_store,
        &service.wisdom_store,
    ] {
        store
            .create_graph_schema(3, kogwistar_store_postgres::GraphTableNames::default())
            .await
            .unwrap();
    }

    let model_fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/workflow-run-node-model-v1.json"
    )))
    .unwrap();
    let payload = model_fixture["payload"].clone();
    let payload_digest = format!(
        "{:x}",
        sha2::Sha256::digest(kogwistar_contracts::canonical_json(&payload, false).as_bytes())
    );
    assert_eq!(payload_digest, model_fixture["canonical_json_sha256"]);

    let mut batch: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/graph-mutation-batch-v1.json"
    )))
    .unwrap();
    let mutation = &mut batch["mutations"][0];
    mutation["scope"] = json!({"namespace":"tenant-a"});
    mutation["event_id"] = json!(format!("{base}-event"));
    mutation["payload"] = payload.clone();
    mutation["record"] = json!({
        "id": payload["id"],
        "metadata": payload["metadata"],
        "embedding": payload["embedding"]
    });
    mutation["embedding_dim"] = json!(3);
    mutation["model_envelope"] = json!({
        "schema_version": 1,
        "model_class": "workflow_run_node",
        "payload": payload
    });

    let request = ApiEffectRequest {
        contract_version: 1,
        method: "POST".to_owned(),
        path_and_query: INTERNAL_GRAPH_MUTATIONS_ROUTE.to_owned(),
        body: serde_json::to_vec(&batch).unwrap(),
        principal: json!({
            "sub":"alice", "role":"rw", "ns":["tenant-a"],
            "storage_ns":"tenant-a", "capabilities":["write_graph"]
        }),
    };
    let response = ApplicationService::execute(&service, request).await;
    assert_eq!(
        response.status,
        StatusCode::OK.as_u16(),
        "{}",
        String::from_utf8_lossy(&response.body)
    );

    let entity_id = model_fixture["payload"]["id"].as_str().unwrap();
    let first_event = read_workflow_run_parity_event(&service, entity_id).await;
    let first_payload: Value = serde_json::from_str(&first_event.payload_json).unwrap();
    assert_eq!(first_payload["metadata"]["_class_name"], "WorkflowRunNode");
    let mut python_payload_view = first_payload.clone();
    python_payload_view["metadata"]
        .as_object_mut()
        .unwrap()
        .remove("_class_name");
    assert_eq!(python_payload_view, model_fixture["payload"]);
    assert_eq!(
        format!(
            "{:x}",
            sha2::Sha256::digest(
                kogwistar_contracts::canonical_json(&python_payload_view, false).as_bytes()
            )
        ),
        model_fixture["canonical_json_sha256"]
    );
    let first_projection = read_workflow_run_parity_projection(&service, entity_id).await;
    assert_eq!(first_projection.len(), 1);
    assert_eq!(first_projection[0].metadata["_class_name"], "WorkflowRunNode");

    drop(service);
    service = PostgresRunApplicationService::from_dsn_with_graph_schemas(
        &dsn,
        &schemas[0],
        &schemas[1],
        &schemas[2],
        &schemas[3],
    )
    .unwrap()
    .with_graph_role("workflow")
    .unwrap();
    service.ensure_schema().await.unwrap();

    let reopened_event = read_workflow_run_parity_event(&service, entity_id).await;
    assert_eq!(reopened_event.payload_json, first_event.payload_json);
    assert_eq!(
        format!(
            "{:x}",
            sha2::Sha256::digest(
                kogwistar_contracts::canonical_json(&python_payload_view, false).as_bytes()
            )
        ),
        model_fixture["canonical_json_sha256"]
    );
    let reopened_projection = read_workflow_run_parity_projection(&service, entity_id).await;
    assert_eq!(reopened_projection, first_projection);

    drop(service);
    let (cleanup, connection) = tokio_postgres::connect(&dsn, tokio_postgres::NoTls)
        .await
        .unwrap();
    let cleanup_task = tokio::spawn(connection);
    cleanup
        .batch_execute(
            &schemas
                .iter()
                .map(|schema| format!("DROP SCHEMA \"{schema}\" CASCADE"))
                .collect::<Vec<_>>()
                .join("; "),
        )
        .await
        .unwrap();
    drop(cleanup);
    cleanup_task.await.unwrap().unwrap();
}
