    #[tokio::test]
    #[ignore = "requires KOGWISTAR_TEST_PG_DSN; run explicitly with --include-ignored"]
    async fn postgres_internal_graph_mutation_route_persists_atomically_when_dsn_available() {
        let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
            .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
        let schema = format!("rust_api_graph_mutation_{}", uuid::Uuid::new_v4().simple());
        let embedding_failure = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (mut client, connection) = tokio_postgres::connect(&dsn, tokio_postgres::NoTls)
            .await
            .unwrap();
        tokio::spawn(async move {
            let _ = connection.await;
        });
        let transaction = client.transaction().await.unwrap();
        transaction
            .query_one(
                "SELECT pg_advisory_xact_lock(hashtext('kogwistar.pgvector.extension'))",
                &[],
            )
            .await
            .unwrap();
        transaction
            .batch_execute(&format!(
                r#"CREATE EXTENSION IF NOT EXISTS vector;
                CREATE SCHEMA "{schema}";
                CREATE TABLE "{schema}"."gke_documents" (
                    id TEXT PRIMARY KEY,
                    document TEXT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding vector(3) NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE TABLE "{schema}"."gke_nodes" (
                    id TEXT PRIMARY KEY,
                    document TEXT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding vector(3) NULL,
                    CHECK (id <> 'public-rollback-node'),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE TABLE "{schema}"."gke_edges" (
                    id TEXT PRIMARY KEY,
                    document TEXT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding vector(3) NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE TABLE "{schema}"."gke_edge_endpoints" (
                    id TEXT PRIMARY KEY,
                    document TEXT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );"#
            ))
            .await
            .unwrap();
        transaction.commit().await.unwrap();
        let embedding_calls = Arc::new(Mutex::new(Vec::new()));
        let embedding_profile = kogwistar_engine::embedding::EmbeddingProfile {
            provider: "test".to_owned(),
            model: "fixed-v1".to_owned(),
            dimension: 3,
            similarity_metric: "cosine".to_owned(),
            endpoint_fingerprint: None,
            max_sequence_length: None,
            crop_token_budget: None,
            tokenizer_fingerprint: None,
            crop_policy: None,
            embedding_kind: Some("single_vector".to_owned()),
            model_revision: None,
            preprocessing_fingerprint: None,
            max_image_patches: None,
        };
        let service = PostgresRunApplicationService::from_dsn(&dsn, &schema)
            .unwrap()
            .with_embedding_provider(Arc::new(FixedEmbeddingProvider {
                profile: embedding_profile.clone(),
                vector: vec![0.0, 0.0, 1.0],
                calls: embedding_calls.clone(),
                fail: Some(embedding_failure.clone()),
            }));
        service.ensure_schema().await.unwrap();
        service
            .store
            .create_graph_schema(3, kogwistar_store_postgres::GraphTableNames::default())
            .await
            .unwrap();
        let embedding_storage_scope = service.embedding_storage_scope.as_ref().unwrap();
        let registered_profile = service
            .store
            .get_named_projection(EMBEDDING_PROFILE_NAMESPACE, embedding_storage_scope)
            .await
            .unwrap()
            .expect("server startup binds the configured profile");
        validate_embedding_profile_projection(
            &registered_profile,
            &embedding_profile,
            embedding_storage_scope,
        )
        .unwrap();
        let mut different_profile = embedding_profile.clone();
        different_profile.model = "other-model-same-dimension".to_owned();
        let mismatched_service = service
            .clone()
            .with_embedding_provider(Arc::new(FixedEmbeddingProvider {
                profile: different_profile,
                vector: vec![0.0, 0.0, 1.0],
                calls: Arc::new(Mutex::new(Vec::new())),
                fail: None,
            }));
        assert!(mismatched_service.ensure_schema().await.is_err());
        let legacy_schema = format!("{schema}_legacy");
        client
            .batch_execute(&format!(
                r#"CREATE SCHEMA "{legacy_schema}";
                CREATE TABLE "{legacy_schema}"."gke_nodes" (
                    id TEXT PRIMARY KEY,
                    document TEXT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding vector(3) NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                INSERT INTO "{legacy_schema}"."gke_nodes" (id, document)
                    VALUES ('legacy-row', 'existing vector store row');"#
            ))
            .await
            .unwrap();
        let legacy_service = PostgresRunApplicationService::from_dsn(&dsn, &legacy_schema)
            .unwrap()
            .with_embedding_provider(Arc::new(FixedEmbeddingProvider {
                profile: embedding_profile.clone(),
                vector: vec![0.0, 0.0, 1.0],
                calls: Arc::new(Mutex::new(Vec::new())),
                fail: None,
            }));
        let legacy_error = legacy_service.ensure_schema().await.unwrap_err();
        assert!(legacy_error.contains("unbound for non-empty PostgreSQL vector storage"));
        assert!(legacy_service
            .store
            .get_named_projection(
                EMBEDDING_PROFILE_NAMESPACE,
                legacy_service.embedding_storage_scope.as_ref().unwrap(),
            )
            .await
            .unwrap()
            .is_none());
        let mut body: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/graph-mutation-batch-v1.json"
        )))
        .unwrap();
        body["mutations"][0]["scope"]["namespace"] = json!("alpha");
        body["mutations"][0]["record"]["document"] = json!("{\"label\":\"Alpha source\"}");
        body["mutations"][0]["record"]["embedding"] = json!([1.0, 0.0, 0.0]);
        body["mutations"][0]["embedding_dim"] = json!(3);
        let mut second_node = body["mutations"][0].clone();
        second_node["event_id"] = json!("graph-route-event-node-2");
        second_node["payload"]["id"] = json!("node-2");
        second_node["record"]["id"] = json!("node-2");
        second_node["record"]["document"] = json!(r#"{"label":"Beta target"}"#);

        let mut edge = body["mutations"][0].clone();
        edge["table"] = json!("gke_edges");
        edge["entity_kind"] = json!("edge");
        edge["event_id"] = json!("graph-route-event-edge-1");
        edge["payload"] = json!({"id": "edge-1"});
        edge["record"]["id"] = json!("edge-1");
        edge["record"]["document"] = json!(
            r#"{"source_ids":["node-1"],"target_ids":["node-2"],"relation":"supports"}"#
        );
        body["mutations"].as_array_mut().unwrap().push(second_node);
        body["mutations"].as_array_mut().unwrap().push(edge);
        for (target_grain, target_id) in [
            ("node", "node-1"),
            ("node", "node-2"),
            ("edge", "edge-1"),
        ] {
            let acl_id = format!("acl-{target_grain}-{target_id}");
            let mut acl_record = body["mutations"][0].clone();
            acl_record["event_id"] = json!(format!("graph-route-event-{acl_id}"));
            acl_record["payload"]["id"] = json!(acl_id);
            acl_record["record"]["id"] = json!(acl_id);
            acl_record["record"]["metadata"] = json!({
                "entity_type": "acl_record",
                "acl_truth_graph": "knowledge",
                "acl_target_grain": target_grain,
                "acl_target_item_id": "",
                "acl_target_entity_id": target_id,
                "acl_version": 1,
                "acl_mode": "private",
                "owner_id": "alice",
                "tombstoned": false
            });
            acl_record["record"]["document"] = Value::Null;
            acl_record["record"]["embedding"] = Value::Null;
            body["mutations"].as_array_mut().unwrap().push(acl_record);
        }
        body["index_jobs"] = json!([{
            "job_id": "graph-route-index-1",
            "namespace": "alpha",
            "entity_kind": "node",
            "entity_id": "node-1",
            "index_kind": "node_embedding",
            "op": "UPSERT",
            "payload_json": "{\"embedding\":[1.0,0.0,0.0]}",
            "max_retries": 3
        }]);
        let key = "graph-route-test-secret";
        let make_token_for = |subject: &str, capabilities: &[&str], storage_ns: &str| {
            encode(
                &Header::new(Algorithm::HS256),
                &json!({
                    "sub": subject,
                    "role": "rw",
                    "roles": ["reader", "rw"],
                    "ns": "workflow",
                    "storage_ns": storage_ns,
                    "capabilities": capabilities,
                    "exp": 4102444800_u64
                }),
                &EncodingKey::from_secret(key.as_bytes()),
            )
            .unwrap()
        };
        let make_token = |capabilities: &[&str], storage_ns: &str| {
            make_token_for("alice", capabilities, storage_ns)
        };
        let make_docs_token_for = |subject: &str, capabilities: &[&str]| {
            encode(
                &Header::new(Algorithm::HS256),
                &json!({
                    "sub": subject,
                    "role": "rw",
                    "roles": ["reader", "rw"],
                    "ns": "docs",
                    "storage_ns": "alpha",
                    "capabilities": capabilities,
                    "exp": 4102444800_u64
                }),
                &EncodingKey::from_secret(key.as_bytes()),
            )
            .unwrap()
        };
        let make_docs_token = |capabilities: &[&str]| make_docs_token_for("alice", capabilities);
        let make_app = || {
            router_with_application(
                ApiState {
                    health: snapshot(),
                    required_roles: vec!["reader".to_owned()],
                    implementation: ImplementationSnapshot::default(),
                    auth: AuthConfig {
                        mode: "jwt".to_owned(),
                        algorithm: Some("HS256".to_owned()),
                        key: Some(key.to_owned()),
                        ..AuthConfig::default()
                    },
                },
                Arc::new(service.clone()),
            )
        };
        let graph_upsert_token_for = |subject: &str| {
            encode(
                &Header::new(Algorithm::HS256),
                &json!({
                "sub": subject,
                "role": "rw",
                "roles": ["reader", "rw"],
                "ns": "workflow",
                "storage_ns": "alpha",
                "workspace": "workspace-a",
                "graph_space": "facts",
                "capabilities": ["write_graph"],
                "exp": 4102444800_u64
            }),
                &EncodingKey::from_secret(key.as_bytes()),
            )
            .unwrap()
        };
        let graph_upsert_token = graph_upsert_token_for("alice");
        let mut extraction: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/graph-extraction-v1.json"
        )))
        .unwrap();
        extraction["doc_id"] = json!("public-graph-upsert-doc");
        extraction["nodes"][0]["references"][0]["doc_id"] =
            json!("public-graph-upsert-doc");
        extraction["nodes"][1]["mentions"][0]["spans"][0]["doc_id"] =
            json!("public-graph-upsert-doc");
        extraction["edges"][0]["references"][0]["doc_id"] =
            json!("public-graph-upsert-doc");
        let denied_upsert = make_app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/graph/upsert")
                    .header("content-type", "application/json")
                    .header(
                        "authorization",
                        format!("Bearer {}", make_token(&[], "alpha")),
                    )
                    .body(Body::from(serde_json::to_vec(&extraction).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(denied_upsert.status(), StatusCode::FORBIDDEN);
        let denied_document_count: i64 = client
            .query_one(
                &format!("SELECT count(*) FROM \"{schema}\".\"gke_documents\" WHERE id=$1"),
                &[&"public-graph-upsert-doc"],
            )
            .await
            .unwrap()
            .get(0);
        assert_eq!(denied_document_count, 0);

        let mut rollback_extraction = extraction.clone();
        rollback_extraction["doc_id"] = json!("public-graph-upsert-rollback-doc");
        rollback_extraction["nodes"][0]["local_id"] = json!("public-rollback-node");
        rollback_extraction["nodes"][0]["id"] = json!("public-rollback-node");
        rollback_extraction["edges"][0]["source_ids"][0] = json!("public-rollback-node");
        rollback_extraction["nodes"][0]["references"][0]["doc_id"] =
            json!("public-graph-upsert-rollback-doc");
        rollback_extraction["nodes"][1]["mentions"][0]["spans"][0]["doc_id"] =
            json!("public-graph-upsert-rollback-doc");
        rollback_extraction["edges"][0]["references"][0]["doc_id"] =
            json!("public-graph-upsert-rollback-doc");
        let rollback_upsert = make_app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/graph/upsert")
                    .header("content-type", "application/json")
                    .header(
                        "authorization",
                        format!("Bearer {graph_upsert_token}"),
                    )
                    .body(Body::from(
                        serde_json::to_vec(&rollback_extraction).unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let rollback_status = rollback_upsert.status();
        let rollback_body = to_bytes(rollback_upsert.into_body(), 65_536)
            .await
            .unwrap();
        assert_eq!(
            rollback_status,
            StatusCode::CONFLICT,
            "{}",
            String::from_utf8_lossy(&rollback_body)
        );
        let rolled_back_document_count: i64 = client
            .query_one(
                &format!("SELECT count(*) FROM \"{schema}\".\"gke_documents\" WHERE id=$1"),
                &[&"public-graph-upsert-rollback-doc"],
            )
            .await
            .unwrap()
            .get(0);
        assert_eq!(rolled_back_document_count, 0);

        let upsert = make_app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/graph/upsert")
                    .header("content-type", "application/json")
                    .header("authorization", format!("Bearer {graph_upsert_token}"))
                    .body(Body::from(serde_json::to_vec(&extraction).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(upsert.status(), StatusCode::OK);
        let upsert_body: Value =
            serde_json::from_slice(&to_bytes(upsert.into_body(), 65_536).await.unwrap()).unwrap();
        assert_eq!(upsert_body["document_id"], "public-graph-upsert-doc");
        let node_ids = upsert_body["node_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap().to_owned())
            .collect::<Vec<_>>();
        let edge_ids = upsert_body["edge_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap().to_owned())
            .collect::<Vec<_>>();
        assert_eq!((node_ids.len(), edge_ids.len()), (2, 1));
        assert_eq!(upsert_body["nodes_added"], 2);
        assert_eq!(upsert_body["edges_added"], 1);
        let stored_document: String = client
            .query_one(
                &format!("SELECT document FROM \"{schema}\".\"gke_documents\" WHERE id=$1"),
                &[&"public-graph-upsert-doc"],
            )
            .await
            .unwrap()
            .get(0);
        assert_eq!(stored_document, extraction["content"].as_str().unwrap());
        let stored_nodes: i64 = client
            .query_one(
                &format!("SELECT count(*) FROM \"{schema}\".\"gke_nodes\" WHERE id=ANY($1)"),
                &[&node_ids],
            )
            .await
            .unwrap()
            .get(0);
        let stored_edges: i64 = client
            .query_one(
                &format!("SELECT count(*) FROM \"{schema}\".\"gke_edges\" WHERE id=ANY($1)"),
                &[&edge_ids],
            )
            .await
            .unwrap()
            .get(0);
        assert_eq!((stored_nodes, stored_edges), (2, 1));
        let stored_jobs: i64 = client
            .query_one(
                &format!("SELECT count(*) FROM \"{schema}\".\"index_jobs\" WHERE entity_id=ANY($1) AND index_kind='node_embedding'"),
                &[&node_ids.iter().chain(edge_ids.iter()).cloned().collect::<Vec<_>>()],
            )
            .await
            .unwrap()
            .get(0);
        assert_eq!(stored_jobs, 3);

        let mut existing_document_extraction = extraction.clone();
        existing_document_extraction
            .as_object_mut()
            .unwrap()
            .remove("content");
        let existing_document_upsert = make_app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/graph/upsert")
                    .header("content-type", "application/json")
                    .header(
                        "authorization",
                        format!("Bearer {graph_upsert_token}"),
                    )
                    .body(Body::from(
                        serde_json::to_vec(&existing_document_extraction).unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(existing_document_upsert.status(), StatusCode::OK);
        let existing_document_body: Value = serde_json::from_slice(
            &to_bytes(existing_document_upsert.into_body(), 65_536)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(existing_document_body["document_id"], "public-graph-upsert-doc");
        assert_eq!(existing_document_body["nodes_added"], 2);
        assert_eq!(existing_document_body["edges_added"], 1);
        let unchanged_document: String = client
            .query_one(
                &format!("SELECT document FROM \"{schema}\".\"gke_documents\" WHERE id=$1"),
                &[&"public-graph-upsert-doc"],
            )
            .await
            .unwrap()
            .get(0);
        assert_eq!(unchanged_document, extraction["content"].as_str().unwrap());

        let mut external_endpoint_extraction = extraction.clone();
        external_endpoint_extraction
            .as_object_mut()
            .unwrap()
            .remove("content");
        external_endpoint_extraction["nodes"] = json!([]);
        let mut external_edge = extraction["edges"][0].clone();
        external_edge["id"] = json!("ne:external-node-edge");
        external_edge["source_ids"] = json!([existing_document_body["node_ids"][0]]);
        external_edge["target_ids"] = json!([existing_document_body["node_ids"][1]]);
        external_endpoint_extraction["edges"] = json!([external_edge]);
        let external_endpoint_upsert = make_app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/graph/upsert")
                    .header("content-type", "application/json")
                    .header(
                        "authorization",
                        format!("Bearer {graph_upsert_token}"),
                    )
                    .body(Body::from(
                        serde_json::to_vec(&external_endpoint_extraction).unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let external_endpoint_status = external_endpoint_upsert.status();
        let external_endpoint_bytes = to_bytes(external_endpoint_upsert.into_body(), 65_536)
            .await
            .unwrap();
        assert_eq!(
            external_endpoint_status,
            StatusCode::OK,
            "{}",
            String::from_utf8_lossy(&external_endpoint_bytes)
        );
        let external_endpoint_body: Value =
            serde_json::from_slice(&external_endpoint_bytes).unwrap();
        assert_eq!(external_endpoint_body["nodes_added"], 0);
        assert_eq!(external_endpoint_body["edges_added"], 1);

        let mut external_edge_endpoint_extraction = extraction.clone();
        external_edge_endpoint_extraction
            .as_object_mut()
            .unwrap()
            .remove("content");
        external_edge_endpoint_extraction["nodes"] = json!([]);
        let mut nested_external_edge = extraction["edges"][0].clone();
        nested_external_edge["id"] = json!("ne:external-edge-reference");
        nested_external_edge["source_ids"] = json!([]);
        nested_external_edge["target_ids"] =
            json!([existing_document_body["node_ids"][1]]);
        nested_external_edge["source_edge_ids"] =
            json!([existing_document_body["edge_ids"][0]]);
        nested_external_edge["target_edge_ids"] = json!([]);
        external_edge_endpoint_extraction["edges"] = json!([nested_external_edge]);
        let external_edge_endpoint_upsert = make_app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/graph/upsert")
                    .header("content-type", "application/json")
                    .header(
                        "authorization",
                        format!("Bearer {graph_upsert_token}"),
                    )
                    .body(Body::from(
                        serde_json::to_vec(&external_edge_endpoint_extraction).unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(external_edge_endpoint_upsert.status(), StatusCode::OK);
        let external_edge_endpoint_body: Value = serde_json::from_slice(
            &to_bytes(external_edge_endpoint_upsert.into_body(), 65_536)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(external_edge_endpoint_body["nodes_added"], 0);
        assert_eq!(external_edge_endpoint_body["edges_added"], 1);
        let events_before_hidden_endpoint = service
            .store
            .replay_raw_events("alpha", 0, 100)
            .await
            .unwrap()
            .len();
        let hidden_endpoint_upsert = make_app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/graph/upsert")
                    .header("content-type", "application/json")
                    .header(
                        "authorization",
                        format!("Bearer {}", graph_upsert_token_for("bob")),
                    )
                    .body(Body::from(
                        serde_json::to_vec(&external_endpoint_extraction).unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(hidden_endpoint_upsert.status(), StatusCode::CONFLICT);
        assert_eq!(
            service
                .store
                .replay_raw_events("alpha", 0, 100)
                .await
                .unwrap()
                .len(),
            events_before_hidden_endpoint
        );
        let hidden_edge_endpoint_upsert = make_app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/graph/upsert")
                    .header("content-type", "application/json")
                    .header(
                        "authorization",
                        format!("Bearer {}", graph_upsert_token_for("bob")),
                    )
                    .body(Body::from(
                        serde_json::to_vec(&external_edge_endpoint_extraction).unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(hidden_edge_endpoint_upsert.status(), StatusCode::CONFLICT);
        assert_eq!(
            service
                .store
                .replay_raw_events("alpha", 0, 100)
                .await
                .unwrap()
                .len(),
            events_before_hidden_endpoint
        );

        let mcp_list = make_app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header(
                        "authorization",
                        format!("Bearer {}", make_docs_token(&["read_graph"])),
                    )
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":"list","method":"tools/list"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(mcp_list.status(), StatusCode::OK);
        let mcp_list_body = to_bytes(mcp_list.into_body(), 65_536).await.unwrap();
        let mcp_list_value: Value = serde_json::from_slice(&mcp_list_body).unwrap();
        assert!(
            mcp_list_value["result"]["tools"]
                .as_array()
                .unwrap()
                .iter()
                .any(|tool| tool["name"] == "kg_neighbors")
        );
        assert!(
            mcp_list_value["result"]["tools"]
                .as_array()
                .unwrap()
                .iter()
                .any(|tool| tool["name"] == "kg_find_edges")
        );

        let mcp_missing_capability = make_app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header(
                        "authorization",
                        format!("Bearer {}", make_docs_token(&[])),
                    )
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":"denied","method":"tools/call","params":{"name":"kg_neighbors","arguments":{"rid":"node-1"}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(mcp_missing_capability.status(), StatusCode::OK);
        let denied_body = to_bytes(mcp_missing_capability.into_body(), 65_536)
            .await
            .unwrap();
        let denied_value: Value = serde_json::from_slice(&denied_body).unwrap();
        assert_eq!(denied_value["error"]["code"], -32602);
        assert_eq!(
            denied_value["error"]["message"],
            "Tool not advertised by this application service"
        );

        // Capability admission precedes tool argument parsing and graph reads.
        let mcp_malformed_arguments = make_app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header(
                        "authorization",
                        format!("Bearer {}", make_docs_token(&[])),
                    )
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":"denied-malformed","method":"tools/call","params":{"name":"kg_neighbors","arguments":null}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(mcp_malformed_arguments.status(), StatusCode::OK);
        let denied_body = to_bytes(mcp_malformed_arguments.into_body(), 65_536)
            .await
            .unwrap();
        let denied_value: Value = serde_json::from_slice(&denied_body).unwrap();
        assert_eq!(denied_value["error"]["code"], -32602);
        assert_eq!(
            denied_value["error"]["message"],
            "Tool not advertised by this application service"
        );

        let find_edges_missing_capability = make_app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header(
                        "authorization",
                        format!("Bearer {}", make_docs_token(&[])),
                    )
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":"find-edges-denied","method":"tools/call","params":{"name":"kg_find_edges","arguments":{}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(find_edges_missing_capability.status(), StatusCode::OK);
        let denied_body = to_bytes(find_edges_missing_capability.into_body(), 65_536)
            .await
            .unwrap();
        let denied_value: Value = serde_json::from_slice(&denied_body).unwrap();
        assert_eq!(denied_value["error"]["code"], -32602);
        assert_eq!(
            denied_value["error"]["message"],
            "Tool not advertised by this application service"
        );

        let send_mutation = |app: Router, token: String, payload: Value| async move {
            app.oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/graph/mutations")
                    .header("content-type", "application/json")
                    .header("authorization", format!("Bearer {token}"))
                    .body(Body::from(serde_json::to_vec(&payload).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap()
        };

        let mut malformed_node_batch = body.clone();
        malformed_node_batch["mutations"] = json!([body["mutations"][0].clone()]);
        let checkpoint_fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/workflow-checkpoint-node-model-v1.json"
        )))
        .unwrap();
        let mut invalid_checkpoint = checkpoint_fixture["payload"].clone();
        invalid_checkpoint["id"] = json!("invalid-workflow-checkpoint-node");
        invalid_checkpoint["metadata"]["step_seq"] = json!(-1);
        malformed_node_batch["mutations"][0]["event_id"] = json!("invalid-node-envelope-event");
        malformed_node_batch["mutations"][0]["payload"] = invalid_checkpoint.clone();
        malformed_node_batch["mutations"][0]["record"]["id"] =
            json!("invalid-workflow-checkpoint-node");
        malformed_node_batch["mutations"][0]["record"]["embedding"] = Value::Null;
        malformed_node_batch["mutations"][0]["model_envelope"] = json!({
            "schema_version": 1,
            "model_class": "workflow_checkpoint_node",
            "payload": invalid_checkpoint,
        });
        malformed_node_batch["index_jobs"] = json!([{
            "job_id": "invalid-node-envelope-index",
            "namespace": "alpha",
            "entity_kind": "node",
            "entity_id": "invalid-workflow-checkpoint-node",
            "index_kind": "node_embedding",
            "op": "UPSERT",
            "payload_json": "{}",
            "max_retries": 3
        }]);
        let malformed_response = send_mutation(
            make_app(),
            graph_upsert_token.clone(),
            malformed_node_batch,
        )
        .await;
        assert_eq!(malformed_response.status(), StatusCode::UNPROCESSABLE_ENTITY);
        assert!(embedding_calls.lock().unwrap().is_empty());
        assert!(service
            .store
            .latest_entity_events(
                "alpha",
                "node",
                &["invalid-workflow-checkpoint-node".to_owned()],
            )
            .await
            .unwrap()
            .is_empty());
        assert!(service
            .store
            .list_index_jobs(
                Some("alpha"),
                None,
                Some("node"),
                Some("invalid-workflow-checkpoint-node"),
                None,
                10,
            )
            .await
            .unwrap()
            .is_empty());
        let invalid_projection_count: i64 = client
            .query_one(
                &format!("SELECT count(*) FROM \"{schema}\".\"gke_nodes\" WHERE id=$1"),
                &[&"invalid-workflow-checkpoint-node"],
            )
            .await
            .unwrap()
            .get(0);
        assert_eq!(invalid_projection_count, 0);

        let mut malformed_edge_batch = body.clone();
        let mut invalid_edge = serde_json::from_str::<Value>(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/workflow-runtime-edge-model-v1.json"
        )))
        .unwrap()["payload"]
            .clone();
        invalid_edge["id"] = json!("invalid-workflow-runtime-edge");
        invalid_edge["metadata"]["relation"] = json!(42);
        malformed_edge_batch["mutations"] = json!([body["mutations"][0].clone()]);
        malformed_edge_batch["mutations"][0]["table"] = json!("gke_edges");
        malformed_edge_batch["mutations"][0]["entity_kind"] = json!("edge");
        malformed_edge_batch["mutations"][0]["event_id"] =
            json!("invalid-edge-envelope-event");
        malformed_edge_batch["mutations"][0]["payload"] = invalid_edge.clone();
        malformed_edge_batch["mutations"][0]["record"]["id"] =
            json!("invalid-workflow-runtime-edge");
        malformed_edge_batch["mutations"][0]["record"]["document"] =
            json!(r#"{"relation":"supports","source_ids":["parity-node"],"target_ids":["external:target"]}"#);
        malformed_edge_batch["mutations"][0]["record"]["embedding"] = Value::Null;
        malformed_edge_batch["mutations"][0]["model_envelope"] = json!({
            "schema_version": 1,
            "model_class": "workflow_runtime_edge",
            "payload": invalid_edge,
        });
        malformed_edge_batch["index_jobs"] = json!([{
            "job_id": "invalid-edge-envelope-index",
            "namespace": "alpha",
            "entity_kind": "edge",
            "entity_id": "invalid-workflow-runtime-edge",
            "index_kind": "edge_embedding",
            "op": "UPSERT",
            "payload_json": "{}",
            "max_retries": 3
        }]);
        let malformed_edge_response = send_mutation(
            make_app(),
            graph_upsert_token.clone(),
            malformed_edge_batch,
        )
        .await;
        assert_eq!(
            malformed_edge_response.status(),
            StatusCode::UNPROCESSABLE_ENTITY
        );
        assert!(embedding_calls.lock().unwrap().is_empty());
        assert!(service
            .store
            .latest_entity_events(
                "alpha",
                "edge",
                &["invalid-workflow-runtime-edge".to_owned()],
            )
            .await
            .unwrap()
            .is_empty());
        assert!(service
            .store
            .list_index_jobs(
                Some("alpha"),
                None,
                Some("edge"),
                Some("invalid-workflow-runtime-edge"),
                None,
                10,
            )
            .await
            .unwrap()
            .is_empty());
        let invalid_edge_projection_count: i64 = client
            .query_one(
                &format!("SELECT count(*) FROM \"{schema}\".\"gke_edges\" WHERE id=$1"),
                &[&"invalid-workflow-runtime-edge"],
            )
            .await
            .unwrap()
            .get(0);
        assert_eq!(invalid_edge_projection_count, 0);

        let missing_capability =
            send_mutation(make_app(), make_token(&[], "alpha"), body.clone()).await;
        assert_eq!(missing_capability.status(), StatusCode::FORBIDDEN);

        let mut cross_scope_body = body.clone();
        for mutation in cross_scope_body["mutations"].as_array_mut().unwrap() {
            mutation["scope"]["namespace"] = json!("beta");
        }
        cross_scope_body["index_jobs"][0]["namespace"] = json!("beta");
        let cross_scope = send_mutation(
            make_app(),
            make_token(&["write_graph"], "alpha"),
            cross_scope_body,
        )
        .await;
        assert_eq!(cross_scope.status(), StatusCode::FORBIDDEN);

        let scoped_key = key;
        let scoped_token = encode(
            &Header::new(Algorithm::HS256),
            &json!({
                "sub": "alice",
                "role": "rw",
                "roles": ["reader", "rw"],
                "ns": "workflow",
                "storage_ns": "alpha",
                "workspace": "workspace-a",
                "graph_space": "facts",
                "capabilities": ["write_graph"],
                "exp": 4102444800_u64
            }),
            &EncodingKey::from_secret(scoped_key.as_bytes()),
        )
        .unwrap();
        let mut workspace_mismatch_body = body.clone();
        for mutation in workspace_mismatch_body["mutations"].as_array_mut().unwrap() {
            mutation["scope"]["workspace_id"] = json!("workspace-b");
        }
        let workspace_mismatch =
            send_mutation(make_app(), scoped_token, workspace_mismatch_body).await;
        assert_eq!(workspace_mismatch.status(), StatusCode::FORBIDDEN);
        let mut graph_space_mismatch_body = body.clone();
        for mutation in graph_space_mismatch_body["mutations"].as_array_mut().unwrap() {
            mutation["scope"]["graph_space"] = json!("private");
        }
        let graph_space_mismatch = send_mutation(
            make_app(),
            encode(
                &Header::new(Algorithm::HS256),
                &json!({
                    "sub": "alice",
                    "role": "rw",
                    "roles": ["reader", "rw"],
                    "ns": "workflow",
                    "storage_ns": "alpha",
                    "workspace": "workspace-a",
                    "graph_space": "facts",
                    "capabilities": ["write_graph"],
                    "exp": 4102444800_u64
                }),
                &EncodingKey::from_secret(scoped_key.as_bytes()),
            )
            .unwrap(),
            graph_space_mismatch_body,
        )
        .await;
        assert_eq!(graph_space_mismatch.status(), StatusCode::FORBIDDEN);

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = make_app();
        let server = tokio::spawn(async move { axum::serve(listener, app).await });
        let http_response = reqwest::Client::new()
            .post(format!("http://{address}/internal/graph/mutations"))
            .bearer_auth(make_token(&["write_graph"], "alpha"))
            .json(&body)
            .send()
            .await
            .unwrap();
        let http_status = http_response.status();
        let response_body = http_response.bytes().await.unwrap();
        assert_eq!(
            http_status,
            StatusCode::OK,
            "{}",
            String::from_utf8_lossy(&response_body)
        );
        let result: Value = serde_json::from_slice(&response_body).unwrap();
        assert_eq!(result["atomic"], true);
        assert!(native_seam_attestation_is_valid(&result));
        assert_eq!(result["implementation_authority"], "rust");
        assert_eq!(result["seam_attestation_version"], 1);
        assert_eq!(result["seam_crossings"], json!([]));
        assert_eq!(result["results"][0]["mutated"], true);

        let replay_response = reqwest::Client::new()
            .post(format!("http://{address}/internal/graph/mutations"))
            .bearer_auth(make_token(&["write_graph"], "alpha"))
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(replay_response.status(), StatusCode::OK);
        let replay: Value = replay_response.json().await.unwrap();
        assert!(
            replay["results"]
                .as_array()
                .unwrap()
                .iter()
                .all(|item| {
                    item["inserted"].as_bool() == Some(false)
                        && item["mutated"].as_bool() == Some(false)
                })
        );

        let mut conflicting_retry = body.clone();
        conflicting_retry["mutations"][0]["payload"]["retry_conflict"] = json!(true);
        let conflicting_response = reqwest::Client::new()
            .post(format!("http://{address}/internal/graph/mutations"))
            .bearer_auth(make_token(&["write_graph"], "alpha"))
            .json(&conflicting_retry)
            .send()
            .await
            .unwrap();
        assert_eq!(conflicting_response.status(), StatusCode::CONFLICT);

        let mut generated = body.clone();
        generated["mutations"] = json!([body["mutations"][0].clone()]);
        generated["mutations"][0]["event_id"] = json!("graph-route-event-rust-embedding");
        generated["mutations"][0]["payload"]["id"] = json!("node-rust-embedded");
        generated["mutations"][0]["record"]["id"] = json!("node-rust-embedded");
        generated["mutations"][0]["record"]["document"] = json!("Rust-generated vector input");
        generated["mutations"][0]["record"]["embedding"] = Value::Null;
        generated["mutations"][0]["embedding_dim"] = json!(0);
        generated["index_jobs"] = json!([]);
        let generated_response = reqwest::Client::new()
            .post(format!("http://{address}/internal/graph/mutations"))
            .bearer_auth(make_token(&["write_graph"], "alpha"))
            .json(&generated)
            .send()
            .await
            .unwrap();
        assert_eq!(generated_response.status(), StatusCode::OK);
        assert_eq!(embedding_calls.lock().unwrap().len(), 1);

        embedding_failure.store(true, std::sync::atomic::Ordering::Relaxed);
        let mut failed_generation = generated.clone();
        failed_generation["mutations"][0]["event_id"] = json!("graph-route-event-provider-failure");
        failed_generation["mutations"][0]["payload"]["id"] = json!("node-provider-failure");
        failed_generation["mutations"][0]["record"]["id"] = json!("node-provider-failure");
        let failed_response = reqwest::Client::new()
            .post(format!("http://{address}/internal/graph/mutations"))
            .bearer_auth(make_token(&["write_graph"], "alpha"))
            .json(&failed_generation)
            .send()
            .await
            .unwrap();
        assert_eq!(failed_response.status(), StatusCode::BAD_GATEWAY);
        server.abort();
        let _ = server.await;

        let events = service
            .store
            .replay_raw_events("alpha", 0, 100)
            .await
            .unwrap();
        assert_eq!(events.len(), 39);
        let jobs = service
            .store
            .list_index_jobs(Some("alpha"), None, None, None, None, 100)
            .await
            .unwrap();
        assert_eq!(jobs.len(), 57);
        assert!(jobs.iter().any(|job| job.job_id == "graph-route-index-1"));

        let records = service
            .store
            .graph_projection_records(kogwistar_store::GraphProjectionRead {
                scope: GraphScope {
                    namespace: "alpha".to_owned(),
                    workspace_id: Some("ws".to_owned()),
                    graph_space: Some("facts".to_owned()),
                },
                table: "gke_nodes".to_owned(),
                ids: Some(vec!["node-1".to_owned()]),
                metadata: Default::default(),
                limit: 10,
            })
            .await
            .unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].id, "node-1");
        let generated_records = service
            .store
            .graph_projection_records(kogwistar_store::GraphProjectionRead {
                scope: GraphScope {
                    namespace: "alpha".to_owned(),
                    workspace_id: Some("ws".to_owned()),
                    graph_space: Some("facts".to_owned()),
                },
                table: "gke_nodes".to_owned(),
                ids: Some(vec!["node-rust-embedded".to_owned()]),
                metadata: Default::default(),
                limit: 10,
            })
            .await
            .unwrap();
        assert_eq!(generated_records.len(), 1);
        assert_eq!(generated_records[0].embedding, Some(vec![0.0, 0.0, 1.0]));
        let failed_records = service
            .store
            .graph_projection_records(kogwistar_store::GraphProjectionRead {
                scope: GraphScope {
                    namespace: "alpha".to_owned(),
                    workspace_id: Some("ws".to_owned()),
                    graph_space: Some("facts".to_owned()),
                },
                table: "gke_nodes".to_owned(),
                ids: Some(vec!["node-provider-failure".to_owned()]),
                metadata: Default::default(),
                limit: 10,
            })
            .await
            .unwrap();
        assert!(failed_records.is_empty());
        let edge_records = service
            .store
            .graph_projection_records(kogwistar_store::GraphProjectionRead {
                scope: GraphScope {
                    namespace: "alpha".to_owned(),
                    workspace_id: Some("ws".to_owned()),
                    graph_space: Some("facts".to_owned()),
                },
                table: "gke_edges".to_owned(),
                ids: Some(vec!["edge-1".to_owned()]),
                metadata: Default::default(),
                limit: 10,
            })
            .await
            .unwrap();
        assert_eq!(edge_records.len(), 1);
        assert_eq!(edge_records[0].id, "edge-1");

        let mcp_neighbors = make_app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header(
                        "authorization",
                        format!("Bearer {}", make_docs_token(&["read_graph"])),
                    )
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":"neighbors","method":"tools/call","params":{"name":"kg_neighbors","arguments":{"rid":"node-1"}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(mcp_neighbors.status(), StatusCode::OK);
        let mcp_neighbors_body = to_bytes(mcp_neighbors.into_body(), 65_536).await.unwrap();
        let mcp_neighbors_value: Value = serde_json::from_slice(&mcp_neighbors_body).unwrap();
        assert_eq!(mcp_neighbors_value["result"]["nodes"], json!(["node-2"]));
        assert_eq!(mcp_neighbors_value["result"]["edges"], json!(["edge-1"]));

        let index_jobs = service
            .store
            .list_index_jobs(
                Some("alpha"),
                Some("PENDING"),
                Some("node"),
                Some("node-1"),
                Some("node_embedding"),
                10,
            )
            .await
            .unwrap();
        assert_eq!(index_jobs.len(), 1);
        assert_eq!(index_jobs[0].job_id, "graph-route-index-1");

        // A transport retry must be idempotent at both authority layers:
        // the graph event is not appended twice and the pending index intent
        // remains one coalesced queue row.
        let retry_response = send_mutation(
            make_app(),
            make_token(&["write_graph"], "alpha"),
            body.clone(),
        )
        .await;
        let retry_status = retry_response.status();
        let retry_body = to_bytes(retry_response.into_body(), 65_536).await.unwrap();
        assert_eq!(
            retry_status,
            StatusCode::OK,
            "{}",
            String::from_utf8_lossy(&retry_body)
        );
        let retry_result: Value = serde_json::from_slice(&retry_body).unwrap();
        assert_eq!(retry_result["results"][0]["mutated"], false);
        assert_eq!(
            service
                .store
                .replay_raw_events("alpha", 0, 100)
                .await
                .unwrap()
                .len(),
            39
        );
        assert_eq!(
            service
                .store
                .list_index_jobs(
                    Some("alpha"),
                    Some("PENDING"),
                    Some("node"),
                    Some("node-1"),
                    Some("node_embedding"),
                    10,
                )
                .await
                .unwrap()
                .len(),
            1
        );

        // A queue primary-key collision must roll back the graph event that
        // preceded it in the same transaction; otherwise a retry could leave
        // graph truth ahead of its durable indexing intent.
        let mut rollback_body = body.clone();
        rollback_body["mutations"][0]["event_id"] = json!("graph-route-event-2");
        rollback_body["mutations"][0]["record"]["id"] = json!("node-rollback");
        rollback_body["mutations"][0]["payload"]["id"] = json!("node-rollback");
        rollback_body["index_jobs"][0]["entity_id"] = json!("node-rollback");
        rollback_body["index_jobs"][0]["job_id"] = json!("graph-route-index-1");
        let rollback_response = send_mutation(
            make_app(),
            make_token(&["write_graph"], "alpha"),
            rollback_body,
        )
        .await;
        assert_eq!(rollback_response.status(), StatusCode::CONFLICT);
        assert!(
            service
                .store
                .graph_projection_records(kogwistar_store::GraphProjectionRead {
                    scope: GraphScope {
                        namespace: "alpha".to_owned(),
                        workspace_id: Some("ws".to_owned()),
                        graph_space: Some("facts".to_owned()),
                    },
                    table: "gke_nodes".to_owned(),
                    ids: Some(vec!["node-rollback".to_owned()]),
                    metadata: Default::default(),
                    limit: 10,
                })
                .await
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            service
                .store
                .replay_raw_events("alpha", 0, 10)
                .await
                .unwrap()
                .len(),
            10
        );
        assert!(
            service
                .store
                .list_index_jobs(
                    Some("alpha"),
                    Some("PENDING"),
                    Some("node"),
                    Some("node-2"),
                    Some("node_embedding"),
                    10,
                )
                .await
                .unwrap()
                .is_empty()
        );

        let adjacency_body = json!({
            "contract_version": 1,
            "atomic": true,
            "mutations": [
                {
                    "scope": {"namespace": "alpha", "workspace_id": "ws", "graph_space": "facts"},
                    "table": "gke_nodes",
                    "entity_kind": "node",
                    "event_id": "graph-route-adjacency-node-2",
                    "op": "ADD",
                    "payload": {"id": "node-2"},
                    "record": {"id": "node-2", "document": "{\"label\":\"Beta target\"}", "metadata": {}, "embedding": [0.0, 1.0, 0.0]},
                    "embedding_dim": 3
                },
                {
                    "scope": {"namespace": "alpha", "workspace_id": "ws", "graph_space": "facts"},
                    "table": "gke_edges",
                    "entity_kind": "edge",
                    "event_id": "graph-route-adjacency-edge-1",
                    "op": "ADD",
                    "payload": {"id": "edge-1", "source_ids": ["node-1"], "target_ids": ["node-2"], "source_edge_ids": [], "target_edge_ids": []},
                    "record": {
                        "id": "edge-1",
                        "document": "{\"relation\":\"links\",\"source_ids\":[\"node-1\"],\"target_ids\":[\"node-2\"],\"source_edge_ids\":[],\"target_edge_ids\":[]}",
                        "metadata": {},
                        "embedding": [0.0, 0.0, 1.0]
                    },
                    "embedding_dim": 3
                },
                {
                    "scope": {"namespace": "alpha", "workspace_id": "ws", "graph_space": "facts"},
                    "table": "gke_nodes",
                    "entity_kind": "node",
                    "event_id": "graph-route-acl-node-1",
                    "op": "ADD",
                    "payload": {"id": "acl-node-1"},
                    "record": {"id": "acl-node-1", "metadata": {"entity_type": "acl_record", "acl_truth_graph": "knowledge", "acl_target_grain": "node", "acl_target_entity_id": "node-1", "acl_version": 1, "acl_mode": "public"}, "embedding": [0.0, 0.0, 0.0]},
                    "embedding_dim": 3
                },
                {
                    "scope": {"namespace": "alpha", "workspace_id": "ws", "graph_space": "facts"},
                    "table": "gke_nodes",
                    "entity_kind": "node",
                    "event_id": "graph-route-acl-node-2",
                    "op": "ADD",
                    "payload": {"id": "acl-node-2"},
                    "record": {"id": "acl-node-2", "metadata": {"entity_type": "acl_record", "acl_truth_graph": "knowledge", "acl_target_grain": "node", "acl_target_entity_id": "node-2", "acl_version": 1, "acl_mode": "public"}, "embedding": [0.0, 0.0, 0.0]},
                    "embedding_dim": 3
                },
                {
                    "scope": {"namespace": "alpha", "workspace_id": "ws", "graph_space": "facts"},
                    "table": "gke_nodes",
                    "entity_kind": "node",
                    "event_id": "graph-route-acl-edge-1",
                    "op": "ADD",
                    "payload": {"id": "acl-edge-1"},
                    "record": {"id": "acl-edge-1", "metadata": {"entity_type": "acl_record", "acl_truth_graph": "knowledge", "acl_target_grain": "edge", "acl_target_entity_id": "edge-1", "acl_version": 1, "acl_mode": "public"}, "embedding": [0.0, 0.0, 0.0]},
                    "embedding_dim": 3
                }
            ],
            "index_jobs": []
        });
        let adjacency_response = send_mutation(
            make_app(),
            make_token(&["write_graph"], "alpha"),
            adjacency_body,
        )
        .await;
        assert_eq!(adjacency_response.status(), StatusCode::OK);

        let mcp_adjacency = make_app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header(
                        "authorization",
                        format!("Bearer {}", make_docs_token(&["read_graph"])),
                    )
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":"adjacency","method":"tools/call","params":{"name":"kg_neighbors","arguments":{"rid":"node-1"}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(mcp_adjacency.status(), StatusCode::OK);
        let adjacency_body = to_bytes(mcp_adjacency.into_body(), 65_536).await.unwrap();
        let adjacency_value: Value = serde_json::from_slice(&adjacency_body).unwrap();
        assert_eq!(
            adjacency_value["result"]["implementation_authority"],
            "rust"
        );
        assert_eq!(adjacency_value["result"]["seam_attestation_version"], 1);
        assert_eq!(adjacency_value["result"]["seam_crossings"], json!([]));
        assert_eq!(adjacency_value["result"]["nodes"], json!(["node-2"]));
        assert_eq!(adjacency_value["result"]["edges"], json!(["edge-1"]));

        let mcp_find_edges = make_app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header(
                        "authorization",
                        format!("Bearer {}", make_docs_token(&["read_graph"])),
                    )
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":"find-edges","method":"tools/call","params":{"name":"kg_find_edges","arguments":{"relation":"links","src_label_contains":"alpha","tgt_label_contains":"TARGET"}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(mcp_find_edges.status(), StatusCode::OK);
        let find_edges_body = to_bytes(mcp_find_edges.into_body(), 65_536).await.unwrap();
        let find_edges_value: Value = serde_json::from_slice(&find_edges_body).unwrap();
        assert_eq!(
            find_edges_value["result"]["implementation_authority"],
            "rust"
        );
        assert_eq!(find_edges_value["result"]["seam_attestation_version"], 1);
        assert_eq!(find_edges_value["result"]["seam_crossings"], json!([]));
        assert_eq!(find_edges_value["result"]["edges"], json!(["edge-1"]));

        let mcp_shortest_path = make_app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header(
                        "authorization",
                        format!("Bearer {}", make_docs_token(&["read_graph"])),
                    )
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":"shortest-path","method":"tools/call","params":{"name":"kg_shortest_path","arguments":{"src_id":"node-1","dst_id":"node-2","max_depth":4}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(mcp_shortest_path.status(), StatusCode::OK);
        let shortest_path_body = to_bytes(mcp_shortest_path.into_body(), 65_536)
            .await
            .unwrap();
        let shortest_path_value: Value = serde_json::from_slice(&shortest_path_body).unwrap();
        assert_eq!(
            shortest_path_value["result"]["implementation_authority"],
            "rust"
        );
        assert_eq!(shortest_path_value["result"]["seam_attestation_version"], 1);
        assert_eq!(shortest_path_value["result"]["seam_crossings"], json!([]));
        assert_eq!(
            shortest_path_value["result"]["path"],
            json!(["node-1", "edge-1", "node-2"])
        );

        let mcp_k_hop = make_app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header(
                        "authorization",
                        format!("Bearer {}", make_docs_token(&["read_graph"])),
                    )
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":"k-hop","method":"tools/call","params":{"name":"kg_k_hop","arguments":{"start_ids":["node-1"],"k":1}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(mcp_k_hop.status(), StatusCode::OK);
        let k_hop_body = to_bytes(mcp_k_hop.into_body(), 65_536).await.unwrap();
        let k_hop_value: Value = serde_json::from_slice(&k_hop_body).unwrap();
        assert_eq!(k_hop_value["result"]["implementation_authority"], "rust");
        assert_eq!(k_hop_value["result"]["seam_attestation_version"], 1);
        assert_eq!(k_hop_value["result"]["seam_crossings"], json!([]));
        assert_eq!(k_hop_value["result"]["layers"][0]["nodes"], json!([]));
        assert_eq!(
            k_hop_value["result"]["layers"][0]["edges"],
            json!(["edge-1"])
        );

        let acl_fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/graph-acl-traversal-v1.json"
        )))
        .unwrap();
        let mut acl_fixture_batch = json!({
            "contract_version": 1,
            "atomic": true,
            "mutations": [],
            "index_jobs": []
        });
        let template = &body["mutations"][0];
        for node in acl_fixture["nodes"].as_array().unwrap() {
            let id = node["id"].as_str().unwrap();
            let mut mutation = template.clone();
            mutation["event_id"] = json!(format!("acl-fixture-node-{id}"));
            mutation["scope"]["namespace"] = json!("alpha");
            mutation["payload"] = json!({"id": id});
            mutation["record"]["id"] = json!(id);
            mutation["record"]["document"] = json!(serde_json::to_string(&json!({
                "label": id,
                "doc_id": acl_fixture["doc_id"]
            })).unwrap());
            mutation["record"]["metadata"] = json!({});
            mutation["record"]["embedding"] = json!([1.0, 0.0, 0.0]);
            acl_fixture_batch["mutations"]
                .as_array_mut()
                .unwrap()
                .push(mutation);
        }
        let edge_fixture = &acl_fixture["edge"];
        let edge_id = edge_fixture["id"].as_str().unwrap();
        let mut edge_mutation = template.clone();
        edge_mutation["table"] = json!("gke_edges");
        edge_mutation["entity_kind"] = json!("edge");
        edge_mutation["event_id"] = json!(format!("acl-fixture-edge-{edge_id}"));
        edge_mutation["scope"]["namespace"] = json!("alpha");
        edge_mutation["payload"] = json!({
            "id": edge_id,
            "source_ids": edge_fixture["source_ids"],
            "target_ids": edge_fixture["target_ids"],
            "source_edge_ids": [],
            "target_edge_ids": []
        });
        edge_mutation["record"]["id"] = json!(edge_id);
        edge_mutation["record"]["document"] = json!(serde_json::to_string(&json!({
            "relation": edge_fixture["relation"],
            "doc_id": acl_fixture["doc_id"],
            "source_ids": edge_fixture["source_ids"],
            "target_ids": edge_fixture["target_ids"],
            "source_edge_ids": [],
            "target_edge_ids": []
        })).unwrap());
        edge_mutation["record"]["metadata"] = json!({});
        edge_mutation["record"]["embedding"] = json!([0.0, 1.0, 0.0]);
        acl_fixture_batch["mutations"]
            .as_array_mut()
            .unwrap()
            .push(edge_mutation);

        for (grain, entity) in acl_fixture["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|node| ("node", node))
            .chain(std::iter::once(("edge", edge_fixture)))
        {
            let target_id = entity["id"].as_str().unwrap();
            let mut acl_record = template.clone();
            let acl_id = format!("acl-fixture-{grain}-{target_id}");
            let mut metadata = json!({
                "entity_type": "acl_record",
                "acl_truth_graph": acl_fixture["truth_graph"],
                "acl_target_grain": grain,
                "acl_target_item_id": "",
                "acl_target_entity_id": target_id,
                "acl_version": 1,
                "acl_mode": entity["mode"],
                "tombstoned": false
            });
            if let Some(owner_id) = entity["owner_id"].as_str() {
                metadata["owner_id"] = json!(owner_id);
            }
            acl_record["event_id"] = json!(format!("acl-fixture-event-{acl_id}"));
            acl_record["scope"]["namespace"] = json!("alpha");
            acl_record["payload"] = json!({"id": acl_id});
            acl_record["record"]["id"] = json!(acl_id);
            acl_record["record"]["metadata"] = metadata;
            acl_record["record"]["document"] = Value::Null;
            acl_record["record"]["embedding"] = Value::Null;
            acl_fixture_batch["mutations"]
                .as_array_mut()
                .unwrap()
                .push(acl_record);
        }
        let acl_write = send_mutation(
            make_app(),
            make_token(&["write_graph"], "alpha"),
            acl_fixture_batch,
        )
        .await;
        assert_eq!(acl_write.status(), StatusCode::OK);

        for principal in acl_fixture["principals"].as_array().unwrap() {
            let subject = principal["id"].as_str().unwrap();
            let requests = [
                (
                    "kg_neighbors",
                    json!({"rid": "source-public", "doc_id": acl_fixture["doc_id"]}),
                ),
                (
                    "kg_k_hop",
                    json!({"start_ids": ["source-public"], "k": 2, "doc_id": acl_fixture["doc_id"]}),
                ),
                (
                    "kg_shortest_path",
                    json!({"src_id": "source-public", "dst_id": "target-private", "max_depth": 8, "doc_id": acl_fixture["doc_id"]}),
                ),
                (
                    "kg_find_edges",
                    json!({"relation": edge_fixture["relation"], "doc_id": acl_fixture["doc_id"]}),
                ),
            ];
            for (tool_name, arguments) in requests {
                let response = make_app()
                    .oneshot(
                        Request::builder()
                            .method("POST")
                            .uri("/mcp")
                            .header("content-type", "application/json")
                            .header(
                                "authorization",
                                format!(
                                    "Bearer {}",
                                    make_docs_token_for(subject, &["read_graph"])
                                ),
                            )
                            .body(Body::from(
                                json!({
                                    "jsonrpc": "2.0",
                                    "id": format!("acl-{subject}-{tool_name}"),
                                    "method": "tools/call",
                                    "params": {"name": tool_name, "arguments": arguments}
                                })
                                .to_string(),
                            ))
                            .unwrap(),
                    )
                    .await
                    .unwrap();
                assert_eq!(response.status(), StatusCode::OK);
                let response_body = to_bytes(response.into_body(), 65_536).await.unwrap();
                let response_value: Value = serde_json::from_slice(&response_body).unwrap();
                assert!(response_value.get("error").is_none(), "{response_value}");
                let result = &response_value["result"];
                match tool_name {
                    "kg_neighbors" => {
                        assert_eq!(
                            result["nodes"], principal["expected_neighbors"]["nodes"],
                            "unexpected visible nodes for {subject}"
                        );
                        assert_eq!(
                            result["edges"], principal["expected_neighbors"]["edges"],
                            "unexpected visible edges for {subject}"
                        );
                    }
                    "kg_k_hop" => {
                        assert_eq!(
                            result["layers"], principal["expected_k_hop"],
                            "unexpected k-hop layers for {subject}"
                        );
                    }
                    "kg_shortest_path" => {
                        assert_eq!(
                            result["path"], principal["expected_shortest_path"],
                            "unexpected path for {subject}"
                        );
                    }
                    "kg_find_edges" => {
                        assert_eq!(
                            result["edges"], principal["expected_find_edges"],
                            "unexpected relation matches for {subject}"
                        );
                    }
                    _ => unreachable!("test request has known MCP tools"),
                }
            }
        }
        let tree_fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/document-upsert-tree-success-v1.json"
        )))
        .unwrap();
        let tree_request = &tree_fixture["request"];
        let tree_response = &tree_fixture["expected"];
        let tree_upsert = make_app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/document.upsert_tree")
                    .header("content-type", "application/json")
                    .header("authorization", format!("Bearer {graph_upsert_token}"))
                    .body(Body::from(serde_json::to_vec(tree_request).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(tree_upsert.status().as_u16(), tree_response["status_code"]);
        let tree_body: Value = serde_json::from_slice(
            &to_bytes(tree_upsert.into_body(), 65_536).await.unwrap(),
        )
        .unwrap();
        assert_eq!(tree_body, tree_response["body"]);

        let mut tree_payload = extraction.clone();
        let tree_payload_object = tree_payload.as_object_mut().unwrap();
        tree_payload_object.remove("contract_version");
        tree_payload_object.remove("content");
        tree_payload_object.remove("doc_type");
        tree_payload_object.remove("insertion_method");
        let tree_upsert = make_app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/document.upsert_tree")
                    .header("content-type", "application/json")
                    .header("authorization", format!("Bearer {graph_upsert_token}"))
                    .body(Body::from(serde_json::to_vec(&tree_payload).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(tree_upsert.status(), StatusCode::OK);
        let tree_body: Value = serde_json::from_slice(
            &to_bytes(tree_upsert.into_body(), 65_536).await.unwrap(),
        )
        .unwrap();
        assert_eq!(tree_body["status"], "ok");
        assert_eq!(tree_body["inserted_nodes"], 2);
        assert_eq!(tree_body["inserted_edges"], 1);
        assert_eq!(tree_body["engine_result"]["document_id"], "public-graph-upsert-doc");
        assert_eq!(tree_body["engine_result"]["node_ids"].as_array().unwrap().len(), 2);
        assert_eq!(tree_body["engine_result"]["edge_ids"].as_array().unwrap().len(), 1);
        drop(mismatched_service);
        drop(legacy_service);
        drop(service);
        client
            .batch_execute(&format!(
                "DROP SCHEMA \"{schema}\" CASCADE; DROP SCHEMA \"{legacy_schema}\" CASCADE"
            ))
            .await
            .unwrap();
    }
