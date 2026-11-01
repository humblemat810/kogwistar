    #[tokio::test]
    async fn sqlite_run_transient_delta_routes_preserve_owner_and_sequence_contract() {
        let sqlite_path = std::env::temp_dir().join(format!(
            "kogwistar-transient-delta-route-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let service = SqliteRunApplicationService::open(&sqlite_path).unwrap();
        service
            .store
            .create_server_run(ServerRunCreate {
                run_id: "delta-route-run".to_owned(),
                conversation_id: "delta-route-conversation".to_owned(),
                workflow_id: "delta-route-workflow".to_owned(),
                user_id: Some("owner".to_owned()),
                user_turn_node_id: "delta-route-turn".to_owned(),
                status: "running".to_owned(),
            })
            .unwrap();
        service
            .transient_run_deltas
            .publish("delta-route-run", 4, "visible chunk")
            .unwrap();
        let principal = json!({
            "sub": "owner",
            "user_id": "owner",
            "ns": ["workflow"],
            "capabilities": ["workflow.run.read"]
        });

        let legacy = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: "/api/runs/delta-route-run/events/poll?after_seq=0".to_owned(),
                body: Vec::new(),
                principal: principal.clone(),
            })
            .await;
        assert_eq!(legacy.status, StatusCode::OK.as_u16());
        assert!(serde_json::from_slice::<Value>(&legacy.body)
            .unwrap()
            .get("client_deltas")
            .is_none());

        let denied = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: "/api/runs/delta-route-run/events/poll?after_seq=0&after_client_sequence=3".to_owned(),
                body: Vec::new(),
                principal: json!({
                    "sub": "not-owner",
                    "user_id": "not-owner",
                    "ns": ["workflow"],
                    "capabilities": ["workflow.run.read"]
                }),
            })
            .await;
        assert_eq!(denied.status, StatusCode::FORBIDDEN.as_u16());

        let poll = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: "/api/runs/delta-route-run/events/poll?after_seq=0&after_client_sequence=3".to_owned(),
                body: Vec::new(),
                principal: principal.clone(),
            })
            .await;
        assert_eq!(poll.status, StatusCode::OK.as_u16());
        let poll_body: Value = serde_json::from_slice(&poll.body).unwrap();
        assert_eq!(poll_body["client_deltas"][0]["client_sse_sequence"], 4);
        assert_eq!(poll_body["client_deltas"][0]["delta"], "visible chunk");

        let sse = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: "/api/runs/delta-route-run/events?after_client_sequence=3".to_owned(),
                body: Vec::new(),
                principal,
            })
            .await;
        assert_eq!(sse.status, StatusCode::OK.as_u16());
        let sse_body = String::from_utf8(sse.body).unwrap();
        assert!(sse_body.contains("id: delta-4\nevent: response.text.delta"));
        assert!(sse_body.contains("visible chunk"));
        assert!(!sse_body.contains("id: 4\n"));

        drop(service);
        remove_test_sqlite(&sqlite_path);
    }

    #[tokio::test]
    async fn workflow_lineage_checks_authority_before_run_lookup() {
        let sqlite_path = std::env::temp_dir().join(format!(
            "kogwistar-lineage-auth-order-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let sqlite = SqliteRunApplicationService::open(&sqlite_path).unwrap();
        let postgres = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "lineage_auth_order",
        )
        .unwrap();

        for service in [&sqlite as &dyn ApplicationService, &postgres] {
            let response = service
                .execute(ApiEffectRequest {
                    contract_version: 1,
                    method: "GET".to_owned(),
                    path_and_query: "/api/workflow/runs/missing/lineage".to_owned(),
                    body: Vec::new(),
                    principal: json!({"ns": ["docs"], "capabilities": []}),
                })
                .await;
            assert_eq!(response.status, StatusCode::FORBIDDEN.as_u16());
            assert_eq!(
                serde_json::from_slice::<Value>(&response.body).unwrap()["code"],
                "KOGWISTAR_NAMESPACE_FORBIDDEN"
            );

            let response = service
                .execute(ApiEffectRequest {
                    contract_version: 1,
                    method: "GET".to_owned(),
                    path_and_query: "/api/workflow/runs/missing/lineage".to_owned(),
                    body: Vec::new(),
                    principal: json!({"ns": ["workflow"], "capabilities": []}),
                })
                .await;
            assert_eq!(response.status, StatusCode::FORBIDDEN.as_u16());
            assert_eq!(
                serde_json::from_slice::<Value>(&response.body).unwrap()["code"],
                "KOGWISTAR_CAPABILITY_FORBIDDEN"
            );
        }

        drop(sqlite);
        remove_test_sqlite(&sqlite_path);
    }

    #[tokio::test]
    async fn run_evidence_checks_authority_before_run_lookup() {
        let sqlite_path = std::env::temp_dir().join(format!(
            "kogwistar-evidence-auth-order-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let sqlite = SqliteRunApplicationService::open(&sqlite_path).unwrap();
        let postgres = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "evidence_auth_order",
        )
        .unwrap();

        for service in [&sqlite as &dyn ApplicationService, &postgres] {
            let response = service
                .execute(ApiEffectRequest {
                    contract_version: 1,
                    method: "GET".to_owned(),
                    path_and_query: "/api/runs/missing/evidence".to_owned(),
                    body: Vec::new(),
                    principal: json!({"ns": ["docs"], "capabilities": []}),
                })
                .await;
            assert_eq!(response.status, StatusCode::FORBIDDEN.as_u16());
            assert_eq!(
                serde_json::from_slice::<Value>(&response.body).unwrap()["code"],
                "KOGWISTAR_NAMESPACE_FORBIDDEN"
            );

            let response = service
                .execute(ApiEffectRequest {
                    contract_version: 1,
                    method: "GET".to_owned(),
                    path_and_query: "/api/runs/missing/evidence".to_owned(),
                    body: Vec::new(),
                    principal: json!({
                        "ns": ["workflow"],
                        "capabilities": ["workflow.run.read"]
                    }),
                })
                .await;
            assert_eq!(response.status, StatusCode::FORBIDDEN.as_u16());
            assert_eq!(
                serde_json::from_slice::<Value>(&response.body).unwrap()["code"],
                "KOGWISTAR_CAPABILITY_FORBIDDEN"
            );
        }

        let unsupported = sqlite
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: "/api/runs/missing/evidence".to_owned(),
                body: Vec::new(),
                principal: json!({
                    "ns": ["workflow"],
                    "capabilities": ["workflow.run.read", "project_view"]
                }),
            })
            .await;
        assert_eq!(unsupported.status, StatusCode::SERVICE_UNAVAILABLE.as_u16());
        assert_eq!(
            serde_json::from_slice::<Value>(&unsupported.body).unwrap()["code"],
            "KOGWISTAR_GRAPH_AUTHORITY_UNSUPPORTED"
        );

        drop(sqlite);
        remove_test_sqlite(&sqlite_path);
    }

    #[test]
    fn rust_run_evidence_projection_matches_shared_v1_contract() {
        let expected: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/run-evidence-v1.json"
        ))
        .unwrap();
        let metadata = serde_json::from_value::<BTreeMap<String, Value>>(
            expected["snapshot_metadata"].clone(),
        )
        .unwrap();
        let digest = serde_json::to_string(&expected["source_summary"]["evidence_pack_digest"])
            .unwrap();
        let snapshot = ContextSnapshotV1 {
            schema_version: 1,
            snapshot_node_id: expected["snapshot_node_id"].as_str().unwrap().to_owned(),
            conversation_id: expected["conversation_id"].as_str().unwrap().to_owned(),
            metadata,
            properties: BTreeMap::from([("evidence_pack_digest".to_owned(), json!(digest))]),
        };
        assert_eq!(
            public_run_evidence_value(expected["run_id"].as_str().unwrap(), snapshot),
            expected
        );
    }

    #[tokio::test]
    #[ignore = "requires KOGWISTAR_TEST_PG_DSN; uses a disposable PostgreSQL schema"]
    async fn postgres_run_evidence_route_returns_only_authorized_run_snapshot() {
        let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
            .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
        let schema = format!("rust_run_evidence_{}", uuid::Uuid::new_v4().simple());
        let service = PostgresRunApplicationService::from_dsn(&dsn, &schema).unwrap();
        service.ensure_schema().await.unwrap();
        service
            .store
            .create_graph_schema(3, kogwistar_store_postgres::GraphTableNames::default())
            .await
            .unwrap();
        service
            .store
            .create_server_run(ServerRunCreate {
                run_id: "evidence-run".to_owned(),
                conversation_id: "evidence-conversation".to_owned(),
                workflow_id: "evidence-workflow".to_owned(),
                user_id: Some("alice".to_owned()),
                user_turn_node_id: "evidence-turn".to_owned(),
                status: "succeeded".to_owned(),
            })
            .await
            .unwrap();

        let scope = GraphScope {
            namespace: "conversation".to_owned(),
            workspace_id: None,
            graph_space: None,
        };
        let snapshot_id = "snapshot|evidence-run|5";
        let mut snapshot_metadata = Map::new();
        for (key, value) in [
            ("entity_type", json!("context_snapshot")),
            ("conversation_id", json!("evidence-conversation")),
            ("run_id", json!("evidence-run")),
            ("run_step_seq", json!(5)),
            ("lifecycle_status", json!("active")),
        ] {
            snapshot_metadata.insert(key.to_owned(), value);
        }
        service
            .store
            .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                record: GraphRecord {
                    id: snapshot_id.to_owned(),
                    document: Some(
                        json!({
                            "metadata": {
                                "entity_type": "context_snapshot",
                                "conversation_id": "evidence-conversation",
                                "run_id": "evidence-run",
                                "run_step_seq": 5,
                                "used_node_ids": ["evidence-node-1", "evidence-node-2"]
                            },
                            "properties": {
                                "evidence_pack_digest": "{\"items\":2}"
                            }
                        })
                        .to_string(),
                    ),
                    metadata: snapshot_metadata,
                    embedding: None,
                },
                embedding_dim: 3,
            })
            .await
            .unwrap();
        let mut acl_metadata = Map::new();
        for (key, value) in [
            ("entity_type", json!("acl_record")),
            ("acl_truth_graph", json!("conversation")),
            ("acl_target_grain", json!("node")),
            ("acl_target_entity_id", json!(snapshot_id)),
            ("acl_mode", json!("private")),
            ("owner_id", json!("alice")),
            ("acl_version", json!(1)),
            ("tombstoned", json!(false)),
        ] {
            acl_metadata.insert(key.to_owned(), value);
        }
        service
            .store
            .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                scope,
                table: "gke_nodes".to_owned(),
                record: GraphRecord {
                    id: format!("acl|{snapshot_id}"),
                    document: Some("{}".to_owned()),
                    metadata: acl_metadata,
                    embedding: None,
                },
                embedding_dim: 3,
            })
            .await
            .unwrap();
        let other_snapshot_id = "snapshot|different-run|99";
        let other_metadata = serde_json::from_value::<Map<String, Value>>(json!({
            "entity_type": "context_snapshot",
            "conversation_id": "evidence-conversation",
            "run_id": "different-run",
            "run_step_seq": 99,
            "lifecycle_status": "active"
        }))
        .unwrap();
        service
            .store
            .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                scope: GraphScope {
                    namespace: "conversation".to_owned(),
                    workspace_id: None,
                    graph_space: None,
                },
                table: "gke_nodes".to_owned(),
                record: GraphRecord {
                    id: other_snapshot_id.to_owned(),
                    document: Some(
                        json!({
                            "metadata": {
                                "entity_type": "context_snapshot",
                                "conversation_id": "evidence-conversation",
                                "run_id": "different-run",
                                "run_step_seq": 99,
                                "used_node_ids": ["wrong-run-evidence"]
                            },
                            "properties": {"evidence_pack_digest": "{\"items\":1}"}
                        })
                        .to_string(),
                    ),
                    metadata: other_metadata,
                    embedding: None,
                },
                embedding_dim: 3,
            })
            .await
            .unwrap();
        let other_acl_metadata = serde_json::from_value::<Map<String, Value>>(json!({
            "entity_type": "acl_record",
            "acl_truth_graph": "conversation",
            "acl_target_grain": "node",
            "acl_target_entity_id": other_snapshot_id,
            "acl_mode": "public",
            "acl_version": 1,
            "tombstoned": false
        }))
        .unwrap();
        service
            .store
            .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                scope: GraphScope {
                    namespace: "conversation".to_owned(),
                    workspace_id: None,
                    graph_space: None,
                },
                table: "gke_nodes".to_owned(),
                record: GraphRecord {
                    id: format!("acl|{other_snapshot_id}"),
                    document: Some("{}".to_owned()),
                    metadata: other_acl_metadata,
                    embedding: None,
                },
                embedding_dim: 3,
            })
            .await
            .unwrap();

        let secret = "run-evidence-route-test-secret";
        let token = encode(
            &Header::new(Algorithm::HS256),
            &json!({
                "user_id": "alice",
                "sub": "alice",
                "roles": ["reader"],
                "ns": ["workflow"],
                "storage_ns": "conversation",
                "capabilities": ["workflow.run.read", "project_view"],
                "exp": 4102444800_u64
            }),
            &EncodingKey::from_secret(secret.as_bytes()),
        )
        .unwrap();
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: vec!["reader".to_owned()],
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig {
                    mode: "jwt".to_owned(),
                    algorithm: Some("HS256".to_owned()),
                    key: Some(secret.to_owned()),
                    ..AuthConfig::default()
                },
            },
            Arc::new(service.clone()),
        );
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/runs/evidence-run/evidence")
                    .header("authorization", format!("Bearer {token}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let payload = serde_json::from_slice::<Value>(
            &to_bytes(response.into_body(), 8192).await.unwrap(),
        )
        .unwrap();
        let expected: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/run-evidence-v1.json"
        ))
        .unwrap();
        assert_eq!(payload, expected);

        let other_user_token = encode(
            &Header::new(Algorithm::HS256),
            &json!({
                "user_id": "bob",
                "sub": "bob",
                "roles": ["reader"],
                "ns": ["workflow"],
                "storage_ns": "conversation",
                "capabilities": ["workflow.run.read", "project_view"],
                "exp": 4102444800_u64
            }),
            &EncodingKey::from_secret(secret.as_bytes()),
        )
        .unwrap();
        let hidden = app
            .oneshot(
                Request::builder()
                    .uri("/api/runs/evidence-run/evidence")
                    .header(
                        "authorization",
                        format!("Bearer {other_user_token}"),
                    )
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(hidden.status(), StatusCode::NOT_FOUND);

        drop(service);
        let (cleanup, connection) =
            tokio_postgres::connect(&dsn, tokio_postgres::NoTls).await.unwrap();
        let cleanup_task = tokio::spawn(connection);
        cleanup
            .batch_execute(&format!("DROP SCHEMA \"{schema}\" CASCADE"))
            .await
            .unwrap();
        drop(cleanup);
        cleanup_task.await.unwrap().unwrap();
    }

    #[tokio::test]
    #[ignore = "requires KOGWISTAR_TEST_PG_DSN; uses a disposable PostgreSQL schema"]
    async fn postgres_workflow_lineage_route_checks_each_workflow_acl() {
        let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
            .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
        let schema = format!("rust_run_lineage_{}", uuid::Uuid::new_v4().simple());
        let auth_path = std::env::temp_dir().join(format!(
            "kogwistar-run-lineage-auth-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let auth_config = AuthConfig {
            auth_db_url: Some(format!("sqlite:///{}", auth_path.display())),
            ..AuthConfig::default()
        };
        let service = PostgresRunApplicationService::from_dsn(&dsn, &schema)
            .unwrap()
            .with_owner_auth_config(&auth_config)
            .await
            .unwrap();
        service.ensure_schema().await.unwrap();
        service
            .store
            .create_graph_schema(3, kogwistar_store_postgres::GraphTableNames::default())
            .await
            .unwrap();
        service
            .store
            .create_server_run(ServerRunCreate {
                run_id: "lineage-child".to_owned(),
                conversation_id: "lineage-conversation".to_owned(),
                workflow_id: "wf-child".to_owned(),
                user_id: Some("alice".to_owned()),
                user_turn_node_id: "lineage-turn".to_owned(),
                status: "completed".to_owned(),
            })
            .await
            .unwrap();

        let auth = SqliteAuthStore::open(&auth_path).unwrap();
        let auth_connection = rusqlite::Connection::open(&auth_path).unwrap();
        auth_connection
            .execute(
                "INSERT INTO users (user_id, email, is_active, created_at) VALUES ('alice', 'alice@example.test', 1, CURRENT_TIMESTAMP)",
                [],
            )
            .unwrap();
        auth_connection
            .execute(
                "INSERT INTO workflow_acl (workflow_id, user_id, role) VALUES ('wf-child', 'alice', 'ro')",
                [],
            )
            .unwrap();
        drop(auth_connection);

        let scope = GraphScope {
            namespace: "conversation".to_owned(),
            workspace_id: None,
            graph_space: None,
        };
        for (run_id, workflow_id, parent_run_id) in [
            ("lineage-child", "wf-child", Some("lineage-parent")),
            ("lineage-parent", "wf-parent", None),
        ] {
            let mut metadata = Map::new();
            metadata.insert("entity_type".to_owned(), json!("workflow_run"));
            metadata.insert("workflow_id".to_owned(), json!(workflow_id));
            metadata.insert("parent_run_id".to_owned(), json!(parent_run_id));
            metadata.insert("status".to_owned(), json!("completed"));
            metadata.insert("lifecycle_status".to_owned(), json!("active"));
            metadata.insert("wf_invoked".to_owned(), json!([]));
            metadata.insert("mentions".to_owned(), json!([]));
            service
                .store
                .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                    scope: scope.clone(),
                    table: "gke_nodes".to_owned(),
                    record: GraphRecord {
                        id: format!("wf_run|{run_id}"),
                        document: Some(json!({"id": run_id}).to_string()),
                        metadata,
                        embedding: None,
                    },
                    embedding_dim: 3,
                })
                .await
                .unwrap();
            let mut acl_metadata = Map::new();
            acl_metadata.insert("entity_type".to_owned(), json!("acl_record"));
            acl_metadata.insert("acl_truth_graph".to_owned(), json!("conversation"));
            acl_metadata.insert("acl_target_grain".to_owned(), json!("node"));
            acl_metadata.insert("acl_target_entity_id".to_owned(), json!(format!("wf_run|{run_id}")));
            acl_metadata.insert("acl_mode".to_owned(), json!("public"));
            acl_metadata.insert("acl_version".to_owned(), json!(1));
            acl_metadata.insert("tombstoned".to_owned(), json!(false));
            service
                .store
                .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                    scope: scope.clone(),
                    table: "gke_nodes".to_owned(),
                    record: GraphRecord {
                        id: format!("acl|wf_run|{run_id}"),
                        document: Some("{}".to_owned()),
                        metadata: acl_metadata,
                        embedding: None,
                    },
                    embedding_dim: 3,
                })
                .await
                .unwrap();
        }
        let stored_child = service
            .store
            .graph_projection_records(GraphProjectionRead {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                ids: Some(vec!["wf_run|lineage-child".to_owned()]),
                metadata: MetadataFilter::default(),
                limit: 1,
            })
            .await
            .unwrap();
        assert_eq!(stored_child.len(), 1, "persisted child run node not found");

        let secret = "lineage-route-test-secret";
        let token = encode(
            &Header::new(Algorithm::HS256),
            &json!({
                "user_id": "alice",
                "sub": "alice",
                "roles": ["reader"],
                "ns": ["workflow"],
                "storage_ns": "conversation",
                "capabilities": ["workflow.run.read"],
                "exp": 4102444800_u64
            }),
            &EncodingKey::from_secret(secret.as_bytes()),
        )
        .unwrap();
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: vec!["reader".to_owned()],
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig {
                    mode: "jwt".to_owned(),
                    algorithm: Some("HS256".to_owned()),
                    key: Some(secret.to_owned()),
                    ..AuthConfig::default()
                },
            },
            Arc::new(service.clone()),
        );
        let request = || {
            Request::builder()
                .uri("/api/workflow/runs/lineage-child/lineage")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::empty())
                .unwrap()
        };
        let denied = app.clone().oneshot(request()).await.unwrap();
        assert_eq!(
            denied.status(),
            StatusCode::FORBIDDEN,
            "unexpected lineage response"
        );
        let denied_body = to_bytes(denied.into_body(), 8192).await.unwrap();
        assert_eq!(serde_json::from_slice::<Value>(&denied_body).unwrap()["code"],
            "KOGWISTAR_WORKFLOW_FORBIDDEN"
        );

        rusqlite::Connection::open(&auth_path)
            .unwrap()
            .execute(
                "INSERT INTO workflow_acl (workflow_id, user_id, role) VALUES ('wf-parent', 'alice', 'ro')",
                [],
            )
            .unwrap();
        let allowed = app.oneshot(request()).await.unwrap();
        assert_eq!(allowed.status(), StatusCode::OK);
        let allowed_body = to_bytes(allowed.into_body(), 8192).await.unwrap();
        let payload = serde_json::from_slice::<Value>(&allowed_body).unwrap();
        assert_eq!(payload["run_id"], "lineage-child");
        assert_eq!(payload["lineage"].as_array().unwrap().len(), 2);
        assert_eq!(payload["lineage"][0]["workflow_id"], "wf-child");
        assert_eq!(payload["lineage"][0]["parent_run_id"], "lineage-parent");
        assert_eq!(payload["lineage"][1]["workflow_id"], "wf-parent");

        drop(service);
        let (cleanup, connection) =
            tokio_postgres::connect(&dsn, tokio_postgres::NoTls).await.unwrap();
        let cleanup_task = tokio::spawn(connection);
        cleanup
            .batch_execute(&format!("DROP SCHEMA \"{schema}\" CASCADE"))
            .await
            .unwrap();
        drop(cleanup);
        cleanup_task.await.unwrap().unwrap();
        drop(auth);
        let _ = std::fs::remove_file(auth_path);
    }

    #[tokio::test]
    #[ignore = "requires KOGWISTAR_TEST_PG_DSN; exercises a disposable wisdom schema"]
    async fn postgres_wisdom_graph_upsert_is_scoped_atomic_and_acl_checked() {
        let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
            .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
        let schema = format!("rust_wisdom_upsert_{}", uuid::Uuid::new_v4().simple());
        let service = PostgresRunApplicationService::from_dsn(&dsn, &schema).unwrap();
        service.ensure_schema().await.unwrap();
        service
            .store
            .create_graph_schema(3, kogwistar_store_postgres::GraphTableNames::default())
            .await
            .unwrap();
        service
            .wisdom_store
            .create_graph_schema(3, kogwistar_store_postgres::GraphTableNames::default())
            .await
            .unwrap();
        let calls = Arc::new(Mutex::new(Vec::new()));
        let profile = kogwistar_engine::embedding::EmbeddingProfile {
            provider: "test".to_owned(),
            model: "wisdom-fixed-v1".to_owned(),
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
        let service = service.with_embedding_provider(Arc::new(FixedEmbeddingProvider {
            profile: profile.clone(),
            vector: vec![0.0, 0.0, 1.0],
            calls: calls.clone(),
            fail: None,
        }));
        service.ensure_embedding_profile_binding(&profile).await.unwrap();
        let primary_embedding_binding = service
            .store
            .get_named_projection(EMBEDDING_PROFILE_NAMESPACE, service.embedding_storage_scope.as_ref().unwrap())
            .await
            .unwrap();
        let wisdom_embedding_binding = service
            .wisdom_store
            .get_named_projection(EMBEDDING_PROFILE_NAMESPACE, service.wisdom_embedding_storage_scope.as_ref().unwrap())
            .await
            .unwrap();
        assert!(primary_embedding_binding.is_some());
        assert!(wisdom_embedding_binding.is_some());

        let mut graph_batch: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/graph-mutation-batch-v1.json"
        )))
        .unwrap();
        let graph_node_id = "regular-semantic-node";
        let graph_node = json!({
            "id": graph_node_id,
            "label": "Rust semantic retrieval fixture",
            "type": "concept",
            "summary": "A normal graph node used to verify semantic MCP retrieval.",
            "metadata": {"namespace": "docs", "workspace_id": null, "graph_space": null}
        });
        graph_batch["mutations"][0]["scope"] = json!({
            "namespace": "docs", "workspace_id": null, "graph_space": null
        });
        graph_batch["mutations"][0]["event_id"] = json!("regular-semantic-event");
        graph_batch["mutations"][0]["payload"] = graph_node.clone();
        graph_batch["mutations"][0]["record"]["id"] = json!(graph_node_id);
        graph_batch["mutations"][0]["record"]["document"] =
            json!(serde_json::to_string(&graph_node).unwrap());
        graph_batch["mutations"][0]["record"]["embedding"] = json!([0.0, 0.0, 1.0]);
        graph_batch["mutations"][0]["embedding_dim"] = json!(3);
        let mut acl_mutation = graph_batch["mutations"][0].clone();
        let acl_metadata = json!({
            "entity_type": "acl_record",
            "acl_truth_graph": "knowledge",
            "acl_target_entity_id": graph_node_id,
            "acl_target_grain": "node",
            "acl_version": 1,
            "acl_mode": "private",
            "owner_id": "alice",
            "tombstoned": false
        });
        let acl_id = "regular-semantic-node-acl";
        acl_mutation["event_id"] = json!("regular-semantic-acl-event");
        acl_mutation["payload"] = json!({"id": acl_id, "metadata": acl_metadata});
        acl_mutation["record"] = json!({
            "id": acl_id,
            "document": serde_json::to_string(&json!({"id": acl_id})).unwrap(),
            "metadata": acl_metadata,
            "embedding": null
        });
        acl_mutation["embedding_dim"] = json!(0);
        graph_batch["mutations"]
            .as_array_mut()
            .unwrap()
            .push(acl_mutation);
        let graph_write = service.execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: INTERNAL_GRAPH_MUTATIONS_ROUTE.to_owned(),
            body: serde_json::to_vec(&graph_batch).unwrap(),
            principal: json!({
                "sub": "alice", "user_id": "alice", "role": "rw",
                "ns": ["docs"], "storage_ns": "docs",
                "capabilities": ["write_graph"]
            }),
        }).await;
        assert_eq!(graph_write.status, StatusCode::OK.as_u16(), "{}", String::from_utf8_lossy(&graph_write.body));
        let inserted_graph_node = service.store.graph_projection_records(GraphProjectionRead {
            scope: GraphScope {
                namespace: "docs".to_owned(),
                workspace_id: None,
                graph_space: None,
            },
            table: "gke_nodes".to_owned(),
            ids: Some(vec![graph_node_id.to_owned()]),
            metadata: MetadataFilter::default(),
            limit: 1,
        }).await.unwrap();
        assert_eq!(inserted_graph_node.len(), 1, "graph mutation did not create projection");
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/python-graph-model-payloads-v1.json"
        )))
        .unwrap();
        let mut node = fixture["cases"][0]["payload"].clone();
        node["id"] = json!("wisdom-private-node");
        node["embedding"] = json!([1.0, 0.0, 0.0]);
        let mut edge = fixture["cases"][1]["payload"].clone();
        edge["id"] = json!("wisdom-private-edge");
        edge["source_ids"] = json!(["wisdom-private-node"]);
        edge["target_ids"] = json!([]);
        edge["embedding"] = json!([0.0, 1.0, 0.0]);
        let alice = json!({
            "sub": "alice", "user_id": "alice", "role": "rw",
            "ns": ["wisdom"], "storage_ns": "wisdom",
            "capabilities": ["write_graph"]
        });
        let response = service.execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: PUBLIC_WISDOM_GRAPH_UPSERT_ROUTE.to_owned(),
            body: serde_json::to_vec(&json!({"inp": {"nodes": [node.clone()], "edges": [edge.clone()]}})).unwrap(),
            principal: alice,
        }).await;
        assert_eq!(response.status, StatusCode::OK.as_u16(), "{}", String::from_utf8_lossy(&response.body));
        assert_eq!(
            *calls.lock().unwrap(),
            vec![
                vec![
                    "Python node: A complete Pydantic node payload crosses the Rust persistence seam."
                        .to_owned(),
                    "supports: The typed edge preserves both endpoint and source-span data."
                        .to_owned(),
                ],
            ]
        );
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
        use tower::ServiceExt;

        let secret = "wisdom-semantic-mcp-test-secret";
        let token = encode(
            &Header::new(Algorithm::HS256),
            &json!({
                "sub": "alice", "user_id": "alice", "role": "ro", "roles": ["reader"],
                "ns": ["wisdom"], "storage_ns": "wisdom",
                "capabilities": ["read_graph"], "exp": 4102444800_u64
            }),
            &EncodingKey::from_secret(secret.as_bytes()),
        )
        .unwrap();
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: vec!["reader".to_owned()],
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig {
                    mode: "jwt".to_owned(),
                    algorithm: Some("HS256".to_owned()),
                    key: Some(secret.to_owned()),
                    ..AuthConfig::default()
                },
            },
            Arc::new(service.clone()),
        );
        let semantic_response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header("authorization", format!("Bearer {token}"))
                    .body(Body::from(serde_json::to_vec(&json!({
                        "jsonrpc": "2.0", "id": 7, "method": "tools/call",
                        "params": {
                            "name": "wisdom.semantic_seed_then_expand",
                            "arguments": {"text": "find the node by meaning", "top_k": 5, "hops": 1}
                        }
                    })).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(semantic_response.status(), StatusCode::OK);
        let semantic_body = to_bytes(semantic_response.into_body(), 65_536).await.unwrap();
        let semantic: Value = serde_json::from_slice(&semantic_body).unwrap();
        assert!(semantic.get("error").is_none(), "{semantic}");
        let semantic = &semantic["result"];
        assert_eq!(semantic["seeds"].as_array().unwrap().len(), 1);
        let seed: Value = serde_json::from_str(semantic["seeds"][0].as_str().unwrap()).unwrap();
        assert_eq!(seed["id"], "wisdom-private-node");
        assert_eq!(semantic["layers"].as_array().unwrap().len(), 1);
        let edge_docs = semantic["layers"][0]["edges"].as_array().unwrap();
        assert_eq!(edge_docs.len(), 1);
        let edge_doc: Value = serde_json::from_str(edge_docs[0].as_str().unwrap()).unwrap();
        assert_eq!(edge_doc["id"], "wisdom-private-edge");

        let graph_semantic_response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header("authorization", format!("Bearer {}", encode(
                        &Header::new(Algorithm::HS256),
                        &json!({
                            "sub": "alice", "user_id": "alice", "role": "ro", "roles": ["reader"],
                            "ns": ["docs"], "storage_ns": "docs",
                            "capabilities": ["read_graph"], "exp": 4102444800_u64
                        }),
                        &EncodingKey::from_secret(secret.as_bytes()),
                    ).unwrap()))
                    .body(Body::from(serde_json::to_vec(&json!({
                        "jsonrpc": "2.0", "id": 8, "method": "tools/call",
                        "params": {
                            "name": "kg_semantic_seed_then_expand_text",
                            "arguments": {"text": "find the fixture concept by meaning"}
                        }
                    })).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(graph_semantic_response.status(), StatusCode::OK);
        let graph_semantic_body = to_bytes(graph_semantic_response.into_body(), 65_536).await.unwrap();
        let graph_semantic: Value = serde_json::from_slice(&graph_semantic_body).unwrap();
        assert!(graph_semantic.get("error").is_none(), "{graph_semantic}");
        let graph_semantic = &graph_semantic["result"];
        let graph_seed = graph_semantic["seeds"].as_array().unwrap();
        assert_eq!(graph_seed.len(), 1, "regular semantic result: {graph_semantic}");
        assert_eq!(serde_json::from_str::<Value>(graph_seed[0].as_str().unwrap()).unwrap()["id"], graph_node_id);

        let wisdom_nodes = service.wisdom_store.graph_projection_records(GraphProjectionRead {
            scope: GraphScope { namespace: "wisdom".to_owned(), workspace_id: None, graph_space: None },
            table: "gke_nodes".to_owned(), ids: Some(vec!["wisdom-private-node".to_owned()]),
            metadata: MetadataFilter::default(), limit: 1,
        }).await.unwrap();
        assert_eq!(wisdom_nodes.len(), 1);
        assert_eq!(wisdom_nodes[0].embedding.as_deref(), Some([0.0, 0.0, 1.0].as_slice()));
        let primary_nodes = service.store.graph_projection_records(GraphProjectionRead {
            scope: GraphScope { namespace: "wisdom".to_owned(), workspace_id: None, graph_space: None },
            table: "gke_nodes".to_owned(), ids: Some(vec!["wisdom-private-node".to_owned()]),
            metadata: MetadataFilter::default(), limit: 1,
        }).await.unwrap();
        assert!(primary_nodes.is_empty(), "wisdom writes stay in the isolated schema");

        let bob = json!({
            "sub": "bob", "user_id": "bob", "role": "rw",
            "ns": ["wisdom"], "storage_ns": "wisdom",
            "capabilities": ["write_graph"]
        });
        let overwrite = service.execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: PUBLIC_WISDOM_GRAPH_UPSERT_ROUTE.to_owned(),
            body: serde_json::to_vec(&json!({"inp": {"nodes": [node.clone()], "edges": []}})).unwrap(),
            principal: bob,
        }).await;
        assert_ne!(overwrite.status, StatusCode::OK.as_u16());

        let mut hidden_reference_edge = edge.clone();
        hidden_reference_edge["id"] = json!("bob-reference-edge");
        hidden_reference_edge["source_ids"] = json!(["wisdom-private-node"]);
        hidden_reference_edge["target_ids"] = json!([]);
        let hidden_reference = service.execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: PUBLIC_WISDOM_GRAPH_UPSERT_ROUTE.to_owned(),
            body: serde_json::to_vec(&json!({"inp": {"nodes": [], "edges": [hidden_reference_edge]}})).unwrap(),
            principal: json!({
                "sub": "bob", "user_id": "bob", "role": "rw", "ns": ["wisdom"],
                "storage_ns": "wisdom", "capabilities": ["write_graph"]
            }),
        }).await;
        assert_ne!(hidden_reference.status, StatusCode::OK.as_u16());

        let missing_endpoint = service.execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: PUBLIC_WISDOM_GRAPH_UPSERT_ROUTE.to_owned(),
            body: serde_json::to_vec(&json!({"inp": {"nodes": [], "edges": [{
                "id": "missing-endpoint-edge", "label": "references", "type": "relationship",
                "summary": "references a missing node", "relation": "references",
                "source_ids": ["absent-wisdom-node"], "target_ids": [],
                "source_edge_ids": [], "target_edge_ids": [], "metadata": {},
                "mentions": [], "embedding": null
            }]}})).unwrap(),
            principal: json!({
                "sub": "alice", "user_id": "alice", "role": "rw", "ns": ["wisdom"],
                "storage_ns": "wisdom", "capabilities": ["write_graph"]
            }),
        }).await;
        assert_eq!(missing_endpoint.status, StatusCode::UNPROCESSABLE_ENTITY.as_u16());

        let shared_service = PostgresRunApplicationService::from_dsn_with_wisdom_schema(
            &dsn,
            &schema,
            &schema,
        )
        .unwrap()
        .with_embedding_provider(Arc::new(FixedEmbeddingProvider {
            profile: profile.clone(),
            vector: vec![0.0, 0.0, 1.0],
            calls: Arc::new(Mutex::new(Vec::new())),
            fail: None,
        }));
        shared_service.ensure_schema().await.unwrap();
        shared_service
            .ensure_embedding_profile_binding(&profile)
            .await
            .unwrap();
        let mut shared_node = node.clone();
        shared_node["id"] = json!("wisdom-shared-schema-node");
        let shared_write = shared_service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: PUBLIC_WISDOM_GRAPH_UPSERT_ROUTE.to_owned(),
                body: serde_json::to_vec(
                    &json!({"inp": {"nodes": [shared_node], "edges": []}}),
                )
                .unwrap(),
                principal: json!({
                    "sub": "alice", "user_id": "alice", "role": "rw",
                    "ns": ["wisdom"], "storage_ns": "wisdom",
                    "capabilities": ["write_graph"]
                }),
            })
            .await;
        assert_eq!(
            shared_write.status,
            StatusCode::OK.as_u16(),
            "{}",
            String::from_utf8_lossy(&shared_write.body)
        );
        let shared_role_records = shared_service
            .store
            .graph_projection_records(GraphProjectionRead {
                scope: GraphScope {
                    namespace: "wisdom".to_owned(),
                    workspace_id: None,
                    graph_space: None,
                },
                table: "gke_nodes".to_owned(),
                ids: Some(vec!["wisdom-shared-schema-node".to_owned()]),
                metadata: MetadataFilter::default(),
                limit: 1,
            })
            .await
            .unwrap();
        assert_eq!(shared_role_records.len(), 1);
        let other_namespace_records = shared_service
            .store
            .graph_projection_records(GraphProjectionRead {
                scope: GraphScope {
                    namespace: "knowledge".to_owned(),
                    workspace_id: None,
                    graph_space: None,
                },
                table: "gke_nodes".to_owned(),
                ids: Some(vec!["wisdom-shared-schema-node".to_owned()]),
                metadata: MetadataFilter::default(),
                limit: 1,
            })
            .await
            .unwrap();
        assert!(other_namespace_records.is_empty());

        drop(service);
        drop(shared_service);
        let (cleanup, connection) = tokio_postgres::connect(&dsn, tokio_postgres::NoTls).await.unwrap();
        let cleanup_task = tokio::spawn(connection);
        cleanup.batch_execute(&format!(
            "DROP SCHEMA \"{schema}\" CASCADE; DROP SCHEMA \"{schema}_wisdom\" CASCADE"
        )).await.unwrap();
        drop(cleanup);
        cleanup_task.await.unwrap().unwrap();
    }

    #[test]
    fn wisdom_mcp_tool_discovery_requires_authorized_wisdom_write_scope() {
        let service = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "wisdom_tool_discovery",
        )
        .unwrap();
        let reader = json!({
            "role": "ro", "ns": ["wisdom"], "capabilities": ["read_graph"]
        });
        assert!(!service.supports_mcp_tool_for_principal(
            "root",
            "wisdom.kg_upsert_graph",
            &json!({"role": "rw", "ns": ["wisdom"], "capabilities": ["write_graph"]}),
        ));
        assert!(!service.supports_mcp_tool_for_principal(
            "root",
            "wisdom.semantic_seed_then_expand",
            &reader,
        ));
        assert!(!service.supports_mcp_tool_for_principal(
            "root",
            "kg_semantic_seed_then_expand_text",
            &json!({"role": "ro", "ns": ["docs"], "capabilities": ["read_graph"]}),
        ));
        let profile = kogwistar_engine::embedding::EmbeddingProfile {
            provider: "test".to_owned(),
            model: "wisdom-discovery-v1".to_owned(),
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
        let service = service.with_embedding_provider(Arc::new(FixedEmbeddingProvider {
            profile,
            vector: vec![0.0, 0.0, 1.0],
            calls: Arc::new(Mutex::new(Vec::new())),
            fail: None,
        }));
        for principal in [
            json!({"role": "rw", "ns": ["wisdom"], "capabilities": ["read_graph"]}),
            json!({"role": "ro", "ns": ["wisdom"], "capabilities": ["write_graph"]}),
            json!({"role": "rw", "ns": ["docs"], "capabilities": ["write_graph"]}),
        ] {
            assert!(!service.supports_mcp_tool_for_principal(
                "root",
                "wisdom.kg_upsert_graph",
                &principal,
            ));
        }
        let authorized = json!({
            "role": "rw", "ns": ["wisdom"], "capabilities": ["write_graph"]
        });
        assert!(service.supports_mcp_tool_for_principal(
            "root",
            "wisdom.kg_upsert_graph",
            &authorized,
        ));
        assert!(!service.supports_mcp_tool_for_principal(
            "root",
            "wisdom.semantic_seed_then_expand",
            &authorized,
        ));
        assert!(service.supports_mcp_tool_for_principal(
            "root",
            "wisdom.semantic_seed_then_expand",
            &reader,
        ));
        let graph_reader = json!({
            "role": "ro", "ns": ["docs"], "storage_ns": "docs",
            "capabilities": ["read_graph"]
        });
        assert!(service.supports_mcp_tool_for_principal(
            "root",
            "kg_semantic_seed_then_expand_text",
            &graph_reader,
        ));
        assert!(!service.supports_mcp_tool_for_principal(
            "root",
            "kg_semantic_seed_then_expand_text",
            &json!({"role": "ro", "ns": ["docs"], "capabilities": []}),
        ));
        assert!(!service.supports_mcp_tool_for_principal(
            "conversation",
            "kg_semantic_seed_then_expand_text",
            &graph_reader,
        ));
        assert!(!service.supports_mcp_tool_for_principal(
            "root",
            "wisdom.semantic_seed_then_expand",
            &json!({"role": "ro", "ns": ["other"], "capabilities": ["read_graph"]}),
        ));
        assert!(!service.supports_mcp_tool_for_principal(
            "conversation",
            "wisdom.kg_upsert_graph",
            &authorized,
        ));
    }

    #[test]
    fn graph_read_mcp_discovery_matches_namespace_and_capability_admission() {
        let service = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "graph_read_discovery",
        )
        .unwrap();
        let tools = [
            "kg_find_edges",
            "kg_k_hop",
            "kg_neighbors",
            "kg_shortest_path",
        ];
        let authorized_reader = json!({
            "role": "ro", "ns": ["docs"], "capabilities": ["read_graph"]
        });
        for tool in tools {
            assert!(
                service.supports_mcp_tool_for_principal("root", tool, &authorized_reader),
                "authorized docs reader should discover {tool}"
            );
            assert!(!service.supports_mcp_tool_for_principal(
                "root",
                tool,
                &json!({"role": "ro", "ns": ["docs"], "capabilities": []}),
            ));
            assert!(!service.supports_mcp_tool_for_principal(
                "root",
                tool,
                &json!({"role": "ro", "ns": ["workflow"], "capabilities": ["read_graph"]}),
            ));
            assert!(!service.supports_mcp_tool_for_principal(
                "root",
                tool,
                &json!({"role": "rw", "ns": ["docs"], "capabilities": []}),
            ));
            assert!(!service.supports_mcp_tool_for_principal(
                "conversation",
                tool,
                &authorized_reader,
            ));
        }
    }
#[test]
fn provider_event_publication_requires_matching_private_graph_event_and_conversation_edge() {
    let legacy_effect: RecordedWorkerSuccessEffect = serde_json::from_value(json!({
        "contract_version": 1,
        "effect_id": "legacy-effect"
    }))
    .unwrap();
    assert!(legacy_effect.provider_event_publications.is_empty());
    assert!(legacy_effect.provider_event_acknowledgements.is_empty());

    let publication_value = json!({
        "namespace": "tenant-a",
        "conversation_id": "conversation-a",
        "event": {
            "event_id": "agent-event:fixture-001",
            "trace": {
                "run_id": "run-a",
                "turn_id": "turn-a",
                "workflow_node_id": "wf:agent.event.v1:request",
                "step_id": "provider-response:response-a",
                "provider_id": "openai-responses",
                "provider_sequence": 3,
                "client_sse_sequence": 12
            },
            "event": {
                "contract_version": "v1",
                "provider_event_id": "response-a:1",
                "provider_sequence": 1,
                "response_id": "response-a",
                "kind": "field_completed",
                "field_name": "text",
                "value": "safe answer"
            }
        }
    });
    let publication: kogwistar_contracts::DurableProviderEventPublicationV1 =
        serde_json::from_value(publication_value).unwrap();
    let message_id = format!("msg:{}", publication.event.event_id);
    let payload_json = serde_json::to_string(&publication.event).unwrap();
    assert_eq!(publication.event.trace.client_sse_sequence, Some(12));
    assert!(payload_json.contains("\"client_sse_sequence\":12"));
    let metadata = serde_json::from_value(json!({
        "artifact_kind": "lane_message",
        "conversation_id": publication.conversation_id,
        "inbox_id": "agent-provider-events",
        "recipient_id": "agent-event-consumer",
        "sender_id": "agent-runtime",
        "msg_type": "agent.provider_event",
        "purpose": "agent_event",
        "run_id": "run-a",
        "step_id": "provider-response:response-a",
        "correlation_id": publication.event.event_id,
        "idempotency_key": publication.event.event_id,
        "visibility": "private",
        "shared_scope": false,
        "payload_json": payload_json
    }))
    .unwrap();
    let node = GraphMutation {
        scope: GraphScope {
            namespace: "tenant-a".to_owned(),
            workspace_id: None,
            graph_space: Some("conversation".to_owned()),
        },
        table: "gke_nodes".to_owned(),
        entity_kind: "node".to_owned(),
        event_id: "node-event-1".to_owned(),
        op: "ADD".to_owned(),
        payload: json!({"id": message_id}),
        record: GraphRecord {
            id: message_id.clone(),
            document: None,
            metadata,
            embedding: None,
        },
        embedding_dim: 0,
    };
    let edge = GraphMutation {
        scope: node.scope.clone(),
        table: "gke_edges".to_owned(),
        entity_kind: "edge".to_owned(),
        event_id: "edge-event-1".to_owned(),
        op: "ADD".to_owned(),
        payload: json!({
            "id": "event-in-conversation",
            "source_ids": [message_id],
            "target_ids": ["conversation-start-a"],
            "relation": "in_conversation"
        }),
        record: GraphRecord {
            id: "event-in-conversation".to_owned(),
            document: None,
            metadata: serde_json::from_value(json!({
                "conversation_id": publication.conversation_id
            }))
            .unwrap(),
            embedding: None,
        },
        embedding_dim: 0,
    };
    let mutations = vec![node, edge];
    assert!(validate_provider_event_publications(
        std::slice::from_ref(&publication),
        &mutations,
        "run-a",
    )
    .is_ok());
    assert!(validate_provider_event_publications(
        std::slice::from_ref(&publication),
        &mutations[..1],
        "run-a",
    )
    .is_err());
    let mut cross_scope_edge = mutations.clone();
    cross_scope_edge[1].scope.namespace = "other-tenant".to_owned();
    assert!(validate_provider_event_publications(
        std::slice::from_ref(&publication),
        &cross_scope_edge,
        "run-a",
    )
    .is_err());
    let mut ambiguous_edge = mutations.clone();
    ambiguous_edge[1].payload["target_ids"] = json!(["conversation-a", "conversation-b"]);
    assert!(validate_provider_event_publications(
        std::slice::from_ref(&publication),
        &ambiguous_edge,
        "run-a",
    )
    .is_err());
    assert!(validate_provider_event_publications(
        std::slice::from_ref(&publication),
        &mutations,
        "another-run",
    )
    .is_err());
    let mut cross_namespace = publication.clone();
    cross_namespace.namespace = "other-tenant".to_owned();
    assert!(validate_provider_event_publications(
        &[cross_namespace],
        &mutations,
        "run-a",
    )
    .is_err());
    let mut transient_delta = publication.clone();
    transient_delta.event.event = serde_json::from_value(json!({
        "kind": "text_delta",
        "delta": "not durable"
    }))
    .unwrap();
    assert!(validate_provider_event_publications(
        &[transient_delta],
        &mutations,
        "run-a",
    )
    .is_err());
}
#[test]
fn provider_event_ack_graph_mutations_link_event_to_ack_step() {
    let scope = GraphScope {
        namespace: "tenant-a".to_owned(),
        workspace_id: Some("workspace-a".to_owned()),
        graph_space: Some("conversation".to_owned()),
    };
    let mutations = provider_event_ack_graph_mutations(
        "event-a",
        "run-a",
        "conversation-a",
        "agent.event.v1",
        "handle-event",
        7,
        "effect-a",
        &scope,
    );

    assert_eq!(mutations.len(), 2);
    let step = &mutations[0];
    assert_eq!(step.entity_kind, "node");
    assert_eq!(step.record.id, "wf_step|run-a|7");
    assert_eq!(step.record.metadata["entity_type"], "workflow_step_exec");
    assert_eq!(step.record.metadata["workflow_node_id"], "handle-event");
    assert_eq!(
        step.record.metadata["result_json"],
        "{\"event_id\":\"event-a\",\"handled\":true}"
    );
    assert_eq!(step.payload["summary"], "step=7, op=handle-event, status=ok");
    assert_eq!(step.scope, scope);

    let edge = &mutations[1];
    assert_eq!(edge.entity_kind, "edge");
    assert_eq!(edge.payload["relation"], "acknowledged_by_step");
    assert_eq!(edge.payload["source_ids"], json!(["msg:event-a"]));
    assert_eq!(edge.payload["target_ids"], json!(["wf_step|run-a|7"]));
    assert_eq!(edge.record.metadata["event_id"], "event-a");
    assert_eq!(edge.record.metadata["step_seq"], 7);
    assert_eq!(edge.scope, scope);

    let replay = provider_event_ack_graph_mutations(
        "event-a",
        "run-a",
        "conversation-a",
        "agent.event.v1",
        "handle-event",
        7,
        "effect-a",
        &scope,
    );
    assert_eq!(mutations, replay, "replay uses deterministic graph identities");
}

#[test]
fn provider_event_links_match_python_producer_and_handler_contract() {
    let scope = GraphScope {
        namespace: "tenant-a".to_owned(),
        workspace_id: Some("workspace-a".to_owned()),
        graph_space: Some("conversation".to_owned()),
    };
    let producer = ProviderEventStepSnapshot {
        step_seq: 3,
        workflow_id: "workflow-a".to_owned(),
        workflow_node_id: "provider.stream".to_owned(),
        status: "success".to_owned(),
        result_json: "{\"published\":true}".to_owned(),
        event_seq: 8,
        transition_id: "transition-producer".to_owned(),
    };
    let handler = ProviderEventStepSnapshot {
        step_seq: 5,
        workflow_id: "workflow-a".to_owned(),
        workflow_node_id: "agent.event.tool".to_owned(),
        status: "success".to_owned(),
        result_json: "{\"tool\":\"read\"}".to_owned(),
        event_seq: 12,
        transition_id: "transition-handler".to_owned(),
    };
    let snapshots = BTreeMap::from([(3, producer), (5, handler)]);
    let mutations = provider_event_step_link_mutations(
        "event-a",
        "run-a",
        "conversation-a",
        Some(3),
        5,
        &snapshots,
        &scope,
    );

    assert_eq!(mutations.len(), 4);
    assert_eq!(mutations[0].record.id, "wf_step|run-a|3");
    assert_eq!(mutations[1].record.id, "wf_step|run-a|5");
    assert_eq!(mutations[0].record.metadata["event_seq"], 8);
    assert_eq!(mutations[1].record.metadata["transition_id"], "transition-handler");
    assert_eq!(mutations[2].payload["relation"], "emitted_by_step");
    assert_eq!(mutations[2].payload["source_ids"], json!(["msg:event-a"]));
    assert_eq!(mutations[2].payload["target_ids"], json!(["wf_step|run-a|3"]));
    assert_eq!(mutations[3].payload["relation"], "handled_by_step");
    assert_eq!(mutations[3].payload["target_ids"], json!(["wf_step|run-a|5"]));
    assert_ne!(mutations[2].record.id, mutations[3].record.id);
    assert_eq!(mutations[0].scope, scope);

    let replay = provider_event_step_link_mutations(
        "event-a",
        "run-a",
        "conversation-a",
        Some(3),
        5,
        &snapshots,
        &scope,
    );
    assert_eq!(mutations, replay, "replay preserves deterministic step/link records");

    let shared_step = BTreeMap::from([(3, snapshots[&3].clone())]);
    let shared_mutations = provider_event_step_link_mutations(
        "event-b",
        "run-a",
        "conversation-a",
        Some(3),
        3,
        &shared_step,
        &scope,
    );
    assert_eq!(
        shared_mutations.iter().filter(|mutation| mutation.entity_kind == "node").count(),
        1,
        "producer and handler sharing a step reuse one step node"
    );
    assert_eq!(
        shared_mutations.iter().filter(|mutation| mutation.entity_kind == "edge").count(),
        2
    );
}

#[test]
fn provider_event_step_snapshot_lookup_rejects_wrong_scope_and_missing_steps() {
    let event = |seq: i64, run_id: &str| ServerRunEvent {
        seq,
        run_id: run_id.to_owned(),
        event_type: "workflow.recorded_transition.v1".to_owned(),
        payload_json: json!({
            "transition_id": format!("transition-{seq}"),
            "reduced": {
                "state": {
                    "run_id": run_id,
                    "conversation_id": "conversation-a",
                    "workflow_id": "workflow-a",
                    "last_step_seq": seq,
                    "last_node_id": "agent.event.receive"
                },
                "errors": [],
                "result": {"ok": true}
            }
        })
        .to_string(),
        created_at_ms: 100 + seq,
    };

    let found = provider_event_step_snapshots(
        vec![event(4, "run-a")],
        "run-a",
        "conversation-a",
        "workflow-a",
        &[4],
    )
    .unwrap();
    assert_eq!(found[&4].workflow_node_id, "agent.event.receive");
    assert!(provider_event_step_snapshots(
        vec![event(4, "other-run")],
        "run-a",
        "conversation-a",
        "workflow-a",
        &[4],
    )
    .is_err());
    assert!(provider_event_step_snapshots(
        Vec::new(),
        "run-a",
        "conversation-a",
        "workflow-a",
        &[4],
    )
    .is_err());
}

#[test]
fn provider_event_acknowledgement_contract_rejects_invalid_or_duplicate_claims() {
    use kogwistar_runtime::ProviderEventAcknowledgementV1;

    let valid = ProviderEventAcknowledgementV1 {
        message_id: "msg:event-1".to_owned(),
        claimed_by: "consumer#lease-1".to_owned(),
        event_id: "event-1".to_owned(),
    };
    assert!(validate_provider_event_acknowledgements(std::slice::from_ref(&valid)).is_ok());
    assert!(validate_provider_event_acknowledgements(&[valid.clone(), valid.clone()]).is_err());
    assert!(validate_provider_event_acknowledgements(&[ProviderEventAcknowledgementV1 {
        message_id: "msg:other".to_owned(),
        claimed_by: "consumer#lease-1".to_owned(),
        event_id: "event-1".to_owned(),
    }])
    .is_err());
    assert!(validate_provider_event_acknowledgements(&[ProviderEventAcknowledgementV1 {
        message_id: "msg:event-1".to_owned(),
        claimed_by: " ".to_owned(),
        event_id: "event-1".to_owned(),
    }])
    .is_err());
    assert!(validate_provider_event_acknowledgements(
        &(0..101)
            .map(|index| ProviderEventAcknowledgementV1 {
                message_id: format!("msg:event-{index}"),
                claimed_by: "consumer#lease-1".to_owned(),
                event_id: format!("event-{index}"),
            })
            .collect::<Vec<_>>()
    )
    .is_err());
}

#[tokio::test]
#[ignore = "requires KOGWISTAR_TEST_PG_DSN; run explicitly against disposable PostgreSQL"]
async fn postgres_provider_event_ack_persists_python_causal_links_atomically() {
    use crate::native_answer_generation::OpenAiCompatibleChatConfig;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN").unwrap();
    let schema = format!("provider_event_causal_{}", uuid::Uuid::new_v4().simple());
    let auth_path = std::env::temp_dir().join(format!(
        "kogwistar-provider-event-causal-auth-{}.sqlite",
        uuid::Uuid::new_v4().simple()
    ));
    let auth = SqliteAuthStore::open(&auth_path).unwrap();
    auth.resolve_external_identity(ResolveExternalIdentity {
        issuer: "test".to_owned(),
        subject: "causal-owner".to_owned(),
        email: "causal-owner@example.test".to_owned(),
        display_name: None,
        new_user_id: "causal-owner".to_owned(),
        default_role: "rw".to_owned(),
        default_ns: "alpha".to_owned(),
    })
    .await
    .unwrap();
    drop(auth);
    let owner_auth = AuthConfig {
        auth_db_url: Some(format!("sqlite:///{}", auth_path.display())),
        ..AuthConfig::default()
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let provider_address = listener.local_addr().unwrap();
    let provider_task = tokio::spawn(async move {
        for _ in 0..2 {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let (header_end, content_length) = loop {
                let mut chunk = [0_u8; 4096];
                let count = stream.read(&mut chunk).await.unwrap();
                assert_ne!(count, 0);
                request.extend_from_slice(&chunk[..count]);
                if let Some(header_end) = request.windows(4).position(|part| part == b"\r\n\r\n") {
                    let header_end = header_end + 4;
                    let headers = String::from_utf8_lossy(&request[..header_end]).to_ascii_lowercase();
                    let content_length = headers
                        .lines()
                        .find_map(|line| line.strip_prefix("content-length: "))
                        .unwrap()
                        .trim()
                        .parse::<usize>()
                        .unwrap();
                    if request.len() >= header_end + content_length {
                        break (header_end, content_length);
                    }
                    while request.len() < header_end + content_length {
                        let mut chunk = [0_u8; 4096];
                        let count = stream.read(&mut chunk).await.unwrap();
                        assert_ne!(count, 0);
                        request.extend_from_slice(&chunk[..count]);
                    }
                    break (header_end, content_length);
                }
            };
            let body: Value = serde_json::from_slice(&request[header_end..header_end + content_length]).unwrap();
            assert_eq!(body["model"], "event-fixture-model");
            assert_eq!(body["stream"], true);
            let mut body = String::new();
            for chunk in [
                json!({"id":"causal-response","choices":[{"delta":{"content":"streamed text"},"finish_reason":null}]}).to_string(),
                json!({"id":"causal-response","choices":[{"delta":{},"finish_reason":"stop"}]}).to_string(),
            ] {
                body.push_str(&format!("data: {chunk}\r\n\r\n"));
            }
            body.push_str("data: [DONE]\r\n\r\n");
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(response.as_bytes()).await.unwrap();
        }
    });
    let provider = OpenAiCompatibleChatConfig::new(
        format!("http://{provider_address}/v1/chat/completions"),
        "event-fixture-model",
        None,
    )
    .unwrap()
    .with_streaming(true);
    let service = PostgresRunApplicationService::from_dsn(&dsn, &schema)
        .unwrap()
        .with_owner_auth_config(&owner_auth)
        .await
        .unwrap()
        .with_chat_completion_provider(provider);
    service.ensure_schema().await.unwrap();
    service
        .store
        .create_graph_schema(3, kogwistar_store_postgres::GraphTableNames::default())
        .await
        .unwrap();

    let run_id = "provider-event-causal-run";
    let workflow_id = "provider-event-causal-workflow";
    let conversation_id = "provider-event-causal-conversation";
    let principal = json!({
        "sub": "causal-owner",
        "user_id": "causal-owner",
        "role": "rw",
        "ns": ["alpha"],
        "storage_ns": "alpha",
        "capabilities": ["read_graph", "write_graph"]
    });
    let binding = runtime_owner_binding_v1(&principal, &service.capabilities).unwrap();
    service
        .store
        .create_server_run(ServerRunCreate {
            run_id: run_id.to_owned(),
            conversation_id: conversation_id.to_owned(),
            workflow_id: workflow_id.to_owned(),
            user_id: Some("causal-owner".to_owned()),
            user_turn_node_id: "causal-user-turn".to_owned(),
            status: "queued".to_owned(),
        })
        .await
        .unwrap();
    service
        .store
        .append_server_run_event(
            run_id,
            "run.created",
            json!({"runtime_owner_binding": binding}).to_string(),
        )
        .await
        .unwrap();

    let start: kogwistar_runtime::RecordedRuntimeTransition = serde_json::from_value(json!({
        "contract_version": 1,
        "transition_id": "causal-start",
        "expected_event_seq": 1,
        "kind": "start",
        "run_id": run_id,
        "workflow_id": workflow_id,
        "conversation_id": conversation_id,
        "user_id": "causal-owner",
        "user_turn_node_id": "causal-user-turn",
        "step_seq": 0,
        "node_id": "provider-step",
        "token_id": run_id,
        "initial_state": {},
        "frontier": {
            "pending": [["provider-step", 0, run_id, null]],
            "suspended": [], "join_node_ids": [], "join_outstanding": [], "join_waiters": {}
        }
    })).unwrap();
    service.store.apply_recorded_runtime_transition(start, false).await.unwrap();
    let provider_request = kogwistar_runtime::RuntimeNativeStepRequest {
        executor_target: kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET.to_owned(),
        operation: "agent.event.request".to_owned(),
        op_version: "v1".to_owned(),
        run_id: run_id.to_owned(),
        workflow_id: workflow_id.to_owned(),
        node_id: "provider-step".to_owned(),
        conversation_id: conversation_id.to_owned(),
        turn_node_id: Some("causal-user-turn".to_owned()),
        token_id: run_id.to_owned(),
        parent_token_id: None,
        step_seq: 0,
        state: serde_json::from_value(json!({
            "provider_request": {
                "contract_version": "v1",
                "model": "event-fixture-model",
                "instructions": "test only",
                "input": "emit one response"
            }
        }))
        .unwrap(),
        runtime_routes: Vec::new(),
        ephemeral_context: Map::from_iter([("_trace_attempt".to_owned(), json!(2))]),
    };
    let provider_result = service
        .execute_native_provider_event_request(
            kogwistar_runtime::RecordedWorkerHandoff {
                message_id: "provider-request-handoff".to_owned(),
                claimed_by: "provider-request-worker#1".to_owned(),
                run_id: run_id.to_owned(),
                step_id: "provider-step".to_owned(),
                correlation_id: run_id.to_owned(),
            },
            provider_request.clone(),
        )
        .await
        .unwrap();
    let provider_effect = provider_result.effect.expect("provider request returns an effect");
    let provider_result_json = provider_effect.result.as_ref().unwrap();
    assert!(provider_result_json["semantic_event_count"].as_i64().unwrap() > 0);
    assert_eq!(provider_result_json["final_state"]["agent_event_route"], "receive");
    let events_after_first_attempt = service.store.server_run_events(run_id, 0, 100).await.unwrap();
    let provider_retry = service
        .execute_native_provider_event_request(
            kogwistar_runtime::RecordedWorkerHandoff {
                message_id: "provider-request-handoff-retry".to_owned(),
                claimed_by: "provider-request-worker#2".to_owned(),
                run_id: run_id.to_owned(),
                step_id: "provider-step".to_owned(),
                correlation_id: run_id.to_owned(),
            },
            provider_request,
        )
        .await
        .unwrap();
    provider_task.await.unwrap();
    let retry_effect = provider_retry.effect.expect("provider retry returns an effect");
    assert_eq!(provider_effect.result, retry_effect.result);
    let events_after_retry = service.store.server_run_events(run_id, 0, 100).await.unwrap();
    assert_eq!(events_after_first_attempt, events_after_retry);
    let producer_event_seq = service
        .store
        .server_run_events(run_id, 0, 100)
        .await
        .unwrap()
        .last()
        .unwrap()
        .seq;

    let producer: kogwistar_runtime::RecordedRuntimeTransition = serde_json::from_value(json!({
        "contract_version": 1,
        "transition_id": "causal-producer-step",
        "expected_event_seq": producer_event_seq,
        "kind": "recorded_step_success",
        "run_id": run_id,
        "workflow_id": workflow_id,
        "conversation_id": conversation_id,
        "step_seq": 0,
        "node_id": "provider-step",
        "token_id": run_id,
        "state_update": provider_effect.state_update,
        "frontier": {
            "pending": [["handler-step", 0, run_id, null]],
            "suspended": [], "join_node_ids": [], "join_outstanding": [], "join_waiters": {}
        },
        "result": provider_effect.result
    })).unwrap();
    service.store.apply_recorded_runtime_transition(producer, false).await.unwrap();
    let handler_event_seq = service
        .store
        .server_run_events(run_id, 0, 100)
        .await
        .unwrap()
        .last()
        .unwrap()
        .seq;
    let receive_request = kogwistar_runtime::RuntimeNativeStepRequest {
        executor_target: kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET.to_owned(),
        operation: "agent.event.receive".to_owned(),
        op_version: "v1".to_owned(),
        run_id: run_id.to_owned(),
        workflow_id: workflow_id.to_owned(),
        node_id: "receive-step".to_owned(),
        conversation_id: conversation_id.to_owned(),
        turn_node_id: Some("causal-user-turn".to_owned()),
        token_id: run_id.to_owned(),
        parent_token_id: None,
        step_seq: 1,
        state: Map::new(),
        runtime_routes: Vec::new(),
        ephemeral_context: Map::new(),
    };
    let receive_principal = service
        .native_read_principal(&receive_request, None)
        .await
        .unwrap_or_else(|_| panic!("the active run owner retains read_graph authority"));
    let claim = service
        .claim_native_provider_event_context(&receive_request, &receive_principal)
        .await
        .unwrap()
        .expect("published event is visible and claimable by its run owner");
    let event_id = claim["event"]["event_id"].as_str().unwrap().to_owned();
    assert_eq!(claim["event"]["event"]["kind"], "response_started");
    assert_eq!(claim["event"]["trace"]["step_id"], "provider-response:causal-response");
    assert_eq!(claim["event"]["trace"]["attempt"], 1);

    let handler: kogwistar_runtime::RecordedRuntimeTransition = serde_json::from_value(json!({
        "contract_version": 1,
        "transition_id": "causal-handler-step",
        "expected_event_seq": handler_event_seq,
        "kind": "recorded_step_success",
        "run_id": run_id,
        "workflow_id": workflow_id,
        "conversation_id": conversation_id,
        "step_seq": 1,
        "node_id": "handler-step",
        "token_id": run_id,
        "state_update": [["u", {
            "agent_event_current": claim["event"],
            "agent_event_claim": claim["claim"],
            "agent_event_handling_step_seq": 1,
            "agent_event_route": "no_action"
        }]],
        "frontier": {
            "pending": [["ack-step", 0, run_id, null]],
            "suspended": [], "join_node_ids": [], "join_outstanding": [], "join_waiters": {}
        },
        "result": {"handled": true}
    })).unwrap();
    service.store.apply_recorded_runtime_transition(handler, false).await.unwrap();

    service.store.project_lane_message(NewProjectedLaneMessage {
        message_id: "causal-ack-runtime-lane".to_owned(),
        namespace: "workflow".to_owned(),
        purpose: "system".to_owned(),
        inbox_id: "workflow-runtime".to_owned(),
        conversation_id: conversation_id.to_owned(),
        recipient_id: "rust-native-worker".to_owned(),
        sender_id: "rust-scheduler".to_owned(),
        msg_type: "workflow.step.execute".to_owned(),
        status: "pending".to_owned(),
        created_at: 1,
        available_at: 1,
        run_id: Some(run_id.to_owned()),
        step_id: Some("ack-step".to_owned()),
        correlation_id: Some(run_id.to_owned()),
        payload_json: Some(json!({
            "workflow_id": workflow_id,
            "conversation_id": conversation_id,
            "token_id": run_id,
            "parent_token_id": null,
            "op": "agent.event.ack"
        }).to_string()),
        error_json: None,
    }).await.unwrap();
    let claimed_runtime = service.store.claim_projected_lane_messages(
        "workflow", "workflow-runtime", "causal-worker", 1, 60,
    ).await.unwrap();
    assert_eq!(claimed_runtime.len(), 1);
    let runtime_lane = &claimed_runtime[0];
    let handoff = kogwistar_runtime::RecordedWorkerHandoff {
        message_id: runtime_lane.message_id.clone(),
        claimed_by: runtime_lane.claimed_by.clone().unwrap(),
        run_id: run_id.to_owned(),
        step_id: "ack-step".to_owned(),
        correlation_id: run_id.to_owned(),
    };
    let ack_request = kogwistar_runtime::RuntimeNativeStepRequest {
        executor_target: kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET.to_owned(),
        operation: "agent.event.ack".to_owned(),
        op_version: "v1".to_owned(),
        run_id: run_id.to_owned(),
        workflow_id: workflow_id.to_owned(),
        node_id: "ack-step".to_owned(),
        conversation_id: conversation_id.to_owned(),
        turn_node_id: Some("causal-user-turn".to_owned()),
        token_id: run_id.to_owned(),
        parent_token_id: None,
        step_seq: 2,
        state: Map::from_iter([
            ("agent_event_current".to_owned(), claim["event"].clone()),
            ("agent_event_claim".to_owned(), claim["claim"].clone()),
            ("agent_event_route".to_owned(), json!("no_action")),
            ("agent_event_handling_step_seq".to_owned(), json!(1)),
        ]),
        runtime_routes: Vec::new(),
        ephemeral_context: Map::new(),
    };
    let ack_effect = kogwistar_runtime::execute_native_step(&ack_request).unwrap();
    let request = |effect: kogwistar_runtime::RecordedWorkerSuccessEffect| RuntimeWorkerResultRequest {
        handoff: handoff.clone(),
        transition: None,
        effect: Some(effect),
    };

    let (fault_setup, fault_connection) = tokio_postgres::connect(&dsn, tokio_postgres::NoTls).await.unwrap();
    let fault_task = tokio::spawn(fault_connection);
    fault_setup.batch_execute(&format!(
        "CREATE FUNCTION \"{schema}\".reject_provider_event_handled() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN IF NEW.event_type='workflow.provider_event.handled.v1' THEN RAISE EXCEPTION 'injected causal ack failure'; END IF; RETURN NEW; END; $$; CREATE TRIGGER reject_provider_event_handled BEFORE INSERT ON \"{schema}\".server_run_events FOR EACH ROW EXECUTE FUNCTION \"{schema}\".reject_provider_event_handled();"
    )).await.unwrap();
    drop(fault_setup);
    fault_task.await.unwrap().unwrap();

    let failed = service.apply_runtime_result_input(
        request(ack_effect.clone()), None, None, kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET,
    ).await;
    assert_eq!(failed.status, StatusCode::CONFLICT.as_u16());
    assert_eq!(
        service
            .store
            .get_projected_lane_message(&format!("msg:{event_id}"))
            .await
            .unwrap()
            .unwrap()
            .status,
        "claimed"
    );
    assert_eq!(service.store.read_recorded_runtime_state(run_id, workflow_id, conversation_id).await.unwrap().unwrap().last_step_seq, 1);
    let scope = GraphScope { namespace: "alpha".to_owned(), workspace_id: None, graph_space: Some("conversation".to_owned()) };
    let causal_edges = service.store.graph_projection_records(GraphProjectionRead {
        scope: scope.clone(), table: "gke_edges".to_owned(), ids: None,
        metadata: MetadataFilter::default(), limit: 1000,
    }).await.unwrap();
    assert!(!causal_edges.iter().any(|record| record.document.as_deref().is_some_and(|doc| doc.contains("emitted_by_step") || doc.contains("handled_by_step"))));

    let (fault_cleanup, fault_connection) = tokio_postgres::connect(&dsn, tokio_postgres::NoTls).await.unwrap();
    let fault_cleanup_task = tokio::spawn(fault_connection);
    fault_cleanup.batch_execute(&format!(
        "DROP TRIGGER reject_provider_event_handled ON \"{schema}\".server_run_events; DROP FUNCTION \"{schema}\".reject_provider_event_handled();"
    )).await.unwrap();
    drop(fault_cleanup);
    fault_cleanup_task.await.unwrap().unwrap();

    let applied = service.apply_runtime_result_input(
        request(ack_effect.clone()), None, None, kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET,
    ).await;
    assert_eq!(
        applied.status,
        StatusCode::OK.as_u16(),
        "{}",
        String::from_utf8_lossy(&applied.body)
    );
    let events_before_exact_retry = service
        .store
        .list_server_run_events(run_id, 0, usize::MAX)
        .await
        .unwrap();
    let exact_retry = service
        .apply_runtime_result_input(
            request(ack_effect),
            None,
            None,
            kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET,
        )
        .await;
    assert_eq!(exact_retry.status, StatusCode::OK.as_u16());
    assert_eq!(
        service
            .store
            .list_server_run_events(run_id, 0, usize::MAX)
            .await
            .unwrap(),
        events_before_exact_retry,
        "an exact post-commit retry must not append another completion or handled event"
    );
    let committed_edges = service.store.graph_projection_records(GraphProjectionRead {
        scope, table: "gke_edges".to_owned(), ids: None,
        metadata: MetadataFilter::default(), limit: 1000,
    }).await.unwrap();
    let relations = committed_edges.iter().filter_map(|record| {
        record.document.as_deref().and_then(|doc| serde_json::from_str::<Value>(doc).ok())
            .and_then(|doc| doc["relation"].as_str().map(str::to_owned))
    }).collect::<Vec<_>>();
    assert!(relations.iter().any(|relation| relation == "emitted_by_step"));
    assert!(relations.iter().any(|relation| relation == "handled_by_step"));
    assert_eq!(
        service.store.get_projected_lane_message(&format!("msg:{event_id}")).await.unwrap().unwrap().status,
        "completed"
    );

    drop(service);
    let (cleanup, connection) = tokio_postgres::connect(&dsn, tokio_postgres::NoTls).await.unwrap();
    let cleanup_task = tokio::spawn(connection);
    cleanup.batch_execute(&format!("DROP SCHEMA \"{schema}\" CASCADE")).await.unwrap();
    drop(cleanup);
    cleanup_task.await.unwrap().unwrap();
    let _ = std::fs::remove_file(auth_path);
}
#[tokio::test]
async fn sqlite_run_admission_persists_authenticated_owner_binding() {
    let sqlite_path = std::env::temp_dir().join(format!(
        "kogwistar-sqlite-owner-binding-{}.sqlite",
        uuid::Uuid::new_v4().simple()
    ));
    let auth_path = std::env::temp_dir().join(format!(
        "kogwistar-sqlite-owner-binding-auth-{}.sqlite",
        uuid::Uuid::new_v4().simple()
    ));
    let auth = SqliteAuthStore::open(&auth_path).unwrap();
    auth.resolve_external_identity(ResolveExternalIdentity {
        issuer: "test".to_owned(),
        subject: "sqlite-owner".to_owned(),
        email: "sqlite-owner@example.test".to_owned(),
        display_name: None,
        new_user_id: "sqlite-owner".to_owned(),
        default_role: "rw".to_owned(),
        default_ns: "workflow".to_owned(),
    })
    .await
    .unwrap();
    drop(auth);

    let auth_config = AuthConfig {
        auth_db_url: Some(format!("sqlite:///{}", auth_path.display())),
        ..AuthConfig::default()
    };
    let service = SqliteRunApplicationService::open(&sqlite_path)
        .unwrap()
        .with_owner_auth_config(&auth_config)
        .await
        .unwrap();
    let principal = json!({
        "user_id": "sqlite-owner",
        "sub": "sqlite-owner@example.test",
        "role": "rw",
        "ns": ["workflow"],
        "storage_ns": "workflow",
        "capabilities": ["workflow.run.write", "write_graph"]
    });
    let response = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/workflow/runs".to_owned(),
            body: serde_json::to_vec(&json!({
                "run_id": "sqlite-owner-bound-run",
                "workflow_id": "sqlite-owner-bound-workflow",
                "conversation_id": "sqlite-owner-bound-conversation",
                "user_id": "sqlite-owner"
            }))
            .unwrap(),
            principal: principal.clone(),
        })
        .await;
    assert_eq!(response.status, StatusCode::ACCEPTED.as_u16());
    let run = service
        .store
        .get_server_run("sqlite-owner-bound-run")
        .unwrap()
        .unwrap();
    assert_eq!(run.user_id.as_deref(), Some("sqlite-owner"));
    let created = service
        .store
        .list_server_run_events("sqlite-owner-bound-run", 0, 1)
        .unwrap()
        .into_iter()
        .find(|event| event.event_type == "run.created")
        .unwrap();
    let payload: Value = serde_json::from_str(&created.payload_json).unwrap();
    assert_eq!(payload["runtime_owner_binding"]["version"], 1);
    assert_eq!(payload["runtime_owner_binding"]["user_id"], "sqlite-owner");
    assert!(payload["runtime_owner_binding"]["admitted_capabilities"]
        .as_array()
        .unwrap()
        .iter()
        .any(|capability| capability == "write_graph"));
    let revalidated = service
        .resolve_inline_native_owner_principal("sqlite-owner-bound-run", "write_graph")
        .await
        .unwrap();
    assert_eq!(authenticated_run_owner_id(&revalidated), Some("sqlite-owner"));
    assert!(principal_strings(&revalidated, "capabilities")
        .iter()
        .any(|capability| capability == "write_graph"));

    let mismatch = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/workflow/runs".to_owned(),
            body: serde_json::to_vec(&json!({
                "run_id": "sqlite-owner-mismatch-run",
                "workflow_id": "sqlite-owner-bound-workflow",
                "conversation_id": "sqlite-owner-bound-conversation",
                "user_id": "another-user"
            }))
            .unwrap(),
            principal,
        })
        .await;
    assert_eq!(mismatch.status, StatusCode::FORBIDDEN.as_u16());
    assert!(service
        .store
        .get_server_run("sqlite-owner-mismatch-run")
        .unwrap()
        .is_none());

    drop(service);
    remove_test_sqlite(&sqlite_path);
    remove_test_sqlite(&auth_path);
}

#[tokio::test]
async fn sqlite_chroma_graph_mutation_commits_canonical_events_and_eventual_jobs() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let (stop_chroma, mut stop_chroma_rx) = tokio::sync::oneshot::channel();
    let fake_chroma = tokio::spawn(async move {
        let mut projected = Vec::new();
        let mut fail_first_upsert = true;
        loop {
            let (mut stream, _) = tokio::select! {
                _ = &mut stop_chroma_rx => break,
                connection = listener.accept() => connection.unwrap(),
            };
            let mut bytes = Vec::new();
            let mut chunk = [0_u8; 2048];
            let (header_end, content_length) = loop {
                let count = stream.read(&mut chunk).await.unwrap();
                assert!(count > 0, "Chroma connection closed before request body");
                bytes.extend_from_slice(&chunk[..count]);
                if let Some(header_end) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
                    let headers = String::from_utf8_lossy(&bytes[..header_end]);
                    let content_length = headers
                        .lines()
                        .find_map(|line| {
                            let (name, value) = line.split_once(':')?;
                            name.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse::<usize>().ok())
                                .flatten()
                        })
                        .unwrap_or(0);
                    if bytes.len() >= header_end + 4 + content_length {
                        break (header_end, content_length);
                    }
                }
            };
            let request: Value = serde_json::from_slice(
                &bytes[header_end + 4..header_end + 4 + content_length],
            )
            .unwrap();
            let request_line = String::from_utf8_lossy(&bytes[..header_end])
                .lines()
                .next()
                .unwrap_or_default()
                .to_owned();
            let (status, response) = if request_line.contains("/collections HTTP") {
                (
                    "200 OK",
                    json!({
                        "id": format!("collection-{}", request["name"].as_str().unwrap()),
                        "metadata": {"hnsw:space":"cosine"}
                    }),
                )
            } else {
                projected.push((request_line, request));
                if fail_first_upsert
                    && projected.last().unwrap().0.contains("/upsert")
                    && projected.last().unwrap().1["ids"][0] == "node-1"
                {
                    fail_first_upsert = false;
                    (
                        "500 Internal Server Error",
                        json!({"error":"injected Chroma projection failure"}),
                    )
                } else {
                    ("200 OK", json!({}))
                }
            };
            let response = response.to_string();
            let wire = format!(
                "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                response.len(),
                response
            );
            stream.write_all(wire.as_bytes()).await.unwrap();
        }
        projected
    });

    let chroma_config = |collection: &str| kogwistar_store_chroma::ChromaStoreConfig {
        endpoint: endpoint.clone(),
        collection: collection.to_owned(),
        ..Default::default()
    };
    let graph = kogwistar_store_chroma::ChromaGraphStore::connect_entity_collections(
        chroma_config("nodes"),
        chroma_config("edges"),
    )
    .await
    .unwrap();
    let sqlite_path = std::env::temp_dir().join(format!(
        "kogwistar-sqlite-chroma-graph-mutation-{}.sqlite",
        uuid::Uuid::new_v4()
    ));
    let embedding_calls = Arc::new(Mutex::new(Vec::new()));
    let provider = Arc::new(FixedEmbeddingProvider {
        profile: kogwistar_engine::embedding::EmbeddingProfile {
            provider: "test".to_owned(),
            model: "fixed-v1".to_owned(),
            dimension: 2,
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
        },
        vector: vec![0.6, 0.8],
        calls: embedding_calls.clone(),
        fail: None,
    });
    let service = SqliteRunApplicationService::open(&sqlite_path)
        .unwrap()
        .with_chroma_graph(graph)
        .with_embedding_provider(provider.clone());
    assert!(service.supports_http_operation("POST", INTERNAL_GRAPH_MUTATIONS_ROUTE));

    let mut batch: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/graph-mutation-batch-v1.json"
    )))
    .unwrap();
    batch["mutations"][0]["scope"] = json!({
        "namespace":"docs", "workspace_id":null, "graph_space":null
    });
    let python_models: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/python-graph-model-payloads-v1.json"
    )))
    .unwrap();
    let mut node_payload = python_models["cases"][0]["payload"].clone();
    node_payload["id"] = json!("node-1");
    node_payload["embedding"] = json!([1.0, 0.0]);
    node_payload["metadata"].as_object_mut().unwrap().remove("workspace_id");
    node_payload["metadata"].as_object_mut().unwrap().remove("graph_space");
    batch["mutations"][0]["payload"] = node_payload.clone();
    batch["mutations"][0]["model_envelope"] = json!({
        "schema_version": 1,
        "model_class": "node",
        "payload": node_payload,
    });
    let mut edge_mutation = batch["mutations"][0].clone();
    edge_mutation["table"] = json!("gke_edges");
    edge_mutation["entity_kind"] = json!("edge");
    edge_mutation["event_id"] = json!("event-edge");
    edge_mutation["record"]["id"] = json!("edge-1");
    let mut edge_payload = python_models["cases"][1]["payload"].clone();
    edge_payload["id"] = json!("edge-1");
    edge_payload["embedding"] = json!([1.0, 0.0]);
    edge_payload["source_ids"] = json!(["node-1"]);
    edge_payload["metadata"].as_object_mut().unwrap().remove("workspace_id");
    edge_payload["metadata"].as_object_mut().unwrap().remove("graph_space");
    edge_mutation["payload"] = edge_payload.clone();
    edge_mutation["model_envelope"] = json!({
        "schema_version": 1,
        "model_class": "edge",
        "payload": edge_payload,
    });
    batch["mutations"].as_array_mut().unwrap().push(edge_mutation);
    let body = serde_json::to_vec(&batch).unwrap();
    let principal = json!({
        "sub":"alice", "user_id":"alice", "role":"rw", "ns":["docs"],
        "storage_ns":"docs", "capabilities":["write_graph"]
    });
    let denied = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: INTERNAL_GRAPH_MUTATIONS_ROUTE.to_owned(),
            body: body.clone(),
            principal: json!({"sub":"alice", "role":"ro", "ns":["docs"], "storage_ns":"docs"}),
        })
        .await;
    assert_eq!(denied.status, StatusCode::FORBIDDEN.as_u16());
    assert!(service.store.replay_raw_events("docs", 0, 10).unwrap().is_empty());

    let request = || ApiEffectRequest {
        contract_version: 1,
        method: "POST".to_owned(),
        path_and_query: INTERNAL_GRAPH_MUTATIONS_ROUTE.to_owned(),
        body: body.clone(),
        principal: principal.clone(),
    };
    let first = service.execute(request()).await;
    assert_eq!(first.status, StatusCode::OK.as_u16(), "{}", String::from_utf8_lossy(&first.body));
    let first_body: Value = serde_json::from_slice(&first.body).unwrap();
    assert_eq!(first_body["projection_consistency"], "eventual");
    assert_eq!(first_body["results"].as_array().unwrap().len(), 2);
    assert_eq!(first_body["results"][0]["inserted"], true);

    let retry = service.execute(request()).await;
    assert_eq!(retry.status, StatusCode::OK.as_u16(), "{}", String::from_utf8_lossy(&retry.body));
    let retry_body: Value = serde_json::from_slice(&retry.body).unwrap();
    assert_eq!(retry_body["results"][0]["mutated"], false);
    let canonical_events = service.store.replay_raw_events("docs", 0, 10).unwrap();
    assert_eq!(canonical_events.len(), 2);
    for event in &canonical_events {
        let payload: Value = serde_json::from_str(&event.payload_json).unwrap();
        let expected_class = if payload["id"] == "node-1" {
            "Node"
        } else {
            "Edge"
        };
        assert_eq!(payload["metadata"]["_class_name"], expected_class);
    }
    let jobs = service
        .store
        .list_index_jobs(Some("docs"), Some("PENDING"), None, None, None, 10)
        .unwrap();
    assert_eq!(jobs.len(), 2);
    assert!(jobs.iter().all(|job| job.index_kind == "node_embedding"));
    assert_eq!(
        jobs.iter()
            .map(|job| job.entity_kind.as_str())
            .collect::<std::collections::BTreeSet<_>>(),
        ["edge", "node"].into_iter().collect()
    );

    // Canonical SQLite events and jobs must survive a service restart before
    // the eventual Chroma projection is attempted.
    drop(service);
    let graph = kogwistar_store_chroma::ChromaGraphStore::connect_entity_collections(
        chroma_config("nodes"),
        chroma_config("edges"),
    )
    .await
    .unwrap();
    let service = SqliteRunApplicationService::open(&sqlite_path)
        .unwrap()
        .with_chroma_graph(graph)
        .with_embedding_provider(provider.clone());
    let recovered_events = service.store.replay_raw_events("docs", 0, 10).unwrap();
    assert_eq!(recovered_events.len(), 2);
    for event in &recovered_events {
        let payload: Value = serde_json::from_str(&event.payload_json).unwrap();
        let expected_class = if payload["id"] == "node-1" {
            "Node"
        } else {
            "Edge"
        };
        assert_eq!(payload["metadata"]["_class_name"], expected_class);
    }
    let recovered_jobs = service
        .store
        .list_index_jobs(Some("docs"), Some("PENDING"), None, None, None, 10)
        .unwrap();
    assert_eq!(recovered_jobs.len(), 2);

    let failed_projection = service
        .run_embedding_projection_jobs_once(10, 30, Some("docs"))
        .await
        .unwrap();
    assert_eq!(failed_projection.claimed, 2);
    assert_eq!(failed_projection.projected, 1);
    assert_eq!(failed_projection.retried, 1);
    assert_eq!(service.store.replay_raw_events("docs", 0, 10).unwrap().len(), 2);
    let after_failure = service
        .store
        .list_index_jobs(Some("docs"), None, None, None, None, 10)
        .unwrap();
    assert_eq!(after_failure.iter().filter(|job| job.status == "DONE").count(), 1);
    assert_eq!(after_failure.iter().filter(|job| job.status == "PENDING").count(), 1);

    // A Chroma failure cannot undo canonical SQLite state. Reopen the service
    // and retry only the durable failed projection job.
    drop(service);
    tokio::time::sleep(std::time::Duration::from_millis(2200)).await;
    let graph = kogwistar_store_chroma::ChromaGraphStore::connect_entity_collections(
        chroma_config("nodes"),
        chroma_config("edges"),
    )
    .await
    .unwrap();
    let service = SqliteRunApplicationService::open(&sqlite_path)
        .unwrap()
        .with_chroma_graph(graph)
        .with_embedding_provider(provider.clone());
    let trigger_connection = rusqlite::Connection::open(&sqlite_path).unwrap();
    trigger_connection
        .execute_batch(
            "CREATE TRIGGER reject_node_projection_ack
             BEFORE UPDATE OF status ON index_jobs
             WHEN NEW.entity_id = 'node-1' AND NEW.status = 'DONE'
             BEGIN SELECT RAISE(ABORT, 'injected projection acknowledgement failure'); END;",
        )
        .unwrap();
    let ack_failure = service
        .run_embedding_projection_jobs_once(10, 1, Some("docs"))
        .await
        .unwrap_err();
    assert!(ack_failure.contains("injected projection acknowledgement failure"));
    let after_ack_failure = service
        .store
        .list_index_jobs(Some("docs"), None, None, None, None, 10)
        .unwrap();
    let node_job = after_ack_failure
        .iter()
        .find(|job| job.entity_id == "node-1")
        .unwrap();
    assert_eq!(node_job.status, "DOING");
    assert!(node_job.accepted_result_json.is_some());
    assert_eq!(
        after_ack_failure
            .iter()
            .find(|job| job.entity_id == "edge-1")
            .unwrap()
            .status,
        "DONE"
    );

    drop(service);
    trigger_connection
        .execute_batch("DROP TRIGGER reject_node_projection_ack")
        .unwrap();
    drop(trigger_connection);
    tokio::time::sleep(std::time::Duration::from_millis(2200)).await;
    let graph = kogwistar_store_chroma::ChromaGraphStore::connect_entity_collections(
        chroma_config("nodes"),
        chroma_config("edges"),
    )
    .await
    .unwrap();
    let service = SqliteRunApplicationService::open(&sqlite_path)
        .unwrap()
        .with_chroma_graph(graph)
        .with_embedding_provider(provider);
    let final_projection = service
        .run_embedding_projection_jobs_once(10, 30, Some("docs"))
        .await
        .unwrap();
    assert_eq!(final_projection.claimed, 1);
    assert_eq!(final_projection.projected, 0);
    assert_eq!(final_projection.reused, 1);
    assert_eq!(service.store.replay_raw_events("docs", 0, 10).unwrap().len(), 2);
    let projection_jobs = service
        .store
        .list_index_jobs(Some("docs"), None, None, None, None, 10)
        .unwrap();
    assert_eq!(projection_jobs.len(), 2);
    assert!(projection_jobs.iter().all(|job| job.status == "DONE"));
    assert_eq!(embedding_calls.lock().unwrap().len(), 1);
    let _ = stop_chroma.send(());
    let chroma_upserts = fake_chroma.await.unwrap();
    assert_eq!(chroma_upserts.len(), 4);
    let node_attempts = chroma_upserts
        .iter()
        .filter(|(_, body)| body["ids"][0] == "node-1")
        .collect::<Vec<_>>();
    let edge_attempts = chroma_upserts
        .iter()
        .filter(|(_, body)| body["ids"][0] == "edge-1")
        .collect::<Vec<_>>();
    assert_eq!(node_attempts.len(), 3);
    assert_eq!(edge_attempts.len(), 1);
    assert_eq!(node_attempts[0].1, node_attempts[1].1);
    assert_eq!(node_attempts[1].1, node_attempts[2].1);
    let node_upsert = chroma_upserts
        .iter()
        .find(|(_, body)| body["ids"][0] == "node-1")
        .unwrap();
    assert!(node_upsert.0.contains("collection-nodes"));
    assert_eq!(node_upsert.1["embeddings"][0], json!([0.6, 0.8]));
    assert_eq!(node_upsert.1["metadatas"][0]["namespace"], "docs");
    assert_eq!(node_upsert.1["metadatas"][0]["_class_name"], "Node");
    let edge_upsert = chroma_upserts
        .iter()
        .find(|(_, body)| body["ids"][0] == "edge-1")
        .unwrap();
    assert!(edge_upsert.0.contains("collection-edges"));
    assert_eq!(edge_upsert.1["embeddings"][0], json!([0.6, 0.8]));
    assert_eq!(edge_upsert.1["metadatas"][0]["namespace"], "docs");
    assert_eq!(edge_upsert.1["metadatas"][0]["_class_name"], "Edge");

    drop(service);
    remove_test_sqlite(&sqlite_path);
}

#[tokio::test]
async fn sqlite_chroma_conversation_create_and_turn_admission_are_canonical_and_atomic() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use kogwistar_store::{
            GraphMutation, GraphMutationBatch, GraphRecord, GraphScope, NamedProjectionWrite,
            WorkflowDesignSnapshotWrite,
        };

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let chat_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let chat_endpoint = format!("http://{}/v1/chat/completions", chat_listener.local_addr().unwrap());
        let chat_server = tokio::spawn(async move {
            let (mut stream, _) = chat_listener.accept().await.unwrap();
            let mut bytes = Vec::new();
            let (header_end, content_length) = loop {
                let mut chunk = [0_u8; 2048];
                let count = stream.read(&mut chunk).await.unwrap();
                assert!(count > 0);
                bytes.extend_from_slice(&chunk[..count]);
                if let Some(end) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
                    let headers = String::from_utf8_lossy(&bytes[..end]);
                    let length = headers.lines().find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().ok())
                            .flatten()
                    }).unwrap_or(0);
                    if bytes.len() >= end + 4 + length {
                        break (end, length);
                    }
                }
            };
            let request: Value = serde_json::from_slice(
                &bytes[header_end + 4..header_end + 4 + content_length],
            ).unwrap();
            assert_eq!(request["stream"], true);
            assert_eq!(request["response_format"]["type"], "json_schema");
            let prompt = request["messages"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|message| message["content"].as_str())
                .collect::<Vec<_>>()
                .join("\n");
            let visible_alias = prompt
                .lines()
                .find_map(|line| line.strip_prefix("NODE "))
                .and_then(|line| line.split_once(" |"))
                .map(|(alias, _)| alias.to_owned())
                .expect("provider prompt contains an aliased retrieved evidence node");
            let content = json!({
                "text":"The runtime persists workflow state in the graph.",
                "reasoning":"The retrieved source supports this statement.",
                "claims":[{
                    "claim":"Workflow state is persisted in the graph.",
                    "citations":[{"source_node_id":visible_alias,"mention_index":0,"span_index":0}]
                }]
            }).to_string();
            let frames = [
                format!("data: {}\r\n\r\n", json!({"id":"sqlite-chroma-answer","choices":[{"delta":{"content":content},"finish_reason":null}]})),
                format!("data: {}\r\n\r\n", json!({"id":"sqlite-chroma-answer","choices":[{"delta":{},"finish_reason":"stop"}]})),
                "data: [DONE]\r\n\r\n".to_owned(),
            ].concat();
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                frames.len(), frames
            );
            stream.write_all(response.as_bytes()).await.unwrap();
        });
        let (stop_tx, mut stop_rx) = tokio::sync::oneshot::channel();
        let fake_chroma = tokio::spawn(async move {
            let mut projection_requests = Vec::new();
            loop {
                let (mut stream, _) = tokio::select! {
                    _ = &mut stop_rx => break,
                    accepted = listener.accept() => accepted.unwrap(),
                };
                let mut bytes = Vec::new();
                let mut chunk = [0_u8; 2048];
                let (header_end, content_length) = loop {
                    let count = stream.read(&mut chunk).await.unwrap();
                    assert!(count > 0, "Chroma closed before receiving request body");
                    bytes.extend_from_slice(&chunk[..count]);
                    if let Some(header_end) =
                        bytes.windows(4).position(|part| part == b"\r\n\r\n")
                    {
                        let headers = String::from_utf8_lossy(&bytes[..header_end]);
                        let content_length = headers
                            .lines()
                            .find_map(|line| {
                                let (name, value) = line.split_once(':')?;
                                name.eq_ignore_ascii_case("content-length")
                                    .then(|| value.trim().parse::<usize>().ok())
                                    .flatten()
                            })
                            .unwrap_or(0);
                        if bytes.len() >= header_end + 4 + content_length {
                            break (header_end, content_length);
                        }
                    }
                };
                let request_line = String::from_utf8_lossy(&bytes[..header_end])
                    .lines()
                    .next()
                    .unwrap_or_default()
                    .to_owned();
                let body: Value = serde_json::from_slice(
                    &bytes[header_end + 4..header_end + 4 + content_length],
                )
                .unwrap();
                let response = if request_line.contains("/collections HTTP") {
                    json!({"id":"collection-id", "metadata":{"hnsw:space":"cosine"}})
                } else if request_line.contains("/query HTTP") {
                    json!({
                        "ids": [["knowledge-node"]],
                        "documents": [["The runtime persists workflow state in the graph."]],
                        "metadatas": [[{
                            "namespace":"conversation",
                            "workspace_id":"workspace-a",
                            "graph_space":"conversation",
                            "entity_type":"node",
                            "label":"Workflow runtime state",
                            "summary":"The runtime persists workflow state in the graph.",
                            "level_from_root":0,
                            "lifecycle_status":"active",
                            "_kogwistar_stage2_ready":true,
                            "_kogwistar_source_revision":1,
                            "_class_name":"Node"
                        }]],
                        "embeddings": [[[0.6, 0.8]]],
                        "distances": [[0.0]]
                    })
                } else if request_line.contains("/get HTTP")
                    && body.to_string().contains("acl_record")
                {
                    json!({
                        "ids": ["knowledge-node-acl"],
                        "documents": ["{}"],
                        "metadatas": [{
                            "namespace":"conversation",
                            "workspace_id":"workspace-a",
                            "graph_space":"conversation",
                            "entity_type":"acl_record",
                            "acl_truth_graph":"knowledge",
                            "acl_target_entity_id":"knowledge-node",
                            "acl_target_grain":"node",
                            "acl_version":1,
                            "acl_mode":"public",
                            "_kogwistar_stage2_ready":true,
                            "_kogwistar_source_revision":2,
                            "_class_name":"Node"
                        }]
                    })
                } else if request_line.contains("/get HTTP") {
                    json!({
                        "ids": ["knowledge-node"],
                        "documents": ["The runtime persists workflow state in the graph."],
                        "metadatas": [{
                            "namespace":"conversation",
                            "workspace_id":"workspace-a",
                            "graph_space":"conversation",
                            "entity_type":"node",
                            "label":"Workflow runtime state",
                            "summary":"The runtime persists workflow state in the graph.",
                            "level_from_root":0,
                            "lifecycle_status":"active",
                            "_kogwistar_stage2_ready":true,
                            "_kogwistar_source_revision":1,
                            "_class_name":"Node"
                        }]
                    })
                } else {
                    if request_line.contains("/upsert") || request_line.contains("/delete") {
                        projection_requests.push(request_line);
                    }
                    json!({})
                };
                let response = response.to_string();
                let wire = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    response.len(), response
                );
                stream.write_all(wire.as_bytes()).await.unwrap();
            }
            projection_requests
        });
        let config = |collection: &str| kogwistar_store_chroma::ChromaStoreConfig {
            endpoint: endpoint.clone(),
            collection: collection.to_owned(),
            ..Default::default()
        };
        let graph = kogwistar_store_chroma::ChromaGraphStore::connect_entity_collections(
            config("conversation-nodes"),
            config("conversation-edges"),
        )
        .await
        .unwrap();
        let path = std::env::temp_dir().join(format!(
            "kogwistar-sqlite-conversation-create-{}.sqlite",
            uuid::Uuid::new_v4().simple()
        ));
        let auth_path = std::env::temp_dir().join(format!(
            "kogwistar-sqlite-conversation-owner-auth-{}.sqlite",
            uuid::Uuid::new_v4().simple()
        ));
        let auth = SqliteAuthStore::open(&auth_path).unwrap();
        auth.resolve_external_identity(ResolveExternalIdentity {
            issuer: "test".to_owned(),
            subject: "conversation-owner".to_owned(),
            email: "conversation-owner@example.test".to_owned(),
            display_name: None,
            new_user_id: "conversation-owner".to_owned(),
            default_role: "rw".to_owned(),
            default_ns: "conversation".to_owned(),
        })
        .await
        .unwrap();
        drop(auth);
        let owner_auth = AuthConfig {
            auth_db_url: Some(format!("sqlite:///{}", auth_path.display())),
            ..AuthConfig::default()
        };
        struct TestEmbeddingProvider {
            profile: kogwistar_engine::embedding::EmbeddingProfile,
        }
        impl kogwistar_engine::embedding::EmbeddingProvider for TestEmbeddingProvider {
            fn profile(&self) -> &kogwistar_engine::embedding::EmbeddingProfile {
                &self.profile
            }

            fn embed<'a>(
                &'a self,
                texts: &'a [String],
            ) -> kogwistar_engine::embedding::EmbeddingFuture<'a> {
                Box::pin(async move { Ok(texts.iter().map(|_| vec![0.6, 0.8]).collect()) })
            }
        }
        let embedding_provider = Arc::new(TestEmbeddingProvider {
            profile: kogwistar_engine::embedding::EmbeddingProfile {
                provider: "test".to_owned(),
                model: "conversation-test".to_owned(),
                dimension: 2,
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
            },
        });
        let service = SqliteRunApplicationService::open(&path)
            .unwrap()
            .with_chroma_graph(graph)
            .with_embedding_provider(embedding_provider)
            .with_chat_completion_provider(
                crate::native_answer_generation::OpenAiCompatibleChatConfig::new(
                    chat_endpoint,
                    "sqlite-chroma-fixture-model",
                    None,
                )
                .unwrap()
                .with_streaming(true),
            )
            .with_owner_auth_config(&owner_auth)
            .await
            .unwrap();

        let mut knowledge_payload: Value = serde_json::from_str::<Value>(
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../../contracts/golden/python-graph-model-payloads-v1.json"
            )),
        )
        .unwrap()["cases"][0]["payload"]
            .clone();
        knowledge_payload["id"] = json!("knowledge-node");
        knowledge_payload["label"] = json!("Workflow runtime state");
        knowledge_payload["summary"] = json!("The runtime persists workflow state in the graph.");
        knowledge_payload["doc_id"] = json!("doc:runtime-state");
        knowledge_payload["embedding"] = Value::Null;
        knowledge_payload["metadata"]["workspace_id"] = json!("workspace-a");
        knowledge_payload["metadata"]["graph_space"] = json!("conversation");
        knowledge_payload["metadata"]["level_from_root"] = json!(0);
        let mut acl_payload = knowledge_payload.clone();
        acl_payload["id"] = json!("knowledge-node-acl");
        acl_payload["label"] = json!("ACL for knowledge-node");
        acl_payload["summary"] = json!("Public node access");
        acl_payload["metadata"]["entity_type"] = json!("acl_record");
        acl_payload["metadata"]["acl_truth_graph"] = json!("knowledge");
        acl_payload["metadata"]["acl_target_entity_id"] = json!("knowledge-node");
        acl_payload["metadata"]["acl_target_grain"] = json!("node");
        acl_payload["metadata"]["acl_version"] = json!(1);
        acl_payload["metadata"]["acl_mode"] = json!("public");
        let graph_scope = GraphScope {
            namespace: "conversation".to_owned(),
            workspace_id: Some("workspace-a".to_owned()),
            graph_space: Some("conversation".to_owned()),
        };
        let mutation = |event_id: &str, payload: Value| {
            let id = payload["id"].as_str().unwrap().to_owned();
            let metadata = payload["metadata"].as_object().unwrap().clone();
            GraphMutation {
                scope: graph_scope.clone(),
                table: "gke_nodes".to_owned(),
                entity_kind: "node".to_owned(),
                event_id: event_id.to_owned(),
                op: "ADD".to_owned(),
                payload: payload.clone(),
                record: GraphRecord {
                    id,
                    document: Some(payload.to_string()),
                    metadata,
                    embedding: None,
                },
                embedding_dim: 2,
            }
        };
        service
            .store
            .apply_canonical_graph_mutation_batch(GraphMutationBatch {
                mutations: vec![
                    mutation("knowledge-node-event", knowledge_payload),
                    mutation("knowledge-node-acl-event", acl_payload),
                ],
                index_jobs: Vec::new(),
            })
            .unwrap();

        let workflow_contract: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/answer-workflow-runtime-contract-v1.json"
        )))
        .unwrap();
        let workflow_id = "debug.rag.v1";
        let workflow = &workflow_contract["workflows"][workflow_id];
        let nodes = workflow["required_ops"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| {
                let op = value.as_str().unwrap();
                json!({
                    "id": op,
                    "label": op,
                    "metadata": {
                        "entity_type": "workflow_node",
                        "workflow_id": workflow_id,
                        "wf_op": op,
                        "wf_version": "v1",
                        "wf_executor_target": "rust-native",
                        "wf_start": op == "start"
                    }
                })
            })
            .collect::<Vec<_>>();
        let edges = workflow["required_routes"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(index, route)| {
                json!({
                    "id": format!("debug-rag-route-{index}"),
                    "source_ids": [route["from"]],
                    "target_ids": [route["to"]],
                    "metadata": {
                        "entity_type": "workflow_edge",
                        "workflow_id": workflow_id,
                        "wf_predicate": route["predicate"],
                        "wf_priority": route["priority"],
                        "wf_is_default": route["default"],
                        "wf_multiplicity": route["multiplicity"]
                    }
                })
            })
            .collect::<Vec<_>>();
        service
            .store
            .replace_named_projection(
                "workflow_design",
                workflow_id,
                NamedProjectionWrite {
                    payload: serde_json::Map::from_iter([
                        ("current_version".to_owned(), json!(1)),
                        ("snapshot_schema_version".to_owned(), json!(1)),
                    ]),
                    last_authoritative_seq: 1,
                    last_materialized_seq: 1,
                    projection_schema_version: 1,
                    materialization_status: "ready".to_owned(),
                },
            )
            .unwrap();
        service
            .store
            .put_workflow_design_snapshot(
                workflow_id,
                WorkflowDesignSnapshotWrite {
                    version: 1,
                    seq: 1,
                    schema_version: 1,
                    payload_json: serde_json::to_string(&json!({ "nodes": nodes, "edges": edges }))
                        .unwrap(),
                },
            )
            .unwrap();
        assert!(service.supports_http_operation("POST", "/api/conversations"));
        assert!(service.supports_mcp_tool("conversation", "conversation.ask"));

        let principal = json!({
            "sub":"conversation-owner", "user_id":"conversation-owner", "role":"rw",
            "ns":["conversation"], "storage_ns":"conversation",
            "workspace":"workspace-a", "graph_space":"conversation",
            "capabilities":["spawn_process", "send_message", "write_graph", "read_graph", "workflow.run.write"]
        });
        let request = || ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/conversations".to_owned(),
            body: serde_json::to_vec(&json!({
                "user_id":"conversation-owner",
                "conversation_id":"conversation-create-contract",
                "start_node_id":"conversation-create-start"
            }))
            .unwrap(),
            principal: principal.clone(),
        };
        let response = service.execute(request()).await;
        assert_eq!(response.status, StatusCode::OK.as_u16());
        let body: Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(body["conversation_id"], "conversation-create-contract");
        assert_eq!(body["start_node_id"], "conversation-create-start");
        assert_eq!(body["turn_count"], 0);

        let events = service.store.replay_raw_events("conversation", 0, 20).unwrap();
        assert!(events.iter().any(|event| {
            event.entity_kind == "node"
                && event.entity_id == "conversation-create-start"
                && serde_json::from_str::<Value>(&event.payload_json)
                    .ok()
                    .is_some_and(|payload| {
                        payload["metadata"]["entity_type"] == "conversation_start"
                            && payload["metadata"]["conversation_id"]
                                == "conversation-create-contract"
                    })
        }));

        let snapshot = service
            .chroma_graph_snapshot(&request(), "conversation")
            .await
            .unwrap();
        assert!(
            snapshot
                .nodes()
                .iter()
                .any(|record| record.id == "conversation-create-start"),
            "newly created conversation start must be visible in the scoped canonical snapshot; stage1={:?}",
            service.store.list_stage1_node_projections("conversation").unwrap()
        );
        let visible_start = snapshot
            .nodes()
            .iter()
            .find(|record| record.id == "conversation-create-start")
            .unwrap();
        assert_eq!(
            visible_start.metadata.get("conversation_id"),
            Some(&json!("conversation-create-contract")),
            "snapshot metadata={:?}, document={:?}",
            visible_start.metadata,
            visible_start.document
        );

        let send_message = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/syscall/v1/send_message".to_owned(),
                body: serde_json::to_vec(&json!({
                    "version":"v1",
                    "op":"send_message",
                    "args":{
                        "conversation_id":"conversation-create-contract",
                        "text":"Where is workflow runtime state persisted?",
                        "workflow_id":workflow_id
                    }
                }))
                .unwrap(),
                principal: principal.clone(),
            })
            .await;
        assert_eq!(
            send_message.status,
            StatusCode::OK.as_u16(),
            "{}",
            String::from_utf8_lossy(&send_message.body)
        );
        let send_result: Value = serde_json::from_slice(&send_message.body).unwrap();
        assert_eq!(send_result["op"], "send_message");
        assert_eq!(send_result["result"]["status"], "queued");
        let run_id = send_result["result"]["run_id"].as_str().unwrap();
        assert!(service.store.get_server_run(run_id).unwrap().is_some());
        let events = service.store.replay_raw_events("conversation", 0, 50).unwrap();
        assert!(events.iter().any(|event| {
            event.entity_kind == "node"
                && serde_json::from_str::<Value>(&event.payload_json)
                    .ok()
                    .is_some_and(|payload| {
                        payload["metadata"]["entity_type"] == "conversation_turn"
                            && payload["summary"] == "Where is workflow runtime state persisted?"
                    })
        }));
        assert!(events.iter().any(|event| {
            event.entity_kind == "edge"
                && serde_json::from_str::<Value>(&event.payload_json)
                    .ok()
                    .is_some_and(|payload| payload["relation"] == "next_turn")
        }));
        assert!(events.iter().any(|event| {
            serde_json::from_str::<Value>(&event.payload_json)
                .ok()
                .is_some_and(|payload| payload["metadata"]["entity_type"] == "acl_record")
        }));

        for _ in 0..32 {
            let status = service
                .store
                .get_server_run(run_id)
                .unwrap()
                .expect("admitted answer run remains durable");
            if matches!(status.status.as_str(), "succeeded" | "failed" | "cancelled") {
                break;
            }
            assert!(
                service.execute_inline_native_work_once().await.unwrap(),
                "native scheduler stopped before the answer run became terminal: {status:#?}"
            );
        }
        let completed_run = service
            .store
            .get_server_run(run_id)
            .unwrap()
            .expect("completed answer run remains durable");
        assert_eq!(completed_run.status, "succeeded");
        let runtime = service
            .store
            .read_recorded_runtime_state(run_id, workflow_id, "conversation-create-contract")
            .unwrap()
            .expect("successful run has durable runtime state");
        assert_eq!(runtime.state["done"], true);
        assert_eq!(runtime.state["answer"]["mode"], "debug_rag");
        assert_eq!(runtime.state["used_node_ids"], json!(["knowledge-node"]));
        assert!(runtime.state["answer"]["text"]
            .as_str()
            .is_some_and(|text| text.contains("knowledge-node")));
        let projected_pointer = runtime.state["projected_pointer_ids"][0]
            .as_str()
            .expect("selected evidence receives a durable conversation pointer")
            .to_owned();
        assert!(service
            .store
            .latest_entity_events("conversation", "node", &[projected_pointer.clone()])
            .unwrap()
            .contains_key(&projected_pointer));
        let assistant_id = runtime.state["agentic_answering_result"]["assistant_turn_node_id"]
            .as_str()
            .expect("answer workflow records the assistant turn")
            .to_owned();
        let assistant = service
            .store
            .latest_entity_events("conversation", "node", &[assistant_id.clone()])
            .unwrap();
        assert!(assistant.contains_key(&assistant_id), "assistant turn is canonical");
        let edges = service.store.replay_raw_events("conversation", 0, 100).unwrap();
        assert!(edges.iter().any(|event| {
            event.entity_kind == "edge"
                && serde_json::from_str::<Value>(&event.payload_json)
                    .ok()
                    .is_some_and(|payload| {
                        payload["relation"] == "next_turn"
                            && payload["source_ids"] == json!([runtime.state["turn_node_id"]])
                            && payload["target_ids"] == json!([assistant_id])
                    })
        }), "assistant turn {assistant_id} is linked to user turn {} by the canonical graph edge", runtime.state["turn_node_id"]);

        let duplicate = service.execute(request()).await;
        assert_eq!(duplicate.status, StatusCode::CONFLICT.as_u16());
        let wrong_owner = ApiEffectRequest {
            principal: json!({
                "sub":"other-user", "user_id":"other-user", "role":"rw",
                "ns":["conversation"], "storage_ns":"conversation",
                "capabilities":["spawn_process"]
            }),
            ..request()
        };
        assert_eq!(
            service.execute(wrong_owner).await.status,
            StatusCode::FORBIDDEN.as_u16()
        );

        let agentic_workflow_id = "agentic_answering.v2";
        let agentic_workflow = &workflow_contract["workflows"][agentic_workflow_id];
        let agentic_nodes = agentic_workflow["required_ops"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| {
                let op = value.as_str().unwrap();
                json!({"id":op,"label":op,"metadata":{
                    "entity_type":"workflow_node","workflow_id":agentic_workflow_id,
                    "wf_op":op,"wf_version":"v1","wf_executor_target":"rust-native",
                    "wf_start":op=="start"
                }})
            })
            .collect::<Vec<_>>();
        let agentic_edges = agentic_workflow["required_routes"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(index, route)| {
                json!({"id":format!("agentic-route-{index}"),"source_ids":[route["from"]],
                    "target_ids":[route["to"]],"metadata":{
                        "entity_type":"workflow_edge","workflow_id":agentic_workflow_id,
                        "wf_predicate":route["predicate"],"wf_priority":route["priority"],
                        "wf_is_default":route["default"],"wf_multiplicity":route["multiplicity"]
                    }})
            })
            .collect::<Vec<_>>();
        service.store.replace_named_projection(
            "workflow_design",
            agentic_workflow_id,
            NamedProjectionWrite {
                payload: serde_json::Map::from_iter([
                    ("current_version".to_owned(), json!(1)),
                    ("snapshot_schema_version".to_owned(), json!(1)),
                ]),
                last_authoritative_seq: 1,
                last_materialized_seq: 1,
                projection_schema_version: 1,
                materialization_status: "ready".to_owned(),
            },
        ).unwrap();
        service.store.put_workflow_design_snapshot(
            agentic_workflow_id,
            WorkflowDesignSnapshotWrite {
                version: 1,
                seq: 1,
                schema_version: 1,
                payload_json: serde_json::to_string(&json!({"nodes":agentic_nodes,"edges":agentic_edges})).unwrap(),
            },
        ).unwrap();
        let agentic_turn_id = runtime.state["turn_node_id"].as_str().unwrap();
        let agentic_submit = service.execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/workflow/runs".to_owned(),
            body: serde_json::to_vec(&json!({
                "run_id":"sqlite-chroma-agentic-provider-run",
                "workflow_id":agentic_workflow_id,
                "conversation_id":"conversation-create-contract",
                "turn_node_id":agentic_turn_id,
                "user_id":"conversation-owner",
                "initial_state":{
                    "conversation_id":"conversation-create-contract",
                    "turn_node_id":agentic_turn_id,
                    "user_id":"conversation-owner",
                    "turn_index":0,
                    "max_candidates":10,
                    "max_used":4,
                    "max_retrieval_level":3,
                    "max_iter":1
                }
            })).unwrap(),
            principal: principal.clone(),
        }).await;
        assert_eq!(agentic_submit.status, StatusCode::ACCEPTED.as_u16(), "{}", String::from_utf8_lossy(&agentic_submit.body));
        let agentic_run_id = "sqlite-chroma-agentic-provider-run";
        for _ in 0..48 {
            let status = service.store.get_server_run(&agentic_run_id).unwrap().unwrap();
            if matches!(status.status.as_str(), "succeeded" | "failed" | "cancelled") {
                break;
            }
            assert!(service.execute_inline_native_work_once().await.unwrap(),
                "native scheduler stopped before the provider-backed run became terminal: {status:#?}");
        }
        let agentic_run = service.store.get_server_run(&agentic_run_id).unwrap().unwrap();
        assert_eq!(agentic_run.status, "succeeded", "provider-backed SQLite+Chroma run should complete natively");
        let agentic_state = service.store.read_recorded_runtime_state(
            agentic_run_id,
            agentic_workflow_id,
            "conversation-create-contract",
        ).unwrap().unwrap();
        assert_eq!(agentic_state.state["used_node_ids"], json!(["knowledge-node"]));
        assert_eq!(agentic_state.state["answer"]["claims"][0]["citations"][0]["source_node_id"], "knowledge-node");
        assert_eq!(agentic_state.state["done"], true);
        let agentic_assistant_id = agentic_state.state["agentic_answering_result"]["assistant_turn_node_id"]
            .as_str()
            .expect("provider-backed run records its assistant turn")
            .to_owned();
        assert!(service.store.latest_entity_events(
            "conversation",
            "node",
            &[agentic_assistant_id.clone()],
        ).unwrap().contains_key(&agentic_assistant_id));
        let agentic_turn_id = agentic_state.state["turn_node_id"].as_str().unwrap();
        assert!(service.store.replay_raw_events("conversation", 0, 200).unwrap().iter().any(|event| {
            event.entity_kind == "edge"
                && serde_json::from_str::<Value>(&event.payload_json)
                    .ok()
                    .is_some_and(|payload| {
                        payload["relation"] == "next_turn"
                            && payload["source_ids"] == json!([agentic_turn_id])
                            && payload["target_ids"] == json!([agentic_assistant_id])
                    })
        }));
        chat_server.await.unwrap();
        let _ = stop_tx.send(());
        assert!(fake_chroma.await.unwrap().is_empty());
        drop(service);
        remove_test_sqlite(&path);
        remove_test_sqlite(&auth_path);

        let metadata_path = std::env::temp_dir().join(format!(
            "kogwistar-sqlite-metadata-conversation-create-{}.sqlite",
            uuid::Uuid::new_v4().simple()
        ));
        let metadata_only = SqliteRunApplicationService::open(&metadata_path).unwrap();
        assert!(!metadata_only.supports_http_operation("POST", "/api/conversations"));
        drop(metadata_only);
        remove_test_sqlite(&metadata_path);
    }

#[tokio::test]
async fn sqlite_chroma_runtime_result_commits_graph_batch_with_worker_effect() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let chroma_setup = tokio::spawn(async move {
        for _ in 0..2 {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut bytes = Vec::new();
            let mut chunk = [0_u8; 2048];
            let (header_end, content_length) = loop {
                let count = stream.read(&mut chunk).await.unwrap();
                assert!(count > 0);
                bytes.extend_from_slice(&chunk[..count]);
                if let Some(header_end) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
                    let headers = String::from_utf8_lossy(&bytes[..header_end]);
                    let content_length = headers
                        .lines()
                        .find_map(|line| {
                            let (name, value) = line.split_once(':')?;
                            name.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse::<usize>().ok())
                                .flatten()
                        })
                        .unwrap_or(0);
                    if bytes.len() >= header_end + 4 + content_length {
                        break (header_end, content_length);
                    }
                }
            };
            let request: Value = serde_json::from_slice(
                &bytes[header_end + 4..header_end + 4 + content_length],
            )
            .unwrap();
            let response = json!({
                "id": format!("collection-{}", request["name"].as_str().unwrap()),
                "metadata": {"hnsw:space": "cosine"}
            })
            .to_string();
            let wire = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                response.len(),
                response
            );
            stream.write_all(wire.as_bytes()).await.unwrap();
        }
    });
    let chroma_config = |collection: &str| kogwistar_store_chroma::ChromaStoreConfig {
        endpoint: endpoint.clone(),
        collection: collection.to_owned(),
        ..Default::default()
    };
    let graph = kogwistar_store_chroma::ChromaGraphStore::connect_entity_collections(
        chroma_config("runtime-nodes"),
        chroma_config("runtime-edges"),
    )
    .await
    .unwrap();
    chroma_setup.await.unwrap();

    let sqlite_path = std::env::temp_dir().join(format!(
        "kogwistar-sqlite-chroma-runtime-graph-{}.sqlite",
        uuid::Uuid::new_v4()
    ));
    let service = SqliteRunApplicationService::open(&sqlite_path)
        .unwrap()
        .with_chroma_graph(graph);
    let principal = json!({
        "sub": "runtime-owner", "user_id": "runtime-owner", "role": "rw",
        "ns": ["conversation"], "storage_ns": "conversation",
        "workspace": "workspace-a", "graph_space": "conversation",
        "capabilities": ["write_graph"]
    });
    let submitted = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/workflow/runs".to_owned(),
            body: serde_json::to_vec(&json!({
                "run_id": "sqlite-runtime-graph-run",
                "workflow_id": "sqlite-runtime-graph-workflow",
                "conversation_id": "sqlite-runtime-graph-conversation",
                "turn_node_id": "turn-1"
            }))
            .unwrap(),
            principal: principal.clone(),
        })
        .await;
    assert_eq!(submitted.status, StatusCode::ACCEPTED.as_u16());

    let claimed = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/internal/runtime/claim".to_owned(),
            body: serde_json::to_vec(&json!({
                "claimed_by": "python-worker", "limit": 1, "lease_seconds": 60
            }))
            .unwrap(),
            principal: json!({"sub":"worker","role":"admin","capabilities":["*"]}),
        })
        .await;
    assert_eq!(claimed.status, StatusCode::OK.as_u16());
    let claimed: Value = serde_json::from_slice(&claimed.body).unwrap();
    let work = &claimed["work"][0];

    let models: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/python-graph-model-payloads-v1.json"
    )))
    .unwrap();
    let mut node = models["cases"][0]["payload"].clone();
    node["id"] = json!("runtime-effect-node");
    node["embedding"] = json!([1.0, 0.0, 0.0]);
    node["metadata"]["namespace"] = json!("conversation");
    node["metadata"]["workspace_id"] = json!("workspace-a");
    node["metadata"]["graph_space"] = json!("conversation");
    let mut batch: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/graph-mutation-batch-v1.json"
    )))
    .unwrap();
    let mutation = &mut batch["mutations"][0];
    mutation["scope"] = json!({
        "namespace": "conversation", "workspace_id": "workspace-a",
        "graph_space": "conversation"
    });
    mutation["event_id"] = json!("runtime-effect-graph-event");
    mutation["payload"] = node.clone();
    mutation["model_envelope"] = json!({
        "schema_version": 1, "model_class": "node", "payload": node
    });
    mutation["record"]["id"] = json!("runtime-effect-node");
    mutation["record"]["embedding"] = json!([1.0, 0.0, 0.0]);
    mutation["embedding_dim"] = json!(3);

    let publication: kogwistar_contracts::DurableProviderEventPublicationV1 =
        serde_json::from_value(json!({
            "namespace": "conversation",
            "conversation_id": "sqlite-runtime-graph-conversation",
            "event": {
                "event_id": "sqlite-runtime-provider-event-1",
                "trace": {
                    "run_id": "sqlite-runtime-graph-run",
                    "turn_id": "turn-1",
                    "workflow_node_id": "provider-stream",
                    "step_id": "provider-response-1",
                    "provider_id": "test-provider"
                },
                "event": {
                    "contract_version": "v1",
                    "provider_event_id": "response-1:1",
                    "provider_sequence": 1,
                    "response_id": "response-1",
                    "kind": "field_completed",
                    "field_name": "text",
                    "value": "durable provider output"
                }
            }
        }))
        .unwrap();
    let provider_mutations = provider_event_graph_mutations(
        &publication,
        &GraphScope {
            namespace: "conversation".to_owned(),
            workspace_id: Some("workspace-a".to_owned()),
            graph_space: Some("conversation".to_owned()),
        },
        "workspace-a",
        1,
    );
    batch["mutations"].as_array_mut().unwrap().extend(
        provider_mutations.into_iter().map(|provider_mutation| {
            json!({
                "scope": {
                    "namespace": provider_mutation.scope.namespace,
                    "workspace_id": provider_mutation.scope.workspace_id,
                    "graph_space": provider_mutation.scope.graph_space
                },
                "table": provider_mutation.table,
                "entity_kind": provider_mutation.entity_kind,
                "event_id": provider_mutation.event_id,
                "op": provider_mutation.op,
                "payload": provider_mutation.payload,
                "record": {
                    "id": provider_mutation.record.id,
                    "document": provider_mutation.record.document,
                    "metadata": provider_mutation.record.metadata,
                    "embedding": provider_mutation.record.embedding
                },
                "embedding_dim": provider_mutation.embedding_dim,
                "model_envelope": null
            })
        }),
    );
    batch["index_jobs"] = json!([{
        "job_id": "runtime-effect-node-index-job",
        "namespace": "conversation",
        "entity_kind": "node",
        "entity_id": "runtime-effect-node",
        "index_kind": "node_embedding",
        "op": "UPSERT",
        "payload_json": null,
        "max_retries": 10
    }]);
    batch["contract_version"] = json!(2);
    batch["default_acl"] = json!({"truth_graph": "conversation"});

    let result_body = serde_json::to_vec(&json!({
                "handoff": {
                    "message_id": work["message_id"],
                    "claimed_by": work["claimed_by"],
                    "run_id": work["run_id"],
                    "step_id": work["step_id"],
                    "correlation_id": work["correlation_id"]
                },
                "effect": {
                    "contract_version": 1,
                    "effect_id": "sqlite-runtime-graph-effect",
                    "state_update": [["u", {"graph_written": true}]],
                    "successors": [],
                    "result": {"done": true},
                    "graph_mutation_batch_v2": batch,
                    "provider_event_publications": [publication]
                }
            }))
    .unwrap();
    let wrong_scope = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/internal/runtime/results".to_owned(),
            body: result_body.clone(),
            principal: json!({
                "sub": "runtime-owner", "user_id": "runtime-owner", "role": "rw",
                "ns": ["conversation"], "storage_ns": "conversation",
                "workspace": "other-workspace", "graph_space": "conversation",
                "capabilities": ["write_graph"]
            }),
        })
        .await;
    assert_eq!(wrong_scope.status, StatusCode::FORBIDDEN.as_u16());
    assert!(service.store.replay_raw_events("conversation", 0, 100).unwrap().is_empty());

    let wrong_owner = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/internal/runtime/results".to_owned(),
            body: result_body.clone(),
            principal: json!({
                "sub": "other-owner", "user_id": "other-owner", "role": "rw",
                "ns": ["conversation"], "storage_ns": "conversation",
                "workspace": "workspace-a", "graph_space": "conversation",
                "capabilities": ["write_graph"]
            }),
        })
        .await;
    assert_eq!(wrong_owner.status, StatusCode::CONFLICT.as_u16());
    assert!(service.store.replay_raw_events("conversation", 0, 100).unwrap().is_empty());

    let mut invalid_handoff: Value = serde_json::from_slice(&result_body).unwrap();
    invalid_handoff["handoff"]["correlation_id"] = json!("wrong-correlation");
    let rolled_back = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/internal/runtime/results".to_owned(),
            body: serde_json::to_vec(&invalid_handoff).unwrap(),
            principal: principal.clone(),
        })
        .await;
    assert_eq!(rolled_back.status, StatusCode::CONFLICT.as_u16());
    assert!(service.store.replay_raw_events("conversation", 0, 100).unwrap().is_empty());
    assert!(service
        .store
        .get_projected_lane_message("msg:sqlite-runtime-provider-event-1")
        .unwrap()
        .is_none());

    let mut forged_acl: Value = serde_json::from_slice(&result_body).unwrap();
    forged_acl["effect"]["graph_mutation_batch_v2"]["mutations"][0]["record"]["metadata"]["entity_type"] = json!("acl_record");
    let denied_acl = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/internal/runtime/results".to_owned(),
            body: serde_json::to_vec(&forged_acl).unwrap(),
            principal: principal.clone(),
        })
        .await;
    assert_eq!(denied_acl.status, StatusCode::UNPROCESSABLE_ENTITY.as_u16());
    assert!(service.store.replay_raw_events("conversation", 0, 100).unwrap().is_empty());

    let result = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/internal/runtime/results".to_owned(),
            body: result_body.clone(),
            principal: principal.clone(),
        })
        .await;
    assert_eq!(result.status, StatusCode::OK.as_u16(), "{}", String::from_utf8_lossy(&result.body));
    let events = service.store.replay_raw_events("conversation", 0, 100).unwrap();
    assert!(events.iter().any(|event| {
        event.entity_id == "runtime-effect-node" && event.event_id == "runtime-effect-graph-event"
    }));
    assert!(events.iter().any(|event| {
        serde_json::from_str::<Value>(&event.payload_json).is_ok_and(|payload| {
            payload["metadata"]["entity_type"] == "acl_record"
                && payload["metadata"]["acl_target_entity_id"] == "runtime-effect-node"
                && payload["metadata"]["acl_truth_graph"] == "conversation"
        })
    }));
    let provider_lane = service
        .store
        .get_projected_lane_message("msg:sqlite-runtime-provider-event-1")
        .unwrap()
        .unwrap();
    assert_eq!(provider_lane.purpose, "agent_event");
    assert_eq!(provider_lane.status, "pending");
    assert_eq!(provider_lane.run_id.as_deref(), Some("sqlite-runtime-graph-run"));
    assert_eq!(provider_lane.msg_type, "agent.provider_event");
    let receive_request = kogwistar_runtime::RuntimeNativeStepRequest {
        executor_target: kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET.to_owned(),
        operation: "agent.event.receive".to_owned(),
        op_version: "v1".to_owned(),
        run_id: "sqlite-runtime-graph-run".to_owned(),
        workflow_id: "sqlite-runtime-graph-workflow".to_owned(),
        node_id: "receive-step".to_owned(),
        conversation_id: "sqlite-runtime-graph-conversation".to_owned(),
        turn_node_id: Some("turn-1".to_owned()),
        token_id: "sqlite-runtime-graph-run".to_owned(),
        parent_token_id: None,
        step_seq: 1,
        state: Map::new(),
        runtime_routes: Vec::new(),
        ephemeral_context: Map::new(),
    };
    let receive_principal = json!({
        "sub": "runtime-owner", "user_id": "runtime-owner", "role": "rw",
        "ns": ["conversation"], "storage_ns": "conversation",
        "workspace": "workspace-a", "graph_space": "conversation",
        "capabilities": ["read_graph"]
    });
    let wrong_receive_principal = json!({
        "sub": "other-owner", "user_id": "other-owner", "role": "rw",
        "ns": ["conversation"], "storage_ns": "conversation",
        "workspace": "workspace-a", "graph_space": "conversation",
        "capabilities": ["read_graph"]
    });
    assert!(service
        .claim_native_provider_event_context(&receive_request, &wrong_receive_principal)
        .await
        .is_err());
    assert_eq!(
        service
            .store
            .get_projected_lane_message("msg:sqlite-runtime-provider-event-1")
            .unwrap()
            .unwrap()
            .status,
        "pending"
    );
    let claimed_event = service
        .claim_native_provider_event_context(&receive_request, &receive_principal)
        .await
        .unwrap()
        .expect("visible event should be claimable by its run owner");
    assert_eq!(
        claimed_event["event"]["event_id"],
        "sqlite-runtime-provider-event-1"
    );
    assert_eq!(
        claimed_event["claim"]["message_id"],
        "msg:sqlite-runtime-provider-event-1"
    );
    assert!(service
        .claim_native_provider_event_context(&receive_request, &receive_principal)
        .await
        .unwrap()
        .is_none());
    assert_eq!(
        service
            .store
            .get_projected_lane_message("msg:sqlite-runtime-provider-event-1")
            .unwrap()
            .unwrap()
            .status,
        "claimed"
    );
    let jobs = service.store.list_index_jobs(
        Some("conversation"),
        Some("PENDING"),
        None,
        Some("runtime-effect-node"),
        None,
        20,
    ).unwrap();
    assert!(!jobs.is_empty());
    assert_eq!(
        service.store.get_server_run("sqlite-runtime-graph-run").unwrap().unwrap().status,
        "succeeded"
    );
    let replay = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/internal/runtime/results".to_owned(),
            body: result_body,
            principal,
        })
        .await;
    assert_eq!(replay.status, StatusCode::OK.as_u16());
    assert_eq!(
        service.store.replay_raw_events("conversation", 0, 100).unwrap().len(),
        events.len()
    );
    assert_eq!(
        service
            .store
            .list_index_jobs(
                Some("conversation"),
                Some("PENDING"),
                None,
                Some("runtime-effect-node"),
                None,
                20,
            )
            .unwrap()
            .len(),
        jobs.len()
    );

    drop(service);
    remove_test_sqlite(&sqlite_path);
}

#[tokio::test]
async fn sqlite_provider_event_ack_commits_handler_transition_and_causal_graph_atomically() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let chroma_setup = tokio::spawn(async move {
        for _ in 0..2 {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut bytes = Vec::new();
            let mut chunk = [0_u8; 2048];
            let (header_end, content_length) = loop {
                let count = stream.read(&mut chunk).await.unwrap();
                assert!(count > 0);
                bytes.extend_from_slice(&chunk[..count]);
                if let Some(header_end) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
                    let headers = String::from_utf8_lossy(&bytes[..header_end]);
                    let content_length = headers
                        .lines()
                        .find_map(|line| {
                            let (name, value) = line.split_once(':')?;
                            name.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse::<usize>().ok())
                                .flatten()
                        })
                        .unwrap_or(0);
                    if bytes.len() >= header_end + 4 + content_length {
                        break (header_end, content_length);
                    }
                }
            };
            let request: Value = serde_json::from_slice(
                &bytes[header_end + 4..header_end + 4 + content_length],
            )
            .unwrap();
            let response = json!({
                "id": format!("collection-{}", request["name"].as_str().unwrap()),
                "metadata": {"hnsw:space": "cosine"}
            })
            .to_string();
            let wire = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                response.len(), response
            );
            stream.write_all(wire.as_bytes()).await.unwrap();
        }
    });
    let chroma_config = |collection: &str| kogwistar_store_chroma::ChromaStoreConfig {
        endpoint: endpoint.clone(),
        collection: collection.to_owned(),
        ..Default::default()
    };
    let graph = kogwistar_store_chroma::ChromaGraphStore::connect_entity_collections(
        chroma_config("provider-event-nodes"),
        chroma_config("provider-event-edges"),
    )
    .await
    .unwrap();
    chroma_setup.await.unwrap();

    let path = std::env::temp_dir().join(format!(
        "kogwistar-sqlite-provider-event-ack-{}.sqlite",
        uuid::Uuid::new_v4().simple()
    ));
    let auth_path = std::env::temp_dir().join(format!(
        "kogwistar-sqlite-provider-event-ack-auth-{}.sqlite",
        uuid::Uuid::new_v4().simple()
    ));
    let auth_store = SqliteAuthStore::open(&auth_path).unwrap();
    auth_store
        .resolve_external_identity(ResolveExternalIdentity {
            issuer: "test".to_owned(),
            subject: "sqlite-ack-owner".to_owned(),
            email: "sqlite-ack-owner@example.test".to_owned(),
            display_name: None,
            new_user_id: "sqlite-ack-owner".to_owned(),
            default_role: "rw".to_owned(),
            default_ns: "conversation".to_owned(),
        })
        .await
        .unwrap();
    drop(auth_store);
    let owner_auth = AuthConfig {
        auth_db_url: Some(format!("sqlite:///{}", auth_path.display())),
        ..AuthConfig::default()
    };
    let service = SqliteRunApplicationService::open(&path)
        .unwrap()
        .with_chroma_graph(graph)
        .with_owner_auth_config(&owner_auth)
        .await
        .unwrap();
    let run_id = "sqlite-provider-ack-run";
    let workflow_id = "sqlite-provider-ack-workflow";
    let conversation_id = "sqlite-provider-ack-conversation";
    let owner_principal = json!({
        "sub": "sqlite-ack-owner", "user_id": "sqlite-ack-owner", "role": "rw",
        "ns": ["conversation"], "storage_ns": "conversation",
        "workspace": "workspace-a", "graph_space": "conversation",
        "capabilities": ["read_graph", "write_graph"]
    });
    let binding = runtime_owner_binding_v1(&owner_principal, &service.capabilities).unwrap();
    service
        .store
        .create_server_run(ServerRunCreate {
            run_id: run_id.to_owned(),
            conversation_id: conversation_id.to_owned(),
            workflow_id: workflow_id.to_owned(),
            user_id: Some("sqlite-ack-owner".to_owned()),
            user_turn_node_id: "turn-1".to_owned(),
            status: "running".to_owned(),
        })
        .unwrap();
    let admission = service
        .store
        .append_server_run_event(
            run_id,
            "run.created",
            json!({"runtime_owner_binding": binding}).to_string(),
        )
        .unwrap();
    let start: RecordedRuntimeTransition = serde_json::from_value(json!({
        "contract_version": 1,
        "transition_id": "sqlite-provider-ack-start",
        "expected_event_seq": admission.seq,
        "kind": "start",
        "run_id": run_id,
        "workflow_id": workflow_id,
        "conversation_id": conversation_id,
        "step_seq": 0,
        "node_id": "start-step",
        "token_id": run_id,
        "initial_state": {},
        "frontier": {
            "pending": [["handler-step", 0, run_id, null]],
            "suspended": [], "join_node_ids": [], "join_outstanding": [], "join_waiters": {}
        }
    }))
    .unwrap();
    let started = service
        .store
        .apply_recorded_runtime_transition(start, false)
        .unwrap();
    let handler: RecordedRuntimeTransition = serde_json::from_value(json!({
        "contract_version": 1,
        "transition_id": "sqlite-provider-ack-handler",
        "expected_event_seq": started.event_seq,
        "kind": "recorded_step_success",
        "run_id": run_id,
        "workflow_id": workflow_id,
        "conversation_id": conversation_id,
        "step_seq": 1,
        "node_id": "handler-step",
        "token_id": run_id,
        "state_update": [["u", {"agent_event_handling_step_seq": 1}]],
        "frontier": {
            "pending": [["ack-step", 0, run_id, null]],
            "suspended": [], "join_node_ids": [], "join_outstanding": [], "join_waiters": {}
        },
        "result": {"handled": true}
    }))
    .unwrap();
    service
        .store
        .apply_recorded_runtime_transition(handler, false)
        .unwrap();

    let event: kogwistar_contracts::DurableProviderEventV1 = serde_json::from_value(json!({
        "event_id": "sqlite-provider-ack-event",
        "trace": {
            "run_id": run_id,
            "turn_id": "turn-1",
            "workflow_node_id": "provider-step",
            "step_id": "provider-response",
            "provider_id": "test-provider"
        },
        "event": {
            "contract_version": "v1",
            "provider_event_id": "response:1",
            "provider_sequence": 1,
            "response_id": "response",
            "kind": "field_completed",
            "field_name": "text",
            "value": "durable answer"
        }
    }))
    .unwrap();
    let event_message_id = format!("msg:{}", event.event_id);
    service
        .store
        .project_lane_message(NewProjectedLaneMessage {
            message_id: event_message_id.clone(),
            namespace: "conversation".to_owned(),
            purpose: "agent_event".to_owned(),
            inbox_id: "agent-provider-events".to_owned(),
            conversation_id: conversation_id.to_owned(),
            recipient_id: "agent-event-consumer".to_owned(),
            sender_id: "agent-runtime".to_owned(),
            msg_type: "agent.provider_event".to_owned(),
            status: "pending".to_owned(),
            created_at: 1,
            available_at: 1,
            run_id: Some(run_id.to_owned()),
            step_id: Some("provider-response".to_owned()),
            correlation_id: Some(event.event_id.clone()),
            payload_json: Some(serde_json::to_string(&event).unwrap()),
            error_json: None,
        })
        .unwrap();
    let event_claim = service
        .store
        .claim_projected_lane_messages_filtered(LaneMessageClaimFilter {
            namespace: "conversation".to_owned(),
            inbox_id: "agent-provider-events".to_owned(),
            claimed_by: "event-consumer-attempt".to_owned(),
            message_ids: Some(vec![event_message_id.clone()]),
            run_id: Some(run_id.to_owned()),
            msg_type: Some("agent.provider_event".to_owned()),
            recipient_id: Some("agent-event-consumer".to_owned()),
            limit: 1,
            lease_seconds: 60,
        })
        .unwrap();
    assert_eq!(event_claim.len(), 1);

    let runtime_message_id = "sqlite-provider-ack-runtime";
    service
        .store
        .project_lane_message(NewProjectedLaneMessage {
            message_id: runtime_message_id.to_owned(),
            namespace: "workflow".to_owned(),
            purpose: "system".to_owned(),
            inbox_id: "workflow-runtime".to_owned(),
            conversation_id: conversation_id.to_owned(),
            recipient_id: "python-worker".to_owned(),
            sender_id: "rust-scheduler".to_owned(),
            msg_type: "workflow.step.execute".to_owned(),
            status: "pending".to_owned(),
            created_at: 1,
            available_at: 1,
            run_id: Some(run_id.to_owned()),
            step_id: Some("ack-step".to_owned()),
            correlation_id: Some(run_id.to_owned()),
            payload_json: Some(json!({
                "workflow_id": workflow_id,
                "conversation_id": conversation_id,
                "token_id": run_id,
                "parent_token_id": null,
                "op": "app/event-ack"
            }).to_string()),
            error_json: None,
        })
        .unwrap();
    let runtime_claim = service
        .store
        .claim_projected_lane_messages_filtered(LaneMessageClaimFilter {
            namespace: "workflow".to_owned(),
            inbox_id: "workflow-runtime".to_owned(),
            claimed_by: "runtime-attempt".to_owned(),
            message_ids: Some(vec![runtime_message_id.to_owned()]),
            run_id: Some(run_id.to_owned()),
            msg_type: Some("workflow.step.execute".to_owned()),
            recipient_id: Some("python-worker".to_owned()),
            limit: 1,
            lease_seconds: 60,
        })
        .unwrap();
    assert_eq!(runtime_claim.len(), 1);
    let handoff = RecordedWorkerHandoff {
        message_id: runtime_message_id.to_owned(),
        claimed_by: "runtime-attempt".to_owned(),
        run_id: run_id.to_owned(),
        step_id: "ack-step".to_owned(),
        correlation_id: run_id.to_owned(),
    };
    let acknowledgement = kogwistar_runtime::ProviderEventAcknowledgementV1 {
        message_id: event_message_id,
        claimed_by: "event-consumer-attempt".to_owned(),
        event_id: event.event_id,
    };
    let effect: RecordedWorkerSuccessEffect = serde_json::from_value(json!({
        "contract_version": 1,
        "effect_id": "sqlite-provider-ack-effect",
        "state_update": [],
        "successors": [],
        "result": {"handled": true},
        "provider_event_acknowledgements": [acknowledgement.clone()]
    }))
    .unwrap();
    let execute = |effect: RecordedWorkerSuccessEffect| {
        let body = serde_json::to_vec(&json!({
            "handoff": handoff,
            "effect": effect
        }))
        .unwrap();
        service.execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/internal/runtime/results".to_owned(),
            body,
            principal: json!({
                "sub": "python-runtime-worker",
                "role": "admin",
                "capabilities": ["runtime.submit_result"]
            }),
        })
    };
    let mut bad_effect = effect.clone();
    bad_effect.provider_event_acknowledgements[0].claimed_by = "wrong-claimer".to_owned();
    assert_eq!(
        execute(bad_effect).await.status,
        StatusCode::CONFLICT.as_u16()
    );
    assert_eq!(
        service
            .store
            .get_projected_lane_message(&format!("msg:{}", acknowledgement.event_id))
            .unwrap()
            .unwrap()
            .status,
        "claimed"
    );
    assert_eq!(
        service
            .store
            .read_recorded_runtime_state(run_id, workflow_id, conversation_id)
            .unwrap()
            .unwrap()
            .last_step_seq,
        1
    );
    let applied = execute(effect.clone()).await;
    assert_eq!(
        applied.status,
        StatusCode::OK.as_u16(),
        "{}",
        String::from_utf8_lossy(&applied.body)
    );
    assert_eq!(
        service
            .store
            .read_recorded_runtime_state(run_id, workflow_id, conversation_id)
            .unwrap()
            .unwrap()
            .last_step_seq,
        2
    );
    assert_eq!(
        service
            .store
            .get_projected_lane_message(&format!("msg:{}", acknowledgement.event_id))
            .unwrap()
            .unwrap()
            .status,
        "completed"
    );
    let graph_events = service
        .store
        .replay_raw_events("conversation", 0, 100)
        .unwrap();
    assert!(graph_events.iter().any(|row| {
        row.entity_id == format!("wf_step|{run_id}|2")
    }));
    assert!(graph_events.iter().any(|row| {
        serde_json::from_str::<Value>(&row.payload_json).is_ok_and(|payload| {
            payload["metadata"]["relation"] == "acknowledged_by_step"
        })
    }));
    assert!(service
        .store
        .list_server_run_events(run_id, 0, 100)
        .unwrap()
        .iter()
        .any(|event| event.event_type == "workflow.provider_event.handled.v1"));
    let graph_event_count = service
        .store
        .replay_raw_events("conversation", 0, 100)
        .unwrap()
        .len();
    let handled_trace_count = service
        .store
        .list_server_run_events(run_id, 0, 100)
        .unwrap()
        .iter()
        .filter(|event| event.event_type == "workflow.provider_event.handled.v1")
        .count();
    let retried = execute(effect).await;
    assert_eq!(retried.status, StatusCode::OK.as_u16());
    assert_eq!(
        service
            .store
            .replay_raw_events("conversation", 0, 100)
            .unwrap()
            .len(),
        graph_event_count
    );
    assert_eq!(
        service
            .store
            .list_server_run_events(run_id, 0, 100)
            .unwrap()
            .iter()
            .filter(|event| event.event_type == "workflow.provider_event.handled.v1")
            .count(),
        handled_trace_count
    );
    drop(service);
    remove_test_sqlite(&path);
    remove_test_sqlite(&auth_path);
}

#[tokio::test]
async fn sqlite_chroma_wisdom_upsert_uses_canonical_events_and_acl_before_projection() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let chroma_setup = tokio::spawn(async move {
        for _ in 0..2 {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut bytes = Vec::new();
            let mut chunk = [0_u8; 2048];
            let (header_end, content_length) = loop {
                let count = stream.read(&mut chunk).await.unwrap();
                assert!(count > 0);
                bytes.extend_from_slice(&chunk[..count]);
                if let Some(header_end) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
                    let headers = String::from_utf8_lossy(&bytes[..header_end]);
                    let content_length = headers
                        .lines()
                        .find_map(|line| {
                            let (name, value) = line.split_once(':')?;
                            name.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse::<usize>().ok())
                                .flatten()
                        })
                        .unwrap_or(0);
                    if bytes.len() >= header_end + 4 + content_length {
                        break (header_end, content_length);
                    }
                }
            };
            let request: Value = serde_json::from_slice(
                &bytes[header_end + 4..header_end + 4 + content_length],
            )
            .unwrap();
            let response = json!({
                "id": format!("collection-{}", request["name"].as_str().unwrap()),
                "metadata": {"hnsw:space": "cosine"}
            })
            .to_string();
            let wire = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                response.len(),
                response
            );
            stream.write_all(wire.as_bytes()).await.unwrap();
        }
    });
    let chroma_config = |collection: &str| kogwistar_store_chroma::ChromaStoreConfig {
        endpoint: endpoint.clone(),
        collection: collection.to_owned(),
        ..Default::default()
    };
    let graph = kogwistar_store_chroma::ChromaGraphStore::connect_entity_collections(
        chroma_config("wisdom-nodes"),
        chroma_config("wisdom-edges"),
    )
    .await
    .unwrap();
    chroma_setup.await.unwrap();

    let sqlite_path = std::env::temp_dir().join(format!(
        "kogwistar-sqlite-chroma-wisdom-{}.sqlite",
        uuid::Uuid::new_v4()
    ));
    let provider = Arc::new(FixedEmbeddingProvider {
        profile: kogwistar_engine::embedding::EmbeddingProfile {
            provider: "test".to_owned(),
            model: "fixed-v1".to_owned(),
            dimension: 2,
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
        },
        vector: vec![0.6, 0.8],
        calls: Arc::new(Mutex::new(Vec::new())),
        fail: None,
    });
    let service = SqliteRunApplicationService::open(&sqlite_path)
        .unwrap()
        .with_chroma_graph(graph)
        .with_embedding_provider(provider);
    assert!(service.supports_mcp_tool("root", "wisdom.kg_upsert_graph"));
    assert!(service.supports_http_operation("POST", PUBLIC_WISDOM_GRAPH_UPSERT_ROUTE));

    let fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/python-graph-model-payloads-v1.json"
    )))
    .unwrap();
    let mut node = fixture["cases"][0]["payload"].clone();
    node["id"] = json!("sqlite-wisdom-node");
    node["embedding"] = Value::Null;
    let mut edge = fixture["cases"][1]["payload"].clone();
    edge["id"] = json!("sqlite-wisdom-edge");
    edge["source_ids"] = json!(["sqlite-wisdom-node"]);
    edge["target_ids"] = json!([]);
    edge["embedding"] = Value::Null;
    let body = serde_json::to_vec(&json!({"inp": {"nodes": [node], "edges": [edge]}})).unwrap();
    let principal = json!({
        "sub": "alice", "user_id": "alice", "role": "rw",
        "ns": ["wisdom"], "storage_ns": "wisdom", "graph_space": "wisdom",
        "capabilities": ["write_graph"]
    });
    assert!(service.supports_mcp_tool_for_principal(
        "root",
        "wisdom.kg_upsert_graph",
        &principal,
    ));
    for denied_principal in [
        json!({"sub":"alice", "role":"ro", "ns":["wisdom"], "capabilities":["write_graph"]}),
        json!({"sub":"alice", "role":"rw", "ns":["docs"], "capabilities":["write_graph"]}),
        json!({"sub":"alice", "role":"rw", "ns":["wisdom"], "capabilities":["read_graph"]}),
    ] {
        assert!(!service.supports_mcp_tool_for_principal(
            "root",
            "wisdom.kg_upsert_graph",
            &denied_principal,
        ));
    }
    assert!(!service.supports_mcp_tool_for_principal(
        "workflow",
        "wisdom.kg_upsert_graph",
        &principal,
    ));
    let denied = service.execute(ApiEffectRequest {
        contract_version: 1,
        method: "POST".to_owned(),
        path_and_query: PUBLIC_WISDOM_GRAPH_UPSERT_ROUTE.to_owned(),
        body: body.clone(),
        principal: json!({"sub": "alice", "role": "ro", "ns": ["wisdom"]}),
    }).await;
    assert_eq!(denied.status, StatusCode::FORBIDDEN.as_u16());
    assert!(service.store.replay_raw_events("wisdom", 0, 100).unwrap().is_empty());

    let response = service.execute(ApiEffectRequest {
        contract_version: 1,
        method: "POST".to_owned(),
        path_and_query: PUBLIC_WISDOM_GRAPH_UPSERT_ROUTE.to_owned(),
        body,
        principal,
    }).await;
    assert_eq!(response.status, StatusCode::OK.as_u16(), "{}", String::from_utf8_lossy(&response.body));
    let payload: Value = serde_json::from_slice(&response.body).unwrap();
    assert_eq!(payload["projection_consistency"], "eventual");
    assert_eq!(payload["node_ids"][0], "sqlite-wisdom-node");
    assert_eq!(payload["edge_ids"][0], "sqlite-wisdom-edge");

    let events = service.store.replay_raw_events("wisdom", 0, 100).unwrap();
    assert!(events.iter().any(|event| event.entity_id == "sqlite-wisdom-node"));
    assert!(events.iter().any(|event| event.entity_id == "sqlite-wisdom-edge"));
    assert!(events.iter().any(|event| {
        serde_json::from_str::<Value>(&event.payload_json)
            .ok()
            .and_then(|payload| payload.get("metadata").cloned())
            .and_then(|metadata| metadata.get("entity_type").cloned())
            == Some(json!("acl_record"))
    }), "host-derived default ACL records are part of the canonical event transaction");
    let jobs = service.store.list_index_jobs(Some("wisdom"), Some("PENDING"), None, None, None, 100).unwrap();
    assert!(jobs.iter().any(|job| job.entity_id == "sqlite-wisdom-node"));
    assert!(jobs.iter().any(|job| job.entity_id == "sqlite-wisdom-edge"));
    assert!(jobs.iter().all(|job| job.namespace == "wisdom"));

    let committed_event_count = events.len();
    let mut private_reference = fixture["cases"][1]["payload"].clone();
    private_reference["id"] = json!("bob-reference-edge");
    private_reference["source_ids"] = json!(["sqlite-wisdom-node"]);
    private_reference["target_ids"] = json!([]);
    private_reference["embedding"] = Value::Null;
    let hidden_reference = service.execute(ApiEffectRequest {
        contract_version: 1,
        method: "POST".to_owned(),
        path_and_query: PUBLIC_WISDOM_GRAPH_UPSERT_ROUTE.to_owned(),
        body: serde_json::to_vec(&json!({"inp": {"nodes": [], "edges": [private_reference]}})).unwrap(),
        principal: json!({
            "sub": "bob", "user_id": "bob", "role": "rw",
            "ns": ["wisdom"], "storage_ns": "wisdom", "graph_space": "wisdom",
            "capabilities": ["write_graph"]
        }),
    }).await;
    assert_eq!(hidden_reference.status, StatusCode::CONFLICT.as_u16());

    let mut missing_reference = fixture["cases"][1]["payload"].clone();
    missing_reference["id"] = json!("missing-reference-edge");
    missing_reference["source_ids"] = json!(["absent-wisdom-node"]);
    missing_reference["target_ids"] = json!([]);
    missing_reference["embedding"] = Value::Null;
    let missing_reference = service.execute(ApiEffectRequest {
        contract_version: 1,
        method: "POST".to_owned(),
        path_and_query: PUBLIC_WISDOM_GRAPH_UPSERT_ROUTE.to_owned(),
        body: serde_json::to_vec(&json!({"inp": {"nodes": [], "edges": [missing_reference]}})).unwrap(),
        principal: json!({
            "sub": "alice", "user_id": "alice", "role": "rw",
            "ns": ["wisdom"], "storage_ns": "wisdom", "graph_space": "wisdom",
            "workspace": "other-workspace", "capabilities": ["write_graph"]
        }),
    }).await;
    assert_eq!(missing_reference.status, StatusCode::CONFLICT.as_u16());
    assert_eq!(service.store.replay_raw_events("wisdom", 0, 100).unwrap().len(), committed_event_count);

    drop(service);
    remove_test_sqlite(&sqlite_path);
}
