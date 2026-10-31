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
                "provider_id": "openai-responses"
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
    let service = PostgresRunApplicationService::from_dsn(&dsn, &schema)
        .unwrap()
        .with_owner_auth_config(&owner_auth)
        .await
        .unwrap();
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
    let producer: kogwistar_runtime::RecordedRuntimeTransition = serde_json::from_value(json!({
        "contract_version": 1,
        "transition_id": "causal-producer-step",
        "expected_event_seq": 2,
        "kind": "recorded_step_success",
        "run_id": run_id,
        "workflow_id": workflow_id,
        "conversation_id": conversation_id,
        "step_seq": 0,
        "node_id": "provider-step",
        "token_id": run_id,
        "state_update": [["u", {}]],
        "frontier": {
            "pending": [["handler-step", 0, run_id, null]],
            "suspended": [], "join_node_ids": [], "join_outstanding": [], "join_waiters": {}
        },
        "result": {"published": true}
    })).unwrap();
    service.store.apply_recorded_runtime_transition(producer, false).await.unwrap();

    let event: kogwistar_contracts::DurableProviderEventV1 = serde_json::from_value(json!({
        "event_id": "provider-event-causal-event",
        "trace": {
            "run_id": run_id,
            "turn_id": "causal-user-turn",
            "workflow_node_id": "provider.request",
            "step_id": "provider-response:causal",
            "producer_step_seq": 0,
            "provider_id": "fixture-provider",
            "provider_response_id": "causal-response"
        },
        "event": {
            "kind": "field_completed",
            "provider_event_id": "causal-response:1",
            "provider_sequence": 1,
            "response_id": "causal-response",
            "field_name": "answer",
            "value": "grounded test"
        }
    })).unwrap();
    service
        .persist_native_provider_event(
            event.clone(),
            conversation_id.to_owned(),
            json!({"event": "provider.semantic_event.observed"}),
        )
        .await
        .unwrap();
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

    let handler: kogwistar_runtime::RecordedRuntimeTransition = serde_json::from_value(json!({
        "contract_version": 1,
        "transition_id": "causal-handler-step",
        "expected_event_seq": 4,
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
            .get_projected_lane_message(&format!("msg:{}", event.event_id))
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
        request(ack_effect), None, None, kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET,
    ).await;
    assert_eq!(
        applied.status,
        StatusCode::OK.as_u16(),
        "{}",
        String::from_utf8_lossy(&applied.body)
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
        service.store.get_projected_lane_message(&format!("msg:{}", event.event_id)).await.unwrap().unwrap().status,
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
