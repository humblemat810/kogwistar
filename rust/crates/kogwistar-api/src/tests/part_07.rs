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
