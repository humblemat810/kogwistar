    #[tokio::test]
    async fn conversation_create_rejects_non_owner_before_storage() {
        let service = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "conversation_create_owner_auth_contract",
        )
        .unwrap();
        let response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/syscall/v1/create_conversation".to_owned(),
                body: serde_json::to_vec(&json!({
                    "version": "v1",
                    "op": "create_conversation",
                    "args": {"user_id": "bob", "conversation_id": "conv-1"}
                }))
                .unwrap(),
                principal: json!({
                    "sub": "alice",
                    "role": "rw",
                    "ns": "conversation",
                    "storage_ns": "conversation",
                    "capabilities": ["spawn_process"]
                }),
            })
            .await;
        assert_eq!(response.status, StatusCode::FORBIDDEN.as_u16());
        assert_eq!(
            serde_json::from_slice::<Value>(&response.body).unwrap()["code"],
            "KOGWISTAR_RUN_OWNER_FORBIDDEN"
        );
    }

    #[tokio::test]
    async fn conversation_create_embedding_failure_fails_before_storage() {
        use std::sync::{Arc, Mutex, atomic::AtomicBool};

        let service = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "conversation_create_embedding_failure_contract",
        )
        .unwrap()
        .with_embedding_provider(Arc::new(FixedEmbeddingProvider {
            profile: kogwistar_engine::embedding::EmbeddingProfile {
                provider: "test".to_owned(),
                model: "failing-conversation-start-v1".to_owned(),
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
            },
            vector: vec![0.6, 0.8, 0.0],
            calls: Arc::new(Mutex::new(Vec::new())),
            fail: Some(Arc::new(AtomicBool::new(true))),
        }));
        let response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/syscall/v1/create_conversation".to_owned(),
                body: serde_json::to_vec(&json!({
                    "version": "v1",
                    "op": "create_conversation",
                    "args": {"user_id": "alice", "conversation_id": "conv-1"}
                }))
                .unwrap(),
                principal: json!({
                    "sub": "alice",
                    "role": "rw",
                    "ns": "conversation",
                    "storage_ns": "conversation",
                    "capabilities": ["spawn_process"]
                }),
            })
            .await;
        assert_eq!(response.status, StatusCode::BAD_GATEWAY.as_u16());
        assert_eq!(
            serde_json::from_slice::<Value>(&response.body).unwrap()["code"],
            "KOGWISTAR_EMBEDDING_FAILED"
        );
    }

    #[tokio::test]
    #[ignore = "requires KOGWISTAR_TEST_PG_DSN; run explicitly with --include-ignored"]
    async fn postgres_conversation_create_persists_private_start_node_when_dsn_available() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use kogwistar_store::EventReadStore;
        use std::sync::{Arc, Mutex};
        use tower::ServiceExt;

        let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
            .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
        let schema = format!("rust_conversation_create_{}", uuid::Uuid::new_v4().simple());
        let embedding_calls = Arc::new(Mutex::new(Vec::new()));
        let service = PostgresRunApplicationService::from_dsn(&dsn, &schema)
            .unwrap()
            .with_embedding_provider(Arc::new(FixedEmbeddingProvider {
                profile: kogwistar_engine::embedding::EmbeddingProfile {
                    provider: "test".to_owned(),
                    model: "fixed-conversation-start-v1".to_owned(),
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
                },
                vector: vec![0.6, 0.8, 0.0],
                calls: Arc::clone(&embedding_calls),
                fail: None,
            }));
        service.ensure_graph_type_schemas().await.unwrap();
        service
            .conversation_store
            .create_graph_schema(3, kogwistar_store_postgres::GraphTableNames::default())
            .await
            .unwrap();

        let create = |user_id: &str| ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/syscall/v1/create_conversation".to_owned(),
            body: serde_json::to_vec(&json!({
                "version": "v1",
                "op": "create_conversation",
                "args": {
                    "user_id": user_id,
                    "conversation_id": "conv-create-test",
                    "start_node_id": "conv-create-start"
                }
            }))
            .unwrap(),
            principal: json!({
                "sub": "alice",
                "role": "rw",
                "ns": "conversation",
                "storage_ns": "conversation",
                "capabilities": ["spawn_process"]
            }),
        };
        let created = service.execute(create("alice")).await;
        assert_eq!(
            created.status,
            StatusCode::OK.as_u16(),
            "conversation creation failed: {}",
            String::from_utf8_lossy(&created.body)
        );
        let created: Value = serde_json::from_slice(&created.body).unwrap();
        assert_eq!(created["status"], "ok");
        let syscall_result_fixture: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/conversation-create-syscall-result-v1.json"
        ))
        .unwrap();
        assert_eq!(created["result"], syscall_result_fixture);
        let start_projection = service
            .conversation_store
            .graph_projection_records(GraphProjectionRead {
                scope: GraphScope {
                    namespace: "conversation".to_owned(),
                    workspace_id: None,
                    graph_space: None,
                },
                table: "gke_nodes".to_owned(),
                ids: Some(vec!["conv-create-start".to_owned()]),
                metadata: MetadataFilter::default(),
                limit: 1,
            })
            .await
            .unwrap();
        assert_eq!(start_projection[0].embedding, Some(vec![0.6, 0.8, 0.0]));
        assert_eq!(embedding_calls.lock().unwrap()[0].len(), 1);
        let events = service
            .conversation_store
            .replay_events("conversation", 0, 100)
            .await
            .unwrap();
        let start_event = events
            .iter()
            .find(|event| event.entity_id == "conv-create-start")
            .expect("conversation start event is durable");
        assert!(start_event.payload.get("embedding").is_none());
        let conversation_create_fixture: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/conversation-create-start-node-v1.json"
        ))
        .unwrap();
        assert_eq!(start_event.payload, conversation_create_fixture["payload"]);

        let details = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: "/api/conversations/conv-create-test".to_owned(),
                body: Vec::new(),
                principal: json!({
                    "sub": "alice",
                    "role": "ro",
                    "ns": "conversation",
                    "storage_ns": "conversation",
                    "capabilities": ["read_graph"]
                }),
            })
            .await;
        assert_eq!(details.status, StatusCode::OK.as_u16());
        let details: Value = serde_json::from_slice(&details.body).unwrap();
        assert_eq!(details["conversation_id"], "conv-create-test");
        assert_eq!(details["user_id"], "alice");
        assert_eq!(details["status"], "active");
        assert_eq!(details["start_node_id"], "conv-create-start");
        assert_eq!(details["tail_node_id"], "conv-create-start");
        assert_eq!(details["turn_count"], 0);

        let outsider_details = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: "/api/conversations/conv-create-test".to_owned(),
                body: Vec::new(),
                principal: json!({
                    "sub": "bob",
                    "role": "ro",
                    "ns": "conversation",
                    "storage_ns": "conversation",
                    "capabilities": ["read_graph"]
                }),
            })
            .await;
        assert_eq!(outsider_details.status, StatusCode::NOT_FOUND.as_u16());

        let transcript = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: "/api/conversations/conv-create-test/turns".to_owned(),
                body: Vec::new(),
                principal: json!({
                    "sub": "alice",
                    "role": "ro",
                    "ns": "conversation",
                    "storage_ns": "conversation",
                    "capabilities": ["read_graph"]
                }),
            })
            .await;
        assert_eq!(transcript.status, StatusCode::OK.as_u16());
        let transcript: Value = serde_json::from_slice(&transcript.body).unwrap();
        assert_eq!(transcript, json!({"conversation_id": "conv-create-test", "turns": []}));

        let owner_mismatch = service.execute(create("bob")).await;
        assert_eq!(owner_mismatch.status, StatusCode::FORBIDDEN.as_u16());
        let duplicate = service.execute(create("alice")).await;
        assert_eq!(duplicate.status, StatusCode::CONFLICT.as_u16());

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let secret = "conversation-create-mcp-test-secret";
        let token = jsonwebtoken::encode(
            &jsonwebtoken::Header::new(jsonwebtoken::Algorithm::HS256),
            &json!({
                "sub": "alice",
                "role": "rw",
                "ns": ["conversation"],
                "storage_ns": "conversation",
                "capabilities": ["spawn_process"],
                "iss": "local",
                "exp": now + 300
            }),
            &jsonwebtoken::EncodingKey::from_secret(secret.as_bytes()),
        )
        .unwrap();
        let no_spawn_process_token = jsonwebtoken::encode(
            &jsonwebtoken::Header::new(jsonwebtoken::Algorithm::HS256),
            &json!({
                "sub": "alice",
                "role": "rw",
                "ns": ["conversation"],
                "storage_ns": "conversation",
                "capabilities": [],
                "iss": "local",
                "exp": now + 300
            }),
            &jsonwebtoken::EncodingKey::from_secret(secret.as_bytes()),
        )
        .unwrap();
        let transcript_read_token = jsonwebtoken::encode(
            &jsonwebtoken::Header::new(jsonwebtoken::Algorithm::HS256),
            &json!({
                "sub": "alice",
                "role": "ro",
                "ns": ["conversation"],
                "storage_ns": "conversation",
                "capabilities": ["read_graph"],
                "iss": "local",
                "exp": now + 300
            }),
            &jsonwebtoken::EncodingKey::from_secret(secret.as_bytes()),
        )
        .unwrap();
        let wrong_namespace_token = jsonwebtoken::encode(
            &jsonwebtoken::Header::new(jsonwebtoken::Algorithm::HS256),
            &json!({
                "sub": "alice",
                "role": "ro",
                "ns": ["workflow"],
                "storage_ns": "workflow",
                "capabilities": ["read_graph"],
                "iss": "local",
                "exp": now + 300
            }),
            &jsonwebtoken::EncodingKey::from_secret(secret.as_bytes()),
        )
        .unwrap();
        let application = Arc::new(service);
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: Vec::new(),
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig {
                    mode: "dev".to_owned(),
                    algorithm: Some("HS256".to_owned()),
                    key: Some(secret.to_owned()),
                    jwks_json: None,
                    issuer: Some("local".to_owned()),
                    audience: None,
                    oidc_providers_json: None,
                    auth_db_url: None,
                },
            },
            application.clone(),
        );
        let denied_create = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/conversation")
                    .header("content-type", "application/json")
                    .header(
                        "authorization",
                        format!("Bearer {no_spawn_process_token}"),
                    )
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":"create-denied","method":"tools/call","params":{"name":"conversation.create","arguments":{"user_id":"alice","conversation_id":"conv-create-denied","start_node_id":"conv-create-denied-start"}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(denied_create.status(), StatusCode::OK);
        let denied_body = to_bytes(denied_create.into_body(), 8192).await.unwrap();
        let denied_result: Value = serde_json::from_slice(&denied_body).unwrap();
        assert_eq!(denied_result["error"]["code"], -32000);
        let denied_start = application
            .conversation_store
            .graph_projection_records(GraphProjectionRead {
                scope: GraphScope {
                    namespace: "conversation".to_owned(),
                    workspace_id: None,
                    graph_space: None,
                },
                table: "gke_nodes".to_owned(),
                ids: Some(vec!["conv-create-denied-start".to_owned()]),
                metadata: MetadataFilter::default(),
                limit: 1,
            })
            .await
            .unwrap();
        assert!(denied_start.is_empty());
        let mcp_create = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/conversation")
                    .header("content-type", "application/json")
                    .header("authorization", format!("Bearer {token}"))
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":"create-live","method":"tools/call","params":{"name":"conversation.create","arguments":{"user_id":"alice","conversation_id":"conv-create-mcp","start_node_id":"conv-create-mcp-start"}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(mcp_create.status(), StatusCode::OK);
        let mcp_body = to_bytes(mcp_create.into_body(), 8192).await.unwrap();
        let mcp_result: Value = serde_json::from_slice(&mcp_body).unwrap();
        assert_eq!(mcp_result["result"]["conversation_id"], "conv-create-mcp");
        assert_eq!(mcp_result["result"]["start_node_id"], "conv-create-mcp-start");
        let mcp_created_details = application
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: "/api/conversations/conv-create-mcp".to_owned(),
                body: Vec::new(),
                principal: json!({
                    "sub": "alice",
                    "role": "ro",
                    "ns": "conversation",
                    "storage_ns": "conversation",
                    "capabilities": ["read_graph"]
                }),
            })
            .await;
        assert_eq!(mcp_created_details.status, StatusCode::OK.as_u16());
        assert_eq!(
            serde_json::from_slice::<Value>(&mcp_created_details.body).unwrap()["start_node_id"],
            "conv-create-mcp-start"
        );

        let transcript_call = |token: &str, request_id: &str| {
            Request::builder()
                .method("POST")
                .uri("/mcp/conversation")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::from(
                    serde_json::to_vec(&json!({
                        "jsonrpc":"2.0",
                        "id":request_id,
                        "method":"tools/call",
                        "params": {
                            "name":"conversation.get_transcript",
                            "arguments":{"conversation_id":"conv-create-mcp"}
                        }
                    }))
                    .unwrap(),
                ))
                .unwrap()
        };
        let mcp_transcript = app
            .clone()
            .oneshot(transcript_call(&transcript_read_token, "transcript-allowed"))
            .await
            .unwrap();
        assert_eq!(mcp_transcript.status(), StatusCode::OK);
        let mcp_transcript: Value = serde_json::from_slice(
            &to_bytes(mcp_transcript.into_body(), 8192).await.unwrap(),
        )
        .unwrap();
        assert_eq!(
            mcp_transcript["result"],
            json!({"conversation_id":"conv-create-mcp", "turns":[]})
        );

        let denied_missing_capability = app
            .clone()
            .oneshot(transcript_call(&no_spawn_process_token, "transcript-no-capability"))
            .await
            .unwrap();
        let denied_body = to_bytes(denied_missing_capability.into_body(), 8192)
            .await
            .unwrap();
        let denied_value: Value = serde_json::from_slice(&denied_body).unwrap();
        assert_eq!(denied_value["error"]["code"], -32602);
        assert_eq!(
            denied_value["error"]["message"],
            "Tool not advertised by this application service"
        );
        assert!(denied_value.get("result").is_none());

        let denied_wrong_namespace = app
            .oneshot(transcript_call(&wrong_namespace_token, "transcript-wrong-namespace"))
            .await
            .unwrap();
        let denied_body = to_bytes(denied_wrong_namespace.into_body(), 8192)
            .await
            .unwrap();
        let denied_value: Value = serde_json::from_slice(&denied_body).unwrap();
        assert_eq!(denied_value["error"]["code"], -32602);
        assert_eq!(
            denied_value["error"]["message"],
            "Tool not advertised by this application service"
        );
        assert!(denied_value.get("result").is_none());

        drop(application);
        let (cleanup, connection) = tokio_postgres::connect(&dsn, tokio_postgres::NoTls)
            .await
            .unwrap();
        tokio::spawn(async move {
            let _ = connection.await;
        });
        cleanup
            .batch_execute(&format!(
                "DROP SCHEMA IF EXISTS \"{schema}_conversation\" CASCADE; \
                 DROP SCHEMA IF EXISTS \"{schema}_workflow\" CASCADE; \
                 DROP SCHEMA IF EXISTS \"{schema}_wisdom\" CASCADE; \
                 DROP SCHEMA IF EXISTS \"{schema}\" CASCADE"
            ))
            .await
            .unwrap();
    }
