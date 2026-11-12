    #[tokio::test]
    async fn postgres_send_message_checks_actor_and_request_before_storage() {
        let service = PostgresRunApplicationService::from_dsn(
            "postgresql://worker:secret@127.0.0.1:1/unreachable",
            "send_message_admission_authority",
        )
        .unwrap();
        let request = |namespace: &str, capabilities: &[&str], text: &str| ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/syscall/v1/send_message".to_owned(),
            body: serde_json::to_vec(&json!({
                "version": "v1",
                "op": "send_message",
                "args": {"conversation_id": "conversation-1", "text": text}
            }))
            .unwrap(),
            principal: json!({
                "user_id": "actor",
                "sub": "actor@example.test",
                "role": "rw",
                "ns": namespace,
                "storage_ns": "conversation",
                "capabilities": capabilities,
            }),
        };
        let denied = service
            .execute(request("conversation", &["read_graph"], "hello"))
            .await;
        assert_eq!(denied.status, StatusCode::FORBIDDEN.as_u16());
        let invalid = service
            .execute(request("conversation", &["send_message"], "  "))
            .await;
        assert_eq!(invalid.status, StatusCode::BAD_REQUEST.as_u16());
        let missing_fields = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/syscall/v1/send_message".to_owned(),
                body: serde_json::to_vec(&json!({
                    "version": "v1",
                    "op": "send_message",
                    "args": {}
                }))
                .unwrap(),
                principal: json!({
                    "user_id": "actor",
                    "sub": "actor@example.test",
                    "role": "rw",
                    "ns": "conversation",
                    "storage_ns": "conversation",
                    "capabilities": ["send_message"]
                }),
            })
            .await;
        assert_eq!(missing_fields.status, StatusCode::BAD_REQUEST.as_u16());
        let wrong_namespace = service
            .execute(request("workflow", &["send_message"], "hello"))
            .await;
        assert_eq!(wrong_namespace.status, StatusCode::FORBIDDEN.as_u16());
    }

    #[tokio::test]
    #[ignore = "requires KOGWISTAR_TEST_PG_DSN; run explicitly against disposable PostgreSQL"]
    async fn postgres_send_message_admission_is_one_graph_run_event_lane_transaction() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use tower::ServiceExt;

        let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
            .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
        let schema = format!("rust_send_message_{}", uuid::Uuid::new_v4().simple());
        let auth_path = std::env::temp_dir().join(format!(
            "kogwistar-send-message-auth-{}.sqlite",
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
            default_ns: "alpha".to_owned(),
        })
        .await
        .unwrap();
        drop(auth);
        let auth_config = AuthConfig {
            auth_db_url: Some(format!("sqlite:///{}", auth_path.display())),
            ..AuthConfig::default()
        };
        let provider = Arc::new(FixedEmbeddingProvider {
            profile: kogwistar_engine::embedding::EmbeddingProfile {
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
            },
            vector: vec![0.6, 0.8, 0.0],
            calls: Arc::new(Mutex::new(Vec::new())),
            fail: None,
        });
        let service = PostgresRunApplicationService::from_dsn(&dsn, &schema)
            .unwrap()
            .with_owner_auth_config(&auth_config)
            .await
            .unwrap()
            .with_embedding_provider(provider);
        service.ensure_graph_type_schemas().await.unwrap();
        service
            .conversation_store
            .create_graph_schema(3, kogwistar_store_postgres::GraphTableNames::default())
            .await
            .unwrap();

        let conversation_id = "send-message-conversation";
        let root_id = "send-message-root";
        service
            .conversation_store
            .apply_graph_mutation(GraphMutation {
                scope: GraphScope {
                    namespace: "alpha".to_owned(),
                    workspace_id: Some("workspace-a".to_owned()),
                    graph_space: Some("conversation".to_owned()),
                },
                table: "gke_nodes".to_owned(),
                entity_kind: "node".to_owned(),
                event_id: format!("{root_id}-add"),
                op: "ADD".to_owned(),
                payload: json!({"id": root_id}),
                record: GraphRecord {
                    id: root_id.to_owned(),
                    document: Some(
                        json!({
                            "id": root_id,
                            "user_id": "conversation-owner",
                            "conversation_id": conversation_id,
                            "summary": "Start of conversation",
                        })
                        .to_string(),
                    ),
                    metadata: serde_json::Map::from_iter([
                        ("entity_type".to_owned(), json!("conversation_start")),
                        ("conversation_id".to_owned(), json!(conversation_id)),
                        ("user_id".to_owned(), json!("conversation-owner")),
                        ("turn_index".to_owned(), json!(-1)),
                        ("in_conversation_chain".to_owned(), json!(true)),
                        ("mentions".to_owned(), json!([])),
                    ]),
                    embedding: None,
                },
                embedding_dim: 3,
            })
            .await
            .unwrap();
        let acl_id = "send-message-root-acl";
        service
            .conversation_store
            .apply_graph_mutation(GraphMutation {
                scope: GraphScope {
                    namespace: "alpha".to_owned(),
                    workspace_id: Some("workspace-a".to_owned()),
                    graph_space: Some("conversation".to_owned()),
                },
                table: "gke_nodes".to_owned(),
                entity_kind: "node".to_owned(),
                event_id: format!("{acl_id}-add"),
                op: "ADD".to_owned(),
                payload: json!({"id": acl_id}),
                record: GraphRecord {
                    id: acl_id.to_owned(),
                    document: Some(json!({"id": acl_id}).to_string()),
                    metadata: serde_json::Map::from_iter([
                        ("entity_type".to_owned(), json!("acl_record")),
                        ("acl_truth_graph".to_owned(), json!("conversation")),
                        ("acl_target_grain".to_owned(), json!("node")),
                        ("acl_target_entity_id".to_owned(), json!(root_id)),
                        ("acl_mode".to_owned(), json!("shared")),
                        ("acl_version".to_owned(), json!(1)),
                        ("owner_id".to_owned(), json!("conversation-owner")),
                        (
                            "shared_with_principals".to_owned(),
                            json!(["request-actor@example.test"]),
                        ),
                    ]),
                    embedding: None,
                },
                embedding_dim: 3,
            })
            .await
            .unwrap();

        let (fault_client, connection) =
            tokio_postgres::connect(&dsn, tokio_postgres::NoTls).await.unwrap();
        let connection_task = tokio::spawn(connection);
        fault_client
            .batch_execute(&format!(
                "CREATE FUNCTION \"{schema}_conversation\".reject_send_message_index_job() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'injected send_message index failure'; END; $$; CREATE TRIGGER reject_send_message_index_job BEFORE INSERT ON \"{schema}_conversation\".index_jobs FOR EACH ROW EXECUTE FUNCTION \"{schema}_conversation\".reject_send_message_index_job();"
            ))
            .await
            .unwrap();
        drop(fault_client);
        connection_task.await.unwrap().unwrap();

        let make_request = |user_id: Option<&str>| {
            let mut args = json!({
                "conversation_id": conversation_id,
                "text": "A user question",
                "workflow_id": "send-message-admission-test"
            });
            if let Some(user_id) = user_id {
                args["user_id"] = json!(user_id);
            }
            ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/syscall/v1/send_message".to_owned(),
                body: serde_json::to_vec(&json!({
                "version": "v1",
                "op": "send_message",
                "args": args
                }))
                .unwrap(),
                principal: json!({
                    "user_id": "request-actor",
                    "sub": "request-actor@example.test",
                    "role": "rw",
                    "ns": "conversation",
                    "storage_ns": "alpha",
                    "workspace": "workspace-a",
                    "graph_space": "conversation",
                    "security_scope": "team-alpha",
                    "capabilities": ["send_message"]
                }),
            }
        };
        let actor_defaulted = service.execute(make_request(None)).await;
        assert_eq!(
            actor_defaulted.status,
            StatusCode::FORBIDDEN.as_u16(),
            "an omitted user_id defaults to the authenticated actor, matching Python"
        );
        let rejected = service.execute(make_request(Some("conversation-owner"))).await;
        assert!(rejected.status >= 500, "{}", String::from_utf8_lossy(&rejected.body));
        let still_only_root = service
            .conversation_store
            .graph_projection_records(GraphProjectionRead {
                scope: GraphScope {
                    namespace: "alpha".to_owned(),
                    workspace_id: Some("workspace-a".to_owned()),
                    graph_space: Some("conversation".to_owned()),
                },
                table: "gke_nodes".to_owned(),
                ids: None,
                metadata: MetadataFilter {
                    equals: BTreeMap::from([(
                        "conversation_id".to_owned(),
                        json!(conversation_id),
                    )]),
                },
                limit: 20,
            })
            .await
            .unwrap();
        assert_eq!(still_only_root.len(), 1);
        assert_eq!(still_only_root[0].id, root_id);
        assert_eq!(
            service
                .conversation_store
                .list_index_jobs(Some("alpha"), None, None, None, None, 20)
                .await
                .unwrap()
                .len(),
            0
        );
        let (verify_client, connection) =
            tokio_postgres::connect(&dsn, tokio_postgres::NoTls).await.unwrap();
        let connection_task = tokio::spawn(connection);
        for table in ["server_runs", "server_run_events", "projected_lane_messages"] {
            let count: i64 = verify_client
                .query_one(&format!("SELECT COUNT(*) FROM \"{schema}\".{table}"), &[])
                .await
                .unwrap()
                .get(0);
            assert_eq!(count, 0, "failed admission left rows in {table}");
        }
        drop(verify_client);
        connection_task.await.unwrap().unwrap();

        let (cleanup_client, connection) =
            tokio_postgres::connect(&dsn, tokio_postgres::NoTls).await.unwrap();
        let connection_task = tokio::spawn(connection);
        cleanup_client
            .batch_execute(&format!(
                "DROP TRIGGER reject_send_message_index_job ON \"{schema}_conversation\".index_jobs; DROP FUNCTION \"{schema}_conversation\".reject_send_message_index_job();"
            ))
            .await
            .unwrap();
        drop(cleanup_client);
        connection_task.await.unwrap().unwrap();

        let http_secret = "send-message-mcp-test-secret";
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: vec!["rw".to_owned()],
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig {
                    mode: "jwt".to_owned(),
                    algorithm: Some("HS256".to_owned()),
                    key: Some(http_secret.to_owned()),
                    ..AuthConfig::default()
                },
            },
            Arc::new(service.clone()),
        );
        let now_seconds = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let make_token = |namespace: &str, capabilities: &[&str]| {
            jsonwebtoken::encode(
                &jsonwebtoken::Header::new(jsonwebtoken::Algorithm::HS256),
                &json!({
                "sub":"conversation-owner@example.test",
                "user_id":"conversation-owner",
                "actor_id":"conversation-owner",
                "role":"rw",
                "roles":["rw"],
                "ns":namespace,
                "storage_ns":"alpha",
                "workspace":"workspace-a",
                "graph_space":"conversation",
                "security_scope":"team-alpha",
                "capabilities":capabilities,
                "exp":now_seconds + 300
            }),
                &jsonwebtoken::EncodingKey::from_secret(http_secret.as_bytes()),
            )
            .unwrap()
        };
        let token = make_token("conversation", &["send_message"]);
        let missing_capability_token = make_token("conversation", &[]);
        let wrong_namespace_token = make_token("workflow", &["send_message"]);
        let mcp_ask = |bearer: &str, request_id: &str| {
            Request::builder()
                .method("POST")
                .uri("/mcp/conversation")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {bearer}"))
                .body(Body::from(
                    json!({
                        "jsonrpc":"2.0",
                        "id":request_id,
                        "method":"tools/call",
                        "params":{
                            "name":"conversation.ask",
                            "arguments":{
                                "conversation_id":conversation_id,
                                "text":"A user question",
                                "user_id":"conversation-owner",
                                "workflow_id":"send-message-admission-test"
                            }
                        }
                    })
                    .to_string(),
                ))
                .unwrap()
        };
        for (bearer, request_id) in [
            (&missing_capability_token, "ask-no-capability"),
            (&wrong_namespace_token, "ask-wrong-namespace"),
        ] {
            let denied = app.clone().oneshot(mcp_ask(bearer, request_id)).await.unwrap();
            assert_eq!(denied.status(), StatusCode::OK);
            let denied_body = to_bytes(denied.into_body(), 8192).await.unwrap();
            let denied_value: Value = serde_json::from_slice(&denied_body).unwrap();
            assert_eq!(denied_value["error"]["code"], -32000);
            assert!(denied_value.get("result").is_none());
        }
        let still_only_root = service
            .conversation_store
            .graph_projection_records(GraphProjectionRead {
                scope: GraphScope {
                    namespace: "alpha".to_owned(),
                    workspace_id: Some("workspace-a".to_owned()),
                    graph_space: Some("conversation".to_owned()),
                },
                table: "gke_nodes".to_owned(),
                ids: None,
                metadata: MetadataFilter {
                    equals: BTreeMap::from([("conversation_id".to_owned(), json!(conversation_id))]),
                },
                limit: 20,
            })
            .await
            .unwrap();
        assert_eq!(still_only_root.len(), 1);
        assert_eq!(still_only_root[0].id, root_id);
        let (verify_client, connection) =
            tokio_postgres::connect(&dsn, tokio_postgres::NoTls).await.unwrap();
        let connection_task = tokio::spawn(connection);
        for table in ["server_runs", "server_run_events", "projected_lane_messages"] {
            let count: i64 = verify_client
                .query_one(&format!("SELECT COUNT(*) FROM \"{schema}\".{table}"), &[])
                .await
                .unwrap()
                .get(0);
            assert_eq!(count, 0, "denied MCP ask left rows in {table}");
        }
        drop(verify_client);
        connection_task.await.unwrap().unwrap();

        let accepted = app
            .clone()
            .oneshot(
                mcp_ask(&token, "send-message-ask"),
            )
            .await
            .unwrap();
        assert_eq!(accepted.status(), StatusCode::OK);
        let accepted_body = to_bytes(accepted.into_body(), 65_536).await.unwrap();
        let response: Value = serde_json::from_slice(&accepted_body).unwrap();
        let run_id = response["result"]["run_id"].as_str().unwrap();
        let turn_id = response["result"]["turn_node_id"].as_str().unwrap();
        let nodes = service
            .conversation_store
            .graph_projection_records(GraphProjectionRead {
                scope: GraphScope {
                    namespace: "alpha".to_owned(),
                    workspace_id: Some("workspace-a".to_owned()),
                    graph_space: Some("conversation".to_owned()),
                },
                table: "gke_nodes".to_owned(),
                ids: None,
                metadata: MetadataFilter {
                    equals: BTreeMap::from([(
                        "conversation_id".to_owned(),
                        json!(conversation_id),
                    )]),
                },
                limit: 20,
            })
            .await
            .unwrap();
        assert_eq!(nodes.len(), 2);
        assert!(nodes.iter().any(|node| node.id == turn_id));
        assert_eq!(
            service
                .conversation_store
                .graph_projection_records(GraphProjectionRead {
                    scope: GraphScope {
                        namespace: "alpha".to_owned(),
                        workspace_id: Some("workspace-a".to_owned()),
                        graph_space: Some("conversation".to_owned()),
                    },
                    table: "gke_edges".to_owned(),
                    ids: None,
                    metadata: MetadataFilter::default(),
                    limit: 20,
                })
                .await
                .unwrap()
                .len(),
            1
        );
        let run = service.store.get_server_run(run_id).await.unwrap().unwrap();
        assert_eq!(run.user_id.as_deref(), Some("conversation-owner"));
        let events = service.store.list_server_run_events(run_id, 0, 10).await.unwrap();
        assert_eq!(events.len(), 1);
        let created: Value = serde_json::from_str(&events[0].payload_json).unwrap();
        assert_eq!(created["admission_actor"]["user_id"], "conversation-owner");
        let lanes = service
            .store
            .list_projected_lane_messages(LaneMessageFilter {
                correlation_id: Some(run_id.to_owned()),
                limit: 10,
                ..LaneMessageFilter::default()
            })
            .await
            .unwrap();
        assert_eq!(lanes.len(), 1);
        assert_eq!(lanes[0].status, "pending");
        let queued_payload: Value =
            serde_json::from_str(lanes[0].payload_json.as_deref().unwrap()).unwrap();
        assert_eq!(queued_payload["initial_state"]["turn_index"], 0);
        assert_eq!(
            queued_payload["initial_state"]["_rt_graph_scope"]["storage_namespace"],
            "alpha"
        );
        assert_eq!(
            queued_payload["initial_state"]["_rt_graph_scope"]["workspace"],
            "workspace-a"
        );
        assert_eq!(
            queued_payload["initial_state"]["_rt_graph_scope"]["graph_space"],
            "conversation"
        );
        assert_eq!(
            queued_payload["initial_state"]["_rt_graph_scope"]["security_scope"],
            "team-alpha"
        );
        assert_eq!(
            service
                .conversation_store
                .list_index_jobs(Some("alpha"), None, None, None, None, 20)
                .await
                .unwrap()
                .iter()
                .map(|job| job.index_kind.as_str())
                .collect::<BTreeSet<_>>(),
            BTreeSet::from(["edge_endpoints", "edge_refs", "node_docs", "node_refs"])
        );

        drop(service);
        let (cleanup, connection) =
            tokio_postgres::connect(&dsn, tokio_postgres::NoTls).await.unwrap();
        let connection_task = tokio::spawn(connection);
        cleanup
            .batch_execute(&format!(
                "DROP SCHEMA IF EXISTS \"{schema}_conversation\" CASCADE; \
                 DROP SCHEMA IF EXISTS \"{schema}_workflow\" CASCADE; \
                 DROP SCHEMA IF EXISTS \"{schema}_wisdom\" CASCADE; \
                 DROP SCHEMA IF EXISTS \"{schema}\" CASCADE"
            ))
            .await
            .unwrap();
        drop(cleanup);
        connection_task.await.unwrap().unwrap();
        let _ = std::fs::remove_file(auth_path);
    }

    #[test]
    fn conversation_message_span_uses_python_character_offsets() {
        let span = conversation_span("conv-1", "turn-1", "A\u{1f642}B");
        assert_eq!(span["start_char"], 0);
        assert_eq!(span["end_char"], 3);
        assert_eq!(span["excerpt"], "A\u{1f642}B");
        assert_eq!(span["verification"]["notes"], "verbatim user input");
    }

    #[test]
    fn conversation_admission_matches_python_single_stage_indexes_and_distances() {
        assert_eq!(
            conversation_index_job_kinds("node"),
            Some(["node_docs", "node_refs"].as_slice())
        );
        assert_eq!(
            conversation_index_job_kinds("edge"),
            Some(["edge_refs", "edge_endpoints"].as_slice())
        );
        assert_eq!(conversation_index_job_kinds("unknown"), None);

        let previous = GraphRecord {
            id: "previous-turn".to_owned(),
            document: Some(r#"{"summary":"A🙂"}"#.to_owned()),
            metadata: serde_json::Map::from_iter([
                ("char_distance_from_last_summary".to_owned(), json!(4)),
                ("turn_distance_from_last_summary".to_owned(), json!(5)),
            ]),
            embedding: None,
        };
        assert_eq!(
            conversation_next_turn_distances(&previous, "B🙂", 6),
            (8, 7, 6)
        );
    }
