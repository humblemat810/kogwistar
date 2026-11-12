    #[tokio::test]
    async fn sqlite_submit_enforces_atomic_queue_backpressure() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use std::time::{SystemTime, UNIX_EPOCH};
        use tower::ServiceExt;

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-backpressure-{nonce}.sqlite3"));
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: vec!["reader".to_owned()],
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig::default(),
            },
            Arc::new(
                SqliteRunApplicationService::open(&path)
                    .unwrap()
                    .with_max_queue(1),
            ),
        );
        let submit = |priority_class: &str| {
            Request::builder()
                .method("POST")
                .uri("/api/workflow/runs")
                .header("content-type", "application/json")
                .header("x-kogwistar-roles", "reader,rw")
                .body(Body::from(
                    serde_json::to_vec(&json!({
                        "workflow_id": "backpressure-wf",
                        "conversation_id": "backpressure-conv",
                        "priority_class": priority_class,
                    }))
                    .unwrap(),
                ))
                .unwrap()
        };
        let first = app.clone().oneshot(submit("foreground")).await.unwrap();
        assert_eq!(first.status(), StatusCode::ACCEPTED);
        let claimed = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/claim")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader,rw")
                    .body(Body::from(
                        r#"{"claimed_by":"backpressure-worker","limit":1,"lease_seconds":60}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(claimed.status(), StatusCode::OK);
        let claimed =
            serde_json::from_slice::<Value>(&to_bytes(claimed.into_body(), 8192).await.unwrap())
                .unwrap();
        assert_eq!(claimed["work"][0]["payload"]["op"], "noop");
        for (priority_class, admission) in [("foreground", "deferred"), ("batch", "rejected")] {
            let response = app.clone().oneshot(submit(priority_class)).await.unwrap();
            assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
            let body = to_bytes(response.into_body(), 8192).await.unwrap();
            let value = serde_json::from_slice::<Value>(&body).unwrap();
            assert_eq!(value["admission"], admission);
            assert_eq!(value["reason"], "queue_full");
            assert_eq!(value["max_queue"], 1);
        }
        let store = SqliteStore::open(&path).unwrap();
        assert_eq!(
            store.list_server_runs(None, None, None, 10).unwrap().len(),
            1
        );
        assert_eq!(
            store
                .list_projected_lane_messages(LaneMessageFilter {
                    namespace: Some("workflow".to_owned()),
                    inbox_id: Some("workflow-runtime".to_owned()),
                    ..LaneMessageFilter::default()
                })
                .unwrap()
                .len(),
            1
        );
        drop(store);
        drop(app);
        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn sqlite_submit_rejects_answer_workflow_without_graph_authority() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use std::time::{SystemTime, UNIX_EPOCH};
        use tower::ServiceExt;

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-answer-not-ready-{nonce}.sqlite3"));
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: vec!["reader".to_owned()],
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig::default(),
            },
            Arc::new(SqliteRunApplicationService::open(&path).unwrap()),
        );
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/workflow/runs")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader,rw")
                    .body(Body::from(
                        r#"{"workflow_id":"agentic_answering.v2","conversation_id":"answer-conversation"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::CONFLICT);
        let body = to_bytes(response.into_body(), 8192).await.unwrap();
        let value = serde_json::from_slice::<Value>(&body).unwrap();
        assert_eq!(value["code"], "KOGWISTAR_ANSWER_WORKFLOW_NOT_READY");

        let store = SqliteStore::open(&path).unwrap();
        assert!(store.list_server_runs(None, None, None, 10).unwrap().is_empty());
        assert!(store
            .list_projected_lane_messages(LaneMessageFilter {
                namespace: Some("workflow".to_owned()),
                inbox_id: Some("workflow-runtime".to_owned()),
                ..LaneMessageFilter::default()
            })
            .unwrap()
            .is_empty());
        drop(store);
        drop(app);
        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn sqlite_chroma_admits_valid_python_answer_workflow_plan() {
        use kogwistar_store::{NamedProjectionWrite, WorkflowDesignSnapshotWrite};
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-answer-chroma-{nonce}.sqlite3"));
        let workflow_id = "debug.rag.v1";
        let contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/answer-workflow-runtime-contract-v1.json"
        ))
        .unwrap();
        let workflow = &contract["workflows"][workflow_id];
        let python_version = workflow["python_op_version"].as_str().unwrap();
        let nodes = workflow["required_ops"]
            .as_array()
            .unwrap()
            .iter()
            .map(|operation| {
                let operation = operation.as_str().unwrap();
                json!({
                    "id": format!("node-{operation}"),
                    "label": operation,
                    "metadata": {
                        "entity_type": "workflow_node",
                        "workflow_id": workflow_id,
                        "wf_op": operation,
                        "wf_version": python_version,
                        "wf_start": operation == "start",
                        "wf_executor_target": "python-worker"
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
                    "id": format!("edge-{index}"),
                    "source_ids": [format!("node-{}", route["from"].as_str().unwrap())],
                    "target_ids": [format!("node-{}", route["to"].as_str().unwrap())],
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
        let store = SqliteStore::open(&path).unwrap();
        store
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
                    materialization_status: "rust_event_only".to_owned(),
                },
            )
            .unwrap();
        store
            .put_workflow_design_snapshot(
                workflow_id,
                WorkflowDesignSnapshotWrite {
                    version: 1,
                    seq: 1,
                    schema_version: 1,
                    payload_json: serde_json::to_string(&json!({"nodes": nodes, "edges": edges}))
                        .unwrap(),
                },
            )
            .unwrap();
        drop(store);

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let chroma_setup = tokio::spawn(async move {
            for _ in 0..2 {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut bytes = Vec::new();
                let mut chunk = [0_u8; 2048];
                let (header_end, content_length) = loop {
                    let count = stream.read(&mut chunk).await.unwrap();
                    assert!(count > 0, "Chroma connection closed before request body");
                    bytes.extend_from_slice(&chunk[..count]);
                    if let Some(header_end) = bytes
                        .windows(4)
                        .position(|window| window == b"\r\n\r\n")
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
            chroma_config("answer-nodes"),
            chroma_config("answer-edges"),
        )
        .await
        .unwrap();
        chroma_setup.await.unwrap();

        let service = SqliteRunApplicationService::open(&path)
            .unwrap()
            .with_chroma_graph(graph);
        let response = service.submit_run(&ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/workflow/runs".to_owned(),
            body: serde_json::to_vec(&json!({
                "workflow_id": workflow_id,
                "conversation_id": "answer-conversation",
                "run_id": "answer-run-python-worker",
                "initial_state": {
                    "_rt_graph_scope": {"storage_namespace":"attacker-scope"},
                    "_rt_run_owner_id": "attacker"
                }
            }))
            .unwrap(),
            principal: json!({
                "sub": "alice", "user_id": "alice", "role": "rw",
                "ns": ["conversation", "workflow"],
                "storage_ns": "conversation",
                "capabilities": ["read_graph", "write_graph"]
            }),
        });
        assert_eq!(response.status, StatusCode::ACCEPTED.as_u16());
        let claimed = service
            .claim_runtime_work_items(
                RuntimeClaimRequest {
                    claimed_by: "python-worker".to_owned(),
                    limit: 1,
                    lease_seconds: 60,
                    run_id: Some("answer-run-python-worker".to_owned()),
                    recipient_ids: vec!["python-worker".to_owned()],
                },
                vec!["python-worker".to_owned()],
            )
            .unwrap();
        assert_eq!(claimed.len(), 1);
        assert_eq!(claimed[0].step_id.as_deref(), Some("node-start"));
        assert_eq!(claimed[0].payload["op"], "start");
        assert_eq!(claimed[0].payload["executor_target"], "python-worker");
        assert_eq!(
            claimed[0].payload["initial_state"]["_rt_graph_scope"]["storage_namespace"],
            "conversation",
            "SQLite admission must replace caller-supplied runtime scope with authenticated scope"
        );
        assert_eq!(
            claimed[0].payload["initial_state"]["_rt_run_owner_id"],
            "alice",
            "SQLite admission must replace caller-supplied runtime owner"
        );

        drop(service);
        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn sqlite_chroma_native_answer_snapshots_before_provider_and_persists_stream_events() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let nonce = uuid::Uuid::new_v4().simple().to_string();
        let db_path = std::env::temp_dir().join(format!("kogwistar-sqlite-native-answer-{nonce}.db"));
        let auth_path = std::env::temp_dir().join(format!("kogwistar-sqlite-native-answer-auth-{nonce}.db"));
        let auth = SqliteAuthStore::open(&auth_path).unwrap();
        auth.resolve_external_identity(ResolveExternalIdentity {
            issuer: "test".to_owned(),
            subject: "alice".to_owned(),
            email: "alice@example.test".to_owned(),
            display_name: None,
            new_user_id: "alice".to_owned(),
            default_role: "rw".to_owned(),
            default_ns: "conversation".to_owned(),
        })
        .await
        .unwrap();
        drop(auth);
        let auth_config = AuthConfig {
            auth_db_url: Some(format!("sqlite:///{}", auth_path.display())),
            ..AuthConfig::default()
        };

        let chroma_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let chroma_endpoint = format!("http://{}", chroma_listener.local_addr().unwrap());
        let chroma_server = tokio::spawn(async move {
            for _ in 0..2 {
                let (mut stream, _) = chroma_listener.accept().await.unwrap();
                let mut bytes = Vec::new();
                let (header_end, content_length) = loop {
                    let mut chunk = [0_u8; 1024];
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
                let request: Value = serde_json::from_slice(&bytes[header_end + 4..header_end + 4 + content_length]).unwrap();
                let body = json!({
                    "id": format!("collection-{}", request["name"].as_str().unwrap()),
                    "metadata": {"hnsw:space":"cosine"}
                }).to_string();
                let response = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                    body.len(), body
                );
                stream.write_all(response.as_bytes()).await.unwrap();
            }
        });
        let chroma_config = |collection: &str| kogwistar_store_chroma::ChromaStoreConfig {
            endpoint: chroma_endpoint.clone(),
            collection: collection.to_owned(),
            ..Default::default()
        };
        let graph = kogwistar_store_chroma::ChromaGraphStore::connect_entity_collections(
            chroma_config("native-answer-nodes"),
            chroma_config("native-answer-edges"),
        ).await.unwrap();
        chroma_server.await.unwrap();

        let chat_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let chat_endpoint = format!("http://{}/v1/chat/completions", chat_listener.local_addr().unwrap());
        let db_path_for_provider = db_path.clone();
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
            let request: Value = serde_json::from_slice(&bytes[header_end + 4..header_end + 4 + content_length]).unwrap();
            assert_eq!(request["stream"], true);
            let sqlite = SqliteStore::open(&db_path_for_provider).unwrap();
            let snapshot_id = kogwistar_contracts::stable_id(
                "conversation.context_snapshot",
                &[
                    "native-answer-conversation".to_owned(),
                    "native-answer-run".to_owned(),
                    "generate_answer_with_citations".to_owned(),
                    "3".to_owned(),
                    "0".to_owned(),
                ],
            )
            .to_string();
            let snapshot_events = sqlite
                .latest_entity_events("conversation", "node", &[snapshot_id.clone()])
                .unwrap();
            assert!(snapshot_events
                .get(&snapshot_id)
                .is_some_and(|event| event.payload_json.contains("context_snapshot")),
                "context snapshot must commit before the provider request");

            let answer = json!({"text":"answer","reasoning":"brief rationale","claims":[]}).to_string();
            let chunks = [
                json!({"id":"chatcmpl-test","choices":[{"delta":{"content":answer},"finish_reason":null}]}).to_string(),
                json!({"id":"chatcmpl-test","choices":[{"delta":{},"finish_reason":"stop"}]}).to_string(),
            ];
            let mut body = String::new();
            for chunk in chunks { body.push_str(&format!("data: {chunk}\r\n\r\n")); }
            body.push_str("data: [DONE]\r\n\r\n");
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                body.len(), body
            );
            for chunk in response.as_bytes().chunks(17) {
                stream.write_all(chunk).await.unwrap();
            }
        });

        let principal = json!({
            "sub":"alice", "user_id":"alice", "role":"rw",
            "ns":["conversation"], "storage_ns":"conversation",
            "capabilities":["read_graph", "write_graph"]
        });
        let service = SqliteRunApplicationService::open(&db_path)
            .unwrap()
            .with_chroma_graph(graph)
            .with_chat_completion_provider(
                crate::native_answer_generation::OpenAiCompatibleChatConfig::new(
                    chat_endpoint, "fixture-answer-model", None,
                ).unwrap().with_streaming(true),
            )
            .with_owner_auth_config(&auth_config).await.unwrap();
        service.store.create_server_run(ServerRunCreate {
            run_id: "native-answer-run".to_owned(),
            conversation_id: "native-answer-conversation".to_owned(),
            workflow_id: "debug.rag.v1".to_owned(),
            user_id: Some("alice".to_owned()),
            user_turn_node_id: "user-turn".to_owned(),
            status: "running".to_owned(),
        }).unwrap();
        let binding = runtime_owner_binding_v1(&principal, &service.capabilities).unwrap();
        service.store.append_server_run_event(
            "native-answer-run",
            "run.created",
            serde_json::to_string(&json!({"runtime_owner_binding":binding})).unwrap(),
        ).unwrap();
        let selection_request = kogwistar_runtime::RuntimeNativeStepRequest {
            executor_target: "rust-native".to_owned(),
            operation: "aa_select_used_evidence".to_owned(),
            op_version: "v1".to_owned(),
            run_id: "native-answer-run".to_owned(),
            workflow_id: "debug.rag.v1".to_owned(),
            node_id: "select-node".to_owned(),
            conversation_id: "native-answer-conversation".to_owned(),
            turn_node_id: Some("user-turn".to_owned()),
            token_id: "selection-token".to_owned(),
            parent_token_id: None,
            step_seq: 2,
            state: serde_json::from_value(json!({
                "conversation_id":"native-answer-conversation",
                "run_id":"native-answer-run",
                "run_step_seq":2,
                "attempt_seq":1,
                "question":"question",
                "system_prompt":"Select relevant evidence.",
                "candidates":[{"id":"candidate-node","label":"Candidate","summary":"summary"}],
                "conversation_context_messages":[{"role":"user","content":"prior context"}],
                "conversation_context_node_ids":[],
                "_rt_graph_scope": {
                    "storage_namespace":"conversation",
                    "workspace":null,
                    "graph_space":"conversation"
                }
            })).unwrap(),
            runtime_routes: Vec::new(),
            ephemeral_context: serde_json::Map::new(),
        };
        service
            .persist_sqlite_native_selection_context_snapshot(&selection_request)
            .await
            .unwrap();
        let selection_snapshot_id = kogwistar_contracts::stable_id(
            "conversation.context_snapshot",
            &[
                "native-answer-conversation".to_owned(),
                "native-answer-run".to_owned(),
                "select_used_evidence".to_owned(),
                "2".to_owned(),
                "1".to_owned(),
            ],
        )
        .to_string();
        let selection_snapshot_events = service
            .store
            .latest_entity_events("conversation", "node", &[selection_snapshot_id.clone()])
            .unwrap();
        assert!(selection_snapshot_events
            .get(&selection_snapshot_id)
            .is_some_and(|event| event.payload_json.contains("select_used_evidence")));
        let request = kogwistar_runtime::RuntimeNativeStepRequest {
            executor_target: "rust-native".to_owned(),
            operation: "aa_generate_answer_with_citations".to_owned(),
            op_version: "v1".to_owned(),
            run_id: "native-answer-run".to_owned(),
            workflow_id: "debug.rag.v1".to_owned(),
            node_id: "answer-node".to_owned(),
            conversation_id: "native-answer-conversation".to_owned(),
            turn_node_id: Some("user-turn".to_owned()),
            token_id: "token-1".to_owned(),
            parent_token_id: None,
            step_seq: 3,
            state: serde_json::from_value(json!({
                "conversation_id":"native-answer-conversation",
                "question":"question",
                "system_prompt":"Use only evidence.",
                "used_node_ids":[],
                "turn_node_id":"user-turn",
                "_rt_graph_scope": {
                    "storage_namespace":"conversation",
                    "workspace":null,
                    "graph_space":"conversation"
                },
                "_rt":{"evidence_pack":{"nodes":[],"edges":[]}}
            })).unwrap(),
            runtime_routes: Vec::new(),
            ephemeral_context: serde_json::Map::new(),
        };
        let snapshot_id = kogwistar_contracts::stable_id(
            "conversation.context_snapshot",
            &[
                "native-answer-conversation".to_owned(),
                "native-answer-run".to_owned(),
                "aa_generate_answer_with_citations".to_owned(),
                "3".to_owned(),
                "0".to_owned(),
            ],
        )
        .to_string();
        let mut unsupported_version = request.clone();
        unsupported_version.op_version = "v2".to_owned();
        let rejected = service
            .execute_sqlite_native_answer_generation(
                RecordedWorkerHandoff {
                    message_id: "lane-invalid-native-answer".to_owned(),
                    claimed_by: "rust-native-worker".to_owned(),
                    run_id: "native-answer-run".to_owned(),
                    step_id: "answer-node".to_owned(),
                    correlation_id: "token-invalid".to_owned(),
                },
                unsupported_version,
            )
            .await;
        assert!(rejected.is_err());
        assert!(service
            .store
            .latest_entity_events("conversation", "node", &[snapshot_id.clone()])
            .unwrap()
            .is_empty(), "unsupported operation version must fail before snapshot persistence");
        let result = service.execute_sqlite_native_answer_generation(
            RecordedWorkerHandoff {
                message_id:"lane-native-answer".to_owned(),
                claimed_by:"rust-native-worker".to_owned(),
                run_id:"native-answer-run".to_owned(),
                step_id:"answer-node".to_owned(),
                correlation_id:"token-1".to_owned(),
            },
            request,
        ).await.unwrap();
        chat_server.await.unwrap();

        assert_eq!(result.effect.as_ref().unwrap().status, kogwistar_runtime::RuntimeWorkerEffectStatus::Success);
        let first_update = serde_json::to_value(&result.effect.as_ref().unwrap().state_update[0])
            .unwrap();
        assert_eq!(first_update[1]["answer"]["text"], "answer");
        let events = service.store.list_server_run_events("native-answer-run", 0, 100).unwrap();
        assert!(events.iter().any(|event| event.event_type == "workflow.provider_event.v1"));
        let lane_events = service.store.list_projected_lane_messages(LaneMessageFilter {
            namespace: Some("conversation".to_owned()),
            inbox_id: Some("agent-provider-events".to_owned()),
            ..LaneMessageFilter::default()
        }).unwrap();
        assert!(!lane_events.is_empty());
        assert!(!service
            .store
            .list_index_jobs(Some("conversation"), None, None, None, None, 100)
            .unwrap()
            .is_empty());

        drop(service);
        let _ = std::fs::remove_file(db_path);
        let _ = std::fs::remove_file(auth_path);
    }

    #[test]
    fn snapshot_syscall_compatibility_drops_rust_only_fields() {
        let expected = serde_json::from_str::<Value>(include_str!(
            "../../../../../contracts/golden/mount-memory-result-v1.json"
        ))
        .unwrap();
        let mut internal_payload = expected.clone();
        let internal_object = internal_payload.as_object_mut().unwrap();
        internal_object.insert("schema_version".to_owned(), json!(1));
        internal_object.insert("implementation_authority".to_owned(), json!("rust"));
        internal_object.insert("seam_attestation_version".to_owned(), json!(1));
        internal_object.insert("seam_crossings".to_owned(), json!([]));
        let mut response = json_effect(StatusCode::OK, internal_payload);
        strip_snapshot_implementation_fields(&mut response);
        let actual = serde_json::from_slice::<Value>(&response.body).unwrap();
        assert_eq!(actual, expected);

        let typed: ContextSnapshotV1 = serde_json::from_value({
            let mut value = expected.clone();
            value
                .as_object_mut()
                .unwrap()
                .insert("schema_version".to_owned(), json!(1));
            value
        })
        .unwrap();
        assert_eq!(public_context_snapshot_value(typed), expected);
        assert_eq!(
            context_snapshot_syscall_field(json!({}).as_object().unwrap(), "stage"),
            Ok(None)
        );
        assert_eq!(
            context_snapshot_syscall_field(json!({"stage": null}).as_object().unwrap(), "stage"),
            Ok(None)
        );
        assert!(
            context_snapshot_syscall_field(json!({"stage": 1}).as_object().unwrap(), "stage")
                .is_err()
        );

        let mut failure = json_effect(
            StatusCode::NOT_FOUND,
            json!({"error": {"code": "KOGWISTAR_CONTEXT_SNAPSHOT_NOT_FOUND"}}),
        );
        let original_body = failure.body.clone();
        strip_snapshot_implementation_fields(&mut failure);
        assert_eq!(failure.body, original_body);
    }

    #[test]
    fn receive_message_compatibility_matches_shared_wire_fixture() {
        let expected = serde_json::from_str::<Value>(include_str!(
            "../../../../../contracts/golden/receive-message-result-v1.json"
        ))
        .unwrap();
        let mut internal_payload = expected.clone();
        let internal_object = internal_payload.as_object_mut().unwrap();
        let mut turns = internal_object.remove("messages").unwrap();
        for turn in turns.as_array_mut().unwrap() {
            turn.as_object_mut()
                .unwrap()
                .insert("schema_version".to_owned(), json!(1));
        }
        internal_object.insert("turns".to_owned(), turns);
        internal_object.insert("schema_version".to_owned(), json!(1));
        internal_object.insert("implementation_authority".to_owned(), json!("rust"));
        internal_object.insert("seam_attestation_version".to_owned(), json!(1));
        internal_object.insert("seam_crossings".to_owned(), json!([]));
        let mut response = json_effect(StatusCode::OK, internal_payload);

        legacy_receive_message_payload(&mut response);

        assert_eq!(response.status, StatusCode::OK.as_u16());
        let actual = serde_json::from_slice::<Value>(&response.body).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn receive_message_invalid_internal_payload_fails_closed() {
        let mut response = json_effect(
            StatusCode::OK,
            json!({
                "schema_version": 1,
                "conversation_id": "conversation-1",
                "turns": [],
                "unexpected": true
            }),
        );

        legacy_receive_message_payload(&mut response);

        assert_eq!(response.status, StatusCode::INTERNAL_SERVER_ERROR.as_u16());
        let error = serde_json::from_slice::<Value>(&response.body).unwrap();
        assert_eq!(error["code"], "KOGWISTAR_INVALID_RESPONSE");
    }

    #[tokio::test]
    async fn sqlite_syscall_spawn_and_terminate_use_authoritative_run_service() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-syscall-{nonce}.sqlite3"));
        let service = SqliteRunApplicationService::open(&path).unwrap();
        let view_request = |capabilities: &[&str]| ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/syscall/v1/project_view".to_owned(),
            body: serde_json::to_vec(&json!({
                "version":"v1",
                "op":"project_view",
                "args":{}
            }))
            .unwrap(),
            principal: json!({
                "sub":"viewer",
                "role":"ro",
                "ns":"workflow",
                "capabilities":capabilities,
            }),
        };
        let denied_view = service.execute_sync(view_request(&["project_view"]));
        assert_eq!(denied_view.status, StatusCode::FORBIDDEN.as_u16());
        let view = service.execute_sync(view_request(&["project_view", "read_security_scope"]));
        assert_eq!(view.status, StatusCode::OK.as_u16());
        let view = serde_json::from_slice::<Value>(&view.body).unwrap();
        assert_eq!(view["version"], "v1");
        assert_eq!(view["op"], "project_view");
        assert_eq!(view["status"], "ok");
        assert!(view["result"]["scheduler"].is_object());
        assert!(view["result"]["runs"].is_object());
        let receive_request = |namespace: &str| ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/syscall/v1/receive_message".to_owned(),
            body: serde_json::to_vec(&json!({
                "version":"v1",
                "op":"receive_message",
                "args":{"conversation_id":"conversation-1"}
            }))
            .unwrap(),
            principal: json!({
                "sub":"viewer",
                "role":"ro",
                "ns":namespace,
                "capabilities":["read_graph"],
            }),
        };
        assert_eq!(
            service.execute_sync(receive_request("workflow")).status,
            StatusCode::FORBIDDEN.as_u16()
        );
        assert_eq!(
            service.execute_sync(receive_request("conversation")).status,
            StatusCode::NOT_IMPLEMENTED.as_u16()
        );
        let mount_request = |namespace: &str| ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/syscall/v1/mount_memory".to_owned(),
            body: serde_json::to_vec(&json!({
                "version":"v1",
                "op":"mount_memory",
                "args":{"conversation_id":"conversation-1"}
            }))
            .unwrap(),
            principal: json!({
                "sub":"viewer",
                "role":"ro",
                "ns":namespace,
                "capabilities":["project_view"],
            }),
        };
        assert_eq!(
            service.execute_sync(mount_request("workflow")).status,
            StatusCode::FORBIDDEN.as_u16()
        );
        assert_eq!(
            service.execute_sync(mount_request("conversation")).status,
            StatusCode::NOT_IMPLEMENTED.as_u16()
        );
        let principal = json!({
            "sub":"operator",
            "role":"rw",
            "ns":"workflow",
            "capabilities":["spawn_process","workflow.run.write","workflow.run.read"],
        });
        let spawned = service.execute_sync(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/syscall/v1/spawn_process".to_owned(),
            body: serde_json::to_vec(&json!({
                "version":"v1",
                "op":"spawn_process",
                "args":{
                    "workflow_id":"wf-syscall",
                    "conversation_id":"conv-syscall",
                    "initial_state":{
                        "source":"syscall",
                        "_rt_requested_capabilities":[
                            "workflow.run.write"
                        ],
                        "_rt_effective_capabilities":["service.manage"]
                    },
                },
            }))
            .unwrap(),
            principal: principal.clone(),
        });
        assert_eq!(spawned.status, StatusCode::OK.as_u16());
        let spawned = serde_json::from_slice::<Value>(&spawned.body).unwrap();
        assert_eq!(spawned["version"], "v1");
        assert_eq!(spawned["op"], "spawn_process");
        assert_eq!(spawned["status"], "ok");
        let run_id = spawned["result"]["run_id"].as_str().unwrap();
        assert!(service.store.get_server_run(run_id).unwrap().is_some());
        let lanes = service
            .store
            .list_projected_lane_messages(LaneMessageFilter {
                namespace: Some("workflow".to_owned()),
                inbox_id: Some("workflow-runtime".to_owned()),
                correlation_id: Some(run_id.to_owned()),
                ..LaneMessageFilter::default()
            })
            .unwrap();
        let payload =
            serde_json::from_str::<Value>(lanes[0].payload_json.as_deref().unwrap()).unwrap();
        assert_eq!(
            payload["initial_state"]["_rt_effective_capabilities"],
            json!(["workflow.run.write"])
        );
        assert_eq!(payload["initial_state"]["_rt_delegated_by"], "operator");
        assert_eq!(
            payload["initial_state"]["_rt_delegation"],
            json!({
                "schema_version": 1,
                "delegated_by": "operator",
                "effective_capabilities": ["workflow.run.write"]
            })
        );
        let terminated = service.execute_sync(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/syscall/v1/terminate_process".to_owned(),
            body: serde_json::to_vec(&json!({
                "version":"v1",
                "op":"terminate_process",
                "args":{"run_id":run_id},
            }))
            .unwrap(),
            principal,
        });
        assert_eq!(terminated.status, StatusCode::OK.as_u16());
        let terminated = serde_json::from_slice::<Value>(&terminated.body).unwrap();
        assert_eq!(terminated["result"]["cancel_requested"], true);
        assert!(
            service
                .store
                .get_server_run(run_id)
                .unwrap()
                .unwrap()
                .cancel_requested
        );
        drop(service);
        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn sqlite_worker_suspend_resume_is_durable_and_scheduler_owned() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use std::time::{SystemTime, UNIX_EPOCH};
        use tower::ServiceExt;

        let resume_contract_fixture: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/run-resume-contract-v1.json"
        ))
        .unwrap();
        let resume_result_fixture: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/workflow-run-resume-result-v1.json"
        ))
        .unwrap();
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-resume-{nonce}.sqlite3"));
        let reopen = || {
            router_with_application(
                ApiState {
                    health: snapshot(),
                    required_roles: vec!["reader".to_owned()],
                    implementation: ImplementationSnapshot::default(),
                    auth: AuthConfig::default(),
                },
                Arc::new(SqliteRunApplicationService::open(&path).unwrap()),
            )
        };
        let mut app = reopen();
        let call = |app: Router, method: &'static str, uri: String, body: Value| async move {
            let response = app
                .oneshot(
                    Request::builder()
                        .method(method)
                        .uri(uri)
                        .header("content-type", "application/json")
                        .header("x-kogwistar-roles", "reader,rw")
                        .body(Body::from(if body.is_null() {
                            Vec::new()
                        } else {
                            serde_json::to_vec(&body).unwrap()
                        }))
                        .unwrap(),
                )
                .await
                .unwrap();
            let status = response.status();
            let bytes = to_bytes(response.into_body(), 32768).await.unwrap();
            (status, serde_json::from_slice::<Value>(&bytes).unwrap())
        };
        let (status, submitted) = call(
            app.clone(),
            "POST",
            "/api/workflow/runs".to_owned(),
            json!({"workflow_id":"resume-wf","conversation_id":"resume-conv","turn_node_id":"turn-original"}),
        )
        .await;
        assert_eq!(status, StatusCode::ACCEPTED);
        let run_id = submitted["run_id"].as_str().unwrap().to_owned();
        drop(app);
        app = reopen();
        let (_, claimed) = call(
            app.clone(),
            "POST",
            "/internal/runtime/claim".to_owned(),
            json!({"claimed_by":"suspender","limit":1,"lease_seconds":60}),
        )
        .await;
        let work = &claimed["work"][0];
        drop(app);
        app = reopen();
        let suspend = json!({
            "handoff": {
                "message_id": work["message_id"],
                "claimed_by": work["claimed_by"],
                "run_id": work["run_id"],
                "step_id": work["step_id"],
                "correlation_id": work["correlation_id"],
            },
            "effect": {
                "contract_version": 1,
                "effect_id": format!("suspend-{run_id}"),
                "status": "suspended",
                "state_update": [["u", {"before_suspend": true}]],
                "successors": [],
                "wait_reason": "approval",
                "resume_payload": {"question": "continue?"},
            }
        });
        let (status, _) = call(
            app.clone(),
            "POST",
            "/internal/runtime/results".to_owned(),
            suspend,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        drop(app);
        app = reopen();
        let (status, contract) = call(
            app.clone(),
            "GET",
            format!("/api/workflow/runs/{run_id}/resume-contract"),
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        for field in resume_contract_fixture["required_fields"]
            .as_array()
            .unwrap()
        {
            let field = field.as_str().unwrap();
            assert!(contract.get(field).is_some(), "missing {field}");
        }
        assert_eq!(contract["compatible"], true);
        assert_eq!(contract["checkpoint_schema_version"], 1);
        assert_eq!(
            contract["supported_wait_reasons"],
            resume_contract_fixture["supported_wait_reasons"]
        );
        assert_eq!(contract["resume_options"][0]["suspended_node_id"], "start");
        assert_eq!(contract["wait_reason"], "approval");
        assert_eq!(contract["suspended"][0][0], "start");
        assert_eq!(contract["runtime_routes"], json!([]));
        assert_eq!(contract["node_ops"], json!({}));
        let token_id = contract["suspended"][0][2].as_str().unwrap().to_owned();
        let resume_body = json!({
            "suspended_node_id": "start",
            "suspended_token_id": token_id,
            "client_result": {
                "status": "success",
                "state_update": [["u", {"approved": true}]],
                "successors": [],
                "route_next": [],
                "result": {"workflow_status": "succeeded"}
            },
            "workflow_id": "resume-wf",
            "conversation_id": "resume-conv",
            "turn_node_id": "unused-compatible-field",
        });
        let denied_resume = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!(
                        "/api/workflow/runs/{run_id}/resume?wait_for_completion=true"
                    ))
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(serde_json::to_vec(&resume_body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(denied_resume.status(), StatusCode::FORBIDDEN);
        let resume_app = app.clone();
        let resume_uri = format!(
            "/api/workflow/runs/{run_id}/resume"
        );
        let resume_future = tokio::spawn(async move {
            call(resume_app, "POST", resume_uri, resume_body).await
        });
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        let (_, claimed) = call(
            app.clone(),
            "POST",
            "/internal/runtime/claim".to_owned(),
            json!({"claimed_by":"finisher","limit":1,"lease_seconds":60}),
        )
        .await;
        let work = &claimed["work"][0];
        assert_eq!(work["step_id"], "start");
        assert_eq!(work["payload"]["turn_node_id"], "turn-original");
        assert_eq!(work["payload"]["resume_effect"]["status"], "success");
        assert_eq!(
            work["payload"]["resume_effect"]["state_update"][0][1]["approved"],
            true
        );
        let finish_result = json!({
            "handoff": {
                "message_id": work["message_id"],
                "claimed_by": work["claimed_by"],
                "run_id": work["run_id"],
                "step_id": work["step_id"],
                "correlation_id": work["correlation_id"],
            },
            "effect": {
                "contract_version": 1,
                "effect_id": format!("finish-{run_id}"),
                "state_update": [["u", {"after_resume": true}]],
                "successors": [],
                "result": {"done": true},
            }
        });
        let (status, first_finish) = call(
            app.clone(),
            "POST",
            "/internal/runtime/results".to_owned(),
            finish_result.clone(),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let (retry_status, retry_finish) = call(
            app.clone(),
            "POST",
            "/internal/runtime/results".to_owned(),
            finish_result,
        )
        .await;
        assert_eq!(retry_status, StatusCode::OK);
        assert_eq!(retry_finish["event_seq"], first_finish["event_seq"]);
        assert_eq!(retry_finish["idempotent"], true);
        let (resume_status, resumed) = resume_future.await.unwrap();
        assert_eq!(resume_status, StatusCode::OK);
        let result_fields = resume_result_fixture["required_fields"]
            .as_array()
            .unwrap()
            .iter()
            .map(|field| field.as_str().unwrap())
            .collect::<std::collections::BTreeSet<_>>();
        let actual_fields = resumed
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(actual_fields, result_fields);
        assert!(resume_result_fixture["completed_statuses"]
            .as_array()
            .unwrap()
            .contains(&resumed["workflow_status"]));
        assert_eq!(resumed["workflow_status"], "succeeded");
        assert_eq!(resumed["final_state"]["before_suspend"], true);
        assert_eq!(resumed["final_state"]["after_resume"], true);
        assert_eq!(resumed["budget"], json!({}));
        drop(app);
        app = reopen();
        let (status, run) = call(
            app,
            "GET",
            format!("/api/workflow/runs/{run_id}"),
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(run["status"], "succeeded");
        assert_eq!(run["result"]["done"], true);
        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn sqlite_submit_freezes_exact_rust_event_graph_snapshot_as_authoritative_routes() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use kogwistar_store::{NamedProjectionWrite, WorkflowDesignSnapshotWrite};
        use std::time::{SystemTime, UNIX_EPOCH};
        use tower::ServiceExt;

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-graph-route-{nonce}.sqlite3"));
        let store = SqliteStore::open(&path).unwrap();
        store
            .replace_named_projection(
                "workflow_design",
                "graph-wf",
                NamedProjectionWrite {
                    payload: serde_json::Map::from_iter([
                        ("current_version".to_owned(), json!(1)),
                        ("snapshot_schema_version".to_owned(), json!(1)),
                    ]),
                    last_authoritative_seq: 1,
                    last_materialized_seq: 1,
                    projection_schema_version: 1,
                    materialization_status: "rust_event_only".to_owned(),
                },
            )
            .unwrap();
        store
            .put_workflow_design_snapshot(
                "graph-wf",
                WorkflowDesignSnapshotWrite {
                    version: 1,
                    seq: 1,
                    schema_version: 1,
                    payload_json: serde_json::to_string(&json!({
                        "nodes": [
                            {"id":"graph-start","metadata":{"wf_start":true,"wf_op":"noop","wf_executor_target":"rust-native"}},
                            {"id":"graph-end","metadata":{"wf_terminal":true}}
                        ],
                        "edges": [{
                            "id":"edge-1",
                            "source_ids":["graph-start"],
                            "target_ids":["graph-end"],
                            "metadata":{}
                        }]
                    }))
                    .unwrap(),
                },
            )
            .unwrap();
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: vec!["reader".to_owned()],
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig::default(),
            },
            Arc::new(SqliteRunApplicationService::open(&path).unwrap()),
        );
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/workflow/runs")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader,rw")
                    .body(Body::from(
                        r#"{"workflow_id":"graph-wf","conversation_id":"graph-conv","initial_state":{"_rt_node_executor_targets":{"graph-end":"rust-native"}},"runtime_routes":[{"source_node_id":"start","target_node_id":"forged","join_mask":0}]}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::ACCEPTED);
        let generic_claim = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/claim")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader,rw")
                    .body(Body::from(
                        r#"{"claimed_by":"graph-worker","limit":1,"lease_seconds":60}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let generic_claim_body = to_bytes(generic_claim.into_body(), 16384).await.unwrap();
        let generic_claim_work =
            serde_json::from_slice::<Value>(&generic_claim_body).unwrap()["work"].clone();
        assert_eq!(generic_claim_work, json!([]));

        let claim = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/claim")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        serde_json::to_vec(&json!({
                            "claimed_by": "rust-native-worker",
                            "limit": 1,
                            "lease_seconds": 60,
                            "recipient_ids": [
                                "rust-native-worker",
                                "python-worker",
                            ],
                        }))
                        .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let claim_body = to_bytes(claim.into_body(), 16384).await.unwrap();
        let work = serde_json::from_slice::<Value>(&claim_body).unwrap()["work"][0].clone();
        assert_eq!(work["step_id"], "graph-start");
        assert_eq!(work["payload"]["op"], "noop");
        assert_eq!(work["payload"]["executor_target"], "rust-native");
        let run_id = work["run_id"].as_str().unwrap();
        let events_before_misrouted_result = store
            .list_server_run_events(run_id, 0, 100)
            .unwrap();
        let result = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/results")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        serde_json::to_vec(&json!({
                            "handoff": {
                                "message_id": work["message_id"],
                                "claimed_by": work["claimed_by"],
                                "run_id": work["run_id"],
                                "step_id": work["step_id"],
                                "correlation_id": work["correlation_id"],
                            },
                            "effect": {
                                "contract_version": 1,
                                "effect_id": format!("graph-start-{}", work["run_id"].as_str().unwrap()),
                                "successors": [{"node_id":"worker-forged","join_mask":0}],
                            }
                        }))
                        .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(result.status(), StatusCode::CONFLICT);
        assert_eq!(
            store.list_server_run_events(run_id, 0, 100).unwrap(),
            events_before_misrouted_result,
            "misrouted result must not append durable events"
        );
        let native_result = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/native/execute")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        serde_json::to_vec(&json!({
                            "handoff": {
                                "message_id": work["message_id"],
                                "claimed_by": work["claimed_by"],
                                "run_id": work["run_id"],
                                "step_id": work["step_id"],
                                "correlation_id": work["correlation_id"],
                            },
                            "payload": work["payload"],
                        }))
                        .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(native_result.status(), StatusCode::OK);
        let next = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/claim")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"claimed_by":"graph-worker-2","limit":1,"lease_seconds":60}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let next_body = to_bytes(next.into_body(), 16384).await.unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&next_body).unwrap()["work"][0]["step_id"],
            "graph-end"
        );
        drop(store);
        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn sqlite_environment_pinned_work_resumes_on_another_compatible_worker() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use kogwistar_store::{NamedProjectionWrite, WorkflowDesignSnapshotWrite};
        use std::time::{SystemTime, UNIX_EPOCH};
        use tower::ServiceExt;

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-worker-env-{nonce}.sqlite3"));
        let workflow_id = "pinned-python-workflow";
        let conversation_id = "pinned-python-conversation";
        let environment_id = "numpy-cpu-v2";
        let store = SqliteStore::open(&path).unwrap();
        store
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
        store
            .put_workflow_design_snapshot(
                workflow_id,
                WorkflowDesignSnapshotWrite {
                    version: 1,
                    seq: 1,
                    schema_version: 1,
                    payload_json: serde_json::to_string(&json!({
                        "nodes": [
                            {"id":"first","metadata":{"wf_start":true,"wf_op":"application.compute","wf_environment_id":environment_id}},
                            {"id":"second","metadata":{"wf_op":"application.compute","wf_environment_id":environment_id}}
                        ],
                        "edges": [{
                            "id":"edge-first-second",
                            "source_ids":["first"],
                            "target_ids":["second"],
                            "metadata":{}
                        }]
                    }))
                    .unwrap(),
                },
            )
            .unwrap();

        let reopen = || {
            router_with_application(
                ApiState {
                    health: snapshot(),
                    required_roles: vec!["reader".to_owned()],
                    implementation: ImplementationSnapshot::default(),
                    auth: AuthConfig::default(),
                },
                Arc::new(SqliteRunApplicationService::open(&path).unwrap()),
            )
        };
        let call = |app: Router, method: &'static str, uri: String, body: Value| async move {
            let response = app
                .oneshot(
                    Request::builder()
                        .method(method)
                        .uri(uri)
                        .header("content-type", "application/json")
                        .header("x-kogwistar-roles", "reader,rw")
                        .body(Body::from(serde_json::to_vec(&body).unwrap()))
                        .unwrap(),
                )
                .await
                .unwrap();
            let status = response.status();
            let bytes = to_bytes(response.into_body(), 32768).await.unwrap();
            (status, serde_json::from_slice::<Value>(&bytes).unwrap())
        };

        let app = reopen();
        let (status, submitted) = call(
            app.clone(),
            "POST",
            "/api/workflow/runs".to_owned(),
            json!({
                "workflow_id": workflow_id,
                "conversation_id": conversation_id,
                "initial_state": {"seed": 7}
            }),
        )
        .await;
        assert_eq!(status, StatusCode::ACCEPTED);
        let run_id = submitted["run_id"].as_str().unwrap().to_owned();

        let generic_worker = call(
            app.clone(),
            "POST",
            "/internal/runtime/claim".to_owned(),
            json!({
                "claimed_by": "generic-worker",
                "limit": 1,
                "lease_seconds": 60,
                "recipient_ids": ["python-worker"],
                "run_id": run_id
            }),
        )
        .await;
        assert_eq!(generic_worker.0, StatusCode::OK);
        assert_eq!(generic_worker.1["work"], json!([]));

        let wrong_environment = call(
            app.clone(),
            "POST",
            "/internal/runtime/claim".to_owned(),
            json!({
                "claimed_by": "worker-wrong-env",
                "limit": 1,
                "lease_seconds": 60,
                "recipient_ids": [crate::runtime_contract::runtime_worker_recipient(Some("numpy-gpu-v1"))]
            }),
        )
        .await;
        assert_eq!(wrong_environment.0, StatusCode::OK);
        assert_eq!(wrong_environment.1["work"], json!([]));

        let (status, claimed) = call(
            app.clone(),
            "POST",
            "/internal/runtime/claim".to_owned(),
            json!({
                "claimed_by": "worker-a",
                "limit": 1,
                "lease_seconds": 60,
                "recipient_ids": [crate::runtime_contract::runtime_worker_recipient(Some(environment_id))],
                "run_id": run_id
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let first_work = claimed["work"][0].clone();
        assert_eq!(first_work["step_id"], "first");
        assert_eq!(first_work["payload"]["op_environment_id"], environment_id);
        let first_result = call(
            app,
            "POST",
            "/internal/runtime/results".to_owned(),
            json!({
                "handoff": {
                    "message_id": first_work["message_id"],
                    "claimed_by": first_work["claimed_by"],
                    "run_id": first_work["run_id"],
                    "step_id": first_work["step_id"],
                    "correlation_id": first_work["correlation_id"]
                },
                "effect": {
                    "contract_version": 1,
                    "effect_id": format!("first-{}", run_id),
                    "state_update": [["u", {"first_worker": "worker-a"}]],
                    "successors": [{"node_id":"second","join_mask":0}],
                    "result": null
                }
            }),
        )
        .await;
        assert_eq!(first_result.0, StatusCode::OK);

        // Reopen the authoritative service between steps. A different worker
        // advertising the same environment must recover the next durable task.
        let app = reopen();
        let (status, resumed) = call(
            app.clone(),
            "POST",
            "/internal/runtime/claim".to_owned(),
            json!({
                "claimed_by": "worker-b",
                "limit": 1,
                "lease_seconds": 60,
                "recipient_ids": [crate::runtime_contract::runtime_worker_recipient(Some(environment_id))],
                "run_id": run_id
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let second_work = resumed["work"][0].clone();
        assert!(second_work["claimed_by"]
            .as_str()
            .unwrap()
            .starts_with("worker-b#"));
        assert_eq!(second_work["step_id"], "second");
        assert_eq!(second_work["payload"]["op_environment_id"], environment_id);
        assert_eq!(second_work["payload"]["state"]["first_worker"], "worker-a");
        let (status, _) = call(
            app.clone(),
            "POST",
            "/internal/runtime/results".to_owned(),
            json!({
                "handoff": {
                    "message_id": second_work["message_id"],
                    "claimed_by": second_work["claimed_by"],
                    "run_id": second_work["run_id"],
                    "step_id": second_work["step_id"],
                    "correlation_id": second_work["correlation_id"]
                },
                "effect": {
                    "contract_version": 1,
                    "effect_id": format!("second-{}", run_id),
                    "successors": [],
                    "result": {"workflow_status":"succeeded"}
                }
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);

        let (_, run) = call(
            app,
            "GET",
            format!("/api/workflow/runs/{run_id}"),
            Value::Null,
        )
        .await;
        assert_eq!(run["status"], "succeeded");
        drop(store);
        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn sqlite_invalid_worker_identity_pin_is_rejected_before_scheduling() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use kogwistar_store::{NamedProjectionWrite, WorkflowDesignSnapshotWrite};
        use std::time::{SystemTime, UNIX_EPOCH};
        use tower::ServiceExt;

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-invalid-worker-pin-{nonce}.sqlite3"));
        let workflow_id = "invalid-worker-pin-workflow";
        let store = SqliteStore::open(&path).unwrap();
        store
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
        store
            .put_workflow_design_snapshot(
                workflow_id,
                WorkflowDesignSnapshotWrite {
                    version: 1,
                    seq: 1,
                    schema_version: 1,
                    payload_json: serde_json::to_string(&json!({
                        "nodes": [
                            {"id":"start","metadata":{
                                "wf_start":true,
                                "wf_op":"application.compute",
                                "wf_environment_id":"python-cpu-v1",
                                "wf_worker_identity":{"schema_version":99}
                            }},
                            {"id":"end","metadata":{"wf_terminal":true}}
                        ],
                        "edges": [{"id":"route","source_ids":["start"],"target_ids":["end"],"metadata":{}}]
                    }))
                    .unwrap(),
                },
            )
            .unwrap();
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: vec!["reader".to_owned()],
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig::default(),
            },
            Arc::new(SqliteRunApplicationService::open(&path).unwrap()),
        );
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/workflow/runs")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader,rw")
                    .body(Body::from(
                        json!({"workflow_id":workflow_id,"conversation_id":"invalid-pin-conv"})
                            .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::CONFLICT);
        let body = to_bytes(response.into_body(), 16384).await.unwrap();
        let error: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(error["code"], "KOGWISTAR_INVALID_WORKER_IDENTITY");
        let queued = store
            .list_projected_lane_messages(LaneMessageFilter {
                namespace: Some("workflow".to_owned()),
                inbox_id: Some("workflow-runtime".to_owned()),
                status: Some("pending".to_owned()),
                limit: 32,
                ..LaneMessageFilter::default()
            })
            .unwrap();
        assert!(queued.is_empty(), "invalid pin must not enqueue runtime work");
        drop(store);
        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn sqlite_worker_identity_pin_survives_service_reopen_and_controls_claim() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
        use kogwistar_store::{NamedProjectionWrite, WorkflowDesignSnapshotWrite};
        use std::time::{SystemTime, UNIX_EPOCH};
        use tower::ServiceExt;

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-worker-pin-reopen-{nonce}.sqlite3"));
        let workflow_id = "worker-pin-reopen-workflow";
        let environment_id = "python-cpu-v1";
        let identity = json!({
            "schema_version":1,
            "environment_id":environment_id,
            "artifact_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "dependency_lock_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        });
        let identity_recipient = runtime_worker_identity_recipient(
            environment_id,
            identity["artifact_sha256"].as_str().unwrap(),
            identity["dependency_lock_sha256"].as_str().unwrap(),
        );
        let auth = AuthConfig {
            key: Some("worker-registration-test-secret".to_owned()),
            issuer: Some("test-worker-registry".to_owned()),
            audience: Some("kogwistar-runtime".to_owned()),
            ..AuthConfig::default()
        };
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let token = |registered_identity: Option<Value>| {
            let mut claims = json!({
                "sub":"identity-worker",
                "role":"reader",
                "capabilities":["spawn_process", "workflow.run.write"],
                "iss":"test-worker-registry",
                "aud":"kogwistar-runtime",
                "exp": now + 600
            });
            if let Some(identity) = registered_identity {
                claims[REGISTERED_WORKER_IDENTITY_CLAIM] = identity;
            }
            encode(
                &Header::new(Algorithm::HS256),
                &claims,
                &EncodingKey::from_secret(b"worker-registration-test-secret"),
            )
            .unwrap()
        };
        let store = SqliteStore::open(&path).unwrap();
        store
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
        store
            .put_workflow_design_snapshot(
                workflow_id,
                WorkflowDesignSnapshotWrite {
                    version: 1,
                    seq: 1,
                    schema_version: 1,
                    payload_json: serde_json::to_string(&json!({
                        "nodes": [
                            {"id":"start","metadata":{
                                "wf_start":true,
                                "wf_op":"application.compute",
                                "wf_environment_id":environment_id,
                                "wf_worker_identity":identity
                            }},
                            {"id":"end","metadata":{"wf_terminal":true}}
                        ],
                        "edges": [{"id":"route","source_ids":["start"],"target_ids":["end"],"metadata":{}}]
                    }))
                    .unwrap(),
                },
            )
            .unwrap();

        let reopen = || {
            router_with_application(
                ApiState {
                    health: snapshot(),
                    required_roles: vec!["reader".to_owned()],
                    implementation: ImplementationSnapshot::default(),
                    auth: auth.clone(),
                },
                Arc::new(SqliteRunApplicationService::open(&path).unwrap()),
            )
        };
        let call = |app: Router,
                    method: &'static str,
                    uri: String,
                    body: Value,
                    bearer: String| async move {
            let response = app
                .oneshot(
                    Request::builder()
                        .method(method)
                        .uri(uri)
                        .header("content-type", "application/json")
                        .header("authorization", format!("Bearer {bearer}"))
                        .body(Body::from(body.to_string()))
                        .unwrap(),
                )
                .await
                .unwrap();
            let status = response.status();
            let bytes = to_bytes(response.into_body(), 32768).await.unwrap();
            (status, serde_json::from_slice::<Value>(&bytes).unwrap())
        };

        let (status, submitted) = call(
            reopen(),
            "POST",
            "/api/workflow/runs".to_owned(),
            json!({"workflow_id":workflow_id,"conversation_id":"identity-reopen-conv"}),
            token(None),
        )
        .await;
        assert_eq!(status, StatusCode::ACCEPTED);
        let run_id = submitted["run_id"].as_str().unwrap().to_owned();
        let claim_uri = "/internal/runtime/claim".to_owned();
        let claim_body = json!({
            "claimed_by":"identity-worker",
            "limit":1,
            "lease_seconds":60,
            "recipient_ids":[identity_recipient],
            "run_id":run_id
        });
        let (status, _) = call(
            reopen(),
            "POST",
            claim_uri.clone(),
            claim_body.clone(),
            token(None),
        )
        .await;
        assert_eq!(status, StatusCode::FORBIDDEN);
        for (field, value) in [
            ("environment_id", json!("other-python-cpu-v1")),
            (
                "artifact_sha256",
                json!("cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"),
            ),
            (
                "dependency_lock_sha256",
                json!("dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"),
            ),
        ] {
            let mut unregistered = identity.clone();
            unregistered[field] = value;
            let (status, _) = call(
                reopen(),
                "POST",
                claim_uri.clone(),
                claim_body.clone(),
                token(Some(unregistered)),
            )
            .await;
            assert_eq!(status, StatusCode::FORBIDDEN, "tampered {field} was admitted");
        }
        let (status, claim) = call(
            reopen(),
            "POST",
            claim_uri,
            claim_body,
            token(Some(identity.clone())),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(claim["work"][0]["step_id"], "start");
        assert_eq!(
            claim["work"][0]["payload"]["initial_state"]["_rt_node_worker_identities"]["start"],
            identity
        );
        assert_eq!(claim["work"][0]["payload"]["required_worker_identity"], identity);
        drop(store);
        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn sqlite_predicate_snapshot_accepts_only_graph_valid_worker_selection() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use kogwistar_store::{NamedProjectionWrite, WorkflowDesignSnapshotWrite};
        use std::time::{SystemTime, UNIX_EPOCH};
        use tower::ServiceExt;

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-predicate-route-{nonce}.sqlite3"));
        let store = SqliteStore::open(&path).unwrap();
        store
            .replace_named_projection(
                "workflow_design",
                "predicate-wf",
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
        store
            .put_workflow_design_snapshot(
                "predicate-wf",
                WorkflowDesignSnapshotWrite {
                    version: 1,
                    seq: 1,
                    schema_version: 1,
                    payload_json: serde_json::to_string(&json!({
                        "nodes": [
                            {"id":"gate","metadata":{"wf_start":true}},
                            {"id":"left","metadata":{"wf_terminal":true}},
                            {"id":"right","metadata":{"wf_terminal":true}}
                        ],
                        "edges": [
                            {"id":"to-left","source_ids":["gate"],"target_ids":["left"],"metadata":{"wf_predicate":"if_true"}},
                            {"id":"to-right","source_ids":["gate"],"target_ids":["right"],"metadata":{"wf_predicate":"if_false","wf_is_default":true}}
                        ]
                    }))
                    .unwrap(),
                },
            )
            .unwrap();
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: vec!["reader".to_owned()],
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig::default(),
            },
            Arc::new(SqliteRunApplicationService::open(&path).unwrap()),
        );
        let submit = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/workflow/runs")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader,rw")
                    .body(Body::from(
                        r#"{"workflow_id":"predicate-wf","conversation_id":"predicate-conv"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(submit.status(), StatusCode::ACCEPTED);
        let claim = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/claim")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"claimed_by":"predicate-worker","limit":1,"lease_seconds":60}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let claim_body = to_bytes(claim.into_body(), 16_384).await.unwrap();
        let work = serde_json::from_slice::<Value>(&claim_body).unwrap()["work"][0].clone();
        let envelope = |successor: Value| {
            json!({
                "handoff": {
                    "message_id": work["message_id"],
                    "claimed_by": work["claimed_by"],
                    "run_id": work["run_id"],
                    "step_id": work["step_id"],
                    "correlation_id": work["correlation_id"],
                },
                "effect": {
                    "contract_version": 1,
                    "effect_id": format!("predicate-{}", work["run_id"].as_str().unwrap()),
                    "successors": [successor],
                }
            })
        };
        let forged = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/results")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        serde_json::to_vec(&envelope(json!({"node_id":"forged","join_mask":0})))
                            .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(forged.status(), StatusCode::CONFLICT);
        let accepted = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/results")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        serde_json::to_vec(&envelope(json!({"node_id":"right","join_mask":0})))
                            .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(accepted.status(), StatusCode::OK);
        let next = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/claim")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"claimed_by":"predicate-finisher","limit":1,"lease_seconds":60}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let next_body = to_bytes(next.into_body(), 16_384).await.unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&next_body).unwrap()["work"][0]["step_id"],
            "right"
        );
        drop(store);
        std::fs::remove_file(path).unwrap();
    }
