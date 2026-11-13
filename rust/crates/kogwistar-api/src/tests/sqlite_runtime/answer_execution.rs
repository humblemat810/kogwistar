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
            "../../../../../../contracts/golden/answer-workflow-runtime-contract-v1.json"
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
