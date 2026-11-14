    #[tokio::test]
    #[ignore = "requires KOGWISTAR_TEST_PG_DSN; full Rust-native answer workflow against disposable PostgreSQL"]
    async fn postgres_agentic_answer_workflow_executes_durably_without_python_worker() {
        use std::process::{Command, Stdio};
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        struct ServerProcess(std::process::Child);

        impl Drop for ServerProcess {
            fn drop(&mut self) {
                let _ = self.0.kill();
                let _ = self.0.wait();
            }
        }

        let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN").unwrap();
        let schema = format!("rust_answer_e2e_{}", uuid::Uuid::new_v4().simple());
        let knowledge_schema = format!("{schema}_knowledge");
        let conversation_schema = format!("{schema}_conversation");
        let workflow_schema = format!("{schema}_workflow");
        let wisdom_schema = format!("{schema}_wisdom");
        let auth_path = std::env::temp_dir().join(format!(
            "kogwistar-answer-e2e-auth-{}.sqlite",
            uuid::Uuid::new_v4().simple()
        ));
        let auth = SqliteAuthStore::open(&auth_path).unwrap();
        auth.resolve_external_identity(ResolveExternalIdentity {
            issuer: "test".to_owned(),
            subject: "alice".to_owned(),
            email: "alice@example.test".to_owned(),
            display_name: None,
            new_user_id: "alice".to_owned(),
            default_role: "rw".to_owned(),
            default_ns: "alpha".to_owned(),
        })
        .await
        .unwrap();
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
        let auth_connection = rusqlite::Connection::open(&auth_path).unwrap();
        auth_connection
            .execute(
                "INSERT INTO workflow_acl (workflow_id, user_id, role) VALUES (?1, ?2, ?3)",
                ("agentic_answering.v2", "alice", "rw"),
            )
            .unwrap();
        auth_connection
            .execute(
                "INSERT INTO workflow_acl (workflow_id, user_id, role) VALUES (?1, ?2, ?3)",
                ("packaged-rust-native-design-e2e", "alice", "rw"),
            )
            .unwrap();
        drop(auth_connection);
        drop(auth);
        let auth_config = AuthConfig {
            auth_db_url: Some(format!("sqlite:///{}", auth_path.display())),
            ..AuthConfig::default()
        };

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let chat_address = listener.local_addr().unwrap();
        let snapshot_dsn = dsn.clone();
        let snapshot_schema = knowledge_schema.clone();
        let conversation_graph_schema = conversation_schema.clone();
        let chat_server = tokio::spawn(async move {
            let (mut stream, _) =
                tokio::time::timeout(std::time::Duration::from_secs(120), listener.accept())
                    .await
                    .unwrap()
                    .unwrap();
            let mut bytes = Vec::new();
            let header_end = loop {
                let mut chunk = [0_u8; 4096];
                let count = stream.read(&mut chunk).await.unwrap();
                assert!(count > 0);
                bytes.extend_from_slice(&chunk[..count]);
                if let Some(position) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
                    break position + 4;
                }
            };
            let headers = String::from_utf8_lossy(&bytes[..header_end]).to_ascii_lowercase();
            let content_length = headers
                .lines()
                .find_map(|line| line.strip_prefix("content-length: "))
                .unwrap()
                .trim()
                .parse::<usize>()
                .unwrap();
            while bytes.len() < header_end + content_length {
                let mut chunk = [0_u8; 4096];
                let count = stream.read(&mut chunk).await.unwrap();
                assert!(count > 0);
                bytes.extend_from_slice(&chunk[..count]);
            }
            let request: Value =
                serde_json::from_slice(&bytes[header_end..header_end + content_length]).unwrap();
            assert_eq!(request["response_format"]["type"], "json_schema");
            assert_eq!(request["stream"], true);
            let snapshot_reader =
                PostgresStore::from_dsn(&snapshot_dsn, &snapshot_schema).unwrap();
            let conversation_reader =
                PostgresStore::from_dsn(&snapshot_dsn, &conversation_graph_schema).unwrap();
            let snapshots = conversation_reader
                .graph_projection_records(GraphProjectionRead {
                    scope: GraphScope {
                        namespace: "alpha".to_owned(),
                        workspace_id: None,
                        graph_space: None,
                    },
                    table: "gke_nodes".to_owned(),
                    ids: None,
                    metadata: MetadataFilter {
                        equals: BTreeMap::from([
                            ("entity_type".to_owned(), json!("context_snapshot")),
                            ("run_id".to_owned(), json!("rust-only-answer-e2e")),
                        ]),
                    },
                    limit: 10,
                })
                .await
                .unwrap();
            assert!(snapshots.iter().any(|record| {
                record.metadata.get("entity_type").and_then(Value::as_str)
                    == Some("context_snapshot")
                    && record.metadata.get("stage").and_then(Value::as_str)
                        == Some("select_used_evidence")
                    && record.metadata.get("run_id").and_then(Value::as_str)
                        == Some("rust-only-answer-e2e")
            }), "default evidence selection snapshot must commit before provider invocation");
            let snapshot = snapshots.iter().find(|record| {
                record.metadata.get("entity_type").and_then(Value::as_str)
                    == Some("context_snapshot")
                    && record.metadata.get("stage").and_then(Value::as_str)
                        == Some("generate_answer_with_citations")
                    && record.metadata.get("run_id").and_then(Value::as_str)
                        == Some("rust-only-answer-e2e")
            });
            let snapshot = snapshot.expect(
                "the model request must not start before its prompt snapshot is durable",
            );
            let snapshot_document: Value = serde_json::from_str(
                snapshot.document.as_deref().expect("snapshot has document"),
            )
            .unwrap();
            let prompt_messages: Value = serde_json::from_str(
                snapshot_document["properties"]["prompt_messages"]
                    .as_str()
                    .expect("snapshot records provider prompt messages"),
            )
            .unwrap();
            assert!(prompt_messages[1]["content"]
                .as_str()
                .unwrap()
                .contains("Where is workflow runtime state stored?"));
            let conversation_context: Value = serde_json::from_str(
                snapshot_document["properties"]["conversation_context_messages"]
                    .as_str()
                    .expect("snapshot records authorized conversation context"),
            )
            .unwrap();
            assert_eq!(conversation_context.as_array().unwrap().len(), 3);
            assert_eq!(conversation_context[1]["content"], "Earlier user question");
            assert_eq!(conversation_context[1]["node_id"], "answer-prior-turn");
            assert_eq!(conversation_context[2]["node_id"], "answer-user-turn");
            assert!(conversation_context[0]["content"]
                .as_str()
                .unwrap()
                .contains("Conversation summary:\nEarlier conversation summary"));
            assert!(conversation_context[0]["content"]
                .as_str()
                .unwrap()
                .contains("Memory context:\nPinned memory detail"));
            assert!(conversation_context[0]["content"]
                .as_str()
                .unwrap()
                .contains("Pinned knowledge references:\n- Architecture reference | A pinned graph reference"));
            assert!(!conversation_context[0]["content"]
                .as_str()
                .unwrap()
                .contains("Bob-private reference"));
            let dependency_edges = conversation_reader
                .graph_projection_records(GraphProjectionRead {
                    scope: GraphScope {
                        namespace: "alpha".to_owned(),
                        workspace_id: None,
                        graph_space: None,
                    },
                    table: "gke_edges".to_owned(),
                    ids: None,
                    metadata: MetadataFilter::default(),
                    limit: 10_000,
                })
                .await
                .unwrap();
            assert!(dependency_edges.iter().any(|record| {
                record
                    .document
                    .as_deref()
                    .and_then(|document| serde_json::from_str::<Value>(document).ok())
                    .is_some_and(|document| {
                        document["relation"] == "depends_on"
                            && document["source_ids"] == json!([snapshot.id])
                            && document["target_ids"] == json!(["answer-evidence-node"])
                    })
            }), "snapshot evidence dependency must commit before the provider call");
            for context_node_id in [
                "answer-memory-summary",
                "answer-prior-turn",
                "answer-user-turn",
                "answer-memory-context",
                "answer-knowledge-pointer",
            ] {
                assert!(dependency_edges.iter().any(|record| {
                    record
                        .document
                        .as_deref()
                        .and_then(|document| serde_json::from_str::<Value>(document).ok())
                        .is_some_and(|document| {
                            document["relation"] == "depends_on"
                                && document["source_ids"] == json!([snapshot.id])
                                && document["target_ids"] == json!([context_node_id])
                        })
                }), "snapshot must durably depend on context node {context_node_id}");
            }
            let answer = json!({
                    "text":"The workflow runtime state is stored in the graph.",
                    "reasoning":"The selected evidence states this directly.",
                    "claims":[{"claim":"Workflow runtime state is stored in the graph.","citations":[{"source_node_id":"N1","mention_index":0,"span_index":0}]}]
                })
                .to_string();
            let (text_field, remaining_fields) = answer
                .split_once(",\"reasoning\"")
                .expect("answer fixture has a top-level reasoning field");
            let chunks = [
                json!({"id":"resp-agentic-e2e","choices":[{"delta":{"content":format!("{text_field},")},"finish_reason":null}]}),
                json!({"id":"resp-agentic-e2e","choices":[{"delta":{"content":format!("\"reasoning\"{remaining_fields}")},"finish_reason":null}]}),
                json!({"id":"resp-agentic-e2e","choices":[{"delta":{},"finish_reason":"stop"}]}),
            ];
            let frames = chunks
                .iter()
                .map(|chunk| format!("data: {chunk}\r\n\r\n"))
                .collect::<Vec<_>>();
            let content_length = frames
                .iter()
                .map(String::len)
                .sum::<usize>()
                + "data: [DONE]\r\n\r\n".len();
            let headers = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {content_length}\r\nconnection: close\r\n\r\n"
            );
            stream.write_all(headers.as_bytes()).await.unwrap();
            stream.flush().await.unwrap();
            for (index, frame) in frames.iter().enumerate() {
                stream.write_all(frame.as_bytes()).await.unwrap();
                stream.flush().await.unwrap();
                if index == 0 {
                    let persisted = tokio::time::timeout(std::time::Duration::from_secs(5), async {
                        loop {
                            let events = snapshot_reader
                                .list_server_run_events("rust-only-answer-e2e", 0, 10_000)
                                .await
                                .unwrap();
                            let trace_was_persisted = events.iter().any(|event| {
                                event.event_type == "workflow.provider_event.v1"
                                    && serde_json::from_str::<Value>(&event.payload_json)
                                        .ok()
                                        .is_some_and(|payload| {
                                            payload["provider_event"]["kind"] == "field_completed"
                                                && payload["provider_event"]["details"]["field_name"] == "text"
                                        })
                            });
                            let lanes = snapshot_reader
                                .list_projected_lane_messages(LaneMessageFilter {
                                    namespace: Some("alpha".to_owned()),
                                    inbox_id: Some("agent-provider-events".to_owned()),
                                    conversation_id: Some("answer-e2e-conversation".to_owned()),
                                    msg_type: Some("agent.provider_event".to_owned()),
                                    limit: 100,
                                    ..LaneMessageFilter::default()
                                })
                                .await
                                .unwrap();
                            let event_nodes = conversation_reader
                                .graph_projection_records(GraphProjectionRead {
                                    scope: GraphScope {
                                        namespace: "alpha".to_owned(),
                                        workspace_id: None,
                                        graph_space: Some("conversation".to_owned()),
                                    },
                                    table: "gke_nodes".to_owned(),
                                    ids: None,
                                    metadata: MetadataFilter {
                                        equals: BTreeMap::from([
                                            ("msg_type".to_owned(), json!("agent.provider_event")),
                                            ("conversation_id".to_owned(), json!("answer-e2e-conversation")),
                                        ]),
                                    },
                                    limit: 100,
                                })
                                .await
                                .unwrap();
                            let lane_node_match = lanes.iter().any(|lane| {
                                event_nodes.iter().any(|node| {
                                    node.id == lane.message_id
                                        && node.metadata.get("payload_json").and_then(Value::as_str)
                                            == lane.payload_json.as_deref()
                                })
                            });
                            let event_node_ids = event_nodes
                                .iter()
                                .map(|node| node.id.clone())
                                .collect::<Vec<_>>();
                            let event_acl = conversation_reader
                                .graph_acl_records_for_entities(
                                    &GraphScope {
                                        namespace: "alpha".to_owned(),
                                        workspace_id: None,
                                        graph_space: Some("conversation".to_owned()),
                                    },
                                    "conversation",
                                    &event_node_ids,
                                )
                                .await
                                .unwrap();
                            let private_acl_match = !event_node_ids.is_empty()
                                && event_node_ids.iter().all(|event_node_id| {
                                    event_acl.iter().any(|acl| {
                                        acl.metadata.get("acl_target_entity_id").and_then(Value::as_str)
                                            == Some(event_node_id.as_str())
                                            && acl.metadata.get("acl_mode").and_then(Value::as_str)
                                                == Some("private")
                                            && acl.metadata.get("owner_id").and_then(Value::as_str)
                                                == Some("alice")
                                    })
                                });
                            if trace_was_persisted && lane_node_match && private_acl_match {
                                break;
                            }
                            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                        }
                    })
                    .await;
                    if persisted.is_err() {
                        let events = snapshot_reader
                            .list_server_run_events("rust-only-answer-e2e", 0, 10_000)
                            .await
                            .unwrap();
                        let event_types = events
                            .iter()
                            .map(|event| event.event_type.as_str())
                            .collect::<Vec<_>>();
                        let lanes = snapshot_reader
                            .list_projected_lane_messages(LaneMessageFilter {
                                namespace: Some("alpha".to_owned()),
                                inbox_id: Some("agent-provider-events".to_owned()),
                                conversation_id: Some("answer-e2e-conversation".to_owned()),
                                msg_type: Some("agent.provider_event".to_owned()),
                                limit: 100,
                                ..LaneMessageFilter::default()
                            })
                            .await
                            .unwrap();
                        let event_nodes = conversation_reader
                            .graph_projection_records(GraphProjectionRead {
                                scope: GraphScope {
                                    namespace: "alpha".to_owned(),
                                    workspace_id: None,
                                    graph_space: Some("conversation".to_owned()),
                                },
                                table: "gke_nodes".to_owned(),
                                ids: None,
                                metadata: MetadataFilter {
                                    equals: BTreeMap::from([(
                                        "msg_type".to_owned(),
                                        json!("agent.provider_event"),
                                    )]),
                                },
                                limit: 100,
                            })
                            .await
                            .unwrap();
                        let event_node_ids = event_nodes
                            .iter()
                            .map(|node| node.id.clone())
                            .collect::<Vec<_>>();
                        let event_acl = conversation_reader
                            .graph_acl_records_for_entities(
                                &GraphScope {
                                    namespace: "alpha".to_owned(),
                                    workspace_id: None,
                                    graph_space: Some("conversation".to_owned()),
                                },
                                "conversation",
                                &event_node_ids,
                            )
                            .await
                            .unwrap();
                        panic!(
                            "complete semantic fields persist before provider completion; run events={event_types:?}, lanes={}, event_nodes={event_node_ids:?}, acl={:?}",
                            lanes.len(),
                            event_acl.iter().map(|acl| &acl.metadata).collect::<Vec<_>>()
                        );
                    }
                    let misplaced_event_nodes = snapshot_reader
                        .graph_projection_records(GraphProjectionRead {
                            scope: GraphScope {
                                namespace: "alpha".to_owned(),
                                workspace_id: None,
                                graph_space: Some("conversation".to_owned()),
                            },
                            table: "gke_nodes".to_owned(),
                            ids: None,
                            metadata: MetadataFilter {
                                equals: BTreeMap::from([
                                    ("msg_type".to_owned(), json!("agent.provider_event")),
                                    (
                                        "conversation_id".to_owned(),
                                        json!("answer-e2e-conversation"),
                                    ),
                                ]),
                            },
                            limit: 100,
                        })
                        .await
                        .unwrap();
                    assert!(
                        misplaced_event_nodes.is_empty(),
                        "conversation provider-event graph nodes must not leak into the knowledge schema"
                    );
                }
            }
            stream.write_all(b"data: [DONE]\r\n\r\n").await.unwrap();
            stream.flush().await.unwrap();
        });
        let embedding_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let embedding_address = embedding_listener.local_addr().unwrap();
        let batch_embedding_requests = Arc::new(Mutex::new(Vec::<Value>::new()));
        let batch_requests = Arc::clone(&batch_embedding_requests);
        let legacy_embedding_requests = Arc::new(Mutex::new(Vec::<Value>::new()));
        let legacy_requests = Arc::clone(&legacy_embedding_requests);
        let embedding_app = axum::Router::new()
            .route(
                "/api/embed",
                axum::routing::post(move |axum::Json(request): axum::Json<Value>| {
                    let requests = Arc::clone(&batch_requests);
                    async move {
                        let input_count = request["input"]
                            .as_array()
                            .expect("Ollama batch input is an array")
                            .len();
                        requests.lock().unwrap().push(request);
                        axum::Json(json!({
                            "embeddings":vec![vec![0.6,0.8,0.0]; input_count]
                        }))
                    }
                }),
            )
            .route(
                "/api/embeddings",
                axum::routing::post(move |axum::Json(request): axum::Json<Value>| {
                    let requests = Arc::clone(&legacy_requests);
                    async move {
                        requests.lock().unwrap().push(request);
                        axum::Json(json!({"embedding":[0.6,0.8,0.0]}))
                    }
                }),
            );
        let embedding_server = tokio::spawn(async move {
            axum::serve(embedding_listener, embedding_app).await.unwrap();
        });
        let chat_provider = crate::native_answer_generation::OpenAiCompatibleChatConfig::new(
            format!("http://{chat_address}/v1/chat/completions"),
            "fixture-answer-model",
            None,
        )
        .unwrap()
        .with_streaming(true);
        let service = PostgresRunApplicationService::from_dsn_with_graph_schemas(
            &dsn,
            &knowledge_schema,
            &conversation_schema,
            &workflow_schema,
            &wisdom_schema,
        )
            .unwrap()
            .with_native_operations(crate::native_answer_persist::registry().unwrap())
            .with_owner_auth_config(&auth_config)
            .await
            .unwrap()
            .with_embedding_provider(Arc::new(FixedEmbeddingProvider {
                profile: kogwistar_engine::embedding::EmbeddingProfile {
                    provider: "ollama".to_owned(),
                    model: "fixture-embedding-model".to_owned(),
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
            }))
            .with_chat_completion_provider(chat_provider);
        service.ensure_graph_type_schemas().await.unwrap();
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

        let service_declaration = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/workflow/services".to_owned(),
                body: serde_json::to_vec(&json!({
                    "service_id":"resource-snapshot-service",
                    "target_kind":"workflow",
                    "target_ref":"resource-snapshot-workflow",
                    "target_config": {
                        "conversation_id":"resource-snapshot-conversation",
                        "runtime_kind":"sync",
                        "user_id":"alice"
                    },
                    "enabled":true
                }))
                .unwrap(),
                principal: json!({
                    "sub":"operator","role":"rw","ns":"workflow",
                    "capabilities":["service.manage","write_graph"]
                }),
            })
            .await;
        assert_eq!(
            service_declaration.status,
            StatusCode::OK.as_u16(),
            "{}",
            String::from_utf8_lossy(&service_declaration.body)
        );

        let resource_snapshot = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/syscall/v1/invoke_tool".to_owned(),
                body: serde_json::to_vec(&json!({
                    "version":"v1",
                    "op":"invoke_tool",
                    "args":{"tool_name":"resource_snapshot"}
                }))
                .unwrap(),
                principal: json!({
                    "sub":"alice","role":"rw","ns":"conversation",
                    "capabilities":["invoke_tool","project_view","read_security_scope"]
                }),
            })
            .await;
        assert_eq!(
            resource_snapshot.status,
            StatusCode::OK.as_u16(),
            "{}",
            String::from_utf8_lossy(&resource_snapshot.body)
        );
        let resource_snapshot: Value = serde_json::from_slice(&resource_snapshot.body).unwrap();
        assert_eq!(resource_snapshot["result"]["tool_name"], "resource_snapshot");
        assert_eq!(
            resource_snapshot["result"]["snapshot"]["services"],
            json!({"total_services":1,"by_health":{"degraded":1}})
        );
        let snapshot_fields = resource_snapshot["result"]["snapshot"].as_object().unwrap();
        assert!(snapshot_fields["storage_usage_bytes"].is_u64());
        assert_eq!(
            snapshot_fields["storage_usage_bytes"],
            json!(configured_storage_usage_bytes()),
            "the live MCP projection must use the configured persistence directories"
        );
        assert_eq!(
            snapshot_fields
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([
                "scheduler",
                "runs",
                "services",
                "storage_usage_bytes",
                "cost_ledger",
                "budget_model",
                "policy_infra",
            ]),
            "legacy resource snapshot keys must match Python without Rust-only diagnostics"
        );

        let put_node = |id: &str,
                        event_id: &str,
                        metadata: serde_json::Map<String, Value>,
                        document: Option<String>,
                        embedding: Option<Vec<f32>>| {
            GraphMutation {
                scope: GraphScope {
                    namespace: "alpha".to_owned(),
                    workspace_id: None,
                    graph_space: None,
                },
                table: "gke_nodes".to_owned(),
                entity_kind: "node".to_owned(),
                event_id: event_id.to_owned(),
                op: "ADD".to_owned(),
                payload: json!({"id":id}),
                record: GraphRecord {
                    id: id.to_owned(),
                    document,
                    metadata,
                    embedding,
                },
                embedding_dim: 3,
            }
        };
        let acl = |target: &str, truth_graph: &str, mode: &str| {
            put_node(
                &format!("acl-{target}"),
                &format!("e2e-acl-{target}"),
                serde_json::Map::from_iter([
                    ("entity_type".to_owned(), json!("acl_record")),
                    ("acl_truth_graph".to_owned(), json!(truth_graph)),
                    ("acl_target_grain".to_owned(), json!("node")),
                    ("acl_target_entity_id".to_owned(), json!(target)),
                    ("acl_version".to_owned(), json!(1)),
                    ("acl_mode".to_owned(), json!(mode)),
                    ("owner_id".to_owned(), json!("alice")),
                ]),
                None,
                None,
            )
        };
        let acl_edge = |target: &str, owner: &str| {
            put_node(
                &format!("acl-edge-{target}"),
                &format!("e2e-acl-edge-{target}"),
                serde_json::Map::from_iter([
                    ("entity_type".to_owned(), json!("acl_record")),
                    ("acl_truth_graph".to_owned(), json!("conversation")),
                    ("acl_target_grain".to_owned(), json!("edge")),
                    ("acl_target_entity_id".to_owned(), json!(target)),
                    ("acl_version".to_owned(), json!(1)),
                    ("acl_mode".to_owned(), json!("private")),
                    ("owner_id".to_owned(), json!(owner)),
                ]),
                None,
                None,
            )
        };
        let put_edge = |id: &str, relation: &str, source: &str, target: &str| {
            let payload = json!({
                "id": id,
                "relation": relation,
                "source_ids": [source],
                "target_ids": [target],
                "doc_id": "conv:answer-e2e-conversation"
            });
            GraphMutation {
                scope: GraphScope {
                    namespace: "alpha".to_owned(),
                    workspace_id: None,
                    graph_space: None,
                },
                table: "gke_edges".to_owned(),
                entity_kind: "edge".to_owned(),
                event_id: format!("e2e-{id}"),
                op: "ADD".to_owned(),
                payload: payload.clone(),
                record: GraphRecord {
                    id: id.to_owned(),
                    document: Some(payload.to_string()),
                    metadata: serde_json::Map::from_iter([
                        ("entity_type".to_owned(), json!("conversation_edge")),
                        ("doc_id".to_owned(), json!("conv:answer-e2e-conversation")),
                        ("relation".to_owned(), json!(relation)),
                        ("source_ids".to_owned(), json!([source])),
                        ("target_ids".to_owned(), json!([target])),
                        ("lifecycle_status".to_owned(), json!("active")),
                    ]),
                    embedding: None,
                },
                embedding_dim: 3,
            }
        };
        let item_acl = |target: &str, truth_graph: &str, grain: &str, item_id: &str, mode: &str| {
            put_node(
                &format!("acl-{grain}-{item_id}"),
                &format!("e2e-acl-{grain}-{item_id}"),
                serde_json::Map::from_iter([
                    ("entity_type".to_owned(), json!("acl_record")),
                    ("acl_truth_graph".to_owned(), json!(truth_graph)),
                    ("acl_target_grain".to_owned(), json!(grain)),
                    ("acl_target_entity_id".to_owned(), json!(target)),
                    ("acl_target_item_id".to_owned(), json!(item_id)),
                    ("acl_version".to_owned(), json!(1)),
                    ("acl_mode".to_owned(), json!(mode)),
                    ("owner_id".to_owned(), json!("alice")),
                ]),
                None,
                None,
            )
        };
        let seed_mutations = vec![
            put_node(
                "answer-prior-turn",
                "e2e-prior-turn",
                serde_json::Map::from_iter([
                    ("entity_type".to_owned(), json!("conversation_turn")),
                    (
                        "conversation_id".to_owned(),
                        json!("answer-e2e-conversation"),
                    ),
                    ("role".to_owned(), json!("user")),
                    ("turn_index".to_owned(), json!(0)),
                    ("in_conversation_chain".to_owned(), json!(true)),
                    ("lifecycle_status".to_owned(), json!("active")),
                ]),
                Some(
                    json!({"summary":"Earlier user question","metadata":{
                        "entity_type":"conversation_turn","conversation_id":"answer-e2e-conversation",
                        "role":"user","turn_index":0,"lifecycle_status":"active"
                    }})
                    .to_string(),
                ),
                None,
            ),
            acl("answer-prior-turn", "conversation", "private"),
            put_node(
                "answer-hidden-turn",
                "e2e-hidden-turn",
                serde_json::Map::from_iter([
                    ("entity_type".to_owned(), json!("conversation_turn")),
                    (
                        "conversation_id".to_owned(),
                        json!("answer-e2e-conversation"),
                    ),
                    ("role".to_owned(), json!("assistant")),
                    ("turn_index".to_owned(), json!(0)),
                    ("in_conversation_chain".to_owned(), json!(true)),
                    ("lifecycle_status".to_owned(), json!("active")),
                ]),
                Some(
                    json!({"summary":"Not visible to Alice","metadata":{
                        "entity_type":"conversation_turn","conversation_id":"answer-e2e-conversation",
                        "role":"assistant","turn_index":0,"lifecycle_status":"active"
                    }})
                    .to_string(),
                ),
                None,
            ),
            put_node(
                "acl-answer-hidden-turn",
                "e2e-acl-hidden-turn",
                serde_json::Map::from_iter([
                    ("entity_type".to_owned(), json!("acl_record")),
                    ("acl_truth_graph".to_owned(), json!("conversation")),
                    ("acl_target_grain".to_owned(), json!("node")),
                    ("acl_target_entity_id".to_owned(), json!("answer-hidden-turn")),
                    ("acl_version".to_owned(), json!(1)),
                    ("acl_mode".to_owned(), json!("private")),
                    ("owner_id".to_owned(), json!("bob")),
                ]),
                None,
                None,
            ),
            put_node(
                "answer-user-turn",
                "e2e-user-turn",
                serde_json::Map::from_iter([
                    ("entity_type".to_owned(), json!("conversation_turn")),
                    (
                        "conversation_id".to_owned(),
                        json!("answer-e2e-conversation"),
                    ),
                    ("role".to_owned(), json!("user")),
                    ("turn_index".to_owned(), json!(1)),
                    ("in_conversation_chain".to_owned(), json!(true)),
                    ("lifecycle_status".to_owned(), json!("active")),
                ]),
                Some(
                    json!({"summary":"Where is workflow runtime state stored?","metadata":{
                        "entity_type":"conversation_turn","conversation_id":"answer-e2e-conversation",
                        "role":"user","turn_index":1,"lifecycle_status":"active"
                    }})
                    .to_string(),
                ),
                None,
            ),
            acl("answer-user-turn", "conversation", "private"),
            put_node(
                "answer-memory-summary",
                "e2e-memory-summary",
                serde_json::Map::from_iter([
                    ("entity_type".to_owned(), json!("memory_summary")),
                    ("conversation_id".to_owned(), json!("answer-e2e-conversation")),
                    ("lifecycle_status".to_owned(), json!("active")),
                ]),
                Some(
                    json!({"type":"memory_summary","summary":"Earlier conversation summary","turn_index":0,"metadata":{
                        "entity_type":"memory_summary","conversation_id":"answer-e2e-conversation",
                        "lifecycle_status":"active"
                    }})
                    .to_string(),
                ),
                None,
            ),
            acl("answer-memory-summary", "conversation", "private"),
            put_node(
                "answer-memory-context",
                "e2e-memory-context",
                serde_json::Map::from_iter([
                    ("entity_type".to_owned(), json!("memory_context")),
                    (
                        "conversation_id".to_owned(),
                        json!("answer-e2e-conversation"),
                    ),
                    ("lifecycle_status".to_owned(), json!("active")),
                ]),
                Some(
                    json!({"summary":"Pinned memory detail","metadata":{
                        "entity_type":"memory_context","conversation_id":"answer-e2e-conversation",
                        "lifecycle_status":"active"
                    }})
                    .to_string(),
                ),
                None,
            ),
            acl("answer-memory-context", "conversation", "private"),
            put_node(
                "answer-knowledge-pointer",
                "e2e-knowledge-pointer",
                serde_json::Map::from_iter([
                    ("entity_type".to_owned(), json!("knowledge_reference")),
                    (
                        "conversation_id".to_owned(),
                        json!("answer-e2e-conversation"),
                    ),
                    ("lifecycle_status".to_owned(), json!("active")),
                ]),
                Some(
                    json!({
                        "label":"Architecture reference",
                        "summary":"A pinned graph reference",
                        "properties":{"refers_to_id":"answer-evidence-node"},
                        "metadata":{
                            "entity_type":"knowledge_reference",
                            "conversation_id":"answer-e2e-conversation",
                            "lifecycle_status":"active"
                        }
                    })
                    .to_string(),
                ),
                None,
            ),
            acl("answer-knowledge-pointer", "conversation", "private"),
            put_node(
                "answer-hidden-knowledge-pointer",
                "e2e-hidden-knowledge-pointer",
                serde_json::Map::from_iter([
                    ("entity_type".to_owned(), json!("knowledge_reference")),
                    ("conversation_id".to_owned(), json!("answer-e2e-conversation")),
                    ("lifecycle_status".to_owned(), json!("active")),
                ]),
                Some(
                    json!({"label":"Bob-private reference","summary":"Must not enter Alice's prompt","metadata":{
                        "entity_type":"knowledge_reference","conversation_id":"answer-e2e-conversation",
                        "lifecycle_status":"active"
                    }})
                    .to_string(),
                ),
                None,
            ),
            put_node(
                "acl-node-answer-hidden-knowledge-pointer",
                "e2e-acl-hidden-knowledge-pointer",
                serde_json::Map::from_iter([
                    ("entity_type".to_owned(), json!("acl_record")),
                    ("acl_truth_graph".to_owned(), json!("conversation")),
                    ("acl_target_grain".to_owned(), json!("node")),
                    ("acl_target_entity_id".to_owned(), json!("answer-hidden-knowledge-pointer")),
                    ("acl_version".to_owned(), json!(1)),
                    ("acl_mode".to_owned(), json!("private")),
                    ("owner_id".to_owned(), json!("bob")),
                ]),
                None,
                None,
            ),
            put_edge(
                "answer-memory-edge",
                "has_memory_context",
                "answer-user-turn",
                "answer-memory-context",
            ),
            acl_edge("answer-memory-edge", "alice"),
            put_edge(
                "answer-reference-edge",
                "references",
                "answer-user-turn",
                "answer-knowledge-pointer",
            ),
            acl_edge("answer-reference-edge", "alice"),
            put_edge(
                "answer-hidden-reference-edge",
                "references",
                "answer-user-turn",
                "answer-hidden-knowledge-pointer",
            ),
            acl_edge("answer-hidden-reference-edge", "bob"),
            put_node(
                "answer-evidence-node",
                "e2e-evidence-node",
                serde_json::Map::from_iter([
                    ("entity_type".to_owned(), json!("knowledge_node")),
                    ("label".to_owned(), json!("Rust workflow runtime")),
                    (
                        "summary".to_owned(),
                        json!("The workflow runtime records state in a durable graph."),
                    ),
                    ("level_from_root".to_owned(), json!(0)),
                    ("lifecycle_status".to_owned(), json!("active")),
                    (
                        "mentions".to_owned(),
                        json!([{"spans":[{"excerpt":"durable graph"}]}]),
                    ),
                ]),
                Some("The workflow runtime records state in a durable graph.".to_owned()),
                Some(vec![0.6_f32, 0.8, 0.0]),
            ),
            acl("answer-evidence-node", "knowledge", "public"),
            item_acl(
                "answer-evidence-node",
                "knowledge",
                "grounding",
                "answer-evidence-node::mention::0",
                "public",
            ),
            item_acl(
                "answer-evidence-node",
                "knowledge",
                "span",
                "answer-evidence-node::mention::0::span::0",
                "public",
            ),
        ];
        let benchmark_fixture_size = std::env::var("KOGWISTAR_AGENT_BENCH_FIXTURE_SIZE")
            .ok()
            .map(|value| value.parse::<usize>().expect("fixture size must be an integer"))
            .unwrap_or(0);
        assert!(
            benchmark_fixture_size <= 10_000,
            "benchmark fixture size exceeds safety limit"
        );
        if benchmark_fixture_size > 0 {
            let benchmark_nodes = (0..benchmark_fixture_size)
                .map(|index| {
                    let id = format!("answer-benchmark-distractor-{index:05}");
                    let text = format!("Unrelated benchmark knowledge record {index}.");
                    put_node(
                        &id,
                        &format!("e2e-{id}"),
                        serde_json::Map::from_iter([
                            ("entity_type".to_owned(), json!("benchmark_knowledge")),
                            ("lifecycle_status".to_owned(), json!("active")),
                        ]),
                        Some(text),
                        Some(vec![-1.0_f32, 0.0, 0.0]),
                    )
                })
                .collect::<Vec<_>>();
            service
                .store
                .apply_graph_mutations_atomic(benchmark_nodes)
                .await
                .expect("seed benchmark knowledge fixture");
        }
        for mutation in seed_mutations {
            let graph_type = mutation
                .record
                .metadata
                .get("acl_truth_graph")
                .and_then(Value::as_str)
                .or_else(|| {
                    mutation
                        .record
                        .metadata
                        .get("entity_type")
                        .and_then(Value::as_str)
                        .filter(|entity_type| entity_type.starts_with("conversation_"))
                        .map(|_| "conversation")
                })
                .or_else(|| {
                    mutation
                        .record
                        .metadata
                        .contains_key("conversation_id")
                        .then_some("conversation")
                })
                .unwrap_or("knowledge");
            let store = match graph_type {
                "conversation" => &service.conversation_store,
                _ => &service.store,
            };
            store.apply_graph_mutation(mutation).await.unwrap();
        }

        let direct_vector_hits = service
            .store
            .graph_projection_vector_query_level_lte_excluding_metadata(
                GraphProjectionVectorQuery {
                    scope: GraphScope {
                        namespace: "alpha".to_owned(),
                        workspace_id: None,
                        graph_space: None,
                    },
                    table: "gke_nodes".to_owned(),
                    query: VectorQuery {
                        embedding: vec![0.6, 0.8, 0.0],
                        limit: 10,
                        metadata: MetadataFilter::default(),
                        metric: DistanceMetric::Cosine,
                    },
                    embedding_dim: 3,
                },
                std::collections::BTreeMap::from([("entity_type".to_owned(), json!("acl_record"))]),
                3,
            )
            .await
            .unwrap();
        assert!(
            direct_vector_hits
                .iter()
                .any(|hit| hit.record.id == "answer-evidence-node"),
            "seeded evidence must be retrievable before workflow admission: {direct_vector_hits:#?}"
        );

        let contract: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/answer-workflow-runtime-contract-v1.json"
        )))
        .unwrap();
        let workflow = &contract["workflows"]["agentic_answering.v2"];
        let ops = workflow["required_ops"].as_array().unwrap();
        let design_nodes = ops.iter().map(|value| {
                let op = value.as_str().unwrap();
                json!({"id":op,"label":op,"metadata":{
                    "wf_op":op,"wf_version":"v1","wf_executor_target":"rust-native","wf_start":op=="start"
                }})
            }).collect::<Vec<_>>();
        let design_edges = workflow["required_routes"].as_array().unwrap().iter().enumerate().map(|(index,route)| json!({
                "id":format!("answer-e2e-route-{index}"),"source_ids":[route["from"]],"target_ids":[route["to"]],
                "metadata":{"wf_predicate":route["predicate"],"wf_is_default":route["default"],
                    "wf_priority":route["priority"],"wf_multiplicity":route["multiplicity"]}
            })).collect::<Vec<_>>();
        service
            .store
            .replace_named_projection(
                "workflow_design",
                "agentic_answering.v2",
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
            .await
            .unwrap();
        service
            .store
            .put_workflow_design_snapshot(
                "agentic_answering.v2",
                WorkflowDesignSnapshotWrite {
                    version: 1,
                    seq: 1,
                    schema_version: 1,
                    payload_json: serde_json::to_string(
                        &json!({"nodes":design_nodes,"edges":design_edges}),
                    )
                    .unwrap(),
                },
            )
            .await
            .unwrap();
        let debug_workflow = &contract["workflows"]["debug.rag.v1"];
        let debug_nodes = debug_workflow["required_ops"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| {
                let op = value.as_str().unwrap();
                json!({"id":op,"label":op,"metadata":{
                    "wf_op":op,"wf_version":"v1","wf_executor_target":"rust-native","wf_start":op=="start"
                }})
            })
            .collect::<Vec<_>>();
        let debug_edges = debug_workflow["required_routes"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(index, route)| {
                json!({
                    "id":format!("debug-e2e-route-{index}"),"source_ids":[route["from"]],"target_ids":[route["to"]],
                    "metadata":{"wf_predicate":route["predicate"],"wf_is_default":route["default"],
                        "wf_priority":route["priority"],"wf_multiplicity":route["multiplicity"]}
                })
            })
            .collect::<Vec<_>>();
        service
            .store
            .replace_named_projection(
                "workflow_design",
                "debug.rag.v1",
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
            .await
            .unwrap();
        service
            .store
            .put_workflow_design_snapshot(
                "debug.rag.v1",
                WorkflowDesignSnapshotWrite {
                    version: 1,
                    seq: 1,
                    schema_version: 1,
                    payload_json: serde_json::to_string(
                        &json!({"nodes":debug_nodes,"edges":debug_edges}),
                    )
                    .unwrap(),
                },
            )
            .await
            .unwrap();
        let http_secret = "rust-only-answer-http-secret";
        let now_seconds = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let token = jsonwebtoken::encode(
            &jsonwebtoken::Header::new(jsonwebtoken::Algorithm::HS256),
            &json!({
                "sub":"alice","user_id":"alice","actor_id":"alice","role":"rw","roles":["rw"],
                "ns":["alpha","workflow"],"storage_ns":"alpha",
                "capabilities":["spawn_process","workflow.run.write","workflow.run.read","workflow.design.write","workflow.design.inspect","read_graph","write_graph"],
                "exp": now_seconds + 300
            }),
            &jsonwebtoken::EncodingKey::from_secret(http_secret.as_bytes()),
        )
        .unwrap();
        let current_exe = std::env::current_exe().unwrap();
        let server_dir = current_exe
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();
        let server_binary = server_dir.join(if cfg!(windows) {
            "kogwistar-server.exe"
        } else {
            "kogwistar-server"
        });
        assert!(
            server_binary.is_file(),
            "packaged Rust server binary is missing at {}; build all targets before this ignored integration test",
            server_binary.display()
        );
        let server_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let server_port = server_listener.local_addr().unwrap().port();
        drop(server_listener);
        let server_log_path = std::env::temp_dir().join(format!(
            "kogwistar-answer-e2e-server-{}.log",
            uuid::Uuid::new_v4().simple()
        ));
        let server_log = std::fs::File::create(&server_log_path).unwrap();
        let mut server_command = Command::new(&server_binary);
        server_command.env_clear();
        for name in ["PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "USERPROFILE"] {
            if let Some(value) = std::env::var_os(name) {
                server_command.env(name, value);
            }
        }
        server_command
            .env("KOGWISTAR_BACKEND", "pg")
            .env("KOGWISTAR_PG_DSN", &dsn)
            .env("KOGWISTAR_PG_SCHEMA_BASE", &schema)
            .env("KOGWISTAR_SERVER_HOST", "127.0.0.1")
            .env("KOGWISTAR_SERVER_PORT", server_port.to_string())
            .env("KOGWISTAR_SERVER_REQUIRED_ROLES", "rw")
            .env("KOGWISTAR_RUNTIME_NATIVE_INLINE", "true")
            .env("AUTH_MODE", "jwt")
            .env("JWT_ALG", "HS256")
            .env("JWT_SECRET", http_secret)
            .env("AUTH_DB_URL", format!("sqlite:///{}", auth_path.display()))
            .env(
                "KOGWISTAR_CHAT_COMPLETIONS_ENDPOINT",
                format!("http://{chat_address}/v1/chat/completions"),
            )
            .env("KOGWISTAR_CHAT_COMPLETIONS_MODEL", "fixture-answer-model")
            .env("KOGWISTAR_CHAT_COMPLETIONS_STREAMING", "true")
            .env("KOGWISTAR_EMBEDDING_PROVIDER", "ollama")
            .env("KOGWISTAR_EMBEDDING_MODEL", "fixture-embedding-model")
            .env("KOGWISTAR_EMBEDDING_DIMENSION", "3")
            .env("KOGWISTAR_EMBEDDING_ENDPOINT", format!("http://{embedding_address}"))
            .stdout(Stdio::null())
            .stderr(Stdio::from(server_log));
        let mut server_process = ServerProcess(server_command.spawn().unwrap());
        let base_url = format!("http://127.0.0.1:{server_port}");
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(3))
            .build()
            .unwrap();
        let mut server_ready = false;
        for _ in 0..100 {
            if let Some(status) = server_process.0.try_wait().unwrap() {
                let log = std::fs::read_to_string(&server_log_path).unwrap_or_default();
                panic!("packaged Rust server exited during startup ({status}): {log}");
            }
            if client
                .get(format!("{base_url}/health"))
                .send()
                .await
                .is_ok_and(|response| response.status().is_success())
            {
                server_ready = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
        assert!(server_ready, "packaged Rust server did not become ready");
        let conversation_token = jsonwebtoken::encode(
            &jsonwebtoken::Header::new(jsonwebtoken::Algorithm::HS256),
            &json!({
                "sub":"conversation-owner","user_id":"conversation-owner","actor_id":"conversation-owner","role":"rw","roles":["rw"],
                "ns":"conversation","storage_ns":"conversation",
                "capabilities":["spawn_process","send_message"],
                "exp": now_seconds + 300
            }),
            &jsonwebtoken::EncodingKey::from_secret(http_secret.as_bytes()),
        )
        .unwrap();
        let conversation_create = client
            .post(format!("{base_url}/api/conversations"))
            .bearer_auth(&conversation_token)
            .json(&json!({
                "user_id":"conversation-owner",
                "conversation_id":"rust-only-rest-create-e2e",
                "start_node_id":"rust-only-rest-create-start"
            }))
            .send()
            .await
            .unwrap();
        let create_status = conversation_create.status();
        let create_body = conversation_create.text().await.unwrap();
        assert_eq!(
            create_status.as_u16(),
            StatusCode::OK.as_u16(),
            "{create_body}"
        );
        let created: Value = serde_json::from_str(&create_body).unwrap();
        assert_eq!(created["conversation_id"], "rust-only-rest-create-e2e");
        assert_eq!(created["user_id"], "conversation-owner");
        assert_eq!(created["start_node_id"], "rust-only-rest-create-start");
        assert_eq!(created["tail_node_id"], "rust-only-rest-create-start");
        assert_eq!(created["turn_count"], 0);
        let created_record = service
            .conversation_store
            .graph_projection_records(GraphProjectionRead {
                scope: GraphScope {
                    namespace: "conversation".to_owned(),
                    workspace_id: None,
                    graph_space: None,
                },
                table: "gke_nodes".to_owned(),
                ids: Some(vec!["rust-only-rest-create-start".to_owned()]),
                metadata: MetadataFilter::default(),
                limit: 1,
            })
            .await
            .unwrap();
        assert_eq!(created_record.len(), 1);
        assert_eq!(
            created_record[0].metadata.get("entity_type").and_then(Value::as_str),
            Some("conversation_start")
        );
        assert_eq!(
            created_record[0].metadata.get("user_id").and_then(Value::as_str),
            Some("conversation-owner")
        );
        let empty_answer = client
            .post(format!(
                "{base_url}/api/conversations/rust-only-rest-create-e2e/turns:answer"
            ))
            .bearer_auth(&conversation_token)
            .json(&json!({"text":""}))
            .send()
            .await
            .unwrap();
        let empty_status = empty_answer.status();
        let empty_body = empty_answer.text().await.unwrap();
        assert_eq!(
            empty_status.as_u16(),
            StatusCode::UNPROCESSABLE_ENTITY.as_u16(),
            "{empty_body}"
        );
        let answer_submission = client
            .post(format!(
                "{base_url}/api/conversations/rust-only-rest-create-e2e/turns:answer"
            ))
            .bearer_auth(&conversation_token)
            .json(&json!({"user_id":"body-spoof","text":"queued native question"}))
            .send()
            .await
            .unwrap();
        let answer_status = answer_submission.status();
        let answer_body = answer_submission.text().await.unwrap();
        assert_eq!(
            answer_status.as_u16(),
            StatusCode::ACCEPTED.as_u16(),
            "{answer_body}"
        );
        let answer: Value = serde_json::from_str(&answer_body).unwrap();
        assert_eq!(answer.as_object().map(serde_json::Map::len), Some(5));
        assert_eq!(answer["conversation_id"], "rust-only-rest-create-e2e");
        assert_eq!(answer["workflow_id"], "debug.rag.v1");
        assert_eq!(answer["status"], "queued");
        let answer_run_id = answer["run_id"].as_str().unwrap();
        let answer_turn_id = answer["user_turn_node_id"].as_str().unwrap();
        let answer_turn = service
            .conversation_store
            .graph_projection_records(GraphProjectionRead {
                scope: GraphScope {
                    namespace: "conversation".to_owned(),
                    workspace_id: None,
                    graph_space: None,
                },
                table: "gke_nodes".to_owned(),
                ids: Some(vec![answer_turn_id.to_owned()]),
                metadata: MetadataFilter::default(),
                limit: 1,
            })
            .await
            .unwrap();
        assert_eq!(answer_turn.len(), 1, "admitted user turn must be visible");
        let answer_turn_document: Value = serde_json::from_str(
            answer_turn[0].document.as_deref().expect("turn has a document"),
        )
        .unwrap();
        assert_eq!(answer_turn_document["conversation_id"], "rust-only-rest-create-e2e");
        assert_eq!(answer_turn_document["role"], "user");
        assert_eq!(
            answer_turn[0].metadata.get("entity_type").and_then(Value::as_str),
            Some("conversation_turn")
        );
        let acl_scope = GraphScope {
            namespace: "conversation".to_owned(),
            workspace_id: None,
            graph_space: None,
        };
        let acl_target_id = answer_turn_id.to_owned();
        let conversation_acl_records = service
            .conversation_store
            .transaction(move |uow| {
                Box::pin(async move {
                    uow.graph_acl_records_for_entities(
                        &acl_scope,
                        "conversation",
                        &[acl_target_id],
                    )
                    .await
                })
            })
            .await
            .unwrap();
        let acl_grains = conversation_acl_records
            .iter()
            .filter_map(|record| record.metadata.get("acl_target_grain").and_then(Value::as_str))
            .collect::<BTreeSet<_>>();
        assert!(
            ["node", "grounding", "span"]
                .into_iter()
                .all(|grain| acl_grains.contains(grain)),
            "conversation admission must persist node and source-span ACLs: {conversation_acl_records:#?}"
        );
        assert!(conversation_acl_records.iter().all(|record| {
            record.metadata.get("owner_id").and_then(Value::as_str)
                == Some("conversation-owner")
                && record.metadata.get("acl_mode").and_then(Value::as_str) == Some("private")
        }));
        let mut admitted_run = None;
        for _ in 0..300 {
            admitted_run = service.store.get_server_run(answer_run_id).await.unwrap();
            if admitted_run
                .as_ref()
                .is_some_and(|run| matches!(run.status.as_str(), "succeeded" | "failed"))
            {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        }
        let admitted_run = admitted_run.expect("answer admission creates a durable run");
        let debug_runtime = service
            .store
            .read_recorded_runtime_state(
                answer_run_id,
                "debug.rag.v1",
                "rust-only-rest-create-e2e",
            )
            .await
            .unwrap();
        let answer_lanes = service
            .store
            .list_projected_lane_messages(LaneMessageFilter {
                correlation_id: Some(answer_run_id.to_owned()),
                limit: 10,
                ..LaneMessageFilter::default()
            })
            .await
            .unwrap();
        assert_eq!(
            admitted_run.status,
            "succeeded",
            "REST-admitted debug RAG run must complete through the Rust-native scheduler; run={admitted_run:#?}, runtime={debug_runtime:#?}, lanes={answer_lanes:#?}, turn_document={answer_turn_document:#?}, turn_metadata={:#?}, acl_records={conversation_acl_records:#?}", answer_turn[0].metadata
        );
        let debug_runtime = debug_runtime
            .expect("completed answer run has durable runtime state");
        assert_eq!(
            debug_runtime.state["conversation_id"],
            "rust-only-rest-create-e2e"
        );
        assert_eq!(debug_runtime.state["max_retrieval_level"], 4);
        assert_eq!(debug_runtime.state["done"], true);
        assert_eq!(debug_runtime.state["answer"]["mode"], "debug_rag");
        let assistant_turn_id = debug_runtime.state["agentic_answering_result"]["assistant_turn_node_id"]
            .as_str()
            .expect("completed debug RAG run records its assistant turn ID")
            .to_owned();
        assert_eq!(
            admitted_run.assistant_turn_node_id.as_deref(),
            Some(assistant_turn_id.as_str()),
            "completed run registry should expose the durable assistant turn ID"
        );
        let assistant_turn = service
            .conversation_store
            .graph_projection_records(GraphProjectionRead {
                scope: GraphScope {
                    namespace: "conversation".to_owned(),
                    workspace_id: None,
                    graph_space: None,
                },
                table: "gke_nodes".to_owned(),
                ids: Some(vec![assistant_turn_id]),
                metadata: MetadataFilter::default(),
                limit: 1,
            })
            .await
            .unwrap();
        assert_eq!(assistant_turn.len(), 1, "assistant response must be durable");
        assert_eq!(
            assistant_turn[0].metadata.get("entity_type").and_then(Value::as_str),
            Some("assistant_turn")
        );
        assert_eq!(
            answer_turn[0].metadata.get("entity_type").and_then(Value::as_str),
            Some("conversation_turn")
        );
        assert_eq!(
            answer_turn[0].metadata.get("role").and_then(Value::as_str),
            Some("user")
        );
        assert!(answer_lanes.iter().any(|lane| lane.msg_type == "workflow.run.execute"));
        assert!(answer_lanes.iter().all(|lane| lane.status == "completed"));
        let spoofed_create = client
            .post(format!("{base_url}/api/conversations"))
            .bearer_auth(&conversation_token)
            .json(&json!({
                "user_id":"bob",
                "conversation_id":"rust-only-rest-create-spoof"
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(
            spoofed_create.status().as_u16(),
            StatusCode::FORBIDDEN.as_u16(),
            "user spoofing must fail through the same native syscall authority"
        );
        let agent_request_started = std::time::Instant::now();
        let submit = client
            .post(format!("{base_url}/api/workflow/runs"))
            .bearer_auth(&token)
            .json(&json!({
                "run_id":"rust-only-answer-e2e","workflow_id":"agentic_answering.v2",
                "conversation_id":"answer-e2e-conversation","turn_node_id":"answer-user-turn","user_id":"alice",
                "initial_state":{"conversation_id":"answer-e2e-conversation","turn_node_id":"answer-user-turn","user_id":"alice",
                    "turn_index":0,"max_candidates":10,"max_used":4,"max_retrieval_level":3,"max_iter":1}
            }))
            .send()
            .await
            .unwrap();
        let submit_status = submit.status();
        let submit_body = submit.text().await.unwrap();
        assert_eq!(
            submit_status.as_u16(),
            StatusCode::ACCEPTED.as_u16(),
            "{}",
            submit_body
        );
        let mut terminal_status = String::new();
        let mut run_response = Value::Null;
        for _ in 0..300 {
            let response = client
                .get(format!("{base_url}/api/workflow/runs/rust-only-answer-e2e"))
                .bearer_auth(&token)
                .send()
                .await
                .unwrap();
            let response_status = response.status();
            let response_body = response.text().await.unwrap();
            assert!(
                response_status.is_success(),
                "run inspection returned {response_status}: {response_body}"
            );
            run_response = serde_json::from_str(&response_body).unwrap();
            terminal_status = run_response["status"].as_str().unwrap_or_default().to_owned();
            if matches!(terminal_status.as_str(), "succeeded" | "failed" | "cancelled") {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        }
        eprintln!(
            "AGENT_BENCH_REQUEST_MS={:.3}",
            agent_request_started.elapsed().as_secs_f64() * 1_000.0
        );
        let run = service
            .store
            .get_server_run("rust-only-answer-e2e")
            .await
            .unwrap();
        if terminal_status != "succeeded" {
            let state = service
                .store
                .read_recorded_runtime_state(
                    "rust-only-answer-e2e",
                    "agentic_answering.v2",
                    "answer-e2e-conversation",
                )
                .await
                .unwrap();
            chat_server.abort();
            embedding_server.abort();
            drop(server_process);
            let server_log = std::fs::read_to_string(&server_log_path).unwrap_or_default();
            panic!(
                "native workflow did not complete through packaged server: status={terminal_status:?}, http_run={run_response:#?}, run={run:#?}, state={state:#?}, server_log={server_log}"
            );
        }
        chat_server.await.unwrap();
        embedding_server.abort();
        let batch_requests = batch_embedding_requests.lock().unwrap().clone();
        assert!(!batch_requests.is_empty(), "native retrieval must embed its query");
        assert!(batch_requests.iter().all(|request| {
            request["model"] == "fixture-embedding-model"
                && request["input"].as_array().is_some_and(|inputs| !inputs.is_empty())
        }));
        assert!(
            batch_requests.iter().any(|request| {
                request["input"]
                    .as_array()
                    .is_some_and(|inputs| inputs.iter().any(|input| input == "Where is workflow runtime state stored?"))
            }),
            "the packaged Rust server must embed the agentic workflow's query"
        );
        assert!(
            legacy_embedding_requests.lock().unwrap().is_empty(),
            "the current Rust provider must not silently pass via Ollama's legacy endpoint"
        );
        let runtime = service
            .store
            .read_recorded_runtime_state(
                "rust-only-answer-e2e",
                "agentic_answering.v2",
                "answer-e2e-conversation",
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            runtime.state["answer"]["text"],
            "The workflow runtime state is stored in the graph."
        );
        assert_eq!(
            runtime.state["used_node_ids"],
            json!(["answer-evidence-node"])
        );
        let op_log = runtime.state["op_log"].as_array().unwrap();
        for operation in ops.iter().filter_map(Value::as_str) {
            assert!(
                op_log
                    .iter()
                    .any(|value| value.as_str().is_some_and(|logged| {
                        logged == operation
                            || logged
                                .strip_prefix(operation)
                                .is_some_and(|suffix| suffix.starts_with(':'))
                    })),
                "workflow operation was not traced: {operation}; log={op_log:#?}"
            );
        }
        assert!(
            runtime.state["_rt_node_executor_targets"]
                .as_object()
                .unwrap()
                .values()
                .all(|target| target == "rust-native")
        );
        assert_eq!(
            runtime.state["answer"]["claims"][0]["citations"][0]["source_node_id"],
            "answer-evidence-node"
        );
        let result_contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/agentic-answer-result-v1.json"
        ))
        .unwrap();
        let canonical_result = json!({
            "assistant_text": runtime.state["answer"]["text"],
            "used_node_ids": runtime.state["used_node_ids"],
            "claims": runtime.state["answer"]["claims"],
        });
        assert_eq!(canonical_result, result_contract["canonical_result"]);
        use sha2::Digest;
        let canonical_json = kogwistar_contracts::canonical_json(&canonical_result, false);
        let digest = sha2::Sha256::digest(canonical_json.as_bytes())
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        assert_eq!(digest, result_contract["canonical_json_sha256"]);
        assert_eq!(runtime.state["done"], true);
        let assistant_id = runtime.state["agentic_answering_result"]["assistant_turn_node_id"]
            .as_str()
            .expect("persist response state contains its assistant turn ID")
            .to_owned();
        let assistant = service
            .conversation_store
            .graph_projection_records(GraphProjectionRead {
                scope: GraphScope {
                    namespace: "alpha".to_owned(),
                    workspace_id: None,
                    graph_space: None,
                },
                table: "gke_nodes".to_owned(),
                ids: Some(vec![assistant_id.to_owned()]),
                metadata: MetadataFilter::default(),
                limit: 1,
            })
            .await
            .unwrap();
        assert_eq!(
            assistant.len(),
            1,
            "assistant response must be durable in the graph"
        );
        let assistant_contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/assistant-turn-graph-v1.json"
        ))
        .unwrap();
        let mut assistant_document: Value = serde_json::from_str(
            assistant[0].document.as_deref().expect("assistant document persisted"),
        )
        .unwrap();
        assistant_document["metadata"] = Value::Object(
            assistant[0]
                .metadata
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
        );
        for field in assistant_contract["assistant_node"]["required_fields"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(Value::as_str)
        {
            assert!(assistant_document.get(field).is_some(), "missing assistant field {field}");
        }
        for (path, expected) in assistant_contract["assistant_node"]["field_values"]
            .as_object()
            .unwrap()
        {
            let actual = path
                .split('.')
                .fold(Some(&assistant_document), |value, key| {
                    value.and_then(|value| value.get(key))
                });
            assert_eq!(actual, Some(expected), "assistant projection mismatch at {path}");
        }
        assert_eq!(
            assistant[0].metadata.get("role").and_then(Value::as_str),
            Some("assistant")
        );
        let edges = service
            .conversation_store
            .graph_projection_records(GraphProjectionRead {
                scope: GraphScope {
                    namespace: "alpha".to_owned(),
                    workspace_id: None,
                    graph_space: None,
                },
                table: "gke_edges".to_owned(),
                ids: None,
                metadata: MetadataFilter::default(),
                limit: 100,
            })
            .await
            .unwrap();
        assert!(
            edges.iter().any(|edge| edge
                .document
                .as_deref()
                .and_then(|document| serde_json::from_str::<Value>(document).ok())
                .is_some_and(|document| {
                    document["relation"] == "next_turn"
                        && document["source_ids"] == json!(["answer-user-turn"])
                        && document["target_ids"] == json!([assistant_id])
                })),
            "assistant turn must be linked to the admitted user turn"
        );
        let generated = edges.iter().find_map(|edge| {
            edge.document
                .as_deref()
                .and_then(|document| serde_json::from_str::<Value>(document).ok())
                .filter(|document| {
                    document["relation"] == "generated"
                        && document["target_ids"] == json!([assistant_id])
                })
        });
        let mut generated = generated.expect("run-to-assistant generated edge must be durable");
        let generated_record = edges
            .iter()
            .find(|edge| {
                edge.document
                    .as_deref()
                    .and_then(|document| serde_json::from_str::<Value>(document).ok())
                    .is_some_and(|document| {
                        document["relation"] == "generated"
                            && document["target_ids"] == json!([assistant_id])
                    })
            })
            .expect("generated edge record exists");
        generated["metadata"] = Value::Object(
            generated_record
                .metadata
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
        );
        for field in assistant_contract["generated_edge"]["required_fields"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(Value::as_str)
        {
            assert!(generated.get(field).is_some(), "missing generated edge field {field}");
        }
        for (path, expected) in assistant_contract["generated_edge"]["field_values"]
            .as_object()
            .unwrap()
        {
            let actual = path
                .split('.')
                .fold(Some(&generated), |value, key| {
                    value.and_then(|value| value.get(key))
                });
            assert_eq!(actual, Some(expected), "generated edge projection mismatch at {path}");
        }
        let lanes = service
            .store
            .list_projected_lane_messages(LaneMessageFilter {
                correlation_id: Some("rust-only-answer-e2e".to_owned()),
                limit: 100,
                ..LaneMessageFilter::default()
            })
            .await
            .unwrap();
        assert!(!lanes.is_empty());
        assert!(lanes.iter().all(|lane| lane.status == "completed"));
        assert!(
            lanes.iter().all(|lane| lane.recipient_id != "python-worker"),
            "Rust-only workflow must not hand a step to a Python worker"
        );
        let durable_events = service
            .store
            .list_server_run_events("rust-only-answer-e2e", 0, 10_000)
            .await
            .unwrap();
        let parse_mcp_sse = |body: &str| -> Value {
            let data = body
                .lines()
                .find_map(|line| line.strip_prefix("data: "))
                .unwrap_or_else(|| panic!("Streamable HTTP MCP response omitted a data event: {body:?}"));
            serde_json::from_str(data).unwrap_or_else(|error| {
                panic!("invalid Streamable HTTP MCP data event ({error}): {data}")
            })
        };
        let public_events_response = client
            .get(format!(
                "{base_url}/api/runs/rust-only-answer-e2e/events/poll?after_seq=0&limit=10000"
            ))
            .bearer_auth(&token)
            .send()
            .await
            .unwrap();
        assert!(
            public_events_response.status().is_success(),
            "authorized run-event polling should expose the durable event stream"
        );
        let public_events: Value = public_events_response.json().await.unwrap();
        let public_provider_event = public_events["events"]
            .as_array()
            .unwrap()
            .iter()
            .find(|event| {
                event["event_type"] == "workflow.provider_event.v1"
                    && event["payload"]["provider_event"]["kind"] == "field_completed"
                    && event["payload"]["provider_event"]["details"]["field_name"] == "text"
            })
            .expect("authorized run-event polling returns streamed semantic event metadata");
        assert_eq!(
            public_provider_event["payload"]["provider_event"]["details"]["field_name"],
            "text"
        );
        assert!(public_provider_event["payload"]["client_sse_sequence"]
            .as_i64()
            .is_some_and(|sequence| sequence >= 0),
            "public provider event omitted client sequence: {public_provider_event:#?}"
        );
        let public_event_payload = serde_json::to_string(public_provider_event).unwrap();
        assert!(!public_event_payload.contains("The workflow runtime state is stored in the graph."));
        assert!(!public_event_payload.contains("The selected evidence states this directly."));
        let mcp_events_response = client
            .post(format!("{base_url}/mcp/workflow"))
            .bearer_auth(&token)
            .header("accept", "application/json, text/event-stream")
            .json(&json!({
                "jsonrpc":"2.0",
                "id":71,
                "method":"tools/call",
                "params":{
                    "name":"workflow.run_events",
                    "arguments":{
                        "run_id":"rust-only-answer-e2e",
                        "after_seq":0,
                        "limit":10_000
                    }
                }
            }))
            .send()
            .await
            .unwrap();
        let mcp_events_status = mcp_events_response.status();
        let mcp_events_content_type = mcp_events_response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("<missing>")
            .to_owned();
        let mcp_events_body = mcp_events_response.text().await.unwrap();
        assert!(
            mcp_events_status.is_success(),
            "authorized workflow.run_events MCP call should expose the durable event stream; status={mcp_events_status}, content_type={mcp_events_content_type}, body={mcp_events_body}"
        );
        let mcp_events = parse_mcp_sse(&mcp_events_body);
        let mcp_provider_event = mcp_events["result"]["structuredContent"]["events"]
            .as_array()
            .unwrap()
            .iter()
            .find(|event| {
                event["event_type"] == "workflow.provider_event.v1"
                    && event["payload"]["provider_event"]["kind"] == "field_completed"
                    && event["payload"]["provider_event"]["details"]["field_name"] == "text"
            })
            .expect("workflow.run_events returns the streamed provider event");
        let mcp_tools_response = client
            .post(format!("{base_url}/mcp/workflow"))
            .bearer_auth(&token)
            .header("accept", "application/json, text/event-stream")
            .json(&json!({"jsonrpc":"2.0","id":70,"method":"tools/list"}))
            .send()
            .await
            .unwrap();
        assert!(mcp_tools_response.status().is_success());
        let mcp_tools_body = mcp_tools_response.text().await.unwrap();
        let mcp_tools = parse_mcp_sse(&mcp_tools_body);
        assert!(
            mcp_tools["result"]["tools"]
                .as_array()
                .unwrap()
                .iter()
                .any(|tool| tool["name"] == "workflow.process_timeline"),
            "PostgreSQL server must advertise process_timeline to the authorized run reader"
        );
        let mcp_timeline_response = client
            .post(format!("{base_url}/mcp/workflow"))
            .bearer_auth(&token)
            .header("accept", "application/json, text/event-stream")
            .json(&json!({
                "jsonrpc":"2.0",
                "id":72,
                "method":"tools/call",
                "params":{
                    "name":"workflow.process_timeline",
                    "arguments":{"run_id":"rust-only-answer-e2e","after_seq":0}
                }
            }))
            .send()
            .await
            .unwrap();
        assert!(
            mcp_timeline_response.status().is_success(),
            "authorized workflow.process_timeline MCP call should expose durable run events"
        );
        let mcp_timeline_body = mcp_timeline_response.text().await.unwrap();
        let mcp_timeline = parse_mcp_sse(&mcp_timeline_body);
        assert_eq!(
            mcp_timeline["result"]["structuredContent"]["run_id"],
            "rust-only-answer-e2e",
            "unexpected process-timeline MCP response: {mcp_timeline:#?}"
        );
        let timeline_provider_event = mcp_timeline["result"]["structuredContent"]["events"]
            .as_array()
            .unwrap()
            .iter()
            .find(|event| {
                event["event_type"] == "workflow.provider_event.v1"
                    && event["payload"]["provider_event"]["kind"] == "field_completed"
                    && event["payload"]["provider_event"]["details"]["field_name"] == "text"
            })
            .expect("workflow.process_timeline returns the persisted semantic provider event");
        assert_eq!(
            timeline_provider_event["payload"]["span_id"],
            public_provider_event["payload"]["span_id"],
            "process timeline and REST expose the same trace identity"
        );
        assert_eq!(
            mcp_provider_event["payload"]["span_id"],
            public_provider_event["payload"]["span_id"],
            "REST and MCP expose the same trace identity"
        );
        let mcp_event_payload = serde_json::to_string(mcp_provider_event).unwrap();
        assert!(!mcp_event_payload.contains("The workflow runtime state is stored in the graph."));
        assert!(!mcp_event_payload.contains("The selected evidence states this directly."));
        let process_table_token = jsonwebtoken::encode(
            &jsonwebtoken::Header::new(jsonwebtoken::Algorithm::HS256),
            &json!({
                "sub":"alice","role":"rw","roles":["rw"],"ns":["workflow"],
                "storage_ns":"alpha","execution_ns":"alpha",
                "capabilities":["workflow.run.read"],"exp":now_seconds + 300
            }),
            &jsonwebtoken::EncodingKey::from_secret(http_secret.as_bytes()),
        )
        .unwrap();
        let process_table_tools = client
            .post(format!("{base_url}/mcp/workflow"))
            .bearer_auth(&process_table_token)
            .header("accept", "application/json, text/event-stream")
            .json(&json!({"jsonrpc":"2.0","id":73,"method":"tools/list"}))
            .send()
            .await
            .unwrap();
        assert!(process_table_tools.status().is_success());
        let process_table_tools_body = process_table_tools.text().await.unwrap();
        let process_table_tools = parse_mcp_sse(&process_table_tools_body);
        assert!(process_table_tools["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .any(|tool| tool["name"] == "workflow.process_table"));
        assert!(process_table_tools["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .any(|tool| tool["name"] == "workflow.blocked_runs"));
        let process_table_response = client
            .post(format!("{base_url}/mcp/workflow"))
            .bearer_auth(&process_table_token)
            .header("accept", "application/json, text/event-stream")
            .json(&json!({
                "jsonrpc":"2.0",
                "id":74,
                "method":"tools/call",
                "params":{
                    "name":"workflow.process_table",
                    "arguments":{
                        "status":"succeeded",
                        "workflow_id":"agentic_answering.v2",
                        "conversation_id":"answer-e2e-conversation",
                        "limit":10
                    }
                }
            }))
            .send()
            .await
            .unwrap();
        assert!(
            process_table_response.status().is_success(),
            "authorized process-table MCP call should succeed"
        );
        let process_table_body = process_table_response.text().await.unwrap();
        let process_table = parse_mcp_sse(&process_table_body);
        let processes = process_table["result"]["structuredContent"]["processes"]
            .as_array()
            .unwrap_or_else(|| panic!("unexpected process-table MCP response: {process_table:#?}"));
        let process = processes
            .iter()
            .find(|process| process["process_id"] == "rust-only-answer-e2e")
            .expect("process table should include the completed Rust-native run");
        assert_eq!(process["process_kind"], "workflow_run");
        assert_eq!(process["status"], "succeeded");
        assert_eq!(process["workflow_id"], "agentic_answering.v2");
        let blocked_runs_response = client
            .post(format!("{base_url}/mcp/workflow"))
            .bearer_auth(&process_table_token)
            .header("accept", "application/json, text/event-stream")
            .json(&json!({
                "jsonrpc":"2.0",
                "id":75,
                "method":"tools/call",
                "params":{
                    "name":"workflow.blocked_runs",
                    "arguments":{"limit":10}
                }
            }))
            .send()
            .await
            .unwrap();
        assert!(
            blocked_runs_response.status().is_success(),
            "authorized blocked-runs MCP call should succeed"
        );
        let blocked_runs_body = blocked_runs_response.text().await.unwrap();
        let blocked_runs = parse_mcp_sse(&blocked_runs_body);
        let blocked_processes = blocked_runs["result"]["structuredContent"]["processes"]
            .as_array()
            .unwrap_or_else(|| panic!("unexpected blocked-runs response: {blocked_runs:#?}"));
        assert!(
            blocked_processes.is_empty(),
            "completed PostgreSQL runs must not appear as blocked"
        );
        let streamed_provider_events = durable_events
            .iter()
            .filter(|event| event.event_type == "workflow.provider_event.v1")
            .filter_map(|event| serde_json::from_str::<Value>(&event.payload_json).ok())
            .collect::<Vec<_>>();
        assert!(
            !streamed_provider_events.is_empty(),
            "semantic provider events should have their own durable run-log entries"
        );
        let streamed_text_field = streamed_provider_events
            .iter()
            .find(|event| {
                event["provider_event"]["kind"] == "field_completed"
                    && event["provider_event"]["details"]["field_name"] == "text"
            })
            .expect("the completed text field event is durable");
        assert_eq!(
            public_provider_event["payload"]["span_id"],
            streamed_text_field["span_id"],
            "public run-event polling preserves canonical trace identity"
        );
        assert_eq!(streamed_text_field["run_id"], "rust-only-answer-e2e");
        assert_eq!(streamed_text_field["trace_id"], "rust-only-answer-e2e");
        assert_eq!(streamed_text_field["workflow_id"], "agentic_answering.v2");
        assert_eq!(
            streamed_text_field["conversation_id"],
            "answer-e2e-conversation"
        );
        assert_eq!(streamed_text_field["turn_node_id"], "answer-user-turn");
        assert_eq!(
            streamed_text_field["provider_id"],
            "openai_compatible_chat_completions"
        );
        assert_eq!(streamed_text_field["executor_target"], "rust-native");
        let event_token_id = streamed_text_field["token_id"].as_str().unwrap();
        let event_step_id = streamed_text_field["step_id"].as_str().unwrap();
        let event_step_seq = streamed_text_field["step_seq"].as_i64().unwrap();
        let event_attempt = streamed_text_field["attempt"].as_u64().unwrap();
        assert_eq!(
            streamed_text_field["span_id"],
            format!(
                "stp:rust-only-answer-e2e:{event_token_id}:{event_step_seq}:{event_attempt}:{event_step_id}"
            )
        );
        assert_eq!(
            streamed_text_field["parent_span_id"],
            format!("tok:rust-only-answer-e2e:{event_token_id}")
        );
        let mut receive_request = kogwistar_runtime::RuntimeNativeStepRequest {
            executor_target: kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET.to_owned(),
            operation: "agent.event.receive".to_owned(),
            op_version: "v1".to_owned(),
            run_id: "rust-only-answer-e2e".to_owned(),
            workflow_id: "agentic_answering.v2".to_owned(),
            node_id: "receive-provider-event".to_owned(),
            conversation_id: "answer-e2e-conversation".to_owned(),
            turn_node_id: Some("answer-user-turn".to_owned()),
            token_id: "event-consumer-token".to_owned(),
            parent_token_id: None,
            step_seq: 99,
            state: Map::new(),
            runtime_routes: Vec::new(),
            ephemeral_context: Map::new(),
        };
        let receive_principal = service
            .native_read_principal(&receive_request, None)
            .await
            .unwrap_or_else(|_| panic!("the admitted run owner retains read_graph authority"));
        let claimed_context = service
            .claim_native_provider_event_context(&receive_request, &receive_principal)
            .await
            .unwrap()
            .expect("a visible semantic event should be claimable");
        assert_eq!(claimed_context["event"]["trace"]["run_id"], "rust-only-answer-e2e");
        assert_eq!(
            claimed_context["event"]["trace"]["attempt"],
            event_attempt,
            "provider event attempt matches its producing workflow attempt"
        );
        assert_eq!(
            claimed_context["event"]["trace"]["producer_attempt"],
            event_attempt,
            "producer attempt metadata remains aligned with the event attempt"
        );
        assert_eq!(
            claimed_context["claim"]["message_id"],
            format!("msg:{}", claimed_context["event"]["event_id"].as_str().unwrap())
        );
        receive_request.ephemeral_context.insert(
            "authorized_provider_event".to_owned(),
            claimed_context["event"].clone(),
        );
        receive_request.ephemeral_context.insert(
            "authorized_provider_event_claim".to_owned(),
            claimed_context["claim"].clone(),
        );
        let receive_effect = kogwistar_runtime::execute_native_step_with_registry(
            &receive_request,
            &kogwistar_runtime::NativeOperationRegistry::default(),
        )
        .unwrap();
        let received_state = kogwistar_contracts::apply_state_update_pairs(
            Value::Object(Map::new()),
            receive_effect.state_update.iter().map(|update| match update {
                kogwistar_runtime::RecordedStateUpdate::Pair((mode, payload)) => {
                    (mode.as_str(), payload)
                }
                kogwistar_runtime::RecordedStateUpdate::Object { mode, payload } => {
                    (mode.as_str(), payload)
                }
            }),
        )
        .unwrap();
        assert_eq!(received_state["agent_event_current"], claimed_context["event"]);
        assert_eq!(received_state["agent_event_claim"], claimed_context["claim"]);
        assert!(receive_effect.provider_event_acknowledgements.is_empty());
        let claimed_lane = service
            .store
            .get_projected_lane_message(
                claimed_context["claim"]["message_id"].as_str().unwrap(),
            )
            .await
            .unwrap()
            .expect("the provider event remains in its durable lane");
        assert_eq!(claimed_lane.status, "claimed");
        assert_eq!(
            claimed_lane.claimed_by.as_deref(),
            claimed_context["claim"]["claimed_by"].as_str()
        );
        let durable_run = run.as_ref().expect("completed run remains durable");
        assert_eq!(durable_run.user_id.as_deref(), Some("alice"));
        let admission_event = durable_events
            .iter()
            .find(|event| event.event_type == "run.created")
            .expect("run admission owner binding remains durable");
        let admission_payload: Value =
            serde_json::from_str(&admission_event.payload_json).unwrap();
        assert_eq!(admission_payload["run_id"], durable_run.run_id);
        assert_eq!(admission_payload["runtime_owner_binding"]["version"], 1);
        assert_eq!(admission_payload["runtime_owner_binding"]["user_id"], "alice");
        let native_traces = durable_events
            .iter()
            .filter(|event| event.event_type == "workflow.trace.v1")
            .filter_map(|event| serde_json::from_str::<Value>(&event.payload_json).ok())
            .filter(|event| event["trace"]["executor_target"] == "rust-native")
            .collect::<Vec<_>>();
        assert!(!native_traces.is_empty(), "native execution traces must be durable");
        let provider_event_traces = native_traces
            .iter()
            .filter(|event| event["trace"]["event"] == "provider.semantic_event.observed")
            .collect::<Vec<_>>();
        assert!(
            provider_event_traces.iter().any(|event| {
                event["trace"]["provider_event"]["kind"] == "field_completed"
                    && event["trace"]["provider_event"]["details"]["field_name"] == "text"
            }),
            "the streamed complete text field should be durably summarized: {provider_event_traces:#?}"
        );
        let provider_event_json = serde_json::to_string(&provider_event_traces).unwrap();
        assert!(!provider_event_json.contains("The workflow runtime state is stored in the graph."));
        assert!(!provider_event_json.contains("The selected evidence states this directly."));
        assert!(!provider_event_json.contains("\"value\""));
        for event in native_traces {
            let trace = &event["trace"];
            let token_id = trace["token_id"].as_str().unwrap();
            let step_seq = trace["step_seq"].as_i64().unwrap();
            let node_id = trace["node_id"].as_str().unwrap();
            let attempt = trace["attempt"].as_u64().unwrap();
            assert_eq!(event["run_id"], "rust-only-answer-e2e");
            assert_eq!(event["run_id"], durable_run.run_id);
            assert_eq!(trace["run_id"], event["run_id"]);
            assert_eq!(trace["trace_id"], event["run_id"]);
            assert_eq!(event["step_id"], node_id);
            assert_eq!(event["token_id"], token_id);
            assert_eq!(
                trace["span_id"],
                format!("stp:rust-only-answer-e2e:{token_id}:{step_seq}:{attempt}:{node_id}")
            );
            assert_eq!(
                trace["parent_span_id"],
                format!("tok:rust-only-answer-e2e:{token_id}")
            );
        }

        let design_id = "packaged-rust-native-design-e2e";
        for node in [
            json!({
                "designer_id":"alice","node_id":"start","label":"Increment",
                "op":"increment-counter","start":true,
                "metadata":{"wf_executor_target":"rust-native"}
            }),
            json!({
                "designer_id":"alice","node_id":"end","label":"Finish",
                "op":"emit-result","terminal":true,
                "metadata":{"wf_executor_target":"rust-native"}
            }),
        ] {
            let response = client
                .post(format!("{base_url}/api/workflow/design/{design_id}/nodes"))
                .bearer_auth(&token)
                .json(&node)
                .send()
                .await
                .unwrap();
            let status = response.status();
            let body = response.text().await.unwrap();
            assert_eq!(status.as_u16(), StatusCode::OK.as_u16(), "{body}");
        }
        let edge_response = client
            .post(format!("{base_url}/api/workflow/design/{design_id}/edges"))
            .bearer_auth(&token)
            .json(&json!({
                "designer_id":"alice","edge_id":"start-to-end",
                "src":"start","dst":"end","is_default":true
            }))
            .send()
            .await
            .unwrap();
        let edge_status = edge_response.status();
        let edge_body = edge_response.text().await.unwrap();
        assert_eq!(edge_status.as_u16(), StatusCode::OK.as_u16(), "{edge_body}");

        let design_graph_response = client
            .get(format!("{base_url}/api/workflow/design/{design_id}/graph"))
            .bearer_auth(&token)
            .send()
            .await
            .unwrap();
        let design_graph_status = design_graph_response.status();
        let design_graph_body = design_graph_response.text().await.unwrap();
        assert_eq!(
            design_graph_status.as_u16(),
            StatusCode::OK.as_u16(),
            "{design_graph_body}"
        );
        let design_graph: Value = serde_json::from_str(&design_graph_body).unwrap();
        assert_eq!(design_graph["nodes"].as_array().unwrap().len(), 2);
        assert_eq!(design_graph["edges"].as_array().unwrap().len(), 1);
        let start = design_graph["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|node| node["id"] == "start")
            .unwrap();
        assert_eq!(start["metadata"]["wf_op"], "increment-counter");
        assert_eq!(start["metadata"]["wf_executor_target"], "rust-native");

        let design_run_id = "packaged-rust-native-design-run-e2e";
        let design_submit = client
            .post(format!("{base_url}/api/workflow/runs"))
            .bearer_auth(&token)
            .json(&json!({
                "run_id":design_run_id,"workflow_id":design_id,
                "conversation_id":"packaged-rust-native-design-conversation",
                "user_id":"alice","initial_state":{"counter":1}
            }))
            .send()
            .await
            .unwrap();
        let design_submit_status = design_submit.status();
        let design_submit_body = design_submit.text().await.unwrap();
        assert_eq!(
            design_submit_status.as_u16(),
            StatusCode::ACCEPTED.as_u16(),
            "{design_submit_body}"
        );
        let mut design_run_status = String::new();
        for _ in 0..300 {
            let response = client
                .get(format!("{base_url}/api/workflow/runs/{design_run_id}"))
                .bearer_auth(&token)
                .send()
                .await
                .unwrap();
            let status = response.status();
            let body = response.text().await.unwrap();
            assert!(status.is_success(), "{status}: {body}");
            let run: Value = serde_json::from_str(&body).unwrap();
            design_run_status = run["status"].as_str().unwrap_or_default().to_owned();
            if matches!(design_run_status.as_str(), "succeeded" | "failed" | "cancelled") {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
        assert_eq!(design_run_status, "succeeded");
        let design_runtime = service
            .store
            .read_recorded_runtime_state(
                design_run_id,
                design_id,
                "packaged-rust-native-design-conversation",
            )
            .await
            .unwrap()
            .expect("packaged Rust design run persists its final runtime state");
        assert_eq!(design_runtime.state["counter"], 2);
        let design_events = service
            .store
            .list_server_run_events(design_run_id, 0, 100)
            .await
            .unwrap();
        assert!(design_events.iter().any(|event| {
            event.event_type == "workflow.trace.v1"
                && event.payload_json.contains("\"seam_crossings\":[]")
        }));
        drop(server_process);
        drop(service);
        let (cleanup, connection) = tokio_postgres::connect(&dsn, tokio_postgres::NoTls)
            .await
            .unwrap();
        let task = tokio::spawn(connection);
        cleanup
            .batch_execute(&format!(
                "DROP SCHEMA IF EXISTS \"{knowledge_schema}\" CASCADE; \
                 DROP SCHEMA IF EXISTS \"{conversation_schema}\" CASCADE; \
                 DROP SCHEMA IF EXISTS \"{workflow_schema}\" CASCADE; \
                 DROP SCHEMA IF EXISTS \"{wisdom_schema}\" CASCADE"
            ))
            .await
            .unwrap();
        drop(cleanup);
        task.await.unwrap().unwrap();
        let _ = std::fs::remove_file(auth_path);
        let _ = std::fs::remove_file(server_log_path);
    }
