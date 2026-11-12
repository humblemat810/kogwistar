#[tokio::test]
async fn sqlite_chroma_conversation_create_and_turn_admission_are_canonical_and_atomic() {
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use kogwistar_store::{
        DistanceMetric, GraphMutation, GraphMutationBatch, GraphReadStore, GraphRecord,
        GraphScope, GraphWriteStore, MetadataFilter, NamedProjectionWrite, VectorQuery,
        WorkflowDesignSnapshotWrite,
    };
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tower::ServiceExt;

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let chat_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let chat_endpoint = format!(
        "http://{}/v1/chat/completions",
        chat_listener.local_addr().unwrap()
    );
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
                let length = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().ok())
                            .flatten()
                    })
                    .unwrap_or(0);
                if bytes.len() >= end + 4 + length {
                    break (end, length);
                }
            }
        };
        let request: Value =
            serde_json::from_slice(&bytes[header_end + 4..header_end + 4 + content_length])
                .unwrap();
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
        })
        .to_string();
        let frames = [
                format!("data: {}\r\n\r\n", json!({"id":"sqlite-chroma-answer","choices":[{"delta":{"content":content},"finish_reason":null}]})),
                format!("data: {}\r\n\r\n", json!({"id":"sqlite-chroma-answer","choices":[{"delta":{},"finish_reason":"stop"}]})),
                "data: [DONE]\r\n\r\n".to_owned(),
            ].concat();
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
            frames.len(),
            frames
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
                if let Some(header_end) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
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
            let body: Value =
                serde_json::from_slice(&bytes[header_end + 4..header_end + 4 + content_length])
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
            } else if request_line.contains("/get HTTP") && body.to_string().contains("acl_record")
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
                response.len(),
                response
            );
            stream.write_all(wire.as_bytes()).await.unwrap();
        }
        projection_requests
    });
    let live_chroma_endpoint = std::env::var("KOGWISTAR_TEST_CHROMA_URL").ok();
    let chroma_suffix = uuid::Uuid::new_v4().simple().to_string();
    let config = |entity_kind: &str| kogwistar_store_chroma::ChromaStoreConfig {
        endpoint: live_chroma_endpoint
            .clone()
            .unwrap_or_else(|| endpoint.clone()),
        collection: format!("conversation-{entity_kind}-{chroma_suffix}"),
        ..Default::default()
    };
    let graph = kogwistar_store_chroma::ChromaGraphStore::connect_entity_collections(
        config("nodes"),
        config("edges"),
    )
    .await
    .unwrap();
    if live_chroma_endpoint.is_some() {
        graph
            .upsert_graph_record_of_kind(
                "conversation",
                kogwistar_store::GraphEntityKind::Node,
                GraphRecord {
                    id: "knowledge-node".to_owned(),
                    document: Some(
                        "The runtime persists workflow state in the graph.".to_owned(),
                    ),
                    metadata: serde_json::from_value(json!({
                        "entity_type": "node",
                        "workspace_id": "workspace-a",
                        "graph_space": "conversation",
                        "label": "Workflow runtime state",
                        "summary": "The runtime persists workflow state in the graph.",
                        "level_from_root": 0,
                        "lifecycle_status": "active"
                    }))
                    .unwrap(),
                    embedding: Some(vec![0.6, 0.8]),
                },
            )
            .await
            .unwrap();
    }
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
        .with_chroma_graph(graph.clone())
        .with_embedding_provider(embedding_provider.clone())
        .with_chat_completion_provider(
            crate::native_answer_generation::OpenAiCompatibleChatConfig::new(
                chat_endpoint.clone(),
                "sqlite-chroma-fixture-model",
                None,
            )
            .unwrap()
            .with_streaming(true),
        )
        .with_owner_auth_config(&owner_auth)
        .await
        .unwrap();

    let mut knowledge_payload: Value = serde_json::from_str::<Value>(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/python-graph-model-payloads-v1.json"
    )))
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
    if live_chroma_endpoint.is_some() {
        graph
            .upsert_graph_record_of_kind(
                "conversation",
                kogwistar_store::GraphEntityKind::Node,
                GraphRecord {
                    id: acl_payload["id"].as_str().unwrap().to_owned(),
                    document: Some(acl_payload["summary"].as_str().unwrap().to_owned()),
                    metadata: serde_json::from_value(acl_payload["metadata"].clone()).unwrap(),
                    embedding: Some(vec![0.6, 0.8]),
                },
            )
            .await
            .unwrap();
    }
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
    if live_chroma_endpoint.is_some() {
        let mut metadata = MetadataFilter::default();
        metadata
            .equals
            .insert("workspace_id".to_owned(), json!("workspace-a"));
        metadata
            .equals
            .insert("graph_space".to_owned(), json!("conversation"));
        let hits = graph
            .vector_query_of_kind_numeric_lte(
                "conversation",
                kogwistar_store::GraphEntityKind::Node,
                &VectorQuery {
                    embedding: vec![0.6, 0.8],
                    limit: 20,
                    metadata,
                    metric: DistanceMetric::Cosine,
                },
                "level_from_root",
                4.0,
            )
            .await
            .unwrap();
        assert!(
            hits.iter().any(|matched| matched.record.id == "knowledge-node"),
            "canonical SQLite admission must not remove the seeded Chroma retrieval hit"
        );
    }

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
    assert!(service.supports_http_operation(
        "POST",
        "/api/conversations/conversation-create-contract/turns:answer"
    ));
    assert!(service.supports_mcp_tool("conversation", "conversation.ask"));

    let principal = json!({
        "sub":"conversation-owner", "user_id":"conversation-owner", "role":"rw",
        "ns":["conversation"], "storage_ns":"conversation",
        "workspace":"workspace-a", "graph_space":"conversation",
        "capabilities":["spawn_process", "send_message", "write_graph", "read_graph", "workflow.run.write"]
    });
    let http_secret = "sqlite-chroma-public-ingress-test-secret";
    let token = jsonwebtoken::encode(
        &jsonwebtoken::Header::new(jsonwebtoken::Algorithm::HS256),
        &json!({
            "sub":"conversation-owner", "user_id":"conversation-owner",
            "actor_id":"conversation-owner", "role":"rw", "roles":["rw"],
            "ns":"conversation", "storage_ns":"conversation",
            "workspace":"workspace-a", "graph_space":"conversation",
            "capabilities":["spawn_process", "send_message", "write_graph", "read_graph", "workflow.run.write"],
            "exp":4102444800_u64
        }),
        &jsonwebtoken::EncodingKey::from_secret(http_secret.as_bytes()),
    )
    .unwrap();
    let app = router_with_application(
        ApiState {
            health: snapshot(),
            required_roles: Vec::new(),
            implementation: ImplementationSnapshot::default(),
            auth: AuthConfig {
                mode: "jwt".to_owned(),
                algorithm: Some("HS256".to_owned()),
                key: Some(http_secret.to_owned()),
                auth_db_url: Some(format!("sqlite:///{}", auth_path.display())),
                ..AuthConfig::default()
            },
        },
        Arc::new(service.clone()),
    );
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
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/conversations")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::from(
                    serde_json::to_vec(&json!({
                        "user_id":"conversation-owner",
                        "conversation_id":"conversation-create-contract",
                        "start_node_id":"conversation-create-start"
                    }))
                    .unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    let response_status = response.status();
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), 65_536).await.unwrap()).unwrap();
    assert_eq!(response_status, StatusCode::OK, "{body}");
    assert_eq!(body["conversation_id"], "conversation-create-contract");
    assert_eq!(body["start_node_id"], "conversation-create-start");
    assert_eq!(body["turn_count"], 0);

    let syscall_create_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/syscall/v1/create_conversation")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::from(
                    serde_json::to_vec(&json!({
                        "version":"v1",
                        "op":"create_conversation",
                        "args":{
                            "user_id":"conversation-owner",
                            "conversation_id":"conversation-syscall-create-contract",
                            "start_node_id":"conversation-syscall-create-start"
                        }
                    }))
                    .unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(syscall_create_response.status(), StatusCode::OK);
    let syscall_create_body: Value = serde_json::from_slice(
        &to_bytes(syscall_create_response.into_body(), 65_536)
            .await
            .unwrap(),
    )
    .unwrap();
    let mut expected_syscall_result: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/conversation-create-syscall-result-v1.json"
    )))
    .unwrap();
    expected_syscall_result["conversation_id"] = json!("conversation-syscall-create-contract");
    expected_syscall_result["user_id"] = json!("conversation-owner");
    expected_syscall_result["start_node_id"] = json!("conversation-syscall-create-start");
    expected_syscall_result["tail_node_id"] = json!("conversation-syscall-create-start");
    assert_eq!(syscall_create_body["version"], "v1");
    assert_eq!(syscall_create_body["op"], "create_conversation");
    assert_eq!(syscall_create_body["status"], "ok");
    assert_eq!(syscall_create_body["result"], expected_syscall_result);

    let events = service
        .store
        .replay_raw_events("conversation", 0, 20)
        .unwrap();
    assert!(events.iter().any(|event| {
        event.entity_kind == "node"
            && event.entity_id == "conversation-create-start"
            && serde_json::from_str::<Value>(&event.payload_json)
                .ok()
                .is_some_and(|payload| {
                    payload["metadata"]["entity_type"] == "conversation_start"
                        && payload["metadata"]["conversation_id"] == "conversation-create-contract"
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
        service
            .store
            .list_stage1_node_projections("conversation")
            .unwrap()
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

    let event_count_before_empty_answer = service
        .store
        .replay_raw_events("conversation", 0, 50)
        .unwrap()
        .len();
    let validation_contract: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/conversation-answer-validation-v1.json"
    )))
    .unwrap();
    assert_eq!(validation_contract["schema_version"], 1);
    for case in validation_contract["cases"].as_array().unwrap() {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/conversations/conversation-create-contract/turns:answer")
                    .header("content-type", "application/json")
                    .header("authorization", format!("Bearer {token}"))
                    .body(Body::from(
                        json!({"text":case["text"]}).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            response.status().as_u16(),
            case["http_status"].as_u64().unwrap() as u16,
            "case {}",
            case["name"]
        );
    }
    assert_eq!(
        service
            .store
            .replay_raw_events("conversation", 0, 50)
            .unwrap()
            .len(),
        event_count_before_empty_answer,
        "invalid REST turn admission must not append canonical graph events"
    );

    let send_message = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/conversations/conversation-create-contract/turns:answer")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::from(
                    serde_json::to_vec(&json!({
                        "text":"Where is workflow runtime state persisted?",
                        "user_id":"body-spoof",
                        "workflow_id":workflow_id
                    }))
                    .unwrap(),
                ))
                .unwrap(),
        )
        .await;
    let send_message = send_message.unwrap();
    assert_eq!(send_message.status(), StatusCode::ACCEPTED);
    let send_response: Value =
        serde_json::from_slice(&to_bytes(send_message.into_body(), 65_536).await.unwrap()).unwrap();
    assert_eq!(send_response.as_object().unwrap().len(), 5);
    assert_eq!(
        send_response["conversation_id"],
        "conversation-create-contract"
    );
    assert_eq!(send_response["workflow_id"], workflow_id);
    assert_eq!(send_response["status"], "queued");
    let run_id = send_response["run_id"].as_str().unwrap();
    assert!(!run_id.is_empty());
    assert!(
        send_response["user_turn_node_id"]
            .as_str()
            .is_some_and(|id| !id.is_empty())
    );
    assert_eq!(
        service
            .store
            .get_server_run(run_id)
            .unwrap()
            .expect("admitted answer run is durable")
            .user_id
            .as_deref(),
        Some("conversation-owner"),
        "authenticated owner takes precedence over body user_id"
    );
    let events = service
        .store
        .replay_raw_events("conversation", 0, 50)
        .unwrap();
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

    let pending_messages = service
        .store
        .list_projected_lane_messages(kogwistar_store::LaneMessageFilter {
            correlation_id: Some(run_id.to_owned()),
            limit: 10,
            ..Default::default()
        })
        .unwrap();
    assert!(
        pending_messages.iter().any(
            |message| message.msg_type == "workflow.run.execute" && message.status == "pending"
        )
    );
    drop(service);

    let recovered_graph = kogwistar_store_chroma::ChromaGraphStore::connect_entity_collections(
        config("nodes"),
        config("edges"),
    )
    .await
    .unwrap();
    let service = SqliteRunApplicationService::open(&path)
        .unwrap()
        .with_chroma_graph(recovered_graph)
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
    assert!(
        service
            .store
            .replay_raw_events("conversation", 0, 50)
            .unwrap()
            .iter()
            .any(|event| event.entity_kind == "node"
                && serde_json::from_str::<Value>(&event.payload_json)
                    .ok()
                    .is_some_and(|payload| payload["metadata"]["entity_type"]
                        == "conversation_turn"
                        && payload["metadata"]["conversation_id"]
                            == "conversation-create-contract"))
    );

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
    assert_eq!(
        runtime.state["_rt_graph_scope"]["workspace"], "workspace-a",
        "the native workflow must retain its admitted workspace scope"
    );
    assert_eq!(runtime.state["done"], true);
    assert_eq!(runtime.state["answer"]["mode"], "debug_rag");
    assert_eq!(runtime.state["used_node_ids"], json!(["knowledge-node"]));
    assert!(
        runtime.state["answer"]["text"]
            .as_str()
            .is_some_and(|text| text.contains("knowledge-node"))
    );
    let projected_pointer = runtime.state["projected_pointer_ids"][0]
        .as_str()
        .expect("selected evidence receives a durable conversation pointer")
        .to_owned();
    assert!(
        service
            .store
            .latest_entity_events("conversation", "node", &[projected_pointer.clone()])
            .unwrap()
            .contains_key(&projected_pointer)
    );
    let assistant_id = runtime.state["agentic_answering_result"]["assistant_turn_node_id"]
        .as_str()
        .expect("answer workflow records the assistant turn")
        .to_owned();
    let assistant = service
        .store
        .latest_entity_events("conversation", "node", &[assistant_id.clone()])
        .unwrap();
    assert!(
        assistant.contains_key(&assistant_id),
        "assistant turn is canonical"
    );
    let assistant_contract: Value = serde_json::from_str(include_str!(
        "../../../../../contracts/golden/assistant-turn-graph-v1.json"
    ))
    .unwrap();
    let assistant_payload: Value =
        serde_json::from_str(&assistant[&assistant_id].payload_json).unwrap();
    for field in assistant_contract["assistant_node"]["required_fields"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(Value::as_str)
    {
        assert!(assistant_payload.get(field).is_some(), "missing assistant field {field}");
    }
    for (path, expected) in assistant_contract["assistant_node"]["field_values"]
        .as_object()
        .unwrap()
    {
        let actual = path
            .split('.')
            .fold(Some(&assistant_payload), |value, key| {
                value.and_then(|value| value.get(key))
            });
        assert_eq!(actual, Some(expected), "assistant projection mismatch at {path}");
    }
    let edges = service
        .store
        .replay_raw_events("conversation", 0, 100)
        .unwrap();
    assert!(
        edges.iter().any(|event| {
            event.entity_kind == "edge"
                && serde_json::from_str::<Value>(&event.payload_json)
                    .ok()
                    .is_some_and(|payload| {
                        payload["relation"] == "next_turn"
                            && payload["source_ids"] == json!([runtime.state["turn_node_id"]])
                            && payload["target_ids"] == json!([assistant_id])
                    })
        }),
        "assistant turn {assistant_id} is linked to user turn {} by the canonical graph edge",
        runtime.state["turn_node_id"]
    );
    let generated_edge = edges.iter().find_map(|event| {
        (event.entity_kind == "edge")
            .then(|| serde_json::from_str::<Value>(&event.payload_json).ok())
            .flatten()
            .filter(|payload| {
                payload["relation"] == "generated"
                    && payload["target_ids"] == json!([assistant_id])
            })
    });
    let generated_edge = generated_edge.expect("run-to-assistant generated edge is canonical");
    for field in assistant_contract["generated_edge"]["required_fields"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(Value::as_str)
    {
        assert!(generated_edge.get(field).is_some(), "missing generated edge field {field}");
    }
    for (path, expected) in assistant_contract["generated_edge"]["field_values"]
        .as_object()
        .unwrap()
    {
        let actual = path
            .split('.')
            .fold(Some(&generated_edge), |value, key| {
                value.and_then(|value| value.get(key))
            });
        assert_eq!(actual, Some(expected), "generated edge projection mismatch at {path}");
    }

    let mcp_ask = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp/conversation")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::from(
                    serde_json::to_vec(&json!({
                        "jsonrpc":"2.0",
                        "id":"sqlite-chroma-conversation-ask",
                        "method":"tools/call",
                        "params":{
                            "name":"conversation.ask",
                            "arguments":{
                                "conversation_id":"conversation-create-contract",
                                "text":"Does MCP use the same durable turn path?",
                                "workflow_id":workflow_id
                            }
                        }
                    }))
                    .unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(mcp_ask.status(), StatusCode::OK);
    let mcp_response: Value =
        serde_json::from_slice(&to_bytes(mcp_ask.into_body(), 65_536).await.unwrap()).unwrap();
    assert_eq!(mcp_response["result"]["status"], "queued");
    let mcp_run_id = mcp_response["result"]["run_id"]
        .as_str()
        .expect("MCP ask returns its queued run id");
    for _ in 0..32 {
        let status = service
            .store
            .get_server_run(mcp_run_id)
            .unwrap()
            .expect("MCP ask run remains durable");
        if matches!(status.status.as_str(), "succeeded" | "failed" | "cancelled") {
            break;
        }
        assert!(
            service.execute_inline_native_work_once().await.unwrap(),
            "native scheduler stopped before MCP ask became terminal: {status:#?}"
        );
    }
    assert_eq!(
        service
            .store
            .get_server_run(mcp_run_id)
            .unwrap()
            .expect("MCP ask run remains durable")
            .status,
        "succeeded"
    );
    let mcp_runtime = service
        .store
        .read_recorded_runtime_state(mcp_run_id, workflow_id, "conversation-create-contract")
        .unwrap()
        .expect("completed MCP run has durable runtime state");
    assert_eq!(mcp_runtime.state["turn_index"], 2);

    let workflow_read_token = jsonwebtoken::encode(
        &jsonwebtoken::Header::new(jsonwebtoken::Algorithm::HS256),
        &json!({
            "sub":"conversation-owner", "user_id":"conversation-owner",
            "actor_id":"conversation-owner", "role":"ro", "roles":["reader"],
            "ns":"workflow", "storage_ns":"workflow",
            "workspace":"workspace-a", "graph_space":"workflow",
            "capabilities":["workflow.run.read"],
            "exp":4102444800_u64
        }),
        &jsonwebtoken::EncodingKey::from_secret(http_secret.as_bytes()),
    )
    .unwrap();
    let conversation_run_read_token = jsonwebtoken::encode(
        &jsonwebtoken::Header::new(jsonwebtoken::Algorithm::HS256),
        &json!({
            "sub":"conversation-owner", "user_id":"conversation-owner",
            "actor_id":"conversation-owner", "role":"ro", "roles":["reader"],
            "ns":"conversation", "storage_ns":"conversation",
            "workspace":"workspace-a", "graph_space":"conversation",
            "capabilities":["workflow.run.read"],
            "exp":4102444800_u64
        }),
        &jsonwebtoken::EncodingKey::from_secret(http_secret.as_bytes()),
    )
    .unwrap();
    let run_status = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/api/workflow/runs/{run_id}"))
                .header(
                    "authorization",
                    format!("Bearer {workflow_read_token}"),
                )
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let run_status_status = run_status.status();
    let run_status_body = to_bytes(run_status.into_body(), 65_536).await.unwrap();
    assert_eq!(
        run_status_status,
        StatusCode::OK,
        "unexpected run-status response: {}",
        String::from_utf8_lossy(&run_status_body)
    );
    let run_status_body: Value = serde_json::from_slice(&run_status_body).unwrap();
    assert_eq!(run_status_body["run_id"], run_id);
    assert_eq!(run_status_body["status"], "succeeded");
    assert_eq!(
        run_status_body["user_turn_node_id"],
        send_response["user_turn_node_id"]
    );

    let run_status_contract: Value = serde_json::from_str(include_str!(
        "../../../../../contracts/golden/run-status-v1.json"
    ))
    .unwrap();
    let direct_run_status = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/api/runs/{run_id}"))
                .header(
                    "authorization",
                    format!("Bearer {conversation_run_read_token}"),
                )
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        direct_run_status.status().as_u16(),
        run_status_contract["success_status"].as_u64().unwrap() as u16
    );
    let direct_run_status_body: Value = serde_json::from_slice(
        &to_bytes(direct_run_status.into_body(), 65_536)
            .await
            .unwrap(),
    )
    .unwrap();
    for field in run_status_contract["required_fields"]
        .as_array()
        .unwrap()
    {
        let field = field.as_str().unwrap();
        assert!(direct_run_status_body.get(field).is_some(), "missing {field}");
    }
    assert_eq!(direct_run_status_body["run_id"], run_id);
    assert_eq!(direct_run_status_body["status"], "succeeded");
    let terminal_statuses = run_status_contract["terminal_statuses"]
        .as_array()
        .unwrap();
    assert!(terminal_statuses.contains(&direct_run_status_body["status"]));
    assert_eq!(direct_run_status_body["terminal"], true);
    assert_eq!(
        direct_run_status_body["user_turn_node_id"],
        send_response["user_turn_node_id"]
    );

    let transcript_snapshot = service
        .chroma_graph_snapshot(&request(), "conversation")
        .await
        .unwrap();
    assert!(
        transcript_snapshot
            .nodes()
            .iter()
            .any(|record| record.id == assistant_id),
        "assistant turn should be visible in the authorized snapshot after durable completion"
    );

    let transcript_response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/conversations/conversation-create-contract/turns")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(transcript_response.status(), StatusCode::OK);
    let transcript: Value = serde_json::from_slice(
        &to_bytes(transcript_response.into_body(), 65_536)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(transcript.as_object().unwrap().len(), 2);
    assert_eq!(
        transcript["conversation_id"],
        "conversation-create-contract"
    );
    let transcript_turns = transcript["turns"].as_array().unwrap();
    assert_eq!(
        transcript_turns
            .iter()
            .map(|turn| turn["role"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["user", "assistant", "user", "assistant"],
        "turns={transcript_turns:#?}"
    );
    assert!(
        transcript_turns.iter().all(|turn| {
            turn.get("event_type").is_none() && turn.get("workflow_node_id").is_none()
        }),
        "workflow and event records must not leak into the user transcript"
    );

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
    service
        .store
        .replace_named_projection(
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
        )
        .unwrap();
    service
        .store
        .put_workflow_design_snapshot(
            agentic_workflow_id,
            WorkflowDesignSnapshotWrite {
                version: 1,
                seq: 1,
                schema_version: 1,
                payload_json: serde_json::to_string(
                    &json!({"nodes":agentic_nodes,"edges":agentic_edges}),
                )
                .unwrap(),
            },
        )
        .unwrap();
    let agentic_turn_id = runtime.state["turn_node_id"].as_str().unwrap();
    let agentic_submit = service
        .execute(ApiEffectRequest {
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
            }))
            .unwrap(),
            principal: principal.clone(),
        })
        .await;
    assert_eq!(
        agentic_submit.status,
        StatusCode::ACCEPTED.as_u16(),
        "{}",
        String::from_utf8_lossy(&agentic_submit.body)
    );
    let agentic_run_id = "sqlite-chroma-agentic-provider-run";
    for _ in 0..48 {
        let status = service
            .store
            .get_server_run(&agentic_run_id)
            .unwrap()
            .unwrap();
        if matches!(status.status.as_str(), "succeeded" | "failed" | "cancelled") {
            break;
        }
        assert!(
            service.execute_inline_native_work_once().await.unwrap(),
            "native scheduler stopped before the provider-backed run became terminal: {status:#?}"
        );
    }
    let agentic_run = service
        .store
        .get_server_run(&agentic_run_id)
        .unwrap()
        .unwrap();
    assert_eq!(
        agentic_run.status, "succeeded",
        "provider-backed SQLite+Chroma run should complete natively"
    );
    let agentic_state = service
        .store
        .read_recorded_runtime_state(
            agentic_run_id,
            agentic_workflow_id,
            "conversation-create-contract",
        )
        .unwrap()
        .unwrap();
    assert_eq!(
        agentic_state.state["used_node_ids"],
        json!(["knowledge-node"])
    );
    assert!(
        agentic_state.state["_rt_node_executor_targets"]
            .as_object()
            .unwrap()
            .values()
            .all(|target| target == "rust-native"),
        "every agentic-answer workflow node must remain on the selected Rust implementation"
    );
    assert_eq!(
        agentic_state.state["answer"]["claims"][0]["citations"][0]["source_node_id"],
        "knowledge-node"
    );
    assert_eq!(agentic_state.state["done"], true);
    let agentic_assistant_id =
        agentic_state.state["agentic_answering_result"]["assistant_turn_node_id"]
            .as_str()
            .expect("provider-backed run records its assistant turn")
            .to_owned();
    assert!(
        service
            .store
            .latest_entity_events("conversation", "node", &[agentic_assistant_id.clone()],)
            .unwrap()
            .contains_key(&agentic_assistant_id)
    );
    let agentic_turn_id = agentic_state.state["turn_node_id"].as_str().unwrap();
    assert!(
        service
            .store
            .replay_raw_events("conversation", 0, 200)
            .unwrap()
            .iter()
            .any(|event| {
                event.entity_kind == "edge"
                    && serde_json::from_str::<Value>(&event.payload_json)
                        .ok()
                        .is_some_and(|payload| {
                            payload["relation"] == "next_turn"
                                && payload["source_ids"] == json!([agentic_turn_id])
                                && payload["target_ids"] == json!([agentic_assistant_id])
                        })
            })
    );
    let execution_lanes = service
        .store
        .list_projected_lane_messages(kogwistar_store::LaneMessageFilter {
            correlation_id: Some(agentic_run_id.to_owned()),
            limit: 100,
            ..Default::default()
        })
        .unwrap();
    assert!(!execution_lanes.is_empty());
    assert!(execution_lanes.iter().all(|lane| {
        lane.status == "completed" && lane.recipient_id != "python-worker"
    }), "Rust-native workflow must not fall back to a Python worker: {execution_lanes:#?}");
    let execution_traces = service
        .store
        .list_server_run_events(agentic_run_id, 0, 10_000)
        .unwrap()
        .into_iter()
        .filter(|event| event.event_type == "workflow.trace.v1")
        .filter_map(|event| serde_json::from_str::<Value>(&event.payload_json).ok())
        .filter(|event| event["trace"]["executor_target"] == "rust-native")
        .collect::<Vec<_>>();
    assert!(
        !execution_traces.is_empty(),
        "the Rust-native backend path must leave native execution trace evidence"
    );
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
