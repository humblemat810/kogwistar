struct InMemoryConversationEmbeddingProvider {
    profile: kogwistar_engine::embedding::EmbeddingProfile,
    store: InMemoryStore,
    concurrent_mutation: std::sync::Mutex<Option<GraphMutation>>,
}

impl kogwistar_engine::embedding::EmbeddingProvider for InMemoryConversationEmbeddingProvider {
    fn profile(&self) -> &kogwistar_engine::embedding::EmbeddingProfile {
        &self.profile
    }

    fn embed<'a>(
        &'a self,
        texts: &'a [String],
    ) -> kogwistar_engine::embedding::EmbeddingFuture<'a> {
        let mutation = self
            .concurrent_mutation
            .lock()
            .expect("test embedding mutex poisoned")
            .take();
        let store = self.store.clone();
        Box::pin(async move {
            if let Some(mutation) = mutation {
                store
                    .apply_graph_mutation(mutation)
                    .await
                    .map_err(|error| {
                        kogwistar_engine::embedding::EmbeddingError::Request(error.to_string())
                    })?;
            }
            Ok(texts.iter().map(|_| vec![0.6, 0.8]).collect())
        })
    }
}

#[tokio::test]
async fn in_memory_create_conversation_is_atomic_and_matches_shared_contract() {
    let service = InMemoryGraphApplicationService::default();
    let path = "/api/syscall/v1/create_conversation";
    assert!(service.supports_http_operation("POST", path));
    let principal = json!({
        "sub":"alice",
        "role":"rw",
        "storage_ns":"conversation",
        "ns":["conversation"],
        "graph_space":"conversation"
    });
    let request = |args: Value| ApiEffectRequest {
        contract_version: 1,
        method: "POST".to_owned(),
        path_and_query: path.to_owned(),
        body: json!({"version":"v1","args":args}).to_string().into_bytes(),
        principal: principal.clone(),
    };
    let body: Value = serde_json::from_str(include_str!(
        "../../../../../contracts/golden/conversation-create-syscall-result-v1.json"
    ))
    .unwrap();
    let args = json!({
        "user_id":"alice",
        "conversation_id":body["conversation_id"],
        "start_node_id":body["start_node_id"]
    });
    let spoofed_owner = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: path.to_owned(),
            body: json!({"version":"v1","args":{
                "user_id":"mallory",
                "conversation_id":"spoofed-conversation",
                "start_node_id":"spoofed-start"
            }})
            .to_string()
            .into_bytes(),
            principal: principal.clone(),
        })
        .await;
    assert_eq!(spoofed_owner.status, StatusCode::FORBIDDEN.as_u16());
    let response = service.execute(request(args.clone())).await;
    assert_eq!(response.status, StatusCode::OK.as_u16());
    let response: Value = serde_json::from_slice(&response.body).unwrap();
    assert_eq!(response["op"], "create_conversation");
    assert_eq!(response["result"], body);

    let scope = GraphScope {
        namespace: "conversation".to_owned(),
        workspace_id: Some(String::new()),
        graph_space: Some("conversation".to_owned()),
    };
    let mut start_filter = MetadataFilter::default();
    start_filter
        .equals
        .insert("entity_type".to_owned(), json!("conversation_start"));
    start_filter
        .equals
        .insert("conversation_id".to_owned(), body["conversation_id"].clone());
    let starts = service
        .store
        .graph_projection_records(GraphProjectionRead {
            scope: scope.clone(),
            table: "gke_nodes".to_owned(),
            ids: None,
            metadata: start_filter,
            limit: 10,
        })
        .await
        .unwrap();
    assert_eq!(starts.len(), 1);
    assert_eq!(starts[0].id, body["start_node_id"]);

    let mut acl_filter = MetadataFilter::default();
    acl_filter
        .equals
        .insert("entity_type".to_owned(), json!("acl_record"));
    acl_filter
        .equals
        .insert("acl_truth_graph".to_owned(), json!("conversation"));
    acl_filter
        .equals
        .insert("acl_target_entity_id".to_owned(), body["start_node_id"].clone());
    let acl_rows = service
        .store
        .graph_projection_records(GraphProjectionRead {
            scope: scope.clone(),
            table: "gke_nodes".to_owned(),
            ids: None,
            metadata: acl_filter,
            limit: 10,
        })
        .await
        .unwrap();
    assert!(!acl_rows.is_empty());
    assert!(acl_rows
        .iter()
        .any(|row| row.metadata.get("owner_id").and_then(Value::as_str) == Some("alice")));
    let jobs = kogwistar_store::IndexJobReadStore::index_jobs(
        &service.store,
        Some("conversation"),
        Some("PENDING"),
        Some("node"),
        Some(body["start_node_id"].as_str().unwrap()),
        Some("node_embedding"),
        10,
    )
    .await
    .unwrap();
    assert_eq!(jobs.len(), 1);

    let duplicate = service.execute(request(args)).await;
    assert_eq!(duplicate.status, StatusCode::CONFLICT.as_u16());
    assert_eq!(
        serde_json::from_slice::<Value>(&duplicate.body).unwrap()["code"],
        "KOGWISTAR_CONVERSATION_CREATE_CONFLICT"
    );
    let start_id_collision = service
        .execute(request(json!({
            "user_id":"alice",
            "conversation_id":"different-conversation",
            "start_node_id":body["start_node_id"]
        })))
        .await;
    assert_eq!(start_id_collision.status, StatusCode::CONFLICT.as_u16());
    let other_workspace_duplicate = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: path.to_owned(),
            body: json!({
                "version":"v1",
                "args":{
                    "user_id":"alice",
                    "conversation_id":body["conversation_id"],
                    "start_node_id":"other-workspace-start"
                }
            })
            .to_string()
            .into_bytes(),
            principal: json!({
                "sub":"alice",
                "role":"rw",
                "storage_ns":"conversation",
                "workspace_id":"workspace-b",
                "ns":["conversation"],
                "graph_space":"conversation"
            }),
        })
        .await;
    assert_eq!(
        other_workspace_duplicate.status,
        StatusCode::CONFLICT.as_u16()
    );

    let concurrent_args = |start_node_id: &str| {
        json!({
            "user_id":"alice",
            "conversation_id":"conversation-race",
            "start_node_id":start_node_id
        })
    };
    let (left, right) = tokio::join!(
        service.execute(request(concurrent_args("conversation-race-start-a"))),
        service.execute(request(concurrent_args("conversation-race-start-b")))
    );
    let statuses = [left.status, right.status];
    assert_eq!(statuses.iter().filter(|status| **status == 200).count(), 1);
    assert_eq!(statuses.iter().filter(|status| **status == 409).count(), 1);
    let mut race_filter = MetadataFilter::default();
    race_filter
        .equals
        .insert("entity_type".to_owned(), json!("conversation_start"));
    race_filter
        .equals
        .insert("conversation_id".to_owned(), json!("conversation-race"));
    let race_starts = service
        .store
        .graph_projection_records(GraphProjectionRead {
            scope,
            table: "gke_nodes".to_owned(),
            ids: None,
            metadata: race_filter,
            limit: 10,
        })
        .await
        .unwrap();
    assert_eq!(race_starts.len(), 1);
}

#[tokio::test]
async fn in_memory_mount_memory_matches_shared_contract_and_authority() {
    let service = InMemoryGraphApplicationService::default();
    let path = "/api/syscall/v1/mount_memory";
    assert!(service.supports_http_operation("POST", path));
    let memory_fixture: Value = serde_json::from_str(include_str!(
        "../../../../../contracts/golden/mount-memory-result-v1.json"
    ))
    .unwrap();
    let scope = GraphScope {
        namespace: "conversation".to_owned(),
        workspace_id: Some("workspace-a".to_owned()),
        graph_space: Some("conversation".to_owned()),
    };
    let metadata = json!({
        "entity_type":"context_snapshot",
        "conversation_id":memory_fixture["conversation_id"],
        "snapshot_seq":memory_fixture["properties"]["snapshot_seq"]
    });
    let acl_metadata = json!({
        "entity_type":"acl_record",
        "acl_truth_graph":"conversation",
        "acl_target_grain":"node",
        "acl_target_entity_id":memory_fixture["snapshot_node_id"],
        "acl_version":1,
        "acl_mode":"public"
    });
    let snapshot_id = memory_fixture["snapshot_node_id"].as_str().unwrap();
    GraphMutationStore::apply_graph_mutations_atomic(
        &service.store,
        vec![
            GraphMutation {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                entity_kind: "node".to_owned(),
                event_id: "event-memory-snapshot".to_owned(),
                op: "ADD".to_owned(),
                payload: json!({"id":snapshot_id,"metadata":metadata}),
                record: GraphRecord {
                    id: snapshot_id.to_owned(),
                    document: Some(memory_fixture.to_string()),
                    metadata: serde_json::from_value(metadata).unwrap(),
                    embedding: None,
                },
                embedding_dim: 0,
            },
            GraphMutation {
                scope,
                table: "gke_nodes".to_owned(),
                entity_kind: "node".to_owned(),
                event_id: "event-memory-snapshot-acl".to_owned(),
                op: "ADD".to_owned(),
                payload: json!({"id":"acl-memory-snapshot","metadata":acl_metadata}),
                record: GraphRecord {
                    id: "acl-memory-snapshot".to_owned(),
                    document: Some(acl_metadata.to_string()),
                    metadata: serde_json::from_value(acl_metadata).unwrap(),
                    embedding: None,
                },
                embedding_dim: 0,
            },
        ],
    )
    .await
    .unwrap();

    let request = |capabilities: &[&str], namespace: &str, args: Value| ApiEffectRequest {
        contract_version: 1,
        method: "POST".to_owned(),
        path_and_query: path.to_owned(),
        body: json!({"version":"v1","args":args}).to_string().into_bytes(),
        principal: json!({
            "sub":"alice",
            "role":"ro",
            "storage_ns":"conversation",
            "workspace_id":"workspace-a",
            "graph_space":"conversation",
            "ns":[namespace],
            "capabilities":capabilities
        }),
    };
    let denied = service
        .execute(request(
            &[],
            "conversation",
            json!({"conversation_id":memory_fixture["conversation_id"]}),
        ))
        .await;
    assert_eq!(denied.status, StatusCode::FORBIDDEN.as_u16());
    let denied_namespace = service
        .execute(request(
            &["project_view"],
            "docs",
            json!({"conversation_id":memory_fixture["conversation_id"]}),
        ))
        .await;
    assert_eq!(denied_namespace.status, StatusCode::FORBIDDEN.as_u16());

    let mounted = service
        .execute(request(
            &["project_view"],
            "conversation",
            json!({"conversation_id":memory_fixture["conversation_id"]}),
        ))
        .await;
    assert_eq!(mounted.status, StatusCode::OK.as_u16());
    let mounted: Value = serde_json::from_slice(&mounted.body).unwrap();
    assert_eq!(mounted["version"], "v1");
    assert_eq!(mounted["op"], "mount_memory");
    assert_eq!(mounted["status"], "ok");
    assert_eq!(mounted["result"], memory_fixture);
}

#[tokio::test]
async fn in_memory_send_message_admits_turn_and_run_atomically() {
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
    use tower::ServiceExt;

    let workflow_contract: Value = serde_json::from_str(include_str!(
        "../../../../../contracts/golden/answer-workflow-runtime-contract-v1.json"
    ))
    .unwrap();
    let workflow_id = "agentic_answering.v2";
    assert!(workflow_contract["workflows"].get(workflow_id).is_some());

    let auth_path = std::env::temp_dir().join(format!(
        "kogwistar-memory-send-auth-{}.sqlite",
        uuid::Uuid::new_v4().simple()
    ));
    let auth = SqliteAuthStore::open(&auth_path).unwrap();
    auth.resolve_external_identity(ResolveExternalIdentity {
        issuer: "test".to_owned(),
        subject: "memory-send-owner".to_owned(),
        email: "memory-send-owner@example.test".to_owned(),
        display_name: None,
        new_user_id: "memory-send-owner".to_owned(),
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
    let service = InMemoryGraphApplicationService::default()
        .with_owner_auth_config(&auth_config)
        .await
        .unwrap();
    let principal = json!({
        "user_id":"memory-send-owner", "sub":"memory-send-owner", "role":"rw",
        "ns":["conversation"], "storage_ns":"conversation",
        "workspace":"workspace-a", "graph_space":"conversation",
        "capabilities":["send_message", "write_graph", "read_graph", "spawn_process", "workflow.run.write"]
    });
    let create_conversation = |conversation_id: &str| ApiEffectRequest {
        contract_version: 1,
        method: "POST".to_owned(),
        path_and_query: "/api/syscall/v1/create_conversation".to_owned(),
        body: serde_json::to_vec(&json!({
            "version":"v1", "op":"create_conversation", "args":{
                "user_id":"memory-send-owner", "conversation_id":conversation_id
            }
        }))
        .unwrap(),
        principal: principal.clone(),
    };
    for conversation_id in ["memory-send-ok", "memory-send-race"] {
        let response = service.execute(create_conversation(conversation_id)).await;
        assert_eq!(response.status, StatusCode::OK.as_u16());
    }

    let invalid_contract: Value = serde_json::from_str(include_str!(
        "../../../../../contracts/golden/syscall-send-message-invalid-v1.json"
    ))
    .unwrap();
    for case in invalid_contract["cases"].as_array().unwrap() {
        let response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/syscall/v1/send_message".to_owned(),
                body: serde_json::to_vec(&case["request"]).unwrap(),
                principal: principal.clone(),
            })
            .await;
        assert_eq!(response.status, case["http_status"].as_u64().unwrap() as u16);
        let body: Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(body, case["response"], "{}", case["name"]);
    }
    assert!(kogwistar_store::ServerRunReadStore::server_runs(
        &service.store,
        None,
        None,
        Some("syscall-send-invalid"),
        100,
    )
    .await
    .unwrap()
    .is_empty());

    let provider = Arc::new(InMemoryConversationEmbeddingProvider {
        profile: kogwistar_engine::embedding::EmbeddingProfile {
            provider: "test".to_owned(),
            model: "in-memory-conversation-test".to_owned(),
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
        store: service.store.clone(),
        concurrent_mutation: std::sync::Mutex::new(None),
    });
    let no_embedding_run_id = conversation_send_message_id(
        "conversation.send_message.run",
        "memory-send-owner",
        "memory-send-ok",
        "no-embedding-request",
    );
    let no_embedding_turn_id = conversation_send_message_id(
        "conversation.send_message.turn",
        "memory-send-owner",
        "memory-send-ok",
        "no-embedding-request",
    );
    let no_embedding = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/syscall/v1/send_message".to_owned(),
            body: serde_json::to_vec(&json!({
                "version":"v1", "op":"send_message", "args":{
                    "conversation_id":"memory-send-ok",
                    "workflow_id":"memory-test-workflow",
                    "text":"embedding is required",
                    "idempotency_key":"no-embedding-request"
                }
            }))
            .unwrap(),
            principal: principal.clone(),
        })
        .await;
    assert_eq!(no_embedding.status, StatusCode::BAD_GATEWAY.as_u16());
    assert!(kogwistar_store::ServerRunReadStore::server_run(&service.store, &no_embedding_run_id)
        .await
        .unwrap()
        .is_none());
    assert!(service
        .store
        .graph_projection_records(GraphProjectionRead {
            scope: GraphScope {
                namespace: "conversation".to_owned(),
                workspace_id: Some("workspace-a".to_owned()),
                graph_space: Some("conversation".to_owned()),
            },
            table: "gke_nodes".to_owned(),
            ids: Some(vec![no_embedding_turn_id]),
            metadata: Default::default(),
            limit: 1,
        })
        .await
        .unwrap()
        .is_empty());
    let service = service.with_embedding_provider(provider.clone());
    let send_message = |conversation_id: &str, text: &str, key: &str| ApiEffectRequest {
        contract_version: 1,
        method: "POST".to_owned(),
        path_and_query: "/api/syscall/v1/send_message".to_owned(),
        body: serde_json::to_vec(&json!({
            "version":"v1", "op":"send_message", "args":{
                "conversation_id":conversation_id,
                "workflow_id":workflow_id,
                "text":text,
                "idempotency_key":key
            }
        }))
        .unwrap(),
        principal: principal.clone(),
    };
    assert!(service.supports_http_operation("POST", "/api/syscall/v1/send_message"));

    let secret = "in-memory-send-message-http-secret";
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let token = encode(
        &Header::new(Algorithm::HS256),
        &json!({
            "sub":"memory-send-owner", "user_id":"memory-send-owner",
            "actor_id":"memory-send-owner", "role":"rw", "roles":["rw"],
            "ns":"conversation", "storage_ns":"conversation",
            "workspace":"workspace-a", "graph_space":"conversation",
            "capabilities":["send_message", "write_graph", "read_graph", "spawn_process", "workflow.run.write"],
            "exp":now + 300
        }),
        &EncodingKey::from_secret(secret.as_bytes()),
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
                key: Some(secret.to_owned()),
                auth_db_url: Some(format!("sqlite:///{}", auth_path.display())),
                ..AuthConfig::default()
            },
        },
        Arc::new(service.clone()),
    );

    let denied_run_id = conversation_send_message_id(
        "conversation.send_message.run",
        "memory-send-owner",
        "memory-send-ok",
        "denied-request",
    );
    let denied_turn_id = conversation_send_message_id(
        "conversation.send_message.turn",
        "memory-send-owner",
        "memory-send-ok",
        "denied-request",
    );
    let mut denied_request = send_message("memory-send-ok", "must be denied", "denied-request");
    denied_request.principal["capabilities"] = json!(["read_graph", "write_graph"]);
    let denied = service.execute(denied_request).await;
    assert_eq!(denied.status, StatusCode::FORBIDDEN.as_u16());
    assert!(kogwistar_store::ServerRunReadStore::server_run(&service.store, &denied_run_id)
        .await
        .unwrap()
        .is_none());
    assert!(service.store.graph_projection_records(GraphProjectionRead {
        scope: GraphScope {
            namespace: "conversation".to_owned(),
            workspace_id: Some("workspace-a".to_owned()),
            graph_space: Some("conversation".to_owned()),
        },
        table: "gke_nodes".to_owned(),
        ids: Some(vec![denied_turn_id]),
        metadata: Default::default(),
        limit: 1,
    }).await.unwrap().is_empty());

    let first = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/syscall/v1/send_message")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::from(
                    send_message("memory-send-ok", "hello from memory", "same-request").body,
                ))
                .unwrap(),
        )
        .await;
    let first = first.unwrap();
    assert_eq!(first.status(), StatusCode::OK);
    let first: Value =
        serde_json::from_slice(&to_bytes(first.into_body(), 65_536).await.unwrap()).unwrap();
    assert_eq!(first["op"], "send_message");
    let run_id = first["result"]["run_id"].as_str().unwrap();
    let turn_id = first["result"]["user_turn_node_id"].as_str().unwrap();
    let run = kogwistar_store::ServerRunReadStore::server_run(&service.store, run_id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(run.user_turn_node_id.as_deref(), Some(turn_id));
    assert_eq!(
        kogwistar_store::ServerRunReadStore::server_run_events(&service.store, run_id, 0, 10)
            .await
            .unwrap()
            .len(),
        2
    );
    assert!(
        kogwistar_store::LaneMessageReadStore::projected_lane_message(
            &service.store,
            &format!("lane|runtime-start|{run_id}"),
        )
            .await
            .unwrap()
            .is_some()
    );
    let scope = GraphScope {
        namespace: "conversation".to_owned(),
        workspace_id: Some("workspace-a".to_owned()),
        graph_space: Some("conversation".to_owned()),
    };
    let mut turn_filter = MetadataFilter::default();
    turn_filter
        .equals
        .insert("conversation_id".to_owned(), json!("memory-send-ok"));
    let turns = service
        .store
        .graph_projection_records(GraphProjectionRead {
            scope: scope.clone(),
            table: "gke_nodes".to_owned(),
            ids: Some(vec![turn_id.to_owned()]),
            metadata: turn_filter,
            limit: 10,
        })
        .await
        .unwrap();
    assert_eq!(turns.len(), 1);
    assert_eq!(turns[0].metadata["role"], "user");
    assert_eq!(turns[0].metadata["turn_index"], 0);
    let edges = service
        .store
        .graph_projection_records(GraphProjectionRead {
            scope: scope.clone(),
            table: "gke_edges".to_owned(),
            ids: None,
            metadata: Default::default(),
            limit: 10,
        })
        .await
        .unwrap();
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].metadata["relation"], "next_turn");
    assert!(edges[0].metadata["target_ids"].as_str().unwrap().contains(turn_id));

    let replay = service
        .execute(send_message("memory-send-ok", "hello from memory", "same-request"))
        .await;
    assert_eq!(replay.status, StatusCode::OK.as_u16());
    assert_eq!(
        kogwistar_store::ServerRunReadStore::server_run_events(&service.store, run_id, 0, 10)
            .await
            .unwrap()
            .len(),
        2
    );
    let conflict = service
        .execute(send_message("memory-send-ok", "changed input", "same-request"))
        .await;
    assert_eq!(conflict.status, StatusCode::CONFLICT.as_u16());

    let race_turn_id = conversation_send_message_id(
        "conversation.send_message.turn",
        "memory-send-owner",
        "memory-send-race",
        "race-request",
    );
    let race_run_id = conversation_send_message_id(
        "conversation.send_message.run",
        "memory-send-owner",
        "memory-send-race",
        "race-request",
    );
    *provider.concurrent_mutation.lock().unwrap() = Some(GraphMutation {
        scope: scope.clone(),
        table: "gke_nodes".to_owned(),
        entity_kind: "node".to_owned(),
        event_id: "concurrent-conversation-tail-event".to_owned(),
        op: "ADD".to_owned(),
        payload: json!({"id":"concurrent-tail", "metadata":{
            "entity_type":"conversation_turn", "conversation_id":"memory-send-race",
            "turn_index":0, "role":"user", "in_conversation_chain":true
        }}),
        record: GraphRecord {
            id: "concurrent-tail".to_owned(),
            document: Some(json!({"id":"concurrent-tail", "summary":"concurrent turn"}).to_string()),
            metadata: serde_json::Map::from_iter([
                ("entity_type".to_owned(), json!("conversation_turn")),
                ("conversation_id".to_owned(), json!("memory-send-race")),
                ("turn_index".to_owned(), json!(0)),
                ("role".to_owned(), json!("user")),
                ("in_conversation_chain".to_owned(), json!(true)),
                ("lifecycle_status".to_owned(), json!("active")),
            ]),
            embedding: None,
        },
        embedding_dim: 0,
    });
    let raced = service
        .execute(send_message("memory-send-race", "must not branch", "race-request"))
        .await;
    assert_eq!(raced.status, StatusCode::CONFLICT.as_u16(), "{}", String::from_utf8_lossy(&raced.body));
    assert!(kogwistar_store::ServerRunReadStore::server_run(&service.store, &race_run_id)
        .await
        .unwrap()
        .is_none());
    assert!(service.store.graph_projection_records(GraphProjectionRead {
        scope: scope.clone(),
        table: "gke_nodes".to_owned(),
        ids: Some(vec![race_turn_id.clone()]),
        metadata: Default::default(),
        limit: 1,
    }).await.unwrap().is_empty());
    assert!(kogwistar_store::IndexJobReadStore::index_jobs(
        &service.store,
        Some("conversation"),
        Some("PENDING"),
        Some("node"),
        Some(&race_turn_id),
        Some("node_docs"),
        10,
    )
    .await
    .unwrap()
    .is_empty());
    assert!(kogwistar_store::LaneMessageReadStore::projected_lane_message(
        &service.store,
        &format!("lane|runtime-start|{race_run_id}")
    )
    .await
    .unwrap()
    .is_none());
}

#[tokio::test]
async fn in_memory_receive_message_matches_shared_python_contract() {
    let service = InMemoryGraphApplicationService::default();
    let path = "/api/syscall/v1/receive_message";
    assert!(service.supports_http_operation("POST", path));

    let scope = GraphScope {
        namespace: "scope-a".to_owned(),
        workspace_id: Some("workspace-a".to_owned()),
        graph_space: Some("conversation".to_owned()),
    };
    let mut mutations = Vec::new();
    let mut add_node = |id: &str, metadata: Value, node_scope: GraphScope| {
        mutations.push(GraphMutation {
            scope: node_scope,
            table: "gke_nodes".to_owned(),
            entity_kind: "node".to_owned(),
            event_id: format!("event-{id}"),
            op: "ADD".to_owned(),
            payload: json!({"id":id,"metadata":metadata}),
            record: GraphRecord {
                id: id.to_owned(),
                document: Some("{}".to_owned()),
                metadata: serde_json::from_value(metadata).unwrap(),
                embedding: None,
            },
            embedding_dim: 0,
        });
    };
    for (id, turn_index, role, summary) in [
        ("turn-1", 1, "user", "first"),
        ("turn-2", 2, "assistant", "second"),
    ] {
        add_node(
            id,
            json!({
                "entity_type":"conversation_turn",
                "conversation_id":"c1",
                "turn_index":turn_index,
                "role":role,
                "summary":summary
            }),
            scope.clone(),
        );
        add_node(
            &format!("acl-{id}"),
            json!({
                "entity_type":"acl_record",
                "acl_truth_graph":"conversation",
                "acl_target_grain":"node",
                "acl_target_entity_id":id,
                "acl_version":1,
                "acl_mode":"public"
            }),
            scope.clone(),
        );
    }
    add_node(
        "turn-private",
        json!({
            "entity_type":"conversation_turn",
            "conversation_id":"c1",
            "turn_index":3,
            "role":"assistant",
            "summary":"private"
        }),
        scope.clone(),
    );
    add_node(
        "acl-turn-private",
        json!({
            "entity_type":"acl_record",
            "acl_truth_graph":"conversation",
            "acl_target_grain":"node",
            "acl_target_entity_id":"turn-private",
            "acl_version":1,
            "acl_mode":"private",
            "owner_id":"someone-else"
        }),
        scope.clone(),
    );
    GraphMutationStore::apply_graph_mutations_atomic(&service.store, mutations)
        .await
        .unwrap();

    let mut out_of_scope_mutations = Vec::new();
    let mut add_out_of_scope_node = |id: &str, metadata: Value, node_scope: GraphScope| {
        out_of_scope_mutations.push(GraphMutation {
            scope: node_scope,
            table: "gke_nodes".to_owned(),
            entity_kind: "node".to_owned(),
            event_id: format!("event-{id}"),
            op: "ADD".to_owned(),
            payload: json!({"id":id,"metadata":metadata}),
            record: GraphRecord {
                id: id.to_owned(),
                document: Some("{}".to_owned()),
                metadata: serde_json::from_value(metadata).unwrap(),
                embedding: None,
            },
            embedding_dim: 0,
        });
    };
    let outside_scope = GraphScope {
        workspace_id: Some("workspace-other".to_owned()),
        ..scope.clone()
    };
    add_out_of_scope_node(
        "turn-outside-scope",
        json!({
            "entity_type":"conversation_turn",
            "conversation_id":"c1",
            "turn_index":4,
            "role":"assistant",
            "summary":"out of scope"
        }),
        outside_scope.clone(),
    );
    add_out_of_scope_node(
        "acl-turn-outside-scope",
        json!({
            "entity_type":"acl_record",
            "acl_truth_graph":"conversation",
            "acl_target_grain":"node",
            "acl_target_entity_id":"turn-outside-scope",
            "acl_version":1,
            "acl_mode":"public"
        }),
        outside_scope,
    );
    GraphMutationStore::apply_graph_mutations_atomic(&service.store, out_of_scope_mutations)
        .await
        .unwrap();

    let request = |args: Value, principal: Value| ApiEffectRequest {
        contract_version: 1,
        method: "POST".to_owned(),
        path_and_query: path.to_owned(),
        body: json!({"version":"v1","args":args}).to_string().into_bytes(),
        principal,
    };
    let principal = json!({
        "sub":"reader-1",
        "role":"ro",
        "storage_ns":"scope-a",
        "workspace_id":"workspace-a",
        "graph_space":"conversation",
        "ns":["conversation"],
        "capabilities":["read_graph"]
    });
    let denied = service
        .execute(request(
            json!({"conversation_id":"c1"}),
            json!({"ns":["conversation"],"capabilities":[]}),
        ))
        .await;
    assert_eq!(denied.status, StatusCode::FORBIDDEN.as_u16());

    let wrong_namespace = service
        .execute(request(
            json!({"conversation_id":"c1"}),
            json!({"ns":["docs"],"capabilities":["read_graph"]}),
        ))
        .await;
    assert_eq!(wrong_namespace.status, StatusCode::FORBIDDEN.as_u16());

    let missing_id = service
        .execute(request(json!({}), principal.clone()))
        .await;
    let missing_id_contract: Value = serde_json::from_str(include_str!(
        "../../../../../contracts/golden/syscall-receive-message-missing-id-v1.json"
    ))
    .unwrap();
    assert_eq!(missing_id.status, missing_id_contract["http_status"]);
    assert_eq!(
        serde_json::from_slice::<Value>(&missing_id.body).unwrap(),
        missing_id_contract["body"]
    );

    let response = service
        .execute(request(json!({"conversation_id":"c1"}), principal))
        .await;
    assert_eq!(response.status, StatusCode::OK.as_u16());
    let response: Value = serde_json::from_slice(&response.body).unwrap();
    let contract: Value = serde_json::from_str(include_str!(
        "../../../../../contracts/golden/receive-message-result-v1.json"
    ))
    .unwrap();
    assert_eq!(response["op"], "receive_message");
    assert_eq!(response["result"], contract);

    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
    use tower::ServiceExt;

    let secret = "in-memory-receive-message-http-secret";
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let token = encode(
        &Header::new(Algorithm::HS256),
        &json!({
            "sub":"reader-1", "user_id":"reader-1", "role":"ro", "roles":["ro"],
            "ns":"conversation", "storage_ns":"scope-a",
            "workspace_id":"workspace-a", "graph_space":"conversation",
            "capabilities":["read_graph"], "exp":now + 300
        }),
        &EncodingKey::from_secret(secret.as_bytes()),
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
                key: Some(secret.to_owned()),
                ..AuthConfig::default()
            },
        },
        Arc::new(service),
    );
    let request_body = json!({
        "version":"v1",
        "op":"receive_message",
        "args":{"conversation_id":"c1"}
    })
    .to_string();
    let unauthenticated = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(path)
                .header("content-type", "application/json")
                .body(Body::from(request_body.clone()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(matches!(
        unauthenticated.status(),
        StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN
    ));
    let authenticated = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(path)
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::from(request_body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(authenticated.status(), StatusCode::OK);
    let authenticated: Value = serde_json::from_slice(
        &to_bytes(authenticated.into_body(), 65_536).await.unwrap(),
    )
    .unwrap();
    assert_eq!(authenticated["result"], contract);
}

#[tokio::test]
async fn in_memory_invoke_builtin_tools_match_shared_contract_and_authority() {
    let service = InMemoryGraphApplicationService::default();
    let path = "/api/syscall/v1/invoke_tool";
    assert!(service.supports_http_operation("POST", path));

    let request = |tool_name: &str, capabilities: &[&str]| ApiEffectRequest {
        contract_version: 1,
        method: "POST".to_owned(),
        path_and_query: path.to_owned(),
        body: json!({
            "version":"v1",
            "op":"invoke_tool",
            "args":{"tool_name":tool_name,"conversation_id":"c1"}
        })
        .to_string()
        .into_bytes(),
        principal: json!({
            "sub":"alice",
            "role":"rw",
            "storage_ns":"conversation",
            "workspace_id":"workspace-a",
            "graph_space":"conversation",
            "ns":["conversation"],
            "capabilities":capabilities
        }),
    };

    let denied = service
        .execute(request("list_transcript", &["invoke_tool"]))
        .await;
    assert_eq!(denied.status, StatusCode::FORBIDDEN.as_u16());

    let denied_snapshot = service
        .execute(request("capability_snapshot", &["invoke_tool"]))
        .await;
    assert_eq!(denied_snapshot.status, StatusCode::FORBIDDEN.as_u16());

    let memory_fixture: Value = serde_json::from_str(include_str!(
        "../../../../../contracts/golden/mount-memory-result-v1.json"
    ))
    .unwrap();
    let snapshot_scope = GraphScope {
        namespace: "conversation".to_owned(),
        workspace_id: Some("workspace-a".to_owned()),
        graph_space: Some("conversation".to_owned()),
    };
    let snapshot_metadata = json!({
        "entity_type":"context_snapshot",
        "conversation_id":memory_fixture["conversation_id"],
        "snapshot_seq":memory_fixture["properties"]["snapshot_seq"]
    });
    let acl_metadata = json!({
        "entity_type":"acl_record",
        "acl_truth_graph":"conversation",
        "acl_target_grain":"node",
        "acl_target_entity_id":memory_fixture["snapshot_node_id"],
        "acl_version":1,
        "acl_mode":"public"
    });
    let snapshot_mutation = |id: &str, metadata: Value, document: String| GraphMutation {
        scope: snapshot_scope.clone(),
        table: "gke_nodes".to_owned(),
        entity_kind: "node".to_owned(),
        event_id: format!("event-{id}"),
        op: "ADD".to_owned(),
        payload: json!({"id":id,"metadata":metadata}),
        record: GraphRecord {
            id: id.to_owned(),
            document: Some(document),
            metadata: serde_json::from_value(metadata).unwrap(),
            embedding: None,
        },
        embedding_dim: 0,
    };
    GraphMutationStore::apply_graph_mutations_atomic(
        &service.store,
        vec![
            snapshot_mutation(
                memory_fixture["snapshot_node_id"].as_str().unwrap(),
                snapshot_metadata,
                memory_fixture.to_string(),
            ),
            snapshot_mutation("acl-snapshot-1", acl_metadata.clone(), acl_metadata.to_string()),
        ],
    )
    .await
    .unwrap();

    let fixture: Value = serde_json::from_str(include_str!(
        "../../../../../contracts/golden/invoke-tool-results-v1.json"
    ))
    .unwrap();
    let list = service
        .execute(request("list_transcript", &["invoke_tool", "read_graph"]))
        .await;
    assert_eq!(list.status, StatusCode::OK.as_u16());
    let list: Value = serde_json::from_slice(&list.body).unwrap();
    assert_eq!(list["op"], fixture["op"]);
    assert_eq!(list["result"]["tool_name"], "list_transcript");
    assert_eq!(list["result"]["conversation_id"], "c1");
    assert_eq!(list["result"]["messages"], json!([]));
    assert_eq!(
        list["result"].as_object().unwrap().keys().cloned().collect::<BTreeSet<_>>(),
        fixture["results"]["list_transcript"]["root_fields"]
            .as_array()
            .unwrap()
            .iter()
            .map(|field| field.as_str().unwrap().to_owned())
            .collect()
    );

    let latest_request = ApiEffectRequest {
        contract_version: 1,
        method: "POST".to_owned(),
        path_and_query: path.to_owned(),
        body: json!({
            "version":"v1",
            "args":{
                "tool_name":"latest_snapshot",
                "conversation_id":memory_fixture["conversation_id"]
            }
        })
        .to_string()
        .into_bytes(),
        principal: json!({
            "sub":"alice",
            "role":"rw",
            "storage_ns":"conversation",
            "workspace_id":"workspace-a",
            "graph_space":"conversation",
            "ns":["conversation"],
            "capabilities":["invoke_tool","project_view"]
        }),
    };
    let latest = service.execute(latest_request).await;
    assert_eq!(latest.status, StatusCode::OK.as_u16());
    let latest: Value = serde_json::from_slice(&latest.body).unwrap();
    assert_eq!(
        latest["result"],
        json!({
            "tool_name":"latest_snapshot",
            "conversation_id":memory_fixture["conversation_id"],
            "snapshot":memory_fixture
        })
    );

    for tool_name in ["capability_snapshot", "resource_snapshot"] {
        let response = service
            .execute(request(
                tool_name,
                &["invoke_tool", "project_view", "read_security_scope"],
            ))
            .await;
        assert_eq!(response.status, StatusCode::OK.as_u16(), "{tool_name}");
        let response: Value = serde_json::from_slice(&response.body).unwrap();
        let expected_fields = fixture["results"][tool_name]["root_fields"]
            .as_array()
            .unwrap()
            .iter()
            .map(|field| field.as_str().unwrap().to_owned())
            .collect::<BTreeSet<_>>();
        assert_eq!(
            response["result"]
                .as_object()
                .unwrap()
                .keys()
                .cloned()
                .collect::<BTreeSet<_>>(),
            expected_fields,
            "{tool_name} result fields"
        );
        let expected_snapshot_fields = fixture["results"][tool_name]["snapshot_fields"]
            .as_array()
            .unwrap()
            .iter()
            .map(|field| field.as_str().unwrap().to_owned())
            .collect::<BTreeSet<_>>();
        assert_eq!(
            response["result"]["snapshot"]
                .as_object()
                .unwrap()
                .keys()
                .cloned()
                .collect::<BTreeSet<_>>(),
            expected_snapshot_fields,
            "{tool_name} snapshot fields"
        );
    }

    let error_fixture: Value = serde_json::from_str(include_str!(
        "../../../../../contracts/golden/syscall-invoke-tool-errors-v1.json"
    ))
    .unwrap();
    for case in error_fixture["cases"].as_array().unwrap() {
        let response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: path.to_owned(),
                body: serde_json::to_vec(&case["request"]).unwrap(),
                principal: request("unused", &["invoke_tool"]).principal,
            })
            .await;
        assert_eq!(response.status, case["http_status"].as_u64().unwrap() as u16);
        let body: Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(body, case["response"], "{}", case["name"]);
    }
}

#[tokio::test]
async fn in_memory_invoke_tool_is_available_through_authenticated_rest() {
    use axum::body::to_bytes;
    use axum::http::Request;
    use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
    use tower::ServiceExt;

    let secret = "in-memory-invoke-tool-rest-secret";
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let bearer = |capabilities: &[&str]| {
        encode(
            &Header::new(Algorithm::HS256),
            &json!({
                "sub":"alice",
                "user_id":"alice",
                "role":"rw",
                "ns":["conversation"],
                "storage_ns":"conversation",
                "workspace_id":"workspace-a",
                "graph_space":"conversation",
                "capabilities":capabilities,
                "exp":now + 300
            }),
            &EncodingKey::from_secret(secret.as_bytes()),
        )
        .unwrap()
    };
    let app = router_with_application(
        ApiState {
            health: HealthSnapshot {
                backend: "memory".to_owned(),
                persist_directory: "memory".to_owned(),
                conversation_persist_directory: "memory".to_owned(),
                workflow_persist_directory: "memory".to_owned(),
                wisdom_persist_directory: "memory".to_owned(),
                pg_schema_base: None,
            },
            required_roles: Vec::new(),
            implementation: ImplementationSnapshot::default(),
            auth: AuthConfig {
                mode: "jwt".to_owned(),
                algorithm: Some("HS256".to_owned()),
                key: Some(secret.to_owned()),
                ..AuthConfig::default()
            },
        },
        Arc::new(InMemoryGraphApplicationService::default()),
    );

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/syscall/v1/invoke_tool")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!("Bearer {}", bearer(&["invoke_tool"])),
                )
                .body(Body::from(
                    json!({
                        "version":"v1",
                        "op":"invoke_tool",
                        "args":{"tool_name":"list_transcript","conversation_id":"c1"}
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::FORBIDDEN);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/syscall/v1/invoke_tool")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!(
                        "Bearer {}",
                        bearer(&["invoke_tool", "read_graph"])
                    ),
                )
                .body(Body::from(
                    json!({
                        "version":"v1",
                        "op":"invoke_tool",
                        "args":{"tool_name":"list_transcript","conversation_id":"c1"}
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let payload: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), 65_536).await.unwrap()).unwrap();
    assert_eq!(payload["version"], "v1");
    assert_eq!(payload["op"], "invoke_tool");
    assert_eq!(payload["result"]["tool_name"], "list_transcript");
    assert_eq!(payload["result"]["messages"], json!([]));
}
