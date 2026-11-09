#[tokio::test]
async fn in_memory_native_run_admission_is_authenticated_and_atomic() {
    let service = InMemoryGraphApplicationService::default();
    assert!(service.supports_http_operation("POST", "/api/workflow/runs"));
    assert!(service.supports_http_operation("GET", "/api/workflow/runs/memory-http-run/events"));
    let request = |principal: Value, body: Value| ApiEffectRequest {
        contract_version: 1,
        method: "POST".to_owned(),
        path_and_query: "/api/workflow/runs".to_owned(),
        body: serde_json::to_vec(&body).unwrap(),
        principal,
    };
    let body = json!({
        "run_id": "memory-http-run",
        "workflow_id": "memory-http-workflow",
        "conversation_id": "memory-http-conversation",
        "initial_state": {
            "seed": 7,
            "_rt_run_owner_id": "forged-owner",
            "_rt_graph_scope": {"storage_namespace": "forged"}
        },
        "node_ops": {"start": "noop"},
        "node_executor_targets": {"start": "rust-native"}
    });
    let unauthorized = service
        .execute(request(
            json!({"sub":"owner","role":"rw","ns":["workflow"],"capabilities":["workflow.run.read"]}),
            body.clone(),
        ))
        .await;
    assert_eq!(unauthorized.status, StatusCode::FORBIDDEN.as_u16());
    assert!(
        kogwistar_store::ServerRunReadStore::server_run(&service.store, "memory-http-run")
            .await
            .unwrap()
            .is_none()
    );

    let principal = json!({
        "sub":"owner",
        "role":"rw",
        "ns":["workflow"],
        "capabilities":["workflow.run.write", "workflow.run.read"]
    });
    let admitted = service
        .execute(request(principal.clone(), body.clone()))
        .await;
    assert_eq!(admitted.status, StatusCode::ACCEPTED.as_u16());
    let replay = service.execute(request(principal.clone(), body)).await;
    assert_eq!(replay.status, StatusCode::ACCEPTED.as_u16());
    let queued_lane = kogwistar_store::LaneMessageReadStore::projected_lane_message(
        &service.store,
        "lane|runtime-start|memory-http-run",
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(queued_lane.status, "pending");
    let run = kogwistar_store::ServerRunReadStore::server_run(&service.store, "memory-http-run")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(run.user_id.as_deref(), Some("owner"));
    assert_eq!(run.status, "running");
    assert!(service.execute_inline_native_work_once().await.unwrap());
    let completed =
        kogwistar_store::ServerRunReadStore::server_run(&service.store, "memory-http-run")
            .await
            .unwrap()
            .unwrap();
    assert_eq!(completed.status, "succeeded");
    let final_state = kogwistar_store::ProjectionReadStore::named_projection(
        &service.store,
        kogwistar_store::RUNTIME_CURRENT_STATE_NAMESPACE,
        "memory-http-run",
    )
    .await
    .unwrap()
    .unwrap()
    .payload;
    assert_eq!(final_state["state"]["_rt_run_owner_id"], "owner");
    assert_eq!(
        final_state["state"]["_rt_graph_scope"]["storage_namespace"],
        "workflow"
    );

    let mut conflicting_body = json!({
        "run_id": "memory-http-run",
        "workflow_id": "memory-http-workflow",
        "conversation_id": "memory-http-conversation",
        "initial_state": {"seed": 8},
        "node_ops": {"start": "noop"},
        "node_executor_targets": {"start": "rust-native"}
    });
    conflicting_body["initial_state"]["_rt_run_owner_id"] = json!("forged-owner");
    let conflict = service
        .execute(request(principal.clone(), conflicting_body))
        .await;
    assert_eq!(conflict.status, StatusCode::CONFLICT.as_u16());

    let owner_read = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "GET".to_owned(),
            path_and_query: "/api/workflow/runs/memory-http-run/events?after_seq=0".to_owned(),
            body: Vec::new(),
            principal: principal.clone(),
        })
        .await;
    assert_eq!(owner_read.status, StatusCode::OK.as_u16());
    let event_result: Value = serde_json::from_slice(&owner_read.body).unwrap();
    let events = event_result["events"].as_array().unwrap();
    assert!(events.len() >= 3);
    assert!(
        events
            .iter()
            .any(|event| event["event_type"] == "workflow.recorded_transition.v1")
    );

    let other_owner = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "GET".to_owned(),
            path_and_query: "/api/workflow/runs/memory-http-run".to_owned(),
            body: Vec::new(),
            principal: json!({
                "sub":"other",
                "role":"rw",
                "ns":["workflow"],
                "capabilities":["workflow.run.read"]
            }),
        })
        .await;
    assert_eq!(other_owner.status, StatusCode::FORBIDDEN.as_u16());

    let unsupported = service
        .execute(request(
            principal,
            json!({
                "run_id":"memory-http-unsupported",
                "workflow_id":"memory-http-workflow",
                "conversation_id":"memory-http-conversation",
                "node_ops":{"start":"python-only-operation"},
                "node_executor_targets":{"start":"python-worker"}
            }),
        ))
        .await;
    assert_eq!(
        unsupported.status,
        StatusCode::UNPROCESSABLE_ENTITY.as_u16()
    );
    assert!(
        kogwistar_store::ServerRunReadStore::server_run(&service.store, "memory-http-unsupported")
            .await
            .unwrap()
            .is_none()
    );
}

#[tokio::test]
async fn in_memory_native_admission_uses_authoritative_design_and_rejects_conflicts() {
    use kogwistar_store::{NamedProjectionWrite, WorkflowDesignSnapshotWrite};

    let service = InMemoryGraphApplicationService::default();
    let workflow_id = "memory-authoritative-workflow";
    kogwistar_store::ProjectionWriteStore::replace_named_projection(
        &service.store,
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
    .await
    .unwrap();
    kogwistar_store::WorkflowDesignHistoryWriteStore::put_workflow_design_snapshot(
        &service.store,
        workflow_id,
        WorkflowDesignSnapshotWrite {
            version: 1,
            seq: 1,
            schema_version: 1,
            payload_json: serde_json::to_string(&json!({
                "nodes": [{
                    "id": "start",
                    "metadata": {
                        "wf_start": true,
                        "wf_terminal": true,
                        "wf_op": "increment-counter",
                        "wf_version": "v1",
                        "wf_executor_target": "rust-native"
                    }
                }],
                "edges": []
            }))
            .unwrap(),
        },
    )
    .await
    .unwrap();

    let principal = json!({
        "sub":"design-owner",
        "role":"rw",
        "ns":["workflow"],
        "capabilities":["workflow.run.write"]
    });
    let submit = |workflow_id: &str, run_id: &str, node_ops: Value| ApiEffectRequest {
        contract_version: 1,
        method: "POST".to_owned(),
        path_and_query: "/api/workflow/runs".to_owned(),
        body: serde_json::to_vec(&json!({
            "run_id": run_id,
            "workflow_id": workflow_id,
            "conversation_id": "memory-authoritative-conversation",
            "node_ops": node_ops
        }))
        .unwrap(),
        principal: principal.clone(),
    };

    let accepted = service
        .execute(submit(workflow_id, "memory-authoritative-run", json!({})))
        .await;
    assert_eq!(accepted.status, StatusCode::ACCEPTED.as_u16());
    let lane = kogwistar_store::LaneMessageReadStore::projected_lane_message(
        &service.store,
        "lane|runtime-start|memory-authoritative-run",
    )
    .await
    .unwrap()
    .unwrap();
    let payload: Value = serde_json::from_str(lane.payload_json.as_deref().unwrap()).unwrap();
    assert_eq!(payload["op"], "increment-counter");
    assert_eq!(payload["executor_target"], "rust-native");

    let conflict = service
        .execute(submit(
            workflow_id,
            "memory-authoritative-conflict",
            json!({"start": "noop"}),
        ))
        .await;
    assert_eq!(conflict.status, StatusCode::CONFLICT.as_u16());
    assert!(
        kogwistar_store::ServerRunReadStore::server_run(
            &service.store,
            "memory-authoritative-conflict"
        )
        .await
        .unwrap()
        .is_none()
    );

    let unsupported_workflow = "memory-multistep-authoritative-workflow";
    kogwistar_store::ProjectionWriteStore::replace_named_projection(
        &service.store,
        "workflow_design",
        unsupported_workflow,
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
    kogwistar_store::WorkflowDesignHistoryWriteStore::put_workflow_design_snapshot(
        &service.store,
        unsupported_workflow,
        WorkflowDesignSnapshotWrite {
            version: 1,
            seq: 1,
            schema_version: 1,
            payload_json: serde_json::to_string(&json!({
                "nodes": [
                    {"id":"start", "metadata":{"wf_start":true, "wf_op":"noop", "wf_executor_target":"rust-native"}},
                    {"id":"finish", "metadata":{"wf_terminal":true, "wf_op":"noop", "wf_executor_target":"rust-native"}}
                ],
                "edges": [{"id":"route", "source_ids":["start"], "target_ids":["finish"], "metadata":{}}]
            }))
            .unwrap(),
        },
    )
    .await
    .unwrap();
    let unsupported = service
        .execute(submit(
            unsupported_workflow,
            "memory-multistep-run",
            json!({}),
        ))
        .await;
    assert_eq!(
        unsupported.status,
        StatusCode::UNPROCESSABLE_ENTITY.as_u16()
    );
    assert!(
        kogwistar_store::ServerRunReadStore::server_run(&service.store, "memory-multistep-run")
            .await
            .unwrap()
            .is_none()
    );
}

#[tokio::test]
async fn in_memory_run_owner_revalidation_rejects_revoked_inactive_and_changed_scope() {
    let auth_path = std::env::temp_dir().join(format!(
        "kogwistar-memory-owner-auth-{}.sqlite",
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
    drop(auth);
    let auth_config = AuthConfig {
        auth_db_url: Some(format!("sqlite:///{}", auth_path.display())),
        ..AuthConfig::default()
    };
    let provider_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let provider_address = provider_listener.local_addr().unwrap();
    let service = InMemoryGraphApplicationService::default()
        .with_owner_auth_config(&auth_config)
        .await
        .unwrap()
        .with_chat_completion_provider(
            crate::native_answer_generation::OpenAiCompatibleChatConfig::new(
                format!("http://{provider_address}/v1/chat/completions"),
                "memory-event-model",
                None,
            )
            .unwrap()
            .with_streaming(true),
        );
    let principal = json!({
        "user_id":"alice",
        "sub":"alice",
        "role":"rw",
        "ns":["workflow", "alpha"],
        "storage_ns":"alpha",
        "capabilities":["workflow.run.write", "write_graph"]
    });
    let admitted = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/workflow/runs".to_owned(),
            body: serde_json::to_vec(&json!({
                "run_id":"memory-owner-revalidation",
                "workflow_id":"memory-owner-workflow",
                "conversation_id":"memory-owner-conversation",
                "node_ops":{"start":"rust-native/agent.event.request"},
                "node_executor_targets":{"start":"rust-native"},
                "initial_state":{"provider_request":{
                    "contract_version":"v1", "model":"memory-event-model",
                    "instructions":"fixture only", "input":"not sent after revocation"
                }}
            }))
            .unwrap(),
            principal,
        })
        .await;
    assert_eq!(admitted.status, StatusCode::ACCEPTED.as_u16());
    assert!(
        service
            .resolve_inline_native_owner_principal("memory-owner-revalidation", "write_graph")
            .await
            .is_ok()
    );

    let auth_connection = rusqlite::Connection::open(&auth_path).unwrap();
    auth_connection
        .execute(
            "UPDATE users SET global_role = 'ro' WHERE user_id = 'alice'",
            [],
        )
        .unwrap();
    assert!(service.execute_inline_native_work_once().await.is_err());
    assert!(
        tokio::time::timeout(
            std::time::Duration::from_millis(100),
            provider_listener.accept()
        )
        .await
        .is_err()
    );
    let provider_messages = kogwistar_store::LaneMessageReadStore::projected_lane_messages(
        &service.store,
        LaneMessageFilter {
            inbox_id: Some("agent-provider-events".to_owned()),
            conversation_id: Some("memory-owner-conversation".to_owned()),
            limit: 10,
            ..LaneMessageFilter::default()
        },
    )
    .await
    .unwrap();
    assert!(provider_messages.is_empty());
    let run_events = kogwistar_store::ServerRunReadStore::server_run_events(
        &service.store,
        "memory-owner-revalidation",
        0,
        100,
    )
    .await
    .unwrap();
    assert!(
        !run_events
            .iter()
            .any(|event| event.event_type == "workflow.provider_event.v1")
    );
    assert!(matches!(
        service
            .resolve_inline_native_owner_principal("memory-owner-revalidation", "write_graph")
            .await,
        Err(InlineOwnerPrincipalError::Denied(_))
    ));

    auth_connection
        .execute(
            "UPDATE users SET global_role = 'rw', global_ns = 'beta' WHERE user_id = 'alice'",
            [],
        )
        .unwrap();
    assert!(matches!(
        service
            .resolve_inline_native_owner_principal("memory-owner-revalidation", "write_graph")
            .await,
        Err(InlineOwnerPrincipalError::Denied(_))
    ));

    auth_connection
        .execute(
            "UPDATE users SET global_ns = 'alpha', is_active = 0 WHERE user_id = 'alice'",
            [],
        )
        .unwrap();
    assert!(matches!(
        service
            .resolve_inline_native_owner_principal("memory-owner-revalidation", "write_graph")
            .await,
        Err(InlineOwnerPrincipalError::Denied(_))
    ));
    drop(auth_connection);
    drop(service);
    let _ = std::fs::remove_file(auth_path);
}

#[tokio::test]
async fn in_memory_native_provider_request_persists_semantic_events_and_keeps_deltas_transient() {
    use crate::native_answer_generation::OpenAiCompatibleChatConfig;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let auth_path = std::env::temp_dir().join(format!(
        "kogwistar-memory-provider-auth-{}.sqlite",
        uuid::Uuid::new_v4().simple()
    ));
    let auth = SqliteAuthStore::open(&auth_path).unwrap();
    auth.resolve_external_identity(ResolveExternalIdentity {
        issuer: "test".to_owned(),
        subject: "memory-provider-owner".to_owned(),
        email: "memory-provider-owner@example.test".to_owned(),
        display_name: None,
        new_user_id: "memory-provider-owner".to_owned(),
        default_role: "rw".to_owned(),
        default_ns: "conversation".to_owned(),
    })
    .await
    .unwrap();
    drop(auth);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let provider_task = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let mut bytes = Vec::new();
        let (header_end, content_length) = loop {
            let mut chunk = [0_u8; 4096];
            let count = stream.read(&mut chunk).await.unwrap();
            assert!(count > 0);
            bytes.extend_from_slice(&chunk[..count]);
            if let Some(index) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
                let end = index + 4;
                let headers = String::from_utf8_lossy(&bytes[..end]).to_ascii_lowercase();
                let length = headers
                    .lines()
                    .find_map(|line| line.strip_prefix("content-length:"))
                    .unwrap()
                    .trim()
                    .parse::<usize>()
                    .unwrap();
                if bytes.len() >= end + length {
                    break (end, length);
                }
            }
        };
        let request: Value =
            serde_json::from_slice(&bytes[header_end..header_end + content_length]).unwrap();
        assert_eq!(request["model"], "memory-event-model");
        assert_eq!(request["stream"], true);
        let body = concat!(
            "data: {\"id\":\"memory-response\",\"choices\":[{\"delta\":{\"content\":\"transient text\"},\"finish_reason\":null}]}\r\n\r\n",
            "data: {\"id\":\"memory-response\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\r\n\r\n",
            "data: [DONE]\r\n\r\n"
        );
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        stream.write_all(response.as_bytes()).await.unwrap();
    });
    let provider = OpenAiCompatibleChatConfig::new(
        format!("http://{address}/v1/chat/completions"),
        "memory-event-model",
        None,
    )
    .unwrap()
    .with_streaming(true);
    let auth_config = AuthConfig {
        auth_db_url: Some(format!("sqlite:///{}", auth_path.display())),
        ..AuthConfig::default()
    };
    let service = InMemoryGraphApplicationService::default()
        .with_owner_auth_config(&auth_config)
        .await
        .unwrap()
        .with_chat_completion_provider(provider);
    let principal = json!({
        "user_id":"memory-provider-owner", "sub":"memory-provider-owner", "role":"rw",
        "ns":["workflow", "conversation"], "storage_ns":"conversation",
        "workspace":"workspace-a", "graph_space":"conversation",
        "capabilities":["workflow.run.write", "workflow.run.read", "write_graph"]
    });
    let admission = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/workflow/runs".to_owned(),
            body: serde_json::to_vec(&json!({
                "run_id":"memory-provider-event-run",
                "workflow_id":"memory-provider-event-workflow",
                "conversation_id":"memory-provider-event-conversation",
                "turn_node_id":"turn-1",
            "start_node_id":"request",
                "node_ops":{"request":"rust-native/agent.event.request"},
                "initial_state":{"provider_request":{
                    "contract_version":"v1", "model":"memory-event-model",
                    "instructions":"fixture only", "input":"emit one response"
                }}
            }))
            .unwrap(),
            principal,
        })
        .await;
    assert_eq!(
        admission.status,
        StatusCode::ACCEPTED.as_u16(),
        "{}",
        String::from_utf8_lossy(&admission.body)
    );
    assert!(service.execute_inline_native_work_once().await.unwrap());
    provider_task.await.unwrap();

    let lane = kogwistar_store::LaneMessageReadStore::projected_lane_messages(
        &service.store,
        LaneMessageFilter {
            namespace: Some("conversation".to_owned()),
            inbox_id: Some("agent-provider-events".to_owned()),
            conversation_id: Some("memory-provider-event-conversation".to_owned()),
            limit: 20,
            ..LaneMessageFilter::default()
        },
    )
    .await
    .unwrap();
    assert!(lane.iter().any(|message| {
        message
            .payload_json
            .as_deref()
            .is_some_and(|payload| payload.contains("response_completed"))
    }));
    assert!(lane.iter().all(|message| {
        message
            .payload_json
            .as_deref()
            .is_some_and(|payload| !payload.contains("text_delta"))
    }));
    let events = kogwistar_store::ServerRunReadStore::server_run_events(
        &service.store,
        "memory-provider-event-run",
        0,
        100,
    )
    .await
    .unwrap();
    assert!(
        events
            .iter()
            .any(|event| event.event_type == "workflow.provider_event.v1"
                && event.payload_json.contains("response_completed"))
    );
    assert!(
        service
            .transient_run_deltas
            .last_sequence("memory-provider-event-run")
            .unwrap()
            > 0
    );
    let run = kogwistar_store::ServerRunReadStore::server_run(
        &service.store,
        "memory-provider-event-run",
    )
    .await
    .unwrap()
    .unwrap();
    let result: Value = serde_json::from_str(run.result_json.as_deref().unwrap()).unwrap();
    assert!(
        result["provider_event_request_result"]["semantic_event_count"]
            .as_i64()
            .unwrap()
            >= 2
    );
    assert!(
        result["provider_event_request_result"]["provider_sequence"]
            .as_i64()
            .unwrap()
            > 0
    );

    drop(service);
    let _ = std::fs::remove_file(auth_path);
}

#[tokio::test]
async fn in_memory_native_run_admission_is_reachable_through_authenticated_http() {
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use tower::ServiceExt;

    let secret = "in-memory-runtime-http-test-secret";
    let auth = AuthConfig {
        mode: "jwt".to_owned(),
        algorithm: Some("HS256".to_owned()),
        key: Some(secret.to_owned()),
        ..AuthConfig::default()
    };
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let token = encode(
        &Header::new(Algorithm::HS256),
        &json!({
            "sub": "runtime-http-owner",
            "role": "rw",
            "ns": ["workflow"],
            "capabilities": ["workflow.run.write", "workflow.run.read"],
            "exp": now + 300
        }),
        &EncodingKey::from_secret(secret.as_bytes()),
    )
    .unwrap();
    let app = router_with_application(
        ApiState {
            health: snapshot(),
            required_roles: Vec::new(),
            implementation: ImplementationSnapshot::default(),
            auth,
        },
        Arc::new(InMemoryGraphApplicationService::default()),
    );
    let body = serde_json::to_vec(&json!({
        "run_id": "memory-router-run",
        "workflow_id": "memory-router-workflow",
        "conversation_id": "memory-router-conversation",
        "node_ops": {"start": "noop"},
        "node_executor_targets": {"start": "rust-native"}
    }))
    .unwrap();

    let unauthenticated = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/workflow/runs")
                .header("content-type", "application/json")
                .body(Body::from(body.clone()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(matches!(
        unauthenticated.status(),
        StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN
    ));

    let admitted = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/workflow/runs")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(admitted.status(), StatusCode::ACCEPTED);
    let admitted_body = to_bytes(admitted.into_body(), 4096).await.unwrap();
    let admitted_json: Value = serde_json::from_slice(&admitted_body).unwrap();
    assert_eq!(admitted_json["run_id"], "memory-router-run");

    let events = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/workflow/runs/memory-router-run/events?after_seq=0")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(events.status(), StatusCode::OK);
    let event_body = to_bytes(events.into_body(), 8192).await.unwrap();
    let event_json: Value = serde_json::from_slice(&event_body).unwrap();
    assert!(event_json["events"].as_array().unwrap().len() >= 2);
}

#[test]
fn failed_inline_graph_effect_clears_provider_event_side_effects() {
    let mut effect: RecordedWorkerSuccessEffect = serde_json::from_value(json!({
        "contract_version": 1,
        "effect_id": "effect-1",
        "status": "success",
        "provider_event_acknowledgements": [{
            "message_id": "message-1",
            "claimed_by": "worker-1",
            "event_id": "event-1"
        }],
        "provider_event_publications": [{
            "namespace": "conversation",
            "conversation_id": "conversation-1",
            "event": {
                "event_id": "event-2",
                "trace": {
                    "run_id": "run-1",
                    "turn_id": "turn-1",
                    "workflow_node_id": "node-1",
                    "step_id": "step-1",
                    "provider_id": "provider-1"
                },
                "event": {"kind": "response_started"}
            }
        }]
    }))
    .expect("valid worker effect");

    fail_inline_graph_effect(
        &mut effect,
        "KOGWISTAR_GRAPH_AUTHORIZATION_DENIED",
        "owner authorization failed",
    );

    assert_eq!(
        effect.status,
        kogwistar_runtime::RuntimeWorkerEffectStatus::Failed
    );
    assert!(effect.provider_event_acknowledgements.is_empty());
    assert!(effect.provider_event_publications.is_empty());
    assert_eq!(effect.errors, ["owner authorization failed"]);
}

#[tokio::test]
async fn in_memory_inline_native_noop_completes_claimed_run_atomically() {
    let service = InMemoryGraphApplicationService::default();
    let noop_fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/runtime-native-noop-v1.json"
    )))
    .unwrap();
    let start: RecordedRuntimeTransition = serde_json::from_value(json!({
        "contract_version": 1,
        "transition_id": "memory-noop-start",
        "expected_event_seq": 0,
        "kind": "start",
        "run_id": "memory-noop-run",
        "workflow_id": "memory-noop-workflow",
        "conversation_id": "memory-noop-conversation",
        "user_id": "operator",
        "user_turn_node_id": "turn-1",
        "step_seq": 0,
        "initial_state": {"seed": 1},
        "frontier": {
            "pending": [["noop", 0, "token-1", null]],
            "suspended": [],
            "join_node_ids": [],
            "join_outstanding": [],
            "join_waiters": {}
        }
    }))
    .unwrap();
    service
        .store
        .apply_recorded_runtime_transition(start, false)
        .unwrap();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;
    kogwistar_store::LaneMessageWriteStore::project_lane_message(
        &service.store,
        NewProjectedLaneMessage {
            message_id: "memory-noop-lane".to_owned(),
            namespace: "workflow".to_owned(),
            purpose: "system".to_owned(),
            inbox_id: "workflow-runtime".to_owned(),
            conversation_id: "memory-noop-conversation".to_owned(),
            recipient_id: "rust-native-worker".to_owned(),
            sender_id: "rust-scheduler".to_owned(),
            msg_type: "workflow.step.execute".to_owned(),
            status: "pending".to_owned(),
            created_at: now,
            available_at: now,
            run_id: Some("memory-noop-run".to_owned()),
            step_id: Some("noop".to_owned()),
            correlation_id: Some("memory-noop-run".to_owned()),
            payload_json: Some(
                json!({
                    "contract_version": 1,
                    "kind": "workflow.step.execute",
                    "run_id": "memory-noop-run",
                    "workflow_id": "memory-noop-workflow",
                    "conversation_id": "memory-noop-conversation",
                    "node_id": "noop",
                    "op": "noop",
                    "executor_target": "rust-native",
                    "join_mask": 0,
                    "token_id": "token-1",
                    "step_seq": 0,
                    "expected_event_seq": 1,
                    "state": {"seed": 1}
                })
                .to_string(),
            ),
            error_json: None,
        },
    )
    .await
    .unwrap();

    assert!(service.execute_inline_native_work_once().await.unwrap());
    let run = kogwistar_store::ServerRunReadStore::server_run(&service.store, "memory-noop-run")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(run.status, "succeeded");
    assert_eq!(
        serde_json::from_str::<Value>(run.result_json.as_deref().unwrap()).unwrap(),
        noop_fixture["result"]
    );
    let projection = kogwistar_store::ProjectionReadStore::named_projection(
        &service.store,
        kogwistar_store::RUNTIME_CURRENT_STATE_NAMESPACE,
        "memory-noop-run",
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(projection.payload["status"], "completed");
    assert_eq!(projection.payload["state"]["seed"], 1);
    let lane = kogwistar_store::LaneMessageReadStore::projected_lane_message(
        &service.store,
        "memory-noop-lane",
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(lane.status, "completed");
    let events = kogwistar_store::ServerRunReadStore::server_run_events(
        &service.store,
        "memory-noop-run",
        0,
        10,
    )
    .await
    .unwrap();
    assert_eq!(
        events
            .iter()
            .map(|event| event.event_type.as_str())
            .collect::<Vec<_>>(),
        [
            "workflow.recorded_transition.v1",
            "workflow.recorded_transition.v1",
            "workflow.recorded_transition.v1",
            "workflow.trace.v1"
        ]
    );
    let step_effect: Value = serde_json::from_str(&events[1].payload_json).unwrap();
    assert_eq!(step_effect["reduced"]["state"]["state"]["seed"], 1);
    let trace: Value = serde_json::from_str(&events[3].payload_json).unwrap();
    assert_eq!(trace["trace"]["event"], "runtime.native.step.completed");
    assert!(!service.execute_inline_native_work_once().await.unwrap());
}

#[tokio::test]
async fn in_memory_inline_native_counter_matches_shared_effect_atomically() {
    let service = InMemoryGraphApplicationService::default();
    let counter_fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/runtime-native-increment-counter-v1.json"
    )))
    .unwrap();
    let start: RecordedRuntimeTransition = serde_json::from_value(json!({
        "contract_version": 1,
        "transition_id": "memory-counter-start",
        "expected_event_seq": 0,
        "kind": "start",
        "run_id": "memory-counter-run",
        "workflow_id": "memory-counter-workflow",
        "conversation_id": "memory-counter-conversation",
        "user_id": "operator",
        "user_turn_node_id": "turn-1",
        "step_seq": 0,
        "initial_state": {"counter": 1},
        "frontier": {
            "pending": [["increment", 0, "token-1", null]],
            "suspended": [],
            "join_node_ids": [],
            "join_outstanding": [],
            "join_waiters": {}
        }
    }))
    .unwrap();
    service
        .store
        .apply_recorded_runtime_transition(start, false)
        .unwrap();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;
    kogwistar_store::LaneMessageWriteStore::project_lane_message(
        &service.store,
        NewProjectedLaneMessage {
            message_id: "memory-counter-lane".to_owned(),
            namespace: "workflow".to_owned(),
            purpose: "system".to_owned(),
            inbox_id: "workflow-runtime".to_owned(),
            conversation_id: "memory-counter-conversation".to_owned(),
            recipient_id: "rust-native-worker".to_owned(),
            sender_id: "rust-scheduler".to_owned(),
            msg_type: "workflow.step.execute".to_owned(),
            status: "pending".to_owned(),
            created_at: now,
            available_at: now,
            run_id: Some("memory-counter-run".to_owned()),
            step_id: Some("increment".to_owned()),
            correlation_id: Some("memory-counter-run".to_owned()),
            payload_json: Some(
                json!({
                    "contract_version": 1,
                    "kind": "workflow.step.execute",
                    "run_id": "memory-counter-run",
                    "workflow_id": "memory-counter-workflow",
                    "conversation_id": "memory-counter-conversation",
                    "node_id": "increment",
                    "op": "increment-counter",
                    "executor_target": "rust-native",
                    "join_mask": 0,
                    "token_id": "token-1",
                    "step_seq": 0,
                    "expected_event_seq": 1,
                    "state": {"counter": 1}
                })
                .to_string(),
            ),
            error_json: None,
        },
    )
    .await
    .unwrap();

    assert!(service.execute_inline_native_work_once().await.unwrap());
    let run = kogwistar_store::ServerRunReadStore::server_run(&service.store, "memory-counter-run")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(run.status, "succeeded");
    assert_eq!(
        serde_json::from_str::<Value>(run.result_json.as_deref().unwrap()).unwrap(),
        counter_fixture["result"]
    );
    let projection = kogwistar_store::ProjectionReadStore::named_projection(
        &service.store,
        kogwistar_store::RUNTIME_CURRENT_STATE_NAMESPACE,
        "memory-counter-run",
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(projection.payload["status"], "completed");
    assert_eq!(projection.payload["state"]["counter"], 2);
    let lane = kogwistar_store::LaneMessageReadStore::projected_lane_message(
        &service.store,
        "memory-counter-lane",
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(lane.status, "completed");
    let events = kogwistar_store::ServerRunReadStore::server_run_events(
        &service.store,
        "memory-counter-run",
        0,
        10,
    )
    .await
    .unwrap();
    let step_effect: Value = serde_json::from_str(&events[1].payload_json).unwrap();
    assert_eq!(step_effect["reduced"]["state"]["state"]["counter"], 2);
    assert_eq!(events[3].event_type, "workflow.trace.v1");
    assert_eq!(
        serde_json::from_str::<Value>(&events[3].payload_json).unwrap()["trace"]["event"],
        "runtime.native.step.completed"
    );
    assert!(!service.execute_inline_native_work_once().await.unwrap());
}

#[tokio::test]
async fn in_memory_inline_native_result_is_persisted_and_completes_run() {
    let service = InMemoryGraphApplicationService::default();
    let result_fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/runtime-native-emit-result-v1.json"
    )))
    .unwrap();
    let start: RecordedRuntimeTransition = serde_json::from_value(json!({
        "contract_version": 1,
        "transition_id": "memory-result-start",
        "expected_event_seq": 0,
        "kind": "start",
        "run_id": "memory-result-run",
        "workflow_id": "memory-result-workflow",
        "conversation_id": "memory-result-conversation",
        "user_id": "operator",
        "user_turn_node_id": "turn-1",
        "step_seq": 0,
        "initial_state": {"seed": 5},
        "frontier": {
            "pending": [["result", 0, "token-1", null]],
            "suspended": [],
            "join_node_ids": [],
            "join_outstanding": [],
            "join_waiters": {}
        }
    }))
    .unwrap();
    service
        .store
        .apply_recorded_runtime_transition(start, false)
        .unwrap();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;
    kogwistar_store::LaneMessageWriteStore::project_lane_message(
        &service.store,
        NewProjectedLaneMessage {
            message_id: "memory-result-lane".to_owned(),
            namespace: "workflow".to_owned(),
            purpose: "system".to_owned(),
            inbox_id: "workflow-runtime".to_owned(),
            conversation_id: "memory-result-conversation".to_owned(),
            recipient_id: "rust-native-worker".to_owned(),
            sender_id: "rust-scheduler".to_owned(),
            msg_type: "workflow.step.execute".to_owned(),
            status: "pending".to_owned(),
            created_at: now,
            available_at: now,
            run_id: Some("memory-result-run".to_owned()),
            step_id: Some("result".to_owned()),
            correlation_id: Some("memory-result-run".to_owned()),
            payload_json: Some(
                json!({
                    "contract_version": 1,
                    "kind": "workflow.step.execute",
                    "run_id": "memory-result-run",
                    "workflow_id": "memory-result-workflow",
                    "conversation_id": "memory-result-conversation",
                    "node_id": "result",
                    "op": "emit-result",
                    "executor_target": "rust-native",
                    "join_mask": 0,
                    "token_id": "token-1",
                    "step_seq": 0,
                    "expected_event_seq": 1,
                    "state": {"seed": 5}
                })
                .to_string(),
            ),
            error_json: None,
        },
    )
    .await
    .unwrap();

    assert!(service.execute_inline_native_work_once().await.unwrap());
    let run = kogwistar_store::ServerRunReadStore::server_run(&service.store, "memory-result-run")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(run.status, "succeeded");
    assert_eq!(
        serde_json::from_str::<Value>(run.result_json.as_deref().unwrap()).unwrap(),
        result_fixture["result"]
    );
    let projection = kogwistar_store::ProjectionReadStore::named_projection(
        &service.store,
        kogwistar_store::RUNTIME_CURRENT_STATE_NAMESPACE,
        "memory-result-run",
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(projection.payload["status"], "completed");
    assert_eq!(projection.payload["state"]["seed"], 5);
    let lane = kogwistar_store::LaneMessageReadStore::projected_lane_message(
        &service.store,
        "memory-result-lane",
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(lane.status, "completed");
    let events = kogwistar_store::ServerRunReadStore::server_run_events(
        &service.store,
        "memory-result-run",
        0,
        10,
    )
    .await
    .unwrap();
    assert_eq!(events[3].event_type, "workflow.trace.v1");
    assert_eq!(
        serde_json::from_str::<Value>(&events[3].payload_json).unwrap()["trace"]["event"],
        "runtime.native.step.completed"
    );
    assert!(!service.execute_inline_native_work_once().await.unwrap());
}

#[tokio::test]
async fn in_memory_native_receive_claims_visible_event_and_routes_it() {
    let auth_path = std::env::temp_dir().join(format!(
        "kogwistar-memory-event-receive-auth-{}.sqlite",
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
        "user_id":"alice",
        "sub":"alice",
        "role":"rw",
        "ns":["workflow", "alpha"],
        "storage_ns":"alpha",
        "capabilities":["workflow.run.write", "read_graph", "write_graph"]
    });
    let admitted = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/workflow/runs".to_owned(),
            body: serde_json::to_vec(&json!({
                "run_id":"memory-event-receive-run",
                "workflow_id":"memory-event-receive-workflow",
                "conversation_id":"memory-event-receive-conversation",
                "node_ops":{"start":"rust-native/agent.event.receive"},
                "node_executor_targets":{"start":"rust-native"}
            }))
            .unwrap(),
            principal,
        })
        .await;
    assert_eq!(admitted.status, StatusCode::ACCEPTED.as_u16());

    let request = kogwistar_runtime::RuntimeNativeStepRequest {
        executor_target: "rust-native".to_owned(),
        operation: "agent.event.request".to_owned(),
        op_version: "v1".to_owned(),
        run_id: "memory-event-receive-run".to_owned(),
        workflow_id: "memory-event-receive-workflow".to_owned(),
        node_id: "request".to_owned(),
        conversation_id: "memory-event-receive-conversation".to_owned(),
        turn_node_id: Some("memory-event-receive-turn".to_owned()),
        token_id: "memory-event-receive-run".to_owned(),
        parent_token_id: None,
        step_seq: 0,
        state: Default::default(),
        runtime_routes: Vec::new(),
        ephemeral_context: Default::default(),
    };
    service
        .persist_in_memory_native_stream_semantic_event(
            &request,
            kogwistar_contracts::ProviderEventV1::ResponseStarted {
                contract_version: "v1".to_owned(),
                provider_event_id: Some("fixture-started".to_owned()),
                provider_sequence: Some(1),
                response_id: Some("memory-response".to_owned()),
            },
            "fixture-provider",
            "fixture-model",
            1,
        )
        .await
        .unwrap();

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;
    kogwistar_store::LaneMessageWriteStore::project_lane_message(
        &service.store,
        NewProjectedLaneMessage {
            message_id: "msg:hidden-event-without-acl".to_owned(),
            namespace: "alpha".to_owned(),
            purpose: "agent_event".to_owned(),
            inbox_id: "agent-provider-events".to_owned(),
            conversation_id: "memory-event-receive-conversation".to_owned(),
            recipient_id: "agent-event-consumer".to_owned(),
            sender_id: "agent-runtime".to_owned(),
            msg_type: "agent.provider_event".to_owned(),
            status: "pending".to_owned(),
            created_at: now,
            available_at: now,
            run_id: Some("memory-event-receive-run".to_owned()),
            step_id: Some("provider-response:hidden".to_owned()),
            correlation_id: Some("hidden-event-without-acl".to_owned()),
            payload_json: Some("{}".to_owned()),
            error_json: None,
        },
    )
    .await
    .unwrap();

    assert!(service.execute_inline_native_work_once().await.unwrap());
    let run =
        kogwistar_store::ServerRunReadStore::server_run(&service.store, "memory-event-receive-run")
            .await
            .unwrap()
            .unwrap();
    assert_eq!(run.status, "succeeded");
    let state = kogwistar_store::ProjectionReadStore::named_projection(
        &service.store,
        kogwistar_store::RUNTIME_CURRENT_STATE_NAMESPACE,
        "memory-event-receive-run",
    )
    .await
    .unwrap()
    .unwrap()
    .payload;
    assert_eq!(
        state["state"]["agent_event_current"]["event"]["kind"],
        "response_started"
    );
    assert_eq!(state["state"]["agent_event_route"], "no_action");
    let event_id = state["state"]["agent_event_current"]["event_id"]
        .as_str()
        .unwrap();
    let event_lane = kogwistar_store::LaneMessageReadStore::projected_lane_message(
        &service.store,
        &format!("msg:{event_id}"),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(event_lane.status, "claimed");
    let hidden_lane = kogwistar_store::LaneMessageReadStore::projected_lane_message(
        &service.store,
        "msg:hidden-event-without-acl",
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(hidden_lane.status, "pending");
}

#[tokio::test]
async fn in_memory_native_provider_event_ack_persists_causal_graph_atomically() {
    let auth_path = std::env::temp_dir().join(format!(
        "kogwistar-memory-event-ack-auth-{}.sqlite",
        uuid::Uuid::new_v4().simple()
    ));
    let auth = SqliteAuthStore::open(&auth_path).unwrap();
    auth.resolve_external_identity(ResolveExternalIdentity {
        issuer: "test".to_owned(),
        subject: "memory-ack-owner".to_owned(),
        email: "memory-ack-owner@example.test".to_owned(),
        display_name: None,
        new_user_id: "memory-ack-owner".to_owned(),
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
    let run_id = "memory-provider-ack-run";
    let workflow_id = "memory-provider-ack-workflow";
    let conversation_id = "memory-provider-ack-conversation";
    let principal = json!({
        "user_id": "memory-ack-owner", "sub": "memory-ack-owner", "role": "rw",
        "ns": ["workflow", "conversation"], "storage_ns": "conversation",
        "workspace": "workspace-a", "graph_space": "conversation",
        "capabilities": ["workflow.run.write", "read_graph", "write_graph"]
    });
    let binding = runtime_owner_binding_v1(&principal, &service.capabilities).unwrap();
    kogwistar_store::ServerRunWriteStore::create_server_run(
        &service.store,
        ServerRunCreate {
            run_id: run_id.to_owned(),
            conversation_id: conversation_id.to_owned(),
            workflow_id: workflow_id.to_owned(),
            user_id: Some("memory-ack-owner".to_owned()),
            user_turn_node_id: "turn-1".to_owned(),
            status: "running".to_owned(),
        },
    )
    .await
    .unwrap();
    let admission = kogwistar_store::ServerRunWriteStore::append_server_run_event(
        &service.store,
        run_id,
        "run.created",
        json!({"runtime_owner_binding": binding}).to_string(),
    )
    .await
    .unwrap();
    let event: kogwistar_contracts::DurableProviderEventV1 = serde_json::from_value(json!({
        "event_id": "memory-provider-ack-event",
        "trace": {
            "run_id": run_id, "turn_id": "turn-1", "workflow_node_id": "provider",
            "step_id": "provider-response", "producer_step_seq": 0,
            "provider_id": "fixture"
        },
        "event": {
            "kind": "response_started", "provider_event_id": "event-1",
            "provider_sequence": 1, "response_id": "response-1"
        }
    }))
    .unwrap();
    let claim = json!({
        "message_id": "msg:memory-provider-ack-event",
        "claimed_by": "memory-event-consumer#1",
        "event_id": event.event_id
    });
    let started = service.store.apply_recorded_runtime_transition(
        serde_json::from_value(json!({
            "contract_version": 1,
            "transition_id": "memory-provider-ack-start",
            "expected_event_seq": admission.seq,
            "kind": "start",
            "run_id": run_id,
            "workflow_id": workflow_id,
            "conversation_id": conversation_id,
            "step_seq": 0,
            "node_id": "start-step",
            "token_id": run_id,
            "initial_state": {
                "_rt_graph_scope": {"storage_namespace": "conversation"},
                "_rt_run_owner_id": "memory-ack-owner",
                "agent_event_current": event,
                "agent_event_claim": claim,
                "agent_event_route": "final",
                "agent_event_handling_step_seq": 0
            },
            "frontier": {
                "pending": [["handler-step", 0, run_id, null]],
                "suspended": [], "join_node_ids": [],
                "join_outstanding": [], "join_waiters": {}
            }
        })).unwrap(),
        false,
    )
    .unwrap();
    service
        .store
        .apply_recorded_runtime_transition(
            serde_json::from_value(json!({
                "contract_version": 1,
                "transition_id": "memory-provider-ack-handler",
                "expected_event_seq": started.event_seq,
                "kind": "recorded_step_success",
                "run_id": run_id,
                "workflow_id": workflow_id,
                "conversation_id": conversation_id,
                "step_seq": 0,
                "node_id": "handler-step",
                "token_id": run_id,
                "state_update": [],
                "frontier": {
                    "pending": [["ack-step", 0, run_id, null]],
                    "suspended": [], "join_node_ids": [],
                    "join_outstanding": [], "join_waiters": {}
                },
                "result": {"handled": true}
            }))
            .unwrap(),
            false,
        )
        .unwrap();
    kogwistar_store::LaneMessageWriteStore::project_lane_message(
        &service.store,
        NewProjectedLaneMessage {
            message_id: "msg:memory-provider-ack-event".to_owned(),
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
        },
    )
    .await
    .unwrap();
    kogwistar_store::LaneMessageWriteStore::claim_projected_lane_messages_filtered(
        &service.store,
        LaneMessageClaimFilter {
            namespace: "conversation".to_owned(),
            inbox_id: "agent-provider-events".to_owned(),
            claimed_by: "memory-event-consumer#1".to_owned(),
            message_ids: Some(vec!["msg:memory-provider-ack-event".to_owned()]),
            run_id: Some(run_id.to_owned()),
            msg_type: Some("agent.provider_event".to_owned()),
            recipient_id: Some("agent-event-consumer".to_owned()),
            limit: 1,
            lease_seconds: 60,
        },
    )
    .await
    .unwrap();
    kogwistar_store::LaneMessageWriteStore::project_lane_message(
        &service.store,
        NewProjectedLaneMessage {
            message_id: "memory-ack-worker-lane".to_owned(),
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
                "workflow_id": workflow_id, "conversation_id": conversation_id,
                "token_id": run_id, "parent_token_id": null
            }).to_string()),
            error_json: None,
        },
    )
    .await
    .unwrap();
    let worker_lane = kogwistar_store::LaneMessageWriteStore::claim_projected_lane_messages_filtered(
        &service.store,
        LaneMessageClaimFilter {
            namespace: "workflow".to_owned(),
            inbox_id: "workflow-runtime".to_owned(),
            claimed_by: "memory-worker#1".to_owned(),
            message_ids: Some(vec!["memory-ack-worker-lane".to_owned()]),
            run_id: Some(run_id.to_owned()),
            msg_type: Some("workflow.step.execute".to_owned()),
            recipient_id: Some("rust-native-worker".to_owned()),
            limit: 1,
            lease_seconds: 60,
        },
    )
    .await
    .unwrap()
    .pop()
    .unwrap();
    let handoff = RecordedWorkerHandoff {
        message_id: worker_lane.message_id,
        claimed_by: worker_lane.claimed_by.unwrap(),
        run_id: run_id.to_owned(),
        step_id: "ack-step".to_owned(),
        correlation_id: run_id.to_owned(),
    };
    let effect = kogwistar_runtime::execute_native_step(
        &kogwistar_runtime::RuntimeNativeStepRequest {
            executor_target: "rust-native".to_owned(),
            operation: "agent.event.ack".to_owned(),
            op_version: "v1".to_owned(),
            run_id: run_id.to_owned(),
            workflow_id: workflow_id.to_owned(),
            node_id: "ack-step".to_owned(),
            conversation_id: conversation_id.to_owned(),
            turn_node_id: Some("turn-1".to_owned()),
            token_id: run_id.to_owned(),
            parent_token_id: None,
            step_seq: 1,
            state: serde_json::from_value(json!({
                "agent_event_current": event,
                "agent_event_claim": claim,
                "agent_event_route": "final",
                "agent_event_handling_step_seq": 0
            })).unwrap(),
            runtime_routes: Vec::new(),
            ephemeral_context: Default::default(),
        },
    ).unwrap();
    let batch = service
        .in_memory_provider_event_ack_graph_batch(&handoff, &effect)
        .await
        .unwrap();
    service.store
        .apply_claimed_recorded_worker_effect_with_graph(handoff, effect, batch)
        .unwrap();

    let graph_scope = GraphScope {
        namespace: "conversation".to_owned(),
        workspace_id: Some("workspace-a".to_owned()),
        graph_space: Some("conversation".to_owned()),
    };
    let graph = kogwistar_store::GraphMutationStore::graph_projection_records(
        &service.store,
        GraphProjectionRead {
            scope: graph_scope.clone(),
            table: "gke_edges".to_owned(),
            ids: None,
            metadata: Default::default(),
            limit: 100,
        },
    ).await.unwrap();
    assert!(graph.iter().any(|edge| {
        edge.metadata["relation"] == "acknowledged_by_step"
            && edge.metadata["in_conversation_chain"] == false
    }));
    assert!(graph.iter().any(|edge| {
        edge.metadata["relation"] == "handled_by_step"
            && edge.metadata["in_conversation_chain"] == false
    }));
    let nodes = kogwistar_store::GraphMutationStore::graph_projection_records(
        &service.store,
        GraphProjectionRead {
            scope: graph_scope,
            table: "gke_nodes".to_owned(),
            ids: None,
            metadata: Default::default(),
            limit: 100,
        },
    )
    .await
    .unwrap();
    assert!(nodes.iter().any(|node| {
        node.id == format!("wf_step|{run_id}|1")
            && node.metadata["entity_type"] == "workflow_step_exec"
            && node.metadata["in_conversation_chain"] == false
    }));
    assert!(nodes.iter().any(|node| {
        node.metadata["entity_type"] == "acl_record"
            && node.metadata["acl_target_entity_id"] == format!("wf_step|{run_id}|1")
    }));
    assert!(kogwistar_store::IndexJobReadStore::index_jobs(
        &service.store,
        Some("conversation"),
        Some("PENDING"),
        None,
        None,
        None,
        100,
    )
    .await
    .unwrap()
    .iter()
    .any(|job| job.entity_kind == "node"));
    assert!(kogwistar_store::ServerRunReadStore::server_run_events(
        &service.store, run_id, 0, 100
    ).await.unwrap().iter().any(|event| event.event_type == "workflow.provider_event.handled.v1"));
    drop(service);
    remove_test_sqlite(&auth_path);
}

#[tokio::test]
async fn in_memory_inline_native_worker_dead_letters_unsupported_operation() {
    let service = InMemoryGraphApplicationService::default();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;
    kogwistar_store::LaneMessageWriteStore::project_lane_message(
        &service.store,
        NewProjectedLaneMessage {
            message_id: "memory-unsupported-lane".to_owned(),
            namespace: "workflow".to_owned(),
            purpose: "system".to_owned(),
            inbox_id: "workflow-runtime".to_owned(),
            conversation_id: "memory-unsupported-conversation".to_owned(),
            recipient_id: "rust-native-worker".to_owned(),
            sender_id: "rust-scheduler".to_owned(),
            msg_type: "workflow.step.execute".to_owned(),
            status: "pending".to_owned(),
            created_at: now,
            available_at: now,
            run_id: Some("memory-unsupported-run".to_owned()),
            step_id: Some("unsupported".to_owned()),
            correlation_id: Some("memory-unsupported-run".to_owned()),
            payload_json: Some(
                json!({
                    "contract_version": 1,
                    "kind": "workflow.step.execute",
                    "run_id": "memory-unsupported-run",
                    "workflow_id": "memory-unsupported-workflow",
                    "conversation_id": "memory-unsupported-conversation",
                    "node_id": "unsupported",
                    "op": "unregistered-operation",
                    "executor_target": "rust-native",
                    "token_id": "token-1"
                })
                .to_string(),
            ),
            error_json: None,
        },
    )
    .await
    .unwrap();

    assert!(service.execute_inline_native_work_once().await.unwrap());
    let lane = kogwistar_store::LaneMessageReadStore::projected_lane_message(
        &service.store,
        "memory-unsupported-lane",
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(lane.status, "dead-letter");
    assert!(!service.execute_inline_native_work_once().await.unwrap());
}

#[tokio::test]
async fn in_memory_semantic_seed_expand_is_available_through_authenticated_http_mcp() {
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use jsonwebtoken::{EncodingKey, Header, encode};
    use tower::ServiceExt;

    let fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/semantic-seed-expand-v1.json"
    )))
    .unwrap();
    assert_eq!(fixture["schema_version"], 1);
    let seed_id = fixture["expected_seed_ids"][0].as_str().unwrap();
    let edge_id = fixture["expected_first_layer_edge_ids"][0]
        .as_str()
        .unwrap();
    let cross_doc_edge_id = fixture["expected_first_layer_edge_ids"][1]
        .as_str()
        .unwrap();
    let doc_id = fixture["scope"]["doc_id"].as_str().unwrap();
    assert_eq!(fixture["expansion_document_filter"], "not_applied");

    struct FixedEmbeddingProvider;
    impl EmbeddingProvider for FixedEmbeddingProvider {
        fn profile(&self) -> &kogwistar_engine::embedding::EmbeddingProfile {
            static PROFILE: std::sync::OnceLock<kogwistar_engine::embedding::EmbeddingProfile> =
                std::sync::OnceLock::new();
            PROFILE.get_or_init(|| kogwistar_engine::embedding::EmbeddingProfile {
                provider: "test".to_owned(),
                model: "in-memory-semantic-test".to_owned(),
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
            })
        }

        fn embed<'a>(
            &'a self,
            texts: &'a [String],
        ) -> kogwistar_engine::embedding::EmbeddingFuture<'a> {
            Box::pin(async move { Ok(texts.iter().map(|_| vec![1.0, 0.0]).collect()) })
        }
    }

    let scope = GraphScope {
        namespace: "tenant-a".to_owned(),
        workspace_id: Some("workspace-a".to_owned()),
        graph_space: Some("facts".to_owned()),
    };
    let inactive = InMemoryGraphApplicationService::default();
    assert!(!inactive.supports_mcp_tool("root", "kg_semantic_seed_then_expand_text"));
    assert!(!inactive.supports_http_operation("POST", GRAPH_SEMANTIC_SEED_EXPAND_ROUTE));

    let service = InMemoryGraphApplicationService::default()
        .with_embedding_provider(Arc::new(FixedEmbeddingProvider));
    assert!(service.supports_mcp_tool("root", "kg_semantic_seed_then_expand_text"));
    assert!(service.supports_http_operation("POST", GRAPH_SEMANTIC_SEED_EXPAND_ROUTE));
    service
        .store
        .apply_graph_mutations_atomic(vec![
            GraphMutation {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                entity_kind: "node".to_owned(),
                event_id: "semantic-seed-node-event".to_owned(),
                op: "ADD".to_owned(),
                payload: json!({"id":seed_id,"label":"Smoking"}),
                record: GraphRecord {
                    id: seed_id.to_owned(),
                    document: Some(json!({"id":seed_id,"label":"Smoking"}).to_string()),
                    metadata: serde_json::Map::from_iter([
                        ("namespace".to_owned(), json!("tenant-a")),
                        ("workspace_id".to_owned(), json!("workspace-a")),
                        ("graph_space".to_owned(), json!("facts")),
                        ("doc_id".to_owned(), json!(doc_id)),
                    ]),
                    embedding: Some(vec![1.0, 0.0]),
                },
                embedding_dim: 2,
            },
            GraphMutation {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                entity_kind: "node".to_owned(),
                event_id: "semantic-neighbor-event".to_owned(),
                op: "ADD".to_owned(),
                payload: json!({"id":"B","label":"Lung Cancer"}),
                record: GraphRecord {
                    id: "B".to_owned(),
                    document: Some(r#"{"id":"B","label":"Lung Cancer"}"#.to_owned()),
                    metadata: serde_json::Map::from_iter([
                        ("namespace".to_owned(), json!("tenant-a")),
                        ("workspace_id".to_owned(), json!("workspace-a")),
                        ("graph_space".to_owned(), json!("facts")),
                        ("doc_id".to_owned(), json!(doc_id)),
                    ]),
                    embedding: Some(vec![0.0, 1.0]),
                },
                embedding_dim: 2,
            },
            GraphMutation {
                scope: scope.clone(),
                table: "gke_edges".to_owned(),
                entity_kind: "edge".to_owned(),
                event_id: "semantic-edge-event".to_owned(),
                op: "ADD".to_owned(),
                payload: json!({"id":edge_id,"source_ids":[seed_id],"target_ids":["B"]}),
                record: GraphRecord {
                    id: edge_id.to_owned(),
                    document: Some(
                        json!({
                            "id":edge_id,
                            "relation":"causes",
                            "source_ids":[seed_id],
                            "target_ids":["B"]
                        })
                        .to_string(),
                    ),
                    metadata: serde_json::Map::from_iter([
                        ("namespace".to_owned(), json!("tenant-a")),
                        ("workspace_id".to_owned(), json!("workspace-a")),
                        ("graph_space".to_owned(), json!("facts")),
                        ("doc_id".to_owned(), json!(doc_id)),
                        ("source_ids".to_owned(), json!([seed_id])),
                        ("target_ids".to_owned(), json!(["B"])),
                        ("relation".to_owned(), json!("causes")),
                    ]),
                    embedding: None,
                },
                embedding_dim: 0,
            },
            GraphMutation {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                entity_kind: "node".to_owned(),
                event_id: "semantic-seed-acl-event".to_owned(),
                op: "ADD".to_owned(),
                payload: json!({"id":"semantic-seed-node-acl"}),
                record: GraphRecord {
                    id: "semantic-seed-node-acl".to_owned(),
                    document: None,
                    metadata: serde_json::Map::from_iter([
                        ("namespace".to_owned(), json!("tenant-a")),
                        ("workspace_id".to_owned(), json!("workspace-a")),
                        ("graph_space".to_owned(), json!("facts")),
                        ("entity_type".to_owned(), json!("acl_record")),
                        ("acl_truth_graph".to_owned(), json!("knowledge")),
                        ("acl_target_grain".to_owned(), json!("node")),
                        ("acl_target_entity_id".to_owned(), json!(seed_id)),
                        ("acl_version".to_owned(), json!(1)),
                        ("acl_mode".to_owned(), json!("public")),
                    ]),
                    embedding: None,
                },
                embedding_dim: 0,
            },
            GraphMutation {
                scope: scope.clone(),
                table: "gke_edges".to_owned(),
                entity_kind: "edge".to_owned(),
                event_id: "semantic-cross-doc-edge-event".to_owned(),
                op: "ADD".to_owned(),
                payload: json!({
                    "id":cross_doc_edge_id,
                    "source_ids":[seed_id],
                    "target_ids":["B"]
                }),
                record: GraphRecord {
                    id: cross_doc_edge_id.to_owned(),
                    document: Some(
                        json!({
                            "id":cross_doc_edge_id,
                            "relation":"cross_doc",
                            "source_ids":[seed_id],
                            "target_ids":["B"],
                            "doc_id":"D2"
                        })
                        .to_string(),
                    ),
                    metadata: serde_json::Map::from_iter([
                        ("namespace".to_owned(), json!("tenant-a")),
                        ("workspace_id".to_owned(), json!("workspace-a")),
                        ("graph_space".to_owned(), json!("facts")),
                        ("doc_id".to_owned(), json!("D2")),
                        ("source_ids".to_owned(), json!([seed_id])),
                        ("target_ids".to_owned(), json!(["B"])),
                        ("relation".to_owned(), json!("cross_doc")),
                    ]),
                    embedding: None,
                },
                embedding_dim: 0,
            },
            GraphMutation {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                entity_kind: "node".to_owned(),
                event_id: "semantic-cross-edge-acl-event".to_owned(),
                op: "ADD".to_owned(),
                payload: json!({"id":"semantic-cross-edge-acl"}),
                record: GraphRecord {
                    id: "semantic-cross-edge-acl".to_owned(),
                    document: None,
                    metadata: serde_json::Map::from_iter([
                        ("namespace".to_owned(), json!("tenant-a")),
                        ("workspace_id".to_owned(), json!("workspace-a")),
                        ("graph_space".to_owned(), json!("facts")),
                        ("entity_type".to_owned(), json!("acl_record")),
                        ("acl_truth_graph".to_owned(), json!("knowledge")),
                        ("acl_target_grain".to_owned(), json!("edge")),
                        ("acl_target_entity_id".to_owned(), json!(cross_doc_edge_id)),
                        ("acl_version".to_owned(), json!(1)),
                        ("acl_mode".to_owned(), json!("public")),
                    ]),
                    embedding: None,
                },
                embedding_dim: 0,
            },
            GraphMutation {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                entity_kind: "node".to_owned(),
                event_id: "semantic-neighbor-acl-event".to_owned(),
                op: "ADD".to_owned(),
                payload: json!({"id":"semantic-neighbor-acl"}),
                record: GraphRecord {
                    id: "semantic-neighbor-acl".to_owned(),
                    document: None,
                    metadata: serde_json::Map::from_iter([
                        ("namespace".to_owned(), json!("tenant-a")),
                        ("workspace_id".to_owned(), json!("workspace-a")),
                        ("graph_space".to_owned(), json!("facts")),
                        ("entity_type".to_owned(), json!("acl_record")),
                        ("acl_truth_graph".to_owned(), json!("knowledge")),
                        ("acl_target_grain".to_owned(), json!("node")),
                        ("acl_target_entity_id".to_owned(), json!("B")),
                        ("acl_version".to_owned(), json!(1)),
                        ("acl_mode".to_owned(), json!("public")),
                    ]),
                    embedding: None,
                },
                embedding_dim: 0,
            },
            GraphMutation {
                scope,
                table: "gke_nodes".to_owned(),
                entity_kind: "node".to_owned(),
                event_id: "semantic-edge-acl-event".to_owned(),
                op: "ADD".to_owned(),
                payload: json!({"id":"semantic-edge-acl"}),
                record: GraphRecord {
                    id: "semantic-edge-acl".to_owned(),
                    document: None,
                    metadata: serde_json::Map::from_iter([
                        ("namespace".to_owned(), json!("tenant-a")),
                        ("workspace_id".to_owned(), json!("workspace-a")),
                        ("graph_space".to_owned(), json!("facts")),
                        ("entity_type".to_owned(), json!("acl_record")),
                        ("acl_truth_graph".to_owned(), json!("knowledge")),
                        ("acl_target_grain".to_owned(), json!("edge")),
                        ("acl_target_entity_id".to_owned(), json!(edge_id)),
                        ("acl_version".to_owned(), json!(1)),
                        ("acl_mode".to_owned(), json!("public")),
                    ]),
                    embedding: None,
                },
                embedding_dim: 0,
            },
        ])
        .await
        .unwrap();

    let secret = "in-memory-semantic-mcp-test-secret";
    let auth = AuthConfig {
        mode: "jwt".to_owned(),
        algorithm: Some("HS256".to_owned()),
        key: Some(secret.to_owned()),
        ..AuthConfig::default()
    };
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let token = encode(
        &Header::new(Algorithm::HS256),
        &json!({
            "sub":"reader",
            "role":"rw",
            "ns":["docs"],
            "storage_ns":"tenant-a",
            "workspace_id":"workspace-a",
            "graph_space":"facts",
            "capabilities":["read_graph"],
            "exp":now + 300
        }),
        &EncodingKey::from_secret(secret.as_bytes()),
    )
    .unwrap();
    let app = router_with_application(
        ApiState {
            health: snapshot(),
            required_roles: vec![],
            implementation: ImplementationSnapshot::default(),
            auth,
        },
        Arc::new(service),
    );
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::from(
                    json!({
                        "jsonrpc":"2.0",
                        "id":"in-memory-semantic",
                        "method":"tools/call",
                        "params":{
                            "name":"kg_semantic_seed_then_expand_text",
                            "arguments":{
                                "text":fixture["query_text"],
                                "top_k":fixture["top_k"],
                                "hops":fixture["hops"],
                                "doc_ids":doc_id
                            }
                        }
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let response: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), 65_536).await.unwrap()).unwrap();
    assert!(response["result"]["seeds"].is_array(), "{response}");
    let seed: Value =
        serde_json::from_str(response["result"]["seeds"][0].as_str().unwrap()).unwrap();
    assert_eq!(seed["id"], seed_id);
    let first_layer_edge_ids = response["result"]["layers"][0]["edges"]
        .as_array()
        .unwrap()
        .iter()
        .map(|edge| {
            serde_json::from_str::<Value>(edge.as_str().unwrap()).unwrap()["id"]
                .as_str()
                .unwrap()
                .to_owned()
        })
        .collect::<Vec<_>>();
    assert_eq!(
        json!(first_layer_edge_ids),
        fixture["expected_first_layer_edge_ids"],
        "response={response}"
    );
}

// PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/tests/part_08.rs#in_memory_search_index_hybrid_is_scoped_and_acl_filtered; python=kogwistar/server_mcp_with_admin.py#search_index_hybrid; fixture=contracts/golden/search-index-hybrid-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/part_08.rs#in_memory_search_index_hybrid_is_scoped_and_acl_filtered; python_test=tests/core/test_search_index_contract.py#test_hybrid_search_fusion_matches_shared_rust_golden_fixture
#[tokio::test]
async fn in_memory_search_index_hybrid_is_scoped_and_acl_filtered() {
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
    use kogwistar_store::{
        EventReadStore, GraphMutation, GraphMutationStore, GraphRecord, NodeIndexStore, VectorMatch,
    };
    use tower::ServiceExt;

    struct FixedEmbeddingProvider;
    impl EmbeddingProvider for FixedEmbeddingProvider {
        fn profile(&self) -> &kogwistar_engine::embedding::EmbeddingProfile {
            static PROFILE: std::sync::OnceLock<kogwistar_engine::embedding::EmbeddingProfile> =
                std::sync::OnceLock::new();
            PROFILE.get_or_init(|| kogwistar_engine::embedding::EmbeddingProfile {
                provider: "test".to_owned(),
                model: "in-memory-search-index-v1".to_owned(),
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
            })
        }

        fn embed<'a>(
            &'a self,
            texts: &'a [String],
        ) -> kogwistar_engine::embedding::EmbeddingFuture<'a> {
            Box::pin(async move { Ok(texts.iter().map(|_| vec![0.6, 0.8]).collect()) })
        }
    }

    let fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/search-index-entry-v1.json"
    )))
    .unwrap();
    let mut request: kogwistar_contracts::SearchIndexUpsertV1 =
        serde_json::from_value(fixture["request"].clone()).unwrap();
    request.index.truncate(1);
    request.index[0].node_id = "in-memory-index-node".to_owned();
    request.index[0].canonical_title = "In-Memory Search Permit".to_owned();
    request.index[0].keywords = vec!["inmemorypermit".to_owned()];
    let mut private_entry = request.index[0].clone();
    private_entry.node_id = "in-memory-private-index-node".to_owned();
    private_entry.canonical_title = "Private In-Memory Search Permit".to_owned();
    request.index.push(private_entry);

    let hybrid_fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/search-index-hybrid-v1.json"
    )))
    .unwrap();
    let lexical = hybrid_fixture["lexical_hits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|hit| crate::search_index_fts::LexicalIndexHit {
            index_key: hit["index_key"].as_str().unwrap().to_owned(),
            node_id: hit["node_id"].as_str().unwrap().to_owned(),
            canonical_title: hit["canonical_title"].as_str().unwrap().to_owned(),
            provision: hit["provision"].as_str().unwrap().to_owned(),
            document_id: hit["document_id"].as_str().map(str::to_owned),
            bm25_score: hit["bm25_score"].as_f64().unwrap(),
        })
        .collect::<Vec<_>>();
    let vectors = hybrid_fixture["vector_hits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|hit| VectorMatch {
            record: GraphRecord {
                id: hit["id"].as_str().unwrap().to_owned(),
                document: None,
                metadata: serde_json::from_value(hit["metadata"].clone()).unwrap(),
                embedding: None,
            },
            distance: hit["distance"].as_f64().unwrap(),
        })
        .collect::<Vec<_>>();
    let fused = crate::search_index_fts::fuse_hybrid_hits(&lexical, &vectors);
    let expected_fused = hybrid_fixture["expected"].as_array().unwrap();
    assert_eq!(fused.len(), expected_fused.len());
    for (actual, expected) in fused.iter().zip(expected_fused) {
        assert_eq!(actual.index_key, expected["index_key"]);
        assert_eq!(actual.node_id, expected["node_id"]);
        assert_eq!(
            actual.hybrid_score,
            expected["hybrid_score"].as_f64().unwrap()
        );
    }

    let unconfigured = InMemoryGraphApplicationService::default();
    assert!(!unconfigured.supports_http_operation("POST", "/api/add_index_entries"));
    let unavailable = unconfigured
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/add_index_entries".to_owned(),
            body: serde_json::to_vec(&request).unwrap(),
            principal: json!({"sub":"index-writer","role":"rw","capabilities":["write_graph"]}),
        })
        .await;
    assert_eq!(unavailable.status, StatusCode::NOT_IMPLEMENTED.as_u16());
    assert!(
        EventReadStore::replay_events(&unconfigured.store, "docs", 0, 10)
            .await
            .unwrap()
            .is_empty()
    );
    assert!(!unconfigured.supports_http_operation("GET", "/api/search_index_hybrid"));
    let unavailable_query = unconfigured
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "GET".to_owned(),
            path_and_query: "/api/search_index_hybrid?q=permit".to_owned(),
            body: Vec::new(),
            principal: json!({
                "sub":"reader",
                "role":"ro",
                "ns":["docs"],
                "storage_ns":"docs",
                "capabilities":["read_graph"]
            }),
        })
        .await;
    assert_eq!(
        unavailable_query.status,
        StatusCode::NOT_IMPLEMENTED.as_u16()
    );

    let suffix = uuid::Uuid::new_v4().simple().to_string();
    let index_path = std::env::temp_dir().join(format!("search-index-memory-{suffix}.sqlite3"));
    let service = InMemoryGraphApplicationService::default()
        .with_embedding_provider(Arc::new(FixedEmbeddingProvider))
        .with_search_index_path(&index_path)
        .unwrap();
    let scope = GraphScope {
        namespace: "docs".to_owned(),
        workspace_id: None,
        graph_space: None,
    };
    let node_mutation = |id: &str, metadata: Value, document: Option<String>| GraphMutation {
        scope: scope.clone(),
        table: "gke_nodes".to_owned(),
        entity_kind: "node".to_owned(),
        event_id: format!("event-{id}"),
        op: "ADD".to_owned(),
        payload: json!({"id":id,"metadata":metadata}),
        record: GraphRecord {
            id: id.to_owned(),
            document,
            metadata: serde_json::from_value(metadata).unwrap(),
            embedding: None,
        },
        embedding_dim: 0,
    };
    let mut mutations = Vec::new();
    for (id, mode, owner) in [
        ("in-memory-index-node", "public", None),
        (
            "in-memory-private-index-node",
            "private",
            Some("other-user"),
        ),
    ] {
        mutations.push(node_mutation(
            id,
            json!({
                "entity_type":"node",
                "label":id,
                "lifecycle_status":"active"
            }),
            Some(format!("Canonical source for {id}")),
        ));
        let mut acl = json!({
            "entity_type":"acl_record",
            "acl_truth_graph":"knowledge",
            "acl_target_grain":"node",
            "acl_target_entity_id":id,
            "acl_version":1,
            "acl_mode":mode
        });
        if let Some(owner) = owner {
            acl["acl_owner_id"] = json!(owner);
        }
        mutations.push(node_mutation(&format!("acl-{id}"), acl, None));
    }
    GraphMutationStore::apply_graph_mutations_atomic(&service.store, mutations)
        .await
        .unwrap();
    assert!(service.supports_http_operation("POST", "/api/add_index_entries"));
    assert!(service.supports_http_operation("GET", "/api/search_index_hybrid"));

    let secret = "in-memory-search-index-test-secret";
    let auth = AuthConfig {
        mode: "jwt".to_owned(),
        algorithm: Some("HS256".to_owned()),
        key: Some(secret.to_owned()),
        ..AuthConfig::default()
    };
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let token = |role: &str, capabilities: &[&str]| {
        encode(
            &Header::new(Algorithm::HS256),
            &json!({
                "sub":"index-writer",
                "role":role,
                "ns":["docs"],
                "storage_ns":"docs",
                "capabilities":capabilities,
                "exp":now + 300
            }),
            &EncodingKey::from_secret(secret.as_bytes()),
        )
        .unwrap()
    };
    let app = router_with_application(
        ApiState {
            health: snapshot(),
            required_roles: Vec::new(),
            implementation: ImplementationSnapshot::default(),
            auth,
        },
        Arc::new(service.clone()),
    );
    let body = serde_json::to_vec(&request).unwrap();

    for (role, capabilities) in [("rw", &[][..]), ("ro", &["write_graph"][..])] {
        let denied = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/add_index_entries")
                    .header(
                        "authorization",
                        format!("Bearer {}", token(role, capabilities)),
                    )
                    .header("content-type", "application/json")
                    .body(Body::from(body.clone()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(denied.status(), StatusCode::FORBIDDEN);
    }

    for _ in 0..2 {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/add_index_entries")
                    .header(
                        "authorization",
                        format!("Bearer {}", token("rw", &["write_graph"])),
                    )
                    .header("content-type", "application/json")
                    .body(Body::from(body.clone()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            serde_json::from_slice::<Value>(
                &to_bytes(response.into_body(), 65_536).await.unwrap(),
            )
            .unwrap(),
            json!({"ok":true})
        );
    }

    let events = EventReadStore::replay_events(&service.store, "docs", 0, 10)
        .await
        .unwrap();
    assert_eq!(
        events
            .iter()
            .filter(|event| event.entity_kind == "search_index" && event.op == "search_index.upsert")
            .count(),
        2,
        "an HTTP retry must not duplicate search-index events"
    );
    let fts = service.search_index_fts.as_ref().unwrap().lock().unwrap();
    let lexical_hits = crate::search_index_fts::search_fts(&fts, "permit", 10).unwrap();
    assert_eq!(
        lexical_hits.len(),
        2,
        "both entries should enter the FTS projection"
    );
    drop(fts);
    let key = kogwistar_contracts::make_search_index_key(
        &request.index[0].node_id,
        &request.index[0].canonical_title,
        &request.index[0].provision,
    );
    let vector =
        NodeIndexStore::get_node_index(&service.store, "nodes_index", &[format!("idx:{key}")])
            .await
            .unwrap();
    assert_eq!(vector.len(), 1);
    assert_eq!(
        vector[0].metadata["target_node_id"],
        request.index[0].node_id
    );

    let search = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/search_index_hybrid?q=permit&limit=10&resolve_node=true")
                .header(
                    "authorization",
                    format!("Bearer {}", token("rw", &["read_graph"])),
                )
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(search.status(), StatusCode::OK);
    let search_payload: Value =
        serde_json::from_slice(&to_bytes(search.into_body(), 65_536).await.unwrap()).unwrap();
    let results = search_payload["results"].as_array().unwrap();
    assert_eq!(
        results.len(),
        1,
        "private indexed nodes must be ACL-filtered: {search_payload}"
    );
    assert_eq!(results[0]["node_id"], "in-memory-index-node");
    assert_eq!(
        results[0]["documents"],
        "Canonical source for in-memory-index-node"
    );

    drop(service);
    let _ = std::fs::remove_file(index_path);
}

#[test]
fn inline_graph_mutation_scope_uses_current_owner_scope() {
    let principal = json!({
        "ns": ["docs"],
        "storage_ns": "docs",
        "workspace": "workspace-a",
        "graph_space": "knowledge"
    });
    let mutation: kogwistar_contracts::GraphMutationV1 = serde_json::from_value(json!({
        "scope": {
            "namespace": "docs",
            "workspace_id": "workspace-a",
            "graph_space": "knowledge"
        },
        "table": "gke_nodes",
        "entity_kind": "node",
        "event_id": "event-1",
        "op": "upsert",
        "payload": {},
        "record": {"id": "node-1", "metadata": {}},
        "embedding_dim": 0
    }))
    .expect("valid graph mutation");

    assert!(!inline_graph_mutations_outside_owner_scope(
        &principal,
        std::slice::from_ref(&mutation)
    ));

    let mut cross_scope = mutation;
    cross_scope.scope.workspace_id = Some("workspace-b".to_owned());
    assert!(inline_graph_mutations_outside_owner_scope(
        &principal,
        &[cross_scope]
    ));
}

#[tokio::test]
async fn sqlite_inline_graph_owner_revocation_fails_claim_durably() {
    let store_path = std::env::temp_dir().join(format!(
        "kogwistar-sqlite-inline-owner-{}.sqlite",
        uuid::Uuid::new_v4().simple()
    ));
    let auth_path = std::env::temp_dir().join(format!(
        "kogwistar-sqlite-inline-owner-auth-{}.sqlite",
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
    drop(auth);

    let mut registry = kogwistar_runtime::NativeOperationRegistry::default();
    registry
        .register_graph("app/inline-owner-test", "v1", |request| {
            let entity_id = format!("inline-node-{}", request.run_id);
            let namespace = if request.run_id == "sqlite-inline-cross-scope" {
                "beta"
            } else {
                "alpha"
            };
            let batch = serde_json::from_value(json!({
                "contract_version": 1,
                "atomic": true,
                "mutations": [{
                    "scope": {"namespace": namespace},
                    "table": "gke_nodes",
                    "entity_kind": "node",
                    "event_id": format!("inline-event-{}", request.run_id),
                    "op": "ADD",
                    "payload": {"id": entity_id},
                    "record": {"id": entity_id, "document": "inline owner test"},
                    "embedding_dim": 0
                }],
                "index_jobs": []
            }))
            .unwrap();
            Ok((
                kogwistar_runtime::NativeOperationOutput {
                    state_update: Vec::new(),
                    result: Some(json!({"inline_graph": true})),
                },
                batch,
            ))
        })
        .unwrap();
    let auth_config = AuthConfig {
        auth_db_url: Some(format!("sqlite:///{}", auth_path.display())),
        ..AuthConfig::default()
    };
    let mut service = SqliteRunApplicationService::open(&store_path)
        .unwrap()
        .with_native_operations(registry)
        .with_owner_auth_config(&auth_config)
        .await
        .unwrap();
    let submit = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/workflow/runs".to_owned(),
            body: serde_json::to_vec(&json!({
                "run_id": "sqlite-inline-owner-revoked",
                "workflow_id": "wf",
                "conversation_id": "conv-inline-owner-revoked",
                "start_node_id": "entry",
                "node_ops": {"entry": "rust-native/app/inline-owner-test"}
            }))
            .unwrap(),
            principal: json!({
                "user_id": "alice",
                "sub": "alice@example.test",
                "role": "rw",
                "ns": "alpha",
                "storage_ns": "alpha",
                "capabilities": ["workflow.run.write", "write_graph"]
            }),
        })
        .await;
    assert_eq!(submit.status, StatusCode::ACCEPTED.as_u16());

    service.owner_auth_store = None;
    assert!(matches!(
        service
            .resolve_inline_native_owner_principal("sqlite-inline-owner-revoked", "write_graph")
            .await,
        Err(InlineOwnerPrincipalError::Unavailable(_))
    ));
    service.owner_auth_store = Some(AuthStore::from_config(&auth_config).await.unwrap());

    tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
    let auth_connection = rusqlite::Connection::open(&auth_path).unwrap();
    let changed = auth_connection
        .execute(
            "UPDATE users SET global_role = 'ro' WHERE user_id = 'alice'",
            [],
        )
        .unwrap();
    assert_eq!(changed, 1);

    assert!(service.execute_inline_native_work_once().await.unwrap());
    let run = service
        .store
        .get_server_run("sqlite-inline-owner-revoked")
        .unwrap()
        .unwrap();
    assert_eq!(run.status, "failed");
    let lanes = service
        .store
        .list_projected_lane_messages(LaneMessageFilter {
            correlation_id: Some("sqlite-inline-owner-revoked".to_owned()),
            limit: 10,
            ..LaneMessageFilter::default()
        })
        .unwrap();
    assert_eq!(lanes.len(), 1);
    assert_eq!(lanes[0].status, "completed");
    assert!(!service.execute_inline_native_work_once().await.unwrap());
    assert!(
        service
            .store
            .replay_raw_events("alpha", 0, 100)
            .unwrap()
            .iter()
            .all(|event| event.event_id != "inline-event-sqlite-inline-owner-revoked")
    );

    let restored = auth_connection
        .execute(
            "UPDATE users SET global_role = 'rw' WHERE user_id = 'alice'",
            [],
        )
        .unwrap();
    assert_eq!(restored, 1);
    let cross_scope_submit = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/workflow/runs".to_owned(),
            body: serde_json::to_vec(&json!({
                "run_id": "sqlite-inline-cross-scope",
                "workflow_id": "wf",
                "conversation_id": "conv-inline-cross-scope",
                "start_node_id": "entry",
                "node_ops": {"entry": "rust-native/app/inline-owner-test"}
            }))
            .unwrap(),
            principal: json!({
                "user_id": "alice",
                "sub": "alice@example.test",
                "role": "rw",
                "ns": "alpha",
                "storage_ns": "alpha",
                "capabilities": ["workflow.run.write", "write_graph"]
            }),
        })
        .await;
    assert_eq!(cross_scope_submit.status, StatusCode::ACCEPTED.as_u16());
    tokio::time::sleep(std::time::Duration::from_millis(1100)).await;

    assert!(service.execute_inline_native_work_once().await.unwrap());
    assert_eq!(
        service
            .store
            .get_server_run("sqlite-inline-cross-scope")
            .unwrap()
            .unwrap()
            .status,
        "failed"
    );
    let cross_scope_lanes = service
        .store
        .list_projected_lane_messages(LaneMessageFilter {
            correlation_id: Some("sqlite-inline-cross-scope".to_owned()),
            limit: 10,
            ..LaneMessageFilter::default()
        })
        .unwrap();
    assert_eq!(cross_scope_lanes.len(), 1);
    assert_eq!(cross_scope_lanes[0].status, "completed");
    assert!(!service.execute_inline_native_work_once().await.unwrap());
    assert!(
        service
            .store
            .replay_raw_events("beta", 0, 100)
            .unwrap()
            .iter()
            .all(|event| event.event_id != "inline-event-sqlite-inline-cross-scope")
    );

    drop(auth_connection);
    drop(service);
    let _ = std::fs::remove_file(store_path);
    let _ = std::fs::remove_file(auth_path);
}

#[test]
fn recorded_step_snapshots_matches_python_fixture() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../../../contracts/golden/workflow-step-trace-context-v1.json"
    ))
    .unwrap();
    let recorded = ServerRunEvent {
        seq: 4,
        run_id: "run-1".to_owned(),
        event_type: "workflow.recorded_transition.v1".to_owned(),
        payload_json: json!({
            "transition_id": "transition-1",
            "reduced": {
                "state": {
                    "last_step_seq": 0,
                    "workflow_id": "workflow-1",
                    "last_node_id": "node-1",
                    "state": {}
                },
                "result": {"ok": true},
                "errors": [],
                "server_status": "running"
            }
        })
        .to_string(),
        created_at_ms: 10,
    };
    let trace = ServerRunEvent {
        seq: 5,
        run_id: "run-1".to_owned(),
        event_type: "workflow.trace.v1".to_owned(),
        payload_json: json!({
            "trace": {
                "event": "runtime.native.step.completed",
                "step_seq": 0,
                "node_id": "node-1",
                "trace_id": fixture["trace_context"]["trace_id"],
                "span_id": fixture["trace_context"]["span_id"],
                "parent_span_id": fixture["trace_context"]["parent_span_id"],
                "attempt": fixture["trace_context"]["attempt"]
            }
        })
        .to_string(),
        created_at_ms: 11,
    };

    let steps = recorded_step_snapshots(vec![recorded.clone(), trace]);
    assert_eq!(steps.len(), 1);
    assert_eq!(
        steps[0].trace_context.as_ref().unwrap(),
        &fixture["trace_context"]
    );
    let legacy_steps = recorded_step_snapshots(vec![recorded]);
    assert!(legacy_steps[0].trace_context.is_none());
    assert!(
        fixture["legacy_step_omits_trace_context"]
            .as_bool()
            .unwrap()
    );
}

// PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/tests/part_08.rs#in_memory_backend_routes_authenticated_graph_write_and_mcp_read; python=kogwistar/server_mcp_with_admin.py#api_viz_d3; fixture=contracts/golden/graph-acl-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/part_08.rs#in_memory_backend_routes_authenticated_graph_write_and_mcp_read; python_test=tests/test_visualization.py#test_visualization_uses_acl_policy_for_selected_python_graph_type
#[tokio::test]
async fn in_memory_backend_routes_authenticated_graph_write_and_mcp_read() {
    let acl_fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/graph-acl-v1.json"
    )))
    .unwrap();
    let isolation = &acl_fixture["graph_type_isolation"];
    let shared_entity_id = isolation["entity_id"].as_str().unwrap();
    let owner_policy = acl_fixture["targets"]
        .as_array()
        .unwrap()
        .iter()
        .find(|target| target["id"] == "node-owner")
        .unwrap();
    let owner_entity_id = owner_policy["id"].as_str().unwrap();
    let owner_id = owner_policy["owner_id"].as_str().unwrap();
    let private_policy = acl_fixture["targets"]
        .as_array()
        .unwrap()
        .iter()
        .find(|target| target["id"] == "node-private")
        .unwrap();
    let private_entity_id = private_policy["id"].as_str().unwrap();
    let private_owner_id = private_policy["owner_id"].as_str().unwrap();
    let no_acl_node_id = acl_fixture["targets"]
        .as_array()
        .unwrap()
        .iter()
        .find(|target| target["id"] == "node-no-acl")
        .unwrap()["id"]
        .as_str()
        .unwrap();
    let no_acl_edge_id = acl_fixture["targets"]
        .as_array()
        .unwrap()
        .iter()
        .find(|target| target["id"] == "edge-no-acl")
        .unwrap()["id"]
        .as_str()
        .unwrap();
    let edge_policy = acl_fixture["targets"]
        .as_array()
        .unwrap()
        .iter()
        .find(|target| target["id"] == "edge-public")
        .unwrap();
    let public_edge_id = edge_policy["id"].as_str().unwrap();
    let private_edge_policy = acl_fixture["targets"]
        .as_array()
        .unwrap()
        .iter()
        .find(|target| target["id"] == "edge-private")
        .unwrap();
    let private_edge_id = private_edge_policy["id"].as_str().unwrap();
    let private_edge_owner_id = private_edge_policy["owner_id"].as_str().unwrap();
    let scope_edge_policy = acl_fixture["targets"]
        .as_array()
        .unwrap()
        .iter()
        .find(|target| target["id"] == "edge-scope")
        .unwrap();
    let scope_edge_id = scope_edge_policy["id"].as_str().unwrap();
    let scope_edge_value = scope_edge_policy["security_scope"].as_str().unwrap();
    let scope_policy = acl_fixture["targets"]
        .as_array()
        .unwrap()
        .iter()
        .find(|target| target["id"] == "node-scope")
        .unwrap();
    let scope_entity_id = scope_policy["id"].as_str().unwrap();
    let scope_value = scope_policy["security_scope"].as_str().unwrap();
    let share_policies: Vec<&Value> = acl_fixture["targets"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|target| {
            target["grain"] == "node"
                && matches!(target["mode"].as_str(), Some("shared" | "group"))
        })
        .collect();
    let edge_share_policies: Vec<&Value> = acl_fixture["targets"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|target| {
            target["grain"] == "edge"
                && matches!(target["mode"].as_str(), Some("shared" | "group"))
        })
        .collect();
    let fixture_principal_id = acl_fixture["principal"]["id"].as_str().unwrap();
    let fixture_groups: Vec<String> = acl_fixture["principal"]["groups"]
        .as_array()
        .unwrap()
        .iter()
        .map(|group| group.as_str().unwrap().to_owned())
        .collect();
    let isolation_cases = isolation["cases"].as_array().unwrap();
    let knowledge_policy = isolation_cases
        .iter()
        .find(|case| case["truth_graph"] == "knowledge")
        .unwrap();
    let conversation_policy = isolation_cases
        .iter()
        .find(|case| case["truth_graph"] == "conversation")
        .unwrap();

    let secret = "in-memory-api-contract-secret";
    let auth = AuthConfig {
        mode: "jwt".to_owned(),
        algorithm: Some("HS256".to_owned()),
        key: Some(secret.to_owned()),
        ..AuthConfig::default()
    };
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let token_for_subject =
        |subject: &str, namespace: &str, capabilities: &[&str], graph_space: &str| {
        encode(
            &Header::new(Algorithm::HS256),
            &json!({
                "sub": subject,
                "user_id": subject,
                "role": "rw",
                "ns": [namespace],
                "storage_ns": "tenant-a",
                "workspace": "workspace-a",
                "workspace_id": "workspace-a",
                "graph_space": graph_space,
                "capabilities": capabilities,
                "exp": now + 300
            }),
            &EncodingKey::from_secret(secret.as_bytes()),
        )
        .unwrap()
    };
    let token_for_graph_space = |namespace: &str, capabilities: &[&str], graph_space: &str| {
        token_for_subject("writer", namespace, capabilities, graph_space)
    };
    let token = |namespace: &str, capabilities: &[&str]| {
        token_for_graph_space(namespace, capabilities, "facts")
    };
    let application = Arc::new(InMemoryGraphApplicationService::default());
    let make_app = || {
        router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: vec![],
                implementation: ImplementationSnapshot::default(),
                auth: auth.clone(),
            },
            application.clone(),
        )
    };
    let app = make_app();
    let scope = json!({
        "namespace": "tenant-a",
        "workspace_id": "workspace-a",
        "graph_space": "facts"
    });
    let mutation = |id: &str, table: &str, kind: &str, metadata: Value, document: Option<&str>| {
        json!({
            "scope": scope.clone(),
            "table": table,
            "entity_kind": kind,
            "event_id": format!("event-{id}"),
            "op": "ADD",
            "payload": {"id": id, "metadata": metadata},
            "record": {
                "id": id,
                "document": document,
                "metadata": metadata,
                "embedding": null
            },
            "embedding_dim": 0
        })
    };
    let mut mutations = vec![
        mutation(
            "n1",
            "gke_nodes",
            "node",
            json!({"label":"one","doc_ids":["doc-a"]}),
            Some(
                r#"{"label":"one","doc_ids":["doc-a"],"mentions":[{"spans":[{"doc_id":"doc-a","insertion_method":"parser"}]}]}"#,
            ),
        ),
        mutation(
            "n2",
            "gke_nodes",
            "node",
            json!({"label":"two","doc_ids":["doc-a"]}),
            Some(
                r#"{"label":"two","doc_ids":["doc-a"],"mentions":[{"spans":[{"doc_id":"doc-a","insertion_method":"parser"}]}]}"#,
            ),
        ),
        mutation(
            "e1",
            "gke_edges",
            "edge",
            json!({"source_ids":["n1"],"target_ids":["n2"],"relation":"links"}),
            Some(
                r#"{"source_ids":["n1"],"target_ids":["n2"],"relation":"links","mentions":[{"spans":[{"doc_id":"doc-a","insertion_method":"parser"}]}]}"#,
            ),
        ),
        mutation(
            "n3",
            "gke_nodes",
            "node",
            json!({"label":"private"}),
            Some("private"),
        ),
        mutation(
            shared_entity_id,
            "gke_nodes",
            "node",
            json!({"label":"same ID in knowledge"}),
            Some("same ID in knowledge"),
        ),
        mutation(
            "acl-n1",
            "gke_nodes",
            "node",
            json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"node","acl_target_entity_id":"n1","acl_version":1,"acl_mode":"public"}),
            None,
        ),
        mutation(
            "acl-n1-mention",
            "gke_nodes",
            "node",
            json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"grounding","acl_target_entity_id":"n1","acl_target_item_id":"n1::mention::0","acl_version":1,"acl_mode":"public"}),
            None,
        ),
        mutation(
            "acl-n1-span",
            "gke_nodes",
            "node",
            json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"span","acl_target_entity_id":"n1","acl_target_item_id":"n1::mention::0::span::0","acl_version":1,"acl_mode":"public"}),
            None,
        ),
        mutation(
            "acl-shared-knowledge",
            "gke_nodes",
            "node",
            json!({
                "entity_type":"acl_record",
                "acl_truth_graph":"knowledge",
                "acl_target_grain":"node",
                "acl_target_entity_id":shared_entity_id,
                "acl_version":1,
                "acl_mode":knowledge_policy["mode"]
            }),
            None,
        ),
        mutation(
            "acl-n2",
            "gke_nodes",
            "node",
            json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"node","acl_target_entity_id":"n2","acl_version":1,"acl_mode":"public"}),
            None,
        ),
        mutation(
            "acl-n2-mention",
            "gke_nodes",
            "node",
            json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"grounding","acl_target_entity_id":"n2","acl_target_item_id":"n2::mention::0","acl_version":1,"acl_mode":"public"}),
            None,
        ),
        mutation(
            "acl-n2-span",
            "gke_nodes",
            "node",
            json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"span","acl_target_entity_id":"n2","acl_target_item_id":"n2::mention::0::span::0","acl_version":1,"acl_mode":"public"}),
            None,
        ),
        mutation(
            "acl-e1",
            "gke_nodes",
            "node",
            json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"edge","acl_target_entity_id":"e1","acl_version":1,"acl_mode":"public"}),
            None,
        ),
        mutation(
            "acl-e1-mention",
            "gke_nodes",
            "node",
            json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"grounding","acl_target_entity_id":"e1","acl_target_item_id":"e1::mention::0","acl_version":1,"acl_mode":"public"}),
            None,
        ),
        mutation(
            "acl-e1-span",
            "gke_nodes",
            "node",
            json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"span","acl_target_entity_id":"e1","acl_target_item_id":"e1::mention::0::span::0","acl_version":1,"acl_mode":"public"}),
            None,
        ),
        mutation(
            "acl-n3",
            "gke_nodes",
            "node",
            json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"node","acl_target_entity_id":"n3","acl_version":1,"acl_mode":"private","acl_owner_id":"another-user"}),
            None,
        ),
        mutation(
            owner_entity_id,
            "gke_nodes",
            "node",
            json!({"label":"owner only"}),
            Some("owner only"),
        ),
        mutation(
            "acl-node-owner",
            "gke_nodes",
            "node",
            json!({
                "entity_type":"acl_record",
                "acl_truth_graph":acl_fixture["truth_graph"],
                "acl_target_grain":owner_policy["grain"],
                "acl_target_entity_id":owner_entity_id,
                "acl_version":1,
                "acl_mode":owner_policy["mode"],
                "owner_id":owner_id
            }),
            None,
        ),
        mutation(
            private_entity_id,
            "gke_nodes",
            "node",
            json!({"label":"another owner's node"}),
            Some("another owner's node"),
        ),
        mutation(
            "acl-node-private",
            "gke_nodes",
            "node",
            json!({
                "entity_type":"acl_record",
                "acl_truth_graph":acl_fixture["truth_graph"],
                "acl_target_grain":private_policy["grain"],
                "acl_target_entity_id":private_entity_id,
                "acl_version":1,
                "acl_mode":private_policy["mode"],
                "owner_id":private_owner_id
            }),
            None,
        ),
        mutation(
            scope_entity_id,
            "gke_nodes",
            "node",
            json!({"label":"team-scoped node"}),
            Some("team-scoped node"),
        ),
        mutation(
            "acl-node-scope",
            "gke_nodes",
            "node",
            json!({
                "entity_type":"acl_record",
                "acl_truth_graph":acl_fixture["truth_graph"],
                "acl_target_grain":scope_policy["grain"],
                "acl_target_entity_id":scope_entity_id,
                "acl_version":1,
                "acl_mode":scope_policy["mode"],
                "security_scope":scope_value
            }),
            None,
        ),
    ];
    for (index, share_policy) in share_policies.iter().enumerate() {
        let shared_entity_id = share_policy["id"].as_str().unwrap();
        mutations.push(mutation(
            shared_entity_id,
            "gke_nodes",
            "node",
            json!({"label":format!("{} graph node", share_policy["mode"].as_str().unwrap())}),
            Some("explicitly shared node"),
        ));
        let mut acl_metadata = json!({
            "entity_type":"acl_record",
            "acl_truth_graph":acl_fixture["truth_graph"],
            "acl_target_grain":share_policy["grain"],
            "acl_target_entity_id":shared_entity_id,
            "acl_version":1,
            "acl_mode":share_policy["mode"]
        });
        if share_policy["mode"] == "shared" {
            acl_metadata["shared_with_principals"] =
                share_policy["shared_with_principals"].clone();
        } else {
            acl_metadata["shared_with_groups"] = share_policy["shared_with_groups"].clone();
        }
        mutations.push(mutation(
            &format!("acl-node-share-{index}"),
            "gke_nodes",
            "node",
            acl_metadata,
            None,
        ));
    }
    let public_edge_data = json!({
        "source_ids":[shared_entity_id],
        "target_ids":[shared_entity_id],
        "source_edge_ids":[],
        "target_edge_ids":[],
        "relation":"references"
    });
    let public_edge_document = public_edge_data.to_string();
    mutations.push(mutation(
        public_edge_id,
        "gke_edges",
        "edge",
        public_edge_data,
        Some(&public_edge_document),
    ));
    mutations.push(mutation(
        "acl-edge-public",
        "gke_nodes",
        "node",
        json!({
            "entity_type":"acl_record",
            "acl_truth_graph":acl_fixture["truth_graph"],
            "acl_target_grain":edge_policy["grain"],
            "acl_target_entity_id":public_edge_id,
            "acl_version":1,
            "acl_mode":edge_policy["mode"]
        }),
        None,
    ));
    let private_edge_data = json!({
        "source_ids":[shared_entity_id],
        "target_ids":[shared_entity_id],
        "source_edge_ids":[],
        "target_edge_ids":[],
        "relation":"private references"
    });
    let private_edge_document = private_edge_data.to_string();
    mutations.push(mutation(
        private_edge_id,
        "gke_edges",
        "edge",
        private_edge_data,
        Some(&private_edge_document),
    ));
    mutations.push(mutation(
        "acl-edge-private",
        "gke_nodes",
        "node",
        json!({
            "entity_type":"acl_record",
            "acl_truth_graph":acl_fixture["truth_graph"],
            "acl_target_grain":private_edge_policy["grain"],
            "acl_target_entity_id":private_edge_id,
            "acl_version":1,
            "acl_mode":private_edge_policy["mode"],
            "owner_id":private_edge_owner_id
        }),
        None,
    ));
    let scope_edge_data = json!({
        "source_ids":[shared_entity_id],
        "target_ids":[shared_entity_id],
        "source_edge_ids":[],
        "target_edge_ids":[],
        "relation":"scope-limited references"
    });
    let scope_edge_document = scope_edge_data.to_string();
    mutations.push(mutation(
        scope_edge_id,
        "gke_edges",
        "edge",
        scope_edge_data,
        Some(&scope_edge_document),
    ));
    mutations.push(mutation(
        "acl-edge-scope",
        "gke_nodes",
        "node",
        json!({
            "entity_type":"acl_record",
            "acl_truth_graph":acl_fixture["truth_graph"],
            "acl_target_grain":scope_edge_policy["grain"],
            "acl_target_entity_id":scope_edge_id,
            "acl_version":1,
            "acl_mode":scope_edge_policy["mode"],
            "security_scope":scope_edge_value
        }),
        None,
    ));
    for (index, policy) in edge_share_policies.iter().enumerate() {
        let entity_id = policy["id"].as_str().unwrap();
        let edge_data = json!({
            "source_ids":[shared_entity_id],
            "target_ids":[shared_entity_id],
            "source_edge_ids":[],
            "target_edge_ids":[],
            "relation":"shared references"
        });
        let edge_document = edge_data.to_string();
        mutations.push(mutation(
            entity_id,
            "gke_edges",
            "edge",
            edge_data,
            Some(&edge_document),
        ));
        let mut acl_record = json!({
            "entity_type":"acl_record",
            "acl_truth_graph":acl_fixture["truth_graph"],
            "acl_target_grain":policy["grain"],
            "acl_target_entity_id":entity_id,
            "acl_version":1,
            "acl_mode":policy["mode"]
        });
        for field in ["shared_with_principals", "shared_with_groups"] {
            if let Some(value) = policy.get(field) {
                acl_record[field] = value.clone();
            }
        }
        mutations.push(mutation(
            &format!("acl-edge-shared-{index}"),
            "gke_nodes",
            "node",
            acl_record,
            None,
        ));
    }
    for (index, embedding) in [
        (0, json!([1.0, 0.0])),
        (1, json!([0.8, 0.2])),
        (3, json!([0.99, 0.01])),
        (5, json!([0.999, 0.001])),
    ] {
        mutations[index]["record"]["embedding"] = embedding;
        mutations[index]["embedding_dim"] = json!(2);
    }
    mutations.push(mutation(
        no_acl_node_id,
        "gke_nodes",
        "node",
        json!({"label":"Node without an ACL record"}),
        Some("Node without an ACL record"),
    ));
    let no_acl_edge = json!({
        "source_ids":[shared_entity_id],
        "target_ids":[shared_entity_id],
        "source_edge_ids":[],
        "target_edge_ids":[],
        "relation":"Edge without an ACL record"
    });
    let no_acl_edge_document = no_acl_edge.to_string();
    mutations.push(mutation(
        no_acl_edge_id,
        "gke_edges",
        "edge",
        no_acl_edge,
        Some(&no_acl_edge_document),
    ));
    let body = json!({"contract_version":1,"atomic":true,"mutations":mutations,"index_jobs":[]});

    let denied_write = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/internal/graph/mutations")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!("Bearer {}", token("docs", &["read_graph"])),
                )
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(denied_write.status(), StatusCode::FORBIDDEN);

    let denied_namespace = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/internal/graph/mutations")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!("Bearer {}", token("workflow", &["write_graph"])),
                )
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(denied_namespace.status(), StatusCode::FORBIDDEN);

    let mut cross_workspace_body = body.clone();
    for mutation in cross_workspace_body["mutations"].as_array_mut().unwrap() {
        mutation["scope"]["workspace_id"] = json!("workspace-b");
    }
    let denied_scope = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/internal/graph/mutations")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!("Bearer {}", token("docs", &["write_graph"])),
                )
                .body(Body::from(
                    serde_json::to_vec(&cross_workspace_body).unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(denied_scope.status(), StatusCode::FORBIDDEN);

    let written = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/internal/graph/mutations")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!("Bearer {}", token("docs", &["write_graph"])),
                )
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(written.status(), StatusCode::OK);
    let written: Value =
        serde_json::from_slice(&to_bytes(written.into_body(), 65_536).await.unwrap()).unwrap();
    assert_eq!(written["atomic"], true);
    assert_eq!(
        written["results"].as_array().unwrap().len(),
        body["mutations"].as_array().unwrap().len()
    );
    let load_persisted = |id: &str, input: Value| {
        Request::builder()
            .method("POST")
            .uri("/mcp")
            .header("content-type", "application/json")
            .header(
                "authorization",
                format!("Bearer {}", token("docs", &["read_graph"])),
            )
            .body(Body::from(
                json!({
                    "jsonrpc":"2.0","id":id,"method":"tools/call",
                    "params":{"name":"kg_load_persisted","arguments":{"inp":input}}
                })
                .to_string(),
            ))
            .unwrap()
    };
    let loaded = app
        .clone()
        .oneshot(load_persisted(
            "load-persisted",
            json!({"doc_ids":["doc-a"],"sid":false}),
        ))
        .await
        .unwrap();
    let loaded: Value =
        serde_json::from_slice(&to_bytes(loaded.into_body(), 65_536).await.unwrap()).unwrap();
    assert_eq!(
        loaded["result"]["node_ids"],
        json!(["n1", "n2"]),
        "{loaded}"
    );
    assert_eq!(loaded["result"]["edge_ids"], json!(["e1"]));
    let loaded_by_method = app
        .clone()
        .oneshot(load_persisted(
            "load-persisted-method",
            json!({"doc_ids":["doc-a"],"insertion_method":"parser","sid":false}),
        ))
        .await
        .unwrap();
    let loaded_by_method: Value = serde_json::from_slice(
        &to_bytes(loaded_by_method.into_body(), 65_536)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(loaded_by_method["result"]["node_ids"], json!(["n1", "n2"]));
    assert_eq!(loaded_by_method["result"]["edge_ids"], json!(["e1"]));

    let brute_force = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!("Bearer {}", token("docs", &["read_graph"])),
                )
                .body(Body::from(
                    json!({
                        "jsonrpc":"2.0","id":"brute-force","method":"tools/call",
                        "params":{"name":"kg_propose_bruteforce","arguments":{"inp":{
                            "pair_kind":"node_node","allowed_docs":["doc-a"]
                        }}}
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(brute_force.status(), StatusCode::OK);
    let brute_force: Value =
        serde_json::from_slice(&to_bytes(brute_force.into_body(), 65_536).await.unwrap()).unwrap();
    assert_eq!(
        brute_force["result"]["pairs"],
        json!([{
            "left_id":"n1","left_kind":"node",
            "right_id":"n2","right_kind":"node"
        }])
    );
    let mixed_pairs = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!("Bearer {}", token("docs", &["read_graph"])),
                )
                .body(Body::from(
                    json!({
                        "jsonrpc":"2.0","id":"mixed-pairs","method":"tools/call",
                        "params":{"name":"kg_propose_bruteforce","arguments":{"inp":{
                            "pair_kind":"any_any","allowed_docs":["doc-a"]
                        }}}
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    let mixed_pairs: Value =
        serde_json::from_slice(&to_bytes(mixed_pairs.into_body(), 65_536).await.unwrap()).unwrap();
    assert_eq!(mixed_pairs["result"]["pairs"].as_array().unwrap().len(), 3);
    assert!(mixed_pairs.to_string().contains("e1"));
    let denied_proposal = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {}", token("docs", &[])))
                .body(Body::from(
                    json!({
                        "jsonrpc":"2.0","id":"denied-proposal","method":"tools/call",
                        "params":{"name":"kg_propose_bruteforce","arguments":{"inp":{}}}
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    let denied_proposal: Value =
        serde_json::from_slice(&to_bytes(denied_proposal.into_body(), 65_536).await.unwrap())
            .unwrap();
    assert!(denied_proposal.get("error").is_some());
    assert!(denied_proposal.get("result").is_none());

    let cross_kind_fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/graph-neighbors-cross-kind-id-v1.json"
    )))
    .unwrap();
    let parent_edge = &cross_kind_fixture["parent_edge"];
    let cross_kind_id = cross_kind_fixture["expected"]["node_ids"][0]
        .as_str()
        .unwrap();
    let parent_edge_id = cross_kind_fixture["rid"].as_str().unwrap();
    let mut cross_kind_node = mutation(
        cross_kind_id,
        "gke_nodes",
        "node",
        json!({"label":"shared node identity"}),
        Some(r#"{"label":"shared node identity"}"#),
    );
    cross_kind_node["event_id"] = json!("event-cross-kind-node");
    let mut cross_kind_edge = mutation(
        cross_kind_id,
        "gke_edges",
        "edge",
        json!({"source_ids":[],"target_ids":[],"relation":"shared edge identity"}),
        Some(r#"{"source_ids":[],"target_ids":[],"relation":"shared edge identity"}"#),
    );
    cross_kind_edge["event_id"] = json!("event-cross-kind-edge");
    let mut parent_edge_mutation = mutation(
        parent_edge_id,
        "gke_edges",
        "edge",
        json!(parent_edge),
        Some(&parent_edge.to_string()),
    );
    parent_edge_mutation["event_id"] = json!("event-cross-kind-parent-edge");
    let cross_kind_mutations = vec![
        cross_kind_node,
        cross_kind_edge,
        parent_edge_mutation,
        mutation(
            "acl-cross-kind-node",
            "gke_nodes",
            "node",
            json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"node","acl_target_entity_id":cross_kind_id,"acl_version":1,"acl_mode":"public"}),
            None,
        ),
        mutation(
            "acl-cross-kind-edge",
            "gke_nodes",
            "node",
            json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"edge","acl_target_entity_id":cross_kind_id,"acl_version":1,"acl_mode":"public"}),
            None,
        ),
        mutation(
            "acl-cross-kind-parent-edge",
            "gke_nodes",
            "node",
            json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"edge","acl_target_entity_id":parent_edge_id,"acl_version":1,"acl_mode":"public"}),
            None,
        ),
    ];
    let cross_kind_body = json!({
        "contract_version":1,
        "atomic":true,
        "mutations":cross_kind_mutations,
        "index_jobs":[]
    });
    let cross_kind_write = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/internal/graph/mutations")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!("Bearer {}", token("docs", &["write_graph"])),
                )
                .body(Body::from(serde_json::to_vec(&cross_kind_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(cross_kind_write.status(), StatusCode::OK);
    let cross_kind_neighbors = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!("Bearer {}", token("docs", &["read_graph"])),
                )
                .body(Body::from(
                    json!({
                        "jsonrpc":"2.0",
                        "id":"cross-kind-neighbors",
                        "method":"tools/call",
                        "params":{"name":"kg_neighbors","arguments":{
                            "rid":parent_edge_id,
                            "truth_graph":"knowledge"
                        }}
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(cross_kind_neighbors.status(), StatusCode::OK);
    let cross_kind_neighbors: Value = serde_json::from_slice(
        &to_bytes(cross_kind_neighbors.into_body(), 65_536)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        cross_kind_neighbors["result"]["implementation_authority"],
        "rust"
    );
    assert_eq!(
        cross_kind_neighbors["result"]["seam_attestation_version"],
        1
    );
    assert_eq!(cross_kind_neighbors["result"]["seam_crossings"], json!([]));
    assert_eq!(
        cross_kind_neighbors["result"]["nodes"],
        cross_kind_fixture["expected"]["node_ids"]
    );
    assert_eq!(
        cross_kind_neighbors["result"]["edges"],
        cross_kind_fixture["expected"]["edge_ids"]
    );

    let redirect_mutations = [
        mutation(
            "redirect-old",
            "gke_nodes",
            "node",
            json!({"label":"old node","lifecycle_status":"tombstoned","redirect_to_id":"redirect-current"}),
            Some("old node"),
        ),
        mutation(
            "redirect-current",
            "gke_nodes",
            "node",
            json!({"label":"current node"}),
            Some("current node"),
        ),
        mutation(
            "redirect-neighbor",
            "gke_nodes",
            "node",
            json!({"label":"neighbor"}),
            Some("neighbor"),
        ),
        mutation(
            "redirect-edge",
            "gke_edges",
            "edge",
            json!({"source_ids":["redirect-current"],"target_ids":["redirect-neighbor"]}),
            Some(r#"{"source_ids":["redirect-current"],"target_ids":["redirect-neighbor"]}"#),
        ),
        mutation(
            "redirect-edge-old",
            "gke_edges",
            "edge",
            json!({"source_ids":[],"target_ids":[],"lifecycle_status":"tombstoned","redirect_to_id":"redirect-edge"}),
            Some(r#"{"source_ids":[],"target_ids":[]}"#),
        ),
        mutation(
            "redirect-edge-source",
            "gke_edges",
            "edge",
            json!({"source_ids":[],"target_ids":[],"lifecycle_status":"tombstoned","redirect_to_id":"redirect-edge"}),
            Some(r#"{"source_ids":[],"target_ids":[]}"#),
        ),
        mutation(
            "redirect-edge-old",
            "gke_nodes",
            "node",
            json!({"label":"same-string tombstoned node"}),
            Some("same-string tombstoned node"),
        ),
        mutation(
            "redirect-old-private",
            "gke_nodes",
            "node",
            json!({"label":"old private node","lifecycle_status":"tombstoned","redirect_to_id":"redirect-private"}),
            Some("old private node"),
        ),
        mutation(
            "redirect-private",
            "gke_nodes",
            "node",
            json!({"label":"private target"}),
            Some("private target"),
        ),
        mutation(
            "redirect-private-edge",
            "gke_edges",
            "edge",
            json!({"source_ids":["redirect-current"],"target_ids":["redirect-neighbor"]}),
            Some(r#"{"source_ids":["redirect-current"],"target_ids":["redirect-neighbor"]}"#),
        ),
        mutation(
            "redirect-private-edge-old",
            "gke_edges",
            "edge",
            json!({"source_ids":[],"target_ids":[],"lifecycle_status":"tombstoned","redirect_to_id":"redirect-private-edge"}),
            Some(r#"{"source_ids":[],"target_ids":[]}"#),
        ),
        mutation(
            "acl-redirect-current",
            "gke_nodes",
            "node",
            json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"node","acl_target_entity_id":"redirect-current","acl_version":1,"acl_mode":"public"}),
            None,
        ),
        mutation(
            "acl-redirect-neighbor",
            "gke_nodes",
            "node",
            json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"node","acl_target_entity_id":"redirect-neighbor","acl_version":1,"acl_mode":"public"}),
            None,
        ),
        mutation(
            "acl-redirect-edge",
            "gke_nodes",
            "node",
            json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"edge","acl_target_entity_id":"redirect-edge","acl_version":1,"acl_mode":"public"}),
            None,
        ),
        mutation(
            "acl-redirect-private",
            "gke_nodes",
            "node",
            json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"node","acl_target_entity_id":"redirect-private","acl_version":1,"acl_mode":"private","acl_owner_id":"other-user"}),
            None,
        ),
        mutation(
            "acl-redirect-private-edge",
            "gke_nodes",
            "node",
            json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"edge","acl_target_entity_id":"redirect-private-edge","acl_version":1,"acl_mode":"private","acl_owner_id":"other-user"}),
            None,
        ),
    ]
    .into_iter()
    .enumerate()
    .map(|(index, mut value)| {
        value["event_id"] = json!(format!("redirect-event-{index}"));
        value
    })
    .collect::<Vec<_>>();
    let redirect_body = json!({
        "contract_version":1,
        "atomic":true,
        "mutations":redirect_mutations,
        "index_jobs":[]
    });
    let redirect_write = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/internal/graph/mutations")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!("Bearer {}", token("docs", &["write_graph"])),
                )
                .body(Body::from(serde_json::to_vec(&redirect_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(redirect_write.status(), StatusCode::OK);

    let read_redirect = |rid: &str, resolve_mode: &str| {
        app.clone().oneshot(
            Request::builder()
                .method("POST")
                .uri("/internal/graph/neighbors")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!("Bearer {}", token("docs", &["read_graph"])),
                )
                .body(Body::from(
                    json!({"rid":rid,"resolve_mode":resolve_mode,"truth_graph":"knowledge"})
                        .to_string(),
                ))
                .unwrap(),
        )
    };
    let redirected = read_redirect("redirect-old", "redirect").await.unwrap();
    assert_eq!(redirected.status(), StatusCode::OK);
    let redirected: Value =
        serde_json::from_slice(&to_bytes(redirected.into_body(), 65_536).await.unwrap()).unwrap();
    assert_eq!(redirected["nodes"], json!(["redirect-neighbor"]));
    assert_eq!(redirected["edges"], json!(["redirect-edge"]));

    let redirected_edge = read_redirect("redirect-edge-old", "redirect").await.unwrap();
    assert_eq!(redirected_edge.status(), StatusCode::OK);
    let redirected_edge: Value = serde_json::from_slice(
        &to_bytes(redirected_edge.into_body(), 65_536).await.unwrap(),
    )
    .unwrap();
    assert_eq!(redirected_edge["nodes"], json!([]));
    assert_eq!(redirected_edge["edges"], json!([]));

    let redirected_edge_source = read_redirect("redirect-edge-source", "redirect")
        .await
        .unwrap();
    assert_eq!(redirected_edge_source.status(), StatusCode::OK);
    let redirected_edge_source: Value = serde_json::from_slice(
        &to_bytes(redirected_edge_source.into_body(), 65_536)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        redirected_edge_source["nodes"],
        json!(["redirect-current", "redirect-neighbor"])
    );
    assert_eq!(redirected_edge_source["edges"], json!([]));

    let private_edge_redirect = read_redirect("redirect-private-edge-old", "redirect")
        .await
        .unwrap();
    assert_eq!(private_edge_redirect.status(), StatusCode::OK);
    let private_edge_redirect: Value = serde_json::from_slice(
        &to_bytes(private_edge_redirect.into_body(), 65_536)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(private_edge_redirect["nodes"], json!([]));
    assert_eq!(private_edge_redirect["edges"], json!([]));

    let private_redirect = read_redirect("redirect-old-private", "redirect")
        .await
        .unwrap();
    assert_eq!(private_redirect.status(), StatusCode::OK);
    let private_redirect: Value = serde_json::from_slice(
        &to_bytes(private_redirect.into_body(), 65_536)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(private_redirect["nodes"], json!([]));
    assert_eq!(private_redirect["edges"], json!([]));

    let conversation_scope = json!({
        "namespace": "tenant-a",
        "workspace_id": "workspace-a",
        "graph_space": "facts"
    });
    let conversation_mutation = |id: &str, metadata: Value, document: Option<&str>| {
        let mut value = mutation(id, "gke_nodes", "node", metadata.clone(), document);
        value["event_id"] = json!(format!("conversation-event-{id}"));
        value["scope"] = conversation_scope.clone();
        value
    };
    let scoped_edge_mutation =
        |id: &str, metadata: Value, document: Option<&str>| {
            let mut value = mutation(id, "gke_edges", "edge", metadata, document);
            value["event_id"] = json!(format!("conversation-edge-event-{id}"));
            value["scope"] = conversation_scope.clone();
            value
        };
    let conversation_parent_edge = json!({
        "source_ids": [shared_entity_id],
        "target_ids": [],
        "source_edge_ids": [shared_entity_id],
        "target_edge_ids": [],
        "relation": "conversation cross-kind reference"
    });
    let workflow_parent_edge = json!({
        "source_ids": [shared_entity_id],
        "target_ids": [],
        "source_edge_ids": [shared_entity_id],
        "target_edge_ids": [],
        "relation": "workflow cross-kind reference"
    });
    let conversation_body = json!({
        "contract_version": 1,
        "atomic": true,
        "index_jobs": [],
        "mutations": [
            conversation_mutation(
                "conversation-node",
                json!({"label":"conversation only"}),
                Some("conversation only"),
            ),
            conversation_mutation(
                "conversation-acl",
                json!({
                    "entity_type":"acl_record",
                    "acl_truth_graph":"conversation",
                    "acl_target_grain":"node",
                    "acl_target_entity_id":"conversation-node",
                    "acl_version":1,
                    "acl_mode":"public"
                }),
                None,
            ),
            conversation_mutation(
                "conversation-acl-shared",
                json!({
                    "entity_type":"acl_record",
                    "acl_truth_graph":"conversation",
                    "acl_target_grain":"node",
                    "acl_target_entity_id":shared_entity_id,
                    "acl_version":1,
                    "acl_mode":conversation_policy["mode"],
                    "owner_id":conversation_policy["owner_id"]
                }),
                None,
            ),
            scoped_edge_mutation(
                shared_entity_id,
                json!({"source_ids":[],"target_ids":[],"relation":"same ID edge"}),
                Some(r#"{"source_ids":[],"target_ids":[],"relation":"same ID edge"}"#),
            ),
            scoped_edge_mutation(
                "conversation-private-edge",
                json!({"source_ids":[],"target_ids":[],"relation":"private conversation edge"}),
                Some(r#"{"source_ids":[],"target_ids":[],"relation":"private conversation edge"}"#),
            ),
            scoped_edge_mutation(
                "conversation-parent-edge",
                conversation_parent_edge.clone(),
                Some(&conversation_parent_edge.to_string()),
            ),
            scoped_edge_mutation(
                "workflow-parent-edge",
                workflow_parent_edge.clone(),
                Some(&workflow_parent_edge.to_string()),
            ),
            scoped_edge_mutation(
                "workflow-private-edge",
                json!({"source_ids":[],"target_ids":[],"relation":"private workflow edge"}),
                Some(r#"{"source_ids":[],"target_ids":[],"relation":"private workflow edge"}"#),
            ),
            conversation_mutation(
                "conversation-acl-cross-kind-edge",
                json!({
                    "entity_type":"acl_record",
                    "acl_truth_graph":"conversation",
                    "acl_target_grain":"edge",
                    "acl_target_entity_id":shared_entity_id,
                    "acl_version":1,
                    "acl_mode":conversation_policy["mode"],
                    "owner_id":conversation_policy["owner_id"]
                }),
                None,
            ),
            conversation_mutation(
                "conversation-acl-parent-edge",
                json!({
                    "entity_type":"acl_record",
                    "acl_truth_graph":"conversation",
                    "acl_target_grain":"edge",
                    "acl_target_entity_id":"conversation-parent-edge",
                    "acl_version":1,
                    "acl_mode":"public"
                }),
                None,
            ),
            conversation_mutation(
                "conversation-acl-private-edge",
                json!({
                    "entity_type":"acl_record",
                    "acl_truth_graph":"conversation",
                    "acl_target_grain":"edge",
                    "acl_target_entity_id":"conversation-private-edge",
                    "acl_version":1,
                    "acl_mode":conversation_policy["mode"],
                    "owner_id":conversation_policy["owner_id"]
                }),
                None,
            ),
            conversation_mutation(
                "workflow-acl-private-edge",
                json!({
                    "entity_type":"acl_record",
                    "acl_truth_graph":"workflow",
                    "acl_target_grain":"edge",
                    "acl_target_entity_id":"workflow-private-edge",
                    "acl_version":1,
                    "acl_mode":"private",
                    "owner_id":"workflow-owner"
                }),
                None,
            ),
        ]
    });
    let conversation_written = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/internal/graph/mutations")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!(
                        "Bearer {}",
                        token_for_graph_space("docs", &["write_graph"], "facts")
                    ),
                )
                .body(Body::from(serde_json::to_vec(&conversation_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(conversation_written.status(), StatusCode::OK);

    for case in isolation_cases
        .iter()
        .filter(|case| matches!(case["truth_graph"].as_str(), Some("workflow" | "wisdom")))
    {
        let truth_graph = case["truth_graph"].as_str().unwrap();
        let mut graph_acl = mutation(
            &format!("{truth_graph}-acl-shared"),
            "gke_nodes",
            "node",
            json!({
                "entity_type":"acl_record",
                "acl_truth_graph":truth_graph,
                "acl_target_grain":"node",
                "acl_target_entity_id":shared_entity_id,
                "acl_version":1,
                "acl_mode":case["mode"]
            }),
            None,
        );
        graph_acl["event_id"] = json!(format!("{truth_graph}-event-acl"));
        let mut graph_mutations = vec![graph_acl];
        if truth_graph == "workflow" {
            let mut edge_acl = mutation(
                "workflow-acl-cross-kind-edge",
                "gke_nodes",
                "node",
                json!({
                    "entity_type":"acl_record",
                    "acl_truth_graph":"workflow",
                    "acl_target_grain":"edge",
                    "acl_target_entity_id":shared_entity_id,
                    "acl_version":1,
                    "acl_mode":"public"
                }),
                None,
            );
            edge_acl["event_id"] = json!("workflow-event-acl-edge");
            graph_mutations.push(edge_acl);
            let mut parent_edge_acl = mutation(
                "workflow-acl-parent-edge",
                "gke_nodes",
                "node",
                json!({
                    "entity_type":"acl_record",
                    "acl_truth_graph":"workflow",
                    "acl_target_grain":"edge",
                    "acl_target_entity_id":"workflow-parent-edge",
                    "acl_version":1,
                    "acl_mode":"public"
                }),
                None,
            );
            parent_edge_acl["event_id"] = json!("workflow-event-acl-parent-edge");
            graph_mutations.push(parent_edge_acl);
        }
        let graph_body = json!({
            "contract_version":1,
            "atomic":true,
            "index_jobs":[],
            "mutations":graph_mutations
        });
        let written = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/graph/mutations")
                    .header("content-type", "application/json")
                    .header(
                        "authorization",
                        format!(
                            "Bearer {}",
                            token_for_graph_space("docs", &["write_graph"], "facts")
                        ),
                    )
                    .body(Body::from(serde_json::to_vec(&graph_body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(written.status(), StatusCode::OK, "graph={truth_graph}");
    }

    let visualization_request = |truth_graph: &str,
                                 graph_space: &str,
                                 security_scope: Option<&str>,
                                 principal_id: &str,
                                 groups: &[String]| {
        let mut principal = json!({
            "sub":principal_id, "role":"rw", "ns":["docs"],
            "storage_ns":"tenant-a", "workspace_id":"workspace-a",
            "graph_space":graph_space, "capabilities":["read_graph"]
        });
        principal["groups"] = json!(groups);
        if let Some(security_scope) = security_scope {
            principal["security_scope"] = json!(security_scope);
        }
        ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: INTERNAL_GRAPH_VIZ_D3_ROUTE.to_owned(),
            body: serde_json::to_vec(&json!({"graph_type":truth_graph})).unwrap(),
            principal,
        }
    };
    let conversation_visualization = application
        .graph_read_effect(&visualization_request("conversation", "facts", None, "writer", &[]))
        .await;
    assert_eq!(conversation_visualization.status, StatusCode::OK.as_u16());
    let conversation_visualization: Value =
        serde_json::from_slice(&conversation_visualization.body).unwrap();
    assert!(
        conversation_visualization["nodes"]
            .to_string()
            .contains("conversation-node"),
        "selected conversation visualization did not return its graph: {conversation_visualization}"
    );
    assert!(
        !conversation_visualization["nodes"]
            .to_string()
            .contains("n1"),
        "conversation visualization leaked knowledge graph data: {conversation_visualization}"
    );
    assert!(
        !conversation_visualization["nodes"]
            .to_string()
            .contains(shared_entity_id),
        "conversation visualization ignored its graph-scoped private ACL: {conversation_visualization}"
    );
    assert!(
        !conversation_visualization["nodes"]
            .to_string()
            .contains("conversation-private-edge"),
        "conversation visualization exposed a private edge to a non-owner: {conversation_visualization}"
    );

    let conversation_owner_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/viz/d3.json?graph_type=conversation")
                .header(
                    "authorization",
                    format!(
                        "Bearer {}",
                        token_for_subject(
                            conversation_policy["owner_id"].as_str().unwrap(),
                            "docs",
                            &["read_graph"],
                            "facts"
                        )
                    ),
                )
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(conversation_owner_response.status(), StatusCode::OK);
    let conversation_owner_payload: Value = serde_json::from_slice(
        &to_bytes(conversation_owner_response.into_body(), 65_536)
            .await
            .unwrap(),
    )
    .unwrap();
    assert!(
        conversation_owner_payload["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .any(|node| node["id"] == shared_entity_id),
        "conversation ACL owner must see its private graph node: {conversation_owner_payload}"
    );
    assert!(
        conversation_owner_payload["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .any(|node| node["id"] == "conversation-private-edge"),
        "conversation ACL owner must see its private graph edge: {conversation_owner_payload}"
    );

    let conversation_cross_kind_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/internal/graph/neighbors")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!(
                        "Bearer {}",
                        token_for_subject(
                            conversation_policy["owner_id"].as_str().unwrap(),
                            "docs",
                            &["read_graph"],
                            "facts"
                        )
                    ),
                )
                .body(Body::from(
                    json!({
                        "rid":"conversation-parent-edge",
                        "truth_graph":"conversation"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(conversation_cross_kind_response.status(), StatusCode::OK);
    let conversation_cross_kind: Value = serde_json::from_slice(
        &to_bytes(conversation_cross_kind_response.into_body(), 65_536)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(conversation_cross_kind["nodes"], json!([shared_entity_id]));
    assert_eq!(conversation_cross_kind["edges"], json!([shared_entity_id]));

    let workflow_cross_kind_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/internal/graph/neighbors")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!(
                        "Bearer {}",
                        token_for_subject("writer", "docs", &["read_graph"], "facts")
                    ),
                )
                .body(Body::from(
                    json!({
                        "rid":"workflow-parent-edge",
                        "truth_graph":"workflow"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(workflow_cross_kind_response.status(), StatusCode::OK);
    let workflow_cross_kind: Value = serde_json::from_slice(
        &to_bytes(workflow_cross_kind_response.into_body(), 65_536)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(workflow_cross_kind["nodes"], json!([shared_entity_id]));
    assert_eq!(workflow_cross_kind["edges"], json!([shared_entity_id]));

    for case in isolation_cases {
        let truth_graph = case["truth_graph"].as_str().unwrap();
        let response = application
            .graph_read_effect(&visualization_request(
                truth_graph,
                "facts",
                None,
                "writer",
                &[],
            ))
            .await;
        assert_eq!(
            response.status,
            StatusCode::OK.as_u16(),
            "graph={truth_graph}"
        );
        let payload: Value = serde_json::from_slice(&response.body).unwrap();
        let rendered_nodes = payload["nodes"].to_string();
        let expected_visible = case["expected_visible"].as_bool().unwrap();
        assert_eq!(
            rendered_nodes.contains(shared_entity_id),
            expected_visible,
            "selected graph={truth_graph} used the wrong ACL policy: {payload}"
        );
        if truth_graph == acl_fixture["truth_graph"].as_str().unwrap() {
            assert!(
                rendered_nodes.contains(public_edge_id),
                "a public edge with visible endpoints must appear in reified graph output: {payload}"
            );
            assert!(
                !rendered_nodes.contains(private_edge_id),
                "a non-owner must not see the private edge: {payload}"
            );
            assert!(
                !rendered_nodes.contains(scope_edge_id),
                "a principal without the matching scope must not see the scoped edge: {payload}"
            );
            for share_policy in &edge_share_policies {
                assert!(
                    !rendered_nodes.contains(share_policy["id"].as_str().unwrap()),
                    "an unshared principal must not see a shared/group edge: {payload}"
                );
            }
            assert!(
                !rendered_nodes.contains(owner_entity_id),
                "a non-owner must not see the private fixture node: {payload}"
            );
            assert!(
                !rendered_nodes.contains(private_entity_id),
                "a non-owner must not see another principal's private node: {payload}"
            );
            assert!(
                !rendered_nodes.contains(scope_entity_id),
                "a principal without the matching security scope must not see the scoped node: {payload}"
            );
            assert!(
                !rendered_nodes.contains(no_acl_node_id),
                "a node without an ACL record must be denied by default: {payload}"
            );
            assert!(
                !rendered_nodes.contains(no_acl_edge_id),
                "an edge without an ACL record must be denied by default: {payload}"
            );
            for share_policy in &share_policies {
                assert!(
                    !rendered_nodes.contains(share_policy["id"].as_str().unwrap()),
                    "an unshared principal must not see a shared/group node: {payload}"
                );
            }
        }
        if truth_graph == "workflow" || truth_graph == "wisdom" {
            assert!(
                !rendered_nodes.contains("n1") && !rendered_nodes.contains("conversation-node"),
                "selected graph={truth_graph} leaked records from another graph: {payload}"
            );
        }
        if truth_graph == "workflow" {
            assert!(
                !rendered_nodes.contains("workflow-private-edge"),
                "workflow visualization exposed a private edge to a non-owner: {payload}"
            );
        }
    }

    let workflow_owner_response = application
        .graph_read_effect(&visualization_request("workflow", "facts", None, "workflow-owner", &[]))
        .await;
    assert_eq!(workflow_owner_response.status, StatusCode::OK.as_u16());
    let workflow_owner_payload: Value =
        serde_json::from_slice(&workflow_owner_response.body).unwrap();
    assert!(
        workflow_owner_payload["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .any(|node| node["id"] == "workflow-private-edge"),
        "workflow ACL owner must see its private graph edge: {workflow_owner_payload}"
    );

    let owner_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/viz/d3.json?graph_type=knowledge")
                .header(
                    "authorization",
                    format!(
                        "Bearer {}",
                        token_for_subject(owner_id, "docs", &["read_graph"], "facts")
                    ),
                )
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(owner_response.status(), StatusCode::OK);
    let owner_payload: Value =
        serde_json::from_slice(&to_bytes(owner_response.into_body(), 65_536).await.unwrap())
            .unwrap();
    assert!(
        owner_payload["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .any(|node| node["id"] == owner_entity_id),
        "the ACL fixture owner must see the private node: {owner_payload}"
    );
    assert!(
        !owner_payload["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .any(|node| node["id"] == private_entity_id),
        "the fixture owner must not see another principal's private node: {owner_payload}"
    );
    assert!(
        !owner_payload["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .any(|node| node["id"] == private_edge_id),
        "Alice must not see Bob's private edge: {owner_payload}"
    );

    let private_edge_owner_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/viz/d3.json?graph_type=knowledge&mode=reify")
                .header(
                    "authorization",
                    format!(
                        "Bearer {}",
                        token_for_subject(private_edge_owner_id, "docs", &["read_graph"], "facts")
                    ),
                )
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(private_edge_owner_response.status(), StatusCode::OK);
    let private_edge_owner_payload: Value = serde_json::from_slice(
        &to_bytes(private_edge_owner_response.into_body(), 65_536)
            .await
            .unwrap(),
    )
    .unwrap();
    assert!(
        private_edge_owner_payload["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .any(|node| node["id"] == private_edge_id),
        "Bob must see the edge he owns: {private_edge_owner_payload}"
    );

    for (requested_scope, expected_visible) in [(scope_value, true), ("team-b", false)] {
        let scoped_response = application
            .graph_read_effect(&visualization_request(
                acl_fixture["truth_graph"].as_str().unwrap(),
                "facts",
                Some(requested_scope),
                "writer",
                &[],
            ))
            .await;
        assert_eq!(scoped_response.status, StatusCode::OK.as_u16());
        let scoped_payload: Value = serde_json::from_slice(&scoped_response.body).unwrap();
        assert_eq!(
            scoped_payload["nodes"]
                .as_array()
                .unwrap()
                .iter()
                .any(|node| node["id"] == scope_entity_id),
            expected_visible,
            "security scope {requested_scope} had incorrect visibility: {scoped_payload}"
        );
        assert_eq!(
            scoped_payload["nodes"]
                .as_array()
                .unwrap()
                .iter()
                .any(|node| node["id"] == scope_edge_id),
            expected_visible,
            "security scope {requested_scope} had incorrect edge visibility: {scoped_payload}"
        );
    }

    for (groups, expected_group_nodes) in [(&[][..], false), (fixture_groups.as_slice(), true)] {
        let shared_response = application
            .graph_read_effect(&visualization_request(
                acl_fixture["truth_graph"].as_str().unwrap(),
                "facts",
                Some(scope_value),
                fixture_principal_id,
                groups,
            ))
            .await;
        assert_eq!(shared_response.status, StatusCode::OK.as_u16());
        let shared_payload: Value = serde_json::from_slice(&shared_response.body).unwrap();
        let visible_nodes = shared_payload["nodes"].to_string();
        for share_policy in &share_policies {
            let should_be_visible = share_policy["mode"] == "shared" || expected_group_nodes;
            assert_eq!(
                visible_nodes.contains(share_policy["id"].as_str().unwrap()),
                should_be_visible,
                "principal groups {groups:?} had incorrect share visibility: {shared_payload}"
            );
        }
        for share_policy in &edge_share_policies {
            let should_be_visible = share_policy["mode"] == "shared" || expected_group_nodes;
            assert_eq!(
                visible_nodes.contains(share_policy["id"].as_str().unwrap()),
                should_be_visible,
                "principal groups {groups:?} had incorrect edge-share visibility: {shared_payload}"
            );
        }
    }

    let deferred_visualization = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/viz/d3.json?graph_type=conversation")
                .header(
                    "authorization",
                    format!(
                        "Bearer {}",
                        token_for_graph_space("docs", &["read_graph"], "facts")
                    ),
                )
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(deferred_visualization.status(), StatusCode::OK);
    let visualization: Value = serde_json::from_slice(
        &to_bytes(deferred_visualization.into_body(), 65_536)
            .await
            .unwrap(),
    )
    .unwrap();
    assert!(
        visualization["nodes"]
            .to_string()
            .contains("conversation-node"),
        "public visualization failed to preserve graph and insertion-method selection: {visualization}"
    );

    for (path, element_path) in [
        ("/api/viz/d3.json?graph_type=knowledge&mode=reify", "nodes"),
        (
            "/api/viz/cytoscape.json?graph_type=knowledge&mode=reify",
            "elements",
        ),
    ] {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri(path)
                    .header(
                        "authorization",
                        format!("Bearer {}", token("docs", &["read_graph"])),
                    )
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK, "route {path}");
        let visualization: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 65_536).await.unwrap()).unwrap();
        let ids = visualization[element_path]
            .as_array()
            .unwrap()
            .iter()
            .map(|item| {
                if element_path == "elements" {
                    item["data"]["id"].as_str().unwrap()
                } else {
                    item["id"].as_str().unwrap()
                }
            })
            .collect::<Vec<_>>();
        assert!(ids.contains(&shared_entity_id), "{path}: {ids:?}");
        assert!(ids.iter().any(|id| id.starts_with("n1")), "{path}: {ids:?}");
        assert!(ids.iter().any(|id| id.starts_with("n2")), "{path}: {ids:?}");
        assert!(ids.iter().any(|id| id.starts_with("e1")), "{path}: {ids:?}");
        assert!(
            ids.iter().all(|id| !id.contains("n3")),
            "private graph records must not be exposed by {path}: {ids:?}"
        );
    }

    let bundle = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/viz/d3.bundle?graph_type=knowledge&mode=reify")
                .header(
                    "authorization",
                    format!("Bearer {}", token("docs", &["read_graph"])),
                )
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(bundle.status(), StatusCode::OK);
    assert_eq!(bundle.headers()["content-type"], "text/html; charset=utf-8");
    let bundle = String::from_utf8(
        to_bytes(bundle.into_body(), 1_048_576)
            .await
            .unwrap()
            .to_vec(),
    )
    .unwrap();
    let embedded = bundle
        .split_once("window.__EMBEDDED_DATA__ = ")
        .and_then(|(_, rest)| rest.split_once(';'))
        .map(|(payload, _)| serde_json::from_str::<Value>(payload).unwrap())
        .expect("bundle includes valid embedded graph data");
    let embedded_node_ids = embedded["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|node| node["id"].as_str())
        .collect::<Vec<_>>();
    assert!(embedded_node_ids.contains(&"n1"));
    assert!(embedded_node_ids.contains(&"n2"));
    assert!(embedded_node_ids.contains(&"e1"));
    assert!(!embedded_node_ids.contains(&"n3"));

    let read = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {}", token("docs", &["read_graph"])))
                .body(Body::from(
                    r#"{"jsonrpc":"2.0","id":"neighbors","method":"tools/call","params":{"name":"kg_neighbors","arguments":{"rid":"n1","truth_graph":"knowledge"}}}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(read.status(), StatusCode::OK);
    let read: Value =
        serde_json::from_slice(&to_bytes(read.into_body(), 65_536).await.unwrap()).unwrap();
    assert_eq!(read["result"]["nodes"], json!(["n2"]), "response: {read}");
    assert_eq!(read["result"]["edges"], json!(["e1"]));

    let standard = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp")
                .header("content-type", "application/json")
                .header("accept", "application/json, text/event-stream")
                .header("authorization", format!("Bearer {}", token("docs", &["read_graph"])))
                .body(Body::from(
                    r#"{"jsonrpc":"2.0","id":"neighbors-standard","method":"tools/call","params":{"name":"kg_neighbors","arguments":{"rid":"n1","truth_graph":"knowledge"}}}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(standard.status(), StatusCode::OK);
    assert_eq!(
        standard
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok()),
        Some("text/event-stream; charset=utf-8")
    );
    let body = to_bytes(standard.into_body(), 65_536).await.unwrap();
    let event = std::str::from_utf8(&body)
        .unwrap()
        .lines()
        .find_map(|line| line.strip_prefix("data: "))
        .expect("standard MCP response has an SSE data frame");
    let standard: Value = serde_json::from_str(event).unwrap();
    assert_eq!(
        standard["result"]["structuredContent"]["nodes"],
        json!(["n2"])
    );
    assert_eq!(
        standard["result"]["structuredContent"]["edges"],
        json!(["e1"])
    );
    assert_eq!(
        serde_json::from_str::<Value>(standard["result"]["content"][0]["text"].as_str().unwrap())
            .unwrap(),
        standard["result"]["structuredContent"]
    );

    for (tool, arguments, expected) in [
        (
            "kg_find_edges",
            json!({"relation":"links","src_label_contains":"one","tgt_label_contains":"two"}),
            json!({"edges":["e1"]}),
        ),
        (
            "kg_shortest_path",
            json!({"src_id":"n1","dst_id":"n2","max_depth":4}),
            json!({"path":["n1","e1","n2"]}),
        ),
        (
            "kg_k_hop",
            json!({"start_ids":["n1"],"k":1}),
            json!({"layers":[{"nodes":[],"edges":["e1"]}]}),
        ),
    ] {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header(
                        "authorization",
                        format!("Bearer {}", token("docs", &["read_graph"])),
                    )
                    .body(Body::from(
                        json!({
                            "jsonrpc":"2.0",
                            "id":tool,
                            "method":"tools/call",
                            "params":{"name":tool,"arguments":arguments}
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let response: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 65_536).await.unwrap()).unwrap();
        assert_eq!(response["result"]["implementation_authority"], "rust");
        assert_eq!(response["result"]["seam_attestation_version"], 1);
        assert_eq!(response["result"]["seam_crossings"], json!([]));
        let mut result = response["result"].clone();
        for field in [
            "implementation_authority",
            "seam_attestation_version",
            "seam_crossings",
        ] {
            result.as_object_mut().unwrap().remove(field);
        }
        assert_eq!(result, expected, "tool={tool}");
    }

    for (tool, key) in [
        ("kg_viz_d3_json", "nodes"),
        ("kg_viz_cytoscape_json", "elements"),
    ] {
        let visualization = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header(
                        "authorization",
                        format!("Bearer {}", token("docs", &["read_graph"])),
                    )
                    .body(Body::from(format!(
                        "{{\"jsonrpc\":\"2.0\",\"id\":\"{tool}\",\"method\":\"tools/call\",\"params\":{{\"name\":\"{tool}\",\"arguments\":{{\"doc_id\":null,\"mode\":\"reify\"}}}}}}"
                    )))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(visualization.status(), StatusCode::OK);
        let visualization: Value =
            serde_json::from_slice(&to_bytes(visualization.into_body(), 65_536).await.unwrap())
                .unwrap();
        let rows = visualization["result"][key].as_array().unwrap();
        let rendered = rows.iter().map(Value::to_string).collect::<String>();
        assert!(rendered.contains("n1") && rendered.contains("n2"));
        assert!(
            !rendered.contains("n3"),
            "private node leaked: {visualization}"
        );
    }

    let proposal_fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/graph-vector-proposal-acl-v1.json"
    )))
    .unwrap();
    let proposals = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!("Bearer {}", token("docs", &["read_graph"])),
                )
                .body(Body::from(
                    json!({
                        "jsonrpc":"2.0",
                        "id":"propose-vector",
                        "method":"tools/call",
                        "params":{
                            "name":"propose_vector",
                            "arguments":proposal_fixture["request"]
                        }
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(proposals.status(), StatusCode::OK);
    let proposals: Value =
        serde_json::from_slice(&to_bytes(proposals.into_body(), 65_536).await.unwrap()).unwrap();
    assert!(proposals.get("error").is_none(), "{proposals}");
    assert_eq!(proposals["result"]["implementation_authority"], "rust");
    assert_eq!(proposals["result"]["seam_crossings"], json!([]));
    assert_eq!(
        proposals["result"]["pairs"],
        proposal_fixture["expected_pairs"]
    );

    let vector_query = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/internal/graph/vector-query")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!("Bearer {}", token("docs", &["read_graph"])),
                )
                .body(Body::from(
                    json!({
                        "entity_kind": "node",
                        "query_embeddings": [[1.0, 0.0], [0.0, 1.0]],
                        "metric": "cosine",
                        "truth_graph": "knowledge",
                        "n_results": 2,
                        "where": {},
                        "include": ["distances"]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(vector_query.status(), StatusCode::OK);
    let vector_query: Value =
        serde_json::from_slice(&to_bytes(vector_query.into_body(), 65_536).await.unwrap()).unwrap();
    // The nearest ACL-denied node consumes a top-K slot; Rust matches Python
    // by filtering it without backfilling from farther candidates.
    assert_eq!(vector_query["ids"], json!([["n1"], ["n2"]]));
    assert_eq!(vector_query["distances"][0].as_array().unwrap().len(), 1);
    assert_eq!(vector_query["distances"][1].as_array().unwrap().len(), 1);
    assert!(vector_query.get("documents").is_none());
}

// PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/tests/part_08.rs#postgres_visualization_read_matches_shared_python_selection_and_acl; python=kogwistar/server_mcp_with_admin.py#api_viz_d3; fixture=contracts/golden/graph-acl-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/part_08.rs#postgres_visualization_read_matches_shared_python_selection_and_acl; python_test=tests/test_visualization.py#test_visualization_uses_acl_policy_for_selected_python_graph_type
#[tokio::test]
#[ignore = "requires KOGWISTAR_TEST_PG_DSN; uses a disposable PostgreSQL schema"]
async fn postgres_visualization_read_matches_shared_python_selection_and_acl() {
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
    use tower::ServiceExt;

    let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
        .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
    let schema = format!("rust_viz_{}", uuid::Uuid::new_v4().simple());
    let knowledge_schema = format!("{schema}_knowledge");
    let conversation_schema = format!("{schema}_conversation");
    let workflow_schema = format!("{schema}_workflow");
    let wisdom_schema = format!("{schema}_wisdom");
    let service = PostgresRunApplicationService::from_dsn_with_graph_schemas(
        &dsn,
        &knowledge_schema,
        &conversation_schema,
        &workflow_schema,
        &wisdom_schema,
    )
    .unwrap();
    service.ensure_schema().await.unwrap();
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

    let fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/visualization-insertion-filter-v1.json"
    )))
    .unwrap();
    let scope = GraphScope {
        namespace: "viz-test".to_owned(),
        workspace_id: None,
        graph_space: Some("facts".to_owned()),
    };
    let mut entities = fixture["nodes"].as_array().unwrap().clone();
    entities.push(json!({
        "id": "n-private",
        "references": [{"doc_id": "d1", "insertion_method": "parser"}]
    }));
    for entity in &entities {
        let id = entity["id"].as_str().unwrap();
        let metadata = Map::from_iter([
            ("entity_type".to_owned(), json!("node")),
            ("doc_id".to_owned(), fixture["doc_id"].clone()),
            ("lifecycle_status".to_owned(), json!("active")),
        ]);
        service
            .store
            .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                record: GraphRecord {
                    id: id.to_owned(),
                    document: Some(
                        json!({
                            "id": id,
                            "label": id,
                            "references": entity["references"]
                        })
                        .to_string(),
                    ),
                    metadata,
                    embedding: None,
                },
                embedding_dim: 3,
            })
            .await
            .unwrap();
        let acl = if id == "n-private" {
            json!({
                "entity_type": "acl_record",
                "acl_truth_graph": "knowledge",
                "acl_target_grain": "node",
                "acl_target_entity_id": id,
                "acl_mode": "private",
                "owner_id": "bob",
                "acl_version": 1,
                "tombstoned": false
            })
        } else {
            json!({
                "entity_type": "acl_record",
                "acl_truth_graph": "knowledge",
                "acl_target_grain": "node",
                "acl_target_entity_id": id,
                "acl_mode": "public",
                "acl_version": 1,
                "tombstoned": false
            })
        };
        service
            .store
            .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                record: GraphRecord {
                    id: format!("acl|{id}"),
                    document: Some("{}".to_owned()),
                    metadata: acl.as_object().unwrap().clone(),
                    embedding: None,
                },
                embedding_dim: 3,
            })
            .await
            .unwrap();
    }

    service
        .conversation_store
        .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
            scope: scope.clone(),
            table: "gke_nodes".to_owned(),
            record: GraphRecord {
                id: "conversation-only-node".to_owned(),
                document: Some(
                    json!({"id":"conversation-only-node","label":"conversation-only-node"})
                        .to_string(),
                ),
                metadata: Map::from_iter([
                    ("entity_type".to_owned(), json!("node")),
                    ("lifecycle_status".to_owned(), json!("active")),
                ]),
                embedding: None,
            },
            embedding_dim: 3,
        })
        .await
        .unwrap();
    service
        .conversation_store
        .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
            scope: scope.clone(),
            table: "gke_nodes".to_owned(),
            record: GraphRecord {
                id: "acl|conversation-only-node".to_owned(),
                document: Some("{}".to_owned()),
                metadata: Map::from_iter([
                    ("entity_type".to_owned(), json!("acl_record")),
                    ("acl_truth_graph".to_owned(), json!("conversation")),
                    ("acl_target_grain".to_owned(), json!("node")),
                    (
                        "acl_target_entity_id".to_owned(),
                        json!("conversation-only-node"),
                    ),
                    ("acl_mode".to_owned(), json!("public")),
                    ("acl_version".to_owned(), json!(1)),
                    ("tombstoned".to_owned(), json!(false)),
                ]),
                embedding: None,
            },
            embedding_dim: 3,
        })
        .await
        .unwrap();

    let acl_fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/graph-acl-v1.json"
    )))
    .unwrap();
    let isolation = &acl_fixture["graph_type_isolation"];
    let isolation_id = isolation["entity_id"].as_str().unwrap();
    for (store, case) in [
        (&service.store, &isolation["cases"][0]),
        (&service.conversation_store, &isolation["cases"][1]),
        (&service.workflow_store, &isolation["cases"][2]),
        (&service.wisdom_store, &isolation["cases"][3]),
    ] {
        let graph_type = case["truth_graph"].as_str().unwrap();
        store
            .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                record: GraphRecord {
                    id: isolation_id.to_owned(),
                    document: Some(json!({"id": isolation_id, "label": isolation_id}).to_string()),
                    metadata: Map::from_iter([
                        ("entity_type".to_owned(), json!("node")),
                        ("lifecycle_status".to_owned(), json!("active")),
                    ]),
                    embedding: None,
                },
                embedding_dim: 3,
            })
            .await
            .unwrap();
        let mut acl = Map::from_iter([
            ("entity_type".to_owned(), json!("acl_record")),
            ("acl_truth_graph".to_owned(), json!(graph_type)),
            ("acl_target_grain".to_owned(), json!("node")),
            ("acl_target_entity_id".to_owned(), json!(isolation_id)),
            ("acl_mode".to_owned(), case["mode"].clone()),
            ("acl_version".to_owned(), json!(1)),
            ("tombstoned".to_owned(), json!(false)),
        ]);
        if let Some(owner_id) = case.get("owner_id") {
            acl.insert("owner_id".to_owned(), owner_id.clone());
        }
        store
            .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                record: GraphRecord {
                    id: format!("acl|{isolation_id}"),
                    document: Some("{}".to_owned()),
                    metadata: acl,
                    embedding: None,
                },
                embedding_dim: 3,
            })
            .await
            .unwrap();
    }

    for case in isolation["cases"].as_array().unwrap() {
        let graph_type = case["truth_graph"].as_str().unwrap();
        let result = ApplicationService::execute(
            &service,
            ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: INTERNAL_GRAPH_VIZ_D3_ROUTE.to_owned(),
                body: serde_json::to_vec(&json!({"graph_type": graph_type})).unwrap(),
                principal: json!({
                    "sub": "alice",
                    "user_id": "alice",
                    "storage_ns": "viz-test",
                    "ns": ["viz-test"],
                    "graph_space": "facts",
                    "capabilities": ["read_graph"]
                }),
            },
        )
        .await;
        assert_eq!(result.status, StatusCode::OK.as_u16(), "{graph_type}");
        let payload: Value = serde_json::from_slice(&result.body).unwrap();
        let visible = payload["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .any(|node| node["id"] == isolation_id);
        assert_eq!(
            visible,
            case["expected_visible"].as_bool().unwrap(),
            "graph type {graph_type} must select its own graph and ACL policy"
        );
    }

    let conversation_result = ApplicationService::execute(
        &service,
        ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: INTERNAL_GRAPH_VIZ_D3_ROUTE.to_owned(),
            body: serde_json::to_vec(&json!({"graph_type":"conversation"})).unwrap(),
            principal: json!({
                "sub": "alice",
                "user_id": "alice",
                "storage_ns": "viz-test",
                "ns": ["viz-test"],
                "graph_space": "facts",
                "capabilities": ["read_graph"]
            }),
        },
    )
    .await;
    assert_eq!(conversation_result.status, StatusCode::OK.as_u16());
    let conversation_payload: Value = serde_json::from_slice(&conversation_result.body).unwrap();
    assert!(
        conversation_payload["nodes"]
            .to_string()
            .contains("conversation-only-node")
    );
    assert!(!conversation_payload["nodes"].to_string().contains("n1"));

    let response = ApplicationService::execute(
        &service,
        ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: INTERNAL_GRAPH_VIZ_D3_ROUTE.to_owned(),
            body: serde_json::to_vec(&json!({
                "doc_id": fixture["doc_id"],
                "mode": "reify",
                "insertion_method": fixture["insertion_method"],
                "graph_type": "knowledge"
            }))
            .unwrap(),
            principal: json!({
                "sub": "alice",
                "user_id": "alice",
                "storage_ns": "viz-test",
                "ns": ["viz-test"],
                "graph_space": "facts",
                "capabilities": ["read_graph"]
            }),
        },
    )
    .await;
    assert_eq!(response.status, StatusCode::OK.as_u16());
    let payload: Value = serde_json::from_slice(&response.body).unwrap();
    let mut returned_ids = payload["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|node| node["id"].as_str())
        .collect::<Vec<_>>();
    returned_ids.sort_unstable();
    assert_eq!(
        returned_ids,
        fixture["expected_matches_by_kind"]["node"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_str().unwrap())
            .collect::<Vec<_>>()
    );

    let secret = "postgres-visualization-public-route-test-secret";
    let auth = AuthConfig {
        mode: "jwt".to_owned(),
        algorithm: Some("HS256".to_owned()),
        key: Some(secret.to_owned()),
        ..AuthConfig::default()
    };
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let token = |capabilities: &[&str]| {
        encode(
            &Header::new(Algorithm::HS256),
            &json!({
                "sub": "alice",
                "role": "rw",
                "ns": ["viz-test"],
                "storage_ns": "viz-test",
                "graph_space": "facts",
                "capabilities": capabilities,
                "exp": now + 300
            }),
            &EncodingKey::from_secret(secret.as_bytes()),
        )
        .unwrap()
    };
    let app = router_with_application(
        ApiState {
            health: snapshot(),
            required_roles: vec![],
            implementation: ImplementationSnapshot::default(),
            auth,
        },
        Arc::new(service.clone()),
    );
    for case in isolation["cases"].as_array().unwrap() {
        let graph_type = case["truth_graph"].as_str().unwrap();
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri(format!("/api/viz/d3.json?graph_type={graph_type}"))
                    .header(
                        "authorization",
                        format!("Bearer {}", token(&["read_graph"])),
                    )
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK, "{graph_type}");
        let payload: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 65_536).await.unwrap()).unwrap();
        let visible = payload["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .any(|node| node["id"] == isolation_id);
        assert_eq!(
            visible,
            case["expected_visible"].as_bool().unwrap(),
            "authenticated HTTP graph type {graph_type} must select its own ACL policy"
        );
    }

    let query = format!(
        "doc_id={}&mode=reify&insertion_method={}&graph_type=knowledge",
        fixture["doc_id"].as_str().unwrap(),
        fixture["insertion_method"].as_str().unwrap()
    );
    let mut expected_node_ids = fixture["expected_matches_by_kind"]["node"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    expected_node_ids.sort_unstable();
    for (path, result_key) in [
        ("/api/viz/d3.json", "nodes"),
        ("/api/viz/cytoscape.json", "elements"),
    ] {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri(format!("{path}?{query}"))
                    .header(
                        "authorization",
                        format!("Bearer {}", token(&["read_graph"])),
                    )
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK, "{path}");
        let payload: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 65_536).await.unwrap()).unwrap();
        let ids = payload[result_key]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|item| {
                item.get("id")
                    .and_then(Value::as_str)
                    .or_else(|| item.get("data").and_then(|data| data["id"].as_str()))
            })
            .collect::<Vec<_>>();
        let mut returned_node_ids = ids
            .iter()
            .copied()
            .filter(|id| id.starts_with('n'))
            .collect::<Vec<_>>();
        returned_node_ids.sort_unstable();
        assert_eq!(returned_node_ids, expected_node_ids, "{path}");
        assert!(!ids.contains(&"n-private"), "{path}: {ids:?}");
    }

    let bundle = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/viz/d3.bundle?{query}"))
                .header(
                    "authorization",
                    format!("Bearer {}", token(&["read_graph"])),
                )
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(bundle.status(), StatusCode::OK);
    assert_eq!(bundle.headers()["content-type"], "text/html; charset=utf-8");
    let bundle = String::from_utf8(
        to_bytes(bundle.into_body(), 1_048_576)
            .await
            .unwrap()
            .to_vec(),
    )
    .unwrap();
    let embedded = bundle
        .split_once("window.__EMBEDDED_DATA__ = ")
        .and_then(|(_, rest)| rest.split_once(';'))
        .map(|(payload, _)| serde_json::from_str::<Value>(payload).unwrap())
        .expect("public bundle contains serialized graph data");
    let bundle_ids = embedded["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|node| node["id"].as_str())
        .collect::<Vec<_>>();
    let mut returned_bundle_node_ids = bundle_ids
        .iter()
        .copied()
        .filter(|id| id.starts_with('n'))
        .collect::<Vec<_>>();
    returned_bundle_node_ids.sort_unstable();
    assert_eq!(returned_bundle_node_ids, expected_node_ids);
    assert!(!bundle_ids.contains(&"n-private"));

    for path in [
        format!("/api/viz/d3.json?{query}"),
        format!("/api/viz/cytoscape.json?{query}"),
        format!("/viz/d3.bundle?{query}"),
    ] {
        let denied = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri(path)
                    .header("authorization", format!("Bearer {}", token(&[])))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(denied.status(), StatusCode::FORBIDDEN);
    }

    drop(service);
    let (cleanup, connection) = tokio_postgres::connect(&dsn, tokio_postgres::NoTls)
        .await
        .unwrap();
    let cleanup_task = tokio::spawn(connection);
    cleanup
        .batch_execute(&format!(
            "DROP SCHEMA \"{knowledge_schema}\" CASCADE; \
             DROP SCHEMA \"{conversation_schema}\" CASCADE; \
             DROP SCHEMA \"{workflow_schema}\" CASCADE; \
             DROP SCHEMA \"{wisdom_schema}\" CASCADE"
        ))
        .await
        .unwrap();
    drop(cleanup);
    cleanup_task.await.unwrap().unwrap();
}

#[tokio::test]
async fn in_memory_vector_query_matches_shared_python_acl_contract() {
    let fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/graph-acl-vector-query-v1.json"
    )))
    .unwrap();
    assert_eq!(fixture["schema_version"], 1);

    let secret = "in-memory-vector-acl-contract-secret";
    let auth = AuthConfig {
        mode: "jwt".to_owned(),
        algorithm: Some("HS256".to_owned()),
        key: Some(secret.to_owned()),
        ..AuthConfig::default()
    };
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let token = |subject: &str, capabilities: &[&str], scope_parts: Option<&Value>| {
        let mut claims = json!({
            "sub": subject,
            "role": "rw",
            "ns": ["docs"],
            "storage_ns": fixture["scope"]["namespace"],
            "workspace_id": fixture["scope"]["workspace_id"],
            "capabilities": capabilities,
            "exp": now + 300
        });
        if let Some(scope_parts) = scope_parts {
            for key in ["tenant", "workspace", "project"] {
                if let Some(value) = scope_parts.get(key) {
                    claims[key] = value.clone();
                }
            }
        }
        encode(
            &Header::new(Algorithm::HS256),
            &claims,
            &EncodingKey::from_secret(secret.as_bytes()),
        )
        .unwrap()
    };
    let app = router_with_application(
        ApiState {
            health: snapshot(),
            required_roles: vec![],
            implementation: ImplementationSnapshot::default(),
            auth,
        },
        Arc::new(InMemoryGraphApplicationService::default()),
    );
    let scope = json!({
        "namespace": fixture["scope"]["namespace"],
        "workspace_id": fixture["scope"]["workspace_id"],
        "graph_space": null
    });
    let mut mutations = Vec::new();
    for (kind, table, records) in [
        ("node", "gke_nodes", &fixture["nodes"]),
        ("edge", "gke_edges", &fixture["edges"]),
    ] {
        for spec in records.as_array().unwrap() {
            let id = spec["id"].as_str().unwrap();
            let mut metadata = serde_json::Map::new();
            metadata.insert("entity_type".to_owned(), json!(kind));
            if kind == "node" {
                metadata.insert("label".to_owned(), json!(id));
            } else {
                metadata.insert("source_ids".to_owned(), spec["source_ids"].clone());
                metadata.insert("target_ids".to_owned(), spec["target_ids"].clone());
            }
            if let Some(lifecycle_status) = spec["lifecycle_status"].as_str() {
                metadata.insert("lifecycle_status".to_owned(), json!(lifecycle_status));
            }
            mutations.push(json!({
                "scope": scope,
                "table": table,
                "entity_kind": kind,
                "event_id": format!("{kind}-{id}"),
                "op": "ADD",
                "payload": {"id": id},
                "record": {
                    "id": id,
                    "document": id,
                    "metadata": metadata,
                    "embedding": spec["embedding"]
                },
                "embedding_dim": 2
            }));

            let mut acl_metadata = serde_json::Map::new();
            for (key, value) in [
                ("entity_type", json!("acl_record")),
                ("acl_truth_graph", fixture["truth_graph"].clone()),
                ("acl_target_entity_id", json!(id)),
                ("acl_target_grain", json!(kind)),
                ("acl_version", json!(1)),
                ("acl_mode", spec["mode"].clone()),
            ] {
                acl_metadata.insert(key.to_owned(), value);
            }
            if let Some(owner_id) = spec["owner_id"].as_str() {
                acl_metadata.insert("owner_id".to_owned(), json!(owner_id));
            }
            if let Some(security_scope) = spec["security_scope"].as_str() {
                acl_metadata.insert("security_scope".to_owned(), json!(security_scope));
            }
            let acl_id = format!("acl-{kind}-{id}");
            mutations.push(json!({
                "scope": scope,
                "table": "gke_nodes",
                "entity_kind": "node",
                "event_id": format!("event-{acl_id}"),
                "op": "ADD",
                "payload": {"id": acl_id},
                "record": {
                    "id": acl_id,
                    "document": null,
                    "metadata": acl_metadata,
                    "embedding": null
                },
                "embedding_dim": 0
            }));
        }
    }
    let write = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/internal/graph/mutations")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!("Bearer {}", token("owner-b", &["write_graph"], None)),
                )
                .body(Body::from(
                    json!({"contract_version":1,"atomic":true,"mutations":mutations,"index_jobs":[]})
                        .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(write.status(), StatusCode::OK);

    for principal in fixture["principals"].as_array().unwrap() {
        let subject = principal["id"].as_str().unwrap();
        for (kind, expected_key) in [("node", "nodes"), ("edge", "edges")] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/internal/graph/vector-query")
                        .header("content-type", "application/json")
                        .header(
                            "authorization",
                            format!(
                                "Bearer {}",
                                token(subject, &["read_graph"], principal.get("scope_parts"))
                            ),
                        )
                        .body(Body::from(
                            json!({
                                "entity_kind": kind,
                                "query_embeddings": fixture["query_embeddings"],
                                "metric": fixture["metric"],
                                "truth_graph": fixture["truth_graph"],
                                "n_results": fixture["n_results"],
                                "include": ["distances"]
                            })
                            .to_string(),
                        ))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let result: Value =
                serde_json::from_slice(&to_bytes(response.into_body(), 65_536).await.unwrap())
                    .unwrap();
            assert_eq!(result["implementation_authority"], "rust");
            assert_eq!(result["seam_attestation_version"], 1);
            assert_eq!(result["seam_crossings"], json!([]));
            assert_eq!(
                result["ids"], principal["expected"][expected_key],
                "unexpected {kind} candidates for {subject}"
            );
            let expected_distances = &principal["expected"]["distances"][expected_key];
            let actual_distances = result["distances"].as_array().unwrap();
            assert_eq!(
                actual_distances.len(),
                expected_distances.as_array().unwrap().len()
            );
            for (actual, expected_values) in actual_distances
                .iter()
                .zip(expected_distances.as_array().unwrap())
            {
                let actual = actual.as_array().unwrap();
                let expected_values = expected_values.as_array().unwrap();
                assert_eq!(actual.len(), expected_values.len());
                for (actual, expected) in actual.iter().zip(expected_values) {
                    assert!(
                        (actual.as_f64().unwrap() - expected.as_f64().unwrap()).abs() < 1e-5,
                        "unexpected {kind} vector distance for {subject}"
                    );
                }
            }
            if subject == "reader-a" && kind == "node" {
                let response = app
                    .clone()
                    .oneshot(
                        Request::builder()
                            .method("POST")
                            .uri("/internal/graph/vector-query")
                            .header("content-type", "application/json")
                            .header(
                                "authorization",
                                format!("Bearer {}", token(subject, &["read_graph"], None)),
                            )
                            .body(Body::from(
                                json!({
                                    "entity_kind": "node",
                                    "query_embeddings": [fixture["query_embeddings"][0]],
                                    "metric": fixture["metric"],
                                    "truth_graph": fixture["truth_graph"],
                                    "n_results": 1,
                                    "include": ["distances"]
                                })
                                .to_string(),
                            ))
                            .unwrap(),
                    )
                    .await
                    .unwrap();
                assert_eq!(response.status(), StatusCode::OK);
                let result: Value =
                    serde_json::from_slice(&to_bytes(response.into_body(), 65_536).await.unwrap())
                        .unwrap();
                assert_eq!(result["ids"], json!([[]]));
            }
        }
    }
    for (kind, expected_ids) in [
        ("node", json!([["near-private", "visible-scope"]])),
        ("edge", json!([["edge-private-endpoint", "edge-scope"]])),
    ] {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/graph/vector-query")
                    .header("content-type", "application/json")
                    .header(
                        "authorization",
                        format!("Bearer {}", token("owner-b", &["read_graph"], None)),
                    )
                    .body(Body::from(
                        json!({
                            "entity_kind": kind,
                            "query_embeddings": [fixture["query_embeddings"][0]],
                            "metric": fixture["metric"],
                            "truth_graph": fixture["truth_graph"],
                            "n_results": 2,
                            "include": ["distances"]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let result: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 65_536).await.unwrap()).unwrap();
        assert_eq!(result["ids"], expected_ids);
    }

    let revoked_node_id = "visible-public";
    let revoked_acl_id = format!("acl-node-{revoked_node_id}");
    let public_read = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/internal/graph/vector-query")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!("Bearer {}", token("reader-a", &["read_graph"], None)),
                )
                .body(Body::from(
                    json!({
                        "entity_kind": "node",
                        "query_embeddings": [fixture["query_embeddings"][0]],
                        "metric": fixture["metric"],
                        "truth_graph": fixture["truth_graph"],
                        "where": {"label": revoked_node_id},
                        "n_results": 1,
                        "include": ["distances"]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(public_read.status(), StatusCode::OK);
    let public_result: Value = serde_json::from_slice(
        &to_bytes(public_read.into_body(), 65_536).await.unwrap(),
    )
    .unwrap();
    assert_eq!(public_result["ids"], json!([[revoked_node_id]]));

    let revoke = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/internal/graph/mutations")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!("Bearer {}", token("owner-b", &["write_graph"], None)),
                )
                .body(Body::from(
                    json!({
                        "contract_version": 1,
                        "atomic": true,
                        "index_jobs": [],
                        "mutations": [{
                            "scope": scope,
                            "table": "gke_nodes",
                            "entity_kind": "node",
                            "event_id": "event-acl-node-visible-public-revoked",
                            "op": "REPLACE",
                            "payload": {"id": revoked_acl_id},
                            "record": {
                                "id": revoked_acl_id,
                                "document": null,
                                "metadata": {
                                    "entity_type": "acl_record",
                                    "acl_truth_graph": fixture["truth_graph"],
                                    "acl_target_entity_id": revoked_node_id,
                                    "acl_target_grain": "node",
                                    "acl_version": 2,
                                    "acl_mode": "private",
                                    "owner_id": "owner-b"
                                },
                                "embedding": null
                            },
                            "embedding_dim": 0
                        }]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(revoke.status(), StatusCode::OK);

    let revoked_read = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/internal/graph/vector-query")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!("Bearer {}", token("reader-a", &["read_graph"], None)),
                )
                .body(Body::from(
                    json!({
                        "entity_kind": "node",
                        "query_embeddings": [fixture["query_embeddings"][0]],
                        "metric": fixture["metric"],
                        "truth_graph": fixture["truth_graph"],
                        "where": {"label": revoked_node_id},
                        "n_results": 1,
                        "include": ["distances"]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(revoked_read.status(), StatusCode::OK);
    let revoked_result: Value = serde_json::from_slice(
        &to_bytes(revoked_read.into_body(), 65_536).await.unwrap(),
    )
    .unwrap();
    assert_eq!(revoked_result["ids"], json!([[]]));

    let item_node_id = "node-source-span-private";
    let item_grounding_id = format!("{item_node_id}::mention::0");
    let item_span_id = format!("{item_grounding_id}::span::0");
    let mut item_mutations = vec![json!({
        "scope": scope,
        "table": "gke_nodes",
        "entity_kind": "node",
        "event_id": item_node_id,
        "op": "ADD",
        "payload": {"id": item_node_id},
        "record": {
            "id": item_node_id,
            "document": item_node_id,
            "metadata": {
                "entity_type": "node",
                "label": item_node_id,
                "mentions": [{"spans": [{}]}]
            },
            "embedding": [1.0, 0.0]
        },
        "embedding_dim": 2
    })];
    for (event_id, grain, target_item_id, mode) in [
        ("acl-source-node", "node", None, "public"),
        (
            "acl-source-grounding",
            "grounding",
            Some(item_grounding_id.as_str()),
            "public",
        ),
        (
            "acl-source-span",
            "span",
            Some(item_span_id.as_str()),
            "private",
        ),
    ] {
        let mut acl_metadata = json!({
            "entity_type": "acl_record",
            "acl_truth_graph": fixture["truth_graph"],
            "acl_target_entity_id": item_node_id,
            "acl_target_grain": grain,
            "acl_version": 1,
            "acl_mode": mode
        });
        if let Some(target_item_id) = target_item_id {
            acl_metadata["acl_target_item_id"] = json!(target_item_id);
        }
        if mode == "private" {
            acl_metadata["owner_id"] = json!("owner-b");
        }
        item_mutations.push(json!({
            "scope": scope,
            "table": "gke_nodes",
            "entity_kind": "node",
            "event_id": event_id,
            "op": "ADD",
            "payload": {"id": event_id},
            "record": {
                "id": event_id,
                "document": null,
                "metadata": acl_metadata,
                "embedding": null
            },
            "embedding_dim": 0
        }));
    }
    let write = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/internal/graph/mutations")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!("Bearer {}", token("owner-b", &["write_graph"], None)),
                )
                .body(Body::from(
                    json!({"contract_version":1,"atomic":true,"mutations":item_mutations,"index_jobs":[]})
                        .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(write.status(), StatusCode::OK);
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/internal/graph/vector-query")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!("Bearer {}", token("reader-a", &["read_graph"], None)),
                )
                .body(Body::from(
                    json!({
                        "entity_kind": "node",
                        "query_embeddings": [[1.0, 0.0]],
                        "metric": "cosine",
                        "truth_graph": fixture["truth_graph"],
                        "where": {"label": item_node_id},
                        "n_results": 1,
                        "include": ["distances"]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let result: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), 65_536).await.unwrap()).unwrap();
    assert_eq!(result["ids"], json!([[]]));
}

#[tokio::test]
async fn sqlite_chroma_vector_query_matches_shared_python_acl_contract() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/graph-acl-vector-query-v1.json"
    )))
    .unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let fixture_for_server = fixture.clone();
    let (stop_server, mut stop_server_rx) = tokio::sync::oneshot::channel();
    let chroma_server = tokio::spawn(async move {
        loop {
            let (mut stream, _) = tokio::select! {
                _ = &mut stop_server_rx => break,
                connection = listener.accept() => connection.unwrap(),
            };
            let mut bytes = Vec::new();
            let mut chunk = [0_u8; 4096];
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
            .unwrap_or_else(|_| json!({}));
            let request_line = String::from_utf8_lossy(&bytes[..header_end])
                .lines()
                .next()
                .unwrap_or_default()
                .to_owned();
            let payload = if request_line.contains("/collections HTTP") {
                let name = request["name"].as_str().unwrap();
                json!({"id":format!("collection-{name}"),"metadata":{"hnsw:space":"cosine"}})
            } else {
                let kind = if request_line.contains("shared-acl-edges") {
                    "edges"
                } else {
                    "nodes"
                };
                if request_line.contains("/query HTTP") {
                    let candidates = fixture_for_server[kind].as_array().unwrap();
                    let query = request["query_embeddings"][0].as_array().unwrap();
                    let query = [query[0].as_f64().unwrap(), query[1].as_f64().unwrap()];
                    let mut ranked = candidates
                        .iter()
                        .filter(|candidate| candidate["lifecycle_status"] != "tombstoned")
                        .map(|candidate| {
                            let embedding = candidate["embedding"].as_array().unwrap();
                            let vector = [
                                embedding[0].as_f64().unwrap(),
                                embedding[1].as_f64().unwrap(),
                            ];
                            let dot = query[0] * vector[0] + query[1] * vector[1];
                            let q_norm = query[0].hypot(query[1]);
                            let v_norm = vector[0].hypot(vector[1]);
                            (candidate, 1.0 - dot / (q_norm * v_norm))
                        })
                        .collect::<Vec<_>>();
                    ranked.sort_by(|left, right| left.1.total_cmp(&right.1));
                    ranked.truncate(request["n_results"].as_u64().unwrap() as usize);
                    let ids = ranked
                        .iter()
                        .map(|(candidate, _)| candidate["id"].clone())
                        .collect::<Vec<_>>();
                    let documents = ids.clone();
                    let embeddings = ranked
                        .iter()
                        .map(|(candidate, _)| candidate["embedding"].clone())
                        .collect::<Vec<_>>();
                    let distances = ranked.iter().map(|(_, distance)| json!(distance)).collect::<Vec<_>>();
                    let metadatas = ranked
                        .iter()
                        .map(|(candidate, _)| {
                            let mut original = json!({
                                "entity_type":if kind == "nodes" {"node"} else {"edge"},
                                "workspace_id":fixture_for_server["scope"]["workspace_id"]
                            });
                            if kind == "nodes" {
                                original["label"] = candidate["id"].clone();
                                original["mentions"] = json!([]);
                            } else {
                                for key in ["source_ids", "target_ids"] {
                                    original[key] = candidate[key].clone();
                                }
                                original["source_edge_ids"] = json!([]);
                                original["target_edge_ids"] = json!([]);
                                original["mentions"] = json!([]);
                            }
                            json!({
                                "_kogwistar_namespace":fixture_for_server["scope"]["namespace"],
                                "_kogwistar_original_metadata":original.to_string()
                            })
                        })
                        .collect::<Vec<_>>();
                    json!({
                        "ids":[ids], "documents":[documents], "metadatas":[metadatas],
                        "embeddings":[embeddings], "distances":[distances]
                    })
                } else if request_line.contains("/get HTTP") {
                    let requested = request["ids"].as_array().cloned().unwrap_or_default();
                    let matching = fixture_for_server["nodes"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .filter(|node| requested.iter().any(|id| id == &node["id"]))
                        .collect::<Vec<_>>();
                    json!({
                        "ids":matching.iter().map(|node| node["id"].clone()).collect::<Vec<_>>(),
                        "documents":matching.iter().map(|node| node["id"].clone()).collect::<Vec<_>>(),
                        "metadatas":matching.iter().map(|node| json!({
                            "_kogwistar_namespace":fixture_for_server["scope"]["namespace"],
                            "_kogwistar_original_metadata":json!({
                                "entity_type":"node", "label":node["id"], "mentions":[],
                                "workspace_id":fixture_for_server["scope"]["workspace_id"]
                            }).to_string()
                        })).collect::<Vec<_>>(),
                        "embeddings":vec![Value::Null;matching.len()]
                    })
                } else {
                    json!({"ids":[],"documents":[],"metadatas":[],"embeddings":[]})
                }
            };
            let response = payload.to_string();
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
        chroma_config("shared-acl-nodes"),
        chroma_config("shared-acl-edges"),
    )
    .await
    .unwrap();
    let sqlite_path = std::env::temp_dir().join(format!(
        "kogwistar-sqlite-chroma-vector-contract-{}.sqlite",
        uuid::Uuid::new_v4().simple()
    ));
    let service = SqliteRunApplicationService::open(&sqlite_path)
        .unwrap()
        .with_chroma_graph(graph);
    let scope = json!({
        "namespace":fixture["scope"]["namespace"],
        "workspace_id":fixture["scope"]["workspace_id"],
        "graph_space":null
    });
    let mut acl_mutations = Vec::new();
    for (kind, records) in [("node", &fixture["nodes"]), ("edge", &fixture["edges"])] {
        for candidate in records.as_array().unwrap() {
            if candidate["lifecycle_status"] == "tombstoned" {
                continue;
            }
            let id = candidate["id"].as_str().unwrap();
            let mut metadata = json!({
                "entity_type":"acl_record", "acl_truth_graph":fixture["truth_graph"],
                "acl_target_entity_id":id, "acl_target_grain":kind,
                "acl_version":1, "acl_mode":candidate["mode"],
                "workspace_id":fixture["scope"]["workspace_id"]
            });
            for key in ["owner_id", "security_scope"] {
                if !candidate[key].is_null() {
                    metadata[key] = candidate[key].clone();
                }
            }
            let acl_id = format!("acl-{kind}-{id}");
            acl_mutations.push(json!({
                "scope":scope, "table":"gke_nodes", "entity_kind":"node",
                "event_id":format!("event-{acl_id}"), "op":"ADD",
                "payload":{"id":acl_id,"metadata":metadata},
                "record":{"id":acl_id,"document":null,"metadata":metadata,"embedding":null},
                "embedding_dim":0
            }));
        }
    }
    let seed = service.execute(ApiEffectRequest {
        contract_version: 1,
        method: "POST".to_owned(),
        path_and_query: "/internal/graph/mutations".to_owned(),
        body: serde_json::to_vec(&json!({
            "contract_version":1,"atomic":true,"mutations":acl_mutations,"index_jobs":[]
        }))
        .unwrap(),
        principal: json!({
            "sub":"fixture-owner", "role":"rw", "storage_ns":fixture["scope"]["namespace"],
            "workspace_id":fixture["scope"]["workspace_id"], "capabilities":["write_graph"]
        }),
    })
    .await;
    assert_eq!(seed.status, StatusCode::OK.as_u16());
    let visible_public_acl = service
        .store
        .immediate_transaction(|uow| {
            uow.graph_acl_records_for_entities(
                &kogwistar_store::GraphScope {
                    namespace: fixture["scope"]["namespace"].as_str().unwrap().to_owned(),
                    workspace_id: Some(
                        fixture["scope"]["workspace_id"]
                            .as_str()
                            .unwrap()
                            .to_owned(),
                    ),
                    graph_space: None,
                },
                fixture["truth_graph"].as_str().unwrap(),
                &["visible-public".to_owned()],
            )
        })
        .unwrap();
    assert_eq!(visible_public_acl.len(), 1);

    for principal_fixture in fixture["principals"].as_array().unwrap() {
        let subject = principal_fixture["id"].as_str().unwrap();
        let mut principal = json!({
            "sub":subject, "role":"ro", "storage_ns":fixture["scope"]["namespace"],
            "workspace_id":fixture["scope"]["workspace_id"], "capabilities":["read_graph"]
        });
        if let Some(scope_parts) = principal_fixture.get("scope_parts") {
            for key in ["tenant", "workspace", "project"] {
                if !scope_parts[key].is_null() {
                    principal[key] = scope_parts[key].clone();
                }
            }
        }
        for (kind, expected_key) in [("node", "nodes"), ("edge", "edges")] {
            let response = service
                .execute(ApiEffectRequest {
                    contract_version: 1,
                    method: "POST".to_owned(),
                    path_and_query: "/internal/graph/vector-query".to_owned(),
                    body: serde_json::to_vec(&json!({
                        "entity_kind":kind,
                        "query_embeddings":fixture["query_embeddings"],
                        "metric":fixture["metric"],
                        "truth_graph":fixture["truth_graph"],
                        "n_results":fixture["n_results"],
                        "include":["distances"]
                    }))
                    .unwrap(),
                    principal: principal.clone(),
                })
                .await;
            assert_eq!(response.status, StatusCode::OK.as_u16(), "{subject}/{kind}");
            let result: Value = serde_json::from_slice(&response.body).unwrap();
            assert_eq!(
                result["ids"], principal_fixture["expected"][expected_key],
                "unexpected {kind} candidates for {subject}: {result}"
            );
            for (actual_query, expected_query) in result["distances"]
                .as_array()
                .unwrap()
                .iter()
                .zip(principal_fixture["expected"]["distances"][expected_key].as_array().unwrap())
            {
                for (actual, expected) in actual_query.as_array().unwrap().iter().zip(expected_query.as_array().unwrap()) {
                    assert!(
                        (actual.as_f64().unwrap() - expected.as_f64().unwrap()).abs() < 1e-5,
                        "unexpected {kind} vector distance for {subject}"
                    );
                }
            }
        }
    }

    drop(service);
    let _ = stop_server.send(());
    chroma_server.await.unwrap();
    let _ = std::fs::remove_file(sqlite_path);
}

#[test]
fn postgres_graph_type_stores_follow_python_schema_partitioning() {
    let service = PostgresRunApplicationService::from_dsn_with_graph_schemas(
        "postgresql://user:password@localhost/kogwistar",
        "kg_knowledge",
        "kg_conversation",
        "kg_workflow",
        "kg_wisdom",
    )
    .expect("valid PostgreSQL configuration should construct graph stores without connecting");

    assert_eq!(
        service.graph_store_for_type("knowledge").schema(),
        "kg_knowledge"
    );
    assert_eq!(
        service.graph_store_for_type("conversation").schema(),
        "kg_conversation"
    );
    assert_eq!(
        service.graph_store_for_type("workflow").schema(),
        "kg_workflow"
    );
    assert_eq!(service.graph_store_for_type("wisdom").schema(), "kg_wisdom");
    assert_eq!(
        service.graph_store_for_type("CONVERSATION").schema(),
        "kg_conversation"
    );

    let legacy = PostgresRunApplicationService::from_dsn_with_wisdom_schema(
        "postgresql://user:password@localhost/kogwistar",
        "legacy_graph",
        "legacy_wisdom",
    )
    .expect("legacy constructor remains supported");
    assert_eq!(legacy.store.schema(), "legacy_graph");
    assert_eq!(
        legacy.conversation_store.schema(),
        "legacy_graph_conversation"
    );
    assert_eq!(legacy.workflow_store.schema(), "legacy_graph_workflow");
    assert_eq!(legacy.wisdom_store.schema(), "legacy_wisdom");
}

#[tokio::test]
#[ignore = "requires KOGWISTAR_TEST_PG_DSN; uses disposable PostgreSQL schemas"]
async fn postgres_kg_load_persisted_matches_document_and_method_lookup_contract() {
    use axum::{body::Body, http::Request};
    use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
    use tower::ServiceExt;

    let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
        .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
    let schema = format!("rust_load_persisted_{}", uuid::Uuid::new_v4().simple());
    let service = PostgresRunApplicationService::from_dsn(&dsn, &schema).unwrap();
    service.ensure_schema().await.unwrap();
    service
        .store
        .create_graph_schema(2, kogwistar_store_postgres::GraphTableNames::default())
        .await
        .unwrap();

    let scope = GraphScope {
        namespace: "docs".to_owned(),
        workspace_id: None,
        graph_space: None,
    };
    let public_acl = |id: &str, grain: &str, target: &str, item: Option<&str>| {
        let mut metadata = serde_json::Map::from_iter([
            ("entity_type".to_owned(), json!("acl_record")),
            ("acl_truth_graph".to_owned(), json!("knowledge")),
            ("acl_target_grain".to_owned(), json!(grain)),
            ("acl_target_entity_id".to_owned(), json!(target)),
            ("acl_version".to_owned(), json!(1)),
            ("acl_mode".to_owned(), json!("public")),
        ]);
        if let Some(item) = item {
            metadata.insert("acl_target_item_id".to_owned(), json!(item));
        }
        let document = serde_json::to_string(&metadata).unwrap();
        GraphMutation {
            scope: scope.clone(),
            table: "gke_nodes".to_owned(),
            entity_kind: "node".to_owned(),
            event_id: format!("{id}-event"),
            op: "ADD".to_owned(),
            payload: json!({"id":id,"metadata":metadata}),
            record: GraphRecord {
                id: id.to_owned(),
                document: Some(document),
                metadata,
                embedding: None,
            },
            embedding_dim: 2,
        }
    };
    let mention_document = |id: &str| {
        json!({
            "id": id,
            "mentions": [{"spans": [{"doc_id":"doc-a","insertion_method":"parser"}]}]
        })
    };
    for id in ["node-a", "node-b"] {
        let document = mention_document(id);
        service
            .store
            .apply_graph_mutation(GraphMutation {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                entity_kind: "node".to_owned(),
                event_id: format!("{id}-event"),
                op: "ADD".to_owned(),
                payload: document.clone(),
                record: GraphRecord {
                    id: id.to_owned(),
                    document: Some(document.to_string()),
                    metadata: serde_json::Map::new(),
                    embedding: None,
                },
                embedding_dim: 2,
            })
            .await
            .unwrap();
        service
            .store
            .apply_graph_mutation(public_acl(&format!("acl-{id}"), "node", id, None))
            .await
            .unwrap();
        let grounding_id = format!("{id}::mention::0");
        let span_id = format!("{grounding_id}::span::0");
        service
            .store
            .apply_graph_mutation(public_acl(
                &format!("acl-{grounding_id}"),
                "grounding",
                id,
                Some(&grounding_id),
            ))
            .await
            .unwrap();
        service
            .store
            .apply_graph_mutation(public_acl(
                &format!("acl-{span_id}"),
                "span",
                id,
                Some(&span_id),
            ))
            .await
            .unwrap();
    }
    let latest = service
        .store
        .latest_entity_events("docs", "node", &["node-a".to_owned()])
        .await
        .unwrap();
    let event = &latest["node-a"];
    let canonical = read_canonical_entity_revision(
        &[EntityEvent {
            namespace: event.namespace.clone(),
            seq: event.seq,
            event_id: event.event_id.clone(),
            entity_kind: event.entity_kind.clone(),
            entity_id: event.entity_id.clone(),
            op: event.op.clone(),
            payload: serde_json::from_str(&event.payload_json).unwrap(),
        }],
        "docs",
        "node",
        "node-a",
    )
    .unwrap()
    .unwrap();
    let fingerprint_payload =
        kogwistar_engine::canonical_revision::canonical_revision_job_payload(&canonical);
    let fingerprint =
        serde_json::from_str::<Value>(&fingerprint_payload).unwrap()["source_fingerprint"]
            .as_str()
            .unwrap()
            .to_owned();
    let staged_document = mention_document("node-a").to_string();
    let staged_metadata = json!({
        "doc_id":"doc-a",
        "entity_type":"node",
        "lifecycle_status":"active",
        "namespace":"docs"
    });
    let staged_metadata_json = staged_metadata.to_string();
    let (stage_setup, stage_connection) = tokio_postgres::connect(&dsn, tokio_postgres::NoTls)
        .await
        .unwrap();
    let stage_task = tokio::spawn(stage_connection);
    stage_setup
        .execute(
            &format!("DELETE FROM \"{schema}\".gke_node_docs WHERE metadata->>'node_id'=$1"),
            &[&"node-a"],
        )
        .await
        .unwrap();
    stage_setup
        .execute(
            &format!(
                "INSERT INTO \"{schema}\".gke_stage1_projections \
                 (namespace,entity_kind,entity_id,document,metadata,source_fingerprint,revision,materialization_status,updated_at_ms) \
                 VALUES ($1,'node',$2,$3,$4::text::jsonb,$5,$6,'pending',0)"
            ),
            &[
                &"docs",
                &"node-a",
                &staged_document,
                &staged_metadata_json,
                &fingerprint,
                &canonical.revision,
            ],
        )
        .await
        .unwrap();
    drop(stage_setup);
    stage_task.await.unwrap().unwrap();
    let edge_document = json!({
        "id":"edge-a",
        "mentions":[{"spans":[{"doc_id":"doc-a","insertion_method":"parser"}]}],
        "source_ids":["node-a"],
        "target_ids":["node-b"],
        "doc_id":"doc-a"
    });
    service
        .store
        .apply_graph_mutation(GraphMutation {
            scope: scope.clone(),
            table: "gke_edges".to_owned(),
            entity_kind: "edge".to_owned(),
            event_id: "edge-a-event".to_owned(),
            op: "ADD".to_owned(),
            payload: edge_document.clone(),
            record: GraphRecord {
                id: "edge-a".to_owned(),
                document: Some(edge_document.to_string()),
                metadata: serde_json::Map::new(),
                embedding: None,
            },
            embedding_dim: 2,
        })
        .await
        .unwrap();
    service
        .store
        .apply_graph_mutation(public_acl("acl-edge-a", "edge", "edge-a", None))
        .await
        .unwrap();
    let edge_grounding = "edge-a::mention::0";
    let edge_span = "edge-a::mention::0::span::0";
    service
        .store
        .apply_graph_mutation(public_acl(
            "acl-edge-grounding",
            "grounding",
            "edge-a",
            Some(edge_grounding),
        ))
        .await
        .unwrap();
    service
        .store
        .apply_graph_mutation(public_acl(
            "acl-edge-span",
            "span",
            "edge-a",
            Some(edge_span),
        ))
        .await
        .unwrap();

    let bearer = encode(
        &Header::new(Algorithm::HS256),
        &json!({
            "sub":"graph-reader","role":"ro","ns":["docs"],"storage_ns":"docs",
            "capabilities":["read_graph"],"exp":4102444800_u64
        }),
        &EncodingKey::from_secret(b"load-persisted-mcp-test"),
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
                key: Some("load-persisted-mcp-test".to_owned()),
                ..AuthConfig::default()
            },
        },
        Arc::new(service.clone()),
    );
    assert!(service.supports_mcp_tool_for_principal(
        "root",
        "kg_load_persisted",
        &json!({"sub":"graph-reader","role":"ro","ns":["docs"],"storage_ns":"docs","capabilities":["read_graph"]})
    ));
    assert!(service.supports_mcp_tool_for_principal(
        "root",
        "kg_propose_bruteforce",
        &json!({"sub":"graph-reader","role":"ro","ns":["docs"],"storage_ns":"docs","capabilities":["read_graph"]})
    ));
    assert!(!service.supports_mcp_tool_for_principal(
        "root",
        "kg_propose_bruteforce",
        &json!({"sub":"graph-reader","role":"ro","ns":["docs"],"storage_ns":"docs","capabilities":[]})
    ));
    let call = |id: &str, inp: Value| {
        Request::builder()
            .method("POST")
            .uri("/mcp")
            .header("content-type", "application/json")
            .header("authorization", format!("Bearer {bearer}"))
            .body(Body::from(
                json!({
                    "jsonrpc":"2.0","id":id,"method":"tools/call",
                    "params":{"name":"kg_load_persisted","arguments":{"inp":inp}}
                })
                .to_string(),
            ))
            .unwrap()
    };
    let short_response = app
        .clone()
        .oneshot(call("short", json!({"doc_ids":["doc-a"]})))
        .await
        .unwrap();
    assert_eq!(short_response.status(), StatusCode::OK);
    let short_body = axum::body::to_bytes(short_response.into_body(), 65_536)
        .await
        .unwrap();
    let short_json: Value = serde_json::from_slice(&short_body).unwrap();
    assert!(
        short_json.get("result").is_some(),
        "MCP response: {short_json}"
    );
    let map_key = crate::short_id_store::short_id_map_key_for_bearer(Some(&bearer));
    let decode_ids = |values: &[Value]| {
        values
            .iter()
            .map(|value| {
                crate::short_id_store::canonical_id_for(
                    ".shortids",
                    &map_key,
                    value.as_str().unwrap(),
                )
                .unwrap()
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(
        decode_ids(short_json["result"]["node_ids"].as_array().unwrap()),
        ["node-a", "node-b"]
    );
    assert_eq!(
        decode_ids(short_json["result"]["edge_ids"].as_array().unwrap()),
        ["edge-a"]
    );

    let canonical_response = app
        .clone()
        .oneshot(call(
            "canonical",
            json!({"doc_ids":["doc-a"],"insertion_method":"parser","sid":false}),
        ))
        .await
        .unwrap();
    let canonical_body = axum::body::to_bytes(canonical_response.into_body(), 65_536)
        .await
        .unwrap();
    let canonical_json: Value = serde_json::from_slice(&canonical_body).unwrap();
    assert_eq!(
        canonical_json["result"]["node_ids"],
        json!(["node-a", "node-b"])
    );
    assert_eq!(canonical_json["result"]["edge_ids"], json!(["edge-a"]));

    let proposal_call = |id: &str, input: Value| {
        Request::builder()
            .method("POST")
            .uri("/mcp")
            .header("content-type", "application/json")
            .header("authorization", format!("Bearer {bearer}"))
            .body(Body::from(
                json!({
                    "jsonrpc":"2.0","id":id,"method":"tools/call",
                    "params":{"name":"kg_propose_bruteforce","arguments":{"inp":input}}
                })
                .to_string(),
            ))
            .unwrap()
    };
    let brute_force_response = app
        .clone()
        .oneshot(proposal_call(
            "brute",
            json!({"pair_kind":"node_node","allowed_docs":["doc-a"]}),
        ))
        .await
        .unwrap();
    let brute_force_body = axum::body::to_bytes(brute_force_response.into_body(), 65_536)
        .await
        .unwrap();
    let brute_force_json: Value = serde_json::from_slice(&brute_force_body).unwrap();
    assert!(
        brute_force_json.get("result").is_some(),
        "MCP response: {brute_force_json}"
    );
    assert_eq!(
        brute_force_json["result"]["pairs"],
        json!([{
            "left_id":"node-a","left_kind":"node",
            "right_id":"node-b","right_kind":"node"
        }])
    );
    let mixed_response = app
        .clone()
        .oneshot(proposal_call(
            "mixed",
            json!({"pair_kind":"any_any","anchor_doc_id":"doc-a"}),
        ))
        .await
        .unwrap();
    let mixed_body = axum::body::to_bytes(mixed_response.into_body(), 65_536)
        .await
        .unwrap();
    let mixed_json: Value = serde_json::from_slice(&mixed_body).unwrap();
    assert!(
        mixed_json.get("result").is_some(),
        "MCP response: {mixed_json}"
    );
    assert_eq!(mixed_json["result"]["pairs"].as_array().unwrap().len(), 3);
    let empty_response = app
        .clone()
        .oneshot(proposal_call(
            "empty-doc",
            json!({"pair_kind":"node_node","allowed_docs":["doc-empty"]}),
        ))
        .await
        .unwrap();
    let empty_body = axum::body::to_bytes(empty_response.into_body(), 65_536)
        .await
        .unwrap();
    let empty_json: Value = serde_json::from_slice(&empty_body).unwrap();
    assert_eq!(empty_json["result"]["pairs"], json!([]));

    let (stale_setup, stale_connection) = tokio_postgres::connect(&dsn, tokio_postgres::NoTls)
        .await
        .unwrap();
    let stale_task = tokio::spawn(stale_connection);
    stale_setup
        .execute(
            &format!(
                "UPDATE \"{schema}\".gke_stage1_projections \
                 SET source_fingerprint='stale' WHERE entity_id='node-a'"
            ),
            &[],
        )
        .await
        .unwrap();
    drop(stale_setup);
    stale_task.await.unwrap().unwrap();
    let stale_response = app
        .clone()
        .oneshot(proposal_call(
            "stale-stage1",
            json!({"pair_kind":"node_node","allowed_docs":["doc-a"]}),
        ))
        .await
        .unwrap();
    let stale_body = axum::body::to_bytes(stale_response.into_body(), 65_536)
        .await
        .unwrap();
    let stale_json: Value = serde_json::from_slice(&stale_body).unwrap();
    assert_eq!(stale_json["result"]["pairs"], json!([]));

    drop(app);
    drop(service);
    let (cleanup, connection) = tokio_postgres::connect(&dsn, tokio_postgres::NoTls)
        .await
        .unwrap();
    let cleanup_task = tokio::spawn(connection);
    cleanup
        .batch_execute(&format!("DROP SCHEMA \"{schema}\" CASCADE"))
        .await
        .unwrap();
    drop(cleanup);
    cleanup_task.await.unwrap().unwrap();
}

#[tokio::test]
#[ignore = "requires KOGWISTAR_TEST_PG_DSN; uses disposable PostgreSQL schemas"]
async fn postgres_graph_mutation_batch_uses_configured_graph_type_store() {
    let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
        .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
    let base = format!("rust_graph_role_{}", uuid::Uuid::new_v4().simple());
    let schemas = [
        format!("{base}_knowledge"),
        format!("{base}_conversation"),
        format!("{base}_workflow"),
        format!("{base}_wisdom"),
    ];
    let service = PostgresRunApplicationService::from_dsn_with_graph_schemas(
        &dsn,
        &schemas[0],
        &schemas[1],
        &schemas[2],
        &schemas[3],
    )
    .unwrap()
    .with_graph_role("conversation")
    .unwrap();
    service.ensure_schema().await.unwrap();
    for store in [
        &service.store,
        &service.conversation_store,
        &service.workflow_store,
        &service.wisdom_store,
    ] {
        store
            .create_graph_schema(2, kogwistar_store_postgres::GraphTableNames::default())
            .await
            .unwrap();
    }

    let mut batch: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/graph-mutation-batch-v1.json"
    )))
    .unwrap();
    batch["mutations"][0]["scope"] = json!({
        "namespace": "tenant-a",
        "workspace_id": "ws",
        "graph_space": "facts"
    });
    batch["mutations"][0]["record"]["id"] = json!("conversation-node");
    batch["mutations"][0]["payload"]["id"] = json!("conversation-node");
    batch["mutations"][0]["event_id"] = json!("conversation-node-event");

    let response = ApplicationService::execute(
        &service,
        ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: INTERNAL_GRAPH_MUTATIONS_ROUTE.to_owned(),
            body: serde_json::to_vec(&batch).unwrap(),
            principal: json!({
                "sub": "alice",
                "role": "rw",
                "ns": ["tenant-a"],
                "storage_ns": "tenant-a",
                "workspace_id": "ws",
                "graph_space": "facts",
                "capabilities": ["write_graph"]
            }),
        },
    )
    .await;
    assert_eq!(
        response.status,
        StatusCode::OK.as_u16(),
        "{}",
        String::from_utf8_lossy(&response.body)
    );

    let scope = GraphScope {
        namespace: "tenant-a".to_owned(),
        workspace_id: Some("ws".to_owned()),
        graph_space: Some("facts".to_owned()),
    };
    let read = |id: &str| GraphProjectionRead {
        scope: scope.clone(),
        table: "gke_nodes".to_owned(),
        ids: Some(vec![id.to_owned()]),
        metadata: MetadataFilter::default(),
        limit: 1,
    };
    let conversation_rows = service
        .conversation_store
        .graph_projection_records(read("conversation-node"))
        .await
        .unwrap();
    let knowledge_rows = service
        .store
        .graph_projection_records(read("conversation-node"))
        .await
        .unwrap();
    assert_eq!(conversation_rows.len(), 1);
    assert!(knowledge_rows.is_empty());

    drop(service);
    let (cleanup, connection) = tokio_postgres::connect(&dsn, tokio_postgres::NoTls)
        .await
        .unwrap();
    let cleanup_task = tokio::spawn(connection);
    cleanup
        .batch_execute(
            &schemas
                .iter()
                .map(|schema| format!("DROP SCHEMA \"{schema}\" CASCADE"))
                .collect::<Vec<_>>()
                .join("; "),
        )
        .await
        .unwrap();
    drop(cleanup);
    cleanup_task.await.unwrap().unwrap();
}

#[tokio::test]
#[ignore = "requires KOGWISTAR_TEST_PG_DSN; uses disposable PostgreSQL schemas"]
async fn postgres_embedding_worker_uses_configured_graph_type_store() {
    struct FixedEmbeddingProvider {
        profile: kogwistar_engine::embedding::EmbeddingProfile,
        inputs: Arc<Mutex<Vec<String>>>,
    }
    impl kogwistar_engine::embedding::EmbeddingProvider for FixedEmbeddingProvider {
        fn profile(&self) -> &kogwistar_engine::embedding::EmbeddingProfile {
            &self.profile
        }

        fn embed<'a>(
            &'a self,
            texts: &'a [String],
        ) -> kogwistar_engine::embedding::EmbeddingFuture<'a> {
            Box::pin(async move {
                self.inputs.lock().unwrap().extend(texts.iter().cloned());
                Ok(texts.iter().map(|_| vec![0.6, 0.8]).collect())
            })
        }
    }

    let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
        .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
    let base = format!("rust_graph_role_worker_{}", uuid::Uuid::new_v4().simple());
    let schemas = [
        format!("{base}_knowledge"),
        format!("{base}_conversation"),
        format!("{base}_workflow"),
        format!("{base}_wisdom"),
    ];
    let inputs = Arc::new(Mutex::new(Vec::new()));
    let profile = kogwistar_engine::embedding::EmbeddingProfile {
        provider: "test".to_owned(),
        model: "graph-role-worker-test".to_owned(),
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
    };
    let service = PostgresRunApplicationService::from_dsn_with_graph_schemas(
        &dsn,
        &schemas[0],
        &schemas[1],
        &schemas[2],
        &schemas[3],
    )
    .unwrap()
    .with_graph_role("conversation")
    .unwrap()
    .with_embedding_provider(Arc::new(FixedEmbeddingProvider {
        profile,
        inputs: inputs.clone(),
    }));
    service.ensure_graph_type_schemas().await.unwrap();
    for store in [
        &service.store,
        &service.conversation_store,
        &service.workflow_store,
        &service.wisdom_store,
    ] {
        store
            .create_graph_schema(2, kogwistar_store_postgres::GraphTableNames::default())
            .await
            .unwrap();
    }

    let scope = GraphScope {
        namespace: "tenant-a".to_owned(),
        workspace_id: Some("workspace-a".to_owned()),
        graph_space: Some("conversation".to_owned()),
    };
    let applied = service
        .conversation_store
        .apply_graph_mutations_atomic(vec![GraphMutation {
            scope: scope.clone(),
            table: "gke_nodes".to_owned(),
            entity_kind: "node".to_owned(),
            event_id: "conversation-embedding-event".to_owned(),
            op: "ADD".to_owned(),
            payload: json!({"id":"conversation-embedding-node","label":"Conversation source"}),
            record: GraphRecord {
                id: "conversation-embedding-node".to_owned(),
                document: Some("Conversation embedding source".to_owned()),
                metadata: serde_json::Map::from_iter([
                    ("namespace".to_owned(), json!("tenant-a")),
                    ("workspace_id".to_owned(), json!("workspace-a")),
                    ("graph_space".to_owned(), json!("conversation")),
                ]),
                embedding: None,
            },
            embedding_dim: 2,
        }])
        .await
        .unwrap();
    let revision = kogwistar_engine::canonical_revision::read_canonical_entity_revision(
        &[applied[0].event.clone()],
        "tenant-a",
        "node",
        "conversation-embedding-node",
    )
    .unwrap()
    .unwrap();
    service
        .conversation_store
        .enqueue_index_job(NewIndexJob {
            job_id: "conversation-embedding-job".to_owned(),
            namespace: "tenant-a".to_owned(),
            entity_kind: "node".to_owned(),
            entity_id: "conversation-embedding-node".to_owned(),
            index_kind: "node_embedding".to_owned(),
            op: "UPSERT".to_owned(),
            payload_json: Some(
                kogwistar_engine::canonical_revision::canonical_revision_job_payload(&revision),
            ),
            max_retries: 3,
        })
        .await
        .unwrap();

    let summary = service
        .run_embedding_projection_jobs_once(10, 30, Some("tenant-a"))
        .await
        .unwrap();
    let job = service
        .conversation_store
        .list_index_jobs(
            Some("tenant-a"),
            None,
            Some("node"),
            Some("conversation-embedding-node"),
            Some("node_embedding"),
            10,
        )
        .await
        .unwrap();
    assert_eq!(summary.claimed, 1);
    assert_eq!(
        summary.projected, 1,
        "projection summary: {summary:?}; jobs: {job:#?}"
    );
    assert_eq!(summary.retried, 0);
    assert_eq!(
        inputs.lock().unwrap().as_slice(),
        ["Conversation embedding source"]
    );
    assert_eq!(job.len(), 1);
    assert_eq!(job[0].status, "DONE");
    let promoted = service
        .conversation_store
        .graph_projection_records(GraphProjectionRead {
            scope,
            table: "gke_nodes".to_owned(),
            ids: Some(vec!["conversation-embedding-node".to_owned()]),
            metadata: MetadataFilter::default(),
            limit: 1,
        })
        .await
        .unwrap();
    assert_eq!(promoted.len(), 1);
    assert_eq!(
        promoted[0].embedding.as_deref(),
        Some([0.6_f32, 0.8].as_slice())
    );
    let knowledge_jobs = service
        .store
        .list_index_jobs(
            Some("tenant-a"),
            None,
            Some("node"),
            Some("conversation-embedding-node"),
            Some("node_embedding"),
            10,
        )
        .await
        .unwrap();
    assert!(knowledge_jobs.is_empty());

    drop(service);
    let (cleanup, connection) = tokio_postgres::connect(&dsn, tokio_postgres::NoTls)
        .await
        .unwrap();
    let cleanup_task = tokio::spawn(connection);
    cleanup
        .batch_execute(
            &schemas
                .iter()
                .map(|schema| format!("DROP SCHEMA \"{schema}\" CASCADE"))
                .collect::<Vec<_>>()
                .join("; "),
        )
        .await
        .unwrap();
    drop(cleanup);
    cleanup_task.await.unwrap().unwrap();
}

#[tokio::test]
async fn in_memory_mcp_traversal_matches_shared_python_contract() {
    let fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/graph-traversal-v1.json"
    )))
    .unwrap();
    assert_eq!(fixture["schema_version"], 1);

    let secret = "in-memory-traversal-contract-secret";
    let auth = AuthConfig {
        mode: "jwt".to_owned(),
        algorithm: Some("HS256".to_owned()),
        key: Some(secret.to_owned()),
        ..AuthConfig::default()
    };
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let token = encode(
        &Header::new(Algorithm::HS256),
        &json!({
            "sub":"fixture-reader",
            "role":"rw",
            "ns":["docs"],
            "storage_ns":"fixture-space",
            "workspace":"fixture-workspace",
            "workspace_id":"fixture-workspace",
            "graph_space":"fixture-graph",
            "capabilities":["read_graph", "write_graph"],
            "exp":now + 300
        }),
        &EncodingKey::from_secret(secret.as_bytes()),
    )
    .unwrap();
    let app = router_with_application(
        ApiState {
            health: snapshot(),
            required_roles: vec![],
            implementation: ImplementationSnapshot::default(),
            auth,
        },
        Arc::new(InMemoryGraphApplicationService::default()),
    );
    let scope = json!({
        "namespace":"fixture-space",
        "workspace_id":"fixture-workspace",
        "graph_space":"fixture-graph"
    });
    let mut mutations = Vec::new();
    for node_id in fixture["graph"]["nodes"].as_array().unwrap() {
        let id = node_id.as_str().unwrap();
        let label = fixture["graph"]["node_labels"][id].as_str().unwrap();
        mutations.push(json!({
            "scope":scope,
            "table":"gke_nodes",
            "entity_kind":"node",
            "event_id":format!("fixture-node-{id}"),
            "op":"ADD",
            "payload":{"id":id,"metadata":{"label":label,"doc_id":"D1"}},
            "record":{
                "id":id,
                "document":json!({"label":label,"doc_id":"D1"}).to_string(),
                "metadata":{"label":label,"doc_id":"D1"},
                "embedding":null
            },
            "embedding_dim":0
        }));
    }
    for edge in fixture["graph"]["edges"].as_array().unwrap() {
        let id = edge["id"].as_str().unwrap();
        let metadata = json!({
            "relation":edge["relation"],
            "source_ids":edge["source_ids"],
            "target_ids":edge["target_ids"],
            "doc_id":edge["doc_id"]
        });
        mutations.push(json!({
            "scope":scope,
            "table":"gke_edges",
            "entity_kind":"edge",
            "event_id":format!("fixture-edge-{id}"),
            "op":"ADD",
            "payload":{"id":id,"metadata":metadata},
            "record":{"id":id,"document":metadata.to_string(),"metadata":metadata,"embedding":null},
            "embedding_dim":0
        }));
    }
    for (grain, id) in [("node", "A"), ("node", "B"), ("edge", "E1")] {
        let metadata = json!({
            "entity_type":"acl_record",
            "acl_truth_graph":"knowledge",
            "acl_target_grain":grain,
            "acl_target_entity_id":id,
            "acl_version":1,
            "acl_mode":"public"
        });
        mutations.push(json!({
            "scope":scope,
            "table":"gke_nodes",
            "entity_kind":"node",
            "event_id":format!("fixture-acl-{grain}-{id}"),
            "op":"ADD",
            "payload":{"id":format!("acl-{grain}-{id}"),"metadata":metadata},
            "record":{"id":format!("acl-{grain}-{id}"),"document":null,"metadata":metadata,"embedding":null},
            "embedding_dim":0
        }));
    }
    let write = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/internal/graph/mutations")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::from(
                    json!({"contract_version":1,"atomic":true,"mutations":mutations,"index_jobs":[]})
                        .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(write.status(), StatusCode::OK);

    let cases = fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|case| (case["name"].as_str().unwrap(), case))
        .collect::<std::collections::BTreeMap<_, _>>();
    let call = |name: &str, arguments: Value| {
        let app = app.clone();
        let token = token.clone();
        let name = name.to_owned();
        async move {
            let response = app
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/mcp")
                        .header("content-type", "application/json")
                        .header("authorization", format!("Bearer {token}"))
                        .body(Body::from(
                            json!({
                                "jsonrpc":"2.0",
                                "id":name,
                                "method":"tools/call",
                                "params":{"name":name,"arguments":arguments}
                            })
                            .to_string(),
                        ))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let response: Value =
                serde_json::from_slice(&to_bytes(response.into_body(), 65_536).await.unwrap())
                    .unwrap();
            assert!(response.get("error").is_none(), "{name}: {response}");
            assert_eq!(response["result"]["implementation_authority"], "rust");
            assert_eq!(response["result"]["seam_attestation_version"], 1);
            assert_eq!(response["result"]["seam_crossings"], json!([]));
            let mut result = response["result"].clone();
            for field in [
                "implementation_authority",
                "seam_attestation_version",
                "seam_crossings",
            ] {
                result.as_object_mut().unwrap().remove(field);
            }
            result
        }
    };

    let edge_case = cases["edge_to_nodes"];
    assert_eq!(
        call(
            "kg_neighbors",
            json!({"rid":edge_case["rid"],"doc_id":"D1"})
        )
        .await,
        edge_case["expected_neighbors"]
    );
    let path_case = cases["shortest_path"];
    assert_eq!(
        call(
            "kg_shortest_path",
            json!({"src_id":path_case["src_id"],"dst_id":path_case["dst_id"],"doc_id":"D1"})
        )
        .await["path"],
        path_case["expected_path"]
    );
    let hop_case = cases["k_hop"];
    assert_eq!(
        call(
            "kg_k_hop",
            json!({"start_ids":hop_case["start_ids"],"k":hop_case["k"],"doc_id":"D1"})
        )
        .await["layers"],
        hop_case["expected_layers"]
    );
    let find_case = cases["find_edges"];
    assert_eq!(
        call(
            "kg_find_edges",
            json!({
                "relation":find_case["relation"],
                "src_label_contains":find_case["source_label_contains"],
                "tgt_label_contains":find_case["target_label_contains"],
                "doc_id":find_case["doc_id"]
            })
        )
        .await["edges"],
        find_case["expected_edge_ids"]
    );
}

#[tokio::test]
async fn in_memory_mcp_acl_traversal_matches_shared_python_contract() {
    let fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/graph-acl-traversal-v1.json"
    )))
    .unwrap();
    assert_eq!(fixture["schema_version"], 1);

    let secret = "in-memory-acl-traversal-contract-secret";
    let auth = AuthConfig {
        mode: "jwt".to_owned(),
        algorithm: Some("HS256".to_owned()),
        key: Some(secret.to_owned()),
        ..AuthConfig::default()
    };
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let token = |subject: &str| {
        encode(
            &Header::new(Algorithm::HS256),
            &json!({
                "sub":subject,
                "role":"rw",
                "ns":["docs"],
                "storage_ns":"acl-fixture-space",
                "workspace":"acl-fixture-workspace",
                "workspace_id":"acl-fixture-workspace",
                "graph_space":"acl-fixture-graph",
                "capabilities":["read_graph", "write_graph"],
                "exp":now + 300
            }),
            &EncodingKey::from_secret(secret.as_bytes()),
        )
        .unwrap()
    };
    let app = router_with_application(
        ApiState {
            health: snapshot(),
            required_roles: vec![],
            implementation: ImplementationSnapshot::default(),
            auth,
        },
        Arc::new(InMemoryGraphApplicationService::default()),
    );
    let scope = json!({
        "namespace":"acl-fixture-space",
        "workspace_id":"acl-fixture-workspace",
        "graph_space":"acl-fixture-graph"
    });
    let mut mutations = Vec::new();
    let mut nodes = Vec::new();
    for node in fixture["nodes"].as_array().unwrap() {
        let id = node["id"].as_str().unwrap();
        let mut metadata = json!({
            "label": if id == "source-public" { "Public source" } else { "Private target" },
            "doc_id": fixture["doc_id"]
        });
        if let Some(owner_id) = node["owner_id"].as_str() {
            metadata["owner_id"] = json!(owner_id);
        }
        mutations.push(json!({
            "scope":scope,
            "table":"gke_nodes",
            "entity_kind":"node",
            "event_id":format!("acl-fixture-node-{id}"),
            "op":"ADD",
            "payload":{"id":id,"metadata":metadata},
            "record":{
                "id":id,
                "document":metadata.to_string(),
                "metadata":metadata,
                "embedding":null
            },
            "embedding_dim":0
        }));
        let mut acl = json!({
            "entity_type":"acl_record",
            "acl_truth_graph":fixture["truth_graph"],
            "acl_target_grain":"node",
            "acl_target_entity_id":id,
            "acl_version":1,
            "acl_mode":node["mode"]
        });
        if let Some(owner_id) = node["owner_id"].as_str() {
            acl["owner_id"] = json!(owner_id);
        }
        nodes.push((format!("acl-node-{id}"), acl));
    }
    let edge = &fixture["edge"];
    let edge_id = edge["id"].as_str().unwrap();
    let edge_metadata = json!({
        "relation":edge["relation"],
        "source_ids":edge["source_ids"],
        "target_ids":edge["target_ids"],
        "doc_id":fixture["doc_id"]
    });
    mutations.push(json!({
        "scope":scope,
        "table":"gke_edges",
        "entity_kind":"edge",
        "event_id":format!("acl-fixture-edge-{edge_id}"),
        "op":"ADD",
        "payload":{"id":edge_id,"metadata":edge_metadata},
        "record":{
            "id":edge_id,
            "document":edge_metadata.to_string(),
            "metadata":edge_metadata,
            "embedding":null
        },
        "embedding_dim":0
    }));
    let mut edge_acl = json!({
        "entity_type":"acl_record",
        "acl_truth_graph":fixture["truth_graph"],
        "acl_target_grain":"edge",
        "acl_target_entity_id":edge_id,
        "acl_version":1,
        "acl_mode":edge["mode"]
    });
    if let Some(owner_id) = edge["owner_id"].as_str() {
        edge_acl["owner_id"] = json!(owner_id);
    }
    nodes.push((format!("acl-edge-{edge_id}"), edge_acl));
    for (id, metadata) in nodes {
        mutations.push(json!({
            "scope":scope,
            "table":"gke_nodes",
            "entity_kind":"node",
            "event_id":format!("acl-fixture-record-{id}"),
            "op":"ADD",
            "payload":{"id":id,"metadata":metadata},
            "record":{"id":id,"document":null,"metadata":metadata,"embedding":null},
            "embedding_dim":0
        }));
    }
    let write = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/internal/graph/mutations")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {}", token("owner-a")))
                .body(Body::from(
                    json!({"contract_version":1,"atomic":true,"mutations":mutations,"index_jobs":[]})
                        .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(write.status(), StatusCode::OK);

    for principal in fixture["principals"].as_array().unwrap() {
        let subject = principal["id"].as_str().unwrap();
        let principal_token = token(subject);
        let call = |name: &str, arguments: Value| {
            let app = app.clone();
            let token = principal_token.clone();
            let name = name.to_owned();
            async move {
                let response = app
                    .oneshot(
                        Request::builder()
                            .method("POST")
                            .uri("/mcp")
                            .header("content-type", "application/json")
                            .header("authorization", format!("Bearer {token}"))
                            .body(Body::from(
                                json!({
                                    "jsonrpc":"2.0",
                                    "id":name,
                                    "method":"tools/call",
                                    "params":{"name":name,"arguments":arguments}
                                })
                                .to_string(),
                            ))
                            .unwrap(),
                    )
                    .await
                    .unwrap();
                assert_eq!(response.status(), StatusCode::OK);
                let response: Value =
                    serde_json::from_slice(&to_bytes(response.into_body(), 65_536).await.unwrap())
                        .unwrap();
                assert!(response.get("error").is_none(), "{name}: {response}");
                assert_eq!(response["result"]["implementation_authority"], "rust");
                assert_eq!(response["result"]["seam_attestation_version"], 1);
                assert_eq!(response["result"]["seam_crossings"], json!([]));
                let mut result = response["result"].clone();
                for field in [
                    "implementation_authority",
                    "seam_attestation_version",
                    "seam_crossings",
                ] {
                    result.as_object_mut().unwrap().remove(field);
                }
                result
            }
        };
        let expected_neighbors = &principal["expected_neighbors"];
        assert_eq!(
            call(
                "kg_neighbors",
                json!({"rid":"source-public","doc_id":fixture["doc_id"]})
            )
            .await,
            *expected_neighbors,
            "principal={subject}"
        );
        assert_eq!(
            call(
                "kg_k_hop",
                json!({"start_ids":["source-public"],"k":2,"doc_id":fixture["doc_id"]})
            )
            .await["layers"],
            principal["expected_k_hop"],
            "principal={subject}"
        );
        assert_eq!(
            call(
                "kg_shortest_path",
                json!({
                    "src_id":"source-public",
                    "dst_id":"target-private",
                    "max_depth":8,
                    "doc_id":fixture["doc_id"]
                })
            )
            .await["path"],
            principal["expected_shortest_path"],
            "principal={subject}"
        );
        assert_eq!(
            call(
                "kg_find_edges",
                json!({"relation":edge["relation"],"doc_id":fixture["doc_id"]})
            )
            .await["edges"],
            principal["expected_find_edges"],
            "principal={subject}"
        );
    }
}

// PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/tests/part_08.rs#postgres_hybrid_search_is_available_over_authenticated_http; python=kogwistar/server_mcp_with_admin.py#search_index_hybrid; fixture=contracts/golden/search-index-hybrid-v1.json; rust_test=rust/crates/kogwistar-api/src/search_index_fts.rs#hybrid_fusion_matches_shared_python_golden_fixture; python_test=tests/core/test_search_index_contract.py#test_hybrid_search_fusion_matches_shared_rust_golden_fixture
#[tokio::test]
#[ignore = "requires KOGWISTAR_TEST_PG_DSN; uses an isolated PostgreSQL schema"]
async fn postgres_hybrid_search_is_available_over_authenticated_http() {
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
    use std::sync::Arc;
    use tower::ServiceExt;

    struct FixedEmbeddingProvider;
    impl EmbeddingProvider for FixedEmbeddingProvider {
        fn profile(&self) -> &kogwistar_engine::embedding::EmbeddingProfile {
            static PROFILE: std::sync::OnceLock<kogwistar_engine::embedding::EmbeddingProfile> =
                std::sync::OnceLock::new();
            PROFILE.get_or_init(|| kogwistar_engine::embedding::EmbeddingProfile {
                provider: "test".to_owned(),
                model: "hybrid-route-fixed-v1".to_owned(),
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
            })
        }

        fn embed<'a>(
            &'a self,
            texts: &'a [String],
        ) -> kogwistar_engine::embedding::EmbeddingFuture<'a> {
            Box::pin(async move { Ok(texts.iter().map(|_| vec![0.6, 0.8]).collect()) })
        }
    }

    let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
        .expect("ignored hybrid-search HTTP test requires KOGWISTAR_TEST_PG_DSN");
    let suffix = uuid::Uuid::new_v4().simple().to_string();
    let schema = format!("rust_hybrid_http_{suffix}");
    let fts_path = std::env::temp_dir().join(format!("rust-hybrid-http-{suffix}.sqlite3"));
    let service = PostgresRunApplicationService::from_dsn(&dsn, &schema)
        .unwrap()
        .with_search_index_path(&fts_path)
        .unwrap()
        .with_embedding_provider(Arc::new(FixedEmbeddingProvider));
    service.ensure_schema().await.unwrap();
    service
        .store
        .create_graph_schema(2, kogwistar_store_postgres::GraphTableNames::default())
        .await
        .unwrap();

    let namespace = format!("hybrid-{suffix}");
    let node_id = format!("hybrid-node-{suffix}");
    let title = format!("uniquehybridpermit{suffix}");
    let provision = "review the unique hybrid permit";
    let scope = GraphScope {
        namespace: namespace.clone(),
        workspace_id: Some("workspace-a".to_owned()),
        graph_space: Some("knowledge".to_owned()),
    };
    service
        .store
        .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
            scope: scope.clone(),
            table: "gke_nodes".to_owned(),
            record: GraphRecord {
                id: node_id.clone(),
                document: Some(json!({"id":node_id,"label":title}).to_string()),
                metadata: Map::from_iter([
                    ("namespace".to_owned(), json!(namespace)),
                    ("workspace_id".to_owned(), json!("workspace-a")),
                    ("graph_space".to_owned(), json!("knowledge")),
                    ("lifecycle_status".to_owned(), json!("active")),
                ]),
                embedding: None,
            },
            embedding_dim: 2,
        })
        .await
        .unwrap();
    service
        .store
        .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
            scope: scope.clone(),
            table: "gke_nodes".to_owned(),
            record: GraphRecord {
                id: format!("acl|{node_id}"),
                document: Some("{}".to_owned()),
                metadata: Map::from_iter([
                    ("entity_type".to_owned(), json!("acl_record")),
                    ("acl_truth_graph".to_owned(), json!("knowledge")),
                    ("acl_target_grain".to_owned(), json!("node")),
                    ("acl_target_entity_id".to_owned(), json!(node_id)),
                    ("acl_mode".to_owned(), json!("public")),
                    ("acl_version".to_owned(), json!(1)),
                    ("tombstoned".to_owned(), json!(false)),
                ]),
                embedding: None,
            },
            embedding_dim: 0,
        })
        .await
        .unwrap();

    let entry: kogwistar_contracts::SearchIndexEntryV1 = serde_json::from_value(json!({
        "node_id": node_id,
        "canonical_title": title,
        "keywords": ["uniquehybrid"],
        "aliases": [],
        "provision": provision,
        "doc_id": "doc-hybrid"
    }))
    .unwrap();
    let fts = std::sync::Mutex::new(crate::search_index_fts::open_index(&fts_path).unwrap());
    crate::search_index_fts::admit_search_index_entries(
        &service.store,
        &service.store,
        &fts,
        service.embedding_provider.as_ref().unwrap().as_ref(),
        &namespace,
        "gke_node_index",
        &[entry.clone()],
    )
    .await
    .unwrap();
    assert!(
        crate::search_index_fts::search_fts(&fts.lock().unwrap(), "review", 5)
            .unwrap()
            .iter()
            .any(|hit| hit.node_id == node_id)
    );
    let direct_hits = crate::search_index_fts::hybrid_search(
        &fts,
        &service.store,
        "gke_node_index",
        service.embedding_provider.as_ref().unwrap().as_ref(),
        "review",
        5,
    )
    .await
    .unwrap();
    assert!(
        direct_hits.iter().any(|hit| hit.node_id == node_id),
        "{direct_hits:?}"
    );
    drop(fts);

    let secret = "hybrid-search-http-test-secret";
    let auth = AuthConfig {
        mode: "jwt".to_owned(),
        algorithm: Some("HS256".to_owned()),
        key: Some(secret.to_owned()),
        ..AuthConfig::default()
    };
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let token = |ns: &str, capabilities: &[&str]| {
        encode(
            &Header::new(Algorithm::HS256),
            &json!({
                "sub":"reader",
                "role":"ro",
                "ns":[ns],
                "storage_ns":ns,
                "workspace_id":"workspace-a",
                "graph_space":"knowledge",
                "capabilities":capabilities,
                "exp":now + 300
            }),
            &EncodingKey::from_secret(secret.as_bytes()),
        )
        .unwrap()
    };
    let app = router_with_application(
        ApiState {
            health: snapshot(),
            required_roles: vec![],
            implementation: ImplementationSnapshot::default(),
            auth,
        },
        Arc::new(service),
    );

    let mut routed_entry = entry.clone();
    routed_entry.aliases.push("http-route-revision".to_owned());
    let writer_token = encode(
        &Header::new(Algorithm::HS256),
        &json!({
            "sub":"writer",
            "role":"rw",
            "ns":[namespace],
            "storage_ns":namespace,
            "workspace_id":"workspace-a",
            "graph_space":"knowledge",
            "capabilities":["write_graph"],
            "exp":now + 300
        }),
        &EncodingKey::from_secret(secret.as_bytes()),
    )
    .unwrap();
    for _ in 0..2 {
        let write = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/add_index_entries")
                    .header("authorization", format!("Bearer {writer_token}"))
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({"index":[routed_entry.clone()]}).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(write.status(), StatusCode::OK);
        let payload: Value =
            serde_json::from_slice(&to_bytes(write.into_body(), 65_536).await.unwrap()).unwrap();
        assert_eq!(payload, json!({"ok":true}));
    }

    for (query, resolve_node) in [("review", false), ("review", true)] {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri(format!(
                        "/api/search_index_hybrid?q={query}&limit=5&resolve_node={resolve_node}"
                    ))
                    .header(
                        "authorization",
                        format!("Bearer {}", token(&namespace, &["read_graph"])),
                    )
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let payload: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 65_536).await.unwrap()).unwrap();
        let results = payload["results"].as_array().unwrap();
        let hit = results
            .iter()
            .find(|hit| hit["node_id"] == node_id)
            .unwrap_or_else(|| panic!("authorized indexed node missing from payload: {payload}"));
        assert_eq!(hit["canonical_title"], title);
        if resolve_node {
            assert!(hit["documents"].is_string());
            assert_eq!(hit["metadatas"]["workspace_id"], "workspace-a");
        } else {
            assert!(hit.get("documents").is_none());
            assert!(hit.get("metadatas").is_none());
        }
    }

    let denied = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/search_index_hybrid?q=review")
                .header(
                    "authorization",
                    format!("Bearer {}", token(&namespace, &[])),
                )
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(denied.status(), StatusCode::FORBIDDEN);

    let outside_scope = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/search_index_hybrid?q=review")
                .header(
                    "authorization",
                    format!("Bearer {}", token("other-namespace", &["read_graph"])),
                )
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(outside_scope.status(), StatusCode::OK);
    let outside_scope: Value =
        serde_json::from_slice(&to_bytes(outside_scope.into_body(), 65_536).await.unwrap())
            .unwrap();
    assert_eq!(outside_scope["results"], json!([]));
}
