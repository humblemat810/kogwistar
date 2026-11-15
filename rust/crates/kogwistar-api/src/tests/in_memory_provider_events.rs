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
        "capabilities":["spawn_process", "workflow.run.write", "workflow.run.read", "write_graph"]
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
            "capabilities": ["spawn_process", "workflow.run.write", "workflow.run.read"],
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
        "capabilities":["spawn_process", "workflow.run.write", "read_graph", "write_graph"]
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
    let event_fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/agent-durable-provider-event-v1.json"
    )))
    .unwrap();
    let provider_event: kogwistar_contracts::ProviderEventV1 =
        serde_json::from_value(event_fixture["envelope"]["event"].clone()).unwrap();
    service
        .persist_in_memory_native_stream_semantic_event(
            &request,
            provider_event,
            event_fixture["envelope"]["trace"]["provider_id"]
                .as_str()
                .unwrap(),
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
        event_fixture["envelope"]["event"]["kind"]
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
    let event_fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/agent-durable-provider-event-v1.json"
    )))
    .unwrap();
    let run_id = event_fixture["envelope"]["trace"]["run_id"]
        .as_str()
        .unwrap();
    let turn_id = event_fixture["envelope"]["trace"]["turn_id"]
        .as_str()
        .unwrap();
    let workflow_id = "memory-provider-ack-workflow";
    let conversation_id = "memory-provider-ack-conversation";
    let event_id = event_fixture["computed_event_id"].as_str().unwrap();
    let message_id = format!("msg:{event_id}");
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
            user_turn_node_id: turn_id.to_owned(),
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
    let mut raw_event = event_fixture["envelope"].clone();
    // This synthetic run persists the handler step but not the fixture's producer step 7.
    raw_event["trace"]["producer_step_seq"] = Value::Null;
    raw_event["event_id"] = json!(event_id);
    let event: kogwistar_contracts::DurableProviderEventV1 =
        serde_json::from_value(raw_event).unwrap();
    let claim = json!({
        "message_id": message_id,
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
            message_id: message_id.clone(),
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
            step_id: Some(event.trace.step_id.clone()),
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
            message_ids: Some(vec![message_id.clone()]),
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
            turn_node_id: Some(turn_id.to_owned()),
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
    let mut canonical_graph_rows = nodes
        .iter()
        .chain(graph.iter())
        .map(|record| {
            json!({
                "id": record.id,
                "document": record.document,
                "metadata": record.metadata,
            })
        })
        .collect::<Vec<_>>();
    canonical_graph_rows.sort_by_key(|row| serde_json::to_string(row).unwrap());
    let graph_digest = format!(
        "{:x}",
        {
            use sha2::Digest;
            sha2::Sha256::digest(serde_json::to_vec(&canonical_graph_rows).unwrap())
        }
    );
    let expected_digests: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/agent-provider-event-graph-digests-v1.json"
    )))
    .unwrap();
    assert_eq!(expected_digests["schema_version"], 1);
    assert_eq!(
        graph_digest,
        expected_digests["digests"]["in_memory"]["sha256"]
            .as_str()
            .unwrap()
    );
    assert!(nodes.iter().any(|node| {
        node.id == format!("wf_step|{run_id}|1")
            && node.metadata["entity_type"] == "workflow_step_exec"
            && node.metadata["in_conversation_chain"] == false
    }));
    assert!(nodes.iter().any(|node| {
        node.metadata["entity_type"] == "acl_record"
            && node.metadata["acl_target_entity_id"] == format!("wf_step|{run_id}|1")
            && node.metadata["owner_id"] == "memory-ack-owner@example.test"
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
