#[tokio::test]
async fn sqlite_native_provider_event_request_persists_semantic_events() {
    use crate::native_answer_generation::OpenAiCompatibleChatConfig;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let chroma_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let chroma_endpoint = format!("http://{}", chroma_listener.local_addr().unwrap());
    let chroma_setup = tokio::spawn(async move {
        for _ in 0..2 {
            let (mut stream, _) = chroma_listener.accept().await.unwrap();
            let mut bytes = Vec::new();
            let (header_end, content_length) = loop {
                let mut chunk = [0_u8; 2048];
                let count = stream.read(&mut chunk).await.unwrap();
                assert!(count > 0);
                bytes.extend_from_slice(&chunk[..count]);
                if let Some(index) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
                    let headers = String::from_utf8_lossy(&bytes[..index]);
                    let content_length = headers
                        .lines()
                        .find_map(|line| {
                            let (name, value) = line.split_once(':')?;
                            name.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse::<usize>().ok())
                                .flatten()
                        })
                        .unwrap_or(0);
                    if bytes.len() >= index + 4 + content_length {
                        break (index + 4, content_length);
                    }
                }
            };
            let request: Value =
                serde_json::from_slice(&bytes[header_end..header_end + content_length]).unwrap();
            let body = json!({
                "id": format!("collection-{}", request["name"].as_str().unwrap()),
                "metadata": {"hnsw:space": "cosine"}
            })
            .to_string();
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                body.len(),
                body
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
        chroma_config("native-provider-events-nodes"),
        chroma_config("native-provider-events-edges"),
    )
    .await
    .unwrap();
    chroma_setup.await.unwrap();

    let provider_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let provider_address = provider_listener.local_addr().unwrap();
    let provider_task = tokio::spawn(async move {
        let (mut stream, _) = provider_listener.accept().await.unwrap();
        let mut bytes = Vec::new();
        let (header_end, content_length) = loop {
            let mut chunk = [0_u8; 4096];
            let count = stream.read(&mut chunk).await.unwrap();
            assert!(count > 0);
            bytes.extend_from_slice(&chunk[..count]);
            if let Some(index) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
                let header_end = index + 4;
                let headers = String::from_utf8_lossy(&bytes[..header_end]).to_ascii_lowercase();
                let content_length = headers
                    .lines()
                    .find_map(|line| line.strip_prefix("content-length:"))
                    .unwrap()
                    .trim()
                    .parse::<usize>()
                    .unwrap();
                if bytes.len() >= header_end + content_length {
                    break (header_end, content_length);
                }
            }
        };
        let request: Value =
            serde_json::from_slice(&bytes[header_end..header_end + content_length]).unwrap();
        assert_eq!(request["model"], "sqlite-event-model");
        assert_eq!(request["stream"], true);
        let mut body = String::new();
        for chunk in [
            json!({"id":"sqlite-response","choices":[{"delta":{"content":"provider text"},"finish_reason":null}]}).to_string(),
            json!({"id":"sqlite-response","choices":[{"delta":{},"finish_reason":"stop"}]}).to_string(),
        ] {
            body.push_str(&format!("data: {chunk}\r\n\r\n"));
        }
        body.push_str("data: [DONE]\r\n\r\n");
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        stream.write_all(response.as_bytes()).await.unwrap();
    });
    let provider = OpenAiCompatibleChatConfig::new(
        format!("http://{provider_address}/v1/chat/completions"),
        "sqlite-event-model",
        None,
    )
    .unwrap()
    .with_streaming(true);

    let sqlite_path = std::env::temp_dir().join(format!(
        "kogwistar-sqlite-native-provider-events-{}.sqlite",
        uuid::Uuid::new_v4().simple()
    ));
    let auth_path = std::env::temp_dir().join(format!(
        "kogwistar-sqlite-native-provider-events-auth-{}.sqlite",
        uuid::Uuid::new_v4().simple()
    ));
    let auth = SqliteAuthStore::open(&auth_path).unwrap();
    auth.resolve_external_identity(ResolveExternalIdentity {
        issuer: "test".to_owned(),
        subject: "sqlite-event-owner".to_owned(),
        email: "sqlite-event-owner@example.test".to_owned(),
        display_name: None,
        new_user_id: "sqlite-event-owner".to_owned(),
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
    let service = SqliteRunApplicationService::open(&sqlite_path)
        .unwrap()
        .with_chroma_graph(graph)
        .with_owner_auth_config(&owner_auth)
        .await
        .unwrap()
        .with_chat_completion_provider(provider);

    let principal = json!({
        "sub": "sqlite-event-owner", "user_id": "sqlite-event-owner", "role": "rw",
        "ns": ["conversation"], "storage_ns": "conversation",
        "workspace": "workspace-a", "graph_space": "conversation",
        "capabilities": ["read_graph", "write_graph", "spawn_process", "workflow.run.write"]
    });
    let run_id = "sqlite-native-provider-event-run";
    let conversation_id = "sqlite-native-provider-event-conversation";
    let submitted = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/workflow/runs".to_owned(),
            body: serde_json::to_vec(&json!({
                "run_id": run_id,
                "workflow_id": "sqlite-native-provider-event-workflow",
                "conversation_id": conversation_id,
                "turn_node_id": "turn-1",
                "start_node_id": "provider-step",
                "node_ops": {
                    "provider-step": "rust-native/agent.event.request",
                    "receive-step": "rust-native/agent.event.receive",
                    "ack-no-action-step": "rust-native/agent.event.ack",
                    "ack-final-step": "rust-native/agent.event.ack",
                    "done-step": "rust-native/emit-result"
                },
                "initial_state": {
                    "provider_request": {
                        "contract_version": "v1",
                        "model": "sqlite-event-model",
                        "instructions": "fixture only",
                        "input": "emit one response"
                    }
                },
                "runtime_routes": [
                    {"source_node_id": "provider-step", "target_node_id": "receive-step", "join_mask": 0},
                    {
                        "source_node_id": "receive-step",
                        "target_node_id": "ack-final-step",
                        "predicate": "event_is_response_complete",
                        "join_mask": 0
                    },
                    {
                        "source_node_id": "receive-step",
                        "target_node_id": "ack-no-action-step",
                        "predicate": "event_requires_no_action",
                        "join_mask": 0
                    },
                    {
                        "source_node_id": "ack-no-action-step",
                        "target_node_id": "receive-step",
                        "predicate": "event_stream_has_more",
                        "join_mask": 0
                    },
                    {
                        "source_node_id": "ack-final-step",
                        "target_node_id": "done-step",
                        "predicate": "event_run_succeeded",
                        "join_mask": 0
                    }
                ]
            }))
            .unwrap(),
            principal: principal.clone(),
        })
        .await;
    assert_eq!(
        submitted.status,
        StatusCode::ACCEPTED.as_u16(),
        "{}",
        String::from_utf8_lossy(&submitted.body)
    );
    tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
    assert!(service.execute_inline_native_work_once().await.unwrap());
    provider_task.await.unwrap();
    assert_eq!(
        service
            .store
            .get_server_run(run_id)
            .unwrap()
            .unwrap()
            .status,
        "running"
    );
    let messages = service
        .store
        .list_projected_lane_messages(LaneMessageFilter {
            namespace: Some("conversation".to_owned()),
            inbox_id: Some("agent-provider-events".to_owned()),
            conversation_id: Some(conversation_id.to_owned()),
            limit: 20,
            ..LaneMessageFilter::default()
        })
        .unwrap();
    assert!(!messages.is_empty());
    assert!(messages.iter().all(|message| message.status == "pending"));
    assert!(messages.iter().any(|message| {
        message
            .payload_json
            .as_deref()
            .is_some_and(|payload| payload.contains("response_completed"))
    }));
    assert!(messages.iter().all(|message| {
        message
            .payload_json
            .as_deref()
            .is_some_and(|payload| !payload.contains("text_delta"))
    }));
    let events = service.store.list_server_run_events(run_id, 0, 100).unwrap();
    assert!(events.iter().any(|event| {
        event.event_type == "workflow.provider_event.v1"
            && event.payload_json.contains("response_completed")
    }));
    assert!(!events.iter().any(|event| event.payload_json.contains("text_delta")));

    for _ in 0..10 {
        if service.store.get_server_run(run_id).unwrap().unwrap().status == "succeeded" {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
        if !service.execute_inline_native_work_once().await.unwrap() {
            break;
        }
    }
    let run = service.store.get_server_run(run_id).unwrap().unwrap();
    assert_eq!(run.status, "succeeded");
    let handled_lanes = service
        .store
        .list_projected_lane_messages(LaneMessageFilter {
            namespace: Some("conversation".to_owned()),
            inbox_id: Some("agent-provider-events".to_owned()),
            conversation_id: Some(conversation_id.to_owned()),
            limit: 20,
            ..LaneMessageFilter::default()
        })
        .unwrap();
    assert!(!handled_lanes.is_empty());
    assert!(handled_lanes.iter().all(|message| message.status == "completed"));
    let run_events = service.store.list_server_run_events(run_id, 0, 100).unwrap();
    let transition_nodes = run_events
        .iter()
        .filter(|event| event.event_type == "workflow.recorded_transition.v1")
        .filter_map(|event| {
            serde_json::from_str::<Value>(&event.payload_json)
                .ok()?
                .get("reduced")?
                .get("state")?
                .get("last_node_id")?
                .as_str()
                .map(str::to_owned)
        })
        .collect::<Vec<_>>();
    let receive_transitions = transition_nodes
        .iter()
        .filter(|node_id| node_id.as_str() == "receive-step")
        .count();
    let acknowledgement_transitions = transition_nodes
        .iter()
        .filter(|node_id| matches!(node_id.as_str(), "ack-no-action-step" | "ack-final-step"))
        .count();
    let handled_traces = run_events
        .iter()
        .filter(|event| event.event_type == "workflow.provider_event.handled.v1")
        .count();
    assert!(receive_transitions >= 1);
    assert!(acknowledgement_transitions >= 1);
    assert_eq!(handled_traces, handled_lanes.len());

    drop(service);
    remove_test_sqlite(&sqlite_path);
    remove_test_sqlite(&auth_path);
}

#[tokio::test]
async fn sqlite_provider_event_ack_commits_handler_transition_and_causal_graph_atomically() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let chroma_setup = tokio::spawn(async move {
        for _ in 0..2 {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut bytes = Vec::new();
            let mut chunk = [0_u8; 2048];
            let (header_end, content_length) = loop {
                let count = stream.read(&mut chunk).await.unwrap();
                assert!(count > 0);
                bytes.extend_from_slice(&chunk[..count]);
                if let Some(header_end) = bytes.windows(4).position(|window| window == b"\r\n\r\n")
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
            let request: Value =
                serde_json::from_slice(&bytes[header_end + 4..header_end + 4 + content_length])
                    .unwrap();
            let response = json!({
                "id": format!("collection-{}", request["name"].as_str().unwrap()),
                "metadata": {"hnsw:space": "cosine"}
            })
            .to_string();
            let wire = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                response.len(),
                response
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
        chroma_config("provider-event-nodes"),
        chroma_config("provider-event-edges"),
    )
    .await
    .unwrap();
    chroma_setup.await.unwrap();

    let path = std::env::temp_dir().join(format!(
        "kogwistar-sqlite-provider-event-ack-{}.sqlite",
        uuid::Uuid::new_v4().simple()
    ));
    let auth_path = std::env::temp_dir().join(format!(
        "kogwistar-sqlite-provider-event-ack-auth-{}.sqlite",
        uuid::Uuid::new_v4().simple()
    ));
    let auth_store = SqliteAuthStore::open(&auth_path).unwrap();
    auth_store
        .resolve_external_identity(ResolveExternalIdentity {
            issuer: "test".to_owned(),
            subject: "sqlite-ack-owner".to_owned(),
            email: "sqlite-ack-owner@example.test".to_owned(),
            display_name: None,
            new_user_id: "sqlite-ack-owner".to_owned(),
            default_role: "rw".to_owned(),
            default_ns: "conversation".to_owned(),
        })
        .await
        .unwrap();
    drop(auth_store);
    let owner_auth = AuthConfig {
        auth_db_url: Some(format!("sqlite:///{}", auth_path.display())),
        ..AuthConfig::default()
    };
    let service = SqliteRunApplicationService::open(&path)
        .unwrap()
        .with_chroma_graph(graph)
        .with_owner_auth_config(&owner_auth)
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
    let workflow_id = "sqlite-provider-ack-workflow";
    let conversation_id = event_fixture["publication"]["conversation_id"]
        .as_str()
        .unwrap();
    let turn_id = event_fixture["envelope"]["trace"]["turn_id"]
        .as_str()
        .unwrap();
    let owner_principal = json!({
        "sub": "sqlite-ack-owner", "user_id": "sqlite-ack-owner", "role": "rw",
        "ns": ["conversation"], "storage_ns": "conversation",
        "workspace": "workspace-a", "graph_space": "conversation",
        "capabilities": ["read_graph", "write_graph"]
    });
    let binding = runtime_owner_binding_v1(&owner_principal, &service.capabilities).unwrap();
    service
        .store
        .create_server_run(ServerRunCreate {
            run_id: run_id.to_owned(),
            conversation_id: conversation_id.to_owned(),
            workflow_id: workflow_id.to_owned(),
            user_id: Some("sqlite-ack-owner".to_owned()),
            user_turn_node_id: turn_id.to_owned(),
            status: "running".to_owned(),
        })
        .unwrap();
    let admission = service
        .store
        .append_server_run_event(
            run_id,
            "run.created",
            json!({"runtime_owner_binding": binding}).to_string(),
        )
        .unwrap();
    let start: RecordedRuntimeTransition = serde_json::from_value(json!({
        "contract_version": 1,
        "transition_id": "sqlite-provider-ack-start",
        "expected_event_seq": admission.seq,
        "kind": "start",
        "run_id": run_id,
        "workflow_id": workflow_id,
        "conversation_id": conversation_id,
        "step_seq": 0,
        "node_id": "start-step",
        "token_id": run_id,
        "initial_state": {},
        "frontier": {
            "pending": [["handler-step", 0, run_id, null]],
            "suspended": [], "join_node_ids": [], "join_outstanding": [], "join_waiters": {}
        }
    }))
    .unwrap();
    let started = service
        .store
        .apply_recorded_runtime_transition(start, false)
        .unwrap();
    let handler: RecordedRuntimeTransition = serde_json::from_value(json!({
        "contract_version": 1,
        "transition_id": "sqlite-provider-ack-handler",
        "expected_event_seq": started.event_seq,
        "kind": "recorded_step_success",
        "run_id": run_id,
        "workflow_id": workflow_id,
        "conversation_id": conversation_id,
        "step_seq": 1,
        "node_id": "handler-step",
        "token_id": run_id,
        "state_update": [["u", {"agent_event_handling_step_seq": 1}]],
        "frontier": {
            "pending": [["ack-step", 0, run_id, null]],
            "suspended": [], "join_node_ids": [], "join_outstanding": [], "join_waiters": {}
        },
        "result": {"handled": true}
    }))
    .unwrap();
    service
        .store
        .apply_recorded_runtime_transition(handler, false)
        .unwrap();

    let mut raw_event = event_fixture["envelope"].clone();
    // This synthetic run persists the handler step but not the fixture's producer step 7.
    raw_event["trace"]["producer_step_seq"] = Value::Null;
    raw_event["event_id"] = event_fixture["computed_event_id"].clone();
    let event: kogwistar_contracts::DurableProviderEventV1 =
        serde_json::from_value(raw_event).unwrap();
    let event_message_id = format!("msg:{}", event.event_id);
    service
        .store
        .project_lane_message(NewProjectedLaneMessage {
            message_id: event_message_id.clone(),
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
        })
        .unwrap();
    let event_claim = service
        .store
        .claim_projected_lane_messages_filtered(LaneMessageClaimFilter {
            namespace: "conversation".to_owned(),
            inbox_id: "agent-provider-events".to_owned(),
            claimed_by: "event-consumer-attempt".to_owned(),
            message_ids: Some(vec![event_message_id.clone()]),
            run_id: Some(run_id.to_owned()),
            msg_type: Some("agent.provider_event".to_owned()),
            recipient_id: Some("agent-event-consumer".to_owned()),
            limit: 1,
            lease_seconds: 60,
        })
        .unwrap();
    assert_eq!(event_claim.len(), 1);

    let runtime_message_id = "sqlite-provider-ack-runtime";
    service
        .store
        .project_lane_message(NewProjectedLaneMessage {
            message_id: runtime_message_id.to_owned(),
            namespace: "workflow".to_owned(),
            purpose: "system".to_owned(),
            inbox_id: "workflow-runtime".to_owned(),
            conversation_id: conversation_id.to_owned(),
            recipient_id: "python-worker".to_owned(),
            sender_id: "rust-scheduler".to_owned(),
            msg_type: "workflow.step.execute".to_owned(),
            status: "pending".to_owned(),
            created_at: 1,
            available_at: 1,
            run_id: Some(run_id.to_owned()),
            step_id: Some("ack-step".to_owned()),
            correlation_id: Some(run_id.to_owned()),
            payload_json: Some(
                json!({
                    "workflow_id": workflow_id,
                    "conversation_id": conversation_id,
                    "token_id": run_id,
                    "parent_token_id": null,
                    "op": "app/event-ack"
                })
                .to_string(),
            ),
            error_json: None,
        })
        .unwrap();
    let runtime_claim = service
        .store
        .claim_projected_lane_messages_filtered(LaneMessageClaimFilter {
            namespace: "workflow".to_owned(),
            inbox_id: "workflow-runtime".to_owned(),
            claimed_by: "runtime-attempt".to_owned(),
            message_ids: Some(vec![runtime_message_id.to_owned()]),
            run_id: Some(run_id.to_owned()),
            msg_type: Some("workflow.step.execute".to_owned()),
            recipient_id: Some("python-worker".to_owned()),
            limit: 1,
            lease_seconds: 60,
        })
        .unwrap();
    assert_eq!(runtime_claim.len(), 1);
    let handoff = RecordedWorkerHandoff {
        message_id: runtime_message_id.to_owned(),
        claimed_by: "runtime-attempt".to_owned(),
        run_id: run_id.to_owned(),
        step_id: "ack-step".to_owned(),
        correlation_id: run_id.to_owned(),
    };
    let acknowledgement = kogwistar_runtime::ProviderEventAcknowledgementV1 {
        message_id: event_message_id,
        claimed_by: "event-consumer-attempt".to_owned(),
        event_id: event.event_id,
    };
    let effect: RecordedWorkerSuccessEffect = serde_json::from_value(json!({
        "contract_version": 1,
        "effect_id": "sqlite-provider-ack-effect",
        "state_update": [],
        "successors": [],
        "result": {"handled": true},
        "provider_event_acknowledgements": [acknowledgement.clone()]
    }))
    .unwrap();
    let execute = |effect: RecordedWorkerSuccessEffect| {
        let body = serde_json::to_vec(&json!({
            "handoff": handoff,
            "effect": effect
        }))
        .unwrap();
        service.execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/internal/runtime/results".to_owned(),
            body,
            principal: json!({
                "sub": "python-runtime-worker",
                "role": "admin",
                "capabilities": ["runtime.submit_result"]
            }),
        })
    };
    let mut bad_effect = effect.clone();
    bad_effect.provider_event_acknowledgements[0].claimed_by = "wrong-claimer".to_owned();
    assert_eq!(
        execute(bad_effect).await.status,
        StatusCode::CONFLICT.as_u16()
    );
    assert_eq!(
        service
            .store
            .get_projected_lane_message(&format!("msg:{}", acknowledgement.event_id))
            .unwrap()
            .unwrap()
            .status,
        "claimed"
    );
    assert_eq!(
        service
            .store
            .read_recorded_runtime_state(run_id, workflow_id, conversation_id)
            .unwrap()
            .unwrap()
            .last_step_seq,
        1
    );
    let applied = execute(effect.clone()).await;
    assert_eq!(
        applied.status,
        StatusCode::OK.as_u16(),
        "{}",
        String::from_utf8_lossy(&applied.body)
    );
    assert_eq!(
        service
            .store
            .read_recorded_runtime_state(run_id, workflow_id, conversation_id)
            .unwrap()
            .unwrap()
            .last_step_seq,
        2
    );
    assert_eq!(
        service
            .store
            .get_projected_lane_message(&format!("msg:{}", acknowledgement.event_id))
            .unwrap()
            .unwrap()
            .status,
        "completed"
    );
    let graph_events = service
        .store
        .replay_raw_events("conversation", 0, 100)
        .unwrap();
    assert!(
        graph_events
            .iter()
            .any(|row| { row.entity_id == format!("wf_step|{run_id}|2") })
    );
    assert!(graph_events.iter().any(|row| {
        serde_json::from_str::<Value>(&row.payload_json)
            .is_ok_and(|payload| payload["metadata"]["relation"] == "acknowledged_by_step")
    }));
    assert!(
        service
            .store
            .list_server_run_events(run_id, 0, 100)
            .unwrap()
            .iter()
            .any(|event| event.event_type == "workflow.provider_event.handled.v1")
    );
    let graph_event_count = service
        .store
        .replay_raw_events("conversation", 0, 100)
        .unwrap()
        .len();
    let handled_trace_count = service
        .store
        .list_server_run_events(run_id, 0, 100)
        .unwrap()
        .iter()
        .filter(|event| event.event_type == "workflow.provider_event.handled.v1")
        .count();
    let retried = execute(effect).await;
    assert_eq!(retried.status, StatusCode::OK.as_u16());
    assert_eq!(
        service
            .store
            .replay_raw_events("conversation", 0, 100)
            .unwrap()
            .len(),
        graph_event_count
    );
    assert_eq!(
        service
            .store
            .list_server_run_events(run_id, 0, 100)
            .unwrap()
            .iter()
            .filter(|event| event.event_type == "workflow.provider_event.handled.v1")
            .count(),
        handled_trace_count
    );
    let mut canonical_graph_state = std::collections::BTreeMap::new();
    for event in service
        .store
        .replay_raw_events("conversation", 0, 100)
        .unwrap()
    {
        let key = format!("{}:{}", event.entity_kind, event.entity_id);
        match event.op.as_str() {
            "ADD" | "UPDATE" | "UPSERT" => {
                let payload: Value = serde_json::from_str(&event.payload_json).unwrap();
                canonical_graph_state.insert(
                    key,
                    json!({
                        "entity_kind": event.entity_kind,
                        "id": event.entity_id,
                        "document": payload["document"],
                        "metadata": payload["metadata"],
                    }),
                );
            }
            "DELETE" | "TOMBSTONE" => {
                canonical_graph_state.remove(&key);
            }
            other => panic!("unexpected graph event operation in frozen trace: {other}"),
        }
    }
    let canonical_graph_rows = canonical_graph_state.into_values().collect::<Vec<_>>();
    assert!(canonical_graph_rows.iter().any(|row| {
        row["metadata"]["entity_type"] == "acl_record"
            && row["metadata"]["owner_id"] == "sqlite-ack-owner@example.test"
    }));
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
        expected_digests["digests"]["sqlite_chroma"]["sha256"]
            .as_str()
            .unwrap()
    );
    drop(service);
    remove_test_sqlite(&path);
    remove_test_sqlite(&auth_path);
}
