#[test]
fn provider_event_publication_requires_matching_private_graph_event_and_conversation_edge() {
    let legacy_effect: RecordedWorkerSuccessEffect = serde_json::from_value(json!({
        "contract_version": 1,
        "effect_id": "legacy-effect"
    }))
    .unwrap();
    assert!(legacy_effect.provider_event_publications.is_empty());
    assert!(legacy_effect.provider_event_acknowledgements.is_empty());

    let publication_value = json!({
        "namespace": "tenant-a",
        "conversation_id": "conversation-a",
        "event": {
            "event_id": "agent-event:fixture-001",
            "trace": {
                "run_id": "run-a",
                "turn_id": "turn-a",
                "workflow_node_id": "wf:agent.event.v1:request",
                "step_id": "provider-response:response-a",
                "provider_id": "openai-responses",
                "provider_sequence": 3,
                "client_sse_sequence": 12
            },
            "event": {
                "contract_version": "v1",
                "provider_event_id": "response-a:1",
                "provider_sequence": 1,
                "response_id": "response-a",
                "kind": "field_completed",
                "field_name": "text",
                "value": "safe answer"
            }
        }
    });
    let publication: kogwistar_contracts::DurableProviderEventPublicationV1 =
        serde_json::from_value(publication_value).unwrap();
    let message_id = format!("msg:{}", publication.event.event_id);
    let payload_json = serde_json::to_string(&publication.event).unwrap();
    assert_eq!(publication.event.trace.client_sse_sequence, Some(12));
    assert!(payload_json.contains("\"client_sse_sequence\":12"));
    let metadata = serde_json::from_value(json!({
        "artifact_kind": "lane_message",
        "conversation_id": publication.conversation_id,
        "inbox_id": "agent-provider-events",
        "recipient_id": "agent-event-consumer",
        "sender_id": "agent-runtime",
        "msg_type": "agent.provider_event",
        "purpose": "agent_event",
        "run_id": "run-a",
        "step_id": "provider-response:response-a",
        "correlation_id": publication.event.event_id,
        "idempotency_key": publication.event.event_id,
        "visibility": "private",
        "shared_scope": false,
        "payload_json": payload_json
    }))
    .unwrap();
    let node = GraphMutation {
        scope: GraphScope {
            namespace: "tenant-a".to_owned(),
            workspace_id: None,
            graph_space: Some("conversation".to_owned()),
        },
        table: "gke_nodes".to_owned(),
        entity_kind: "node".to_owned(),
        event_id: "node-event-1".to_owned(),
        op: "ADD".to_owned(),
        payload: json!({"id": message_id}),
        record: GraphRecord {
            id: message_id.clone(),
            document: None,
            metadata,
            embedding: None,
        },
        embedding_dim: 0,
    };
    let edge = GraphMutation {
        scope: node.scope.clone(),
        table: "gke_edges".to_owned(),
        entity_kind: "edge".to_owned(),
        event_id: "edge-event-1".to_owned(),
        op: "ADD".to_owned(),
        payload: json!({
            "id": "event-in-conversation",
            "source_ids": [message_id],
            "target_ids": ["conversation-start-a"],
            "relation": "in_conversation"
        }),
        record: GraphRecord {
            id: "event-in-conversation".to_owned(),
            document: None,
            metadata: serde_json::from_value(json!({
                "conversation_id": publication.conversation_id
            }))
            .unwrap(),
            embedding: None,
        },
        embedding_dim: 0,
    };
    let mutations = vec![node, edge];
    assert!(
        validate_provider_event_publications(
            std::slice::from_ref(&publication),
            &mutations,
            "run-a",
        )
        .is_ok()
    );
    assert!(
        validate_provider_event_publications(
            std::slice::from_ref(&publication),
            &mutations[..1],
            "run-a",
        )
        .is_err()
    );
    let mut cross_scope_edge = mutations.clone();
    cross_scope_edge[1].scope.namespace = "other-tenant".to_owned();
    assert!(
        validate_provider_event_publications(
            std::slice::from_ref(&publication),
            &cross_scope_edge,
            "run-a",
        )
        .is_err()
    );
    let mut ambiguous_edge = mutations.clone();
    ambiguous_edge[1].payload["target_ids"] = json!(["conversation-a", "conversation-b"]);
    assert!(
        validate_provider_event_publications(
            std::slice::from_ref(&publication),
            &ambiguous_edge,
            "run-a",
        )
        .is_err()
    );
    assert!(
        validate_provider_event_publications(
            std::slice::from_ref(&publication),
            &mutations,
            "another-run",
        )
        .is_err()
    );
    let mut cross_namespace = publication.clone();
    cross_namespace.namespace = "other-tenant".to_owned();
    assert!(
        validate_provider_event_publications(&[cross_namespace], &mutations, "run-a",).is_err()
    );
    let mut transient_delta = publication.clone();
    transient_delta.event.event = serde_json::from_value(json!({
        "kind": "text_delta",
        "delta": "not durable"
    }))
    .unwrap();
    assert!(
        validate_provider_event_publications(&[transient_delta], &mutations, "run-a",).is_err()
    );
}
#[test]
fn provider_event_ack_graph_mutations_link_event_to_ack_step() {
    let scope = GraphScope {
        namespace: "tenant-a".to_owned(),
        workspace_id: Some("workspace-a".to_owned()),
        graph_space: Some("conversation".to_owned()),
    };
    let mutations = provider_event_ack_graph_mutations(
        "event-a",
        "run-a",
        "conversation-a",
        "agent.event.v1",
        "handle-event",
        7,
        "effect-a",
        &scope,
    );

    assert_eq!(mutations.len(), 2);
    let step = &mutations[0];
    assert_eq!(step.entity_kind, "node");
    assert_eq!(step.record.id, "wf_step|run-a|7");
    assert_eq!(step.record.metadata["entity_type"], "workflow_step_exec");
    assert_eq!(step.record.metadata["workflow_node_id"], "handle-event");
    assert_eq!(
        step.record.metadata["result_json"],
        "{\"event_id\":\"event-a\",\"handled\":true}"
    );
    assert_eq!(
        step.payload["summary"],
        "step=7, op=handle-event, status=ok"
    );
    assert_eq!(step.scope, scope);

    let edge = &mutations[1];
    assert_eq!(edge.entity_kind, "edge");
    assert_eq!(edge.payload["relation"], "acknowledged_by_step");
    assert_eq!(edge.payload["source_ids"], json!(["msg:event-a"]));
    assert_eq!(edge.payload["target_ids"], json!(["wf_step|run-a|7"]));
    assert_eq!(edge.record.metadata["event_id"], "event-a");
    assert_eq!(edge.record.metadata["step_seq"], 7);
    assert_eq!(edge.scope, scope);

    let replay = provider_event_ack_graph_mutations(
        "event-a",
        "run-a",
        "conversation-a",
        "agent.event.v1",
        "handle-event",
        7,
        "effect-a",
        &scope,
    );
    assert_eq!(
        mutations, replay,
        "replay uses deterministic graph identities"
    );
}

#[test]
fn provider_event_links_match_python_producer_and_handler_contract() {
    let scope = GraphScope {
        namespace: "tenant-a".to_owned(),
        workspace_id: Some("workspace-a".to_owned()),
        graph_space: Some("conversation".to_owned()),
    };
    let producer = ProviderEventStepSnapshot {
        step_seq: 3,
        workflow_id: "workflow-a".to_owned(),
        workflow_node_id: "provider.stream".to_owned(),
        status: "success".to_owned(),
        result_json: "{\"published\":true}".to_owned(),
        event_seq: 8,
        transition_id: "transition-producer".to_owned(),
    };
    let handler = ProviderEventStepSnapshot {
        step_seq: 5,
        workflow_id: "workflow-a".to_owned(),
        workflow_node_id: "agent.event.tool".to_owned(),
        status: "success".to_owned(),
        result_json: "{\"tool\":\"read\"}".to_owned(),
        event_seq: 12,
        transition_id: "transition-handler".to_owned(),
    };
    let snapshots = BTreeMap::from([(3, producer), (5, handler)]);
    let mutations = provider_event_step_link_mutations(
        "event-a",
        "run-a",
        "conversation-a",
        Some(3),
        5,
        &snapshots,
        &scope,
    );

    assert_eq!(mutations.len(), 4);
    assert_eq!(mutations[0].record.id, "wf_step|run-a|3");
    assert_eq!(mutations[1].record.id, "wf_step|run-a|5");
    assert_eq!(mutations[0].record.metadata["event_seq"], 8);
    assert_eq!(
        mutations[1].record.metadata["transition_id"],
        "transition-handler"
    );
    assert_eq!(mutations[2].payload["relation"], "emitted_by_step");
    assert_eq!(mutations[2].payload["source_ids"], json!(["msg:event-a"]));
    assert_eq!(
        mutations[2].payload["target_ids"],
        json!(["wf_step|run-a|3"])
    );
    assert_eq!(mutations[3].payload["relation"], "handled_by_step");
    assert_eq!(
        mutations[3].payload["target_ids"],
        json!(["wf_step|run-a|5"])
    );
    assert_ne!(mutations[2].record.id, mutations[3].record.id);
    assert_eq!(mutations[0].scope, scope);

    let replay = provider_event_step_link_mutations(
        "event-a",
        "run-a",
        "conversation-a",
        Some(3),
        5,
        &snapshots,
        &scope,
    );
    assert_eq!(
        mutations, replay,
        "replay preserves deterministic step/link records"
    );

    let shared_step = BTreeMap::from([(3, snapshots[&3].clone())]);
    let shared_mutations = provider_event_step_link_mutations(
        "event-b",
        "run-a",
        "conversation-a",
        Some(3),
        3,
        &shared_step,
        &scope,
    );
    assert_eq!(
        shared_mutations
            .iter()
            .filter(|mutation| mutation.entity_kind == "node")
            .count(),
        1,
        "producer and handler sharing a step reuse one step node"
    );
    assert_eq!(
        shared_mutations
            .iter()
            .filter(|mutation| mutation.entity_kind == "edge")
            .count(),
        2
    );
}

#[test]
fn provider_event_step_snapshot_lookup_rejects_wrong_scope_and_missing_steps() {
    let event = |seq: i64, run_id: &str| ServerRunEvent {
        seq,
        run_id: run_id.to_owned(),
        event_type: "workflow.recorded_transition.v1".to_owned(),
        payload_json: json!({
            "transition_id": format!("transition-{seq}"),
            "reduced": {
                "state": {
                    "run_id": run_id,
                    "conversation_id": "conversation-a",
                    "workflow_id": "workflow-a",
                    "last_step_seq": seq,
                    "last_node_id": "agent.event.receive"
                },
                "errors": [],
                "result": {"ok": true}
            }
        })
        .to_string(),
        created_at_ms: 100 + seq,
    };

    let found = provider_event_step_snapshots(
        vec![event(4, "run-a")],
        "run-a",
        "conversation-a",
        "workflow-a",
        &[4],
    )
    .unwrap();
    assert_eq!(found[&4].workflow_node_id, "agent.event.receive");
    assert!(
        provider_event_step_snapshots(
            vec![event(4, "other-run")],
            "run-a",
            "conversation-a",
            "workflow-a",
            &[4],
        )
        .is_err()
    );
    assert!(
        provider_event_step_snapshots(Vec::new(), "run-a", "conversation-a", "workflow-a", &[4],)
            .is_err()
    );
}

#[test]
fn provider_event_acknowledgement_contract_rejects_invalid_or_duplicate_claims() {
    use kogwistar_runtime::ProviderEventAcknowledgementV1;

    let valid = ProviderEventAcknowledgementV1 {
        message_id: "msg:event-1".to_owned(),
        claimed_by: "consumer#lease-1".to_owned(),
        event_id: "event-1".to_owned(),
    };
    assert!(validate_provider_event_acknowledgements(std::slice::from_ref(&valid)).is_ok());
    assert!(validate_provider_event_acknowledgements(&[valid.clone(), valid.clone()]).is_err());
    assert!(
        validate_provider_event_acknowledgements(&[ProviderEventAcknowledgementV1 {
            message_id: "msg:other".to_owned(),
            claimed_by: "consumer#lease-1".to_owned(),
            event_id: "event-1".to_owned(),
        }])
        .is_err()
    );
    assert!(
        validate_provider_event_acknowledgements(&[ProviderEventAcknowledgementV1 {
            message_id: "msg:event-1".to_owned(),
            claimed_by: " ".to_owned(),
            event_id: "event-1".to_owned(),
        }])
        .is_err()
    );
    assert!(
        validate_provider_event_acknowledgements(
            &(0..101)
                .map(|index| ProviderEventAcknowledgementV1 {
                    message_id: format!("msg:event-{index}"),
                    claimed_by: "consumer#lease-1".to_owned(),
                    event_id: format!("event-{index}"),
                })
                .collect::<Vec<_>>()
        )
        .is_err()
    );
}

#[tokio::test]
#[ignore = "requires KOGWISTAR_TEST_PG_DSN; run explicitly against disposable PostgreSQL"]
async fn postgres_provider_event_ack_persists_python_causal_links_atomically() {
    use crate::native_answer_generation::OpenAiCompatibleChatConfig;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN").unwrap();
    let schema = format!("pec_{}", uuid::Uuid::new_v4().simple());
    let auth_path = std::env::temp_dir().join(format!(
        "kogwistar-provider-event-causal-auth-{}.sqlite",
        uuid::Uuid::new_v4().simple()
    ));
    let auth = SqliteAuthStore::open(&auth_path).unwrap();
    auth.resolve_external_identity(ResolveExternalIdentity {
        issuer: "test".to_owned(),
        subject: "causal-owner".to_owned(),
        email: "causal-owner@example.test".to_owned(),
        display_name: None,
        new_user_id: "causal-owner".to_owned(),
        default_role: "rw".to_owned(),
        default_ns: "alpha".to_owned(),
    })
    .await
    .unwrap();
    drop(auth);
    let owner_auth = AuthConfig {
        auth_db_url: Some(format!("sqlite:///{}", auth_path.display())),
        ..AuthConfig::default()
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let provider_address = listener.local_addr().unwrap();
    let provider_task = tokio::spawn(async move {
        for _ in 0..2 {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let (header_end, content_length) = loop {
                let mut chunk = [0_u8; 4096];
                let count = stream.read(&mut chunk).await.unwrap();
                assert_ne!(count, 0);
                request.extend_from_slice(&chunk[..count]);
                if let Some(header_end) = request.windows(4).position(|part| part == b"\r\n\r\n") {
                    let header_end = header_end + 4;
                    let headers =
                        String::from_utf8_lossy(&request[..header_end]).to_ascii_lowercase();
                    let content_length = headers
                        .lines()
                        .find_map(|line| line.strip_prefix("content-length: "))
                        .unwrap()
                        .trim()
                        .parse::<usize>()
                        .unwrap();
                    if request.len() >= header_end + content_length {
                        break (header_end, content_length);
                    }
                    while request.len() < header_end + content_length {
                        let mut chunk = [0_u8; 4096];
                        let count = stream.read(&mut chunk).await.unwrap();
                        assert_ne!(count, 0);
                        request.extend_from_slice(&chunk[..count]);
                    }
                    break (header_end, content_length);
                }
            };
            let body: Value =
                serde_json::from_slice(&request[header_end..header_end + content_length]).unwrap();
            assert_eq!(body["model"], "event-fixture-model");
            assert_eq!(body["stream"], true);
            let mut body = String::new();
            for chunk in [
                json!({"id":"causal-response","choices":[{"delta":{"content":"streamed text"},"finish_reason":null}]}).to_string(),
                json!({"id":"causal-response","choices":[{"delta":{},"finish_reason":"stop"}]}).to_string(),
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
        }
    });
    let provider = OpenAiCompatibleChatConfig::new(
        format!("http://{provider_address}/v1/chat/completions"),
        "event-fixture-model",
        None,
    )
    .unwrap()
    .with_streaming(true);
    let service = PostgresRunApplicationService::from_dsn(&dsn, &schema)
        .unwrap()
        .with_owner_auth_config(&owner_auth)
        .await
        .unwrap()
        .with_chat_completion_provider(provider);
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

    let run_id = "provider-event-causal-run";
    let workflow_id = "provider-event-causal-workflow";
    let conversation_id = "provider-event-causal-conversation";
    let principal = json!({
        "sub": "causal-owner",
        "user_id": "causal-owner",
        "role": "rw",
        "ns": ["alpha"],
        "storage_ns": "alpha",
        "capabilities": ["read_graph", "write_graph"]
    });
    let binding = runtime_owner_binding_v1(&principal, &service.capabilities).unwrap();
    service
        .store
        .create_server_run(ServerRunCreate {
            run_id: run_id.to_owned(),
            conversation_id: conversation_id.to_owned(),
            workflow_id: workflow_id.to_owned(),
            user_id: Some("causal-owner".to_owned()),
            user_turn_node_id: "causal-user-turn".to_owned(),
            status: "queued".to_owned(),
        })
        .await
        .unwrap();
    service
        .store
        .append_server_run_event(
            run_id,
            "run.created",
            json!({"runtime_owner_binding": binding}).to_string(),
        )
        .await
        .unwrap();

    let start: kogwistar_runtime::RecordedRuntimeTransition = serde_json::from_value(json!({
        "contract_version": 1,
        "transition_id": "causal-start",
        "expected_event_seq": 1,
        "kind": "start",
        "run_id": run_id,
        "workflow_id": workflow_id,
        "conversation_id": conversation_id,
        "user_id": "causal-owner",
        "user_turn_node_id": "causal-user-turn",
        "step_seq": 0,
        "node_id": "provider-step",
        "token_id": run_id,
        "initial_state": {},
        "frontier": {
            "pending": [["provider-step", 0, run_id, null]],
            "suspended": [], "join_node_ids": [], "join_outstanding": [], "join_waiters": {}
        }
    }))
    .unwrap();
    service
        .store
        .apply_recorded_runtime_transition(start, false)
        .await
        .unwrap();
    let provider_request = kogwistar_runtime::RuntimeNativeStepRequest {
        executor_target: kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET.to_owned(),
        operation: "agent.event.request".to_owned(),
        op_version: "v1".to_owned(),
        run_id: run_id.to_owned(),
        workflow_id: workflow_id.to_owned(),
        node_id: "provider-step".to_owned(),
        conversation_id: conversation_id.to_owned(),
        turn_node_id: Some("causal-user-turn".to_owned()),
        token_id: run_id.to_owned(),
        parent_token_id: None,
        step_seq: 0,
        state: serde_json::from_value(json!({
            "provider_request": {
                "contract_version": "v1",
                "model": "event-fixture-model",
                "instructions": "test only",
                "input": "emit one response"
            }
        }))
        .unwrap(),
        runtime_routes: Vec::new(),
        ephemeral_context: Map::from_iter([("_trace_attempt".to_owned(), json!(2))]),
    };
    let provider_result = service
        .execute_native_provider_event_request(
            kogwistar_runtime::RecordedWorkerHandoff {
                message_id: "provider-request-handoff".to_owned(),
                claimed_by: "provider-request-worker#1".to_owned(),
                run_id: run_id.to_owned(),
                step_id: "provider-step".to_owned(),
                correlation_id: run_id.to_owned(),
            },
            provider_request.clone(),
        )
        .await
        .unwrap();
    let provider_effect = provider_result
        .effect
        .expect("provider request returns an effect");
    let provider_result_json = provider_effect.result.as_ref().unwrap();
    assert!(
        provider_result_json["semantic_event_count"]
            .as_i64()
            .unwrap()
            > 0
    );
    assert_eq!(
        provider_result_json["final_state"]["agent_event_route"],
        "receive"
    );
    let events_after_first_attempt = service
        .store
        .server_run_events(run_id, 0, 100)
        .await
        .unwrap();
    let provider_retry = service
        .execute_native_provider_event_request(
            kogwistar_runtime::RecordedWorkerHandoff {
                message_id: "provider-request-handoff-retry".to_owned(),
                claimed_by: "provider-request-worker#2".to_owned(),
                run_id: run_id.to_owned(),
                step_id: "provider-step".to_owned(),
                correlation_id: run_id.to_owned(),
            },
            provider_request,
        )
        .await
        .unwrap();
    provider_task.await.unwrap();
    let retry_effect = provider_retry
        .effect
        .expect("provider retry returns an effect");
    assert_eq!(provider_effect.result, retry_effect.result);
    let events_after_retry = service
        .store
        .server_run_events(run_id, 0, 100)
        .await
        .unwrap();
    assert_eq!(events_after_first_attempt, events_after_retry);
    let producer_event_seq = service
        .store
        .server_run_events(run_id, 0, 100)
        .await
        .unwrap()
        .last()
        .unwrap()
        .seq;

    let producer: kogwistar_runtime::RecordedRuntimeTransition = serde_json::from_value(json!({
        "contract_version": 1,
        "transition_id": "causal-producer-step",
        "expected_event_seq": producer_event_seq,
        "kind": "recorded_step_success",
        "run_id": run_id,
        "workflow_id": workflow_id,
        "conversation_id": conversation_id,
        "step_seq": 0,
        "node_id": "provider-step",
        "token_id": run_id,
        "state_update": provider_effect.state_update,
        "frontier": {
            "pending": [["handler-step", 0, run_id, null]],
            "suspended": [], "join_node_ids": [], "join_outstanding": [], "join_waiters": {}
        },
        "result": provider_effect.result
    }))
    .unwrap();
    service
        .store
        .apply_recorded_runtime_transition(producer, false)
        .await
        .unwrap();
    let handler_event_seq = service
        .store
        .server_run_events(run_id, 0, 100)
        .await
        .unwrap()
        .last()
        .unwrap()
        .seq;
    let mut receive_request = kogwistar_runtime::RuntimeNativeStepRequest {
        executor_target: kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET.to_owned(),
        operation: "agent.event.receive".to_owned(),
        op_version: "v1".to_owned(),
        run_id: run_id.to_owned(),
        workflow_id: workflow_id.to_owned(),
        node_id: "receive-step".to_owned(),
        conversation_id: conversation_id.to_owned(),
        turn_node_id: Some("causal-user-turn".to_owned()),
        token_id: run_id.to_owned(),
        parent_token_id: None,
        step_seq: 1,
        state: Map::new(),
        runtime_routes: vec![kogwistar_runtime::RuntimeStaticRoute {
            edge_id: "receive-to-ack".to_owned(),
            source_node_id: "receive-step".to_owned(),
            target_node_id: "ack-step".to_owned(),
            aliases: Vec::new(),
            join_mask: 0,
            predicate: Some("event_requires_no_action".to_owned()),
            multiplicity: "one".to_owned(),
            is_default: false,
            priority: 100,
            source_fanout: false,
        }],
        ephemeral_context: Map::new(),
    };
    let receive_principal = service
        .native_read_principal(&receive_request, None)
        .await
        .unwrap_or_else(|_| panic!("the active run owner retains read_graph authority"));
    let claim = service
        .claim_native_provider_event_context(&receive_request, &receive_principal)
        .await
        .unwrap()
        .expect("published event is visible and claimable by its run owner");
    let event_id = claim["event"]["event_id"].as_str().unwrap().to_owned();
    assert_eq!(claim["event"]["event"]["kind"], "response_started");
    assert_eq!(
        claim["event"]["trace"]["step_id"],
        "provider-response:causal-response"
    );
    assert_eq!(claim["event"]["trace"]["attempt"], 1);

    receive_request.ephemeral_context.insert(
        "authorized_provider_event".to_owned(),
        claim["event"].clone(),
    );
    receive_request.ephemeral_context.insert(
        "authorized_provider_event_claim".to_owned(),
        claim["claim"].clone(),
    );
    let received = native_result_from_request(
        RecordedWorkerHandoff {
            message_id: "provider-receive-handoff".to_owned(),
            claimed_by: "rust-native-worker#1".to_owned(),
            run_id: run_id.to_owned(),
            step_id: "receive-step".to_owned(),
            correlation_id: run_id.to_owned(),
        },
        receive_request,
        &kogwistar_runtime::NativeOperationRegistry::default(),
    )
    .unwrap();
    let receive_effect = received.effect.expect("native receive returns a workflow effect");
    assert_eq!(receive_effect.successors.len(), 1);
    assert_eq!(receive_effect.successors[0].node_id, "ack-step");
    assert!(receive_effect.provider_event_acknowledgements.is_empty());

    let handler: kogwistar_runtime::RecordedRuntimeTransition = serde_json::from_value(json!({
        "contract_version": 1,
        "transition_id": "causal-handler-step",
        "expected_event_seq": handler_event_seq,
        "kind": "recorded_step_success",
        "run_id": run_id,
        "workflow_id": workflow_id,
        "conversation_id": conversation_id,
        "step_seq": 1,
        "node_id": "handler-step",
        "token_id": run_id,
        "state_update": receive_effect.state_update,
        "frontier": {
            "pending": [["ack-step", 0, run_id, null]],
            "suspended": [], "join_node_ids": [], "join_outstanding": [], "join_waiters": {}
        },
        "result": receive_effect.result
    }))
    .unwrap();
    service
        .store
        .apply_recorded_runtime_transition(handler, false)
        .await
        .unwrap();

    service
        .store
        .project_lane_message(NewProjectedLaneMessage {
            message_id: "causal-ack-runtime-lane".to_owned(),
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
            payload_json: Some(
                json!({
                    "workflow_id": workflow_id,
                    "conversation_id": conversation_id,
                    "token_id": run_id,
                    "parent_token_id": null,
                    "op": "agent.event.ack"
                })
                .to_string(),
            ),
            error_json: None,
        })
        .await
        .unwrap();
    let claimed_runtime = service
        .store
        .claim_projected_lane_messages("workflow", "workflow-runtime", "causal-worker", 1, 60)
        .await
        .unwrap();
    assert_eq!(claimed_runtime.len(), 1);
    let runtime_lane = &claimed_runtime[0];
    let handoff = kogwistar_runtime::RecordedWorkerHandoff {
        message_id: runtime_lane.message_id.clone(),
        claimed_by: runtime_lane.claimed_by.clone().unwrap(),
        run_id: run_id.to_owned(),
        step_id: "ack-step".to_owned(),
        correlation_id: run_id.to_owned(),
    };
    let ack_request = kogwistar_runtime::RuntimeNativeStepRequest {
        executor_target: kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET.to_owned(),
        operation: "agent.event.ack".to_owned(),
        op_version: "v1".to_owned(),
        run_id: run_id.to_owned(),
        workflow_id: workflow_id.to_owned(),
        node_id: "ack-step".to_owned(),
        conversation_id: conversation_id.to_owned(),
        turn_node_id: Some("causal-user-turn".to_owned()),
        token_id: run_id.to_owned(),
        parent_token_id: None,
        step_seq: 2,
        state: Map::from_iter([
            ("agent_event_current".to_owned(), claim["event"].clone()),
            ("agent_event_claim".to_owned(), claim["claim"].clone()),
            ("agent_event_route".to_owned(), json!("no_action")),
            ("agent_event_handling_step_seq".to_owned(), json!(1)),
        ]),
        runtime_routes: Vec::new(),
        ephemeral_context: Map::new(),
    };
    let ack_effect = kogwistar_runtime::execute_native_step(&ack_request).unwrap();
    let request =
        |effect: kogwistar_runtime::RecordedWorkerSuccessEffect| RuntimeWorkerResultRequest {
            handoff: handoff.clone(),
            transition: None,
            effect: Some(effect),
        };

    let mut wrong_claim_effect = ack_effect.clone();
    wrong_claim_effect.provider_event_acknowledgements[0].claimed_by =
        "wrong-claimer".to_owned();
    let wrong_claim = service
        .apply_runtime_result_input(
            request(wrong_claim_effect),
            None,
            None,
            kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET,
        )
        .await;
    assert_eq!(wrong_claim.status, StatusCode::CONFLICT.as_u16());
    assert_eq!(
        service
            .store
            .get_projected_lane_message(&format!("msg:{event_id}"))
            .await
            .unwrap()
            .unwrap()
            .status,
        "claimed"
    );
    assert_eq!(
        service
            .store
            .read_recorded_runtime_state(run_id, workflow_id, conversation_id)
            .await
            .unwrap()
            .unwrap()
            .last_step_seq,
        1
    );

    let (fault_setup, fault_connection) = tokio_postgres::connect(&dsn, tokio_postgres::NoTls)
        .await
        .unwrap();
    let fault_task = tokio::spawn(fault_connection);
    fault_setup.batch_execute(&format!(
        "CREATE FUNCTION \"{schema}\".reject_provider_event_handled() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN IF NEW.event_type='workflow.provider_event.handled.v1' THEN RAISE EXCEPTION 'injected causal ack failure'; END IF; RETURN NEW; END; $$; CREATE TRIGGER reject_provider_event_handled BEFORE INSERT ON \"{schema}\".server_run_events FOR EACH ROW EXECUTE FUNCTION \"{schema}\".reject_provider_event_handled();"
    )).await.unwrap();
    drop(fault_setup);
    fault_task.await.unwrap().unwrap();

    let failed = service
        .apply_runtime_result_input(
            request(ack_effect.clone()),
            None,
            None,
            kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET,
        )
        .await;
    assert_eq!(failed.status, StatusCode::CONFLICT.as_u16());
    assert_eq!(
        service
            .store
            .get_projected_lane_message(&format!("msg:{event_id}"))
            .await
            .unwrap()
            .unwrap()
            .status,
        "claimed"
    );
    assert_eq!(
        service
            .store
            .read_recorded_runtime_state(run_id, workflow_id, conversation_id)
            .await
            .unwrap()
            .unwrap()
            .last_step_seq,
        1
    );
    let scope = GraphScope {
        namespace: "alpha".to_owned(),
        workspace_id: None,
        graph_space: Some("conversation".to_owned()),
    };
    let causal_edges = service
        .conversation_store
        .graph_projection_records(GraphProjectionRead {
            scope: scope.clone(),
            table: "gke_edges".to_owned(),
            ids: None,
            metadata: MetadataFilter::default(),
            limit: 1000,
        })
        .await
        .unwrap();
    assert!(!causal_edges.iter().any(|record| {
        record
            .document
            .as_deref()
            .is_some_and(|doc| doc.contains("emitted_by_step") || doc.contains("handled_by_step"))
    }));

    let (fault_cleanup, fault_connection) = tokio_postgres::connect(&dsn, tokio_postgres::NoTls)
        .await
        .unwrap();
    let fault_cleanup_task = tokio::spawn(fault_connection);
    fault_cleanup.batch_execute(&format!(
        "DROP TRIGGER reject_provider_event_handled ON \"{schema}\".server_run_events; DROP FUNCTION \"{schema}\".reject_provider_event_handled();"
    )).await.unwrap();
    drop(fault_cleanup);
    fault_cleanup_task.await.unwrap().unwrap();

    let applied = service
        .apply_runtime_result_input(
            request(ack_effect.clone()),
            None,
            None,
            kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET,
        )
        .await;
    assert_eq!(
        applied.status,
        StatusCode::OK.as_u16(),
        "{}",
        String::from_utf8_lossy(&applied.body)
    );
    let events_before_exact_retry = service
        .store
        .list_server_run_events(run_id, 0, usize::MAX)
        .await
        .unwrap();
    let exact_retry = service
        .apply_runtime_result_input(
            request(ack_effect),
            None,
            None,
            kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET,
        )
        .await;
    assert_eq!(exact_retry.status, StatusCode::OK.as_u16());
    assert_eq!(
        service
            .store
            .list_server_run_events(run_id, 0, usize::MAX)
            .await
            .unwrap(),
        events_before_exact_retry,
        "an exact post-commit retry must not append another completion or handled event"
    );
    let committed_edges = service
        .conversation_store
        .graph_projection_records(GraphProjectionRead {
            scope,
            table: "gke_edges".to_owned(),
            ids: None,
            metadata: MetadataFilter::default(),
            limit: 1000,
        })
        .await
        .unwrap();
    let relations = committed_edges
        .iter()
        .filter_map(|record| {
            record
                .document
                .as_deref()
                .and_then(|doc| serde_json::from_str::<Value>(doc).ok())
                .and_then(|doc| doc["relation"].as_str().map(str::to_owned))
        })
        .collect::<Vec<_>>();
    assert!(
        relations
            .iter()
            .any(|relation| relation == "emitted_by_step")
    );
    assert!(
        relations
            .iter()
            .any(|relation| relation == "handled_by_step")
    );
    assert_eq!(
        service
            .store
            .get_projected_lane_message(&format!("msg:{event_id}"))
            .await
            .unwrap()
            .unwrap()
            .status,
        "completed"
    );

    let fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/agent-durable-provider-event-v1.json"
    )))
    .unwrap();
    let fixture_scope = GraphScope {
        namespace: "alpha".to_owned(),
        workspace_id: None,
        graph_space: Some("conversation".to_owned()),
    };
    let baseline_node_ids = service
        .conversation_store
        .graph_projection_records(GraphProjectionRead {
            scope: fixture_scope.clone(),
            table: "gke_nodes".to_owned(),
            ids: None,
            metadata: MetadataFilter::default(),
            limit: 1000,
        })
        .await
        .unwrap()
        .into_iter()
        .map(|record| record.id)
        .collect::<std::collections::HashSet<_>>();
    let baseline_edge_ids = service
        .conversation_store
        .graph_projection_records(GraphProjectionRead {
            scope: fixture_scope.clone(),
            table: "gke_edges".to_owned(),
            ids: None,
            metadata: MetadataFilter::default(),
            limit: 1000,
        })
        .await
        .unwrap()
        .into_iter()
        .map(|record| record.id)
        .collect::<std::collections::HashSet<_>>();
    // The disposable schema isolates this fixed run ID, making fixture-derived IDs stable.
    let fixture_run_id = "provider-event-fixture-run".to_owned();
    let fixture_workflow_id = "provider-event-fixture-workflow";
    let fixture_conversation_id = fixture["publication"]["conversation_id"]
        .as_str()
        .unwrap();
    let fixture_turn_id = fixture["envelope"]["trace"]["turn_id"]
        .as_str()
        .unwrap();
    let fixture_principal = json!({
        "sub": "causal-owner", "user_id": "causal-owner", "role": "rw",
        "ns": ["alpha"], "storage_ns": "alpha",
        "capabilities": ["read_graph", "write_graph"]
    });
    let fixture_binding = runtime_owner_binding_v1(&fixture_principal, &service.capabilities)
        .unwrap();
    service
        .store
        .create_server_run(ServerRunCreate {
            run_id: fixture_run_id.clone(),
            conversation_id: fixture_conversation_id.to_owned(),
            workflow_id: fixture_workflow_id.to_owned(),
            user_id: Some("causal-owner".to_owned()),
            user_turn_node_id: fixture_turn_id.to_owned(),
            status: "queued".to_owned(),
        })
        .await
        .unwrap();
    let fixture_admission = service
        .store
        .append_server_run_event(
            &fixture_run_id,
            "run.created",
            json!({"runtime_owner_binding": fixture_binding}).to_string(),
        )
        .await
        .unwrap();
    let fixture_start: kogwistar_runtime::RecordedRuntimeTransition =
        serde_json::from_value(json!({
            "contract_version": 1,
            "transition_id": "fixture-run-start",
            "expected_event_seq": fixture_admission.seq,
            "kind": "start",
            "run_id": fixture_run_id,
            "workflow_id": fixture_workflow_id,
            "conversation_id": fixture_conversation_id,
            "user_id": "causal-owner",
            "user_turn_node_id": fixture_turn_id,
            "step_seq": 0,
            "node_id": "fixture-provider-step",
            "token_id": fixture_run_id,
            "initial_state": {},
            "frontier": {
                "pending": [["fixture-handler-step", 0, fixture_run_id, null]],
                "suspended": [], "join_node_ids": [],
                "join_outstanding": [], "join_waiters": {}
            }
        }))
        .unwrap();
    let _fixture_started = service
        .store
        .apply_recorded_runtime_transition(fixture_start, false)
        .await
        .unwrap();
    let mut fixture_trace: kogwistar_contracts::EventTraceMetadataV1 =
        serde_json::from_value(fixture["envelope"]["trace"].clone()).unwrap();
    fixture_trace.run_id = fixture_run_id.clone();
    fixture_trace.turn_id = fixture_turn_id.to_owned();
    fixture_trace.workflow_node_id = "fixture-provider-step".to_owned();
    fixture_trace.producer_step_seq = None;
    let fixture_event = kogwistar_contracts::durable_provider_event(
        &fixture_trace,
        serde_json::from_value(fixture["envelope"]["event"].clone()).unwrap(),
    );
    assert_eq!(
        serde_json::to_value(&fixture_event.event).unwrap()["kind"],
        "field_completed"
    );
    let fixture_message_id = format!("msg:{}", fixture_event.event_id);
    service
        .store
        .project_lane_message(NewProjectedLaneMessage {
            message_id: fixture_message_id.clone(),
            namespace: "alpha".to_owned(),
            purpose: "agent_event".to_owned(),
            inbox_id: "agent-provider-events".to_owned(),
            conversation_id: fixture_conversation_id.to_owned(),
            recipient_id: "agent-event-consumer".to_owned(),
            sender_id: "agent-runtime".to_owned(),
            msg_type: "agent.provider_event".to_owned(),
            status: "pending".to_owned(),
            created_at: 2,
            available_at: 2,
            run_id: Some(fixture_run_id.clone()),
            step_id: Some(fixture_event.trace.step_id.clone()),
            correlation_id: Some(fixture_event.event_id.clone()),
            payload_json: Some(serde_json::to_string(&fixture_event).unwrap()),
            error_json: None,
        })
        .await
        .unwrap();
    let fixture_claimed_by = "fixture-event-consumer#1";
    let fixture_claimed = service
        .store
        .claim_projected_lane_messages_filtered(LaneMessageClaimFilter {
            namespace: "alpha".to_owned(),
            inbox_id: "agent-provider-events".to_owned(),
            claimed_by: fixture_claimed_by.to_owned(),
            message_ids: Some(vec![fixture_message_id.clone()]),
            run_id: Some(fixture_run_id.clone()),
            msg_type: Some("agent.provider_event".to_owned()),
            recipient_id: Some("agent-event-consumer".to_owned()),
            limit: 1,
            lease_seconds: 60,
        })
        .await
        .unwrap();
    assert_eq!(fixture_claimed.len(), 1);
    let fixture_claim = json!({
        "message_id": fixture_message_id,
        "claimed_by": fixture_claimed_by,
        "event_id": fixture_event.event_id
    });
    let previous_events = service
        .store
        .list_server_run_events(&fixture_run_id, 0, usize::MAX)
        .await
        .unwrap();
    let fixture_handler_seq = 1;
    let fixture_handler: kogwistar_runtime::RecordedRuntimeTransition =
        serde_json::from_value(json!({
            "contract_version": 1,
            "transition_id": "fixture-causal-handler",
            "expected_event_seq": previous_events.last().unwrap().seq,
            "kind": "recorded_step_success",
            "run_id": fixture_run_id,
            "workflow_id": fixture_workflow_id,
            "conversation_id": fixture_conversation_id,
            "step_seq": fixture_handler_seq,
            "node_id": "fixture-handler-step",
            "token_id": fixture_run_id,
            "state_update": [["u", {
                "agent_event_current": fixture_event.clone(),
                "agent_event_claim": fixture_claim.clone(),
                "agent_event_route": "final",
                "agent_event_handling_step_seq": fixture_handler_seq
            }]],
            "frontier": {
                "pending": [["fixture-ack-step", 0, fixture_run_id, null]],
                "suspended": [], "join_node_ids": [],
                "join_outstanding": [], "join_waiters": {}
            },
            "result": {"handled": true}
        }))
        .unwrap();
    service
        .store
        .apply_recorded_runtime_transition(fixture_handler, false)
        .await
        .unwrap();

    let fixture_ack_message_id = "fixture-causal-ack-runtime";
    service
        .store
        .project_lane_message(NewProjectedLaneMessage {
            message_id: fixture_ack_message_id.to_owned(),
            namespace: "workflow".to_owned(),
            purpose: "system".to_owned(),
            inbox_id: "workflow-runtime".to_owned(),
            conversation_id: fixture_conversation_id.to_owned(),
            recipient_id: "rust-native-worker".to_owned(),
            sender_id: "rust-scheduler".to_owned(),
            msg_type: "workflow.step.execute".to_owned(),
            status: "pending".to_owned(),
            created_at: 2,
            available_at: 2,
            run_id: Some(fixture_run_id.clone()),
            step_id: Some("fixture-ack-step".to_owned()),
            correlation_id: Some(fixture_run_id.clone()),
            payload_json: Some(json!({
                "workflow_id": fixture_workflow_id,
                "conversation_id": fixture_conversation_id,
                "token_id": fixture_run_id,
                "parent_token_id": null,
                "op": "agent.event.ack"
            }).to_string()),
            error_json: None,
        })
        .await
        .unwrap();
    let fixture_runtime_claim = service
        .store
        .claim_projected_lane_messages_filtered(LaneMessageClaimFilter {
            namespace: "workflow".to_owned(),
            inbox_id: "workflow-runtime".to_owned(),
            claimed_by: "fixture-runtime-worker#1".to_owned(),
            message_ids: Some(vec![fixture_ack_message_id.to_owned()]),
            run_id: Some(fixture_run_id.clone()),
            msg_type: Some("workflow.step.execute".to_owned()),
            recipient_id: Some("rust-native-worker".to_owned()),
            limit: 1,
            lease_seconds: 60,
        })
        .await
        .unwrap();
    assert_eq!(fixture_runtime_claim.len(), 1);
    let fixture_handoff = kogwistar_runtime::RecordedWorkerHandoff {
        message_id: fixture_ack_message_id.to_owned(),
        claimed_by: "fixture-runtime-worker#1".to_owned(),
        run_id: fixture_run_id.clone(),
        step_id: "fixture-ack-step".to_owned(),
        correlation_id: fixture_run_id.clone(),
    };
    let fixture_ack_effect = kogwistar_runtime::execute_native_step(
        &kogwistar_runtime::RuntimeNativeStepRequest {
            executor_target: kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET.to_owned(),
            operation: "agent.event.ack".to_owned(),
            op_version: "v1".to_owned(),
            run_id: fixture_run_id.clone(),
            workflow_id: fixture_workflow_id.to_owned(),
            node_id: "fixture-ack-step".to_owned(),
            conversation_id: fixture_conversation_id.to_owned(),
            turn_node_id: Some(fixture_turn_id.to_owned()),
            token_id: fixture_run_id.clone(),
            parent_token_id: None,
            step_seq: fixture_handler_seq + 1,
            state: serde_json::from_value(json!({
                "agent_event_current": fixture_event.clone(),
                "agent_event_claim": fixture_claim,
                "agent_event_route": "final",
                "agent_event_handling_step_seq": fixture_handler_seq
            })).unwrap(),
            runtime_routes: Vec::new(),
            ephemeral_context: Map::new(),
        },
    )
    .unwrap();
    let fixture_ack_response = service
        .apply_runtime_result_input(
            RuntimeWorkerResultRequest {
                handoff: fixture_handoff,
                transition: None,
                effect: Some(fixture_ack_effect),
            },
            None,
            None,
            kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET,
        )
        .await;
    assert_eq!(
        fixture_ack_response.status,
        StatusCode::OK.as_u16(),
        "{}",
        String::from_utf8_lossy(&fixture_ack_response.body)
    );
    let fixture_lane = service
        .store
        .get_projected_lane_message(&fixture_message_id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(fixture_lane.status, "completed");
    let fixture_payload: Value =
        serde_json::from_str(fixture_lane.payload_json.as_deref().unwrap()).unwrap();
    assert_eq!(fixture_payload["event"], fixture["envelope"]["event"]);
    assert_eq!(fixture_payload["trace"]["run_id"], fixture_run_id);
    let fixture_links = service
        .conversation_store
        .graph_projection_records(GraphProjectionRead {
            scope: GraphScope {
                namespace: "alpha".to_owned(),
                workspace_id: None,
                graph_space: Some("conversation".to_owned()),
            },
            table: "gke_edges".to_owned(),
            ids: None,
            metadata: MetadataFilter::default(),
            limit: 1000,
        })
        .await
        .unwrap()
        .into_iter()
        .filter_map(|record| {
            let document = record.document?;
            let payload = serde_json::from_str::<Value>(&document).ok()?;
            let targets_fixture_event = payload["source_ids"]
                .as_array()?
                .iter()
                .any(|source| source.as_str() == Some(fixture_message_id.as_str()));
            targets_fixture_event
                .then(|| payload["relation"].as_str().map(str::to_owned))
                .flatten()
        })
        .collect::<Vec<_>>();
    assert!(fixture_links.iter().any(|relation| relation == "handled_by_step"));
    assert_eq!(
        service
            .store
            .get_projected_lane_message(&format!("msg:{}", fixture_event.event_id))
            .await
            .unwrap()
            .unwrap()
        .status,
        "completed"
    );
    let fixture_nodes = service
        .conversation_store
        .graph_projection_records(GraphProjectionRead {
            scope: fixture_scope.clone(),
            table: "gke_nodes".to_owned(),
            ids: None,
            metadata: MetadataFilter::default(),
            limit: 1000,
        })
        .await
        .unwrap();
    let fixture_edges = service
        .conversation_store
        .graph_projection_records(GraphProjectionRead {
            scope: fixture_scope,
            table: "gke_edges".to_owned(),
            ids: None,
            metadata: MetadataFilter::default(),
            limit: 1000,
        })
        .await
        .unwrap();
    let mut canonical_graph_rows = fixture_nodes
        .iter()
        .filter(|record| !baseline_node_ids.contains(&record.id))
        .map(|record| {
            json!({
                "table": "gke_nodes",
                "id": record.id,
                "document": record.document,
                "metadata": record.metadata,
            })
        })
        .chain(
            fixture_edges
                .iter()
                .filter(|record| !baseline_edge_ids.contains(&record.id))
                .map(|record| {
                    json!({
                        "table": "gke_edges",
                        "id": record.id,
                        "document": record.document,
                        "metadata": record.metadata,
                    })
                }),
        )
        .collect::<Vec<_>>();
    assert!(!canonical_graph_rows.is_empty());
    assert!(canonical_graph_rows.iter().any(|row| {
        row["metadata"]["entity_type"] == "acl_record"
            && row["metadata"]["owner_id"] == "causal-owner@example.test"
    }));
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
        expected_digests["digests"]["postgresql"]["sha256"]
            .as_str()
            .unwrap()
    );

    drop(service);
    let (cleanup, connection) = tokio_postgres::connect(&dsn, tokio_postgres::NoTls)
        .await
        .unwrap();
    let cleanup_task = tokio::spawn(connection);
    cleanup
        .batch_execute(&format!(
            "DROP SCHEMA IF EXISTS \"{schema}\" CASCADE; \
             DROP SCHEMA IF EXISTS \"{schema}_conversation\" CASCADE; \
             DROP SCHEMA IF EXISTS \"{schema}_workflow\" CASCADE; \
             DROP SCHEMA IF EXISTS \"{schema}_wisdom\" CASCADE"
        ))
        .await
        .unwrap();
    drop(cleanup);
    cleanup_task.await.unwrap().unwrap();
    let _ = std::fs::remove_file(auth_path);
}
