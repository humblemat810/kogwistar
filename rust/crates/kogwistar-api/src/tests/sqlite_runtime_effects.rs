#[tokio::test]
async fn sqlite_chroma_runtime_result_commits_graph_batch_with_worker_effect() {
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
        chroma_config("runtime-nodes"),
        chroma_config("runtime-edges"),
    )
    .await
    .unwrap();
    chroma_setup.await.unwrap();

    let sqlite_path = std::env::temp_dir().join(format!(
        "kogwistar-sqlite-chroma-runtime-graph-{}.sqlite",
        uuid::Uuid::new_v4()
    ));
    let service = SqliteRunApplicationService::open(&sqlite_path)
        .unwrap()
        .with_chroma_graph(graph);
    let principal = json!({
        "sub": "runtime-owner", "user_id": "runtime-owner", "role": "rw",
        "ns": ["conversation"], "storage_ns": "conversation",
        "workspace": "workspace-a", "graph_space": "conversation",
        "capabilities": ["spawn_process", "workflow.run.write", "write_graph"]
    });
    let submitted = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/workflow/runs".to_owned(),
            body: serde_json::to_vec(&json!({
                "run_id": "sqlite-runtime-graph-run",
                "workflow_id": "sqlite-runtime-graph-workflow",
                "conversation_id": "sqlite-runtime-graph-conversation",
                "turn_node_id": "turn-1"
            }))
            .unwrap(),
            principal: principal.clone(),
        })
        .await;
    assert_eq!(submitted.status, StatusCode::ACCEPTED.as_u16());

    let claimed = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/internal/runtime/claim".to_owned(),
            body: serde_json::to_vec(&json!({
                "claimed_by": "python-worker", "limit": 1, "lease_seconds": 60
            }))
            .unwrap(),
            principal: json!({"sub":"worker","role":"admin","capabilities":["*"]}),
        })
        .await;
    assert_eq!(claimed.status, StatusCode::OK.as_u16());
    let claimed: Value = serde_json::from_slice(&claimed.body).unwrap();
    let work = &claimed["work"][0];

    let models: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/python-graph-model-payloads-v1.json"
    )))
    .unwrap();
    let mut node = models["cases"][0]["payload"].clone();
    node["id"] = json!("runtime-effect-node");
    node["embedding"] = json!([1.0, 0.0, 0.0]);
    node["metadata"]["namespace"] = json!("conversation");
    node["metadata"]["workspace_id"] = json!("workspace-a");
    node["metadata"]["graph_space"] = json!("conversation");
    let mut batch: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/graph-mutation-batch-v1.json"
    )))
    .unwrap();
    let mutation = &mut batch["mutations"][0];
    mutation["scope"] = json!({
        "namespace": "conversation", "workspace_id": "workspace-a",
        "graph_space": "conversation"
    });
    mutation["event_id"] = json!("runtime-effect-graph-event");
    mutation["payload"] = node.clone();
    mutation["model_envelope"] = json!({
        "schema_version": 1, "model_class": "node", "payload": node
    });
    mutation["record"]["id"] = json!("runtime-effect-node");
    mutation["record"]["embedding"] = json!([1.0, 0.0, 0.0]);
    mutation["embedding_dim"] = json!(3);

    let publication: kogwistar_contracts::DurableProviderEventPublicationV1 =
        serde_json::from_value(json!({
            "namespace": "conversation",
            "conversation_id": "sqlite-runtime-graph-conversation",
            "event": {
                "event_id": "sqlite-runtime-provider-event-1",
                "trace": {
                    "run_id": "sqlite-runtime-graph-run",
                    "turn_id": "turn-1",
                    "workflow_node_id": "provider-stream",
                    "step_id": "provider-response-1",
                    "provider_id": "test-provider"
                },
                "event": {
                    "contract_version": "v1",
                    "provider_event_id": "response-1:1",
                    "provider_sequence": 1,
                    "response_id": "response-1",
                    "kind": "field_completed",
                    "field_name": "text",
                    "value": "durable provider output"
                }
            }
        }))
        .unwrap();
    let provider_mutations = provider_event_graph_mutations(
        &publication,
        &GraphScope {
            namespace: "conversation".to_owned(),
            workspace_id: Some("workspace-a".to_owned()),
            graph_space: Some("conversation".to_owned()),
        },
        "workspace-a",
        1,
    );
    batch["mutations"]
        .as_array_mut()
        .unwrap()
        .extend(provider_mutations.into_iter().map(|provider_mutation| {
            json!({
                "scope": {
                    "namespace": provider_mutation.scope.namespace,
                    "workspace_id": provider_mutation.scope.workspace_id,
                    "graph_space": provider_mutation.scope.graph_space
                },
                "table": provider_mutation.table,
                "entity_kind": provider_mutation.entity_kind,
                "event_id": provider_mutation.event_id,
                "op": provider_mutation.op,
                "payload": provider_mutation.payload,
                "record": {
                    "id": provider_mutation.record.id,
                    "document": provider_mutation.record.document,
                    "metadata": provider_mutation.record.metadata,
                    "embedding": provider_mutation.record.embedding
                },
                "embedding_dim": provider_mutation.embedding_dim,
                "model_envelope": null
            })
        }));
    batch["index_jobs"] = json!([{
        "job_id": "runtime-effect-node-index-job",
        "namespace": "conversation",
        "entity_kind": "node",
        "entity_id": "runtime-effect-node",
        "index_kind": "node_embedding",
        "op": "UPSERT",
        "payload_json": null,
        "max_retries": 10
    }]);
    batch["contract_version"] = json!(2);
    batch["default_acl"] = json!({"truth_graph": "conversation"});

    let result_body = serde_json::to_vec(&json!({
        "handoff": {
            "message_id": work["message_id"],
            "claimed_by": work["claimed_by"],
            "run_id": work["run_id"],
            "step_id": work["step_id"],
            "correlation_id": work["correlation_id"]
        },
        "effect": {
            "contract_version": 1,
            "effect_id": "sqlite-runtime-graph-effect",
            "state_update": [["u", {"graph_written": true}]],
            "successors": [],
            "result": {"done": true},
            "graph_mutation_batch_v2": batch,
            "provider_event_publications": [publication]
        }
    }))
    .unwrap();
    let wrong_scope = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/internal/runtime/results".to_owned(),
            body: result_body.clone(),
            principal: json!({
                "sub": "runtime-owner", "user_id": "runtime-owner", "role": "rw",
                "ns": ["conversation"], "storage_ns": "conversation",
                "workspace": "other-workspace", "graph_space": "conversation",
                "capabilities": ["write_graph"]
            }),
        })
        .await;
    assert_eq!(wrong_scope.status, StatusCode::FORBIDDEN.as_u16());
    assert!(
        service
            .store
            .replay_raw_events("conversation", 0, 100)
            .unwrap()
            .is_empty()
    );

    let wrong_owner = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/internal/runtime/results".to_owned(),
            body: result_body.clone(),
            principal: json!({
                "sub": "other-owner", "user_id": "other-owner", "role": "rw",
                "ns": ["conversation"], "storage_ns": "conversation",
                "workspace": "workspace-a", "graph_space": "conversation",
                "capabilities": ["write_graph"]
            }),
        })
        .await;
    assert_eq!(wrong_owner.status, StatusCode::CONFLICT.as_u16());
    assert!(
        service
            .store
            .replay_raw_events("conversation", 0, 100)
            .unwrap()
            .is_empty()
    );

    let mut invalid_handoff: Value = serde_json::from_slice(&result_body).unwrap();
    invalid_handoff["handoff"]["correlation_id"] = json!("wrong-correlation");
    let rolled_back = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/internal/runtime/results".to_owned(),
            body: serde_json::to_vec(&invalid_handoff).unwrap(),
            principal: principal.clone(),
        })
        .await;
    assert_eq!(rolled_back.status, StatusCode::CONFLICT.as_u16());
    assert!(
        service
            .store
            .replay_raw_events("conversation", 0, 100)
            .unwrap()
            .is_empty()
    );
    assert!(
        service
            .store
            .get_projected_lane_message("msg:sqlite-runtime-provider-event-1")
            .unwrap()
            .is_none()
    );

    let mut forged_acl: Value = serde_json::from_slice(&result_body).unwrap();
    forged_acl["effect"]["graph_mutation_batch_v2"]["mutations"][0]["record"]["metadata"]["entity_type"] =
        json!("acl_record");
    let denied_acl = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/internal/runtime/results".to_owned(),
            body: serde_json::to_vec(&forged_acl).unwrap(),
            principal: principal.clone(),
        })
        .await;
    assert_eq!(denied_acl.status, StatusCode::UNPROCESSABLE_ENTITY.as_u16());
    assert!(
        service
            .store
            .replay_raw_events("conversation", 0, 100)
            .unwrap()
            .is_empty()
    );

    let result = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/internal/runtime/results".to_owned(),
            body: result_body.clone(),
            principal: principal.clone(),
        })
        .await;
    assert_eq!(
        result.status,
        StatusCode::OK.as_u16(),
        "{}",
        String::from_utf8_lossy(&result.body)
    );
    let events = service
        .store
        .replay_raw_events("conversation", 0, 100)
        .unwrap();
    assert!(events.iter().any(|event| {
        event.entity_id == "runtime-effect-node" && event.event_id == "runtime-effect-graph-event"
    }));
    assert!(events.iter().any(|event| {
        serde_json::from_str::<Value>(&event.payload_json).is_ok_and(|payload| {
            payload["metadata"]["entity_type"] == "acl_record"
                && payload["metadata"]["acl_target_entity_id"] == "runtime-effect-node"
                && payload["metadata"]["acl_truth_graph"] == "conversation"
        })
    }));
    let provider_lane = service
        .store
        .get_projected_lane_message("msg:sqlite-runtime-provider-event-1")
        .unwrap()
        .unwrap();
    assert_eq!(provider_lane.purpose, "agent_event");
    assert_eq!(provider_lane.status, "pending");
    assert_eq!(
        provider_lane.run_id.as_deref(),
        Some("sqlite-runtime-graph-run")
    );
    assert_eq!(provider_lane.msg_type, "agent.provider_event");
    let receive_request = kogwistar_runtime::RuntimeNativeStepRequest {
        executor_target: kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET.to_owned(),
        operation: "agent.event.receive".to_owned(),
        op_version: "v1".to_owned(),
        run_id: "sqlite-runtime-graph-run".to_owned(),
        workflow_id: "sqlite-runtime-graph-workflow".to_owned(),
        node_id: "receive-step".to_owned(),
        conversation_id: "sqlite-runtime-graph-conversation".to_owned(),
        turn_node_id: Some("turn-1".to_owned()),
        token_id: "sqlite-runtime-graph-run".to_owned(),
        parent_token_id: None,
        step_seq: 1,
        state: Map::new(),
        runtime_routes: Vec::new(),
        ephemeral_context: Map::new(),
    };
    let receive_principal = json!({
        "sub": "runtime-owner", "user_id": "runtime-owner", "role": "rw",
        "ns": ["conversation"], "storage_ns": "conversation",
        "workspace": "workspace-a", "graph_space": "conversation",
        "capabilities": ["read_graph"]
    });
    let wrong_receive_principal = json!({
        "sub": "other-owner", "user_id": "other-owner", "role": "rw",
        "ns": ["conversation"], "storage_ns": "conversation",
        "workspace": "workspace-a", "graph_space": "conversation",
        "capabilities": ["read_graph"]
    });
    assert!(
        service
            .claim_native_provider_event_context(&receive_request, &wrong_receive_principal)
            .await
            .is_err()
    );
    assert_eq!(
        service
            .store
            .get_projected_lane_message("msg:sqlite-runtime-provider-event-1")
            .unwrap()
            .unwrap()
            .status,
        "pending"
    );
    let claimed_event = service
        .claim_native_provider_event_context(&receive_request, &receive_principal)
        .await
        .unwrap()
        .expect("visible event should be claimable by its run owner");
    assert_eq!(
        claimed_event["event"]["event_id"],
        "sqlite-runtime-provider-event-1"
    );
    assert_eq!(
        claimed_event["claim"]["message_id"],
        "msg:sqlite-runtime-provider-event-1"
    );
    assert!(
        service
            .claim_native_provider_event_context(&receive_request, &receive_principal)
            .await
            .unwrap()
            .is_none()
    );
    assert_eq!(
        service
            .store
            .get_projected_lane_message("msg:sqlite-runtime-provider-event-1")
            .unwrap()
            .unwrap()
            .status,
        "claimed"
    );
    let jobs = service
        .store
        .list_index_jobs(
            Some("conversation"),
            Some("PENDING"),
            None,
            Some("runtime-effect-node"),
            None,
            20,
        )
        .unwrap();
    assert!(!jobs.is_empty());
    assert_eq!(
        service
            .store
            .get_server_run("sqlite-runtime-graph-run")
            .unwrap()
            .unwrap()
            .status,
        "succeeded"
    );
    let replay = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/internal/runtime/results".to_owned(),
            body: result_body,
            principal,
        })
        .await;
    assert_eq!(replay.status, StatusCode::OK.as_u16());
    assert_eq!(
        service
            .store
            .replay_raw_events("conversation", 0, 100)
            .unwrap()
            .len(),
        events.len()
    );
    assert_eq!(
        service
            .store
            .list_index_jobs(
                Some("conversation"),
                Some("PENDING"),
                None,
                Some("runtime-effect-node"),
                None,
                20,
            )
            .unwrap()
            .len(),
        jobs.len()
    );

    drop(service);
    remove_test_sqlite(&sqlite_path);
}
