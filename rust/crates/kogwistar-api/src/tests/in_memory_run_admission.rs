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
        "expected_run_owner_id": "owner",
        "initial_state": {
            "seed": 7,
            "_rt_run_owner_id": "forged-owner",
            "_rt_graph_scope": {"storage_namespace": "forged"}
        },
        "node_ops": {"start": "noop"},
        "node_executor_targets": {"start": "rust-native"}
    });
    let admission_contract: Value = serde_json::from_str(include_str!(
        "../../../../../contracts/golden/workflow-run-submit-auth-v1.json"
    ))
    .unwrap();
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

    let mut wrong_expected_owner = body.clone();
    wrong_expected_owner["run_id"] = json!("memory-wrong-expected-owner");
    wrong_expected_owner["expected_run_owner_id"] = json!("attacker");
    let owner_mismatch = service
        .execute(request(
            json!({
                "sub":"owner",
                "role":"rw",
                "ns":["workflow"],
                "capabilities":["spawn_process", "workflow.run.write", "workflow.run.read"]
            }),
            wrong_expected_owner,
        ))
        .await;
    assert_eq!(owner_mismatch.status, StatusCode::FORBIDDEN.as_u16());
    assert!(
        kogwistar_store::ServerRunReadStore::server_run(
            &service.store,
            "memory-wrong-expected-owner"
        )
        .await
        .unwrap()
        .is_none()
    );

    for missing_capability in admission_contract["required_capabilities"]
        .as_array()
        .unwrap()
    {
        let granted = admission_contract["required_capabilities"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|capability| *capability != missing_capability)
            .cloned()
            .collect::<Vec<_>>();
        let denied_run_id = format!("missing-{}", missing_capability.as_str().unwrap());
        let mut denied_body = body.clone();
        denied_body["run_id"] = json!(denied_run_id);
        let response = service
            .execute(request(
                json!({
                    "sub":"owner",
                    "role":"rw",
                    "ns":["workflow"],
                    "capabilities":granted
                }),
                denied_body,
            ))
            .await;
        assert_eq!(
            response.status,
            admission_contract["denied_status"].as_u64().unwrap() as u16
        );
        assert!(
            kogwistar_store::ServerRunReadStore::server_run(&service.store, &denied_run_id)
                .await
                .unwrap()
                .is_none()
        );
    }

    let principal = json!({
        "sub":"owner",
        "role":"rw",
        "ns":["workflow"],
        "capabilities":["spawn_process", "workflow.run.write", "workflow.run.read"]
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
        "capabilities":["spawn_process", "workflow.run.write"]
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

    let mut authoritative_request = submit(workflow_id, "memory-authoritative-run", json!({}));
    let mut authoritative_body: Value = serde_json::from_slice(&authoritative_request.body).unwrap();
    authoritative_body["initial_state"]["counter"] = json!(0);
    authoritative_request.body = serde_json::to_vec(&authoritative_body).unwrap();
    let accepted = service.execute(authoritative_request).await;
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
    assert!(service.execute_inline_native_work_once().await.unwrap());

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

    let multistep_workflow = "memory-multistep-workflow";
    let mut multistep_request = submit(
        multistep_workflow,
        "memory-multistep-run",
        json!({"start": "noop", "finish": "noop"}),
    );
    let mut multistep_body: Value = serde_json::from_slice(&multistep_request.body).unwrap();
    multistep_body["start_node_id"] = json!("start");
    multistep_body["runtime_routes"] = json!([{
        "edge_id": "route",
        "source_node_id": "start",
        "target_node_id": "finish",
        "aliases": [],
        "join_mask": 0,
        "predicate": null,
        "multiplicity": "all",
        "is_default": false,
        "priority": 0,
        "source_fanout": false
    }]);
    multistep_body["node_executor_targets"] = json!({
        "start": "rust-native",
        "finish": "rust-native"
    });
    multistep_request.body = serde_json::to_vec(&multistep_body).unwrap();
    let accepted_multistep = service.execute(multistep_request).await;
    assert_eq!(
        accepted_multistep.status,
        StatusCode::ACCEPTED.as_u16()
    );
    assert!(service.execute_inline_native_work_once().await.unwrap());
    let successor_lanes = kogwistar_store::LaneMessageReadStore::projected_lane_messages(
        &service.store,
        kogwistar_store::LaneMessageFilter {
            namespace: Some("workflow".to_owned()),
            inbox_id: Some("workflow-runtime".to_owned()),
            correlation_id: Some("memory-multistep-run".to_owned()),
            ..kogwistar_store::LaneMessageFilter::default()
        },
    )
    .await
    .unwrap();
    assert!(successor_lanes.iter().any(|lane| {
        lane.step_id.as_deref() == Some("finish") && lane.status == "pending"
    }));
    assert!(service.execute_inline_native_work_once().await.unwrap());
    assert!(!service.execute_inline_native_work_once().await.unwrap());
    let completed = kogwistar_store::ServerRunReadStore::server_run(
        &service.store,
        "memory-multistep-run",
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(completed.status, "succeeded");
    let events = kogwistar_store::ServerRunReadStore::server_run_events(
        &service.store,
        "memory-multistep-run",
        0,
        20,
    )
    .await
    .unwrap();
    assert_eq!(
        events
            .iter()
            .filter(|event| event.event_type == "workflow.recorded_transition.v1")
            .count(),
        4,
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
        "capabilities":["spawn_process", "workflow.run.write", "write_graph"]
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
