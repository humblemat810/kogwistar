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
                "capabilities": ["spawn_process", "workflow.run.write", "write_graph"]
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
                "capabilities": ["spawn_process", "workflow.run.write", "write_graph"]
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
    let legacy_steps = recorded_step_snapshots(vec![recorded.clone()]);
    assert!(legacy_steps[0].trace_context.is_none());
    assert!(
        fixture["legacy_step_omits_trace_context"]
            .as_bool()
            .unwrap()
    );

    let completion = ServerRunEvent {
        seq: 6,
        run_id: "run-1".to_owned(),
        event_type: "workflow.recorded_transition.v1".to_owned(),
        payload_json: json!({
            "transition_id": "complete-transition-1",
            "reduced": {
                "kind": "complete",
                "state": {
                    "last_step_seq": 0,
                    "workflow_id": "workflow-1",
                    "last_node_id": "node-1",
                    "state": {}
                },
                "result": {"ok": true},
                "errors": [],
                "server_status": "succeeded"
            }
        })
        .to_string(),
        created_at_ms: 12,
    };
    let steps = recorded_step_snapshots(vec![recorded, completion]);
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0].transition_id, "transition-1");
}

#[tokio::test]
async fn sqlite_legacy_step_without_trace_survives_reopen_without_invented_context() {
    use axum::body::{to_bytes, Body};
    use axum::http::Request;
    use std::time::{SystemTime, UNIX_EPOCH};
    use tower::ServiceExt;

    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("kogwistar-legacy-trace-{nonce}.sqlite3"));
    {
        let store = SqliteStore::open(&path).unwrap();
        store
            .create_server_run(kogwistar_store::ServerRunCreate {
                run_id: "legacy-trace-run".to_owned(),
                conversation_id: "legacy-trace-conversation".to_owned(),
                workflow_id: "legacy-trace-workflow".to_owned(),
                user_id: Some("legacy-trace-owner".to_owned()),
                user_turn_node_id: "legacy-trace-turn".to_owned(),
                status: "running".to_owned(),
            })
            .unwrap();
        store
            .append_server_run_event(
                "legacy-trace-run",
                "workflow.recorded_transition.v1",
                json!({
                    "transition_id": "legacy-transition",
                    "reduced": {
                        "state": {
                            "last_step_seq": 0,
                            "workflow_id": "legacy-trace-workflow",
                            "last_node_id": "legacy-step",
                            "state": {}
                        },
                        "result": {"ok": true},
                        "errors": [],
                        "server_status": "running"
                    }
                })
                .to_string(),
            )
            .unwrap();
    }

    let service = SqliteRunApplicationService::open(&path).unwrap();
    let app = router_with_application(
        ApiState {
            health: snapshot(),
            required_roles: vec!["reader".to_owned()],
            implementation: ImplementationSnapshot::default(),
            auth: AuthConfig::default(),
        },
        Arc::new(service),
    );
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/runs/legacy-trace-run/steps")
                .header("x-kogwistar-roles", "reader")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let payload: Value = serde_json::from_slice(
        &to_bytes(response.into_body(), 16_384).await.unwrap(),
    )
    .unwrap();
    assert_eq!(payload["steps"].as_array().unwrap().len(), 1);
    assert_eq!(payload["steps"][0]["workflow_node_id"], "legacy-step");
    assert!(payload["steps"][0].get("trace_context").is_none());

    let _ = std::fs::remove_file(path);
}

// PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/tests/in_memory_graph_routes.rs#in_memory_backend_routes_authenticated_graph_write_and_mcp_read; python=kogwistar/server_mcp_with_admin.py#api_viz_d3; fixture=contracts/golden/graph-acl-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/in_memory_graph_routes.rs#in_memory_backend_routes_authenticated_graph_write_and_mcp_read; python_test=tests/test_visualization.py#test_visualization_uses_acl_policy_for_selected_python_graph_type
