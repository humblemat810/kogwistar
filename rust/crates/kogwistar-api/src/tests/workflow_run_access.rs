#[tokio::test]
async fn in_memory_run_evidence_route_matches_shared_contract() {
    let service = InMemoryGraphApplicationService::default();
    let expected: Value = serde_json::from_str(include_str!(
        "../../../../../contracts/golden/run-evidence-v1.json"
    ))
    .unwrap();
    assert!(service.supports_http_operation(
        "GET",
        "/api/runs/evidence-run/evidence"
    ));
    assert!(service.supports_mcp_tool("workflow", "workflow.run_evidence"));
    let mcp_principal = json!({
        "role":"ro",
        "ns":["workflow"],
        "capabilities":["workflow.run.read","project_view"]
    });
    assert!(service.supports_mcp_tool_for_principal(
        "workflow",
        "workflow.run_evidence",
        &mcp_principal
    ));
    for denied_principal in [
        json!({"role":"ro","ns":["conversation"],"capabilities":["workflow.run.read","project_view"]}),
        json!({"role":"ro","ns":["workflow"],"capabilities":["workflow.run.read"]}),
        json!({"role":"rw","ns":["workflow"],"capabilities":["project_view"]}),
    ] {
        assert!(!service.supports_mcp_tool_for_principal(
            "workflow",
            "workflow.run_evidence",
            &denied_principal
        ));
    }

    kogwistar_store::ServerRunWriteStore::create_server_run(
        &service.store,
        ServerRunCreate {
            run_id: expected["run_id"].as_str().unwrap().to_owned(),
            conversation_id: expected["conversation_id"].as_str().unwrap().to_owned(),
            workflow_id: "evidence-workflow".to_owned(),
            user_id: Some("alice".to_owned()),
            user_turn_node_id: "evidence-turn".to_owned(),
            status: "succeeded".to_owned(),
        },
    )
        .await
        .unwrap();

    let scope = GraphScope {
        namespace: "conversation".to_owned(),
        workspace_id: Some("workspace-a".to_owned()),
        graph_space: Some("conversation".to_owned()),
    };
    let add_node = |id: &str, metadata: Value, document: Value| GraphMutation {
        scope: scope.clone(),
        table: "gke_nodes".to_owned(),
        entity_kind: "node".to_owned(),
        event_id: format!("event-{id}"),
        op: "ADD".to_owned(),
        payload: json!({"id":id,"metadata":metadata}),
        record: GraphRecord {
            id: id.to_owned(),
            document: Some(document.to_string()),
            metadata: serde_json::from_value(metadata).unwrap(),
            embedding: None,
        },
        embedding_dim: 0,
    };
    let snapshot_id = expected["snapshot_node_id"].as_str().unwrap();
    let snapshot_metadata = expected["snapshot_metadata"].clone();
    let digest = expected["source_summary"]["evidence_pack_digest"].to_string();
    let acl_id = format!("acl-{snapshot_id}");
    let acl_metadata = json!({
        "entity_type":"acl_record",
        "acl_truth_graph":"conversation",
        "acl_target_grain":"node",
        "acl_target_entity_id":snapshot_id,
        "acl_version":1,
        "acl_mode":"private",
        "owner_id":"alice",
        "workspace_id":"workspace-a",
        "graph_space":"conversation"
    });
    GraphMutationStore::apply_graph_mutations_atomic(
        &service.store,
        vec![
            add_node(
                snapshot_id,
                snapshot_metadata.clone(),
                json!({
                    "metadata":snapshot_metadata,
                    "properties":{"evidence_pack_digest":digest}
                }),
            ),
            add_node(&acl_id, acl_metadata.clone(), json!({"metadata":acl_metadata})),
        ],
    )
    .await
    .unwrap();

    let response = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "GET".to_owned(),
            path_and_query: format!("/api/runs/{}/evidence", expected["run_id"].as_str().unwrap()),
            body: Vec::new(),
            principal: json!({
                "sub":"alice",
                "user_id":"alice",
                "role":"ro",
                "storage_ns":"conversation",
                "workspace_id":"workspace-a",
                "graph_space":"conversation",
                "ns":["workflow"],
                "capabilities":["workflow.run.read","project_view"]
            }),
        })
        .await;
    assert_eq!(response.status, StatusCode::OK.as_u16());
    assert_eq!(serde_json::from_slice::<Value>(&response.body).unwrap(), expected);

    for (namespace, capabilities, expected_code) in [
        ("conversation", json!(["workflow.run.read", "project_view"]), "KOGWISTAR_NAMESPACE_FORBIDDEN"),
        ("workflow", json!(["workflow.run.read"]), "KOGWISTAR_CAPABILITY_FORBIDDEN"),
    ] {
        let denied = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: format!(
                    "/api/runs/{}/evidence",
                    expected["run_id"].as_str().unwrap()
                ),
                body: Vec::new(),
                principal: json!({
                    "sub":"alice",
                    "user_id":"alice",
                    "role":"ro",
                    "storage_ns":"conversation",
                    "workspace_id":"workspace-a",
                    "graph_space":"conversation",
                    "ns":[namespace],
                    "capabilities":capabilities
                }),
            })
            .await;
        assert_eq!(denied.status, StatusCode::FORBIDDEN.as_u16());
        let denied_body: Value = serde_json::from_slice(&denied.body).unwrap();
        assert_eq!(denied_body["code"], expected_code);
        assert!(!denied_body.to_string().contains(expected["run_id"].as_str().unwrap()));
    }

    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
    use std::sync::Arc;
    use tower::ServiceExt;

    let secret = b"in-memory-run-evidence-secret";
    let token = encode(
        &Header::new(Algorithm::HS256),
        &json!({
            "sub":"alice",
            "user_id":"alice",
            "role":"ro",
            "storage_ns":"conversation",
            "workspace_id":"workspace-a",
            "graph_space":"conversation",
            "ns":["workflow"],
            "capabilities":["workflow.run.read","project_view"],
            "iss":"local",
            "exp":4102444800_u64
        }),
        &EncodingKey::from_secret(secret),
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
                key: Some(String::from_utf8(secret.to_vec()).unwrap()),
                issuer: Some("local".to_owned()),
                ..AuthConfig::default()
            },
        },
        Arc::new(service),
    );
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!(
                    "/api/runs/{}/evidence",
                    expected["run_id"].as_str().unwrap()
                ))
                .header("authorization", format!("Bearer {token}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), 65_536).await.unwrap();
    assert_eq!(serde_json::from_slice::<Value>(&body).unwrap(), expected);

    let mcp_response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp/workflow")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::from(format!(
                    r#"{{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{{"name":"workflow.run_evidence","arguments":{{"run_id":"{}"}}}}}}"#,
                    expected["run_id"].as_str().unwrap()
                )))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(mcp_response.status(), StatusCode::OK);
    let mcp_body = to_bytes(mcp_response.into_body(), 65_536).await.unwrap();
    assert_eq!(
        serde_json::from_slice::<Value>(&mcp_body).unwrap()["result"],
        expected
    );
}

#[tokio::test]
async fn in_memory_run_events_poll_matches_shared_contract() {
    let service = InMemoryGraphApplicationService::default();
    let contract: Value = serde_json::from_str(include_str!(
        "../../../../../contracts/golden/run-events-poll-v1.json"
    ))
    .unwrap();
    kogwistar_store::ServerRunWriteStore::create_server_run(
        &service.store,
        ServerRunCreate {
            run_id: "run-1".to_owned(),
            conversation_id: "conversation-1".to_owned(),
            workflow_id: "workflow-1".to_owned(),
            user_id: Some("user-1".to_owned()),
            user_turn_node_id: "turn-1".to_owned(),
            status: "running".to_owned(),
        },
    )
    .await
    .unwrap();
    kogwistar_store::ServerRunWriteStore::append_server_run_event(
        &service.store,
        "run-1",
        "run.started",
        r#"{"source":"test"}"#.to_owned(),
    )
    .await
    .unwrap();

    let response = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "GET".to_owned(),
            path_and_query: "/api/runs/run-1/events/poll?after_seq=0&limit=10".to_owned(),
            body: Vec::new(),
            principal: json!({
                "sub":"user-1",
                "user_id":"user-1",
                "role":"ro",
                "ns":["conversation"],
                "capabilities":["workflow.run.read"]
            }),
        })
        .await;
    assert_eq!(response.status, contract["success_status"].as_u64().unwrap() as u16);
    let body: Value = serde_json::from_slice(&response.body).unwrap();
    for field in contract["required_fields"].as_array().unwrap() {
        assert!(body.get(field.as_str().unwrap()).is_some());
    }
    assert_eq!(body["run_id"], "run-1");
    assert_eq!(body["events"].as_array().unwrap().len(), 1);
    let event = &body["events"][0];
    for field in contract["required_event_fields"].as_array().unwrap() {
        assert!(event.get(field.as_str().unwrap()).is_some());
    }
    assert_eq!(event["event_type"], "run.started");
    assert_eq!(event["payload"], json!({"source":"test"}));

    let denied = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "GET".to_owned(),
            path_and_query: "/api/runs/run-1/events/poll".to_owned(),
            body: Vec::new(),
            principal: json!({
                "sub":"user-1",
                "user_id":"user-1",
                "role":"ro",
                "ns":["workflow"],
                "capabilities":["workflow.run.read"]
            }),
        })
        .await;
    assert_eq!(denied.status, StatusCode::FORBIDDEN.as_u16());
}

#[tokio::test]
async fn workflow_run_status_denies_wrong_scope_before_storage_access() {
    let service = PostgresRunApplicationService::from_dsn(
        "postgres://unused:unused@127.0.0.1:1/unused",
        "run_status_authorization",
    )
    .unwrap();

    for (principal, expected_code) in [
        (
            json!({"role":"ro", "ns":["workflow"], "capabilities":[], "_kogwistar_internal_transport":"mcp"}),
            "KOGWISTAR_CAPABILITY_FORBIDDEN",
        ),
        (
            json!({"role":"ro", "ns":["workflow"], "capabilities":["project_view"], "_kogwistar_internal_transport":"mcp"}),
            "KOGWISTAR_CAPABILITY_FORBIDDEN",
        ),
        (
            json!({"role":"ro", "ns":["conversation"], "capabilities":["workflow.run.read"], "_kogwistar_internal_transport":"mcp"}),
            "KOGWISTAR_NAMESPACE_FORBIDDEN",
        ),
    ] {
        let response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: "/api/workflow/runs/private-run".to_owned(),
                body: Vec::new(),
                principal,
            })
            .await;
        assert_eq!(response.status, StatusCode::FORBIDDEN.as_u16());
        assert_eq!(
            serde_json::from_slice::<Value>(&response.body).unwrap()["code"],
            expected_code
        );
    }
}

#[test]
fn public_http_principal_cannot_supply_internal_mcp_transport_marker() {
    let principal = without_internal_transport_marker(json!({
        "sub": "reader",
        "_kogwistar_internal_transport": "mcp"
    }));
    assert!(principal.get(INTERNAL_MCP_TRANSPORT_CLAIM).is_none());
}

#[test]
fn only_mcp_workflow_status_request_gets_acl_parity_exception() {
    let principal = mcp_workflow_status_principal(&json!({"user_id":"reader"}));
    assert_eq!(principal[INTERNAL_MCP_TRANSPORT_CLAIM], "mcp");
    let status = ApiEffectRequest {
        contract_version: 1,
        method: "GET".to_owned(),
        path_and_query: "/api/workflow/runs/run-1".to_owned(),
        body: Vec::new(),
        principal: principal.clone(),
    };
    assert!(is_mcp_workflow_status_read(&status));
    let steps = ApiEffectRequest {
        path_and_query: "/api/workflow/runs/run-1/steps".to_owned(),
        ..status.clone()
    };
    assert!(!is_mcp_workflow_status_read(&steps));
    let conversation = ApiEffectRequest {
        path_and_query: "/api/runs/run-1".to_owned(),
        ..status
    };
    assert!(!is_mcp_workflow_status_read(&conversation));
}

#[tokio::test]
async fn workflow_checkpoint_get_denies_wrong_scope_before_storage_access() {
    let service = PostgresRunApplicationService::from_dsn(
        "postgres://unused:unused@127.0.0.1:1/unused",
        "checkpoint_get_authorization",
    )
    .unwrap();

    for path in [
        "/api/workflow/runs/private-run/checkpoints/0",
        "/api/workflow/runs/private-run/replay?target_step_seq=0",
    ] {
        for (principal, expected_code) in [
            (
                json!({"role":"ro", "ns":["workflow"], "capabilities":[]}),
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
            ),
            (
                json!({"role":"ro", "ns":["conversation"], "capabilities":["workflow.run.read"]}),
                "KOGWISTAR_NAMESPACE_FORBIDDEN",
            ),
        ] {
            let response = service
                .execute(ApiEffectRequest {
                    contract_version: 1,
                    method: "GET".to_owned(),
                    path_and_query: path.to_owned(),
                    body: Vec::new(),
                    principal,
                })
                .await;
            assert_eq!(response.status, StatusCode::FORBIDDEN.as_u16());
            assert_eq!(
                serde_json::from_slice::<Value>(&response.body).unwrap()["code"],
                expected_code
            );
        }
    }
}

#[tokio::test]
async fn workflow_run_events_denies_wrong_scope_before_storage_access() {
    let service = PostgresRunApplicationService::from_dsn(
        "postgres://unused:unused@127.0.0.1:1/unused",
        "run_events_authorization",
    )
    .unwrap();

    for (principal, expected_code) in [
        (
            json!({"role":"ro", "ns":["workflow"], "capabilities":[]}),
            "KOGWISTAR_CAPABILITY_FORBIDDEN",
        ),
        (
            json!({"role":"ro", "ns":["conversation"], "capabilities":["workflow.run.read"]}),
            "KOGWISTAR_NAMESPACE_FORBIDDEN",
        ),
    ] {
        let response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query:
                    "/api/workflow/runs/private-run/events/poll?after_seq=0&limit=10".to_owned(),
                body: Vec::new(),
                principal,
            })
            .await;
        assert_eq!(response.status, StatusCode::FORBIDDEN.as_u16());
        assert_eq!(
            serde_json::from_slice::<Value>(&response.body).unwrap()["code"],
            expected_code
        );
    }
}

#[tokio::test]
async fn workflow_run_steps_denies_wrong_scope_before_storage_access() {
    let service = PostgresRunApplicationService::from_dsn(
        "postgres://unused:unused@127.0.0.1:1/unused",
        "run_steps_authorization",
    )
    .unwrap();

    for (principal, expected_code) in [
        (
            json!({"role":"ro", "ns":["workflow"], "capabilities":[]}),
            "KOGWISTAR_CAPABILITY_FORBIDDEN",
        ),
        (
            json!({"role":"ro", "ns":["conversation"], "capabilities":["workflow.run.read"]}),
            "KOGWISTAR_NAMESPACE_FORBIDDEN",
        ),
    ] {
        let response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: "/api/workflow/runs/private-run/steps?limit=25".to_owned(),
                body: Vec::new(),
                principal,
            })
            .await;
        assert_eq!(response.status, StatusCode::FORBIDDEN.as_u16());
        assert_eq!(
            serde_json::from_slice::<Value>(&response.body).unwrap()["code"],
            expected_code
        );
    }
}

#[tokio::test]
async fn sqlite_workflow_run_status_and_steps_enforce_per_workflow_read_acl() {
    let sqlite_path = std::env::temp_dir().join(format!(
        "kogwistar-sqlite-run-steps-acl-{}.sqlite",
        uuid::Uuid::new_v4().simple()
    ));
    let auth_path = std::env::temp_dir().join(format!(
        "kogwistar-sqlite-run-steps-acl-auth-{}.sqlite",
        uuid::Uuid::new_v4().simple()
    ));
    let auth = SqliteAuthStore::open(&auth_path).unwrap();
    auth.resolve_external_identity(ResolveExternalIdentity {
        issuer: "test".to_owned(),
        subject: "steps-acl-user".to_owned(),
        email: "steps-acl-user@example.test".to_owned(),
        display_name: None,
        new_user_id: "steps-acl-user".to_owned(),
        default_role: "rw".to_owned(),
        default_ns: "workflow".to_owned(),
    })
    .await
    .unwrap();
    auth.resolve_external_identity(ResolveExternalIdentity {
        issuer: "test".to_owned(),
        subject: "steps-acl-reader".to_owned(),
        email: "steps-acl-reader@example.test".to_owned(),
        display_name: None,
        new_user_id: "steps-acl-reader".to_owned(),
        default_role: "ro".to_owned(),
        default_ns: "workflow".to_owned(),
    })
    .await
    .unwrap();
    drop(auth);

    let auth_config = AuthConfig {
        auth_db_url: Some(format!("sqlite:///{}", auth_path.display())),
        ..AuthConfig::default()
    };
    let service = SqliteRunApplicationService::open(&sqlite_path)
        .unwrap()
        .with_owner_auth_config(&auth_config)
        .await
        .unwrap();
    service
        .store
        .create_server_run(ServerRunCreate {
            run_id: "private-steps-run".to_owned(),
            conversation_id: "steps-conversation".to_owned(),
            workflow_id: "private-steps-workflow".to_owned(),
            user_id: Some("steps-acl-user".to_owned()),
            user_turn_node_id: "steps-turn".to_owned(),
            status: "completed".to_owned(),
        })
        .unwrap();

    let status = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "GET".to_owned(),
            path_and_query: "/api/workflow/runs/private-steps-run".to_owned(),
            body: Vec::new(),
            principal: json!({
                "user_id": "steps-acl-user",
                "role": "rw",
                "ns": ["workflow"],
                "capabilities": ["workflow.run.read"]
            }),
        })
        .await;

    assert_eq!(status.status, StatusCode::FORBIDDEN.as_u16());
    assert_eq!(
        serde_json::from_slice::<Value>(&status.body).unwrap()["code"],
        "KOGWISTAR_WORKFLOW_FORBIDDEN"
    );

    let mcp_status = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "GET".to_owned(),
            path_and_query: "/api/workflow/runs/private-steps-run".to_owned(),
            body: Vec::new(),
            principal: json!({
                "user_id": "steps-acl-user",
                "role": "rw",
                "ns": ["workflow"],
                "capabilities": ["workflow.run.read"],
                "_kogwistar_internal_transport": "mcp"
            }),
        })
        .await;
    assert_eq!(mcp_status.status, StatusCode::OK.as_u16());
    assert_eq!(
        serde_json::from_slice::<Value>(&mcp_status.body).unwrap()["run_id"],
        "private-steps-run"
    );

    let response = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "GET".to_owned(),
            path_and_query: "/api/workflow/runs/private-steps-run/steps?limit=25".to_owned(),
            body: Vec::new(),
            principal: json!({
                "user_id": "steps-acl-user",
                "role": "rw",
                "ns": ["workflow"],
                "capabilities": ["workflow.run.read"]
            }),
        })
        .await;

    assert_eq!(response.status, StatusCode::FORBIDDEN.as_u16());
    assert_eq!(
        serde_json::from_slice::<Value>(&response.body).unwrap()["code"],
        "KOGWISTAR_WORKFLOW_FORBIDDEN"
    );

    rusqlite::Connection::open(&auth_path)
        .unwrap()
        .execute(
            "INSERT INTO workflow_acl (workflow_id, user_id, role) VALUES ('private-steps-workflow', 'steps-acl-reader', 'ro')",
            [],
        )
        .unwrap();
    let delegated_principal = json!({
        "user_id": "steps-acl-reader",
        "role": "ro",
        "ns": ["workflow"],
        "capabilities": ["workflow.run.read"]
    });
    for (path, expected_field) in [
        ("/api/workflow/runs/private-steps-run", "workflow_id"),
        ("/api/workflow/runs/private-steps-run/steps?limit=25", "steps"),
    ] {
        let delegated = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: path.to_owned(),
                body: Vec::new(),
                principal: delegated_principal.clone(),
            })
            .await;
        assert_eq!(delegated.status, StatusCode::OK.as_u16(), "{path}");
        let payload: Value = serde_json::from_slice(&delegated.body).unwrap();
        assert!(payload.get(expected_field).is_some(), "{path}");
    }
    drop(service);
    let _ = std::fs::remove_file(sqlite_path);
    let _ = std::fs::remove_file(auth_path);
}
