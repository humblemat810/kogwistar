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
        .clone()
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

// PYTHON_PARITY_NOTE_V1: InMemory lineage matches Python REST and MCP projections, admission, missing-run status, limit handling, and ACL-visible prefix. Python=kogwistar/server/chat_service_run_inspection.py#ChatRunService.workflow_run_lineage+kogwistar/server/runtime_api.py#create_runtime_router.get_workflow_run_lineage+kogwistar/server/chat_mcp.py#build_workflow_mcp.workflow_run_lineage; fixture=contracts/golden/workflow-run-lineage-v1.json; python_test=kogwistar/tests/server/test_workflow_lineage_api.py#test_python_lineage_stops_at_parent_hidden_by_graph_acl+test_lineage_returns_404_for_missing_run+kogwistar/tests/mcp/test_chat_mcp_workflow_run_access.py#test_mcp_workflow_run_lineage_matches_shared_contract+test_mcp_workflow_run_lineage_denies_scope_capability_and_child_acl.
#[tokio::test]
async fn in_memory_workflow_lineage_route_matches_shared_contract() {
    let contract: Value = serde_json::from_str(include_str!(
        "../../../../../contracts/golden/workflow-run-lineage-v1.json"
    ))
    .unwrap();
    let auth_path = std::env::temp_dir().join(format!(
        "kogwistar-memory-lineage-auth-{}.sqlite",
        uuid::Uuid::new_v4().simple()
    ));
    let auth = SqliteAuthStore::open(&auth_path).unwrap();
    auth.resolve_external_identity(ResolveExternalIdentity {
        issuer: "test".to_owned(),
        subject: "lineage-alice".to_owned(),
        email: "lineage-alice@example.test".to_owned(),
        display_name: None,
        new_user_id: "alice".to_owned(),
        default_role: "ro".to_owned(),
        default_ns: "workflow".to_owned(),
    })
    .await
    .unwrap();
    drop(auth);
    let auth_connection = rusqlite::Connection::open(&auth_path).unwrap();
    for workflow_id in ["wf-child", "wf-parent", "wf-visible"] {
        auth_connection
            .execute(
                "INSERT INTO workflow_acl (workflow_id, user_id, role) VALUES (?1, 'alice', 'ro')",
                [workflow_id],
            )
            .unwrap();
    }
    drop(auth_connection);
    let auth_config = AuthConfig {
        auth_db_url: Some(format!("sqlite:///{}", auth_path.display())),
        ..AuthConfig::default()
    };
    let service = InMemoryGraphApplicationService::default()
        .with_owner_auth_config(&auth_config)
        .await
        .unwrap();
    let route = "/api/workflow/runs/lineage-child/lineage";
    assert!(service.supports_http_operation("GET", route));
    let lineage_principal = json!({
        "sub":"alice", "user_id":"alice", "role":"ro",
        "storage_ns":"conversation", "workspace_id":"workspace-a",
        "graph_space":"conversation", "ns":["workflow"],
        "capabilities":["workflow.run.read"]
    });
    assert!(service.supports_mcp_tool_for_principal(
        "workflow",
        "workflow.run_lineage",
        &lineage_principal
    ));
    assert!(!service.supports_mcp_tool_for_principal(
        "workflow",
        "workflow.run_lineage",
        &json!({"role":"ro","ns":["conversation"],"capabilities":["workflow.run.read"]})
    ));

    let scope = GraphScope {
        namespace: "conversation".to_owned(),
        workspace_id: Some("workspace-a".to_owned()),
        graph_space: Some("conversation".to_owned()),
    };
    let mut mutations = Vec::new();
    for item in contract["response"]["lineage"].as_array().unwrap() {
        let run_id = item["run_id"].as_str().unwrap();
        let workflow_id = item["workflow_id"].as_str().unwrap();
        kogwistar_store::ServerRunWriteStore::create_server_run(
            &service.store,
            ServerRunCreate {
                run_id: run_id.to_owned(),
                conversation_id: "lineage-conversation".to_owned(),
                workflow_id: workflow_id.to_owned(),
                user_id: Some("alice".to_owned()),
                user_turn_node_id: "lineage-turn".to_owned(),
                status: "completed".to_owned(),
            },
        )
        .await
        .unwrap();

        let run_node_id = format!("wf_run|{run_id}");
        let mut metadata = json!({
            "entity_type":"workflow_run",
            "workflow_id":workflow_id,
            "parent_run_id":item["parent_run_id"],
            "status":item["status"],
            "wf_invoked":item["wf_invoked"],
            "workspace_id":"workspace-a",
            "graph_space":"conversation"
        });
        mutations.push(GraphMutation {
            scope: scope.clone(),
            table: "gke_nodes".to_owned(),
            entity_kind: "node".to_owned(),
            event_id: format!("event-{run_node_id}"),
            op: "ADD".to_owned(),
            payload: json!({"id":run_node_id,"metadata":metadata}),
            record: GraphRecord {
                id: run_node_id.clone(),
                document: Some("{}".to_owned()),
                metadata: serde_json::from_value(metadata.take()).unwrap(),
                embedding: None,
            },
            embedding_dim: 0,
        });
        let acl_id = format!("acl-{run_node_id}");
        let acl_metadata = json!({
            "entity_type":"acl_record",
            "acl_truth_graph":"conversation",
            "acl_target_grain":"node",
            "acl_target_entity_id":run_node_id,
            "acl_version":1,
            "acl_mode":"public",
            "workspace_id":"workspace-a",
            "graph_space":"conversation"
        });
        mutations.push(GraphMutation {
            scope: scope.clone(),
            table: "gke_nodes".to_owned(),
            entity_kind: "node".to_owned(),
            event_id: format!("event-{acl_id}"),
            op: "ADD".to_owned(),
            payload: json!({"id":acl_id,"metadata":acl_metadata}),
            record: GraphRecord {
                id: acl_id,
                document: Some("{}".to_owned()),
                metadata: serde_json::from_value(acl_metadata).unwrap(),
                embedding: None,
            },
            embedding_dim: 0,
        });
    }
    GraphMutationStore::apply_graph_mutations_atomic(&service.store, mutations)
        .await
        .unwrap();

    let response = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "GET".to_owned(),
            path_and_query: route.to_owned(),
            body: Vec::new(),
            principal: json!({
                "sub":"alice", "user_id":"alice", "role":"ro",
                "storage_ns":"conversation", "workspace_id":"workspace-a",
                "graph_space":"conversation", "ns":["workflow"],
                "capabilities":["workflow.run.read"]
            }),
        })
        .await;
    assert_eq!(response.status, contract["success_status"].as_u64().unwrap() as u16);
    assert_eq!(serde_json::from_slice::<Value>(&response.body).unwrap(), contract["response"]);

    for (principal, expected_code) in [
        (
            json!({"sub":"alice","user_id":"alice","role":"ro","storage_ns":"conversation","workspace_id":"workspace-a","graph_space":"conversation","ns":["conversation"],"capabilities":["workflow.run.read"]}),
            "KOGWISTAR_NAMESPACE_FORBIDDEN",
        ),
        (
            json!({"sub":"alice","user_id":"alice","role":"ro","storage_ns":"conversation","workspace_id":"workspace-a","graph_space":"conversation","ns":["workflow"],"capabilities":[]}),
            "KOGWISTAR_CAPABILITY_FORBIDDEN",
        ),
    ] {
        let denied = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: route.to_owned(),
                body: Vec::new(),
                principal,
            })
            .await;
        assert_eq!(denied.status, StatusCode::FORBIDDEN.as_u16());
        let body: Value = serde_json::from_slice(&denied.body).unwrap();
        assert_eq!(body["code"], expected_code);
    }

    let limited = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "GET".to_owned(),
            path_and_query: format!("{route}?limit=1"),
            body: Vec::new(),
            principal: json!({
                "sub":"alice", "user_id":"alice", "role":"ro",
                "storage_ns":"conversation", "workspace_id":"workspace-a",
                "graph_space":"conversation", "ns":["workflow"],
                "capabilities":["workflow.run.read"]
            }),
        })
        .await;
    assert_eq!(limited.status, StatusCode::OK.as_u16());
    let limited: Value = serde_json::from_slice(&limited.body).unwrap();
    assert_eq!(limited["lineage"].as_array().unwrap().len(), 1);
    assert_eq!(limited["truncated"], true);

    let missing = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "GET".to_owned(),
            path_and_query: "/api/workflow/runs/missing/lineage".to_owned(),
            body: Vec::new(),
            principal: json!({
                "sub":"alice", "user_id":"alice", "role":"ro",
                "storage_ns":"conversation", "workspace_id":"workspace-a",
                "graph_space":"conversation", "ns":["workflow"],
                "capabilities":["workflow.run.read"]
            }),
        })
        .await;
    assert_eq!(missing.status, StatusCode::NOT_FOUND.as_u16());

    kogwistar_store::ServerRunWriteStore::create_server_run(
        &service.store,
        ServerRunCreate {
            run_id: "visible-child".to_owned(),
            conversation_id: "lineage-conversation".to_owned(),
            workflow_id: "wf-visible".to_owned(),
            user_id: Some("alice".to_owned()),
            user_turn_node_id: "lineage-turn".to_owned(),
            status: "completed".to_owned(),
        },
    )
    .await
    .unwrap();
    let add_scoped_node = |id: &str, metadata: Value| GraphMutation {
        scope: scope.clone(),
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
    };
    let visible_child = json!({
        "entity_type":"workflow_run", "workflow_id":"wf-visible",
        "parent_run_id":"hidden-parent", "status":"completed",
        "workspace_id":"workspace-a", "graph_space":"conversation"
    });
    let hidden_parent = json!({
        "entity_type":"workflow_run", "workflow_id":"wf-hidden",
        "parent_run_id":null, "status":"completed",
        "workspace_id":"workspace-a", "graph_space":"conversation"
    });
    GraphMutationStore::apply_graph_mutations_atomic(
        &service.store,
        vec![
            add_scoped_node("wf_run|visible-child", visible_child),
            add_scoped_node(
                "acl-visible-child",
                json!({
                    "entity_type":"acl_record", "acl_truth_graph":"conversation",
                    "acl_target_grain":"node", "acl_target_entity_id":"wf_run|visible-child",
                    "acl_version":1, "acl_mode":"public",
                    "workspace_id":"workspace-a", "graph_space":"conversation"
                }),
            ),
            add_scoped_node("wf_run|hidden-parent", hidden_parent),
            add_scoped_node(
                "acl-hidden-parent",
                json!({
                    "entity_type":"acl_record", "acl_truth_graph":"conversation",
                    "acl_target_grain":"node", "acl_target_entity_id":"wf_run|hidden-parent",
                    "acl_version":1, "acl_mode":"private", "owner_id":"bob",
                    "workspace_id":"workspace-a", "graph_space":"conversation"
                }),
            ),
        ],
    )
    .await
    .unwrap();
    let visible_prefix = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "GET".to_owned(),
            path_and_query: "/api/workflow/runs/visible-child/lineage".to_owned(),
            body: Vec::new(),
            principal: json!({
                "sub":"alice", "user_id":"alice", "role":"ro",
                "storage_ns":"conversation", "workspace_id":"workspace-a",
                "graph_space":"conversation", "ns":["workflow"],
                "capabilities":["workflow.run.read"]
            }),
        })
        .await;
    assert_eq!(visible_prefix.status, StatusCode::OK.as_u16());
    let visible_prefix: Value = serde_json::from_slice(&visible_prefix.body).unwrap();
    assert_eq!(visible_prefix["lineage"].as_array().unwrap().len(), 1);
    assert_eq!(visible_prefix["lineage"][0]["run_id"], "visible-child");

    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
    use std::sync::Arc;
    use tower::ServiceExt;

    let secret = "in-memory-lineage-mcp-secret";
    let token = encode(
        &Header::new(Algorithm::HS256),
        &json!({
            "sub":"alice", "user_id":"alice", "role":"ro",
            "storage_ns":"conversation", "workspace_id":"workspace-a",
            "graph_space":"conversation", "ns":["workflow"],
            "capabilities":["workflow.run.read"], "iss":"local",
            "exp":4102444800_u64
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
                issuer: Some("local".to_owned()),
                ..AuthConfig::default()
            },
        },
        Arc::new(service),
    );
    let mcp_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp/workflow")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::from(
                    r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"workflow.run_lineage","arguments":{"run_id":"lineage-child","limit":1}}}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(mcp_response.status(), StatusCode::OK);
    let mcp_body = to_bytes(mcp_response.into_body(), 65_536).await.unwrap();
    let mcp_payload: Value = serde_json::from_slice(&mcp_body).unwrap();
    assert_eq!(mcp_payload["result"]["run_id"], "lineage-child");
    assert_eq!(mcp_payload["result"]["lineage"].as_array().unwrap().len(), 1);
    assert_eq!(mcp_payload["result"]["truncated"], true);

    let denied_token = encode(
        &Header::new(Algorithm::HS256),
        &json!({
            "sub":"alice", "user_id":"alice", "role":"ro",
            "storage_ns":"conversation", "workspace_id":"workspace-a",
            "graph_space":"conversation", "ns":["workflow"],
            "capabilities":[], "iss":"local", "exp":4102444800_u64
        }),
        &EncodingKey::from_secret(secret.as_bytes()),
    )
    .unwrap();
    let denied_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp/workflow")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {denied_token}"))
                .body(Body::from(
                    r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"workflow.run_lineage","arguments":{"run_id":"lineage-child"}}}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(denied_response.status(), StatusCode::OK);
    let denied_body = to_bytes(denied_response.into_body(), 65_536)
        .await
        .unwrap();
    let denied_payload: Value = serde_json::from_slice(&denied_body).unwrap();
    assert_eq!(
        denied_payload["error"]["data"]["code"],
        "KOGWISTAR_CAPABILITY_FORBIDDEN"
    );

    let unauthorized_token = encode(
        &Header::new(Algorithm::HS256),
        &json!({
            "sub":"charlie", "user_id":"charlie", "role":"ro",
            "storage_ns":"conversation", "workspace_id":"workspace-a",
            "graph_space":"conversation", "ns":["workflow"],
            "capabilities":["workflow.run.read"], "iss":"local",
            "exp":4102444800_u64
        }),
        &EncodingKey::from_secret(secret.as_bytes()),
    )
    .unwrap();
    let unauthorized_response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp/workflow")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {unauthorized_token}"))
                .body(Body::from(
                    r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"workflow.run_lineage","arguments":{"run_id":"lineage-child"}}}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(unauthorized_response.status(), StatusCode::OK);
    let unauthorized_body = to_bytes(unauthorized_response.into_body(), 65_536)
        .await
        .unwrap();
    let unauthorized_payload: Value = serde_json::from_slice(&unauthorized_body).unwrap();
    assert_eq!(
        unauthorized_payload["error"]["data"]["code"],
        "KOGWISTAR_WORKFLOW_FORBIDDEN"
    );
    let _ = std::fs::remove_file(auth_path);
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

    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
    use std::sync::Arc;
    use tower::ServiceExt;

    let secret = "run-events-poll-query-secret";
    let token = encode(
        &Header::new(Algorithm::HS256),
        &json!({
            "sub":"user-1", "user_id":"user-1", "role":"ro",
            "ns":["conversation"], "capabilities":["workflow.run.read"],
            "exp":4102444800_u64
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
    let invalid = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/runs/run-1/events/poll?limit=invalid")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(invalid.status(), StatusCode::UNPROCESSABLE_ENTITY);
    let invalid_body: Value =
        serde_json::from_slice(&to_bytes(invalid.into_body(), 8192).await.unwrap()).unwrap();
    assert_eq!(invalid_body["detail"][0]["type"], "int_parsing");
    assert_eq!(invalid_body["detail"][0]["loc"], json!(["query", "limit"]));
    assert_eq!(invalid_body["detail"][0]["input"], "invalid");

    let invalid_after_seq = app
        .oneshot(
            Request::builder()
                .uri("/api/runs/run-1/events/poll?after_seq=invalid")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(invalid_after_seq.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn in_memory_workflow_run_steps_mcp_matches_shared_contract() {
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
    use tower::ServiceExt;

    let contract: Value = serde_json::from_str(include_str!(
        "../../../../../contracts/golden/workflow-run-steps-mcp-v1.json"
    ))
    .unwrap();
    let run_id = contract["arguments"]["run_id"].as_str().unwrap();
    let service = InMemoryGraphApplicationService::default();
    kogwistar_store::ServerRunWriteStore::create_server_run(
        &service.store,
        ServerRunCreate {
            run_id: run_id.to_owned(),
            conversation_id: "steps-conversation".to_owned(),
            workflow_id: "workflow-1".to_owned(),
            user_id: Some("alice".to_owned()),
            user_turn_node_id: "steps-turn".to_owned(),
            status: "succeeded".to_owned(),
        },
    )
    .await
    .unwrap();
    for (event_type, payload) in [
        (
            "workflow.recorded_transition.v1",
            json!({
                "transition_id":"steps-transition-1",
                "reduced": {
                    "state": {
                        "last_step_seq":0,
                        "workflow_id":"workflow-1",
                        "last_node_id":"entry",
                        "state": {"frontier":{"pending":[]}}
                    },
                    "state_update":contract["result"]["steps"][0]["result"]["state_update"],
                    "result":contract["result"]["steps"][0]["result"],
                    "errors":[],
                    "server_status":"succeeded"
                }
            }),
        ),
        (
            "workflow.trace.v1",
            json!({"trace": {
                "event":"runtime.native.step.completed",
                "step_seq":0,
                "node_id":"entry",
                "operation":"begin",
                "duration_ms":3,
                "trace_id":"trace-1",
                "span_id":"span-step-1",
                "parent_span_id":"span-run-1",
                "attempt":1
            }}),
        ),
    ] {
        kogwistar_store::ServerRunWriteStore::append_server_run_event(
            &service.store,
            run_id,
            event_type,
            payload.to_string(),
        )
        .await
        .unwrap();
    }

    let secret = "in-memory-run-steps-mcp-secret";
    let token = encode(
        &Header::new(Algorithm::HS256),
        &json!({
            "sub":"alice", "user_id":"alice", "role":"ro",
            "ns":["workflow"], "capabilities":["workflow.run.read"],
            "iss":"local", "exp":4102444800_u64
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
                issuer: Some("local".to_owned()),
                ..AuthConfig::default()
            },
        },
        Arc::new(service),
    );
    let request_body = json!({
        "jsonrpc":"2.0", "id":1, "method":"tools/call",
        "params":{"name":contract["tool"],"arguments":contract["arguments"]}
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp/workflow")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::from(request_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = serde_json::from_slice(
        &to_bytes(response.into_body(), 65_536).await.unwrap(),
    )
    .unwrap();
    assert_eq!(body["result"], contract["result"]);
}

#[tokio::test]
async fn sqlite_workflow_run_steps_mcp_matches_shared_contract() {
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
    use std::time::{SystemTime, UNIX_EPOCH};
    use tower::ServiceExt;

    let contract: Value = serde_json::from_str(include_str!(
        "../../../../../contracts/golden/workflow-run-steps-mcp-v1.json"
    ))
    .unwrap();
    let run_id = contract["arguments"]["run_id"].as_str().unwrap();
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("kogwistar-run-steps-{nonce}.sqlite3"));
    let service = SqliteRunApplicationService::open(&path).unwrap();
    service
        .store
        .create_server_run(ServerRunCreate {
            run_id: run_id.to_owned(),
            conversation_id: "steps-conversation".to_owned(),
            workflow_id: "workflow-1".to_owned(),
            user_id: Some("alice".to_owned()),
            user_turn_node_id: "steps-turn".to_owned(),
            status: "succeeded".to_owned(),
        })
        .unwrap();
    for (event_type, payload) in [
        (
            "workflow.recorded_transition.v1",
            json!({
                "transition_id":"steps-transition-1",
                "reduced": {
                    "state": {
                        "last_step_seq":0,
                        "workflow_id":"workflow-1",
                        "last_node_id":"entry",
                        "state": {"frontier":{"pending":[]}}
                    },
                    "state_update":contract["result"]["steps"][0]["result"]["state_update"],
                    "result":contract["result"]["steps"][0]["result"],
                    "errors":[],
                    "server_status":"succeeded"
                }
            }),
        ),
        (
            "workflow.trace.v1",
            json!({"trace": {
                "event":"runtime.native.step.completed",
                "step_seq":0,
                "node_id":"entry",
                "operation":"begin",
                "duration_ms":3,
                "trace_id":"trace-1",
                "span_id":"span-step-1",
                "parent_span_id":"span-run-1",
                "attempt":1
            }}),
        ),
    ] {
        service
            .store
            .append_server_run_event(
                run_id,
                event_type,
                payload.to_string(),
            )
            .unwrap();
    }

    let secret = "sqlite-run-steps-mcp-secret";
    let token = encode(
        &Header::new(Algorithm::HS256),
        &json!({
            "sub":"alice", "user_id":"alice", "role":"ro",
            "ns":["workflow"], "capabilities":["workflow.run.read"],
            "iss":"local", "exp":4102444800_u64
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
                issuer: Some("local".to_owned()),
                ..AuthConfig::default()
            },
        },
        Arc::new(service.clone()),
    );
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp/workflow")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::from(
                    json!({
                        "jsonrpc":"2.0", "id":1, "method":"tools/call",
                        "params":{"name":contract["tool"],"arguments":contract["arguments"]}
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = serde_json::from_slice(
        &to_bytes(response.into_body(), 65_536).await.unwrap(),
    )
    .unwrap();
    assert_eq!(body["result"], contract["result"]);
    drop(service);
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(format!("{}-wal", path.display()));
    let _ = std::fs::remove_file(format!("{}-shm", path.display()));
}

#[tokio::test]
async fn in_memory_run_steps_route_matches_shared_python_projection() {
    let contract: Value = serde_json::from_str(include_str!(
        "../../../../../contracts/golden/workflow-step-inspection-v1.json"
    ))
    .unwrap();
    let seed = &contract["parity_seed"];
    let run_id = seed["run_id"].as_str().unwrap();
    let workflow_id = seed["workflow_id"].as_str().unwrap();
    let operation = seed["op"].as_str().unwrap();
    let workflow_node_id = format!("wf:{workflow_id}:{operation}");
    let service = InMemoryGraphApplicationService::default();
    kogwistar_store::ServerRunWriteStore::create_server_run(
        &service.store,
        ServerRunCreate {
            run_id: run_id.to_owned(),
            conversation_id: "conversation-1".to_owned(),
            workflow_id: workflow_id.to_owned(),
            user_id: Some("user-1".to_owned()),
            user_turn_node_id: "turn-1".to_owned(),
            status: "running".to_owned(),
        },
    )
    .await
    .unwrap();
    for (event_type, payload) in [
        (
            "workflow.recorded_transition.v1",
            json!({
                "transition_id":"transition-1",
                "reduced": {
                    "state": {
                        "last_step_seq":seed["step_seq"],
                        "workflow_id":workflow_id,
                        "last_node_id":workflow_node_id,
                        "state": {
                            "_rt_node_ops": {workflow_node_id.clone():operation},
                            "frontier":{"pending":[]}
                        }
                    },
                    "state_update":seed["state_update"],
                    "update":null,
                    "result":seed["result_fields"],
                    "errors":[],
                    "server_status":"running"
                }
            }),
        ),
        (
            "workflow.trace.v1",
            json!({
                "trace": {
                    "event":"runtime.native.step.completed",
                    "step_seq":seed["step_seq"],
                    "node_id":workflow_node_id,
                    "operation":operation,
                    "duration_ms":1,
                    "trace_id":seed["trace_context"]["trace_id"],
                    "span_id":seed["trace_context"]["span_id"],
                    "parent_span_id":seed["trace_context"]["parent_span_id"],
                    "attempt":seed["trace_context"]["attempt"]
                }
            }),
        ),
    ] {
        kogwistar_store::ServerRunWriteStore::append_server_run_event(
            &service.store,
            run_id,
            event_type,
            payload.to_string(),
        )
        .await
        .unwrap();
    }
    let response = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "GET".to_owned(),
            path_and_query: format!("/api/runs/{run_id}/steps"),
            body: Vec::new(),
            principal: json!({
                "sub":"user-1", "user_id":"user-1", "role":"ro",
                "ns":["workflow"], "capabilities":["workflow.run.read"]
            }),
        })
        .await;
    assert_eq!(response.status as u64, contract["success_status"]);
    let body: Value = serde_json::from_slice(&response.body).unwrap();
    assert_eq!(body["run_id"], run_id);
    assert_eq!(body["steps"].as_array().unwrap().len(), 1);
    let expected = contract["parity_step"].as_object().unwrap();
    for (field, value) in expected {
        assert_eq!(&body["steps"][0][field], value, "{field}");
    }
}

#[tokio::test]
async fn in_memory_checkpoint_and_replay_routes_match_python_contracts() {
    let service = InMemoryGraphApplicationService::default();
    let run_id = "checkpoint-run";
    kogwistar_store::ServerRunWriteStore::create_server_run(
        &service.store,
        ServerRunCreate {
            run_id: run_id.to_owned(),
            conversation_id: "checkpoint-conversation".to_owned(),
            workflow_id: "checkpoint-workflow".to_owned(),
            user_id: Some("alice".to_owned()),
            user_turn_node_id: "checkpoint-turn".to_owned(),
            status: "running".to_owned(),
        },
    )
    .await
    .unwrap();
    for (step_seq, state) in [(0, json!({"cursor":"start"})), (3, json!({"cursor":"later"}))] {
        kogwistar_store::ServerRunWriteStore::append_server_run_event(
            &service.store,
            run_id,
            "workflow.recorded_transition.v1",
            json!({
                "transition_id": format!("checkpoint-transition-{step_seq}"),
                "reduced": {
                    "state": {
                        "last_step_seq": step_seq,
                        "workflow_id": "checkpoint-workflow",
                        "last_node_id": "prepare",
                        "state": state
                    },
                    "state_update": [],
                    "result": null,
                    "errors": [],
                    "server_status": "running"
                }
            })
            .to_string(),
        )
        .await
        .unwrap();
    }
    let principal = json!({
        "sub":"alice", "user_id":"alice", "role":"ro",
        "ns":["workflow"], "capabilities":["workflow.run.read"]
    });
    for (suffix, fixture_name) in [
        ("checkpoints", "run-checkpoints-v1.json"),
        ("checkpoints/3", "run-checkpoint-detail-v1.json"),
        ("replay?target_step_seq=3", "run-replay-v1.json"),
    ] {
        let path = format!("/api/runs/{run_id}/{suffix}");
        assert!(service.supports_http_operation("GET", &path), "{path}");
        let fixture: Value = serde_json::from_str(
            &std::fs::read_to_string(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../../contracts/golden")
                    .join(fixture_name),
            )
            .unwrap(),
        )
        .unwrap();
        let response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: path.clone(),
                body: Vec::new(),
                principal: principal.clone(),
            })
            .await;
        assert_eq!(response.status as u64, fixture["success_status"], "{path}");
        let body: Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(body["run_id"], run_id);
        if suffix == "checkpoints" {
            assert_eq!(body["checkpoints"].as_array().unwrap().len(), 2);
            assert_eq!(body["checkpoints"][0]["step_seq"], 0);
            assert_eq!(body["checkpoints"][1]["step_seq"], 3);
            assert_eq!(body["checkpoints"][1]["state"], json!({"cursor":"later"}));
        } else {
            assert_eq!(body["state"], json!({"cursor":"later"}));
        }
        let denied = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: path,
                body: Vec::new(),
                principal: json!({
                    "sub":"alice", "user_id":"alice", "role":"ro",
                    "ns":["conversation"], "capabilities":["workflow.run.read"]
                }),
            })
            .await;
        assert_eq!(denied.status, StatusCode::FORBIDDEN.as_u16());
    }
    for (path, expected_status) in [
        ("/api/runs/checkpoint-run/checkpoints/99", StatusCode::NOT_FOUND),
        ("/api/runs/checkpoint-run/checkpoints/not-a-number", StatusCode::UNPROCESSABLE_ENTITY),
        ("/api/runs/checkpoint-run/replay?target_step_seq=99", StatusCode::NOT_FOUND),
        ("/api/runs/checkpoint-run/replay?target_step_seq=bad", StatusCode::UNPROCESSABLE_ENTITY),
    ] {
        let response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: path.to_owned(),
                body: Vec::new(),
                principal: principal.clone(),
            })
            .await;
        assert_eq!(response.status, expected_status.as_u16(), "{path}");
    }
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

// PYTHON_PARITY_NOTE_V1: Exercise REST namespace/capability denial before run storage for the supported application backends. Python route-policy oracle is the workflow/run inspection and runtime API suite; this test complements operation-specific success/ACL tests listed in ADR-022.
#[tokio::test]
async fn workflow_run_rest_authority_matrix_matches_python() {
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
    use std::sync::Arc;
    use tower::ServiceExt;

    let sqlite_path = std::env::temp_dir().join(format!(
        "kogwistar-rest-authority-matrix-{}.sqlite",
        uuid::Uuid::new_v4().simple()
    ));
    let sqlite = SqliteRunApplicationService::open(&sqlite_path).unwrap();
    let memory = InMemoryGraphApplicationService::default();
    let postgres = PostgresRunApplicationService::from_dsn(
        "postgres://unused:unused@127.0.0.1:1/unused",
        "rest_authority_matrix",
    )
    .unwrap();
    let services: [(&str, Arc<dyn ApplicationService>); 3] = [
        ("in_memory", Arc::new(memory)),
        ("sqlite_metadata", Arc::new(sqlite.clone())),
        ("postgres", Arc::new(postgres)),
    ];
    let routes = [
        ("submit", "POST", "/api/workflow/runs", json!({
            "workflow_id":"matrix-workflow",
            "conversation_id":"matrix-conversation",
            "initial_state":{}
        }), "workflow.run.write"),
        ("status", "GET", "/api/workflow/runs/matrix-run", json!({}), "workflow.run.read"),
        ("steps", "GET", "/api/workflow/runs/matrix-run/steps", json!({}), "workflow.run.read"),
        ("evidence", "GET", "/api/runs/matrix-run/evidence", json!({}), "workflow.run.read"),
        ("lineage", "GET", "/api/workflow/runs/matrix-run/lineage", json!({}), "workflow.run.read"),
        ("resume", "POST", "/api/workflow/runs/matrix-run/resume", json!({}), "workflow.run.write"),
        ("cancel", "POST", "/api/workflow/runs/matrix-run/cancel", json!({}), "workflow.run.write"),
    ];

    let secret = b"workflow-run-rest-authority-matrix";
    for (backend, service) in services {
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
            service,
        );
        for (operation, method, path, body, required_capability) in &routes {
            let mut denied_principals = vec![
                (
                    json!({"sub":"matrix-user", "user_id":"matrix-user", "role":"rw", "ns":["conversation"], "capabilities":[required_capability, "spawn_process", "project_view"]}),
                    "KOGWISTAR_NAMESPACE_FORBIDDEN",
                ),
            ];
            // Python permits an idempotent cancel of an already-terminal run
            // before checking workflow.run.write; active-run denial is covered
            // by the backend-specific cancellation contract tests.
            if *operation != "cancel" {
                denied_principals.push((
                    json!({"sub":"matrix-user", "user_id":"matrix-user", "role":"rw", "ns":["workflow"], "capabilities":["project_view"]}),
                    "KOGWISTAR_CAPABILITY_FORBIDDEN",
                ));
            }
            if matches!(*operation, "submit" | "resume" | "cancel") {
                denied_principals.push((
                    json!({"sub":"matrix-user", "user_id":"matrix-user", "role":"ro", "ns":["workflow"], "capabilities":["workflow.run.write", "spawn_process", "project_view"]}),
                    "PYTHON_ROLE_DETAIL",
                ));
            }
            for (principal, expected_code) in denied_principals {
                let token = encode(
                    &Header::new(Algorithm::HS256),
                    &json!({
                        "sub":principal["sub"],
                        "user_id":principal["user_id"],
                        "role":principal["role"],
                        "ns":principal["ns"],
                        "capabilities":principal["capabilities"],
                        "iss":"local",
                        "exp":4102444800_u64
                    }),
                    &EncodingKey::from_secret(secret),
                )
                .unwrap();
                let response = app
                    .clone()
                    .oneshot(
                        Request::builder()
                            .method(*method)
                            .uri(*path)
                            .header("authorization", format!("Bearer {token}"))
                            .header("content-type", "application/json")
                            .body(if *method == "POST" {
                                Body::from(serde_json::to_vec(body).unwrap())
                            } else {
                                Body::empty()
                            })
                            .unwrap(),
                    )
                    .await
                    .unwrap();
                assert_eq!(
                    response.status().as_u16(),
                    StatusCode::FORBIDDEN.as_u16(),
                    "{backend}/{operation}"
                );
                let response_body: Value = serde_json::from_slice(
                    &to_bytes(response.into_body(), 65_536).await.unwrap(),
                )
                .unwrap();
                if expected_code == "PYTHON_ROLE_DETAIL" {
                    assert_eq!(
                        response_body["detail"],
                        "Forbidden: requires role 'rw', you have 'ro'",
                        "{backend}/{operation}"
                    );
                } else {
                    assert_eq!(response_body["code"], expected_code, "{backend}/{operation}");
                }
                assert!(
                    !response_body.to_string().contains("matrix-run"),
                    "{backend}/{operation} leaked run id"
                );
            }
        }
    }

    drop(sqlite);
    remove_test_sqlite(&sqlite_path);
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
