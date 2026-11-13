#[tokio::test]
async fn sqlite_workflow_process_table_is_discovered_with_scope_and_capability() {
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
    use tower::ServiceExt;

    let fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/workflow-blocked-runs-mcp-v1.json"
    )))
    .unwrap();
    let process_table_fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/workflow-process-table-mcp-v1.json"
    )))
    .unwrap();
    let run = &fixture["run"];
    let run_id = run["run_id"].as_str().unwrap();
    let sqlite_path = std::env::temp_dir().join(format!(
        "kogwistar-sqlite-process-table-{}.sqlite",
        uuid::Uuid::new_v4().simple()
    ));
    let service = SqliteRunApplicationService::open(&sqlite_path).unwrap();
    service
        .store
        .create_server_run(ServerRunCreate {
            run_id: "workflow-process-table-completed-run".to_owned(),
            conversation_id: "workflow-process-table-completed-conversation".to_owned(),
            workflow_id: "workflow-process-table-completed-workflow".to_owned(),
            user_id: Some("alice".to_owned()),
            user_turn_node_id: "workflow-process-table-completed-turn".to_owned(),
            status: "succeeded".to_owned(),
        })
        .unwrap();
    service
        .store
        .create_server_run(ServerRunCreate {
            run_id: run_id.to_owned(),
            conversation_id: run["conversation_id"].as_str().unwrap().to_owned(),
            workflow_id: run["workflow_id"].as_str().unwrap().to_owned(),
            user_id: Some(run["user_id"].as_str().unwrap().to_owned()),
            user_turn_node_id: run["user_turn_node_id"].as_str().unwrap().to_owned(),
            status: run["status"].as_str().unwrap().to_owned(),
        })
        .unwrap();

    let authorized = json!({
        "sub":"alice", "role":"ro", "ns":["workflow"],
        "capabilities":["workflow.run.read"]
    });
    assert!(service.supports_mcp_tool("workflow", "workflow.process_table"));
    assert!(service.supports_mcp_tool_for_principal(
        "workflow",
        "workflow.process_table",
        &authorized,
    ));
    assert!(service.supports_mcp_tool("workflow", "workflow.blocked_runs"));
    assert!(service.supports_mcp_tool_for_principal(
        "workflow",
        "workflow.blocked_runs",
        &authorized,
    ));
    assert!(!service.supports_mcp_tool("workflow", "workflow.operator_inbox"));
    for (surface, principal) in [
        ("root", authorized.clone()),
        (
            "workflow",
            json!({"sub":"alice", "role":"rw", "ns":["workflow"], "capabilities":["read_graph"]}),
        ),
        (
            "workflow",
            json!({"sub":"alice", "role":"ro", "ns":["other"], "capabilities":["workflow.run.read"]}),
        ),
        (
            "workflow",
            json!({"sub":"alice", "role":"ro", "ns":["workflow"], "capabilities":["read_graph"]}),
        ),
    ] {
        assert!(
            !service.supports_mcp_tool_for_principal(surface, "workflow.process_table", &principal)
                && !service.supports_mcp_tool_for_principal(
                    surface,
                    "workflow.blocked_runs",
                    &principal
                ),
            "unexpected workflow monitor access for {surface} principal {principal}"
        );
    }
    let project_view_reader = json!({
        "sub":"alice", "role":"ro", "ns":["*"], "capabilities":["project_view"]
    });
    assert!(service.supports_mcp_tool_for_principal(
        "workflow",
        "workflow.process_table",
        &project_view_reader,
    ));
    assert!(service.supports_mcp_tool_for_principal(
        "workflow",
        "workflow.blocked_runs",
        &project_view_reader,
    ));

    let response = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "GET".to_owned(),
            path_and_query: "/api/workflow/process-table?status=succeeded&limit=10".to_owned(),
            body: Vec::new(),
            principal: authorized,
        })
        .await;
    assert_eq!(response.status, StatusCode::OK.as_u16(), "{}", String::from_utf8_lossy(&response.body));
    let body: Value = serde_json::from_slice(&response.body).unwrap();
    assert!(body["processes"].as_array().unwrap().iter().any(|process| {
        process["process_id"] == "workflow-process-table-completed-run"
            && process["process_kind"] == "workflow_run"
            && process["status"] == "succeeded"
    }));
    let blocked = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "GET".to_owned(),
            path_and_query: "/api/workflow/blocked-runs?limit=10".to_owned(),
            body: Vec::new(),
            principal: json!({
                "sub":"alice", "role":"ro", "ns":["workflow"],
                "capabilities":["project_view"]
            }),
        })
        .await;
    assert_eq!(blocked.status, StatusCode::OK.as_u16());
    let blocked: Value = serde_json::from_slice(&blocked.body).unwrap();
    let blocked_processes = blocked["processes"].as_array().unwrap();
    assert_eq!(blocked_processes.len(), 1);
    assert_eq!(blocked_processes[0]["process_id"], run_id);
    assert_eq!(blocked_processes[0]["status"], run["status"]);

    let secret = "workflow-blocked-mcp-test-secret";
    let bearer = encode(
        &Header::new(Algorithm::HS256),
        &json!({
            "sub":"alice", "role":"ro", "ns":["workflow"],
            "capabilities":["workflow.run.read"], "exp":4102444800_u64
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
        Arc::new(service.clone()),
    );
    let arguments = fixture["arguments"].clone();
    let no_capability_bearer = encode(
        &Header::new(Algorithm::HS256),
        &json!({
            "sub":"alice", "role":"ro", "ns":["workflow"],
            "capabilities":[], "exp":4102444800_u64
        }),
        &EncodingKey::from_secret(secret.as_bytes()),
    )
    .unwrap();
    let denied = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp/workflow")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!("Bearer {no_capability_bearer}"),
                )
                .body(Body::from(
                    json!({
                        "jsonrpc":"2.0", "id":"blocked-runs-denied",
                        "method":"tools/call",
                        "params":{"name":fixture["operation"], "arguments":arguments}
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(denied.status(), StatusCode::OK);
    let denied_body = to_bytes(denied.into_body(), 65_536).await.unwrap();
    let denied: Value = serde_json::from_slice(&denied_body).unwrap();
    assert_eq!(denied["error"]["code"], -32602);
    let process_table_denied = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp/workflow")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!("Bearer {no_capability_bearer}"),
                )
                .body(Body::from(
                    json!({
                        "jsonrpc":"2.0", "id":"process-table-denied",
                        "method":"tools/call",
                        "params":{
                            "name":process_table_fixture["operation"],
                            "arguments":process_table_fixture["arguments"]
                        }
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(process_table_denied.status(), StatusCode::OK);
    let denied_body = to_bytes(process_table_denied.into_body(), 65_536)
        .await
        .unwrap();
    let denied: Value = serde_json::from_slice(&denied_body).unwrap();
    assert_eq!(denied["error"]["code"], -32602);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp/workflow")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {bearer}"))
                .header("accept", "application/json, text/event-stream")
                .body(Body::from(
                    json!({
                        "jsonrpc":"2.0", "id":"blocked-runs-contract",
                        "method":"tools/call",
                        "params":{"name":fixture["operation"], "arguments":arguments}
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), 65_536).await.unwrap();
    let wire = String::from_utf8(body.to_vec()).unwrap();
    let data = wire
        .lines()
        .find_map(|line| line.strip_prefix("data: "))
        .unwrap_or_else(|| panic!("missing MCP SSE data frame: {wire}"));
    let response: Value = serde_json::from_str(data).unwrap();
    let actual = response["result"]["structuredContent"]["processes"]
        .as_array()
        .unwrap_or_else(|| panic!("unexpected blocked-runs MCP response: {response:#?}"))
        .iter()
        .filter(|item| item["process_id"] == run_id)
        .map(|item| {
            json!({
                "process_id": item["process_id"],
                "process_kind": item["process_kind"],
                "status": item["status"]
            })
        })
        .collect::<Vec<_>>();
    assert_eq!(json!(actual), fixture["expected_processes"]);
    let process_table_response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp/workflow")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {bearer}"))
                .header("accept", "application/json, text/event-stream")
                .body(Body::from(
                    json!({
                        "jsonrpc":"2.0", "id":"process-table-contract",
                        "method":"tools/call",
                        "params":{
                            "name":process_table_fixture["operation"],
                            "arguments":process_table_fixture["arguments"]
                        }
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(process_table_response.status(), StatusCode::OK);
    let body = to_bytes(process_table_response.into_body(), 65_536)
        .await
        .unwrap();
    let wire = String::from_utf8(body.to_vec()).unwrap();
    let data = wire
        .lines()
        .find_map(|line| line.strip_prefix("data: "))
        .unwrap_or_else(|| panic!("missing process-table MCP SSE frame: {wire}"));
    let response: Value = serde_json::from_str(data).unwrap();
    let actual = response["result"]["structuredContent"]["processes"]
        .as_array()
        .unwrap_or_else(|| panic!("unexpected process-table MCP response: {response:#?}"))
        .iter()
        .map(|item| {
            json!({
                "process_id": item["process_id"],
                "process_kind": item["process_kind"],
                "status": item["status"],
                "workflow_id": item["workflow_id"],
                "conversation_id": item["conversation_id"]
            })
        })
        .collect::<Vec<_>>();
    assert_eq!(json!(actual), process_table_fixture["expected_processes"]);

    drop(service);
    let _ = std::fs::remove_file(sqlite_path);
}

#[test]
fn postgres_workflow_monitor_tools_require_workflow_scope_and_read_capability() {
    let service = PostgresRunApplicationService::from_dsn(
        "postgres://unused:unused@127.0.0.1:1/unused",
        "workflow_monitor_discovery",
    )
    .unwrap();
    for tool in ["workflow.process_table", "workflow.blocked_runs"] {
        for principal in [
            json!({"role":"ro", "ns":["workflow"], "capabilities":["read_graph"]}),
            json!({"role":"rw", "ns":["other"], "capabilities":["workflow.run.read"]}),
        ] {
            assert!(!service.supports_mcp_tool_for_principal("workflow", tool, &principal));
        }
        for capability in ["project_view", "workflow.run.read"] {
            let principal = json!({
                "role":"ro", "ns":["workflow"], "capabilities":[capability]
            });
            assert!(service.supports_mcp_tool_for_principal("workflow", tool, &principal));
        }
        assert!(!service.supports_mcp_tool_for_principal(
            "root",
            tool,
            &json!({"role":"ro", "ns":["workflow"], "capabilities":["project_view"]}),
        ));
    }
    let operator_inbox = "workflow.operator_inbox";
    assert!(!service.supports_mcp_tool_for_principal(
        "workflow",
        operator_inbox,
        &json!({"role":"ro", "ns":["workflow"], "capabilities":["workflow.run.read"]}),
    ));
    assert!(service.supports_mcp_tool_for_principal(
        "workflow",
        operator_inbox,
        &json!({"role":"ro", "ns":["workflow"], "capabilities":["project_view"]}),
    ));
    assert!(!service.supports_mcp_tool_for_principal(
        "root",
        operator_inbox,
        &json!({"role":"ro", "ns":["workflow"], "capabilities":["project_view"]}),
    ));
}

#[tokio::test]
#[ignore = "requires KOGWISTAR_TEST_PG_DSN; uses disposable PostgreSQL schemas"]
async fn postgres_operator_inbox_joins_acl_visible_conversation_nodes_and_filters_security_scope() {
    let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
        .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
    let schema = format!("rust_operator_inbox_{}", uuid::Uuid::new_v4().simple());
    let service = PostgresRunApplicationService::from_dsn(&dsn, &schema).unwrap();
    service.ensure_graph_type_schemas().await.unwrap();
    service
        .conversation_store
        .create_graph_schema(3, kogwistar_store_postgres::GraphTableNames::default())
        .await
        .unwrap();
    let scope = GraphScope {
        namespace: "workflow".to_owned(),
        workspace_id: None,
        graph_space: None,
    };
    for (message_id, target_scope, acl_mode, owner) in [
        ("visible-message", "team-a", "public", None),
        ("hidden-by-scope", "team-b", "public", None),
        ("hidden-by-acl", "team-a", "private", Some("bob")),
    ] {
        let metadata = Map::from_iter([
            ("entity_type".to_owned(), json!("lane_message")),
            ("conversation_id".to_owned(), json!("inbox-conversation")),
            ("security_scope".to_owned(), json!(target_scope)),
            ("shared_scope".to_owned(), json!(false)),
            ("shared_inbox".to_owned(), json!(false)),
            ("visibility".to_owned(), json!("private")),
        ]);
        service
            .conversation_store
            .upsert_graph_projection(PostgresGraphProjectionUpsert {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                record: GraphRecord {
                    id: message_id.to_owned(),
                    document: None,
                    metadata,
                    embedding: None,
                },
                embedding_dim: 3,
            })
            .await
            .unwrap();
        let mut acl_metadata = Map::from_iter([
            ("entity_type".to_owned(), json!("acl_record")),
            ("acl_truth_graph".to_owned(), json!("conversation")),
            ("acl_target_grain".to_owned(), json!("node")),
            ("acl_target_entity_id".to_owned(), json!(message_id)),
            ("acl_version".to_owned(), json!(1)),
            ("acl_mode".to_owned(), json!(acl_mode)),
        ]);
        if let Some(owner) = owner {
            acl_metadata.insert("acl_owner_id".to_owned(), json!(owner));
        }
        service
            .conversation_store
            .upsert_graph_projection(PostgresGraphProjectionUpsert {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                record: GraphRecord {
                    id: format!("acl:{message_id}"),
                    document: None,
                    metadata: acl_metadata,
                    embedding: None,
                },
                embedding_dim: 3,
            })
            .await
            .unwrap();
        service
            .store
            .project_lane_message(NewProjectedLaneMessage {
                message_id: message_id.to_owned(),
                namespace: "workflow".to_owned(),
                purpose: "operator".to_owned(),
                inbox_id: "ops".to_owned(),
                conversation_id: "inbox-conversation".to_owned(),
                recipient_id: "operator".to_owned(),
                sender_id: "worker".to_owned(),
                msg_type: "maintenance.result".to_owned(),
                status: "pending".to_owned(),
                created_at: 1,
                available_at: 1,
                run_id: None,
                step_id: None,
                correlation_id: None,
                payload_json: Some("not-returned".to_owned()),
                error_json: None,
            })
            .await
            .unwrap();
    }
    let principal = json!({
        "sub":"alice", "role":"ro", "ns":["workflow"],
        "storage_ns":"workflow", "security_scope":"team-a",
        "capabilities":["project_view","workflow.run.read","read_security_scope"]
    });
    let response = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "GET".to_owned(),
            path_and_query: "/api/workflow/operator/inbox?inbox_id=ops&status=pending&limit=10"
                .to_owned(),
            body: Vec::new(),
            principal: principal.clone(),
        })
        .await;
    assert_eq!(
        response.status,
        StatusCode::OK.as_u16(),
        "{}",
        String::from_utf8_lossy(&response.body)
    );
    let result: Value = serde_json::from_slice(&response.body).unwrap();
    let messages = result["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0]["message_id"], "visible-message");
    assert_eq!(messages[0]["message_security_scope"], "team-a");
    assert!(!response.body.windows(12).any(|window| window == b"not-returned"));

    let dashboard = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "GET".to_owned(),
            path_and_query: "/api/workflow/operator/dashboard?limit=10".to_owned(),
            body: Vec::new(),
            principal,
        })
        .await;
    assert_eq!(dashboard.status, StatusCode::OK.as_u16());
    let dashboard: Value = serde_json::from_slice(&dashboard.body).unwrap();
    assert_eq!(dashboard["operator_inbox"].as_array().unwrap().len(), 1);
    assert_eq!(dashboard["operator_inbox"][0]["message_id"], "visible-message");
    assert_eq!(dashboard["capabilities"]["current_subject"], "alice");
    assert!(dashboard["capabilities"]["specs"]
        .as_array()
        .is_some_and(|items| !items.is_empty()));
    assert!(dashboard["capabilities"]["effective_capabilities"]
        .as_array()
        .unwrap()
        .contains(&json!("project_view")));
}

#[tokio::test]
#[ignore = "requires KOGWISTAR_TEST_CHROMA_URL; uses a unique Chroma collection"]
async fn sqlite_chroma_operator_inbox_joins_acl_visible_conversation_nodes() {
    use kogwistar_store::GraphWriteStore;

    let endpoint = std::env::var("KOGWISTAR_TEST_CHROMA_URL")
        .expect("ignored Chroma test requires KOGWISTAR_TEST_CHROMA_URL");
    let suffix = uuid::Uuid::new_v4().simple().to_string();
    let collection = |kind: &str| kogwistar_store_chroma::ChromaStoreConfig {
        endpoint: endpoint.clone(),
        collection: format!("operator-inbox-{kind}-{suffix}"),
        ..Default::default()
    };
    let graph = kogwistar_store_chroma::ChromaGraphStore::connect_entity_collections(
        collection("nodes"),
        collection("edges"),
    )
    .await
    .unwrap();
    let node_metadata = serde_json::from_value(json!({
        "entity_type":"lane_message",
        "conversation_id":"operator-conversation",
        "workspace_id":"workspace-a",
        "graph_space":"conversation",
        "security_scope":"team-a",
        "shared_scope":false,
        "shared_inbox":false,
        "visibility":"private",
        "lifecycle_status":"active"
    }))
    .unwrap();
    graph
        .upsert_graph_record_of_kind(
            "workflow",
            kogwistar_store::GraphEntityKind::Node,
            GraphRecord {
                id: "sqlite-visible-message".to_owned(),
                document: Some("authorized operator message".to_owned()),
                metadata: node_metadata,
                embedding: Some(vec![0.6, 0.8]),
            },
        )
        .await
        .unwrap();
    let acl_metadata = serde_json::from_value(json!({
        "entity_type":"acl_record",
        "workspace_id":"workspace-a",
        "graph_space":"conversation",
        "acl_truth_graph":"conversation",
        "acl_target_entity_id":"sqlite-visible-message",
        "acl_target_grain":"node",
        "acl_version":1,
        "acl_mode":"public"
    }))
    .unwrap();
    graph
        .upsert_graph_record_of_kind(
            "workflow",
            kogwistar_store::GraphEntityKind::Node,
            GraphRecord {
                id: "sqlite-visible-message-acl".to_owned(),
                document: None,
                metadata: acl_metadata,
                embedding: Some(vec![0.6, 0.8]),
            },
        )
        .await
        .unwrap();

    let sqlite_path = std::env::temp_dir().join(format!(
        "kogwistar-sqlite-chroma-operator-inbox-{}.sqlite",
        suffix
    ));
    let service = SqliteRunApplicationService::open(&sqlite_path)
        .unwrap()
        .with_chroma_graph(graph);
    service
        .store
        .project_lane_message(NewProjectedLaneMessage {
            message_id: "sqlite-visible-message".to_owned(),
            namespace: "workflow".to_owned(),
            purpose: "operator".to_owned(),
            inbox_id: "ops".to_owned(),
            conversation_id: "operator-conversation".to_owned(),
            recipient_id: "operator".to_owned(),
            sender_id: "worker".to_owned(),
            msg_type: "maintenance.result".to_owned(),
            status: "pending".to_owned(),
            created_at: 1,
            available_at: 1,
            run_id: None,
            step_id: None,
            correlation_id: None,
            payload_json: Some("not-returned".to_owned()),
            error_json: None,
        })
        .unwrap();
    let principal = json!({
        "sub":"alice", "role":"ro", "ns":["workflow"],
        "storage_ns":"workflow", "workspace_id":"workspace-a",
        "graph_space":"conversation", "security_scope":"team-a",
        "capabilities":["project_view"]
    });
    let response = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "GET".to_owned(),
            path_and_query: "/api/workflow/operator/inbox?inbox_id=ops&status=pending&limit=10"
                .to_owned(),
            body: Vec::new(),
            principal: principal.clone(),
        })
        .await;
    assert_eq!(
        response.status,
        StatusCode::OK.as_u16(),
        "operator inbox response: {}",
        String::from_utf8_lossy(&response.body)
    );
    let result: Value = serde_json::from_slice(&response.body).unwrap();
    let messages = result["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0]["message_id"], "sqlite-visible-message");
    assert_eq!(messages[0]["message_security_scope"], "team-a");
    assert!(!response.body.windows(12).any(|window| window == b"not-returned"));

    let dashboard = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "GET".to_owned(),
            path_and_query: "/api/workflow/operator/dashboard?limit=10".to_owned(),
            body: Vec::new(),
            principal,
        })
        .await;
    assert_eq!(dashboard.status, StatusCode::OK.as_u16());
    let dashboard: Value = serde_json::from_slice(&dashboard.body).unwrap();
    assert_eq!(dashboard["operator_inbox"].as_array().unwrap().len(), 1);
    assert_eq!(
        dashboard["operator_inbox"][0]["message_id"],
        "sqlite-visible-message"
    );
}

#[tokio::test]
async fn sqlite_operator_dashboard_requires_project_view_and_separates_blocked_runs_from_inbox() {
    let dashboard_contract: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/operator-dashboard-v1.json"
    )))
    .unwrap();
    let sqlite_path = std::env::temp_dir().join(format!(
        "kogwistar-sqlite-operator-dashboard-{}.sqlite",
        uuid::Uuid::new_v4().simple()
    ));
    let service = SqliteRunApplicationService::open(&sqlite_path).unwrap();
    for status in dashboard_contract["blocked_statuses"].as_array().unwrap() {
        let status = status.as_str().unwrap();
        service
            .store
            .create_server_run(ServerRunCreate {
                run_id: format!("dashboard-{status}-run"),
                conversation_id: "dashboard-conversation".to_owned(),
                workflow_id: "dashboard-workflow".to_owned(),
                user_id: Some("alice".to_owned()),
                user_turn_node_id: format!("dashboard-{status}-turn"),
                status: status.to_owned(),
            })
            .unwrap();
    }
    for (message_id, inbox_id, status, status_namespace) in [
        ("dashboard-pending-a", "ops", "pending", "tenant-workflow"),
        ("dashboard-pending-b", "ops", "pending", "tenant-workflow"),
        ("dashboard-failed", "retry", "failed", "tenant-workflow"),
        ("dashboard-other-namespace", "other", "failed", "workflow"),
    ] {
        service
            .store
            .project_lane_message(NewProjectedLaneMessage {
                message_id: message_id.to_owned(),
                namespace: status_namespace.to_owned(),
                purpose: "operator".to_owned(),
                inbox_id: inbox_id.to_owned(),
                conversation_id: "dashboard-conversation".to_owned(),
                recipient_id: "operator".to_owned(),
                sender_id: "worker".to_owned(),
                msg_type: "maintenance.result".to_owned(),
                status: status.to_owned(),
                created_at: 1,
                available_at: 1,
                run_id: None,
                step_id: None,
                correlation_id: Some(format!("correlation-{message_id}")),
                payload_json: None,
                error_json: None,
            })
            .unwrap();
    }
    let request = |capabilities: Value| ApiEffectRequest {
        contract_version: 1,
        method: "GET".to_owned(),
        path_and_query: "/api/workflow/operator/dashboard?limit=10".to_owned(),
        body: Vec::new(),
        principal: json!({
            "sub":"alice", "role":"ro", "ns":["workflow"],
            "storage_ns":"tenant-workflow",
            "capabilities":capabilities
        }),
    };
    let denied = service
        .execute(request(json!(["workflow.run.read"])))
        .await;
    assert_eq!(denied.status, StatusCode::FORBIDDEN.as_u16());

    let view_only = service.execute(request(json!(["project_view"]))).await;
    assert_eq!(view_only.status, StatusCode::FORBIDDEN.as_u16());

    let allowed = service
        .execute(request(json!(dashboard_contract["required_capabilities"])))
        .await;
    assert_eq!(allowed.status, StatusCode::OK.as_u16());
    let dashboard: Value = serde_json::from_slice(&allowed.body).unwrap();
    assert_eq!(dashboard["process_table"].as_array().unwrap().len(), 4);
    assert_eq!(dashboard["blocked_runs"].as_array().unwrap().len(), 4);
    let dashboard_fields = dashboard_contract["dashboard_fields"]
        .as_array()
        .unwrap()
        .iter()
        .map(|field| field.as_str().unwrap())
        .collect::<Vec<_>>();
    assert!(dashboard_fields.iter().all(|field| dashboard.get(*field).is_some()));
    let blocked_nodes = dashboard["blocked_flow_graph"]["nodes"].as_array().unwrap();
    assert_eq!(dashboard["blocked_flow_graph"]["blocked_count"], 4);
    assert_eq!(blocked_nodes.len(), 4);
    let expected_node_fields = dashboard_contract["blocked_flow_node_fields"]
        .as_array()
        .unwrap()
        .iter()
        .map(|field| field.as_str().unwrap())
        .collect::<std::collections::BTreeSet<_>>();
    let expected_statuses = dashboard_contract["blocked_statuses"]
        .as_array()
        .unwrap()
        .iter()
        .map(|status| status.as_str().unwrap().to_owned())
        .collect::<std::collections::BTreeSet<_>>();
    let actual_statuses = blocked_nodes
        .iter()
        .map(|node| node["status"].as_str().unwrap().to_owned())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(actual_statuses, expected_statuses);
    for node in blocked_nodes {
        let fields = node
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(fields, expected_node_fields);
    }
    assert_eq!(dashboard["operator_inbox"], json!([]));
    assert_eq!(dashboard["message_queue"]["total"], 3);
    assert_eq!(dashboard["message_queue"]["by_status"]["pending"], 2);
    assert_eq!(dashboard["message_queue"]["by_status"]["failed"], 1);
    assert_eq!(dashboard["message_queue"]["by_inbox"]["ops"], 2);
    assert_eq!(dashboard["message_queue"]["by_inbox"]["retry"], 1);
    assert_eq!(dashboard["message_queue"]["failed"].as_array().unwrap().len(), 1);
    assert_eq!(
        dashboard["message_queue"]["failed"][0]["message_id"],
        "dashboard-failed"
    );
    let capability_snapshot = &dashboard["capabilities"];
    assert!(capability_snapshot["specs"].as_array().is_some_and(|items| !items.is_empty()));
    assert_eq!(capability_snapshot["current_subject"], "alice");
    assert!(capability_snapshot["effective_capabilities"]
        .as_array()
        .unwrap()
        .contains(&json!("project_view")));
    assert!(capability_snapshot["approvals"].is_array());
    assert!(capability_snapshot["revoked"].is_array());
    assert!(capability_snapshot["audit_log"].is_array());
}

#[tokio::test]
async fn sqlite_resources_route_matches_python_contract_and_requires_scope_capabilities() {
    let contract: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/workflow-resources-http-v1.json"
    )))
    .unwrap();
    let sqlite_path = std::env::temp_dir().join(format!(
        "kogwistar-sqlite-resources-contract-{}.sqlite",
        uuid::Uuid::new_v4().simple()
    ));
    let service = SqliteRunApplicationService::open(&sqlite_path).unwrap();
    let request = |capabilities: Value| ApiEffectRequest {
        contract_version: 1,
        method: "GET".to_owned(),
        path_and_query: "/api/workflow/resources".to_owned(),
        body: Vec::new(),
        principal: json!({
            "sub": "alice",
            "role": "ro",
            "ns": ["workflow"],
            "storage_ns": "tenant-workflow",
            "capabilities": capabilities
        }),
    };
    for missing_capabilities in [json!(["project_view"]), json!(["read_security_scope"])] {
        let denied = service.execute(request(missing_capabilities)).await;
        assert_eq!(denied.status, StatusCode::FORBIDDEN.as_u16());
    }

    let allowed = service
        .execute(request(json!(contract["required_capabilities"])))
        .await;
    assert_eq!(allowed.status, StatusCode::OK.as_u16());
    let resources: Value = serde_json::from_slice(&allowed.body).unwrap();
    let required_fields = contract["required_fields"]
        .as_array()
        .unwrap()
        .iter()
        .map(|field| field.as_str().unwrap())
        .collect::<std::collections::BTreeSet<_>>();
    let additive_fields = contract["rust_additive_fields"]
        .as_array()
        .unwrap()
        .iter()
        .map(|field| field.as_str().unwrap())
        .collect::<std::collections::BTreeSet<_>>();
    let actual_fields = resources
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(actual_fields, required_fields.union(&additive_fields).copied().collect());
    for (key, fields) in contract["required_object_fields"].as_object().unwrap() {
        let actual = resources[key]
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<std::collections::BTreeSet<_>>();
        let expected = fields
            .as_array()
            .unwrap()
            .iter()
            .map(|field| field.as_str().unwrap())
            .collect::<std::collections::BTreeSet<_>>();
        assert!(expected.is_subset(&actual), "resource section {key}");
    }
    assert!(resources["storage_usage_bytes"].is_u64());
}

#[tokio::test]
async fn sqlite_run_transient_delta_routes_preserve_owner_and_sequence_contract() {
    let sqlite_path = std::env::temp_dir().join(format!(
        "kogwistar-transient-delta-route-{}.sqlite",
        uuid::Uuid::new_v4()
    ));
    let service = SqliteRunApplicationService::open(&sqlite_path).unwrap();
    service
        .store
        .create_server_run(ServerRunCreate {
            run_id: "delta-route-run".to_owned(),
            conversation_id: "delta-route-conversation".to_owned(),
            workflow_id: "delta-route-workflow".to_owned(),
            user_id: Some("owner".to_owned()),
            user_turn_node_id: "delta-route-turn".to_owned(),
            status: "running".to_owned(),
        })
        .unwrap();
    service
        .transient_run_deltas
        .publish("delta-route-run", 4, "visible chunk")
        .unwrap();
    let principal = json!({
        "sub": "owner",
        "user_id": "owner",
        "ns": ["workflow"],
        "capabilities": ["workflow.run.read"]
    });

    let legacy = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "GET".to_owned(),
            path_and_query: "/api/runs/delta-route-run/events/poll?after_seq=0".to_owned(),
            body: Vec::new(),
            principal: principal.clone(),
        })
        .await;
    assert_eq!(legacy.status, StatusCode::OK.as_u16());
    assert!(
        serde_json::from_slice::<Value>(&legacy.body)
            .unwrap()
            .get("client_deltas")
            .is_none()
    );

    let denied = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "GET".to_owned(),
            path_and_query:
                "/api/runs/delta-route-run/events/poll?after_seq=0&after_client_sequence=3"
                    .to_owned(),
            body: Vec::new(),
            principal: json!({
                "sub": "not-owner",
                "user_id": "not-owner",
                "ns": ["workflow"],
                "capabilities": ["workflow.run.read"]
            }),
        })
        .await;
    assert_eq!(denied.status, StatusCode::FORBIDDEN.as_u16());

    let poll = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "GET".to_owned(),
            path_and_query:
                "/api/runs/delta-route-run/events/poll?after_seq=0&after_client_sequence=3"
                    .to_owned(),
            body: Vec::new(),
            principal: principal.clone(),
        })
        .await;
    assert_eq!(poll.status, StatusCode::OK.as_u16());
    let poll_body: Value = serde_json::from_slice(&poll.body).unwrap();
    assert_eq!(poll_body["client_deltas"][0]["client_sse_sequence"], 4);
    assert_eq!(poll_body["client_deltas"][0]["delta"], "visible chunk");

    let sse = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "GET".to_owned(),
            path_and_query: "/api/runs/delta-route-run/events?after_client_sequence=3".to_owned(),
            body: Vec::new(),
            principal,
        })
        .await;
    assert_eq!(sse.status, StatusCode::OK.as_u16());
    let sse_body = String::from_utf8(sse.body).unwrap();
    assert!(sse_body.contains("id: delta-4\nevent: response.text.delta"));
    assert!(sse_body.contains("visible chunk"));
    assert!(!sse_body.contains("id: 4\n"));

    drop(service);
    remove_test_sqlite(&sqlite_path);
}
