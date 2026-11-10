#[tokio::test]
async fn sqlite_workflow_process_table_is_discovered_with_scope_and_capability() {
    let sqlite_path = std::env::temp_dir().join(format!(
        "kogwistar-sqlite-process-table-{}.sqlite",
        uuid::Uuid::new_v4().simple()
    ));
    let service = SqliteRunApplicationService::open(&sqlite_path).unwrap();
    service
        .store
        .create_server_run(ServerRunCreate {
            run_id: "process-table-run".to_owned(),
            conversation_id: "process-table-conversation".to_owned(),
            workflow_id: "process-table-workflow".to_owned(),
            user_id: Some("alice".to_owned()),
            user_turn_node_id: "process-table-turn".to_owned(),
            status: "succeeded".to_owned(),
        })
        .unwrap();
    service
        .store
        .create_server_run(ServerRunCreate {
            run_id: "process-table-blocked-run".to_owned(),
            conversation_id: "process-table-blocked-conversation".to_owned(),
            workflow_id: "process-table-blocked-workflow".to_owned(),
            user_id: Some("alice".to_owned()),
            user_turn_node_id: "process-table-blocked-turn".to_owned(),
            status: "suspended".to_owned(),
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
    assert_eq!(response.status, StatusCode::OK.as_u16());
    let body: Value = serde_json::from_slice(&response.body).unwrap();
    assert!(body["processes"].as_array().unwrap().iter().any(|process| {
        process["process_id"] == "process-table-run"
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
    assert_eq!(blocked_processes[0]["process_id"], "process-table-blocked-run");
    assert_eq!(blocked_processes[0]["status"], "suspended");

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
    assert_eq!(response.status, StatusCode::OK.as_u16());
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
    assert_eq!(response.status, StatusCode::OK.as_u16());
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

    let allowed = service.execute(request(json!(["project_view"]))).await;
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

#[tokio::test]
async fn workflow_lineage_checks_authority_before_run_lookup() {
    let sqlite_path = std::env::temp_dir().join(format!(
        "kogwistar-lineage-auth-order-{}.sqlite",
        uuid::Uuid::new_v4()
    ));
    let sqlite = SqliteRunApplicationService::open(&sqlite_path).unwrap();
    let postgres = PostgresRunApplicationService::from_dsn(
        "postgres://unused:unused@127.0.0.1:1/unused",
        "lineage_auth_order",
    )
    .unwrap();

    for service in [&sqlite as &dyn ApplicationService, &postgres] {
        let response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: "/api/workflow/runs/missing/lineage".to_owned(),
                body: Vec::new(),
                principal: json!({"ns": ["docs"], "capabilities": []}),
            })
            .await;
        assert_eq!(response.status, StatusCode::FORBIDDEN.as_u16());
        assert_eq!(
            serde_json::from_slice::<Value>(&response.body).unwrap()["code"],
            "KOGWISTAR_NAMESPACE_FORBIDDEN"
        );

        let response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: "/api/workflow/runs/missing/lineage".to_owned(),
                body: Vec::new(),
                principal: json!({"ns": ["workflow"], "capabilities": []}),
            })
            .await;
        assert_eq!(response.status, StatusCode::FORBIDDEN.as_u16());
        assert_eq!(
            serde_json::from_slice::<Value>(&response.body).unwrap()["code"],
            "KOGWISTAR_CAPABILITY_FORBIDDEN"
        );
    }

    drop(sqlite);
    remove_test_sqlite(&sqlite_path);
}

#[tokio::test]
async fn run_evidence_checks_authority_before_run_lookup() {
    let sqlite_path = std::env::temp_dir().join(format!(
        "kogwistar-evidence-auth-order-{}.sqlite",
        uuid::Uuid::new_v4()
    ));
    let sqlite = SqliteRunApplicationService::open(&sqlite_path).unwrap();
    let postgres = PostgresRunApplicationService::from_dsn(
        "postgres://unused:unused@127.0.0.1:1/unused",
        "evidence_auth_order",
    )
    .unwrap();

    for service in [&sqlite as &dyn ApplicationService, &postgres] {
        let response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: "/api/runs/missing/evidence".to_owned(),
                body: Vec::new(),
                principal: json!({"ns": ["docs"], "capabilities": []}),
            })
            .await;
        assert_eq!(response.status, StatusCode::FORBIDDEN.as_u16());
        assert_eq!(
            serde_json::from_slice::<Value>(&response.body).unwrap()["code"],
            "KOGWISTAR_NAMESPACE_FORBIDDEN"
        );

        let response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: "/api/runs/missing/evidence".to_owned(),
                body: Vec::new(),
                principal: json!({
                    "ns": ["workflow"],
                    "capabilities": ["workflow.run.read"]
                }),
            })
            .await;
        assert_eq!(response.status, StatusCode::FORBIDDEN.as_u16());
        assert_eq!(
            serde_json::from_slice::<Value>(&response.body).unwrap()["code"],
            "KOGWISTAR_CAPABILITY_FORBIDDEN"
        );
    }

    let unsupported = sqlite
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "GET".to_owned(),
            path_and_query: "/api/runs/missing/evidence".to_owned(),
            body: Vec::new(),
            principal: json!({
                "ns": ["workflow"],
                "capabilities": ["workflow.run.read", "project_view"]
            }),
        })
        .await;
    assert_eq!(unsupported.status, StatusCode::SERVICE_UNAVAILABLE.as_u16());
    assert_eq!(
        serde_json::from_slice::<Value>(&unsupported.body).unwrap()["code"],
        "KOGWISTAR_GRAPH_AUTHORITY_UNSUPPORTED"
    );

    drop(sqlite);
    remove_test_sqlite(&sqlite_path);
}

#[test]
fn runtime_run_route_decodes_encoded_run_id_without_splitting_it() {
    assert_eq!(
        runtime_run_route("/api/runs/run%2F42%3Fx/evidence"),
        Some(("run/42?x".to_owned(), vec!["evidence".to_owned()]))
    );
    assert_eq!(
        runtime_run_route("/api/workflow/runs/plain-run/steps"),
        Some(("plain-run".to_owned(), vec!["steps".to_owned()]))
    );
    assert_eq!(runtime_run_route("/api/runs/run%FF/evidence"), None);
}

#[test]
fn rust_run_evidence_projection_matches_shared_v1_contract() {
    let expected: Value = serde_json::from_str(include_str!(
        "../../../../../contracts/golden/run-evidence-v1.json"
    ))
    .unwrap();
    let metadata =
        serde_json::from_value::<BTreeMap<String, Value>>(expected["snapshot_metadata"].clone())
            .unwrap();
    let digest =
        serde_json::to_string(&expected["source_summary"]["evidence_pack_digest"]).unwrap();
    let snapshot = ContextSnapshotV1 {
        schema_version: 1,
        snapshot_node_id: expected["snapshot_node_id"].as_str().unwrap().to_owned(),
        conversation_id: expected["conversation_id"].as_str().unwrap().to_owned(),
        metadata,
        properties: BTreeMap::from([("evidence_pack_digest".to_owned(), json!(digest))]),
    };
    assert_eq!(
        public_run_evidence_value(expected["run_id"].as_str().unwrap(), snapshot),
        expected
    );
}

#[tokio::test]
#[ignore = "requires KOGWISTAR_TEST_PG_DSN; uses a disposable PostgreSQL schema"]
async fn postgres_run_evidence_route_returns_only_authorized_run_snapshot() {
    let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
        .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
    let schema = format!("rust_run_evidence_{}", uuid::Uuid::new_v4().simple());
    let service = PostgresRunApplicationService::from_dsn(&dsn, &schema).unwrap();
    service.ensure_graph_type_schemas().await.unwrap();
    service
        .conversation_store
        .create_graph_schema(3, kogwistar_store_postgres::GraphTableNames::default())
        .await
        .unwrap();
    service
        .store
        .create_server_run(ServerRunCreate {
            run_id: "evidence-run".to_owned(),
            conversation_id: "evidence-conversation".to_owned(),
            workflow_id: "evidence-workflow".to_owned(),
            user_id: Some("alice".to_owned()),
            user_turn_node_id: "evidence-turn".to_owned(),
            status: "succeeded".to_owned(),
        })
        .await
        .unwrap();

    let scope = GraphScope {
        namespace: "conversation".to_owned(),
        workspace_id: None,
        graph_space: None,
    };
    let snapshot_id = "snapshot|evidence-run|5";
    let mut snapshot_metadata = Map::new();
    for (key, value) in [
        ("entity_type", json!("context_snapshot")),
        ("conversation_id", json!("evidence-conversation")),
        ("run_id", json!("evidence-run")),
        ("run_step_seq", json!(5)),
        ("lifecycle_status", json!("active")),
    ] {
        snapshot_metadata.insert(key.to_owned(), value);
    }
    service
        .conversation_store
        .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
            scope: scope.clone(),
            table: "gke_nodes".to_owned(),
            record: GraphRecord {
                id: snapshot_id.to_owned(),
                document: Some(
                    json!({
                        "metadata": {
                            "entity_type": "context_snapshot",
                            "conversation_id": "evidence-conversation",
                            "run_id": "evidence-run",
                            "run_step_seq": 5,
                            "used_node_ids": ["evidence-node-1", "evidence-node-2"]
                        },
                        "properties": {
                            "evidence_pack_digest": "{\"items\":2}"
                        }
                    })
                    .to_string(),
                ),
                metadata: snapshot_metadata,
                embedding: None,
            },
            embedding_dim: 3,
        })
        .await
        .unwrap();
    let mut acl_metadata = Map::new();
    for (key, value) in [
        ("entity_type", json!("acl_record")),
        ("acl_truth_graph", json!("conversation")),
        ("acl_target_grain", json!("node")),
        ("acl_target_entity_id", json!(snapshot_id)),
        ("acl_mode", json!("private")),
        ("owner_id", json!("alice")),
        ("acl_version", json!(1)),
        ("tombstoned", json!(false)),
    ] {
        acl_metadata.insert(key.to_owned(), value);
    }
    service
        .conversation_store
        .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
            scope,
            table: "gke_nodes".to_owned(),
            record: GraphRecord {
                id: format!("acl|{snapshot_id}"),
                document: Some("{}".to_owned()),
                metadata: acl_metadata,
                embedding: None,
            },
            embedding_dim: 3,
        })
        .await
        .unwrap();
    let other_snapshot_id = "snapshot|different-run|99";
    let other_metadata = serde_json::from_value::<Map<String, Value>>(json!({
        "entity_type": "context_snapshot",
        "conversation_id": "evidence-conversation",
        "run_id": "different-run",
        "run_step_seq": 99,
        "lifecycle_status": "active"
    }))
    .unwrap();
    service
        .conversation_store
        .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
            scope: GraphScope {
                namespace: "conversation".to_owned(),
                workspace_id: None,
                graph_space: None,
            },
            table: "gke_nodes".to_owned(),
            record: GraphRecord {
                id: other_snapshot_id.to_owned(),
                document: Some(
                    json!({
                        "metadata": {
                            "entity_type": "context_snapshot",
                            "conversation_id": "evidence-conversation",
                            "run_id": "different-run",
                            "run_step_seq": 99,
                            "used_node_ids": ["wrong-run-evidence"]
                        },
                        "properties": {"evidence_pack_digest": "{\"items\":1}"}
                    })
                    .to_string(),
                ),
                metadata: other_metadata,
                embedding: None,
            },
            embedding_dim: 3,
        })
        .await
        .unwrap();
    let other_acl_metadata = serde_json::from_value::<Map<String, Value>>(json!({
        "entity_type": "acl_record",
        "acl_truth_graph": "conversation",
        "acl_target_grain": "node",
        "acl_target_entity_id": other_snapshot_id,
        "acl_mode": "public",
        "acl_version": 1,
        "tombstoned": false
    }))
    .unwrap();
    service
        .conversation_store
        .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
            scope: GraphScope {
                namespace: "conversation".to_owned(),
                workspace_id: None,
                graph_space: None,
            },
            table: "gke_nodes".to_owned(),
            record: GraphRecord {
                id: format!("acl|{other_snapshot_id}"),
                document: Some("{}".to_owned()),
                metadata: other_acl_metadata,
                embedding: None,
            },
            embedding_dim: 3,
        })
        .await
        .unwrap();

    let secret = "run-evidence-route-test-secret";
    let token = encode(
        &Header::new(Algorithm::HS256),
        &json!({
            "user_id": "alice",
            "sub": "alice",
            "roles": ["reader"],
            "ns": ["workflow"],
            "storage_ns": "conversation",
            "capabilities": ["workflow.run.read", "project_view"],
            "exp": 4102444800_u64
        }),
        &EncodingKey::from_secret(secret.as_bytes()),
    )
    .unwrap();
    let app = router_with_application(
        ApiState {
            health: snapshot(),
            required_roles: vec!["reader".to_owned()],
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
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/runs/evidence-run/evidence")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let payload =
        serde_json::from_slice::<Value>(&to_bytes(response.into_body(), 8192).await.unwrap())
            .unwrap();
    let expected: Value = serde_json::from_str(include_str!(
        "../../../../../contracts/golden/run-evidence-v1.json"
    ))
    .unwrap();
    assert_eq!(payload, expected);

    let mcp_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp/workflow")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::from(
                    r#"{"jsonrpc":"2.0","id":"evidence-mcp","method":"tools/call","params":{"name":"workflow.run_evidence","arguments":{"run_id":"evidence-run"}}}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(mcp_response.status(), StatusCode::OK);
    let mcp_payload = serde_json::from_slice::<Value>(
        &to_bytes(mcp_response.into_body(), 8192).await.unwrap(),
    )
    .unwrap();
    assert_eq!(mcp_payload["id"], "evidence-mcp");
    assert_eq!(mcp_payload["result"], expected);

    let other_user_token = encode(
        &Header::new(Algorithm::HS256),
        &json!({
            "user_id": "bob",
            "sub": "bob",
            "roles": ["reader"],
            "ns": ["workflow"],
            "storage_ns": "conversation",
            "capabilities": ["workflow.run.read", "project_view"],
            "exp": 4102444800_u64
        }),
        &EncodingKey::from_secret(secret.as_bytes()),
    )
    .unwrap();
    let hidden = app
        .oneshot(
            Request::builder()
                .uri("/api/runs/evidence-run/evidence")
                .header("authorization", format!("Bearer {other_user_token}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(hidden.status(), StatusCode::NOT_FOUND);

    drop(service);
    let (cleanup, connection) = tokio_postgres::connect(&dsn, tokio_postgres::NoTls)
        .await
        .unwrap();
    let cleanup_task = tokio::spawn(connection);
    cleanup
        .batch_execute(&format!(
            "DROP SCHEMA IF EXISTS \"{schema}_conversation\" CASCADE; \
             DROP SCHEMA IF EXISTS \"{schema}_workflow\" CASCADE; \
             DROP SCHEMA IF EXISTS \"{schema}_wisdom\" CASCADE; \
             DROP SCHEMA IF EXISTS \"{schema}\" CASCADE"
        ))
        .await
        .unwrap();
    drop(cleanup);
    cleanup_task.await.unwrap().unwrap();
}

#[tokio::test]
#[ignore = "requires KOGWISTAR_TEST_PG_DSN; uses a disposable PostgreSQL schema"]
async fn postgres_workflow_lineage_route_checks_each_workflow_acl() {
    let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
        .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
    let schema = format!("rust_run_lineage_{}", uuid::Uuid::new_v4().simple());
    let auth_path = std::env::temp_dir().join(format!(
        "kogwistar-run-lineage-auth-{}.sqlite",
        uuid::Uuid::new_v4()
    ));
    let auth_config = AuthConfig {
        auth_db_url: Some(format!("sqlite:///{}", auth_path.display())),
        ..AuthConfig::default()
    };
    let service = PostgresRunApplicationService::from_dsn(&dsn, &schema)
        .unwrap()
        .with_owner_auth_config(&auth_config)
        .await
        .unwrap();
    service.ensure_schema().await.unwrap();
    service
        .store
        .create_graph_schema(3, kogwistar_store_postgres::GraphTableNames::default())
        .await
        .unwrap();
    service
        .store
        .create_server_run(ServerRunCreate {
            run_id: "lineage-child".to_owned(),
            conversation_id: "lineage-conversation".to_owned(),
            workflow_id: "wf-child".to_owned(),
            user_id: Some("alice".to_owned()),
            user_turn_node_id: "lineage-turn".to_owned(),
            status: "completed".to_owned(),
        })
        .await
        .unwrap();

    let auth = SqliteAuthStore::open(&auth_path).unwrap();
    let auth_connection = rusqlite::Connection::open(&auth_path).unwrap();
    auth_connection
            .execute(
                "INSERT INTO users (user_id, email, is_active, created_at) VALUES ('alice', 'alice@example.test', 1, CURRENT_TIMESTAMP)",
                [],
            )
            .unwrap();
    auth_connection
            .execute(
                "INSERT INTO workflow_acl (workflow_id, user_id, role) VALUES ('wf-child', 'alice', 'ro')",
                [],
            )
            .unwrap();
    drop(auth_connection);

    let scope = GraphScope {
        namespace: "conversation".to_owned(),
        workspace_id: None,
        graph_space: None,
    };
    for (run_id, workflow_id, parent_run_id) in [
        ("lineage-child", "wf-child", Some("lineage-parent")),
        ("lineage-parent", "wf-parent", None),
    ] {
        let mut metadata = Map::new();
        metadata.insert("entity_type".to_owned(), json!("workflow_run"));
        metadata.insert("workflow_id".to_owned(), json!(workflow_id));
        metadata.insert("parent_run_id".to_owned(), json!(parent_run_id));
        metadata.insert("status".to_owned(), json!("completed"));
        metadata.insert("lifecycle_status".to_owned(), json!("active"));
        metadata.insert("wf_invoked".to_owned(), json!([]));
        metadata.insert("mentions".to_owned(), json!([]));
        service
            .store
            .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                record: GraphRecord {
                    id: format!("wf_run|{run_id}"),
                    document: Some(json!({"id": run_id}).to_string()),
                    metadata,
                    embedding: None,
                },
                embedding_dim: 3,
            })
            .await
            .unwrap();
        let mut acl_metadata = Map::new();
        acl_metadata.insert("entity_type".to_owned(), json!("acl_record"));
        acl_metadata.insert("acl_truth_graph".to_owned(), json!("conversation"));
        acl_metadata.insert("acl_target_grain".to_owned(), json!("node"));
        acl_metadata.insert(
            "acl_target_entity_id".to_owned(),
            json!(format!("wf_run|{run_id}")),
        );
        acl_metadata.insert("acl_mode".to_owned(), json!("public"));
        acl_metadata.insert("acl_version".to_owned(), json!(1));
        acl_metadata.insert("tombstoned".to_owned(), json!(false));
        service
            .store
            .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                record: GraphRecord {
                    id: format!("acl|wf_run|{run_id}"),
                    document: Some("{}".to_owned()),
                    metadata: acl_metadata,
                    embedding: None,
                },
                embedding_dim: 3,
            })
            .await
            .unwrap();
    }
    let stored_child = service
        .store
        .graph_projection_records(GraphProjectionRead {
            scope: scope.clone(),
            table: "gke_nodes".to_owned(),
            ids: Some(vec!["wf_run|lineage-child".to_owned()]),
            metadata: MetadataFilter::default(),
            limit: 1,
        })
        .await
        .unwrap();
    assert_eq!(stored_child.len(), 1, "persisted child run node not found");

    let secret = "lineage-route-test-secret";
    let token = encode(
        &Header::new(Algorithm::HS256),
        &json!({
            "user_id": "alice",
            "sub": "alice",
            "roles": ["reader"],
            "ns": ["workflow"],
            "storage_ns": "conversation",
            "capabilities": ["workflow.run.read"],
            "exp": 4102444800_u64
        }),
        &EncodingKey::from_secret(secret.as_bytes()),
    )
    .unwrap();
    let app = router_with_application(
        ApiState {
            health: snapshot(),
            required_roles: vec!["reader".to_owned()],
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
    let request = || {
        Request::builder()
            .uri("/api/workflow/runs/lineage-child/lineage")
            .header("authorization", format!("Bearer {token}"))
            .body(Body::empty())
            .unwrap()
    };
    let denied = app.clone().oneshot(request()).await.unwrap();
    assert_eq!(
        denied.status(),
        StatusCode::FORBIDDEN,
        "unexpected lineage response"
    );
    let denied_body = to_bytes(denied.into_body(), 8192).await.unwrap();
    assert_eq!(
        serde_json::from_slice::<Value>(&denied_body).unwrap()["code"],
        "KOGWISTAR_WORKFLOW_FORBIDDEN"
    );

    rusqlite::Connection::open(&auth_path)
            .unwrap()
            .execute(
                "INSERT INTO workflow_acl (workflow_id, user_id, role) VALUES ('wf-parent', 'alice', 'ro')",
                [],
            )
            .unwrap();
    let allowed = app.clone().oneshot(request()).await.unwrap();
    assert_eq!(allowed.status(), StatusCode::OK);
    let allowed_body = to_bytes(allowed.into_body(), 8192).await.unwrap();
    let payload = serde_json::from_slice::<Value>(&allowed_body).unwrap();
    assert_eq!(payload["run_id"], "lineage-child");
    assert_eq!(payload["lineage"].as_array().unwrap().len(), 2);
    assert_eq!(payload["lineage"][0]["workflow_id"], "wf-child");
    assert_eq!(payload["lineage"][0]["parent_run_id"], "lineage-parent");
    assert_eq!(payload["lineage"][1]["workflow_id"], "wf-parent");

    let mcp_response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp/workflow")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::from(
                    r#"{"jsonrpc":"2.0","id":"lineage-mcp","method":"tools/call","params":{"name":"workflow.run_lineage","arguments":{"run_id":"lineage-child","limit":1}}}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(mcp_response.status(), StatusCode::OK);
    let mcp_payload = serde_json::from_slice::<Value>(
        &to_bytes(mcp_response.into_body(), 8192).await.unwrap(),
    )
    .unwrap();
    assert_eq!(mcp_payload["id"], "lineage-mcp");
    assert_eq!(mcp_payload["result"]["run_id"], "lineage-child");
    assert_eq!(mcp_payload["result"]["lineage"].as_array().unwrap().len(), 1);
    assert_eq!(mcp_payload["result"]["truncated"], true);
    assert_eq!(mcp_payload["result"]["lineage"][0]["workflow_id"], "wf-child");

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
    drop(auth);
    let _ = std::fs::remove_file(auth_path);
}

#[tokio::test]
#[ignore = "requires KOGWISTAR_TEST_PG_DSN; exercises a disposable wisdom schema"]
async fn postgres_wisdom_graph_upsert_is_scoped_atomic_and_acl_checked() {
    let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
        .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
    let schema = format!("rust_wisdom_upsert_{}", uuid::Uuid::new_v4().simple());
    let service = PostgresRunApplicationService::from_dsn(&dsn, &schema).unwrap();
    service.ensure_schema().await.unwrap();
    service
        .store
        .create_graph_schema(3, kogwistar_store_postgres::GraphTableNames::default())
        .await
        .unwrap();
    service
        .wisdom_store
        .create_graph_schema(3, kogwistar_store_postgres::GraphTableNames::default())
        .await
        .unwrap();
    let calls = Arc::new(Mutex::new(Vec::new()));
    let profile = kogwistar_engine::embedding::EmbeddingProfile {
        provider: "test".to_owned(),
        model: "wisdom-fixed-v1".to_owned(),
        dimension: 3,
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
    let service = service.with_embedding_provider(Arc::new(FixedEmbeddingProvider {
        profile: profile.clone(),
        vector: vec![0.0, 0.0, 1.0],
        calls: calls.clone(),
        fail: None,
    }));
    service
        .ensure_embedding_profile_binding(&profile)
        .await
        .unwrap();
    let primary_embedding_binding = service
        .store
        .get_named_projection(
            EMBEDDING_PROFILE_NAMESPACE,
            service.embedding_storage_scope.as_ref().unwrap(),
        )
        .await
        .unwrap();
    let wisdom_embedding_binding = service
        .wisdom_store
        .get_named_projection(
            EMBEDDING_PROFILE_NAMESPACE,
            service.wisdom_embedding_storage_scope.as_ref().unwrap(),
        )
        .await
        .unwrap();
    assert!(primary_embedding_binding.is_some());
    assert!(wisdom_embedding_binding.is_some());

    let mut graph_batch: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/graph-mutation-batch-v1.json"
    )))
    .unwrap();
    let graph_node_id = "regular-semantic-node";
    let graph_node = json!({
        "id": graph_node_id,
        "label": "Rust semantic retrieval fixture",
        "type": "concept",
        "summary": "A normal graph node used to verify semantic MCP retrieval.",
        "metadata": {"namespace": "docs", "workspace_id": null, "graph_space": null, "doc_id": "semantic-seed-doc"}
    });
    graph_batch["mutations"][0]["scope"] = json!({
        "namespace": "docs", "workspace_id": null, "graph_space": null
    });
    graph_batch["mutations"][0]["event_id"] = json!("regular-semantic-event");
    graph_batch["mutations"][0]["payload"] = graph_node.clone();
    graph_batch["mutations"][0]["record"]["id"] = json!(graph_node_id);
    graph_batch["mutations"][0]["record"]["metadata"]["doc_id"] =
        json!("semantic-seed-doc");
    graph_batch["mutations"][0]["record"]["document"] =
        json!(serde_json::to_string(&graph_node).unwrap());
    graph_batch["mutations"][0]["record"]["embedding"] = json!([0.0, 0.0, 1.0]);
    graph_batch["mutations"][0]["embedding_dim"] = json!(3);
    let mut acl_mutation = graph_batch["mutations"][0].clone();
    let acl_metadata = json!({
        "entity_type": "acl_record",
        "acl_truth_graph": "knowledge",
        "acl_target_entity_id": graph_node_id,
        "acl_target_grain": "node",
        "acl_version": 1,
        "acl_mode": "private",
        "owner_id": "alice",
        "tombstoned": false
    });
    let acl_id = "regular-semantic-node-acl";
    acl_mutation["event_id"] = json!("regular-semantic-acl-event");
    acl_mutation["payload"] = json!({"id": acl_id, "metadata": acl_metadata});
    acl_mutation["record"] = json!({
        "id": acl_id,
        "document": serde_json::to_string(&json!({"id": acl_id})).unwrap(),
        "metadata": acl_metadata,
        "embedding": null
    });
    acl_mutation["embedding_dim"] = json!(0);
    graph_batch["mutations"]
        .as_array_mut()
        .unwrap()
        .push(acl_mutation);
    let graph_write = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: INTERNAL_GRAPH_MUTATIONS_ROUTE.to_owned(),
            body: serde_json::to_vec(&graph_batch).unwrap(),
            principal: json!({
                "sub": "alice", "user_id": "alice", "role": "rw",
                "ns": ["docs"], "storage_ns": "docs",
                "capabilities": ["write_graph"]
            }),
        })
        .await;
    assert_eq!(
        graph_write.status,
        StatusCode::OK.as_u16(),
        "{}",
        String::from_utf8_lossy(&graph_write.body)
    );
    let cross_document_edge_id = "regular-semantic-cross-document-edge";
    let graph_scope = GraphScope {
        namespace: "docs".to_owned(),
        workspace_id: None,
        graph_space: None,
    };
    let cross_document_edge_document = json!({
        "id": cross_document_edge_id,
        "relation": "related_to",
        "source_ids": [graph_node_id],
        "target_ids": [graph_node_id],
        "doc_id": "semantic-expansion-doc"
    });
    let cross_document_edge = GraphMutation {
        scope: graph_scope.clone(),
        table: "gke_edges".to_owned(),
        entity_kind: "edge".to_owned(),
        event_id: "regular-semantic-cross-document-edge-event".to_owned(),
        op: "ADD".to_owned(),
        payload: cross_document_edge_document.clone(),
        record: GraphRecord {
            id: cross_document_edge_id.to_owned(),
            document: Some(cross_document_edge_document.to_string()),
            metadata: serde_json::from_value(json!({
                "namespace": "docs",
                "doc_id": "semantic-expansion-doc",
                "source_ids": [graph_node_id],
                "target_ids": [graph_node_id],
                "relation": "related_to"
            }))
            .unwrap(),
            embedding: None,
        },
        embedding_dim: 0,
    };
    let cross_document_edge_acl_metadata = json!({
        "entity_type": "acl_record",
        "acl_truth_graph": "knowledge",
        "acl_target_entity_id": cross_document_edge_id,
        "acl_target_grain": "edge",
        "acl_version": 1,
        "acl_mode": "private",
        "owner_id": "alice",
        "tombstoned": false
    });
    let cross_document_edge_acl = GraphMutation {
        scope: graph_scope,
        table: "gke_nodes".to_owned(),
        entity_kind: "node".to_owned(),
        event_id: "regular-semantic-cross-document-edge-acl-event".to_owned(),
        op: "ADD".to_owned(),
        payload: json!({"id": "regular-semantic-cross-document-edge-acl", "metadata": cross_document_edge_acl_metadata}),
        record: GraphRecord {
            id: "regular-semantic-cross-document-edge-acl".to_owned(),
            document: None,
            metadata: serde_json::from_value(cross_document_edge_acl_metadata).unwrap(),
            embedding: None,
        },
        embedding_dim: 0,
    };
    service
        .store
        .apply_graph_mutations_atomic(vec![cross_document_edge, cross_document_edge_acl])
        .await
        .unwrap();
    let inserted_graph_node = service
        .store
        .graph_projection_records(GraphProjectionRead {
            scope: GraphScope {
                namespace: "docs".to_owned(),
                workspace_id: None,
                graph_space: None,
            },
            table: "gke_nodes".to_owned(),
            ids: Some(vec![graph_node_id.to_owned()]),
            metadata: MetadataFilter::default(),
            limit: 1,
        })
        .await
        .unwrap();
    assert_eq!(
        inserted_graph_node.len(),
        1,
        "graph mutation did not create projection"
    );
    let fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/python-graph-model-payloads-v1.json"
    )))
    .unwrap();
    let mut node = fixture["cases"][0]["payload"].clone();
    node["id"] = json!("wisdom-private-node");
    node["embedding"] = json!([1.0, 0.0, 0.0]);
    let mut edge = fixture["cases"][1]["payload"].clone();
    edge["id"] = json!("wisdom-private-edge");
    edge["source_ids"] = json!(["wisdom-private-node"]);
    edge["target_ids"] = json!([]);
    edge["embedding"] = json!([0.0, 1.0, 0.0]);
    let alice = json!({
        "sub": "alice", "user_id": "alice", "role": "rw",
        "ns": ["wisdom"], "storage_ns": "wisdom",
        "capabilities": ["write_graph"]
    });
    let response = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: PUBLIC_WISDOM_GRAPH_UPSERT_ROUTE.to_owned(),
            body: serde_json::to_vec(
                &json!({"inp": {"nodes": [node.clone()], "edges": [edge.clone()]}}),
            )
            .unwrap(),
            principal: alice,
        })
        .await;
    assert_eq!(
        response.status,
        StatusCode::OK.as_u16(),
        "{}",
        String::from_utf8_lossy(&response.body)
    );
    assert_eq!(
        *calls.lock().unwrap(),
        vec![vec![
            "Python node: A complete Pydantic node payload crosses the Rust persistence seam."
                .to_owned(),
            "supports: The typed edge preserves both endpoint and source-span data.".to_owned(),
        ],]
    );
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
    use tower::ServiceExt;

    let secret = "wisdom-semantic-mcp-test-secret";
    let token = encode(
        &Header::new(Algorithm::HS256),
        &json!({
            "sub": "alice", "user_id": "alice", "role": "ro", "roles": ["reader"],
            "ns": ["wisdom"], "storage_ns": "wisdom",
            "capabilities": ["read_graph"], "exp": 4102444800_u64
        }),
        &EncodingKey::from_secret(secret.as_bytes()),
    )
    .unwrap();
    let app = router_with_application(
        ApiState {
            health: snapshot(),
            required_roles: vec!["reader".to_owned()],
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
    let semantic_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::from(
                    serde_json::to_vec(&json!({
                        "jsonrpc": "2.0", "id": 7, "method": "tools/call",
                        "params": {
                            "name": "wisdom.semantic_seed_then_expand",
                            "arguments": {"text": "find the node by meaning", "top_k": 5, "hops": 1}
                        }
                    }))
                    .unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(semantic_response.status(), StatusCode::OK);
    let semantic_body = to_bytes(semantic_response.into_body(), 65_536)
        .await
        .unwrap();
    let semantic: Value = serde_json::from_slice(&semantic_body).unwrap();
    assert!(semantic.get("error").is_none(), "{semantic}");
    let semantic = &semantic["result"];
    assert_eq!(semantic["seeds"].as_array().unwrap().len(), 1);
    let seed: Value = serde_json::from_str(semantic["seeds"][0].as_str().unwrap()).unwrap();
    assert_eq!(seed["id"], "wisdom-private-node");
    assert_eq!(semantic["layers"].as_array().unwrap().len(), 1);
    let edge_docs = semantic["layers"][0]["edges"].as_array().unwrap();
    assert_eq!(edge_docs.len(), 1);
    let edge_doc: Value = serde_json::from_str(edge_docs[0].as_str().unwrap()).unwrap();
    assert_eq!(edge_doc["id"], "wisdom-private-edge");

    let graph_semantic_response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!("Bearer {}", encode(
                        &Header::new(Algorithm::HS256),
                        &json!({
                            "sub": "alice", "user_id": "alice", "role": "ro", "roles": ["reader"],
                            "ns": ["docs"], "storage_ns": "docs",
                            "capabilities": ["read_graph"], "exp": 4102444800_u64
                        }),
                        &EncodingKey::from_secret(secret.as_bytes()),
                    ).unwrap()),
                )
                .body(Body::from(
                    serde_json::to_vec(&json!({
                        "jsonrpc": "2.0", "id": 8, "method": "tools/call",
                        "params": {
                            "name": "kg_semantic_seed_then_expand_text",
                            "arguments": {"text": "find the fixture concept by meaning", "top_k": 1, "hops": 1, "doc_ids": "semantic-seed-doc"}
                        }
                    }))
                    .unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(graph_semantic_response.status(), StatusCode::OK);
    let graph_semantic_body = to_bytes(graph_semantic_response.into_body(), 65_536)
        .await
        .unwrap();
    let graph_semantic: Value = serde_json::from_slice(&graph_semantic_body).unwrap();
    assert!(graph_semantic.get("error").is_none(), "{graph_semantic}");
    let graph_semantic = &graph_semantic["result"];
    let graph_seed = graph_semantic["seeds"].as_array().unwrap();
    assert_eq!(
        graph_seed.len(),
        1,
        "regular semantic result: {graph_semantic}"
    );
    assert_eq!(
        serde_json::from_str::<Value>(graph_seed[0].as_str().unwrap()).unwrap()["id"],
        graph_node_id
    );
    let expanded_edge_ids = graph_semantic["layers"][0]["edges"]
        .as_array()
        .unwrap()
        .iter()
        .map(|edge| {
            serde_json::from_str::<Value>(edge.as_str().unwrap()).unwrap()["id"]
                .as_str()
                .unwrap()
                .to_owned()
        })
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(expanded_edge_ids, [cross_document_edge_id.to_owned()].into());

    let wisdom_nodes = service
        .wisdom_store
        .graph_projection_records(GraphProjectionRead {
            scope: GraphScope {
                namespace: "wisdom".to_owned(),
                workspace_id: None,
                graph_space: None,
            },
            table: "gke_nodes".to_owned(),
            ids: Some(vec!["wisdom-private-node".to_owned()]),
            metadata: MetadataFilter::default(),
            limit: 1,
        })
        .await
        .unwrap();
    assert_eq!(wisdom_nodes.len(), 1);
    assert_eq!(
        wisdom_nodes[0].embedding.as_deref(),
        Some([0.0, 0.0, 1.0].as_slice())
    );
    let primary_nodes = service
        .store
        .graph_projection_records(GraphProjectionRead {
            scope: GraphScope {
                namespace: "wisdom".to_owned(),
                workspace_id: None,
                graph_space: None,
            },
            table: "gke_nodes".to_owned(),
            ids: Some(vec!["wisdom-private-node".to_owned()]),
            metadata: MetadataFilter::default(),
            limit: 1,
        })
        .await
        .unwrap();
    assert!(
        primary_nodes.is_empty(),
        "wisdom writes stay in the isolated schema"
    );

    let bob = json!({
        "sub": "bob", "user_id": "bob", "role": "rw",
        "ns": ["wisdom"], "storage_ns": "wisdom",
        "capabilities": ["write_graph"]
    });
    let overwrite = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: PUBLIC_WISDOM_GRAPH_UPSERT_ROUTE.to_owned(),
            body: serde_json::to_vec(&json!({"inp": {"nodes": [node.clone()], "edges": []}}))
                .unwrap(),
            principal: bob,
        })
        .await;
    assert_ne!(overwrite.status, StatusCode::OK.as_u16());

    let mut hidden_reference_edge = edge.clone();
    hidden_reference_edge["id"] = json!("bob-reference-edge");
    hidden_reference_edge["source_ids"] = json!(["wisdom-private-node"]);
    hidden_reference_edge["target_ids"] = json!([]);
    let hidden_reference = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: PUBLIC_WISDOM_GRAPH_UPSERT_ROUTE.to_owned(),
            body: serde_json::to_vec(
                &json!({"inp": {"nodes": [], "edges": [hidden_reference_edge]}}),
            )
            .unwrap(),
            principal: json!({
                "sub": "bob", "user_id": "bob", "role": "rw", "ns": ["wisdom"],
                "storage_ns": "wisdom", "capabilities": ["write_graph"]
            }),
        })
        .await;
    assert_ne!(hidden_reference.status, StatusCode::OK.as_u16());

    let missing_endpoint = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: PUBLIC_WISDOM_GRAPH_UPSERT_ROUTE.to_owned(),
            body: serde_json::to_vec(&json!({"inp": {"nodes": [], "edges": [{
                "id": "missing-endpoint-edge", "label": "references", "type": "relationship",
                "summary": "references a missing node", "relation": "references",
                "source_ids": ["absent-wisdom-node"], "target_ids": [],
                "source_edge_ids": [], "target_edge_ids": [], "metadata": {},
                "mentions": [], "embedding": null
            }]}}))
            .unwrap(),
            principal: json!({
                "sub": "alice", "user_id": "alice", "role": "rw", "ns": ["wisdom"],
                "storage_ns": "wisdom", "capabilities": ["write_graph"]
            }),
        })
        .await;
    assert_eq!(
        missing_endpoint.status,
        StatusCode::UNPROCESSABLE_ENTITY.as_u16()
    );

    let shared_service =
        PostgresRunApplicationService::from_dsn_with_wisdom_schema(&dsn, &schema, &schema)
            .unwrap()
            .with_embedding_provider(Arc::new(FixedEmbeddingProvider {
                profile: profile.clone(),
                vector: vec![0.0, 0.0, 1.0],
                calls: Arc::new(Mutex::new(Vec::new())),
                fail: None,
            }));
    shared_service.ensure_schema().await.unwrap();
    shared_service
        .ensure_embedding_profile_binding(&profile)
        .await
        .unwrap();
    let mut shared_node = node.clone();
    shared_node["id"] = json!("wisdom-shared-schema-node");
    let shared_write = shared_service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: PUBLIC_WISDOM_GRAPH_UPSERT_ROUTE.to_owned(),
            body: serde_json::to_vec(&json!({"inp": {"nodes": [shared_node], "edges": []}}))
                .unwrap(),
            principal: json!({
                "sub": "alice", "user_id": "alice", "role": "rw",
                "ns": ["wisdom"], "storage_ns": "wisdom",
                "capabilities": ["write_graph"]
            }),
        })
        .await;
    assert_eq!(
        shared_write.status,
        StatusCode::OK.as_u16(),
        "{}",
        String::from_utf8_lossy(&shared_write.body)
    );
    let shared_role_records = shared_service
        .store
        .graph_projection_records(GraphProjectionRead {
            scope: GraphScope {
                namespace: "wisdom".to_owned(),
                workspace_id: None,
                graph_space: None,
            },
            table: "gke_nodes".to_owned(),
            ids: Some(vec!["wisdom-shared-schema-node".to_owned()]),
            metadata: MetadataFilter::default(),
            limit: 1,
        })
        .await
        .unwrap();
    assert_eq!(shared_role_records.len(), 1);
    let other_namespace_records = shared_service
        .store
        .graph_projection_records(GraphProjectionRead {
            scope: GraphScope {
                namespace: "knowledge".to_owned(),
                workspace_id: None,
                graph_space: None,
            },
            table: "gke_nodes".to_owned(),
            ids: Some(vec!["wisdom-shared-schema-node".to_owned()]),
            metadata: MetadataFilter::default(),
            limit: 1,
        })
        .await
        .unwrap();
    assert!(other_namespace_records.is_empty());

    drop(service);
    drop(shared_service);
    let (cleanup, connection) = tokio_postgres::connect(&dsn, tokio_postgres::NoTls)
        .await
        .unwrap();
    let cleanup_task = tokio::spawn(connection);
    cleanup
        .batch_execute(&format!(
            "DROP SCHEMA \"{schema}\" CASCADE; DROP SCHEMA \"{schema}_wisdom\" CASCADE"
        ))
        .await
        .unwrap();
    drop(cleanup);
    cleanup_task.await.unwrap().unwrap();
}

#[test]
fn wisdom_mcp_tool_discovery_requires_authorized_wisdom_write_scope() {
    let service = PostgresRunApplicationService::from_dsn(
        "postgres://unused:unused@127.0.0.1:1/unused",
        "wisdom_tool_discovery",
    )
    .unwrap();
    let reader = json!({
        "role": "ro", "ns": ["wisdom"], "capabilities": ["read_graph"]
    });
    assert!(!service.supports_mcp_tool_for_principal(
        "root",
        "wisdom.kg_upsert_graph",
        &json!({"role": "rw", "ns": ["wisdom"], "capabilities": ["write_graph"]}),
    ));
    assert!(!service.supports_mcp_tool_for_principal(
        "root",
        "wisdom.semantic_seed_then_expand",
        &reader,
    ));
    assert!(!service.supports_mcp_tool_for_principal(
        "root",
        "kg_semantic_seed_then_expand_text",
        &json!({"role": "ro", "ns": ["docs"], "capabilities": ["read_graph"]}),
    ));
    let profile = kogwistar_engine::embedding::EmbeddingProfile {
        provider: "test".to_owned(),
        model: "wisdom-discovery-v1".to_owned(),
        dimension: 3,
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
    let service = service.with_embedding_provider(Arc::new(FixedEmbeddingProvider {
        profile,
        vector: vec![0.0, 0.0, 1.0],
        calls: Arc::new(Mutex::new(Vec::new())),
        fail: None,
    }));
    for principal in [
        json!({"role": "rw", "ns": ["wisdom"], "capabilities": ["read_graph"]}),
        json!({"role": "ro", "ns": ["wisdom"], "capabilities": ["write_graph"]}),
        json!({"role": "rw", "ns": ["docs"], "capabilities": ["write_graph"]}),
    ] {
        assert!(!service.supports_mcp_tool_for_principal(
            "root",
            "wisdom.kg_upsert_graph",
            &principal,
        ));
    }
    let authorized = json!({
        "role": "rw", "ns": ["wisdom"], "capabilities": ["write_graph"]
    });
    assert!(
        service.supports_mcp_tool_for_principal("root", "wisdom.kg_upsert_graph", &authorized,)
    );
    assert!(!service.supports_mcp_tool_for_principal(
        "root",
        "wisdom.semantic_seed_then_expand",
        &authorized,
    ));
    assert!(service.supports_mcp_tool_for_principal(
        "root",
        "wisdom.semantic_seed_then_expand",
        &reader,
    ));
    let graph_reader = json!({
        "role": "ro", "ns": ["docs"], "storage_ns": "docs",
        "capabilities": ["read_graph"]
    });
    assert!(service.supports_mcp_tool_for_principal(
        "root",
        "kg_semantic_seed_then_expand_text",
        &graph_reader,
    ));
    assert!(!service.supports_mcp_tool_for_principal(
        "root",
        "kg_semantic_seed_then_expand_text",
        &json!({"role": "ro", "ns": ["docs"], "capabilities": []}),
    ));
    assert!(!service.supports_mcp_tool_for_principal(
        "conversation",
        "kg_semantic_seed_then_expand_text",
        &graph_reader,
    ));
    assert!(!service.supports_mcp_tool_for_principal(
        "root",
        "wisdom.semantic_seed_then_expand",
        &json!({"role": "ro", "ns": ["other"], "capabilities": ["read_graph"]}),
    ));
    assert!(!service.supports_mcp_tool_for_principal(
        "conversation",
        "wisdom.kg_upsert_graph",
        &authorized,
    ));
}

#[test]
fn graph_read_mcp_discovery_matches_namespace_and_capability_admission() {
    let service = PostgresRunApplicationService::from_dsn(
        "postgres://unused:unused@127.0.0.1:1/unused",
        "graph_read_discovery",
    )
    .unwrap();
    let tools = [
        "kg_find_edges",
        "kg_k_hop",
        "kg_neighbors",
        "kg_shortest_path",
    ];
    let authorized_reader = json!({
        "role": "ro", "ns": ["docs"], "capabilities": ["read_graph"]
    });
    for tool in tools {
        assert!(
            service.supports_mcp_tool_for_principal("root", tool, &authorized_reader),
            "authorized docs reader should discover {tool}"
        );
        assert!(!service.supports_mcp_tool_for_principal(
            "root",
            tool,
            &json!({"role": "ro", "ns": ["docs"], "capabilities": []}),
        ));
        assert!(!service.supports_mcp_tool_for_principal(
            "root",
            tool,
            &json!({"role": "ro", "ns": ["workflow"], "capabilities": ["read_graph"]}),
        ));
        assert!(!service.supports_mcp_tool_for_principal(
            "root",
            tool,
            &json!({"role": "rw", "ns": ["docs"], "capabilities": []}),
        ));
        assert!(
            !service.supports_mcp_tool_for_principal("conversation", tool, &authorized_reader,)
        );
    }
}
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
#[tokio::test]
async fn sqlite_run_admission_persists_authenticated_owner_binding() {
    let sqlite_path = std::env::temp_dir().join(format!(
        "kogwistar-sqlite-owner-binding-{}.sqlite",
        uuid::Uuid::new_v4().simple()
    ));
    let auth_path = std::env::temp_dir().join(format!(
        "kogwistar-sqlite-owner-binding-auth-{}.sqlite",
        uuid::Uuid::new_v4().simple()
    ));
    let auth = SqliteAuthStore::open(&auth_path).unwrap();
    auth.resolve_external_identity(ResolveExternalIdentity {
        issuer: "test".to_owned(),
        subject: "sqlite-owner".to_owned(),
        email: "sqlite-owner@example.test".to_owned(),
        display_name: None,
        new_user_id: "sqlite-owner".to_owned(),
        default_role: "rw".to_owned(),
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
    let principal = json!({
        "user_id": "sqlite-owner",
        "sub": "sqlite-owner@example.test",
        "role": "rw",
        "ns": ["workflow"],
        "storage_ns": "workflow",
        "capabilities": ["workflow.run.write", "write_graph"]
    });
    let response = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/workflow/runs".to_owned(),
            body: serde_json::to_vec(&json!({
                "run_id": "sqlite-owner-bound-run",
                "workflow_id": "sqlite-owner-bound-workflow",
                "conversation_id": "sqlite-owner-bound-conversation",
                "user_id": "sqlite-owner"
            }))
            .unwrap(),
            principal: principal.clone(),
        })
        .await;
    assert_eq!(response.status, StatusCode::ACCEPTED.as_u16());
    let run = service
        .store
        .get_server_run("sqlite-owner-bound-run")
        .unwrap()
        .unwrap();
    assert_eq!(run.user_id.as_deref(), Some("sqlite-owner"));
    let created = service
        .store
        .list_server_run_events("sqlite-owner-bound-run", 0, 1)
        .unwrap()
        .into_iter()
        .find(|event| event.event_type == "run.created")
        .unwrap();
    let payload: Value = serde_json::from_str(&created.payload_json).unwrap();
    assert_eq!(payload["runtime_owner_binding"]["version"], 1);
    assert_eq!(payload["runtime_owner_binding"]["user_id"], "sqlite-owner");
    assert!(
        payload["runtime_owner_binding"]["admitted_capabilities"]
            .as_array()
            .unwrap()
            .iter()
            .any(|capability| capability == "write_graph")
    );
    let revalidated = service
        .resolve_inline_native_owner_principal("sqlite-owner-bound-run", "write_graph")
        .await
        .unwrap();
    assert_eq!(
        authenticated_run_owner_id(&revalidated),
        Some("sqlite-owner")
    );
    assert!(
        principal_strings(&revalidated, "capabilities")
            .iter()
            .any(|capability| capability == "write_graph")
    );

    let mismatch = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/workflow/runs".to_owned(),
            body: serde_json::to_vec(&json!({
                "run_id": "sqlite-owner-mismatch-run",
                "workflow_id": "sqlite-owner-bound-workflow",
                "conversation_id": "sqlite-owner-bound-conversation",
                "user_id": "another-user"
            }))
            .unwrap(),
            principal,
        })
        .await;
    assert_eq!(mismatch.status, StatusCode::FORBIDDEN.as_u16());
    assert!(
        service
            .store
            .get_server_run("sqlite-owner-mismatch-run")
            .unwrap()
            .is_none()
    );

    drop(service);
    remove_test_sqlite(&sqlite_path);
    remove_test_sqlite(&auth_path);
}

#[tokio::test]
async fn sqlite_chroma_public_document_upsert_is_scope_isolated_and_durable() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let chroma_setup = tokio::spawn(async move {
        for _ in 0..2 {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut bytes = Vec::new();
            let (header_end, content_length) = loop {
                let mut chunk = [0_u8; 1024];
                let count = stream.read(&mut chunk).await.unwrap();
                assert!(count > 0);
                bytes.extend_from_slice(&chunk[..count]);
                if let Some(header_end) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
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
            let body = json!({
                "id": format!("collection-{}", request["name"].as_str().unwrap()),
                "metadata": {"hnsw:space": "cosine"}
            })
            .to_string();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(response.as_bytes()).await.unwrap();
        }
    });
    let config = |collection: &str| kogwistar_store_chroma::ChromaStoreConfig {
        endpoint: endpoint.clone(),
        collection: collection.to_owned(),
        ..Default::default()
    };
    let graph = kogwistar_store_chroma::ChromaGraphStore::connect_entity_collections(
        config("source-doc-nodes"),
        config("source-doc-edges"),
    )
    .await
    .unwrap();
    chroma_setup.await.unwrap();

    let path = std::env::temp_dir().join(format!(
        "kogwistar-sqlite-chroma-source-document-{}.sqlite",
        uuid::Uuid::new_v4()
    ));
    let service = SqliteRunApplicationService::open(&path)
        .unwrap()
        .with_chroma_graph(graph);
    assert!(service.supports_http_operation("POST", "/api/document"));
    let body = json!({
        "contract_version": 1,
        "doc_id": "source-doc-1",
        "doc_type": "text",
        "insertion_method": "parser-v1",
        "content": "The authoritative source text.",
        "metadata": {"origin": "test"}
    });
    let principal = json!({
        "sub": "alice",
        "user_id": "alice",
        "role": "rw",
        "ns": ["docs"],
        "storage_ns": "docs",
        "workspace": "workspace-a",
        "graph_space": "knowledge",
        "capabilities": ["write_graph"]
    });
    let response = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/document".to_owned(),
            body: serde_json::to_vec(&body).unwrap(),
            principal: principal.clone(),
        })
        .await;
    assert_eq!(response.status, StatusCode::OK.as_u16());
    let scope = GraphScope {
        namespace: "docs".to_owned(),
        workspace_id: Some("workspace-a".to_owned()),
        graph_space: Some("knowledge".to_owned()),
    };
    let stored = service
        .store
        .source_document_projection(&scope, "source-doc-1")
        .unwrap()
        .unwrap();
    assert_eq!(
        stored.document.as_deref(),
        Some("The authoritative source text.")
    );
    assert_eq!(stored.metadata["origin"], "test");
    assert_eq!(stored.metadata["doc_type"], "text");
    assert!(
        service
            .store
            .source_document_projection(
                &GraphScope {
                    graph_space: Some("wisdom".to_owned()),
                    ..scope.clone()
                },
                "source-doc-1",
            )
            .unwrap()
            .is_none()
    );

    let denied = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/document".to_owned(),
            body: serde_json::to_vec(&json!({
                "contract_version": 1,
                "doc_id": "denied-doc",
                "content": "must not persist"
            }))
            .unwrap(),
            principal: json!({"role": "ro", "ns": ["docs"]}),
        })
        .await;
    assert_eq!(denied.status, StatusCode::FORBIDDEN.as_u16());
    assert!(
        service
            .store
            .source_document_projection(&scope, "denied-doc")
            .unwrap()
            .is_none()
    );

    drop(service);
    let reopened = SqliteStore::open(&path).unwrap();
    assert_eq!(
        reopened
            .source_document_projection(&scope, "source-doc-1")
            .unwrap()
            .unwrap()
            .document
            .as_deref(),
        Some("The authoritative source text.")
    );
    drop(reopened);
    remove_test_sqlite(&path);
}

#[tokio::test]
async fn sqlite_chroma_graph_mutation_commits_canonical_events_and_eventual_jobs() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let (stop_chroma, mut stop_chroma_rx) = tokio::sync::oneshot::channel();
    let fake_chroma = tokio::spawn(async move {
        let mut projected = Vec::new();
        let mut fail_first_upsert = true;
        loop {
            let (mut stream, _) = tokio::select! {
                _ = &mut stop_chroma_rx => break,
                connection = listener.accept() => connection.unwrap(),
            };
            let mut bytes = Vec::new();
            let mut chunk = [0_u8; 2048];
            let (header_end, content_length) = loop {
                let count = stream.read(&mut chunk).await.unwrap();
                assert!(count > 0, "Chroma connection closed before request body");
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
            let request_line = String::from_utf8_lossy(&bytes[..header_end])
                .lines()
                .next()
                .unwrap_or_default()
                .to_owned();
            let (status, response) = if request_line.contains("/collections HTTP") {
                (
                    "200 OK",
                    json!({
                        "id": format!("collection-{}", request["name"].as_str().unwrap()),
                        "metadata": {"hnsw:space":"cosine"}
                    }),
                )
            } else {
                projected.push((request_line, request));
                if fail_first_upsert
                    && projected.last().unwrap().0.contains("/upsert")
                    && projected.last().unwrap().1["ids"][0] == "node-1"
                {
                    fail_first_upsert = false;
                    (
                        "500 Internal Server Error",
                        json!({"error":"injected Chroma projection failure"}),
                    )
                } else {
                    ("200 OK", json!({}))
                }
            };
            let response = response.to_string();
            let wire = format!(
                "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                response.len(),
                response
            );
            stream.write_all(wire.as_bytes()).await.unwrap();
        }
        projected
    });

    let chroma_config = |collection: &str| kogwistar_store_chroma::ChromaStoreConfig {
        endpoint: endpoint.clone(),
        collection: collection.to_owned(),
        ..Default::default()
    };
    let graph = kogwistar_store_chroma::ChromaGraphStore::connect_entity_collections(
        chroma_config("nodes"),
        chroma_config("edges"),
    )
    .await
    .unwrap();
    let sqlite_path = std::env::temp_dir().join(format!(
        "kogwistar-sqlite-chroma-graph-mutation-{}.sqlite",
        uuid::Uuid::new_v4()
    ));
    let metadata_only = SqliteRunApplicationService::open(&sqlite_path).unwrap();
    assert!(!metadata_only.supports_http_operation("POST", INTERNAL_GRAPH_MUTATIONS_ROUTE));
    drop(metadata_only);
    let embedding_calls = Arc::new(Mutex::new(Vec::new()));
    let provider = Arc::new(FixedEmbeddingProvider {
        profile: kogwistar_engine::embedding::EmbeddingProfile {
            provider: "test".to_owned(),
            model: "fixed-v1".to_owned(),
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
        },
        vector: vec![0.6, 0.8],
        calls: embedding_calls.clone(),
        fail: None,
    });
    let service = SqliteRunApplicationService::open(&sqlite_path)
        .unwrap()
        .with_chroma_graph(graph)
        .with_embedding_provider(provider.clone());
    assert!(service.supports_http_operation("POST", INTERNAL_GRAPH_MUTATIONS_ROUTE));

    let mut batch: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/graph-mutation-batch-v1.json"
    )))
    .unwrap();
    batch["mutations"][0]["record"]["metadata"] = json!({
        "workspace_id":"ws", "graph_space":"facts"
    });
    let python_models: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/python-graph-model-payloads-v1.json"
    )))
    .unwrap();
    let mut node_payload = python_models["cases"][0]["payload"].clone();
    node_payload["id"] = json!("node-1");
    node_payload["embedding"] = json!([1.0, 0.0]);
    batch["mutations"][0]["payload"] = node_payload.clone();
    batch["mutations"][0]["model_envelope"] = json!({
        "schema_version": 1,
        "model_class": "node",
        "payload": node_payload,
    });
    let mut edge_mutation = batch["mutations"][0].clone();
    edge_mutation["table"] = json!("gke_edges");
    edge_mutation["entity_kind"] = json!("edge");
    edge_mutation["event_id"] = json!("event-edge");
    edge_mutation["record"]["id"] = json!("edge-1");
    let mut edge_payload = python_models["cases"][1]["payload"].clone();
    edge_payload["id"] = json!("edge-1");
    edge_payload["embedding"] = json!([1.0, 0.0]);
    edge_payload["source_ids"] = json!(["node-1"]);
    edge_mutation["payload"] = edge_payload.clone();
    edge_mutation["model_envelope"] = json!({
        "schema_version": 1,
        "model_class": "edge",
        "payload": edge_payload,
    });
    batch["mutations"]
        .as_array_mut()
        .unwrap()
        .push(edge_mutation);
    let body = serde_json::to_vec(&batch).unwrap();
    let principal = json!({
        "sub":"alice", "user_id":"alice", "role":"rw", "ns":["tenant-a"],
        "storage_ns":"tenant-a", "workspace":"ws", "graph_space":"facts",
        "capabilities":["write_graph"]
    });
    let denied = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: INTERNAL_GRAPH_MUTATIONS_ROUTE.to_owned(),
            body: body.clone(),
            principal: json!({"sub":"alice", "role":"ro", "ns":["tenant-a"], "storage_ns":"tenant-a", "workspace":"ws", "graph_space":"facts"}),
        })
        .await;
    assert_eq!(denied.status, StatusCode::FORBIDDEN.as_u16());
    assert!(
        service
            .store
            .replay_raw_events("tenant-a", 0, 10)
            .unwrap()
            .is_empty()
    );

    for (scope_key, value) in [
        ("namespace", "tenant-b"),
        ("workspace_id", "workspace-b"),
        ("graph_space", "other"),
    ] {
        let mut out_of_scope_batch = batch.clone();
        for mutation in out_of_scope_batch["mutations"].as_array_mut().unwrap() {
            mutation["scope"][scope_key] = json!(value);
            if scope_key != "namespace" {
                let metadata_key = if scope_key == "workspace_id" {
                    "workspace_id"
                } else {
                    "graph_space"
                };
                mutation["record"]["metadata"][metadata_key] = json!(value);
                mutation["payload"]["metadata"][metadata_key] = json!(value);
                mutation["model_envelope"]["payload"]["metadata"][metadata_key] =
                    json!(value);
            }
        }
        let out_of_scope = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: INTERNAL_GRAPH_MUTATIONS_ROUTE.to_owned(),
                body: serde_json::to_vec(&out_of_scope_batch).unwrap(),
                principal: principal.clone(),
            })
            .await;
        assert_eq!(out_of_scope.status, StatusCode::FORBIDDEN.as_u16(), "{scope_key}");
        assert!(service.store.replay_raw_events("tenant-a", 0, 10).unwrap().is_empty());
    }

    let request = || ApiEffectRequest {
        contract_version: 1,
        method: "POST".to_owned(),
        path_and_query: INTERNAL_GRAPH_MUTATIONS_ROUTE.to_owned(),
        body: body.clone(),
        principal: principal.clone(),
    };
    let first = service.execute(request()).await;
    assert_eq!(
        first.status,
        StatusCode::OK.as_u16(),
        "{}",
        String::from_utf8_lossy(&first.body)
    );
    let first_body: Value = serde_json::from_slice(&first.body).unwrap();
    assert_eq!(first_body["projection_consistency"], "eventual");
    assert_eq!(first_body["results"].as_array().unwrap().len(), 2);
    assert_eq!(first_body["results"][0]["inserted"], true);

    let retry = service.execute(request()).await;
    assert_eq!(
        retry.status,
        StatusCode::OK.as_u16(),
        "{}",
        String::from_utf8_lossy(&retry.body)
    );
    let retry_body: Value = serde_json::from_slice(&retry.body).unwrap();
    assert_eq!(retry_body["results"][0]["mutated"], false);
    let canonical_events = service.store.replay_raw_events("tenant-a", 0, 10).unwrap();
    assert_eq!(canonical_events.len(), 2);
    for event in &canonical_events {
        let payload: Value = serde_json::from_str(&event.payload_json).unwrap();
        let expected_class = if payload["id"] == "node-1" {
            "Node"
        } else {
            "Edge"
        };
        assert_eq!(payload["metadata"]["_class_name"], expected_class);
        assert_eq!(payload["metadata"]["workspace_id"], "ws");
        assert_eq!(payload["metadata"]["graph_space"], "facts");
    }
    let jobs = service
        .store
        .list_index_jobs(Some("tenant-a"), Some("PENDING"), None, None, None, 10)
        .unwrap();
    assert_eq!(jobs.len(), 2);
    assert!(jobs.iter().all(|job| job.index_kind == "node_embedding"));
    assert_eq!(
        jobs.iter()
            .map(|job| job.entity_kind.as_str())
            .collect::<std::collections::BTreeSet<_>>(),
        ["edge", "node"].into_iter().collect()
    );

    // Canonical SQLite events and jobs must survive a service restart before
    // the eventual Chroma projection is attempted.
    drop(service);
    let graph = kogwistar_store_chroma::ChromaGraphStore::connect_entity_collections(
        chroma_config("nodes"),
        chroma_config("edges"),
    )
    .await
    .unwrap();
    let service = SqliteRunApplicationService::open(&sqlite_path)
        .unwrap()
        .with_chroma_graph(graph)
        .with_embedding_provider(provider.clone());
    let recovered_events = service.store.replay_raw_events("tenant-a", 0, 10).unwrap();
    assert_eq!(recovered_events.len(), 2);
    for event in &recovered_events {
        let payload: Value = serde_json::from_str(&event.payload_json).unwrap();
        let expected_class = if payload["id"] == "node-1" {
            "Node"
        } else {
            "Edge"
        };
        assert_eq!(payload["metadata"]["_class_name"], expected_class);
    }
    let recovered_jobs = service
        .store
        .list_index_jobs(Some("tenant-a"), Some("PENDING"), None, None, None, 10)
        .unwrap();
    assert_eq!(recovered_jobs.len(), 2);

    let failed_projection = service
        .run_embedding_projection_jobs_once(10, 30, Some("tenant-a"))
        .await
        .unwrap();
    assert_eq!(failed_projection.claimed, 2);
    assert_eq!(failed_projection.projected, 1);
    assert_eq!(failed_projection.retried, 1);
    assert_eq!(
        service
            .store
            .replay_raw_events("tenant-a", 0, 10)
            .unwrap()
            .len(),
        2
    );
    let after_failure = service
        .store
        .list_index_jobs(Some("tenant-a"), None, None, None, None, 10)
        .unwrap();
    assert_eq!(
        after_failure
            .iter()
            .filter(|job| job.status == "DONE")
            .count(),
        1
    );
    assert_eq!(
        after_failure
            .iter()
            .filter(|job| job.status == "PENDING")
            .count(),
        1
    );

    // A Chroma failure cannot undo canonical SQLite state. Reopen the service
    // and retry only the durable failed projection job.
    drop(service);
    tokio::time::sleep(std::time::Duration::from_millis(2200)).await;
    let graph = kogwistar_store_chroma::ChromaGraphStore::connect_entity_collections(
        chroma_config("nodes"),
        chroma_config("edges"),
    )
    .await
    .unwrap();
    let service = SqliteRunApplicationService::open(&sqlite_path)
        .unwrap()
        .with_chroma_graph(graph)
        .with_embedding_provider(provider.clone());
    let trigger_connection = rusqlite::Connection::open(&sqlite_path).unwrap();
    trigger_connection
        .execute_batch(
            "CREATE TRIGGER reject_node_projection_ack
             BEFORE UPDATE OF status ON index_jobs
             WHEN NEW.entity_id = 'node-1' AND NEW.status = 'DONE'
             BEGIN SELECT RAISE(ABORT, 'injected projection acknowledgement failure'); END;",
        )
        .unwrap();
    let ack_failure = service
        .run_embedding_projection_jobs_once(10, 1, Some("tenant-a"))
        .await
        .unwrap_err();
    assert!(ack_failure.contains("injected projection acknowledgement failure"));
    let after_ack_failure = service
        .store
        .list_index_jobs(Some("tenant-a"), None, None, None, None, 10)
        .unwrap();
    let node_job = after_ack_failure
        .iter()
        .find(|job| job.entity_id == "node-1")
        .unwrap();
    assert_eq!(node_job.status, "DOING");
    assert!(node_job.accepted_result_json.is_some());
    assert_eq!(
        after_ack_failure
            .iter()
            .find(|job| job.entity_id == "edge-1")
            .unwrap()
            .status,
        "DONE"
    );

    drop(service);
    trigger_connection
        .execute_batch("DROP TRIGGER reject_node_projection_ack")
        .unwrap();
    drop(trigger_connection);
    tokio::time::sleep(std::time::Duration::from_millis(2200)).await;
    let graph = kogwistar_store_chroma::ChromaGraphStore::connect_entity_collections(
        chroma_config("nodes"),
        chroma_config("edges"),
    )
    .await
    .unwrap();
    let service = SqliteRunApplicationService::open(&sqlite_path)
        .unwrap()
        .with_chroma_graph(graph)
        .with_embedding_provider(provider);
    let final_projection = service
        .run_embedding_projection_jobs_once(10, 30, Some("tenant-a"))
        .await
        .unwrap();
    assert_eq!(final_projection.claimed, 1);
    assert_eq!(final_projection.projected, 0);
    assert_eq!(final_projection.reused, 1);
    assert_eq!(
        service
            .store
            .replay_raw_events("tenant-a", 0, 10)
            .unwrap()
            .len(),
        2
    );
    let projection_jobs = service
        .store
        .list_index_jobs(Some("tenant-a"), None, None, None, None, 10)
        .unwrap();
    assert_eq!(projection_jobs.len(), 2);
    assert!(projection_jobs.iter().all(|job| job.status == "DONE"));
    assert_eq!(embedding_calls.lock().unwrap().len(), 1);
    let _ = stop_chroma.send(());
    let chroma_upserts = fake_chroma.await.unwrap();
    assert_eq!(chroma_upserts.len(), 4);
    let node_attempts = chroma_upserts
        .iter()
        .filter(|(_, body)| body["ids"][0] == "node-1")
        .collect::<Vec<_>>();
    let edge_attempts = chroma_upserts
        .iter()
        .filter(|(_, body)| body["ids"][0] == "edge-1")
        .collect::<Vec<_>>();
    assert_eq!(node_attempts.len(), 3);
    assert_eq!(edge_attempts.len(), 1);
    assert_eq!(node_attempts[0].1, node_attempts[1].1);
    assert_eq!(node_attempts[1].1, node_attempts[2].1);
    let node_upsert = chroma_upserts
        .iter()
        .find(|(_, body)| body["ids"][0] == "node-1")
        .unwrap();
    assert!(node_upsert.0.contains("collection-nodes"));
    assert_eq!(node_upsert.1["embeddings"][0], json!([0.6, 0.8]));
    assert_eq!(node_upsert.1["metadatas"][0]["namespace"], "tenant-a");
    assert_eq!(node_upsert.1["metadatas"][0]["workspace_id"], "ws");
    assert_eq!(node_upsert.1["metadatas"][0]["graph_space"], "facts");
    assert_eq!(node_upsert.1["metadatas"][0]["_class_name"], "Node");
    let edge_upsert = chroma_upserts
        .iter()
        .find(|(_, body)| body["ids"][0] == "edge-1")
        .unwrap();
    assert!(edge_upsert.0.contains("collection-edges"));
    assert_eq!(edge_upsert.1["embeddings"][0], json!([0.6, 0.8]));
    assert_eq!(edge_upsert.1["metadatas"][0]["namespace"], "tenant-a");
    assert_eq!(edge_upsert.1["metadatas"][0]["workspace_id"], "ws");
    assert_eq!(edge_upsert.1["metadatas"][0]["graph_space"], "facts");
    assert_eq!(edge_upsert.1["metadatas"][0]["_class_name"], "Edge");

    drop(service);
    remove_test_sqlite(&sqlite_path);
}

#[tokio::test]
async fn sqlite_chroma_vector_query_enforces_source_span_acl() {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let vector_query_count = Arc::new(AtomicUsize::new(0));
    let vector_query_count_for_server = Arc::clone(&vector_query_count);
    let (stop_chroma, mut stop_chroma_rx) = tokio::sync::oneshot::channel();
    let fake_chroma = tokio::spawn(async move {
        loop {
            let (mut stream, _) = tokio::select! {
                _ = &mut stop_chroma_rx => break,
                connection = listener.accept() => connection.unwrap(),
            };
            let mut bytes = Vec::new();
            let mut chunk = [0_u8; 2048];
            let (header_end, content_length) = loop {
                let count = stream.read(&mut chunk).await.unwrap();
                assert!(count > 0, "Chroma connection closed before request body");
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
            let request_line = String::from_utf8_lossy(&bytes[..header_end])
                .lines()
                .next()
                .unwrap_or_default()
                .to_owned();
            let payload = if request_line.contains("/collections HTTP") {
                json!({
                    "id": format!("collection-{}", request["name"].as_str().unwrap()),
                    "metadata": {"hnsw:space":"cosine"}
                })
            } else if request_line.contains("collection-source-acl-nodes/query") {
                vector_query_count_for_server.fetch_add(1, Ordering::SeqCst);
                assert_eq!(request["n_results"], 1);
                let query_filter = request["where"].to_string();
                assert!(
                    query_filter.contains("\"entity_type\":{\"$ne\":\"acl_record\"}"),
                    "ACL helper exclusion must be pushed into vector top-k: {query_filter}"
                );
                assert!(
                    query_filter.contains("\"lifecycle_status\":{\"$ne\":\"tombstoned\"}"),
                    "tombstones must be excluded before vector top-k: {query_filter}"
                );
                assert!(
                    query_filter.contains("\"lifecycle_status\":{\"$ne\":\"deleted\"}"),
                    "legacy deleted records must be excluded before vector top-k: {query_filter}"
                );
                let revoked_query = request["where"].to_string().contains("node-revoked");
                let (node_id, original) = if revoked_query {
                    (
                        "node-revoked",
                        json!({"entity_type":"node","label":"node-revoked","mentions":[]}),
                    )
                } else {
                    (
                        "node-source-span-private",
                        json!({
                            "entity_type":"node",
                            "label":"node-source-span-private",
                            "mentions":[{"spans":[{}]}]
                        }),
                    )
                };
                json!({
                    "ids":[[node_id]],
                    "documents":[[node_id]],
                    "metadatas":[[{
                        "_kogwistar_namespace":"docs",
                        "_kogwistar_original_metadata":original.to_string()
                    }]],
                    "embeddings":[[[1.0,0.0]]],
                    "distances":[[0.0]]
                })
            } else if request_line.contains("collection-source-acl-edges/query") {
                vector_query_count_for_server.fetch_add(1, Ordering::SeqCst);
                assert_eq!(request["n_results"], 1);
                let original = json!({
                    "entity_type":"edge",
                    "source_ids":["private-endpoint"],
                    "target_ids":["public-endpoint"],
                    "source_edge_ids":[],
                    "target_edge_ids":[],
                    "mentions":[]
                });
                json!({
                    "ids":[["edge-with-private-endpoint"]],
                    "documents":[["edge-with-private-endpoint"]],
                    "metadatas":[[{
                        "_kogwistar_namespace":"docs",
                        "_kogwistar_original_metadata":original.to_string()
                    }]],
                    "embeddings":[[[1.0,0.0]]],
                    "distances":[[0.0]]
                })
            } else if request_line.contains("collection-source-acl-nodes/get") {
                if let Some(ids) = request.get("ids").and_then(Value::as_array) {
                    let nodes = [
                        (
                            "private-endpoint",
                            json!({
                                "entity_type":"node",
                                "label":"private-endpoint",
                                "mentions":[]
                            }),
                        ),
                        (
                            "public-endpoint",
                            json!({
                                "entity_type":"node",
                                "label":"public-endpoint",
                                "mentions":[]
                            }),
                        ),
                    ];
                    let matching = nodes
                        .iter()
                        .filter(|(id, _)| ids.iter().any(|requested| requested == id))
                        .collect::<Vec<_>>();
                    json!({
                        "ids":matching.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
                        "documents":matching.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
                        "metadatas":matching.iter().map(|(_, original)| json!({
                            "_kogwistar_namespace":"docs",
                            "_kogwistar_original_metadata":original.to_string()
                        })).collect::<Vec<_>>(),
                        "embeddings":vec![Value::Null; matching.len()]
                    })
                } else {
                let records = [
                    (
                        "acl-source-node",
                        json!({
                            "entity_type":"acl_record",
                            "acl_truth_graph":"knowledge",
                            "acl_target_entity_id":"node-source-span-private",
                            "acl_target_grain":"node",
                            "acl_version":1,
                            "acl_mode":"public"
                        }),
                    ),
                    (
                        "acl-source-grounding",
                        json!({
                            "entity_type":"acl_record",
                            "acl_truth_graph":"knowledge",
                            "acl_target_entity_id":"node-source-span-private",
                            "acl_target_grain":"grounding",
                            "acl_target_item_id":"node-source-span-private::mention::0",
                            "acl_version":1,
                            "acl_mode":"public"
                        }),
                    ),
                    (
                        "acl-source-span",
                        json!({
                            "entity_type":"acl_record",
                            "acl_truth_graph":"knowledge",
                            "acl_target_entity_id":"node-source-span-private",
                            "acl_target_grain":"span",
                            "acl_target_item_id":"node-source-span-private::mention::0::span::0",
                            "acl_version":1,
                            "acl_mode":"private",
                            "owner_id":"bob"
                        }),
                    ),
                    (
                        "acl-node-revoked",
                        json!({
                            "entity_type":"acl_record",
                            "acl_truth_graph":"knowledge",
                            "acl_target_entity_id":"node-revoked",
                            "acl_target_grain":"node",
                            "acl_version":1,
                            "acl_mode":"public"
                        }),
                    ),
                    (
                        "acl-edge-public",
                        json!({
                            "entity_type":"acl_record",
                            "acl_truth_graph":"knowledge",
                            "acl_target_entity_id":"edge-with-private-endpoint",
                            "acl_target_grain":"edge",
                            "acl_version":1,
                            "acl_mode":"public"
                        }),
                    ),
                    (
                        "acl-private-endpoint",
                        json!({
                            "entity_type":"acl_record",
                            "acl_truth_graph":"knowledge",
                            "acl_target_entity_id":"private-endpoint",
                            "acl_target_grain":"node",
                            "acl_version":1,
                            "acl_mode":"private",
                            "owner_id":"bob"
                        }),
                    ),
                    (
                        "acl-public-endpoint",
                        json!({
                            "entity_type":"acl_record",
                            "acl_truth_graph":"knowledge",
                            "acl_target_entity_id":"public-endpoint",
                            "acl_target_grain":"node",
                            "acl_version":1,
                            "acl_mode":"public"
                        }),
                    ),
                ];
                json!({
                    "ids": records.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
                    "documents": vec![Value::Null; records.len()],
                    "metadatas": records.iter().map(|(_, original)| json!({
                        "_kogwistar_namespace":"docs",
                        "_kogwistar_original_metadata":original.to_string()
                    })).collect::<Vec<_>>(),
                    "embeddings": vec![Value::Null; records.len()]
                })
                }
            } else {
                json!({"ids":[],"documents":[],"metadatas":[],"embeddings":[]})
            };
            let response = payload.to_string();
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
        chroma_config("source-acl-nodes"),
        chroma_config("source-acl-edges"),
    )
    .await
    .unwrap();
    let sqlite_path = std::env::temp_dir().join(format!(
        "kogwistar-sqlite-chroma-source-acl-{}.sqlite",
        uuid::Uuid::new_v4().simple()
    ));
    let service = SqliteRunApplicationService::open(&sqlite_path)
        .unwrap()
        .with_chroma_graph(graph);
    let mut canonical_acl_mutations = Vec::new();
    for (id, entity_id, grain, item_id, mode, owner_id) in [
        (
            "acl-source-node",
            "node-source-span-private",
            "node",
            None,
            "public",
            None,
        ),
        (
            "acl-source-grounding",
            "node-source-span-private",
            "grounding",
            Some("node-source-span-private::mention::0"),
            "public",
            None,
        ),
        (
            "acl-source-span",
            "node-source-span-private",
            "span",
            Some("node-source-span-private::mention::0::span::0"),
            "private",
            Some("bob"),
        ),
        (
            "acl-edge-public",
            "edge-with-private-endpoint",
            "edge",
            None,
            "public",
            None,
        ),
        (
            "acl-private-endpoint",
            "private-endpoint",
            "node",
            None,
            "private",
            Some("bob"),
        ),
        (
            "acl-public-endpoint",
            "public-endpoint",
            "node",
            None,
            "public",
            None,
        ),
        (
            "acl-node-revoked",
            "node-revoked",
            "node",
            None,
            "public",
            None,
        ),
    ] {
        let mut metadata = json!({
            "entity_type":"acl_record",
            "acl_truth_graph":"knowledge",
            "acl_target_entity_id":entity_id,
            "acl_target_grain":grain,
            "acl_version":1,
            "acl_mode":mode
        });
        if let Some(item_id) = item_id {
            metadata["acl_target_item_id"] = json!(item_id);
        }
        if let Some(owner_id) = owner_id {
            metadata["owner_id"] = json!(owner_id);
        }
        canonical_acl_mutations.push(json!({
            "scope":{"namespace":"docs","workspace_id":null,"graph_space":null},
            "table":"gke_nodes",
            "entity_kind":"node",
            "event_id":format!("seed-{id}"),
            "op":"ADD",
            "payload":{"id":id,"document":null,"metadata":metadata.clone()},
            "record":{"id":id,"document":null,"metadata":metadata,"embedding":null},
            "embedding_dim":0
        }));
    }
    let seed_acl = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/internal/graph/mutations".to_owned(),
            body: serde_json::to_vec(&json!({
                "contract_version":1,
                "atomic":true,
                "mutations":canonical_acl_mutations,
                "index_jobs":[]
            }))
            .unwrap(),
            principal: json!({
                "sub":"owner",
                "role":"rw",
                "storage_ns":"docs",
                "capabilities":["write_graph"]
            }),
        })
        .await;
    assert_eq!(seed_acl.status, StatusCode::OK.as_u16());
    let response = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/internal/graph/vector-query".to_owned(),
            body: serde_json::to_vec(&json!({
                "entity_kind":"node",
                "query_embeddings":[[1.0,0.0]],
                "metric":"cosine",
                "truth_graph":"knowledge",
                "where":{"label":"node-source-span-private"},
                "n_results":1,
                "include":["distances"]
            }))
            .unwrap(),
            principal: json!({
                "sub":"alice",
                "role":"ro",
                "storage_ns":"docs",
                "capabilities":["read_graph"]
            }),
        })
        .await;
    assert_eq!(response.status, StatusCode::OK.as_u16());
    let result: Value = serde_json::from_slice(&response.body).unwrap();
    assert_eq!(result["ids"], json!([[]]));
    assert_eq!(vector_query_count.load(Ordering::SeqCst), 1);

    let edge_response = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/internal/graph/vector-query".to_owned(),
            body: serde_json::to_vec(&json!({
                "entity_kind":"edge",
                "query_embeddings":[[1.0,0.0]],
                "metric":"cosine",
                "truth_graph":"knowledge",
                "n_results":1,
                "include":["distances"]
            }))
            .unwrap(),
            principal: json!({
                "sub":"alice",
                "role":"ro",
                "storage_ns":"docs",
                "capabilities":["read_graph"]
            }),
        })
        .await;
    assert_eq!(edge_response.status, StatusCode::OK.as_u16());
    let edge_result: Value = serde_json::from_slice(&edge_response.body).unwrap();
    assert_eq!(edge_result["ids"], json!([[]]));
    assert_eq!(vector_query_count.load(Ordering::SeqCst), 2);

    let revoked_query = || ApiEffectRequest {
        contract_version: 1,
        method: "POST".to_owned(),
        path_and_query: "/internal/graph/vector-query".to_owned(),
        body: serde_json::to_vec(&json!({
            "entity_kind":"node",
            "query_embeddings":[[1.0,0.0]],
            "metric":"cosine",
            "truth_graph":"knowledge",
            "where":{"label":"node-revoked"},
            "n_results":1,
            "include":["distances"]
        }))
        .unwrap(),
        principal: json!({
            "sub":"alice",
            "role":"ro",
            "storage_ns":"docs",
            "capabilities":["read_graph"]
        }),
    };
    let public_before_revoke = service.execute(revoked_query()).await;
    assert_eq!(public_before_revoke.status, StatusCode::OK.as_u16());
    let public_before_revoke: Value = serde_json::from_slice(&public_before_revoke.body).unwrap();
    assert_eq!(public_before_revoke["ids"], json!([["node-revoked"]]));
    assert_eq!(vector_query_count.load(Ordering::SeqCst), 3);

    let revoke = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/internal/graph/mutations".to_owned(),
            body: serde_json::to_vec(&json!({
                "contract_version":1,
                "atomic":true,
                "index_jobs":[],
                "mutations":[{
                    "scope":{"namespace":"docs","workspace_id":null,"graph_space":null},
                    "table":"gke_nodes",
                    "entity_kind":"node",
                    "event_id":"revoke-acl-node-revoked-v2",
                    "op":"REPLACE",
                    "payload":{
                        "id":"acl-node-revoked",
                        "document":null,
                        "metadata":{
                            "entity_type":"acl_record",
                            "acl_truth_graph":"knowledge",
                            "acl_target_entity_id":"node-revoked",
                            "acl_target_grain":"node",
                            "acl_version":2,
                            "acl_mode":"private",
                            "owner_id":"bob"
                        }
                    },
                    "record":{
                        "id":"acl-node-revoked",
                        "document":null,
                        "metadata":{
                            "entity_type":"acl_record",
                            "acl_truth_graph":"knowledge",
                            "acl_target_entity_id":"node-revoked",
                            "acl_target_grain":"node",
                            "acl_version":2,
                            "acl_mode":"private",
                            "owner_id":"bob"
                        },
                        "embedding":null
                    },
                    "embedding_dim":0
                }]
            }))
            .unwrap(),
            principal: json!({
                "sub":"owner",
                "role":"rw",
                "storage_ns":"docs",
                "capabilities":["write_graph"]
            }),
        })
        .await;
    assert_eq!(revoke.status, StatusCode::OK.as_u16());

    let hidden_after_revoke = service.execute(revoked_query()).await;
    assert_eq!(hidden_after_revoke.status, StatusCode::OK.as_u16());
    let hidden_after_revoke: Value = serde_json::from_slice(&hidden_after_revoke.body).unwrap();
    assert_eq!(hidden_after_revoke["ids"], json!([[]]));
    assert_eq!(vector_query_count.load(Ordering::SeqCst), 4);

    drop(service);
    let _ = stop_chroma.send(());
    fake_chroma.await.unwrap();
    remove_test_sqlite(&sqlite_path);
}

#[tokio::test]
async fn sqlite_chroma_conversation_create_and_turn_admission_are_canonical_and_atomic() {
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use kogwistar_store::{
        DistanceMetric, GraphMutation, GraphMutationBatch, GraphReadStore, GraphRecord,
        GraphScope, GraphWriteStore, MetadataFilter, NamedProjectionWrite, VectorQuery,
        WorkflowDesignSnapshotWrite,
    };
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tower::ServiceExt;

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let chat_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let chat_endpoint = format!(
        "http://{}/v1/chat/completions",
        chat_listener.local_addr().unwrap()
    );
    let chat_server = tokio::spawn(async move {
        let (mut stream, _) = chat_listener.accept().await.unwrap();
        let mut bytes = Vec::new();
        let (header_end, content_length) = loop {
            let mut chunk = [0_u8; 2048];
            let count = stream.read(&mut chunk).await.unwrap();
            assert!(count > 0);
            bytes.extend_from_slice(&chunk[..count]);
            if let Some(end) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
                let headers = String::from_utf8_lossy(&bytes[..end]);
                let length = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().ok())
                            .flatten()
                    })
                    .unwrap_or(0);
                if bytes.len() >= end + 4 + length {
                    break (end, length);
                }
            }
        };
        let request: Value =
            serde_json::from_slice(&bytes[header_end + 4..header_end + 4 + content_length])
                .unwrap();
        assert_eq!(request["stream"], true);
        assert_eq!(request["response_format"]["type"], "json_schema");
        let prompt = request["messages"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|message| message["content"].as_str())
            .collect::<Vec<_>>()
            .join("\n");
        let visible_alias = prompt
            .lines()
            .find_map(|line| line.strip_prefix("NODE "))
            .and_then(|line| line.split_once(" |"))
            .map(|(alias, _)| alias.to_owned())
            .expect("provider prompt contains an aliased retrieved evidence node");
        let content = json!({
            "text":"The runtime persists workflow state in the graph.",
            "reasoning":"The retrieved source supports this statement.",
            "claims":[{
                "claim":"Workflow state is persisted in the graph.",
                "citations":[{"source_node_id":visible_alias,"mention_index":0,"span_index":0}]
            }]
        })
        .to_string();
        let frames = [
                format!("data: {}\r\n\r\n", json!({"id":"sqlite-chroma-answer","choices":[{"delta":{"content":content},"finish_reason":null}]})),
                format!("data: {}\r\n\r\n", json!({"id":"sqlite-chroma-answer","choices":[{"delta":{},"finish_reason":"stop"}]})),
                "data: [DONE]\r\n\r\n".to_owned(),
            ].concat();
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
            frames.len(),
            frames
        );
        stream.write_all(response.as_bytes()).await.unwrap();
    });
    let (stop_tx, mut stop_rx) = tokio::sync::oneshot::channel();
    let fake_chroma = tokio::spawn(async move {
        let mut projection_requests = Vec::new();
        loop {
            let (mut stream, _) = tokio::select! {
                _ = &mut stop_rx => break,
                accepted = listener.accept() => accepted.unwrap(),
            };
            let mut bytes = Vec::new();
            let mut chunk = [0_u8; 2048];
            let (header_end, content_length) = loop {
                let count = stream.read(&mut chunk).await.unwrap();
                assert!(count > 0, "Chroma closed before receiving request body");
                bytes.extend_from_slice(&chunk[..count]);
                if let Some(header_end) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
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
            let request_line = String::from_utf8_lossy(&bytes[..header_end])
                .lines()
                .next()
                .unwrap_or_default()
                .to_owned();
            let body: Value =
                serde_json::from_slice(&bytes[header_end + 4..header_end + 4 + content_length])
                    .unwrap();
            let response = if request_line.contains("/collections HTTP") {
                json!({"id":"collection-id", "metadata":{"hnsw:space":"cosine"}})
            } else if request_line.contains("/query HTTP") {
                json!({
                    "ids": [["knowledge-node"]],
                    "documents": [["The runtime persists workflow state in the graph."]],
                    "metadatas": [[{
                        "namespace":"conversation",
                        "workspace_id":"workspace-a",
                        "graph_space":"conversation",
                        "entity_type":"node",
                        "label":"Workflow runtime state",
                        "summary":"The runtime persists workflow state in the graph.",
                        "level_from_root":0,
                        "lifecycle_status":"active",
                        "_kogwistar_stage2_ready":true,
                        "_kogwistar_source_revision":1,
                        "_class_name":"Node"
                    }]],
                    "embeddings": [[[0.6, 0.8]]],
                    "distances": [[0.0]]
                })
            } else if request_line.contains("/get HTTP") && body.to_string().contains("acl_record")
            {
                json!({
                    "ids": ["knowledge-node-acl"],
                    "documents": ["{}"],
                    "metadatas": [{
                        "namespace":"conversation",
                        "workspace_id":"workspace-a",
                        "graph_space":"conversation",
                        "entity_type":"acl_record",
                        "acl_truth_graph":"knowledge",
                        "acl_target_entity_id":"knowledge-node",
                        "acl_target_grain":"node",
                        "acl_version":1,
                        "acl_mode":"public",
                        "_kogwistar_stage2_ready":true,
                        "_kogwistar_source_revision":2,
                        "_class_name":"Node"
                    }]
                })
            } else if request_line.contains("/get HTTP") {
                json!({
                    "ids": ["knowledge-node"],
                    "documents": ["The runtime persists workflow state in the graph."],
                    "metadatas": [{
                        "namespace":"conversation",
                        "workspace_id":"workspace-a",
                        "graph_space":"conversation",
                        "entity_type":"node",
                        "label":"Workflow runtime state",
                        "summary":"The runtime persists workflow state in the graph.",
                        "level_from_root":0,
                        "lifecycle_status":"active",
                        "_kogwistar_stage2_ready":true,
                        "_kogwistar_source_revision":1,
                        "_class_name":"Node"
                    }]
                })
            } else {
                if request_line.contains("/upsert") || request_line.contains("/delete") {
                    projection_requests.push(request_line);
                }
                json!({})
            };
            let response = response.to_string();
            let wire = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                response.len(),
                response
            );
            stream.write_all(wire.as_bytes()).await.unwrap();
        }
        projection_requests
    });
    let live_chroma_endpoint = std::env::var("KOGWISTAR_TEST_CHROMA_URL").ok();
    let chroma_suffix = uuid::Uuid::new_v4().simple().to_string();
    let config = |entity_kind: &str| kogwistar_store_chroma::ChromaStoreConfig {
        endpoint: live_chroma_endpoint
            .clone()
            .unwrap_or_else(|| endpoint.clone()),
        collection: format!("conversation-{entity_kind}-{chroma_suffix}"),
        ..Default::default()
    };
    let graph = kogwistar_store_chroma::ChromaGraphStore::connect_entity_collections(
        config("nodes"),
        config("edges"),
    )
    .await
    .unwrap();
    if live_chroma_endpoint.is_some() {
        graph
            .upsert_graph_record_of_kind(
                "conversation",
                kogwistar_store::GraphEntityKind::Node,
                GraphRecord {
                    id: "knowledge-node".to_owned(),
                    document: Some(
                        "The runtime persists workflow state in the graph.".to_owned(),
                    ),
                    metadata: serde_json::from_value(json!({
                        "entity_type": "node",
                        "workspace_id": "workspace-a",
                        "graph_space": "conversation",
                        "label": "Workflow runtime state",
                        "summary": "The runtime persists workflow state in the graph.",
                        "level_from_root": 0,
                        "lifecycle_status": "active"
                    }))
                    .unwrap(),
                    embedding: Some(vec![0.6, 0.8]),
                },
            )
            .await
            .unwrap();
    }
    let path = std::env::temp_dir().join(format!(
        "kogwistar-sqlite-conversation-create-{}.sqlite",
        uuid::Uuid::new_v4().simple()
    ));
    let auth_path = std::env::temp_dir().join(format!(
        "kogwistar-sqlite-conversation-owner-auth-{}.sqlite",
        uuid::Uuid::new_v4().simple()
    ));
    let auth = SqliteAuthStore::open(&auth_path).unwrap();
    auth.resolve_external_identity(ResolveExternalIdentity {
        issuer: "test".to_owned(),
        subject: "conversation-owner".to_owned(),
        email: "conversation-owner@example.test".to_owned(),
        display_name: None,
        new_user_id: "conversation-owner".to_owned(),
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
    struct TestEmbeddingProvider {
        profile: kogwistar_engine::embedding::EmbeddingProfile,
    }
    impl kogwistar_engine::embedding::EmbeddingProvider for TestEmbeddingProvider {
        fn profile(&self) -> &kogwistar_engine::embedding::EmbeddingProfile {
            &self.profile
        }

        fn embed<'a>(
            &'a self,
            texts: &'a [String],
        ) -> kogwistar_engine::embedding::EmbeddingFuture<'a> {
            Box::pin(async move { Ok(texts.iter().map(|_| vec![0.6, 0.8]).collect()) })
        }
    }
    let embedding_provider = Arc::new(TestEmbeddingProvider {
        profile: kogwistar_engine::embedding::EmbeddingProfile {
            provider: "test".to_owned(),
            model: "conversation-test".to_owned(),
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
        },
    });
    let service = SqliteRunApplicationService::open(&path)
        .unwrap()
        .with_chroma_graph(graph.clone())
        .with_embedding_provider(embedding_provider.clone())
        .with_chat_completion_provider(
            crate::native_answer_generation::OpenAiCompatibleChatConfig::new(
                chat_endpoint.clone(),
                "sqlite-chroma-fixture-model",
                None,
            )
            .unwrap()
            .with_streaming(true),
        )
        .with_owner_auth_config(&owner_auth)
        .await
        .unwrap();

    let mut knowledge_payload: Value = serde_json::from_str::<Value>(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/python-graph-model-payloads-v1.json"
    )))
    .unwrap()["cases"][0]["payload"]
        .clone();
    knowledge_payload["id"] = json!("knowledge-node");
    knowledge_payload["label"] = json!("Workflow runtime state");
    knowledge_payload["summary"] = json!("The runtime persists workflow state in the graph.");
    knowledge_payload["doc_id"] = json!("doc:runtime-state");
    knowledge_payload["embedding"] = Value::Null;
    knowledge_payload["metadata"]["workspace_id"] = json!("workspace-a");
    knowledge_payload["metadata"]["graph_space"] = json!("conversation");
    knowledge_payload["metadata"]["level_from_root"] = json!(0);
    let mut acl_payload = knowledge_payload.clone();
    acl_payload["id"] = json!("knowledge-node-acl");
    acl_payload["label"] = json!("ACL for knowledge-node");
    acl_payload["summary"] = json!("Public node access");
    acl_payload["metadata"]["entity_type"] = json!("acl_record");
    acl_payload["metadata"]["acl_truth_graph"] = json!("knowledge");
    acl_payload["metadata"]["acl_target_entity_id"] = json!("knowledge-node");
    acl_payload["metadata"]["acl_target_grain"] = json!("node");
    acl_payload["metadata"]["acl_version"] = json!(1);
    acl_payload["metadata"]["acl_mode"] = json!("public");
    if live_chroma_endpoint.is_some() {
        graph
            .upsert_graph_record_of_kind(
                "conversation",
                kogwistar_store::GraphEntityKind::Node,
                GraphRecord {
                    id: acl_payload["id"].as_str().unwrap().to_owned(),
                    document: Some(acl_payload["summary"].as_str().unwrap().to_owned()),
                    metadata: serde_json::from_value(acl_payload["metadata"].clone()).unwrap(),
                    embedding: Some(vec![0.6, 0.8]),
                },
            )
            .await
            .unwrap();
    }
    let graph_scope = GraphScope {
        namespace: "conversation".to_owned(),
        workspace_id: Some("workspace-a".to_owned()),
        graph_space: Some("conversation".to_owned()),
    };
    let mutation = |event_id: &str, payload: Value| {
        let id = payload["id"].as_str().unwrap().to_owned();
        let metadata = payload["metadata"].as_object().unwrap().clone();
        GraphMutation {
            scope: graph_scope.clone(),
            table: "gke_nodes".to_owned(),
            entity_kind: "node".to_owned(),
            event_id: event_id.to_owned(),
            op: "ADD".to_owned(),
            payload: payload.clone(),
            record: GraphRecord {
                id,
                document: Some(payload.to_string()),
                metadata,
                embedding: None,
            },
            embedding_dim: 2,
        }
    };
    service
        .store
        .apply_canonical_graph_mutation_batch(GraphMutationBatch {
            mutations: vec![
                mutation("knowledge-node-event", knowledge_payload),
                mutation("knowledge-node-acl-event", acl_payload),
            ],
            index_jobs: Vec::new(),
        })
        .unwrap();
    if live_chroma_endpoint.is_some() {
        let mut metadata = MetadataFilter::default();
        metadata
            .equals
            .insert("workspace_id".to_owned(), json!("workspace-a"));
        metadata
            .equals
            .insert("graph_space".to_owned(), json!("conversation"));
        let hits = graph
            .vector_query_of_kind_numeric_lte(
                "conversation",
                kogwistar_store::GraphEntityKind::Node,
                &VectorQuery {
                    embedding: vec![0.6, 0.8],
                    limit: 20,
                    metadata,
                    metric: DistanceMetric::Cosine,
                },
                "level_from_root",
                4.0,
            )
            .await
            .unwrap();
        assert!(
            hits.iter().any(|matched| matched.record.id == "knowledge-node"),
            "canonical SQLite admission must not remove the seeded Chroma retrieval hit"
        );
    }

    let workflow_contract: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/answer-workflow-runtime-contract-v1.json"
    )))
    .unwrap();
    let workflow_id = "debug.rag.v1";
    let workflow = &workflow_contract["workflows"][workflow_id];
    let nodes = workflow["required_ops"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| {
            let op = value.as_str().unwrap();
            json!({
                "id": op,
                "label": op,
                "metadata": {
                    "entity_type": "workflow_node",
                    "workflow_id": workflow_id,
                    "wf_op": op,
                    "wf_version": "v1",
                    "wf_executor_target": "rust-native",
                    "wf_start": op == "start"
                }
            })
        })
        .collect::<Vec<_>>();
    let edges = workflow["required_routes"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(index, route)| {
            json!({
                "id": format!("debug-rag-route-{index}"),
                "source_ids": [route["from"]],
                "target_ids": [route["to"]],
                "metadata": {
                    "entity_type": "workflow_edge",
                    "workflow_id": workflow_id,
                    "wf_predicate": route["predicate"],
                    "wf_priority": route["priority"],
                    "wf_is_default": route["default"],
                    "wf_multiplicity": route["multiplicity"]
                }
            })
        })
        .collect::<Vec<_>>();
    service
        .store
        .replace_named_projection(
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
        .unwrap();
    service
        .store
        .put_workflow_design_snapshot(
            workflow_id,
            WorkflowDesignSnapshotWrite {
                version: 1,
                seq: 1,
                schema_version: 1,
                payload_json: serde_json::to_string(&json!({ "nodes": nodes, "edges": edges }))
                    .unwrap(),
            },
        )
        .unwrap();
    assert!(service.supports_http_operation("POST", "/api/conversations"));
    assert!(service.supports_http_operation(
        "POST",
        "/api/conversations/conversation-create-contract/turns:answer"
    ));
    assert!(service.supports_mcp_tool("conversation", "conversation.ask"));

    let principal = json!({
        "sub":"conversation-owner", "user_id":"conversation-owner", "role":"rw",
        "ns":["conversation"], "storage_ns":"conversation",
        "workspace":"workspace-a", "graph_space":"conversation",
        "capabilities":["spawn_process", "send_message", "write_graph", "read_graph", "workflow.run.write"]
    });
    let http_secret = "sqlite-chroma-public-ingress-test-secret";
    let token = jsonwebtoken::encode(
        &jsonwebtoken::Header::new(jsonwebtoken::Algorithm::HS256),
        &json!({
            "sub":"conversation-owner", "user_id":"conversation-owner",
            "actor_id":"conversation-owner", "role":"rw", "roles":["rw"],
            "ns":"conversation", "storage_ns":"conversation",
            "workspace":"workspace-a", "graph_space":"conversation",
            "capabilities":["spawn_process", "send_message", "write_graph", "read_graph", "workflow.run.write"],
            "exp":4102444800_u64
        }),
        &jsonwebtoken::EncodingKey::from_secret(http_secret.as_bytes()),
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
                key: Some(http_secret.to_owned()),
                auth_db_url: Some(format!("sqlite:///{}", auth_path.display())),
                ..AuthConfig::default()
            },
        },
        Arc::new(service.clone()),
    );
    let request = || ApiEffectRequest {
        contract_version: 1,
        method: "POST".to_owned(),
        path_and_query: "/api/conversations".to_owned(),
        body: serde_json::to_vec(&json!({
            "user_id":"conversation-owner",
            "conversation_id":"conversation-create-contract",
            "start_node_id":"conversation-create-start"
        }))
        .unwrap(),
        principal: principal.clone(),
    };
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/conversations")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::from(
                    serde_json::to_vec(&json!({
                        "user_id":"conversation-owner",
                        "conversation_id":"conversation-create-contract",
                        "start_node_id":"conversation-create-start"
                    }))
                    .unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    let response_status = response.status();
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), 65_536).await.unwrap()).unwrap();
    assert_eq!(response_status, StatusCode::OK, "{body}");
    assert_eq!(body["conversation_id"], "conversation-create-contract");
    assert_eq!(body["start_node_id"], "conversation-create-start");
    assert_eq!(body["turn_count"], 0);

    let events = service
        .store
        .replay_raw_events("conversation", 0, 20)
        .unwrap();
    assert!(events.iter().any(|event| {
        event.entity_kind == "node"
            && event.entity_id == "conversation-create-start"
            && serde_json::from_str::<Value>(&event.payload_json)
                .ok()
                .is_some_and(|payload| {
                    payload["metadata"]["entity_type"] == "conversation_start"
                        && payload["metadata"]["conversation_id"] == "conversation-create-contract"
                })
    }));

    let snapshot = service
        .chroma_graph_snapshot(&request(), "conversation")
        .await
        .unwrap();
    assert!(
        snapshot
            .nodes()
            .iter()
            .any(|record| record.id == "conversation-create-start"),
        "newly created conversation start must be visible in the scoped canonical snapshot; stage1={:?}",
        service
            .store
            .list_stage1_node_projections("conversation")
            .unwrap()
    );
    let visible_start = snapshot
        .nodes()
        .iter()
        .find(|record| record.id == "conversation-create-start")
        .unwrap();
    assert_eq!(
        visible_start.metadata.get("conversation_id"),
        Some(&json!("conversation-create-contract")),
        "snapshot metadata={:?}, document={:?}",
        visible_start.metadata,
        visible_start.document
    );

    let event_count_before_empty_answer = service
        .store
        .replay_raw_events("conversation", 0, 50)
        .unwrap()
        .len();
    let validation_contract: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/conversation-answer-validation-v1.json"
    )))
    .unwrap();
    assert_eq!(validation_contract["schema_version"], 1);
    for case in validation_contract["cases"].as_array().unwrap() {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/conversations/conversation-create-contract/turns:answer")
                    .header("content-type", "application/json")
                    .header("authorization", format!("Bearer {token}"))
                    .body(Body::from(
                        json!({"text":case["text"]}).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            response.status().as_u16(),
            case["http_status"].as_u64().unwrap() as u16,
            "case {}",
            case["name"]
        );
    }
    assert_eq!(
        service
            .store
            .replay_raw_events("conversation", 0, 50)
            .unwrap()
            .len(),
        event_count_before_empty_answer,
        "invalid REST turn admission must not append canonical graph events"
    );

    let send_message = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/conversations/conversation-create-contract/turns:answer")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::from(
                    serde_json::to_vec(&json!({
                        "text":"Where is workflow runtime state persisted?",
                        "user_id":"body-spoof",
                        "workflow_id":workflow_id
                    }))
                    .unwrap(),
                ))
                .unwrap(),
        )
        .await;
    let send_message = send_message.unwrap();
    assert_eq!(send_message.status(), StatusCode::ACCEPTED);
    let send_response: Value =
        serde_json::from_slice(&to_bytes(send_message.into_body(), 65_536).await.unwrap()).unwrap();
    assert_eq!(send_response.as_object().unwrap().len(), 5);
    assert_eq!(
        send_response["conversation_id"],
        "conversation-create-contract"
    );
    assert_eq!(send_response["workflow_id"], workflow_id);
    assert_eq!(send_response["status"], "queued");
    let run_id = send_response["run_id"].as_str().unwrap();
    assert!(!run_id.is_empty());
    assert!(
        send_response["user_turn_node_id"]
            .as_str()
            .is_some_and(|id| !id.is_empty())
    );
    assert_eq!(
        service
            .store
            .get_server_run(run_id)
            .unwrap()
            .expect("admitted answer run is durable")
            .user_id
            .as_deref(),
        Some("conversation-owner"),
        "authenticated owner takes precedence over body user_id"
    );
    let events = service
        .store
        .replay_raw_events("conversation", 0, 50)
        .unwrap();
    assert!(events.iter().any(|event| {
        event.entity_kind == "node"
            && serde_json::from_str::<Value>(&event.payload_json)
                .ok()
                .is_some_and(|payload| {
                    payload["metadata"]["entity_type"] == "conversation_turn"
                        && payload["summary"] == "Where is workflow runtime state persisted?"
                })
    }));
    assert!(events.iter().any(|event| {
        event.entity_kind == "edge"
            && serde_json::from_str::<Value>(&event.payload_json)
                .ok()
                .is_some_and(|payload| payload["relation"] == "next_turn")
    }));
    assert!(events.iter().any(|event| {
        serde_json::from_str::<Value>(&event.payload_json)
            .ok()
            .is_some_and(|payload| payload["metadata"]["entity_type"] == "acl_record")
    }));

    let pending_messages = service
        .store
        .list_projected_lane_messages(kogwistar_store::LaneMessageFilter {
            correlation_id: Some(run_id.to_owned()),
            limit: 10,
            ..Default::default()
        })
        .unwrap();
    assert!(
        pending_messages.iter().any(
            |message| message.msg_type == "workflow.run.execute" && message.status == "pending"
        )
    );
    drop(service);

    let recovered_graph = kogwistar_store_chroma::ChromaGraphStore::connect_entity_collections(
        config("nodes"),
        config("edges"),
    )
    .await
    .unwrap();
    let service = SqliteRunApplicationService::open(&path)
        .unwrap()
        .with_chroma_graph(recovered_graph)
        .with_embedding_provider(embedding_provider)
        .with_chat_completion_provider(
            crate::native_answer_generation::OpenAiCompatibleChatConfig::new(
                chat_endpoint,
                "sqlite-chroma-fixture-model",
                None,
            )
            .unwrap()
            .with_streaming(true),
        )
        .with_owner_auth_config(&owner_auth)
        .await
        .unwrap();
    assert!(
        service
            .store
            .replay_raw_events("conversation", 0, 50)
            .unwrap()
            .iter()
            .any(|event| event.entity_kind == "node"
                && serde_json::from_str::<Value>(&event.payload_json)
                    .ok()
                    .is_some_and(|payload| payload["metadata"]["entity_type"]
                        == "conversation_turn"
                        && payload["metadata"]["conversation_id"]
                            == "conversation-create-contract"))
    );

    for _ in 0..32 {
        let status = service
            .store
            .get_server_run(run_id)
            .unwrap()
            .expect("admitted answer run remains durable");
        if matches!(status.status.as_str(), "succeeded" | "failed" | "cancelled") {
            break;
        }
        assert!(
            service.execute_inline_native_work_once().await.unwrap(),
            "native scheduler stopped before the answer run became terminal: {status:#?}"
        );
    }
    let completed_run = service
        .store
        .get_server_run(run_id)
        .unwrap()
        .expect("completed answer run remains durable");
    assert_eq!(completed_run.status, "succeeded");
    let runtime = service
        .store
        .read_recorded_runtime_state(run_id, workflow_id, "conversation-create-contract")
        .unwrap()
        .expect("successful run has durable runtime state");
    assert_eq!(
        runtime.state["_rt_graph_scope"]["workspace"], "workspace-a",
        "the native workflow must retain its admitted workspace scope"
    );
    assert_eq!(runtime.state["done"], true);
    assert_eq!(runtime.state["answer"]["mode"], "debug_rag");
    assert_eq!(runtime.state["used_node_ids"], json!(["knowledge-node"]));
    assert!(
        runtime.state["answer"]["text"]
            .as_str()
            .is_some_and(|text| text.contains("knowledge-node"))
    );
    let projected_pointer = runtime.state["projected_pointer_ids"][0]
        .as_str()
        .expect("selected evidence receives a durable conversation pointer")
        .to_owned();
    assert!(
        service
            .store
            .latest_entity_events("conversation", "node", &[projected_pointer.clone()])
            .unwrap()
            .contains_key(&projected_pointer)
    );
    let assistant_id = runtime.state["agentic_answering_result"]["assistant_turn_node_id"]
        .as_str()
        .expect("answer workflow records the assistant turn")
        .to_owned();
    let assistant = service
        .store
        .latest_entity_events("conversation", "node", &[assistant_id.clone()])
        .unwrap();
    assert!(
        assistant.contains_key(&assistant_id),
        "assistant turn is canonical"
    );
    let assistant_contract: Value = serde_json::from_str(include_str!(
        "../../../../../contracts/golden/assistant-turn-graph-v1.json"
    ))
    .unwrap();
    let assistant_payload: Value =
        serde_json::from_str(&assistant[&assistant_id].payload_json).unwrap();
    for field in assistant_contract["assistant_node"]["required_fields"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(Value::as_str)
    {
        assert!(assistant_payload.get(field).is_some(), "missing assistant field {field}");
    }
    for (path, expected) in assistant_contract["assistant_node"]["field_values"]
        .as_object()
        .unwrap()
    {
        let actual = path
            .split('.')
            .fold(Some(&assistant_payload), |value, key| {
                value.and_then(|value| value.get(key))
            });
        assert_eq!(actual, Some(expected), "assistant projection mismatch at {path}");
    }
    let edges = service
        .store
        .replay_raw_events("conversation", 0, 100)
        .unwrap();
    assert!(
        edges.iter().any(|event| {
            event.entity_kind == "edge"
                && serde_json::from_str::<Value>(&event.payload_json)
                    .ok()
                    .is_some_and(|payload| {
                        payload["relation"] == "next_turn"
                            && payload["source_ids"] == json!([runtime.state["turn_node_id"]])
                            && payload["target_ids"] == json!([assistant_id])
                    })
        }),
        "assistant turn {assistant_id} is linked to user turn {} by the canonical graph edge",
        runtime.state["turn_node_id"]
    );
    let generated_edge = edges.iter().find_map(|event| {
        (event.entity_kind == "edge")
            .then(|| serde_json::from_str::<Value>(&event.payload_json).ok())
            .flatten()
            .filter(|payload| {
                payload["relation"] == "generated"
                    && payload["target_ids"] == json!([assistant_id])
            })
    });
    let generated_edge = generated_edge.expect("run-to-assistant generated edge is canonical");
    for field in assistant_contract["generated_edge"]["required_fields"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(Value::as_str)
    {
        assert!(generated_edge.get(field).is_some(), "missing generated edge field {field}");
    }
    for (path, expected) in assistant_contract["generated_edge"]["field_values"]
        .as_object()
        .unwrap()
    {
        let actual = path
            .split('.')
            .fold(Some(&generated_edge), |value, key| {
                value.and_then(|value| value.get(key))
            });
        assert_eq!(actual, Some(expected), "generated edge projection mismatch at {path}");
    }

    let mcp_ask = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp/conversation")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::from(
                    serde_json::to_vec(&json!({
                        "jsonrpc":"2.0",
                        "id":"sqlite-chroma-conversation-ask",
                        "method":"tools/call",
                        "params":{
                            "name":"conversation.ask",
                            "arguments":{
                                "conversation_id":"conversation-create-contract",
                                "text":"Does MCP use the same durable turn path?",
                                "workflow_id":workflow_id
                            }
                        }
                    }))
                    .unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(mcp_ask.status(), StatusCode::OK);
    let mcp_response: Value =
        serde_json::from_slice(&to_bytes(mcp_ask.into_body(), 65_536).await.unwrap()).unwrap();
    assert_eq!(mcp_response["result"]["status"], "queued");
    let mcp_run_id = mcp_response["result"]["run_id"]
        .as_str()
        .expect("MCP ask returns its queued run id");
    for _ in 0..32 {
        let status = service
            .store
            .get_server_run(mcp_run_id)
            .unwrap()
            .expect("MCP ask run remains durable");
        if matches!(status.status.as_str(), "succeeded" | "failed" | "cancelled") {
            break;
        }
        assert!(
            service.execute_inline_native_work_once().await.unwrap(),
            "native scheduler stopped before MCP ask became terminal: {status:#?}"
        );
    }
    assert_eq!(
        service
            .store
            .get_server_run(mcp_run_id)
            .unwrap()
            .expect("MCP ask run remains durable")
            .status,
        "succeeded"
    );
    let mcp_runtime = service
        .store
        .read_recorded_runtime_state(mcp_run_id, workflow_id, "conversation-create-contract")
        .unwrap()
        .expect("completed MCP run has durable runtime state");
    assert_eq!(mcp_runtime.state["turn_index"], 2);

    let run_status = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/api/workflow/runs/{run_id}"))
                .header("authorization", format!("Bearer {token}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(run_status.status(), StatusCode::OK);
    let run_status_body: Value =
        serde_json::from_slice(&to_bytes(run_status.into_body(), 65_536).await.unwrap()).unwrap();
    assert_eq!(run_status_body["run_id"], run_id);
    assert_eq!(run_status_body["status"], "succeeded");
    assert_eq!(
        run_status_body["user_turn_node_id"],
        send_response["user_turn_node_id"]
    );

    let run_status_contract: Value = serde_json::from_str(include_str!(
        "../../../../../contracts/golden/run-status-v1.json"
    ))
    .unwrap();
    let direct_run_status = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/api/runs/{run_id}"))
                .header("authorization", format!("Bearer {token}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        direct_run_status.status().as_u16(),
        run_status_contract["success_status"].as_u64().unwrap() as u16
    );
    let direct_run_status_body: Value = serde_json::from_slice(
        &to_bytes(direct_run_status.into_body(), 65_536)
            .await
            .unwrap(),
    )
    .unwrap();
    for field in run_status_contract["required_fields"]
        .as_array()
        .unwrap()
    {
        let field = field.as_str().unwrap();
        assert!(direct_run_status_body.get(field).is_some(), "missing {field}");
    }
    assert_eq!(direct_run_status_body["run_id"], run_id);
    assert_eq!(direct_run_status_body["status"], "succeeded");
    let terminal_statuses = run_status_contract["terminal_statuses"]
        .as_array()
        .unwrap();
    assert!(terminal_statuses.contains(&direct_run_status_body["status"]));
    assert_eq!(direct_run_status_body["terminal"], true);
    assert_eq!(
        direct_run_status_body["user_turn_node_id"],
        send_response["user_turn_node_id"]
    );

    let transcript_snapshot = service
        .chroma_graph_snapshot(&request(), "conversation")
        .await
        .unwrap();
    assert!(
        transcript_snapshot
            .nodes()
            .iter()
            .any(|record| record.id == assistant_id),
        "assistant turn should be visible in the authorized snapshot after durable completion"
    );

    let transcript_response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/conversations/conversation-create-contract/turns")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(transcript_response.status(), StatusCode::OK);
    let transcript: Value = serde_json::from_slice(
        &to_bytes(transcript_response.into_body(), 65_536)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(transcript.as_object().unwrap().len(), 2);
    assert_eq!(
        transcript["conversation_id"],
        "conversation-create-contract"
    );
    let transcript_turns = transcript["turns"].as_array().unwrap();
    assert_eq!(
        transcript_turns
            .iter()
            .map(|turn| turn["role"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["user", "assistant", "user", "assistant"],
        "turns={transcript_turns:#?}"
    );
    assert!(
        transcript_turns.iter().all(|turn| {
            turn.get("event_type").is_none() && turn.get("workflow_node_id").is_none()
        }),
        "workflow and event records must not leak into the user transcript"
    );

    let duplicate = service.execute(request()).await;
    assert_eq!(duplicate.status, StatusCode::CONFLICT.as_u16());
    let wrong_owner = ApiEffectRequest {
        principal: json!({
            "sub":"other-user", "user_id":"other-user", "role":"rw",
            "ns":["conversation"], "storage_ns":"conversation",
            "capabilities":["spawn_process"]
        }),
        ..request()
    };
    assert_eq!(
        service.execute(wrong_owner).await.status,
        StatusCode::FORBIDDEN.as_u16()
    );

    let agentic_workflow_id = "agentic_answering.v2";
    let agentic_workflow = &workflow_contract["workflows"][agentic_workflow_id];
    let agentic_nodes = agentic_workflow["required_ops"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| {
            let op = value.as_str().unwrap();
            json!({"id":op,"label":op,"metadata":{
                "entity_type":"workflow_node","workflow_id":agentic_workflow_id,
                "wf_op":op,"wf_version":"v1","wf_executor_target":"rust-native",
                "wf_start":op=="start"
            }})
        })
        .collect::<Vec<_>>();
    let agentic_edges = agentic_workflow["required_routes"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(index, route)| {
            json!({"id":format!("agentic-route-{index}"),"source_ids":[route["from"]],
            "target_ids":[route["to"]],"metadata":{
                "entity_type":"workflow_edge","workflow_id":agentic_workflow_id,
                "wf_predicate":route["predicate"],"wf_priority":route["priority"],
                "wf_is_default":route["default"],"wf_multiplicity":route["multiplicity"]
            }})
        })
        .collect::<Vec<_>>();
    service
        .store
        .replace_named_projection(
            "workflow_design",
            agentic_workflow_id,
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
        .unwrap();
    service
        .store
        .put_workflow_design_snapshot(
            agentic_workflow_id,
            WorkflowDesignSnapshotWrite {
                version: 1,
                seq: 1,
                schema_version: 1,
                payload_json: serde_json::to_string(
                    &json!({"nodes":agentic_nodes,"edges":agentic_edges}),
                )
                .unwrap(),
            },
        )
        .unwrap();
    let agentic_turn_id = runtime.state["turn_node_id"].as_str().unwrap();
    let agentic_submit = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/workflow/runs".to_owned(),
            body: serde_json::to_vec(&json!({
                "run_id":"sqlite-chroma-agentic-provider-run",
                "workflow_id":agentic_workflow_id,
                "conversation_id":"conversation-create-contract",
                "turn_node_id":agentic_turn_id,
                "user_id":"conversation-owner",
                "initial_state":{
                    "conversation_id":"conversation-create-contract",
                    "turn_node_id":agentic_turn_id,
                    "user_id":"conversation-owner",
                    "turn_index":0,
                    "max_candidates":10,
                    "max_used":4,
                    "max_retrieval_level":3,
                    "max_iter":1
                }
            }))
            .unwrap(),
            principal: principal.clone(),
        })
        .await;
    assert_eq!(
        agentic_submit.status,
        StatusCode::ACCEPTED.as_u16(),
        "{}",
        String::from_utf8_lossy(&agentic_submit.body)
    );
    let agentic_run_id = "sqlite-chroma-agentic-provider-run";
    for _ in 0..48 {
        let status = service
            .store
            .get_server_run(&agentic_run_id)
            .unwrap()
            .unwrap();
        if matches!(status.status.as_str(), "succeeded" | "failed" | "cancelled") {
            break;
        }
        assert!(
            service.execute_inline_native_work_once().await.unwrap(),
            "native scheduler stopped before the provider-backed run became terminal: {status:#?}"
        );
    }
    let agentic_run = service
        .store
        .get_server_run(&agentic_run_id)
        .unwrap()
        .unwrap();
    assert_eq!(
        agentic_run.status, "succeeded",
        "provider-backed SQLite+Chroma run should complete natively"
    );
    let agentic_state = service
        .store
        .read_recorded_runtime_state(
            agentic_run_id,
            agentic_workflow_id,
            "conversation-create-contract",
        )
        .unwrap()
        .unwrap();
    assert_eq!(
        agentic_state.state["used_node_ids"],
        json!(["knowledge-node"])
    );
    assert!(
        agentic_state.state["_rt_node_executor_targets"]
            .as_object()
            .unwrap()
            .values()
            .all(|target| target == "rust-native"),
        "every agentic-answer workflow node must remain on the selected Rust implementation"
    );
    assert_eq!(
        agentic_state.state["answer"]["claims"][0]["citations"][0]["source_node_id"],
        "knowledge-node"
    );
    assert_eq!(agentic_state.state["done"], true);
    let agentic_assistant_id =
        agentic_state.state["agentic_answering_result"]["assistant_turn_node_id"]
            .as_str()
            .expect("provider-backed run records its assistant turn")
            .to_owned();
    assert!(
        service
            .store
            .latest_entity_events("conversation", "node", &[agentic_assistant_id.clone()],)
            .unwrap()
            .contains_key(&agentic_assistant_id)
    );
    let agentic_turn_id = agentic_state.state["turn_node_id"].as_str().unwrap();
    assert!(
        service
            .store
            .replay_raw_events("conversation", 0, 200)
            .unwrap()
            .iter()
            .any(|event| {
                event.entity_kind == "edge"
                    && serde_json::from_str::<Value>(&event.payload_json)
                        .ok()
                        .is_some_and(|payload| {
                            payload["relation"] == "next_turn"
                                && payload["source_ids"] == json!([agentic_turn_id])
                                && payload["target_ids"] == json!([agentic_assistant_id])
                        })
            })
    );
    let execution_lanes = service
        .store
        .list_projected_lane_messages(kogwistar_store::LaneMessageFilter {
            correlation_id: Some(agentic_run_id.to_owned()),
            limit: 100,
            ..Default::default()
        })
        .unwrap();
    assert!(!execution_lanes.is_empty());
    assert!(execution_lanes.iter().all(|lane| {
        lane.status == "completed" && lane.recipient_id != "python-worker"
    }), "Rust-native workflow must not fall back to a Python worker: {execution_lanes:#?}");
    let execution_traces = service
        .store
        .list_server_run_events(agentic_run_id, 0, 10_000)
        .unwrap()
        .into_iter()
        .filter(|event| event.event_type == "workflow.trace.v1")
        .filter_map(|event| serde_json::from_str::<Value>(&event.payload_json).ok())
        .filter(|event| event["trace"]["executor_target"] == "rust-native")
        .collect::<Vec<_>>();
    assert!(
        !execution_traces.is_empty(),
        "the Rust-native backend path must leave native execution trace evidence"
    );
    chat_server.await.unwrap();
    let _ = stop_tx.send(());
    assert!(fake_chroma.await.unwrap().is_empty());
    drop(service);
    remove_test_sqlite(&path);
    remove_test_sqlite(&auth_path);

    let metadata_path = std::env::temp_dir().join(format!(
        "kogwistar-sqlite-metadata-conversation-create-{}.sqlite",
        uuid::Uuid::new_v4().simple()
    ));
    let metadata_only = SqliteRunApplicationService::open(&metadata_path).unwrap();
    assert!(!metadata_only.supports_http_operation("POST", "/api/conversations"));
    drop(metadata_only);
    remove_test_sqlite(&metadata_path);
}

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
        "capabilities": ["write_graph"]
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
        "capabilities": ["read_graph", "write_graph", "workflow.run.write"]
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

#[tokio::test]
async fn sqlite_chroma_wisdom_upsert_uses_canonical_events_and_acl_before_projection() {
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
        chroma_config("wisdom-nodes"),
        chroma_config("wisdom-edges"),
    )
    .await
    .unwrap();
    chroma_setup.await.unwrap();

    let sqlite_path = std::env::temp_dir().join(format!(
        "kogwistar-sqlite-chroma-wisdom-{}.sqlite",
        uuid::Uuid::new_v4()
    ));
    let provider = Arc::new(FixedEmbeddingProvider {
        profile: kogwistar_engine::embedding::EmbeddingProfile {
            provider: "test".to_owned(),
            model: "fixed-v1".to_owned(),
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
        },
        vector: vec![0.6, 0.8],
        calls: Arc::new(Mutex::new(Vec::new())),
        fail: None,
    });
    let service = SqliteRunApplicationService::open(&sqlite_path)
        .unwrap()
        .with_chroma_graph(graph)
        .with_embedding_provider(provider);
    let mcp_contract: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/wisdom-graph-upsert-mcp-v1.json"
    )))
    .unwrap();
    let mcp_secret = "mcp-wisdom-contract-test-secret";
    let mcp_token = encode(
        &Header::new(Algorithm::HS256),
        &json!({
            "sub": "alice", "user_id": "alice", "role": "rw", "roles": ["writer"],
            "ns": ["wisdom"], "storage_ns": "wisdom", "graph_space": "wisdom",
            "capabilities": ["write_graph"], "exp": 4102444800_u64
        }),
        &EncodingKey::from_secret(mcp_secret.as_bytes()),
    )
    .unwrap();
    let mcp_app = router_with_application(
        ApiState {
            health: snapshot(),
            required_roles: vec!["writer".to_owned()],
            implementation: ImplementationSnapshot::default(),
            auth: AuthConfig {
                mode: "jwt".to_owned(),
                algorithm: Some("HS256".to_owned()),
                key: Some(mcp_secret.to_owned()),
                ..AuthConfig::default()
            },
        },
        Arc::new(service.clone()),
    );
    let mcp_response = mcp_app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp")
                .header("content-type", "application/json")
                .header("accept", "application/json, text/event-stream")
                .header("authorization", format!("Bearer {mcp_token}"))
                .body(Body::from(
                    serde_json::to_vec(&json!({
                        "jsonrpc": "2.0", "id": "wisdom-upsert-contract",
                        "method": "tools/call",
                        "params": {
                            "name": mcp_contract["operation"],
                            "arguments": mcp_contract["arguments"]
                        }
                    }))
                    .unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(mcp_response.status(), StatusCode::OK);
    assert_eq!(
        mcp_response.headers()["content-type"],
        "text/event-stream; charset=utf-8"
    );
    let mcp_body = axum::body::to_bytes(mcp_response.into_body(), 65_536)
        .await
        .unwrap();
    let mcp_wire = String::from_utf8(mcp_body.to_vec()).unwrap();
    let mcp_json = mcp_wire
        .lines()
        .find_map(|line| line.strip_prefix("data: "))
        .expect("Streamable HTTP wraps the JSON-RPC response in an SSE data field");
    let mcp_payload: Value = serde_json::from_str(mcp_json).unwrap();
    assert!(mcp_payload.get("error").is_none(), "{mcp_payload}");
    assert_eq!(
        mcp_payload["result"]["structuredContent"]["node_ids"],
        mcp_contract["expected_structured_content"]["node_ids"]
    );
    assert_eq!(
        mcp_payload["result"]["structuredContent"]["edge_ids"],
        mcp_contract["expected_structured_content"]["edge_ids"]
    );
    assert_eq!(
        mcp_payload["result"]["structuredContent"]["nodes_added"],
        mcp_contract["expected_structured_content"]["nodes_added"]
    );
    assert_eq!(
        mcp_payload["result"]["structuredContent"]["edges_added"],
        mcp_contract["expected_structured_content"]["edges_added"]
    );
    assert!(service.supports_mcp_tool("root", "wisdom.kg_upsert_graph"));
    assert!(service.supports_http_operation("POST", PUBLIC_WISDOM_GRAPH_UPSERT_ROUTE));

    let fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/python-graph-model-payloads-v1.json"
    )))
    .unwrap();
    let mut node = fixture["cases"][0]["payload"].clone();
    node["id"] = json!("sqlite-wisdom-node");
    node["embedding"] = Value::Null;
    let mut edge = fixture["cases"][1]["payload"].clone();
    edge["id"] = json!("sqlite-wisdom-edge");
    edge["source_ids"] = json!(["sqlite-wisdom-node"]);
    edge["target_ids"] = json!([]);
    edge["embedding"] = Value::Null;
    let body = serde_json::to_vec(&json!({"inp": {"nodes": [node], "edges": [edge]}})).unwrap();
    let principal = json!({
        "sub": "alice", "user_id": "alice", "role": "rw",
        "ns": ["wisdom"], "storage_ns": "wisdom", "graph_space": "wisdom",
        "capabilities": ["write_graph"]
    });
    assert!(service.supports_mcp_tool_for_principal("root", "wisdom.kg_upsert_graph", &principal,));
    for denied_principal in [
        json!({"sub":"alice", "role":"ro", "ns":["wisdom"], "capabilities":["write_graph"]}),
        json!({"sub":"alice", "role":"rw", "ns":["docs"], "capabilities":["write_graph"]}),
        json!({"sub":"alice", "role":"rw", "ns":["wisdom"], "capabilities":["read_graph"]}),
    ] {
        assert!(!service.supports_mcp_tool_for_principal(
            "root",
            "wisdom.kg_upsert_graph",
            &denied_principal,
        ));
    }
    assert!(!service.supports_mcp_tool_for_principal(
        "workflow",
        "wisdom.kg_upsert_graph",
        &principal,
    ));
    let denied = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: PUBLIC_WISDOM_GRAPH_UPSERT_ROUTE.to_owned(),
            body: body.clone(),
            principal: json!({"sub": "alice", "role": "ro", "ns": ["wisdom"]}),
        })
        .await;
    assert_eq!(denied.status, StatusCode::FORBIDDEN.as_u16());
    assert!(
        service
            .store
            .replay_raw_events("wisdom", 0, 100)
            .unwrap()
            .is_empty()
    );

    let response = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: PUBLIC_WISDOM_GRAPH_UPSERT_ROUTE.to_owned(),
            body,
            principal,
        })
        .await;
    assert_eq!(
        response.status,
        StatusCode::OK.as_u16(),
        "{}",
        String::from_utf8_lossy(&response.body)
    );
    let payload: Value = serde_json::from_slice(&response.body).unwrap();
    assert_eq!(payload["projection_consistency"], "eventual");
    assert_eq!(payload["node_ids"][0], "sqlite-wisdom-node");
    assert_eq!(payload["edge_ids"][0], "sqlite-wisdom-edge");

    let events = service.store.replay_raw_events("wisdom", 0, 100).unwrap();
    assert!(
        events
            .iter()
            .any(|event| event.entity_id == "sqlite-wisdom-node")
    );
    assert!(
        events
            .iter()
            .any(|event| event.entity_id == "sqlite-wisdom-edge")
    );
    assert!(
        events.iter().any(|event| {
            serde_json::from_str::<Value>(&event.payload_json)
                .ok()
                .and_then(|payload| payload.get("metadata").cloned())
                .and_then(|metadata| metadata.get("entity_type").cloned())
                == Some(json!("acl_record"))
        }),
        "host-derived default ACL records are part of the canonical event transaction"
    );
    let jobs = service
        .store
        .list_index_jobs(Some("wisdom"), Some("PENDING"), None, None, None, 100)
        .unwrap();
    assert!(jobs.iter().any(|job| job.entity_id == "sqlite-wisdom-node"));
    assert!(jobs.iter().any(|job| job.entity_id == "sqlite-wisdom-edge"));
    assert!(jobs.iter().all(|job| job.namespace == "wisdom"));

    let committed_event_count = events.len();
    let mut private_reference = fixture["cases"][1]["payload"].clone();
    private_reference["id"] = json!("bob-reference-edge");
    private_reference["source_ids"] = json!(["sqlite-wisdom-node"]);
    private_reference["target_ids"] = json!([]);
    private_reference["embedding"] = Value::Null;
    let hidden_reference = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: PUBLIC_WISDOM_GRAPH_UPSERT_ROUTE.to_owned(),
            body: serde_json::to_vec(&json!({"inp": {"nodes": [], "edges": [private_reference]}}))
                .unwrap(),
            principal: json!({
                "sub": "bob", "user_id": "bob", "role": "rw",
                "ns": ["wisdom"], "storage_ns": "wisdom", "graph_space": "wisdom",
                "capabilities": ["write_graph"]
            }),
        })
        .await;
    assert_eq!(hidden_reference.status, StatusCode::CONFLICT.as_u16());

    let mut missing_reference = fixture["cases"][1]["payload"].clone();
    missing_reference["id"] = json!("missing-reference-edge");
    missing_reference["source_ids"] = json!(["absent-wisdom-node"]);
    missing_reference["target_ids"] = json!([]);
    missing_reference["embedding"] = Value::Null;
    let missing_reference = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: PUBLIC_WISDOM_GRAPH_UPSERT_ROUTE.to_owned(),
            body: serde_json::to_vec(&json!({"inp": {"nodes": [], "edges": [missing_reference]}}))
                .unwrap(),
            principal: json!({
                "sub": "alice", "user_id": "alice", "role": "rw",
                "ns": ["wisdom"], "storage_ns": "wisdom", "graph_space": "wisdom",
                "workspace": "other-workspace", "capabilities": ["write_graph"]
            }),
        })
        .await;
    assert_eq!(missing_reference.status, StatusCode::CONFLICT.as_u16());
    assert_eq!(
        service
            .store
            .replay_raw_events("wisdom", 0, 100)
            .unwrap()
            .len(),
        committed_event_count
    );

    drop(service);
    remove_test_sqlite(&sqlite_path);
}

#[tokio::test]
async fn sqlite_chroma_public_graph_upsert_uses_shared_extraction_contract() {
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tower::ServiceExt;

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let setup = tokio::spawn(async move {
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
        chroma_config("knowledge-nodes"),
        chroma_config("knowledge-edges"),
    )
    .await
    .unwrap();
    setup.await.unwrap();

    let sqlite_path = std::env::temp_dir().join(format!(
        "kogwistar-sqlite-chroma-public-upsert-{}.sqlite",
        uuid::Uuid::new_v4()
    ));
    let service = SqliteRunApplicationService::open(&sqlite_path)
        .unwrap()
        .with_chroma_graph(graph)
        .with_graph_role("conversation")
        .unwrap();
    let standalone_path = std::env::temp_dir().join(format!(
        "kogwistar-sqlite-standalone-public-upsert-{}.sqlite",
        uuid::Uuid::new_v4()
    ));
    let standalone = SqliteRunApplicationService::open(&standalone_path).unwrap();
    assert_eq!(standalone.graph_role, "knowledge");
    assert!(!standalone.supports_http_operation("POST", PUBLIC_GRAPH_UPSERT_ROUTE));
    assert!(service.supports_http_operation("POST", PUBLIC_GRAPH_UPSERT_ROUTE));

    let mut extraction: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/graph-extraction-v1.json"
    )))
    .unwrap();
    extraction["nodes"][0]["metadata"]["engine_type"] = json!("attacker-selected");
    let body = serde_json::to_vec(&extraction).unwrap();
    let principal = json!({
        "sub": "alice", "user_id": "alice", "role": "rw",
        "ns": ["docs"], "storage_ns": "docs", "capabilities": ["write_graph"]
    });
    for denied_principal in [
        json!({
            "sub": "alice", "user_id": "alice", "role": "ro",
            "ns": ["docs"], "storage_ns": "docs", "capabilities": ["write_graph"]
        }),
        json!({
            "sub": "alice", "user_id": "alice", "role": "rw",
            "ns": ["docs"], "storage_ns": "docs", "capabilities": ["read_graph"]
        }),
        json!({
            "role": "rw", "ns": ["docs"], "storage_ns": "docs",
            "capabilities": ["write_graph"]
        }),
    ] {
        let denied = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: PUBLIC_GRAPH_UPSERT_ROUTE.to_owned(),
                body: body.clone(),
                principal: denied_principal,
            })
            .await;
        assert_eq!(denied.status, StatusCode::FORBIDDEN.as_u16());
        assert!(
            service
                .store
                .replay_raw_events("docs", 0, 100)
                .unwrap()
                .is_empty()
        );
    }
    let secret = "sqlite-chroma-public-upsert-test-secret";
    let token = encode(
        &Header::new(Algorithm::HS256),
        &json!({
            "sub": "alice", "user_id": "alice", "role": "rw",
            "roles": ["reader", "rw"], "ns": ["docs"],
            "storage_ns": "docs", "capabilities": ["write_graph"],
            "exp": 4102444800_u64
        }),
        &EncodingKey::from_secret(secret.as_bytes()),
    )
    .unwrap();
    let app = router_with_application(
        ApiState {
            health: snapshot(),
            required_roles: vec!["reader".to_owned()],
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
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(PUBLIC_GRAPH_UPSERT_ROUTE)
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::from(body.clone()))
                .unwrap(),
        )
        .await;
    let response = response.unwrap();
    let response_status = response.status();
    let response_body = to_bytes(response.into_body(), 65_536).await.unwrap();
    assert_eq!(
        response_status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&response_body)
    );
    let payload: Value = serde_json::from_slice(&response_body).unwrap();
    let response_contract: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/graph-upsert-success-v1.json"
    )))
    .unwrap();
    assert_eq!(
        response_status.as_u16(),
        response_contract["http_status"].as_u64().unwrap() as u16
    );
    let expected_body = &response_contract["body"];
    assert_eq!(payload["document_id"], expected_body["document_id"]);
    assert_eq!(
        payload["node_ids"].as_array().unwrap().len(),
        expected_body["node_ids_count"].as_u64().unwrap() as usize
    );
    assert_eq!(
        payload["edge_ids"].as_array().unwrap().len(),
        expected_body["edge_ids_count"].as_u64().unwrap() as usize
    );
    assert_eq!(payload["nodes_added"], expected_body["nodes_added"]);
    assert_eq!(payload["edges_added"], expected_body["edges_added"]);

    let events = service.store.replay_raw_events("docs", 0, 100).unwrap();
    assert!(events.iter().any(|event| {
        event.entity_id == payload["node_ids"][0]
            && serde_json::from_str::<Value>(&event.payload_json)
                .ok()
                .and_then(|record| record.get("metadata").cloned())
                .and_then(|metadata| metadata.get("engine_type").cloned())
                == Some(json!("conversation"))
    }));
    assert!(events.iter().any(|event| {
        serde_json::from_str::<Value>(&event.payload_json)
            .ok()
            .and_then(|record| record.get("metadata").cloned())
            .and_then(|metadata| metadata.get("entity_type").cloned())
            == Some(json!("acl_record"))
    }));
    assert!(
        events.iter().any(|event| {
            serde_json::from_str::<Value>(&event.payload_json)
                .ok()
                .and_then(|record| record.get("metadata").cloned())
                .and_then(|metadata| metadata.get("acl_truth_graph").cloned())
                == Some(json!("conversation"))
        }),
        "expected ACL event for configured conversation graph role, got {:?}",
        events
            .iter()
            .map(|event| &event.payload_json)
            .collect::<Vec<_>>()
    );
    let document = service
        .store
        .source_document_projection(
            &GraphScope {
                namespace: "docs".to_owned(),
                workspace_id: Some(String::new()),
                graph_space: Some(String::new()),
            },
            "doc-1",
        )
        .unwrap();
    assert_eq!(
        document.and_then(|record| record.document),
        Some("Alpha connects to Beta.".to_owned())
    );
    let index_jobs = service
        .store
        .list_index_jobs(Some("docs"), Some("PENDING"), None, None, None, 100)
        .unwrap();
    assert!(
        index_jobs
            .iter()
            .any(|job| job.entity_id == payload["node_ids"][0])
    );
    assert!(
        index_jobs
            .iter()
            .any(|job| job.entity_id == payload["node_ids"][1])
    );
    assert!(index_jobs.iter().all(|job| job.namespace == "docs"));

    let committed_events = events.len();
    let mut hidden_edge = serde_json::from_slice::<Value>(&body).unwrap()["edges"][0].clone();
    hidden_edge["id"] = json!("bob-reference-edge");
    hidden_edge["local_id"] = json!("ne:bob-reference-edge");
    hidden_edge["source_ids"] = json!([payload["node_ids"][0]]);
    hidden_edge["target_ids"] = json!([]);
    let mut hidden_request = serde_json::from_slice::<Value>(&body).unwrap();
    hidden_request["nodes"] = json!([]);
    hidden_request["edges"] = json!([hidden_edge]);
    let hidden_reference = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: PUBLIC_GRAPH_UPSERT_ROUTE.to_owned(),
            body: serde_json::to_vec(&hidden_request).unwrap(),
            principal: json!({
                "sub": "bob", "user_id": "bob", "role": "rw",
                "ns": ["docs"], "storage_ns": "docs", "capabilities": ["write_graph"]
            }),
        })
        .await;
    assert_eq!(hidden_reference.status, StatusCode::CONFLICT.as_u16());
    assert_eq!(
        service
            .store
            .replay_raw_events("docs", 0, 100)
            .unwrap()
            .len(),
        committed_events,
        "cross-owner endpoint denial must not append graph or ACL events"
    );

    let document_before_invalid = service
        .store
        .source_document_projection(
            &GraphScope {
                namespace: "docs".to_owned(),
                workspace_id: Some(String::new()),
                graph_space: Some(String::new()),
            },
            "doc-1",
        )
        .unwrap()
        .and_then(|record| record.document);
    let jobs_before_invalid = service
        .store
        .list_index_jobs(Some("docs"), Some("PENDING"), None, None, None, 100)
        .unwrap()
        .into_iter()
        .map(|job| (job.entity_id, job.status))
        .collect::<Vec<_>>();
    let mut invalid: Value = serde_json::from_slice(&body).unwrap();
    invalid["nodes"][0]["references"][0]["excerpt"] = json!("forged");
    let rejected = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: PUBLIC_GRAPH_UPSERT_ROUTE.to_owned(),
            body: serde_json::to_vec(&invalid).unwrap(),
            principal,
        })
        .await;
    assert_ne!(rejected.status, StatusCode::OK.as_u16());
    assert_eq!(
        service
            .store
            .replay_raw_events("docs", 0, 100)
            .unwrap()
            .len(),
        committed_events
    );
    let document_after_invalid = service
        .store
        .source_document_projection(
            &GraphScope {
                namespace: "docs".to_owned(),
                workspace_id: Some(String::new()),
                graph_space: Some(String::new()),
            },
            "doc-1",
        )
        .unwrap()
        .and_then(|record| record.document);
    assert_eq!(document_after_invalid, document_before_invalid);
    let jobs_after_invalid = service
        .store
        .list_index_jobs(Some("docs"), Some("PENDING"), None, None, None, 100)
        .unwrap()
        .into_iter()
        .map(|job| (job.entity_id, job.status))
        .collect::<Vec<_>>();
    assert_eq!(jobs_after_invalid, jobs_before_invalid);

    let malformed_hyperedge: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/graph-extraction-invalid-hyperedge-v1.json"
    )))
    .unwrap();
    let rejected_hyperedge = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: PUBLIC_GRAPH_UPSERT_ROUTE.to_owned(),
            body: serde_json::to_vec(&malformed_hyperedge).unwrap(),
            principal: json!({
                "sub": "alice", "user_id": "alice", "role": "rw",
                "ns": ["docs"], "storage_ns": "docs", "capabilities": ["write_graph"]
            }),
        })
        .await;
    assert_eq!(
        rejected_hyperedge.status,
        StatusCode::UNPROCESSABLE_ENTITY.as_u16()
    );
    assert_eq!(
        service.store.replay_raw_events("docs", 0, 100).unwrap().len(),
        committed_events,
        "malformed hyperedge must not append graph or ACL events"
    );
    let document_after_hyperedge_rejection = service
        .store
        .source_document_projection(
            &GraphScope {
                namespace: "docs".to_owned(),
                workspace_id: Some(String::new()),
                graph_space: Some(String::new()),
            },
            "doc-invalid-hyperedge",
        )
        .unwrap();
    assert!(document_after_hyperedge_rejection.is_none());
    let jobs_after_hyperedge_rejection = service
        .store
        .list_index_jobs(Some("docs"), Some("PENDING"), None, None, None, 100)
        .unwrap()
        .into_iter()
        .map(|job| (job.entity_id, job.status))
        .collect::<Vec<_>>();
    assert_eq!(jobs_after_hyperedge_rejection, jobs_before_invalid);

    drop(service);
    drop(standalone);
    remove_test_sqlite(&sqlite_path);
    remove_test_sqlite(&standalone_path);
}

#[tokio::test]
async fn postgres_identity_claim_rejects_unregistered_principal_before_storage() {
    let service = PostgresRunApplicationService::from_dsn(
        "postgres://unused:unused@127.0.0.1:1/unused",
        "worker_identity_registration",
    )
    .unwrap();
    let recipient = runtime_worker_identity_recipient(
        "python-cpu-v1",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    );
    let response = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/internal/runtime/claim".to_owned(),
            body: json!({
                "claimed_by":"unregistered-worker",
                "recipient_ids":[recipient]
            })
            .to_string()
            .into_bytes(),
            principal: json!({"sub":"unregistered-worker", "role":"reader"}),
        })
        .await;

    assert_eq!(response.status, StatusCode::FORBIDDEN.as_u16());
    let body: Value = serde_json::from_slice(&response.body).unwrap();
    assert_eq!(body["code"], "KOGWISTAR_WORKER_IDENTITY_FORBIDDEN");
}
