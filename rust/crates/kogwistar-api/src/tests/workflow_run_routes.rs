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
async fn resume_contract_checks_authority_before_run_lookup() {
    let sqlite_path = std::env::temp_dir().join(format!(
        "kogwistar-resume-contract-auth-order-{}.sqlite",
        uuid::Uuid::new_v4()
    ));
    let sqlite = SqliteRunApplicationService::open(&sqlite_path).unwrap();
    let postgres = PostgresRunApplicationService::from_dsn(
        "postgres://unused:unused@127.0.0.1:1/unused",
        "resume_contract_auth_order",
    )
    .unwrap();

    for service in [&sqlite as &dyn ApplicationService, &postgres] {
        for (principal, expected_code) in [
            (
                json!({"ns": ["conversation"], "capabilities": ["workflow.run.read"]}),
                "KOGWISTAR_NAMESPACE_FORBIDDEN",
            ),
            (
                json!({"ns": ["workflow"], "capabilities": []}),
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
            ),
        ] {
            let response = service
                .execute(ApiEffectRequest {
                    contract_version: 1,
                    method: "GET".to_owned(),
                    path_and_query:
                        "/api/workflow/runs/missing/resume-contract".to_owned(),
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

    drop(sqlite);
    remove_test_sqlite(&sqlite_path);
}

#[tokio::test]
async fn workflow_run_submit_checks_both_capabilities_before_admission() {
    let sqlite_path = std::env::temp_dir().join(format!(
        "kogwistar-run-submit-auth-{}.sqlite",
        uuid::Uuid::new_v4()
    ));
    let service = SqliteRunApplicationService::open(&sqlite_path).unwrap();
    for (run_id, capabilities) in [
        ("missing-spawn", json!(["workflow.run.write"])),
        ("missing-write", json!(["spawn_process"])),
    ] {
        let response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/workflow/runs".to_owned(),
                body: serde_json::to_vec(&json!({
                    "run_id": run_id,
                    "workflow_id": "submit-auth-workflow",
                    "conversation_id": "submit-auth-conversation"
                }))
                .unwrap(),
                principal: json!({
                    "sub": "alice",
                    "ns": ["workflow"],
                    "capabilities": capabilities
                }),
            })
            .await;
        assert_eq!(response.status, StatusCode::FORBIDDEN.as_u16());
        assert_eq!(
            serde_json::from_slice::<Value>(&response.body).unwrap()["code"],
            "KOGWISTAR_CAPABILITY_FORBIDDEN"
        );
        assert!(service.store.get_server_run(run_id).unwrap().is_none());
    }

    drop(service);
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
        let conversation_scope = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: "/api/workflow/runs/missing/evidence".to_owned(),
                body: Vec::new(),
                principal: json!({
                    "ns": ["conversation"],
                    "capabilities": ["workflow.run.read", "project_view"]
                }),
            })
            .await;
        assert_eq!(
            conversation_scope.status,
            StatusCode::FORBIDDEN.as_u16()
        );
        assert_eq!(
            serde_json::from_slice::<Value>(&conversation_scope.body).unwrap()["code"],
            "KOGWISTAR_NAMESPACE_FORBIDDEN"
        );

        let response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: "/api/workflow/runs/missing/evidence".to_owned(),
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
                path_and_query: "/api/workflow/runs/missing/evidence".to_owned(),
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

        let response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: "/api/workflow/runs/missing/evidence".to_owned(),
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
            path_and_query: "/api/workflow/runs/missing/evidence".to_owned(),
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
        runtime_run_namespace("/api/runs/legacy-run?include=events"),
        Some("conversation")
    );
    assert_eq!(
        runtime_run_namespace("/api/workflow/runs/workflow-run"),
        Some("workflow")
    );
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
async fn postgres_workflow_run_cancel_contract_enforces_scope_and_write_capability() {
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use tower::ServiceExt;

    let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
        .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
    let schema = format!("rust_run_cancel_{}", uuid::Uuid::new_v4().simple());
    let service = PostgresRunApplicationService::from_dsn(&dsn, &schema).unwrap();
    service.ensure_graph_type_schemas().await.unwrap();
    service
        .store
        .create_server_run(ServerRunCreate {
            run_id: "cancel-auth-run".to_owned(),
            conversation_id: "cancel-auth-conversation".to_owned(),
            workflow_id: "cancel-auth-workflow".to_owned(),
            user_id: Some("alice".to_owned()),
            user_turn_node_id: "cancel-auth-turn".to_owned(),
            status: "running".to_owned(),
        })
        .await
        .unwrap();

    for (principal, expected_code) in [
        (
            json!({"sub":"alice", "ns":["conversation"], "capabilities":["workflow.run.write"]}),
            "KOGWISTAR_NAMESPACE_FORBIDDEN",
        ),
        (
            json!({"sub":"alice", "ns":["workflow"], "capabilities":[]}),
            "KOGWISTAR_CAPABILITY_FORBIDDEN",
        ),
    ] {
        let denied = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/workflow/runs/cancel-auth-run/cancel".to_owned(),
                body: Vec::new(),
                principal,
            })
            .await;
        assert_eq!(denied.status, StatusCode::FORBIDDEN.as_u16());
        assert_eq!(
            serde_json::from_slice::<Value>(&denied.body).unwrap()["code"],
            expected_code
        );
    }
    let unchanged = service
        .store
        .get_server_run("cancel-auth-run")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(unchanged.status, "running");
    assert!(!unchanged.cancel_requested);

    let secret = "workflow-run-cancel-mcp-test-secret";
    let token = encode(
        &Header::new(Algorithm::HS256),
        &json!({
            "sub": "alice",
            "role": "rw",
            "ns": ["workflow"],
            "capabilities": [],
            "iss": "local",
            "exp": 4102444800_u64
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
    let mcp_denied = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp/workflow")
                .header("content-type", "application/json")
                .header("accept", "application/json, text/event-stream")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::from(
                    r#"{"jsonrpc":"2.0","id":"cancel-denied","method":"tools/call","params":{"name":"workflow.run_cancel","arguments":{"run_id":"cancel-auth-run"}}}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(mcp_denied.status(), StatusCode::OK);
    let mcp_wire = String::from_utf8(
        to_bytes(mcp_denied.into_body(), 8192).await.unwrap().to_vec(),
    )
    .unwrap();
    let mcp_data = mcp_wire
        .lines()
        .find_map(|line| line.strip_prefix("data: "))
        .expect("MCP Streamable HTTP result data");
    let mcp_denied: Value = serde_json::from_str(mcp_data).unwrap();
    assert_eq!(mcp_denied["result"]["isError"], true);
    assert!(!mcp_denied["result"]["content"][0]["text"]
        .as_str()
        .unwrap()
        .is_empty());
    let unchanged_after_mcp = service
        .store
        .get_server_run("cancel-auth-run")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(unchanged_after_mcp.status, "running");
    assert!(!unchanged_after_mcp.cancel_requested);

    let accepted = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/workflow/runs/cancel-auth-run/cancel".to_owned(),
            body: Vec::new(),
            principal: json!({"sub":"alice", "ns":["workflow"], "capabilities":["workflow.run.write"]}),
        })
        .await;
    let contract: Value = serde_json::from_str(include_str!(
        "../../../../../contracts/golden/run-cancel-v1.json"
    ))
    .unwrap();
    assert_eq!(accepted.status, contract["accepted_status"].as_u64().unwrap() as u16);
    assert_eq!(contract["workflow_route"], "POST /api/workflow/runs/{run_id}/cancel");
    let body: Value = serde_json::from_slice(&accepted.body).unwrap();
    for field in contract["required_fields"].as_array().unwrap() {
        let field = field.as_str().unwrap();
        assert!(body.get(field).is_some(), "missing {field}");
    }
    assert!(contract["accepted_run_statuses"].as_array().unwrap().contains(&body["status"]));
    assert_eq!(body["cancel_requested"], true);
}

#[tokio::test]
#[ignore = "requires KOGWISTAR_TEST_PG_DSN; uses a disposable PostgreSQL schema"]
async fn postgres_conversation_run_routes_enforce_conversation_scope() {
    let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
        .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
    let schema = format!("rust_conversation_run_scope_{}", uuid::Uuid::new_v4().simple());
    let service = PostgresRunApplicationService::from_dsn(&dsn, &schema).unwrap();
    service.ensure_graph_type_schemas().await.unwrap();
    service
        .store
        .create_server_run(ServerRunCreate {
            run_id: "conversation-scope-run".to_owned(),
            conversation_id: "conversation-scope-conversation".to_owned(),
            workflow_id: "agentic_answering.v2".to_owned(),
            user_id: Some("alice".to_owned()),
            user_turn_node_id: "conversation-scope-turn".to_owned(),
            status: "running".to_owned(),
        })
        .await
        .unwrap();

    let status = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "GET".to_owned(),
            path_and_query: "/api/runs/conversation-scope-run".to_owned(),
            body: Vec::new(),
            principal: json!({"sub":"alice", "ns":["conversation"], "capabilities":["workflow.run.read"]}),
        })
        .await;
    assert_eq!(status.status, StatusCode::OK.as_u16());
    assert_eq!(serde_json::from_slice::<Value>(&status.body).unwrap()["status"], "running");

    let denied_status = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "GET".to_owned(),
            path_and_query: "/api/runs/conversation-scope-run".to_owned(),
            body: Vec::new(),
            principal: json!({"sub":"alice", "ns":["conversation"], "capabilities":[]}),
        })
        .await;
    assert_eq!(denied_status.status, StatusCode::FORBIDDEN.as_u16());
    assert_eq!(
        serde_json::from_slice::<Value>(&denied_status.body).unwrap()["code"],
        "KOGWISTAR_CAPABILITY_FORBIDDEN"
    );

    let wrong_route = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "GET".to_owned(),
            path_and_query: "/api/workflow/runs/conversation-scope-run".to_owned(),
            body: Vec::new(),
            principal: json!({"sub":"alice", "ns":["conversation"], "capabilities":["workflow.run.read"]}),
        })
        .await;
    assert_eq!(wrong_route.status, StatusCode::FORBIDDEN.as_u16());
    assert_eq!(
        serde_json::from_slice::<Value>(&wrong_route.body).unwrap()["code"],
        "KOGWISTAR_NAMESPACE_FORBIDDEN"
    );

    let denied_cancel = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/runs/conversation-scope-run/cancel".to_owned(),
            body: Vec::new(),
            principal: json!({"sub":"alice", "ns":["conversation"], "capabilities":[]}),
        })
        .await;
    assert_eq!(denied_cancel.status, StatusCode::FORBIDDEN.as_u16());
    assert_eq!(
        serde_json::from_slice::<Value>(&denied_cancel.body).unwrap()["code"],
        "KOGWISTAR_CAPABILITY_FORBIDDEN"
    );
    let unchanged = service
        .store
        .get_server_run("conversation-scope-run")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(unchanged.status, "running");
    assert!(!unchanged.cancel_requested);

    let accepted = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/runs/conversation-scope-run/cancel".to_owned(),
            body: Vec::new(),
            principal: json!({"sub":"alice", "ns":["conversation"], "capabilities":["workflow.run.write"]}),
        })
        .await;
    assert_eq!(accepted.status, StatusCode::ACCEPTED.as_u16());
    let cancelled: Value = serde_json::from_slice(&accepted.body).unwrap();
    assert_eq!(cancelled["status"], "cancelling");
    assert_eq!(cancelled["cancel_requested"], true);
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

    for (principal, expected_code) in [
        (
            json!({
                "sub":"alice", "user_id":"alice", "role":"ro",
                "ns":["workflow"], "capabilities":[]
            }),
            "KOGWISTAR_CAPABILITY_FORBIDDEN",
        ),
        (
            json!({
                "sub":"alice", "user_id":"alice", "role":"ro",
                "ns":["conversation"], "capabilities":["workflow.run.read"]
            }),
            "KOGWISTAR_NAMESPACE_FORBIDDEN",
        ),
    ] {
        let denied = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: "/api/workflow/runs/evidence-run".to_owned(),
                body: Vec::new(),
                principal,
            })
            .await;
        assert_eq!(denied.status, StatusCode::FORBIDDEN.as_u16());
        assert_eq!(
            serde_json::from_slice::<Value>(&denied.body).unwrap()["code"],
            expected_code
        );
    }
    let status = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "GET".to_owned(),
            path_and_query: "/api/workflow/runs/evidence-run".to_owned(),
            body: Vec::new(),
            principal: json!({
                "sub":"alice", "user_id":"alice", "role":"ro",
                "ns":["workflow"], "capabilities":["workflow.run.read"]
            }),
        })
        .await;
    let status_contract: Value = serde_json::from_str(include_str!(
        "../../../../../contracts/golden/workflow-run-status-http-v1.json"
    ))
    .unwrap();
    assert_eq!(
        status.status,
        status_contract["success_status"].as_u64().unwrap() as u16
    );
    let status_body: Value = serde_json::from_slice(&status.body).unwrap();
    for field in status_contract["required_fields"].as_array().unwrap() {
        let field = field.as_str().unwrap();
        assert!(status_body.get(field).is_some(), "missing {field}");
    }
    assert_eq!(status_body["run_id"], "evidence-run");
    assert_eq!(status_body["status"], "succeeded");
    assert_eq!(status_body["terminal"], true);

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
    let lineage_contract: Value = serde_json::from_str(include_str!(
        "../../../../../contracts/golden/workflow-run-lineage-v1.json"
    ))
    .unwrap();
    assert_eq!(
        allowed.status().as_u16(),
        lineage_contract["success_status"].as_u64().unwrap() as u16
    );
    let allowed_body = to_bytes(allowed.into_body(), 8192).await.unwrap();
    let payload = serde_json::from_slice::<Value>(&allowed_body).unwrap();
    assert_eq!(payload, lineage_contract["response"]);

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
