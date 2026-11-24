#[test]
fn resume_wait_for_completion_defaults_to_sync_and_accepts_explicit_false() {
    assert!(query_boolean(
        "/api/workflow/runs/run-1/resume",
        "wait_for_completion",
        true
    ));
    assert!(query_boolean(
        "/api/workflow/runs/run-1/resume?wait_for_completion=true",
        "wait_for_completion",
        true
    ));
    assert!(!query_boolean(
        "/api/workflow/runs/run-1/resume?wait_for_completion=false",
        "wait_for_completion",
        true
    ));
    assert!(query_boolean(
        "/api/workflow/runs/run-1/resume?wait_for_completion=invalid",
        "wait_for_completion",
        true
    ));
}

#[tokio::test]
#[ignore = "requires KOGWISTAR_TEST_PG_DSN; uses a disposable PostgreSQL schema"]
async fn postgres_resume_syscall_resumes_persisted_suspended_token() {
    use kogwistar_store::{ProjectionReadStore, RUNTIME_CURRENT_STATE_NAMESPACE};

    let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
        .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
    let schema = format!("rust_syscall_resume_{}", uuid::Uuid::new_v4().simple());
    let service = PostgresRunApplicationService::from_dsn(&dsn, &schema).unwrap();
    service.ensure_graph_type_schemas().await.unwrap();
    let run_id = "postgres-resume-syscall-run";
    let workflow_id = "postgres-resume-syscall-workflow";
    let conversation_id = "postgres-resume-syscall-conversation";
    let principal = json!({
        "sub":"postgres-resume-owner", "role":"rw", "ns":["workflow"],
        "capabilities":["spawn_process", "workflow.run.write"]
    });
    let submitted = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/workflow/runs".to_owned(),
            body: serde_json::to_vec(&json!({
                "run_id":run_id,
                "workflow_id":workflow_id,
                "conversation_id":conversation_id,
                "turn_node_id":"postgres-resume-turn"
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
                "claimed_by":"postgres-test-suspender", "limit":1, "lease_seconds":60
            }))
            .unwrap(),
            principal: principal.clone(),
        })
        .await;
    assert_eq!(claimed.status, StatusCode::OK.as_u16());
    let claimed: Value = serde_json::from_slice(&claimed.body).unwrap();
    let work = &claimed["work"][0];
    let suspended = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/internal/runtime/results".to_owned(),
            body: serde_json::to_vec(&json!({
                "handoff": {
                    "message_id":work["message_id"], "claimed_by":work["claimed_by"],
                    "run_id":work["run_id"], "step_id":work["step_id"],
                    "correlation_id":work["correlation_id"]
                },
                "effect": {
                    "contract_version":1, "effect_id":"postgres-suspend-for-syscall-resume",
                    "status":"suspended", "state_update":[["u", {"before_suspend":true}]],
                    "successors":[], "wait_reason":"approval",
                    "resume_payload":{"question":"continue?"}
                }
            }))
            .unwrap(),
            principal: principal.clone(),
        })
        .await;
    assert_eq!(
        suspended.status,
        StatusCode::OK.as_u16(),
        "{}",
        String::from_utf8_lossy(&suspended.body)
    );
    let projection = ProjectionReadStore::named_projection(
        &service.store,
        RUNTIME_CURRENT_STATE_NAMESPACE,
        run_id,
    )
    .await
    .unwrap()
    .unwrap();
    let state: RecordedRuntimeState =
        serde_json::from_value(Value::Object(projection.payload)).unwrap();
    let (node_id, _, token_id, _) = state.frontier.suspended[0].clone();

    let resume_service = service.clone();
    let resume_future = tokio::spawn(async move {
        resume_service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/syscall/v1/resume".to_owned(),
                body: serde_json::to_vec(&json!({
                    "version":"v1", "op":"resume",
                    "args": {
                        "run_id":run_id, "workflow_id":workflow_id,
                        "conversation_id":conversation_id, "turn_node_id":"postgres-resume-turn",
                        "suspended_node_id":node_id, "suspended_token_id":token_id,
                        "client_result":{"approved":true}
                    }
                }))
                .unwrap(),
                principal,
            })
            .await
    });
    let mut claimed_work = None;
    for _ in 0..200 {
        let claimed = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/internal/runtime/claim".to_owned(),
                body: serde_json::to_vec(&json!({
                    "claimed_by":"postgres-test-finisher", "limit":1,
                    "lease_seconds":60, "run_id":run_id
                }))
                .unwrap(),
                principal: json!({
                    "sub":"postgres-resume-owner", "role":"rw", "ns":["workflow"],
                    "capabilities":["workflow.run.write"]
                }),
            })
            .await;
        assert_eq!(claimed.status, StatusCode::OK.as_u16());
        let claimed: Value = serde_json::from_slice(&claimed.body).unwrap();
        if let Some(work) = claimed["work"].as_array().and_then(|items| items.first()) {
            claimed_work = Some(work.clone());
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    let Some(work) = claimed_work else {
        resume_future.abort();
        panic!("PostgreSQL resume did not enqueue claimable continuation within 5 seconds");
    };
    assert_eq!(work["run_id"], run_id);
    let finished = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/internal/runtime/results".to_owned(),
            body: serde_json::to_vec(&json!({
                "handoff": {
                    "message_id":work["message_id"], "claimed_by":work["claimed_by"],
                    "run_id":work["run_id"], "step_id":work["step_id"],
                    "correlation_id":work["correlation_id"]
                },
                "effect": {
                    "contract_version":1,
                    "effect_id":"postgres-finish-after-syscall-resume",
                    "state_update":[["u", {"after_resume":true}]], "successors":[],
                    "result":{"done":true}
                }
            }))
            .unwrap(),
            principal: json!({"role":"rw", "ns":["workflow"]}),
        })
        .await;
    assert_eq!(finished.status, StatusCode::OK.as_u16());

    let resumed = resume_future.await.unwrap();
    assert_eq!(
        resumed.status,
        StatusCode::OK.as_u16(),
        "{}",
        String::from_utf8_lossy(&resumed.body)
    );
    let envelope: Value = serde_json::from_slice(&resumed.body).unwrap();
    let contract: Value = serde_json::from_str(include_str!(
        "../../../../../contracts/golden/workflow-run-resume-result-v1.json"
    ))
    .unwrap();
    assert_eq!(envelope["version"], "v1");
    assert_eq!(envelope["op"], "resume");
    assert_eq!(envelope["status"], "ok");
    for field in contract["required_fields"].as_array().unwrap() {
        let field = field.as_str().unwrap();
        assert!(envelope["result"].get(field).is_some(), "missing {field}");
    }
    assert_eq!(envelope["result"]["workflow_status"], "succeeded");
    assert_eq!(envelope["result"]["final_state"]["after_resume"], true);
    assert_eq!(envelope["result"]["budget"], json!({}));

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

// PYTHON_PARITY_NOTE_V1: SQLite MCP submit/cancel persist effects and terminal retry adds no run event; Python counterparts verify submit/cancel behavior and authorization. rust=rust/crates/kogwistar-api/src/tests/workflow_run_routes.rs#sqlite_mcp_run_submit_persists_admission; python=tests/mcp/test_chat_mcp_tools.py#test_mcp_workflow_runtime_submit_cancel+test_mcp_workflow_run_submit_matches_shared_contract+test_mcp_workflow_run_submit_denies_missing_capabilities_before_admission; fixture=contracts/golden/workflow-run-submit-mcp-v1.json
#[tokio::test]
async fn sqlite_mcp_run_submit_persists_admission() {
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
    use std::sync::Arc;
    use tower::ServiceExt;

    let path = std::env::temp_dir().join(format!(
        "kogwistar-sqlite-mcp-submit-{}.sqlite",
        uuid::Uuid::new_v4()
    ));
    let service = SqliteRunApplicationService::open(&path).unwrap();
    let contract: Value = serde_json::from_str(include_str!(
        "../../../../../contracts/golden/workflow-run-submit-mcp-v1.json"
    ))
    .unwrap();
    let expected_owner = contract["python_service_arguments"]["user_id"]
        .as_str()
        .unwrap();
    let secret = b"sqlite-mcp-submit-secret";
    let token = encode(
        &Header::new(Algorithm::HS256),
        &json!({
            "sub":expected_owner, "user_id":expected_owner, "role":"rw", "ns":["workflow"],
            "capabilities":["spawn_process", "workflow.run.write"],
            "iss":"local", "exp":4102444800_u64
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
        Arc::new(service.clone()),
    );
    let mut arguments = contract["arguments"].clone();
    arguments["user_id"] = json!("caller-controlled-owner");

    let submit_without_spawn_token = encode(
        &Header::new(Algorithm::HS256),
        &json!({
            "sub":expected_owner, "user_id":expected_owner, "role":"rw", "ns":["workflow"],
            "capabilities":["workflow.run.write"],
            "iss":"local", "exp":4102444800_u64
        }),
        &EncodingKey::from_secret(secret),
    )
    .unwrap();
    let denied_submit = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp/workflow")
                .header("content-type", "application/json")
                .header("accept", "application/json, text/event-stream")
                .header(
                    "authorization",
                    format!("Bearer {submit_without_spawn_token}"),
                )
                .body(Body::from(
                    json!({
                        "jsonrpc":"2.0", "id":"sqlite-submit-denied", "method":"tools/call",
                        "params":{"name":contract["tool"], "arguments":arguments.clone()}
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(denied_submit.status(), StatusCode::OK);
    let denied_submit_body = to_bytes(denied_submit.into_body(), 8192).await.unwrap();
    let denied_submit_data = String::from_utf8(denied_submit_body.to_vec()).unwrap();
    let denied_submit_message = denied_submit_data
        .lines()
        .find_map(|line| line.strip_prefix("data: "))
        .unwrap();
    let denied_submit_payload: Value = serde_json::from_str(denied_submit_message).unwrap();
    assert_eq!(
        denied_submit_payload["result"]["isError"],
        true,
        "{denied_submit_payload}"
    );
    let after_denied_submit = kogwistar_store_sqlite::SqliteStore::open(&path).unwrap();
    assert!(after_denied_submit
        .list_server_runs(None, Some("workflow-contract"), Some("conversation-contract"), 20)
        .unwrap()
        .is_empty());
    drop(after_denied_submit);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp/workflow")
                .header("content-type", "application/json")
                .header("accept", "application/json, text/event-stream")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::from(
                    json!({
                        "jsonrpc":"2.0", "id":"sqlite-submit", "method":"tools/call",
                        "params":{"name":contract["tool"], "arguments":arguments}
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), 8192).await.unwrap();
    let data = String::from_utf8(body.to_vec()).unwrap();
    let message = data
        .lines()
        .find_map(|line| line.strip_prefix("data: "))
        .unwrap();
    let payload: Value = serde_json::from_str(message).unwrap();
    assert_ne!(payload["result"]["isError"], true, "{payload}");
    let result = &payload["result"]["structuredContent"];
    let expected = &contract["result"];
    for field in [
        "conversation_id",
        "workflow_id",
        "turn_node_id",
        "status",
        "priority_class",
        "token_budget",
        "time_budget_ms",
        "admission",
    ] {
        assert_eq!(result[field], expected[field], "field {field}");
    }
    let run_id = result["run_id"]
        .as_str()
        .unwrap();
    assert!(!run_id.is_empty());

    let reopened = kogwistar_store_sqlite::SqliteStore::open(&path).unwrap();
    let persisted = reopened.get_server_run(run_id).unwrap().unwrap();
    assert_eq!(persisted.workflow_id, "workflow-contract");
    assert_eq!(persisted.conversation_id, "conversation-contract");
    assert_eq!(persisted.user_id.as_deref(), Some(expected_owner));
    assert_eq!(persisted.status, "queued");
    let before_denial_event_count = kogwistar_store::ServerRunReadStore::server_run_events(
        &reopened,
        run_id,
        0,
        100,
    )
    .await
    .unwrap()
    .len();

    drop(reopened);

    let read_only_token = encode(
        &Header::new(Algorithm::HS256),
        &json!({
            "sub":"alice", "user_id":"alice", "role":"rw", "ns":["workflow"],
            "capabilities":["workflow.run.read"],
            "iss":"local", "exp":4102444800_u64
        }),
        &EncodingKey::from_secret(secret),
    )
    .unwrap();

    let cancel = |bearer: &str| {
        Request::builder()
            .method("POST")
            .uri("/mcp/workflow")
            .header("content-type", "application/json")
            .header("accept", "application/json, text/event-stream")
            .header("authorization", format!("Bearer {bearer}"))
            .body(Body::from(
                json!({
                    "jsonrpc":"2.0", "id":"sqlite-cancel", "method":"tools/call",
                    "params":{"name":"workflow.run_cancel", "arguments":{"run_id":run_id}}
                })
                .to_string(),
            ))
            .unwrap()
    };

    let denied = app
        .clone()
        .oneshot(cancel(&read_only_token))
        .await
        .unwrap();
    assert_eq!(denied.status(), StatusCode::OK);
    let denied_body = to_bytes(denied.into_body(), 8192).await.unwrap();
    let denied_data = String::from_utf8(denied_body.to_vec()).unwrap();
    let denied_message = denied_data
        .lines()
        .find_map(|line| line.strip_prefix("data: "))
        .unwrap();
    let denied_payload: Value = serde_json::from_str(denied_message).unwrap();
    assert_eq!(denied_payload["result"]["isError"], true, "{denied_payload}");

    let after_denial = kogwistar_store_sqlite::SqliteStore::open(&path).unwrap();
    assert_eq!(after_denial.get_server_run(run_id).unwrap().unwrap().status, "queued");
    let before_cancel_event_count = kogwistar_store::ServerRunReadStore::server_run_events(
        &after_denial,
        run_id,
        0,
        100,
    )
    .await
    .unwrap()
    .len();
    assert_eq!(
        before_cancel_event_count,
        before_denial_event_count,
        "denied cancel must not append a durable run event",
    );
    drop(after_denial);

    let cancelled = app.clone().oneshot(cancel(&token)).await.unwrap();
    assert_eq!(cancelled.status(), StatusCode::OK);
    let cancelled_body = to_bytes(cancelled.into_body(), 8192).await.unwrap();
    let cancelled_data = String::from_utf8(cancelled_body.to_vec()).unwrap();
    let cancelled_message = cancelled_data
        .lines()
        .find_map(|line| line.strip_prefix("data: "))
        .unwrap();
    let cancelled_payload: Value = serde_json::from_str(cancelled_message).unwrap();
    assert_eq!(
        cancelled_payload["result"]["structuredContent"]["status"],
        "cancelled"
    );

    let after_cancel = kogwistar_store_sqlite::SqliteStore::open(&path).unwrap();
    assert_eq!(after_cancel.get_server_run(run_id).unwrap().unwrap().status, "cancelled");
    let first_event_count = kogwistar_store::ServerRunReadStore::server_run_events(
        &after_cancel,
        run_id,
        0,
        100,
    )
    .await
    .unwrap();
    let cancel_contract: Value = serde_json::from_str(include_str!(
        "../../../../../contracts/golden/run-cancel-v1.json"
    ))
    .unwrap();
    assert!(first_event_count
        .iter()
        .any(|event| event.event_type == cancel_contract["nonterminal_cancel_event"]));
    let first_event_count = first_event_count.len();
    assert!(
        first_event_count > before_cancel_event_count,
        "authorized cancel must append its durable run event",
    );
    drop(after_cancel);

    let retried = app.oneshot(cancel(&token)).await.unwrap();
    assert_eq!(retried.status(), StatusCode::OK);
    let retry_body = to_bytes(retried.into_body(), 8192).await.unwrap();
    let retry_data = String::from_utf8(retry_body.to_vec()).unwrap();
    let retry_message = retry_data
        .lines()
        .find_map(|line| line.strip_prefix("data: "))
        .unwrap();
    let retry_payload: Value = serde_json::from_str(retry_message).unwrap();
    assert_eq!(
        retry_payload["result"]["structuredContent"]["status"],
        "cancelled"
    );

    let after_retry = kogwistar_store_sqlite::SqliteStore::open(&path).unwrap();
    let retry_event_count = kogwistar_store::ServerRunReadStore::server_run_events(
        &after_retry,
        run_id,
        0,
        100,
    )
    .await
    .unwrap()
    .len();
    assert_eq!(cancel_contract["terminal_retry_appends_event"], false);
    assert_eq!(retry_event_count, first_event_count, "retry must not add a duplicate event");
    drop(after_retry);
    drop(service);
    remove_test_sqlite(&path);
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
// PYTHON_PARITY_NOTE_V1: PostgreSQL REST/MCP/syscall cancellation preserves scope; denied calls leave state unchanged, authorized nonterminal termination appends run.cancelling, terminal retries add no event. rust=rust/crates/kogwistar-api/src/tests/workflow_run_routes.rs#postgres_workflow_run_cancel_contract_enforces_scope_and_write_capability; python=tests/mcp/test_chat_mcp_tools.py#test_mcp_workflow_run_submit_matches_shared_contract+test_mcp_workflow_run_submit_uses_authenticated_user_over_argument+test_mcp_workflow_run_cancel_denies_missing_write_capability+test_mcp_workflow_runtime_submit_cancel+tests/server/test_chat_server_api.py#test_conversation_syscalls_use_python_postgres_backend; fixture=contracts/golden/workflow-run-submit-mcp-v1.json,contracts/golden/run-cancel-v1.json
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
    let submit_denial_token = encode(
        &Header::new(Algorithm::HS256),
        &json!({
            "sub":"alice", "user_id":"alice", "role":"rw", "ns":["workflow"],
            "capabilities":["workflow.run.write"],
            "iss":"local", "exp":4102444800_u64
        }),
        &EncodingKey::from_secret(secret.as_bytes()),
    )
    .unwrap();
    let submit_denied = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp/workflow")
                .header("content-type", "application/json")
                .header("accept", "application/json, text/event-stream")
                .header("authorization", format!("Bearer {submit_denial_token}"))
                .body(Body::from(
                    json!({
                        "jsonrpc":"2.0", "id":"pg-submit-denied", "method":"tools/call",
                        "params":{
                            "name":"workflow.run_submit",
                            "arguments":{
                                "workflow_id":"denied-submit-workflow",
                                "conversation_id":"denied-submit-conversation",
                                "initial_state":{"counter":1}
                            }
                        }
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(submit_denied.status(), StatusCode::OK);
    let submit_denied_wire = String::from_utf8(
        to_bytes(submit_denied.into_body(), 8192).await.unwrap().to_vec(),
    )
    .unwrap();
    let submit_denied_data = submit_denied_wire
        .lines()
        .find_map(|line| line.strip_prefix("data: "))
        .expect("MCP Streamable HTTP submit denial data");
    let submit_denied_payload: Value = serde_json::from_str(submit_denied_data).unwrap();
    assert_eq!(submit_denied_payload["result"]["isError"], true);
    assert!(service
        .store
        .list_server_runs(
            None,
            Some("denied-submit-workflow"),
            Some("denied-submit-conversation"),
            20,
        )
        .await
        .unwrap()
        .is_empty());

    let submit_contract: Value = serde_json::from_str(include_str!(
        "../../../../../contracts/golden/workflow-run-submit-mcp-v1.json"
    ))
    .unwrap();
    let mut submit_arguments = submit_contract["arguments"].clone();
    submit_arguments["user_id"] = json!("caller-controlled-owner");
    let submit_token = encode(
        &Header::new(Algorithm::HS256),
        &json!({
            "sub":"alice", "user_id":"alice", "role":"rw", "ns":["workflow"],
            "capabilities":["spawn_process", "workflow.run.write"],
            "iss":"local", "exp":4102444800_u64
        }),
        &EncodingKey::from_secret(secret.as_bytes()),
    )
    .unwrap();
    let submitted = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp/workflow")
                .header("content-type", "application/json")
                .header("accept", "application/json, text/event-stream")
                .header("authorization", format!("Bearer {submit_token}"))
                .body(Body::from(
                    json!({
                        "jsonrpc":"2.0", "id":"pg-submit-accepted", "method":"tools/call",
                        "params":{
                            "name":submit_contract["tool"],
                            "arguments":submit_arguments
                        }
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(submitted.status(), StatusCode::OK);
    let submitted_wire = String::from_utf8(
        to_bytes(submitted.into_body(), 8192).await.unwrap().to_vec(),
    )
    .unwrap();
    let submitted_data = submitted_wire
        .lines()
        .find_map(|line| line.strip_prefix("data: "))
        .expect("MCP Streamable HTTP submit data");
    let submitted_payload: Value = serde_json::from_str(submitted_data).unwrap();
    assert_ne!(submitted_payload["result"]["isError"], true, "{submitted_payload}");
    let submitted_result = &submitted_payload["result"]["structuredContent"];
    let expected_result = &submit_contract["result"];
    for field in [
        "conversation_id",
        "workflow_id",
        "turn_node_id",
        "status",
        "priority_class",
        "token_budget",
        "time_budget_ms",
        "admission",
    ] {
        assert_eq!(submitted_result[field], expected_result[field], "field {field}");
    }
    let submitted_run_id = submitted_result["run_id"].as_str().unwrap();
    assert!(!submitted_run_id.is_empty());
    let persisted_submit = service
        .store
        .get_server_run(submitted_run_id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(persisted_submit.workflow_id, "workflow-contract");
    assert_eq!(persisted_submit.conversation_id, "conversation-contract");
    assert_eq!(persisted_submit.user_id.as_deref(), Some("alice"));
    assert_eq!(persisted_submit.status, "queued");
    let submit_events = kogwistar_store::ServerRunReadStore::server_run_events(
        &service.store,
        submitted_run_id,
        0,
        100,
    )
    .await
    .unwrap();
    let created_event = submit_events.first().expect("durable run.created event");
    assert_eq!(created_event.seq, 1);
    assert_eq!(created_event.event_type, "run.created");
    let created_payload: Value = serde_json::from_str(&created_event.payload_json).unwrap();
    assert_eq!(created_payload["run_id"], submitted_run_id);
    assert_eq!(created_payload["workflow_id"], "workflow-contract");
    assert_eq!(created_payload["conversation_id"], "conversation-contract");
    assert_eq!(created_payload["status"], "queued");

    let mcp_denied = app
        .clone()
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

    let first_cancel_events = kogwistar_store::ServerRunReadStore::server_run_events(
        &service.store,
        "cancel-auth-run",
        0,
        100,
    )
    .await
    .unwrap();
    assert_eq!(first_cancel_events.len(), 1);
    assert_eq!(
        first_cancel_events[0].event_type,
        contract["nonterminal_cancel_event"]
    );

    let event_count_before_retry = kogwistar_store::ServerRunReadStore::server_run_events(
        &service.store,
        "cancel-auth-run",
        0,
        100,
    )
    .await
    .unwrap()
    .len();
    let retry_token = encode(
        &Header::new(Algorithm::HS256),
        &json!({
            "sub":"alice", "role":"rw", "ns":["workflow"],
            "capabilities":["workflow.run.write"], "iss":"local", "exp":4102444800_u64
        }),
        &EncodingKey::from_secret(secret.as_bytes()),
    )
    .unwrap();
    let mcp_retry = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp/workflow")
                .header("content-type", "application/json")
                .header("accept", "application/json, text/event-stream")
                .header("authorization", format!("Bearer {retry_token}"))
                .body(Body::from(
                    r#"{"jsonrpc":"2.0","id":"cancel-retry","method":"tools/call","params":{"name":"workflow.run_cancel","arguments":{"run_id":"cancel-auth-run"}}}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(mcp_retry.status(), StatusCode::OK);
    let retry_wire = String::from_utf8(
        to_bytes(mcp_retry.into_body(), 8192).await.unwrap().to_vec(),
    )
    .unwrap();
    let retry_data = retry_wire
        .lines()
        .find_map(|line| line.strip_prefix("data: "))
        .expect("MCP Streamable HTTP retry result data");
    let retry_value: Value = serde_json::from_str(retry_data).unwrap();
    assert_ne!(retry_value["result"]["isError"], true, "{retry_value}");
    let retry_result = &retry_value["result"]["structuredContent"];
    for field in contract["required_fields"].as_array().unwrap() {
        let field = field.as_str().unwrap();
        if field != "updated_at_ms" {
            assert_eq!(retry_result[field], body[field], "retry field {field}");
        }
    }
    assert!(
        retry_result["updated_at_ms"].as_i64().unwrap()
            >= body["updated_at_ms"].as_i64().unwrap()
    );
    let event_count_after_retry = kogwistar_store::ServerRunReadStore::server_run_events(
        &service.store,
        "cancel-auth-run",
        0,
        100,
    )
    .await
    .unwrap()
    .len();
    assert_eq!(event_count_after_retry, event_count_before_retry + 1);
    let retry_events = kogwistar_store::ServerRunReadStore::server_run_events(
        &service.store,
        "cancel-auth-run",
        0,
        100,
    )
    .await
    .unwrap();
    assert_eq!(
        retry_events.last().unwrap().event_type,
        contract["nonterminal_cancel_event"]
    );

    let syscall_principal = json!({
        "sub":"alice",
        "role":"rw",
        "ns":["workflow"],
        "capabilities":["spawn_process", "workflow.run.write"]
    });
    let spawned = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/syscall/v1/spawn_process".to_owned(),
            body: serde_json::to_vec(&json!({
                "version":"v1",
                "op":"spawn_process",
                "args":{
                    "workflow_id":"postgres-syscall-lifecycle",
                    "conversation_id":"postgres-syscall-conversation",
                    "user_id":"alice",
                    "initial_state":{"source":"postgres-syscall"}
                }
            }))
            .unwrap(),
            principal: syscall_principal.clone(),
        })
        .await;
    assert_eq!(spawned.status, StatusCode::OK.as_u16());
    let spawned: Value = serde_json::from_slice(&spawned.body).unwrap();
    assert_eq!(spawned["version"], "v1");
    assert_eq!(spawned["op"], "spawn_process");
    assert_eq!(spawned["status"], "ok");
    let submit_contract: Value = serde_json::from_str(include_str!(
        "../../../../../contracts/golden/workflow-run-submit-http-v1.json"
    ))
    .unwrap();
    let spawned_result = &spawned["result"];
    assert!(submit_contract["required_common_fields"]
        .as_array()
        .unwrap()
        .iter()
        .all(|field| spawned_result.get(field.as_str().unwrap()).is_some()));
    for (field, value) in submit_contract["expected_common_values"].as_object().unwrap() {
        assert_eq!(&spawned_result[field], value, "field {field}");
    }
    let syscall_run_id = spawned["result"]["run_id"].as_str().unwrap();
    assert!(service
        .store
        .get_server_run(syscall_run_id)
        .await
        .unwrap()
        .is_some());
    let terminate_errors: Value = serde_json::from_str(include_str!(
        "../../../../../contracts/golden/syscall-terminate-errors-v1.json"
    ))
    .unwrap();
    for (args, expected_status) in [
        (
            json!({}),
            terminate_errors["missing_run_id_status"].as_u64().unwrap() as u16,
        ),
        (
            json!({"run_id":"unknown-postgres-run"}),
            terminate_errors["unknown_run_id_status"].as_u64().unwrap() as u16,
        ),
    ] {
        let invalid_termination = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/syscall/v1/terminate_process".to_owned(),
                body: serde_json::to_vec(&json!({"version":"v1", "op":"terminate_process", "args":args}))
                    .unwrap(),
                principal: syscall_principal.clone(),
            })
            .await;
        assert_eq!(invalid_termination.status, expected_status);
    }
    assert!(!service
        .store
        .get_server_run(syscall_run_id)
        .await
        .unwrap()
        .unwrap()
        .cancel_requested);
    let resume_errors: Value = serde_json::from_str(include_str!(
        "../../../../../contracts/golden/syscall-resume-errors-v1.json"
    ))
    .unwrap();
    for (args, expected_status) in [
        (json!({}), resume_errors["missing_run_id_status"].as_u64().unwrap() as u16),
        (
            json!({
                "run_id":"unknown-postgres-run", "workflow_id":"unknown-workflow",
                "conversation_id":"unknown-conversation", "turn_node_id":"unknown-turn",
                "suspended_node_id":"unknown-node", "suspended_token_id":"unknown-token",
                "client_result":{}
            }),
            resume_errors["unknown_run_id_status"].as_u64().unwrap() as u16,
        ),
    ] {
        let invalid_resume = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/syscall/v1/resume".to_owned(),
                body: serde_json::to_vec(&json!({"version":"v1", "op":"resume", "args":args}))
                    .unwrap(),
                principal: syscall_principal.clone(),
            })
            .await;
        assert_eq!(invalid_resume.status, expected_status);
    }
    assert!(!service
        .store
        .get_server_run(syscall_run_id)
        .await
        .unwrap()
        .unwrap()
        .cancel_requested);

    let cancel_contract: Value = serde_json::from_str(include_str!(
        "../../../../../contracts/golden/run-cancel-v1.json"
    ))
    .unwrap();
    let events_before_termination = kogwistar_store::ServerRunReadStore::server_run_events(
        &service.store,
        syscall_run_id,
        0,
        100,
    )
    .await
    .unwrap();
    let denied_termination = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/syscall/v1/terminate_process".to_owned(),
            body: serde_json::to_vec(&json!({
                "version":"v1",
                "op":"terminate_process",
                "args":{"run_id":syscall_run_id}
            }))
            .unwrap(),
            principal: json!({
                "sub":"alice", "role":"rw", "ns":["workflow"],
                "capabilities":["workflow.run.read"]
            }),
        })
        .await;
    assert_eq!(denied_termination.status, StatusCode::FORBIDDEN.as_u16());
    assert!(!service
        .store
        .get_server_run(syscall_run_id)
        .await
        .unwrap()
        .unwrap()
        .cancel_requested);

    let terminated = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/syscall/v1/terminate_process".to_owned(),
            body: serde_json::to_vec(&json!({
                "version":"v1",
                "op":"terminate_process",
                "args":{"run_id":syscall_run_id}
            }))
            .unwrap(),
            principal: syscall_principal.clone(),
        })
        .await;
    assert_eq!(terminated.status, StatusCode::OK.as_u16());
    let terminated: Value = serde_json::from_slice(&terminated.body).unwrap();
    assert_eq!(terminated["version"], "v1");
    assert_eq!(terminated["op"], "terminate_process");
    assert_eq!(terminated["status"], "ok");
    for field in cancel_contract["required_fields"].as_array().unwrap() {
        let field = field.as_str().unwrap();
        assert!(terminated["result"].get(field).is_some(), "missing {field}");
    }
    assert!(cancel_contract["accepted_run_statuses"]
        .as_array()
        .unwrap()
        .contains(&terminated["result"]["status"]));
    assert_eq!(terminated["result"]["cancel_requested"], true);
    let events_after_termination = kogwistar_store::ServerRunReadStore::server_run_events(
        &service.store,
        syscall_run_id,
        0,
        100,
    )
    .await
    .unwrap();
    let cancel_event = cancel_contract["nonterminal_cancel_event"]
        .as_str()
        .unwrap();
    let count_cancel_events = |events: &[kogwistar_store::ServerRunEvent]| {
        events
            .iter()
            .filter(|event| event.event_type == cancel_event)
            .count()
    };
    assert_eq!(
        count_cancel_events(&events_after_termination),
        count_cancel_events(&events_before_termination) + 1
    );
    assert!(service
        .store
        .get_server_run(syscall_run_id)
        .await
        .unwrap()
        .unwrap()
        .cancel_requested);
    let repeated_termination = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/syscall/v1/terminate_process".to_owned(),
            body: serde_json::to_vec(&json!({
                "version":"v1",
                "op":"terminate_process",
                "args":{"run_id":syscall_run_id}
            }))
            .unwrap(),
            principal: syscall_principal,
        })
        .await;
    assert_eq!(repeated_termination.status, StatusCode::OK.as_u16());
    let repeated_termination: Value =
        serde_json::from_slice(&repeated_termination.body).unwrap();
    assert!(cancel_contract["accepted_run_statuses"]
        .as_array()
        .unwrap()
        .contains(&repeated_termination["result"]["status"]));
    assert_eq!(repeated_termination["result"]["cancel_requested"], true);
    let events_after_terminal_retry = kogwistar_store::ServerRunReadStore::server_run_events(
        &service.store,
        syscall_run_id,
        0,
        100,
    )
    .await
    .unwrap();
    assert_eq!(
        count_cancel_events(&events_after_terminal_retry),
        count_cancel_events(&events_after_termination)
    );
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

#[tokio::test]
#[ignore = "requires KOGWISTAR_TEST_PG_DSN; uses a disposable PostgreSQL schema"]
async fn postgres_dead_letter_replay_mcp_is_single_claim_under_concurrency() {
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use tower::ServiceExt;

    let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
        .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
    let schema = format!("rust_dead_replay_{}", uuid::Uuid::new_v4().simple());
    let service = PostgresRunApplicationService::from_dsn(&dsn, &schema).unwrap();
    service.ensure_graph_type_schemas().await.unwrap();
    service
        .store
        .project_lane_message(NewProjectedLaneMessage {
            message_id: "dead-letter-concurrent".to_owned(),
            namespace: "workflow".to_owned(),
            purpose: "system".to_owned(),
            inbox_id: "workflow-runtime".to_owned(),
            conversation_id: "conversation-dead-replay".to_owned(),
            recipient_id: "worker".to_owned(),
            sender_id: "scheduler".to_owned(),
            msg_type: "workflow.run.execute".to_owned(),
            status: "dead-letter".to_owned(),
            created_at: 1,
            available_at: 1,
            run_id: Some("dead-replay-run".to_owned()),
            step_id: None,
            correlation_id: None,
            payload_json: None,
            error_json: Some("failed".to_owned()),
        })
        .await
        .unwrap();

    let secret = "postgres-dead-replay-test-secret";
    let token = encode(
        &Header::new(Algorithm::HS256),
        &json!({
            "sub":"alice", "role":"rw", "ns":["workflow"],
            "capabilities":["service.manage"], "iss":"local", "exp":4102444800_u64
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
    let request = |id: &str| {
        Request::builder()
            .method("POST")
            .uri("/mcp/workflow")
            .header("content-type", "application/json")
            .header("authorization", format!("Bearer {token}"))
            .body(Body::from(
                json!({
                    "jsonrpc":"2.0", "id":id, "method":"tools/call",
                    "params":{"name":"workflow.dead_letter_replay", "arguments":{"run_id":"dead-replay-run"}}
                })
                .to_string(),
            ))
            .unwrap()
    };
    let (first, second) = tokio::join!(
        app.clone().oneshot(request("replay-a")),
        app.clone().oneshot(request("replay-b")),
    );
    let first = serde_json::from_slice::<Value>(
        &to_bytes(first.unwrap().into_body(), 16_384).await.unwrap(),
    )
    .unwrap();
    let second = serde_json::from_slice::<Value>(
        &to_bytes(second.unwrap().into_body(), 16_384).await.unwrap(),
    )
    .unwrap();
    let mut outcomes = [
        first["result"]["replayed"].as_bool().unwrap(),
        second["result"]["replayed"].as_bool().unwrap(),
    ];
    outcomes.sort_unstable();
    assert_eq!(outcomes, [false, true]);
    let lane = service
        .store
        .get_projected_lane_message("dead-letter-concurrent")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(lane.status, "pending");

    drop(app);
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
}
