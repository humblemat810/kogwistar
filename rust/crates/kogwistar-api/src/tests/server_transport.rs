    #[test]
    fn frozen_route_registry_covers_committed_openapi() {
        assert_eq!(FROZEN_OPENAPI_ROUTES.len(), 84);
        assert!(FROZEN_OPENAPI_ROUTES.contains(&("GET", "/health")));
        assert!(FROZEN_OPENAPI_ROUTES.contains(&("POST", "/api/workflow/runs")));
        assert!(FROZEN_OPENAPI_ROUTES.contains(&("GET", "/api/runs/{run_id}/events")));
        assert!(FROZEN_OPENAPI_ROUTES.contains(&("GET", "/api/runs/{run_id}/evidence")));
        let embedded: Value = serde_json::from_str(FROZEN_OPENAPI_JSON).unwrap();
        assert_eq!(embedded["paths"].as_object().unwrap().len(), 82);
    }

    #[test]
    fn sqlite_application_route_inventory_has_no_unclassified_cutover_drift() {
        let path = std::env::temp_dir().join(format!(
            "kogwistar-api-route-inventory-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let fixture_store = kogwistar_store_sqlite::SqliteStore::open(&path).unwrap();
        let mut run_ids = std::collections::BTreeMap::new();
        for (index, &(_, template)) in FROZEN_OPENAPI_ROUTES.iter().enumerate() {
            if template.contains("{run_id}") {
                let run_id = format!("inventory-run-{index}");
                fixture_store
                    .create_server_run(kogwistar_store::ServerRunCreate {
                        run_id: run_id.clone(),
                        conversation_id: format!("inventory-conversation-{index}"),
                        workflow_id: format!("inventory-workflow-{index}"),
                        user_id: Some("inventory-admin".to_owned()),
                        user_turn_node_id: format!("inventory-turn-{index}"),
                        status: "running".to_owned(),
                    })
                    .unwrap();
                fixture_store
                    .append_server_run_event(
                        &run_id,
                        "run.started",
                        r#"{"source":"route-inventory"}"#.to_owned(),
                    )
                    .unwrap();
                run_ids.insert(index, run_id);
            }
        }
        drop(fixture_store);
        let service = SqliteRunApplicationService::open(&path).unwrap();
        let mut unavailable = Vec::new();
        for (index, &(method, template)) in FROZEN_OPENAPI_ROUTES.iter().enumerate() {
            if TRANSPORT_OWNED_HTTP_ROUTES.contains(&(method, template)) {
                continue;
            }
            let path_and_query = template
                .replace(
                    "{run_id}",
                    run_ids
                        .get(&index)
                        .map(String::as_str)
                        .unwrap_or("inventory-run"),
                )
                .replace("{conversation_id}", "inventory-conversation")
                .replace("{workflow_id}", "inventory-workflow")
                .replace("{service_id}", "inventory-service")
                .replace("{step_seq}", "0")
                .replace("{node_id}", "inventory-node")
                .replace("{edge_id}", "inventory-edge")
                .replace("{doc_id}", "inventory-document")
                .replace("{op}", "checkpoint");
            let response = service.execute_sync(ApiEffectRequest {
                contract_version: 1,
                method: method.to_owned(),
                path_and_query,
                body: b"{}".to_vec(),
                principal: json!({
                    "sub": "inventory-admin",
                    "role": "admin",
                    "capabilities": ["*"]
                }),
            });
            if response.status == StatusCode::NOT_IMPLEMENTED.as_u16() {
                unavailable.push(format!("{method} {template}"));
            }
        }
        drop(service);
        let _ = std::fs::remove_file(&path);
        let mut expected_unavailable = PENDING_SERVER_CUTOVER_ROUTES
            .iter()
            .map(|(method, path)| format!("{method} {path}"))
            .chain(
                SQLITE_UNSUPPORTED_HTTP_ROUTES
                    .iter()
                    .map(|(method, path)| format!("{method} {path}")),
            )
            .collect::<Vec<_>>();
        unavailable.sort();
        expected_unavailable.sort();
        assert_eq!(unavailable, expected_unavailable);
    }

    #[tokio::test]
    async fn sqlite_graph_routes_fail_closed_with_explicit_authority_status() {
        let path = std::env::temp_dir().join(format!(
            "kogwistar-api-graph-authority-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let service = open_sqlite_application_service(&path, None).await.unwrap();
        for route in [
            "/internal/graph/neighbors",
            "/internal/graph/vector-query",
            "/internal/graph/mutations",
        ] {
            let response = service.execute_sync(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: route.to_owned(),
                body: br#"{}"#.to_vec(),
                principal: json!({"sub":"alice","role":"rw","capabilities":["*"]}),
            });
            assert_eq!(response.status, StatusCode::NOT_IMPLEMENTED.as_u16());
            let body: Value = serde_json::from_slice(&response.body).unwrap();
            assert_eq!(body["code"], "KOGWISTAR_GRAPH_AUTHORITY_UNSUPPORTED");
        }
        let worker_graph_effect = service.execute_sync(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/internal/runtime/results".to_owned(),
            body: serde_json::to_vec(&json!({
                "handoff": {
                    "message_id": "lane-1",
                    "claimed_by": "worker-1",
                    "run_id": "run-1",
                    "step_id": "step-1",
                    "correlation_id": "run-1"
                },
                "effect": {
                    "contract_version": 1,
                    "effect_id": "effect-1",
                    "graph_mutation_batch": {
                        "contract_version": 1,
                        "atomic": true,
                        "mutations": [{
                            "scope": {"namespace": "alpha"},
                            "table": "gke_nodes",
                            "entity_kind": "node",
                            "event_id": "event-1",
                            "op": "ADD",
                            "payload": {"id": "node-1"},
                            "record": {"id": "node-1"},
                            "embedding_dim": 0
                        }]
                    }
                }
            }))
            .unwrap(),
            principal: json!({"sub":"alice","role":"rw","capabilities":["*"]}),
        });
        assert_eq!(
            worker_graph_effect.status,
            StatusCode::NOT_IMPLEMENTED.as_u16()
        );
        let worker_graph_body: Value =
            serde_json::from_slice(&worker_graph_effect.body).unwrap();
        assert_eq!(
            worker_graph_body["code"],
            "KOGWISTAR_GRAPH_AUTHORITY_UNSUPPORTED"
        );
        let worker_event_ack = service.execute_sync(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/internal/runtime/results".to_owned(),
            body: serde_json::to_vec(&json!({
                "handoff": {
                    "message_id": "lane-1",
                    "claimed_by": "worker-1",
                    "run_id": "run-1",
                    "step_id": "step-1",
                    "correlation_id": "run-1"
                },
                "effect": {
                    "contract_version": 1,
                    "effect_id": "effect-event-ack",
                    "provider_event_acknowledgements": [{
                        "message_id": "msg:event-1",
                        "claimed_by": "event-worker#lease-1",
                        "event_id": "event-1"
                    }]
                }
            }))
            .unwrap(),
            principal: json!({"sub":"alice","role":"rw","capabilities":["*"]}),
        });
        assert_eq!(
            worker_event_ack.status,
            StatusCode::NOT_IMPLEMENTED.as_u16()
        );
        let worker_event_ack_body: Value =
            serde_json::from_slice(&worker_event_ack.body).unwrap();
        assert_eq!(
            worker_event_ack_body["code"],
            "KOGWISTAR_GRAPH_AUTHORITY_UNSUPPORTED"
        );
        drop(service);
        remove_test_sqlite(&path);
    }

    #[tokio::test]
    async fn router_serves_health_auth_sse_and_mcp() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use tower::ServiceExt;

        let app = router(ApiState {
            health: snapshot(),
            required_roles: vec!["reader".to_owned()],
            implementation: ImplementationSnapshot::default(),
            auth: AuthConfig::default(),
        });
        let health = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let health_contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/health-v1.json"
        ))
        .unwrap();
        assert_eq!(
            health.status().as_u16(),
            health_contract["success_status"].as_u64().unwrap() as u16
        );
        let health_body = to_bytes(health.into_body(), 4096).await.unwrap();
        let health_value: Value = serde_json::from_slice(&health_body).unwrap();
        assert_eq!(health_value["ok"], health_contract["ok"]);
        for field in health_contract["required_fields"].as_array().unwrap() {
            assert!(health_value.get(field.as_str().unwrap()).is_some());
        }
        assert_eq!(
            health_value["backend"],
            "pg"
        );
        assert!(health_value["embedding_inline_worker_enabled"].is_boolean());
        assert_eq!(
            health_value["implementation"]["runtime_cutover_ready"],
            true
        );
        assert_eq!(
            health_value["implementation"]["server_cutover_ready"],
            false
        );
        assert_eq!(
            health_value["implementation"]["effective_server_cutover_ready"],
            false
        );
        assert_eq!(health_value["graph_authority"]["level"], "unsupported");
        assert_eq!(health_value["graph_authority"]["single_mutation"], false);
        assert_eq!(health_value["document_authority"]["level"], "unsupported");
        assert_eq!(
            health_value["document_authority"]["projection_upsert"],
            false
        );

        let openapi = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/openapi.json")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(openapi.status(), StatusCode::OK);
        let openapi_body = to_bytes(openapi.into_body(), 2_000_000).await.unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&openapi_body).unwrap(),
            serde_json::from_str::<Value>(FROZEN_OPENAPI_JSON).unwrap()
        );

        let forbidden = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/events")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(forbidden.status(), StatusCode::FORBIDDEN);

        let events = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/events?event=run&id=7")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(events.status(), StatusCode::OK);
        assert_eq!(
            events.headers()["content-type"],
            "text/event-stream; charset=utf-8"
        );

        let mcp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(mcp.status(), StatusCode::OK);
        let mcp_body = to_bytes(mcp.into_body(), 65_536).await.unwrap();
        let mcp_value = serde_json::from_slice::<Value>(&mcp_body).unwrap();
        assert_eq!(mcp_value["result"]["tools"].as_array().unwrap().len(), 0);

        let initialize = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":10,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"contract-test","version":"1"}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(initialize.status(), StatusCode::OK);
        assert_eq!(initialize.headers()["mcp-protocol-version"], "2025-06-18");
        let session_id = initialize.headers()["mcp-session-id"]
            .to_str()
            .unwrap()
            .to_owned();
        let initialize_body = to_bytes(initialize.into_body(), 4096).await.unwrap();
        let initialize_value: Value = serde_json::from_slice(&initialize_body).unwrap();
        assert_eq!(initialize_value["result"]["protocolVersion"], "2025-06-18");
        assert_eq!(
            initialize_value["result"]["capabilities"]["tools"]["listChanged"],
            false
        );
        assert_eq!(initialize_value["result"]["capabilities"]["logging"], json!({}));

        let invalid_initialize = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":17,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(invalid_initialize.status(), StatusCode::BAD_REQUEST);
        assert!(!invalid_initialize.headers().contains_key("mcp-session-id"));

        let initialized = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header("mcp-protocol-version", "2025-06-18")
                    .header("mcp-session-id", &session_id)
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(initialized.status(), StatusCode::ACCEPTED);

        let ping_sse = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header("accept", "application/json, text/event-stream")
                    .header("mcp-protocol-version", "2025-06-18")
                    .header("mcp-session-id", &session_id)
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(r#"{"jsonrpc":"2.0","id":11,"method":"ping"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(ping_sse.status(), StatusCode::OK);
        assert_eq!(
            ping_sse.headers()["content-type"],
            "text/event-stream; charset=utf-8"
        );
        assert_eq!(ping_sse.headers()["mcp-session-id"], session_id);
        assert_eq!(ping_sse.headers()["mcp-protocol-version"], "2025-06-18");
        let ping_sse_body = to_bytes(ping_sse.into_body(), 4096).await.unwrap();
        assert!(String::from_utf8(ping_sse_body.to_vec())
            .unwrap()
            .contains("data: {\"jsonrpc\":\"2.0\",\"id\":11,\"result\":{}}"));

        let other_session_owner = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header("mcp-protocol-version", "2025-06-18")
                    .header("mcp-session-id", &session_id)
                    .header("x-kogwistar-roles", "reader,writer")
                    .body(Body::from(r#"{"jsonrpc":"2.0","id":15,"method":"ping"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(other_session_owner.status(), StatusCode::NOT_FOUND);

        let owner_after_foreign_probe = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header("mcp-protocol-version", "2025-06-18")
                    .header("mcp-session-id", &session_id)
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(r#"{"jsonrpc":"2.0","id":18,"method":"ping"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(owner_after_foreign_probe.status(), StatusCode::OK);

        let rejected_accept = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header("accept", "text/plain")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(r#"{"jsonrpc":"2.0","id":12,"method":"ping"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(rejected_accept.status(), StatusCode::NOT_ACCEPTABLE);

        let unsupported_protocol = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header("mcp-protocol-version", "2026-07-28")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(r#"{"jsonrpc":"2.0","id":13,"method":"ping"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(unsupported_protocol.status(), StatusCode::BAD_REQUEST);

        let unknown_session = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header("mcp-session-id", "not-issued-by-this-server")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(r#"{"jsonrpc":"2.0","id":14,"method":"ping"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(unknown_session.status(), StatusCode::NOT_FOUND);

        let terminated_session = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("DELETE")
                    .uri("/mcp")
                    .header("mcp-session-id", &session_id)
                    .header("mcp-protocol-version", "2025-06-18")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(terminated_session.status(), StatusCode::OK);
        let request_after_termination = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header("mcp-protocol-version", "2025-06-18")
                    .header("mcp-session-id", &session_id)
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(r#"{"jsonrpc":"2.0","id":16,"method":"ping"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(request_after_termination.status(), StatusCode::NOT_FOUND);

        let unadvertised_mcp_call = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":"unadvertised","method":"tools/call","params":{"name":"kg_neighbors","arguments":{"rid":"node-1"}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(unadvertised_mcp_call.status(), StatusCode::OK);
        let unadvertised_body = to_bytes(unadvertised_mcp_call.into_body(), 65_536)
            .await
            .unwrap();
        let unadvertised_value = serde_json::from_slice::<Value>(&unadvertised_body).unwrap();
        assert_eq!(unadvertised_value["error"]["code"], -32602);

        let wildcard_accept_mcp_call = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header("accept", "*/*")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":"wildcard","method":"tools/call","params":{"name":"kg_neighbors","arguments":{"rid":"node-1"}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(wildcard_accept_mcp_call.status(), StatusCode::OK);
        let wildcard_body = to_bytes(wildcard_accept_mcp_call.into_body(), 65_536)
            .await
            .unwrap();
        let wildcard_value = serde_json::from_slice::<Value>(&wildcard_body).unwrap();
        assert_eq!(wildcard_value["error"]["code"], -32602);

        let negotiated_tool_error = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header("accept", "application/json, text/event-stream")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":"negotiated-error","method":"tools/call","params":{"name":"kg_neighbors","arguments":{"rid":"node-1"}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(negotiated_tool_error.status(), StatusCode::OK);
        assert_eq!(
            negotiated_tool_error
                .headers()
                .get("content-type")
                .unwrap(),
            "text/event-stream; charset=utf-8"
        );
        let negotiated_body = to_bytes(negotiated_tool_error.into_body(), 65_536)
            .await
            .unwrap();
        let negotiated_sse = std::str::from_utf8(&negotiated_body).unwrap();
        let data = negotiated_sse
            .lines()
            .find_map(|line| line.strip_prefix("data: "))
            .unwrap();
        let negotiated_value = serde_json::from_str::<Value>(data).unwrap();
        assert_eq!(negotiated_value["id"], "negotiated-error");
        assert!(negotiated_value["result"]["isError"].as_bool().unwrap());
        assert_eq!(negotiated_value["result"]["content"][0]["type"], "text");
        assert_eq!(
            negotiated_value["result"]["content"][0]["text"],
            "Tool not advertised by this application service"
        );
        assert!(negotiated_value.get("error").is_none());

        let workflow_mcp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/workflow")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(workflow_mcp.status(), StatusCode::OK);
        let workflow_body = to_bytes(workflow_mcp.into_body(), 65_536).await.unwrap();
        let workflow_value = serde_json::from_slice::<Value>(&workflow_body).unwrap();
        assert_eq!(
            workflow_value["result"]["tools"].as_array().unwrap().len(),
            0
        );

        let conversation_mcp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/conversation")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":3,"method":"tools/list"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(conversation_mcp.status(), StatusCode::OK);
        let conversation_body = to_bytes(conversation_mcp.into_body(), 65_536)
            .await
            .unwrap();
        let conversation_value = serde_json::from_slice::<Value>(&conversation_body).unwrap();
        assert_eq!(
            conversation_value["result"]["tools"]
                .as_array()
                .unwrap()
                .len(),
            0
        );

        let pending_cutover = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/workflow/runs")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(pending_cutover.status(), StatusCode::NOT_IMPLEMENTED);
    }
