    #[tokio::test]
    #[ignore = "requires KOGWISTAR_TEST_PG_DSN; run explicitly with --include-ignored"]
    async fn postgres_native_runtime_route_uses_shared_result_reducer_when_dsn_available() {
        let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
            .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
        let schema = format!("rust_native_runtime_{}", uuid::Uuid::new_v4().simple());
        let service = PostgresRunApplicationService::from_dsn(&dsn, &schema).unwrap();
        service.ensure_schema().await.unwrap();

        let request = |method: &str, path: &str, body: Value| ApiEffectRequest {
            contract_version: 1,
            method: method.to_owned(),
            path_and_query: path.to_owned(),
            body: serde_json::to_vec(&body).unwrap(),
            principal: json!({
                "sub": "native-worker",
                "role": "rw",
                "ns": "workflow",
                "capabilities": ["workflow.run.write"]
            }),
        };
        let submit = service
            .execute(request(
                "POST",
                "/api/workflow/runs",
                json!({
                    "run_id": "pg-native-run",
                    "workflow_id": "pg-native-workflow",
                    "conversation_id": "pg-native-conversation",
                    "initial_state": {"seed": 1},
                    "runtime_kind": "sync",
                    "start_node_id": "entry",
                    "node_ops": {"entry": "rust-native/emit-result"}
                }),
            ))
            .await;
        assert_eq!(submit.status, StatusCode::ACCEPTED.as_u16());

        let claim = service
            .execute(request(
                "POST",
                "/internal/runtime/claim",
                json!({
                    "claimed_by": "rust-native-worker",
                    "limit": 1,
                    "lease_seconds": 1,
                    "recipient_ids": ["rust-native-worker"]
                }),
            ))
            .await;
        assert_eq!(claim.status, StatusCode::OK.as_u16());
        let work = serde_json::from_slice::<Value>(&claim.body).unwrap()["work"][0].clone();
        assert_eq!(work["payload"]["op"], "rust-native/emit-result");
        assert_eq!(
            work["payload"]["executor_target"],
            kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET
        );

        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        let reclaimed = service
            .execute(request(
                "POST",
                "/internal/runtime/claim",
                json!({
                    "claimed_by": "rust-native-worker",
                    "limit": 1,
                    "lease_seconds": 60,
                    "recipient_ids": ["rust-native-worker"]
                }),
            ))
            .await;
        assert_eq!(reclaimed.status, StatusCode::OK.as_u16());
        let reclaimed_work = serde_json::from_slice::<Value>(&reclaimed.body).unwrap()["work"][0]
            .clone();
        assert_eq!(reclaimed_work["message_id"], work["message_id"]);
        assert_ne!(reclaimed_work["claimed_by"], work["claimed_by"]);

        let stale_execute = service
            .execute(request(
                "POST",
                "/internal/runtime/native/execute",
                json!({
                    "handoff": {
                        "message_id": work["message_id"],
                        "claimed_by": work["claimed_by"],
                        "run_id": work["run_id"],
                        "step_id": work["step_id"],
                        "correlation_id": work["correlation_id"]
                    },
                    "payload": work["payload"]
                }),
            ))
            .await;
        assert_eq!(stale_execute.status, StatusCode::CONFLICT);

        let execute = service
            .execute(request(
                "POST",
                "/internal/runtime/native/execute",
                json!({
                    "handoff": {
                        "message_id": reclaimed_work["message_id"],
                        "claimed_by": reclaimed_work["claimed_by"],
                        "run_id": reclaimed_work["run_id"],
                        "step_id": reclaimed_work["step_id"],
                        "correlation_id": reclaimed_work["correlation_id"]
                    },
                    "payload": reclaimed_work["payload"]
                }),
            ))
            .await;
        assert_eq!(execute.status, StatusCode::OK.as_u16());
        let events = service
            .store
            .list_server_run_events("pg-native-run", 0, 100)
            .await
            .unwrap();
        assert!(events.iter().any(|event| {
            event.event_type == "workflow.trace.v1"
                && event.payload_json.contains("runtime.native.step.completed")
        }));

        drop(service);
        let cleanup_dsn = dsn
            .strip_prefix("postgresql+psycopg://")
            .map(|rest| format!("postgresql://{rest}"))
            .or_else(|| {
                dsn.strip_prefix("postgresql+psycopg2://")
                    .map(|rest| format!("postgresql://{rest}"))
            })
            .unwrap_or(dsn);
        let (client, connection) = tokio_postgres::connect(&cleanup_dsn, tokio_postgres::NoTls)
            .await
            .unwrap();
        let connection_task = tokio::spawn(connection);
        client
            .batch_execute(&format!("DROP SCHEMA \"{schema}\" CASCADE"))
            .await
            .unwrap();
        drop(client);
        connection_task.await.unwrap().unwrap();
    }

    #[test]
    fn delegated_capabilities_are_attenuated_and_reserved_state_wins() {
        let shared = Arc::new(Mutex::new(CapabilityState::default()));
        let principal = json!({
            "sub": "operator",
            "role": "rw",
            "ns": "workflow",
            "capabilities": ["workflow.run.write", "read_graph"]
        });
        let args = json!({
            "workflow_id": "wf",
            "conversation_id": "conv",
            "initial_state": {
                "_rt_requested_capabilities": ["workflow.run.write"],
                "_rt_effective_capabilities": ["service.manage"]
            }
        });
        let delegated = prepare_delegated_run_args(&args, &principal, &shared).unwrap();
        assert_eq!(
            delegated["initial_state"]["_rt_effective_capabilities"],
            json!(["workflow.run.write"])
        );
        assert_eq!(delegated["initial_state"]["_rt_delegated_by"], "operator");
        assert_eq!(
            delegated["initial_state"]["_rt_delegation"]["schema_version"],
            1
        );
        let widening = json!({
            "workflow_id": "wf",
            "conversation_id": "conv",
            "initial_state": {
                "_rt_requested_capabilities": ["service.manage"]
            }
        });
        assert!(prepare_delegated_run_args(&widening, &principal, &shared).is_err());
    }

    #[test]
    fn auth_sse_mcp_and_cli_contracts_are_versioned_shapes() {
        assert!(authorize(&ApiAuthRequest {
            roles: vec!["admin".to_owned()],
            required_roles: vec!["admin".to_owned()],
        }));
        assert_eq!(
            sse_frame("run", &json!({"ok": true}), Some("7")),
            "id: 7\nevent: run\ndata: {\"ok\":true}\n\n"
        );
        assert_eq!(mcp_result(json!(1), json!({"ok": true}))["jsonrpc"], "2.0");
        assert_eq!(cli_health(&snapshot()), "ok backend=pg");
    }

    #[test]
    fn jwt_auth_verifies_signature_claims_expiry_and_role_lattice() {
        use axum::http::HeaderValue;
        use jsonwebtoken::{EncodingKey, Header, encode};
        use std::time::{SystemTime, UNIX_EPOCH};

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let config = AuthConfig {
            mode: "oidc".to_owned(),
            algorithm: Some("HS256".to_owned()),
            key: Some("test-secret".to_owned()),
            jwks_json: None,
            issuer: Some("issuer".to_owned()),
            audience: Some("audience".to_owned()),
            oidc_providers_json: None,
            auth_db_url: None,
        };
        let token = encode(
            &Header::new(Algorithm::HS256),
            &json!({
                "sub": "user-1",
                "role": "rw",
                "iss": "issuer",
                "aud": "audience",
                "exp": now + 300,
            }),
            &EncodingKey::from_secret(b"test-secret"),
        )
        .unwrap();
        let mut headers = HeaderMap::new();
        headers.insert(
            "authorization",
            HeaderValue::from_str(&format!("Bearer {token}")).unwrap(),
        );
        assert_eq!(jwt_roles(&headers, &config).unwrap(), vec!["rw"]);
        assert!(has_required_role(&headers, &["ro".to_owned()], &config));

        let expired = encode(
            &Header::new(Algorithm::HS256),
            &json!({
                "role": "rw",
                "iss": "issuer",
                "aud": "audience",
                "exp": now - 300,
            }),
            &EncodingKey::from_secret(b"test-secret"),
        )
        .unwrap();
        headers.insert(
            "authorization",
            HeaderValue::from_str(&format!("Bearer {expired}")).unwrap(),
        );
        assert!(jwt_roles(&headers, &config).is_err());

        let mut forged_headers = HeaderMap::new();
        forged_headers.insert("x-kogwistar-roles", HeaderValue::from_static("rw"));
        assert!(!has_required_role(
            &forged_headers,
            &["ro".to_owned()],
            &config
        ));
    }

    #[test]
    fn jwks_kid_selection_supports_rotation_and_rejects_unknown_keys() {
        use axum::http::HeaderValue;
        use jsonwebtoken::{EncodingKey, Header, encode};
        use std::time::{SystemTime, UNIX_EPOCH};

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let claims = json!({"sub": "user", "role": "ro", "exp": now + 300});
        let token = |kid: &str, secret: &[u8]| {
            let mut header = Header::new(Algorithm::HS256);
            header.kid = Some(kid.to_owned());
            encode(&header, &claims, &EncodingKey::from_secret(secret)).unwrap()
        };
        let config = |jwks_json: Value| AuthConfig {
            mode: "oidc".to_owned(),
            algorithm: Some("HS256".to_owned()),
            key: None,
            jwks_json: Some(serde_json::to_string(&jwks_json).unwrap()),
            issuer: None,
            audience: None,
            oidc_providers_json: None,
            auth_db_url: None,
        };
        let mut headers = HeaderMap::new();
        headers.insert(
            "authorization",
            HeaderValue::from_str(&format!("Bearer {}", token("new", b"new-secret"))).unwrap(),
        );
        let rotating = config(json!({
            "keys": [
                {"kty": "oct", "kid": "old", "alg": "HS256", "k": "b2xkLXNlY3JldA"},
                {"kty": "oct", "kid": "new", "alg": "HS256", "k": "bmV3LXNlY3JldA"}
            ]
        }));
        assert_eq!(jwt_roles(&headers, &rotating).unwrap(), vec!["ro"]);

        let old_only = config(json!({
            "keys": [
                {"kty": "oct", "kid": "old", "alg": "HS256", "k": "b2xkLXNlY3JldA"}
            ]
        }));
        assert_eq!(
            jwt_roles(&headers, &old_only),
            Err("JWT kid is not trusted")
        );

        headers.insert(
            "authorization",
            HeaderValue::from_str(&format!("Bearer {}", token("new", b"forged-secret"))).unwrap(),
        );
        assert_eq!(jwt_roles(&headers, &rotating), Err("Invalid bearer token"));
    }

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
        let service = SqliteRunApplicationService::open(&path).unwrap();
        let transport_routes = [
            ("GET", "/health"),
            ("GET", "/api/auth/login"),
            ("GET", "/api/auth/callback"),
            ("GET", "/api/auth/me"),
            ("POST", "/api/auth/logout"),
            ("POST", "/auth/dev-token"),
        ];
        let mut unavailable = Vec::new();
        for &(method, template) in FROZEN_OPENAPI_ROUTES {
            if transport_routes.contains(&(method, template)) {
                continue;
            }
            let path_and_query = template
                .replace("{run_id}", "inventory-run")
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

    #[test]
    fn sqlite_graph_routes_fail_closed_with_explicit_authority_status() {
        let path = std::env::temp_dir().join(format!(
            "kogwistar-api-graph-authority-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let service = SqliteRunApplicationService::open(&path).unwrap();
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
        drop(service);
        remove_test_sqlite(&path);
    }

    #[test]
    fn frozen_mcp_tools_are_unique_and_versioned() {
        let contract: Value = serde_json::from_str(FROZEN_MCP_TOOLS_JSON).unwrap();
        assert_eq!(contract["contract_version"], "1.0.0");
        for (surface, expected) in [("root", 11), ("conversation", 5), ("workflow", 44)] {
            let tools = contract["surfaces"][surface].as_array().unwrap();
            assert_eq!(tools.len(), expected);
            let mut names = tools
                .iter()
                .map(|tool| tool["name"].as_str().unwrap())
                .collect::<Vec<_>>();
            names.sort_unstable();
            names.dedup();
            assert_eq!(names.len(), tools.len());
        }
    }

    #[test]
    fn rust_mcp_advertisement_is_subset_of_implemented_handlers() {
        let contract: Value = serde_json::from_str(FROZEN_MCP_TOOLS_JSON).unwrap();
        let root = contract["surfaces"]["root"].as_array().unwrap();
        let root_supported = root
            .iter()
            .filter(|tool| rust_mcp_tool_supported("root", tool["name"].as_str().unwrap()))
            .count();
        assert_eq!(root_supported, 4);
        assert!(rust_mcp_tool_supported("root", "kg_find_edges"));
        assert!(rust_mcp_tool_supported("root", "kg_k_hop"));
        assert!(rust_mcp_tool_supported("root", "kg_neighbors"));
        assert!(rust_mcp_tool_supported("root", "kg_shortest_path"));
        let conversation = contract["surfaces"]["conversation"].as_array().unwrap();
        let conversation_supported = conversation
            .iter()
            .filter(|tool| rust_mcp_tool_supported("conversation", tool["name"].as_str().unwrap()))
            .count();
        assert_eq!(conversation_supported, 4);
        assert!(rust_mcp_tool_supported("conversation", "conversation.ask"));
        assert!(rust_mcp_tool_supported(
            "conversation",
            "conversation.get_transcript"
        ));
        let workflow = contract["surfaces"]["workflow"].as_array().unwrap();
        let supported = workflow
            .iter()
            .filter(|tool| rust_mcp_tool_supported("workflow", tool["name"].as_str().unwrap()))
            .count();
        assert_eq!(supported, 37);
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
        assert_eq!(health.status(), StatusCode::OK);
        let health_body = to_bytes(health.into_body(), 4096).await.unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&health_body).unwrap()["backend"],
            "pg"
        );
        let health_value: Value = serde_json::from_slice(&health_body).unwrap();
        assert!(health_value["embedding_inline_worker_enabled"].is_boolean());
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

    #[tokio::test]
    async fn dev_token_me_and_logout_share_verified_jwt_contract() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use tower::ServiceExt;

        let app = router(ApiState {
            health: snapshot(),
            required_roles: Vec::new(),
            implementation: ImplementationSnapshot::default(),
            auth: AuthConfig {
                mode: "dev".to_owned(),
                algorithm: Some("HS256".to_owned()),
                key: Some("dev-secret".to_owned()),
                jwks_json: None,
                issuer: Some("local".to_owned()),
                audience: None,
                oidc_providers_json: None,
                auth_db_url: None,
            },
        });
        let token_response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/auth/dev-token")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"username":"dev@example.com","role":"rw","ns":"docs,workflow","capabilities":"project_view,workflow.run.read"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(token_response.status(), StatusCode::OK);
        let token_body = to_bytes(token_response.into_body(), 8192).await.unwrap();
        let token = serde_json::from_slice::<Value>(&token_body).unwrap()["token"]
            .as_str()
            .unwrap()
            .to_owned();
        let me = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/auth/me")
                    .header("authorization", format!("Bearer {token}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(me.status(), StatusCode::OK);
        let me_body = to_bytes(me.into_body(), 8192).await.unwrap();
        let me_value = serde_json::from_slice::<Value>(&me_body).unwrap();
        assert_eq!(me_value["email"], "dev@example.com");
        assert_eq!(me_value["role"], "rw");
        assert_eq!(me_value["ns"], json!(["docs", "workflow"]));

        let forged = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/auth/me")
                    .header("authorization", "Bearer forged")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(forged.status(), StatusCode::UNAUTHORIZED);
        let logout = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/auth/logout")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(logout.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn oidc_login_preserves_provider_pkce_and_cookie_contract() {
        use axum::body::Body;
        use axum::http::Request;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tower::ServiceExt;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let discovery_task = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 2048];
            let size = stream.read(&mut request).await.unwrap();
            assert!(String::from_utf8_lossy(&request[..size]).starts_with("GET /discovery "));
            let body = format!(r#"{{"authorization_endpoint":"http://{address}/authorize"}}"#);
            stream
                .write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                        body.len()
                    )
                    .as_bytes(),
                )
                .await
                .unwrap();
        });
        let app = router(ApiState {
            health: snapshot(),
            required_roles: Vec::new(),
            implementation: ImplementationSnapshot::default(),
            auth: AuthConfig {
                mode: "oidc".to_owned(),
                algorithm: None,
                key: None,
                jwks_json: None,
                issuer: None,
                audience: None,
                oidc_providers_json: Some(
                    json!({
                        "default_provider": "test",
                        "providers": {
                            "test": {
                                "name": "test",
                                "discovery_url": format!("http://{address}/discovery"),
                                "redirect_uri": "http://localhost/api/auth/callback",
                                "client_id": "client-1",
                                "client_secret": "secret",
                                "scopes": ["openid", "email", "profile"],
                                "allowed": true
                            }
                        }
                    })
                    .to_string(),
                ),
                auth_db_url: None,
            },
        });
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/auth/login")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        discovery_task.await.unwrap();
        assert_eq!(response.status(), StatusCode::TEMPORARY_REDIRECT);
        let location = response.headers()[header::LOCATION].to_str().unwrap();
        let authorization_url = url::Url::parse(location).unwrap();
        assert_eq!(authorization_url.path(), "/authorize");
        let query = authorization_url
            .query_pairs()
            .into_owned()
            .collect::<BTreeMap<_, _>>();
        assert_eq!(query["client_id"], "client-1");
        assert_eq!(query["response_type"], "code");
        assert_eq!(query["scope"], "openid email profile");
        assert_eq!(query["redirect_uri"], "http://localhost/api/auth/callback");
        assert_eq!(query["code_challenge_method"], "S256");
        assert_eq!(query["state"].len(), 64);
        assert_eq!(query["nonce"].len(), 64);
        assert!(!query["code_challenge"].contains('='));
        let cookies = response
            .headers()
            .get_all(header::SET_COOKIE)
            .iter()
            .map(|value| value.to_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(cookies.len(), 4);
        for cookie_name in [
            "auth_state=",
            "auth_pkce_verifier=",
            "auth_nonce=",
            "auth_provider=test",
        ] {
            assert!(cookies.iter().any(|cookie| {
                cookie.starts_with(cookie_name)
                    && cookie.contains("HttpOnly")
                    && cookie.contains("Max-Age=600")
            }));
        }
    }

    #[tokio::test]
    async fn oidc_login_rejects_production_redirect_override_before_network_io() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use tower::ServiceExt;

        let app = router(ApiState {
            health: snapshot(),
            required_roles: Vec::new(),
            implementation: ImplementationSnapshot::default(),
            auth: AuthConfig {
                mode: "oidc".to_owned(),
                oidc_providers_json: Some(r#"{"providers":{}}"#.to_owned()),
                auth_db_url: None,
                ..AuthConfig::default()
            },
        });
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/auth/login?redirect_uri=https://attacker.example")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(response.into_body(), 4096).await.unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&body).unwrap()["detail"],
            "redirect_uri override is only allowed when AUTH_MODE=dev"
        );
    }

    #[tokio::test]
    async fn oidc_callback_validates_rs256_and_persists_identity_before_minting() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use std::time::{SystemTime, UNIX_EPOCH};
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::time::{Duration, timeout};
        use tower::ServiceExt;

        const PRIVATE_KEY: &str = r#"-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDZEMrHkpSKPPCq
DATB6Q7At9YZVbLq4acttd+yM1ffjpEXfaMfpurxcpHOIoZOO4ddHUDqtDj+cPbH
T1CyNwrSl1jHLe1o+qZRkhPuuKS5ferI9vB0w2vsTynB7rqR45SPO3+ViCJDPmsq
x2qsX/pTXC4a9RRkOTXdEYg7t47jDxim822/AIRu2mSjA7GjGLoDpMTFm1/wTotm
DVb6/hy9iaSf5SO8qd8ROHH1vbRAtY1lcc4Sx/P2+Ngm94QEdKV/tCqQAV5yCPTR
2OcM5aDjyY7gyTY0kd9Gkw2He0o49INcdsGQSeIVE75ITC/vqommk4OgI2gCtd3g
MWDWm4dDAgMBAAECggEAPxjpASNkR1zYjm2o8l8XWUD3HO0y8aD/kkOEj43qNMOB
/KSaRuij8eSeap/Rj6sxOYl35eHWkWvv6Fbve6aRYE77UQbSNMprj1mZrrKAu6TV
G27gzehClnIajtOg6yiO9iXS+/oTD5300/4czZemshWhF1f3gfy5YhYnFkjQ4cJo
E1+5FnB8dgQLSk9clD47zVkFySWoSN1H7OOfwojtxfpgr/wN+/wHuJY425gaFc+7
VmQUg5RGtdciVYmwTV0tQIsqcqeM82/f2e554EzAOck+vX1PKhq3Ll8Z9S0rln57
o1HADCHBzHQ2APrr518BT/jYwsl+kGQPKI+IcmaE2QKBgQDwIkxejcjHvbg140s4
PyJ6yq3bthNF8XaXXMoa14lPQtTy316mn+uoNBwL9YFhEm9ByHBJ1aLugv2ZulMj
sC837+VZye6nl8fCiebGP8Guy4vHFgm4/Lk3HOkbr4Zd2Iwj5A9tlF/iMyjbHkdN
vAUCmkyhyJNpYk26EKijH0CG2QKBgQDnaE7Rx0/mXqhuNjAL9PypuVpqsA/3j3ZL
sJP+vrkEL8DoQGeOEFHGHGJlxmK/lgSmHcov34Zt4nhxa04BgpsIcd6C3OCaSJpX
Cw+KKit1CKPm5Bka2L9XtxaDKHSFH67K9pngc+QPlrEQeXH3xcCKxv0T50SdyZQ+
HSfvEWyFewKBgHx3OqBT2zr0sjN0QXvA9a0xupXENQ8uzeo8lSD+kNQ9bsUIVDYH
dA02HUdxlALtnC87pkAO9KmtyabRteAspPzYYkd87C9/83F5Kt2dFFX2eNfTK2zv
yUywtn68JugjotfDkN+aZWyIWefhNNIs32fu9ENzBD0+T81ebxpFy5tZAoGAG5rb
3DaUl3yvRwZ70NFW2sBbwuJh5Txd9kWIQhlqZM91ib81G0NjHekA6/cwjH5O66oe
Fnvpw24CxDTyx0dXSziaPK4wtPb4Qm31WpwRNxLiyoZnYEZ+/O3AZ8EJtV/EMD4e
uSHaEOn/EWILcG1MvMFkK12pV9FWN9quitxfP8UCgYEAvffs3ep6KgW8zGtQxlWt
emV55w55nDC3OYHQpvv41roViMihNU5EhxF0YNjus744wLxTSyWc29YgkSqZGRwK
NGj8qC7iDj7eHst5dr2KCfToUOTBidV7ynv8RZ5LyChcKzOiEDh/EqlEfGt5xYEI
6Qv6B43kPAqwE3JgGUyb4Wo=
-----END PRIVATE KEY-----"#;
        const JWK: &str = r#"{"kty":"RSA","kid":"test-key","alg":"RS256","use":"sig","n":"2RDKx5KUijzwqgwEwekOwLfWGVWy6uGnLbXfsjNX346RF32jH6bq8XKRziKGTjuHXR1A6rQ4_nD2x09QsjcK0pdYxy3taPqmUZIT7rikuX3qyPbwdMNr7E8pwe66keOUjzt_lYgiQz5rKsdqrF_6U1wuGvUUZDk13RGIO7eO4w8YpvNtvwCEbtpkowOxoxi6A6TExZtf8E6LZg1W-v4cvYmkn-UjvKnfEThx9b20QLWNZXHOEsfz9vjYJveEBHSlf7QqkAFecgj00djnDOWg48mO4Mk2NJHfRpMNh3tKOPSDXHbBkEniFRO-SEwv76qJppODoCNoArXd4DFg1puHQw","e":"AQAB"}"#;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let issuer = format!("http://{address}");
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let mut id_header = Header::new(Algorithm::RS256);
        id_header.kid = Some("test-key".to_owned());
        let id_token = encode(
            &id_header,
            &json!({
                "iss": issuer,
                "aud": "client-1",
                "sub": "subject-1",
                "email": "alice@example.com",
                "name": "Alice",
                "nonce": "nonce-1",
                "iat": now,
                "exp": now + 300,
            }),
            &EncodingKey::from_rsa_pem(PRIVATE_KEY.as_bytes()).unwrap(),
        )
        .unwrap();
        let mock_issuer = issuer.clone();
        let mock = tokio::spawn(async move {
            for _ in 0..4 {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut request = vec![0_u8; 8192];
                let size = stream.read(&mut request).await.unwrap();
                let request = String::from_utf8_lossy(&request[..size]);
                let (status, content_type, body) = if request.starts_with("GET /discovery ") {
                    (
                        "200 OK",
                        "application/json",
                        json!({
                            "issuer": mock_issuer,
                            "authorization_endpoint": format!("http://{address}/authorize"),
                            "token_endpoint": format!("http://{address}/token"),
                            "jwks_uri": format!("http://{address}/jwks"),
                            "userinfo_endpoint": format!("http://{address}/userinfo"),
                        })
                        .to_string(),
                    )
                } else if request.starts_with("POST /token ") {
                    assert!(request.contains("code=auth-code"));
                    assert!(request.contains("code_verifier=verifier-1"));
                    (
                        "200 OK",
                        "application/json",
                        json!({"access_token": "access-1", "id_token": id_token}).to_string(),
                    )
                } else if request.starts_with("GET /jwks ") {
                    (
                        "200 OK",
                        "application/json",
                        format!(r#"{{"keys":[{JWK}]}}"#),
                    )
                } else if request.starts_with("GET /userinfo ") {
                    assert!(request.contains("authorization: Bearer access-1"));
                    (
                        "200 OK",
                        "application/json",
                        json!({"sub": "subject-1", "email": "alice@example.com", "name": "Alice"})
                            .to_string(),
                    )
                } else {
                    ("404 Not Found", "text/plain", "not found".to_owned())
                };
                stream
                    .write_all(
                        format!(
                            "HTTP/1.1 {status}\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                            body.len()
                        )
                        .as_bytes(),
                    )
                    .await
                    .unwrap();
            }
        });
        let auth_path = std::env::temp_dir().join(format!(
            "kogwistar-oidc-callback-{}.sqlite3",
            uuid::Uuid::new_v4()
        ));
        let config = AuthConfig {
            mode: "oidc".to_owned(),
            algorithm: Some("HS256".to_owned()),
            key: Some("app-secret".to_owned()),
            jwks_json: None,
            issuer: Some("local".to_owned()),
            audience: None,
            oidc_providers_json: Some(
                json!({
                    "default_provider": "test",
                    "providers": {
                        "test": {
                            "name": "test",
                            "discovery_url": format!("http://{address}/discovery"),
                            "redirect_uri": "http://localhost/api/auth/callback",
                            "issuer": issuer,
                            "client_id": "client-1",
                            "client_secret": "secret",
                            "required_email": true,
                            "allowed": true,
                        }
                    }
                })
                .to_string(),
            ),
            auth_db_url: Some(format!("sqlite:///{}", auth_path.display())),
        };
        let app = router(ApiState {
            health: snapshot(),
            required_roles: Vec::new(),
            implementation: ImplementationSnapshot::default(),
            auth: config.clone(),
        });
        let response = timeout(
            Duration::from_secs(10),
            app.oneshot(
                Request::builder()
                    .uri("/api/auth/callback?code=auth-code&state=state-1")
                    .header(
                        header::COOKIE,
                        "auth_state=state-1; auth_pkce_verifier=verifier-1; auth_nonce=nonce-1; auth_provider=test",
                    )
                    .body(Body::empty())
                    .unwrap(),
            ),
        )
        .await
        .expect("OIDC callback exceeded the local test timeout")
        .unwrap();
        if response.status() != StatusCode::TEMPORARY_REDIRECT {
            let status = response.status();
            let body = to_bytes(response.into_body(), 16_384).await.unwrap();
            panic!(
                "OIDC callback failed with {status}: {}",
                String::from_utf8_lossy(&body)
            );
        }
        let location = response.headers()[header::LOCATION].to_str().unwrap();
        let token = location.split_once("?token=").unwrap().1;
        let mut token_headers = HeaderMap::new();
        token_headers.insert(
            header::AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {token}")).unwrap(),
        );
        let app_claims = jwt_claims(&token_headers, &config).unwrap();
        assert_eq!(app_claims["email"], "alice@example.com");
        assert_eq!(app_claims["role"], "ro");
        assert_eq!(app_claims["ns"], "docs");
        assert_eq!(
            response
                .headers()
                .get_all(header::SET_COOKIE)
                .iter()
                .count(),
            4
        );
        let auth_store = SqliteAuthStore::open(&auth_path).unwrap();
        let identity = auth_store
            .external_identity(&issuer, "subject-1")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(identity.email.as_deref(), Some("alice@example.com"));
        assert_eq!(
            auth_store
                .auth_user(&identity.user_id)
                .await
                .unwrap()
                .unwrap()
                .display_name
                .as_deref(),
            Some("Alice")
        );
        drop(auth_store);
        std::fs::remove_file(auth_path).unwrap();
        timeout(Duration::from_secs(2), mock)
            .await
            .expect("OIDC mock issuer did not receive its expected requests")
            .unwrap();
    }
