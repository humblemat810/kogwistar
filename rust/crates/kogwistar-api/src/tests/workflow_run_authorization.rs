    #[test]
    fn postgres_visibility_snapshot_mcp_admission_matches_python_policy() {
        let service = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "visibility_snapshot_mcp_admission",
        )
        .unwrap();
        let authorized = json!({
            "role": "ro",
            "ns": ["workflow"],
            "capabilities": ["project_view", "read_security_scope"]
        });

        assert!(service.supports_mcp_tool_for_principal(
            "workflow",
            "workflow.visibility_snapshot",
            &authorized
        ));
        assert!(!service.supports_mcp_tool_for_principal(
            "workflow",
            "workflow.visibility_snapshot",
            &json!({"role": "ro", "ns": ["workflow"], "capabilities": ["project_view"]})
        ));
        assert!(!service.supports_mcp_tool_for_principal(
            "workflow",
            "workflow.visibility_snapshot",
            &json!({"role": "ro", "ns": ["conversation"], "capabilities": ["project_view", "read_security_scope"]})
        ));
        assert!(!service.supports_mcp_tool_for_principal(
            "root",
            "workflow.visibility_snapshot",
            &authorized
        ));
    }

    #[test]
    fn workflow_run_events_mcp_admission_matches_python_policy_for_all_stores() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-run-events-mcp-{nonce}.sqlite3"));
        let sqlite = SqliteRunApplicationService::open(&path).unwrap();
        let postgres = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "workflow_run_events_mcp_admission",
        )
        .unwrap();
        let services: [&dyn ApplicationService; 2] = [&sqlite, &postgres];

        for service in services {
            assert!(service.supports_mcp_tool_for_principal(
                "workflow",
                "workflow.run_events",
                &json!({"role":"ro", "ns":["workflow"], "capabilities":["workflow.run.read"]})
            ));
            assert!(!service.supports_mcp_tool_for_principal(
                "workflow",
                "workflow.run_events",
                &json!({"role":"ro", "ns":["workflow"], "capabilities":[]})
            ));
            assert!(!service.supports_mcp_tool_for_principal(
                "workflow",
                "workflow.run_events",
                &json!({"role":"rw", "ns":["workflow"], "capabilities":["write_graph"]})
            ));
            assert!(!service.supports_mcp_tool_for_principal(
                "workflow",
                "workflow.run_events",
                &json!({"role":"ro", "ns":["conversation"], "capabilities":["workflow.run.read"]})
            ));
            assert!(!service.supports_mcp_tool_for_principal(
                "root",
                "workflow.run_events",
                &json!({"role":"ro", "ns":["workflow"], "capabilities":["workflow.run.read"]})
            ));
            assert!(service.supports_mcp_tool_for_principal(
                "workflow",
                "workflow.run_events",
                &json!({"role":"ro", "ns":["workflow"]})
            ));
        }

        drop(sqlite);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn workflow_capabilities_snapshot_mcp_admission_matches_python_policy_for_all_stores() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "kogwistar-capabilities-snapshot-mcp-{nonce}.sqlite3"
        ));
        let sqlite = SqliteRunApplicationService::open(&path).unwrap();
        let postgres = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "capabilities_snapshot_mcp_admission",
        )
        .unwrap();
        let services: [&dyn ApplicationService; 2] = [&sqlite, &postgres];

        for service in services {
            for tool in [
                "workflow.visibility_snapshot",
                "workflow.capabilities_snapshot",
            ] {
                assert!(service.supports_mcp_tool_for_principal(
                    "workflow",
                    tool,
                    &json!({
                        "role": "ro",
                        "ns": ["workflow"],
                        "capabilities": ["project_view", "read_security_scope"]
                    })
                ), "authorized principal must see {tool}");
                assert!(!service.supports_mcp_tool_for_principal(
                    "workflow",
                    tool,
                    &json!({"role":"ro", "ns":["workflow"], "capabilities":["project_view"]})
                ), "missing read_security_scope must hide {tool}");
                assert!(!service.supports_mcp_tool_for_principal(
                    "workflow",
                    tool,
                    &json!({"role":"ro", "ns":["workflow"], "capabilities":["read_security_scope"]})
                ), "missing project_view must hide {tool}");
                assert!(!service.supports_mcp_tool_for_principal(
                    "workflow",
                    tool,
                    &json!({"role":"ro", "ns":["conversation"], "capabilities":["project_view", "read_security_scope"]})
                ), "wrong namespace must hide {tool}");
            }
        }

        drop(sqlite);
        let _ = std::fs::remove_file(path);
    }
    #[test]
    fn workflow_dead_letter_replay_mcp_admission_matches_python_policy_for_all_stores() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-dead-letter-mcp-{nonce}.sqlite3"));
        let sqlite = SqliteRunApplicationService::open(&path).unwrap();
        let postgres = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "dead_letter_mcp_admission",
        )
        .unwrap();
        let services: [&dyn ApplicationService; 2] = [&sqlite, &postgres];

        for service in services {
            let tool = "workflow.dead_letter_replay";
            assert!(service.supports_mcp_tool_for_principal(
                "workflow",
                tool,
                &json!({"role":"rw", "ns":["workflow"], "capabilities":["service.manage"]})
            ));
            for principal in [
                json!({"role":"rw", "ns":["workflow"], "capabilities":[]}),
                json!({"role":"ro", "ns":["workflow"], "capabilities":["service.manage"]}),
                json!({"role":"rw", "ns":["conversation"], "capabilities":["service.manage"]}),
            ] {
                assert!(!service.supports_mcp_tool_for_principal(
                    "workflow", tool, &principal
                ));
            }
            assert!(!service.supports_mcp_tool_for_principal(
                "root",
                tool,
                &json!({"role":"rw", "ns":["workflow"], "capabilities":["service.manage"]})
            ));
        }

        drop(sqlite);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn dead_letter_replay_effect_rechecks_role_and_namespace_for_all_stores() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-dead-letter-effect-{nonce}.sqlite3"));
        let sqlite = SqliteRunApplicationService::open(&path).unwrap();
        let postgres = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "dead_letter_effect_auth",
        )
        .unwrap();
        let services: [&dyn ApplicationService; 2] = [&sqlite, &postgres];

        for service in services {
            for principal in [
                json!({"role":"ro", "ns":["workflow"], "capabilities":["service.manage"]}),
                json!({"role":"rw", "ns":["conversation"], "capabilities":["service.manage"]}),
            ] {
                let response = service
                    .execute(ApiEffectRequest {
                        contract_version: 1,
                        method: "POST".to_owned(),
                        path_and_query: "/api/workflow/dead-letters/never-read/replay".to_owned(),
                        body: Vec::new(),
                        principal: principal.clone(),
                    })
                    .await;
                assert_eq!(response.status, StatusCode::FORBIDDEN.as_u16());
            }
        }

        drop(sqlite);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn mcp_workflow_dead_letter_replay_matches_shared_contract() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use std::future::Future;
        use std::pin::Pin;
        use std::sync::{Arc, Mutex};
        use tower::ServiceExt;

        #[derive(Clone)]
        struct RequestRecorder(Arc<Mutex<Option<ApiEffectRequest>>>, Value);

        impl ApplicationService for RequestRecorder {
            fn supports_mcp_tool(&self, surface: &str, name: &str) -> bool {
                surface == "workflow" && name == "workflow.dead_letter_replay"
            }

            fn execute(
                &self,
                request: ApiEffectRequest,
            ) -> Pin<Box<dyn Future<Output = ApiEffectResponse> + Send + '_>> {
                let seen = Arc::clone(&self.0);
                let result = self.1.clone();
                Box::pin(async move {
                    *seen.lock().unwrap() = Some(request);
                    ApiEffectResponse {
                        status: StatusCode::OK.as_u16(),
                        content_type: "application/json".to_owned(),
                        body: serde_json::to_vec(&result).unwrap(),
                    }
                })
            }
        }

        let contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/workflow-dead-letter-replay-mcp-v1.json"
        ))
        .unwrap();
        let seen = Arc::new(Mutex::new(None));
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: Vec::new(),
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig::default(),
            },
            Arc::new(RequestRecorder(
                Arc::clone(&seen),
                contract["result"].clone(),
            )),
        );
        let request_body = serde_json::to_string(&json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": contract["tool"],
                "arguments": contract["arguments"],
            }
        }))
        .unwrap();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/workflow")
                    .header("content-type", "application/json")
                    .body(Body::from(request_body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let wire: Value = serde_json::from_slice(
            &to_bytes(response.into_body(), 8192).await.unwrap(),
        )
        .unwrap();
        assert_eq!(wire["result"], contract["result"]);

        let request = seen.lock().unwrap().clone().unwrap();
        assert_eq!(request.method, contract["rust_request"]["method"]);
        assert_eq!(
            request.path_and_query,
            contract["rust_request"]["path_and_query"]
        );
        assert_eq!(request.body, contract["rust_request"]["body"].as_str().unwrap().as_bytes());
    }

    #[test]
    fn workflow_dead_letters_mcp_admission_matches_python_policy_for_all_stores() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-dead-letters-mcp-{nonce}.sqlite3"));
        let sqlite = SqliteRunApplicationService::open(&path).unwrap();
        let postgres = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "dead_letters_mcp_admission",
        )
        .unwrap();
        let services: [&dyn ApplicationService; 2] = [&sqlite, &postgres];

        for service in services {
            let tool = "workflow.dead_letters";
            assert!(service.supports_mcp_tool_for_principal(
                "workflow",
                tool,
                &json!({"role":"ro", "ns":["workflow"], "capabilities":["project_view"]})
            ));
            for principal in [
                json!({"role":"rw", "ns":["workflow"], "capabilities":[]}),
                json!({"role":"rw", "ns":["conversation"], "capabilities":["project_view"]}),
            ] {
                assert!(!service.supports_mcp_tool_for_principal(
                    "workflow", tool, &principal
                ));
            }
            assert!(!service.supports_mcp_tool_for_principal(
                "root",
                tool,
                &json!({"role":"ro", "ns":["workflow"], "capabilities":["project_view"]})
            ));
        }

        drop(sqlite);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn dead_letters_effect_rechecks_role_namespace_and_capability_for_all_stores() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-dead-letters-effect-{nonce}.sqlite3"));
        let sqlite = SqliteRunApplicationService::open(&path).unwrap();
        let postgres = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "dead_letters_effect_auth",
        )
        .unwrap();
        let services: [&dyn ApplicationService; 2] = [&sqlite, &postgres];

        for service in services {
            for principal in [
                json!({"role":"ro", "ns":["workflow"], "capabilities":[]}),
                json!({"role":"ro", "ns":["conversation"], "capabilities":["project_view"]}),
            ] {
                let response = service
                    .execute(ApiEffectRequest {
                        contract_version: 1,
                        method: "GET".to_owned(),
                        path_and_query: "/api/workflow/dead-letters?limit=10".to_owned(),
                        body: Vec::new(),
                        principal,
                    })
                    .await;
                assert_eq!(response.status, StatusCode::FORBIDDEN.as_u16());
            }
        }

        drop(sqlite);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn mcp_workflow_dead_letters_matches_shared_contract() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use std::future::Future;
        use std::pin::Pin;
        use std::sync::{Arc, Mutex};
        use tower::ServiceExt;

        #[derive(Clone)]
        struct RequestRecorder(Arc<Mutex<Option<ApiEffectRequest>>>, Value);

        impl ApplicationService for RequestRecorder {
            fn supports_mcp_tool(&self, surface: &str, name: &str) -> bool {
                surface == "workflow" && name == "workflow.dead_letters"
            }

            fn execute(
                &self,
                request: ApiEffectRequest,
            ) -> Pin<Box<dyn Future<Output = ApiEffectResponse> + Send + '_>> {
                let seen = Arc::clone(&self.0);
                let result = self.1.clone();
                Box::pin(async move {
                    *seen.lock().unwrap() = Some(request);
                    ApiEffectResponse {
                        status: StatusCode::OK.as_u16(),
                        content_type: "application/json".to_owned(),
                        body: serde_json::to_vec(&result).unwrap(),
                    }
                })
            }
        }

        let contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/workflow-dead-letters-mcp-v1.json"
        ))
        .unwrap();
        let seen = Arc::new(Mutex::new(None));
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: Vec::new(),
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig::default(),
            },
            Arc::new(RequestRecorder(
                Arc::clone(&seen),
                contract["result"].clone(),
            )),
        );
        let request_body = serde_json::to_string(&json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": contract["tool"],
                "arguments": contract["arguments"],
            }
        }))
        .unwrap();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/workflow")
                    .header("content-type", "application/json")
                    .body(Body::from(request_body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let wire: Value = serde_json::from_slice(
            &to_bytes(response.into_body(), 8192).await.unwrap(),
        )
        .unwrap();
        assert_eq!(wire["result"], contract["result"]);

        let request = seen.lock().unwrap().clone().unwrap();
        assert_eq!(request.method, contract["rust_request"]["method"]);
        assert_eq!(
            request.path_and_query,
            contract["rust_request"]["path_and_query"]
        );
        assert_eq!(request.body, contract["rust_request"]["body"].as_str().unwrap().as_bytes());
    }

    #[test]
    fn workflow_design_edge_delete_mcp_admission_matches_python_policy_for_all_stores() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-design-edge-mcp-{nonce}.sqlite3"));
        let sqlite = SqliteRunApplicationService::open(&path).unwrap();
        let postgres = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "design_edge_delete_mcp_admission",
        )
        .unwrap();
        let services: [&dyn ApplicationService; 2] = [&sqlite, &postgres];

        for service in services {
            let tool = "workflow.design_edge_delete";
            let authorized = json!({
                "role":"rw", "ns":["workflow"],
                "capabilities":["workflow.design.write", "write_graph"]
            });
            assert!(service.supports_mcp_tool_for_principal("workflow", tool, &authorized));
            for principal in [
                json!({"role":"rw", "ns":["workflow"], "capabilities":["workflow.design.write"]}),
                json!({"role":"rw", "ns":["workflow"], "capabilities":["write_graph"]}),
                json!({"role":"ro", "ns":["workflow"], "capabilities":["workflow.design.write", "write_graph"]}),
                json!({"role":"rw", "ns":["conversation"], "capabilities":["workflow.design.write", "write_graph"]}),
            ] {
                assert!(!service.supports_mcp_tool_for_principal("workflow", tool, &principal));
            }
            assert!(!service.supports_mcp_tool_for_principal("root", tool, &authorized));
        }

        drop(sqlite);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn workflow_design_edge_delete_effect_rechecks_scope_and_capabilities_for_all_stores() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-design-edge-effect-{nonce}.sqlite3"));
        let sqlite = SqliteRunApplicationService::open(&path).unwrap();
        let postgres = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "design_edge_delete_effect_auth",
        )
        .unwrap();
        let services: [&dyn ApplicationService; 2] = [&sqlite, &postgres];
        for service in services {
            for principal in [
                json!({"role":"rw", "ns":["workflow"], "capabilities":["workflow.design.write"]}),
                json!({"role":"ro", "ns":["workflow"], "capabilities":["workflow.design.write", "write_graph"]}),
                json!({"role":"rw", "ns":["conversation"], "capabilities":["workflow.design.write", "write_graph"]}),
            ] {
                let response = service
                    .execute(ApiEffectRequest {
                        contract_version: 1,
                        method: "DELETE".to_owned(),
                        path_and_query: "/api/workflow/design/wf-mcp-delete/edges/edge-mcp-delete".to_owned(),
                        body: br#"{"designer_id":"alice","edge_id":"edge-mcp-delete","workflow_id":"wf-mcp-delete"}"#.to_vec(),
                        principal,
                    })
                    .await;
                assert_eq!(response.status, StatusCode::FORBIDDEN.as_u16());
            }
        }
        drop(sqlite);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn mcp_workflow_design_edge_delete_matches_shared_contract() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use std::future::Future;
        use std::pin::Pin;
        use std::sync::{Arc, Mutex};
        use tower::ServiceExt;

        #[derive(Clone)]
        struct RequestRecorder(Arc<Mutex<Option<ApiEffectRequest>>>, Value);

        impl ApplicationService for RequestRecorder {
            fn supports_mcp_tool(&self, surface: &str, name: &str) -> bool {
                surface == "workflow" && name == "workflow.design_edge_delete"
            }

            fn execute(
                &self,
                request: ApiEffectRequest,
            ) -> Pin<Box<dyn Future<Output = ApiEffectResponse> + Send + '_>> {
                let seen = Arc::clone(&self.0);
                let result = self.1.clone();
                Box::pin(async move {
                    *seen.lock().unwrap() = Some(request);
                    ApiEffectResponse {
                        status: StatusCode::OK.as_u16(),
                        content_type: "application/json".to_owned(),
                        body: serde_json::to_vec(&result).unwrap(),
                    }
                })
            }
        }

        let contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/workflow-design-edge-delete-mcp-v1.json"
        ))
        .unwrap();
        let seen = Arc::new(Mutex::new(None));
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: Vec::new(),
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig::default(),
            },
            Arc::new(RequestRecorder(Arc::clone(&seen), contract["result"].clone())),
        );
        let request_body = serde_json::to_string(&json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": contract["tool"],
                "arguments": contract["arguments"],
            }
        }))
        .unwrap();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/workflow")
                    .header("content-type", "application/json")
                    .body(Body::from(request_body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let wire: Value = serde_json::from_slice(
            &to_bytes(response.into_body(), 8192).await.unwrap(),
        )
        .unwrap();
        assert_eq!(wire["result"], contract["result"]);

        let request = seen.lock().unwrap().clone().unwrap();
        assert_eq!(request.method, contract["rust_request"]["method"]);
        assert_eq!(request.path_and_query, contract["rust_request"]["path_and_query"]);
        assert_eq!(request.body, contract["rust_request"]["body"].as_str().unwrap().as_bytes());
    }

    #[test]
    fn workflow_design_edge_upsert_mcp_admission_matches_python_policy_for_all_stores() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-design-edge-upsert-mcp-{nonce}.sqlite3"));
        let sqlite = SqliteRunApplicationService::open(&path).unwrap();
        let postgres = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "design_edge_upsert_mcp_admission",
        )
        .unwrap();
        let services: [&dyn ApplicationService; 2] = [&sqlite, &postgres];

        for service in services {
            let tool = "workflow.design_edge_upsert";
            let authorized = json!({
                "role":"rw", "ns":["workflow"],
                "capabilities":["workflow.design.write", "write_graph"]
            });
            assert!(service.supports_mcp_tool_for_principal("workflow", tool, &authorized));
            for principal in [
                json!({"role":"rw", "ns":["workflow"], "capabilities":["workflow.design.write"]}),
                json!({"role":"rw", "ns":["workflow"], "capabilities":["write_graph"]}),
                json!({"role":"ro", "ns":["workflow"], "capabilities":["workflow.design.write", "write_graph"]}),
                json!({"role":"rw", "ns":["conversation"], "capabilities":["workflow.design.write", "write_graph"]}),
            ] {
                assert!(!service.supports_mcp_tool_for_principal("workflow", tool, &principal));
            }
            assert!(!service.supports_mcp_tool_for_principal("root", tool, &authorized));
        }

        drop(sqlite);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn workflow_design_edge_upsert_effect_rechecks_scope_and_capabilities_for_all_stores() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-design-edge-upsert-effect-{nonce}.sqlite3"));
        let sqlite = SqliteRunApplicationService::open(&path).unwrap();
        let postgres = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "design_edge_upsert_effect_auth",
        )
        .unwrap();
        let services: [&dyn ApplicationService; 2] = [&sqlite, &postgres];
        for service in services {
            for principal in [
                json!({"role":"rw", "ns":["workflow"], "capabilities":["workflow.design.write"]}),
                json!({"role":"ro", "ns":["workflow"], "capabilities":["workflow.design.write", "write_graph"]}),
                json!({"role":"rw", "ns":["conversation"], "capabilities":["workflow.design.write", "write_graph"]}),
            ] {
                let response = service
                    .execute(ApiEffectRequest {
                        contract_version: 1,
                        method: "POST".to_owned(),
                        path_and_query: "/api/workflow/design/wf-mcp-upsert/edges".to_owned(),
                        body: br#"{"workflow_id":"wf-mcp-upsert","designer_id":"alice","edge_id":"edge-mcp-upsert","src":"node-start","dst":"node-end","relation":"wf_next","predicate":null,"priority":100,"is_default":true,"multiplicity":"one","metadata":{}}"#.to_vec(),
                        principal,
                    })
                    .await;
                assert_eq!(response.status, StatusCode::FORBIDDEN.as_u16());
            }
        }
        drop(sqlite);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn mcp_workflow_design_edge_upsert_matches_shared_contract() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use std::future::Future;
        use std::pin::Pin;
        use std::sync::{Arc, Mutex};
        use tower::ServiceExt;

        #[derive(Clone)]
        struct RequestRecorder(Arc<Mutex<Option<ApiEffectRequest>>>, Value);

        impl ApplicationService for RequestRecorder {
            fn supports_mcp_tool(&self, surface: &str, name: &str) -> bool {
                surface == "workflow" && name == "workflow.design_edge_upsert"
            }

            fn execute(
                &self,
                request: ApiEffectRequest,
            ) -> Pin<Box<dyn Future<Output = ApiEffectResponse> + Send + '_>> {
                let seen = Arc::clone(&self.0);
                let result = self.1.clone();
                Box::pin(async move {
                    *seen.lock().unwrap() = Some(request);
                    ApiEffectResponse {
                        status: StatusCode::OK.as_u16(),
                        content_type: "application/json".to_owned(),
                        body: serde_json::to_vec(&result).unwrap(),
                    }
                })
            }
        }

        let contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/workflow-design-edge-upsert-mcp-v1.json"
        ))
        .unwrap();
        let seen = Arc::new(Mutex::new(None));
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: Vec::new(),
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig::default(),
            },
            Arc::new(RequestRecorder(Arc::clone(&seen), contract["result"].clone())),
        );
        let request_body = serde_json::to_string(&json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": contract["tool"], "arguments": contract["arguments"]}
        }))
        .unwrap();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/workflow")
                    .header("content-type", "application/json")
                    .body(Body::from(request_body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let wire: Value = serde_json::from_slice(
            &to_bytes(response.into_body(), 8192).await.unwrap(),
        )
        .unwrap();
        assert_eq!(wire["result"], contract["result"]);

        let request = seen.lock().unwrap().clone().unwrap();
        assert_eq!(request.method, contract["rust_request"]["method"]);
        assert_eq!(request.path_and_query, contract["rust_request"]["path_and_query"]);
        assert_eq!(request.body, contract["rust_request"]["body"].as_str().unwrap().as_bytes());
    }

    #[test]
    fn workflow_design_history_mcp_admission_matches_python_policy_for_all_stores() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-design-history-mcp-{nonce}.sqlite3"));
        let sqlite = SqliteRunApplicationService::open(&path).unwrap();
        let postgres = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "design_history_mcp_admission",
        )
        .unwrap();
        let services: [&dyn ApplicationService; 2] = [&sqlite, &postgres];

        for service in services {
            let tool = "workflow.design_history";
            for role in ["ro", "rw"] {
                let authorized = json!({
                    "role":role, "ns":["workflow"],
                    "capabilities":["workflow.design.inspect"]
                });
                assert!(service.supports_mcp_tool_for_principal("workflow", tool, &authorized));
            }
            for principal in [
                json!({"role":"ro", "ns":["workflow"], "capabilities":[]}),
                json!({"role":"rw", "ns":["conversation"], "capabilities":["workflow.design.inspect"]}),
                json!({"role":"admin", "ns":["workflow"], "capabilities":["workflow.design.inspect"]}),
            ] {
                assert!(!service.supports_mcp_tool_for_principal("workflow", tool, &principal));
            }
            assert!(!service.supports_mcp_tool_for_principal(
                "root",
                tool,
                &json!({"role":"ro", "ns":["workflow"], "capabilities":["workflow.design.inspect"]})
            ));
        }

        drop(sqlite);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn workflow_design_history_effect_rechecks_scope_and_capability_for_all_stores() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-design-history-effect-{nonce}.sqlite3"));
        let sqlite = SqliteRunApplicationService::open(&path).unwrap();
        let postgres = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "design_history_effect_auth",
        )
        .unwrap();
        let services: [&dyn ApplicationService; 2] = [&sqlite, &postgres];
        for service in services {
            for principal in [
                json!({"role":"ro", "ns":["workflow"], "capabilities":[]}),
                json!({"role":"rw", "ns":["conversation"], "capabilities":["workflow.design.inspect"]}),
                json!({"role":"admin", "ns":["workflow"], "capabilities":["workflow.design.inspect"]}),
            ] {
                let response = service
                    .execute(ApiEffectRequest {
                        contract_version: 1,
                        method: "GET".to_owned(),
                        path_and_query: "/api/workflow/design/wf-mcp-history/history".to_owned(),
                        body: Vec::new(),
                        principal,
                    })
                    .await;
                assert_eq!(response.status, StatusCode::FORBIDDEN.as_u16());
            }
        }
        drop(sqlite);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn mcp_workflow_design_history_matches_shared_contract() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use std::future::Future;
        use std::pin::Pin;
        use std::sync::{Arc, Mutex};
        use tower::ServiceExt;

        #[derive(Clone)]
        struct RequestRecorder(Arc<Mutex<Option<ApiEffectRequest>>>, Value);

        impl ApplicationService for RequestRecorder {
            fn supports_mcp_tool(&self, surface: &str, name: &str) -> bool {
                surface == "workflow" && name == "workflow.design_history"
            }

            fn execute(
                &self,
                request: ApiEffectRequest,
            ) -> Pin<Box<dyn Future<Output = ApiEffectResponse> + Send + '_>> {
                let seen = Arc::clone(&self.0);
                let result = self.1.clone();
                Box::pin(async move {
                    *seen.lock().unwrap() = Some(request);
                    ApiEffectResponse {
                        status: StatusCode::OK.as_u16(),
                        content_type: "application/json".to_owned(),
                        body: serde_json::to_vec(&result).unwrap(),
                    }
                })
            }
        }

        let contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/workflow-design-history-mcp-v1.json"
        ))
        .unwrap();
        let seen = Arc::new(Mutex::new(None));
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: Vec::new(),
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig::default(),
            },
            Arc::new(RequestRecorder(Arc::clone(&seen), contract["result"].clone())),
        );
        let request_body = serde_json::to_string(&json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": contract["tool"], "arguments": contract["arguments"]}
        }))
        .unwrap();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/workflow")
                    .header("content-type", "application/json")
                    .body(Body::from(request_body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let wire: Value = serde_json::from_slice(
            &to_bytes(response.into_body(), 8192).await.unwrap(),
        )
        .unwrap();
        assert_eq!(wire["result"], contract["result"]);

        let request = seen.lock().unwrap().clone().unwrap();
        assert_eq!(request.method, contract["rust_request"]["method"]);
        assert_eq!(request.path_and_query, contract["rust_request"]["path_and_query"]);
        assert!(request.body.is_empty());
    }

    #[tokio::test]
    async fn workflow_design_history_empty_state_matches_shared_contract() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-design-history-contract-{nonce}.sqlite3"));
        let service = SqliteRunApplicationService::open(&path).unwrap();
        let contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/workflow-design-history-mcp-v1.json"
        ))
        .unwrap();
        let response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: contract["rust_request"]["path_and_query"]
                    .as_str()
                    .unwrap()
                    .to_owned(),
                body: Vec::new(),
                principal: json!({
                    "role":"ro", "ns":["workflow"],
                    "capabilities":contract["caller_capabilities"]
                }),
            })
            .await;

        assert_eq!(response.status, StatusCode::OK.as_u16());
        assert_eq!(serde_json::from_slice::<Value>(&response.body).unwrap(), contract["result"]);
        drop(service);
        let _ = std::fs::remove_file(path);
    }
