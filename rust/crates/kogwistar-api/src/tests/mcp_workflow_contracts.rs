    #[tokio::test]
    async fn mcp_workflow_run_replay_matches_shared_contract() {
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
                surface == "workflow" && name == "workflow.run_replay"
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
            "../../../../../contracts/golden/workflow-run-replay-mcp-v1.json"
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
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/workflow")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::to_vec(&json!({
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "tools/call",
                            "params": {
                                "name": contract["tool"],
                                "arguments": contract["arguments"]
                            }
                        }))
                        .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 4096).await.unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["result"], contract["result"]);
        let request = seen.lock().unwrap().clone().unwrap();
        assert_eq!(request.method, "GET");
        assert_eq!(
            request.path_and_query,
            "/api/workflow/runs/replay-contract-run/replay?target_step_seq=3"
        );
        assert!(request.body.is_empty());
    }

    #[tokio::test]
    async fn mcp_workflow_run_cancel_matches_shared_contract() {
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
                surface == "workflow" && name == "workflow.run_cancel"
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
                        status: StatusCode::ACCEPTED.as_u16(),
                        content_type: "application/json".to_owned(),
                        body: serde_json::to_vec(&result).unwrap(),
                    }
                })
            }
        }

        let contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/workflow-run-cancel-mcp-v1.json"
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
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/workflow")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::to_vec(&json!({
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "tools/call",
                            "params": {
                                "name": contract["tool"],
                                "arguments": contract["arguments"]
                            }
                        }))
                        .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 4096).await.unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["result"], contract["result"]);
        let request = seen.lock().unwrap().clone().unwrap();
        assert_eq!(request.method, "POST");
        assert_eq!(
            request.path_and_query,
            "/api/workflow/runs/mcp-cancel-contract-run/cancel"
        );
        assert!(request.body.is_empty());
    }

    #[tokio::test]
    async fn mcp_workflow_run_submit_matches_shared_contract() {
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
                surface == "workflow" && name == "workflow.run_submit"
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
                        status: StatusCode::ACCEPTED.as_u16(),
                        content_type: "application/json".to_owned(),
                        body: serde_json::to_vec(&result).unwrap(),
                    }
                })
            }
        }

        let contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/workflow-run-submit-mcp-v1.json"
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
            .clone()
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
        let body = to_bytes(response.into_body(), 8192).await.unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["result"], contract["result"]);

        let request = seen.lock().unwrap().clone().unwrap();
        assert_eq!(request.method, "POST");
        assert_eq!(request.path_and_query, "/api/workflow/runs");
        assert_eq!(
            serde_json::from_slice::<Value>(&request.body).unwrap(),
            contract["arguments"]
        );
    }

    #[tokio::test]
    async fn mcp_workflow_scheduler_timeline_matches_shared_contract() {
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
                surface == "workflow" && name == "workflow.scheduler_timeline"
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
            "../../../../../contracts/golden/workflow-scheduler-timeline-mcp-v1.json"
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
        let body = to_bytes(response.into_body(), 8192).await.unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["result"], contract["result"]);

        let request = seen.lock().unwrap().clone().unwrap();
        assert_eq!(request.method, contract["rust_request"]["method"]);
        assert_eq!(
            request.path_and_query,
            contract["rust_request"]["path_and_query"]
        );
    }

    #[tokio::test]
    async fn mcp_workflow_service_declare_matches_shared_contract() {
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
                surface == "workflow" && name == "workflow.service_declare"
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
            "../../../../../contracts/golden/workflow-service-declare-mcp-v1.json"
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
        let body = to_bytes(response.into_body(), 8192).await.unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["result"], contract["result"]);

        let request = seen.lock().unwrap().clone().unwrap();
        assert_eq!(request.method, contract["rust_request"]["method"]);
        assert_eq!(
            request.path_and_query,
            contract["rust_request"]["path_and_query"]
        );
        assert_eq!(
            serde_json::from_slice::<Value>(&request.body).unwrap(),
            contract["arguments"]
        );
    }

    #[tokio::test]
    async fn mcp_workflow_service_disable_matches_shared_contract() {
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
                surface == "workflow" && name == "workflow.service_disable"
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
            "../../../../../contracts/golden/workflow-service-disable-mcp-v1.json"
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
        let body = to_bytes(response.into_body(), 8192).await.unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["result"], contract["result"]);

        let request = seen.lock().unwrap().clone().unwrap();
        assert_eq!(request.method, contract["rust_request"]["method"]);
        assert_eq!(
            request.path_and_query,
            contract["rust_request"]["path_and_query"]
        );
        assert!(request.body.is_empty());
    }

    #[tokio::test]
    async fn mcp_workflow_service_enable_matches_shared_contract() {
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
                surface == "workflow" && name == "workflow.service_enable"
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
            "../../../../../contracts/golden/workflow-service-enable-mcp-v1.json"
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
        let body = to_bytes(response.into_body(), 8192).await.unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["result"], contract["result"]);

        let request = seen.lock().unwrap().clone().unwrap();
        assert_eq!(request.method, contract["rust_request"]["method"]);
        assert_eq!(
            request.path_and_query,
            contract["rust_request"]["path_and_query"]
        );
        assert!(request.body.is_empty());
    }

    #[tokio::test]
    async fn mcp_workflow_service_events_matches_shared_contract() {
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
                surface == "workflow" && name == "workflow.service_events"
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
            "../../../../../contracts/golden/workflow-service-events-mcp-v1.json"
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
        let body = to_bytes(response.into_body(), 8192).await.unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["result"], contract["result"]);

        let request = seen.lock().unwrap().clone().unwrap();
        assert_eq!(request.method, contract["rust_request"]["method"]);
        assert_eq!(
            request.path_and_query,
            contract["rust_request"]["path_and_query"]
        );
        assert!(request.body.is_empty());
    }

    #[tokio::test]
    async fn mcp_workflow_service_get_matches_shared_contract() {
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
                surface == "workflow" && name == "workflow.service_get"
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
            "../../../../../contracts/golden/workflow-service-get-mcp-v1.json"
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
        let body = to_bytes(response.into_body(), 8192).await.unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["result"], contract["result"]);

        let request = seen.lock().unwrap().clone().unwrap();
        assert_eq!(request.method, contract["rust_request"]["method"]);
        assert_eq!(
            request.path_and_query,
            contract["rust_request"]["path_and_query"]
        );
        assert!(request.body.is_empty());
    }

    #[tokio::test]
    async fn mcp_workflow_service_heartbeat_matches_shared_contract() {
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
                surface == "workflow" && name == "workflow.service_heartbeat"
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
            "../../../../../contracts/golden/workflow-service-heartbeat-mcp-v1.json"
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
        let body = to_bytes(response.into_body(), 8192).await.unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["result"], contract["result"]);

        let request = seen.lock().unwrap().clone().unwrap();
        assert_eq!(request.method, contract["rust_request"]["method"]);
        assert_eq!(
            request.path_and_query,
            contract["rust_request"]["path_and_query"]
        );
        assert_eq!(
            serde_json::from_slice::<Value>(&request.body).unwrap(),
            contract["rust_request"]["body"]
        );

    }

    #[tokio::test]
    async fn mcp_workflow_service_list_matches_shared_contract() {
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
                surface == "workflow" && name == "workflow.service_list"
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
            "../../../../../contracts/golden/workflow-service-list-mcp-v1.json"
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
        let body = to_bytes(response.into_body(), 8192).await.unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["result"], contract["result"]);

        let request = seen.lock().unwrap().clone().unwrap();
        assert_eq!(request.method, contract["rust_request"]["method"]);
        assert_eq!(
            request.path_and_query,
            contract["rust_request"]["path_and_query"]
        );
        assert!(request.body.is_empty());
    }

    #[tokio::test]
    async fn mcp_workflow_service_repair_matches_shared_contract() {
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
                surface == "workflow" && name == "workflow.service_repair"
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
            "../../../../../contracts/golden/workflow-service-repair-mcp-v1.json"
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
        let body = to_bytes(response.into_body(), 8192).await.unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["result"], contract["result"]);

        let request = seen.lock().unwrap().clone().unwrap();
        assert_eq!(request.method, contract["rust_request"]["method"]);
        assert_eq!(
            request.path_and_query,
            contract["rust_request"]["path_and_query"]
        );
        assert!(request.body.is_empty());
    }

    #[tokio::test]
    async fn mcp_workflow_visibility_snapshot_matches_shared_contract() {
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
                surface == "workflow" && name == "workflow.visibility_snapshot"
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
            "../../../../../contracts/golden/workflow-visibility-snapshot-mcp-v1.json"
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
        let body = to_bytes(response.into_body(), 8192).await.unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["result"], contract["result"]);

        let request = seen.lock().unwrap().clone().unwrap();
        assert_eq!(request.method, contract["rust_request"]["method"]);
        assert_eq!(request.path_and_query, contract["rust_request"]["path_and_query"]);
        assert_eq!(
            request.body,
            contract["rust_request"]["body"].as_str().unwrap().as_bytes()
        );
    }

    #[tokio::test]
    async fn mcp_workflow_capabilities_snapshot_matches_shared_contract() {
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
                surface == "workflow" && name == "workflow.capabilities_snapshot"
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
            "../../../../../contracts/golden/workflow-capabilities-snapshot-mcp-v1.json"
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
        let body = to_bytes(response.into_body(), 8192).await.unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["result"], contract["result"]);

        let request = seen.lock().unwrap().clone().unwrap();
        assert_eq!(request.method, contract["rust_request"]["method"]);
        assert_eq!(request.path_and_query, contract["rust_request"]["path_and_query"]);
        assert_eq!(
            request.body,
            contract["rust_request"]["body"].as_str().unwrap().as_bytes()
        );
    }

    #[tokio::test]
    async fn mcp_workflow_capability_approve_matches_shared_contract() {
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
                surface == "workflow" && name == "workflow.capability_approve"
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
                        status: StatusCode::ACCEPTED.as_u16(),
                        content_type: "application/json".to_owned(),
                        body: serde_json::to_vec(&result).unwrap(),
                    }
                })
            }
        }

        let contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/workflow-capability-approve-mcp-v1.json"
        ))
        .unwrap();
        let expected = &contract["expected"];
        let result = json!({
            "specs": [],
            "approvals": [expected["approval"]],
            "revoked": [],
            "audit_log": expected["audit_actions"].as_array().unwrap().iter().enumerate().map(|(index, action)| json!({
                "action": action,
                "required": expected["audit_required"][index],
                "outcome": "allow"
            })).collect::<Vec<_>>(),
            "current_subject": expected["current_subject"],
            "effective_capabilities": expected["effective_capabilities_include"]
        });
        let seen = Arc::new(Mutex::new(None));
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: Vec::new(),
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig::default(),
            },
            Arc::new(RequestRecorder(Arc::clone(&seen), result)),
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
        let body = to_bytes(response.into_body(), 8192).await.unwrap();
        let wire: Value = serde_json::from_slice(&body).unwrap();
        let result = &wire["result"];
        for field in expected["required_fields"].as_array().unwrap() {
            let field = field.as_str().unwrap();
            assert!(result.get(field).is_some(), "missing {field}");
        }
        assert_eq!(result["current_subject"], expected["current_subject"]);
        assert!(result["approvals"]
            .as_array()
            .unwrap()
            .contains(&expected["approval"]));
        for capability in expected["effective_capabilities_include"].as_array().unwrap() {
            assert!(result["effective_capabilities"]
                .as_array()
                .unwrap()
                .contains(capability));
        }
        let audit = result["audit_log"].as_array().unwrap();
        let actions = audit.iter().rev().take(2).rev().map(|entry| entry["action"].clone()).collect::<Vec<_>>();
        assert_eq!(actions, *expected["audit_actions"].as_array().unwrap());
        assert!(audit.iter().rev().take(2).all(|entry| entry["outcome"] == "allow"));
        let required = audit
            .iter()
            .rev()
            .take(2)
            .rev()
            .map(|entry| entry["required"].clone())
            .collect::<Vec<_>>();
        assert_eq!(required, *expected["audit_required"].as_array().unwrap());

        let request = seen.lock().unwrap().clone().unwrap();
        assert_eq!(request.method, contract["rust_request"]["method"]);
        assert_eq!(request.path_and_query, contract["rust_request"]["path_and_query"]);
        assert_eq!(request.body, contract["rust_request"]["body"].as_str().unwrap().as_bytes());
    }

    #[tokio::test]
    async fn mcp_workflow_capability_revoke_matches_shared_contract() {
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
                surface == "workflow" && name == "workflow.capability_revoke"
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
                        status: StatusCode::ACCEPTED.as_u16(),
                        content_type: "application/json".to_owned(),
                        body: serde_json::to_vec(&result).unwrap(),
                    }
                })
            }
        }

        let contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/workflow-capability-revoke-mcp-v1.json"
        ))
        .unwrap();
        let expected = &contract["expected"];
        let result = json!({
            "specs": [],
            "approvals": [],
            "revoked": [expected["revocation"]],
            "audit_log": expected["audit_actions"].as_array().unwrap().iter().enumerate().map(|(index, action)| json!({
                "action": action,
                "required": expected["audit_required"][index],
                "outcome": "allow"
            })).collect::<Vec<_>>(),
            "current_subject": expected["current_subject"],
            "effective_capabilities": []
        });
        let seen = Arc::new(Mutex::new(None));
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: Vec::new(),
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig::default(),
            },
            Arc::new(RequestRecorder(Arc::clone(&seen), result)),
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
        let result = &wire["result"];
        for field in expected["required_fields"].as_array().unwrap() {
            let field = field.as_str().unwrap();
            assert!(result.get(field).is_some(), "missing {field}");
        }
        assert_eq!(result["current_subject"], expected["current_subject"]);
        assert!(result["revoked"]
            .as_array()
            .unwrap()
            .contains(&expected["revocation"]));
        assert!(expected["effective_capabilities_exclude"]
            .as_array()
            .unwrap()
            .iter()
            .all(|capability| !result["effective_capabilities"]
                .as_array()
                .unwrap()
                .contains(capability)));
        let audit = result["audit_log"].as_array().unwrap();
        let actions = audit
            .iter()
            .rev()
            .take(2)
            .rev()
            .map(|entry| entry["action"].clone())
            .collect::<Vec<_>>();
        assert_eq!(actions, *expected["audit_actions"].as_array().unwrap());
        assert!(audit.iter().rev().take(2).all(|entry| entry["outcome"] == "allow"));
        let required = audit
            .iter()
            .rev()
            .take(2)
            .rev()
            .map(|entry| entry["required"].clone())
            .collect::<Vec<_>>();
        assert_eq!(required, *expected["audit_required"].as_array().unwrap());

        let request = seen.lock().unwrap().clone().unwrap();
        assert_eq!(request.method, contract["rust_request"]["method"]);
        assert_eq!(request.path_and_query, contract["rust_request"]["path_and_query"]);
        assert_eq!(request.body, contract["rust_request"]["body"].as_str().unwrap().as_bytes());
    }
