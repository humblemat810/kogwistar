    #[tokio::test]
    async fn mcp_run_event_timeline_tools_build_the_canonical_application_effect_route() {
        let run_events_contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/workflow-run-events-mcp-v1.json"
        ))
        .unwrap();
        let process_timeline_contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/workflow-process-timeline-mcp-v1.json"
        ))
        .unwrap();
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use std::future::Future;
        use std::pin::Pin;
        use std::sync::{Arc, Mutex};
        use tower::ServiceExt;

        #[derive(Clone)]
        struct RequestRecorder(Arc<Mutex<Option<ApiEffectRequest>>>);

        impl ApplicationService for RequestRecorder {
            fn supports_mcp_tool(&self, surface: &str, name: &str) -> bool {
                surface == "workflow"
                    && matches!(
                        name,
                        "workflow.run_events"
                            | "workflow.process_timeline"
                            | "workflow.process_table"
                            | "workflow.blocked_runs"
                            | "workflow.operator_inbox"
                            | "workflow.run_resume"
                            | "workflow.run_resume_contract"
                    )
            }

            fn execute(
                &self,
                request: ApiEffectRequest,
            ) -> Pin<Box<dyn Future<Output = ApiEffectResponse> + Send + '_>> {
                let seen = Arc::clone(&self.0);
                Box::pin(async move {
                    let result = json!({
                        "method": request.method.clone(),
                        "path_and_query": request.path_and_query.clone(),
                    });
                    *seen.lock().unwrap() = Some(request);
                    ApiEffectResponse {
                        status: StatusCode::OK.as_u16(),
                        content_type: "application/json".to_owned(),
                        body: serde_json::to_vec(&result).unwrap(),
                    }
                })
            }
        }

        let seen = Arc::new(Mutex::new(None));
        let fixture_arguments = run_events_contract["arguments"].clone();
        let make_app = || {
            router_with_application(
                ApiState {
                    health: snapshot(),
                    required_roles: Vec::new(),
                    implementation: ImplementationSnapshot::default(),
                    auth: AuthConfig::default(),
                },
                Arc::new(RequestRecorder(Arc::clone(&seen))),
            )
        };
        let response = make_app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/workflow")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::to_vec(&json!({
                            "jsonrpc":"2.0",
                            "id":1,
                            "method":"tools/call",
                            "params":{
                                "name":run_events_contract["operation"],
                                "arguments":fixture_arguments,
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
        assert_eq!(value["result"]["method"], "GET");
        assert_eq!(
            value["result"]["path_and_query"],
            "/api/workflow/runs/mcp%2Devents%2Dcontract%2Drun/events/poll?after_seq=4&limit=200"
        );
        assert_eq!(run_events_contract["route"], "GET /api/workflow/runs/{run_id}/events/poll");
        assert_eq!(
            run_events_contract["response_fields"],
            json!(["run_id", "events"])
        );
        let request = seen.lock().unwrap().clone().unwrap();
        assert_eq!(request.contract_version, 1);
        assert_eq!(request.method, "GET");
        assert!(request.body.is_empty());

        let default_response = make_app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/workflow")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"workflow.run_events","arguments":{"run_id":"run-1"}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let default_body = to_bytes(default_response.into_body(), 4096).await.unwrap();
        let default_value: Value = serde_json::from_slice(&default_body).unwrap();
        assert_eq!(
            default_value["result"]["path_and_query"],
            "/api/workflow/runs/run%2D1/events/poll?after_seq=0&limit=200"
        );

        let response = make_app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/workflow")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"workflow.run_resume","arguments":{"run_id":"run-6","suspended_node_id":"approve","suspended_token_id":"token-6","client_result":{"status":"success"},"workflow_id":"wf-6","conversation_id":"conv-6","turn_node_id":"turn-6"}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 4096).await.unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            value["result"]["path_and_query"],
            "/api/workflow/runs/run-6/resume?wait_for_completion=true"
        );
        let request = seen.lock().unwrap().clone().unwrap();
        assert_eq!(request.contract_version, 1);
        assert_eq!(request.method, "POST");
        let request_body: Value = serde_json::from_slice(&request.body).unwrap();
        assert_eq!(request_body["suspended_node_id"], "approve");
        assert_eq!(request_body["suspended_token_id"], "token-6");
        assert_eq!(request_body["client_result"]["status"], "success");
        assert_eq!(request_body["workflow_id"], "wf-6");
        assert_eq!(request_body["conversation_id"], "conv-6");

        let response = make_app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/workflow")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"workflow.run_resume_contract","arguments":{"run_id":"run-7"}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 4096).await.unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["result"]["method"], "GET");
        assert_eq!(
            value["result"]["path_and_query"],
            "/api/workflow/runs/run-7/resume-contract"
        );
        let request = seen.lock().unwrap().clone().unwrap();
        assert_eq!(request.contract_version, 1);
        assert_eq!(request.method, "GET");
        assert!(request.body.is_empty());

        let response = make_app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/workflow")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"workflow.blocked_runs","arguments":{"limit":23}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 4096).await.unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["result"]["method"], "GET");
        assert_eq!(
            value["result"]["path_and_query"],
            "/api/workflow/blocked-runs?limit=23"
        );
        let request = seen.lock().unwrap().clone().unwrap();
        assert_eq!(request.contract_version, 1);
        assert!(request.body.is_empty());

        let response = make_app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/workflow")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"workflow.operator_inbox","arguments":{"inbox_id":"ops & triage","status":"pending","limit":12}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 4096).await.unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["result"]["method"], "GET");
        assert_eq!(
            value["result"]["path_and_query"],
            "/api/workflow/operator/inbox?limit=12&inbox_id=ops+%26+triage&status=pending"
        );

        let response = make_app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/workflow")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"workflow.process_table","arguments":{"workflow_id":"wf & one","status":"succeeded","limit":7}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 4096).await.unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["result"]["method"], "GET");
        let request = seen.lock().unwrap().clone().unwrap();
        let parsed_query = url::form_urlencoded::parse(
            request.path_and_query.split_once('?').unwrap().1.as_bytes(),
        )
        .collect::<std::collections::BTreeMap<_, _>>();
        assert_eq!(request.path_and_query.split('?').next(), Some("/api/workflow/process-table"));
        assert_eq!(parsed_query.get("workflow_id").map(|value| value.as_ref()), Some("wf & one"));
        assert_eq!(parsed_query.get("status").map(|value| value.as_ref()), Some("succeeded"));
        assert_eq!(parsed_query.get("limit").map(|value| value.as_ref()), Some("7"));
        assert!(request.body.is_empty());

        let timeline_arguments = process_timeline_contract["arguments"].clone();
        let response = make_app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/workflow")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::to_vec(&json!({
                            "jsonrpc":"2.0",
                            "id":2,
                            "method":"tools/call",
                            "params":{
                                "name":process_timeline_contract["operation"],
                                "arguments":timeline_arguments,
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
        assert_eq!(value["result"]["method"], "GET");
        assert_eq!(
            value["result"]["path_and_query"],
            "/api/workflow/runs/mcp%2Dtimeline%2Dcontract%2Drun/events/poll?after_seq=4&limit=200"
        );
        assert_eq!(
            process_timeline_contract["route"],
            "GET /api/workflow/runs/{run_id}/events/poll"
        );
        assert_eq!(
            process_timeline_contract["response_fields"],
            json!(["run_id", "events"])
        );
        let request = seen.lock().unwrap().clone().unwrap();
        assert_eq!(request.contract_version, 1);
        assert_eq!(request.method, "GET");
        assert!(request.body.is_empty());
    }

    #[test]
    fn process_table_projects_filtered_runs_and_service_event_counts() {
        let request = ApiEffectRequest {
            contract_version: 1,
            method: "GET".to_owned(),
            path_and_query: "/api/workflow/process-table?workflow_id=wf%20one&limit=5"
                .to_owned(),
            body: Vec::new(),
            principal: json!({"sub":"alice","ns":["workflow"]}),
        };
        let run = ServerRun {
            run_id: "run-1".to_owned(),
            conversation_id: "conv-1".to_owned(),
            workflow_id: "wf one".to_owned(),
            user_id: Some("alice".to_owned()),
            user_turn_node_id: Some("user-turn".to_owned()),
            assistant_turn_node_id: Some("assistant-turn".to_owned()),
            status: "succeeded".to_owned(),
            cancel_requested: false,
            result_json: None,
            error_json: None,
            created_at_ms: 10,
            updated_at_ms: 20,
            started_at_ms: Some(11),
            finished_at_ms: Some(19),
        };
        let run_event = ServerRunEvent {
            seq: 3,
            run_id: "run-1".to_owned(),
            event_type: "run.completed".to_owned(),
            payload_json: "{}".to_owned(),
            created_at_ms: 19,
        };
        let service = json!({
            "service_id":"svc-1",
            "service_kind":"workflow",
            "target_kind":"workflow",
            "target_ref":"wf one",
            "target_config":{"conversation_id":"conv-1"},
            "lifecycle_status":"active",
            "enabled":true,
            "created_at_ms":12,
            "updated_at_ms":21,
        });
        let service_event = json!({
            "metadata":{"entity_type":"service_event","service_id":"svc-1"}
        });
        let response = process_table_effect(
            &request,
            vec![run],
            vec![service],
            vec![run_event],
            vec![(1, "node".to_owned(), "event-1".to_owned(), "ADD".to_owned(), service_event.to_string())],
        );

        assert_eq!(response.status, StatusCode::OK.as_u16());
        let value: Value = serde_json::from_slice(&response.body).unwrap();
        let processes = value["processes"].as_array().unwrap();
        assert_eq!(processes.len(), 2);
        assert_eq!(processes[0]["process_id"], "svc-1");
        assert_eq!(processes[0]["process_kind"], "service");
        assert_eq!(processes[0]["event_count"], 1);
        assert_eq!(processes[1]["process_id"], "run-1");
        assert_eq!(processes[1]["process_kind"], "workflow_run");
        assert_eq!(processes[1]["event_count"], 1);
        assert_eq!(processes[1]["last_event_type"], "run.completed");
        assert_eq!(processes[1]["terminal"], true);
    }

    #[test]
    fn operator_inbox_filters_graph_visibility_and_security_scope_before_limit() {
        let request = ApiEffectRequest {
            contract_version: 1,
            method: "GET".to_owned(),
            path_and_query: "/api/workflow/operator/inbox?inbox_id=ops&limit=1".to_owned(),
            body: Vec::new(),
            principal: json!({
                "sub":"alice",
                "ns":["workflow"],
                "storage_ns":"tenant-a",
                "execution_ns":"workflow",
                "security_scope":"team-a",
            }),
        };
        let lane = |message_id: &str, seq: i64| ProjectedLaneMessage {
            message_id: message_id.to_owned(),
            namespace: "tenant-a".to_owned(),
            purpose: "operator".to_owned(),
            inbox_id: "ops".to_owned(),
            conversation_id: "conversation-1".to_owned(),
            recipient_id: "operator".to_owned(),
            sender_id: "worker".to_owned(),
            msg_type: "maintenance.result".to_owned(),
            status: "pending".to_owned(),
            seq,
            conversation_seq: seq,
            claimed_by: None,
            lease_until: None,
            retry_count: 0,
            created_at: seq,
            available_at: seq,
            run_id: None,
            step_id: None,
            correlation_id: None,
            payload_json: None,
            error_json: None,
            prev_message_id: None,
            next_message_id: None,
            inbox_tail_message_id: None,
            conversation_tail_message_id: None,
        };
        let node = |id: &str, security_scope: &str, shared: bool| GraphRecord {
            id: id.to_owned(),
            document: None,
            metadata: serde_json::Map::from_iter([
                ("security_scope".to_owned(), json!(security_scope)),
                ("shared_inbox".to_owned(), json!(shared)),
                ("visibility".to_owned(), json!(if shared { "shared" } else { "private" })),
            ]),
            embedding: None,
        };
        let response = operator_inbox_effect(
            &request,
            &[lane("hidden-by-scope", 1), lane("shared-visible", 2), lane("private-visible", 3)],
            &[
                node("hidden-by-scope", "team-b", false),
                node("shared-visible", "team-b", true),
                node("private-visible", "team-a", false),
            ],
        );
        assert_eq!(response.status, StatusCode::OK.as_u16());
        let value: Value = serde_json::from_slice(&response.body).unwrap();
        let messages = value["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["message_id"], "shared-visible");
        assert_eq!(messages[0]["visibility"], "shared");
        assert_eq!(messages[0]["storage_namespace"], "tenant-a");
    }

    #[test]
    fn operator_inbox_omits_missing_acl_visible_graph_nodes() {
        let request = ApiEffectRequest {
            contract_version: 1,
            method: "GET".to_owned(),
            path_and_query: "/api/workflow/operator/inbox".to_owned(),
            body: Vec::new(),
            principal: json!({"sub":"alice", "ns":["workflow"], "security_scope":"team-a"}),
        };
        let lane = ProjectedLaneMessage {
            message_id: "orphaned-projection".to_owned(),
            namespace: "workflow".to_owned(),
            purpose: "operator".to_owned(),
            inbox_id: "ops".to_owned(),
            conversation_id: "conversation-1".to_owned(),
            recipient_id: "operator".to_owned(),
            sender_id: "worker".to_owned(),
            msg_type: "maintenance.result".to_owned(),
            status: "pending".to_owned(),
            seq: 1,
            conversation_seq: 1,
            claimed_by: None,
            lease_until: None,
            retry_count: 0,
            created_at: 1,
            available_at: 1,
            run_id: None,
            step_id: None,
            correlation_id: None,
            payload_json: Some("must-not-leak".to_owned()),
            error_json: None,
            prev_message_id: None,
            next_message_id: None,
            inbox_tail_message_id: None,
            conversation_tail_message_id: None,
        };
        let response = operator_inbox_effect(&request, &[lane], &[]);
        let value: Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(value["messages"], json!([]));
        assert!(!response.body.windows(14).any(|window| window == b"must-not-leak"));
    }

    #[test]
    fn operator_inbox_requires_workflow_scope_read_role_and_project_view() {
        let capabilities = Arc::new(Mutex::new(CapabilityState::default()));
        let request = |principal| ApiEffectRequest {
            contract_version: 1,
            method: "GET".to_owned(),
            path_and_query: "/api/workflow/operator/inbox".to_owned(),
            body: Vec::new(),
            principal,
        };
        assert!(operator_inbox_authorization_error(
            &request(json!({"role":"ro", "ns":["workflow"], "capabilities":["project_view"]})),
            &capabilities,
        ).is_none());
        let wrong_namespace = operator_inbox_authorization_error(
            &request(json!({"role":"ro", "ns":["knowledge"], "capabilities":["project_view"]})),
            &capabilities,
        ).unwrap();
        assert_eq!(wrong_namespace.status, StatusCode::FORBIDDEN.as_u16());
        let missing_capability = operator_inbox_authorization_error(
            &request(json!({"role":"ro", "ns":["workflow"], "capabilities":["read_graph"]})),
            &capabilities,
        ).unwrap();
        assert_eq!(missing_capability.status, StatusCode::FORBIDDEN.as_u16());
    }

    #[tokio::test]
    async fn mcp_operator_inbox_matches_shared_contract() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use std::future::Future;
        use std::pin::Pin;
        use std::sync::{Arc, Mutex};
        use tower::ServiceExt;

        let contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/workflow-operator-inbox-mcp-v1.json"
        ))
        .unwrap();

        #[derive(Clone)]
        struct ContractService(Arc<Mutex<Option<ApiEffectRequest>>>, Value);

        impl ApplicationService for ContractService {
            fn supports_mcp_tool(&self, surface: &str, name: &str) -> bool {
                surface == "workflow" && name == "workflow.operator_inbox"
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

        let seen = Arc::new(Mutex::new(None));
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: Vec::new(),
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig::default(),
            },
            Arc::new(ContractService(
                Arc::clone(&seen),
                contract["expected_result"].clone(),
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
                            "jsonrpc":"2.0",
                            "id":1,
                            "method":"tools/call",
                            "params":{
                                "name":contract["operation"],
                                "arguments":contract["arguments"]
                            }
                        }))
                        .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 8192).await.unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["result"], contract["expected_result"]);
        let request = seen.lock().unwrap().clone().unwrap();
        assert_eq!(request.contract_version, 1);
        let (method, path) = contract["route"].as_str().unwrap().split_once(' ').unwrap();
        assert_eq!(request.method, method);
        assert_eq!(request.path_and_query, path);
        assert!(request.body.is_empty());
    }

    #[tokio::test]
    async fn mcp_run_evidence_builds_encoded_read_only_route() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use std::future::Future;
        use std::pin::Pin;
        use std::sync::{Arc, Mutex};
        use tower::ServiceExt;

        #[derive(Clone)]
        struct RequestRecorder(Arc<Mutex<Option<ApiEffectRequest>>>);

        impl ApplicationService for RequestRecorder {
            fn supports_mcp_tool(&self, surface: &str, name: &str) -> bool {
                surface == "workflow" && name == "workflow.run_evidence"
            }

            fn execute(
                &self,
                request: ApiEffectRequest,
            ) -> Pin<Box<dyn Future<Output = ApiEffectResponse> + Send + '_>> {
                let seen = Arc::clone(&self.0);
                Box::pin(async move {
                    let result: Value = serde_json::from_str(include_str!(
                        "../../../../../contracts/golden/run-evidence-v1.json"
                    ))
                    .unwrap();
                    *seen.lock().unwrap() = Some(request);
                    ApiEffectResponse {
                        status: StatusCode::OK.as_u16(),
                        content_type: "application/json".to_owned(),
                        body: serde_json::to_vec(&result).unwrap(),
                    }
                })
            }
        }

        let seen = Arc::new(Mutex::new(None));
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: Vec::new(),
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig::default(),
            },
            Arc::new(RequestRecorder(Arc::clone(&seen))),
        );
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/workflow")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"workflow.run_evidence","arguments":{"run_id":"run/42?x"}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 4096).await.unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        let expected: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/run-evidence-v1.json"
        ))
        .unwrap();
        assert_eq!(value["result"], expected);
        let request = seen.lock().unwrap().clone().unwrap();
        assert_eq!(request.contract_version, 1);
        assert_eq!(request.method, "GET");
        assert_eq!(
            request.path_and_query,
            "/api/workflow/runs/run%2F42%3Fx/evidence"
        );
        assert!(request.body.is_empty());

        let invalid = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/workflow")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"workflow.run_evidence","arguments":{"run_id":"run-1","unexpected":true}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let invalid_body = to_bytes(invalid.into_body(), 4096).await.unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&invalid_body).unwrap()["error"]["code"],
            -32602
        );
        let seen_request = seen.lock().unwrap();
        assert_eq!(
            seen_request.as_ref().unwrap().path_and_query,
            "/api/workflow/runs/run%2F42%3Fx/evidence"
        );
    }

    #[tokio::test]
    async fn mcp_run_lineage_builds_bounded_encoded_read_only_route() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use std::future::Future;
        use std::pin::Pin;
        use std::sync::{Arc, Mutex};
        use tower::ServiceExt;

        #[derive(Clone)]
        struct RequestRecorder(Arc<Mutex<Option<ApiEffectRequest>>>, Value, Value);

        impl ApplicationService for RequestRecorder {
            fn supports_mcp_tool(&self, surface: &str, name: &str) -> bool {
                surface == "workflow"
                    && matches!(name, "workflow.run_lineage" | "workflow.run_steps")
            }

            fn execute(
                &self,
                request: ApiEffectRequest,
            ) -> Pin<Box<dyn Future<Output = ApiEffectResponse> + Send + '_>> {
                let seen = Arc::clone(&self.0);
                let steps_result = self.1.clone();
                let lineage_result = self.2.clone();
                Box::pin(async move {
                    let result = if request.path_and_query.ends_with("/lineage?limit=2") {
                        lineage_result
                    } else if request.path_and_query.ends_with("/steps?limit=25") {
                        steps_result
                    } else {
                        json!({"path_and_query": request.path_and_query.clone()})
                    };
                    *seen.lock().unwrap() = Some(request);
                    ApiEffectResponse {
                        status: StatusCode::OK.as_u16(),
                        content_type: "application/json".to_owned(),
                        body: serde_json::to_vec(&result).unwrap(),
                    }
                })
            }
        }

        let seen = Arc::new(Mutex::new(None));
        let steps_contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/workflow-run-steps-mcp-v1.json"
        ))
        .unwrap();
        let lineage_contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/workflow-run-lineage-v1.json"
        ))
        .unwrap();
        let mut lineage_result = lineage_contract["response"].clone();
        lineage_result["truncated"] = json!(false);
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: Vec::new(),
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig::default(),
            },
            Arc::new(RequestRecorder(
                Arc::clone(&seen),
                steps_contract["result"].clone(),
                lineage_result,
            )),
        );
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/workflow")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"workflow.run_lineage","arguments":{"run_id":"run/42?x","limit":25}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 4096).await.unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            value["result"]["path_and_query"],
            "/api/workflow/runs/run%2F42%3Fx/lineage?limit=25"
        );
        let request = seen.lock().unwrap().clone().unwrap();
        assert_eq!(request.method, "GET");
        assert!(request.body.is_empty());

        let lineage_response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/workflow")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"workflow.run_lineage","arguments":{"run_id":"lineage-child","limit":2}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(lineage_response.status(), StatusCode::OK);
        let lineage_body = to_bytes(lineage_response.into_body(), 4096).await.unwrap();
        let lineage_value: Value = serde_json::from_slice(&lineage_body).unwrap();
        let expected_lineage = {
            let mut expected = lineage_contract["response"].clone();
            expected["truncated"] = json!(false);
            expected
        };
        assert_eq!(lineage_value["result"], expected_lineage);
        assert_eq!(
            seen.lock().unwrap().as_ref().unwrap().path_and_query,
            "/api/workflow/runs/lineage%2Dchild/lineage?limit=2"
        );

        let steps_response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/workflow")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::to_string(&json!({
                            "jsonrpc": "2.0",
                            "id": 6,
                            "method": "tools/call",
                            "params": {
                                "name": steps_contract["tool"],
                                "arguments": steps_contract["arguments"]
                            }
                        }))
                        .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(steps_response.status(), StatusCode::OK);
        let steps_body = to_bytes(steps_response.into_body(), 4096).await.unwrap();
        let steps_value: Value = serde_json::from_slice(&steps_body).unwrap();
        assert_eq!(steps_value["result"], steps_contract["result"]);
        assert_eq!(
            seen.lock().unwrap().as_ref().unwrap().path_and_query,
            "/api/workflow/runs/run%2F42%3Fx/steps?limit=25"
        );

        let invalid = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/workflow")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"workflow.run_lineage","arguments":{"run_id":"run-1","limit":1001}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let invalid_body = to_bytes(invalid.into_body(), 4096).await.unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&invalid_body).unwrap()["error"]["code"],
            -32602
        );
    }
