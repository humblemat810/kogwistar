    #[test]
    fn run_steps_limit_is_mcp_scoped_and_rest_shape_remains_unbounded() {
        let event = |seq: i64| ServerRunEvent {
            seq,
            run_id: "steps-run".to_owned(),
            event_type: "workflow.recorded_transition.v1".to_owned(),
            payload_json: json!({
                "transition_id": format!("transition-{seq}"),
                "reduced": {
                    "state": {
                        "workflow_id": "workflow-a",
                        "last_step_seq": seq,
                        "last_node_id": format!("node-{seq}"),
                    },
                    "result": {"step": seq},
                    "errors": [],
                    "server_status": "running",
                }
            })
            .to_string(),
            created_at_ms: seq,
        };
        let request = |path_and_query: &str| ApiEffectRequest {
            contract_version: 1,
            method: "GET".to_owned(),
            path_and_query: path_and_query.to_owned(),
            body: Vec::new(),
            principal: json!({}),
        };
        let events = vec![event(1), event(2)];

        let limited = runtime_inspection_effect(
            &request("/api/workflow/runs/steps-run/steps?limit=1"),
            "steps-run",
            &["steps"],
            events.clone(),
        );
        let limited: Value = serde_json::from_slice(&limited.body).unwrap();
        assert_eq!(limited["steps"].as_array().unwrap().len(), 1);
        assert_eq!(limited["steps"][0]["step_seq"], 1);

        let rest = runtime_inspection_effect(
            &request("/api/workflow/runs/steps-run/steps"),
            "steps-run",
            &["steps"],
            events,
        );
        let rest: Value = serde_json::from_slice(&rest.body).unwrap();
        assert_eq!(rest["steps"].as_array().unwrap().len(), 2);
        assert!(rest.get("truncated").is_none());
    }

    #[tokio::test]
    async fn mcp_conversation_ask_uses_authorized_send_message_syscall() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use std::future::Future;
        use std::pin::Pin;
        use std::sync::{Arc, Mutex};
        use tower::ServiceExt;

        let ask_contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/conversation-ask-mcp-v1.json"
        ))
        .unwrap();

        #[derive(Clone)]
        struct RequestRecorder(Arc<Mutex<Option<ApiEffectRequest>>>);

        impl ApplicationService for RequestRecorder {
            fn supports_mcp_tool(&self, surface: &str, name: &str) -> bool {
                surface == "conversation" && name == "conversation.ask"
            }

            fn execute(
                &self,
                request: ApiEffectRequest,
            ) -> Pin<Box<dyn Future<Output = ApiEffectResponse> + Send + '_>> {
                let seen = Arc::clone(&self.0);
                Box::pin(async move {
                    *seen.lock().unwrap() = Some(request);
                    ApiEffectResponse {
                        status: StatusCode::OK.as_u16(),
                        content_type: "application/json".to_owned(),
                        body: br#"{"version":"v1","op":"send_message","status":"ok","result":{"run_id":"run-1","conversation_id":"conv-ask-contract","workflow_id":"agentic_answering.v2","turn_node_id":"turn-1","status":"queued","admission":"accepted","lane_message_id":"lane-1"}}"#.to_vec(),
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
        let listed = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/conversation")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":"list-1","method":"tools/list"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let listed_body = to_bytes(listed.into_body(), 4096).await.unwrap();
        let listed_value: Value = serde_json::from_slice(&listed_body).unwrap();
        let listed_names = listed_value["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|tool| tool["name"].as_str())
            .collect::<Vec<_>>();
        assert!(listed_names.contains(&"conversation.ask"));
        assert!(!listed_names.contains(&"conversation.create"));
        assert!(!listed_names.contains(&"conversation.get_transcript"));

        let malformed = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/conversation")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":"ask-invalid","method":"tools/call","params":{"name":"conversation.ask","arguments":{"conversation_id":"conv-1","text":"hello","unexpected":true}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let malformed_body = to_bytes(malformed.into_body(), 4096).await.unwrap();
        let malformed_value: Value = serde_json::from_slice(&malformed_body).unwrap();
        assert_eq!(malformed_value["error"]["code"], -32602);
        assert!(seen.lock().unwrap().is_none());

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/conversation")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "jsonrpc": "2.0",
                            "id": "ask-1",
                            "method": "tools/call",
                            "params": {
                                "name": ask_contract["tool"],
                                "arguments": ask_contract["arguments"],
                            }
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 4096).await.unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        let mut normalized = value["result"].clone();
        normalized["run_id"] = json!("<generated>");
        normalized["user_turn_node_id"] = json!("<generated>");
        assert_eq!(normalized, ask_contract["result"]);

        let request = seen.lock().unwrap().clone().unwrap();
        assert_eq!(request.contract_version, 1);
        assert_eq!(request.method, "POST");
        assert_eq!(request.path_and_query, "/api/syscall/v1/send_message");
        assert_eq!(
            serde_json::from_slice::<Value>(&request.body).unwrap(),
            json!({
                "version":"v1",
                "op":"send_message",
                "args": {
                    "conversation_id": ask_contract["arguments"]["conversation_id"],
                    "text": ask_contract["arguments"]["text"],
                    "user_id": ask_contract["arguments"]["user_id"],
                    "workflow_id":"agentic_answering.v2"
                }
            })
        );
    }

    #[tokio::test]
    async fn mcp_conversation_cancel_run_matches_shared_contract() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use std::future::Future;
        use std::pin::Pin;
        use std::sync::{Arc, Mutex};
        use tower::ServiceExt;

        let contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/conversation-cancel-run-mcp-v1.json"
        ))
        .unwrap();
        let seen = Arc::new(Mutex::new(None));

        #[derive(Clone)]
        struct CancelRecorder(Arc<Mutex<Option<ApiEffectRequest>>>);

        impl ApplicationService for CancelRecorder {
            fn supports_mcp_tool(&self, surface: &str, name: &str) -> bool {
                surface == "conversation" && name == "conversation.cancel_run"
            }

            fn execute(
                &self,
                request: ApiEffectRequest,
            ) -> Pin<Box<dyn Future<Output = ApiEffectResponse> + Send + '_>> {
                let seen = Arc::clone(&self.0);
                Box::pin(async move {
                    *seen.lock().unwrap() = Some(request);
                    ApiEffectResponse {
                        status: StatusCode::ACCEPTED.as_u16(),
                        content_type: "application/json".to_owned(),
                        body: br#"{"run_id":"run-cancel-contract","conversation_id":"conv-cancel-contract","workflow_id":"agentic_answering.v2","user_id":"alice","user_turn_node_id":"turn-cancel-contract","assistant_turn_node_id":null,"status":"cancelling","cancel_requested":true,"result":null,"error":null,"created_at_ms":1700000000000,"updated_at_ms":1700000000001,"started_at_ms":null,"finished_at_ms":null,"terminal":false}"#.to_vec(),
                    }
                })
            }
        }

        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: Vec::new(),
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig::default(),
            },
            Arc::new(CancelRecorder(Arc::clone(&seen))),
        );
        let malformed = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/conversation")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":"cancel-invalid","method":"tools/call","params":{"name":"conversation.cancel_run","arguments":{"run_id":"run-cancel-contract","unexpected":true}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let malformed_body = to_bytes(malformed.into_body(), 4096).await.unwrap();
        let malformed_value: Value = serde_json::from_slice(&malformed_body).unwrap();
        assert_eq!(malformed_value["error"]["code"], -32602);
        assert!(seen.lock().unwrap().is_none());

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/conversation")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "jsonrpc":"2.0",
                            "id":"cancel-1",
                            "method":"tools/call",
                            "params":{
                                "name":contract["tool"],
                                "arguments":{"run_id":"run-cancel-contract"}
                            }
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 4096).await.unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        let mut normalized = value["result"].clone();
        normalized["created_at_ms"] = json!("<generated>");
        normalized["updated_at_ms"] = json!("<generated>");
        assert_eq!(normalized, contract["result"]);

        let request = seen.lock().unwrap().clone().unwrap();
        assert_eq!(request.method, "POST");
        assert_eq!(request.path_and_query, "/api/runs/run-cancel-contract/cancel");
        assert!(request.body.is_empty());
    }

    #[tokio::test]
    async fn mcp_conversation_run_operations_use_conversation_scope_end_to_end() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
        use std::time::{SystemTime, UNIX_EPOCH};
        use tower::ServiceExt;

        let status_contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/conversation-run-status-mcp-v1.json"
        ))
        .unwrap();
        let cancel_contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/conversation-cancel-run-mcp-v1.json"
        ))
        .unwrap();
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-mcp-conversation-{nonce}.sqlite3"));
        let service = SqliteRunApplicationService::open(&path).unwrap();
        let denied_run = json!({
            "run_id":"cancel-denied-run",
            "conversation_id":"cancel-denied-conversation",
            "workflow_id":"agentic_answering.v2",
            "user_id":"alice",
            "user_turn_node_id":"cancel-denied-turn",
            "status":"running"
        });
        for run in [
            &status_contract["result"],
            &cancel_contract["result"],
            &denied_run,
        ] {
            service
                .store
                .create_server_run(kogwistar_store::ServerRunCreate {
                    run_id: run["run_id"].as_str().unwrap().to_owned(),
                    conversation_id: run["conversation_id"].as_str().unwrap().to_owned(),
                    workflow_id: run["workflow_id"].as_str().unwrap().to_owned(),
                    user_id: run["user_id"].as_str().map(str::to_owned),
                    user_turn_node_id: run["user_turn_node_id"].as_str().unwrap().to_owned(),
                    status: run["status"].as_str().unwrap().to_owned(),
                })
                .unwrap();
        }

        let key = b"conversation-mcp-auth-secret";
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let make_token = |role: &str, namespace: &str, capabilities: &[&str]| {
            encode(
                &Header::new(Algorithm::HS256),
                &json!({
                    "sub":"alice",
                    "role":role,
                    "ns":[namespace],
                    "capabilities":capabilities,
                    "iss":"local",
                    "exp":now + 300
                }),
                &EncodingKey::from_secret(key),
            )
            .unwrap()
        };
        let request = |id: &str, tool: &str, arguments: Value, bearer: &str| {
            Request::builder()
                .method("POST")
                .uri("/mcp/conversation")
                .header("authorization", format!("Bearer {bearer}"))
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "jsonrpc":"2.0",
                        "id":id,
                        "method":"tools/call",
                        "params":{"name":tool,"arguments":arguments}
                    })
                    .to_string(),
                ))
                .unwrap()
        };
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: Vec::new(),
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig {
                    mode: "dev".to_owned(),
                    algorithm: Some("HS256".to_owned()),
                    key: Some(String::from_utf8_lossy(key).into_owned()),
                    jwks_json: None,
                    issuer: Some("local".to_owned()),
                    audience: None,
                    oidc_providers_json: None,
                    auth_db_url: None,
                },
            },
            Arc::new(service.clone()),
        );

        let status_response = app
            .clone()
            .oneshot(request(
                "status-allowed",
                status_contract["tool"].as_str().unwrap(),
                status_contract["arguments"].clone(),
                &make_token("ro", "conversation", &["workflow.run.read"]),
            ))
            .await
            .unwrap();
        let status_body = to_bytes(status_response.into_body(), 8192).await.unwrap();
        let status_value: Value = serde_json::from_slice(&status_body).unwrap();
        let mut status_result = status_value["result"].clone();
        status_result["created_at_ms"] = json!("<generated>");
        status_result["updated_at_ms"] = json!("<generated>");
        assert_eq!(status_result, status_contract["result"]);

        for (denied_token, expected_code) in [
            (make_token("ro", "conversation", &[]), -32000),
            (
                make_token("ro", "workflow", &["workflow.run.read"]),
                -32602,
            ),
        ] {
            let denied = app
                .clone()
                .oneshot(request(
                    "status-denied",
                    status_contract["tool"].as_str().unwrap(),
                    status_contract["arguments"].clone(),
                    &denied_token,
                ))
                .await
                .unwrap();
            let body = to_bytes(denied.into_body(), 8192).await.unwrap();
            let value: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(value["error"]["code"], expected_code);
            assert!(value.get("result").is_none());
        }

        let denied_cancel = app
            .clone()
            .oneshot(request(
                "cancel-denied",
                "conversation.cancel_run",
                json!({"run_id":"cancel-denied-run"}),
                &make_token("rw", "conversation", &[]),
            ))
            .await
            .unwrap();
        let denied_body = to_bytes(denied_cancel.into_body(), 8192).await.unwrap();
        let denied_value: Value = serde_json::from_slice(&denied_body).unwrap();
        assert_eq!(denied_value["error"]["code"], -32000);
        let unchanged = service
            .store
            .get_server_run("cancel-denied-run")
            .unwrap()
            .unwrap();
        assert_eq!(unchanged.status, "running");
        assert!(!unchanged.cancel_requested);

        let cancel_response = app
            .oneshot(request(
                "cancel-allowed",
                cancel_contract["tool"].as_str().unwrap(),
                cancel_contract["arguments"].clone(),
                &make_token("rw", "conversation", &["workflow.run.write"]),
            ))
            .await
            .unwrap();
        let cancel_body = to_bytes(cancel_response.into_body(), 8192).await.unwrap();
        let cancel_value: Value = serde_json::from_slice(&cancel_body).unwrap();
        let mut cancel_result = cancel_value["result"].clone();
        cancel_result["created_at_ms"] = json!("<generated>");
        cancel_result["updated_at_ms"] = json!("<generated>");
        assert_eq!(cancel_result, cancel_contract["result"]);

        drop(service);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn mcp_conversation_transcript_uses_authorized_postgres_read_route() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use std::future::Future;
        use std::pin::Pin;
        use std::sync::{Arc, Mutex};
        use tower::ServiceExt;

        let transcript_contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/conversation-get-transcript-mcp-v1.json"
        ))
        .unwrap();

        #[derive(Clone)]
        struct TranscriptRecorder(Arc<Mutex<Option<ApiEffectRequest>>>);

        impl ApplicationService for TranscriptRecorder {
            fn supports_mcp_tool(&self, surface: &str, name: &str) -> bool {
                surface == "conversation" && name == "conversation.get_transcript"
            }

            fn execute(
                &self,
                request: ApiEffectRequest,
            ) -> Pin<Box<dyn Future<Output = ApiEffectResponse> + Send + '_>> {
                let seen = Arc::clone(&self.0);
                Box::pin(async move {
                    *seen.lock().unwrap() = Some(request);
                    ApiEffectResponse {
                        status: StatusCode::OK.as_u16(),
                        content_type: "application/json".to_owned(),
                        body: br#"{"conversation_id":"transcript-contract-conversation","turns":[],"schema_version":"v1","implementation_authority":"postgres"}"#.to_vec(),
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
            Arc::new(TranscriptRecorder(Arc::clone(&seen))),
        );
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/mcp-tool-call-error-v1.json"
        ))
        .unwrap();
        let negotiated_error = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header("accept", "application/json, text/event-stream")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        serde_json::to_vec(&json!({
                            "jsonrpc":"2.0",
                            "id":"invalid-tool-call",
                            "method":"tools/call",
                            "params":{
                                "name":fixture["tool_name"],
                                "arguments":fixture["arguments"],
                            }
                        }))
                        .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            negotiated_error.status().as_u16(),
            fixture["expected"]["http_status"].as_u64().unwrap() as u16
        );
        assert!(negotiated_error
            .headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap()
            .starts_with(fixture["expected"]["content_type"].as_str().unwrap()));
        let negotiated_body = to_bytes(negotiated_error.into_body(), 4096)
            .await
            .unwrap();
        let negotiated_sse = std::str::from_utf8(&negotiated_body).unwrap();
        let data = negotiated_sse
            .lines()
            .find_map(|line| line.strip_prefix("data: "))
            .unwrap();
        let message: Value = serde_json::from_str(data).unwrap();
        let result = &message["result"];
        assert_eq!(message["id"], "invalid-tool-call");
        assert_eq!(result["isError"], fixture["expected"]["result_is_error"]);
        assert_eq!(
            result["content"][0]["type"],
            fixture["expected"]["content_item_type"]
        );
        assert_eq!(
            !result["content"][0]["text"].as_str().unwrap_or_default().is_empty(),
            fixture["expected"]["content_text_non_empty"]
        );
        assert_eq!(message.get("error").is_some(), fixture["expected"]["top_level_error"]);
        assert!(seen.lock().unwrap().is_none());

        let malformed = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/conversation")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":"transcript-invalid","method":"tools/call","params":{"name":"conversation.get_transcript","arguments":{"conversation_id":"conv-1","unexpected":true}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let malformed_body = to_bytes(malformed.into_body(), 4096).await.unwrap();
        let malformed_value: Value = serde_json::from_slice(&malformed_body).unwrap();
        assert_eq!(malformed_value["error"]["code"], -32602);
        assert!(seen.lock().unwrap().is_none());

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/conversation")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "jsonrpc":"2.0",
                            "id":"transcript-1",
                            "method":"tools/call",
                            "params": {
                                "name": transcript_contract["tool"],
                                "arguments": transcript_contract["arguments"]
                            }
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 4096).await.unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["result"], transcript_contract["result"]);
        let request = seen.lock().unwrap().clone().unwrap();
        assert_eq!(request.method, "GET");
        assert_eq!(
            request.path_and_query,
            "/api/conversations/transcript%2Dcontract%2Dconversation/turns"
        );
        assert!(request.body.is_empty());
    }

    #[tokio::test]
    async fn mcp_conversation_create_uses_native_postgres_syscall() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use std::future::Future;
        use std::pin::Pin;
        use std::sync::{Arc, Mutex};
        use tower::ServiceExt;

        let contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/conversation-create-mcp-v1.json"
        ))
        .unwrap();

        #[derive(Clone)]
        struct CreateRecorder(Arc<Mutex<Option<ApiEffectRequest>>>);

        impl ApplicationService for CreateRecorder {
            fn supports_mcp_tool(&self, surface: &str, name: &str) -> bool {
                surface == "conversation" && name == "conversation.create"
            }

            fn execute(
                &self,
                request: ApiEffectRequest,
            ) -> Pin<Box<dyn Future<Output = ApiEffectResponse> + Send + '_>> {
                let seen = Arc::clone(&self.0);
                Box::pin(async move {
                    *seen.lock().unwrap() = Some(request);
                    ApiEffectResponse {
                        status: StatusCode::OK.as_u16(),
                        content_type: "application/json".to_owned(),
                        body: br#"{"version":"v1","op":"create_conversation","status":"ok","result":{"conversation_id":"conv-create-test","user_id":"alice","status":"active","start_node_id":"conv-create-start","tail_node_id":"conv-create-start","turn_count":0}}"#.to_vec(),
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
            Arc::new(CreateRecorder(Arc::clone(&seen))),
        );
        let malformed = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/conversation")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":"create-invalid","method":"tools/call","params":{"name":"conversation.create","arguments":{"user_id":"alice","start_node_id":17}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let malformed_body = to_bytes(malformed.into_body(), 4096).await.unwrap();
        let malformed_value: Value = serde_json::from_slice(&malformed_body).unwrap();
        assert_eq!(malformed_value["error"]["code"], -32602);
        assert!(seen.lock().unwrap().is_none());

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/conversation")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "jsonrpc":"2.0",
                            "id":"create-1",
                            "method":"tools/call",
                            "params": {
                                "name": contract["tool"],
                                "arguments": contract["arguments"]
                            }
                        })
                        .to_string(),
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
        assert_eq!(request.path_and_query, "/api/syscall/v1/create_conversation");
        assert_eq!(
            serde_json::from_slice::<Value>(&request.body).unwrap(),
            json!({
                "version":"v1",
                "op":"create_conversation",
                "args": contract["arguments"]
            })
        );
    }
