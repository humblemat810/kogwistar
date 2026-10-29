    #[tokio::test]
    async fn inline_runtime_iteration_uses_typed_hook_not_generic_effect_path() {
        use std::future::Future;
        use std::pin::Pin;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        struct TypedOnlyService {
            typed_calls: Arc<AtomicUsize>,
            generic_effect_calls: Arc<AtomicUsize>,
        }

        impl ApplicationService for TypedOnlyService {
            fn execute_inline_native_work_once(
                &self,
            ) -> Pin<Box<dyn Future<Output = Result<bool, String>> + Send + '_>> {
                let typed_calls = Arc::clone(&self.typed_calls);
                Box::pin(async move {
                    typed_calls.fetch_add(1, Ordering::SeqCst);
                    Ok(true)
                })
            }

            fn execute(
                &self,
                _request: ApiEffectRequest,
            ) -> Pin<Box<dyn Future<Output = ApiEffectResponse> + Send + '_>> {
                let generic_effect_calls = Arc::clone(&self.generic_effect_calls);
                Box::pin(async move {
                    generic_effect_calls.fetch_add(1, Ordering::SeqCst);
                    ApiEffectResponse {
                        status: StatusCode::OK.as_u16(),
                        content_type: "application/json".to_owned(),
                        body: b"{}".to_vec(),
                    }
                })
            }
        }

        let service = TypedOnlyService {
            typed_calls: Arc::new(AtomicUsize::new(0)),
            generic_effect_calls: Arc::new(AtomicUsize::new(0)),
        };

        assert!(run_one_inline_native_worker_iteration(&service)
            .await
            .unwrap());
        assert_eq!(service.typed_calls.load(Ordering::SeqCst), 1);
        assert_eq!(service.generic_effect_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn inline_runtime_supervisor_restarts_worker_after_panic() {
        use std::future::Future;
        use std::pin::Pin;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;
        use std::time::Duration;

        struct PanicOnceService(Arc<AtomicUsize>);

        impl ApplicationService for PanicOnceService {
            fn execute_inline_native_work_once(
                &self,
            ) -> Pin<Box<dyn Future<Output = Result<bool, String>> + Send + '_>> {
                let calls = Arc::clone(&self.0);
                Box::pin(async move {
                    if calls.fetch_add(1, Ordering::SeqCst) == 0 {
                        panic!("injected native worker panic");
                    }
                    Ok(false)
                })
            }

            fn execute(
                &self,
                _request: ApiEffectRequest,
            ) -> Pin<Box<dyn Future<Output = ApiEffectResponse> + Send + '_>> {
                Box::pin(async {
                    ApiEffectResponse {
                        status: StatusCode::OK.as_u16(),
                        content_type: "application/json".to_owned(),
                        body: b"{}".to_vec(),
                    }
                })
            }
        }

        let calls = Arc::new(AtomicUsize::new(0));
        let (shutdown, shutdown_rx) = tokio::sync::oneshot::channel();
        let supervisor = tokio::spawn(supervise_inline_native_worker(
            Arc::new(PanicOnceService(Arc::clone(&calls))),
            shutdown_rx,
        ));

        tokio::time::timeout(Duration::from_secs(5), async {
            while calls.load(Ordering::SeqCst) < 2 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("supervisor should restart the worker after panic");

        shutdown.send(()).expect("supervisor should still be running");
        tokio::time::timeout(Duration::from_secs(2), supervisor)
            .await
            .expect("supervisor should stop on shutdown")
            .expect("supervisor task should complete");
        let calls_after_shutdown = calls.load(Ordering::SeqCst);
        assert!(calls_after_shutdown >= 2);
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(calls.load(Ordering::SeqCst), calls_after_shutdown);
    }

    #[tokio::test]
    async fn mcp_run_events_builds_the_canonical_application_effect_route() {
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
                surface == "workflow" && name == "workflow.run_events"
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
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/workflow")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"workflow.run_events","arguments":{"run_id":"run-1","after_seq":7,"limit":25}}}"#,
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
            "/api/runs/run-1/events/poll?after_seq=7&limit=25"
        );
        let request = seen.lock().unwrap().clone().unwrap();
        assert_eq!(request.contract_version, 1);
        assert_eq!(request.method, "GET");
        assert!(request.body.is_empty());
    }

    #[tokio::test]
    async fn mcp_conversation_ask_uses_authorized_send_message_syscall() {
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
                        body: br#"{"version":"v1","op":"send_message","status":"ok","result":{"run_id":"run-1","turn_node_id":"turn-1"}}"#.to_vec(),
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
        assert_eq!(
            listed_value["result"]["tools"][0]["name"],
            "conversation.ask"
        );

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
                        r#"{"jsonrpc":"2.0","id":"ask-1","method":"tools/call","params":{"name":"conversation.ask","arguments":{"conversation_id":"conv-1","text":"hello","user_id":"alice"}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 4096).await.unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["result"], json!({"run_id":"run-1","turn_node_id":"turn-1"}));

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
                    "conversation_id":"conv-1",
                    "text":"hello",
                    "user_id":"alice",
                    "workflow_id":"agentic_answering.v2"
                }
            })
        );
    }

    #[tokio::test]
    async fn sqlite_run_service_handles_get_events_and_cancel() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use kogwistar_store::NamedProjectionWrite;
        use std::time::{SystemTime, UNIX_EPOCH};
        use tower::ServiceExt;

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-api-{nonce}.sqlite3"));
        let store = SqliteStore::open(&path).unwrap();
        store
            .create_server_run(kogwistar_store::ServerRunCreate {
                run_id: "run-1".to_owned(),
                conversation_id: "conversation-1".to_owned(),
                workflow_id: "workflow-1".to_owned(),
                user_id: Some("user-1".to_owned()),
                user_turn_node_id: "turn-1".to_owned(),
                status: "running".to_owned(),
            })
            .unwrap();
        store
            .append_server_run_event("run-1", "run.started", r#"{"source":"test"}"#.to_owned())
            .unwrap();
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: vec!["reader".to_owned()],
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig::default(),
            },
            Arc::new(SqliteRunApplicationService::open(&path).unwrap()),
        );

        let submit = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/workflow/runs")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"run_id":"caller-run-id","workflow_id":"workflow-submit","conversation_id":"conversation-submit","initial_state":{"seed":1,"effective_capabilities":["admin"],"_rt_effective_capabilities":["admin"]},"priority_class":"foreground","runtime_kind":"sync","start_node_id":"entry","node_ops":{"entry":"begin"}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(submit.status(), StatusCode::ACCEPTED);
        let submit_body = to_bytes(submit.into_body(), 8192).await.unwrap();
        let submitted = serde_json::from_slice::<Value>(&submit_body).unwrap();
        let submitted_run_id = submitted["run_id"].as_str().unwrap();
        assert_eq!(submitted_run_id, "caller-run-id");
        let queued = store
            .list_projected_lane_messages(LaneMessageFilter {
                namespace: Some("workflow".to_owned()),
                inbox_id: Some("workflow-runtime".to_owned()),
                correlation_id: Some(submitted_run_id.to_owned()),
                ..LaneMessageFilter::default()
            })
            .unwrap();
        let queued_payload =
            serde_json::from_str::<Value>(queued[0].payload_json.as_deref().unwrap()).unwrap();
        assert!(
            !queued_payload["initial_state"]["effective_capabilities"]
                .as_array()
                .unwrap()
                .iter()
                .any(|value| value == "admin")
        );
        assert_eq!(
            queued_payload["initial_state"]["_rt_effective_capabilities"],
            queued_payload["initial_state"]["effective_capabilities"]
        );
        let retried_submit = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/workflow/runs")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"run_id":"caller-run-id","workflow_id":"workflow-submit","conversation_id":"conversation-submit","initial_state":{"seed":1,"effective_capabilities":["admin"],"_rt_effective_capabilities":["admin"]},"priority_class":"foreground","runtime_kind":"sync","start_node_id":"entry","node_ops":{"entry":"begin"}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(retried_submit.status(), StatusCode::ACCEPTED);
        let retried_submit = serde_json::from_slice::<Value>(
            &to_bytes(retried_submit.into_body(), 8192).await.unwrap(),
        )
        .unwrap();
        assert_eq!(retried_submit["idempotent"], true);
        assert_eq!(retried_submit["run_id"], "caller-run-id");

        let conflicting_submit = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/workflow/runs")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"run_id":"caller-run-id","workflow_id":"workflow-submit","conversation_id":"conversation-submit","initial_state":{"seed":2,"effective_capabilities":["admin"],"_rt_effective_capabilities":["admin"]},"priority_class":"foreground","runtime_kind":"sync","start_node_id":"entry","node_ops":{"entry":"begin"}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(conflicting_submit.status(), StatusCode::CONFLICT);
        assert_eq!(
            store
                .get_server_run(submitted_run_id)
                .unwrap()
                .unwrap()
                .status,
            "queued"
        );
        let created_events = store
            .list_server_run_events(submitted_run_id, 0, 10)
            .unwrap();
        assert_eq!(created_events[0].event_type, "run.created");
        let implementation_attestation: Value =
            serde_json::from_str(&created_events[0].payload_json).unwrap();
        assert_eq!(
            implementation_attestation["implementation_attestation"]["schema_version"],
            1
        );
        assert_eq!(
            implementation_attestation["implementation_attestation"]["source"],
            "rust_server_admission"
        );
        assert_eq!(
            implementation_attestation["implementation_attestation"]["scheduler"]["language"],
            "rust"
        );
        assert_eq!(
            implementation_attestation["implementation_attestation"]["scheduler"]["package"],
            "kogwistar-api"
        );
        assert_eq!(
            implementation_attestation["implementation_attestation"]["worker_contract"]["protocol"],
            "sync-v1"
        );
        let claimed = store
            .claim_projected_lane_messages("workflow", "workflow-runtime", "python-worker-1", 1, 60)
            .unwrap();
        assert_eq!(claimed.len(), 1);
        assert_eq!(claimed[0].run_id.as_deref(), Some(submitted_run_id));
        assert_eq!(claimed[0].msg_type, "workflow.run.execute");
        assert_eq!(claimed[0].claimed_by.as_deref(), Some("python-worker-1"));

        let submitted_message_id = claimed[0].message_id.clone();
        // Requeue the direct-store claim; HTTP claim owns the tested handoff.
        store
            .requeue_projected_lane_message(&submitted_message_id, "python-worker-1", None, 0)
            .unwrap();
        let claim = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/claim")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"claimed_by":"python-worker-2","limit":1,"lease_seconds":60}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let claim_status = claim.status();
        let claim_body = to_bytes(claim.into_body(), 16384).await.unwrap();
        assert_eq!(
            claim_status,
            StatusCode::OK,
            "{}",
            String::from_utf8_lossy(&claim_body)
        );
        let claim_value = serde_json::from_slice::<Value>(&claim_body).unwrap();
        let work = &claim_value["work"][0];
        assert_eq!(work["run_id"], submitted_run_id);
        assert_eq!(work["step_id"], "entry");
        assert_eq!(work["payload"]["start_node_id"], "entry");
        assert_eq!(work["payload"]["op"], "begin");
        let result_payload = json!({
            "handoff": {
                "message_id": work["message_id"],
                "claimed_by": work["claimed_by"],
                "run_id": work["run_id"],
                "step_id": work["step_id"],
                "correlation_id": work["correlation_id"],
            },
            "transition": {
                "contract_version": 1,
                "transition_id": format!("result-{submitted_run_id}"),
                "expected_event_seq": work["expected_event_seq"],
                "kind": "recorded_step_success",
                "run_id": submitted_run_id,
                "workflow_id": "workflow-submit",
                "conversation_id": "conversation-submit",
                "step_seq": 0,
                "node_id": "entry",
                "token_id": submitted_run_id,
                "parent_token_id": null,
                "state_update": [["u", {"answer": "worker"}]],
                "frontier": {
                    "pending": [],
                    "suspended": [],
                    "join_node_ids": [],
                    "join_outstanding": [],
                    "join_waiters": {},
                },
                "result": {"answer": "worker"},
            }
        });
        let result_bytes = serde_json::to_vec(&result_payload).unwrap();
        let result = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/results")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(result_bytes.clone()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let result_status = result.status();
        let result_body = to_bytes(result.into_body(), 16384).await.unwrap();
        assert_eq!(
            result_status,
            StatusCode::OK,
            "{}",
            String::from_utf8_lossy(&result_body)
        );
        let duplicate_result = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/results")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(result_bytes))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(duplicate_result.status(), StatusCode::OK);
        assert_eq!(
            store
                .get_server_run(submitted_run_id)
                .unwrap()
                .unwrap()
                .status,
            "succeeded"
        );
        assert_eq!(
            store
                .get_projected_lane_message(&submitted_message_id)
                .unwrap()
                .unwrap()
                .status,
            "completed"
        );
        assert_eq!(
            store
                .read_recorded_runtime_state(
                    submitted_run_id,
                    "workflow-submit",
                    "conversation-submit",
                )
                .unwrap()
                .unwrap()
                .state["answer"],
            "worker"
        );
        for (path, required_keys) in [
            (
                "/api/workflow/resources",
                &[
                    "scheduler",
                    "runs",
                    "cost_ledger",
                    "budget_model",
                    "migration",
                ][..],
            ),
            ("/api/workflow/budget", &["cost_ledger", "budget_model"][..]),
            (
                "/api/workflow/budget/history?limit=10",
                &["cost_ledger", "events"][..],
            ),
            (
                "/api/workflow/lane/progress?conversation_id=conversation-submit&limit=10",
                &["items", "total"][..],
            ),
            (
                "/api/workflow/operator/dashboard?limit=10",
                &["process_table", "message_queue", "resources"][..],
            ),
            (
                "/api/workflow/scheduler/timeline?limit=10",
                &["run_id", "events"][..],
            ),
            (
                "/api/workflow/dead-letters?limit=10",
                &["runs", "limit"][..],
            ),
            ("/api/workflow/catalog/ops", &[][..]),
        ] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .uri(path)
                        .header("x-kogwistar-roles", "reader")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK, "{path}");
            let body = to_bytes(response.into_body(), 65_536).await.unwrap();
            let value = serde_json::from_slice::<Value>(&body).unwrap();
            if path == "/api/workflow/catalog/ops" {
                assert_eq!(value[0]["op"], "start");
                assert_eq!(value[1]["op"], "llm_call");
            }
            for key in required_keys {
                assert!(value.get(key).is_some(), "{path} lacks {key}: {value}");
            }
        }
        let dead_lane = |message_id: &str, run_id: &str| NewProjectedLaneMessage {
            message_id: message_id.to_owned(),
            namespace: "workflow".to_owned(),
            purpose: "system".to_owned(),
            inbox_id: "workflow-runtime".to_owned(),
            conversation_id: "dead-conversation".to_owned(),
            recipient_id: "python-worker".to_owned(),
            sender_id: "rust-scheduler".to_owned(),
            msg_type: "workflow.run.execute".to_owned(),
            status: "dead-letter".to_owned(),
            created_at: 1,
            available_at: 1,
            run_id: Some(run_id.to_owned()),
            step_id: Some("start".to_owned()),
            correlation_id: Some(format!("corr-{run_id}")),
            payload_json: Some("{}".to_owned()),
            error_json: Some(r#"{"error":"boom"}"#.to_owned()),
        };
        store
            .project_lane_message(dead_lane("dead-rest-message", "dead-rest-run"))
            .unwrap();
        let replay = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/workflow/dead-letters/dead-rest-run/replay")
                    .header("x-kogwistar-roles", "reader,rw")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(replay.status(), StatusCode::OK);
        let replay =
            serde_json::from_slice::<Value>(&to_bytes(replay.into_body(), 16_384).await.unwrap())
                .unwrap();
        assert_eq!(replay["replayed"], true);
        assert_eq!(replay["dead_letter"]["status"], "dead-letter");
        assert_eq!(
            store
                .get_projected_lane_message("dead-rest-message")
                .unwrap()
                .unwrap()
                .status,
            "pending"
        );
        store
            .update_projected_lane_message_status("dead-rest-message", "completed", None)
            .unwrap();
        let missing_replay = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/workflow/dead-letters/missing/replay")
                    .header("x-kogwistar-roles", "reader,rw")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let missing_replay = serde_json::from_slice::<Value>(
            &to_bytes(missing_replay.into_body(), 16_384).await.unwrap(),
        )
        .unwrap();
        assert_eq!(
            missing_replay,
            json!({"run_id": "missing", "replayed": false})
        );
        for prefix in ["/api/runs", "/api/workflow/runs"] {
            let steps = app
                .clone()
                .oneshot(
                    Request::builder()
                        .uri(format!("{prefix}/{submitted_run_id}/steps"))
                        .header("x-kogwistar-roles", "reader")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(steps.status(), StatusCode::OK);
            let steps_body = to_bytes(steps.into_body(), 16_384).await.unwrap();
            let steps_value = serde_json::from_slice::<Value>(&steps_body).unwrap();
            assert_eq!(steps_value["steps"].as_array().unwrap().len(), 1);
            assert_eq!(steps_value["steps"][0]["workflow_node_id"], "entry");
            assert_eq!(steps_value["steps"][0]["result"]["answer"], "worker");

            let checkpoints = app
                .clone()
                .oneshot(
                    Request::builder()
                        .uri(format!("{prefix}/{submitted_run_id}/checkpoints"))
                        .header("x-kogwistar-roles", "reader")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(checkpoints.status(), StatusCode::OK);
            let checkpoints_body = to_bytes(checkpoints.into_body(), 16_384).await.unwrap();
            let checkpoints_value = serde_json::from_slice::<Value>(&checkpoints_body).unwrap();
            assert_eq!(
                checkpoints_value["checkpoints"].as_array().unwrap().len(),
                1
            );
            assert_eq!(
                checkpoints_value["checkpoints"][0]["state"]["answer"],
                "worker"
            );

            for suffix in ["checkpoints/0", "replay?target_step_seq=0"] {
                let response = app
                    .clone()
                    .oneshot(
                        Request::builder()
                            .uri(format!("{prefix}/{submitted_run_id}/{suffix}"))
                            .header("x-kogwistar-roles", "reader")
                            .body(Body::empty())
                            .unwrap(),
                    )
                    .await
                    .unwrap();
                assert_eq!(response.status(), StatusCode::OK);
                let body = to_bytes(response.into_body(), 16_384).await.unwrap();
                assert_eq!(
                    serde_json::from_slice::<Value>(&body).unwrap()["state"]["answer"],
                    "worker"
                );
            }
        }
        for (tool_name, arguments, expected_key) in [
            (
                "workflow.run_checkpoint_get",
                json!({"run_id": submitted_run_id, "step_seq": 0}),
                "state",
            ),
            (
                "workflow.run_replay",
                json!({"run_id": submitted_run_id, "target_step_seq": 0}),
                "state",
            ),
            ("workflow.budget_snapshot", json!({}), "cost_ledger"),
            (
                "workflow.operator_dashboard",
                json!({"limit": 10}),
                "resources",
            ),
            (
                "workflow.scheduler_timeline",
                json!({"run_id": submitted_run_id, "limit": 10}),
                "events",
            ),
            ("workflow.dead_letters", json!({"limit": 10}), "runs"),
        ] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/mcp/workflow")
                        .header("content-type", "application/json")
                        .header("x-kogwistar-roles", "reader")
                        .body(Body::from(
                            serde_json::to_vec(&json!({
                                "jsonrpc": "2.0",
                                "id": tool_name,
                                "method": "tools/call",
                                "params": {"name": tool_name, "arguments": arguments},
                            }))
                            .unwrap(),
                        ))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let body = to_bytes(response.into_body(), 65_536).await.unwrap();
            let value = serde_json::from_slice::<Value>(&body).unwrap();
            assert!(
                value["result"].get(expected_key).is_some(),
                "{tool_name} failed: {value}"
            );
        }
        store
            .project_lane_message(dead_lane("dead-mcp-message", "dead-mcp-run"))
            .unwrap();
        let replay = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/workflow")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader,rw")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":"dead-replay","method":"tools/call","params":{"name":"workflow.dead_letter_replay","arguments":{"run_id":"dead-mcp-run"}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let replay =
            serde_json::from_slice::<Value>(&to_bytes(replay.into_body(), 16_384).await.unwrap())
                .unwrap();
        assert_eq!(replay["result"]["replayed"], true);
        assert_eq!(
            store
                .get_projected_lane_message("dead-mcp-message")
                .unwrap()
                .unwrap()
                .status,
            "pending"
        );
        store
            .update_projected_lane_message_status("dead-mcp-message", "completed", None)
            .unwrap();

        let orphan_lane = |message_id: &str, inbox_id: &str| NewProjectedLaneMessage {
            message_id: message_id.to_owned(),
            namespace: "workflow".to_owned(),
            purpose: "system".to_owned(),
            inbox_id: inbox_id.to_owned(),
            conversation_id: "orphan-conversation".to_owned(),
            recipient_id: "python-worker".to_owned(),
            sender_id: "rust-scheduler".to_owned(),
            msg_type: "workflow.run.execute".to_owned(),
            status: "pending".to_owned(),
            created_at: 1,
            available_at: 0,
            run_id: Some(format!("run-{message_id}")),
            step_id: Some("start".to_owned()),
            correlation_id: Some(format!("corr-{message_id}")),
            payload_json: Some("{}".to_owned()),
            error_json: None,
        };
        store
            .project_lane_message(orphan_lane("expired-orphan", "repair-rest"))
            .unwrap();
        store
            .project_lane_message(orphan_lane("active-claim", "repair-active"))
            .unwrap();
        store
            .project_lane_message(orphan_lane("mcp-orphan", "repair-mcp"))
            .unwrap();
        store
            .claim_projected_lane_messages("workflow", "repair-rest", "expired-owner", 1, -1)
            .unwrap();
        store
            .claim_projected_lane_messages("workflow", "repair-active", "active-owner", 1, 60)
            .unwrap();
        store
            .claim_projected_lane_messages("workflow", "repair-mcp", "mcp-owner", 1, -1)
            .unwrap();
        let repaired = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/workflow/messages/repair-orphans?inbox_id=repair-rest&limit=10")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(repaired.status(), StatusCode::OK);
        let repaired =
            serde_json::from_slice::<Value>(&to_bytes(repaired.into_body(), 16_384).await.unwrap())
                .unwrap();
        assert_eq!(repaired["repaired_message_ids"], json!(["expired-orphan"]));
        assert_eq!(
            store
                .get_projected_lane_message("active-claim")
                .unwrap()
                .unwrap()
                .status,
            "claimed"
        );
        let repaired = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/workflow")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":"orphan-repair","method":"tools/call","params":{"name":"workflow.message_orphans_repair","arguments":{"inbox_id":"repair-mcp","limit":10}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let repaired =
            serde_json::from_slice::<Value>(&to_bytes(repaired.into_body(), 16_384).await.unwrap())
                .unwrap();
        assert_eq!(
            repaired["result"]["repaired_message_ids"],
            json!(["mcp-orphan"])
        );
        for message_id in ["expired-orphan", "active-claim", "mcp-orphan"] {
            store
                .update_projected_lane_message_status(message_id, "completed", None)
                .unwrap();
        }

        store
            .replace_named_projection(
                "service_registry",
                "svc-b",
                NamedProjectionWrite {
                    payload: serde_json::Map::from_iter([
                        ("service_id".to_owned(), json!("svc-b")),
                        ("enabled".to_owned(), json!(true)),
                        ("health_status".to_owned(), json!("healthy")),
                    ]),
                    last_authoritative_seq: 1,
                    last_materialized_seq: 1,
                    projection_schema_version: 1,
                    materialization_status: "ready".to_owned(),
                },
            )
            .unwrap();
        store
            .replace_named_projection(
                "service_registry",
                "svc-a",
                NamedProjectionWrite {
                    payload: serde_json::Map::from_iter([
                        ("service_id".to_owned(), json!("svc-a")),
                        ("enabled".to_owned(), json!(false)),
                        ("health_status".to_owned(), json!("stopped")),
                    ]),
                    last_authoritative_seq: 2,
                    last_materialized_seq: 2,
                    projection_schema_version: 1,
                    materialization_status: "ready".to_owned(),
                },
            )
            .unwrap();
        let services = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/workflow/services?limit=1")
                    .header("x-kogwistar-roles", "reader,rw")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(services.status(), StatusCode::OK);
        let services =
            serde_json::from_slice::<Value>(&to_bytes(services.into_body(), 16_384).await.unwrap())
                .unwrap();
        assert_eq!(services["services"][0]["service_id"], "svc-a");
        let service = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/workflow")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader,rw")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":"service-get","method":"tools/call","params":{"name":"workflow.service_get","arguments":{"service_id":"svc-b"}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let service =
            serde_json::from_slice::<Value>(&to_bytes(service.into_body(), 16_384).await.unwrap())
                .unwrap();
        assert_eq!(service["result"]["health_status"], "healthy");
        let missing_service = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/workflow/services/missing")
                    .header("x-kogwistar-roles", "reader,rw")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(missing_service.status(), StatusCode::NOT_FOUND);

        // A non-terminal result must durably schedule exactly one successor.
        let two_step_submit = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/workflow/runs")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"workflow_id":"workflow-two-step","conversation_id":"conversation-two-step","initial_state":{"seed":2}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let two_step_body = to_bytes(two_step_submit.into_body(), 8192).await.unwrap();
        let two_step_run_id = serde_json::from_slice::<Value>(&two_step_body).unwrap()["run_id"]
            .as_str()
            .unwrap()
            .to_owned();
        let first_claim = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/claim")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"claimed_by":"python-worker-3","limit":10,"lease_seconds":1}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let first_claim_body = to_bytes(first_claim.into_body(), 16384).await.unwrap();
        let first_claim_value = serde_json::from_slice::<Value>(&first_claim_body).unwrap();
        let first_work = first_claim_value["work"]
            .as_array()
            .unwrap()
            .iter()
            .find(|work| work["run_id"] == two_step_run_id)
            .unwrap();
        let successor_token_id = format!("{two_step_run_id}:finish");
        let mut first_result_payload = json!({
            "handoff": {
                "message_id": first_work["message_id"],
                "claimed_by": first_work["claimed_by"],
                "run_id": first_work["run_id"],
                "step_id": first_work["step_id"],
                "correlation_id": first_work["correlation_id"],
            },
            "transition": {
                "contract_version": 1,
                "transition_id": format!("first-{two_step_run_id}"),
                "expected_event_seq": first_work["expected_event_seq"],
                "kind": "recorded_step_success",
                "run_id": two_step_run_id,
                "workflow_id": "workflow-two-step",
                "conversation_id": "conversation-two-step",
                "step_seq": 0,
                "node_id": "start",
                "token_id": two_step_run_id,
                "parent_token_id": null,
                "state_update": [["u", {"first": true}]],
                "frontier": {
                    "pending": [["finish", 0, successor_token_id, two_step_run_id]],
                    "suspended": [],
                    "join_node_ids": [],
                    "join_outstanding": [],
                    "join_waiters": {},
                },
            }
        });
        let mut forged_owner_payload = first_result_payload.clone();
        forged_owner_payload["handoff"]["claimed_by"] = json!("forged-worker");
        let forged_owner = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/results")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        serde_json::to_vec(&forged_owner_payload).unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(forged_owner.status(), StatusCode::CONFLICT);
        let first_message_id = first_work["message_id"].as_str().unwrap();
        let lane_after_forgery = store
            .get_projected_lane_message(first_message_id)
            .unwrap()
            .unwrap();
        assert_eq!(lane_after_forgery.status, "claimed");
        assert_eq!(
            lane_after_forgery.claimed_by.as_deref(),
            first_work["claimed_by"].as_str()
        );

        let mut invalid_transition_payload = first_result_payload.clone();
        invalid_transition_payload["transition"]["expected_event_seq"] = json!(999_999);
        let invalid_transition = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/results")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        serde_json::to_vec(&invalid_transition_payload).unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(invalid_transition.status(), StatusCode::CONFLICT);
        let lane_after_invalid = store
            .get_projected_lane_message(first_message_id)
            .unwrap()
            .unwrap();
        assert_eq!(lane_after_invalid.status, "claimed");
        assert_eq!(
            lane_after_invalid.claimed_by.as_deref(),
            first_work["claimed_by"].as_str()
        );
        assert!(
            store
                .read_recorded_runtime_state(
                    &two_step_run_id,
                    "workflow-two-step",
                    "conversation-two-step",
                )
                .unwrap()
                .unwrap()
                .state
                .get("first")
                .is_none(),
            "rejected result must not partially update state"
        );
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        let reclaimed = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/claim")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"claimed_by":"python-worker-3","limit":10,"lease_seconds":60}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let reclaimed_status = reclaimed.status();
        let reclaimed_body = to_bytes(reclaimed.into_body(), 16384).await.unwrap();
        assert_eq!(
            reclaimed_status,
            StatusCode::OK,
            "{}",
            String::from_utf8_lossy(&reclaimed_body)
        );
        let reclaimed_value = serde_json::from_slice::<Value>(&reclaimed_body).unwrap();
        let reclaimed_work = &reclaimed_value["work"][0];
        assert_eq!(reclaimed_work["message_id"], first_work["message_id"]);
        assert!(reclaimed_work["claimed_by"]
            .as_str()
            .unwrap()
            .starts_with("python-worker-3#"));
        assert_ne!(reclaimed_work["claimed_by"], first_work["claimed_by"]);
        let stale_owner = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/results")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        serde_json::to_vec(&first_result_payload).unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(stale_owner.status(), StatusCode::CONFLICT);
        first_result_payload["handoff"]["claimed_by"] = reclaimed_work["claimed_by"].clone();
        first_result_payload["transition"]["expected_event_seq"] =
            reclaimed_work["expected_event_seq"].clone();
        let first_result_bytes = serde_json::to_vec(&first_result_payload).unwrap();
        for body in [first_result_bytes.clone(), first_result_bytes] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/internal/runtime/results")
                        .header("content-type", "application/json")
                        .header("x-kogwistar-roles", "reader")
                        .body(Body::from(body))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
        }
        let second_claim = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/claim")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"claimed_by":"python-worker-4","limit":10,"lease_seconds":60}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let second_claim_body = to_bytes(second_claim.into_body(), 16384).await.unwrap();
        let second_claim_value = serde_json::from_slice::<Value>(&second_claim_body).unwrap();
        let second_work = second_claim_value["work"].as_array().unwrap();
        assert_eq!(
            second_work.len(),
            1,
            "exact retry duplicated successor lane"
        );
        assert_eq!(second_work[0]["run_id"], two_step_run_id);
        assert_eq!(second_work[0]["step_id"], "finish");
        assert_eq!(second_work[0]["payload"]["token_id"], successor_token_id);
        let second_result_payload = json!({
            "handoff": {
                "message_id": second_work[0]["message_id"],
                "claimed_by": second_work[0]["claimed_by"],
                "run_id": second_work[0]["run_id"],
                "step_id": second_work[0]["step_id"],
                "correlation_id": second_work[0]["correlation_id"],
            },
            "transition": {
                "contract_version": 1,
                "transition_id": format!("second-{two_step_run_id}"),
                "expected_event_seq": second_work[0]["expected_event_seq"],
                "kind": "recorded_step_success",
                "run_id": two_step_run_id,
                "workflow_id": "workflow-two-step",
                "conversation_id": "conversation-two-step",
                "step_seq": 1,
                "node_id": "finish",
                "token_id": successor_token_id,
                "parent_token_id": two_step_run_id,
                "state_update": [["u", {"second": true}]],
                "frontier": {
                    "pending": [],
                    "suspended": [],
                    "join_node_ids": [],
                    "join_outstanding": [],
                    "join_waiters": {},
                },
                "result": {"done": true},
            }
        });
        let second_result = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/results")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        serde_json::to_vec(&second_result_payload).unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let second_result_status = second_result.status();
        let second_result_body = to_bytes(second_result.into_body(), 16384).await.unwrap();
        assert_eq!(
            second_result_status,
            StatusCode::OK,
            "{}",
            String::from_utf8_lossy(&second_result_body)
        );
        assert_eq!(
            store
                .get_server_run(&two_step_run_id)
                .unwrap()
                .unwrap()
                .status,
            "succeeded"
        );
        let reopened = SqliteStore::open(&path).unwrap();
        let reopened_state = reopened
            .read_recorded_runtime_state(
                &two_step_run_id,
                "workflow-two-step",
                "conversation-two-step",
            )
            .unwrap()
            .unwrap();
        assert_eq!(reopened_state.state["first"], true);
        assert_eq!(reopened_state.state["second"], true);
        drop(reopened);

        // Restricted worker effects cannot replace the scheduler frontier.
        let effect_submit = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/workflow/runs")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"workflow_id":"workflow-effect","conversation_id":"conversation-effect","join_node_ids":["join"],"start_join_mask":1,"runtime_routes":[{"source_node_id":"start","target_node_id":"left","join_mask":1},{"source_node_id":"start","target_node_id":"right","join_mask":1},{"source_node_id":"left","target_node_id":"join","join_mask":1},{"source_node_id":"right","target_node_id":"join","join_mask":1}]}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let effect_submit_body = to_bytes(effect_submit.into_body(), 8192).await.unwrap();
        let effect_run_id = serde_json::from_slice::<Value>(&effect_submit_body).unwrap()["run_id"]
            .as_str()
            .unwrap()
            .to_owned();
        let effect_claim = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/claim")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"claimed_by":"effect-worker","limit":10,"lease_seconds":60}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let effect_claim_body = to_bytes(effect_claim.into_body(), 16384).await.unwrap();
        let effect_claim_value = serde_json::from_slice::<Value>(&effect_claim_body).unwrap();
        let effect_work = effect_claim_value["work"]
            .as_array()
            .unwrap()
            .iter()
            .find(|work| work["run_id"] == effect_run_id)
            .unwrap();
        let effect_id = format!("effect-{effect_run_id}");
        let effect_payload = json!({
            "handoff": {
                "message_id": effect_work["message_id"],
                "claimed_by": effect_work["claimed_by"],
                "run_id": effect_work["run_id"],
                "step_id": effect_work["step_id"],
                "correlation_id": effect_work["correlation_id"],
            },
            "effect": {
                "contract_version": 1,
                "effect_id": effect_id,
                "state_update": [["u", {"effect_first": true}]],
                "usage": {"input_tokens": 3, "output_tokens": 2, "total_cost": 0.01},
                "trace_events": [{
                    "schema_version": 1,
                    "event": "runtime.python_worker.result_prepared",
                    "executor_target": "python-worker",
                    "worker_id": "effect-worker",
                    "identity_source": "worker_configuration",
                    "lease_claim_id": effect_work["claimed_by"]
                }, {
                    "schema_version": 1,
                    "event": "runtime.python_worker.result_submitted",
                    "executor_target": "python-worker",
                    "worker_id": "effect-worker",
                    "identity_source": "worker_configuration",
                    "lease_claim_id": effect_work["claimed_by"]
                }],
                "successors": [{"node_id": "forged", "join_mask": 0}],
            }
        });
        let mut mismatched_trace = effect_payload.clone();
        mismatched_trace["effect"]["trace_events"][1]["lease_claim_id"] =
            json!("other-worker#other-lease");
        let mismatched_response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/results")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(serde_json::to_vec(&mismatched_trace).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            mismatched_response.status(),
            StatusCode::UNPROCESSABLE_ENTITY
        );
        store
            .append_server_run_event(&effect_run_id, "test.competing", "{}".to_owned())
            .unwrap();
        let effect_bytes = serde_json::to_vec(&effect_payload).unwrap();
        for body in [effect_bytes.clone(), effect_bytes] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/internal/runtime/results")
                        .header("content-type", "application/json")
                        .header("x-kogwistar-roles", "reader")
                        .body(Body::from(body))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
        }
        let effect_event_types = store
            .list_server_run_events(&effect_run_id, 0, usize::MAX)
            .unwrap()
            .into_iter()
            .map(|event| event.event_type)
            .collect::<Vec<_>>();
        assert_eq!(
            effect_event_types
                .iter()
                .filter(|event_type| event_type.as_str() == "workflow.usage.v1")
                .count(),
            1
        );
        assert_eq!(
            effect_event_types
                .iter()
                .filter(|event_type| event_type.as_str() == "workflow.trace.v1")
                .count(),
            2
        );
        let persisted_worker_traces = store
            .list_server_run_events(&effect_run_id, 0, usize::MAX)
            .unwrap()
            .into_iter()
            .filter(|event| event.event_type == "workflow.trace.v1")
            .map(|event| serde_json::from_str::<Value>(&event.payload_json).unwrap())
            .collect::<Vec<_>>();
        let persisted_worker_trace = persisted_worker_traces
            .iter()
            .find(|event| event["trace"]["event"] == "runtime.python_worker.result_prepared")
            .unwrap();
        assert_eq!(
            persisted_worker_trace["trace"]["worker_id"],
            "effect-worker"
        );
        assert_eq!(
            persisted_worker_trace["trace"]["lease_claim_id"],
            effect_work["claimed_by"]
        );
        let persisted_submission_trace = persisted_worker_traces
            .iter()
            .find(|event| event["trace"]["event"] == "runtime.python_worker.result_submitted")
            .unwrap();
        assert_eq!(
            persisted_submission_trace["trace"]["lease_claim_id"],
            effect_work["claimed_by"]
        );
        let effect_state = store
            .read_recorded_runtime_state(&effect_run_id, "workflow-effect", "conversation-effect")
            .unwrap()
            .unwrap();
        assert_eq!(effect_state.frontier.pending.len(), 2);
        let left = effect_state
            .frontier
            .pending
            .iter()
            .find(|token| token.0 == "left")
            .unwrap();
        let right = effect_state
            .frontier
            .pending
            .iter()
            .find(|token| token.0 == "right")
            .unwrap();
        assert_eq!(left.2, effect_run_id);
        assert_eq!(right.3.as_deref(), Some(effect_run_id.as_str()));
        assert_ne!(right.2, effect_run_id);
        let pending_effect_lanes = store
            .list_projected_lane_messages(kogwistar_store::LaneMessageFilter {
                namespace: Some("workflow".to_owned()),
                inbox_id: Some("workflow-runtime".to_owned()),
                status: Some("pending".to_owned()),
                correlation_id: Some(effect_run_id.clone()),
                ..kogwistar_store::LaneMessageFilter::default()
            })
            .unwrap();
        assert_eq!(
            pending_effect_lanes.len(),
            2,
            "effect retry duplicated dispatch"
        );
        assert_eq!(pending_effect_lanes[0].step_id.as_deref(), Some("left"));
        assert_eq!(pending_effect_lanes[1].step_id.as_deref(), Some("right"));

        let parallel_claim = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/claim")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"claimed_by":"effect-parallel-worker","limit":10,"lease_seconds":60}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let parallel_claim_body = to_bytes(parallel_claim.into_body(), 16384).await.unwrap();
        let parallel_claim_value = serde_json::from_slice::<Value>(&parallel_claim_body).unwrap();
        let parallel_work = parallel_claim_value["work"].as_array().unwrap();
        assert_eq!(parallel_work.len(), 2);
        assert_eq!(parallel_work[0]["step_id"], "left");
        assert_eq!(parallel_work[1]["step_id"], "right");

        let submit_effect = |app: Router, work: Value, successors: Value, final_result: Value| {
            let effect_run_id = effect_run_id.clone();
            async move {
                let node = work["step_id"].as_str().unwrap().to_owned();
                let mut effect = json!({
                    "contract_version": 1,
                    "effect_id": format!("effect-{node}-{effect_run_id}"),
                    "state_update": [["u", {(format!("effect_{node}")): true}]],
                    "successors": successors,
                });
                if !final_result.is_null() {
                    effect["result"] = final_result;
                }
                app.clone()
                    .oneshot(
                        Request::builder()
                            .method("POST")
                            .uri("/internal/runtime/results")
                            .header("content-type", "application/json")
                            .header("x-kogwistar-roles", "reader")
                            .body(Body::from(
                                serde_json::to_vec(&json!({
                                    "handoff": {
                                        "message_id": work["message_id"],
                                        "claimed_by": work["claimed_by"],
                                        "run_id": work["run_id"],
                                        "step_id": work["step_id"],
                                        "correlation_id": work["correlation_id"],
                                    },
                                    "effect": effect,
                                }))
                                .unwrap(),
                            ))
                            .unwrap(),
                    )
                    .await
                    .unwrap()
            }
        };
        let to_join = json!([{"node_id": "forged", "join_mask": 0}]);
        let early_right = submit_effect(
            app.clone(),
            parallel_work[1].clone(),
            to_join.clone(),
            Value::Null,
        )
        .await;
        assert_eq!(early_right.status(), StatusCode::CONFLICT);
        let left_result = submit_effect(
            app.clone(),
            parallel_work[0].clone(),
            to_join.clone(),
            Value::Null,
        )
        .await;
        let result_status = left_result.status();
        let result_body = to_bytes(left_result.into_body(), 16384).await.unwrap();
        assert_eq!(
            result_status,
            StatusCode::OK,
            "{}",
            String::from_utf8_lossy(&result_body)
        );
        let restarted_app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: vec!["reader".to_owned()],
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig::default(),
            },
            Arc::new(SqliteRunApplicationService::open(&path).unwrap()),
        );
        let right_result = submit_effect(
            restarted_app.clone(),
            parallel_work[1].clone(),
            to_join,
            Value::Null,
        )
        .await;
        let result_status = right_result.status();
        let result_body = to_bytes(right_result.into_body(), 16384).await.unwrap();
        assert_eq!(
            result_status,
            StatusCode::OK,
            "{}",
            String::from_utf8_lossy(&result_body)
        );
        let join_claim = restarted_app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/claim")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"claimed_by":"effect-join-worker","limit":10,"lease_seconds":60}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let join_claim_body = to_bytes(join_claim.into_body(), 16384).await.unwrap();
        let join_claim_value = serde_json::from_slice::<Value>(&join_claim_body).unwrap();
        let join_work = join_claim_value["work"].as_array().unwrap();
        assert_eq!(join_work.len(), 1);
        assert_eq!(join_work[0]["step_id"], "join");
        let join_result = submit_effect(
            restarted_app,
            join_work[0].clone(),
            json!([]),
            json!({"effect_done": true}),
        )
        .await;
        let join_status = join_result.status();
        let join_body = to_bytes(join_result.into_body(), 16384).await.unwrap();
        assert_eq!(
            join_status,
            StatusCode::OK,
            "{}",
            String::from_utf8_lossy(&join_body)
        );
        let final_effect_state = store
            .read_recorded_runtime_state(&effect_run_id, "workflow-effect", "conversation-effect")
            .unwrap()
            .unwrap();
        assert!(final_effect_state.frontier.pending.is_empty());
        assert!(final_effect_state.status.is_terminal());
        assert_eq!(final_effect_state.state["effect_first"], true);
        assert_eq!(final_effect_state.state["effect_left"], true);
        assert_eq!(final_effect_state.state["effect_right"], true);
        assert_eq!(final_effect_state.state["effect_join"], true);
        assert!(final_effect_state.frontier.join_outstanding.is_empty());
        assert!(final_effect_state.frontier.join_waiters.is_empty());
        assert_eq!(
            store
                .get_server_run(&effect_run_id)
                .unwrap()
                .unwrap()
                .status,
            "succeeded"
        );

        let cancelled_submit = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/workflow/runs")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"workflow_id":"workflow-cancel","conversation_id":"conversation-cancel"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let cancelled_submit_body = to_bytes(cancelled_submit.into_body(), 8192).await.unwrap();
        let cancelled_run_id =
            serde_json::from_slice::<Value>(&cancelled_submit_body).unwrap()["run_id"]
                .as_str()
                .unwrap()
                .to_owned();
        let cancel_before_claim = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/workflow/runs/{cancelled_run_id}/cancel"))
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(cancel_before_claim.status(), StatusCode::ACCEPTED);
        let cancel_body = to_bytes(cancel_before_claim.into_body(), 8192)
            .await
            .unwrap();
        let cancelled_run = serde_json::from_slice::<Value>(&cancel_body).unwrap();
        assert_eq!(cancelled_run["status"], "cancelled");
        assert_eq!(cancelled_run["terminal"], true);
        let cancelled_claim = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/claim")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"claimed_by":"must-not-run","limit":10,"lease_seconds":60}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let cancelled_claim_body = to_bytes(cancelled_claim.into_body(), 8192).await.unwrap();
        assert!(
            serde_json::from_slice::<Value>(&cancelled_claim_body).unwrap()["work"]
                .as_array()
                .unwrap()
                .is_empty()
        );
        let cancelled_lanes = store
            .list_projected_lane_messages(kogwistar_store::LaneMessageFilter {
                correlation_id: Some(cancelled_run_id.clone()),
                ..kogwistar_store::LaneMessageFilter::default()
            })
            .unwrap();
        assert_eq!(cancelled_lanes.len(), 1);
        assert_eq!(cancelled_lanes[0].status, "cancelled");
        let cancelled_state = store
            .read_recorded_runtime_state(
                &cancelled_run_id,
                "workflow-cancel",
                "conversation-cancel",
            )
            .unwrap()
            .unwrap();
        assert_eq!(
            cancelled_state.status,
            kogwistar_runtime::RecordedRunStatus::Cancelled
        );
        assert!(cancelled_state.frontier.pending.is_empty());

        let claimed_cancel_submit = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/workflow/runs")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"workflow_id":"workflow-claimed-cancel","conversation_id":"conversation-claimed-cancel"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let claimed_cancel_submit_body = to_bytes(claimed_cancel_submit.into_body(), 8192)
            .await
            .unwrap();
        let claimed_cancel_run_id = serde_json::from_slice::<Value>(&claimed_cancel_submit_body)
            .unwrap()["run_id"]
            .as_str()
            .unwrap()
            .to_owned();
        let claimed_cancel_claim = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/claim")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"claimed_by":"cancelled-worker","limit":1,"lease_seconds":60}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let claimed_cancel_claim_body = to_bytes(claimed_cancel_claim.into_body(), 8192)
            .await
            .unwrap();
        let claimed_cancel_work =
            serde_json::from_slice::<Value>(&claimed_cancel_claim_body).unwrap()["work"][0].clone();
        let claimed_cancel = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/workflow/runs/{claimed_cancel_run_id}/cancel"))
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(claimed_cancel.status(), StatusCode::ACCEPTED);
        let stale_result = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/results")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        serde_json::to_vec(&json!({
                            "handoff": {
                                "message_id": claimed_cancel_work["message_id"],
                                "claimed_by": claimed_cancel_work["claimed_by"],
                                "run_id": claimed_cancel_work["run_id"],
                                "step_id": claimed_cancel_work["step_id"],
                                "correlation_id": claimed_cancel_work["correlation_id"],
                            },
                            "effect": {
                                "contract_version": 1,
                                "effect_id": format!("stale-{claimed_cancel_run_id}"),
                                "state_update": [["u", {"must_not_apply": true}]],
                                "successors": [],
                                "result": {"must_not_apply": true},
                            }
                        }))
                        .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(stale_result.status(), StatusCode::CONFLICT);
        let claimed_cancel_state = store
            .read_recorded_runtime_state(
                &claimed_cancel_run_id,
                "workflow-claimed-cancel",
                "conversation-claimed-cancel",
            )
            .unwrap()
            .unwrap();
        assert_eq!(
            claimed_cancel_state.status,
            kogwistar_runtime::RecordedRunStatus::Cancelled
        );
        assert!(claimed_cancel_state.state.get("must_not_apply").is_none());

        drop(app);
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: vec!["reader".to_owned()],
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig::default(),
            },
            Arc::new(SqliteRunApplicationService::open(&path).unwrap()),
        );
        let get = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/workflow/runs/run-1")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(get.status(), StatusCode::OK);
        let get_body = to_bytes(get.into_body(), 8192).await.unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&get_body).unwrap()["status"],
            "running"
        );

        let events = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/runs/run-1/events/poll?after_seq=0&limit=10")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(events.status(), StatusCode::OK);
        let event_body = to_bytes(events.into_body(), 8192).await.unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&event_body).unwrap()["events"][0]["event_type"],
            "run.started"
        );

        let cancel = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/workflow/runs/run-1/cancel")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(cancel.status(), StatusCode::ACCEPTED);
        let cancel_body = to_bytes(cancel.into_body(), 8192).await.unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&cancel_body).unwrap()["status"],
            "cancelling"
        );

        let sse = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/workflow/runs/run-1/events")
                    .header("x-kogwistar-roles", "reader")
                    .header("last-event-id", "0")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(sse.status(), StatusCode::OK);
        assert_eq!(
            sse.headers()["content-type"],
            "text/event-stream; charset=utf-8"
        );
        let sse_body =
            String::from_utf8(to_bytes(sse.into_body(), 8192).await.unwrap().to_vec()).unwrap();
        assert!(sse_body.contains("event: run.started"));
        assert!(sse_body.contains("id: 1"));

        let mcp_status = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/workflow")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"workflow.run_status","arguments":{"run_id":"run-1"}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(mcp_status.status(), StatusCode::OK);
        let mcp_body = to_bytes(mcp_status.into_body(), 8192).await.unwrap();
        let mcp_value = serde_json::from_slice::<Value>(&mcp_body).unwrap();
        assert_eq!(mcp_value["id"], 7);
        assert_eq!(mcp_value["result"]["run_id"], "run-1");

        let mcp_events = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/workflow")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":8,"method":"tools/call","params":{"name":"workflow.run_events","arguments":{"run_id":"run-1","after_seq":0,"limit":10}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let mcp_event_body = to_bytes(mcp_events.into_body(), 8192).await.unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&mcp_event_body).unwrap()["result"]["events"][0]["event_type"],
            "run.started"
        );
        drop(app);
        drop(store);
        remove_test_sqlite(&path);
    }
