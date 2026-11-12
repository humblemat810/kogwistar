    #[tokio::test]
    async fn sqlite_native_noop_result_is_durable_through_runtime_route() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use std::time::{SystemTime, UNIX_EPOCH};
        use tower::ServiceExt;

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-native-noop-{nonce}.sqlite3"));
        let store = SqliteStore::open(&path).unwrap();
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
                    .header("x-kogwistar-roles", "reader,rw")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "run_id": "native-noop-run",
                            "workflow_id": "native-noop-workflow",
                            "conversation_id": "native-noop-conversation",
                            "initial_state": {"seed": 1},
                            "runtime_kind": "sync",
                            "start_node_id": "start",
                            "node_ops": {
                                "start": "rust-native/noop",
                                "finish": "rust-native/emit-result"
                            },
                            "runtime_routes": [{
                                "source_node_id": "start",
                                "target_node_id": "finish",
                                "join_mask": 0
                            }]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(submit.status(), StatusCode::ACCEPTED);

        let claim = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/claim")
                    .header("x-kogwistar-roles", "reader")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "claimed_by": "rust-native-worker",
                            "limit": 1,
                            "lease_seconds": 60,
                            "recipient_ids": ["rust-native-worker"]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(claim.status(), StatusCode::OK);
        let claim_body: Value = serde_json::from_slice(
            &to_bytes(claim.into_body(), 1_048_576).await.unwrap(),
        )
        .unwrap();
        let work = claim_body["work"][0].clone();
        assert_eq!(work["payload"]["op"], "rust-native/noop");

        let execute = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/native/execute")
                    .header("x-kogwistar-roles", "reader")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "handoff": {
                                "message_id": work["message_id"],
                                "claimed_by": work["claimed_by"],
                                "run_id": work["run_id"],
                                "step_id": work["step_id"],
                                "correlation_id": work["correlation_id"]
                            },
                            "payload": work["payload"]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(execute.status(), StatusCode::OK);

        let run = store.get_server_run("native-noop-run").unwrap().unwrap();
        assert_eq!(run.status, "running");
        let events = store
            .list_server_run_events("native-noop-run", 0, 100)
            .unwrap();
        let transition = events
            .iter()
            .find(|event| {
                if event.event_type != "workflow.recorded_transition.v1" {
                    return false;
                }
                serde_json::from_str::<Value>(&event.payload_json)
                    .map(|payload| {
                        payload["reduced"]["state"]["last_node_id"] == "start"
                            && payload["reduced"]["state"]["last_step_seq"] == 0
                    })
                    .unwrap_or(false)
            })
            .expect("native noop transition must be durable");
        let transition_payload: Value = serde_json::from_str(&transition.payload_json).unwrap();
        assert_eq!(
            transition_payload["reduced"]["result"]["workflow_status"],
            "succeeded",
            "unexpected persisted noop transition: {transition_payload}"
        );
        assert_eq!(
            transition_payload["reduced"]["result"]["final_state"]["seed"],
            1
        );
        assert!(events.iter().any(|event| {
            event.event_type == "workflow.trace.v1"
                && event.payload_json.contains("runtime.native.step.completed")
        }));

        drop(app);
        drop(store);
        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn sqlite_native_runtime_route_uses_shared_result_reducer() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-native-runtime-{nonce}.sqlite3"));
        let store = SqliteStore::open(&path).unwrap();
        let mut native_operations = kogwistar_runtime::NativeOperationRegistry::default();
        native_operations
            .register("app/custom-result", "v1", |request| {
                Ok(kogwistar_runtime::NativeOperationOutput {
                    state_update: Vec::new(),
                    result: Some(json!({
                        "custom_result": true,
                        "counter": request.state.get("counter"),
                    })),
                })
            })
            .unwrap();
        native_operations
            .register("app/custom-result", "v2", |request| {
                Ok(kogwistar_runtime::NativeOperationOutput {
                    state_update: Vec::new(),
                    result: Some(json!({
                        "custom_result": true,
                        "operation_version": request.op_version,
                        "counter": request.state.get("counter"),
                    })),
                })
            })
            .unwrap();
        let service = SqliteRunApplicationService::open(&path)
            .unwrap()
            .with_native_operations(native_operations);
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: vec!["reader".to_owned()],
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig::default(),
            },
            Arc::new(service.clone()),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let client = reqwest::Client::new();
        let base_url = format!("http://{address}");

        for (run_id, operation, version, expected_code) in [
            (
                "unknown-native-run",
                "rust-native/app/missing",
                "v1",
                "KOGWISTAR_NATIVE_OPERATION_UNREGISTERED",
            ),
            (
                "unsupported-native-version-run",
                "rust-native/increment-counter",
                "v9",
                "KOGWISTAR_NATIVE_OPERATION_UNREGISTERED",
            ),
            (
                "malformed-native-run",
                "rust-native/",
                "v1",
                "KOGWISTAR_INVALID_NATIVE_OPERATION",
            ),
        ] {
            let response = client
                .post(format!("{base_url}/api/workflow/runs"))
                .header("x-kogwistar-roles", "reader,rw")
                .json(&json!({
                    "run_id": run_id,
                    "workflow_id": "native-admission-workflow",
                    "conversation_id": "native-admission-conversation",
                    "runtime_kind": "sync",
                    "start_node_id": "entry",
                    "node_ops": {"entry": operation},
                    "node_versions": {"entry": version}
                }))
                .send()
                .await
                .unwrap();
            let status = response.status();
            let response_body = response.text().await.unwrap();
            assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{response_body}");
            let body: Value = serde_json::from_str(&response_body).unwrap();
            assert_eq!(body["code"], expected_code);
            assert!(store.get_server_run(run_id).unwrap().is_none());
            assert!(
                store
                    .list_projected_lane_messages(LaneMessageFilter {
                        namespace: Some("workflow".to_owned()),
                        inbox_id: Some("workflow-runtime".to_owned()),
                        correlation_id: Some(run_id.to_owned()),
                        limit: 10,
                        ..LaneMessageFilter::default()
                    })
                    .unwrap()
                    .is_empty()
            );
        }

        let submit = client
            .post(format!("{base_url}/api/workflow/runs"))
            .header("x-kogwistar-roles", "reader,rw")
            .header("content-type", "application/json")
            .body(
                r#"{"run_id":"native-run","workflow_id":"native-workflow","conversation_id":"native-conversation","initial_state":{"counter":1},"runtime_kind":"sync","start_node_id":"entry","node_ops":{"entry":"rust-native/increment-counter","finish":"rust-native/emit-result","custom":"rust-native/app/custom-result"},"node_versions":{"custom":"v2"},"runtime_routes":[{"source_node_id":"entry","target_node_id":"finish","join_mask":0},{"source_node_id":"finish","target_node_id":"custom","join_mask":0}]}"#,
            )
            .send()
            .await
            .unwrap();
        assert_eq!(submit.status(), StatusCode::ACCEPTED);

        let claim = client
            .post(format!("{base_url}/internal/runtime/claim"))
            .header("x-kogwistar-roles", "reader")
            .header("content-type", "application/json")
            .body(
                r#"{"claimed_by":"rust-native-worker","limit":1,"lease_seconds":60,"recipient_ids":["rust-native-worker"]}"#,
            )
            .send()
            .await
            .unwrap();
        assert_eq!(claim.status(), StatusCode::OK);
        let work = claim.json::<Value>().await.unwrap()["work"][0].clone();
        assert_eq!(work["payload"]["op"], "rust-native/increment-counter");
        assert_eq!(
            work["payload"]["executor_target"],
            kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET
        );
        assert!(work["claimed_by"]
            .as_str()
            .unwrap()
            .starts_with("rust-native-worker#"));

        let mut malformed_version_payload = work["payload"].clone();
        malformed_version_payload["op_version"] = json!(2);
        let malformed_version = client
            .post(format!("{base_url}/internal/runtime/native/execute"))
            .header("x-kogwistar-roles", "reader")
            .json(&json!({
                "handoff": {
                    "message_id": work["message_id"],
                    "claimed_by": work["claimed_by"],
                    "run_id": work["run_id"],
                    "step_id": work["step_id"],
                    "correlation_id": work["correlation_id"]
                },
                "payload": malformed_version_payload
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(malformed_version.status(), StatusCode::CONFLICT);

        let execute_payload = json!({
            "handoff": {
                "message_id": work["message_id"],
                "claimed_by": work["claimed_by"],
                "run_id": work["run_id"],
                "step_id": work["step_id"],
                "correlation_id": work["correlation_id"]
            },
            "payload": work["payload"]
        });
        let fault_connection = rusqlite::Connection::open(&path).unwrap();
        fault_connection
            .execute_batch(
                "CREATE TRIGGER reject_native_transition BEFORE INSERT ON server_run_events \
                 WHEN NEW.run_id = 'native-run' \
                 AND NEW.event_type = 'workflow.recorded_transition.v1' \
                 BEGIN SELECT RAISE(ABORT, 'injected native transition failure'); END;",
            )
            .unwrap();
        let event_count_before_failed_commit = store
            .list_server_run_events("native-run", 0, 100)
            .unwrap()
            .len();
        let failed_execute = client
            .post(format!("{base_url}/internal/runtime/native/execute"))
            .header("x-kogwistar-roles", "reader")
            .json(&execute_payload)
            .send()
            .await
            .unwrap();
        assert_eq!(failed_execute.status(), StatusCode::CONFLICT);
        let failed_body: Value = failed_execute.json().await.unwrap();
        assert_eq!(failed_body["code"], "KOGWISTAR_RUNTIME_RESULT_CONFLICT");
        assert_eq!(
            store
                .list_server_run_events("native-run", 0, 100)
                .unwrap()
                .len(),
            event_count_before_failed_commit,
            "failed transition must leave canonical run history unchanged"
        );
        assert_eq!(
            store
                .get_projected_lane_message(work["message_id"].as_str().unwrap())
                .unwrap()
                .unwrap()
                .status,
            "claimed",
            "failed result commit must not acknowledge the native work item"
        );
        fault_connection
            .execute_batch("DROP TRIGGER reject_native_transition;")
            .unwrap();

        let execute = client
            .post(format!("{base_url}/internal/runtime/native/execute"))
            .header("x-kogwistar-roles", "reader")
            .json(&execute_payload)
            .send()
            .await
            .unwrap();
        let execute_status = execute.status();
        let execute_body = execute.text().await.unwrap();
        assert_eq!(execute_status, StatusCode::OK, "{}", execute_body);
        assert_eq!(
            store.get_server_run("native-run").unwrap().unwrap().status,
            "running"
        );
        let event_count_after_first_execute = store
            .list_server_run_events("native-run", 0, 100)
            .unwrap()
            .len();
        let retried_execute = client
            .post(format!("{base_url}/internal/runtime/native/execute"))
            .header("x-kogwistar-roles", "reader")
            .json(&execute_payload)
            .send()
            .await
            .unwrap();
        assert_eq!(retried_execute.status(), StatusCode::OK);
        assert_eq!(
            store
                .list_server_run_events("native-run", 0, 100)
                .unwrap()
                .len(),
            event_count_after_first_execute,
            "retry must not duplicate durable runtime events"
        );
        let continuation_claim = client
            .post(format!("{base_url}/internal/runtime/claim"))
            .header("x-kogwistar-roles", "reader")
            .header("content-type", "application/json")
            .body(
                r#"{"claimed_by":"rust-native-worker","limit":1,"lease_seconds":60,"recipient_ids":["rust-native-worker"]}"#,
            )
            .send()
            .await
            .unwrap();
        assert_eq!(continuation_claim.status(), StatusCode::OK);
        let continuation = continuation_claim.json::<Value>().await.unwrap()["work"][0].clone();
        assert_eq!(continuation["payload"]["op"], "rust-native/emit-result");
        assert_eq!(
            continuation["payload"]["executor_target"],
            kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET
        );
        let continuation_execute = client
            .post(format!("{base_url}/internal/runtime/native/execute"))
            .header("x-kogwistar-roles", "reader")
            .json(&json!({
                "handoff": {
                    "message_id": continuation["message_id"],
                    "claimed_by": continuation["claimed_by"],
                    "run_id": continuation["run_id"],
                    "step_id": continuation["step_id"],
                    "correlation_id": continuation["correlation_id"]
                },
                "payload": continuation["payload"]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(continuation_execute.status(), StatusCode::OK);
        assert_eq!(
            store.get_server_run("native-run").unwrap().unwrap().status,
            "running"
        );
        let events = store.list_server_run_events("native-run", 0, 100).unwrap();
        let native_trace = events
            .iter()
            .find(|event| {
                event.event_type == "workflow.trace.v1"
                    && event.payload_json.contains("runtime.native.step.completed")
            })
            .expect("native trace event must survive the HTTP/API durable lane");
        let native_trace_payload: Value = serde_json::from_str(&native_trace.payload_json).unwrap();
        assert!(native_seam_attestation_is_valid(
            &native_trace_payload["trace"]
        ));
        assert_eq!(
            native_trace_payload["trace"]["implementation_authority"],
            kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET
        );
        assert_eq!(
            native_trace_payload["trace"]["seam_attestation_version"],
            kogwistar_runtime::RUST_NATIVE_SEAM_ATTESTATION_VERSION
        );
        assert_eq!(native_trace_payload["trace"]["seam_crossings"], json!([]));
        assert!(events.iter().any(|event| {
            event.event_type == "workflow.recorded_transition.v1"
                && event.payload_json.contains("increment-counter")
                && event.payload_json.contains("counter")
                && event.payload_json.contains(":2")
        }));
        assert!(events.iter().any(|event| {
            event.event_type == "workflow.recorded_transition.v1"
                && event.payload_json.contains("emit-result")
                && event.payload_json.contains("rust-native")
        }));

        let custom_claim = client
            .post(format!("{base_url}/internal/runtime/claim"))
            .header("x-kogwistar-roles", "reader")
            .json(&json!({
                "claimed_by": "rust-native-worker",
                "limit": 1,
                "lease_seconds": 60,
                "recipient_ids": ["rust-native-worker"]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(custom_claim.status(), StatusCode::OK);
        let custom_work = custom_claim.json::<Value>().await.unwrap()["work"][0].clone();
        assert_eq!(
            custom_work["payload"]["op"],
            "rust-native/app/custom-result"
        );
        let custom_execute = client
            .post(format!("{base_url}/internal/runtime/native/execute"))
            .header("x-kogwistar-roles", "reader")
            .json(&json!({
                "handoff": {
                    "message_id": custom_work["message_id"],
                    "claimed_by": custom_work["claimed_by"],
                    "run_id": custom_work["run_id"],
                    "step_id": custom_work["step_id"],
                    "correlation_id": custom_work["correlation_id"]
                },
                "payload": custom_work["payload"]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(custom_execute.status(), StatusCode::OK);
        assert_eq!(
            store.get_server_run("native-run").unwrap().unwrap().status,
            "succeeded"
        );
        let events = store.list_server_run_events("native-run", 0, 100).unwrap();
        assert!(events.iter().any(|event| {
            event.event_type == "workflow.recorded_transition.v1"
                && event.payload_json.contains("app/custom-result")
                && event.payload_json.contains("custom_result")
                && event.payload_json.contains("v2")
        }));
        assert!(events.iter().any(|event| {
            event.event_type == "workflow.trace.v1"
                && event.payload_json.contains("app/custom-result")
                && event.payload_json.contains("seam_crossings")
        }));

        // One durable run can hand off from the Python worker protocol to
        // Rust-native execution and back without changing its state contract.
        let mixed_submit = client
            .post(format!("{base_url}/api/workflow/runs"))
            .header("x-kogwistar-roles", "reader,rw")
            .json(&json!({
                "run_id": "mixed-language-run",
                "workflow_id": "mixed-language-workflow",
                "conversation_id": "mixed-language-conversation",
                "initial_state": {"counter": 4},
                "runtime_kind": "sync",
                "start_node_id": "python-before",
                "node_ops": {
                    "python-before": "application.before",
                    "native-middle": "rust-native/increment-counter",
                    "python-after": "application.after"
                },
                "runtime_routes": [
                    {"source_node_id": "python-before", "target_node_id": "native-middle", "join_mask": 0},
                    {"source_node_id": "native-middle", "target_node_id": "python-after", "join_mask": 0}
                ]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(mixed_submit.status(), StatusCode::ACCEPTED);

        let python_before_claim = client
            .post(format!("{base_url}/internal/runtime/claim"))
            .header("x-kogwistar-roles", "reader")
            .json(&json!({
                "claimed_by": "python-worker",
                "limit": 1,
                "lease_seconds": 60,
                "recipient_ids": ["python-worker"]
            }))
            .send()
            .await
            .unwrap();
        let python_before_work =
            python_before_claim.json::<Value>().await.unwrap()["work"][0].clone();
        assert_eq!(python_before_work["step_id"], "python-before");
        assert!(python_before_work["payload"]["executor_target"].is_null());
        let events_before_wrong_executor = store
            .list_server_run_events("mixed-language-run", 0, 100)
            .unwrap();
        let wrong_executor_result = client
            .post(format!("{base_url}/internal/runtime/native/execute"))
            .header("x-kogwistar-roles", "reader")
            .json(&json!({
                "handoff": {
                    "message_id": python_before_work["message_id"],
                    "claimed_by": python_before_work["claimed_by"],
                    "run_id": python_before_work["run_id"],
                    "step_id": python_before_work["step_id"],
                    "correlation_id": python_before_work["correlation_id"]
                },
                "payload": python_before_work["payload"]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(wrong_executor_result.status(), StatusCode::CONFLICT);
        assert_eq!(
            store
                .list_server_run_events("mixed-language-run", 0, 100)
                .unwrap(),
            events_before_wrong_executor,
            "Python-target work rejected by the native route must remain unconsumed"
        );
        let python_before_effect = json!({
            "contract_version": 1,
            "effect_id": "mixed-python-before",
            "state_update": [["u", {"python_before_seen": true}]],
            "successors": [],
            "result": null
        });
        let events_before_invalid_handoff = store
            .list_server_run_events("mixed-language-run", 0, 100)
            .unwrap();
        for (body, expected_status) in [
            (b"{".to_vec(), StatusCode::UNPROCESSABLE_ENTITY),
            (
                vec![b' '; MAX_RUNTIME_WORKER_RESULT_BYTES + 1],
                StatusCode::PAYLOAD_TOO_LARGE,
            ),
        ] {
            let rejected = service
                .execute(ApiEffectRequest {
                    contract_version: 1,
                    method: "POST".to_owned(),
                    path_and_query: "/internal/runtime/results".to_owned(),
                    body,
                    principal: json!({"roles": ["reader"]}),
                })
                .await;
            assert_eq!(rejected.status, expected_status.as_u16());
            assert_eq!(
                store
                    .list_server_run_events("mixed-language-run", 0, 100)
                    .unwrap(),
                events_before_invalid_handoff,
                "malformed or oversized result must not append durable events"
            );
            assert_eq!(
                store
                    .get_projected_lane_message(
                        python_before_work["message_id"].as_str().unwrap()
                    )
                    .unwrap()
                    .unwrap()
                    .status,
                "claimed",
                "malformed or oversized result must preserve the active claim"
            );
        }
        for field in ["message_id", "claimed_by", "run_id", "step_id", "correlation_id"] {
            let mut invalid_handoff = json!({
                "handoff": {
                    "message_id": python_before_work["message_id"],
                    "claimed_by": python_before_work["claimed_by"],
                    "run_id": python_before_work["run_id"],
                    "step_id": python_before_work["step_id"],
                    "correlation_id": python_before_work["correlation_id"]
                },
                "effect": python_before_effect
            });
            invalid_handoff["handoff"][field] = json!(format!("forged-{field}"));
            let rejected = client
                .post(format!("{base_url}/internal/runtime/results"))
                .header("x-kogwistar-roles", "reader")
                .json(&invalid_handoff)
                .send()
                .await
                .unwrap();
            assert_eq!(
                rejected.status(),
                StatusCode::CONFLICT,
                "mismatched handoff field {field} must not be accepted"
            );
            assert_eq!(
                store
                    .list_server_run_events("mixed-language-run", 0, 100)
                    .unwrap(),
                events_before_invalid_handoff,
                "mismatched handoff field {field} must not append durable events"
            );
            assert_eq!(
                store
                    .get_projected_lane_message(
                        python_before_work["message_id"].as_str().unwrap()
                    )
                    .unwrap()
                    .unwrap()
                    .status,
                "claimed",
                "mismatched handoff field {field} must not acknowledge the active claim"
            );
        }
        let python_before_result = client
            .post(format!("{base_url}/internal/runtime/results"))
            .header("x-kogwistar-roles", "reader")
            .json(&json!({
                "handoff": {
                    "message_id": python_before_work["message_id"],
                    "claimed_by": python_before_work["claimed_by"],
                    "run_id": python_before_work["run_id"],
                    "step_id": python_before_work["step_id"],
                    "correlation_id": python_before_work["correlation_id"]
                },
                "effect": python_before_effect
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(python_before_result.status(), StatusCode::OK);

        let native_middle_claim = client
            .post(format!("{base_url}/internal/runtime/claim"))
            .header("x-kogwistar-roles", "reader")
            .json(&json!({
                "claimed_by": "rust-native-worker",
                "limit": 1,
                "lease_seconds": 60,
                "recipient_ids": ["rust-native-worker"]
            }))
            .send()
            .await
            .unwrap();
        let native_middle_work =
            native_middle_claim.json::<Value>().await.unwrap()["work"][0].clone();
        assert_eq!(native_middle_work["payload"]["node_id"], "native-middle");
        assert_eq!(
            native_middle_work["payload"]["executor_target"],
            "rust-native"
        );
        assert_eq!(native_middle_work["payload"]["state"]["counter"], 4);
        assert_eq!(
            native_middle_work["payload"]["state"]["python_before_seen"],
            true
        );
        let events_before_python_result_bypass = store
            .list_server_run_events("mixed-language-run", 0, 100)
            .unwrap();
        let python_result_bypass = client
            .post(format!("{base_url}/internal/runtime/results"))
            .header("x-kogwistar-roles", "reader")
            .json(&json!({
                "handoff": {
                    "message_id": native_middle_work["message_id"],
                    "claimed_by": native_middle_work["claimed_by"],
                    "run_id": native_middle_work["run_id"],
                    "step_id": native_middle_work["step_id"],
                    "correlation_id": native_middle_work["correlation_id"]
                },
                "effect": {
                    "contract_version": 1,
                    "effect_id": "forged-python-result-for-native-step",
                    "state_update": [["u", {"counter": 999}]],
                    "successors": []
                }
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(python_result_bypass.status(), StatusCode::CONFLICT);
        assert_eq!(
            store
                .list_server_run_events("mixed-language-run", 0, 100)
                .unwrap(),
            events_before_python_result_bypass,
            "generic Python results must not consume Rust-targeted work"
        );
        let events_before_wrong_target = store
            .list_server_run_events("mixed-language-run", 0, 100)
            .unwrap();
        let mut wrong_target_payload = native_middle_work["payload"].clone();
        wrong_target_payload["executor_target"] = json!("python-worker");
        let wrong_target_result = client
            .post(format!("{base_url}/internal/runtime/native/execute"))
            .header("x-kogwistar-roles", "reader")
            .json(&json!({
                "handoff": {
                    "message_id": native_middle_work["message_id"],
                    "claimed_by": native_middle_work["claimed_by"],
                    "run_id": native_middle_work["run_id"],
                    "step_id": native_middle_work["step_id"],
                    "correlation_id": native_middle_work["correlation_id"]
                },
                "payload": wrong_target_payload
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(wrong_target_result.status(), StatusCode::CONFLICT);
        assert_eq!(
            store
                .list_server_run_events("mixed-language-run", 0, 100)
                .unwrap(),
            events_before_wrong_target,
            "wrong-target rejection must not persist a worker result or trace"
        );
        let native_middle_result = client
            .post(format!("{base_url}/internal/runtime/native/execute"))
            .header("x-kogwistar-roles", "reader")
            .json(&json!({
                "handoff": {
                    "message_id": native_middle_work["message_id"],
                    "claimed_by": native_middle_work["claimed_by"],
                    "run_id": native_middle_work["run_id"],
                    "step_id": native_middle_work["step_id"],
                    "correlation_id": native_middle_work["correlation_id"]
                },
                "payload": native_middle_work["payload"]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(native_middle_result.status(), StatusCode::OK);

        let python_after_claim = client
            .post(format!("{base_url}/internal/runtime/claim"))
            .header("x-kogwistar-roles", "reader")
            .json(&json!({
                "claimed_by": "python-worker",
                "limit": 1,
                "lease_seconds": 60,
                "recipient_ids": ["python-worker"]
            }))
            .send()
            .await
            .unwrap();
        let python_after_work =
            python_after_claim.json::<Value>().await.unwrap()["work"][0].clone();
        assert_eq!(python_after_work["payload"]["node_id"], "python-after");
        assert!(python_after_work["payload"]["executor_target"].is_null());
        assert_eq!(python_after_work["payload"]["state"]["counter"], 5);
        assert_eq!(
            python_after_work["payload"]["state"]["python_before_seen"],
            true
        );
        let python_after_result = client
            .post(format!("{base_url}/internal/runtime/results"))
            .header("x-kogwistar-roles", "reader")
            .json(&json!({
                "handoff": {
                    "message_id": python_after_work["message_id"],
                    "claimed_by": python_after_work["claimed_by"],
                    "run_id": python_after_work["run_id"],
                    "step_id": python_after_work["step_id"],
                    "correlation_id": python_after_work["correlation_id"]
                },
                "effect": {
                    "contract_version": 1,
                    "effect_id": "mixed-python-after",
                    "state_update": [["u", {"python_after_seen": true}]],
                    "successors": [],
                    "result": {"workflow_status": "succeeded", "answer": "mixed"}
                }
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(python_after_result.status(), StatusCode::OK);
        let mixed_run = store.get_server_run("mixed-language-run").unwrap().unwrap();
        assert_eq!(mixed_run.status, "succeeded");
        let mixed_events = store
            .list_server_run_events("mixed-language-run", 0, 100)
            .unwrap();
        assert!(mixed_events.iter().any(|event| {
            event.event_type == "workflow.trace.v1"
                && event.payload_json.contains("native-middle")
                && event.payload_json.contains("seam_crossings")
        }));
        let native_trace = mixed_events
            .iter()
            .filter(|event| event.event_type == "workflow.trace.v1")
            .map(|event| serde_json::from_str::<Value>(&event.payload_json).unwrap())
            .find(|event| event["trace"]["event"] == "runtime.native.step.completed")
            .expect("successful Rust-native step must persist its trace");
        assert_eq!(native_trace["trace"]["executor_target"], "rust-native");
        let trace = &native_trace["trace"];
        let token_id = native_middle_work["payload"]["token_id"]
            .as_str()
            .expect("native step payload has its runtime token");
        let step_seq = native_middle_work["payload"]["step_seq"]
            .as_i64()
            .expect("native step payload has its sequence");
        assert_eq!(trace["trace_id"], "mixed-language-run");
        assert_eq!(trace["attempt"], 1);
        assert_eq!(trace["token_id"], token_id);
        assert_eq!(trace["step_seq"], step_seq);
        assert_eq!(trace["node_id"], "native-middle");
        assert_eq!(trace["span_id"], format!("stp:mixed-language-run:{token_id}:{step_seq}:1:native-middle"));
        assert_eq!(trace["parent_span_id"], format!("tok:mixed-language-run:{token_id}"));
        assert_eq!(native_trace["run_id"], "mixed-language-run");
        assert_eq!(native_trace["step_id"], "native-middle");
        assert_eq!(native_trace["token_id"], token_id);

        // A non-native operation must keep the legacy Python-worker lane and
        // must not acquire native executor metadata implicitly.
        let legacy_submit = client
            .post(format!("{base_url}/api/workflow/runs"))
            .header("x-kogwistar-roles", "reader,rw")
            .header("content-type", "application/json")
            .json(&json!({
                "run_id": "legacy-run",
                "workflow_id": "legacy-workflow",
                "conversation_id": "legacy-conversation",
                "initial_state": {"seed": 1},
                "runtime_kind": "sync",
                "start_node_id": "entry",
                "node_ops": {"entry": "application.process"}
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(legacy_submit.status(), StatusCode::ACCEPTED);
        let legacy_claim = client
            .post(format!("{base_url}/internal/runtime/claim"))
            .header("x-kogwistar-roles", "reader")
            .json(&json!({
                "claimed_by": "python-worker",
                "limit": 1,
                "lease_seconds": 60,
                "recipient_ids": ["python-worker"]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(legacy_claim.status(), StatusCode::OK);
        let legacy_work = legacy_claim.json::<Value>().await.unwrap()["work"][0].clone();
        assert_eq!(legacy_work["payload"]["op"], "application.process");
        assert!(legacy_work["payload"].get("executor_target").is_none());

        let native_events_before_restart = store
            .list_server_run_events("native-run", 0, 100)
            .unwrap();
        let final_transition = native_events_before_restart
            .iter()
            .rev()
            .find(|event| event.event_type == "workflow.recorded_transition.v1")
            .expect("native run must have a terminal recorded transition");
        let final_transition_payload: Value =
            serde_json::from_str(&final_transition.payload_json).unwrap();
        let expected_state_digest = final_transition_payload["reduced"]["state_digest"]
            .as_str()
            .expect("terminal transition must persist a state digest")
            .to_owned();
        let expected_runtime_state = store
            .read_recorded_runtime_state(
                "native-run",
                "native-workflow",
                "native-conversation",
            )
            .unwrap()
            .expect("native runtime state must be reconstructable before reopen");
        assert_eq!(expected_runtime_state.status.server_status(), "succeeded");
        assert_eq!(expected_runtime_state.state["counter"], json!(2));

        server.abort();
        let _ = server.await;
        drop(service);
        drop(store);

        let reopened_store = SqliteStore::open(&path).unwrap();
        let reopened_run = reopened_store
            .get_server_run("native-run")
            .unwrap()
            .expect("completed native run must survive SQLite reopen");
        assert_eq!(reopened_run.status, "succeeded");
        let reopened_runtime_state = reopened_store
            .read_recorded_runtime_state(
                "native-run",
                "native-workflow",
                "native-conversation",
            )
            .unwrap()
            .expect("runtime state must reconstruct from durable transitions");
        assert_eq!(reopened_runtime_state, expected_runtime_state);
        assert_eq!(
            kogwistar_runtime::state_digest(&reopened_runtime_state).unwrap(),
            expected_state_digest
        );
        let reopened_events = reopened_store
            .list_server_run_events("native-run", 0, 100)
            .unwrap();
        let reopened_final_transition = reopened_events
            .iter()
            .rev()
            .find(|event| event.event_type == "workflow.recorded_transition.v1")
            .expect("terminal recorded transition must survive SQLite reopen");
        let reopened_transition_payload: Value =
            serde_json::from_str(&reopened_final_transition.payload_json).unwrap();
        assert_eq!(
            reopened_transition_payload["reduced"]["state_digest"],
            expected_state_digest
        );
        drop(reopened_store);
        drop(fault_connection);
        std::fs::remove_file(path).unwrap();
    }
