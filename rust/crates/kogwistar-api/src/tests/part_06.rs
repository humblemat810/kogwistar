    #[tokio::test]
    async fn sqlite_submit_enforces_atomic_queue_backpressure() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use std::time::{SystemTime, UNIX_EPOCH};
        use tower::ServiceExt;

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-backpressure-{nonce}.sqlite3"));
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: vec!["reader".to_owned()],
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig::default(),
            },
            Arc::new(
                SqliteRunApplicationService::open(&path)
                    .unwrap()
                    .with_max_queue(1),
            ),
        );
        let submit = |priority_class: &str| {
            Request::builder()
                .method("POST")
                .uri("/api/workflow/runs")
                .header("content-type", "application/json")
                .header("x-kogwistar-roles", "reader")
                .body(Body::from(
                    serde_json::to_vec(&json!({
                        "workflow_id": "backpressure-wf",
                        "conversation_id": "backpressure-conv",
                        "priority_class": priority_class,
                    }))
                    .unwrap(),
                ))
                .unwrap()
        };
        let first = app.clone().oneshot(submit("foreground")).await.unwrap();
        assert_eq!(first.status(), StatusCode::ACCEPTED);
        let claimed = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/claim")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"claimed_by":"backpressure-worker","limit":1,"lease_seconds":60}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(claimed.status(), StatusCode::OK);
        let claimed =
            serde_json::from_slice::<Value>(&to_bytes(claimed.into_body(), 8192).await.unwrap())
                .unwrap();
        assert_eq!(claimed["work"][0]["payload"]["op"], "noop");
        for (priority_class, admission) in [("foreground", "deferred"), ("batch", "rejected")] {
            let response = app.clone().oneshot(submit(priority_class)).await.unwrap();
            assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
            let body = to_bytes(response.into_body(), 8192).await.unwrap();
            let value = serde_json::from_slice::<Value>(&body).unwrap();
            assert_eq!(value["admission"], admission);
            assert_eq!(value["reason"], "queue_full");
            assert_eq!(value["max_queue"], 1);
        }
        let store = SqliteStore::open(&path).unwrap();
        assert_eq!(
            store.list_server_runs(None, None, None, 10).unwrap().len(),
            1
        );
        assert_eq!(
            store
                .list_projected_lane_messages(LaneMessageFilter {
                    namespace: Some("workflow".to_owned()),
                    inbox_id: Some("workflow-runtime".to_owned()),
                    ..LaneMessageFilter::default()
                })
                .unwrap()
                .len(),
            1
        );
        drop(store);
        drop(app);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn snapshot_syscall_compatibility_drops_rust_only_fields() {
        let expected = serde_json::from_str::<Value>(include_str!(
            "../../../../../contracts/golden/mount-memory-result-v1.json"
        ))
        .unwrap();
        let mut internal_payload = expected.clone();
        let internal_object = internal_payload.as_object_mut().unwrap();
        internal_object.insert("schema_version".to_owned(), json!(1));
        internal_object.insert("implementation_authority".to_owned(), json!("rust"));
        internal_object.insert("seam_attestation_version".to_owned(), json!(1));
        internal_object.insert("seam_crossings".to_owned(), json!([]));
        let mut response = json_effect(StatusCode::OK, internal_payload);
        strip_snapshot_implementation_fields(&mut response);
        let actual = serde_json::from_slice::<Value>(&response.body).unwrap();
        assert_eq!(actual, expected);

        let mut failure = json_effect(
            StatusCode::NOT_FOUND,
            json!({"error": {"code": "KOGWISTAR_CONTEXT_SNAPSHOT_NOT_FOUND"}}),
        );
        let original_body = failure.body.clone();
        strip_snapshot_implementation_fields(&mut failure);
        assert_eq!(failure.body, original_body);
    }

    #[test]
    fn receive_message_compatibility_matches_shared_wire_fixture() {
        let expected = serde_json::from_str::<Value>(include_str!(
            "../../../../../contracts/golden/receive-message-result-v1.json"
        ))
        .unwrap();
        let mut internal_payload = expected.clone();
        let internal_object = internal_payload.as_object_mut().unwrap();
        let mut turns = internal_object.remove("messages").unwrap();
        for turn in turns.as_array_mut().unwrap() {
            turn.as_object_mut()
                .unwrap()
                .insert("schema_version".to_owned(), json!(1));
        }
        internal_object.insert("turns".to_owned(), turns);
        internal_object.insert("schema_version".to_owned(), json!(1));
        internal_object.insert("implementation_authority".to_owned(), json!("rust"));
        internal_object.insert("seam_attestation_version".to_owned(), json!(1));
        internal_object.insert("seam_crossings".to_owned(), json!([]));
        let mut response = json_effect(StatusCode::OK, internal_payload);

        legacy_receive_message_payload(&mut response);

        assert_eq!(response.status, StatusCode::OK.as_u16());
        let actual = serde_json::from_slice::<Value>(&response.body).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn receive_message_invalid_internal_payload_fails_closed() {
        let mut response = json_effect(
            StatusCode::OK,
            json!({
                "schema_version": 1,
                "conversation_id": "conversation-1",
                "turns": [],
                "unexpected": true
            }),
        );

        legacy_receive_message_payload(&mut response);

        assert_eq!(response.status, StatusCode::INTERNAL_SERVER_ERROR.as_u16());
        let error = serde_json::from_slice::<Value>(&response.body).unwrap();
        assert_eq!(error["code"], "KOGWISTAR_INVALID_RESPONSE");
    }

    #[tokio::test]
    async fn sqlite_syscall_spawn_and_terminate_use_authoritative_run_service() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-syscall-{nonce}.sqlite3"));
        let service = SqliteRunApplicationService::open(&path).unwrap();
        let view_request = |capabilities: &[&str]| ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/syscall/v1/project_view".to_owned(),
            body: serde_json::to_vec(&json!({
                "version":"v1",
                "op":"project_view",
                "args":{}
            }))
            .unwrap(),
            principal: json!({
                "sub":"viewer",
                "role":"ro",
                "ns":"workflow",
                "capabilities":capabilities,
            }),
        };
        let denied_view = service.execute_sync(view_request(&["project_view"]));
        assert_eq!(denied_view.status, StatusCode::FORBIDDEN.as_u16());
        let view = service.execute_sync(view_request(&["project_view", "read_security_scope"]));
        assert_eq!(view.status, StatusCode::OK.as_u16());
        let view = serde_json::from_slice::<Value>(&view.body).unwrap();
        assert_eq!(view["version"], "v1");
        assert_eq!(view["op"], "project_view");
        assert_eq!(view["status"], "ok");
        assert!(view["result"]["scheduler"].is_object());
        assert!(view["result"]["runs"].is_object());
        let receive_request = |namespace: &str| ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/syscall/v1/receive_message".to_owned(),
            body: serde_json::to_vec(&json!({
                "version":"v1",
                "op":"receive_message",
                "args":{"conversation_id":"conversation-1"}
            }))
            .unwrap(),
            principal: json!({
                "sub":"viewer",
                "role":"ro",
                "ns":namespace,
                "capabilities":["read_graph"],
            }),
        };
        assert_eq!(
            service.execute_sync(receive_request("workflow")).status,
            StatusCode::FORBIDDEN.as_u16()
        );
        assert_eq!(
            service.execute_sync(receive_request("conversation")).status,
            StatusCode::NOT_IMPLEMENTED.as_u16()
        );
        let mount_request = |namespace: &str| ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/syscall/v1/mount_memory".to_owned(),
            body: serde_json::to_vec(&json!({
                "version":"v1",
                "op":"mount_memory",
                "args":{"conversation_id":"conversation-1"}
            }))
            .unwrap(),
            principal: json!({
                "sub":"viewer",
                "role":"ro",
                "ns":namespace,
                "capabilities":["project_view"],
            }),
        };
        assert_eq!(
            service.execute_sync(mount_request("workflow")).status,
            StatusCode::FORBIDDEN.as_u16()
        );
        assert_eq!(
            service.execute_sync(mount_request("conversation")).status,
            StatusCode::NOT_IMPLEMENTED.as_u16()
        );
        let principal = json!({
            "sub":"operator",
            "role":"rw",
            "ns":"workflow",
            "capabilities":["spawn_process","workflow.run.write","workflow.run.read"],
        });
        let spawned = service.execute_sync(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/syscall/v1/spawn_process".to_owned(),
            body: serde_json::to_vec(&json!({
                "version":"v1",
                "op":"spawn_process",
                "args":{
                    "workflow_id":"wf-syscall",
                    "conversation_id":"conv-syscall",
                    "initial_state":{
                        "source":"syscall",
                        "_rt_requested_capabilities":[
                            "workflow.run.write"
                        ],
                        "_rt_effective_capabilities":["service.manage"]
                    },
                },
            }))
            .unwrap(),
            principal: principal.clone(),
        });
        assert_eq!(spawned.status, StatusCode::OK.as_u16());
        let spawned = serde_json::from_slice::<Value>(&spawned.body).unwrap();
        assert_eq!(spawned["version"], "v1");
        assert_eq!(spawned["op"], "spawn_process");
        assert_eq!(spawned["status"], "ok");
        let run_id = spawned["result"]["run_id"].as_str().unwrap();
        assert!(service.store.get_server_run(run_id).unwrap().is_some());
        let lanes = service
            .store
            .list_projected_lane_messages(LaneMessageFilter {
                namespace: Some("workflow".to_owned()),
                inbox_id: Some("workflow-runtime".to_owned()),
                correlation_id: Some(run_id.to_owned()),
                ..LaneMessageFilter::default()
            })
            .unwrap();
        let payload =
            serde_json::from_str::<Value>(lanes[0].payload_json.as_deref().unwrap()).unwrap();
        assert_eq!(
            payload["initial_state"]["_rt_effective_capabilities"],
            json!(["workflow.run.write"])
        );
        assert_eq!(payload["initial_state"]["_rt_delegated_by"], "operator");
        assert_eq!(
            payload["initial_state"]["_rt_delegation"],
            json!({
                "schema_version": 1,
                "delegated_by": "operator",
                "effective_capabilities": ["workflow.run.write"]
            })
        );
        let terminated = service.execute_sync(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/syscall/v1/terminate_process".to_owned(),
            body: serde_json::to_vec(&json!({
                "version":"v1",
                "op":"terminate_process",
                "args":{"run_id":run_id},
            }))
            .unwrap(),
            principal,
        });
        assert_eq!(terminated.status, StatusCode::OK.as_u16());
        let terminated = serde_json::from_slice::<Value>(&terminated.body).unwrap();
        assert_eq!(terminated["result"]["cancel_requested"], true);
        assert!(
            service
                .store
                .get_server_run(run_id)
                .unwrap()
                .unwrap()
                .cancel_requested
        );
        drop(service);
        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn sqlite_worker_suspend_resume_is_durable_and_scheduler_owned() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use std::time::{SystemTime, UNIX_EPOCH};
        use tower::ServiceExt;

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-resume-{nonce}.sqlite3"));
        let reopen = || {
            router_with_application(
                ApiState {
                    health: snapshot(),
                    required_roles: vec!["reader".to_owned()],
                    implementation: ImplementationSnapshot::default(),
                    auth: AuthConfig::default(),
                },
                Arc::new(SqliteRunApplicationService::open(&path).unwrap()),
            )
        };
        let mut app = reopen();
        let call = |app: Router, method: &'static str, uri: String, body: Value| async move {
            let response = app
                .oneshot(
                    Request::builder()
                        .method(method)
                        .uri(uri)
                        .header("content-type", "application/json")
                        .header("x-kogwistar-roles", "reader")
                        .body(Body::from(if body.is_null() {
                            Vec::new()
                        } else {
                            serde_json::to_vec(&body).unwrap()
                        }))
                        .unwrap(),
                )
                .await
                .unwrap();
            let status = response.status();
            let bytes = to_bytes(response.into_body(), 32768).await.unwrap();
            (status, serde_json::from_slice::<Value>(&bytes).unwrap())
        };
        let (status, submitted) = call(
            app.clone(),
            "POST",
            "/api/workflow/runs".to_owned(),
            json!({"workflow_id":"resume-wf","conversation_id":"resume-conv","turn_node_id":"turn-original"}),
        )
        .await;
        assert_eq!(status, StatusCode::ACCEPTED);
        let run_id = submitted["run_id"].as_str().unwrap().to_owned();
        drop(app);
        app = reopen();
        let (_, claimed) = call(
            app.clone(),
            "POST",
            "/internal/runtime/claim".to_owned(),
            json!({"claimed_by":"suspender","limit":1,"lease_seconds":60}),
        )
        .await;
        let work = &claimed["work"][0];
        drop(app);
        app = reopen();
        let suspend = json!({
            "handoff": {
                "message_id": work["message_id"],
                "claimed_by": work["claimed_by"],
                "run_id": work["run_id"],
                "step_id": work["step_id"],
                "correlation_id": work["correlation_id"],
            },
            "effect": {
                "contract_version": 1,
                "effect_id": format!("suspend-{run_id}"),
                "status": "suspended",
                "state_update": [["u", {"before_suspend": true}]],
                "successors": [],
                "wait_reason": "approval",
                "resume_payload": {"question": "continue?"},
            }
        });
        let (status, _) = call(
            app.clone(),
            "POST",
            "/internal/runtime/results".to_owned(),
            suspend,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        drop(app);
        app = reopen();
        let (status, contract) = call(
            app.clone(),
            "GET",
            format!("/api/workflow/runs/{run_id}/resume-contract"),
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(contract["wait_reason"], "approval");
        assert_eq!(contract["suspended"][0][0], "start");
        assert_eq!(contract["runtime_routes"], json!([]));
        assert_eq!(contract["node_ops"], json!({}));
        let token_id = contract["suspended"][0][2].as_str().unwrap().to_owned();
        let resume_body = json!({
            "suspended_node_id": "start",
            "suspended_token_id": token_id,
            "client_result": {
                "status": "success",
                "state_update": [["u", {"approved": true}]],
                "successors": [],
                "route_next": [],
                "result": {"workflow_status": "succeeded"}
            },
            "workflow_id": "resume-wf",
            "conversation_id": "resume-conv",
            "turn_node_id": "unused-compatible-field",
        });
        let (status, resumed) = call(
            app.clone(),
            "POST",
            format!("/api/workflow/runs/{run_id}/resume"),
            resume_body,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(resumed["state"]["status"], "running");
        assert_eq!(resumed["state"]["wait_reason"], Value::Null);
        drop(app);
        app = reopen();
        let (_, claimed) = call(
            app.clone(),
            "POST",
            "/internal/runtime/claim".to_owned(),
            json!({"claimed_by":"finisher","limit":1,"lease_seconds":60}),
        )
        .await;
        let work = &claimed["work"][0];
        assert_eq!(work["step_id"], "start");
        assert_eq!(work["payload"]["turn_node_id"], "turn-original");
        assert_eq!(work["payload"]["resume_effect"]["status"], "success");
        assert_eq!(
            work["payload"]["resume_effect"]["state_update"][0][1]["approved"],
            true
        );
        let (status, _) = call(
            app.clone(),
            "POST",
            "/internal/runtime/results".to_owned(),
            json!({
                "handoff": {
                    "message_id": work["message_id"],
                    "claimed_by": work["claimed_by"],
                    "run_id": work["run_id"],
                    "step_id": work["step_id"],
                    "correlation_id": work["correlation_id"],
                },
                "effect": {
                    "contract_version": 1,
                    "effect_id": format!("finish-{run_id}"),
                    "state_update": [["u", {"after_resume": true}]],
                    "successors": [],
                    "result": {"done": true},
                }
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        drop(app);
        app = reopen();
        let (status, run) = call(
            app,
            "GET",
            format!("/api/workflow/runs/{run_id}"),
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(run["status"], "succeeded");
        assert_eq!(run["result"]["done"], true);
        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn sqlite_submit_freezes_exact_ready_graph_snapshot_as_authoritative_routes() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use kogwistar_store::{NamedProjectionWrite, WorkflowDesignSnapshotWrite};
        use std::time::{SystemTime, UNIX_EPOCH};
        use tower::ServiceExt;

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-graph-route-{nonce}.sqlite3"));
        let store = SqliteStore::open(&path).unwrap();
        store
            .replace_named_projection(
                "workflow_design",
                "graph-wf",
                NamedProjectionWrite {
                    payload: serde_json::Map::from_iter([
                        ("current_version".to_owned(), json!(1)),
                        ("snapshot_schema_version".to_owned(), json!(1)),
                    ]),
                    last_authoritative_seq: 1,
                    last_materialized_seq: 1,
                    projection_schema_version: 1,
                    materialization_status: "ready".to_owned(),
                },
            )
            .unwrap();
        store
            .put_workflow_design_snapshot(
                "graph-wf",
                WorkflowDesignSnapshotWrite {
                    version: 1,
                    seq: 1,
                    schema_version: 1,
                    payload_json: serde_json::to_string(&json!({
                        "nodes": [
                            {"id":"graph-start","metadata":{"wf_start":true,"wf_op":"noop","wf_executor_target":"rust-native"}},
                            {"id":"graph-end","metadata":{"wf_terminal":true}}
                        ],
                        "edges": [{
                            "id":"edge-1",
                            "source_ids":["graph-start"],
                            "target_ids":["graph-end"],
                            "metadata":{}
                        }]
                    }))
                    .unwrap(),
                },
            )
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
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/workflow/runs")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"workflow_id":"graph-wf","conversation_id":"graph-conv","initial_state":{"_rt_node_executor_targets":{"graph-end":"rust-native"}},"runtime_routes":[{"source_node_id":"start","target_node_id":"forged","join_mask":0}]}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::ACCEPTED);
        let generic_claim = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/claim")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"claimed_by":"graph-worker","limit":1,"lease_seconds":60}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let generic_claim_body = to_bytes(generic_claim.into_body(), 16384).await.unwrap();
        let generic_claim_work =
            serde_json::from_slice::<Value>(&generic_claim_body).unwrap()["work"].clone();
        assert_eq!(generic_claim_work, json!([]));

        let claim = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/claim")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        serde_json::to_vec(&json!({
                            "claimed_by": "numpy-worker",
                            "limit": 1,
                            "lease_seconds": 60,
                            "recipient_ids": [
                                "rust-native-worker",
                                "python-worker",
                            ],
                        }))
                        .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let claim_body = to_bytes(claim.into_body(), 16384).await.unwrap();
        let work = serde_json::from_slice::<Value>(&claim_body).unwrap()["work"][0].clone();
        assert_eq!(work["step_id"], "graph-start");
        assert_eq!(work["payload"]["op"], "noop");
        assert_eq!(work["payload"]["executor_target"], "rust-native");
        let result = app
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
                                "message_id": work["message_id"],
                                "claimed_by": work["claimed_by"],
                                "run_id": work["run_id"],
                                "step_id": work["step_id"],
                                "correlation_id": work["correlation_id"],
                            },
                            "effect": {
                                "contract_version": 1,
                                "effect_id": format!("graph-start-{}", work["run_id"].as_str().unwrap()),
                                "successors": [{"node_id":"worker-forged","join_mask":0}],
                            }
                        }))
                        .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(result.status(), StatusCode::OK);
        let next = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/claim")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"claimed_by":"graph-worker-2","limit":1,"lease_seconds":60}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let next_body = to_bytes(next.into_body(), 16384).await.unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&next_body).unwrap()["work"][0]["step_id"],
            "graph-end"
        );
        drop(store);
        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn sqlite_predicate_snapshot_accepts_only_graph_valid_worker_selection() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use kogwistar_store::{NamedProjectionWrite, WorkflowDesignSnapshotWrite};
        use std::time::{SystemTime, UNIX_EPOCH};
        use tower::ServiceExt;

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-predicate-route-{nonce}.sqlite3"));
        let store = SqliteStore::open(&path).unwrap();
        store
            .replace_named_projection(
                "workflow_design",
                "predicate-wf",
                NamedProjectionWrite {
                    payload: serde_json::Map::from_iter([
                        ("current_version".to_owned(), json!(1)),
                        ("snapshot_schema_version".to_owned(), json!(1)),
                    ]),
                    last_authoritative_seq: 1,
                    last_materialized_seq: 1,
                    projection_schema_version: 1,
                    materialization_status: "ready".to_owned(),
                },
            )
            .unwrap();
        store
            .put_workflow_design_snapshot(
                "predicate-wf",
                WorkflowDesignSnapshotWrite {
                    version: 1,
                    seq: 1,
                    schema_version: 1,
                    payload_json: serde_json::to_string(&json!({
                        "nodes": [
                            {"id":"gate","metadata":{"wf_start":true}},
                            {"id":"left","metadata":{"wf_terminal":true}},
                            {"id":"right","metadata":{"wf_terminal":true}}
                        ],
                        "edges": [
                            {"id":"to-left","source_ids":["gate"],"target_ids":["left"],"metadata":{"wf_predicate":"if_true"}},
                            {"id":"to-right","source_ids":["gate"],"target_ids":["right"],"metadata":{"wf_predicate":"if_false","wf_is_default":true}}
                        ]
                    }))
                    .unwrap(),
                },
            )
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
                        r#"{"workflow_id":"predicate-wf","conversation_id":"predicate-conv"}"#,
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
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"claimed_by":"predicate-worker","limit":1,"lease_seconds":60}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let claim_body = to_bytes(claim.into_body(), 16_384).await.unwrap();
        let work = serde_json::from_slice::<Value>(&claim_body).unwrap()["work"][0].clone();
        let envelope = |successor: Value| {
            json!({
                "handoff": {
                    "message_id": work["message_id"],
                    "claimed_by": work["claimed_by"],
                    "run_id": work["run_id"],
                    "step_id": work["step_id"],
                    "correlation_id": work["correlation_id"],
                },
                "effect": {
                    "contract_version": 1,
                    "effect_id": format!("predicate-{}", work["run_id"].as_str().unwrap()),
                    "successors": [successor],
                }
            })
        };
        let forged = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/results")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        serde_json::to_vec(&envelope(json!({"node_id":"forged","join_mask":0})))
                            .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(forged.status(), StatusCode::CONFLICT);
        let accepted = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/results")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        serde_json::to_vec(&envelope(json!({"node_id":"right","join_mask":0})))
                            .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(accepted.status(), StatusCode::OK);
        let next = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/claim")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"claimed_by":"predicate-finisher","limit":1,"lease_seconds":60}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let next_body = to_bytes(next.into_body(), 16_384).await.unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&next_body).unwrap()["work"][0]["step_id"],
            "right"
        );
        drop(store);
        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn tcp_transport_serves_health_over_real_socket() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(serve(
            listener,
            ApiState {
                health: snapshot(),
                required_roles: vec![],
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig::default(),
            },
        ));
        let mut stream = tokio::net::TcpStream::connect(address).await.unwrap();
        stream
            .write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
            .await
            .unwrap();
        let mut response = Vec::new();
        stream.read_to_end(&mut response).await.unwrap();
        let response = String::from_utf8(response).unwrap();
        assert!(response.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(response.contains("\"backend\":\"pg\""));
        server.abort();
    }
