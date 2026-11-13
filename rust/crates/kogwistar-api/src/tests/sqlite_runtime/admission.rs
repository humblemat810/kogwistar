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
                .header("x-kogwistar-roles", "reader,rw")
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
                    .header("x-kogwistar-roles", "reader,rw")
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

    #[tokio::test]
    async fn sqlite_submit_rejects_answer_workflow_without_graph_authority() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use std::time::{SystemTime, UNIX_EPOCH};
        use tower::ServiceExt;

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-answer-not-ready-{nonce}.sqlite3"));
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
                    .header("x-kogwistar-roles", "reader,rw")
                    .body(Body::from(
                        r#"{"workflow_id":"agentic_answering.v2","conversation_id":"answer-conversation"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::CONFLICT);
        let body = to_bytes(response.into_body(), 8192).await.unwrap();
        let value = serde_json::from_slice::<Value>(&body).unwrap();
        assert_eq!(value["code"], "KOGWISTAR_ANSWER_WORKFLOW_NOT_READY");

        let store = SqliteStore::open(&path).unwrap();
        assert!(store.list_server_runs(None, None, None, 10).unwrap().is_empty());
        assert!(store
            .list_projected_lane_messages(LaneMessageFilter {
                namespace: Some("workflow".to_owned()),
                inbox_id: Some("workflow-runtime".to_owned()),
                ..LaneMessageFilter::default()
            })
            .unwrap()
            .is_empty());
        drop(store);
        drop(app);
        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn sqlite_submit_freezes_exact_rust_event_graph_snapshot_as_authoritative_routes() {
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
                    materialization_status: "rust_event_only".to_owned(),
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
                    .header("x-kogwistar-roles", "reader,rw")
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
                    .header("x-kogwistar-roles", "reader,rw")
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
                            "claimed_by": "rust-native-worker",
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
        let run_id = work["run_id"].as_str().unwrap();
        let events_before_misrouted_result = store
            .list_server_run_events(run_id, 0, 100)
            .unwrap();
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
        assert_eq!(result.status(), StatusCode::CONFLICT);
        assert_eq!(
            store.list_server_run_events(run_id, 0, 100).unwrap(),
            events_before_misrouted_result,
            "misrouted result must not append durable events"
        );
        let native_result = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/native/execute")
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
                            "payload": work["payload"],
                        }))
                        .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(native_result.status(), StatusCode::OK);
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
