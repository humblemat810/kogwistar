    #[tokio::test]
    async fn sqlite_environment_pinned_work_resumes_on_another_compatible_worker() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use kogwistar_store::{NamedProjectionWrite, WorkflowDesignSnapshotWrite};
        use std::time::{SystemTime, UNIX_EPOCH};
        use tower::ServiceExt;

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-worker-env-{nonce}.sqlite3"));
        let workflow_id = "pinned-python-workflow";
        let conversation_id = "pinned-python-conversation";
        let environment_id = "numpy-cpu-v2";
        let store = SqliteStore::open(&path).unwrap();
        store
            .replace_named_projection(
                "workflow_design",
                workflow_id,
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
                workflow_id,
                WorkflowDesignSnapshotWrite {
                    version: 1,
                    seq: 1,
                    schema_version: 1,
                    payload_json: serde_json::to_string(&json!({
                        "nodes": [
                            {"id":"first","metadata":{"wf_start":true,"wf_op":"application.compute","wf_environment_id":environment_id}},
                            {"id":"second","metadata":{"wf_op":"application.compute","wf_environment_id":environment_id}}
                        ],
                        "edges": [{
                            "id":"edge-first-second",
                            "source_ids":["first"],
                            "target_ids":["second"],
                            "metadata":{}
                        }]
                    }))
                    .unwrap(),
                },
            )
            .unwrap();

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
        let call = |app: Router, method: &'static str, uri: String, body: Value| async move {
            let response = app
                .oneshot(
                    Request::builder()
                        .method(method)
                        .uri(uri)
                        .header("content-type", "application/json")
                        .header("x-kogwistar-roles", "reader,rw")
                        .body(Body::from(serde_json::to_vec(&body).unwrap()))
                        .unwrap(),
                )
                .await
                .unwrap();
            let status = response.status();
            let bytes = to_bytes(response.into_body(), 32768).await.unwrap();
            (status, serde_json::from_slice::<Value>(&bytes).unwrap())
        };

        let app = reopen();
        let (status, submitted) = call(
            app.clone(),
            "POST",
            "/api/workflow/runs".to_owned(),
            json!({
                "workflow_id": workflow_id,
                "conversation_id": conversation_id,
                "initial_state": {"seed": 7}
            }),
        )
        .await;
        assert_eq!(status, StatusCode::ACCEPTED);
        let run_id = submitted["run_id"].as_str().unwrap().to_owned();

        let generic_worker = call(
            app.clone(),
            "POST",
            "/internal/runtime/claim".to_owned(),
            json!({
                "claimed_by": "generic-worker",
                "limit": 1,
                "lease_seconds": 60,
                "recipient_ids": ["python-worker"],
                "run_id": run_id
            }),
        )
        .await;
        assert_eq!(generic_worker.0, StatusCode::OK);
        assert_eq!(generic_worker.1["work"], json!([]));

        let wrong_environment = call(
            app.clone(),
            "POST",
            "/internal/runtime/claim".to_owned(),
            json!({
                "claimed_by": "worker-wrong-env",
                "limit": 1,
                "lease_seconds": 60,
                "recipient_ids": [crate::runtime_contract::runtime_worker_recipient(Some("numpy-gpu-v1"))]
            }),
        )
        .await;
        assert_eq!(wrong_environment.0, StatusCode::OK);
        assert_eq!(wrong_environment.1["work"], json!([]));

        let (status, claimed) = call(
            app.clone(),
            "POST",
            "/internal/runtime/claim".to_owned(),
            json!({
                "claimed_by": "worker-a",
                "limit": 1,
                "lease_seconds": 60,
                "recipient_ids": [crate::runtime_contract::runtime_worker_recipient(Some(environment_id))],
                "run_id": run_id
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let first_work = claimed["work"][0].clone();
        assert_eq!(first_work["step_id"], "first");
        assert_eq!(first_work["payload"]["op_environment_id"], environment_id);
        let first_result = call(
            app,
            "POST",
            "/internal/runtime/results".to_owned(),
            json!({
                "handoff": {
                    "message_id": first_work["message_id"],
                    "claimed_by": first_work["claimed_by"],
                    "run_id": first_work["run_id"],
                    "step_id": first_work["step_id"],
                    "correlation_id": first_work["correlation_id"]
                },
                "effect": {
                    "contract_version": 1,
                    "effect_id": format!("first-{}", run_id),
                    "state_update": [["u", {"first_worker": "worker-a"}]],
                    "successors": [{"node_id":"second","join_mask":0}],
                    "result": null
                }
            }),
        )
        .await;
        assert_eq!(first_result.0, StatusCode::OK);

        // Reopen the authoritative service between steps. A different worker
        // advertising the same environment must recover the next durable task.
        let app = reopen();
        let (status, resumed) = call(
            app.clone(),
            "POST",
            "/internal/runtime/claim".to_owned(),
            json!({
                "claimed_by": "worker-b",
                "limit": 1,
                "lease_seconds": 60,
                "recipient_ids": [crate::runtime_contract::runtime_worker_recipient(Some(environment_id))],
                "run_id": run_id
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let second_work = resumed["work"][0].clone();
        assert!(second_work["claimed_by"]
            .as_str()
            .unwrap()
            .starts_with("worker-b#"));
        assert_eq!(second_work["step_id"], "second");
        assert_eq!(second_work["payload"]["op_environment_id"], environment_id);
        assert_eq!(second_work["payload"]["state"]["first_worker"], "worker-a");
        let (status, _) = call(
            app.clone(),
            "POST",
            "/internal/runtime/results".to_owned(),
            json!({
                "handoff": {
                    "message_id": second_work["message_id"],
                    "claimed_by": second_work["claimed_by"],
                    "run_id": second_work["run_id"],
                    "step_id": second_work["step_id"],
                    "correlation_id": second_work["correlation_id"]
                },
                "effect": {
                    "contract_version": 1,
                    "effect_id": format!("second-{}", run_id),
                    "successors": [],
                    "result": {"workflow_status":"succeeded"}
                }
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);

        let (_, run) = call(
            app,
            "GET",
            format!("/api/workflow/runs/{run_id}"),
            Value::Null,
        )
        .await;
        assert_eq!(run["status"], "succeeded");
        drop(store);
        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn sqlite_invalid_worker_identity_pin_is_rejected_before_scheduling() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use kogwistar_store::{NamedProjectionWrite, WorkflowDesignSnapshotWrite};
        use std::time::{SystemTime, UNIX_EPOCH};
        use tower::ServiceExt;

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-invalid-worker-pin-{nonce}.sqlite3"));
        let workflow_id = "invalid-worker-pin-workflow";
        let store = SqliteStore::open(&path).unwrap();
        store
            .replace_named_projection(
                "workflow_design",
                workflow_id,
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
                workflow_id,
                WorkflowDesignSnapshotWrite {
                    version: 1,
                    seq: 1,
                    schema_version: 1,
                    payload_json: serde_json::to_string(&json!({
                        "nodes": [
                            {"id":"start","metadata":{
                                "wf_start":true,
                                "wf_op":"application.compute",
                                "wf_environment_id":"python-cpu-v1",
                                "wf_worker_identity":{"schema_version":99}
                            }},
                            {"id":"end","metadata":{"wf_terminal":true}}
                        ],
                        "edges": [{"id":"route","source_ids":["start"],"target_ids":["end"],"metadata":{}}]
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
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/workflow/runs")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader,rw")
                    .body(Body::from(
                        json!({"workflow_id":workflow_id,"conversation_id":"invalid-pin-conv"})
                            .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::CONFLICT);
        let body = to_bytes(response.into_body(), 16384).await.unwrap();
        let error: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(error["code"], "KOGWISTAR_INVALID_WORKER_IDENTITY");
        let queued = store
            .list_projected_lane_messages(LaneMessageFilter {
                namespace: Some("workflow".to_owned()),
                inbox_id: Some("workflow-runtime".to_owned()),
                status: Some("pending".to_owned()),
                limit: 32,
                ..LaneMessageFilter::default()
            })
            .unwrap();
        assert!(queued.is_empty(), "invalid pin must not enqueue runtime work");
        drop(store);
        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn sqlite_worker_identity_pin_survives_service_reopen_and_controls_claim() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
        use kogwistar_store::{NamedProjectionWrite, WorkflowDesignSnapshotWrite};
        use std::time::{SystemTime, UNIX_EPOCH};
        use tower::ServiceExt;

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-worker-pin-reopen-{nonce}.sqlite3"));
        let workflow_id = "worker-pin-reopen-workflow";
        let environment_id = "python-cpu-v1";
        let identity = json!({
            "schema_version":1,
            "environment_id":environment_id,
            "artifact_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "dependency_lock_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        });
        let identity_recipient = runtime_worker_identity_recipient(
            environment_id,
            identity["artifact_sha256"].as_str().unwrap(),
            identity["dependency_lock_sha256"].as_str().unwrap(),
        );
        let auth = AuthConfig {
            key: Some("worker-registration-test-secret".to_owned()),
            issuer: Some("test-worker-registry".to_owned()),
            audience: Some("kogwistar-runtime".to_owned()),
            ..AuthConfig::default()
        };
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let token = |registered_identity: Option<Value>| {
            let mut claims = json!({
                "sub":"identity-worker",
                "role":"reader",
                "capabilities":["spawn_process", "workflow.run.write"],
                "iss":"test-worker-registry",
                "aud":"kogwistar-runtime",
                "exp": now + 600
            });
            if let Some(identity) = registered_identity {
                claims[REGISTERED_WORKER_IDENTITY_CLAIM] = identity;
            }
            encode(
                &Header::new(Algorithm::HS256),
                &claims,
                &EncodingKey::from_secret(b"worker-registration-test-secret"),
            )
            .unwrap()
        };
        let store = SqliteStore::open(&path).unwrap();
        store
            .replace_named_projection(
                "workflow_design",
                workflow_id,
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
                workflow_id,
                WorkflowDesignSnapshotWrite {
                    version: 1,
                    seq: 1,
                    schema_version: 1,
                    payload_json: serde_json::to_string(&json!({
                        "nodes": [
                            {"id":"start","metadata":{
                                "wf_start":true,
                                "wf_op":"application.compute",
                                "wf_environment_id":environment_id,
                                "wf_worker_identity":identity
                            }},
                            {"id":"end","metadata":{"wf_terminal":true}}
                        ],
                        "edges": [{"id":"route","source_ids":["start"],"target_ids":["end"],"metadata":{}}]
                    }))
                    .unwrap(),
                },
            )
            .unwrap();

        let reopen = || {
            router_with_application(
                ApiState {
                    health: snapshot(),
                    required_roles: vec!["reader".to_owned()],
                    implementation: ImplementationSnapshot::default(),
                    auth: auth.clone(),
                },
                Arc::new(SqliteRunApplicationService::open(&path).unwrap()),
            )
        };
        let call = |app: Router,
                    method: &'static str,
                    uri: String,
                    body: Value,
                    bearer: String| async move {
            let response = app
                .oneshot(
                    Request::builder()
                        .method(method)
                        .uri(uri)
                        .header("content-type", "application/json")
                        .header("authorization", format!("Bearer {bearer}"))
                        .body(Body::from(body.to_string()))
                        .unwrap(),
                )
                .await
                .unwrap();
            let status = response.status();
            let bytes = to_bytes(response.into_body(), 32768).await.unwrap();
            (status, serde_json::from_slice::<Value>(&bytes).unwrap())
        };

        let (status, submitted) = call(
            reopen(),
            "POST",
            "/api/workflow/runs".to_owned(),
            json!({"workflow_id":workflow_id,"conversation_id":"identity-reopen-conv"}),
            token(None),
        )
        .await;
        assert_eq!(status, StatusCode::ACCEPTED);
        let run_id = submitted["run_id"].as_str().unwrap().to_owned();
        let claim_uri = "/internal/runtime/claim".to_owned();
        let claim_body = json!({
            "claimed_by":"identity-worker",
            "limit":1,
            "lease_seconds":60,
            "recipient_ids":[identity_recipient],
            "run_id":run_id
        });
        let (status, _) = call(
            reopen(),
            "POST",
            claim_uri.clone(),
            claim_body.clone(),
            token(None),
        )
        .await;
        assert_eq!(status, StatusCode::FORBIDDEN);
        for (field, value) in [
            ("environment_id", json!("other-python-cpu-v1")),
            (
                "artifact_sha256",
                json!("cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"),
            ),
            (
                "dependency_lock_sha256",
                json!("dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"),
            ),
        ] {
            let mut unregistered = identity.clone();
            unregistered[field] = value;
            let (status, _) = call(
                reopen(),
                "POST",
                claim_uri.clone(),
                claim_body.clone(),
                token(Some(unregistered)),
            )
            .await;
            assert_eq!(status, StatusCode::FORBIDDEN, "tampered {field} was admitted");
        }
        let (status, claim) = call(
            reopen(),
            "POST",
            claim_uri,
            claim_body,
            token(Some(identity.clone())),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(claim["work"][0]["step_id"], "start");
        assert_eq!(
            claim["work"][0]["payload"]["initial_state"]["_rt_node_worker_identities"]["start"],
            identity
        );
        assert_eq!(claim["work"][0]["payload"]["required_worker_identity"], identity);
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
                    .header("x-kogwistar-roles", "reader,rw")
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
