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

        let resume_contract_fixture: Value = serde_json::from_str(include_str!(
            "../../../../../../contracts/golden/run-resume-contract-v1.json"
        ))
        .unwrap();
        let resume_result_fixture: Value = serde_json::from_str(include_str!(
            "../../../../../../contracts/golden/workflow-run-resume-result-v1.json"
        ))
        .unwrap();
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
                        .header("x-kogwistar-roles", "reader,rw")
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
        for field in resume_contract_fixture["required_fields"]
            .as_array()
            .unwrap()
        {
            let field = field.as_str().unwrap();
            assert!(contract.get(field).is_some(), "missing {field}");
        }
        assert_eq!(contract["compatible"], true);
        assert_eq!(contract["checkpoint_schema_version"], 1);
        assert_eq!(
            contract["supported_wait_reasons"],
            resume_contract_fixture["supported_wait_reasons"]
        );
        assert_eq!(contract["resume_options"][0]["suspended_node_id"], "start");
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
        let denied_resume = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!(
                        "/api/workflow/runs/{run_id}/resume?wait_for_completion=true"
                    ))
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(serde_json::to_vec(&resume_body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(denied_resume.status(), StatusCode::FORBIDDEN);
        let resume_app = app.clone();
        let resume_uri = format!(
            "/api/workflow/runs/{run_id}/resume"
        );
        let resume_future = tokio::spawn(async move {
            call(resume_app, "POST", resume_uri, resume_body).await
        });
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
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
        let finish_result = json!({
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
        });
        let (status, first_finish) = call(
            app.clone(),
            "POST",
            "/internal/runtime/results".to_owned(),
            finish_result.clone(),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let (retry_status, retry_finish) = call(
            app.clone(),
            "POST",
            "/internal/runtime/results".to_owned(),
            finish_result,
        )
        .await;
        assert_eq!(retry_status, StatusCode::OK);
        assert_eq!(retry_finish["event_seq"], first_finish["event_seq"]);
        assert_eq!(retry_finish["idempotent"], true);
        let (resume_status, resumed) = resume_future.await.unwrap();
        assert_eq!(resume_status, StatusCode::OK);
        let result_fields = resume_result_fixture["required_fields"]
            .as_array()
            .unwrap()
            .iter()
            .map(|field| field.as_str().unwrap())
            .collect::<std::collections::BTreeSet<_>>();
        let actual_fields = resumed
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(actual_fields, result_fields);
        assert!(resume_result_fixture["completed_statuses"]
            .as_array()
            .unwrap()
            .contains(&resumed["workflow_status"]));
        assert_eq!(resumed["workflow_status"], "succeeded");
        assert_eq!(resumed["final_state"]["before_suspend"], true);
        assert_eq!(resumed["final_state"]["after_resume"], true);
        assert_eq!(resumed["budget"], json!({}));
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
