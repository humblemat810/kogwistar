    #[tokio::test]
    #[ignore = "requires KOGWISTAR_TEST_PG_DSN; run explicitly with --include-ignored"]
    async fn postgres_native_runtime_route_uses_shared_result_reducer_when_dsn_available() {
        let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
            .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
        let resume_contract_fixture: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/run-resume-contract-v1.json"
        ))
        .unwrap();
        let schema = format!("rust_native_runtime_{}", uuid::Uuid::new_v4().simple());
        let service = PostgresRunApplicationService::from_dsn(&dsn, &schema).unwrap();
        service.ensure_schema().await.unwrap();

        let request = |method: &str, path: &str, body: Value| ApiEffectRequest {
            contract_version: 1,
            method: method.to_owned(),
            path_and_query: path.to_owned(),
            body: serde_json::to_vec(&body).unwrap(),
            principal: json!({
                "sub": "native-worker",
                "role": "rw",
                "ns": "workflow",
                "capabilities": ["spawn_process", "workflow.run.write"]
            }),
        };
        let submit = service
            .execute(request(
                "POST",
                "/api/workflow/runs",
                json!({
                    "run_id": "pg-native-run",
                    "workflow_id": "pg-native-workflow",
                    "conversation_id": "pg-native-conversation",
                    "initial_state": {"counter": 1, "seed": 17},
                    "runtime_kind": "sync",
                    "start_node_id": "noop",
                    "node_ops": {
                        "noop": "rust-native/noop",
                        "entry": "rust-native/increment-counter",
                        "finish": "rust-native/emit-result"
                    },
                    "runtime_routes": [
                        {"source_node_id": "noop", "target_node_id": "entry", "join_mask": 0},
                        {"source_node_id": "entry", "target_node_id": "finish", "join_mask": 0}
                    ]
                }),
            ))
            .await;
        assert_eq!(submit.status, StatusCode::ACCEPTED.as_u16());
        let resume_contract = service
            .execute(request(
                "GET",
                "/api/runs/pg-native-run/resume-contract",
                Value::Null,
            ))
            .await;
        assert_eq!(
            resume_contract.status,
            resume_contract_fixture["success_status"]
                .as_u64()
                .unwrap() as u16
        );
        let resume_contract_body =
            serde_json::from_slice::<Value>(&resume_contract.body).unwrap();
        for field in resume_contract_fixture["required_fields"]
            .as_array()
            .unwrap()
        {
            let field = field.as_str().unwrap();
            assert!(resume_contract_body.get(field).is_some(), "missing {field}");
        }
        assert_eq!(resume_contract_body["run_id"], "pg-native-run");
        for (field, expected) in resume_contract_fixture["no_checkpoint"]
            .as_object()
            .unwrap()
        {
            assert_eq!(resume_contract_body[field], *expected);
        }
        assert_eq!(
            resume_contract_body["supported_wait_reasons"],
            resume_contract_fixture["supported_wait_reasons"]
        );

        let claim = service
            .execute(request(
                "POST",
                "/internal/runtime/claim",
                json!({
                    "claimed_by": "rust-native-worker",
                    "limit": 1,
                    "lease_seconds": 1,
                    "recipient_ids": ["rust-native-worker"]
                }),
            ))
            .await;
        assert_eq!(claim.status, StatusCode::OK.as_u16());
        let work = serde_json::from_slice::<Value>(&claim.body).unwrap()["work"][0].clone();
        assert_eq!(work["payload"]["op"], "rust-native/noop");
        assert_eq!(
            work["payload"]["executor_target"],
            kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET
        );

        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        let reclaimed = service
            .execute(request(
                "POST",
                "/internal/runtime/claim",
                json!({
                    "claimed_by": "rust-native-worker",
                    "limit": 1,
                    "lease_seconds": 60,
                    "recipient_ids": ["rust-native-worker"]
                }),
            ))
            .await;
        assert_eq!(reclaimed.status, StatusCode::OK.as_u16());
        let reclaimed_work = serde_json::from_slice::<Value>(&reclaimed.body).unwrap()["work"][0]
            .clone();
        assert_eq!(reclaimed_work["message_id"], work["message_id"]);
        assert_ne!(reclaimed_work["claimed_by"], work["claimed_by"]);

        let stale_execute = service
            .execute(request(
                "POST",
                "/internal/runtime/native/execute",
                json!({
                    "handoff": {
                        "message_id": work["message_id"],
                        "claimed_by": work["claimed_by"],
                        "run_id": work["run_id"],
                        "step_id": work["step_id"],
                        "correlation_id": work["correlation_id"]
                    },
                    "payload": work["payload"]
                }),
            ))
            .await;
        assert_eq!(stale_execute.status, StatusCode::CONFLICT);

        let execute = service
            .execute(request(
                "POST",
                "/internal/runtime/native/execute",
                json!({
                    "handoff": {
                        "message_id": reclaimed_work["message_id"],
                        "claimed_by": reclaimed_work["claimed_by"],
                        "run_id": reclaimed_work["run_id"],
                        "step_id": reclaimed_work["step_id"],
                        "correlation_id": reclaimed_work["correlation_id"]
                    },
                    "payload": reclaimed_work["payload"]
                }),
            ))
            .await;
        assert_eq!(execute.status, StatusCode::OK.as_u16());
        let continuation = service
            .execute(request(
                "POST",
                "/internal/runtime/claim",
                json!({
                    "claimed_by": "rust-native-worker",
                    "limit": 1,
                    "lease_seconds": 60,
                    "recipient_ids": ["rust-native-worker"]
                }),
            ))
            .await;
        assert_eq!(continuation.status, StatusCode::OK.as_u16());
        let counter_work =
            serde_json::from_slice::<Value>(&continuation.body).unwrap()["work"][0].clone();
        assert_eq!(
            counter_work["payload"]["op"],
            "rust-native/increment-counter"
        );
        let counter_execute = service
            .execute(request(
                "POST",
                "/internal/runtime/native/execute",
                json!({
                    "handoff": {
                        "message_id": counter_work["message_id"],
                        "claimed_by": counter_work["claimed_by"],
                        "run_id": counter_work["run_id"],
                        "step_id": counter_work["step_id"],
                        "correlation_id": counter_work["correlation_id"]
                    },
                    "payload": counter_work["payload"]
                }),
            ))
            .await;
        assert_eq!(counter_execute.status, StatusCode::OK.as_u16());
        let finish = service
            .execute(request(
                "POST",
                "/internal/runtime/claim",
                json!({
                    "claimed_by": "rust-native-worker",
                    "limit": 1,
                    "lease_seconds": 60,
                    "recipient_ids": ["rust-native-worker"]
                }),
            ))
            .await;
        assert_eq!(finish.status, StatusCode::OK.as_u16());
        let finish_work = serde_json::from_slice::<Value>(&finish.body).unwrap()["work"][0].clone();
        assert_eq!(finish_work["payload"]["op"], "rust-native/emit-result");
        let finish_execute = service
            .execute(request(
                "POST",
                "/internal/runtime/native/execute",
                json!({
                    "handoff": {
                        "message_id": finish_work["message_id"],
                        "claimed_by": finish_work["claimed_by"],
                        "run_id": finish_work["run_id"],
                        "step_id": finish_work["step_id"],
                        "correlation_id": finish_work["correlation_id"]
                    },
                    "payload": finish_work["payload"]
                }),
            ))
            .await;
        assert_eq!(finish_execute.status, StatusCode::OK.as_u16());
        let events = service
            .store
            .list_server_run_events("pg-native-run", 0, 100)
            .await
            .unwrap();
        let noop_transition = events
            .iter()
            .find(|event| {
                event.event_type == "workflow.recorded_transition.v1"
                    && serde_json::from_str::<Value>(&event.payload_json)
                        .ok()
                        .is_some_and(|payload| {
                            payload["reduced"]["result"]["workflow_status"] == "succeeded"
                                && payload["reduced"]["result"]["final_state"]["counter"] == 1
                                && payload["reduced"]["result"]["final_state"]["seed"] == 17
                        })
            })
            .expect("native noop transition must be durably recorded");
        let noop_payload: Value = serde_json::from_str(&noop_transition.payload_json).unwrap();
        assert_eq!(noop_payload["reduced"]["result"]["workflow_status"], "succeeded");
        assert_eq!(noop_payload["reduced"]["result"]["final_state"]["counter"], 1);
        assert_eq!(noop_payload["reduced"]["result"]["final_state"]["seed"], 17);
        assert!(events.iter().any(|event| {
            event.event_type == "workflow.trace.v1"
                && event.payload_json.contains("runtime.native.step.completed")
        }));
        assert!(events.iter().any(|event| {
            event.event_type == "workflow.recorded_transition.v1"
                && event.payload_json.contains("increment-counter")
                && event.payload_json.contains("counter")
                && event.payload_json.contains(":2")
        }));
        let run = service
            .store
            .get_server_run("pg-native-run")
            .await
            .unwrap()
            .expect("native result must leave its run durably inspectable");
        assert_eq!(run.status, "succeeded");
        let result: Value = serde_json::from_str(
            run.result_json
                .as_deref()
                .expect("terminal native run must persist its result"),
        )
        .unwrap();
        assert_eq!(result["operation"], "emit-result");
        assert_eq!(result["schema_version"], 1);

        let before_restart_events = service
            .store
            .list_server_run_events("pg-native-run", 0, 100)
            .await
            .unwrap();
        let counter_transition = before_restart_events
            .iter()
            .find(|event| {
                if event.event_type != "workflow.recorded_transition.v1" {
                    return false;
                }
                serde_json::from_str::<Value>(&event.payload_json)
                    .map(|payload| payload["reduced"]["state"]["state"]["counter"] == 2)
                    .unwrap_or(false)
            })
            .expect("counter transition must be durably recorded before restart");
        let counter_payload: Value =
            serde_json::from_str(&counter_transition.payload_json).unwrap();
        assert_eq!(counter_payload["reduced"]["state"]["state"]["counter"], 2);
        let expected_state_digest = counter_payload["reduced"]["state_digest"]
            .as_str()
            .expect("recorded state must carry its canonical digest")
            .to_owned();

        drop(service);
        let reopened = PostgresRunApplicationService::from_dsn(&dsn, &schema).unwrap();
        reopened.ensure_schema().await.unwrap();
        let reopened_run = reopened
            .store
            .get_server_run("pg-native-run")
            .await
            .unwrap()
            .expect("completed native run must survive service reconstruction");
        assert_eq!(reopened_run.status, "succeeded");
        let reopened_result: Value = serde_json::from_str(
            reopened_run
                .result_json
                .as_deref()
                .expect("reopened run must retain its terminal result"),
        )
        .unwrap();
        assert_eq!(reopened_result["operation"], "emit-result");
        let reopened_events = reopened
            .store
            .list_server_run_events("pg-native-run", 0, 100)
            .await
            .unwrap();
        let reopened_counter = reopened_events
            .iter()
            .find(|event| {
                if event.event_type != "workflow.recorded_transition.v1" {
                    return false;
                }
                serde_json::from_str::<Value>(&event.payload_json)
                    .map(|payload| payload["reduced"]["state"]["state"]["counter"] == 2)
                    .unwrap_or(false)
            })
            .expect("counter transition must survive service reconstruction");
        let reopened_counter_payload: Value =
            serde_json::from_str(&reopened_counter.payload_json).unwrap();
        assert_eq!(
            reopened_counter_payload["reduced"]["state"]["state"]["counter"],
            2
        );
        assert_eq!(
            reopened_counter_payload["reduced"]["state_digest"],
            expected_state_digest
        );
        drop(reopened);
        let cleanup_dsn = dsn
            .strip_prefix("postgresql+psycopg://")
            .map(|rest| format!("postgresql://{rest}"))
            .or_else(|| {
                dsn.strip_prefix("postgresql+psycopg2://")
                    .map(|rest| format!("postgresql://{rest}"))
            })
            .unwrap_or(dsn);
        let (client, connection) = tokio_postgres::connect(&cleanup_dsn, tokio_postgres::NoTls)
            .await
            .unwrap();
        let connection_task = tokio::spawn(connection);
        client
            .batch_execute(&format!("DROP SCHEMA \"{schema}\" CASCADE"))
            .await
            .unwrap();
        drop(client);
        connection_task.await.unwrap().unwrap();
    }
