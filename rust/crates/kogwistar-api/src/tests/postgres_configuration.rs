    #[test]
    fn postgres_multi_host_dsn_remains_constructible_without_embedding_provider() {
        let service = PostgresRunApplicationService::from_dsn(
            "postgresql://writer@db-a.example,db-b.example:5544/graph",
            "tenant",
        )
        .expect("the opt-in embedding profile guard must not narrow legacy DSN support");
        assert!(service.embedding_provider.is_none());
        assert!(service.embedding_storage_scope.is_err());
    }

    #[tokio::test]
    #[ignore = "requires KOGWISTAR_TEST_PG_DSN; run explicitly with --include-ignored"]
    async fn postgres_design_capability_mutates_and_undoes_when_dsn_available() {
        let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
            .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
        let schema = format!("rust_api_design_write_{}", uuid::Uuid::new_v4().simple());
        let service = PostgresRunApplicationService::from_dsn(&dsn, &schema).unwrap();
        service.ensure_schema().await.unwrap();
        let principal = json!({"sub":"alice","role":"rw","ns":"workflow","capabilities":["workflow.design.write","workflow.design.inspect"]});
        for body in [
            json!({"designer_id":"alice","node_id":"start","label":"Start","op":"increment-counter","start":true,"metadata":{"wf_executor_target":"rust-native"}}),
            json!({"designer_id":"alice","node_id":"end","label":"End","op":"emit-result","terminal":true,"metadata":{"wf_executor_target":"rust-native"}}),
        ] {
            let response = service
                .execute_async(ApiEffectRequest {
                    contract_version: 1,
                    method: "POST".to_owned(),
                    path_and_query: "/api/workflow/design/wf-pg-write/nodes".to_owned(),
                    body: serde_json::to_vec(&body).unwrap(),
                    principal: principal.clone(),
                })
                .await;
            assert_eq!(response.status, 200);
        }
        let edge = service
            .execute_async(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/workflow/design/wf-pg-write/edges".to_owned(),
                body: serde_json::to_vec(
                    &json!({"designer_id":"alice","edge_id":"edge","src":"start","dst":"end"}),
                )
                .unwrap(),
                principal: principal.clone(),
            })
            .await;
        assert_eq!(edge.status, 200);
        let snapshot_reader = PostgresStore::from_dsn(&dsn, &schema).unwrap();
        let current_snapshot = snapshot_reader
            .get_workflow_design_snapshot("wf-pg-write", 3, 1)
            .await
            .unwrap()
            .expect("PostgreSQL design mutation publishes an exact current snapshot");
        assert_eq!(current_snapshot.version, 3);
        assert!(runtime_graph_plan_from_snapshot(&current_snapshot.payload_json).is_some());
        let request = |method: &str, path: &str, body: Value| ApiEffectRequest {
            contract_version: 1,
            method: method.to_owned(),
            path_and_query: path.to_owned(),
            body: serde_json::to_vec(&body).unwrap(),
            principal: principal.clone(),
        };
        let submit = service
            .execute_async(request(
                "POST",
                "/api/workflow/runs",
                json!({
                    "run_id": "run-from-pg-design",
                    "workflow_id": "wf-pg-write",
                    "conversation_id": "conversation-from-pg-design",
                    "initial_state": {"counter": 1}
                }),
            ))
            .await;
        assert_eq!(submit.status, StatusCode::ACCEPTED);
        let first_claim = service
            .execute_async(request(
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
        assert_eq!(first_claim.status, StatusCode::OK);
        let first_claim: Value = serde_json::from_slice(&first_claim.body).unwrap();
        let first_work = &first_claim["work"][0];
        assert_eq!(first_work["step_id"], "start");
        assert_eq!(first_work["payload"]["op"], "increment-counter");
        let native_execute_request = |work: &Value| {
            request(
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
            )
        };
        let first_execute = service
            .execute_async(native_execute_request(first_work))
            .await;
        assert_eq!(first_execute.status, StatusCode::OK);
        let second_claim = service
            .execute_async(request(
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
        assert_eq!(second_claim.status, StatusCode::OK);
        let second_claim: Value = serde_json::from_slice(&second_claim.body).unwrap();
        let second_work = &second_claim["work"][0];
        assert_eq!(second_work["step_id"], "end");
        assert_eq!(second_work["payload"]["op"], "emit-result");
        let second_execute = service
            .execute_async(native_execute_request(second_work))
            .await;
        assert_eq!(second_execute.status, StatusCode::OK);
        let completed = snapshot_reader
            .get_server_run("run-from-pg-design")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(completed.status, "succeeded");
        let run_events = snapshot_reader
            .list_server_run_events("run-from-pg-design", 0, 100)
            .await
            .unwrap();
        assert!(run_events.iter().any(|event| {
            event.event_type == "workflow.recorded_transition.v1"
                && event.payload_json.contains("increment-counter")
                && event.payload_json.contains(":2")
        }));
        assert!(run_events.iter().any(|event| {
            event.event_type == "workflow.trace.v1"
                && event.payload_json.contains("runtime.native.step.completed")
                && event.payload_json.contains("\"seam_crossings\":[]")
        }));
        let undo = service
            .execute_async(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/workflow/design/wf-pg-write/undo".to_owned(),
                body: serde_json::to_vec(&json!({"designer_id":"alice"})).unwrap(),
                principal: principal.clone(),
            })
            .await;
        assert_eq!(undo.status, 200);
        assert!(snapshot_reader
            .get_workflow_design_snapshot("wf-pg-write", 2, 1)
            .await
            .unwrap()
            .is_some());
        let graph = service
            .execute_async(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: "/api/workflow/design/wf-pg-write/graph".to_owned(),
                body: Vec::new(),
                principal,
            })
            .await;
        let graph: Value = serde_json::from_slice(&graph.body).unwrap();
        assert!(graph["edges"].as_array().unwrap().is_empty());
        drop(service);
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
            .batch_execute(&format!("DROP SCHEMA {schema} CASCADE"))
            .await
            .unwrap();
        drop(client);
        connection_task.await.unwrap().unwrap();
    }
