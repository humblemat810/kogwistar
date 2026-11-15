    #[tokio::test]
    #[ignore = "requires KOGWISTAR_TEST_PG_DSN; run explicitly with --include-ignored"]
    async fn postgres_design_history_folds_and_repairs_projection_when_dsn_available() {
        use kogwistar_store_postgres::NewRawEntityEvent;

        let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
            .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
        let schema = format!("rust_api_design_history_{}", uuid::Uuid::new_v4().simple());
        let service = PostgresRunApplicationService::from_dsn(&dsn, &schema).unwrap();
        service.ensure_schema().await.unwrap();
        for (event_id, op, payload) in [
            (
                "commit-1",
                "MUTATION_COMMITTED",
                json!({"version": 1, "prev_version": 0, "target_seq": 1, "ts_ms": 10, "designer_id": "alice", "action": "node_upsert", "entity_id": "n1"}),
            ),
            (
                "commit-2",
                "MUTATION_COMMITTED",
                json!({"version": 2, "prev_version": 1, "target_seq": 2, "ts_ms": 20, "designer_id": "alice", "action": "edge_upsert", "entity_id": "e2"}),
            ),
            (
                "undo-1",
                "UNDO_APPLIED",
                json!({"from_version": 2, "to_version": 1, "ts_ms": 30, "designer_id": "alice"}),
            ),
        ] {
            service
                .store
                .append_raw_entity_event(
                    "wf_design:wf-pg-history",
                    NewRawEntityEvent {
                        event_id: event_id.to_owned(),
                        entity_kind: "design_control".to_owned(),
                        entity_id: "wf-pg-history".to_owned(),
                        op: op.to_owned(),
                        payload_json: payload.to_string(),
                    },
                )
                .await
                .unwrap();
        }
        let response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: "/api/workflow/design/wf-pg-history/history".to_owned(),
                body: Vec::new(),
                principal: json!({
                    "sub": "alice",
                    "role": "ro",
                    "ns": "workflow",
                    "capabilities": ["workflow.design.inspect"],
                }),
            })
            .await;
        assert_eq!(response.status, StatusCode::OK.as_u16());
        let history = serde_json::from_slice::<Value>(&response.body).unwrap();
        assert_eq!(history["current_version"], 1);
        assert_eq!(history["active_tip_version"], 2);
        assert_eq!(history["can_redo"], true);
        assert_eq!(history["latest_seq"], 3);
        let projection = service
            .store
            .get_named_projection("workflow_design", "wf-pg-history")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(projection.payload["current_version"], 1);
        assert_eq!(projection.payload["active_tip_version"], 2);
        assert_eq!(projection.last_authoritative_seq, 3);
        assert_eq!(projection.last_materialized_seq, 1);
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
