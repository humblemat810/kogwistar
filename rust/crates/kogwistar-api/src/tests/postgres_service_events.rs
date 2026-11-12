    #[tokio::test]
    #[ignore = "requires KOGWISTAR_TEST_PG_DSN; run explicitly with --include-ignored"]
    async fn postgres_service_events_fold_authoritative_history_when_dsn_available() {
        use kogwistar_store::{NamedProjectionWrite, WorkflowDesignSnapshotWrite};
        use kogwistar_store_postgres::{GraphTableNames, NewRawEntityEvent};
        use rusqlite::Connection;

        let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
            .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
        let schema = format!("rust_api_service_events_{}", uuid::Uuid::new_v4().simple());
        let auth_path = std::env::temp_dir().join(format!(
            "kogwistar-service-trigger-auth-{}.sqlite",
            uuid::Uuid::new_v4().simple()
        ));
        let auth = SqliteAuthStore::open(&auth_path).unwrap();
        auth.resolve_external_identity(ResolveExternalIdentity {
            issuer: "test".to_owned(),
            subject: "operator".to_owned(),
            email: "operator@example.test".to_owned(),
            display_name: None,
            new_user_id: "operator".to_owned(),
            default_role: "rw".to_owned(),
            default_ns: "workflow".to_owned(),
        })
        .await
        .unwrap();
        drop(auth);
        let auth_config = AuthConfig {
            auth_db_url: Some(format!("sqlite:///{}", auth_path.display())),
            ..AuthConfig::default()
        };
        let mut native_operations = kogwistar_runtime::NativeOperationRegistry::default();
        native_operations
            .register_graph("app/service-owner-graph", "v1", |request| {
                let entity_id = format!("service-native-node-{}", request.run_id);
                let batch = serde_json::from_value(json!({
                    "contract_version": 1,
                    "atomic": true,
                    "mutations": [{
                        "scope": {"namespace": "workflow"},
                        "table": "gke_nodes",
                        "entity_kind": "node",
                        "event_id": format!("service-native-event-{}", request.run_id),
                        "op": "ADD",
                        "payload": {"id": entity_id},
                        "record": {"id": entity_id, "document": "service-triggered native graph"},
                        "embedding_dim": 0
                    }],
                    "index_jobs": [{
                        "job_id": format!("service-native-job-{}", request.run_id),
                        "namespace": "workflow",
                        "entity_kind": "node",
                        "entity_id": entity_id,
                        "index_kind": "node_embedding",
                        "op": "UPSERT"
                    }]
                }))
                .unwrap();
                Ok((
                    kogwistar_runtime::NativeOperationOutput {
                        state_update: Vec::new(),
                        result: Some(json!({"service_native_graph": true})),
                    },
                    batch,
                ))
            })
            .unwrap();
        let service = PostgresRunApplicationService::from_dsn(&dsn, &schema)
            .unwrap()
            .with_native_operations(native_operations)
            .with_owner_auth_config(&auth_config)
            .await
            .unwrap();
        service.ensure_schema().await.unwrap();
        service
            .store
            .create_graph_schema(3, GraphTableNames::default())
            .await
            .unwrap();
        service
            .store
            .replace_named_projection(
                "workflow_design",
                "wf-pg-trigger",
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
            .await
            .unwrap();
        service
            .store
            .put_workflow_design_snapshot(
                "wf-pg-trigger",
                WorkflowDesignSnapshotWrite {
                    version: 1,
                    seq: 1,
                    schema_version: 1,
                    payload_json: serde_json::to_string(&json!({
                        "nodes": [{
                            "id": "service-start",
                            "metadata": {
                                "wf_start": true,
                                "wf_op": "rust-native/app/service-owner-graph",
                                "wf_executor_target": "rust-native"
                            }
                        }],
                        "edges": []
                    }))
                    .unwrap(),
                },
            )
            .await
            .unwrap();
        for (event_id, entity_id, op, payload) in [
            (
                "service-event-1",
                "service_evt:svc-pg:1",
                "ADD",
                json!({
                    "id": "service_evt:svc-pg:1",
                    "metadata": {
                        "entity_type": "service_event",
                        "service_id": "svc-pg",
                        "service_event_type": "service.starting",
                        "ts_ms": 20,
                    },
                    "properties": {"payload_json": r#"{"phase":"old"}"#},
                }),
            ),
            (
                "service-event-2",
                "service_evt:svc-pg:1",
                "REPLACE",
                json!({
                    "id": "service_evt:svc-pg:1",
                    "metadata": {
                        "entity_type": "service_event",
                        "service_id": "svc-pg",
                        "service_event_type": "service.started",
                        "ts_ms": 20,
                    },
                    "properties": {"payload_json": r#"{"phase":"new"}"#},
                }),
            ),
            (
                "service-event-other",
                "service_evt:other:1",
                "ADD",
                json!({
                    "id": "service_evt:other:1",
                    "metadata": {
                        "entity_type": "service_event",
                        "service_id": "other",
                        "service_event_type": "service.healthy",
                        "ts_ms": 1,
                    },
                    "properties": {"payload_json": "{}"},
                }),
            ),
        ] {
            service
                .store
                .append_raw_entity_event(
                    "workflow",
                    NewRawEntityEvent {
                        event_id: event_id.to_owned(),
                        entity_kind: "node".to_owned(),
                        entity_id: entity_id.to_owned(),
                        op: op.to_owned(),
                        payload_json: payload.to_string(),
                    },
                )
                .await
                .unwrap();
        }
        let response = service
            .execute_async(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: "/api/workflow/services/svc-pg/events?limit=10".to_owned(),
                body: Vec::new(),
                principal: json!({
                    "sub": "alice",
                    "role": "ro",
                    "ns": "workflow",
                    "capabilities": ["service.inspect", "project_view"],
                }),
            })
            .await;
        assert_eq!(response.status, StatusCode::OK.as_u16());
        let body = serde_json::from_slice::<Value>(&response.body).unwrap();
        assert_eq!(body["service_id"], "svc-pg");
        assert_eq!(body["events"].as_array().unwrap().len(), 1);
        assert_eq!(body["events"][0]["event_type"], "service.started");
        assert_eq!(body["events"][0]["payload"]["phase"], "new");
        let principal = json!({
            "sub": "operator",
            "role": "rw",
            "ns": "workflow",
            "capabilities": ["service.manage", "service.inspect", "project_view", "write_graph"],
        });
        let declared = service
            .execute_async(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/workflow/services".to_owned(),
                body: serde_json::to_vec(&json!({
                    "service_id": "svc-pg-trigger",
                    "target_kind": "workflow",
                    "target_ref": "wf-pg-trigger",
                    "target_config": {
                        "conversation_id":"conv-pg-trigger",
                        "runtime_kind":"async",
                        "user_id":"operator"
                    },
                }))
                .unwrap(),
                principal: principal.clone(),
            })
            .await;
        assert_eq!(declared.status, StatusCode::OK.as_u16());
        let trigger = || ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/workflow/services/svc-pg-trigger/trigger".to_owned(),
            body: serde_json::to_vec(&json!({
                "trigger_type": "external event",
                "payload": {"source":"postgres-live"},
            }))
            .unwrap(),
            principal: principal.clone(),
        };
        let service_clone = service.clone();
        let (first, second) = tokio::join!(
            service.execute_async(trigger()),
            service_clone.execute_async(trigger()),
        );
        assert_eq!(first.status, StatusCode::OK.as_u16());
        assert_eq!(second.status, StatusCode::OK.as_u16());
        let first = serde_json::from_slice::<Value>(&first.body).unwrap();
        let second = serde_json::from_slice::<Value>(&second.body).unwrap();
        let run_id = first["current_child_run_id"].as_str().unwrap();
        assert_eq!(second["current_child_run_id"], run_id);
        let run = service
            .store
            .get_server_run(run_id)
            .await
            .unwrap()
            .expect("service trigger run");
        assert_eq!(run.user_id.as_deref(), Some("operator"));
        let run_events = service.store.server_run_events(run_id, 0, 10).await.unwrap();
        let created_payload: Value =
            serde_json::from_str(&run_events[0].payload_json).expect("run.created payload");
        assert_eq!(created_payload["runtime_owner_binding"]["user_id"], "operator");
        let lanes = service
            .store
            .list_projected_lane_messages(LaneMessageFilter {
                correlation_id: Some(run_id.to_owned()),
                limit: 10,
                ..LaneMessageFilter::default()
            })
            .await
            .unwrap();
        assert_eq!(lanes.len(), 1);
        let payload: Value = serde_json::from_str(
            lanes[0]
                .payload_json
                .as_deref()
                .expect("service lane payload"),
        )
        .unwrap();
        assert_eq!(payload["worker_protocol"], "async-v2");
        assert_eq!(payload["initial_state"]["_rt_worker_protocol"], "async-v2");
        assert!(service.execute_inline_native_work_once().await.unwrap());
        let entity_id = format!("service-native-node-{run_id}");
        let graph_rows = service
            .store
            .graph_projection_records(GraphProjectionRead {
                scope: GraphScope {
                    namespace: "workflow".to_owned(),
                    workspace_id: None,
                    graph_space: None,
                },
                table: "gke_nodes".to_owned(),
                ids: Some(vec![entity_id.clone()]),
                metadata: MetadataFilter::default(),
                limit: 10,
            })
            .await
            .unwrap();
        assert_eq!(graph_rows.len(), 1);
        assert_eq!(graph_rows[0].id, entity_id);
        let graph_events = service
            .store
            .replay_raw_events("workflow", 0, 100)
            .await
            .unwrap();
        assert!(graph_events
            .iter()
            .any(|event| event.event_id == format!("service-native-event-{run_id}")));
        let index_jobs = service
            .store
            .list_index_jobs(Some("workflow"), None, None, None, None, 100)
            .await
            .unwrap();
        assert!(index_jobs.iter().any(|job| {
            job.job_id == format!("service-native-job-{run_id}")
                && job.entity_id == entity_id
                && job.status == "PENDING"
        }));
        assert_eq!(
            service
                .store
                .get_server_run(run_id)
                .await
                .unwrap()
                .unwrap()
                .status,
            "succeeded"
        );
        let completed_lane = service
            .store
            .list_projected_lane_messages(LaneMessageFilter {
                correlation_id: Some(run_id.to_owned()),
                limit: 10,
                ..LaneMessageFilter::default()
            })
            .await
            .unwrap();
        assert_eq!(completed_lane.len(), 1);
        assert_eq!(completed_lane[0].status, "completed");
        assert!(!service.execute_inline_native_work_once().await.unwrap());

        let auth_connection = Connection::open(&auth_path).unwrap();
        for (case, column, changed_value, restored_value) in [
            ("deactivated", "is_active", "0", "1"),
            ("role-downgraded", "global_role", "ro", "rw"),
            ("namespace-moved", "global_ns", "other", "workflow"),
        ] {
            let service_id = format!("svc-pg-trigger-{case}");
            let conversation_id = format!("conv-pg-trigger-{case}");
            let declared = service
                .execute_async(ApiEffectRequest {
                    contract_version: 1,
                    method: "POST".to_owned(),
                    path_and_query: "/api/workflow/services".to_owned(),
                    body: serde_json::to_vec(&json!({
                        "service_id": service_id,
                        "target_kind": "workflow",
                        "target_ref": "wf-pg-trigger",
                        "target_config": {
                            "conversation_id": conversation_id,
                            "runtime_kind": "async",
                            "user_id": "operator"
                        }
                    }))
                    .unwrap(),
                    principal: principal.clone(),
                })
                .await;
            assert_eq!(declared.status, StatusCode::OK.as_u16(), "{case}");
            let triggered = service
                .execute_async(ApiEffectRequest {
                    contract_version: 1,
                    method: "POST".to_owned(),
                    path_and_query: format!("/api/workflow/services/{service_id}/trigger"),
                    body: serde_json::to_vec(&json!({
                        "trigger_type": "external event",
                        "payload": {"source": format!("revalidate-{case}")}
                    }))
                    .unwrap(),
                    principal: principal.clone(),
                })
                .await;
            assert_eq!(triggered.status, StatusCode::OK.as_u16(), "{case}");
            let triggered: Value = serde_json::from_slice(&triggered.body).unwrap();
            let run_id = triggered["current_child_run_id"]
                .as_str()
                .expect("service trigger admitted a child run")
                .to_owned();

            let changed = auth_connection
                .execute(
                    &format!("UPDATE users SET {column} = ?1 WHERE user_id = ?2"),
                    [changed_value, "operator"],
                )
                .unwrap();
            assert_eq!(changed, 1, "test must apply the {case} change");

            assert!(service.execute_inline_native_work_once().await.unwrap());
            let entity_id = format!("service-native-node-{run_id}");
            let graph_rows = service
                .store
                .graph_projection_records(GraphProjectionRead {
                    scope: GraphScope {
                        namespace: "workflow".to_owned(),
                        workspace_id: None,
                        graph_space: None,
                    },
                    table: "gke_nodes".to_owned(),
                    ids: Some(vec![entity_id]),
                    metadata: MetadataFilter::default(),
                    limit: 10,
                })
                .await
                .unwrap();
            assert!(graph_rows.is_empty(), "{case} owner must not write graph data");
            let events = service
                .store
                .replay_raw_events("workflow", 0, 100)
                .await
                .unwrap();
            assert!(!events
                .iter()
                .any(|event| event.event_id == format!("service-native-event-{run_id}")));
            let jobs = service
                .store
                .list_index_jobs(Some("workflow"), None, None, None, None, 100)
                .await
                .unwrap();
            assert!(!jobs
                .iter()
                .any(|job| job.job_id == format!("service-native-job-{run_id}")));
            assert_eq!(
                service
                    .store
                    .get_server_run(&run_id)
                    .await
                    .unwrap()
                    .unwrap()
                    .status,
                "failed",
                "{case} denial must be recorded as a durable failed run"
            );
            let lane = service
                .store
                .list_projected_lane_messages(LaneMessageFilter {
                    correlation_id: Some(run_id),
                    limit: 10,
                    ..LaneMessageFilter::default()
                })
                .await
                .unwrap();
            assert_eq!(lane.len(), 1, "{case}");
            assert_eq!(lane[0].status, "completed", "{case}");

            let restored = auth_connection
                .execute(
                    &format!("UPDATE users SET {column} = ?1 WHERE user_id = ?2"),
                    [restored_value, "operator"],
                )
                .unwrap();
            assert_eq!(restored, 1, "restore auth state after {case}");
        }

        let spoof_service = service
            .execute_async(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/workflow/services".to_owned(),
                body: serde_json::to_vec(&json!({
                    "service_id": "svc-pg-trigger-spoof",
                    "target_kind": "workflow",
                    "target_ref": "wf-pg-trigger",
                    "target_config": {
                        "conversation_id": "conv-pg-trigger-spoof",
                        "user_id": "victim"
                    }
                }))
                .unwrap(),
                principal: principal.clone(),
            })
            .await;
        assert_eq!(spoof_service.status, StatusCode::OK.as_u16());
        let spoof_trigger = service
            .execute_async(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/workflow/services/svc-pg-trigger-spoof/trigger".to_owned(),
                body: serde_json::to_vec(&json!({
                    "trigger_type": "external event",
                    "payload": {}
                }))
                .unwrap(),
                principal: principal.clone(),
            })
            .await;
        assert_eq!(spoof_trigger.status, StatusCode::FORBIDDEN.as_u16());
        let spoof_error: Value = serde_json::from_slice(&spoof_trigger.body).unwrap();
        assert_eq!(spoof_error["code"], "KOGWISTAR_RUN_OWNER_FORBIDDEN");
        assert!(service
            .store
            .list_projected_lane_messages(LaneMessageFilter {
                conversation_id: Some("conv-pg-trigger-spoof".to_owned()),
                limit: 10,
                ..LaneMessageFilter::default()
            })
            .await
            .unwrap()
            .is_empty());
        drop(service);
        let _ = std::fs::remove_file(auth_path);

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
