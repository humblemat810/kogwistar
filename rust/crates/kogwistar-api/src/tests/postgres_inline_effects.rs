    #[tokio::test]
    #[ignore = "requires KOGWISTAR_TEST_PG_DSN; run explicitly against disposable PostgreSQL"]
    async fn postgres_inline_native_graph_effect_revalidates_current_owner() {
        let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN").unwrap();
        let schema = format!("rust_inline_owner_{}", uuid::Uuid::new_v4().simple());
        let auth_path = std::env::temp_dir().join(format!(
            "kogwistar-inline-auth-{}.sqlite",
            uuid::Uuid::new_v4().simple()
        ));
        let auth = SqliteAuthStore::open(&auth_path).unwrap();
        auth.resolve_external_identity(ResolveExternalIdentity {
            issuer: "test".to_owned(),
            subject: "alice".to_owned(),
            email: "alice@example.test".to_owned(),
            display_name: None,
            new_user_id: "alice".to_owned(),
            default_role: "rw".to_owned(),
            default_ns: "alpha".to_owned(),
        })
        .await
        .unwrap();
        drop(auth);

        let mut registry = kogwistar_runtime::NativeOperationRegistry::default();
        registry
            .register_graph("app/inline-graph", "v1", |request| {
                let entity_id = format!("inline-node-{}", request.run_id);
                let namespace = if request.run_id == "inline-cross-scope" {
                    "beta"
                } else {
                    "alpha"
                };
                let batch = serde_json::from_value(json!({
                    "contract_version": 1,
                    "atomic": true,
                    "mutations": [{
                        "scope": {"namespace": namespace},
                        "table": "gke_nodes",
                        "entity_kind": "node",
                        "event_id": format!("inline-event-{}", request.run_id),
                        "op": "ADD",
                        "payload": {"id": entity_id},
                        "record": {"id": entity_id, "document": "inline native graph"},
                        "embedding_dim": 0
                    }],
                    "index_jobs": [{
                        "job_id": format!("inline-job-{}", request.run_id),
                        "namespace": namespace,
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
                        result: Some(json!({"native_graph": true})),
                    },
                    batch,
                ))
            })
            .unwrap();
        registry
            .register_graph_v2("app/inline-graph-v2", "v1", |request| {
                let entity_id = "inline-v2-node-stable";
                let batch = serde_json::from_value(json!({
                    "contract_version": 2,
                    "atomic": true,
                    "mutations": [{
                        "scope": {"namespace": "alpha"},
                        "table": "gke_nodes",
                        "entity_kind": "node",
                        "event_id": format!("inline-v2-event-{}", request.run_id),
                        "op": "ADD",
                        "payload": {
                            "id": entity_id,
                            "label": "native ACL entity",
                            "type": "entity",
                            "summary": "native ACL entity",
                            "doc_id": "native-acl-doc",
                            "mentions": [],
                            "metadata": {},
                            "properties": {},
                            "embedding": null
                        },
                        "record": {"id": entity_id, "document": "native ACL entity"},
                        "embedding_dim": 0
                    }],
                    "default_acl": {"truth_graph": "conversation"}
                }))
                .unwrap();
                Ok((
                    kogwistar_runtime::NativeOperationOutput {
                        state_update: Vec::new(),
                        result: Some(json!({"native_acl_graph": true})),
                    },
                    Some(batch),
                ))
            })
            .unwrap();
        let auth_config = AuthConfig {
            auth_db_url: Some(format!("sqlite:///{}", auth_path.display())),
            ..AuthConfig::default()
        };
        let mut service = PostgresRunApplicationService::from_dsn(&dsn, &schema)
            .unwrap()
            .with_native_operations(registry.clone())
            .with_owner_auth_config(&auth_config)
            .await
            .unwrap();
        service.ensure_graph_type_schemas().await.unwrap();
        service
            .store
            .create_graph_schema(3, kogwistar_store_postgres::GraphTableNames::default())
            .await
            .unwrap();
        service
            .conversation_store
            .create_graph_schema(3, kogwistar_store_postgres::GraphTableNames::default())
            .await
            .unwrap();

        let submit = |run_id: &str| ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/workflow/runs".to_owned(),
            body: serde_json::to_vec(&json!({
                "run_id": run_id,
                "workflow_id": "wf",
                "conversation_id": format!("conv-{run_id}"),
                "start_node_id": "entry",
                "node_ops": {"entry": "rust-native/app/inline-graph"}
            }))
            .unwrap(),
            principal: json!({
                "user_id": "alice",
                "sub": "alice",
                "role": "rw",
                "ns": "alpha",
                "storage_ns": "alpha",
                "capabilities": ["spawn_process", "workflow.run.write", "write_graph"]
            }),
        };
        let submit_v2 = |run_id: &str| {
            let mut request = submit(run_id);
            let mut payload: Value = serde_json::from_slice(&request.body).unwrap();
            payload["node_ops"]["entry"] = json!("rust-native/app/inline-graph-v2");
            request.body = serde_json::to_vec(&payload).unwrap();
            request
        };
        assert_eq!(
            service.execute(submit("inline-authorized")).await.status,
            StatusCode::ACCEPTED.as_u16()
        );
        tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
        assert!(service.execute_inline_native_work_once().await.unwrap());
        let authorized = service
            .store
            .graph_projection_records(GraphProjectionRead {
                scope: GraphScope {
                    namespace: "alpha".to_owned(),
                    workspace_id: None,
                    graph_space: None,
                },
                table: "gke_nodes".to_owned(),
                ids: Some(vec!["inline-node-inline-authorized".to_owned()]),
                metadata: MetadataFilter::default(),
                limit: 10,
            })
            .await
            .unwrap();
        assert_eq!(authorized.len(), 1);

        assert_eq!(
            service.execute(submit_v2("inline-acl-authorized")).await.status,
            StatusCode::ACCEPTED.as_u16()
        );
        tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
        assert!(service.execute_inline_native_work_once().await.unwrap());
        let acl_target_id = "inline-v2-node-stable".to_owned();
        let acl_records = service
            .conversation_store
            .graph_acl_records_for_entities(
                &GraphScope {
                    namespace: "alpha".to_owned(),
                    workspace_id: None,
                    graph_space: None,
                },
                "conversation",
                std::slice::from_ref(&acl_target_id),
            )
            .await
            .unwrap();
        assert_eq!(acl_records.len(), 1);
        assert_eq!(
            acl_records[0].metadata.get("acl_mode").and_then(Value::as_str),
            Some("private")
        );
        assert_eq!(
            acl_records[0].metadata.get("owner_id").and_then(Value::as_str),
            Some("alice")
        );
        let acl_jobs = service
            .conversation_store
            .list_index_jobs(Some("alpha"), None, None, None, None, 100)
            .await
            .unwrap();
        assert!(acl_jobs.iter().any(|job| job.entity_id == acl_records[0].id));

        assert_eq!(
            service.execute(submit_v2("inline-acl-repeat")).await.status,
            StatusCode::ACCEPTED.as_u16()
        );
        tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
        assert!(service.execute_inline_native_work_once().await.unwrap());
        let replayed_acl_records = service
            .conversation_store
            .graph_acl_records_for_entities(
                &GraphScope {
                    namespace: "alpha".to_owned(),
                    workspace_id: None,
                    graph_space: None,
                },
                "conversation",
                std::slice::from_ref(&acl_target_id),
            )
            .await
            .unwrap();
        assert_eq!(
            replayed_acl_records.len(),
            1,
            "matching ACL defaults must not mint a new version"
        );

        assert_eq!(
            service.execute(submit("inline-cross-scope")).await.status,
            StatusCode::ACCEPTED.as_u16()
        );
        tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
        assert!(service.execute_inline_native_work_once().await.unwrap());
        assert_eq!(
            service
                .store
                .get_server_run("inline-cross-scope")
                .await
                .unwrap()
                .unwrap()
                .status,
            "failed",
            "cross-scope native graph mutation must fail the run"
        );
        let cross_scope_lane = service
            .store
            .list_projected_lane_messages(LaneMessageFilter {
                correlation_id: Some("inline-cross-scope".to_owned()),
                limit: 10,
                ..LaneMessageFilter::default()
            })
            .await
            .unwrap();
        assert_eq!(cross_scope_lane.len(), 1);
        assert_eq!(cross_scope_lane[0].status, "completed");
        for namespace in ["alpha", "beta"] {
            let rows = service
                .store
                .graph_projection_records(GraphProjectionRead {
                    scope: GraphScope {
                        namespace: namespace.to_owned(),
                        workspace_id: None,
                        graph_space: None,
                    },
                    table: "gke_nodes".to_owned(),
                    ids: Some(vec!["inline-node-inline-cross-scope".to_owned()]),
                    metadata: MetadataFilter::default(),
                    limit: 10,
                })
                .await
                .unwrap();
            assert!(rows.is_empty(), "cross-scope write leaked into {namespace}");
            let jobs = service
                .store
                .list_index_jobs(Some(namespace), None, None, None, None, 10)
                .await
                .unwrap();
            assert!(!jobs
                .iter()
                .any(|job| job.job_id == "inline-job-inline-cross-scope"));
        }
        for namespace in ["alpha", "beta"] {
            let events = service
                .store
                .replay_raw_events(namespace, 0, 100)
                .await
                .unwrap()
                ;
            assert!(!events
                .iter()
                .any(|event| event.event_id == "inline-event-inline-cross-scope"));
        }

        let (fault_setup, connection) =
            tokio_postgres::connect(&dsn, tokio_postgres::NoTls).await.unwrap();
        let task = tokio::spawn(connection);
        fault_setup
            .batch_execute(&format!(
                "CREATE FUNCTION \"{schema}\".reject_inline_index_job() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'injected inline index-job failure'; END; $$; CREATE TRIGGER reject_inline_index_job BEFORE INSERT ON \"{schema}\".index_jobs FOR EACH ROW EXECUTE FUNCTION \"{schema}\".reject_inline_index_job();"
            ))
            .await
            .unwrap();
        drop(fault_setup);
        task.await.unwrap().unwrap();
        assert_eq!(
            service.execute(submit("inline-transaction-failure")).await.status,
            StatusCode::ACCEPTED.as_u16()
        );
        tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
        assert!(service
            .execute_inline_native_work_once()
            .await
            .is_err());
        let failure_events = service
            .store
            .replay_raw_events("alpha", 0, 100)
            .await
            .unwrap();
        assert!(!failure_events
            .iter()
            .any(|event| event.event_id == "inline-event-inline-transaction-failure"));
        let failure_rows = service
            .store
            .graph_projection_records(GraphProjectionRead {
                scope: GraphScope {
                    namespace: "alpha".to_owned(),
                    workspace_id: None,
                    graph_space: None,
                },
                table: "gke_nodes".to_owned(),
                ids: Some(vec!["inline-node-inline-transaction-failure".to_owned()]),
                metadata: MetadataFilter::default(),
                limit: 10,
            })
            .await
            .unwrap();
        assert!(failure_rows.is_empty());
        let failure_jobs = service
            .store
            .list_index_jobs(Some("alpha"), None, None, None, None, 10)
            .await
            .unwrap();
        assert!(!failure_jobs
            .iter()
            .any(|job| job.job_id == "inline-job-inline-transaction-failure"));
        let failure_lane = service
            .store
            .list_projected_lane_messages(LaneMessageFilter {
                correlation_id: Some("inline-transaction-failure".to_owned()),
                limit: 10,
                ..LaneMessageFilter::default()
            })
            .await
            .unwrap();
        assert_eq!(failure_lane.len(), 1);
        assert_ne!(failure_lane[0].status, "completed");
        let (fault_cleanup, connection) =
            tokio_postgres::connect(&dsn, tokio_postgres::NoTls).await.unwrap();
        let task = tokio::spawn(connection);
        fault_cleanup
            .batch_execute(&format!(
                "DROP TRIGGER reject_inline_index_job ON \"{schema}\".index_jobs; DROP FUNCTION \"{schema}\".reject_inline_index_job();"
            ))
            .await
            .unwrap();
        drop(fault_cleanup);
        task.await.unwrap().unwrap();

        drop(service);
        let (reclaim, connection) =
            tokio_postgres::connect(&dsn, tokio_postgres::NoTls).await.unwrap();
        let task = tokio::spawn(connection);
        let expired = reclaim
            .execute(
                &format!(
                    "UPDATE \"{schema}\".projected_lane_messages SET lease_until = NOW() - INTERVAL '1 second' WHERE correlation_id = $1 AND status = 'claimed'"
                ),
                &[&"inline-transaction-failure"],
            )
            .await
            .unwrap();
        assert_eq!(expired, 1, "the failed transaction must leave a reclaimable lane");
        drop(reclaim);
        task.await.unwrap().unwrap();
        service = PostgresRunApplicationService::from_dsn(&dsn, &schema)
            .unwrap()
            .with_native_operations(registry.clone())
            .with_owner_auth_config(&auth_config)
            .await
            .unwrap();
        assert!(service.execute_inline_native_work_once().await.unwrap());
        assert_eq!(
            service
                .store
                .get_server_run("inline-transaction-failure")
                .await
                .unwrap()
                .unwrap()
                .status,
            "succeeded"
        );
        let recovered_lane = service
            .store
            .list_projected_lane_messages(LaneMessageFilter {
                correlation_id: Some("inline-transaction-failure".to_owned()),
                limit: 10,
                ..LaneMessageFilter::default()
            })
            .await
            .unwrap();
        assert_eq!(recovered_lane.len(), 1);
        assert_eq!(recovered_lane[0].status, "completed");
        let recovered_events = service
            .store
            .replay_raw_events("alpha", 0, 100)
            .await
            .unwrap();
        assert_eq!(
            recovered_events
                .iter()
                .filter(|event| event.event_id == "inline-event-inline-transaction-failure")
                .count(),
            1
        );
        let recovered_rows = service
            .store
            .graph_projection_records(GraphProjectionRead {
                scope: GraphScope {
                    namespace: "alpha".to_owned(),
                    workspace_id: None,
                    graph_space: None,
                },
                table: "gke_nodes".to_owned(),
                ids: Some(vec!["inline-node-inline-transaction-failure".to_owned()]),
                metadata: MetadataFilter::default(),
                limit: 10,
            })
            .await
            .unwrap();
        assert_eq!(recovered_rows.len(), 1);
        let recovered_jobs = service
            .store
            .list_index_jobs(Some("alpha"), None, None, None, None, 10)
            .await
            .unwrap();
        assert_eq!(
            recovered_jobs
                .iter()
                .filter(|job| job.job_id == "inline-job-inline-transaction-failure")
                .count(),
            1
        );

        for (run_id, binding_update) in [
            (
                "inline-binding-missing",
                "payload_json = (payload_json::jsonb - 'runtime_owner_binding')::text",
            ),
            (
                "inline-binding-mismatched",
                "payload_json = jsonb_set(payload_json::jsonb, '{runtime_owner_binding,user_id}', '\"mallory\"'::jsonb)::text",
            ),
        ] {
            assert_eq!(
                service.execute(submit(run_id)).await.status,
                StatusCode::ACCEPTED.as_u16()
            );
            let (tamper, connection) =
                tokio_postgres::connect(&dsn, tokio_postgres::NoTls).await.unwrap();
            let task = tokio::spawn(connection);
            let updated = tamper
                .execute(
                    &format!(
                        "UPDATE \"{schema}\".server_run_events SET {binding_update} WHERE run_id = $1 AND event_type = 'run.created'"
                    ),
                    &[&run_id],
                )
                .await
                .unwrap();
            assert_eq!(updated, 1, "expected to tamper with the admission event");
            drop(tamper);
            task.await.unwrap().unwrap();

            tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
            assert!(service.execute_inline_native_work_once().await.unwrap());
            assert_eq!(
                service
                    .store
                    .get_server_run(run_id)
                    .await
                    .unwrap()
                    .unwrap()
                    .status,
                "failed",
                "an absent or untrusted owner binding must durably fail the run"
            );
            let lane = service
                .store
                .list_projected_lane_messages(LaneMessageFilter {
                    correlation_id: Some(run_id.to_owned()),
                    limit: 10,
                    ..LaneMessageFilter::default()
                })
                .await
                .unwrap();
            assert_eq!(lane.len(), 1);
            assert_eq!(lane[0].status, "completed");
            let events = service
                .store
                .replay_raw_events("alpha", 0, 100)
                .await
                .unwrap();
            assert!(!events.iter().any(|event| {
                event.event_id == format!("inline-event-{run_id}")
            }));
            let rows = service
                .store
                .graph_projection_records(GraphProjectionRead {
                    scope: GraphScope {
                        namespace: "alpha".to_owned(),
                        workspace_id: None,
                        graph_space: None,
                    },
                    table: "gke_nodes".to_owned(),
                    ids: Some(vec![format!("inline-node-{run_id}")]),
                    metadata: MetadataFilter::default(),
                    limit: 10,
                })
                .await
                .unwrap();
            assert!(rows.is_empty());
            let jobs = service
                .store
                .list_index_jobs(Some("alpha"), None, None, None, None, 10)
                .await
                .unwrap();
            assert!(!jobs
                .iter()
                .any(|job| job.job_id == format!("inline-job-{run_id}")));
        }

        assert_eq!(
            service.execute(submit("inline-revoked")).await.status,
            StatusCode::ACCEPTED.as_u16()
        );
        tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
        service
            .capabilities
            .lock()
            .unwrap()
            .revoked
            .entry("alice@example.test".to_owned())
            .or_default()
            .insert("write_graph".to_owned());
        assert!(service.execute_inline_native_work_once().await.unwrap());
        let denied = service
            .store
            .graph_projection_records(GraphProjectionRead {
                scope: GraphScope {
                    namespace: "alpha".to_owned(),
                    workspace_id: None,
                    graph_space: None,
                },
                table: "gke_nodes".to_owned(),
                ids: Some(vec!["inline-node-inline-revoked".to_owned()]),
                metadata: MetadataFilter::default(),
                limit: 10,
            })
            .await
            .unwrap();
        assert!(denied.is_empty());
        assert_eq!(
            service
                .store
                .get_server_run("inline-revoked")
                .await
                .unwrap()
                .unwrap()
                .status,
            "failed",
            "deterministic authorization denial must be acknowledged as a failed step"
        );
        drop(service);
        let _ = std::fs::remove_file(auth_path);
        let (cleanup, connection) = tokio_postgres::connect(&dsn, tokio_postgres::NoTls)
            .await
            .unwrap();
        let task = tokio::spawn(connection);
        cleanup
            .batch_execute(&format!(
                "DROP SCHEMA IF EXISTS \"{schema}_conversation\" CASCADE; \
                 DROP SCHEMA IF EXISTS \"{schema}_workflow\" CASCADE; \
                 DROP SCHEMA IF EXISTS \"{schema}_wisdom\" CASCADE; \
                 DROP SCHEMA IF EXISTS \"{schema}\" CASCADE"
            ))
            .await
            .unwrap();
        drop(cleanup);
        task.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn postgres_worker_graph_effect_requires_write_graph_before_storage() {
        let service = PostgresRunApplicationService::from_dsn(
            "postgresql://worker:secret@127.0.0.1:1/unreachable",
            "worker_effect_authority",
        )
        .unwrap();
        let response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/internal/runtime/results".to_owned(),
                body: serde_json::to_vec(&json!({
                    "handoff": {
                        "message_id": "lane-1",
                        "claimed_by": "worker-1",
                        "run_id": "run-1",
                        "step_id": "step-1",
                        "correlation_id": "run-1"
                    },
                    "effect": {
                        "contract_version": 1,
                        "effect_id": "effect-1",
                        "graph_mutation_batch": {
                            "contract_version": 1,
                            "atomic": true,
                            "mutations": [{
                                "scope": {"namespace": "alpha"},
                                "table": "gke_nodes",
                                "entity_kind": "node",
                                "event_id": "event-1",
                                "op": "ADD",
                                "payload": {"id": "node-1"},
                                "record": {"id": "node-1"},
                                "embedding_dim": 0
                            }]
                        }
                    }
                }))
                .unwrap(),
                principal: json!({
                    "sub": "read-only-worker",
                    "role": "reader",
                    "storage_ns": "alpha",
                    "capabilities": ["read_graph"]
                }),
            })
            .await;
        assert_eq!(response.status, StatusCode::FORBIDDEN.as_u16());
        let body: Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(body["code"], "KOGWISTAR_CAPABILITY_FORBIDDEN");
    }
