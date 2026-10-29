    #[test]
    fn sqlite_chroma_stage2_read_fence_requires_matching_active_event_revision() {
        let event = SqliteRawEntityEvent {
            namespace: "tenant-a".to_owned(),
            seq: 7,
            event_id: "node-7".to_owned(),
            entity_kind: "node".to_owned(),
            entity_id: "n1".to_owned(),
            op: "REPLACE".to_owned(),
            payload_json: r#"{"lifecycle_patch":{"lifecycle_status":"active"}}"#.to_owned(),
            created_at: 0,
        };
        let stage2_record = |ready: Value, revision: Option<Value>| {
            let mut metadata = serde_json::Map::new();
            metadata.insert("_kogwistar_stage2_ready".to_owned(), ready);
            if let Some(revision) = revision {
                metadata.insert("_kogwistar_source_revision".to_owned(), revision);
            }
            GraphRecord {
                id: "n1".to_owned(),
                document: Some("node payload".to_owned()),
                metadata,
                embedding: None,
            }
        };

        assert!(SqliteRunApplicationService::stage2_record_is_current(
            &stage2_record(Value::Bool(true), Some(json!(7))),
            Some(&event),
        ));
        assert!(!SqliteRunApplicationService::stage2_record_is_current(
            &stage2_record(Value::Bool(true), Some(json!(6))),
            Some(&event),
        ));
        assert!(!SqliteRunApplicationService::stage2_record_is_current(
            &stage2_record(Value::Bool(true), None),
            Some(&event),
        ));
        assert!(!SqliteRunApplicationService::stage2_record_is_current(
            &stage2_record(Value::Bool(true), Some(json!(7))),
            None,
        ));
        assert!(!SqliteRunApplicationService::stage2_record_is_current(
            &stage2_record(Value::Bool(true), Some(Value::Bool(true))),
            Some(&event),
        ));
        assert!(!SqliteRunApplicationService::stage2_record_is_current(
            &stage2_record(Value::Bool(false), Some(json!(7))),
            Some(&event),
        ));

        let deleted = SqliteRawEntityEvent {
            op: "DELETE".to_owned(),
            ..event.clone()
        };
        assert!(!SqliteRunApplicationService::stage2_record_is_current(
            &stage2_record(Value::Bool(true), Some(json!(7))),
            Some(&deleted),
        ));
        let legacy_single_stage = GraphRecord {
            id: "legacy".to_owned(),
            document: None,
            metadata: serde_json::Map::new(),
            embedding: None,
        };
        assert!(SqliteRunApplicationService::stage2_record_is_current(
            &legacy_single_stage,
            None,
        ));
    }

    #[test]
    fn authenticated_run_owner_uses_stable_user_id_and_rejects_anonymous() {
        assert_eq!(
            authenticated_run_owner_id(&json!({"user_id":"User-42", "sub":"alice@example.test"})),
            Some("User-42")
        );
        assert_eq!(
            authenticated_run_owner_id(&json!({"sub":"alice@example.test"})),
            Some("alice@example.test")
        );
        assert_eq!(authenticated_run_owner_id(&json!({"sub":"anonymous"})), None);
        assert_eq!(authenticated_run_owner_id(&json!({"role":"rw"})), None);
    }

    #[test]
    fn service_trigger_owner_must_match_authenticated_actor() {
        let capabilities = Arc::new(Mutex::new(CapabilityState::default()));
        let principal = json!({
            "user_id": "alice",
            "sub": "alice@example.test",
            "role": "rw",
            "ns": "alpha",
            "capabilities": ["service.manage", "write_graph"]
        });
        let admitted = runtime_owner_binding_v1(&principal, &capabilities).unwrap();

        let (owner, binding) = PostgresRunApplicationService::service_run_owner_binding(
            Some(" alice "),
            Some(&admitted),
        )
        .unwrap();
        assert_eq!(owner.as_deref(), Some("alice"));
        assert_eq!(binding.as_ref().map(|value| value.user_id.as_str()), Some("alice"));

        let (owner, binding) =
            PostgresRunApplicationService::service_run_owner_binding(None, Some(&admitted))
                .unwrap();
        assert!(owner.is_none() && binding.is_none());
        assert!(PostgresRunApplicationService::service_run_owner_binding(
            Some("victim"),
            Some(&admitted)
        )
        .is_err());
        assert!(PostgresRunApplicationService::service_run_owner_binding(
            Some("victim"),
            None
        )
        .is_err());
    }

    #[test]
    fn inline_run_owner_revalidation_uses_current_identity_and_admission_ceiling() {
        let capabilities = Arc::new(Mutex::new(CapabilityState::default()));
        let admitted = json!({
            "user_id": "alice",
            "sub": "alice@example.test",
            "role": "rw",
            "ns": ["alpha", "beta"],
            "storage_ns": "alpha",
            "capabilities": ["read_graph", "write_graph"]
        });
        let binding = runtime_owner_binding_v1(&admitted, &capabilities).unwrap();
        let current = AuthUser {
            user_id: "alice".to_owned(),
            email: "alice@example.test".to_owned(),
            display_name: None,
            is_active: true,
            global_role: Some("rw".to_owned()),
            global_ns: Some("alpha".to_owned()),
        };
        let principal =
            revalidated_runtime_owner_principal(&binding, &current, &capabilities).unwrap();
        assert_eq!(principal["ns"], json!(["alpha"]));
        assert!(principal_strings(&principal, "capabilities").contains(&"write_graph".to_owned()));

        let downgraded = AuthUser {
            global_role: Some("ro".to_owned()),
            ..current.clone()
        };
        assert!(revalidated_runtime_owner_principal(&binding, &downgraded, &capabilities).is_err());
        let deactivated = AuthUser {
            is_active: false,
            ..current.clone()
        };
        assert!(revalidated_runtime_owner_principal(&binding, &deactivated, &capabilities).is_err());
        let moved_namespace = AuthUser {
            global_ns: Some("beta".to_owned()),
            ..current.clone()
        };
        assert!(
            revalidated_runtime_owner_principal(&binding, &moved_namespace, &capabilities).is_err()
        );

        let narrow_admission = json!({
            "user_id": "alice",
            "sub": "alice@example.test",
            "role": "rw",
            "ns": "alpha",
            "capabilities": ["read_graph"]
        });
        let narrow_binding = runtime_owner_binding_v1(&narrow_admission, &capabilities).unwrap();
        capabilities
            .lock()
            .unwrap()
            .approvals
            .insert(("alice".to_owned(), "later-approval".to_owned()), BTreeSet::from(["write_graph".to_owned()]));
        assert!(
            revalidated_runtime_owner_principal(&narrow_binding, &current, &capabilities).is_err(),
            "a post-admission capability approval must not widen the run"
        );
    }

    #[tokio::test]
    async fn postgres_submit_rejects_user_id_spoofing_before_storage() {
        let service = PostgresRunApplicationService::from_dsn(
            "postgresql://worker:secret@127.0.0.1:1/unreachable",
            "worker_owner_authority",
        )
        .unwrap();
        let response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/workflow/runs".to_owned(),
                body: serde_json::to_vec(&json!({
                    "workflow_id": "wf",
                    "conversation_id": "conv",
                    "user_id": "victim"
                }))
                .unwrap(),
                principal: json!({"user_id":"attacker", "sub":"attacker", "role":"rw"}),
            })
            .await;
        assert_eq!(response.status, StatusCode::FORBIDDEN.as_u16());
        let body: Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(body["code"], "KOGWISTAR_RUN_OWNER_FORBIDDEN");

        let anonymous_claim = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/workflow/runs".to_owned(),
                body: serde_json::to_vec(&json!({
                    "workflow_id": "wf",
                    "conversation_id": "conv",
                    "user_id": "victim"
                }))
                .unwrap(),
                principal: json!({"sub":"anonymous", "role":"rw"}),
            })
            .await;
        assert_eq!(anonymous_claim.status, StatusCode::FORBIDDEN.as_u16());
        assert_eq!(
            serde_json::from_slice::<Value>(&anonymous_claim.body).unwrap()["code"],
            "KOGWISTAR_RUN_OWNER_FORBIDDEN"
        );
    }

    #[tokio::test]
    async fn postgres_worker_graph_effect_rejects_anonymous_principal_even_with_write_capability() {
        let service = PostgresRunApplicationService::from_dsn(
            "postgresql://worker:secret@127.0.0.1:1/unreachable",
            "worker_anonymous_authority",
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
                    "sub":"anonymous",
                    "role":"rw",
                    "storage_ns":"alpha",
                    "capabilities":["write_graph"]
                }),
            })
            .await;
        assert_eq!(response.status, StatusCode::FORBIDDEN.as_u16());
        let body: Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(body["code"], "KOGWISTAR_RUN_OWNER_FORBIDDEN");
    }

    #[tokio::test]
    #[ignore = "requires KOGWISTAR_TEST_PG_DSN; run explicitly with --include-ignored"]
    async fn postgres_worker_graph_effect_rechecks_durable_run_owner_in_transaction() {
        let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
            .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
        let schema = format!("rust_run_owner_{}", uuid::Uuid::new_v4().simple());
        let service = PostgresRunApplicationService::from_dsn(&dsn, &schema).unwrap();
        service.ensure_schema().await.unwrap();
        service
            .store
            .create_graph_schema(3, kogwistar_store_postgres::GraphTableNames::default())
            .await
            .unwrap();
        service
            .store
            .create_server_run(ServerRunCreate {
                run_id: "owned-run".to_owned(),
                conversation_id: "conversation".to_owned(),
                workflow_id: "workflow".to_owned(),
                user_id: Some("run-owner".to_owned()),
                user_turn_node_id: "turn".to_owned(),
                status: "queued".to_owned(),
            })
            .await
            .unwrap();
        service
            .store
            .project_lane_message(NewProjectedLaneMessage {
                message_id: "lane-1".to_owned(),
                namespace: "workflow".to_owned(),
                purpose: "system".to_owned(),
                inbox_id: "workflow-runtime".to_owned(),
                conversation_id: "conversation".to_owned(),
                recipient_id: "python-worker".to_owned(),
                sender_id: "rust-scheduler".to_owned(),
                msg_type: "workflow.step.execute".to_owned(),
                status: "pending".to_owned(),
                created_at: 1,
                available_at: 1,
                run_id: Some("owned-run".to_owned()),
                step_id: Some("step-1".to_owned()),
                correlation_id: Some("owned-run".to_owned()),
                payload_json: Some("{}".to_owned()),
                error_json: None,
            })
            .await
            .unwrap();
        let claimed = service
            .store
            .claim_projected_lane_messages("workflow", "workflow-runtime", "worker-1", 1, 30)
            .await
            .unwrap();
        assert_eq!(claimed.len(), 1);
        assert_eq!(claimed[0].message_id, "lane-1");
        assert_eq!(claimed[0].status, "claimed");

        let response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/internal/runtime/results".to_owned(),
                body: serde_json::to_vec(&json!({
                    "handoff": {
                        "message_id": "lane-1",
                        "claimed_by": "worker-1",
                        "run_id": "owned-run",
                        "step_id": "step-1",
                        "correlation_id": "owned-run"
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
                                "event_id": "owner-check-event",
                                "op": "ADD",
                                "payload": {"id": "owner-check-node"},
                                "record": {"id": "owner-check-node"},
                                "embedding_dim": 0
                            }]
                        }
                    }
                }))
                .unwrap(),
                principal: json!({
                    "user_id":"different-user",
                    "sub":"different-user",
                    "role":"rw",
                    "storage_ns":"alpha",
                    "capabilities":["write_graph"]
                }),
            })
            .await;
        assert_eq!(response.status, StatusCode::FORBIDDEN.as_u16());
        assert_eq!(
            serde_json::from_slice::<Value>(&response.body).unwrap()["code"],
            "KOGWISTAR_RUN_OWNER_FORBIDDEN"
        );
        let records = service
            .store
            .graph_projection_records(GraphProjectionRead {
                scope: GraphScope {
                    namespace: "alpha".to_owned(),
                    workspace_id: None,
                    graph_space: None,
                },
                table: "gke_nodes".to_owned(),
                ids: Some(vec!["owner-check-node".to_owned()]),
                metadata: MetadataFilter::default(),
                limit: 10,
            })
            .await
            .unwrap();
        assert!(records.is_empty(), "unauthorized graph effects must not persist");
    }

    #[tokio::test]
    #[ignore = "requires KOGWISTAR_TEST_PG_DSN; run explicitly against disposable PostgreSQL"]
    async fn postgres_authenticated_native_graph_operation_commits_and_replays() {
        let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN").unwrap();
        let schema = format!("rust_native_graph_{}", uuid::Uuid::new_v4().simple());
        let mut registry = kogwistar_runtime::NativeOperationRegistry::default();
        registry.register_graph("app/graph", "v1", |request| {
            let batch = serde_json::from_value(json!({
                "contract_version": 1,
                "atomic": true,
                "mutations": [{
                    "scope": {"namespace": "alpha"},
                    "table": "gke_nodes",
                    "entity_kind": "node",
                    "event_id": format!("graph-event-{}", request.run_id),
                    "op": "ADD",
                    "payload": {"id": "native-node"},
                    "record": {"id": "native-node", "document": "native graph"},
                    "embedding_dim": 0
                }],
                "index_jobs": [{
                    "job_id": format!("graph-job-{}", request.run_id),
                    "namespace": "alpha",
                    "entity_kind": "node",
                    "entity_id": "native-node",
                    "index_kind": "node_embedding",
                    "op": "UPSERT"
                }]
            })).unwrap();
            Ok((kogwistar_runtime::NativeOperationOutput {
                state_update: Vec::new(),
                result: Some(json!({"native_graph": true})),
            }, batch))
        }).unwrap();
        let service = PostgresRunApplicationService::from_dsn(&dsn, &schema)
            .unwrap().with_native_operations(registry);
        service.ensure_schema().await.unwrap();
        service.store.create_graph_schema(3, kogwistar_store_postgres::GraphTableNames::default()).await.unwrap();

        let secret = "native-graph-test-secret";
        let token = |subject: &str, namespace: &str, caps: &[&str]| encode(
            &Header::new(Algorithm::HS256),
            &json!({"sub": subject, "role": "rw", "roles": ["reader"], "ns": "workflow",
                "storage_ns": namespace, "capabilities": caps, "exp": 4102444800_u64}),
            &EncodingKey::from_secret(secret.as_bytes()),
        ).unwrap();
        let owner = token("alice", "alpha", &["workflow.run.write", "write_graph"]);
        let other = token("bob", "alpha", &["workflow.run.write", "write_graph"]);
        let wrong_scope = token("alice", "beta", &["workflow.run.write", "write_graph"]);
        let no_write = token("alice", "alpha", &["workflow.run.write"]);
        let app = router_with_application(ApiState {
            health: snapshot(), required_roles: vec!["reader".to_owned()],
            implementation: ImplementationSnapshot::default(),
            auth: AuthConfig { mode: "jwt".to_owned(), algorithm: Some("HS256".to_owned()),
                key: Some(secret.to_owned()), ..AuthConfig::default() },
        }, Arc::new(service.clone()));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let client = reqwest::Client::new();
        let base = format!("http://{address}");
        let submit = client.post(format!("{base}/api/workflow/runs")).bearer_auth(&owner)
            .json(&json!({"run_id":"native-graph-run", "workflow_id":"wf", "conversation_id":"conv",
                "start_node_id":"entry", "node_ops":{"entry":"rust-native/app/graph"}}))
            .send().await.unwrap();
        assert_eq!(submit.status(), StatusCode::ACCEPTED);
        tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
        let claim = client.post(format!("{base}/internal/runtime/claim")).bearer_auth(&owner)
            .json(&json!({"claimed_by":"rust-native-worker", "limit":1, "lease_seconds":60,
                "recipient_ids":["rust-native-worker"]})).send().await.unwrap();
        assert_eq!(claim.status(), StatusCode::OK);
        let work = claim.json::<Value>().await.unwrap()["work"][0].clone();
        let execution = json!({"handoff": {"message_id":work["message_id"], "claimed_by":work["claimed_by"],
            "run_id":work["run_id"], "step_id":work["step_id"], "correlation_id":work["correlation_id"]},
            "payload":work["payload"]});
        assert!(
            execution["handoff"]["message_id"].is_string(),
            "claim must return work before exercising duplicate execution: {execution}"
        );
        assert_eq!(execution["payload"]["executor_target"], "rust-native");
        let python_result_bypass = client
            .post(format!("{base}/internal/runtime/results"))
            .bearer_auth(&owner)
            .json(&json!({
                "handoff": execution["handoff"],
                "effect": {
                    "contract_version": 1,
                    "effect_id": "forged-python-graph-effect",
                    "graph_mutation_batch": {
                        "contract_version": 1,
                        "atomic": true,
                        "mutations": [{
                            "scope": {"namespace": "alpha"},
                            "table": "gke_nodes",
                            "entity_kind": "node",
                            "event_id": "forged-python-graph-event",
                            "op": "ADD",
                            "payload": {"id": "forged-python-node"},
                            "record": {"id": "forged-python-node", "document": "forged"},
                            "embedding_dim": 0
                        }],
                        "index_jobs": [{
                            "job_id": "forged-python-graph-job",
                            "namespace": "alpha",
                            "entity_kind": "node",
                            "entity_id": "forged-python-node",
                            "index_kind": "node_embedding",
                            "op": "UPSERT"
                        }]
                    }
                }
            }))
            .send()
            .await
            .unwrap();
        let python_result_bypass_status = python_result_bypass.status();
        let python_result_bypass_body = python_result_bypass.text().await.unwrap();
        assert_eq!(
            python_result_bypass_status,
            StatusCode::CONFLICT,
            "{python_result_bypass_body}"
        );
        assert!(service.store.replay_raw_events("alpha", 0, 100).await.unwrap().is_empty());
        assert!(service.store.list_index_jobs(Some("alpha"), None, None, None, None, 100).await.unwrap().is_empty());
        let mut wrong_target_execution = execution.clone();
        wrong_target_execution["payload"]["executor_target"] = json!("python-worker");
        let wrong_target = client
            .post(format!("{base}/internal/runtime/native/execute"))
            .bearer_auth(&owner)
            .json(&wrong_target_execution)
            .send()
            .await
            .unwrap();
        let wrong_target_status = wrong_target.status();
        let wrong_target_body = wrong_target.text().await.unwrap();
        assert_eq!(
            wrong_target_status,
            StatusCode::CONFLICT,
            "{wrong_target_body}"
        );
        assert!(service.store.replay_raw_events("alpha", 0, 100).await.unwrap().is_empty());
        assert!(service.store.list_index_jobs(Some("alpha"), None, None, None, None, 100).await.unwrap().is_empty());
        let pending_lane = service.store.get_projected_lane_message(
            work["message_id"].as_str().unwrap(),
        ).await.unwrap().unwrap();
        assert_eq!(pending_lane.status, "claimed");
        for denied in [&other, &wrong_scope, &no_write] {
            let response = client.post(format!("{base}/internal/runtime/native/execute"))
                .bearer_auth(denied).json(&execution).send().await.unwrap();
            let status = response.status();
            let body = response.text().await.unwrap();
            assert_eq!(status, StatusCode::FORBIDDEN, "{body}");
        }
        assert!(service.store.replay_raw_events("alpha", 0, 100).await.unwrap().is_empty());
        assert!(service.store.list_index_jobs(Some("alpha"), None, None, None, None, 100).await.unwrap().is_empty());
        for _ in 0..2 {
            let response = client.post(format!("{base}/internal/runtime/native/execute"))
                .bearer_auth(&owner).json(&execution).send().await.unwrap();
            let status = response.status();
            let body = response.text().await.unwrap();
            assert_eq!(status, StatusCode::OK, "{body}");
        }
        let records = service.store.graph_projection_records(GraphProjectionRead {
            scope: GraphScope { namespace:"alpha".to_owned(), workspace_id:None, graph_space:None },
            table:"gke_nodes".to_owned(), ids:Some(vec!["native-node".to_owned()]),
            metadata:MetadataFilter::default(), limit:10,
        }).await.unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].document.as_deref(), Some("native graph"));
        assert_eq!(service.store.list_index_jobs(Some("alpha"), None, None, None, None, 100).await.unwrap().len(), 1);
        let graph_events = service.store.replay_raw_events("alpha", 0, 100).await.unwrap();
        assert_eq!(graph_events.len(), 1);
        assert_eq!(graph_events[0].event_id, "graph-event-native-graph-run");
        let lane = service.store.get_projected_lane_message(work["message_id"].as_str().unwrap()).await.unwrap().unwrap();
        assert_eq!(lane.status, "completed");
        assert_eq!(service.store.get_server_run("native-graph-run").await.unwrap().unwrap().status, "succeeded");
        server.abort();
        let (cleanup, connection) = tokio_postgres::connect(&dsn, tokio_postgres::NoTls).await.unwrap();
        let task = tokio::spawn(connection);
        cleanup.batch_execute(&format!("DROP SCHEMA \"{schema}\" CASCADE")).await.unwrap();
        drop(cleanup);
        task.await.unwrap().unwrap();
    }

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
        service.ensure_schema().await.unwrap();
        service
            .store
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
                "capabilities": ["workflow.run.write", "write_graph"]
            }),
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
            .batch_execute(&format!("DROP SCHEMA \"{schema}\" CASCADE"))
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

    #[tokio::test]
    async fn postgres_send_message_checks_actor_and_request_before_storage() {
        let service = PostgresRunApplicationService::from_dsn(
            "postgresql://worker:secret@127.0.0.1:1/unreachable",
            "send_message_admission_authority",
        )
        .unwrap();
        let request = |namespace: &str, capabilities: &[&str], text: &str| ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/syscall/v1/send_message".to_owned(),
            body: serde_json::to_vec(&json!({
                "version": "v1",
                "op": "send_message",
                "args": {"conversation_id": "conversation-1", "text": text}
            }))
            .unwrap(),
            principal: json!({
                "user_id": "actor",
                "sub": "actor@example.test",
                "role": "rw",
                "ns": namespace,
                "storage_ns": "conversation",
                "capabilities": capabilities,
            }),
        };
        let denied = service
            .execute(request("conversation", &["read_graph"], "hello"))
            .await;
        assert_eq!(denied.status, StatusCode::FORBIDDEN.as_u16());
        let invalid = service
            .execute(request("conversation", &["send_message"], "  "))
            .await;
        assert_eq!(invalid.status, StatusCode::BAD_REQUEST.as_u16());
        let missing_fields = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/syscall/v1/send_message".to_owned(),
                body: serde_json::to_vec(&json!({
                    "version": "v1",
                    "op": "send_message",
                    "args": {}
                }))
                .unwrap(),
                principal: json!({
                    "user_id": "actor",
                    "sub": "actor@example.test",
                    "role": "rw",
                    "ns": "conversation",
                    "storage_ns": "conversation",
                    "capabilities": ["send_message"]
                }),
            })
            .await;
        assert_eq!(missing_fields.status, StatusCode::BAD_REQUEST.as_u16());
        let wrong_namespace = service
            .execute(request("workflow", &["send_message"], "hello"))
            .await;
        assert_eq!(wrong_namespace.status, StatusCode::FORBIDDEN.as_u16());
    }

    #[tokio::test]
    #[ignore = "requires KOGWISTAR_TEST_PG_DSN; run explicitly against disposable PostgreSQL"]
    async fn postgres_send_message_admission_is_one_graph_run_event_lane_transaction() {
        let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
            .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
        let schema = format!("rust_send_message_{}", uuid::Uuid::new_v4().simple());
        let auth_path = std::env::temp_dir().join(format!(
            "kogwistar-send-message-auth-{}.sqlite",
            uuid::Uuid::new_v4().simple()
        ));
        let auth = SqliteAuthStore::open(&auth_path).unwrap();
        auth.resolve_external_identity(ResolveExternalIdentity {
            issuer: "test".to_owned(),
            subject: "conversation-owner".to_owned(),
            email: "conversation-owner@example.test".to_owned(),
            display_name: None,
            new_user_id: "conversation-owner".to_owned(),
            default_role: "rw".to_owned(),
            default_ns: "alpha".to_owned(),
        })
        .await
        .unwrap();
        drop(auth);
        let auth_config = AuthConfig {
            auth_db_url: Some(format!("sqlite:///{}", auth_path.display())),
            ..AuthConfig::default()
        };
        let provider = Arc::new(FixedEmbeddingProvider {
            profile: kogwistar_engine::embedding::EmbeddingProfile {
                provider: "test".to_owned(),
                model: "fixed-v1".to_owned(),
                dimension: 3,
                similarity_metric: "cosine".to_owned(),
                endpoint_fingerprint: None,
                max_sequence_length: None,
                crop_token_budget: None,
                tokenizer_fingerprint: None,
                crop_policy: None,
                embedding_kind: Some("single_vector".to_owned()),
                model_revision: None,
                preprocessing_fingerprint: None,
                max_image_patches: None,
            },
            vector: vec![0.6, 0.8, 0.0],
            calls: Arc::new(Mutex::new(Vec::new())),
            fail: None,
        });
        let service = PostgresRunApplicationService::from_dsn(&dsn, &schema)
            .unwrap()
            .with_owner_auth_config(&auth_config)
            .await
            .unwrap()
            .with_embedding_provider(provider);
        service.ensure_schema().await.unwrap();
        service
            .store
            .create_graph_schema(3, kogwistar_store_postgres::GraphTableNames::default())
            .await
            .unwrap();

        let conversation_id = "send-message-conversation";
        let root_id = "send-message-root";
        service
            .store
            .apply_graph_mutation(GraphMutation {
                scope: GraphScope {
                    namespace: "alpha".to_owned(),
                    workspace_id: None,
                    graph_space: None,
                },
                table: "gke_nodes".to_owned(),
                entity_kind: "node".to_owned(),
                event_id: format!("{root_id}-add"),
                op: "ADD".to_owned(),
                payload: json!({"id": root_id}),
                record: GraphRecord {
                    id: root_id.to_owned(),
                    document: Some(
                        json!({
                            "id": root_id,
                            "user_id": "conversation-owner",
                            "conversation_id": conversation_id,
                            "summary": "Start of conversation",
                        })
                        .to_string(),
                    ),
                    metadata: serde_json::Map::from_iter([
                        ("entity_type".to_owned(), json!("conversation_start")),
                        ("conversation_id".to_owned(), json!(conversation_id)),
                        ("user_id".to_owned(), json!("conversation-owner")),
                        ("turn_index".to_owned(), json!(-1)),
                        ("in_conversation_chain".to_owned(), json!(true)),
                        ("mentions".to_owned(), json!([])),
                    ]),
                    embedding: None,
                },
                embedding_dim: 3,
            })
            .await
            .unwrap();
        let acl_id = "send-message-root-acl";
        service
            .store
            .apply_graph_mutation(GraphMutation {
                scope: GraphScope {
                    namespace: "alpha".to_owned(),
                    workspace_id: None,
                    graph_space: None,
                },
                table: "gke_nodes".to_owned(),
                entity_kind: "node".to_owned(),
                event_id: format!("{acl_id}-add"),
                op: "ADD".to_owned(),
                payload: json!({"id": acl_id}),
                record: GraphRecord {
                    id: acl_id.to_owned(),
                    document: Some(json!({"id": acl_id}).to_string()),
                    metadata: serde_json::Map::from_iter([
                        ("entity_type".to_owned(), json!("acl_record")),
                        ("acl_truth_graph".to_owned(), json!("conversation")),
                        ("acl_target_grain".to_owned(), json!("node")),
                        ("acl_target_entity_id".to_owned(), json!(root_id)),
                        ("acl_mode".to_owned(), json!("shared")),
                        ("acl_version".to_owned(), json!(1)),
                        ("owner_id".to_owned(), json!("conversation-owner")),
                        (
                            "shared_with_principals".to_owned(),
                            json!(["request-actor@example.test"]),
                        ),
                    ]),
                    embedding: None,
                },
                embedding_dim: 3,
            })
            .await
            .unwrap();

        let (fault_client, connection) =
            tokio_postgres::connect(&dsn, tokio_postgres::NoTls).await.unwrap();
        let connection_task = tokio::spawn(connection);
        fault_client
            .batch_execute(&format!(
                "CREATE FUNCTION \"{schema}\".reject_send_message_index_job() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'injected send_message index failure'; END; $$; CREATE TRIGGER reject_send_message_index_job BEFORE INSERT ON \"{schema}\".index_jobs FOR EACH ROW EXECUTE FUNCTION \"{schema}\".reject_send_message_index_job();"
            ))
            .await
            .unwrap();
        drop(fault_client);
        connection_task.await.unwrap().unwrap();

        let make_request = || ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/syscall/v1/send_message".to_owned(),
            body: serde_json::to_vec(&json!({
                "version": "v1",
                "op": "send_message",
                "args": {
                    "conversation_id": conversation_id,
                    "text": "A user question",
                    "workflow_id": "send-message-admission-test"
                }
            }))
            .unwrap(),
            principal: json!({
                "user_id": "request-actor",
                "sub": "request-actor@example.test",
                "role": "rw",
                "ns": "conversation",
                "storage_ns": "alpha",
                "capabilities": ["send_message"]
            }),
        };
        let rejected = service.execute(make_request()).await;
        assert!(rejected.status >= 500, "{}", String::from_utf8_lossy(&rejected.body));
        let still_only_root = service
            .store
            .graph_projection_records(GraphProjectionRead {
                scope: GraphScope {
                    namespace: "alpha".to_owned(),
                    workspace_id: None,
                    graph_space: None,
                },
                table: "gke_nodes".to_owned(),
                ids: None,
                metadata: MetadataFilter {
                    equals: BTreeMap::from([(
                        "conversation_id".to_owned(),
                        json!(conversation_id),
                    )]),
                },
                limit: 20,
            })
            .await
            .unwrap();
        assert_eq!(still_only_root.len(), 1);
        assert_eq!(still_only_root[0].id, root_id);
        assert_eq!(
            service
                .store
                .list_index_jobs(Some("alpha"), None, None, None, None, 20)
                .await
                .unwrap()
                .len(),
            0
        );
        let (verify_client, connection) =
            tokio_postgres::connect(&dsn, tokio_postgres::NoTls).await.unwrap();
        let connection_task = tokio::spawn(connection);
        for table in ["server_runs", "server_run_events", "projected_lane_messages"] {
            let count: i64 = verify_client
                .query_one(&format!("SELECT COUNT(*) FROM \"{schema}\".{table}"), &[])
                .await
                .unwrap()
                .get(0);
            assert_eq!(count, 0, "failed admission left rows in {table}");
        }
        drop(verify_client);
        connection_task.await.unwrap().unwrap();

        let (cleanup_client, connection) =
            tokio_postgres::connect(&dsn, tokio_postgres::NoTls).await.unwrap();
        let connection_task = tokio::spawn(connection);
        cleanup_client
            .batch_execute(&format!(
                "DROP TRIGGER reject_send_message_index_job ON \"{schema}\".index_jobs; DROP FUNCTION \"{schema}\".reject_send_message_index_job();"
            ))
            .await
            .unwrap();
        drop(cleanup_client);
        connection_task.await.unwrap().unwrap();

        let accepted = service.execute(make_request()).await;
        assert_eq!(accepted.status, StatusCode::OK.as_u16());
        let response: Value = serde_json::from_slice(&accepted.body).unwrap();
        let run_id = response["result"]["run_id"].as_str().unwrap();
        let turn_id = response["result"]["turn_node_id"].as_str().unwrap();
        let nodes = service
            .store
            .graph_projection_records(GraphProjectionRead {
                scope: GraphScope {
                    namespace: "alpha".to_owned(),
                    workspace_id: None,
                    graph_space: None,
                },
                table: "gke_nodes".to_owned(),
                ids: None,
                metadata: MetadataFilter {
                    equals: BTreeMap::from([(
                        "conversation_id".to_owned(),
                        json!(conversation_id),
                    )]),
                },
                limit: 20,
            })
            .await
            .unwrap();
        assert_eq!(nodes.len(), 2);
        assert!(nodes.iter().any(|node| node.id == turn_id));
        assert_eq!(
            service
                .store
                .graph_projection_records(GraphProjectionRead {
                    scope: GraphScope {
                        namespace: "alpha".to_owned(),
                        workspace_id: None,
                        graph_space: None,
                    },
                    table: "gke_edges".to_owned(),
                    ids: None,
                    metadata: MetadataFilter::default(),
                    limit: 20,
                })
                .await
                .unwrap()
                .len(),
            1
        );
        let run = service.store.get_server_run(run_id).await.unwrap().unwrap();
        assert_eq!(run.user_id.as_deref(), Some("conversation-owner"));
        let events = service.store.list_server_run_events(run_id, 0, 10).await.unwrap();
        assert_eq!(events.len(), 1);
        let created: Value = serde_json::from_str(&events[0].payload_json).unwrap();
        assert_eq!(created["admission_actor"]["user_id"], "request-actor");
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
        assert_eq!(lanes[0].status, "pending");
        assert_eq!(
            service
                .store
                .list_index_jobs(Some("alpha"), None, None, None, None, 20)
                .await
                .unwrap()
                .iter()
                .map(|job| job.index_kind.as_str())
                .collect::<BTreeSet<_>>(),
            BTreeSet::from(["edge_endpoints", "edge_refs", "node_docs", "node_refs"])
        );

        drop(service);
        let (cleanup, connection) =
            tokio_postgres::connect(&dsn, tokio_postgres::NoTls).await.unwrap();
        let connection_task = tokio::spawn(connection);
        cleanup
            .batch_execute(&format!("DROP SCHEMA \"{schema}\" CASCADE"))
            .await
            .unwrap();
        drop(cleanup);
        connection_task.await.unwrap().unwrap();
        let _ = std::fs::remove_file(auth_path);
    }

    #[test]
    fn conversation_message_span_uses_python_character_offsets() {
        let span = conversation_span("conv-1", "turn-1", "A\u{1f642}B");
        assert_eq!(span["start_char"], 0);
        assert_eq!(span["end_char"], 3);
        assert_eq!(span["excerpt"], "A\u{1f642}B");
        assert_eq!(span["verification"]["notes"], "verbatim user input");
    }

    #[test]
    fn conversation_admission_matches_python_single_stage_indexes_and_distances() {
        assert_eq!(
            conversation_index_job_kinds("node"),
            Some(["node_docs", "node_refs"].as_slice())
        );
        assert_eq!(
            conversation_index_job_kinds("edge"),
            Some(["edge_refs", "edge_endpoints"].as_slice())
        );
        assert_eq!(conversation_index_job_kinds("unknown"), None);

        let previous = GraphRecord {
            id: "previous-turn".to_owned(),
            document: Some(r#"{"summary":"A🙂"}"#.to_owned()),
            metadata: serde_json::Map::from_iter([
                ("char_distance_from_last_summary".to_owned(), json!(4)),
                ("turn_distance_from_last_summary".to_owned(), json!(5)),
            ]),
            embedding: None,
        };
        assert_eq!(
            conversation_next_turn_distances(&previous, "B🙂", 6),
            (8, 7, 6)
        );
    }

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
                    .header("x-kogwistar-roles", "reader")
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

        let typed: ContextSnapshotV1 = serde_json::from_value({
            let mut value = expected.clone();
            value
                .as_object_mut()
                .unwrap()
                .insert("schema_version".to_owned(), json!(1));
            value
        })
        .unwrap();
        assert_eq!(public_context_snapshot_value(typed), expected);
        assert_eq!(
            context_snapshot_syscall_field(json!({}).as_object().unwrap(), "stage"),
            Ok(None)
        );
        assert_eq!(
            context_snapshot_syscall_field(json!({"stage": null}).as_object().unwrap(), "stage"),
            Ok(None)
        );
        assert!(
            context_snapshot_syscall_field(json!({"stage": 1}).as_object().unwrap(), "stage")
                .is_err()
        );

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
                        .header("x-kogwistar-roles", "reader")
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

    #[tokio::test]
    async fn sqlite_chroma_graph_topology_reads_are_scoped_and_acl_filtered() {
        use kogwistar_store_chroma::{ChromaGraphStore, ChromaStoreConfig};
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use jsonwebtoken::{EncodingKey, Header, encode};
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tower::ServiceExt;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let fake = tokio::spawn(async move {
            let mut observed = Vec::new();
            for _ in 0..20 {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut bytes = Vec::new();
                let mut chunk = [0_u8; 4096];
                let (split, length) = loop {
                    let count = stream.read(&mut chunk).await.unwrap();
                    assert!(count > 0);
                    bytes.extend_from_slice(&chunk[..count]);
                    if let Some(split) = bytes.windows(4).position(|v| v == b"\r\n\r\n") {
                        let head = String::from_utf8_lossy(&bytes[..split]);
                        let length = head.lines().find_map(|line| {
                            let (name, value) = line.split_once(':')?;
                            name.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse::<usize>().ok())
                                .flatten()
                        }).unwrap_or(0);
                        if bytes.len() >= split + 4 + length { break (split, length); }
                    }
                };
                let head = String::from_utf8_lossy(&bytes[..split]);
                let line = head.lines().next().unwrap().to_owned();
                let input: Value = serde_json::from_slice(&bytes[split + 4..split + 4 + length]).unwrap();
                let payload = if line.contains("/collections HTTP") {
                    json!({"id": format!("collection-{}", input["name"].as_str().unwrap()), "metadata":{"hnsw:space":"cosine"}})
                } else if line.contains("collection-nodes/query") {
                    let scoped = |document: &str| {
                        let original = if document == "doc2" {
                            json!({"_kogwistar_stage2_ready":true,"_kogwistar_source_revision":1})
                        } else {
                            json!({"_kogwistar_stage2_ready":true,"_kogwistar_source_revision":2})
                        };
                        json!({
                            "_kogwistar_namespace":"scope-a",
                            "workspace_id":"workspace-a",
                            "graph_space":"knowledge",
                            "_kogwistar_original_metadata":original.to_string(),
                            "_test_document": document
                        })
                    };
                    json!({
                        "ids":[["n4","n2"]],
                        "documents":[["doc4","doc2"]],
                        "metadatas":[[scoped("doc4"),scoped("doc2")]],
                        "embeddings":[[[0.1,0.2],[0.1,0.2]]],
                        "distances":[[0.03,0.1]]
                    })
                } else if line.contains("collection-edges/query") {
                    let scoped = |document: &str| json!({
                        "_kogwistar_namespace":"scope-a",
                        "workspace_id":"workspace-a",
                        "graph_space":"knowledge",
                        "_kogwistar_original_metadata":"{}",
                        "_test_document": document
                    });
                    json!({
                        "ids":[["e2","e1"]],
                        "documents":[[
                            r#"{"source_ids":["n1"],"target_ids":["n3"]}"#,
                            r#"{"source_ids":["n1"],"target_ids":["n2"]}"#
                        ]],
                        "metadatas":[[scoped("edge2"),scoped("edge1")]],
                        "embeddings":[[[0.1,0.2],[0.1,0.2]]],
                        "distances":[[0.02,0.1]]
                    })
                } else if line.contains("collection-nodes/get") {
                    let metas = [
                        json!({}),
                        json!({"_kogwistar_stage2_ready":true,"_kogwistar_source_revision":1}),
                        json!({}),
                        json!({"_kogwistar_stage2_ready":true,"_kogwistar_source_revision":2}),
                        json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"node","acl_target_entity_id":"n1","acl_version":1,"acl_mode":"public"}),
                        json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"node","acl_target_entity_id":"n2","acl_version":1,"acl_mode":"public"}),
                        json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"node","acl_target_entity_id":"n3","acl_version":1,"acl_mode":"private"}),
                        json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"node","acl_target_entity_id":"n4","acl_version":1,"acl_mode":"public"}),
                        json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"edge","acl_target_entity_id":"e1","acl_version":1,"acl_mode":"public"}),
                    ];
                    json!({"ids":["n1","n2","n3","n4","acl-n1","acl-n2","acl-n3","acl-n4","acl-e1"],
                        "documents":["{}","{}","{}","{}","{}","{}","{}","{}","{}"],
                        "metadatas":metas.into_iter().map(|m| json!({"_kogwistar_namespace":"scope-a","workspace_id":"workspace-a","graph_space":"knowledge","_kogwistar_original_metadata":m.to_string()})).collect::<Vec<_>>(),
                        "embeddings":[null,null,null,null,null,null,null,null,null]})
                } else {
                    json!({"ids":["e1"],
                        "documents":[r#"{"source_ids":["n1"],"target_ids":["n2"]}"#],
                        "metadatas":[json!({"_kogwistar_namespace":"scope-a","workspace_id":"workspace-a","graph_space":"knowledge","_kogwistar_original_metadata":"{}"})],
                        "embeddings":[null]})
                };
                let body = payload.to_string();
                let response = format!("HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}", body.len(), body);
                stream.write_all(response.as_bytes()).await.unwrap();
                observed.push((line, input));
            }
            observed
        });
        let shared = ChromaStoreConfig { endpoint, ..Default::default() };
        let graph = ChromaGraphStore::connect_entity_collections(
            ChromaStoreConfig { collection: "nodes".to_owned(), ..shared.clone() },
            ChromaStoreConfig { collection: "edges".to_owned(), ..shared },
        ).await.unwrap();
        let filter = MetadataFilter::default();
        let (nodes, edges) = tokio::try_join!(
            graph.graph_records_of_kind_bounded("scope-a", GraphEntityKind::Node, &filter, 100),
            graph.graph_records_of_kind_bounded("scope-a", GraphEntityKind::Edge, &filter, 100),
        )
        .unwrap();
        assert_eq!(nodes.len(), 9, "decoded node records: {nodes:?}");
        assert_eq!(edges.len(), 1, "decoded edge records: {edges:?}");
        let principal = GraphAclPrincipal { principal_id: "reader-1".to_owned(), ..Default::default() };
        let (visible_nodes, visible_edges) = graph_records_visible_by_acl(&nodes, &edges, "knowledge", &principal);
        assert_eq!(visible_nodes.len(), 3, "ACL-visible nodes: {visible_nodes:?}");
        assert_eq!(visible_edges.len(), 1, "ACL-visible edges: {visible_edges:?}");
        assert_eq!(
            graph_neighbors_from_records("n1", &visible_nodes, &visible_edges, None).nodes,
            ["n2".to_owned()].into()
        );
        let requested = vec!["n2".to_owned(), "n2".to_owned(), "missing".to_owned()];
        let batched = graph
            .graph_records_by_ids_of_kind("scope-a", GraphEntityKind::Node, &requested)
            .await
            .unwrap();
        assert_eq!(batched.iter().map(|record| record.id.as_str()).collect::<Vec<_>>(), ["n2"]);
        let path = std::env::temp_dir().join(format!("kogwistar-sqlite-chroma-{}.sqlite3", uuid::Uuid::new_v4()));
        let service = SqliteRunApplicationService::open(&path).unwrap().with_chroma_graph(graph);
        for (event_id, entity_id, op) in [
            ("n2-current", "n2", "ADD"),
            ("n4-old", "n4", "ADD"),
            ("n4-current", "n4", "REPLACE"),
        ] {
            service.store.append_raw_entity_event(
                "scope-a",
                SqliteNewRawEntityEvent {
                    event_id: event_id.to_owned(),
                    entity_kind: "node".to_owned(),
                    entity_id: entity_id.to_owned(),
                    op: op.to_owned(),
                    payload_json: "{}".to_owned(),
                },
            ).unwrap();
        }
        let graph_read_tools = ["kg_neighbors", "kg_find_edges", "kg_shortest_path", "kg_k_hop"];
        assert!(graph_read_tools.iter().all(|name| service.supports_mcp_tool("root", name)));
        assert!(!service.supports_mcp_tool("root", "kg_graph_mutate"));
        assert!(!SqliteRunApplicationService::open(&path).unwrap().supports_mcp_tool("root", "kg_neighbors"));

        let auth = AuthConfig {
            mode: "dev".to_owned(),
            algorithm: Some("HS256".to_owned()),
            key: Some("sqlite-chroma-mcp-secret".to_owned()),
            ..AuthConfig::default()
        };
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let token = |capabilities: &[&str]| {
            encode(
                &Header::new(jsonwebtoken::Algorithm::HS256),
                &json!({
                    "sub":"reader-1", "storage_ns":"scope-a", "workspace_id":"workspace-a",
                    "graph_space":"knowledge", "capabilities":capabilities, "exp":now + 300
                }),
                &EncodingKey::from_secret(b"sqlite-chroma-mcp-secret"),
            )
            .unwrap()
        };
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: vec![],
                implementation: ImplementationSnapshot::default(),
                auth,
            },
            Arc::new(service.clone()),
        );
        let listing = app.clone().oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {}", token(&["read_graph"])))
                .body(Body::from(r#"{"jsonrpc":"2.0","id":"list","method":"tools/list"}"#))
                .unwrap(),
        ).await.unwrap();
        assert_eq!(listing.status(), StatusCode::OK);
        let listing: Value = serde_json::from_slice(&to_bytes(listing.into_body(), 65_536).await.unwrap()).unwrap();
        let advertised = listing["result"]["tools"].as_array().unwrap().iter()
            .filter_map(|tool| tool["name"].as_str()).collect::<Vec<_>>();
        let advertised_graph_tools = advertised
            .iter()
            .copied()
            .filter(|name| name.starts_with("kg_"))
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            advertised_graph_tools,
            graph_read_tools.into_iter().collect(),
            "SQLite+Chroma must advertise exactly the implemented read-only graph surface"
        );

        let denied = app.clone().oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {}", token(&[])))
                .body(Body::from(r#"{"jsonrpc":"2.0","id":"denied","method":"tools/call","params":{"name":"kg_neighbors","arguments":{"rid":"n1"}}}"#))
                .unwrap(),
        ).await.unwrap();
        let denied: Value = serde_json::from_slice(&to_bytes(denied.into_body(), 65_536).await.unwrap()).unwrap();
        assert_eq!(denied["error"]["data"]["code"], "KOGWISTAR_CAPABILITY_FORBIDDEN");

        let mcp_read = app.oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {}", token(&["read_graph"])))
                .body(Body::from(r#"{"jsonrpc":"2.0","id":"neighbors","method":"tools/call","params":{"name":"kg_neighbors","arguments":{"rid":"n1"}}}"#))
                .unwrap(),
        ).await.unwrap();
        assert_eq!(mcp_read.status(), StatusCode::OK);
        let mcp_read: Value = serde_json::from_slice(&to_bytes(mcp_read.into_body(), 65_536).await.unwrap()).unwrap();
        assert_eq!(mcp_read["result"]["nodes"], json!(["n2"]));
        assert_eq!(mcp_read["result"]["edges"], json!(["e1"]));

        let request = |path: &str, body: Value, capabilities: Value| ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: path.to_owned(),
            body: serde_json::to_vec(&body).unwrap(),
            principal: json!({"sub":"reader-1","storage_ns":"scope-a","workspace_id":"workspace-a","graph_space":"knowledge","capabilities":capabilities}),
        };
        let neighbor_body = json!({"rid":"n1","truth_graph":"knowledge"});
        let read_capability = json!(["read_graph"]);
        for route in [
            INTERNAL_GRAPH_NEIGHBORS_ROUTE,
            INTERNAL_GRAPH_VECTOR_QUERY_ROUTE,
            INTERNAL_GRAPH_FIND_EDGES_ROUTE,
            INTERNAL_GRAPH_SHORTEST_PATH_ROUTE,
            INTERNAL_GRAPH_K_HOP_ROUTE,
        ] {
            assert!(service.supports_http_operation("POST", route), "route {route}");
        }
        assert_eq!(
            service
                .execute(request(
                    INTERNAL_GRAPH_NEIGHBORS_ROUTE,
                    neighbor_body.clone(),
                    json!([]),
                ))
                .await
                .status,
            StatusCode::FORBIDDEN.as_u16()
        );
        let result = service
            .execute(request(
                INTERNAL_GRAPH_NEIGHBORS_ROUTE,
                neighbor_body,
                read_capability.clone(),
            ))
            .await;
        assert_eq!(result.status, StatusCode::OK.as_u16());
        let result: Value = serde_json::from_slice(&result.body).unwrap();
        assert_eq!(result["nodes"], json!(["n2"]), "response: {result}");
        assert_eq!(result["edges"], json!(["e1"]), "response: {result}");

        let find_edges = service
            .execute(request(
                INTERNAL_GRAPH_FIND_EDGES_ROUTE,
                json!({}),
                read_capability.clone(),
            ))
            .await;
        assert_eq!(find_edges.status, StatusCode::OK.as_u16());
        assert_eq!(serde_json::from_slice::<Value>(&find_edges.body).unwrap()["edges"], json!(["e1"]));

        let shortest_path = service
            .execute(request(
                INTERNAL_GRAPH_SHORTEST_PATH_ROUTE,
                json!({"src_id":"n1","dst_id":"n2"}),
                read_capability.clone(),
            ))
            .await;
        assert_eq!(shortest_path.status, StatusCode::OK.as_u16());
        assert_eq!(
            serde_json::from_slice::<Value>(&shortest_path.body).unwrap()["path"],
            json!(["n1", "e1", "n2"])
        );

        let k_hop = service
            .execute(request(
                INTERNAL_GRAPH_K_HOP_ROUTE,
                json!({"start_ids":["n1"],"k":1}),
                read_capability.clone(),
            ))
            .await;
        assert_eq!(k_hop.status, StatusCode::OK.as_u16());
        assert_eq!(
            serde_json::from_slice::<Value>(&k_hop.body).unwrap()["layers"],
            json!([{"nodes":[],"edges":["e1"]}])
        );

        let vector_query = service
            .execute(request(
                INTERNAL_GRAPH_VECTOR_QUERY_ROUTE,
                json!({
                    "entity_kind":"node",
                    "query_embeddings":[[0.1,0.2]],
                    "metric":"cosine",
                    "n_results":2,
                    "include":["documents","distances"]
                }),
                read_capability,
            ))
            .await;
        assert_eq!(vector_query.status, StatusCode::OK.as_u16());
        let vector_query: Value = serde_json::from_slice(&vector_query.body).unwrap();
        assert_eq!(vector_query["ids"], json!([["n2"]]));
        assert_eq!(vector_query["documents"], json!([["doc2"]]));
        assert_eq!(vector_query["distances"], json!([[0.1]]));

        let edge_vector_query = service
            .execute(request(
                INTERNAL_GRAPH_VECTOR_QUERY_ROUTE,
                json!({
                    "entity_kind":"edge",
                    "query_embeddings":[[0.1,0.2]],
                    "metric":"cosine",
                    "n_results":2,
                    "include":["documents","distances"]
                }),
                json!(["read_graph"]),
            ))
            .await;
        assert_eq!(edge_vector_query.status, StatusCode::OK.as_u16());
        let edge_vector_query: Value = serde_json::from_slice(&edge_vector_query.body).unwrap();
        assert_eq!(edge_vector_query["ids"], json!([["e1"]]));
        assert_eq!(edge_vector_query["documents"], json!([[
            r#"{"source_ids":["n1"],"target_ids":["n2"]}"#
        ]]));
        assert_eq!(edge_vector_query["distances"], json!([[0.1]]));

        let observed = fake.await.unwrap();
        assert_eq!(observed.len(), 20);
        assert!(observed.iter().any(|(line, _)| line.contains("collection-nodes/get")));
        assert!(observed.iter().any(|(line, _)| line.contains("collection-edges/get")));
        assert!(observed.iter().any(|(line, payload)| {
            line.contains("collection-nodes/get") && payload["ids"] == json!(["n2", "missing"])
        }));
        let scoped_reads = observed
            .iter()
            .filter(|(line, payload)| {
                line.contains("/get HTTP") && payload["where"].to_string().contains("workspace_id")
            })
            .collect::<Vec<_>>();
        assert_eq!(scoped_reads.len(), 12);
        assert!(
            scoped_reads.iter().all(|(_, payload)| {
                payload["where"].to_string().contains("workspace_id")
                    && payload["where"].to_string().contains("graph_space")
            }),
            "scoped Chroma query payloads: {scoped_reads:#?}"
        );
        let vector_request = observed
            .iter()
            .find(|(line, _)| line.contains("collection-nodes/query"))
            .expect("scoped node vector request");
        assert_eq!(vector_request.1["n_results"], 2);
        assert!(vector_request.1["where"].to_string().contains("workspace_id"));
        assert!(vector_request.1["where"].to_string().contains("graph_space"));
        drop(service);
        let _ = std::fs::remove_file(path);
    }
