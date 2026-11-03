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
        let principal = revalidated_runtime_owner_principal(
            &binding,
            &current,
            &capabilities,
            "write_graph",
        )
        .unwrap();
        assert_eq!(principal["ns"], json!(["alpha"]));
        assert!(principal_strings(&principal, "capabilities").contains(&"write_graph".to_owned()));

        let downgraded = AuthUser {
            global_role: Some("ro".to_owned()),
            ..current.clone()
        };
        assert!(revalidated_runtime_owner_principal(
            &binding,
            &downgraded,
            &capabilities,
            "write_graph"
        )
        .is_err());
        let deactivated = AuthUser {
            is_active: false,
            ..current.clone()
        };
        assert!(revalidated_runtime_owner_principal(
            &binding,
            &deactivated,
            &capabilities,
            "write_graph"
        )
        .is_err());
        let moved_namespace = AuthUser {
            global_ns: Some("beta".to_owned()),
            ..current.clone()
        };
        assert!(
            revalidated_runtime_owner_principal(
                &binding,
                &moved_namespace,
                &capabilities,
                "write_graph"
            )
            .is_err()
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
            revalidated_runtime_owner_principal(
                &narrow_binding,
                &current,
                &capabilities,
                "write_graph"
            )
            .is_err(),
            "a post-admission capability approval must not widen the run"
        );
    }

    #[test]
    fn inline_run_owner_revalidation_supports_read_only_native_capability() {
        let capabilities = Arc::new(Mutex::new(CapabilityState::default()));
        let admitted = json!({
            "user_id": "alice",
            "sub": "alice@example.test",
            "role": "ro",
            "ns": "alpha",
            "capabilities": ["read_graph"]
        });
        let binding = runtime_owner_binding_v1(&admitted, &capabilities).unwrap();
        let current = AuthUser {
            user_id: "alice".to_owned(),
            email: "alice@example.test".to_owned(),
            display_name: None,
            is_active: true,
            global_role: Some("ro".to_owned()),
            global_ns: Some("alpha".to_owned()),
        };

        let principal = revalidated_runtime_owner_principal(
            &binding,
            &current,
            &capabilities,
            "read_graph",
        )
        .unwrap();
        assert!(principal_strings(&principal, "capabilities").contains(&"read_graph".to_owned()));
        assert!(revalidated_runtime_owner_principal(
            &binding,
            &current,
            &capabilities,
            "write_graph"
        )
        .is_err());
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

    struct HoldFirstWorkerResultResponse {
        inner: Arc<dyn ApplicationService>,
        committed: std::sync::Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
        held: std::sync::atomic::AtomicBool,
    }

    impl ApplicationService for HoldFirstWorkerResultResponse {
        fn execute(
            &self,
            request: ApiEffectRequest,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = ApiEffectResponse> + Send + '_>,
        > {
            let hold_response = request.method == "POST"
                && request.path_and_query == "/internal/runtime/results";
            Box::pin(async move {
                let response = self.inner.execute(request).await;
                if hold_response
                    && response.status == StatusCode::OK.as_u16()
                    && !self.held.swap(true, std::sync::atomic::Ordering::SeqCst)
                {
                    if let Some(committed) = self.committed.lock().unwrap().take() {
                        let _ = committed.send(());
                    }
                    std::future::pending::<()>().await;
                }
                response
            })
        }
    }

    #[tokio::test]
    #[ignore = "requires KOGWISTAR_TEST_PG_DSN; run explicitly with --include-ignored"]
    async fn postgres_worker_graph_effect_rechecks_durable_run_owner_in_transaction() {
        use tokio::io::AsyncWriteExt;

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
        let runtime_start: kogwistar_runtime::RecordedRuntimeTransition =
            serde_json::from_value(json!({
                "contract_version": 1,
                "transition_id": "owned-run-start",
                "expected_event_seq": 0,
                "kind": "start",
                "run_id": "owned-run",
                "workflow_id": "workflow",
                "conversation_id": "conversation",
                "user_id": "run-owner",
                "user_turn_node_id": "turn",
                "step_seq": 0,
                "node_id": "step-1",
                "token_id": "owned-run",
                "initial_state": {},
                "frontier": {
                    "pending": [["step-1", 0, "owned-run", null]],
                    "suspended": [],
                    "join_node_ids": [],
                    "join_outstanding": [],
                    "join_waiters": {}
                }
            }))
            .unwrap();
        service
            .store
            .apply_recorded_runtime_transition(runtime_start, false)
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
                payload_json: Some(
                    json!({
                        "workflow_id": "workflow",
                        "conversation_id": "conversation",
                        "token_id": "owned-run",
                        "parent_token_id": null
                    })
                    .to_string(),
                ),
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

        let worker_result = json!({
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
                            }],
                            "index_jobs": [{
                                "job_id": "owner-check-job",
                                "namespace": "alpha",
                                "entity_kind": "node",
                                "entity_id": "owner-check-node",
                                "index_kind": "node_embedding",
                                "op": "UPSERT"
                            }]
                        }
                    }
                });
        let secret = "worker-result-retry-test-secret";
        let token = |subject: &str| {
            encode(
                &Header::new(Algorithm::HS256),
                &json!({
                    "sub": subject,
                    "user_id": subject,
                    "role": "rw",
                    "roles": ["reader"],
                    "ns": "alpha",
                    "storage_ns": "alpha",
                    "capabilities": ["write_graph"],
                    "exp": 4102444800_u64
                }),
                &EncodingKey::from_secret(secret.as_bytes()),
            )
            .unwrap()
        };
        let owner_token = token("run-owner");
        let other_owner_token = token("different-user");
        let (committed_tx, committed_rx) = tokio::sync::oneshot::channel();
        let application = Arc::new(HoldFirstWorkerResultResponse {
            inner: Arc::new(service.clone()),
            committed: std::sync::Mutex::new(Some(committed_tx)),
            held: std::sync::atomic::AtomicBool::new(false),
        });
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: vec!["reader".to_owned()],
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig {
                    mode: "jwt".to_owned(),
                    algorithm: Some("HS256".to_owned()),
                    key: Some(secret.to_owned()),
                    ..AuthConfig::default()
                },
            },
            application,
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let client = reqwest::Client::new();
        let base = format!("http://{address}");

        let denied = client
            .post(format!("{base}/internal/runtime/results"))
            .bearer_auth(&other_owner_token)
            .json(&worker_result)
            .send()
            .await
            .unwrap();
        assert_eq!(denied.status(), StatusCode::FORBIDDEN);
        assert_eq!(
            denied.json::<Value>().await.unwrap()["code"],
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

        let (fault_setup, fault_connection) =
            tokio_postgres::connect(&dsn, tokio_postgres::NoTls).await.unwrap();
        let fault_connection_task = tokio::spawn(fault_connection);
        fault_setup
            .batch_execute(&format!(
                "CREATE FUNCTION \"{schema}\".reject_worker_result_index_job() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'injected worker-result index-job failure'; END; $$; CREATE TRIGGER reject_worker_result_index_job BEFORE INSERT ON \"{schema}\".index_jobs FOR EACH ROW EXECUTE FUNCTION \"{schema}\".reject_worker_result_index_job();"
            ))
            .await
            .unwrap();
        drop(fault_setup);
        fault_connection_task.await.unwrap().unwrap();

        let run_events_before_fault = service
            .store
            .list_server_run_events("owned-run", 0, usize::MAX)
            .await
            .unwrap()
            .len();
        let failed = client
            .post(format!("{base}/internal/runtime/results"))
            .bearer_auth(&owner_token)
            .json(&worker_result)
            .send()
            .await
            .unwrap();
        assert_eq!(failed.status(), StatusCode::CONFLICT);
        assert_eq!(
            failed.json::<Value>().await.unwrap()["code"],
            "KOGWISTAR_RUNTIME_RESULT_CONFLICT"
        );
        let failed_events = service.store.replay_raw_events("alpha", 0, 100).await.unwrap();
        assert!(!failed_events
            .iter()
            .any(|event| event.event_id == "owner-check-event"));
        let failed_records = service
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
        assert!(failed_records.is_empty(), "failed transaction must roll back graph projections");
        assert!(service
            .store
            .list_index_jobs(Some("alpha"), None, None, None, None, 100)
            .await
            .unwrap()
            .iter()
            .all(|job| job.job_id != "owner-check-job"));
        assert_eq!(
            service
                .store
                .list_server_run_events("owned-run", 0, usize::MAX)
                .await
                .unwrap()
                .len(),
            run_events_before_fault,
            "failed transaction must not append a runtime event"
        );
        assert_eq!(
            service
                .store
                .get_projected_lane_message("lane-1")
                .await
                .unwrap()
                .unwrap()
                .status,
            "claimed",
            "failed transaction must leave the worker lane unacknowledged"
        );

        let (fault_cleanup, fault_connection) =
            tokio_postgres::connect(&dsn, tokio_postgres::NoTls).await.unwrap();
        let fault_connection_task = tokio::spawn(fault_connection);
        fault_cleanup
            .batch_execute(&format!(
                "DROP TRIGGER reject_worker_result_index_job ON \"{schema}\".index_jobs; DROP FUNCTION \"{schema}\".reject_worker_result_index_job();"
            ))
            .await
            .unwrap();
        drop(fault_cleanup);
        fault_connection_task.await.unwrap().unwrap();

        let request_body = serde_json::to_vec(&worker_result).unwrap();
        let mut disconnected_client = tokio::net::TcpStream::connect(address).await.unwrap();
        let request_head = format!(
            "POST /internal/runtime/results HTTP/1.1\r\nHost: {address}\r\nAuthorization: Bearer {owner_token}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            request_body.len()
        );
        disconnected_client
            .write_all(request_head.as_bytes())
            .await
            .unwrap();
        disconnected_client
            .write_all(&request_body)
            .await
            .unwrap();
        committed_rx.await.unwrap();
        drop(disconnected_client);

        assert_eq!(
            service
                .store
                .get_projected_lane_message("lane-1")
                .await
                .unwrap()
                .unwrap()
                .status,
            "completed",
            "the disconnected HTTP request must commit before its response is lost"
        );
        let run_events_after_commit = service
            .store
            .list_server_run_events("owned-run", 0, usize::MAX)
            .await
            .unwrap()
            .len();

        // The caller did not observe the first response and retries the exact
        // request. Graph, index intent, runtime event, and lane ack share one
        // PostgreSQL transaction and must remain single-application effects.
        let retry = client
            .post(format!("{base}/internal/runtime/results"))
            .bearer_auth(&owner_token)
            .json(&worker_result)
            .send()
            .await
            .unwrap();
        assert_eq!(retry.status(), StatusCode::OK);
        assert_eq!(
            service
                .store
                .list_server_run_events("owned-run", 0, usize::MAX)
                .await
                .unwrap()
                .len(),
            run_events_after_commit
        );
        let replayed_events = service
            .store
            .replay_raw_events("alpha", 0, 100)
            .await
            .unwrap();
        assert_eq!(
            replayed_events
                .iter()
                .filter(|event| event.event_id == "owner-check-event")
                .count(),
            1
        );
        let replayed_records = service
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
        assert_eq!(replayed_records.len(), 1);
        let replayed_jobs = service
            .store
            .list_index_jobs(Some("alpha"), None, None, None, None, 100)
            .await
            .unwrap();
        assert_eq!(
            replayed_jobs
                .iter()
                .filter(|job| job.job_id == "owner-check-job")
                .count(),
            1
        );
        assert_eq!(
            service
                .store
                .get_projected_lane_message("lane-1")
                .await
                .unwrap()
                .unwrap()
                .status,
            "completed"
        );
        server.abort();
    }

    #[tokio::test]
    #[ignore = "requires KOGWISTAR_TEST_PG_DSN; run explicitly against disposable PostgreSQL"]
    async fn postgres_authenticated_native_graph_operation_commits_and_replays() {
        let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN").unwrap();
        let schema = format!("rust_native_graph_{}", uuid::Uuid::new_v4().simple());
        let mut registry = crate::native_answer_persist::registry().unwrap();
        registry.register_graph_v2("app/graph", "v1", |request| {
            let batch = serde_json::from_value(json!({
                "contract_version": 2,
                "atomic": true,
                "default_acl": {"truth_graph": "knowledge"},
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
            }, Some(batch)))
        }).unwrap();
        let auth_path = std::env::temp_dir().join(format!(
            "kogwistar-native-graph-auth-{}.sqlite",
            uuid::Uuid::new_v4().simple()
        ));
        let auth_store = SqliteAuthStore::open(&auth_path).unwrap();
        auth_store
            .resolve_external_identity(ResolveExternalIdentity {
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
        drop(auth_store);
        let owner_auth_config = AuthConfig {
            auth_db_url: Some(format!("sqlite:///{}", auth_path.display())),
            ..AuthConfig::default()
        };
        let embedding_calls = Arc::new(Mutex::new(Vec::new()));
        let embedding_provider = Arc::new(FixedEmbeddingProvider {
            profile: kogwistar_engine::embedding::EmbeddingProfile {
                provider: "test".to_owned(),
                model: "fixed-native-search-v1".to_owned(),
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
            calls: embedding_calls.clone(),
            fail: None,
        });
        let service = PostgresRunApplicationService::from_dsn(&dsn, &schema)
            .unwrap()
            .with_native_operations(registry)
            .with_owner_auth_config(&owner_auth_config)
            .await
            .unwrap()
            .with_embedding_provider(embedding_provider);
        service.ensure_graph_type_schemas().await.unwrap();
        service.store.create_graph_schema(3, kogwistar_store_postgres::GraphTableNames::default()).await.unwrap();
        service.conversation_store.create_graph_schema(3, kogwistar_store_postgres::GraphTableNames::default()).await.unwrap();

        let secret = "native-graph-test-secret";
        let token = |subject: &str, namespace: &str, caps: &[&str]| encode(
            &Header::new(Algorithm::HS256),
            &json!({"sub": subject, "role": "rw", "roles": ["reader"], "ns": namespace,
                "storage_ns": namespace, "capabilities": caps, "exp": 4102444800_u64}),
            &EncodingKey::from_secret(secret.as_bytes()),
        ).unwrap();
        let owner = token("alice", "alpha", &["workflow.run.write", "write_graph"]);
        let owner_read = token("alice", "alpha", &["workflow.run.write", "write_graph", "read_graph"]);
        let other = token("bob", "alpha", &["workflow.run.write", "write_graph"]);
        let other_read = token("bob", "alpha", &["workflow.run.write", "read_graph"]);
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
        let graph_jobs = service.store.list_index_jobs(Some("alpha"), None, None, None, None, 100).await.unwrap();
        assert!(graph_jobs.iter().any(|job| job.job_id == "graph-job-native-graph-run"));
        let graph_events = service.store.replay_raw_events("alpha", 0, 100).await.unwrap();
        assert!(graph_events.iter().any(|event| event.event_id == "graph-event-native-graph-run"));

        for (id, entity_type, role) in [
            ("native-user-turn", "conversation_turn", "user"),
            ("native-assistant-turn", "assistant_turn", "assistant"),
        ] {
            service
                .conversation_store
                .apply_graph_mutation(GraphMutation {
                    scope: GraphScope {
                        namespace: "alpha".to_owned(),
                        workspace_id: None,
                        graph_space: None,
                    },
                    table: "gke_nodes".to_owned(),
                    entity_kind: "node".to_owned(),
                    event_id: format!("assistant-link-turn-{id}"),
                    op: "ADD".to_owned(),
                    payload: json!({"id": id}),
                    record: GraphRecord {
                        id: id.to_owned(),
                        document: Some(
                            json!({
                                "summary": if role == "user" { "What is the Rust-native path?" } else { "Native answer." },
                                "metadata": {
                                    "entity_type": entity_type,
                                    "conversation_id": "conv-native-assistant-link",
                                    "role": role,
                                    "lifecycle_status": "active"
                                }
                            })
                            .to_string(),
                        ),
                        metadata: serde_json::Map::from_iter([
                            ("entity_type".to_owned(), json!(entity_type)),
                            ("conversation_id".to_owned(), json!("conv-native-assistant-link")),
                            ("role".to_owned(), json!(role)),
                            ("lifecycle_status".to_owned(), json!("active")),
                        ]),
                        embedding: None,
                    },
                    embedding_dim: 0,
                })
                .await
                .unwrap();
            service
                .conversation_store
                .apply_graph_mutation(GraphMutation {
                    scope: GraphScope {
                        namespace: "alpha".to_owned(),
                        workspace_id: None,
                        graph_space: None,
                    },
                    table: "gke_nodes".to_owned(),
                    entity_kind: "node".to_owned(),
                    event_id: format!("assistant-link-acl-{id}"),
                    op: "ADD".to_owned(),
                    payload: json!({"id": format!("acl-{id}")}),
                    record: GraphRecord {
                        id: format!("acl-{id}"),
                        document: None,
                        metadata: serde_json::Map::from_iter([
                            ("entity_type".to_owned(), json!("acl_record")),
                            ("acl_truth_graph".to_owned(), json!("conversation")),
                            ("acl_target_grain".to_owned(), json!("node")),
                            ("acl_target_entity_id".to_owned(), json!(id)),
                            ("acl_version".to_owned(), json!(1)),
                            ("acl_mode".to_owned(), json!("private")),
                            ("owner_id".to_owned(), json!("alice")),
                        ]),
                        embedding: None,
                    },
                    embedding_dim: 0,
                })
                .await
                .unwrap();
        }
        let view_submit = client
            .post(format!("{base}/api/workflow/runs"))
            .bearer_auth(&owner_read)
            .json(&json!({
                "run_id": "native-answer-view-run",
                "workflow_id": "wf",
                "conversation_id": "conv-native-assistant-link",
                "turn_node_id": "native-user-turn",
                "start_node_id": "entry",
                "node_ops": {"entry": "rust-native/aa_get_view_and_question"},
                "initial_state": {
                    "conversation_id": "conv-native-assistant-link",
                    "user_id": "alice",
                    "turn_node_id": "native-user-turn"
                }
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(view_submit.status(), StatusCode::ACCEPTED);
        tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
        let view_claim = client
            .post(format!("{base}/internal/runtime/claim"))
            .bearer_auth(&owner_read)
            .json(&json!({
                "claimed_by": "rust-native-worker",
                "limit": 1,
                "lease_seconds": 60,
                "run_id": "native-answer-view-run",
                "recipient_ids": ["rust-native-worker"]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(view_claim.status(), StatusCode::OK);
        let view_work = view_claim.json::<Value>().await.unwrap()["work"][0].clone();
        let view_execution = json!({
            "handoff": {
                "message_id": view_work["message_id"],
                "claimed_by": view_work["claimed_by"],
                "run_id": view_work["run_id"],
                "step_id": view_work["step_id"],
                "correlation_id": view_work["correlation_id"]
            },
            "payload": view_work["payload"]
        });
        let view_result = client
            .post(format!("{base}/internal/runtime/native/execute"))
            .bearer_auth(&owner_read)
            .json(&view_execution)
            .send()
            .await
            .unwrap();
        let view_status = view_result.status();
        let view_body = view_result.text().await.unwrap();
        assert_eq!(view_status, StatusCode::OK, "{view_body}");
        let view_state = service
            .store
            .read_recorded_runtime_state(
                "native-answer-view-run",
                "wf",
                "conv-native-assistant-link",
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            view_state.state.get("question").and_then(Value::as_str),
            Some("What is the Rust-native path?")
        );
        assert_eq!(
            view_state.state.get("system_prompt").and_then(Value::as_str),
            Some("You are a helpful assistant. Answer the user using the conversation and any provided evidence.")
        );

        let link_submit = client
            .post(format!("{base}/api/workflow/runs"))
            .bearer_auth(&owner_read)
            .json(&json!({
                "run_id": "native-assistant-link-run",
                "workflow_id": "wf",
                "conversation_id": "conv-native-assistant-link",
                "start_node_id": "entry",
                "node_ops": {"entry": "rust-native/link_assistant_turn"},
                "initial_state": {
                    "conversation_id": "conv-native-assistant-link",
                    "user_id": "alice",
                    "turn_node_id": "native-user-turn",
                    "turn_index": 2,
                    "answer": {"response_node_id": "native-assistant-turn"},
                    "prev_turn_meta_summary": {
                        "prev_node_char_distance_from_last_summary": 18,
                        "prev_node_distance_from_last_summary": 2,
                        "tail_turn_index": 2
                    }
                }
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(link_submit.status(), StatusCode::ACCEPTED);
        tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
        let link_claim = client
            .post(format!("{base}/internal/runtime/claim"))
            .bearer_auth(&owner)
            .json(&json!({"claimed_by":"rust-native-worker", "limit":1, "lease_seconds":60,
                "run_id":"native-assistant-link-run", "recipient_ids":["rust-native-worker"]}))
            .send()
            .await
            .unwrap();
        assert_eq!(link_claim.status(), StatusCode::OK);
        let link_work = link_claim.json::<Value>().await.unwrap()["work"][0].clone();
        let link_execution = json!({
            "handoff": {
                "message_id": link_work["message_id"],
                "claimed_by": link_work["claimed_by"],
                "run_id": link_work["run_id"],
                "step_id": link_work["step_id"],
                "correlation_id": link_work["correlation_id"]
            },
            "payload": link_work["payload"]
        });
        let mut link_response_bodies = Vec::new();
        for _ in 0..2 {
            let link_result = client
                .post(format!("{base}/internal/runtime/native/execute"))
                .bearer_auth(&owner)
                .json(&link_execution)
                .send()
                .await
                .unwrap();
            let status = link_result.status();
            let body = link_result.text().await.unwrap();
            assert_eq!(status, StatusCode::OK, "{body}");
            link_response_bodies.push(body);
        }
        let linked_state = service
            .store
            .read_recorded_runtime_state(
                "native-assistant-link-run",
                "wf",
                "conv-native-assistant-link",
            )
            .await
            .unwrap()
            .unwrap();
        assert!(
            linked_state.state.contains_key("assistant_turn_edge_id"),
            "native link op failed to update run state: {:?}; responses: {:?}",
            linked_state.state,
            link_response_bodies
        );
        let linked_edge_id = linked_state.state["assistant_turn_edge_id"]
            .as_str()
            .unwrap();
        let linked_edges = service
            .conversation_store
            .graph_projection_records(GraphProjectionRead {
                scope: GraphScope {
                    namespace: "alpha".to_owned(),
                    workspace_id: None,
                    graph_space: None,
                },
                table: "gke_edges".to_owned(),
                ids: Some(vec![linked_edge_id.to_owned()]),
                metadata: MetadataFilter::default(),
                limit: 10,
            })
            .await
            .unwrap();
        assert_eq!(linked_edges.len(), 1);
        let linked_edge_doc: Value = serde_json::from_str(
            linked_edges[0].document.as_deref().unwrap(),
        )
        .unwrap();
        assert_eq!(linked_edge_doc["relation"], "next_turn");
        assert_eq!(linked_edge_doc["source_ids"], json!(["native-user-turn"]));
        assert_eq!(linked_edge_doc["target_ids"], json!(["native-assistant-turn"]));
        let link_events = service.conversation_store.replay_raw_events("alpha", 0, 500).await.unwrap();
        assert_eq!(
            link_events
                .iter()
                .filter(|event| event.entity_id == linked_edge_id)
                .count(),
            1,
            "retry must not append a duplicate assistant-turn edge event"
        );
        let link_jobs = service
            .conversation_store
            .list_index_jobs(Some("alpha"), None, None, None, None, 500)
            .await
            .unwrap();
        assert_eq!(
            link_jobs
                .iter()
                .filter(|job| job.entity_id == linked_edge_id)
                .count(),
            2,
            "the atomic edge write should create exactly the edge index intents"
        );

        for (id, entity_type, truth_graph) in [
            (
                "native-pointer-source",
                "knowledge_node",
                "knowledge",
            ),
            (
                "native-pointer-run-anchor",
                "agent_run",
                "conversation",
            ),
        ] {
            let conversation_id = (truth_graph == "conversation")
                .then_some("conv-native-pointer");
            let graph_store = if truth_graph == "conversation" {
                &service.conversation_store
            } else {
                &service.store
            };
            graph_store
                .apply_graph_mutation(GraphMutation {
                    scope: GraphScope {
                        namespace: "alpha".to_owned(),
                        workspace_id: None,
                        graph_space: None,
                    },
                    table: "gke_nodes".to_owned(),
                    entity_kind: "node".to_owned(),
                    event_id: format!("native-pointer-seed-{id}"),
                    op: "ADD".to_owned(),
                    payload: json!({"id":id}),
                    record: GraphRecord {
                        id: id.to_owned(),
                        document: Some(format!("document {id}")),
                        metadata: serde_json::Map::from_iter([
                            ("entity_type".to_owned(), json!(entity_type)),
                            ("conversation_id".to_owned(), json!(conversation_id)),
                            ("label".to_owned(), json!("Pointer source")),
                            ("summary".to_owned(), json!("Pointer source summary")),
                            ("type".to_owned(), json!("entity")),
                            ("canonical_entity_id".to_owned(), Value::Null),
                            ("lifecycle_status".to_owned(), json!("active")),
                            ("mentions".to_owned(), json!([])),
                        ]),
                        embedding: None,
                    },
                    embedding_dim: 0,
                })
                .await
                .unwrap();
            graph_store
                .apply_graph_mutation(GraphMutation {
                    scope: GraphScope {
                        namespace: "alpha".to_owned(),
                        workspace_id: None,
                        graph_space: None,
                    },
                    table: "gke_nodes".to_owned(),
                    entity_kind: "node".to_owned(),
                    event_id: format!("native-pointer-acl-{id}"),
                    op: "ADD".to_owned(),
                    payload: json!({"id":format!("acl-{id}")}),
                    record: GraphRecord {
                        id: format!("acl-{id}"),
                        document: None,
                        metadata: serde_json::Map::from_iter([
                            ("entity_type".to_owned(), json!("acl_record")),
                            ("acl_truth_graph".to_owned(), json!(truth_graph)),
                            ("acl_target_grain".to_owned(), json!("node")),
                            ("acl_target_entity_id".to_owned(), json!(id)),
                            ("acl_version".to_owned(), json!(1)),
                            ("acl_mode".to_owned(), json!("private")),
                            ("owner_id".to_owned(), json!("alice")),
                        ]),
                        embedding: None,
                    },
                    embedding_dim: 0,
                })
                .await
                .unwrap();
        }
        let pointer_submit = client
            .post(format!("{base}/api/workflow/runs"))
            .bearer_auth(&owner_read)
            .json(&json!({
                "run_id":"native-pointer-projection-run",
                "workflow_id":"wf",
                "conversation_id":"conv-native-pointer",
                "start_node_id":"entry",
                "node_ops":{"entry":"rust-native/aa_project_pointers"},
                "initial_state":{
                    "conversation_id":"conv-native-pointer",
                    "user_id":"alice",
                    "run_node_id":"native-pointer-run-anchor",
                    "used_node_ids":["native-pointer-source"]
                }
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(pointer_submit.status(), StatusCode::ACCEPTED);
        tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
        let pointer_claim = client
            .post(format!("{base}/internal/runtime/claim"))
            .bearer_auth(&owner)
            .json(&json!({"claimed_by":"rust-native-worker", "limit":1, "lease_seconds":60,
                "run_id":"native-pointer-projection-run", "recipient_ids":["rust-native-worker"]}))
            .send()
            .await
            .unwrap();
        assert_eq!(pointer_claim.status(), StatusCode::OK);
        let pointer_work = pointer_claim.json::<Value>().await.unwrap()["work"][0].clone();
        let pointer_execution = json!({
            "handoff": {
                "message_id": pointer_work["message_id"],
                "claimed_by": pointer_work["claimed_by"],
                "run_id": pointer_work["run_id"],
                "step_id": pointer_work["step_id"],
                "correlation_id": pointer_work["correlation_id"]
            },
            "payload": pointer_work["payload"]
        });
        for replay_principal in [&owner, &no_write] {
            let response = client
                .post(format!("{base}/internal/runtime/native/execute"))
                .bearer_auth(replay_principal)
                .json(&pointer_execution)
                .send()
                .await
                .unwrap();
            let status = response.status();
            let body = response.text().await.unwrap();
            assert_eq!(status, StatusCode::OK, "{body}");
        }
        let pointer_id = crate::native_answer_persist::stable_conversation_id(
            "ptr",
            &[
                ("scope", "conv:conv-native-pointer"),
                ("pointer_kind", "kg_node"),
                ("target_kind", "node"),
                ("target_id", "native-pointer-source"),
            ],
        );
        let pointer_records = service
            .conversation_store
            .graph_projection_records(GraphProjectionRead {
                scope: GraphScope {
                    namespace: "alpha".to_owned(),
                    workspace_id: None,
                    graph_space: None,
                },
                table: "gke_nodes".to_owned(),
                ids: Some(vec![pointer_id.clone()]),
                metadata: MetadataFilter::default(),
                limit: 10,
            })
            .await
            .unwrap();
        assert_eq!(pointer_records.len(), 1);
        let pointer_document: Value =
            serde_json::from_str(pointer_records[0].document.as_deref().unwrap()).unwrap();
        assert_eq!(pointer_document["properties"]["target_id"], "native-pointer-source");
        let usage_edge_id = crate::native_answer_persist::stable_conversation_id(
            "e",
            &[
                ("scope", "conv:conv-native-pointer"),
                ("rel", "used_evidence"),
                ("src", "native-pointer-run-anchor"),
                ("dst", &pointer_id),
            ],
        );
        let usage_edges = service
            .conversation_store
            .graph_projection_records(GraphProjectionRead {
                scope: GraphScope {
                    namespace: "alpha".to_owned(),
                    workspace_id: None,
                    graph_space: None,
                },
                table: "gke_edges".to_owned(),
                ids: Some(vec![usage_edge_id.clone()]),
                metadata: MetadataFilter::default(),
                limit: 10,
            })
            .await
            .unwrap();
        assert_eq!(usage_edges.len(), 1);
        assert_eq!(
            service
                .conversation_store
                .replay_raw_events("alpha", 0, 1000)
                .await
                .unwrap()
                .iter()
                .filter(|event| event.entity_id == pointer_id || event.entity_id == usage_edge_id)
                .count(),
            2,
            "retry must not duplicate the pointer or its usage edge event"
        );
        let pointer_state = service
            .store
            .read_recorded_runtime_state(
                "native-pointer-projection-run",
                "wf",
                "conv-native-pointer",
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(pointer_state.state["projected_pointer_ids"], json!([pointer_id]));

        let add_candidate = |id: &str, level: i64, lifecycle_status: &str| {
            let mut metadata = serde_json::Map::from_iter([
                ("entity_type".to_owned(), json!("knowledge_node")),
                ("level_from_root".to_owned(), json!(level)),
                ("label".to_owned(), json!(id)),
                ("summary".to_owned(), json!(format!("summary {id}"))),
                ("lifecycle_status".to_owned(), json!(lifecycle_status)),
            ]);
            metadata.insert("mentions".to_owned(), json!([]));
            GraphMutation {
                scope: GraphScope {
                    namespace: "alpha".to_owned(),
                    workspace_id: None,
                    graph_space: None,
                },
                table: "gke_nodes".to_owned(),
                entity_kind: "node".to_owned(),
                event_id: format!("retrieval-node-{id}"),
                op: "ADD".to_owned(),
                payload: json!({"id": id}),
                record: GraphRecord {
                    id: id.to_owned(),
                    document: Some(format!("document {id}")),
                    metadata,
                    embedding: Some(vec![0.6, 0.8, 0.0]),
                },
                embedding_dim: 3,
            }
        };
        let add_acl = |id: &str, mode: &str, owner: &str| GraphMutation {
            scope: GraphScope {
                namespace: "alpha".to_owned(),
                workspace_id: None,
                graph_space: None,
            },
            table: "gke_nodes".to_owned(),
            entity_kind: "node".to_owned(),
            event_id: format!("retrieval-acl-{id}"),
            op: "ADD".to_owned(),
            payload: json!({"id": format!("acl-{id}")}),
            record: GraphRecord {
                id: format!("acl-{id}"),
                document: None,
                metadata: serde_json::Map::from_iter([
                    ("entity_type".to_owned(), json!("acl_record")),
                    ("acl_truth_graph".to_owned(), json!("knowledge")),
                    ("acl_target_grain".to_owned(), json!("node")),
                    ("acl_target_entity_id".to_owned(), json!(id)),
                    ("acl_version".to_owned(), json!(1)),
                    ("acl_mode".to_owned(), json!(mode)),
                    ("owner_id".to_owned(), json!(owner)),
                ]),
                embedding: None,
            },
            embedding_dim: 0,
        };
        for mutation in [
            add_candidate("retrieval-own", 0, "active"),
            add_acl("retrieval-own", "private", "alice"),
            add_candidate("retrieval-foreign", 0, "active"),
            add_acl("retrieval-foreign", "private", "bob"),
            add_candidate("retrieval-deep", 5, "active"),
            add_acl("retrieval-deep", "public", "alice"),
            add_candidate("retrieval-tombstone", 0, "tombstoned"),
            add_acl("retrieval-tombstone", "public", "alice"),
        ] {
            service.store.apply_graph_mutation(mutation).await.unwrap();
        }
        let retrieval_submit = client
            .post(format!("{base}/api/workflow/runs"))
            .bearer_auth(&owner_read)
            .json(&json!({
                "run_id": "native-candidate-retrieval-run",
                "workflow_id": "wf",
                "conversation_id": "conv-native-candidate-retrieval",
                "start_node_id": "entry",
                "node_ops": {"entry": "rust-native/aa_retrieve_candidates"},
                "initial_state": {
                    "question": "native candidate search",
                    "max_candidates": 20,
                    "max_retrieval_level": 4
                }
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(retrieval_submit.status(), StatusCode::ACCEPTED);
        tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
        assert!(service.execute_inline_native_work_once().await.unwrap());
        let retrieval_state = service
            .store
            .read_recorded_runtime_state(
                "native-candidate-retrieval-run",
                "wf",
                "conv-native-candidate-retrieval",
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            retrieval_state.state["candidates"],
            json!([{
                "id": "retrieval-own",
                "label": "retrieval-own",
                "summary": "summary retrieval-own",
                "doc": "document retrieval-own"
            }])
        );
        assert_eq!(
            *embedding_calls.lock().unwrap(),
            vec![vec!["native candidate search".to_owned()]]
        );
        assert!(retrieval_state
            .state
            .get("authorized_candidate_records")
            .is_none());

        let foreign_principal = json!({
            "user_id": "bob",
            "actor_id": "bob",
            "sub": "bob",
            "role": "rw",
            "ns": "alpha",
            "storage_ns": "alpha",
            "capabilities": ["read_graph"]
        });
        assert!(service
            .authorized_evidence_records(
                &foreign_principal,
                &["native-node".to_owned()],
                &[],
                "knowledge",
            )
            .await
            .is_err(), "private knowledge ACL must hide the selected node from another principal");
        let lane = service.store.get_projected_lane_message(work["message_id"].as_str().unwrap()).await.unwrap().unwrap();
        assert_eq!(lane.status, "completed");
        assert_eq!(service.store.get_server_run("native-graph-run").await.unwrap().unwrap().status, "succeeded");

        let evidence_submit = client
            .post(format!("{base}/api/workflow/runs"))
            .bearer_auth(&owner_read)
            .json(&json!({
                "run_id": "native-evidence-materialize-run",
                "workflow_id": "wf",
                "conversation_id": "conv-native-evidence-materialize",
                "start_node_id": "entry",
                "node_ops": {"entry": "rust-native/aa_materialize_evidence_pack"},
                "initial_state": {
                    "used_node_ids": ["native-node"],
                    "used_edge_ids": [],
                    "materialize_depth": "deep"
                }
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(evidence_submit.status(), StatusCode::ACCEPTED);
        tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
        let owner_guard_submit = client
            .post(format!("{base}/api/workflow/runs"))
            .bearer_auth(&owner_read)
            .json(&json!({
                "run_id": "native-evidence-owner-guard-run",
                "workflow_id": "wf",
                "conversation_id": "conv-native-evidence-owner-guard",
                "start_node_id": "entry",
                "node_ops": {"entry": "rust-native/aa_materialize_evidence_pack"},
                "initial_state": {
                    "used_node_ids": ["native-node"],
                    "used_edge_ids": [],
                    "materialize_depth": "deep"
                }
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(owner_guard_submit.status(), StatusCode::ACCEPTED);
        tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
        let owner_guard_claim = client
            .post(format!("{base}/internal/runtime/claim"))
            .bearer_auth(&owner_read)
            .json(&json!({"claimed_by":"rust-native-worker", "limit":1, "lease_seconds":60,
                "run_id":"native-evidence-owner-guard-run", "recipient_ids":["rust-native-worker"]}))
            .send()
            .await
            .unwrap();
        assert_eq!(owner_guard_claim.status(), StatusCode::OK);
        let owner_guard_work = owner_guard_claim.json::<Value>().await.unwrap()["work"][0].clone();
        let owner_guard_execution = json!({
            "handoff": {
                "message_id": owner_guard_work["message_id"],
                "claimed_by": owner_guard_work["claimed_by"],
                "run_id": owner_guard_work["run_id"],
                "step_id": owner_guard_work["step_id"],
                "correlation_id": owner_guard_work["correlation_id"]
            },
            "payload": owner_guard_work["payload"]
        });
        let wrong_evidence_owner = client
            .post(format!("{base}/internal/runtime/native/execute"))
            .bearer_auth(&other_read)
            .json(&owner_guard_execution)
            .send()
            .await
            .unwrap();
        let wrong_evidence_owner_status = wrong_evidence_owner.status();
        let wrong_evidence_owner_body = wrong_evidence_owner.text().await.unwrap();
        assert_eq!(
            wrong_evidence_owner_status,
            StatusCode::FORBIDDEN,
            "{wrong_evidence_owner_body}"
        );
        assert_eq!(
            service
                .store
                .get_projected_lane_message(owner_guard_work["message_id"].as_str().unwrap())
                .await
                .unwrap()
                .unwrap()
                .status,
            "claimed",
            "a non-owner must not mutate or acknowledge the run"
        );
        assert!(service.execute_inline_native_work_once().await.unwrap());
        let evidence_state = service
            .store
            .read_recorded_runtime_state(
                "native-evidence-materialize-run",
                "wf",
                "conv-native-evidence-materialize",
            )
            .await
            .unwrap()
            .unwrap();
        let Some(evidence_pack) = evidence_state.state.get("_rt").and_then(|state| state.get("evidence_pack")) else {
            panic!(
                "evidence pack missing; state={:?}; run={:?}",
                evidence_state.state,
                service.store.get_server_run("native-evidence-materialize-run").await.unwrap().unwrap()
            );
        };
        assert_eq!(evidence_pack["nodes"][0]["text"], "native graph");
        assert!(evidence_state.state.get("authorized_evidence_records").is_none());
        let evidence_lanes = service
            .store
            .list_projected_lane_messages(LaneMessageFilter {
                correlation_id: Some("native-evidence-materialize-run".to_owned()),
                limit: 10,
                ..LaneMessageFilter::default()
            })
            .await
            .unwrap();
        assert_eq!(evidence_lanes.len(), 1);
        assert_eq!(evidence_lanes[0].status, "completed");
        assert_eq!(
            service
                .store
                .get_server_run("native-evidence-materialize-run")
                .await
                .unwrap()
                .unwrap()
                .status,
            "succeeded"
        );

        let memory_submit = client
            .post(format!("{base}/api/workflow/runs"))
            .bearer_auth(&owner)
            .json(&json!({
                "run_id": "native-memory-pin-run",
                "workflow_id": "wf",
                "conversation_id": "conv-native-memory-pin",
                "turn_node_id": "turn-native-memory-pin",
                "start_node_id": "entry",
                "node_ops": {"entry": "rust-native/memory_pin"},
                "initial_state": {
                    "conversation_id": "conv-native-memory-pin",
                    "turn_node_id": "turn-native-memory-pin",
                    "turn_index": 7,
                    "mem_id": "native-memory-context",
                    "user_id": "forged-owner",
                    "_rt_run_owner_id": "forged-owner",
                    "_rt_graph_scope": {
                        "storage_namespace": "beta",
                        "security_scope": "forged-scope"
                    },
                    "self_span": {
                        "collection_page_url": "conversation/1",
                        "document_page_url": "conversation/1",
                        "doc_id": "conv-native-memory-pin",
                        "insertion_method": "conversation",
                        "start_char": 0,
                        "end_char": 3,
                        "excerpt": "ask"
                    },
                    "prev_turn_meta_summary": {
                        "prev_node_char_distance_from_last_summary": 2,
                        "prev_node_distance_from_last_summary": 1,
                        "tail_turn_index": 6
                    },
                    "memory": {
                        "memory_context_text": "pinned context",
                        "selected": {
                            "nodes": [{"id": "memory-source-node"}],
                            "edges": [{"id": "memory-source-edge"}]
                        }
                    }
                }
            }))
            .send()
            .await
            .unwrap();
        let memory_submit_status = memory_submit.status();
        let memory_submit_body = memory_submit.text().await.unwrap();
        assert_eq!(memory_submit_status, StatusCode::ACCEPTED, "{memory_submit_body}");
        tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
        let memory_claim = client
            .post(format!("{base}/internal/runtime/claim"))
            .bearer_auth(&owner)
            .json(&json!({
                "claimed_by": "rust-native-worker",
                "limit": 1,
                "lease_seconds": 60,
                "run_id": "native-memory-pin-run",
                "recipient_ids": ["rust-native-worker"]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(memory_claim.status(), StatusCode::OK);
        let memory_work = memory_claim.json::<Value>().await.unwrap()["work"][0].clone();
        let memory_execution = json!({
            "handoff": {
                "message_id": memory_work["message_id"],
                "claimed_by": memory_work["claimed_by"],
                "run_id": memory_work["run_id"],
                "step_id": memory_work["step_id"],
                "correlation_id": memory_work["correlation_id"]
            },
            "payload": memory_work["payload"]
        });
        let memory_result = client
            .post(format!("{base}/internal/runtime/native/execute"))
            .bearer_auth(&owner)
            .json(&memory_execution)
            .send()
            .await
            .unwrap();
        let memory_result_status = memory_result.status();
        let memory_result_body = memory_result.text().await.unwrap();
        assert_eq!(memory_result_status, StatusCode::OK, "{memory_result_body}");

        let memory_node = service
            .conversation_store
            .graph_projection_records(GraphProjectionRead {
                scope: GraphScope {
                    namespace: "alpha".to_owned(),
                    workspace_id: None,
                    graph_space: None,
                },
                table: "gke_nodes".to_owned(),
                ids: Some(vec!["native-memory-context".to_owned()]),
                metadata: MetadataFilter::default(),
                limit: 10,
            })
            .await
            .unwrap();
        assert_eq!(memory_node.len(), 1);
        assert_eq!(memory_node[0].metadata["user_id"], "alice");
        assert_eq!(memory_node[0].metadata["doc_id"], "native-memory-context");
        let spoofed_namespace_node = service
            .conversation_store
            .graph_projection_records(GraphProjectionRead {
                scope: GraphScope {
                    namespace: "beta".to_owned(),
                    workspace_id: None,
                    graph_space: None,
                },
                table: "gke_nodes".to_owned(),
                ids: Some(vec!["native-memory-context".to_owned()]),
                metadata: MetadataFilter::default(),
                limit: 10,
            })
            .await
            .unwrap();
        assert!(spoofed_namespace_node.is_empty());
        let memory_edges = service
            .conversation_store
            .graph_projection_records(GraphProjectionRead {
                scope: GraphScope {
                    namespace: "alpha".to_owned(),
                    workspace_id: None,
                    graph_space: None,
                },
                table: "gke_edges".to_owned(),
                ids: Some(vec![
                    "turn-native-memory-pin::hm::native-memory-context".to_owned(),
                    "native-memory-context::n::memory-source-node".to_owned(),
                    "native-memory-context::e::memory-source-edge".to_owned(),
                ]),
                metadata: MetadataFilter::default(),
                limit: 10,
            })
            .await
            .unwrap();
        assert_eq!(memory_edges.len(), 3);
        assert!(memory_edges.iter().all(|edge| {
            edge.metadata
                .get("user_id")
                .and_then(Value::as_str)
                .is_none()
                && edge.metadata["owner_agent_id"] == "alice"
                && edge.metadata["visibility"] == "private"
                && edge.metadata["doc_id"] == "conv:conv-native-memory-pin"
        }));
        let memory_acl = service
            .conversation_store
            .graph_acl_records_for_entities(
                &GraphScope {
                    namespace: "alpha".to_owned(),
                    workspace_id: None,
                    graph_space: None,
                },
                "conversation",
                &[
                    "native-memory-context".to_owned(),
                    "turn-native-memory-pin::hm::native-memory-context".to_owned(),
                    "native-memory-context::n::memory-source-node".to_owned(),
                    "native-memory-context::e::memory-source-edge".to_owned(),
                ],
            )
            .await
            .unwrap();
        assert_eq!(memory_acl.len(), 12);
        assert!(memory_acl.iter().all(|record| {
            record.metadata["acl_mode"] == "private"
                && record.metadata["owner_id"] == "alice"
                && record.metadata["created_by"] == "alice"
        }));
        let memory_entity_ids = [
            "native-memory-context",
            "turn-native-memory-pin::hm::native-memory-context",
            "native-memory-context::n::memory-source-node",
            "native-memory-context::e::memory-source-edge",
        ];
        let memory_index_jobs = service
            .conversation_store
            .list_index_jobs(Some("alpha"), None, None, None, None, 100)
            .await
            .unwrap()
            .into_iter()
            .filter(|job| memory_entity_ids.contains(&job.entity_id.as_str()))
            .collect::<Vec<_>>();
        assert_eq!(memory_index_jobs.len(), 12);
        assert!(memory_index_jobs.iter().any(|job| {
            job.entity_kind == "edge" && job.index_kind == "node_embedding"
        }));
        let memory_lane = service
            .store
            .get_projected_lane_message(memory_work["message_id"].as_str().unwrap())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(memory_lane.status, "completed");
        assert_eq!(
            service
                .store
                .get_server_run("native-memory-pin-run")
                .await
                .unwrap()
                .unwrap()
                .status,
            "succeeded"
        );

        let answer_submit = client
            .post(format!("{base}/api/workflow/runs"))
            .bearer_auth(&owner)
            .json(&json!({
                "run_id": "native-answer-persist-run",
                "workflow_id": "wf",
                "conversation_id": "conv-native-answer-persist",
                "start_node_id": "persist",
                "node_ops": {"persist": "rust-native/aa_persist_response"},
                "initial_state": {
                    "conversation_id": "conv-native-answer-persist",
                    "run_node_id": "run-node-native-answer-persist",
                    "turn_index": 4,
                    "answer": {"text": "Native response 🌱", "claim_citations": []},
                    "used_node_ids": ["source-node-1"],
                    "projected_pointer_ids": ["pointer-node-1"],
                    "evaluation": {"is_sufficient": true},
                    "prev_turn_meta_summary": {
                        "prev_node_char_distance_from_last_summary": 10,
                        "prev_node_distance_from_last_summary": 2,
                        "tail_turn_index": 4
                    }
                }
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(answer_submit.status(), StatusCode::ACCEPTED);
        tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
        let answer_claim = client
            .post(format!("{base}/internal/runtime/claim"))
            .bearer_auth(&owner)
            .json(&json!({
                "claimed_by": "rust-native-worker",
                "limit": 1,
                "lease_seconds": 60,
                "run_id": "native-answer-persist-run",
                "recipient_ids": ["rust-native-worker"]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(answer_claim.status(), StatusCode::OK);
        let answer_work = answer_claim.json::<Value>().await.unwrap()["work"][0].clone();
        let answer_execution = json!({
            "handoff": {
                "message_id": answer_work["message_id"],
                "claimed_by": answer_work["claimed_by"],
                "run_id": answer_work["run_id"],
                "step_id": answer_work["step_id"],
                "correlation_id": answer_work["correlation_id"]
            },
            "payload": answer_work["payload"]
        });
        let answer_response = client
            .post(format!("{base}/internal/runtime/native/execute"))
            .bearer_auth(&owner)
            .json(&answer_execution)
            .send()
            .await
            .unwrap();
        let answer_status = answer_response.status();
        let answer_body = answer_response.text().await.unwrap();
        assert_eq!(answer_status, StatusCode::OK, "{answer_body}");
        let answer_turn_id = "ptr:uvpzlbcvkjixsq7sbwra";
        let answer_edge_id = "e:t7ewoy536jqdo3qakhta";
        let answer_nodes = service
            .conversation_store
            .graph_projection_records(GraphProjectionRead {
                scope: GraphScope {
                    namespace: "alpha".to_owned(),
                    workspace_id: None,
                    graph_space: None,
                },
                table: "gke_nodes".to_owned(),
                ids: Some(vec![answer_turn_id.to_owned()]),
                metadata: MetadataFilter::default(),
                limit: 10,
            })
            .await
            .unwrap();
        assert_eq!(answer_nodes.len(), 1);
        assert_eq!(answer_nodes[0].metadata["_class_name"], "ConversationNode");
        assert_eq!(answer_nodes[0].metadata["role"], "assistant");
        assert_eq!(answer_nodes[0].metadata["conversation_id"], "conv-native-answer-persist");
        assert!(answer_nodes[0]
            .document
            .as_deref()
            .unwrap()
            .contains("Native response 🌱"));
        let answer_edges = service
            .conversation_store
            .graph_projection_records(GraphProjectionRead {
                scope: GraphScope {
                    namespace: "alpha".to_owned(),
                    workspace_id: None,
                    graph_space: None,
                },
                table: "gke_edges".to_owned(),
                ids: Some(vec![answer_edge_id.to_owned()]),
                metadata: MetadataFilter::default(),
                limit: 10,
            })
            .await
            .unwrap();
        assert_eq!(answer_edges.len(), 1);
        assert_eq!(answer_edges[0].metadata["_class_name"], "ConversationEdge");
        assert_eq!(answer_edges[0].metadata["doc_id"], "conv:conv-native-answer-persist");
        let answer_entity_ids = [answer_turn_id.to_owned(), answer_edge_id.to_owned()];
        let answer_acl = service
            .conversation_store
            .graph_acl_records_for_entities(
                &GraphScope {
                    namespace: "alpha".to_owned(),
                    workspace_id: None,
                    graph_space: None,
                },
                "conversation",
                &answer_entity_ids,
            )
            .await
            .unwrap();
        assert_eq!(answer_acl.len(), 6);
        assert!(answer_acl.iter().all(|record| {
            record.metadata["acl_mode"] == "private"
                && record.metadata["owner_id"] == "alice"
        }));
        let answer_jobs = service
            .conversation_store
            .list_index_jobs(Some("alpha"), None, None, None, None, 100)
            .await
            .unwrap()
            .into_iter()
            .filter(|job| answer_entity_ids.contains(&job.entity_id))
            .collect::<Vec<_>>();
        assert_eq!(answer_jobs.len(), 6);
        assert_eq!(
            service
                .store
                .get_projected_lane_message(answer_work["message_id"].as_str().unwrap())
                .await
                .unwrap()
                .unwrap()
                .status,
            "completed"
        );
        assert_eq!(
            service
                .store
                .get_server_run("native-answer-persist-run")
                .await
                .unwrap()
                .unwrap()
                .status,
            "succeeded"
        );

        let prepare_submit = client
            .post(format!("{base}/api/workflow/runs"))
            .bearer_auth(&owner)
            .json(&json!({
                "run_id": "native-answer-prepare-run",
                "workflow_id": "wf",
                "conversation_id": "conv-native-aa-prepare",
                "start_node_id": "prepare",
                "node_ops": {"prepare": "rust-native/aa_prepare"},
                "initial_state": {
                    "conversation_id": "conv-native-aa-prepare",
                    "run_id": "agent-run-1"
                }
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(prepare_submit.status(), StatusCode::ACCEPTED);
        tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
        let prepare_claim = client
            .post(format!("{base}/internal/runtime/claim"))
            .bearer_auth(&owner)
            .json(&json!({
                "claimed_by": "rust-native-worker",
                "limit": 1,
                "lease_seconds": 60,
                "run_id": "native-answer-prepare-run",
                "recipient_ids": ["rust-native-worker"]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(prepare_claim.status(), StatusCode::OK);
        let prepare_work = prepare_claim.json::<Value>().await.unwrap()["work"][0].clone();
        let prepare_execution = json!({
            "handoff": {
                "message_id": prepare_work["message_id"],
                "claimed_by": prepare_work["claimed_by"],
                "run_id": prepare_work["run_id"],
                "step_id": prepare_work["step_id"],
                "correlation_id": prepare_work["correlation_id"]
            },
            "payload": prepare_work["payload"]
        });
        let prepare_response = client
            .post(format!("{base}/internal/runtime/native/execute"))
            .bearer_auth(&owner)
            .json(&prepare_execution)
            .send()
            .await
            .unwrap();
        let prepare_status = prepare_response.status();
        let prepare_body = prepare_response.text().await.unwrap();
        assert_eq!(prepare_status, StatusCode::OK, "{prepare_body}");
        let run_anchor_id = "ptr:7apfcktwl35dka7476tq";
        let run_anchor = service
            .conversation_store
            .graph_projection_records(GraphProjectionRead {
                scope: GraphScope {
                    namespace: "alpha".to_owned(),
                    workspace_id: None,
                    graph_space: None,
                },
                table: "gke_nodes".to_owned(),
                ids: Some(vec![run_anchor_id.to_owned()]),
                metadata: MetadataFilter::default(),
                limit: 10,
            })
            .await
            .unwrap();
        assert_eq!(run_anchor.len(), 1);
        assert_eq!(run_anchor[0].metadata["_class_name"], "ConversationNode");
        assert_eq!(run_anchor[0].metadata["entity_type"], "agent_run");
        let run_anchor_acl = service
            .conversation_store
            .graph_acl_records_for_entities(
                &GraphScope {
                    namespace: "alpha".to_owned(),
                    workspace_id: None,
                    graph_space: None,
                },
                "conversation",
                &[run_anchor_id.to_owned()],
            )
            .await
            .unwrap();
        assert_eq!(run_anchor_acl.len(), 3);
        assert!(run_anchor_acl.iter().all(|record| {
            record.metadata["acl_mode"] == "private"
                && record.metadata["owner_id"] == "alice"
        }));
        assert_eq!(
            service
                .store
                .get_projected_lane_message(prepare_work["message_id"].as_str().unwrap())
                .await
                .unwrap()
                .unwrap()
                .status,
            "completed"
        );
        assert_eq!(
            service
                .store
                .get_server_run("native-answer-prepare-run")
                .await
                .unwrap()
                .unwrap()
                .status,
            "succeeded"
        );

        let selector_submit = client
            .post(format!("{base}/api/workflow/runs"))
            .bearer_auth(&owner)
            .json(&json!({
                "run_id": "native-answer-bm25-run",
                "workflow_id": "wf",
                "conversation_id": "conv-native-bm25",
                "start_node_id": "select",
                "node_ops": {"select": "rust-native/aa_select_used_evidence"},
                "initial_state": {
                    "evidence_selector": "bm25",
                    "question": "Explain graph vector retrieval none",
                    "max_used": 2,
                    "candidates": [
                        {"id": "node-first", "label": "Graph query", "summary": "vector access"},
                        {"id": "node-best", "label": "retrieval", "summary": "query vector graph"},
                        {"id": "node-irrelevant", "label": "unrelated", "summary": "nothing"},
                        {"id": "node-tied", "label": "Graph", "summary": "vector"},
                        {"id": "node-missing-fields"}
                    ]
                }
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(selector_submit.status(), StatusCode::ACCEPTED);
        tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
        let selector_claim = client
            .post(format!("{base}/internal/runtime/claim"))
            .bearer_auth(&owner)
            .json(&json!({
                "claimed_by": "rust-native-worker",
                "limit": 1,
                "lease_seconds": 60,
                "run_id": "native-answer-bm25-run",
                "recipient_ids": ["rust-native-worker"]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(selector_claim.status(), StatusCode::OK);
        let selector_work = selector_claim.json::<Value>().await.unwrap()["work"][0].clone();
        let selector_execution = json!({
            "handoff": {
                "message_id": selector_work["message_id"],
                "claimed_by": selector_work["claimed_by"],
                "run_id": selector_work["run_id"],
                "step_id": selector_work["step_id"],
                "correlation_id": selector_work["correlation_id"]
            },
            "payload": selector_work["payload"]
        });
        assert_eq!(selector_execution["payload"]["executor_target"], "rust-native");
        let selector_response = client
            .post(format!("{base}/internal/runtime/native/execute"))
            .bearer_auth(&owner)
            .json(&selector_execution)
            .send()
            .await
            .unwrap();
        let selector_status = selector_response.status();
        let selector_body = selector_response.text().await.unwrap();
        assert_eq!(selector_status, StatusCode::OK, "{selector_body}");
        let selector_state = service
            .store
            .read_recorded_runtime_state(
                "native-answer-bm25-run",
                "wf",
                "conv-native-bm25",
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            selector_state.state["selection"]["used_node_ids"],
            json!(["node-best", "node-first"])
        );
        assert_eq!(
            selector_state.state["op_log"],
            json!(["aa_select_used_evidence:2"])
        );
        assert_eq!(
            service
                .store
                .get_projected_lane_message(selector_work["message_id"].as_str().unwrap())
                .await
                .unwrap()
                .unwrap()
                .status,
            "completed"
        );
        assert_eq!(
            service
                .store
                .get_server_run("native-answer-bm25-run")
                .await
                .unwrap()
                .unwrap()
                .status,
            "succeeded"
        );
        server.abort();
        let (cleanup, connection) = tokio_postgres::connect(&dsn, tokio_postgres::NoTls).await.unwrap();
        let task = tokio::spawn(connection);
        cleanup.batch_execute(&format!(
            "DROP SCHEMA IF EXISTS \"{schema}_conversation\" CASCADE; \
             DROP SCHEMA IF EXISTS \"{schema}_workflow\" CASCADE; \
             DROP SCHEMA IF EXISTS \"{schema}_wisdom\" CASCADE; \
             DROP SCHEMA IF EXISTS \"{schema}\" CASCADE"
        )).await.unwrap();
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
                "capabilities": ["workflow.run.write", "write_graph"]
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
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use tower::ServiceExt;

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
        service.ensure_graph_type_schemas().await.unwrap();
        service
            .conversation_store
            .create_graph_schema(3, kogwistar_store_postgres::GraphTableNames::default())
            .await
            .unwrap();

        let conversation_id = "send-message-conversation";
        let root_id = "send-message-root";
        service
            .conversation_store
            .apply_graph_mutation(GraphMutation {
                scope: GraphScope {
                    namespace: "alpha".to_owned(),
                    workspace_id: Some("workspace-a".to_owned()),
                    graph_space: Some("conversation".to_owned()),
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
            .conversation_store
            .apply_graph_mutation(GraphMutation {
                scope: GraphScope {
                    namespace: "alpha".to_owned(),
                    workspace_id: Some("workspace-a".to_owned()),
                    graph_space: Some("conversation".to_owned()),
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
                "CREATE FUNCTION \"{schema}_conversation\".reject_send_message_index_job() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'injected send_message index failure'; END; $$; CREATE TRIGGER reject_send_message_index_job BEFORE INSERT ON \"{schema}_conversation\".index_jobs FOR EACH ROW EXECUTE FUNCTION \"{schema}_conversation\".reject_send_message_index_job();"
            ))
            .await
            .unwrap();
        drop(fault_client);
        connection_task.await.unwrap().unwrap();

        let make_request = |user_id: Option<&str>| {
            let mut args = json!({
                "conversation_id": conversation_id,
                "text": "A user question",
                "workflow_id": "send-message-admission-test"
            });
            if let Some(user_id) = user_id {
                args["user_id"] = json!(user_id);
            }
            ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/syscall/v1/send_message".to_owned(),
                body: serde_json::to_vec(&json!({
                "version": "v1",
                "op": "send_message",
                "args": args
                }))
                .unwrap(),
                principal: json!({
                    "user_id": "request-actor",
                    "sub": "request-actor@example.test",
                    "role": "rw",
                    "ns": "conversation",
                    "storage_ns": "alpha",
                    "workspace": "workspace-a",
                    "graph_space": "conversation",
                    "security_scope": "team-alpha",
                    "capabilities": ["send_message"]
                }),
            }
        };
        let actor_defaulted = service.execute(make_request(None)).await;
        assert_eq!(
            actor_defaulted.status,
            StatusCode::FORBIDDEN.as_u16(),
            "an omitted user_id defaults to the authenticated actor, matching Python"
        );
        let rejected = service.execute(make_request(Some("conversation-owner"))).await;
        assert!(rejected.status >= 500, "{}", String::from_utf8_lossy(&rejected.body));
        let still_only_root = service
            .conversation_store
            .graph_projection_records(GraphProjectionRead {
                scope: GraphScope {
                    namespace: "alpha".to_owned(),
                    workspace_id: Some("workspace-a".to_owned()),
                    graph_space: Some("conversation".to_owned()),
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
                .conversation_store
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
                "DROP TRIGGER reject_send_message_index_job ON \"{schema}_conversation\".index_jobs; DROP FUNCTION \"{schema}_conversation\".reject_send_message_index_job();"
            ))
            .await
            .unwrap();
        drop(cleanup_client);
        connection_task.await.unwrap().unwrap();

        let http_secret = "send-message-mcp-test-secret";
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: vec!["rw".to_owned()],
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig {
                    mode: "jwt".to_owned(),
                    algorithm: Some("HS256".to_owned()),
                    key: Some(http_secret.to_owned()),
                    ..AuthConfig::default()
                },
            },
            Arc::new(service.clone()),
        );
        let now_seconds = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let token = jsonwebtoken::encode(
            &jsonwebtoken::Header::new(jsonwebtoken::Algorithm::HS256),
            &json!({
                "sub":"conversation-owner@example.test",
                "user_id":"conversation-owner",
                "actor_id":"conversation-owner",
                "role":"rw",
                "roles":["rw"],
                "ns":"conversation",
                "storage_ns":"alpha",
                "workspace":"workspace-a",
                "graph_space":"conversation",
                "security_scope":"team-alpha",
                "capabilities":["send_message"],
                "exp":now_seconds + 300
            }),
            &jsonwebtoken::EncodingKey::from_secret(http_secret.as_bytes()),
        )
        .unwrap();
        let accepted = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/conversation")
                    .header("content-type", "application/json")
                    .header("authorization", format!("Bearer {token}"))
                    .body(Body::from(
                        json!({
                            "jsonrpc":"2.0",
                            "id":"send-message-ask",
                            "method":"tools/call",
                            "params":{
                                "name":"conversation.ask",
                                "arguments":{
                                    "conversation_id":conversation_id,
                                    "text":"A user question",
                                    "user_id":"conversation-owner",
                                    "workflow_id":"send-message-admission-test"
                                }
                            }
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(accepted.status(), StatusCode::OK);
        let accepted_body = to_bytes(accepted.into_body(), 65_536).await.unwrap();
        let response: Value = serde_json::from_slice(&accepted_body).unwrap();
        let run_id = response["result"]["run_id"].as_str().unwrap();
        let turn_id = response["result"]["turn_node_id"].as_str().unwrap();
        let nodes = service
            .conversation_store
            .graph_projection_records(GraphProjectionRead {
                scope: GraphScope {
                    namespace: "alpha".to_owned(),
                    workspace_id: Some("workspace-a".to_owned()),
                    graph_space: Some("conversation".to_owned()),
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
                .conversation_store
                .graph_projection_records(GraphProjectionRead {
                    scope: GraphScope {
                        namespace: "alpha".to_owned(),
                        workspace_id: Some("workspace-a".to_owned()),
                        graph_space: Some("conversation".to_owned()),
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
        assert_eq!(created["admission_actor"]["user_id"], "conversation-owner");
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
        let queued_payload: Value =
            serde_json::from_str(lanes[0].payload_json.as_deref().unwrap()).unwrap();
        assert_eq!(queued_payload["initial_state"]["turn_index"], 0);
        assert_eq!(
            queued_payload["initial_state"]["_rt_graph_scope"]["storage_namespace"],
            "alpha"
        );
        assert_eq!(
            queued_payload["initial_state"]["_rt_graph_scope"]["workspace"],
            "workspace-a"
        );
        assert_eq!(
            queued_payload["initial_state"]["_rt_graph_scope"]["graph_space"],
            "conversation"
        );
        assert_eq!(
            queued_payload["initial_state"]["_rt_graph_scope"]["security_scope"],
            "team-alpha"
        );
        assert_eq!(
            service
                .conversation_store
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
            .batch_execute(&format!(
                "DROP SCHEMA IF EXISTS \"{schema}_conversation\" CASCADE; \
                 DROP SCHEMA IF EXISTS \"{schema}_workflow\" CASCADE; \
                 DROP SCHEMA IF EXISTS \"{schema}_wisdom\" CASCADE; \
                 DROP SCHEMA IF EXISTS \"{schema}\" CASCADE"
            ))
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

    #[tokio::test]
    async fn sqlite_chroma_admits_valid_python_answer_workflow_plan() {
        use kogwistar_store::{NamedProjectionWrite, WorkflowDesignSnapshotWrite};
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-answer-chroma-{nonce}.sqlite3"));
        let workflow_id = "debug.rag.v1";
        let contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/answer-workflow-runtime-contract-v1.json"
        ))
        .unwrap();
        let workflow = &contract["workflows"][workflow_id];
        let python_version = workflow["python_op_version"].as_str().unwrap();
        let nodes = workflow["required_ops"]
            .as_array()
            .unwrap()
            .iter()
            .map(|operation| {
                let operation = operation.as_str().unwrap();
                json!({
                    "id": format!("node-{operation}"),
                    "label": operation,
                    "metadata": {
                        "entity_type": "workflow_node",
                        "workflow_id": workflow_id,
                        "wf_op": operation,
                        "wf_version": python_version,
                        "wf_start": operation == "start",
                        "wf_executor_target": "python-worker"
                    }
                })
            })
            .collect::<Vec<_>>();
        let edges = workflow["required_routes"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(index, route)| {
                json!({
                    "id": format!("edge-{index}"),
                    "source_ids": [format!("node-{}", route["from"].as_str().unwrap())],
                    "target_ids": [format!("node-{}", route["to"].as_str().unwrap())],
                    "metadata": {
                        "entity_type": "workflow_edge",
                        "workflow_id": workflow_id,
                        "wf_predicate": route["predicate"],
                        "wf_priority": route["priority"],
                        "wf_is_default": route["default"],
                        "wf_multiplicity": route["multiplicity"]
                    }
                })
            })
            .collect::<Vec<_>>();
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
                    materialization_status: "rust_event_only".to_owned(),
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
                    payload_json: serde_json::to_string(&json!({"nodes": nodes, "edges": edges}))
                        .unwrap(),
                },
            )
            .unwrap();
        drop(store);

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let chroma_setup = tokio::spawn(async move {
            for _ in 0..2 {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut bytes = Vec::new();
                let mut chunk = [0_u8; 2048];
                let (header_end, content_length) = loop {
                    let count = stream.read(&mut chunk).await.unwrap();
                    assert!(count > 0, "Chroma connection closed before request body");
                    bytes.extend_from_slice(&chunk[..count]);
                    if let Some(header_end) = bytes
                        .windows(4)
                        .position(|window| window == b"\r\n\r\n")
                    {
                        let headers = String::from_utf8_lossy(&bytes[..header_end]);
                        let content_length = headers
                            .lines()
                            .find_map(|line| {
                                let (name, value) = line.split_once(':')?;
                                name.eq_ignore_ascii_case("content-length")
                                    .then(|| value.trim().parse::<usize>().ok())
                                    .flatten()
                            })
                            .unwrap_or(0);
                        if bytes.len() >= header_end + 4 + content_length {
                            break (header_end, content_length);
                        }
                    }
                };
                let request: Value = serde_json::from_slice(
                    &bytes[header_end + 4..header_end + 4 + content_length],
                )
                .unwrap();
                let response = json!({
                    "id": format!("collection-{}", request["name"].as_str().unwrap()),
                    "metadata": {"hnsw:space": "cosine"}
                })
                .to_string();
                let wire = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    response.len(), response
                );
                stream.write_all(wire.as_bytes()).await.unwrap();
            }
        });
        let chroma_config = |collection: &str| kogwistar_store_chroma::ChromaStoreConfig {
            endpoint: endpoint.clone(),
            collection: collection.to_owned(),
            ..Default::default()
        };
        let graph = kogwistar_store_chroma::ChromaGraphStore::connect_entity_collections(
            chroma_config("answer-nodes"),
            chroma_config("answer-edges"),
        )
        .await
        .unwrap();
        chroma_setup.await.unwrap();

        let service = SqliteRunApplicationService::open(&path)
            .unwrap()
            .with_chroma_graph(graph);
        let response = service.submit_run(&ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/workflow/runs".to_owned(),
            body: serde_json::to_vec(&json!({
                "workflow_id": workflow_id,
                "conversation_id": "answer-conversation",
                "run_id": "answer-run-python-worker",
                "initial_state": {
                    "_rt_graph_scope": {"storage_namespace":"attacker-scope"},
                    "_rt_run_owner_id": "attacker"
                }
            }))
            .unwrap(),
            principal: json!({
                "sub": "alice", "user_id": "alice", "role": "rw",
                "ns": ["conversation", "workflow"],
                "storage_ns": "conversation",
                "capabilities": ["read_graph", "write_graph"]
            }),
        });
        assert_eq!(response.status, StatusCode::ACCEPTED.as_u16());
        let claimed = service
            .claim_runtime_work_items(
                RuntimeClaimRequest {
                    claimed_by: "python-worker".to_owned(),
                    limit: 1,
                    lease_seconds: 60,
                    run_id: Some("answer-run-python-worker".to_owned()),
                    recipient_ids: vec!["python-worker".to_owned()],
                },
                vec!["python-worker".to_owned()],
            )
            .unwrap();
        assert_eq!(claimed.len(), 1);
        assert_eq!(claimed[0].step_id.as_deref(), Some("node-start"));
        assert_eq!(claimed[0].payload["op"], "start");
        assert_eq!(claimed[0].payload["executor_target"], "python-worker");
        assert_eq!(
            claimed[0].payload["initial_state"]["_rt_graph_scope"]["storage_namespace"],
            "conversation",
            "SQLite admission must replace caller-supplied runtime scope with authenticated scope"
        );
        assert_eq!(
            claimed[0].payload["initial_state"]["_rt_run_owner_id"],
            "alice",
            "SQLite admission must replace caller-supplied runtime owner"
        );

        drop(service);
        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn sqlite_chroma_native_answer_snapshots_before_provider_and_persists_stream_events() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let nonce = uuid::Uuid::new_v4().simple().to_string();
        let db_path = std::env::temp_dir().join(format!("kogwistar-sqlite-native-answer-{nonce}.db"));
        let auth_path = std::env::temp_dir().join(format!("kogwistar-sqlite-native-answer-auth-{nonce}.db"));
        let auth = SqliteAuthStore::open(&auth_path).unwrap();
        auth.resolve_external_identity(ResolveExternalIdentity {
            issuer: "test".to_owned(),
            subject: "alice".to_owned(),
            email: "alice@example.test".to_owned(),
            display_name: None,
            new_user_id: "alice".to_owned(),
            default_role: "rw".to_owned(),
            default_ns: "conversation".to_owned(),
        })
        .await
        .unwrap();
        drop(auth);
        let auth_config = AuthConfig {
            auth_db_url: Some(format!("sqlite:///{}", auth_path.display())),
            ..AuthConfig::default()
        };

        let chroma_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let chroma_endpoint = format!("http://{}", chroma_listener.local_addr().unwrap());
        let chroma_server = tokio::spawn(async move {
            for _ in 0..2 {
                let (mut stream, _) = chroma_listener.accept().await.unwrap();
                let mut bytes = Vec::new();
                let (header_end, content_length) = loop {
                    let mut chunk = [0_u8; 1024];
                    let count = stream.read(&mut chunk).await.unwrap();
                    assert!(count > 0);
                    bytes.extend_from_slice(&chunk[..count]);
                    if let Some(end) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
                        let headers = String::from_utf8_lossy(&bytes[..end]);
                        let length = headers.lines().find_map(|line| {
                            let (name, value) = line.split_once(':')?;
                            name.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse::<usize>().ok())
                                .flatten()
                        }).unwrap_or(0);
                        if bytes.len() >= end + 4 + length {
                            break (end, length);
                        }
                    }
                };
                let request: Value = serde_json::from_slice(&bytes[header_end + 4..header_end + 4 + content_length]).unwrap();
                let body = json!({
                    "id": format!("collection-{}", request["name"].as_str().unwrap()),
                    "metadata": {"hnsw:space":"cosine"}
                }).to_string();
                let response = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                    body.len(), body
                );
                stream.write_all(response.as_bytes()).await.unwrap();
            }
        });
        let chroma_config = |collection: &str| kogwistar_store_chroma::ChromaStoreConfig {
            endpoint: chroma_endpoint.clone(),
            collection: collection.to_owned(),
            ..Default::default()
        };
        let graph = kogwistar_store_chroma::ChromaGraphStore::connect_entity_collections(
            chroma_config("native-answer-nodes"),
            chroma_config("native-answer-edges"),
        ).await.unwrap();
        chroma_server.await.unwrap();

        let chat_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let chat_endpoint = format!("http://{}/v1/chat/completions", chat_listener.local_addr().unwrap());
        let db_path_for_provider = db_path.clone();
        let chat_server = tokio::spawn(async move {
            let (mut stream, _) = chat_listener.accept().await.unwrap();
            let mut bytes = Vec::new();
            let (header_end, content_length) = loop {
                let mut chunk = [0_u8; 2048];
                let count = stream.read(&mut chunk).await.unwrap();
                assert!(count > 0);
                bytes.extend_from_slice(&chunk[..count]);
                if let Some(end) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
                    let headers = String::from_utf8_lossy(&bytes[..end]);
                    let length = headers.lines().find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().ok())
                            .flatten()
                    }).unwrap_or(0);
                    if bytes.len() >= end + 4 + length {
                        break (end, length);
                    }
                }
            };
            let request: Value = serde_json::from_slice(&bytes[header_end + 4..header_end + 4 + content_length]).unwrap();
            assert_eq!(request["stream"], true);
            let sqlite = SqliteStore::open(&db_path_for_provider).unwrap();
            let snapshot_id = kogwistar_contracts::stable_id(
                "conversation.context_snapshot",
                &[
                    "native-answer-conversation".to_owned(),
                    "native-answer-run".to_owned(),
                    "generate_answer_with_citations".to_owned(),
                    "3".to_owned(),
                    "0".to_owned(),
                ],
            )
            .to_string();
            let snapshot_events = sqlite
                .latest_entity_events("conversation", "node", &[snapshot_id.clone()])
                .unwrap();
            assert!(snapshot_events
                .get(&snapshot_id)
                .is_some_and(|event| event.payload_json.contains("context_snapshot")),
                "context snapshot must commit before the provider request");

            let answer = json!({"text":"answer","reasoning":"brief rationale","claims":[]}).to_string();
            let chunks = [
                json!({"id":"chatcmpl-test","choices":[{"delta":{"content":answer},"finish_reason":null}]}).to_string(),
                json!({"id":"chatcmpl-test","choices":[{"delta":{},"finish_reason":"stop"}]}).to_string(),
            ];
            let mut body = String::new();
            for chunk in chunks { body.push_str(&format!("data: {chunk}\r\n\r\n")); }
            body.push_str("data: [DONE]\r\n\r\n");
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                body.len(), body
            );
            for chunk in response.as_bytes().chunks(17) {
                stream.write_all(chunk).await.unwrap();
            }
        });

        let principal = json!({
            "sub":"alice", "user_id":"alice", "role":"rw",
            "ns":["conversation"], "storage_ns":"conversation",
            "capabilities":["read_graph", "write_graph"]
        });
        let service = SqliteRunApplicationService::open(&db_path)
            .unwrap()
            .with_chroma_graph(graph)
            .with_chat_completion_provider(
                crate::native_answer_generation::OpenAiCompatibleChatConfig::new(
                    chat_endpoint, "fixture-answer-model", None,
                ).unwrap().with_streaming(true),
            )
            .with_owner_auth_config(&auth_config).await.unwrap();
        service.store.create_server_run(ServerRunCreate {
            run_id: "native-answer-run".to_owned(),
            conversation_id: "native-answer-conversation".to_owned(),
            workflow_id: "debug.rag.v1".to_owned(),
            user_id: Some("alice".to_owned()),
            user_turn_node_id: "user-turn".to_owned(),
            status: "running".to_owned(),
        }).unwrap();
        let binding = runtime_owner_binding_v1(&principal, &service.capabilities).unwrap();
        service.store.append_server_run_event(
            "native-answer-run",
            "run.created",
            serde_json::to_string(&json!({"runtime_owner_binding":binding})).unwrap(),
        ).unwrap();
        let selection_request = kogwistar_runtime::RuntimeNativeStepRequest {
            executor_target: "rust-native".to_owned(),
            operation: "aa_select_used_evidence".to_owned(),
            op_version: "v1".to_owned(),
            run_id: "native-answer-run".to_owned(),
            workflow_id: "debug.rag.v1".to_owned(),
            node_id: "select-node".to_owned(),
            conversation_id: "native-answer-conversation".to_owned(),
            turn_node_id: Some("user-turn".to_owned()),
            token_id: "selection-token".to_owned(),
            parent_token_id: None,
            step_seq: 2,
            state: serde_json::from_value(json!({
                "conversation_id":"native-answer-conversation",
                "run_id":"native-answer-run",
                "run_step_seq":2,
                "attempt_seq":1,
                "question":"question",
                "system_prompt":"Select relevant evidence.",
                "candidates":[{"id":"candidate-node","label":"Candidate","summary":"summary"}],
                "conversation_context_messages":[{"role":"user","content":"prior context"}],
                "conversation_context_node_ids":[],
                "_rt_graph_scope": {
                    "storage_namespace":"conversation",
                    "workspace":null,
                    "graph_space":"conversation"
                }
            })).unwrap(),
            runtime_routes: Vec::new(),
            ephemeral_context: serde_json::Map::new(),
        };
        service
            .persist_sqlite_native_selection_context_snapshot(&selection_request)
            .await
            .unwrap();
        let selection_snapshot_id = kogwistar_contracts::stable_id(
            "conversation.context_snapshot",
            &[
                "native-answer-conversation".to_owned(),
                "native-answer-run".to_owned(),
                "select_used_evidence".to_owned(),
                "2".to_owned(),
                "1".to_owned(),
            ],
        )
        .to_string();
        let selection_snapshot_events = service
            .store
            .latest_entity_events("conversation", "node", &[selection_snapshot_id.clone()])
            .unwrap();
        assert!(selection_snapshot_events
            .get(&selection_snapshot_id)
            .is_some_and(|event| event.payload_json.contains("select_used_evidence")));
        let request = kogwistar_runtime::RuntimeNativeStepRequest {
            executor_target: "rust-native".to_owned(),
            operation: "aa_generate_answer_with_citations".to_owned(),
            op_version: "v1".to_owned(),
            run_id: "native-answer-run".to_owned(),
            workflow_id: "debug.rag.v1".to_owned(),
            node_id: "answer-node".to_owned(),
            conversation_id: "native-answer-conversation".to_owned(),
            turn_node_id: Some("user-turn".to_owned()),
            token_id: "token-1".to_owned(),
            parent_token_id: None,
            step_seq: 3,
            state: serde_json::from_value(json!({
                "conversation_id":"native-answer-conversation",
                "question":"question",
                "system_prompt":"Use only evidence.",
                "used_node_ids":[],
                "turn_node_id":"user-turn",
                "_rt_graph_scope": {
                    "storage_namespace":"conversation",
                    "workspace":null,
                    "graph_space":"conversation"
                },
                "_rt":{"evidence_pack":{"nodes":[],"edges":[]}}
            })).unwrap(),
            runtime_routes: Vec::new(),
            ephemeral_context: serde_json::Map::new(),
        };
        let snapshot_id = kogwistar_contracts::stable_id(
            "conversation.context_snapshot",
            &[
                "native-answer-conversation".to_owned(),
                "native-answer-run".to_owned(),
                "aa_generate_answer_with_citations".to_owned(),
                "3".to_owned(),
                "0".to_owned(),
            ],
        )
        .to_string();
        let mut unsupported_version = request.clone();
        unsupported_version.op_version = "v2".to_owned();
        let rejected = service
            .execute_sqlite_native_answer_generation(
                RecordedWorkerHandoff {
                    message_id: "lane-invalid-native-answer".to_owned(),
                    claimed_by: "rust-native-worker".to_owned(),
                    run_id: "native-answer-run".to_owned(),
                    step_id: "answer-node".to_owned(),
                    correlation_id: "token-invalid".to_owned(),
                },
                unsupported_version,
            )
            .await;
        assert!(rejected.is_err());
        assert!(service
            .store
            .latest_entity_events("conversation", "node", &[snapshot_id.clone()])
            .unwrap()
            .is_empty(), "unsupported operation version must fail before snapshot persistence");
        let result = service.execute_sqlite_native_answer_generation(
            RecordedWorkerHandoff {
                message_id:"lane-native-answer".to_owned(),
                claimed_by:"rust-native-worker".to_owned(),
                run_id:"native-answer-run".to_owned(),
                step_id:"answer-node".to_owned(),
                correlation_id:"token-1".to_owned(),
            },
            request,
        ).await.unwrap();
        chat_server.await.unwrap();

        assert_eq!(result.effect.as_ref().unwrap().status, kogwistar_runtime::RuntimeWorkerEffectStatus::Success);
        let first_update = serde_json::to_value(&result.effect.as_ref().unwrap().state_update[0])
            .unwrap();
        assert_eq!(first_update[1]["answer"]["text"], "answer");
        let events = service.store.list_server_run_events("native-answer-run", 0, 100).unwrap();
        assert!(events.iter().any(|event| event.event_type == "workflow.provider_event.v1"));
        let lane_events = service.store.list_projected_lane_messages(LaneMessageFilter {
            namespace: Some("conversation".to_owned()),
            inbox_id: Some("agent-provider-events".to_owned()),
            ..LaneMessageFilter::default()
        }).unwrap();
        assert!(!lane_events.is_empty());
        assert!(!service
            .store
            .list_index_jobs(Some("conversation"), None, None, None, None, 100)
            .unwrap()
            .is_empty());

        drop(service);
        let _ = std::fs::remove_file(db_path);
        let _ = std::fs::remove_file(auth_path);
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
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use jsonwebtoken::{EncodingKey, Header, encode};
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tower::ServiceExt;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let fake = tokio::spawn(async move {
            let mut observed = Vec::new();
            loop {
                let Ok(Ok((mut stream, _))) =
                    tokio::time::timeout(std::time::Duration::from_secs(1), listener.accept())
                        .await
                else {
                    break;
                };
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
                        let workspace = if document == "doc5" {"workspace-b"} else {"workspace-a"};
                        let original = if document == "doc5" {
                            json!({"workspace_id":workspace,"graph_space":"facts"})
                        } else if document == "doc2" {
                            json!({"workspace_id":workspace,"graph_space":"facts","_kogwistar_stage2_ready":true,"_kogwistar_source_revision":1})
                        } else {
                            json!({"workspace_id":workspace,"graph_space":"facts","_kogwistar_stage2_ready":true,"_kogwistar_source_revision":2})
                        };
                        json!({
                            "_kogwistar_namespace":"scope-a",
                            "workspace_id":workspace,
                            "graph_space":"facts",
                            "_kogwistar_original_metadata":original.to_string(),
                            "_test_document": document
                        })
                    };
                    json!({
                        "ids":[["n4","n2","n5"]],
                        "documents":[["doc4","doc2","doc5"]],
                        "metadatas":[[scoped("doc4"),scoped("doc2"),scoped("doc5")]],
                        "embeddings":[[[0.1,0.2],[0.1,0.2],[0.1,0.2]]],
                        "distances":[[0.03,0.1,0.01]]
                    })
                } else if line.contains("collection-edges/query") {
                    let scoped = |document: &str| json!({
                        "_kogwistar_namespace":"scope-a",
                        "workspace_id":"workspace-a",
                        "graph_space":"facts",
                        "_kogwistar_original_metadata":"{\"workspace_id\":\"workspace-a\",\"graph_space\":\"facts\"}",
                        "_test_document": document
                    });
                    json!({
                        "ids":[["e2","e1","e3"]],
                        "documents":[[
                            r#"{"source_ids":["n1"],"target_ids":["n3"]}"#,
                            r#"{"source_ids":["n1"],"target_ids":["n2"]}"#,
                            r#"{"source_ids":["n1"],"target_ids":["n5"]}"#
                        ]],
                        "metadatas":[[scoped("edge2"),scoped("edge1"),scoped("edge3")]],
                        "embeddings":[[[0.1,0.2],[0.1,0.2],[0.1,0.2]]],
                        "distances":[[0.02,0.1,0.01]]
                    })
                } else if line.contains("collection-nodes/get") {
                    let metas = [
                        json!({}),
                        json!({"_kogwistar_stage2_ready":true,"_kogwistar_source_revision":1}),
                        json!({}),
                        json!({"_kogwistar_stage2_ready":true,"_kogwistar_source_revision":2}),
                        json!({}),
                        json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"node","acl_target_entity_id":"n1","acl_version":1,"acl_mode":"public"}),
                        json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"node","acl_target_entity_id":"n2","acl_version":1,"acl_mode":"public"}),
                        json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"node","acl_target_entity_id":"n3","acl_version":1,"acl_mode":"private"}),
                        json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"node","acl_target_entity_id":"n4","acl_version":1,"acl_mode":"public"}),
                        json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"node","acl_target_entity_id":"n5","acl_version":1,"acl_mode":"public"}),
                        json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"edge","acl_target_entity_id":"e1","acl_version":1,"acl_mode":"public"}),
                        json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"edge","acl_target_entity_id":"e3","acl_version":1,"acl_mode":"public"}),
                    ];
                    let ids = ["n1","n2","n3","n4","n5","acl-n1","acl-n2","acl-n3","acl-n4","acl-n5","acl-e1","acl-e3"];
                    json!({"ids":ids,
                        "documents":["{}","{}","{}","{}","{}","{}","{}","{}","{}","{}","{}","{}"],
                        "metadatas":metas.into_iter().enumerate().map(|(index, m)| {
                            let workspace = if index == 4 || index == 9 {"workspace-b"} else {"workspace-a"};
                            let mut original = m.as_object().unwrap().clone();
                            original.insert("workspace_id".to_owned(), json!(workspace));
                            original.insert("graph_space".to_owned(), json!("facts"));
                            json!({"_kogwistar_namespace":"scope-a","workspace_id":workspace,"graph_space":"facts","_kogwistar_original_metadata":Value::Object(original).to_string()})
                        }).collect::<Vec<_>>(),
                        "embeddings":[null,null,null,null,null,null,null,null,null,null,null,null]})
                } else {
                    json!({"ids":["e1"],
                        "documents":[r#"{"source_ids":["n1"],"target_ids":["n2"]}"#],
                        "metadatas":[json!({"_kogwistar_namespace":"scope-a","workspace_id":"workspace-a","graph_space":"facts","_kogwistar_original_metadata":"{\"workspace_id\":\"workspace-a\",\"graph_space\":\"facts\"}"})],
                        "embeddings":[null]})
                };
                let body = payload.to_string();
                let response = format!("HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}", body.len(), body);
                stream.write_all(response.as_bytes()).await.unwrap();
                observed.push((line, input));
            }
            observed
        });
        let path = std::env::temp_dir().join(format!(
            "kogwistar-sqlite-chroma-{}.sqlite3",
            uuid::Uuid::new_v4()
        ));
        let service = open_sqlite_application_service(
            &path,
            Some(SqliteChromaGraphConfig {
                endpoint,
                metric: DistanceMetric::Cosine,
                node_collection: "nodes".to_owned(),
                edge_collection: "edges".to_owned(),
            }),
        )
        .await
        .unwrap();
        let graph = service.chroma_graph.as_ref().unwrap();
        let filter = MetadataFilter::default();
        let (nodes, edges) = tokio::try_join!(
            graph.graph_records_of_kind_bounded("scope-a", GraphEntityKind::Node, &filter, 100),
            graph.graph_records_of_kind_bounded("scope-a", GraphEntityKind::Edge, &filter, 100),
        )
        .unwrap();
        assert_eq!(nodes.len(), 12, "decoded node records: {nodes:?}");
        assert_eq!(edges.len(), 1, "decoded edge records: {edges:?}");
        let principal = GraphAclPrincipal { principal_id: "reader-1".to_owned(), ..Default::default() };
        let (visible_nodes, visible_edges) = graph_records_visible_by_acl(&nodes, &edges, "knowledge", &principal);
        assert_eq!(visible_nodes.len(), 4, "ACL-visible nodes: {visible_nodes:?}");
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
        let graph_read_tools = [
            "kg_neighbors",
            "kg_find_edges",
            "kg_shortest_path",
            "kg_k_hop",
            "kg_viz_d3_json",
            "kg_viz_cytoscape_json",
        ];
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
                    "graph_space":"facts", "capabilities":capabilities, "exp":now + 300
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

        let mcp_read = app.clone().oneshot(
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

        for (tool, expected_key) in [
            ("kg_viz_d3_json", "nodes"),
            ("kg_viz_cytoscape_json", "elements"),
        ] {
            let visualization = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/mcp")
                        .header("content-type", "application/json")
                        .header(
                            "authorization",
                            format!("Bearer {}", token(&["read_graph"])),
                        )
                        .body(Body::from(format!(
                            "{{\"jsonrpc\":\"2.0\",\"id\":\"{tool}\",\"method\":\"tools/call\",\"params\":{{\"name\":\"{tool}\",\"arguments\":{{\"doc_id\":null,\"mode\":\"reify\"}}}}}}"
                        )))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(visualization.status(), StatusCode::OK);
            let visualization: Value = serde_json::from_slice(
                &to_bytes(visualization.into_body(), 65_536).await.unwrap(),
            )
            .unwrap();
            let rows = visualization["result"][expected_key].as_array().unwrap();
            assert!(rows.iter().any(|row| row.to_string().contains("n1")));
            assert!(rows.iter().any(|row| row.to_string().contains("n2")));
            assert!(
                !visualization.to_string().contains("n3"),
                "private graph endpoint must not leak through {tool}: {visualization}"
            );
            assert!(
                !visualization.to_string().contains("n5"),
                "cross-workspace graph entity must not leak through {tool}: {visualization}"
            );
        }

        for (path, result_key) in [
            ("/api/viz/d3.json", "nodes"),
            ("/api/viz/cytoscape.json", "elements"),
        ] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("GET")
                        .uri(format!("{path}?graph_type=knowledge&mode=reify"))
                        .header(
                            "authorization",
                            format!("Bearer {}", token(&["read_graph"])),
                        )
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK, "{path}");
            let visualization: Value = serde_json::from_slice(
                &to_bytes(response.into_body(), 65_536).await.unwrap(),
            )
            .unwrap();
            let rows = visualization[result_key].as_array().unwrap();
            let serialized = visualization.to_string();
            assert!(serialized.contains("n1"), "{path}: {visualization}");
            assert!(serialized.contains("n2"), "{path}: {visualization}");
            assert!(serialized.contains("e1"), "{path}: {visualization}");
            assert!(!serialized.contains("n3"), "private node leaked: {path}");
            assert!(!serialized.contains("n5"), "cross-workspace node leaked: {path}");
            assert!(!rows.is_empty(), "{path} returned no graph records");
        }

        let bundle = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/viz/d3.bundle?graph_type=knowledge&mode=reify")
                    .header(
                        "authorization",
                        format!("Bearer {}", token(&["read_graph"])),
                    )
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(bundle.status(), StatusCode::OK);
        assert_eq!(bundle.headers()["content-type"], "text/html; charset=utf-8");
        let bundle = String::from_utf8(
            to_bytes(bundle.into_body(), 1_048_576)
                .await
                .unwrap()
                .to_vec(),
        )
        .unwrap();
        let embedded = bundle
            .split_once("window.__EMBEDDED_DATA__ = ")
            .and_then(|(_, rest)| rest.split_once(';'))
            .map(|(payload, _)| serde_json::from_str::<Value>(payload).unwrap())
            .expect("SQLite+Chroma D3 bundle embeds the authorized projection");
        let embedded_json = embedded.to_string();
        assert!(embedded_json.contains("n1"));
        assert!(embedded_json.contains("n2"));
        assert!(embedded_json.contains("e1"));
        assert!(!embedded_json.contains("n3"));
        assert!(!embedded_json.contains("n5"));

        let request = |path: &str, body: Value, capabilities: Value| ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: path.to_owned(),
            body: serde_json::to_vec(&body).unwrap(),
            principal: json!({"sub":"reader-1","storage_ns":"scope-a","workspace_id":"workspace-a","graph_space":"facts","capabilities":capabilities}),
        };
        let neighbor_body = json!({"rid":"n1","truth_graph":"knowledge"});
        let read_capability = json!(["read_graph"]);
        for route in [
            INTERNAL_GRAPH_NEIGHBORS_ROUTE,
            INTERNAL_GRAPH_VECTOR_QUERY_ROUTE,
            INTERNAL_GRAPH_FIND_EDGES_ROUTE,
            INTERNAL_GRAPH_SHORTEST_PATH_ROUTE,
            INTERNAL_GRAPH_K_HOP_ROUTE,
            INTERNAL_GRAPH_VIZ_D3_ROUTE,
            INTERNAL_GRAPH_VIZ_CYTOSCAPE_ROUTE,
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

        for (route, key) in [
            (INTERNAL_GRAPH_VIZ_D3_ROUTE, "nodes"),
            (INTERNAL_GRAPH_VIZ_CYTOSCAPE_ROUTE, "elements"),
        ] {
            let output = service
                .execute(request(route, json!({"mode":"reify"}), read_capability.clone()))
                .await;
            assert_eq!(
                output.status,
                StatusCode::OK.as_u16(),
                "visualization route {route} failed: {}",
                String::from_utf8_lossy(&output.body)
            );
            let output: Value = serde_json::from_slice(&output.body).unwrap();
            assert!(output[key].as_array().is_some());
            assert!(!output.to_string().contains("n3"));
        }

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
        assert!(observed.len() >= 20);
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
        assert!(scoped_reads.len() >= 12);
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

    #[tokio::test]
    async fn sqlite_chroma_receive_message_returns_scoped_acl_filtered_transcript() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use jsonwebtoken::{EncodingKey, Header, encode};
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tower::ServiceExt;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let (stop_chroma, mut stop_chroma_rx) = tokio::sync::oneshot::channel();
        let fake = tokio::spawn(async move {
            let mut requests = Vec::new();
            loop {
                let accepted = tokio::select! {
                    _ = &mut stop_chroma_rx => break,
                    accepted = listener.accept() => accepted.unwrap(),
                };
                let (mut stream, _) = accepted;
                let mut bytes = Vec::new();
                let mut chunk = [0_u8; 4096];
                let (split, length) = loop {
                    let count = stream.read(&mut chunk).await.unwrap();
                    assert!(count > 0);
                    bytes.extend_from_slice(&chunk[..count]);
                    if let Some(split) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
                        let head = String::from_utf8_lossy(&bytes[..split]);
                        let length = head
                            .lines()
                            .find_map(|line| {
                                let (name, value) = line.split_once(':')?;
                                name.eq_ignore_ascii_case("content-length")
                                    .then(|| value.trim().parse::<usize>().ok())
                                    .flatten()
                            })
                            .unwrap_or(0);
                        if bytes.len() >= split + 4 + length {
                            break (split, length);
                        }
                    }
                };
                let head = String::from_utf8_lossy(&bytes[..split]).to_string();
                let body: Value = serde_json::from_slice(&bytes[split + 4..split + 4 + length])
                    .unwrap();
                let payload = if head.lines().next().unwrap().contains("/collections HTTP") {
                    json!({
                        "id": format!("collection-{}", body["name"].as_str().unwrap()),
                        "metadata": {"hnsw:space": "cosine"}
                    })
                } else if head.lines().next().unwrap().contains("collection-nodes/get") {
                    let nodes = [
                        ("start", json!({"entity_type":"conversation_start","conversation_id":"c1","user_id":"reader-1","status":"active"})),
                        ("turn-2", json!({"entity_type":"conversation_turn","conversation_id":"c1","turn_index":2,"role":"assistant","summary":"second"})),
                        ("turn-1", json!({"entity_type":"conversation_turn","conversation_id":"c1","turn_index":1,"role":"user","summary":"first"})),
                        ("turn-private", json!({"entity_type":"conversation_turn","conversation_id":"c1","turn_index":3,"role":"assistant","summary":"private"})),
                        ("snapshot", json!({"entity_type":"context_snapshot","conversation_id":"c1","run_id":"run-a","stage":"answer","snapshot_seq":7})),
                        ("acl-start", json!({"entity_type":"acl_record","acl_truth_graph":"conversation","acl_target_grain":"node","acl_target_entity_id":"start","acl_version":1,"acl_mode":"public"})),
                        ("acl-turn-1", json!({"entity_type":"acl_record","acl_truth_graph":"conversation","acl_target_grain":"node","acl_target_entity_id":"turn-1","acl_version":1,"acl_mode":"public"})),
                        ("acl-turn-2", json!({"entity_type":"acl_record","acl_truth_graph":"conversation","acl_target_grain":"node","acl_target_entity_id":"turn-2","acl_version":1,"acl_mode":"public"})),
                        ("acl-turn-private", json!({"entity_type":"acl_record","acl_truth_graph":"conversation","acl_target_grain":"node","acl_target_entity_id":"turn-private","acl_version":1,"acl_mode":"private","acl_owner_id":"someone-else"})),
                        ("acl-snapshot", json!({"entity_type":"acl_record","acl_truth_graph":"conversation","acl_target_grain":"node","acl_target_entity_id":"snapshot","acl_version":1,"acl_mode":"public"})),
                    ];
                    let metadatas = nodes
                        .iter()
                        .map(|(_, metadata)| {
                            let mut original = metadata.as_object().unwrap().clone();
                            original.insert("workspace_id".to_owned(), json!("workspace-a"));
                            original.insert("graph_space".to_owned(), json!("conversation"));
                            json!({
                                "_kogwistar_namespace":"scope-a",
                                "workspace_id":"workspace-a",
                                "graph_space":"conversation",
                                "_kogwistar_original_metadata":Value::Object(original).to_string()
                            })
                        })
                        .collect::<Vec<_>>();
                    json!({
                        "ids": nodes.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
                        "documents": nodes.iter().map(|(id, _)| {
                            if *id == "snapshot" {
                                r#"{"metadata":{"summary":"stored summary"},"properties":{"references":["evidence-1"]}}"#.to_owned()
                            } else {
                                "{}".to_owned()
                            }
                        }).collect::<Vec<_>>(),
                        "metadatas": metadatas,
                        "embeddings": [null,null,null,null,null,null,null,null,null,null]
                    })
                } else {
                    json!({"ids": [], "documents": [], "metadatas": [], "embeddings": []})
                };
                let response_body = payload.to_string();
                let response = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                    response_body.len(),
                    response_body
                );
                stream.write_all(response.as_bytes()).await.unwrap();
                requests.push((head, body));
            }
            requests
        });
        let path = std::env::temp_dir().join(format!(
            "kogwistar-sqlite-chroma-transcript-{}.sqlite3",
            uuid::Uuid::new_v4()
        ));
        let service = open_sqlite_application_service(
            &path,
            Some(SqliteChromaGraphConfig {
                endpoint,
                metric: DistanceMetric::Cosine,
                node_collection: "nodes".to_owned(),
                edge_collection: "edges".to_owned(),
            }),
        )
        .await
        .unwrap();
        let request = |operation: &str, args: Value, principal: Value| ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: format!("/api/syscall/v1/{operation}"),
            body: json!({"version":"v1","args":args}).to_string().into_bytes(),
            principal,
        };
        let read_request = |path: &str, principal: Value| ApiEffectRequest {
            contract_version: 1,
            method: "GET".to_owned(),
            path_and_query: path.to_owned(),
            body: Vec::new(),
            principal,
        };
        let principal = json!({
            "sub":"reader-1", "storage_ns":"scope-a", "workspace_id":"workspace-a",
            "graph_space":"conversation", "ns":["conversation"], "capabilities":["read_graph"]
        });
        let denied = service
            .execute(request(
                "receive_message",
                json!({"conversation_id":"c1"}),
                json!({"ns":["conversation"],"capabilities":[]}),
            ))
            .await;
        assert_eq!(denied.status, StatusCode::FORBIDDEN.as_u16());
        let missing_id = service
            .execute(request("receive_message", json!({}), principal.clone()))
            .await;
        assert_eq!(missing_id.status, StatusCode::BAD_REQUEST.as_u16());

        let response = service
            .execute(request("receive_message", json!({"conversation_id":"c1"}), principal.clone()))
            .await;
        assert_eq!(response.status, StatusCode::OK.as_u16());
        let response: Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(response["op"], "receive_message");
        assert_eq!(response["result"]["conversation_id"], "c1");
        assert_eq!(response["result"]["messages"][0]["node_id"], "turn-1");
        assert_eq!(response["result"]["messages"][1]["node_id"], "turn-2");

        let empty_transcript = service
            .execute(request("receive_message", json!({"conversation_id":"missing"}), principal.clone()))
            .await;
        assert_eq!(empty_transcript.status, StatusCode::OK.as_u16());
        assert_eq!(
            serde_json::from_slice::<Value>(&empty_transcript.body).unwrap()["result"]["messages"],
            json!([])
        );

        let denied_mount = service
            .execute(request(
                "mount_memory",
                json!({"conversation_id":"c1"}),
                json!({"ns":["conversation"],"capabilities":[]}),
            ))
            .await;
        assert_eq!(denied_mount.status, StatusCode::FORBIDDEN.as_u16());

        let mounted = service
            .execute(request(
                "mount_memory",
                json!({"conversation_id":"c1","run_id":"run-a","stage":"answer"}),
                json!({
                    "sub":"reader-1", "storage_ns":"scope-a", "workspace_id":"workspace-a",
                    "graph_space":"conversation", "ns":["conversation"], "capabilities":["project_view"]
                }),
            ))
            .await;
        assert_eq!(mounted.status, StatusCode::OK.as_u16());
        let mounted: Value = serde_json::from_slice(&mounted.body).unwrap();
        assert_eq!(mounted["op"], "mount_memory");
        assert_eq!(mounted["result"]["snapshot_node_id"], "snapshot");
        assert_eq!(mounted["result"]["metadata"]["summary"], "stored summary");
        assert_eq!(mounted["result"]["properties"]["references"], json!(["evidence-1"]));
        assert!(mounted["result"].get("schema_version").is_none());

        let invoked_transcript = service
            .execute(request(
                "invoke_tool",
                json!({"tool_name":"list_transcript","conversation_id":"c1"}),
                json!({
                    "sub":"reader-1", "role":"rw", "storage_ns":"scope-a",
                    "workspace_id":"workspace-a", "graph_space":"conversation",
                    "ns":["conversation"], "capabilities":["invoke_tool","read_graph"]
                }),
            ))
            .await;
        let invoke_contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/invoke-tool-results-v1.json"
        ))
        .unwrap();
        assert_eq!(invoke_contract["version"], "v1");
        assert_eq!(invoke_contract["op"], "invoke_tool");
        let assert_invoke_result_shape = |tool_name: &str, result: &Value| {
            let expected = &invoke_contract["results"][tool_name];
            let actual_root = result
                .as_object()
                .unwrap()
                .keys()
                .cloned()
                .collect::<BTreeSet<_>>();
            let expected_root = expected["root_fields"]
                .as_array()
                .unwrap()
                .iter()
                .map(|field| field.as_str().unwrap().to_owned())
                .collect::<BTreeSet<_>>();
            assert_eq!(actual_root, expected_root, "{tool_name} root result fields");
            if let Some(snapshot_fields) = expected["snapshot_fields"].as_array() {
                let actual_snapshot = result["snapshot"]
                    .as_object()
                    .unwrap()
                    .keys()
                    .cloned()
                    .collect::<BTreeSet<_>>();
                let expected_snapshot = snapshot_fields
                    .iter()
                    .map(|field| field.as_str().unwrap().to_owned())
                    .collect::<BTreeSet<_>>();
                assert_eq!(
                    actual_snapshot, expected_snapshot,
                    "{tool_name} nested snapshot fields"
                );
            }
        };
        assert_eq!(invoked_transcript.status, StatusCode::OK.as_u16());
        let invoked_transcript: Value = serde_json::from_slice(&invoked_transcript.body).unwrap();
        assert_invoke_result_shape("list_transcript", &invoked_transcript["result"]);
        assert_eq!(invoked_transcript["result"]["tool_name"], "list_transcript");
        assert_eq!(invoked_transcript["result"]["messages"].as_array().unwrap().len(), 2);

        let invoked_snapshot = service
            .execute(request(
                "invoke_tool",
                json!({"tool_name":"latest_snapshot","conversation_id":"c1","run_id":"run-a","stage":"answer"}),
                json!({
                    "sub":"reader-1", "role":"rw", "storage_ns":"scope-a",
                    "workspace_id":"workspace-a", "graph_space":"conversation",
                    "ns":["conversation"], "capabilities":["invoke_tool","project_view"]
                }),
            ))
            .await;
        assert_eq!(invoked_snapshot.status, StatusCode::OK.as_u16());
        let invoked_snapshot: Value = serde_json::from_slice(&invoked_snapshot.body).unwrap();
        assert_invoke_result_shape("latest_snapshot", &invoked_snapshot["result"]);
        assert_eq!(invoked_snapshot["result"]["tool_name"], "latest_snapshot");
        assert_eq!(invoked_snapshot["result"]["snapshot"]["snapshot_node_id"], "snapshot");

        let denied_builtin = service
            .execute(request(
                "invoke_tool",
                json!({"tool_name":"capability_snapshot"}),
                json!({"role":"rw","ns":["conversation"],"capabilities":["invoke_tool"]}),
            ))
            .await;
        assert_eq!(denied_builtin.status, StatusCode::FORBIDDEN.as_u16());

        let capability_snapshot = service
            .execute(request(
                "invoke_tool",
                json!({"tool_name":"capability_snapshot"}),
                json!({
                    "sub":"reader-1", "role":"rw", "ns":["conversation"],
                    "capabilities":["invoke_tool","project_view","read_security_scope"]
                }),
            ))
            .await;
        assert_eq!(capability_snapshot.status, StatusCode::OK.as_u16());
        let capability_snapshot: Value = serde_json::from_slice(&capability_snapshot.body).unwrap();
        assert_invoke_result_shape("capability_snapshot", &capability_snapshot["result"]);
        assert_eq!(capability_snapshot["result"]["tool_name"], "capability_snapshot");
        assert_eq!(
            capability_snapshot["result"]["snapshot"]["effective_capabilities"][0],
            "invoke_tool"
        );

        let resource_snapshot = service
            .execute(request(
                "invoke_tool",
                json!({"tool_name":"resource_snapshot"}),
                json!({
                    "sub":"reader-1", "role":"rw", "ns":["conversation"],
                    "capabilities":["invoke_tool","project_view","read_security_scope"]
                }),
            ))
            .await;
        assert_eq!(resource_snapshot.status, StatusCode::OK.as_u16());
        let resource_snapshot: Value = serde_json::from_slice(&resource_snapshot.body).unwrap();
        assert_invoke_result_shape("resource_snapshot", &resource_snapshot["result"]);
        assert_eq!(resource_snapshot["result"]["tool_name"], "resource_snapshot");
        assert_eq!(resource_snapshot["result"]["snapshot"]["services"]["total_services"], 0);

        let syscall_audit = service
            .execute(read_request(
            "/api/syscall/v1/audit?limit=100",
                json!({"role":"ro","ns":["workflow"],"capabilities":["project_view"]}),
            ))
            .await;
        assert_eq!(syscall_audit.status, StatusCode::OK.as_u16());
        let syscall_audit: Value = serde_json::from_slice(&syscall_audit.body).unwrap();
        let events = syscall_audit["events"].as_array().unwrap();
        let audit_for = |operation: &str| {
            events
                .iter()
                .filter(|event| event["op"] == operation)
                .collect::<Vec<_>>()
        };
        let receive_audit = audit_for("receive_message");
        assert_eq!(receive_audit.len(), 4);
        assert_eq!(
            receive_audit
                .iter()
                .filter(|event| event["status"] == "error")
                .count(),
            2
        );
        let mount_audit = audit_for("mount_memory");
        assert_eq!(mount_audit.len(), 2);
        assert_eq!(mount_audit[0]["status"], "error");
        assert_eq!(mount_audit[1]["status"], "ok");
        let invoke_audit = audit_for("invoke_tool");
        assert_eq!(invoke_audit.len(), 5);
        assert_eq!(
            invoke_audit.iter().filter(|event| event["status"] == "error").count(),
            1
        );

        let auth = AuthConfig {
            mode: "dev".to_owned(),
            algorithm: Some("HS256".to_owned()),
            key: Some("sqlite-chroma-transcript-secret".to_owned()),
            ..AuthConfig::default()
        };
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let token = encode(
            &Header::new(jsonwebtoken::Algorithm::HS256),
            &json!({
                "sub":"reader-1", "role":"ro", "storage_ns":"scope-a",
                "workspace_id":"workspace-a", "graph_space":"conversation",
                "ns":["conversation"], "capabilities":["read_graph","project_view"], "exp":now + 300
            }),
            &EncodingKey::from_secret(b"sqlite-chroma-transcript-secret"),
        )
        .unwrap();
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: vec![],
                implementation: ImplementationSnapshot::default(),
                auth,
            },
            Arc::new(service.clone()),
        );
        let http_response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/syscall/v1/receive_message")
                    .header("content-type", "application/json")
                    .header("authorization", format!("Bearer {token}"))
                    .body(Body::from(r#"{"version":"v1","args":{"conversation_id":"c1"}}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        let http_status = http_response.status();
        let http_body = to_bytes(http_response.into_body(), 65_536).await.unwrap();
        assert_eq!(
            http_status,
            StatusCode::OK,
            "response body: {}",
            String::from_utf8_lossy(&http_body)
        );
        let http_payload: Value = serde_json::from_slice(&http_body).unwrap();
        assert_eq!(http_payload["result"]["messages"][0]["node_id"], "turn-1");
        assert_eq!(http_payload["result"]["messages"][1]["node_id"], "turn-2");
        assert_eq!(http_payload["result"]["messages"].as_array().unwrap().len(), 2);

        let http_mount_memory = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: vec![],
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig {
                    mode: "dev".to_owned(),
                    algorithm: Some("HS256".to_owned()),
                    key: Some("sqlite-chroma-transcript-secret".to_owned()),
                    ..AuthConfig::default()
                },
            },
            Arc::new(service.clone()),
        )
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/syscall/v1/mount_memory")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::from(
                    r#"{"version":"v1","args":{"conversation_id":"c1","run_id":"run-a","stage":"answer"}}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
        let mount_status = http_mount_memory.status();
        let mount_body = to_bytes(http_mount_memory.into_body(), 65_536).await.unwrap();
        assert_eq!(
            mount_status,
            StatusCode::OK,
            "mount_memory response: {}",
            String::from_utf8_lossy(&mount_body)
        );
        let mount_payload: Value = serde_json::from_slice(&mount_body).unwrap();
        assert_eq!(mount_payload["result"]["snapshot_node_id"], "snapshot");
        assert_eq!(mount_payload["result"]["metadata"]["summary"], "stored summary");
        assert!(mount_payload["result"].get("schema_version").is_none());

        let transcript_route = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: vec![],
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig {
                    mode: "dev".to_owned(),
                    algorithm: Some("HS256".to_owned()),
                    key: Some("sqlite-chroma-transcript-secret".to_owned()),
                    ..AuthConfig::default()
                },
            },
            Arc::new(service.clone()),
        )
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/conversations/c1/turns")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
        let transcript_status = transcript_route.status();
        let transcript_body = to_bytes(transcript_route.into_body(), 65_536).await.unwrap();
        assert_eq!(
            transcript_status,
            StatusCode::OK,
            "transcript response: {}",
            String::from_utf8_lossy(&transcript_body)
        );
        let transcript_payload: Value = serde_json::from_slice(&transcript_body).unwrap();
        assert_eq!(transcript_payload["conversation_id"], "c1");
        assert_eq!(transcript_payload["turns"].as_array().unwrap().len(), 2);
        assert_eq!(transcript_payload["turns"][0]["node_id"], "turn-1");
        assert!(transcript_payload.get("schema_version").is_none());
        assert!(transcript_payload["turns"][0].get("schema_version").is_none());
        assert!(service.supports_http_operation("GET", "/api/conversations/c1/turns"));

        assert!(service.supports_mcp_tool("conversation", "conversation.get_transcript"));
        let mcp_app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: vec![],
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig {
                    mode: "dev".to_owned(),
                    algorithm: Some("HS256".to_owned()),
                    key: Some("sqlite-chroma-transcript-secret".to_owned()),
                    ..AuthConfig::default()
                },
            },
            Arc::new(service.clone()),
        );
        let mcp_list = mcp_app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/conversation")
                    .header("content-type", "application/json")
                    .header("authorization", format!("Bearer {token}"))
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":"list-transcript","method":"tools/list"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let mcp_list_body = to_bytes(mcp_list.into_body(), 65_536).await.unwrap();
        let mcp_list_value: Value = serde_json::from_slice(&mcp_list_body).unwrap();
        assert!(mcp_list_value["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .any(|tool| tool["name"] == "conversation.get_transcript"));

        let mcp_transcript = mcp_app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/conversation")
                    .header("content-type", "application/json")
                    .header("authorization", format!("Bearer {token}"))
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":"get-transcript","method":"tools/call","params":{"name":"conversation.get_transcript","arguments":{"conversation_id":"c1"}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(mcp_transcript.status(), StatusCode::OK);
        let mcp_transcript_body = to_bytes(mcp_transcript.into_body(), 65_536).await.unwrap();
        let mcp_transcript_value: Value = serde_json::from_slice(&mcp_transcript_body).unwrap();
        assert_eq!(mcp_transcript_value["result"]["conversation_id"], "c1");
        assert_eq!(mcp_transcript_value["result"]["turns"].as_array().unwrap().len(), 2);
        assert_eq!(mcp_transcript_value["result"]["turns"][0]["node_id"], "turn-1");

        for (uri, expected_path) in [
            ("/api/conversations/c1", "detail"),
            ("/api/conversations", "list"),
        ] {
            let response = router_with_application(
                ApiState {
                    health: snapshot(),
                    required_roles: vec![],
                    implementation: ImplementationSnapshot::default(),
                    auth: AuthConfig {
                        mode: "dev".to_owned(),
                        algorithm: Some("HS256".to_owned()),
                        key: Some("sqlite-chroma-transcript-secret".to_owned()),
                        ..AuthConfig::default()
                    },
                },
                Arc::new(service.clone()),
            )
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri(uri)
                    .header("authorization", format!("Bearer {token}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
            let status = response.status();
            let body = to_bytes(response.into_body(), 65_536).await.unwrap();
            assert_eq!(
                status,
                StatusCode::OK,
                "conversation {expected_path} response: {}",
                String::from_utf8_lossy(&body)
            );
            let payload: Value = serde_json::from_slice(&body).unwrap();
            match expected_path {
                "detail" => {
                    assert_eq!(payload["conversation_id"], "c1");
                    assert_eq!(payload["turn_count"], 2);
                }
                "list" => {
                    assert_eq!(payload["conversations"].as_array().unwrap().len(), 1);
                    assert_eq!(payload["conversations"][0]["turn_count"], 2);
                }
                _ => unreachable!(),
            }
        }

        let detail = service
            .execute(read_request("/api/conversations/c1", principal.clone()))
            .await;
        assert_eq!(detail.status, StatusCode::OK.as_u16());
        let detail: Value = serde_json::from_slice(&detail.body).unwrap();
        assert_eq!(detail["conversation_id"], "c1");
        assert_eq!(detail["user_id"], "reader-1");
        assert_eq!(detail["turn_count"], 2);
        assert!(detail.get("schema_version").is_none());

        let conversations = service
            .execute(read_request("/api/conversations", principal.clone()))
            .await;
        assert_eq!(
            conversations.status,
            StatusCode::OK.as_u16(),
            "conversation list response: {}",
            String::from_utf8_lossy(&conversations.body)
        );
        let conversations: Value = serde_json::from_slice(&conversations.body).unwrap();
        assert_eq!(conversations["conversations"].as_array().unwrap().len(), 1);
        assert_eq!(conversations["conversations"][0]["id"], "c1");
        assert_eq!(conversations["conversations"][0]["turn_count"], 2);
        assert!(service.supports_http_operation("GET", "/api/conversations/c1"));
        assert!(service.supports_http_operation("GET", "/api/conversations"));

        let denied_detail = service
            .execute(read_request(
                "/api/conversations/c1",
                json!({"ns":["conversation"],"capabilities":[]}),
            ))
            .await;
        assert_eq!(denied_detail.status, StatusCode::FORBIDDEN.as_u16());
        let denied_list = service
            .execute(read_request(
                "/api/conversations",
                json!({"sub":"reader-1","ns":["docs"],"capabilities":["read_graph"]}),
            ))
            .await;
        assert_eq!(denied_list.status, StatusCode::FORBIDDEN.as_u16());
        let no_identity = service
            .execute(read_request(
                "/api/conversations",
                json!({"ns":["conversation"],"capabilities":["read_graph"]}),
            ))
            .await;
        assert_eq!(no_identity.status, StatusCode::UNAUTHORIZED.as_u16());

        let _ = stop_chroma.send(());
        let requests = fake.await.unwrap();
        assert!(requests
            .iter()
            .any(|(head, _)| head.contains("collection-edges/get")));
        assert!(requests.iter().filter(|(head, _)| head.contains("collection-nodes/get")).all(
            |(_, body)| body["where"].to_string().contains("workspace_id")
                && body["where"].to_string().contains("conversation")
        ));
        drop(service);
        let _ = std::fs::remove_file(path);

        let metadata_only_path = std::env::temp_dir().join(format!(
            "kogwistar-sqlite-metadata-only-{}.sqlite3",
            uuid::Uuid::new_v4()
        ));
        let metadata_only = SqliteRunApplicationService::open(&metadata_only_path).unwrap();
        assert!(!metadata_only.supports_mcp_tool("conversation", "conversation.get_transcript"));
        let unsupported = metadata_only
            .execute(request(
                "receive_message",
                json!({"conversation_id":"c1"}),
                json!({
                    "sub":"reader-1", "storage_ns":"scope-a", "ns":["conversation"],
                    "capabilities":["read_graph"]
                }),
            ))
            .await;
        assert_eq!(unsupported.status, StatusCode::NOT_IMPLEMENTED.as_u16());
        assert_eq!(
            serde_json::from_slice::<Value>(&unsupported.body).unwrap()["code"],
            "KOGWISTAR_API_CAPABILITY_NOT_CUT_OVER"
        );
        let unsupported_tool = metadata_only
            .execute(request(
                "invoke_tool",
                json!({"tool_name":"list_transcript","conversation_id":"c1"}),
                json!({
                    "sub":"reader-1", "role":"rw", "storage_ns":"scope-a",
                    "ns":["conversation"], "capabilities":["invoke_tool","read_graph"]
                }),
            ))
            .await;
        assert_eq!(unsupported_tool.status, StatusCode::NOT_IMPLEMENTED.as_u16());
        drop(metadata_only);
        let _ = std::fs::remove_file(metadata_only_path);
    }

    #[tokio::test]
    #[ignore = "requires KOGWISTAR_TEST_PG_DSN; full Rust-native answer workflow against disposable PostgreSQL"]
    async fn postgres_agentic_answer_workflow_executes_durably_without_python_worker() {
        use std::process::{Command, Stdio};
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        struct ServerProcess(std::process::Child);

        impl Drop for ServerProcess {
            fn drop(&mut self) {
                let _ = self.0.kill();
                let _ = self.0.wait();
            }
        }

        let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN").unwrap();
        let schema = format!("rust_answer_e2e_{}", uuid::Uuid::new_v4().simple());
        let knowledge_schema = format!("{schema}_knowledge");
        let conversation_schema = format!("{schema}_conversation");
        let workflow_schema = format!("{schema}_workflow");
        let wisdom_schema = format!("{schema}_wisdom");
        let auth_path = std::env::temp_dir().join(format!(
            "kogwistar-answer-e2e-auth-{}.sqlite",
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
        auth.resolve_external_identity(ResolveExternalIdentity {
            issuer: "test".to_owned(),
            subject: "conversation-owner".to_owned(),
            email: "conversation-owner@example.test".to_owned(),
            display_name: None,
            new_user_id: "conversation-owner".to_owned(),
            default_role: "rw".to_owned(),
            default_ns: "conversation".to_owned(),
        })
        .await
        .unwrap();
        drop(auth);
        let auth_config = AuthConfig {
            auth_db_url: Some(format!("sqlite:///{}", auth_path.display())),
            ..AuthConfig::default()
        };

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let chat_address = listener.local_addr().unwrap();
        let snapshot_dsn = dsn.clone();
        let snapshot_schema = knowledge_schema.clone();
        let conversation_graph_schema = conversation_schema.clone();
        let chat_server = tokio::spawn(async move {
            let (mut stream, _) =
                tokio::time::timeout(std::time::Duration::from_secs(120), listener.accept())
                    .await
                    .unwrap()
                    .unwrap();
            let mut bytes = Vec::new();
            let header_end = loop {
                let mut chunk = [0_u8; 4096];
                let count = stream.read(&mut chunk).await.unwrap();
                assert!(count > 0);
                bytes.extend_from_slice(&chunk[..count]);
                if let Some(position) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
                    break position + 4;
                }
            };
            let headers = String::from_utf8_lossy(&bytes[..header_end]).to_ascii_lowercase();
            let content_length = headers
                .lines()
                .find_map(|line| line.strip_prefix("content-length: "))
                .unwrap()
                .trim()
                .parse::<usize>()
                .unwrap();
            while bytes.len() < header_end + content_length {
                let mut chunk = [0_u8; 4096];
                let count = stream.read(&mut chunk).await.unwrap();
                assert!(count > 0);
                bytes.extend_from_slice(&chunk[..count]);
            }
            let request: Value =
                serde_json::from_slice(&bytes[header_end..header_end + content_length]).unwrap();
            assert_eq!(request["response_format"]["type"], "json_schema");
            assert_eq!(request["stream"], true);
            let snapshot_reader =
                PostgresStore::from_dsn(&snapshot_dsn, &snapshot_schema).unwrap();
            let conversation_reader =
                PostgresStore::from_dsn(&snapshot_dsn, &conversation_graph_schema).unwrap();
            let snapshots = conversation_reader
                .graph_projection_records(GraphProjectionRead {
                    scope: GraphScope {
                        namespace: "alpha".to_owned(),
                        workspace_id: None,
                        graph_space: None,
                    },
                    table: "gke_nodes".to_owned(),
                    ids: None,
                    metadata: MetadataFilter {
                        equals: BTreeMap::from([
                            ("entity_type".to_owned(), json!("context_snapshot")),
                            ("run_id".to_owned(), json!("rust-only-answer-e2e")),
                        ]),
                    },
                    limit: 10,
                })
                .await
                .unwrap();
            assert!(snapshots.iter().any(|record| {
                record.metadata.get("entity_type").and_then(Value::as_str)
                    == Some("context_snapshot")
                    && record.metadata.get("stage").and_then(Value::as_str)
                        == Some("select_used_evidence")
                    && record.metadata.get("run_id").and_then(Value::as_str)
                        == Some("rust-only-answer-e2e")
            }), "default evidence selection snapshot must commit before provider invocation");
            let snapshot = snapshots.iter().find(|record| {
                record.metadata.get("entity_type").and_then(Value::as_str)
                    == Some("context_snapshot")
                    && record.metadata.get("stage").and_then(Value::as_str)
                        == Some("generate_answer_with_citations")
                    && record.metadata.get("run_id").and_then(Value::as_str)
                        == Some("rust-only-answer-e2e")
            });
            let snapshot = snapshot.expect(
                "the model request must not start before its prompt snapshot is durable",
            );
            let snapshot_document: Value = serde_json::from_str(
                snapshot.document.as_deref().expect("snapshot has document"),
            )
            .unwrap();
            let prompt_messages: Value = serde_json::from_str(
                snapshot_document["properties"]["prompt_messages"]
                    .as_str()
                    .expect("snapshot records provider prompt messages"),
            )
            .unwrap();
            assert!(prompt_messages[1]["content"]
                .as_str()
                .unwrap()
                .contains("Where is workflow runtime state stored?"));
            let conversation_context: Value = serde_json::from_str(
                snapshot_document["properties"]["conversation_context_messages"]
                    .as_str()
                    .expect("snapshot records authorized conversation context"),
            )
            .unwrap();
            assert_eq!(conversation_context.as_array().unwrap().len(), 3);
            assert_eq!(conversation_context[1]["content"], "Earlier user question");
            assert_eq!(conversation_context[1]["node_id"], "answer-prior-turn");
            assert_eq!(conversation_context[2]["node_id"], "answer-user-turn");
            assert!(conversation_context[0]["content"]
                .as_str()
                .unwrap()
                .contains("Conversation summary:\nEarlier conversation summary"));
            assert!(conversation_context[0]["content"]
                .as_str()
                .unwrap()
                .contains("Memory context:\nPinned memory detail"));
            assert!(conversation_context[0]["content"]
                .as_str()
                .unwrap()
                .contains("Pinned knowledge references:\n- Architecture reference | A pinned graph reference"));
            assert!(!conversation_context[0]["content"]
                .as_str()
                .unwrap()
                .contains("Bob-private reference"));
            let dependency_edges = conversation_reader
                .graph_projection_records(GraphProjectionRead {
                    scope: GraphScope {
                        namespace: "alpha".to_owned(),
                        workspace_id: None,
                        graph_space: None,
                    },
                    table: "gke_edges".to_owned(),
                    ids: None,
                    metadata: MetadataFilter::default(),
                    limit: 10_000,
                })
                .await
                .unwrap();
            assert!(dependency_edges.iter().any(|record| {
                record
                    .document
                    .as_deref()
                    .and_then(|document| serde_json::from_str::<Value>(document).ok())
                    .is_some_and(|document| {
                        document["relation"] == "depends_on"
                            && document["source_ids"] == json!([snapshot.id])
                            && document["target_ids"] == json!(["answer-evidence-node"])
                    })
            }), "snapshot evidence dependency must commit before the provider call");
            for context_node_id in [
                "answer-memory-summary",
                "answer-prior-turn",
                "answer-user-turn",
                "answer-memory-context",
                "answer-knowledge-pointer",
            ] {
                assert!(dependency_edges.iter().any(|record| {
                    record
                        .document
                        .as_deref()
                        .and_then(|document| serde_json::from_str::<Value>(document).ok())
                        .is_some_and(|document| {
                            document["relation"] == "depends_on"
                                && document["source_ids"] == json!([snapshot.id])
                                && document["target_ids"] == json!([context_node_id])
                        })
                }), "snapshot must durably depend on context node {context_node_id}");
            }
            let answer = json!({
                    "text":"The workflow runtime state is stored in the graph.",
                    "reasoning":"The selected evidence states this directly.",
                    "claims":[{"claim":"Workflow runtime state is stored in the graph.","citations":[{"source_node_id":"N1","mention_index":0,"span_index":0}]}]
                })
                .to_string();
            let (text_field, remaining_fields) = answer
                .split_once(",\"reasoning\"")
                .expect("answer fixture has a top-level reasoning field");
            let chunks = [
                json!({"id":"resp-agentic-e2e","choices":[{"delta":{"content":format!("{text_field},")},"finish_reason":null}]}),
                json!({"id":"resp-agentic-e2e","choices":[{"delta":{"content":format!("\"reasoning\"{remaining_fields}")},"finish_reason":null}]}),
                json!({"id":"resp-agentic-e2e","choices":[{"delta":{},"finish_reason":"stop"}]}),
            ];
            let frames = chunks
                .iter()
                .map(|chunk| format!("data: {chunk}\r\n\r\n"))
                .collect::<Vec<_>>();
            let content_length = frames
                .iter()
                .map(String::len)
                .sum::<usize>()
                + "data: [DONE]\r\n\r\n".len();
            let headers = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {content_length}\r\nconnection: close\r\n\r\n"
            );
            stream.write_all(headers.as_bytes()).await.unwrap();
            stream.flush().await.unwrap();
            for (index, frame) in frames.iter().enumerate() {
                stream.write_all(frame.as_bytes()).await.unwrap();
                stream.flush().await.unwrap();
                if index == 0 {
                    let persisted = tokio::time::timeout(std::time::Duration::from_secs(5), async {
                        loop {
                            let events = snapshot_reader
                                .list_server_run_events("rust-only-answer-e2e", 0, 10_000)
                                .await
                                .unwrap();
                            let trace_was_persisted = events.iter().any(|event| {
                                event.event_type == "workflow.provider_event.v1"
                                    && serde_json::from_str::<Value>(&event.payload_json)
                                        .ok()
                                        .is_some_and(|payload| {
                                            payload["provider_event"]["kind"] == "field_completed"
                                                && payload["provider_event"]["details"]["field_name"] == "text"
                                        })
                            });
                            let lanes = snapshot_reader
                                .list_projected_lane_messages(LaneMessageFilter {
                                    namespace: Some("alpha".to_owned()),
                                    inbox_id: Some("agent-provider-events".to_owned()),
                                    conversation_id: Some("answer-e2e-conversation".to_owned()),
                                    msg_type: Some("agent.provider_event".to_owned()),
                                    limit: 100,
                                    ..LaneMessageFilter::default()
                                })
                                .await
                                .unwrap();
                            let event_nodes = conversation_reader
                                .graph_projection_records(GraphProjectionRead {
                                    scope: GraphScope {
                                        namespace: "alpha".to_owned(),
                                        workspace_id: None,
                                        graph_space: Some("conversation".to_owned()),
                                    },
                                    table: "gke_nodes".to_owned(),
                                    ids: None,
                                    metadata: MetadataFilter {
                                        equals: BTreeMap::from([
                                            ("msg_type".to_owned(), json!("agent.provider_event")),
                                            ("conversation_id".to_owned(), json!("answer-e2e-conversation")),
                                        ]),
                                    },
                                    limit: 100,
                                })
                                .await
                                .unwrap();
                            let lane_node_match = lanes.iter().any(|lane| {
                                event_nodes.iter().any(|node| {
                                    node.id == lane.message_id
                                        && node.metadata.get("payload_json").and_then(Value::as_str)
                                            == lane.payload_json.as_deref()
                                })
                            });
                            let event_node_ids = event_nodes
                                .iter()
                                .map(|node| node.id.clone())
                                .collect::<Vec<_>>();
                            let event_acl = conversation_reader
                                .graph_acl_records_for_entities(
                                    &GraphScope {
                                        namespace: "alpha".to_owned(),
                                        workspace_id: None,
                                        graph_space: Some("conversation".to_owned()),
                                    },
                                    "conversation",
                                    &event_node_ids,
                                )
                                .await
                                .unwrap();
                            let private_acl_match = !event_node_ids.is_empty()
                                && event_node_ids.iter().all(|event_node_id| {
                                    event_acl.iter().any(|acl| {
                                        acl.metadata.get("acl_target_entity_id").and_then(Value::as_str)
                                            == Some(event_node_id.as_str())
                                            && acl.metadata.get("acl_mode").and_then(Value::as_str)
                                                == Some("private")
                                            && acl.metadata.get("owner_id").and_then(Value::as_str)
                                                == Some("alice")
                                    })
                                });
                            if trace_was_persisted && lane_node_match && private_acl_match {
                                break;
                            }
                            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                        }
                    })
                    .await;
                    if persisted.is_err() {
                        let events = snapshot_reader
                            .list_server_run_events("rust-only-answer-e2e", 0, 10_000)
                            .await
                            .unwrap();
                        let event_types = events
                            .iter()
                            .map(|event| event.event_type.as_str())
                            .collect::<Vec<_>>();
                        let lanes = snapshot_reader
                            .list_projected_lane_messages(LaneMessageFilter {
                                namespace: Some("alpha".to_owned()),
                                inbox_id: Some("agent-provider-events".to_owned()),
                                conversation_id: Some("answer-e2e-conversation".to_owned()),
                                msg_type: Some("agent.provider_event".to_owned()),
                                limit: 100,
                                ..LaneMessageFilter::default()
                            })
                            .await
                            .unwrap();
                        let event_nodes = conversation_reader
                            .graph_projection_records(GraphProjectionRead {
                                scope: GraphScope {
                                    namespace: "alpha".to_owned(),
                                    workspace_id: None,
                                    graph_space: Some("conversation".to_owned()),
                                },
                                table: "gke_nodes".to_owned(),
                                ids: None,
                                metadata: MetadataFilter {
                                    equals: BTreeMap::from([(
                                        "msg_type".to_owned(),
                                        json!("agent.provider_event"),
                                    )]),
                                },
                                limit: 100,
                            })
                            .await
                            .unwrap();
                        let event_node_ids = event_nodes
                            .iter()
                            .map(|node| node.id.clone())
                            .collect::<Vec<_>>();
                        let event_acl = conversation_reader
                            .graph_acl_records_for_entities(
                                &GraphScope {
                                    namespace: "alpha".to_owned(),
                                    workspace_id: None,
                                    graph_space: Some("conversation".to_owned()),
                                },
                                "conversation",
                                &event_node_ids,
                            )
                            .await
                            .unwrap();
                        panic!(
                            "complete semantic fields persist before provider completion; run events={event_types:?}, lanes={}, event_nodes={event_node_ids:?}, acl={:?}",
                            lanes.len(),
                            event_acl.iter().map(|acl| &acl.metadata).collect::<Vec<_>>()
                        );
                    }
                    let misplaced_event_nodes = snapshot_reader
                        .graph_projection_records(GraphProjectionRead {
                            scope: GraphScope {
                                namespace: "alpha".to_owned(),
                                workspace_id: None,
                                graph_space: Some("conversation".to_owned()),
                            },
                            table: "gke_nodes".to_owned(),
                            ids: None,
                            metadata: MetadataFilter {
                                equals: BTreeMap::from([
                                    ("msg_type".to_owned(), json!("agent.provider_event")),
                                    (
                                        "conversation_id".to_owned(),
                                        json!("answer-e2e-conversation"),
                                    ),
                                ]),
                            },
                            limit: 100,
                        })
                        .await
                        .unwrap();
                    assert!(
                        misplaced_event_nodes.is_empty(),
                        "conversation provider-event graph nodes must not leak into the knowledge schema"
                    );
                }
            }
            stream.write_all(b"data: [DONE]\r\n\r\n").await.unwrap();
            stream.flush().await.unwrap();
        });
        let embedding_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let embedding_address = embedding_listener.local_addr().unwrap();
        let batch_embedding_requests = Arc::new(Mutex::new(Vec::<Value>::new()));
        let batch_requests = Arc::clone(&batch_embedding_requests);
        let legacy_embedding_requests = Arc::new(Mutex::new(Vec::<Value>::new()));
        let legacy_requests = Arc::clone(&legacy_embedding_requests);
        let embedding_app = axum::Router::new()
            .route(
                "/api/embed",
                axum::routing::post(move |axum::Json(request): axum::Json<Value>| {
                    let requests = Arc::clone(&batch_requests);
                    async move {
                        let input_count = request["input"]
                            .as_array()
                            .expect("Ollama batch input is an array")
                            .len();
                        requests.lock().unwrap().push(request);
                        axum::Json(json!({
                            "embeddings":vec![vec![0.6,0.8,0.0]; input_count]
                        }))
                    }
                }),
            )
            .route(
                "/api/embeddings",
                axum::routing::post(move |axum::Json(request): axum::Json<Value>| {
                    let requests = Arc::clone(&legacy_requests);
                    async move {
                        requests.lock().unwrap().push(request);
                        axum::Json(json!({"embedding":[0.6,0.8,0.0]}))
                    }
                }),
            );
        let embedding_server = tokio::spawn(async move {
            axum::serve(embedding_listener, embedding_app).await.unwrap();
        });
        let chat_provider = crate::native_answer_generation::OpenAiCompatibleChatConfig::new(
            format!("http://{chat_address}/v1/chat/completions"),
            "fixture-answer-model",
            None,
        )
        .unwrap()
        .with_streaming(true);
        let service = PostgresRunApplicationService::from_dsn_with_graph_schemas(
            &dsn,
            &knowledge_schema,
            &conversation_schema,
            &workflow_schema,
            &wisdom_schema,
        )
            .unwrap()
            .with_native_operations(crate::native_answer_persist::registry().unwrap())
            .with_owner_auth_config(&auth_config)
            .await
            .unwrap()
            .with_embedding_provider(Arc::new(FixedEmbeddingProvider {
                profile: kogwistar_engine::embedding::EmbeddingProfile {
                    provider: "ollama".to_owned(),
                    model: "fixture-embedding-model".to_owned(),
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
            }))
            .with_chat_completion_provider(chat_provider);
        service.ensure_graph_type_schemas().await.unwrap();
        for store in [
            &service.store,
            &service.conversation_store,
            &service.workflow_store,
            &service.wisdom_store,
        ] {
            store
                .create_graph_schema(3, kogwistar_store_postgres::GraphTableNames::default())
                .await
                .unwrap();
        }

        let service_declaration = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/workflow/services".to_owned(),
                body: serde_json::to_vec(&json!({
                    "service_id":"resource-snapshot-service",
                    "target_kind":"workflow",
                    "target_ref":"resource-snapshot-workflow",
                    "target_config": {
                        "conversation_id":"resource-snapshot-conversation",
                        "runtime_kind":"sync",
                        "user_id":"alice"
                    },
                    "enabled":true
                }))
                .unwrap(),
                principal: json!({
                    "sub":"operator","role":"rw","ns":"workflow",
                    "capabilities":["service.manage","write_graph"]
                }),
            })
            .await;
        assert_eq!(
            service_declaration.status,
            StatusCode::OK.as_u16(),
            "{}",
            String::from_utf8_lossy(&service_declaration.body)
        );

        let resource_snapshot = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/syscall/v1/invoke_tool".to_owned(),
                body: serde_json::to_vec(&json!({
                    "version":"v1",
                    "op":"invoke_tool",
                    "args":{"tool_name":"resource_snapshot"}
                }))
                .unwrap(),
                principal: json!({
                    "sub":"alice","role":"rw","ns":"conversation",
                    "capabilities":["invoke_tool","project_view","read_security_scope"]
                }),
            })
            .await;
        assert_eq!(
            resource_snapshot.status,
            StatusCode::OK.as_u16(),
            "{}",
            String::from_utf8_lossy(&resource_snapshot.body)
        );
        let resource_snapshot: Value = serde_json::from_slice(&resource_snapshot.body).unwrap();
        assert_eq!(resource_snapshot["result"]["tool_name"], "resource_snapshot");
        assert_eq!(
            resource_snapshot["result"]["snapshot"]["services"],
            json!({"total_services":1,"by_health":{"degraded":1}})
        );
        let snapshot_fields = resource_snapshot["result"]["snapshot"].as_object().unwrap();
        assert!(snapshot_fields["storage_usage_bytes"].is_u64());
        assert_eq!(
            snapshot_fields["storage_usage_bytes"],
            json!(configured_storage_usage_bytes()),
            "the live MCP projection must use the configured persistence directories"
        );
        assert_eq!(
            snapshot_fields
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([
                "scheduler",
                "runs",
                "services",
                "storage_usage_bytes",
                "cost_ledger",
                "budget_model",
                "policy_infra",
            ]),
            "legacy resource snapshot keys must match Python without Rust-only diagnostics"
        );

        let put_node = |id: &str,
                        event_id: &str,
                        metadata: serde_json::Map<String, Value>,
                        document: Option<String>,
                        embedding: Option<Vec<f32>>| {
            GraphMutation {
                scope: GraphScope {
                    namespace: "alpha".to_owned(),
                    workspace_id: None,
                    graph_space: None,
                },
                table: "gke_nodes".to_owned(),
                entity_kind: "node".to_owned(),
                event_id: event_id.to_owned(),
                op: "ADD".to_owned(),
                payload: json!({"id":id}),
                record: GraphRecord {
                    id: id.to_owned(),
                    document,
                    metadata,
                    embedding,
                },
                embedding_dim: 3,
            }
        };
        let acl = |target: &str, truth_graph: &str, mode: &str| {
            put_node(
                &format!("acl-{target}"),
                &format!("e2e-acl-{target}"),
                serde_json::Map::from_iter([
                    ("entity_type".to_owned(), json!("acl_record")),
                    ("acl_truth_graph".to_owned(), json!(truth_graph)),
                    ("acl_target_grain".to_owned(), json!("node")),
                    ("acl_target_entity_id".to_owned(), json!(target)),
                    ("acl_version".to_owned(), json!(1)),
                    ("acl_mode".to_owned(), json!(mode)),
                    ("owner_id".to_owned(), json!("alice")),
                ]),
                None,
                None,
            )
        };
        let acl_edge = |target: &str, owner: &str| {
            put_node(
                &format!("acl-edge-{target}"),
                &format!("e2e-acl-edge-{target}"),
                serde_json::Map::from_iter([
                    ("entity_type".to_owned(), json!("acl_record")),
                    ("acl_truth_graph".to_owned(), json!("conversation")),
                    ("acl_target_grain".to_owned(), json!("edge")),
                    ("acl_target_entity_id".to_owned(), json!(target)),
                    ("acl_version".to_owned(), json!(1)),
                    ("acl_mode".to_owned(), json!("private")),
                    ("owner_id".to_owned(), json!(owner)),
                ]),
                None,
                None,
            )
        };
        let put_edge = |id: &str, relation: &str, source: &str, target: &str| {
            let payload = json!({
                "id": id,
                "relation": relation,
                "source_ids": [source],
                "target_ids": [target],
                "doc_id": "conv:answer-e2e-conversation"
            });
            GraphMutation {
                scope: GraphScope {
                    namespace: "alpha".to_owned(),
                    workspace_id: None,
                    graph_space: None,
                },
                table: "gke_edges".to_owned(),
                entity_kind: "edge".to_owned(),
                event_id: format!("e2e-{id}"),
                op: "ADD".to_owned(),
                payload: payload.clone(),
                record: GraphRecord {
                    id: id.to_owned(),
                    document: Some(payload.to_string()),
                    metadata: serde_json::Map::from_iter([
                        ("entity_type".to_owned(), json!("conversation_edge")),
                        ("doc_id".to_owned(), json!("conv:answer-e2e-conversation")),
                        ("relation".to_owned(), json!(relation)),
                        ("source_ids".to_owned(), json!([source])),
                        ("target_ids".to_owned(), json!([target])),
                        ("lifecycle_status".to_owned(), json!("active")),
                    ]),
                    embedding: None,
                },
                embedding_dim: 3,
            }
        };
        let item_acl = |target: &str, truth_graph: &str, grain: &str, item_id: &str, mode: &str| {
            put_node(
                &format!("acl-{grain}-{item_id}"),
                &format!("e2e-acl-{grain}-{item_id}"),
                serde_json::Map::from_iter([
                    ("entity_type".to_owned(), json!("acl_record")),
                    ("acl_truth_graph".to_owned(), json!(truth_graph)),
                    ("acl_target_grain".to_owned(), json!(grain)),
                    ("acl_target_entity_id".to_owned(), json!(target)),
                    ("acl_target_item_id".to_owned(), json!(item_id)),
                    ("acl_version".to_owned(), json!(1)),
                    ("acl_mode".to_owned(), json!(mode)),
                    ("owner_id".to_owned(), json!("alice")),
                ]),
                None,
                None,
            )
        };
        for mutation in [
            put_node(
                "answer-prior-turn",
                "e2e-prior-turn",
                serde_json::Map::from_iter([
                    ("entity_type".to_owned(), json!("conversation_turn")),
                    (
                        "conversation_id".to_owned(),
                        json!("answer-e2e-conversation"),
                    ),
                    ("role".to_owned(), json!("user")),
                    ("turn_index".to_owned(), json!(0)),
                    ("in_conversation_chain".to_owned(), json!(true)),
                    ("lifecycle_status".to_owned(), json!("active")),
                ]),
                Some(
                    json!({"summary":"Earlier user question","metadata":{
                        "entity_type":"conversation_turn","conversation_id":"answer-e2e-conversation",
                        "role":"user","turn_index":0,"lifecycle_status":"active"
                    }})
                    .to_string(),
                ),
                None,
            ),
            acl("answer-prior-turn", "conversation", "private"),
            put_node(
                "answer-hidden-turn",
                "e2e-hidden-turn",
                serde_json::Map::from_iter([
                    ("entity_type".to_owned(), json!("conversation_turn")),
                    (
                        "conversation_id".to_owned(),
                        json!("answer-e2e-conversation"),
                    ),
                    ("role".to_owned(), json!("assistant")),
                    ("turn_index".to_owned(), json!(0)),
                    ("in_conversation_chain".to_owned(), json!(true)),
                    ("lifecycle_status".to_owned(), json!("active")),
                ]),
                Some(
                    json!({"summary":"Not visible to Alice","metadata":{
                        "entity_type":"conversation_turn","conversation_id":"answer-e2e-conversation",
                        "role":"assistant","turn_index":0,"lifecycle_status":"active"
                    }})
                    .to_string(),
                ),
                None,
            ),
            put_node(
                "acl-answer-hidden-turn",
                "e2e-acl-hidden-turn",
                serde_json::Map::from_iter([
                    ("entity_type".to_owned(), json!("acl_record")),
                    ("acl_truth_graph".to_owned(), json!("conversation")),
                    ("acl_target_grain".to_owned(), json!("node")),
                    ("acl_target_entity_id".to_owned(), json!("answer-hidden-turn")),
                    ("acl_version".to_owned(), json!(1)),
                    ("acl_mode".to_owned(), json!("private")),
                    ("owner_id".to_owned(), json!("bob")),
                ]),
                None,
                None,
            ),
            put_node(
                "answer-user-turn",
                "e2e-user-turn",
                serde_json::Map::from_iter([
                    ("entity_type".to_owned(), json!("conversation_turn")),
                    (
                        "conversation_id".to_owned(),
                        json!("answer-e2e-conversation"),
                    ),
                    ("role".to_owned(), json!("user")),
                    ("turn_index".to_owned(), json!(1)),
                    ("in_conversation_chain".to_owned(), json!(true)),
                    ("lifecycle_status".to_owned(), json!("active")),
                ]),
                Some(
                    json!({"summary":"Where is workflow runtime state stored?","metadata":{
                        "entity_type":"conversation_turn","conversation_id":"answer-e2e-conversation",
                        "role":"user","turn_index":1,"lifecycle_status":"active"
                    }})
                    .to_string(),
                ),
                None,
            ),
            acl("answer-user-turn", "conversation", "private"),
            put_node(
                "answer-memory-summary",
                "e2e-memory-summary",
                serde_json::Map::from_iter([
                    ("entity_type".to_owned(), json!("memory_summary")),
                    ("conversation_id".to_owned(), json!("answer-e2e-conversation")),
                    ("lifecycle_status".to_owned(), json!("active")),
                ]),
                Some(
                    json!({"type":"memory_summary","summary":"Earlier conversation summary","turn_index":0,"metadata":{
                        "entity_type":"memory_summary","conversation_id":"answer-e2e-conversation",
                        "lifecycle_status":"active"
                    }})
                    .to_string(),
                ),
                None,
            ),
            acl("answer-memory-summary", "conversation", "private"),
            put_node(
                "answer-memory-context",
                "e2e-memory-context",
                serde_json::Map::from_iter([
                    ("entity_type".to_owned(), json!("memory_context")),
                    (
                        "conversation_id".to_owned(),
                        json!("answer-e2e-conversation"),
                    ),
                    ("lifecycle_status".to_owned(), json!("active")),
                ]),
                Some(
                    json!({"summary":"Pinned memory detail","metadata":{
                        "entity_type":"memory_context","conversation_id":"answer-e2e-conversation",
                        "lifecycle_status":"active"
                    }})
                    .to_string(),
                ),
                None,
            ),
            acl("answer-memory-context", "conversation", "private"),
            put_node(
                "answer-knowledge-pointer",
                "e2e-knowledge-pointer",
                serde_json::Map::from_iter([
                    ("entity_type".to_owned(), json!("knowledge_reference")),
                    (
                        "conversation_id".to_owned(),
                        json!("answer-e2e-conversation"),
                    ),
                    ("lifecycle_status".to_owned(), json!("active")),
                ]),
                Some(
                    json!({
                        "label":"Architecture reference",
                        "summary":"A pinned graph reference",
                        "properties":{"refers_to_id":"answer-evidence-node"},
                        "metadata":{
                            "entity_type":"knowledge_reference",
                            "conversation_id":"answer-e2e-conversation",
                            "lifecycle_status":"active"
                        }
                    })
                    .to_string(),
                ),
                None,
            ),
            acl("answer-knowledge-pointer", "conversation", "private"),
            put_node(
                "answer-hidden-knowledge-pointer",
                "e2e-hidden-knowledge-pointer",
                serde_json::Map::from_iter([
                    ("entity_type".to_owned(), json!("knowledge_reference")),
                    ("conversation_id".to_owned(), json!("answer-e2e-conversation")),
                    ("lifecycle_status".to_owned(), json!("active")),
                ]),
                Some(
                    json!({"label":"Bob-private reference","summary":"Must not enter Alice's prompt","metadata":{
                        "entity_type":"knowledge_reference","conversation_id":"answer-e2e-conversation",
                        "lifecycle_status":"active"
                    }})
                    .to_string(),
                ),
                None,
            ),
            put_node(
                "acl-node-answer-hidden-knowledge-pointer",
                "e2e-acl-hidden-knowledge-pointer",
                serde_json::Map::from_iter([
                    ("entity_type".to_owned(), json!("acl_record")),
                    ("acl_truth_graph".to_owned(), json!("conversation")),
                    ("acl_target_grain".to_owned(), json!("node")),
                    ("acl_target_entity_id".to_owned(), json!("answer-hidden-knowledge-pointer")),
                    ("acl_version".to_owned(), json!(1)),
                    ("acl_mode".to_owned(), json!("private")),
                    ("owner_id".to_owned(), json!("bob")),
                ]),
                None,
                None,
            ),
            put_edge(
                "answer-memory-edge",
                "has_memory_context",
                "answer-user-turn",
                "answer-memory-context",
            ),
            acl_edge("answer-memory-edge", "alice"),
            put_edge(
                "answer-reference-edge",
                "references",
                "answer-user-turn",
                "answer-knowledge-pointer",
            ),
            acl_edge("answer-reference-edge", "alice"),
            put_edge(
                "answer-hidden-reference-edge",
                "references",
                "answer-user-turn",
                "answer-hidden-knowledge-pointer",
            ),
            acl_edge("answer-hidden-reference-edge", "bob"),
            put_node(
                "answer-evidence-node",
                "e2e-evidence-node",
                serde_json::Map::from_iter([
                    ("entity_type".to_owned(), json!("knowledge_node")),
                    ("label".to_owned(), json!("Rust workflow runtime")),
                    (
                        "summary".to_owned(),
                        json!("The workflow runtime records state in a durable graph."),
                    ),
                    ("level_from_root".to_owned(), json!(0)),
                    ("lifecycle_status".to_owned(), json!("active")),
                    (
                        "mentions".to_owned(),
                        json!([{"spans":[{"excerpt":"durable graph"}]}]),
                    ),
                ]),
                Some("The workflow runtime records state in a durable graph.".to_owned()),
                Some(vec![0.6_f32, 0.8, 0.0]),
            ),
            acl("answer-evidence-node", "knowledge", "public"),
            item_acl(
                "answer-evidence-node",
                "knowledge",
                "grounding",
                "answer-evidence-node::mention::0",
                "public",
            ),
            item_acl(
                "answer-evidence-node",
                "knowledge",
                "span",
                "answer-evidence-node::mention::0::span::0",
                "public",
            ),
        ] {
            let graph_type = mutation
                .record
                .metadata
                .get("acl_truth_graph")
                .and_then(Value::as_str)
                .or_else(|| {
                    mutation
                        .record
                        .metadata
                        .get("entity_type")
                        .and_then(Value::as_str)
                        .filter(|entity_type| entity_type.starts_with("conversation_"))
                        .map(|_| "conversation")
                })
                .or_else(|| {
                    mutation
                        .record
                        .metadata
                        .contains_key("conversation_id")
                        .then_some("conversation")
                })
                .unwrap_or("knowledge");
            let store = match graph_type {
                "conversation" => &service.conversation_store,
                _ => &service.store,
            };
            store.apply_graph_mutation(mutation).await.unwrap();
        }

        let direct_vector_hits = service
            .store
            .graph_projection_vector_query_level_lte_excluding_metadata(
                GraphProjectionVectorQuery {
                    scope: GraphScope {
                        namespace: "alpha".to_owned(),
                        workspace_id: None,
                        graph_space: None,
                    },
                    table: "gke_nodes".to_owned(),
                    query: VectorQuery {
                        embedding: vec![0.6, 0.8, 0.0],
                        limit: 10,
                        metadata: MetadataFilter::default(),
                        metric: DistanceMetric::Cosine,
                    },
                    embedding_dim: 3,
                },
                std::collections::BTreeMap::from([("entity_type".to_owned(), json!("acl_record"))]),
                3,
            )
            .await
            .unwrap();
        assert!(
            direct_vector_hits
                .iter()
                .any(|hit| hit.record.id == "answer-evidence-node"),
            "seeded evidence must be retrievable before workflow admission: {direct_vector_hits:#?}"
        );

        let contract: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/answer-workflow-runtime-contract-v1.json"
        )))
        .unwrap();
        let workflow = &contract["workflows"]["agentic_answering.v2"];
        let ops = workflow["required_ops"].as_array().unwrap();
        let design_nodes = ops.iter().map(|value| {
                let op = value.as_str().unwrap();
                json!({"id":op,"label":op,"metadata":{
                    "wf_op":op,"wf_version":"v1","wf_executor_target":"rust-native","wf_start":op=="start"
                }})
            }).collect::<Vec<_>>();
        let design_edges = workflow["required_routes"].as_array().unwrap().iter().enumerate().map(|(index,route)| json!({
                "id":format!("answer-e2e-route-{index}"),"source_ids":[route["from"]],"target_ids":[route["to"]],
                "metadata":{"wf_predicate":route["predicate"],"wf_is_default":route["default"],
                    "wf_priority":route["priority"],"wf_multiplicity":route["multiplicity"]}
            })).collect::<Vec<_>>();
        service
            .store
            .replace_named_projection(
                "workflow_design",
                "agentic_answering.v2",
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
                "agentic_answering.v2",
                WorkflowDesignSnapshotWrite {
                    version: 1,
                    seq: 1,
                    schema_version: 1,
                    payload_json: serde_json::to_string(
                        &json!({"nodes":design_nodes,"edges":design_edges}),
                    )
                    .unwrap(),
                },
            )
            .await
            .unwrap();
        let debug_workflow = &contract["workflows"]["debug.rag.v1"];
        let debug_nodes = debug_workflow["required_ops"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| {
                let op = value.as_str().unwrap();
                json!({"id":op,"label":op,"metadata":{
                    "wf_op":op,"wf_version":"v1","wf_executor_target":"rust-native","wf_start":op=="start"
                }})
            })
            .collect::<Vec<_>>();
        let debug_edges = debug_workflow["required_routes"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(index, route)| {
                json!({
                    "id":format!("debug-e2e-route-{index}"),"source_ids":[route["from"]],"target_ids":[route["to"]],
                    "metadata":{"wf_predicate":route["predicate"],"wf_is_default":route["default"],
                        "wf_priority":route["priority"],"wf_multiplicity":route["multiplicity"]}
                })
            })
            .collect::<Vec<_>>();
        service
            .store
            .replace_named_projection(
                "workflow_design",
                "debug.rag.v1",
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
                "debug.rag.v1",
                WorkflowDesignSnapshotWrite {
                    version: 1,
                    seq: 1,
                    schema_version: 1,
                    payload_json: serde_json::to_string(
                        &json!({"nodes":debug_nodes,"edges":debug_edges}),
                    )
                    .unwrap(),
                },
            )
            .await
            .unwrap();
        let http_secret = "rust-only-answer-http-secret";
        let now_seconds = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let token = jsonwebtoken::encode(
            &jsonwebtoken::Header::new(jsonwebtoken::Algorithm::HS256),
            &json!({
                "sub":"alice","user_id":"alice","actor_id":"alice","role":"rw","roles":["rw"],
                "ns":"alpha","storage_ns":"alpha",
                "capabilities":["workflow.run.write","workflow.design.write","workflow.design.inspect","read_graph","write_graph"],
                "exp": now_seconds + 300
            }),
            &jsonwebtoken::EncodingKey::from_secret(http_secret.as_bytes()),
        )
        .unwrap();
        let current_exe = std::env::current_exe().unwrap();
        let server_dir = current_exe
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();
        let server_binary = server_dir.join(if cfg!(windows) {
            "kogwistar-server.exe"
        } else {
            "kogwistar-server"
        });
        assert!(
            server_binary.is_file(),
            "packaged Rust server binary is missing at {}; build all targets before this ignored integration test",
            server_binary.display()
        );
        let server_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let server_port = server_listener.local_addr().unwrap().port();
        drop(server_listener);
        let server_log_path = std::env::temp_dir().join(format!(
            "kogwistar-answer-e2e-server-{}.log",
            uuid::Uuid::new_v4().simple()
        ));
        let server_log = std::fs::File::create(&server_log_path).unwrap();
        let mut server_command = Command::new(&server_binary);
        server_command.env_clear();
        for name in ["PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "USERPROFILE"] {
            if let Some(value) = std::env::var_os(name) {
                server_command.env(name, value);
            }
        }
        server_command
            .env("KOGWISTAR_BACKEND", "pg")
            .env("KOGWISTAR_PG_DSN", &dsn)
            .env("KOGWISTAR_PG_SCHEMA_BASE", &schema)
            .env("KOGWISTAR_SERVER_HOST", "127.0.0.1")
            .env("KOGWISTAR_SERVER_PORT", server_port.to_string())
            .env("KOGWISTAR_SERVER_REQUIRED_ROLES", "rw")
            .env("KOGWISTAR_RUNTIME_NATIVE_INLINE", "true")
            .env("AUTH_MODE", "jwt")
            .env("JWT_ALG", "HS256")
            .env("JWT_SECRET", http_secret)
            .env("AUTH_DB_URL", format!("sqlite:///{}", auth_path.display()))
            .env(
                "KOGWISTAR_CHAT_COMPLETIONS_ENDPOINT",
                format!("http://{chat_address}/v1/chat/completions"),
            )
            .env("KOGWISTAR_CHAT_COMPLETIONS_MODEL", "fixture-answer-model")
            .env("KOGWISTAR_CHAT_COMPLETIONS_STREAMING", "true")
            .env("KOGWISTAR_EMBEDDING_PROVIDER", "ollama")
            .env("KOGWISTAR_EMBEDDING_MODEL", "fixture-embedding-model")
            .env("KOGWISTAR_EMBEDDING_DIMENSION", "3")
            .env("KOGWISTAR_EMBEDDING_ENDPOINT", format!("http://{embedding_address}"))
            .stdout(Stdio::null())
            .stderr(Stdio::from(server_log));
        let mut server_process = ServerProcess(server_command.spawn().unwrap());
        let base_url = format!("http://127.0.0.1:{server_port}");
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(3))
            .build()
            .unwrap();
        let mut server_ready = false;
        for _ in 0..100 {
            if let Some(status) = server_process.0.try_wait().unwrap() {
                let log = std::fs::read_to_string(&server_log_path).unwrap_or_default();
                panic!("packaged Rust server exited during startup ({status}): {log}");
            }
            if client
                .get(format!("{base_url}/health"))
                .send()
                .await
                .is_ok_and(|response| response.status().is_success())
            {
                server_ready = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
        assert!(server_ready, "packaged Rust server did not become ready");
        let conversation_token = jsonwebtoken::encode(
            &jsonwebtoken::Header::new(jsonwebtoken::Algorithm::HS256),
            &json!({
                "sub":"conversation-owner","user_id":"conversation-owner","actor_id":"conversation-owner","role":"rw","roles":["rw"],
                "ns":"conversation","storage_ns":"conversation",
                "capabilities":["spawn_process","send_message"],
                "exp": now_seconds + 300
            }),
            &jsonwebtoken::EncodingKey::from_secret(http_secret.as_bytes()),
        )
        .unwrap();
        let conversation_create = client
            .post(format!("{base_url}/api/conversations"))
            .bearer_auth(&conversation_token)
            .json(&json!({
                "user_id":"conversation-owner",
                "conversation_id":"rust-only-rest-create-e2e",
                "start_node_id":"rust-only-rest-create-start"
            }))
            .send()
            .await
            .unwrap();
        let create_status = conversation_create.status();
        let create_body = conversation_create.text().await.unwrap();
        assert_eq!(
            create_status.as_u16(),
            StatusCode::OK.as_u16(),
            "{create_body}"
        );
        let created: Value = serde_json::from_str(&create_body).unwrap();
        assert_eq!(created["conversation_id"], "rust-only-rest-create-e2e");
        assert_eq!(created["user_id"], "conversation-owner");
        assert_eq!(created["start_node_id"], "rust-only-rest-create-start");
        assert_eq!(created["tail_node_id"], "rust-only-rest-create-start");
        assert_eq!(created["turn_count"], 0);
        let created_record = service
            .conversation_store
            .graph_projection_records(GraphProjectionRead {
                scope: GraphScope {
                    namespace: "conversation".to_owned(),
                    workspace_id: None,
                    graph_space: None,
                },
                table: "gke_nodes".to_owned(),
                ids: Some(vec!["rust-only-rest-create-start".to_owned()]),
                metadata: MetadataFilter::default(),
                limit: 1,
            })
            .await
            .unwrap();
        assert_eq!(created_record.len(), 1);
        assert_eq!(
            created_record[0].metadata.get("entity_type").and_then(Value::as_str),
            Some("conversation_start")
        );
        assert_eq!(
            created_record[0].metadata.get("user_id").and_then(Value::as_str),
            Some("conversation-owner")
        );
        let empty_answer = client
            .post(format!(
                "{base_url}/api/conversations/rust-only-rest-create-e2e/turns:answer"
            ))
            .bearer_auth(&conversation_token)
            .json(&json!({"text":""}))
            .send()
            .await
            .unwrap();
        let empty_status = empty_answer.status();
        let empty_body = empty_answer.text().await.unwrap();
        assert_eq!(
            empty_status.as_u16(),
            StatusCode::UNPROCESSABLE_ENTITY.as_u16(),
            "{empty_body}"
        );
        let answer_submission = client
            .post(format!(
                "{base_url}/api/conversations/rust-only-rest-create-e2e/turns:answer"
            ))
            .bearer_auth(&conversation_token)
            .json(&json!({"user_id":"body-spoof","text":"queued native question"}))
            .send()
            .await
            .unwrap();
        let answer_status = answer_submission.status();
        let answer_body = answer_submission.text().await.unwrap();
        assert_eq!(
            answer_status.as_u16(),
            StatusCode::ACCEPTED.as_u16(),
            "{answer_body}"
        );
        let answer: Value = serde_json::from_str(&answer_body).unwrap();
        assert_eq!(answer.as_object().map(serde_json::Map::len), Some(5));
        assert_eq!(answer["conversation_id"], "rust-only-rest-create-e2e");
        assert_eq!(answer["workflow_id"], "debug.rag.v1");
        assert_eq!(answer["status"], "queued");
        let answer_run_id = answer["run_id"].as_str().unwrap();
        let answer_turn_id = answer["user_turn_node_id"].as_str().unwrap();
        let answer_turn = service
            .conversation_store
            .graph_projection_records(GraphProjectionRead {
                scope: GraphScope {
                    namespace: "conversation".to_owned(),
                    workspace_id: None,
                    graph_space: None,
                },
                table: "gke_nodes".to_owned(),
                ids: Some(vec![answer_turn_id.to_owned()]),
                metadata: MetadataFilter::default(),
                limit: 1,
            })
            .await
            .unwrap();
        assert_eq!(answer_turn.len(), 1, "admitted user turn must be visible");
        let answer_turn_document: Value = serde_json::from_str(
            answer_turn[0].document.as_deref().expect("turn has a document"),
        )
        .unwrap();
        assert_eq!(answer_turn_document["conversation_id"], "rust-only-rest-create-e2e");
        assert_eq!(answer_turn_document["role"], "user");
        assert_eq!(
            answer_turn[0].metadata.get("entity_type").and_then(Value::as_str),
            Some("conversation_turn")
        );
        let acl_scope = GraphScope {
            namespace: "conversation".to_owned(),
            workspace_id: None,
            graph_space: None,
        };
        let acl_target_id = answer_turn_id.to_owned();
        let conversation_acl_records = service
            .conversation_store
            .transaction(move |uow| {
                Box::pin(async move {
                    uow.graph_acl_records_for_entities(
                        &acl_scope,
                        "conversation",
                        &[acl_target_id],
                    )
                    .await
                })
            })
            .await
            .unwrap();
        let acl_grains = conversation_acl_records
            .iter()
            .filter_map(|record| record.metadata.get("acl_target_grain").and_then(Value::as_str))
            .collect::<BTreeSet<_>>();
        assert!(
            ["node", "grounding", "span"]
                .into_iter()
                .all(|grain| acl_grains.contains(grain)),
            "conversation admission must persist node and source-span ACLs: {conversation_acl_records:#?}"
        );
        assert!(conversation_acl_records.iter().all(|record| {
            record.metadata.get("owner_id").and_then(Value::as_str)
                == Some("conversation-owner")
                && record.metadata.get("acl_mode").and_then(Value::as_str) == Some("private")
        }));
        let mut admitted_run = None;
        for _ in 0..300 {
            admitted_run = service.store.get_server_run(answer_run_id).await.unwrap();
            if admitted_run
                .as_ref()
                .is_some_and(|run| matches!(run.status.as_str(), "succeeded" | "failed"))
            {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        }
        let admitted_run = admitted_run.expect("answer admission creates a durable run");
        let debug_runtime = service
            .store
            .read_recorded_runtime_state(
                answer_run_id,
                "debug.rag.v1",
                "rust-only-rest-create-e2e",
            )
            .await
            .unwrap();
        let answer_lanes = service
            .store
            .list_projected_lane_messages(LaneMessageFilter {
                correlation_id: Some(answer_run_id.to_owned()),
                limit: 10,
                ..LaneMessageFilter::default()
            })
            .await
            .unwrap();
        assert_eq!(
            admitted_run.status,
            "succeeded",
            "REST-admitted debug RAG run must complete through the Rust-native scheduler; run={admitted_run:#?}, runtime={debug_runtime:#?}, lanes={answer_lanes:#?}, turn_document={answer_turn_document:#?}, turn_metadata={:#?}, acl_records={conversation_acl_records:#?}", answer_turn[0].metadata
        );
        let debug_runtime = debug_runtime
            .expect("completed answer run has durable runtime state");
        assert_eq!(
            debug_runtime.state["conversation_id"],
            "rust-only-rest-create-e2e"
        );
        assert_eq!(debug_runtime.state["max_retrieval_level"], 4);
        assert_eq!(debug_runtime.state["done"], true);
        assert_eq!(debug_runtime.state["answer"]["mode"], "debug_rag");
        let assistant_turn_id = debug_runtime.state["agentic_answering_result"]["assistant_turn_node_id"]
            .as_str()
            .expect("completed debug RAG run records its assistant turn ID")
            .to_owned();
        assert_eq!(
            admitted_run.assistant_turn_node_id.as_deref(),
            Some(assistant_turn_id.as_str()),
            "completed run registry should expose the durable assistant turn ID"
        );
        let assistant_turn = service
            .conversation_store
            .graph_projection_records(GraphProjectionRead {
                scope: GraphScope {
                    namespace: "conversation".to_owned(),
                    workspace_id: None,
                    graph_space: None,
                },
                table: "gke_nodes".to_owned(),
                ids: Some(vec![assistant_turn_id]),
                metadata: MetadataFilter::default(),
                limit: 1,
            })
            .await
            .unwrap();
        assert_eq!(assistant_turn.len(), 1, "assistant response must be durable");
        assert_eq!(
            assistant_turn[0].metadata.get("entity_type").and_then(Value::as_str),
            Some("assistant_turn")
        );
        assert_eq!(
            answer_turn[0].metadata.get("entity_type").and_then(Value::as_str),
            Some("conversation_turn")
        );
        assert_eq!(
            answer_turn[0].metadata.get("role").and_then(Value::as_str),
            Some("user")
        );
        assert!(answer_lanes.iter().any(|lane| lane.msg_type == "workflow.run.execute"));
        assert!(answer_lanes.iter().all(|lane| lane.status == "completed"));
        let spoofed_create = client
            .post(format!("{base_url}/api/conversations"))
            .bearer_auth(&conversation_token)
            .json(&json!({
                "user_id":"bob",
                "conversation_id":"rust-only-rest-create-spoof"
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(
            spoofed_create.status().as_u16(),
            StatusCode::FORBIDDEN.as_u16(),
            "user spoofing must fail through the same native syscall authority"
        );
        let submit = client
            .post(format!("{base_url}/api/workflow/runs"))
            .bearer_auth(&token)
            .json(&json!({
                "run_id":"rust-only-answer-e2e","workflow_id":"agentic_answering.v2",
                "conversation_id":"answer-e2e-conversation","turn_node_id":"answer-user-turn","user_id":"alice",
                "initial_state":{"conversation_id":"answer-e2e-conversation","turn_node_id":"answer-user-turn","user_id":"alice",
                    "turn_index":0,"max_candidates":10,"max_used":4,"max_retrieval_level":3,"max_iter":1}
            }))
            .send()
            .await
            .unwrap();
        let submit_status = submit.status();
        let submit_body = submit.text().await.unwrap();
        assert_eq!(
            submit_status.as_u16(),
            StatusCode::ACCEPTED.as_u16(),
            "{}",
            submit_body
        );
        let mut terminal_status = String::new();
        let mut run_response = Value::Null;
        for _ in 0..300 {
            let response = client
                .get(format!("{base_url}/api/workflow/runs/rust-only-answer-e2e"))
                .bearer_auth(&token)
                .send()
                .await
                .unwrap();
            let response_status = response.status();
            let response_body = response.text().await.unwrap();
            assert!(
                response_status.is_success(),
                "run inspection returned {response_status}: {response_body}"
            );
            run_response = serde_json::from_str(&response_body).unwrap();
            terminal_status = run_response["status"].as_str().unwrap_or_default().to_owned();
            if matches!(terminal_status.as_str(), "succeeded" | "failed" | "cancelled") {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        }
        let run = service
            .store
            .get_server_run("rust-only-answer-e2e")
            .await
            .unwrap();
        if terminal_status != "succeeded" {
            let state = service
                .store
                .read_recorded_runtime_state(
                    "rust-only-answer-e2e",
                    "agentic_answering.v2",
                    "answer-e2e-conversation",
                )
                .await
                .unwrap();
            chat_server.abort();
            embedding_server.abort();
            drop(server_process);
            let server_log = std::fs::read_to_string(&server_log_path).unwrap_or_default();
            panic!(
                "native workflow did not complete through packaged server: status={terminal_status:?}, http_run={run_response:#?}, run={run:#?}, state={state:#?}, server_log={server_log}"
            );
        }
        chat_server.await.unwrap();
        embedding_server.abort();
        let batch_requests = batch_embedding_requests.lock().unwrap().clone();
        assert!(!batch_requests.is_empty(), "native retrieval must embed its query");
        assert!(batch_requests.iter().all(|request| {
            request["model"] == "fixture-embedding-model"
                && request["input"].as_array().is_some_and(|inputs| !inputs.is_empty())
        }));
        assert!(
            batch_requests.iter().any(|request| {
                request["input"]
                    .as_array()
                    .is_some_and(|inputs| inputs.iter().any(|input| input == "Where is workflow runtime state stored?"))
            }),
            "the packaged Rust server must embed the agentic workflow's query"
        );
        assert!(
            legacy_embedding_requests.lock().unwrap().is_empty(),
            "the current Rust provider must not silently pass via Ollama's legacy endpoint"
        );
        let runtime = service
            .store
            .read_recorded_runtime_state(
                "rust-only-answer-e2e",
                "agentic_answering.v2",
                "answer-e2e-conversation",
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            runtime.state["answer"]["text"],
            "The workflow runtime state is stored in the graph."
        );
        assert_eq!(
            runtime.state["used_node_ids"],
            json!(["answer-evidence-node"])
        );
        let op_log = runtime.state["op_log"].as_array().unwrap();
        for operation in ops.iter().filter_map(Value::as_str) {
            assert!(
                op_log
                    .iter()
                    .any(|value| value.as_str().is_some_and(|logged| {
                        logged == operation
                            || logged
                                .strip_prefix(operation)
                                .is_some_and(|suffix| suffix.starts_with(':'))
                    })),
                "workflow operation was not traced: {operation}; log={op_log:#?}"
            );
        }
        assert!(
            runtime.state["_rt_node_executor_targets"]
                .as_object()
                .unwrap()
                .values()
                .all(|target| target == "rust-native")
        );
        assert_eq!(
            runtime.state["answer"]["claims"][0]["citations"][0]["source_node_id"],
            "answer-evidence-node"
        );
        assert_eq!(runtime.state["done"], true);
        let assistant_id = runtime.state["agentic_answering_result"]["assistant_turn_node_id"]
            .as_str()
            .expect("persist response state contains its assistant turn ID")
            .to_owned();
        let assistant = service
            .conversation_store
            .graph_projection_records(GraphProjectionRead {
                scope: GraphScope {
                    namespace: "alpha".to_owned(),
                    workspace_id: None,
                    graph_space: None,
                },
                table: "gke_nodes".to_owned(),
                ids: Some(vec![assistant_id.to_owned()]),
                metadata: MetadataFilter::default(),
                limit: 1,
            })
            .await
            .unwrap();
        assert_eq!(
            assistant.len(),
            1,
            "assistant response must be durable in the graph"
        );
        let assistant_contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/assistant-turn-graph-v1.json"
        ))
        .unwrap();
        let mut assistant_document: Value = serde_json::from_str(
            assistant[0].document.as_deref().expect("assistant document persisted"),
        )
        .unwrap();
        assistant_document["metadata"] = Value::Object(
            assistant[0]
                .metadata
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
        );
        for field in assistant_contract["assistant_node"]["required_fields"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(Value::as_str)
        {
            assert!(assistant_document.get(field).is_some(), "missing assistant field {field}");
        }
        for (path, expected) in assistant_contract["assistant_node"]["field_values"]
            .as_object()
            .unwrap()
        {
            let actual = path
                .split('.')
                .fold(Some(&assistant_document), |value, key| {
                    value.and_then(|value| value.get(key))
                });
            assert_eq!(actual, Some(expected), "assistant projection mismatch at {path}");
        }
        assert_eq!(
            assistant[0].metadata.get("role").and_then(Value::as_str),
            Some("assistant")
        );
        let edges = service
            .conversation_store
            .graph_projection_records(GraphProjectionRead {
                scope: GraphScope {
                    namespace: "alpha".to_owned(),
                    workspace_id: None,
                    graph_space: None,
                },
                table: "gke_edges".to_owned(),
                ids: None,
                metadata: MetadataFilter::default(),
                limit: 100,
            })
            .await
            .unwrap();
        assert!(
            edges.iter().any(|edge| edge
                .document
                .as_deref()
                .and_then(|document| serde_json::from_str::<Value>(document).ok())
                .is_some_and(|document| {
                    document["relation"] == "next_turn"
                        && document["source_ids"] == json!(["answer-user-turn"])
                        && document["target_ids"] == json!([assistant_id])
                })),
            "assistant turn must be linked to the admitted user turn"
        );
        let generated = edges.iter().find_map(|edge| {
            edge.document
                .as_deref()
                .and_then(|document| serde_json::from_str::<Value>(document).ok())
                .filter(|document| {
                    document["relation"] == "generated"
                        && document["target_ids"] == json!([assistant_id])
                })
        });
        let mut generated = generated.expect("run-to-assistant generated edge must be durable");
        let generated_record = edges
            .iter()
            .find(|edge| {
                edge.document
                    .as_deref()
                    .and_then(|document| serde_json::from_str::<Value>(document).ok())
                    .is_some_and(|document| {
                        document["relation"] == "generated"
                            && document["target_ids"] == json!([assistant_id])
                    })
            })
            .expect("generated edge record exists");
        generated["metadata"] = Value::Object(
            generated_record
                .metadata
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
        );
        for field in assistant_contract["generated_edge"]["required_fields"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(Value::as_str)
        {
            assert!(generated.get(field).is_some(), "missing generated edge field {field}");
        }
        for (path, expected) in assistant_contract["generated_edge"]["field_values"]
            .as_object()
            .unwrap()
        {
            let actual = path
                .split('.')
                .fold(Some(&generated), |value, key| {
                    value.and_then(|value| value.get(key))
                });
            assert_eq!(actual, Some(expected), "generated edge projection mismatch at {path}");
        }
        let lanes = service
            .store
            .list_projected_lane_messages(LaneMessageFilter {
                correlation_id: Some("rust-only-answer-e2e".to_owned()),
                limit: 100,
                ..LaneMessageFilter::default()
            })
            .await
            .unwrap();
        assert!(!lanes.is_empty());
        assert!(lanes.iter().all(|lane| lane.status == "completed"));
        assert!(
            lanes.iter().all(|lane| lane.recipient_id != "python-worker"),
            "Rust-only workflow must not hand a step to a Python worker"
        );
        let durable_events = service
            .store
            .list_server_run_events("rust-only-answer-e2e", 0, 10_000)
            .await
            .unwrap();
        let public_events_response = client
            .get(format!(
                "{base_url}/api/runs/rust-only-answer-e2e/events/poll?after_seq=0&limit=10000"
            ))
            .bearer_auth(&token)
            .send()
            .await
            .unwrap();
        assert!(
            public_events_response.status().is_success(),
            "authorized run-event polling should expose the durable event stream"
        );
        let public_events: Value = public_events_response.json().await.unwrap();
        let public_provider_event = public_events["events"]
            .as_array()
            .unwrap()
            .iter()
            .find(|event| {
                event["event_type"] == "workflow.provider_event.v1"
                    && event["payload"]["provider_event"]["kind"] == "field_completed"
                    && event["payload"]["provider_event"]["details"]["field_name"] == "text"
            })
            .expect("authorized run-event polling returns streamed semantic event metadata");
        assert_eq!(
            public_provider_event["payload"]["provider_event"]["details"]["field_name"],
            "text"
        );
        assert!(public_provider_event["payload"]["client_sse_sequence"]
            .as_i64()
            .is_some_and(|sequence| sequence >= 0),
            "public provider event omitted client sequence: {public_provider_event:#?}"
        );
        let public_event_payload = serde_json::to_string(public_provider_event).unwrap();
        assert!(!public_event_payload.contains("The workflow runtime state is stored in the graph."));
        assert!(!public_event_payload.contains("The selected evidence states this directly."));
        let mcp_events_response = client
            .post(format!("{base_url}/mcp/workflow"))
            .bearer_auth(&token)
            .json(&json!({
                "jsonrpc":"2.0",
                "id":71,
                "method":"tools/call",
                "params":{
                    "name":"workflow.run_events",
                    "arguments":{
                        "run_id":"rust-only-answer-e2e",
                        "after_seq":0,
                        "limit":10_000
                    }
                }
            }))
            .send()
            .await
            .unwrap();
        assert!(
            mcp_events_response.status().is_success(),
            "authorized workflow.run_events MCP call should expose the durable event stream"
        );
        let mcp_events: Value = mcp_events_response.json().await.unwrap();
        let mcp_provider_event = mcp_events["result"]["events"]
            .as_array()
            .unwrap()
            .iter()
            .find(|event| {
                event["event_type"] == "workflow.provider_event.v1"
                    && event["payload"]["provider_event"]["kind"] == "field_completed"
                    && event["payload"]["provider_event"]["details"]["field_name"] == "text"
            })
            .expect("workflow.run_events returns the streamed provider event");
        let mcp_tools_response = client
            .post(format!("{base_url}/mcp/workflow"))
            .bearer_auth(&token)
            .json(&json!({"jsonrpc":"2.0","id":70,"method":"tools/list"}))
            .send()
            .await
            .unwrap();
        assert!(mcp_tools_response.status().is_success());
        let mcp_tools: Value = mcp_tools_response.json().await.unwrap();
        assert!(
            mcp_tools["result"]["tools"]
                .as_array()
                .unwrap()
                .iter()
                .any(|tool| tool["name"] == "workflow.process_timeline"),
            "PostgreSQL server must advertise process_timeline to the authorized run reader"
        );
        let mcp_timeline_response = client
            .post(format!("{base_url}/mcp/workflow"))
            .bearer_auth(&token)
            .json(&json!({
                "jsonrpc":"2.0",
                "id":72,
                "method":"tools/call",
                "params":{
                    "name":"workflow.process_timeline",
                    "arguments":{"run_id":"rust-only-answer-e2e","after_seq":0}
                }
            }))
            .send()
            .await
            .unwrap();
        assert!(
            mcp_timeline_response.status().is_success(),
            "authorized workflow.process_timeline MCP call should expose durable run events"
        );
        let mcp_timeline: Value = mcp_timeline_response.json().await.unwrap();
        assert_eq!(
            mcp_timeline["result"]["run_id"],
            "rust-only-answer-e2e",
            "unexpected process-timeline MCP response: {mcp_timeline:#?}"
        );
        let timeline_provider_event = mcp_timeline["result"]["events"]
            .as_array()
            .unwrap()
            .iter()
            .find(|event| {
                event["event_type"] == "workflow.provider_event.v1"
                    && event["payload"]["provider_event"]["kind"] == "field_completed"
                    && event["payload"]["provider_event"]["details"]["field_name"] == "text"
            })
            .expect("workflow.process_timeline returns the persisted semantic provider event");
        assert_eq!(
            timeline_provider_event["payload"]["span_id"],
            public_provider_event["payload"]["span_id"],
            "process timeline and REST expose the same trace identity"
        );
        assert_eq!(
            mcp_provider_event["payload"]["span_id"],
            public_provider_event["payload"]["span_id"],
            "REST and MCP expose the same trace identity"
        );
        let mcp_event_payload = serde_json::to_string(mcp_provider_event).unwrap();
        assert!(!mcp_event_payload.contains("The workflow runtime state is stored in the graph."));
        assert!(!mcp_event_payload.contains("The selected evidence states this directly."));
        let process_table_token = jsonwebtoken::encode(
            &jsonwebtoken::Header::new(jsonwebtoken::Algorithm::HS256),
            &json!({
                "sub":"alice","role":"rw","roles":["rw"],"ns":["workflow"],
                "storage_ns":"alpha","execution_ns":"alpha",
                "capabilities":["workflow.run.read"],"exp":now_seconds + 300
            }),
            &jsonwebtoken::EncodingKey::from_secret(http_secret.as_bytes()),
        )
        .unwrap();
        let process_table_tools = client
            .post(format!("{base_url}/mcp/workflow"))
            .bearer_auth(&process_table_token)
            .json(&json!({"jsonrpc":"2.0","id":73,"method":"tools/list"}))
            .send()
            .await
            .unwrap();
        assert!(process_table_tools.status().is_success());
        let process_table_tools: Value = process_table_tools.json().await.unwrap();
        assert!(process_table_tools["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .any(|tool| tool["name"] == "workflow.process_table"));
        assert!(process_table_tools["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .any(|tool| tool["name"] == "workflow.blocked_runs"));
        let process_table_response = client
            .post(format!("{base_url}/mcp/workflow"))
            .bearer_auth(&process_table_token)
            .json(&json!({
                "jsonrpc":"2.0",
                "id":74,
                "method":"tools/call",
                "params":{
                    "name":"workflow.process_table",
                    "arguments":{
                        "status":"succeeded",
                        "workflow_id":"agentic_answering.v2",
                        "conversation_id":"answer-e2e-conversation",
                        "limit":10
                    }
                }
            }))
            .send()
            .await
            .unwrap();
        assert!(
            process_table_response.status().is_success(),
            "authorized process-table MCP call should succeed"
        );
        let process_table: Value = process_table_response.json().await.unwrap();
        let processes = process_table["result"]["processes"]
            .as_array()
            .unwrap_or_else(|| panic!("unexpected process-table MCP response: {process_table:#?}"));
        let process = processes
            .iter()
            .find(|process| process["process_id"] == "rust-only-answer-e2e")
            .expect("process table should include the completed Rust-native run");
        assert_eq!(process["process_kind"], "workflow_run");
        assert_eq!(process["status"], "succeeded");
        assert_eq!(process["workflow_id"], "agentic_answering.v2");
        let blocked_runs_response = client
            .post(format!("{base_url}/mcp/workflow"))
            .bearer_auth(&process_table_token)
            .json(&json!({
                "jsonrpc":"2.0",
                "id":75,
                "method":"tools/call",
                "params":{
                    "name":"workflow.blocked_runs",
                    "arguments":{"limit":10}
                }
            }))
            .send()
            .await
            .unwrap();
        assert!(
            blocked_runs_response.status().is_success(),
            "authorized blocked-runs MCP call should succeed"
        );
        let blocked_runs: Value = blocked_runs_response.json().await.unwrap();
        let blocked_processes = blocked_runs["result"]["processes"]
            .as_array()
            .unwrap_or_else(|| panic!("unexpected blocked-runs response: {blocked_runs:#?}"));
        assert!(
            blocked_processes.is_empty(),
            "completed PostgreSQL runs must not appear as blocked"
        );
        let streamed_provider_events = durable_events
            .iter()
            .filter(|event| event.event_type == "workflow.provider_event.v1")
            .filter_map(|event| serde_json::from_str::<Value>(&event.payload_json).ok())
            .collect::<Vec<_>>();
        assert!(
            !streamed_provider_events.is_empty(),
            "semantic provider events should have their own durable run-log entries"
        );
        let streamed_text_field = streamed_provider_events
            .iter()
            .find(|event| {
                event["provider_event"]["kind"] == "field_completed"
                    && event["provider_event"]["details"]["field_name"] == "text"
            })
            .expect("the completed text field event is durable");
        assert_eq!(
            public_provider_event["payload"]["span_id"],
            streamed_text_field["span_id"],
            "public run-event polling preserves canonical trace identity"
        );
        assert_eq!(streamed_text_field["run_id"], "rust-only-answer-e2e");
        assert_eq!(streamed_text_field["trace_id"], "rust-only-answer-e2e");
        assert_eq!(streamed_text_field["workflow_id"], "agentic_answering.v2");
        assert_eq!(
            streamed_text_field["conversation_id"],
            "answer-e2e-conversation"
        );
        assert_eq!(streamed_text_field["turn_node_id"], "answer-user-turn");
        assert_eq!(
            streamed_text_field["provider_id"],
            "openai_compatible_chat_completions"
        );
        assert_eq!(streamed_text_field["executor_target"], "rust-native");
        let event_token_id = streamed_text_field["token_id"].as_str().unwrap();
        let event_step_id = streamed_text_field["step_id"].as_str().unwrap();
        let event_step_seq = streamed_text_field["step_seq"].as_i64().unwrap();
        let event_attempt = streamed_text_field["attempt"].as_u64().unwrap();
        assert_eq!(
            streamed_text_field["span_id"],
            format!(
                "stp:rust-only-answer-e2e:{event_token_id}:{event_step_seq}:{event_attempt}:{event_step_id}"
            )
        );
        assert_eq!(
            streamed_text_field["parent_span_id"],
            format!("tok:rust-only-answer-e2e:{event_token_id}")
        );
        let mut receive_request = kogwistar_runtime::RuntimeNativeStepRequest {
            executor_target: kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET.to_owned(),
            operation: "agent.event.receive".to_owned(),
            op_version: "v1".to_owned(),
            run_id: "rust-only-answer-e2e".to_owned(),
            workflow_id: "agentic_answering.v2".to_owned(),
            node_id: "receive-provider-event".to_owned(),
            conversation_id: "answer-e2e-conversation".to_owned(),
            turn_node_id: Some("answer-user-turn".to_owned()),
            token_id: "event-consumer-token".to_owned(),
            parent_token_id: None,
            step_seq: 99,
            state: Map::new(),
            runtime_routes: Vec::new(),
            ephemeral_context: Map::new(),
        };
        let receive_principal = service
            .native_read_principal(&receive_request, None)
            .await
            .unwrap_or_else(|_| panic!("the admitted run owner retains read_graph authority"));
        let claimed_context = service
            .claim_native_provider_event_context(&receive_request, &receive_principal)
            .await
            .unwrap()
            .expect("a visible semantic event should be claimable");
        assert_eq!(claimed_context["event"]["trace"]["run_id"], "rust-only-answer-e2e");
        assert_eq!(
            claimed_context["event"]["trace"]["attempt"],
            event_attempt,
            "provider event attempt matches its producing workflow attempt"
        );
        assert_eq!(
            claimed_context["event"]["trace"]["producer_attempt"],
            event_attempt,
            "producer attempt metadata remains aligned with the event attempt"
        );
        assert_eq!(
            claimed_context["claim"]["message_id"],
            format!("msg:{}", claimed_context["event"]["event_id"].as_str().unwrap())
        );
        receive_request.ephemeral_context.insert(
            "authorized_provider_event".to_owned(),
            claimed_context["event"].clone(),
        );
        receive_request.ephemeral_context.insert(
            "authorized_provider_event_claim".to_owned(),
            claimed_context["claim"].clone(),
        );
        let receive_effect = kogwistar_runtime::execute_native_step_with_registry(
            &receive_request,
            &kogwistar_runtime::NativeOperationRegistry::default(),
        )
        .unwrap();
        let received_state = kogwistar_contracts::apply_state_update_pairs(
            Value::Object(Map::new()),
            receive_effect.state_update.iter().map(|update| match update {
                kogwistar_runtime::RecordedStateUpdate::Pair((mode, payload)) => {
                    (mode.as_str(), payload)
                }
                kogwistar_runtime::RecordedStateUpdate::Object { mode, payload } => {
                    (mode.as_str(), payload)
                }
            }),
        )
        .unwrap();
        assert_eq!(received_state["agent_event_current"], claimed_context["event"]);
        assert_eq!(received_state["agent_event_claim"], claimed_context["claim"]);
        assert!(receive_effect.provider_event_acknowledgements.is_empty());
        let claimed_lane = service
            .store
            .get_projected_lane_message(
                claimed_context["claim"]["message_id"].as_str().unwrap(),
            )
            .await
            .unwrap()
            .expect("the provider event remains in its durable lane");
        assert_eq!(claimed_lane.status, "claimed");
        assert_eq!(
            claimed_lane.claimed_by.as_deref(),
            claimed_context["claim"]["claimed_by"].as_str()
        );
        let durable_run = run.as_ref().expect("completed run remains durable");
        assert_eq!(durable_run.user_id.as_deref(), Some("alice"));
        let admission_event = durable_events
            .iter()
            .find(|event| event.event_type == "run.created")
            .expect("run admission owner binding remains durable");
        let admission_payload: Value =
            serde_json::from_str(&admission_event.payload_json).unwrap();
        assert_eq!(admission_payload["run_id"], durable_run.run_id);
        assert_eq!(admission_payload["runtime_owner_binding"]["version"], 1);
        assert_eq!(admission_payload["runtime_owner_binding"]["user_id"], "alice");
        let native_traces = durable_events
            .iter()
            .filter(|event| event.event_type == "workflow.trace.v1")
            .filter_map(|event| serde_json::from_str::<Value>(&event.payload_json).ok())
            .filter(|event| event["trace"]["executor_target"] == "rust-native")
            .collect::<Vec<_>>();
        assert!(!native_traces.is_empty(), "native execution traces must be durable");
        let provider_event_traces = native_traces
            .iter()
            .filter(|event| event["trace"]["event"] == "provider.semantic_event.observed")
            .collect::<Vec<_>>();
        assert!(
            provider_event_traces.iter().any(|event| {
                event["trace"]["provider_event"]["kind"] == "field_completed"
                    && event["trace"]["provider_event"]["details"]["field_name"] == "text"
            }),
            "the streamed complete text field should be durably summarized: {provider_event_traces:#?}"
        );
        let provider_event_json = serde_json::to_string(&provider_event_traces).unwrap();
        assert!(!provider_event_json.contains("The workflow runtime state is stored in the graph."));
        assert!(!provider_event_json.contains("The selected evidence states this directly."));
        assert!(!provider_event_json.contains("\"value\""));
        for event in native_traces {
            let trace = &event["trace"];
            let token_id = trace["token_id"].as_str().unwrap();
            let step_seq = trace["step_seq"].as_i64().unwrap();
            let node_id = trace["node_id"].as_str().unwrap();
            let attempt = trace["attempt"].as_u64().unwrap();
            assert_eq!(event["run_id"], "rust-only-answer-e2e");
            assert_eq!(event["run_id"], durable_run.run_id);
            assert_eq!(trace["run_id"], event["run_id"]);
            assert_eq!(trace["trace_id"], event["run_id"]);
            assert_eq!(event["step_id"], node_id);
            assert_eq!(event["token_id"], token_id);
            assert_eq!(
                trace["span_id"],
                format!("stp:rust-only-answer-e2e:{token_id}:{step_seq}:{attempt}:{node_id}")
            );
            assert_eq!(
                trace["parent_span_id"],
                format!("tok:rust-only-answer-e2e:{token_id}")
            );
        }

        let design_id = "packaged-rust-native-design-e2e";
        for node in [
            json!({
                "designer_id":"alice","node_id":"start","label":"Increment",
                "op":"increment-counter","start":true,
                "metadata":{"wf_executor_target":"rust-native"}
            }),
            json!({
                "designer_id":"alice","node_id":"end","label":"Finish",
                "op":"emit-result","terminal":true,
                "metadata":{"wf_executor_target":"rust-native"}
            }),
        ] {
            let response = client
                .post(format!("{base_url}/api/workflow/design/{design_id}/nodes"))
                .bearer_auth(&token)
                .json(&node)
                .send()
                .await
                .unwrap();
            let status = response.status();
            let body = response.text().await.unwrap();
            assert_eq!(status.as_u16(), StatusCode::OK.as_u16(), "{body}");
        }
        let edge_response = client
            .post(format!("{base_url}/api/workflow/design/{design_id}/edges"))
            .bearer_auth(&token)
            .json(&json!({
                "designer_id":"alice","edge_id":"start-to-end",
                "src":"start","dst":"end","is_default":true
            }))
            .send()
            .await
            .unwrap();
        let edge_status = edge_response.status();
        let edge_body = edge_response.text().await.unwrap();
        assert_eq!(edge_status.as_u16(), StatusCode::OK.as_u16(), "{edge_body}");

        let design_graph_response = client
            .get(format!("{base_url}/api/workflow/design/{design_id}/graph"))
            .bearer_auth(&token)
            .send()
            .await
            .unwrap();
        let design_graph_status = design_graph_response.status();
        let design_graph_body = design_graph_response.text().await.unwrap();
        assert_eq!(
            design_graph_status.as_u16(),
            StatusCode::OK.as_u16(),
            "{design_graph_body}"
        );
        let design_graph: Value = serde_json::from_str(&design_graph_body).unwrap();
        assert_eq!(design_graph["nodes"].as_array().unwrap().len(), 2);
        assert_eq!(design_graph["edges"].as_array().unwrap().len(), 1);
        let start = design_graph["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|node| node["id"] == "start")
            .unwrap();
        assert_eq!(start["metadata"]["wf_op"], "increment-counter");
        assert_eq!(start["metadata"]["wf_executor_target"], "rust-native");

        let design_run_id = "packaged-rust-native-design-run-e2e";
        let design_submit = client
            .post(format!("{base_url}/api/workflow/runs"))
            .bearer_auth(&token)
            .json(&json!({
                "run_id":design_run_id,"workflow_id":design_id,
                "conversation_id":"packaged-rust-native-design-conversation",
                "user_id":"alice","initial_state":{"counter":1}
            }))
            .send()
            .await
            .unwrap();
        let design_submit_status = design_submit.status();
        let design_submit_body = design_submit.text().await.unwrap();
        assert_eq!(
            design_submit_status.as_u16(),
            StatusCode::ACCEPTED.as_u16(),
            "{design_submit_body}"
        );
        let mut design_run_status = String::new();
        for _ in 0..300 {
            let response = client
                .get(format!("{base_url}/api/workflow/runs/{design_run_id}"))
                .bearer_auth(&token)
                .send()
                .await
                .unwrap();
            let status = response.status();
            let body = response.text().await.unwrap();
            assert!(status.is_success(), "{status}: {body}");
            let run: Value = serde_json::from_str(&body).unwrap();
            design_run_status = run["status"].as_str().unwrap_or_default().to_owned();
            if matches!(design_run_status.as_str(), "succeeded" | "failed" | "cancelled") {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
        assert_eq!(design_run_status, "succeeded");
        let design_runtime = service
            .store
            .read_recorded_runtime_state(
                design_run_id,
                design_id,
                "packaged-rust-native-design-conversation",
            )
            .await
            .unwrap()
            .expect("packaged Rust design run persists its final runtime state");
        assert_eq!(design_runtime.state["counter"], 2);
        let design_events = service
            .store
            .list_server_run_events(design_run_id, 0, 100)
            .await
            .unwrap();
        assert!(design_events.iter().any(|event| {
            event.event_type == "workflow.trace.v1"
                && event.payload_json.contains("\"seam_crossings\":[]")
        }));
        drop(server_process);
        drop(service);
        let (cleanup, connection) = tokio_postgres::connect(&dsn, tokio_postgres::NoTls)
            .await
            .unwrap();
        let task = tokio::spawn(connection);
        cleanup
            .batch_execute(&format!(
                "DROP SCHEMA IF EXISTS \"{knowledge_schema}\" CASCADE; \
                 DROP SCHEMA IF EXISTS \"{conversation_schema}\" CASCADE; \
                 DROP SCHEMA IF EXISTS \"{workflow_schema}\" CASCADE; \
                 DROP SCHEMA IF EXISTS \"{wisdom_schema}\" CASCADE"
            ))
            .await
            .unwrap();
        drop(cleanup);
        task.await.unwrap().unwrap();
        let _ = std::fs::remove_file(auth_path);
        let _ = std::fs::remove_file(server_log_path);
    }

    #[test]
    fn sqlite_native_answer_view_projects_only_the_admitted_conversation() {
        let record = |id: &str, document: Value, metadata: serde_json::Map<String, Value>| {
            GraphRecord {
                id: id.to_owned(),
                document: Some(serde_json::to_string(&document).unwrap()),
                metadata,
                embedding: None,
            }
        };
        let nodes = vec![
            record("turn-c1", json!({
                "entity_type": "conversation_turn",
                "conversation_id": "c1",
                "role": "user",
                "content": "question",
            }), serde_json::Map::new()),
            record("context-c1", json!({
                "entity_type": "conversation_turn",
                "conversation_id": "c1",
                "role": "assistant",
                "content": "answer",
            }), serde_json::Map::new()),
            record("turn-c2", json!({
                "entity_type": "conversation_turn",
                "conversation_id": "c2",
                "role": "user",
                "content": "other question",
            }), serde_json::Map::new()),
        ];
        let edges = vec![
            record("edge-c1", json!({"source": "turn-c1", "target": "context-c1"}),
                serde_json::Map::from_iter([("doc_id".to_owned(), json!("conv:c1"))])),
            record("edge-c2", json!({"source": "turn-c2", "target": "context-c2"}),
                serde_json::Map::from_iter([("doc_id".to_owned(), json!("conv:c2"))])),
        ];

        let view = match sqlite_native_answer_view_from_visible_records(
            &nodes,
            &edges,
            "c1",
            "turn-c1",
        ) {
            Ok(view) => view,
            Err(_) => panic!("valid visible conversation context should be accepted"),
        };
        assert_eq!(view["turn"]["id"], "turn-c1");
        assert_eq!(view["context_nodes"].as_array().unwrap().len(), 2);
        assert_eq!(view["context_edges"].as_array().unwrap().len(), 1);

        let inactive_turn = record("turn-c1", json!({
            "entity_type": "conversation_turn",
            "conversation_id": "c1",
            "role": "user",
            "content": "question",
            "lifecycle_status": "tombstoned",
        }), serde_json::Map::new());
        assert!(sqlite_native_answer_view_from_visible_records(
            &[inactive_turn],
            &[],
            "c1",
            "turn-c1",
        )
        .is_err());
    }

    #[test]
    fn sqlite_native_candidates_exclude_acl_private_and_tombstoned_records() {
        let node = |id: &str, lifecycle: Option<&str>| {
            let mut metadata = serde_json::Map::new();
            metadata.insert("label".to_owned(), json!(id));
            metadata.insert("summary".to_owned(), json!(format!("summary {id}")));
            if let Some(status) = lifecycle {
                metadata.insert("lifecycle_status".to_owned(), json!(status));
            }
            GraphRecord {
                id: id.to_owned(),
                document: Some(format!("document {id}")),
                metadata,
                embedding: None,
            }
        };
        let acl = |id: &str, target: &str, mode: &str| {
            let mut metadata = serde_json::Map::new();
            metadata.insert("entity_type".to_owned(), json!("acl_record"));
            metadata.insert("acl_truth_graph".to_owned(), json!("knowledge"));
            metadata.insert("acl_target_grain".to_owned(), json!("node"));
            metadata.insert("acl_target_entity_id".to_owned(), json!(target));
            metadata.insert("acl_version".to_owned(), json!(1));
            metadata.insert("acl_mode".to_owned(), json!(mode));
            if mode == "private" {
                metadata.insert("acl_owner_id".to_owned(), json!("other-user"));
            }
            GraphRecord {
                id: id.to_owned(),
                document: None,
                metadata,
                embedding: None,
            }
        };
        let candidates = vec![
            node("visible", None),
            node("private", None),
            node("deleted", Some("tombstoned")),
        ];
        let acl_records = vec![
            acl("acl-visible", "visible", "public"),
            acl("acl-private", "private", "private"),
            acl("acl-deleted", "deleted", "public"),
        ];
        let principal = GraphAclPrincipal {
            principal_id: "reader".to_owned(),
            ..Default::default()
        };
        let projected = sqlite_native_candidate_values(
            &candidates,
            &acl_records,
            "knowledge",
            &principal,
        );
        assert_eq!(projected.as_array().unwrap().len(), 1);
        assert_eq!(projected[0]["id"], "visible");
        assert_eq!(projected[0]["doc"], "document visible");
    }
