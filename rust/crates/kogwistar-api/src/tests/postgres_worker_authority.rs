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
        service
            .store
            .create_server_run(ServerRunCreate {
                run_id: "other-owned-run".to_owned(),
                conversation_id: "other-conversation".to_owned(),
                workflow_id: "other-workflow".to_owned(),
                user_id: Some("run-owner".to_owned()),
                user_turn_node_id: "other-turn".to_owned(),
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

        let mut worker_result = json!({
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

        let run_events_before_invalid_handoff = service
            .store
            .list_server_run_events("owned-run", 0, usize::MAX)
            .await
            .unwrap();
        for (body, expected_status) in [
            (b"{".to_vec(), StatusCode::UNPROCESSABLE_ENTITY),
            (
                vec![b' '; MAX_RUNTIME_WORKER_RESULT_BYTES + 1],
                StatusCode::PAYLOAD_TOO_LARGE,
            ),
        ] {
            let rejected = service
                .execute(ApiEffectRequest {
                    contract_version: 1,
                    method: "POST".to_owned(),
                    path_and_query: "/internal/runtime/results".to_owned(),
                    body,
                    principal: json!({}),
                })
                .await;
            assert_eq!(rejected.status, expected_status.as_u16());
            assert_eq!(
                service
                    .store
                    .list_server_run_events("owned-run", 0, usize::MAX)
                    .await
                    .unwrap(),
                run_events_before_invalid_handoff,
                "malformed or oversized result must not append runtime events"
            );
            assert!(service
                .store
                .replay_raw_events("alpha", 0, 100)
                .await
                .unwrap()
                .iter()
                .all(|event| event.event_id != "owner-check-event"));
            assert_eq!(
                service
                    .store
                    .get_projected_lane_message("lane-1")
                    .await
                    .unwrap()
                    .unwrap()
                    .status,
                "claimed",
                "malformed or oversized result must preserve the active claim"
            );
        }
        for field in ["message_id", "claimed_by", "run_id", "step_id", "correlation_id"] {
            let mut invalid_handoff = worker_result.clone();
            invalid_handoff["handoff"][field] = if field == "run_id" {
                json!("other-owned-run")
            } else {
                json!(format!("forged-{field}"))
            };
            let rejected = client
                .post(format!("{base}/internal/runtime/results"))
                .bearer_auth(&owner_token)
                .json(&invalid_handoff)
                .send()
                .await
                .unwrap();
            assert_eq!(
                rejected.status(),
                StatusCode::CONFLICT,
                "mismatched handoff field {field} must not be accepted"
            );
            assert_eq!(
                service
                    .store
                    .list_server_run_events("owned-run", 0, usize::MAX)
                    .await
                    .unwrap(),
                run_events_before_invalid_handoff,
                "mismatched handoff field {field} must not append run events"
            );
            assert!(service
                .store
                .replay_raw_events("alpha", 0, 100)
                .await
                .unwrap()
                .iter()
                .all(|event| event.event_id != "owner-check-event"));
            assert_eq!(
                service
                    .store
                    .get_projected_lane_message("lane-1")
                    .await
                    .unwrap()
                    .unwrap()
                    .status,
                "claimed",
                "mismatched handoff field {field} must preserve the active claim"
            );
        }

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

        let (lease_expiry, connection) =
            tokio_postgres::connect(&dsn, tokio_postgres::NoTls).await.unwrap();
        let lease_expiry_task = tokio::spawn(connection);
        let expired = lease_expiry
            .execute(
                &format!(
                    "UPDATE \"{schema}\".projected_lane_messages SET lease_until = NOW() - INTERVAL '1 second' WHERE message_id = $1 AND status = 'claimed' AND claimed_by = $2"
                ),
                &[&"lane-1", &"worker-1"],
            )
            .await
            .unwrap();
        assert_eq!(expired, 1, "test must expire the original worker lease");
        drop(lease_expiry);
        lease_expiry_task.await.unwrap().unwrap();
        let reclaimed = service
            .store
            .claim_projected_lane_messages("workflow", "workflow-runtime", "worker-2", 1, 30)
            .await
            .unwrap();
        assert_eq!(reclaimed.len(), 1);
        assert_eq!(reclaimed[0].message_id, "lane-1");
        assert_ne!(reclaimed[0].claimed_by.as_deref(), Some("worker-1"));
        let run_events_before_stale_result = service
            .store
            .list_server_run_events("owned-run", 0, usize::MAX)
            .await
            .unwrap();
        let stale = client
            .post(format!("{base}/internal/runtime/results"))
            .bearer_auth(&owner_token)
            .json(&worker_result)
            .send()
            .await
            .unwrap();
        assert_eq!(stale.status(), StatusCode::CONFLICT);
        assert_eq!(
            service
                .store
                .list_server_run_events("owned-run", 0, usize::MAX)
                .await
                .unwrap(),
            run_events_before_stale_result,
            "stale worker result must not append runtime events"
        );
        assert!(service
            .store
            .replay_raw_events("alpha", 0, 100)
            .await
            .unwrap()
            .iter()
            .all(|event| event.event_id != "owner-check-event"));
        assert_eq!(
            service
                .store
                .get_projected_lane_message("lane-1")
                .await
                .unwrap()
                .unwrap()
                .claimed_by
                .as_deref(),
            reclaimed[0].claimed_by.as_deref(),
            "stale worker must not disturb the newer claim"
        );
        worker_result["handoff"]["claimed_by"] =
            json!(reclaimed[0].claimed_by.as_deref().unwrap());

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
