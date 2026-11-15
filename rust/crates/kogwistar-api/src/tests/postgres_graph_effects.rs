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
        let owner = token("alice", "alpha", &["spawn_process", "workflow.run.write", "write_graph"]);
        let owner_read = token("alice", "alpha", &["spawn_process", "workflow.run.write", "write_graph", "read_graph"]);
        let missing_spawn_process = token("alice", "alpha", &["workflow.run.write"]);
        let missing_run_write = token("alice", "alpha", &["spawn_process"]);
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
        let submit_auth_contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/workflow-run-submit-auth-v1.json"
        ))
        .unwrap();
        for (token, run_id) in [
            (&missing_spawn_process, "pg-missing-spawn"),
            (&missing_run_write, "pg-missing-write"),
        ] {
            let denied = client
                .post(format!("{base}/api/workflow/runs"))
                .bearer_auth(token)
                .json(&json!({
                    "run_id": run_id,
                    "workflow_id": "wf",
                    "conversation_id": "conv"
                }))
                .send()
                .await
                .unwrap();
            assert_eq!(
                denied.status().as_u16(),
                submit_auth_contract["denied_status"].as_u64().unwrap() as u16
            );
            assert!(service.store.get_server_run(run_id).await.unwrap().is_none());
        }
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
