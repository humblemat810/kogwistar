// Domain-focused tests; shared fixtures live in the parent test module.
use super::*;

#[tokio::test]
#[ignore = "requires KOGWISTAR_TEST_PG_DSN; run explicitly with --include-ignored"]
async fn graph_run_event_and_worker_lane_share_one_transaction() -> PostgresStoreResult<()> {
    let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
        .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
    let schema = test_schema("graph_run_lane_uow");
    let store = PostgresStore::from_dsn(&dsn, &schema)?;
    store.ensure_schema().await?;
    store
        .create_graph_schema(3, GraphTableNames::default())
        .await?;

    let namespace = format!("graph-run-lane-uow-{}", std::process::id());
    let bundle = |suffix: &str| {
        let run_id = format!("run-{suffix}");
        let turn_id = format!("turn-{suffix}");
        let message_id = format!("lane-{suffix}");
        let graph_mutation = GraphMutation {
            scope: GraphScope {
                namespace: namespace.clone(),
                workspace_id: None,
                graph_space: None,
            },
            table: "gke_nodes".to_owned(),
            entity_kind: "node".to_owned(),
            event_id: format!("event-{suffix}"),
            op: "ADD".to_owned(),
            payload: serde_json::json!({"id": turn_id, "metadata": {"entity_type": "conversation_turn"}}),
            record: GraphRecord {
                id: turn_id.clone(),
                document: Some("user turn".to_owned()),
                metadata: Map::from_iter([(
                    "entity_type".to_owned(),
                    Value::String("conversation_turn".to_owned()),
                )]),
                embedding: Some(vec![0.1, 0.2, 0.3]),
            },
            embedding_dim: 3,
        };
        let run = ServerRunCreate {
            run_id: run_id.clone(),
            conversation_id: "conversation-uow".to_owned(),
            workflow_id: "workflow-uow".to_owned(),
            user_id: Some("owner-uow".to_owned()),
            user_turn_node_id: turn_id.clone(),
            status: "queued".to_owned(),
        };
        let lane = NewProjectedLaneMessage {
            message_id,
            namespace: "workflow".to_owned(),
            purpose: "system".to_owned(),
            inbox_id: "workflow-runtime".to_owned(),
            conversation_id: "conversation-uow".to_owned(),
            recipient_id: "python-worker".to_owned(),
            sender_id: "rust-scheduler".to_owned(),
            msg_type: "workflow.run.execute".to_owned(),
            status: "pending".to_owned(),
            created_at: 1,
            available_at: 1,
            run_id: Some(run_id.clone()),
            step_id: Some("start".to_owned()),
            correlation_id: Some(run_id.clone()),
            payload_json: Some(serde_json::json!({"run_id": run_id}).to_string()),
            error_json: None,
        };
        (graph_mutation, run, lane)
    };

    let (graph_mutation, run, lane) = bundle("rollback");
    let rollback: PostgresStoreResult<()> = store
        .transaction(|uow| {
            Box::pin(async move {
                uow.apply_graph_mutation(graph_mutation).await?;
                uow.create_server_run(run).await?;
                uow.append_server_run_event(
                    "run-rollback",
                    "run.created",
                    "{\"status\":\"queued\"}".to_owned(),
                )
                .await?;
                uow.project_lane_message(lane).await?;
                Err(PostgresStoreError::TransactionAborted(
                    "injected rollback".to_owned(),
                ))
            })
        })
        .await;
    assert!(matches!(
        rollback,
        Err(PostgresStoreError::TransactionAborted(_))
    ));
    assert_eq!(
        EventReadStore::latest_event_seq(&store, &namespace).await?,
        0
    );
    assert!(store.get_server_run("run-rollback").await?.is_none());
    assert!(
        store
            .list_server_run_events("run-rollback", 0, 10)
            .await?
            .is_empty()
    );
    assert!(
        store
            .get_projected_lane_message("lane-rollback")
            .await?
            .is_none()
    );
    assert!(
        store
            .graph_projection_records(GraphProjectionRead {
                scope: GraphScope {
                    namespace: namespace.clone(),
                    workspace_id: None,
                    graph_space: None,
                },
                table: "gke_nodes".to_owned(),
                ids: Some(vec!["turn-rollback".to_owned()]),
                metadata: kogwistar_store::MetadataFilter::default(),
                limit: 1,
            })
            .await?
            .is_empty()
    );

    let (graph_mutation, run, lane) = bundle("commit");
    store
        .transaction(|uow| {
            Box::pin(async move {
                uow.apply_graph_mutation(graph_mutation).await?;
                uow.create_server_run(run).await?;
                uow.append_server_run_event(
                    "run-commit",
                    "run.created",
                    "{\"status\":\"queued\"}".to_owned(),
                )
                .await?;
                uow.project_lane_message(lane).await?;
                Ok(())
            })
        })
        .await?;
    assert_eq!(
        EventReadStore::latest_event_seq(&store, &namespace).await?,
        1
    );
    assert_eq!(
        store.get_server_run("run-commit").await?.unwrap().status,
        "queued"
    );
    assert_eq!(
        store
            .list_server_run_events("run-commit", 0, 10)
            .await?
            .len(),
        1
    );
    assert_eq!(
        store
            .get_projected_lane_message("lane-commit")
            .await?
            .unwrap()
            .status,
        "pending"
    );
    assert_eq!(
        store
            .graph_projection_records(GraphProjectionRead {
                scope: GraphScope {
                    namespace,
                    workspace_id: None,
                    graph_space: None,
                },
                table: "gke_nodes".to_owned(),
                ids: Some(vec!["turn-commit".to_owned()]),
                metadata: kogwistar_store::MetadataFilter::default(),
                limit: 1,
            })
            .await?
            .len(),
        1
    );
    Ok(())
}

#[tokio::test]
#[ignore = "requires KOGWISTAR_TEST_PG_DSN; run explicitly with --include-ignored"]
async fn rust_projection_worker_uses_postgres_native_writes_and_tombstone_gate()
-> PostgresStoreResult<()> {
    let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
        .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
    let schema = test_schema("native_projection_worker");
    let store = PostgresStore::from_dsn(&dsn, &schema)?;
    store.ensure_schema().await?;
    store
        .create_graph_schema(3, GraphTableNames::default())
        .await?;
    let namespace = format!("native-projection-worker-{}", std::process::id());
    let scope = GraphScope {
        namespace: namespace.clone(),
        workspace_id: None,
        graph_space: None,
    };
    let node_id = "native-worker-node";
    let graph_mutation = |event_id: &str, op: &str| {
        let mut metadata = Map::new();
        if op == "TOMBSTONE" {
            metadata.insert(
                "lifecycle_status".to_owned(),
                Value::String("tombstoned".to_owned()),
            );
        }
        GraphMutation {
            scope: scope.clone(),
            table: "gke_nodes".to_owned(),
            entity_kind: "node".to_owned(),
            event_id: event_id.to_owned(),
            op: op.to_owned(),
            payload: serde_json::json!({"id": node_id}),
            record: GraphRecord {
                id: node_id.to_owned(),
                document: Some("native projection worker text".to_owned()),
                metadata,
                embedding: None,
            },
            embedding_dim: 3,
        }
    };
    store
        .apply_graph_mutation(graph_mutation("native-worker-add", "ADD"))
        .await?;
    store
        .enqueue_index_job(NewIndexJob {
            job_id: "native-worker-upsert".to_owned(),
            namespace: namespace.clone(),
            entity_kind: "node".to_owned(),
            entity_id: node_id.to_owned(),
            index_kind: "node_embedding".to_owned(),
            op: "UPSERT".to_owned(),
            payload_json: None,
            max_retries: 3,
        })
        .await?;

    let provider_calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let calls = provider_calls.clone();
    let summary =
        kogwistar_engine::projection_worker::run_graph_projection_jobs_once_with_provider_input(
            &store,
            &store,
            1,
            30,
            Some(&namespace),
            move |input| {
                let calls = calls.clone();
                async move {
                    calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    Ok(GraphRecord {
                        id: input.entity_id,
                        document: Some("native projection worker text".to_owned()),
                        metadata: Map::new(),
                        embedding: Some(vec![0.0, 1.0, 0.0]),
                    })
                }
            },
        )
        .await?;
    assert_eq!(summary.projected, 1);
    assert_eq!(provider_calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(
        store
            .index_jobs(Some(&namespace), Some("DONE"), None, None, None, 10)
            .await?
            .len(),
        1
    );
    let projected = store
        .graph_projection_records(GraphProjectionRead {
            scope: scope.clone(),
            table: "gke_nodes".to_owned(),
            ids: Some(vec![node_id.to_owned()]),
            metadata: kogwistar_store::MetadataFilter::default(),
            limit: 1,
        })
        .await?;
    assert_eq!(projected[0].embedding, Some(vec![0.0, 1.0, 0.0]));

    store
        .enqueue_index_job(NewIndexJob {
            job_id: "native-worker-stale-provider".to_owned(),
            namespace: namespace.clone(),
            entity_kind: "node".to_owned(),
            entity_id: node_id.to_owned(),
            index_kind: "node_embedding".to_owned(),
            op: "UPSERT".to_owned(),
            payload_json: None,
            max_retries: 3,
        })
        .await?;
    let stale_summary =
        kogwistar_engine::projection_worker::run_graph_projection_jobs_once_with_provider_input(
            &store,
            &store,
            1,
            30,
            Some(&namespace),
            move |input| async move {
                Ok(GraphRecord {
                    id: input.entity_id,
                    document: Some("stale source text".to_owned()),
                    metadata: Map::new(),
                    embedding: Some(vec![1.0, 0.0, 0.0]),
                })
            },
        )
        .await?;
    assert_eq!(stale_summary.projected, 1);
    let after_stale_result = store
        .graph_projection_records(GraphProjectionRead {
            scope: scope.clone(),
            table: "gke_nodes".to_owned(),
            ids: Some(vec![node_id.to_owned()]),
            metadata: kogwistar_store::MetadataFilter::default(),
            limit: 1,
        })
        .await?;
    assert_eq!(
        after_stale_result[0].document.as_deref(),
        Some("native projection worker text")
    );
    assert_eq!(after_stale_result[0].embedding, Some(vec![0.0, 1.0, 0.0]));

    // A stale DELETE must not remove a live replacement.
    store
        .enqueue_index_job(NewIndexJob {
            job_id: "native-worker-stale-delete".to_owned(),
            namespace: namespace.clone(),
            entity_kind: "node".to_owned(),
            entity_id: node_id.to_owned(),
            index_kind: "node_embedding".to_owned(),
            op: "DELETE".to_owned(),
            payload_json: None,
            max_retries: 3,
        })
        .await?;
    let delete_summary =
        kogwistar_engine::projection_worker::run_graph_projection_jobs_once_with_provider_input(
            &store,
            &store,
            1,
            30,
            Some(&namespace),
            |_input| async { panic!("DELETE must not invoke the embedding provider") },
        )
        .await?;
    assert_eq!(delete_summary.projected, 1);
    assert_eq!(
        store
            .graph_projection_records(GraphProjectionRead {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                ids: Some(vec![node_id.to_owned()]),
                metadata: kogwistar_store::MetadataFilter::default(),
                limit: 1,
            })
            .await?
            .len(),
        1
    );

    store
        .apply_graph_mutation(graph_mutation("native-worker-tombstone", "TOMBSTONE"))
        .await?;
    store
        .enqueue_index_job(NewIndexJob {
            job_id: "native-worker-tombstone-delete".to_owned(),
            namespace: namespace.clone(),
            entity_kind: "node".to_owned(),
            entity_id: node_id.to_owned(),
            index_kind: "node_embedding".to_owned(),
            op: "DELETE".to_owned(),
            payload_json: None,
            max_retries: 3,
        })
        .await?;
    let tombstone_summary =
        kogwistar_engine::projection_worker::run_graph_projection_jobs_once_with_provider_input(
            &store,
            &store,
            1,
            30,
            Some(&namespace),
            |_input| async { panic!("DELETE must not invoke the embedding provider") },
        )
        .await?;
    assert_eq!(tombstone_summary.projected, 1);
    assert!(
        store
            .graph_projection_records(GraphProjectionRead {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                ids: Some(vec![node_id.to_owned()]),
                metadata: kogwistar_store::MetadataFilter::default(),
                limit: 1,
            })
            .await?
            .is_empty()
    );

    let client = store.client().await?;
    client
        .batch_execute(&format!(
            "DROP SCHEMA {} CASCADE",
            Tables::new(&schema)?.quoted_schema
        ))
        .await
        .map_err(backend)?;
    Ok(())
}
#[tokio::test]
#[ignore = "requires KOGWISTAR_TEST_PG_DSN; run explicitly with --include-ignored"]
async fn recorded_runtime_and_claimed_handoff_are_atomic_when_dsn_available()
-> PostgresStoreResult<()> {
    let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
        .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
    let schema = test_schema("recorded_runtime");
    let store = PostgresStore::from_dsn(&dsn, &schema)?;
    let tables = Tables::new(&schema)?;
    store.ensure_schema().await?;

    let planner = store.client().await?;
    planner
        .batch_execute("SET enable_seqscan=off")
        .await
        .map_err(backend)?;
    let plan = planner
        .query(
            &format!(
                "EXPLAIN SELECT payload_json, seq FROM {} \
                     WHERE event_type='workflow.recorded_transition.v1' \
                     AND payload_json::jsonb->>'transition_id'=$1 LIMIT 1",
                tables.server_run_events
            ),
            &[&"index-plan-probe"],
        )
        .await
        .map_err(backend)?
        .iter()
        .map(|row| row.get::<_, String>(0))
        .collect::<Vec<_>>()
        .join(" ");
    assert!(
        plan.contains("idx_server_run_events_recorded_transition_id"),
        "{plan}"
    );

    let start_1 = store
        .apply_recorded_runtime_transition(runtime_start("run-1", "start-1"), false)
        .await?;
    store
        .project_lane_message(runtime_lane("run-1", "request-1"))
        .await?;
    let claimed = store
        .claim_projected_lane_messages("runtime", "python-workers", "worker-1", 1, 30)
        .await?;
    assert_eq!(claimed.len(), 1);
    let transition = runtime_result("run-1", "result-1", start_1.event_seq, "worker");
    let handoff = runtime_handoff("run-1", "request-1", "worker-1");
    let first = store
        .apply_claimed_recorded_runtime_transition(handoff.clone(), transition.clone(), false)
        .await?;
    let retry = store
        .apply_claimed_recorded_runtime_transition(handoff.clone(), transition.clone(), false)
        .await?;
    assert!(!first.idempotent && retry.idempotent);
    assert_eq!(first.event_seq, retry.event_seq);
    assert_eq!(
        store
            .get_projected_lane_message("request-1")
            .await?
            .unwrap()
            .status,
        "completed"
    );
    assert_eq!(
        store
            .read_recorded_runtime_state("run-1", "wf-1", "conv-1")
            .await?
            .unwrap()
            .state["answer"],
        serde_json::json!("worker")
    );
    assert!(matches!(
        store
            .apply_claimed_recorded_runtime_transition(
                handoff,
                runtime_result("run-1", "result-1", start_1.event_seq, "changed"),
                false,
            )
            .await,
        Err(PostgresStoreError::RecordedRuntimeConflict(_))
    ));

    let start_2 = store
        .apply_recorded_runtime_transition(runtime_start("run-2", "start-2"), false)
        .await?;
    store
        .project_lane_message(runtime_lane("run-2", "request-2"))
        .await?;
    store
        .claim_projected_lane_messages("runtime", "python-workers", "worker-2", 1, 30)
        .await?;
    assert!(matches!(
        store
            .apply_claimed_recorded_runtime_transition(
                runtime_handoff("run-2", "request-2", "wrong-worker"),
                runtime_result("run-2", "result-2", start_2.event_seq, "worker"),
                false,
            )
            .await,
        Err(PostgresStoreError::RecordedRuntimeConflict(_))
    ));

    let start_3 = store
        .apply_recorded_runtime_transition(runtime_start("run-3", "start-3"), false)
        .await?;
    store
        .project_lane_message(runtime_lane("run-3", "request-3"))
        .await?;
    store
        .claim_projected_lane_messages("runtime", "python-workers", "dead-worker", 1, -1)
        .await?;
    store
        .claim_projected_lane_messages("runtime", "python-workers", "new-worker", 1, 30)
        .await?;
    assert!(matches!(
        store
            .apply_claimed_recorded_runtime_transition(
                runtime_handoff("run-3", "request-3", "dead-worker"),
                runtime_result("run-3", "result-3", start_3.event_seq, "worker"),
                false,
            )
            .await,
        Err(PostgresStoreError::RecordedRuntimeConflict(_))
    ));
    let reclaimed = store
        .apply_claimed_recorded_runtime_transition(
            runtime_handoff("run-3", "request-3", "new-worker"),
            runtime_result("run-3", "result-3", start_3.event_seq, "worker"),
            false,
        )
        .await?;
    assert!(!reclaimed.idempotent);

    let start_4 = store
        .apply_recorded_runtime_transition(runtime_start("run-4", "start-4"), false)
        .await?;
    let before = store
        .list_server_run_events("run-4", 0, usize::MAX)
        .await?
        .len();
    assert!(matches!(
        store
            .apply_recorded_runtime_transition(
                runtime_result("run-4", "result-4", start_4.event_seq, "rolled"),
                true,
            )
            .await,
        Err(PostgresStoreError::TransactionAborted(_))
    ));
    assert_eq!(
        store
            .list_server_run_events("run-4", 0, usize::MAX)
            .await?
            .len(),
        before
    );

    let start_5 = store
        .apply_recorded_runtime_transition(runtime_start("run-5", "start-5"), false)
        .await?;
    store
        .project_lane_message(runtime_lane("run-5", "request-5"))
        .await?;
    store
        .claim_projected_lane_messages("runtime", "python-workers", "worker-5", 1, 30)
        .await?;
    store.request_server_run_cancel("run-5").await?;
    assert_eq!(
        store.get_server_run("run-5").await?.unwrap().status,
        "cancelled"
    );
    assert_eq!(
        store
            .read_recorded_runtime_state("run-5", "wf-1", "conv-1")
            .await?
            .unwrap()
            .status,
        kogwistar_runtime::RecordedRunStatus::Cancelled
    );
    assert_eq!(
        store
            .get_projected_lane_message("request-5")
            .await?
            .unwrap()
            .status,
        "cancelled"
    );
    assert!(matches!(
        store
            .apply_claimed_recorded_runtime_transition(
                runtime_handoff("run-5", "request-5", "worker-5"),
                runtime_result("run-5", "result-5", start_5.event_seq, "stale"),
                false,
            )
            .await,
        Err(PostgresStoreError::RecordedRuntimeConflict(_))
    ));

    for (run_id, message_id, owner) in [
        ("handled", "handled-request", "handled-worker"),
        ("unhandled", "unhandled-request", "unhandled-worker"),
        ("forged", "forged-request", "forged-worker"),
    ] {
        store
            .apply_recorded_runtime_transition(
                runtime_failure_start(run_id, &format!("start-{run_id}")),
                false,
            )
            .await?;
        store
            .project_lane_message(runtime_worker_lane(run_id, message_id))
            .await?;
        assert_eq!(
            store
                .claim_projected_lane_messages("runtime", "python-workers", owner, 1, 30,)
                .await?
                .len(),
            1
        );
    }

    let handled_handoff = runtime_worker_handoff("handled", "handled-request", "handled-worker");
    let handled_effect =
        runtime_failure_effect("handled-effect", &["recover"], serde_json::json!([]));
    let handled = store
        .apply_claimed_recorded_worker_effect(handled_handoff.clone(), handled_effect.clone())
        .await?;
    let handled_retry = store
        .apply_claimed_recorded_worker_effect(handled_handoff, handled_effect)
        .await?;
    assert!(!handled.idempotent && handled_retry.idempotent);
    assert_eq!(handled.event_seq, handled_retry.event_seq);
    assert_eq!(
        handled.reduced.state.status,
        kogwistar_runtime::RecordedRunStatus::Running
    );
    assert_eq!(handled.reduced.state.frontier.pending[0].0, "recover");
    assert_eq!(
        store
            .get_projected_lane_message("handled-request")
            .await?
            .unwrap()
            .status,
        "completed"
    );

    let unhandled = store
        .apply_claimed_recorded_worker_effect(
            runtime_worker_handoff("unhandled", "unhandled-request", "unhandled-worker"),
            runtime_failure_effect("unhandled-effect", &[], serde_json::json!([])),
        )
        .await?;
    assert_eq!(
        unhandled.reduced.state.status,
        kogwistar_runtime::RecordedRunStatus::Failed
    );
    assert!(unhandled.reduced.state.frontier.pending.is_empty());
    assert_eq!(
        store.get_server_run("unhandled").await?.unwrap().status,
        "failed"
    );

    assert!(matches!(
        store
            .apply_claimed_recorded_worker_effect(
                runtime_worker_handoff("forged", "forged-request", "forged-worker"),
                runtime_failure_effect(
                    "forged-effect",
                    &[],
                    serde_json::json!([{"node_id": "forged", "join_mask": 0}]),
                ),
            )
            .await,
        Err(PostgresStoreError::RecordedRuntime(_))
    ));
    assert_eq!(
        store
            .list_server_run_events("forged", 0, usize::MAX)
            .await?
            .len(),
        1
    );
    assert_eq!(
        store
            .get_projected_lane_message("forged-request")
            .await?
            .unwrap()
            .status,
        "claimed"
    );

    store
        .client()
        .await?
        .batch_execute(&format!("DROP SCHEMA {} CASCADE", tables.quoted_schema))
        .await
        .map_err(backend)
}
