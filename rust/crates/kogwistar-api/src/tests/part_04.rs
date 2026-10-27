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
            json!({"designer_id":"alice","node_id":"start","label":"Start","op":"start","start":true}),
            json!({"designer_id":"alice","node_id":"end","label":"End","op":"end","terminal":true}),
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

    #[tokio::test]
    async fn sqlite_service_events_fold_authoritative_history_for_rest_and_mcp() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use kogwistar_store_sqlite::NewRawEntityEvent;
        use std::time::{SystemTime, UNIX_EPOCH};
        use tower::ServiceExt;

        fn node(
            id: &str,
            service_id: &str,
            event_type: &str,
            ts_ms: i64,
            event_payload_json: &str,
        ) -> Value {
            json!({
                "id": id,
                "metadata": {
                    "entity_type": "service_event",
                    "service_id": service_id,
                    "service_event_type": event_type,
                    "ts_ms": ts_ms,
                },
                "properties": {"payload_json": event_payload_json},
            })
        }

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-service-events-{nonce}.sqlite3"));
        let store = SqliteStore::open(&path).unwrap();
        let append = |event_id: &str, entity_id: &str, op: &str, payload: Value| {
            store
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
                .unwrap();
        };
        append(
            "e1-add",
            "service_evt:svc-a:1",
            "ADD",
            node(
                "service_evt:svc-a:1",
                "svc-a",
                "service.starting",
                20,
                r#"{"phase":"old"}"#,
            ),
        );
        append(
            "e2-replace",
            "service_evt:svc-a:1",
            "REPLACE",
            node(
                "service_evt:svc-a:1",
                "svc-a",
                "service.started",
                20,
                r#"{"phase":"new"}"#,
            ),
        );
        append(
            "e3-other",
            "service_evt:svc-b:1",
            "ADD",
            node("service_evt:svc-b:1", "svc-b", "service.healthy", 1, "{}"),
        );
        append(
            "e4-earlier",
            "service_evt:svc-a:2",
            "ADD",
            node(
                "service_evt:svc-a:2",
                "svc-a",
                "service.healthy",
                10,
                "not-json",
            ),
        );
        append("e5-delete", "service_evt:svc-a:2", "DELETE", json!({}));
        append(
            "e6-unrelated",
            "node-a",
            "ADD",
            json!({"id":"node-a","metadata":{"entity_type":"node"}}),
        );

        let service = SqliteRunApplicationService::open(&path).unwrap();
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: Vec::new(),
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig::default(),
            },
            Arc::new(service.clone()),
        );
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/workflow/services/svc-a/events?limit=1")
                    .header("x-kogwistar-roles", "reader,rw")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body =
            serde_json::from_slice::<Value>(&to_bytes(response.into_body(), 16_384).await.unwrap())
                .unwrap();
        assert_eq!(body["service_id"], "svc-a");
        assert_eq!(body["events"].as_array().unwrap().len(), 1);
        assert_eq!(body["events"][0]["event_type"], "service.started");
        assert_eq!(body["events"][0]["payload"]["phase"], "new");

        let missing = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/workflow/services/missing/events")
                    .header("x-kogwistar-roles", "reader,rw")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(missing.status(), StatusCode::OK);
        let missing =
            serde_json::from_slice::<Value>(&to_bytes(missing.into_body(), 16_384).await.unwrap())
                .unwrap();
        assert_eq!(missing["events"], json!([]));

        let forbidden = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: "/api/workflow/services/svc-a/events".to_owned(),
                body: Vec::new(),
                principal: json!({"sub":"alice","role":"ro","capabilities":[]}),
            })
            .await;
        assert_eq!(forbidden.status, StatusCode::FORBIDDEN.as_u16());

        let mcp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/workflow")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader,rw")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":"service-events","method":"tools/call","params":{"name":"workflow.service_events","arguments":{"service_id":"svc-a","limit":10}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(mcp.status(), StatusCode::OK);
        let mcp =
            serde_json::from_slice::<Value>(&to_bytes(mcp.into_body(), 16_384).await.unwrap())
                .unwrap();
        assert_eq!(mcp["result"]["events"][0]["event_type"], "service.started");

        drop(service);
        drop(store);
        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn sqlite_service_repair_rebuilds_disposable_projection_for_rest_and_mcp() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use kogwistar_store_sqlite::NewRawEntityEvent;
        use std::time::{SystemTime, UNIX_EPOCH};
        use tower::ServiceExt;

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-service-repair-{nonce}.sqlite3"));
        let store = SqliteStore::open(&path).unwrap();
        let definition = json!({
            "id": "service_def:svc-repair:1",
            "metadata": {
                "entity_type": "service_definition",
                "service_id": "svc-repair",
                "updated_at_ms": 10,
                "created_at_ms": 10,
            },
            "properties": {
                "service_id": "svc-repair",
                "service_kind": "daemon",
                "target_kind": "workflow",
                "target_ref": "wf-repair",
                "target_config_json": "{}",
                "enabled": true,
                "autostart": false,
                "restart_policy_json": "{}",
                "heartbeat_ttl_ms": 60000,
                "trigger_specs_json": "[]",
                "security_scope": "workflow",
                "storage_namespace": "workflow",
                "execution_namespace": "workflow",
                "created_at_ms": 10,
                "updated_at_ms": 10,
            },
        });
        store
            .append_raw_entity_event(
                "workflow",
                NewRawEntityEvent {
                    event_id: "definition".to_owned(),
                    entity_kind: "node".to_owned(),
                    entity_id: "service_def:svc-repair:1".to_owned(),
                    op: "ADD".to_owned(),
                    payload_json: definition.to_string(),
                },
            )
            .unwrap();
        store
            .append_raw_entity_event(
                "workflow",
                NewRawEntityEvent {
                    event_id: "disabled".to_owned(),
                    entity_kind: "node".to_owned(),
                    entity_id: "service_evt:svc-repair:2".to_owned(),
                    op: "ADD".to_owned(),
                    payload_json: json!({
                        "id": "service_evt:svc-repair:2",
                        "metadata": {
                            "entity_type": "service_event",
                            "service_id": "svc-repair",
                            "service_event_type": "service.stopped",
                            "ts_ms": 20,
                        },
                        "properties": {"payload_json": r#"{"enabled":false}"#},
                    })
                    .to_string(),
                },
            )
            .unwrap();
        let service = SqliteRunApplicationService::open(&path).unwrap();
        let principal = json!({
            "sub": "operator",
            "role": "rw",
            "ns": "workflow",
            "capabilities": ["service.manage", "service.inspect", "project_view"],
        });
        let declared = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/workflow/services".to_owned(),
                body: serde_json::to_vec(&json!({
                    "service_id": "svc-declared",
                    "service_kind": "daemon",
                    "target_kind": "workflow",
                    "target_ref": "wf-declared",
                    "target_config": {
                        "conversation_id": "conv-declared",
                        "runtime_kind": "async"
                    },
                    "enabled": true,
                    "autostart": true,
                    "heartbeat_ttl_ms": 1234,
                    "trigger_specs": [{"type":"external event","enabled":true}],
                }))
                .unwrap(),
                principal: principal.clone(),
            })
            .await;
        assert_eq!(declared.status, StatusCode::OK.as_u16());
        let declared = serde_json::from_slice::<Value>(&declared.body).unwrap();
        assert_eq!(declared["service_id"], "svc-declared");
        assert_eq!(declared["target_ref"], "wf-declared");
        assert_eq!(declared["autostart"], true);
        assert_eq!(declared["health_status"], "degraded");
        assert_eq!(declared["heartbeat_ttl_ms"], 1234);
        let triggered = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/workflow/services/svc-declared/trigger".to_owned(),
                body: serde_json::to_vec(&json!({
                    "trigger_type": "external event",
                    "payload": {"source": "test"},
                }))
                .unwrap(),
                principal: principal.clone(),
            })
            .await;
        assert_eq!(triggered.status, StatusCode::OK.as_u16());
        let triggered = serde_json::from_slice::<Value>(&triggered.body).unwrap();
        let run_id = triggered["current_child_run_id"]
            .as_str()
            .unwrap()
            .to_owned();
        assert_eq!(triggered["last_trigger_type"], "external event");
        assert_eq!(triggered["current_child_status"], "queued");
        assert!(store.get_server_run(&run_id).unwrap().is_some());
        let lanes = store
            .list_projected_lane_messages(kogwistar_store::LaneMessageFilter {
                correlation_id: Some(run_id.clone()),
                limit: 10,
                ..kogwistar_store::LaneMessageFilter::default()
            })
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
        let retriggered = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/workflow/services/svc-declared/trigger".to_owned(),
                body: serde_json::to_vec(&json!({
                    "trigger_type": "external event",
                    "payload": {},
                }))
                .unwrap(),
                principal: principal.clone(),
            })
            .await;
        let retriggered = serde_json::from_slice::<Value>(&retriggered.body).unwrap();
        assert_eq!(retriggered["current_child_run_id"], run_id);
        assert_eq!(
            store
                .list_projected_lane_messages(kogwistar_store::LaneMessageFilter {
                    correlation_id: Some(run_id.clone()),
                    limit: 10,
                    ..kogwistar_store::LaneMessageFilter::default()
                })
                .unwrap()
                .len(),
            1
        );
        let repaired = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/workflow/services/svc-repair/repair".to_owned(),
                body: Vec::new(),
                principal: principal.clone(),
            })
            .await;
        assert_eq!(repaired.status, StatusCode::OK.as_u16());
        let body = serde_json::from_slice::<Value>(&repaired.body).unwrap();
        assert_eq!(body["projection"]["service_id"], "svc-repair");
        assert_eq!(body["projection"]["enabled"], false);
        assert_eq!(body["projection"]["lifecycle_status"], "stopped");
        store
            .clear_named_projection("service_registry", "svc-repair")
            .unwrap();

        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: Vec::new(),
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig::default(),
            },
            Arc::new(service.clone()),
        );
        let mcp_trigger = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/workflow")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader,rw")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":"trigger","method":"tools/call","params":{"name":"workflow.service_trigger","arguments":{"service_id":"svc-declared","trigger_type":"external event","payload":{"source":"mcp"}}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(mcp_trigger.status(), StatusCode::OK);
        let mcp_trigger = serde_json::from_slice::<Value>(
            &to_bytes(mcp_trigger.into_body(), 16_384).await.unwrap(),
        )
        .unwrap();
        assert_eq!(mcp_trigger["result"]["current_child_run_id"], run_id);
        let mcp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/workflow")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader,rw")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":"repair","method":"tools/call","params":{"name":"workflow.service_repair","arguments":{"service_id":"svc-repair"}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(mcp.status(), StatusCode::OK);
        let mcp =
            serde_json::from_slice::<Value>(&to_bytes(mcp.into_body(), 16_384).await.unwrap())
                .unwrap();
        assert_eq!(mcp["result"]["projection"]["enabled"], false);
        let forbidden = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/workflow/services/repair".to_owned(),
                body: Vec::new(),
                principal: json!({"sub":"reader","role":"ro","capabilities":["service.inspect"]}),
            })
            .await;
        assert_eq!(forbidden.status, StatusCode::FORBIDDEN.as_u16());
        let disabled = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/workflow/services/svc-repair/disable".to_owned(),
                body: Vec::new(),
                principal: principal.clone(),
            })
            .await;
        assert_eq!(disabled.status, StatusCode::OK.as_u16());
        let disabled = serde_json::from_slice::<Value>(&disabled.body).unwrap();
        assert_eq!(disabled["enabled"], false);
        assert_eq!(disabled["lifecycle_status"], "stopped");
        let enabled = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/workflow/services/svc-repair/enable".to_owned(),
                body: Vec::new(),
                principal,
            })
            .await;
        assert_eq!(enabled.status, StatusCode::OK.as_u16());
        let enabled = serde_json::from_slice::<Value>(&enabled.body).unwrap();
        assert_eq!(enabled["enabled"], true);
        // Python rebuild keeps the prior stopped lifecycle until the next
        // supervisor health tick emits a new lifecycle event.
        assert_eq!(enabled["lifecycle_status"], "stopped");
        let heartbeat = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/workflow/services/svc-repair/heartbeat".to_owned(),
                body: serde_json::to_vec(&json!({
                    "instance_id": "worker-1",
                    "payload": {"last_error": "recovering"},
                }))
                .unwrap(),
                principal: json!({
                    "sub": "worker",
                    "role": "rw",
                    "capabilities": ["service.heartbeat", "service.manage"],
                }),
            })
            .await;
        assert_eq!(heartbeat.status, StatusCode::OK.as_u16());
        let heartbeat = serde_json::from_slice::<Value>(&heartbeat.body).unwrap();
        assert_eq!(heartbeat["instance_id"], "worker-1");
        assert_eq!(heartbeat["health_status"], "healthy");
        assert_eq!(heartbeat["last_error"], "recovering");
        let history = store.replay_raw_events("workflow", 0, usize::MAX).unwrap();
        assert!(history.iter().any(|event| {
            event.payload_json.contains("service.enabled")
                && event.entity_id.starts_with("service_evt:svc-repair:")
        }));
        assert!(history.iter().any(|event| {
            event.payload_json.contains("service.error_changed")
                && event.entity_id.starts_with("service_evt:svc-repair:")
        }));
        drop(service);
        drop(store);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn service_route_and_event_fold_reject_unrelated_inputs() {
        assert_eq!(
            service_route("/api/workflow/services/svc/events?limit=1"),
            Some(("svc".to_owned(), ServiceRoute::Events))
        );
        assert_eq!(service_route("/api/workflow/services/svc/unknown"), None);
        assert_eq!(
            service_repair_route("/api/workflow/services/repair?limit=2"),
            Some(None)
        );
        assert_eq!(
            service_repair_route("/api/workflow/services/svc/repair"),
            Some(Some("svc".to_owned()))
        );
        assert_eq!(
            service_enabled_route("/api/workflow/services/svc/disable"),
            Some(("svc".to_owned(), false))
        );
        assert!(
            service_event_values(
                [(
                    1,
                    "node".to_owned(),
                    "bad".to_owned(),
                    "ADD".to_owned(),
                    "not-json".to_owned(),
                )],
                "svc",
                10,
            )
            .is_empty()
        );
        let projection = json!({"last_triggered_at_ms": 900});
        assert!(service_trigger_is_suppressed(
            &json!({"properties":{"enabled":false,"trigger_specs_json":"[]"}}),
            &projection,
            "external event",
            1_000,
        ));
        assert!(service_trigger_is_suppressed(
            &json!({"properties":{"enabled":true,"trigger_specs_json":r#"[{"type":"external event","enabled":true,"cooldown_ms":200}]"#}}),
            &projection,
            "external event",
            1_000,
        ));
        assert!(!service_trigger_is_suppressed(
            &json!({"properties":{"enabled":true,"trigger_specs_json":r#"[{"type":"graph change","enabled":false}]"#}}),
            &projection,
            "external event",
            1_000,
        ));
        assert_eq!(PENDING_SYSCALL_CUTOVER_OPS.len(), 4);
        assert!(PENDING_SYSCALL_CUTOVER_OPS.contains(&"invoke_tool"));
        assert!(!PENDING_SYSCALL_CUTOVER_OPS.contains(&"project_view"));
        assert!(!PENDING_SYSCALL_CUTOVER_OPS.contains(&"spawn_process"));
    }

    #[test]
    fn document_graph_validation_matches_provenance_and_pointer_contract() {
        let request = |body: Value| ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/document.validate_graph".to_owned(),
            body: serde_json::to_vec(&body).unwrap(),
            principal: json!({}),
        };
        let valid = document_graph_validation_effect(&request(json!({
            "doc_id":"doc-validate",
            "insertion_method":"pytest",
            "nodes":[{
                "id":"node-1",
                "label":"Node",
                "type":"entity",
                "summary":"Valid node",
                "metadata":{"pointers":[{
                    "source_cluster_id":"cluster-1",
                    "start_char":0,
                    "end_char":4,
                    "verbatim_text":"Node",
                }]},
            }],
            "edges":[{
                "id":"edge-1",
                "label":"Edge",
                "type":"relationship",
                "summary":"Valid edge",
                "relation":"links",
                "source_ids":["node-1"],
                "target_ids":["node-1"],
                "source_edge_ids":[],
                "target_edge_ids":[],
                "mentions":[{"spans":[{
                    "collection_page_url":"doc://doc-validate",
                    "document_page_url":"doc://doc-validate#cluster-1",
                    "doc_id":"doc-validate",
                    "insertion_method":"pytest",
                    "start_char":0,
                    "end_char":4,
                }]}],
            }],
        })))
        .unwrap();
        let valid = serde_json::from_slice::<Value>(&valid.body).unwrap();
        assert_eq!(valid, json!({"ok":true,"node_errors":{},"edge_errors":{}}));

        let invalid = document_graph_validation_effect(&request(json!({
            "doc_id":"doc-invalid",
            "nodes":[{"id":"node-bad","label":"Bad","type":"entity","summary":"","mentions":[]}],
            "edges":[{"id":"edge-bad","label":"Bad","type":"relationship","summary":"","mentions":[],"relation":"links","source_ids":[],"target_ids":[],"source_edge_ids":[],"target_edge_ids":[]}],
        })))
        .unwrap();
        let invalid = serde_json::from_slice::<Value>(&invalid.body).unwrap();
        assert_eq!(invalid["ok"], false);
        assert!(invalid["node_errors"]["node-bad"].is_string());
        assert!(invalid["edge_errors"]["edge-bad"].is_string());
    }

    #[test]
    fn internal_document_projection_reuses_wire_contract_and_write_authority() {
        let capabilities = Arc::new(Mutex::new(CapabilityState::default()));
        let request = |principal: Value| ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: INTERNAL_DOCUMENT_PROJECTION_ROUTE.to_owned(),
            body: serde_json::to_vec(&json!({
                "doc_id": "doc-projection",
                "doc_type": "text",
                "insertion_method": "parser",
                "content": "hello",
                "metadata": {"source": "fixture"}
            }))
            .unwrap(),
            principal,
        };
        let denied = document_projection_write(
            &request(json!({"sub":"reader","role":"ro","capabilities":[]})),
            &capabilities,
        )
        .unwrap_err();
        assert_eq!(denied.status, StatusCode::FORBIDDEN);

        let write = document_projection_write(
            &request(json!({
                "sub":"writer",
                "role":"rw",
                "ns":["docs"],
                "capabilities":["write_graph"]
            })),
            &capabilities,
        )
        .unwrap();
        assert_eq!(write.record.id, "doc-projection");
        assert_eq!(write.record.document.as_deref(), Some("hello"));
        assert_eq!(write.scope.namespace, "docs");
        assert_eq!(write.record.metadata["doc_type"], "text");
        assert_eq!(write.record.metadata["insertion_method"], "parser");

        let ocr_request = ApiEffectRequest {
            body: serde_json::to_vec(&json!({
                "doc_id": "doc-ocr",
                "doc_type": "ocr",
                "content": "{}"
            }))
            .unwrap(),
            ..request(json!({
                "sub":"writer",
                "role":"rw",
                "ns":["docs"],
                "capabilities":["write_graph"]
            }))
        };
        let ocr = document_projection_write(&ocr_request, &capabilities).unwrap_err();
        assert_eq!(ocr.status, StatusCode::NOT_IMPLEMENTED);
    }

    #[test]
    fn internal_graph_mutation_contract_requires_atomic_versioned_batches() {
        let body: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/graph-mutation-batch-v1.json"
        )))
        .unwrap();
        let input: GraphMutationBatchInput = serde_json::from_value(body.clone()).unwrap();
        let mutations = graph_mutation_batch_value(input).unwrap();
        assert_eq!(mutations.len(), 1);
        assert_eq!(mutations[0].scope.namespace, "tenant-a");
        assert_eq!(
            mutations[0].record.embedding.as_deref(),
            Some([1.0, 0.0].as_slice())
        );

        let graph_fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/python-graph-model-payloads-v1.json"
        )))
        .unwrap();
        let typed_payload = graph_fixture["cases"][0]["payload"].clone();
        let mut typed_body = body.clone();
        typed_body["mutations"][0]["payload"] = typed_payload.clone();
        typed_body["mutations"][0]["record"]["id"] = typed_payload["id"].clone();
        typed_body["mutations"][0]["model_envelope"] = json!({
            "schema_version": 1,
            "model_class": "node",
            "payload": typed_payload,
        });
        let typed_input: GraphMutationBatchInput =
            serde_json::from_value(typed_body.clone()).unwrap();
        let typed_mutations = graph_mutation_batch_value(typed_input.clone()).unwrap();
        assert!(validate_graph_mutation_envelopes(&typed_input, &typed_mutations).is_ok());

        typed_body["mutations"][0]["model_envelope"]["model_class"] = json!("edge");
        let wrong_kind: GraphMutationBatchInput = serde_json::from_value(typed_body).unwrap();
        let wrong_kind_mutations = graph_mutation_batch_value(wrong_kind.clone()).unwrap();
        assert!(validate_graph_mutation_envelopes(&wrong_kind, &wrong_kind_mutations).is_err());

        let mut invalid_table = body.clone();
        invalid_table["mutations"][0]["table"] = json!("gke_edges");
        let parsed: GraphMutationBatchInput = serde_json::from_value(invalid_table).unwrap();
        assert!(graph_mutation_batch_value(parsed).is_err());

        let mut invalid_dimension = body.clone();
        invalid_dimension["mutations"][0]["embedding_dim"] = json!(3);
        let parsed: GraphMutationBatchInput = serde_json::from_value(invalid_dimension).unwrap();
        assert!(graph_mutation_batch_value(parsed).is_err());

        let mut identity_drift = body.clone();
        identity_drift["mutations"][0]["record"]["id"] = json!("different-node");
        let parsed: GraphMutationBatchInput = serde_json::from_value(identity_drift).unwrap();
        assert!(graph_mutation_batch_value(parsed).is_err());

        let mut physical_delete = body.clone();
        physical_delete["mutations"][0]["op"] = json!("DELETE");
        let parsed: GraphMutationBatchInput = serde_json::from_value(physical_delete).unwrap();
        assert!(parsed.validate().is_err());

        for (field, value) in [("atomic", json!(false)), ("contract_version", json!(2))] {
            let mut invalid = json!({
                "contract_version": 1,
                "atomic": true,
                "mutations": [],
            });
            invalid[field] = value;
            let parsed: GraphMutationBatchInput = serde_json::from_value(invalid).unwrap();
            assert!(graph_mutation_batch_value(parsed).is_err());
        }
    }

    #[test]
    fn v1_graph_mutation_preserves_arbitrary_integer_tokens_through_store_dto() {
        let mut body: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/graph-mutation-batch-v1.json"
        )))
        .unwrap();
        body["mutations"][0]["payload"]["level_from_root"] = json!(1);
        body["mutations"][0]["record"]["metadata"]["level_from_root"] = json!(1);
        let safe = serde_json::to_string(&body).unwrap();
        let huge = "1606938044258990275541962092341162602522202993782792835301376";
        let wire = safe.replace(
            "\"level_from_root\":1",
            &format!("\"level_from_root\":{huge}"),
        );
        let input: GraphMutationBatchInput = serde_json::from_str(&wire).unwrap();
        let mutations = graph_mutation_batch_value(input).unwrap();

        assert!(serde_json::to_string(&mutations[0].payload)
            .unwrap()
            .contains(huge));
        assert!(serde_json::to_string(&mutations[0].record.metadata)
            .unwrap()
            .contains(huge));
    }

    #[test]
    fn graph_mutation_index_intents_are_explicit_and_scope_checked() {
        let mut body: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/graph-mutation-batch-v1.json"
        )))
        .unwrap();
        body["index_jobs"] = json!([{
            "job_id": "index-node-1",
            "namespace": "tenant-a",
            "entity_kind": "node",
            "entity_id": "node-1",
            "index_kind": "node_embedding",
            "op": "UPSERT",
            "payload_json": "{\"text\":\"hello\"}",
            "max_retries": 4
        }]);
        let input: GraphMutationBatchInput = serde_json::from_value(body.clone()).unwrap();
        let mutations = graph_mutation_batch_value(input.clone()).unwrap();
        let jobs = graph_index_jobs_value(&input, &mutations).unwrap();
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].coalesce_key(), "node:node-1:node_embedding");
        assert_eq!(jobs[0].max_retries, 4);

        for (field, value) in [
            ("namespace", json!("other-tenant")),
            ("op", json!("REPLACE")),
            ("max_retries", json!(-1)),
        ] {
            let mut invalid = body.clone();
            invalid["index_jobs"][0][field] = value;
            let input: GraphMutationBatchInput = serde_json::from_value(invalid).unwrap();
            assert!(graph_mutation_batch_value(input).is_err(), "{field}");
        }

        let mut invalid = body.clone();
        invalid["index_jobs"][0]["max_retries"] = json!(i32::MAX as i64 + 1);
        let input: GraphMutationBatchInput = serde_json::from_value(invalid).unwrap();
        assert!(graph_mutation_batch_value(input).is_err());
    }

    #[test]
    fn visualization_shells_embed_packaged_templates_without_jinja_drift() {
        for (path, marker) in [
            ("/viz/cytoscape?doc_id=doc-1&mode=graph", "Cytoscape"),
            ("/viz/d3?doc_id=doc-1&mode=graph", "D3 graph viewer"),
            ("/viz/go?doc_id=doc-1&mode=graph", "GoJS"),
        ] {
            let response = visualization_shell_effect(&ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: path.to_owned(),
                body: Vec::new(),
                principal: json!({}),
            })
            .unwrap();
            assert_eq!(response.status, StatusCode::OK.as_u16());
            assert_eq!(response.content_type, "text/html; charset=utf-8");
            let body = String::from_utf8(response.body).unwrap();
            assert!(body.contains(marker));
            assert!(!body.contains("{{"));
            assert!(!body.contains("{%"));
        }
    }

    #[test]
    fn runtime_start_tracks_only_reachable_join_obligations() {
        assert_eq!(runtime_join_outstanding(4, 0b0101), vec![1, 0, 1, 0]);
    }

    #[test]
    fn python_runtime_wire_vectors_round_trip_through_rust_dtos() {
        let vectors: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/adr015-runtime-wire-v1.json"
        )))
        .unwrap();
        let submit: RuntimeSubmitRun = serde_json::from_value(vectors["submit"].clone()).unwrap();
        let claim: RuntimeClaimRequest = serde_json::from_value(vectors["claim"].clone()).unwrap();
        let result: RuntimeWorkerResultRequest =
            serde_json::from_value(vectors["result_effect"].clone()).unwrap();
        let transition_result: RuntimeWorkerResultRequest =
            serde_json::from_value(vectors["result_transition"].clone()).unwrap();
        let mut legacy_worker_result = vectors["result_effect"].clone();
        legacy_worker_result
            .as_object_mut()
            .unwrap()
            .remove("transition");
        let _: RuntimeWorkerResultRequest = serde_json::from_value(legacy_worker_result).unwrap();
        let resume: RuntimeResumeRequest =
            serde_json::from_value(vectors["resume"].clone()).unwrap();
        let sqlite_work: RuntimeClaimedWork =
            serde_json::from_value(vectors["claimed_work_sqlite"].clone()).unwrap();
        let postgres_work: RuntimeClaimedWork =
            serde_json::from_value(vectors["claimed_work_postgres"].clone()).unwrap();
        let step_execute: kogwistar_runtime::RuntimeStepExecutePayload =
            serde_json::from_value(vectors["step_execute"].clone()).unwrap();
        assert_eq!(serde_json::to_value(submit).unwrap(), vectors["submit"]);
        assert_eq!(serde_json::to_value(claim).unwrap(), vectors["claim"]);
        assert_eq!(
            serde_json::to_value(result).unwrap(),
            vectors["result_effect"]
        );
        assert_eq!(
            serde_json::to_value(transition_result).unwrap(),
            vectors["result_transition"]
        );
        assert_eq!(serde_json::to_value(resume).unwrap(), vectors["resume"]);
        assert_eq!(
            serde_json::to_value(sqlite_work).unwrap(),
            vectors["claimed_work_sqlite"]
        );
        assert_eq!(
            serde_json::to_value(postgres_work).unwrap(),
            vectors["claimed_work_postgres"]
        );
        assert_eq!(
            serde_json::to_value(step_execute).unwrap(),
            vectors["step_execute"]
        );
    }

    #[test]
    fn runtime_wire_dtos_reject_unknown_root_and_nested_fields() {
        let vectors: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/adr015-runtime-wire-v1.json"
        )))
        .unwrap();
        macro_rules! assert_unknown_field_rejected {
            ($dto:ty, $value:expr) => {{
                let mut value = $value;
                value["unexpected"] = json!(true);
                let error = serde_json::from_value::<$dto>(value).unwrap_err();
                assert!(error.to_string().contains("unknown field"));
            }};
        }

        assert_unknown_field_rejected!(RuntimeSubmitRun, vectors["submit"].clone());
        assert_unknown_field_rejected!(RuntimeClaimRequest, vectors["claim"].clone());
        assert_unknown_field_rejected!(
            RuntimeWorkerResultRequest,
            vectors["result_effect"].clone()
        );
        assert_unknown_field_rejected!(RuntimeResumeRequest, vectors["resume"].clone());
        assert_unknown_field_rejected!(RuntimeClaimedWork, vectors["claimed_work_sqlite"].clone());
        assert_unknown_field_rejected!(
            kogwistar_runtime::RuntimeStepExecutePayload,
            vectors["step_execute"].clone()
        );

        let mut nested_route = vectors["submit"].clone();
        nested_route["runtime_routes"][0]["unexpected"] = json!(true);
        let error = serde_json::from_value::<RuntimeSubmitRun>(nested_route).unwrap_err();
        assert!(error.to_string().contains("unknown field"));
    }

    #[test]
    fn runtime_start_builder_preserves_all_python_wire_identity() {
        let vectors: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/adr015-runtime-wire-v1.json"
        )))
        .unwrap();
        let mut payload = vectors["submit"].clone();
        payload["expected_event_seq"] = json!(7);
        let transition = runtime_start_transition(&payload, "run-1");
        assert_eq!(transition.run_id, "run-1");
        assert_eq!(transition.workflow_id, "wf-1");
        assert_eq!(transition.conversation_id, "conv-1");
        assert_eq!(transition.user_id.as_deref(), Some("user-1"));
        assert_eq!(transition.user_turn_node_id, None);
        assert_eq!(transition.node_id.as_deref(), Some("entry"));
        assert_eq!(transition.token_id.as_deref(), Some("run-1"));
        assert_eq!(transition.expected_event_seq, 7);
        let frontier = transition.frontier.unwrap();
        assert_eq!(frontier.pending[0].0, "entry");
        assert_eq!(frontier.pending[0].1, 1);
        assert_eq!(frontier.join_node_ids, vec!["join"]);
        assert_eq!(frontier.join_outstanding, vec![1]);
    }

    #[test]
    fn worker_protocol_tracks_runtime_kind_and_service_payload_state() {
        assert_eq!(worker_protocol_for_runtime_kind("sync"), "sync-v1");
        assert_eq!(worker_protocol_for_runtime_kind("async"), "async-v2");
        assert_eq!(worker_protocol_for_runtime_kind("unknown"), "sync-v1");
        let async_attestation = runtime_implementation_attestation("async");
        assert_eq!(async_attestation["schema_version"], 1);
        assert_eq!(async_attestation["scheduler"]["language"], "rust");
        assert_eq!(
            async_attestation["scheduler"]["version"],
            env!("CARGO_PKG_VERSION")
        );
        assert_eq!(async_attestation["worker_contract"]["protocol"], "async-v2");

        let mut sync_attestation = runtime_implementation_attestation("sync");
        sync_attestation["scheduler"]
            .as_object_mut()
            .unwrap()
            .remove("version");
        let shared_attestation: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/runtime-implementation-attestation-v1.json"
        )))
        .unwrap();
        assert_eq!(sync_attestation, shared_attestation);

        let mut state = serde_json::Map::new();
        state.insert(
            "_rt_worker_protocol".to_owned(),
            Value::String(worker_protocol_for_runtime_kind("async").to_owned()),
        );
        let recorded = kogwistar_runtime::RecordedRuntimeState {
            contract_version: kogwistar_runtime::RECORDED_RUNTIME_CONTRACT_VERSION,
            run_id: "run".to_owned(),
            workflow_id: "workflow".to_owned(),
            conversation_id: "conversation".to_owned(),
            user_turn_node_id: None,
            state,
            static_routes: Vec::new(),
            frontier: kogwistar_runtime::RuntimeFrontier {
                pending: Vec::new(),
                suspended: Vec::new(),
                join_node_ids: Vec::new(),
                join_outstanding: Vec::new(),
                join_waiters: BTreeMap::new(),
            },
            status: kogwistar_runtime::RecordedRunStatus::Running,
            last_step_seq: 0,
            wait_reason: None,
            resume_payload: None,
            last_node_id: None,
            last_token_id: None,
            last_parent_token_id: None,
        };
        let payload = kogwistar_runtime::RuntimeStepExecutePayload::from_recorded_state(
            &recorded,
            kogwistar_runtime::RuntimeStepExecuteRequest {
                node_id: "node".to_owned(),
                op: json!("op"),
                join_mask: 0,
                token_id: "token".to_owned(),
                parent_token_id: None,
                step_seq: 1,
                expected_event_seq: 1,
                resume_effect: None,
            },
        );
        assert_eq!(payload.worker_protocol, "async-v2");
    }
    struct FixedEmbeddingProvider {
        profile: kogwistar_engine::embedding::EmbeddingProfile,
        vector: Vec<f64>,
        calls: Arc<Mutex<Vec<Vec<String>>>>,
        fail: Option<Arc<std::sync::atomic::AtomicBool>>,
    }

    impl kogwistar_engine::embedding::EmbeddingProvider for FixedEmbeddingProvider {
        fn profile(&self) -> &kogwistar_engine::embedding::EmbeddingProfile {
            &self.profile
        }

        fn embed<'a>(
            &'a self,
            texts: &'a [String],
        ) -> kogwistar_engine::embedding::EmbeddingFuture<'a> {
            Box::pin(async move {
                self.calls.lock().unwrap().push(texts.to_vec());
                if self
                    .fail
                    .as_ref()
                    .is_some_and(|fail| fail.load(std::sync::atomic::Ordering::Relaxed))
                {
                    return Err(kogwistar_engine::embedding::EmbeddingError::Request(
                        "test provider failure".to_owned(),
                    ));
                }
                Ok(texts.iter().map(|_| self.vector.clone()).collect())
            })
        }
    }

    fn graph_mutation_batch_without_vector() -> GraphMutationBatchInput {
        let mut body: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/graph-mutation-batch-v1.json"
        )))
        .unwrap();
        body["mutations"][0]["record"]["document"] = json!("a source-grounded node");
        body["mutations"][0]["record"]["embedding"] = Value::Null;
        body["mutations"][0]["embedding_dim"] = json!(0);
        serde_json::from_value(body).unwrap()
    }

    #[tokio::test]
    async fn configured_rust_provider_fills_missing_graph_vector_before_atomic_store_call() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let provider = Arc::new(FixedEmbeddingProvider {
            profile: kogwistar_engine::embedding::EmbeddingProfile {
                provider: "test".to_owned(),
                model: "fixed-v1".to_owned(),
                dimension: 2,
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
            vector: vec![0.6, 0.8],
            calls: calls.clone(),
            fail: None,
        });
        let service = PostgresRunApplicationService::from_dsn(
            "postgresql://test:test@127.0.0.1:1/test",
            "public",
        )
        .unwrap()
        .with_embedding_provider(provider);
        let mut batch = graph_mutation_batch_without_vector();

        service.embed_missing_graph_vectors(&mut batch).await.unwrap();

        assert_eq!(batch.mutations[0].record.embedding, Some(vec![0.6, 0.8]));
        assert_eq!(batch.mutations[0].embedding_dim, 2);
        assert_eq!(calls.lock().unwrap().as_slice(), &[vec!["a source-grounded node".to_owned()]]);
    }

    #[tokio::test]
    async fn provider_does_not_turn_durable_two_stage_batch_into_synchronous_embedding() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let provider = Arc::new(FixedEmbeddingProvider {
            profile: kogwistar_engine::embedding::EmbeddingProfile {
                provider: "test".to_owned(),
                model: "fixed-v1".to_owned(),
                dimension: 2,
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
            vector: vec![0.6, 0.8],
            calls: calls.clone(),
            fail: None,
        });
        let service = PostgresRunApplicationService::from_dsn(
            "postgresql://test:test@127.0.0.1:1/test",
            "public",
        )
        .unwrap()
        .with_embedding_provider(provider);
        let mut batch = graph_mutation_batch_without_vector();
        batch.index_jobs.push(kogwistar_contracts::GraphIndexJobV1 {
            job_id: "index-node-1".to_owned(),
            namespace: "tenant-a".to_owned(),
            entity_kind: "node".to_owned(),
            entity_id: "node-1".to_owned(),
            index_kind: "node_embedding".to_owned(),
            op: "UPSERT".to_owned(),
            payload_json: None,
            max_retries: 3,
        });

        service.embed_missing_graph_vectors(&mut batch).await.unwrap();

        assert_eq!(batch.mutations[0].record.embedding, None);
        assert!(calls.lock().unwrap().is_empty());
    }
