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
        let resources = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: "/api/workflow/resources".to_owned(),
                body: Vec::new(),
                principal: principal.clone(),
            })
            .await;
        assert_eq!(resources.status, StatusCode::OK.as_u16());
        let resources: Value = serde_json::from_slice(&resources.body).unwrap();
        assert_eq!(resources["services"]["total_services"], 1);
        assert_eq!(resources["services"]["by_health"]["degraded"], 1);
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
    fn native_claim_attempt_is_internal_and_matches_trace_fixture() {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/runtime-trace-context-v1.json"
        )))
        .unwrap();
        let context = &fixture["context"];
        let payload = json!({
            "contract_version": 1,
            "kind": "workflow.step.execute",
            "worker_protocol": "sync-v1",
            "run_id": context["run_id"],
            "workflow_id": "workflow",
            "conversation_id": "conversation",
            "turn_node_id": null,
            "node_id": context["node_id"],
            "op": "noop",
            "executor_target": kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET,
            "op_version": "v1",
            "runtime_routes": [],
            "join_mask": 0,
            "token_id": context["token_id"],
            "parent_token_id": null,
            "step_seq": context["step_seq"],
            "expected_event_seq": 0,
            "state": {},
            "resume_effect": null
        });
        let lane = ProjectedLaneMessage {
            message_id: "message".to_owned(),
            namespace: "namespace".to_owned(),
            purpose: "runtime".to_owned(),
            inbox_id: "inbox".to_owned(),
            conversation_id: "conversation".to_owned(),
            recipient_id: "worker".to_owned(),
            sender_id: "scheduler".to_owned(),
            msg_type: "workflow.step.execute".to_owned(),
            status: "claimed".to_owned(),
            seq: 1,
            conversation_seq: 1,
            claimed_by: Some("worker".to_owned()),
            lease_until: None,
            retry_count: 1,
            created_at: 0,
            available_at: 0,
            run_id: Some("run-42".to_owned()),
            step_id: Some("parse".to_owned()),
            correlation_id: Some("correlation".to_owned()),
            payload_json: None,
            error_json: None,
            prev_message_id: None,
            next_message_id: None,
            inbox_tail_message_id: None,
            conversation_tail_message_id: None,
        };
        let python_retry_lane = ProjectedLaneMessage {
            recipient_id: "python-worker-env-test".to_owned(),
            ..lane.clone()
        };
        let python_retry_work = runtime_claimed_work(
            python_retry_lane,
            "python-process-1",
            "run-42".to_owned(),
            json!({"kind":"workflow.step.execute"}),
            0,
        );
        assert_eq!(python_retry_work.payload["attempt"], 2);
        let python_retry_wire = serde_json::to_value(&python_retry_work).unwrap();
        assert!(python_retry_wire.get("attempt_number").is_none());
        assert_eq!(python_retry_wire["payload"]["attempt"], 2);

        let first_python_lane = ProjectedLaneMessage {
            recipient_id: "python-worker".to_owned(),
            retry_count: 0,
            ..lane.clone()
        };
        let first_python_work = runtime_claimed_work(
            first_python_lane,
            "python-process-1",
            "run-42".to_owned(),
            json!({"kind":"workflow.step.execute"}),
            0,
        );
        assert!(first_python_work.payload.get("attempt").is_none());

        let work = runtime_claimed_work(
            lane,
            "rust-native-worker",
            "run-42".to_owned(),
            payload,
            0,
        );
        let worker_wire = serde_json::to_value(&work).unwrap();
        assert!(worker_wire.get("attempt_number").is_none());

        let (handoff, request) = native_request_for_claimed_work(&work).unwrap();
        assert_eq!(request.ephemeral_context["_trace_attempt"], context["attempt"]);
        let effect = kogwistar_runtime::execute_native_step(&request).unwrap();
        assert_eq!(effect.trace_events[0]["event"], "runtime.native.step.completed");
        for (field, expected) in fixture["expected"].as_object().unwrap() {
            assert_eq!(&effect.trace_events[0][field], expected, "field {field}");
        }
        let failed = failed_native_result(handoff, &request, "TEST_FAILURE", "test failure");
        for (field, expected) in fixture["expected"].as_object().unwrap() {
            assert_eq!(
                &failed.effect.as_ref().unwrap().trace_events[0][field],
                expected,
                "failed trace field {field}"
            );
        }
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

    #[tokio::test]
    async fn embedding_worker_requires_provider_before_opening_or_claiming_postgres_jobs() {
        let service = PostgresRunApplicationService::from_dsn(
            "postgresql://test:test@127.0.0.1:1/test",
            "public",
        )
        .unwrap();
        let error = service
            .run_embedding_projection_jobs_once(1, 30, Some("tenant-a"))
            .await
            .unwrap_err();
        assert_eq!(error, "Rust embedding provider is not configured");
    }

    #[test]
    fn sqlite_chroma_stage1_recovery_projects_edge_and_hyperedge_fields() {
        let model_fixture: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/python-graph-model-payloads-v1.json"
        ))
        .unwrap();
        let recovery_fixture: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/stage1-recovery-document-v2.json"
        ))
        .unwrap();
        let mut payload = model_fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|case| case["kind"] == "edge")
            .unwrap()["payload"]
            .clone();
        let entity_id = recovery_fixture["edge_entity_id"].as_str().unwrap();
        payload["id"] = json!(entity_id);
        let revision = kogwistar_engine::canonical_revision::CanonicalEntityRevision {
            namespace: "tenant-a".to_owned(),
            entity_kind: "edge".to_owned(),
            entity_id: entity_id.to_owned(),
            revision: 7,
            state: "active".to_owned(),
            payload,
        };

        let recovered = RevisionFencedChromaProjectionStore::restore_stage1_from_revision(
            &revision,
            "fixture-fingerprint",
        )
        .unwrap();
        assert_eq!(
            recovered.payload["document"].as_str(),
            recovery_fixture["edge_expected_document"].as_str()
        );
        let metadata = &recovered.payload["metadata"];
        assert_eq!(metadata["namespace"], "tenant-a");
        assert_eq!(metadata["node_endpoint_count"], 2);
        assert_eq!(metadata["edge_endpoint_count"], 0);
        assert_eq!(metadata["source_ids"], "[\"parity-node\"]");
        assert!(metadata.get("references").and_then(Value::as_str).is_some());
        assert_eq!(recovered.projection_schema_version, 2);
    }

    #[tokio::test]
    async fn sqlite_chroma_embedding_worker_projects_revision_fenced_stage1_document() {
        use kogwistar_store::NamedProjectionWrite;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let chroma = tokio::spawn(async move {
            let mut observed = Vec::new();
            for _ in 0..5 {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut bytes = Vec::new();
                let mut chunk = [0_u8; 4096];
                let (split, length) = loop {
                    let count = stream.read(&mut chunk).await.unwrap();
                    assert!(count > 0, "Chroma client closed before sending a full request");
                    bytes.extend_from_slice(&chunk[..count]);
                    if let Some(split) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
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
                let head = String::from_utf8_lossy(&bytes[..split]);
                let request_line = head.lines().next().unwrap().to_owned();
                let body: Value = serde_json::from_slice(&bytes[split + 4..split + 4 + length])
                    .unwrap();
                let response = if request_line.contains("/collections HTTP") {
                    json!({
                        "id": format!("collection-{}", body["name"].as_str().unwrap()),
                        "metadata": {"hnsw:space":"cosine"}
                    })
                } else {
                    json!({})
                };
                let response = response.to_string();
                let response = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                    response.len(),
                    response
                );
                stream.write_all(response.as_bytes()).await.unwrap();
                observed.push((request_line, body));
            }
            observed
        });

        let path = std::env::temp_dir().join(format!(
            "kogwistar-sqlite-chroma-embedding-{}.sqlite3",
            uuid::Uuid::new_v4()
        ));
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
        let service = open_sqlite_application_service(
            &path,
            Some(SqliteChromaGraphConfig {
                endpoint: endpoint.clone(),
                metric: DistanceMetric::Cosine,
                node_collection: "worker-nodes".to_owned(),
                edge_collection: "worker-edges".to_owned(),
            }),
        )
        .await
        .unwrap()
        .with_embedding_provider(provider.clone());

        let namespace = "worker-tenant";
        let recovery_fixture: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/stage1-recovery-document-v2.json"
        ))
        .unwrap();
        let entity_id = recovery_fixture["entity_id"].as_str().unwrap();
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/python-graph-model-payloads-v1.json"
        ))
        .unwrap();
        let mut entity_payload = fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|case| case["kind"] == "node")
            .unwrap()["payload"]
            .clone();
        entity_payload["id"] = json!(entity_id);
        entity_payload["metadata"]["namespace"] = Value::Null;
        let mut expected_document_payload = entity_payload.clone();
        expected_document_payload.as_object_mut().unwrap().remove("embedding");
        expected_document_payload.as_object_mut().unwrap().remove("metadata");
        let document = kogwistar_contracts::canonical_json(&expected_document_payload, false);
        assert_eq!(document, recovery_fixture["expected_document"].as_str().unwrap());
        let appended = service
            .store
            .append_raw_entity_event(
                namespace,
                SqliteNewRawEntityEvent {
                    event_id: "worker-node-add".to_owned(),
                    entity_kind: "node".to_owned(),
                    entity_id: entity_id.to_owned(),
                    op: "ADD".to_owned(),
                    payload_json: entity_payload.to_string(),
                },
            )
            .unwrap();
        let revision = sqlite_canonical_entity_revision(
            &service.store,
            namespace,
            "node",
            entity_id,
        )
        .unwrap()
        .unwrap();
        let job_payload = kogwistar_engine::canonical_revision::canonical_revision_job_payload(
            &revision,
        );
        let fingerprint = serde_json::from_str::<Value>(&job_payload).unwrap()
            ["source_fingerprint"]
            .as_str()
            .unwrap()
            .to_owned();
        service
            .store
            .enqueue_index_job(NewIndexJob {
                job_id: "worker-node-embedding".to_owned(),
                namespace: namespace.to_owned(),
                entity_kind: "node".to_owned(),
                entity_id: entity_id.to_owned(),
                index_kind: "node_embedding".to_owned(),
                op: "UPSERT".to_owned(),
                payload_json: Some(job_payload),
                max_retries: 3,
            })
            .unwrap();
        service
            .store
            .enqueue_index_job(NewIndexJob {
                job_id: "worker-node-refs".to_owned(),
                namespace: namespace.to_owned(),
                entity_kind: "node".to_owned(),
                entity_id: entity_id.to_owned(),
                index_kind: "node_refs".to_owned(),
                op: "UPSERT".to_owned(),
                payload_json: None,
                max_retries: 3,
            })
            .unwrap();

        // The SQLite queue and canonical event are durable across a Rust
        // application-service restart; Chroma promotion remains eventual.
        drop(service);
        let service = open_sqlite_application_service(
            &path,
            Some(SqliteChromaGraphConfig {
                endpoint,
                metric: DistanceMetric::Cosine,
                node_collection: "worker-nodes".to_owned(),
                edge_collection: "worker-edges".to_owned(),
            }),
        )
        .await
        .unwrap()
        .with_embedding_provider(provider);
        assert_eq!(
            sqlite_canonical_entity_revision(&service.store, namespace, "node", entity_id)
                .unwrap()
                .unwrap()
                .revision,
            appended.event.seq
        );
        assert_eq!(
            service
                .store
                .list_index_jobs(Some(namespace), Some("PENDING"), None, None, None, 10)
                .unwrap()
                .iter()
                .filter(|job| job.job_id == "worker-node-embedding")
                .count(),
            1
        );

        let summary = service
            .run_embedding_projection_jobs_once(1, 30, Some(namespace))
            .await
            .unwrap();
        assert_eq!(summary.claimed, 1);
        assert_eq!(summary.projected, 1);
        assert!(service
            .store
            .get_stage1_node_projection(namespace, &format!("node:{entity_id}"))
            .unwrap()
            .is_none());
        assert_eq!(
            calls.lock().unwrap().as_slice(),
            &[vec![document.to_owned()]]
        );
        let jobs = service
            .store
            .list_index_jobs(Some(namespace), None, None, None, None, 10)
            .unwrap();
        assert_eq!(
            jobs.iter()
                .find(|job| job.job_id == "worker-node-embedding")
                .unwrap()
                .status,
            "DONE"
        );
        assert_eq!(
            jobs.iter()
                .find(|job| job.job_id == "worker-node-refs")
                .unwrap()
                .status,
            "PENDING"
        );
        let requests = chroma.await.unwrap();
        let (upsert_path, upsert) = requests
            .iter()
            .find(|(line, _)| line.contains("/upsert"))
            .expect("the worker writes the embedded record to Chroma");
        assert!(upsert_path.contains("worker-nodes"));
        assert_eq!(upsert["ids"], json!([entity_id]));
        assert_eq!(upsert["documents"], json!([document]));
        assert_eq!(upsert["embeddings"][0], json!([0.6, 0.8]));
        let metadata: Value = serde_json::from_str(
            upsert["metadatas"][0]["_kogwistar_original_metadata"]
                .as_str()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(metadata["namespace"], namespace);
        assert_eq!(metadata["doc_id"], "doc:rust-python-parity");
        assert_eq!(
            metadata["properties"],
            "{\"attributes\":[\"unicode-雪\",\"nested-payload\"]}"
        );
        assert!(metadata.get("redirect_to_id").is_none());
        assert_eq!(metadata["_kogwistar_stage2_ready"], true);
        assert_eq!(metadata["_kogwistar_source_revision"], appended.event.seq);
        assert_eq!(metadata["_kogwistar_source_fingerprint"], fingerprint);

        let stale_id = "stale-stage1-node";
        let stale_appended = service
            .store
            .append_raw_entity_event(
                namespace,
                SqliteNewRawEntityEvent {
                    event_id: "stale-stage1-node-add".to_owned(),
                    entity_kind: "node".to_owned(),
                    entity_id: stale_id.to_owned(),
                    op: "ADD".to_owned(),
                    payload_json: json!({"id":stale_id,"document":"canonical v1"}).to_string(),
                },
            )
            .unwrap();
        let stale_revision = sqlite_canonical_entity_revision(
            &service.store,
            namespace,
            "node",
            stale_id,
        )
        .unwrap()
        .unwrap();
        let stale_job_payload =
            kogwistar_engine::canonical_revision::canonical_revision_job_payload(&stale_revision);
        service
            .store
            .replace_stage1_node_projection(
                namespace,
                "node:stale-stage1-node",
                NamedProjectionWrite {
                    payload: json!({
                        "document":"stale stage one text",
                        "metadata":{},
                        "source_fingerprint":"not-the-canonical-fingerprint"
                    })
                    .as_object()
                    .unwrap()
                    .clone(),
                    last_authoritative_seq: stale_appended.event.seq,
                    last_materialized_seq: stale_appended.event.seq,
                    projection_schema_version: 1,
                    materialization_status: "ready".to_owned(),
                },
            )
            .unwrap();
        service
            .store
            .enqueue_index_job(NewIndexJob {
                job_id: "stale-stage1-embedding".to_owned(),
                namespace: namespace.to_owned(),
                entity_kind: "node".to_owned(),
                entity_id: stale_id.to_owned(),
                index_kind: "node_embedding".to_owned(),
                op: "UPSERT".to_owned(),
                payload_json: Some(stale_job_payload),
                max_retries: 3,
            })
            .unwrap();
        let stale_summary = service
            .run_embedding_projection_jobs_once(1, 30, Some(namespace))
            .await
            .unwrap();
        assert_eq!(stale_summary.claimed, 1);
        assert_eq!(stale_summary.projected, 0);
        assert_eq!(stale_summary.retried, 1);
        assert_eq!(calls.lock().unwrap().as_slice(), &[vec![document.to_owned()]]);
        let stale_job = service
            .store
            .list_index_jobs(Some(namespace), Some("PENDING"), None, None, None, 10)
            .unwrap()
            .into_iter()
            .find(|job| job.job_id == "stale-stage1-embedding")
            .expect("mismatched Stage-1 revision remains retryable");
        assert!(stale_job.last_error.as_deref().unwrap_or_default().contains("revisions do not match"));

        drop(service);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn sqlite_chroma_reconciliation_removes_orphaned_tombstone_projection() {
        use kogwistar_store::NamedProjectionWrite;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let namespace = "reconcile-tenant";
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let chroma = tokio::spawn(async move {
            let mut observed = Vec::new();
            let mut orphan_deleted = false;
            for _ in 0..9 {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut bytes = Vec::new();
                let mut chunk = [0_u8; 4096];
                let (split, length) = loop {
                    let count = stream.read(&mut chunk).await.unwrap();
                    assert!(count > 0, "Chroma client closed before sending a full request");
                    bytes.extend_from_slice(&chunk[..count]);
                    if let Some(split) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
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
                let head = String::from_utf8_lossy(&bytes[..split]);
                let request_line = head.lines().next().unwrap().to_owned();
                let body: Value = serde_json::from_slice(&bytes[split + 4..split + 4 + length])
                    .unwrap();
                let response = if request_line.contains("/collections HTTP") {
                    json!({
                        "id": format!("collection-{}", body["name"].as_str().unwrap()),
                        "metadata": {"hnsw:space":"cosine"}
                    })
                } else if request_line.contains("/get")
                    && request_line.contains("collection-reconcile-nodes")
                {
                    let row = |id: &str, revision: i64| {
                        let original = json!({
                            "namespace": namespace,
                            "_kogwistar_stage2_ready": true,
                            "_kogwistar_source_revision": revision,
                            "_kogwistar_source_fingerprint": "stale-fingerprint"
                        });
                        json!({
                            "id": id,
                            "document": "stale",
                            "metadata": {
                                "_kogwistar_namespace": namespace,
                                "_kogwistar_original_metadata": original.to_string()
                            },
                            "embedding": [0.2, 0.8]
                        })
                    };
                    let rows = if let Some(ids) = body.get("ids").and_then(Value::as_array) {
                        ids.iter()
                            .filter_map(Value::as_str)
                            .filter(|id| *id != "orphan" || !orphan_deleted)
                            .map(|id| row(id, 1))
                            .collect::<Vec<_>>()
                    } else {
                        let mut all_rows = Vec::new();
                        if !orphan_deleted {
                            all_rows.push(row("orphan", 1));
                        }
                        all_rows.push(row("stale-active", 1));
                        let offset = body["offset"].as_u64().unwrap_or(0) as usize;
                        let limit = body["limit"].as_u64().unwrap_or(2) as usize;
                        all_rows.into_iter().skip(offset).take(limit).collect()
                    };
                    json!({
                        "ids": rows.iter().map(|row| row["id"].clone()).collect::<Vec<_>>(),
                        "documents": rows.iter().map(|row| row["document"].clone()).collect::<Vec<_>>(),
                        "metadatas": rows.iter().map(|row| row["metadata"].clone()).collect::<Vec<_>>(),
                        "embeddings": rows.iter().map(|row| row["embedding"].clone()).collect::<Vec<_>>()
                    })
                } else {
                    json!({"ids":[],"documents":[],"metadatas":[],"embeddings":[]})
                };
                if request_line.contains("/delete") && body["ids"] == json!(["orphan"]) {
                    orphan_deleted = true;
                }
                let response = response.to_string();
                let response = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                    response.len(),
                    response
                );
                stream.write_all(response.as_bytes()).await.unwrap();
                observed.push((request_line, body));
            }
            observed
        });

        let path = std::env::temp_dir().join(format!(
            "kogwistar-sqlite-chroma-reconcile-{}.sqlite3",
            uuid::Uuid::new_v4()
        ));
        let service = open_sqlite_application_service(
            &path,
            Some(SqliteChromaGraphConfig {
                endpoint,
                metric: DistanceMetric::Cosine,
                node_collection: "reconcile-nodes".to_owned(),
                edge_collection: "reconcile-edges".to_owned(),
            }),
        )
        .await
        .unwrap();
        service
            .store
            .append_raw_entity_event(
                namespace,
                SqliteNewRawEntityEvent {
                    event_id: "orphan-tombstone".to_owned(),
                    entity_kind: "node".to_owned(),
                    entity_id: "orphan".to_owned(),
                    op: "TOMBSTONE".to_owned(),
                    payload_json: json!({"id":"orphan"}).to_string(),
                },
            )
            .unwrap();
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/python-graph-model-payloads-v1.json"
        ))
        .unwrap();
        let mut active_payload = fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|case| case["kind"] == "node")
            .unwrap()["payload"]
            .clone();
        active_payload["id"] = json!("stale-active");
        service
            .store
            .append_raw_entity_event(
                namespace,
                SqliteNewRawEntityEvent {
                    event_id: "stale-active-add".to_owned(),
                    entity_kind: "node".to_owned(),
                    entity_id: "stale-active".to_owned(),
                    op: "ADD".to_owned(),
                    payload_json: active_payload.to_string(),
                },
            )
            .unwrap();
        let mut stage1_only_payload = active_payload.clone();
        stage1_only_payload["id"] = json!("stage1-only-stale");
        service
            .store
            .append_raw_entity_event(
                namespace,
                SqliteNewRawEntityEvent {
                    event_id: "stage1-only-add".to_owned(),
                    entity_kind: "node".to_owned(),
                    entity_id: "stage1-only-stale".to_owned(),
                    op: "ADD".to_owned(),
                    payload_json: stage1_only_payload.to_string(),
                },
            )
            .unwrap();
        service
            .store
            .replace_stage1_node_projection(
                namespace,
                "node:stage1-only-stale",
                NamedProjectionWrite {
                    payload: json!({
                        "id":"stage1-only-stale",
                        "entity_kind":"node",
                        "source_fingerprint":"stale-fingerprint"
                    })
                    .as_object()
                    .unwrap()
                    .clone(),
                    last_authoritative_seq: 2,
                    last_materialized_seq: 2,
                    projection_schema_version: 2,
                    materialization_status: "pending".to_owned(),
                },
            )
            .unwrap();

        assert_eq!(
            ApplicationService::reconcile_eventual_projection_once(
                &service,
                namespace.to_owned(),
                2,
            )
            .await
            .unwrap(),
            3
        );
        let pending_jobs = service
            .store
            .list_index_jobs(Some(namespace), Some("PENDING"), None, None, None, 10)
            .unwrap()
            .len();
        assert_eq!(
            ApplicationService::reconcile_eventual_projection_once(
                &service,
                namespace.to_owned(),
                2,
            )
            .await
            .unwrap(),
            0
        );
        assert_eq!(
            service
                .store
                .list_index_jobs(Some(namespace), Some("PENDING"), None, None, None, 10)
                .unwrap()
                .len(),
            pending_jobs,
            "repeat reconciliation must not enqueue an already-staged current revision"
        );
        let recovered_stage1 = service
            .store
            .get_stage1_node_projection(namespace, "node:stage1-only-stale")
            .unwrap()
            .expect("active stale Stage-1 row is rebuilt without a Chroma row");
        assert_ne!(
            recovered_stage1
                .payload
                .get("source_fingerprint")
                .and_then(Value::as_str),
            Some("stale-fingerprint")
        );
        assert!(service
            .store
            .get_stage1_node_projection(namespace, "node:stale-active")
            .unwrap()
            .is_some());
        assert!(service
            .store
            .list_index_jobs(Some(namespace), Some("PENDING"), None, None, None, 10)
            .unwrap()
            .iter()
            .any(|job| job.entity_id == "stale-active" && job.op == "UPSERT"));
        assert!(service
            .store
            .list_index_jobs(Some(namespace), Some("PENDING"), None, None, None, 10)
            .unwrap()
            .iter()
            .any(|job| job.entity_id == "stage1-only-stale" && job.op == "UPSERT"));
        let requests = chroma.await.unwrap();
        let (delete_path, delete_body) = requests
            .iter()
            .find(|(line, _)| line.contains("/delete"))
            .expect("orphaned Stage-2 row is physically removed");
        assert!(delete_path.contains("reconcile-nodes"));
        assert_eq!(delete_body["ids"], json!(["orphan"]));
        assert_eq!(delete_body["where"]["$or"][0]["_kogwistar_namespace"]["$eq"], namespace);
        assert!(requests.iter().any(|(line, body)| {
            line.contains("/get") && body.get("offset").and_then(Value::as_u64) == Some(2)
        }));
    }

    #[tokio::test]
    async fn sqlite_chroma_projection_fence_replaces_stale_delete_after_resurrection() {
        use kogwistar_store::NamedProjectionWrite;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let race_store: Arc<Mutex<Option<kogwistar_store_sqlite::SqliteStore>>> =
            Arc::new(Mutex::new(None));
        let race_store_for_chroma = race_store.clone();
        let chroma = tokio::spawn(async move {
            let mut observed = Vec::new();
            for _ in 0..5 {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut bytes = Vec::new();
                let mut chunk = [0_u8; 4096];
                let (split, length) = loop {
                    let count = stream.read(&mut chunk).await.unwrap();
                    assert!(count > 0);
                    bytes.extend_from_slice(&chunk[..count]);
                    if let Some(split) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
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
                let head = String::from_utf8_lossy(&bytes[..split]);
                let request_line = head.lines().next().unwrap().to_owned();
                let body: Value = serde_json::from_slice(&bytes[split + 4..split + 4 + length])
                    .unwrap();
                if request_line.contains("/upsert") {
                    let store = race_store_for_chroma.lock().unwrap().clone().unwrap();
                    store
                        .append_raw_entity_event(
                            "fenced-tenant",
                            SqliteNewRawEntityEvent {
                                event_id: "resurrected-node-changed-during-upsert".to_owned(),
                                entity_kind: "node".to_owned(),
                                entity_id: "resurrected-node".to_owned(),
                                op: "REPLACE".to_owned(),
                                payload_json: json!({
                                    "id":"resurrected-node",
                                    "document":"newer source during Chroma write"
                                })
                                .to_string(),
                            },
                        )
                        .unwrap();
                    let latest = sqlite_canonical_entity_revision(
                        &store,
                        "fenced-tenant",
                        "node",
                        "resurrected-node",
                    )
                    .unwrap()
                    .unwrap();
                    let job_payload =
                        kogwistar_engine::canonical_revision::canonical_revision_job_payload(
                            &latest,
                        );
                    let fingerprint = serde_json::from_str::<Value>(&job_payload).unwrap()
                        ["source_fingerprint"]
                        .as_str()
                        .unwrap()
                        .to_owned();
                    store
                        .replace_stage1_node_projection(
                            "fenced-tenant",
                            "node:resurrected-node",
                            NamedProjectionWrite {
                                payload: json!({
                                    "document":"newer source during Chroma write",
                                    "metadata":{},
                                    "source_fingerprint":fingerprint
                                })
                                .as_object()
                                .unwrap()
                                .clone(),
                                last_authoritative_seq: latest.revision,
                                last_materialized_seq: latest.revision,
                                projection_schema_version: 1,
                                materialization_status: "ready".to_owned(),
                            },
                        )
                        .unwrap();
                }
                let response = if request_line.contains("/get") && body["ids"].is_array() {
                    json!({
                        "ids":body["ids"],
                        "documents":[null],
                        "metadatas":[{}]
                    })
                    .to_string()
                } else {
                    body["name"].as_str().map_or_else(
                        || json!({}).to_string(),
                        |name| {
                            json!({
                                "id": format!("collection-{name}"),
                                "metadata": {"hnsw:space":"cosine"}
                            })
                            .to_string()
                        },
                    )
                };
                let response = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                    response.len(),
                    response
                );
                stream.write_all(response.as_bytes()).await.unwrap();
                observed.push((request_line, body));
            }
            observed
        });

        let path = std::env::temp_dir().join(format!(
            "kogwistar-sqlite-chroma-stale-delete-{}.sqlite3",
            uuid::Uuid::new_v4()
        ));
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
        let service = open_sqlite_application_service(
            &path,
            Some(SqliteChromaGraphConfig {
                endpoint,
                metric: DistanceMetric::Cosine,
                node_collection: "fenced-nodes".to_owned(),
                edge_collection: "fenced-edges".to_owned(),
            }),
        )
        .await
        .unwrap()
        .with_embedding_provider(provider);
        *race_store.lock().unwrap() = Some(service.store.clone());

        let namespace = "fenced-tenant";
        let entity_id = "resurrected-node";
        let deleted = service
            .store
            .append_raw_entity_event(
                namespace,
                SqliteNewRawEntityEvent {
                    event_id: "resurrected-node-delete".to_owned(),
                    entity_kind: "node".to_owned(),
                    entity_id: entity_id.to_owned(),
                    op: "DELETE".to_owned(),
                    payload_json: json!({"id":entity_id}).to_string(),
                },
            )
            .unwrap();
        let deleted_revision = kogwistar_engine::canonical_revision::read_canonical_entity_revision(
            &[EntityEvent {
                namespace: deleted.event.namespace.clone(),
                seq: deleted.event.seq,
                event_id: deleted.event.event_id.clone(),
                entity_kind: deleted.event.entity_kind.clone(),
                entity_id: deleted.event.entity_id.clone(),
                op: deleted.event.op.clone(),
                payload: serde_json::from_str(&deleted.event.payload_json).unwrap(),
            }],
            namespace,
            "node",
            entity_id,
        )
        .unwrap()
        .unwrap();
        let stale_delete_payload =
            kogwistar_engine::canonical_revision::canonical_revision_job_payload(&deleted_revision);

        let active = service
            .store
            .append_raw_entity_event(
                namespace,
                SqliteNewRawEntityEvent {
                    event_id: "resurrected-node-reactivate".to_owned(),
                    entity_kind: "node".to_owned(),
                    entity_id: entity_id.to_owned(),
                    op: "REPLACE".to_owned(),
                    payload_json: json!({"id":entity_id,"document":"current source"}).to_string(),
                },
            )
            .unwrap();
        let active_revision = sqlite_canonical_entity_revision(
            &service.store,
            namespace,
            "node",
            entity_id,
        )
        .unwrap()
        .unwrap();
        let active_job_payload =
            kogwistar_engine::canonical_revision::canonical_revision_job_payload(&active_revision);
        let active_fingerprint = serde_json::from_str::<Value>(&active_job_payload).unwrap()
            ["source_fingerprint"]
            .as_str()
            .unwrap()
            .to_owned();
        service
            .store
            .replace_stage1_node_projection(
                namespace,
                "node:resurrected-node",
                NamedProjectionWrite {
                    payload: json!({
                        "document":"current source",
                        "metadata":{},
                        "source_fingerprint":active_fingerprint
                    })
                    .as_object()
                    .unwrap()
                    .clone(),
                    last_authoritative_seq: active.event.seq,
                    last_materialized_seq: active.event.seq,
                    projection_schema_version: 1,
                    materialization_status: "ready".to_owned(),
                },
            )
            .unwrap();
        service
            .store
            .enqueue_index_job(NewIndexJob {
                job_id: "stale-resurrection-delete".to_owned(),
                namespace: namespace.to_owned(),
                entity_kind: "node".to_owned(),
                entity_id: entity_id.to_owned(),
                index_kind: "node_embedding".to_owned(),
                op: "DELETE".to_owned(),
                payload_json: Some(stale_delete_payload.clone()),
                max_retries: 3,
            })
            .unwrap();

        let summary = service
            .run_embedding_projection_jobs_once(1, 30, Some(namespace))
            .await
            .unwrap();
        assert_eq!(summary.claimed, 1);
        assert_eq!(summary.projected, 0);
        assert!(calls.lock().unwrap().is_empty(), "DELETE must not invoke embeddings");
        let jobs = service
            .store
            .list_index_jobs(Some(namespace), None, Some("node"), Some(entity_id), Some("node_embedding"), 10)
            .unwrap();
        assert_eq!(
            jobs.iter()
                .find(|job| job.job_id == "stale-resurrection-delete")
                .unwrap()
                .status,
            "DONE"
        );
        let replacement = jobs
            .iter()
            .find(|job| job.job_id != "stale-resurrection-delete" && job.status == "PENDING")
            .expect("the active revision must be requeued before the stale delete is acknowledged");
        assert_eq!(replacement.op, "UPSERT");
        assert_eq!(replacement.payload_json.as_deref(), Some(active_job_payload.as_str()));
        let fenced = RevisionFencedChromaProjectionStore {
            queue: service.store.clone(),
            graph: service.chroma_graph.as_ref().unwrap().clone(),
        };
        let deleted_fingerprint = serde_json::from_str::<Value>(&stale_delete_payload).unwrap()
            ["source_fingerprint"]
            .as_str()
            .unwrap()
            .to_owned();
        let stale_upsert = kogwistar_store::GraphProjectionWriteStore::upsert_graph_projection_record_with_outcome(
            &fenced,
            namespace,
            "node",
            GraphRecord {
                id: entity_id.to_owned(),
                document: Some("stale tombstoned content".to_owned()),
                metadata: serde_json::from_value(json!({
                    "_kogwistar_stage2_ready":true,
                    "_kogwistar_source_revision":deleted.event.seq,
                    "_kogwistar_source_fingerprint":deleted_fingerprint
                }))
                .unwrap(),
                embedding: Some(vec![0.0, 1.0]),
            },
        )
        .await
        .unwrap();
        assert_eq!(
            stale_upsert,
            kogwistar_store::ProjectionWriteOutcome::Superseded
        );
        let pending = service
            .store
            .list_index_jobs(
                Some(namespace),
                Some("PENDING"),
                Some("node"),
                Some(entity_id),
                Some("node_embedding"),
                10,
            )
            .unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].op, "UPSERT");
        assert_eq!(pending[0].payload_json.as_deref(), Some(active_job_payload.as_str()));

        let active_fingerprint = serde_json::from_str::<Value>(&active_job_payload).unwrap()
            ["source_fingerprint"]
            .as_str()
            .unwrap()
            .to_owned();
        let raced_upsert = kogwistar_store::GraphProjectionWriteStore::upsert_graph_projection_record_with_outcome(
            &fenced,
            namespace,
            "node",
            GraphRecord {
                id: entity_id.to_owned(),
                document: Some("current before Chroma race".to_owned()),
                metadata: serde_json::from_value(json!({
                    "_kogwistar_stage2_ready":true,
                    "_kogwistar_source_revision":active.event.seq,
                    "_kogwistar_source_fingerprint":active_fingerprint
                }))
                .unwrap(),
                embedding: Some(vec![0.6, 0.8]),
            },
        )
        .await
        .unwrap();
        assert_eq!(
            raced_upsert,
            kogwistar_store::ProjectionWriteOutcome::Superseded
        );
        let latest = sqlite_canonical_entity_revision(
            &service.store,
            namespace,
            "node",
            entity_id,
        )
        .unwrap()
        .unwrap();
        let latest_payload =
            kogwistar_engine::canonical_revision::canonical_revision_job_payload(&latest);
        let pending = service
            .store
            .list_index_jobs(
                Some(namespace),
                Some("PENDING"),
                Some("node"),
                Some(entity_id),
                Some("node_embedding"),
                10,
            )
            .unwrap();
        assert_eq!(pending.len(), 1, "race recovery should coalesce into one job");
        assert_eq!(pending[0].payload_json.as_deref(), Some(latest_payload.as_str()));

        service
            .store
            .append_raw_entity_event(
                namespace,
                SqliteNewRawEntityEvent {
                    event_id: "resurrected-node-delete-after-race".to_owned(),
                    entity_kind: "node".to_owned(),
                    entity_id: entity_id.to_owned(),
                    op: "DELETE".to_owned(),
                    payload_json: json!({"id":entity_id}).to_string(),
                },
            )
            .unwrap();
        let deleted_outcome = kogwistar_store::GraphProjectionWriteStore::delete_graph_projection_record_with_outcome(
            &fenced,
            namespace,
            "node",
            entity_id,
        )
        .await
        .unwrap();
        assert_eq!(
            deleted_outcome,
            kogwistar_store::ProjectionWriteOutcome::Applied
        );
        assert!(service
            .store
            .get_stage1_node_projection(namespace, &format!("node:{entity_id}"))
            .unwrap()
            .is_none());
        let requests = chroma.await.unwrap();
        assert_eq!(requests.len(), 5);
        assert!(
            requests.iter().any(|(line, _)| line.contains("/upsert")),
            "the simulated race must occur during a real Chroma mutation"
        );
        assert!(requests.iter().any(|(line, _)| line.contains("/delete")));

        drop(service);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    #[ignore = "requires KOGWISTAR_TEST_PG_DSN; run explicitly against disposable PostgreSQL"]
    async fn postgres_embedding_worker_uses_configured_provider_and_leaves_other_jobs_pending() {
        let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
            .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
        let schema = format!("rust_embedding_worker_{}", uuid::Uuid::new_v4().simple());
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
        let service = PostgresRunApplicationService::from_dsn(&dsn, &schema)
            .unwrap()
            .with_embedding_provider(provider.clone());
        service.ensure_schema().await.unwrap();
        service
            .store
            .create_graph_schema(2, kogwistar_store_postgres::GraphTableNames::default())
            .await
            .unwrap();

        let namespace = format!("embedding-worker-{}", uuid::Uuid::new_v4().simple());
        let node_id = "provider-worker-node";
        service
            .store
            .apply_graph_mutation(GraphMutation {
                scope: GraphScope {
                    namespace: namespace.clone(),
                    workspace_id: None,
                    graph_space: None,
                },
                table: "gke_nodes".to_owned(),
                entity_kind: "node".to_owned(),
                event_id: format!("{node_id}-add"),
                op: "ADD".to_owned(),
                payload: json!({"id": node_id}),
                record: GraphRecord {
                    id: node_id.to_owned(),
                    document: Some("canonical text for provider".to_owned()),
                    metadata: Default::default(),
                    embedding: None,
                },
                embedding_dim: 2,
            })
            .await
            .unwrap();
        let edge_id = "provider-worker-edge";
        let edge_document = json!({
            "id": edge_id,
            "source_ids": ["source"],
            "target_ids": ["target"]
        })
        .to_string();
        service
            .store
            .apply_graph_mutation(GraphMutation {
                scope: GraphScope {
                    namespace: namespace.clone(),
                    workspace_id: None,
                    graph_space: None,
                },
                table: "gke_edges".to_owned(),
                entity_kind: "edge".to_owned(),
                event_id: format!("{edge_id}-add"),
                op: "ADD".to_owned(),
                payload: json!({"id": edge_id, "source_ids": ["source"], "target_ids": ["target"]}),
                record: GraphRecord {
                    id: edge_id.to_owned(),
                    document: Some(edge_document.clone()),
                    metadata: Default::default(),
                    embedding: None,
                },
                embedding_dim: 2,
            })
            .await
            .unwrap();
        for (job_id, entity_kind, entity_id, index_kind) in [
            ("provider-worker-node-embedding", "node", node_id, "node_embedding"),
            ("provider-worker-edge-embedding", "edge", edge_id, "node_embedding"),
            ("provider-worker-refs", "node", node_id, "node_refs"),
        ] {
            service
                .store
                .enqueue_index_job(NewIndexJob {
                    job_id: format!("{job_id}-{}", uuid::Uuid::new_v4().simple()),
                    namespace: namespace.clone(),
                    entity_kind: entity_kind.to_owned(),
                    entity_id: entity_id.to_owned(),
                    index_kind: index_kind.to_owned(),
                    op: "UPSERT".to_owned(),
                    payload_json: None,
                    max_retries: 3,
                })
                .await
                .unwrap();
        }

        let cleanup_dsn = dsn
            .strip_prefix("postgresql+psycopg://")
            .map(|rest| format!("postgresql://{rest}"))
            .or_else(|| {
                dsn.strip_prefix("postgresql+psycopg2://")
                    .map(|rest| format!("postgresql://{rest}"))
            })
            .unwrap_or_else(|| dsn.clone());
        let (admin, connection) = tokio_postgres::connect(&cleanup_dsn, tokio_postgres::NoTls)
            .await
            .unwrap();
        let connection_task = tokio::spawn(connection);
        admin
            .batch_execute(&format!(
                "CREATE FUNCTION \"{schema}\".fail_embedding_projection() RETURNS trigger \
                 LANGUAGE plpgsql AS $$ BEGIN \
                 IF NEW.id = '{node_id}' AND NEW.embedding IS NOT NULL THEN \
                 RAISE EXCEPTION 'injected projection failure'; END IF; \
                 RETURN NEW; END $$; \
                 CREATE TRIGGER fail_embedding_projection BEFORE UPDATE ON \"{schema}\".gke_nodes \
                 FOR EACH ROW EXECUTE FUNCTION \"{schema}\".fail_embedding_projection();"
            ))
            .await
            .unwrap();

        let failed_projection = service
            .run_embedding_projection_jobs_once(4, 30, Some(&namespace))
            .await
            .unwrap();
        assert_eq!(failed_projection.claimed, 2);
        assert_eq!(failed_projection.projected, 1);
        assert_eq!(failed_projection.retried, 1);
        let mut provider_texts = calls
            .lock()
            .unwrap()
            .iter()
            .flatten()
            .cloned()
            .collect::<Vec<_>>();
        provider_texts.sort();
        assert_eq!(
            provider_texts,
            vec![
                "canonical text for provider".to_owned(),
                edge_document.clone()
            ]
        );

        let retryable = service
            .store
            .list_index_jobs(Some(&namespace), Some("PENDING"), None, None, None, 10)
            .await
            .unwrap();
        let embedding_job = retryable
            .iter()
            .find(|job| job.index_kind == "node_embedding")
            .expect("failed embedding projection remains pending");
        assert!(embedding_job.accepted_result_json.is_some());
        assert!(embedding_job.accepted_result_sha256.is_some());
        assert!(retryable.iter().any(|job| job.index_kind == "node_refs"));

        drop(service);
        let service = PostgresRunApplicationService::from_dsn(&dsn, &schema)
            .unwrap()
            .with_embedding_provider(provider);
        service.ensure_schema().await.unwrap();

        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        admin
            .batch_execute(&format!(
                "DROP TRIGGER fail_embedding_projection ON \"{schema}\".gke_nodes; \
                 DROP FUNCTION \"{schema}\".fail_embedding_projection();"
            ))
            .await
            .unwrap();
        let (worker_shutdown, worker_shutdown_rx) = tokio::sync::oneshot::channel();
        let worker = tokio::spawn(run_inline_embedding_worker(
            Arc::new(service.clone()),
            worker_shutdown_rx,
            true,
            std::time::Duration::ZERO,
            String::new(),
        ));
        let jobs = tokio::time::timeout(std::time::Duration::from_secs(10), async {
            loop {
                let jobs = service
                    .store
                    .list_index_jobs(Some(&namespace), None, None, None, None, 10)
                    .await
                    .unwrap();
                if jobs.iter().filter(|job| job.status == "DONE").count() == 2 {
                    break jobs;
                }
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }
        })
        .await
        .expect("inline worker completes the recovered embedding job");
        let _ = worker_shutdown.send(());
        worker.await.unwrap();
        let mut provider_texts = calls
            .lock()
            .unwrap()
            .iter()
            .flatten()
            .cloned()
            .collect::<Vec<_>>();
        provider_texts.sort();
        assert_eq!(
            provider_texts,
            vec![
                "canonical text for provider".to_owned(),
                edge_document.clone()
            ]
        );

        let projection = service
            .store
            .graph_projection_records(GraphProjectionRead {
                scope: GraphScope {
                    namespace: namespace.clone(),
                    workspace_id: None,
                    graph_space: None,
                },
                table: "gke_nodes".to_owned(),
                ids: Some(vec![node_id.to_owned()]),
                metadata: MetadataFilter::default(),
                limit: 1,
            })
            .await
            .unwrap();
        assert_eq!(projection[0].embedding, Some(vec![0.6, 0.8]));
        let edge_projection = service
            .store
            .graph_projection_records(GraphProjectionRead {
                scope: GraphScope {
                    namespace: namespace.clone(),
                    workspace_id: None,
                    graph_space: None,
                },
                table: "gke_edges".to_owned(),
                ids: Some(vec![edge_id.to_owned()]),
                metadata: MetadataFilter::default(),
                limit: 1,
            })
            .await
            .unwrap();
        assert_eq!(edge_projection[0].embedding, Some(vec![0.6, 0.8]));
        assert_eq!(jobs.iter().filter(|job| job.status == "DONE").count(), 2);
        assert_eq!(jobs.iter().filter(|job| job.status == "PENDING").count(), 1);

        drop(service);
        admin
            .batch_execute(&format!("DROP SCHEMA \"{schema}\" CASCADE"))
            .await
            .unwrap();
        drop(admin);
        connection_task.await.unwrap().unwrap();
    }

    #[test]
    fn rust_server_embedding_configuration_is_opt_in_and_fails_closed() {
        assert!(!parse_inline_worker_flag("KOGWISTAR_EMBEDDING_WORKER_INLINE", None).unwrap());
        assert!(!parse_inline_worker_flag("KOGWISTAR_EMBEDDING_WORKER_INLINE", Some("off"))
            .unwrap());
        assert!(parse_inline_worker_flag("KOGWISTAR_EMBEDDING_WORKER_INLINE", Some(" YES "))
            .unwrap());
        assert!(parse_inline_worker_flag("KOGWISTAR_EMBEDDING_WORKER_INLINE", Some("maybe"))
            .is_err());
        assert!(embedding_provider_from_settings(None, None, None, None)
            .unwrap()
            .is_none());
        assert!(embedding_provider_from_settings(Some("  "), None, None, None)
            .unwrap()
            .is_none());
        assert!(embedding_provider_from_settings(Some("openai"), Some("m"), Some("3"), None)
            .is_err());
        assert!(embedding_provider_from_settings(Some("ollama"), None, Some("3"), None)
            .is_err());
        assert!(embedding_provider_from_settings(Some("ollama"), Some("m"), Some("0"), None)
            .is_err());
        assert!(embedding_provider_from_settings(Some("ollama"), Some("m"), Some("x"), None)
            .is_err());

        let provider = embedding_provider_from_settings(
            Some("Ollama"),
            Some("nomic-embed-text"),
            Some("768"),
            Some("http://ollama:11434"),
        )
        .unwrap()
        .unwrap();
        assert_eq!(provider.profile().provider, "ollama");
        assert_eq!(provider.profile().model, "nomic-embed-text");
        assert_eq!(provider.profile().dimension, 768);
        assert!(embedding_provider_from_settings(
            Some("ollama"),
            Some("nomic-embed-text"),
            Some("768"),
            Some("not-a-url"),
        )
        .is_err());
    }

    #[test]
    fn rust_postgres_embedding_scope_matches_python_physical_scope() {
        let config: tokio_postgres::Config =
            "postgresql://writer:secret@EXAMPLE.com:5544/dbname".parse().unwrap();
        assert_eq!(
            rust_pgvector_embedding_scope(&config, "tenant").unwrap(),
            "pgvector:5eafd1a9e29e59fdabd83f04e36ed6ae"
        );
    }

    #[test]
    fn registered_embedding_profile_matches_by_full_fingerprint() {
        let profile: kogwistar_engine::embedding::EmbeddingProfile = serde_json::from_str(
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../../contracts/golden/embedding-profile-v1.json"
            )),
        )
        .unwrap();
        let payload = json!({
            "profile_schema_version": 1,
            "embedding_profile": profile.as_value(),
            "embedding_fingerprint": profile.fingerprint(),
            "legacy_adopted": false,
        })
        .as_object()
        .cloned()
        .unwrap();
        let projection = NamedProjection {
            namespace: EMBEDDING_PROFILE_NAMESPACE.to_owned(),
            key: "pgvector:fixture".to_owned(),
            payload,
            last_authoritative_seq: 0,
            last_materialized_seq: 0,
            projection_schema_version: 1,
            materialization_status: "bound".to_owned(),
            updated_at_ms: 0,
        };
        validate_embedding_profile_projection(&projection, &profile, "pgvector:fixture").unwrap();

        let mut changed_profile = profile;
        changed_profile.model = "different-model-same-dimension".to_owned();
        assert!(validate_embedding_profile_projection(
            &projection,
            &changed_profile,
            "pgvector:fixture"
        )
        .is_err());
    }
