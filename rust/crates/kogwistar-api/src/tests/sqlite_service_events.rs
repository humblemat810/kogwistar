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
        let contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/workflow-service-events-http-v1.json"
        ))
        .unwrap();
        assert_eq!(
            response.status().as_u16(),
            contract["success_status"].as_u64().unwrap() as u16
        );
        let body =
            serde_json::from_slice::<Value>(&to_bytes(response.into_body(), 16_384).await.unwrap())
                .unwrap();
        for field in contract["required_envelope_fields"].as_array().unwrap() {
            let field = field.as_str().unwrap();
            assert!(body.get(field).is_some(), "missing {field}");
        }
        assert_eq!(body["service_id"], "svc-a");
        assert_eq!(body["events"].as_array().unwrap().len(), 1);
        for field in contract["required_event_fields"].as_array().unwrap() {
            let field = field.as_str().unwrap();
            assert!(body["events"][0].get(field).is_some(), "missing {field}");
        }
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
    async fn sqlite_service_declare_matches_shared_http_contract() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-service-declare-{nonce}.sqlite3"));
        let service = SqliteRunApplicationService::open(&path).unwrap();
        let contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/workflow-service-declare-http-v1.json"
        ))
        .unwrap();
        let response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/workflow/services".to_owned(),
                body: serde_json::to_vec(&contract["request"]).unwrap(),
                principal: json!({
                    "sub": "operator",
                    "role": "rw",
                    "ns": "workflow",
                    "capabilities": ["service.manage"],
                }),
            })
            .await;
        assert_eq!(
            response.status,
            contract["http_status"].as_u64().unwrap() as u16
        );
        let payload: Value = serde_json::from_slice(&response.body).unwrap();
        for (field, expected) in contract["expected_body_fields"].as_object().unwrap() {
            assert_eq!(&payload[field], expected);
        }
        drop(service);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn sqlite_service_get_rebuilds_missing_projection_from_history() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-service-get-{nonce}.sqlite3"));
        let service = SqliteRunApplicationService::open(&path).unwrap();
        let contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/workflow-service-get-http-v1.json"
        ))
        .unwrap();
        let service_id = contract["request"]["service_id"].as_str().unwrap();
        let declared = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/workflow/services".to_owned(),
                body: serde_json::to_vec(&contract["request"]).unwrap(),
                principal: json!({
                    "sub": "operator",
                    "role": "rw",
                    "ns": "workflow",
                    "capabilities": ["service.manage", "service.inspect", "project_view"],
                }),
            })
            .await;
        assert_eq!(declared.status, contract["http_status"].as_u64().unwrap() as u16);
        let declared: Value = serde_json::from_slice(&declared.body).unwrap();
        SqliteStore::open(&path)
            .unwrap()
            .clear_named_projection("service_registry", service_id)
            .unwrap();
        let response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: format!("/api/workflow/services/{service_id}"),
                body: Vec::new(),
                principal: json!({
                    "sub": "operator",
                    "role": "rw",
                    "ns": "workflow",
                    "capabilities": ["service.inspect", "project_view"],
                }),
            })
            .await;
        assert_eq!(response.status, contract["http_status"].as_u64().unwrap() as u16);
        let payload: Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(payload, declared);
        for (field, expected) in contract["expected_body_fields"].as_object().unwrap() {
            assert_eq!(&payload[field], expected);
        }
        assert!(SqliteStore::open(&path)
            .unwrap()
            .get_named_projection("service_registry", service_id)
            .unwrap()
            .is_some());

        drop(service);
        let _ = std::fs::remove_file(path);
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
            "capabilities": ["service.manage", "service.inspect", "project_view", "read_security_scope"],
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
        let trigger_contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/workflow-service-trigger-http-v1.json"
        ))
        .unwrap();
        let triggered = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: trigger_contract["method"].as_str().unwrap().to_owned(),
                path_and_query: trigger_contract["path"]
                    .as_str()
                    .unwrap()
                    .replace("{service_id}", "svc-declared"),
                body: serde_json::to_vec(&trigger_contract["request_body"]).unwrap(),
                principal: principal.clone(),
            })
            .await;
        assert_eq!(
            triggered.status,
            trigger_contract["success_status"].as_u64().unwrap() as u16
        );
        let triggered = serde_json::from_slice::<Value>(&triggered.body).unwrap();
        for (field, expected) in trigger_contract["expected_response_fields"]
            .as_object()
            .unwrap()
        {
            assert_eq!(&triggered[field], expected);
        }
        for field in trigger_contract["required_response_fields"].as_array().unwrap() {
            let field = field.as_str().unwrap();
            assert!(triggered.get(field).is_some(), "missing {field}");
        }
        for field in trigger_contract["dynamic_snapshot_fields"].as_array().unwrap() {
            let field = field.as_str().unwrap();
            assert!(triggered[field].as_str().is_some(), "missing string {field}");
        }
        assert_eq!(triggered["service_id"], "svc-declared");
        assert!(triggered["last_triggered_at_ms"].as_i64().is_some());
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
        let repair_contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/workflow-service-repair-http-v1.json"
        ))
        .unwrap();
        let repaired = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: repair_contract["method"].as_str().unwrap().to_owned(),
                path_and_query: repair_contract["path"]
                    .as_str()
                    .unwrap()
                    .replace("{service_id}", "svc-repair"),
                body: Vec::new(),
                principal: principal.clone(),
            })
            .await;
        assert_eq!(
            repaired.status,
            repair_contract["success_status"].as_u64().unwrap() as u16
        );
        let body = serde_json::from_slice::<Value>(&repaired.body).unwrap();
        for field in repair_contract["required_body_fields"].as_array().unwrap() {
            let field = field.as_str().unwrap();
            assert!(body.get(field).is_some(), "missing {field}");
        }
        assert_eq!(body["service_id"], "svc-repair");
        for field in repair_contract["required_projection_fields"].as_array().unwrap() {
            let field = field.as_str().unwrap();
            assert!(body["projection"].get(field).is_some(), "missing projection.{field}");
        }
        assert_eq!(body["projection"]["service_id"], "svc-repair");
        assert_eq!(body["projection"]["enabled"], false);
        assert_eq!(body["projection"]["lifecycle_status"], "stopped");
        store
            .clear_named_projection("service_registry", "svc-repair")
            .unwrap();

        // PYTHON_PARITY_V1: bulk HTTP response uses ChatRunService.repair_service_projections.
        let bulk_repair_contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/workflow-services-repair-http-v1.json"
        ))
        .unwrap();
        let bulk_repaired = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: bulk_repair_contract["method"].as_str().unwrap().to_owned(),
                path_and_query: bulk_repair_contract["path"].as_str().unwrap().to_owned(),
                body: Vec::new(),
                principal: principal.clone(),
            })
            .await;
        assert_eq!(
            bulk_repaired.status,
            bulk_repair_contract["success_status"].as_u64().unwrap() as u16
        );
        let bulk_body: Value = serde_json::from_slice(&bulk_repaired.body).unwrap();
        for field in bulk_repair_contract["required_body_fields"].as_array().unwrap() {
            let field = field.as_str().unwrap();
            assert!(bulk_body.get(field).is_some(), "missing {field}");
        }
        assert_eq!(
            bulk_body["repaired_service_ids"].as_array().unwrap().len(),
            bulk_repair_contract["expected_repaired_service_count"]
                .as_u64()
                .unwrap() as usize
        );
        assert!(bulk_body["repaired_service_ids"]
            .as_array()
            .unwrap()
            .iter()
            .any(|service_id| service_id == "svc-declared"));

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
        let toggle_contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/workflow-service-toggle-http-v1.json"
        ))
        .unwrap();
        let disabled = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: toggle_contract["disable"]["path"]
                    .as_str()
                    .unwrap()
                    .to_owned(),
                body: Vec::new(),
                principal: principal.clone(),
            })
            .await;
        assert_eq!(
            disabled.status,
            toggle_contract["http_status"].as_u64().unwrap() as u16
        );
        let disabled = serde_json::from_slice::<Value>(&disabled.body).unwrap();
        for (field, expected) in toggle_contract["disable"]["expected_fields"]
            .as_object()
            .unwrap()
        {
            assert_eq!(&disabled[field], expected);
        }
        let enabled = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: toggle_contract["enable"]["path"]
                    .as_str()
                    .unwrap()
                    .to_owned(),
                body: Vec::new(),
                principal,
            })
            .await;
        assert_eq!(
            enabled.status,
            toggle_contract["http_status"].as_u64().unwrap() as u16
        );
        let enabled = serde_json::from_slice::<Value>(&enabled.body).unwrap();
        for (field, expected) in toggle_contract["enable"]["expected_fields"]
            .as_object()
            .unwrap()
        {
            assert_eq!(&enabled[field], expected);
        }
        assert_eq!(enabled["lifecycle_status"], "degraded");
        assert_eq!(enabled["health_status"], "degraded");
        let heartbeat_contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/workflow-service-heartbeat-http-v1.json"
        ))
        .unwrap();
        let heartbeat = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: heartbeat_contract["method"].as_str().unwrap().to_owned(),
                path_and_query: heartbeat_contract["path"]
                    .as_str()
                    .unwrap()
                    .replace("{service_id}", "svc-repair"),
                body: serde_json::to_vec(&heartbeat_contract["request_body"]).unwrap(),
                principal: json!({
                    "sub": "worker",
                    "role": "rw",
                    "capabilities": ["service.heartbeat", "service.manage"],
                }),
            })
            .await;
        assert_eq!(
            heartbeat.status,
            heartbeat_contract["success_status"].as_u64().unwrap() as u16
        );
        let heartbeat = serde_json::from_slice::<Value>(&heartbeat.body).unwrap();
        for (field, expected) in heartbeat_contract["expected_response_fields"]
            .as_object()
            .unwrap()
        {
            assert_eq!(&heartbeat[field], expected);
        }
        for field in heartbeat_contract["required_timestamp_fields"]
            .as_array()
            .unwrap()
        {
            let field = field.as_str().unwrap();
            assert!(heartbeat[field].as_i64().is_some(), "missing integer {field}");
        }
        let history = store.replay_raw_events("workflow", 0, usize::MAX).unwrap();
        assert!(history.iter().any(|event| {
            event.payload_json.contains("service.enabled")
                && event.entity_id.starts_with("service_evt:svc-repair:")
        }));
        assert!(history.iter().any(|event| {
            event.payload_json.contains("service.degraded")
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
