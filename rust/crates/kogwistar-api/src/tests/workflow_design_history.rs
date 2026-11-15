    #[tokio::test]
    async fn sqlite_design_history_folds_branch_and_repairs_projection() {
        use axum::body::{Body, to_bytes};
        use kogwistar_store_sqlite::NewRawEntityEvent;
        use std::time::{SystemTime, UNIX_EPOCH};
        use tower::ServiceExt;

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-design-history-{nonce}.sqlite3"));
        let store = SqliteStore::open(&path).unwrap();
        let namespace = "wf_design:wf-history";
        let append = |event_id: &str, op: &str, payload: Value| {
            store
                .append_raw_entity_event(
                    namespace,
                    NewRawEntityEvent {
                        event_id: event_id.to_owned(),
                        entity_kind: "design_control".to_owned(),
                        entity_id: "wf-history".to_owned(),
                        op: op.to_owned(),
                        payload_json: payload.to_string(),
                    },
                )
                .unwrap();
        };
        append(
            "commit-1",
            "MUTATION_COMMITTED",
            json!({"version": 1, "prev_version": 0, "target_seq": 1, "ts_ms": 10, "designer_id": "alice", "action": "node_upsert", "entity_id": "n1"}),
        );
        append(
            "commit-2",
            "MUTATION_COMMITTED",
            json!({"version": 2, "prev_version": 1, "target_seq": 2, "ts_ms": 20, "designer_id": "alice", "action": "node_upsert", "entity_id": "n2"}),
        );
        append(
            "undo-1",
            "UNDO_APPLIED",
            json!({"from_version": 2, "to_version": 1, "ts_ms": 30, "designer_id": "alice"}),
        );
        append(
            "drop-2",
            "BRANCH_DROPPED",
            json!({"drop_from_version": 2, "drop_to_version": 2, "drop_from_seq": 2, "drop_to_seq": 2, "ts_ms": 40, "designer_id": "alice"}),
        );
        append(
            "commit-3",
            "MUTATION_COMMITTED",
            json!({"version": 3, "prev_version": 1, "target_seq": 5, "ts_ms": 50, "designer_id": "alice", "action": "edge_upsert", "entity_id": "e3"}),
        );
        store
            .replace_named_projection(
                "workflow_design",
                "wf-history",
                NamedProjectionWrite {
                    payload: json!({"current_version": 999}).as_object().unwrap().clone(),
                    last_authoritative_seq: 5,
                    last_materialized_seq: 999,
                    projection_schema_version: 1,
                    materialization_status: "ready".to_owned(),
                },
            )
            .unwrap();
        let service = SqliteRunApplicationService::open(&path).unwrap();
        let response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: "/api/workflow/design/wf-history/history".to_owned(),
                body: Vec::new(),
                principal: json!({
                    "sub": "alice",
                    "role": "ro",
                    "ns": "workflow",
                    "capabilities": ["workflow.design.inspect"],
                }),
            })
            .await;
        assert_eq!(response.status, StatusCode::OK.as_u16());
        let history = serde_json::from_slice::<Value>(&response.body).unwrap();
        assert_eq!(history["current_version"], 3);
        assert_eq!(history["active_tip_version"], 3);
        assert_eq!(history["allocated_max_version"], 3);
        assert_eq!(history["current_seq"], 5);
        assert_eq!(history["can_undo"], true);
        assert_eq!(history["can_redo"], false);
        assert_eq!(
            history["versions"]
                .as_array()
                .unwrap()
                .iter()
                .map(|item| item["version"].as_i64().unwrap())
                .collect::<Vec<_>>(),
            [0, 1, 3]
        );
        assert_eq!(history["selected_versions"].as_array().unwrap().len(), 3);
        assert_eq!(history["dropped_ranges"][0]["start_version"], 2);
        assert_eq!(history["timeline"].as_array().unwrap().len(), 5);
        let repaired = store
            .get_named_projection("workflow_design", "wf-history")
            .unwrap()
            .unwrap();
        assert_eq!(repaired.payload["current_version"], 3);
        assert_eq!(repaired.payload["active_tip_version"], 3);
        assert_eq!(repaired.last_authoritative_seq, 5);
        assert_eq!(repaired.last_materialized_seq, 5);
        assert_eq!(repaired.materialization_status, "rust_event_only");
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: vec!["reader".to_owned()],
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig::default(),
            },
            Arc::new(service.clone()),
        );
        let mcp = app
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/mcp/workflow")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":"history","method":"tools/call","params":{"name":"workflow.design_history","arguments":{"workflow_id":"wf-history"}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(mcp.status(), StatusCode::OK);
        let mcp_body = to_bytes(mcp.into_body(), 65_536).await.unwrap();
        let mcp_value = serde_json::from_slice::<Value>(&mcp_body).unwrap();
        assert_eq!(mcp_value["result"]["current_version"], 3);
        drop(service);
        drop(store);
        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn sqlite_design_capability_mutates_graph_and_undoes_atomically() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
        use tower::ServiceExt;

        let path = std::env::temp_dir().join(format!(
            "kogwistar-design-write-{}.sqlite3",
            uuid::Uuid::new_v4()
        ));
        let edge_delete_contract: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/workflow-design-edge-delete-http-v1.json"
        )))
        .unwrap();
        let service = SqliteRunApplicationService::open(&path).unwrap();
        let store = SqliteStore::open(&path).unwrap();
        let key = "design-secret";
        let auth = AuthConfig {
            mode: "jwt".to_owned(),
            algorithm: Some("HS256".to_owned()),
            key: Some(key.to_owned()),
            ..AuthConfig::default()
        };
        let claims = json!({"sub":"alice","role":"rw","roles":["reader","rw"],"ns":"workflow","capabilities":["spawn_process","workflow.run.write","workflow.design.write","workflow.design.inspect","approve_action","write_graph"],"exp":4102444800_u64});
        let token = encode(
            &Header::new(Algorithm::HS256),
            &claims,
            &EncodingKey::from_secret(key.as_bytes()),
        )
        .unwrap();
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: vec!["reader".to_owned()],
                implementation: ImplementationSnapshot::default(),
                auth,
            },
            Arc::new(service.clone()),
        );
        macro_rules! public_call {
            ($method:expr, $path:expr, $body:expr) => {{
                app.clone()
                    .oneshot(
                        Request::builder()
                            .method($method)
                            .uri($path)
                            .header("content-type", "application/json")
                            .header("authorization", format!("Bearer {token}"))
                            .body(Body::from(serde_json::to_vec(&$body).unwrap()))
                            .unwrap(),
                    )
                    .await
                    .unwrap()
            }};
        }
        let empty_graph_contract: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/workflow-design-empty-graph-http-v1.json"
        )))
        .unwrap();
        let empty_graph_response = public_call!(
            "GET",
            "/api/workflow/design/wf-empty-graph/graph",
            json!({})
        );
        assert_eq!(
            empty_graph_response.status().as_u16(),
            empty_graph_contract["success_status"]
                .as_u64()
                .unwrap() as u16
        );
        let empty_graph: Value = serde_json::from_slice(
            &to_bytes(empty_graph_response.into_body(), 32_768)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(empty_graph, empty_graph_contract["response_body"]);
        let empty_history_contract: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/workflow-design-empty-history-http-v1.json"
        )))
        .unwrap();
        let empty_history_response = public_call!(
            "GET",
            "/api/workflow/design/wf-empty-history/history",
            json!({})
        );
        assert_eq!(
            empty_history_response.status().as_u16(),
            empty_history_contract["success_status"]
                .as_u64()
                .unwrap() as u16
        );
        let empty_history: Value = serde_json::from_slice(
            &to_bytes(empty_history_response.into_body(), 32_768)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(empty_history, empty_history_contract["response_body"]);
        let node_upsert_contract: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/workflow-design-node-upsert-http-v1.json"
        )))
        .unwrap();
        let node_upsert_response = public_call!(
            "POST",
            "/api/workflow/design/wf-node-contract/nodes",
            json!({
                "designer_id":"alice",
                "node_id":"node-contract",
                "label":"Start",
                "op":"start",
                "start":true
            })
        );
        assert_eq!(
            node_upsert_response.status().as_u16(),
            node_upsert_contract["success_status"]
                .as_u64()
                .unwrap() as u16
        );
        let node_upsert: Value = serde_json::from_slice(
            &to_bytes(node_upsert_response.into_body(), 32_768)
                .await
                .unwrap(),
        )
        .unwrap();
        for (field, expected) in node_upsert_contract["fixed_response_fields"]
            .as_object()
            .unwrap()
        {
            assert_eq!(&node_upsert[field], expected, "field {field}");
        }
        for field in node_upsert_contract["integer_response_fields"]
            .as_array()
            .unwrap()
        {
            let field = field.as_str().unwrap();
            assert!(node_upsert[field].as_i64().is_some(), "field {field}");
        }
        for field in node_upsert_contract["absent_response_fields"]
            .as_array()
            .unwrap()
        {
            let field = field.as_str().unwrap();
            assert!(node_upsert.get(field).is_none(), "field {field}");
        }
        let node_delete_contract: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/workflow-design-node-delete-http-v1.json"
        )))
        .unwrap();
        let node_delete_create = public_call!(
            "POST",
            "/api/workflow/design/wf-node-delete-contract/nodes",
            json!({
                "designer_id":"alice",
                "node_id":"node-delete-contract",
                "label":"Start",
                "op":"start",
                "start":true
            })
        );
        assert_eq!(node_delete_create.status(), StatusCode::OK);
        let node_delete_response = public_call!(
            "DELETE",
            "/api/workflow/design/wf-node-delete-contract/nodes/node-delete-contract",
            json!({"designer_id":"alice"})
        );
        assert_eq!(
            node_delete_response.status().as_u16(),
            node_delete_contract["success_status"]
                .as_u64()
                .unwrap() as u16
        );
        let node_delete: Value = serde_json::from_slice(
            &to_bytes(node_delete_response.into_body(), 32_768)
                .await
                .unwrap(),
        )
        .unwrap();
        for (field, expected) in node_delete_contract["fixed_response_fields"]
            .as_object()
            .unwrap()
        {
            assert_eq!(&node_delete[field], expected, "field {field}");
        }
        for field in node_delete_contract["integer_response_fields"]
            .as_array()
            .unwrap()
        {
            let field = field.as_str().unwrap();
            assert!(node_delete[field].as_i64().is_some(), "field {field}");
        }
        for field in node_delete_contract["absent_response_fields"]
            .as_array()
            .unwrap()
        {
            let field = field.as_str().unwrap();
            assert!(node_delete.get(field).is_none(), "field {field}");
        }
        let redo_contract: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/workflow-design-redo-http-v1.json"
        )))
        .unwrap();
        let redo_workflow = redo_contract["workflow_id"].as_str().unwrap();
        let redo_node = redo_contract["node_id"].as_str().unwrap();
        let redo_create = public_call!(
            "POST",
            &format!("/api/workflow/design/{redo_workflow}/nodes"),
            json!({
                "designer_id":"alice",
                "node_id":redo_node,
                "label":"Start",
                "op":"start",
                "start":true
            })
        );
        assert_eq!(redo_create.status(), StatusCode::OK);
        let redo_undo = public_call!(
            "POST",
            &format!("/api/workflow/design/{redo_workflow}/undo"),
            json!({"designer_id":"alice"})
        );
        assert_eq!(redo_undo.status(), StatusCode::OK);
        let redo_response = public_call!(
            "POST",
            &format!("/api/workflow/design/{redo_workflow}/redo"),
            json!({"designer_id":"alice"})
        );
        assert_eq!(
            redo_response.status().as_u16(),
            redo_contract["success_status"].as_u64().unwrap() as u16
        );
        let redo: Value = serde_json::from_slice(
            &to_bytes(redo_response.into_body(), 32_768).await.unwrap(),
        )
        .unwrap();
        for (field, expected) in redo_contract["fixed_response_fields"].as_object().unwrap() {
            assert_eq!(&redo[field], expected, "field {field}");
        }
        for field in redo_contract["integer_response_fields"].as_array().unwrap() {
            let field = field.as_str().unwrap();
            assert!(redo[field].as_i64().is_some(), "field {field}");
        }
        for (field, expected_length) in redo_contract["array_lengths"].as_object().unwrap() {
            assert_eq!(
                redo[field].as_array().unwrap().len(),
                expected_length.as_u64().unwrap() as usize,
                "field {field}"
            );
        }
        assert_eq!(redo["commits"]["1"]["action"], redo_contract["version_one_action"]);
        for field in redo_contract["absent_response_fields"].as_array().unwrap() {
            let field = field.as_str().unwrap();
            assert!(redo.get(field).is_none(), "field {field}");
        }
        let undo_contract: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/workflow-design-undo-http-v1.json"
        )))
        .unwrap();
        let undo_workflow = undo_contract["workflow_id"].as_str().unwrap();
        let undo_node = undo_contract["node_id"].as_str().unwrap();
        let undo_create = public_call!(
            "POST",
            &format!("/api/workflow/design/{undo_workflow}/nodes"),
            json!({
                "designer_id":"alice",
                "node_id":undo_node,
                "label":"Start",
                "op":"start",
                "start":true
            })
        );
        assert_eq!(undo_create.status(), StatusCode::OK);
        let undo_response = public_call!(
            "POST",
            &format!("/api/workflow/design/{undo_workflow}/undo"),
            json!({"designer_id":"alice"})
        );
        assert_eq!(
            undo_response.status().as_u16(),
            undo_contract["success_status"].as_u64().unwrap() as u16
        );
        let undo: Value = serde_json::from_slice(
            &to_bytes(undo_response.into_body(), 32_768).await.unwrap(),
        )
        .unwrap();
        for (field, expected) in undo_contract["fixed_response_fields"].as_object().unwrap() {
            assert_eq!(&undo[field], expected, "field {field}");
        }
        for field in undo_contract["integer_response_fields"].as_array().unwrap() {
            let field = field.as_str().unwrap();
            assert!(undo[field].as_i64().is_some(), "field {field}");
        }
        for (field, expected_length) in undo_contract["array_lengths"].as_object().unwrap() {
            assert_eq!(
                undo[field].as_array().unwrap().len(),
                expected_length.as_u64().unwrap() as usize,
                "field {field}"
            );
        }
        assert_eq!(undo["commits"]["1"]["action"], undo_contract["version_one_action"]);
        for field in undo_contract["absent_response_fields"].as_array().unwrap() {
            let field = field.as_str().unwrap();
            assert!(undo.get(field).is_none(), "field {field}");
        }
        let principal = json!({"sub":"alice","role":"rw","ns":"workflow","capabilities":["workflow.design.write","workflow.design.inspect"]});
        let call = |method: &str, path: &str, body: Value| {
            service.execute_sync(ApiEffectRequest {
                contract_version: 1,
                method: method.to_owned(),
                path_and_query: path.to_owned(),
                body: serde_json::to_vec(&body).unwrap(),
                principal: principal.clone(),
            })
        };
        let edge_upsert_contract: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/workflow-design-edge-upsert-http-v1.json"
        )))
        .unwrap();
        for node in [
            json!({
                "designer_id": "alice",
                "node_id": "start",
                "label": "Start",
                "op": "start",
                "start": true
            }),
            json!({
                "designer_id": "alice",
                "node_id": "end",
                "label": "End",
                "op": "end",
                "terminal": true
            }),
        ] {
            let response = public_call!(
                "POST",
                "/api/workflow/design/wf-edge-contract/nodes",
                node
            );
            assert_eq!(response.status(), StatusCode::OK);
        }
        let edge_upsert = public_call!(
            "POST",
            "/api/workflow/design/wf-edge-contract/edges",
            json!({
                "designer_id": "alice",
                "edge_id": "edge-contract",
                "src": "start",
                "dst": "end",
                "is_default": true
            })
        );
        assert_eq!(
            edge_upsert.status().as_u16(),
            edge_upsert_contract["success_status"]
                .as_u64()
                .unwrap() as u16
        );
        let edge_upsert: Value = serde_json::from_slice(
            &to_bytes(edge_upsert.into_body(), 32_768).await.unwrap(),
        )
        .unwrap();
        for (field, expected) in edge_upsert_contract["fixed_response_fields"]
            .as_object()
            .unwrap()
        {
            assert_eq!(&edge_upsert[field], expected, "field {field}");
        }
        for field in edge_upsert_contract["integer_response_fields"]
            .as_array()
            .unwrap()
        {
            let field = field.as_str().unwrap();
            assert!(edge_upsert[field].as_i64().is_some(), "field {field}");
        }
        for field in edge_upsert_contract["absent_response_fields"]
            .as_array()
            .unwrap()
        {
            let field = field.as_str().unwrap();
            assert!(edge_upsert.get(field).is_none(), "field {field}");
        }
        for body in [
            json!({"designer_id":"alice","node_id":"start","label":"Start","op":"increment-counter","start":true,"metadata":{"wf_executor_target":"rust-native"}}),
            json!({"designer_id":"alice","node_id":"end","label":"End","op":"emit-result","terminal":true,"metadata":{"wf_executor_target":"rust-native"}}),
        ] {
            let response = public_call!("POST", "/api/workflow/design/wf-write/nodes", body);
            assert_eq!(response.status(), StatusCode::OK);
        }
        let response = public_call!("POST", "/api/workflow/design/wf-write/edges", json!({
            "designer_id":"alice","edge_id":"edge","src":"start","dst":"end","is_default":true
        }));
        assert_eq!(response.status(), StatusCode::OK);
        let response = public_call!("GET", "/api/workflow/design/wf-write/graph", json!({}));
        let graph: Value = serde_json::from_slice(
            &to_bytes(response.into_body(), 65_536).await.unwrap(),
        )
        .unwrap();
        assert_eq!(graph["nodes"].as_array().unwrap().len(), 2);
        assert_eq!(graph["edges"].as_array().unwrap().len(), 1);
        let projection = store
            .get_named_projection("workflow_design", "wf-write")
            .unwrap()
            .unwrap();
        assert_eq!(projection.materialization_status, "rust_event_only");
        let design_snapshot = store
            .get_workflow_design_snapshot("wf-write", 3, 1)
            .unwrap()
            .expect("every Rust-published design version has an executable snapshot");
        assert_eq!(design_snapshot.version, 3);
        assert!(runtime_graph_plan_from_snapshot(&design_snapshot.payload_json).is_some());
        assert!(exact_runtime_graph_plan(Some(projection), Some(design_snapshot)).is_some());
        let submitted = public_call!("POST", "/api/workflow/runs", json!({
            "run_id": "run-from-rust-design",
            "workflow_id": "wf-write",
            "conversation_id": "conversation-from-rust-design",
            "initial_state": {"counter": 1}
        }));
        assert_eq!(submitted.status(), StatusCode::ACCEPTED);
        let submitted: Value = serde_json::from_slice(
            &to_bytes(submitted.into_body(), 65_536).await.unwrap(),
        )
        .unwrap();
        let lane = store
            .list_projected_lane_messages(LaneMessageFilter {
                namespace: Some("workflow".to_owned()),
                inbox_id: Some("workflow-runtime".to_owned()),
                correlation_id: Some("run-from-rust-design".to_owned()),
                limit: 1,
                ..LaneMessageFilter::default()
            })
            .unwrap()
            .into_iter()
            .next()
            .expect("design submission durably creates a runtime lane message");
        let worker_payload: Value = serde_json::from_str(lane.payload_json.as_deref().unwrap()).unwrap();
        assert_eq!(worker_payload["run_id"], submitted["run_id"]);
        assert_eq!(worker_payload["start_node_id"], "start");
        assert_eq!(worker_payload["runtime_routes"][0]["source_node_id"], "start");
        assert_eq!(worker_payload["runtime_routes"][0]["target_node_id"], "end");
        assert_eq!(worker_payload["executor_target"], "rust-native");

        let execute_claimed_native_step = |work: &Value| {
            call(
                "POST",
                "/internal/runtime/native/execute",
                json!({
                    "handoff": {
                        "message_id": work["message_id"],
                        "claimed_by": work["claimed_by"],
                        "run_id": work["run_id"],
                        "step_id": work["step_id"],
                        "correlation_id": work["correlation_id"]
                    },
                    "payload": work["payload"]
                }),
            )
        };
        let first_claim = call(
            "POST",
            "/internal/runtime/claim",
            json!({
                "claimed_by": "rust-native-worker",
                "limit": 1,
                "lease_seconds": 60,
                "recipient_ids": ["rust-native-worker"]
            }),
        );
        assert_eq!(first_claim.status, 200);
        let first_work: Value = serde_json::from_slice(&first_claim.body).unwrap();
        let first_work = &first_work["work"][0];
        assert_eq!(first_work["step_id"], "start");
        assert_eq!(first_work["payload"]["op"], "increment-counter");
        assert_eq!(execute_claimed_native_step(first_work).status, 200);

        let second_claim = call(
            "POST",
            "/internal/runtime/claim",
            json!({
                "claimed_by": "rust-native-worker",
                "limit": 1,
                "lease_seconds": 60,
                "recipient_ids": ["rust-native-worker"]
            }),
        );
        assert_eq!(second_claim.status, 200);
        let second_work: Value = serde_json::from_slice(&second_claim.body).unwrap();
        let second_work = &second_work["work"][0];
        assert_eq!(second_work["step_id"], "end");
        assert_eq!(second_work["payload"]["op"], "emit-result");
        assert_eq!(execute_claimed_native_step(second_work).status, 200);
        assert_eq!(store.get_server_run("run-from-rust-design").unwrap().unwrap().status, "succeeded");
        let run_events = store
            .list_server_run_events("run-from-rust-design", 0, 100)
            .unwrap();
        assert!(run_events.iter().any(|event| {
            event.event_type == "workflow.recorded_transition.v1"
                && event.payload_json.contains("increment-counter")
                && event.payload_json.contains(":2")
        }));
        assert!(run_events.iter().any(|event| {
            event.event_type == "workflow.trace.v1"
                && event.payload_json.contains("runtime.native.step.completed")
                && event.payload_json.contains("\"seam_crossings\":[]")
        }));
        let events = store
            .replay_raw_events("wf_design:wf-write", 0, usize::MAX)
            .unwrap();
        assert_eq!(
            events
                .iter()
                .map(|event| (event.entity_kind.as_str(), event.op.as_str()))
                .collect::<Vec<_>>(),
            [
                ("node", "ADD"),
                ("design_control", "MUTATION_COMMITTED"),
                ("node", "ADD"),
                ("design_control", "MUTATION_COMMITTED"),
                ("edge", "ADD"),
                ("design_control", "MUTATION_COMMITTED"),
            ]
        );
        let delta = store
            .get_workflow_design_delta("wf-write", 3, 1)
            .unwrap()
            .unwrap();
        assert_eq!(delta.target_seq, 5);
        assert_eq!(
            serde_json::from_str::<Value>(&delta.forward_json).unwrap()["upsert_edges"][0]["id"],
            "edge"
        );
        assert_eq!(
            store
                .get_named_projection("workflow_design", "wf-write")
                .unwrap()
                .unwrap()
                .materialization_status,
            "rust_event_only"
        );
        let response = public_call!(
            "POST",
            "/api/workflow/design/wf-write/undo",
            json!({"designer_id":"alice"})
        );
        assert_eq!(response.status(), StatusCode::OK);
        let response = public_call!("GET", "/api/workflow/design/wf-write/graph", json!({}));
        let graph: Value = serde_json::from_slice(
            &to_bytes(response.into_body(), 65_536).await.unwrap(),
        )
        .unwrap();
        assert!(graph["edges"].as_array().unwrap().is_empty());
        assert!(store.get_workflow_design_snapshot("wf-write", 2, 1).unwrap().is_some());
        let response = public_call!(
            "POST",
            "/api/workflow/design/wf-write/redo",
            json!({"designer_id":"alice"})
        );
        assert_eq!(response.status(), StatusCode::OK);
        assert!(store.get_workflow_design_snapshot("wf-write", 3, 1).unwrap().is_some());
        let response = public_call!(
            "DELETE",
            "/api/workflow/design/wf-write/edges/edge",
            json!({"designer_id":"alice"})
        );
        assert_eq!(
            response.status().as_u16(),
            edge_delete_contract["success_status"].as_u64().unwrap() as u16
        );
        let deleted: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 32_768).await.unwrap())
                .unwrap();
        for (field, expected) in edge_delete_contract["fixed_response_fields"]
            .as_object()
            .unwrap()
        {
            assert_eq!(&deleted[field], expected, "field {field}");
        }
        for field in edge_delete_contract["integer_response_fields"]
            .as_array()
            .unwrap()
        {
            let field = field.as_str().unwrap();
            assert!(deleted[field].as_i64().is_some(), "field {field}");
        }
        for field in edge_delete_contract["absent_response_fields"]
            .as_array()
            .unwrap()
        {
            let field = field.as_str().unwrap();
            assert!(deleted.get(field).is_none(), "field {field}");
        }
        let node_upsert_mcp: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/workflow-design-node-upsert-mcp-v1.json"
        )))
        .unwrap();
        let mcp_request = serde_json::to_vec(&json!({
            "jsonrpc":"2.0",
            "id":"node",
            "method":"tools/call",
            "params": {
                "name": node_upsert_mcp["tool"],
                "arguments": node_upsert_mcp["arguments"]
            }
        }))
        .unwrap();
        let mcp = app.clone().oneshot(Request::builder().method("POST").uri("/mcp/workflow")
            .header("content-type", "application/json").header("authorization", format!("Bearer {token}"))
            .body(Body::from(mcp_request)).unwrap()).await.unwrap();
        assert_eq!(mcp.status(), StatusCode::OK);
        let body: Value =
            serde_json::from_slice(&to_bytes(mcp.into_body(), 65_536).await.unwrap()).unwrap();
        for (field, expected) in node_upsert_mcp["fixed_result_fields"].as_object().unwrap() {
            assert_eq!(&body["result"][field], expected, "field {field}");
        }
        for field in node_upsert_mcp["integer_result_fields"].as_array().unwrap() {
            let field = field.as_str().unwrap();
            assert!(body["result"][field].as_i64().is_some(), "field {field}");
        }
        let undo_mcp: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/workflow-design-undo-mcp-v1.json"
        )))
        .unwrap();
        let undo_args = &undo_mcp["arguments"];
        let created = public_call!(
            "POST",
            "/api/workflow/design/wf-undo-mcp-contract/nodes",
            json!({
                "workflow_id":undo_args["workflow_id"],
                "designer_id":undo_args["designer_id"],
                "node_id":"undo-node-contract",
                "label":"Start",
                "op":"start",
                "start":true
            })
        );
        assert_eq!(created.status(), StatusCode::OK);
        let undo_request = serde_json::to_vec(&json!({
            "jsonrpc":"2.0",
            "id":"undo",
            "method":"tools/call",
            "params": {
                "name":undo_mcp["tool"],
                "arguments":undo_args
            }
        }))
        .unwrap();
        let undone = app.clone().oneshot(Request::builder().method("POST").uri("/mcp/workflow")
            .header("content-type", "application/json").header("authorization", format!("Bearer {token}"))
            .body(Body::from(undo_request)).unwrap()).await.unwrap();
        assert_eq!(undone.status(), StatusCode::OK);
        let body: Value = serde_json::from_slice(
            &to_bytes(undone.into_body(), 65_536).await.unwrap()
        ).unwrap();
        for (field, expected) in undo_mcp["fixed_result_fields"].as_object().unwrap() {
            assert_eq!(&body["result"][field], expected, "field {field}");
        }
        for field in undo_mcp["integer_result_fields"].as_array().unwrap() {
            let field = field.as_str().unwrap();
            assert!(body["result"][field].as_i64().is_some(), "field {field}");
        }
        for (field, length) in undo_mcp["array_lengths"].as_object().unwrap() {
            assert_eq!(body["result"][field].as_array().unwrap().len(), length.as_u64().unwrap() as usize);
        }
        assert_eq!(body["result"]["commits"]["1"]["action"], undo_mcp["version_one_action"]);
        for field in undo_mcp["absent_result_fields"].as_array().unwrap() {
            let field = field.as_str().unwrap();
            assert!(body["result"].get(field).is_none(), "field {field}");
        }
        // PYTHON_PARITY_NOTE_V1: MCP redo response matches Python's workflow MCP projection; rust=rust/crates/kogwistar-api/src/sections/http_transport.rs#mcp_effect_request; python=kogwistar/server/chat_mcp.py#build_workflow_mcp.workflow_design_redo; fixture=contracts/golden/workflow-design-redo-mcp-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/workflow_design_history.rs#sqlite_design_capability_mutates_graph_and_undoes_atomically; python_test=tests/mcp/test_chat_mcp_tools.py#test_mcp_workflow_design_redo_matches_shared_contract
        let redo_mcp: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/workflow-design-redo-mcp-v1.json"
        )))
        .unwrap();
        let redo_args = &redo_mcp["arguments"];
        let redo_workflow_id = redo_args["workflow_id"].as_str().unwrap();
        let redo_created = public_call!(
            "POST",
            &format!("/api/workflow/design/{redo_workflow_id}/nodes"),
            json!({
                "designer_id":redo_args["designer_id"],
                "node_id":"redo-node-contract",
                "label":"Start",
                "op":"start",
                "start":true
            })
        );
        assert_eq!(redo_created.status(), StatusCode::OK);
        let redo_undone = public_call!(
            "POST",
            &format!("/api/workflow/design/{redo_workflow_id}/undo"),
            json!({"designer_id":redo_args["designer_id"]})
        );
        assert_eq!(redo_undone.status(), StatusCode::OK);
        let redo_request = serde_json::to_vec(&json!({
            "jsonrpc":"2.0",
            "id":"redo",
            "method":"tools/call",
            "params": {
                "name":redo_mcp["tool"],
                "arguments":redo_args
            }
        }))
        .unwrap();
        let redone = app.clone().oneshot(Request::builder().method("POST").uri("/mcp/workflow")
            .header("content-type", "application/json").header("authorization", format!("Bearer {token}"))
            .body(Body::from(redo_request)).unwrap()).await.unwrap();
        assert_eq!(redone.status(), StatusCode::OK);
        let body: Value = serde_json::from_slice(
            &to_bytes(redone.into_body(), 65_536).await.unwrap()
        ).unwrap();
        for (field, expected) in redo_mcp["fixed_result_fields"].as_object().unwrap() {
            assert_eq!(&body["result"][field], expected, "field {field}");
        }
        for field in redo_mcp["integer_result_fields"].as_array().unwrap() {
            let field = field.as_str().unwrap();
            assert!(body["result"][field].as_i64().is_some(), "field {field}");
        }
        for (field, length) in redo_mcp["array_lengths"].as_object().unwrap() {
            assert_eq!(body["result"][field].as_array().unwrap().len(), length.as_u64().unwrap() as usize);
        }
        assert_eq!(body["result"]["commits"]["1"]["action"], redo_mcp["version_one_action"]);
        for field in redo_mcp["absent_result_fields"].as_array().unwrap() {
            let field = field.as_str().unwrap();
            assert!(body["result"].get(field).is_none(), "field {field}");
        }
        drop(app);
        drop(service);
        drop(store);
        std::fs::remove_file(path).unwrap();
    }
