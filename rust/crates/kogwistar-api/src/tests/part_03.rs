    #[tokio::test]
    async fn sqlite_visibility_and_capabilities_enforce_claims_across_rest_and_mcp() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use jsonwebtoken::{EncodingKey, Header, encode};
        use std::time::{SystemTime, UNIX_EPOCH};
        use tower::ServiceExt;

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-capabilities-{nonce}.sqlite3"));
        let auth = AuthConfig {
            mode: "dev".to_owned(),
            algorithm: Some("HS256".to_owned()),
            key: Some("capability-secret".to_owned()),
            jwks_json: None,
            issuer: Some("local".to_owned()),
            audience: None,
            oidc_providers_json: None,
            auth_db_url: None,
        };
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: Vec::new(),
                implementation: ImplementationSnapshot::default(),
                auth,
            },
            Arc::new(SqliteRunApplicationService::open(&path).unwrap()),
        );
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let token = |claims: Value| {
            encode(
                &Header::new(Algorithm::HS256),
                &claims,
                &EncodingKey::from_secret(b"capability-secret"),
            )
            .unwrap()
        };
        let read_token = token(json!({
            "sub": "reader-1",
            "role": "ro",
            "ns": ["workflow"],
            "capabilities": ["project_view", "read_security_scope"],
            "tenant": "tenant-a",
            "workspace": "workspace-b",
            "project": "project-c",
            "iss": "local",
            "exp": now + 300,
        }));
        let visibility = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/workflow/visibility")
                    .header("authorization", format!("Bearer {read_token}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(visibility.status(), StatusCode::OK);
        let visibility = serde_json::from_slice::<Value>(
            &to_bytes(visibility.into_body(), 32768).await.unwrap(),
        )
        .unwrap();
        assert_eq!(visibility["current_subject"], "reader-1");
        assert_eq!(visibility["namespaces"]["execution_namespace"], "workflow");
        assert_eq!(visibility["security_scope"], "tenant-a");
        assert_eq!(
            visibility["storage_security_mapping"]["security_scope_path"],
            "tenant-a/workspace-b/project-c"
        );

        let denied_token = token(json!({
            "sub": "denied",
            "role": "ro",
            "ns": ["workflow"],
            "capabilities": ["workflow.run.read"],
            "iss": "local",
            "exp": now + 300,
        }));
        let denied = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/workflow/capabilities")
                    .header("authorization", format!("Bearer {denied_token}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(denied.status(), StatusCode::FORBIDDEN);

        let admin_token = token(json!({
            "sub": "admin-1",
            "role": "rw",
            "ns": ["workflow"],
            "capabilities": ["project_view", "approve_action"],
            "iss": "local",
            "exp": now + 300,
        }));
        let approve = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/workflow/capabilities/approve")
                    .header("authorization", format!("Bearer {admin_token}"))
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"action":"spawn_process","capabilities":["spawn_process","workflow.run.write"]}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(approve.status(), StatusCode::ACCEPTED);
        let approve =
            serde_json::from_slice::<Value>(&to_bytes(approve.into_body(), 32768).await.unwrap())
                .unwrap();
        assert!(
            approve["effective_capabilities"]
                .as_array()
                .unwrap()
                .contains(&json!("spawn_process"))
        );
        assert_eq!(
            approve["audit_log"].as_array().unwrap().last().unwrap()["outcome"],
            "allow"
        );

        let revoke = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/workflow/capabilities/revoke")
                    .header("authorization", format!("Bearer {admin_token}"))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"capability":"spawn_process"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(revoke.status(), StatusCode::ACCEPTED);
        let revoke =
            serde_json::from_slice::<Value>(&to_bytes(revoke.into_body(), 32768).await.unwrap())
                .unwrap();
        assert!(
            !revoke["effective_capabilities"]
                .as_array()
                .unwrap()
                .contains(&json!("spawn_process"))
        );

        let mcp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/workflow")
                    .header("authorization", format!("Bearer {read_token}"))
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"workflow.visibility_snapshot","arguments":{}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(mcp.status(), StatusCode::OK);
        let mcp = serde_json::from_slice::<Value>(&to_bytes(mcp.into_body(), 32768).await.unwrap())
            .unwrap();
        assert_eq!(mcp["result"]["current_subject"], "reader-1");

        let syscall_list = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/syscall/v1")
                    .header("authorization", format!("Bearer {read_token}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(syscall_list.status(), StatusCode::OK);
        let syscall_list = serde_json::from_slice::<Value>(
            &to_bytes(syscall_list.into_body(), 16_384).await.unwrap(),
        )
        .unwrap();
        assert_eq!(syscall_list["version"], "v1");
        assert!(
            syscall_list["ops"]
                .as_array()
                .unwrap()
                .contains(&json!("resume"))
        );
        let blocked_approval = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/syscall/v1/request_approval")
                    .header("authorization", format!("Bearer {admin_token}"))
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"version":"v1","op":"request_approval","args":{"action":"deny","reason":"manual deny"}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(blocked_approval.status(), StatusCode::OK);
        let blocked_approval = serde_json::from_slice::<Value>(
            &to_bytes(blocked_approval.into_body(), 16_384)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(blocked_approval["status"], "blocked");
        assert_eq!(blocked_approval["error"]["reason"], "manual deny");
        let unsupported = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/syscall/v1/send_message")
                    .header("authorization", format!("Bearer {admin_token}"))
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"version":"v1","op":"send_message","args":{}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(unsupported.status(), StatusCode::NOT_IMPLEMENTED);
        let syscall_audit = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/syscall/v1/audit?limit=1")
                    .header("authorization", format!("Bearer {read_token}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(syscall_audit.status(), StatusCode::OK);
        let syscall_audit = serde_json::from_slice::<Value>(
            &to_bytes(syscall_audit.into_body(), 16_384).await.unwrap(),
        )
        .unwrap();
        assert_eq!(syscall_audit["events"][0]["op"], "send_message");
        assert_eq!(syscall_audit["events"][0]["status"], "error");

        let designer = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/designer/capabilities")
                    .header("authorization", format!("Bearer {read_token}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(designer.status(), StatusCode::FORBIDDEN);
        let designer_token = token(json!({
            "sub": "designer-1",
            "role": "ro",
            "ns": ["workflow"],
            "capabilities": ["workflow.design.inspect"],
            "iss": "local",
            "exp": now + 300,
        }));
        let designer = app
            .oneshot(
                Request::builder()
                    .uri("/designer/capabilities")
                    .header("authorization", format!("Bearer {designer_token}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(designer.status(), StatusCode::OK);
        let designer =
            serde_json::from_slice::<Value>(&to_bytes(designer.into_body(), 65_536).await.unwrap())
                .unwrap();
        assert_eq!(
            designer["schema_version"],
            "workflow-designer-capabilities/v1"
        );
        assert_eq!(designer["node_types"][0]["type"], "workflow_node");
        assert!(
            designer["node_types"][0]["metadata_schema"]["properties"]
                .get("wf_join")
                .is_some()
        );
        assert!(
            designer["edge_types"][0]["metadata_schema"]["properties"]
                .get("wf_predicate")
                .is_some()
        );
        std::fs::remove_file(path).unwrap();
    }

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
        let path = std::env::temp_dir().join(format!(
            "kogwistar-design-write-{}.sqlite3",
            uuid::Uuid::new_v4()
        ));
        let service = SqliteRunApplicationService::open(&path).unwrap();
        let store = SqliteStore::open(&path).unwrap();
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
        for body in [
            json!({"designer_id":"alice","node_id":"start","label":"Start","op":"start","start":true}),
            json!({"designer_id":"alice","node_id":"end","label":"End","op":"end","terminal":true}),
        ] {
            assert_eq!(
                call("POST", "/api/workflow/design/wf-write/nodes", body).status,
                200
            );
        }
        assert_eq!(call("POST", "/api/workflow/design/wf-write/edges", json!({
            "designer_id":"alice","edge_id":"edge","src":"start","dst":"end","is_default":true
        })).status, 200);
        let graph = call("GET", "/api/workflow/design/wf-write/graph", json!({}));
        let graph: Value = serde_json::from_slice(&graph.body).unwrap();
        assert_eq!(graph["nodes"].as_array().unwrap().len(), 2);
        assert_eq!(graph["edges"].as_array().unwrap().len(), 1);
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
        assert_eq!(
            call(
                "POST",
                "/api/workflow/design/wf-write/undo",
                json!({"designer_id":"alice"})
            )
            .status,
            200
        );
        let graph = call("GET", "/api/workflow/design/wf-write/graph", json!({}));
        let graph: Value = serde_json::from_slice(&graph.body).unwrap();
        assert!(graph["edges"].as_array().unwrap().is_empty());
        assert_eq!(
            call(
                "POST",
                "/api/workflow/design/wf-write/redo",
                json!({"designer_id":"alice"})
            )
            .status,
            200
        );
        assert_eq!(
            call(
                "DELETE",
                "/api/workflow/design/wf-write/edges/edge",
                json!({"designer_id":"alice"})
            )
            .status,
            200
        );
        let key = "design-secret";
        let auth = AuthConfig {
            mode: "jwt".to_owned(),
            algorithm: Some("HS256".to_owned()),
            key: Some(key.to_owned()),
            ..AuthConfig::default()
        };
        let claims = json!({"sub":"alice","role":"rw","roles":["reader","rw"],"ns":"workflow","capabilities":["workflow.design.write","workflow.design.inspect"],"exp":4102444800_u64});
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
        let mcp = app.oneshot(Request::builder().method("POST").uri("/mcp/workflow")
            .header("content-type", "application/json").header("authorization", format!("Bearer {token}"))
            .body(Body::from(r#"{"jsonrpc":"2.0","id":"node","method":"tools/call","params":{"name":"workflow.design_node_upsert","arguments":{"workflow_id":"wf-mcp-design","designer_id":"alice","node_id":"mcp-node","label":"MCP"}}}"#)).unwrap()).await.unwrap();
        assert_eq!(mcp.status(), StatusCode::OK);
        let body: Value =
            serde_json::from_slice(&to_bytes(mcp.into_body(), 65_536).await.unwrap()).unwrap();
        assert_eq!(body["result"]["node_id"], "mcp-node");
        drop(service);
        drop(store);
        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    #[ignore = "requires KOGWISTAR_TEST_PG_DSN; run explicitly with --include-ignored"]
    async fn postgres_internal_graph_mutation_route_persists_atomically_when_dsn_available() {
        let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
            .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
        let schema = format!("rust_api_graph_mutation_{}", uuid::Uuid::new_v4().simple());
        let (client, connection) = tokio_postgres::connect(&dsn, tokio_postgres::NoTls)
            .await
            .unwrap();
        tokio::spawn(async move {
            let _ = connection.await;
        });
        client
            .batch_execute(&format!(
                r#"CREATE EXTENSION IF NOT EXISTS vector;
                CREATE SCHEMA "{schema}";
                CREATE TABLE "{schema}"."gke_nodes" (
                    id TEXT PRIMARY KEY,
                    document TEXT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding vector(3) NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE TABLE "{schema}"."gke_edges" (
                    id TEXT PRIMARY KEY,
                    document TEXT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding vector(3) NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE TABLE "{schema}"."gke_edge_endpoints" (
                    id TEXT PRIMARY KEY,
                    document TEXT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );"#
            ))
            .await
            .unwrap();
        let service = PostgresRunApplicationService::from_dsn(&dsn, &schema).unwrap();
        service.ensure_schema().await.unwrap();
        let mut body: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/graph-mutation-batch-v1.json"
        )))
        .unwrap();
        body["mutations"][0]["scope"]["namespace"] = json!("alpha");
        body["mutations"][0]["record"]["document"] = json!("{\"label\":\"Alpha source\"}");
        body["mutations"][0]["record"]["embedding"] = json!([1.0, 0.0, 0.0]);
        body["mutations"][0]["embedding_dim"] = json!(3);
        let mut second_node = body["mutations"][0].clone();
        second_node["event_id"] = json!("graph-route-event-node-2");
        second_node["payload"]["id"] = json!("node-2");
        second_node["record"]["id"] = json!("node-2");
        second_node["record"]["document"] = json!(r#"{"label":"Beta target"}"#);

        let mut edge = body["mutations"][0].clone();
        edge["table"] = json!("gke_edges");
        edge["entity_kind"] = json!("edge");
        edge["event_id"] = json!("graph-route-event-edge-1");
        edge["payload"] = json!({"id": "edge-1"});
        edge["record"]["id"] = json!("edge-1");
        edge["record"]["document"] = json!(
            r#"{"source_ids":["node-1"],"target_ids":["node-2"],"relation":"supports"}"#
        );
        body["mutations"].as_array_mut().unwrap().push(second_node);
        body["mutations"].as_array_mut().unwrap().push(edge);
        for (target_grain, target_id) in [
            ("node", "node-1"),
            ("node", "node-2"),
            ("edge", "edge-1"),
        ] {
            let acl_id = format!("acl-{target_grain}-{target_id}");
            let mut acl_record = body["mutations"][0].clone();
            acl_record["event_id"] = json!(format!("graph-route-event-{acl_id}"));
            acl_record["payload"]["id"] = json!(acl_id);
            acl_record["record"]["id"] = json!(acl_id);
            acl_record["record"]["metadata"] = json!({
                "entity_type": "acl_record",
                "acl_truth_graph": "knowledge",
                "acl_target_grain": target_grain,
                "acl_target_item_id": "",
                "acl_target_entity_id": target_id,
                "acl_version": 1,
                "acl_mode": "private",
                "owner_id": "alice",
                "tombstoned": false
            });
            acl_record["record"]["document"] = Value::Null;
            acl_record["record"]["embedding"] = Value::Null;
            body["mutations"].as_array_mut().unwrap().push(acl_record);
        }
        body["index_jobs"] = json!([{
            "job_id": "graph-route-index-1",
            "namespace": "alpha",
            "entity_kind": "node",
            "entity_id": "node-1",
            "index_kind": "node_embedding",
            "op": "UPSERT",
            "payload_json": "{\"embedding\":[1.0,0.0,0.0]}",
            "max_retries": 3
        }]);
        let key = "graph-route-test-secret";
        let make_token_for = |subject: &str, capabilities: &[&str], storage_ns: &str| {
            encode(
                &Header::new(Algorithm::HS256),
                &json!({
                    "sub": subject,
                    "role": "rw",
                    "roles": ["reader", "rw"],
                    "ns": "workflow",
                    "storage_ns": storage_ns,
                    "capabilities": capabilities,
                    "exp": 4102444800_u64
                }),
                &EncodingKey::from_secret(key.as_bytes()),
            )
            .unwrap()
        };
        let make_token = |capabilities: &[&str], storage_ns: &str| {
            make_token_for("alice", capabilities, storage_ns)
        };
        let make_app = || {
            router_with_application(
                ApiState {
                    health: snapshot(),
                    required_roles: vec!["reader".to_owned()],
                    implementation: ImplementationSnapshot::default(),
                    auth: AuthConfig {
                        mode: "jwt".to_owned(),
                        algorithm: Some("HS256".to_owned()),
                        key: Some(key.to_owned()),
                        ..AuthConfig::default()
                    },
                },
                Arc::new(service.clone()),
            )
        };
        let mcp_list = make_app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header(
                        "authorization",
                        format!("Bearer {}", make_token(&["read_graph"], "alpha")),
                    )
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":"list","method":"tools/list"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(mcp_list.status(), StatusCode::OK);
        let mcp_list_body = to_bytes(mcp_list.into_body(), 65_536).await.unwrap();
        let mcp_list_value: Value = serde_json::from_slice(&mcp_list_body).unwrap();
        assert!(
            mcp_list_value["result"]["tools"]
                .as_array()
                .unwrap()
                .iter()
                .any(|tool| tool["name"] == "kg_neighbors")
        );

        let mcp_missing_capability = make_app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header("authorization", format!("Bearer {}", make_token(&[], "alpha")))
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":"denied","method":"tools/call","params":{"name":"kg_neighbors","arguments":{"rid":"node-1"}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(mcp_missing_capability.status(), StatusCode::OK);
        let denied_body = to_bytes(mcp_missing_capability.into_body(), 65_536)
            .await
            .unwrap();
        let denied_value: Value = serde_json::from_slice(&denied_body).unwrap();
        assert_eq!(denied_value["error"]["code"], -32000);
        assert_eq!(
            denied_value["error"]["data"]["code"],
            "KOGWISTAR_CAPABILITY_FORBIDDEN"
        );

        let send_mutation = |app: Router, token: String, payload: Value| async move {
            app.oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/graph/mutations")
                    .header("content-type", "application/json")
                    .header("authorization", format!("Bearer {token}"))
                    .body(Body::from(serde_json::to_vec(&payload).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap()
        };

        let missing_capability =
            send_mutation(make_app(), make_token(&[], "alpha"), body.clone()).await;
        assert_eq!(missing_capability.status(), StatusCode::FORBIDDEN);

        let mut cross_scope_body = body.clone();
        for mutation in cross_scope_body["mutations"].as_array_mut().unwrap() {
            mutation["scope"]["namespace"] = json!("beta");
        }
        cross_scope_body["index_jobs"][0]["namespace"] = json!("beta");
        let cross_scope = send_mutation(
            make_app(),
            make_token(&["write_graph"], "alpha"),
            cross_scope_body,
        )
        .await;
        assert_eq!(cross_scope.status(), StatusCode::FORBIDDEN);

        let scoped_key = key;
        let scoped_token = encode(
            &Header::new(Algorithm::HS256),
            &json!({
                "sub": "alice",
                "role": "rw",
                "roles": ["reader", "rw"],
                "ns": "workflow",
                "storage_ns": "alpha",
                "workspace": "workspace-a",
                "graph_space": "facts",
                "capabilities": ["write_graph"],
                "exp": 4102444800_u64
            }),
            &EncodingKey::from_secret(scoped_key.as_bytes()),
        )
        .unwrap();
        let mut workspace_mismatch_body = body.clone();
        for mutation in workspace_mismatch_body["mutations"].as_array_mut().unwrap() {
            mutation["scope"]["workspace_id"] = json!("workspace-b");
        }
        let workspace_mismatch =
            send_mutation(make_app(), scoped_token, workspace_mismatch_body).await;
        assert_eq!(workspace_mismatch.status(), StatusCode::FORBIDDEN);
        let mut graph_space_mismatch_body = body.clone();
        for mutation in graph_space_mismatch_body["mutations"].as_array_mut().unwrap() {
            mutation["scope"]["graph_space"] = json!("private");
        }
        let graph_space_mismatch = send_mutation(
            make_app(),
            encode(
                &Header::new(Algorithm::HS256),
                &json!({
                    "sub": "alice",
                    "role": "rw",
                    "roles": ["reader", "rw"],
                    "ns": "workflow",
                    "storage_ns": "alpha",
                    "workspace": "workspace-a",
                    "graph_space": "facts",
                    "capabilities": ["write_graph"],
                    "exp": 4102444800_u64
                }),
                &EncodingKey::from_secret(scoped_key.as_bytes()),
            )
            .unwrap(),
            graph_space_mismatch_body,
        )
        .await;
        assert_eq!(graph_space_mismatch.status(), StatusCode::FORBIDDEN);

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = make_app();
        let server = tokio::spawn(async move { axum::serve(listener, app).await });
        let http_response = reqwest::Client::new()
            .post(format!("http://{address}/internal/graph/mutations"))
            .bearer_auth(make_token(&["write_graph"], "alpha"))
            .json(&body)
            .send()
            .await
            .unwrap();
        let http_status = http_response.status();
        let response_body = http_response.bytes().await.unwrap();
        assert_eq!(
            http_status,
            StatusCode::OK,
            "{}",
            String::from_utf8_lossy(&response_body)
        );
        let result: Value = serde_json::from_slice(&response_body).unwrap();
        assert_eq!(result["atomic"], true);
        assert!(native_seam_attestation_is_valid(&result));
        assert_eq!(result["implementation_authority"], "rust");
        assert_eq!(result["seam_attestation_version"], 1);
        assert_eq!(result["seam_crossings"], json!([]));
        assert_eq!(result["results"][0]["mutated"], true);

        let replay_response = reqwest::Client::new()
            .post(format!("http://{address}/internal/graph/mutations"))
            .bearer_auth(make_token(&["write_graph"], "alpha"))
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(replay_response.status(), StatusCode::OK);
        let replay: Value = replay_response.json().await.unwrap();
        assert!(
            replay["results"]
                .as_array()
                .unwrap()
                .iter()
                .all(|item| {
                    item["inserted"].as_bool() == Some(false)
                        && item["mutated"].as_bool() == Some(false)
                })
        );

        let mut conflicting_retry = body.clone();
        conflicting_retry["mutations"][0]["payload"]["retry_conflict"] = json!(true);
        let conflicting_response = reqwest::Client::new()
            .post(format!("http://{address}/internal/graph/mutations"))
            .bearer_auth(make_token(&["write_graph"], "alpha"))
            .json(&conflicting_retry)
            .send()
            .await
            .unwrap();
        assert_eq!(conflicting_response.status(), StatusCode::CONFLICT);
        server.abort();
        let _ = server.await;

        let events = service
            .store
            .replay_raw_events("alpha", 0, 100)
            .await
            .unwrap();
        assert_eq!(events.len(), 6);
        let jobs = service
            .store
            .list_index_jobs(Some("alpha"), None, None, None, None, 10)
            .await
            .unwrap();
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].job_id, "graph-route-index-1");

        let records = service
            .store
            .graph_projection_records(kogwistar_store::GraphProjectionRead {
                scope: GraphScope {
                    namespace: "alpha".to_owned(),
                    workspace_id: Some("ws".to_owned()),
                    graph_space: Some("facts".to_owned()),
                },
                table: "gke_nodes".to_owned(),
                ids: Some(vec!["node-1".to_owned()]),
                metadata: Default::default(),
                limit: 10,
            })
            .await
            .unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].id, "node-1");
        let edge_records = service
            .store
            .graph_projection_records(kogwistar_store::GraphProjectionRead {
                scope: GraphScope {
                    namespace: "alpha".to_owned(),
                    workspace_id: Some("ws".to_owned()),
                    graph_space: Some("facts".to_owned()),
                },
                table: "gke_edges".to_owned(),
                ids: Some(vec!["edge-1".to_owned()]),
                metadata: Default::default(),
                limit: 10,
            })
            .await
            .unwrap();
        assert_eq!(edge_records.len(), 1);
        assert_eq!(edge_records[0].id, "edge-1");

        let mcp_neighbors = make_app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header(
                        "authorization",
                        format!("Bearer {}", make_token(&["read_graph"], "alpha")),
                    )
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":"neighbors","method":"tools/call","params":{"name":"kg_neighbors","arguments":{"rid":"node-1"}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(mcp_neighbors.status(), StatusCode::OK);
        let mcp_neighbors_body = to_bytes(mcp_neighbors.into_body(), 65_536).await.unwrap();
        let mcp_neighbors_value: Value = serde_json::from_slice(&mcp_neighbors_body).unwrap();
        assert_eq!(mcp_neighbors_value["result"]["nodes"], json!(["node-2"]));
        assert_eq!(mcp_neighbors_value["result"]["edges"], json!(["edge-1"]));

        let index_jobs = service
            .store
            .list_index_jobs(
                Some("alpha"),
                Some("PENDING"),
                Some("node"),
                Some("node-1"),
                Some("node_embedding"),
                10,
            )
            .await
            .unwrap();
        assert_eq!(index_jobs.len(), 1);
        assert_eq!(index_jobs[0].job_id, "graph-route-index-1");

        // A transport retry must be idempotent at both authority layers:
        // the graph event is not appended twice and the pending index intent
        // remains one coalesced queue row.
        let retry_response = send_mutation(
            make_app(),
            make_token(&["write_graph"], "alpha"),
            body.clone(),
        )
        .await;
        let retry_status = retry_response.status();
        let retry_body = to_bytes(retry_response.into_body(), 65_536).await.unwrap();
        assert_eq!(
            retry_status,
            StatusCode::OK,
            "{}",
            String::from_utf8_lossy(&retry_body)
        );
        let retry_result: Value = serde_json::from_slice(&retry_body).unwrap();
        assert_eq!(retry_result["results"][0]["mutated"], false);
        assert_eq!(
            service
                .store
                .replay_raw_events("alpha", 0, 10)
                .await
                .unwrap()
                .len(),
            6
        );
        assert_eq!(
            service
                .store
                .list_index_jobs(
                    Some("alpha"),
                    Some("PENDING"),
                    Some("node"),
                    Some("node-1"),
                    Some("node_embedding"),
                    10,
                )
                .await
                .unwrap()
                .len(),
            1
        );

        // A queue primary-key collision must roll back the graph event that
        // preceded it in the same transaction; otherwise a retry could leave
        // graph truth ahead of its durable indexing intent.
        let mut rollback_body = body.clone();
        rollback_body["mutations"][0]["event_id"] = json!("graph-route-event-2");
        rollback_body["mutations"][0]["record"]["id"] = json!("node-rollback");
        rollback_body["mutations"][0]["payload"]["id"] = json!("node-rollback");
        rollback_body["index_jobs"][0]["entity_id"] = json!("node-rollback");
        rollback_body["index_jobs"][0]["job_id"] = json!("graph-route-index-1");
        let rollback_response = send_mutation(
            make_app(),
            make_token(&["write_graph"], "alpha"),
            rollback_body,
        )
        .await;
        assert_eq!(rollback_response.status(), StatusCode::CONFLICT);
        assert!(
            service
                .store
                .graph_projection_records(kogwistar_store::GraphProjectionRead {
                    scope: GraphScope {
                        namespace: "alpha".to_owned(),
                        workspace_id: Some("ws".to_owned()),
                        graph_space: Some("facts".to_owned()),
                    },
                    table: "gke_nodes".to_owned(),
                    ids: Some(vec!["node-rollback".to_owned()]),
                    metadata: Default::default(),
                    limit: 10,
                })
                .await
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            service
                .store
                .replay_raw_events("alpha", 0, 10)
                .await
                .unwrap()
                .len(),
            6
        );
        assert!(
            service
                .store
                .list_index_jobs(
                    Some("alpha"),
                    Some("PENDING"),
                    Some("node"),
                    Some("node-2"),
                    Some("node_embedding"),
                    10,
                )
                .await
                .unwrap()
                .is_empty()
        );

        let adjacency_body = json!({
            "contract_version": 1,
            "atomic": true,
            "mutations": [
                {
                    "scope": {"namespace": "alpha", "workspace_id": "ws", "graph_space": "facts"},
                    "table": "gke_nodes",
                    "entity_kind": "node",
                    "event_id": "graph-route-adjacency-node-2",
                    "op": "ADD",
                    "payload": {"id": "node-2"},
                    "record": {"id": "node-2", "document": "{\"label\":\"Beta target\"}", "metadata": {}, "embedding": [0.0, 1.0, 0.0]},
                    "embedding_dim": 3
                },
                {
                    "scope": {"namespace": "alpha", "workspace_id": "ws", "graph_space": "facts"},
                    "table": "gke_edges",
                    "entity_kind": "edge",
                    "event_id": "graph-route-adjacency-edge-1",
                    "op": "ADD",
                    "payload": {"id": "edge-1", "source_ids": ["node-1"], "target_ids": ["node-2"], "source_edge_ids": [], "target_edge_ids": []},
                    "record": {
                        "id": "edge-1",
                        "document": "{\"relation\":\"links\",\"source_ids\":[\"node-1\"],\"target_ids\":[\"node-2\"],\"source_edge_ids\":[],\"target_edge_ids\":[]}",
                        "metadata": {},
                        "embedding": [0.0, 0.0, 1.0]
                    },
                    "embedding_dim": 3
                },
                {
                    "scope": {"namespace": "alpha", "workspace_id": "ws", "graph_space": "facts"},
                    "table": "gke_nodes",
                    "entity_kind": "node",
                    "event_id": "graph-route-acl-node-1",
                    "op": "ADD",
                    "payload": {"id": "acl-node-1"},
                    "record": {"id": "acl-node-1", "metadata": {"entity_type": "acl_record", "acl_truth_graph": "knowledge", "acl_target_grain": "node", "acl_target_entity_id": "node-1", "acl_version": 1, "acl_mode": "public"}, "embedding": [0.0, 0.0, 0.0]},
                    "embedding_dim": 3
                },
                {
                    "scope": {"namespace": "alpha", "workspace_id": "ws", "graph_space": "facts"},
                    "table": "gke_nodes",
                    "entity_kind": "node",
                    "event_id": "graph-route-acl-node-2",
                    "op": "ADD",
                    "payload": {"id": "acl-node-2"},
                    "record": {"id": "acl-node-2", "metadata": {"entity_type": "acl_record", "acl_truth_graph": "knowledge", "acl_target_grain": "node", "acl_target_entity_id": "node-2", "acl_version": 1, "acl_mode": "public"}, "embedding": [0.0, 0.0, 0.0]},
                    "embedding_dim": 3
                },
                {
                    "scope": {"namespace": "alpha", "workspace_id": "ws", "graph_space": "facts"},
                    "table": "gke_nodes",
                    "entity_kind": "node",
                    "event_id": "graph-route-acl-edge-1",
                    "op": "ADD",
                    "payload": {"id": "acl-edge-1"},
                    "record": {"id": "acl-edge-1", "metadata": {"entity_type": "acl_record", "acl_truth_graph": "knowledge", "acl_target_grain": "edge", "acl_target_entity_id": "edge-1", "acl_version": 1, "acl_mode": "public"}, "embedding": [0.0, 0.0, 0.0]},
                    "embedding_dim": 3
                }
            ],
            "index_jobs": []
        });
        let adjacency_response = send_mutation(
            make_app(),
            make_token(&["write_graph"], "alpha"),
            adjacency_body,
        )
        .await;
        assert_eq!(adjacency_response.status(), StatusCode::OK);

        let mcp_adjacency = make_app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header(
                        "authorization",
                        format!("Bearer {}", make_token(&["read_graph"], "alpha")),
                    )
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":"adjacency","method":"tools/call","params":{"name":"kg_neighbors","arguments":{"rid":"node-1"}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(mcp_adjacency.status(), StatusCode::OK);
        let adjacency_body = to_bytes(mcp_adjacency.into_body(), 65_536).await.unwrap();
        let adjacency_value: Value = serde_json::from_slice(&adjacency_body).unwrap();
        assert_eq!(
            adjacency_value["result"]["implementation_authority"],
            "rust"
        );
        assert_eq!(adjacency_value["result"]["seam_attestation_version"], 1);
        assert_eq!(adjacency_value["result"]["seam_crossings"], json!([]));
        assert_eq!(adjacency_value["result"]["nodes"], json!(["node-2"]));
        assert_eq!(adjacency_value["result"]["edges"], json!(["edge-1"]));

        let mcp_find_edges = make_app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header(
                        "authorization",
                        format!("Bearer {}", make_token(&["read_graph"], "alpha")),
                    )
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":"find-edges","method":"tools/call","params":{"name":"kg_find_edges","arguments":{"relation":"links","src_label_contains":"alpha","tgt_label_contains":"TARGET"}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(mcp_find_edges.status(), StatusCode::OK);
        let find_edges_body = to_bytes(mcp_find_edges.into_body(), 65_536).await.unwrap();
        let find_edges_value: Value = serde_json::from_slice(&find_edges_body).unwrap();
        assert_eq!(
            find_edges_value["result"]["implementation_authority"],
            "rust"
        );
        assert_eq!(find_edges_value["result"]["seam_attestation_version"], 1);
        assert_eq!(find_edges_value["result"]["seam_crossings"], json!([]));
        assert_eq!(find_edges_value["result"]["edges"], json!(["edge-1"]));

        let mcp_shortest_path = make_app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header(
                        "authorization",
                        format!("Bearer {}", make_token(&["read_graph"], "alpha")),
                    )
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":"shortest-path","method":"tools/call","params":{"name":"kg_shortest_path","arguments":{"src_id":"node-1","dst_id":"node-2","max_depth":4}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(mcp_shortest_path.status(), StatusCode::OK);
        let shortest_path_body = to_bytes(mcp_shortest_path.into_body(), 65_536)
            .await
            .unwrap();
        let shortest_path_value: Value = serde_json::from_slice(&shortest_path_body).unwrap();
        assert_eq!(
            shortest_path_value["result"]["implementation_authority"],
            "rust"
        );
        assert_eq!(shortest_path_value["result"]["seam_attestation_version"], 1);
        assert_eq!(shortest_path_value["result"]["seam_crossings"], json!([]));
        assert_eq!(
            shortest_path_value["result"]["path"],
            json!(["node-1", "edge-1", "node-2"])
        );

        let mcp_k_hop = make_app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header(
                        "authorization",
                        format!("Bearer {}", make_token(&["read_graph"], "alpha")),
                    )
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":"k-hop","method":"tools/call","params":{"name":"kg_k_hop","arguments":{"start_ids":["node-1"],"k":1}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(mcp_k_hop.status(), StatusCode::OK);
        let k_hop_body = to_bytes(mcp_k_hop.into_body(), 65_536).await.unwrap();
        let k_hop_value: Value = serde_json::from_slice(&k_hop_body).unwrap();
        assert_eq!(k_hop_value["result"]["implementation_authority"], "rust");
        assert_eq!(k_hop_value["result"]["seam_attestation_version"], 1);
        assert_eq!(k_hop_value["result"]["seam_crossings"], json!([]));
        assert_eq!(k_hop_value["result"]["layers"][0]["nodes"], json!([]));
        assert_eq!(
            k_hop_value["result"]["layers"][0]["edges"],
            json!(["edge-1"])
        );

        let acl_fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/graph-acl-traversal-v1.json"
        )))
        .unwrap();
        let mut acl_fixture_batch = json!({
            "contract_version": 1,
            "atomic": true,
            "mutations": [],
            "index_jobs": []
        });
        let template = &body["mutations"][0];
        for node in acl_fixture["nodes"].as_array().unwrap() {
            let id = node["id"].as_str().unwrap();
            let mut mutation = template.clone();
            mutation["event_id"] = json!(format!("acl-fixture-node-{id}"));
            mutation["scope"]["namespace"] = json!("alpha");
            mutation["payload"] = json!({"id": id});
            mutation["record"]["id"] = json!(id);
            mutation["record"]["document"] = json!(
                serde_json::to_string(&json!({"label": id})).unwrap()
            );
            mutation["record"]["metadata"] = json!({});
            mutation["record"]["embedding"] = json!([1.0, 0.0, 0.0]);
            acl_fixture_batch["mutations"]
                .as_array_mut()
                .unwrap()
                .push(mutation);
        }
        let edge_fixture = &acl_fixture["edge"];
        let edge_id = edge_fixture["id"].as_str().unwrap();
        let mut edge_mutation = template.clone();
        edge_mutation["table"] = json!("gke_edges");
        edge_mutation["entity_kind"] = json!("edge");
        edge_mutation["event_id"] = json!(format!("acl-fixture-edge-{edge_id}"));
        edge_mutation["scope"]["namespace"] = json!("alpha");
        edge_mutation["payload"] = json!({
            "id": edge_id,
            "source_ids": edge_fixture["source_ids"],
            "target_ids": edge_fixture["target_ids"],
            "source_edge_ids": [],
            "target_edge_ids": []
        });
        edge_mutation["record"]["id"] = json!(edge_id);
        edge_mutation["record"]["document"] = json!(serde_json::to_string(&json!({
            "relation": edge_fixture["relation"],
            "source_ids": edge_fixture["source_ids"],
            "target_ids": edge_fixture["target_ids"],
            "source_edge_ids": [],
            "target_edge_ids": []
        })).unwrap());
        edge_mutation["record"]["metadata"] = json!({});
        edge_mutation["record"]["embedding"] = json!([0.0, 1.0, 0.0]);
        acl_fixture_batch["mutations"]
            .as_array_mut()
            .unwrap()
            .push(edge_mutation);

        for (grain, entity) in acl_fixture["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|node| ("node", node))
            .chain(std::iter::once(("edge", edge_fixture)))
        {
            let target_id = entity["id"].as_str().unwrap();
            let mut acl_record = template.clone();
            let acl_id = format!("acl-fixture-{grain}-{target_id}");
            let mut metadata = json!({
                "entity_type": "acl_record",
                "acl_truth_graph": acl_fixture["truth_graph"],
                "acl_target_grain": grain,
                "acl_target_item_id": "",
                "acl_target_entity_id": target_id,
                "acl_version": 1,
                "acl_mode": entity["mode"],
                "tombstoned": false
            });
            if let Some(owner_id) = entity["owner_id"].as_str() {
                metadata["owner_id"] = json!(owner_id);
            }
            acl_record["event_id"] = json!(format!("acl-fixture-event-{acl_id}"));
            acl_record["scope"]["namespace"] = json!("alpha");
            acl_record["payload"] = json!({"id": acl_id});
            acl_record["record"]["id"] = json!(acl_id);
            acl_record["record"]["metadata"] = metadata;
            acl_record["record"]["document"] = Value::Null;
            acl_record["record"]["embedding"] = Value::Null;
            acl_fixture_batch["mutations"]
                .as_array_mut()
                .unwrap()
                .push(acl_record);
        }
        let acl_write = send_mutation(
            make_app(),
            make_token(&["write_graph"], "alpha"),
            acl_fixture_batch,
        )
        .await;
        assert_eq!(acl_write.status(), StatusCode::OK);

        for principal in acl_fixture["principals"].as_array().unwrap() {
            let subject = principal["id"].as_str().unwrap();
            let response = make_app()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/mcp")
                        .header("content-type", "application/json")
                        .header(
                            "authorization",
                            format!(
                                "Bearer {}",
                                make_token_for(subject, &["read_graph"], "alpha")
                            ),
                        )
                        .body(Body::from(format!(
                            r#"{{"jsonrpc":"2.0","id":"acl-{subject}","method":"tools/call","params":{{"name":"kg_neighbors","arguments":{{"rid":"source-public"}}}}}}"#
                        )))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let response_body = to_bytes(response.into_body(), 65_536).await.unwrap();
            let response_value: Value = serde_json::from_slice(&response_body).unwrap();
            assert_eq!(
                response_value["result"]["nodes"],
                principal["expected_neighbors"]["nodes"],
                "unexpected visible nodes for {subject}"
            );
            assert_eq!(
                response_value["result"]["edges"],
                principal["expected_neighbors"]["edges"],
                "unexpected visible edges for {subject}"
            );
        }
    }

    #[tokio::test]
    #[ignore = "requires KOGWISTAR_TEST_PG_DSN; run explicitly with --include-ignored"]
    async fn postgres_public_mcp_graph_reads_cross_projection_page_boundary() {
        use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};

        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/graph-traversal-v1.json"
        )))
        .unwrap();
        let cases = fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .map(|case| (case["name"].as_str().unwrap(), case))
            .collect::<std::collections::BTreeMap<_, _>>();
        let source_id = "z-source";
        let target_id = "z-target";
        let edge_id = "z-edge";
        let map_fixture_id = |id: &str| match id {
            "A" => source_id.to_owned(),
            "B" => target_id.to_owned(),
            "E1" => edge_id.to_owned(),
            other => other.to_owned(),
        };
        let map_fixture_ids = |ids: &Value| {
            ids.as_array()
                .unwrap()
                .iter()
                .map(|id| map_fixture_id(id.as_str().unwrap()))
                .collect::<Vec<_>>()
        };

        let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN").unwrap();
        let schema = format!("rust_mcp_graph_pages_{}", uuid::Uuid::new_v4().simple());
        let service = PostgresRunApplicationService::from_dsn(&dsn, &schema).unwrap();
        service.ensure_schema().await.unwrap();
        service
            .store
            .create_graph_schema(3, kogwistar_store_postgres::GraphTableNames::default())
            .await
            .unwrap();
        let scope = GraphScope {
            namespace: "alpha".to_owned(),
            workspace_id: None,
            graph_space: None,
        };
        for (table, id, document, metadata) in [
            ("gke_nodes", source_id, json!({"label":"Smoking","doc_id":"D1"}), json!({})),
            ("gke_nodes", target_id, json!({"label":"Lung Cancer","doc_id":"D1"}), json!({})),
            ("gke_nodes", "z-third", json!({"label":"Third node","doc_id":"D1"}), json!({})),
            ("gke_nodes", "z-d2-target", json!({"label":"Lung Cancer","doc_id":"D2"}), json!({})),
            ("gke_edges", edge_id, json!({"relation":"causes","source_ids":[source_id],"target_ids":[target_id],"doc_id":"D1"}), json!({})),
            ("gke_edges", "z-d2-edge", json!({"relation":"causes","source_ids":[source_id],"target_ids":["z-d2-target"],"doc_id":"D2"}), json!({})),
            ("gke_nodes", "acl-source", Value::Null, json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"node","acl_target_entity_id":source_id,"acl_version":1,"acl_mode":"public"})),
            ("gke_nodes", "acl-target", Value::Null, json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"node","acl_target_entity_id":target_id,"acl_version":1,"acl_mode":"public"})),
            ("gke_nodes", "acl-third", Value::Null, json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"node","acl_target_entity_id":"z-third","acl_version":1,"acl_mode":"public"})),
            ("gke_nodes", "acl-d2-target", Value::Null, json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"node","acl_target_entity_id":"z-d2-target","acl_version":1,"acl_mode":"public"})),
            ("gke_nodes", "acl-edge", Value::Null, json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"edge","acl_target_entity_id":edge_id,"acl_version":1,"acl_mode":"public"})),
            ("gke_nodes", "acl-d2-edge", Value::Null, json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"edge","acl_target_entity_id":"z-d2-edge","acl_version":1,"acl_mode":"public"})),
            ("gke_nodes", "acl-new-edge", Value::Null, json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"edge","acl_target_entity_id":"z-new-edge","acl_version":1,"acl_mode":"public"})),
        ] {
            service.store.upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                scope: scope.clone(),
                table: table.to_owned(),
                record: GraphRecord {
                    id: id.to_owned(),
                    document: (!document.is_null()).then(|| document.to_string()),
                    metadata: metadata.as_object().unwrap().clone(),
                    embedding: None,
                },
                embedding_dim: 3,
            }).await.unwrap();
        }
        let (mut client, connection) = tokio_postgres::connect(&dsn, tokio_postgres::NoTls).await.unwrap();
        let connection_task = tokio::spawn(connection);
        let (observer, observer_connection) =
            tokio_postgres::connect(&dsn, tokio_postgres::NoTls).await.unwrap();
        let observer_task = tokio::spawn(observer_connection);
        client.execute(
            &format!("INSERT INTO \"{schema}\".gke_nodes(id, metadata) SELECT 'm' || lpad(n::text, 5, '0'), '{{\"namespace\":\"alpha\"}}'::jsonb FROM generate_series(0, 10000) AS n"),
            &[],
        ).await.unwrap();
        client.execute(
            &format!("INSERT INTO \"{schema}\".gke_edges(id, metadata) SELECT 'm' || lpad(n::text, 5, '0'), '{{\"namespace\":\"alpha\"}}'::jsonb FROM generate_series(0, 10000) AS n"),
            &[],
        ).await.unwrap();
        let key = "graph-pages-test-secret";
        let token = encode(
            &Header::new(Algorithm::HS256),
            &json!({"sub":"alice","role":"ro","roles":["reader"],"ns":"workflow","storage_ns":"alpha","capabilities":["read_graph"],"exp":4102444800_u64}),
            &EncodingKey::from_secret(key.as_bytes()),
        ).unwrap();
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: vec!["reader".to_owned()],
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig {mode:"jwt".to_owned(), algorithm:Some("HS256".to_owned()), key:Some(key.to_owned()), ..AuthConfig::default()},
            },
            Arc::new(service.clone()),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, app).await });
        let http = reqwest::Client::new();

        // Hold the edge table while the reader establishes its repeatable-read
        // snapshot on nodes. Commit a new edge only after its edge-page SELECT
        // is waiting on the lock; that same read must not observe the edge.
        let transaction = client.transaction().await.unwrap();
        transaction
            .batch_execute(&format!("LOCK TABLE \"{schema}\".gke_edges IN ACCESS EXCLUSIVE MODE"))
            .await
            .unwrap();
        transaction
            .execute(
                &format!("INSERT INTO \"{schema}\".gke_edges(id, document, metadata) VALUES ('z-new-edge', $1, '{{\"namespace\":\"alpha\"}}'::jsonb)"),
                &[&json!({"relation":"causes","source_ids":[source_id],"target_ids":["z-third"],"doc_id":"D1"}).to_string()],
            )
            .await
            .unwrap();
        let concurrent_read = {
            let http = http.clone();
            let token = token.clone();
            tokio::spawn(async move {
                http.post(format!("http://{address}/mcp"))
                    .bearer_auth(token)
                    .json(&json!({
                        "jsonrpc": "2.0",
                        "id": "snapshot-before-edge-commit",
                        "method": "tools/call",
                        "params": {"name": "kg_neighbors", "arguments": {"rid": source_id, "doc_id": "D1"}}
                    }))
                    .send()
                    .await
                    .unwrap()
                    .json::<Value>()
                    .await
                    .unwrap()
            })
        };
        let waiting_for_edge_page = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            async {
                loop {
                    let waiting: bool = observer
                        .query_one(
                            "SELECT EXISTS (SELECT 1 FROM pg_stat_activity WHERE datname = current_database() AND state = 'active' AND wait_event_type = 'Lock' AND query LIKE '%SELECT id, document, metadata::TEXT, embedding::TEXT FROM%' AND query LIKE '%gke_edges%')",
                            &[],
                        )
                        .await
                        .unwrap()
                        .get(0);
                    if waiting {
                        break;
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(2)).await;
                }
            },
        )
        .await;
        assert!(waiting_for_edge_page.is_ok(), "reader did not reach the locked edge projection query");
        transaction.commit().await.unwrap();
        let first_read = concurrent_read.await.unwrap();
        assert!(first_read.get("error").is_none(), "{first_read}");
        assert_eq!(first_read["result"]["implementation_authority"], "rust");
        assert_eq!(first_read["result"]["seam_attestation_version"], 1);
        assert_eq!(first_read["result"]["seam_crossings"], json!([]));
        assert!(!first_read["result"]["nodes"].as_array().unwrap().contains(&json!("z-third")));

        let second_read: Value = http
            .post(format!("http://{address}/mcp"))
            .bearer_auth(&token)
            .json(&json!({
                "jsonrpc": "2.0",
                "id": "snapshot-after-edge-commit",
                "method": "tools/call",
                "params": {"name": "kg_neighbors", "arguments": {"rid": source_id, "doc_id": "D1"}}
            }))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert!(second_read["result"]["nodes"].as_array().unwrap().contains(&json!("z-third")));
        assert_eq!(second_read["result"]["implementation_authority"], "rust");
        assert_eq!(second_read["result"]["seam_attestation_version"], 1);
        assert_eq!(second_read["result"]["seam_crossings"], json!([]));
        client
            .execute(
                &format!("DELETE FROM \"{schema}\".gke_edges WHERE id = 'z-new-edge'"),
                &[],
            )
            .await
            .unwrap();

        let neighbor_case = cases["edge_to_nodes"];
        let find_case = cases["find_edges"];
        let path_case = cases["shortest_path"];
        let hop_case = cases["k_hop"];
        for (name, arguments, expected_field, expected) in [
            (
                "kg_neighbors",
                json!({"rid":edge_id,"doc_id":"D1"}),
                "nodes",
                json!(map_fixture_ids(&neighbor_case["expected_neighbors"]["nodes"])),
            ),
            (
                "kg_find_edges",
                json!({"relation":find_case["relation"],"src_label_contains":find_case["source_label_contains"],"tgt_label_contains":find_case["target_label_contains"],"doc_id":find_case["doc_id"]}),
                "edges",
                json!(map_fixture_ids(&find_case["expected_edge_ids"])),
            ),
            (
                "kg_shortest_path",
                json!({"src_id":source_id,"dst_id":target_id,"doc_id":"D1"}),
                "path",
                json!(map_fixture_ids(&path_case["expected_path"])),
            ),
            (
                "kg_k_hop",
                json!({"start_ids":[source_id],"k":hop_case["k"],"doc_id":"D1"}),
                "layers",
                Value::Null,
            ),
        ] {
            let response = http
                .post(format!("http://{address}/mcp"))
                .bearer_auth(&token)
                .json(&json!({
                    "jsonrpc": "2.0",
                    "id": name,
                    "method": "tools/call",
                    "params": {"name": name, "arguments": arguments}
                }))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let payload: Value = response.json().await.unwrap();
            assert!(payload.get("error").is_none(), "{name}: {payload}");
            assert_eq!(payload["result"]["implementation_authority"], "rust");
            assert_eq!(payload["result"]["seam_attestation_version"], 1);
            assert_eq!(payload["result"]["seam_crossings"], json!([]));
            if name == "kg_k_hop" {
                let expected_layers = hop_case["expected_layers"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|layer| {
                        json!({
                            "nodes": map_fixture_ids(&layer["nodes"]),
                            "edges": map_fixture_ids(&layer["edges"])
                        })
                    })
                    .collect::<Vec<_>>();
                assert_eq!(payload["result"]["layers"], json!(expected_layers));
            } else {
                assert_eq!(payload["result"][expected_field], expected, "{name}");
                if name == "kg_neighbors" {
                    assert_eq!(
                        payload["result"]["edges"],
                        json!(map_fixture_ids(&neighbor_case["expected_neighbors"]["edges"]))
                    );
                }
            }
        }
        server.abort();
        let _ = server.await;
        drop(service);
        client.batch_execute(&format!("DROP SCHEMA \"{schema}\" CASCADE")).await.unwrap();
        drop(client);
        drop(observer);
        observer_task.await.unwrap().unwrap();
        connection_task.await.unwrap().unwrap();
    }

    #[tokio::test]
    #[ignore = "requires KOGWISTAR_TEST_PG_DSN; run explicitly with --include-ignored"]
    async fn postgres_document_projection_route_persists_reopens_and_retries_when_dsn_available() {
        let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
            .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
        let schema = format!("rust_api_document_{}", uuid::Uuid::new_v4().simple());
        let (client, connection) = tokio_postgres::connect(&dsn, tokio_postgres::NoTls)
            .await
            .unwrap();
        tokio::spawn(async move {
            let _ = connection.await;
        });
        client
            .batch_execute(&format!(
                r#"CREATE EXTENSION IF NOT EXISTS vector;
                CREATE SCHEMA "{schema}";
                CREATE TABLE "{schema}"."gke_documents" (
                    id TEXT PRIMARY KEY,
                    document TEXT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding vector(3) NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );"#
            ))
            .await
            .unwrap();

        let service = PostgresRunApplicationService::from_dsn(&dsn, &schema).unwrap();
        service.ensure_schema().await.unwrap();
        let key = "document-route-test-secret";
        let token = encode(
            &Header::new(Algorithm::HS256),
            &json!({
                "sub": "alice",
                "role": "rw",
                "roles": ["reader", "rw"],
                "ns": "workflow",
                "storage_ns": "docs",
                "workspace": "workspace-a",
                "graph_space": "knowledge",
                "capabilities": ["write_graph"],
                "exp": 4102444800_u64
            }),
            &EncodingKey::from_secret(key.as_bytes()),
        )
        .unwrap();
        let make_app = || {
            router_with_application(
                ApiState {
                    health: snapshot(),
                    required_roles: vec!["reader".to_owned()],
                    implementation: ImplementationSnapshot::default(),
                    auth: AuthConfig {
                        mode: "jwt".to_owned(),
                        algorithm: Some("HS256".to_owned()),
                        key: Some(key.to_owned()),
                        ..AuthConfig::default()
                    },
                },
                Arc::new(service.clone()),
            )
        };
        let send = |app: Router, body: Value| {
            let token = token.clone();
            async move {
                app.oneshot(
                    Request::builder()
                        .method("POST")
                        .uri(INTERNAL_DOCUMENT_PROJECTION_ROUTE)
                        .header("content-type", "application/json")
                        .header("authorization", format!("Bearer {token}"))
                        .body(Body::from(serde_json::to_vec(&body).unwrap()))
                        .unwrap(),
                )
                .await
                .unwrap()
            }
        };

        let first = send(
            make_app(),
            json!({
                "doc_id": "document-1",
                "content": "first body",
                "doc_type": "text",
                "insertion_method": "route-test"
            }),
        )
        .await;
        assert_eq!(first.status(), StatusCode::OK);
        let retry = send(
            make_app(),
            json!({
                "doc_id": "document-1",
                "content": "updated body",
                "doc_type": "text",
                "insertion_method": "route-test"
            }),
        )
        .await;
        assert_eq!(retry.status(), StatusCode::OK);

        let reopened = PostgresRunApplicationService::from_dsn(&dsn, &schema).unwrap();
        let rows = reopened
            .store
            .graph_projection_records(kogwistar_store::GraphProjectionRead {
                scope: GraphScope {
                    namespace: "docs".to_owned(),
                    workspace_id: Some("workspace-a".to_owned()),
                    graph_space: Some("knowledge".to_owned()),
                },
                table: "gke_documents".to_owned(),
                ids: Some(vec!["document-1".to_owned()]),
                metadata: MetadataFilter::default(),
                limit: 10,
            })
            .await
            .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].document.as_deref(), Some("updated body"));
        assert_eq!(rows[0].metadata["namespace"], "docs");
        assert_eq!(rows[0].metadata["workspace_id"], "workspace-a");
        assert_eq!(rows[0].metadata["graph_space"], "knowledge");
        assert_eq!(rows[0].metadata["insertion_method"], "route-test");

        let (cleanup, cleanup_connection) = tokio_postgres::connect(&dsn, tokio_postgres::NoTls)
            .await
            .unwrap();
        tokio::spawn(async move {
            let _ = cleanup_connection.await;
        });
        cleanup
            .batch_execute(&format!("DROP SCHEMA \"{schema}\" CASCADE"))
            .await
            .unwrap();
    }

    #[tokio::test]
    #[ignore = "requires KOGWISTAR_TEST_PG_DSN; run explicitly with --include-ignored"]
    async fn postgres_design_history_folds_and_repairs_projection_when_dsn_available() {
        use kogwistar_store_postgres::NewRawEntityEvent;

        let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
            .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
        let schema = format!("rust_api_design_history_{}", uuid::Uuid::new_v4().simple());
        let service = PostgresRunApplicationService::from_dsn(&dsn, &schema).unwrap();
        service.ensure_schema().await.unwrap();
        for (event_id, op, payload) in [
            (
                "commit-1",
                "MUTATION_COMMITTED",
                json!({"version": 1, "prev_version": 0, "target_seq": 1, "ts_ms": 10, "designer_id": "alice", "action": "node_upsert", "entity_id": "n1"}),
            ),
            (
                "commit-2",
                "MUTATION_COMMITTED",
                json!({"version": 2, "prev_version": 1, "target_seq": 2, "ts_ms": 20, "designer_id": "alice", "action": "edge_upsert", "entity_id": "e2"}),
            ),
            (
                "undo-1",
                "UNDO_APPLIED",
                json!({"from_version": 2, "to_version": 1, "ts_ms": 30, "designer_id": "alice"}),
            ),
        ] {
            service
                .store
                .append_raw_entity_event(
                    "wf_design:wf-pg-history",
                    NewRawEntityEvent {
                        event_id: event_id.to_owned(),
                        entity_kind: "design_control".to_owned(),
                        entity_id: "wf-pg-history".to_owned(),
                        op: op.to_owned(),
                        payload_json: payload.to_string(),
                    },
                )
                .await
                .unwrap();
        }
        let response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: "/api/workflow/design/wf-pg-history/history".to_owned(),
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
        assert_eq!(history["current_version"], 1);
        assert_eq!(history["active_tip_version"], 2);
        assert_eq!(history["can_redo"], true);
        assert_eq!(history["latest_seq"], 3);
        let projection = service
            .store
            .get_named_projection("workflow_design", "wf-pg-history")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(projection.payload["current_version"], 1);
        assert_eq!(projection.payload["active_tip_version"], 2);
        assert_eq!(projection.last_authoritative_seq, 3);
        assert_eq!(projection.last_materialized_seq, 1);
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
    #[ignore = "requires KOGWISTAR_TEST_PG_DSN; run explicitly with --include-ignored"]
    async fn postgres_service_events_fold_authoritative_history_when_dsn_available() {
        use kogwistar_store_postgres::NewRawEntityEvent;

        let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
            .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
        let schema = format!("rust_api_service_events_{}", uuid::Uuid::new_v4().simple());
        let service = PostgresRunApplicationService::from_dsn(&dsn, &schema).unwrap();
        service.ensure_schema().await.unwrap();
        for (event_id, entity_id, op, payload) in [
            (
                "service-event-1",
                "service_evt:svc-pg:1",
                "ADD",
                json!({
                    "id": "service_evt:svc-pg:1",
                    "metadata": {
                        "entity_type": "service_event",
                        "service_id": "svc-pg",
                        "service_event_type": "service.starting",
                        "ts_ms": 20,
                    },
                    "properties": {"payload_json": r#"{"phase":"old"}"#},
                }),
            ),
            (
                "service-event-2",
                "service_evt:svc-pg:1",
                "REPLACE",
                json!({
                    "id": "service_evt:svc-pg:1",
                    "metadata": {
                        "entity_type": "service_event",
                        "service_id": "svc-pg",
                        "service_event_type": "service.started",
                        "ts_ms": 20,
                    },
                    "properties": {"payload_json": r#"{"phase":"new"}"#},
                }),
            ),
            (
                "service-event-other",
                "service_evt:other:1",
                "ADD",
                json!({
                    "id": "service_evt:other:1",
                    "metadata": {
                        "entity_type": "service_event",
                        "service_id": "other",
                        "service_event_type": "service.healthy",
                        "ts_ms": 1,
                    },
                    "properties": {"payload_json": "{}"},
                }),
            ),
        ] {
            service
                .store
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
                .await
                .unwrap();
        }
        let response = service
            .execute_async(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: "/api/workflow/services/svc-pg/events?limit=10".to_owned(),
                body: Vec::new(),
                principal: json!({
                    "sub": "alice",
                    "role": "ro",
                    "ns": "workflow",
                    "capabilities": ["service.inspect", "project_view"],
                }),
            })
            .await;
        assert_eq!(response.status, StatusCode::OK.as_u16());
        let body = serde_json::from_slice::<Value>(&response.body).unwrap();
        assert_eq!(body["service_id"], "svc-pg");
        assert_eq!(body["events"].as_array().unwrap().len(), 1);
        assert_eq!(body["events"][0]["event_type"], "service.started");
        assert_eq!(body["events"][0]["payload"]["phase"], "new");
        let principal = json!({
            "sub": "operator",
            "role": "rw",
            "ns": "workflow",
            "capabilities": ["service.manage", "service.inspect", "project_view"],
        });
        let declared = service
            .execute_async(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/workflow/services".to_owned(),
                body: serde_json::to_vec(&json!({
                    "service_id": "svc-pg-trigger",
                    "target_kind": "workflow",
                    "target_ref": "wf-pg-trigger",
                    "target_config": {
                        "conversation_id":"conv-pg-trigger",
                        "runtime_kind":"async"
                    },
                }))
                .unwrap(),
                principal: principal.clone(),
            })
            .await;
        assert_eq!(declared.status, StatusCode::OK.as_u16());
        let trigger = || ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/workflow/services/svc-pg-trigger/trigger".to_owned(),
            body: serde_json::to_vec(&json!({
                "trigger_type": "external event",
                "payload": {"source":"postgres-live"},
            }))
            .unwrap(),
            principal: principal.clone(),
        };
        let service_clone = service.clone();
        let (first, second) = tokio::join!(
            service.execute_async(trigger()),
            service_clone.execute_async(trigger()),
        );
        assert_eq!(first.status, StatusCode::OK.as_u16());
        assert_eq!(second.status, StatusCode::OK.as_u16());
        let first = serde_json::from_slice::<Value>(&first.body).unwrap();
        let second = serde_json::from_slice::<Value>(&second.body).unwrap();
        let run_id = first["current_child_run_id"].as_str().unwrap();
        assert_eq!(second["current_child_run_id"], run_id);
        assert!(
            service
                .store
                .get_server_run(run_id)
                .await
                .unwrap()
                .is_some()
        );
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
        let payload: Value = serde_json::from_str(
            lanes[0]
                .payload_json
                .as_deref()
                .expect("service lane payload"),
        )
        .unwrap();
        assert_eq!(payload["worker_protocol"], "async-v2");
        assert_eq!(payload["initial_state"]["_rt_worker_protocol"], "async-v2");
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
