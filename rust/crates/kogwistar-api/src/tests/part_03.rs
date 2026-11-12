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
        let denied_tokens = [
            token(json!({
                "sub": "reader-no-visibility-capability",
                "role": "ro",
                "ns": ["workflow"],
                "capabilities": ["project_view"],
                "iss": "local",
                "exp": now + 300,
            })),
            token(json!({
                "sub": "reader-wrong-visibility-namespace",
                "role": "ro",
                "ns": ["conversation"],
                "capabilities": ["project_view", "read_security_scope"],
                "iss": "local",
                "exp": now + 300,
            })),
        ];
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
            visibility["storage_security_mapping"],
            json!({
                "storage_namespace":"workflow",
                "execution_namespace":"workflow",
                "security_scope":"tenant-a",
                "security_scope_path":"tenant-a/workspace-b/project-c",
                "tenant":"tenant-a",
                "workspace":"workspace-b",
                "project":"project-c"
            })
        );

        let capabilities = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/workflow/capabilities")
                    .header("authorization", format!("Bearer {read_token}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let capabilities_contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/workflow-capabilities-http-v1.json"
        ))
        .unwrap();
        assert_eq!(
            capabilities.status().as_u16(),
            capabilities_contract["success_status"].as_u64().unwrap() as u16
        );
        let capabilities = serde_json::from_slice::<Value>(
            &to_bytes(capabilities.into_body(), 32768).await.unwrap(),
        )
        .unwrap();
        for field in capabilities_contract["required_fields"].as_array().unwrap() {
            let field = field.as_str().unwrap();
            assert!(capabilities.get(field).is_some(), "missing {field}");
        }
        assert_eq!(capabilities["current_subject"], "reader-1");
        for capability in capabilities_contract["required_effective_capabilities"]
            .as_array()
            .unwrap()
        {
            assert!(capabilities["effective_capabilities"]
                .as_array()
                .unwrap()
                .contains(capability));
        }
        assert_eq!(
            capabilities["audit_log"].as_array().unwrap().last().unwrap()["action"],
            capabilities_contract["audit_action"]
        );
        assert_eq!(
            capabilities["audit_log"].as_array().unwrap().last().unwrap()["outcome"],
            capabilities_contract["allowed_outcome"]
        );

        let denied_capabilities = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/workflow/capabilities")
                    .header("authorization", format!("Bearer {}", denied_tokens[0]))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(denied_capabilities.status(), StatusCode::FORBIDDEN);

        let syscall_catalog_contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/syscall-catalog-http-v1.json"
        ))
        .unwrap();
        let syscall_catalog = app
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
        assert_eq!(
            syscall_catalog.status().as_u16(),
            syscall_catalog_contract["success_status"].as_u64().unwrap() as u16
        );
        let syscall_catalog: Value = serde_json::from_slice(
            &to_bytes(syscall_catalog.into_body(), 32768).await.unwrap(),
        )
        .unwrap();
        assert_eq!(syscall_catalog, syscall_catalog_contract["response_body"]);

        let budget_contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/workflow-budget-http-v1.json"
        ))
        .unwrap();
        let budget = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/workflow/budget")
                    .header("authorization", format!("Bearer {read_token}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            budget.status().as_u16(),
            budget_contract["success_status"].as_u64().unwrap() as u16
        );
        let budget: Value =
            serde_json::from_slice(&to_bytes(budget.into_body(), 32768).await.unwrap()).unwrap();
        for field in budget_contract["required_fields"].as_array().unwrap() {
            let field = field.as_str().unwrap();
            assert!(budget.get(field).is_some(), "missing {field}");
        }
        for field in budget_contract["required_cost_ledger_fields"].as_array().unwrap() {
            let field = field.as_str().unwrap();
            assert!(budget["cost_ledger"].get(field).is_some(), "missing {field}");
        }
        assert_eq!(budget["budget_model"], budget_contract["budget_model"]);

        let budget_history_contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/workflow-budget-history-http-v1.json"
        ))
        .unwrap();
        let budget_history = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/workflow/budget/history?limit=1")
                    .header("authorization", format!("Bearer {read_token}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            budget_history.status().as_u16(),
            budget_history_contract["success_status"].as_u64().unwrap() as u16
        );
        let budget_history: Value = serde_json::from_slice(
            &to_bytes(budget_history.into_body(), 32768).await.unwrap(),
        )
        .unwrap();
        for field in budget_history_contract["required_fields"].as_array().unwrap() {
            let field = field.as_str().unwrap();
            assert!(budget_history.get(field).is_some(), "missing {field}");
        }
        for field in budget_history_contract["required_cost_ledger_fields"]
            .as_array()
            .unwrap()
        {
            let field = field.as_str().unwrap();
            assert!(budget_history["cost_ledger"].get(field).is_some(), "missing {field}");
        }
        assert!(budget_history["events"].is_array());
        for event in budget_history["events"].as_array().unwrap() {
            for field in budget_history_contract["required_event_fields"]
                .as_array()
                .unwrap()
            {
                let field = field.as_str().unwrap();
                assert!(event.get(field).is_some(), "event missing {field}: {event}");
            }
        }

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
        assert_eq!(
            denied.status().as_u16(),
            capabilities_contract["denied_status"].as_u64().unwrap() as u16
        );
        let budget_missing_scope_token = token(json!({
            "sub": "budget-reader-without-security-scope",
            "role": "ro",
            "ns": ["workflow"],
            "capabilities": ["project_view"],
            "iss": "local",
            "exp": now + 300,
        }));
        let denied_budget = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/workflow/budget")
                    .header("authorization", format!("Bearer {budget_missing_scope_token}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            denied_budget.status().as_u16(),
            budget_contract["denied_status"].as_u64().unwrap() as u16
        );
        let denied_budget_history = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/workflow/budget/history")
                    .header("authorization", format!("Bearer {budget_missing_scope_token}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            denied_budget_history.status().as_u16(),
            budget_history_contract["denied_status"].as_u64().unwrap() as u16
        );

        let admin_token = token(json!({
            "sub": "admin-1",
            "role": "rw",
            "ns": ["workflow"],
            "capabilities": ["project_view", "read_security_scope", "approve_action"],
            "iss": "local",
            "exp": now + 300,
        }));
        let capability_approve_contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/workflow-capability-approve-http-v1.json"
        ))
        .unwrap();
        let approval_without_view_token = token(json!({
            "sub": "limited-approver",
            "role": "rw",
            "ns": ["workflow"],
            "capabilities": ["approve_action"],
            "iss": "local",
            "exp": now + 300,
        }));
        let denied_approval = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/workflow/capabilities/approve")
                    .header("authorization", format!("Bearer {approval_without_view_token}"))
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "action": "must_not_be_granted",
                            "capabilities": ["workflow.run.write"],
                            "subject": "target-denied"
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(denied_approval.status(), StatusCode::FORBIDDEN);
        let unchanged = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/workflow/capabilities")
                    .header("authorization", format!("Bearer {admin_token}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let unchanged: Value =
            serde_json::from_slice(&to_bytes(unchanged.into_body(), 32768).await.unwrap())
                .unwrap();
        assert!(!unchanged["approvals"].as_array().unwrap().iter().any(|approval| {
            approval["subject"] == "target-denied"
                && approval["action"] == "must_not_be_granted"
        }));
        let approve = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/workflow/capabilities/approve")
                    .header("authorization", format!("Bearer {admin_token}"))
                    .header("content-type", "application/json")
                    .body(Body::from(capability_approve_contract["request"].to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            approve.status().as_u16(),
            capability_approve_contract["http_status"]
                .as_u64()
                .unwrap() as u16
        );
        let approve =
            serde_json::from_slice::<Value>(&to_bytes(approve.into_body(), 32768).await.unwrap())
                .unwrap();
        for field in capability_approve_contract["required_body_fields"]
            .as_array()
            .unwrap()
        {
            let field = field.as_str().unwrap();
            assert!(approve.get(field).is_some(), "missing {field}");
        }
        assert_eq!(
            approve["current_subject"],
            capability_approve_contract["expected_subject"]
        );
        assert!(approve["approvals"]
            .as_array()
            .unwrap()
            .contains(&capability_approve_contract["expected_approval"]));
        for capability in capability_approve_contract["effective_capabilities_include"]
            .as_array()
            .unwrap()
        {
            assert!(approve["effective_capabilities"]
                .as_array()
                .unwrap()
                .contains(capability));
        }
        assert!(
            approve["effective_capabilities"]
                .as_array()
                .unwrap()
                .contains(&json!("spawn_process"))
        );
        let approve_audit = approve["audit_log"].as_array().unwrap();
        let approve_audit_actions = approve_audit
            .iter()
            .rev()
            .take(2)
            .rev()
            .map(|entry| entry["action"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(approve_audit_actions, ["approve_action", "project_view"]);
        assert_eq!(
            approve_audit.last().unwrap()["outcome"],
            "allow"
        );

        let capability_revoke_contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/workflow-capability-revoke-http-v1.json"
        ))
        .unwrap();
        let revoke_without_view_token = token(json!({
            "sub": "limited-revoker",
            "role": "rw",
            "ns": ["workflow"],
            "capabilities": ["approve_action"],
            "iss": "local",
            "exp": now + 300,
        }));
        let denied_revoke = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/workflow/capabilities/revoke")
                    .header("authorization", format!("Bearer {revoke_without_view_token}"))
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({"capability":"spawn_process","subject":"admin-1"})
                            .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(denied_revoke.status(), StatusCode::FORBIDDEN);
        let unchanged = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/workflow/capabilities")
                    .header("authorization", format!("Bearer {admin_token}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let unchanged: Value =
            serde_json::from_slice(&to_bytes(unchanged.into_body(), 32768).await.unwrap())
                .unwrap();
        assert!(!unchanged["revoked"].as_array().unwrap().iter().any(|revocation| {
            revocation["subject"] == "admin-1"
                && revocation["capabilities"]
                    .as_array()
                    .unwrap()
                    .contains(&json!("spawn_process"))
        }));
        let revoke = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/workflow/capabilities/revoke")
                    .header("authorization", format!("Bearer {admin_token}"))
                    .header("content-type", "application/json")
                    .body(Body::from(capability_revoke_contract["request"].to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            revoke.status().as_u16(),
            capability_revoke_contract["http_status"]
                .as_u64()
                .unwrap() as u16
        );
        let revoke =
            serde_json::from_slice::<Value>(&to_bytes(revoke.into_body(), 32768).await.unwrap())
                .unwrap();
        for field in capability_revoke_contract["required_body_fields"]
            .as_array()
            .unwrap()
        {
            let field = field.as_str().unwrap();
            assert!(revoke.get(field).is_some(), "missing {field}");
        }
        assert_eq!(
            revoke["current_subject"],
            capability_revoke_contract["expected_subject"]
        );
        assert!(revoke["revoked"]
            .as_array()
            .unwrap()
            .contains(&capability_revoke_contract["expected_revocation"]));
        for capability in capability_revoke_contract["effective_capabilities_exclude"]
            .as_array()
            .unwrap()
        {
            assert!(!revoke["effective_capabilities"]
                .as_array()
                .unwrap()
                .contains(capability));
        }
        let revoke_audit = revoke["audit_log"].as_array().unwrap();
        let revoke_audit_actions = revoke_audit
            .iter()
            .rev()
            .take(2)
            .rev()
            .map(|entry| entry["action"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(revoke_audit_actions, ["approve_action", "project_view"]);

        let self_revoke = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/workflow/capabilities/revoke")
                    .header("authorization", format!("Bearer {admin_token}"))
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({"capability":"project_view","subject":"admin-1"}).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(self_revoke.status(), StatusCode::ACCEPTED);
        let self_revoke: Value =
            serde_json::from_slice(&to_bytes(self_revoke.into_body(), 32768).await.unwrap())
                .unwrap();
        assert!(self_revoke["revoked"].as_array().unwrap().iter().any(|entry| {
            entry["subject"] == "admin-1"
                && entry["capabilities"]
                    .as_array()
                    .unwrap()
                    .contains(&json!("project_view"))
        }));
        assert!(!self_revoke["effective_capabilities"]
            .as_array()
            .unwrap()
            .contains(&json!("project_view")));

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

        for denied_token in denied_tokens {
            let denied = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/mcp/workflow")
                        .header("authorization", format!("Bearer {denied_token}"))
                        .header("content-type", "application/json")
                        .body(Body::from(
                            r#"{"jsonrpc":"2.0","id":8,"method":"tools/call","params":{"name":"workflow.visibility_snapshot","arguments":{}}}"#,
                        ))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(denied.status(), StatusCode::OK);
            let denied = serde_json::from_slice::<Value>(
                &to_bytes(denied.into_body(), 32768).await.unwrap(),
            )
            .unwrap();
            assert_eq!(denied["error"]["code"], -32602);
            assert!(denied.get("result").is_none());
            assert!(!denied.to_string().contains("tenant-a"));
        }

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
        let approval_contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/syscall-request-approval-result-v1.json"
        ))
        .unwrap();
        let blocked_approval = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/syscall/v1/request_approval")
                    .header("authorization", format!("Bearer {admin_token}"))
                    .header("content-type", "application/json")
                    .body(Body::from(approval_contract["request"].to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            blocked_approval.status().as_u16(),
            approval_contract["http_status"].as_u64().unwrap() as u16
        );
        let blocked_approval = serde_json::from_slice::<Value>(
            &to_bytes(blocked_approval.into_body(), 16_384)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(blocked_approval, approval_contract["response"]);
        let syscall_audit_contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/syscall-audit-http-v1.json"
        ))
        .unwrap();
        let approval_audit = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri(format!(
                        "/api/syscall/v1/audit?limit={}",
                        syscall_audit_contract["limit"].as_u64().unwrap()
                    ))
                    .header("authorization", format!("Bearer {read_token}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            approval_audit.status().as_u16(),
            syscall_audit_contract["success_status"].as_u64().unwrap() as u16
        );
        let approval_audit: Value = serde_json::from_slice(
            &to_bytes(approval_audit.into_body(), 16_384).await.unwrap(),
        )
        .unwrap();
        for field in syscall_audit_contract["required_fields"]
            .as_array()
            .unwrap()
        {
            let field = field.as_str().unwrap();
            assert!(approval_audit.get(field).is_some(), "missing {field}");
        }
        assert_eq!(approval_audit["version"], "v1");
        let events = approval_audit["events"].as_array().unwrap();
        assert_eq!(events.len(), 1);
        for field in syscall_audit_contract["required_event_fields"]
            .as_array()
            .unwrap()
        {
            let field = field.as_str().unwrap();
            assert!(events[0].get(field).is_some(), "missing event field {field}");
        }
        for (field, expected) in syscall_audit_contract["expected_event"]
            .as_object()
            .unwrap()
        {
            assert_eq!(&events[0][field], expected);
        }
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
        let contract_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../contracts/golden/workflow-designer-capabilities-schema-v1.json");
        let contract: Value = serde_json::from_slice(
            &std::fs::read(contract_path).expect("read shared designer schema contract"),
        )
        .expect("parse shared designer schema contract");
        assert_eq!(
            designer["schema_version"],
            contract["schema_version"]
        );
        assert_eq!(
            designer["node_types"][0]["metadata_schema"],
            contract["node_metadata_schema"]
        );
        assert_eq!(
            designer["edge_types"][0]["metadata_schema"],
            contract["edge_metadata_schema"]
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
        // PYTHON_PARITY_V1: MCP redo response matches Python's workflow MCP projection; rust=rust/crates/kogwistar-api/src/sections/http_transport.rs#mcp_effect_request; python=kogwistar/server/chat_mcp.py#build_workflow_mcp.workflow_design_redo; fixture=contracts/golden/workflow-design-redo-mcp-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/part_03.rs#sqlite_design_capability_mutates_graph_and_undoes_atomically; python_test=tests/mcp/test_chat_mcp_tools.py#test_mcp_workflow_design_redo_matches_shared_contract
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

    #[tokio::test]
    #[ignore = "requires KOGWISTAR_TEST_PG_DSN; run explicitly with --include-ignored"]
    async fn postgres_internal_graph_mutation_route_persists_atomically_when_dsn_available() {
        let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
            .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
        let schema = format!("rust_api_graph_mutation_{}", uuid::Uuid::new_v4().simple());
        let embedding_failure = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (mut client, connection) = tokio_postgres::connect(&dsn, tokio_postgres::NoTls)
            .await
            .unwrap();
        tokio::spawn(async move {
            let _ = connection.await;
        });
        let transaction = client.transaction().await.unwrap();
        transaction
            .query_one(
                "SELECT pg_advisory_xact_lock(hashtext('kogwistar.pgvector.extension'))",
                &[],
            )
            .await
            .unwrap();
        transaction
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
                );
                CREATE TABLE "{schema}"."gke_nodes" (
                    id TEXT PRIMARY KEY,
                    document TEXT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding vector(3) NULL,
                    CHECK (id <> 'public-rollback-node'),
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
        transaction.commit().await.unwrap();
        let embedding_calls = Arc::new(Mutex::new(Vec::new()));
        let embedding_profile = kogwistar_engine::embedding::EmbeddingProfile {
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
        };
        let service = PostgresRunApplicationService::from_dsn(&dsn, &schema)
            .unwrap()
            .with_embedding_provider(Arc::new(FixedEmbeddingProvider {
                profile: embedding_profile.clone(),
                vector: vec![0.0, 0.0, 1.0],
                calls: embedding_calls.clone(),
                fail: Some(embedding_failure.clone()),
            }));
        service.ensure_schema().await.unwrap();
        service
            .store
            .create_graph_schema(3, kogwistar_store_postgres::GraphTableNames::default())
            .await
            .unwrap();
        let embedding_storage_scope = service.embedding_storage_scope.as_ref().unwrap();
        let registered_profile = service
            .store
            .get_named_projection(EMBEDDING_PROFILE_NAMESPACE, embedding_storage_scope)
            .await
            .unwrap()
            .expect("server startup binds the configured profile");
        validate_embedding_profile_projection(
            &registered_profile,
            &embedding_profile,
            embedding_storage_scope,
        )
        .unwrap();
        let mut different_profile = embedding_profile.clone();
        different_profile.model = "other-model-same-dimension".to_owned();
        let mismatched_service = service
            .clone()
            .with_embedding_provider(Arc::new(FixedEmbeddingProvider {
                profile: different_profile,
                vector: vec![0.0, 0.0, 1.0],
                calls: Arc::new(Mutex::new(Vec::new())),
                fail: None,
            }));
        assert!(mismatched_service.ensure_schema().await.is_err());
        let legacy_schema = format!("{schema}_legacy");
        client
            .batch_execute(&format!(
                r#"CREATE SCHEMA "{legacy_schema}";
                CREATE TABLE "{legacy_schema}"."gke_nodes" (
                    id TEXT PRIMARY KEY,
                    document TEXT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding vector(3) NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                INSERT INTO "{legacy_schema}"."gke_nodes" (id, document)
                    VALUES ('legacy-row', 'existing vector store row');"#
            ))
            .await
            .unwrap();
        let legacy_service = PostgresRunApplicationService::from_dsn(&dsn, &legacy_schema)
            .unwrap()
            .with_embedding_provider(Arc::new(FixedEmbeddingProvider {
                profile: embedding_profile.clone(),
                vector: vec![0.0, 0.0, 1.0],
                calls: Arc::new(Mutex::new(Vec::new())),
                fail: None,
            }));
        let legacy_error = legacy_service.ensure_schema().await.unwrap_err();
        assert!(legacy_error.contains("unbound for non-empty PostgreSQL vector storage"));
        assert!(legacy_service
            .store
            .get_named_projection(
                EMBEDDING_PROFILE_NAMESPACE,
                legacy_service.embedding_storage_scope.as_ref().unwrap(),
            )
            .await
            .unwrap()
            .is_none());
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
        let make_docs_token_for = |subject: &str, capabilities: &[&str]| {
            encode(
                &Header::new(Algorithm::HS256),
                &json!({
                    "sub": subject,
                    "role": "rw",
                    "roles": ["reader", "rw"],
                    "ns": "docs",
                    "storage_ns": "alpha",
                    "capabilities": capabilities,
                    "exp": 4102444800_u64
                }),
                &EncodingKey::from_secret(key.as_bytes()),
            )
            .unwrap()
        };
        let make_docs_token = |capabilities: &[&str]| make_docs_token_for("alice", capabilities);
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
        let graph_upsert_token_for = |subject: &str| {
            encode(
                &Header::new(Algorithm::HS256),
                &json!({
                "sub": subject,
                "role": "rw",
                "roles": ["reader", "rw"],
                "ns": "workflow",
                "storage_ns": "alpha",
                "workspace": "workspace-a",
                "graph_space": "facts",
                "capabilities": ["write_graph"],
                "exp": 4102444800_u64
            }),
                &EncodingKey::from_secret(key.as_bytes()),
            )
            .unwrap()
        };
        let graph_upsert_token = graph_upsert_token_for("alice");
        let mut extraction: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/graph-extraction-v1.json"
        )))
        .unwrap();
        extraction["doc_id"] = json!("public-graph-upsert-doc");
        extraction["nodes"][0]["references"][0]["doc_id"] =
            json!("public-graph-upsert-doc");
        extraction["nodes"][1]["mentions"][0]["spans"][0]["doc_id"] =
            json!("public-graph-upsert-doc");
        extraction["edges"][0]["references"][0]["doc_id"] =
            json!("public-graph-upsert-doc");
        let denied_upsert = make_app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/graph/upsert")
                    .header("content-type", "application/json")
                    .header(
                        "authorization",
                        format!("Bearer {}", make_token(&[], "alpha")),
                    )
                    .body(Body::from(serde_json::to_vec(&extraction).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(denied_upsert.status(), StatusCode::FORBIDDEN);
        let denied_document_count: i64 = client
            .query_one(
                &format!("SELECT count(*) FROM \"{schema}\".\"gke_documents\" WHERE id=$1"),
                &[&"public-graph-upsert-doc"],
            )
            .await
            .unwrap()
            .get(0);
        assert_eq!(denied_document_count, 0);

        let mut rollback_extraction = extraction.clone();
        rollback_extraction["doc_id"] = json!("public-graph-upsert-rollback-doc");
        rollback_extraction["nodes"][0]["local_id"] = json!("public-rollback-node");
        rollback_extraction["nodes"][0]["id"] = json!("public-rollback-node");
        rollback_extraction["edges"][0]["source_ids"][0] = json!("public-rollback-node");
        rollback_extraction["nodes"][0]["references"][0]["doc_id"] =
            json!("public-graph-upsert-rollback-doc");
        rollback_extraction["nodes"][1]["mentions"][0]["spans"][0]["doc_id"] =
            json!("public-graph-upsert-rollback-doc");
        rollback_extraction["edges"][0]["references"][0]["doc_id"] =
            json!("public-graph-upsert-rollback-doc");
        let rollback_upsert = make_app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/graph/upsert")
                    .header("content-type", "application/json")
                    .header(
                        "authorization",
                        format!("Bearer {graph_upsert_token}"),
                    )
                    .body(Body::from(
                        serde_json::to_vec(&rollback_extraction).unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let rollback_status = rollback_upsert.status();
        let rollback_body = to_bytes(rollback_upsert.into_body(), 65_536)
            .await
            .unwrap();
        assert_eq!(
            rollback_status,
            StatusCode::CONFLICT,
            "{}",
            String::from_utf8_lossy(&rollback_body)
        );
        let rolled_back_document_count: i64 = client
            .query_one(
                &format!("SELECT count(*) FROM \"{schema}\".\"gke_documents\" WHERE id=$1"),
                &[&"public-graph-upsert-rollback-doc"],
            )
            .await
            .unwrap()
            .get(0);
        assert_eq!(rolled_back_document_count, 0);

        let upsert = make_app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/graph/upsert")
                    .header("content-type", "application/json")
                    .header("authorization", format!("Bearer {graph_upsert_token}"))
                    .body(Body::from(serde_json::to_vec(&extraction).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(upsert.status(), StatusCode::OK);
        let upsert_body: Value =
            serde_json::from_slice(&to_bytes(upsert.into_body(), 65_536).await.unwrap()).unwrap();
        assert_eq!(upsert_body["document_id"], "public-graph-upsert-doc");
        let node_ids = upsert_body["node_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap().to_owned())
            .collect::<Vec<_>>();
        let edge_ids = upsert_body["edge_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap().to_owned())
            .collect::<Vec<_>>();
        assert_eq!((node_ids.len(), edge_ids.len()), (2, 1));
        assert_eq!(upsert_body["nodes_added"], 2);
        assert_eq!(upsert_body["edges_added"], 1);
        let stored_document: String = client
            .query_one(
                &format!("SELECT document FROM \"{schema}\".\"gke_documents\" WHERE id=$1"),
                &[&"public-graph-upsert-doc"],
            )
            .await
            .unwrap()
            .get(0);
        assert_eq!(stored_document, extraction["content"].as_str().unwrap());
        let stored_nodes: i64 = client
            .query_one(
                &format!("SELECT count(*) FROM \"{schema}\".\"gke_nodes\" WHERE id=ANY($1)"),
                &[&node_ids],
            )
            .await
            .unwrap()
            .get(0);
        let stored_edges: i64 = client
            .query_one(
                &format!("SELECT count(*) FROM \"{schema}\".\"gke_edges\" WHERE id=ANY($1)"),
                &[&edge_ids],
            )
            .await
            .unwrap()
            .get(0);
        assert_eq!((stored_nodes, stored_edges), (2, 1));
        let stored_jobs: i64 = client
            .query_one(
                &format!("SELECT count(*) FROM \"{schema}\".\"index_jobs\" WHERE entity_id=ANY($1) AND index_kind='node_embedding'"),
                &[&node_ids.iter().chain(edge_ids.iter()).cloned().collect::<Vec<_>>()],
            )
            .await
            .unwrap()
            .get(0);
        assert_eq!(stored_jobs, 3);

        let mut existing_document_extraction = extraction.clone();
        existing_document_extraction
            .as_object_mut()
            .unwrap()
            .remove("content");
        let existing_document_upsert = make_app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/graph/upsert")
                    .header("content-type", "application/json")
                    .header(
                        "authorization",
                        format!("Bearer {graph_upsert_token}"),
                    )
                    .body(Body::from(
                        serde_json::to_vec(&existing_document_extraction).unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(existing_document_upsert.status(), StatusCode::OK);
        let existing_document_body: Value = serde_json::from_slice(
            &to_bytes(existing_document_upsert.into_body(), 65_536)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(existing_document_body["document_id"], "public-graph-upsert-doc");
        assert_eq!(existing_document_body["nodes_added"], 2);
        assert_eq!(existing_document_body["edges_added"], 1);
        let unchanged_document: String = client
            .query_one(
                &format!("SELECT document FROM \"{schema}\".\"gke_documents\" WHERE id=$1"),
                &[&"public-graph-upsert-doc"],
            )
            .await
            .unwrap()
            .get(0);
        assert_eq!(unchanged_document, extraction["content"].as_str().unwrap());

        let mut external_endpoint_extraction = extraction.clone();
        external_endpoint_extraction
            .as_object_mut()
            .unwrap()
            .remove("content");
        external_endpoint_extraction["nodes"] = json!([]);
        let mut external_edge = extraction["edges"][0].clone();
        external_edge["id"] = json!("ne:external-node-edge");
        external_edge["source_ids"] = json!([existing_document_body["node_ids"][0]]);
        external_edge["target_ids"] = json!([existing_document_body["node_ids"][1]]);
        external_endpoint_extraction["edges"] = json!([external_edge]);
        let external_endpoint_upsert = make_app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/graph/upsert")
                    .header("content-type", "application/json")
                    .header(
                        "authorization",
                        format!("Bearer {graph_upsert_token}"),
                    )
                    .body(Body::from(
                        serde_json::to_vec(&external_endpoint_extraction).unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let external_endpoint_status = external_endpoint_upsert.status();
        let external_endpoint_bytes = to_bytes(external_endpoint_upsert.into_body(), 65_536)
            .await
            .unwrap();
        assert_eq!(
            external_endpoint_status,
            StatusCode::OK,
            "{}",
            String::from_utf8_lossy(&external_endpoint_bytes)
        );
        let external_endpoint_body: Value =
            serde_json::from_slice(&external_endpoint_bytes).unwrap();
        assert_eq!(external_endpoint_body["nodes_added"], 0);
        assert_eq!(external_endpoint_body["edges_added"], 1);

        let mut external_edge_endpoint_extraction = extraction.clone();
        external_edge_endpoint_extraction
            .as_object_mut()
            .unwrap()
            .remove("content");
        external_edge_endpoint_extraction["nodes"] = json!([]);
        let mut nested_external_edge = extraction["edges"][0].clone();
        nested_external_edge["id"] = json!("ne:external-edge-reference");
        nested_external_edge["source_ids"] = json!([]);
        nested_external_edge["target_ids"] =
            json!([existing_document_body["node_ids"][1]]);
        nested_external_edge["source_edge_ids"] =
            json!([existing_document_body["edge_ids"][0]]);
        nested_external_edge["target_edge_ids"] = json!([]);
        external_edge_endpoint_extraction["edges"] = json!([nested_external_edge]);
        let external_edge_endpoint_upsert = make_app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/graph/upsert")
                    .header("content-type", "application/json")
                    .header(
                        "authorization",
                        format!("Bearer {graph_upsert_token}"),
                    )
                    .body(Body::from(
                        serde_json::to_vec(&external_edge_endpoint_extraction).unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(external_edge_endpoint_upsert.status(), StatusCode::OK);
        let external_edge_endpoint_body: Value = serde_json::from_slice(
            &to_bytes(external_edge_endpoint_upsert.into_body(), 65_536)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(external_edge_endpoint_body["nodes_added"], 0);
        assert_eq!(external_edge_endpoint_body["edges_added"], 1);
        let events_before_hidden_endpoint = service
            .store
            .replay_raw_events("alpha", 0, 100)
            .await
            .unwrap()
            .len();
        let hidden_endpoint_upsert = make_app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/graph/upsert")
                    .header("content-type", "application/json")
                    .header(
                        "authorization",
                        format!("Bearer {}", graph_upsert_token_for("bob")),
                    )
                    .body(Body::from(
                        serde_json::to_vec(&external_endpoint_extraction).unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(hidden_endpoint_upsert.status(), StatusCode::CONFLICT);
        assert_eq!(
            service
                .store
                .replay_raw_events("alpha", 0, 100)
                .await
                .unwrap()
                .len(),
            events_before_hidden_endpoint
        );
        let hidden_edge_endpoint_upsert = make_app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/graph/upsert")
                    .header("content-type", "application/json")
                    .header(
                        "authorization",
                        format!("Bearer {}", graph_upsert_token_for("bob")),
                    )
                    .body(Body::from(
                        serde_json::to_vec(&external_edge_endpoint_extraction).unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(hidden_edge_endpoint_upsert.status(), StatusCode::CONFLICT);
        assert_eq!(
            service
                .store
                .replay_raw_events("alpha", 0, 100)
                .await
                .unwrap()
                .len(),
            events_before_hidden_endpoint
        );

        let mcp_list = make_app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header(
                        "authorization",
                        format!("Bearer {}", make_docs_token(&["read_graph"])),
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
        assert!(
            mcp_list_value["result"]["tools"]
                .as_array()
                .unwrap()
                .iter()
                .any(|tool| tool["name"] == "kg_find_edges")
        );

        let mcp_missing_capability = make_app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header(
                        "authorization",
                        format!("Bearer {}", make_docs_token(&[])),
                    )
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
        assert_eq!(denied_value["error"]["code"], -32602);
        assert_eq!(
            denied_value["error"]["message"],
            "Tool not advertised by this application service"
        );

        let find_edges_missing_capability = make_app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header(
                        "authorization",
                        format!("Bearer {}", make_docs_token(&[])),
                    )
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":"find-edges-denied","method":"tools/call","params":{"name":"kg_find_edges","arguments":{}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(find_edges_missing_capability.status(), StatusCode::OK);
        let denied_body = to_bytes(find_edges_missing_capability.into_body(), 65_536)
            .await
            .unwrap();
        let denied_value: Value = serde_json::from_slice(&denied_body).unwrap();
        assert_eq!(denied_value["error"]["code"], -32602);
        assert_eq!(
            denied_value["error"]["message"],
            "Tool not advertised by this application service"
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

        let mut generated = body.clone();
        generated["mutations"] = json!([body["mutations"][0].clone()]);
        generated["mutations"][0]["event_id"] = json!("graph-route-event-rust-embedding");
        generated["mutations"][0]["payload"]["id"] = json!("node-rust-embedded");
        generated["mutations"][0]["record"]["id"] = json!("node-rust-embedded");
        generated["mutations"][0]["record"]["document"] = json!("Rust-generated vector input");
        generated["mutations"][0]["record"]["embedding"] = Value::Null;
        generated["mutations"][0]["embedding_dim"] = json!(0);
        generated["index_jobs"] = json!([]);
        let generated_response = reqwest::Client::new()
            .post(format!("http://{address}/internal/graph/mutations"))
            .bearer_auth(make_token(&["write_graph"], "alpha"))
            .json(&generated)
            .send()
            .await
            .unwrap();
        assert_eq!(generated_response.status(), StatusCode::OK);
        assert_eq!(embedding_calls.lock().unwrap().len(), 1);

        embedding_failure.store(true, std::sync::atomic::Ordering::Relaxed);
        let mut failed_generation = generated.clone();
        failed_generation["mutations"][0]["event_id"] = json!("graph-route-event-provider-failure");
        failed_generation["mutations"][0]["payload"]["id"] = json!("node-provider-failure");
        failed_generation["mutations"][0]["record"]["id"] = json!("node-provider-failure");
        let failed_response = reqwest::Client::new()
            .post(format!("http://{address}/internal/graph/mutations"))
            .bearer_auth(make_token(&["write_graph"], "alpha"))
            .json(&failed_generation)
            .send()
            .await
            .unwrap();
        assert_eq!(failed_response.status(), StatusCode::BAD_GATEWAY);
        server.abort();
        let _ = server.await;

        let events = service
            .store
            .replay_raw_events("alpha", 0, 100)
            .await
            .unwrap();
        assert_eq!(events.len(), 39);
        let jobs = service
            .store
            .list_index_jobs(Some("alpha"), None, None, None, None, 100)
            .await
            .unwrap();
        assert_eq!(jobs.len(), 57);
        assert!(jobs.iter().any(|job| job.job_id == "graph-route-index-1"));

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
        let generated_records = service
            .store
            .graph_projection_records(kogwistar_store::GraphProjectionRead {
                scope: GraphScope {
                    namespace: "alpha".to_owned(),
                    workspace_id: Some("ws".to_owned()),
                    graph_space: Some("facts".to_owned()),
                },
                table: "gke_nodes".to_owned(),
                ids: Some(vec!["node-rust-embedded".to_owned()]),
                metadata: Default::default(),
                limit: 10,
            })
            .await
            .unwrap();
        assert_eq!(generated_records.len(), 1);
        assert_eq!(generated_records[0].embedding, Some(vec![0.0, 0.0, 1.0]));
        let failed_records = service
            .store
            .graph_projection_records(kogwistar_store::GraphProjectionRead {
                scope: GraphScope {
                    namespace: "alpha".to_owned(),
                    workspace_id: Some("ws".to_owned()),
                    graph_space: Some("facts".to_owned()),
                },
                table: "gke_nodes".to_owned(),
                ids: Some(vec!["node-provider-failure".to_owned()]),
                metadata: Default::default(),
                limit: 10,
            })
            .await
            .unwrap();
        assert!(failed_records.is_empty());
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
                        format!("Bearer {}", make_docs_token(&["read_graph"])),
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
                .replay_raw_events("alpha", 0, 100)
                .await
                .unwrap()
                .len(),
            39
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
            10
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
                        format!("Bearer {}", make_docs_token(&["read_graph"])),
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
                        format!("Bearer {}", make_docs_token(&["read_graph"])),
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
                        format!("Bearer {}", make_docs_token(&["read_graph"])),
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
                        format!("Bearer {}", make_docs_token(&["read_graph"])),
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
            mutation["record"]["document"] = json!(serde_json::to_string(&json!({
                "label": id,
                "doc_id": acl_fixture["doc_id"]
            })).unwrap());
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
            "doc_id": acl_fixture["doc_id"],
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
            let requests = [
                (
                    "kg_neighbors",
                    json!({"rid": "source-public", "doc_id": acl_fixture["doc_id"]}),
                ),
                (
                    "kg_k_hop",
                    json!({"start_ids": ["source-public"], "k": 2, "doc_id": acl_fixture["doc_id"]}),
                ),
                (
                    "kg_shortest_path",
                    json!({"src_id": "source-public", "dst_id": "target-private", "max_depth": 8, "doc_id": acl_fixture["doc_id"]}),
                ),
                (
                    "kg_find_edges",
                    json!({"relation": edge_fixture["relation"], "doc_id": acl_fixture["doc_id"]}),
                ),
            ];
            for (tool_name, arguments) in requests {
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
                                    make_docs_token_for(subject, &["read_graph"])
                                ),
                            )
                            .body(Body::from(
                                json!({
                                    "jsonrpc": "2.0",
                                    "id": format!("acl-{subject}-{tool_name}"),
                                    "method": "tools/call",
                                    "params": {"name": tool_name, "arguments": arguments}
                                })
                                .to_string(),
                            ))
                            .unwrap(),
                    )
                    .await
                    .unwrap();
                assert_eq!(response.status(), StatusCode::OK);
                let response_body = to_bytes(response.into_body(), 65_536).await.unwrap();
                let response_value: Value = serde_json::from_slice(&response_body).unwrap();
                assert!(response_value.get("error").is_none(), "{response_value}");
                let result = &response_value["result"];
                match tool_name {
                    "kg_neighbors" => {
                        assert_eq!(
                            result["nodes"], principal["expected_neighbors"]["nodes"],
                            "unexpected visible nodes for {subject}"
                        );
                        assert_eq!(
                            result["edges"], principal["expected_neighbors"]["edges"],
                            "unexpected visible edges for {subject}"
                        );
                    }
                    "kg_k_hop" => {
                        assert_eq!(
                            result["layers"], principal["expected_k_hop"],
                            "unexpected k-hop layers for {subject}"
                        );
                    }
                    "kg_shortest_path" => {
                        assert_eq!(
                            result["path"], principal["expected_shortest_path"],
                            "unexpected path for {subject}"
                        );
                    }
                    "kg_find_edges" => {
                        assert_eq!(
                            result["edges"], principal["expected_find_edges"],
                            "unexpected relation matches for {subject}"
                        );
                    }
                    _ => unreachable!("test request has known MCP tools"),
                }
            }
        }
        let tree_fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/document-upsert-tree-success-v1.json"
        )))
        .unwrap();
        let tree_request = &tree_fixture["request"];
        let tree_response = &tree_fixture["expected"];
        let tree_upsert = make_app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/document.upsert_tree")
                    .header("content-type", "application/json")
                    .header("authorization", format!("Bearer {graph_upsert_token}"))
                    .body(Body::from(serde_json::to_vec(tree_request).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(tree_upsert.status().as_u16(), tree_response["status_code"]);
        let tree_body: Value = serde_json::from_slice(
            &to_bytes(tree_upsert.into_body(), 65_536).await.unwrap(),
        )
        .unwrap();
        assert_eq!(tree_body, tree_response["body"]);

        let mut tree_payload = extraction.clone();
        let tree_payload_object = tree_payload.as_object_mut().unwrap();
        tree_payload_object.remove("contract_version");
        tree_payload_object.remove("content");
        tree_payload_object.remove("doc_type");
        tree_payload_object.remove("insertion_method");
        let tree_upsert = make_app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/document.upsert_tree")
                    .header("content-type", "application/json")
                    .header("authorization", format!("Bearer {graph_upsert_token}"))
                    .body(Body::from(serde_json::to_vec(&tree_payload).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(tree_upsert.status(), StatusCode::OK);
        let tree_body: Value = serde_json::from_slice(
            &to_bytes(tree_upsert.into_body(), 65_536).await.unwrap(),
        )
        .unwrap();
        assert_eq!(tree_body["status"], "ok");
        assert_eq!(tree_body["inserted_nodes"], 2);
        assert_eq!(tree_body["inserted_edges"], 1);
        assert_eq!(tree_body["engine_result"]["document_id"], "public-graph-upsert-doc");
        assert_eq!(tree_body["engine_result"]["node_ids"].as_array().unwrap().len(), 2);
        assert_eq!(tree_body["engine_result"]["edge_ids"].as_array().unwrap().len(), 1);
        drop(mismatched_service);
        drop(legacy_service);
        drop(service);
        client
            .batch_execute(&format!(
                "DROP SCHEMA \"{schema}\" CASCADE; DROP SCHEMA \"{legacy_schema}\" CASCADE"
            ))
            .await
            .unwrap();
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
            &json!({"sub":"alice","role":"ro","roles":["reader"],"ns":"docs","storage_ns":"alpha","capabilities":["read_graph"],"exp":4102444800_u64}),
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
        let preflight: Value = http
            .post(format!("http://{address}/mcp"))
            .bearer_auth(&token)
            .json(&json!({
                "jsonrpc": "2.0",
                "id": "snapshot-preflight",
                "method": "tools/call",
                "params": {"name": "kg_neighbors", "arguments": {"rid": source_id, "doc_id": "D1"}}
            }))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert!(preflight.get("error").is_none(), "{preflight}");

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
        client
            .batch_execute(&format!("DROP SCHEMA \"{schema}\" CASCADE"))
            .await
            .unwrap();
        drop(client);
        drop(observer);
        observer_task.await.unwrap().unwrap();
        connection_task.await.unwrap().unwrap();
    }

    #[tokio::test]
    #[ignore = "requires KOGWISTAR_TEST_PG_DSN; run explicitly with --include-ignored"]
    async fn postgres_document_projection_route_persists_reopens_and_retries_when_dsn_available() {
        struct FixedEmbeddingProvider {
            profile: kogwistar_engine::embedding::EmbeddingProfile,
            inputs: Arc<Mutex<Vec<String>>>,
            fail_next: Arc<std::sync::atomic::AtomicBool>,
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
                    if self
                        .fail_next
                        .swap(false, std::sync::atomic::Ordering::SeqCst)
                    {
                        return Err(kogwistar_engine::embedding::EmbeddingError::Request(
                            "synthetic provider failure".to_owned(),
                        ));
                    }
                    self.inputs.lock().unwrap().extend(texts.iter().cloned());
                    Ok(texts.iter().map(|_| vec![0.6, 0.8, 0.0]).collect())
                })
            }
        }

        let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
            .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
        let schema = format!("rust_api_document_{}", uuid::Uuid::new_v4().simple());
        let (mut client, connection) = tokio_postgres::connect(&dsn, tokio_postgres::NoTls)
            .await
            .unwrap();
        tokio::spawn(async move {
            let _ = connection.await;
        });
        let transaction = client.transaction().await.unwrap();
        transaction
            .query_one(
                "SELECT pg_advisory_xact_lock(hashtext('kogwistar.pgvector.extension'))",
                &[],
            )
            .await
            .unwrap();
        transaction
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
        transaction.commit().await.unwrap();

        let embedding_inputs = Arc::new(Mutex::new(Vec::new()));
        let fail_next_embedding = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let service = PostgresRunApplicationService::from_dsn(&dsn, &schema)
            .unwrap()
            .with_embedding_provider(Arc::new(FixedEmbeddingProvider {
                profile: kogwistar_engine::embedding::EmbeddingProfile {
                    provider: "test".to_owned(),
                    model: "document-route-fixture".to_owned(),
                    dimension: 3,
                    similarity_metric: "cosine".to_owned(),
                    endpoint_fingerprint: None,
                    max_sequence_length: None,
                    crop_token_budget: None,
                    tokenizer_fingerprint: None,
                    crop_policy: None,
                    embedding_kind: None,
                    model_revision: None,
                    preprocessing_fingerprint: None,
                    max_image_patches: None,
                },
                inputs: embedding_inputs.clone(),
                fail_next: fail_next_embedding,
            }));
        service.ensure_schema().await.unwrap();
        service
            .store
            .create_graph_schema(
                3,
                kogwistar_store_postgres::GraphTableNames::default(),
            )
            .await
            .unwrap();
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
                        .uri(PUBLIC_DOCUMENT_UPSERT_ROUTE)
                        .header("content-type", "application/json")
                        .header("authorization", format!("Bearer {token}"))
                        .body(Body::from(serde_json::to_vec(&body).unwrap()))
                        .unwrap(),
                )
                .await
                .unwrap()
            }
        };

        let first_payload: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/document-upsert-v1.json"
        )))
        .unwrap();
        let failed = send(make_app(), first_payload.clone()).await;
        assert_eq!(failed.status(), StatusCode::BAD_GATEWAY);
        let failed_body = to_bytes(failed.into_body(), 4096).await.unwrap();
        let failure: Value = serde_json::from_slice(&failed_body).unwrap();
        assert_eq!(failure["code"], "KOGWISTAR_EMBEDDING_FAILED");
        let unpersisted = service
            .store
            .graph_projection_records(GraphProjectionRead {
                scope: GraphScope {
                    namespace: "docs".to_owned(),
                    workspace_id: Some("workspace-a".to_owned()),
                    graph_space: Some("knowledge".to_owned()),
                },
                table: "gke_documents".to_owned(),
                ids: Some(vec![first_payload["doc_id"].as_str().unwrap().to_owned()]),
                metadata: MetadataFilter::default(),
                limit: 10,
            })
            .await
            .unwrap();
        assert!(unpersisted.is_empty());

        let first = send(make_app(), first_payload.clone()).await;
        let first_status = first.status();
        let first_body = to_bytes(first.into_body(), 4096).await.unwrap();
        assert_eq!(
            first_status,
            StatusCode::OK,
            "document upsert failed: {}",
            String::from_utf8_lossy(&first_body)
        );
        assert_eq!(
            serde_json::from_slice::<Value>(&first_body).unwrap(),
            json!({"status": "ok"})
        );
        let mut retry_payload = first_payload.clone();
        retry_payload["content"] = json!("updated body");
        let retry = send(make_app(), retry_payload).await;
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
                ids: Some(vec![first_payload["doc_id"].as_str().unwrap().to_owned()]),
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
        assert_eq!(
            rows[0].metadata["insertion_method"],
            first_payload["insertion_method"]
        );
        assert_eq!(rows[0].embedding.as_deref(), Some(&[0.6, 0.8, 0.0][..]));
        assert_eq!(
            *embedding_inputs.lock().unwrap(),
            vec![
                first_payload["content"].as_str().unwrap().to_owned(),
                "updated body".to_owned()
            ]
        );

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
        use kogwistar_store::{NamedProjectionWrite, WorkflowDesignSnapshotWrite};
        use kogwistar_store_postgres::{GraphTableNames, NewRawEntityEvent};
        use rusqlite::Connection;

        let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
            .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
        let schema = format!("rust_api_service_events_{}", uuid::Uuid::new_v4().simple());
        let auth_path = std::env::temp_dir().join(format!(
            "kogwistar-service-trigger-auth-{}.sqlite",
            uuid::Uuid::new_v4().simple()
        ));
        let auth = SqliteAuthStore::open(&auth_path).unwrap();
        auth.resolve_external_identity(ResolveExternalIdentity {
            issuer: "test".to_owned(),
            subject: "operator".to_owned(),
            email: "operator@example.test".to_owned(),
            display_name: None,
            new_user_id: "operator".to_owned(),
            default_role: "rw".to_owned(),
            default_ns: "workflow".to_owned(),
        })
        .await
        .unwrap();
        drop(auth);
        let auth_config = AuthConfig {
            auth_db_url: Some(format!("sqlite:///{}", auth_path.display())),
            ..AuthConfig::default()
        };
        let mut native_operations = kogwistar_runtime::NativeOperationRegistry::default();
        native_operations
            .register_graph("app/service-owner-graph", "v1", |request| {
                let entity_id = format!("service-native-node-{}", request.run_id);
                let batch = serde_json::from_value(json!({
                    "contract_version": 1,
                    "atomic": true,
                    "mutations": [{
                        "scope": {"namespace": "workflow"},
                        "table": "gke_nodes",
                        "entity_kind": "node",
                        "event_id": format!("service-native-event-{}", request.run_id),
                        "op": "ADD",
                        "payload": {"id": entity_id},
                        "record": {"id": entity_id, "document": "service-triggered native graph"},
                        "embedding_dim": 0
                    }],
                    "index_jobs": [{
                        "job_id": format!("service-native-job-{}", request.run_id),
                        "namespace": "workflow",
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
                        result: Some(json!({"service_native_graph": true})),
                    },
                    batch,
                ))
            })
            .unwrap();
        let service = PostgresRunApplicationService::from_dsn(&dsn, &schema)
            .unwrap()
            .with_native_operations(native_operations)
            .with_owner_auth_config(&auth_config)
            .await
            .unwrap();
        service.ensure_schema().await.unwrap();
        service
            .store
            .create_graph_schema(3, GraphTableNames::default())
            .await
            .unwrap();
        service
            .store
            .replace_named_projection(
                "workflow_design",
                "wf-pg-trigger",
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
                "wf-pg-trigger",
                WorkflowDesignSnapshotWrite {
                    version: 1,
                    seq: 1,
                    schema_version: 1,
                    payload_json: serde_json::to_string(&json!({
                        "nodes": [{
                            "id": "service-start",
                            "metadata": {
                                "wf_start": true,
                                "wf_op": "rust-native/app/service-owner-graph",
                                "wf_executor_target": "rust-native"
                            }
                        }],
                        "edges": []
                    }))
                    .unwrap(),
                },
            )
            .await
            .unwrap();
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
            "capabilities": ["service.manage", "service.inspect", "project_view", "write_graph"],
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
                        "runtime_kind":"async",
                        "user_id":"operator"
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
        let run = service
            .store
            .get_server_run(run_id)
            .await
            .unwrap()
            .expect("service trigger run");
        assert_eq!(run.user_id.as_deref(), Some("operator"));
        let run_events = service.store.server_run_events(run_id, 0, 10).await.unwrap();
        let created_payload: Value =
            serde_json::from_str(&run_events[0].payload_json).expect("run.created payload");
        assert_eq!(created_payload["runtime_owner_binding"]["user_id"], "operator");
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
        assert!(service.execute_inline_native_work_once().await.unwrap());
        let entity_id = format!("service-native-node-{run_id}");
        let graph_rows = service
            .store
            .graph_projection_records(GraphProjectionRead {
                scope: GraphScope {
                    namespace: "workflow".to_owned(),
                    workspace_id: None,
                    graph_space: None,
                },
                table: "gke_nodes".to_owned(),
                ids: Some(vec![entity_id.clone()]),
                metadata: MetadataFilter::default(),
                limit: 10,
            })
            .await
            .unwrap();
        assert_eq!(graph_rows.len(), 1);
        assert_eq!(graph_rows[0].id, entity_id);
        let graph_events = service
            .store
            .replay_raw_events("workflow", 0, 100)
            .await
            .unwrap();
        assert!(graph_events
            .iter()
            .any(|event| event.event_id == format!("service-native-event-{run_id}")));
        let index_jobs = service
            .store
            .list_index_jobs(Some("workflow"), None, None, None, None, 100)
            .await
            .unwrap();
        assert!(index_jobs.iter().any(|job| {
            job.job_id == format!("service-native-job-{run_id}")
                && job.entity_id == entity_id
                && job.status == "PENDING"
        }));
        assert_eq!(
            service
                .store
                .get_server_run(run_id)
                .await
                .unwrap()
                .unwrap()
                .status,
            "succeeded"
        );
        let completed_lane = service
            .store
            .list_projected_lane_messages(LaneMessageFilter {
                correlation_id: Some(run_id.to_owned()),
                limit: 10,
                ..LaneMessageFilter::default()
            })
            .await
            .unwrap();
        assert_eq!(completed_lane.len(), 1);
        assert_eq!(completed_lane[0].status, "completed");
        assert!(!service.execute_inline_native_work_once().await.unwrap());

        let auth_connection = Connection::open(&auth_path).unwrap();
        for (case, column, changed_value, restored_value) in [
            ("deactivated", "is_active", "0", "1"),
            ("role-downgraded", "global_role", "ro", "rw"),
            ("namespace-moved", "global_ns", "other", "workflow"),
        ] {
            let service_id = format!("svc-pg-trigger-{case}");
            let conversation_id = format!("conv-pg-trigger-{case}");
            let declared = service
                .execute_async(ApiEffectRequest {
                    contract_version: 1,
                    method: "POST".to_owned(),
                    path_and_query: "/api/workflow/services".to_owned(),
                    body: serde_json::to_vec(&json!({
                        "service_id": service_id,
                        "target_kind": "workflow",
                        "target_ref": "wf-pg-trigger",
                        "target_config": {
                            "conversation_id": conversation_id,
                            "runtime_kind": "async",
                            "user_id": "operator"
                        }
                    }))
                    .unwrap(),
                    principal: principal.clone(),
                })
                .await;
            assert_eq!(declared.status, StatusCode::OK.as_u16(), "{case}");
            let triggered = service
                .execute_async(ApiEffectRequest {
                    contract_version: 1,
                    method: "POST".to_owned(),
                    path_and_query: format!("/api/workflow/services/{service_id}/trigger"),
                    body: serde_json::to_vec(&json!({
                        "trigger_type": "external event",
                        "payload": {"source": format!("revalidate-{case}")}
                    }))
                    .unwrap(),
                    principal: principal.clone(),
                })
                .await;
            assert_eq!(triggered.status, StatusCode::OK.as_u16(), "{case}");
            let triggered: Value = serde_json::from_slice(&triggered.body).unwrap();
            let run_id = triggered["current_child_run_id"]
                .as_str()
                .expect("service trigger admitted a child run")
                .to_owned();

            let changed = auth_connection
                .execute(
                    &format!("UPDATE users SET {column} = ?1 WHERE user_id = ?2"),
                    [changed_value, "operator"],
                )
                .unwrap();
            assert_eq!(changed, 1, "test must apply the {case} change");

            assert!(service.execute_inline_native_work_once().await.unwrap());
            let entity_id = format!("service-native-node-{run_id}");
            let graph_rows = service
                .store
                .graph_projection_records(GraphProjectionRead {
                    scope: GraphScope {
                        namespace: "workflow".to_owned(),
                        workspace_id: None,
                        graph_space: None,
                    },
                    table: "gke_nodes".to_owned(),
                    ids: Some(vec![entity_id]),
                    metadata: MetadataFilter::default(),
                    limit: 10,
                })
                .await
                .unwrap();
            assert!(graph_rows.is_empty(), "{case} owner must not write graph data");
            let events = service
                .store
                .replay_raw_events("workflow", 0, 100)
                .await
                .unwrap();
            assert!(!events
                .iter()
                .any(|event| event.event_id == format!("service-native-event-{run_id}")));
            let jobs = service
                .store
                .list_index_jobs(Some("workflow"), None, None, None, None, 100)
                .await
                .unwrap();
            assert!(!jobs
                .iter()
                .any(|job| job.job_id == format!("service-native-job-{run_id}")));
            assert_eq!(
                service
                    .store
                    .get_server_run(&run_id)
                    .await
                    .unwrap()
                    .unwrap()
                    .status,
                "failed",
                "{case} denial must be recorded as a durable failed run"
            );
            let lane = service
                .store
                .list_projected_lane_messages(LaneMessageFilter {
                    correlation_id: Some(run_id),
                    limit: 10,
                    ..LaneMessageFilter::default()
                })
                .await
                .unwrap();
            assert_eq!(lane.len(), 1, "{case}");
            assert_eq!(lane[0].status, "completed", "{case}");

            let restored = auth_connection
                .execute(
                    &format!("UPDATE users SET {column} = ?1 WHERE user_id = ?2"),
                    [restored_value, "operator"],
                )
                .unwrap();
            assert_eq!(restored, 1, "restore auth state after {case}");
        }

        let spoof_service = service
            .execute_async(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/workflow/services".to_owned(),
                body: serde_json::to_vec(&json!({
                    "service_id": "svc-pg-trigger-spoof",
                    "target_kind": "workflow",
                    "target_ref": "wf-pg-trigger",
                    "target_config": {
                        "conversation_id": "conv-pg-trigger-spoof",
                        "user_id": "victim"
                    }
                }))
                .unwrap(),
                principal: principal.clone(),
            })
            .await;
        assert_eq!(spoof_service.status, StatusCode::OK.as_u16());
        let spoof_trigger = service
            .execute_async(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/workflow/services/svc-pg-trigger-spoof/trigger".to_owned(),
                body: serde_json::to_vec(&json!({
                    "trigger_type": "external event",
                    "payload": {}
                }))
                .unwrap(),
                principal: principal.clone(),
            })
            .await;
        assert_eq!(spoof_trigger.status, StatusCode::FORBIDDEN.as_u16());
        let spoof_error: Value = serde_json::from_slice(&spoof_trigger.body).unwrap();
        assert_eq!(spoof_error["code"], "KOGWISTAR_RUN_OWNER_FORBIDDEN");
        assert!(service
            .store
            .list_projected_lane_messages(LaneMessageFilter {
                conversation_id: Some("conv-pg-trigger-spoof".to_owned()),
                limit: 10,
                ..LaneMessageFilter::default()
            })
            .await
            .unwrap()
            .is_empty());
        drop(service);
        let _ = std::fs::remove_file(auth_path);

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
