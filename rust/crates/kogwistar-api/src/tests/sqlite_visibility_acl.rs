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
        let visibility_contract_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../contracts/golden/workflow-visibility-http-v1.json");
        let visibility_contract: Value = serde_json::from_slice(
            &std::fs::read(visibility_contract_path)
                .expect("read shared workflow visibility HTTP contract"),
        )
        .expect("parse shared workflow visibility HTTP contract");
        assert_eq!(StatusCode::OK.as_u16(), visibility_contract["success_status"]);
        assert_eq!(visibility, visibility_contract["response"]);
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
