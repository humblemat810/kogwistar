    #[tokio::test]
    async fn workflow_design_node_upsert_authorization_matches_python_policy_for_all_stores() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "kogwistar-design-node-upsert-auth-{nonce}.sqlite3"
        ));
        let sqlite = SqliteRunApplicationService::open(&path).unwrap();
        let postgres = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "design_node_upsert_auth",
        )
        .unwrap();
        let services: [&dyn ApplicationService; 2] = [&sqlite, &postgres];
        let contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/workflow-design-node-upsert-mcp-v1.json"
        ))
        .unwrap();
        let tool = contract["tool"].as_str().unwrap();
        let authorized = json!({
            "role":"rw", "ns":["workflow"],
            "capabilities":contract["caller_capabilities"]
        });

        for service in services {
            assert!(service.supports_mcp_tool_for_principal("workflow", tool, &authorized));
            assert!(!service.supports_mcp_tool_for_principal("root", tool, &authorized));
            for principal in [
                json!({"role":"rw", "ns":["workflow"], "capabilities":["workflow.design.write"]}),
                json!({"role":"rw", "ns":["workflow"], "capabilities":["write_graph"]}),
                json!({"role":"ro", "ns":["workflow"], "capabilities":contract["caller_capabilities"]}),
                json!({"role":"rw", "ns":["conversation"], "capabilities":contract["caller_capabilities"]}),
            ] {
                assert!(!service.supports_mcp_tool_for_principal("workflow", tool, &principal));
                let response = service.execute(ApiEffectRequest {
                    contract_version: 1,
                    method: contract["rust_request"]["method"].as_str().unwrap().to_owned(),
                    path_and_query: contract["rust_request"]["path_and_query"]
                        .as_str().unwrap().to_owned(),
                    body: serde_json::to_vec(&contract["arguments"]).unwrap(),
                    principal,
                }).await;
                assert_eq!(response.status, StatusCode::FORBIDDEN.as_u16());
            }
        }

        drop(sqlite);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn workflow_design_undo_authorization_matches_python_policy_for_all_stores() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-design-undo-auth-{nonce}.sqlite3"));
        let sqlite = SqliteRunApplicationService::open(&path).unwrap();
        let postgres = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "design_undo_auth",
        )
        .unwrap();
        let services: [&dyn ApplicationService; 2] = [&sqlite, &postgres];
        let contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/workflow-design-undo-mcp-v1.json"
        ))
        .unwrap();
        let tool = contract["tool"].as_str().unwrap();
        let authorized = json!({
            "role":"rw", "ns":["workflow"],
            "capabilities":contract["caller_capabilities"]
        });

        for service in services {
            assert!(service.supports_mcp_tool_for_principal("workflow", tool, &authorized));
            for principal in [
                json!({"role":"rw", "ns":["workflow"], "capabilities":["workflow.design.write", "write_graph"]}),
                json!({"role":"rw", "ns":["workflow"], "capabilities":["approve_action", "write_graph"]}),
                json!({"role":"ro", "ns":["workflow"], "capabilities":contract["caller_capabilities"]}),
                json!({"role":"rw", "ns":["conversation"], "capabilities":contract["caller_capabilities"]}),
            ] {
                assert!(!service.supports_mcp_tool_for_principal("workflow", tool, &principal));
                let response = service.execute(ApiEffectRequest {
                    contract_version: 1,
                    method: contract["rust_request"]["method"].as_str().unwrap().to_owned(),
                    path_and_query: contract["rust_request"]["path_and_query"]
                        .as_str().unwrap().to_owned(),
                    body: serde_json::to_vec(&contract["arguments"]).unwrap(),
                    principal,
                }).await;
                assert_eq!(response.status, StatusCode::FORBIDDEN.as_u16());
            }
        }

        drop(sqlite);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn workflow_design_redo_authorization_matches_python_policy_for_all_stores() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-design-redo-auth-{nonce}.sqlite3"));
        let sqlite = SqliteRunApplicationService::open(&path).unwrap();
        let postgres = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "design_redo_auth",
        )
        .unwrap();
        let services: [&dyn ApplicationService; 2] = [&sqlite, &postgres];
        let contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/workflow-design-redo-mcp-v1.json"
        ))
        .unwrap();
        let tool = contract["tool"].as_str().unwrap();
        let authorized = json!({
            "role":"rw", "ns":["workflow"],
            "capabilities":contract["caller_capabilities"]
        });

        for service in services {
            assert!(service.supports_mcp_tool_for_principal("workflow", tool, &authorized));
            for principal in [
                json!({"role":"rw", "ns":["workflow"], "capabilities":["workflow.design.write"]}),
                json!({"role":"rw", "ns":["workflow"], "capabilities":["approve_action"]}),
                json!({"role":"ro", "ns":["workflow"], "capabilities":contract["caller_capabilities"]}),
                json!({"role":"rw", "ns":["conversation"], "capabilities":contract["caller_capabilities"]}),
            ] {
                assert!(!service.supports_mcp_tool_for_principal("workflow", tool, &principal));
                let response = service.execute(ApiEffectRequest {
                    contract_version: 1,
                    method: contract["rust_request"]["method"].as_str().unwrap().to_owned(),
                    path_and_query: contract["rust_request"]["path_and_query"]
                        .as_str().unwrap().to_owned(),
                    body: serde_json::to_vec(&contract["arguments"]).unwrap(),
                    principal,
                }).await;
                assert_eq!(response.status, StatusCode::FORBIDDEN.as_u16());
            }
        }

        drop(sqlite);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn workflow_message_orphans_repair_authorization_matches_python_policy_for_all_stores() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-message-repair-auth-{nonce}.sqlite3"));
        let sqlite = SqliteRunApplicationService::open(&path).unwrap();
        let postgres = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "message_orphan_repair_auth",
        )
        .unwrap();
        let services: [&dyn ApplicationService; 2] = [&sqlite, &postgres];
        let tool = "workflow.message_orphans_repair";
        let authorized = json!({
            "role":"rw", "ns":["workflow"], "capabilities":["project_view"]
        });

        for service in services {
            assert!(service.supports_mcp_tool_for_principal("workflow", tool, &authorized));
            for principal in [
                json!({"role":"rw", "ns":["workflow"], "capabilities":[]}),
                json!({"role":"ro", "ns":["workflow"], "capabilities":["project_view"]}),
                json!({"role":"rw", "ns":["conversation"], "capabilities":["project_view"]}),
            ] {
                assert!(!service.supports_mcp_tool_for_principal("workflow", tool, &principal));
                let response = service.execute(ApiEffectRequest {
                    contract_version: 1,
                    method: "POST".to_owned(),
                    path_and_query: "/api/workflow/messages/repair-orphans?limit=10&inbox_id=repair-auth".to_owned(),
                    body: Vec::new(),
                    principal,
                }).await;
                assert_eq!(response.status, StatusCode::FORBIDDEN.as_u16());
            }
        }

        drop(sqlite);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn workflow_operator_dashboard_mcp_admission_matches_python_policy_for_all_stores() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "kogwistar-operator-dashboard-auth-{nonce}.sqlite3"
        ));
        let sqlite = SqliteRunApplicationService::open(&path).unwrap();
        let postgres = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "operator_dashboard_auth",
        )
        .unwrap();
        let services: [&dyn ApplicationService; 2] = [&sqlite, &postgres];
        let authorized = json!({
            "role":"ro",
            "ns":["workflow"],
            "capabilities":["project_view", "workflow.run.read", "read_security_scope"]
        });
        let required = ["project_view", "workflow.run.read", "read_security_scope"];

        for service in services {
            for role in ["ro", "rw"] {
                let mut principal = authorized.clone();
                principal["role"] = json!(role);
                assert!(service.supports_mcp_tool_for_principal(
                    "workflow",
                    "workflow.operator_dashboard",
                    &principal
                ));
            }
            let invalid_principals = required
                .iter()
                .map(|missing| {
                    json!({
                        "role":"ro",
                        "ns":["workflow"],
                        "capabilities":required.iter().filter(|capability| *capability != missing).collect::<Vec<_>>()
                    })
                })
                .chain([
                    json!({"role":"ro", "ns":["conversation"], "capabilities":required}),
                ]);
            for principal in invalid_principals {
                assert!(!service.supports_mcp_tool_for_principal(
                    "workflow",
                    "workflow.operator_dashboard",
                    &principal
                ));
                let response = service
                    .execute(ApiEffectRequest {
                        contract_version: 1,
                        method: "GET".to_owned(),
                        path_and_query: "/api/workflow/operator/dashboard?limit=10".to_owned(),
                        body: Vec::new(),
                        principal,
                    })
                    .await;
                assert_eq!(response.status, StatusCode::FORBIDDEN.as_u16());
            }
            assert!(!service.supports_mcp_tool_for_principal(
                "root",
                "workflow.operator_dashboard",
                &authorized
            ));
        }

        drop(sqlite);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn workflow_operator_inbox_authorization_denies_before_backend_access() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "kogwistar-operator-inbox-auth-{nonce}.sqlite3"
        ));
        let sqlite = SqliteRunApplicationService::open(&path).unwrap();
        let postgres = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "operator_inbox_auth",
        )
        .unwrap();
        let authorized = json!({
            "role":"ro", "ns":["workflow"], "capabilities":["project_view"]
        });
        assert!(postgres.supports_mcp_tool_for_principal(
            "workflow",
            "workflow.operator_inbox",
            &authorized
        ));
        assert!(!postgres.supports_mcp_tool_for_principal(
            "root",
            "workflow.operator_inbox",
            &authorized
        ));

        let denied_principals = [
            json!({"role":"ro", "ns":["workflow"], "capabilities":[]}),
            json!({"role":"ro", "ns":["conversation"], "capabilities":["project_view"]}),
        ];
        for principal in &denied_principals {
            assert!(!postgres.supports_mcp_tool_for_principal(
                "workflow",
                "workflow.operator_inbox",
                principal
            ));
            for service in [
                &sqlite as &dyn ApplicationService,
                &postgres as &dyn ApplicationService,
            ] {
                let response = service
                    .execute(ApiEffectRequest {
                        contract_version: 1,
                        method: "GET".to_owned(),
                        path_and_query: "/api/workflow/operator/inbox?inbox_id=auth-test".to_owned(),
                        body: Vec::new(),
                        principal: principal.clone(),
                    })
                    .await;
                assert_eq!(response.status, StatusCode::FORBIDDEN.as_u16());
            }
        }
        let unavailable = sqlite
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: "/api/workflow/operator/inbox?inbox_id=auth-test".to_owned(),
                body: Vec::new(),
                principal: authorized,
            })
            .await;
        assert_eq!(unavailable.status, StatusCode::NOT_IMPLEMENTED.as_u16());

        drop(sqlite);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn workflow_budget_reads_authorization_denies_before_backend_access() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-budget-auth-{nonce}.sqlite3"));
        let sqlite = SqliteRunApplicationService::open(&path).unwrap();
        let postgres = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "workflow_budget_auth",
        )
        .unwrap();
        let services: [&dyn ApplicationService; 2] = [&sqlite, &postgres];
        let authorized = json!({
            "role":"ro",
            "ns":["workflow"],
            "capabilities":["project_view", "read_security_scope"]
        });
        let required = ["project_view", "read_security_scope"];
        let operations = [
            ("workflow.budget_snapshot", "/api/workflow/budget"),
            (
                "workflow.budget_history",
                "/api/workflow/budget/history?limit=10",
            ),
        ];

        for service in services {
            for (tool, path) in operations {
                for role in ["ro", "rw"] {
                    let mut principal = authorized.clone();
                    principal["role"] = json!(role);
                    assert!(service.supports_mcp_tool_for_principal(
                        "workflow", tool, &principal
                    ));
                }
                assert!(!service.supports_mcp_tool_for_principal("root", tool, &authorized));

                let denied_principals = required
                    .iter()
                    .map(|missing| {
                        json!({
                            "role":"ro",
                            "ns":["workflow"],
                            "capabilities":required.iter().filter(|capability| *capability != missing).collect::<Vec<_>>()
                        })
                    })
                    .chain([json!({
                        "role":"ro",
                        "ns":["conversation"],
                        "capabilities":required
                    })]);

                for principal in denied_principals {
                    let response = service
                        .execute(ApiEffectRequest {
                            contract_version: 1,
                            method: "GET".to_owned(),
                            path_and_query: path.to_owned(),
                            body: Vec::new(),
                            principal,
                        })
                        .await;
                    assert_eq!(response.status, StatusCode::FORBIDDEN.as_u16());
                }
            }
        }

        for (_, path) in operations {
            for role in ["ro", "rw"] {
                let mut principal = authorized.clone();
                principal["role"] = json!(role);
                let authorized_response = sqlite
                    .execute(ApiEffectRequest {
                        contract_version: 1,
                        method: "GET".to_owned(),
                        path_and_query: path.to_owned(),
                        body: Vec::new(),
                        principal,
                    })
                    .await;
                assert_eq!(authorized_response.status, StatusCode::OK.as_u16());
            }
        }

        drop(sqlite);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn workflow_scheduler_timeline_authorization_denies_before_store_reads() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-scheduler-auth-{nonce}.sqlite3"));
        let sqlite = SqliteRunApplicationService::open(&path).unwrap();
        let postgres = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "workflow_scheduler_auth",
        )
        .unwrap();
        let services: [&dyn ApplicationService; 2] = [&sqlite, &postgres];
        let authorized = json!({
            "role":"ro",
            "ns":["workflow"],
            "capabilities":["project_view"]
        });

        for service in services {
            for role in ["ro", "rw"] {
                let mut principal = authorized.clone();
                principal["role"] = json!(role);
                assert!(service.supports_mcp_tool_for_principal(
                    "workflow",
                    "workflow.scheduler_timeline",
                    &principal
                ));
            }
            assert!(!service.supports_mcp_tool_for_principal(
                "root",
                "workflow.scheduler_timeline",
                &authorized
            ));

            for principal in [
                json!({"role":"ro", "ns":["workflow"], "capabilities":[]}),
                json!({"role":"ro", "ns":["conversation"], "capabilities":["project_view"]}),
            ] {
                let response = service
                    .execute(ApiEffectRequest {
                        contract_version: 1,
                        method: "GET".to_owned(),
                        path_and_query: "/api/workflow/scheduler/timeline?limit=10".to_owned(),
                        body: Vec::new(),
                        principal,
                    })
                    .await;
                assert_eq!(response.status, StatusCode::FORBIDDEN.as_u16());
            }
        }

        let authorized_response = sqlite
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: "/api/workflow/scheduler/timeline?limit=10".to_owned(),
                body: Vec::new(),
                principal: authorized,
            })
            .await;
        assert_eq!(authorized_response.status, StatusCode::OK.as_u16());

        drop(sqlite);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn workflow_service_declare_mcp_admission_matches_python_policy_for_all_stores() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-service-declare-auth-{nonce}.sqlite3"));
        let sqlite = SqliteRunApplicationService::open(&path).unwrap();
        let postgres = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "workflow_service_declare_auth",
        )
        .unwrap();
        let services: [&dyn ApplicationService; 2] = [&sqlite, &postgres];

        for service in services {
            let tool = "workflow.service_declare";
            let authorized = json!({"role":"rw", "ns":["workflow"]});
            assert!(service.supports_mcp_tool_for_principal(
                "workflow", tool, &authorized
            ));
            for principal in [
                json!({"role":"ro", "ns":["workflow"]}),
                json!({"role":"rw", "ns":["conversation"]}),
            ] {
                assert!(!service.supports_mcp_tool_for_principal(
                    "workflow", tool, &principal
                ));
            }
            assert!(!service.supports_mcp_tool_for_principal(
                "root", tool, &authorized
            ));

            let denied = service
                .execute(ApiEffectRequest {
                    contract_version: 1,
                    method: "POST".to_owned(),
                    path_and_query: "/api/workflow/services".to_owned(),
                    body: br#"{"service_id":"svc-auth-check","service_kind":"daemon","target_kind":"workflow","target_ref":"wf.auth-check","enabled":false,"autostart":false,"heartbeat_ttl_ms":1000}"#.to_vec(),
                    principal: json!({"role":"rw", "ns":["workflow"], "capabilities":[]}),
                })
                .await;
            assert_eq!(denied.status, StatusCode::FORBIDDEN.as_u16());
        }

        let allowed = sqlite
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/workflow/services".to_owned(),
                body: br#"{"service_id":"svc-auth-check","service_kind":"daemon","target_kind":"workflow","target_ref":"wf.auth-check","enabled":false,"autostart":false,"heartbeat_ttl_ms":1000}"#.to_vec(),
                principal: json!({"role":"rw", "ns":["workflow"], "capabilities":["service.manage"]}),
            })
            .await;
        assert_eq!(allowed.status, StatusCode::OK.as_u16());

        drop(sqlite);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn workflow_service_lifecycle_mcp_admission_matches_python_policy_for_all_stores() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-service-lifecycle-auth-{nonce}.sqlite3"));
        let sqlite = SqliteRunApplicationService::open(&path).unwrap();
        let postgres = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "workflow_service_lifecycle_auth",
        )
        .unwrap();
        let services: [&dyn ApplicationService; 2] = [&sqlite, &postgres];
        let operations = [
            (
                "workflow.service_enable",
                "/api/workflow/services/svc-lifecycle-auth/enable",
                true,
            ),
            (
                "workflow.service_disable",
                "/api/workflow/services/svc-lifecycle-auth/disable",
                false,
            ),
        ];

        for service in services {
            for (tool, path, _) in operations {
                let authorized = json!({"role":"rw", "ns":["workflow"]});
                assert!(service.supports_mcp_tool_for_principal(
                    "workflow", tool, &authorized
                ));
                for principal in [
                    json!({"role":"ro", "ns":["workflow"]}),
                    json!({"role":"rw", "ns":["conversation"]}),
                ] {
                    assert!(!service.supports_mcp_tool_for_principal(
                        "workflow", tool, &principal
                    ));
                }
                assert!(!service.supports_mcp_tool_for_principal(
                    "root", tool, &authorized
                ));
                let denied = service
                    .execute(ApiEffectRequest {
                        contract_version: 1,
                        method: "POST".to_owned(),
                        path_and_query: path.to_owned(),
                        body: Vec::new(),
                        principal: json!({"role":"rw", "ns":["workflow"], "capabilities":[]}),
                    })
                    .await;
                assert_eq!(denied.status, StatusCode::FORBIDDEN.as_u16());
            }
        }

        let declared = sqlite
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/workflow/services".to_owned(),
                body: br#"{"service_id":"svc-lifecycle-auth","service_kind":"daemon","target_kind":"workflow","target_ref":"wf.lifecycle-auth","enabled":false,"autostart":false,"heartbeat_ttl_ms":1000}"#.to_vec(),
                principal: json!({"role":"rw", "ns":["workflow"], "capabilities":["service.manage"]}),
            })
            .await;
        assert_eq!(declared.status, StatusCode::OK.as_u16());

        for (_, path, expected_enabled) in operations {
            let allowed = sqlite
                .execute(ApiEffectRequest {
                    contract_version: 1,
                    method: "POST".to_owned(),
                    path_and_query: path.to_owned(),
                    body: Vec::new(),
                    principal: json!({"role":"rw", "ns":["workflow"], "capabilities":["service.manage"]}),
                })
                .await;
            assert_eq!(allowed.status, StatusCode::OK.as_u16());
            let body: Value = serde_json::from_slice(&allowed.body).unwrap();
            assert_eq!(body["enabled"], expected_enabled);
        }

        drop(sqlite);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn workflow_service_heartbeat_mcp_admission_matches_python_policy_for_all_stores() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-service-heartbeat-auth-{nonce}.sqlite3"));
        let sqlite = SqliteRunApplicationService::open(&path).unwrap();
        let postgres = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "workflow_service_heartbeat_auth",
        )
        .unwrap();
        let services: [&dyn ApplicationService; 2] = [&sqlite, &postgres];
        let tool = "workflow.service_heartbeat";
        let service_id = "svc-heartbeat-auth";
        let heartbeat_path = format!("/api/workflow/services/{service_id}/heartbeat");

        for service in services {
            let authorized = json!({"role":"rw", "ns":["workflow"]});
            assert!(service.supports_mcp_tool_for_principal(
                "workflow", tool, &authorized
            ));
            for principal in [
                json!({"role":"ro", "ns":["workflow"]}),
                json!({"role":"rw", "ns":["conversation"]}),
            ] {
                assert!(!service.supports_mcp_tool_for_principal(
                    "workflow", tool, &principal
                ));
            }
            assert!(!service.supports_mcp_tool_for_principal(
                "root", tool, &authorized
            ));
            for capabilities in [
                json!([]),
                json!(["service.heartbeat"]),
                json!(["service.manage"]),
            ] {
                let denied = service
                    .execute(ApiEffectRequest {
                        contract_version: 1,
                        method: "POST".to_owned(),
                        path_and_query: heartbeat_path.clone(),
                        body: br#"{"instance_id":"worker-1","payload":{"beat":1}}"#.to_vec(),
                        principal: json!({"role":"rw", "ns":["workflow"], "capabilities":capabilities}),
                    })
                    .await;
                assert_eq!(denied.status, StatusCode::FORBIDDEN.as_u16());
            }
        }

        let declared = sqlite
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/workflow/services".to_owned(),
                body: serde_json::to_vec(&json!({
                    "service_id": service_id,
                    "service_kind": "daemon",
                    "target_kind": "workflow",
                    "target_ref": "wf.heartbeat-auth",
                    "enabled": false,
                    "autostart": false,
                    "heartbeat_ttl_ms": 1000
                }))
                .unwrap(),
                principal: json!({"role":"rw", "ns":["workflow"], "capabilities":["service.manage"]}),
            })
            .await;
        assert_eq!(declared.status, StatusCode::OK.as_u16());

        let allowed = sqlite
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: heartbeat_path,
                body: br#"{"instance_id":"worker-1","payload":{"beat":1}}"#.to_vec(),
                principal: json!({"role":"rw", "ns":["workflow"], "capabilities":["service.heartbeat", "service.manage"]}),
            })
            .await;
        assert_eq!(allowed.status, StatusCode::OK.as_u16());

        drop(sqlite);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn workflow_service_trigger_mcp_admission_matches_python_policy_for_all_stores() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-service-trigger-auth-{nonce}.sqlite3"));
        let sqlite = SqliteRunApplicationService::open(&path).unwrap();
        let postgres = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "workflow_service_trigger_auth",
        )
        .unwrap();
        let services: [&dyn ApplicationService; 2] = [&sqlite, &postgres];
        let tool = "workflow.service_trigger";
        let service_id = "svc-trigger-auth";
        let trigger_path = format!("/api/workflow/services/{service_id}/trigger");

        for service in services {
            let authorized = json!({"role":"rw", "ns":["workflow"]});
            assert!(service.supports_mcp_tool_for_principal(
                "workflow", tool, &authorized
            ));
            for principal in [
                json!({"role":"ro", "ns":["workflow"]}),
                json!({"role":"rw", "ns":["conversation"]}),
            ] {
                assert!(!service.supports_mcp_tool_for_principal(
                    "workflow", tool, &principal
                ));
            }
            assert!(!service.supports_mcp_tool_for_principal(
                "root", tool, &authorized
            ));
            let denied = service
                .execute(ApiEffectRequest {
                    contract_version: 1,
                    method: "POST".to_owned(),
                    path_and_query: trigger_path.clone(),
                    body: br#"{"trigger_type":"external event","payload":{"source":"auth-test"}}"#.to_vec(),
                    principal: json!({"role":"rw", "ns":["workflow"], "capabilities":[]}),
                })
                .await;
            assert_eq!(denied.status, StatusCode::FORBIDDEN.as_u16());
        }

        let declared = sqlite
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/workflow/services".to_owned(),
                body: serde_json::to_vec(&json!({
                    "service_id": service_id,
                    "service_kind": "daemon",
                    "target_kind": "workflow",
                    "target_ref": "wf.trigger-auth",
                    "enabled": false,
                    "autostart": false,
                    "heartbeat_ttl_ms": 1000
                }))
                .unwrap(),
                principal: json!({"role":"rw", "ns":["workflow"], "capabilities":["service.manage"]}),
            })
            .await;
        assert_eq!(declared.status, StatusCode::OK.as_u16());

        let allowed = sqlite
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: trigger_path,
                body: br#"{"trigger_type":"external event","payload":{"source":"auth-test"}}"#.to_vec(),
                principal: json!({"role":"rw", "ns":["workflow"], "capabilities":["service.manage"]}),
            })
            .await;
        assert_eq!(allowed.status, StatusCode::OK.as_u16());
        let projection: Value = serde_json::from_slice(&allowed.body).unwrap();
        assert_eq!(projection["service_id"], service_id);
        assert_eq!(projection["enabled"], false);
        assert_eq!(projection["current_child_run_id"], Value::Null);

        drop(sqlite);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn workflow_service_repair_mcp_admission_matches_python_policy_for_all_stores() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir()
            .join(format!("kogwistar-service-repair-auth-{nonce}.sqlite3"));
        let sqlite = SqliteRunApplicationService::open(&path).unwrap();
        let postgres = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "workflow_service_repair_auth",
        )
        .unwrap();
        let services: [&dyn ApplicationService; 2] = [&sqlite, &postgres];
        let tool = "workflow.service_repair";
        let service_id = "svc-repair-auth";
        let repair_path = format!("/api/workflow/services/{service_id}/repair");

        for service in services {
            let authorized = json!({"role":"rw", "ns":["workflow"]});
            assert!(service.supports_mcp_tool_for_principal(
                "workflow", tool, &authorized
            ));
            for principal in [
                json!({"role":"ro", "ns":["workflow"]}),
                json!({"role":"rw", "ns":["conversation"]}),
            ] {
                assert!(!service.supports_mcp_tool_for_principal(
                    "workflow", tool, &principal
                ));
            }
            assert!(!service.supports_mcp_tool_for_principal(
                "root", tool, &authorized
            ));
            let denied = service
                .execute(ApiEffectRequest {
                    contract_version: 1,
                    method: "POST".to_owned(),
                    path_and_query: repair_path.clone(),
                    body: Vec::new(),
                    principal: json!({"role":"rw", "ns":["workflow"], "capabilities":[]}),
                })
                .await;
            assert_eq!(denied.status, StatusCode::FORBIDDEN.as_u16());
        }

        let declared = sqlite
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/workflow/services".to_owned(),
                body: serde_json::to_vec(&json!({
                    "service_id": service_id,
                    "service_kind": "daemon",
                    "target_kind": "workflow",
                    "target_ref": "wf.repair-auth",
                    "enabled": false,
                    "autostart": false,
                    "heartbeat_ttl_ms": 1000
                }))
                .unwrap(),
                principal: json!({"role":"rw", "ns":["workflow"], "capabilities":["service.manage"]}),
            })
            .await;
        assert_eq!(declared.status, StatusCode::OK.as_u16());

        let allowed = sqlite
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: repair_path,
                body: Vec::new(),
                principal: json!({"role":"rw", "ns":["workflow"], "capabilities":["service.manage"]}),
            })
            .await;
        assert_eq!(allowed.status, StatusCode::OK.as_u16());
        let projection: Value = serde_json::from_slice(&allowed.body).unwrap();
        assert_eq!(projection["service_id"], service_id);
        assert_eq!(projection["projection"]["service_id"], service_id);

        drop(sqlite);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn workflow_services_repair_mcp_admission_matches_python_policy_for_all_stores() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir()
            .join(format!("kogwistar-services-repair-auth-{nonce}.sqlite3"));
        let sqlite = SqliteRunApplicationService::open(&path).unwrap();
        let postgres = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "workflow_services_repair_auth",
        )
        .unwrap();
        let services: [&dyn ApplicationService; 2] = [&sqlite, &postgres];
        let tool = "workflow.services_repair";
        let repair_path = "/api/workflow/services/repair?limit=37";

        for service in services {
            let authorized = json!({"role":"rw", "ns":["workflow"]});
            assert!(service.supports_mcp_tool_for_principal(
                "workflow", tool, &authorized
            ));
            for principal in [
                json!({"role":"ro", "ns":["workflow"]}),
                json!({"role":"rw", "ns":["conversation"]}),
            ] {
                assert!(!service.supports_mcp_tool_for_principal(
                    "workflow", tool, &principal
                ));
            }
            assert!(!service.supports_mcp_tool_for_principal(
                "root", tool, &authorized
            ));
            let denied = service
                .execute(ApiEffectRequest {
                    contract_version: 1,
                    method: "POST".to_owned(),
                    path_and_query: repair_path.to_owned(),
                    body: Vec::new(),
                    principal: json!({"role":"rw", "ns":["workflow"], "capabilities":[]}),
                })
                .await;
            assert_eq!(denied.status, StatusCode::FORBIDDEN.as_u16());
        }

        let allowed = sqlite
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: repair_path.to_owned(),
                body: Vec::new(),
                principal: json!({"role":"rw", "ns":["workflow"], "capabilities":["service.manage"]}),
            })
            .await;
        assert_eq!(allowed.status, StatusCode::OK.as_u16());
        let result: Value = serde_json::from_slice(&allowed.body).unwrap();
        assert_eq!(result["repaired_service_ids"], json!([]));

        drop(sqlite);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn workflow_tool_audit_mcp_admission_matches_python_policy() {
        let service = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "workflow_tool_audit_admission",
        )
        .unwrap();
        let sqlite_path = std::env::temp_dir().join(format!(
            "kogwistar-tool-audit-admission-{}.sqlite3",
            uuid::Uuid::new_v4()
        ));
        let sqlite = SqliteRunApplicationService::open(&sqlite_path).unwrap();
        let tool = "workflow.tool_audit";
        let authorized = json!({
            "role":"ro",
            "ns":["workflow"],
            "capabilities":["project_view", "read_graph"]
        });
        assert!(service.supports_mcp_tool_for_principal(
            "workflow", tool, &authorized
        ));
        for (surface, principal) in [
            ("workflow", json!({"role":"ro", "ns":["conversation"], "capabilities":["project_view", "read_graph"]})),
            ("root", authorized.clone()),
            ("workflow", json!({"role":"ro", "ns":["workflow"], "capabilities":["project_view"]})),
            ("workflow", json!({"role":"rw", "ns":["workflow"], "capabilities":["read_graph"]})),
        ] {
            assert!(!service.supports_mcp_tool_for_principal(surface, tool, &principal));
        }
        assert!(!sqlite.supports_mcp_tool_for_principal(
            "workflow", tool, &authorized
        ));

        for principal in [
            json!({"ns":"workflow", "capabilities":["read_graph"]}),
            json!({"ns":"workflow", "capabilities":["project_view"]}),
            json!({"ns":"docs", "capabilities":["project_view", "read_graph"]}),
        ] {
            let response = service
                .execute(ApiEffectRequest {
                    contract_version: 1,
                    method: "GET".to_owned(),
                    path_and_query: "/api/workflow/tools/audit?limit=5".to_owned(),
                    body: Vec::new(),
                    principal,
                })
                .await;
            assert_eq!(response.status, StatusCode::FORBIDDEN.as_u16());
        }

        drop(sqlite);
        let _ = std::fs::remove_file(sqlite_path);
    }

    #[tokio::test]
    async fn workflow_service_read_mcp_admission_matches_python_policy_for_all_stores() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-service-read-auth-{nonce}.sqlite3"));
        let sqlite = SqliteRunApplicationService::open(&path).unwrap();
        let postgres = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "workflow_service_read_auth",
        )
        .unwrap();
        let services: [&dyn ApplicationService; 2] = [&sqlite, &postgres];
        let operations = [
            (
                "workflow.service_get",
                "/api/workflow/services/svc-missing",
            ),
            ("workflow.service_list", "/api/workflow/services?limit=10"),
        ];

        for service in services {
            for (tool, path) in operations {
                let authorized = json!({
                    "sub":"service-read-authorized",
                    "role":"ro",
                    "ns":["workflow"],
                    "capabilities":["service.inspect", "project_view"]
                });
                assert!(service.supports_mcp_tool_for_principal(
                    "workflow", tool, &authorized
                ));
                for principal in [
                    json!({"sub":"service-read-no-caps", "role":"ro", "ns":["workflow"], "capabilities":[]}),
                    json!({"sub":"service-read-no-project-view", "role":"ro", "ns":["workflow"], "capabilities":["service.inspect"]}),
                    json!({"sub":"service-read-no-inspect", "role":"ro", "ns":["workflow"], "capabilities":["project_view"]}),
                    json!({"sub":"service-read-wrong-scope", "role":"ro", "ns":["conversation"], "capabilities":["service.inspect", "project_view"]}),
                ] {
                    assert!(!service.supports_mcp_tool_for_principal(
                        "workflow", tool, &principal
                    ), "unexpected MCP admission: {tool} {principal}");
                    let response = service
                        .execute(ApiEffectRequest {
                            contract_version: 1,
                            method: "GET".to_owned(),
                            path_and_query: path.to_owned(),
                            body: Vec::new(),
                            principal,
                        })
                        .await;
                    assert_eq!(response.status, StatusCode::FORBIDDEN.as_u16());
                }
                assert!(!service.supports_mcp_tool_for_principal(
                    "root",
                    tool,
                    &json!({"role":"ro", "ns":["workflow"], "capabilities":["service.inspect"]})
                ));
            }
        }

        for (tool, path) in operations {
            let response = sqlite
                .execute(ApiEffectRequest {
                    contract_version: 1,
                    method: "GET".to_owned(),
                    path_and_query: path.to_owned(),
                    body: Vec::new(),
                    principal: json!({
                        "sub":"service-read-authorized",
                        "role":"rw",
                        "ns":["workflow"],
                        "capabilities":["service.inspect", "project_view"]
                    }),
                })
                .await;
            if tool == "workflow.service_get" {
                assert_eq!(response.status, StatusCode::NOT_FOUND.as_u16());
            } else {
                assert_eq!(response.status, StatusCode::OK.as_u16());
            }
        }

        drop(sqlite);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn workflow_service_events_mcp_admission_matches_python_policy_for_all_stores() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-service-events-auth-{nonce}.sqlite3"));
        let sqlite = SqliteRunApplicationService::open(&path).unwrap();
        let postgres = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "workflow_service_events_auth",
        )
        .unwrap();
        let services: [&dyn ApplicationService; 2] = [&sqlite, &postgres];
        let authorized = json!({
            "sub":"service-events-authorized",
            "role":"ro",
            "ns":["workflow"],
            "capabilities":["service.inspect", "project_view"]
        });
        let path_and_query = "/api/workflow/services/svc-1/events?limit=5";

        for service in services {
            for role in ["ro", "rw"] {
                let mut principal = authorized.clone();
                principal["role"] = json!(role);
                assert!(service.supports_mcp_tool_for_principal(
                    "workflow",
                    "workflow.service_events",
                    &principal
                ));
            }
            for principal in [
                json!({"sub":"events-no-caps", "role":"ro", "ns":["workflow"], "capabilities":[]}),
                json!({"sub":"events-no-inspect", "role":"ro", "ns":["workflow"], "capabilities":["project_view"]}),
                json!({"sub":"events-no-project-view", "role":"ro", "ns":["workflow"], "capabilities":["service.inspect"]}),
                json!({"sub":"events-wrong-scope", "role":"ro", "ns":["conversation"], "capabilities":["service.inspect", "project_view"]}),
            ] {
                assert!(!service.supports_mcp_tool_for_principal(
                    "workflow",
                    "workflow.service_events",
                    &principal
                ));
                let response = service
                    .execute(ApiEffectRequest {
                        contract_version: 1,
                        method: "GET".to_owned(),
                        path_and_query: path_and_query.to_owned(),
                        body: Vec::new(),
                        principal,
                    })
                    .await;
                assert_eq!(response.status, StatusCode::FORBIDDEN.as_u16());
            }
            assert!(!service.supports_mcp_tool_for_principal(
                "root",
                "workflow.service_events",
                &authorized
            ));
        }

        let authorized_response = sqlite
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: path_and_query.to_owned(),
                body: Vec::new(),
                principal: authorized,
            })
            .await;
        assert_eq!(authorized_response.status, StatusCode::OK.as_u16());

        drop(sqlite);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn workflow_design_node_delete_mcp_admission_matches_python_policy_for_all_stores() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-design-node-delete-mcp-{nonce}.sqlite3"));
        let sqlite = SqliteRunApplicationService::open(&path).unwrap();
        let postgres = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "design_node_delete_mcp_admission",
        )
        .unwrap();
        let services: [&dyn ApplicationService; 2] = [&sqlite, &postgres];

        for service in services {
            let tool = "workflow.design_node_delete";
            let authorized = json!({
                "role":"rw", "ns":["workflow"],
                "capabilities":["workflow.design.write", "write_graph"]
            });
            assert!(service.supports_mcp_tool_for_principal("workflow", tool, &authorized));
            for principal in [
                json!({"role":"rw", "ns":["workflow"], "capabilities":["workflow.design.write"]}),
                json!({"role":"rw", "ns":["workflow"], "capabilities":["write_graph"]}),
                json!({"role":"ro", "ns":["workflow"], "capabilities":["workflow.design.write", "write_graph"]}),
                json!({"role":"rw", "ns":["conversation"], "capabilities":["workflow.design.write", "write_graph"]}),
            ] {
                assert!(!service.supports_mcp_tool_for_principal("workflow", tool, &principal));
            }
            assert!(!service.supports_mcp_tool_for_principal("root", tool, &authorized));
        }

        drop(sqlite);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn workflow_design_node_delete_effect_rechecks_scope_and_capabilities_for_all_stores() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-design-node-delete-effect-{nonce}.sqlite3"));
        let sqlite = SqliteRunApplicationService::open(&path).unwrap();
        let postgres = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "design_node_delete_effect_auth",
        )
        .unwrap();
        let services: [&dyn ApplicationService; 2] = [&sqlite, &postgres];
        for service in services {
            for principal in [
                json!({"role":"rw", "ns":["workflow"], "capabilities":["workflow.design.write"]}),
                json!({"role":"ro", "ns":["workflow"], "capabilities":["workflow.design.write", "write_graph"]}),
                json!({"role":"rw", "ns":["conversation"], "capabilities":["workflow.design.write", "write_graph"]}),
            ] {
                let response = service
                    .execute(ApiEffectRequest {
                        contract_version: 1,
                        method: "DELETE".to_owned(),
                        path_and_query: "/api/workflow/design/wf-mcp-node-delete/nodes/node-mcp-delete".to_owned(),
                        body: br#"{"workflow_id":"wf-mcp-node-delete","node_id":"node-mcp-delete","designer_id":"alice"}"#.to_vec(),
                        principal,
                    })
                    .await;
                assert_eq!(response.status, StatusCode::FORBIDDEN.as_u16());
            }
        }
        drop(sqlite);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn mcp_workflow_design_node_delete_matches_shared_contract() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use std::future::Future;
        use std::pin::Pin;
        use std::sync::{Arc, Mutex};
        use tower::ServiceExt;

        #[derive(Clone)]
        struct RequestRecorder(Arc<Mutex<Option<ApiEffectRequest>>>, Value);

        impl ApplicationService for RequestRecorder {
            fn supports_mcp_tool(&self, surface: &str, name: &str) -> bool {
                surface == "workflow" && name == "workflow.design_node_delete"
            }

            fn execute(
                &self,
                request: ApiEffectRequest,
            ) -> Pin<Box<dyn Future<Output = ApiEffectResponse> + Send + '_>> {
                let seen = Arc::clone(&self.0);
                let result = self.1.clone();
                Box::pin(async move {
                    *seen.lock().unwrap() = Some(request);
                    ApiEffectResponse {
                        status: StatusCode::OK.as_u16(),
                        content_type: "application/json".to_owned(),
                        body: serde_json::to_vec(&result).unwrap(),
                    }
                })
            }
        }

        let contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/workflow-design-node-delete-mcp-v1.json"
        ))
        .unwrap();
        let seen = Arc::new(Mutex::new(None));
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: Vec::new(),
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig::default(),
            },
            Arc::new(RequestRecorder(Arc::clone(&seen), contract["result"].clone())),
        );
        let request_body = serde_json::to_string(&json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": contract["tool"], "arguments": contract["arguments"]}
        }))
        .unwrap();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/workflow")
                    .header("content-type", "application/json")
                    .body(Body::from(request_body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let wire: Value = serde_json::from_slice(
            &to_bytes(response.into_body(), 8192).await.unwrap(),
        )
        .unwrap();
        assert_eq!(wire["result"], contract["result"]);

        let request = seen.lock().unwrap().clone().unwrap();
        assert_eq!(request.method, contract["rust_request"]["method"]);
        assert_eq!(request.path_and_query, contract["rust_request"]["path_and_query"]);
        assert_eq!(request.body, contract["rust_request"]["body"].as_str().unwrap().as_bytes());
    }
