    #[test]
    fn conversation_reads_share_frozen_storage_scope_and_acl_principal() {
        let (scope, acl) = conversation_graph_scope_and_acl_principal(&json!({
            "sub": "Alice",
            "ns": "conversation",
            "storage_ns": "ConversationData",
            "workspace_id": "Workspace-A",
            "graph_space": "graph-space-a",
            "security_scope": "Tenant-A",
            "groups": [" Team-B ", "team-a"],
            "principal_groups": ["team-a", "team-c"]
        }));

        assert_eq!(scope.namespace, "conversationdata");
        assert_eq!(scope.workspace_id.as_deref(), Some("Workspace-A"));
        assert_eq!(scope.graph_space.as_deref(), Some("graph-space-a"));
        assert_eq!(acl.principal_id, "alice");
        assert_eq!(
            acl.groups,
            BTreeSet::from([
                "team-a".to_owned(),
                "team-b".to_owned(),
                "team-c".to_owned()
            ])
        );
        assert_eq!(acl.security_scope.as_deref(), Some("tenant-a"));
    }

    #[test]
    fn graph_acl_principal_identity_matches_shared_python_contract() {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/graph-acl-principal-identity-v1.json"
        )))
        .expect("valid ACL principal identity fixture");
        assert_eq!(fixture["schema_version"], 1);
        for case in fixture["cases"].as_array().expect("identity cases") {
            assert_eq!(
                graph_acl_principal_id(&case["claims"]),
                case["expected_principal_id"]
                    .as_str()
                    .expect("expected principal ID"),
                "{}",
                case["name"].as_str().expect("case name")
            );
        }
        for case in fixture["group_cases"].as_array().expect("group cases") {
            let claims = &case["claims"];
            let groups = principal_group_set(claims);
            let (_, acl) = conversation_graph_scope_and_acl_principal(claims);
            let expected = case["expected_groups"]
                .as_array()
                .expect("expected groups")
                .iter()
                .map(|value| value.as_str().expect("group").to_owned())
                .collect::<BTreeSet<_>>();
            assert_eq!(
                groups,
                acl.groups,
                "group projection and ACL scope construction must share one normalized set"
            );
            assert_eq!(
                acl.groups,
                expected,
                "{}",
                case["name"].as_str().expect("case name")
            );
        }
    }

    #[test]
    fn security_scope_projection_matches_shared_python_contract() {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/auth-scope-projection-v1.json"
        )))
        .expect("valid auth scope projection fixture");
        assert_eq!(fixture["schema_version"], 1);
        for case in fixture["cases"].as_array().expect("scope cases") {
            let actual = security_scope_value(&case["claims"]);
            for (key, expected) in case["expected"].as_object().expect("expected mapping") {
                assert_eq!(actual[key], *expected, "{}: {key}", case["name"]);
            }
        }
    }
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use kogwistar_contracts::native_seam_attestation_is_valid;
    use tower::ServiceExt;

    fn remove_test_sqlite(path: &std::path::Path) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
        loop {
            match std::fs::remove_file(path) {
                Ok(()) => return,
                Err(error) if std::time::Instant::now() < deadline => {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                    let _ = error;
                }
                Err(error) => panic!("failed to remove test SQLite file {path:?}: {error}"),
            }
        }
    }

    fn snapshot() -> HealthSnapshot {
        HealthSnapshot {
            backend: "pg".to_owned(),
            persist_directory: "root".to_owned(),
            conversation_persist_directory: "conversation".to_owned(),
            workflow_persist_directory: "workflow".to_owned(),
            wisdom_persist_directory: "wisdom".to_owned(),
            pg_schema_base: Some("kogwistar".to_owned()),
        }
    }

    #[test]
    fn health_preserves_python_shape() {
        let value = health_response(&snapshot());
        assert_eq!(value["ok"], true);
        assert_eq!(value["backend"], "pg");
        assert_eq!(value["pg_schema_base"], "kogwistar");
    }

    #[test]
    fn latest_snapshot_route_parsing_is_exact_and_preserves_query_values() {
        assert_eq!(
            latest_snapshot_conversation_id(
                "/api/conversations/conversation-1/snapshots/latest?run_id=run-1"
            ),
            Some("conversation-1".to_owned())
        );
        assert_eq!(
            latest_snapshot_conversation_id(
                "/api/conversations/conversation-1/extra/snapshots/latest"
            ),
            None
        );
        assert_eq!(
            latest_snapshot_conversation_id("/api/conversations//snapshots/latest"),
            None
        );
        assert_eq!(
            latest_snapshot_conversation_id(
                "/api/conversations/id%20with%20spaces/snapshots/latest"
            ),
            Some("id with spaces".to_owned())
        );
        assert_eq!(
            latest_snapshot_conversation_id("/api/conversations/id%2Fother/snapshots/latest"),
            None
        );
        assert_eq!(
            context_snapshot_query(
                "/api/conversations/c/snapshots/latest?run_id=run%2F1&stage=answer+final"
            ),
            (Some("run/1".to_owned()), Some("answer final".to_owned()))
        );
        assert_eq!(
            context_snapshot_query("/api/conversations/c/snapshots/latest?run_id="),
            (Some(String::new()), None)
        );
    }

    #[test]
    fn conversation_detail_route_parsing_is_exact_and_decodes_path_id() {
        assert!(is_conversation_list_route("/api/conversations"));
        assert!(is_conversation_list_route("/api/conversations?limit=10"));
        assert!(!is_conversation_list_route("/api/conversations/"));
        assert_eq!(
            conversation_detail_id("/api/conversations/conversation-1?view=summary"),
            Some("conversation-1".to_owned())
        );
        assert_eq!(
            conversation_detail_id("/api/conversations/id%20with%20spaces"),
            Some("id with spaces".to_owned())
        );
        assert_eq!(conversation_detail_id("/api/conversations/"), None);
        assert_eq!(conversation_detail_id("/api/conversations/a/b"), None);
        assert_eq!(
            conversation_detail_id("/api/conversations/id%2Fother"),
            None
        );
        assert_eq!(conversation_detail_id("/api/conversations/c/turns"), None);
        assert_eq!(
            conversation_detail_id("/api/conversations/c/snapshots/latest"),
            None
        );
    }

    #[test]
    fn tool_audit_query_decodes_scope_and_validates_limits() {
        assert_eq!(
            tool_audit_query("/api/workflow/tools/audit?conversation_id=conversation%3A1&limit=12")
                .unwrap(),
            (Some("conversation:1".to_owned()), 12)
        );
        assert_eq!(
            tool_audit_query("/api/workflow/tools/audit").unwrap(),
            (None, 200)
        );
        assert!(tool_audit_query("/api/workflow/tools/audit?limit=-1").is_err());
        assert!(tool_audit_query("/api/workflow/tools/audit?limit=10001").is_err());
        assert!(tool_audit_query("/api/workflow/tools/audit?conversation_id=").is_err());
    }

    #[test]
    fn conversation_transcript_route_parsing_is_exact_and_decodes_path_id() {
        assert_eq!(
            conversation_transcript_id("/api/conversations/conversation-1/turns?limit=10"),
            Some("conversation-1".to_owned())
        );
        assert_eq!(
            conversation_transcript_id("/api/conversations/id%20with%20spaces/turns"),
            Some("id with spaces".to_owned())
        );
        assert_eq!(
            conversation_transcript_id("/api/conversations//turns"),
            None
        );
        assert_eq!(
            conversation_transcript_id("/api/conversations/a/b/turns"),
            None
        );
        assert_eq!(
            conversation_transcript_id("/api/conversations/id%2Fother/turns"),
            None
        );
        assert_eq!(conversation_transcript_id("/api/conversations/c"), None);
        assert_eq!(
            conversation_transcript_id("/api/conversations/c/snapshots/latest"),
            None
        );
    }

    #[tokio::test]
    async fn conversation_detail_route_checks_namespace_and_capability_before_storage() {
        let service = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "conversation_detail_auth_contract",
        )
        .unwrap();
        let make_request = |principal| ApiEffectRequest {
            contract_version: 1,
            method: "GET".to_owned(),
            path_and_query: "/api/conversations/c-1".to_owned(),
            body: Vec::new(),
            principal,
        };
        let namespace_denied = service
            .execute(make_request(json!({
                "sub": "alice",
                "ns": "docs",
                "capabilities": ["read_graph"]
            })))
            .await;
        assert_eq!(namespace_denied.status, StatusCode::FORBIDDEN.as_u16());
        assert_eq!(
            serde_json::from_slice::<Value>(&namespace_denied.body).unwrap()["code"],
            "KOGWISTAR_NAMESPACE_FORBIDDEN"
        );

        let capability_denied = service
            .execute(make_request(json!({
                "sub": "alice",
                "ns": "conversation",
                "capabilities": []
            })))
            .await;
        assert_eq!(capability_denied.status, StatusCode::FORBIDDEN.as_u16());
        assert_eq!(
            serde_json::from_slice::<Value>(&capability_denied.body).unwrap()["code"],
            "KOGWISTAR_CAPABILITY_FORBIDDEN"
        );
    }

    #[tokio::test]
    async fn tool_audit_route_checks_namespace_and_capability_before_storage() {
        let service = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "tool_audit_auth_contract",
        )
        .unwrap();
        assert!(service.supports_http_operation("GET", "/api/workflow/tools/audit"));
        let make_request = |principal| ApiEffectRequest {
            contract_version: 1,
            method: "GET".to_owned(),
            path_and_query: "/api/workflow/tools/audit?limit=5".to_owned(),
            body: Vec::new(),
            principal,
        };
        let namespace_denied = service
            .execute(make_request(json!({
                "sub": "alice",
                "ns": "docs",
                "capabilities": ["project_view"]
            })))
            .await;
        assert_eq!(namespace_denied.status, StatusCode::FORBIDDEN.as_u16());
        assert_eq!(
            serde_json::from_slice::<Value>(&namespace_denied.body).unwrap()["code"],
            "KOGWISTAR_NAMESPACE_FORBIDDEN"
        );

        let capability_denied = service
            .execute(make_request(json!({
                "sub": "alice",
                "ns": "conversation",
                "capabilities": []
            })))
            .await;
        assert_eq!(capability_denied.status, StatusCode::FORBIDDEN.as_u16());
        assert_eq!(
            serde_json::from_slice::<Value>(&capability_denied.body).unwrap()["code"],
            "KOGWISTAR_CAPABILITY_FORBIDDEN"
        );
    }

    #[tokio::test]
    async fn router_exposes_tool_audit_route_and_sqlite_fails_closed() {
        use axum::body::Body;
        use axum::http::Request;
        use tower::ServiceExt;

        let postgres = Arc::new(
            PostgresRunApplicationService::from_dsn(
                "postgres://unused:unused@127.0.0.1:1/unused",
                "tool_audit_router_contract",
            )
            .unwrap(),
        );
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: Vec::new(),
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig {
                    key: Some("route-test-secret".to_owned()),
                    ..AuthConfig::default()
                },
            },
            postgres,
        );
        let response = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/api/workflow/tools/audit")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert!(matches!(
            response.status(),
            StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN
        ));

        let path = std::env::temp_dir().join(format!(
            "kogwistar-tool-audit-sqlite-{}.sqlite3",
            uuid::Uuid::new_v4()
        ));
        let sqlite = SqliteRunApplicationService::open(&path).unwrap();
        assert!(!sqlite.supports_http_operation("GET", "/api/workflow/tools/audit"));
        let unsupported = sqlite
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: "/api/workflow/tools/audit".to_owned(),
                body: Vec::new(),
                principal: json!({"ns": "conversation", "capabilities": ["project_view"]}),
            })
            .await;
        assert_eq!(unsupported.status, StatusCode::NOT_IMPLEMENTED.as_u16());
        drop(sqlite);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn conversation_list_route_checks_namespace_capability_and_identity_before_storage() {
        let service = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "conversation_list_auth_contract",
        )
        .unwrap();
        let make_request = |principal| ApiEffectRequest {
            contract_version: 1,
            method: "GET".to_owned(),
            path_and_query: "/api/conversations".to_owned(),
            body: Vec::new(),
            principal,
        };
        for (principal, expected_status) in [
            (
                json!({"sub": "alice", "ns": "docs", "capabilities": ["read_graph"]}),
                StatusCode::FORBIDDEN,
            ),
            (
                json!({"sub": "alice", "ns": "conversation", "capabilities": []}),
                StatusCode::FORBIDDEN,
            ),
            (
                json!({"ns": "conversation", "capabilities": ["read_graph"]}),
                StatusCode::UNAUTHORIZED,
            ),
        ] {
            let response = service.execute(make_request(principal)).await;
            assert_eq!(response.status, expected_status.as_u16());
        }
    }

    #[tokio::test]
    async fn conversation_transcript_route_checks_namespace_and_capability_before_storage() {
        let service = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "conversation_transcript_auth_contract",
        )
        .unwrap();
        let make_request = |principal| ApiEffectRequest {
            contract_version: 1,
            method: "GET".to_owned(),
            path_and_query: "/api/conversations/c-1/turns".to_owned(),
            body: Vec::new(),
            principal,
        };
        let namespace_denied = service
            .execute(make_request(json!({
                "sub": "alice",
                "ns": "docs",
                "capabilities": ["read_graph"]
            })))
            .await;
        assert_eq!(namespace_denied.status, StatusCode::FORBIDDEN.as_u16());
        assert_eq!(
            serde_json::from_slice::<Value>(&namespace_denied.body).unwrap()["code"],
            "KOGWISTAR_NAMESPACE_FORBIDDEN"
        );

        let capability_denied = service
            .execute(make_request(json!({
                "sub": "alice",
                "ns": "conversation",
                "capabilities": []
            })))
            .await;
        assert_eq!(capability_denied.status, StatusCode::FORBIDDEN.as_u16());
        assert_eq!(
            serde_json::from_slice::<Value>(&capability_denied.body).unwrap()["code"],
            "KOGWISTAR_CAPABILITY_FORBIDDEN"
        );
    }

    #[tokio::test]
    async fn conversation_transcript_authorization_is_bound_to_request_principal() {
        let service = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "conversation_transcript_principal_binding",
        )
        .unwrap();
        let authorized_request = ApiEffectRequest {
            contract_version: 1,
            method: "GET".to_owned(),
            path_and_query: "/api/conversations/c-1/turns".to_owned(),
            body: Vec::new(),
            principal: json!({
                "sub": "alice",
                "ns": "conversation",
                "capabilities": ["read_graph"]
            }),
        };
        let authorization = service
            .authorize_conversation_transcript(&authorized_request)
            .unwrap();
        let other_principal_request = ApiEffectRequest {
            principal: json!({
                "sub": "bob",
                "ns": "conversation",
                "capabilities": ["read_graph"]
            }),
            ..authorized_request
        };

        let result = service
            .conversation_transcript_projection(
                &other_principal_request,
                InternalConversationTranscriptRequest {
                    conversation_id: "c-1".to_owned(),
                    limit: 10,
                },
                authorization,
            )
            .await;

        let Err(response) = result else {
            panic!("authorization for one principal must not authorize another");
        };
        assert_eq!(response.status, StatusCode::FORBIDDEN.as_u16());
        assert_eq!(
            serde_json::from_slice::<Value>(&response.body).unwrap()["code"],
            "KOGWISTAR_CAPABILITY_FORBIDDEN"
        );
    }

    #[tokio::test]
    async fn latest_snapshot_route_checks_namespace_and_capability_before_storage() {
        let service = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "snapshot_route_auth_contract",
        )
        .unwrap();
        let make_request = |principal| ApiEffectRequest {
            contract_version: 1,
            method: "GET".to_owned(),
            path_and_query: "/api/conversations/c-1/snapshots/latest".to_owned(),
            body: Vec::new(),
            principal,
        };
        let namespace_denied = service
            .execute(make_request(json!({
                "sub": "alice",
                "ns": "docs",
                "capabilities": ["project_view"]
            })))
            .await;
        assert_eq!(
            namespace_denied.status,
            StatusCode::FORBIDDEN.as_u16(),
            "{}",
            String::from_utf8_lossy(&namespace_denied.body)
        );
        assert_eq!(
            serde_json::from_slice::<Value>(&namespace_denied.body).unwrap()["code"],
            "KOGWISTAR_NAMESPACE_FORBIDDEN"
        );

        let capability_denied = service
            .execute(make_request(json!({
                "sub": "alice",
                "ns": "workflow",
                "capabilities": []
            })))
            .await;
        assert_eq!(capability_denied.status, StatusCode::FORBIDDEN.as_u16());
        assert_eq!(
            serde_json::from_slice::<Value>(&capability_denied.body).unwrap()["code"],
            "KOGWISTAR_CAPABILITY_FORBIDDEN"
        );
    }

    #[tokio::test]
    async fn mount_memory_syscall_checks_namespace_and_capability_before_storage() {
        let service = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "mount_memory_auth_contract",
        )
        .unwrap();
        let make_request = |principal| ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/syscall/v1/mount_memory".to_owned(),
            body: serde_json::to_vec(&json!({
                "version": "v1",
                "op": "mount_memory",
                "args": {"conversation_id": "c-1"}
            }))
            .unwrap(),
            principal,
        };
        let namespace_denied = service
            .execute(make_request(json!({
                "sub": "alice",
                "ns": "docs",
                "capabilities": ["project_view"]
            })))
            .await;
        assert_eq!(namespace_denied.status, StatusCode::FORBIDDEN.as_u16());
        assert_eq!(
            serde_json::from_slice::<Value>(&namespace_denied.body).unwrap()["code"],
            "KOGWISTAR_CAPABILITY_FORBIDDEN"
        );

        let capability_denied = service
            .execute(make_request(json!({
                "sub": "alice",
                "ns": "conversation",
                "capabilities": []
            })))
            .await;
        assert_eq!(capability_denied.status, StatusCode::FORBIDDEN.as_u16());
        assert_eq!(
            serde_json::from_slice::<Value>(&capability_denied.body).unwrap()["code"],
            "KOGWISTAR_CAPABILITY_FORBIDDEN"
        );
    }

    #[tokio::test]
    async fn invoke_tool_checks_builtin_capabilities_before_storage() {
        let service = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "invoke_tool_auth_contract",
        )
        .unwrap();
        let make_request = |tool_name: &str, args: Value| {
            let mut args = args.as_object().cloned().unwrap_or_default();
            args.insert("tool_name".to_owned(), json!(tool_name));
            ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/syscall/v1/invoke_tool".to_owned(),
                body: serde_json::to_vec(&json!({
                    "version": "v1",
                    "op": "invoke_tool",
                    "args": args
                }))
                .unwrap(),
                principal: json!({
                    "sub": "alice",
                    "role": "rw",
                    "ns": "conversation",
                    "capabilities": []
                }),
            }
        };

        let transcript_denied = service
            .execute(make_request("list_transcript", json!({"conversation_id": "c-1"})))
            .await;
        assert_eq!(transcript_denied.status, StatusCode::FORBIDDEN.as_u16());
        assert_eq!(
            serde_json::from_slice::<Value>(&transcript_denied.body).unwrap()["code"],
            "KOGWISTAR_CAPABILITY_FORBIDDEN"
        );

        let snapshot_denied = service
            .execute(make_request("latest_snapshot", json!({"conversation_id": "c-1"})))
            .await;
        assert_eq!(snapshot_denied.status, StatusCode::FORBIDDEN.as_u16());
        assert_eq!(
            serde_json::from_slice::<Value>(&snapshot_denied.body).unwrap()["code"],
            "KOGWISTAR_CAPABILITY_FORBIDDEN"
        );

        let capability_snapshot_denied = service
            .execute(make_request("capability_snapshot", json!({})))
            .await;
        assert_eq!(
            capability_snapshot_denied.status,
            StatusCode::FORBIDDEN.as_u16()
        );
        assert_eq!(
            serde_json::from_slice::<Value>(&capability_snapshot_denied.body).unwrap()["code"],
            "KOGWISTAR_CAPABILITY_FORBIDDEN"
        );
    }

    #[tokio::test]
    async fn invoke_tool_capability_snapshot_uses_existing_authority_projection() {
        let service = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "invoke_tool_capability_snapshot",
        )
        .unwrap();
        let response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/syscall/v1/invoke_tool".to_owned(),
                body: serde_json::to_vec(&json!({
                    "version": "v1",
                    "op": "invoke_tool",
                    "args": {"tool_name": "capability_snapshot"}
                }))
                .unwrap(),
                principal: json!({
                    "sub": "alice",
                    "role": "rw",
                    "ns": "conversation",
                    "capabilities": ["project_view", "read_security_scope"]
                }),
            })
            .await;

        assert_eq!(response.status, StatusCode::OK.as_u16());
        let payload: Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(payload["op"], "invoke_tool");
        assert_eq!(payload["result"]["tool_name"], "capability_snapshot");
        assert_eq!(payload["result"]["current_subject"], "alice");
        assert!(payload["result"]["specs"].as_array().is_some());
        assert!(payload["result"]["effective_capabilities"]
            .as_array()
            .is_some_and(|capabilities| capabilities
                .iter()
                .any(|capability| capability == "project_view")));
    }

    #[tokio::test]
    async fn context_snapshot_authorization_is_bound_to_request_principal() {
        let service = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "context_snapshot_principal_binding",
        )
        .unwrap();
        let authorized_request = ApiEffectRequest {
            contract_version: 1,
            method: "GET".to_owned(),
            path_and_query: "/api/conversations/c-1/snapshots/latest".to_owned(),
            body: Vec::new(),
            principal: json!({
                "sub": "alice",
                "ns": "conversation",
                "capabilities": ["project_view"]
            }),
        };
        let authorization = service.authorize_context_snapshot(&authorized_request).unwrap();
        let other_principal_request = ApiEffectRequest {
            principal: json!({
                "sub": "bob",
                "ns": "conversation",
                "capabilities": ["project_view"]
            }),
            ..authorized_request
        };

        let result = service
            .context_snapshot_projection(
                &other_principal_request,
                InternalContextSnapshotRequest {
                    conversation_id: "c-1".to_owned(),
                    run_id: None,
                    stage: None,
                },
                authorization,
            )
            .await;

        let Err(response) = result else {
            panic!("authorization for one principal must not authorize another");
        };
        assert_eq!(response.status, StatusCode::FORBIDDEN.as_u16());
        assert_eq!(
            serde_json::from_slice::<Value>(&response.body).unwrap()["code"],
            "KOGWISTAR_CAPABILITY_FORBIDDEN"
        );
    }

    #[tokio::test]
    #[ignore = "requires KOGWISTAR_TEST_PG_DSN; run explicitly with --include-ignored"]
    async fn postgres_latest_snapshot_route_projects_authorized_snapshot_when_dsn_available() {
        let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
            .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
        let schema = format!("rust_snapshot_route_{}", uuid::Uuid::new_v4().simple());
        let service = PostgresRunApplicationService::from_dsn(&dsn, &schema).unwrap();
        service.ensure_schema().await.unwrap();
        service
            .store
            .create_graph_schema(3, kogwistar_store_postgres::GraphTableNames::default())
            .await
            .unwrap();
        let scope = GraphScope {
            namespace: "conversation".to_owned(),
            workspace_id: None,
            graph_space: None,
        };
        let expected = serde_json::from_str::<Value>(include_str!(
            "../../../../../contracts/golden/mount-memory-result-v1.json"
        ))
        .unwrap();
        let mut snapshot_metadata = Map::new();
        snapshot_metadata.insert("entity_type".to_owned(), json!("context_snapshot"));
        snapshot_metadata.insert(
            "conversation_id".to_owned(),
            expected["conversation_id"].clone(),
        );
        snapshot_metadata.insert("run_id".to_owned(), json!("run-1"));
        snapshot_metadata.insert("stage".to_owned(), json!("final"));
        service
            .store
            .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                record: GraphRecord {
                    id: expected["snapshot_node_id"].as_str().unwrap().to_owned(),
                    document: Some(expected.to_string()),
                    metadata: snapshot_metadata,
                    embedding: None,
                },
                embedding_dim: 3,
            })
            .await
            .unwrap();

        let mut private_snapshot_metadata = Map::new();
        private_snapshot_metadata.insert("entity_type".to_owned(), json!("context_snapshot"));
        private_snapshot_metadata.insert(
            "conversation_id".to_owned(),
            expected["conversation_id"].clone(),
        );
        private_snapshot_metadata.insert("run_id".to_owned(), json!("run-private"));
        private_snapshot_metadata.insert("stage".to_owned(), json!("final"));
        service
            .store
            .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                record: GraphRecord {
                    id: "snapshot-private".to_owned(),
                    document: Some(json!({"summary": "private snapshot"}).to_string()),
                    metadata: private_snapshot_metadata,
                    embedding: None,
                },
                embedding_dim: 3,
            })
            .await
            .unwrap();
        let mut private_acl_metadata = Map::new();
        for (key, value) in [
            ("entity_type", json!("acl_record")),
            ("acl_truth_graph", json!("conversation")),
            ("acl_target_entity_id", json!("snapshot-private")),
            ("acl_target_grain", json!("node")),
            ("acl_version", json!(1)),
            ("acl_mode", json!("private")),
            ("owner_id", json!("bob")),
        ] {
            private_acl_metadata.insert(key.to_owned(), value);
        }
        service
            .store
            .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                record: GraphRecord {
                    id: "acl-snapshot-private".to_owned(),
                    document: None,
                    metadata: private_acl_metadata,
                    embedding: None,
                },
                embedding_dim: 3,
            })
            .await
            .unwrap();
        let mut acl_metadata = Map::new();
        for (key, value) in [
            ("entity_type", json!("acl_record")),
            ("acl_truth_graph", json!("conversation")),
            ("acl_target_entity_id", json!("snapshot-1")),
            ("acl_target_grain", json!("node")),
            ("acl_version", json!(1)),
            ("acl_mode", json!("public")),
        ] {
            acl_metadata.insert(key.to_owned(), value);
        }
        service
            .store
            .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                record: GraphRecord {
                    id: "acl-snapshot-1".to_owned(),
                    document: None,
                    metadata: acl_metadata,
                    embedding: None,
                },
                embedding_dim: 3,
            })
            .await
            .unwrap();

        let response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query:
                    "/api/conversations/conversation-1/snapshots/latest?run_id=run-1&stage=final"
                        .to_owned(),
                body: Vec::new(),
                principal: json!({
                    "sub": "alice",
                    "role": "ro",
                    "ns": "workflow",
                    "storage_ns": "conversation",
                    "capabilities": ["project_view"]
                }),
            })
            .await;
        assert_eq!(response.status, StatusCode::OK.as_u16());
        let payload: Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(payload["snapshot_node_id"], "snapshot-1");
        assert_eq!(payload["conversation_id"], "conversation-1");
        assert_eq!(payload["metadata"]["source"], "integration");
        assert_eq!(payload["properties"]["summary"], "latest snapshot");
        assert_eq!(
            payload
                .as_object()
                .unwrap()
                .keys()
                .cloned()
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([
                "conversation_id".to_owned(),
                "metadata".to_owned(),
                "properties".to_owned(),
                "snapshot_node_id".to_owned(),
            ])
        );

        let syscall_response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/syscall/v1/mount_memory".to_owned(),
                body: serde_json::to_vec(&json!({
                    "version": "v1",
                    "op": "mount_memory",
                    "args": {
                        "conversation_id": expected["conversation_id"],
                        "run_id": "run-1",
                        "stage": "final"
                    }
                }))
                .unwrap(),
                principal: json!({
                    "sub": "alice",
                    "role": "ro",
                    "ns": "conversation",
                    "storage_ns": "conversation",
                    "capabilities": ["project_view"]
                }),
            })
            .await;
        assert_eq!(syscall_response.status, StatusCode::OK.as_u16());
        let syscall_payload: Value = serde_json::from_slice(&syscall_response.body).unwrap();
        assert_eq!(syscall_payload["version"], "v1");
        assert_eq!(syscall_payload["op"], "mount_memory");
        assert_eq!(syscall_payload["status"], "ok");
        assert_eq!(syscall_payload["result"], expected);

        let private_syscall_response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/syscall/v1/mount_memory".to_owned(),
                body: serde_json::to_vec(&json!({
                    "version": "v1",
                    "op": "mount_memory",
                    "args": {
                        "conversation_id": expected["conversation_id"],
                        "run_id": "run-private",
                        "stage": "final"
                    }
                }))
                .unwrap(),
                principal: json!({
                    "sub": "alice",
                    "role": "ro",
                    "ns": "conversation",
                    "storage_ns": "conversation",
                    "capabilities": ["project_view"]
                }),
            })
            .await;
        assert_eq!(
            private_syscall_response.status,
            StatusCode::NOT_FOUND.as_u16()
        );
        let private_payload: Value =
            serde_json::from_slice(&private_syscall_response.body).unwrap();
        assert_eq!(
            private_payload["code"],
            "KOGWISTAR_CONTEXT_SNAPSHOT_NOT_FOUND"
        );
        assert!(!private_payload.to_string().contains("private snapshot"));

        let invoke_snapshot_response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/syscall/v1/invoke_tool".to_owned(),
                body: serde_json::to_vec(&json!({
                    "version": "v1",
                    "op": "invoke_tool",
                    "args": {
                        "tool_name": "latest_snapshot",
                        "conversation_id": expected["conversation_id"],
                        "run_id": "run-1",
                        "stage": "final"
                    }
                }))
                .unwrap(),
                principal: json!({
                    "sub": "alice",
                    "role": "rw",
                    "ns": "conversation",
                    "storage_ns": "conversation",
                    "capabilities": ["project_view"]
                }),
            })
            .await;
        assert_eq!(invoke_snapshot_response.status, StatusCode::OK.as_u16());
        let invoke_snapshot: Value =
            serde_json::from_slice(&invoke_snapshot_response.body).unwrap();
        assert_eq!(invoke_snapshot["version"], "v1");
        assert_eq!(invoke_snapshot["op"], "invoke_tool");
        assert_eq!(invoke_snapshot["status"], "ok");
        assert_eq!(
            invoke_snapshot["result"],
            json!({
                "tool_name": "latest_snapshot",
                "conversation_id": expected["conversation_id"],
                "snapshot": expected
            })
        );

        drop(service);
        let (cleanup, connection) = tokio_postgres::connect(&dsn, tokio_postgres::NoTls)
            .await
            .unwrap();
        tokio::spawn(async move {
            let _ = connection.await;
        });
        cleanup
            .batch_execute(&format!("DROP SCHEMA \"{schema}\" CASCADE"))
            .await
            .unwrap();
    }

    #[tokio::test]
    #[ignore = "requires KOGWISTAR_TEST_PG_DSN; run explicitly with --include-ignored"]
    async fn postgres_conversation_routes_project_authorized_summary_and_transcript_when_dsn_available()
     {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use tower::ServiceExt;

        let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
            .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
        let schema = format!("rust_conversation_detail_{}", uuid::Uuid::new_v4().simple());
        let service = PostgresRunApplicationService::from_dsn(&dsn, &schema).unwrap();
        service.ensure_schema().await.unwrap();
        service
            .store
            .create_graph_schema(3, kogwistar_store_postgres::GraphTableNames::default())
            .await
            .unwrap();
        let scope = GraphScope {
            namespace: "conversation".to_owned(),
            workspace_id: None,
            graph_space: None,
        };
        for (id, conversation_id, entity_type, extra) in [
            (
                "start-1",
                "conversation-1",
                "conversation_start",
                json!({"user_id": "alice", "status": "active"}),
            ),
            (
                "turn-1",
                "conversation-1",
                "conversation_turn",
                json!({"turn_index": 0, "role": "user", "summary": "hello"}),
            ),
            (
                "start-other-user",
                "conversation-other",
                "conversation_start",
                json!({"user_id": "bob", "status": "active"}),
            ),
        ] {
            let mut metadata = Map::new();
            metadata.insert("entity_type".to_owned(), json!(entity_type));
            metadata.insert("conversation_id".to_owned(), json!(conversation_id));
            metadata.extend(extra.as_object().unwrap().clone());
            service
                .store
                .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                    scope: scope.clone(),
                    table: "gke_nodes".to_owned(),
                    record: GraphRecord {
                        id: id.to_owned(),
                        document: None,
                        metadata,
                        embedding: None,
                    },
                    embedding_dim: 3,
                })
                .await
                .unwrap();
            let mut acl_metadata = Map::new();
            for (key, value) in [
                ("entity_type", json!("acl_record")),
                ("acl_truth_graph", json!("conversation")),
                ("acl_target_entity_id", json!(id)),
                ("acl_target_grain", json!("node")),
                ("acl_version", json!(1)),
                ("acl_mode", json!("public")),
            ] {
                acl_metadata.insert(key.to_owned(), value);
            }
            service
                .store
                .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                    scope: scope.clone(),
                    table: "gke_nodes".to_owned(),
                    record: GraphRecord {
                        id: format!("acl-{id}"),
                        document: None,
                        metadata: acl_metadata,
                        embedding: None,
                    },
                    embedding_dim: 3,
                })
                .await
                .unwrap();
        }

        for (id, entity_type, summary, status, acl_mode) in [
            (
                "tool-call-1",
                "tool_call",
                "Read graph neighbors",
                "requested",
                "public",
            ),
            (
                "tool-result-1",
                "tool_result",
                "Returned one neighbor",
                "completed",
                "public",
            ),
            (
                "tool-call-other",
                "tool_call",
                "Other user's private call",
                "requested",
                "public",
            ),
            (
                "tool-call-private",
                "tool_call",
                "Private call owned by another user",
                "requested",
                "private",
            ),
        ] {
            let conversation_id = if id == "tool-call-other" {
                "conversation-other"
            } else {
                "conversation-1"
            };
            let mut metadata = Map::new();
            metadata.insert("entity_type".to_owned(), json!(entity_type));
            metadata.insert("conversation_id".to_owned(), json!(conversation_id));
            metadata.insert("tool_name".to_owned(), json!("kg_neighbors"));
            metadata.insert("summary".to_owned(), json!(summary));
            metadata.insert("turn_index".to_owned(), json!(0));
            let record = GraphRecord {
                id: id.to_owned(),
                document: Some(json!({"properties": {"status": status}}).to_string()),
                metadata,
                embedding: None,
            };
            service
                .store
                .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                    scope: scope.clone(),
                    table: "gke_nodes".to_owned(),
                    record,
                    embedding_dim: 3,
                })
                .await
                .unwrap();
            let mut acl_metadata = Map::new();
            for (key, value) in [
                ("entity_type", json!("acl_record")),
                ("acl_truth_graph", json!("conversation")),
                ("acl_target_entity_id", json!(id)),
                ("acl_target_grain", json!("node")),
                ("acl_version", json!(1)),
                ("acl_mode", json!(acl_mode)),
            ] {
                acl_metadata.insert(key.to_owned(), value);
            }
            if acl_mode == "private" {
                acl_metadata.insert("owner_id".to_owned(), json!("bob"));
            }
            service
                .store
                .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                    scope: scope.clone(),
                    table: "gke_nodes".to_owned(),
                    record: GraphRecord {
                        id: format!("acl-{id}"),
                        document: None,
                        metadata: acl_metadata,
                        embedding: None,
                    },
                    embedding_dim: 3,
                })
                .await
                .unwrap();
        }

        let response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: "/api/conversations/conversation-1".to_owned(),
                body: Vec::new(),
                principal: json!({
                    "sub": "alice",
                    "role": "ro",
                    "ns": "conversation",
                    "storage_ns": "conversation",
                    "capabilities": ["read_graph"]
                }),
            })
            .await;
        assert_eq!(response.status, StatusCode::OK.as_u16());
        let payload: Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(payload["conversation_id"], "conversation-1");
        assert_eq!(payload["user_id"], "alice");
        assert_eq!(payload["status"], "active");
        assert_eq!(payload["start_node_id"], "start-1");
        assert_eq!(payload["tail_node_id"], "turn-1");
        assert_eq!(payload["turn_count"], 1);
        assert!(payload.get("schema_version").is_none());
        assert!(payload.get("implementation_authority").is_none());
        assert!(payload.get("seam_attestation_version").is_none());
        assert!(payload.get("seam_crossings").is_none());
        assert_eq!(
            payload
                .as_object()
                .unwrap()
                .keys()
                .cloned()
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([
                "conversation_id".to_owned(),
                "start_node_id".to_owned(),
                "status".to_owned(),
                "tail_node_id".to_owned(),
                "turn_count".to_owned(),
                "user_id".to_owned(),
            ])
        );

        let list_response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: "/api/conversations".to_owned(),
                body: Vec::new(),
                principal: json!({
                    "sub": "alice",
                    "role": "ro",
                    "ns": "conversation",
                    "storage_ns": "conversation",
                    "capabilities": ["read_graph"]
                }),
            })
            .await;
        assert_eq!(list_response.status, StatusCode::OK.as_u16());
        let list: Value = serde_json::from_slice(&list_response.body).unwrap();
        assert_eq!(
            list,
            json!({
                "conversations": [{
                    "id": "conversation-1",
                    "start_node_id": "start-1",
                    "status": "active",
                    "turn_count": 1
                }]
            })
        );

        let transcript_response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: "/api/conversations/conversation-1/turns".to_owned(),
                body: Vec::new(),
                principal: json!({
                    "sub": "alice",
                    "role": "ro",
                    "ns": "conversation",
                    "storage_ns": "conversation",
                    "capabilities": ["read_graph"]
                }),
            })
            .await;
        assert_eq!(transcript_response.status, StatusCode::OK.as_u16());
        let transcript: Value = serde_json::from_slice(&transcript_response.body).unwrap();
        assert_eq!(transcript["conversation_id"], "conversation-1");
        assert_eq!(transcript["turns"].as_array().unwrap().len(), 1);
        assert_eq!(transcript["turns"][0]["node_id"], "turn-1");
        assert_eq!(transcript["turns"][0]["turn_index"], 0);
        assert_eq!(transcript["turns"][0]["role"], "user");
        assert_eq!(transcript["turns"][0]["content"], "hello");
        assert_eq!(transcript["turns"][0]["entity_type"], "conversation_turn");
        assert!(transcript.get("schema_version").is_none());
        assert!(transcript.get("implementation_authority").is_none());
        assert!(transcript["turns"][0].get("schema_version").is_none());
        assert_eq!(
            transcript
                .as_object()
                .unwrap()
                .keys()
                .cloned()
                .collect::<BTreeSet<_>>(),
            BTreeSet::from(["conversation_id".to_owned(), "turns".to_owned()])
        );

        let audit_response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: "/api/workflow/tools/audit?conversation_id=conversation-1&limit=10"
                    .to_owned(),
                body: Vec::new(),
                principal: json!({
                    "sub": "alice",
                    "role": "ro",
                    "ns": "conversation",
                    "storage_ns": "conversation",
                    "capabilities": ["project_view", "read_graph"]
                }),
            })
            .await;
        assert_eq!(audit_response.status, StatusCode::OK.as_u16());
        let audit: Value = serde_json::from_slice(&audit_response.body).unwrap();
        assert_eq!(audit["conversation_id"], "conversation-1");
        assert_eq!(audit["items"].as_array().unwrap().len(), 2);
        assert_eq!(audit["items"][0]["node_id"], "tool-call-1");
        assert_eq!(audit["items"][1]["node_id"], "tool-result-1");
        assert_eq!(audit["items"][1]["properties"]["status"], "completed");
        assert!(audit["items"].as_array().unwrap().iter().all(|item| {
            item["node_id"] != "tool-call-private" && item["node_id"] != "tool-call-other"
        }));
        assert!(audit.get("schema_version").is_none());
        assert!(audit["items"][0].get("schema_version").is_none());

        let receive_response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/syscall/v1/receive_message".to_owned(),
                body: serde_json::to_vec(&json!({
                    "version": "v1",
                    "op": "receive_message",
                    "args": {"conversation_id": "conversation-1", "limit": 0}
                }))
                .unwrap(),
                principal: json!({
                    "sub": "alice",
                    "role": "ro",
                    "ns": "conversation",
                    "storage_ns": "conversation",
                    "capabilities": ["read_graph"]
                }),
            })
            .await;
        assert_eq!(receive_response.status, StatusCode::OK.as_u16());
        let receive_payload: Value = serde_json::from_slice(&receive_response.body).unwrap();
        let expected_receive = serde_json::from_str::<Value>(include_str!(
            "../../../../../contracts/golden/receive-message-result-v1.json"
        ))
        .unwrap();
        assert_eq!(receive_payload["version"], "v1");
        assert_eq!(receive_payload["op"], "receive_message");
        assert_eq!(receive_payload["status"], "ok");
        assert_eq!(receive_payload["result"], expected_receive);

        let invoke_transcript_response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/syscall/v1/invoke_tool".to_owned(),
                body: serde_json::to_vec(&json!({
                    "version": "v1",
                    "op": "invoke_tool",
                    "args": {
                        "tool_name": "list_transcript",
                        "conversation_id": "conversation-1"
                    }
                }))
                .unwrap(),
                principal: json!({
                    "sub": "alice",
                    "role": "rw",
                    "ns": "conversation",
                    "storage_ns": "conversation",
                    "capabilities": ["read_graph"]
                }),
            })
            .await;
        assert_eq!(invoke_transcript_response.status, StatusCode::OK.as_u16());
        let invoke_transcript: Value =
            serde_json::from_slice(&invoke_transcript_response.body).unwrap();
        assert_eq!(invoke_transcript["version"], "v1");
        assert_eq!(invoke_transcript["op"], "invoke_tool");
        assert_eq!(invoke_transcript["status"], "ok");
        assert_eq!(
            invoke_transcript["result"],
            json!({
                "tool_name": "list_transcript",
                "conversation_id": "conversation-1",
                "messages": expected_receive["messages"]
            })
        );

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let auth = AuthConfig {
            mode: "dev".to_owned(),
            algorithm: Some("HS256".to_owned()),
            key: Some("tool-audit-mcp-secret".to_owned()),
            jwks_json: None,
            issuer: Some("local".to_owned()),
            audience: None,
            oidc_providers_json: None,
            auth_db_url: None,
        };
        let token = encode(
            &Header::new(Algorithm::HS256),
            &json!({
                "sub": "alice",
                "role": "ro",
                "ns": ["workflow"],
                "storage_ns": "conversation",
                "capabilities": ["project_view", "read_graph"],
                "iss": "local",
                "exp": now + 300
            }),
            &EncodingKey::from_secret(b"tool-audit-mcp-secret"),
        )
        .unwrap();
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: Vec::new(),
                implementation: ImplementationSnapshot::default(),
                auth,
            },
            Arc::new(service),
        );
        let http_audit = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/api/workflow/tools/audit?conversation_id=conversation-1&limit=10")
                    .header("authorization", format!("Bearer {token}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(http_audit.status(), StatusCode::OK);
        let http_audit: Value =
            serde_json::from_slice(&to_bytes(http_audit.into_body(), 65_536).await.unwrap())
                .unwrap();
        assert_eq!(http_audit, audit);

        let mcp_audit = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/workflow")
                    .header("content-type", "application/json")
                    .header("authorization", format!("Bearer {token}"))
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":"audit","method":"tools/call","params":{"name":"workflow.tool_audit","arguments":{"conversation_id":"conversation-1","limit":10}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(mcp_audit.status(), StatusCode::OK);
        let mcp_audit: Value =
            serde_json::from_slice(&to_bytes(mcp_audit.into_body(), 65_536).await.unwrap())
                .unwrap();
        assert_eq!(mcp_audit["result"]["items"].as_array().unwrap().len(), 2);
        assert_eq!(mcp_audit["result"]["items"][0]["node_id"], "tool-call-1");
        assert_eq!(mcp_audit["result"]["items"][1]["node_id"], "tool-result-1");

        let (cleanup, connection) = tokio_postgres::connect(&dsn, tokio_postgres::NoTls)
            .await
            .unwrap();
        tokio::spawn(async move {
            let _ = connection.await;
        });
        cleanup
            .batch_execute(&format!("DROP SCHEMA \"{schema}\" CASCADE"))
            .await
            .unwrap();
    }

    #[test]
    fn server_readiness_cannot_overclaim_pending_routes() {
        let mut implementation = ImplementationSnapshot::default();
        assert!(!implementation.effective_server_cutover_ready());

        implementation.server_cutover_ready = true;
        assert!(!implementation.effective_server_cutover_ready());

        implementation.implemented_route_operations = implementation.frozen_route_operations;
        assert!(!implementation.effective_server_cutover_ready());
    }

    #[test]
    fn unavailable_application_does_not_claim_registered_http_routes() {
        let service = UnavailableApplicationService;
        assert_eq!(service.implemented_http_operations(), 0);
    }

    #[test]
    fn application_route_accounting_excludes_transport_owned_handlers() {
        assert_eq!(
            super::cutover::frozen_application_route_operations()
                + TRANSPORT_OWNED_HTTP_ROUTES.len(),
            FROZEN_OPENAPI_ROUTES.len()
        );
        let service = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "route_accounting_contract",
        )
        .unwrap();
        assert!(!service.supports_http_operation("GET", "/health"));
        assert!(!service.supports_http_operation("GET", "/api/auth/me"));
        assert!(!service.supports_http_operation("POST", "/internal/conversation/transcript"));
        assert!(!service.supports_http_operation("GET", "/api/not-registered"));
        assert!(service.supports_http_operation("POST", "/api/workflow/runs"));
        assert!(service.supports_http_operation("GET", "/api/conversations"));
        assert!(service.supports_http_operation("GET", "/api/workflow/tools/audit"));
        assert!(service.supports_http_operation("GET", "/api/conversations/{conversation_id}"));
        assert!(
            service.supports_http_operation("GET", "/api/conversations/{conversation_id}/turns")
        );
        assert!(service.supports_http_operation(
            "GET",
            "/api/conversations/{conversation_id}/snapshots/latest"
        ));
    }

    #[test]
    fn mcp_advertisement_requires_concrete_authority() {
        let postgres = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "mcp_authority_contract",
        )
        .unwrap();
        assert!(postgres.supports_mcp_tool("root", "kg_neighbors"));
        assert!(postgres.supports_mcp_tool("workflow", "workflow.run_status"));
        assert!(postgres.supports_mcp_tool("workflow", "workflow.tool_audit"));

        let sqlite_path = std::env::temp_dir().join(format!(
            "kogwistar-api-mcp-authority-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let sqlite = SqliteRunApplicationService::open(&sqlite_path).unwrap();
        assert!(!sqlite.supports_mcp_tool("root", "kg_neighbors"));
        assert!(sqlite.supports_mcp_tool("workflow", "workflow.run_status"));
        assert!(!sqlite.supports_mcp_tool("workflow", "workflow.tool_audit"));
        assert!(!sqlite.supports_http_operation("GET", "/api/conversations"));
        drop(sqlite);
        remove_test_sqlite(&sqlite_path);
    }

    #[tokio::test]
    async fn pending_routes_fail_closed_for_every_rust_service() {
        let sqlite_path = std::env::temp_dir().join(format!(
            "kogwistar-api-document-route-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let sqlite = SqliteRunApplicationService::open(&sqlite_path).unwrap();
        let postgres = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "document_route_contract",
        )
        .unwrap();

        for service in [&sqlite as &dyn ApplicationService, &postgres] {
            for &(method, path) in PENDING_SERVER_CUTOVER_ROUTES {
                assert!(
                    !service.supports_http_operation(method, path),
                    "deferred route was advertised: {method} {path}"
                );
                let concrete_path = path
                    .replace("{doc_id}", "deferred-probe")
                    .replace("{conversation_id}", "deferred-probe");
                let response = service
                    .execute(ApiEffectRequest {
                        contract_version: 1,
                        method: method.to_owned(),
                        path_and_query: concrete_path,
                        body: Vec::new(),
                        principal: json!({
                            "sub": "alice",
                            "ns": ["workflow", "conversation"],
                            "capabilities": ["*"]
                        }),
                    })
                    .await;
                assert_eq!(
                    response.status,
                    StatusCode::NOT_IMPLEMENTED.as_u16(),
                    "deferred route executed: {method} {path}"
                );
                assert_eq!(
                    serde_json::from_slice::<Value>(&response.body).unwrap()["code"],
                    "KOGWISTAR_ROUTE_DEFERRED",
                    "deferred route returned the wrong failure: {method} {path}"
                );
            }
        }
        assert_eq!(
            sqlite.document_authority_capability(),
            DocumentAuthorityCapability::Unsupported
        );
        assert_eq!(
            postgres.document_authority_capability(),
            DocumentAuthorityCapability::ProjectionUpsert
        );
        assert!(!sqlite.supports_http_operation(
            "GET",
            "/api/conversations/{conversation_id}/snapshots/latest"
        ));

        drop(sqlite);
        remove_test_sqlite(&sqlite_path);
    }

    #[tokio::test]
    async fn sqlite_metadata_store_cannot_advertise_or_route_graph_authority() {
        let sqlite_path = std::env::temp_dir().join(format!(
            "kogwistar-api-sqlite-graph-authority-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let sqlite = SqliteRunApplicationService::open(&sqlite_path).unwrap();

        assert_eq!(
            sqlite.graph_authority_capability(),
            GraphAuthorityCapability::Unsupported
        );
        assert!(!sqlite.supports_mcp_tool("root", "kg_neighbors"));
        assert!(!sqlite.supports_http_operation("POST", "/internal/graph/neighbors"));
        assert!(!sqlite.supports_http_operation("POST", "/internal/conversation/transcript"));
        assert!(!sqlite.supports_http_operation("GET", "/api/conversations/{conversation_id}"));
        assert!(
            !sqlite.supports_http_operation("GET", "/api/conversations/{conversation_id}/turns")
        );

        let response = sqlite
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/internal/graph/neighbors".to_owned(),
                body: br#"{}"#.to_vec(),
                principal: json!({"sub": "test"}),
            })
            .await;
        assert_eq!(response.status, StatusCode::NOT_IMPLEMENTED.as_u16());
        assert_eq!(
            serde_json::from_slice::<Value>(&response.body).unwrap()["code"],
            "KOGWISTAR_GRAPH_AUTHORITY_UNSUPPORTED"
        );

        let response = sqlite
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: "/api/conversations/c-1/turns".to_owned(),
                body: Vec::new(),
                principal: json!({
                    "sub": "alice",
                    "ns": "conversation",
                    "capabilities": ["read_graph"]
                }),
            })
            .await;
        assert_eq!(response.status, StatusCode::NOT_IMPLEMENTED.as_u16());
        assert_eq!(
            serde_json::from_slice::<Value>(&response.body).unwrap()["code"],
            "KOGWISTAR_GRAPH_AUTHORITY_UNSUPPORTED"
        );

        let response = sqlite
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: "/api/conversations/c-1".to_owned(),
                body: Vec::new(),
                principal: json!({
                    "sub": "alice",
                    "ns": "conversation",
                    "capabilities": ["read_graph"]
                }),
            })
            .await;
        assert_eq!(response.status, StatusCode::NOT_IMPLEMENTED.as_u16());
        assert_eq!(
            serde_json::from_slice::<Value>(&response.body).unwrap()["code"],
            "KOGWISTAR_GRAPH_AUTHORITY_UNSUPPORTED"
        );

        drop(sqlite);
        remove_test_sqlite(&sqlite_path);
    }

    #[test]
    fn graph_neighbor_request_is_typed_and_closed_to_unknown_fields() {
        let request: GraphNeighborsRequestV1 =
            serde_json::from_str(r#"{"rid":" n1 ","doc_id":"doc-1"}"#).unwrap();
        assert_eq!(request.rid, " n1 ");
        assert_eq!(request.doc_id.as_deref(), Some("doc-1"));
        assert_eq!(request.truth_graph, "knowledge");
        assert!(
            serde_json::from_str::<GraphNeighborsRequestV1>(r#"{"rid":"n1","write":true}"#)
                .is_err()
        );
        assert!(parse_graph_neighbors_request(br#"{"rid":""}"#).is_err());
    }

    #[test]
    fn graph_vector_query_contract_is_batched_bounded_and_closed() {
        let input = parse_graph_vector_query_request(
            br#"{"entity_kind":"node","query_embeddings":[[0.1,0.2],[0.3,0.4]],"metric":"ip","where":{"kind":"note"},"n_results":5,"include":["metadatas","distances"]}"#,
        )
        .unwrap();
        assert_eq!(input.entity_kind, "node");
        assert_eq!(input.query_embeddings.len(), 2);
        assert_eq!(input.metric, DistanceMetric::InnerProduct);
        assert_eq!(input.n_results, 5);
        assert_eq!(input.where_filter.equals.get("kind"), Some(&json!("note")));
        assert_eq!(input.include, ["metadatas", "distances"]);

        assert!(parse_graph_vector_query_request(
            br#"{"entity_kind":"document","query_embeddings":[[0.1]],"metric":"cosine"}"#
        )
        .is_err());
        assert!(parse_graph_vector_query_request(
            br#"{"entity_kind":"node","query_embeddings":[[]],"metric":"cosine"}"#
        )
        .is_err());
        assert!(parse_graph_vector_query_request(
            br#"{"entity_kind":"node","query_embeddings":[[0.1]],"metric":"cosine","where":{"$gt":2}}"#
        )
        .is_err());
        assert!(parse_graph_vector_query_request(
            br#"{"entity_kind":"node","query_embeddings":[[0.1]],"metric":"cosine","unknown":true}"#
        )
        .is_err());
        assert!(parse_graph_vector_query_request(
            br#"{"entity_kind":"node","query_embeddings":[[0.1]],"metric":"cosine","include":["implementation_authority"]}"#
        )
        .is_err());
    }

    #[tokio::test]
    async fn graph_vector_query_route_authorizes_and_validates_before_storage() {
        let service = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "graph_vector_query_contract",
        )
        .unwrap();
        let denied = service
            .execute_async(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/internal/graph/vector-query".to_owned(),
                body: br#"{"entity_kind":"node","query_embeddings":[[0.1]],"metric":"cosine"}"#.to_vec(),
                principal: json!({"sub":"alice","capabilities":[]}),
            })
            .await;
        assert_eq!(denied.status, StatusCode::FORBIDDEN.as_u16());

        let malformed = service
            .execute_async(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/internal/graph/vector-query".to_owned(),
                body: br#"{"entity_kind":"node","query_embeddings":[[0.1]],"metric":"cosine","extra":true}"#.to_vec(),
                principal: json!({"sub":"alice","capabilities":["read_graph"]}),
            })
            .await;
        assert_eq!(malformed.status, StatusCode::UNPROCESSABLE_ENTITY.as_u16());
    }

    #[tokio::test]
    #[ignore = "requires KOGWISTAR_TEST_PG_DSN; run explicitly against a disposable database"]
    async fn postgres_graph_vector_query_is_scoped_acl_filtered_and_rank_stable_when_dsn_available()
    {
        let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
            .expect("ignored test requires KOGWISTAR_TEST_PG_DSN");
        let schema = format!("rust_vector_query_{}", uuid::Uuid::new_v4().simple());
        let service = PostgresRunApplicationService::from_dsn(&dsn, &schema).unwrap();
        service.ensure_schema().await.unwrap();
        service
            .store
            .create_graph_schema(2, kogwistar_store_postgres::GraphTableNames::default())
            .await
            .unwrap();
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/graph-acl-vector-query-v1.json"
        )))
        .unwrap();
        let scope = GraphScope {
            namespace: fixture["scope"]["namespace"].as_str().unwrap().to_owned(),
            workspace_id: fixture["scope"]["workspace_id"]
                .as_str()
                .map(str::to_owned),
            graph_space: None,
        };
        for (entity_kind, table, entities) in [
            ("node", "gke_nodes", &fixture["nodes"]),
            ("edge", "gke_edges", &fixture["edges"]),
        ] {
            for entity in entities.as_array().unwrap() {
                let entity_id = entity["id"].as_str().unwrap();
                let mut metadata = Map::new();
                metadata.insert("entity_type".to_owned(), json!(entity_kind));
                if entity_kind == "node" {
                    metadata.insert("label".to_owned(), json!(entity_id));
                } else {
                    metadata.insert("source_ids".to_owned(), entity["source_ids"].clone());
                    metadata.insert("target_ids".to_owned(), entity["target_ids"].clone());
                }
                service
                    .store
                    .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                        scope: scope.clone(),
                        table: table.to_owned(),
                        record: GraphRecord {
                            id: entity_id.to_owned(),
                            document: Some(entity_id.to_owned()),
                            metadata,
                            embedding: Some(
                                serde_json::from_value(entity["embedding"].clone()).unwrap(),
                            ),
                        },
                        embedding_dim: 2,
                    })
                    .await
                    .unwrap();

                let mut acl_metadata = Map::new();
                for (key, value) in [
                    ("entity_type", json!("acl_record")),
                    ("acl_truth_graph", fixture["truth_graph"].clone()),
                    ("acl_target_entity_id", json!(entity_id)),
                    ("acl_target_grain", json!(entity_kind)),
                    ("acl_version", json!(1)),
                    ("acl_mode", entity["mode"].clone()),
                ] {
                    acl_metadata.insert(key.to_owned(), value);
                }
                if let Some(owner_id) = entity["owner_id"].as_str() {
                    acl_metadata.insert("owner_id".to_owned(), json!(owner_id));
                }
                service
                    .store
                    .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                        scope: scope.clone(),
                        table: "gke_nodes".to_owned(),
                        record: GraphRecord {
                            id: format!("acl-{entity_id}"),
                            document: None,
                            metadata: acl_metadata,
                            embedding: None,
                        },
                        embedding_dim: 2,
                    })
                    .await
                    .unwrap();
            }
        }

        for principal in fixture["principals"].as_array().unwrap() {
            let subject = principal["id"].as_str().unwrap();
            for (entity_kind, expected_key) in [("node", "nodes"), ("edge", "edges")] {
                let request_body = json!({
                    "entity_kind": entity_kind,
                    "query_embeddings": fixture["query_embeddings"],
                    "metric": fixture["metric"],
                    "n_results": fixture["n_results"],
                    "include": ["distances"]
                });
                let response = service
                    .execute_async(ApiEffectRequest {
                        contract_version: 1,
                        method: "POST".to_owned(),
                        path_and_query: "/internal/graph/vector-query".to_owned(),
                        body: serde_json::to_vec(&request_body).unwrap(),
                        principal: json!({
                            "sub": subject,
                            "ns": scope.namespace,
                            "storage_ns": scope.namespace,
                            "workspace": scope.workspace_id,
                            "capabilities": ["read_graph"]
                        }),
                    })
                    .await;
                assert_eq!(response.status, StatusCode::OK.as_u16());
                let result: Value = serde_json::from_slice(&response.body).unwrap();
                assert_eq!(
                    result["ids"],
                    principal["expected"][expected_key],
                    "unexpected {entity_kind} vector results for {subject}"
                );
                let actual_distances = result["distances"].as_array().unwrap();
                let expected_distances = principal["expected"]["distances"][expected_key]
                    .as_array()
                    .unwrap();
                assert_eq!(actual_distances.len(), expected_distances.len());
                for (actual, expected) in actual_distances.iter().zip(expected_distances) {
                    let actual = actual.as_array().unwrap();
                    let expected = expected.as_array().unwrap();
                    assert_eq!(actual.len(), expected.len());
                    for (actual, expected) in actual.iter().zip(expected) {
                        let actual = actual.as_f64().unwrap();
                        let expected = expected.as_f64().unwrap();
                        assert!(
                            (actual - expected).abs() < 1e-5,
                            "unexpected {entity_kind} vector distance for {subject}: {actual} != {expected}"
                        );
                    }
                }
            }
        }

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
            .batch_execute(&format!("DROP SCHEMA \"{schema}\" CASCADE"))
            .await
            .unwrap();
        drop(client);
        connection_task.await.unwrap().unwrap();
    }

    #[test]
    fn graph_traversal_requests_preserve_defaults_and_fail_closed() {
        let shortest: GraphShortestPathRequest =
            serde_json::from_str(r#"{"src_id":"n1","dst_id":"n2"}"#).unwrap();
        assert_eq!(shortest.max_depth, 8);
        assert!(
            serde_json::from_str::<GraphShortestPathRequest>(
                r#"{"src_id":"n1","dst_id":"n2","extra":true}"#
            )
            .is_err()
        );

        let hop: GraphKHopRequest = serde_json::from_str(r#"{"start_ids":["n1"]}"#).unwrap();
        assert_eq!(hop.k, 1);
        assert!(
            serde_json::from_str::<GraphKHopRequest>(r#"{"start_ids":["n1"],"unknown":true}"#)
                .is_err()
        );
    }

    #[tokio::test]
    async fn graph_neighbor_route_authorizes_before_storage_access() {
        let service = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "graph_neighbor_contract",
        )
        .unwrap();
        let denied = service
            .graph_neighbors_effect(&ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/internal/graph/neighbors".to_owned(),
                body: br#"{"rid":"n1"}"#.to_vec(),
                principal: json!({"sub":"alice","capabilities":[]}),
            })
            .await;
        assert_eq!(denied.status, StatusCode::FORBIDDEN.as_u16());

        let malformed = service
            .graph_neighbors_effect(&ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/internal/graph/neighbors".to_owned(),
                body: br#"{"rid":"n1","unknown":true}"#.to_vec(),
                principal: json!({"sub":"alice","capabilities":["read_graph"]}),
            })
            .await;
        assert_eq!(malformed.status, StatusCode::UNPROCESSABLE_ENTITY.as_u16());
    }

    #[tokio::test]
    async fn router_exposes_graph_read_effect_routes() {
        let service = Arc::new(
            PostgresRunApplicationService::from_dsn(
                "postgres://unused:unused@127.0.0.1:1/unused",
                "graph_neighbor_router_contract",
            )
            .unwrap(),
        );
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: Vec::new(),
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig {
                    key: Some("route-test-secret".to_owned()),
                    ..AuthConfig::default()
                },
            },
            service,
        );
        for (path, body) in [
            ("/internal/graph/neighbors", r#"{"rid":"n1"}"#),
            (
                "/internal/graph/vector-query",
                r#"{"entity_kind":"node","query_embeddings":[[0.1]],"metric":"cosine"}"#,
            ),
        ] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri(path)
                        .header("content-type", "application/json")
                        .body(Body::from(body))
                        .unwrap(),
                )
                .await
                .unwrap();

            // A 403 proves the route reached application authorization; 404
            // would mean the transport seam is missing.
            let status = response.status();
            let body = to_bytes(response.into_body(), 4096).await.unwrap();
            assert_eq!(status, StatusCode::FORBIDDEN, "body={:?}", body);
        }
    }

    #[tokio::test]
    async fn internal_effect_route_catalog_is_registered_in_transport() {
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: Vec::new(),
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig::default(),
            },
            Arc::new(UnavailableApplicationService),
        );
        let unique = INTERNAL_APPLICATION_EFFECT_ROUTES
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        assert_eq!(unique.len(), INTERNAL_APPLICATION_EFFECT_ROUTES.len());

        for path in INTERNAL_APPLICATION_EFFECT_ROUTES {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri(*path)
                        .header("content-type", "application/json")
                        .body(Body::from("{}"))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_ne!(
                response.status(),
                StatusCode::NOT_FOUND,
                "internal route is missing from Axum transport: {path}"
            );
            assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
        }
    }

    #[tokio::test]
    async fn router_exposes_document_projection_effect_route() {
        let service = Arc::new(
            PostgresRunApplicationService::from_dsn(
                "postgres://unused:unused@127.0.0.1:1/unused",
                "document_projection_router_contract",
            )
            .unwrap(),
        );
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: Vec::new(),
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig {
                    key: Some("route-test-secret".to_owned()),
                    ..AuthConfig::default()
                },
            },
            service,
        );
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(INTERNAL_DOCUMENT_PROJECTION_ROUTE)
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"doc_id":"doc-1","content":"text","doc_type":"text"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        let status = response.status();
        let body = to_bytes(response.into_body(), 4096).await.unwrap();
        assert_eq!(status, StatusCode::FORBIDDEN, "body={:?}", body);
    }

    #[tokio::test]
    async fn router_exposes_internal_conversation_transcript_route() {
        let service = Arc::new(
            PostgresRunApplicationService::from_dsn(
                "postgres://unused:unused@127.0.0.1:1/unused",
                "conversation_transcript_router_contract",
            )
            .unwrap(),
        );
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: Vec::new(),
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig {
                    key: Some("route-test-secret".to_owned()),
                    ..AuthConfig::default()
                },
            },
            service,
        );
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/conversation/transcript")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"conversation_id":"conversation-1"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
    }

    #[test]
    fn native_runtime_routing_supports_explicit_targets_and_legacy_prefixes() {
        assert_eq!(native_operation_name("rust-native/noop"), Some("noop"));
        assert_eq!(native_operation_name("rust-native/"), None);
        assert_eq!(native_operation_name("noop"), None);
        assert_eq!(
            runtime_worker_recipient_for_operation("rust-native/noop", None, None),
            "rust-native-worker"
        );
        assert_eq!(
            runtime_worker_recipient_for_operation("noop", None, None),
            "python-worker"
        );
        assert_eq!(
            runtime_worker_recipient_for_operation("rust-native/noop", Some("numpy-cpu-2"), None,),
            "rust-native-worker"
        );
        assert_eq!(
            runtime_worker_recipient_for_operation("noop", Some("numpy-cpu-2"), None),
            runtime_worker_recipient(Some("numpy-cpu-2"))
        );
        assert_eq!(
            runtime_worker_recipient_for_operation("increment", None, Some("rust-native")),
            "rust-native-worker"
        );
        assert_eq!(
            runtime_worker_recipient_for_operation("increment", None, Some("python-worker")),
            "python-worker"
        );
        assert!(runtime_graph_plan_from_snapshot(
            r#"{"nodes":[{"id":"start","metadata":{"wf_start":true,"wf_executor_target":"gpu"}}],"edges":[]}"#
        )
        .is_none());
    }

    #[tokio::test]
    async fn sqlite_native_runtime_route_uses_shared_result_reducer() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-native-runtime-{nonce}.sqlite3"));
        let store = SqliteStore::open(&path).unwrap();
        let mut native_operations = kogwistar_runtime::NativeOperationRegistry::default();
        native_operations
            .register("app/custom-result", "v1", |request| {
                Ok(kogwistar_runtime::NativeOperationOutput {
                    state_update: Vec::new(),
                    result: Some(json!({
                        "custom_result": true,
                        "counter": request.state.get("counter"),
                    })),
                })
            })
            .unwrap();
        native_operations
            .register("app/custom-result", "v2", |request| {
                Ok(kogwistar_runtime::NativeOperationOutput {
                    state_update: Vec::new(),
                    result: Some(json!({
                        "custom_result": true,
                        "operation_version": request.op_version,
                        "counter": request.state.get("counter"),
                    })),
                })
            })
            .unwrap();
        let service = SqliteRunApplicationService::open(&path)
            .unwrap()
            .with_native_operations(native_operations);
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: vec!["reader".to_owned()],
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig::default(),
            },
            Arc::new(service),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let client = reqwest::Client::new();
        let base_url = format!("http://{address}");

        for (run_id, operation, version, expected_code) in [
            (
                "unknown-native-run",
                "rust-native/app/missing",
                "v1",
                "KOGWISTAR_NATIVE_OPERATION_UNREGISTERED",
            ),
            (
                "unsupported-native-version-run",
                "rust-native/increment-counter",
                "v9",
                "KOGWISTAR_NATIVE_OPERATION_UNREGISTERED",
            ),
            (
                "malformed-native-run",
                "rust-native/",
                "v1",
                "KOGWISTAR_INVALID_NATIVE_OPERATION",
            ),
        ] {
            let response = client
                .post(format!("{base_url}/api/workflow/runs"))
                .header("x-kogwistar-roles", "reader")
                .json(&json!({
                    "run_id": run_id,
                    "workflow_id": "native-admission-workflow",
                    "conversation_id": "native-admission-conversation",
                    "runtime_kind": "sync",
                    "start_node_id": "entry",
                    "node_ops": {"entry": operation},
                    "node_versions": {"entry": version}
                }))
                .send()
                .await
                .unwrap();
            let status = response.status();
            let response_body = response.text().await.unwrap();
            assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{response_body}");
            let body: Value = serde_json::from_str(&response_body).unwrap();
            assert_eq!(body["code"], expected_code);
            assert!(store.get_server_run(run_id).unwrap().is_none());
            assert!(
                store
                    .list_projected_lane_messages(LaneMessageFilter {
                        namespace: Some("workflow".to_owned()),
                        inbox_id: Some("workflow-runtime".to_owned()),
                        correlation_id: Some(run_id.to_owned()),
                        limit: 10,
                        ..LaneMessageFilter::default()
                    })
                    .unwrap()
                    .is_empty()
            );
        }

        let submit = client
            .post(format!("{base_url}/api/workflow/runs"))
            .header("x-kogwistar-roles", "reader")
            .header("content-type", "application/json")
            .body(
                r#"{"run_id":"native-run","workflow_id":"native-workflow","conversation_id":"native-conversation","initial_state":{"counter":1},"runtime_kind":"sync","start_node_id":"entry","node_ops":{"entry":"rust-native/increment-counter","finish":"rust-native/emit-result","custom":"rust-native/app/custom-result"},"node_versions":{"custom":"v2"},"runtime_routes":[{"source_node_id":"entry","target_node_id":"finish","join_mask":0},{"source_node_id":"finish","target_node_id":"custom","join_mask":0}]}"#,
            )
            .send()
            .await
            .unwrap();
        assert_eq!(submit.status(), StatusCode::ACCEPTED);

        let claim = client
            .post(format!("{base_url}/internal/runtime/claim"))
            .header("x-kogwistar-roles", "reader")
            .header("content-type", "application/json")
            .body(
                r#"{"claimed_by":"rust-native-worker","limit":1,"lease_seconds":60,"recipient_ids":["rust-native-worker"]}"#,
            )
            .send()
            .await
            .unwrap();
        assert_eq!(claim.status(), StatusCode::OK);
        let work = claim.json::<Value>().await.unwrap()["work"][0].clone();
        assert_eq!(work["payload"]["op"], "rust-native/increment-counter");
        assert_eq!(
            work["payload"]["executor_target"],
            kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET
        );
        assert!(work["claimed_by"]
            .as_str()
            .unwrap()
            .starts_with("rust-native-worker#"));

        let mut malformed_version_payload = work["payload"].clone();
        malformed_version_payload["op_version"] = json!(2);
        let malformed_version = client
            .post(format!("{base_url}/internal/runtime/native/execute"))
            .header("x-kogwistar-roles", "reader")
            .json(&json!({
                "handoff": {
                    "message_id": work["message_id"],
                    "claimed_by": work["claimed_by"],
                    "run_id": work["run_id"],
                    "step_id": work["step_id"],
                    "correlation_id": work["correlation_id"]
                },
                "payload": malformed_version_payload
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(malformed_version.status(), StatusCode::CONFLICT);

        let execute_payload = json!({
            "handoff": {
                "message_id": work["message_id"],
                "claimed_by": work["claimed_by"],
                "run_id": work["run_id"],
                "step_id": work["step_id"],
                "correlation_id": work["correlation_id"]
            },
            "payload": work["payload"]
        });
        let execute = client
            .post(format!("{base_url}/internal/runtime/native/execute"))
            .header("x-kogwistar-roles", "reader")
            .json(&execute_payload)
            .send()
            .await
            .unwrap();
        let execute_status = execute.status();
        let execute_body = execute.text().await.unwrap();
        assert_eq!(execute_status, StatusCode::OK, "{}", execute_body);
        assert_eq!(
            store.get_server_run("native-run").unwrap().unwrap().status,
            "running"
        );
        let event_count_after_first_execute = store
            .list_server_run_events("native-run", 0, 100)
            .unwrap()
            .len();
        let retried_execute = client
            .post(format!("{base_url}/internal/runtime/native/execute"))
            .header("x-kogwistar-roles", "reader")
            .json(&execute_payload)
            .send()
            .await
            .unwrap();
        assert_eq!(retried_execute.status(), StatusCode::OK);
        assert_eq!(
            store
                .list_server_run_events("native-run", 0, 100)
                .unwrap()
                .len(),
            event_count_after_first_execute,
            "retry must not duplicate durable runtime events"
        );
        let continuation_claim = client
            .post(format!("{base_url}/internal/runtime/claim"))
            .header("x-kogwistar-roles", "reader")
            .header("content-type", "application/json")
            .body(
                r#"{"claimed_by":"rust-native-worker","limit":1,"lease_seconds":60,"recipient_ids":["rust-native-worker"]}"#,
            )
            .send()
            .await
            .unwrap();
        assert_eq!(continuation_claim.status(), StatusCode::OK);
        let continuation = continuation_claim.json::<Value>().await.unwrap()["work"][0].clone();
        assert_eq!(continuation["payload"]["op"], "rust-native/emit-result");
        assert_eq!(
            continuation["payload"]["executor_target"],
            kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET
        );
        let continuation_execute = client
            .post(format!("{base_url}/internal/runtime/native/execute"))
            .header("x-kogwistar-roles", "reader")
            .json(&json!({
                "handoff": {
                    "message_id": continuation["message_id"],
                    "claimed_by": continuation["claimed_by"],
                    "run_id": continuation["run_id"],
                    "step_id": continuation["step_id"],
                    "correlation_id": continuation["correlation_id"]
                },
                "payload": continuation["payload"]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(continuation_execute.status(), StatusCode::OK);
        assert_eq!(
            store.get_server_run("native-run").unwrap().unwrap().status,
            "running"
        );
        let events = store.list_server_run_events("native-run", 0, 100).unwrap();
        let native_trace = events
            .iter()
            .find(|event| {
                event.event_type == "workflow.trace.v1"
                    && event.payload_json.contains("runtime.native.step.completed")
            })
            .expect("native trace event must survive the HTTP/API durable lane");
        let native_trace_payload: Value = serde_json::from_str(&native_trace.payload_json).unwrap();
        assert!(native_seam_attestation_is_valid(
            &native_trace_payload["trace"]
        ));
        assert_eq!(
            native_trace_payload["trace"]["implementation_authority"],
            kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET
        );
        assert_eq!(
            native_trace_payload["trace"]["seam_attestation_version"],
            kogwistar_runtime::RUST_NATIVE_SEAM_ATTESTATION_VERSION
        );
        assert_eq!(native_trace_payload["trace"]["seam_crossings"], json!([]));
        assert!(events.iter().any(|event| {
            event.event_type == "workflow.recorded_transition.v1"
                && event.payload_json.contains("increment-counter")
                && event.payload_json.contains("counter")
                && event.payload_json.contains(":2")
        }));
        assert!(events.iter().any(|event| {
            event.event_type == "workflow.recorded_transition.v1"
                && event.payload_json.contains("emit-result")
                && event.payload_json.contains("rust-native")
        }));

        let custom_claim = client
            .post(format!("{base_url}/internal/runtime/claim"))
            .header("x-kogwistar-roles", "reader")
            .json(&json!({
                "claimed_by": "rust-native-worker",
                "limit": 1,
                "lease_seconds": 60,
                "recipient_ids": ["rust-native-worker"]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(custom_claim.status(), StatusCode::OK);
        let custom_work = custom_claim.json::<Value>().await.unwrap()["work"][0].clone();
        assert_eq!(
            custom_work["payload"]["op"],
            "rust-native/app/custom-result"
        );
        let custom_execute = client
            .post(format!("{base_url}/internal/runtime/native/execute"))
            .header("x-kogwistar-roles", "reader")
            .json(&json!({
                "handoff": {
                    "message_id": custom_work["message_id"],
                    "claimed_by": custom_work["claimed_by"],
                    "run_id": custom_work["run_id"],
                    "step_id": custom_work["step_id"],
                    "correlation_id": custom_work["correlation_id"]
                },
                "payload": custom_work["payload"]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(custom_execute.status(), StatusCode::OK);
        assert_eq!(
            store.get_server_run("native-run").unwrap().unwrap().status,
            "succeeded"
        );
        let events = store.list_server_run_events("native-run", 0, 100).unwrap();
        assert!(events.iter().any(|event| {
            event.event_type == "workflow.recorded_transition.v1"
                && event.payload_json.contains("app/custom-result")
                && event.payload_json.contains("custom_result")
                && event.payload_json.contains("v2")
        }));
        assert!(events.iter().any(|event| {
            event.event_type == "workflow.trace.v1"
                && event.payload_json.contains("app/custom-result")
                && event.payload_json.contains("seam_crossings")
        }));

        // One durable run can hand off from the Python worker protocol to
        // Rust-native execution and back without changing its state contract.
        let mixed_submit = client
            .post(format!("{base_url}/api/workflow/runs"))
            .header("x-kogwistar-roles", "reader")
            .json(&json!({
                "run_id": "mixed-language-run",
                "workflow_id": "mixed-language-workflow",
                "conversation_id": "mixed-language-conversation",
                "initial_state": {"counter": 4},
                "runtime_kind": "sync",
                "start_node_id": "python-before",
                "node_ops": {
                    "python-before": "application.before",
                    "native-middle": "rust-native/increment-counter",
                    "python-after": "application.after"
                },
                "runtime_routes": [
                    {"source_node_id": "python-before", "target_node_id": "native-middle", "join_mask": 0},
                    {"source_node_id": "native-middle", "target_node_id": "python-after", "join_mask": 0}
                ]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(mixed_submit.status(), StatusCode::ACCEPTED);

        let python_before_claim = client
            .post(format!("{base_url}/internal/runtime/claim"))
            .header("x-kogwistar-roles", "reader")
            .json(&json!({
                "claimed_by": "python-worker",
                "limit": 1,
                "lease_seconds": 60,
                "recipient_ids": ["python-worker"]
            }))
            .send()
            .await
            .unwrap();
        let python_before_work =
            python_before_claim.json::<Value>().await.unwrap()["work"][0].clone();
        assert_eq!(python_before_work["step_id"], "python-before");
        assert!(python_before_work["payload"]["executor_target"].is_null());
        let python_before_result = client
            .post(format!("{base_url}/internal/runtime/results"))
            .header("x-kogwistar-roles", "reader")
            .json(&json!({
                "handoff": {
                    "message_id": python_before_work["message_id"],
                    "claimed_by": python_before_work["claimed_by"],
                    "run_id": python_before_work["run_id"],
                    "step_id": python_before_work["step_id"],
                    "correlation_id": python_before_work["correlation_id"]
                },
                "effect": {
                    "contract_version": 1,
                    "effect_id": "mixed-python-before",
                    "state_update": [["u", {"python_before_seen": true}]],
                    "successors": [],
                    "result": null
                }
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(python_before_result.status(), StatusCode::OK);

        let native_middle_claim = client
            .post(format!("{base_url}/internal/runtime/claim"))
            .header("x-kogwistar-roles", "reader")
            .json(&json!({
                "claimed_by": "rust-native-worker",
                "limit": 1,
                "lease_seconds": 60,
                "recipient_ids": ["rust-native-worker"]
            }))
            .send()
            .await
            .unwrap();
        let native_middle_work =
            native_middle_claim.json::<Value>().await.unwrap()["work"][0].clone();
        assert_eq!(native_middle_work["payload"]["node_id"], "native-middle");
        assert_eq!(
            native_middle_work["payload"]["executor_target"],
            "rust-native"
        );
        assert_eq!(native_middle_work["payload"]["state"]["counter"], 4);
        assert_eq!(
            native_middle_work["payload"]["state"]["python_before_seen"],
            true
        );
        let events_before_wrong_target = store
            .list_server_run_events("mixed-language-run", 0, 100)
            .unwrap();
        let mut wrong_target_payload = native_middle_work["payload"].clone();
        wrong_target_payload["executor_target"] = json!("python-worker");
        let wrong_target_result = client
            .post(format!("{base_url}/internal/runtime/native/execute"))
            .header("x-kogwistar-roles", "reader")
            .json(&json!({
                "handoff": {
                    "message_id": native_middle_work["message_id"],
                    "claimed_by": native_middle_work["claimed_by"],
                    "run_id": native_middle_work["run_id"],
                    "step_id": native_middle_work["step_id"],
                    "correlation_id": native_middle_work["correlation_id"]
                },
                "payload": wrong_target_payload
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(wrong_target_result.status(), StatusCode::CONFLICT);
        assert_eq!(
            store
                .list_server_run_events("mixed-language-run", 0, 100)
                .unwrap(),
            events_before_wrong_target,
            "wrong-target rejection must not persist a worker result or trace"
        );
        let native_middle_result = client
            .post(format!("{base_url}/internal/runtime/native/execute"))
            .header("x-kogwistar-roles", "reader")
            .json(&json!({
                "handoff": {
                    "message_id": native_middle_work["message_id"],
                    "claimed_by": native_middle_work["claimed_by"],
                    "run_id": native_middle_work["run_id"],
                    "step_id": native_middle_work["step_id"],
                    "correlation_id": native_middle_work["correlation_id"]
                },
                "payload": native_middle_work["payload"]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(native_middle_result.status(), StatusCode::OK);

        let python_after_claim = client
            .post(format!("{base_url}/internal/runtime/claim"))
            .header("x-kogwistar-roles", "reader")
            .json(&json!({
                "claimed_by": "python-worker",
                "limit": 1,
                "lease_seconds": 60,
                "recipient_ids": ["python-worker"]
            }))
            .send()
            .await
            .unwrap();
        let python_after_work =
            python_after_claim.json::<Value>().await.unwrap()["work"][0].clone();
        assert_eq!(python_after_work["payload"]["node_id"], "python-after");
        assert!(python_after_work["payload"]["executor_target"].is_null());
        assert_eq!(python_after_work["payload"]["state"]["counter"], 5);
        assert_eq!(
            python_after_work["payload"]["state"]["python_before_seen"],
            true
        );
        let python_after_result = client
            .post(format!("{base_url}/internal/runtime/results"))
            .header("x-kogwistar-roles", "reader")
            .json(&json!({
                "handoff": {
                    "message_id": python_after_work["message_id"],
                    "claimed_by": python_after_work["claimed_by"],
                    "run_id": python_after_work["run_id"],
                    "step_id": python_after_work["step_id"],
                    "correlation_id": python_after_work["correlation_id"]
                },
                "effect": {
                    "contract_version": 1,
                    "effect_id": "mixed-python-after",
                    "state_update": [["u", {"python_after_seen": true}]],
                    "successors": [],
                    "result": {"workflow_status": "succeeded", "answer": "mixed"}
                }
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(python_after_result.status(), StatusCode::OK);
        let mixed_run = store.get_server_run("mixed-language-run").unwrap().unwrap();
        assert_eq!(mixed_run.status, "succeeded");
        let mixed_events = store
            .list_server_run_events("mixed-language-run", 0, 100)
            .unwrap();
        assert!(mixed_events.iter().any(|event| {
            event.event_type == "workflow.trace.v1"
                && event.payload_json.contains("native-middle")
                && event.payload_json.contains("seam_crossings")
        }));
        let native_trace = mixed_events
            .iter()
            .filter(|event| event.event_type == "workflow.trace.v1")
            .map(|event| serde_json::from_str::<Value>(&event.payload_json).unwrap())
            .find(|event| event["trace"]["event"] == "runtime.native.step.completed")
            .expect("successful Rust-native step must persist its trace");
        assert_eq!(native_trace["trace"]["executor_target"], "rust-native");

        // A non-native operation must keep the legacy Python-worker lane and
        // must not acquire native executor metadata implicitly.
        let legacy_submit = client
            .post(format!("{base_url}/api/workflow/runs"))
            .header("x-kogwistar-roles", "reader")
            .header("content-type", "application/json")
            .json(&json!({
                "run_id": "legacy-run",
                "workflow_id": "legacy-workflow",
                "conversation_id": "legacy-conversation",
                "initial_state": {"seed": 1},
                "runtime_kind": "sync",
                "start_node_id": "entry",
                "node_ops": {"entry": "application.process"}
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(legacy_submit.status(), StatusCode::ACCEPTED);
        let legacy_claim = client
            .post(format!("{base_url}/internal/runtime/claim"))
            .header("x-kogwistar-roles", "reader")
            .json(&json!({
                "claimed_by": "python-worker",
                "limit": 1,
                "lease_seconds": 60,
                "recipient_ids": ["python-worker"]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(legacy_claim.status(), StatusCode::OK);
        let legacy_work = legacy_claim.json::<Value>().await.unwrap()["work"][0].clone();
        assert_eq!(legacy_work["payload"]["op"], "application.process");
        assert!(legacy_work["payload"].get("executor_target").is_none());

        server.abort();
        let _ = server.await;
        drop(store);
        std::fs::remove_file(path).unwrap();
    }
