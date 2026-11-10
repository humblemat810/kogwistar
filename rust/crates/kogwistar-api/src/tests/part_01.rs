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
    fn wisdom_storage_schema_is_configurable_with_legacy_default() {
        assert_eq!(
            configured_wisdom_schema("kg", None).unwrap(),
            "kg_wisdom"
        );
        assert_eq!(
            configured_wisdom_schema("kg", Some("kg")).unwrap(),
            "kg",
            "the generic graph store can be shared by graph roles"
        );
        assert_eq!(
            configured_wisdom_schema("kg", Some("tenant_wisdom")).unwrap(),
            "tenant_wisdom"
        );
        assert!(configured_wisdom_schema("kg", Some("  ")).is_err());
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

    #[test]
    fn configured_search_index_fts_sidecar_reopens() {
        let nonce = uuid::Uuid::new_v4();
        let graph_path = std::env::temp_dir().join(format!(
            "kogwistar-search-index-graph-{nonce}.sqlite3"
        ));
        let index_path = std::env::temp_dir().join(format!(
            "kogwistar-search-index-sidecar-{nonce}.sqlite3"
        ));
        let service = SqliteRunApplicationService::open(&graph_path)
            .expect("open SQLite graph service")
            .with_search_index_path(&index_path)
            .expect("configure persistent FTS sidecar");
        assert!(index_path.exists());
        drop(service);

        let reopened = crate::search_index_fts::open_index(&index_path)
            .expect("reopen configured FTS sidecar");
        let table_count: i64 = reopened
            .query_row(
                "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='semantic_index_fts'",
                [],
                |row| row.get(0),
            )
            .expect("query FTS schema");
        assert_eq!(table_count, 1);
        drop(reopened);
        remove_test_sqlite(&index_path);
        remove_test_sqlite(&graph_path);
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

        let resource_snapshot_denied = service
            .execute(make_request("resource_snapshot", json!({})))
            .await;
        assert_eq!(
            resource_snapshot_denied.status,
            StatusCode::FORBIDDEN.as_u16()
        );
        assert_eq!(
            serde_json::from_slice::<Value>(&resource_snapshot_denied.body).unwrap()["code"],
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
        let invoke_contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/invoke-tool-results-v1.json"
        ))
        .unwrap();
        assert_eq!(invoke_contract["version"], "v1");
        assert_eq!(invoke_contract["op"], "invoke_tool");
        let result_fields = payload["result"]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>();
        let expected_result_fields = invoke_contract["results"]["capability_snapshot"]["root_fields"]
            .as_array()
            .unwrap()
            .iter()
            .map(|field| field.as_str().unwrap().to_owned())
            .collect::<BTreeSet<_>>();
        assert_eq!(result_fields, expected_result_fields);
        let snapshot_fields = payload["result"]["snapshot"]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>();
        let expected_snapshot_fields = invoke_contract["results"]["capability_snapshot"]["snapshot_fields"]
            .as_array()
            .unwrap()
            .iter()
            .map(|field| field.as_str().unwrap().to_owned())
            .collect::<BTreeSet<_>>();
        assert_eq!(snapshot_fields, expected_snapshot_fields);
        assert_eq!(payload["op"], "invoke_tool");
        assert_eq!(payload["result"]["tool_name"], "capability_snapshot");
        assert_eq!(payload["result"]["snapshot"]["current_subject"], "alice");
        assert!(payload["result"]["snapshot"]["specs"].as_array().is_some());
        assert!(payload["result"]["snapshot"]["effective_capabilities"]
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
        use sha2::Digest;

        let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
            .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
        let schema = format!("rust_snapshot_route_{}", uuid::Uuid::new_v4().simple());
        let service = PostgresRunApplicationService::from_dsn(&dsn, &schema).unwrap();
        service.ensure_graph_type_schemas().await.unwrap();
        service
            .conversation_store
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
            .conversation_store
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
            .conversation_store
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
            .conversation_store
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
            .conversation_store
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
        let initial_invoke_digest = format!(
            "{:x}",
            sha2::Sha256::digest(
                kogwistar_contracts::canonical_json(&invoke_snapshot["result"], false).as_bytes(),
            )
        );

        drop(service);
        let reopened = PostgresRunApplicationService::from_dsn(&dsn, &schema).unwrap();
        reopened.ensure_graph_type_schemas().await.unwrap();
        reopened
            .conversation_store
            .create_graph_schema(3, kogwistar_store_postgres::GraphTableNames::default())
            .await
            .unwrap();
        let retried_invoke_snapshot = reopened
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
        assert_eq!(
            retried_invoke_snapshot.status,
            StatusCode::OK.as_u16()
        );
        let retried_invoke_snapshot: Value =
            serde_json::from_slice(&retried_invoke_snapshot.body).unwrap();
        assert_eq!(retried_invoke_snapshot["op"], "invoke_tool");
        assert_eq!(
            format!(
                "{:x}",
                sha2::Sha256::digest(
                    kogwistar_contracts::canonical_json(
                        &retried_invoke_snapshot["result"],
                        false,
                    )
                    .as_bytes(),
                )
            ),
            initial_invoke_digest,
            "PostgreSQL syscall retry returns the same result after service reopen"
        );

        drop(reopened);
        let (cleanup, connection) = tokio_postgres::connect(&dsn, tokio_postgres::NoTls)
            .await
            .unwrap();
        tokio::spawn(async move {
            let _ = connection.await;
        });
        cleanup
            .batch_execute(&format!(
                "DROP SCHEMA IF EXISTS \"{schema}_conversation\" CASCADE; \
                 DROP SCHEMA IF EXISTS \"{schema}_workflow\" CASCADE; \
                 DROP SCHEMA IF EXISTS \"{schema}_wisdom\" CASCADE; \
                 DROP SCHEMA IF EXISTS \"{schema}\" CASCADE"
            ))
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
        service.ensure_graph_type_schemas().await.unwrap();
        service
            .conversation_store
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
                json!({"turn_index": 1, "role": "user", "summary": "first"}),
            ),
            (
                "turn-2",
                "conversation-1",
                "conversation_turn",
                json!({"turn_index": 2, "role": "assistant", "summary": "second"}),
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
                .conversation_store
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
                .conversation_store
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

        let mut private_turn_metadata = Map::new();
        for (key, value) in [
            ("entity_type", json!("conversation_turn")),
            ("conversation_id", json!("conversation-1")),
            ("turn_index", json!(3)),
            ("role", json!("assistant")),
            ("summary", json!("private")),
        ] {
            private_turn_metadata.insert(key.to_owned(), value);
        }
        service
            .conversation_store
            .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                record: GraphRecord {
                    id: "turn-private".to_owned(),
                    document: None,
                    metadata: private_turn_metadata,
                    embedding: None,
                },
                embedding_dim: 3,
            })
            .await
            .unwrap();
        let private_acl = Map::from_iter([
            ("entity_type".to_owned(), json!("acl_record")),
            ("acl_truth_graph".to_owned(), json!("conversation")),
            ("acl_target_entity_id".to_owned(), json!("turn-private")),
            ("acl_target_grain".to_owned(), json!("node")),
            ("acl_version".to_owned(), json!(1)),
            ("acl_mode".to_owned(), json!("private")),
            ("owner_id".to_owned(), json!("bob")),
        ]);
        service
            .conversation_store
            .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                record: GraphRecord {
                    id: "acl-turn-private".to_owned(),
                    document: None,
                    metadata: private_acl,
                    embedding: None,
                },
                embedding_dim: 3,
            })
            .await
            .unwrap();

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
                .conversation_store
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
                .conversation_store
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
        assert_eq!(payload["tail_node_id"], "turn-2");
        assert_eq!(payload["turn_count"], 2);
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
                    "turn_count": 2
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
        assert_eq!(transcript["turns"].as_array().unwrap().len(), 2);
        assert_eq!(transcript["turns"][0]["node_id"], "turn-1");
        assert_eq!(transcript["turns"][0]["turn_index"], 1);
        assert_eq!(transcript["turns"][0]["role"], "user");
        assert_eq!(transcript["turns"][0]["content"], "first");
        assert_eq!(transcript["turns"][0]["entity_type"], "conversation_turn");
        assert_eq!(transcript["turns"][1]["node_id"], "turn-2");
        assert_eq!(transcript["turns"][1]["turn_index"], 2);
        assert_eq!(transcript["turns"][1]["role"], "assistant");
        assert_eq!(transcript["turns"][1]["content"], "second");
        assert_eq!(transcript["turns"][1]["entity_type"], "conversation_turn");
        assert!(transcript.get("schema_version").is_none());
        assert!(transcript.get("implementation_authority").is_none());
        assert!(transcript["turns"].as_array().unwrap().iter().all(|turn| {
            turn.get("schema_version").is_none()
        }));
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

        let receive_request = |principal: Value| ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/syscall/v1/receive_message".to_owned(),
                body: serde_json::to_vec(&json!({
                    "version": "v1",
                    "op": "receive_message",
                    "args": {"conversation_id": "conversation-1", "limit": 0}
                }))
                .unwrap(),
                principal
            };
        let missing_id_response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/syscall/v1/receive_message".to_owned(),
                body: serde_json::to_vec(&json!({
                    "version": "v1",
                    "op": "receive_message",
                    "args": {}
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
        let receive_error_contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/syscall-receive-message-missing-id-v1.json"
        ))
        .unwrap();
        assert_eq!(missing_id_response.status, receive_error_contract["http_status"]);
        assert_eq!(
            serde_json::from_slice::<Value>(&missing_id_response.body).unwrap(),
            receive_error_contract["body"]
        );
        let denied_receive = service
            .execute(receive_request(json!({
                "sub": "alice",
                "role": "ro",
                "ns": "conversation",
                "storage_ns": "conversation",
                "capabilities": []
            })))
            .await;
        assert_eq!(denied_receive.status, StatusCode::FORBIDDEN.as_u16());
        let wrong_namespace_receive = service
            .execute(receive_request(json!({
                "sub": "alice",
                "role": "ro",
                "ns": "workflow",
                "storage_ns": "conversation",
                "capabilities": ["read_graph"]
            })))
            .await;
        assert_eq!(
            wrong_namespace_receive.status,
            StatusCode::FORBIDDEN.as_u16()
        );
        let receive_response = service
            .execute(receive_request(json!({
                    "sub": "alice",
                    "role": "ro",
                    "ns": "conversation",
                    "storage_ns": "conversation",
                    "capabilities": ["read_graph"]
                })))
            .await;
        assert_eq!(receive_response.status, StatusCode::OK.as_u16());
        let receive_payload: Value = serde_json::from_slice(&receive_response.body).unwrap();
        let mut expected_receive = serde_json::from_str::<Value>(include_str!(
            "../../../../../contracts/golden/receive-message-result-v1.json"
        ))
        .unwrap();
        // The fixture's conversation ID is illustrative; the response echoes this request ID.
        expected_receive["conversation_id"] = json!("conversation-1");
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
            Arc::new(service.clone()),
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

        for (id, turn_index, workspace_id, graph_space, summary) in [
            ("scoped-turn-1", 1, "workspace-a", "conversation", "first scoped"),
            ("scoped-turn-2", 2, "workspace-a", "conversation", "second scoped"),
            (
                "foreign-workspace-turn",
                3,
                "workspace-b",
                "conversation",
                "foreign workspace",
            ),
            (
                "foreign-graph-space-turn",
                4,
                "workspace-a",
                "knowledge",
                "foreign graph space",
            ),
        ] {
            let scoped_record = GraphScope {
                namespace: "conversation".to_owned(),
                workspace_id: Some(workspace_id.to_owned()),
                graph_space: Some(graph_space.to_owned()),
            };
            for (record_id, metadata) in [
                (
                    id.to_owned(),
                    Map::from_iter([
                        ("entity_type".to_owned(), json!("conversation_turn")),
                        ("conversation_id".to_owned(), json!("scoped-conversation")),
                        ("turn_index".to_owned(), json!(turn_index)),
                        ("role".to_owned(), json!("assistant")),
                        ("summary".to_owned(), json!(summary)),
                    ]),
                ),
                (
                    format!("acl-{id}"),
                    Map::from_iter([
                        ("entity_type".to_owned(), json!("acl_record")),
                        ("acl_truth_graph".to_owned(), json!("conversation")),
                        ("acl_target_entity_id".to_owned(), json!(id)),
                        ("acl_target_grain".to_owned(), json!("node")),
                        ("acl_version".to_owned(), json!(1)),
                        ("acl_mode".to_owned(), json!("public")),
                    ]),
                ),
            ] {
                service
                    .conversation_store
                    .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                        scope: scoped_record.clone(),
                        table: "gke_nodes".to_owned(),
                        record: GraphRecord {
                            id: record_id,
                            document: None,
                            metadata,
                            embedding: None,
                        },
                        embedding_dim: 3,
                    })
                    .await
                    .unwrap();
            }
        }
        let scoped_receive = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/syscall/v1/receive_message".to_owned(),
                body: serde_json::to_vec(&json!({
                    "version": "v1",
                    "op": "receive_message",
                    "args": {"conversation_id": "scoped-conversation", "limit": 0}
                }))
                .unwrap(),
                principal: json!({
                    "sub": "alice",
                    "role": "ro",
                    "ns": "conversation",
                    "storage_ns": "conversation",
                    "workspace_id": "workspace-a",
                    "graph_space": "conversation",
                    "capabilities": ["read_graph"]
                }),
            })
            .await;
        assert_eq!(scoped_receive.status, StatusCode::OK.as_u16());
        let scoped_receive: Value = serde_json::from_slice(&scoped_receive.body).unwrap();
        assert_eq!(scoped_receive["result"]["messages"].as_array().unwrap().len(), 2);
        assert_eq!(scoped_receive["result"]["messages"][0]["node_id"], "scoped-turn-1");
        assert_eq!(scoped_receive["result"]["messages"][1]["node_id"], "scoped-turn-2");

        let (cleanup, connection) = tokio_postgres::connect(&dsn, tokio_postgres::NoTls)
            .await
            .unwrap();
        tokio::spawn(async move {
            let _ = connection.await;
        });
        cleanup
            .batch_execute(&format!(
                "DROP SCHEMA IF EXISTS \"{schema}_conversation\" CASCADE; \
                 DROP SCHEMA IF EXISTS \"{schema}_workflow\" CASCADE; \
                 DROP SCHEMA IF EXISTS \"{schema}_wisdom\" CASCADE; \
                 DROP SCHEMA IF EXISTS \"{schema}\" CASCADE"
            ))
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn conversation_create_rejects_non_owner_before_storage() {
        let service = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "conversation_create_owner_auth_contract",
        )
        .unwrap();
        let response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/syscall/v1/create_conversation".to_owned(),
                body: serde_json::to_vec(&json!({
                    "version": "v1",
                    "op": "create_conversation",
                    "args": {"user_id": "bob", "conversation_id": "conv-1"}
                }))
                .unwrap(),
                principal: json!({
                    "sub": "alice",
                    "role": "rw",
                    "ns": "conversation",
                    "storage_ns": "conversation",
                    "capabilities": ["spawn_process"]
                }),
            })
            .await;
        assert_eq!(response.status, StatusCode::FORBIDDEN.as_u16());
        assert_eq!(
            serde_json::from_slice::<Value>(&response.body).unwrap()["code"],
            "KOGWISTAR_RUN_OWNER_FORBIDDEN"
        );
    }

    #[tokio::test]
    async fn conversation_create_embedding_failure_fails_before_storage() {
        use std::sync::{Arc, Mutex, atomic::AtomicBool};

        let service = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "conversation_create_embedding_failure_contract",
        )
        .unwrap()
        .with_embedding_provider(Arc::new(FixedEmbeddingProvider {
            profile: kogwistar_engine::embedding::EmbeddingProfile {
                provider: "test".to_owned(),
                model: "failing-conversation-start-v1".to_owned(),
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
            },
            vector: vec![0.6, 0.8, 0.0],
            calls: Arc::new(Mutex::new(Vec::new())),
            fail: Some(Arc::new(AtomicBool::new(true))),
        }));
        let response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/syscall/v1/create_conversation".to_owned(),
                body: serde_json::to_vec(&json!({
                    "version": "v1",
                    "op": "create_conversation",
                    "args": {"user_id": "alice", "conversation_id": "conv-1"}
                }))
                .unwrap(),
                principal: json!({
                    "sub": "alice",
                    "role": "rw",
                    "ns": "conversation",
                    "storage_ns": "conversation",
                    "capabilities": ["spawn_process"]
                }),
            })
            .await;
        assert_eq!(response.status, StatusCode::BAD_GATEWAY.as_u16());
        assert_eq!(
            serde_json::from_slice::<Value>(&response.body).unwrap()["code"],
            "KOGWISTAR_EMBEDDING_FAILED"
        );
    }

    #[tokio::test]
    #[ignore = "requires KOGWISTAR_TEST_PG_DSN; run explicitly with --include-ignored"]
    async fn postgres_conversation_create_persists_private_start_node_when_dsn_available() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use kogwistar_store::EventReadStore;
        use std::sync::{Arc, Mutex};
        use tower::ServiceExt;

        let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
            .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
        let schema = format!("rust_conversation_create_{}", uuid::Uuid::new_v4().simple());
        let embedding_calls = Arc::new(Mutex::new(Vec::new()));
        let service = PostgresRunApplicationService::from_dsn(&dsn, &schema)
            .unwrap()
            .with_embedding_provider(Arc::new(FixedEmbeddingProvider {
                profile: kogwistar_engine::embedding::EmbeddingProfile {
                    provider: "test".to_owned(),
                    model: "fixed-conversation-start-v1".to_owned(),
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
                },
                vector: vec![0.6, 0.8, 0.0],
                calls: Arc::clone(&embedding_calls),
                fail: None,
            }));
        service.ensure_graph_type_schemas().await.unwrap();
        service
            .conversation_store
            .create_graph_schema(3, kogwistar_store_postgres::GraphTableNames::default())
            .await
            .unwrap();

        let create = |user_id: &str| ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/syscall/v1/create_conversation".to_owned(),
            body: serde_json::to_vec(&json!({
                "version": "v1",
                "op": "create_conversation",
                "args": {
                    "user_id": user_id,
                    "conversation_id": "conv-create-test",
                    "start_node_id": "conv-create-start"
                }
            }))
            .unwrap(),
            principal: json!({
                "sub": "alice",
                "role": "rw",
                "ns": "conversation",
                "storage_ns": "conversation",
                "capabilities": ["spawn_process"]
            }),
        };
        let created = service.execute(create("alice")).await;
        assert_eq!(
            created.status,
            StatusCode::OK.as_u16(),
            "conversation creation failed: {}",
            String::from_utf8_lossy(&created.body)
        );
        let created: Value = serde_json::from_slice(&created.body).unwrap();
        assert_eq!(created["status"], "ok");
        let syscall_result_fixture: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/conversation-create-syscall-result-v1.json"
        ))
        .unwrap();
        assert_eq!(created["result"], syscall_result_fixture);
        let start_projection = service
            .conversation_store
            .graph_projection_records(GraphProjectionRead {
                scope: GraphScope {
                    namespace: "conversation".to_owned(),
                    workspace_id: None,
                    graph_space: None,
                },
                table: "gke_nodes".to_owned(),
                ids: Some(vec!["conv-create-start".to_owned()]),
                metadata: MetadataFilter::default(),
                limit: 1,
            })
            .await
            .unwrap();
        assert_eq!(start_projection[0].embedding, Some(vec![0.6, 0.8, 0.0]));
        assert_eq!(embedding_calls.lock().unwrap()[0].len(), 1);
        let events = service
            .conversation_store
            .replay_events("conversation", 0, 100)
            .await
            .unwrap();
        let start_event = events
            .iter()
            .find(|event| event.entity_id == "conv-create-start")
            .expect("conversation start event is durable");
        assert!(start_event.payload.get("embedding").is_none());
        let conversation_create_fixture: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/conversation-create-start-node-v1.json"
        ))
        .unwrap();
        assert_eq!(start_event.payload, conversation_create_fixture["payload"]);

        let details = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: "/api/conversations/conv-create-test".to_owned(),
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
        assert_eq!(details.status, StatusCode::OK.as_u16());
        let details: Value = serde_json::from_slice(&details.body).unwrap();
        assert_eq!(details["conversation_id"], "conv-create-test");
        assert_eq!(details["user_id"], "alice");
        assert_eq!(details["status"], "active");
        assert_eq!(details["start_node_id"], "conv-create-start");
        assert_eq!(details["tail_node_id"], "conv-create-start");
        assert_eq!(details["turn_count"], 0);

        let outsider_details = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: "/api/conversations/conv-create-test".to_owned(),
                body: Vec::new(),
                principal: json!({
                    "sub": "bob",
                    "role": "ro",
                    "ns": "conversation",
                    "storage_ns": "conversation",
                    "capabilities": ["read_graph"]
                }),
            })
            .await;
        assert_eq!(outsider_details.status, StatusCode::NOT_FOUND.as_u16());

        let transcript = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: "/api/conversations/conv-create-test/turns".to_owned(),
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
        assert_eq!(transcript.status, StatusCode::OK.as_u16());
        let transcript: Value = serde_json::from_slice(&transcript.body).unwrap();
        assert_eq!(transcript, json!({"conversation_id": "conv-create-test", "turns": []}));

        let owner_mismatch = service.execute(create("bob")).await;
        assert_eq!(owner_mismatch.status, StatusCode::FORBIDDEN.as_u16());
        let duplicate = service.execute(create("alice")).await;
        assert_eq!(duplicate.status, StatusCode::CONFLICT.as_u16());

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let secret = "conversation-create-mcp-test-secret";
        let token = jsonwebtoken::encode(
            &jsonwebtoken::Header::new(jsonwebtoken::Algorithm::HS256),
            &json!({
                "sub": "alice",
                "role": "rw",
                "ns": ["conversation"],
                "storage_ns": "conversation",
                "capabilities": ["spawn_process"],
                "iss": "local",
                "exp": now + 300
            }),
            &jsonwebtoken::EncodingKey::from_secret(secret.as_bytes()),
        )
        .unwrap();
        let application = Arc::new(service);
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: Vec::new(),
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig {
                    mode: "dev".to_owned(),
                    algorithm: Some("HS256".to_owned()),
                    key: Some(secret.to_owned()),
                    jwks_json: None,
                    issuer: Some("local".to_owned()),
                    audience: None,
                    oidc_providers_json: None,
                    auth_db_url: None,
                },
            },
            application.clone(),
        );
        let mcp_create = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/conversation")
                    .header("content-type", "application/json")
                    .header("authorization", format!("Bearer {token}"))
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":"create-live","method":"tools/call","params":{"name":"conversation.create","arguments":{"user_id":"alice","conversation_id":"conv-create-mcp","start_node_id":"conv-create-mcp-start"}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(mcp_create.status(), StatusCode::OK);
        let mcp_body = to_bytes(mcp_create.into_body(), 8192).await.unwrap();
        let mcp_result: Value = serde_json::from_slice(&mcp_body).unwrap();
        assert_eq!(mcp_result["result"]["conversation_id"], "conv-create-mcp");
        assert_eq!(mcp_result["result"]["start_node_id"], "conv-create-mcp-start");
        let mcp_created_details = application
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: "/api/conversations/conv-create-mcp".to_owned(),
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
        assert_eq!(mcp_created_details.status, StatusCode::OK.as_u16());
        assert_eq!(
            serde_json::from_slice::<Value>(&mcp_created_details.body).unwrap()["start_node_id"],
            "conv-create-mcp-start"
        );

        drop(application);
        let (cleanup, connection) = tokio_postgres::connect(&dsn, tokio_postgres::NoTls)
            .await
            .unwrap();
        tokio::spawn(async move {
            let _ = connection.await;
        });
        cleanup
            .batch_execute(&format!(
                "DROP SCHEMA IF EXISTS \"{schema}_conversation\" CASCADE; \
                 DROP SCHEMA IF EXISTS \"{schema}_workflow\" CASCADE; \
                 DROP SCHEMA IF EXISTS \"{schema}_wisdom\" CASCADE; \
                 DROP SCHEMA IF EXISTS \"{schema}\" CASCADE"
            ))
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
        assert!(service.supports_http_operation("POST", "/api/conversations"));
        assert!(service.supports_http_operation("POST", "/api/document"));
        assert!(service.supports_http_operation(
            "POST",
            "/api/conversations/{conversation_id}/turns:answer"
        ));
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
        assert!(postgres.supports_mcp_tool("workflow", "workflow.run_evidence"));
        assert!(postgres.supports_mcp_tool("workflow", "workflow.run_lineage"));
        assert!(postgres.supports_mcp_tool("workflow", "workflow.tool_audit"));
        assert!(postgres.supports_mcp_tool("conversation", "conversation.ask"));
        assert!(postgres.supports_mcp_tool("conversation", "conversation.create"));
        assert!(postgres.supports_mcp_tool(
            "conversation",
            "conversation.get_transcript"
        ));

        let sqlite_path = std::env::temp_dir().join(format!(
            "kogwistar-api-mcp-authority-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let sqlite = SqliteRunApplicationService::open(&sqlite_path).unwrap();
        assert!(!sqlite.supports_mcp_tool("root", "kg_neighbors"));
        assert!(sqlite.supports_mcp_tool("workflow", "workflow.run_status"));
        assert!(!sqlite.supports_mcp_tool("workflow", "workflow.run_evidence"));
        assert!(!sqlite.supports_mcp_tool("workflow", "workflow.run_lineage"));
        assert!(!sqlite.supports_mcp_tool("workflow", "workflow.tool_audit"));
        assert!(!sqlite.supports_mcp_tool("conversation", "conversation.ask"));
        assert!(!sqlite.supports_mcp_tool("conversation", "conversation.create"));
        assert!(!sqlite.supports_mcp_tool(
            "conversation",
            "conversation.get_transcript"
        ));
        assert!(!sqlite.supports_http_operation("GET", "/api/conversations"));
        assert!(!sqlite.supports_http_operation(
            "GET",
            "/api/workflow/runs/{run_id}/lineage"
        ));
        assert!(!sqlite.supports_http_operation("POST", "/api/conversations"));
        assert!(!sqlite.supports_http_operation("POST", "/api/document"));
        assert!(!sqlite.supports_http_operation(
            "POST",
            "/api/conversations/{conversation_id}/turns:answer"
        ));
        drop(sqlite);
        remove_test_sqlite(&sqlite_path);
    }

    #[tokio::test]
    async fn pending_routes_fail_closed_for_every_rust_service() {
        use kogwistar_engine::embedding::{EmbeddingProfile, OllamaEmbeddingProvider};
        use std::sync::Arc;

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
        let search_index_path = std::env::temp_dir().join(format!(
            "kogwistar-api-pending-search-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let configured_postgres = postgres
            .clone()
            .with_search_index_path(&search_index_path)
            .unwrap()
            .with_embedding_provider(Arc::new(
                OllamaEmbeddingProvider::new(
                    EmbeddingProfile {
                        provider: "ollama".to_owned(),
                        model: "pending-route-test".to_owned(),
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
                    "http://127.0.0.1:1",
                    std::time::Duration::from_secs(1),
                )
                .unwrap(),
            ));
        assert!(postgres.supports_http_operation(
            "GET",
            "/api/workflow/runs/{run_id}/lineage"
        ));
        assert!(postgres.supports_http_operation("POST", "/api/document.upsert_tree"));
        assert!(!sqlite.supports_http_operation("POST", "/api/document.upsert_tree"));

        assert!(configured_postgres
            .supports_http_operation("GET", "/api/search_index_hybrid"));

        for service in [
            &sqlite as &dyn ApplicationService,
            &postgres,
            &configured_postgres,
        ] {
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
        drop(configured_postgres);
        let _ = std::fs::remove_file(search_index_path);
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
                if let Some(lifecycle_status) = entity["lifecycle_status"].as_str() {
                    metadata.insert("lifecycle_status".to_owned(), json!(lifecycle_status));
                }
                if let Some(redirect_to_id) = entity["redirect_to_id"].as_str() {
                    metadata.insert("redirect_to_id".to_owned(), json!(redirect_to_id));
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
                if let Some(security_scope) = entity["security_scope"].as_str() {
                    acl_metadata.insert("security_scope".to_owned(), json!(security_scope));
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

        let foreign_scope = GraphScope {
            workspace_id: Some("workspace-b".to_owned()),
            ..scope.clone()
        };
        service
            .store
            .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                scope: foreign_scope,
                table: "gke_nodes".to_owned(),
                record: GraphRecord {
                    id: "foreign-workspace-nearest".to_owned(),
                    document: Some(
                        json!({"id":"foreign-workspace-nearest"}).to_string(),
                    ),
                    metadata: Map::from_iter([
                        ("entity_type".to_owned(), json!("node")),
                        ("workspace_id".to_owned(), json!("workspace-b")),
                        ("label".to_owned(), json!("foreign-workspace-nearest")),
                    ]),
                    embedding: Some(vec![1.0, 0.0]),
                },
                embedding_dim: 2,
            })
            .await
            .unwrap();
        service
            .store
            .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                record: GraphRecord {
                    id: "acl-foreign-workspace-nearest".to_owned(),
                    document: None,
                    metadata: Map::from_iter([
                        ("entity_type".to_owned(), json!("acl_record")),
                        ("acl_truth_graph".to_owned(), fixture["truth_graph"].clone()),
                        (
                            "acl_target_entity_id".to_owned(),
                            json!("foreign-workspace-nearest"),
                        ),
                        ("acl_target_grain".to_owned(), json!("node")),
                        ("acl_version".to_owned(), json!(1)),
                        ("acl_mode".to_owned(), json!("public")),
                    ]),
                    embedding: None,
                },
                embedding_dim: 2,
            })
            .await
            .unwrap();

        for principal in fixture["principals"].as_array().unwrap() {
            let subject = principal["id"].as_str().unwrap();
            let mut principal_claims = json!({
                "sub": subject,
                "ns": scope.namespace,
                "storage_ns": scope.namespace,
                "workspace_id": scope.workspace_id,
                "capabilities": ["read_graph"]
            });
            if let Some(scope_parts) = principal.get("scope_parts") {
                for key in ["tenant", "workspace", "project"] {
                    if let Some(value) = scope_parts.get(key) {
                        principal_claims[key] = value.clone();
                    }
                }
            }
            for (entity_kind, expected_key) in [("node", "nodes"), ("edge", "edges")] {
                let request_body = json!({
                    "entity_kind": entity_kind,
                    "query_embeddings": fixture["query_embeddings"],
                    "metric": fixture["metric"],
                    "truth_graph": fixture["truth_graph"],
                    "n_results": fixture["n_results"],
                    "include": ["distances"]
                });
                let response = service
                    .execute_async(ApiEffectRequest {
                        contract_version: 1,
                        method: "POST".to_owned(),
                        path_and_query: "/internal/graph/vector-query".to_owned(),
                        body: serde_json::to_vec(&request_body).unwrap(),
                        principal: principal_claims.clone(),
                    })
                    .await;
                assert_eq!(response.status, StatusCode::OK.as_u16());
                let result: Value = serde_json::from_slice(&response.body).unwrap();
                assert_eq!(
                    result["ids"],
                    principal["expected"][expected_key],
                    "unexpected {entity_kind} vector results for {subject}"
                );
                let redirect = &fixture["current_vector_redirect_contract"][entity_kind];
                for row in result["ids"].as_array().unwrap() {
                    let ids = row.as_array().unwrap();
                    assert!(!ids.contains(&redirect["source_id"]));
                    assert!(ids.contains(&redirect["target_id"]));
                }
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

        let revoked_node_id = "visible-public";
        let revoked_acl_metadata = Map::from_iter([
            ("entity_type".to_owned(), json!("acl_record")),
            (
                "acl_truth_graph".to_owned(),
                fixture["truth_graph"].clone(),
            ),
            (
                "acl_target_entity_id".to_owned(),
                json!(revoked_node_id),
            ),
            ("acl_target_grain".to_owned(), json!("node")),
            ("acl_version".to_owned(), json!(2)),
            ("acl_mode".to_owned(), json!("private")),
            ("owner_id".to_owned(), json!("owner-b")),
        ]);
        service
            .store
            .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                record: GraphRecord {
                    id: format!("acl-{revoked_node_id}"),
                    document: None,
                    metadata: revoked_acl_metadata,
                    embedding: None,
                },
                embedding_dim: 0,
            })
            .await
            .unwrap();
        let revoked_response = service
            .execute_async(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/internal/graph/vector-query".to_owned(),
                body: serde_json::to_vec(&json!({
                    "entity_kind": "node",
                    "query_embeddings": [fixture["query_embeddings"][0]],
                    "metric": fixture["metric"],
                    "truth_graph": fixture["truth_graph"],
                    "where": {"label": revoked_node_id},
                    "n_results": 1,
                    "include": ["distances"]
                }))
                .unwrap(),
                principal: json!({
                    "sub": "reader-a",
                    "ns": scope.namespace,
                    "storage_ns": scope.namespace,
                    "workspace_id": scope.workspace_id,
                    "capabilities": ["read_graph"]
                }),
            })
            .await;
        assert_eq!(revoked_response.status, StatusCode::OK.as_u16());
        let revoked_result: Value = serde_json::from_slice(&revoked_response.body).unwrap();
        assert_eq!(revoked_result["ids"], json!([[]]));

        let item_node_id = "node-source-span-private";
        let item_grounding_id = format!("{item_node_id}::mention::0");
        let item_span_id = format!("{item_grounding_id}::span::0");
        service
            .store
            .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                record: GraphRecord {
                    id: item_node_id.to_owned(),
                    document: Some(item_node_id.to_owned()),
                    metadata: Map::from_iter([
                        ("entity_type".to_owned(), json!("node")),
                        ("label".to_owned(), json!(item_node_id)),
                        ("mentions".to_owned(), json!([{"spans": [{}]}])),
                    ]),
                    embedding: Some(vec![1.0, 0.0]),
                },
                embedding_dim: 2,
            })
            .await
            .unwrap();
        for (record_id, grain, target_item_id, mode) in [
            ("acl-source-node", "node", None, "public"),
            (
                "acl-source-grounding",
                "grounding",
                Some(item_grounding_id.as_str()),
                "public",
            ),
            (
                "acl-source-span",
                "span",
                Some(item_span_id.as_str()),
                "private",
            ),
        ] {
            let mut metadata = Map::from_iter([
                ("entity_type".to_owned(), json!("acl_record")),
                ("acl_truth_graph".to_owned(), fixture["truth_graph"].clone()),
                ("acl_target_entity_id".to_owned(), json!(item_node_id)),
                ("acl_target_grain".to_owned(), json!(grain)),
                ("acl_version".to_owned(), json!(1)),
                ("acl_mode".to_owned(), json!(mode)),
            ]);
            if let Some(target_item_id) = target_item_id {
                metadata.insert("acl_target_item_id".to_owned(), json!(target_item_id));
            }
            if mode == "private" {
                metadata.insert("owner_id".to_owned(), json!("owner-b"));
            }
            service
                .store
                .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                    scope: scope.clone(),
                    table: "gke_nodes".to_owned(),
                    record: GraphRecord {
                        id: record_id.to_owned(),
                        document: None,
                        metadata,
                        embedding: None,
                    },
                    embedding_dim: 0,
                })
                .await
                .unwrap();
        }
        let response = service
            .execute_async(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/internal/graph/vector-query".to_owned(),
                body: serde_json::to_vec(&json!({
                    "entity_kind": "node",
                    "query_embeddings": [[1.0, 0.0]],
                    "metric": "cosine",
                    "truth_graph": fixture["truth_graph"],
                    "where": {"label": item_node_id},
                    "n_results": 1,
                    "include": ["distances"]
                }))
                .unwrap(),
                principal: json!({
                    "sub": "reader-a",
                    "ns": scope.namespace,
                    "storage_ns": scope.namespace,
                    "workspace": scope.workspace_id,
                    "capabilities": ["read_graph"]
                }),
            })
            .await;
        assert_eq!(response.status, StatusCode::OK.as_u16());
        let result: Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(result["ids"], json!([[]]));

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

    #[tokio::test]
    #[ignore = "requires KOGWISTAR_TEST_PG_DSN; uses a disposable PostgreSQL schema"]
    async fn postgres_graph_neighbors_omit_cross_scope_endpoint() {
        let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
            .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
        let schema = format!("rust_neighbor_scope_{}", uuid::Uuid::new_v4().simple());
        let service = PostgresRunApplicationService::from_dsn(&dsn, &schema).unwrap();
        service.ensure_schema().await.unwrap();
        service
            .store
            .create_graph_schema(2, kogwistar_store_postgres::GraphTableNames::default())
            .await
            .unwrap();

        let namespace = format!("neighbor-scope-{}", uuid::Uuid::new_v4().simple());
        let source_scope = GraphScope {
            namespace: namespace.clone(),
            workspace_id: Some("workspace-a".to_owned()),
            graph_space: Some("facts".to_owned()),
        };
        let foreign_scope = GraphScope {
            workspace_id: Some("workspace-b".to_owned()),
            ..source_scope.clone()
        };
        for (scope, table, id, metadata, document) in [
            (
                source_scope.clone(),
                "gke_nodes",
                "source-a",
                json!({"entity_type":"node","workspace_id":"workspace-a","graph_space":"facts"}),
                json!({"id":"source-a","label":"Source"}),
            ),
            (
                foreign_scope,
                "gke_nodes",
                "target-b",
                json!({"entity_type":"node","workspace_id":"workspace-b","graph_space":"facts"}),
                json!({"id":"target-b","label":"Foreign target"}),
            ),
            (
                source_scope.clone(),
                "gke_edges",
                "edge-a-b",
                json!({"entity_type":"edge","workspace_id":"workspace-a","graph_space":"facts","source_ids":["source-a"],"target_ids":["target-b"]}),
                json!({"id":"edge-a-b","source_ids":["source-a"],"target_ids":["target-b"]}),
            ),
        ] {
            service
                .store
                .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                    scope,
                    table: table.to_owned(),
                    record: GraphRecord {
                        id: id.to_owned(),
                        document: Some(document.to_string()),
                        metadata: metadata.as_object().unwrap().clone(),
                        embedding: None,
                    },
                    embedding_dim: 0,
                })
                .await
                .unwrap();
        }

        let response = service
            .graph_neighbors_effect(&ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/internal/graph/neighbors".to_owned(),
                body: br#"{"rid":"source-a"}"#.to_vec(),
                principal: json!({
                    "sub":"reader-a",
                    "storage_ns":namespace,
                    "workspace":"workspace-a",
                    "workspace_id":"workspace-a",
                    "graph_space":"facts",
                    "capabilities":["read_graph"]
                }),
            })
            .await;
        assert_eq!(response.status, StatusCode::OK.as_u16());
        let result: Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(result["nodes"], json!([]));
        assert_eq!(result["edges"], json!([]));

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
            .batch_execute(&format!(
                "DROP SCHEMA IF EXISTS \"{schema}_wisdom\" CASCADE; DROP SCHEMA \"{schema}\" CASCADE"
            ))
            .await
            .unwrap();
        drop(client);
        connection_task.await.unwrap().unwrap();
    }

    #[tokio::test]
    #[ignore = "requires KOGWISTAR_TEST_PG_DSN; uses a disposable PostgreSQL schema"]
    async fn postgres_graph_neighbors_enforce_project_and_source_evidence_acl() {
        let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
            .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
        let schema = format!("rust_neighbor_acl_{}", uuid::Uuid::new_v4().simple());
        let service = PostgresRunApplicationService::from_dsn(&dsn, &schema).unwrap();
        service.ensure_schema().await.unwrap();
        service
            .store
            .create_graph_schema(2, kogwistar_store_postgres::GraphTableNames::default())
            .await
            .unwrap();

        let namespace = format!("neighbor-acl-{}", uuid::Uuid::new_v4().simple());
        let scope = GraphScope {
            namespace: namespace.clone(),
            workspace_id: Some("workspace-a".to_owned()),
            graph_space: Some("facts".to_owned()),
        };
        let mut records = Vec::new();
        for (id, mentions) in [
            ("project-root", json!([])),
            ("project-target", json!([])),
            ("evidence-root", json!([])),
            (
                "evidence-target",
                json!([{"spans":[{"doc_id":"restricted-source"}]}]),
            ),
            ("control-root", json!([])),
            ("control-target", json!([])),
            (
                "same-id",
                json!([{"spans":[{"doc_id":"shared-source"}]}]),
            ),
        ] {
            records.push((
                "gke_nodes",
                id,
                json!({
                    "entity_type":"node",
                    "workspace_id":"workspace-a",
                    "graph_space":"facts",
                    "mentions":mentions
                }),
                json!({"id":id,"label":id,"mentions":mentions}),
            ));
        }
        for (id, source_id, target_id, mentions) in [
            ("project-edge", "project-root", "project-target", json!([])),
            ("evidence-edge", "evidence-root", "evidence-target", json!([])),
            ("control-edge", "control-root", "control-target", json!([])),
            ("same-node-link", "control-root", "same-id", json!([])),
            (
                "same-id",
                "control-root",
                "same-id",
                json!([{"spans":[{"doc_id":"shared-source"}]}]),
            ),
        ] {
            records.push((
                "gke_edges",
                id,
                json!({
                    "entity_type":"edge",
                    "workspace_id":"workspace-a",
                    "graph_space":"facts",
                    "mentions":mentions,
                    "source_ids":[source_id],
                    "target_ids":[target_id]
                }),
                json!({"id":id,"source_ids":[source_id],"target_ids":[target_id]}),
            ));
        }
        for (table, id, metadata, document) in records {
            service
                .store
                .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                    scope: scope.clone(),
                    table: table.to_owned(),
                    record: GraphRecord {
                        id: id.to_owned(),
                        document: Some(document.to_string()),
                        metadata: metadata.as_object().unwrap().clone(),
                        embedding: None,
                    },
                    embedding_dim: 0,
                })
                .await
                .unwrap();
        }

        let mut acl_cases = Vec::new();
        for id in [
            "project-root",
            "evidence-root",
            "evidence-target",
            "control-root",
            "control-target",
        ] {
            acl_cases.push((format!("acl-{id}"), "node", id.to_owned(), None, "public", None));
        }
        for id in ["project-edge", "evidence-edge", "control-edge"] {
            acl_cases.push((format!("acl-{id}"), "edge", id.to_owned(), None, "public", None));
        }
        acl_cases.push((
            "acl-same-node-link".to_owned(),
            "edge",
            "same-node-link".to_owned(),
            None,
            "public",
            None,
        ));
        acl_cases.push((
            "acl-project-target".to_owned(),
            "node",
            "project-target".to_owned(),
            None,
            "scope",
            Some("project-b"),
        ));
        acl_cases.push((
            "acl-evidence-grounding".to_owned(),
            "grounding",
            "evidence-target".to_owned(),
            Some("evidence-target::mention::0"),
            "public",
            None,
        ));
        acl_cases.push((
            "acl-evidence-span".to_owned(),
            "span",
            "evidence-target".to_owned(),
            Some("evidence-target::mention::0::span::0"),
            "private",
            Some("another-principal"),
        ));
        acl_cases.extend([
            (
                "acl-same-node".to_owned(),
                "node",
                "same-id".to_owned(),
                None,
                "public",
                None,
            ),
            (
                "acl-same-edge".to_owned(),
                "edge",
                "same-id".to_owned(),
                None,
                "public",
                None,
            ),
            (
                "acl-same-node-grounding".to_owned(),
                "grounding",
                "same-id".to_owned(),
                Some("__kogwistar_acl_usage_v2__:node:same-id::mention::0"),
                "public",
                None,
            ),
            (
                "acl-same-node-span".to_owned(),
                "span",
                "same-id".to_owned(),
                Some("__kogwistar_acl_usage_v2__:node:same-id::mention::0::span::0"),
                "public",
                None,
            ),
            (
                "acl-same-edge-grounding".to_owned(),
                "grounding",
                "same-id".to_owned(),
                Some("__kogwistar_acl_usage_v2__:edge:same-id::mention::0"),
                "public",
                None,
            ),
            (
                "acl-same-edge-span".to_owned(),
                "span",
                "same-id".to_owned(),
                Some("__kogwistar_acl_usage_v2__:edge:same-id::mention::0::span::0"),
                "private",
                Some("another-principal"),
            ),
        ]);
        for (acl_id, grain, entity_id, item_id, mode, security_scope) in acl_cases {
            let mut metadata = json!({
                "entity_type":"acl_record",
                "acl_truth_graph":"knowledge",
                "acl_target_grain":grain,
                "acl_target_entity_id":entity_id,
                "acl_version":1,
                "acl_mode":mode
            });
            if let Some(item_id) = item_id {
                metadata["acl_target_item_id"] = json!(item_id);
            }
            if let Some(security_scope) = security_scope {
                metadata["security_scope"] = json!(security_scope);
            }
            if mode == "private" {
                metadata["owner_id"] = json!("another-principal");
            }
            service
                .store
                .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                    scope: scope.clone(),
                    table: "gke_nodes".to_owned(),
                    record: GraphRecord {
                        id: acl_id,
                        document: None,
                        metadata: metadata.as_object().unwrap().clone(),
                        embedding: None,
                    },
                    embedding_dim: 0,
                })
                .await
                .unwrap();
        }

        let read_neighbors = |root: &str| ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/internal/graph/neighbors".to_owned(),
            body: serde_json::to_vec(&json!({"rid":root,"truth_graph":"knowledge"})).unwrap(),
            principal: json!({
                "sub":"reader-a",
                "storage_ns":namespace,
                "workspace":"workspace-a",
                "workspace_id":"workspace-a",
                "graph_space":"facts",
                "security_scope":"project-a",
                "capabilities":["read_graph"]
            }),
        };
        for root in ["project-root", "evidence-root"] {
            let response = service.graph_neighbors_effect(&read_neighbors(root)).await;
            assert_eq!(response.status, StatusCode::OK.as_u16());
            let result: Value = serde_json::from_slice(&response.body).unwrap();
            assert_eq!(result["nodes"], json!([]), "root={root}: {result}");
            assert_eq!(result["edges"], json!([]), "root={root}: {result}");
        }
        let control = service
            .graph_neighbors_effect(&read_neighbors("control-root"))
            .await;
        assert_eq!(control.status, StatusCode::OK.as_u16());
        let control: Value = serde_json::from_slice(&control.body).unwrap();
        assert_eq!(control["nodes"], json!(["control-target", "same-id"]));
        assert_eq!(control["edges"], json!(["control-edge", "same-node-link"]));

        let mut owner_request = read_neighbors("control-root");
        let owner_principal = json!({
            "sub":"another-principal",
            "storage_ns":namespace,
            "workspace":"workspace-a",
            "workspace_id":"workspace-a",
            "graph_space":"facts",
            "security_scope":"project-a",
            "capabilities":["read_graph"]
        });
        owner_request.principal = owner_principal;
        let same_id_owner = service.graph_neighbors_effect(&owner_request).await;
        assert_eq!(same_id_owner.status, StatusCode::OK.as_u16());
        let same_id_owner: Value = serde_json::from_slice(&same_id_owner.body).unwrap();
        assert_eq!(same_id_owner["nodes"], json!(["control-target", "same-id"]));
        assert_eq!(
            same_id_owner["edges"],
            json!(["control-edge", "same-id", "same-node-link"])
        );

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
            .batch_execute(&format!(
                "DROP SCHEMA IF EXISTS \"{schema}_wisdom\" CASCADE; DROP SCHEMA \"{schema}\" CASCADE"
            ))
            .await
            .unwrap();
        drop(client);
        connection_task.await.unwrap().unwrap();
    }

    #[tokio::test]
    #[ignore = "requires KOGWISTAR_TEST_PG_DSN; uses a disposable PostgreSQL schema"]
    async fn postgres_graph_snapshot_omits_conflicting_workspace_node() {
        let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
            .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
        let schema = format!("rust_snapshot_scope_{}", uuid::Uuid::new_v4().simple());
        let service = PostgresRunApplicationService::from_dsn(&dsn, &schema).unwrap();
        service.ensure_schema().await.unwrap();
        service
            .store
            .create_graph_schema(2, kogwistar_store_postgres::GraphTableNames::default())
            .await
            .unwrap();

        let namespace = format!("snapshot-scope-{}", uuid::Uuid::new_v4().simple());
        for (workspace, id) in [
            ("workspace-a", "source-a"),
            ("workspace-b", "foreign-node-b"),
        ] {
            service
                .store
                .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                    scope: GraphScope {
                        namespace: namespace.clone(),
                        workspace_id: Some(workspace.to_owned()),
                        graph_space: Some("facts".to_owned()),
                    },
                    table: "gke_nodes".to_owned(),
                    record: GraphRecord {
                        id: id.to_owned(),
                        document: Some(json!({"id":id,"label":id}).to_string()),
                        metadata: Map::from_iter([
                            ("entity_type".to_owned(), json!("node")),
                            ("workspace_id".to_owned(), json!(workspace)),
                            ("graph_space".to_owned(), json!("facts")),
                        ]),
                        embedding: None,
                    },
                    embedding_dim: 0,
                })
                .await
                .unwrap();
            service
                .store
                .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                    scope: GraphScope {
                        namespace: namespace.clone(),
                        workspace_id: Some(workspace.to_owned()),
                        graph_space: Some("facts".to_owned()),
                    },
                    table: "gke_nodes".to_owned(),
                    record: GraphRecord {
                        id: format!("acl-{id}"),
                        document: None,
                        metadata: Map::from_iter([
                            ("entity_type".to_owned(), json!("acl_record")),
                            ("acl_truth_graph".to_owned(), json!("knowledge")),
                            ("acl_target_grain".to_owned(), json!("node")),
                            ("acl_target_entity_id".to_owned(), json!(id)),
                            ("acl_mode".to_owned(), json!("public")),
                            ("acl_version".to_owned(), json!(1)),
                        ]),
                        embedding: None,
                    },
                    embedding_dim: 0,
                })
                .await
                .unwrap();
        }

        for (table, id, metadata, document) in [
            (
                "gke_edges",
                "edge-to-foreign-node",
                json!({
                    "entity_type":"edge",
                    "workspace_id":"workspace-a",
                    "graph_space":"facts",
                    "source_ids":["source-a"],
                    "target_ids":["foreign-node-b"]
                }),
                json!({
                    "id":"edge-to-foreign-node",
                    "source_ids":["source-a"],
                    "target_ids":["foreign-node-b"]
                }),
            ),
            (
                "gke_nodes",
                "acl-edge-to-foreign-node",
                json!({
                    "entity_type":"acl_record",
                    "acl_truth_graph":"knowledge",
                    "acl_target_grain":"edge",
                    "acl_target_entity_id":"edge-to-foreign-node",
                    "acl_mode":"public",
                    "acl_version":1
                }),
                Value::Null,
            ),
            (
                "gke_edges",
                "edge-malformed-endpoint",
                json!({
                    "entity_type":"edge",
                    "workspace_id":"workspace-a",
                    "graph_space":"facts",
                    "source_ids":"source-a",
                    "target_ids":["source-a"]
                }),
                json!({
                    "id":"edge-malformed-endpoint",
                    "source_ids":"source-a",
                    "target_ids":["source-a"]
                }),
            ),
            (
                "gke_nodes",
                "acl-edge-malformed-endpoint",
                json!({
                    "entity_type":"acl_record",
                    "acl_truth_graph":"knowledge",
                    "acl_target_grain":"edge",
                    "acl_target_entity_id":"edge-malformed-endpoint",
                    "acl_mode":"public",
                    "acl_version":1
                }),
                Value::Null,
            ),
        ] {
            service
                .store
                .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                    scope: GraphScope {
                        namespace: namespace.clone(),
                        workspace_id: Some("workspace-a".to_owned()),
                        graph_space: Some("facts".to_owned()),
                    },
                    table: table.to_owned(),
                    record: GraphRecord {
                        id: id.to_owned(),
                        document: (!document.is_null()).then(|| document.to_string()),
                        metadata: metadata.as_object().unwrap().clone(),
                        embedding: None,
                    },
                    embedding_dim: 0,
                })
                .await
                .unwrap();
        }

        let response = service
            .execute_async(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: INTERNAL_GRAPH_VIZ_D3_ROUTE.to_owned(),
                body: br#"{"graph_type":"knowledge"}"#.to_vec(),
                principal: json!({
                    "sub":"reader-a",
                    "storage_ns":namespace,
                    "workspace":"workspace-a",
                    "workspace_id":"workspace-a",
                    "graph_space":"facts",
                    "capabilities":["read_graph"]
                }),
            })
            .await;
        let response_status = response.status;
        let response_body = response.body;

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
            .batch_execute(&format!(
                "DROP SCHEMA IF EXISTS \"{schema}_wisdom\" CASCADE; DROP SCHEMA \"{schema}\" CASCADE"
            ))
            .await
            .unwrap();
        drop(client);
        connection_task.await.unwrap().unwrap();

        assert_eq!(response_status, StatusCode::OK.as_u16());
        let result: Value = serde_json::from_slice(&response_body).unwrap();
        let node_ids = result["nodes"]
            .as_array()
            .expect("D3 snapshot has node array")
            .iter()
            .filter_map(|node| node["id"].as_str())
            .collect::<Vec<_>>();
        assert!(node_ids.contains(&"source-a"), "snapshot: {result}");
        assert!(!node_ids.contains(&"foreign-node-b"), "snapshot: {result}");
        assert!(
            !node_ids.contains(&"edge-to-foreign-node"),
            "snapshot reified an edge with an out-of-scope endpoint: {result}"
        );
        assert!(
            !node_ids.contains(&"edge-malformed-endpoint"),
            "snapshot reified an edge with malformed endpoint data: {result}"
        );
        assert!(
            result["links"].as_array().unwrap().iter().all(|link| {
                link["source"] != "edge-to-foreign-node"
                    && link["target"] != "edge-to-foreign-node"
                    && link["source"] != "edge-malformed-endpoint"
                    && link["target"] != "edge-malformed-endpoint"
            }),
            "snapshot linked an edge with an invalid endpoint: {result}"
        );
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
            runtime_worker_recipient_for_operation("rust-native/noop", None, None, None),
            "rust-native-worker"
        );
        assert_eq!(
            runtime_worker_recipient_for_operation("noop", None, None, None),
            "python-worker"
        );
        assert_eq!(
            runtime_worker_recipient_for_operation(
                "rust-native/noop",
                Some("numpy-cpu-2"),
                None,
                None,
            ),
            "rust-native-worker"
        );
        assert_eq!(
            runtime_worker_recipient_for_operation("noop", Some("numpy-cpu-2"), None, None),
            runtime_worker_recipient(Some("numpy-cpu-2"))
        );
        assert_eq!(
            runtime_worker_recipient_for_operation(
                "increment",
                None,
                Some("rust-native"),
                None,
            ),
            "rust-native-worker"
        );
        assert_eq!(
            runtime_worker_recipient_for_operation(
                "increment",
                None,
                Some("python-worker"),
                None,
            ),
            "python-worker"
        );
        assert!(runtime_graph_plan_from_snapshot(
            r#"{"nodes":[{"id":"start","metadata":{"wf_start":true,"wf_executor_target":"gpu"}}],"edges":[]}"#
        )
        .is_none());

        let identity = json!({
            "schema_version": 1,
            "environment_id": "python-cpu-v1",
            "artifact_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "dependency_lock_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        });
        assert_eq!(
            runtime_worker_recipient_for_operation(
                "parse",
                Some("python-cpu-v1"),
                None,
                Some(&identity),
            ),
            "python-worker-identity-Pg8kjaa_r5hcsnNN8gIUMKd6x7EO4Sugfo92UM4XSVg"
        );
    }

    #[test]
    fn runtime_graph_plan_matches_python_design_topology_invariants() {
        let snapshot = json!({
            "nodes": [
                {"id": "start", "metadata": {"wf_start": true}},
                {"id": "end", "metadata": {"wf_terminal": true}},
                {"id": "unused", "metadata": {}}
            ],
            "edges": [{
                "id": "route",
                "source_ids": ["start", "unused"],
                "target_ids": ["end", "unused"],
                "metadata": {}
            }]
        });
        let plan = runtime_graph_plan_from_snapshot(&snapshot.to_string())
            .expect("first source/target form the Python-compatible transition");
        assert_eq!(plan.routes.len(), 1);
        assert_eq!(plan.routes[0].source_node_id, "start");
        assert_eq!(plan.routes[0].target_node_id, "end");

        let duplicate_start = json!({
            "nodes": [
                {"id": "start", "metadata": {"wf_start": true}},
                {"id": "second", "metadata": {"wf_start": true}}
            ],
            "edges": []
        });
        assert!(runtime_graph_plan_from_snapshot(&duplicate_start.to_string()).is_none());

        let missing_node_id = json!({
            "nodes": [
                {"id": "start", "metadata": {"wf_start": true}},
                {"metadata": {}}
            ],
            "edges": []
        });
        assert!(runtime_graph_plan_from_snapshot(&missing_node_id.to_string()).is_none());

        let dangling_edge = json!({
            "nodes": [{"id": "start", "metadata": {"wf_start": true}}],
            "edges": [{"id": "bad", "source_ids": ["start"], "target_ids": ["missing"]}]
        });
        assert!(runtime_graph_plan_from_snapshot(&dangling_edge.to_string()).is_none());

        let empty_endpoints = json!({
            "nodes": [{"id": "start", "metadata": {"wf_start": true}}],
            "edges": [{"id": "bad", "source_ids": [], "target_ids": ["start"]}]
        });
        assert!(runtime_graph_plan_from_snapshot(&empty_endpoints.to_string()).is_none());

        let no_reachable_terminal = json!({
            "nodes": [
                {"id": "start", "metadata": {"wf_start": true}},
                {"id": "loop", "metadata": {}}
            ],
            "edges": [
                {"id": "to-loop", "source_ids": ["start"], "target_ids": ["loop"]},
                {"id": "to-start", "source_ids": ["loop"], "target_ids": ["start"]}
            ]
        });
        assert!(runtime_graph_plan_from_snapshot(&no_reachable_terminal.to_string()).is_none());
    }

    #[test]
    fn runtime_graph_plan_freezes_only_valid_python_worker_identity() {
        let identity = json!({
            "schema_version": 1,
            "environment_id": "python-cpu-v1",
            "artifact_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "dependency_lock_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        });
        let mut snapshot = json!({
            "nodes": [
                {"id": "start", "metadata": {
                    "wf_start": true,
                    "wf_op": "parse",
                    "wf_environment_id": "python-cpu-v1",
                    "wf_worker_identity": identity
                }},
                {"id": "end", "metadata": {"wf_terminal": true}}
            ],
            "edges": [{"id": "route", "source_ids": ["start"], "target_ids": ["end"]}]
        });
        let plan = runtime_graph_plan_from_snapshot(&snapshot.to_string())
            .expect("valid worker identity is frozen in the workflow plan");
        assert_eq!(plan.node_worker_identities.get("start"), Some(&identity));

        snapshot["nodes"][0]["metadata"]["wf_worker_identity"]["environment_id"] =
            json!("stale-env");
        assert!(runtime_graph_plan_from_snapshot(&snapshot.to_string()).is_none());

        snapshot["nodes"][0]["metadata"]["wf_worker_identity"] = identity;
        snapshot["nodes"][0]["metadata"]["wf_executor_target"] = json!("rust-native");
        assert!(runtime_graph_plan_from_snapshot(&snapshot.to_string()).is_none());
    }

    #[test]
    fn explicit_worker_identity_map_must_match_environment_and_executor() {
        let identity = json!({
            "schema_version": 1,
            "environment_id": "python-cpu-v1",
            "artifact_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "dependency_lock_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        });
        let identities = BTreeMap::from([("start".to_owned(), identity.clone())]);
        let environments = BTreeMap::from([("start".to_owned(), "python-cpu-v1".to_owned())]);
        assert!(runtime_worker_identity_map_is_valid(
            &identities,
            &environments,
            &BTreeMap::new()
        ));

        let stale_environment = BTreeMap::from([("start".to_owned(), "stale-env".to_owned())]);
        assert!(!runtime_worker_identity_map_is_valid(
            &identities,
            &stale_environment,
            &BTreeMap::new()
        ));

        let native_target = BTreeMap::from([("start".to_owned(), "rust-native".to_owned())]);
        assert!(!runtime_worker_identity_map_is_valid(
            &identities,
            &environments,
            &native_target
        ));

        let malformed = BTreeMap::from([("start".to_owned(), json!({"schema_version": 99}))]);
        assert!(!runtime_worker_identity_map_is_valid(
            &malformed,
            &environments,
            &BTreeMap::new()
        ));

        let snapshot = WorkflowDesignSnapshot {
            workflow_id: "wf".to_owned(),
            version: 1,
            seq: 1,
            payload_json: json!({
                "nodes": [{"metadata": {"wf_worker_identity": {"schema_version": 99}}}],
                "edges": []
            })
            .to_string(),
            schema_version: 1,
            created_at_ms: 0,
        };
        assert!(runtime_snapshot_declares_worker_identity(Some(&snapshot)));
        assert!(!runtime_snapshot_declares_worker_identity(None));
    }

    #[tokio::test]
    async fn sqlite_native_noop_result_is_durable_through_runtime_route() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use std::time::{SystemTime, UNIX_EPOCH};
        use tower::ServiceExt;

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-native-noop-{nonce}.sqlite3"));
        let store = SqliteStore::open(&path).unwrap();
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: vec!["reader".to_owned()],
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig::default(),
            },
            Arc::new(SqliteRunApplicationService::open(&path).unwrap()),
        );

        let submit = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/workflow/runs")
                    .header("x-kogwistar-roles", "reader")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "run_id": "native-noop-run",
                            "workflow_id": "native-noop-workflow",
                            "conversation_id": "native-noop-conversation",
                            "initial_state": {"seed": 1},
                            "runtime_kind": "sync",
                            "start_node_id": "start",
                            "node_ops": {
                                "start": "rust-native/noop",
                                "finish": "rust-native/emit-result"
                            },
                            "runtime_routes": [{
                                "source_node_id": "start",
                                "target_node_id": "finish",
                                "join_mask": 0
                            }]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(submit.status(), StatusCode::ACCEPTED);

        let claim = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/claim")
                    .header("x-kogwistar-roles", "reader")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "claimed_by": "rust-native-worker",
                            "limit": 1,
                            "lease_seconds": 60,
                            "recipient_ids": ["rust-native-worker"]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(claim.status(), StatusCode::OK);
        let claim_body: Value = serde_json::from_slice(
            &to_bytes(claim.into_body(), 1_048_576).await.unwrap(),
        )
        .unwrap();
        let work = claim_body["work"][0].clone();
        assert_eq!(work["payload"]["op"], "rust-native/noop");

        let execute = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/native/execute")
                    .header("x-kogwistar-roles", "reader")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "handoff": {
                                "message_id": work["message_id"],
                                "claimed_by": work["claimed_by"],
                                "run_id": work["run_id"],
                                "step_id": work["step_id"],
                                "correlation_id": work["correlation_id"]
                            },
                            "payload": work["payload"]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(execute.status(), StatusCode::OK);

        let run = store.get_server_run("native-noop-run").unwrap().unwrap();
        assert_eq!(run.status, "running");
        let events = store
            .list_server_run_events("native-noop-run", 0, 100)
            .unwrap();
        let transition = events
            .iter()
            .find(|event| {
                if event.event_type != "workflow.recorded_transition.v1" {
                    return false;
                }
                serde_json::from_str::<Value>(&event.payload_json)
                    .map(|payload| {
                        payload["reduced"]["state"]["last_node_id"] == "start"
                            && payload["reduced"]["state"]["last_step_seq"] == 0
                    })
                    .unwrap_or(false)
            })
            .expect("native noop transition must be durable");
        let transition_payload: Value = serde_json::from_str(&transition.payload_json).unwrap();
        assert_eq!(
            transition_payload["reduced"]["result"]["workflow_status"],
            "succeeded",
            "unexpected persisted noop transition: {transition_payload}"
        );
        assert_eq!(
            transition_payload["reduced"]["result"]["final_state"]["seed"],
            1
        );
        assert!(events.iter().any(|event| {
            event.event_type == "workflow.trace.v1"
                && event.payload_json.contains("runtime.native.step.completed")
        }));

        drop(app);
        drop(store);
        std::fs::remove_file(path).unwrap();
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
            Arc::new(service.clone()),
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
        let fault_connection = rusqlite::Connection::open(&path).unwrap();
        fault_connection
            .execute_batch(
                "CREATE TRIGGER reject_native_transition BEFORE INSERT ON server_run_events \
                 WHEN NEW.run_id = 'native-run' \
                 AND NEW.event_type = 'workflow.recorded_transition.v1' \
                 BEGIN SELECT RAISE(ABORT, 'injected native transition failure'); END;",
            )
            .unwrap();
        let event_count_before_failed_commit = store
            .list_server_run_events("native-run", 0, 100)
            .unwrap()
            .len();
        let failed_execute = client
            .post(format!("{base_url}/internal/runtime/native/execute"))
            .header("x-kogwistar-roles", "reader")
            .json(&execute_payload)
            .send()
            .await
            .unwrap();
        assert_eq!(failed_execute.status(), StatusCode::CONFLICT);
        let failed_body: Value = failed_execute.json().await.unwrap();
        assert_eq!(failed_body["code"], "KOGWISTAR_RUNTIME_RESULT_CONFLICT");
        assert_eq!(
            store
                .list_server_run_events("native-run", 0, 100)
                .unwrap()
                .len(),
            event_count_before_failed_commit,
            "failed transition must leave canonical run history unchanged"
        );
        assert_eq!(
            store
                .get_projected_lane_message(work["message_id"].as_str().unwrap())
                .unwrap()
                .unwrap()
                .status,
            "claimed",
            "failed result commit must not acknowledge the native work item"
        );
        fault_connection
            .execute_batch("DROP TRIGGER reject_native_transition;")
            .unwrap();

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
        let events_before_wrong_executor = store
            .list_server_run_events("mixed-language-run", 0, 100)
            .unwrap();
        let wrong_executor_result = client
            .post(format!("{base_url}/internal/runtime/native/execute"))
            .header("x-kogwistar-roles", "reader")
            .json(&json!({
                "handoff": {
                    "message_id": python_before_work["message_id"],
                    "claimed_by": python_before_work["claimed_by"],
                    "run_id": python_before_work["run_id"],
                    "step_id": python_before_work["step_id"],
                    "correlation_id": python_before_work["correlation_id"]
                },
                "payload": python_before_work["payload"]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(wrong_executor_result.status(), StatusCode::CONFLICT);
        assert_eq!(
            store
                .list_server_run_events("mixed-language-run", 0, 100)
                .unwrap(),
            events_before_wrong_executor,
            "Python-target work rejected by the native route must remain unconsumed"
        );
        let python_before_effect = json!({
            "contract_version": 1,
            "effect_id": "mixed-python-before",
            "state_update": [["u", {"python_before_seen": true}]],
            "successors": [],
            "result": null
        });
        let events_before_invalid_handoff = store
            .list_server_run_events("mixed-language-run", 0, 100)
            .unwrap();
        for (body, expected_status) in [
            (b"{".to_vec(), StatusCode::UNPROCESSABLE_ENTITY),
            (
                vec![b' '; MAX_RUNTIME_WORKER_RESULT_BYTES + 1],
                StatusCode::PAYLOAD_TOO_LARGE,
            ),
        ] {
            let rejected = service
                .execute(ApiEffectRequest {
                    contract_version: 1,
                    method: "POST".to_owned(),
                    path_and_query: "/internal/runtime/results".to_owned(),
                    body,
                    principal: json!({"roles": ["reader"]}),
                })
                .await;
            assert_eq!(rejected.status, expected_status.as_u16());
            assert_eq!(
                store
                    .list_server_run_events("mixed-language-run", 0, 100)
                    .unwrap(),
                events_before_invalid_handoff,
                "malformed or oversized result must not append durable events"
            );
            assert_eq!(
                store
                    .get_projected_lane_message(
                        python_before_work["message_id"].as_str().unwrap()
                    )
                    .unwrap()
                    .unwrap()
                    .status,
                "claimed",
                "malformed or oversized result must preserve the active claim"
            );
        }
        for field in ["message_id", "claimed_by", "run_id", "step_id", "correlation_id"] {
            let mut invalid_handoff = json!({
                "handoff": {
                    "message_id": python_before_work["message_id"],
                    "claimed_by": python_before_work["claimed_by"],
                    "run_id": python_before_work["run_id"],
                    "step_id": python_before_work["step_id"],
                    "correlation_id": python_before_work["correlation_id"]
                },
                "effect": python_before_effect
            });
            invalid_handoff["handoff"][field] = json!(format!("forged-{field}"));
            let rejected = client
                .post(format!("{base_url}/internal/runtime/results"))
                .header("x-kogwistar-roles", "reader")
                .json(&invalid_handoff)
                .send()
                .await
                .unwrap();
            assert_eq!(
                rejected.status(),
                StatusCode::CONFLICT,
                "mismatched handoff field {field} must not be accepted"
            );
            assert_eq!(
                store
                    .list_server_run_events("mixed-language-run", 0, 100)
                    .unwrap(),
                events_before_invalid_handoff,
                "mismatched handoff field {field} must not append durable events"
            );
            assert_eq!(
                store
                    .get_projected_lane_message(
                        python_before_work["message_id"].as_str().unwrap()
                    )
                    .unwrap()
                    .unwrap()
                    .status,
                "claimed",
                "mismatched handoff field {field} must not acknowledge the active claim"
            );
        }
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
                "effect": python_before_effect
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
        let events_before_python_result_bypass = store
            .list_server_run_events("mixed-language-run", 0, 100)
            .unwrap();
        let python_result_bypass = client
            .post(format!("{base_url}/internal/runtime/results"))
            .header("x-kogwistar-roles", "reader")
            .json(&json!({
                "handoff": {
                    "message_id": native_middle_work["message_id"],
                    "claimed_by": native_middle_work["claimed_by"],
                    "run_id": native_middle_work["run_id"],
                    "step_id": native_middle_work["step_id"],
                    "correlation_id": native_middle_work["correlation_id"]
                },
                "effect": {
                    "contract_version": 1,
                    "effect_id": "forged-python-result-for-native-step",
                    "state_update": [["u", {"counter": 999}]],
                    "successors": []
                }
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(python_result_bypass.status(), StatusCode::CONFLICT);
        assert_eq!(
            store
                .list_server_run_events("mixed-language-run", 0, 100)
                .unwrap(),
            events_before_python_result_bypass,
            "generic Python results must not consume Rust-targeted work"
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
        let trace = &native_trace["trace"];
        let token_id = native_middle_work["payload"]["token_id"]
            .as_str()
            .expect("native step payload has its runtime token");
        let step_seq = native_middle_work["payload"]["step_seq"]
            .as_i64()
            .expect("native step payload has its sequence");
        assert_eq!(trace["trace_id"], "mixed-language-run");
        assert_eq!(trace["attempt"], 1);
        assert_eq!(trace["token_id"], token_id);
        assert_eq!(trace["step_seq"], step_seq);
        assert_eq!(trace["node_id"], "native-middle");
        assert_eq!(trace["span_id"], format!("stp:mixed-language-run:{token_id}:{step_seq}:1:native-middle"));
        assert_eq!(trace["parent_span_id"], format!("tok:mixed-language-run:{token_id}"));
        assert_eq!(native_trace["run_id"], "mixed-language-run");
        assert_eq!(native_trace["step_id"], "native-middle");
        assert_eq!(native_trace["token_id"], token_id);

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

        let native_events_before_restart = store
            .list_server_run_events("native-run", 0, 100)
            .unwrap();
        let final_transition = native_events_before_restart
            .iter()
            .rev()
            .find(|event| event.event_type == "workflow.recorded_transition.v1")
            .expect("native run must have a terminal recorded transition");
        let final_transition_payload: Value =
            serde_json::from_str(&final_transition.payload_json).unwrap();
        let expected_state_digest = final_transition_payload["reduced"]["state_digest"]
            .as_str()
            .expect("terminal transition must persist a state digest")
            .to_owned();
        let expected_runtime_state = store
            .read_recorded_runtime_state(
                "native-run",
                "native-workflow",
                "native-conversation",
            )
            .unwrap()
            .expect("native runtime state must be reconstructable before reopen");
        assert_eq!(expected_runtime_state.status.server_status(), "succeeded");
        assert_eq!(expected_runtime_state.state["counter"], json!(2));

        server.abort();
        let _ = server.await;
        drop(service);
        drop(store);

        let reopened_store = SqliteStore::open(&path).unwrap();
        let reopened_run = reopened_store
            .get_server_run("native-run")
            .unwrap()
            .expect("completed native run must survive SQLite reopen");
        assert_eq!(reopened_run.status, "succeeded");
        let reopened_runtime_state = reopened_store
            .read_recorded_runtime_state(
                "native-run",
                "native-workflow",
                "native-conversation",
            )
            .unwrap()
            .expect("runtime state must reconstruct from durable transitions");
        assert_eq!(reopened_runtime_state, expected_runtime_state);
        assert_eq!(
            kogwistar_runtime::state_digest(&reopened_runtime_state).unwrap(),
            expected_state_digest
        );
        let reopened_events = reopened_store
            .list_server_run_events("native-run", 0, 100)
            .unwrap();
        let reopened_final_transition = reopened_events
            .iter()
            .rev()
            .find(|event| event.event_type == "workflow.recorded_transition.v1")
            .expect("terminal recorded transition must survive SQLite reopen");
        let reopened_transition_payload: Value =
            serde_json::from_str(&reopened_final_transition.payload_json).unwrap();
        assert_eq!(
            reopened_transition_payload["reduced"]["state_digest"],
            expected_state_digest
        );
        drop(reopened_store);
        drop(fault_connection);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn answer_workflow_admission_supports_per_node_python_and_registered_rust_executors() {
        let native_operations = crate::native_answer_persist::registry().unwrap();
        let contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/answer-workflow-runtime-contract-v1.json"
        ))
        .unwrap();
        for (workflow_id, workflow) in contract["workflows"].as_object().unwrap() {
            let required_ops = workflow["required_ops"]
                .as_array()
                .unwrap()
                .iter()
                .map(|op| op.as_str().unwrap().to_owned())
                .collect::<Vec<_>>();
            let python_ops = workflow["python_executor_ops"]
                .as_array()
                .unwrap()
                .iter()
                .map(|op| op.as_str().unwrap().to_owned())
                .collect::<std::collections::BTreeSet<_>>();
            let required_routes = workflow["required_routes"].as_array().unwrap();
            let python_op_version = workflow["python_op_version"].as_str().unwrap();
            let nodes = required_ops
                .iter()
                .map(|op| {
                    json!({
                        "id": format!("node-{op}"),
                        "label": op,
                        "metadata": {
                            "entity_type": "workflow_node",
                            "workflow_id": workflow_id,
                            "wf_op": op,
                            "wf_version": python_op_version,
                            "wf_start": op == "start",
                            "wf_fanout": false,
                            "wf_join": false,
                            "wf_executor_target": "python-worker"
                        }
                    })
                })
                .collect::<Vec<_>>();
            let edges = required_routes
                .iter()
                .enumerate()
                .map(|(index, route)| {
                    let relation = if route["predicate"].is_null() {
                        "wf_next"
                    } else {
                        "wf_conditional"
                    };
                    json!({
                        "id": format!("edge-{index}"),
                        "label": relation,
                        "source_ids": [format!("node-{}", route["from"].as_str().unwrap())],
                        "target_ids": [format!("node-{}", route["to"].as_str().unwrap())],
                        "metadata": {
                            "entity_type": "workflow_edge",
                            "workflow_id": workflow_id,
                            "wf_predicate": route["predicate"],
                            "wf_priority": route["priority"],
                            "wf_is_default": route["default"],
                            "wf_multiplicity": route["multiplicity"]
                        }
                    })
                })
                .collect::<Vec<_>>();
            let snapshot = json!({"nodes": nodes, "edges": edges});
            let plan = runtime_graph_plan_from_snapshot(&snapshot.to_string())
                .expect("shared Python answer design contract parses as a Rust runtime graph");
            assert!(answer_workflow_plan_is_executable(workflow_id, &plan, &native_operations));
            assert!(answer_workflow_admission_ready(
                workflow_id,
                Some(&plan),
                &native_operations
            ));
            assert!(!answer_workflow_admission_ready(
                workflow_id,
                None,
                &native_operations
            ));

            let mut missing_operation = plan.clone();
            missing_operation
                .node_ops
                .remove(&format!("node-{}", required_ops[1]));
            assert!(!answer_workflow_plan_is_executable(
                workflow_id,
                &missing_operation,
                &native_operations
            ));

            let mut altered_route_contract = plan.clone();
            if let Some(route) = altered_route_contract
                .routes
                .iter_mut()
                .find(|route| route.predicate.is_some())
            {
                route.predicate = Some("unknown_predicate".to_owned());
            } else {
                altered_route_contract.routes[0].is_default = false;
            }
            assert!(!answer_workflow_plan_is_executable(
                workflow_id,
                &altered_route_contract,
                &native_operations
            ));

            let mut wrong_route_target = plan.clone();
            wrong_route_target.routes[0].target_node_id = "node-start".to_owned();
            assert!(!answer_workflow_plan_is_executable(
                workflow_id,
                &wrong_route_target,
                &native_operations
            ));

            let mut unexpected_fanout = plan.clone();
            unexpected_fanout.routes[0].source_fanout = true;
            assert!(!answer_workflow_plan_is_executable(
                workflow_id,
                &unexpected_fanout,
                &native_operations
            ));

            let mut unexpected_join = plan.clone();
            unexpected_join.join_node_ids.push("node-start".to_owned());
            assert!(!answer_workflow_plan_is_executable(
                workflow_id,
                &unexpected_join,
                &native_operations
            ));

            let mut incompatible_operation_version = plan.clone();
            incompatible_operation_version
                .node_versions
                .insert("node-start".to_owned(), "v2".to_owned());
            assert!(!answer_workflow_plan_is_executable(
                workflow_id,
                &incompatible_operation_version,
                &native_operations
            ));

            let mut unknown_executor = plan.clone();
            let python_node = plan
                .node_ops
                .iter()
                .find(|(_, op)| python_ops.contains(*op))
                .unwrap()
                .0;
            unknown_executor
                .node_executor_targets
                .insert(python_node.clone(), "unknown-worker".to_owned());
            assert!(!answer_workflow_plan_is_executable(
                workflow_id,
                &unknown_executor,
                &native_operations
            ));

            if workflow_id == "agentic_answering.v2" {
                let native_node = plan
                    .node_ops
                    .iter()
                    .find(|(_, op)| op.as_str() == "aa_persist_response")
                    .unwrap()
                    .0;
                let mut mixed_executor_plan = plan.clone();
                mixed_executor_plan
                    .node_executor_targets
                    .insert(native_node.clone(), "rust-native".to_owned());
                assert!(answer_workflow_plan_is_executable(
                    workflow_id,
                    &mixed_executor_plan,
                    &native_operations
                ));

                let selector_node = plan
                    .node_ops
                    .iter()
                    .find(|(_, op)| op.as_str() == "aa_select_used_evidence")
                    .unwrap()
                    .0;
                let mut native_bm25_plan = plan.clone();
                native_bm25_plan
                    .node_executor_targets
                    .insert(selector_node.clone(), "rust-native".to_owned());
                assert!(answer_workflow_plan_is_executable(
                    workflow_id,
                    &native_bm25_plan,
                    &native_operations
                ));

                let mut rust_only_plan = plan.clone();
                for node_id in rust_only_plan.node_ops.keys().cloned().collect::<Vec<_>>() {
                    rust_only_plan
                        .node_executor_targets
                        .insert(node_id, "rust-native".to_owned());
                }
                assert!(answer_workflow_plan_is_executable(
                    workflow_id,
                    &rust_only_plan,
                    &native_operations
                ), "the complete answer workflow must support an all-Rust path");

                let mut native_answer_plan = mixed_executor_plan.clone();
                native_answer_plan.node_executor_targets.insert(
                    "node-aa_generate_answer_with_citations".to_owned(),
                    "rust-native".to_owned(),
                );
                assert!(answer_workflow_plan_is_executable(
                    workflow_id,
                    &native_answer_plan,
                    &native_operations
                ));

                let mut native_repair_plan = mixed_executor_plan.clone();
                native_repair_plan.node_executor_targets.insert(
                    "node-aa_validate_or_repair_citations".to_owned(),
                    "rust-native".to_owned(),
                );
                assert!(answer_workflow_plan_is_executable(
                    workflow_id,
                    &native_repair_plan,
                    &native_operations
                ));

                let mut unsupported_native_plan = mixed_executor_plan.clone();
                unsupported_native_plan.node_ops.insert(
                    "node-aa_validate_or_repair_citations".to_owned(),
                    "unregistered_citation_operation".to_owned(),
                );
                assert!(!answer_workflow_plan_is_executable(
                    workflow_id,
                    &unsupported_native_plan,
                    &native_operations
                ));
            }

            let mut implicit_noop = plan;
            implicit_noop
                .node_ops
                .insert("node-accidental".to_owned(), "noop".to_owned());
            assert!(!answer_workflow_plan_is_executable(
                workflow_id,
                &implicit_noop,
                &native_operations
            ));
        }
        assert!(!is_answer_workflow_id("ordinary.workflow"));
        assert!(answer_workflow_admission_ready(
            "ordinary.workflow",
            None,
            &native_operations
        ));
    }
