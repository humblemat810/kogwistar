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
