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
    async fn deferred_admin_document_delete_preserves_graph_events() {
        let sqlite_path = std::env::temp_dir().join(format!(
            "kogwistar-deferred-admin-delete-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let service = SqliteRunApplicationService::open(&sqlite_path).unwrap();
        let node_id = "admin-delete-retained-node";
        let doc_id = "admin-delete-retained-doc";
        let metadata = json!({
            "entity_type": "node",
            "label": "Retained node",
            "namespace": "docs",
            "doc_id": doc_id,
            "lifecycle_status": "active"
        });
        service
            .store
            .apply_canonical_graph_mutation_batch(GraphMutationBatch {
                mutations: vec![GraphMutation {
                    scope: GraphScope {
                        namespace: "docs".to_owned(),
                        workspace_id: None,
                        graph_space: None,
                    },
                    table: "gke_nodes".to_owned(),
                    entity_kind: "node".to_owned(),
                    event_id: "admin-delete-retained-event".to_owned(),
                    op: "ADD".to_owned(),
                    payload: json!({"id": node_id, "doc_id": doc_id, "metadata": metadata}),
                    record: GraphRecord {
                        id: node_id.to_owned(),
                        document: None,
                        metadata: serde_json::from_value(metadata).unwrap(),
                        embedding: None,
                    },
                    embedding_dim: 0,
                }],
                index_jobs: Vec::new(),
            })
            .unwrap();
        let before = service.store.replay_raw_events("docs", 0, 10).unwrap();
        assert_eq!(before.len(), 1);
        assert!(!service.supports_http_operation("DELETE", "/admin/doc/{doc_id}"));

        let response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "DELETE".to_owned(),
                path_and_query: format!("/admin/doc/{doc_id}"),
                body: Vec::new(),
                principal: json!({"sub": "admin", "role": "rw", "ns": ["docs"]}),
            })
            .await;

        assert_eq!(response.status, StatusCode::NOT_IMPLEMENTED.as_u16());
        assert_eq!(
            serde_json::from_slice::<Value>(&response.body).unwrap()["code"],
            "KOGWISTAR_ROUTE_DEFERRED"
        );
        assert_eq!(
            service.store.replay_raw_events("docs", 0, 10).unwrap(),
            before,
            "the unsupported Rust route must not tombstone or mutate canonical graph events"
        );

        drop(service);
        remove_test_sqlite(&sqlite_path);
    }

    #[tokio::test]
    async fn metadata_only_sqlite_mcp_discovery_hides_graph_operations() {
        use axum::{
            body::{Body, to_bytes},
            http::Request,
        };
        use std::sync::Arc;
        use tower::ServiceExt;

        let sqlite_path = std::env::temp_dir().join(format!(
            "kogwistar-api-sqlite-graph-authority-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let sqlite = Arc::new(SqliteRunApplicationService::open(&sqlite_path).unwrap());

        assert_eq!(
            sqlite.graph_authority_capability(),
            GraphAuthorityCapability::Unsupported
        );
        assert!(!sqlite.supports_mcp_tool("root", "kg_neighbors"));
        assert!(!sqlite.supports_mcp_tool("root", "kg_load_persisted"));
        assert!(!sqlite.supports_mcp_tool("root", "kg_propose_bruteforce"));
        assert!(!sqlite.supports_http_operation("POST", "/internal/graph/neighbors"));
        assert!(!sqlite.supports_http_operation("POST", "/internal/conversation/transcript"));
        assert!(!sqlite.supports_http_operation("GET", "/api/conversations/{conversation_id}"));
        assert!(
            !sqlite.supports_http_operation("GET", "/api/conversations/{conversation_id}/turns")
        );

        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: Vec::new(),
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig::default(),
            },
            sqlite.clone(),
        );
        let listing = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":"list","method":"tools/list"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(listing.status(), StatusCode::OK);
        let listing: Value = serde_json::from_slice(
            &to_bytes(listing.into_body(), 65_536).await.unwrap(),
        )
        .unwrap();
        let names = listing["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|tool| tool["name"].as_str())
            .collect::<Vec<_>>();
        for unsupported in [
            "kg_neighbors",
            "kg_find_edges",
            "kg_load_persisted",
            "kg_propose_bruteforce",
            "propose_vector",
        ] {
            assert!(!names.contains(&unsupported), "advertised {unsupported}: {names:?}");
        }

        for (id, name, arguments) in [
            (
                "denied-neighbors",
                "kg_neighbors",
                json!({"node_id":"missing"}),
            ),
            (
                "denied-load",
                "kg_load_persisted",
                json!({"inp":{"doc_ids":["doc-a"]}}),
            ),
            (
                "denied-propose",
                "kg_propose_bruteforce",
                json!({"inp":{}}),
            ),
        ] {
            let rejected = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/mcp")
                        .header("content-type", "application/json")
                        .header("x-kogwistar-roles", "reader")
                        .body(Body::from(
                            json!({
                                "jsonrpc":"2.0",
                                "id":id,
                                "method":"tools/call",
                                "params":{"name":name,"arguments":arguments}
                            })
                            .to_string(),
                        ))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(rejected.status(), StatusCode::OK);
            let rejected: Value = serde_json::from_slice(
                &to_bytes(rejected.into_body(), 65_536).await.unwrap(),
            )
            .unwrap();
            assert_eq!(rejected["error"]["code"], -32602, "{name}: {rejected}");
            assert_eq!(
                rejected["error"]["message"],
                "Tool not advertised by this application service",
                "{name}: {rejected}"
            );
        }

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

        drop(app);
        drop(sqlite);
        remove_test_sqlite(&sqlite_path);
    }
