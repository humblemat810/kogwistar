    #[test]
    fn document_graph_validation_matches_provenance_and_pointer_contract() {
        let request = |body: Value| ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/document.validate_graph".to_owned(),
            body: serde_json::to_vec(&body).unwrap(),
            principal: json!({}),
        };
        let valid = document_graph_validation_effect(&request(json!({
            "doc_id":"doc-validate",
            "insertion_method":"pytest",
            "nodes":[{
                "id":"node-1",
                "label":"Node",
                "type":"entity",
                "summary":"Valid node",
                "metadata":{"pointers":[{
                    "source_cluster_id":"cluster-1",
                    "start_char":0,
                    "end_char":4,
                    "verbatim_text":"Node",
                }]},
            }],
            "edges":[{
                "id":"edge-1",
                "label":"Edge",
                "type":"relationship",
                "summary":"Valid edge",
                "relation":"links",
                "source_ids":["node-1"],
                "target_ids":["node-1"],
                "source_edge_ids":[],
                "target_edge_ids":[],
                "mentions":[{"spans":[{
                    "collection_page_url":"doc://doc-validate",
                    "document_page_url":"doc://doc-validate#cluster-1",
                    "doc_id":"doc-validate",
                    "insertion_method":"pytest",
                    "start_char":0,
                    "end_char":4,
                }]}],
            }],
        })))
        .unwrap();
        let valid = serde_json::from_slice::<Value>(&valid.body).unwrap();
        assert_eq!(valid, json!({"ok":true,"node_errors":{},"edge_errors":{}}));

        let invalid = document_graph_validation_effect(&request(json!({
            "doc_id":"doc-invalid",
            "nodes":[{"id":"node-bad","label":"Bad","type":"entity","summary":"","mentions":[]}],
            "edges":[{"id":"edge-bad","label":"Bad","type":"relationship","summary":"","mentions":[],"relation":"links","source_ids":[],"target_ids":[],"source_edge_ids":[],"target_edge_ids":[]}],
        })))
        .unwrap();
        let invalid = serde_json::from_slice::<Value>(&invalid.body).unwrap();
        assert_eq!(invalid["ok"], false);
        assert!(invalid["node_errors"]["node-bad"].is_string());
        assert!(invalid["edge_errors"]["edge-bad"].is_string());
    }

    #[tokio::test]
    async fn document_validate_graph_http_route_matches_shared_success_contract() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/document-validate-graph-success-v1.json"
        ))
        .expect("valid document validation fixture");
        let service = Arc::new(
            PostgresRunApplicationService::from_dsn(
                "postgres://unused:unused@127.0.0.1:1/unused",
                "document_validate_graph_route_contract",
            )
            .unwrap(),
        );
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: Vec::new(),
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig::default(),
            },
            service,
        );
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/document.validate_graph")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&fixture["request"]).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status().as_u16(), fixture["status"]);
        let body = to_bytes(response.into_body(), 4096).await.unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&body).unwrap(),
            fixture["response"]
        );
    }

    #[test]
    fn document_graph_validation_missing_nodes_matches_python_error_detail() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/document-graph-validation-error-v1.json"
        ))
        .unwrap();
        let response = document_graph_validation_effect(&ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/document.validate_graph".to_owned(),
            body: serde_json::to_vec(&fixture["request"]).unwrap(),
            principal: json!({}),
        })
        .unwrap();

        assert_eq!(response.status, StatusCode::UNPROCESSABLE_ENTITY.as_u16());
        assert_eq!(response.content_type, "application/json");
        let body: Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(body["detail"], fixture["python_detail"]);
        assert_eq!(body["code"], fixture["rust_legacy_fields"]["code"]);
        assert_eq!(body["message"], fixture["rust_legacy_fields"]["message"]);
    }

    #[test]
    fn internal_document_projection_reuses_wire_contract_and_write_authority() {
        let capabilities = Arc::new(Mutex::new(CapabilityState::default()));
        let request = |principal: Value| ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: INTERNAL_DOCUMENT_PROJECTION_ROUTE.to_owned(),
            body: serde_json::to_vec(&json!({
                "doc_id": "doc-projection",
                "doc_type": "text",
                "insertion_method": "parser",
                "content": "hello",
                "metadata": {"source": "fixture"}
            }))
            .unwrap(),
            principal,
        };
        let denied = document_projection_write(
            &request(json!({"sub":"reader","role":"ro","capabilities":[]})),
            &capabilities,
        )
        .unwrap_err();
        assert_eq!(denied.status, StatusCode::FORBIDDEN);

        let public_request = ApiEffectRequest {
            path_and_query: PUBLIC_DOCUMENT_UPSERT_ROUTE.to_owned(),
            ..request(json!({"sub":"reader","role":"ro","capabilities":[]}))
        };
        assert!(is_document_projection_route(
            &public_request.method,
            &public_request.path_and_query
        ));
        let public_denied = document_projection_write(&public_request, &capabilities).unwrap_err();
        assert_eq!(public_denied.status, StatusCode::FORBIDDEN);

        let write = document_projection_write(
            &request(json!({
                "sub":"writer",
                "role":"rw",
                "ns":["docs"],
                "capabilities":["write_graph"]
            })),
            &capabilities,
        )
        .unwrap();
        assert_eq!(write.record.id, "doc-projection");
        assert_eq!(write.record.document.as_deref(), Some("hello"));
        assert_eq!(write.scope.namespace, "docs");
        assert_eq!(write.record.metadata["doc_type"], "text");
        assert_eq!(write.record.metadata["insertion_method"], "parser");

        let ocr_request = ApiEffectRequest {
            body: serde_json::to_vec(&json!({
                "doc_id": "doc-ocr",
                "doc_type": "ocr",
                "content": "{}"
            }))
            .unwrap(),
            ..request(json!({
                "sub":"writer",
                "role":"rw",
                "ns":["docs"],
                "capabilities":["write_graph"]
            }))
        };
        let ocr = document_projection_write(&ocr_request, &capabilities).unwrap_err();
        assert_eq!(ocr.status, StatusCode::NOT_IMPLEMENTED);
    }

    #[tokio::test]
    async fn internal_document_lookup_requires_read_authority_and_uses_principal_scope() {
        let nonce = uuid::Uuid::new_v4();
        let path = std::env::temp_dir().join(format!("kogwistar-document-lookup-{nonce}.sqlite3"));
        let store = kogwistar_store_sqlite::SqliteStore::open(&path).unwrap();
        let scope = GraphScope {
            namespace: "docs".to_owned(),
            workspace_id: Some("workspace-a".to_owned()),
            graph_space: None,
        };
        DocumentStore::upsert_document_projection(
            &store,
            DocumentProjectionUpsert {
                scope,
                record: GraphRecord {
                    id: "doc-1".to_owned(),
                    document: Some("fixture source".to_owned()),
                    metadata: Default::default(),
                    embedding: None,
                },
                embedding_dim: 0,
            },
        )
        .await
        .unwrap();
        let capabilities = Arc::new(Mutex::new(CapabilityState::default()));
        let request = |principal: Value| ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: INTERNAL_DOCUMENT_LOOKUP_ROUTE.to_owned(),
            body: serde_json::to_vec(&json!({"file_name":"doc-1"})).unwrap(),
            principal,
        };

        let denied = document_id_lookup_effect(
            &request(json!({
                "sub":"reader",
                "role":"ro",
                "ns":["docs"],
                "storage_ns":"docs",
                "workspace":"workspace-a",
                "capabilities":[]
            })),
            &store,
            &capabilities,
        )
        .await;
        assert_eq!(denied.status, StatusCode::FORBIDDEN);

        let found = document_id_lookup_effect(
            &request(json!({
                "sub":"reader",
                "role":"ro",
                "ns":["docs"],
                "storage_ns":"docs",
                "workspace":"workspace-a",
                "capabilities":["read_graph"]
            })),
            &store,
            &capabilities,
        )
        .await;
        assert_eq!(found.status, StatusCode::OK);
        assert_eq!(serde_json::from_slice::<Value>(&found.body).unwrap()["ids"], json!(["doc-1"]));

        let outside_scope = document_id_lookup_effect(
            &request(json!({
                "sub":"reader",
                "role":"ro",
                "ns":["docs"],
                "storage_ns":"other-docs",
                "workspace":"workspace-a",
                "capabilities":["read_graph"]
            })),
            &store,
            &capabilities,
        )
        .await;
        assert_eq!(outside_scope.status, StatusCode::OK);
        assert_eq!(serde_json::from_slice::<Value>(&outside_scope.body).unwrap()["ids"], json!([]));

        drop(store);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn sqlite_document_id_mcp_returns_python_compatible_short_id_mapping() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
        use tower::ServiceExt;

        let contract: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/document-id-from-file-name-mcp-v1.json"
        )))
        .unwrap();

        let nonce = uuid::Uuid::new_v4();
        let path = std::env::temp_dir().join(format!("kogwistar-document-mcp-{nonce}.sqlite3"));
        let service = SqliteRunApplicationService::open(&path).unwrap();
        let bearer = encode(
            &Header::new(Algorithm::HS256),
            &json!({
                "sub":"document-reader",
                "role":"ro",
                "ns":["docs"],
                "storage_ns":"docs",
                "capabilities":["read_graph"],
                "exp":4102444800_u64
            }),
            &EncodingKey::from_secret(b"document-mcp-test-secret"),
        )
        .unwrap();
        let scope = GraphScope {
            namespace: "docs".to_owned(),
            workspace_id: None,
            graph_space: None,
        };
        DocumentStore::upsert_document_projection(
            &service.store,
            DocumentProjectionUpsert {
                scope,
                record: GraphRecord {
                    id: contract["expected_canonical_id"]
                        .as_str()
                        .unwrap()
                        .to_owned(),
                    document: Some("source text".to_owned()),
                    metadata: Default::default(),
                    embedding: None,
                },
                embedding_dim: 0,
            },
        )
        .await
        .unwrap();

        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: Vec::new(),
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig {
                    mode: "jwt".to_owned(),
                    algorithm: Some("HS256".to_owned()),
                    key: Some("document-mcp-test-secret".to_owned()),
                    ..AuthConfig::default()
                },
            },
            Arc::new(service.clone()),
        );
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header("authorization", format!("Bearer {bearer}"))
                    .body(Body::from(
                        json!({
                            "jsonrpc":"2.0",
                            "id":"lookup",
                            "method":"tools/call",
                            "params": {
                                "name": contract["operation"],
                                "arguments": contract["arguments"]
                            }
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 65_536).await.unwrap();
        let payload: Value = serde_json::from_slice(&body).unwrap();
        let mapping = &payload["result"]["id_mapping"][0];
        assert_eq!(mapping["file_name"], contract["expected_file_name"]);
        let short_id = mapping["id"].as_str().unwrap();
        assert!(short_id.starts_with(contract["short_id_prefix"].as_str().unwrap()));
        let map_key = crate::short_id_store::short_id_map_key_for_bearer(Some(&bearer));
        assert_eq!(
            crate::short_id_store::canonical_id_for(".shortids", &map_key, short_id).unwrap(),
            contract["expected_canonical_id"].as_str().unwrap()
        );

        drop(service);
        let short_id_root = std::path::Path::new(".shortids");
        let _ = std::fs::remove_file(short_id_root.join(format!("{map_key}.json")));
        let _ = std::fs::remove_file(short_id_root.join(format!("{map_key}.lock")));
        let _ = std::fs::remove_file(path);
    }
