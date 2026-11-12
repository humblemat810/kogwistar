#[tokio::test]
#[ignore = "requires KOGWISTAR_TEST_PG_DSN; uses an isolated PostgreSQL schema"]
async fn postgres_hybrid_search_is_available_over_authenticated_http() {
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
    use std::sync::Arc;
    use tower::ServiceExt;

    struct FixedEmbeddingProvider;
    impl EmbeddingProvider for FixedEmbeddingProvider {
        fn profile(&self) -> &kogwistar_engine::embedding::EmbeddingProfile {
            static PROFILE: std::sync::OnceLock<kogwistar_engine::embedding::EmbeddingProfile> =
                std::sync::OnceLock::new();
            PROFILE.get_or_init(|| kogwistar_engine::embedding::EmbeddingProfile {
                provider: "test".to_owned(),
                model: "hybrid-route-fixed-v1".to_owned(),
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
            })
        }

        fn embed<'a>(
            &'a self,
            texts: &'a [String],
        ) -> kogwistar_engine::embedding::EmbeddingFuture<'a> {
            Box::pin(async move { Ok(texts.iter().map(|_| vec![0.6, 0.8]).collect()) })
        }
    }

    let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
        .expect("ignored hybrid-search HTTP test requires KOGWISTAR_TEST_PG_DSN");
    let suffix = uuid::Uuid::new_v4().simple().to_string();
    let schema = format!("rust_hybrid_http_{suffix}");
    let fts_path = std::env::temp_dir().join(format!("rust-hybrid-http-{suffix}.sqlite3"));
    let service = PostgresRunApplicationService::from_dsn(&dsn, &schema)
        .unwrap()
        .with_search_index_path(&fts_path)
        .unwrap()
        .with_embedding_provider(Arc::new(FixedEmbeddingProvider));
    service.ensure_schema().await.unwrap();
    service
        .store
        .create_graph_schema(2, kogwistar_store_postgres::GraphTableNames::default())
        .await
        .unwrap();

    let namespace = format!("hybrid-{suffix}");
    let node_id = format!("hybrid-node-{suffix}");
    let title = format!("uniquehybridpermit{suffix}");
    let provision = "review the unique hybrid permit";
    let scope = GraphScope {
        namespace: namespace.clone(),
        workspace_id: Some("workspace-a".to_owned()),
        graph_space: Some("knowledge".to_owned()),
    };
    service
        .store
        .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
            scope: scope.clone(),
            table: "gke_nodes".to_owned(),
            record: GraphRecord {
                id: node_id.clone(),
                document: Some(json!({"id":node_id,"label":title}).to_string()),
                metadata: Map::from_iter([
                    ("namespace".to_owned(), json!(namespace)),
                    ("workspace_id".to_owned(), json!("workspace-a")),
                    ("graph_space".to_owned(), json!("knowledge")),
                    ("lifecycle_status".to_owned(), json!("active")),
                ]),
                embedding: None,
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
                id: format!("acl|{node_id}"),
                document: Some("{}".to_owned()),
                metadata: Map::from_iter([
                    ("entity_type".to_owned(), json!("acl_record")),
                    ("acl_truth_graph".to_owned(), json!("knowledge")),
                    ("acl_target_grain".to_owned(), json!("node")),
                    ("acl_target_entity_id".to_owned(), json!(node_id)),
                    ("acl_mode".to_owned(), json!("public")),
                    ("acl_version".to_owned(), json!(1)),
                    ("tombstoned".to_owned(), json!(false)),
                ]),
                embedding: None,
            },
            embedding_dim: 0,
        })
        .await
        .unwrap();

    let entry: kogwistar_contracts::SearchIndexEntryV1 = serde_json::from_value(json!({
        "node_id": node_id,
        "canonical_title": title,
        "keywords": ["uniquehybrid"],
        "aliases": [],
        "provision": provision,
        "doc_id": "doc-hybrid"
    }))
    .unwrap();
    let fts = std::sync::Mutex::new(crate::search_index_fts::open_index(&fts_path).unwrap());
    crate::search_index_fts::admit_search_index_entries(
        &service.store,
        &service.store,
        &fts,
        service.embedding_provider.as_ref().unwrap().as_ref(),
        &namespace,
        "gke_node_index",
        &[entry.clone()],
    )
    .await
    .unwrap();
    assert!(
        crate::search_index_fts::search_fts(&fts.lock().unwrap(), "review", 5)
            .unwrap()
            .iter()
            .any(|hit| hit.node_id == node_id)
    );
    let direct_hits = crate::search_index_fts::hybrid_search(
        &fts,
        &service.store,
        "gke_node_index",
        service.embedding_provider.as_ref().unwrap().as_ref(),
        "review",
        5,
    )
    .await
    .unwrap();
    assert!(
        direct_hits.iter().any(|hit| hit.node_id == node_id),
        "{direct_hits:?}"
    );
    drop(fts);

    let secret = "hybrid-search-http-test-secret";
    let auth = AuthConfig {
        mode: "jwt".to_owned(),
        algorithm: Some("HS256".to_owned()),
        key: Some(secret.to_owned()),
        ..AuthConfig::default()
    };
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let token = |ns: &str, capabilities: &[&str]| {
        encode(
            &Header::new(Algorithm::HS256),
            &json!({
                "sub":"reader",
                "role":"ro",
                "ns":[ns],
                "storage_ns":ns,
                "workspace_id":"workspace-a",
                "graph_space":"knowledge",
                "capabilities":capabilities,
                "exp":now + 300
            }),
            &EncodingKey::from_secret(secret.as_bytes()),
        )
        .unwrap()
    };
    let app = router_with_application(
        ApiState {
            health: snapshot(),
            required_roles: vec![],
            implementation: ImplementationSnapshot::default(),
            auth,
        },
        Arc::new(service),
    );

    let mut routed_entry = entry.clone();
    routed_entry.aliases.push("http-route-revision".to_owned());
    let writer_token = encode(
        &Header::new(Algorithm::HS256),
        &json!({
            "sub":"writer",
            "role":"rw",
            "ns":[namespace],
            "storage_ns":namespace,
            "workspace_id":"workspace-a",
            "graph_space":"knowledge",
            "capabilities":["write_graph"],
            "exp":now + 300
        }),
        &EncodingKey::from_secret(secret.as_bytes()),
    )
    .unwrap();
    for _ in 0..2 {
        let write = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/add_index_entries")
                    .header("authorization", format!("Bearer {writer_token}"))
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({"index":[routed_entry.clone()]}).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(write.status(), StatusCode::OK);
        let payload: Value =
            serde_json::from_slice(&to_bytes(write.into_body(), 65_536).await.unwrap()).unwrap();
        assert_eq!(payload, json!({"ok":true}));
    }

    for (query, resolve_node) in [("review", false), ("review", true)] {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri(format!(
                        "/api/search_index_hybrid?q={query}&limit=5&resolve_node={resolve_node}"
                    ))
                    .header(
                        "authorization",
                        format!("Bearer {}", token(&namespace, &["read_graph"])),
                    )
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let payload: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 65_536).await.unwrap()).unwrap();
        let results = payload["results"].as_array().unwrap();
        let hit = results
            .iter()
            .find(|hit| hit["node_id"] == node_id)
            .unwrap_or_else(|| panic!("authorized indexed node missing from payload: {payload}"));
        assert_eq!(hit["canonical_title"], title);
        if resolve_node {
            assert!(hit["documents"].is_string());
            assert_eq!(hit["metadatas"]["workspace_id"], "workspace-a");
        } else {
            assert!(hit.get("documents").is_none());
            assert!(hit.get("metadatas").is_none());
        }
    }

    let denied = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/search_index_hybrid?q=review")
                .header(
                    "authorization",
                    format!("Bearer {}", token(&namespace, &[])),
                )
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(denied.status(), StatusCode::FORBIDDEN);

    let outside_scope = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/search_index_hybrid?q=review")
                .header(
                    "authorization",
                    format!("Bearer {}", token("other-namespace", &["read_graph"])),
                )
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(outside_scope.status(), StatusCode::OK);
    let outside_scope: Value =
        serde_json::from_slice(&to_bytes(outside_scope.into_body(), 65_536).await.unwrap())
            .unwrap();
    assert_eq!(outside_scope["results"], json!([]));
}
