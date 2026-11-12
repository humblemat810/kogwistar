#[tokio::test]
async fn in_memory_semantic_seed_expand_is_available_through_authenticated_http_mcp() {
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use jsonwebtoken::{EncodingKey, Header, encode};
    use tower::ServiceExt;

    let fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/semantic-seed-expand-v1.json"
    )))
    .unwrap();
    assert_eq!(fixture["schema_version"], 1);
    let seed_id = fixture["expected_seed_ids"][0].as_str().unwrap();
    let edge_id = fixture["expected_first_layer_edge_ids"][0]
        .as_str()
        .unwrap();
    let cross_doc_edge_id = fixture["expected_first_layer_edge_ids"][1]
        .as_str()
        .unwrap();
    let doc_id = fixture["scope"]["doc_id"].as_str().unwrap();
    assert_eq!(fixture["expansion_document_filter"], "not_applied");

    struct FixedEmbeddingProvider {
        calls: Arc<Mutex<Vec<String>>>,
    }
    impl EmbeddingProvider for FixedEmbeddingProvider {
        fn profile(&self) -> &kogwistar_engine::embedding::EmbeddingProfile {
            static PROFILE: std::sync::OnceLock<kogwistar_engine::embedding::EmbeddingProfile> =
                std::sync::OnceLock::new();
            PROFILE.get_or_init(|| kogwistar_engine::embedding::EmbeddingProfile {
                provider: "test".to_owned(),
                model: "in-memory-semantic-test".to_owned(),
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
            let calls = self.calls.clone();
            let inputs = texts.to_vec();
            Box::pin(async move {
                calls.lock().unwrap().extend(inputs);
                Ok(texts.iter().map(|_| vec![1.0, 0.0]).collect())
            })
        }
    }

    let scope = GraphScope {
        namespace: "tenant-a".to_owned(),
        workspace_id: Some("workspace-a".to_owned()),
        graph_space: Some("facts".to_owned()),
    };
    let inactive = InMemoryGraphApplicationService::default();
    assert!(!inactive.supports_mcp_tool("root", "kg_semantic_seed_then_expand_text"));
    assert!(!inactive.supports_http_operation("POST", GRAPH_SEMANTIC_SEED_EXPAND_ROUTE));

    let embedding_calls = Arc::new(Mutex::new(Vec::new()));
    let service = InMemoryGraphApplicationService::default().with_embedding_provider(Arc::new(
        FixedEmbeddingProvider {
            calls: embedding_calls.clone(),
        },
    ));
    assert!(service.supports_mcp_tool("root", "kg_semantic_seed_then_expand_text"));
    assert!(service.supports_http_operation("POST", GRAPH_SEMANTIC_SEED_EXPAND_ROUTE));
    assert!(service.supports_mcp_tool("root", "wisdom.semantic_seed_then_expand"));
    assert!(service.supports_http_operation("POST", WISDOM_SEMANTIC_SEED_EXPAND_ROUTE));
    service
        .store
        .apply_graph_mutations_atomic(vec![
            GraphMutation {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                entity_kind: "node".to_owned(),
                event_id: "semantic-seed-node-event".to_owned(),
                op: "ADD".to_owned(),
                payload: json!({"id":seed_id,"label":"Smoking"}),
                record: GraphRecord {
                    id: seed_id.to_owned(),
                    document: Some(json!({"id":seed_id,"label":"Smoking"}).to_string()),
                    metadata: serde_json::Map::from_iter([
                        ("namespace".to_owned(), json!("tenant-a")),
                        ("workspace_id".to_owned(), json!("workspace-a")),
                        ("graph_space".to_owned(), json!("facts")),
                        ("doc_id".to_owned(), json!(doc_id)),
                    ]),
                    embedding: Some(vec![1.0, 0.0]),
                },
                embedding_dim: 2,
            },
            GraphMutation {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                entity_kind: "node".to_owned(),
                event_id: "semantic-neighbor-event".to_owned(),
                op: "ADD".to_owned(),
                payload: json!({"id":"B","label":"Lung Cancer"}),
                record: GraphRecord {
                    id: "B".to_owned(),
                    document: Some(r#"{"id":"B","label":"Lung Cancer"}"#.to_owned()),
                    metadata: serde_json::Map::from_iter([
                        ("namespace".to_owned(), json!("tenant-a")),
                        ("workspace_id".to_owned(), json!("workspace-a")),
                        ("graph_space".to_owned(), json!("facts")),
                        ("doc_id".to_owned(), json!(doc_id)),
                    ]),
                    embedding: Some(vec![0.0, 1.0]),
                },
                embedding_dim: 2,
            },
            GraphMutation {
                scope: scope.clone(),
                table: "gke_edges".to_owned(),
                entity_kind: "edge".to_owned(),
                event_id: "semantic-edge-event".to_owned(),
                op: "ADD".to_owned(),
                payload: json!({"id":edge_id,"source_ids":[seed_id],"target_ids":["B"]}),
                record: GraphRecord {
                    id: edge_id.to_owned(),
                    document: Some(
                        json!({
                            "id":edge_id,
                            "relation":"causes",
                            "source_ids":[seed_id],
                            "target_ids":["B"]
                        })
                        .to_string(),
                    ),
                    metadata: serde_json::Map::from_iter([
                        ("namespace".to_owned(), json!("tenant-a")),
                        ("workspace_id".to_owned(), json!("workspace-a")),
                        ("graph_space".to_owned(), json!("facts")),
                        ("doc_id".to_owned(), json!(doc_id)),
                        ("source_ids".to_owned(), json!([seed_id])),
                        ("target_ids".to_owned(), json!(["B"])),
                        ("relation".to_owned(), json!("causes")),
                    ]),
                    embedding: None,
                },
                embedding_dim: 0,
            },
            GraphMutation {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                entity_kind: "node".to_owned(),
                event_id: "semantic-seed-acl-event".to_owned(),
                op: "ADD".to_owned(),
                payload: json!({"id":"semantic-seed-node-acl"}),
                record: GraphRecord {
                    id: "semantic-seed-node-acl".to_owned(),
                    document: None,
                    metadata: serde_json::Map::from_iter([
                        ("namespace".to_owned(), json!("tenant-a")),
                        ("workspace_id".to_owned(), json!("workspace-a")),
                        ("graph_space".to_owned(), json!("facts")),
                        ("entity_type".to_owned(), json!("acl_record")),
                        ("acl_truth_graph".to_owned(), json!("knowledge")),
                        ("acl_target_grain".to_owned(), json!("node")),
                        ("acl_target_entity_id".to_owned(), json!(seed_id)),
                        ("acl_version".to_owned(), json!(1)),
                        ("acl_mode".to_owned(), json!("public")),
                    ]),
                    embedding: None,
                },
                embedding_dim: 0,
            },
            GraphMutation {
                scope: scope.clone(),
                table: "gke_edges".to_owned(),
                entity_kind: "edge".to_owned(),
                event_id: "semantic-cross-doc-edge-event".to_owned(),
                op: "ADD".to_owned(),
                payload: json!({
                    "id":cross_doc_edge_id,
                    "source_ids":[seed_id],
                    "target_ids":["B"]
                }),
                record: GraphRecord {
                    id: cross_doc_edge_id.to_owned(),
                    document: Some(
                        json!({
                            "id":cross_doc_edge_id,
                            "relation":"cross_doc",
                            "source_ids":[seed_id],
                            "target_ids":["B"],
                            "doc_id":"D2"
                        })
                        .to_string(),
                    ),
                    metadata: serde_json::Map::from_iter([
                        ("namespace".to_owned(), json!("tenant-a")),
                        ("workspace_id".to_owned(), json!("workspace-a")),
                        ("graph_space".to_owned(), json!("facts")),
                        ("doc_id".to_owned(), json!("D2")),
                        ("source_ids".to_owned(), json!([seed_id])),
                        ("target_ids".to_owned(), json!(["B"])),
                        ("relation".to_owned(), json!("cross_doc")),
                    ]),
                    embedding: None,
                },
                embedding_dim: 0,
            },
            GraphMutation {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                entity_kind: "node".to_owned(),
                event_id: "semantic-cross-edge-acl-event".to_owned(),
                op: "ADD".to_owned(),
                payload: json!({"id":"semantic-cross-edge-acl"}),
                record: GraphRecord {
                    id: "semantic-cross-edge-acl".to_owned(),
                    document: None,
                    metadata: serde_json::Map::from_iter([
                        ("namespace".to_owned(), json!("tenant-a")),
                        ("workspace_id".to_owned(), json!("workspace-a")),
                        ("graph_space".to_owned(), json!("facts")),
                        ("entity_type".to_owned(), json!("acl_record")),
                        ("acl_truth_graph".to_owned(), json!("knowledge")),
                        ("acl_target_grain".to_owned(), json!("edge")),
                        ("acl_target_entity_id".to_owned(), json!(cross_doc_edge_id)),
                        ("acl_version".to_owned(), json!(1)),
                        ("acl_mode".to_owned(), json!("public")),
                    ]),
                    embedding: None,
                },
                embedding_dim: 0,
            },
            GraphMutation {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                entity_kind: "node".to_owned(),
                event_id: "semantic-neighbor-acl-event".to_owned(),
                op: "ADD".to_owned(),
                payload: json!({"id":"semantic-neighbor-acl"}),
                record: GraphRecord {
                    id: "semantic-neighbor-acl".to_owned(),
                    document: None,
                    metadata: serde_json::Map::from_iter([
                        ("namespace".to_owned(), json!("tenant-a")),
                        ("workspace_id".to_owned(), json!("workspace-a")),
                        ("graph_space".to_owned(), json!("facts")),
                        ("entity_type".to_owned(), json!("acl_record")),
                        ("acl_truth_graph".to_owned(), json!("knowledge")),
                        ("acl_target_grain".to_owned(), json!("node")),
                        ("acl_target_entity_id".to_owned(), json!("B")),
                        ("acl_version".to_owned(), json!(1)),
                        ("acl_mode".to_owned(), json!("public")),
                    ]),
                    embedding: None,
                },
                embedding_dim: 0,
            },
            GraphMutation {
                scope,
                table: "gke_nodes".to_owned(),
                entity_kind: "node".to_owned(),
                event_id: "semantic-edge-acl-event".to_owned(),
                op: "ADD".to_owned(),
                payload: json!({"id":"semantic-edge-acl"}),
                record: GraphRecord {
                    id: "semantic-edge-acl".to_owned(),
                    document: None,
                    metadata: serde_json::Map::from_iter([
                        ("namespace".to_owned(), json!("tenant-a")),
                        ("workspace_id".to_owned(), json!("workspace-a")),
                        ("graph_space".to_owned(), json!("facts")),
                        ("entity_type".to_owned(), json!("acl_record")),
                        ("acl_truth_graph".to_owned(), json!("knowledge")),
                        ("acl_target_grain".to_owned(), json!("edge")),
                        ("acl_target_entity_id".to_owned(), json!(edge_id)),
                        ("acl_version".to_owned(), json!(1)),
                        ("acl_mode".to_owned(), json!("public")),
                    ]),
                    embedding: None,
                },
                embedding_dim: 0,
            },
        ])
        .await
        .unwrap();

    let wisdom_contract: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/wisdom-semantic-seed-expand-mcp-v1.json"
    )))
    .unwrap();
    let wisdom_scope = GraphScope {
        namespace: "wisdom".to_owned(),
        workspace_id: None,
        graph_space: None,
    };
    let wisdom_seed_id = wisdom_contract["expected_seed_ids"][0].as_str().unwrap();
    let wisdom_edge_id = wisdom_contract["expected_first_layer_edge_ids"][0]
        .as_str()
        .unwrap();
    let wisdom_neighbor_id = "wisdom-related-node";
    let mut wisdom_seed_metadata = serde_json::Map::new();
    wisdom_seed_metadata.insert("namespace".to_owned(), json!("wisdom"));
    wisdom_seed_metadata.insert("doc_id".to_owned(), json!("wisdom-doc"));
    let mut wisdom_neighbor_metadata = wisdom_seed_metadata.clone();
    wisdom_neighbor_metadata.insert("doc_id".to_owned(), json!("wisdom-related-doc"));
    let mut wisdom_edge_metadata = wisdom_seed_metadata.clone();
    wisdom_edge_metadata.insert("source_ids".to_owned(), json!([wisdom_seed_id]));
    wisdom_edge_metadata.insert("target_ids".to_owned(), json!([wisdom_neighbor_id]));
    wisdom_edge_metadata.insert("relation".to_owned(), json!("related_to"));
    let wisdom_acl_metadata = |grain: &str, target: &str| {
        serde_json::Map::from_iter([
            ("namespace".to_owned(), json!("wisdom")),
            ("entity_type".to_owned(), json!("acl_record")),
            ("acl_truth_graph".to_owned(), json!("wisdom")),
            ("acl_target_grain".to_owned(), json!(grain)),
            ("acl_target_entity_id".to_owned(), json!(target)),
            ("acl_version".to_owned(), json!(1)),
            ("acl_mode".to_owned(), json!("public")),
        ])
    };
    let wisdom_graph_mutations = vec![
        GraphMutation {
            scope: wisdom_scope.clone(),
            table: "gke_nodes".to_owned(),
            entity_kind: "node".to_owned(),
            event_id: "wisdom-semantic-seed-event".to_owned(),
            op: "ADD".to_owned(),
            payload: json!({"id":wisdom_seed_id,"label":"Wisdom seed"}),
            record: GraphRecord {
                id: wisdom_seed_id.to_owned(),
                document: Some(json!({"id":wisdom_seed_id,"label":"Wisdom seed"}).to_string()),
                metadata: wisdom_seed_metadata,
                embedding: Some(vec![1.0, 0.0]),
            },
            embedding_dim: 2,
        },
        GraphMutation {
            scope: wisdom_scope.clone(),
            table: "gke_nodes".to_owned(),
            entity_kind: "node".to_owned(),
            event_id: "wisdom-semantic-neighbor-event".to_owned(),
            op: "ADD".to_owned(),
            payload: json!({"id":wisdom_neighbor_id,"label":"Related"}),
            record: GraphRecord {
                id: wisdom_neighbor_id.to_owned(),
                document: Some(json!({"id":wisdom_neighbor_id,"label":"Related"}).to_string()),
                metadata: wisdom_neighbor_metadata,
                embedding: None,
            },
            embedding_dim: 0,
        },
        GraphMutation {
            scope: wisdom_scope.clone(),
            table: "gke_edges".to_owned(),
            entity_kind: "edge".to_owned(),
            event_id: "wisdom-semantic-edge-event".to_owned(),
            op: "ADD".to_owned(),
            payload: json!({"id":wisdom_edge_id,"source_ids":[wisdom_seed_id],"target_ids":[wisdom_neighbor_id]}),
            record: GraphRecord {
                id: wisdom_edge_id.to_owned(),
                document: Some(json!({"id":wisdom_edge_id,"source_ids":[wisdom_seed_id],"target_ids":[wisdom_neighbor_id]}).to_string()),
                metadata: wisdom_edge_metadata,
                embedding: None,
            },
            embedding_dim: 0,
        },
        GraphMutation {
            scope: wisdom_scope.clone(),
            table: "gke_nodes".to_owned(),
            entity_kind: "node".to_owned(),
            event_id: "wisdom-semantic-seed-acl-event".to_owned(),
            op: "ADD".to_owned(),
            payload: json!({"id":"wisdom-semantic-seed-acl"}),
            record: GraphRecord {
                id: "wisdom-semantic-seed-acl".to_owned(),
                document: None,
                metadata: wisdom_acl_metadata("node", wisdom_seed_id),
                embedding: None,
            },
            embedding_dim: 0,
        },
        GraphMutation {
            scope: wisdom_scope.clone(),
            table: "gke_nodes".to_owned(),
            entity_kind: "node".to_owned(),
            event_id: "wisdom-semantic-neighbor-acl-event".to_owned(),
            op: "ADD".to_owned(),
            payload: json!({"id":"wisdom-semantic-neighbor-acl"}),
            record: GraphRecord {
                id: "wisdom-semantic-neighbor-acl".to_owned(),
                document: None,
                metadata: wisdom_acl_metadata("node", wisdom_neighbor_id),
                embedding: None,
            },
            embedding_dim: 0,
        },
        GraphMutation {
            scope: wisdom_scope,
            table: "gke_nodes".to_owned(),
            entity_kind: "node".to_owned(),
            event_id: "wisdom-semantic-edge-acl-event".to_owned(),
            op: "ADD".to_owned(),
            payload: json!({"id":"wisdom-semantic-edge-acl"}),
            record: GraphRecord {
                id: "wisdom-semantic-edge-acl".to_owned(),
                document: None,
                metadata: wisdom_acl_metadata("edge", wisdom_edge_id),
                embedding: None,
            },
            embedding_dim: 0,
        },
    ];
    service
        .store
        .apply_graph_mutations_atomic(wisdom_graph_mutations)
        .await
        .unwrap();

    let secret = "in-memory-semantic-mcp-test-secret";
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
    let token = encode(
        &Header::new(Algorithm::HS256),
        &json!({
            "sub":"reader",
            "role":"rw",
            "ns":["docs"],
            "storage_ns":"tenant-a",
            "workspace_id":"workspace-a",
            "graph_space":"facts",
            "capabilities":["read_graph"],
            "exp":now + 300
        }),
        &EncodingKey::from_secret(secret.as_bytes()),
    )
    .unwrap();
    let app = router_with_application(
        ApiState {
            health: snapshot(),
            required_roles: vec![],
            implementation: ImplementationSnapshot::default(),
            auth,
        },
        Arc::new(service.clone()),
    );
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::from(
                    json!({
                        "jsonrpc":"2.0",
                        "id":"in-memory-semantic",
                        "method":"tools/call",
                        "params":{
                            "name":"kg_semantic_seed_then_expand_text",
                            "arguments":{
                                "text":fixture["query_text"],
                                "top_k":fixture["top_k"],
                                "hops":fixture["hops"],
                                "doc_ids":doc_id
                            }
                        }
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let response: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), 65_536).await.unwrap()).unwrap();
    assert!(response["result"]["seeds"].is_array(), "{response}");
    let seed: Value =
        serde_json::from_str(response["result"]["seeds"][0].as_str().unwrap()).unwrap();
    assert_eq!(seed["id"], seed_id);
    let first_layer_edge_ids = response["result"]["layers"][0]["edges"]
        .as_array()
        .unwrap()
        .iter()
        .map(|edge| {
            serde_json::from_str::<Value>(edge.as_str().unwrap()).unwrap()["id"]
                .as_str()
                .unwrap()
                .to_owned()
        })
        .collect::<Vec<_>>();
    assert_eq!(
        json!(first_layer_edge_ids),
        fixture["expected_first_layer_edge_ids"],
        "response={response}"
    );

    let embedding_calls_before_denial = embedding_calls.lock().unwrap().len();
    let denied_token = encode(
        &Header::new(Algorithm::HS256),
        &json!({
            "sub":"reader", "role":"ro", "ns":["wisdom"],
            "storage_ns":"wisdom", "capabilities":[], "exp":now + 300
        }),
        &EncodingKey::from_secret(secret.as_bytes()),
    )
    .unwrap();
    let denied = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {denied_token}"))
                .body(Body::from(
                    json!({
                        "jsonrpc":"2.0", "id":"wisdom-semantic-denied", "method":"tools/call",
                        "params":{"name":wisdom_contract["operation"],"arguments":wisdom_contract["arguments"]}
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    let denied: Value = serde_json::from_slice(&to_bytes(denied.into_body(), 65_536).await.unwrap())
        .unwrap();
    assert!(denied.get("error").is_some(), "{denied}");
    assert_eq!(embedding_calls.lock().unwrap().len(), embedding_calls_before_denial);

    let wisdom_token = encode(
        &Header::new(Algorithm::HS256),
        &json!({
            "sub":"reader", "role":"ro", "ns":["wisdom"],
            "storage_ns":"wisdom", "capabilities":["read_graph"], "exp":now + 300
        }),
        &EncodingKey::from_secret(secret.as_bytes()),
    )
    .unwrap();
    let wisdom_response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {wisdom_token}"))
                .body(Body::from(
                    json!({
                        "jsonrpc":"2.0", "id":"wisdom-semantic", "method":"tools/call",
                        "params":{"name":wisdom_contract["operation"],"arguments":wisdom_contract["arguments"]}
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(wisdom_response.status(), StatusCode::OK);
    let wisdom_response: Value = serde_json::from_slice(
        &to_bytes(wisdom_response.into_body(), 65_536).await.unwrap(),
    )
    .unwrap();
    assert!(wisdom_response.get("error").is_none(), "{wisdom_response}");
    let wisdom_result = &wisdom_response["result"];
    let wisdom_seed: Value =
        serde_json::from_str(wisdom_result["seeds"][0].as_str().unwrap()).unwrap();
    assert_eq!(wisdom_seed["id"], wisdom_contract["expected_seed_ids"][0]);
    let wisdom_edge_ids = wisdom_result["layers"][0]["edges"]
        .as_array()
        .unwrap()
        .iter()
        .map(|edge| {
            serde_json::from_str::<Value>(edge.as_str().unwrap()).unwrap()["id"]
                .as_str()
                .unwrap()
                .to_owned()
        })
        .collect::<Vec<_>>();
    assert_eq!(
        json!(wisdom_edge_ids),
        wisdom_contract["expected_first_layer_edge_ids"]
    );
}

// PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/tests/in_memory_semantic_retrieval.rs#in_memory_search_index_hybrid_is_scoped_and_acl_filtered; python=kogwistar/server_mcp_with_admin.py#search_index_hybrid; fixture=contracts/golden/search-index-hybrid-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/in_memory_semantic_retrieval.rs#in_memory_search_index_hybrid_is_scoped_and_acl_filtered; python_test=tests/core/test_search_index_contract.py#test_hybrid_search_fusion_matches_shared_rust_golden_fixture
#[tokio::test]
async fn in_memory_search_index_hybrid_is_scoped_and_acl_filtered() {
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
    use kogwistar_store::{
        EventReadStore, GraphMutation, GraphMutationStore, GraphRecord, NodeIndexStore, VectorMatch,
    };
    use tower::ServiceExt;

    struct FixedEmbeddingProvider;
    impl EmbeddingProvider for FixedEmbeddingProvider {
        fn profile(&self) -> &kogwistar_engine::embedding::EmbeddingProfile {
            static PROFILE: std::sync::OnceLock<kogwistar_engine::embedding::EmbeddingProfile> =
                std::sync::OnceLock::new();
            PROFILE.get_or_init(|| kogwistar_engine::embedding::EmbeddingProfile {
                provider: "test".to_owned(),
                model: "in-memory-search-index-v1".to_owned(),
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

    let fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/search-index-entry-v1.json"
    )))
    .unwrap();
    let mut request: kogwistar_contracts::SearchIndexUpsertV1 =
        serde_json::from_value(fixture["request"].clone()).unwrap();
    request.index.truncate(1);
    request.index[0].node_id = "in-memory-index-node".to_owned();
    request.index[0].canonical_title = "In-Memory Search Permit".to_owned();
    request.index[0].keywords = vec!["inmemorypermit".to_owned()];
    let mut private_entry = request.index[0].clone();
    private_entry.node_id = "in-memory-private-index-node".to_owned();
    private_entry.canonical_title = "Private In-Memory Search Permit".to_owned();
    request.index.push(private_entry);

    let hybrid_fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/search-index-hybrid-v1.json"
    )))
    .unwrap();
    let lexical = hybrid_fixture["lexical_hits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|hit| crate::search_index_fts::LexicalIndexHit {
            index_key: hit["index_key"].as_str().unwrap().to_owned(),
            node_id: hit["node_id"].as_str().unwrap().to_owned(),
            canonical_title: hit["canonical_title"].as_str().unwrap().to_owned(),
            provision: hit["provision"].as_str().unwrap().to_owned(),
            document_id: hit["document_id"].as_str().map(str::to_owned),
            bm25_score: hit["bm25_score"].as_f64().unwrap(),
        })
        .collect::<Vec<_>>();
    let vectors = hybrid_fixture["vector_hits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|hit| VectorMatch {
            record: GraphRecord {
                id: hit["id"].as_str().unwrap().to_owned(),
                document: None,
                metadata: serde_json::from_value(hit["metadata"].clone()).unwrap(),
                embedding: None,
            },
            distance: hit["distance"].as_f64().unwrap(),
        })
        .collect::<Vec<_>>();
    let fused = crate::search_index_fts::fuse_hybrid_hits(&lexical, &vectors);
    let expected_fused = hybrid_fixture["expected"].as_array().unwrap();
    assert_eq!(fused.len(), expected_fused.len());
    for (actual, expected) in fused.iter().zip(expected_fused) {
        assert_eq!(actual.index_key, expected["index_key"]);
        assert_eq!(actual.node_id, expected["node_id"]);
        assert_eq!(
            actual.hybrid_score,
            expected["hybrid_score"].as_f64().unwrap()
        );
    }

    let unconfigured = InMemoryGraphApplicationService::default();
    assert!(!unconfigured.supports_http_operation("POST", "/api/add_index_entries"));
    let unavailable = unconfigured
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/add_index_entries".to_owned(),
            body: serde_json::to_vec(&request).unwrap(),
            principal: json!({"sub":"index-writer","role":"rw","capabilities":["write_graph"]}),
        })
        .await;
    assert_eq!(unavailable.status, StatusCode::NOT_IMPLEMENTED.as_u16());
    assert!(
        EventReadStore::replay_events(&unconfigured.store, "docs", 0, 10)
            .await
            .unwrap()
            .is_empty()
    );
    assert!(!unconfigured.supports_http_operation("GET", "/api/search_index_hybrid"));
    let unavailable_query = unconfigured
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "GET".to_owned(),
            path_and_query: "/api/search_index_hybrid?q=permit".to_owned(),
            body: Vec::new(),
            principal: json!({
                "sub":"reader",
                "role":"ro",
                "ns":["docs"],
                "storage_ns":"docs",
                "capabilities":["read_graph"]
            }),
        })
        .await;
    assert_eq!(
        unavailable_query.status,
        StatusCode::NOT_IMPLEMENTED.as_u16()
    );

    let suffix = uuid::Uuid::new_v4().simple().to_string();
    let index_path = std::env::temp_dir().join(format!("search-index-memory-{suffix}.sqlite3"));
    let service = InMemoryGraphApplicationService::default()
        .with_embedding_provider(Arc::new(FixedEmbeddingProvider))
        .with_search_index_path(&index_path)
        .unwrap();
    let scope = GraphScope {
        namespace: "docs".to_owned(),
        workspace_id: None,
        graph_space: None,
    };
    let node_mutation = |id: &str, metadata: Value, document: Option<String>| GraphMutation {
        scope: scope.clone(),
        table: "gke_nodes".to_owned(),
        entity_kind: "node".to_owned(),
        event_id: format!("event-{id}"),
        op: "ADD".to_owned(),
        payload: json!({"id":id,"metadata":metadata}),
        record: GraphRecord {
            id: id.to_owned(),
            document,
            metadata: serde_json::from_value(metadata).unwrap(),
            embedding: None,
        },
        embedding_dim: 0,
    };
    let mut mutations = Vec::new();
    for (id, mode, owner) in [
        ("in-memory-index-node", "public", None),
        (
            "in-memory-private-index-node",
            "private",
            Some("other-user"),
        ),
    ] {
        mutations.push(node_mutation(
            id,
            json!({
                "entity_type":"node",
                "label":id,
                "lifecycle_status":"active"
            }),
            Some(format!("Canonical source for {id}")),
        ));
        let mut acl = json!({
            "entity_type":"acl_record",
            "acl_truth_graph":"knowledge",
            "acl_target_grain":"node",
            "acl_target_entity_id":id,
            "acl_version":1,
            "acl_mode":mode
        });
        if let Some(owner) = owner {
            acl["acl_owner_id"] = json!(owner);
        }
        mutations.push(node_mutation(&format!("acl-{id}"), acl, None));
    }
    GraphMutationStore::apply_graph_mutations_atomic(&service.store, mutations)
        .await
        .unwrap();
    assert!(service.supports_http_operation("POST", "/api/add_index_entries"));
    assert!(service.supports_http_operation("GET", "/api/search_index_hybrid"));

    let secret = "in-memory-search-index-test-secret";
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
    let token = |role: &str, capabilities: &[&str]| {
        encode(
            &Header::new(Algorithm::HS256),
            &json!({
                "sub":"index-writer",
                "role":role,
                "ns":["docs"],
                "storage_ns":"docs",
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
            required_roles: Vec::new(),
            implementation: ImplementationSnapshot::default(),
            auth,
        },
        Arc::new(service.clone()),
    );
    let body = serde_json::to_vec(&request).unwrap();

    for (role, capabilities) in [("rw", &[][..]), ("ro", &["write_graph"][..])] {
        let denied = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/add_index_entries")
                    .header(
                        "authorization",
                        format!("Bearer {}", token(role, capabilities)),
                    )
                    .header("content-type", "application/json")
                    .body(Body::from(body.clone()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(denied.status(), StatusCode::FORBIDDEN);
    }

    for _ in 0..2 {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/add_index_entries")
                    .header(
                        "authorization",
                        format!("Bearer {}", token("rw", &["write_graph"])),
                    )
                    .header("content-type", "application/json")
                    .body(Body::from(body.clone()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            serde_json::from_slice::<Value>(
                &to_bytes(response.into_body(), 65_536).await.unwrap(),
            )
            .unwrap(),
            json!({"ok":true})
        );
    }

    let events = EventReadStore::replay_events(&service.store, "docs", 0, 10)
        .await
        .unwrap();
    assert_eq!(
        events
            .iter()
            .filter(|event| event.entity_kind == "search_index" && event.op == "search_index.upsert")
            .count(),
        2,
        "an HTTP retry must not duplicate search-index events"
    );
    let fts = service.search_index_fts.as_ref().unwrap().lock().unwrap();
    let lexical_hits = crate::search_index_fts::search_fts(&fts, "permit", 10).unwrap();
    assert_eq!(
        lexical_hits.len(),
        2,
        "both entries should enter the FTS projection"
    );
    drop(fts);
    let key = kogwistar_contracts::make_search_index_key(
        &request.index[0].node_id,
        &request.index[0].canonical_title,
        &request.index[0].provision,
    );
    let vector =
        NodeIndexStore::get_node_index(&service.store, "nodes_index", &[format!("idx:{key}")])
            .await
            .unwrap();
    assert_eq!(vector.len(), 1);
    assert_eq!(
        vector[0].metadata["target_node_id"],
        request.index[0].node_id
    );

    let search = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/search_index_hybrid?q=permit&limit=10&resolve_node=true")
                .header(
                    "authorization",
                    format!("Bearer {}", token("rw", &["read_graph"])),
                )
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let http_fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/search-index-hybrid-http-v1.json"
    )))
    .unwrap();
    assert_eq!(search.status().as_u16(), http_fixture["success_status"]);
    let search_payload: Value =
        serde_json::from_slice(&to_bytes(search.into_body(), 65_536).await.unwrap()).unwrap();
    assert_eq!(search_payload["query"], http_fixture["query"]);
    for field in http_fixture["required_response_fields"].as_array().unwrap() {
        assert!(search_payload.get(field.as_str().unwrap()).is_some());
    }
    let results = search_payload["results"].as_array().unwrap();
    assert_eq!(
        results.len(),
        1,
        "private indexed nodes must be ACL-filtered: {search_payload}"
    );
    assert_eq!(results[0]["node_id"], "in-memory-index-node");
    for field in http_fixture["required_hit_fields"].as_array().unwrap() {
        assert!(results[0].get(field.as_str().unwrap()).is_some());
    }
    for field in http_fixture["resolved_hit_fields"].as_array().unwrap() {
        assert!(results[0].get(field.as_str().unwrap()).is_some());
    }
    assert_eq!(
        results[0]["documents"],
        "Canonical source for in-memory-index-node"
    );

    drop(service);
    let _ = std::fs::remove_file(index_path);
}
