#[tokio::test]
#[ignore = "requires KOGWISTAR_TEST_PG_DSN; exercises a disposable wisdom schema"]
async fn postgres_wisdom_graph_upsert_is_scoped_atomic_and_acl_checked() {
    let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
        .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
    let schema = format!("rust_wisdom_upsert_{}", uuid::Uuid::new_v4().simple());
    let service = PostgresRunApplicationService::from_dsn(&dsn, &schema).unwrap();
    service.ensure_schema().await.unwrap();
    service
        .store
        .create_graph_schema(3, kogwistar_store_postgres::GraphTableNames::default())
        .await
        .unwrap();
    service
        .wisdom_store
        .create_graph_schema(3, kogwistar_store_postgres::GraphTableNames::default())
        .await
        .unwrap();
    let calls = Arc::new(Mutex::new(Vec::new()));
    let profile = kogwistar_engine::embedding::EmbeddingProfile {
        provider: "test".to_owned(),
        model: "wisdom-fixed-v1".to_owned(),
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
    let service = service.with_embedding_provider(Arc::new(FixedEmbeddingProvider {
        profile: profile.clone(),
        vector: vec![0.0, 0.0, 1.0],
        calls: calls.clone(),
        fail: None,
    }));
    service
        .ensure_embedding_profile_binding(&profile)
        .await
        .unwrap();
    let primary_embedding_binding = service
        .store
        .get_named_projection(
            EMBEDDING_PROFILE_NAMESPACE,
            service.embedding_storage_scope.as_ref().unwrap(),
        )
        .await
        .unwrap();
    let wisdom_embedding_binding = service
        .wisdom_store
        .get_named_projection(
            EMBEDDING_PROFILE_NAMESPACE,
            service.wisdom_embedding_storage_scope.as_ref().unwrap(),
        )
        .await
        .unwrap();
    assert!(primary_embedding_binding.is_some());
    assert!(wisdom_embedding_binding.is_some());

    let mut graph_batch: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/graph-mutation-batch-v1.json"
    )))
    .unwrap();
    let graph_node_id = "regular-semantic-node";
    let graph_node = json!({
        "id": graph_node_id,
        "label": "Rust semantic retrieval fixture",
        "type": "concept",
        "summary": "A normal graph node used to verify semantic MCP retrieval.",
        "metadata": {"namespace": "docs", "workspace_id": null, "graph_space": null, "doc_id": "semantic-seed-doc"}
    });
    graph_batch["mutations"][0]["scope"] = json!({
        "namespace": "docs", "workspace_id": null, "graph_space": null
    });
    graph_batch["mutations"][0]["event_id"] = json!("regular-semantic-event");
    graph_batch["mutations"][0]["payload"] = graph_node.clone();
    graph_batch["mutations"][0]["record"]["id"] = json!(graph_node_id);
    graph_batch["mutations"][0]["record"]["metadata"]["doc_id"] =
        json!("semantic-seed-doc");
    graph_batch["mutations"][0]["record"]["document"] =
        json!(serde_json::to_string(&graph_node).unwrap());
    graph_batch["mutations"][0]["record"]["embedding"] = json!([0.0, 0.0, 1.0]);
    graph_batch["mutations"][0]["embedding_dim"] = json!(3);
    let mut acl_mutation = graph_batch["mutations"][0].clone();
    let acl_metadata = json!({
        "entity_type": "acl_record",
        "acl_truth_graph": "knowledge",
        "acl_target_entity_id": graph_node_id,
        "acl_target_grain": "node",
        "acl_version": 1,
        "acl_mode": "private",
        "owner_id": "alice",
        "tombstoned": false
    });
    let acl_id = "regular-semantic-node-acl";
    acl_mutation["event_id"] = json!("regular-semantic-acl-event");
    acl_mutation["payload"] = json!({"id": acl_id, "metadata": acl_metadata});
    acl_mutation["record"] = json!({
        "id": acl_id,
        "document": serde_json::to_string(&json!({"id": acl_id})).unwrap(),
        "metadata": acl_metadata,
        "embedding": null
    });
    acl_mutation["embedding_dim"] = json!(0);
    graph_batch["mutations"]
        .as_array_mut()
        .unwrap()
        .push(acl_mutation);
    let graph_write = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: INTERNAL_GRAPH_MUTATIONS_ROUTE.to_owned(),
            body: serde_json::to_vec(&graph_batch).unwrap(),
            principal: json!({
                "sub": "alice", "user_id": "alice", "role": "rw",
                "ns": ["docs"], "storage_ns": "docs",
                "capabilities": ["write_graph"]
            }),
        })
        .await;
    assert_eq!(
        graph_write.status,
        StatusCode::OK.as_u16(),
        "{}",
        String::from_utf8_lossy(&graph_write.body)
    );
    let cross_document_edge_id = "regular-semantic-cross-document-edge";
    let graph_scope = GraphScope {
        namespace: "docs".to_owned(),
        workspace_id: None,
        graph_space: None,
    };
    let cross_document_edge_document = json!({
        "id": cross_document_edge_id,
        "relation": "related_to",
        "source_ids": [graph_node_id],
        "target_ids": [graph_node_id],
        "doc_id": "semantic-expansion-doc"
    });
    let cross_document_edge = GraphMutation {
        scope: graph_scope.clone(),
        table: "gke_edges".to_owned(),
        entity_kind: "edge".to_owned(),
        event_id: "regular-semantic-cross-document-edge-event".to_owned(),
        op: "ADD".to_owned(),
        payload: cross_document_edge_document.clone(),
        record: GraphRecord {
            id: cross_document_edge_id.to_owned(),
            document: Some(cross_document_edge_document.to_string()),
            metadata: serde_json::from_value(json!({
                "namespace": "docs",
                "doc_id": "semantic-expansion-doc",
                "source_ids": [graph_node_id],
                "target_ids": [graph_node_id],
                "relation": "related_to"
            }))
            .unwrap(),
            embedding: None,
        },
        embedding_dim: 0,
    };
    let cross_document_edge_acl_metadata = json!({
        "entity_type": "acl_record",
        "acl_truth_graph": "knowledge",
        "acl_target_entity_id": cross_document_edge_id,
        "acl_target_grain": "edge",
        "acl_version": 1,
        "acl_mode": "private",
        "owner_id": "alice",
        "tombstoned": false
    });
    let cross_document_edge_acl = GraphMutation {
        scope: graph_scope,
        table: "gke_nodes".to_owned(),
        entity_kind: "node".to_owned(),
        event_id: "regular-semantic-cross-document-edge-acl-event".to_owned(),
        op: "ADD".to_owned(),
        payload: json!({"id": "regular-semantic-cross-document-edge-acl", "metadata": cross_document_edge_acl_metadata}),
        record: GraphRecord {
            id: "regular-semantic-cross-document-edge-acl".to_owned(),
            document: None,
            metadata: serde_json::from_value(cross_document_edge_acl_metadata).unwrap(),
            embedding: None,
        },
        embedding_dim: 0,
    };
    service
        .store
        .apply_graph_mutations_atomic(vec![cross_document_edge, cross_document_edge_acl])
        .await
        .unwrap();
    let inserted_graph_node = service
        .store
        .graph_projection_records(GraphProjectionRead {
            scope: GraphScope {
                namespace: "docs".to_owned(),
                workspace_id: None,
                graph_space: None,
            },
            table: "gke_nodes".to_owned(),
            ids: Some(vec![graph_node_id.to_owned()]),
            metadata: MetadataFilter::default(),
            limit: 1,
        })
        .await
        .unwrap();
    assert_eq!(
        inserted_graph_node.len(),
        1,
        "graph mutation did not create projection"
    );
    let fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/python-graph-model-payloads-v1.json"
    )))
    .unwrap();
    let mut node = fixture["cases"][0]["payload"].clone();
    node["id"] = json!("wisdom-private-node");
    node["embedding"] = json!([1.0, 0.0, 0.0]);
    let mut edge = fixture["cases"][1]["payload"].clone();
    edge["id"] = json!("wisdom-private-edge");
    edge["source_ids"] = json!(["wisdom-private-node"]);
    edge["target_ids"] = json!([]);
    edge["embedding"] = json!([0.0, 1.0, 0.0]);
    let alice = json!({
        "sub": "alice", "user_id": "alice", "role": "rw",
        "ns": ["wisdom"], "storage_ns": "wisdom",
        "capabilities": ["write_graph"]
    });
    let response = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: PUBLIC_WISDOM_GRAPH_UPSERT_ROUTE.to_owned(),
            body: serde_json::to_vec(
                &json!({"inp": {"nodes": [node.clone()], "edges": [edge.clone()]}}),
            )
            .unwrap(),
            principal: alice,
        })
        .await;
    assert_eq!(
        response.status,
        StatusCode::OK.as_u16(),
        "{}",
        String::from_utf8_lossy(&response.body)
    );
    assert_eq!(
        *calls.lock().unwrap(),
        vec![vec![
            "Python node: A complete Pydantic node payload crosses the Rust persistence seam."
                .to_owned(),
            "supports: The typed edge preserves both endpoint and source-span data.".to_owned(),
        ],]
    );
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
    use tower::ServiceExt;

    let secret = "wisdom-semantic-mcp-test-secret";
    let token = encode(
        &Header::new(Algorithm::HS256),
        &json!({
            "sub": "alice", "user_id": "alice", "role": "ro", "roles": ["reader"],
            "ns": ["wisdom"], "storage_ns": "wisdom",
            "capabilities": ["read_graph"], "exp": 4102444800_u64
        }),
        &EncodingKey::from_secret(secret.as_bytes()),
    )
    .unwrap();
    let app = router_with_application(
        ApiState {
            health: snapshot(),
            required_roles: vec!["reader".to_owned()],
            implementation: ImplementationSnapshot::default(),
            auth: AuthConfig {
                mode: "jwt".to_owned(),
                algorithm: Some("HS256".to_owned()),
                key: Some(secret.to_owned()),
                ..AuthConfig::default()
            },
        },
        Arc::new(service.clone()),
    );
    let wisdom_contract: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/wisdom-semantic-seed-expand-mcp-v1.json"
    )))
    .unwrap();
    let denied_token = encode(
        &Header::new(Algorithm::HS256),
        &json!({
            "sub": "alice", "user_id": "alice", "role": "ro", "roles": ["reader"],
            "ns": ["wisdom"], "storage_ns": "wisdom",
            "capabilities": [], "exp": 4102444800_u64
        }),
        &EncodingKey::from_secret(secret.as_bytes()),
    )
    .unwrap();
    let embedding_calls_before_denied_read = calls.lock().unwrap().len();
    let denied_semantic = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {denied_token}"))
                .body(Body::from(
                    json!({
                        "jsonrpc": "2.0", "id": 6, "method": "tools/call",
                        "params": {
                            "name": wisdom_contract["operation"],
                            "arguments": wisdom_contract["arguments"]
                        }
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    let denied_semantic: Value = serde_json::from_slice(
        &to_bytes(denied_semantic.into_body(), 65_536).await.unwrap(),
    )
    .unwrap();
    assert!(denied_semantic.get("error").is_some(), "{denied_semantic}");
    assert_eq!(
        calls.lock().unwrap().len(),
        embedding_calls_before_denied_read,
        "unauthorized Wisdom retrieval must not invoke embedding"
    );
    let semantic_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::from(
                    serde_json::to_vec(&json!({
                        "jsonrpc": "2.0", "id": 7, "method": "tools/call",
                        "params": {
                            "name": wisdom_contract["operation"],
                            "arguments": wisdom_contract["arguments"]
                        }
                    }))
                    .unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(semantic_response.status(), StatusCode::OK);
    let semantic_body = to_bytes(semantic_response.into_body(), 65_536)
        .await
        .unwrap();
    let semantic: Value = serde_json::from_slice(&semantic_body).unwrap();
    assert!(semantic.get("error").is_none(), "{semantic}");
    let semantic = &semantic["result"];
    assert_eq!(semantic["seeds"].as_array().unwrap().len(), 1);
    let seed: Value = serde_json::from_str(semantic["seeds"][0].as_str().unwrap()).unwrap();
    assert_eq!(seed["id"], wisdom_contract["expected_seed_ids"][0]);
    assert_eq!(semantic["layers"].as_array().unwrap().len(), 1);
    let edge_docs = semantic["layers"][0]["edges"].as_array().unwrap();
    assert_eq!(edge_docs.len(), 1);
    let edge_doc: Value = serde_json::from_str(edge_docs[0].as_str().unwrap()).unwrap();
    assert_eq!(edge_doc["id"], wisdom_contract["expected_first_layer_edge_ids"][0]);

    let graph_semantic_response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!("Bearer {}", encode(
                        &Header::new(Algorithm::HS256),
                        &json!({
                            "sub": "alice", "user_id": "alice", "role": "ro", "roles": ["reader"],
                            "ns": ["docs"], "storage_ns": "docs",
                            "capabilities": ["read_graph"], "exp": 4102444800_u64
                        }),
                        &EncodingKey::from_secret(secret.as_bytes()),
                    ).unwrap()),
                )
                .body(Body::from(
                    serde_json::to_vec(&json!({
                        "jsonrpc": "2.0", "id": 8, "method": "tools/call",
                        "params": {
                            "name": "kg_semantic_seed_then_expand_text",
                            "arguments": {"text": "find the fixture concept by meaning", "top_k": 1, "hops": 1, "doc_ids": "semantic-seed-doc"}
                        }
                    }))
                    .unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(graph_semantic_response.status(), StatusCode::OK);
    let graph_semantic_body = to_bytes(graph_semantic_response.into_body(), 65_536)
        .await
        .unwrap();
    let graph_semantic: Value = serde_json::from_slice(&graph_semantic_body).unwrap();
    assert!(graph_semantic.get("error").is_none(), "{graph_semantic}");
    let graph_semantic = &graph_semantic["result"];
    let graph_seed = graph_semantic["seeds"].as_array().unwrap();
    assert_eq!(
        graph_seed.len(),
        1,
        "regular semantic result: {graph_semantic}"
    );
    assert_eq!(
        serde_json::from_str::<Value>(graph_seed[0].as_str().unwrap()).unwrap()["id"],
        graph_node_id
    );
    let expanded_edge_ids = graph_semantic["layers"][0]["edges"]
        .as_array()
        .unwrap()
        .iter()
        .map(|edge| {
            serde_json::from_str::<Value>(edge.as_str().unwrap()).unwrap()["id"]
                .as_str()
                .unwrap()
                .to_owned()
        })
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(expanded_edge_ids, [cross_document_edge_id.to_owned()].into());

    let wisdom_nodes = service
        .wisdom_store
        .graph_projection_records(GraphProjectionRead {
            scope: GraphScope {
                namespace: "wisdom".to_owned(),
                workspace_id: None,
                graph_space: None,
            },
            table: "gke_nodes".to_owned(),
            ids: Some(vec!["wisdom-private-node".to_owned()]),
            metadata: MetadataFilter::default(),
            limit: 1,
        })
        .await
        .unwrap();
    assert_eq!(wisdom_nodes.len(), 1);
    assert_eq!(
        wisdom_nodes[0].embedding.as_deref(),
        Some([0.0, 0.0, 1.0].as_slice())
    );
    let primary_nodes = service
        .store
        .graph_projection_records(GraphProjectionRead {
            scope: GraphScope {
                namespace: "wisdom".to_owned(),
                workspace_id: None,
                graph_space: None,
            },
            table: "gke_nodes".to_owned(),
            ids: Some(vec!["wisdom-private-node".to_owned()]),
            metadata: MetadataFilter::default(),
            limit: 1,
        })
        .await
        .unwrap();
    assert!(
        primary_nodes.is_empty(),
        "wisdom writes stay in the isolated schema"
    );

    let bob = json!({
        "sub": "bob", "user_id": "bob", "role": "rw",
        "ns": ["wisdom"], "storage_ns": "wisdom",
        "capabilities": ["write_graph"]
    });
    let overwrite = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: PUBLIC_WISDOM_GRAPH_UPSERT_ROUTE.to_owned(),
            body: serde_json::to_vec(&json!({"inp": {"nodes": [node.clone()], "edges": []}}))
                .unwrap(),
            principal: bob,
        })
        .await;
    assert_ne!(overwrite.status, StatusCode::OK.as_u16());

    let mut hidden_reference_edge = edge.clone();
    hidden_reference_edge["id"] = json!("bob-reference-edge");
    hidden_reference_edge["source_ids"] = json!(["wisdom-private-node"]);
    hidden_reference_edge["target_ids"] = json!([]);
    let hidden_reference = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: PUBLIC_WISDOM_GRAPH_UPSERT_ROUTE.to_owned(),
            body: serde_json::to_vec(
                &json!({"inp": {"nodes": [], "edges": [hidden_reference_edge]}}),
            )
            .unwrap(),
            principal: json!({
                "sub": "bob", "user_id": "bob", "role": "rw", "ns": ["wisdom"],
                "storage_ns": "wisdom", "capabilities": ["write_graph"]
            }),
        })
        .await;
    assert_ne!(hidden_reference.status, StatusCode::OK.as_u16());

    let missing_endpoint = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: PUBLIC_WISDOM_GRAPH_UPSERT_ROUTE.to_owned(),
            body: serde_json::to_vec(&json!({"inp": {"nodes": [], "edges": [{
                "id": "missing-endpoint-edge", "label": "references", "type": "relationship",
                "summary": "references a missing node", "relation": "references",
                "source_ids": ["absent-wisdom-node"], "target_ids": [],
                "source_edge_ids": [], "target_edge_ids": [], "metadata": {},
                "mentions": [], "embedding": null
            }]}}))
            .unwrap(),
            principal: json!({
                "sub": "alice", "user_id": "alice", "role": "rw", "ns": ["wisdom"],
                "storage_ns": "wisdom", "capabilities": ["write_graph"]
            }),
        })
        .await;
    assert_eq!(
        missing_endpoint.status,
        StatusCode::UNPROCESSABLE_ENTITY.as_u16()
    );

    let shared_service =
        PostgresRunApplicationService::from_dsn_with_wisdom_schema(&dsn, &schema, &schema)
            .unwrap()
            .with_embedding_provider(Arc::new(FixedEmbeddingProvider {
                profile: profile.clone(),
                vector: vec![0.0, 0.0, 1.0],
                calls: Arc::new(Mutex::new(Vec::new())),
                fail: None,
            }));
    shared_service.ensure_schema().await.unwrap();
    shared_service
        .ensure_embedding_profile_binding(&profile)
        .await
        .unwrap();
    let mut shared_node = node.clone();
    shared_node["id"] = json!("wisdom-shared-schema-node");
    let shared_write = shared_service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: PUBLIC_WISDOM_GRAPH_UPSERT_ROUTE.to_owned(),
            body: serde_json::to_vec(&json!({"inp": {"nodes": [shared_node], "edges": []}}))
                .unwrap(),
            principal: json!({
                "sub": "alice", "user_id": "alice", "role": "rw",
                "ns": ["wisdom"], "storage_ns": "wisdom",
                "capabilities": ["write_graph"]
            }),
        })
        .await;
    assert_eq!(
        shared_write.status,
        StatusCode::OK.as_u16(),
        "{}",
        String::from_utf8_lossy(&shared_write.body)
    );
    let shared_role_records = shared_service
        .store
        .graph_projection_records(GraphProjectionRead {
            scope: GraphScope {
                namespace: "wisdom".to_owned(),
                workspace_id: None,
                graph_space: None,
            },
            table: "gke_nodes".to_owned(),
            ids: Some(vec!["wisdom-shared-schema-node".to_owned()]),
            metadata: MetadataFilter::default(),
            limit: 1,
        })
        .await
        .unwrap();
    assert_eq!(shared_role_records.len(), 1);
    let other_namespace_records = shared_service
        .store
        .graph_projection_records(GraphProjectionRead {
            scope: GraphScope {
                namespace: "knowledge".to_owned(),
                workspace_id: None,
                graph_space: None,
            },
            table: "gke_nodes".to_owned(),
            ids: Some(vec!["wisdom-shared-schema-node".to_owned()]),
            metadata: MetadataFilter::default(),
            limit: 1,
        })
        .await
        .unwrap();
    assert!(other_namespace_records.is_empty());

    drop(service);
    drop(shared_service);
    let (cleanup, connection) = tokio_postgres::connect(&dsn, tokio_postgres::NoTls)
        .await
        .unwrap();
    let cleanup_task = tokio::spawn(connection);
    cleanup
        .batch_execute(&format!(
            "DROP SCHEMA \"{schema}\" CASCADE; DROP SCHEMA \"{schema}_wisdom\" CASCADE"
        ))
        .await
        .unwrap();
    drop(cleanup);
    cleanup_task.await.unwrap().unwrap();
}

#[test]
fn wisdom_mcp_tool_discovery_requires_authorized_wisdom_write_scope() {
    let service = PostgresRunApplicationService::from_dsn(
        "postgres://unused:unused@127.0.0.1:1/unused",
        "wisdom_tool_discovery",
    )
    .unwrap();
    let reader = json!({
        "role": "ro", "ns": ["wisdom"], "capabilities": ["read_graph"]
    });
    assert!(!service.supports_mcp_tool_for_principal(
        "root",
        "wisdom.kg_upsert_graph",
        &json!({"role": "rw", "ns": ["wisdom"], "capabilities": ["write_graph"]}),
    ));
    assert!(!service.supports_mcp_tool_for_principal(
        "root",
        "wisdom.semantic_seed_then_expand",
        &reader,
    ));
    assert!(!service.supports_mcp_tool_for_principal(
        "root",
        "kg_semantic_seed_then_expand_text",
        &json!({"role": "ro", "ns": ["docs"], "capabilities": ["read_graph"]}),
    ));
    let profile = kogwistar_engine::embedding::EmbeddingProfile {
        provider: "test".to_owned(),
        model: "wisdom-discovery-v1".to_owned(),
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
    let service = service.with_embedding_provider(Arc::new(FixedEmbeddingProvider {
        profile,
        vector: vec![0.0, 0.0, 1.0],
        calls: Arc::new(Mutex::new(Vec::new())),
        fail: None,
    }));
    for principal in [
        json!({"role": "rw", "ns": ["wisdom"], "capabilities": ["read_graph"]}),
        json!({"role": "ro", "ns": ["wisdom"], "capabilities": ["write_graph"]}),
        json!({"role": "rw", "ns": ["docs"], "capabilities": ["write_graph"]}),
    ] {
        assert!(!service.supports_mcp_tool_for_principal(
            "root",
            "wisdom.kg_upsert_graph",
            &principal,
        ));
    }
    let authorized = json!({
        "role": "rw", "ns": ["wisdom"], "capabilities": ["write_graph"]
    });
    assert!(
        service.supports_mcp_tool_for_principal("root", "wisdom.kg_upsert_graph", &authorized,)
    );
    assert!(!service.supports_mcp_tool_for_principal(
        "root",
        "wisdom.semantic_seed_then_expand",
        &authorized,
    ));
    assert!(service.supports_mcp_tool_for_principal(
        "root",
        "wisdom.semantic_seed_then_expand",
        &reader,
    ));
    let graph_reader = json!({
        "role": "ro", "ns": ["docs"], "storage_ns": "docs",
        "capabilities": ["read_graph"]
    });
    assert!(service.supports_mcp_tool_for_principal(
        "root",
        "kg_semantic_seed_then_expand_text",
        &graph_reader,
    ));
    assert!(!service.supports_mcp_tool_for_principal(
        "root",
        "kg_semantic_seed_then_expand_text",
        &json!({"role": "ro", "ns": ["docs"], "capabilities": []}),
    ));
    assert!(!service.supports_mcp_tool_for_principal(
        "conversation",
        "kg_semantic_seed_then_expand_text",
        &graph_reader,
    ));
    assert!(!service.supports_mcp_tool_for_principal(
        "root",
        "wisdom.semantic_seed_then_expand",
        &json!({"role": "ro", "ns": ["other"], "capabilities": ["read_graph"]}),
    ));
    assert!(!service.supports_mcp_tool_for_principal(
        "conversation",
        "wisdom.kg_upsert_graph",
        &authorized,
    ));
}

#[test]
fn graph_read_mcp_discovery_matches_namespace_and_capability_admission() {
    let service = PostgresRunApplicationService::from_dsn(
        "postgres://unused:unused@127.0.0.1:1/unused",
        "graph_read_discovery",
    )
    .unwrap();
    let tools = [
        "kg_find_edges",
        "kg_k_hop",
        "kg_neighbors",
        "kg_shortest_path",
    ];
    let authorized_reader = json!({
        "role": "ro", "ns": ["docs"], "capabilities": ["read_graph"]
    });
    for tool in tools {
        assert!(
            service.supports_mcp_tool_for_principal("root", tool, &authorized_reader),
            "authorized docs reader should discover {tool}"
        );
        assert!(!service.supports_mcp_tool_for_principal(
            "root",
            tool,
            &json!({"role": "ro", "ns": ["docs"], "capabilities": []}),
        ));
        assert!(!service.supports_mcp_tool_for_principal(
            "root",
            tool,
            &json!({"role": "ro", "ns": ["workflow"], "capabilities": ["read_graph"]}),
        ));
        assert!(!service.supports_mcp_tool_for_principal(
            "root",
            tool,
            &json!({"role": "rw", "ns": ["docs"], "capabilities": []}),
        ));
        assert!(
            !service.supports_mcp_tool_for_principal("conversation", tool, &authorized_reader,)
        );
    }
}
