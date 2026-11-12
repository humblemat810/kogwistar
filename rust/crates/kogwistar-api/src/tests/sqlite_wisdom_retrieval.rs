#[tokio::test]
async fn sqlite_chroma_wisdom_upsert_uses_canonical_events_and_acl_before_projection() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let chroma_setup = tokio::spawn(async move {
        for _ in 0..2 {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut bytes = Vec::new();
            let mut chunk = [0_u8; 2048];
            let (header_end, content_length) = loop {
                let count = stream.read(&mut chunk).await.unwrap();
                assert!(count > 0);
                bytes.extend_from_slice(&chunk[..count]);
                if let Some(header_end) = bytes.windows(4).position(|window| window == b"\r\n\r\n")
                {
                    let headers = String::from_utf8_lossy(&bytes[..header_end]);
                    let content_length = headers
                        .lines()
                        .find_map(|line| {
                            let (name, value) = line.split_once(':')?;
                            name.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse::<usize>().ok())
                                .flatten()
                        })
                        .unwrap_or(0);
                    if bytes.len() >= header_end + 4 + content_length {
                        break (header_end, content_length);
                    }
                }
            };
            let request: Value =
                serde_json::from_slice(&bytes[header_end + 4..header_end + 4 + content_length])
                    .unwrap();
            let response = json!({
                "id": format!("collection-{}", request["name"].as_str().unwrap()),
                "metadata": {"hnsw:space": "cosine"}
            })
            .to_string();
            let wire = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                response.len(),
                response
            );
            stream.write_all(wire.as_bytes()).await.unwrap();
        }
    });
    let chroma_config = |collection: &str| kogwistar_store_chroma::ChromaStoreConfig {
        endpoint: endpoint.clone(),
        collection: collection.to_owned(),
        ..Default::default()
    };
    let graph = kogwistar_store_chroma::ChromaGraphStore::connect_entity_collections(
        chroma_config("wisdom-nodes"),
        chroma_config("wisdom-edges"),
    )
    .await
    .unwrap();
    chroma_setup.await.unwrap();

    let sqlite_path = std::env::temp_dir().join(format!(
        "kogwistar-sqlite-chroma-wisdom-{}.sqlite",
        uuid::Uuid::new_v4()
    ));
    let provider = Arc::new(FixedEmbeddingProvider {
        profile: kogwistar_engine::embedding::EmbeddingProfile {
            provider: "test".to_owned(),
            model: "fixed-v1".to_owned(),
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
        vector: vec![0.6, 0.8],
        calls: Arc::new(Mutex::new(Vec::new())),
        fail: None,
    });
    let service = SqliteRunApplicationService::open(&sqlite_path)
        .unwrap()
        .with_chroma_graph(graph)
        .with_embedding_provider(provider);
    let mcp_contract: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/wisdom-graph-upsert-mcp-v1.json"
    )))
    .unwrap();
    let mcp_secret = "mcp-wisdom-contract-test-secret";
    let mcp_token = encode(
        &Header::new(Algorithm::HS256),
        &json!({
            "sub": "alice", "user_id": "alice", "role": "rw", "roles": ["writer"],
            "ns": ["wisdom"], "storage_ns": "wisdom", "graph_space": "wisdom",
            "capabilities": ["write_graph"], "exp": 4102444800_u64
        }),
        &EncodingKey::from_secret(mcp_secret.as_bytes()),
    )
    .unwrap();
    let mcp_app = router_with_application(
        ApiState {
            health: snapshot(),
            required_roles: vec!["writer".to_owned()],
            implementation: ImplementationSnapshot::default(),
            auth: AuthConfig {
                mode: "jwt".to_owned(),
                algorithm: Some("HS256".to_owned()),
                key: Some(mcp_secret.to_owned()),
                ..AuthConfig::default()
            },
        },
        Arc::new(service.clone()),
    );
    let mcp_response = mcp_app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp")
                .header("content-type", "application/json")
                .header("accept", "application/json, text/event-stream")
                .header("authorization", format!("Bearer {mcp_token}"))
                .body(Body::from(
                    serde_json::to_vec(&json!({
                        "jsonrpc": "2.0", "id": "wisdom-upsert-contract",
                        "method": "tools/call",
                        "params": {
                            "name": mcp_contract["operation"],
                            "arguments": mcp_contract["arguments"]
                        }
                    }))
                    .unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(mcp_response.status(), StatusCode::OK);
    assert_eq!(
        mcp_response.headers()["content-type"],
        "text/event-stream; charset=utf-8"
    );
    let mcp_body = axum::body::to_bytes(mcp_response.into_body(), 65_536)
        .await
        .unwrap();
    let mcp_wire = String::from_utf8(mcp_body.to_vec()).unwrap();
    let mcp_json = mcp_wire
        .lines()
        .find_map(|line| line.strip_prefix("data: "))
        .expect("Streamable HTTP wraps the JSON-RPC response in an SSE data field");
    let mcp_payload: Value = serde_json::from_str(mcp_json).unwrap();
    assert!(mcp_payload.get("error").is_none(), "{mcp_payload}");
    assert_eq!(
        mcp_payload["result"]["structuredContent"]["node_ids"],
        mcp_contract["expected_structured_content"]["node_ids"]
    );
    assert_eq!(
        mcp_payload["result"]["structuredContent"]["edge_ids"],
        mcp_contract["expected_structured_content"]["edge_ids"]
    );
    assert_eq!(
        mcp_payload["result"]["structuredContent"]["nodes_added"],
        mcp_contract["expected_structured_content"]["nodes_added"]
    );
    assert_eq!(
        mcp_payload["result"]["structuredContent"]["edges_added"],
        mcp_contract["expected_structured_content"]["edges_added"]
    );
    assert!(service.supports_mcp_tool("root", "wisdom.kg_upsert_graph"));
    assert!(service.supports_http_operation("POST", PUBLIC_WISDOM_GRAPH_UPSERT_ROUTE));

    let fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/python-graph-model-payloads-v1.json"
    )))
    .unwrap();
    let mut node = fixture["cases"][0]["payload"].clone();
    node["id"] = json!("sqlite-wisdom-node");
    node["embedding"] = Value::Null;
    let mut edge = fixture["cases"][1]["payload"].clone();
    edge["id"] = json!("sqlite-wisdom-edge");
    edge["source_ids"] = json!(["sqlite-wisdom-node"]);
    edge["target_ids"] = json!([]);
    edge["embedding"] = Value::Null;
    let body = serde_json::to_vec(&json!({"inp": {"nodes": [node], "edges": [edge]}})).unwrap();
    let principal = json!({
        "sub": "alice", "user_id": "alice", "role": "rw",
        "ns": ["wisdom"], "storage_ns": "wisdom", "graph_space": "wisdom",
        "capabilities": ["write_graph"]
    });
    assert!(service.supports_mcp_tool_for_principal("root", "wisdom.kg_upsert_graph", &principal,));
    for denied_principal in [
        json!({"sub":"alice", "role":"ro", "ns":["wisdom"], "capabilities":["write_graph"]}),
        json!({"sub":"alice", "role":"rw", "ns":["docs"], "capabilities":["write_graph"]}),
        json!({"sub":"alice", "role":"rw", "ns":["wisdom"], "capabilities":["read_graph"]}),
    ] {
        assert!(!service.supports_mcp_tool_for_principal(
            "root",
            "wisdom.kg_upsert_graph",
            &denied_principal,
        ));
    }
    assert!(!service.supports_mcp_tool_for_principal(
        "workflow",
        "wisdom.kg_upsert_graph",
        &principal,
    ));
    let denied = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: PUBLIC_WISDOM_GRAPH_UPSERT_ROUTE.to_owned(),
            body: body.clone(),
            principal: json!({"sub": "alice", "role": "ro", "ns": ["wisdom"]}),
        })
        .await;
    assert_eq!(denied.status, StatusCode::FORBIDDEN.as_u16());
    assert!(
        service
            .store
            .replay_raw_events("wisdom", 0, 100)
            .unwrap()
            .is_empty()
    );

    let response = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: PUBLIC_WISDOM_GRAPH_UPSERT_ROUTE.to_owned(),
            body,
            principal,
        })
        .await;
    assert_eq!(
        response.status,
        StatusCode::OK.as_u16(),
        "{}",
        String::from_utf8_lossy(&response.body)
    );
    let payload: Value = serde_json::from_slice(&response.body).unwrap();
    assert_eq!(payload["projection_consistency"], "eventual");
    assert_eq!(payload["node_ids"][0], "sqlite-wisdom-node");
    assert_eq!(payload["edge_ids"][0], "sqlite-wisdom-edge");

    let events = service.store.replay_raw_events("wisdom", 0, 100).unwrap();
    assert!(
        events
            .iter()
            .any(|event| event.entity_id == "sqlite-wisdom-node")
    );
    assert!(
        events
            .iter()
            .any(|event| event.entity_id == "sqlite-wisdom-edge")
    );
    assert!(
        events.iter().any(|event| {
            serde_json::from_str::<Value>(&event.payload_json)
                .ok()
                .and_then(|payload| payload.get("metadata").cloned())
                .and_then(|metadata| metadata.get("entity_type").cloned())
                == Some(json!("acl_record"))
        }),
        "host-derived default ACL records are part of the canonical event transaction"
    );
    let jobs = service
        .store
        .list_index_jobs(Some("wisdom"), Some("PENDING"), None, None, None, 100)
        .unwrap();
    assert!(jobs.iter().any(|job| job.entity_id == "sqlite-wisdom-node"));
    assert!(jobs.iter().any(|job| job.entity_id == "sqlite-wisdom-edge"));
    assert!(jobs.iter().all(|job| job.namespace == "wisdom"));

    let committed_event_count = events.len();
    let mut private_reference = fixture["cases"][1]["payload"].clone();
    private_reference["id"] = json!("bob-reference-edge");
    private_reference["source_ids"] = json!(["sqlite-wisdom-node"]);
    private_reference["target_ids"] = json!([]);
    private_reference["embedding"] = Value::Null;
    let hidden_reference = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: PUBLIC_WISDOM_GRAPH_UPSERT_ROUTE.to_owned(),
            body: serde_json::to_vec(&json!({"inp": {"nodes": [], "edges": [private_reference]}}))
                .unwrap(),
            principal: json!({
                "sub": "bob", "user_id": "bob", "role": "rw",
                "ns": ["wisdom"], "storage_ns": "wisdom", "graph_space": "wisdom",
                "capabilities": ["write_graph"]
            }),
        })
        .await;
    assert_eq!(hidden_reference.status, StatusCode::CONFLICT.as_u16());

    let mut missing_reference = fixture["cases"][1]["payload"].clone();
    missing_reference["id"] = json!("missing-reference-edge");
    missing_reference["source_ids"] = json!(["absent-wisdom-node"]);
    missing_reference["target_ids"] = json!([]);
    missing_reference["embedding"] = Value::Null;
    let missing_reference = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: PUBLIC_WISDOM_GRAPH_UPSERT_ROUTE.to_owned(),
            body: serde_json::to_vec(&json!({"inp": {"nodes": [], "edges": [missing_reference]}}))
                .unwrap(),
            principal: json!({
                "sub": "alice", "user_id": "alice", "role": "rw",
                "ns": ["wisdom"], "storage_ns": "wisdom", "graph_space": "wisdom",
                "workspace": "other-workspace", "capabilities": ["write_graph"]
            }),
        })
        .await;
    assert_eq!(missing_reference.status, StatusCode::CONFLICT.as_u16());
    assert_eq!(
        service
            .store
            .replay_raw_events("wisdom", 0, 100)
            .unwrap()
            .len(),
        committed_event_count
    );

    drop(service);
    remove_test_sqlite(&sqlite_path);
}

#[tokio::test]
async fn sqlite_chroma_wisdom_semantic_retrieval_requires_scope_and_embedding() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let seed_id = "wisdom-semantic-seed".to_owned();
    let neighbor_id = "wisdom-semantic-neighbor".to_owned();
    let edge_id = "wisdom-semantic-edge".to_owned();

    struct FixedEmbeddingProvider {
        profile: kogwistar_engine::embedding::EmbeddingProfile,
        vector: Vec<f64>,
        calls: Arc<Mutex<Vec<String>>>,
    }
    impl kogwistar_engine::embedding::EmbeddingProvider for FixedEmbeddingProvider {
        fn profile(&self) -> &kogwistar_engine::embedding::EmbeddingProfile {
            &self.profile
        }

        fn embed<'a>(
            &'a self,
            texts: &'a [String],
        ) -> kogwistar_engine::embedding::EmbeddingFuture<'a> {
            Box::pin(async move {
                self.calls.lock().unwrap().extend(texts.iter().cloned());
                Ok(texts.iter().map(|_| self.vector.clone()).collect())
            })
        }
    }

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let queries = Arc::new(Mutex::new(Vec::<Value>::new()));
    let queries_for_server = queries.clone();
    let server = tokio::spawn(async move {
        for _ in 0..5 {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut bytes = Vec::new();
            let mut chunk = [0_u8; 4096];
            let (header_end, content_length) = loop {
                let count = stream.read(&mut chunk).await.unwrap();
                assert!(count > 0);
                bytes.extend_from_slice(&chunk[..count]);
                if let Some(header_end) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
                    let headers = String::from_utf8_lossy(&bytes[..header_end]);
                    let content_length = headers
                        .lines()
                        .find_map(|line| {
                            let (name, value) = line.split_once(':')?;
                            name.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse::<usize>().ok())
                                .flatten()
                        })
                        .unwrap_or(0);
                    if bytes.len() >= header_end + 4 + content_length {
                        break (header_end, content_length);
                    }
                }
            };
            let request: Value = serde_json::from_slice(
                &bytes[header_end + 4..header_end + 4 + content_length],
            )
            .unwrap();
            let request_line = String::from_utf8_lossy(&bytes[..header_end])
                .lines()
                .next()
                .unwrap_or_default()
                .to_owned();
            let payload = if request_line.contains("/collections HTTP") {
                json!({
                    "id":format!("collection-{}", request["name"].as_str().unwrap()),
                    "metadata":{"hnsw:space":"cosine"}
                })
            } else if request_line.contains("/query HTTP") {
                queries_for_server.lock().unwrap().push(request);
                let metadata = json!({"entity_type":"node","namespace":"wisdom","label":"Seed","lifecycle_status":"active"});
                json!({
                    "ids":[["wisdom-semantic-seed"]],
                    "documents":[[json!({"id":"wisdom-semantic-seed","label":"Seed"}).to_string()]],
                    "metadatas":[[{
                        "_kogwistar_namespace":"wisdom",
                        "_kogwistar_original_metadata":metadata.to_string()
                    }]],
                    "embeddings":[[[0.6,0.8]]],
                    "distances":[[0.0]]
                })
            } else if request_line.contains("wisdom-semantic-edges") {
                let metadata = json!({
                    "entity_type":"edge", "namespace":"wisdom", "lifecycle_status":"active",
                    "source_ids":["wisdom-semantic-seed"],
                    "target_ids":["wisdom-semantic-neighbor"], "relation":"related_to"
                });
                json!({
                    "ids":["wisdom-semantic-edge"],
                    "documents":[json!({
                        "id":"wisdom-semantic-edge",
                        "source_ids":["wisdom-semantic-seed"],
                        "target_ids":["wisdom-semantic-neighbor"]
                    }).to_string()],
                    "metadatas":[{
                        "_kogwistar_namespace":"wisdom",
                        "_kogwistar_original_metadata":metadata.to_string()
                    }],
                    "embeddings":[null]
                })
            } else {
                let seed_metadata = json!({"entity_type":"node","namespace":"wisdom","label":"Seed","lifecycle_status":"active"});
                let neighbor_metadata = json!({"entity_type":"node","namespace":"wisdom","label":"Neighbor","lifecycle_status":"active"});
                json!({
                    "ids":["wisdom-semantic-seed","wisdom-semantic-neighbor"],
                    "documents":[
                        json!({"id":"wisdom-semantic-seed","label":"Seed"}).to_string(),
                        json!({"id":"wisdom-semantic-neighbor","label":"Neighbor"}).to_string()
                    ],
                    "metadatas":[
                        {"_kogwistar_namespace":"wisdom","_kogwistar_original_metadata":seed_metadata.to_string()},
                        {"_kogwistar_namespace":"wisdom","_kogwistar_original_metadata":neighbor_metadata.to_string()}
                    ],
                    "embeddings":[[0.6,0.8],null]
                })
            };
            let response = payload.to_string();
            let wire = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                response.len(), response
            );
            stream.write_all(wire.as_bytes()).await.unwrap();
        }
    });
    let chroma_config = |collection: &str| kogwistar_store_chroma::ChromaStoreConfig {
        endpoint: endpoint.clone(),
        collection: collection.to_owned(),
        ..Default::default()
    };
    let graph = kogwistar_store_chroma::ChromaGraphStore::connect_entity_collections(
        chroma_config("wisdom-semantic-nodes"),
        chroma_config("wisdom-semantic-edges"),
    )
    .await
    .unwrap();
    let sqlite_path = std::env::temp_dir().join(format!(
        "kogwistar-sqlite-chroma-wisdom-semantic-{}.sqlite",
        uuid::Uuid::new_v4().simple()
    ));
    let embedding_calls = Arc::new(Mutex::new(Vec::<String>::new()));
    let provider = Arc::new(FixedEmbeddingProvider {
        profile: kogwistar_engine::embedding::EmbeddingProfile {
            provider: "test".to_owned(),
            model: "fixed-v1".to_owned(),
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
        vector: vec![0.6, 0.8],
        calls: embedding_calls.clone(),
    });
    let service = SqliteRunApplicationService::open(&sqlite_path)
        .unwrap()
        .with_chroma_graph(graph)
        .with_embedding_provider(provider);
    let scope = GraphScope {
        namespace: "wisdom".to_owned(),
        workspace_id: None,
        graph_space: None,
    };
    let mutation = |event_id: &str,
                    table: &str,
                    entity_kind: &str,
                    id: &str,
                    document: Option<String>,
                    metadata: serde_json::Map<String, Value>,
                    embedding: Option<Vec<f32>>,
                    embedding_dim: usize| {
        let mut payload = json!({
            "id":id,
            "label":metadata
                .get("label")
                .cloned()
                .unwrap_or_else(|| json!(id)),
            "type":if entity_kind == "edge" { "relationship" } else { "entity" },
            "summary":"Wisdom semantic retrieval fixture",
            "doc_id":"wisdom-test-doc",
            "domain_id":null,
            "properties":null,
            "metadata":metadata.clone(),
            "mentions":[{
                "multimodal_spans":[],
                "spans":[{
                    "collection_page_url":"document_collection/wisdom-test-doc",
                    "document_page_url":"document/wisdom-test-doc",
                    "doc_id":"wisdom-test-doc",
                    "insertion_method":"contract-test",
                    "start_char":0,
                    "end_char":1,
                    "excerpt":id
                }]
            }]
        });
        if entity_kind == "edge" {
            payload["source_ids"] = metadata
                .get("source_ids")
                .cloned()
                .unwrap_or_else(|| json!([]));
            payload["target_ids"] = metadata
                .get("target_ids")
                .cloned()
                .unwrap_or_else(|| json!([]));
            payload["relation"] = metadata
                .get("relation")
                .cloned()
                .unwrap_or_else(|| json!("related_to"));
            payload["source_edge_ids"] = json!([]);
            payload["target_edge_ids"] = json!([]);
        }
        GraphMutation {
            scope: scope.clone(),
            table: table.to_owned(),
            entity_kind: entity_kind.to_owned(),
            event_id: event_id.to_owned(),
            op: "ADD".to_owned(),
            payload,
            record: GraphRecord {
            id: id.to_owned(),
            document,
            metadata,
            embedding,
            },
            embedding_dim,
        }
    };
    let mut seed_metadata = serde_json::Map::new();
    seed_metadata.insert("namespace".to_owned(), json!("wisdom"));
    seed_metadata.insert("label".to_owned(), json!("Seed"));
    seed_metadata.insert("lifecycle_status".to_owned(), json!("active"));
    let mut neighbor_metadata = serde_json::Map::new();
    neighbor_metadata.insert("namespace".to_owned(), json!("wisdom"));
    neighbor_metadata.insert("label".to_owned(), json!("Neighbor"));
    neighbor_metadata.insert("lifecycle_status".to_owned(), json!("active"));
    let mut edge_metadata = serde_json::Map::new();
    edge_metadata.insert("namespace".to_owned(), json!("wisdom"));
    edge_metadata.insert("lifecycle_status".to_owned(), json!("active"));
    edge_metadata.insert("source_ids".to_owned(), json!([seed_id]));
    edge_metadata.insert("target_ids".to_owned(), json!([neighbor_id]));
    edge_metadata.insert("relation".to_owned(), json!("related_to"));
    let acl_metadata = |grain: &str, target: &str| {
        serde_json::Map::from_iter([
            ("entity_type".to_owned(), json!("acl_record")),
            ("namespace".to_owned(), json!("wisdom")),
            ("acl_truth_graph".to_owned(), json!("wisdom")),
            ("acl_target_grain".to_owned(), json!(grain)),
            ("acl_target_entity_id".to_owned(), json!(target)),
            ("acl_version".to_owned(), json!(1)),
            ("acl_mode".to_owned(), json!("public")),
        ])
    };
    service
        .store
        .apply_canonical_graph_mutation_batch(kogwistar_store::GraphMutationBatch {
            mutations: vec![
            mutation(
                "wisdom-semantic-seed-event",
                "gke_nodes",
                "node",
                &seed_id,
                Some(json!({"id":seed_id,"label":"Seed"}).to_string()),
                seed_metadata,
                Some(vec![0.6_f32, 0.8_f32]),
                2,
            ),
            mutation(
                "wisdom-semantic-neighbor-event",
                "gke_nodes",
                "node",
                &neighbor_id,
                Some(json!({"id":neighbor_id,"label":"Neighbor"}).to_string()),
                neighbor_metadata,
                None,
                0,
            ),
            mutation(
                "wisdom-semantic-edge-event",
                "gke_edges",
                "edge",
                &edge_id,
                Some(json!({"id":edge_id,"source_ids":[seed_id],"target_ids":[neighbor_id]}).to_string()),
                edge_metadata,
                None,
                0,
            ),
            mutation(
                "wisdom-semantic-seed-acl-event",
                "gke_nodes",
                "node",
                "acl-seed",
                None,
                acl_metadata("node", &seed_id),
                None,
                0,
            ),
            mutation(
                "wisdom-semantic-neighbor-acl-event",
                "gke_nodes",
                "node",
                "acl-neighbor",
                None,
                acl_metadata("node", &neighbor_id),
                None,
                0,
            ),
            mutation(
                "wisdom-semantic-edge-acl-event",
                "gke_nodes",
                "node",
                "acl-edge",
                None,
                acl_metadata("edge", &edge_id),
                None,
                0,
            ),
            ],
            index_jobs: Vec::new(),
        })
        .unwrap();
    let authorized = json!({
        "sub":"reader", "role":"ro", "ns":["wisdom"],
        "storage_ns":"wisdom", "capabilities":["read_graph"]
    });
    assert!(service.supports_mcp_tool("root", "wisdom.semantic_seed_then_expand"));
    assert!(service.supports_mcp_tool_for_principal(
        "root",
        "wisdom.semantic_seed_then_expand",
        &authorized,
    ));
    assert!(service.supports_http_operation("POST", WISDOM_SEMANTIC_SEED_EXPAND_ROUTE));
    assert!(!service.supports_mcp_tool_for_principal(
        "root",
        "wisdom.semantic_seed_then_expand",
        &json!({"role":"ro", "ns":["wisdom"], "capabilities":[]}),
    ));

    let request_body = serde_json::to_vec(&json!({"text":"a relevant memory"})).unwrap();
    let denied = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: WISDOM_SEMANTIC_SEED_EXPAND_ROUTE.to_owned(),
            body: request_body.clone(),
            principal: json!({"role":"ro", "ns":["wisdom"], "storage_ns":"wisdom", "capabilities":[]}),
        })
        .await;
    assert_eq!(denied.status, StatusCode::FORBIDDEN.as_u16());
    assert!(embedding_calls.lock().unwrap().is_empty());
    assert!(queries.lock().unwrap().is_empty());

    let response = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: WISDOM_SEMANTIC_SEED_EXPAND_ROUTE.to_owned(),
            body: request_body,
            principal: authorized,
        })
        .await;
    assert_eq!(response.status, StatusCode::OK.as_u16());
    let payload: Value = serde_json::from_slice(&response.body).unwrap();
    let seed_json = payload["seeds"][0]
        .as_str()
        .unwrap_or_else(|| panic!("expected Chroma seed in response: {payload}"));
    let seed: Value = serde_json::from_str(seed_json).unwrap();
    assert_eq!(seed["id"], seed_id);
    assert_eq!(payload["layers"].as_array().unwrap().len(), 2);
    let first_layer_edges = payload["layers"][0]["edges"]
        .as_array()
        .unwrap()
        .iter()
        .map(|edge| serde_json::from_str::<Value>(edge.as_str().unwrap()).unwrap()["id"].clone())
        .collect::<Vec<_>>();
    assert_eq!(first_layer_edges, [json!(edge_id)]);
    assert_eq!(embedding_calls.lock().unwrap().as_slice(), ["a relevant memory"]);
    let queries = queries.lock().unwrap();
    assert_eq!(queries.len(), 1);
    assert_eq!(queries[0]["n_results"], 10);

    drop(service);
    server.await.unwrap();
    remove_test_sqlite(&sqlite_path);
}

#[tokio::test]
async fn sqlite_chroma_public_graph_upsert_uses_shared_extraction_contract() {
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tower::ServiceExt;

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let setup = tokio::spawn(async move {
        for _ in 0..2 {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut bytes = Vec::new();
            let mut chunk = [0_u8; 2048];
            let (header_end, content_length) = loop {
                let count = stream.read(&mut chunk).await.unwrap();
                assert!(count > 0);
                bytes.extend_from_slice(&chunk[..count]);
                if let Some(header_end) = bytes.windows(4).position(|window| window == b"\r\n\r\n")
                {
                    let headers = String::from_utf8_lossy(&bytes[..header_end]);
                    let content_length = headers
                        .lines()
                        .find_map(|line| {
                            let (name, value) = line.split_once(':')?;
                            name.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse::<usize>().ok())
                                .flatten()
                        })
                        .unwrap_or(0);
                    if bytes.len() >= header_end + 4 + content_length {
                        break (header_end, content_length);
                    }
                }
            };
            let request: Value =
                serde_json::from_slice(&bytes[header_end + 4..header_end + 4 + content_length])
                    .unwrap();
            let response = json!({
                "id": format!("collection-{}", request["name"].as_str().unwrap()),
                "metadata": {"hnsw:space": "cosine"}
            })
            .to_string();
            let wire = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                response.len(),
                response
            );
            stream.write_all(wire.as_bytes()).await.unwrap();
        }
    });
    let chroma_config = |collection: &str| kogwistar_store_chroma::ChromaStoreConfig {
        endpoint: endpoint.clone(),
        collection: collection.to_owned(),
        ..Default::default()
    };
    let graph = kogwistar_store_chroma::ChromaGraphStore::connect_entity_collections(
        chroma_config("knowledge-nodes"),
        chroma_config("knowledge-edges"),
    )
    .await
    .unwrap();
    setup.await.unwrap();

    let sqlite_path = std::env::temp_dir().join(format!(
        "kogwistar-sqlite-chroma-public-upsert-{}.sqlite",
        uuid::Uuid::new_v4()
    ));
    let service = SqliteRunApplicationService::open(&sqlite_path)
        .unwrap()
        .with_chroma_graph(graph)
        .with_graph_role("conversation")
        .unwrap();
    let standalone_path = std::env::temp_dir().join(format!(
        "kogwistar-sqlite-standalone-public-upsert-{}.sqlite",
        uuid::Uuid::new_v4()
    ));
    let standalone = SqliteRunApplicationService::open(&standalone_path).unwrap();
    assert_eq!(standalone.graph_role, "knowledge");
    assert!(!standalone.supports_http_operation("POST", PUBLIC_GRAPH_UPSERT_ROUTE));
    assert!(!standalone.supports_http_operation("POST", PUBLIC_DOCUMENT_TREE_UPSERT_ROUTE));
    assert!(service.supports_http_operation("POST", PUBLIC_GRAPH_UPSERT_ROUTE));
    assert!(service.supports_http_operation("POST", PUBLIC_DOCUMENT_TREE_UPSERT_ROUTE));

    let mut extraction: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/graph-extraction-v1.json"
    )))
    .unwrap();
    extraction["nodes"][0]["metadata"]["engine_type"] = json!("attacker-selected");
    let body = serde_json::to_vec(&extraction).unwrap();
    let principal = json!({
        "sub": "alice", "user_id": "alice", "role": "rw",
        "ns": ["docs"], "storage_ns": "docs", "capabilities": ["write_graph"]
    });
    for denied_principal in [
        json!({
            "sub": "alice", "user_id": "alice", "role": "ro",
            "ns": ["docs"], "storage_ns": "docs", "capabilities": ["write_graph"]
        }),
        json!({
            "sub": "alice", "user_id": "alice", "role": "rw",
            "ns": ["docs"], "storage_ns": "docs", "capabilities": ["read_graph"]
        }),
        json!({
            "role": "rw", "ns": ["docs"], "storage_ns": "docs",
            "capabilities": ["write_graph"]
        }),
    ] {
        let denied = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: PUBLIC_GRAPH_UPSERT_ROUTE.to_owned(),
                body: body.clone(),
                principal: denied_principal,
            })
            .await;
        assert_eq!(denied.status, StatusCode::FORBIDDEN.as_u16());
        assert!(
            service
                .store
                .replay_raw_events("docs", 0, 100)
                .unwrap()
                .is_empty()
        );
    }
    let secret = "sqlite-chroma-public-upsert-test-secret";
    let token = encode(
        &Header::new(Algorithm::HS256),
        &json!({
            "sub": "alice", "user_id": "alice", "role": "rw",
            "roles": ["reader", "rw"], "ns": ["docs"],
            "storage_ns": "docs", "capabilities": ["write_graph"],
            "exp": 4102444800_u64
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
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(PUBLIC_GRAPH_UPSERT_ROUTE)
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::from(body.clone()))
                .unwrap(),
        )
        .await;
    let response = response.unwrap();
    let response_status = response.status();
    let response_body = to_bytes(response.into_body(), 65_536).await.unwrap();
    assert_eq!(
        response_status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&response_body)
    );
    let payload: Value = serde_json::from_slice(&response_body).unwrap();
    let response_contract: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/graph-upsert-success-v1.json"
    )))
    .unwrap();
    assert_eq!(
        response_status.as_u16(),
        response_contract["http_status"].as_u64().unwrap() as u16
    );
    let expected_body = &response_contract["body"];
    assert_eq!(payload["document_id"], expected_body["document_id"]);
    assert_eq!(
        payload["node_ids"].as_array().unwrap().len(),
        expected_body["node_ids_count"].as_u64().unwrap() as usize
    );
    assert_eq!(
        payload["edge_ids"].as_array().unwrap().len(),
        expected_body["edge_ids_count"].as_u64().unwrap() as usize
    );
    assert_eq!(payload["nodes_added"], expected_body["nodes_added"]);
    assert_eq!(payload["edges_added"], expected_body["edges_added"]);

    let mut legacy_nodes = extraction["nodes"].as_array().unwrap().clone();
    let mut legacy_edges = extraction["edges"].as_array().unwrap().clone();
    for entity in legacy_nodes.iter_mut().chain(legacy_edges.iter_mut()) {
        let object = entity.as_object_mut().unwrap();
        if !object.contains_key("id") {
            if let Some(local_id) = object.get("local_id").cloned() {
                object.insert("id".to_owned(), local_id);
            }
        }
        object.remove("local_id");
    }
    let legacy_tree = json!({
        "doc_id": extraction["doc_id"],
        "insertion_method": extraction["insertion_method"],
        "nodes": legacy_nodes,
        "edges": legacy_edges,
    });
    let legacy_tree_response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(PUBLIC_DOCUMENT_TREE_UPSERT_ROUTE)
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::from(serde_json::to_vec(&legacy_tree).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    let legacy_tree_status = legacy_tree_response.status();
    let legacy_tree_body = to_bytes(legacy_tree_response.into_body(), 65_536)
        .await
        .unwrap();
    assert_eq!(
        legacy_tree_status,
        StatusCode::OK,
        "legacy tree upsert failed: {}",
        String::from_utf8_lossy(&legacy_tree_body)
    );
    let legacy_result: Value = serde_json::from_slice(&legacy_tree_body).unwrap();
    assert_eq!(legacy_result["status"], "ok");
    assert_eq!(legacy_result["inserted_nodes"], 2);
    assert_eq!(legacy_result["inserted_edges"], 1);
    assert_eq!(legacy_result["engine_result"]["document_id"], "doc-1");
    let legacy_node_ids = legacy_result["engine_result"]["node_ids"]
        .as_array()
        .unwrap();
    let legacy_edge_ids = legacy_result["engine_result"]["edge_ids"]
        .as_array()
        .unwrap();
    assert_eq!(legacy_node_ids.len(), 2);
    assert_eq!(legacy_edge_ids.len(), 1);

    let events = service.store.replay_raw_events("docs", 0, 100).unwrap();
    for id in legacy_node_ids.iter().chain(legacy_edge_ids) {
        assert!(events.iter().any(|event| event.entity_id == *id));
    }
    assert!(events.iter().any(|event| {
        event.entity_id == payload["node_ids"][0]
            && serde_json::from_str::<Value>(&event.payload_json)
                .ok()
                .and_then(|record| record.get("metadata").cloned())
                .and_then(|metadata| metadata.get("engine_type").cloned())
                == Some(json!("conversation"))
    }));
    assert!(events.iter().any(|event| {
        serde_json::from_str::<Value>(&event.payload_json)
            .ok()
            .and_then(|record| record.get("metadata").cloned())
            .and_then(|metadata| metadata.get("entity_type").cloned())
            == Some(json!("acl_record"))
    }));
    assert!(
        events.iter().any(|event| {
            serde_json::from_str::<Value>(&event.payload_json)
                .ok()
                .and_then(|record| record.get("metadata").cloned())
                .and_then(|metadata| metadata.get("acl_truth_graph").cloned())
                == Some(json!("conversation"))
        }),
        "expected ACL event for configured conversation graph role, got {:?}",
        events
            .iter()
            .map(|event| &event.payload_json)
            .collect::<Vec<_>>()
    );
    let document = service
        .store
        .source_document_projection(
            &GraphScope {
                namespace: "docs".to_owned(),
                workspace_id: Some(String::new()),
                graph_space: Some(String::new()),
            },
            "doc-1",
        )
        .unwrap();
    assert_eq!(
        document.and_then(|record| record.document),
        Some("Alpha connects to Beta.".to_owned())
    );
    let index_jobs = service
        .store
        .list_index_jobs(Some("docs"), Some("PENDING"), None, None, None, 100)
        .unwrap();
    assert!(
        index_jobs
            .iter()
            .any(|job| job.entity_id == payload["node_ids"][0])
    );
    assert!(
        index_jobs
            .iter()
            .any(|job| job.entity_id == payload["node_ids"][1])
    );
    assert!(index_jobs.iter().all(|job| job.namespace == "docs"));

    let committed_events = events.len();
    let mut hidden_edge = serde_json::from_slice::<Value>(&body).unwrap()["edges"][0].clone();
    hidden_edge["id"] = json!("bob-reference-edge");
    hidden_edge["local_id"] = json!("ne:bob-reference-edge");
    hidden_edge["source_ids"] = json!([payload["node_ids"][0]]);
    hidden_edge["target_ids"] = json!([]);
    let mut hidden_request = serde_json::from_slice::<Value>(&body).unwrap();
    hidden_request["nodes"] = json!([]);
    hidden_request["edges"] = json!([hidden_edge]);
    let hidden_reference = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: PUBLIC_GRAPH_UPSERT_ROUTE.to_owned(),
            body: serde_json::to_vec(&hidden_request).unwrap(),
            principal: json!({
                "sub": "bob", "user_id": "bob", "role": "rw",
                "ns": ["docs"], "storage_ns": "docs", "capabilities": ["write_graph"]
            }),
        })
        .await;
    assert_eq!(hidden_reference.status, StatusCode::CONFLICT.as_u16());
    assert_eq!(
        service
            .store
            .replay_raw_events("docs", 0, 100)
            .unwrap()
            .len(),
        committed_events,
        "cross-owner endpoint denial must not append graph or ACL events"
    );

    let document_before_invalid = service
        .store
        .source_document_projection(
            &GraphScope {
                namespace: "docs".to_owned(),
                workspace_id: Some(String::new()),
                graph_space: Some(String::new()),
            },
            "doc-1",
        )
        .unwrap()
        .and_then(|record| record.document);
    let jobs_before_invalid = service
        .store
        .list_index_jobs(Some("docs"), Some("PENDING"), None, None, None, 100)
        .unwrap()
        .into_iter()
        .map(|job| (job.entity_id, job.status))
        .collect::<Vec<_>>();
    let mut invalid: Value = serde_json::from_slice(&body).unwrap();
    invalid["nodes"][0]["references"][0]["excerpt"] = json!("forged");
    let rejected = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: PUBLIC_GRAPH_UPSERT_ROUTE.to_owned(),
            body: serde_json::to_vec(&invalid).unwrap(),
            principal,
        })
        .await;
    assert_ne!(rejected.status, StatusCode::OK.as_u16());
    assert_eq!(
        service
            .store
            .replay_raw_events("docs", 0, 100)
            .unwrap()
            .len(),
        committed_events
    );
    let document_after_invalid = service
        .store
        .source_document_projection(
            &GraphScope {
                namespace: "docs".to_owned(),
                workspace_id: Some(String::new()),
                graph_space: Some(String::new()),
            },
            "doc-1",
        )
        .unwrap()
        .and_then(|record| record.document);
    assert_eq!(document_after_invalid, document_before_invalid);
    let jobs_after_invalid = service
        .store
        .list_index_jobs(Some("docs"), Some("PENDING"), None, None, None, 100)
        .unwrap()
        .into_iter()
        .map(|job| (job.entity_id, job.status))
        .collect::<Vec<_>>();
    assert_eq!(jobs_after_invalid, jobs_before_invalid);

    let malformed_hyperedge: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/graph-extraction-invalid-hyperedge-v1.json"
    )))
    .unwrap();
    let rejected_hyperedge = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: PUBLIC_GRAPH_UPSERT_ROUTE.to_owned(),
            body: serde_json::to_vec(&malformed_hyperedge).unwrap(),
            principal: json!({
                "sub": "alice", "user_id": "alice", "role": "rw",
                "ns": ["docs"], "storage_ns": "docs", "capabilities": ["write_graph"]
            }),
        })
        .await;
    assert_eq!(
        rejected_hyperedge.status,
        StatusCode::UNPROCESSABLE_ENTITY.as_u16()
    );
    assert_eq!(
        service.store.replay_raw_events("docs", 0, 100).unwrap().len(),
        committed_events,
        "malformed hyperedge must not append graph or ACL events"
    );
    let document_after_hyperedge_rejection = service
        .store
        .source_document_projection(
            &GraphScope {
                namespace: "docs".to_owned(),
                workspace_id: Some(String::new()),
                graph_space: Some(String::new()),
            },
            "doc-invalid-hyperedge",
        )
        .unwrap();
    assert!(document_after_hyperedge_rejection.is_none());
    let jobs_after_hyperedge_rejection = service
        .store
        .list_index_jobs(Some("docs"), Some("PENDING"), None, None, None, 100)
        .unwrap()
        .into_iter()
        .map(|job| (job.entity_id, job.status))
        .collect::<Vec<_>>();
    assert_eq!(jobs_after_hyperedge_rejection, jobs_before_invalid);

    drop(service);
    drop(standalone);
    remove_test_sqlite(&sqlite_path);
    remove_test_sqlite(&standalone_path);
}
