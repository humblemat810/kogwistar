#[tokio::test]
async fn in_memory_mcp_traversal_matches_shared_python_contract() {
    let fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/graph-traversal-v1.json"
    )))
    .unwrap();
    assert_eq!(fixture["schema_version"], 1);

    let secret = "in-memory-traversal-contract-secret";
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
            "sub":"fixture-reader",
            "role":"rw",
            "ns":["docs"],
            "storage_ns":"fixture-space",
            "workspace":"fixture-workspace",
            "workspace_id":"fixture-workspace",
            "graph_space":"fixture-graph",
            "capabilities":["read_graph", "write_graph"],
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
        Arc::new(InMemoryGraphApplicationService::default()),
    );
    let scope = json!({
        "namespace":"fixture-space",
        "workspace_id":"fixture-workspace",
        "graph_space":"fixture-graph"
    });
    let mut mutations = Vec::new();
    for node_id in fixture["graph"]["nodes"].as_array().unwrap() {
        let id = node_id.as_str().unwrap();
        let label = fixture["graph"]["node_labels"][id].as_str().unwrap();
        mutations.push(json!({
            "scope":scope,
            "table":"gke_nodes",
            "entity_kind":"node",
            "event_id":format!("fixture-node-{id}"),
            "op":"ADD",
            "payload":{"id":id,"metadata":{"label":label,"doc_id":"D1"}},
            "record":{
                "id":id,
                "document":json!({"label":label,"doc_id":"D1"}).to_string(),
                "metadata":{"label":label,"doc_id":"D1"},
                "embedding":null
            },
            "embedding_dim":0
        }));
    }
    for edge in fixture["graph"]["edges"].as_array().unwrap() {
        let id = edge["id"].as_str().unwrap();
        let metadata = json!({
            "relation":edge["relation"],
            "source_ids":edge["source_ids"],
            "target_ids":edge["target_ids"],
            "doc_id":edge["doc_id"]
        });
        mutations.push(json!({
            "scope":scope,
            "table":"gke_edges",
            "entity_kind":"edge",
            "event_id":format!("fixture-edge-{id}"),
            "op":"ADD",
            "payload":{"id":id,"metadata":metadata},
            "record":{"id":id,"document":metadata.to_string(),"metadata":metadata,"embedding":null},
            "embedding_dim":0
        }));
    }
    for (grain, id) in [("node", "A"), ("node", "B"), ("edge", "E1")] {
        let metadata = json!({
            "entity_type":"acl_record",
            "acl_truth_graph":"knowledge",
            "acl_target_grain":grain,
            "acl_target_entity_id":id,
            "acl_version":1,
            "acl_mode":"public"
        });
        mutations.push(json!({
            "scope":scope,
            "table":"gke_nodes",
            "entity_kind":"node",
            "event_id":format!("fixture-acl-{grain}-{id}"),
            "op":"ADD",
            "payload":{"id":format!("acl-{grain}-{id}"),"metadata":metadata},
            "record":{"id":format!("acl-{grain}-{id}"),"document":null,"metadata":metadata,"embedding":null},
            "embedding_dim":0
        }));
    }
    let write = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/internal/graph/mutations")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::from(
                    json!({"contract_version":1,"atomic":true,"mutations":mutations,"index_jobs":[]})
                        .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(write.status(), StatusCode::OK);

    let cases = fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|case| (case["name"].as_str().unwrap(), case))
        .collect::<std::collections::BTreeMap<_, _>>();
    let call = |name: &str, arguments: Value| {
        let app = app.clone();
        let token = token.clone();
        let name = name.to_owned();
        async move {
            let response = app
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/mcp")
                        .header("content-type", "application/json")
                        .header("authorization", format!("Bearer {token}"))
                        .body(Body::from(
                            json!({
                                "jsonrpc":"2.0",
                                "id":name,
                                "method":"tools/call",
                                "params":{"name":name,"arguments":arguments}
                            })
                            .to_string(),
                        ))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let response: Value =
                serde_json::from_slice(&to_bytes(response.into_body(), 65_536).await.unwrap())
                    .unwrap();
            assert!(response.get("error").is_none(), "{name}: {response}");
            assert_eq!(response["result"]["implementation_authority"], "rust");
            assert_eq!(response["result"]["seam_attestation_version"], 1);
            assert_eq!(response["result"]["seam_crossings"], json!([]));
            let mut result = response["result"].clone();
            for field in [
                "implementation_authority",
                "seam_attestation_version",
                "seam_crossings",
            ] {
                result.as_object_mut().unwrap().remove(field);
            }
            result
        }
    };

    let edge_case = cases["edge_to_nodes"];
    assert_eq!(
        call(
            "kg_neighbors",
            json!({"rid":edge_case["rid"],"doc_id":"D1"})
        )
        .await,
        edge_case["expected_neighbors"]
    );
    let path_case = cases["shortest_path"];
    assert_eq!(
        call(
            "kg_shortest_path",
            json!({"src_id":path_case["src_id"],"dst_id":path_case["dst_id"],"doc_id":"D1"})
        )
        .await["path"],
        path_case["expected_path"]
    );
    let hop_case = cases["k_hop"];
    assert_eq!(
        call(
            "kg_k_hop",
            json!({"start_ids":hop_case["start_ids"],"k":hop_case["k"],"doc_id":"D1"})
        )
        .await["layers"],
        hop_case["expected_layers"]
    );
    let find_case = cases["find_edges"];
    assert_eq!(
        call(
            "kg_find_edges",
            json!({
                "relation":find_case["relation"],
                "src_label_contains":find_case["source_label_contains"],
                "tgt_label_contains":find_case["target_label_contains"],
                "doc_id":find_case["doc_id"]
            })
        )
        .await["edges"],
        find_case["expected_edge_ids"]
    );
}

#[tokio::test]
async fn in_memory_mcp_acl_traversal_matches_shared_python_contract() {
    let fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/graph-acl-traversal-v1.json"
    )))
    .unwrap();
    assert_eq!(fixture["schema_version"], 1);

    let secret = "in-memory-acl-traversal-contract-secret";
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
    let token = |subject: &str| {
        encode(
            &Header::new(Algorithm::HS256),
            &json!({
                "sub":subject,
                "role":"rw",
                "ns":["docs"],
                "storage_ns":"acl-fixture-space",
                "workspace":"acl-fixture-workspace",
                "workspace_id":"acl-fixture-workspace",
                "graph_space":"acl-fixture-graph",
                "capabilities":["read_graph", "write_graph"],
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
        Arc::new(InMemoryGraphApplicationService::default()),
    );
    let scope = json!({
        "namespace":"acl-fixture-space",
        "workspace_id":"acl-fixture-workspace",
        "graph_space":"acl-fixture-graph"
    });
    let mut mutations = Vec::new();
    let mut nodes = Vec::new();
    for node in fixture["nodes"].as_array().unwrap() {
        let id = node["id"].as_str().unwrap();
        let mut metadata = json!({
            "label": if id == "source-public" { "Public source" } else { "Private target" },
            "doc_id": fixture["doc_id"]
        });
        if let Some(owner_id) = node["owner_id"].as_str() {
            metadata["owner_id"] = json!(owner_id);
        }
        mutations.push(json!({
            "scope":scope,
            "table":"gke_nodes",
            "entity_kind":"node",
            "event_id":format!("acl-fixture-node-{id}"),
            "op":"ADD",
            "payload":{"id":id,"metadata":metadata},
            "record":{
                "id":id,
                "document":metadata.to_string(),
                "metadata":metadata,
                "embedding":null
            },
            "embedding_dim":0
        }));
        let mut acl = json!({
            "entity_type":"acl_record",
            "acl_truth_graph":fixture["truth_graph"],
            "acl_target_grain":"node",
            "acl_target_entity_id":id,
            "acl_version":1,
            "acl_mode":node["mode"]
        });
        if let Some(owner_id) = node["owner_id"].as_str() {
            acl["owner_id"] = json!(owner_id);
        }
        nodes.push((format!("acl-node-{id}"), acl));
    }
    let edge = &fixture["edge"];
    let edge_id = edge["id"].as_str().unwrap();
    let edge_metadata = json!({
        "relation":edge["relation"],
        "source_ids":edge["source_ids"],
        "target_ids":edge["target_ids"],
        "doc_id":fixture["doc_id"]
    });
    mutations.push(json!({
        "scope":scope,
        "table":"gke_edges",
        "entity_kind":"edge",
        "event_id":format!("acl-fixture-edge-{edge_id}"),
        "op":"ADD",
        "payload":{"id":edge_id,"metadata":edge_metadata},
        "record":{
            "id":edge_id,
            "document":edge_metadata.to_string(),
            "metadata":edge_metadata,
            "embedding":null
        },
        "embedding_dim":0
    }));
    let mut edge_acl = json!({
        "entity_type":"acl_record",
        "acl_truth_graph":fixture["truth_graph"],
        "acl_target_grain":"edge",
        "acl_target_entity_id":edge_id,
        "acl_version":1,
        "acl_mode":edge["mode"]
    });
    if let Some(owner_id) = edge["owner_id"].as_str() {
        edge_acl["owner_id"] = json!(owner_id);
    }
    nodes.push((format!("acl-edge-{edge_id}"), edge_acl));
    for (id, metadata) in nodes {
        mutations.push(json!({
            "scope":scope,
            "table":"gke_nodes",
            "entity_kind":"node",
            "event_id":format!("acl-fixture-record-{id}"),
            "op":"ADD",
            "payload":{"id":id,"metadata":metadata},
            "record":{"id":id,"document":null,"metadata":metadata,"embedding":null},
            "embedding_dim":0
        }));
    }
    let write = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/internal/graph/mutations")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {}", token("owner-a")))
                .body(Body::from(
                    json!({"contract_version":1,"atomic":true,"mutations":mutations,"index_jobs":[]})
                        .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(write.status(), StatusCode::OK);

    for principal in fixture["principals"].as_array().unwrap() {
        let subject = principal["id"].as_str().unwrap();
        let principal_token = token(subject);
        let call = |name: &str, arguments: Value| {
            let app = app.clone();
            let token = principal_token.clone();
            let name = name.to_owned();
            async move {
                let response = app
                    .oneshot(
                        Request::builder()
                            .method("POST")
                            .uri("/mcp")
                            .header("content-type", "application/json")
                            .header("authorization", format!("Bearer {token}"))
                            .body(Body::from(
                                json!({
                                    "jsonrpc":"2.0",
                                    "id":name,
                                    "method":"tools/call",
                                    "params":{"name":name,"arguments":arguments}
                                })
                                .to_string(),
                            ))
                            .unwrap(),
                    )
                    .await
                    .unwrap();
                assert_eq!(response.status(), StatusCode::OK);
                let response: Value =
                    serde_json::from_slice(&to_bytes(response.into_body(), 65_536).await.unwrap())
                        .unwrap();
                assert!(response.get("error").is_none(), "{name}: {response}");
                assert_eq!(response["result"]["implementation_authority"], "rust");
                assert_eq!(response["result"]["seam_attestation_version"], 1);
                assert_eq!(response["result"]["seam_crossings"], json!([]));
                let mut result = response["result"].clone();
                for field in [
                    "implementation_authority",
                    "seam_attestation_version",
                    "seam_crossings",
                ] {
                    result.as_object_mut().unwrap().remove(field);
                }
                result
            }
        };
        let expected_neighbors = &principal["expected_neighbors"];
        assert_eq!(
            call(
                "kg_neighbors",
                json!({"rid":"source-public","doc_id":fixture["doc_id"]})
            )
            .await,
            *expected_neighbors,
            "principal={subject}"
        );
        assert_eq!(
            call(
                "kg_k_hop",
                json!({"start_ids":["source-public"],"k":2,"doc_id":fixture["doc_id"]})
            )
            .await["layers"],
            principal["expected_k_hop"],
            "principal={subject}"
        );
        assert_eq!(
            call(
                "kg_shortest_path",
                json!({
                    "src_id":"source-public",
                    "dst_id":"target-private",
                    "max_depth":8,
                    "doc_id":fixture["doc_id"]
                })
            )
            .await["path"],
            principal["expected_shortest_path"],
            "principal={subject}"
        );
        assert_eq!(
            call(
                "kg_find_edges",
                json!({"relation":edge["relation"],"doc_id":fixture["doc_id"]})
            )
            .await["edges"],
            principal["expected_find_edges"],
            "principal={subject}"
        );
    }
}

// PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/tests/postgres_hybrid_search.rs#postgres_hybrid_search_is_available_over_authenticated_http; python=kogwistar/server_mcp_with_admin.py#search_index_hybrid; fixture=contracts/golden/search-index-hybrid-v1.json; rust_test=rust/crates/kogwistar-api/src/search_index_fts.rs#hybrid_fusion_matches_shared_python_golden_fixture; python_test=tests/core/test_search_index_contract.py#test_hybrid_search_fusion_matches_shared_rust_golden_fixture
