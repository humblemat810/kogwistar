#[tokio::test]
async fn in_memory_vector_query_requires_exact_source_span_acl_reference() {
    let fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/graph-vector-source-acl-v1.json"
    )))
    .unwrap();
    let service = InMemoryGraphApplicationService::default();
    let scope = json!({
        "namespace": fixture["scope"]["namespace"],
        "workspace_id": fixture["scope"]["workspace_id"],
        "graph_space": null
    });
    let mut mutations = Vec::new();
    for case in fixture["cases"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        let item_id = format!("{id}::mention::0");
        let node_metadata = json!({
            "entity_type":"node",
            "label":id,
            "lifecycle_status":"active",
            "mentions":[{"spans":[{"doc_id":"source-doc"}]}]
        });
        mutations.push(json!({
            "scope":scope, "table":"gke_nodes", "entity_kind":"node",
            "event_id":format!("node-{id}"), "op":"ADD", "payload":{"id":id},
            "record":{"id":id,"document":id,"metadata":node_metadata,
                "embedding":fixture["query_embedding"]},
            "embedding_dim":2
        }));

        for (grain, target_entity, target_item) in [
            ("node", id.to_owned(), None),
            ("grounding", id.to_owned(), Some(item_id)),
            (
                "span",
                case["span_acl_entity_id"].as_str().unwrap().to_owned(),
                Some(case["span_acl_target_item_id"].as_str().unwrap().to_owned()),
            ),
        ] {
            let acl_id = format!("acl-{id}-{grain}");
            let mut acl_metadata = json!({
                "entity_type":"acl_record",
                "acl_truth_graph":fixture["truth_graph"],
                "acl_target_entity_id":target_entity,
                "acl_target_grain":grain,
                "acl_version":1,
                "acl_mode":"public"
            });
            if let Some(target_item_id) = target_item {
                acl_metadata["acl_target_item_id"] = json!(target_item_id);
            }
            mutations.push(json!({
                "scope":scope, "table":"gke_nodes", "entity_kind":"node",
                "event_id":format!("event-{acl_id}"), "op":"ADD",
                "payload":{"id":acl_id},
                "record":{"id":acl_id,"document":null,"metadata":acl_metadata,
                    "embedding":null},
                "embedding_dim":0
            }));
        }
    }
    let write = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/internal/graph/mutations".to_owned(),
            body: serde_json::to_vec(&json!({
                "contract_version":1,"atomic":true,"mutations":mutations,"index_jobs":[]
            }))
            .unwrap(),
            principal: json!({
                "sub":"writer", "role":"rw", "storage_ns":fixture["scope"]["namespace"],
                "ns":["docs"],
                "workspace_id":fixture["scope"]["workspace_id"],
                "capabilities":["write_graph"]
            }),
        })
        .await;
    assert_eq!(write.status, StatusCode::OK.as_u16());

    for case in fixture["cases"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        let response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/internal/graph/vector-query".to_owned(),
                body: serde_json::to_vec(&json!({
                    "entity_kind":"node",
                    "query_embeddings":[fixture["query_embedding"]],
                    "metric":"cosine",
                    "truth_graph":fixture["truth_graph"],
                    "where":{"label":id},
                    "n_results":1,
                    "include":["distances"]
                }))
                .unwrap(),
                principal: json!({
                    "sub":"reader", "role":"ro", "storage_ns":fixture["scope"]["namespace"],
                    "ns":["docs"],
                    "workspace_id":fixture["scope"]["workspace_id"],
                    "capabilities":["read_graph"]
                }),
            })
            .await;
        assert_eq!(response.status, StatusCode::OK.as_u16());
        let result: Value = serde_json::from_slice(&response.body).unwrap();
        let expected = if case["expected_visible"].as_bool().unwrap() {
            json!([[id]])
        } else {
            json!([[]])
        };
        assert_eq!(result["ids"], expected, "case {id}");
    }
}

#[tokio::test]
async fn in_memory_vector_query_matches_shared_python_acl_contract() {
    let fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/graph-acl-vector-query-v1.json"
    )))
    .unwrap();
    assert_eq!(fixture["schema_version"], 1);

    let secret = "in-memory-vector-acl-contract-secret";
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
    let token = |subject: &str, capabilities: &[&str], scope_parts: Option<&Value>| {
        let mut claims = json!({
            "sub": subject,
            "role": "rw",
            "ns": ["docs"],
            "storage_ns": fixture["scope"]["namespace"],
            "workspace_id": fixture["scope"]["workspace_id"],
            "capabilities": capabilities,
            "exp": now + 300
        });
        if let Some(scope_parts) = scope_parts {
            for key in ["tenant", "workspace", "project"] {
                if let Some(value) = scope_parts.get(key) {
                    claims[key] = value.clone();
                }
            }
        }
        encode(
            &Header::new(Algorithm::HS256),
            &claims,
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
        "namespace": fixture["scope"]["namespace"],
        "workspace_id": fixture["scope"]["workspace_id"],
        "graph_space": null
    });
    let mut mutations = Vec::new();
    for (kind, table, records) in [
        ("node", "gke_nodes", &fixture["nodes"]),
        ("edge", "gke_edges", &fixture["edges"]),
    ] {
        for spec in records.as_array().unwrap() {
            let id = spec["id"].as_str().unwrap();
            let mut metadata = serde_json::Map::new();
            metadata.insert("entity_type".to_owned(), json!(kind));
            if kind == "node" {
                metadata.insert("label".to_owned(), json!(id));
            } else {
                metadata.insert("source_ids".to_owned(), spec["source_ids"].clone());
                metadata.insert("target_ids".to_owned(), spec["target_ids"].clone());
            }
            if let Some(lifecycle_status) = spec["lifecycle_status"].as_str() {
                metadata.insert("lifecycle_status".to_owned(), json!(lifecycle_status));
            }
            if let Some(redirect_to_id) = spec["redirect_to_id"].as_str() {
                metadata.insert("redirect_to_id".to_owned(), json!(redirect_to_id));
            }
            mutations.push(json!({
                "scope": scope,
                "table": table,
                "entity_kind": kind,
                "event_id": format!("{kind}-{id}"),
                "op": "ADD",
                "payload": {"id": id},
                "record": {
                    "id": id,
                    "document": id,
                    "metadata": metadata,
                    "embedding": spec["embedding"]
                },
                "embedding_dim": 2
            }));

            let mut acl_metadata = serde_json::Map::new();
            for (key, value) in [
                ("entity_type", json!("acl_record")),
                ("acl_truth_graph", fixture["truth_graph"].clone()),
                ("acl_target_entity_id", json!(id)),
                ("acl_target_grain", json!(kind)),
                ("acl_version", json!(1)),
                ("acl_mode", spec["mode"].clone()),
            ] {
                acl_metadata.insert(key.to_owned(), value);
            }
            if let Some(owner_id) = spec["owner_id"].as_str() {
                acl_metadata.insert("owner_id".to_owned(), json!(owner_id));
            }
            if let Some(security_scope) = spec["security_scope"].as_str() {
                acl_metadata.insert("security_scope".to_owned(), json!(security_scope));
            }
            let acl_id = format!("acl-{kind}-{id}");
            mutations.push(json!({
                "scope": scope,
                "table": "gke_nodes",
                "entity_kind": "node",
                "event_id": format!("event-{acl_id}"),
                "op": "ADD",
                "payload": {"id": acl_id},
                "record": {
                    "id": acl_id,
                    "document": null,
                    "metadata": acl_metadata,
                    "embedding": null
                },
                "embedding_dim": 0
            }));
        }
    }
    let write = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/internal/graph/mutations")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!("Bearer {}", token("owner-b", &["write_graph"], None)),
                )
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
        for (kind, expected_key) in [("node", "nodes"), ("edge", "edges")] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/internal/graph/vector-query")
                        .header("content-type", "application/json")
                        .header(
                            "authorization",
                            format!(
                                "Bearer {}",
                                token(subject, &["read_graph"], principal.get("scope_parts"))
                            ),
                        )
                        .body(Body::from(
                            json!({
                                "entity_kind": kind,
                                "query_embeddings": fixture["query_embeddings"],
                                "metric": fixture["metric"],
                                "truth_graph": fixture["truth_graph"],
                                "n_results": fixture["n_results"],
                                "include": ["distances"]
                            })
                            .to_string(),
                        ))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let result: Value =
                serde_json::from_slice(&to_bytes(response.into_body(), 65_536).await.unwrap())
                    .unwrap();
            assert_eq!(result["implementation_authority"], "rust");
            assert_eq!(result["seam_attestation_version"], 1);
            assert_eq!(result["seam_crossings"], json!([]));
            assert_eq!(
                result["ids"], principal["expected"][expected_key],
                "unexpected {kind} candidates for {subject}"
            );
            let redirect = &fixture["current_vector_redirect_contract"][kind];
            for row in result["ids"].as_array().unwrap() {
                let ids = row.as_array().unwrap();
                assert!(!ids.contains(&redirect["source_id"]));
                assert!(ids.contains(&redirect["target_id"]));
            }
            let expected_distances = &principal["expected"]["distances"][expected_key];
            let actual_distances = result["distances"].as_array().unwrap();
            assert_eq!(
                actual_distances.len(),
                expected_distances.as_array().unwrap().len()
            );
            for (actual, expected_values) in actual_distances
                .iter()
                .zip(expected_distances.as_array().unwrap())
            {
                let actual = actual.as_array().unwrap();
                let expected_values = expected_values.as_array().unwrap();
                assert_eq!(actual.len(), expected_values.len());
                for (actual, expected) in actual.iter().zip(expected_values) {
                    assert!(
                        (actual.as_f64().unwrap() - expected.as_f64().unwrap()).abs() < 1e-5,
                        "unexpected {kind} vector distance for {subject}"
                    );
                }
            }
            if subject == "reader-a" && kind == "node" {
                let response = app
                    .clone()
                    .oneshot(
                        Request::builder()
                            .method("POST")
                            .uri("/internal/graph/vector-query")
                            .header("content-type", "application/json")
                            .header(
                                "authorization",
                                format!("Bearer {}", token(subject, &["read_graph"], None)),
                            )
                            .body(Body::from(
                                json!({
                                    "entity_kind": "node",
                                    "query_embeddings": [fixture["query_embeddings"][0]],
                                    "metric": fixture["metric"],
                                    "truth_graph": fixture["truth_graph"],
                                    "n_results": 1,
                                    "include": ["distances"]
                                })
                                .to_string(),
                            ))
                            .unwrap(),
                    )
                    .await
                    .unwrap();
                assert_eq!(response.status(), StatusCode::OK);
                let result: Value =
                    serde_json::from_slice(&to_bytes(response.into_body(), 65_536).await.unwrap())
                        .unwrap();
                assert_eq!(result["ids"], json!([[]]));
            }
        }
    }
    for (kind, expected_ids) in [
        ("node", json!([["near-private", "visible-scope"]])),
        ("edge", json!([["edge-private-endpoint", "edge-scope"]])),
    ] {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/graph/vector-query")
                    .header("content-type", "application/json")
                    .header(
                        "authorization",
                        format!("Bearer {}", token("owner-b", &["read_graph"], None)),
                    )
                    .body(Body::from(
                        json!({
                            "entity_kind": kind,
                            "query_embeddings": [fixture["query_embeddings"][0]],
                            "metric": fixture["metric"],
                            "truth_graph": fixture["truth_graph"],
                            "n_results": 2,
                            "include": ["distances"]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let result: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 65_536).await.unwrap()).unwrap();
        assert_eq!(result["ids"], expected_ids);
    }

    let revoked_node_id = "visible-public";
    let revoked_acl_id = format!("acl-node-{revoked_node_id}");
    let public_read = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/internal/graph/vector-query")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!("Bearer {}", token("reader-a", &["read_graph"], None)),
                )
                .body(Body::from(
                    json!({
                        "entity_kind": "node",
                        "query_embeddings": [fixture["query_embeddings"][0]],
                        "metric": fixture["metric"],
                        "truth_graph": fixture["truth_graph"],
                        "where": {"label": revoked_node_id},
                        "n_results": 1,
                        "include": ["distances"]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(public_read.status(), StatusCode::OK);
    let public_result: Value = serde_json::from_slice(
        &to_bytes(public_read.into_body(), 65_536).await.unwrap(),
    )
    .unwrap();
    assert_eq!(public_result["ids"], json!([[revoked_node_id]]));

    let revoke = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/internal/graph/mutations")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!("Bearer {}", token("owner-b", &["write_graph"], None)),
                )
                .body(Body::from(
                    json!({
                        "contract_version": 1,
                        "atomic": true,
                        "index_jobs": [],
                        "mutations": [{
                            "scope": scope,
                            "table": "gke_nodes",
                            "entity_kind": "node",
                            "event_id": "event-acl-node-visible-public-revoked",
                            "op": "REPLACE",
                            "payload": {"id": revoked_acl_id},
                            "record": {
                                "id": revoked_acl_id,
                                "document": null,
                                "metadata": {
                                    "entity_type": "acl_record",
                                    "acl_truth_graph": fixture["truth_graph"],
                                    "acl_target_entity_id": revoked_node_id,
                                    "acl_target_grain": "node",
                                    "acl_version": 2,
                                    "acl_mode": "private",
                                    "owner_id": "owner-b"
                                },
                                "embedding": null
                            },
                            "embedding_dim": 0
                        }]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(revoke.status(), StatusCode::OK);

    let revoked_read = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/internal/graph/vector-query")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!("Bearer {}", token("reader-a", &["read_graph"], None)),
                )
                .body(Body::from(
                    json!({
                        "entity_kind": "node",
                        "query_embeddings": [fixture["query_embeddings"][0]],
                        "metric": fixture["metric"],
                        "truth_graph": fixture["truth_graph"],
                        "where": {"label": revoked_node_id},
                        "n_results": 1,
                        "include": ["distances"]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(revoked_read.status(), StatusCode::OK);
    let revoked_result: Value = serde_json::from_slice(
        &to_bytes(revoked_read.into_body(), 65_536).await.unwrap(),
    )
    .unwrap();
    assert_eq!(revoked_result["ids"], json!([[]]));

    let item_node_id = "node-source-span-private";
    let item_grounding_id = format!("{item_node_id}::mention::0");
    let item_span_id = format!("{item_grounding_id}::span::0");
    let mut item_mutations = vec![json!({
        "scope": scope,
        "table": "gke_nodes",
        "entity_kind": "node",
        "event_id": item_node_id,
        "op": "ADD",
        "payload": {"id": item_node_id},
        "record": {
            "id": item_node_id,
            "document": item_node_id,
            "metadata": {
                "entity_type": "node",
                "label": item_node_id,
                "mentions": [{"spans": [{}]}]
            },
            "embedding": [1.0, 0.0]
        },
        "embedding_dim": 2
    })];
    for (event_id, grain, target_item_id, mode) in [
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
        let mut acl_metadata = json!({
            "entity_type": "acl_record",
            "acl_truth_graph": fixture["truth_graph"],
            "acl_target_entity_id": item_node_id,
            "acl_target_grain": grain,
            "acl_version": 1,
            "acl_mode": mode
        });
        if let Some(target_item_id) = target_item_id {
            acl_metadata["acl_target_item_id"] = json!(target_item_id);
        }
        if mode == "private" {
            acl_metadata["owner_id"] = json!("owner-b");
        }
        item_mutations.push(json!({
            "scope": scope,
            "table": "gke_nodes",
            "entity_kind": "node",
            "event_id": event_id,
            "op": "ADD",
            "payload": {"id": event_id},
            "record": {
                "id": event_id,
                "document": null,
                "metadata": acl_metadata,
                "embedding": null
            },
            "embedding_dim": 0
        }));
    }
    let write = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/internal/graph/mutations")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!("Bearer {}", token("owner-b", &["write_graph"], None)),
                )
                .body(Body::from(
                    json!({"contract_version":1,"atomic":true,"mutations":item_mutations,"index_jobs":[]})
                        .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(write.status(), StatusCode::OK);
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/internal/graph/vector-query")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!("Bearer {}", token("reader-a", &["read_graph"], None)),
                )
                .body(Body::from(
                    json!({
                        "entity_kind": "node",
                        "query_embeddings": [[1.0, 0.0]],
                        "metric": "cosine",
                        "truth_graph": fixture["truth_graph"],
                        "where": {"label": item_node_id},
                        "n_results": 1,
                        "include": ["distances"]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let result: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), 65_536).await.unwrap()).unwrap();
    assert_eq!(result["ids"], json!([[]]));
}

#[tokio::test]
async fn sqlite_chroma_vector_query_matches_shared_python_acl_contract() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/graph-acl-vector-query-v1.json"
    )))
    .unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let fixture_for_server = fixture.clone();
    let (stop_server, mut stop_server_rx) = tokio::sync::oneshot::channel();
    let chroma_server = tokio::spawn(async move {
        loop {
            let (mut stream, _) = tokio::select! {
                _ = &mut stop_server_rx => break,
                connection = listener.accept() => connection.unwrap(),
            };
            let mut bytes = Vec::new();
            let mut chunk = [0_u8; 4096];
            let (header_end, content_length) = loop {
                let count = stream.read(&mut chunk).await.unwrap();
                assert!(count > 0, "Chroma connection closed before request body");
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
            .unwrap_or_else(|_| json!({}));
            let request_line = String::from_utf8_lossy(&bytes[..header_end])
                .lines()
                .next()
                .unwrap_or_default()
                .to_owned();
            let payload = if request_line.contains("/collections HTTP") {
                let name = request["name"].as_str().unwrap();
                json!({"id":format!("collection-{name}"),"metadata":{"hnsw:space":"cosine"}})
            } else {
                let kind = if request_line.contains("shared-acl-edges") {
                    "edges"
                } else {
                    "nodes"
                };
                if request_line.contains("/query HTTP") {
                    let candidates = fixture_for_server[kind].as_array().unwrap();
                    let query = request["query_embeddings"][0].as_array().unwrap();
                    let query = [query[0].as_f64().unwrap(), query[1].as_f64().unwrap()];
                    let mut ranked = candidates
                        .iter()
                        .filter(|candidate| candidate["lifecycle_status"] != "tombstoned")
                        .map(|candidate| {
                            let embedding = candidate["embedding"].as_array().unwrap();
                            let vector = [
                                embedding[0].as_f64().unwrap(),
                                embedding[1].as_f64().unwrap(),
                            ];
                            let dot = query[0] * vector[0] + query[1] * vector[1];
                            let q_norm = query[0].hypot(query[1]);
                            let v_norm = vector[0].hypot(vector[1]);
                            (candidate, 1.0 - dot / (q_norm * v_norm))
                        })
                        .collect::<Vec<_>>();
                    ranked.sort_by(|left, right| left.1.total_cmp(&right.1));
                    ranked.truncate(request["n_results"].as_u64().unwrap() as usize);
                    let ids = ranked
                        .iter()
                        .map(|(candidate, _)| candidate["id"].clone())
                        .collect::<Vec<_>>();
                    let documents = ids.clone();
                    let embeddings = ranked
                        .iter()
                        .map(|(candidate, _)| candidate["embedding"].clone())
                        .collect::<Vec<_>>();
                    let distances = ranked.iter().map(|(_, distance)| json!(distance)).collect::<Vec<_>>();
                    let metadatas = ranked
                        .iter()
                        .map(|(candidate, _)| {
                            let mut original = json!({
                                "entity_type":if kind == "nodes" {"node"} else {"edge"},
                                "workspace_id":fixture_for_server["scope"]["workspace_id"]
                            });
                            if kind == "nodes" {
                                original["label"] = candidate["id"].clone();
                                original["mentions"] = json!([]);
                            } else {
                                for key in ["source_ids", "target_ids"] {
                                    original[key] = candidate[key].clone();
                                }
                                original["source_edge_ids"] = json!([]);
                                original["target_edge_ids"] = json!([]);
                                original["mentions"] = json!([]);
                            }
                            json!({
                                "_kogwistar_namespace":fixture_for_server["scope"]["namespace"],
                                "_kogwistar_original_metadata":original.to_string()
                            })
                        })
                        .collect::<Vec<_>>();
                    json!({
                        "ids":[ids], "documents":[documents], "metadatas":[metadatas],
                        "embeddings":[embeddings], "distances":[distances]
                    })
                } else if request_line.contains("/get HTTP") {
                    let requested = request["ids"].as_array().cloned().unwrap_or_default();
                    let matching = fixture_for_server["nodes"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .filter(|node| requested.iter().any(|id| id == &node["id"]))
                        .collect::<Vec<_>>();
                    json!({
                        "ids":matching.iter().map(|node| node["id"].clone()).collect::<Vec<_>>(),
                        "documents":matching.iter().map(|node| node["id"].clone()).collect::<Vec<_>>(),
                        "metadatas":matching.iter().map(|node| json!({
                            "_kogwistar_namespace":fixture_for_server["scope"]["namespace"],
                            "_kogwistar_original_metadata":json!({
                                "entity_type":"node", "label":node["id"], "mentions":[],
                                "workspace_id":fixture_for_server["scope"]["workspace_id"]
                            }).to_string()
                        })).collect::<Vec<_>>(),
                        "embeddings":vec![Value::Null;matching.len()]
                    })
                } else {
                    json!({"ids":[],"documents":[],"metadatas":[],"embeddings":[]})
                }
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
        chroma_config("shared-acl-nodes"),
        chroma_config("shared-acl-edges"),
    )
    .await
    .unwrap();
    let sqlite_path = std::env::temp_dir().join(format!(
        "kogwistar-sqlite-chroma-vector-contract-{}.sqlite",
        uuid::Uuid::new_v4().simple()
    ));
    let service = SqliteRunApplicationService::open(&sqlite_path)
        .unwrap()
        .with_chroma_graph(graph);
    let scope = json!({
        "namespace":fixture["scope"]["namespace"],
        "workspace_id":fixture["scope"]["workspace_id"],
        "graph_space":null
    });
    let mut acl_mutations = Vec::new();
    for (kind, records) in [("node", &fixture["nodes"]), ("edge", &fixture["edges"])] {
        for candidate in records.as_array().unwrap() {
            if candidate["lifecycle_status"] == "tombstoned" {
                continue;
            }
            let id = candidate["id"].as_str().unwrap();
            let mut metadata = json!({
                "entity_type":"acl_record", "acl_truth_graph":fixture["truth_graph"],
                "acl_target_entity_id":id, "acl_target_grain":kind,
                "acl_version":1, "acl_mode":candidate["mode"],
                "workspace_id":fixture["scope"]["workspace_id"]
            });
            for key in ["owner_id", "security_scope"] {
                if !candidate[key].is_null() {
                    metadata[key] = candidate[key].clone();
                }
            }
            let acl_id = format!("acl-{kind}-{id}");
            acl_mutations.push(json!({
                "scope":scope, "table":"gke_nodes", "entity_kind":"node",
                "event_id":format!("event-{acl_id}"), "op":"ADD",
                "payload":{"id":acl_id,"metadata":metadata},
                "record":{"id":acl_id,"document":null,"metadata":metadata,"embedding":null},
                "embedding_dim":0
            }));
        }
    }
    let seed = service.execute(ApiEffectRequest {
        contract_version: 1,
        method: "POST".to_owned(),
        path_and_query: "/internal/graph/mutations".to_owned(),
        body: serde_json::to_vec(&json!({
            "contract_version":1,"atomic":true,"mutations":acl_mutations,"index_jobs":[]
        }))
        .unwrap(),
        principal: json!({
            "sub":"fixture-owner", "role":"rw", "storage_ns":fixture["scope"]["namespace"],
            "workspace_id":fixture["scope"]["workspace_id"], "capabilities":["write_graph"]
        }),
    })
    .await;
    assert_eq!(seed.status, StatusCode::OK.as_u16());
    let visible_public_acl = service
        .store
        .immediate_transaction(|uow| {
            uow.graph_acl_records_for_entities(
                &kogwistar_store::GraphScope {
                    namespace: fixture["scope"]["namespace"].as_str().unwrap().to_owned(),
                    workspace_id: Some(
                        fixture["scope"]["workspace_id"]
                            .as_str()
                            .unwrap()
                            .to_owned(),
                    ),
                    graph_space: None,
                },
                fixture["truth_graph"].as_str().unwrap(),
                &["visible-public".to_owned()],
            )
        })
        .unwrap();
    assert_eq!(visible_public_acl.len(), 1);

    for principal_fixture in fixture["principals"].as_array().unwrap() {
        let subject = principal_fixture["id"].as_str().unwrap();
        let mut principal = json!({
            "sub":subject, "role":"ro", "storage_ns":fixture["scope"]["namespace"],
            "workspace_id":fixture["scope"]["workspace_id"], "capabilities":["read_graph"]
        });
        if let Some(scope_parts) = principal_fixture.get("scope_parts") {
            for key in ["tenant", "workspace", "project"] {
                if !scope_parts[key].is_null() {
                    principal[key] = scope_parts[key].clone();
                }
            }
        }
        for (kind, expected_key) in [("node", "nodes"), ("edge", "edges")] {
            let response = service
                .execute(ApiEffectRequest {
                    contract_version: 1,
                    method: "POST".to_owned(),
                    path_and_query: "/internal/graph/vector-query".to_owned(),
                    body: serde_json::to_vec(&json!({
                        "entity_kind":kind,
                        "query_embeddings":fixture["query_embeddings"],
                        "metric":fixture["metric"],
                        "truth_graph":fixture["truth_graph"],
                        "n_results":fixture["n_results"],
                        "include":["distances"]
                    }))
                    .unwrap(),
                    principal: principal.clone(),
                })
                .await;
            assert_eq!(response.status, StatusCode::OK.as_u16(), "{subject}/{kind}");
            let result: Value = serde_json::from_slice(&response.body).unwrap();
            assert_eq!(
                result["ids"], principal_fixture["expected"][expected_key],
                "unexpected {kind} candidates for {subject}: {result}"
            );
            let redirect = &fixture["current_vector_redirect_contract"][kind];
            for row in result["ids"].as_array().unwrap() {
                let ids = row.as_array().unwrap();
                assert!(!ids.contains(&redirect["source_id"]));
                assert!(ids.contains(&redirect["target_id"]));
            }
            for (actual_query, expected_query) in result["distances"]
                .as_array()
                .unwrap()
                .iter()
                .zip(principal_fixture["expected"]["distances"][expected_key].as_array().unwrap())
            {
                for (actual, expected) in actual_query.as_array().unwrap().iter().zip(expected_query.as_array().unwrap()) {
                    assert!(
                        (actual.as_f64().unwrap() - expected.as_f64().unwrap()).abs() < 1e-5,
                        "unexpected {kind} vector distance for {subject}"
                    );
                }
            }
        }
    }

    drop(service);
    let _ = stop_server.send(());
    chroma_server.await.unwrap();
    let _ = std::fs::remove_file(sqlite_path);
}
