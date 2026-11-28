fn assert_visualization_route_matches_fixture(payload: &Value, fixture: &Value, mode: &str) {
    let expected = &fixture["expected"];
    let d3 = payload["d3"].as_object().is_some();
    if d3 {
        let expected_node_ids = if mode == "reify" {
            &expected["reified_d3_node_order"]
        } else {
            &expected["classic_d3_node_order"]
        };
        let actual_node_ids = payload["d3"]["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|node| node["id"].as_str().unwrap().to_owned())
            .collect::<std::collections::BTreeSet<_>>();
        let expected_node_ids = expected_node_ids
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_str().unwrap().to_owned())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(actual_node_ids, expected_node_ids);

        let actual_links = payload["d3"]["links"]
            .as_array()
            .unwrap()
            .iter()
            .map(|link| {
                (
                    link["source"].as_str().unwrap().to_owned(),
                    link["target"].as_str().unwrap().to_owned(),
                    link.get("role").and_then(Value::as_str).map(str::to_owned),
                    link["relation"].as_str().unwrap().to_owned(),
                )
            })
            .collect::<std::collections::BTreeSet<_>>();
        let expected_links = if mode == "reify" {
            expected["reified_d3_links"]
                .as_array()
                .unwrap()
                .iter()
                .map(|link| {
                    (
                        link["source"].as_str().unwrap().to_owned(),
                        link["target"].as_str().unwrap().to_owned(),
                        link["role"].as_str().map(str::to_owned),
                        link["relation"].as_str().unwrap().to_owned(),
                    )
                })
                .collect()
        } else {
            expected["classic_d3_links"]
                .as_array()
                .unwrap()
                .iter()
                .map(|link| {
                    (
                        link["source"].as_str().unwrap().to_owned(),
                        link["target"].as_str().unwrap().to_owned(),
                        None,
                        link["relation"].as_str().unwrap().to_owned(),
                    )
                })
                .collect()
        };
        assert_eq!(actual_links, expected_links);
        return;
    }

    let expected_element_ids = if mode == "reify" {
        &expected["reified_cytoscape_element_order"]
    } else {
        &expected["classic_cytoscape_element_order"]
    };
    let actual_element_ids = payload["cytoscape"]["elements"]
        .as_array()
        .unwrap()
        .iter()
        .map(|element| element["data"]["id"].as_str().unwrap().to_owned())
        .collect::<std::collections::BTreeSet<_>>();
    let expected_element_ids = expected_element_ids
        .as_array()
        .unwrap()
        .iter()
        .map(|id| id.as_str().unwrap().to_owned())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(actual_element_ids, expected_element_ids);

    let actual_links = payload["cytoscape"]["elements"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|element| element["data"].get("source").is_some())
        .map(|element| {
            let data = &element["data"];
            (
                data["id"].as_str().unwrap().to_owned(),
                data["source"].as_str().unwrap().to_owned(),
                data["target"].as_str().unwrap().to_owned(),
                data["label"].as_str().unwrap().to_owned(),
                element["classes"].as_str().unwrap().to_owned(),
            )
        })
        .collect::<std::collections::BTreeSet<_>>();
    let expected_links = if mode == "reify" {
        &expected["reified_cytoscape_links"]
    } else {
        &expected["classic_cytoscape_links"]
    };
    let expected_links = expected_links
        .as_array()
        .unwrap()
        .iter()
        .map(|link| {
            (
                link["id"].as_str().unwrap().to_owned(),
                link["source"].as_str().unwrap().to_owned(),
                link["target"].as_str().unwrap().to_owned(),
                link["label"].as_str().unwrap().to_owned(),
                link["class"].as_str().unwrap().to_owned(),
            )
        })
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(actual_links, expected_links);
}

async fn assert_visualization_http_routes(
    app: &Router,
    token: &str,
    fixture: &Value,
) {
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use tower::ServiceExt;

    for mode in ["reify", "classic"] {
        for (surface, path) in [
            ("d3", "/api/viz/d3.json"),
            ("cytoscape", "/api/viz/cytoscape.json"),
        ] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("GET")
                        .uri(format!("{path}?graph_type=knowledge&mode={mode}"))
                        .header("authorization", format!("Bearer {token}"))
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK, "{surface}/{mode}");
            let result: Value = serde_json::from_slice(
                &to_bytes(response.into_body(), 65_536).await.unwrap(),
            )
            .unwrap();
            let payload = if surface == "d3" {
                json!({"d3": result})
            } else {
                json!({"cytoscape": result})
            };
            assert_visualization_route_matches_fixture(&payload, fixture, mode);
        }
    }
}

fn graph_record_mutation(
    scope: &GraphScope,
    id: &str,
    kind: &str,
    document: Option<String>,
    metadata: serde_json::Map<String, Value>,
) -> GraphMutation {
    let table = if kind == "node" {
        "gke_nodes"
    } else {
        "gke_edges"
    };
    GraphMutation {
        scope: scope.clone(),
        table: table.to_owned(),
        entity_kind: kind.to_owned(),
        event_id: format!("viz-{id}"),
        op: "ADD".to_owned(),
        payload: json!({"id": id, "metadata": metadata}),
        record: GraphRecord {
            id: id.to_owned(),
            document,
            metadata,
            embedding: None,
        },
        embedding_dim: 0,
    }
}

fn append_public_acl_rows(store: &SqliteStore, fixture: &Value) {
    for (grain, field) in [("node", "nodes"), ("edge", "edges")] {
        for entity in fixture[field].as_array().unwrap() {
            let id = entity["id"].as_str().unwrap();
            let acl_id = format!("acl-{grain}-{id}");
            let metadata = json!({
                "entity_type": "acl_record",
                "acl_truth_graph": "knowledge",
                "acl_target_grain": grain,
                "acl_target_entity_id": id,
                "acl_version": 1,
                "acl_mode": "public",
                "workspace_id": "workspace-a",
                "graph_space": "facts"
            });
            store
                .append_raw_entity_event(
                    "tenant-a",
                    SqliteNewRawEntityEvent {
                        event_id: format!("event-{acl_id}"),
                        entity_kind: "node".to_owned(),
                        entity_id: acl_id.clone(),
                        op: "ADD".to_owned(),
                        payload_json: json!({"id": acl_id, "metadata": metadata}).to_string(),
                    },
                )
                .expect("persist fixture ACL event");
        }
    }
}

fn visualization_auth(secret: &str) -> AuthConfig {
    AuthConfig {
        mode: "jwt".to_owned(),
        algorithm: Some("HS256".to_owned()),
        key: Some(secret.to_owned()),
        ..AuthConfig::default()
    }
}

fn visualization_token(secret: &str, now: u64) -> String {
    use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};

    encode(
        &Header::new(Algorithm::HS256),
        &json!({
            "sub": "reader",
            "user_id": "reader",
            "role": "ro",
            "ns": ["docs"],
            "storage_ns": "tenant-a",
            "workspace": "workspace-a",
            "workspace_id": "workspace-a",
            "graph_space": "facts",
            "capabilities": ["read_graph"],
            "exp": now + 300
        }),
        &EncodingKey::from_secret(secret.as_bytes()),
    )
    .unwrap()
}

// PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/in_memory_service/graph_operations.rs#graph_read_effect; python=kogwistar/server_mcp_with_admin.py#api_viz_d3; fixture=contracts/golden/visualization-hyperedge-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/visualization_hyperedge_routes.rs#in_memory_visualization_http_hyperedges_match_shared_fixture; python_test=tests/test_visualization.py#test_hyperedge_visualization_matches_shared_python_contract
#[tokio::test]
async fn in_memory_visualization_http_hyperedges_match_shared_fixture() {
    let fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/visualization-hyperedge-v1.json"
    )))
    .expect("visualization hyperedge fixture is valid");
    let scope = GraphScope {
        namespace: "tenant-a".to_owned(),
        workspace_id: Some("workspace-a".to_owned()),
        graph_space: Some("facts".to_owned()),
    };
    let service = Arc::new(InMemoryGraphApplicationService::default());
    let mut mutations = Vec::new();
    for (kind, field) in [("node", "nodes"), ("edge", "edges")] {
        for entity in fixture[field].as_array().unwrap() {
            let id = entity["id"].as_str().unwrap();
            let metadata = json!({"workspace_id":"workspace-a","graph_space":"facts"})
                .as_object()
                .unwrap()
                .clone();
            mutations.push(graph_record_mutation(
                &scope,
                id,
                kind,
                Some(entity.to_string()),
                metadata,
            ));
            let acl_id = format!("acl-{kind}-{id}");
            let acl_metadata = json!({
                "entity_type":"acl_record",
                "acl_truth_graph":"knowledge",
                "acl_target_grain":kind,
                "acl_target_entity_id":id,
                "acl_version":1,
                "acl_mode":"public",
                "workspace_id":"workspace-a",
                "graph_space":"facts"
            })
            .as_object()
            .unwrap()
            .clone();
            mutations.push(graph_record_mutation(
                &scope,
                &acl_id,
                "node",
                None,
                acl_metadata,
            ));
        }
    }
    service
        .store
        .apply_graph_mutations_atomic(mutations)
        .await
        .expect("seed same graph and ACL records as shared fixture");

    let secret = "in-memory-hyperedge-visualization-secret";
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let app = router_with_application(
        ApiState {
            health: snapshot(),
            required_roles: Vec::new(),
            implementation: ImplementationSnapshot::default(),
            auth: visualization_auth(secret),
        },
        service,
    );
    let token = visualization_token(secret, now);
    assert_visualization_http_routes(&app, &token, &fixture).await;
}

async fn fake_chroma_for_visualization(
    listener: tokio::net::TcpListener,
    fixture: Value,
) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    loop {
        let Ok((mut stream, _)) = listener.accept().await else {
            return;
        };
        let mut bytes = Vec::new();
        let mut chunk = [0_u8; 4096];
        let (header_end, content_length) = loop {
            let Ok(count) = stream.read(&mut chunk).await else {
                return;
            };
            if count == 0 {
                return;
            }
            bytes.extend_from_slice(&chunk[..count]);
            let Some(header_end) = bytes.windows(4).position(|window| window == b"\r\n\r\n")
            else {
                continue;
            };
            let headers = String::from_utf8_lossy(&bytes[..header_end]);
            let content_length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().ok())
                        .flatten()
                })
                .unwrap_or_default();
            if bytes.len() >= header_end + 4 + content_length {
                break (header_end, content_length);
            }
        };
        let headers = String::from_utf8_lossy(&bytes[..header_end]);
        let request_line = headers.lines().next().unwrap_or_default();
        let body = serde_json::from_slice::<Value>(
            &bytes[header_end + 4..header_end + 4 + content_length],
        )
        .unwrap_or(Value::Null);
        let response = if request_line.contains("/collections HTTP") {
            let name = body["name"].as_str().unwrap_or_default();
            json!({
                "id": format!("collection-{name}"),
                "metadata": body["metadata"]
            })
        } else if request_line.contains("/collection-nodes/get HTTP") {
            chroma_fixture_page(&fixture, "nodes")
        } else if request_line.contains("/collection-edges/get HTTP") {
            chroma_fixture_page(&fixture, "edges")
        } else {
            json!({"error":"unexpected fixture request"})
        };
        let response = response.to_string();
        let status = if response.contains("unexpected fixture request") {
            "404 Not Found"
        } else {
            "200 OK"
        };
        let head = format!(
            "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            response.len()
        );
        if stream.write_all(head.as_bytes()).await.is_err()
            || stream.write_all(response.as_bytes()).await.is_err()
        {
            return;
        }
    }
}

fn chroma_fixture_page(fixture: &Value, field: &str) -> Value {
    let records = fixture[field].as_array().unwrap();
    let metadata = json!({"workspace_id":"workspace-a","graph_space":"facts"});
    json!({
        "ids": records.iter().map(|record| record["id"].clone()).collect::<Vec<_>>(),
        "documents": records.iter().map(Value::to_string).collect::<Vec<_>>(),
        "metadatas": records.iter().map(|_| json!({
            "_kogwistar_namespace":"tenant-a",
            "_kogwistar_original_metadata":metadata.to_string()
        })).collect::<Vec<_>>(),
        "embeddings": records.iter().map(|_| Value::Null).collect::<Vec<_>>()
    })
}

// PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/sqlite_graph_reads.rs#chroma_graph_visualization_effect; python=kogwistar/server_mcp_with_admin.py#api_viz_d3; fixture=contracts/golden/visualization-hyperedge-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/visualization_hyperedge_routes.rs#sqlite_chroma_visualization_http_hyperedges_match_shared_fixture; python_test=tests/test_visualization.py#test_hyperedge_visualization_matches_shared_python_contract
#[tokio::test]
async fn sqlite_chroma_visualization_http_hyperedges_match_shared_fixture() {
    use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};

    let fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/visualization-hyperedge-v1.json"
    )))
    .expect("visualization hyperedge fixture is valid");
    let listener = bind_reachable_loopback_listener().await;
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let fake = tokio::spawn(fake_chroma_for_visualization(listener, fixture.clone()));
    let database_path = std::env::temp_dir().join(format!(
        "kogwistar-viz-hyperedge-{}.sqlite3",
        uuid::Uuid::new_v4().simple()
    ));
    let service = open_sqlite_application_service(
        &database_path,
        Some(SqliteChromaGraphConfig {
            endpoint,
            metric: DistanceMetric::Cosine,
            node_collection: "nodes".to_owned(),
            edge_collection: "edges".to_owned(),
        }),
    )
    .await
    .expect("open SQLite service with fixture Chroma projection");
    append_public_acl_rows(&service.store, &fixture);

    let secret = "sqlite-chroma-hyperedge-visualization-secret";
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let auth = AuthConfig {
        mode: "jwt".to_owned(),
        algorithm: Some("HS256".to_owned()),
        key: Some(secret.to_owned()),
        ..AuthConfig::default()
    };
    let token = encode(
        &Header::new(Algorithm::HS256),
        &json!({
            "sub":"reader",
            "user_id":"reader",
            "role":"ro",
            "ns":["docs"],
            "storage_ns":"tenant-a",
            "workspace":"workspace-a",
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
            required_roles: Vec::new(),
            implementation: ImplementationSnapshot::default(),
            auth,
        },
        Arc::new(service),
    );
    assert_visualization_http_routes(&app, &token, &fixture).await;
    fake.abort();
    drop(app);
    let _ = std::fs::remove_file(&database_path);
}
