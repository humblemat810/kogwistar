#[tokio::test]
#[ignore = "requires KOGWISTAR_TEST_PG_DSN; uses a disposable PostgreSQL schema"]
async fn postgres_visualization_read_matches_shared_python_selection_and_acl() {
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
    use tower::ServiceExt;

    let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
        .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
    let schema = format!("rust_viz_{}", uuid::Uuid::new_v4().simple());
    let knowledge_schema = format!("{schema}_knowledge");
    let conversation_schema = format!("{schema}_conversation");
    let workflow_schema = format!("{schema}_workflow");
    let wisdom_schema = format!("{schema}_wisdom");
    let service = PostgresRunApplicationService::from_dsn_with_graph_schemas(
        &dsn,
        &knowledge_schema,
        &conversation_schema,
        &workflow_schema,
        &wisdom_schema,
    )
    .unwrap();
    service.ensure_schema().await.unwrap();
    for store in [
        &service.store,
        &service.conversation_store,
        &service.workflow_store,
        &service.wisdom_store,
    ] {
        store
            .create_graph_schema(3, kogwistar_store_postgres::GraphTableNames::default())
            .await
            .unwrap();
    }

    let fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/visualization-insertion-filter-v1.json"
    )))
    .unwrap();
    let scope = GraphScope {
        namespace: "viz-test".to_owned(),
        workspace_id: None,
        graph_space: Some("facts".to_owned()),
    };
    let mut entities = fixture["nodes"].as_array().unwrap().clone();
    entities.push(json!({
        "id": "n-private",
        "references": [{"doc_id": "d1", "insertion_method": "parser"}]
    }));
    for entity in &entities {
        let id = entity["id"].as_str().unwrap();
        let metadata = Map::from_iter([
            ("entity_type".to_owned(), json!("node")),
            ("doc_id".to_owned(), fixture["doc_id"].clone()),
            ("lifecycle_status".to_owned(), json!("active")),
        ]);
        service
            .store
            .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                record: GraphRecord {
                    id: id.to_owned(),
                    document: Some(
                        json!({
                            "id": id,
                            "label": id,
                            "references": entity["references"]
                        })
                        .to_string(),
                    ),
                    metadata,
                    embedding: None,
                },
                embedding_dim: 3,
            })
            .await
            .unwrap();
        let acl = if id == "n-private" {
            json!({
                "entity_type": "acl_record",
                "acl_truth_graph": "knowledge",
                "acl_target_grain": "node",
                "acl_target_entity_id": id,
                "acl_mode": "private",
                "owner_id": "bob",
                "acl_version": 1,
                "tombstoned": false
            })
        } else {
            json!({
                "entity_type": "acl_record",
                "acl_truth_graph": "knowledge",
                "acl_target_grain": "node",
                "acl_target_entity_id": id,
                "acl_mode": "public",
                "acl_version": 1,
                "tombstoned": false
            })
        };
        service
            .store
            .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                record: GraphRecord {
                    id: format!("acl|{id}"),
                    document: Some("{}".to_owned()),
                    metadata: acl.as_object().unwrap().clone(),
                    embedding: None,
                },
                embedding_dim: 3,
            })
            .await
            .unwrap();
    }

    service
        .conversation_store
        .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
            scope: scope.clone(),
            table: "gke_nodes".to_owned(),
            record: GraphRecord {
                id: "conversation-only-node".to_owned(),
                document: Some(
                    json!({"id":"conversation-only-node","label":"conversation-only-node"})
                        .to_string(),
                ),
                metadata: Map::from_iter([
                    ("entity_type".to_owned(), json!("node")),
                    ("lifecycle_status".to_owned(), json!("active")),
                ]),
                embedding: None,
            },
            embedding_dim: 3,
        })
        .await
        .unwrap();
    service
        .conversation_store
        .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
            scope: scope.clone(),
            table: "gke_nodes".to_owned(),
            record: GraphRecord {
                id: "acl|conversation-only-node".to_owned(),
                document: Some("{}".to_owned()),
                metadata: Map::from_iter([
                    ("entity_type".to_owned(), json!("acl_record")),
                    ("acl_truth_graph".to_owned(), json!("conversation")),
                    ("acl_target_grain".to_owned(), json!("node")),
                    (
                        "acl_target_entity_id".to_owned(),
                        json!("conversation-only-node"),
                    ),
                    ("acl_mode".to_owned(), json!("public")),
                    ("acl_version".to_owned(), json!(1)),
                    ("tombstoned".to_owned(), json!(false)),
                ]),
                embedding: None,
            },
            embedding_dim: 3,
        })
        .await
        .unwrap();

    let acl_fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/graph-acl-v1.json"
    )))
    .unwrap();
    let isolation = &acl_fixture["graph_type_isolation"];
    let isolation_id = isolation["entity_id"].as_str().unwrap();
    for (store, case) in [
        (&service.store, &isolation["cases"][0]),
        (&service.conversation_store, &isolation["cases"][1]),
        (&service.workflow_store, &isolation["cases"][2]),
        (&service.wisdom_store, &isolation["cases"][3]),
    ] {
        let graph_type = case["truth_graph"].as_str().unwrap();
        store
            .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                record: GraphRecord {
                    id: isolation_id.to_owned(),
                    document: Some(json!({"id": isolation_id, "label": isolation_id}).to_string()),
                    metadata: Map::from_iter([
                        ("entity_type".to_owned(), json!("node")),
                        ("lifecycle_status".to_owned(), json!("active")),
                    ]),
                    embedding: None,
                },
                embedding_dim: 3,
            })
            .await
            .unwrap();
        let mut acl = Map::from_iter([
            ("entity_type".to_owned(), json!("acl_record")),
            ("acl_truth_graph".to_owned(), json!(graph_type)),
            ("acl_target_grain".to_owned(), json!("node")),
            ("acl_target_entity_id".to_owned(), json!(isolation_id)),
            ("acl_mode".to_owned(), case["mode"].clone()),
            ("acl_version".to_owned(), json!(1)),
            ("tombstoned".to_owned(), json!(false)),
        ]);
        if let Some(owner_id) = case.get("owner_id") {
            acl.insert("owner_id".to_owned(), owner_id.clone());
        }
        store
            .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                record: GraphRecord {
                    id: format!("acl|{isolation_id}"),
                    document: Some("{}".to_owned()),
                    metadata: acl,
                    embedding: None,
                },
                embedding_dim: 3,
            })
            .await
            .unwrap();
    }

    for case in isolation["cases"].as_array().unwrap() {
        let graph_type = case["truth_graph"].as_str().unwrap();
        let result = ApplicationService::execute(
            &service,
            ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: INTERNAL_GRAPH_VIZ_D3_ROUTE.to_owned(),
                body: serde_json::to_vec(&json!({"graph_type": graph_type})).unwrap(),
                principal: json!({
                    "sub": "alice",
                    "user_id": "alice",
                    "storage_ns": "viz-test",
                    "ns": ["viz-test"],
                    "graph_space": "facts",
                    "capabilities": ["read_graph"]
                }),
            },
        )
        .await;
        assert_eq!(result.status, StatusCode::OK.as_u16(), "{graph_type}");
        let payload: Value = serde_json::from_slice(&result.body).unwrap();
        let visible = payload["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .any(|node| node["id"] == isolation_id);
        assert_eq!(
            visible,
            case["expected_visible"].as_bool().unwrap(),
            "graph type {graph_type} must select its own graph and ACL policy"
        );
    }

    let conversation_result = ApplicationService::execute(
        &service,
        ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: INTERNAL_GRAPH_VIZ_D3_ROUTE.to_owned(),
            body: serde_json::to_vec(&json!({"graph_type":"conversation"})).unwrap(),
            principal: json!({
                "sub": "alice",
                "user_id": "alice",
                "storage_ns": "viz-test",
                "ns": ["viz-test"],
                "graph_space": "facts",
                "capabilities": ["read_graph"]
            }),
        },
    )
    .await;
    assert_eq!(conversation_result.status, StatusCode::OK.as_u16());
    let conversation_payload: Value = serde_json::from_slice(&conversation_result.body).unwrap();
    assert!(
        conversation_payload["nodes"]
            .to_string()
            .contains("conversation-only-node")
    );
    assert!(!conversation_payload["nodes"].to_string().contains("n1"));

    let response = ApplicationService::execute(
        &service,
        ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: INTERNAL_GRAPH_VIZ_D3_ROUTE.to_owned(),
            body: serde_json::to_vec(&json!({
                "doc_id": fixture["doc_id"],
                "mode": "reify",
                "insertion_method": fixture["insertion_method"],
                "graph_type": "knowledge"
            }))
            .unwrap(),
            principal: json!({
                "sub": "alice",
                "user_id": "alice",
                "storage_ns": "viz-test",
                "ns": ["viz-test"],
                "graph_space": "facts",
                "capabilities": ["read_graph"]
            }),
        },
    )
    .await;
    assert_eq!(response.status, StatusCode::OK.as_u16());
    let payload: Value = serde_json::from_slice(&response.body).unwrap();
    let mut returned_ids = payload["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|node| node["id"].as_str())
        .collect::<Vec<_>>();
    returned_ids.sort_unstable();
    assert_eq!(
        returned_ids,
        fixture["expected_matches_by_kind"]["node"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_str().unwrap())
            .collect::<Vec<_>>()
    );

    let secret = "postgres-visualization-public-route-test-secret";
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
    let token = |capabilities: &[&str]| {
        encode(
            &Header::new(Algorithm::HS256),
            &json!({
                "sub": "alice",
                "role": "rw",
                "ns": ["viz-test"],
                "storage_ns": "viz-test",
                "graph_space": "facts",
                "capabilities": capabilities,
                "exp": now + 300
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
        Arc::new(service.clone()),
    );
    for case in isolation["cases"].as_array().unwrap() {
        let graph_type = case["truth_graph"].as_str().unwrap();
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri(format!("/api/viz/d3.json?graph_type={graph_type}"))
                    .header(
                        "authorization",
                        format!("Bearer {}", token(&["read_graph"])),
                    )
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK, "{graph_type}");
        let payload: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 65_536).await.unwrap()).unwrap();
        let visible = payload["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .any(|node| node["id"] == isolation_id);
        assert_eq!(
            visible,
            case["expected_visible"].as_bool().unwrap(),
            "authenticated HTTP graph type {graph_type} must select its own ACL policy"
        );
    }

    let query = format!(
        "doc_id={}&mode=reify&insertion_method={}&graph_type=knowledge",
        fixture["doc_id"].as_str().unwrap(),
        fixture["insertion_method"].as_str().unwrap()
    );
    let mut expected_node_ids = fixture["expected_matches_by_kind"]["node"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    expected_node_ids.sort_unstable();
    for (path, result_key) in [
        ("/api/viz/d3.json", "nodes"),
        ("/api/viz/cytoscape.json", "elements"),
    ] {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri(format!("{path}?{query}"))
                    .header(
                        "authorization",
                        format!("Bearer {}", token(&["read_graph"])),
                    )
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK, "{path}");
        let payload: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 65_536).await.unwrap()).unwrap();
        let ids = payload[result_key]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|item| {
                item.get("id")
                    .and_then(Value::as_str)
                    .or_else(|| item.get("data").and_then(|data| data["id"].as_str()))
            })
            .collect::<Vec<_>>();
        let mut returned_node_ids = ids
            .iter()
            .copied()
            .filter(|id| id.starts_with('n'))
            .collect::<Vec<_>>();
        returned_node_ids.sort_unstable();
        assert_eq!(returned_node_ids, expected_node_ids, "{path}");
        assert!(!ids.contains(&"n-private"), "{path}: {ids:?}");
    }

    let bundle = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/viz/d3.bundle?{query}"))
                .header(
                    "authorization",
                    format!("Bearer {}", token(&["read_graph"])),
                )
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(bundle.status(), StatusCode::OK);
    assert_eq!(bundle.headers()["content-type"], "text/html; charset=utf-8");
    let bundle = String::from_utf8(
        to_bytes(bundle.into_body(), 1_048_576)
            .await
            .unwrap()
            .to_vec(),
    )
    .unwrap();
    let embedded = bundle
        .split_once("window.__EMBEDDED_DATA__ = ")
        .and_then(|(_, rest)| rest.split_once(';'))
        .map(|(payload, _)| serde_json::from_str::<Value>(payload).unwrap())
        .expect("public bundle contains serialized graph data");
    let bundle_ids = embedded["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|node| node["id"].as_str())
        .collect::<Vec<_>>();
    let mut returned_bundle_node_ids = bundle_ids
        .iter()
        .copied()
        .filter(|id| id.starts_with('n'))
        .collect::<Vec<_>>();
    returned_bundle_node_ids.sort_unstable();
    assert_eq!(returned_bundle_node_ids, expected_node_ids);
    assert!(!bundle_ids.contains(&"n-private"));

    for path in [
        format!("/api/viz/d3.json?{query}"),
        format!("/api/viz/cytoscape.json?{query}"),
        format!("/viz/d3.bundle?{query}"),
    ] {
        let denied = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri(path)
                    .header("authorization", format!("Bearer {}", token(&[])))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(denied.status(), StatusCode::FORBIDDEN);
    }

    drop(service);
    let (cleanup, connection) = tokio_postgres::connect(&dsn, tokio_postgres::NoTls)
        .await
        .unwrap();
    let cleanup_task = tokio::spawn(connection);
    cleanup
        .batch_execute(&format!(
            "DROP SCHEMA \"{knowledge_schema}\" CASCADE; \
             DROP SCHEMA \"{conversation_schema}\" CASCADE; \
             DROP SCHEMA \"{workflow_schema}\" CASCADE; \
             DROP SCHEMA \"{wisdom_schema}\" CASCADE"
        ))
        .await
        .unwrap();
    drop(cleanup);
    cleanup_task.await.unwrap().unwrap();
}
