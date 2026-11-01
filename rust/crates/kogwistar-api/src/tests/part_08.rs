#[tokio::test]
async fn in_memory_backend_routes_authenticated_graph_write_and_mcp_read() {
    let secret = "in-memory-api-contract-secret";
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
    let token = |namespace: &str, capabilities: &[&str]| {
        encode(
            &Header::new(Algorithm::HS256),
            &json!({
                "sub": "writer",
                "role": "rw",
                "ns": [namespace],
                "storage_ns": "tenant-a",
                "workspace": "workspace-a",
                "workspace_id": "workspace-a",
                "graph_space": "facts",
                "capabilities": capabilities,
                "exp": now + 300
            }),
            &EncodingKey::from_secret(secret.as_bytes()),
        )
        .unwrap()
    };
    let make_app = || {
        router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: vec![],
                implementation: ImplementationSnapshot::default(),
                auth: auth.clone(),
            },
            Arc::new(InMemoryGraphApplicationService::default()),
        )
    };
    let app = make_app();
    let scope = json!({
        "namespace": "tenant-a",
        "workspace_id": "workspace-a",
        "graph_space": "facts"
    });
    let mutation = |id: &str, table: &str, kind: &str, metadata: Value, document: Option<&str>| {
        json!({
            "scope": scope.clone(),
            "table": table,
            "entity_kind": kind,
            "event_id": format!("event-{id}"),
            "op": "ADD",
            "payload": {"id": id, "metadata": metadata},
            "record": {
                "id": id,
                "document": document,
                "metadata": metadata,
                "embedding": null
            },
            "embedding_dim": 0
        })
    };
    let mut mutations = vec![
        mutation(
            "n1",
            "gke_nodes",
            "node",
            json!({"label":"one"}),
            Some(r#"{"label":"one"}"#),
        ),
        mutation(
            "n2",
            "gke_nodes",
            "node",
            json!({"label":"two"}),
            Some(r#"{"label":"two"}"#),
        ),
        mutation(
            "e1",
            "gke_edges",
            "edge",
            json!({"source_ids":["n1"],"target_ids":["n2"],"relation":"links"}),
            Some(r#"{"source_ids":["n1"],"target_ids":["n2"],"relation":"links"}"#),
        ),
        mutation(
            "n3",
            "gke_nodes",
            "node",
            json!({"label":"private"}),
            Some("private"),
        ),
        mutation(
            "acl-n1",
            "gke_nodes",
            "node",
            json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"node","acl_target_entity_id":"n1","acl_version":1,"acl_mode":"public"}),
            None,
        ),
        mutation(
            "acl-n2",
            "gke_nodes",
            "node",
            json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"node","acl_target_entity_id":"n2","acl_version":1,"acl_mode":"public"}),
            None,
        ),
        mutation(
            "acl-e1",
            "gke_nodes",
            "node",
            json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"edge","acl_target_entity_id":"e1","acl_version":1,"acl_mode":"public"}),
            None,
        ),
        mutation(
            "acl-n3",
            "gke_nodes",
            "node",
            json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"node","acl_target_entity_id":"n3","acl_version":1,"acl_mode":"private","acl_owner_id":"another-user"}),
            None,
        ),
    ];
    for (index, embedding) in [
        (0, json!([1.0, 0.0])),
        (1, json!([0.8, 0.2])),
        (3, json!([0.99, 0.01])),
        (4, json!([0.999, 0.001])),
    ] {
        mutations[index]["record"]["embedding"] = embedding;
        mutations[index]["embedding_dim"] = json!(2);
    }
    let body = json!({"contract_version":1,"atomic":true,"mutations":mutations,"index_jobs":[]});

    let denied_write = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/internal/graph/mutations")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!("Bearer {}", token("docs", &["read_graph"])),
                )
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(denied_write.status(), StatusCode::FORBIDDEN);

    let denied_namespace = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/internal/graph/mutations")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!("Bearer {}", token("workflow", &["write_graph"])),
                )
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(denied_namespace.status(), StatusCode::FORBIDDEN);

    let mut cross_workspace_body = body.clone();
    for mutation in cross_workspace_body["mutations"].as_array_mut().unwrap() {
        mutation["scope"]["workspace_id"] = json!("workspace-b");
    }
    let denied_scope = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/internal/graph/mutations")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!("Bearer {}", token("docs", &["write_graph"])),
                )
                .body(Body::from(
                    serde_json::to_vec(&cross_workspace_body).unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(denied_scope.status(), StatusCode::FORBIDDEN);

    let written = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/internal/graph/mutations")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!("Bearer {}", token("docs", &["write_graph"])),
                )
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(written.status(), StatusCode::OK);
    let written: Value =
        serde_json::from_slice(&to_bytes(written.into_body(), 65_536).await.unwrap()).unwrap();
    assert_eq!(written["atomic"], true);
    assert_eq!(written["results"].as_array().unwrap().len(), 8);

    let read = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {}", token("docs", &["read_graph"])))
                .body(Body::from(
                    r#"{"jsonrpc":"2.0","id":"neighbors","method":"tools/call","params":{"name":"kg_neighbors","arguments":{"rid":"n1","truth_graph":"knowledge"}}}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(read.status(), StatusCode::OK);
    let read: Value =
        serde_json::from_slice(&to_bytes(read.into_body(), 65_536).await.unwrap()).unwrap();
    assert_eq!(read["result"]["nodes"], json!(["n2"]), "response: {read}");
    assert_eq!(read["result"]["edges"], json!(["e1"]));

    for (tool, arguments, expected) in [
        (
            "kg_find_edges",
            json!({"relation":"links","src_label_contains":"one","tgt_label_contains":"two"}),
            json!({"edges":["e1"]}),
        ),
        (
            "kg_shortest_path",
            json!({"src_id":"n1","dst_id":"n2","max_depth":4}),
            json!({"path":["n1","e1","n2"]}),
        ),
        (
            "kg_k_hop",
            json!({"start_ids":["n1"],"k":1}),
            json!({"layers":[{"nodes":[],"edges":["e1"]}]}),
        ),
    ] {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header(
                        "authorization",
                        format!("Bearer {}", token("docs", &["read_graph"])),
                    )
                    .body(Body::from(
                        json!({
                            "jsonrpc":"2.0",
                            "id":tool,
                            "method":"tools/call",
                            "params":{"name":tool,"arguments":arguments}
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
        let mut result = response["result"].clone();
        for field in [
            "implementation_authority",
            "seam_attestation_version",
            "seam_crossings",
        ] {
            result.as_object_mut().unwrap().remove(field);
        }
        assert_eq!(result, expected, "tool={tool}");
    }

    for (tool, key) in [
        ("kg_viz_d3_json", "nodes"),
        ("kg_viz_cytoscape_json", "elements"),
    ] {
        let visualization = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header(
                        "authorization",
                        format!("Bearer {}", token("docs", &["read_graph"])),
                    )
                    .body(Body::from(format!(
                        "{{\"jsonrpc\":\"2.0\",\"id\":\"{tool}\",\"method\":\"tools/call\",\"params\":{{\"name\":\"{tool}\",\"arguments\":{{\"doc_id\":null,\"mode\":\"reify\"}}}}}}"
                    )))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(visualization.status(), StatusCode::OK);
        let visualization: Value =
            serde_json::from_slice(&to_bytes(visualization.into_body(), 65_536).await.unwrap())
                .unwrap();
        let rows = visualization["result"][key].as_array().unwrap();
        let rendered = rows.iter().map(Value::to_string).collect::<String>();
        assert!(rendered.contains("n1") && rendered.contains("n2"));
        assert!(
            !rendered.contains("n3"),
            "private node leaked: {visualization}"
        );
    }

    let vector_query = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/internal/graph/vector-query")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!("Bearer {}", token("docs", &["read_graph"])),
                )
                .body(Body::from(
                    json!({
                        "entity_kind": "node",
                        "query_embeddings": [[1.0, 0.0]],
                        "metric": "cosine",
                        "truth_graph": "knowledge",
                        "n_results": 2,
                        "where": {},
                        "include": ["distances"]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(vector_query.status(), StatusCode::OK);
    let vector_query: Value =
        serde_json::from_slice(&to_bytes(vector_query.into_body(), 65_536).await.unwrap()).unwrap();
    assert_eq!(vector_query["ids"], json!([["n1", "n2"]]));
    assert_eq!(vector_query["distances"][0].as_array().unwrap().len(), 2);
    assert!(vector_query.get("documents").is_none());
}
