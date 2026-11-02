#[test]
fn failed_inline_graph_effect_clears_provider_event_side_effects() {
    let mut effect: RecordedWorkerSuccessEffect = serde_json::from_value(json!({
        "contract_version": 1,
        "effect_id": "effect-1",
        "status": "success",
        "provider_event_acknowledgements": [{
            "message_id": "message-1",
            "claimed_by": "worker-1",
            "event_id": "event-1"
        }],
        "provider_event_publications": [{
            "namespace": "conversation",
            "conversation_id": "conversation-1",
            "event": {
                "event_id": "event-2",
                "trace": {
                    "run_id": "run-1",
                    "turn_id": "turn-1",
                    "workflow_node_id": "node-1",
                    "step_id": "step-1",
                    "provider_id": "provider-1"
                },
                "event": {"kind": "response_started"}
            }
        }]
    }))
    .expect("valid worker effect");

    fail_inline_graph_effect(
        &mut effect,
        "KOGWISTAR_GRAPH_AUTHORIZATION_DENIED",
        "owner authorization failed",
    );

    assert_eq!(
        effect.status,
        kogwistar_runtime::RuntimeWorkerEffectStatus::Failed
    );
    assert!(effect.provider_event_acknowledgements.is_empty());
    assert!(effect.provider_event_publications.is_empty());
    assert_eq!(effect.errors, ["owner authorization failed"]);
}

#[test]
fn inline_graph_mutation_scope_uses_current_owner_scope() {
    let principal = json!({
        "ns": ["docs"],
        "storage_ns": "docs",
        "workspace": "workspace-a",
        "graph_space": "knowledge"
    });
    let mutation: kogwistar_contracts::GraphMutationV1 = serde_json::from_value(json!({
        "scope": {
            "namespace": "docs",
            "workspace_id": "workspace-a",
            "graph_space": "knowledge"
        },
        "table": "gke_nodes",
        "entity_kind": "node",
        "event_id": "event-1",
        "op": "upsert",
        "payload": {},
        "record": {"id": "node-1", "metadata": {}},
        "embedding_dim": 0
    }))
    .expect("valid graph mutation");

    assert!(!inline_graph_mutations_outside_owner_scope(
        &principal,
        std::slice::from_ref(&mutation)
    ));

    let mut cross_scope = mutation;
    cross_scope.scope.workspace_id = Some("workspace-b".to_owned());
    assert!(inline_graph_mutations_outside_owner_scope(
        &principal,
        &[cross_scope]
    ));
}

#[test]
fn recorded_step_snapshots_matches_python_fixture() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../../../contracts/golden/workflow-step-trace-context-v1.json"
    ))
    .unwrap();
    let recorded = ServerRunEvent {
        seq: 4,
        run_id: "run-1".to_owned(),
        event_type: "workflow.recorded_transition.v1".to_owned(),
        payload_json: json!({
            "transition_id": "transition-1",
            "reduced": {
                "state": {
                    "last_step_seq": 0,
                    "workflow_id": "workflow-1",
                    "last_node_id": "node-1",
                    "state": {}
                },
                "result": {"ok": true},
                "errors": [],
                "server_status": "running"
            }
        })
        .to_string(),
        created_at_ms: 10,
    };
    let trace = ServerRunEvent {
        seq: 5,
        run_id: "run-1".to_owned(),
        event_type: "workflow.trace.v1".to_owned(),
        payload_json: json!({
            "trace": {
                "event": "runtime.native.step.completed",
                "step_seq": 0,
                "node_id": "node-1",
                "trace_id": fixture["trace_context"]["trace_id"],
                "span_id": fixture["trace_context"]["span_id"],
                "parent_span_id": fixture["trace_context"]["parent_span_id"],
                "attempt": fixture["trace_context"]["attempt"]
            }
        })
        .to_string(),
        created_at_ms: 11,
    };

    let steps = recorded_step_snapshots(vec![recorded.clone(), trace]);
    assert_eq!(steps.len(), 1);
    assert_eq!(
        steps[0].trace_context.as_ref().unwrap(),
        &fixture["trace_context"]
    );
    let legacy_steps = recorded_step_snapshots(vec![recorded]);
    assert!(legacy_steps[0].trace_context.is_none());
    assert!(fixture["legacy_step_omits_trace_context"].as_bool().unwrap());
}

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
                        "query_embeddings": [[1.0, 0.0], [0.0, 1.0]],
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
    assert_eq!(vector_query["ids"], json!([["n1", "n2"], ["n2", "n1"]]));
    assert_eq!(vector_query["distances"][0].as_array().unwrap().len(), 2);
    assert_eq!(vector_query["distances"][1].as_array().unwrap().len(), 2);
    assert!(vector_query.get("documents").is_none());
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
    let token = |subject: &str, capabilities: &[&str]| {
        encode(
            &Header::new(Algorithm::HS256),
            &json!({
                "sub": subject,
                "role": "rw",
                "ns": ["docs"],
                "storage_ns": fixture["scope"]["namespace"],
                "workspace": fixture["scope"]["workspace_id"],
                "workspace_id": fixture["scope"]["workspace_id"],
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
                    format!("Bearer {}", token("owner-b", &["write_graph"])),
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
                            format!("Bearer {}", token(subject, &["read_graph"])),
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
        }
    }
}

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
