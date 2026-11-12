#[tokio::test]
async fn in_memory_backend_routes_authenticated_graph_write_and_mcp_read() {
    let acl_fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/graph-acl-v1.json"
    )))
    .unwrap();
    let isolation = &acl_fixture["graph_type_isolation"];
    let shared_entity_id = isolation["entity_id"].as_str().unwrap();
    let owner_policy = acl_fixture["targets"]
        .as_array()
        .unwrap()
        .iter()
        .find(|target| target["id"] == "node-owner")
        .unwrap();
    let owner_entity_id = owner_policy["id"].as_str().unwrap();
    let owner_id = owner_policy["owner_id"].as_str().unwrap();
    let private_policy = acl_fixture["targets"]
        .as_array()
        .unwrap()
        .iter()
        .find(|target| target["id"] == "node-private")
        .unwrap();
    let private_entity_id = private_policy["id"].as_str().unwrap();
    let private_owner_id = private_policy["owner_id"].as_str().unwrap();
    let no_acl_node_id = acl_fixture["targets"]
        .as_array()
        .unwrap()
        .iter()
        .find(|target| target["id"] == "node-no-acl")
        .unwrap()["id"]
        .as_str()
        .unwrap();
    let no_acl_edge_id = acl_fixture["targets"]
        .as_array()
        .unwrap()
        .iter()
        .find(|target| target["id"] == "edge-no-acl")
        .unwrap()["id"]
        .as_str()
        .unwrap();
    let edge_policy = acl_fixture["targets"]
        .as_array()
        .unwrap()
        .iter()
        .find(|target| target["id"] == "edge-public")
        .unwrap();
    let public_edge_id = edge_policy["id"].as_str().unwrap();
    let private_edge_policy = acl_fixture["targets"]
        .as_array()
        .unwrap()
        .iter()
        .find(|target| target["id"] == "edge-private")
        .unwrap();
    let private_edge_id = private_edge_policy["id"].as_str().unwrap();
    let private_edge_owner_id = private_edge_policy["owner_id"].as_str().unwrap();
    let scope_edge_policy = acl_fixture["targets"]
        .as_array()
        .unwrap()
        .iter()
        .find(|target| target["id"] == "edge-scope")
        .unwrap();
    let scope_edge_id = scope_edge_policy["id"].as_str().unwrap();
    let scope_edge_value = scope_edge_policy["security_scope"].as_str().unwrap();
    let scope_policy = acl_fixture["targets"]
        .as_array()
        .unwrap()
        .iter()
        .find(|target| target["id"] == "node-scope")
        .unwrap();
    let scope_entity_id = scope_policy["id"].as_str().unwrap();
    let scope_value = scope_policy["security_scope"].as_str().unwrap();
    let share_policies: Vec<&Value> = acl_fixture["targets"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|target| {
            target["grain"] == "node"
                && matches!(target["mode"].as_str(), Some("shared" | "group"))
        })
        .collect();
    let edge_share_policies: Vec<&Value> = acl_fixture["targets"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|target| {
            target["grain"] == "edge"
                && matches!(target["mode"].as_str(), Some("shared" | "group"))
        })
        .collect();
    let fixture_principal_id = acl_fixture["principal"]["id"].as_str().unwrap();
    let fixture_groups: Vec<String> = acl_fixture["principal"]["groups"]
        .as_array()
        .unwrap()
        .iter()
        .map(|group| group.as_str().unwrap().to_owned())
        .collect();
    let isolation_cases = isolation["cases"].as_array().unwrap();
    let knowledge_policy = isolation_cases
        .iter()
        .find(|case| case["truth_graph"] == "knowledge")
        .unwrap();
    let conversation_policy = isolation_cases
        .iter()
        .find(|case| case["truth_graph"] == "conversation")
        .unwrap();

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
    let token_for_subject =
        |subject: &str, namespace: &str, capabilities: &[&str], graph_space: &str| {
        encode(
            &Header::new(Algorithm::HS256),
            &json!({
                "sub": subject,
                "user_id": subject,
                "role": "rw",
                "ns": [namespace],
                "storage_ns": "tenant-a",
                "workspace": "workspace-a",
                "workspace_id": "workspace-a",
                "graph_space": graph_space,
                "capabilities": capabilities,
                "exp": now + 300
            }),
            &EncodingKey::from_secret(secret.as_bytes()),
        )
        .unwrap()
    };
    let token_for_graph_space = |namespace: &str, capabilities: &[&str], graph_space: &str| {
        token_for_subject("writer", namespace, capabilities, graph_space)
    };
    let token = |namespace: &str, capabilities: &[&str]| {
        token_for_graph_space(namespace, capabilities, "facts")
    };
    let application = Arc::new(InMemoryGraphApplicationService::default());
    let make_app = || {
        router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: vec![],
                implementation: ImplementationSnapshot::default(),
                auth: auth.clone(),
            },
            application.clone(),
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
            json!({"label":"one","doc_ids":["doc-a"]}),
            Some(
                r#"{"label":"one","doc_ids":["doc-a"],"mentions":[{"spans":[{"doc_id":"doc-a","insertion_method":"parser"}]}]}"#,
            ),
        ),
        mutation(
            "n2",
            "gke_nodes",
            "node",
            json!({"label":"two","doc_ids":["doc-a"]}),
            Some(
                r#"{"label":"two","doc_ids":["doc-a"],"mentions":[{"spans":[{"doc_id":"doc-a","insertion_method":"parser"}]}]}"#,
            ),
        ),
        mutation(
            "e1",
            "gke_edges",
            "edge",
            json!({"source_ids":["n1"],"target_ids":["n2"],"relation":"links"}),
            Some(
                r#"{"source_ids":["n1"],"target_ids":["n2"],"relation":"links","mentions":[{"spans":[{"doc_id":"doc-a","insertion_method":"parser"}]}]}"#,
            ),
        ),
        mutation(
            "n3",
            "gke_nodes",
            "node",
            json!({"label":"private"}),
            Some("private"),
        ),
        mutation(
            shared_entity_id,
            "gke_nodes",
            "node",
            json!({"label":"same ID in knowledge"}),
            Some("same ID in knowledge"),
        ),
        mutation(
            "acl-n1",
            "gke_nodes",
            "node",
            json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"node","acl_target_entity_id":"n1","acl_version":1,"acl_mode":"public"}),
            None,
        ),
        mutation(
            "acl-n1-mention",
            "gke_nodes",
            "node",
            json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"grounding","acl_target_entity_id":"n1","acl_target_item_id":"n1::mention::0","acl_version":1,"acl_mode":"public"}),
            None,
        ),
        mutation(
            "acl-n1-span",
            "gke_nodes",
            "node",
            json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"span","acl_target_entity_id":"n1","acl_target_item_id":"n1::mention::0::span::0","acl_version":1,"acl_mode":"public"}),
            None,
        ),
        mutation(
            "acl-shared-knowledge",
            "gke_nodes",
            "node",
            json!({
                "entity_type":"acl_record",
                "acl_truth_graph":"knowledge",
                "acl_target_grain":"node",
                "acl_target_entity_id":shared_entity_id,
                "acl_version":1,
                "acl_mode":knowledge_policy["mode"]
            }),
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
            "acl-n2-mention",
            "gke_nodes",
            "node",
            json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"grounding","acl_target_entity_id":"n2","acl_target_item_id":"n2::mention::0","acl_version":1,"acl_mode":"public"}),
            None,
        ),
        mutation(
            "acl-n2-span",
            "gke_nodes",
            "node",
            json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"span","acl_target_entity_id":"n2","acl_target_item_id":"n2::mention::0::span::0","acl_version":1,"acl_mode":"public"}),
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
            "acl-e1-mention",
            "gke_nodes",
            "node",
            json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"grounding","acl_target_entity_id":"e1","acl_target_item_id":"e1::mention::0","acl_version":1,"acl_mode":"public"}),
            None,
        ),
        mutation(
            "acl-e1-span",
            "gke_nodes",
            "node",
            json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"span","acl_target_entity_id":"e1","acl_target_item_id":"e1::mention::0::span::0","acl_version":1,"acl_mode":"public"}),
            None,
        ),
        mutation(
            "acl-n3",
            "gke_nodes",
            "node",
            json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"node","acl_target_entity_id":"n3","acl_version":1,"acl_mode":"private","acl_owner_id":"another-user"}),
            None,
        ),
        mutation(
            owner_entity_id,
            "gke_nodes",
            "node",
            json!({"label":"owner only"}),
            Some("owner only"),
        ),
        mutation(
            "acl-node-owner",
            "gke_nodes",
            "node",
            json!({
                "entity_type":"acl_record",
                "acl_truth_graph":acl_fixture["truth_graph"],
                "acl_target_grain":owner_policy["grain"],
                "acl_target_entity_id":owner_entity_id,
                "acl_version":1,
                "acl_mode":owner_policy["mode"],
                "owner_id":owner_id
            }),
            None,
        ),
        mutation(
            private_entity_id,
            "gke_nodes",
            "node",
            json!({"label":"another owner's node"}),
            Some("another owner's node"),
        ),
        mutation(
            "acl-node-private",
            "gke_nodes",
            "node",
            json!({
                "entity_type":"acl_record",
                "acl_truth_graph":acl_fixture["truth_graph"],
                "acl_target_grain":private_policy["grain"],
                "acl_target_entity_id":private_entity_id,
                "acl_version":1,
                "acl_mode":private_policy["mode"],
                "owner_id":private_owner_id
            }),
            None,
        ),
        mutation(
            scope_entity_id,
            "gke_nodes",
            "node",
            json!({"label":"team-scoped node"}),
            Some("team-scoped node"),
        ),
        mutation(
            "acl-node-scope",
            "gke_nodes",
            "node",
            json!({
                "entity_type":"acl_record",
                "acl_truth_graph":acl_fixture["truth_graph"],
                "acl_target_grain":scope_policy["grain"],
                "acl_target_entity_id":scope_entity_id,
                "acl_version":1,
                "acl_mode":scope_policy["mode"],
                "security_scope":scope_value
            }),
            None,
        ),
    ];
    for (index, share_policy) in share_policies.iter().enumerate() {
        let shared_entity_id = share_policy["id"].as_str().unwrap();
        mutations.push(mutation(
            shared_entity_id,
            "gke_nodes",
            "node",
            json!({"label":format!("{} graph node", share_policy["mode"].as_str().unwrap())}),
            Some("explicitly shared node"),
        ));
        let mut acl_metadata = json!({
            "entity_type":"acl_record",
            "acl_truth_graph":acl_fixture["truth_graph"],
            "acl_target_grain":share_policy["grain"],
            "acl_target_entity_id":shared_entity_id,
            "acl_version":1,
            "acl_mode":share_policy["mode"]
        });
        if share_policy["mode"] == "shared" {
            acl_metadata["shared_with_principals"] =
                share_policy["shared_with_principals"].clone();
        } else {
            acl_metadata["shared_with_groups"] = share_policy["shared_with_groups"].clone();
        }
        mutations.push(mutation(
            &format!("acl-node-share-{index}"),
            "gke_nodes",
            "node",
            acl_metadata,
            None,
        ));
    }
    let public_edge_data = json!({
        "source_ids":[shared_entity_id],
        "target_ids":[shared_entity_id],
        "source_edge_ids":[],
        "target_edge_ids":[],
        "relation":"references"
    });
    let public_edge_document = public_edge_data.to_string();
    mutations.push(mutation(
        public_edge_id,
        "gke_edges",
        "edge",
        public_edge_data,
        Some(&public_edge_document),
    ));
    mutations.push(mutation(
        "acl-edge-public",
        "gke_nodes",
        "node",
        json!({
            "entity_type":"acl_record",
            "acl_truth_graph":acl_fixture["truth_graph"],
            "acl_target_grain":edge_policy["grain"],
            "acl_target_entity_id":public_edge_id,
            "acl_version":1,
            "acl_mode":edge_policy["mode"]
        }),
        None,
    ));
    let private_edge_data = json!({
        "source_ids":[shared_entity_id],
        "target_ids":[shared_entity_id],
        "source_edge_ids":[],
        "target_edge_ids":[],
        "relation":"private references"
    });
    let private_edge_document = private_edge_data.to_string();
    mutations.push(mutation(
        private_edge_id,
        "gke_edges",
        "edge",
        private_edge_data,
        Some(&private_edge_document),
    ));
    mutations.push(mutation(
        "acl-edge-private",
        "gke_nodes",
        "node",
        json!({
            "entity_type":"acl_record",
            "acl_truth_graph":acl_fixture["truth_graph"],
            "acl_target_grain":private_edge_policy["grain"],
            "acl_target_entity_id":private_edge_id,
            "acl_version":1,
            "acl_mode":private_edge_policy["mode"],
            "owner_id":private_edge_owner_id
        }),
        None,
    ));
    let scope_edge_data = json!({
        "source_ids":[shared_entity_id],
        "target_ids":[shared_entity_id],
        "source_edge_ids":[],
        "target_edge_ids":[],
        "relation":"scope-limited references"
    });
    let scope_edge_document = scope_edge_data.to_string();
    mutations.push(mutation(
        scope_edge_id,
        "gke_edges",
        "edge",
        scope_edge_data,
        Some(&scope_edge_document),
    ));
    mutations.push(mutation(
        "acl-edge-scope",
        "gke_nodes",
        "node",
        json!({
            "entity_type":"acl_record",
            "acl_truth_graph":acl_fixture["truth_graph"],
            "acl_target_grain":scope_edge_policy["grain"],
            "acl_target_entity_id":scope_edge_id,
            "acl_version":1,
            "acl_mode":scope_edge_policy["mode"],
            "security_scope":scope_edge_value
        }),
        None,
    ));
    for (index, policy) in edge_share_policies.iter().enumerate() {
        let entity_id = policy["id"].as_str().unwrap();
        let edge_data = json!({
            "source_ids":[shared_entity_id],
            "target_ids":[shared_entity_id],
            "source_edge_ids":[],
            "target_edge_ids":[],
            "relation":"shared references"
        });
        let edge_document = edge_data.to_string();
        mutations.push(mutation(
            entity_id,
            "gke_edges",
            "edge",
            edge_data,
            Some(&edge_document),
        ));
        let mut acl_record = json!({
            "entity_type":"acl_record",
            "acl_truth_graph":acl_fixture["truth_graph"],
            "acl_target_grain":policy["grain"],
            "acl_target_entity_id":entity_id,
            "acl_version":1,
            "acl_mode":policy["mode"]
        });
        for field in ["shared_with_principals", "shared_with_groups"] {
            if let Some(value) = policy.get(field) {
                acl_record[field] = value.clone();
            }
        }
        mutations.push(mutation(
            &format!("acl-edge-shared-{index}"),
            "gke_nodes",
            "node",
            acl_record,
            None,
        ));
    }
    for (index, embedding) in [
        (0, json!([1.0, 0.0])),
        (1, json!([0.8, 0.2])),
        (3, json!([0.99, 0.01])),
        (5, json!([0.999, 0.001])),
    ] {
        mutations[index]["record"]["embedding"] = embedding;
        mutations[index]["embedding_dim"] = json!(2);
    }
    mutations.push(mutation(
        no_acl_node_id,
        "gke_nodes",
        "node",
        json!({"label":"Node without an ACL record"}),
        Some("Node without an ACL record"),
    ));
    let no_acl_edge = json!({
        "source_ids":[shared_entity_id],
        "target_ids":[shared_entity_id],
        "source_edge_ids":[],
        "target_edge_ids":[],
        "relation":"Edge without an ACL record"
    });
    let no_acl_edge_document = no_acl_edge.to_string();
    mutations.push(mutation(
        no_acl_edge_id,
        "gke_edges",
        "edge",
        no_acl_edge,
        Some(&no_acl_edge_document),
    ));
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
    assert_eq!(
        written["results"].as_array().unwrap().len(),
        body["mutations"].as_array().unwrap().len()
    );
    let load_persisted = |id: &str, input: Value| {
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
                    "jsonrpc":"2.0","id":id,"method":"tools/call",
                    "params":{"name":"kg_load_persisted","arguments":{"inp":input}}
                })
                .to_string(),
            ))
            .unwrap()
    };
    let loaded = app
        .clone()
        .oneshot(load_persisted(
            "load-persisted",
            json!({"doc_ids":["doc-a"],"sid":false}),
        ))
        .await
        .unwrap();
    let loaded: Value =
        serde_json::from_slice(&to_bytes(loaded.into_body(), 65_536).await.unwrap()).unwrap();
    assert_eq!(
        loaded["result"]["node_ids"],
        json!(["n1", "n2"]),
        "{loaded}"
    );
    assert_eq!(loaded["result"]["edge_ids"], json!(["e1"]));
    let loaded_by_method = app
        .clone()
        .oneshot(load_persisted(
            "load-persisted-method",
            json!({"doc_ids":["doc-a"],"insertion_method":"parser","sid":false}),
        ))
        .await
        .unwrap();
    let loaded_by_method: Value = serde_json::from_slice(
        &to_bytes(loaded_by_method.into_body(), 65_536)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(loaded_by_method["result"]["node_ids"], json!(["n1", "n2"]));
    assert_eq!(loaded_by_method["result"]["edge_ids"], json!(["e1"]));

    let brute_force = app
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
                        "jsonrpc":"2.0","id":"brute-force","method":"tools/call",
                        "params":{"name":"kg_propose_bruteforce","arguments":{"inp":{
                            "pair_kind":"node_node","allowed_docs":["doc-a"]
                        }}}
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(brute_force.status(), StatusCode::OK);
    let brute_force: Value =
        serde_json::from_slice(&to_bytes(brute_force.into_body(), 65_536).await.unwrap()).unwrap();
    assert_eq!(
        brute_force["result"]["pairs"],
        json!([{
            "left_id":"n1","left_kind":"node",
            "right_id":"n2","right_kind":"node"
        }])
    );
    let mixed_pairs = app
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
                        "jsonrpc":"2.0","id":"mixed-pairs","method":"tools/call",
                        "params":{"name":"kg_propose_bruteforce","arguments":{"inp":{
                            "pair_kind":"any_any","allowed_docs":["doc-a"]
                        }}}
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    let mixed_pairs: Value =
        serde_json::from_slice(&to_bytes(mixed_pairs.into_body(), 65_536).await.unwrap()).unwrap();
    assert_eq!(mixed_pairs["result"]["pairs"].as_array().unwrap().len(), 3);
    assert!(mixed_pairs.to_string().contains("e1"));
    let denied_proposal = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {}", token("docs", &[])))
                .body(Body::from(
                    json!({
                        "jsonrpc":"2.0","id":"denied-proposal","method":"tools/call",
                        "params":{"name":"kg_propose_bruteforce","arguments":{"inp":{}}}
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    let denied_proposal: Value =
        serde_json::from_slice(&to_bytes(denied_proposal.into_body(), 65_536).await.unwrap())
            .unwrap();
    assert!(denied_proposal.get("error").is_some());
    assert!(denied_proposal.get("result").is_none());

    let cross_kind_fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/graph-neighbors-cross-kind-id-v1.json"
    )))
    .unwrap();
    let parent_edge = &cross_kind_fixture["parent_edge"];
    let cross_kind_id = cross_kind_fixture["expected"]["node_ids"][0]
        .as_str()
        .unwrap();
    let parent_edge_id = cross_kind_fixture["rid"].as_str().unwrap();
    let mut cross_kind_node = mutation(
        cross_kind_id,
        "gke_nodes",
        "node",
        json!({"label":"shared node identity","visibility":"public"}),
        Some(r#"{"label":"shared node identity","mentions":[{"spans":[{"doc_id":"D-cross-kind"}]}]}"#),
    );
    cross_kind_node["payload"]["mentions"] =
        json!([{"spans":[{"doc_id":"D-cross-kind"}]}]);
    cross_kind_node["event_id"] = json!("event-cross-kind-node");
    let mut cross_kind_edge = mutation(
        cross_kind_id,
        "gke_edges",
        "edge",
        json!({"source_ids":[cross_kind_id],"target_ids":[cross_kind_id],"relation":"shared edge identity","visibility":"private","owner_id":"writer"}),
        Some(r#"{"source_ids":["shared-id"],"target_ids":["shared-id"],"relation":"shared edge identity","mentions":[{"spans":[{"doc_id":"D-cross-kind"}]}]}"#),
    );
    cross_kind_edge["payload"]["mentions"] =
        json!([{"spans":[{"doc_id":"D-cross-kind"}]}]);
    cross_kind_edge["event_id"] = json!("event-cross-kind-edge");
    let mut public_parent_edge = parent_edge.clone();
    public_parent_edge["source_edge_ids"] = json!([]);
    let mut parent_edge_mutation = mutation(
        parent_edge_id,
        "gke_edges",
        "edge",
        public_parent_edge.clone(),
        Some(&public_parent_edge.to_string()),
    );
    parent_edge_mutation["record"]["metadata"]["visibility"] = json!("public");
    parent_edge_mutation["event_id"] = json!("event-cross-kind-parent-edge");
    let node_grounding_id = format!(
        "__kogwistar_acl_usage_v2__:node:{cross_kind_id}::mention::0"
    );
    let node_span_id = format!("{node_grounding_id}::span::0");
    let edge_grounding_id = format!(
        "__kogwistar_acl_usage_v2__:edge:{cross_kind_id}::mention::0"
    );
    let edge_span_id = format!("{edge_grounding_id}::span::0");
    let cross_kind_mutations = vec![
        cross_kind_node,
        cross_kind_edge,
        parent_edge_mutation,
        mutation(
            "acl-cross-kind-node",
            "gke_nodes",
            "node",
            json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"node","acl_target_entity_id":cross_kind_id,"acl_version":1,"acl_mode":"public"}),
            None,
        ),
        mutation(
            "acl-cross-kind-node-grounding",
            "gke_nodes",
            "node",
            json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"grounding","acl_target_entity_id":cross_kind_id,"acl_target_item_id":node_grounding_id,"acl_version":1,"acl_mode":"public"}),
            None,
        ),
        mutation(
            "acl-cross-kind-node-span",
            "gke_nodes",
            "node",
            json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"span","acl_target_entity_id":cross_kind_id,"acl_target_item_id":node_span_id,"acl_version":1,"acl_mode":"public"}),
            None,
        ),
        mutation(
            "acl-cross-kind-edge",
            "gke_nodes",
            "node",
            json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"edge","acl_target_entity_id":cross_kind_id,"acl_version":1,"acl_mode":"private","owner_id":"writer"}),
            None,
        ),
        mutation(
            "acl-cross-kind-edge-grounding",
            "gke_nodes",
            "node",
            json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"grounding","acl_target_entity_id":cross_kind_id,"acl_target_item_id":edge_grounding_id,"acl_version":1,"acl_mode":"private","owner_id":"writer"}),
            None,
        ),
        mutation(
            "acl-cross-kind-edge-span",
            "gke_nodes",
            "node",
            json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"span","acl_target_entity_id":cross_kind_id,"acl_target_item_id":edge_span_id,"acl_version":1,"acl_mode":"private","owner_id":"writer"}),
            None,
        ),
        mutation(
            "acl-cross-kind-parent-edge",
            "gke_nodes",
            "node",
            json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"edge","acl_target_entity_id":parent_edge_id,"acl_version":1,"acl_mode":"public"}),
            None,
        ),
    ];
    let cross_kind_body = json!({
        "contract_version":1,
        "atomic":true,
        "mutations":cross_kind_mutations,
        "index_jobs":[]
    });
    let cross_kind_write = app
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
                .body(Body::from(serde_json::to_vec(&cross_kind_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(cross_kind_write.status(), StatusCode::OK);
    let cross_kind_reader_neighbors = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!(
                        "Bearer {}",
                        token_for_subject("reader", "docs", &["read_graph"], "facts")
                    ),
                )
                .body(Body::from(
                    json!({
                        "jsonrpc":"2.0",
                        "id":"cross-kind-reader-neighbors",
                        "method":"tools/call",
                        "params":{"name":"kg_neighbors","arguments":{
                            "rid":parent_edge_id,
                            "truth_graph":"knowledge"
                        }}
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(cross_kind_reader_neighbors.status(), StatusCode::OK);
    let cross_kind_reader_neighbors: Value = serde_json::from_slice(
        &to_bytes(cross_kind_reader_neighbors.into_body(), 65_536)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        cross_kind_reader_neighbors["result"]["nodes"],
        cross_kind_fixture["expected"]["node_ids"]
    );
    assert_eq!(cross_kind_reader_neighbors["result"]["edges"], json!([]));
    let cross_kind_reader_by_node_id = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!(
                        "Bearer {}",
                        token_for_subject("reader", "docs", &["read_graph"], "facts")
                    ),
                )
                .body(Body::from(
                    json!({
                        "jsonrpc":"2.0",
                        "id":"cross-kind-reader-by-node-id",
                        "method":"tools/call",
                        "params":{"name":"kg_neighbors","arguments":{
                            "rid":cross_kind_id,
                            "truth_graph":"knowledge"
                        }}
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    let cross_kind_reader_by_node_id: Value = serde_json::from_slice(
        &to_bytes(cross_kind_reader_by_node_id.into_body(), 65_536)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(cross_kind_reader_by_node_id["result"]["nodes"], json!([]));
    assert_eq!(
        cross_kind_reader_by_node_id["result"]["edges"],
        json!([parent_edge_id])
    );
    let cross_kind_neighbors = app
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
                        "id":"cross-kind-neighbors",
                        "method":"tools/call",
                        "params":{"name":"kg_neighbors","arguments":{
                            "rid":cross_kind_id,
                            "truth_graph":"knowledge"
                        }}
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(cross_kind_neighbors.status(), StatusCode::OK);
    let cross_kind_neighbors: Value = serde_json::from_slice(
        &to_bytes(cross_kind_neighbors.into_body(), 65_536)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        cross_kind_neighbors["result"]["implementation_authority"],
        "rust"
    );
    assert_eq!(
        cross_kind_neighbors["result"]["seam_attestation_version"],
        1
    );
    assert_eq!(cross_kind_neighbors["result"]["seam_crossings"], json!([]));
    assert_eq!(
        cross_kind_neighbors["result"]["nodes"],
        json!([cross_kind_id])
    );
    assert_eq!(
        cross_kind_neighbors["result"]["edges"],
        cross_kind_fixture["edge_ids"]
    );

    let redirect_mutations = [
        mutation(
            "redirect-old",
            "gke_nodes",
            "node",
            json!({"label":"old node","lifecycle_status":"tombstoned","redirect_to_id":"redirect-current"}),
            Some("old node"),
        ),
        mutation(
            "redirect-current",
            "gke_nodes",
            "node",
            json!({"label":"current node"}),
            Some("current node"),
        ),
        mutation(
            "redirect-neighbor",
            "gke_nodes",
            "node",
            json!({"label":"neighbor"}),
            Some("neighbor"),
        ),
        mutation(
            "redirect-edge",
            "gke_edges",
            "edge",
            json!({"source_ids":["redirect-current"],"target_ids":["redirect-neighbor"]}),
            Some(r#"{"source_ids":["redirect-current"],"target_ids":["redirect-neighbor"]}"#),
        ),
        mutation(
            "redirect-edge-old",
            "gke_edges",
            "edge",
            json!({"source_ids":[],"target_ids":[],"lifecycle_status":"tombstoned","redirect_to_id":"redirect-edge"}),
            Some(r#"{"source_ids":[],"target_ids":[]}"#),
        ),
        mutation(
            "redirect-edge-source",
            "gke_edges",
            "edge",
            json!({"source_ids":[],"target_ids":[],"lifecycle_status":"tombstoned","redirect_to_id":"redirect-edge"}),
            Some(r#"{"source_ids":[],"target_ids":[]}"#),
        ),
        mutation(
            "redirect-edge-old",
            "gke_nodes",
            "node",
            json!({"label":"same-string tombstoned node"}),
            Some("same-string tombstoned node"),
        ),
        mutation(
            "redirect-old-private",
            "gke_nodes",
            "node",
            json!({"label":"old private node","lifecycle_status":"tombstoned","redirect_to_id":"redirect-private"}),
            Some("old private node"),
        ),
        mutation(
            "redirect-private",
            "gke_nodes",
            "node",
            json!({"label":"private target"}),
            Some("private target"),
        ),
        mutation(
            "redirect-private-edge",
            "gke_edges",
            "edge",
            json!({"source_ids":["redirect-current"],"target_ids":["redirect-neighbor"]}),
            Some(r#"{"source_ids":["redirect-current"],"target_ids":["redirect-neighbor"]}"#),
        ),
        mutation(
            "redirect-private-edge-old",
            "gke_edges",
            "edge",
            json!({"source_ids":[],"target_ids":[],"lifecycle_status":"tombstoned","redirect_to_id":"redirect-private-edge"}),
            Some(r#"{"source_ids":[],"target_ids":[]}"#),
        ),
        mutation(
            "acl-redirect-current",
            "gke_nodes",
            "node",
            json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"node","acl_target_entity_id":"redirect-current","acl_version":1,"acl_mode":"public"}),
            None,
        ),
        mutation(
            "acl-redirect-neighbor",
            "gke_nodes",
            "node",
            json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"node","acl_target_entity_id":"redirect-neighbor","acl_version":1,"acl_mode":"public"}),
            None,
        ),
        mutation(
            "acl-redirect-edge",
            "gke_nodes",
            "node",
            json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"edge","acl_target_entity_id":"redirect-edge","acl_version":1,"acl_mode":"public"}),
            None,
        ),
        mutation(
            "acl-redirect-private",
            "gke_nodes",
            "node",
            json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"node","acl_target_entity_id":"redirect-private","acl_version":1,"acl_mode":"private","acl_owner_id":"other-user"}),
            None,
        ),
        mutation(
            "acl-redirect-private-edge",
            "gke_nodes",
            "node",
            json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"edge","acl_target_entity_id":"redirect-private-edge","acl_version":1,"acl_mode":"private","acl_owner_id":"other-user"}),
            None,
        ),
    ]
    .into_iter()
    .enumerate()
    .map(|(index, mut value)| {
        value["event_id"] = json!(format!("redirect-event-{index}"));
        value
    })
    .collect::<Vec<_>>();
    let redirect_body = json!({
        "contract_version":1,
        "atomic":true,
        "mutations":redirect_mutations,
        "index_jobs":[]
    });
    let redirect_write = app
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
                .body(Body::from(serde_json::to_vec(&redirect_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(redirect_write.status(), StatusCode::OK);

    let read_redirect = |rid: &str, resolve_mode: &str| {
        app.clone().oneshot(
            Request::builder()
                .method("POST")
                .uri("/internal/graph/neighbors")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!("Bearer {}", token("docs", &["read_graph"])),
                )
                .body(Body::from(
                    json!({"rid":rid,"resolve_mode":resolve_mode,"truth_graph":"knowledge"})
                        .to_string(),
                ))
                .unwrap(),
        )
    };
    let redirected = read_redirect("redirect-old", "redirect").await.unwrap();
    assert_eq!(redirected.status(), StatusCode::OK);
    let redirected: Value =
        serde_json::from_slice(&to_bytes(redirected.into_body(), 65_536).await.unwrap()).unwrap();
    assert_eq!(redirected["nodes"], json!(["redirect-neighbor"]));
    assert_eq!(redirected["edges"], json!(["redirect-edge"]));

    let redirected_edge = read_redirect("redirect-edge-old", "redirect").await.unwrap();
    assert_eq!(redirected_edge.status(), StatusCode::OK);
    let redirected_edge: Value = serde_json::from_slice(
        &to_bytes(redirected_edge.into_body(), 65_536).await.unwrap(),
    )
    .unwrap();
    assert_eq!(redirected_edge["nodes"], json!([]));
    assert_eq!(redirected_edge["edges"], json!([]));

    let redirected_edge_source = read_redirect("redirect-edge-source", "redirect")
        .await
        .unwrap();
    assert_eq!(redirected_edge_source.status(), StatusCode::OK);
    let redirected_edge_source: Value = serde_json::from_slice(
        &to_bytes(redirected_edge_source.into_body(), 65_536)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        redirected_edge_source["nodes"],
        json!(["redirect-current", "redirect-neighbor"])
    );
    assert_eq!(redirected_edge_source["edges"], json!([]));

    let private_edge_redirect = read_redirect("redirect-private-edge-old", "redirect")
        .await
        .unwrap();
    assert_eq!(private_edge_redirect.status(), StatusCode::OK);
    let private_edge_redirect: Value = serde_json::from_slice(
        &to_bytes(private_edge_redirect.into_body(), 65_536)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(private_edge_redirect["nodes"], json!([]));
    assert_eq!(private_edge_redirect["edges"], json!([]));

    let private_redirect = read_redirect("redirect-old-private", "redirect")
        .await
        .unwrap();
    assert_eq!(private_redirect.status(), StatusCode::OK);
    let private_redirect: Value = serde_json::from_slice(
        &to_bytes(private_redirect.into_body(), 65_536)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(private_redirect["nodes"], json!([]));
    assert_eq!(private_redirect["edges"], json!([]));

    let conversation_scope = json!({
        "namespace": "tenant-a",
        "workspace_id": "workspace-a",
        "graph_space": "facts"
    });
    let conversation_mutation = |id: &str, metadata: Value, document: Option<&str>| {
        let mut value = mutation(id, "gke_nodes", "node", metadata.clone(), document);
        value["event_id"] = json!(format!("conversation-event-{id}"));
        value["scope"] = conversation_scope.clone();
        value
    };
    let scoped_edge_mutation =
        |id: &str, metadata: Value, document: Option<&str>| {
            let mut value = mutation(id, "gke_edges", "edge", metadata, document);
            value["event_id"] = json!(format!("conversation-edge-event-{id}"));
            value["scope"] = conversation_scope.clone();
            value
        };
    let conversation_parent_edge = json!({
        "source_ids": [shared_entity_id],
        "target_ids": [],
        "source_edge_ids": [shared_entity_id],
        "target_edge_ids": [],
        "relation": "conversation cross-kind reference"
    });
    let workflow_parent_edge = json!({
        "source_ids": [shared_entity_id],
        "target_ids": [],
        "source_edge_ids": [shared_entity_id],
        "target_edge_ids": [],
        "relation": "workflow cross-kind reference"
    });
    let conversation_body = json!({
        "contract_version": 1,
        "atomic": true,
        "index_jobs": [],
        "mutations": [
            conversation_mutation(
                "conversation-node",
                json!({"label":"conversation only"}),
                Some("conversation only"),
            ),
            conversation_mutation(
                "conversation-acl",
                json!({
                    "entity_type":"acl_record",
                    "acl_truth_graph":"conversation",
                    "acl_target_grain":"node",
                    "acl_target_entity_id":"conversation-node",
                    "acl_version":1,
                    "acl_mode":"public"
                }),
                None,
            ),
            conversation_mutation(
                "conversation-acl-shared",
                json!({
                    "entity_type":"acl_record",
                    "acl_truth_graph":"conversation",
                    "acl_target_grain":"node",
                    "acl_target_entity_id":shared_entity_id,
                    "acl_version":1,
                    "acl_mode":conversation_policy["mode"],
                    "owner_id":conversation_policy["owner_id"]
                }),
                None,
            ),
            scoped_edge_mutation(
                shared_entity_id,
                json!({"source_ids":[],"target_ids":[],"relation":"same ID edge"}),
                Some(r#"{"source_ids":[],"target_ids":[],"relation":"same ID edge"}"#),
            ),
            scoped_edge_mutation(
                "conversation-private-edge",
                json!({"source_ids":[],"target_ids":[],"relation":"private conversation edge"}),
                Some(r#"{"source_ids":[],"target_ids":[],"relation":"private conversation edge"}"#),
            ),
            scoped_edge_mutation(
                "conversation-parent-edge",
                conversation_parent_edge.clone(),
                Some(&conversation_parent_edge.to_string()),
            ),
            scoped_edge_mutation(
                "workflow-parent-edge",
                workflow_parent_edge.clone(),
                Some(&workflow_parent_edge.to_string()),
            ),
            scoped_edge_mutation(
                "workflow-private-edge",
                json!({"source_ids":[],"target_ids":[],"relation":"private workflow edge"}),
                Some(r#"{"source_ids":[],"target_ids":[],"relation":"private workflow edge"}"#),
            ),
            conversation_mutation(
                "conversation-acl-cross-kind-edge",
                json!({
                    "entity_type":"acl_record",
                    "acl_truth_graph":"conversation",
                    "acl_target_grain":"edge",
                    "acl_target_entity_id":shared_entity_id,
                    "acl_version":1,
                    "acl_mode":conversation_policy["mode"],
                    "owner_id":conversation_policy["owner_id"]
                }),
                None,
            ),
            conversation_mutation(
                "conversation-acl-parent-edge",
                json!({
                    "entity_type":"acl_record",
                    "acl_truth_graph":"conversation",
                    "acl_target_grain":"edge",
                    "acl_target_entity_id":"conversation-parent-edge",
                    "acl_version":1,
                    "acl_mode":"public"
                }),
                None,
            ),
            conversation_mutation(
                "conversation-acl-private-edge",
                json!({
                    "entity_type":"acl_record",
                    "acl_truth_graph":"conversation",
                    "acl_target_grain":"edge",
                    "acl_target_entity_id":"conversation-private-edge",
                    "acl_version":1,
                    "acl_mode":conversation_policy["mode"],
                    "owner_id":conversation_policy["owner_id"]
                }),
                None,
            ),
            conversation_mutation(
                "workflow-acl-private-edge",
                json!({
                    "entity_type":"acl_record",
                    "acl_truth_graph":"workflow",
                    "acl_target_grain":"edge",
                    "acl_target_entity_id":"workflow-private-edge",
                    "acl_version":1,
                    "acl_mode":"private",
                    "owner_id":"workflow-owner"
                }),
                None,
            ),
        ]
    });
    let conversation_written = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/internal/graph/mutations")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!(
                        "Bearer {}",
                        token_for_graph_space("docs", &["write_graph"], "facts")
                    ),
                )
                .body(Body::from(serde_json::to_vec(&conversation_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(conversation_written.status(), StatusCode::OK);

    for case in isolation_cases
        .iter()
        .filter(|case| matches!(case["truth_graph"].as_str(), Some("workflow" | "wisdom")))
    {
        let truth_graph = case["truth_graph"].as_str().unwrap();
        let mut graph_acl = mutation(
            &format!("{truth_graph}-acl-shared"),
            "gke_nodes",
            "node",
            json!({
                "entity_type":"acl_record",
                "acl_truth_graph":truth_graph,
                "acl_target_grain":"node",
                "acl_target_entity_id":shared_entity_id,
                "acl_version":1,
                "acl_mode":case["mode"]
            }),
            None,
        );
        graph_acl["event_id"] = json!(format!("{truth_graph}-event-acl"));
        let mut graph_mutations = vec![graph_acl];
        if truth_graph == "workflow" {
            let mut edge_acl = mutation(
                "workflow-acl-cross-kind-edge",
                "gke_nodes",
                "node",
                json!({
                    "entity_type":"acl_record",
                    "acl_truth_graph":"workflow",
                    "acl_target_grain":"edge",
                    "acl_target_entity_id":shared_entity_id,
                    "acl_version":1,
                    "acl_mode":"public"
                }),
                None,
            );
            edge_acl["event_id"] = json!("workflow-event-acl-edge");
            graph_mutations.push(edge_acl);
            let mut parent_edge_acl = mutation(
                "workflow-acl-parent-edge",
                "gke_nodes",
                "node",
                json!({
                    "entity_type":"acl_record",
                    "acl_truth_graph":"workflow",
                    "acl_target_grain":"edge",
                    "acl_target_entity_id":"workflow-parent-edge",
                    "acl_version":1,
                    "acl_mode":"public"
                }),
                None,
            );
            parent_edge_acl["event_id"] = json!("workflow-event-acl-parent-edge");
            graph_mutations.push(parent_edge_acl);
        }
        let graph_body = json!({
            "contract_version":1,
            "atomic":true,
            "index_jobs":[],
            "mutations":graph_mutations
        });
        let written = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/graph/mutations")
                    .header("content-type", "application/json")
                    .header(
                        "authorization",
                        format!(
                            "Bearer {}",
                            token_for_graph_space("docs", &["write_graph"], "facts")
                        ),
                    )
                    .body(Body::from(serde_json::to_vec(&graph_body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(written.status(), StatusCode::OK, "graph={truth_graph}");
    }

    let visualization_request = |truth_graph: &str,
                                 graph_space: &str,
                                 security_scope: Option<&str>,
                                 principal_id: &str,
                                 groups: &[String]| {
        let mut principal = json!({
            "sub":principal_id, "role":"rw", "ns":["docs"],
            "storage_ns":"tenant-a", "workspace_id":"workspace-a",
            "graph_space":graph_space, "capabilities":["read_graph"]
        });
        principal["groups"] = json!(groups);
        if let Some(security_scope) = security_scope {
            principal["security_scope"] = json!(security_scope);
        }
        ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: INTERNAL_GRAPH_VIZ_D3_ROUTE.to_owned(),
            body: serde_json::to_vec(&json!({"graph_type":truth_graph})).unwrap(),
            principal,
        }
    };
    let conversation_visualization = application
        .graph_read_effect(&visualization_request("conversation", "facts", None, "writer", &[]))
        .await;
    assert_eq!(conversation_visualization.status, StatusCode::OK.as_u16());
    let conversation_visualization: Value =
        serde_json::from_slice(&conversation_visualization.body).unwrap();
    assert!(
        conversation_visualization["nodes"]
            .to_string()
            .contains("conversation-node"),
        "selected conversation visualization did not return its graph: {conversation_visualization}"
    );
    assert!(
        !conversation_visualization["nodes"]
            .to_string()
            .contains("n1"),
        "conversation visualization leaked knowledge graph data: {conversation_visualization}"
    );
    assert!(
        !conversation_visualization["nodes"]
            .to_string()
            .contains(shared_entity_id),
        "conversation visualization ignored its graph-scoped private ACL: {conversation_visualization}"
    );
    assert!(
        !conversation_visualization["nodes"]
            .to_string()
            .contains("conversation-private-edge"),
        "conversation visualization exposed a private edge to a non-owner: {conversation_visualization}"
    );

    let conversation_owner_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/viz/d3.json?graph_type=conversation")
                .header(
                    "authorization",
                    format!(
                        "Bearer {}",
                        token_for_subject(
                            conversation_policy["owner_id"].as_str().unwrap(),
                            "docs",
                            &["read_graph"],
                            "facts"
                        )
                    ),
                )
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(conversation_owner_response.status(), StatusCode::OK);
    let conversation_owner_payload: Value = serde_json::from_slice(
        &to_bytes(conversation_owner_response.into_body(), 65_536)
            .await
            .unwrap(),
    )
    .unwrap();
    assert!(
        conversation_owner_payload["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .any(|node| node["id"] == shared_entity_id),
        "conversation ACL owner must see its private graph node: {conversation_owner_payload}"
    );
    assert!(
        conversation_owner_payload["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .any(|node| node["id"] == "conversation-private-edge"),
        "conversation ACL owner must see its private graph edge: {conversation_owner_payload}"
    );

    let conversation_cross_kind_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/internal/graph/neighbors")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!(
                        "Bearer {}",
                        token_for_subject(
                            conversation_policy["owner_id"].as_str().unwrap(),
                            "docs",
                            &["read_graph"],
                            "facts"
                        )
                    ),
                )
                .body(Body::from(
                    json!({
                        "rid":"conversation-parent-edge",
                        "truth_graph":"conversation"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(conversation_cross_kind_response.status(), StatusCode::OK);
    let conversation_cross_kind: Value = serde_json::from_slice(
        &to_bytes(conversation_cross_kind_response.into_body(), 65_536)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(conversation_cross_kind["nodes"], json!([shared_entity_id]));
    assert_eq!(conversation_cross_kind["edges"], json!([shared_entity_id]));

    let workflow_cross_kind_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/internal/graph/neighbors")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!(
                        "Bearer {}",
                        token_for_subject("writer", "docs", &["read_graph"], "facts")
                    ),
                )
                .body(Body::from(
                    json!({
                        "rid":"workflow-parent-edge",
                        "truth_graph":"workflow"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(workflow_cross_kind_response.status(), StatusCode::OK);
    let workflow_cross_kind: Value = serde_json::from_slice(
        &to_bytes(workflow_cross_kind_response.into_body(), 65_536)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(workflow_cross_kind["nodes"], json!([shared_entity_id]));
    assert_eq!(workflow_cross_kind["edges"], json!([shared_entity_id]));

    for case in isolation_cases {
        let truth_graph = case["truth_graph"].as_str().unwrap();
        let response = application
            .graph_read_effect(&visualization_request(
                truth_graph,
                "facts",
                None,
                "writer",
                &[],
            ))
            .await;
        assert_eq!(
            response.status,
            StatusCode::OK.as_u16(),
            "graph={truth_graph}"
        );
        let payload: Value = serde_json::from_slice(&response.body).unwrap();
        let rendered_nodes = payload["nodes"].to_string();
        let expected_visible = case["expected_visible"].as_bool().unwrap();
        assert_eq!(
            rendered_nodes.contains(shared_entity_id),
            expected_visible,
            "selected graph={truth_graph} used the wrong ACL policy: {payload}"
        );
        if truth_graph == acl_fixture["truth_graph"].as_str().unwrap() {
            assert!(
                rendered_nodes.contains(public_edge_id),
                "a public edge with visible endpoints must appear in reified graph output: {payload}"
            );
            assert!(
                !rendered_nodes.contains(private_edge_id),
                "a non-owner must not see the private edge: {payload}"
            );
            assert!(
                !rendered_nodes.contains(scope_edge_id),
                "a principal without the matching scope must not see the scoped edge: {payload}"
            );
            for share_policy in &edge_share_policies {
                assert!(
                    !rendered_nodes.contains(share_policy["id"].as_str().unwrap()),
                    "an unshared principal must not see a shared/group edge: {payload}"
                );
            }
            assert!(
                !rendered_nodes.contains(owner_entity_id),
                "a non-owner must not see the private fixture node: {payload}"
            );
            assert!(
                !rendered_nodes.contains(private_entity_id),
                "a non-owner must not see another principal's private node: {payload}"
            );
            assert!(
                !rendered_nodes.contains(scope_entity_id),
                "a principal without the matching security scope must not see the scoped node: {payload}"
            );
            assert!(
                !rendered_nodes.contains(no_acl_node_id),
                "a node without an ACL record must be denied by default: {payload}"
            );
            assert!(
                !rendered_nodes.contains(no_acl_edge_id),
                "an edge without an ACL record must be denied by default: {payload}"
            );
            for share_policy in &share_policies {
                assert!(
                    !rendered_nodes.contains(share_policy["id"].as_str().unwrap()),
                    "an unshared principal must not see a shared/group node: {payload}"
                );
            }
        }
        if truth_graph == "workflow" || truth_graph == "wisdom" {
            assert!(
                !rendered_nodes.contains("n1") && !rendered_nodes.contains("conversation-node"),
                "selected graph={truth_graph} leaked records from another graph: {payload}"
            );
        }
        if truth_graph == "workflow" {
            assert!(
                !rendered_nodes.contains("workflow-private-edge"),
                "workflow visualization exposed a private edge to a non-owner: {payload}"
            );
        }
    }

    let workflow_owner_response = application
        .graph_read_effect(&visualization_request("workflow", "facts", None, "workflow-owner", &[]))
        .await;
    assert_eq!(workflow_owner_response.status, StatusCode::OK.as_u16());
    let workflow_owner_payload: Value =
        serde_json::from_slice(&workflow_owner_response.body).unwrap();
    assert!(
        workflow_owner_payload["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .any(|node| node["id"] == "workflow-private-edge"),
        "workflow ACL owner must see its private graph edge: {workflow_owner_payload}"
    );

    let owner_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/viz/d3.json?mode=reify")
                .header(
                    "authorization",
                    format!(
                        "Bearer {}",
                        token_for_subject(owner_id, "docs", &["read_graph"], "facts")
                    ),
                )
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let d3_contract: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/graph-viz-d3-http-success-v1.json"
    )))
    .unwrap();
    assert_eq!(
        owner_response.status().as_u16() as u64,
        d3_contract["success_status"].as_u64().unwrap()
    );
    let owner_payload: Value =
        serde_json::from_slice(&to_bytes(owner_response.into_body(), 65_536).await.unwrap())
            .unwrap();
    for field in d3_contract["required_fields"].as_array().unwrap() {
        let field = field.as_str().unwrap();
        assert!(owner_payload.get(field).is_some(), "missing {field}: {owner_payload}");
    }
    for field in d3_contract["array_fields"].as_array().unwrap() {
        let field = field.as_str().unwrap();
        assert!(owner_payload[field].is_array(), "{field} must be an array: {owner_payload}");
    }
    assert_eq!(owner_payload["mode"], d3_contract["mode"]);
    assert_eq!(owner_payload["doc_id"], d3_contract["doc_id"]);
    assert!(
        owner_payload["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .any(|node| node["id"] == owner_entity_id),
        "the ACL fixture owner must see the private node: {owner_payload}"
    );
    assert!(
        !owner_payload["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .any(|node| node["id"] == private_entity_id),
        "the fixture owner must not see another principal's private node: {owner_payload}"
    );
    assert!(
        !owner_payload["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .any(|node| node["id"] == private_edge_id),
        "Alice must not see Bob's private edge: {owner_payload}"
    );

    let private_edge_owner_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/viz/d3.json?graph_type=knowledge&mode=reify")
                .header(
                    "authorization",
                    format!(
                        "Bearer {}",
                        token_for_subject(private_edge_owner_id, "docs", &["read_graph"], "facts")
                    ),
                )
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(private_edge_owner_response.status(), StatusCode::OK);
    let private_edge_owner_payload: Value = serde_json::from_slice(
        &to_bytes(private_edge_owner_response.into_body(), 65_536)
            .await
            .unwrap(),
    )
    .unwrap();
    assert!(
        private_edge_owner_payload["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .any(|node| node["id"] == private_edge_id),
        "Bob must see the edge he owns: {private_edge_owner_payload}"
    );

    for (requested_scope, expected_visible) in [(scope_value, true), ("team-b", false)] {
        let scoped_response = application
            .graph_read_effect(&visualization_request(
                acl_fixture["truth_graph"].as_str().unwrap(),
                "facts",
                Some(requested_scope),
                "writer",
                &[],
            ))
            .await;
        assert_eq!(scoped_response.status, StatusCode::OK.as_u16());
        let scoped_payload: Value = serde_json::from_slice(&scoped_response.body).unwrap();
        assert_eq!(
            scoped_payload["nodes"]
                .as_array()
                .unwrap()
                .iter()
                .any(|node| node["id"] == scope_entity_id),
            expected_visible,
            "security scope {requested_scope} had incorrect visibility: {scoped_payload}"
        );
        assert_eq!(
            scoped_payload["nodes"]
                .as_array()
                .unwrap()
                .iter()
                .any(|node| node["id"] == scope_edge_id),
            expected_visible,
            "security scope {requested_scope} had incorrect edge visibility: {scoped_payload}"
        );
    }

    for (groups, expected_group_nodes) in [(&[][..], false), (fixture_groups.as_slice(), true)] {
        let shared_response = application
            .graph_read_effect(&visualization_request(
                acl_fixture["truth_graph"].as_str().unwrap(),
                "facts",
                Some(scope_value),
                fixture_principal_id,
                groups,
            ))
            .await;
        assert_eq!(shared_response.status, StatusCode::OK.as_u16());
        let shared_payload: Value = serde_json::from_slice(&shared_response.body).unwrap();
        let visible_nodes = shared_payload["nodes"].to_string();
        for share_policy in &share_policies {
            let should_be_visible = share_policy["mode"] == "shared" || expected_group_nodes;
            assert_eq!(
                visible_nodes.contains(share_policy["id"].as_str().unwrap()),
                should_be_visible,
                "principal groups {groups:?} had incorrect share visibility: {shared_payload}"
            );
        }
        for share_policy in &edge_share_policies {
            let should_be_visible = share_policy["mode"] == "shared" || expected_group_nodes;
            assert_eq!(
                visible_nodes.contains(share_policy["id"].as_str().unwrap()),
                should_be_visible,
                "principal groups {groups:?} had incorrect edge-share visibility: {shared_payload}"
            );
        }
    }

    let deferred_visualization = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/viz/d3.json?graph_type=conversation")
                .header(
                    "authorization",
                    format!(
                        "Bearer {}",
                        token_for_graph_space("docs", &["read_graph"], "facts")
                    ),
                )
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(deferred_visualization.status(), StatusCode::OK);
    let visualization: Value = serde_json::from_slice(
        &to_bytes(deferred_visualization.into_body(), 65_536)
            .await
            .unwrap(),
    )
    .unwrap();
    assert!(
        visualization["nodes"]
            .to_string()
            .contains("conversation-node"),
        "public visualization failed to preserve graph and insertion-method selection: {visualization}"
    );

    let cytoscape_contract: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/graph-viz-cytoscape-http-success-v1.json"
    )))
    .unwrap();
    for (path, element_path) in [
        ("/api/viz/d3.json?graph_type=knowledge&mode=reify", "nodes"),
        (
            "/api/viz/cytoscape.json?graph_type=knowledge&mode=reify",
            "elements",
        ),
    ] {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri(path)
                    .header(
                        "authorization",
                        format!("Bearer {}", token("docs", &["read_graph"])),
                    )
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        if path.starts_with("/api/viz/cytoscape.json") {
            assert_eq!(
                response.status().as_u16(),
                cytoscape_contract["success_status"].as_u64().unwrap() as u16,
                "route {path}"
            );
        } else {
            assert_eq!(response.status(), StatusCode::OK, "route {path}");
        }
        let visualization: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 65_536).await.unwrap()).unwrap();
        if path.starts_with("/api/viz/cytoscape.json") {
            for field in cytoscape_contract["required_fields"]
                .as_array()
                .unwrap()
            {
                let field = field.as_str().unwrap();
                assert!(visualization.get(field).is_some(), "missing {field}");
            }
            for field in cytoscape_contract["array_fields"].as_array().unwrap() {
                let field = field.as_str().unwrap();
                assert!(visualization[field].is_array(), "{field} must be an array");
            }
            assert_eq!(visualization["mode"], cytoscape_contract["mode"]);
            if cytoscape_contract["doc_id_query_echo"] == true {
                assert!(visualization["doc_id"].is_null());
            }
        }
        let ids = visualization[element_path]
            .as_array()
            .unwrap()
            .iter()
            .map(|item| {
                if element_path == "elements" {
                    item["data"]["id"].as_str().unwrap()
                } else {
                    item["id"].as_str().unwrap()
                }
            })
            .collect::<Vec<_>>();
        assert!(ids.contains(&shared_entity_id), "{path}: {ids:?}");
        assert!(ids.iter().any(|id| id.starts_with("n1")), "{path}: {ids:?}");
        assert!(ids.iter().any(|id| id.starts_with("n2")), "{path}: {ids:?}");
        assert!(ids.iter().any(|id| id.starts_with("e1")), "{path}: {ids:?}");
        assert!(
            ids.iter().all(|id| !id.contains("n3")),
            "private graph records must not be exposed by {path}: {ids:?}"
        );
    }

    let bundle = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/viz/d3.bundle?graph_type=knowledge&mode=reify")
                .header(
                    "authorization",
                    format!("Bearer {}", token("docs", &["read_graph"])),
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
        .expect("bundle includes valid embedded graph data");
    let embedded_node_ids = embedded["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|node| node["id"].as_str())
        .collect::<Vec<_>>();
    assert!(embedded_node_ids.contains(&"n1"));
    assert!(embedded_node_ids.contains(&"n2"));
    assert!(embedded_node_ids.contains(&"e1"));
    assert!(!embedded_node_ids.contains(&"n3"));

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

    let standard = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp")
                .header("content-type", "application/json")
                .header("accept", "application/json, text/event-stream")
                .header("authorization", format!("Bearer {}", token("docs", &["read_graph"])))
                .body(Body::from(
                    r#"{"jsonrpc":"2.0","id":"neighbors-standard","method":"tools/call","params":{"name":"kg_neighbors","arguments":{"rid":"n1","truth_graph":"knowledge"}}}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(standard.status(), StatusCode::OK);
    assert_eq!(
        standard
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok()),
        Some("text/event-stream; charset=utf-8")
    );
    let body = to_bytes(standard.into_body(), 65_536).await.unwrap();
    let event = std::str::from_utf8(&body)
        .unwrap()
        .lines()
        .find_map(|line| line.strip_prefix("data: "))
        .expect("standard MCP response has an SSE data frame");
    let standard: Value = serde_json::from_str(event).unwrap();
    assert_eq!(
        standard["result"]["structuredContent"]["nodes"],
        json!(["n2"])
    );
    assert_eq!(
        standard["result"]["structuredContent"]["edges"],
        json!(["e1"])
    );
    assert_eq!(
        serde_json::from_str::<Value>(standard["result"]["content"][0]["text"].as_str().unwrap())
            .unwrap(),
        standard["result"]["structuredContent"]
    );

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

    let proposal_fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/graph-vector-proposal-acl-v1.json"
    )))
    .unwrap();
    let proposals = app
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
                        "id":"propose-vector",
                        "method":"tools/call",
                        "params":{
                            "name":proposal_fixture["tool"],
                            "arguments":proposal_fixture["mcp_arguments"]
                        }
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    let unauthorized_proposal = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!("Bearer {}", token("docs", &[])),
                )
                .body(Body::from(
                    json!({
                        "jsonrpc":"2.0",
                        "id":"propose-vector-denied",
                        "method":"tools/call",
                        "params":{
                            "name":proposal_fixture["tool"],
                            "arguments":{"inp":proposal_fixture["request"]}
                        }
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(unauthorized_proposal.status(), StatusCode::OK);
    let unauthorized_proposal: Value = serde_json::from_slice(
        &to_bytes(unauthorized_proposal.into_body(), 65_536)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        unauthorized_proposal["error"]["code"],
        json!(-32602)
    );
    assert!(unauthorized_proposal["error"]["message"]
        .as_str()
        .unwrap()
        .contains("not advertised"));
    assert_eq!(proposals.status(), StatusCode::OK);
    let proposals: Value =
        serde_json::from_slice(&to_bytes(proposals.into_body(), 65_536).await.unwrap()).unwrap();
    assert!(proposals.get("error").is_none(), "{proposals}");
    assert_eq!(proposals["result"]["implementation_authority"], "rust");
    assert_eq!(proposals["result"]["seam_crossings"], json!([]));
    assert_eq!(
        proposals["result"]["pairs"],
        proposal_fixture["expected_result"]["pairs"]
    );

    let flat_proposals = app
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
                        "id":"propose-vector-legacy-flat",
                        "method":"tools/call",
                        "params":{
                            "name":proposal_fixture["tool"],
                            "arguments":proposal_fixture["request"]
                        }
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(flat_proposals.status(), StatusCode::OK);
    let flat_proposals: Value = serde_json::from_slice(
        &to_bytes(flat_proposals.into_body(), 65_536).await.unwrap(),
    )
    .unwrap();
    assert!(flat_proposals.get("error").is_none(), "{flat_proposals}");
    assert_eq!(
        flat_proposals["result"]["pairs"],
        proposal_fixture["expected_result"]["pairs"]
    );

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
    // The nearest ACL-denied node consumes a top-K slot; Rust matches Python
    // by filtering it without backfilling from farther candidates.
    assert_eq!(vector_query["ids"], json!([["n1"], ["n2"]]));
    assert_eq!(vector_query["distances"][0].as_array().unwrap().len(), 1);
    assert_eq!(vector_query["distances"][1].as_array().unwrap().len(), 1);
    assert!(vector_query.get("documents").is_none());
}

// PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/tests/postgres_visualization.rs#postgres_visualization_read_matches_shared_python_selection_and_acl; python=kogwistar/server_mcp_with_admin.py#api_viz_d3; fixture=contracts/golden/graph-acl-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/postgres_visualization.rs#postgres_visualization_read_matches_shared_python_selection_and_acl; python_test=tests/test_visualization.py#test_visualization_uses_acl_policy_for_selected_python_graph_type
