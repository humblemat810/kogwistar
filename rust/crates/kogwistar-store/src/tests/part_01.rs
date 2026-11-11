#[test]
fn graph_scope_metadata_rejects_unrequested_workspace_and_graph_space() {
    let scope = GraphScope {
        namespace: "tenant-a".to_owned(),
        workspace_id: None,
        graph_space: None,
    };
    assert!(
        scope
            .validate_metadata_scope(&Map::from_iter([(
                "workspace_id".to_owned(),
                Value::String("workspace-a".to_owned()),
            )]))
            .is_err()
    );
    assert!(
        scope
            .validate_metadata_scope(&Map::from_iter([(
                "graph_space".to_owned(),
                Value::String("conversation".to_owned()),
            )]))
            .is_err()
    );
    assert!(
        scope
            .validate_metadata_scope(&Map::from_iter([("workspace_id".to_owned(), Value::Null,)]))
            .is_ok()
    );
}

#[test]
fn standardized_vector_distance_scores_are_backend_neutral() {
    assert_eq!(
        similarity_from_distance(Some(0.25), DistanceMetric::Cosine),
        Some(0.75)
    );
    assert_eq!(
        similarity_from_distance(Some(3.0), DistanceMetric::L2),
        Some(0.25)
    );
    assert_eq!(
        similarity_from_distance(Some(-0.8), DistanceMetric::InnerProduct),
        Some(0.8)
    );
    assert_eq!(similarity_from_distance(None, DistanceMetric::Cosine), None);
    assert_eq!(
        similarity_from_distance(Some(f64::NAN), DistanceMetric::Cosine),
        None
    );
}

fn edge_record(id: &str, document: Value) -> GraphRecord {
    GraphRecord {
        id: id.to_owned(),
        document: Some(document.to_string()),
        metadata: Map::new(),
        embedding: None,
    }
}

fn node_record(id: &str) -> GraphRecord {
    GraphRecord::new(id)
}

// PYTHON_PARITY_V1: rust=rust/crates/kogwistar-store/src/lib.rs#graph_viz_d3_from_records; python=kogwistar/visualization/graph_viz.py#to_d3_force; fixture=contracts/golden/python-graph-model-payloads-v1.json; rust_test=rust/crates/kogwistar-store/src/lib.rs#graph_visualization_model_payloads_match_python_fixture; python_test=tests/test_visualization.py#test_graph_visualization_payloads_match_shared_python_model_fixture
#[test]
fn graph_visualization_model_payloads_match_python_fixture() {
    let fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/python-graph-model-payloads-v1.json"
    )))
    .expect("Python graph-model fixture is valid JSON");
    let payload = |kind: &str| {
        fixture["cases"]
            .as_array()
            .expect("fixture cases")
            .iter()
            .find(|case| case["kind"] == kind)
            .expect("fixture graph model")["payload"]
            .clone()
    };
    let source = payload("node");
    let mut target = source.clone();
    target["id"] = json!("parity-target");
    target["label"] = json!("Target node");
    let mut edge = payload("edge");
    edge["source_ids"] = json!([source["id"]]);
    edge["target_ids"] = json!([target["id"]]);

    let record = |payload: &Value| GraphRecord {
        id: payload["id"].as_str().expect("model ID").to_owned(),
        document: Some(payload.to_string()),
        metadata: payload["metadata"].as_object().cloned().unwrap_or_default(),
        embedding: None,
    };
    let nodes = vec![record(&source), record(&target)];
    let edges = vec![record(&edge)];
    let d3 = graph_viz_d3_from_records(&nodes, &edges, None, "reify", None);

    let mut expected_nodes = vec![source.clone(), target.clone(), edge.clone()];
    for node in &mut expected_nodes[..2] {
        node["type"] = json!("entity");
        node["properties"] = node["properties"]
            .as_object()
            .cloned()
            .map(Value::Object)
            .unwrap_or_else(|| json!({}));
    }
    expected_nodes[2]["type"] = json!("edge-node");
    expected_nodes[2]["label"] = edge["relation"].clone();
    expected_nodes.sort_by_key(|node| node["id"].as_str().unwrap_or_default().to_owned());
    let mut actual_nodes = d3["nodes"].as_array().unwrap().clone();
    actual_nodes.sort_by_key(|node| node["id"].as_str().unwrap_or_default().to_owned());
    assert_eq!(actual_nodes, expected_nodes);

    let mut actual_links = d3["links"].as_array().unwrap().clone();
    actual_links.sort_by_key(|link| {
        (
            link["source"].as_str().unwrap_or_default().to_owned(),
            link["target"].as_str().unwrap_or_default().to_owned(),
        )
    });
    let mut expected_links = vec![
        json!({"source":source["id"],"target":edge["id"],"relation":edge["relation"],"role":"src","properties":edge["properties"]}),
        json!({"source":edge["id"],"target":target["id"],"relation":edge["relation"],"role":"tgt","properties":edge["properties"]}),
    ];
    expected_links.sort_by_key(|link| {
        (
            link["source"].as_str().unwrap_or_default().to_owned(),
            link["target"].as_str().unwrap_or_default().to_owned(),
        )
    });
    assert_eq!(actual_links, expected_links);

    let classic_d3 = graph_viz_d3_from_records(&nodes, &edges, None, "classic", None);
    let mut expected_classic_nodes = vec![source.clone(), target.clone()];
    for node in &mut expected_classic_nodes {
        node["type"] = json!("entity");
        node["properties"] = node["properties"]
            .as_object()
            .cloned()
            .map(Value::Object)
            .unwrap_or_else(|| json!({}));
    }
    expected_classic_nodes.sort_by_key(|node| node["id"].as_str().unwrap_or_default().to_owned());
    let mut actual_classic_nodes = classic_d3["nodes"].as_array().unwrap().clone();
    actual_classic_nodes.sort_by_key(|node| node["id"].as_str().unwrap_or_default().to_owned());
    assert_eq!(actual_classic_nodes, expected_classic_nodes);
    assert_eq!(
        classic_d3["links"],
        json!([{
            "source": source["id"],
            "target": target["id"],
            "relation": edge["relation"],
            "properties": edge["properties"]
        }])
    );

    let cytoscape = graph_viz_cytoscape_from_records(&nodes, &edges, None, "reify", None);
    let mut expected_elements = vec![
        json!({"data":{"id":source["id"],"label":source["label"],"type":"entity"},"classes":""}),
        json!({"data":{"id":target["id"],"label":target["label"],"type":"entity"},"classes":""}),
        json!({"data":{"id":edge["id"],"label":edge["relation"],"type":"edge-node"},"classes":"edge-node"}),
        json!({"data":{"id":format!("{}::src::{}",edge["id"].as_str().unwrap(),source["id"].as_str().unwrap()),"source":source["id"],"target":edge["id"],"label":edge["relation"]},"classes":"src"}),
        json!({"data":{"id":format!("{}::tgt::{}",edge["id"].as_str().unwrap(),target["id"].as_str().unwrap()),"source":edge["id"],"target":target["id"],"label":edge["relation"]},"classes":"tgt"}),
    ];
    expected_elements.sort_by_key(|element| {
        element["data"]["id"]
            .as_str()
            .unwrap_or_default()
            .to_owned()
    });
    let mut actual_elements = cytoscape["elements"].as_array().unwrap().clone();
    actual_elements.sort_by_key(|element| {
        element["data"]["id"]
            .as_str()
            .unwrap_or_default()
            .to_owned()
    });
    assert_eq!(actual_elements, expected_elements);
    assert_eq!(cytoscape["mode"], "reify");
    assert!(cytoscape["doc_id"].is_null());

    let classic_cytoscape = graph_viz_cytoscape_from_records(&nodes, &edges, None, "classic", None);
    assert_eq!(classic_cytoscape["elements"].as_array().unwrap().len(), 3);
    let expected_classic_edge_id = format!(
        "{}::{}->{}",
        edge["id"].as_str().unwrap(),
        source["id"].as_str().unwrap(),
        target["id"].as_str().unwrap()
    );
    assert!(
        classic_cytoscape["elements"]
            .as_array()
            .unwrap()
            .iter()
            .any(|element| {
                element["data"]["id"].as_str() == Some(expected_classic_edge_id.as_str())
            })
    );
    assert_eq!(classic_cytoscape["mode"], "classic");

    let incomplete_edge = record(&payload("edge"));
    let incomplete_nodes = vec![record(&source)];
    let incomplete_edges = vec![incomplete_edge];
    let incomplete_d3 =
        graph_viz_d3_from_records(&incomplete_nodes, &incomplete_edges, None, "reify", None);
    assert_eq!(incomplete_d3["nodes"].as_array().unwrap().len(), 1);
    assert!(incomplete_d3["links"].as_array().unwrap().is_empty());
    let incomplete_cytoscape =
        graph_viz_cytoscape_from_records(&incomplete_nodes, &incomplete_edges, None, "reify", None);
    assert_eq!(
        incomplete_cytoscape["elements"].as_array().unwrap().len(),
        1
    );
}

// PYTHON_PARITY_V1: rust=rust/crates/kogwistar-store/src/lib.rs#graph_visualization_edge_label_fallback_matches_python_contract; python=kogwistar/visualization/graph_viz.py#to_d3_force; fixture=contracts/golden/visualization-label-fallback-v1.json; rust_test=rust/crates/kogwistar-store/src/lib.rs#graph_visualization_edge_label_fallback_matches_python_contract; python_test=tests/test_visualization.py#test_visualization_edge_label_fallback_matches_shared_contract
#[test]
fn graph_visualization_edge_label_fallback_matches_python_contract() {
    let contract: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/visualization-label-fallback-v1.json"
    )))
    .expect("visualization label fallback contract is valid JSON");
    let model_fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/python-graph-model-payloads-v1.json"
    )))
    .expect("Python graph model fixture is valid JSON");
    let model_payload = |kind: &str| {
        model_fixture["cases"]
            .as_array()
            .expect("model fixture cases")
            .iter()
            .find(|case| case["kind"] == kind)
            .expect("graph model fixture case")["payload"]
            .clone()
    };
    let record = |payload: &Value| GraphRecord {
        id: payload["id"].as_str().expect("record ID").to_owned(),
        document: Some(payload.to_string()),
        metadata: payload["metadata"].as_object().cloned().unwrap_or_default(),
        embedding: None,
    };
    let source = model_payload("node");
    let mut target = source.clone();
    target["id"] = json!("target");
    let nodes = vec![record(&source), record(&target)];
    for case in contract["cases"].as_array().expect("contract cases") {
        let mut edge = model_payload("edge");
        edge["id"] = case["name"].clone();
        edge["relation"] = case["relation"].clone();
        edge["label"] = case["label"].clone();
        edge["source_ids"] = json!([source["id"]]);
        edge["target_ids"] = json!([target["id"]]);
        let edges = vec![record(&edge)];
        let expected_label = &case["expected"];

        let d3 = graph_viz_d3_from_records(&nodes, &edges, None, "reify", None);
        assert!(
            d3["nodes"]
                .as_array()
                .unwrap()
                .iter()
                .any(|node| { node["id"] == edge["id"] && &node["label"] == expected_label })
        );
        assert!(
            d3["links"]
                .as_array()
                .unwrap()
                .iter()
                .all(|link| { &link["relation"] == expected_label })
        );

        let classic = graph_viz_d3_from_records(&nodes, &edges, None, "classic", None);
        assert_eq!(classic["links"][0]["relation"], *expected_label);

        let reified_cytoscape =
            graph_viz_cytoscape_from_records(&nodes, &edges, None, "reify", None);
        assert!(
            reified_cytoscape["elements"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|element| {
                    element["data"]["source"] == edge["id"]
                        || element["data"]["target"] == edge["id"]
                })
                .all(|element| &element["data"]["label"] == expected_label)
        );

        let classic_cytoscape =
            graph_viz_cytoscape_from_records(&nodes, &edges, None, "classic", None);
        assert!(
            classic_cytoscape["elements"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|element| element["data"]["source"] == source["id"])
                .all(|element| &element["data"]["label"] == expected_label)
        );
    }
}

// PYTHON_PARITY_V1: rust=rust/crates/kogwistar-store/src/lib.rs#graph_visualization_empty_doc_id_matches_python_contract; python=kogwistar/visualization/graph_viz.py#_ids_by_doc; fixture=contracts/golden/visualization-doc-filter-v1.json; rust_test=rust/crates/kogwistar-store/src/lib.rs#graph_visualization_empty_doc_id_matches_python_contract; python_test=tests/test_visualization.py#test_visualization_empty_doc_id_matches_python_route_selection_contract
#[test]
fn graph_visualization_empty_doc_id_matches_python_contract() {
    let contract: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/visualization-doc-filter-v1.json"
    )))
    .expect("visualization doc filter contract is valid JSON");
    let model_fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/python-graph-model-payloads-v1.json"
    )))
    .expect("Python graph model fixture is valid JSON");
    let node_template = model_fixture["cases"]
        .as_array()
        .expect("model fixture cases")
        .iter()
        .find(|case| case["kind"] == "node")
        .expect("node model fixture")["payload"]
        .clone();
    let edge_template = model_fixture["cases"]
        .as_array()
        .expect("model fixture cases")
        .iter()
        .find(|case| case["kind"] == "edge")
        .expect("edge model fixture")["payload"]
        .clone();
    let record = |payload: &Value| GraphRecord {
        id: payload["id"].as_str().expect("record ID").to_owned(),
        document: Some(payload.to_string()),
        metadata: payload["metadata"].as_object().cloned().unwrap_or_default(),
        embedding: None,
    };
    let mut first_node = node_template.clone();
    first_node["id"] = json!("node-a");
    first_node["doc_id"] = json!("doc-a");
    first_node["metadata"]["doc_id"] = json!("doc-a");
    let mut second_node = node_template;
    second_node["id"] = json!("node-b");
    second_node["doc_id"] = json!("doc-b");
    second_node["metadata"]["doc_id"] = json!("doc-b");
    let nodes = vec![record(&first_node), record(&second_node)];
    let mut first_edge = edge_template.clone();
    first_edge["id"] = json!("edge-a");
    first_edge["doc_id"] = json!("doc-a");
    first_edge["metadata"]["doc_id"] = json!("doc-a");
    first_edge["source_ids"] = json!(["node-a"]);
    first_edge["target_ids"] = json!(["node-a"]);
    let mut second_edge = edge_template;
    second_edge["id"] = json!("edge-b");
    second_edge["doc_id"] = json!("doc-b");
    second_edge["metadata"]["doc_id"] = json!("doc-b");
    second_edge["source_ids"] = json!(["node-b"]);
    second_edge["target_ids"] = json!(["node-b"]);
    let edges = vec![record(&first_edge), record(&second_edge)];

    for case in contract["cases"].as_array().expect("contract cases") {
        let doc_id = case["doc_id"].as_str();
        let payload = graph_viz_d3_from_records(&nodes, &edges, doc_id, "reify", None);
        assert_eq!(payload["doc_id"], case["response_doc_id"]);
        let mut actual_doc_ids = payload["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|node| node["type"] == "entity")
            .map(|node| node["doc_id"].as_str().unwrap().to_owned())
            .collect::<Vec<_>>();
        actual_doc_ids.sort();
        let mut expected_doc_ids = case["selected_doc_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|doc_id| doc_id.as_str().unwrap().to_owned())
            .collect::<Vec<_>>();
        expected_doc_ids.sort();
        assert_eq!(actual_doc_ids, expected_doc_ids, "case {}", case["name"]);

        let mut actual_edge_doc_ids = payload["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|node| node["type"] == "edge-node")
            .map(|node| node["doc_id"].as_str().unwrap().to_owned())
            .collect::<Vec<_>>();
        actual_edge_doc_ids.sort();
        let mut expected_edge_doc_ids = case["selected_edge_doc_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|doc_id| doc_id.as_str().unwrap().to_owned())
            .collect::<Vec<_>>();
        expected_edge_doc_ids.sort();
        assert_eq!(
            actual_edge_doc_ids, expected_edge_doc_ids,
            "edge selection case {}",
            case["name"]
        );
    }
}

// PYTHON_PARITY_V1: rust=rust/crates/kogwistar-store/src/lib.rs#graph_visualization_hyperedges_match_python_contract; python=kogwistar/visualization/graph_viz.py#to_d3_force; fixture=contracts/golden/visualization-hyperedge-v1.json; rust_test=rust/crates/kogwistar-store/src/lib.rs#graph_visualization_hyperedges_match_python_contract; python_test=tests/test_visualization.py#test_hyperedge_visualization_matches_shared_python_contract
#[test]
fn graph_visualization_hyperedges_match_python_contract() {
    let fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/visualization-hyperedge-v1.json"
    )))
    .expect("visualization hyperedge contract is valid JSON");
    let records = |field: &str| {
        fixture[field]
            .as_array()
            .expect("fixture graph records")
            .iter()
            .map(|payload| GraphRecord {
                id: payload["id"].as_str().expect("record ID").to_owned(),
                document: Some(payload.to_string()),
                metadata: Map::new(),
                embedding: None,
            })
            .collect::<Vec<_>>()
    };
    let nodes = records("nodes");
    let edges = records("edges");
    let expected = &fixture["expected"];

    let d3 = graph_viz_d3_from_records(&nodes, &edges, None, "reify", None);
    let actual_node_ids = d3["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|node| node["id"].as_str().unwrap())
        .collect::<Vec<_>>();
    let expected_node_ids = expected["reified_d3_node_order"]
        .as_array()
        .unwrap()
        .iter()
        .map(|id| id.as_str().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(actual_node_ids, expected_node_ids);
    let actual_reified_links = d3["links"]
        .as_array()
        .unwrap()
        .iter()
        .map(|link| {
            (
                link["source"].as_str().unwrap(),
                link["target"].as_str().unwrap(),
                link["role"].as_str().unwrap(),
                link["relation"].as_str().unwrap(),
            )
        })
        .collect::<BTreeSet<_>>();
    let expected_reified_links = expected["reified_d3_links"]
        .as_array()
        .unwrap()
        .iter()
        .map(|link| {
            (
                link["source"].as_str().unwrap(),
                link["target"].as_str().unwrap(),
                link["role"].as_str().unwrap(),
                link["relation"].as_str().unwrap(),
            )
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(actual_reified_links, expected_reified_links);

    let classic_d3 = graph_viz_d3_from_records(&nodes, &edges, None, "classic", None);
    assert_eq!(
        classic_d3["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|node| node["id"].as_str().unwrap())
            .collect::<Vec<_>>(),
        expected["classic_d3_node_order"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_str().unwrap())
            .collect::<Vec<_>>()
    );
    let actual_classic_links = classic_d3["links"]
        .as_array()
        .unwrap()
        .iter()
        .map(|link| {
            (
                link["source"].as_str().unwrap(),
                link["target"].as_str().unwrap(),
                link["relation"].as_str().unwrap(),
            )
        })
        .collect::<BTreeSet<_>>();
    let expected_classic_links = expected["classic_d3_links"]
        .as_array()
        .unwrap()
        .iter()
        .map(|link| {
            (
                link["source"].as_str().unwrap(),
                link["target"].as_str().unwrap(),
                link["relation"].as_str().unwrap(),
            )
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(actual_classic_links, expected_classic_links);

    let cytoscape_links =
        |mode: &str| graph_viz_cytoscape_from_records(&nodes, &edges, None, mode, None);
    let reified_cytoscape = cytoscape_links("reify");
    assert_eq!(
        reified_cytoscape["elements"]
            .as_array()
            .unwrap()
            .iter()
            .map(|element| element["data"]["id"].as_str().unwrap())
            .collect::<Vec<_>>(),
        expected["reified_cytoscape_element_order"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_str().unwrap())
            .collect::<Vec<_>>()
    );
    let actual_reified_cytoscape_links = reified_cytoscape["elements"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|element| matches!(element["classes"].as_str(), Some("src" | "tgt")))
        .map(|element| {
            (
                element["data"]["id"].as_str().unwrap(),
                element["data"]["source"].as_str().unwrap(),
                element["data"]["target"].as_str().unwrap(),
                element["data"]["label"].as_str().unwrap(),
                element["classes"].as_str().unwrap(),
            )
        })
        .collect::<BTreeSet<_>>();
    let expected_reified_cytoscape_links = expected["reified_cytoscape_links"]
        .as_array()
        .unwrap()
        .iter()
        .map(|element| {
            (
                element["id"].as_str().unwrap(),
                element["source"].as_str().unwrap(),
                element["target"].as_str().unwrap(),
                element["label"].as_str().unwrap(),
                element["class"].as_str().unwrap(),
            )
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(
        actual_reified_cytoscape_links,
        expected_reified_cytoscape_links
    );

    let classic_cytoscape = cytoscape_links("classic");
    assert_eq!(
        classic_cytoscape["elements"]
            .as_array()
            .unwrap()
            .iter()
            .map(|element| element["data"]["id"].as_str().unwrap())
            .collect::<Vec<_>>(),
        expected["classic_cytoscape_element_order"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_str().unwrap())
            .collect::<Vec<_>>()
    );
    let actual_classic_cytoscape_links = classic_cytoscape["elements"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|element| element["data"].get("source").is_some())
        .map(|element| {
            (
                element["data"]["id"].as_str().unwrap(),
                element["data"]["source"].as_str().unwrap(),
                element["data"]["target"].as_str().unwrap(),
                element["data"]["label"].as_str().unwrap(),
                element["classes"].as_str().unwrap(),
            )
        })
        .collect::<BTreeSet<_>>();
    let expected_classic_cytoscape_links = expected["classic_cytoscape_links"]
        .as_array()
        .unwrap()
        .iter()
        .map(|element| {
            (
                element["id"].as_str().unwrap(),
                element["source"].as_str().unwrap(),
                element["target"].as_str().unwrap(),
                element["label"].as_str().unwrap(),
                element["class"].as_str().unwrap(),
            )
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(
        actual_classic_cytoscape_links,
        expected_classic_cytoscape_links
    );
}

#[test]
fn document_authority_capability_is_explicit_and_fail_closed() {
    assert_eq!(
        DocumentAuthorityCapability::default(),
        DocumentAuthorityCapability::Unsupported
    );
    assert_eq!(
        DocumentAuthorityCapability::Unsupported.as_str(),
        "unsupported"
    );
    assert!(!DocumentAuthorityCapability::Unsupported.supports_projection_upsert());
    assert!(DocumentAuthorityCapability::ProjectionUpsert.supports_projection_upsert());
}

#[test]
fn graph_neighbors_match_node_and_edge_hypergraph_semantics() {
    let edges = vec![
        edge_record(
            "e0",
            serde_json::json!({"source_ids": [], "target_ids": []}),
        ),
        edge_record(
            "e1",
            serde_json::json!({
                "source_ids": ["n1", "n2"],
                "target_ids": ["n3"],
                "source_edge_ids": ["e0"],
                "target_edge_ids": ["e2"]
            }),
        ),
        edge_record(
            "e2",
            serde_json::json!({"source_ids": [], "target_ids": []}),
        ),
    ];
    let nodes = vec![node_record("n1"), node_record("n2"), node_record("n3")];

    assert_eq!(
        graph_neighbors_from_records("n1", &nodes, &edges, None),
        GraphNeighbors {
            nodes: BTreeSet::from(["n2".to_owned(), "n3".to_owned()]),
            edges: BTreeSet::from(["e1".to_owned()]),
        }
    );
    assert_eq!(
        graph_neighbors_from_records("e1", &nodes, &edges, None),
        GraphNeighbors {
            nodes: BTreeSet::from(["n1".to_owned(), "n2".to_owned(), "n3".to_owned()]),
            edges: BTreeSet::from(["e0".to_owned(), "e2".to_owned()]),
        }
    );
}

#[test]
fn node_redirect_resolution_matches_python_lifecycle_modes() {
    let mut redirected = node_record("old");
    redirected
        .metadata
        .insert("lifecycle_status".to_owned(), json!("tombstoned"));
    redirected
        .metadata
        .insert("redirect_to_id".to_owned(), json!("middle"));
    let mut middle = node_record("middle");
    middle
        .metadata
        .insert("lifecycle_status".to_owned(), json!("tombstoned"));
    middle
        .metadata
        .insert("redirect_to_id".to_owned(), json!("current"));
    let nodes = vec![redirected.clone(), middle, node_record("current")];

    assert_eq!(
        resolve_graph_node_redirect_id("old", &nodes),
        Some("current".into())
    );
    assert_eq!(
        resolve_graph_node_redirect_id("current", &nodes),
        Some("current".into())
    );
    assert_eq!(resolve_graph_node_redirect_id("missing", &nodes), None);

    let mut tombstone = node_record("gone");
    tombstone
        .metadata
        .insert("lifecycle_status".to_owned(), json!("tombstoned"));
    let mut broken = redirected;
    broken
        .metadata
        .insert("redirect_to_id".to_owned(), json!("missing"));
    assert_eq!(resolve_graph_node_redirect_id("gone", &[tombstone]), None);
    assert_eq!(resolve_graph_node_redirect_id("old", &[broken]), None);

    let mut cycle_a = node_record("cycle-a");
    cycle_a
        .metadata
        .insert("redirect_to_id".to_owned(), json!("cycle-b"));
    let mut cycle_b = node_record("cycle-b");
    cycle_b
        .metadata
        .insert("redirect_to_id".to_owned(), json!("cycle-a"));
    assert_eq!(
        resolve_graph_node_redirect_id("cycle-a", &[cycle_a, cycle_b]),
        None
    );
}

#[test]
fn edge_redirect_resolution_matches_python_lifecycle_modes() {
    let mut redirected = edge_record("old-edge", json!({"source_ids": [], "target_ids": []}));
    redirected
        .metadata
        .insert("lifecycle_status".to_owned(), json!("tombstoned"));
    redirected
        .metadata
        .insert("redirect_to_id".to_owned(), json!("middle-edge"));
    let mut middle = edge_record("middle-edge", json!({"source_ids": [], "target_ids": []}));
    middle
        .metadata
        .insert("lifecycle_status".to_owned(), json!("tombstoned"));
    middle
        .metadata
        .insert("redirect_to_id".to_owned(), json!("current-edge"));
    let edges = vec![
        redirected.clone(),
        middle,
        edge_record("current-edge", json!({})),
    ];

    assert_eq!(
        resolve_graph_edge_redirect_id("old-edge", &edges),
        Some("current-edge".into())
    );
    assert_eq!(
        resolve_graph_edge_redirect_id("current-edge", &edges),
        Some("current-edge".into())
    );
    assert_eq!(resolve_graph_edge_redirect_id("missing", &edges), None);

    let mut tombstone = edge_record("gone-edge", json!({}));
    tombstone
        .metadata
        .insert("lifecycle_status".to_owned(), json!("tombstoned"));
    let mut broken = redirected;
    broken
        .metadata
        .insert("redirect_to_id".to_owned(), json!("missing-edge"));
    assert_eq!(
        resolve_graph_edge_redirect_id("gone-edge", &[tombstone]),
        None
    );
    assert_eq!(resolve_graph_edge_redirect_id("old-edge", &[broken]), None);

    let mut cycle_a = edge_record("cycle-edge-a", json!({}));
    cycle_a
        .metadata
        .insert("redirect_to_id".to_owned(), json!("cycle-edge-b"));
    let mut cycle_b = edge_record("cycle-edge-b", json!({}));
    cycle_b
        .metadata
        .insert("redirect_to_id".to_owned(), json!("cycle-edge-a"));
    assert_eq!(
        resolve_graph_edge_redirect_id("cycle-edge-a", &[cycle_a, cycle_b]),
        None
    );
}

#[test]
fn graph_neighbors_preserve_cross_kind_same_id_from_shared_fixture() {
    let fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/graph-neighbors-cross-kind-id-v1.json"
    )))
    .expect("cross-kind graph-neighbor fixture is valid JSON");
    let shared_id = fixture["expected"]["node_ids"][0]
        .as_str()
        .expect("expected shared node ID");
    let parent_edge = &fixture["parent_edge"];
    let nodes = vec![node_record(shared_id)];
    let edges = vec![
        edge_record(parent_edge["id"].as_str().unwrap(), parent_edge.clone()),
        edge_record(shared_id, json!({"source_ids": [], "target_ids": []})),
    ];

    let actual =
        graph_neighbors_from_records(fixture["rid"].as_str().unwrap(), &nodes, &edges, None);
    let expected = GraphNeighbors {
        nodes: fixture["expected"]["node_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_str().unwrap().to_owned())
            .collect(),
        edges: fixture["expected"]["edge_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_str().unwrap().to_owned())
            .collect(),
    };

    assert_eq!(actual, expected);
}

#[test]
fn graph_visualization_projects_acl_visible_records_in_both_formats() {
    let mut source = node_record("n1");
    source.document =
        Some(json!({"label":"Source","summary":"source summary","doc_id":"d1"}).to_string());
    let mut target = node_record("n2");
    target.document = Some(json!({"label":"Target","doc_id":"d1"}).to_string());
    let mut hidden = node_record("private-node");
    hidden.document = Some(json!({"label":"Hidden","doc_id":"d1"}).to_string());
    let visible_edge = edge_record(
        "e1",
        json!({
            "relation":"connects",
            "doc_id":"d1",
            "source_ids":["n1", "private-node"],
            "target_ids":["n2"],
            "properties":{"weight":2}
        }),
    );
    let unrelated_edge = edge_record(
        "e2",
        json!({"relation":"other","doc_id":"d2","source_ids":["n1"],"target_ids":["n2"]}),
    );
    let mut tombstoned = edge_record(
        "e3",
        json!({"relation":"deleted","doc_id":"d1","source_ids":["n1"],"target_ids":["n2"]}),
    );
    tombstoned
        .metadata
        .insert("lifecycle_status".to_owned(), json!("tombstoned"));
    let nodes = vec![source, target, hidden];
    let edges = vec![visible_edge.clone(), unrelated_edge, tombstoned];

    let d3 = graph_viz_d3_from_records(&nodes[..2], &edges, Some("d1"), "reify", None);
    assert_eq!(d3["nodes"].as_array().unwrap().len(), 2);
    assert!(d3["links"].as_array().unwrap().is_empty());
    assert!(!d3.to_string().contains("private-node"));
    assert!(!d3.to_string().contains("e2"));
    assert!(!d3.to_string().contains("e3"));

    let cy = graph_viz_cytoscape_from_records(&nodes[..2], &edges, Some("d1"), "classic", None);
    let elements = cy["elements"].as_array().unwrap();
    assert_eq!(elements.len(), 2);
    assert_eq!(cy["mode"], "classic");
    assert_eq!(cy["doc_id"], "d1");

    let complete_edge = edge_record(
        "e1",
        json!({
            "relation":"connects",
            "doc_id":"d1",
            "source_ids":["n1"],
            "target_ids":["n2"],
            "properties":{"weight":2}
        }),
    );
    let source_edge = edge_record(
        "e4",
        json!({
            "relation":"composes",
            "doc_id":"d1",
            "source_edge_ids":["e1"],
            "target_edge_ids":["e5"]
        }),
    );
    let target_edge = edge_record("e5", json!({"relation":"terminal","doc_id":"d1"}));
    let reified = graph_viz_cytoscape_from_records(
        &nodes[..2],
        &[complete_edge, source_edge, target_edge],
        Some("d1"),
        "reify",
        None,
    );
    let reified_elements = reified["elements"].as_array().unwrap();
    assert!(reified_elements.iter().any(|element| {
        element["data"]["id"] == "e4::srcE::e1"
            && element["data"]["source"] == "e1"
            && element["data"]["target"] == "e4"
    }));
    assert!(reified_elements.iter().any(|element| {
        element["data"]["id"] == "e4::tgtE::e5"
            && element["data"]["source"] == "e4"
            && element["data"]["target"] == "e5"
    }));
}

// PYTHON_PARITY_V1: rust=rust/crates/kogwistar-store/src/lib.rs#graph_record_has_reference; python=kogwistar/visualization/graph_viz.py#_filter_by_insertion_method; fixture=contracts/golden/visualization-insertion-filter-v1.json; rust_test=rust/crates/kogwistar-store/src/lib.rs#graph_visualization_filters_by_reference_method_and_preserves_python_fallback; python_test=tests/test_visualization.py#test_visualization_insertion_method_filter_matches_shared_rust_fixture
#[test]
fn graph_visualization_filters_by_reference_method_and_preserves_python_fallback() {
    let fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/visualization-insertion-filter-v1.json"
    )))
    .unwrap();
    let make_record = |value: &Value, edge: bool| {
        let id = value["id"].as_str().unwrap();
        let document = if edge {
            json!({
                "label": id,
                "source_ids": ["n1"],
                "target_ids": ["n2"],
                "mentions": value["references"].clone()
            })
        } else {
            json!({
                "label": id,
                "references": value["references"].clone()
            })
        };
        let mut record = GraphRecord {
            id: id.to_owned(),
            document: Some(document.to_string()),
            metadata: Map::from_iter([("doc_id".to_owned(), fixture["doc_id"].clone())]),
            embedding: None,
        };
        if edge {
            record
                .metadata
                .insert("entity_type".to_owned(), json!("edge"));
        }
        record
    };
    let nodes = fixture["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|record| make_record(record, false))
        .collect::<Vec<_>>();
    let edges = fixture["edges"]
        .as_array()
        .unwrap()
        .iter()
        .map(|record| make_record(record, true))
        .collect::<Vec<_>>();
    let doc_id = fixture["doc_id"].as_str();
    let insertion_method = fixture["insertion_method"].as_str();

    let filtered_node_ids = graph_viz_selected_records(&nodes, doc_id, insertion_method)
        .into_iter()
        .map(|node| node.id.as_str())
        .collect::<BTreeSet<_>>();
    let filtered_edge_ids = graph_viz_selected_records(&edges, doc_id, insertion_method)
        .into_iter()
        .map(|edge| edge.id.as_str())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        filtered_node_ids,
        fixture["expected_matches_by_kind"]["node"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_str().unwrap())
            .collect::<BTreeSet<_>>()
    );
    assert_eq!(
        filtered_edge_ids,
        fixture["expected_matches_by_kind"]["edge"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_str().unwrap())
            .collect::<BTreeSet<_>>()
    );

    let mut empty_method_ids = graph_viz_selected_records(&nodes, doc_id, Some(""))
        .into_iter()
        .chain(graph_viz_selected_records(&edges, doc_id, Some("")))
        .map(|record| record.id.as_str())
        .collect::<Vec<_>>();
    empty_method_ids.sort_unstable();
    assert_eq!(
        empty_method_ids,
        fixture["expected_empty_method_filter"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_str().unwrap())
            .collect::<Vec<_>>()
    );

    let mut fallback_ids = graph_viz_selected_records(&nodes, doc_id, Some("missing-method"))
        .into_iter()
        .chain(graph_viz_selected_records(
            &edges,
            doc_id,
            Some("missing-method"),
        ))
        .map(|record| record.id.as_str())
        .collect::<Vec<_>>();
    fallback_ids.sort_unstable();
    assert_eq!(
        fallback_ids,
        fixture["expected_empty_match_fallback"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_str().unwrap())
            .collect::<Vec<_>>()
    );
}

#[test]
fn graph_neighbors_excludes_stale_and_other_document_edges() {
    let mut stale = edge_record(
        "stale",
        serde_json::json!({
            "doc_id": "doc-requested",
            "source_ids": ["n1"],
            "target_ids": ["n2"]
        }),
    );
    stale.metadata.insert(
        "lifecycle_status".to_owned(),
        serde_json::json!("tombstoned"),
    );
    let other = edge_record(
        "other",
        serde_json::json!({
            "doc_id": "doc-other",
            "source_ids": ["n1"],
            "target_ids": ["n3"]
        }),
    );
    let nodes = vec![node_record("n1"), node_record("n2"), node_record("n3")];

    assert_eq!(
        graph_neighbors_from_records("n1", &nodes, &[stale, other], Some("doc-requested")),
        GraphNeighbors::default()
    );
}

#[test]
fn graph_neighbors_excludes_tombstoned_and_dangling_endpoints() {
    let mut tombstoned_node = node_record("n2");
    tombstoned_node.metadata.insert(
        "lifecycle_status".to_owned(),
        serde_json::json!("tombstoned"),
    );
    let edges = vec![edge_record(
        "e1",
        serde_json::json!({
            "source_ids": ["n1", "n2", "missing"],
            "target_ids": [],
            "source_edge_ids": ["missing-edge"]
        }),
    )];
    let nodes = vec![node_record("n1"), tombstoned_node];

    assert_eq!(
        graph_neighbors_from_records("n1", &nodes, &edges, None),
        GraphNeighbors {
            nodes: BTreeSet::new(),
            edges: BTreeSet::from(["e1".to_owned()]),
        }
    );
    assert_eq!(
        graph_neighbors_from_records("e1", &nodes, &edges, None),
        GraphNeighbors {
            nodes: BTreeSet::from(["n1".to_owned()]),
            edges: BTreeSet::new(),
        }
    );
}

#[test]
fn graph_find_edges_matches_relation_document_and_label_filters() {
    let mut source = node_record("n1");
    source.document = Some(serde_json::json!({"label": "Alpha source"}).to_string());
    let mut target = node_record("n2");
    target.document = Some(serde_json::json!({"label": "Beta target"}).to_string());
    let edges = vec![
        edge_record(
            "e1",
            serde_json::json!({
                "relation": "links",
                "doc_id": "doc-1",
                "source_ids": ["n1"],
                "target_ids": ["n2"]
            }),
        ),
        edge_record(
            "e2",
            serde_json::json!({
                "relation": "mentions",
                "doc_id": "doc-1",
                "source_ids": ["n1"],
                "target_ids": ["n2"]
            }),
        ),
    ];
    assert_eq!(
        graph_find_edges_from_records(
            &GraphFindEdgesQuery {
                relation: Some("links".to_owned()),
                src_label_contains: Some("alpha".to_owned()),
                tgt_label_contains: Some("TARGET".to_owned()),
                doc_id: Some("doc-1".to_owned()),
            },
            &[source, target],
            &edges,
        ),
        vec!["e1".to_owned()]
    );
}

#[test]
fn graph_shortest_path_is_bounded_and_respects_document_and_tombstone_filters() {
    let nodes = vec![node_record("n1"), node_record("n2"), node_record("n3")];
    let edges = vec![
        edge_record(
            "e1",
            serde_json::json!({"doc_id": "doc-1", "source_ids": ["n1"], "target_ids": ["n2"]}),
        ),
        edge_record(
            "e2",
            serde_json::json!({"doc_id": "doc-1", "source_ids": ["n2"], "target_ids": ["n3"]}),
        ),
        edge_record(
            "other-doc",
            serde_json::json!({"doc_id": "doc-2", "source_ids": ["n1"], "target_ids": ["n3"]}),
        ),
    ];
    let query = GraphShortestPathQuery {
        src_id: "n1".to_owned(),
        dst_id: "n3".to_owned(),
        doc_id: Some("doc-1".to_owned()),
        max_depth: 4,
    };
    assert_eq!(
        graph_shortest_path_from_records(&query, &nodes, &edges),
        vec!["n1", "e1", "n2", "e2", "n3"]
    );
    let too_shallow = GraphShortestPathQuery {
        max_depth: 1,
        ..query.clone()
    };
    assert!(graph_shortest_path_from_records(&too_shallow, &nodes, &edges).is_empty());
    let mut tombstoned = edges[1].clone();
    tombstoned.metadata.insert(
        "lifecycle_status".to_owned(),
        serde_json::json!("tombstoned"),
    );
    assert!(
        graph_shortest_path_from_records(&query, &nodes, &[edges[0].clone(), tombstoned])
            .is_empty()
    );
}

#[test]
fn graph_k_hop_returns_stable_layers_without_reexpanding_visited_ids() {
    let nodes = vec![node_record("n1"), node_record("n2"), node_record("n3")];
    let edges = vec![
        edge_record(
            "e1",
            serde_json::json!({"doc_id": "doc-1", "source_ids": ["n1"], "target_ids": ["n2"]}),
        ),
        edge_record(
            "e2",
            serde_json::json!({"doc_id": "doc-1", "source_ids": ["n2"], "target_ids": ["n3"]}),
        ),
    ];
    let layers = graph_k_hop_from_records(
        &GraphKHopQuery {
            start_ids: vec!["n1".to_owned()],
            k: 2,
            doc_id: Some("doc-1".to_owned()),
        },
        &nodes,
        &edges,
    );
    assert_eq!(layers.len(), 2);
    assert!(layers[0].nodes.is_empty());
    assert_eq!(layers[0].edges, BTreeSet::from(["e1".to_owned()]));
    assert_eq!(
        layers[1].nodes,
        BTreeSet::from(["n1".to_owned(), "n2".to_owned()])
    );
    assert!(layers[1].edges.is_empty());
}

#[test]
fn graph_traversal_matches_shared_golden_contract() {
    let fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/graph-traversal-v1.json"
    )))
    .expect("valid graph traversal fixture");
    assert_eq!(fixture["schema_version"], 1);
    let string_set = |value: &Value| {
        value
            .as_array()
            .expect("fixture string array")
            .iter()
            .map(|item| item.as_str().expect("fixture string").to_owned())
            .collect::<BTreeSet<_>>()
    };
    let nodes = fixture["graph"]["nodes"]
        .as_array()
        .expect("fixture nodes")
        .iter()
        .map(|value| {
            let id = value.as_str().expect("node ID");
            let mut record = node_record(id);
            let label = fixture["graph"]["node_labels"][id]
                .as_str()
                .expect("node label");
            record.document = Some(serde_json::json!({"label": label}).to_string());
            record
        })
        .collect::<Vec<_>>();
    let edges = fixture["graph"]["edges"]
        .as_array()
        .expect("fixture edges")
        .iter()
        .map(|value| {
            edge_record(
                value["id"].as_str().expect("edge ID"),
                serde_json::json!({
                    "doc_id": value["doc_id"],
                    "relation": value["relation"],
                    "source_ids": value["source_ids"],
                    "target_ids": value["target_ids"]
                }),
            )
        })
        .collect::<Vec<_>>();
    let cases = fixture["cases"].as_array().expect("fixture cases");

    let node_case = &cases[0];
    let neighbors = graph_neighbors_from_records_with_jump(
        node_case["rid"].as_str().expect("neighbor ID"),
        &nodes,
        &edges,
        Some("D1"),
        false,
    );
    assert_eq!(
        neighbors.nodes,
        string_set(&node_case["expected_neighbors"]["nodes"])
    );
    assert_eq!(
        neighbors.edges,
        string_set(&node_case["expected_neighbors"]["edges"])
    );

    let edge_case = &cases[1];
    let neighbors = graph_neighbors_from_records(
        edge_case["rid"].as_str().expect("edge ID"),
        &nodes,
        &edges,
        Some("D1"),
    );
    assert_eq!(
        neighbors.nodes,
        string_set(&edge_case["expected_neighbors"]["nodes"])
    );
    assert_eq!(
        neighbors.edges,
        string_set(&edge_case["expected_neighbors"]["edges"])
    );

    let path_case = &cases[2];
    let path = graph_shortest_path_from_records(
        &GraphShortestPathQuery {
            src_id: path_case["src_id"].as_str().unwrap().to_owned(),
            dst_id: path_case["dst_id"].as_str().unwrap().to_owned(),
            doc_id: Some("D1".to_owned()),
            max_depth: 8,
        },
        &nodes,
        &edges,
    );
    let expected_path = path_case["expected_path"]
        .as_array()
        .expect("fixture path")
        .iter()
        .map(|value| value.as_str().expect("fixture path ID").to_owned())
        .collect::<Vec<_>>();
    assert_eq!(path, expected_path);

    let hop_case = &cases[3];
    let layers = graph_k_hop_from_records(
        &GraphKHopQuery {
            start_ids: vec!["A".to_owned()],
            k: hop_case["k"].as_u64().unwrap() as usize,
            doc_id: Some("D1".to_owned()),
        },
        &nodes,
        &edges,
    );
    let expected_layers = hop_case["expected_layers"]
        .as_array()
        .expect("fixture layers");
    assert_eq!(layers.len(), expected_layers.len());
    for (actual, expected) in layers.iter().zip(expected_layers) {
        assert_eq!(actual.nodes, string_set(&expected["nodes"]));
        assert_eq!(actual.edges, string_set(&expected["edges"]));
    }

    let find_case = &cases[4];
    let matching_edges = graph_find_edges_from_records(
        &GraphFindEdgesQuery {
            relation: find_case["relation"].as_str().map(str::to_owned),
            src_label_contains: find_case["source_label_contains"]
                .as_str()
                .map(str::to_owned),
            tgt_label_contains: find_case["target_label_contains"]
                .as_str()
                .map(str::to_owned),
            doc_id: find_case["doc_id"].as_str().map(str::to_owned),
        },
        &nodes,
        &edges,
    );
    let expected_edges = find_case["expected_edge_ids"]
        .as_array()
        .expect("fixture edge IDs")
        .iter()
        .map(|value| value.as_str().expect("fixture edge ID").to_owned())
        .collect::<Vec<_>>();
    assert_eq!(matching_edges, expected_edges);
}

#[test]
fn graph_acl_traversal_matches_shared_python_contract() {
    let fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/graph-acl-traversal-v1.json"
    )))
    .expect("valid graph ACL traversal fixture");
    assert_eq!(fixture["schema_version"], 1);
    let truth_graph = fixture["truth_graph"].as_str().expect("truth graph");
    let edge_spec = &fixture["edge"];
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    for node_spec in fixture["nodes"].as_array().expect("nodes") {
        let id = node_spec["id"].as_str().expect("node ID");
        let record = node_record(id);
        nodes.push(record.clone());

        let mut acl = GraphRecord::new(format!("acl-{id}"));
        acl.metadata
            .insert("entity_type".into(), "acl_record".into());
        acl.metadata
            .insert("acl_truth_graph".into(), truth_graph.into());
        acl.metadata
            .insert("acl_target_entity_id".into(), id.into());
        acl.metadata
            .insert("acl_target_grain".into(), "node".into());
        acl.metadata.insert("acl_version".into(), 1.into());
        acl.metadata
            .insert("acl_mode".into(), node_spec["mode"].clone());
        if let Some(owner) = node_spec["owner_id"].as_str() {
            acl.metadata.insert("owner_id".into(), owner.into());
        }
        nodes.push(acl);
    }

    let edge_id = edge_spec["id"].as_str().expect("edge ID");
    let edge_document = serde_json::json!({
        "doc_id": fixture["doc_id"],
        "relation": edge_spec["relation"],
        "source_ids": edge_spec["source_ids"],
        "target_ids": edge_spec["target_ids"],
        "source_edge_ids": [],
        "target_edge_ids": []
    });
    let edge = edge_record(edge_id, edge_document);
    edges.push(edge.clone());
    let mut edge_acl = GraphRecord::new(format!("acl-{edge_id}"));
    edge_acl
        .metadata
        .insert("entity_type".into(), "acl_record".into());
    edge_acl
        .metadata
        .insert("acl_truth_graph".into(), truth_graph.into());
    edge_acl
        .metadata
        .insert("acl_target_entity_id".into(), edge_id.into());
    edge_acl
        .metadata
        .insert("acl_target_grain".into(), "edge".into());
    edge_acl.metadata.insert("acl_version".into(), 1.into());
    edge_acl
        .metadata
        .insert("acl_mode".into(), edge_spec["mode"].clone());
    if let Some(owner) = edge_spec["owner_id"].as_str() {
        edge_acl.metadata.insert("owner_id".into(), owner.into());
    }
    nodes.push(edge_acl);

    for principal_spec in fixture["principals"].as_array().expect("principals") {
        let principal = GraphAclPrincipal {
            principal_id: principal_spec["id"]
                .as_str()
                .expect("principal ID")
                .to_owned(),
            ..Default::default()
        };
        let (visible_nodes, visible_edges) =
            graph_records_visible_by_acl(&nodes, &edges, truth_graph, &principal);
        let neighbors = graph_neighbors_from_records(
            fixture["nodes"][0]["id"].as_str().expect("source ID"),
            &visible_nodes,
            &visible_edges,
            fixture["doc_id"].as_str(),
        );
        let expected_nodes = principal_spec["expected_neighbors"]["nodes"]
            .as_array()
            .expect("expected nodes")
            .iter()
            .map(|value| value.as_str().expect("node ID").to_owned())
            .collect::<BTreeSet<_>>();
        let expected_edges = principal_spec["expected_neighbors"]["edges"]
            .as_array()
            .expect("expected edges")
            .iter()
            .map(|value| value.as_str().expect("edge ID").to_owned())
            .collect::<BTreeSet<_>>();
        assert_eq!(neighbors.nodes, expected_nodes, "{}", principal_spec["id"]);
        assert_eq!(neighbors.edges, expected_edges, "{}", principal_spec["id"]);

        let k_hop = graph_k_hop_from_records(
            &GraphKHopQuery {
                start_ids: vec![
                    fixture["nodes"][0]["id"]
                        .as_str()
                        .expect("source ID")
                        .to_owned(),
                ],
                k: 2,
                doc_id: fixture["doc_id"].as_str().map(str::to_owned),
            },
            &visible_nodes,
            &visible_edges,
        );
        let expected_k_hop = principal_spec["expected_k_hop"]
            .as_array()
            .expect("expected k-hop layers")
            .iter()
            .map(|layer| {
                serde_json::json!({
                    "nodes": layer["nodes"],
                    "edges": layer["edges"]
                })
            })
            .collect::<Vec<_>>();
        let actual_k_hop = k_hop
            .iter()
            .map(|layer| {
                serde_json::json!({
                    "nodes": layer.nodes,
                    "edges": layer.edges
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(actual_k_hop, expected_k_hop, "{}", principal_spec["id"]);

        let shortest_path = graph_shortest_path_from_records(
            &GraphShortestPathQuery {
                src_id: fixture["nodes"][0]["id"]
                    .as_str()
                    .expect("source ID")
                    .to_owned(),
                dst_id: fixture["nodes"][1]["id"]
                    .as_str()
                    .expect("target ID")
                    .to_owned(),
                doc_id: fixture["doc_id"].as_str().map(str::to_owned),
                max_depth: 8,
            },
            &visible_nodes,
            &visible_edges,
        );
        assert_eq!(
            shortest_path,
            principal_spec["expected_shortest_path"]
                .as_array()
                .expect("expected path")
                .iter()
                .map(|value| value.as_str().expect("path ID").to_owned())
                .collect::<Vec<_>>(),
            "{}",
            principal_spec["id"]
        );

        let find_edges = graph_find_edges_from_records(
            &GraphFindEdgesQuery {
                relation: edge_spec["relation"].as_str().map(str::to_owned),
                doc_id: fixture["doc_id"].as_str().map(str::to_owned),
                ..Default::default()
            },
            &visible_nodes,
            &visible_edges,
        );
        assert_eq!(
            find_edges,
            principal_spec["expected_find_edges"]
                .as_array()
                .expect("expected matching edges")
                .iter()
                .map(|value| value.as_str().expect("edge ID").to_owned())
                .collect::<Vec<_>>(),
            "{}",
            principal_spec["id"]
        );
    }
}

#[test]
fn graph_record_acl_matches_python_visibility_and_fails_closed() {
    let mut acl = GraphRecord::new("acl-n1");
    acl.metadata
        .insert("entity_type".into(), "acl_record".into());
    acl.metadata
        .insert("acl_truth_graph".into(), "knowledge".into());
    acl.metadata
        .insert("acl_target_entity_id".into(), "n1".into());
    acl.metadata
        .insert("acl_target_grain".into(), "node".into());
    acl.metadata.insert("acl_version".into(), 1.into());
    acl.metadata.insert("acl_mode".into(), "shared".into());
    acl.document = Some(
        serde_json::json!({
            "properties": {"shared_with_principals": ["alice"]}
        })
        .to_string(),
    );
    let nodes = vec![acl, GraphRecord::new("n1"), GraphRecord::new("n2")];
    let edges = vec![GraphRecord::new("e1")];
    let alice = GraphAclPrincipal {
        principal_id: "alice".to_owned(),
        ..Default::default()
    };
    let bob = GraphAclPrincipal {
        principal_id: "bob".to_owned(),
        ..Default::default()
    };
    let (alice_nodes, alice_edges) =
        graph_records_visible_by_acl(&nodes, &edges, "knowledge", &alice);
    assert_eq!(
        alice_nodes
            .iter()
            .map(|r| r.id.as_str())
            .collect::<Vec<_>>(),
        ["n1"]
    );
    assert!(alice_edges.is_empty());
    let (bob_nodes, bob_edges) = graph_records_visible_by_acl(&nodes, &edges, "knowledge", &bob);
    assert!(bob_nodes.is_empty());
    assert!(bob_edges.is_empty());
}

#[test]
fn acl_projection_reuses_index_after_helper_records_are_removed() {
    let mut acl = GraphRecord::new("acl-n1");
    acl.metadata.extend(
        serde_json::json!({
            "entity_type": "acl_record",
            "acl_truth_graph": "knowledge",
            "acl_target_entity_id": "n1",
            "acl_target_grain": "node",
            "acl_version": 1,
            "acl_mode": "shared"
        })
        .as_object()
        .expect("ACL metadata is an object")
        .clone(),
    );
    acl.document = Some(
        serde_json::json!({
            "properties": {"shared_with_principals": ["alice"]}
        })
        .to_string(),
    );
    let complete_snapshot = vec![acl, GraphRecord::new("n1"), GraphRecord::new("n2")];
    let alice = GraphAclPrincipal {
        principal_id: "alice".to_owned(),
        ..Default::default()
    };

    let projection =
        GraphAclProjection::from_complete_snapshot(&complete_snapshot, &[], "knowledge", &alice);
    assert_eq!(
        projection
            .nodes()
            .iter()
            .map(|record| record.id.as_str())
            .collect::<Vec<_>>(),
        ["n1"]
    );
    assert!(projection.edges().is_empty());

    // The filtered view deliberately lacks the ACL helper node. A later
    // projection reuses the full snapshot's index through the wrapper.
    let reprojected = projection.filter_records(projection.nodes(), &[]);
    assert_eq!(reprojected.nodes(), projection.nodes());

    // Rebuilding from the filtered view has no ACL rule and correctly
    // fails closed; services must not mistake that for the canonical ACL.
    let (incorrectly_rebuilt_nodes, _) =
        graph_records_visible_by_acl(projection.nodes(), &[], "knowledge", &alice);
    assert!(incorrectly_rebuilt_nodes.is_empty());
}

#[test]
fn paged_conversation_acl_uses_rules_from_other_pages() {
    let acl_record = |id: &str, target_id: &str| {
        let mut record = GraphRecord::new(id);
        record.metadata.extend(
            serde_json::json!({
                "entity_type": "acl_record",
                "acl_truth_graph": "conversation",
                "acl_target_entity_id": target_id,
                "acl_target_grain": "node",
                "acl_version": 1,
                "acl_mode": "private",
                "owner_id": "alice"
            })
            .as_object()
            .expect("ACL metadata is an object")
            .clone(),
        );
        record
    };
    let mut start = GraphRecord::new("start-alice");
    start.document = Some(
        serde_json::json!({
            "metadata": {
                "entity_type": "conversation_start",
                "conversation_id": "conversation-1",
                "user_id": "alice",
                "status": "active"
            }
        })
        .to_string(),
    );
    let mut turn = GraphRecord::new("turn-1");
    turn.metadata.extend(
        serde_json::json!({
            "entity_type": "conversation_turn",
            "conversation_id": "conversation-1",
            "turn_index": 0
        })
        .as_object()
        .expect("turn metadata is an object")
        .clone(),
    );
    let acl_page = [
        acl_record("acl-start", "start-alice"),
        acl_record("acl-turn", "turn-1"),
    ];
    let data_page = [start, turn];

    let mut acl_index = GraphAclIndex::default();
    acl_index.extend_from_records(&acl_page, "conversation");
    let compact = compact_conversation_list_candidates(&data_page);
    assert!(compact.iter().all(|record| record.document.is_none()));

    let alice = GraphAclPrincipal {
        principal_id: "alice".to_owned(),
        ..Default::default()
    };
    let bob = GraphAclPrincipal {
        principal_id: "bob".to_owned(),
        ..Default::default()
    };
    let (visible_alice, _) =
        graph_records_visible_by_acl_with_index(&compact, &[], &acl_index, &alice);
    let (visible_bob, _) = graph_records_visible_by_acl_with_index(&compact, &[], &acl_index, &bob);
    let mut alice_list = ConversationListAccumulator::new("alice");
    alice_list.extend_visible_records(&visible_alice);

    assert_eq!(
        alice_list.finish(),
        vec![ConversationListEntry {
            id: "conversation-1".to_owned(),
            start_node_id: "start-alice".to_owned(),
            status: "active".to_owned(),
            turn_count: 1,
        }]
    );
    let mut bob_list = ConversationListAccumulator::new("bob");
    bob_list.extend_visible_records(&visible_bob);
    assert!(bob_list.finish().is_empty());
}

#[test]
fn graph_record_acl_covers_owner_scope_group_private_and_stale_item_rules() {
    let acl_record = |id: &str, target: &str, mode: &str, version: i64| {
        let mut record = GraphRecord::new(id);
        record
            .metadata
            .insert("entity_type".into(), "acl_record".into());
        record
            .metadata
            .insert("acl_truth_graph".into(), "knowledge".into());
        record
            .metadata
            .insert("acl_target_entity_id".into(), target.into());
        record
            .metadata
            .insert("acl_target_grain".into(), "node".into());
        record.metadata.insert("acl_version".into(), version.into());
        record.metadata.insert("acl_mode".into(), mode.into());
        record
    };
    let mut owner = acl_record("acl-owner", "owner", "private", 1);
    owner.metadata.insert("owner_id".into(), "alice".into());
    let mut scoped = acl_record("acl-scope", "scope", "scope", 1);
    scoped
        .metadata
        .insert("security_scope".into(), "team-a".into());
    let mut grouped = acl_record("acl-group", "group", "group", 1);
    grouped.document =
        Some(serde_json::json!({"properties": {"shared_with_groups": ["reviewers"]}}).to_string());
    let mut stale_item = acl_record("acl-span", "span", "public", 1);
    stale_item
        .metadata
        .insert("acl_target_item_id".into(), "span-1".into());
    let nodes = vec![
        owner,
        scoped,
        grouped,
        stale_item,
        GraphRecord::new("owner"),
        GraphRecord::new("scope"),
        GraphRecord::new("group"),
        GraphRecord::new("span"),
    ];
    let principal = GraphAclPrincipal {
        principal_id: "alice".into(),
        groups: BTreeSet::from(["reviewers".into()]),
        security_scope: Some("team-a".into()),
    };
    let (visible, _) = graph_records_visible_by_acl(&nodes, &[], "knowledge", &principal);
    assert_eq!(
        visible
            .iter()
            .map(|record| record.id.as_str())
            .collect::<Vec<_>>(),
        ["owner", "scope", "group"]
    );
}
