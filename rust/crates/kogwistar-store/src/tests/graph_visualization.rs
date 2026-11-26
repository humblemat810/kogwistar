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

// PYTHON_PARITY_NOTE_V1: Rust document selection accepts metadata.doc_ids and payload.metadata.doc_id like Python `_ids_by_doc`; rust=rust/crates/kogwistar-store/src/sections/graph_traversal.rs#graph_record_matches_doc_id; python=kogwistar/visualization/graph_viz.py#_ids_by_doc; rust_test=rust/crates/kogwistar-store/src/tests/graph_visualization.rs#graph_visualization_matches_python_document_membership; python_test=tests/test_visualization.py#test_graph_visualization_doc_filter_matches_python_contract
#[test]
fn graph_visualization_matches_python_document_membership() {
    let record = |id: &str, payload: Value, metadata: Map<String, Value>| GraphRecord {
        id: id.to_owned(),
        document: Some(payload.to_string()),
        metadata,
        embedding: None,
    };
    let nodes = vec![
        record(
            "node-doc-ids",
            json!({"id":"node-doc-ids","label":"metadata doc_ids"}),
            Map::from_iter([("doc_ids".to_owned(), json!(["doc-a"]))]),
        ),
        record(
            "node-payload-metadata",
            json!({"id":"node-payload-metadata","label":"payload metadata","metadata":{"doc_id":"doc-a"}}),
            Map::new(),
        ),
        record(
            "node-other-doc",
            json!({"id":"node-other-doc","label":"other document"}),
            Map::from_iter([("doc_ids".to_owned(), json!(["doc-b"]))]),
        ),
    ];
    let edges = vec![
        record(
            "edge-selected",
            json!({"id":"edge-selected","label":"connects","source_ids":["node-doc-ids"],"target_ids":["node-payload-metadata"],"doc_id":"doc-a"}),
            Map::new(),
        ),
        record(
            "edge-incomplete",
            json!({"id":"edge-incomplete","label":"hidden endpoint","source_ids":["node-doc-ids"],"target_ids":["node-other-doc"],"doc_id":"doc-a"}),
            Map::new(),
        ),
    ];

    let result = graph_viz_d3_from_records(&nodes, &edges, Some("doc-a"), "reify", None);
    let node_ids = result["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|node| node["id"].as_str())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        node_ids,
        ["edge-selected", "node-doc-ids", "node-payload-metadata"]
            .into_iter()
            .collect()
    );
    assert_eq!(result["links"].as_array().unwrap().len(), 2);
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
    let nodes = [source, target, hidden];
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
