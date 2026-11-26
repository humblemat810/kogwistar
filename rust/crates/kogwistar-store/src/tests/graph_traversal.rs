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
fn node_as_of_resolution_matches_python_lifecycle_boundaries() {
    let as_of = "2026-01-02T00:00:00Z";
    let mut old = node_record("old");
    old.metadata
        .insert("lifecycle_status".to_owned(), json!("tombstoned"));
    old.metadata
        .insert("deleted_at".to_owned(), json!("2026-01-02T00:00:00Z"));
    old.metadata
        .insert("redirect_to_id".to_owned(), json!("current"));
    let current = node_record("current");

    assert_eq!(
        resolve_graph_node_as_of_id("old", &[old.clone(), current.clone()], as_of, true, 16),
        Ok(Some("current".to_owned())),
        "redirect applies at the deletion instant"
    );
    assert_eq!(
        resolve_graph_node_as_of_id(
            "old",
            &[old.clone(), current.clone()],
            "2026-01-02T01:00:00+01:00",
            true,
            16
        ),
        Ok(Some("current".to_owned())),
        "offset timestamps compare as the same UTC instant"
    );

    let before = "2026-01-01T23:59:59Z";
    assert_eq!(
        resolve_graph_node_as_of_id("old", &[old.clone(), current.clone()], before, true, 16),
        Ok(Some("old".to_owned())),
        "tombstone remains historically visible before deletion"
    );

    let mut future = node_record("future");
    future
        .metadata
        .insert("effective_from".to_owned(), json!("2026-01-03T00:00:00Z"));
    assert_eq!(
        resolve_graph_node_as_of_id("future", &[future], as_of, true, 16),
        Ok(None),
        "future-effective node is absent"
    );

    assert_eq!(
        resolve_graph_node_as_of_id("old", &[old.clone()], as_of, true, 16),
        Ok(None),
        "missing redirect target fails closed"
    );
    assert_eq!(
        resolve_graph_node_as_of_id("old", &[old.clone(), current.clone()], as_of, false, 16),
        Ok(None),
        "disabled redirect following excludes tombstoned node"
    );
    assert_eq!(
        resolve_graph_node_as_of_id("old", &[old.clone(), current.clone()], as_of, true, 0),
        Ok(None),
        "redirect hop limit is enforced"
    );
    let mut cycle_a = node_record("cycle-a");
    cycle_a
        .metadata
        .insert("lifecycle_status".to_owned(), json!("tombstoned"));
    cycle_a
        .metadata
        .insert("deleted_at".to_owned(), json!("2026-01-01T00:00:00Z"));
    cycle_a
        .metadata
        .insert("redirect_to_id".to_owned(), json!("cycle-b"));
    let mut cycle_b = cycle_a.clone();
    cycle_b.id = "cycle-b".to_owned();
    cycle_b
        .metadata
        .insert("redirect_to_id".to_owned(), json!("cycle-a"));
    assert_eq!(
        resolve_graph_node_as_of_id("cycle-a", &[cycle_a, cycle_b], as_of, true, 16),
        Ok(None),
        "redirect cycle fails closed"
    );
    assert_eq!(
        resolve_graph_node_as_of_id("old", &[old, current], "not-a-timestamp", true, 16),
        Err("invalid as-of timestamp"),
        "invalid as-of time is reported, not treated as empty results"
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
fn graph_find_edges_preserves_supplied_backend_order() {
    let edges = vec![
        edge_record("z-edge", serde_json::json!({"relation": "links"})),
        edge_record("a-edge", serde_json::json!({"relation": "links"})),
    ];

    assert_eq!(
        graph_find_edges_from_records(
            &GraphFindEdgesQuery {
                relation: Some("links".to_owned()),
                ..Default::default()
            },
            &[],
            &edges,
        ),
        vec!["z-edge".to_owned(), "a-edge".to_owned()]
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
