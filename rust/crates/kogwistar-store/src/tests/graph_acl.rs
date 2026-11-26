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

#[test]
fn graph_acl_latest_version_is_scoped_to_selected_truth_graph() {
    let entity_id = "shared-entity-id";
    let acl = |record_id: &str, truth_graph: &str, version: i64, mode: &str| {
        let mut record = GraphRecord::new(record_id);
        record.metadata.extend(
            serde_json::json!({
                "entity_type": "acl_record",
                "acl_truth_graph": truth_graph,
                "acl_target_entity_id": entity_id,
                "acl_target_grain": "node",
                "acl_version": version,
                "acl_mode": mode,
                "owner_id": if mode == "private" { Some("wisdom-owner") } else { None },
            })
            .as_object()
            .expect("ACL metadata")
            .clone(),
        );
        record
    };
    let records = vec![
        GraphRecord::new(entity_id),
        acl("knowledge-v1", "knowledge", 1, "public"),
        acl("wisdom-v1", "wisdom", 1, "public"),
        acl("wisdom-v2", "wisdom", 2, "private"),
    ];
    let reader = GraphAclPrincipal {
        principal_id: "reader".into(),
        ..Default::default()
    };
    let owner = GraphAclPrincipal {
        principal_id: "wisdom-owner".into(),
        ..Default::default()
    };

    let (knowledge_reader, _) = graph_records_visible_by_acl(&records, &[], "knowledge", &reader);
    let (wisdom_reader, _) = graph_records_visible_by_acl(&records, &[], "wisdom", &reader);
    let (wisdom_owner, _) = graph_records_visible_by_acl(&records, &[], "wisdom", &owner);

    assert!(knowledge_reader.iter().any(|record| record.id == entity_id));
    assert!(!wisdom_reader.iter().any(|record| record.id == entity_id));
    assert!(wisdom_owner.iter().any(|record| record.id == entity_id));
}

#[test]
fn graph_acl_matches_shared_golden_contract() {
    let fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/graph-acl-v1.json"
    )))
    .expect("valid graph ACL fixture");
    assert_eq!(fixture["schema_version"], 1);
    let principal = GraphAclPrincipal {
        principal_id: fixture["principal"]["id"]
            .as_str()
            .expect("principal ID")
            .to_owned(),
        groups: fixture["principal"]["groups"]
            .as_array()
            .expect("principal groups")
            .iter()
            .map(|value| value.as_str().expect("group").to_owned())
            .collect(),
        security_scope: fixture["principal"]["security_scope"]
            .as_str()
            .map(str::to_owned),
    };
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    for target in fixture["targets"].as_array().expect("targets") {
        let id = target["id"].as_str().expect("target ID");
        let grain = target["grain"].as_str().expect("target grain");
        let target_record = GraphRecord::new(id);
        let mut acl = GraphRecord::new(format!("acl-{id}"));
        acl.metadata
            .insert("entity_type".into(), "acl_record".into());
        acl.metadata
            .insert("acl_truth_graph".into(), fixture["truth_graph"].clone());
        acl.metadata
            .insert("acl_target_entity_id".into(), id.into());
        acl.metadata.insert("acl_target_grain".into(), grain.into());
        acl.metadata.insert("acl_version".into(), 1.into());
        if let Some(mode) = target["mode"].as_str() {
            acl.metadata.insert("acl_mode".into(), mode.into());
        }
        if let Some(owner) = target["owner_id"].as_str() {
            acl.metadata.insert("owner_id".into(), owner.into());
        }
        if let Some(scope) = target["security_scope"].as_str() {
            acl.metadata.insert("security_scope".into(), scope.into());
        }
        let mut properties = serde_json::Map::new();
        if let Some(values) = target["shared_with_principals"].as_array() {
            properties.insert(
                "shared_with_principals".into(),
                Value::Array(values.clone()),
            );
        }
        if let Some(values) = target["shared_with_groups"].as_array() {
            properties.insert("shared_with_groups".into(), Value::Array(values.clone()));
        }
        if target["tombstoned"].as_bool().unwrap_or(false) {
            acl.metadata.insert("tombstoned".into(), true.into());
        }
        if !properties.is_empty() {
            acl.document = Some(serde_json::json!({"properties": properties}).to_string());
        }
        nodes.push(acl);
        if grain == "node" {
            nodes.push(target_record);
        } else {
            edges.push(target_record);
        }
    }
    let (visible_nodes, visible_edges) =
        graph_records_visible_by_acl(&nodes, &edges, "knowledge", &principal);
    let node_ids = visible_nodes
        .iter()
        .map(|record| record.id.as_str())
        .collect::<Vec<_>>();
    let edge_ids = visible_edges
        .iter()
        .map(|record| record.id.as_str())
        .collect::<Vec<_>>();
    let expected_nodes = fixture["expected_visible"]["nodes"]
        .as_array()
        .expect("expected nodes")
        .iter()
        .map(|value| value.as_str().expect("node ID").to_owned())
        .collect::<Vec<_>>();
    let expected_edges = fixture["expected_visible"]["edges"]
        .as_array()
        .expect("expected edges")
        .iter()
        .map(|value| value.as_str().expect("edge ID").to_owned())
        .collect::<Vec<_>>();
    assert_eq!(node_ids, expected_nodes);
    assert_eq!(edge_ids, expected_edges);

    let isolation = &fixture["graph_type_isolation"];
    let entity_id = isolation["entity_id"].as_str().expect("entity ID");
    let mut scoped_nodes = vec![GraphRecord::new(entity_id)];
    for case in isolation["cases"].as_array().expect("graph-type cases") {
        let mut acl = GraphRecord::new(format!(
            "acl-{}",
            case["truth_graph"].as_str().expect("truth graph")
        ));
        acl.metadata.extend(
            serde_json::json!({
                "entity_type": "acl_record",
                "acl_truth_graph": case["truth_graph"],
                "acl_target_entity_id": entity_id,
                "acl_target_grain": case["grain"],
                "acl_version": 1,
                "acl_mode": case["mode"],
                "owner_id": case.get("owner_id")
            })
            .as_object()
            .expect("ACL metadata object")
            .clone(),
        );
        scoped_nodes.push(acl);
    }
    for case in isolation["cases"].as_array().expect("graph-type cases") {
        let truth_graph = case["truth_graph"].as_str().expect("truth graph");
        let (visible, _) =
            graph_records_visible_by_acl(&scoped_nodes, &[], truth_graph, &principal);
        assert_eq!(
            visible.iter().any(|record| record.id == entity_id),
            case["expected_visible"]
                .as_bool()
                .expect("visibility expectation"),
            "ACL decision leaked across graph type {truth_graph}"
        );
    }
}

#[test]
fn wisdom_acl_keeps_same_id_node_and_edge_grains_independent() {
    let entity_id = "same-id-in-wisdom";
    let acl = |id: &str, grain: &str, mode: &str, owner: Option<&str>| {
        let mut record = GraphRecord::new(format!("acl-{grain}"));
        record
            .metadata
            .insert("entity_type".into(), "acl_record".into());
        record
            .metadata
            .insert("acl_truth_graph".into(), "wisdom".into());
        record
            .metadata
            .insert("acl_target_entity_id".into(), id.into());
        record
            .metadata
            .insert("acl_target_grain".into(), grain.into());
        record
            .metadata
            .insert("acl_version".into(), serde_json::json!(1));
        record.metadata.insert("acl_mode".into(), mode.into());
        if let Some(owner) = owner {
            record.metadata.insert("owner_id".into(), owner.into());
        }
        record
    };
    let node = GraphRecord::new(entity_id);
    let mut edge = GraphRecord::new(entity_id);
    edge.metadata
        .insert("source_ids".into(), serde_json::json!([]));
    edge.metadata
        .insert("target_ids".into(), serde_json::json!([]));
    let nodes = vec![
        node,
        acl(entity_id, "node", "public", None),
        acl(entity_id, "edge", "private", Some("alice")),
    ];

    let reader = GraphAclPrincipal {
        principal_id: "bob".to_owned(),
        groups: BTreeSet::new(),
        security_scope: None,
    };
    let (reader_nodes, reader_edges) =
        graph_records_visible_by_acl(&nodes, &[edge.clone()], "wisdom", &reader);
    assert_eq!(
        reader_nodes
            .iter()
            .map(|record| record.id.as_str())
            .collect::<Vec<_>>(),
        [entity_id]
    );
    assert!(reader_edges.is_empty());

    let owner = GraphAclPrincipal {
        principal_id: "alice".to_owned(),
        groups: BTreeSet::new(),
        security_scope: None,
    };
    let (owner_nodes, owner_edges) =
        graph_records_visible_by_acl(&nodes, &[edge], "wisdom", &owner);
    assert_eq!(
        owner_nodes
            .iter()
            .map(|record| record.id.as_str())
            .collect::<Vec<_>>(),
        [entity_id]
    );
    assert_eq!(
        owner_edges
            .iter()
            .map(|record| record.id.as_str())
            .collect::<Vec<_>>(),
        [entity_id]
    );
}

#[test]
fn wisdom_node_acl_does_not_override_same_id_edge_acl() {
    let entity_id = "same-id-in-wisdom";
    let acl = |grain: &str, mode: &str| {
        let mut record = GraphRecord::new(format!("acl-{grain}"));
        record.metadata.extend(
            serde_json::json!({
                "entity_type": "acl_record",
                "acl_truth_graph": "wisdom",
                "acl_target_entity_id": entity_id,
                "acl_target_grain": grain,
                "acl_version": 1,
                "acl_mode": mode
            })
            .as_object()
            .expect("ACL metadata")
            .clone(),
        );
        record
    };
    let node = GraphRecord::new(entity_id);
    let mut edge = GraphRecord::new(entity_id);
    edge.metadata
        .insert("source_ids".into(), serde_json::json!([]));
    edge.metadata
        .insert("target_ids".into(), serde_json::json!([]));
    let records = vec![node, acl("node", "private"), acl("edge", "public")];
    let reader = GraphAclPrincipal {
        principal_id: "reader".into(),
        ..Default::default()
    };

    let (visible_nodes, visible_edges) =
        graph_records_visible_by_acl(&records, &[edge], "wisdom", &reader);

    assert!(visible_nodes.is_empty());
    assert_eq!(
        visible_edges
            .iter()
            .map(|record| record.id.as_str())
            .collect::<Vec<_>>(),
        [entity_id]
    );
}

#[test]
fn wisdom_shared_and_group_acl_grants_are_truth_graph_scoped() {
    let direct_id = "wisdom-direct-share-node";
    let group_id = "wisdom-group-share-node";
    let acl = |id: &str, mode: &str, properties: Value| {
        let mut record = GraphRecord::new(format!("acl-{id}"));
        record.metadata.extend(
            serde_json::json!({
                "entity_type": "acl_record",
                "acl_truth_graph": "wisdom",
                "acl_target_entity_id": id,
                "acl_target_grain": "node",
                "acl_version": 1,
                "acl_mode": mode
            })
            .as_object()
            .expect("ACL metadata object")
            .clone(),
        );
        record.document = Some(serde_json::json!({"properties": properties}).to_string());
        record
    };
    let nodes = vec![
        GraphRecord::new(direct_id),
        acl(
            direct_id,
            "shared",
            serde_json::json!({"shared_with_principals": ["alice"]}),
        ),
        GraphRecord::new(group_id),
        acl(
            group_id,
            "group",
            serde_json::json!({"shared_with_groups": ["wisdom-reviewers"]}),
        ),
    ];
    let alice = GraphAclPrincipal {
        principal_id: "alice".to_owned(),
        groups: BTreeSet::new(),
        security_scope: None,
    };
    let (wisdom_direct_nodes, _) = graph_records_visible_by_acl(&nodes, &[], "wisdom", &alice);
    assert!(wisdom_direct_nodes.iter().any(|node| node.id == direct_id));
    assert!(!wisdom_direct_nodes.iter().any(|node| node.id == group_id));
    let (other_graph_nodes, _) = graph_records_visible_by_acl(&nodes, &[], "knowledge", &alice);
    assert!(other_graph_nodes.is_empty());

    let reviewer = GraphAclPrincipal {
        principal_id: "bob".to_owned(),
        groups: BTreeSet::from(["wisdom-reviewers".to_owned()]),
        security_scope: None,
    };
    let (wisdom_group_nodes, _) = graph_records_visible_by_acl(&nodes, &[], "wisdom", &reviewer);
    assert!(wisdom_group_nodes.iter().any(|node| node.id == group_id));
    assert!(!wisdom_group_nodes.iter().any(|node| node.id == direct_id));
}

#[test]
fn graph_node_item_usage_matches_shared_python_contract() {
    let fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/graph-acl-item-usage-v1.json"
    )))
    .expect("valid graph ACL item-use fixture");
    assert_eq!(fixture["schema_version"], 1);
    let principal = GraphAclPrincipal {
        principal_id: fixture["principal"]["id"]
            .as_str()
            .expect("principal ID")
            .to_owned(),
        groups: fixture["principal"]["groups"]
            .as_array()
            .expect("principal groups")
            .iter()
            .map(|value| value.as_str().expect("group").to_owned())
            .collect(),
        security_scope: fixture["principal"]["security_scope"]
            .as_str()
            .map(str::to_owned),
    };
    for case in fixture["cases"].as_array().expect("cases") {
        let node_id = case["node_id"].as_str().expect("node ID");
        let mention_span_counts = case["mention_span_counts"]
            .as_array()
            .expect("mention span counts");
        let mentions = mention_span_counts
            .iter()
            .map(|span_count| {
                let span_count = span_count.as_u64().expect("span count") as usize;
                serde_json::json!({"spans": vec![serde_json::json!({}); span_count]})
            })
            .collect::<Vec<_>>();
        let mut records = Vec::new();
        for acl_record in case["acl_records"].as_array().expect("ACL records") {
            let mut record = GraphRecord::new(format!("acl-{node_id}-{}", records.len()));
            record
                .metadata
                .insert("entity_type".into(), "acl_record".into());
            record
                .metadata
                .insert("acl_truth_graph".into(), fixture["truth_graph"].clone());
            record
                .metadata
                .insert("acl_target_entity_id".into(), node_id.into());
            record.metadata.insert(
                "acl_target_grain".into(),
                acl_record["grain"].as_str().expect("ACL grain").into(),
            );
            record.metadata.insert(
                "acl_version".into(),
                acl_record
                    .get("version")
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!(1)),
            );
            record.metadata.insert(
                "acl_mode".into(),
                acl_record["mode"].as_str().expect("ACL mode").into(),
            );
            if let Some(item_id) = acl_record["target_item_id"].as_str() {
                record
                    .metadata
                    .insert("acl_target_item_id".into(), item_id.into());
            }
            if let Some(owner_id) = acl_record["owner_id"].as_str() {
                record.metadata.insert("owner_id".into(), owner_id.into());
            }
            if acl_record.get("shared_with_principals").is_some()
                || acl_record.get("shared_with_groups").is_some()
            {
                record.document = Some(
                    serde_json::json!({
                        "properties": {
                            "shared_with_principals": acl_record
                                .get("shared_with_principals")
                                .cloned()
                                .unwrap_or_else(|| serde_json::json!([])),
                            "shared_with_groups": acl_record
                                .get("shared_with_groups")
                                .cloned()
                                .unwrap_or_else(|| serde_json::json!([]))
                        }
                    })
                    .to_string(),
                );
            }
            records.push(record);
        }
        let mut target = GraphRecord::new(node_id);
        let encoded_mentions = serde_json::to_string(&mentions).unwrap();
        if case["mentions_storage"].as_str() == Some("document_root") {
            target.document = Some(serde_json::json!({"mentions": encoded_mentions}).to_string());
        } else {
            target.metadata.insert(
                "mentions".into(),
                serde_json::Value::String(encoded_mentions),
            );
        }
        records.push(target);
        let (visible, _) = graph_records_visible_by_acl(&records, &[], "knowledge", &principal);
        let expected = case["expected_visible"]
            .as_bool()
            .expect("expected visibility");
        assert_eq!(
            visible.iter().any(|record| record.id == node_id),
            expected,
            "{node_id}"
        );
    }
}

#[test]
fn graph_edge_visibility_requires_visible_usage_and_existing_endpoints() {
    let principal = GraphAclPrincipal {
        principal_id: "alice".into(),
        groups: BTreeSet::new(),
        security_scope: None,
    };
    let acl_record = |id: &str, grain: &str, item_id: Option<&str>, mode: &str| {
        let mut record =
            GraphRecord::new(format!("acl-{id}-{grain}-{}", item_id.unwrap_or("entity")));
        record
            .metadata
            .insert("entity_type".into(), "acl_record".into());
        record
            .metadata
            .insert("acl_truth_graph".into(), "knowledge".into());
        record
            .metadata
            .insert("acl_target_entity_id".into(), id.into());
        record
            .metadata
            .insert("acl_target_grain".into(), grain.into());
        record
            .metadata
            .insert("acl_version".into(), serde_json::json!(1));
        record.metadata.insert("acl_mode".into(), mode.into());
        if let Some(item_id) = item_id {
            record
                .metadata
                .insert("acl_target_item_id".into(), item_id.into());
        }
        record
    };

    let edge_id = "edge-private-span";
    let grounding_id = format!("{edge_id}::mention::0");
    let span_id = format!("{grounding_id}::span::0");
    let mut edge = GraphRecord::new(edge_id);
    edge.metadata
        .insert("source_ids".into(), serde_json::json!(["visible-node"]));
    edge.metadata
        .insert("mentions".into(), serde_json::json!([{"spans": [{}]}]));
    let mut nodes = vec![
        acl_record(edge_id, "edge", None, "public"),
        acl_record(edge_id, "grounding", Some(&grounding_id), "public"),
        acl_record(edge_id, "span", Some(&span_id), "private"),
        acl_record("visible-node", "node", None, "public"),
        GraphRecord::new("visible-node"),
    ];
    let (visible_nodes, visible_edges) =
        graph_records_visible_by_acl(&nodes, &[edge], "knowledge", &principal);
    assert_eq!(visible_nodes.len(), 1);
    assert!(
        visible_edges.is_empty(),
        "private edge span must hide the edge"
    );

    let mut endpoint_edge = GraphRecord::new("edge-hidden-endpoint");
    endpoint_edge
        .metadata
        .insert("source_ids".into(), serde_json::json!(["hidden-node"]));
    nodes = vec![
        acl_record("edge-hidden-endpoint", "edge", None, "public"),
        acl_record("hidden-node", "node", None, "private"),
        GraphRecord::new("hidden-node"),
    ];
    let (_, visible_edges) =
        graph_records_visible_by_acl(&nodes, &[endpoint_edge.clone()], "knowledge", &principal);
    assert!(
        visible_edges.is_empty(),
        "an edge must not disclose an existing hidden endpoint"
    );

    nodes = vec![
        acl_record("edge-hidden-endpoint", "edge", None, "public"),
        acl_record("visible-node", "node", None, "public"),
        GraphRecord::new("visible-node"),
    ];
    endpoint_edge
        .metadata
        .insert("source_ids".into(), serde_json::json!(["visible-node"]));
    let (_, visible_edges) =
        graph_records_visible_by_acl(&nodes, &[endpoint_edge], "knowledge", &principal);
    assert_eq!(
        visible_edges.len(),
        1,
        "legacy edges without mention metadata remain readable with visible endpoints"
    );

    let mut malformed_reference = GraphRecord::new("edge-malformed-reference");
    malformed_reference
        .metadata
        .insert("source_edge_ids".into(), "not-json".into());
    nodes.push(acl_record(
        "edge-malformed-reference",
        "edge",
        None,
        "public",
    ));
    let (_, visible_edges) =
        graph_records_visible_by_acl(&nodes, &[malformed_reference], "knowledge", &principal);
    assert!(
        visible_edges.is_empty(),
        "malformed edge-reference metadata must fail closed"
    );
}

#[test]
fn graph_acl_projection_excludes_tombstoned_entities_and_their_edges() {
    let acl_record = |id: &str, grain: &str| {
        let mut record = GraphRecord::new(format!("acl-{id}-{grain}"));
        record.metadata.extend(
            serde_json::json!({
                "entity_type": "acl_record",
                "acl_truth_graph": "knowledge",
                "acl_target_entity_id": id,
                "acl_target_grain": grain,
                "acl_version": 1,
                "acl_mode": "public"
            })
            .as_object()
            .expect("ACL metadata is an object")
            .clone(),
        );
        record
    };
    let mut tombstoned_node = GraphRecord::new("node-deleted");
    tombstoned_node
        .metadata
        .insert("lifecycle_status".into(), "tombstoned".into());
    let mut edge = GraphRecord::new("edge-to-deleted-node");
    edge.metadata
        .insert("source_ids".into(), serde_json::json!(["node-active"]));
    edge.metadata
        .insert("target_ids".into(), serde_json::json!(["node-deleted"]));
    let nodes = vec![
        acl_record("node-active", "node"),
        acl_record("node-deleted", "node"),
        acl_record("edge-to-deleted-node", "edge"),
        GraphRecord::new("node-active"),
        tombstoned_node,
    ];
    let principal = GraphAclPrincipal::default();

    let (visible_nodes, visible_edges) =
        graph_records_visible_by_acl(&nodes, &[edge], "knowledge", &principal);

    assert_eq!(
        visible_nodes
            .iter()
            .map(|record| record.id.as_str())
            .collect::<Vec<_>>(),
        ["node-active"]
    );
    assert!(visible_edges.is_empty());
}

#[test]
fn graph_edge_visibility_fails_closed_on_non_string_endpoint_ids() {
    let fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/graph-acl-malformed-endpoint-v1.json"
    )))
    .expect("valid malformed endpoint ACL fixture");
    let edge_id = "malformed-endpoint-edge";
    let endpoint_id = fixture["source_endpoint_id"].as_str().unwrap();
    let acl_record = |id: &str, grain: &str, mode: &str| {
        let mut record = GraphRecord::new(format!("acl-{id}-{grain}"));
        record
            .metadata
            .insert("entity_type".into(), "acl_record".into());
        record
            .metadata
            .insert("acl_truth_graph".into(), fixture["truth_graph"].clone());
        record
            .metadata
            .insert("acl_target_entity_id".into(), id.into());
        record
            .metadata
            .insert("acl_target_grain".into(), grain.into());
        record
            .metadata
            .insert("acl_version".into(), serde_json::json!(1));
        record.metadata.insert("acl_mode".into(), mode.into());
        if mode == "private" {
            record
                .metadata
                .insert("owner_id".into(), "someone-else".into());
        }
        record
    };

    let mut edge = GraphRecord::new(edge_id);
    edge.metadata.insert(
        "source_ids".into(),
        serde_json::json!([fixture["source_endpoint_value"]]),
    );
    edge.metadata
        .insert("target_ids".into(), serde_json::json!([]));
    let nodes = vec![
        acl_record(edge_id, "edge", "public"),
        acl_record(endpoint_id, "node", "private"),
        GraphRecord::new(endpoint_id),
    ];
    let principal = GraphAclPrincipal {
        principal_id: fixture["principal"]["id"].as_str().unwrap().into(),
        groups: BTreeSet::new(),
        security_scope: fixture["principal"]["security_scope"]
            .as_str()
            .map(str::to_owned),
    };
    let (_, visible_edges) = graph_records_visible_by_acl(
        &nodes,
        &[edge],
        fixture["truth_graph"].as_str().unwrap(),
        &principal,
    );
    assert_eq!(
        !visible_edges.is_empty(),
        fixture["expected_edge_visible"].as_bool().unwrap()
    );
}

#[test]
fn graph_edge_usage_matches_shared_python_golden_contract() {
    let fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/graph-acl-edge-usage-v1.json"
    )))
    .expect("valid graph edge ACL fixture");
    assert_eq!(fixture["schema_version"], 1);
    let principal = GraphAclPrincipal {
        principal_id: fixture["principal"]["id"]
            .as_str()
            .expect("principal ID")
            .to_owned(),
        groups: BTreeSet::new(),
        security_scope: fixture["principal"]["security_scope"]
            .as_str()
            .map(str::to_owned),
    };
    let acl_record = |entity_id: &str, grain: &str, item_id: Option<&str>, mode: &str| {
        let mut record = GraphRecord::new(format!(
            "acl-{entity_id}-{grain}-{}",
            item_id.unwrap_or("entity")
        ));
        record
            .metadata
            .insert("entity_type".into(), "acl_record".into());
        record
            .metadata
            .insert("acl_truth_graph".into(), fixture["truth_graph"].clone());
        record
            .metadata
            .insert("acl_target_entity_id".into(), entity_id.into());
        record
            .metadata
            .insert("acl_target_grain".into(), grain.into());
        record
            .metadata
            .insert("acl_version".into(), serde_json::json!(1));
        record.metadata.insert("acl_mode".into(), mode.into());
        if let Some(item_id) = item_id {
            record
                .metadata
                .insert("acl_target_item_id".into(), item_id.into());
        }
        if mode == "private" {
            record.metadata.insert(
                "owner_id".into(),
                fixture["owner_id"].as_str().expect("owner ID").into(),
            );
        } else if mode == "shared" {
            record.metadata.insert(
                "shared_with_principals".into(),
                fixture["shared_with_principals"].clone(),
            );
        }
        record
    };

    for case in fixture["cases"].as_array().expect("fixture cases") {
        let edge_id = case["edge_id"].as_str().expect("edge ID");
        let source_id = case["source_node_id"].as_str().expect("source ID");
        let target_id = case["target_node_id"].as_str().expect("target ID");
        let mut nodes = vec![acl_record(edge_id, "edge", None, "shared")];
        for (endpoint_id, exists, mode) in [
            (
                source_id,
                case["source_node_exists"].as_bool().unwrap_or(true),
                case["source_node_mode"].as_str().expect("source ACL mode"),
            ),
            (
                target_id,
                case["target_node_exists"].as_bool().unwrap_or(true),
                case["target_node_mode"].as_str().expect("target ACL mode"),
            ),
        ] {
            if exists {
                nodes.push(acl_record(endpoint_id, "node", None, mode));
                nodes.push(GraphRecord::new(endpoint_id));
            }
        }
        let mut edge = GraphRecord::new(edge_id);
        edge.metadata
            .insert("source_ids".into(), serde_json::json!([source_id]));
        edge.metadata
            .insert("target_ids".into(), serde_json::json!([target_id]));
        let mut mentions = Vec::new();
        for (mention_index, span_count) in case["mention_span_counts"]
            .as_array()
            .expect("mention span counts")
            .iter()
            .enumerate()
        {
            let grounding_id = format!("{edge_id}::mention::{mention_index}");
            nodes.push(acl_record(
                edge_id,
                "grounding",
                Some(&grounding_id),
                "public",
            ));
            let span_count = span_count.as_u64().expect("span count") as usize;
            let mut spans = Vec::new();
            for span_index in 0..span_count {
                let span_id = format!("{grounding_id}::span::{span_index}");
                let private = case["private_span"].as_array().is_some_and(|indices| {
                    indices[0].as_u64() == Some(mention_index as u64)
                        && indices[1].as_u64() == Some(span_index as u64)
                });
                nodes.push(acl_record(
                    edge_id,
                    "span",
                    Some(&span_id),
                    if private { "private" } else { "public" },
                ));
                spans.push(serde_json::json!({}));
            }
            mentions.push(serde_json::json!({"spans": spans}));
        }
        if !mentions.is_empty() {
            edge.metadata
                .insert("mentions".into(), serde_json::json!(mentions));
        }
        let mut edges = vec![edge];
        if let Some(referenced_edge_id) = case["source_edge_id"].as_str() {
            edges[0].metadata.insert(
                "source_edge_ids".into(),
                serde_json::json!([referenced_edge_id]),
            );
            if case["source_edge_exists"].as_bool().unwrap_or(true) {
                nodes.push(acl_record(referenced_edge_id, "edge", None, "private"));
                edges.push(GraphRecord::new(referenced_edge_id));
            }
        }
        let (_, visible_edges) = graph_records_visible_by_acl(
            &nodes,
            &edges,
            fixture["truth_graph"].as_str().expect("truth graph"),
            &principal,
        );
        assert_eq!(
            visible_edges.iter().any(|record| record.id == edge_id),
            case["expected_visible"]
                .as_bool()
                .expect("expected visibility"),
            "{edge_id}"
        );
    }
}

#[test]
fn graph_node_malformed_item_usage_fails_closed_but_legacy_absence_remains_readable() {
    let principal = GraphAclPrincipal {
        principal_id: "alice".into(),
        groups: BTreeSet::new(),
        security_scope: None,
    };
    let acl = |node_id: &str| {
        let mut record = GraphRecord::new(format!("acl-{node_id}"));
        record
            .metadata
            .insert("entity_type".into(), "acl_record".into());
        record
            .metadata
            .insert("acl_truth_graph".into(), "knowledge".into());
        record
            .metadata
            .insert("acl_target_entity_id".into(), node_id.into());
        record
            .metadata
            .insert("acl_target_grain".into(), "node".into());
        record
            .metadata
            .insert("acl_version".into(), serde_json::json!(1));
        record.metadata.insert("acl_mode".into(), "public".into());
        record
    };
    let mut malformed_json = GraphRecord::new("malformed-json");
    malformed_json
        .metadata
        .insert("mentions".into(), "not-json".into());
    let mut wrong_shape = GraphRecord::new("wrong-shape");
    wrong_shape
        .metadata
        .insert("mentions".into(), serde_json::json!({"unexpected": true}));
    let legacy_without_mentions = GraphRecord::new("legacy-without-mentions");
    let nodes = vec![
        acl("malformed-json"),
        malformed_json,
        acl("wrong-shape"),
        wrong_shape,
        acl("legacy-without-mentions"),
        legacy_without_mentions,
    ];

    let (visible, _) = graph_records_visible_by_acl(&nodes, &[], "knowledge", &principal);
    let visible_ids = visible
        .iter()
        .map(|record| record.id.as_str())
        .collect::<BTreeSet<_>>();
    assert_eq!(visible_ids, BTreeSet::from(["legacy-without-mentions"]));
}

#[test]
fn graph_acl_history_matches_shared_golden_contract() {
    let fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/graph-acl-history-v1.json"
    )))
    .expect("valid graph ACL history fixture");
    assert_eq!(fixture["schema_version"], 1);
    let principal = GraphAclPrincipal {
        principal_id: fixture["principal"]["id"]
            .as_str()
            .expect("principal ID")
            .to_owned(),
        groups: fixture["principal"]["groups"]
            .as_array()
            .expect("principal groups")
            .iter()
            .map(|value| value.as_str().expect("group").to_owned())
            .collect(),
        security_scope: fixture["principal"]["security_scope"]
            .as_str()
            .map(str::to_owned),
    };
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    for target in fixture["targets"].as_array().expect("targets") {
        let id = target["id"].as_str().expect("target ID");
        let grain = target["grain"].as_str().expect("target grain");
        let target_record = GraphRecord::new(id);
        for (index, rule) in target["acl_records"]
            .as_array()
            .expect("ACL record history")
            .iter()
            .enumerate()
        {
            let mut acl = GraphRecord::new(format!("acl-{id}-{index}"));
            acl.metadata
                .insert("entity_type".into(), "acl_record".into());
            acl.metadata
                .insert("acl_truth_graph".into(), fixture["truth_graph"].clone());
            acl.metadata
                .insert("acl_target_entity_id".into(), id.into());
            acl.metadata.insert("acl_target_grain".into(), grain.into());
            acl.metadata.insert(
                "acl_version".into(),
                rule["version"].as_i64().expect("version").into(),
            );
            acl.metadata.insert(
                "acl_mode".into(),
                rule["mode"].as_str().expect("mode").into(),
            );
            if let Some(owner) = rule["owner_id"].as_str() {
                acl.metadata.insert("owner_id".into(), owner.into());
            }
            if let Some(scope) = rule["security_scope"].as_str() {
                acl.metadata.insert("security_scope".into(), scope.into());
            }
            if rule["tombstoned"].as_bool().unwrap_or(false) {
                acl.metadata.insert("tombstoned".into(), true.into());
            }
            let mut properties = serde_json::Map::new();
            for field in ["shared_with_principals", "shared_with_groups"] {
                if let Some(values) = rule[field].as_array() {
                    properties.insert(field.into(), Value::Array(values.clone()));
                }
            }
            if !properties.is_empty() {
                acl.document = Some(serde_json::json!({"properties": properties}).to_string());
            }
            nodes.push(acl);
        }
        if grain == "node" {
            nodes.push(target_record);
        } else {
            edges.push(target_record);
        }
    }

    let (visible_nodes, visible_edges) = graph_records_visible_by_acl(
        &nodes,
        &edges,
        fixture["truth_graph"].as_str().expect("truth graph"),
        &principal,
    );
    let node_ids = visible_nodes
        .iter()
        .map(|record| record.id.as_str())
        .collect::<Vec<_>>();
    let edge_ids = visible_edges
        .iter()
        .map(|record| record.id.as_str())
        .collect::<Vec<_>>();
    let expected_nodes = fixture["expected_visible"]["nodes"]
        .as_array()
        .expect("expected nodes")
        .iter()
        .map(|value| value.as_str().expect("node ID"))
        .collect::<Vec<_>>();
    let expected_edges = fixture["expected_visible"]["edges"]
        .as_array()
        .expect("expected edges")
        .iter()
        .map(|value| value.as_str().expect("edge ID"))
        .collect::<Vec<_>>();
    assert_eq!(node_ids, expected_nodes);
    assert_eq!(edge_ids, expected_edges);

    let reversed_nodes = nodes.iter().rev().cloned().collect::<Vec<_>>();
    let reversed_edges = edges.iter().rev().cloned().collect::<Vec<_>>();
    let (reordered_visible_nodes, reordered_visible_edges) = graph_records_visible_by_acl(
        &reversed_nodes,
        &reversed_edges,
        fixture["truth_graph"].as_str().expect("truth graph"),
        &principal,
    );
    let mut reordered_node_ids = reordered_visible_nodes
        .iter()
        .map(|record| record.id.as_str())
        .collect::<Vec<_>>();
    let mut reordered_edge_ids = reordered_visible_edges
        .iter()
        .map(|record| record.id.as_str())
        .collect::<Vec<_>>();
    reordered_node_ids.sort_unstable();
    reordered_edge_ids.sort_unstable();
    let mut expected_node_ids = expected_nodes.clone();
    let mut expected_edge_ids = expected_edges.clone();
    expected_node_ids.sort_unstable();
    expected_edge_ids.sort_unstable();
    assert_eq!(reordered_node_ids, expected_node_ids);
    assert_eq!(reordered_edge_ids, expected_edge_ids);
}
