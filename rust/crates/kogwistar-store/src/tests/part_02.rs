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
            record
                .metadata
                .insert("acl_version".into(), serde_json::json!(1));
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

#[test]
fn tool_audit_projection_is_scope_filtered_and_canonical() {
    let mut call = GraphRecord::new("tool-call-1");
    call.metadata
        .insert("entity_type".into(), "tool_call".into());
    call.metadata
        .insert("tool_name".into(), "graph.read".into());
    call.metadata
        .insert("conversation_id".into(), "conversation-1".into());
    call.metadata.insert("turn_index".into(), 2.into());
    call.document = Some(
        serde_json::json!({
            "properties": {"status": "completed", "count": 1}
        })
        .to_string(),
    );

    let mut result = GraphRecord::new("tool-result-1");
    result
        .metadata
        .insert("entity_type".into(), "tool_result".into());
    result
        .metadata
        .insert("tool_name".into(), "graph.read".into());
    result
        .metadata
        .insert("conversation_id".into(), "conversation-1".into());
    result.document = Some(
        serde_json::json!({
            "properties": {"status": "completed"}
        })
        .to_string(),
    );

    let mut other = GraphRecord::new("tool-result-other");
    other
        .metadata
        .insert("entity_type".into(), "tool_result".into());
    other
        .metadata
        .insert("tool_name".into(), "graph.read".into());
    other
        .metadata
        .insert("conversation_id".into(), "conversation-2".into());

    let mut tombstoned = GraphRecord::new("tool-call-deleted");
    tombstoned
        .metadata
        .insert("entity_type".into(), "tool_call".into());
    tombstoned
        .metadata
        .insert("tool_name".into(), "secret".into());
    tombstoned
        .metadata
        .insert("conversation_id".into(), "conversation-1".into());
    tombstoned
        .metadata
        .insert("lifecycle_status".into(), "tombstoned".into());

    let items = tool_audit_items_from_visible_records(
        &[result, call, other, tombstoned],
        Some("conversation-1"),
        10,
    )
    .unwrap();
    assert_eq!(items.len(), 2);
    assert_eq!(items[0].node_id, "tool-call-1");
    assert_eq!(items[0].turn_index, Some(2));
    assert_eq!(items[0].properties["status"], "completed");
    assert_eq!(items[1].node_id, "tool-result-1");
    assert!(
        tool_audit_items_from_visible_records(&[GraphRecord::new("ignored")], None, 0)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn tool_audit_projection_fails_closed_for_malformed_visible_records() {
    let mut malformed = GraphRecord::new("tool-call-invalid");
    malformed
        .metadata
        .insert("entity_type".into(), "tool_call".into());
    malformed
        .metadata
        .insert("conversation_id".into(), "conversation-1".into());
    let error = tool_audit_items_from_visible_records(&[malformed], None, 10).unwrap_err();
    assert_eq!(
        error,
        kogwistar_contracts::ToolAuditContractError::EmptyField { field: "tool_name" }
    );
}

#[test]
fn conversation_transcript_projection_is_ordered_and_scope_filtered() {
    let mut assistant = GraphRecord::new("assistant-1");
    assistant
        .metadata
        .insert("entity_type".into(), "assistant_turn".into());
    assistant
        .metadata
        .insert("conversation_id".into(), "conversation-1".into());
    assistant.metadata.insert("turn_index".into(), 1.into());
    assistant.metadata.insert("role".into(), "assistant".into());
    assistant.metadata.insert("summary".into(), "answer".into());

    let mut user = GraphRecord::new("user-0");
    user.metadata
        .insert("entity_type".into(), "conversation_turn".into());
    user.metadata
        .insert("conversation_id".into(), "conversation-1".into());
    user.metadata.insert("turn_index".into(), 0.into());
    user.metadata.insert("role".into(), "user".into());
    user.metadata.insert("summary".into(), "question".into());

    let mut other = user.clone();
    other.id = "other-user".into();
    other
        .metadata
        .insert("conversation_id".into(), "conversation-2".into());

    let transcript = conversation_transcript_from_visible_records(
        &[assistant, other, user],
        "conversation-1",
        10,
    )
    .unwrap();
    assert_eq!(transcript.turns.len(), 2);
    assert_eq!(transcript.turns[0].node_id, "user-0");
    assert_eq!(transcript.turns[1].node_id, "assistant-1");
}

#[test]
fn conversation_transcript_projection_rejects_duplicate_turn_indexes() {
    let mut first = GraphRecord::new("turn-a");
    first.metadata.extend(
        serde_json::json!({
            "entity_type": "conversation_turn",
            "conversation_id": "conversation-1",
            "turn_index": 0,
            "role": "user",
            "summary": "one"
        })
        .as_object()
        .unwrap()
        .clone(),
    );
    let mut second = first.clone();
    second.id = "turn-b".into();
    let error = conversation_transcript_from_visible_records(&[first, second], "conversation-1", 0)
        .unwrap_err();
    assert_eq!(error, ConversationContractError::InvalidTurnOrder);
}

#[test]
fn conversation_summary_projection_derives_owner_status_and_tail() {
    let mut start = GraphRecord::new("conversation-start");
    start.metadata.extend(
        serde_json::json!({
            "entity_type": "conversation_start",
            "conversation_id": "conversation-1",
            "user_id": "alice",
            "status": "active"
        })
        .as_object()
        .unwrap()
        .clone(),
    );
    let mut turn = GraphRecord::new("turn-0");
    turn.metadata.extend(
        serde_json::json!({
            "entity_type": "conversation_turn",
            "conversation_id": "conversation-1",
            "turn_index": 0,
            "role": "user",
            "summary": "question"
        })
        .as_object()
        .unwrap()
        .clone(),
    );
    let summary =
        conversation_summary_from_visible_records(&[start, turn], "conversation-1").unwrap();
    assert_eq!(summary.user_id, "alice");
    assert_eq!(summary.start_node_id, "conversation-start");
    assert_eq!(summary.tail_node_id, "turn-0");
    assert_eq!(summary.turn_count, 1);
}

#[test]
fn conversation_list_projection_is_user_scoped_and_counts_visible_turns() {
    let mut start_alice = GraphRecord::new("start-alice");
    start_alice.metadata.extend(
        serde_json::json!({
            "entity_type": "conversation_start",
            "conversation_id": "conversation-alice",
            "user_id": "alice",
            "status": "active"
        })
        .as_object()
        .unwrap()
        .clone(),
    );
    let mut start_bob = start_alice.clone();
    start_bob.id = "start-bob".to_owned();
    start_bob
        .metadata
        .insert("conversation_id".to_owned(), "conversation-bob".into());
    start_bob
        .metadata
        .insert("user_id".to_owned(), "bob".into());
    let mut first_turn = GraphRecord::new("turn-1");
    first_turn.metadata.extend(
        serde_json::json!({
            "entity_type": "conversation_turn",
            "conversation_id": "conversation-alice",
            "turn_index": 0
        })
        .as_object()
        .unwrap()
        .clone(),
    );
    let mut second_turn = first_turn.clone();
    second_turn.id = "turn-2".to_owned();
    second_turn
        .metadata
        .insert("entity_type".to_owned(), "assistant_turn".into());
    second_turn
        .metadata
        .insert("turn_index".to_owned(), 1.into());

    let records = [start_alice, start_bob, first_turn, second_turn];
    let entries = conversation_list_for_user_from_visible_records(&records, "alice");
    assert_eq!(
        entries,
        vec![ConversationListEntry {
            id: "conversation-alice".to_owned(),
            start_node_id: "start-alice".to_owned(),
            status: "active".to_owned(),
            turn_count: 2,
        }]
    );

    let mut chunked = ConversationListAccumulator::new("alice");
    chunked.extend_visible_records(&records[..1]);
    chunked.extend_visible_records(&records[1..3]);
    chunked.extend_visible_records(&records[3..]);
    assert_eq!(chunked.finish(), entries);
}

#[test]
fn context_snapshot_projection_selects_latest_matching_scope() {
    let mut older = GraphRecord::new("snapshot-old");
    older
        .metadata
        .insert("entity_type".into(), "context_snapshot".into());
    older
        .metadata
        .insert("conversation_id".into(), "conversation-1".into());
    older.metadata.insert("run_id".into(), "run-1".into());
    older.metadata.insert("stage".into(), "answer".into());
    older.metadata.insert("snapshot_seq".into(), 3.into());
    older.document = Some(
        serde_json::json!({
            "metadata": {"stage": "answer"},
            "properties": {"value": "old"}
        })
        .to_string(),
    );

    let mut newer = older.clone();
    newer.id = "snapshot-new".into();
    newer.metadata.insert("snapshot_seq".into(), 4.into());
    newer.document = Some(
        serde_json::json!({
            "metadata": {"stage": "answer"},
            "properties": {"value": "new"}
        })
        .to_string(),
    );

    let snapshot = context_snapshot_from_visible_records(
        &[older, newer],
        "conversation-1",
        Some("run-1"),
        Some("answer"),
    )
    .unwrap();
    assert_eq!(snapshot.snapshot_node_id, "snapshot-new");
    assert_eq!(snapshot.properties["value"], "new");
}

#[test]
fn context_snapshot_projection_matches_python_run_and_attempt_order_fixture() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/context-snapshot-order-v1.json"
    )))
    .expect("valid shared Python context-snapshot ordering fixture");
    let conversation_id = fixture["conversation_id"].as_str().unwrap();
    let run_id = fixture["run_id"].as_str().unwrap();
    let stage = fixture["stage"].as_str().unwrap();
    let records = fixture["candidates"]
        .as_array()
        .unwrap()
        .iter()
        .map(|candidate| {
            let id = candidate["id"].as_str().unwrap().to_owned();
            let metadata = serde_json::json!({
                "entity_type": "context_snapshot",
                "conversation_id": conversation_id,
                "run_id": run_id,
                "stage": stage,
                "run_step_seq": candidate["run_step_seq"],
                "attempt_seq": candidate["attempt_seq"]
            })
            .as_object()
            .unwrap()
            .clone();
            GraphRecord {
                id,
                document: Some(
                    serde_json::json!({
                        "metadata": metadata,
                        "properties": candidate["properties"]
                    })
                    .to_string(),
                ),
                metadata,
                embedding: None,
            }
        })
        .collect::<Vec<_>>();

    let snapshot =
        context_snapshot_from_visible_records(&records, conversation_id, Some(run_id), Some(stage))
            .expect("Python run_step_seq and attempt_seq metadata is orderable");
    assert_eq!(
        snapshot.snapshot_node_id,
        fixture["expected_snapshot_node_id"].as_str().unwrap()
    );
    assert_eq!(snapshot.properties["prompt_messages"], "latest retry");
}

#[test]
fn context_snapshot_projection_rejects_unordered_candidates() {
    let mut record = GraphRecord::new("snapshot-invalid");
    record
        .metadata
        .insert("entity_type".into(), "context_snapshot".into());
    record
        .metadata
        .insert("conversation_id".into(), "conversation-1".into());
    let error =
        context_snapshot_from_visible_records(&[record], "conversation-1", None, None).unwrap_err();
    assert_eq!(error, ConversationContractError::MissingSnapshotOrder);
}
use kogwistar_contracts::{GraphModelClass, GraphModelEnvelope};

fn fixture_payload(index: usize) -> Value {
    let fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/python-graph-model-payloads-v1.json"
    )))
    .expect("valid graph model fixture");
    fixture["cases"][index]["payload"].clone()
}

fn typed_mutation(entity_kind: &str, payload: Value) -> GraphMutation {
    let id = payload["id"].as_str().expect("fixture id").to_owned();
    GraphMutation {
        scope: GraphScope {
            namespace: "ns".to_owned(),
            workspace_id: Some("ws".to_owned()),
            graph_space: Some("facts".to_owned()),
        },
        table: if entity_kind == "edge" {
            "gke_edges".to_owned()
        } else {
            "gke_nodes".to_owned()
        },
        entity_kind: entity_kind.to_owned(),
        event_id: "event-1".to_owned(),
        op: "ADD".to_owned(),
        payload: payload.clone(),
        record: GraphRecord {
            id,
            document: None,
            metadata: payload["metadata"].as_object().cloned().unwrap_or_default(),
            embedding: None,
        },
        embedding_dim: 0,
    }
}

#[test]
fn runtime_projection_contract_has_one_stable_shape() {
    let mut payload = Map::new();
    payload.insert("status".to_owned(), Value::String("running".to_owned()));
    let projection = runtime_projection_write(payload, 7, "ready");
    assert_eq!(
        runtime_checkpoint_namespace("conv"),
        "conv:workflow_checkpoint_latest"
    );
    assert_eq!(runtime_status_namespace("conv"), "conv:workflow_run_status");
    assert_eq!(projection.last_authoritative_seq, 7);
    assert_eq!(projection.last_materialized_seq, 7);
    assert_eq!(
        projection.projection_schema_version,
        RUNTIME_PROJECTION_SCHEMA_VERSION
    );
    assert_eq!(projection.materialization_status, "ready");
}

#[test]
fn typed_graph_mutation_accepts_matching_payload_and_identity() {
    let payload = fixture_payload(1);
    let mutation = typed_mutation("edge", payload.clone());
    let envelope = GraphModelEnvelope::new(GraphModelClass::Edge, payload);
    validate_typed_graph_mutation(&mutation, &envelope).expect("valid typed mutation");
}

#[test]
fn typed_graph_mutation_rejects_kind_and_identity_confusion() {
    let edge_payload = fixture_payload(1);
    let edge_mutation = typed_mutation("node", edge_payload.clone());
    let edge_envelope = GraphModelEnvelope::new(GraphModelClass::Edge, edge_payload);
    assert!(matches!(
        validate_typed_graph_mutation(&edge_mutation, &edge_envelope),
        Err(StoreError::InvalidEntityEventPayload { .. })
    ));

    let node_payload = fixture_payload(0);
    let wrong_edge_envelope = GraphModelEnvelope::new(GraphModelClass::Edge, node_payload.clone());
    let wrong_edge_mutation = typed_mutation("node", node_payload);
    let error = validate_typed_graph_mutation(&wrong_edge_mutation, &wrong_edge_envelope)
        .expect_err("malformed edge family must fail before the later kind mismatch");
    assert!(matches!(
        error,
        StoreError::InvalidEntityEventPayload { message } if message.contains("source_ids")
    ));

    let node_payload = fixture_payload(0);
    let mut node_mutation = typed_mutation("node", node_payload.clone());
    node_mutation.record.id = "different-id".to_owned();
    let node_envelope = GraphModelEnvelope::new(GraphModelClass::Node, node_payload);
    assert!(matches!(
        validate_typed_graph_mutation(&node_mutation, &node_envelope),
        Err(StoreError::InvalidEntityEventPayload { .. })
    ));
}

#[test]
fn typed_graph_mutation_cannot_bypass_base_mutation_validation() {
    let payload = fixture_payload(0);
    let envelope = GraphModelEnvelope::new(GraphModelClass::Node, payload.clone());
    let valid = typed_mutation("node", payload);

    let mut invalid = valid.clone();
    invalid.table = "gke_edges".to_owned();
    assert!(validate_typed_graph_mutation(&invalid, &envelope).is_err());

    let mut invalid = valid.clone();
    invalid.scope.namespace.clear();
    assert!(validate_typed_graph_mutation(&invalid, &envelope).is_err());

    let mut invalid = valid.clone();
    invalid.op = "DELETE".to_owned();
    assert!(validate_typed_graph_mutation(&invalid, &envelope).is_err());

    let mut invalid = valid.clone();
    invalid.payload = Value::Array(Vec::new());
    assert!(validate_typed_graph_mutation(&invalid, &envelope).is_err());
}

#[test]
fn graph_projection_target_mapping_is_shared_and_exact() {
    let targets = [
        ("node", "gke_nodes"),
        ("edge", "gke_edges"),
        ("document", "gke_documents"),
        ("domain", "gke_domains"),
    ];
    for (entity_kind, table) in targets {
        assert_eq!(graph_projection_table(entity_kind), Ok(table));
        validate_graph_projection_target(table, entity_kind)
            .expect("matching projection target should be accepted");
        let mutation = GraphMutation {
            scope: GraphScope {
                namespace: "ns".to_owned(),
                workspace_id: None,
                graph_space: None,
            },
            table: table.to_owned(),
            entity_kind: entity_kind.to_owned(),
            event_id: format!("{entity_kind}-event"),
            op: "ADD".to_owned(),
            payload: if matches!(entity_kind, "node" | "edge") {
                serde_json::json!({"id": format!("{entity_kind}-id")})
            } else {
                serde_json::json!({})
            },
            record: GraphRecord::new(format!("{entity_kind}-id")),
            embedding_dim: 0,
        };
        validate_graph_mutation(&mutation).expect("projection mutation should be valid");
        for (_, other_table) in targets {
            if other_table != table {
                assert!(validate_graph_projection_target(other_table, entity_kind).is_err());
            }
        }
    }
    assert!(graph_projection_table("unknown").is_err());
    assert!(validate_graph_lifecycle_target("gke_documents", "document").is_err());
    assert!(validate_graph_lifecycle_target("gke_domains", "domain").is_err());
    assert!(validate_graph_lifecycle_target("gke_nodes", "node").is_ok());
    assert!(validate_graph_lifecycle_target("gke_edges", "edge").is_ok());
}

#[test]
fn direct_graph_mutation_validation_rejects_effect_bypass_inputs() {
    let mut mutation = typed_mutation("node", fixture_payload(0));
    mutation.table = "gke_edges".to_owned();
    assert!(matches!(
        validate_graph_mutation(&mutation),
        Err(StoreError::InvalidEntityEventPayload { .. })
    ));

    mutation.table = "gke_nodes".to_owned();
    mutation.payload = Value::Array(Vec::new());
    assert!(matches!(
        validate_graph_mutation(&mutation),
        Err(StoreError::InvalidEntityEventPayload { .. })
    ));
}

#[test]
fn direct_graph_mutation_validation_rejects_unrequested_scope_dimensions() {
    let mut mutation = typed_mutation("node", fixture_payload(0));
    mutation.scope.workspace_id = None;
    mutation.scope.graph_space = None;
    mutation.record.metadata.insert(
        "workspace_id".to_owned(),
        Value::String("foreign".to_owned()),
    );
    assert!(validate_graph_mutation(&mutation).is_err());

    mutation.record.metadata.remove("workspace_id");
    mutation.payload["metadata"]["graph_space"] = Value::String("foreign".to_owned());
    assert!(validate_graph_mutation(&mutation).is_err());
}

#[test]
fn graph_mutation_replay_requires_identical_effect_identity_and_payload() {
    let mutation = typed_mutation("node", fixture_payload(0));
    let event = EntityEvent {
        namespace: mutation.scope.namespace.clone(),
        seq: 1,
        event_id: mutation.event_id.clone(),
        entity_kind: mutation.entity_kind.clone(),
        entity_id: mutation.record.id.clone(),
        op: mutation.op.clone(),
        payload: mutation.payload.clone(),
    };
    assert_eq!(validate_graph_mutation_replay(&mutation, &event), Ok(()));

    let mut altered = event.clone();
    altered.entity_kind = "edge".to_owned();
    assert!(validate_graph_mutation_replay(&mutation, &altered).is_err());
    let mut altered = event.clone();
    altered.entity_id.push_str("-other");
    assert!(validate_graph_mutation_replay(&mutation, &altered).is_err());
    let mut altered = event.clone();
    altered.op = "REPLACE".to_owned();
    assert!(validate_graph_mutation_replay(&mutation, &altered).is_err());
    let mut altered = event.clone();
    altered.payload["summary"] = Value::String("different".to_owned());
    assert!(validate_graph_mutation_replay(&mutation, &altered).is_err());
    let mut altered = event.clone();
    altered.event_id.push_str("-other");
    assert!(validate_graph_mutation_replay(&mutation, &altered).is_err());
    let mut altered = event;
    altered.namespace.push_str("-other");
    assert!(validate_graph_mutation_replay(&mutation, &altered).is_err());
}

#[test]
fn graph_mutation_batch_validation_rejects_scope_and_index_drift() {
    let first = typed_mutation("node", fixture_payload(0));
    let mut second = typed_mutation("edge", fixture_payload(1));
    second.scope.namespace = "other".to_owned();
    assert!(matches!(
        validate_graph_mutation_batch(&GraphMutationBatch {
            mutations: vec![first.clone(), second],
            index_jobs: Vec::new(),
        }),
        Err(StoreError::InvalidEntityEventPayload { .. })
    ));

    assert!(matches!(
        validate_graph_mutation_batch(&GraphMutationBatch {
            mutations: vec![first],
            index_jobs: vec![NewIndexJob {
                job_id: "job-1".to_owned(),
                namespace: "other".to_owned(),
                entity_kind: "node".to_owned(),
                entity_id: "n-1".to_owned(),
                index_kind: "doc".to_owned(),
                op: "UPSERT".to_owned(),
                payload_json: None,
                max_retries: 0,
            }],
        }),
        Err(StoreError::InvalidEntityEventPayload { .. })
    ));
}

#[test]
fn opaque_payload_metadata_does_not_select_storage_scope() {
    // Python's kg_graph_type is engine configuration, not an entity field.
    // A similarly named payload value must remain opaque to the store.
    for metadata_value in [
        "knowledge",
        "conversation",
        "workflow",
        "wisdom",
        "custom-role",
    ] {
        for (entity_kind, fixture_index) in [("node", 0), ("edge", 1)] {
            let mut mutation = typed_mutation(entity_kind, fixture_payload(fixture_index));
            mutation.payload["metadata"]["engine_type"] = Value::String(metadata_value.into());
            mutation.record.metadata.insert(
                "engine_type".to_owned(),
                Value::String(metadata_value.into()),
            );

            assert_eq!(mutation.scope.graph_space.as_deref(), Some("facts"));
            assert!(
                validate_graph_mutation_batch(&GraphMutationBatch {
                    mutations: vec![mutation],
                    index_jobs: Vec::new(),
                })
                .is_ok(),
                "opaque metadata {metadata_value:?} must not specialize {entity_kind} storage"
            );
        }
    }
}

#[test]
fn graph_mutation_batch_rejects_index_retry_counts_outside_portable_range() {
    let mutation = typed_mutation("node", fixture_payload(0));
    for max_retries in [-1, i32::MAX as i64 + 1] {
        let result = validate_graph_mutation_batch(&GraphMutationBatch {
            mutations: vec![mutation.clone()],
            index_jobs: vec![NewIndexJob {
                job_id: "job-1".to_owned(),
                namespace: mutation.scope.namespace.clone(),
                entity_kind: "node".to_owned(),
                entity_id: mutation.record.id.clone(),
                index_kind: "doc".to_owned(),
                op: "UPSERT".to_owned(),
                payload_json: None,
                max_retries,
            }],
        });
        assert!(matches!(
            result,
            Err(StoreError::InvalidEntityEventPayload { .. })
        ));
    }

    let boundary = validate_graph_mutation_batch(&GraphMutationBatch {
        mutations: vec![mutation.clone()],
        index_jobs: vec![NewIndexJob {
            job_id: "job-boundary".to_owned(),
            namespace: mutation.scope.namespace.clone(),
            entity_kind: "node".to_owned(),
            entity_id: mutation.record.id.clone(),
            index_kind: "doc".to_owned(),
            op: "UPSERT".to_owned(),
            payload_json: None,
            max_retries: i32::MAX as i64,
        }],
    });
    assert!(boundary.is_ok());
}

#[test]
fn graph_mutation_rejects_payload_and_projection_identity_drift() {
    let payload = fixture_payload(0);
    let mut mutation = typed_mutation("node", payload);
    mutation.record.id = "different-node".to_owned();

    assert!(matches!(
        validate_graph_mutation(&mutation),
        Err(StoreError::InvalidEntityEventPayload { .. })
    ));
}
