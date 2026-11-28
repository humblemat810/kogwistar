    #[test]
    fn strict_edge_extends_the_shared_graph_entity_interface() {
        fn assert_persistent_capabilities<T>()
        where
            T: GraphEntity + HasMetadata + HasLifecycle + HasLevel,
        {
        }

        fn assert_edge_capabilities<T>()
        where
            T: EdgeEntity + HasMetadata + HasLifecycle + HasLevel,
        {
        }

        assert_persistent_capabilities::<PersistentNodeV1>();
        assert_persistent_capabilities::<PersistentEdgeV1>();
        assert_edge_capabilities::<PersistentEdgeV1>();

        fn inspect(entity: &impl GraphEntity) -> (&'static str, Option<&str>) {
            let kind = match entity.graph_entity_kind() {
                GraphEntityKind::Node => "node",
                GraphEntityKind::Edge => "edge",
            };
            (kind, entity.graph_entity_id())
        }

        let fixture = fixture();
        let node = GraphModelEnvelope::new(
            GraphModelClass::Node,
            fixture["cases"][0]["payload"].clone(),
        )
        .decode_persistent_v1()
        .unwrap();
        let edge = GraphModelEnvelope::new(
            GraphModelClass::Edge,
            fixture["cases"][1]["payload"].clone(),
        )
        .decode_persistent_v1()
        .unwrap();
        assert_eq!(inspect(&node), ("node", Some("parity-node")));
        assert_eq!(inspect(&edge), ("edge", Some("parity-edge")));

        let PersistentGraphEntityV1::Edge(edge) = edge else {
            panic!("expected strict edge");
        };
        assert_eq!(edge.source_node_ids().unwrap().len(), 1);
        assert_eq!(edge.relation(), Some("supports"));
    }

    #[test]
    fn persistent_v2_keeps_shared_entity_capabilities_and_exact_hierarchy_level() {
        fn assert_common_capabilities<T>()
        where
            T: GraphEntity + HasMetadata + HasLifecycle + HasLevel + HasExactLevel,
        {
        }

        assert_common_capabilities::<PersistentGraphEntityV2>();

        let fixture = fixture();
        let huge = "1606938044258990275541962092341162602522202993782792835301376";
        let huge_number: Value = serde_json::from_str(huge).expect("arbitrary JSON integer");

        for (case_index, model_class, expected_id) in [
            (0, GraphModelClass::Node, "parity-node"),
            (1, GraphModelClass::Edge, "parity-edge"),
        ] {
            let mut payload = fixture["cases"][case_index]["payload"].clone();
            let object = payload.as_object_mut().expect("entity payload object");
            object.insert("level_from_root".to_owned(), huge_number.clone());
            object
                .get_mut("metadata")
                .and_then(Value::as_object_mut)
                .expect("entity metadata object")
                .insert("level_from_root".to_owned(), huge_number.clone());
            let payload_json = serde_json::to_string(&payload).expect("serialize graph payload");
            let envelope_json = format!(
                "{{\"schema_version\":1,\"model_class\":{},\"payload\":{payload_json}}}",
                serde_json::to_string(&model_class).expect("serialize model class")
            );
            let decoded = RawGraphModelEnvelopeV1::from_json(&envelope_json)
                .expect("raw envelope parses")
                .decode_persistent_v2()
                .expect("V2 entity validates");

            assert_eq!(decoded.graph_entity_id(), Some(expected_id));
            assert_eq!(
                decoded
                    .metadata()
                    .get("lifecycle_status")
                    .and_then(Value::as_str),
                Some("active")
            );
            assert_eq!(
                decoded.level_from_root(),
                None,
                "legacy i64 view stays bounded"
            );
            assert_eq!(
                decoded
                    .exact_level_from_root()
                    .expect("exact level remains available")
                    .as_decimal_str(),
                huge
            );

            match model_class {
                GraphModelClass::Node => assert!(decoded.as_node().is_some()),
                GraphModelClass::Edge => {
                    let edge = decoded
                        .as_edge()
                        .expect("edge variant exposes edge capability");
                    assert_eq!(edge.relation(), Some("supports"));
                }
                _ => unreachable!("only persistent nodes and edges are tested"),
            }
        }

        let mut invalid_payload = fixture["cases"][0]["payload"].clone();
        invalid_payload["metadata"]["lifecycle_status"] = Value::String("purged".to_owned());
        let invalid_wire = format!(
            "{{\"schema_version\":1,\"model_class\":\"node\",\"payload\":{}}}",
            serde_json::to_string(&invalid_payload).expect("serialize invalid payload")
        );
        assert!(
            RawGraphModelEnvelopeV1::from_json(&invalid_wire)
                .expect("invalid family payload remains valid JSON")
                .decode_persistent_v2()
                .is_err(),
            "V2 decoding must retain fail-closed lifecycle validation"
        );
    }

    #[test]
    fn every_rust_graph_model_pair_keeps_node_capabilities_on_edges() {
        fn assert_node<T: GraphEntity>() {}
        fn assert_edge<T: GraphEntity + EdgeEntity>() {}

        assert_node::<GraphNodePayload>();
        assert_edge::<GraphEdgePayload>();
        assert_node::<PersistentNodeV1>();
        assert_edge::<PersistentEdgeV1>();
        assert_node::<LlmNodeV1>();
        assert_edge::<LlmEdgeV1>();
        assert_edge::<LlmEdgeExtractionV1>();
        assert_node::<AssocFlattenedLlmNodeV1>();
        assert_edge::<AssocFlattenedLlmEdgeV1>();
        assert_node::<FlattenedLlmNodeV1>();
        assert_edge::<FlattenedLlmEdgeV1>();
        assert_node::<PureChromaNodeV1>();
        assert_edge::<PureChromaEdgeV1>();

        // These assertions are intentionally compile-time contracts: an edge
        // is a graph entity plus incidence capabilities, never a narrower
        // replacement for its corresponding node family.
    }

    #[test]
    fn specialized_edges_resolve_to_their_matching_node_family() {
        let pairs = [
            (GraphModelClass::Edge, GraphModelClass::Node),
            (GraphModelClass::AclEdge, GraphModelClass::AclNode),
            (
                GraphModelClass::ConversationEdge,
                GraphModelClass::ConversationNode,
            ),
            (GraphModelClass::WorkflowEdge, GraphModelClass::WorkflowNode),
            (
                GraphModelClass::WorkflowRuntimeEdge,
                GraphModelClass::WorkflowRunNode,
            ),
            (
                GraphModelClass::PureChromaEdge,
                GraphModelClass::PureChromaNode,
            ),
            (GraphModelClass::LlmEdge, GraphModelClass::LlmNode),
            (
                GraphModelClass::LlmEdgeExtraction,
                GraphModelClass::LlmNodeExtraction,
            ),
            (
                GraphModelClass::FlattenedLlmEdge,
                GraphModelClass::FlattenedLlmNode,
            ),
            (
                GraphModelClass::AssocFlattenedLlmEdge,
                GraphModelClass::AssocFlattenedLlmNode,
            ),
        ];
        for (edge, node) in pairs {
            assert_eq!(
                edge.corresponding_class(GraphEntityKind::Node),
                Some(node),
                "edge {:?} must retain its specialized node family",
                edge
            );
            assert!(edge.edge_extends_node_capabilities());
        }
    }

    #[test]
    fn document_node_does_not_claim_persistent_capabilities() {
        let capabilities = GraphModelClass::DocNode.capabilities();
        assert!(capabilities.graph_entity);
        assert!(!capabilities.edge_entity);
        assert!(!capabilities.metadata);
        assert!(!capabilities.lifecycle);
        assert!(!capabilities.level);
    }

    #[test]
    fn capability_report_covers_every_declared_graph_class() {
        let profiles = graph_model_capability_profiles();
        assert_eq!(profiles.len(), GraphModelClass::all().len());
        assert_eq!(
            profiles
                .iter()
                .map(|profile| profile.model_class.as_str())
                .collect::<Vec<_>>(),
            GraphModelClass::all()
                .iter()
                .map(|model_class| model_class.wire_name())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            profiles
                .iter()
                .find(|profile| profile.model_class == "doc_node")
                .unwrap()
                .capabilities,
            GraphModelClass::DocNode.capabilities()
        );
    }

    // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-contracts/src/graph_model.rs#graph_model_capability_profiles; python=tests/core/test_graph_model_adapter.py#test_graph_model_capability_contract_matches_shared_fixture; fixture=contracts/golden/graph-model-capabilities-v1.json; rust_test=rust/crates/kogwistar-contracts/src/graph_model_tests/entity_capability_contracts.rs#graph_model_capabilities_match_shared_cross_language_fixture; python_test=tests/core/test_graph_model_adapter.py#test_graph_model_capability_contract_matches_shared_fixture
    #[test]
    fn graph_model_capabilities_match_shared_cross_language_fixture() {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/graph-model-capabilities-v1.json"
        )))
        .expect("valid shared graph capability fixture");
        assert_eq!(fixture["schema_version"], GRAPH_MODEL_ENVELOPE_VERSION);
        let identity_fields: BTreeSet<_> = fixture["node_contract"]["identity_fields"]
            .as_array()
            .expect("node identity fields")
            .iter()
            .map(|field| field.as_str().expect("identity field"))
            .collect();
        assert_eq!(identity_fields, BTreeSet::from(["id"]));
        let identity_exceptions = fixture["node_contract"]["identity_exceptions"]
            .as_object()
            .expect("node identity exceptions");
        let doc_node_exception = &identity_exceptions["doc_node"];
        assert_eq!(
            doc_node_exception["absent_fields"][0],
            Value::String("id".to_owned())
        );
        assert_eq!(
            doc_node_exception["present_fields"][0],
            Value::String("doc_id".to_owned())
        );

        let mut expected = BTreeMap::new();
        for group in fixture["profiles"].as_array().expect("profile groups") {
            let entity_kind = group["entity_kind"].as_str().expect("entity kind");
            for model_class in group["classes"].as_array().expect("class names") {
                let model_class = model_class.as_str().expect("model class name");
                let mut profile = group["capabilities"].clone();
                let object = profile.as_object_mut().expect("capability object");
                object.insert(
                    "schema_version".to_owned(),
                    Value::from(GRAPH_MODEL_ENVELOPE_VERSION),
                );
                object.insert(
                    "model_class".to_owned(),
                    Value::String(model_class.to_owned()),
                );
                object.insert(
                    "entity_kind".to_owned(),
                    Value::String(entity_kind.to_owned()),
                );
                let typed: GraphModelCapabilityProfileV1 =
                    serde_json::from_value(profile).expect("typed fixture profile");
                assert!(expected.insert(model_class.to_owned(), typed).is_none());
            }
        }

        let actual: BTreeMap<_, _> = graph_model_capability_profiles()
            .into_iter()
            .map(|profile| (profile.model_class.clone(), profile))
            .collect();
        assert_eq!(actual, expected);

        let edge_pairs = fixture["edge_node_pairs"]
            .as_array()
            .expect("edge/node pairs");
        let paired_edges: BTreeSet<_> = edge_pairs
            .iter()
            .map(|pair| pair[0].as_str().expect("edge model name"))
            .collect();
        let registered_edges: BTreeSet<_> = expected
            .values()
            .filter(|profile| profile.entity_kind == GraphEntityKind::Edge)
            .map(|profile| profile.model_class.as_str())
            .collect();
        assert_eq!(
            paired_edges.len(),
            edge_pairs.len(),
            "edge pairs must be unique"
        );
        assert_eq!(paired_edges, registered_edges);

        let paired_nodes: BTreeSet<_> = edge_pairs
            .iter()
            .map(|pair| pair[1].as_str().expect("node model name"))
            .collect();
        assert_eq!(
            paired_nodes.len(),
            edge_pairs.len(),
            "node pairs must be unique"
        );
        let node_only: BTreeSet<_> = fixture["node_only_classes"]
            .as_array()
            .expect("node-only classes")
            .iter()
            .map(|model| model.as_str().expect("node-only class name"))
            .collect();
        let registered_nodes: BTreeSet<_> = expected
            .values()
            .filter(|profile| profile.entity_kind == GraphEntityKind::Node)
            .map(|profile| profile.model_class.as_str())
            .collect();
        let node_identity_exceptions: BTreeSet<_> = fixture["node_contract"]["identity_exceptions"]
            .as_object()
            .expect("node identity exceptions")
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(node_identity_exceptions, BTreeSet::from(["doc_node"]));
        assert!(node_identity_exceptions.is_subset(&registered_nodes));
        for node_name in &registered_nodes {
            let node = GraphModelClass::all()
                .iter()
                .copied()
                .find(|model| model.wire_name() == *node_name)
                .expect("fixture node exists in Rust registry");
            assert_eq!(node.entity_kind(), GraphEntityKind::Node);
        }
        assert_eq!(
            registered_nodes
                .difference(&paired_nodes)
                .copied()
                .collect::<BTreeSet<_>>(),
            node_only
        );

        for pair in edge_pairs {
            let edge_name = pair[0].as_str().expect("edge model name");
            let node_name = pair[1].as_str().expect("node model name");
            let edge = GraphModelClass::all()
                .iter()
                .copied()
                .find(|model| model.wire_name() == edge_name)
                .expect("fixture edge exists in Rust registry");
            let node = GraphModelClass::all()
                .iter()
                .copied()
                .find(|model| model.wire_name() == node_name)
                .expect("fixture node exists in Rust registry");
            assert_eq!(edge.corresponding_class(GraphEntityKind::Node), Some(node));
            assert!(edge.edge_extends_node_capabilities());
        }
    }

    #[test]
    fn model_family_classification_keeps_specialized_shapes_separate() {
        assert_eq!(GraphModelClass::Node.family(), GraphModelFamily::Persistent);
        assert_eq!(
            GraphModelClass::ConversationEdge.family(),
            GraphModelFamily::Conversation
        );
        assert_eq!(
            GraphModelClass::WorkflowNode.family(),
            GraphModelFamily::Workflow
        );
        assert_eq!(
            GraphModelClass::PureChromaEdge.family(),
            GraphModelFamily::PureChroma
        );
        assert_eq!(
            GraphModelClass::LlmEdgeExtraction.family(),
            GraphModelFamily::Llm
        );
    }

    #[test]
    fn every_edge_family_explicitly_extends_its_node_capabilities() {
        let edge_classes: Vec<_> = GraphModelClass::all()
            .iter()
            .copied()
            .filter(|model_class| model_class.entity_kind() == GraphEntityKind::Edge)
            .collect();

        assert!(
            !edge_classes.is_empty(),
            "the graph contract must define edges"
        );
        for edge_class in edge_classes {
            let node_class = edge_class
                .corresponding_class(GraphEntityKind::Node)
                .unwrap_or_else(|| {
                    panic!(
                        "edge class {} has no node counterpart",
                        edge_class.wire_name()
                    )
                });
            assert_eq!(edge_class.entity_kind(), GraphEntityKind::Edge);
            assert_eq!(node_class.entity_kind(), GraphEntityKind::Node);
            assert_eq!(edge_class.family(), node_class.family());
            assert!(
                edge_class.edge_extends_node_capabilities(),
                "edge class {} must include the capabilities of node class {}",
                edge_class.wire_name(),
                node_class.wire_name()
            );
        }

        assert!(
            GraphModelClass::DocNode
                .corresponding_class(GraphEntityKind::Edge)
                .is_none()
        );
    }
