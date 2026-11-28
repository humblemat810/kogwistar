    #[test]
    fn every_python_node_and_edge_fixture_has_a_typed_rust_view() {
        for case in fixture()["cases"].as_array().expect("fixture cases") {
            let model_class = match case["kind"].as_str().expect("kind") {
                "node" => GraphModelClass::Node,
                "edge" | "edge_with_edge_endpoints" => GraphModelClass::Edge,
                other => panic!("unexpected fixture graph model kind: {other}"),
            };
            let envelope = GraphModelEnvelope::new(model_class, case["payload"].clone());
            assert_persistent_v2_validation_matches_v1(model_class, &case["payload"]);
            let typed = envelope.inspect().expect("valid typed graph view");
            match typed {
                TypedGraphEntity::Node(node) => {
                    assert_eq!(node.common.id.as_deref(), Some("parity-node"));
                }
                TypedGraphEntity::Edge(edge) => {
                    assert_eq!(edge.source_ids.as_ref().unwrap()[0].0, "parity-node");
                    if case["kind"] == "edge_with_edge_endpoints" {
                        assert_eq!(edge.source_edge_ids.as_ref().unwrap()[0].0, "parity-edge");
                        assert_eq!(
                            edge.target_edge_ids.as_ref().unwrap()[0].0,
                            "parity-edge-target"
                        );
                    }
                }
            }
            assert_eq!(envelope.payload, case["payload"]);
        }
    }

    #[test]
    fn persistent_node_typed_round_trip_matches_python_canonical_wire_json() {
        let graph_fixture = fixture();
        let case = graph_fixture["cases"]
            .as_array()
            .expect("fixture cases")
            .iter()
            .find(|case| case["kind"] == "node")
            .expect("Python Node fixture");
        let node = GraphModelEnvelope::new(GraphModelClass::Node, case["payload"].clone())
            .decode_persistent_v1()
            .expect("valid persistent Node DTO");
        let PersistentGraphEntityV1::Node(node) = node else {
            panic!("expected Node DTO");
        };
        let serialized = serde_json::to_value(node).expect("serialize typed Rust Node DTO");
        assert_eq!(
            crate::canonical_json(&serialized, false),
            case["canonical_json"]
                .as_str()
                .expect("Python canonical Node JSON")
        );

        let mut payload_with_non_backend_fields = case["payload"].clone();
        payload_with_non_backend_fields["local_id"] = serde_json::json!("nn:pending-node");
        payload_with_non_backend_fields["mentions"][0]["spans"][0]["chunk_id"] =
            serde_json::json!("chunk:prompt-only");
        let node = GraphModelEnvelope::new(GraphModelClass::Node, payload_with_non_backend_fields)
            .decode_persistent_v1()
            .expect("legacy payload with non-backend fields remains readable");
        let PersistentGraphEntityV1::Node(node) = node else {
            panic!("expected Node DTO");
        };
        let serialized = serde_json::to_value(node).expect("serialize typed Rust Node DTO");
        assert_eq!(
            crate::canonical_json(&serialized, false),
            case["canonical_json"]
                .as_str()
                .expect("Python canonical Node JSON")
        );
    }

    #[test]
    fn persistent_edge_typed_round_trip_matches_python_canonical_wire_json() {
        let graph_fixture = fixture();
        let case = graph_fixture["cases"]
            .as_array()
            .expect("fixture cases")
            .iter()
            .find(|case| case["kind"] == "edge")
            .expect("Python Edge fixture");
        let payload_json = |payload: Value| {
            let edge = GraphModelEnvelope::new(GraphModelClass::Edge, payload)
                .decode_persistent_v1()
                .expect("valid persistent Edge DTO");
            let PersistentGraphEntityV1::Edge(edge) = edge else {
                panic!("expected Edge DTO");
            };
            let serialized = serde_json::to_value(edge).expect("serialize typed Rust Edge DTO");
            crate::canonical_json(&serialized, false)
        };
        let expected = case["canonical_json"]
            .as_str()
            .expect("Python canonical Edge JSON");
        assert_eq!(payload_json(case["payload"].clone()), expected);

        let mut payload_with_non_backend_fields = case["payload"].clone();
        payload_with_non_backend_fields["local_id"] = serde_json::json!("ne:pending-edge");
        payload_with_non_backend_fields["mentions"][0]["spans"][0]["chunk_id"] =
            serde_json::json!("chunk:prompt-only");
        assert_eq!(payload_json(payload_with_non_backend_fields), expected);
    }

    #[test]
    fn persistent_hyperedge_typed_round_trip_preserves_edge_incidences() {
        let graph_fixture = fixture();
        let case = graph_fixture["cases"]
            .as_array()
            .expect("fixture cases")
            .iter()
            .find(|case| case["kind"] == "edge_with_edge_endpoints")
            .expect("Python hyperedge fixture");
        let envelope = GraphModelEnvelope::new(GraphModelClass::Edge, case["payload"].clone());
        let PersistentGraphEntityV1::Edge(edge) = envelope
            .decode_persistent_v1()
            .expect("valid persistent hyperedge DTO")
        else {
            panic!("expected Edge DTO for hyperedge");
        };
        let serialized = serde_json::to_value(edge).expect("serialize typed Rust hyperedge");
        assert_eq!(
            serialized["source_edge_ids"],
            serde_json::json!(["parity-edge"])
        );
        assert_eq!(
            serialized["target_edge_ids"],
            serde_json::json!(["parity-edge-target"])
        );
        assert_eq!(
            crate::canonical_json(&serialized, false),
            case["canonical_json"]
                .as_str()
                .expect("Python canonical hyperedge JSON")
        );
    }

    #[test]
    fn workflow_runtime_edge_matches_python_metadata_defaults_and_requirements() {
        let contract: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/workflow-runtime-edge-metadata-v1.json"
        )))
        .expect("valid shared workflow runtime edge metadata contract");
        assert_eq!(contract["schema_version"], 1);

        let baseline = fixture()["cases"][1]["payload"].clone();
        for case in contract["cases"].as_array().expect("contract cases") {
            let mut payload = baseline.clone();
            payload["metadata"] = case["metadata"].clone();
            let result = GraphModelEnvelope::new(GraphModelClass::WorkflowRuntimeEdge, payload)
                .validate_family_v1();
            assert_eq!(
                result.is_ok(),
                case["valid"].as_bool().expect("validity flag"),
                "workflow runtime edge metadata case {}",
                case["id"]
            );
        }
    }

    #[test]
    fn workflow_runtime_node_metadata_matches_python_model_contract() {
        let contract: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/workflow-runtime-node-metadata-v1.json"
        )))
        .expect("valid shared workflow runtime node metadata contract");
        assert_eq!(contract["schema_version"], 1);

        let baseline = fixture()["cases"][0]["payload"].clone();
        for case in contract["cases"].as_array().expect("contract cases") {
            let model_class: GraphModelClass = serde_json::from_value(case["model_class"].clone())
                .expect("known runtime node model class");
            let mut payload = baseline.clone();
            payload["metadata"] = case["metadata"].clone();
            let result = GraphModelEnvelope::new(model_class, payload).validate_family_v1();
            assert_eq!(
                result.is_ok(),
                case["valid"].as_bool().expect("validity flag"),
                "workflow runtime node metadata case {}",
                case["id"]
            );
        }
    }

    #[test]
    fn conversation_metadata_matches_python_model_contract() {
        let contract: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/conversation-graph-metadata-v1.json"
        )))
        .expect("valid shared conversation graph metadata contract");
        assert_eq!(contract["schema_version"], 1);

        let graph_fixture = fixture();
        for case in contract["cases"].as_array().expect("contract cases") {
            let model_class: GraphModelClass = serde_json::from_value(case["model_class"].clone())
                .expect("known conversation model class");
            let baseline_index = usize::from(model_class == GraphModelClass::ConversationEdge);
            let mut payload = graph_fixture["cases"][baseline_index]["payload"].clone();
            payload["metadata"] = case["metadata"].clone();
            if let Some(top_level) = case["top_level"].as_object() {
                for (field, value) in top_level {
                    payload[field] = value.clone();
                }
            }
            let result = GraphModelEnvelope::new(model_class, payload).validate_family_v1();
            assert_eq!(
                result.is_ok(),
                case["valid"].as_bool().expect("validity flag"),
                "conversation metadata case {}",
                case["id"]
            );
        }
    }

    #[test]
    fn workflow_design_metadata_matches_python_model_contract() {
        let contract: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/workflow-design-metadata-v1.json"
        )))
        .expect("valid shared workflow design metadata contract");
        assert_eq!(contract["schema_version"], 1);

        let graph_fixture = fixture();
        for case in contract["cases"].as_array().expect("contract cases") {
            let model_class: GraphModelClass = serde_json::from_value(case["model_class"].clone())
                .expect("known workflow design model class");
            let baseline_index = usize::from(model_class == GraphModelClass::WorkflowEdge);
            let mut payload = graph_fixture["cases"][baseline_index]["payload"].clone();
            payload["metadata"] = if case["valid"].as_bool() == Some(true) {
                case["serialized_metadata"].clone()
            } else {
                case["input_metadata"].clone()
            };
            let result = GraphModelEnvelope::new(model_class, payload).validate_family_v1();
            assert_eq!(
                result.is_ok(),
                case["valid"].as_bool().expect("validity flag"),
                "workflow design metadata case {}: {result:?}",
                case["id"]
            );
        }
    }

    #[test]
    fn persistent_graph_invalid_cases_match_shared_python_rejection_fixture() {
        let baseline: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/python-graph-model-payloads-v1.json"
        )))
        .expect("valid Python graph payload fixture");
        let invalid: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/graph-model-invalid-v1.json"
        )))
        .expect("valid graph invalid-case fixture");
        assert_eq!(invalid["schema_version"], 1);

        for case in invalid["cases"].as_array().expect("invalid cases") {
            let case_id = case.as_str().expect("case id");
            let (kind, mutation) = case_id.split_once('.').expect("kind.mutation");
            let baseline_kind = match kind {
                "pure_chroma_node" => "node",
                "pure_chroma_edge" => "edge",
                _ => kind,
            };
            let baseline_case = baseline["cases"]
                .as_array()
                .expect("baseline cases")
                .iter()
                .find(|candidate| candidate["kind"] == baseline_kind)
                .expect("matching baseline model");
            let mut payload = baseline_case["payload"].clone();
            if kind.starts_with("pure_chroma_") {
                payload
                    .as_object_mut()
                    .expect("PureChroma payload object")
                    .remove("mentions");
            }
            let model_class = match kind {
                "node" => GraphModelClass::Node,
                "edge" | "edge_with_edge_endpoints" => GraphModelClass::Edge,
                "pure_chroma_node" => GraphModelClass::PureChromaNode,
                "pure_chroma_edge" => GraphModelClass::PureChromaEdge,
                _ => panic!("unexpected fixture model class: {kind}"),
            };
            match mutation {
                "invalid_lifecycle_status" => {
                    payload["metadata"]["lifecycle_status"] = Value::String("purged".into());
                }
                "empty_id" => {
                    payload["id"] = Value::String(String::new());
                }
                "invalid_redirect_id" => {
                    payload["metadata"]["redirect_to_id"] = Value::Number(7.into());
                }
                "empty_node_endpoint_id" if kind == "edge" => {
                    payload["source_ids"] = serde_json::json!([""]);
                }
                "empty_edge_endpoint_id" if kind == "edge" => {
                    payload["source_edge_ids"] = serde_json::json!([""]);
                }
                "empty_source_edge_endpoint_id" if kind == "edge_with_edge_endpoints" => {
                    payload["source_edge_ids"] = serde_json::json!([""]);
                }
                "empty_target_edge_endpoint_id" if kind == "edge_with_edge_endpoints" => {
                    payload["target_edge_ids"] = serde_json::json!([""]);
                }
                "missing_relation" if kind == "edge" => {
                    payload
                        .as_object_mut()
                        .expect("edge payload object")
                        .remove("relation");
                }
                "negative_span_start_char" if kind == "node" => {
                    payload["mentions"][0]["spans"][0]["start_char"] = serde_json::json!(-1);
                }
                "span_end_not_after_start" if kind == "node" => {
                    payload["mentions"][0]["spans"][0]["start_char"] = serde_json::json!(8);
                    payload["mentions"][0]["spans"][0]["end_char"] = serde_json::json!(8);
                }
                "span_offset_exceeds_u64" if kind == "node" => {
                    payload["mentions"][0]["spans"][0]["end_char"] =
                        serde_json::json!(18446744073709551616_u128);
                }
                "properties_not_mapping" => {
                    payload["properties"] = serde_json::json!(["not", "a", "mapping"]);
                }
                "properties_nested_mapping" if kind == "node" => {
                    payload["properties"] = serde_json::json!({
                        "invalid": {"nested": {"value": 1}}
                    });
                }
                "properties_nested_list" if kind == "edge" => {
                    payload["properties"] = serde_json::json!({"invalid": [[1]]});
                }
                "properties_nested_mapping" if kind == "pure_chroma_node" => {
                    payload["properties"] = serde_json::json!({
                        "invalid": {"nested": {"value": 1}}
                    });
                }
                "properties_nested_list" if kind == "pure_chroma_edge" => {
                    payload["properties"] = serde_json::json!({"invalid": [[1]]});
                }
                _ => panic!("unexpected graph-model invalid fixture: {case_id}"),
            }
            assert!(
                GraphModelEnvelope::new(model_class, payload.clone())
                    .validate_family_v1()
                    .is_err(),
                "Rust accepted shared invalid graph-model case {case_id}"
            );
            if matches!(model_class, GraphModelClass::Node | GraphModelClass::Edge) {
                assert_persistent_v2_validation_matches_v1(model_class, &payload);
            }
        }
    }

    #[test]
    fn persistent_graph_legacy_defaults_match_shared_compatibility_fixture() {
        let baseline: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/python-graph-model-payloads-v1.json"
        )))
        .expect("valid Python graph payload fixture");
        let compatibility: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/graph-model-legacy-compat-v1.json"
        )))
        .expect("valid graph compatibility fixture");
        assert_eq!(compatibility["schema_version"], 1);

        for case in compatibility["cases"]
            .as_array()
            .expect("compatibility cases")
        {
            let case_id = case.as_str().expect("case id");
            let (kind, compatibility_case) = case_id.split_once('.').expect("kind.case");
            let baseline_case = baseline["cases"]
                .as_array()
                .expect("baseline cases")
                .iter()
                .find(|candidate| candidate["kind"] == kind)
                .expect("matching baseline model");
            let mut payload = baseline_case["payload"].clone();
            let model_class = match kind {
                "node" => GraphModelClass::Node,
                "edge" => GraphModelClass::Edge,
                _ => panic!("unexpected fixture model class: {kind}"),
            };
            let expected_status = match compatibility_case {
                "omitted_lifecycle_defaults" => {
                    let metadata = payload
                        .get_mut("metadata")
                        .and_then(Value::as_object_mut)
                        .expect("fixture metadata");
                    metadata.remove("lifecycle_status");
                    metadata.remove("redirect_to_id");
                    "active"
                }
                "deleted_alias_normalizes" => {
                    payload["metadata"]["lifecycle_status"] = Value::String("deleted".into());
                    "tombstoned"
                }
                "level_metadata_fallback" => {
                    payload
                        .as_object_mut()
                        .expect("graph payload object")
                        .remove("level_from_root");
                    payload["metadata"]["level_from_root"] = Value::Number(2.into());
                    "active"
                }
                "level_top_level_precedence" => {
                    payload["level_from_root"] = Value::Number(3.into());
                    payload["metadata"]["level_from_root"] = Value::Number(2.into());
                    "active"
                }
                _ => panic!("unexpected graph-model compatibility case: {case_id}"),
            };
            let decoded = GraphModelEnvelope::new(model_class, payload)
                .decode_persistent_v1()
                .expect("legacy graph payload remains accepted");
            let core = match decoded {
                PersistentGraphEntityV1::Node(node) => node.core,
                PersistentGraphEntityV1::Edge(edge) => edge.core,
            };
            assert_eq!(core.metadata["lifecycle_status"], expected_status);
            assert_eq!(core.metadata["redirect_to_id"], Value::Null);
            match compatibility_case {
                "level_metadata_fallback" => {
                    assert_eq!(core.level_from_root, Some(2));
                    assert_eq!(core.metadata["level_from_root"], 2);
                }
                "level_top_level_precedence" => {
                    assert_eq!(core.level_from_root, Some(3));
                    assert_eq!(core.metadata["level_from_root"], 3);
                }
                _ => {}
            }
        }
    }

    #[test]
    fn model_class_tag_is_out_of_band_and_preserves_legacy_payload_shape() {
        let payload = fixture()["cases"][0]["payload"].clone();
        let envelope = GraphModelEnvelope::new(GraphModelClass::AclNode, payload.clone());
        assert_eq!(envelope.payload, payload);
        assert_eq!(envelope.model_class.entity_kind(), GraphEntityKind::Node);
        assert!(
            !serde_json::to_value(envelope)
                .unwrap()
                .get("payload")
                .unwrap()
                .as_object()
                .unwrap()
                .contains_key("model_class")
        );
    }

    #[test]
    fn acl_node_family_round_trip_preserves_python_tag_and_extension_metadata() {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/acl-node-model-v1.json"
        )))
        .expect("valid Python ACLNode fixture");
        let wire = serde_json::json!({
            "schema_version": fixture["schema_version"],
            "model_class": fixture["model_class"],
            "payload": fixture["payload"],
        })
        .to_string();

        let envelope = RawGraphModelEnvelopeV1::from_json(&wire).expect("valid ACLNode envelope");
        assert_eq!(envelope.model_class, GraphModelClass::AclNode);
        envelope
            .validate_family_v1()
            .expect("valid typed ACLNode metadata");

        let round_trip: Value = serde_json::from_str(
            &round_trip_raw_graph_model_envelope_from_str(&wire)
                .expect("lossless Rust envelope round trip"),
        )
        .expect("valid Rust round-trip JSON");
        assert_eq!(round_trip["model_class"], fixture["model_class"]);
        assert_eq!(round_trip["payload"], fixture["payload"]);
        assert_eq!(
            round_trip["payload"]["metadata"]["share_extension"],
            "retained"
        );
        use sha2::Digest;
        let canonical_payload = crate::canonical_json(&round_trip["payload"], false);
        assert_eq!(
            hex::encode(sha2::Sha256::digest(canonical_payload.as_bytes())),
            fixture["canonical_json_sha256"]
        );
    }

    #[test]
    fn node_and_edge_endpoint_ids_are_distinct_rust_types() {
        let payload = fixture()["cases"][2]["payload"].clone();
        let envelope = GraphModelEnvelope::new(GraphModelClass::Edge, payload);
        let TypedGraphEntity::Edge(edge) = envelope.inspect().unwrap() else {
            panic!("edge model class must decode to an edge");
        };
        assert_eq!(
            edge.source_ids.unwrap()[0],
            NodeId("parity-node".to_owned())
        );
        assert_eq!(
            edge.source_edge_ids.unwrap()[0],
            EdgeId("parity-edge".to_owned())
        );
    }

    #[test]
    fn hyperedges_reuse_edge_identity_and_preserve_all_incidences() {
        let mut payload = fixture()["cases"][1]["payload"].clone();
        payload["source_ids"] = serde_json::json!(["node-a", "node-b"]);
        payload["target_ids"] = serde_json::json!(["node-c", "node-d"]);
        payload["source_edge_ids"] = serde_json::json!(["edge-parent"]);
        let envelope = GraphModelEnvelope::new(GraphModelClass::Edge, payload);
        let TypedGraphEntity::Edge(edge) = envelope.inspect().unwrap() else {
            panic!("hyperedge must retain the ordinary edge identity");
        };
        assert_eq!(edge.source_ids.as_ref().unwrap().len(), 2);
        assert_eq!(edge.target_ids.as_ref().unwrap().len(), 2);
        assert_eq!(edge.source_edge_ids.as_ref().unwrap().len(), 1);
        assert!(edge.is_hyperedge());
        assert_eq!(edge.graph_entity_id(), Some("parity-edge"));
    }

    #[test]
    fn nodes_and_edges_share_the_graph_entity_interface() {
        fn inspect_entity(entity: &impl GraphEntity) -> (&'static str, Option<&str>) {
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
        .inspect()
        .unwrap();
        let edge = GraphModelEnvelope::new(
            GraphModelClass::Edge,
            fixture["cases"][1]["payload"].clone(),
        )
        .inspect()
        .unwrap();

        assert_eq!(inspect_entity(&node), ("node", Some("parity-node")));
        assert_eq!(inspect_entity(&edge), ("edge", Some("parity-edge")));
    }

    #[test]
    fn explicit_model_class_rejects_node_edge_shape_confusion() {
        let edge_payload = fixture()["cases"][1]["payload"].clone();
        let wrongly_tagged = GraphModelEnvelope::new(GraphModelClass::Node, edge_payload);
        assert!(matches!(
            wrongly_tagged.inspect(),
            Err(GraphModelError::InvalidEntityFields {
                reason: "node model payload contains edge-only fields",
                ..
            })
        ));
        assert!(matches!(
            wrongly_tagged.decode_persistent_v1(),
            Err(GraphModelError::InvalidEntityFields {
                reason: "node model payload contains edge-only fields",
                ..
            })
        ));

        let mut incomplete = fixture()["cases"][1]["payload"].clone();
        incomplete.as_object_mut().unwrap().remove("relation");
        let wrongly_shaped = GraphModelEnvelope::new(GraphModelClass::Edge, incomplete);
        assert!(matches!(
            wrongly_shaped.inspect(),
            Err(GraphModelError::InvalidEntityFields {
                reason: "edge requires source_ids, target_ids, and relation",
                ..
            })
        ));
    }

    #[test]
    fn strict_persistent_models_preserve_required_nullable_edge_fields() {
        let fixture = fixture();
        let payload = fixture["cases"][1]["payload"].clone();
        let decoded = GraphModelEnvelope::new(GraphModelClass::Edge, payload.clone())
            .decode_persistent_v1()
            .expect("strict edge payload");
        let PersistentGraphEntityV1::Edge(edge) = decoded else {
            panic!("expected strict edge");
        };
        assert!(matches!(edge.source_edge_ids, RequiredNullable::Value(_)));
        assert!(matches!(edge.target_edge_ids, RequiredNullable::Value(_)));
        assert_eq!(edge.lifecycle_status(), LifecycleStatus::Active);
        assert_eq!(edge.level_from_root(), None);

        let mut explicit_null = fixture["cases"][1]["payload"].clone();
        explicit_null["source_edge_ids"] = Value::Null;
        explicit_null["target_edge_ids"] = Value::Null;
        let decoded = GraphModelEnvelope::new(GraphModelClass::Edge, explicit_null)
            .decode_persistent_v1()
            .expect("explicit null remains valid");
        let PersistentGraphEntityV1::Edge(edge) = decoded else {
            panic!("expected strict edge");
        };
        assert!(edge.source_edge_ids.is_null());
        assert!(edge.target_edge_ids.is_null());

        let mut missing = fixture["cases"][1]["payload"].clone();
        missing
            .as_object_mut()
            .expect("object payload")
            .remove("source_edge_ids");
        assert!(matches!(
            GraphModelEnvelope::new(GraphModelClass::Edge, missing).decode_persistent_v1(),
            Err(GraphModelError::InvalidPayload(_))
        ));
    }

    #[test]
    fn strict_persistent_models_reject_invalid_grounding_and_lifecycle() {
        let fixture = fixture();
        let mut payload = fixture["cases"][0]["payload"].clone();
        payload["mentions"] = Value::Array(Vec::new());
        assert!(matches!(
            GraphModelEnvelope::new(GraphModelClass::Node, payload).decode_persistent_v1(),
            Err(GraphModelError::InvalidEntityFields {
                reason: "persistent entity requires at least one grounding",
                ..
            })
        ));

        let mut invalid_lifecycle = fixture["cases"][0]["payload"].clone();
        invalid_lifecycle["metadata"]["lifecycle_status"] = Value::String("purged".into());
        assert!(matches!(
            GraphModelEnvelope::new(GraphModelClass::Node, invalid_lifecycle)
                .decode_persistent_v1(),
            Err(GraphModelError::InvalidEntityFields {
                reason: "lifecycle_status must be active or tombstoned",
                ..
            })
        ));
    }

    #[test]
    fn persistent_node_and_edge_omitted_lifecycle_metadata_keep_python_defaults() {
        let fixture = fixture();
        for (model_class, case_name) in [
            (GraphModelClass::Node, "node"),
            (GraphModelClass::Edge, "edge"),
        ] {
            let mut payload = fixture["cases"]
                .as_array()
                .expect("fixture cases")
                .iter()
                .find(|case| case["kind"] == case_name)
                .expect("fixture model case")["payload"]
                .clone();
            let metadata = payload
                .get_mut("metadata")
                .and_then(Value::as_object_mut)
                .expect("fixture metadata");
            metadata.remove("lifecycle_status");
            metadata.remove("redirect_to_id");

            let decoded = GraphModelEnvelope::new(model_class, payload)
                .decode_persistent_v1()
                .expect("omitted lifecycle metadata remains backward compatible");
            match decoded {
                PersistentGraphEntityV1::Node(node) => {
                    assert_eq!(node.lifecycle_status(), LifecycleStatus::Active);
                    assert_eq!(node.core.metadata["redirect_to_id"], Value::Null);
                }
                PersistentGraphEntityV1::Edge(edge) => {
                    assert_eq!(edge.lifecycle_status(), LifecycleStatus::Active);
                    assert_eq!(edge.core.metadata["redirect_to_id"], Value::Null);
                }
            }
        }
    }

    #[test]
    fn persistent_node_and_edge_normalize_python_mixin_fields() {
        let fixture = fixture();
        for (model_class, case_name) in [
            (GraphModelClass::Node, "node"),
            (GraphModelClass::Edge, "edge"),
        ] {
            let mut payload = fixture["cases"]
                .as_array()
                .expect("fixture cases")
                .iter()
                .find(|case| case["kind"] == case_name)
                .expect("fixture model case")["payload"]
                .clone();
            payload["metadata"]["lifecycle_status"] = Value::String("deleted".to_owned());
            payload["metadata"]["level_from_root"] = Value::Number(2.into());
            payload
                .as_object_mut()
                .expect("object payload")
                .remove("level_from_root");
            let envelope = GraphModelEnvelope::new(model_class, payload.clone());

            let decoded = envelope
                .decode_persistent_v1()
                .expect("legacy model fields normalize like Python mixins");
            let core = match decoded {
                PersistentGraphEntityV1::Node(node) => node.core,
                PersistentGraphEntityV1::Edge(edge) => edge.core,
            };
            assert_eq!(core.metadata["lifecycle_status"], "tombstoned");
            assert_eq!(core.metadata["redirect_to_id"], Value::Null);
            assert_eq!(core.level_from_root, Some(2));
            assert_eq!(core.metadata["level_from_root"], 2);
            assert_eq!(envelope.payload["metadata"]["lifecycle_status"], "deleted");

            let mut conflicting = fixture["cases"]
                .as_array()
                .expect("fixture cases")
                .iter()
                .find(|case| case["kind"] == case_name)
                .expect("fixture model case")["payload"]
                .clone();
            conflicting["level_from_root"] = Value::Number(3.into());
            conflicting["metadata"]["level_from_root"] = Value::Number(2.into());
            let decoded = GraphModelEnvelope::new(model_class, conflicting)
                .decode_persistent_v1()
                .expect("explicit level field overrides metadata like Python mixin");
            let core = match decoded {
                PersistentGraphEntityV1::Node(node) => node.core,
                PersistentGraphEntityV1::Edge(edge) => edge.core,
            };
            assert_eq!(core.level_from_root, Some(3));
            assert_eq!(core.metadata["level_from_root"], 3);

            for (raw_level, expected) in [
                (Value::String("2".to_owned()), 2),
                (Value::Bool(true), 1),
                (serde_json::json!(2.0), 2),
            ] {
                let mut coerced = fixture["cases"]
                    .as_array()
                    .expect("fixture cases")
                    .iter()
                    .find(|case| case["kind"] == case_name)
                    .expect("fixture model case")["payload"]
                    .clone();
                coerced["metadata"]["level_from_root"] = raw_level;
                coerced
                    .as_object_mut()
                    .expect("object payload")
                    .remove("level_from_root");
                let decoded = GraphModelEnvelope::new(model_class, coerced)
                    .decode_persistent_v1()
                    .expect("coerce metadata level as Python's integer field does");
                let core = match decoded {
                    PersistentGraphEntityV1::Node(node) => node.core,
                    PersistentGraphEntityV1::Edge(edge) => edge.core,
                };
                assert_eq!(core.level_from_root, Some(expected));
                assert_eq!(core.metadata["level_from_root"], expected);
            }

            let mut fractional = fixture["cases"]
                .as_array()
                .expect("fixture cases")
                .iter()
                .find(|case| case["kind"] == case_name)
                .expect("fixture model case")["payload"]
                .clone();
            fractional["metadata"]["level_from_root"] = serde_json::json!(2.5);
            fractional
                .as_object_mut()
                .expect("object payload")
                .remove("level_from_root");
            assert!(
                GraphModelEnvelope::new(model_class, fractional)
                    .decode_persistent_v1()
                    .is_err()
            );

            let mut negative = fixture["cases"]
                .as_array()
                .expect("fixture cases")
                .iter()
                .find(|case| case["kind"] == case_name)
                .expect("fixture model case")["payload"]
                .clone();
            negative["level_from_root"] = Value::Number((-1).into());
            let decoded = GraphModelEnvelope::new(model_class, negative)
                .decode_persistent_v1()
                .expect("preserve Python's existing integer level acceptance");
            let core = match decoded {
                PersistentGraphEntityV1::Node(node) => node.core,
                PersistentGraphEntityV1::Edge(edge) => edge.core,
            };
            assert_eq!(core.level_from_root, Some(-1));
            assert_eq!(core.metadata["level_from_root"], -1);

            let mut explicit_null = fixture["cases"]
                .as_array()
                .expect("fixture cases")
                .iter()
                .find(|case| case["kind"] == case_name)
                .expect("fixture model case")["payload"]
                .clone();
            explicit_null["level_from_root"] = Value::Null;
            explicit_null["metadata"]["level_from_root"] = Value::Number(2.into());
            let decoded = GraphModelEnvelope::new(model_class, explicit_null)
                .decode_persistent_v1()
                .expect("explicit null remains distinct from omitted field");
            let core = match decoded {
                PersistentGraphEntityV1::Node(node) => node.core,
                PersistentGraphEntityV1::Edge(edge) => edge.core,
            };
            assert_eq!(core.level_from_root, None);
            assert_eq!(core.metadata["level_from_root"], 2);
        }
    }

    #[test]
    fn tombstoned_edge_preserves_redirect_target_from_shared_fixture() {
        let contract_fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/graph-model-lifecycle-redirect-v1.json"
        )))
        .expect("valid edge redirect lifecycle fixture");
        assert_eq!(contract_fixture["schema_version"], 1);
        assert_eq!(contract_fixture["model_class"], "edge");

        let baseline = fixture();
        let mut payload = baseline["cases"]
            .as_array()
            .expect("graph model cases")
            .iter()
            .find(|case| case["kind"] == contract_fixture["source_fixture_kind"])
            .expect("edge model case")["payload"]
            .clone();
        payload["metadata"]["lifecycle_status"] =
            contract_fixture["input_lifecycle_status"].clone();
        payload["metadata"]["redirect_to_id"] = contract_fixture["redirect_to_id"].clone();

        let decoded = GraphModelEnvelope::new(GraphModelClass::Edge, payload)
            .decode_persistent_v1()
            .expect("legacy deleted edge with redirect remains valid");
        let PersistentGraphEntityV1::Edge(edge) = decoded else {
            panic!("edge fixture decoded as a non-edge model");
        };
        assert_eq!(
            edge.core.metadata["lifecycle_status"],
            contract_fixture["expected_lifecycle_status"]
        );
        assert_eq!(
            edge.core.metadata["redirect_to_id"],
            contract_fixture["redirect_to_id"]
        );
        assert_eq!(edge.source_ids[0].0, "parity-node");
        assert_eq!(edge.target_ids[0].0, "external:target");
        let serialized = serde_json::to_value(&edge).expect("serialize persistent Edge DTO");
        assert_eq!(serialized["source_ids"], serde_json::json!(["parity-node"]));
        assert_eq!(
            serialized["target_ids"],
            serde_json::json!(["external:target"])
        );
    }
