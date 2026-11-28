    #[test]
    fn family_dtos_share_graph_entity_and_edge_entity_capabilities() {
        fn identify(entity: &impl GraphEntity) -> (GraphEntityKind, Option<&str>) {
            (entity.graph_entity_kind(), entity.graph_entity_id())
        }

        let fixture = fixture();
        let mut llm_node_payload = fixture["cases"][0]["payload"].clone();
        llm_node_payload
            .as_object_mut()
            .unwrap()
            .remove("embedding");
        let llm_node: LlmNodeV1 = serde_json::from_value(llm_node_payload).unwrap();
        assert_eq!(
            identify(&llm_node),
            (GraphEntityKind::Node, Some("parity-node"))
        );

        let mut llm_edge_payload = fixture["cases"][1]["payload"].clone();
        llm_edge_payload
            .as_object_mut()
            .unwrap()
            .remove("embedding");
        let llm_edge: LlmEdgeV1 = serde_json::from_value(llm_edge_payload).unwrap();
        assert_eq!(
            identify(&llm_edge),
            (GraphEntityKind::Edge, Some("parity-edge"))
        );
        assert_eq!(llm_edge.source_node_ids().unwrap().len(), 1);

        let mut pure_payload = fixture["cases"][0]["payload"].clone();
        pure_payload.as_object_mut().unwrap().remove("mentions");
        let pure: PureChromaNodeV1 = serde_json::from_value(pure_payload).unwrap();
        assert_eq!(
            identify(&pure),
            (GraphEntityKind::Node, Some("parity-node"))
        );

        let doc = DocumentNodeV1 {
            id: Some("doc-node:1".to_owned()),
            label: "Page 1".to_owned(),
            node_type: DocumentNodeTypeV1::Page,
            summary: "A page".to_owned(),
            doc_id: "document:1".to_owned(),
            mentions: vec![],
        };
        assert_eq!(identify(&doc), (GraphEntityKind::Node, Some("doc-node:1")));
    }

    #[test]
    fn acl_edge_validation_preserves_the_persistent_edge_shape() {
        let fixture = fixture();
        let mut acl_edge = fixture["cases"][1]["payload"].clone();
        acl_edge["metadata"] = serde_json::json!({
            "entity_type": "acl_targets_truth",
            "acl_truth_graph": "truth:main",
            "acl_target_entity_id": "parity-node",
            "acl_target_grain": "node",
            "acl_version": 1,
            "acl_mode": "shared",
            "level_from_root": 0
        });
        assert!(
            GraphModelEnvelope::new(GraphModelClass::AclEdge, acl_edge.clone())
                .validate_family_v1()
                .is_ok()
        );

        acl_edge
            .as_object_mut()
            .expect("edge payload object")
            .remove("source_ids");
        assert!(matches!(
            GraphModelEnvelope::new(GraphModelClass::AclEdge, acl_edge).validate_family_v1(),
            Err(GraphModelError::InvalidPayload(_))
        ));
    }

    #[test]
    fn acl_edge_typed_round_trip_matches_python_backend_fixture() {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/acl-edge-model-v1.json"
        )))
        .expect("valid shared Python ACLEdge fixture");
        let payload = fixture["payload"].clone();
        let envelope = GraphModelEnvelope::new(GraphModelClass::AclEdge, payload.clone());

        envelope
            .validate_family_v1()
            .expect("valid Python ACL-specific edge metadata");
        let edge = envelope
            .decode_persistent_edge_shape()
            .expect("valid typed persistent edge shape");
        let serialized = serde_json::to_value(edge).expect("serialize typed ACL edge");

        assert_eq!(serialized["metadata"]["entity_type"], "acl_targets_truth");
        assert_eq!(
            serialized["metadata"]["vendor_extension"],
            serde_json::json!({"source": "fixture"})
        );
        assert_eq!(
            crate::canonical_json(&serialized, false),
            fixture["canonical_json"]
                .as_str()
                .expect("Python canonical ACL edge JSON")
        );
        use sha2::Digest;
        assert_eq!(
            hex::encode(sha2::Sha256::digest(
                crate::canonical_json(&serialized, false).as_bytes()
            )),
            fixture["canonical_json_sha256"]
        );
        assert_eq!(serialized["source_ids"], serde_json::json!(["parity-node"]));
        assert_eq!(
            serialized["target_ids"],
            serde_json::json!(["external:target"])
        );
    }

    #[test]
    fn conversation_edge_typed_round_trip_matches_python_backend_fixture() {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/conversation-edge-model-v1.json"
        )))
        .expect("valid shared Python ConversationEdge fixture");
        let payload = fixture["payload"].clone();
        let envelope = GraphModelEnvelope::new(GraphModelClass::ConversationEdge, payload);

        envelope
            .validate_family_v1()
            .expect("valid conversation-specific edge metadata");
        let edge = envelope
            .decode_persistent_edge_shape()
            .expect("valid typed persistent edge shape");
        let serialized = serde_json::to_value(edge).expect("serialize typed conversation edge");

        assert_eq!(serialized["metadata"]["causal_type"], "chain");
        assert_eq!(serialized["metadata"]["char_distance_from_last_summary"], 4);
        assert_eq!(serialized["metadata"]["turn_distance_from_last_summary"], 1);
        assert_eq!(
            serialized["metadata"]["vendor_extension"],
            serde_json::json!({"source": "fixture"})
        );
        assert_eq!(
            crate::canonical_json(&serialized, false),
            fixture["canonical_json"]
                .as_str()
                .expect("Python canonical conversation edge JSON")
        );
        assert_eq!(serialized["source_ids"], serde_json::json!(["parity-node"]));
        assert_eq!(
            serialized["target_ids"],
            serde_json::json!(["external:target"])
        );
    }

    #[test]
    fn workflow_edge_typed_round_trip_matches_python_backend_fixture() {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/workflow-edge-model-v1.json"
        )))
        .expect("valid shared Python WorkflowEdge fixture");
        let payload = fixture["payload"].clone();
        let envelope = GraphModelEnvelope::new(GraphModelClass::WorkflowEdge, payload);

        envelope
            .validate_family_v1()
            .expect("valid workflow-specific edge metadata");
        let edge = envelope
            .decode_persistent_edge_shape()
            .expect("valid typed persistent edge shape");
        let serialized = serde_json::to_value(edge).expect("serialize typed workflow edge");

        assert_eq!(serialized["metadata"]["workflow_id"], "wf:python-fixture");
        assert_eq!(serialized["metadata"]["wf_predicate"], "ok");
        assert_eq!(serialized["metadata"]["wf_priority"], 1);
        assert_eq!(serialized["metadata"]["wf_multiplicity"], "many");
        assert_eq!(
            serialized["metadata"]["vendor_extension"],
            serde_json::json!({"source": "fixture"})
        );
        assert_eq!(
            crate::canonical_json(&serialized, false),
            fixture["canonical_json"]
                .as_str()
                .expect("Python canonical workflow edge JSON")
        );
        assert_eq!(serialized["source_ids"], serde_json::json!(["parity-node"]));
        assert_eq!(
            serialized["target_ids"],
            serde_json::json!(["external:target"])
        );
    }

    #[test]
    fn workflow_runtime_edge_typed_round_trip_matches_python_backend_fixture() {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/workflow-runtime-edge-model-v1.json"
        )))
        .expect("valid shared Python WorkflowRuntimeEdge fixture");
        let payload = fixture["payload"].clone();
        let envelope = GraphModelEnvelope::new(GraphModelClass::WorkflowRuntimeEdge, payload);

        envelope
            .validate_family_v1()
            .expect("valid workflow runtime edge metadata");
        let edge = envelope
            .decode_persistent_edge_shape()
            .expect("valid typed persistent edge shape");
        let serialized = serde_json::to_value(edge).expect("serialize typed runtime edge");

        assert_eq!(serialized["metadata"]["relation"], "next");
        assert_eq!(
            serialized["metadata"]["conversation_id"],
            "conversation:python-fixture"
        );
        assert_eq!(serialized["metadata"]["run_id"], "run:python-fixture");
        assert_eq!(serialized["metadata"]["entity_type"], "conversation_edge");
        assert_eq!(
            serialized["metadata"]["vendor_extension"],
            serde_json::json!({"source": "fixture"})
        );
        assert_eq!(
            crate::canonical_json(&serialized, false),
            fixture["canonical_json"]
                .as_str()
                .expect("Python canonical workflow runtime edge JSON")
        );
        assert_eq!(serialized["source_ids"], serde_json::json!(["parity-node"]));
        assert_eq!(
            serialized["target_ids"],
            serde_json::json!(["external:target"])
        );
    }

    #[test]
    fn declared_slice_projects_canonical_payload_and_requires_declaration() {
        let manifest = GraphSliceManifestV1 {
            schema_version: GRAPH_SLICE_MANIFEST_VERSION,
            model_name: Some("Node".to_owned()),
            retain_unlisted_modes: BTreeSet::new(),
            fields: BTreeMap::from([(
                "summary".to_owned(),
                GraphSliceFieldV1 {
                    modes: BTreeSet::from([GraphSliceMode::LlmIn]),
                    child: None,
                    child_location: Default::default(),
                },
            )]),
        };
        let envelope = GraphModelEnvelope::new(
            GraphModelClass::Node,
            serde_json::json!({"summary": "visible", "secret": "backend-only"}),
        )
        .with_slice(GraphSliceDeclarationV1 {
            schema_version: GRAPH_SLICE_MANIFEST_VERSION,
            mode: GraphSliceMode::LlmIn,
            manifest,
        });

        let projected = envelope.project_declared_slice().unwrap();
        assert_eq!(projected.payload, serde_json::json!({"summary": "visible"}));
        let projected_wire =
            project_graph_model_slice_from_str(&serde_json::to_string(&envelope).unwrap()).unwrap();
        let projected_value: Value = serde_json::from_str(&projected_wire).unwrap();
        assert_eq!(
            projected_value["payload"],
            serde_json::json!({"summary": "visible"})
        );
        assert!(
            GraphModelEnvelope::new(
                GraphModelClass::Node,
                serde_json::json!({"summary": "visible"}),
            )
            .project_declared_slice()
            .is_err()
        );

        let mismatched = GraphModelEnvelope::new(
            GraphModelClass::Edge,
            serde_json::json!({"summary": "visible"}),
        )
        .with_slice(GraphSliceDeclarationV1 {
            schema_version: GRAPH_SLICE_MANIFEST_VERSION,
            mode: GraphSliceMode::LlmIn,
            manifest: envelope.slice.as_ref().unwrap().manifest.clone(),
        });
        assert!(matches!(
            mismatched.project_declared_slice(),
            Err(GraphModelError::InvalidPayload(message))
                if message.contains("does not match envelope model Edge")
        ));
    }

    #[test]
    fn batch_slice_projection_matches_individual_projection_and_is_fail_closed() {
        let manifest = GraphSliceManifestV1 {
            schema_version: GRAPH_SLICE_MANIFEST_VERSION,
            model_name: Some("Node".to_owned()),
            retain_unlisted_modes: BTreeSet::new(),
            fields: BTreeMap::from([(
                "summary".to_owned(),
                GraphSliceFieldV1 {
                    modes: BTreeSet::from([GraphSliceMode::LlmIn]),
                    child: None,
                    child_location: Default::default(),
                },
            )]),
        };
        let envelopes: Vec<_> = ["one", "two"]
            .into_iter()
            .map(|summary| {
                GraphModelEnvelope::new(
                    GraphModelClass::Node,
                    serde_json::json!({"summary": summary, "secret": "hidden"}),
                )
                .with_slice(GraphSliceDeclarationV1 {
                    schema_version: GRAPH_SLICE_MANIFEST_VERSION,
                    mode: GraphSliceMode::LlmIn,
                    manifest: manifest.clone(),
                })
            })
            .collect();
        let request = serde_json::to_string(&envelopes).unwrap();
        let batch: Vec<GraphModelSliceV1> =
            serde_json::from_str(&project_graph_model_slices_from_str(&request).unwrap()).unwrap();
        let individual: Vec<GraphModelSliceV1> = envelopes
            .iter()
            .map(|envelope| {
                serde_json::from_str(
                    &project_graph_model_slice_from_str(&serde_json::to_string(envelope).unwrap())
                        .unwrap(),
                )
                .unwrap()
            })
            .collect();
        assert_eq!(batch, individual);

        let mut invalid = envelopes;
        invalid[1].slice.as_mut().unwrap().manifest.model_name = Some("Edge".to_owned());
        assert!(
            project_graph_model_slices_from_str(&serde_json::to_string(&invalid).unwrap()).is_err()
        );
    }
