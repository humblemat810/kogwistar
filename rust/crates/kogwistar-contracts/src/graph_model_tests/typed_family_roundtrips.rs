    #[test]
    fn workflow_and_conversation_family_metadata_is_typed_and_validated() {
        let fixture = fixture();
        let mut workflow_node = fixture["cases"][0]["payload"].clone();
        workflow_node["metadata"] = serde_json::json!({
            "entity_type": "workflow_node",
            "workflow_id": "wf:demo",
            "wf_op": "parse",
            "wf_version": "v1",
            "wf_start": true
        });
        assert!(
            GraphModelEnvelope::new(GraphModelClass::WorkflowNode, workflow_node)
                .validate_family_v1()
                .is_ok()
        );

        let mut conversation_node = fixture["cases"][0]["payload"].clone();
        conversation_node["metadata"] = serde_json::json!({
            "entity_type": "conversation_node",
            "level_from_root": 0,
            "role": "user",
            "turn_index": 1,
            "conversation_id": "conversation:demo",
            "user_id": "user:demo"
        });
        assert!(
            GraphModelEnvelope::new(GraphModelClass::ConversationNode, conversation_node)
                .validate_family_v1()
                .is_ok()
        );

        for (entity_type, role, turn_index) in [
            ("conversation_start", "system", -1),
            ("conversation_turn", "user", 0),
            ("conversation_summary", "system", 1),
            ("assistant_turn", "assistant", 2),
        ] {
            let mut conversation_subtype = fixture["cases"][0]["payload"].clone();
            conversation_subtype["metadata"] = serde_json::json!({
                "entity_type": entity_type,
                "level_from_root": 0,
                "role": role,
                "turn_index": turn_index,
                "conversation_id": "conversation:demo",
                "user_id": "user:demo"
            });
            assert!(
                GraphModelEnvelope::new(GraphModelClass::ConversationNode, conversation_subtype)
                    .validate_family_v1()
                    .is_ok(),
                "Python ConversationNode subtype {entity_type:?} must remain valid"
            );
        }

        let mut invalid_conversation = fixture["cases"][0]["payload"].clone();
        invalid_conversation["metadata"] = serde_json::json!({
            "entity_type": "conversation_node",
            "level_from_root": 0,
            "turn_distance_from_last_summary": 1
        });
        assert!(matches!(
            GraphModelEnvelope::new(GraphModelClass::ConversationNode, invalid_conversation)
                .validate_family_v1(),
            Err(GraphModelError::InvalidEntityFields {
                model_class: GraphModelClass::ConversationNode,
                ..
            })
        ));
    }

    #[test]
    fn conversation_node_round_trip_preserves_python_subtype_and_extension_fields() {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/conversation-create-start-node-v1.json"
        )))
        .expect("valid Python ConversationNode fixture");
        let wire = serde_json::json!({
            "schema_version": GRAPH_MODEL_ENVELOPE_VERSION,
            "model_class": "conversation_node",
            "payload": fixture["payload"],
        })
        .to_string();

        let envelope =
            RawGraphModelEnvelopeV1::from_json(&wire).expect("valid ConversationNode envelope");
        assert_eq!(envelope.model_class, GraphModelClass::ConversationNode);
        envelope
            .validate_family_v1()
            .expect("valid ConversationNode subtype metadata");

        let round_trip: Value = serde_json::from_str(
            &round_trip_raw_graph_model_envelope_from_str(&wire)
                .expect("lossless Rust envelope round trip"),
        )
        .expect("valid Rust round-trip JSON");
        assert_eq!(round_trip["model_class"], "conversation_node");
        assert_eq!(round_trip["payload"], fixture["payload"]);
        assert_eq!(round_trip["payload"]["role"], "system");
        assert_eq!(round_trip["payload"]["turn_index"], -1);
        assert_eq!(
            round_trip["payload"]["metadata"]["entity_type"],
            "conversation_start"
        );
    }

    #[test]
    fn workflow_node_round_trip_preserves_python_tag_and_extension_metadata() {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/workflow-node-model-v1.json"
        )))
        .expect("valid Python WorkflowNode fixture");
        let wire = serde_json::json!({
            "schema_version": fixture["schema_version"],
            "model_class": fixture["model_class"],
            "payload": fixture["payload"],
        })
        .to_string();

        let envelope =
            RawGraphModelEnvelopeV1::from_json(&wire).expect("valid WorkflowNode envelope");
        assert_eq!(envelope.model_class, GraphModelClass::WorkflowNode);
        envelope
            .validate_family_v1()
            .expect("valid typed WorkflowNode metadata");

        let round_trip: Value = serde_json::from_str(
            &round_trip_raw_graph_model_envelope_from_str(&wire)
                .expect("lossless Rust envelope round trip"),
        )
        .expect("valid Rust round-trip JSON");
        assert_eq!(round_trip["schema_version"], fixture["schema_version"]);
        assert_eq!(round_trip["model_class"], "workflow_node");
        assert_eq!(round_trip["payload"], fixture["payload"]);
        assert_eq!(
            round_trip["payload"]["metadata"]["vendor_extension"],
            fixture["payload"]["metadata"]["vendor_extension"]
        );
        use sha2::Digest;
        let canonical_payload = crate::canonical_json(&round_trip["payload"], false);
        assert_eq!(
            hex::encode(sha2::Sha256::digest(canonical_payload.as_bytes())),
            fixture["canonical_json_sha256"]
        );
    }

    #[test]
    fn workflow_run_node_round_trip_preserves_python_tag_and_extension_metadata() {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/workflow-run-node-model-v1.json"
        )))
        .expect("valid Python WorkflowRunNode fixture");
        let wire = serde_json::json!({
            "schema_version": fixture["schema_version"],
            "model_class": fixture["model_class"],
            "payload": fixture["payload"],
        })
        .to_string();

        let envelope =
            RawGraphModelEnvelopeV1::from_json(&wire).expect("valid WorkflowRunNode envelope");
        assert_eq!(envelope.model_class, GraphModelClass::WorkflowRunNode);
        envelope
            .validate_family_v1()
            .expect("valid typed WorkflowRunNode metadata");

        let round_trip: Value = serde_json::from_str(
            &round_trip_raw_graph_model_envelope_from_str(&wire)
                .expect("lossless Rust envelope round trip"),
        )
        .expect("valid Rust round-trip JSON");
        assert_eq!(round_trip["schema_version"], fixture["schema_version"]);
        assert_eq!(round_trip["model_class"], "workflow_run_node");
        assert_eq!(round_trip["payload"], fixture["payload"]);
        assert_eq!(round_trip["payload"]["metadata"]["status"], "running");
        use sha2::Digest;
        let canonical_payload = crate::canonical_json(&round_trip["payload"], false);
        assert_eq!(
            hex::encode(sha2::Sha256::digest(canonical_payload.as_bytes())),
            fixture["canonical_json_sha256"]
        );
    }

    #[test]
    fn workflow_step_exec_node_round_trip_preserves_python_tag_and_result() {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/workflow-step-exec-node-model-v1.json"
        )))
        .expect("valid Python WorkflowStepExecNode fixture");
        let wire = serde_json::json!({
            "schema_version": fixture["schema_version"],
            "model_class": fixture["model_class"],
            "payload": fixture["payload"],
        })
        .to_string();

        let envelope =
            RawGraphModelEnvelopeV1::from_json(&wire).expect("valid WorkflowStepExecNode envelope");
        assert_eq!(envelope.model_class, GraphModelClass::WorkflowStepExecNode);
        envelope
            .validate_family_v1()
            .expect("valid typed WorkflowStepExecNode metadata");

        let round_trip: Value = serde_json::from_str(
            &round_trip_raw_graph_model_envelope_from_str(&wire)
                .expect("lossless Rust envelope round trip"),
        )
        .expect("valid Rust round-trip JSON");
        assert_eq!(round_trip["schema_version"], fixture["schema_version"]);
        assert_eq!(round_trip["model_class"], "workflow_step_exec_node");
        assert_eq!(round_trip["payload"], fixture["payload"]);
        assert_eq!(round_trip["payload"]["metadata"]["step_seq"], 2);
        assert_eq!(
            round_trip["payload"]["metadata"]["result_json"],
            "{\"parsed\":true}"
        );
        use sha2::Digest;
        let canonical_payload = crate::canonical_json(&round_trip["payload"], false);
        assert_eq!(
            hex::encode(sha2::Sha256::digest(canonical_payload.as_bytes())),
            fixture["canonical_json_sha256"]
        );
    }

    #[test]
    fn workflow_checkpoint_node_round_trip_preserves_python_tag_and_state() {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/workflow-checkpoint-node-model-v1.json"
        )))
        .expect("valid Python WorkflowCheckpointNode fixture");
        let wire = serde_json::json!({
            "schema_version": fixture["schema_version"],
            "model_class": fixture["model_class"],
            "payload": fixture["payload"],
        })
        .to_string();

        let envelope = RawGraphModelEnvelopeV1::from_json(&wire)
            .expect("valid WorkflowCheckpointNode envelope");
        assert_eq!(
            envelope.model_class,
            GraphModelClass::WorkflowCheckpointNode
        );
        envelope
            .validate_family_v1()
            .expect("valid typed WorkflowCheckpointNode metadata");

        let round_trip: Value = serde_json::from_str(
            &round_trip_raw_graph_model_envelope_from_str(&wire)
                .expect("lossless Rust envelope round trip"),
        )
        .expect("valid Rust round-trip JSON");
        assert_eq!(round_trip["schema_version"], fixture["schema_version"]);
        assert_eq!(round_trip["model_class"], "workflow_checkpoint_node");
        assert_eq!(round_trip["payload"], fixture["payload"]);
        assert_eq!(
            round_trip["payload"]["metadata"]["state_json"],
            "{\"cursor\":\"section-2\",\"items\":3}"
        );
    }

    #[test]
    fn workflow_completed_node_round_trip_preserves_python_tag_and_terminal_state() {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/workflow-completed-node-model-v1.json"
        )))
        .expect("valid Python WorkflowCompletedNode fixture");
        let wire = serde_json::json!({
            "schema_version": fixture["schema_version"],
            "model_class": fixture["model_class"],
            "payload": fixture["payload"],
        })
        .to_string();

        let envelope = RawGraphModelEnvelopeV1::from_json(&wire)
            .expect("valid WorkflowCompletedNode envelope");
        assert_eq!(envelope.model_class, GraphModelClass::WorkflowCompletedNode);
        envelope
            .validate_family_v1()
            .expect("valid typed WorkflowCompletedNode metadata");

        let round_trip: Value = serde_json::from_str(
            &round_trip_raw_graph_model_envelope_from_str(&wire)
                .expect("lossless Rust envelope round trip"),
        )
        .expect("valid Rust round-trip JSON");
        assert_eq!(round_trip["schema_version"], fixture["schema_version"]);
        assert_eq!(round_trip["model_class"], "workflow_completed_node");
        assert_eq!(round_trip["payload"], fixture["payload"]);
        assert_eq!(round_trip["payload"]["metadata"]["accepted_step_seq"], 3);
    }

    #[test]
    fn workflow_cancelled_node_round_trip_preserves_python_tag_and_terminal_state() {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/workflow-cancelled-node-model-v1.json"
        )))
        .expect("valid Python WorkflowCancelledNode fixture");
        let wire = serde_json::json!({
            "schema_version": fixture["schema_version"],
            "model_class": fixture["model_class"],
            "payload": fixture["payload"],
        })
        .to_string();

        let envelope = RawGraphModelEnvelopeV1::from_json(&wire)
            .expect("valid WorkflowCancelledNode envelope");
        assert_eq!(envelope.model_class, GraphModelClass::WorkflowCancelledNode);
        envelope
            .validate_family_v1()
            .expect("valid typed WorkflowCancelledNode metadata");

        let round_trip: Value = serde_json::from_str(
            &round_trip_raw_graph_model_envelope_from_str(&wire)
                .expect("lossless Rust envelope round trip"),
        )
        .expect("valid Rust round-trip JSON");
        assert_eq!(round_trip["schema_version"], fixture["schema_version"]);
        assert_eq!(round_trip["model_class"], "workflow_cancelled_node");
        assert_eq!(round_trip["payload"], fixture["payload"]);
        assert_eq!(round_trip["payload"]["metadata"]["accepted_step_seq"], 2);
    }

    #[test]
    fn workflow_failed_node_round_trip_preserves_python_tag_errors_and_extensions() {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/workflow-failed-node-model-v1.json"
        )))
        .expect("valid Python WorkflowFailedNode fixture");
        let wire = serde_json::json!({
            "schema_version": fixture["schema_version"],
            "model_class": fixture["model_class"],
            "payload": fixture["payload"],
        })
        .to_string();

        let envelope =
            RawGraphModelEnvelopeV1::from_json(&wire).expect("valid WorkflowFailedNode envelope");
        assert_eq!(envelope.model_class, GraphModelClass::WorkflowFailedNode);
        envelope
            .validate_family_v1()
            .expect("valid typed WorkflowFailedNode metadata");

        let round_trip: Value = serde_json::from_str(
            &round_trip_raw_graph_model_envelope_from_str(&wire)
                .expect("lossless Rust envelope round trip"),
        )
        .expect("valid Rust round-trip JSON");
        assert_eq!(round_trip["schema_version"], fixture["schema_version"]);
        assert_eq!(round_trip["model_class"], "workflow_failed_node");
        assert_eq!(round_trip["payload"], fixture["payload"]);
        assert_eq!(
            round_trip["payload"]["metadata"]["errors"],
            serde_json::json!(["step parse failed", "retry limit reached"])
        );
    }

    #[test]
    fn llm_node_round_trip_preserves_python_grouped_groundings() {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/llm-node-model-v1.json"
        )))
        .expect("valid Python LLMNode fixture");
        let wire = serde_json::json!({
            "schema_version": fixture["schema_version"],
            "model_class": fixture["model_class"],
            "payload": fixture["payload"],
        })
        .to_string();

        let envelope = RawGraphModelEnvelopeV1::from_json(&wire).expect("valid LLMNode envelope");
        assert_eq!(envelope.model_class, GraphModelClass::LlmNode);
        envelope
            .validate_family_v1()
            .expect("valid typed grouped LLMNode groundings");

        let round_trip: Value = serde_json::from_str(
            &round_trip_raw_graph_model_envelope_from_str(&wire)
                .expect("lossless Rust envelope round trip"),
        )
        .expect("valid Rust round-trip JSON");
        assert_eq!(round_trip["schema_version"], fixture["schema_version"]);
        assert_eq!(round_trip["model_class"], "llm_node");
        assert_eq!(round_trip["payload"], fixture["payload"]);
    }

    #[test]
    fn llm_node_extraction_round_trip_preserves_flat_excerpt_spans() {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/llm-node-extraction-model-v1.json"
        )))
        .expect("valid Python LLMNodeExtraction fixture");
        let wire = serde_json::json!({
            "schema_version": fixture["schema_version"],
            "model_class": fixture["model_class"],
            "payload": fixture["payload"],
        })
        .to_string();

        let envelope =
            RawGraphModelEnvelopeV1::from_json(&wire).expect("valid extraction-node envelope");
        assert_eq!(envelope.model_class, GraphModelClass::LlmNodeExtraction);
        envelope
            .validate_family_v1()
            .expect("valid flat extraction spans");
        let typed: LlmNodeExtractionV1 =
            serde_json::from_value(fixture["payload"].clone()).expect("typed flat spans");
        assert_eq!(typed.mentions.len(), 1);
        assert_eq!(typed.mentions[0].excerpt, "extracted");

        let round_trip: Value = serde_json::from_str(
            &round_trip_raw_graph_model_envelope_from_str(&wire)
                .expect("lossless Rust envelope round trip"),
        )
        .expect("valid Rust round-trip JSON");
        assert_eq!(round_trip["model_class"], "llm_node_extraction");
        assert_eq!(round_trip["payload"], fixture["payload"]);

        let mut missing_excerpt = fixture["payload"].clone();
        missing_excerpt["mentions"][0]
            .as_object_mut()
            .expect("span object")
            .remove("excerpt");
        let invalid_wire = serde_json::json!({
            "schema_version": fixture["schema_version"],
            "model_class": fixture["model_class"],
            "payload": missing_excerpt,
        })
        .to_string();
        assert!(
            RawGraphModelEnvelopeV1::from_json(&invalid_wire)
                .expect("valid JSON envelope")
                .validate_family_v1()
                .is_err()
        );
    }

    #[test]
    fn llm_edge_extraction_round_trip_preserves_flat_spans_and_endpoints() {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/llm-edge-extraction-model-v1.json"
        )))
        .expect("valid Python LLMEdgeExtraction fixture");
        let wire = serde_json::json!({
            "schema_version": fixture["schema_version"],
            "model_class": fixture["model_class"],
            "payload": fixture["payload"],
        })
        .to_string();

        let envelope =
            RawGraphModelEnvelopeV1::from_json(&wire).expect("valid extraction-edge envelope");
        assert_eq!(envelope.model_class, GraphModelClass::LlmEdgeExtraction);
        envelope
            .validate_family_v1()
            .expect("valid flat extraction spans and endpoints");
        let typed: LlmEdgeExtractionV1 =
            serde_json::from_value(fixture["payload"].clone()).expect("typed extraction edge");
        assert_eq!(typed.mentions.len(), 1);
        assert_eq!(typed.mentions[0].excerpt, "extracted");
        assert_eq!(typed.source_ids[0].0, "nn:extract-source");
        assert_eq!(typed.source_edge_ids.as_option().unwrap()[0].0, "ne:parent");

        let round_trip: Value = serde_json::from_str(
            &round_trip_raw_graph_model_envelope_from_str(&wire)
                .expect("lossless Rust envelope round trip"),
        )
        .expect("valid Rust round-trip JSON");
        assert_eq!(round_trip["model_class"], "llm_edge_extraction");
        assert_eq!(round_trip["payload"], fixture["payload"]);

        let mut empty_endpoint = fixture["payload"].clone();
        empty_endpoint["source_ids"][0] = Value::String(String::new());
        let invalid_wire = serde_json::json!({
            "schema_version": fixture["schema_version"],
            "model_class": fixture["model_class"],
            "payload": empty_endpoint,
        })
        .to_string();
        assert!(
            RawGraphModelEnvelopeV1::from_json(&invalid_wire)
                .expect("valid JSON envelope")
                .validate_family_v1()
                .is_err()
        );
    }

    #[test]
    fn flattened_llm_node_round_trip_preserves_python_evidence_ids() {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/llm-node-flattened-model-v1.json"
        )))
        .expect("valid Python FlattenedLLMNode fixture");
        let wire = serde_json::json!({
            "schema_version": fixture["schema_version"],
            "model_class": fixture["model_class"],
            "payload": fixture["payload"],
        })
        .to_string();

        let envelope =
            RawGraphModelEnvelopeV1::from_json(&wire).expect("valid flattened-node envelope");
        assert_eq!(envelope.model_class, GraphModelClass::FlattenedLlmNode);
        envelope
            .validate_family_v1()
            .expect("valid flattened evidence references");
        let typed: FlattenedLlmNodeV1 =
            serde_json::from_value(fixture["payload"].clone()).expect("typed flattened node");
        assert_eq!(typed.mentions.len(), 1);
        assert_eq!(typed.mentions[0].span_ids, ["span:node-1".to_string()]);
        assert!(typed.mentions[0].multimodal_span_ids.is_empty());

        let round_trip: Value = serde_json::from_str(
            &round_trip_raw_graph_model_envelope_from_str(&wire)
                .expect("lossless Rust envelope round trip"),
        )
        .expect("valid Rust round-trip JSON");
        assert_eq!(round_trip["model_class"], "flattened_llm_node");
        assert_eq!(round_trip["payload"], fixture["payload"]);
    }

    #[test]
    fn flattened_llm_edge_round_trip_matches_python_and_rejects_blank_endpoints() {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/llm-edge-flattened-model-v1.json"
        )))
        .expect("valid Python FlattenedLLMEdge fixture");
        let wire = serde_json::json!({
            "schema_version": fixture["schema_version"],
            "model_class": fixture["model_class"],
            "payload": fixture["payload"],
        })
        .to_string();

        let envelope =
            RawGraphModelEnvelopeV1::from_json(&wire).expect("valid flattened-edge envelope");
        assert_eq!(envelope.model_class, GraphModelClass::FlattenedLlmEdge);
        envelope
            .validate_family_v1()
            .expect("valid flattened evidence references and endpoints");
        let typed: FlattenedLlmEdgeV1 =
            serde_json::from_value(fixture["payload"].clone()).expect("typed flattened edge");
        assert_eq!(typed.mentions.len(), 1);
        assert_eq!(typed.mentions[0].span_ids, ["span:edge-1"]);
        assert_eq!(typed.source_ids[0].0, "nn:flat-source");
        assert_eq!(typed.source_edge_ids.as_option().unwrap()[0].0, "ne:parent");

        let round_trip: Value = serde_json::from_str(
            &round_trip_raw_graph_model_envelope_from_str(&wire)
                .expect("lossless Rust envelope round trip"),
        )
        .expect("valid Rust round-trip JSON");
        assert_eq!(round_trip["model_class"], "flattened_llm_edge");
        assert_eq!(round_trip["payload"], fixture["payload"]);

        let mut invalid_payload = fixture["payload"].clone();
        invalid_payload["source_edge_ids"][0] = Value::String(String::new());
        let invalid_wire = serde_json::json!({
            "schema_version": fixture["schema_version"],
            "model_class": fixture["model_class"],
            "payload": invalid_payload,
        })
        .to_string();
        assert!(
            RawGraphModelEnvelopeV1::from_json(&invalid_wire)
                .expect("valid JSON envelope")
                .validate_family_v1()
                .is_err()
        );
    }

    #[test]
    fn assoc_flattened_llm_node_round_trip_has_no_inline_mentions() {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/llm-node-assoc-flattened-model-v1.json"
        )))
        .expect("valid Python AssocFlattenedLLMNode fixture");
        let wire = serde_json::json!({
            "schema_version": fixture["schema_version"],
            "model_class": fixture["model_class"],
            "payload": fixture["payload"],
        })
        .to_string();

        let envelope = RawGraphModelEnvelopeV1::from_json(&wire)
            .expect("valid associated flattened-node envelope");
        assert_eq!(envelope.model_class, GraphModelClass::AssocFlattenedLlmNode);
        envelope
            .validate_family_v1()
            .expect("associated grounding belongs to the separate association tables");
        let typed: AssocFlattenedLlmNodeV1 = serde_json::from_value(fixture["payload"].clone())
            .expect("typed associated flattened node");
        assert_eq!(typed.fields.local_id.as_deref(), Some("nn:assoc-flat-node"));

        let round_trip: Value = serde_json::from_str(
            &round_trip_raw_graph_model_envelope_from_str(&wire)
                .expect("lossless Rust envelope round trip"),
        )
        .expect("valid Rust round-trip JSON");
        assert_eq!(round_trip["model_class"], "assoc_flattened_llm_node");
        assert_eq!(round_trip["payload"], fixture["payload"]);
    }

    #[test]
    fn assoc_flattened_llm_edge_round_trip_preserves_python_endpoints() {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/llm-edge-assoc-flattened-model-v1.json"
        )))
        .expect("valid Python AssocFlattenedLLMEdge fixture");
        let wire = serde_json::json!({
            "schema_version": fixture["schema_version"],
            "model_class": fixture["model_class"],
            "payload": fixture["payload"],
        })
        .to_string();

        let envelope = RawGraphModelEnvelopeV1::from_json(&wire)
            .expect("valid associated flattened-edge envelope");
        assert_eq!(envelope.model_class, GraphModelClass::AssocFlattenedLlmEdge);
        envelope
            .validate_family_v1()
            .expect("associated grounding belongs to separate association tables");
        let typed: AssocFlattenedLlmEdgeV1 = serde_json::from_value(fixture["payload"].clone())
            .expect("typed associated flattened edge");
        assert_eq!(typed.graph_entity_kind(), GraphEntityKind::Edge);
        assert_eq!(typed.graph_entity_id(), Some("ne:assoc-flat-edge"));
        assert_eq!(typed.source_node_ids().unwrap()[0].0, "nn:source");
        assert_eq!(typed.target_node_ids().unwrap()[0].0, "nn:target");
        assert_eq!(typed.relation(), Some("connects_to"));

        let round_trip: Value = serde_json::from_str(
            &round_trip_raw_graph_model_envelope_from_str(&wire)
                .expect("lossless Rust envelope round trip"),
        )
        .expect("valid Rust round-trip JSON");
        assert_eq!(round_trip["model_class"], "assoc_flattened_llm_edge");
        assert_eq!(round_trip["payload"], fixture["payload"]);

        let mut invalid_payload = fixture["payload"].clone();
        invalid_payload["source_ids"] = serde_json::json!([""]);
        assert!(
            GraphModelEnvelope::new(GraphModelClass::AssocFlattenedLlmEdge, invalid_payload)
                .validate_family_v1()
                .is_err()
        );
    }

    #[test]
    fn llm_edge_typed_round_trip_matches_python_llm_fixture() {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/llm-edge-model-v1.json"
        )))
        .expect("valid shared Python LLMEdge fixture");
        let payload = fixture["payload"].clone();
        let envelope = GraphModelEnvelope::new(GraphModelClass::LlmEdge, payload.clone());

        envelope
            .validate_family_v1()
            .expect("valid LLM edge payload and grounding");
        let edge: LlmEdgeV1 = serde_json::from_value(payload).expect("typed Rust LLM edge");
        let serialized = serde_json::to_value(edge).expect("serialize typed LLM edge");

        assert_eq!(serialized["id"], "ne:llm-edge");
        assert_eq!(serialized["local_id"], "ne:llm-edge");
        assert_eq!(serialized["source_ids"], serde_json::json!(["nn:source"]));
        assert_eq!(
            serialized["target_ids"],
            serde_json::json!(["node:existing"])
        );
        assert_eq!(
            serialized["source_edge_ids"],
            serde_json::json!(["ne:parent"])
        );
        assert_eq!(serialized["target_edge_ids"], Value::Null);
        assert_eq!(serialized["properties"]["confidence_label"], "high");
        assert!(serialized.get("embedding").is_none());
        assert!(serialized.get("metadata").is_none());
        assert_eq!(
            crate::canonical_json(&serialized, false),
            fixture["canonical_json"]
                .as_str()
                .expect("Python canonical LLM edge JSON")
        );

        let mut legacy_payload = fixture["payload"].clone();
        legacy_payload["doc_id"] = serde_json::json!("doc:legacy-top-level");
        let legacy_edge: LlmEdgeV1 =
            serde_json::from_value(legacy_payload).expect("legacy optional document ID");
        let legacy_serialized =
            serde_json::to_value(legacy_edge).expect("serialize legacy-compatible LLM edge");
        assert_eq!(legacy_serialized["doc_id"], "doc:legacy-top-level");
    }

    #[test]
    fn llm_node_typed_serialization_matches_python_llm_slice_omissions() {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/llm-node-model-v1.json"
        )))
        .expect("valid shared Python LLMNode fixture");
        let node: LlmNodeV1 =
            serde_json::from_value(fixture["payload"].clone()).expect("typed Rust LLM node");

        assert_eq!(
            serde_json::to_value(node).expect("serialize typed LLM node"),
            fixture["payload"]
        );
    }
