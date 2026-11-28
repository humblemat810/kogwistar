    #[test]
    fn workflow_operation_version_matches_shared_python_contract_fixture() {
        let versions: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/workflow-operation-version-v1.json"
        )))
        .expect("valid workflow operation version fixture");

        let validate_version = |version: Value| {
            let metadata: BTreeMap<String, Value> = serde_json::from_value(serde_json::json!({
                "entity_type": "workflow_node",
                "workflow_id": "wf:version-fixture",
                "wf_op": "compute",
                "wf_version": version
            }))
            .expect("metadata object");
            validate_workflow_node_metadata(&metadata)
        };

        for version in versions["valid"].as_array().expect("valid versions") {
            assert!(validate_version(version.clone()).is_ok());
        }
        for version in versions["invalid"].as_array().expect("invalid versions") {
            assert!(
                validate_version(version.clone()).is_err(),
                "Rust accepted invalid operation version {version}"
            );
        }

        let legacy_metadata: BTreeMap<String, Value> = serde_json::from_value(serde_json::json!({
            "entity_type": "workflow_node",
            "workflow_id": "wf:version-fixture",
            "wf_op": "compute"
        }))
        .expect("legacy metadata object");
        let parsed: WorkflowNodeMetadataV1 = serde_json::from_value(serde_json::Value::Object(
            legacy_metadata.clone().into_iter().collect(),
        ))
        .expect("legacy metadata defaults");
        assert_eq!(
            parsed.wf_version,
            versions["legacy_default"]
                .as_str()
                .expect("legacy default version string")
        );
        assert!(validate_workflow_node_metadata(&legacy_metadata).is_ok());
    }

    #[test]
    fn workflow_edge_and_conversation_edge_metadata_preserve_edge_shape() {
        let fixture = fixture();
        let mut workflow_edge = fixture["cases"][1]["payload"].clone();
        workflow_edge["metadata"] = serde_json::json!({
            "entity_type": "workflow_edge",
            "workflow_id": "wf:demo",
            "wf_predicate": "ok",
            "wf_priority": 10,
            "wf_multiplicity": "many"
        });
        assert!(
            GraphModelEnvelope::new(GraphModelClass::WorkflowEdge, workflow_edge)
                .validate_family_v1()
                .is_ok()
        );

        let mut conversation_edge = fixture["cases"][1]["payload"].clone();
        conversation_edge["metadata"] = serde_json::json!({
            "causal_type": "chain",
            "char_distance_from_last_summary": 12,
            "turn_distance_from_last_summary": 1
        });
        assert!(
            GraphModelEnvelope::new(GraphModelClass::ConversationEdge, conversation_edge)
                .validate_family_v1()
                .is_ok()
        );
    }

    #[test]
    fn workflow_runtime_subtypes_preserve_node_edge_shape_and_metadata_tag() {
        let fixture = fixture();
        let runtime_contract: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/workflow-runtime-node-metadata-v1.json"
        )))
        .expect("valid shared workflow runtime node metadata contract");
        let node_payload = fixture["cases"][0]["payload"].clone();
        for (model_class, wire_name) in [
            (GraphModelClass::WorkflowRunNode, "workflow_run_node"),
            (
                GraphModelClass::WorkflowStepExecNode,
                "workflow_step_exec_node",
            ),
            (
                GraphModelClass::WorkflowCheckpointNode,
                "workflow_checkpoint_node",
            ),
            (
                GraphModelClass::WorkflowCompletedNode,
                "workflow_completed_node",
            ),
            (
                GraphModelClass::WorkflowCancelledNode,
                "workflow_cancelled_node",
            ),
            (GraphModelClass::WorkflowFailedNode, "workflow_failed_node"),
        ] {
            let mut payload = node_payload.clone();
            payload["metadata"] = runtime_contract["cases"]
                .as_array()
                .unwrap()
                .iter()
                .find(|case| {
                    case["model_class"] == wire_name && case["valid"].as_bool() == Some(true)
                })
                .expect("shared fixture has valid runtime-node case")["metadata"]
                .clone();
            assert!(
                GraphModelEnvelope::new(model_class, payload)
                    .validate_family_v1()
                    .is_ok(),
                "runtime node should validate: {model_class:?}"
            );
        }

        let mut edge_payload = fixture["cases"][1]["payload"].clone();
        edge_payload["metadata"] = serde_json::json!({"relation": "next"});
        assert!(
            GraphModelEnvelope::new(GraphModelClass::WorkflowRuntimeEdge, edge_payload)
                .validate_family_v1()
                .is_ok()
        );

        let mut wrong = fixture["cases"][0]["payload"].clone();
        wrong["metadata"] = serde_json::json!({"entity_type": "workflow_failed"});
        assert!(matches!(
            GraphModelEnvelope::new(GraphModelClass::WorkflowRunNode, wrong).validate_family_v1(),
            Err(GraphModelError::InvalidEntityFields { .. })
        ));
    }

    #[test]
    fn llm_pure_chroma_and_document_families_have_distinct_requirements() {
        let fixture = fixture();

        let mut llm = fixture["cases"][0]["payload"].clone();
        llm.as_object_mut().unwrap().remove("embedding");
        llm["mentions"] = serde_json::json!([{
            "collection_page_url": "https://example.test/collection",
            "document_page_url": "https://example.test/document",
            "doc_id": "doc:extraction",
            "page_number": 1,
            "start_char": 0,
            "end_char": 1,
            "excerpt": "x"
        }]);
        assert!(
            GraphModelEnvelope::new(GraphModelClass::LlmNodeExtraction, llm)
                .validate_family_v1()
                .is_ok()
        );

        let llm_with_embedding = fixture["cases"][0]["payload"].clone();
        assert!(matches!(
            GraphModelEnvelope::new(GraphModelClass::LlmNode, llm_with_embedding)
                .validate_family_v1(),
            Err(GraphModelError::InvalidEntityFields {
                reason: "LLM graph payload must not contain an embedding",
                ..
            })
        ));

        let mut pure_chroma = fixture["cases"][0]["payload"].clone();
        pure_chroma.as_object_mut().unwrap().remove("mentions");
        assert!(
            GraphModelEnvelope::new(GraphModelClass::PureChromaNode, pure_chroma)
                .validate_family_v1()
                .is_ok()
        );

        let mut doc = serde_json::json!({
            "id": "doc-node:1",
            "label": "Page 1",
            "type": "page",
            "summary": "A document page",
            "doc_id": "document:1"
        });
        doc["mentions"] = fixture["cases"][0]["payload"]["mentions"].clone();
        assert!(
            GraphModelEnvelope::new(GraphModelClass::DocNode, doc)
                .validate_family_v1()
                .is_ok()
        );

        let invalid_doc = serde_json::json!({"type": "section", "doc_id": "document:1"});
        assert!(
            GraphModelEnvelope::new(GraphModelClass::DocNode, invalid_doc)
                .validate_family_v1()
                .is_err()
        );

        let mut acl = fixture["cases"][0]["payload"].clone();
        acl["metadata"] = serde_json::json!({
            "entity_type": "acl_record",
            "acl_truth_graph": "truth:main",
            "acl_target_entity_id": "parity-node",
            "acl_target_grain": "node",
            "acl_version": 1,
            "acl_mode": "private",
            "level_from_root": 0
        });
        assert!(
            GraphModelEnvelope::new(GraphModelClass::AclNode, acl)
                .validate_family_v1()
                .is_ok()
        );

        let mut flattened = fixture["cases"][0]["payload"].clone();
        flattened.as_object_mut().unwrap().remove("embedding");
        flattened["mentions"] = serde_json::json!([{"span_ids": ["sp:1"]}]);
        assert!(
            GraphModelEnvelope::new(GraphModelClass::FlattenedLlmNode, flattened)
                .validate_family_v1()
                .is_ok()
        );

        let mut invalid_acl = fixture["cases"][0]["payload"].clone();
        invalid_acl["metadata"] = serde_json::json!({
            "entity_type": "acl_record",
            "acl_truth_graph": "truth:main",
            "acl_target_entity_id": "parity-node",
            "acl_target_grain": "unknown-grain",
            "acl_version": 1,
            "acl_mode": "private"
        });
        assert!(
            GraphModelEnvelope::new(GraphModelClass::AclNode, invalid_acl)
                .validate_family_v1()
                .is_err()
        );
    }
