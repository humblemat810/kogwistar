    #[test]
    fn internal_graph_mutation_contract_requires_atomic_versioned_batches() {
        let body: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/graph-mutation-batch-v1.json"
        )))
        .unwrap();
        let input: GraphMutationBatchInput = serde_json::from_value(body.clone()).unwrap();
        let mutations = graph_mutation_batch_value(input).unwrap();
        assert_eq!(mutations.len(), 1);
        assert_eq!(mutations[0].scope.namespace, "tenant-a");
        assert_eq!(
            mutations[0].record.embedding.as_deref(),
            Some([1.0, 0.0].as_slice())
        );
        assert!(mutations[0].record.metadata.is_empty());

        let graph_fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/python-graph-model-payloads-v1.json"
        )))
        .unwrap();
        let typed_payload = graph_fixture["cases"][0]["payload"].clone();
        let mut typed_body = body.clone();
        typed_body["mutations"][0]["payload"] = typed_payload.clone();
        typed_body["mutations"][0]["record"]["id"] = typed_payload["id"].clone();
        typed_body["mutations"][0]["model_envelope"] = json!({
            "schema_version": 1,
            "model_class": "node",
            "payload": typed_payload,
        });
        let typed_input: GraphMutationBatchInput =
            serde_json::from_value(typed_body.clone()).unwrap();
        let typed_mutations = graph_mutation_batch_value(typed_input.clone()).unwrap();
        assert_eq!(typed_mutations[0].record.metadata["_class_name"], "Node");
        assert!(typed_mutations[0].payload["metadata"].get("_class_name").is_none());
        assert!(validate_graph_mutation_envelopes(&typed_input, &typed_mutations).is_ok());

        let runtime_metadata: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/workflow-runtime-node-metadata-v1.json"
        )))
        .unwrap();
        let mut workflow_payload = typed_payload.clone();
        workflow_payload["metadata"] = runtime_metadata["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|case| {
                case["model_class"] == "workflow_run_node" && case["valid"] == true
            })
            .unwrap()["metadata"]
            .clone();
        let mut workflow_body = body.clone();
        workflow_body["mutations"][0]["payload"] = workflow_payload.clone();
        workflow_body["mutations"][0]["record"]["id"] = workflow_payload["id"].clone();
        workflow_body["mutations"][0]["model_envelope"] = json!({
            "schema_version": 1,
            "model_class": "workflow_run_node",
            "payload": workflow_payload,
        });
        let workflow_input: GraphMutationBatchInput =
            serde_json::from_value(workflow_body.clone()).unwrap();
        let workflow_mutations = graph_mutation_batch_value(workflow_input.clone()).unwrap();
        assert_eq!(
            workflow_mutations[0].record.metadata["_class_name"],
            "WorkflowRunNode"
        );
        assert!(workflow_mutations[0].payload["metadata"]
            .get("_class_name")
            .is_none());
        assert!(
            validate_graph_mutation_envelopes(&workflow_input, &workflow_mutations).is_ok()
        );

        workflow_body["mutations"][0]["record"]["metadata"]["_class_name"] = json!("Node");
        let conflicting_input: GraphMutationBatchInput =
            serde_json::from_value(workflow_body).unwrap();
        assert!(graph_mutation_batch_value(conflicting_input).is_err());

        let mut conflicting_payload = typed_body.clone();
        conflicting_payload["mutations"][0]["payload"]["metadata"]["_class_name"] =
            json!("Edge");
        let conflicting_payload: GraphMutationBatchInput =
            serde_json::from_value(conflicting_payload).unwrap();
        assert!(graph_mutation_batch_value(conflicting_payload).is_err());

        typed_body["mutations"][0]["model_envelope"]["model_class"] = json!("edge");
        let wrong_kind: GraphMutationBatchInput = serde_json::from_value(typed_body).unwrap();
        let wrong_kind_mutations = graph_mutation_batch_value(wrong_kind.clone()).unwrap();
        assert!(validate_graph_mutation_envelopes(&wrong_kind, &wrong_kind_mutations).is_err());

        let mut invalid_table = body.clone();
        invalid_table["mutations"][0]["table"] = json!("gke_edges");
        let parsed: GraphMutationBatchInput = serde_json::from_value(invalid_table).unwrap();
        assert!(graph_mutation_batch_value(parsed).is_err());

        let mut invalid_dimension = body.clone();
        invalid_dimension["mutations"][0]["embedding_dim"] = json!(3);
        let parsed: GraphMutationBatchInput = serde_json::from_value(invalid_dimension).unwrap();
        assert!(graph_mutation_batch_value(parsed).is_err());

        let mut identity_drift = body.clone();
        identity_drift["mutations"][0]["record"]["id"] = json!("different-node");
        let parsed: GraphMutationBatchInput = serde_json::from_value(identity_drift).unwrap();
        assert!(graph_mutation_batch_value(parsed).is_err());

        let mut physical_delete = body.clone();
        physical_delete["mutations"][0]["op"] = json!("DELETE");
        let parsed: GraphMutationBatchInput = serde_json::from_value(physical_delete).unwrap();
        assert!(parsed.validate().is_err());

        for (field, value) in [("atomic", json!(false)), ("contract_version", json!(2))] {
            let mut invalid = json!({
                "contract_version": 1,
                "atomic": true,
                "mutations": [],
            });
            invalid[field] = value;
            let parsed: GraphMutationBatchInput = serde_json::from_value(invalid).unwrap();
            assert!(graph_mutation_batch_value(parsed).is_err());
        }
    }

    #[test]
    fn v1_graph_mutation_preserves_arbitrary_integer_tokens_through_store_dto() {
        let mut body: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/graph-mutation-batch-v1.json"
        )))
        .unwrap();
        body["mutations"][0]["payload"]["level_from_root"] = json!(1);
        body["mutations"][0]["record"]["metadata"]["level_from_root"] = json!(1);
        let safe = serde_json::to_string(&body).unwrap();
        let huge = "1606938044258990275541962092341162602522202993782792835301376";
        let wire = safe.replace(
            "\"level_from_root\":1",
            &format!("\"level_from_root\":{huge}"),
        );
        let input: GraphMutationBatchInput = serde_json::from_str(&wire).unwrap();
        let mutations = graph_mutation_batch_value(input).unwrap();

        assert!(serde_json::to_string(&mutations[0].payload)
            .unwrap()
            .contains(huge));
        assert!(serde_json::to_string(&mutations[0].record.metadata)
            .unwrap()
            .contains(huge));
    }

    #[test]
    fn graph_mutation_index_intents_are_explicit_and_scope_checked() {
        let mut body: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/graph-mutation-batch-v1.json"
        )))
        .unwrap();
        body["index_jobs"] = json!([{
            "job_id": "index-node-1",
            "namespace": "tenant-a",
            "entity_kind": "node",
            "entity_id": "node-1",
            "index_kind": "node_embedding",
            "op": "UPSERT",
            "payload_json": "{\"text\":\"hello\"}",
            "max_retries": 4
        }]);
        let input: GraphMutationBatchInput = serde_json::from_value(body.clone()).unwrap();
        let mutations = graph_mutation_batch_value(input.clone()).unwrap();
        let jobs = graph_index_jobs_value(&input, &mutations).unwrap();
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].coalesce_key(), "node:node-1:node_embedding");
        assert_eq!(jobs[0].max_retries, 4);

        for (field, value) in [
            ("namespace", json!("other-tenant")),
            ("op", json!("REPLACE")),
            ("max_retries", json!(-1)),
        ] {
            let mut invalid = body.clone();
            invalid["index_jobs"][0][field] = value;
            let input: GraphMutationBatchInput = serde_json::from_value(invalid).unwrap();
            assert!(graph_mutation_batch_value(input).is_err(), "{field}");
        }

        let mut invalid = body.clone();
        invalid["index_jobs"][0]["max_retries"] = json!(i32::MAX as i64 + 1);
        let input: GraphMutationBatchInput = serde_json::from_value(invalid).unwrap();
        assert!(graph_mutation_batch_value(input).is_err());
    }
