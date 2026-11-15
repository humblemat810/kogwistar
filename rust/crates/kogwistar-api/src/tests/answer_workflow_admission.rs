    #[test]
    fn answer_workflow_admission_supports_per_node_python_and_registered_rust_executors() {
        let native_operations = crate::native_answer_persist::registry().unwrap();
        let contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/answer-workflow-runtime-contract-v1.json"
        ))
        .unwrap();
        for (workflow_id, workflow) in contract["workflows"].as_object().unwrap() {
            let required_ops = workflow["required_ops"]
                .as_array()
                .unwrap()
                .iter()
                .map(|op| op.as_str().unwrap().to_owned())
                .collect::<Vec<_>>();
            let python_ops = workflow["python_executor_ops"]
                .as_array()
                .unwrap()
                .iter()
                .map(|op| op.as_str().unwrap().to_owned())
                .collect::<std::collections::BTreeSet<_>>();
            let required_routes = workflow["required_routes"].as_array().unwrap();
            let python_op_version = workflow["python_op_version"].as_str().unwrap();
            let nodes = required_ops
                .iter()
                .map(|op| {
                    json!({
                        "id": format!("node-{op}"),
                        "label": op,
                        "metadata": {
                            "entity_type": "workflow_node",
                            "workflow_id": workflow_id,
                            "wf_op": op,
                            "wf_version": python_op_version,
                            "wf_start": op == "start",
                            "wf_fanout": false,
                            "wf_join": false,
                            "wf_executor_target": "python-worker"
                        }
                    })
                })
                .collect::<Vec<_>>();
            let edges = required_routes
                .iter()
                .enumerate()
                .map(|(index, route)| {
                    let relation = if route["predicate"].is_null() {
                        "wf_next"
                    } else {
                        "wf_conditional"
                    };
                    json!({
                        "id": format!("edge-{index}"),
                        "label": relation,
                        "source_ids": [format!("node-{}", route["from"].as_str().unwrap())],
                        "target_ids": [format!("node-{}", route["to"].as_str().unwrap())],
                        "metadata": {
                            "entity_type": "workflow_edge",
                            "workflow_id": workflow_id,
                            "wf_predicate": route["predicate"],
                            "wf_priority": route["priority"],
                            "wf_is_default": route["default"],
                            "wf_multiplicity": route["multiplicity"]
                        }
                    })
                })
                .collect::<Vec<_>>();
            let snapshot = json!({"nodes": nodes, "edges": edges});
            let plan = runtime_graph_plan_from_snapshot(&snapshot.to_string())
                .expect("shared Python answer design contract parses as a Rust runtime graph");
            assert!(answer_workflow_plan_is_executable(workflow_id, &plan, &native_operations));
            assert!(answer_workflow_admission_ready(
                workflow_id,
                Some(&plan),
                &native_operations
            ));
            assert!(!answer_workflow_admission_ready(
                workflow_id,
                None,
                &native_operations
            ));

            let mut missing_operation = plan.clone();
            missing_operation
                .node_ops
                .remove(&format!("node-{}", required_ops[1]));
            assert!(!answer_workflow_plan_is_executable(
                workflow_id,
                &missing_operation,
                &native_operations
            ));

            let mut altered_route_contract = plan.clone();
            if let Some(route) = altered_route_contract
                .routes
                .iter_mut()
                .find(|route| route.predicate.is_some())
            {
                route.predicate = Some("unknown_predicate".to_owned());
            } else {
                altered_route_contract.routes[0].is_default = false;
            }
            assert!(!answer_workflow_plan_is_executable(
                workflow_id,
                &altered_route_contract,
                &native_operations
            ));

            let mut wrong_route_target = plan.clone();
            wrong_route_target.routes[0].target_node_id = "node-start".to_owned();
            assert!(!answer_workflow_plan_is_executable(
                workflow_id,
                &wrong_route_target,
                &native_operations
            ));

            let mut unexpected_fanout = plan.clone();
            unexpected_fanout.routes[0].source_fanout = true;
            assert!(!answer_workflow_plan_is_executable(
                workflow_id,
                &unexpected_fanout,
                &native_operations
            ));

            let mut unexpected_join = plan.clone();
            unexpected_join.join_node_ids.push("node-start".to_owned());
            assert!(!answer_workflow_plan_is_executable(
                workflow_id,
                &unexpected_join,
                &native_operations
            ));

            let mut incompatible_operation_version = plan.clone();
            incompatible_operation_version
                .node_versions
                .insert("node-start".to_owned(), "v2".to_owned());
            assert!(!answer_workflow_plan_is_executable(
                workflow_id,
                &incompatible_operation_version,
                &native_operations
            ));

            let mut unknown_executor = plan.clone();
            let python_node = plan
                .node_ops
                .iter()
                .find(|(_, op)| python_ops.contains(*op))
                .unwrap()
                .0;
            unknown_executor
                .node_executor_targets
                .insert(python_node.clone(), "unknown-worker".to_owned());
            assert!(!answer_workflow_plan_is_executable(
                workflow_id,
                &unknown_executor,
                &native_operations
            ));

            if workflow_id == "agentic_answering.v2" {
                let native_node = plan
                    .node_ops
                    .iter()
                    .find(|(_, op)| op.as_str() == "aa_persist_response")
                    .unwrap()
                    .0;
                let mut mixed_executor_plan = plan.clone();
                mixed_executor_plan
                    .node_executor_targets
                    .insert(native_node.clone(), "rust-native".to_owned());
                assert!(answer_workflow_plan_is_executable(
                    workflow_id,
                    &mixed_executor_plan,
                    &native_operations
                ));

                let selector_node = plan
                    .node_ops
                    .iter()
                    .find(|(_, op)| op.as_str() == "aa_select_used_evidence")
                    .unwrap()
                    .0;
                let mut native_bm25_plan = plan.clone();
                native_bm25_plan
                    .node_executor_targets
                    .insert(selector_node.clone(), "rust-native".to_owned());
                assert!(answer_workflow_plan_is_executable(
                    workflow_id,
                    &native_bm25_plan,
                    &native_operations
                ));

                let mut rust_only_plan = plan.clone();
                for node_id in rust_only_plan.node_ops.keys().cloned().collect::<Vec<_>>() {
                    rust_only_plan
                        .node_executor_targets
                        .insert(node_id, "rust-native".to_owned());
                }
                assert!(answer_workflow_plan_is_executable(
                    workflow_id,
                    &rust_only_plan,
                    &native_operations
                ), "the complete answer workflow must support an all-Rust path");

                let mut native_answer_plan = mixed_executor_plan.clone();
                native_answer_plan.node_executor_targets.insert(
                    "node-aa_generate_answer_with_citations".to_owned(),
                    "rust-native".to_owned(),
                );
                assert!(answer_workflow_plan_is_executable(
                    workflow_id,
                    &native_answer_plan,
                    &native_operations
                ));

                let mut native_repair_plan = mixed_executor_plan.clone();
                native_repair_plan.node_executor_targets.insert(
                    "node-aa_validate_or_repair_citations".to_owned(),
                    "rust-native".to_owned(),
                );
                assert!(answer_workflow_plan_is_executable(
                    workflow_id,
                    &native_repair_plan,
                    &native_operations
                ));

                let mut unsupported_native_plan = mixed_executor_plan.clone();
                unsupported_native_plan.node_ops.insert(
                    "node-aa_validate_or_repair_citations".to_owned(),
                    "unregistered_citation_operation".to_owned(),
                );
                assert!(!answer_workflow_plan_is_executable(
                    workflow_id,
                    &unsupported_native_plan,
                    &native_operations
                ));
            }

            let mut implicit_noop = plan;
            implicit_noop
                .node_ops
                .insert("node-accidental".to_owned(), "noop".to_owned());
            assert!(!answer_workflow_plan_is_executable(
                workflow_id,
                &implicit_noop,
                &native_operations
            ));
        }
        assert!(!is_answer_workflow_id("ordinary.workflow"));
        assert!(answer_workflow_admission_ready(
            "ordinary.workflow",
            None,
            &native_operations
        ));
    }
