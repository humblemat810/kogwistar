
    fn transition(
        kind: RecordedTransitionKind,
        id: &str,
        expected_event_seq: i64,
        step_seq: i64,
    ) -> RecordedRuntimeTransition {
        RecordedRuntimeTransition {
            contract_version: RECORDED_RUNTIME_CONTRACT_VERSION,
            transition_id: id.to_owned(),
            expected_event_seq,
            kind,
            run_id: "run".to_owned(),
            workflow_id: "workflow".to_owned(),
            conversation_id: "conversation".to_owned(),
            user_id: Some("user".to_owned()),
            user_turn_node_id: Some("turn".to_owned()),
            step_seq,
            node_id: Some("node".to_owned()),
            token_id: Some("token".to_owned()),
            parent_token_id: None,
            initial_state: None,
            state_update: Vec::new(),
            update: None,
            state_schema: Map::new(),
            frontier: None,
            result: None,
            wait_reason: None,
            resume_payload: None,
            errors: Vec::new(),
        }
    }

    #[test]
    fn step_execute_payload_preserves_runtime_identity_and_resume_effect() {
        let mut start = transition(RecordedTransitionKind::Start, "start", 0, 0);
        let mut initial_state = Map::new();
        initial_state.insert("seed".to_owned(), json!(1));
        start.initial_state = Some(initial_state);
        start.frontier = Some(RuntimeFrontier {
            pending: vec![("node".to_owned(), 3, "token".to_owned(), None)],
            ..RuntimeFrontier::default()
        });
        let reduced = reduce_recorded_transition(None, &start).unwrap();
        let payload = RuntimeStepExecutePayload::from_recorded_state(
            &reduced.state,
            RuntimeStepExecuteRequest {
                node_id: "node".to_owned(),
                op: json!("op"),
                join_mask: 3,
                token_id: "token".to_owned(),
                parent_token_id: None,
                step_seq: 1,
                expected_event_seq: 9,
                resume_effect: Some(json!({"status": "success"})),
            },
        );
        assert_eq!(payload.turn_node_id.as_deref(), Some("turn"));
        assert_eq!(payload.expected_event_seq, 9);
        assert_eq!(payload.op_version, "v1");
        assert_eq!(payload.resume_effect.as_ref().unwrap()["status"], "success");
        assert!(
            serde_json::to_value(&payload)
                .unwrap()
                .get("op_version")
                .is_none()
        );
        assert_eq!(
            serde_json::from_value::<RuntimeStepExecutePayload>(
                serde_json::to_value(payload).unwrap()
            )
            .unwrap()
            .state["seed"],
            1
        );
    }

    #[test]
    fn native_noop_emits_the_shared_worker_effect_and_static_successor() {
        let effect = execute_native_step(&RuntimeNativeStepRequest {
            executor_target: RUST_NATIVE_EXECUTOR_TARGET.to_owned(),
            operation: "noop".to_owned(),
            op_version: "v1".to_owned(),
            run_id: "run".to_owned(),
            workflow_id: "workflow".to_owned(),
            node_id: "start".to_owned(),
            conversation_id: "conversation".to_owned(),
            turn_node_id: Some("turn".to_owned()),
            token_id: "token".to_owned(),
            parent_token_id: None,
            step_seq: 1,
            state: Map::from_iter([(String::from("seed"), json!(1))]),
            runtime_routes: vec![RuntimeStaticRoute {
                edge_id: "edge".to_owned(),
                source_node_id: "start".to_owned(),
                target_node_id: "finish".to_owned(),
                aliases: vec![],
                join_mask: 0,
                predicate: None,
                multiplicity: "one".to_owned(),
                is_default: true,
                priority: 100,
                source_fanout: false,
            }],
        })
        .unwrap();

        assert_eq!(effect.contract_version, RECORDED_RUNTIME_CONTRACT_VERSION);
        assert_eq!(effect.status, RuntimeWorkerEffectStatus::Success);
        assert_eq!(effect.successors[0].node_id, "finish");
        assert!(effect.state_update.is_empty());
        assert_eq!(
            effect.result,
            Some(json!({
                "workflow_status": "succeeded",
                "final_state": {"seed": 1},
            }))
        );
        assert_eq!(
            effect.trace_events[0]["event"],
            "runtime.native.step.completed"
        );
        assert_eq!(effect.trace_events[0]["schema_version"], 1);
        assert_eq!(effect.trace_events[0]["op_version"], "v1");
        assert_eq!(
            effect.trace_events[0]["implementation_authority"],
            RUST_NATIVE_EXECUTOR_TARGET
        );
        assert_eq!(
            effect.trace_events[0]["seam_attestation_version"],
            RUST_NATIVE_SEAM_ATTESTATION_VERSION
        );
        assert_eq!(effect.trace_events[0]["seam_crossings"], json!([]));

        let fixture: Value = serde_json::from_str(include_str!(
            "../../../../contracts/golden/runtime-native-noop-v1.json"
        ))
        .unwrap();
        let durable = json!({
            "contract_version": effect.contract_version,
            "status": "success",
            "state_update": effect.state_update,
            "update": effect.update,
            "state_schema": effect.state_schema,
            "successors": effect.successors,
            "route_next": effect.route_next,
            "result": effect.result,
        });
        assert_eq!(durable, fixture);
    }

    #[test]
    fn native_executor_fails_closed_for_wrong_target_and_unregistered_operations() {
        let request = RuntimeNativeStepRequest {
            executor_target: "python-worker".to_owned(),
            operation: "noop".to_owned(),
            op_version: "v1".to_owned(),
            run_id: "run".to_owned(),
            workflow_id: "workflow".to_owned(),
            node_id: "node".to_owned(),
            conversation_id: "conversation".to_owned(),
            turn_node_id: None,
            token_id: "token".to_owned(),
            parent_token_id: None,
            step_seq: 0,
            state: Map::new(),
            runtime_routes: vec![],
        };
        assert!(matches!(
            execute_native_step(&request),
            Err(RuntimeNativeExecutionError::WrongExecutorTarget { .. })
        ));
        let emit_result = RuntimeNativeStepRequest {
            executor_target: RUST_NATIVE_EXECUTOR_TARGET.to_owned(),
            operation: "emit-result".to_owned(),
            ..request.clone()
        };
        let effect = execute_native_step(&emit_result).unwrap();
        assert_eq!(
            effect.result,
            Some(json!({
                "executor": RUST_NATIVE_EXECUTOR_TARGET,
                "operation": "emit-result",
                "schema_version": RUST_NATIVE_OPERATION_SCHEMA_VERSION,
            }))
        );
        let increment = RuntimeNativeStepRequest {
            executor_target: RUST_NATIVE_EXECUTOR_TARGET.to_owned(),
            operation: "increment-counter".to_owned(),
            state: Map::from_iter([(String::from("counter"), json!(1))]),
            ..request.clone()
        };
        let increment_effect = execute_native_step(&increment).unwrap();
        assert_eq!(
            serde_json::to_value(&increment_effect.state_update).unwrap(),
            json!([["u", {"counter": 2}]])
        );
        assert_eq!(
            increment_effect.result,
            Some(json!({
                "workflow_status": "succeeded",
                "final_state": {"counter": 2},
            }))
        );
        let invalid_increment = RuntimeNativeStepRequest {
            state: Map::new(),
            ..increment
        };
        assert!(matches!(
            execute_native_step(&invalid_increment),
            Err(RuntimeNativeExecutionError::InvalidState {
                field: "counter",
                ..
            })
        ));
        let unsupported = RuntimeNativeStepRequest {
            executor_target: RUST_NATIVE_EXECUTOR_TARGET.to_owned(),
            operation: "arbitrary-callback".to_owned(),
            ..request
        };
        assert!(matches!(
            execute_native_step(&unsupported),
            Err(RuntimeNativeExecutionError::UnsupportedOperation(operation))
                if operation == "arbitrary-callback"
        ));
        let unsupported_version = RuntimeNativeStepRequest {
            executor_target: RUST_NATIVE_EXECUTOR_TARGET.to_owned(),
            operation: "noop".to_owned(),
            op_version: "v2".to_owned(),
            ..emit_result
        };
        assert!(matches!(
            execute_native_step(&unsupported_version),
            Err(RuntimeNativeExecutionError::UnsupportedOperationVersion {
                operation,
                version
            }) if operation == "noop" && version == "v2"
        ));
    }

    #[test]
    fn native_operation_recognition_is_strict_and_shared() {
        assert_eq!(native_operation_name("rust-native/noop"), Some("noop"));
        assert_eq!(native_operation_name("rust-native/"), None);
        assert_eq!(native_operation_name("rust-native"), None);
        assert_eq!(native_operation_name("python/noop"), None);
        assert_eq!(
            native_operation_for_target("increment-counter", Some(RUST_NATIVE_EXECUTOR_TARGET)),
            Some("increment-counter")
        );
        assert_eq!(
            native_operation_for_target("rust-native/", Some(RUST_NATIVE_EXECUTOR_TARGET)),
            None
        );
        assert_eq!(
            native_operation_for_target("rust-native/increment-counter", None),
            Some("increment-counter")
        );
        assert_eq!(
            native_operation_for_target("increment-counter", Some(PYTHON_EXECUTOR_TARGET)),
            None
        );
    }

    #[test]
    fn native_executor_routing_matches_shared_python_contract_fixture() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/runtime-executor-routing-v1.json"
        )))
        .unwrap();
        assert_eq!(fixture["schema_version"], 1);
        for case in fixture["cases"].as_array().unwrap() {
            let operation = case["operation"].as_str().unwrap();
            let target = case["target"].as_str();
            let expected = case["native_operation"].as_str();
            assert_eq!(
                native_operation_for_target(operation, target),
                expected,
                "case {}",
                case["name"].as_str().unwrap()
            );
        }
    }

    #[test]
    fn native_registry_runs_versioned_application_operation_and_preserves_builtins() {
        let mut registry = NativeOperationRegistry::default();
        let append_fixture: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/runtime-native-app-append-v1.json"
        )))
        .unwrap();
        registry
            .register("app/append-item", "v1", |request| {
                let mut items = request
                    .state
                    .get("items")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                items.push(json!("appended"));
                Ok(NativeOperationOutput {
                    state_update: vec![RecordedStateUpdate::Pair((
                        "a".to_owned(),
                        Map::from_iter([("items".to_owned(), json!("appended"))]),
                    ))],
                    result: Some(json!({
                        "workflow_status": "succeeded",
                        "final_state": {"items": items},
                    })),
                })
            })
            .unwrap();
        registry
            .register("app/label", "v2", |request| {
                Ok(NativeOperationOutput {
                    state_update: Vec::new(),
                    result: Some(json!({
                        "label": request.state.get("label").cloned().unwrap_or(Value::Null),
                        "operation_version": request.op_version,
                    })),
                })
            })
            .unwrap();

        let append_request = RuntimeNativeStepRequest {
            executor_target: RUST_NATIVE_EXECUTOR_TARGET.to_owned(),
            operation: "app/append-item".to_owned(),
            op_version: "v1".to_owned(),
            run_id: "run".to_owned(),
            workflow_id: "workflow".to_owned(),
            node_id: "append".to_owned(),
            conversation_id: "conversation".to_owned(),
            turn_node_id: None,
            token_id: "token".to_owned(),
            parent_token_id: None,
            step_seq: 1,
            state: serde_json::from_value(append_fixture["initial_state"].clone()).unwrap(),
            runtime_routes: Vec::new(),
        };
        let append_effect = execute_native_step_with_registry(&append_request, &registry).unwrap();
        assert_eq!(
            serde_json::json!({
                "status": append_effect.status,
                "state_update": append_effect.state_update,
                "state_schema": append_effect.state_schema,
                "successors": append_effect.successors,
                "route_next": append_effect.route_next,
                "result": append_effect.result,
            }),
            append_fixture["effect"]
        );

        let request = RuntimeNativeStepRequest {
            executor_target: RUST_NATIVE_EXECUTOR_TARGET.to_owned(),
            operation: "app/label".to_owned(),
            op_version: "v2".to_owned(),
            run_id: "run".to_owned(),
            workflow_id: "workflow".to_owned(),
            node_id: "label".to_owned(),
            conversation_id: "conversation".to_owned(),
            turn_node_id: None,
            token_id: "token".to_owned(),
            parent_token_id: None,
            step_seq: 1,
            state: Map::from_iter([(String::from("label"), json!("native"))]),
            runtime_routes: Vec::new(),
        };
        let effect = execute_native_step_with_registry(&request, &registry).unwrap();
        assert_eq!(
            effect.result,
            Some(json!({"label": "native", "operation_version": "v2"}))
        );
        assert_eq!(effect.trace_events[0]["op_version"], "v2");
        assert_eq!(effect.trace_events[0]["seam_crossings"], json!([]));

        let unregistered_version = RuntimeNativeStepRequest {
            op_version: "v1".to_owned(),
            ..request.clone()
        };
        assert!(matches!(
            execute_native_step_with_registry(&unregistered_version, &registry),
            Err(RuntimeNativeExecutionError::UnsupportedOperationVersion {
                operation,
                version
            }) if operation == "app/label" && version == "v1"
        ));

        let builtin = RuntimeNativeStepRequest {
            operation: "noop".to_owned(),
            op_version: "v1".to_owned(),
            ..request.clone()
        };
        assert!(execute_native_step_with_registry(&builtin, &registry).is_ok());
        assert!(matches!(
            registry.register("app/label", "v2", |_| Ok(NativeOperationOutput {
                state_update: Vec::new(),
                result: None,
            })),
            Err(NativeOperationRegistrationError::Duplicate { .. })
        ));
        assert!(matches!(
            registry.register(" rust-native/unsafe", "v1", |_| Ok(NativeOperationOutput {
                state_update: Vec::new(),
                result: None,
            })),
            Err(NativeOperationRegistrationError::InvalidIdentity)
        ));
    }

    #[test]
    fn native_request_legacy_wire_defaults_state_without_changing_routing() {
        let request: RuntimeNativeStepRequest = serde_json::from_value(json!({
            "executor_target": RUST_NATIVE_EXECUTOR_TARGET,
            "operation": "noop",
            "run_id": "run",
            "workflow_id": "workflow",
            "node_id": "start",
            "conversation_id": "conversation",
            "token_id": "token",
            "step_seq": 0,
            "runtime_routes": []
        }))
        .unwrap();
        assert!(request.state.is_empty());
        assert_eq!(request.op_version, "v1");
        let effect = execute_native_step(&request).unwrap();
        assert!(effect.successors.is_empty());
        assert_eq!(
            effect.result,
            Some(json!({
                "workflow_status": "succeeded",
                "final_state": {},
            }))
        );
    }

    #[test]
    fn step_execute_payload_carries_non_default_operation_version() {
        let mut start = transition(RecordedTransitionKind::Start, "start", 0, 0);
        let mut initial_state = Map::new();
        initial_state.insert("_rt_node_versions".to_owned(), json!({"node": "v2"}));
        start.initial_state = Some(initial_state);
        start.frontier = Some(RuntimeFrontier {
            pending: vec![("node".to_owned(), 0, "token".to_owned(), None)],
            ..RuntimeFrontier::default()
        });
        let reduced = reduce_recorded_transition(None, &start).unwrap();
        let payload = RuntimeStepExecutePayload::from_recorded_state(
            &reduced.state,
            RuntimeStepExecuteRequest {
                node_id: "node".to_owned(),
                op: json!("process"),
                join_mask: 0,
                token_id: "token".to_owned(),
                parent_token_id: None,
                step_seq: 1,
                expected_event_seq: 2,
                resume_effect: None,
            },
        );

        assert_eq!(payload.op_version, "v2");
        assert_eq!(serde_json::to_value(payload).unwrap()["op_version"], "v2");
    }

    #[test]
    fn step_execute_payload_carries_required_worker_environment() {
        let mut start = transition(RecordedTransitionKind::Start, "start", 0, 0);
        let mut initial_state = Map::new();
        initial_state.insert(
            "_rt_node_environment_ids".to_owned(),
            json!({"node": "numpy-cpu-2"}),
        );
        initial_state.insert(
            "_rt_node_executor_targets".to_owned(),
            json!({"node": RUST_NATIVE_EXECUTOR_TARGET}),
        );
        start.initial_state = Some(initial_state);
        start.frontier = Some(RuntimeFrontier {
            pending: vec![("node".to_owned(), 0, "token".to_owned(), None)],
            ..RuntimeFrontier::default()
        });
        let reduced = reduce_recorded_transition(None, &start).unwrap();
        let payload = RuntimeStepExecutePayload::from_recorded_state(
            &reduced.state,
            RuntimeStepExecuteRequest {
                node_id: "node".to_owned(),
                op: json!("numerical"),
                join_mask: 0,
                token_id: "token".to_owned(),
                parent_token_id: None,
                step_seq: 1,
                expected_event_seq: 2,
                resume_effect: None,
            },
        );
        assert_eq!(payload.op_environment_id.as_deref(), Some("numpy-cpu-2"));
        assert_eq!(
            payload.executor_target.as_deref(),
            Some(RUST_NATIVE_EXECUTOR_TARGET)
        );
        let serialized = serde_json::to_value(payload).unwrap();
        assert_eq!(serialized["op_environment_id"], "numpy-cpu-2");
        assert_eq!(serialized["executor_target"], RUST_NATIVE_EXECUTOR_TARGET);
    }

    #[test]
    fn public_runtime_json_helpers_reject_unknown_root_and_nested_dto_fields() {
        let cases = [
            (
                r#"{"edges":[],"explicit_next":[],"fanout":false,"failure_only":false,"unexpected":true}"#,
                select_runtime_route_from_str as fn(&str) -> Result<String, serde_json::Error>,
            ),
            (
                r#"{"token_id":"token","step_seq":1,"current_join_mask":0,"join_outstanding":[],"successors":[],"unexpected":true}"#,
                plan_runtime_successors_from_str,
            ),
            (
                r#"{"join_index":0,"join_outstanding":[],"arrival":{"join_mask":0,"token_id":"token"},"unexpected":true}"#,
                apply_runtime_join_arrival_from_str,
            ),
            (
                r#"{"retry_budget":1,"attempt_number":0,"unexpected":true}"#,
                decide_runtime_retry_from_str,
            ),
            (
                r#"{"parent_run_id":"parent","workflow_id":"workflow","parent_conversation_id":"conversation","parent_turn_node_id":"turn","unexpected":true}"#,
                plan_runtime_nested_invocation_from_str,
            ),
            (
                r#"{"max_workers":1,"inflight":0,"pending":0,"unexpected":true}"#,
                decide_runtime_dispatch_from_str,
            ),
            (
                r#"{"token_budget":0,"token_used":0,"time_budget_ms":0,"time_used_ms":0,"rate_limit":0,"rate_used":0,"step_budget":0,"step_used":0,"call_budget":0,"call_used":0,"cost_budget":0.0,"cost_used":0.0,"unexpected":true}"#,
                decide_runtime_budget_suspend_from_str,
            ),
            (
                r#"{"pending":[],"inflight":0,"max_workers":1,"unexpected":true}"#,
                tick_runtime_scheduler_from_str,
            ),
        ];
        for (payload, helper) in cases {
            let error = helper(payload).unwrap_err();
            assert!(error.to_string().contains("unknown field"), "{error}");
        }

        let error = select_runtime_route_from_str(
            r#"{"edges":[{"edge_id":"edge","unexpected":true}],"explicit_next":[]}"#,
        )
        .unwrap_err();
        assert!(error.to_string().contains("unknown field"), "{error}");
    }

    fn route_edge(
        edge_id: &str,
        target: &str,
        predicate: Option<&str>,
        result: Option<bool>,
        priority: i64,
    ) -> RuntimeRouteEdge {
        RuntimeRouteEdge {
            edge_id: edge_id.to_owned(),
            target_ids: vec![target.to_owned()],
            aliases: vec![
                target.to_owned(),
                target.split('|').next_back().unwrap().to_owned(),
            ],
            predicate: predicate.map(ToOwned::to_owned),
            multiplicity: "one".to_owned(),
            is_default: false,
            priority,
            predicate_result: result,
            base_result: result,
        }
    }

    #[test]
    fn route_selection_preserves_explicit_predicate_fanout_failure_and_default_rules() {
        let edges = vec![
            route_edge("low", "wf|low", Some("p"), Some(true), 1),
            route_edge("high", "wf|high", Some("p"), Some(true), 9),
        ];
        let explicit = select_runtime_route(&RuntimeRouteRequest {
            edges: edges.clone(),
            explicit_next: vec!["low".to_owned()],
            fanout: false,
            failure_only: false,
        });
        assert_eq!(explicit.next_node_ids, vec!["wf|low"]);
        assert_eq!(explicit.selected_edge_indices, vec![0]);

        let priority = select_runtime_route(&RuntimeRouteRequest {
            edges: edges.clone(),
            explicit_next: vec![],
            fanout: false,
            failure_only: false,
        });
        assert_eq!(priority.next_node_ids, vec!["wf|high"]);
        assert_eq!(priority.selected_edge_indices, vec![1]);

        let fanout = select_runtime_route(&RuntimeRouteRequest {
            edges,
            explicit_next: vec![],
            fanout: true,
            failure_only: false,
        });
        assert_eq!(fanout.next_node_ids, vec!["wf|high", "wf|low"]);
        assert_eq!(fanout.selected_edge_indices, vec![1, 0]);

        let failure = select_runtime_route(&RuntimeRouteRequest {
            edges: vec![route_edge("base", "base", None, Some(true), 1)],
            explicit_next: vec![],
            fanout: false,
            failure_only: true,
        });
        assert!(failure.next_node_ids.is_empty());

        let mut default = route_edge("default", "a", None, Some(false), 1);
        default.target_ids.push("b".to_owned());
        default.is_default = true;
        let default = select_runtime_route(&RuntimeRouteRequest {
            edges: vec![default],
            explicit_next: vec![],
            fanout: true,
            failure_only: false,
        });
        assert_eq!(default.next_node_ids, vec!["a", "b"]);
    }

    #[test]
    fn authoritative_successors_delegate_only_predicate_choice_with_graph_validation() {
        let route = |target: &str, predicate: Option<&str>, multiplicity: &str, fanout: bool| {
            RuntimeStaticRoute {
                edge_id: format!("edge-{target}"),
                source_node_id: "gate".to_owned(),
                target_node_id: target.to_owned(),
                aliases: vec![target.to_owned()],
                join_mask: if target == "join" { 1 } else { 0 },
                predicate: predicate.map(str::to_owned),
                multiplicity: multiplicity.to_owned(),
                is_default: false,
                priority: 100,
                source_fanout: fanout,
            }
        };
        let static_routes = vec![
            route("left", None, "many", true),
            route("right", None, "many", true),
        ];
        assert_eq!(
            authoritative_runtime_successors(
                &static_routes,
                "gate",
                &[RuntimeSuccessor {
                    node_id: "forged".to_owned(),
                    join_mask: 0,
                }],
            )
            .unwrap()
            .into_iter()
            .map(|successor| successor.node_id)
            .collect::<Vec<_>>(),
            vec!["left", "right"]
        );

        let predicate_routes = vec![
            route("left", Some("if_true"), "one", false),
            route("join", Some("if_false"), "one", false),
        ];
        let selected = vec![RuntimeSuccessor {
            node_id: "join".to_owned(),
            join_mask: 1,
        }];
        assert_eq!(
            authoritative_runtime_successors(&predicate_routes, "gate", &selected).unwrap()[0]
                .node_id,
            "join"
        );
        assert!(
            authoritative_runtime_successors(
                &predicate_routes,
                "gate",
                &[RuntimeSuccessor {
                    node_id: "join".to_owned(),
                    join_mask: 0,
                }],
            )
            .is_err()
        );
        assert!(
            authoritative_runtime_successors(
                &predicate_routes,
                "gate",
                &[
                    RuntimeSuccessor {
                        node_id: "left".to_owned(),
                        join_mask: 0,
                    },
                    selected[0].clone(),
                ],
            )
            .is_err()
        );
    }

    #[test]
    fn authoritative_successors_preserve_python_static_explicit_and_failure_routing() {
        let route = |edge: &str,
                     target: &str,
                     aliases: &[&str],
                     predicate: Option<&str>,
                     priority: i64,
                     is_default: bool| RuntimeStaticRoute {
            edge_id: edge.to_owned(),
            source_node_id: "gate".to_owned(),
            target_node_id: target.to_owned(),
            aliases: aliases.iter().map(|value| (*value).to_owned()).collect(),
            join_mask: if target == "join" { 1 } else { 0 },
            predicate: predicate.map(str::to_owned),
            multiplicity: "one".to_owned(),
            is_default,
            priority,
            source_fanout: false,
        };

        let static_routes = vec![
            route("first", "left", &["left", "choose-left"], None, 10, false),
            route(
                "second",
                "right",
                &["right", "choose-right"],
                None,
                20,
                true,
            ),
        ];
        assert_eq!(
            authoritative_runtime_successors_for_result(&static_routes, "gate", &[], &[], false,)
                .unwrap()
                .into_iter()
                .map(|successor| successor.node_id)
                .collect::<Vec<_>>(),
            vec!["left"]
        );
        assert_eq!(
            authoritative_runtime_successors_for_result(
                &static_routes,
                "gate",
                &[],
                &["choose-right".to_owned()],
                false,
            )
            .unwrap()[0]
                .node_id,
            "right"
        );
        assert!(
            authoritative_runtime_successors_for_result(&static_routes, "gate", &[], &[], true,)
                .unwrap()
                .is_empty()
        );

        let failure_routes = vec![
            route(
                "recover",
                "join",
                &["recover"],
                Some("on_failure"),
                10,
                false,
            ),
            route("fallback", "right", &["right"], None, 20, true),
        ];
        let handled = authoritative_runtime_successors_for_result(
            &failure_routes,
            "gate",
            &[RuntimeSuccessor {
                node_id: "join".to_owned(),
                join_mask: 1,
            }],
            &[],
            true,
        )
        .unwrap();
        assert_eq!(handled[0].node_id, "join");
        assert_eq!(handled[0].join_mask, 1);
    }

    #[test]
    fn successor_plan_preserves_continuation_spawns_children_and_updates_join_counts() {
        let plan = plan_runtime_successors(&RuntimeSuccessorPlanRequest {
            token_id: "parent".to_owned(),
            parent_token_id: Some("root".to_owned()),
            step_seq: 7,
            current_join_mask: 0b0011,
            join_outstanding: vec![1, 1, 0],
            successors: vec![
                RuntimeSuccessor {
                    node_id: "left".to_owned(),
                    join_mask: 0b0101,
                },
                RuntimeSuccessor {
                    node_id: "right".to_owned(),
                    join_mask: 0b0110,
                },
            ],
        });
        assert_eq!(plan.join_outstanding, vec![1, 1, 2]);
        assert_eq!(plan.tokens[0].token_id, "parent");
        assert_eq!(plan.tokens[0].parent_token_id.as_deref(), Some("root"));
        assert!(!plan.tokens[0].spawned);
        assert_eq!(plan.tokens[1].parent_token_id.as_deref(), Some("parent"));
        assert!(plan.tokens[1].spawned);
        assert_eq!(
            plan.tokens[1].token_id,
            kogwistar_contracts::stable_id("token_id", &["parent/7:1:right".to_owned()])
                .simple()
                .to_string()
        );
    }

    #[test]
    fn join_arrival_waits_then_merges_masks_and_obligations() {
        let first = apply_runtime_join_arrival(&RuntimeJoinArrivalRequest {
            join_index: 0,
            join_outstanding: vec![2, 2],
            waiters: vec![],
            arrival: RuntimeJoinWaiter {
                join_mask: 0b11,
                token_id: "one".to_owned(),
                parent_token_id: None,
            },
            merge: true,
        });
        assert_eq!(first.outstanding, 1);
        assert!(first.released.is_none());
        assert_eq!(first.waiters[0].join_mask, 0b10);

        let second = apply_runtime_join_arrival(&RuntimeJoinArrivalRequest {
            join_index: 0,
            join_outstanding: first.join_outstanding,
            waiters: first.waiters,
            arrival: RuntimeJoinWaiter {
                join_mask: 0b11,
                token_id: "two".to_owned(),
                parent_token_id: Some("one".to_owned()),
            },
            merge: true,
        });
        assert_eq!(second.outstanding, 0);
        assert_eq!(second.collapsed_work_items, 1);
        assert!(second.waiters.is_empty());
        assert_eq!(second.released.unwrap().join_mask, 0b10);
        assert_eq!(second.join_outstanding, vec![0, 1]);
    }

    #[test]
    fn retry_decision_normalizes_budget_and_marks_exhaustion() {
        assert_eq!(
            decide_runtime_retry(&RuntimeRetryDecisionRequest {
                retry_budget: 0,
                attempt_number: 1,
            }),
            RuntimeRetryDecision {
                retry_budget: 1,
                attempt_number: 1,
                should_retry: false,
                exhausted: true,
                next_attempt_number: None,
            }
        );
        assert_eq!(
            decide_runtime_retry(&RuntimeRetryDecisionRequest {
                retry_budget: 3,
                attempt_number: 2,
            })
            .next_attempt_number,
            Some(3)
        );
    }

    #[test]
    fn nested_invocation_plan_is_deterministic_and_honors_overrides() {
        let request = RuntimeNestedInvocationPlanRequest {
            parent_run_id: "parent".to_owned(),
            workflow_id: "child".to_owned(),
            result_state_key: None,
            run_id: None,
            invocation_key: None,
            parent_conversation_id: "conversation".to_owned(),
            conversation_id: Some("child-conversation".to_owned()),
            parent_turn_node_id: "turn".to_owned(),
            turn_node_id: None,
        };
        let first = plan_runtime_nested_invocation(&request);
        assert_eq!(first, plan_runtime_nested_invocation(&request));
        assert_eq!(first.conversation_id, "child-conversation");
        assert_eq!(first.turn_node_id, "turn");
        assert_eq!(first.result_state_key, "workflow_result::child");
        let keyed = plan_runtime_nested_invocation(&RuntimeNestedInvocationPlanRequest {
            invocation_key: Some("loop:2:act:1".to_owned()),
            ..request
        });
        assert_ne!(first.child_run_id, keyed.child_run_id);
    }

    #[test]
    fn dispatch_decision_bounds_launches_and_stops_on_cancel() {
        assert_eq!(
            decide_runtime_dispatch(&RuntimeDispatchDecisionRequest {
                max_workers: 4,
                inflight: 3,
                pending: 7,
                cancelling: false,
            })
            .launch_capacity,
            1
        );
        let cancelling = decide_runtime_dispatch(&RuntimeDispatchDecisionRequest {
            max_workers: 0,
            inflight: 1,
            pending: 0,
            cancelling: true,
        });
        assert_eq!(cancelling.worker_limit, 1);
        assert_eq!(cancelling.launch_capacity, 0);
        assert!(cancelling.should_drain);
        assert!(!cancelling.cancellation_complete);
    }

    #[test]
    fn budget_suspend_decision_includes_cost_limit() {
        let decision = decide_runtime_budget_suspend(&RuntimeBudgetSuspendRequest {
            token_budget: 0,
            token_used: 0,
            time_budget_ms: 0,
            time_used_ms: 0,
            rate_limit: 0,
            rate_used: 0,
            step_budget: 0,
            step_used: 0,
            call_budget: 0,
            call_used: 0,
            cost_budget: 2.5,
            cost_used: 2.5,
        });
        assert!(decision.should_suspend);
        assert_eq!(decision.reason.as_deref(), Some("cost"));
    }

    #[test]
    fn scheduler_tick_owns_fifo_dispatch_and_cancel_stop() {
        let token = |node: &str| RuntimeSchedulerToken {
            node_id: node.to_owned(),
            join_mask: 0,
            token_id: format!("token-{node}"),
            parent_token_id: None,
        };
        let tick = tick_runtime_scheduler(&RuntimeSchedulerTickRequest {
            pending: vec![token("a"), token("b"), token("c")],
            inflight: 1,
            max_workers: 3,
            cancelling: false,
        });
        assert_eq!(tick.dispatch, vec![token("a"), token("b")]);
        assert_eq!(tick.pending, vec![token("c")]);

        let cancelled = tick_runtime_scheduler(&RuntimeSchedulerTickRequest {
            pending: vec![token("a")],
            inflight: 1,
            max_workers: 3,
            cancelling: true,
        });
        assert!(cancelled.dispatch.is_empty());
        assert_eq!(cancelled.pending, vec![token("a")]);
        assert!(cancelled.should_drain);
    }

    #[test]
    fn reducer_canonicalizes_frontier_and_drops_dependencies() {
        let mut start = transition(RecordedTransitionKind::Start, "start", 0, 0);
        start.initial_state = Some(Map::from_iter([
            ("_deps".to_owned(), json!({"not": "durable"})),
            ("value".to_owned(), json!(1)),
        ]));
        start.frontier = Some(RuntimeFrontier {
            pending: vec![
                ("z".to_owned(), 0, "z-token".to_owned(), None),
                ("a".to_owned(), 0, "a-token".to_owned(), None),
            ],
            ..RuntimeFrontier::default()
        });
        let started = reduce_recorded_transition(None, &start).unwrap();
        assert_eq!(started.state.last_step_seq, -1);
        assert_eq!(started.state.frontier.pending[0].0, "a");
        assert!(!started.state.state.contains_key("_deps"));
        assert_eq!(started.checkpoint["node_id"], "wf_ckpt|run|0");
        assert_eq!(started.run_status["status"], "running");
    }

    #[test]
    fn terminal_transition_may_share_last_accepted_step_only() {
        let mut start = transition(RecordedTransitionKind::Start, "start", 0, 0);
        start.frontier = Some(RuntimeFrontier {
            pending: vec![("node".to_owned(), 0, "token".to_owned(), None)],
            ..RuntimeFrontier::default()
        });
        let started = reduce_recorded_transition(None, &start).unwrap();
        let step = transition(RecordedTransitionKind::RecordedStepSuccess, "step", 1, 0);
        let stepped = reduce_recorded_transition(Some(&started.state), &step).unwrap();
        let complete = transition(RecordedTransitionKind::Complete, "complete", 2, 0);
        let completed = reduce_recorded_transition(Some(&stepped.state), &complete).unwrap();
        assert_eq!(completed.state.status, RecordedRunStatus::Completed);
        assert_eq!(completed.server_status, "succeeded");
        assert_eq!(completed.run_status["status"], "completed");
        assert!(reduce_recorded_transition(Some(&completed.state), &complete).is_err());
    }

    #[test]
    fn empty_legacy_update_matches_python_state_update_truthiness() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/runtime-state-update-v1.json"
        )))
        .unwrap();
        let case = fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|case| {
                case["name"] == "empty_legacy_update_does_not_conflict_with_state_delta"
            })
            .unwrap();
        let initial_state: Map<String, serde_json::Value> =
            serde_json::from_value(case["state"].clone()).unwrap();
        let mut start = transition(RecordedTransitionKind::Start, "empty-update-start", 0, 0);
        start.initial_state = Some(initial_state);
        start.frontier = Some(RuntimeFrontier {
            pending: vec![("node".to_owned(), 0, "token".to_owned(), None)],
            ..RuntimeFrontier::default()
        });
        let started = reduce_recorded_transition(None, &start).unwrap();

        let mut step = transition(
            RecordedTransitionKind::RecordedStepSuccess,
            "empty-update-step",
            1,
            0,
        );
        step.state_update = serde_json::from_value(case["state_update"].clone()).unwrap();
        step.update = serde_json::from_value(case["update"].clone()).unwrap();
        step.state_schema = serde_json::from_value(case["state_schema"].clone()).unwrap();
        let stepped = reduce_recorded_transition(Some(&started.state), &step).unwrap();

        assert_eq!(
            serde_json::Value::Object(stepped.state.state)["items"],
            case["expected"]["items"]
        );

        step.update = Some(Map::from_iter([("status".to_owned(), json!("done"))]));
        assert!(matches!(
            reduce_recorded_transition(Some(&started.state), &step),
            Err(RecordedRuntimeError::StateUpdateConflict)
        ));
    }

    #[test]
    fn terminal_transition_rejects_unfinished_join_work() {
        let mut start = transition(RecordedTransitionKind::Start, "start", 0, 0);
        start.frontier = Some(RuntimeFrontier {
            pending: vec![("node".to_owned(), 1, "token".to_owned(), None)],
            join_node_ids: vec!["join".to_owned()],
            join_outstanding: vec![1],
            join_waiters: BTreeMap::from_iter([(
                "join".to_owned(),
                vec![(1, "waiter".to_owned(), None)],
            )]),
            ..RuntimeFrontier::default()
        });
        let started = reduce_recorded_transition(None, &start).unwrap();
        let mut complete = transition(RecordedTransitionKind::Complete, "complete", 1, 0);
        complete.frontier = Some(RuntimeFrontier {
            join_node_ids: vec!["join".to_owned()],
            join_outstanding: vec![1],
            join_waiters: BTreeMap::from_iter([(
                "join".to_owned(),
                vec![(1, "waiter".to_owned(), None)],
            )]),
            ..RuntimeFrontier::default()
        });

        assert!(matches!(
            reduce_recorded_transition(Some(&started.state), &complete),
            Err(RecordedRuntimeError::TerminalFrontierNotEmpty)
        ));
    }

    #[test]
    fn worker_success_effect_preserves_authoritative_frontier_and_plans_children() {
        let frontier = RuntimeFrontier {
            pending: vec![
                ("start".to_owned(), 1, "token".to_owned(), None),
                ("other".to_owned(), 0, "other-token".to_owned(), None),
            ],
            suspended: vec![("sleep".to_owned(), 0, "sleep-token".to_owned(), None)],
            join_node_ids: vec!["join".to_owned()],
            join_outstanding: vec![1],
            join_waiters: BTreeMap::from_iter([("join".to_owned(), Vec::new())]),
        };
        let next = frontier_after_worker_success(
            &frontier,
            "start",
            "token",
            None,
            4,
            &RuntimeWorkerSuccessEffect {
                successors: vec![
                    RuntimeSuccessor {
                        node_id: "left".to_owned(),
                        join_mask: 1,
                    },
                    RuntimeSuccessor {
                        node_id: "right".to_owned(),
                        join_mask: 1,
                    },
                ],
            },
        )
        .unwrap();

        assert!(next.pending.iter().any(|token| token.0 == "other"));
        assert!(
            next.pending
                .iter()
                .any(|token| token.0 == "left" && token.2 == "token")
        );
        let right = next
            .pending
            .iter()
            .find(|token| token.0 == "right")
            .unwrap();
        assert_eq!(right.3.as_deref(), Some("token"));
        assert_ne!(right.2, "token");
        assert_eq!(next.suspended, frontier.suspended);
        assert_eq!(next.join_outstanding, vec![2]);
        assert!(matches!(
            frontier_after_worker_success(
                &frontier,
                "forged",
                "token",
                None,
                4,
                &RuntimeWorkerSuccessEffect { successors: vec![] },
            ),
            Err(RecordedRuntimeError::TokenNotInFrontier)
        ));
    }

    #[test]
    fn worker_success_effect_holds_join_arrivals_until_barrier_releases_once() {
        let root = RuntimeFrontier {
            pending: vec![("start".to_owned(), 1, "root".to_owned(), None)],
            join_node_ids: vec!["join".to_owned()],
            join_outstanding: vec![1],
            join_waiters: BTreeMap::from_iter([("join".to_owned(), Vec::new())]),
            ..RuntimeFrontier::default()
        };
        let branches = frontier_after_worker_success(
            &root,
            "start",
            "root",
            None,
            0,
            &RuntimeWorkerSuccessEffect {
                successors: vec![
                    RuntimeSuccessor {
                        node_id: "left".to_owned(),
                        join_mask: 1,
                    },
                    RuntimeSuccessor {
                        node_id: "right".to_owned(),
                        join_mask: 1,
                    },
                ],
            },
        )
        .unwrap();
        assert_eq!(branches.join_outstanding, vec![2]);
        let left = branches
            .pending
            .iter()
            .find(|token| token.0 == "left")
            .unwrap()
            .clone();
        let after_left = frontier_after_worker_success(
            &branches,
            &left.0,
            &left.2,
            left.3.as_deref(),
            1,
            &RuntimeWorkerSuccessEffect {
                successors: vec![RuntimeSuccessor {
                    node_id: "join".to_owned(),
                    join_mask: 1,
                }],
            },
        )
        .unwrap();
        assert_eq!(after_left.join_outstanding, vec![1]);
        assert!(!after_left.pending.iter().any(|token| token.0 == "join"));
        assert_eq!(after_left.join_waiters["join"].len(), 1);

        let right = after_left
            .pending
            .iter()
            .find(|token| token.0 == "right")
            .unwrap()
            .clone();
        let released = frontier_after_worker_success(
            &after_left,
            &right.0,
            &right.2,
            right.3.as_deref(),
            2,
            &RuntimeWorkerSuccessEffect {
                successors: vec![RuntimeSuccessor {
                    node_id: "join".to_owned(),
                    join_mask: 1,
                }],
            },
        )
        .unwrap();
        assert_eq!(released.join_outstanding, vec![0]);
        assert!(released.join_waiters["join"].is_empty());
        assert_eq!(
            released
                .pending
                .iter()
                .filter(|token| token.0 == "join")
                .count(),
            1
        );
    }

    #[test]
    fn suspend_resume_moves_only_the_owned_token_and_clears_wait_reason() {
        let mut start = transition(RecordedTransitionKind::Start, "start", 0, 0);
        start.frontier = Some(RuntimeFrontier {
            pending: vec![("node".to_owned(), 0, "token".to_owned(), None)],
            ..RuntimeFrontier::default()
        });
        let started = reduce_recorded_transition(None, &start).unwrap();
        let suspended_frontier =
            frontier_after_worker_suspend(&started.state.frontier, "node", "token", None).unwrap();
        let mut suspend = transition(RecordedTransitionKind::Suspend, "suspend", 1, 0);
        suspend.frontier = Some(suspended_frontier.clone());
        suspend.wait_reason = Some("approval".to_owned());
        suspend.resume_payload = Some(json!({"request": 1}));
        let suspended = reduce_recorded_transition(Some(&started.state), &suspend).unwrap();
        assert_eq!(suspended.state.status, RecordedRunStatus::Suspended);
        assert_eq!(suspended.state.wait_reason.as_deref(), Some("approval"));

        let mut resume = transition(RecordedTransitionKind::ResumeResult, "resume", 2, 1);
        resume.frontier = Some(
            frontier_after_worker_resume(&suspended.state.frontier, "node", "token", None).unwrap(),
        );
        resume.resume_payload = Some(json!({"approved": true}));
        let resumed = reduce_recorded_transition(Some(&suspended.state), &resume).unwrap();
        assert_eq!(resumed.state.status, RecordedRunStatus::Running);
        assert_eq!(resumed.state.wait_reason, None);
        assert_eq!(
            resumed.state.resume_payload,
            Some(json!({"approved": true}))
        );
        assert_eq!(
            resumed.state.frontier.pending,
            started.state.frontier.pending
        );
        assert!(resumed.state.frontier.suspended.is_empty());
    }
