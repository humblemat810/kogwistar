    #[test]
    fn native_runtime_routing_supports_explicit_targets_and_legacy_prefixes() {
        assert_eq!(native_operation_name("rust-native/noop"), Some("noop"));
        assert_eq!(native_operation_name("rust-native/"), None);
        assert_eq!(native_operation_name("noop"), None);
        assert_eq!(
            runtime_worker_recipient_for_operation("rust-native/noop", None, None, None),
            "rust-native-worker"
        );
        assert_eq!(
            runtime_worker_recipient_for_operation("noop", None, None, None),
            "python-worker"
        );
        assert_eq!(
            runtime_worker_recipient_for_operation(
                "rust-native/noop",
                Some("numpy-cpu-2"),
                None,
                None,
            ),
            "rust-native-worker"
        );
        assert_eq!(
            runtime_worker_recipient_for_operation("noop", Some("numpy-cpu-2"), None, None),
            runtime_worker_recipient(Some("numpy-cpu-2"))
        );
        assert_eq!(
            runtime_worker_recipient_for_operation(
                "increment",
                None,
                Some("rust-native"),
                None,
            ),
            "rust-native-worker"
        );
        assert_eq!(
            runtime_worker_recipient_for_operation(
                "increment",
                None,
                Some("python-worker"),
                None,
            ),
            "python-worker"
        );
        assert!(runtime_graph_plan_from_snapshot(
            r#"{"nodes":[{"id":"start","metadata":{"wf_start":true,"wf_executor_target":"gpu"}}],"edges":[]}"#
        )
        .is_none());

        let identity = json!({
            "schema_version": 1,
            "environment_id": "python-cpu-v1",
            "artifact_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "dependency_lock_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        });
        assert_eq!(
            runtime_worker_recipient_for_operation(
                "parse",
                Some("python-cpu-v1"),
                None,
                Some(&identity),
            ),
            "python-worker-identity-Pg8kjaa_r5hcsnNN8gIUMKd6x7EO4Sugfo92UM4XSVg"
        );
    }

    #[test]
    fn runtime_graph_plan_matches_python_design_topology_invariants() {
        let snapshot = json!({
            "nodes": [
                {"id": "start", "metadata": {"wf_start": true}},
                {"id": "end", "metadata": {"wf_terminal": true}},
                {"id": "unused", "metadata": {}}
            ],
            "edges": [{
                "id": "route",
                "source_ids": ["start", "unused"],
                "target_ids": ["end", "unused"],
                "metadata": {}
            }]
        });
        let plan = runtime_graph_plan_from_snapshot(&snapshot.to_string())
            .expect("first source/target form the Python-compatible transition");
        assert_eq!(plan.routes.len(), 1);
        assert_eq!(plan.routes[0].source_node_id, "start");
        assert_eq!(plan.routes[0].target_node_id, "end");

        let duplicate_start = json!({
            "nodes": [
                {"id": "start", "metadata": {"wf_start": true}},
                {"id": "second", "metadata": {"wf_start": true}}
            ],
            "edges": []
        });
        assert!(runtime_graph_plan_from_snapshot(&duplicate_start.to_string()).is_none());

        let missing_node_id = json!({
            "nodes": [
                {"id": "start", "metadata": {"wf_start": true}},
                {"metadata": {}}
            ],
            "edges": []
        });
        assert!(runtime_graph_plan_from_snapshot(&missing_node_id.to_string()).is_none());

        let dangling_edge = json!({
            "nodes": [{"id": "start", "metadata": {"wf_start": true}}],
            "edges": [{"id": "bad", "source_ids": ["start"], "target_ids": ["missing"]}]
        });
        assert!(runtime_graph_plan_from_snapshot(&dangling_edge.to_string()).is_none());

        let empty_endpoints = json!({
            "nodes": [{"id": "start", "metadata": {"wf_start": true}}],
            "edges": [{"id": "bad", "source_ids": [], "target_ids": ["start"]}]
        });
        assert!(runtime_graph_plan_from_snapshot(&empty_endpoints.to_string()).is_none());

        let no_reachable_terminal = json!({
            "nodes": [
                {"id": "start", "metadata": {"wf_start": true}},
                {"id": "loop", "metadata": {}}
            ],
            "edges": [
                {"id": "to-loop", "source_ids": ["start"], "target_ids": ["loop"]},
                {"id": "to-start", "source_ids": ["loop"], "target_ids": ["start"]}
            ]
        });
        assert!(runtime_graph_plan_from_snapshot(&no_reachable_terminal.to_string()).is_none());
    }

    #[test]
    fn runtime_graph_plan_freezes_only_valid_python_worker_identity() {
        let identity = json!({
            "schema_version": 1,
            "environment_id": "python-cpu-v1",
            "artifact_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "dependency_lock_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        });
        let mut snapshot = json!({
            "nodes": [
                {"id": "start", "metadata": {
                    "wf_start": true,
                    "wf_op": "parse",
                    "wf_environment_id": "python-cpu-v1",
                    "wf_worker_identity": identity
                }},
                {"id": "end", "metadata": {"wf_terminal": true}}
            ],
            "edges": [{"id": "route", "source_ids": ["start"], "target_ids": ["end"]}]
        });
        let plan = runtime_graph_plan_from_snapshot(&snapshot.to_string())
            .expect("valid worker identity is frozen in the workflow plan");
        assert_eq!(plan.node_worker_identities.get("start"), Some(&identity));

        snapshot["nodes"][0]["metadata"]["wf_worker_identity"]["environment_id"] =
            json!("stale-env");
        assert!(runtime_graph_plan_from_snapshot(&snapshot.to_string()).is_none());

        snapshot["nodes"][0]["metadata"]["wf_worker_identity"] = identity;
        snapshot["nodes"][0]["metadata"]["wf_executor_target"] = json!("rust-native");
        assert!(runtime_graph_plan_from_snapshot(&snapshot.to_string()).is_none());
    }

    #[test]
    fn explicit_worker_identity_map_must_match_environment_and_executor() {
        let identity = json!({
            "schema_version": 1,
            "environment_id": "python-cpu-v1",
            "artifact_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "dependency_lock_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        });
        let identities = BTreeMap::from([("start".to_owned(), identity.clone())]);
        let environments = BTreeMap::from([("start".to_owned(), "python-cpu-v1".to_owned())]);
        assert!(runtime_worker_identity_map_is_valid(
            &identities,
            &environments,
            &BTreeMap::new()
        ));

        let stale_environment = BTreeMap::from([("start".to_owned(), "stale-env".to_owned())]);
        assert!(!runtime_worker_identity_map_is_valid(
            &identities,
            &stale_environment,
            &BTreeMap::new()
        ));

        let native_target = BTreeMap::from([("start".to_owned(), "rust-native".to_owned())]);
        assert!(!runtime_worker_identity_map_is_valid(
            &identities,
            &environments,
            &native_target
        ));

        let malformed = BTreeMap::from([("start".to_owned(), json!({"schema_version": 99}))]);
        assert!(!runtime_worker_identity_map_is_valid(
            &malformed,
            &environments,
            &BTreeMap::new()
        ));

        let snapshot = WorkflowDesignSnapshot {
            workflow_id: "wf".to_owned(),
            version: 1,
            seq: 1,
            payload_json: json!({
                "nodes": [{"metadata": {"wf_worker_identity": {"schema_version": 99}}}],
                "edges": []
            })
            .to_string(),
            schema_version: 1,
            created_at_ms: 0,
        };
        assert!(runtime_snapshot_declares_worker_identity(Some(&snapshot)));
        assert!(!runtime_snapshot_declares_worker_identity(None));
    }
