    #[test]
    fn conversation_reads_share_frozen_storage_scope_and_acl_principal() {
        let (scope, acl) = conversation_graph_scope_and_acl_principal(&json!({
            "sub": "Alice",
            "ns": "conversation",
            "storage_ns": "ConversationData",
            "workspace_id": "Workspace-A",
            "graph_space": "graph-space-a",
            "security_scope": "Tenant-A",
            "groups": [" Team-B ", "team-a"],
            "principal_groups": ["team-a", "team-c"]
        }));

        assert_eq!(scope.namespace, "conversationdata");
        assert_eq!(scope.workspace_id.as_deref(), Some("Workspace-A"));
        assert_eq!(scope.graph_space.as_deref(), Some("graph-space-a"));
        assert_eq!(acl.principal_id, "alice");
        assert_eq!(
            acl.groups,
            BTreeSet::from([
                "team-a".to_owned(),
                "team-b".to_owned(),
                "team-c".to_owned()
            ])
        );
        assert_eq!(acl.security_scope.as_deref(), Some("tenant-a"));
    }

    #[test]
    fn wisdom_storage_schema_is_configurable_with_legacy_default() {
        assert_eq!(
            configured_wisdom_schema("kg", None).unwrap(),
            "kg_wisdom"
        );
        assert_eq!(
            configured_wisdom_schema("kg", Some("kg")).unwrap(),
            "kg",
            "the generic graph store can be shared by graph roles"
        );
        assert_eq!(
            configured_wisdom_schema("kg", Some("tenant_wisdom")).unwrap(),
            "tenant_wisdom"
        );
        assert!(configured_wisdom_schema("kg", Some("  ")).is_err());
    }

    #[test]
    fn graph_acl_principal_identity_matches_shared_python_contract() {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/graph-acl-principal-identity-v1.json"
        )))
        .expect("valid ACL principal identity fixture");
        assert_eq!(fixture["schema_version"], 1);
        for case in fixture["cases"].as_array().expect("identity cases") {
            assert_eq!(
                graph_acl_principal_id(&case["claims"]),
                case["expected_principal_id"]
                    .as_str()
                    .expect("expected principal ID"),
                "{}",
                case["name"].as_str().expect("case name")
            );
        }
        for case in fixture["group_cases"].as_array().expect("group cases") {
            let claims = &case["claims"];
            let groups = principal_group_set(claims);
            let (_, acl) = conversation_graph_scope_and_acl_principal(claims);
            let expected = case["expected_groups"]
                .as_array()
                .expect("expected groups")
                .iter()
                .map(|value| value.as_str().expect("group").to_owned())
                .collect::<BTreeSet<_>>();
            assert_eq!(
                groups,
                acl.groups,
                "group projection and ACL scope construction must share one normalized set"
            );
            assert_eq!(
                acl.groups,
                expected,
                "{}",
                case["name"].as_str().expect("case name")
            );
        }
    }

    #[test]
    fn security_scope_projection_matches_shared_python_contract() {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/auth-scope-projection-v1.json"
        )))
        .expect("valid auth scope projection fixture");
        assert_eq!(fixture["schema_version"], 1);
        for case in fixture["cases"].as_array().expect("scope cases") {
            let actual = security_scope_value(&case["claims"]);
            for (key, expected) in case["expected"].as_object().expect("expected mapping") {
                assert_eq!(actual[key], *expected, "{}: {key}", case["name"]);
            }
        }
    }
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use kogwistar_contracts::native_seam_attestation_is_valid;
    use tower::ServiceExt;

    fn remove_test_sqlite(path: &std::path::Path) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
        loop {
            match std::fs::remove_file(path) {
                Ok(()) => return,
                Err(error) if std::time::Instant::now() < deadline => {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                    let _ = error;
                }
                Err(error) => panic!("failed to remove test SQLite file {path:?}: {error}"),
            }
        }
    }
