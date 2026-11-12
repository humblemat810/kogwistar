    #[test]
    fn authenticated_run_owner_uses_stable_user_id_and_rejects_anonymous() {
        assert_eq!(
            authenticated_run_owner_id(&json!({"user_id":"User-42", "sub":"alice@example.test"})),
            Some("User-42")
        );
        assert_eq!(
            authenticated_run_owner_id(&json!({"sub":"alice@example.test"})),
            Some("alice@example.test")
        );
        assert_eq!(authenticated_run_owner_id(&json!({"sub":"anonymous"})), None);
        assert_eq!(authenticated_run_owner_id(&json!({"role":"rw"})), None);
    }

    #[test]
    fn service_trigger_owner_must_match_authenticated_actor() {
        let capabilities = Arc::new(Mutex::new(CapabilityState::default()));
        let principal = json!({
            "user_id": "alice",
            "sub": "alice@example.test",
            "role": "rw",
            "ns": "alpha",
            "capabilities": ["service.manage", "write_graph"]
        });
        let admitted = runtime_owner_binding_v1(&principal, &capabilities).unwrap();

        let (owner, binding) = PostgresRunApplicationService::service_run_owner_binding(
            Some(" alice "),
            Some(&admitted),
        )
        .unwrap();
        assert_eq!(owner.as_deref(), Some("alice"));
        assert_eq!(binding.as_ref().map(|value| value.user_id.as_str()), Some("alice"));

        let (owner, binding) =
            PostgresRunApplicationService::service_run_owner_binding(None, Some(&admitted))
                .unwrap();
        assert!(owner.is_none() && binding.is_none());
        assert!(PostgresRunApplicationService::service_run_owner_binding(
            Some("victim"),
            Some(&admitted)
        )
        .is_err());
        assert!(PostgresRunApplicationService::service_run_owner_binding(
            Some("victim"),
            None
        )
        .is_err());
    }

    #[test]
    fn inline_run_owner_revalidation_uses_current_identity_and_admission_ceiling() {
        let capabilities = Arc::new(Mutex::new(CapabilityState::default()));
        let admitted = json!({
            "user_id": "alice",
            "sub": "alice@example.test",
            "role": "rw",
            "ns": ["alpha", "beta"],
            "storage_ns": "alpha",
            "capabilities": ["read_graph", "write_graph"]
        });
        let binding = runtime_owner_binding_v1(&admitted, &capabilities).unwrap();
        let current = AuthUser {
            user_id: "alice".to_owned(),
            email: "alice@example.test".to_owned(),
            display_name: None,
            is_active: true,
            global_role: Some("rw".to_owned()),
            global_ns: Some("alpha".to_owned()),
        };
        let principal = revalidated_runtime_owner_principal(
            &binding,
            &current,
            &capabilities,
            "write_graph",
        )
        .unwrap();
        assert_eq!(principal["ns"], json!(["alpha"]));
        assert!(principal_strings(&principal, "capabilities").contains(&"write_graph".to_owned()));

        let downgraded = AuthUser {
            global_role: Some("ro".to_owned()),
            ..current.clone()
        };
        assert!(revalidated_runtime_owner_principal(
            &binding,
            &downgraded,
            &capabilities,
            "write_graph"
        )
        .is_err());
        let deactivated = AuthUser {
            is_active: false,
            ..current.clone()
        };
        assert!(revalidated_runtime_owner_principal(
            &binding,
            &deactivated,
            &capabilities,
            "write_graph"
        )
        .is_err());
        let moved_namespace = AuthUser {
            global_ns: Some("beta".to_owned()),
            ..current.clone()
        };
        assert!(
            revalidated_runtime_owner_principal(
                &binding,
                &moved_namespace,
                &capabilities,
                "write_graph"
            )
            .is_err()
        );

        let narrow_admission = json!({
            "user_id": "alice",
            "sub": "alice@example.test",
            "role": "rw",
            "ns": "alpha",
            "capabilities": ["read_graph"]
        });
        let narrow_binding = runtime_owner_binding_v1(&narrow_admission, &capabilities).unwrap();
        capabilities
            .lock()
            .unwrap()
            .approvals
            .insert(("alice".to_owned(), "later-approval".to_owned()), BTreeSet::from(["write_graph".to_owned()]));
        assert!(
            revalidated_runtime_owner_principal(
                &narrow_binding,
                &current,
                &capabilities,
                "write_graph"
            )
            .is_err(),
            "a post-admission capability approval must not widen the run"
        );
    }

    #[test]
    fn inline_run_owner_revalidation_supports_read_only_native_capability() {
        let capabilities = Arc::new(Mutex::new(CapabilityState::default()));
        let admitted = json!({
            "user_id": "alice",
            "sub": "alice@example.test",
            "role": "ro",
            "ns": "alpha",
            "capabilities": ["read_graph"]
        });
        let binding = runtime_owner_binding_v1(&admitted, &capabilities).unwrap();
        let current = AuthUser {
            user_id: "alice".to_owned(),
            email: "alice@example.test".to_owned(),
            display_name: None,
            is_active: true,
            global_role: Some("ro".to_owned()),
            global_ns: Some("alpha".to_owned()),
        };

        let principal = revalidated_runtime_owner_principal(
            &binding,
            &current,
            &capabilities,
            "read_graph",
        )
        .unwrap();
        assert!(principal_strings(&principal, "capabilities").contains(&"read_graph".to_owned()));
        assert!(revalidated_runtime_owner_principal(
            &binding,
            &current,
            &capabilities,
            "write_graph"
        )
        .is_err());
    }

    #[tokio::test]
    async fn postgres_submit_rejects_user_id_spoofing_before_storage() {
        let service = PostgresRunApplicationService::from_dsn(
            "postgresql://worker:secret@127.0.0.1:1/unreachable",
            "worker_owner_authority",
        )
        .unwrap();
        let response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/workflow/runs".to_owned(),
                body: serde_json::to_vec(&json!({
                    "workflow_id": "wf",
                    "conversation_id": "conv",
                    "user_id": "victim"
                }))
                .unwrap(),
                principal: json!({"user_id":"attacker", "sub":"attacker", "role":"rw"}),
            })
            .await;
        assert_eq!(response.status, StatusCode::FORBIDDEN.as_u16());
        let body: Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(body["code"], "KOGWISTAR_RUN_OWNER_FORBIDDEN");

        let anonymous_claim = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/workflow/runs".to_owned(),
                body: serde_json::to_vec(&json!({
                    "workflow_id": "wf",
                    "conversation_id": "conv",
                    "user_id": "victim"
                }))
                .unwrap(),
                principal: json!({"sub":"anonymous", "role":"rw"}),
            })
            .await;
        assert_eq!(anonymous_claim.status, StatusCode::FORBIDDEN.as_u16());
        assert_eq!(
            serde_json::from_slice::<Value>(&anonymous_claim.body).unwrap()["code"],
            "KOGWISTAR_RUN_OWNER_FORBIDDEN"
        );
    }
