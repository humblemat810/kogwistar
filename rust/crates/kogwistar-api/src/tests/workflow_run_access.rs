#[tokio::test]
async fn workflow_run_status_denies_wrong_scope_before_storage_access() {
    let service = PostgresRunApplicationService::from_dsn(
        "postgres://unused:unused@127.0.0.1:1/unused",
        "run_status_authorization",
    )
    .unwrap();

    for (principal, expected_code) in [
        (
            json!({"role":"ro", "ns":["workflow"], "capabilities":[]}),
            "KOGWISTAR_CAPABILITY_FORBIDDEN",
        ),
        (
            json!({"role":"ro", "ns":["conversation"], "capabilities":["workflow.run.read"]}),
            "KOGWISTAR_NAMESPACE_FORBIDDEN",
        ),
    ] {
        let response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: "/api/workflow/runs/private-run".to_owned(),
                body: Vec::new(),
                principal,
            })
            .await;
        assert_eq!(response.status, StatusCode::FORBIDDEN.as_u16());
        assert_eq!(
            serde_json::from_slice::<Value>(&response.body).unwrap()["code"],
            expected_code
        );
    }
}

#[tokio::test]
async fn workflow_checkpoint_get_denies_wrong_scope_before_storage_access() {
    let service = PostgresRunApplicationService::from_dsn(
        "postgres://unused:unused@127.0.0.1:1/unused",
        "checkpoint_get_authorization",
    )
    .unwrap();

    for path in [
        "/api/workflow/runs/private-run/checkpoints/0",
        "/api/workflow/runs/private-run/replay?target_step_seq=0",
    ] {
        for (principal, expected_code) in [
            (
                json!({"role":"ro", "ns":["workflow"], "capabilities":[]}),
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
            ),
            (
                json!({"role":"ro", "ns":["conversation"], "capabilities":["workflow.run.read"]}),
                "KOGWISTAR_NAMESPACE_FORBIDDEN",
            ),
        ] {
            let response = service
                .execute(ApiEffectRequest {
                    contract_version: 1,
                    method: "GET".to_owned(),
                    path_and_query: path.to_owned(),
                    body: Vec::new(),
                    principal,
                })
                .await;
            assert_eq!(response.status, StatusCode::FORBIDDEN.as_u16());
            assert_eq!(
                serde_json::from_slice::<Value>(&response.body).unwrap()["code"],
                expected_code
            );
        }
    }
}

#[tokio::test]
async fn workflow_run_events_denies_wrong_scope_before_storage_access() {
    let service = PostgresRunApplicationService::from_dsn(
        "postgres://unused:unused@127.0.0.1:1/unused",
        "run_events_authorization",
    )
    .unwrap();

    for (principal, expected_code) in [
        (
            json!({"role":"ro", "ns":["workflow"], "capabilities":[]}),
            "KOGWISTAR_CAPABILITY_FORBIDDEN",
        ),
        (
            json!({"role":"ro", "ns":["conversation"], "capabilities":["workflow.run.read"]}),
            "KOGWISTAR_NAMESPACE_FORBIDDEN",
        ),
    ] {
        let response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query:
                    "/api/workflow/runs/private-run/events/poll?after_seq=0&limit=10".to_owned(),
                body: Vec::new(),
                principal,
            })
            .await;
        assert_eq!(response.status, StatusCode::FORBIDDEN.as_u16());
        assert_eq!(
            serde_json::from_slice::<Value>(&response.body).unwrap()["code"],
            expected_code
        );
    }
}

#[tokio::test]
async fn workflow_run_steps_denies_wrong_scope_before_storage_access() {
    let service = PostgresRunApplicationService::from_dsn(
        "postgres://unused:unused@127.0.0.1:1/unused",
        "run_steps_authorization",
    )
    .unwrap();

    for (principal, expected_code) in [
        (
            json!({"role":"ro", "ns":["workflow"], "capabilities":[]}),
            "KOGWISTAR_CAPABILITY_FORBIDDEN",
        ),
        (
            json!({"role":"ro", "ns":["conversation"], "capabilities":["workflow.run.read"]}),
            "KOGWISTAR_NAMESPACE_FORBIDDEN",
        ),
    ] {
        let response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: "/api/workflow/runs/private-run/steps?limit=25".to_owned(),
                body: Vec::new(),
                principal,
            })
            .await;
        assert_eq!(response.status, StatusCode::FORBIDDEN.as_u16());
        assert_eq!(
            serde_json::from_slice::<Value>(&response.body).unwrap()["code"],
            expected_code
        );
    }
}

#[tokio::test]
async fn sqlite_workflow_run_steps_enforces_per_workflow_read_acl() {
    let sqlite_path = std::env::temp_dir().join(format!(
        "kogwistar-sqlite-run-steps-acl-{}.sqlite",
        uuid::Uuid::new_v4().simple()
    ));
    let auth_path = std::env::temp_dir().join(format!(
        "kogwistar-sqlite-run-steps-acl-auth-{}.sqlite",
        uuid::Uuid::new_v4().simple()
    ));
    let auth = SqliteAuthStore::open(&auth_path).unwrap();
    auth.resolve_external_identity(ResolveExternalIdentity {
        issuer: "test".to_owned(),
        subject: "steps-acl-user".to_owned(),
        email: "steps-acl-user@example.test".to_owned(),
        display_name: None,
        new_user_id: "steps-acl-user".to_owned(),
        default_role: "rw".to_owned(),
        default_ns: "workflow".to_owned(),
    })
    .await
    .unwrap();
    drop(auth);

    let auth_config = AuthConfig {
        auth_db_url: Some(format!("sqlite:///{}", auth_path.display())),
        ..AuthConfig::default()
    };
    let service = SqliteRunApplicationService::open(&sqlite_path)
        .unwrap()
        .with_owner_auth_config(&auth_config)
        .await
        .unwrap();
    service
        .store
        .create_server_run(ServerRunCreate {
            run_id: "private-steps-run".to_owned(),
            conversation_id: "steps-conversation".to_owned(),
            workflow_id: "private-steps-workflow".to_owned(),
            user_id: Some("steps-acl-user".to_owned()),
            user_turn_node_id: "steps-turn".to_owned(),
            status: "completed".to_owned(),
        })
        .unwrap();

    let response = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "GET".to_owned(),
            path_and_query: "/api/workflow/runs/private-steps-run/steps?limit=25".to_owned(),
            body: Vec::new(),
            principal: json!({
                "user_id": "steps-acl-user",
                "role": "rw",
                "ns": ["workflow"],
                "capabilities": ["workflow.run.read"]
            }),
        })
        .await;

    assert_eq!(response.status, StatusCode::FORBIDDEN.as_u16());
    assert_eq!(
        serde_json::from_slice::<Value>(&response.body).unwrap()["code"],
        "KOGWISTAR_WORKFLOW_FORBIDDEN"
    );
    drop(service);
    let _ = std::fs::remove_file(sqlite_path);
    let _ = std::fs::remove_file(auth_path);
}
