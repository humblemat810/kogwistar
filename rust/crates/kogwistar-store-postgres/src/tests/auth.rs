// Domain-focused tests; shared fixtures live in the parent test module.
use super::*;

#[tokio::test]
#[ignore = "requires KOGWISTAR_TEST_PG_DSN; run explicitly with --include-ignored"]
async fn auth_store_links_python_user_and_retries_identity_when_dsn_available()
-> PostgresStoreResult<()> {
    let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
        .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
    let schema = test_schema("auth_identity");
    let quoted_schema = quote_identifier(&schema)?;
    let base_config = dsn
        .parse::<Config>()
        .map_err(|error| PostgresStoreError::Backend(error.to_string()))?;
    let (client, connection) = base_config.connect(NoTls).await.map_err(backend)?;
    tokio::spawn(async move {
        let _ = connection.await;
    });
    client
        .batch_execute(&format!("CREATE SCHEMA {quoted_schema}"))
        .await
        .map_err(backend)?;
    let mut auth_config = base_config;
    auth_config.options(format!("-c search_path={schema}"));
    let store = PostgresAuthStore::from_config(auth_config)?;
    store.ensure_schema().await?;
    client
            .execute(
                &format!(
                    "INSERT INTO {quoted_schema}.users (user_id, email, display_name, is_active, global_role, global_ns, created_at) VALUES ('python-user', 'alice@example.com', 'Python Alice', TRUE, 'rw', 'docs,workflow', CURRENT_TIMESTAMP)"
                ),
                &[],
            )
            .await
            .map_err(backend)?;
    let request = ResolveExternalIdentity {
        issuer: "https://issuer.example".to_owned(),
        subject: "subject-1".to_owned(),
        email: "alice@example.com".to_owned(),
        display_name: Some("Ignored".to_owned()),
        new_user_id: "rust-user".to_owned(),
        default_role: "ro".to_owned(),
        default_ns: "docs".to_owned(),
    };
    let linked = store.resolve_external_identity(request.clone()).await?;
    assert_eq!(linked.user_id, "python-user");
    assert_eq!(linked.global_role.as_deref(), Some("rw"));
    let retried = store
        .resolve_external_identity(ResolveExternalIdentity {
            new_user_id: "must-not-win".to_owned(),
            email: "changed@example.com".to_owned(),
            ..request
        })
        .await?;
    assert_eq!(retried.user_id, "python-user");
    assert_eq!(
        store
            .external_identity("https://issuer.example", "subject-1")
            .await?
            .unwrap()
            .email
            .as_deref(),
        Some("alice@example.com")
    );
    client
        .batch_execute(&format!("DROP SCHEMA {quoted_schema} CASCADE"))
        .await
        .map_err(backend)?;
    Ok(())
}
