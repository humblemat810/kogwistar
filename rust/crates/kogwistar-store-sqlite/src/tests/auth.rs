// Domain-focused tests; shared fixtures live in the parent test module.
use super::*;

#[test]
fn auth_store_opens_python_schema_and_resolves_identity_atomically() {
    let path = database_path("auth");
    let store = SqliteAuthStore::open(&path).unwrap();
    let request = ResolveExternalIdentity {
        issuer: "https://issuer.example".to_owned(),
        subject: "subject-1".to_owned(),
        email: "alice@example.com".to_owned(),
        display_name: Some("Alice".to_owned()),
        new_user_id: "generated-1".to_owned(),
        default_role: "ro".to_owned(),
        default_ns: "docs".to_owned(),
    };
    let created = block_on(store.resolve_external_identity(request.clone())).unwrap();
    assert_eq!(created.user_id, "generated-1");
    assert_eq!(created.email, "alice@example.com");
    assert_eq!(created.display_name.as_deref(), Some("Alice"));
    assert_eq!(created.global_role.as_deref(), Some("ro"));
    let retried = block_on(store.resolve_external_identity(ResolveExternalIdentity {
        new_user_id: "must-not-win".to_owned(),
        email: "changed@example.com".to_owned(),
        ..request
    }))
    .unwrap();
    assert_eq!(retried.user_id, "generated-1");
    let identity = block_on(store.external_identity("https://issuer.example", "subject-1"))
        .unwrap()
        .unwrap();
    assert_eq!(identity.user_id, "generated-1");
    assert_eq!(identity.email.as_deref(), Some("alice@example.com"));
    let connection = Connection::open(&path).unwrap();
    let table_names = connection
        .prepare("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")
        .unwrap()
        .query_map([], |row| row.get::<_, String>(0))
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(
        table_names,
        ["external_identities", "users", "workflow_acl"]
    );
    drop(connection);
    drop(store);
    fs::remove_file(path).unwrap();
}
#[test]
fn auth_store_links_existing_python_user_by_email() {
    let path = database_path("auth-python-user");
    let store = SqliteAuthStore::open(&path).unwrap();
    let connection = Connection::open(&path).unwrap();
    connection
            .execute(
                "INSERT INTO users (user_id, email, display_name, is_active, global_role, global_ns, created_at) VALUES ('python-user', 'alice@example.com', 'Python Alice', 1, 'rw', 'docs,workflow', CURRENT_TIMESTAMP)",
                [],
            )
            .unwrap();
    drop(connection);
    let user = block_on(store.resolve_external_identity(ResolveExternalIdentity {
        issuer: "issuer".to_owned(),
        subject: "subject".to_owned(),
        email: "alice@example.com".to_owned(),
        display_name: Some("Ignored".to_owned()),
        new_user_id: "rust-user".to_owned(),
        default_role: "ro".to_owned(),
        default_ns: "docs".to_owned(),
    }))
    .unwrap();
    assert_eq!(user.user_id, "python-user");
    assert_eq!(user.display_name.as_deref(), Some("Python Alice"));
    assert_eq!(user.global_role.as_deref(), Some("rw"));
    assert_eq!(
        block_on(store.external_identity("issuer", "subject"))
            .unwrap()
            .unwrap()
            .user_id,
        "python-user"
    );
    drop(store);
    fs::remove_file(path).unwrap();
}
