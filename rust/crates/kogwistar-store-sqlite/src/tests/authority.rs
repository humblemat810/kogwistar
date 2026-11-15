// Domain-focused tests; shared fixtures live in the parent test module.
use super::*;

#[test]
fn private_memory_database_preserves_sqlite_memory_semantics() {
    let first = SqliteStore::open(":memory:").unwrap();
    let second = SqliteStore::open(":memory:").unwrap();

    assert_eq!(first.next_global_seq().unwrap(), 1);
    assert_eq!(first.current_global_seq().unwrap(), 1);
    assert_eq!(second.current_global_seq().unwrap(), 0);

    let first_connection = first.connection.lock().unwrap();
    let first_filename: String = first_connection
        .query_row("PRAGMA database_list", [], |row| row.get(2))
        .unwrap();
    assert!(first_filename.is_empty());
    first_connection
        .execute("CREATE TABLE private_memory_probe (value INTEGER)", [])
        .unwrap();
    drop(first_connection);

    let second_connection = second.connection.lock().unwrap();
    let second_filename: String = second_connection
        .query_row("PRAGMA database_list", [], |row| row.get(2))
        .unwrap();
    assert!(second_filename.is_empty());
    let table_count: i64 = second_connection
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'private_memory_probe'",
                [],
                |row| row.get(0),
            )
            .unwrap();
    assert_eq!(table_count, 0);
}
#[test]
fn active_path_identity_is_shared_and_replacement_fails_closed() {
    let path = database_path("identity-registry");
    register_active_database_identity(path.clone(), "file-identity-a", &path).unwrap();
    register_active_database_identity(path.clone(), "file-identity-a", &path).unwrap();
    assert!(matches!(
        register_active_database_identity(path.clone(), "file-identity-b", &path),
        Err(SqliteStoreError::AuthorityConflict { .. })
    ));

    unregister_active_database_identity(&path, "file-identity-a");
    assert!(matches!(
        register_active_database_identity(path.clone(), "file-identity-b", &path),
        Err(SqliteStoreError::AuthorityConflict { .. })
    ));
    unregister_active_database_identity(&path, "file-identity-a");
    register_active_database_identity(path.clone(), "file-identity-b", &path).unwrap();
    unregister_active_database_identity(&path, "file-identity-b");
}
#[test]
fn concurrent_file_identity_replacement_has_one_active_winner() {
    let path = database_path("identity-race");
    let barrier = Arc::new(Barrier::new(3));
    let workers = ["file-identity-a", "file-identity-b"].map(|identity| {
        let path = path.clone();
        let barrier = Arc::clone(&barrier);
        thread::spawn(move || {
            barrier.wait();
            let result = register_active_database_identity(path.clone(), identity, &path);
            (identity, result.is_ok())
        })
    });
    barrier.wait();
    let outcomes = workers.map(|worker| worker.join().unwrap());
    let winners: Vec<_> = outcomes
        .iter()
        .filter_map(|(identity, accepted)| accepted.then_some(*identity))
        .collect();
    assert_eq!(winners.len(), 1);

    let active = ACTIVE_DATABASE_IDENTITIES.get().unwrap().lock().unwrap();
    assert_eq!(active.get(&path).map(|(_, count)| *count), Some(1));
    drop(active);
    unregister_active_database_identity(&path, winners[0]);
}
#[test]
fn standalone_rust_process_cannot_open_an_authority_owned_database() {
    let Ok(path) = std::env::var("KOGWISTAR_SQLITE_LOCK_CHILD_PATH") else {
        let path = database_path("cross-process-authority");
        let ready_path = PathBuf::from(format!("{}.ready", path.display()));
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args([
                "tests::authority::standalone_rust_process_cannot_open_an_authority_owned_database",
                "--nocapture",
            ])
            .env("KOGWISTAR_SQLITE_LOCK_CHILD_PATH", &path)
            .env("KOGWISTAR_SQLITE_LOCK_CHILD_READY", &ready_path)
            .stdin(Stdio::piped())
            .spawn()
            .unwrap();
        for _ in 0..50 {
            if ready_path.exists() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        assert!(
            ready_path.exists(),
            "child did not acquire SQLite authority"
        );
        let error = SqliteStore::open(&path).unwrap_err();
        assert!(matches!(error, SqliteStoreError::AuthorityConflict { .. }));
        drop(child.stdin.take());
        assert!(child.wait().unwrap().success());
        let _ = fs::remove_file(&path);
        let _ = fs::remove_file(&ready_path);
        let _ = fs::remove_file(format!("{}.kogwistar-authority.lock", path.display()));
        return;
    };

    let path = PathBuf::from(path);
    let expect_conflict = std::env::var_os("KOGWISTAR_SQLITE_LOCK_EXPECT_CONFLICT").is_some();
    if expect_conflict {
        assert!(matches!(
            SqliteStore::open(&path),
            Err(SqliteStoreError::AuthorityConflict { .. })
        ));
    } else {
        let store = SqliteStore::open(&path).unwrap();
        std::hint::black_box(&store);
        fs::write(
            std::env::var("KOGWISTAR_SQLITE_LOCK_CHILD_READY").unwrap(),
            b"ready",
        )
        .unwrap();
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();
        drop(store);
        return;
    }
    fs::write(
        std::env::var("KOGWISTAR_SQLITE_LOCK_CHILD_READY").unwrap(),
        b"ready",
    )
    .unwrap();
    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();
}
#[cfg(unix)]
#[test]
fn active_rust_authority_rejects_database_path_replacement() {
    let (store, path) = store("active-path-replacement");
    let original = path.with_extension("original.db");
    let replacement = path.with_extension("replacement.db");
    let replacement_connection = Connection::open(&replacement).unwrap();
    replacement_connection
        .execute_batch("CREATE TABLE replacement_probe (value INTEGER);")
        .unwrap();
    drop(replacement_connection);

    fs::rename(&path, &original).unwrap();
    fs::rename(&replacement, &path).unwrap();
    let error = SqliteStore::open(&path).unwrap_err();
    assert!(matches!(error, SqliteStoreError::AuthorityConflict { .. }));

    drop(store);
    fs::remove_file(original).unwrap();
    fs::remove_file(path).unwrap();
}
