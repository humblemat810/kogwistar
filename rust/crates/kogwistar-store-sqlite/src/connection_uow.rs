use super::*;

pub(super) fn configured_connection(path: &Path) -> SqliteStoreResult<Connection> {
    let conn = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE,
    )?;
    conn.busy_timeout(std::time::Duration::from_millis(BUSY_TIMEOUT_MS))?;
    conn.execute_batch("PRAGMA foreign_keys = ON; PRAGMA synchronous = NORMAL;")?;
    let mut journal_mode: String = conn.query_row("PRAGMA journal_mode", [], |row| row.get(0))?;
    if !journal_mode.eq_ignore_ascii_case("wal") {
        conn.execute_batch("PRAGMA journal_mode = WAL;")?;
        journal_mode = conn.query_row("PRAGMA journal_mode", [], |row| row.get(0))?;
    }
    if std::env::var_os("KOGWISTAR_SQLITE_DIAGNOSTICS").as_deref()
        == Some(std::ffi::OsStr::new("1"))
    {
        let version = unsafe {
            CStr::from_ptr(rusqlite::ffi::sqlite3_libversion())
                .to_string_lossy()
                .into_owned()
        };
        let source_id = unsafe {
            CStr::from_ptr(rusqlite::ffi::sqlite3_sourceid())
                .to_string_lossy()
                .into_owned()
        };
        eprintln!(
            "[sqlite-rust-open] conn={:p} path={} version={} source_id={} journal_mode={} {}",
            unsafe { conn.handle() },
            path.display(),
            version,
            source_id,
            journal_mode,
            sqlite_sidecars(path)
        );
    }
    Ok(conn)
}

pub(super) fn sqlite_lifecycle_diagnostic(event: &str, path: &Path) {
    if std::env::var_os("KOGWISTAR_SQLITE_DIAGNOSTICS").as_deref()
        != Some(std::ffi::OsStr::new("1"))
    {
        return;
    }
    eprintln!(
        "[sqlite-rust-{event}] path={} {}",
        path.display(),
        sqlite_sidecars(path)
    );
}

pub(super) fn sqlite_sidecars(path: &Path) -> String {
    [
        path.to_path_buf(),
        PathBuf::from(format!("{}-wal", path.display())),
        PathBuf::from(format!("{}-shm", path.display())),
    ]
    .iter()
    .map(|candidate| match fs::metadata(candidate) {
        Ok(metadata) => format!(
            "{}=size:{}",
            candidate.file_name().unwrap_or_default().to_string_lossy(),
            metadata.len()
        ),
        Err(_) => format!(
            "{}=absent",
            candidate.file_name().unwrap_or_default().to_string_lossy()
        ),
    })
    .collect::<Vec<_>>()
    .join(" ")
}

/// Operations sharing one `BEGIN IMMEDIATE` transaction.
pub struct SqliteUnitOfWork<'connection> {
    pub(super) transaction: &'connection Connection,
}

mod design_and_runs;
mod events_recovery;
mod index_jobs;
mod lane_messages;
mod projections;
mod recorded_runtime;
