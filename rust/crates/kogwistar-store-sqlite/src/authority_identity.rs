//! In-process tracking for active SQLite file identities.
//!
//! This is separate from the OS-level authority lock: it detects a database
//! path being replaced while an authority handle is still live.

use super::*;

pub(super) static ACTIVE_DATABASE_IDENTITIES: OnceLock<Mutex<BTreeMap<PathBuf, (String, usize)>>> =
    OnceLock::new();

pub(super) fn register_active_database_identity(
    path: PathBuf,
    identity: &str,
    display_path: &Path,
) -> SqliteStoreResult<()> {
    let identities = ACTIVE_DATABASE_IDENTITIES.get_or_init(|| Mutex::new(BTreeMap::new()));
    let mut active = identities
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    match active.get_mut(&path) {
        Some((existing, _)) if existing != identity => Err(SqliteStoreError::AuthorityConflict {
            path: display_path.display().to_string(),
            message: "database file identity changed while its authority was active".to_owned(),
        }),
        Some((_, count)) => {
            *count += 1;
            Ok(())
        }
        None => {
            active.insert(path, (identity.to_owned(), 1));
            Ok(())
        }
    }
}

pub(super) fn unregister_active_database_identity(path: &Path, identity: &str) {
    let identities = ACTIVE_DATABASE_IDENTITIES.get_or_init(|| Mutex::new(BTreeMap::new()));
    let mut active = identities
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some((current, count)) = active.get_mut(path)
        && current == identity
    {
        if *count <= 1 {
            active.remove(path);
        } else {
            *count -= 1;
        }
    }
}
