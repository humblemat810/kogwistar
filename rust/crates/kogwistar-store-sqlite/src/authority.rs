use super::*;

use fs2::FileExt;

static PROCESS_AUTHORITY_LOCKS: OnceLock<Mutex<BTreeMap<PathBuf, Weak<File>>>> = OnceLock::new();
static DATABASE_INITIALIZATION_LOCKS: OnceLock<Mutex<BTreeMap<PathBuf, Weak<Mutex<()>>>>> =
    OnceLock::new();

pub(super) fn database_initialization_lock(database: &Path) -> Arc<Mutex<()>> {
    let key = normalized_database_path(database);
    let locks = DATABASE_INITIALIZATION_LOCKS.get_or_init(|| Mutex::new(BTreeMap::new()));
    let mut locks = locks
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    locks.retain(|_, lock| lock.strong_count() > 0);
    if let Some(lock) = locks.get(&key).and_then(Weak::upgrade) {
        return lock;
    }
    let lock = Arc::new(Mutex::new(()));
    locks.insert(key, Arc::downgrade(&lock));
    lock
}

#[derive(Debug)]
struct AuthorityLockHandle {
    _file: Arc<File>,
}

fn acquire_authority_lock_handle(
    lock_path: PathBuf,
    _authority_key: &str,
    display_path: &Path,
) -> SqliteStoreResult<AuthorityLockHandle> {
    let gate_path = PathBuf::from(format!("{}.admission", lock_path.display()));
    let own_path = PathBuf::from(format!("{}.rust", lock_path.display()));
    let other_path = PathBuf::from(format!("{}.python", lock_path.display()));
    let locks = PROCESS_AUTHORITY_LOCKS.get_or_init(|| Mutex::new(BTreeMap::new()));
    let mut active = locks
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(file) = active.get(&own_path).and_then(Weak::upgrade) {
        return Ok(AuthorityLockHandle { _file: file });
    }

    let gate = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&gate_path)?;
    gate.lock_exclusive()?;
    let file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&own_path)?;
    file.lock_shared()?;
    let other = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&other_path)?;
    if let Err(error) = other.try_lock_exclusive() {
        let _ = FileExt::unlock(&file);
        let _ = FileExt::unlock(&gate);
        return Err(SqliteStoreError::AuthorityConflict {
            path: display_path.display().to_string(),
            message: error.to_string(),
        });
    }
    FileExt::unlock(&other)?;
    FileExt::unlock(&gate)?;
    let file = Arc::new(file);
    active.insert(own_path, Arc::downgrade(&file));
    Ok(AuthorityLockHandle { _file: file })
}

#[derive(Debug)]
pub(super) struct ProcessAuthorityLock {
    _path: Option<AuthorityLockHandle>,
    _file_identity: Option<AuthorityLockHandle>,
    active_path_identity: Option<(PathBuf, String)>,
}

impl ProcessAuthorityLock {
    pub(super) fn acquire(database: &Path) -> SqliteStoreResult<Self> {
        if is_private_memory_database(database) {
            return Ok(Self {
                _path: None,
                _file_identity: None,
                active_path_identity: None,
            });
        }
        let database = normalized_database_path(database);
        let lock_path = PathBuf::from(format!("{}.kogwistar-authority.lock", database.display()));
        let authority_key = normalized_authority_path_key(&database);
        let path = acquire_authority_lock_handle(lock_path, &authority_key, &database)?;
        Ok(Self {
            _path: Some(path),
            _file_identity: None,
            active_path_identity: None,
        })
    }

    pub(super) fn attach_file_identity(&mut self, database: &Path) -> SqliteStoreResult<()> {
        if self._path.is_none() {
            return Ok(());
        }
        let Some(identity_key) = file_identity_authority_key(database)? else {
            return Ok(());
        };
        let identifier = Uuid::new_v5(&Uuid::NAMESPACE_OID, identity_key.as_bytes());
        let lock_path = std::env::temp_dir().join(format!(
            "kogwistar-sqlite-authority-{}.lock",
            identifier.simple()
        ));
        let file_identity_lock = acquire_authority_lock_handle(lock_path, &identity_key, database)?;
        let database_path = normalized_database_path(database);
        register_active_database_identity(database_path.clone(), &identity_key, database)?;
        self._file_identity = Some(file_identity_lock);
        self.active_path_identity = Some((database_path, identity_key));
        Ok(())
    }
}

impl Drop for ProcessAuthorityLock {
    fn drop(&mut self) {
        let Some((path, identity)) = self.active_path_identity.take() else {
            return;
        };
        unregister_active_database_identity(&path, &identity);
    }
}

mod path_identity;
pub(super) use path_identity::*;
