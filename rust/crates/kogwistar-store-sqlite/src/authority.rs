use super::*;

#[cfg(not(windows))]
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

#[cfg(windows)]
static WINDOWS_PROCESS_AUTHORITY_MUTEXES: OnceLock<
    Mutex<BTreeMap<String, Weak<WindowsAuthorityMutex>>>,
> = OnceLock::new();

#[cfg(windows)]
#[derive(Debug)]
struct WindowsAuthorityMutex {
    handle: winapi::shared::ntdef::HANDLE,
}

#[cfg(windows)]
unsafe impl Send for WindowsAuthorityMutex {}

#[cfg(windows)]
unsafe impl Sync for WindowsAuthorityMutex {}

#[cfg(windows)]
impl Drop for WindowsAuthorityMutex {
    fn drop(&mut self) {
        unsafe {
            winapi::um::handleapi::CloseHandle(self.handle);
        }
    }
}

#[cfg(windows)]
fn acquire_windows_authority_mutex(
    authority_key: &str,
    display_path: &Path,
) -> SqliteStoreResult<Arc<WindowsAuthorityMutex>> {
    let key = format!(
        "Local\\KogwistarSQLiteAuthority-{}",
        Uuid::new_v5(&Uuid::NAMESPACE_OID, authority_key.as_bytes())
    );
    let mutexes = WINDOWS_PROCESS_AUTHORITY_MUTEXES.get_or_init(|| Mutex::new(BTreeMap::new()));
    let mut active = mutexes
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(mutex) = active.get(&key).and_then(Weak::upgrade) {
        return Ok(mutex);
    }

    let name: Vec<u16> = key.encode_utf16().chain(std::iter::once(0)).collect();
    let existing = unsafe { winapi::um::synchapi::OpenMutexW(0x001F0001, 0, name.as_ptr()) };
    if !existing.is_null() {
        unsafe {
            winapi::um::handleapi::CloseHandle(existing);
        }
        return Err(SqliteStoreError::AuthorityConflict {
            path: display_path.display().to_string(),
            message: "named Windows authority mutex is already owned".to_owned(),
        });
    }
    let handle =
        unsafe { winapi::um::synchapi::CreateMutexW(std::ptr::null_mut(), 1, name.as_ptr()) };
    if handle.is_null() {
        return Err(std::io::Error::last_os_error().into());
    }
    let already_exists = unsafe { winapi::um::errhandlingapi::GetLastError() }
        == winapi::shared::winerror::ERROR_ALREADY_EXISTS;
    if already_exists {
        unsafe {
            winapi::um::handleapi::CloseHandle(handle);
        }
        return Err(SqliteStoreError::AuthorityConflict {
            path: display_path.display().to_string(),
            message: "named Windows authority mutex is already owned".to_owned(),
        });
    }
    let mutex = Arc::new(WindowsAuthorityMutex { handle });
    active.insert(key, Arc::downgrade(&mutex));
    Ok(mutex)
}

#[derive(Debug)]
struct AuthorityLockHandle {
    _file: Arc<File>,
    #[cfg(windows)]
    _mutex: Arc<WindowsAuthorityMutex>,
}

fn acquire_authority_lock_handle(
    lock_path: PathBuf,
    _authority_key: &str,
    display_path: &Path,
) -> SqliteStoreResult<AuthorityLockHandle> {
    let locks = PROCESS_AUTHORITY_LOCKS.get_or_init(|| Mutex::new(BTreeMap::new()));
    let mut active = locks
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(file) = active.get(&lock_path).and_then(Weak::upgrade) {
        #[cfg(windows)]
        {
            let mutex = acquire_windows_authority_mutex(_authority_key, display_path)?;
            return Ok(AuthorityLockHandle {
                _file: file,
                _mutex: mutex,
            });
        }
        #[cfg(not(windows))]
        return Ok(AuthorityLockHandle { _file: file });
    }

    let file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&lock_path)?;
    #[cfg(windows)]
    {
        let mutex = acquire_windows_authority_mutex(_authority_key, display_path)?;
        let file = Arc::new(file);
        active.insert(lock_path, Arc::downgrade(&file));
        Ok(AuthorityLockHandle {
            _file: file,
            _mutex: mutex,
        })
    }
    #[cfg(not(windows))]
    {
        file.try_lock_exclusive()
            .map_err(|error| SqliteStoreError::AuthorityConflict {
                path: display_path.display().to_string(),
                message: error.to_string(),
            })?;
        let file = Arc::new(file);
        active.insert(lock_path, Arc::downgrade(&file));
        Ok(AuthorityLockHandle { _file: file })
    }
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

pub(super) fn is_private_memory_database(database: &Path) -> bool {
    database.as_os_str().is_empty() || database == Path::new(":memory:")
}

fn normalized_authority_path_key(database: &Path) -> String {
    let database_string = database.to_string_lossy().into_owned();
    #[cfg(windows)]
    {
        database_string
            .strip_prefix(r"\\?\")
            .unwrap_or(&database_string)
            .to_ascii_lowercase()
    }
    #[cfg(not(windows))]
    {
        database_string
    }
}

fn file_identity_authority_key(database: &Path) -> SqliteStoreResult<Option<String>> {
    let _metadata = match fs::metadata(database) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        use winapi::shared::minwindef::FALSE;
        use winapi::um::fileapi::{
            BY_HANDLE_FILE_INFORMATION, CreateFileW, GetFileInformationByHandle, OPEN_EXISTING,
        };
        use winapi::um::handleapi::{CloseHandle, INVALID_HANDLE_VALUE};
        use winapi::um::winbase::FILE_FLAG_BACKUP_SEMANTICS;
        use winapi::um::winnt::{
            FILE_READ_ATTRIBUTES, FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE,
        };

        let path = database
            .as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect::<Vec<_>>();
        let handle = unsafe {
            CreateFileW(
                path.as_ptr(),
                FILE_READ_ATTRIBUTES,
                FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                std::ptr::null_mut(),
                OPEN_EXISTING,
                FILE_FLAG_BACKUP_SEMANTICS,
                std::ptr::null_mut(),
            )
        };
        if handle == INVALID_HANDLE_VALUE {
            return Err(std::io::Error::last_os_error().into());
        }
        let mut information: BY_HANDLE_FILE_INFORMATION = unsafe { std::mem::zeroed() };
        let result = unsafe { GetFileInformationByHandle(handle, &mut information) };
        let error = if result == FALSE {
            Some(std::io::Error::last_os_error())
        } else {
            None
        };
        unsafe {
            CloseHandle(handle);
        }
        if let Some(error) = error {
            return Err(error.into());
        }
        let file_index =
            (u64::from(information.nFileIndexHigh) << 32) | u64::from(information.nFileIndexLow);
        Ok(Some(format!(
            "file-id:{:x}:{file_index:x}",
            information.dwVolumeSerialNumber
        )))
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        Ok(Some(format!(
            "file-id:{:x}:{:x}",
            _metadata.dev(),
            _metadata.ino()
        )))
    }
    #[cfg(not(any(windows, unix)))]
    {
        let _ = _metadata;
        Ok(None)
    }
}

fn normalized_database_path(database: &Path) -> PathBuf {
    let absolute = if database.is_absolute() {
        database.to_path_buf()
    } else {
        std::env::current_dir()
            .map(|directory| directory.join(database))
            .unwrap_or_else(|_| database.to_path_buf())
    };
    if let Ok(canonical) = fs::canonicalize(&absolute) {
        return canonical;
    }
    let Some(file_name) = absolute.file_name() else {
        return absolute;
    };
    absolute
        .parent()
        .and_then(|parent| fs::canonicalize(parent).ok())
        .map(|parent| parent.join(file_name))
        .unwrap_or(absolute)
}
