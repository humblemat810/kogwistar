//! Durable SQLite implementation of the Phase-3 event-store slice.
//!
//! Schema names and event semantics intentionally match Python's
//! `EngineSQLite`; this crate does not own queues, projections, or graph data.

#[cfg(not(windows))]
use fs2::FileExt;
use kogwistar_contracts::EntityEventEnvelope;
use kogwistar_engine::EntityProjection;
use kogwistar_runtime::{
    PersistedRecordedTransition, RECORDED_RUNTIME_CONTRACT_VERSION, RecordedRuntimeError,
    RecordedRuntimeState, RecordedRuntimeTransition, RecordedTransitionResult,
    RecordedWorkerHandoff, RecordedWorkerSuccessEffect, RuntimeFrontier, RuntimeStepExecutePayload,
    RuntimeStepExecuteRequest, RuntimeWorkerEffectStatus, frontier_after_worker_resume,
    frontier_after_worker_success, frontier_after_worker_suspend, reduce_recorded_transition,
    transition_digest, worker_effect_digest,
};
use kogwistar_store::{
    AcceptedIndexJobResult, AppendedEvent, AuthIdentityStore, AuthUser, EntityEvent,
    EntityRebuildRequest, EntityRecoveryReport, EntityRecoveryRequest, EventPruneStore,
    EventReadStore, EventWriteStore, ExternalIdentity, IndexJob, IndexJobReadStore,
    IndexJobWriteStore, LaneMessageClaimFilter, LaneMessageFilter, LaneMessageReadStore,
    LaneMessageWriteStore, NamedProjection, NamedProjectionWrite, NewEntityEvent, NewIndexJob,
    NewProjectedLaneMessage, ProjectedLaneMessage, ProjectionReadStore, ProjectionWriteStore,
    RUNTIME_CURRENT_STATE_NAMESPACE, ReplayCursor, ResolveExternalIdentity, ServerRun,
    ServerRunCreate, ServerRunEvent, ServerRunReadStore, ServerRunUpdate, ServerRunWriteStore,
    StoreError, StoreResult, WorkflowDesignDelta, WorkflowDesignDeltaWrite,
    WorkflowDesignHistoryReadStore, WorkflowDesignHistoryWriteStore, WorkflowDesignSnapshot,
    WorkflowDesignSnapshotWrite, runtime_checkpoint_namespace, runtime_projection_write,
    runtime_status_namespace, validate_entity_rebuild_request, validate_entity_recovery_request,
};
use rusqlite::{Connection, OpenFlags, OptionalExtension, TransactionBehavior, params};
use serde_json::{Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::CStr;
use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;
use uuid::Uuid;

const BUSY_TIMEOUT_MS: u64 = 30_000;
// Bump this when initialize_schema gains a migration. The marker lets ordinary
// opens avoid taking a schema write lock after initialization, including while
// another connection has an active data transaction.
const STORE_SCHEMA_VERSION: i64 = 1;
static NEXT_CLAIM_TOKEN: AtomicU64 = AtomicU64::new(0);

static PROCESS_AUTHORITY_LOCKS: OnceLock<Mutex<BTreeMap<PathBuf, Weak<File>>>> = OnceLock::new();
static ACTIVE_DATABASE_IDENTITIES: OnceLock<Mutex<BTreeMap<PathBuf, (String, usize)>>> =
    OnceLock::new();
static DATABASE_INITIALIZATION_LOCKS: OnceLock<Mutex<BTreeMap<PathBuf, Weak<Mutex<()>>>>> =
    OnceLock::new();

fn database_initialization_lock(database: &Path) -> Arc<Mutex<()>> {
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

fn register_active_database_identity(
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

fn unregister_active_database_identity(path: &Path, identity: &str) {
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
struct ProcessAuthorityLock {
    _path: Option<AuthorityLockHandle>,
    _file_identity: Option<AuthorityLockHandle>,
    active_path_identity: Option<(PathBuf, String)>,
}

impl ProcessAuthorityLock {
    fn acquire(database: &Path) -> SqliteStoreResult<Self> {
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

    fn attach_file_identity(&mut self, database: &Path) -> SqliteStoreResult<()> {
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

fn is_private_memory_database(database: &Path) -> bool {
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

/// Errors for direct SQLite APIs. Trait methods retain `kogwistar-store`'s
/// existing `StoreError` contract for its validation and cursor errors.
#[derive(Debug, Error)]
pub enum SqliteStoreError {
    #[error("SQLite operation failed: {0}")]
    Sql(#[from] rusqlite::Error),
    #[error("filesystem operation failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("SQLite database authority is held by another process for {path}: {message}")]
    AuthorityConflict { path: String, message: String },
    #[error("namespace must not be empty")]
    EmptyNamespace,
    #[error(
        "event_id {event_id:?} already belongs to namespace {existing_namespace:?}, not {requested_namespace:?}"
    )]
    EventIdNamespaceCollision {
        event_id: String,
        existing_namespace: String,
        requested_namespace: String,
    },
    #[error("event payload is not valid JSON: {0}")]
    InvalidPayload(#[from] serde_json::Error),
    #[error("sequence value must not be negative: {value}")]
    NegativeSequenceValue { value: i64 },
    #[error("transaction aborted: {0}")]
    TransactionAborted(String),
    #[error("SQLite cached session belongs to another execution context")]
    ContextMismatch,
    #[error("invalid recorded runtime transition: {0}")]
    RecordedRuntime(#[from] RecordedRuntimeError),
    #[error("recorded runtime transition conflict: {0}")]
    RecordedRuntimeConflict(String),
    #[error(transparent)]
    Store(#[from] StoreError),
}

/// One compare-and-swap mutation for an atomic named-projection batch.
pub struct NamedProjectionCas {
    pub namespace: String,
    pub key: String,
    pub expected_last_authoritative_seq: Option<i64>,
    pub expected_last_materialized_seq: Option<i64>,
    pub projection: NamedProjectionWrite,
}

pub type SqliteStoreResult<T> = Result<T, SqliteStoreError>;

/// Event row with the raw SQLite JSON text retained byte-for-byte as stored.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawEntityEvent {
    pub namespace: String,
    pub seq: i64,
    pub event_id: String,
    pub entity_kind: String,
    pub entity_id: String,
    pub op: String,
    pub payload_json: String,
    pub created_at: i64,
}

/// New event for callers that must retain Python's original `payload_json` text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NewRawEntityEvent {
    pub event_id: String,
    pub entity_kind: String,
    pub entity_id: String,
    pub op: String,
    pub payload_json: String,
}

/// An event imported from a portable archive with authoritative identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RestoredRawEntityEvent {
    pub seq: i64,
    pub event_id: String,
    pub entity_kind: String,
    pub entity_id: String,
    pub op: String,
    pub payload_json: String,
    pub created_at: i64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AppendedRawEvent {
    pub event: RawEntityEvent,
    pub inserted: bool,
}

/// Cloneable durable store handle. Clones share one serialized SQLite
/// connection; SQLite is a single-writer store and repeated open/WAL teardown
/// otherwise dominates short authoritative operations.
#[derive(Clone, Debug)]
pub struct SqliteStore {
    path: Arc<PathBuf>,
    connection: Arc<Mutex<Connection>>,
    _authority_lock: Arc<ProcessAuthorityLock>,
}

impl Drop for SqliteStore {
    fn drop(&mut self) {
        if std::env::var_os("KOGWISTAR_SQLITE_DIAGNOSTICS").as_deref()
            != Some(std::ffi::OsStr::new("1"))
            || Arc::strong_count(&self.connection) != 1
        {
            return;
        }
        let connection = self
            .connection
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        eprintln!(
            "[sqlite-rust-close] conn={:p} path={} {}",
            unsafe { connection.handle() },
            self.path.display(),
            sqlite_sidecars(self.path.as_ref())
        );
    }
}

/// Independent Python-compatible `AUTH_DB_URL=sqlite:///...` store. Opening
/// it creates only auth relations and never engine metadata relations.
#[derive(Clone, Debug)]
pub struct SqliteAuthStore {
    path: Arc<PathBuf>,
}

mod auth;
mod connection_uow;
mod events_recovery;
mod index_jobs;
mod lane_messages;
mod projections;
mod row_mapping;
mod schema;
mod store;
mod trait_impls;
mod workflow_runtime;

pub use connection_uow::SqliteUnitOfWork;
use connection_uow::*;
use events_recovery::*;
use index_jobs::*;
use lane_messages::*;
use projections::*;
use row_mapping::*;
use schema::*;
use workflow_runtime::*;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

fn unix_epoch_seconds() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is before Unix epoch")
        .as_secs() as i64
}

fn unix_epoch_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock precedes Unix epoch")
        .as_millis() as i64
}

fn require_trait_namespace(namespace: &str) -> StoreResult<()> {
    if namespace.is_empty() {
        Err(StoreError::EmptyNamespace)
    } else {
        Ok(())
    }
}

fn trait_error(error: SqliteStoreError) -> StoreError {
    match error {
        SqliteStoreError::Store(error) => error,
        SqliteStoreError::EmptyNamespace => StoreError::EmptyNamespace,
        SqliteStoreError::EventIdNamespaceCollision {
            event_id,
            existing_namespace,
            requested_namespace,
        } => StoreError::EventIdNamespaceCollision {
            event_id,
            existing_namespace,
            requested_namespace,
        },
        error => StoreError::Backend {
            backend: "sqlite".to_owned(),
            message: error.to_string(),
        },
    }
}
