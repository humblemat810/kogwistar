use super::super::*;

impl SqliteStore {
    /// Open (or create) a Python-compatible EngineSQLite database.
    ///
    /// `rusqlite` uses its `bundled` feature here so native SQLite behavior is
    /// reproducible instead of depending on a host-provided SQLite library.
    pub fn open(path: impl AsRef<Path>) -> SqliteStoreResult<Self> {
        let path = path.as_ref().to_path_buf();
        let initialization_lock = database_initialization_lock(&path);
        let _initialization_guard = initialization_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let private_memory = is_private_memory_database(&path);
        if !private_memory
            && let Some(parent) = path
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)?;
        }
        let mut authority_lock = ProcessAuthorityLock::acquire(&path)?;
        if !private_memory {
            OpenOptions::new()
                .create(true)
                .truncate(false)
                .read(true)
                .write(true)
                .open(&path)?;
            authority_lock.attach_file_identity(&path)?;
        }
        let conn = configured_connection(&path)?;
        let authority_lock = Arc::new(authority_lock);
        let schema_version: i64 = conn.query_row("PRAGMA user_version", [], |row| row.get(0))?;
        if schema_version < STORE_SCHEMA_VERSION {
            initialize_schema(&conn)?;
            conn.execute_batch(&format!("PRAGMA user_version = {STORE_SCHEMA_VERSION};"))?;
        }
        Ok(Self {
            path: Arc::new(path),
            connection: Arc::new(Mutex::new(conn)),
            _authority_lock: authority_lock,
        })
    }

    pub fn path(&self) -> &Path {
        self.path.as_ref()
    }

    /// Run a short UoW under `BEGIN IMMEDIATE`. Returning an error rolls back
    /// every sequence, event, and cursor change performed through `uow`.
    pub fn immediate_transaction<T, F>(&self, operation: F) -> SqliteStoreResult<T>
    where
        F: FnOnce(&mut SqliteUnitOfWork<'_>) -> SqliteStoreResult<T>,
    {
        let mut conn = self
            .connection
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if conn.is_autocommit() && !is_private_memory_database(self.path.as_ref()) {
            // Python EngineSQLite may commit between Rust calls. Do not start
            // an immediate transaction from the cached handle, whose read
            // snapshot can predate that external commit. Keep the store mutex
            // held while refreshing so native calls remain serialized.
            let mut fresh = configured_connection(self.path.as_ref())?;
            let transaction = fresh.transaction_with_behavior(TransactionBehavior::Immediate)?;
            let result = {
                let mut uow = SqliteUnitOfWork {
                    transaction: &transaction,
                };
                operation(&mut uow)
            };
            match result {
                Ok(value) => {
                    transaction.commit()?;
                    Ok(value)
                }
                Err(error) => Err(error),
            }
        } else {
            let savepoint = conn.savepoint()?;
            let result = {
                let mut uow = SqliteUnitOfWork {
                    transaction: &savepoint,
                };
                operation(&mut uow)
            };
            match result {
                Ok(value) => {
                    savepoint.commit()?;
                    Ok(value)
                }
                Err(error) => Err(error),
            }
        }
    }

    /// Begin a Python-facade unit of work on the cached connection. Calls made
    /// before commit/rollback join it through per-operation savepoints.
    pub fn begin_external_transaction(&self) -> SqliteStoreResult<()> {
        let conn = self
            .connection
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if !conn.is_autocommit() {
            return Err(SqliteStoreError::TransactionAborted(
                "SQLite external transaction is already active".to_owned(),
            ));
        }
        conn.execute_batch("BEGIN IMMEDIATE")?;
        sqlite_lifecycle_diagnostic("begin", self.path.as_ref());
        Ok(())
    }

    pub fn commit_external_transaction(&self) -> SqliteStoreResult<()> {
        let conn = self
            .connection
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if conn.is_autocommit() {
            return Err(SqliteStoreError::TransactionAborted(
                "SQLite external transaction is not active".to_owned(),
            ));
        }
        conn.execute_batch("COMMIT")?;
        sqlite_lifecycle_diagnostic("commit", self.path.as_ref());
        Ok(())
    }

    pub fn rollback_external_transaction(&self) -> SqliteStoreResult<()> {
        let conn = self
            .connection
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if conn.is_autocommit() {
            return Err(SqliteStoreError::TransactionAborted(
                "SQLite external transaction is not active".to_owned(),
            ));
        }
        conn.execute_batch("ROLLBACK")?;
        sqlite_lifecycle_diagnostic("rollback", self.path.as_ref());
        Ok(())
    }

    /// Alias for `immediate_transaction`; all store UoWs are immediate.
    pub fn transaction<T, F>(&self, operation: F) -> SqliteStoreResult<T>
    where
        F: FnOnce(&mut SqliteUnitOfWork<'_>) -> SqliteStoreResult<T>,
    {
        self.immediate_transaction(operation)
    }

    pub(super) fn with_connection<T>(
        &self,
        operation: impl FnOnce(&Connection) -> SqliteStoreResult<T>,
    ) -> SqliteStoreResult<T> {
        let conn = self
            .connection
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if conn.is_autocommit() && !is_private_memory_database(self.path.as_ref()) {
            drop(conn);
            // A Python EngineSQLite connection may commit between native
            // calls. A fresh read handle prevents an autocommit snapshot from
            // hiding those external commits while transaction reads still
            // use the Rust-owned connection above.
            let fresh = configured_connection(self.path.as_ref())?;
            operation(&fresh)
        } else {
            operation(&conn)
        }
    }
}
