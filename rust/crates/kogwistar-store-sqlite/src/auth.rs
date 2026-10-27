use super::*;

impl SqliteAuthStore {
    pub fn open(path: impl AsRef<Path>) -> SqliteStoreResult<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent().filter(|value| !value.as_os_str().is_empty()) {
            fs::create_dir_all(parent)?;
        }
        let store = Self {
            path: Arc::new(path),
        };
        let connection = store.connection()?;
        initialize_auth_schema(&connection)?;
        Ok(store)
    }

    pub fn path(&self) -> &Path {
        self.path.as_ref()
    }

    fn connection(&self) -> SqliteStoreResult<Connection> {
        let connection = Connection::open_with_flags(
            self.path.as_ref(),
            OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE,
        )?;
        connection.busy_timeout(std::time::Duration::from_millis(BUSY_TIMEOUT_MS))?;
        connection.execute_batch("PRAGMA foreign_keys = ON;")?;
        Ok(connection)
    }
}

fn sqlite_auth_user(row: &rusqlite::Row<'_>) -> rusqlite::Result<AuthUser> {
    Ok(AuthUser {
        user_id: row.get(0)?,
        email: row.get(1)?,
        display_name: row.get(2)?,
        is_active: row.get(3)?,
        global_role: row.get(4)?,
        global_ns: row.get(5)?,
    })
}

fn sqlite_store_error(error: SqliteStoreError) -> StoreError {
    StoreError::Backend {
        backend: "sqlite".to_owned(),
        message: error.to_string(),
    }
}

impl AuthIdentityStore for SqliteAuthStore {
    async fn auth_user(&self, user_id: &str) -> StoreResult<Option<AuthUser>> {
        let connection = self.connection().map_err(sqlite_store_error)?;
        connection
            .query_row(
                "SELECT user_id, email, display_name, is_active, global_role, global_ns FROM users WHERE user_id = ?1",
                [user_id],
                sqlite_auth_user,
            )
            .optional()
            .map_err(|error| sqlite_store_error(error.into()))
    }

    async fn external_identity(
        &self,
        issuer: &str,
        subject: &str,
    ) -> StoreResult<Option<ExternalIdentity>> {
        let connection = self.connection().map_err(sqlite_store_error)?;
        connection
            .query_row(
                "SELECT issuer, subject, user_id, email FROM external_identities WHERE issuer = ?1 AND subject = ?2",
                params![issuer, subject],
                |row| {
                    Ok(ExternalIdentity {
                        issuer: row.get(0)?,
                        subject: row.get(1)?,
                        user_id: row.get(2)?,
                        email: row.get(3)?,
                    })
                },
            )
            .optional()
            .map_err(|error| sqlite_store_error(error.into()))
    }

    async fn resolve_external_identity(
        &self,
        request: ResolveExternalIdentity,
    ) -> StoreResult<AuthUser> {
        let mut connection = self.connection().map_err(sqlite_store_error)?;
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|error| sqlite_store_error(error.into()))?;
        let existing_user_id = transaction
            .query_row(
                "SELECT user_id FROM external_identities WHERE issuer = ?1 AND subject = ?2",
                params![request.issuer, request.subject],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .map_err(|error| sqlite_store_error(error.into()))?;
        let user_id = if let Some(user_id) = existing_user_id {
            user_id
        } else {
            let email_user_id = transaction
                .query_row(
                    "SELECT user_id FROM users WHERE email = ?1",
                    [&request.email],
                    |row| row.get::<_, String>(0),
                )
                .optional()
                .map_err(|error| sqlite_store_error(error.into()))?;
            let user_id = email_user_id.unwrap_or_else(|| request.new_user_id.clone());
            transaction
                .execute(
                    "INSERT OR IGNORE INTO users (user_id, email, display_name, is_active, global_role, global_ns, created_at) VALUES (?1, ?2, ?3, 1, ?4, ?5, CURRENT_TIMESTAMP)",
                    params![user_id, request.email, request.display_name, request.default_role, request.default_ns],
                )
                .map_err(|error| sqlite_store_error(error.into()))?;
            transaction
                .execute(
                    "INSERT INTO external_identities (issuer, subject, user_id, email) VALUES (?1, ?2, ?3, ?4)",
                    params![request.issuer, request.subject, user_id, request.email],
                )
                .map_err(|error| sqlite_store_error(error.into()))?;
            user_id
        };
        transaction
            .execute(
                "UPDATE users SET last_login_at = CURRENT_TIMESTAMP WHERE user_id = ?1",
                [&user_id],
            )
            .map_err(|error| sqlite_store_error(error.into()))?;
        let user = transaction
            .query_row(
                "SELECT user_id, email, display_name, is_active, global_role, global_ns FROM users WHERE user_id = ?1",
                [&user_id],
                sqlite_auth_user,
            )
            .map_err(|error| sqlite_store_error(error.into()))?;
        transaction
            .commit()
            .map_err(|error| sqlite_store_error(error.into()))?;
        Ok(user)
    }
}
