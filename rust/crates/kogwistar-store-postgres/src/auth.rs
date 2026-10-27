//! PostgreSQL identity-store adapter, isolated from graph/event persistence.

use super::*;

/// Independent Python-compatible PostgreSQL `AUTH_DB_URL` store. Relations
/// remain unqualified so PostgreSQL DSN `search_path` keeps deployment behavior.
#[derive(Clone)]
pub struct PostgresAuthStore {
    pool: Pool,
}

impl std::fmt::Debug for PostgresAuthStore {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PostgresAuthStore")
            .finish_non_exhaustive()
    }
}

impl PostgresAuthStore {
    pub fn from_config(config: Config) -> PostgresStoreResult<Self> {
        let manager = Manager::new(config, NoTls);
        let pool = Pool::builder(manager)
            .build()
            .map_err(|error| PostgresStoreError::Backend(error.to_string()))?;
        Ok(Self { pool })
    }

    pub fn from_dsn(dsn: &str) -> PostgresStoreResult<Self> {
        let config = dsn
            .parse::<Config>()
            .map_err(|error| PostgresStoreError::Backend(error.to_string()))?;
        Self::from_config(config)
    }

    pub async fn ensure_schema(&self) -> PostgresStoreResult<()> {
        let client = self.client().await?;
        client
            .batch_execute(
                "CREATE TABLE IF NOT EXISTS users (
                    user_id VARCHAR(64) NOT NULL PRIMARY KEY,
                    email VARCHAR(255) NOT NULL UNIQUE,
                    display_name VARCHAR(255),
                    is_active BOOLEAN NOT NULL,
                    global_role VARCHAR(32),
                    global_ns VARCHAR(255),
                    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
                    last_login_at TIMESTAMP WITHOUT TIME ZONE
                );
                CREATE INDEX IF NOT EXISTS ix_users_email ON users (email);
                CREATE TABLE IF NOT EXISTS external_identities (
                    issuer VARCHAR(255) NOT NULL,
                    subject VARCHAR(255) NOT NULL,
                    user_id VARCHAR(64) NOT NULL REFERENCES users(user_id),
                    email VARCHAR(255),
                    PRIMARY KEY (issuer, subject)
                );
                CREATE INDEX IF NOT EXISTS ix_external_identities_user_id ON external_identities (user_id);
                CREATE TABLE IF NOT EXISTS workflow_acl (
                    workflow_id VARCHAR(255) NOT NULL,
                    user_id VARCHAR(64) NOT NULL REFERENCES users(user_id),
                    role VARCHAR(32) NOT NULL,
                    PRIMARY KEY (workflow_id, user_id)
                );",
            )
            .await
            .map_err(backend)
    }

    async fn client(&self) -> PostgresStoreResult<deadpool_postgres::Object> {
        self.pool
            .get()
            .await
            .map_err(|error| PostgresStoreError::Backend(error.to_string()))
    }
}

fn user_from_row(row: &Row) -> AuthUser {
    AuthUser {
        user_id: row.get(0),
        email: row.get(1),
        display_name: row.get(2),
        is_active: row.get(3),
        global_role: row.get(4),
        global_ns: row.get(5),
    }
}

fn auth_store_error(error: PostgresStoreError) -> StoreError {
    StoreError::Backend {
        backend: "postgres".to_owned(),
        message: error.to_string(),
    }
}

impl AuthIdentityStore for PostgresAuthStore {
    async fn auth_user(&self, user_id: &str) -> StoreResult<Option<AuthUser>> {
        let client = self.client().await.map_err(auth_store_error)?;
        client
            .query_opt(
                "SELECT user_id, email, display_name, is_active, global_role, global_ns FROM users WHERE user_id = $1",
                &[&user_id],
            )
            .await
            .map(|row| row.as_ref().map(user_from_row))
            .map_err(|error| auth_store_error(backend(error)))
    }

    async fn external_identity(
        &self,
        issuer: &str,
        subject: &str,
    ) -> StoreResult<Option<ExternalIdentity>> {
        let client = self.client().await.map_err(auth_store_error)?;
        client
            .query_opt(
                "SELECT issuer, subject, user_id, email FROM external_identities WHERE issuer = $1 AND subject = $2",
                &[&issuer, &subject],
            )
            .await
            .map(|row| {
                row.map(|row| ExternalIdentity {
                    issuer: row.get(0),
                    subject: row.get(1),
                    user_id: row.get(2),
                    email: row.get(3),
                })
            })
            .map_err(|error| auth_store_error(backend(error)))
    }

    async fn resolve_external_identity(
        &self,
        request: ResolveExternalIdentity,
    ) -> StoreResult<AuthUser> {
        let mut client = self.client().await.map_err(auth_store_error)?;
        let transaction = client
            .transaction()
            .await
            .map_err(|error| auth_store_error(backend(error)))?;
        transaction
            .query_one(
                "SELECT pg_advisory_xact_lock(hashtextextended($1::text, hashtextextended($2::text, 0)))",
                &[&request.issuer, &request.subject],
            )
            .await
            .map_err(|error| auth_store_error(backend(error)))?;
        let existing_identity = transaction
            .query_opt(
                "SELECT user_id FROM external_identities WHERE issuer = $1 AND subject = $2 FOR UPDATE",
                &[&request.issuer, &request.subject],
            )
            .await
            .map_err(|error| auth_store_error(backend(error)))?;
        let user_id = if let Some(row) = existing_identity {
            row.get::<_, String>(0)
        } else {
            transaction
                .execute(
                    "INSERT INTO users (user_id, email, display_name, is_active, global_role, global_ns, created_at) VALUES ($1, $2, $3, TRUE, $4, $5, CURRENT_TIMESTAMP) ON CONFLICT (email) DO NOTHING",
                    &[&request.new_user_id, &request.email, &request.display_name, &request.default_role, &request.default_ns],
                )
                .await
                .map_err(|error| auth_store_error(backend(error)))?;
            let user_id = transaction
                .query_one(
                    "SELECT user_id FROM users WHERE email = $1",
                    &[&request.email],
                )
                .await
                .map_err(|error| auth_store_error(backend(error)))?
                .get::<_, String>(0);
            transaction
                .execute(
                    "INSERT INTO external_identities (issuer, subject, user_id, email) VALUES ($1, $2, $3, $4) ON CONFLICT (issuer, subject) DO NOTHING",
                    &[&request.issuer, &request.subject, &user_id, &request.email],
                )
                .await
                .map_err(|error| auth_store_error(backend(error)))?;
            transaction
                .query_one(
                    "SELECT user_id FROM external_identities WHERE issuer = $1 AND subject = $2",
                    &[&request.issuer, &request.subject],
                )
                .await
                .map_err(|error| auth_store_error(backend(error)))?
                .get::<_, String>(0)
        };
        let row = transaction
            .query_one(
                "UPDATE users SET last_login_at = CURRENT_TIMESTAMP WHERE user_id = $1 RETURNING user_id, email, display_name, is_active, global_role, global_ns",
                &[&user_id],
            )
            .await
            .map_err(|error| auth_store_error(backend(error)))?;
        let user = user_from_row(&row);
        transaction
            .commit()
            .await
            .map_err(|error| auth_store_error(backend(error)))?;
        Ok(user)
    }
}
