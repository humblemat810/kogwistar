use super::*;

pub(super) fn initialize_auth_schema(conn: &Connection) -> SqliteStoreResult<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS users (
            user_id VARCHAR(64) NOT NULL PRIMARY KEY,
            email VARCHAR(255) NOT NULL UNIQUE,
            display_name VARCHAR(255),
            is_active BOOLEAN NOT NULL,
            global_role VARCHAR(32),
            global_ns VARCHAR(255),
            created_at DATETIME NOT NULL,
            last_login_at DATETIME
        );
        CREATE INDEX IF NOT EXISTS ix_users_email ON users (email);
        CREATE TABLE IF NOT EXISTS external_identities (
            issuer VARCHAR(255) NOT NULL,
            subject VARCHAR(255) NOT NULL,
            user_id VARCHAR(64) NOT NULL,
            email VARCHAR(255),
            PRIMARY KEY (issuer, subject),
            FOREIGN KEY(user_id) REFERENCES users (user_id)
        );
        CREATE INDEX IF NOT EXISTS ix_external_identities_user_id ON external_identities (user_id);
        CREATE TABLE IF NOT EXISTS workflow_acl (
            workflow_id VARCHAR(255) NOT NULL,
            user_id VARCHAR(64) NOT NULL,
            role VARCHAR(32) NOT NULL,
            PRIMARY KEY (workflow_id, user_id),
            FOREIGN KEY(user_id) REFERENCES users (user_id)
        );",
    )?;
    Ok(())
}
