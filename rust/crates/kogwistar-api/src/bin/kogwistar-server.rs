use std::sync::Arc;

use kogwistar_api::{
    ApiState, AuthConfig, HealthSnapshot, ImplementationSnapshot, PostgresRunApplicationService,
    SqliteRunApplicationService, UnavailableApplicationService, router_with_application,
};

fn environment(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_owned())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let host = environment("KOGWISTAR_SERVER_HOST", "127.0.0.1");
    let port = environment("KOGWISTAR_SERVER_PORT", "8000");
    let listener = tokio::net::TcpListener::bind(format!("{host}:{port}")).await?;
    let required_roles = environment("KOGWISTAR_SERVER_REQUIRED_ROLES", "")
        .split(',')
        .map(str::trim)
        .filter(|role| !role.is_empty())
        .map(str::to_owned)
        .collect();
    let backend = environment("KOGWISTAR_BACKEND", "in_memory");
    let state = ApiState {
        health: HealthSnapshot {
            backend: backend.clone(),
            persist_directory: environment("KOGWISTAR_PERSIST_DIRECTORY", ".kogwistar"),
            conversation_persist_directory: environment(
                "KOGWISTAR_CONVERSATION_PERSIST_DIRECTORY",
                ".kogwistar/conversation",
            ),
            workflow_persist_directory: environment(
                "KOGWISTAR_WORKFLOW_PERSIST_DIRECTORY",
                ".kogwistar/workflow",
            ),
            wisdom_persist_directory: environment(
                "KOGWISTAR_WISDOM_PERSIST_DIRECTORY",
                ".kogwistar/wisdom",
            ),
            pg_schema_base: (backend == "pg")
                .then(|| environment("KOGWISTAR_PG_SCHEMA_BASE", "kogwistar")),
        },
        required_roles,
        implementation: ImplementationSnapshot::default(),
        auth: AuthConfig::from_environment(),
    };
    let application: Arc<dyn kogwistar_api::ApplicationService> = if backend == "pg" {
        match std::env::var("KOGWISTAR_PG_DSN") {
            Ok(dsn) if !dsn.trim().is_empty() => {
                let service = PostgresRunApplicationService::from_dsn(
                    &dsn,
                    &environment("KOGWISTAR_PG_SCHEMA_BASE", "kogwistar"),
                )
                .map_err(std::io::Error::other)?;
                service
                    .ensure_schema()
                    .await
                    .map_err(std::io::Error::other)?;
                Arc::new(service)
            }
            _ => Arc::new(UnavailableApplicationService),
        }
    } else {
        match std::env::var("KOGWISTAR_META_SQLITE_PATH") {
            Ok(path) if !path.trim().is_empty() => {
                Arc::new(SqliteRunApplicationService::open(path).map_err(std::io::Error::other)?)
            }
            _ => Arc::new(UnavailableApplicationService),
        }
    };
    axum::serve(listener, router_with_application(state, application)).await?;
    Ok(())
}
