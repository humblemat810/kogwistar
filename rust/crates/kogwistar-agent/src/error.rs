use thiserror::Error;

#[derive(Debug, Error)]
pub enum AgentError {
    #[error("validation error: {0}")]
    Validation(String),
    #[error("permission error: {0}")]
    Permission(String),
    #[error("runtime error: {0}")]
    Runtime(String),
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

pub type AgentResult<T> = Result<T, AgentError>;
