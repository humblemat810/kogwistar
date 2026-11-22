use crate::error::AgentError;
use crate::util::{canonical_json, sha256_hex};
use kogwistar_runtime::RECORDED_RUNTIME_CONTRACT_VERSION;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Compression {
    Low,
    #[default]
    Medium,
    High,
    Ultra,
}

impl std::fmt::Display for Compression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let value = match self {
            Compression::Low => "low",
            Compression::Medium => "medium",
            Compression::High => "high",
            Compression::Ultra => "ultra",
        };
        write!(f, "{value}")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContextPolicy {
    #[serde(default = "bool_true")]
    pub include_current_conversation: bool,
    #[serde(default)]
    pub allow_cross_conversation: bool,
    #[serde(default = "default_max_items")]
    pub max_items: usize,
    #[serde(default = "default_max_chars")]
    pub max_chars: usize,
    #[serde(default)]
    pub compression: Compression,
    #[serde(default)]
    pub memory_query: Option<String>,
    #[serde(default)]
    pub knowledge_query: Option<String>,
    #[serde(default)]
    pub wisdom_query: Option<String>,
}

impl Default for ContextPolicy {
    fn default() -> Self {
        Self {
            include_current_conversation: true,
            allow_cross_conversation: false,
            max_items: 32,
            max_chars: 32_000,
            compression: Compression::Medium,
            memory_query: None,
            knowledge_query: None,
            wisdom_query: None,
        }
    }
}

impl ContextPolicy {
    pub fn validate(&self) -> Result<(), AgentError> {
        if self.max_items < 1 || self.max_chars < 1 {
            return Err(AgentError::Validation(
                "context bounds must be positive".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContextSnapshot {
    #[serde(default = "default_policy_version")]
    pub policy_version: String,
    #[serde(default)]
    pub conversation_id: Option<String>,
    #[serde(default)]
    pub items: Vec<serde_json::Value>,
    #[serde(default)]
    pub source_refs: Vec<String>,
    #[serde(default)]
    pub truncated: bool,
    #[serde(default = "default_compression")]
    pub compression: String,
}

impl Default for ContextSnapshot {
    fn default() -> Self {
        Self {
            policy_version: default_policy_version(),
            conversation_id: None,
            items: Vec::new(),
            source_refs: Vec::new(),
            truncated: false,
            compression: default_compression(),
        }
    }
}

impl ContextSnapshot {
    pub fn stable_fingerprint(&self) -> String {
        let value = serde_json::to_value(self).unwrap_or(serde_json::Value::Null);
        sha256_hex(canonical_json(&value).as_bytes())
    }
}

fn bool_true() -> bool {
    true
}

fn default_max_items() -> usize {
    32
}

fn default_max_chars() -> usize {
    32_000
}

fn default_policy_version() -> String {
    format!("v{RECORDED_RUNTIME_CONTRACT_VERSION}")
}

fn default_compression() -> String {
    "medium".to_owned()
}
