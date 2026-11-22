use crate::error::{AgentError, AgentResult};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

pub type HookCallback = Arc<dyn Fn(&Value) -> Result<Value, AgentError> + Send + Sync>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HookFailureMode {
    FailOpen,
    FailClosed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HookEffect {
    Observe,
    Annotate,
}

#[derive(Clone)]
pub struct HookSpec {
    pub hook_id: String,
    pub callback: HookCallback,
    pub order: i64,
    pub timeout_ms: u64,
    pub failure_mode: HookFailureMode,
    pub effect: HookEffect,
    pub required_capabilities: Vec<String>,
}

impl std::fmt::Debug for HookSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HookSpec")
            .field("hook_id", &self.hook_id)
            .field("order", &self.order)
            .field("timeout_ms", &self.timeout_ms)
            .field("failure_mode", &self.failure_mode)
            .field("effect", &self.effect)
            .field("required_capabilities", &self.required_capabilities)
            .finish()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookResult {
    pub hook_id: String,
    pub status: String,
    pub annotations: serde_json::Map<String, Value>,
    pub error: Option<String>,
}

#[derive(Default)]
pub struct HookRegistry {
    hooks: RwLock<HashMap<String, HookSpec>>,
}

impl HookRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&self, spec: HookSpec) -> Result<HookSpec, AgentError> {
        if spec.hook_id.trim().is_empty() {
            return Err(AgentError::Validation(
                "hook_id must be non-empty".to_owned(),
            ));
        }
        if spec.timeout_ms < 1 {
            return Err(AgentError::Validation(
                "hook timeout must be positive".to_owned(),
            ));
        }
        let mut hooks = self.hooks.write().unwrap();
        if hooks.contains_key(&spec.hook_id) {
            return Err(AgentError::Validation(format!(
                "hook already registered: {}",
                spec.hook_id
            )));
        }
        hooks.insert(spec.hook_id.clone(), spec.clone());
        Ok(spec)
    }

    pub fn unregister(&self, hook_id: &str) {
        self.hooks.write().unwrap().remove(hook_id);
    }

    pub fn list(&self) -> Vec<HookSpec> {
        let mut hooks: Vec<HookSpec> = self.hooks.read().unwrap().values().cloned().collect();
        hooks.sort_by_key(|a| (a.order, a.hook_id.clone()));
        hooks
    }

    pub fn run(
        &self,
        payload: &Value,
        effective_capabilities: &[String],
    ) -> AgentResult<Vec<HookResult>> {
        let capabilities: std::collections::BTreeSet<String> =
            effective_capabilities.iter().cloned().collect();
        let mut results = Vec::new();
        for spec in self.list() {
            let allowed = spec
                .required_capabilities
                .iter()
                .all(|item| capabilities.contains(item));
            if !allowed {
                results.push(HookResult {
                    hook_id: spec.hook_id,
                    status: "skipped".to_owned(),
                    annotations: serde_json::Map::new(),
                    error: Some("capability denied".to_owned()),
                });
                continue;
            }
            match (spec.callback)(payload) {
                Ok(Value::Object(annotations)) => results.push(HookResult {
                    hook_id: spec.hook_id,
                    status: "applied".to_owned(),
                    annotations,
                    error: None,
                }),
                Ok(_) => results.push(HookResult {
                    hook_id: spec.hook_id,
                    status: "applied".to_owned(),
                    annotations: serde_json::Map::new(),
                    error: None,
                }),
                Err(error) if spec.failure_mode == HookFailureMode::FailClosed => {
                    return Err(error);
                }
                Err(error) => results.push(HookResult {
                    hook_id: spec.hook_id,
                    status: "failed".to_owned(),
                    annotations: serde_json::Map::new(),
                    error: Some(error.to_string()),
                }),
            }
        }
        Ok(results)
    }
}
