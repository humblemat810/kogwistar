use crate::error::{AgentError, AgentResult};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DelegationSpec {
    pub model_profile: String,
    #[serde(default)]
    pub requested_capabilities: Vec<String>,
    #[serde(default)]
    pub context_keys: Vec<String>,
    #[serde(default = "default_max_output_bytes")]
    pub max_output_bytes: usize,
    #[serde(default)]
    pub max_steps: Option<i64>,
    #[serde(default)]
    pub max_model_calls: Option<i64>,
    #[serde(default)]
    pub max_tokens: Option<i64>,
    #[serde(default)]
    pub max_cost: Option<f64>,
}

impl Default for DelegationSpec {
    fn default() -> Self {
        Self {
            model_profile: String::new(),
            requested_capabilities: Vec::new(),
            context_keys: Vec::new(),
            max_output_bytes: default_max_output_bytes(),
            max_steps: None,
            max_model_calls: None,
            max_tokens: None,
            max_cost: None,
        }
    }
}

impl DelegationSpec {
    pub fn validate(&self) -> AgentResult<()> {
        if self.model_profile.trim().is_empty() {
            return Err(AgentError::Validation(
                "child model_profile must be non-empty".to_owned(),
            ));
        }
        if self.max_output_bytes < 1 {
            return Err(AgentError::Validation(
                "max_output_bytes must be positive".to_owned(),
            ));
        }
        for (name, value) in [
            ("max_steps", self.max_steps),
            ("max_model_calls", self.max_model_calls),
            ("max_tokens", self.max_tokens),
        ] {
            if value.is_some_and(|item| item < 1) {
                return Err(AgentError::Validation(format!("{name} must be positive")));
            }
        }
        if self.max_cost.is_some_and(|item| item <= 0.0) {
            return Err(AgentError::Validation(
                "max_cost must be positive".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct DelegationContext {
    pub run_id: String,
    pub step_seq: i64,
    pub conversation_id: Option<String>,
    pub turn_node_id: Option<String>,
    pub state_view: Map<String, Value>,
}

pub fn delegated_initial_state(
    parent_state: &Map<String, Value>,
    spec: &DelegationSpec,
) -> AgentResult<crate::AgentState> {
    spec.validate()?;
    let parent_caps = match parent_state.get("effective_capabilities") {
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(|item| item.as_str().map(str::to_owned))
            .collect::<BTreeSet<String>>(),
        _ => BTreeSet::new(),
    };
    let requested: BTreeSet<String> = spec.requested_capabilities.iter().cloned().collect();
    let missing: Vec<String> = requested.difference(&parent_caps).cloned().collect();
    if !missing.is_empty() {
        return Err(AgentError::Permission(format!(
            "child requested capabilities absent from parent: {}",
            missing.join(",")
        )));
    }
    let mut child = Map::new();
    child.insert(
        "agent_model_profile".to_owned(),
        Value::String(spec.model_profile.clone()),
    );
    child.insert(
        "effective_capabilities".to_owned(),
        Value::Array(requested.into_iter().map(Value::String).collect()),
    );
    for key in &spec.context_keys {
        if !key.starts_with('_')
            && let Some(value) = parent_state.get(key)
        {
            child.insert(key.clone(), value.clone());
        }
    }
    for (key, value) in [
        ("step_budget", spec.max_steps.map(Value::from)),
        ("call_budget", spec.max_model_calls.map(Value::from)),
        ("token_budget", spec.max_tokens.map(Value::from)),
        ("cost_budget", spec.max_cost.map(Value::from)),
    ] {
        if let Some(value) = value {
            child.insert(key.to_owned(), value);
        }
    }
    Ok(child)
}

pub fn bounded_child_result(value: &Value, max_bytes: usize) -> AgentResult<Map<String, Value>> {
    if max_bytes < 1 {
        return Err(AgentError::Validation(
            "max_bytes must be positive".to_owned(),
        ));
    }
    let payload = if value.is_object() {
        value.clone()
    } else {
        serde_json::json!({"value": value})
    };
    let encoded = serde_json::to_vec(&payload)?;
    let digest = {
        let mut hasher = Sha256::new();
        hasher.update(&encoded);
        format!("{:x}", hasher.finalize())
    };
    let mut result = Map::new();
    if encoded.len() <= max_bytes {
        result.insert("result".to_owned(), payload);
        result.insert("truncated".to_owned(), Value::Bool(false));
        result.insert("digest".to_owned(), Value::String(digest));
    } else {
        result.insert("result".to_owned(), Value::Null);
        result.insert("truncated".to_owned(), Value::Bool(true));
        result.insert("digest".to_owned(), Value::String(digest));
        result.insert("size_bytes".to_owned(), Value::from(encoded.len()));
    }
    Ok(result)
}

fn default_max_output_bytes() -> usize {
    16 * 1024
}
