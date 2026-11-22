use crate::error::AgentError;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgentBudgetPolicy {
    #[serde(default)]
    pub max_steps: Option<i64>,
    #[serde(default)]
    pub max_model_calls: Option<i64>,
    #[serde(default)]
    pub max_tokens: Option<i64>,
    #[serde(default)]
    pub max_time_ms: Option<i64>,
    #[serde(default)]
    pub max_cost: Option<f64>,
}

impl AgentBudgetPolicy {
    pub fn from_map(map: &Map<String, Value>) -> Result<Self, AgentError> {
        let allowed = [
            "max_steps",
            "max_model_calls",
            "max_tokens",
            "max_time_ms",
            "max_cost",
        ];
        for key in map.keys() {
            if !allowed.contains(&key.as_str()) {
                return Err(AgentError::Validation(format!(
                    "unknown budget policy keys: {key}"
                )));
            }
        }
        let integer = |key: &str| -> Result<Option<i64>, AgentError> {
            match map.get(key) {
                None | Some(Value::Null) => Ok(None),
                Some(Value::Number(value)) => value.as_i64().map(Some).ok_or_else(|| {
                    AgentError::Validation(format!("budget policy value must be an integer: {key}"))
                }),
                Some(_) => Err(AgentError::Validation(format!(
                    "budget policy value must be an integer: {key}"
                ))),
            }
        };
        let max_cost = match map.get("max_cost") {
            None | Some(Value::Null) => None,
            Some(Value::Number(value)) => Some(value.as_f64().ok_or_else(|| {
                AgentError::Validation("budget policy value must be numeric: max_cost".to_owned())
            })?),
            Some(_) => {
                return Err(AgentError::Validation(
                    "budget policy value must be numeric: max_cost".to_owned(),
                ));
            }
        };
        let policy = Self {
            max_steps: integer("max_steps")?,
            max_model_calls: integer("max_model_calls")?,
            max_tokens: integer("max_tokens")?,
            max_time_ms: integer("max_time_ms")?,
            max_cost,
        };
        policy.validate()?;
        Ok(policy)
    }

    pub fn validate(&self) -> Result<(), AgentError> {
        for (name, value) in [
            ("max_steps", self.max_steps),
            ("max_model_calls", self.max_model_calls),
            ("max_tokens", self.max_tokens),
            ("max_time_ms", self.max_time_ms),
        ] {
            if value.is_some_and(|item| item < 1) {
                return Err(AgentError::Validation(format!(
                    "{name} must be positive when supplied"
                )));
            }
        }
        if self.max_cost.is_some_and(|item| item <= 0.0) {
            return Err(AgentError::Validation(
                "max_cost must be positive when supplied".to_owned(),
            ));
        }
        Ok(())
    }

    pub fn limits(&self) -> Map<String, Value> {
        let mut result = Map::new();
        for (key, value) in [
            ("step_budget", self.max_steps.map(Value::from)),
            ("call_budget", self.max_model_calls.map(Value::from)),
            ("token_budget", self.max_tokens.map(Value::from)),
            ("time_budget_ms", self.max_time_ms.map(Value::from)),
            ("cost_budget", self.max_cost.map(Value::from)),
        ] {
            if let Some(value) = value {
                result.insert(key.to_owned(), value);
            }
        }
        result
    }

    pub fn seed(&self, state: &mut crate::AgentState) {
        for (key, requested) in self.limits() {
            let current = state.get(&key).cloned();
            let next = match (current, requested) {
                (Some(Value::Number(current)), requested)
                    if current.as_f64().unwrap_or(0.0) > 0.0 =>
                {
                    if key == "cost_budget" {
                        Value::from(
                            current
                                .as_f64()
                                .unwrap_or(0.0)
                                .min(requested.as_f64().unwrap_or(0.0)),
                        )
                    } else {
                        Value::from(
                            current
                                .as_i64()
                                .unwrap_or(0)
                                .min(requested.as_i64().unwrap_or(0)),
                        )
                    }
                }
                (_, requested) => requested,
            };
            state.insert(key, next);
        }
        for (key, value) in [
            ("step_used", Value::from(0)),
            ("call_used", Value::from(0)),
            ("token_used", Value::from(0)),
            ("time_used_ms", Value::from(0)),
            ("cost_used", Value::from(0.0)),
        ] {
            state.entry(key.to_owned()).or_insert(value);
        }
    }
}

fn numeric(state: &crate::AgentState, key: &str) -> f64 {
    match state.get(key) {
        Some(Value::Number(value)) => value.as_f64().unwrap_or(0.0),
        _ => 0.0,
    }
}

fn remaining(limit: f64, used: f64) -> Value {
    if limit == 0.0 {
        Value::Null
    } else {
        Value::from((limit - used).max(0.0))
    }
}

pub fn budget_hints(state: &crate::AgentState) -> Map<String, Value> {
    let mut hints = Map::new();
    hints.insert(
        "steps_remaining".to_owned(),
        remaining(numeric(state, "step_budget"), numeric(state, "step_used")),
    );
    hints.insert(
        "model_calls_remaining".to_owned(),
        remaining(numeric(state, "call_budget"), numeric(state, "call_used")),
    );
    hints.insert(
        "tokens_remaining".to_owned(),
        remaining(numeric(state, "token_budget"), numeric(state, "token_used")),
    );
    hints.insert(
        "time_remaining_ms".to_owned(),
        remaining(
            numeric(state, "time_budget_ms"),
            numeric(state, "time_used_ms"),
        ),
    );
    hints.insert(
        "cost_remaining".to_owned(),
        remaining(numeric(state, "cost_budget"), numeric(state, "cost_used")),
    );
    hints.insert("authoritative".to_owned(), Value::Bool(false));
    hints
}

pub fn refresh_budget_hints(state: &mut crate::AgentState) -> Map<String, Value> {
    let hints = budget_hints(state);
    state.insert(
        "agent_budget_hints".to_owned(),
        Value::Object(hints.clone()),
    );
    hints
}
