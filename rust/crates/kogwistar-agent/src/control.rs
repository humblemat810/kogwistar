use crate::error::AgentError;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlPolicy {
    #[default]
    Steer,
    Queue,
    CancelAndReplace,
}

impl std::fmt::Display for ControlPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let value = match self {
            ControlPolicy::Steer => "steer",
            ControlPolicy::Queue => "queue",
            ControlPolicy::CancelAndReplace => "cancel_and_replace",
        };
        write!(f, "{value}")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ControlPointSpec {
    pub inbox_id: String,
    #[serde(default)]
    pub policy: ControlPolicy,
    #[serde(default = "default_accepted_message_types")]
    pub accepted_message_types: Vec<String>,
    #[serde(default)]
    pub recipient_id: Option<String>,
    #[serde(default = "default_max_claims")]
    pub max_claims: i64,
    #[serde(default = "default_lease_seconds")]
    pub lease_seconds: i64,
    #[serde(default)]
    pub run_specific_inbox: bool,
    #[serde(default = "default_claimed_by_key")]
    pub claimed_by_key: String,
    #[serde(default = "default_output_key")]
    pub output_key: String,
    #[serde(default = "default_pending_claims_key")]
    pub pending_claims_key: String,
    #[serde(default = "default_high_water_key")]
    pub high_water_key: String,
}

impl Default for ControlPointSpec {
    fn default() -> Self {
        Self {
            inbox_id: String::new(),
            policy: ControlPolicy::Steer,
            accepted_message_types: default_accepted_message_types(),
            recipient_id: None,
            max_claims: 1,
            lease_seconds: 60,
            run_specific_inbox: false,
            claimed_by_key: default_claimed_by_key(),
            output_key: default_output_key(),
            pending_claims_key: default_pending_claims_key(),
            high_water_key: default_high_water_key(),
        }
    }
}

impl ControlPointSpec {
    pub fn validate(&self) -> Result<(), AgentError> {
        if self.inbox_id.trim().is_empty() {
            return Err(AgentError::Validation(
                "inbox_id must be non-empty".to_owned(),
            ));
        }
        if self.accepted_message_types.is_empty()
            || self
                .accepted_message_types
                .iter()
                .any(|item| item.trim().is_empty())
        {
            return Err(AgentError::Validation(
                "accepted_message_types must be non-empty".to_owned(),
            ));
        }
        if self.max_claims != 1 {
            return Err(AgentError::Validation(
                "v1 control points support one FIFO claim".to_owned(),
            ));
        }
        if self.lease_seconds <= 0 {
            return Err(AgentError::Validation(
                "lease_seconds must be positive".to_owned(),
            ));
        }
        Ok(())
    }

    pub fn resolved_inbox_id(&self, run_id: &str) -> String {
        if self.run_specific_inbox {
            format!("{}:run:{}", self.inbox_id, run_id)
        } else {
            self.inbox_id.clone()
        }
    }
}

fn default_accepted_message_types() -> Vec<String> {
    vec!["agent.steer".to_owned()]
}

fn default_max_claims() -> i64 {
    1
}

fn default_lease_seconds() -> i64 {
    60
}

fn default_claimed_by_key() -> String {
    "agent_claimed_by".to_owned()
}

fn default_output_key() -> String {
    "agent_control_input".to_owned()
}

fn default_pending_claims_key() -> String {
    "agent_control_pending_claims".to_owned()
}

fn default_high_water_key() -> String {
    "agent_control_high_water".to_owned()
}
