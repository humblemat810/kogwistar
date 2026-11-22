use crate::error::AgentError;
use crate::util::canonical_json;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;

fn clean_name(value: &str) -> String {
    value.trim().to_lowercase()
}

fn clean_names(values: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut result = Vec::new();
    for value in values {
        let name = clean_name(&value);
        if !name.is_empty() && name != "*" && !result.contains(&name) {
            result.push(name);
        }
    }
    result
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentWorkflowMode {
    #[default]
    Normal,
    Plan,
    Goal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WhisperConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_knowledge_depth")]
    pub knowledge_depth: u8,
    #[serde(default = "default_tool_limit")]
    pub tool_limit: u8,
    #[serde(default = "default_knowledge_limit")]
    pub knowledge_limit: u8,
    #[serde(default = "default_seed_limit")]
    pub seed_limit: u8,
    #[serde(default = "default_summary_max_chars")]
    pub summary_max_chars: usize,
    #[serde(default = "default_payload_max_chars")]
    pub payload_max_chars: usize,
}

impl Default for WhisperConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            knowledge_depth: 1,
            tool_limit: 3,
            knowledge_limit: 10,
            seed_limit: 3,
            summary_max_chars: 320,
            payload_max_chars: 6000,
        }
    }
}

impl WhisperConfig {
    pub fn validate(&self) -> Result<(), AgentError> {
        if self.tool_limit > 3 {
            return Err(AgentError::Validation(
                "tool_limit must be between 0 and 3".to_owned(),
            ));
        }
        if self.knowledge_limit > 10 {
            return Err(AgentError::Validation(
                "knowledge_limit must be between 0 and 10".to_owned(),
            ));
        }
        if self.seed_limit > 3 {
            return Err(AgentError::Validation(
                "seed_limit must be between 0 and 3".to_owned(),
            ));
        }
        if self.summary_max_chars < 1 || self.summary_max_chars > 320 {
            return Err(AgentError::Validation(
                "summary_max_chars must be between 1 and 320".to_owned(),
            ));
        }
        if self.payload_max_chars < 128 || self.payload_max_chars > 6000 {
            return Err(AgentError::Validation(
                "payload_max_chars must be between 128 and 6000".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgentProfile {
    pub agent_id: String,
    #[serde(default)]
    pub workflow_id: Option<String>,
    #[serde(default)]
    pub workflow_mode: AgentWorkflowMode,
    #[serde(default = "default_model_profile")]
    pub model_profile: String,
    #[serde(default)]
    pub requested_tool_capabilities: Vec<String>,
    #[serde(default)]
    pub context_policy: serde_json::Map<String, Value>,
    #[serde(default)]
    pub budget_policy: serde_json::Map<String, Value>,
    #[serde(default)]
    pub skill_providers: Vec<String>,
    #[serde(default)]
    pub plugin_ids: Vec<String>,
    #[serde(default)]
    pub hook_ids: Vec<String>,
    #[serde(default)]
    pub whisper: WhisperConfig,
    #[serde(default = "bool_true")]
    pub acl_required: bool,
}

impl AgentProfile {
    pub fn new(agent_id: impl Into<String>) -> Self {
        Self {
            agent_id: agent_id.into(),
            workflow_id: None,
            workflow_mode: AgentWorkflowMode::Normal,
            model_profile: "default".to_owned(),
            requested_tool_capabilities: Vec::new(),
            context_policy: serde_json::Map::new(),
            budget_policy: serde_json::Map::new(),
            skill_providers: Vec::new(),
            plugin_ids: Vec::new(),
            hook_ids: Vec::new(),
            whisper: WhisperConfig::default(),
            acl_required: true,
        }
    }

    pub fn validate(&self) -> Result<(), AgentError> {
        if self.agent_id.trim().is_empty()
            || self.model_profile.trim().is_empty()
            || self
                .workflow_id
                .as_ref()
                .is_some_and(|item| item.trim().is_empty())
        {
            return Err(AgentError::Validation(
                "profile identifiers must not be empty".to_owned(),
            ));
        }
        self.whisper.validate()?;
        Ok(())
    }

    pub fn effective_capabilities(
        &self,
        caller_capabilities: &[String],
        revoked_capabilities: &[String],
    ) -> Vec<String> {
        let requested: BTreeSet<String> =
            self.requested_tool_capabilities.iter().cloned().collect();
        let caller: BTreeSet<String> = clean_names(caller_capabilities.iter().cloned())
            .into_iter()
            .collect();
        let revoked: BTreeSet<String> = clean_names(revoked_capabilities.iter().cloned())
            .into_iter()
            .collect();
        requested
            .intersection(&caller)
            .filter(|item| !revoked.contains(*item))
            .cloned()
            .collect()
    }

    pub fn stable_json(&self) -> String {
        let value = serde_json::to_value(self).unwrap_or(Value::Null);
        canonical_json(&value)
    }

    pub fn validate_bindings(
        &self,
        known_workflows: Option<&BTreeSet<String>>,
        known_providers: Option<&BTreeSet<String>>,
        known_model_profiles: Option<&BTreeSet<String>>,
        known_hooks: Option<&BTreeSet<String>>,
        caller_capabilities: &[String],
    ) -> Result<(), AgentError> {
        self.validate()?;
        if !self.acl_required {
            return Err(AgentError::Permission(
                "AgentProfile cannot disable host ACL enforcement".to_owned(),
            ));
        }
        if let Some(known) = known_workflows
            && let Some(workflow_id) = &self.workflow_id
            && !known.contains(workflow_id)
        {
            return Err(AgentError::Validation(format!(
                "unknown workflow: {workflow_id}"
            )));
        }
        if let Some(known) = known_providers {
            let unknown: Vec<String> = self
                .skill_providers
                .iter()
                .filter(|item| !known.contains(*item))
                .cloned()
                .collect();
            if !unknown.is_empty() {
                return Err(AgentError::Validation(format!(
                    "unknown skill providers: {}",
                    unknown.join(",")
                )));
            }
        }
        if let Some(known) = known_model_profiles
            && !known.contains(&self.model_profile)
        {
            return Err(AgentError::Validation(format!(
                "unknown model profile: {}",
                self.model_profile
            )));
        }
        if let Some(known) = known_hooks {
            let unknown: Vec<String> = self
                .hook_ids
                .iter()
                .filter(|item| !known.contains(*item))
                .cloned()
                .collect();
            if !unknown.is_empty() {
                return Err(AgentError::Validation(format!(
                    "unknown hooks: {}",
                    unknown.join(",")
                )));
            }
        }
        let requested: BTreeSet<String> =
            self.requested_tool_capabilities.iter().cloned().collect();
        let caller: BTreeSet<String> = clean_names(caller_capabilities.iter().cloned())
            .into_iter()
            .collect();
        let missing: Vec<String> = requested.difference(&caller).cloned().collect();
        if !missing.is_empty() {
            return Err(AgentError::Permission(format!(
                "profile requests capabilities absent from caller authority: {}",
                missing.join(",")
            )));
        }
        Ok(())
    }
}

fn bool_true() -> bool {
    true
}

fn default_model_profile() -> String {
    "default".to_owned()
}

fn default_knowledge_depth() -> u8 {
    1
}

fn default_tool_limit() -> u8 {
    3
}

fn default_knowledge_limit() -> u8 {
    10
}

fn default_seed_limit() -> u8 {
    3
}

fn default_summary_max_chars() -> usize {
    320
}

fn default_payload_max_chars() -> usize {
    6000
}
