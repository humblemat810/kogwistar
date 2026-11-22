use crate::error::{AgentError, AgentResult};
use crate::limits::AgentBudgetPolicy;
use crate::profile::AgentProfile;
use serde_json::{Map, Value};
use std::collections::BTreeSet;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

#[derive(Debug, Clone)]
pub struct AgentInvocationRequest {
    pub workflow_id: String,
    pub conversation_id: String,
    pub initial_state: crate::AgentState,
    pub authority_context: serde_json::Value,
    pub extra: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct AgentResumeRequest {
    pub authority_context: serde_json::Value,
    pub extra: serde_json::Value,
}

pub trait WorkflowRuntimeLike: Send + Sync {
    fn run<'a>(&'a self, request: AgentInvocationRequest) -> BoxFuture<'a, AgentResult<Value>>;
    fn resume_from_latest_checkpoint<'a>(
        &'a self,
        request: AgentResumeRequest,
    ) -> BoxFuture<'a, AgentResult<Value>>;
}

#[derive(Debug, Clone, Default)]
pub struct AgentHarnessOptions {
    pub known_workflows: Option<BTreeSet<String>>,
    pub known_providers: Option<BTreeSet<String>>,
    pub known_model_profiles: Option<BTreeSet<String>>,
    pub known_hooks: Option<BTreeSet<String>>,
    pub caller_capabilities: Vec<String>,
    pub revoked_capabilities: Vec<String>,
    pub principal_id: Option<String>,
    pub tenant_id: Option<String>,
    pub project_id: Option<String>,
    pub security_scope: Option<String>,
}

#[derive(Clone)]
pub struct AgentHarness {
    profile: AgentProfile,
    runtime: Arc<dyn WorkflowRuntimeLike>,
    known_workflows: Option<BTreeSet<String>>,
    effective_capabilities: Vec<String>,
    budget_policy: AgentBudgetPolicy,
    authority_context: serde_json::Value,
}

impl AgentHarness {
    pub fn new(
        profile: AgentProfile,
        runtime: Arc<dyn WorkflowRuntimeLike>,
        options: AgentHarnessOptions,
    ) -> AgentResult<Self> {
        profile.validate_bindings(
            options.known_workflows.as_ref(),
            options.known_providers.as_ref(),
            options.known_model_profiles.as_ref(),
            options.known_hooks.as_ref(),
            &options.caller_capabilities,
        )?;
        let budget_policy = AgentBudgetPolicy::from_map(&profile.budget_policy)?;
        let effective_capabilities = profile
            .effective_capabilities(&options.caller_capabilities, &options.revoked_capabilities);
        let mut authority = Map::new();
        authority.insert(
            "principal_id".to_owned(),
            Value::String(
                options
                    .principal_id
                    .unwrap_or_else(|| profile.agent_id.clone()),
            ),
        );
        if let Some(tenant_id) = options.tenant_id {
            authority.insert("tenant_id".to_owned(), Value::String(tenant_id));
        }
        if let Some(project_id) = options.project_id {
            authority.insert("project_id".to_owned(), Value::String(project_id));
        }
        if let Some(security_scope) = options.security_scope {
            authority.insert("security_scope".to_owned(), Value::String(security_scope));
        }
        authority.insert(
            "effective_capabilities".to_owned(),
            Value::Array(
                effective_capabilities
                    .iter()
                    .cloned()
                    .map(Value::String)
                    .collect(),
            ),
        );
        authority.insert(
            "budget_limits".to_owned(),
            Value::Object(budget_policy.limits()),
        );
        Ok(Self {
            profile,
            runtime,
            known_workflows: options.known_workflows,
            effective_capabilities,
            budget_policy,
            authority_context: Value::Object(authority),
        })
    }

    pub fn profile(&self) -> &AgentProfile {
        &self.profile
    }

    pub fn authority_context(&self) -> &serde_json::Value {
        &self.authority_context
    }

    pub fn effective_capabilities(&self) -> &[String] {
        &self.effective_capabilities
    }

    pub fn budget_policy(&self) -> &AgentBudgetPolicy {
        &self.budget_policy
    }

    fn select_workflow(&self, workflow_id: Option<String>) -> AgentResult<String> {
        let selected = workflow_id
            .or_else(|| self.profile.workflow_id.clone())
            .ok_or_else(|| {
                AgentError::Validation(
                    "workflow_id is required by the agent profile or run call".to_owned(),
                )
            })?;
        if let Some(known) = &self.known_workflows
            && !known.contains(&selected)
        {
            return Err(AgentError::Validation(format!(
                "unknown workflow: {selected}"
            )));
        }
        Ok(selected)
    }

    fn prepared_state(&self, mut state: crate::AgentState) -> crate::AgentState {
        if !self.effective_capabilities.is_empty() {
            state.insert(
                "effective_capabilities".to_owned(),
                Value::Array(
                    self.effective_capabilities
                        .iter()
                        .cloned()
                        .map(Value::String)
                        .collect(),
                ),
            );
        }
        if !self.budget_policy.limits().is_empty() {
            self.budget_policy.seed(&mut state);
        }
        state
    }

    pub fn run(
        &self,
        initial_state: crate::AgentState,
        conversation_id: impl Into<String>,
        workflow_id: Option<String>,
        extra: Option<Map<String, Value>>,
    ) -> BoxFuture<'_, AgentResult<Value>> {
        let selected_workflow = self.select_workflow(workflow_id);
        let conversation_id = conversation_id.into();
        let extra = extra.map(Value::Object).unwrap_or(Value::Null);
        Box::pin(async move {
            let workflow_id = selected_workflow?;
            let request = AgentInvocationRequest {
                workflow_id,
                conversation_id,
                initial_state: self.prepared_state(initial_state),
                authority_context: self.authority_context.clone(),
                extra,
            };
            self.runtime.run(request).await
        })
    }

    pub fn resume_from_latest_checkpoint(
        &self,
        extra: Option<Map<String, Value>>,
    ) -> BoxFuture<'_, AgentResult<Value>> {
        let request = AgentResumeRequest {
            authority_context: self.authority_context.clone(),
            extra: extra.map(Value::Object).unwrap_or(Value::Null),
        };
        self.runtime.resume_from_latest_checkpoint(request)
    }
}

#[derive(Clone)]
pub struct AsyncAgentHarness {
    inner: AgentHarness,
}

impl AsyncAgentHarness {
    pub fn new(
        profile: AgentProfile,
        runtime: Arc<dyn WorkflowRuntimeLike>,
        options: AgentHarnessOptions,
    ) -> AgentResult<Self> {
        Ok(Self {
            inner: AgentHarness::new(profile, runtime, options)?,
        })
    }

    pub fn inner(&self) -> &AgentHarness {
        &self.inner
    }

    pub fn run(
        &self,
        initial_state: crate::AgentState,
        conversation_id: impl Into<String>,
        workflow_id: Option<String>,
        extra: Option<Map<String, Value>>,
    ) -> BoxFuture<'_, AgentResult<Value>> {
        self.inner
            .run(initial_state, conversation_id, workflow_id, extra)
    }

    pub fn resume_from_latest_checkpoint(
        &self,
        extra: Option<Map<String, Value>>,
    ) -> BoxFuture<'_, AgentResult<Value>> {
        self.inner.resume_from_latest_checkpoint(extra)
    }
}
