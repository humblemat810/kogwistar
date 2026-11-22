pub mod context;
pub mod control;
pub mod delegation;
pub mod error;
pub mod harness;
pub mod hooks;
pub mod limits;
pub mod profile;
pub mod providers;
pub mod util;

pub type AgentState = serde_json::Map<String, serde_json::Value>;

pub use context::{Compression, ContextPolicy, ContextSnapshot};
pub use control::{ControlPointSpec, ControlPolicy};
pub use delegation::{
    DelegationContext, DelegationSpec, bounded_child_result, delegated_initial_state,
};
pub use error::{AgentError, AgentResult};
pub use harness::{
    AgentHarness, AgentHarnessOptions, AgentInvocationRequest, AgentResumeRequest,
    AsyncAgentHarness, BoxFuture, WorkflowRuntimeLike,
};
pub use hooks::{HookEffect, HookFailureMode, HookRegistry, HookResult, HookSpec};
pub use limits::{AgentBudgetPolicy, budget_hints, refresh_budget_hints};
pub use profile::{AgentProfile, AgentWorkflowMode};
pub use providers::{
    Provider, ProviderIdentity, ProviderInactiveError, ProviderRegistration, ProviderRegistry,
};
pub use util::canonical_json;
