use std::collections::BTreeSet;
use std::sync::Arc;

use kogwistar_agent::{
    AgentBudgetPolicy, AgentError, AgentHarness, AgentHarnessOptions, AgentProfile,
    ContextSnapshot, ControlPointSpec, ControlPolicy, DelegationSpec, HookFailureMode,
    HookRegistry, HookSpec, Provider, ProviderRegistry,
};
use serde_json::{Value, json};

struct FakeRuntime {
    result: Value,
}

impl kogwistar_agent::WorkflowRuntimeLike for FakeRuntime {
    fn run<'a>(
        &'a self,
        request: kogwistar_agent::AgentInvocationRequest,
    ) -> kogwistar_agent::BoxFuture<'a, Result<Value, AgentError>> {
        Box::pin(async move {
            Ok(json!({
                "authority_context": request.authority_context,
                "conversation_id": request.conversation_id,
                "initial_state": request.initial_state,
                "result": self.result,
                "workflow_id": request.workflow_id,
            }))
        })
    }

    fn resume_from_latest_checkpoint<'a>(
        &'a self,
        request: kogwistar_agent::AgentResumeRequest,
    ) -> kogwistar_agent::BoxFuture<'a, Result<Value, AgentError>> {
        Box::pin(async move { Ok(json!({ "authority_context": request.authority_context })) })
    }
}

struct NoopProvider {
    id: String,
}

impl Provider for NoopProvider {
    fn provider_id(&self) -> &str {
        &self.id
    }
}

fn profile() -> AgentProfile {
    let mut profile = AgentProfile::new("agent");
    profile.workflow_id = Some("workflow".to_owned());
    profile.requested_tool_capabilities = vec!["tool.a".to_owned()];
    profile.budget_policy = serde_json::from_value(json!({"max_steps": 2})).unwrap();
    profile
}

fn options() -> AgentHarnessOptions {
    AgentHarnessOptions {
        known_workflows: Some(BTreeSet::from(["workflow".to_owned()])),
        caller_capabilities: vec!["tool.a".to_owned()],
        ..Default::default()
    }
}

#[test]
fn profile_capability_intersection_is_authority_preserving() {
    let mut profile = profile();
    profile.requested_tool_capabilities = vec![" TOOL.A ".to_owned(), "tool.b".to_owned()];
    assert_eq!(
        profile.effective_capabilities(&["tool.b".to_owned()], &[]),
        vec!["tool.b".to_owned()]
    );
    assert_eq!(
        profile.effective_capabilities(&["tool.a".to_owned()], &["tool.a".to_owned()]),
        Vec::<String>::new()
    );
    assert_eq!(profile.stable_json(), profile.stable_json());
}

#[test]
fn profile_binding_validation_fails_closed() {
    let profile = profile();
    let unknown_model = options();
    let unknown_model = AgentHarnessOptions {
        known_model_profiles: Some(BTreeSet::from(["other".to_owned()])),
        ..unknown_model
    };
    assert!(
        profile
            .validate_bindings(
                unknown_model.known_workflows.as_ref(),
                unknown_model.known_providers.as_ref(),
                unknown_model.known_model_profiles.as_ref(),
                unknown_model.known_hooks.as_ref(),
                &unknown_model.caller_capabilities,
            )
            .is_err()
    );

    let escalated = AgentHarnessOptions {
        caller_capabilities: Vec::new(),
        ..options()
    };
    assert!(
        profile
            .validate_bindings(
                escalated.known_workflows.as_ref(),
                escalated.known_providers.as_ref(),
                escalated.known_model_profiles.as_ref(),
                escalated.known_hooks.as_ref(),
                &escalated.caller_capabilities,
            )
            .is_err()
    );
}

#[test]
fn profile_rejects_empty_identity_and_disabled_acl() {
    let empty = AgentProfile::new(" ");
    assert!(matches!(empty.validate(), Err(AgentError::Validation(_))));

    let mut unprotected = AgentProfile::new("agent");
    unprotected.acl_required = false;
    assert!(matches!(
        unprotected.validate_bindings(None, None, None, None, &[]),
        Err(AgentError::Permission(_))
    ));
}

#[tokio::test]
async fn harness_delegates_to_runtime_with_authority_context() {
    let runtime = Arc::new(FakeRuntime {
        result: json!({"ok": true}),
    });
    let harness = AgentHarness::new(profile(), runtime, options()).unwrap();
    let state = serde_json::Map::new();
    let result = harness
        .run(state, "conversation", None, None)
        .await
        .unwrap();

    assert_eq!(result["conversation_id"], "conversation");
    assert_eq!(result["workflow_id"], "workflow");
    assert_eq!(result["result"], json!({"ok": true}));
    assert_eq!(
        result["authority_context"]["effective_capabilities"],
        json!(["tool.a"])
    );
    assert_eq!(result["authority_context"]["principal_id"], "agent");
    assert_eq!(result["initial_state"]["step_budget"], 2);
}

#[tokio::test]
async fn harness_rejects_runtime_workflow_override_outside_allowlist() {
    let runtime = Arc::new(FakeRuntime {
        result: Value::Null,
    });
    let harness = AgentHarness::new(profile(), runtime, options()).unwrap();
    let error = harness
        .run(
            serde_json::Map::new(),
            "conversation",
            Some("other".to_owned()),
            None,
        )
        .await
        .unwrap_err();
    assert!(matches!(error, AgentError::Validation(_)));
}

#[tokio::test]
async fn harness_overwrites_untrusted_state_authority_claims() {
    let runtime = Arc::new(FakeRuntime {
        result: Value::Null,
    });
    let harness = AgentHarness::new(profile(), runtime, options()).unwrap();
    let mut state = serde_json::Map::new();
    state.insert("effective_capabilities".to_owned(), json!(["tool.admin"]));
    state.insert("step_budget".to_owned(), json!(99));
    state.insert("step_used".to_owned(), json!(99));

    let result = harness
        .run(state, "conversation", None, None)
        .await
        .unwrap();
    assert_eq!(
        result["initial_state"]["effective_capabilities"],
        json!(["tool.a"])
    );
    assert_eq!(result["initial_state"]["step_budget"].as_i64(), Some(2));
}

#[tokio::test]
async fn harness_resume_carries_runtime_authority() {
    let runtime = Arc::new(FakeRuntime {
        result: Value::Null,
    });
    let harness = AgentHarness::new(profile(), runtime, options()).unwrap();
    let result = harness.resume_from_latest_checkpoint(None).await.unwrap();
    assert_eq!(result["authority_context"]["principal_id"], "agent");
    assert_eq!(
        result["authority_context"]["effective_capabilities"],
        json!(["tool.a"])
    );
}

#[tokio::test]
async fn harness_requires_workflow_id_from_profile_or_call() {
    let runtime = Arc::new(FakeRuntime {
        result: Value::Null,
    });
    let mut no_workflow = profile();
    no_workflow.workflow_id = None;
    let harness = AgentHarness::new(no_workflow, runtime, options()).unwrap();
    let error = harness
        .run(serde_json::Map::new(), "conversation", None, None)
        .await
        .unwrap_err();
    assert!(matches!(error, AgentError::Validation(_)));
}

#[test]
fn budget_policy_seeds_and_reports_non_authoritative_hints() {
    let policy = AgentBudgetPolicy::from_map(
        &serde_json::from_value(json!({"max_steps": 3, "max_cost": 1.5})).unwrap(),
    )
    .unwrap();
    let mut state = serde_json::Map::new();
    policy.seed(&mut state);
    assert_eq!(state["step_budget"].as_i64(), Some(3));
    assert_eq!(state["cost_budget"].as_f64(), Some(1.5));
    let hints = kogwistar_agent::refresh_budget_hints(&mut state);
    assert_eq!(hints["steps_remaining"].as_f64(), Some(3.0));
    assert_eq!(hints["authoritative"], false);
}

#[test]
fn budget_policy_rejects_unknown_and_nonpositive_limits() {
    assert!(matches!(
        AgentBudgetPolicy::from_map(&serde_json::from_value(json!({"unknown": 1})).unwrap(),),
        Err(AgentError::Validation(_))
    ));
    for key in ["max_steps", "max_model_calls", "max_tokens", "max_time_ms"] {
        let policy = AgentBudgetPolicy::from_map(&serde_json::from_value(json!({key: 0})).unwrap());
        assert!(matches!(policy, Err(AgentError::Validation(_))));
    }
    assert!(matches!(
        AgentBudgetPolicy::from_map(&serde_json::from_value(json!({"max_cost": 0.0})).unwrap(),),
        Err(AgentError::Validation(_))
    ));
}

#[test]
fn context_policy_rejects_zero_bounds() {
    let zero_items = kogwistar_agent::ContextPolicy {
        max_items: 0,
        ..Default::default()
    };
    assert!(zero_items.validate().is_err());
    let zero_chars = kogwistar_agent::ContextPolicy {
        max_chars: 0,
        ..Default::default()
    };
    assert!(zero_chars.validate().is_err());
}

#[test]
fn context_snapshot_fingerprint_is_stable_and_sensitive() {
    let mut snapshot = ContextSnapshot {
        conversation_id: Some("conversation".to_owned()),
        ..Default::default()
    };
    let first = snapshot.stable_fingerprint();
    assert_eq!(first, snapshot.stable_fingerprint());
    snapshot.conversation_id = Some("other".to_owned());
    assert_ne!(first, snapshot.stable_fingerprint());
}

#[test]
fn control_point_spec_validates_and_resolves_inbox() {
    let mut spec = ControlPointSpec {
        inbox_id: "inbox".to_owned(),
        ..Default::default()
    };
    spec.validate().unwrap();
    assert_eq!(spec.policy, ControlPolicy::Steer);
    assert_eq!(spec.resolved_inbox_id("run"), "inbox");
    spec.run_specific_inbox = true;
    assert_eq!(spec.resolved_inbox_id("run"), "inbox:run:run");
    spec.max_claims = 2;
    assert!(spec.validate().is_err());
}

#[test]
fn delegation_builds_isolated_child_state() {
    let spec = DelegationSpec {
        model_profile: "child".to_owned(),
        requested_capabilities: vec!["tool.a".to_owned()],
        context_keys: vec!["shared".to_owned(), "_private".to_owned()],
        ..Default::default()
    };
    let mut parent = serde_json::Map::new();
    parent.insert("effective_capabilities".to_owned(), json!(["tool.a"]));
    parent.insert("shared".to_owned(), json!("value"));
    parent.insert("_private".to_owned(), json!("hidden"));
    let child = kogwistar_agent::delegated_initial_state(&parent, &spec).unwrap();
    assert_eq!(child["agent_model_profile"], "child");
    assert_eq!(child["effective_capabilities"], json!(["tool.a"]));
    assert_eq!(child["shared"], "value");
    assert!(!child.contains_key("_private"));
}

#[test]
fn bounded_child_result_truncates_large_payloads() {
    let result = kogwistar_agent::bounded_child_result(&json!({"value": 1}), 1024).unwrap();
    assert_eq!(result["truncated"], false);
    assert_eq!(result["result"], json!({"value": 1}));

    let long = "x".repeat(2048);
    let result = kogwistar_agent::bounded_child_result(&json!(long), 64).unwrap();
    assert_eq!(result["truncated"], true);
    assert_eq!(result["result"], Value::Null);
    assert!(result["size_bytes"].as_u64().unwrap() > 64);
}

#[test]
fn provider_registry_enforces_identity_and_lifecycle() {
    let registry = ProviderRegistry::new();
    let provider = Arc::new(NoopProvider {
        id: "provider".to_owned(),
    });
    let first = registry.register(provider.clone(), "v1").unwrap();
    assert!(registry.register(provider, "v1").is_err());
    assert!(registry.run_if_active(&first, || "active").is_ok());
    registry.dispose(&first).unwrap();
    assert!(registry.run_if_active(&first, || "active").is_err());
}

#[test]
fn provider_registry_rejects_empty_identity() {
    let registry = ProviderRegistry::new();
    assert!(matches!(
        registry.register(Arc::new(NoopProvider { id: " ".to_owned() }), "v1",),
        Err(AgentError::Validation(_))
    ));
    assert!(matches!(
        registry.register(
            Arc::new(NoopProvider {
                id: "provider".to_owned()
            }),
            " ",
        ),
        Err(AgentError::Validation(_))
    ));
}

#[test]
fn hook_registry_gates_capabilities_and_propagates_fail_closed() {
    let registry = HookRegistry::new();
    registry
        .register(HookSpec {
            hook_id: "annotator".to_owned(),
            callback: Arc::new(|_| Ok(json!({"annotation": true}))),
            order: 1,
            timeout_ms: 100,
            failure_mode: HookFailureMode::FailOpen,
            effect: kogwistar_agent::HookEffect::Annotate,
            required_capabilities: vec!["tool.a".to_owned()],
        })
        .unwrap();
    let denied = registry.run(&json!({}), &[]).unwrap();
    assert_eq!(denied[0].status, "skipped");
    let applied = registry.run(&json!({}), &["tool.a".to_owned()]).unwrap();
    assert_eq!(applied[0].status, "applied");

    let failing = HookRegistry::new();
    failing
        .register(HookSpec {
            hook_id: "failing".to_owned(),
            callback: Arc::new(|_| Err(AgentError::Runtime("boom".to_owned()))),
            order: 1,
            timeout_ms: 100,
            failure_mode: HookFailureMode::FailClosed,
            effect: kogwistar_agent::HookEffect::Observe,
            required_capabilities: Vec::new(),
        })
        .unwrap();
    assert!(failing.run(&json!({}), &[]).is_err());
}

#[test]
fn hook_registry_rejects_invalid_and_duplicate_specs() {
    let registry = HookRegistry::new();
    let spec = |hook_id: &str, timeout_ms: u64| HookSpec {
        hook_id: hook_id.to_owned(),
        callback: Arc::new(|_| Ok(json!({}))),
        order: 1,
        timeout_ms,
        failure_mode: HookFailureMode::FailOpen,
        effect: kogwistar_agent::HookEffect::Observe,
        required_capabilities: Vec::new(),
    };
    assert!(registry.register(spec("", 1)).is_err());
    assert!(registry.register(spec("hook", 0)).is_err());
    registry.register(spec("hook", 1)).unwrap();
    assert!(registry.register(spec("hook", 1)).is_err());
}

#[test]
fn control_point_spec_rejects_invalid_bounds() {
    let spec = |control: ControlPointSpec| {
        control.validate().unwrap_err();
        control
    };
    spec(ControlPointSpec {
        inbox_id: " ".to_owned(),
        ..Default::default()
    });
    let bad_types = ControlPointSpec {
        accepted_message_types: Vec::new(),
        ..ControlPointSpec::default()
    };
    spec(ControlPointSpec {
        inbox_id: "inbox".to_owned(),
        ..bad_types
    });

    spec(ControlPointSpec {
        lease_seconds: 0,
        ..Default::default()
    });
}
