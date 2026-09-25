from __future__ import annotations

import pytest

from kogwistar.agent import (
    AgentBudgetPolicy,
    AgentProfile,
    AgentReadTools,
    CatalogEntry,
    CatalogStore,
    FunctionFakeTool,
    ReadScope,
    SequenceFakeModel,
    build_goal_workflow,
    build_normal_workflow,
    build_plan_workflow,
    dynamic_invocation,
    graph_signature,
    register_catalog_search_step,
    register_model_step,
    register_tool_step,
    static_invocation,
    validate_invocation_request,
    without_agent_mode_metadata,
)
from kogwistar.runtime import MappingStepResolver, StepContext
from kogwistar.runtime.base_runtime import apply_state_update_inplace
from kogwistar.runtime.base_runtime import BaseRuntime, checkpointable_state_copy
from kogwistar.runtime.models import RunSuccess, WorkflowInvocationRequest
from kogwistar.runtime import WorkflowRuntime
from kogwistar.engine_core.engine import GraphKnowledgeEngine
from tests._helpers.embeddings import ConstantEmbeddingFunction
from tests._helpers.fake_backend import build_fake_backend

pytestmark = [pytest.mark.ci, pytest.mark.core, pytest.mark.unit]


def _ctx(state: dict, *, op: str) -> StepContext:
    return StepContext(
        run_id="run-agent-c",
        workflow_id="agent-test",
        workflow_node_id=f"node-{op}",
        op=op,
        token_id="token-agent-c",
        attempt=1,
        step_seq=1,
        cache_dir=None,
        state=state,
    )


def test_normal_plan_goal_are_graph_templates_and_metadata_is_advisory() -> None:
    designs = (
        build_normal_workflow(),
        build_plan_workflow(),
        build_goal_workflow(),
    )
    assert [design.nodes[0].metadata["wf_mode"] for design in designs] == [
        "normal",
        "plan",
        "goal",
    ]
    for design in designs:
        stripped = without_agent_mode_metadata(design)
        assert graph_signature(stripped) == graph_signature(design)
        assert "wf_mode" not in stripped.nodes[0].metadata
        assert "wf_agent_contract_version" not in stripped.nodes[0].metadata


def test_profile_declares_mode_without_granting_authority() -> None:
    profile = AgentProfile(
        agent_id="agent-1",
        workflow_mode="goal",
        requested_tool_capabilities=["read.project"],
    )
    assert profile.workflow_mode == "goal"
    assert profile.effective_capabilities(caller_capabilities=["other"]) == ()
    assert profile.effective_capabilities(caller_capabilities=["read.project"]) == (
        "read.project",
    )


def test_model_binding_is_deterministic_and_budgeted() -> None:
    state: dict = {"agent_prompt": "fixed prompt"}
    ledger = AgentBudgetPolicy(
        max_model_calls=1, max_tokens=2, max_cost=1.0
    ).install(state)
    model = SequenceFakeModel(
        [{"answer": "ok", "usage": {"total_tokens": 2, "total_cost": 0.25}}]
    )
    resolver = MappingStepResolver()
    register_model_step(resolver, model, estimated_tokens=2)

    first = resolver.resolve("agent.model_call")(_ctx(state, op="agent.model_call"))
    assert isinstance(first, RunSuccess)
    apply_state_update_inplace(state, first.state_update)
    assert state["agent_model_output"]["answer"] == "ok"
    assert ledger.call_used == 1
    assert state["agent_budget_hints"]["model_calls_remaining"] == 0
    assert state["agent_budget_hints"]["tokens_remaining"] == 0
    assert ledger.cost_used == 0.25
    assert any(event.kind == "cost" for event in ledger.events)
    assert model.contexts == [
        {
            "budget": {
                "steps_remaining": None,
                "model_calls_remaining": 1,
                "tokens_remaining": 2,
                "time_remaining_ms": None,
                "cost_remaining": 1.0,
                "authoritative": False,
            }
        }
    ]

    second = resolver.resolve("agent.model_call")(_ctx(state, op="agent.model_call"))
    assert second.status == "failure"
    assert len(model.prompts) == 1


def test_budget_ledger_rehydrates_from_checkpointable_state() -> None:
    state: dict = {}
    AgentBudgetPolicy(max_steps=3, max_model_calls=2).install(state)
    persisted = checkpointable_state_copy(state)
    assert "_deps" not in persisted
    ledger = BaseRuntime.ensure_budget_ledger(persisted)
    assert ledger is not None
    assert ledger.step_budget == 3
    assert ledger.call_budget == 2

    nested: dict = {"budget": {}}
    nested_ledger = AgentBudgetPolicy(max_steps=4).install(nested)
    restored_nested = checkpointable_state_copy(nested)
    restored_ledger = BaseRuntime.ensure_budget_ledger(restored_nested)
    assert restored_ledger is not None
    assert restored_ledger.state is restored_nested["budget"]
    assert nested_ledger.step_budget == restored_ledger.step_budget == 4


def test_tool_binding_is_acl_fail_closed_and_does_not_self_grant() -> None:
    tool = FunctionFakeTool(lambda args: {"echo": args["value"]})
    resolver = MappingStepResolver()
    register_tool_step(
        resolver,
        tool,
        required_capability="tool.echo",
    )
    state = {"agent_tool_arguments": {"value": "x"}, "effective_capabilities": []}
    denied = resolver.resolve("agent.tool_call")(_ctx(state, op="agent.tool_call"))
    assert denied.status == "failure"
    assert tool.calls == []

    state["effective_capabilities"] = ["tool.echo"]
    allowed = resolver.resolve("agent.tool_call")(_ctx(state, op="agent.tool_call"))
    assert allowed.status == "success"
    assert tool.calls == [{"value": "x"}]


def test_catalog_binding_uses_descriptor_first_read() -> None:
    catalog = CatalogStore(acl_enabled=False)
    catalog.upsert(
        CatalogEntry(
            logical_id="skill:echo",
            provider_id="provider",
            provider_local_id="echo",
            kind="skill",
            name="Echo",
            summary="echo project text",
            source_fingerprint="fp-1",
        )
    )
    reads = AgentReadTools(catalog=catalog, acl_required=False)
    scope = ReadScope(principal_id="agent")
    resolver = MappingStepResolver()
    register_catalog_search_step(resolver, reads, scope)
    state = {"agent_catalog_query": "echo"}
    result = resolver.resolve("agent.catalog_search")(_ctx(state, op="agent.catalog_search"))
    assert result.status == "success"
    apply_state_update_inplace(state, result.state_update)
    assert state["agent_catalog_results"]["items"][0]["logical_id"] == "skill:echo"


def test_static_dynamic_nested_action_requests_validate_once() -> None:
    static = static_invocation("ordinary.workflow")(_ctx({}, op="agent.act"))
    assert static.workflow_design is None
    design = build_normal_workflow(workflow_id="dynamic.workflow")
    dynamic = dynamic_invocation(design)(_ctx({}, op="agent.act"))
    assert dynamic.workflow_design is not None
    assert dynamic.workflow_id == "dynamic.workflow"
    with pytest.raises(ValueError, match="must match"):
        validate_invocation_request(
            WorkflowInvocationRequest(
                workflow_id="other.workflow",
                workflow_design=design,
            )
        )


@pytest.fixture
def engine_pair(tmp_path):
    embedding = ConstantEmbeddingFunction(dim=4)
    engines = (
        GraphKnowledgeEngine(
            persist_directory=str(tmp_path / "workflow"),
            kg_graph_type="workflow",
            embedding_function=embedding,
            backend_factory=build_fake_backend,
        ),
        GraphKnowledgeEngine(
            persist_directory=str(tmp_path / "conversation"),
            kg_graph_type="conversation",
            embedding_function=embedding,
            backend_factory=build_fake_backend,
        ),
    )
    try:
        yield engines
    finally:
        for engine in engines:
            engine.close()


def test_templates_execute_as_ordinary_runtime_graphs_and_goal_is_bounded(
    engine_pair,
    monkeypatch,
) -> None:
    # This contract test exercises Python graph execution only; no native
    # authority endpoint or provider telemetry is part of the fixture.
    monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "python")
    workflow_engine, conversation_engine = engine_pair

    def run_design(
        design,
        *,
        state=None,
        run_id: str,
        goal_satisfy_after: int = 2,
    ):
        for node in design.nodes:
            workflow_engine.write.add_node(node)
        for edge in design.edges:
            workflow_engine.write.add_edge(edge)
        resolver = MappingStepResolver()
        next_by_op = {
            "agent.observe": "agent.decide",
            "agent.decide": "agent.act",
            "agent.act": "agent.check",
            "agent.execute": "agent.done",
            "agent.plan": "agent.approve",
            "agent.approve": "agent.execute",
        }

        for op in {node.op for node in design.nodes}:
            def make_handler(current_op: str):
                @resolver.register(current_op)
                def _handler(ctx):
                    if current_op == "agent.decide":
                        with ctx.state_write as mutable:
                            mutable["goal_iteration"] = int(
                                mutable.get("goal_iteration", 0)
                            ) + 1
                    if current_op == "agent.check":
                        satisfied = (
                            int(ctx.state_view.get("goal_iteration", 0))
                            >= goal_satisfy_after
                        )
                        with ctx.state_write as mutable:
                            mutable["goal_status"] = (
                                "satisfied" if satisfied else "active"
                            )
                        return RunSuccess(
                            state_update=[],
                            _route_next=[
                                "agent.done" if satisfied else "agent.observe"
                            ],
                        )
                    target = next_by_op.get(current_op)
                    return RunSuccess(
                        state_update=[], _route_next=[target] if target else []
                    )

                return _handler

            make_handler(op)

        return WorkflowRuntime(
            workflow_engine=workflow_engine,
            conversation_engine=conversation_engine,
            step_resolver=resolver,
            predicate_registry={
                "approved": lambda _info, _state, _result: True,
                "goal_satisfied": lambda _info, state, _result: state.get(
                    "goal_status"
                )
                == "satisfied",
                "goal_not_satisfied": lambda _info, state, _result: state.get(
                    "goal_status"
                )
                != "satisfied",
            },
            checkpoint_every_n_steps=1,
        ).run(
            workflow_id=design.workflow_id,
            conversation_id="agent-c-conversation",
            turn_node_id="agent-c-turn",
            initial_state=state or {"_deps": {}},
            run_id=run_id,
        )

    normal = run_design(build_normal_workflow(workflow_id="agent-c-normal"), run_id="run-normal")
    assert normal.status == "succeeded"

    plan = run_design(build_plan_workflow(workflow_id="agent-c-plan"), run_id="run-plan")
    assert plan.status == "succeeded"

    first_goal = run_design(
        build_goal_workflow(workflow_id="agent-c-goal-first"),
        state={"_deps": {}},
        run_id="run-goal-first",
        goal_satisfy_after=1,
    )
    assert first_goal.status == "succeeded"
    assert first_goal.final_state["goal_iteration"] == 1
    assert first_goal.final_state["goal_status"] == "satisfied"

    goal_state = {"_deps": {}}
    goal = run_design(
        build_goal_workflow(workflow_id="agent-c-goal"),
        state=goal_state,
        run_id="run-goal",
    )
    assert goal.status == "succeeded"
    assert goal.final_state["goal_iteration"] == 2
    assert goal.final_state["goal_status"] == "satisfied"

    limited_state = {"_deps": {}}
    AgentBudgetPolicy(max_steps=2).install(limited_state)
    limited = run_design(
        build_goal_workflow(workflow_id="agent-c-limited"),
        state=limited_state,
        run_id="run-limited",
    )
    assert limited.status == "suspended"
    assert limited.final_state["step_used"] == 2
    assert limited.final_state["step_used"] <= limited.final_state["step_budget"]
