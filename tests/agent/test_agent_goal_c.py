from __future__ import annotations

from dataclasses import replace

import pytest

from kogwistar.agent import (
    AgentBudgetPolicy,
    AgentProfile,
    AgentReadTools,
    CatalogEntry,
    CatalogStore,
    ControlPointSpec,
    ReadScope,
    SequenceFakeModel,
    build_goal_workflow,
    build_normal_workflow,
    build_plan_workflow,
    FunctionFakeTool,
    register_tool_step,
    dynamic_invocation,
    graph_signature,
    register_catalog_search_step,
    register_control_ack_step,
    register_control_point,
    validate_control_point_placement,
    terminalize_control_messages,
    register_model_step,
    register_tool_step,
    static_invocation,
    validate_invocation_request,
    without_agent_mode_metadata,
)
from kogwistar.runtime import MappingStepResolver, StepContext
from kogwistar.runtime.base_runtime import apply_state_update_inplace
from kogwistar.runtime.base_runtime import BaseRuntime, checkpointable_state_copy
from kogwistar.runtime.models import RunFailure, RunSuccess, WorkflowInvocationRequest
from kogwistar.runtime import WorkflowRuntime
from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.messaging.models import ProjectedLaneMessageRow
from kogwistar.messaging.service import LaneMessagingService
from tests._helpers.embeddings import ConstantEmbeddingFunction
from tests._helpers.fake_backend import build_fake_backend

pytestmark = [
    pytest.mark.ci,
    pytest.mark.core,
    pytest.mark.workflow,
    pytest.mark.runtime_sync,
    pytest.mark.unit,
    pytest.mark.e2e,
]


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


def test_plan_has_distinct_approval_and_rejection_terminal_paths() -> None:
    design = build_plan_workflow(workflow_id="agent-plan-approval")
    ops = {node.op: node for node in design.nodes}
    assert ops["agent.rejected"].metadata["wf_terminal"] is True
    approve_edges = [
        edge for edge in design.edges if edge.source_ids == [ops["agent.approve"].id]
    ]
    assert {edge.metadata["wf_predicate"] for edge in approve_edges} == {
        "approved",
        "rejected",
    }


def test_control_points_are_explicit_graph_nodes_not_hidden_runtime_hooks() -> None:
    designs = (
        build_normal_workflow(workflow_id="agent.control-normal", control_op="agent.control_point"),
        build_plan_workflow(workflow_id="agent.control-plan", control_op="agent.control_point"),
        build_goal_workflow(
            workflow_id="agent.control-goal",
            control_op="agent.control_point",
            control_metadata={
                "wf_control_policy": "queue",
                "wf_accepted_message_types": ["agent.steer"],
            },
        ),
    )
    for design in designs:
        control_nodes = [
            node for node in design.nodes if node.metadata.get("wf_control_point") is True
        ]
        assert len(control_nodes) == 1
        control_id = str(control_nodes[0].safe_get_id())
        assert control_nodes[0].op == "agent.control_point"
        if design.workflow_id == "agent.control-goal":
            assert control_nodes[0].metadata["wf_control_policy"] == "queue"
            assert control_nodes[0].metadata["wf_accepted_message_types"] == [
                "agent.steer"
            ]
        assert any(
            str(edge.source_ids[0]) == control_id
            for edge in design.edges
        )
        assert any(
            str(edge.target_ids[0]) == control_id
            for edge in design.edges
        )


def test_workflow_without_control_point_has_no_control_consumer() -> None:
    design = build_normal_workflow(workflow_id="agent.no-control")
    assert not any(node.metadata.get("wf_control_point") for node in design.nodes)


def test_control_point_placement_only_changes_observation_and_cannot_override_guards() -> None:
    before = build_normal_workflow(
        workflow_id="agent.control-placement",
        control_op="agent.control_point",
    )
    assert validate_control_point_placement(before) == ()
    terminal = before.model_copy(deep=True)
    control = next(node for node in terminal.nodes if node.metadata.get("wf_control_point"))
    control.metadata["wf_terminal"] = True
    assert "cannot be terminal" in validate_control_point_placement(terminal)[0]
    forbidden = before.model_copy(deep=True)
    control = next(node for node in forbidden.nodes if node.metadata.get("wf_control_point"))
    control.metadata["wf_budget_override"] = True
    assert "cannot override runtime guards" in validate_control_point_placement(forbidden)[0]


def test_steering_updates_future_state_without_rewriting_completed_output() -> None:
    messaging = _FakeControlMessaging()
    resolver = MappingStepResolver()
    register_control_point(
        resolver,
        messaging,
        ControlPointSpec(inbox_id="inbox:agent", recipient_id="agent-1"),
    )
    state = {
        "agent_claimed_by": "worker-1",
        "agent_model_output": {"answer": "already emitted"},
    }
    result = resolver.resolve("agent.control_point")(
        _ctx(state, op="agent.control_point")
    )
    assert isinstance(result, RunSuccess)
    updates = result.state_update[0][1]
    assert updates["agent_control_input"] == {"direction": "short"}
    assert "agent_model_output" not in updates
    assert state["agent_model_output"] == {"answer": "already emitted"}


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


def test_tool_failure_retries_within_bounded_step_attempt() -> None:
    calls = {"count": 0}

    def flaky(_args):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("temporary")
        return {"ok": True}

    tool = FunctionFakeTool(flaky)
    resolver = MappingStepResolver()
    register_tool_step(
        resolver,
        tool,
        required_capability="tool.echo",
        max_attempts=2,
    )
    result = resolver.resolve("agent.tool_call")(
        _ctx(
            {
                "agent_tool_arguments": {},
                "effective_capabilities": ["tool.echo"],
            },
            op="agent.tool_call",
        )
    )
    assert isinstance(result, RunSuccess)
    assert result.state_update[0][1]["agent_tool_attempts"] == 2
    assert len(tool.calls) == 2


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


class _FakeControlMessaging:
    def __init__(self) -> None:
        self.rows = [
            ProjectedLaneMessageRow(
                message_id="m-1",
                namespace="default",
                purpose="user_visible",
                inbox_id="inbox:agent",
                conversation_id="conv",
                recipient_id="agent-1",
                sender_id="user",
                msg_type="agent.steer",
                status="pending",
                seq=1,
                conversation_seq=1,
                claimed_by=None,
                lease_until=None,
                retry_count=0,
                created_at=1,
                available_at=1,
                run_id="run-agent-c",
                step_id=None,
                correlation_id="corr-1",
                payload_json='{"direction":"short"}',
            ),
            ProjectedLaneMessageRow(
                message_id="m-2",
                namespace="default",
                purpose="user_visible",
                inbox_id="inbox:agent",
                conversation_id="conv",
                recipient_id="agent-1",
                sender_id="user",
                msg_type="agent.steer",
                status="pending",
                seq=2,
                conversation_seq=2,
                claimed_by=None,
                lease_until=None,
                retry_count=0,
                created_at=2,
                available_at=1,
                run_id="run-agent-c",
                step_id=None,
                correlation_id="corr-2",
                payload_json='{"direction":"long"}',
            ),
        ]
        self.acks: list[tuple[str, str]] = []
        self.fail_ack_once = False
        self.clock = 100

    def list_projected(self, **kwargs):
        return [
            row
            for row in self.rows
            if row.status not in {"completed", "cancelled"}
            and (kwargs.get("inbox_id") is None or row.inbox_id == kwargs["inbox_id"])
            and (kwargs.get("run_id") is None or row.run_id == kwargs["run_id"])
            and (
                kwargs.get("recipient_id") is None
                or row.recipient_id == kwargs["recipient_id"]
            )
        ]

    def claim_pending(self, *, message_ids, claimed_by, **kwargs):
        selected = [
            row
            for row in self.rows
            if row.message_id in message_ids
            and (
                row.status == "pending"
                or (
                    row.status == "claimed"
                    and row.lease_until is not None
                    and int(row.lease_until) <= self.clock
                )
            )
            and (kwargs.get("run_id") is None or row.run_id == kwargs["run_id"])
            and (kwargs.get("msg_type") is None or row.msg_type == kwargs["msg_type"])
            and (
                kwargs.get("recipient_id") is None
                or row.recipient_id == kwargs["recipient_id"]
            )
        ]
        if not selected:
            return []
        row = selected[0]
        updated = replace(
            row,
            status="claimed",
            claimed_by=claimed_by,
            lease_until=self.clock + int(kwargs.get("lease_seconds", 60)),
        )
        self.rows[self.rows.index(row)] = updated
        return [updated]

    def ack(self, *, message_id, claimed_by):
        if self.fail_ack_once:
            self.fail_ack_once = False
            raise RuntimeError("ack interrupted")
        self.acks.append((message_id, claimed_by))
        for index, row in enumerate(self.rows):
            if row.message_id == message_id:
                self.rows[index] = replace(row, status="completed", claimed_by=None)

    def update_message_status(self, **kwargs):
        for index, row in enumerate(self.rows):
            if row.message_id == kwargs["message_id"]:
                self.rows[index] = replace(
                    row, status=kwargs["status"], claimed_by=None
                )


def test_control_point_is_fifo_checkpoint_then_ack_and_filters_target() -> None:
    messaging = _FakeControlMessaging()
    resolver = MappingStepResolver()
    spec = ControlPointSpec(
        inbox_id="inbox:agent",
        recipient_id="agent-1",
        accepted_message_types=("agent.steer",),
    )
    register_control_point(resolver, messaging, spec)
    register_control_ack_step(resolver, messaging)
    state: dict[str, object] = {"agent_claimed_by": "worker-1"}

    first = resolver.resolve("agent.control_point")(_ctx(state, op="agent.control_point"))
    assert isinstance(first, RunSuccess)
    apply_state_update_inplace(state, first.state_update)
    assert state["agent_control_input"] == {"direction": "short"}
    assert state["agent_control_received"] is True
    assert state["agent_control_pending_claims"][0]["conversation_id"] == "conv"
    assert state["agent_control_pending_claims"][0]["correlation_id"] == "corr-1"
    assert state["agent_control_pending_claims"][0]["run_id"] == "run-agent-c"
    assert messaging.acks == []

    acked = resolver.resolve("agent.control_ack")(_ctx(state, op="agent.control_ack"))
    assert isinstance(acked, RunSuccess)
    apply_state_update_inplace(state, acked.state_update)
    assert messaging.acks == [("m-1", "worker-1")]

    second = resolver.resolve("agent.control_point")(_ctx(state, op="agent.control_point"))
    assert isinstance(second, RunSuccess)
    apply_state_update_inplace(state, second.state_update)
    assert state["agent_control_input"] == {"direction": "long"}
    assert state["agent_control_high_water"] == 2


def test_control_point_cancel_replace_is_policy_state_not_hidden_runtime_control() -> None:
    messaging = _FakeControlMessaging()
    resolver = MappingStepResolver()
    register_control_point(
        resolver,
        messaging,
        ControlPointSpec(
            inbox_id="inbox:agent",
            recipient_id="agent-1",
            policy="cancel_and_replace",
        ),
    )
    state: dict[str, object] = {"agent_claimed_by": "worker-1"}
    result = resolver.resolve("agent.control_point")(
        _ctx(state, op="agent.control_point")
    )
    assert isinstance(result, RunSuccess)
    apply_state_update_inplace(state, result.state_update)
    assert state["agent_control_action"] == "cancel_and_replace"
    assert messaging.acks == []


def test_control_point_queue_preserves_fifo_and_high_water_deduplicates() -> None:
    messaging = _FakeControlMessaging()
    messaging.rows[0] = replace(messaging.rows[0], status="claimed", claimed_by="other")
    resolver = MappingStepResolver()
    register_control_point(
        resolver,
        messaging,
        ControlPointSpec(
            inbox_id="inbox:agent",
            recipient_id="agent-1",
            policy="queue",
        ),
    )
    blocked = resolver.resolve("agent.control_point")(
        _ctx({"agent_claimed_by": "worker-1"}, op="agent.control_point")
    )
    assert isinstance(blocked, RunSuccess)
    assert blocked.state_update[0][1]["agent_control_received"] is False
    assert messaging.rows[1].status == "pending"

    messaging.rows[0] = replace(messaging.rows[0], status="pending", claimed_by=None)
    duplicate = resolver.resolve("agent.control_point")(
        _ctx(
            {
                "agent_claimed_by": "worker-1",
                "agent_control_high_water": 1,
            },
            op="agent.control_point",
        )
    )
    assert isinstance(duplicate, RunSuccess)
    update = duplicate.state_update[0][1]
    assert update["agent_control_received"] is False
    assert update["agent_control_duplicate"] is True


def test_control_point_reclaims_after_crash_before_checkpoint() -> None:
    messaging = _FakeControlMessaging()
    resolver = MappingStepResolver()
    register_control_point(
        resolver,
        messaging,
        ControlPointSpec(inbox_id="inbox:agent", recipient_id="agent-1"),
    )
    first = resolver.resolve("agent.control_point")(
        _ctx({"agent_claimed_by": "worker-1"}, op="agent.control_point")
    )
    assert isinstance(first, RunSuccess)
    assert messaging.rows[0].status == "claimed"

    # Simulate process death before applying/checkpointing the returned state.
    messaging.rows[0] = replace(messaging.rows[0], lease_until=0)
    recovered = resolver.resolve("agent.control_point")(
        _ctx({"agent_claimed_by": "worker-2"}, op="agent.control_point")
    )
    assert isinstance(recovered, RunSuccess)
    assert recovered.state_update[0][1]["agent_control_input"] == {
        "direction": "short"
    }
    assert messaging.rows[0].claimed_by == "worker-2"


def test_control_point_ack_is_retried_after_checkpoint_before_ack_crash() -> None:
    messaging = _FakeControlMessaging()
    resolver = MappingStepResolver()
    register_control_point(
        resolver,
        messaging,
        ControlPointSpec(inbox_id="inbox:agent", recipient_id="agent-1"),
    )
    register_control_ack_step(resolver, messaging)
    state: dict[str, object] = {"agent_claimed_by": "worker-1"}
    claimed = resolver.resolve("agent.control_point")(
        _ctx(state, op="agent.control_point")
    )
    assert isinstance(claimed, RunSuccess)
    apply_state_update_inplace(state, claimed.state_update)
    persisted = dict(state)

    messaging.fail_ack_once = True
    failed_ack = resolver.resolve("agent.control_ack")(
        _ctx(persisted, op="agent.control_ack")
    )
    assert failed_ack.status == "failure"
    assert messaging.rows[0].status == "claimed"

    resumed = resolver.resolve("agent.control_point")(
        _ctx(persisted, op="agent.control_point")
    )
    assert isinstance(resumed, RunSuccess)
    apply_state_update_inplace(persisted, resumed.state_update)
    assert messaging.acks == [("m-1", "worker-1")]
    assert messaging.rows[0].status == "completed"


def test_control_point_rejects_wrong_kind_even_when_run_and_recipient_match() -> None:
    messaging = _FakeControlMessaging()
    messaging.rows.append(
        replace(
            messaging.rows[0],
            message_id="wrong-kind",
            msg_type="agent.queue",
            seq=0,
        )
    )
    resolver = MappingStepResolver()
    register_control_point(
        resolver,
        messaging,
        ControlPointSpec(
            inbox_id="inbox:agent",
            recipient_id="agent-1",
            accepted_message_types=("agent.steer",),
        ),
    )
    result = resolver.resolve("agent.control_point")(
        _ctx({"agent_claimed_by": "worker-1"}, op="agent.control_point")
    )
    assert isinstance(result, RunSuccess)
    assert result.state_update[0][1]["agent_control_input"] == {
        "direction": "short"
    }
    assert messaging.rows[-1].status == "pending"


def test_terminalize_control_messages_only_targets_run() -> None:
    messaging = _FakeControlMessaging()
    cancelled = terminalize_control_messages(messaging, run_id="run-agent-c")
    assert cancelled == 2
    target_status = {
        row.message_id: row.status
        for row in messaging.rows
        if row.run_id == "run-agent-c"
    }
    assert target_status == {"m-1": "cancelled", "m-2": "cancelled"}
    assert terminalize_control_messages(messaging, run_id="run-agent-c") == 0


def test_run_specific_control_inbox_isolated_before_claim() -> None:
    messaging = _FakeControlMessaging()
    messaging.rows[0] = replace(
        messaging.rows[0], inbox_id="inbox:agent:run:run-agent-c"
    )
    resolver = MappingStepResolver()
    register_control_point(
        resolver,
        messaging,
        ControlPointSpec(
            inbox_id="inbox:agent",
            recipient_id="agent-1",
            run_specific_inbox=True,
        ),
    )
    result = resolver.resolve("agent.control_point")(
        _ctx({"agent_claimed_by": "worker-1"}, op="agent.control_point")
    )
    assert isinstance(result, RunSuccess)
    assert result.state_update[0][1]["agent_control_received"] is True


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
                    if current_op == "agent.approve":
                        approval_target = (
                            "agent.execute"
                            if ctx.state_view.get("approval", "approved") == "approved"
                            else "agent.rejected"
                        )
                        return RunSuccess(state_update=[], _route_next=[approval_target])
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
                "rejected": lambda _info, state, _result: state.get("approval") != "approved",
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
    rejected_plan = run_design(
        build_plan_workflow(workflow_id="agent-c-plan-rejected"),
        state={"_deps": {}, "approval": "rejected"},
        run_id="run-plan-rejected",
    )
    assert rejected_plan.status == "succeeded"

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


def test_control_point_checkpoint_resume_ack_is_durable_with_real_runtime(
    engine_pair,
    monkeypatch,
) -> None:
    workflow_engine, conversation_engine = engine_pair
    workflow_id = "agent-control-runtime"
    run_id = "run-control-runtime"
    design = build_normal_workflow(
        workflow_id=workflow_id,
        execute_op="agent.apply",
        control_op="agent.control_point",
        ack_op="agent.control_ack",
    )
    for node in design.nodes:
        workflow_engine.write.add_node(node)
    for edge in design.edges:
        workflow_engine.write.add_edge(edge)

    messaging = LaneMessagingService(conversation_engine)
    messaging.send_message(
        conversation_id="agent-control-conversation",
        inbox_id="inbox:agent",
        sender_id="user",
        recipient_id="agent-1",
        msg_type="agent.steer",
        payload={"directive": "safe-mode"},
        run_id=run_id,
        step_id="control",
        correlation_id="corr-runtime",
    )

    resolver = MappingStepResolver()
    register_control_point(
        resolver,
        messaging,
        ControlPointSpec(inbox_id="inbox:agent", recipient_id="agent-1"),
    )
    register_control_ack_step(resolver, messaging)

    def _step(ctx):
        if ctx.op == "agent.apply":
            directive = ctx.state_view.get("agent_control_input")
            if not directive:
                return RunFailure(
                    conversation_node_id=None,
                    state_update=[],
                    errors=["control input was not checkpointed before apply"],
                )
            return RunSuccess(
                conversation_node_id=None,
                state_update=[("u", {"applied_directive": directive})],
            )
        return RunSuccess(conversation_node_id=None, state_update=[])

    for op in {node.op for node in design.nodes}:
        if op not in {"agent.control_point", "agent.control_ack"}:
            resolver.register(op)(_step)
    runtime = WorkflowRuntime(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        step_resolver=resolver,
        predicate_registry={},
        checkpoint_every_n_steps=1,
    )

    original_ack = messaging.ack
    failed_once = {"value": False}

    def fail_ack_once(*, message_id: str, claimed_by: str) -> None:
        if not failed_once["value"]:
            failed_once["value"] = True
            raise RuntimeError("simulated ack crash")
        original_ack(message_id=message_id, claimed_by=claimed_by)

    monkeypatch.setattr(messaging, "ack", fail_ack_once)
    first = runtime.run(
        workflow_id=workflow_id,
        conversation_id="agent-control-conversation",
        turn_node_id="turn-control",
        initial_state={
            "_deps": {},
            "agent_recipient_id": "agent-1",
            "agent_claimed_by": "worker-1",
        },
        run_id=run_id,
    )
    assert first.status == "failure"
    assert any(row.status == "claimed" for row in messaging.list_projected(
        inbox_id="inbox:agent", run_id=run_id, status=None, limit=10
    ))

    monkeypatch.setattr(messaging, "ack", original_ack)
    resumed = runtime.resume_from_latest_checkpoint(
        run_id=run_id,
        workflow_id=workflow_id,
        conversation_id="agent-control-conversation",
        turn_node_id="turn-control",
    )
    assert resumed.status == "succeeded"
    assert resumed.final_state["applied_directive"] == {"directive": "safe-mode"}
    rows = messaging.list_projected(
        inbox_id="inbox:agent", run_id=run_id, status=None, limit=10
    )
    assert len(rows) == 1
    assert rows[0].status == "completed"
