from __future__ import annotations

import pytest

from kogwistar.acl import ACLInput
from kogwistar.agent import SequenceFakeModel, register_model_step, register_tool_step
from kogwistar.runtime import MappingStepResolver, StepContext
from kogwistar.runtime.base_runtime import apply_state_update_inplace


pytestmark = [pytest.mark.ci, pytest.mark.core, pytest.mark.workflow]


def _ctx(state: dict, *, authority_context: dict | None = None) -> StepContext:
    return StepContext(
        run_id="run-acl-taint",
        workflow_id="acl-taint",
        workflow_node_id="node-model",
        op="agent.model_call",
        token_id="token-acl-taint",
        attempt=1,
        step_seq=1,
        cache_dir=None,
        state=state,
        authority_context=authority_context or {},
    )


def test_model_output_uses_strict_join_and_does_not_trust_payload_acl() -> None:
    resolver = MappingStepResolver()
    register_model_step(
        resolver,
        SequenceFakeModel([{"answer": "derived", "acl": "public"}]),
        prompt_acl_mode="public",
    )
    state = {"agent_prompt": "public question"}
    result = resolver.resolve("agent.model_call")(
        _ctx(
            state,
            authority_context={
                "acl_inputs": [ACLInput("private-context", "private", owner_id="u1")]
            },
        )
    )

    assert result.status == "success"
    apply_state_update_inplace(state, result.state_update)
    assert state["agent_model_output"]["acl"] == "public"
    assert state["agent_model_output_acl"]["acl_mode"] == "private"
    assert state["agent_model_output_provenance"]["visible_context_acl_join"] == "private"


def test_mutable_state_cannot_downgrade_acl() -> None:
    resolver = MappingStepResolver()
    register_model_step(resolver, SequenceFakeModel([{"answer": "safe"}]), prompt_acl_mode="public")
    state = {
        "agent_prompt": "public question",
        "agent_acl_inputs": [{"id": "secret", "acl_mode": "public"}],
    }
    result = resolver.resolve("agent.model_call")(_ctx(state))
    assert result.status == "success"
    apply_state_update_inplace(state, result.state_update)
    assert state["agent_model_output_acl"]["acl_mode"] == "private"


def test_tool_output_is_private_by_default_and_can_be_host_declared_public() -> None:
    resolver = MappingStepResolver()
    register_tool_step(
        resolver,
        type("Tool", (), {"invoke": lambda _self, _args: {"answer": "x"}})(),
        required_capability="tool.read",
        output_acl_mode="public",
    )
    result = resolver.resolve("agent.tool_call")(
        StepContext(
            run_id="run-acl-taint",
            workflow_id="acl-taint",
            workflow_node_id="node-tool",
            op="agent.tool_call",
            token_id="token-acl-taint",
            attempt=1,
            step_seq=1,
            cache_dir=None,
            state={"agent_tool_arguments": {}},
            authority_context={"effective_capabilities": ("tool.read",)},
        )
    )
    assert result.status == "success"
    assert result.state_update[0][1]["agent_tool_output_acl"]["acl_mode"] == "public"
