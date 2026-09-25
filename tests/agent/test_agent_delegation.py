"""Bounded subagent delegation contracts; no live model or network."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from kogwistar.agent import (
    DelegationSpec,
    bounded_child_result,
    build_delegated_invocation,
    delegated_initial_state,
)


pytestmark = [pytest.mark.ci, pytest.mark.core, pytest.mark.regression]


def _ctx() -> SimpleNamespace:
    return SimpleNamespace(
        run_id="parent-run",
        step_seq=4,
        conversation_id="conversation",
        turn_node_id="turn",
        state_view={
            "effective_capabilities": ["knowledge.read", "vision.read"],
            "private_context": "must not leak",
            "selected_context": {"node_id": "n1"},
            "_deps": {"secret": object()},
        },
    )


def test_delegation_intersects_capabilities_and_isolates_context() -> None:
    spec = DelegationSpec(
        model_profile="vision-local",
        requested_capabilities=("vision.read",),
        context_keys=("selected_context", "private_context", "_deps"),
        max_steps=3,
        max_tokens=100,
    )
    state = delegated_initial_state(_ctx().state_view, spec)
    assert state["effective_capabilities"] == ["vision.read"]
    assert state["selected_context"] == {"node_id": "n1"}
    # Explicitly selected context is allowed; unrelated parent state is not copied.
    assert state["private_context"] == "must not leak"
    assert "_deps" not in state
    assert state["step_budget"] == 3

    request = build_delegated_invocation(
        _ctx(), workflow_id="child.workflow", spec=spec
    )
    assert request.invocation_key == "parent-run:4:child.workflow:vision-local"
    assert request.initial_state["agent_model_profile"] == "vision-local"

    repeated = build_delegated_invocation(
        _ctx(), workflow_id="child.workflow", spec=spec
    )
    assert repeated.invocation_key == request.invocation_key


def test_delegation_cannot_escalate_capability_or_output_context() -> None:
    with pytest.raises(PermissionError):
        delegated_initial_state(
            _ctx().state_view,
            DelegationSpec(
                model_profile="lookup",
                requested_capabilities=("process.execute",),
            ),
        )
    bounded = bounded_child_result({"answer": "x" * 100}, max_bytes=10)
    assert bounded["truncated"] is True
    assert bounded["result"] is None
    assert len(str(bounded["digest"])) == 64
    complete = bounded_child_result({"answer": "ok"}, max_bytes=1024)
    assert complete["truncated"] is False
    assert complete["result"] == {"answer": "ok"}
    artifact = bounded_child_result(
        {"artifact_ref": "file://vision/preview.png", "mime_type": "image/png"},
        max_bytes=1024,
    )
    assert artifact["result"]["artifact_ref"].startswith("file:")


def test_delegation_rejects_model_profile_outside_host_allowlist() -> None:
    with pytest.raises(ValueError, match="unknown child model profile"):
        build_delegated_invocation(
            _ctx(),
            workflow_id="child.workflow",
            spec=DelegationSpec(model_profile="unapproved"),
            allowed_model_profiles={"vision-local"},
        )
