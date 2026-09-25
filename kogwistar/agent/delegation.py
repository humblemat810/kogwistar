"""Bounded subagent composition over ordinary nested workflow invocation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from kogwistar.runtime.models import RunSuccess, WorkflowDesignArtifact, WorkflowInvocationRequest


@dataclass(frozen=True, slots=True)
class DelegationSpec:
    """Child-run policy; it grants no capability beyond the parent."""

    model_profile: str
    requested_capabilities: tuple[str, ...] = ()
    context_keys: tuple[str, ...] = ()
    max_output_bytes: int = 16 * 1024
    max_steps: int | None = None
    max_model_calls: int | None = None
    max_tokens: int | None = None
    max_cost: float | None = None

    def __post_init__(self) -> None:
        if not str(self.model_profile).strip():
            raise ValueError("child model_profile must be non-empty")
        if int(self.max_output_bytes) < 1:
            raise ValueError("max_output_bytes must be positive")
        for name in ("max_steps", "max_model_calls", "max_tokens"):
            value = getattr(self, name)
            if value is not None and int(value) < 1:
                raise ValueError(f"{name} must be positive")
        if self.max_cost is not None and float(self.max_cost) <= 0:
            raise ValueError("max_cost must be positive")


def _effective_capabilities(
    requested: tuple[str, ...], parent: Mapping[str, Any]
) -> tuple[str, ...]:
    parent_caps = {str(item) for item in parent.get("effective_capabilities", ())}
    requested_caps = {str(item) for item in requested}
    if not requested_caps <= parent_caps:
        raise PermissionError(
            "child requested capabilities absent from parent: "
            + ", ".join(sorted(requested_caps - parent_caps))
        )
    return tuple(sorted(requested_caps))


def delegated_initial_state(
    parent_state: Mapping[str, Any], spec: DelegationSpec
) -> dict[str, Any]:
    """Copy only explicitly selected ordinary state into child context."""

    child: dict[str, Any] = {
        "agent_model_profile": spec.model_profile,
        "effective_capabilities": list(_effective_capabilities(spec.requested_capabilities, parent_state)),
    }
    for key in spec.context_keys:
        if key in parent_state and not str(key).startswith("_"):
            child[str(key)] = parent_state[key]
    for key, value in {
        "step_budget": spec.max_steps,
        "call_budget": spec.max_model_calls,
        "token_budget": spec.max_tokens,
        "cost_budget": spec.max_cost,
    }.items():
        if value is not None:
            child[key] = value
    return child


def build_delegated_invocation(
    ctx: Any,
    *,
    workflow_id: str,
    spec: DelegationSpec,
    workflow_design: WorkflowDesignArtifact | None = None,
    result_state_key: str = "agent_subagent_result",
) -> WorkflowInvocationRequest:
    """Build deterministic invoke-and-await request with isolated child state."""

    if workflow_design is not None and workflow_design.workflow_id != workflow_id:
        raise ValueError("workflow_design.workflow_id must match workflow_id")
    invocation_key = f"{ctx.run_id}:{ctx.step_seq}:{workflow_id}:{spec.model_profile}"
    return WorkflowInvocationRequest(
        workflow_id=workflow_id,
        workflow_design=workflow_design,
        initial_state=delegated_initial_state(ctx.state_view, spec),
        result_state_key=result_state_key,
        invocation_key=invocation_key,
        conversation_id=getattr(ctx, "conversation_id", None),
        turn_node_id=getattr(ctx, "turn_node_id", None),
    )


def bounded_child_result(value: Any, *, max_bytes: int) -> dict[str, Any]:
    """Return structured bounded evidence; never expose complete child context."""

    if int(max_bytes) < 1:
        raise ValueError("max_bytes must be positive")
    payload: Any = value if isinstance(value, Mapping) else {"value": value}
    encoded = json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    if len(encoded) <= max_bytes:
        return {"result": json.loads(encoded.decode("utf-8")), "truncated": False, "digest": digest}
    return {
        "result": None,
        "truncated": True,
        "digest": digest,
        "size_bytes": len(encoded),
    }


def make_delegation_handler(
    *,
    workflow_id: str,
    spec: DelegationSpec,
    workflow_design: WorkflowDesignArtifact | None = None,
    result_state_key: str = "agent_subagent_result",
) -> Any:
    """Create ordinary resolver handler returning one nested invocation."""

    def _handler(ctx: Any) -> RunSuccess:
        return RunSuccess(
            state_update=[],
            workflow_invocations=[
                build_delegated_invocation(
                    ctx,
                    workflow_id=workflow_id,
                    spec=spec,
                    workflow_design=workflow_design,
                    result_state_key=result_state_key,
                )
            ],
        )

    return _handler


__all__ = [
    "DelegationSpec",
    "bounded_child_result",
    "build_delegated_invocation",
    "delegated_initial_state",
    "make_delegation_handler",
]
