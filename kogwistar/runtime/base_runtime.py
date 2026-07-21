from __future__ import annotations

import copy
import logging
import warnings
from typing import Any

from .._rust_bridge import (
    RustParityError,
    json_contract_compatible,
    runtime_apply_state_update,
    runtime_decide_dispatch,
    runtime_implementation_mode,
    runtime_plan_nested_invocation,
    runtime_scheduler_tick,
)
from ..id_provider import stable_id
from .models import StateUpdate, WorkflowDesignArtifact, WorkflowInvocationRequest, WorkflowState
from .routing import RouteComputation, compute_route_next

RESERVED_ROOT_KEYS = {
    "_deps",
    "_rt_join",
}

RESERVED_PREFIXES = ("_", "__")
NON_CHECKPOINT_STATE_KEYS = {
    "_deps",
    "dream_deps",  # legacy dream DI key; keep out of checkpoints
}


def _native_state_update_payload(
    state: WorkflowState,
    state_update: list[tuple[str, dict[str, Any]]] | list[StateUpdate],
    update: dict | None,
    state_schema: dict[str, Any] | None,
) -> dict[str, Any]:
    """JSON transport form; state-update pairs are tuples in public Python API."""
    return {
        "state": copy.deepcopy(state),
        "state_update": [list(item) for item in state_update],
        "update": update,
        "state_schema": state_schema or {},
    }


def _native_state_update_safe(
    state: WorkflowState,
    state_update: list[tuple[str, dict[str, Any]]] | list[StateUpdate],
    update: dict | None,
    state_schema: dict[str, Any] | None,
) -> bool:
    """Keep Python for inputs whose observable legacy failure is not a contract fold."""
    for item in state_update:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            return False
        mode, payload = item
        if not isinstance(payload, dict):
            return False
        if mode == "a" and any(key in state and not isinstance(state[key], list) for key in payload):
            return False
        if mode == "e":
            if any(key in state and not isinstance(state[key], list) for key in payload):
                return False
            if any(not isinstance(value, (list, str, dict)) for value in payload.values()):
                return False
    if update:
        schema = state_schema or {}
        append_keys = (key for key in update if schema.get(key) == "a")
        for key in append_keys:
            if key in state and not isinstance(state[key], list):
                return False
            if not isinstance(update[key], (list, str, dict)):
                return False
    return True


def validate_initial_state(initial_state: WorkflowState):
    """Validate user-provided initial workflow state.

    Workflow state is user-land except for a small set of underscore-prefixed
    keys reserved for runtime/DI plumbing.
    """
    allowed_underscore = {"_deps", "_rt_join"}

    for key in initial_state:
        if key in allowed_underscore:
            warnings.warn(
                f"Using advanced underscore state key '{key}'. This key is reserved for runtime/DI plumbing.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        if key.startswith(RESERVED_PREFIXES):
            raise ValueError(
                f"Keys starting with '_' or '__' are reserved. Invalid key: '{key}'"
            )


def apply_state_update_inplace(
    mute_state: WorkflowState,
    state_update: list[tuple[str, dict[str, Any]]] | list[StateUpdate],
    update: dict | None = None,
    *,
    state_schema: dict[str, Any] | None = None,
):
    """Apply a workflow-runtime state delta in place.

    Single reducer for sync runtime, async runtime, and replay.
    """
    if update and state_update:
        error = Exception("Either update or state_update can be used")
        error.code = "KOGWISTAR_CONTRACT_STATE_UPDATE_CONFLICT"
        raise error

    mode = runtime_implementation_mode()
    # Inspect live inputs before copying them.  Runtime state can contain
    # process-local plumbing (notably ``_deps``) such as locks; deepcopy would
    # raise before the JSON boundary can decline that state and retain the
    # legacy Python reducer.
    native_transport = {
        "state": mute_state,
        "state_update": [list(item) for item in state_update],
        "update": update,
        "state_schema": state_schema or {},
    }
    json_compatible = json_contract_compatible(native_transport) and _native_state_update_safe(
        mute_state, state_update, update, state_schema
    )
    native_state: dict[str, Any] | None = None
    if mode != "python" and json_compatible:
        native_payload = _native_state_update_payload(
            mute_state, state_update, update, state_schema
        )
        native_state = runtime_apply_state_update(payload=native_payload)

    for update_item in state_update:
        update_item: tuple[str, dict[str, Any]] | StateUpdate
        if update_item[0] == "a":
            append_dict: dict = update_item[1]
            for k, v in append_dict.items():
                mute_state.setdefault(k, []).append(v)
        elif update_item[0] == "u":
            update_dict: dict = update_item[1]
            for k, v in update_dict.items():
                mute_state[k] = v
        elif update_item[0] == "e":
            update_dict: dict = update_item[1]
            for k, v in update_dict.items():
                mute_state.setdefault(k, []).extend(v)
    if update:
        schema = state_schema or {}
        for k, v in update.items():
            if op := schema.get(k):
                pass
            else:
                op = "u"
            if op == "a":
                mute_state.setdefault(k, []).extend(v)
            else:
                mute_state[k] = v

    if native_state is not None:
        # Rust owns the canonical fold.  The thin Python facade materializes the
        # same delta in place so untouched and inserted mutable objects retain
        # the identity guarantees of the existing public API.
        if native_state != mute_state:
            raise RustParityError(
                "Rust parity mismatch for runtime_state_update: "
                f"python={mute_state!r}, rust={native_state!r}"
            )


def checkpointable_state_copy(state: WorkflowState) -> WorkflowState:
    """Return checkpoint-safe state by dropping process-local DI plumbing."""
    return {
        k: v for k, v in state.items() if k not in NON_CHECKPOINT_STATE_KEYS
    }


class BaseRuntime:
    """Pure shared runtime helpers.

    Keep only logic that is scheduler-agnostic and backend-agnostic so sync and
    async runtimes can inherit it without semantic drift.
    """

    workflow_engine: Any
    step_resolver: Any
    predicate_registry: dict[str, Any]

    validate_initial_state = staticmethod(validate_initial_state)
    apply_state_update_inplace = staticmethod(apply_state_update_inplace)
    checkpointable_state_copy = staticmethod(checkpointable_state_copy)

    @staticmethod
    def _edge_priority(edge: Any) -> int:
        md = getattr(edge, "metadata", {}) or {}
        try:
            return int(md.get("wf_priority", 100))
        except Exception:
            return 100

    @staticmethod
    def _compute_route_next_shared(
        *,
        edges: list[Any],
        state: WorkflowState,
        last_result: Any,
        fanout: bool,
        predicate_registry: dict[str, Any],
        nodes: dict[str, Any] | None = None,
        sort_edges: bool = False,
    ) -> RouteComputation:
        route_edges = list(edges)
        if sort_edges:
            route_edges = sorted(route_edges, key=BaseRuntime._edge_priority)
        return compute_route_next(
            edges=route_edges,
            state=dict(state),
            last_result=last_result,
            fanout=fanout,
            predicate_registry=predicate_registry,
            nodes=nodes,
        )

    def _close_sandbox_run(self, run_id: str) -> None:
        close_run = getattr(self.step_resolver, "close_sandbox_run", None)
        if callable(close_run):
            try:
                close_run(str(run_id))
            except Exception:
                logging.getLogger("workflow.runtime").exception(
                    "Failed to clean up sandbox resources for run %s",
                    run_id,
                )

    def _persist_workflow_design_artifact(
        self, design: WorkflowDesignArtifact
    ) -> None:
        for node in design.nodes:
            self.workflow_engine.write.add_node(node)
        for edge in design.edges:
            self.workflow_engine.write.add_edge(edge)

    def _child_workflow_initial_state(
        self,
        *,
        parent_state: WorkflowState,
        invocation: WorkflowInvocationRequest,
    ) -> WorkflowState:
        child_state: WorkflowState = dict(parent_state)  # type: ignore[arg-type]
        child_state.pop("_rt_join", None)
        if invocation.initial_state:
            child_state.update(copy.deepcopy(invocation.initial_state))  # type: ignore[arg-type]

        deps = dict(child_state.get("_deps") or parent_state.get("_deps") or {})  # type: ignore[union-attr]
        deps["workflow_runtime"] = self  # type: ignore[index]
        child_state["_deps"] = deps  # type: ignore[index]
        return child_state

    @staticmethod
    def _workflow_invocation_plan(
        *,
        invocation: WorkflowInvocationRequest,
        conversation_id: str,
        turn_node_id: str,
        parent_run_id: str,
    ) -> dict[str, str]:
        effective_turn_node_id = invocation.turn_node_id or turn_node_id
        python_value = {
            "child_run_id": invocation.run_id
            or str(
                stable_id(
                    "workflow.child_run",
                    parent_run_id,
                    invocation.workflow_id,
                    invocation.result_state_key or "",
                    effective_turn_node_id,
                )
            ),
            "conversation_id": invocation.conversation_id or conversation_id,
            "turn_node_id": effective_turn_node_id,
            "result_state_key": invocation.result_state_key
            or f"workflow_result::{invocation.workflow_id}",
        }
        return runtime_plan_nested_invocation(
            payload={
                "parent_run_id": parent_run_id,
                "workflow_id": invocation.workflow_id,
                "result_state_key": invocation.result_state_key,
                "run_id": invocation.run_id,
                "parent_conversation_id": conversation_id,
                "conversation_id": invocation.conversation_id,
                "parent_turn_node_id": turn_node_id,
                "turn_node_id": invocation.turn_node_id,
            },
            python_value=python_value,
        )

    @staticmethod
    def _runtime_dispatch_decision(
        *, max_workers: int, inflight: int, pending: int, cancelling: bool
    ) -> dict[str, Any]:
        worker_limit = max(1, int(max_workers))
        launch_capacity = (
            0
            if cancelling
            else min(max(0, worker_limit - int(inflight)), int(pending))
        )
        python_value = {
            "worker_limit": worker_limit,
            "launch_capacity": launch_capacity,
            "should_launch": launch_capacity > 0,
            "should_drain": bool(cancelling and inflight > 0),
            "cancellation_complete": bool(
                cancelling and inflight == 0 and pending == 0
            ),
        }
        return runtime_decide_dispatch(
            payload={
                "max_workers": int(max_workers),
                "inflight": int(inflight),
                "pending": int(pending),
                "cancelling": bool(cancelling),
            },
            python_value=python_value,
        )

    @staticmethod
    def _runtime_scheduler_tick(
        *,
        pending: list[tuple[str, int, str, str | None]],
        inflight: int,
        max_workers: int,
        cancelling: bool,
    ) -> dict[str, Any]:
        payload_pending = [
            {
                "node_id": str(node_id),
                "join_mask": int(join_mask),
                "token_id": str(token_id),
                "parent_token_id": parent_token_id,
            }
            for node_id, join_mask, token_id, parent_token_id in pending
        ]
        worker_limit = max(1, int(max_workers))
        dispatch_count = (
            0
            if cancelling
            else min(max(0, worker_limit - int(inflight)), len(pending))
        )
        python_value = {
            "dispatch": payload_pending[:dispatch_count],
            "pending": payload_pending[dispatch_count:],
            "should_drain": bool(cancelling and inflight > 0),
            "cancellation_complete": bool(
                cancelling and inflight == 0 and not pending
            ),
        }
        return runtime_scheduler_tick(
            payload={
                "pending": payload_pending,
                "inflight": int(inflight),
                "max_workers": int(max_workers),
                "cancelling": bool(cancelling),
            },
            python_value=python_value,
        )

    def _apply_workflow_invocation_result(
        self,
        *,
        state: WorkflowState,
        invocation: WorkflowInvocationRequest,
        child_result: Any,
    ) -> None:
        result_key = (
            invocation.result_state_key or f"workflow_result::{invocation.workflow_id}"
        )
        child_state = dict(child_result.final_state)
        child_state.pop("_deps", None)
        child_state.pop("_rt_join", None)
        state[result_key] = copy.deepcopy(child_state)
        state[f"{result_key}__run_id"] = str(child_result.run_id)
        state[f"{result_key}__status"] = str(child_result.status)
        state[f"{result_key}__workflow_id"] = str(invocation.workflow_id)
