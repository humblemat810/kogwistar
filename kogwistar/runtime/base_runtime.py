from __future__ import annotations

import copy
import logging
import warnings
from typing import Any

from .models import StateUpdate, WorkflowDesignArtifact, WorkflowInvocationRequest, WorkflowState
from .routing import RouteComputation, compute_route_next

RESERVED_ROOT_KEYS = {
    "_deps",
    "_rt_join",
}

RESERVED_PREFIXES = ("_", "__")


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
    """Apply runtime state delta in place.

    Single reducer for sync runtime, async runtime, and replay.
    """
    if update and state_update:
        raise Exception("Either update or state_update can be used")

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
