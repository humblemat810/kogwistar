from __future__ import annotations

import copy
import logging
from typing import Any

from .models import WorkflowDesignArtifact, WorkflowInvocationRequest, WorkflowState
from .routing import RouteComputation, compute_route_next


class BaseRuntime:
    """Pure shared runtime helpers.

    Keep only logic that is scheduler-agnostic and backend-agnostic so sync and
    async runtimes can inherit it without semantic drift.
    """

    workflow_engine: Any
    step_resolver: Any
    predicate_registry: dict[str, Any]

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
