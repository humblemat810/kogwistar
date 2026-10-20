"""Synchronous runtime adapter for shared workflow route computation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from .contract import Predicate
from .models import StepRunResult, WorkflowEdge, WorkflowNode, WorkflowState
from .routing import RouteComputation
from .step_context import RouteDecision


class RuntimeRoutingHost(Protocol):
    """Minimum inherited runtime surface required by the routing adapter."""

    predicate_registry: dict[str, Predicate]

    def _compute_route_next_shared(
        self,
        *,
        edges: list[object],
        state: WorkflowState,
        last_result: object,
        fanout: bool,
        predicate_registry: dict[str, Predicate],
        nodes: Mapping[str, object] | None = None,
        sort_edges: bool = False,
    ) -> RouteComputation: ...


class WorkflowRouteNextMixin:
    """Expose route computation in the sync runtime's public result shape."""

    def _route_next(
        self: RuntimeRoutingHost,
        edges: list[WorkflowEdge],
        state: WorkflowState,
        last_result: StepRunResult,
        fanout: bool,
        nodes: Mapping[str, WorkflowNode] | None = None,
    ) -> tuple[list[str], RouteDecision]:
        """Select destinations while preserving predicate evaluation evidence.

        Tracing remains the caller's responsibility. The shared route engine
        remains authoritative for predicate ordering and default selection.
        """
        computed = self._compute_route_next_shared(
            edges=list(edges),
            state=state,
            last_result=last_result,
            fanout=fanout,
            predicate_registry=self.predicate_registry,
            nodes=nodes,
        )
        return computed.next_node_ids, RouteDecision(
            evaluated=list(computed.evaluated),
            selected=list(computed.selected),
        )
