"""Durable nested-workflow invocation and lineage operations."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from os import PathLike
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol

from kogwistar.engine_core.models import Grounding, Span
from kogwistar.json_types import JsonValue
from kogwistar.runtime.models import (
    WorkflowCheckpointNode,
    WorkflowDesignArtifact,
    WorkflowInvocationRequest,
    WorkflowRunNode,
    WorkflowState,
)
from kogwistar.runtime.telemetry import TraceContext
from .runtime_artifacts import _make_runtime_edge, _make_trace_span
from .step_context import RunResult

if TYPE_CHECKING:
    from kogwistar.engine_core.engine import GraphKnowledgeEngine


def derive_child_authority_context(
    parent: Mapping[str, Any] | None,
    child_state: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Carry trusted execution authority into a nested run without expansion."""

    if parent is None:
        return None
    context = dict(parent)
    if "effective_capabilities" in context:
        parent_caps = {str(item) for item in context.get("effective_capabilities", ())}
        requested = {
            str(item)
            for item in child_state.get("effective_capabilities", parent_caps)
        }
        context["effective_capabilities"] = tuple(sorted(parent_caps & requested))
    if "budget_limits" in context:
        context["budget_limits"] = dict(context["budget_limits"] or {})
    return MappingProxyType(context)


class InvocationRuntimeHost(Protocol):
    @property
    def workflow_engine(self) -> GraphKnowledgeEngine: ...

    @property
    def conversation_engine(self) -> GraphKnowledgeEngine: ...

    def _persist_workflow_design_artifact(
        self, design: WorkflowDesignArtifact
    ) -> None: ...

    def _child_workflow_initial_state(
        self,
        *,
        parent_state: WorkflowState,
        invocation: WorkflowInvocationRequest,
    ) -> WorkflowState: ...

    def _workflow_invocation_plan(
        self,
        *,
        invocation: WorkflowInvocationRequest,
        conversation_id: str,
        turn_node_id: str,
        parent_run_id: str,
    ) -> dict[str, str]: ...

    def _terminal_run_result(
        self, *, conversation_id: str, run_id: str
    ) -> RunResult | None: ...

    def _latest_checkpoint_for_run(
        self, *, conversation_id: str, run_id: str
    ) -> WorkflowCheckpointNode | None: ...

    def _conversation_node_exists(self, node_id: str) -> bool: ...

    def _persist_workflow_run(
        self,
        conversation_id: str,
        workflow_id: str,
        run_id: str,
        turn_node_id: str,
        status: str,
        run_metadata: Mapping[str, JsonValue] | None = None,
        trace_context: TraceContext | None = None,
    ) -> WorkflowRunNode: ...

    def _persist_workflow_invocation_lineage(
        self,
        *,
        conversation_id: str,
        parent_run_id: str,
        child_run_id: str,
        invocation: WorkflowInvocationRequest,
        result_state_key: str,
        required: bool = False,
    ) -> bool: ...

    def _run_workflow_invocation(
        self,
        *,
        invocation: WorkflowInvocationRequest,
        parent_state: WorkflowState,
        conversation_id: str,
        turn_node_id: str,
        parent_run_id: str,
        cache_dir: str | PathLike[str] | None = None,
        parent_trace_context: TraceContext | None = None,
        parent_authority_context: Mapping[str, Any] | None = None,
        parent_lease_renew: Callable[[], None] | None = None,
    ) -> RunResult: ...

    def resume_from_latest_checkpoint(
        self,
        *,
        run_id: str,
        workflow_id: str,
        conversation_id: str,
        turn_node_id: str,
        cache_dir: str | PathLike[str] | None = None,
        _parent_trace_context: TraceContext | None = None,
        _parent_authority_context: Mapping[str, Any] | None = None,
        _parent_lease_renew: Callable[[], None] | None = None,
    ) -> RunResult: ...

    def run(
        self,
        *,
        workflow_id: str,
        conversation_id: str,
        turn_node_id: str | None,
        initial_state: WorkflowState,
        run_id: str | None = None,
        cache_dir: str | PathLike[str] | None = None,
        _run_metadata: Mapping[str, JsonValue] | None = None,
        _trace_context: TraceContext | None = None,
        _authority_context: Mapping[str, Any] | None = None,
        _parent_lease_renew: Callable[[], None] | None = None,
    ) -> RunResult: ...

    def _trace_write_mode(self) -> AbstractContextManager[None]: ...


class WorkflowInvocationMixin:
    """Execute child workflows and persist their durable parent lineage."""

    def _run_workflow_invocation(
        self: InvocationRuntimeHost,
        *,
        invocation: WorkflowInvocationRequest,
        parent_state: WorkflowState,
        conversation_id: str,
        turn_node_id: str,
        parent_run_id: str,
        cache_dir: str | PathLike[str] | None = None,
        parent_trace_context: TraceContext | None = None,
        parent_authority_context: Mapping[str, Any] | None = None,
        parent_lease_renew: Callable[[], None] | None = None,
    ) -> RunResult:
        if invocation.workflow_design is not None:
            if str(invocation.workflow_design.workflow_id) != str(
                invocation.workflow_id
            ):
                raise ValueError(
                    "workflow_design.workflow_id must match workflow_id on the invocation"
                )
            self._persist_workflow_design_artifact(invocation.workflow_design)

        child_state = self._child_workflow_initial_state(
            parent_state=parent_state, invocation=invocation
        )
        child_authority_context = derive_child_authority_context(
            parent_authority_context, child_state
        )
        plan = self._workflow_invocation_plan(
            invocation=invocation,
            conversation_id=conversation_id,
            turn_node_id=turn_node_id,
            parent_run_id=parent_run_id,
        )
        child_result = self._terminal_run_result(
            conversation_id=plan["conversation_id"], run_id=plan["child_run_id"]
        )
        # Failed children with durable checkpoints remain retryable; successful
        # and cancelled children are terminal for idempotent parent resumes.
        if child_result is not None and child_result.status == "failure":
            child_result = None
        if child_result is None:
            child_checkpoint = self._latest_checkpoint_for_run(
                conversation_id=plan["conversation_id"],
                run_id=plan["child_run_id"],
            )
            child_run_node_id = f"wf_run|{plan['child_run_id']}"
            if child_checkpoint is None or not self._conversation_node_exists(
                child_run_node_id
            ):
                # Recreate the durable run node if a crash preserved only its checkpoint.
                self._persist_workflow_run(
                    conversation_id=plan["conversation_id"],
                    workflow_id=invocation.workflow_id,
                    run_id=plan["child_run_id"],
                    turn_node_id=plan["turn_node_id"],
                    status="planned",
                    run_metadata={
                        "parent_run_id": parent_run_id,
                        "invocation_key": invocation.invocation_key,
                        "result_state_key": plan["result_state_key"],
                    },
                )
            self._persist_workflow_invocation_lineage(
                conversation_id=conversation_id,
                parent_run_id=parent_run_id,
                child_run_id=plan["child_run_id"],
                invocation=invocation,
                result_state_key=plan["result_state_key"],
                required=True,
            )
            if child_checkpoint is not None:
                child_result = self.resume_from_latest_checkpoint(
                    run_id=plan["child_run_id"],
                    workflow_id=invocation.workflow_id,
                    conversation_id=plan["conversation_id"],
                    turn_node_id=plan["turn_node_id"],
                    cache_dir=cache_dir,
                    _parent_trace_context=parent_trace_context,
                    _parent_authority_context=child_authority_context,
                    _parent_lease_renew=parent_lease_renew,
                )
            else:
                child_result = self.run(
                    workflow_id=invocation.workflow_id,
                    conversation_id=plan["conversation_id"],
                    turn_node_id=plan["turn_node_id"],
                    initial_state=child_state,
                    run_id=plan["child_run_id"],
                    cache_dir=cache_dir,
                    _run_metadata={
                        "parent_run_id": parent_run_id,
                        "invocation_key": invocation.invocation_key,
                        "result_state_key": plan["result_state_key"],
                    },
                    _trace_context=parent_trace_context,
                    _authority_context=child_authority_context,
                    _parent_lease_renew=parent_lease_renew,
                )
        self._persist_workflow_invocation_lineage(
            conversation_id=conversation_id,
            parent_run_id=parent_run_id,
            child_run_id=str(child_result.run_id),
            invocation=invocation,
            result_state_key=plan["result_state_key"],
        )
        return child_result

    def _persist_workflow_invocation_lineage(
        self: InvocationRuntimeHost,
        *,
        conversation_id: str,
        parent_run_id: str,
        child_run_id: str,
        invocation: WorkflowInvocationRequest,
        result_state_key: str,
        required: bool = False,
    ) -> bool:
        """Record generic run lineage without making it workflow authority."""
        parent_id = f"wf_run|{parent_run_id}"
        child_id = f"wf_run|{child_run_id}"
        content = f"{parent_id} invoked {child_id}"
        try:
            parent_exists = self.conversation_engine.read.get_nodes(
                ids=[parent_id], limit=1
            )
        except Exception:
            parent_exists = []
        if not parent_exists:
            return False
        try:
            span = Span(
                **_make_trace_span(
                    conversation_id=conversation_id,
                    excerpt=content,
                    doc_id=f"conv:{conversation_id}",
                )
            )
            edge = _make_runtime_edge(
                edge_id=f"wf_invoked|{parent_run_id}|{child_run_id}",
                relation="wf_invoked",
                label="workflow invocation",
                summary=content,
                doc_id=f"wf_invoked|{parent_run_id}|{child_run_id}",
                conversation_id=conversation_id,
                source_ids=[parent_id],
                target_ids=[child_id],
                mentions=[Grounding(spans=[span])],
                run_id=parent_run_id,
                metadata={
                    "parent_run_id": parent_run_id,
                    "child_run_id": child_run_id,
                    "workflow_id": invocation.workflow_id,
                    "invocation_key": invocation.invocation_key,
                    "result_state_key": result_state_key,
                },
            )
            with self._trace_write_mode():
                self.conversation_engine.write.add_edge(edge)
            return True
        except Exception:
            logging.getLogger(__name__).exception(
                "workflow invocation lineage persistence failed"
            )
            if required:
                raise
            return False

    def run_subworkflow(
        self: InvocationRuntimeHost,
        *,
        workflow_id: str,
        parent_state: WorkflowState,
        conversation_id: str,
        turn_node_id: str,
        parent_run_id: str,
        result_state_key: str | None = None,
        workflow_design: WorkflowDesignArtifact | None = None,
        initial_state: dict[str, Any] | None = None,
        run_id: str | None = None,
        cache_dir: str | PathLike[str] | None = None,
    ) -> RunResult:
        """Public helper for explicit nested workflow execution."""
        invocation = WorkflowInvocationRequest(
            workflow_id=workflow_id,
            initial_state=dict(initial_state or {}),
            result_state_key=result_state_key,
            workflow_design=workflow_design,
            run_id=run_id,
        )
        return self._run_workflow_invocation(
            invocation=invocation,
            parent_state=parent_state,
            conversation_id=conversation_id,
            turn_node_id=turn_node_id,
            parent_run_id=parent_run_id,
            cache_dir=cache_dir,
        )
