"""Typed synchronous workflow-step execution, separate from scheduling."""

from __future__ import annotations

import queue
import threading
import traceback
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from os import PathLike
from typing import Any, Protocol, TypeAlias

from kogwistar.json_types import JsonValue

from .models import (
    RunFailure,
    RunSuccess,
    StepRunResult,
    WorkflowInvocationRequest,
    WorkflowNode,
    WorkflowState,
)
from .step_context import RunResult
from .step_context import (
    LaneMessageEventSinkLike,
    LaneMessageSenderLike,
    StepContext,
    StepFn,
    StepResolver,
)
from .telemetry import EventEmitter, TraceContext
from kogwistar.utils.log import bind_log_context

CompletedStep: TypeAlias = tuple[str, StepRunResult, int, str, str | None, str, int]


class StepWorkerConversationHost(Protocol):
    """Transactional engine surface used by a workflow step."""

    def uow(self) -> AbstractContextManager[object]: ...


class StepWorkerRuntime(Protocol):
    """Runtime services needed by one worker, excluding scheduler ownership."""

    @property
    def step_resolver(self) -> StepResolver: ...

    @property
    def conversation_engine(self) -> StepWorkerConversationHost: ...

    @property
    def lane_message_sender(
        self,
    ) -> LaneMessageSenderLike | Callable[..., object] | None: ...

    @property
    def lane_message_event_sink(
        self,
    ) -> LaneMessageEventSinkLike | Callable[..., object] | None: ...

    @property
    def emitter(self) -> EventEmitter: ...

    def _should_step_uow(self, op: str, state: WorkflowState) -> bool: ...

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


@dataclass
class SyncStepWorker:
    """Execute one resolver step and publish its completion to the scheduler."""

    runtime: StepWorkerRuntime
    workflow_id: str
    run_id: str
    conversation_id: str | None
    turn_node_id: str | None
    nodes: Mapping[str, WorkflowNode]
    trace_context: TraceContext | None
    message_queue: queue.Queue[dict[str, JsonValue]]
    done_queue: queue.Queue[CompletedStep]
    work_condition: threading.Condition
    authority_context: Mapping[str, Any] | None
    cancel_requested: Callable[[], bool]
    parent_lease_renew: Callable[[], None] | None
    renew_run_ownership: Callable[[], None]
    apply_child_result: Callable[
        [WorkflowState, WorkflowInvocationRequest, RunResult], None
    ]
    now_ms: Callable[[], int]

    def __call__(
        self,
        node_id: str,
        state: WorkflowState,
        token_id: str,
        parent_token_id: str | None,
        step_seq: int,
        mask: int,
        cache_dir: str | PathLike[str] | None = None,
    ) -> None:
        thread = threading.current_thread()
        previous_name = thread.name
        context: StepContext | None = None
        try:
            thread.name = (
                f"rt-wf-{self.workflow_id}-wf-{self.run_id}-step-{step_seq}-{node_id}"
            )
            workflow_node = self.nodes[node_id]
            op = workflow_node.op
            started_at = self.now_ms()
            status = "ok"
            with bind_log_context(step_id=f"{step_seq}-{node_id}", op=op):
                try:
                    context = StepContext(
                        run_id=str(self.run_id),
                        workflow_id=str(self.workflow_id),
                        workflow_node_id=str(node_id),
                        op=str(op),
                        token_id=str(token_id),
                        attempt=1,
                        step_seq=int(step_seq),
                        conversation_id=(
                            str(self.conversation_id)
                            if self.conversation_id is not None
                            else None
                        ),
                        turn_node_id=(
                            str(self.turn_node_id)
                            if self.turn_node_id is not None
                            else None
                        ),
                        trace_context=(
                            self.trace_context.child_span(
                                token_id=str(token_id),
                                step_seq=int(step_seq),
                                node_id=str(node_id),
                                attempt=1,
                            )
                            if self.trace_context is not None
                            else None
                        ),
                        state=state,
                        message_queue=self.message_queue,
                        lane_message_sender=self.runtime.lane_message_sender,
                        lane_message_event_sink=self.runtime.lane_message_event_sink,
                        events=self.runtime.emitter,
                        cache_dir=cache_dir,
                        authority_context=(
                            self.authority_context
                            if self.authority_context is not None
                            else {"effective_capabilities": ()}
                        ),
                    )
                    self.runtime.emitter.step_started(context.trace_ctx)

                    if self.cancel_requested():
                        status = "cancelled"
                        result: StepRunResult = RunFailure(
                            conversation_node_id=None,
                            status="failure",
                            errors=["Run cancelled"],
                            state_update=[],
                        )
                    else:
                        if self.parent_lease_renew is not None:
                            self.parent_lease_renew()
                        step_fn: StepFn = self.runtime.step_resolver(op)
                        with (
                            self.runtime.conversation_engine.uow()
                            if self.runtime._should_step_uow(op, state)
                            else nullcontext()
                        ):
                            result = step_fn(context)
                        if result.status == "suspended":
                            status = "suspended"
                        elif result.status == "failure":
                            status = "failure"
                        else:
                            status = "ok"
                        if isinstance(result, RunSuccess):
                            for invocation in result.workflow_invocations:
                                self.renew_run_ownership()
                                child_result = self.runtime._run_workflow_invocation(
                                    invocation=invocation,
                                    parent_state=state,
                                    conversation_id=str(self.conversation_id),
                                    turn_node_id=str(self.turn_node_id),
                                    parent_run_id=str(self.run_id),
                                    cache_dir=cache_dir,
                                    parent_trace_context=context.trace_ctx,
                                    parent_authority_context=context.authority_context,
                                    parent_lease_renew=self.renew_run_ownership,
                                )
                                self.renew_run_ownership()
                                if child_result.status != "succeeded":
                                    status = "failure"
                                    result = RunFailure(
                                        conversation_node_id=None,
                                        status="failure",
                                        errors=[
                                            "Nested workflow "
                                            f"{invocation.workflow_id!r} "
                                            f"returned status {child_result.status!r}"
                                        ]
                                        + list(child_result.errors or []),
                                        state_update=[],
                                    )
                                    break
                                self.apply_child_result(state, invocation, child_result)
                except Exception as exc:
                    status = "error"
                    result = RunFailure(
                        conversation_node_id=None,
                        status="failure",
                        errors=[str(exc) + "\n" + traceback.format_exc()],
                        state_update=[],
                    )

                try:
                    self.message_queue.put_nowait(result.model_dump())
                except queue.Full:
                    raise
                completed_at = self.now_ms()
                try:
                    if context is None:
                        raise ValueError("ctx is None")
                    self.runtime.emitter.step_completed(
                        context.trace_ctx,
                        status=str(status),
                        duration_ms=max(0, completed_at - started_at),
                    )
                except Exception:
                    status = "status_report_error"
                if status in ["error", "status_report_error"]:
                    error_message = f"error on {node_id}, {result}"
                    # Preserve the existing resolver-node failure policy lookup.
                    if getattr(workflow_node, "fail_actiion", "raise") == "raise":
                        raise Exception(error_message)

                self.done_queue.put(
                    (
                        node_id,
                        result,
                        max(0, completed_at - started_at),
                        token_id,
                        parent_token_id,
                        status,
                        mask,
                    )
                )
                with self.work_condition:
                    self.work_condition.notify_all()
        finally:
            thread.name = previous_name
