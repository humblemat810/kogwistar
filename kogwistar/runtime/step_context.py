"""Typed execution context and result contracts for workflow steps."""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable, Mapping
from dataclasses import InitVar, dataclass, field
from os import PathLike
from types import MappingProxyType, TracebackType
from typing import Any, Protocol, TypeAlias

from kogwistar.json_types import JsonValue

from .models import StepRunResult, WorkflowState
from .telemetry import BoundLoggerAdapter, EventEmitter, TraceContext, bind_logger

Json = JsonValue


class LaneMessageSenderLike(Protocol):
    """Structural contract for durable cross-lane message submission."""

    def __call__(self, **kwargs: Json) -> object: ...


class LaneMessageEventSinkLike(Protocol):
    """Structural contract for best-effort lifecycle event mirroring."""

    def __call__(self, event: dict[str, Json]) -> object: ...


StepFn: TypeAlias = Callable[["StepContext"], StepRunResult]


class StepResolver(Protocol):
    """Resolve a persisted workflow operation to a typed step handler."""

    def __call__(self, op: str) -> StepFn: ...


@dataclass
class RunResult:
    run_id: str
    final_state: WorkflowState
    mq: queue.Queue[dict[str, Json]]
    status: str = "succeeded"
    errors: list[str] = field(default_factory=list)


class _StateWriteTxn:
    def __init__(self, ctx: StepContext) -> None:
        self._ctx = ctx

    def __enter__(self) -> WorkflowState:
        self._ctx._state_lock.acquire()
        return self._ctx._state

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._ctx._state_lock.release()


@dataclass
class RouteDecision:
    """Structured routing outcome used for trace emission."""

    evaluated: list[tuple[str, bool]]
    selected: list[tuple[str, str, str]]


@dataclass
class StepContext:
    run_id: str
    workflow_id: str
    workflow_node_id: str
    op: str
    token_id: str
    attempt: int
    step_seq: int
    cache_dir: str | PathLike | None
    op_version: str = "v1"
    op_environment_id: str | None = None
    conversation_id: str | None = None
    turn_node_id: str | None = None
    trace_context: TraceContext | None = field(default=None, repr=False)
    authority_context: Mapping[str, Any] = field(default_factory=dict, repr=False)
    message_queue: queue.Queue[dict[str, Json]] = field(
        repr=False, default_factory=queue.Queue
    )
    lane_message_sender: LaneMessageSenderLike | None = field(repr=False, default=None)
    lane_message_event_sink: LaneMessageEventSinkLike | None = field(
        repr=False, default=None
    )
    events: EventEmitter | None = field(repr=False, default=None)
    state: InitVar[WorkflowState] = None  # type: ignore[assignment]
    idempotency_key: str | None = None
    _state: WorkflowState = field(init=False, repr=False)
    _state_lock: threading.RLock = field(
        default_factory=threading.RLock, init=False, repr=False
    )
    log: BoundLoggerAdapter = field(init=False, repr=False)

    def __post_init__(self, state: WorkflowState) -> None:
        self._state = state
        self.log = bind_logger(logging.getLogger("workflow.resolver"), self.trace_ctx)

    @property
    def state_view(self) -> Mapping[str, object]:
        return MappingProxyType(self._state)

    @property
    def state_write(self) -> _StateWriteTxn:
        return _StateWriteTxn(self)

    def publish(self, msg: dict[str, Json]) -> None:
        self.message_queue.put(msg)

    def send_lane_message(self, **kwargs: Json) -> object:
        sender = self.lane_message_sender
        if not callable(sender):
            raise RuntimeError(
                "lane message sender not configured; durable cross-lane delivery unavailable"
            )
        result = sender(**kwargs)
        sink = self.lane_message_event_sink
        if callable(sink):
            message_id = getattr(result, "message_id", None)
            if not message_id and isinstance(result, dict):
                message_id = result.get("message_id")
            try:
                sink(
                    {
                        "event_type": "worker.requested",
                        "run_id": str(kwargs.get("run_id") or self.run_id),
                        "message_id": str(message_id or ""),
                        "conversation_id": str(
                            kwargs.get("conversation_id") or self.conversation_id or ""
                        ),
                        "inbox_id": str(kwargs.get("inbox_id") or ""),
                        "sender_id": str(kwargs.get("sender_id") or ""),
                        "recipient_id": str(kwargs.get("recipient_id") or ""),
                        "msg_type": str(kwargs.get("msg_type") or ""),
                        "status": "pending",
                        "correlation_id": kwargs.get("correlation_id"),
                        "step_id": kwargs.get("step_id"),
                        "workflow_id": self.workflow_id,
                        "workflow_node_id": self.workflow_node_id,
                        "step_seq": int(self.step_seq),
                    }
                )
            except Exception:
                logging.getLogger(__name__).exception(
                    "lane message event sink failed: run_id=%s", self.run_id
                )
        return result

    def emit_lane_message_event(self, event: dict[str, Json]) -> None:
        sink = self.lane_message_event_sink
        if not callable(sink):
            raise RuntimeError(
                "lane message event sink not configured; lifecycle mirroring unavailable"
            )
        sink(event)

    def drain(self, max_items: int = 200) -> list[dict[str, Json]]:
        raise Exception(
            "current design does not allow mq drained in context, but only by orchestrator"
        )

    @property
    def trace_ctx(self) -> TraceContext:
        if self.trace_context is not None:
            return self.trace_context
        return TraceContext(
            run_id=self.run_id,
            token_id=self.token_id,
            step_seq=self.step_seq,
            node_id=self.workflow_node_id,
            attempt=self.attempt,
            conversation_id=self.conversation_id,
            turn_node_id=self.turn_node_id,
        )

    @property
    def conv_trace_ctx(self) -> TraceContext:
        return self.trace_ctx
