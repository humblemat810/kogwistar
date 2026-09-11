"""Optional, best-effort OpenTelemetry projection for runtime telemetry events."""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from collections.abc import Mapping
from typing import Any, Optional, Protocol


class _Span(Protocol):
    def add_event(
        self, name: str, attributes: Optional[Mapping[str, Any]] = None
    ) -> None: ...

    def end(self) -> None: ...


class _Tracer(Protocol):
    def start_span(
        self,
        name: str,
        *,
        context: Any = None,
        attributes: Optional[Mapping[str, Any]] = None,
    ) -> _Span: ...


def opentelemetry_available() -> bool:
    """Return whether the optional OpenTelemetry API and SDK are installed."""
    try:
        import opentelemetry.trace  # noqa: F401
        import opentelemetry.sdk.trace  # noqa: F401
    except ImportError:
        return False
    return True


def try_create_opentelemetry_sink(**kwargs: Any) -> "OpenTelemetrySink | None":
    """Create optional sink, or disable observability when OTel is absent."""
    try:
        return OpenTelemetrySink.from_opentelemetry(**kwargs)
    except ImportError:
        return None


class OpenTelemetrySink:
    """Project selected workflow lifecycle events into OpenTelemetry spans.

    ``emit`` only enqueues a copied event. Exporter and SDK failures are contained
    in the worker thread so runtime execution remains authoritative.
    """

    _RUN_END_EVENTS = {
        "workflow_run_completed",
        "workflow_run_failed",
        "workflow_run_cancelled",
        "workflow_run_suspended",
    }
    _STEP_EVENTS = {"step_attempt_started", "step_attempt_completed"}

    def __init__(
        self,
        tracer: _Tracer,
        *,
        context_factory: Any = None,
        queue_max: int = 1_000,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if queue_max <= 0:
            raise ValueError("queue_max must be positive")
        self._tracer = tracer
        self._context_factory = context_factory
        self._log = logger or logging.getLogger(__name__)
        self._queue: "queue.Queue[dict[str, Any]]" = queue.Queue(maxsize=queue_max)
        self._stop = threading.Event()
        self._closed = False
        self._lock = threading.Lock()
        self._run_spans: dict[str, _Span] = {}
        self._step_spans: dict[tuple[str, str, int, str, int], _Span] = {}
        self.dropped_events = 0
        self.export_errors = 0
        self._thread = threading.Thread(
            target=self._run,
            name="kogwistar-otel-exporter",
            daemon=True,
        )
        self._thread.start()

    @classmethod
    def from_opentelemetry(
        cls,
        *,
        instrumentation_name: str = "kogwistar.runtime",
        **kwargs: Any,
    ) -> "OpenTelemetrySink":
        """Create sink using installed OTel API; raises ImportError if absent."""
        from opentelemetry import trace

        return cls(
            trace.get_tracer(instrumentation_name),
            context_factory=trace.set_span_in_context,
            **kwargs,
        )

    def emit(self, evt: dict[str, Any]) -> None:
        """Best-effort non-blocking enqueue; full queues drop newest event."""
        if self._stop.is_set():
            return
        try:
            self._queue.put_nowait(dict(evt))
        except queue.Full:
            self.dropped_events += 1
            self._log.warning("OpenTelemetrySink queue full; dropping type=%s", evt.get("type"))

    def flush(self, timeout: float = 1.0) -> bool:
        """Wait bounded time for queued events; false means remaining work dropped/later."""
        deadline = time.monotonic() + max(timeout, 0.0)
        while self._queue.unfinished_tasks:
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.005)
        return True

    def close(self, timeout: float = 1.0) -> None:
        """Bounded shutdown. It never raises into the runtime caller."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self.flush(timeout)
            self._stop.set()
        self._thread.join(timeout=max(timeout, 0.0))
        if not self._thread.is_alive():
            # A dropped terminal event can leave spans open in the local maps.
            # Close them on shutdown so a long-lived process does not retain
            # span objects indefinitely.
            for step in tuple(self._step_spans.values()):
                try:
                    step.end()
                except Exception:
                    self.export_errors += 1
            for span in tuple(self._run_spans.values()):
                try:
                    span.end()
                except Exception:
                    self.export_errors += 1
            self._step_spans.clear()
            self._run_spans.clear()

    @property
    def is_alive(self) -> bool:
        """Expose worker liveness for diagnostics and shutdown tests."""
        return self._thread.is_alive()

    def _run(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                evt = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                self._project(evt)
            except Exception:
                self.export_errors += 1
                self._log.exception("OpenTelemetry projection failed; event discarded")
            finally:
                self._queue.task_done()

    @staticmethod
    def _step_key(evt: Mapping[str, Any]) -> tuple[str, str, int, str, int]:
        return (
            str(evt.get("run_id", "")),
            str(evt.get("token_id", "")),
            int(evt.get("step_seq", 0)),
            str(evt.get("node_id", "")),
            int(evt.get("attempt", 1)),
        )

    @staticmethod
    def _attributes(evt: Mapping[str, Any]) -> dict[str, Any]:
        attrs: dict[str, Any] = {}
        for field in (
            "event_id",
            "run_id",
            "token_id",
            "node_id",
            "conversation_id",
            "turn_node_id",
            "trace_id",
            "span_id",
            "parent_span_id",
        ):
            value = evt.get(field)
            if value is not None:
                attrs[f"kogwistar.{field}"] = str(value)
        for field in ("step_seq", "attempt", "ts_ms"):
            value = evt.get(field)
            if value is not None:
                attrs[f"kogwistar.{field}"] = int(value)
        payload = evt.get("payload_json")
        if payload:
            attrs["kogwistar.payload_json"] = str(payload)
            try:
                payload_object = json.loads(str(payload))
            except (TypeError, ValueError):
                payload_object = None
            if isinstance(payload_object, Mapping):
                workflow_id = payload_object.get("workflow_id")
                if workflow_id is not None:
                    attrs["kogwistar.workflow_id"] = str(workflow_id)
        return attrs

    def _start_run(self, evt: Mapping[str, Any]) -> _Span:
        run_id = str(evt["run_id"])
        span = self._run_spans.get(run_id)
        if span is None:
            parent = self._known_span(str(evt.get("parent_span_id", "")))
            context = self._context_factory(parent) if parent and self._context_factory else None
            span = self._tracer.start_span(
                "kogwistar.workflow.run",
                context=context,
                attributes=self._attributes(evt),
            )
            self._run_spans[run_id] = span
        return span

    def _known_span(self, span_id: str) -> _Span | None:
        """Find an in-process parent span from the Kogwistar correlation ID."""
        if not span_id:
            return None
        for span in (*self._run_spans.values(), *self._step_spans.values()):
            try:
                if self._attributes_for_span(span).get("kogwistar.span_id") == span_id:
                    return span
            except Exception:
                continue
        return None

    @staticmethod
    def _attributes_for_span(span: _Span) -> Mapping[str, Any]:
        # The protocol intentionally stays tiny; test and SDK spans expose
        # attributes differently, so only use the optional public attribute.
        return getattr(span, "attributes", {}) or {}

    def _project(self, evt: dict[str, Any]) -> None:
        event_type = str(evt.get("type", ""))
        run_id = str(evt.get("run_id", ""))
        if event_type == "workflow_run_started":
            self._start_run(evt)
            return

        run_span = self._run_spans.get(run_id)
        if run_span is None:
            run_span = self._start_run(evt)

        if event_type == "step_attempt_started":
            key = self._step_key(evt)
            if key not in self._step_spans:
                context = self._context_factory(run_span) if self._context_factory else None
                self._step_spans[key] = self._tracer.start_span(
                    "kogwistar.workflow.step_attempt",
                    context=context,
                    attributes=self._attributes(evt),
                )
            return

        if event_type == "step_attempt_completed":
            step = self._step_spans.pop(self._step_key(evt), None)
            if step is not None:
                step.add_event(event_type, self._attributes(evt))
                step.end()
            else:
                run_span.add_event(event_type, self._attributes(evt))
            return

        if event_type in self._RUN_END_EVENTS:
            for key, step in list(self._step_spans.items()):
                if key[0] == run_id:
                    step.end()
                    del self._step_spans[key]
            run_span.add_event(event_type, self._attributes(evt))
            run_span.end()
            self._run_spans.pop(run_id, None)
            return

        run_span.add_event(event_type or "kogwistar.runtime.event", self._attributes(evt))
