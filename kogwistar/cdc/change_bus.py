# knowledge_graph_engine/changes/change_bus.py
from __future__ import annotations

import queue
import threading
from collections.abc import Mapping
from typing import Protocol, cast

import requests

from kogwistar.json_types import JsonValue
from kogwistar.utils import log as logmod
from kogwistar.utils.log import bind_log_context

from .change_event import ChangeEvent


class ChangeSink(Protocol):
    def publish(self, event: ChangeEvent) -> None: ...


def _context_text(value: JsonValue) -> str | None:
    return value if isinstance(value, str) else None


class ChangeBus:
    """
    Sync in-process change bus:
      - monotonic seq
      - ring buffer for replay
      - per-subscriber bounded queues
      - never blocks engine on slow subscribers
    """

    def __init__(self) -> None:
        self._sinks: list[ChangeSink] = []
        self._seq_lock = threading.Lock()
        self._seq = 0
        self._closed = False

    def next_seq(self) -> int:
        with self._seq_lock:
            self._seq += 1
            return self._seq

    def add_sink(self, sink: ChangeSink) -> None:
        if self._closed:
            close = getattr(sink, "close", None)
            if callable(close):
                close()
            raise RuntimeError("change bus already closed")
        self._sinks.append(sink)

    def emit(self, event: ChangeEvent) -> None:
        if self._closed:
            return
        # existing internal behavior stays
        # e.g. in-memory tracking, counters, etc.

        for sink in self._sinks:
            sink.publish(event)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for sink in self._sinks:
            close = getattr(sink, "close", None)
            if callable(close):
                close()
        self._sinks.clear()


class FastAPIChangeSink:
    _STOP = object()

    def __init__(
        self, endpoint: str, *, max_queue: int = 5000, name: str = "fastapi sink"
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        # The queue also carries the private stop sentinel used by ``close``.
        self.q: queue.Queue[object] = queue.Queue(maxsize=max_queue)
        self._closed = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True, name=name)
        self._t.start()

    def publish(self, event: ChangeEvent) -> None:
        if self._closed.is_set():
            return
        try:
            payload: dict[str, JsonValue] = dict(event.to_jsonable())
            # Capture current logging context (from the caller thread)
            payload["_log_ctx"] = {
                "engine_type": logmod._ctx_engine_type.get(),
                "engine_id": logmod._ctx_engine_id.get(),
                "conversation_id": logmod._ctx_conversation_id.get(),
                # prefer event fields if provided
                "workflow_run_id": event.run_id,
                "step_id": event.step_id,
            }
            self.q.put_nowait(payload)
        except queue.Full:
            pass

    def _run(self) -> None:
        url = f"{self.endpoint}/ingest"
        with requests.Session() as session:
            while True:
                item = self.q.get()
                if item is self._STOP:
                    return
                if not isinstance(item, dict):
                    continue
                ev = cast(dict[str, JsonValue], item)
                raw_context = ev.pop("_log_ctx", None)
                ctx: Mapping[str, JsonValue] = (
                    raw_context if isinstance(raw_context, Mapping) else {}
                )
                try:
                    with bind_log_context(
                        engine_type=_context_text(ctx.get("engine_type")),
                        engine_id=_context_text(ctx.get("engine_id")),
                        conversation_id=_context_text(ctx.get("conversation_id")),
                        workflow_run_id=_context_text(ctx.get("workflow_run_id")),
                        step_id=_context_text(ctx.get("step_id")),
                    ):
                        session.post(url, json=ev, timeout=2.5)
                except (
                    requests.exceptions.ConnectTimeout,
                    requests.exceptions.ConnectionError,
                ):
                    pass
                except Exception:
                    raise

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        try:
            self.q.put_nowait(self._STOP)
        except queue.Full:
            try:
                self.q.get_nowait()
            except queue.Empty:
                pass
            try:
                self.q.put_nowait(self._STOP)
            except queue.Full:
                return
        self._t.join(timeout=1.0)
