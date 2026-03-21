# knowledge_graph_engine/changes/change_bus.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import threading
import queue
from kogwistar.utils import log as logmod
from kogwistar.utils.log import bind_log_context
from typing import Protocol


class ChangeSink(Protocol):
    def publish(self, event: ChangeEvent) -> None: ...


@dataclass(frozen=True, slots=True)
class ChangeEvent:
    seq: int
    op: str
    ts_unix_ms: int
    entity: Optional[dict] = None
    payload: object = None
    run_id: Optional[str] = None
    step_id: Optional[str] = None

    def to_jsonable(self) -> dict:
        return {
            "seq": self.seq,
            "op": self.op,
            "ts_unix_ms": self.ts_unix_ms,
            "entity": self.entity,
            "payload": self.payload,
            "run_id": self.run_id,
            "step_id": self.step_id,
        }


class ChangeBus:
    """
    Sync in-process change bus:
      - monotonic seq
      - ring buffer for replay
      - per-subscriber bounded queues
      - never blocks engine on slow subscribers
    """

    def __init__(self):
        self._sinks: list[ChangeSink] = []
        self._seq_lock = threading.Lock()
        self._seq = 0

    def next_seq(self) -> int:
        with self._seq_lock:
            self._seq += 1
            return self._seq

    def add_sink(self, sink: ChangeSink) -> None:
        self._sinks.append(sink)

    def emit(self, event: ChangeEvent) -> None:
        # existing internal behavior stays
        # e.g. in-memory tracking, counters, etc.

        for sink in self._sinks:
            sink.publish(event)


import requests


class FastAPIChangeSink:
    def __init__(
        self, endpoint: str, *, max_queue: int = 5000, name: str = "fastapi sink"
    ):
        self.endpoint = endpoint.rstrip("/")
        self.q: queue.Queue[dict] = queue.Queue(maxsize=max_queue)
        self._t = threading.Thread(target=self._run, daemon=True, name=name)
        self._t.start()

    def publish(self, event) -> None:
        try:
            payload = event.to_jsonable()
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

    def _run(self):
        session = requests.Session()
        url = f"{self.endpoint}/ingest"
        while True:
            ev = self.q.get()
            ctx = ev.pop("_log_ctx", None) or {}
            try:
                with bind_log_context(
                    engine_type=ctx.get("engine_type"),
                    engine_id=ctx.get("engine_id"),
                    conversation_id=ctx.get("conversation_id"),
                    workflow_run_id=ctx.get("workflow_run_id"),
                    step_id=ctx.get("step_id"),
                ):
                    session.post(url, json=ev, timeout=2.5)
            except (
                requests.exceptions.ConnectTimeout,
                requests.exceptions.ConnectionError,
            ):
                pass
            except Exception:
                raise
