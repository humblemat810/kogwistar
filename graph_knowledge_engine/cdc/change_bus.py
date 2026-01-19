# knowledge_graph_engine/changes/change_bus.py
from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, Optional, List
import threading
import queue
import time
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

    # def next_seq(self) -> int:
    #     with self._seq_lock:
    #         self._seq += 1
    #         return self._seq

    def add_sink(self, sink: ChangeSink) -> None:
        self._sinks.append(sink)

    def emit(self, event: ChangeEvent) -> None:
        # existing internal behavior stays
        # e.g. in-memory tracking, counters, etc.

        for sink in self._sinks:
            sink.publish(event)

import requests
class FastAPIChangeSink:
    def __init__(self, endpoint: str, *, max_queue: int = 5000):
        self.endpoint = endpoint.rstrip("/")
        self.q: queue.Queue[dict] = queue.Queue(maxsize=max_queue)
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def publish(self, event) -> None:
        try:
            self.q.put_nowait(event.to_jsonable())
        except queue.Full:
            pass  # debug channel, drop is OK

    def _run(self):
        session = requests.Session()
        url = f"{self.endpoint}/ingest"
        while True:
            ev = self.q.get()
            try:
                session.post(url, json=ev, timeout=0.5)
            except requests.exceptions.ConnectTimeout:
                # allow bridge not set up usage
                pass
            except Exception as _e:
                raise