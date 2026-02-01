from __future__ import annotations

import atexit
import json
import logging
import queue
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


# -----------------------------
# Trace / event schema helpers
# -----------------------------

def _now_ms() -> int:
    return int(time.time() * 1000)


def _new_id(prefix: str = "evt") -> str:
    return f"{prefix}|{uuid.uuid4()}"


@dataclass(frozen=True)
class TraceContext:
    """
    Minimal correlation IDs that make trace<->logs mapping work.
    This is cheap to pass around (and you already have these in runtime).
    """
    run_id: str
    token_id: str
    step_seq: int
    node_id: str
    attempt: int = 1

    conversation_id: Optional[str] = None
    turn_node_id: Optional[str] = None

    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None

    def with_span_defaults(self) -> "TraceContext":
        """
        Provide conventional defaults:
          trace_id = run_id
          token span = f"tok:{run_id}:{token_id}"
          step span  = f"stp:{run_id}:{token_id}:{step_seq}:{attempt}:{node_id}"
        """
        trace_id = self.trace_id or self.run_id
        token_span = f"tok:{self.run_id}:{self.token_id}"
        step_span = f"stp:{self.run_id}:{self.token_id}:{self.step_seq}:{self.attempt}:{self.node_id}"
        return TraceContext(
            run_id=self.run_id,
            token_id=self.token_id,
            step_seq=self.step_seq,
            node_id=self.node_id,
            attempt=self.attempt,
            conversation_id=self.conversation_id,
            turn_node_id=self.turn_node_id,
            trace_id=trace_id,
            span_id=self.span_id or step_span,
            parent_span_id=self.parent_span_id or token_span,
        )

    def as_fields(self) -> Dict[str, Any]:
        ctx = self.with_span_defaults()
        return {
            "run_id": ctx.run_id,
            "token_id": ctx.token_id,
            "step_seq": ctx.step_seq,
            "node_id": ctx.node_id,
            "attempt": ctx.attempt,
            "conversation_id": ctx.conversation_id,
            "turn_node_id": ctx.turn_node_id,
            "trace_id": ctx.trace_id,
            "span_id": ctx.span_id,
            "parent_span_id": ctx.parent_span_id,
        }


# -----------------------------
# SQLite sink (async writer)
# -----------------------------

class SQLiteEventSink:
    """
    Durable append-only event store with an async writer thread.

    - emit() is non-blocking unless queue is full
    - writer batches inserts
    - schema is trace-friendly: trace_id/span_id/parent_span_id
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        table: str = "wf_trace_events",
        queue_max: int = 10_000,
        batch_size: int = 200,
        flush_interval_ms: int = 250,
        drop_when_full: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.db_path = str(db_path)
        self.table = table
        self.batch_size = batch_size
        self.flush_interval_ms = flush_interval_ms
        self.drop_when_full = drop_when_full
        self._log = logger or logging.getLogger(__name__)

        self._q: "queue.Queue[dict]" = queue.Queue(maxsize=queue_max)
        self._stop = threading.Event()
        self._thr = threading.Thread(target=self._run, name="wf-trace-sqlite-writer", daemon=True)

        self._init_db()
        self._thr.start()
        atexit.register(self.close)

    def _init_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    event_id TEXT PRIMARY KEY,
                    ts_ms INTEGER NOT NULL,
                    type TEXT NOT NULL,

                    trace_id TEXT NOT NULL,
                    span_id TEXT NOT NULL,
                    parent_span_id TEXT,

                    run_id TEXT NOT NULL,
                    token_id TEXT NOT NULL,
                    step_seq INTEGER NOT NULL,
                    node_id TEXT NOT NULL,
                    attempt INTEGER NOT NULL,

                    conversation_id TEXT,
                    turn_node_id TEXT,

                    payload_json TEXT NOT NULL
                );
                """
            )
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.table}_run_seq ON {self.table}(run_id, step_seq);")
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.table}_span ON {self.table}(span_id);")
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.table}_trace ON {self.table}(trace_id);")
            conn.commit()
        finally:
            conn.close()

    def emit(self, evt: Dict[str, Any]) -> None:
        """
        Enqueue a fully-formed event dict (already includes required fields).
        """
        if self._stop.is_set():
            return

        try:
            self._q.put_nowait(evt)
        except queue.Full:
            if self.drop_when_full:
                # best-effort: trace may drop in overload, but we still log it
                self._log.warning("SQLiteEventSink queue full; dropping trace event type=%s", evt.get("type"))
            else:
                # backpressure: block until space
                self._q.put(evt)

    def _run(self) -> None:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        cur = conn.cursor()

        batch: list[dict] = []
        last_flush = _now_ms()

        def flush() -> None:
            nonlocal batch, last_flush
            if not batch:
                last_flush = _now_ms()
                return

            rows = []
            for e in batch:
                rows.append(
                    (
                        e["event_id"],
                        e["ts_ms"],
                        e["type"],
                        e["trace_id"],
                        e["span_id"],
                        e.get("parent_span_id"),
                        e["run_id"],
                        e["token_id"],
                        int(e["step_seq"]),
                        e["node_id"],
                        int(e.get("attempt", 1)),
                        e.get("conversation_id"),
                        e.get("turn_node_id"),
                        e["payload_json"],
                    )
                )

            cur.executemany(
                f"""
                INSERT OR REPLACE INTO {self.table} (
                    event_id, ts_ms, type,
                    trace_id, span_id, parent_span_id,
                    run_id, token_id, step_seq, node_id, attempt,
                    conversation_id, turn_node_id,
                    payload_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                rows,
            )
            conn.commit()
            batch = []
            last_flush = _now_ms()

        try:
            while not self._stop.is_set():
                timeout = max(0.01, self.flush_interval_ms / 1000.0)
                try:
                    e = self._q.get(timeout=timeout)
                    batch.append(e)
                    if len(batch) >= self.batch_size:
                        flush()
                except queue.Empty:
                    # periodic flush
                    if batch and (_now_ms() - last_flush) >= self.flush_interval_ms:
                        flush()

            # drain on stop
            while True:
                try:
                    batch.append(self._q.get_nowait())
                except queue.Empty:
                    break
            flush()
        finally:
            conn.close()

    def close(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        try:
            self._thr.join(timeout=2.0)
        except Exception:
            pass


# -----------------------------
# Emitter: logging + sqlite sink
# -----------------------------

class EventEmitter:
    """
    Main entry point used by runtime/resolvers.
    Emits:
      - structured logs (logging.info)
      - durable SQLite events (async)
    """

    def __init__(
        self,
        *,
        sink: Optional[SQLiteEventSink] = None,
        logger: Optional[logging.Logger] = None,
        log_prefix: str = "WF_EVT",
    ) -> None:
        self.sink = sink
        self.log = logger or logging.getLogger("workflow.telemetry")
        self.log_prefix = log_prefix

    def emit(self, *, type: str, ctx: TraceContext, payload: Mapping[str, Any] | None = None) -> str:
        ctx2 = ctx.with_span_defaults()
        event_id = _new_id("evt")
        payload_obj = dict(payload or {})

        evt = {
            "event_id": event_id,
            "ts_ms": _now_ms(),
            "type": type,
            **ctx2.as_fields(),
            "payload_json": json.dumps(payload_obj, ensure_ascii=False, separators=(",", ":")),
        }

        # human/ops log mirror (best effort)
        # (Still structured JSON so it can be shipped to ELK/etc later.)
        self.log.info("%s %s", self.log_prefix, json.dumps({k: v for k, v in evt.items() if k != "payload_json"}))

        # durable sink
        if self.sink is not None:
            self.sink.emit(evt)

        return event_id

    # convenience helpers used by runtime
    def step_started(self, ctx: TraceContext, *, extra: Mapping[str, Any] | None = None) -> str:
        return self.emit(type="step_attempt_started", ctx=ctx, payload=extra)

    def step_completed(self, ctx: TraceContext, *, status: str, duration_ms: int, extra: Mapping[str, Any] | None = None) -> str:
        payload = {"status": status, "duration_ms": duration_ms, **(extra or {})}
        return self.emit(type="step_attempt_completed", ctx=ctx, payload=payload)

    def predicate_evaluated(self, ctx: TraceContext, *, predicate: str, value: bool, error: str | None = None) -> str:
        payload = {"predicate": predicate, "value": bool(value)}
        if error:
            payload["error"] = error
        return self.emit(type="predicate_evaluated", ctx=ctx, payload=payload)

    def edge_selected(self, ctx: TraceContext, *, edge_id: str, to_node_id: str, reason: str) -> str:
        payload = {"edge_id": edge_id, "to_node_id": to_node_id, "reason": reason}
        return self.emit(type="edge_selected", ctx=ctx, payload=payload)

    def join_event(self, ctx: TraceContext, *, join_node_id: str, kind: str, outstanding: int) -> str:
        payload = {"join_node_id": join_node_id, "kind": kind, "outstanding": int(outstanding)}
        return self.emit(type=f"join_{kind}", ctx=ctx, payload=payload)


# -----------------------------
# Bound logger for resolver code
# -----------------------------

class BoundLoggerAdapter(logging.LoggerAdapter):
    """
    Ensures all logs contain correlation fields.
    """
    def process(self, msg, kwargs):
        extra = dict(kwargs.get("extra") or {})
        extra.update(self.extra)  # correlation fields win
        kwargs["extra"] = extra
        return msg, kwargs


def bind_logger(base: logging.Logger, ctx: TraceContext) -> BoundLoggerAdapter:
    """
    Create a resolver-friendly logger adapter that automatically includes trace/run fields.
    """
    fields = ctx.with_span_defaults().as_fields()
    # reduce noise: a few are enough for correlation
    keep = {k: fields[k] for k in ["run_id", "trace_id", "span_id", "token_id", "step_seq", "node_id", "attempt"] if fields.get(k) is not None}
    if ctx.conversation_id:
        keep["conversation_id"] = ctx.conversation_id
    if ctx.turn_node_id:
        keep["turn_node_id"] = ctx.turn_node_id
    return BoundLoggerAdapter(base, extra=keep)