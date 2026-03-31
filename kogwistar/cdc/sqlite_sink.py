from __future__ import annotations

import threading

from kogwistar.runtime.telemetry import SQLiteEventSink

# ------------------------------------------------------------------
# Shared trace sink cache (process-local)
# ------------------------------------------------------------------
# Multiple WorkflowRuntime instances (including nested runs) may point at the same
# persist_directory. Creating multiple SQLiteEventSink writers for the same DB path
# is a common source of contention. For "quick fix" safety, we share a single sink
# instance per db_path inside this process.
_SINK_CACHE: dict[str, SQLiteEventSink] = {}
_SINK_CACHE_LOCK = threading.Lock()


def _get_shared_sqlite_sink(
    db_path: str, *, drop_when_full: bool = True
) -> SQLiteEventSink:
    with _SINK_CACHE_LOCK:
        sink = _SINK_CACHE.get(db_path)
        if sink is None:
            sink = SQLiteEventSink(db_path=db_path, drop_when_full=drop_when_full)
            _SINK_CACHE[db_path] = sink
        return sink
