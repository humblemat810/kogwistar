from __future__ import annotations

import sqlite3

from kogwistar.runtime.telemetry import (
    EventEmitter,
    FanoutEventSink,
    SQLiteEventSink,
    TraceContext,
)


def _ctx() -> TraceContext:
    return TraceContext(run_id="r", token_id="t", step_seq=1, node_id="n")


def test_event_emitter_isolates_sink_exception() -> None:
    class _BrokenSink:
        def emit(self, evt):
            raise RuntimeError("exporter unavailable")

        def close(self):
            return None

    event_id = EventEmitter(sink=_BrokenSink()).emit(type="step_attempt_started", ctx=_ctx())
    assert event_id.startswith("evt|")


def test_fanout_keeps_healthy_sink_when_peer_fails() -> None:
    seen: list[dict] = []

    class _BrokenSink:
        def emit(self, evt):
            raise RuntimeError("broken")

        def close(self):
            return None

    class _HealthySink:
        def emit(self, evt):
            seen.append(evt)

        def close(self):
            return None

    emitter = EventEmitter(sink=FanoutEventSink([_BrokenSink(), _HealthySink()]))
    emitter.emit(type="workflow_run_started", ctx=_ctx())
    assert [event["type"] for event in seen] == ["workflow_run_started"]


def test_sqlite_sink_flush_commits_prior_events_and_close_stops_writer(tmp_path) -> None:
    path = tmp_path / "telemetry.sqlite"
    sink = SQLiteEventSink(path, batch_size=100, flush_interval_ms=60_000)
    EventEmitter(sink=sink).emit(type="workflow_run_started", ctx=_ctx())

    assert sink.flush(1.0)
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT type FROM wf_trace_events").fetchall() == [
            ("workflow_run_started",)
        ]

    sink.close(1.0)
    assert not sink._thr.is_alive()


def test_fanout_flush_isolates_peer_failure() -> None:
    class _BrokenSink:
        def flush(self, timeout=1.0):
            raise RuntimeError("flush unavailable")

        def emit(self, _evt):
            return None

        def close(self):
            return None

    class _HealthySink:
        flushed = 0

        def flush(self, timeout=1.0):
            self.flushed += 1
            return True

        def emit(self, _evt):
            return None

        def close(self):
            return None

    healthy = _HealthySink()
    assert not FanoutEventSink([_BrokenSink(), healthy]).flush(1.0)
    assert healthy.flushed == 1


def test_fanout_close_forwards_remaining_timeout() -> None:
    class _Sink:
        timeout = None

        def flush(self, timeout=1.0):
            return True

        def emit(self, _evt):
            return None

        def close(self, timeout=1.0):
            self.timeout = timeout

    sink = _Sink()
    FanoutEventSink([sink]).close(0.25)
    assert sink.timeout is not None
    assert 0.0 <= sink.timeout <= 0.25
