from __future__ import annotations

import json
import threading
import time
from typing import Any

from kogwistar.server.run_scheduler import RunScheduler


def run_scheduler_priority_demo() -> dict[str, Any]:
    sched = RunScheduler(max_active=1, max_queue=8)
    timeline: list[dict[str, Any]] = []
    order: list[str] = []
    timeline_lock = threading.Lock()
    switch_lock = threading.Lock()
    allow_low_finish = False

    low_started = threading.Event()
    low_paused = threading.Event()
    high_finished = threading.Event()
    low_finished = threading.Event()

    def _record(event: str) -> None:
        with timeline_lock:
            timeline.append({"event": event, "ts_ms": int(time.time() * 1000)})

    def _read_switch() -> bool:
        with switch_lock:
            return allow_low_finish

    def _write_switch(value: bool) -> None:
        nonlocal allow_low_finish
        with switch_lock:
            allow_low_finish = value

    def _low_run() -> None:
        _record("low.started")
        low_started.set()
        if not _read_switch():
            _record("low.blocked")
            sched.mark_paused("low-run")
            low_paused.set()
            return
        _record("low.resumed")
        order.append("low")
        _record("low.finished")
        low_finished.set()

    def _high_run() -> None:
        _record("high.started")
        order.append("high")
        _record("high.finished")
        high_finished.set()

    low_submit = sched.submit(
        run_id="low-run",
        priority_class="background",
        start_fn=_low_run,
    )
    if low_submit["admission"] != "accepted":
        raise AssertionError(f"unexpected low admission: {low_submit}")
    if not low_started.wait(timeout=5.0):
        raise AssertionError("low run did not start")
    if not low_paused.wait(timeout=5.0):
        raise AssertionError("low run did not yield at cooperative block")

    high_submit = sched.submit(
        run_id="high-run",
        priority_class="latency-sensitive",
        start_fn=_high_run,
    )
    if high_submit["admission"] != "accepted":
        raise AssertionError(f"unexpected high admission: {high_submit}")
    if not high_finished.wait(timeout=5.0):
        raise AssertionError("high run did not finish while low run was paused")

    _write_switch(True)
    sched.resume("low-run")
    low_resume = sched.submit(
        run_id="low-run",
        priority_class="background",
        start_fn=_low_run,
    )
    if low_resume["admission"] != "accepted":
        raise AssertionError(f"unexpected low resume admission: {low_resume}")
    if not low_finished.wait(timeout=5.0):
        raise AssertionError("low run did not finish after resume")

    sched.close()
    return {
        "scheduler": {
            "max_active": 1,
            "cooperative_pause_required": True,
        },
        "submissions": {
            "low": low_submit,
            "high": high_submit,
            "low_resume": low_resume,
        },
        "order": order,
        "timeline": timeline,
        "result": {
            "high_finished_before_low": order == ["high", "low"],
            "low_blocked_before_high_started": any(
                item["event"] == "low.blocked" for item in timeline
            )
            and any(item["event"] == "high.started" for item in timeline),
        },
    }


def main() -> None:
    print(json.dumps(run_scheduler_priority_demo(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
