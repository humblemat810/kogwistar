from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import Any

from kogwistar.runtime.budget import RateBudgetWindow
from kogwistar.server.run_scheduler import RunScheduler


@dataclass
class FakeClock:
    now_ms: int = 0

    def advance(self, amount_ms: int) -> int:
        self.now_ms += int(amount_ms or 0)
        return self.now_ms


class FakeTokenGenerator:
    def __init__(self, *, limit: int, window_ms: int, clock: FakeClock) -> None:
        self.clock = clock
        self.window = RateBudgetWindow(
            limit=int(limit or 0),
            used=0,
            window_ms=int(window_ms or 0),
            window_started_ms=int(clock.now_ms),
        )
        self._lock = threading.Lock()

    def remaining(self) -> int:
        with self._lock:
            return self.window.remaining(now_ms=self.clock.now_ms)

    def can_consume(self, amount: int) -> bool:
        return self.remaining() >= int(amount or 0)

    def consume(self, amount: int) -> bool:
        with self._lock:
            try:
                self.window.debit(int(amount or 0), now_ms=self.clock.now_ms)
                return True
            except Exception:
                return False

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            self.window.refresh(now_ms=self.clock.now_ms)
            return {
                "limit": int(self.window.limit),
                "used": int(self.window.used),
                "remaining": max(0, int(self.window.limit) - int(self.window.used)),
                "window_ms": int(self.window.window_ms),
                "window_started_ms": int(self.window.window_started_ms),
                "now_ms": int(self.clock.now_ms),
            }


def run_budget_rate_switch_demo() -> dict[str, Any]:
    clock = FakeClock(now_ms=1000)
    tokens = FakeTokenGenerator(limit=3, window_ms=10_000, clock=clock)
    sched = RunScheduler(max_active=1, max_queue=8)
    timeline: list[dict[str, Any]] = []
    order: list[str] = []
    heavy_phase = {"step": 0}
    heavy_paused = threading.Event()
    tiny_done = threading.Event()
    free_done = threading.Event()
    heavy_done = threading.Event()
    lock = threading.Lock()

    def _record(event: str, **extra: Any) -> None:
        with lock:
            timeline.append(
                {
                    "event": event,
                    "ts_ms": int(clock.now_ms),
                    "tokens_remaining": tokens.remaining(),
                    **extra,
                }
            )

    def _heavy_run() -> None:
        if heavy_phase["step"] == 0:
            _record("heavy.started")
            if not tokens.consume(2):
                _record("heavy.start_blocked")
                sched.mark_paused("heavy-run")
                heavy_paused.set()
                return
            heavy_phase["step"] = 1
            _record("heavy.step1")
            if not tokens.can_consume(2):
                _record("heavy.rate_blocked")
                sched.mark_paused("heavy-run")
                heavy_paused.set()
                return
        if not tokens.consume(2):
            _record("heavy.resume_blocked")
            sched.mark_paused("heavy-run")
            heavy_paused.set()
            return
        heavy_phase["step"] = 2
        order.append("heavy")
        _record("heavy.finished")
        heavy_done.set()

    def _tiny_run() -> None:
        _record("tiny.started")
        if not tokens.consume(1):
            _record("tiny.blocked")
            return
        order.append("tiny")
        _record("tiny.finished")
        tiny_done.set()

    def _free_run() -> None:
        _record("free.started")
        order.append("free")
        _record("free.finished")
        free_done.set()

    heavy_submit = sched.submit(
        run_id="heavy-run",
        priority_class="background",
        start_fn=_heavy_run,
    )
    if heavy_submit["admission"] != "accepted":
        raise AssertionError(f"unexpected heavy admission: {heavy_submit}")
    if not heavy_paused.wait(timeout=5.0):
        raise AssertionError("heavy run did not pause on rate exhaustion")

    tiny_submit = sched.submit(
        run_id="tiny-run",
        priority_class="foreground",
        start_fn=_tiny_run,
    )
    if tiny_submit["admission"] != "accepted":
        raise AssertionError(f"unexpected tiny admission: {tiny_submit}")
    if not tiny_done.wait(timeout=5.0):
        raise AssertionError("tiny run did not finish with leftover tokens")

    free_submit = sched.submit(
        run_id="free-run",
        priority_class="foreground",
        start_fn=_free_run,
    )
    if free_submit["admission"] != "accepted":
        raise AssertionError(f"unexpected free admission: {free_submit}")
    if not free_done.wait(timeout=5.0):
        raise AssertionError("free run did not finish without tokens")

    clock.advance(10_001)
    _record("token_window.refreshed", **tokens.snapshot())
    sched.resume("heavy-run")
    heavy_resume = sched.submit(
        run_id="heavy-run",
        priority_class="background",
        start_fn=_heavy_run,
    )
    if heavy_resume["admission"] != "accepted":
        raise AssertionError(f"unexpected heavy resume admission: {heavy_resume}")
    if not heavy_done.wait(timeout=5.0):
        raise AssertionError("heavy run did not finish after refresh")

    sched.close()
    return {
        "rate_window": tokens.snapshot(),
        "submissions": {
            "heavy": heavy_submit,
            "tiny": tiny_submit,
            "free": free_submit,
            "heavy_resume": heavy_resume,
        },
        "order": order,
        "timeline": timeline,
        "result": {
            "tiny_before_refresh": order.index("tiny") < order.index("heavy"),
            "free_before_refresh": order.index("free") < order.index("heavy"),
            "heavy_resumed_after_refresh": any(
                item["event"] == "token_window.refreshed" for item in timeline
            )
            and timeline[-1]["event"] == "heavy.finished",
        },
    }


def main() -> None:
    print(json.dumps(run_budget_rate_switch_demo(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
