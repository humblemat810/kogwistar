from __future__ import annotations

import threading
import time

from kogwistar.server.run_scheduler import RunScheduler


def test_run_scheduler_prefers_higher_priority_first() -> None:
    sched = RunScheduler(max_active=1)
    seen: list[str] = []
    done = threading.Event()

    def _mk(name: str, delay: float = 0.0):
        def _run():
            time.sleep(delay)
            seen.append(name)
            if len(seen) == 2:
                done.set()

        return _run

    sched.submit(run_id="batch-1", priority_class="batch", start_fn=_mk("batch"))
    sched.submit(
        run_id="foreground-1",
        priority_class="foreground",
        start_fn=_mk("foreground"),
    )
    assert done.wait(timeout=5.0)
    assert seen == ["foreground", "batch"]


def test_run_scheduler_accepts_latency_sensitive() -> None:
    sched = RunScheduler(max_active=2)
    seen: list[str] = []
    done = threading.Event()

    def _run():
        seen.append("latency-sensitive")
        done.set()

    sched.submit(
        run_id="ls-1",
        priority_class="latency-sensitive",
        start_fn=_run,
    )
    assert done.wait(timeout=5.0)
    assert seen == ["latency-sensitive"]


def test_run_scheduler_rejects_background_when_queue_full() -> None:
    sched = RunScheduler(max_active=1, max_queue=1)
    started = threading.Event()
    release = threading.Event()

    def _block():
        started.set()
        release.wait(timeout=5.0)

    first = sched.submit(
        run_id="foreground-1",
        priority_class="foreground",
        start_fn=_block,
    )
    assert first["admission"] == "accepted"
    assert started.wait(timeout=5.0)

    second = sched.submit(
        run_id="batch-1",
        priority_class="batch",
        start_fn=lambda: None,
    )
    assert second["admission"] == "accepted"

    third = sched.submit(
        run_id="batch-2",
        priority_class="batch",
        start_fn=lambda: None,
    )
    assert third["admission"] == "rejected"
    assert third["reason"] == "queue_full"
    release.set()


def test_run_scheduler_defers_foreground_when_queue_full() -> None:
    sched = RunScheduler(max_active=1, max_queue=1)
    started = threading.Event()
    release = threading.Event()

    def _block():
        started.set()
        release.wait(timeout=5.0)

    first = sched.submit(
        run_id="foreground-1",
        priority_class="foreground",
        start_fn=_block,
    )
    assert first["admission"] == "accepted"
    assert started.wait(timeout=5.0)

    second = sched.submit(
        run_id="foreground-2",
        priority_class="foreground",
        start_fn=lambda: None,
    )
    assert second["admission"] == "accepted"

    third = sched.submit(
        run_id="foreground-3",
        priority_class="foreground",
        start_fn=lambda: None,
    )
    assert third["admission"] == "deferred"
    assert third["reason"] == "queue_full"
    release.set()


def test_run_scheduler_honors_class_concurrency_limit() -> None:
    sched = RunScheduler(max_active=2, max_queue=4, max_active_by_class={"batch": 1})
    started = []
    release = threading.Event()

    def _block(name: str):
        def _run():
            started.append(name)
            release.wait(timeout=5.0)

        return _run

    a = sched.submit(run_id="batch-a", priority_class="batch", start_fn=_block("a"))
    b = sched.submit(run_id="batch-b", priority_class="batch", start_fn=_block("b"))
    assert a["admission"] == "accepted"
    assert b["admission"] == "deferred"
    assert b["reason"] == "class_concurrency_limit"
    release.set()


def test_run_scheduler_prioritizes_latency_sensitive_over_batch() -> None:
    sched = RunScheduler(max_active=1, max_queue=8)
    seen: list[str] = []
    done = threading.Event()

    def _run(name: str):
        def _fn():
            seen.append(name)
            if len(seen) == 2:
                done.set()

        return _fn

    sched.submit(run_id="batch-1", priority_class="batch", start_fn=_run("batch"))
    sched.submit(
        run_id="latency-1",
        priority_class="latency-sensitive",
        start_fn=_run("latency-sensitive"),
    )
    assert done.wait(timeout=5.0)
    assert seen == ["latency-sensitive", "batch"]


def test_run_scheduler_defaults_unknown_class_to_foreground() -> None:
    sched = RunScheduler(max_active=1, max_queue=4)
    seen: list[str] = []
    done = threading.Event()

    def _run():
        seen.append("unknown")
        done.set()

    out = sched.submit(run_id="u-1", priority_class="mystery", start_fn=_run)
    assert out["priority_class"] == "mystery"
    assert out["admission"] == "accepted"
    assert done.wait(timeout=5.0)
    assert seen == ["unknown"]


def test_run_scheduler_retries_then_dead_letters() -> None:
    sched = RunScheduler(max_active=1, max_queue=4)
    attempts = {"count": 0}
    done = threading.Event()

    def _boom():
        attempts["count"] += 1
        raise RuntimeError("boom")

    sched.submit(
        run_id="retry-me",
        priority_class="foreground",
        start_fn=_boom,
        max_retries=1,
    )
    deadline = time.time() + 5.0
    while time.time() < deadline and attempts["count"] < 2:
        time.sleep(0.05)
    assert attempts["count"] >= 2
    assert sched.dead_letter() and sched.dead_letter()[0]["run_id"] == "retry-me"
    done.set()


def test_run_scheduler_external_hook_can_rewrite_priority() -> None:
    sched = RunScheduler(max_active=1, max_queue=4)
    seen: list[str] = []
    done = threading.Event()

    def _run():
        seen.append("ran")
        done.set()

    out = sched.submit(
        run_id="hook-me",
        priority_class="batch",
        start_fn=_run,
        external_hook=lambda req: {**req, "priority_class": "latency-sensitive"},
    )
    assert out["priority_class"] == "latency-sensitive"
    assert done.wait(timeout=5.0)
    assert seen == ["ran"]


def test_run_scheduler_cooperative_pause_blocks_start_until_resumed() -> None:
    sched = RunScheduler(max_active=1, max_queue=4)
    seen: list[str] = []
    done = threading.Event()

    def _run():
        seen.append("ran")
        done.set()

    sched.request_pause("paused-run")
    out = sched.submit(
        run_id="paused-run",
        priority_class="foreground",
        start_fn=_run,
    )
    assert out["admission"] == "deferred"
    assert out["reason"] == "paused"
    assert not done.wait(timeout=0.2)
    sched.resume("paused-run")
    out2 = sched.submit(
        run_id="paused-run",
        priority_class="foreground",
        start_fn=_run,
    )
    assert out2["admission"] == "accepted"
    assert done.wait(timeout=5.0)
    assert seen == ["ran"]


def test_run_scheduler_preemptive_pause_is_request_only() -> None:
    sched = RunScheduler(max_active=1, max_queue=4)
    seen = []
    release = threading.Event()

    def _run():
        seen.append("started")
        release.wait(timeout=5.0)

    sched.submit(
        run_id="active-run",
        priority_class="foreground",
        start_fn=_run,
    )
    time.sleep(0.05)
    out = sched.request_pause("active-run")
    assert out["pause_state"] == "requested"
    assert seen == ["started"]
    release.set()
