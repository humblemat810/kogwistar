from __future__ import annotations

import heapq
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable


_PRIORITY_RANK = {
    "latency-sensitive": 0,
    "foreground": 1,
    "background": 2,
    "batch": 3,
}


@dataclass(order=True)
class _QueuedRun:
    sort_key: tuple[int, int, int]
    run_id: str = field(compare=False)
    priority_class: str = field(compare=False)
    start_fn: Callable[[], None] = field(compare=False)
    retry_count: int = field(default=0, compare=False)
    max_retries: int = field(default=3, compare=False)
    backoff_s: float = field(default=0.0, compare=False)


class RunScheduler:
    def __init__(
        self,
        *,
        max_active: int = 2,
        max_queue: int = 128,
        max_active_by_class: dict[str, int] | None = None,
    ) -> None:
        self.max_active = max(1, int(max_active or 1))
        self.max_queue = max(1, int(max_queue or 1))
        self.max_active_by_class = {
            str(k): max(0, int(v))
            for k, v in (max_active_by_class or {}).items()
        }
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._queue: list[_QueuedRun] = []
        self._active = 0
        self._active_by_class: dict[str, int] = {k: 0 for k in _PRIORITY_RANK}
        self._queued_by_class: dict[str, int] = {k: 0 for k in _PRIORITY_RANK}
        self._seq = 0
        self._closed = False
        self._dispatch_not_before = 0.0
        self._dispatch_window_s = 0.02
        self._dead_letter: list[dict[str, Any]] = []
        self._paused: set[str] = set()
        self._pause_requested: set[str] = set()
        self._thr = threading.Thread(
            target=self._loop, daemon=True, name="kogwistar-run-scheduler"
        )
        self._thr.start()

    def submit(
        self,
        *,
        run_id: str,
        priority_class: str,
        start_fn: Callable[[], None],
        retry_count: int = 0,
        max_retries: int = 3,
        external_hook: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
    ) -> dict[str, Any]:
        cls = str(priority_class or "foreground")
        pr = _PRIORITY_RANK.get(cls, 1)
        if callable(external_hook):
            maybe = external_hook(
                {
                    "run_id": run_id,
                    "priority_class": cls,
                    "retry_count": max(0, int(retry_count or 0)),
                    "max_retries": max(0, int(max_retries or 0)),
                }
            )
            if isinstance(maybe, dict):
                run_id = str(maybe.get("run_id") or run_id)
                cls = str(maybe.get("priority_class") or cls)
                retry_count = int(maybe.get("retry_count", retry_count) or retry_count or 0)
                max_retries = int(maybe.get("max_retries", max_retries) or max_retries or 0)
        with self._cv:
            if run_id in self._paused or run_id in self._pause_requested:
                return {
                    "run_id": run_id,
                    "priority_class": cls,
                    "queued_at_ms": int(time.time() * 1000),
                    "admission": "deferred",
                    "reason": "paused",
                }
            class_cap = self.max_active_by_class.get(cls)
            class_inflight = self._active_by_class.get(cls, 0) + self._queued_by_class.get(
                cls, 0
            )
            if class_cap is not None and class_inflight >= class_cap:
                return {
                    "run_id": run_id,
                    "priority_class": cls,
                    "queued_at_ms": int(time.time() * 1000),
                    "admission": "deferred",
                    "reason": "class_concurrency_limit",
                }
            if len(self._queue) >= self.max_queue:
                if pr >= _PRIORITY_RANK["background"]:
                    return {
                        "run_id": run_id,
                        "priority_class": cls,
                        "queued_at_ms": int(time.time() * 1000),
                        "admission": "rejected",
                        "reason": "queue_full",
                    }
                return {
                    "run_id": run_id,
                    "priority_class": cls,
                    "queued_at_ms": int(time.time() * 1000),
                    "admission": "deferred",
                    "reason": "queue_full",
                }
            self._seq += 1
            heapq.heappush(
                self._queue,
                _QueuedRun(
                    (pr, self._seq, 0),
                    run_id=run_id,
                    priority_class=cls,
                    start_fn=start_fn,
                    retry_count=max(0, int(retry_count or 0)),
                    max_retries=max(0, int(max_retries or 0)),
                    backoff_s=0.0,
                ),
            )
            self._queued_by_class[cls] = self._queued_by_class.get(cls, 0) + 1
            if self._active < self.max_active and self._dispatch_not_before <= 0:
                self._dispatch_not_before = time.time() + self._dispatch_window_s
            self._cv.notify_all()
            return {
                "run_id": run_id,
                "priority_class": cls,
                "queued_at_ms": int(time.time() * 1000),
                "admission": "accepted",
                "retry_policy": {
                    "retry_count": max(0, int(retry_count or 0)),
                    "max_retries": max(0, int(max_retries or 0)),
                },
                "external_hook": bool(external_hook),
            }

    def request_pause(self, run_id: str) -> dict[str, Any]:
        with self._cv:
            self._pause_requested.add(run_id)
            self._cv.notify_all()
            return {"run_id": run_id, "pause_state": "requested"}

    def mark_paused(self, run_id: str) -> dict[str, Any]:
        with self._cv:
            self._paused.add(run_id)
            self._pause_requested.discard(run_id)
            self._cv.notify_all()
            return {"run_id": run_id, "pause_state": "paused"}

    def resume(self, run_id: str) -> dict[str, Any]:
        with self._cv:
            self._paused.discard(run_id)
            self._pause_requested.discard(run_id)
            self._cv.notify_all()
            return {"run_id": run_id, "pause_state": "resumed"}

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()

    def _loop(self) -> None:
        while True:
            with self._cv:
                while not self._closed and (
                    self._active >= self.max_active or not self._queue
                ):
                    self._cv.wait(timeout=0.1)
                if self._closed:
                    return
                now = time.time()
                if self._dispatch_not_before > now:
                    self._cv.wait(timeout=max(0.0, self._dispatch_not_before - now))
                    continue
                item_idx = self._pick_runnable_index()
                if item_idx is None:
                    self._cv.wait(timeout=self._dispatch_window_s)
                    continue
                item = self._queue.pop(item_idx)
                heapq.heapify(self._queue)
                self._active += 1
                self._queued_by_class[item.priority_class] = max(
                    0, self._queued_by_class.get(item.priority_class, 0) - 1
                )
                self._active_by_class[item.priority_class] = (
                    self._active_by_class.get(item.priority_class, 0) + 1
                )
                self._dispatch_not_before = 0.0

            def _run_item(item: _QueuedRun) -> None:
                try:
                    item.start_fn()
                except Exception as exc:  # noqa: BLE001
                    with self._cv:
                        if item.retry_count < item.max_retries:
                            self._seq += 1
                            delay = min(5.0, 0.1 * (2**item.retry_count))
                            heapq.heappush(
                                self._queue,
                                _QueuedRun(
                                    (_PRIORITY_RANK.get(item.priority_class, 1), self._seq, 0),
                                    run_id=item.run_id,
                                    priority_class=item.priority_class,
                                    start_fn=item.start_fn,
                                    retry_count=item.retry_count + 1,
                                    max_retries=item.max_retries,
                                    backoff_s=delay,
                                ),
                            )
                            self._queued_by_class[item.priority_class] = (
                                self._queued_by_class.get(item.priority_class, 0) + 1
                            )
                        else:
                            self._dead_letter.append(
                                {
                                    "run_id": item.run_id,
                                    "priority_class": item.priority_class,
                                    "error": str(exc),
                                    "retry_count": item.retry_count,
                                    "max_retries": item.max_retries,
                                }
                            )
                        self._cv.notify_all()
                finally:
                    with self._cv:
                        self._active -= 1
                        self._active_by_class[item.priority_class] = max(
                            0, self._active_by_class.get(item.priority_class, 0) - 1
                        )
                        if self._queue and self._dispatch_not_before <= 0:
                            self._dispatch_not_before = time.time() + self._dispatch_window_s
                        self._cv.notify_all()

            threading.Thread(
                target=_run_item,
                args=(item,),
                daemon=True,
                name=f"kogwistar-run-{item.run_id}",
            ).start()

    def dead_letter(self) -> list[dict[str, Any]]:
        with self._cv:
            return list(self._dead_letter)

    def snapshot(self) -> dict[str, Any]:
        with self._cv:
            return {
                "max_active": self.max_active,
                "max_queue": self.max_queue,
                "active": self._active,
                "queued": len(self._queue),
                "active_by_class": dict(self._active_by_class),
                "queued_by_class": dict(self._queued_by_class),
                "dead_letter_count": len(self._dead_letter),
                "paused_count": len(self._paused),
                "pause_requested_count": len(self._pause_requested),
            }

    def _pick_runnable_index(self) -> int | None:
        for idx, item in enumerate(self._queue):
            if item.run_id in self._paused or item.run_id in self._pause_requested:
                continue
            class_cap = self.max_active_by_class.get(item.priority_class)
            if class_cap is not None and self._active_by_class.get(item.priority_class, 0) >= class_cap:
                continue
            return idx
        return None
