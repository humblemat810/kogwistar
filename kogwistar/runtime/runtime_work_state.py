"""Thread-safe live-work accounting for the synchronous workflow scheduler."""

from __future__ import annotations

import threading


class RuntimeWorkState:
    """Track scheduled and in-flight work under one scheduler condition."""

    def __init__(self) -> None:
        self.condition = threading.Condition()
        self._count = 0

    def add(self, count: int = 1) -> None:
        if count <= 0:
            return
        with self.condition:
            self._count += count
            self.condition.notify_all()

    def done(self, count: int = 1) -> None:
        if count <= 0:
            return
        with self.condition:
            self._count = max(0, self._count - count)
            self.condition.notify_all()

    def snapshot(self) -> int:
        with self.condition:
            return self._count
