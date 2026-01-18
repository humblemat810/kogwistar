from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

from .change_event import ChangeEvent


@dataclass(frozen=True, slots=True)
class ReplayResult:
    ok: bool
    # if ok == False, caller should request snapshot
    reason: Optional[str] = None


class ChangeBus:
    """
    In-process change feed with:
      - monotonic seq allocation
      - ring-buffer for replay
      - per-subscriber bounded queues (backpressure protection)
    """

    def __init__(
        self,
        *,
        buffer_max_events: int = 50_000,
        subscriber_queue_max: int = 2_000,
        drop_slow_subscribers: bool = True,
    ) -> None:
        self._seq = 0
        self._seq_lock = asyncio.Lock()

        self._buffer: Deque[ChangeEvent] = deque(maxlen=buffer_max_events)
        self._subscribers: set[asyncio.Queue[ChangeEvent]] = set()

        self._subscriber_queue_max = subscriber_queue_max
        self._drop_slow = drop_slow_subscribers

    @property
    def last_seq(self) -> int:
        return self._seq

    async def next_seq(self) -> int:
        async with self._seq_lock:
            self._seq += 1
            return self._seq

    def _buffer_oldest_seq(self) -> int:
        if not self._buffer:
            return self._seq
        return self._buffer[0].seq

    def _buffer_newest_seq(self) -> int:
        if not self._buffer:
            return self._seq
        return self._buffer[-1].seq

    def emit(self, event: ChangeEvent) -> None:
        # Must already have seq assigned (engine calls bus.next_seq())
        self._buffer.append(event)

        # Fan-out (never block engine)
        dead: list[asyncio.Queue[ChangeEvent]] = []
        for q in self._subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                if self._drop_slow:
                    dead.append(q)
                # else: drop this event for this subscriber (do nothing)
        for q in dead:
            self._subscribers.discard(q)

    def subscribe(self) -> asyncio.Queue[ChangeEvent]:
        q: asyncio.Queue[ChangeEvent] = asyncio.Queue(maxsize=self._subscriber_queue_max)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[ChangeEvent]) -> None:
        self._subscribers.discard(q)

    def replay_into(self, q: asyncio.Queue[ChangeEvent], *, since_seq: int) -> ReplayResult:
        """
        Best-effort replay from ring buffer into q. Non-async; no disk IO.
        If since_seq is too old (already evicted), returns ok=False.
        """
        if not self._buffer:
            return ReplayResult(ok=True)

        oldest = self._buffer_oldest_seq()
        newest = self._buffer_newest_seq()

        if since_seq < oldest - 1:
            return ReplayResult(ok=False, reason=f"since_seq too old for buffer (oldest={oldest}, newest={newest})")

        # Put all events with seq > since_seq
        for ev in self._buffer:
            if ev.seq > since_seq:
                try:
                    q.put_nowait(ev)
                except asyncio.QueueFull:
                    # Subscriber too slow even during replay
                    if self._drop_slow:
                        self.unsubscribe(q)
                        return ReplayResult(ok=False, reason="subscriber too slow during replay; disconnected")
                    break

        return ReplayResult(ok=True)
