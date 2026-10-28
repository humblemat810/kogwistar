from __future__ import annotations

import asyncio
import inspect
import threading
from collections.abc import Callable, Iterable


class SyncLeaseHeartbeat:
    """Renew a claimed batch while synchronous provider/store work runs."""

    def __init__(
        self,
        metadata: object,
        claims: Iterable[tuple[str, str | None]],
        lease_seconds: int,
    ) -> None:
        renew = getattr(metadata, "renew_index_job_lease", None)
        self._renew: Callable[..., object] | None = renew if callable(renew) else None
        self._claims = tuple(
            (job_id, token) for job_id, token in claims if token is not None
        )
        self._interval = max(0.1, max(1, int(lease_seconds)) / 3)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._lost: set[str] = set()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> SyncLeaseHeartbeat:
        if callable(self._renew) and self._claims:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self._interval * 2))

    def owns(self, job_id: str) -> bool:
        with self._lock:
            return job_id not in self._lost

    def _run(self) -> None:
        renew = self._renew
        if renew is None:
            return
        while not self._stop.wait(self._interval):
            for job_id, claim_token in self._claims:
                if self._stop.is_set():
                    return
                try:
                    renewed = renew(
                        job_id,
                        claim_token=claim_token,
                        lease_seconds=max(1, round(self._interval * 3)),
                    )
                    if renewed is not True:
                        self._mark_lost(job_id)
                except Exception:
                    self._mark_lost(job_id)

    def _mark_lost(self, job_id: str) -> None:
        with self._lock:
            self._lost.add(job_id)


class AsyncLeaseHeartbeat:
    """Renew a claimed batch while asynchronous provider/store work runs."""

    def __init__(
        self,
        metadata: object,
        claims: Iterable[tuple[str, str | None]],
        lease_seconds: int,
    ) -> None:
        self._metadata = metadata
        self._claims = tuple(
            (job_id, token) for job_id, token in claims if token is not None
        )
        self._lease_seconds = max(1, int(lease_seconds))
        self._interval = max(0.1, self._lease_seconds / 3)
        self._lost: set[str] = set()
        self._task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> AsyncLeaseHeartbeat:
        if callable(getattr(self._metadata, "renew_index_job_lease", None)) and self._claims:
            self._task = asyncio.create_task(self._run())
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    def owns(self, job_id: str) -> bool:
        return job_id not in self._lost

    async def _run(self) -> None:
        renew = getattr(self._metadata, "renew_index_job_lease")
        while True:
            await asyncio.sleep(self._interval)
            for job_id, claim_token in self._claims:
                try:
                    if inspect.iscoroutinefunction(renew):
                        result = await renew(
                            job_id,
                            claim_token=claim_token,
                            lease_seconds=self._lease_seconds,
                        )
                    else:
                        result = await asyncio.to_thread(
                            renew,
                            job_id,
                            claim_token=claim_token,
                            lease_seconds=self._lease_seconds,
                        )
                        if inspect.isawaitable(result):
                            result = await result
                    if result is not True:
                        self._lost.add(job_id)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    self._lost.add(job_id)
