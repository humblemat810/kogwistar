"""Bounded observation/enrichment hooks; hooks cannot become runtime authority."""

from __future__ import annotations

import inspect
import threading
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal


HookFailureMode = Literal["fail_open", "fail_closed"]
HookEffect = Literal["observe", "annotate"]
HookCallback = Callable[[Mapping[str, Any]], Any]


@dataclass(frozen=True, slots=True)
class HookSpec:
    hook_id: str
    callback: HookCallback
    order: int = 100
    timeout_ms: int = 1_000
    failure_mode: HookFailureMode = "fail_open"
    effect: HookEffect = "observe"
    required_capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not str(self.hook_id).strip():
            raise ValueError("hook_id must be non-empty")
        if int(self.timeout_ms) < 1:
            raise ValueError("hook timeout must be positive")
        if self.effect not in {"observe", "annotate"}:
            raise ValueError("hooks may only observe or annotate")
        if self.failure_mode not in {"fail_open", "fail_closed"}:
            raise ValueError("unsupported hook failure mode")


@dataclass(frozen=True, slots=True)
class HookResult:
    hook_id: str
    status: Literal["applied", "failed", "skipped"]
    annotations: dict[str, Any]
    error: str | None = None


class HookRegistry:
    """Deterministic registry with capability checks and bounded disposal."""

    def __init__(self) -> None:
        self._hooks: dict[str, HookSpec] = {}

    def register(self, spec: HookSpec) -> HookSpec:
        if spec.hook_id in self._hooks:
            raise ValueError(f"hook already registered: {spec.hook_id}")
        self._hooks[spec.hook_id] = spec
        return spec

    def unregister(self, hook_id: str) -> None:
        spec = self._hooks.pop(str(hook_id), None)
        if spec is None:
            return
        close = getattr(spec.callback, "close", None)
        if callable(close):
            close()

    def list(self) -> tuple[HookSpec, ...]:
        return tuple(sorted(self._hooks.values(), key=lambda item: (item.order, item.hook_id)))

    @staticmethod
    def _allowed(spec: HookSpec, capabilities: set[str]) -> bool:
        return set(spec.required_capabilities) <= capabilities

    @staticmethod
    def _run_sync(spec: HookSpec, payload: Mapping[str, Any]) -> Any:
        """Run sync hooks with a bounded caller wait.

        Python cannot safely kill an arbitrary callback.  A timed-out callback
        therefore finishes in a daemon thread, while the agent path returns
        according to the hook failure policy and never holds a worker hostage.
        """
        result_box: list[Any] = []
        error_box: list[BaseException] = []
        finished = threading.Event()

        def invoke() -> None:
            try:
                result_box.append(spec.callback(payload))
            except BaseException as exc:  # propagate callback failures below
                error_box.append(exc)
            finally:
                finished.set()

        thread = threading.Thread(
            target=invoke,
            name=f"kogwistar-hook-{spec.hook_id}",
            daemon=True,
        )
        thread.start()
        if not finished.wait(spec.timeout_ms / 1000):
            raise TimeoutError(f"hook timed out after {spec.timeout_ms} ms: {spec.hook_id}")
        if error_box:
            raise error_box[0]
        result = result_box[0] if result_box else None
        if inspect.isawaitable(result):
            raise RuntimeError("async hook requires HookRegistry.arun()")
        return result

    async def arun(
        self,
        payload: Mapping[str, Any],
        *,
        effective_capabilities: tuple[str, ...] = (),
    ) -> tuple[HookResult, ...]:
        import asyncio

        capabilities = set(effective_capabilities)
        results: list[HookResult] = []
        for spec in self.list():
            if not self._allowed(spec, capabilities):
                results.append(HookResult(spec.hook_id, "skipped", {}, "capability denied"))
                continue
            try:
                if _is_async_callable(spec.callback):
                    value = await asyncio.wait_for(
                        _await_value(spec.callback(payload)), spec.timeout_ms / 1000
                    )
                else:
                    value = await _run_sync_async(spec, payload)
                    if inspect.isawaitable(value):
                        value = await asyncio.wait_for(
                            _await_value(value), spec.timeout_ms / 1000
                        )
                annotations = dict(value) if isinstance(value, Mapping) else {}
                results.append(HookResult(spec.hook_id, "applied", annotations))
            except Exception as exc:
                if spec.failure_mode == "fail_closed":
                    raise
                results.append(HookResult(spec.hook_id, "failed", {}, str(exc)))
        return tuple(results)

    def run(
        self,
        payload: Mapping[str, Any],
        *,
        effective_capabilities: tuple[str, ...] = (),
    ) -> tuple[HookResult, ...]:
        import asyncio

        if any(inspect.iscoroutinefunction(spec.callback) for spec in self.list()):
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(self.arun(payload, effective_capabilities=effective_capabilities))
            raise RuntimeError("async hook requires await HookRegistry.arun()")
        capabilities = set(effective_capabilities)
        results: list[HookResult] = []
        for spec in self.list():
            if not self._allowed(spec, capabilities):
                results.append(HookResult(spec.hook_id, "skipped", {}, "capability denied"))
                continue
            try:
                value = self._run_sync(spec, payload)
                annotations = dict(value) if isinstance(value, Mapping) else {}
                results.append(HookResult(spec.hook_id, "applied", annotations))
            except Exception as exc:
                if spec.failure_mode == "fail_closed":
                    raise
                results.append(HookResult(spec.hook_id, "failed", {}, str(exc)))
        return tuple(results)


async def _await_value(value: Any) -> Any:
    if isinstance(value, Awaitable):
        return await value
    return value


def _is_async_callable(callback: HookCallback) -> bool:
    return inspect.iscoroutinefunction(callback) or inspect.iscoroutinefunction(
        getattr(callback, "__call__", None)
    )


async def _run_sync_async(spec: HookSpec, payload: Mapping[str, Any]) -> Any:
    """Run sync callback in a daemon thread without executor shutdown waits."""

    import asyncio

    result_box: list[Any] = []
    error_box: list[BaseException] = []
    finished = threading.Event()

    def invoke() -> None:
        try:
            result_box.append(spec.callback(payload))
        except BaseException as exc:
            error_box.append(exc)
        finally:
            finished.set()

    threading.Thread(
        target=invoke,
        name=f"kogwistar-hook-{spec.hook_id}",
        daemon=True,
    ).start()
    deadline = asyncio.get_running_loop().time() + spec.timeout_ms / 1000
    while not finished.is_set():
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise TimeoutError(f"hook timed out after {spec.timeout_ms} ms: {spec.hook_id}")
        await asyncio.sleep(min(0.01, remaining))
    if error_box:
        raise error_box[0]
    return result_box[0] if result_box else None


__all__ = ["HookFailureMode", "HookEffect", "HookSpec", "HookResult", "HookRegistry"]
