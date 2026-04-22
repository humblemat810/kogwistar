from __future__ import annotations
import asyncio
import functools
import warnings

from kogwistar.utils.log import bind_log_context

from typing import TYPE_CHECKING

"""Workflow step resolvers.

This module provides a registry-based step resolver that can be used by
`WorkflowRuntime` (or any workflow executor) that expects:

    step_resolver(op_name: str) -> Callable[[StepContext], RunResult]

Design goals
------------
* Keep step implementations out of the orchestrator.
* Allow the orchestrator to inject runtime dependencies via `ctx.state["_deps"]`.
* Keep the step resolver contract stable: *handlers return RunResult*.

Dependency injection
--------------------
Handlers are expected to retrieve dependencies from `ctx.state["_deps"]`, e.g.:

    deps = ctx.state["_deps"]
    conversation_engine = deps["conversation_engine"]

The orchestrator should populate `_deps` in the workflow initial_state.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Union

# Best-effort self-inspection for state schema inference
import ast
import inspect

Json = Any
if TYPE_CHECKING:
    from .models import StepRunResult
    from .runtime import StepContext
    from .sandbox import Sandbox

    RawStepFn = Callable[[StepContext], Union[Json, StepRunResult]]

from kogwistar.runtime.models import RunSuccess, RunFailure, RunSuspended
from kogwistar.runtime.sandbox import SandboxRequest

# Import your real RunResult types from kogwistar.runtime/models




class BaseResolver:
    ops: set


_LEGACY_UPDATE_WARNING_EMITTED = False


@dataclass
class MappingStepResolver(BaseResolver):
    handlers: Dict[str, RawStepFn]
    default: Optional[RawStepFn] = None

    @property
    def ops(self):
        return set(self.handlers)

    def __init__(
        self,
        handlers: Optional[Mapping[str, RawStepFn]] = None,
        *,
        default: Optional[RawStepFn] = None,
    ) -> None:
        self.handlers = dict(handlers or {})
        self.default = default
        # Ops that are allowed to execute nested workflows (i.e., may call back into
        # WorkflowRuntime). Runtime may use this to avoid holding a step-UoW open.
        self.nested_ops: set[str] = set()
        # Ops that should run in a sandbox
        self.sandboxed_ops: set[str] = set()
        # Preferred merge mode per state key: 'u' overwrite, 'a' append, 'e' extend
        self._state_schema: dict[str, str] = {}
        # The sandbox to use
        self._sandbox: Optional["Sandbox"] = None

    def set_sandbox(self, sandbox: "Sandbox"):
        self._sandbox = sandbox

    def close_sandbox_run(self, run_id: str) -> None:
        if self._sandbox is not None and hasattr(self._sandbox, "close_run"):
            self._sandbox.close_run(str(run_id))

    def close_sandbox(self) -> None:
        if self._sandbox is not None and hasattr(self._sandbox, "close"):
            self._sandbox.close()

    def register(
        self, op: str, *, is_nested: bool = False, is_sandboxed: bool = False
    ) -> Callable[[RawStepFn], RawStepFn]:
        def _decorator(fn: RawStepFn) -> RawStepFn:
            @functools.wraps(fn)
            def wrapped_fun(*arg, **kwarg):
                ctx: StepContext = arg[0]
                with bind_log_context(
                    op=op,
                    conversation_id=ctx.conversation_id,
                    workflow_run_id=f"{ctx.workflow_id}--{ctx.run_id}",
                    step_id=ctx.workflow_node_id,
                ):
                    return fn(*arg, **kwarg)

            self.handlers[op] = fn  # wrapped_fun
            if is_nested:
                self.nested_ops.add(str(op))
            if is_sandboxed:
                self.sandboxed_ops.add(str(op))
            return wrapped_fun

        return _decorator

    def resolve(self, op: str) -> Callable[[StepContext], StepRunResult]:
        raw = self.handlers.get(op) or self.default
        from kogwistar.runtime.models import RunFailure

        if raw is None:
            raise KeyError(f"No step handler registered for op={op!r}")

        def _wrapped(ctx: StepContext) -> StepRunResult:
            try:
                out = raw(ctx)
                out = self._maybe_execute_sandboxed(op=op, ctx=ctx, out=out)
                if getattr(out, "update", None) is not None:
                    global _LEGACY_UPDATE_WARNING_EMITTED
                    if not _LEGACY_UPDATE_WARNING_EMITTED:
                        warnings.warn(
                            "legacy update detected, use state_update if you need to append list state multiple times",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        _LEGACY_UPDATE_WARNING_EMITTED = True
                if isinstance(out, (RunSuccess, RunFailure, RunSuspended)):
                    return out
                else:
                    raise TypeError("Resolver must return StepRunResult")
            except Exception as e:
                import traceback

                return RunFailure(
                    conversation_node_id=ctx.state_view.get("workflow_node_id"),
                    state_update=[("a", {"op_log": str(e)})],
                    errors=[str(e), traceback.format_exc()],
                )

        return _wrapped

    def _maybe_execute_sandboxed(self, *, op: str, ctx: "StepContext", out: Any) -> Any:
        if op not in self.sandboxed_ops:
            return out

        req = self._coerce_sandbox_request(out)
        if req is None:
            return out
        if self._sandbox is None:
            return RunFailure(
                conversation_node_id=ctx.workflow_node_id,
                state_update=[],
                errors=[
                    f"Sandboxed op '{op}' produced sandbox code but no sandbox runtime is configured"
                ],
            )

        sandbox_state = (
            dict(req.state) if req.state is not None else dict(ctx.state_view)
        )
        sandbox_context = self._sandbox_context(ctx)
        sandbox_context.update(req.context)
        return self._sandbox.run(req.code, sandbox_state, sandbox_context)

    @staticmethod
    def _coerce_sandbox_request(out: Any) -> SandboxRequest | None:
        if isinstance(out, SandboxRequest):
            return out
        if isinstance(out, str):
            return SandboxRequest(code=out)
        if isinstance(out, dict) and "sandbox_code" in out:
            code = out.get("sandbox_code")
            if not isinstance(code, str) or not code.strip():
                raise ValueError("sandbox_code must be a non-empty string")
            state = out.get("sandbox_state")
            if state is not None and not isinstance(state, dict):
                raise ValueError("sandbox_state must be a dict when provided")
            context = out.get("sandbox_context") or {}
            if not isinstance(context, dict):
                raise ValueError("sandbox_context must be a dict when provided")
            return SandboxRequest(code=code, state=state, context=context)
        return None

    @staticmethod
    def _sandbox_context(ctx: "StepContext") -> dict[str, Any]:
        return {
            "run_id": ctx.run_id,
            "workflow_id": ctx.workflow_id,
            "workflow_node_id": ctx.workflow_node_id,
            "op": ctx.op,
            "token_id": ctx.token_id,
            "attempt": ctx.attempt,
            "step_seq": ctx.step_seq,
            "conversation_id": ctx.conversation_id,
            "turn_node_id": ctx.turn_node_id,
        }

    # ------------------------------------------------------------------
    # State schema for native updates + LangGraph conversion
    # ------------------------------------------------------------------

    def set_state_schema(self, schema: Mapping[str, str]) -> None:
        """Set preferred merge mode per state key.

        Values should be one of: 'u' (overwrite), 'a' (append), 'e' (extend).
        """
        self._state_schema = {str(k): str(v) for k, v in dict(schema).items()}

    def describe_state(self) -> dict[str, str]:
        """Return the state schema for native updates / LangGraph conversion."""
        return dict(self._state_schema)

    def infer_state_schema_best_effort(self) -> dict[str, str]:
        """Best-effort inference of keys touched by resolvers.

        We intentionally keep this conservative:
        - Any key assigned via state["k"] = ...  -> 'u'
        - Any key via state.setdefault("k", []).append(...) -> 'a'
        - Any key via state.setdefault("k", []).extend(...) -> 'e'

        If inference fails, returns the existing schema unchanged.
        """
        inferred: dict[str, str] = dict(self._state_schema)

        def _note(k: str, mode: str) -> None:
            if k and mode in ("u", "a", "e"):
                # if conflicts, prefer 'u' (safest) over list modes
                prev = inferred.get(k)
                if prev is None:
                    inferred[k] = mode
                elif prev != mode:
                    inferred[k] = "u"

        for op, fn in list(self.handlers.items()):
            try:
                src = inspect.getsource(fn)
            except Exception:
                continue
            try:
                tree = ast.parse(src)
            except Exception:
                continue

            # We look for patterns, not full correctness.
            for node in ast.walk(tree):
                # state["k"] = ...
                if isinstance(node, ast.Assign):
                    for tgt in node.targets:
                        if (
                            isinstance(tgt, ast.Subscript)
                            and isinstance(tgt.value, ast.Name)
                            and tgt.value.id == "state"
                        ):
                            sl = tgt.slice
                            if isinstance(sl, ast.Constant) and isinstance(
                                sl.value, str
                            ):
                                _note(sl.value, "u")

                # state.setdefault("k", []).append/extend
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    attr = node.func.attr
                    if attr not in ("append", "extend"):
                        continue
                    recv = node.func.value
                    if not (
                        isinstance(recv, ast.Call)
                        and isinstance(recv.func, ast.Attribute)
                    ):
                        continue
                    if recv.func.attr != "setdefault":
                        continue
                    base = recv.func.value
                    if not (isinstance(base, ast.Name) and base.id == "state"):
                        continue
                    if not recv.args:
                        continue
                    k0 = recv.args[0]
                    if isinstance(k0, ast.Constant) and isinstance(k0.value, str):
                        _note(k0.value, "a" if attr == "append" else "e")

        self._state_schema = dict(inferred)
        return dict(self._state_schema)

    def __call__(self, op: str) -> Callable[[StepContext], StepRunResult]:
        return self.resolve(op)


def _deps(ctx: StepContext) -> Dict[str, Any]:
    deps = ctx.state_view.get("_deps")
    if not isinstance(deps, dict):
        raise RuntimeError(
            "StepContext.state['_deps'] must be a dict of injected dependencies"
        )
    return deps


class AsyncMappingStepResolver(MappingStepResolver):
    """Async-friendly resolver wrapper.

    - Keeps same registry surface as MappingStepResolver.
    - `resolve_async(op)` always returns an async callable.
    - Sync handlers are adapted to async via direct return.
    """

    def resolve_async(self, op: str):
        raw = self.handlers.get(op) or self.default
        if raw is None:
            raise KeyError(f"No step handler registered for op={op!r}")

        async def _wrapped(ctx: StepContext) -> StepRunResult:
            try:
                if inspect.iscoroutinefunction(raw):
                    out = await raw(ctx)
                else:
                    # Keep async runtime responsive by offloading blocking sync
                    # handlers onto a worker thread.
                    out = await asyncio.to_thread(raw, ctx)
                out = self._maybe_execute_sandboxed(op=op, ctx=ctx, out=out)
                if getattr(out, "update", None) is not None:
                    global _LEGACY_UPDATE_WARNING_EMITTED
                    if not _LEGACY_UPDATE_WARNING_EMITTED:
                        warnings.warn(
                            "legacy update detected, use state_update if you need to append list state multiple times",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        _LEGACY_UPDATE_WARNING_EMITTED = True
                if isinstance(out, (RunSuccess, RunFailure, RunSuspended)):
                    return out
                raise TypeError("Resolver must return StepRunResult")
            except Exception as e:
                import traceback

                return RunFailure(
                    conversation_node_id=ctx.state_view.get("workflow_node_id"),
                    state_update=[("a", {"op_log": str(e)})],
                    errors=[str(e), traceback.format_exc()],
                )

        return _wrapped
