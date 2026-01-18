from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Union


Json = Any

# Import your real RunResult types from runtime/models
from graph_knowledge_engine.workflow.runtime import RunFailure, RunResult, RunSuccess, StepContext  # adjust import path

RawStepFn = Callable[[StepContext], Union[Json, RunResult]]

@dataclass
class MappingStepResolver:
    handlers: Dict[str, RawStepFn]
    default: Optional[RawStepFn] = None

    def __init__(self, handlers: Optional[Mapping[str, RawStepFn]] = None, *, default: Optional[RawStepFn] = None) -> None:
        self.handlers = dict(handlers or {})
        self.default = default

    def register(self, op: str) -> Callable[[RawStepFn], RawStepFn]:
        def _decorator(fn: RawStepFn) -> RawStepFn:
            self.handlers[op] = fn
            return fn
        return _decorator

    def resolve(self, op: str) -> Callable[[StepContext], RunResult]:
        raw = self.handlers.get(op) or self.default
        if raw is None:
            raise KeyError(f"No step handler registered for op={op!r}")

        def _wrapped(ctx: StepContext) -> RunResult:
            try:
                out = raw(ctx)
                if isinstance(out, (RunSuccess, RunFailure)):
                    return out
                return out  # field name might be `data`/`payload` in your codebase
            except Exception as e:
                return RunFailure(errors=[str(e)])  # match your RunFailure fields
        return _wrapped

    def __call__(self, op: str) -> Callable[[StepContext], RunResult]:
        return self.resolve(op)