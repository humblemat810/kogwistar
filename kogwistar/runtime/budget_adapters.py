from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .budget import BudgetEvent


@runtime_checkable
class BudgetAdapter(Protocol):
    name: str

    def can_adapt(self, source: Any) -> bool: ...

    def adapt(self, source: Any, *, run_id: str, scope: str = "run") -> list[BudgetEvent]: ...


@dataclass(frozen=True)
class GenericUsageAdapter:
    name: str = "generic-usage"

    def can_adapt(self, source: Any) -> bool:
        return isinstance(source, dict) and "usage" in source

    def adapt(self, source: Any, *, run_id: str, scope: str = "run") -> list[BudgetEvent]:
        usage = source.get("usage") if isinstance(source, dict) else None
        if not isinstance(usage, dict):
            return []
        out: list[BudgetEvent] = []
        for key in ("input_tokens", "output_tokens", "total_tokens", "total_cost"):
            value = usage.get(key)
            if value is None:
                continue
            kind = "cost" if key.endswith("cost") else "token"
            out.append(
                BudgetEvent(
                    run_id=run_id,
                    source=self.name,
                    kind=kind,
                    amount=float(value),
                    unit=key,
                    scope=scope,
                    meta={"raw_key": key},
                )
            )
        return out


DEFAULT_BUDGET_ADAPTERS: list[BudgetAdapter] = [GenericUsageAdapter()]


def adapt_budget_events(
    source: Any, *, run_id: str, scope: str = "run", adapters: list[BudgetAdapter] | None = None
) -> list[BudgetEvent]:
    for adapter in adapters or DEFAULT_BUDGET_ADAPTERS:
        if adapter.can_adapt(source):
            return adapter.adapt(source, run_id=run_id, scope=scope)
    return []

