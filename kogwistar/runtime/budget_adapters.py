from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .budget import BudgetAttribution, BudgetEvent


@runtime_checkable
class BudgetAdapter(Protocol):
    name: str

    def can_adapt(self, source: Any) -> bool: ...

    def adapt(
        self,
        source: Any,
        *,
        run_id: str,
        scope: str = "run",
        attribution: BudgetAttribution | None = None,
    ) -> list[BudgetEvent]: ...


@dataclass(frozen=True)
class GenericUsageAdapter:
    name: str = "generic-usage"

    def can_adapt(self, source: Any) -> bool:
        return isinstance(source, dict) and "usage" in source

    def adapt(
        self,
        source: Any,
        *,
        run_id: str,
        scope: str = "run",
        attribution: BudgetAttribution | None = None,
    ) -> list[BudgetEvent]:
        usage = source.get("usage") if isinstance(source, dict) else None
        if not isinstance(usage, dict):
            return []
        out: list[BudgetEvent] = []
        token_keys = [
            key
            for key in ("input_tokens", "cached_input_tokens", "output_tokens")
            if usage.get(key) is not None
        ]
        if not token_keys and usage.get("total_tokens") is not None:
            token_keys.append("total_tokens")
        for key in (*token_keys, "total_cost"):
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
                    attribution=attribution,
                )
            )
        return out


DEFAULT_BUDGET_ADAPTERS: list[BudgetAdapter] = [GenericUsageAdapter()]


def adapt_budget_events(
    source: Any,
    *,
    run_id: str,
    scope: str = "run",
    attribution: BudgetAttribution | None = None,
    adapters: list[BudgetAdapter] | None = None,
) -> list[BudgetEvent]:
    for adapter in adapters or DEFAULT_BUDGET_ADAPTERS:
        if adapter.can_adapt(source):
            if attribution is None:
                # Preserve compatibility with adapters implemented before the
                # attribution context was added.
                return adapter.adapt(source, run_id=run_id, scope=scope)
            return adapter.adapt(source, run_id=run_id, scope=scope, attribution=attribution)
    return []


def summarize_budget_events(events: list[BudgetEvent]) -> dict[str, Any]:
    input_tokens = 0
    cached_input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    total_cost = 0.0
    time_ms = 0
    event_counts: Counter[str] = Counter()
    by_unit: Counter[str] = Counter()
    cost_provenance: Counter[str] = Counter()
    for event in events:
        kind = str(getattr(event, "kind", "unknown"))
        unit = str(getattr(event, "unit", ""))
        amount = getattr(event, "amount", 0)
        event_counts[kind] += 1
        if unit:
            by_unit[unit] += 1
        if kind == "cost":
            provenance = getattr(event, "meta", {}).get("cost_provenance")
            if provenance:
                cost_provenance[str(provenance)] += 1
        if kind in {"debit", "token"} and unit == "input_tokens":
            input_tokens += int(amount or 0)
            total_tokens += int(amount or 0)
        elif kind in {"debit", "token"} and unit == "output_tokens":
            output_tokens += int(amount or 0)
            total_tokens += int(amount or 0)
        elif kind in {"debit", "token"} and unit == "cached_input_tokens":
            cached_input_tokens += int(amount or 0)
        elif kind in {"debit", "token"} and unit == "total_tokens":
            total_tokens += int(amount or 0)
        elif kind == "cost" or unit == "total_cost":
            total_cost += float(amount or 0.0)
        elif unit == "ms":
            time_ms += int(amount or 0)
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "total_cost": round(total_cost, 6),
        "time_ms": time_ms,
        "event_count": len(events),
        "event_counts": dict(event_counts),
        "by_unit": dict(by_unit),
        "cost_provenance": dict(cost_provenance),
    }

