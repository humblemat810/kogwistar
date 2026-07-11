from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any


class BudgetExhaustedError(RuntimeError):
    pass


@dataclass(frozen=True)
class BudgetAttribution:
    """Optional dimensions used by read-side usage projections."""

    workspace_id: str | None = None
    source_document_id: str | None = None
    operation_id: str | None = None
    operation_kind: str | None = None
    maintenance_job_id: str | None = None
    dream_job_id: str | None = None
    provider: str | None = None
    model: str | None = None

    def as_dict(self) -> dict[str, str]:
        return {
            key: value
            for key, value in {
                "workspace_id": self.workspace_id,
                "source_document_id": self.source_document_id,
                "operation_id": self.operation_id,
                "operation_kind": self.operation_kind,
                "maintenance_job_id": self.maintenance_job_id,
                "dream_job_id": self.dream_job_id,
                "provider": self.provider,
                "model": self.model,
            }.items()
            if value is not None
        }


@dataclass(frozen=True)
class BudgetEvent:
    run_id: str
    source: str
    kind: str
    amount: float
    unit: str
    scope: str = "run"
    ts_ms: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    event_id: str | None = None
    attribution: BudgetAttribution | None = None


def budget_event_to_dict(event: BudgetEvent) -> dict[str, Any]:
    """Serialize a budget event without exposing dataclass implementation details."""

    payload: dict[str, Any] = {
        "event_id": event.event_id,
        "run_id": event.run_id,
        "source": event.source,
        "kind": event.kind,
        "amount": event.amount,
        "unit": event.unit,
        "scope": event.scope,
        "ts_ms": event.ts_ms,
        "meta": dict(event.meta),
    }
    if event.attribution is not None:
        payload["attribution"] = event.attribution.as_dict()
    return payload


def budget_event_from_dict(payload: dict[str, Any]) -> BudgetEvent:
    """Deserialize the stable event envelope used by raw usage projections."""

    raw_attribution = payload.get("attribution")
    attribution = (
        BudgetAttribution(**dict(raw_attribution))
        if isinstance(raw_attribution, dict)
        else None
    )
    amount = float(payload.get("amount") or 0.0)
    if not math.isfinite(amount) or amount < 0:
        raise ValueError("budget event amount must be finite and >= 0")
    return BudgetEvent(
        event_id=str(payload["event_id"]) if payload.get("event_id") is not None else None,
        run_id=str(payload.get("run_id") or ""),
        source=str(payload.get("source") or "unknown"),
        kind=str(payload.get("kind") or "unknown"),
        amount=amount,
        unit=str(payload.get("unit") or ""),
        scope=str(payload.get("scope") or "run"),
        ts_ms=int(payload["ts_ms"]) if payload.get("ts_ms") is not None else None,
        meta=dict(payload.get("meta") or {}),
        attribution=attribution,
    )


@dataclass
class RateBudgetWindow:
    limit: int = 0
    used: int = 0
    window_ms: int = 0
    window_started_ms: int = 0

    def refresh(self, *, now_ms: int) -> None:
        if self.window_ms and self.window_started_ms and now_ms - self.window_started_ms >= self.window_ms:
            self.window_started_ms = now_ms
            self.used = 0

    def remaining(self, *, now_ms: int) -> int:
        self.refresh(now_ms=now_ms)
        return max(0, int(self.limit) - int(self.used))

    def debit(self, amount: int, *, now_ms: int) -> None:
        self.refresh(now_ms=now_ms)
        if amount < 0:
            raise ValueError("amount must be >= 0")
        if self.limit and self.used + amount > self.limit:
            raise BudgetExhaustedError(
                f"rate budget exhausted: used={self.used} total={self.limit}"
            )
        self.used += amount

    def next_refresh_ms(self) -> int | None:
        if not self.window_ms or not self.window_started_ms:
            return None
        return int(self.window_started_ms + self.window_ms)

    def is_pinned_until_refresh(self, *, now_ms: int) -> bool:
        next_refresh = self.next_refresh_ms()
        if next_refresh is None:
            return False
        return int(now_ms) < int(next_refresh)


@dataclass
class BudgetLedger:
    total: int
    used: int = 0
    events: list[BudgetEvent] = field(default_factory=list)

    def debit(
        self,
        amount: int | float,
        *,
        reason: str = "step",
        source: str = "runtime",
        run_id: str = "",
        unit: str = "token",
        attribution: BudgetAttribution | None = None,
        event_id: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        amount = int(amount or 0)
        if amount < 0:
            raise ValueError("amount must be >= 0")
        if self.used + amount > self.total:
            self.events.append(
                BudgetEvent(
                    run_id=run_id,
                    source=source,
                    kind="exhausted",
                    amount=float(amount),
                    unit=unit,
                    scope="run",
                    meta={"reason": reason, **(meta or {})},
                    event_id=event_id,
                    attribution=attribution,
                )
            )
            raise BudgetExhaustedError(f"budget exhausted: used={self.used} total={self.total}")
        self.used += amount
        self.events.append(
            BudgetEvent(
                run_id=run_id,
                source=source,
                kind="debit",
                amount=float(amount),
                unit=unit,
                scope="run",
                meta={"reason": reason, **(meta or {})},
                event_id=event_id,
                attribution=attribution,
            )
        )

    @property
    def remaining(self) -> int:
        return max(0, int(self.total) - int(self.used))

    def ingest(self, event: BudgetEvent) -> None:
        if event.kind in {"debit", "token"} and event.unit != "ms":
            self.debit(
                int(event.amount),
                reason=str(event.meta.get("reason") or event.kind or "event"),
                source=event.source,
                run_id=event.run_id,
                unit=event.unit or "token",
                attribution=event.attribution,
                event_id=event.event_id,
                meta=event.meta,
            )
            return
        if event.kind == "time" or event.unit == "ms":
            self.events.append(event)
            return
        if event.kind == "cost":
            self.events.append(event)
            return
        self.events.append(event)


@dataclass
class StateBackedBudgetLedger:
    state: dict[str, Any]
    events: list[BudgetEvent] = field(default_factory=list)

    @property
    def total(self) -> int:
        return int(self.state.get("token_budget", 0) or 0)

    @property
    def used(self) -> int:
        return int(self.state.get("token_used", 0) or 0)

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.used)

    @property
    def time_budget_ms(self) -> int:
        return int(self.state.get("time_budget_ms", 0) or 0)

    @property
    def time_used_ms(self) -> int:
        return int(self.state.get("time_used_ms", 0) or 0)

    @property
    def rate_limit(self) -> int:
        return int(self.state.get("rate_limit", 0) or 0)

    @property
    def rate_used(self) -> int:
        return int(self.state.get("rate_used", 0) or 0)

    @property
    def rate_window_ms(self) -> int:
        return int(self.state.get("rate_window_ms", 0) or 0)

    @property
    def rate_window_started_ms(self) -> int:
        return int(self.state.get("rate_window_started_ms", 0) or 0)

    @property
    def rate_window_ready_ms(self) -> int | None:
        if not self.rate_limit or not self.rate_window_ms or not self.rate_window_started_ms:
            return None
        return int(self.rate_window_started_ms + self.rate_window_ms)

    @property
    def step_budget(self) -> int:
        return int(self.state.get("step_budget", 0) or 0)

    @property
    def step_used(self) -> int:
        return int(self.state.get("step_used", 0) or 0)

    @property
    def call_budget(self) -> int:
        return int(self.state.get("call_budget", 0) or 0)

    @property
    def call_used(self) -> int:
        return int(self.state.get("call_used", 0) or 0)

    def debit_step(
        self,
        amount: int = 1,
        *,
        reason: str = "workflow_step",
        run_id: str = "",
        attribution: BudgetAttribution | None = None,
    ) -> None:
        amount = int(amount or 0)
        if amount < 0:
            raise ValueError("step amount must be >= 0")
        next_used = self.step_used + amount
        if self.step_budget and next_used > self.step_budget:
            raise BudgetExhaustedError(
                f"step budget exhausted: used={self.step_used} total={self.step_budget}"
            )
        self.state["step_used"] = next_used
        self.events.append(
            BudgetEvent(
                run_id=run_id,
                source="runtime",
                kind="debit",
                amount=float(amount),
                unit="step",
                scope=str(self.state.get("budget_scope") or "run"),
                meta={"reason": reason},
                attribution=attribution,
            )
        )

    def debit_call(
        self,
        amount: int = 1,
        *,
        reason: str = "llm_call",
        run_id: str = "",
        attribution: BudgetAttribution | None = None,
    ) -> None:
        amount = int(amount or 0)
        if amount < 0:
            raise ValueError("call amount must be >= 0")
        next_used = self.call_used + amount
        if self.call_budget and next_used > self.call_budget:
            raise BudgetExhaustedError(
                f"call budget exhausted: used={self.call_used} total={self.call_budget}"
            )
        self.state["call_used"] = next_used
        self.events.append(
            BudgetEvent(
                run_id=run_id,
                source="runtime",
                kind="debit",
                amount=float(amount),
                unit="call",
                scope=str(self.state.get("budget_scope") or "run"),
                meta={"reason": reason},
                attribution=attribution,
            )
        )

    def debit(
        self,
        amount: int | float,
        *,
        reason: str = "step",
        source: str = "runtime",
        run_id: str = "",
        unit: str | None = None,
        attribution: BudgetAttribution | None = None,
        event_id: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        amount = int(amount or 0)
        if amount < 0:
            raise ValueError("amount must be >= 0")
        next_used = self.used + amount
        if next_used > self.total:
            self.events.append(
                BudgetEvent(
                    run_id=run_id,
                    source=source,
                    kind="exhausted",
                    amount=float(amount),
                    unit=unit or str(self.state.get("budget_kind") or "token"),
                    scope=str(self.state.get("budget_scope") or "run"),
                    meta={"reason": reason, **(meta or {})},
                    event_id=event_id,
                    attribution=attribution,
                )
            )
            raise BudgetExhaustedError(
                f"budget exhausted: used={self.used} total={self.total}"
            )
        now_ms = int(self.state.get("now_ms", 0) or 0)
        if self.rate_limit:
            window = RateBudgetWindow(
                limit=self.rate_limit,
                used=self.rate_used,
                window_ms=self.rate_window_ms,
                window_started_ms=self.rate_window_started_ms or now_ms,
            )
            window.debit(amount, now_ms=now_ms or window.window_started_ms)
            self.state["rate_used"] = window.used
            self.state["rate_window_started_ms"] = window.window_started_ms
        self.state["token_used"] = next_used
        self.events.append(
            BudgetEvent(
                run_id=run_id,
                source=source,
                kind="debit",
                amount=float(amount),
                unit=unit or str(self.state.get("budget_kind") or "token"),
                scope=str(self.state.get("budget_scope") or "run"),
                meta={"reason": reason, **(meta or {})},
                event_id=event_id,
                attribution=attribution,
            )
        )

    def ingest(self, event: BudgetEvent) -> None:
        if event.kind in {"debit", "token"} and event.unit != "ms":
            self.debit(
                int(event.amount),
                reason=str(event.meta.get("reason") or event.kind or "event"),
                source=event.source,
                run_id=event.run_id,
                unit=event.unit or None,
                attribution=event.attribution,
                event_id=event.event_id,
                meta=event.meta,
            )
            return
        if event.kind == "time" or event.unit == "ms":
            self.debit_time(
                int(event.amount),
                reason=str(event.meta.get("reason") or event.kind or "event"),
                source=event.source,
                run_id=event.run_id,
                attribution=event.attribution,
                event_id=event.event_id,
                meta=event.meta,
            )
            return
        if event.kind == "cost":
            self.state["cost_used"] = float(self.state.get("cost_used", 0.0) or 0.0) + float(
                event.amount or 0
            )
            self.events.append(event)
            return
        self.events.append(event)

    def debit_time(
        self,
        amount_ms: int | float,
        *,
        reason: str = "step",
        source: str = "runtime",
        run_id: str = "",
        attribution: BudgetAttribution | None = None,
        event_id: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        amount_ms = int(amount_ms or 0)
        if amount_ms < 0:
            raise ValueError("amount_ms must be >= 0")
        next_used = self.time_used_ms + amount_ms
        if self.time_budget_ms and next_used > self.time_budget_ms:
            self.events.append(
                BudgetEvent(
                    run_id=run_id,
                    source=source,
                    kind="exhausted",
                    amount=float(amount_ms),
                    unit="ms",
                    scope=str(self.state.get("budget_scope") or "run"),
                    meta={"reason": reason, **(meta or {})},
                    event_id=event_id,
                    attribution=attribution,
                )
            )
            raise BudgetExhaustedError(
                f"time budget exhausted: used={self.time_used_ms} total={self.time_budget_ms}"
            )
        self.state["time_used_ms"] = next_used
        self.events.append(
            BudgetEvent(
                run_id=run_id,
                source=source,
                kind="debit",
                amount=float(amount_ms),
                unit="ms",
                scope=str(self.state.get("budget_scope") or "run"),
                meta={"reason": reason, **(meta or {})},
                event_id=event_id,
                attribution=attribution,
            )
        )

    def should_suspend_for_budget(self) -> bool:
        if self.total and self.used >= self.total:
            return True
        if self.time_budget_ms and self.time_used_ms >= self.time_budget_ms:
            return True
        if self.rate_limit and self.rate_used >= self.rate_limit:
            return True
        if self.step_budget and self.step_used >= self.step_budget:
            return True
        if self.call_budget and self.call_used >= self.call_budget:
            return True
        return False

    def is_pinned_until_refresh(self, *, now_ms: int) -> bool:
        ready_ms = self.rate_window_ready_ms
        if ready_ms is None:
            return False
        return bool(self.rate_limit and self.rate_used >= self.rate_limit and now_ms < ready_ms)
