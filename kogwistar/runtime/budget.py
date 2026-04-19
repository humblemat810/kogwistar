from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class BudgetExhaustedError(RuntimeError):
    pass


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


@dataclass
class BudgetLedger:
    total: int
    used: int = 0
    events: list[BudgetEvent] = field(default_factory=list)

    def debit(self, amount: int | float, *, reason: str = "step", source: str = "runtime", run_id: str = "") -> None:
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
                    unit="token",
                    scope="run",
                    meta={"reason": reason},
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
                unit="token",
                scope="run",
                meta={"reason": reason},
            )
        )

    @property
    def remaining(self) -> int:
        return max(0, int(self.total) - int(self.used))

    def ingest(self, event: BudgetEvent) -> None:
        if event.kind in {"debit", "token"}:
            self.debit(
                int(event.amount),
                reason=str(event.meta.get("reason") or event.kind or "event"),
                source=event.source,
                run_id=event.run_id,
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

    def debit(
        self,
        amount: int | float,
        *,
        reason: str = "step",
        source: str = "runtime",
        run_id: str = "",
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
                    unit=str(self.state.get("budget_kind") or "token"),
                    scope=str(self.state.get("budget_scope") or "run"),
                    meta={"reason": reason},
                )
            )
            raise BudgetExhaustedError(
                f"budget exhausted: used={self.used} total={self.total}"
            )
        self.state["token_used"] = next_used
        self.events.append(
            BudgetEvent(
                run_id=run_id,
                source=source,
                kind="debit",
                amount=float(amount),
                unit=str(self.state.get("budget_kind") or "token"),
                scope=str(self.state.get("budget_scope") or "run"),
                meta={"reason": reason},
            )
        )

    def ingest(self, event: BudgetEvent) -> None:
        if event.kind in {"debit", "token"}:
            self.debit(
                int(event.amount),
                reason=str(event.meta.get("reason") or event.kind or "event"),
                source=event.source,
                run_id=event.run_id,
            )
            return
        if event.kind == "time" or event.unit == "ms":
            self.debit_time(
                int(event.amount),
                reason=str(event.meta.get("reason") or event.kind or "event"),
                source=event.source,
                run_id=event.run_id,
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
                    meta={"reason": reason},
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
                meta={"reason": reason},
            )
        )
