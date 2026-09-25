"""Agent budget policy helpers backed by Kogwistar's generic ledger."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kogwistar.runtime.budget import StateBackedBudgetLedger


@dataclass(frozen=True, slots=True)
class AgentBudgetPolicy:
    """Requested ceilings; lower caller ceilings always win."""

    max_steps: int | None = None
    max_model_calls: int | None = None
    max_tokens: int | None = None
    max_time_ms: int | None = None
    max_cost: float | None = None

    def __post_init__(self) -> None:
        for name in ("max_steps", "max_model_calls", "max_tokens", "max_time_ms"):
            value = getattr(self, name)
            if value is not None and int(value) < 1:
                raise ValueError(f"{name} must be positive when supplied")
        if self.max_cost is not None and float(self.max_cost) <= 0:
            raise ValueError("max_cost must be positive when supplied")

    def install(self, state: dict[str, Any]) -> StateBackedBudgetLedger:
        """Install ceilings and a process-local ledger in existing DI state."""

        budget_state = state.get("budget")
        if not isinstance(budget_state, dict):
            budget_state = state
        limits = {
            "step_budget": self.max_steps,
            "call_budget": self.max_model_calls,
            "token_budget": self.max_tokens,
            "time_budget_ms": self.max_time_ms,
            "cost_budget": self.max_cost,
        }
        for key, requested in limits.items():
            if requested is None:
                continue
            current = budget_state.get(key)
            if current:
                budget_state[key] = min(float(current), float(requested))
                if key != "cost_budget":
                    budget_state[key] = int(budget_state[key])
            else:
                budget_state[key] = (
                    int(requested) if key != "cost_budget" else float(requested)
                )
        budget_state.setdefault("step_used", 0)
        budget_state.setdefault("call_used", 0)
        budget_state.setdefault("token_used", 0)
        budget_state.setdefault("time_used_ms", 0)
        budget_state.setdefault("cost_used", 0.0)
        deps = state.setdefault("_deps", {})
        if not isinstance(deps, dict):
            raise ValueError("workflow state _deps must be a dict")
        ledger = deps.get("budget_ledger")
        if not isinstance(ledger, StateBackedBudgetLedger) or ledger.state is not budget_state:
            ledger = StateBackedBudgetLedger(budget_state)
            deps["budget_ledger"] = ledger
        refresh_budget_hints(state, ledger=ledger)
        return ledger


def _remaining(limit: int | float, used: int | float) -> int | float | None:
    if not limit:
        return None
    return max(0, limit - used)


def budget_hints(
    state: dict[str, Any], *, ledger: StateBackedBudgetLedger | None = None
) -> dict[str, int | float | None]:
    """Return bounded display/context hints, never authority."""

    ledger = ledger or StateBackedBudgetLedger(state)
    return {
        "steps_remaining": _remaining(ledger.step_budget, ledger.step_used),
        "model_calls_remaining": _remaining(ledger.call_budget, ledger.call_used),
        "tokens_remaining": _remaining(ledger.total, ledger.used),
        "time_remaining_ms": _remaining(ledger.time_budget_ms, ledger.time_used_ms),
        "cost_remaining": _remaining(ledger.cost_budget, ledger.cost_used),
        "authoritative": False,
    }


def refresh_budget_hints(
    state: dict[str, Any], *, ledger: StateBackedBudgetLedger | None = None
) -> dict[str, int | float | None]:
    hints = budget_hints(state, ledger=ledger)
    state["agent_budget_hints"] = hints
    return hints
