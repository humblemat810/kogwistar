from __future__ import annotations

import pytest

from kogwistar.runtime.budget import (
    BudgetExhaustedError,
    BudgetLedger,
    StateBackedBudgetLedger,
)
from kogwistar.runtime.budget_adapters import adapt_budget_events
from kogwistar.runtime.cost_ledger import CostLedger
from kogwistar.conversation.conversation_state_contracts import (
    PrevTurnMetaSummaryModel,
    WorkflowStateModel,
)


def test_budget_ledger_debit_and_remaining() -> None:
    ledger = BudgetLedger(total=10)
    ledger.debit(3, reason="step")
    assert ledger.used == 3
    assert ledger.remaining == 7
    assert ledger.events[0].kind == "debit"


def test_budget_ledger_exhausts_loudly() -> None:
    ledger = BudgetLedger(total=1)
    ledger.debit(1)
    with pytest.raises(BudgetExhaustedError):
        ledger.debit(1)


def test_cost_ledger_records_events() -> None:
    ledger = CostLedger(workspace_id="ws-1")
    evt = ledger.add_event(kind="token", amount=5, source="run")
    assert evt["workspace_id"] == "ws-1"
    assert ledger.events[0]["amount"] == 5
    assert ledger.snapshot()["event_count"] == 1


def test_budget_ledger_ingests_canonical_event() -> None:
    ledger = BudgetLedger(total=10)
    event = adapt_budget_events({"usage": {"input_tokens": 4}}, run_id="run-1")[0]
    ledger.ingest(event)
    assert ledger.used == 4
    assert ledger.events[-1].kind == "debit"


def test_workflow_state_model_carries_budget_shape() -> None:
    state = WorkflowStateModel.model_construct(
        conversation_id="c1",
        user_id="u1",
        turn_node_id="t1",
        turn_index=0,
        mem_id="m1",
        self_span=None,
        role="user",
        user_text="hi",
        embedding=None,
        prev_turn_meta_summary=PrevTurnMetaSummaryModel(
            prev_node_char_distance_from_last_summary=0,
            prev_node_distance_from_last_summary=0,
            tail_turn_index=0,
        ),
        _deps={},
    )
    dumped = state.dump_state()
    assert dumped["budget"]["token_budget"] == 0
    assert dumped["budget"]["budget_kind"] == "token"


def test_state_backed_budget_ledger_mutates_persisted_state() -> None:
    state = {
        "token_budget": 10,
        "token_used": 1,
        "time_budget_ms": 50,
        "time_used_ms": 5,
        "budget_kind": "token",
        "budget_scope": "run",
    }
    ledger = StateBackedBudgetLedger(state)
    ledger.debit(3, reason="step", run_id="run-1")
    assert state["token_used"] == 4
    assert ledger.remaining == 6


def test_state_backed_budget_ledger_tracks_time_budget() -> None:
    state = {
        "token_budget": 10,
        "token_used": 1,
        "time_budget_ms": 50,
        "time_used_ms": 5,
        "budget_kind": "ms",
        "budget_scope": "run",
    }
    ledger = StateBackedBudgetLedger(state)
    ledger.debit_time(7, reason="step", run_id="run-1")
    assert state["time_used_ms"] == 12
    assert ledger.time_budget_ms == 50


def test_state_backed_budget_ledger_tracks_cost_events() -> None:
    state = {
        "token_budget": 10,
        "token_used": 1,
        "time_budget_ms": 50,
        "time_used_ms": 5,
        "cost_budget": 3.0,
        "cost_used": 0.5,
        "budget_kind": "token",
        "budget_scope": "run",
    }
    ledger = StateBackedBudgetLedger(state)
    event = adapt_budget_events({"usage": {"total_cost": 1.25}}, run_id="run-1")[0]
    ledger.ingest(event)
    assert state["cost_used"] == pytest.approx(1.75)


def test_cost_ledger_can_ingest_canonical_budget_event() -> None:
    ledger = CostLedger(workspace_id="ws-2")
    event = adapt_budget_events({"usage": {"total_cost": 3}}, run_id="run-2")[0]
    ledger.ingest(event)
    assert ledger.snapshot()["total_amount"] == 3
