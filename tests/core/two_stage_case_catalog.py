"""Shared case IDs for ADR-018 backend parity tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T", bound=Callable[..., Any])

TWO_STAGE_COMMON_CASES = frozenset(
    {
        "pending_visibility",
        "promotion_handoff",
        "batch_embedding",
        "stale_revision",
        "delete_race",
        "recovery_reconciliation",
    }
)


def two_stage_case(case_id: str) -> Callable[[T], T]:
    if case_id not in TWO_STAGE_COMMON_CASES:
        raise ValueError(f"unknown two-stage parity case: {case_id}")

    def decorate(test: T) -> T:
        cases = set(getattr(test, "__two_stage_cases__", ()))
        cases.add(case_id)
        setattr(test, "__two_stage_cases__", frozenset(cases))
        return test

    return decorate
