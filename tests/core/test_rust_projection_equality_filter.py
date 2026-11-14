from __future__ import annotations

from kogwistar.engine_core.subsystems.read import ReadSubsystem


def test_native_equality_filter_accepts_flat_conjunction() -> None:
    assert ReadSubsystem._native_equality_filter(
        {"$and": [{"entity_type": "workflow_completed"}, {"run_id": "run-1"}]}
    ) == {"entity_type": "workflow_completed", "run_id": "run-1"}


def test_native_equality_filter_rejects_unsupported_or_ambiguous_filters() -> None:
    assert ReadSubsystem._native_equality_filter({"$or": [{"run_id": "run-1"}]}) is None
    assert ReadSubsystem._native_equality_filter(
        {"$and": [{"run_id": "run-1"}, {"run_id": "run-2"}]}
    ) is None
    assert ReadSubsystem._native_equality_filter({"run_id": {"$ne": "run-1"}}) is None
