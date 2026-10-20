from __future__ import annotations

import pytest

from kogwistar.runtime.runtime_work_state import RuntimeWorkState

pytestmark = pytest.mark.ci


def test_runtime_work_state_tracks_live_work_and_clamps_at_zero() -> None:
    state = RuntimeWorkState()

    state.add(3)
    assert state.snapshot() == 3

    state.done(2)
    assert state.snapshot() == 1

    state.done(4)
    assert state.snapshot() == 0


def test_runtime_work_state_ignores_nonpositive_adjustments() -> None:
    state = RuntimeWorkState()

    state.add(0)
    state.add(-1)
    state.done(0)
    state.done(-1)

    assert state.snapshot() == 0
