
import math
import pytest

# These imports assume you replaced the module file in your repo with the patched version.
from graph_knowledge_engine.conversation.conversation_orchestrator import (
    ExecClock,
    _estimate_tokens_from_chars,
    _stamp_meta,
)

def test_estimate_tokens_from_chars_default_proxy():
    assert _estimate_tokens_from_chars(0) == 0
    assert _estimate_tokens_from_chars(1) == 1
    assert _estimate_tokens_from_chars(4) == 1
    assert _estimate_tokens_from_chars(5) == 2
    assert _estimate_tokens_from_chars(8) == 2

def test_estimate_tokens_from_chars_custom_estimator_scales():
    # estimator: 1 token per 2 chars on the sample
    def est(s: str) -> int:
        return max(1, len(s) // 2)

    # For small sizes, scaling should be close
    assert _estimate_tokens_from_chars(10, est) in (5, 6)

    # For larger sizes, it should scale roughly linearly.
    t1 = _estimate_tokens_from_chars(4096, est)
    t2 = _estimate_tokens_from_chars(8192, est)
    assert t2 >= t1 * 2 - 2
    assert t2 <= t1 * 2 + 2

def test_stamp_meta_sets_defaults_without_overwrite():
    m = _stamp_meta({}, run_id="r", run_step_seq=7, attempt_seq=2)
    assert m["run_id"] == "r"
    assert m["run_step_seq"] == 7
    assert m["attempt_seq"] == 2

    m2 = _stamp_meta({"run_id": "x", "run_step_seq": 1}, run_id="r", run_step_seq=7, attempt_seq=2)
    # must not overwrite existing
    assert m2["run_id"] == "x"
    assert m2["run_step_seq"] == 1
    # missing attempt_seq should be filled
    assert m2["attempt_seq"] == 2

def test_exec_clock_bumps():
    c = ExecClock(run_id="conv1", run_step_seq=0, attempt_seq=0)
    c1 = c.bump_step()
    assert c1.run_step_seq == 1
    assert c1.attempt_seq == 0
    c2 = c1.bump_attempt()
    assert c2.run_step_seq == 1
    assert c2.attempt_seq == 1
