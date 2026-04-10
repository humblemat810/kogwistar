from __future__ import annotations

import json

import pytest

from kogwistar.runtime.perf_profile import (
    format_profile_report,
    profile_in_memory_index_job_breakdown,
    profile_in_memory_checkpoint_write,
    profile_in_memory_checkpoint_write_mode,
)


pytestmark = [pytest.mark.manual, pytest.mark.core]


def test_manual_profile_in_memory_checkpoint_write(tmp_path) -> None:
    report = profile_in_memory_checkpoint_write(
        tmp_path / "profile_case",
        iterations=2,
        compare_fast_path=True,
        include_monitoring=False,
    )

    print(format_profile_report(report))
    print(json.dumps(report, indent=2, sort_keys=True))

    assert report["iterations"] == 2
    assert "eager_reconcile" in report["scenarios"]
    assert "fast_inline" in report["scenarios"]
    assert report["scenarios"]["eager_reconcile"]["method_timings"]
    assert report["scenarios"]["fast_inline"]["method_timings"]

    eager_ms = float(report["scenarios"]["eager_reconcile"]["scenario_total_ms"])
    fast_ms = float(report["scenarios"]["fast_inline"]["scenario_total_ms"])
    speedup = eager_ms / fast_ms if fast_ms > 0 else float("inf")
    print(f"Speedup: {speedup:.2f}x (eager={eager_ms:.3f} ms, fast={fast_ms:.3f} ms)")

    # Manual benchmark guard, not a strict microbenchmark.
    assert fast_ms < eager_ms


def test_manual_profile_in_memory_checkpoint_write_total_speedup(tmp_path) -> None:
    baseline = profile_in_memory_checkpoint_write_mode(
        tmp_path / "profile_case_baseline",
        iterations=1,
        fast_trace_persistence=False,
        include_monitoring=False,
        use_validation_cache=False,
    )
    optimized = profile_in_memory_checkpoint_write_mode(
        tmp_path / "profile_case_optimized",
        iterations=1,
        fast_trace_persistence=True,
        include_monitoring=False,
        use_validation_cache=True,
    )

    print("[baseline]")
    print(format_profile_report(baseline))
    print(json.dumps(baseline, indent=2, sort_keys=True))
    print("[optimized]")
    print(format_profile_report(optimized))
    print(json.dumps(optimized, indent=2, sort_keys=True))

    base_eager = float(next(iter(baseline["scenarios"].values()))["scenario_total_ms"])
    opt_fast = float(next(iter(optimized["scenarios"].values()))["scenario_total_ms"])
    combined_speedup = base_eager / opt_fast if opt_fast > 0 else float("inf")

    print(
        f"Combined end-to-end speedup: {combined_speedup:.2f}x "
        f"(baseline={base_eager:.3f} ms, optimized={opt_fast:.3f} ms)"
    )

    assert opt_fast < base_eager


def test_manual_profile_in_memory_index_job_breakdown(tmp_path) -> None:
    uncached_report = profile_in_memory_index_job_breakdown(
        tmp_path / "job_breakdown_case_uncached",
        iterations=1,
        include_monitoring=False,
        use_validation_cache=False,
    )
    cached_report = profile_in_memory_index_job_breakdown(
        tmp_path / "job_breakdown_case",
        iterations=1,
        include_monitoring=False,
        use_validation_cache=True,
    )

    print("[uncached]")
    print(format_profile_report(uncached_report))
    print(json.dumps(uncached_report, indent=2, sort_keys=True))
    print("[cached]")
    print(format_profile_report(cached_report))
    print(json.dumps(cached_report, indent=2, sort_keys=True))

    assert uncached_report["iterations"] == 1
    assert cached_report["iterations"] == 1
    for report in (uncached_report, cached_report):
        assert "claim_only" in report["scenarios"]
        assert "apply_only" in report["scenarios"]
        assert "eager_reconcile" in report["scenarios"]
        assert report["scenarios"]["claim_only"]["method_timings"]
        assert report["scenarios"]["apply_only"]["method_timings"]
        assert report["scenarios"]["apply_only"]["internal_timings"]
        assert report["scenarios"]["eager_reconcile"]["method_timings"]
        assert report["scenarios"]["eager_reconcile"]["internal_timings"]

    uncached_apply = float(uncached_report["scenarios"]["apply_only"]["scenario_total_ms"])
    cached_apply = float(cached_report["scenarios"]["apply_only"]["scenario_total_ms"])
    uncached_eager = float(uncached_report["scenarios"]["eager_reconcile"]["scenario_total_ms"])
    cached_eager = float(cached_report["scenarios"]["eager_reconcile"]["scenario_total_ms"])

    print(
        f"Apply-only loop speedup: {uncached_apply / cached_apply:.2f}x "
        f"(uncached={uncached_apply:.3f} ms, cached={cached_apply:.3f} ms)"
    )
    print(
        f"Eager loop speedup: {uncached_eager / cached_eager:.2f}x "
        f"(uncached={uncached_eager:.3f} ms, cached={cached_eager:.3f} ms)"
    )

    assert cached_apply < uncached_apply
    assert cached_eager < uncached_eager

    uncached_models = uncached_report["scenarios"]["apply_only"]["internal_timings"]
    cached_models = cached_report["scenarios"]["apply_only"]["internal_timings"]

    def _count(timings: dict[str, dict[str, int | float]], key: str) -> int:
        return int((timings.get(key) or {}).get("count", 0))

    assert _count(cached_models, "apply.node_docs.model_validate") <= _count(
        uncached_models, "apply.node_docs.model_validate"
    )
    assert _count(cached_models, "apply.node_refs.model_validate") <= _count(
        uncached_models, "apply.node_refs.model_validate"
    )
    assert _count(cached_models, "apply.edge_refs.model_validate") <= _count(
        uncached_models, "apply.edge_refs.model_validate"
    )
    assert _count(cached_models, "apply.edge_endpoints.model_validate") <= _count(
        uncached_models, "apply.edge_endpoints.model_validate"
    )
