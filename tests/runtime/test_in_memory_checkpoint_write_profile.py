from __future__ import annotations

import json

import pytest

from kogwistar.runtime.perf_profile import (
    format_profile_report,
    profile_in_memory_index_job_breakdown,
    profile_in_memory_index_job_worker_parallel,
    profile_in_memory_checkpoint_write,
    profile_in_memory_checkpoint_write_mode,
    profile_simple_resolver_workflow,
)


pytestmark = [pytest.mark.manual, pytest.mark.core]


def _profile_backend_args(request, backend_kind: str) -> dict[str, object]:
    sa_engine = None
    pg_schema = None
    if backend_kind == "pg":
        pytest.importorskip("pgvector")
        try:
            sa_engine = request.getfixturevalue("sa_engine")
            pg_schema = request.getfixturevalue("pg_schema")
        except Exception as exc:
            pytest.skip(f"pg backend unavailable for profiling benchmark: {exc}")
    return {
        "backend_kind": backend_kind,
        "sa_engine": sa_engine,
        "pg_schema": pg_schema,
    }

@pytest.mark.parametrize(
    "backend_kind",
    [
        pytest.param("fake", id="fake"),
        pytest.param("chroma", id="chroma"),
        pytest.param("pg", id="pg"),
    ],
)
def test_manual_profile_in_memory_checkpoint_write(
    tmp_path, request, backend_kind: str
) -> None:
    report = profile_in_memory_checkpoint_write(
        tmp_path / f"profile_case_{backend_kind}",
        **_profile_backend_args(request, backend_kind),
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
    print(
        f"[backend={backend_kind}] Speedup: {speedup:.2f}x "
        f"(eager={eager_ms:.3f} ms, fast={fast_ms:.3f} ms)"
    )

    # Manual benchmark guard, not a strict microbenchmark.
    assert fast_ms > 0
    assert eager_ms > 0

@pytest.mark.parametrize(
    "backend_kind",
    [
        pytest.param("fake", id="fake"),
        pytest.param("chroma", id="chroma"),
        pytest.param("pg", id="pg"),
    ],
)
def test_manual_profile_in_memory_checkpoint_write_total_speedup(
    tmp_path, request, backend_kind: str
) -> None:
    backend_args = _profile_backend_args(request, backend_kind)
    baseline = profile_in_memory_checkpoint_write_mode(
        tmp_path / f"profile_case_baseline_{backend_kind}",
        **backend_args,
        iterations=1,
        fast_trace_persistence=False,
        include_monitoring=False,
        use_validation_cache=False,
    )
    optimized = profile_in_memory_checkpoint_write_mode(
        tmp_path / f"profile_case_optimized_{backend_kind}",
        **backend_args,
        iterations=1,
        fast_trace_persistence=True,
        include_monitoring=False,
        use_validation_cache=True,
    )

    print("[baseline]")
    print(f"[backend={backend_kind}]")
    print(format_profile_report(baseline))
    print(json.dumps(baseline, indent=2, sort_keys=True))
    print("[optimized]")
    print(format_profile_report(optimized))
    print(json.dumps(optimized, indent=2, sort_keys=True))

    base_eager = float(next(iter(baseline["scenarios"].values()))["scenario_total_ms"])
    opt_fast = float(next(iter(optimized["scenarios"].values()))["scenario_total_ms"])
    combined_speedup = base_eager / opt_fast if opt_fast > 0 else float("inf")

    print(
        f"[backend={backend_kind}] Combined end-to-end speedup: {combined_speedup:.2f}x "
        f"(baseline={base_eager:.3f} ms, optimized={opt_fast:.3f} ms)"
    )

    assert opt_fast < base_eager


@pytest.mark.parametrize(
    "backend_kind",
    [
        pytest.param("fake", id="fake"),
        pytest.param("chroma", id="chroma"),
        pytest.param("pg", id="pg"),
    ],
)
def test_manual_profile_in_memory_index_job_breakdown(
    tmp_path, request, backend_kind: str
) -> None:
    backend_args = _profile_backend_args(request, backend_kind)
    uncached_report = profile_in_memory_index_job_breakdown(
        tmp_path / f"job_breakdown_case_uncached_{backend_kind}",
        **backend_args,
        iterations=1,
        include_monitoring=False,
        use_validation_cache=False,
    )
    cached_report = profile_in_memory_index_job_breakdown(
        tmp_path / f"job_breakdown_case_{backend_kind}",
        **backend_args,
        iterations=1,
        include_monitoring=False,
        use_validation_cache=True,
    )

    print("[uncached]")
    print(f"[backend={backend_kind}]")
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
        f"[backend={backend_kind}] Apply-only loop speedup: {uncached_apply / cached_apply:.2f}x "
        f"(uncached={uncached_apply:.3f} ms, cached={cached_apply:.3f} ms)"
    )
    print(
        f"[backend={backend_kind}] Eager loop speedup: {uncached_eager / cached_eager:.2f}x "
        f"(uncached={uncached_eager:.3f} ms, cached={cached_eager:.3f} ms)"
    )

    assert uncached_apply > 0
    assert cached_apply > 0
    assert uncached_eager > 0
    assert cached_eager > 0

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


@pytest.mark.parametrize(
    "backend_kind",
    [
        pytest.param("fake", id="fake"),
        pytest.param("chroma", id="chroma"),
        pytest.param("pg", id="pg"),
    ],
)
def test_manual_profile_in_memory_index_job_worker_parallel_speedup(
    tmp_path, request, backend_kind: str
) -> None:
    backend_args = _profile_backend_args(request, backend_kind)

    baseline = profile_in_memory_index_job_breakdown(
        tmp_path / f"worker_parallel_baseline_{backend_kind}",
        **backend_args,
        iterations=1,
        include_monitoring=False,
        use_validation_cache=False,
    )
    worker_1 = profile_in_memory_index_job_worker_parallel(
        tmp_path / f"worker_parallel_1_{backend_kind}",
        **backend_args,
        iterations=1,
        worker_count=1,
        include_monitoring=False,
        use_validation_cache=True,
    )
    worker_4 = profile_in_memory_index_job_worker_parallel(
        tmp_path / f"worker_parallel_4_{backend_kind}",
        **backend_args,
        iterations=1,
        worker_count=4,
        include_monitoring=False,
        use_validation_cache=True,
    )

    print(f"[backend={backend_kind}]")
    print("[baseline_eager]")
    print(format_profile_report(baseline))
    print(json.dumps(baseline, indent=2, sort_keys=True))
    print("[worker_1]")
    print(format_profile_report(worker_1))
    print(json.dumps(worker_1, indent=2, sort_keys=True))
    print("[worker_4]")
    print(format_profile_report(worker_4))
    print(json.dumps(worker_4, indent=2, sort_keys=True))

    baseline_ms = float(baseline["scenarios"]["eager_reconcile"]["scenario_total_ms"])
    worker1_ms = float(worker_1["scenarios"]["worker_1"]["scenario_total_ms"])
    worker4_ms = float(worker_4["scenarios"]["worker_4"]["scenario_total_ms"])

    worker1_wall_ms = float(worker_1["scenarios"]["worker_1"]["wall_total_ms"])
    worker4_wall_ms = float(worker_4["scenarios"]["worker_4"]["wall_total_ms"])

    print(
        f"Baseline eager -> 1 worker drain speedup: {baseline_ms / worker1_ms:.2f}x "
        f"(baseline={baseline_ms:.3f} ms, worker_1={worker1_ms:.3f} ms)"
    )
    print(
        f"Baseline eager -> 4 worker drain speedup: {baseline_ms / worker4_ms:.2f}x "
        f"(baseline={baseline_ms:.3f} ms, worker_4={worker4_ms:.3f} ms)"
    )
    print(
        f"Worker wall times: 1 worker={worker1_wall_ms:.3f} ms, "
        f"4 workers={worker4_wall_ms:.3f} ms"
    )

    assert baseline_ms > 0
    assert worker1_ms > 0
    assert worker4_ms > 0


@pytest.mark.parametrize(
    "backend_kind",
    [
        pytest.param("fake", id="fake"),
        pytest.param("chroma", id="chroma"),
        pytest.param("pg", id="pg"),
    ],
)
def test_manual_profile_simple_resolver_workflow_speedup(
    tmp_path, request, backend_kind: str
) -> None:
    report = profile_simple_resolver_workflow(
        tmp_path / f"simple_resolver_workflow_{backend_kind}",
        **_profile_backend_args(request, backend_kind),
        iterations=2,
        include_monitoring=False,
    )

    print(f"[backend={backend_kind}]")
    print(format_profile_report(report))
    print(json.dumps(report, indent=2, sort_keys=True))

    baseline = report["scenarios"]["baseline"]
    optimized = report["scenarios"]["optimized"]
    baseline_ms = float(baseline["scenario_total_ms"])
    optimized_ms = float(optimized["scenario_total_ms"])
    speedup = baseline_ms / optimized_ms if optimized_ms > 0 else float("inf")

    print(
        f"[backend={backend_kind}] Simple resolver workflow speedup: {speedup:.2f}x "
        f"(baseline={baseline_ms:.3f} ms, optimized={optimized_ms:.3f} ms)"
    )

    assert baseline_ms > 0
    assert optimized_ms > 0
    assert float(baseline["seed_total_ms"]) >= 0
    assert float(optimized["seed_total_ms"]) >= 0
    assert baseline["method_timings"]
    assert optimized["method_timings"]
