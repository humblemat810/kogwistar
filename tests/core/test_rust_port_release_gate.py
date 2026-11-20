from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.ci


def _gate():
    path = ROOT / "scripts" / "rust_port_release_gate.py"
    spec = importlib.util.spec_from_file_location("rust_port_release_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _compatibility(identity: str, source_sha256: str | None = None) -> list[dict]:
    source_sha256 = source_sha256 or _gate().candidate_source_fingerprint(ROOT)[0]
    return [
        {
            "candidate": {
                "identity_sha256": identity,
                "candidate_source_sha256": source_sha256,
                "backend": "docker-clean-dual-venv-sharded-pgvector",
                "storage_root": "container-local-isolated",
            },
            "status": "passed",
            "layers": [
                {
                    "name": layer,
                    "status": "passed",
                    "returncode": 0,
                    "target_group_count": 1,
                    "selected_group_indexes": [0],
                    "group_assignments": [0],
                    "commands": [["pytest", f"tests/{layer}"]],
                    "runs": [
                        {
                            "group_index": 0,
                            "group_key": f"{layer}/test_contract.py",
                            "command": ["pytest", f"tests/{layer}"],
                            "returncode": 0,
                        }
                    ],
                }
            ],
        }
        for layer in ("core", "parser", "sink", "application")
    ]


def _evaluate_compatibility(reports: list[dict]) -> dict:
    manifest = json.loads((ROOT / "contracts" / "rust-port-v1.json").read_text(encoding="utf-8"))
    return _gate().evaluate_release_gate(
        manifest=manifest,
        compatibility_reports=reports,
        runtime_performance={"gate": {"status": "blocked"}},
        phase3_capability_evidence={},
        phase4_runtime_evidence={},
        phase5_server_evidence={},
    )


def _runtime_performance() -> dict:
    path = ROOT / "contracts" / "benchmarks" / "rust-runtime-serving-current-windows.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _persistent_v4_report() -> dict:
    report = json.loads(
        (ROOT / "contracts" / "benchmarks" / "rust-persistent-sqlite-current-linux.json")
        .read_text(encoding="utf-8")
    )
    report["benchmark_version"] = 4
    report["items"] = 100
    report["read_cycles"] = 20
    report["samples_per_mode"] = 5
    workloads = _gate().PERSISTENT_PERFORMANCE_WORKLOADS
    expected_counts = {
        "event_append": 500,
        "event_replay": 100,
        "projection_replace": 500,
        "projection_list": 100,
        "mixed_interleaved": 1500,
    }
    for mode_name in ("python", "rust"):
        mode = report["modes"][mode_name]
        for workload in workloads:
            metric = mode["metrics"].setdefault(workload, {})
            metric["median_throughput_ops_per_sec"] = 100.0
            metric["operation_latency_ms"] = {
                "operation_count": expected_counts[workload],
                "p50_ms": 1.0,
                "p95_ms": 1.0,
                "p99_ms": 1.0,
            }
    report["ratios"] = {
        "rust": {
            workload: {
                "p95_latency_over_python": 1.0,
                "throughput_over_python": 1.0,
            }
            for workload in workloads
        }
    }
    report["gate"]["status"] = "passed"
    return report


def _persistent_v5_report() -> dict:
    report = _persistent_v4_report()
    report["benchmark_version"] = 5
    report["performance_gate_mode"] = "rust_native"
    report["modes"]["rust_native"] = copy.deepcopy(report["modes"]["rust"])
    report["ratios"]["rust_native"] = report["ratios"].pop("rust")
    report["peak_rss_rust_native_over_python"] = report.pop(
        "peak_rss_rust_over_python"
    )
    report["gate"]["mode"] = "rust_native"
    return report


def test_release_gate_rejects_duplicate_layer_report() -> None:
    reports = _compatibility("a" * 64)
    reports[0]["layers"].append(dict(reports[0]["layers"][0]))

    result = _evaluate_compatibility(reports)

    assert "core layer appears more than once" in {
        row["detail"] for row in result["blockers"]
    }


def test_release_gate_rejects_status_only_performance_report() -> None:
    gate = _gate()
    manifest = json.loads((ROOT / "contracts" / "rust-port-v1.json").read_text())

    result = gate.evaluate_release_gate(
        manifest=manifest,
        compatibility_reports=_compatibility("a" * 64),
        runtime_performance={"gate": {"status": "passed"}},
        phase3_capability_evidence={},
        phase4_runtime_evidence={},
        phase5_server_evidence={},
    )

    performance_details = {
        row["detail"] for row in result["blockers"] if row["gate"] == "performance"
    }
    assert "runtime serving scale benchmark checks are incomplete" in performance_details


@pytest.mark.parametrize("ratio", [0.0, -0.01, True, float("nan"), float("inf")])
def test_runtime_performance_gate_rejects_non_positive_or_non_finite_ratios(
    ratio: float,
) -> None:
    gate = _gate()
    report = _runtime_performance()
    metric = next(iter(report["ratios"]))
    report["ratios"][metric] = ratio
    report["gate"]["checks"][metric] = True

    errors = gate._validate_runtime_performance(report)

    assert f"runtime serving scale benchmark check failed: {metric}" in errors


def test_runtime_performance_gate_accepts_positive_ratio_at_threshold() -> None:
    gate = _gate()
    report = _runtime_performance()
    metric = next(iter(report["ratios"]))
    report["ratios"][metric] = report["gate"]["max_growth_ratio"]
    report["gate"]["checks"][metric] = True

    assert gate._validate_runtime_performance(report) == []


@pytest.mark.parametrize(
    ("latency_ratio", "throughput_ratio", "rss_ratio", "expected_check"),
    [
        (0.0, 1.0, 1.0, "p95_latency_no_worse_than_110_percent"),
        (-0.1, 1.0, 1.0, "p95_latency_no_worse_than_110_percent"),
        (True, 1.0, 1.0, "p95_latency_no_worse_than_110_percent"),
        (float("nan"), 1.0, 1.0, "p95_latency_no_worse_than_110_percent"),
        (1.0, True, 1.0, "throughput_at_least_95_percent"),
        (1.0, float("inf"), 1.0, "throughput_at_least_95_percent"),
        (1.0, 1.0, 0.0, "peak_rss_no_worse_than_110_percent"),
        (1.0, 1.0, -0.1, "peak_rss_no_worse_than_110_percent"),
        (1.0, 1.0, True, "peak_rss_no_worse_than_110_percent"),
        (1.0, 1.0, float("nan"), "peak_rss_no_worse_than_110_percent"),
    ],
)
def test_persistent_performance_gate_rejects_invalid_ratios(
    latency_ratio: float,
    throughput_ratio: float,
    rss_ratio: float,
    expected_check: str,
) -> None:
    benchmark = importlib.util.spec_from_file_location(
        "rust_port_persistent_benchmark",
        ROOT / "scripts" / "rust_port_persistent_benchmark.py",
    )
    assert benchmark is not None and benchmark.loader is not None
    module = importlib.util.module_from_spec(benchmark)
    benchmark.loader.exec_module(module)

    result = module.evaluate_persistent_performance_gate(
        {
            "rust": {
                "mixed_interleaved": {
                    "p95_latency_over_python": latency_ratio,
                    "throughput_over_python": throughput_ratio,
                }
            }
        },
        rss_ratio,
    )

    assert result["status"] == "failed"
    assert result["checks"][expected_check] is False


@pytest.mark.parametrize(
    ("metric", "expected_error"),
    [
        ("latency", "persistent-store operation measurements are invalid: event_append"),
        ("rss", "persistent-store peak RSS threshold failed"),
    ],
)
def test_persistent_release_validator_rejects_zero_measurements_even_if_marked_passed(
    metric: str, expected_error: str
) -> None:
    gate = _gate()
    report = _persistent_v5_report()
    if metric == "latency":
        report["modes"]["rust_native"]["metrics"]["event_append"][
            "operation_latency_ms"
        ]["p95_ms"] = 0
    else:
        report["modes"]["rust_native"]["p95_peak_rss_bytes"] *= 2
        report["peak_rss_rust_native_over_python"] *= 2

    assert expected_error in gate._validate_persistent_performance(report)


def test_persistent_performance_validator_recomputes_numeric_thresholds() -> None:
    gate = _gate()
    report = _persistent_v5_report()
    assert gate._validate_persistent_performance(report) == []

    report["modes"]["rust_native"]["metrics"]["event_append"][
        "operation_latency_ms"
    ]["p95_ms"] = 99.0
    report["ratios"]["rust_native"]["event_append"][
        "p95_latency_over_python"
    ] = 99.0
    report["gate"]["status"] = "passed"
    report["gate"]["checks"]["p95_latency_no_worse_than_110_percent"] = True
    errors = gate._validate_persistent_performance(report)

    assert "persistent-store latency threshold failed: event_append" in errors


def test_legacy_hybrid_report_is_diagnostic_not_a_performance_gate() -> None:
    gate = _gate()
    report = _persistent_v4_report()
    assert gate._validate_persistent_performance(report) == []

    report["modes"]["rust"]["metrics"]["event_append"]["operation_latency_ms"][
        "p95_ms"
    ] = 2.0
    report["gate"]["status"] = "failed"
    assert gate._validate_persistent_performance(report) == []


def test_legacy_hybrid_threshold_regression_does_not_block_release_gate() -> None:
    gate = _gate()
    report = _persistent_v4_report()
    report["modes"]["rust"]["metrics"]["event_append"]["operation_latency_ms"][
        "p95_ms"
    ] = 1.2
    report["ratios"]["rust"]["event_append"]["p95_latency_over_python"] = 1.2
    report["gate"]["checks"]["p95_latency_no_worse_than_110_percent"] = True

    assert gate._validate_persistent_performance(report) == []


def test_legacy_hybrid_report_is_not_reinterpreted_as_gate_evidence() -> None:
    gate = _gate()
    report = _persistent_v4_report()
    report["modes"]["rust"]["metrics"]["projection_list"]["operation_latency_ms"][
        "operation_count"
    ] -= 1

    assert gate._validate_persistent_performance(report) == []


def test_persistent_v5_release_gate_uses_pure_rust_native_mode() -> None:
    gate = _gate()
    report = _persistent_v5_report()

    assert gate._validate_persistent_performance(report) == []

    report["modes"]["rust"]["metrics"]["event_append"]["operation_latency_ms"][
        "p95_ms"
    ] = 99.0
    assert gate._validate_persistent_performance(report) == []

    report["modes"]["rust_native"]["metrics"]["event_append"][
        "operation_latency_ms"
    ]["p95_ms"] = 99.0
    report["ratios"]["rust_native"]["event_append"][
        "p95_latency_over_python"
    ] = 99.0
    assert "persistent-store latency threshold failed: event_append" in (
        gate._validate_persistent_performance(report)
    )


def test_persistent_v5_release_gate_rejects_hybrid_performance_target() -> None:
    gate = _gate()
    report = _persistent_v5_report()
    report["performance_gate_mode"] = "rust"

    assert gate._validate_persistent_performance(report) == [
        "persistent-store performance gate must target pure Rust-native mode"
    ]


def test_release_gate_rejects_missing_test_group_run() -> None:
    reports = _compatibility("a" * 64)
    reports[0]["layers"][0]["runs"] = []

    result = _evaluate_compatibility(reports)

    assert "core layer test-group runs are missing or duplicated" in {
        row["detail"] for row in result["blockers"]
    }


def test_release_gate_accepts_milestone_group_with_no_selected_tests() -> None:
    reports = _compatibility("a" * 64)
    layer = reports[0]["layers"][0]
    layer["test_profile"] = "milestone"
    layer["runs"][0]["returncode"] = 5

    result = _evaluate_compatibility(reports)

    assert "core layer contains an unverified test-group run" not in {
        row["detail"] for row in result["blockers"]
    }


def test_release_gate_rejects_empty_selection_outside_supported_profile() -> None:
    reports = _compatibility("a" * 64)
    reports[0]["layers"][0]["runs"][0]["returncode"] = 5

    result = _evaluate_compatibility(reports)

    assert "core layer contains an unverified test-group run" in {
        row["detail"] for row in result["blockers"]
    }


def _clause_test_references() -> list[dict[str, str]]:
    oracle_map = json.loads(
        (ROOT / "contracts/adr015-capability-test-oracles-v1.json").read_text(
            encoding="utf-8"
        )
    )["capabilities"]
    references: set[str] = set()
    for clauses in oracle_map.values():
        for selectors in clauses.values():
            for selector in selectors:
                references.add(selector)
    results = []
    for selector in sorted(references):
        source, test_path = selector.split(":", 1)
        relative_path, function = test_path.rsplit("::", 1)
        result = {"source": source, "function": function, "status": "passed"}
        if source == "pytest":
            result["classname"] = relative_path.removesuffix(".py").replace("/", ".")
            result["name"] = function
        else:
            module_path = relative_path.split("/src/", 1)[-1].removesuffix(".rs")
            result["name"] = "::".join((*module_path.split("/"), function))
        results.append(result)
    return results


def _phase5_evidence(source_sha256: str | None = None) -> dict:
    source_sha256 = source_sha256 or _gate().candidate_source_fingerprint(ROOT)[0]
    groups = []
    for name in sorted(_gate().REQUIRED_PHASE5_GROUPS):
        group = {
            "name": name,
            "status": "passed",
            "returncode": 0,
            "live_async_sse_required": name == "async-sse",
            "counts": {"tests": 1, "passed": 1, "failures": 0, "errors": 0, "skipped": 0},
            "test_results": [
                {
                    "source": "pytest",
                    "classname": "tests.fixture",
                    "name": "fixture_test",
                    "function": "fixture_test",
                    "status": "passed",
                }
            ],
        }
        if name == "rust-api":
            references = _clause_test_references()
            group["excluded_test_filters"] = sorted(_gate().RUST_API_EXCLUDED_TESTS)
            group["required_services"] = ["PostgreSQL", "Chroma"]
            group["test_profile"] = "debug-ci"
            group["counts"].update(
                {"tests": len(references), "passed": len(references), "filtered_out": 1}
            )
            group["test_results"] = references
        groups.append(group)
    return {
        "schema": "adr015-phase5-server-gate/v4",
        "candidate_source_sha256": source_sha256,
        "server_authority": "python-baseline-before-rust-cutover",
        "production_rolling_upgrade": "not-proven-local",
        "groups": groups,
        "status": "passed",
    }


def _phase3_capability_evidence(source_sha256: str | None = None) -> dict:
    source_sha256 = source_sha256 or _gate().candidate_source_fingerprint(ROOT)[0]
    groups = []
    for name in sorted(_gate().REQUIRED_PHASE3_CAPABILITY_GROUPS):
        groups.append(
            {
                "name": name,
                "status": "passed",
                "returncode": 0,
                "live_postgres_required": True,
                "counts": {"tests": 1, "passed": 1, "failures": 0, "errors": 0, "skipped": 0},
                "test_results": [
                    {
                        "source": "pytest",
                        "classname": "tests.fixture",
                        "name": "fixture_test",
                        "function": "fixture_test",
                        "status": "passed",
                    }
                ],
            }
        )
    return {
        "schema": "adr015-phase3-capability-gate/v2",
        "candidate_source_sha256": source_sha256,
        "production_authority": "not-promoted-local-evidence-only",
        "groups": groups,
        "status": "passed",
    }


def _phase4_runtime_evidence(source_sha256: str | None = None) -> dict:
    source_sha256 = source_sha256 or _gate().candidate_source_fingerprint(ROOT)[0]
    groups = []
    for name in sorted(_gate().REQUIRED_PHASE4_RUNTIME_GROUPS):
        groups.append(
            {
                "name": name,
                "status": "passed",
                "returncode": 0,
                "live_postgres_required": name == "postgres",
                "counts": {"tests": 1, "passed": 1, "failures": 0, "errors": 0, "skipped": 0},
                "test_results": [
                    {
                        "source": "pytest",
                        "classname": "tests.fixture",
                        "name": "fixture_test",
                        "function": "fixture_test",
                        "status": "passed",
                    }
                ],
            }
        )
    return {
        "schema": "adr015-phase4-runtime-gate/v2",
        "candidate_source_sha256": source_sha256,
        "async_rust_authority": "async-v2-worker-protocol",
        "groups": groups,
        "status": "passed",
    }


def test_capability_clause_results_follow_named_test_outcomes() -> None:
    gate = _gate()
    phase3 = _phase3_capability_evidence()
    phase4 = _phase4_runtime_evidence()
    phase5 = _phase5_evidence()

    results, errors = gate.evaluate_capability_clause_tests(
        phase3=phase3,
        phase4=phase4,
        phase5=phase5,
        current_source_sha256=gate.candidate_source_fingerprint(ROOT)[0],
    )
    assert len(results) == 22
    assert all(row["status"] == "passed" for row in results)
    assert errors == []

    rust_group = next(group for group in phase5["groups"] if group["name"] == "rust-api")
    rust_group["test_results"][0]["status"] = "failed"
    results, errors = gate.evaluate_capability_clause_tests(
        phase3=phase3,
        phase4=phase4,
        phase5=phase5,
        current_source_sha256=gate.candidate_source_fingerprint(ROOT)[0],
    )
    assert any(row["status"] == "blocked" for row in results)
    assert errors


@pytest.mark.parametrize("source", ["pytest", "rust"])
def test_capability_oracle_requires_exact_test_module_and_function(source: str) -> None:
    gate = _gate()
    phase3 = _phase3_capability_evidence()
    phase4 = _phase4_runtime_evidence()
    phase5 = _phase5_evidence()
    rust_group = next(group for group in phase5["groups"] if group["name"] == "rust-api")
    target = next(
        result for result in rust_group["test_results"] if result["source"] == source
    )
    if source == "pytest":
        target["classname"] = "wrong.module"
    else:
        target["function"] = "wrong_function"

    results, errors = gate.evaluate_capability_clause_tests(
        phase3=phase3,
        phase4=phase4,
        phase5=phase5,
        current_source_sha256=gate.candidate_source_fingerprint(ROOT)[0],
    )

    assert any(row["status"] == "blocked" for row in results)
    assert errors


def test_capability_test_oracle_map_covers_normative_clauses_and_existing_tests() -> None:
    gate = _gate()
    contract = json.loads(
        (ROOT / "contracts/dual-language-capabilities-v1.json").read_text(encoding="utf-8")
    )
    oracles = json.loads(
        (ROOT / "contracts/adr015-capability-test-oracles-v1.json").read_text(
            encoding="utf-8"
        )
    )
    relevant = gate.REQUIRED_PHASE3_CAPABILITY_GROUPS | {
        "workflow-runtime",
        "server-rest-sse-mcp-cli",
    }
    expected = {
        (capability, clause)
        for capability, clauses in contract["conformance_profiles"].items()
        if capability in relevant
        for clause in clauses
    }
    actual = {
        (capability, clause)
        for capability, clauses in oracles["capabilities"].items()
        for clause in clauses
    }

    assert actual == expected
    results, errors = gate.evaluate_capability_clause_tests(
        phase3=_phase3_capability_evidence(),
        phase4=_phase4_runtime_evidence(),
        phase5=_phase5_evidence(),
        current_source_sha256=gate.candidate_source_fingerprint(ROOT)[0],
    )
    assert len(results) == len(expected) == 22
    assert all(row["tests"] for row in results)
    assert errors == []


def test_capability_readiness_is_derived_from_test_evidence_not_manifest_flags() -> None:
    gate = _gate()
    results, errors = gate.evaluate_capability_clause_tests(
        phase3=_phase3_capability_evidence(),
        phase4=_phase4_runtime_evidence(),
        phase5=_phase5_evidence(),
        current_source_sha256=gate.candidate_source_fingerprint(ROOT)[0],
    )
    assert errors == []

    ready = gate.derive_capability_test_readiness(
        clause_results=results,
        phase_statuses={"phase3": "passed", "phase4": "passed", "phase5": "passed"},
        library_uat_passed=True,
        deferred_capabilities=set(),
    )

    assert len(ready) == 6
    assert all(item["status"] == "passed" for item in ready)
    assert all(item["cutover_readiness"] == "ready" for item in ready)
    results[0]["status"] = "blocked"
    blocked = gate.derive_capability_test_readiness(
        clause_results=results,
        phase_statuses={"phase3": "passed", "phase4": "passed", "phase5": "passed"},
        library_uat_passed=True,
        deferred_capabilities=set(),
    )
    assert blocked[0]["status"] == "blocked"


def test_capability_test_gate_pass_is_not_hidden_by_shared_cutover_blockers() -> None:
    gate = _gate()
    results, errors = gate.evaluate_capability_clause_tests(
        phase3=_phase3_capability_evidence(),
        phase4=_phase4_runtime_evidence(),
        phase5=_phase5_evidence(),
        current_source_sha256=gate.candidate_source_fingerprint(ROOT)[0],
    )
    assert errors == []

    reports = gate.derive_capability_test_readiness(
        clause_results=results,
        phase_statuses={"phase3": "blocked", "phase4": "blocked", "phase5": "blocked"},
        library_uat_passed=False,
        deferred_capabilities={"postgres-sequence-event-log", "workflow-runtime"},
    )
    by_capability = {item["capability"]: item for item in reports}

    postgres = by_capability["postgres-sequence-event-log"]
    assert postgres["status"] == "passed"
    assert postgres["passed_clauses"] == postgres["required_clauses"]
    assert postgres["blockers"] == []
    assert postgres["cutover_readiness"] == "blocked"
    assert "phase3 evidence blocked" in postgres["cutover_blockers"]
    assert "same-candidate Docker library UAT not passed" in postgres["cutover_blockers"]
    assert "required operations remain intentionally deferred" in postgres[
        "cutover_blockers"
    ]

    workflow = by_capability["workflow-runtime"]
    assert workflow["status"] == "passed"
    assert workflow["cutover_readiness"] == "blocked"


def test_capability_specific_current_report_closes_named_test_gate() -> None:
    gate = _gate()
    current_source = gate.candidate_source_fingerprint(ROOT)[0]
    phase3 = _phase3_capability_evidence()
    phase4 = _phase4_runtime_evidence()
    phase5 = _phase5_evidence()
    selectors_by_clause = json.loads(
        (ROOT / "contracts/adr015-capability-test-oracles-v1.json").read_text()
    )["capabilities"]["projections-snapshots-run-registry"]
    target_names = {
        selector.rsplit("::", 1)[1]
        for selectors in selectors_by_clause.values()
        for selector in selectors
    }
    for report in (phase3, phase4, phase5):
        for group in report["groups"]:
            group["test_results"] = [
                result
                for result in group["test_results"]
                if result.get("function") not in target_names
            ]

    test_results = []
    for clause_selectors in selectors_by_clause.values():
        for selector in clause_selectors:
            _, location = selector.split(":", 1)
            test_path, function = location.rsplit("::", 1)
            test_results.append(
                {
                    "source": "pytest",
                    "classname": test_path.removesuffix(".py").replace("/", "."),
                    "function": function,
                    "status": "passed",
                }
            )
    supplemental = {
        "schema": "adr015-phase3-capability-gate/v2",
        "candidate_source_sha256": current_source,
        "status": "passed",
        "groups": [
            {
                "name": "projections-snapshots-run-registry",
                "status": "passed",
                "counts": {
                    "tests": len(test_results),
                    "passed": len(test_results),
                    "failures": 0,
                    "errors": 0,
                    "skipped": 0,
                },
                "test_results": test_results,
            }
        ],
    }

    results, errors = gate.evaluate_capability_clause_tests(
        phase3=phase3,
        phase4=phase4,
        phase5=phase5,
        current_source_sha256=current_source,
        capability_test_evidence=[supplemental],
    )

    projection = [
        row for row in results
        if row["capability"] == "projections-snapshots-run-registry"
    ]
    assert len(projection) == 3
    assert all(row["status"] == "passed" for row in projection)
    assert errors == []

    manifest = json.loads(
        (ROOT / "contracts/rust-port-v1.json").read_text(encoding="utf-8")
    )
    release = gate.evaluate_release_gate(
        manifest=manifest,
        compatibility_reports=_compatibility("a" * 64, current_source),
        runtime_performance=_runtime_performance(),
        phase3_capability_evidence=phase3,
        phase4_runtime_evidence=phase4,
        phase5_server_evidence=phase5,
        capability_test_evidence=[supplemental],
    )
    release_by_capability = {
        item["capability"]: item for item in release["capability_test_readiness"]
    }
    assert release_by_capability["projections-snapshots-run-registry"]["status"] == "passed"

    supplemental["groups"][0]["counts"]["skipped"] = 1
    rejected, errors = gate.evaluate_capability_clause_tests(
        phase3=phase3,
        phase4=phase4,
        phase5=phase5,
        current_source_sha256=current_source,
        capability_test_evidence=[supplemental],
    )
    assert all(
        row["status"] == "blocked"
        for row in rejected
        if row["capability"] == "projections-snapshots-run-registry"
    )
    assert "supplemental capability test report is stale, skipped, or failed" in errors


def test_capability_readiness_fails_closed_on_duplicate_or_missing_clause_results() -> None:
    gate = _gate()
    results, errors = gate.evaluate_capability_clause_tests(
        phase3=_phase3_capability_evidence(),
        phase4=_phase4_runtime_evidence(),
        phase5=_phase5_evidence(),
        current_source_sha256=gate.candidate_source_fingerprint(ROOT)[0],
    )
    assert errors == []

    duplicate = dict(results[0])
    results[0]["status"] = "blocked"
    results.append(duplicate)
    readiness = gate.derive_capability_test_readiness(
        clause_results=results,
        phase_statuses={"phase3": "passed", "phase4": "passed", "phase5": "passed"},
        library_uat_passed=True,
        deferred_capabilities=set(),
    )

    assert readiness[0]["status"] == "blocked"
    assert results[0]["clause"] in readiness[0]["blockers"][0]

    missing = gate.derive_capability_test_readiness(
        clause_results=[],
        phase_statuses={"phase3": "passed", "phase4": "passed", "phase5": "passed"},
        library_uat_passed=True,
        deferred_capabilities=set(),
    )
    assert all(item["status"] == "blocked" for item in missing)


def test_stale_phase_report_does_not_override_current_named_test_results() -> None:
    gate = _gate()
    phase3 = _phase3_capability_evidence()
    phase4 = _phase4_runtime_evidence()
    phase5 = _phase5_evidence()
    phase4["candidate_source_sha256"] = "0" * 64

    results, errors = gate.evaluate_capability_clause_tests(
        phase3=phase3,
        phase4=phase4,
        phase5=phase5,
        current_source_sha256=gate.candidate_source_fingerprint(ROOT)[0],
    )

    assert len(results) == 22
    # Each selector also appears in a current-candidate test report. The stale
    # aggregate phase report remains a separate global cutover blocker.
    assert all(row["status"] == "passed" for row in results)
    assert errors == []


def test_stale_phase3_report_does_not_override_current_named_test_results() -> None:
    gate = _gate()
    phase3 = _phase3_capability_evidence()
    phase4 = _phase4_runtime_evidence()
    phase5 = _phase5_evidence()
    phase3["candidate_source_sha256"] = "0" * 64

    results, errors = gate.evaluate_capability_clause_tests(
        phase3=phase3,
        phase4=phase4,
        phase5=phase5,
        current_source_sha256=gate.candidate_source_fingerprint(ROOT)[0],
    )

    assert all(row["status"] == "passed" for row in results)
    assert errors == []


def test_capability_test_gate_rejects_evidence_when_all_reports_are_stale() -> None:
    gate = _gate()
    phase3 = _phase3_capability_evidence()
    phase4 = _phase4_runtime_evidence()
    phase5 = _phase5_evidence()
    for report in (phase3, phase4, phase5):
        report["candidate_source_sha256"] = "0" * 64

    results, errors = gate.evaluate_capability_clause_tests(
        phase3=phase3,
        phase4=phase4,
        phase5=phase5,
        current_source_sha256=gate.candidate_source_fingerprint(ROOT)[0],
    )

    assert all(row["status"] == "blocked" for row in results)
    assert all(
        test["status"] == "stale" for row in results for test in row["tests"]
    )
    assert errors


def test_current_release_gate_reports_real_blockers() -> None:
    manifest = json.loads((ROOT / "contracts" / "rust-port-v1.json").read_text(encoding="utf-8"))
    performance = json.loads(
        (ROOT / "contracts" / "benchmarks" / "rust-runtime-serving-current-windows.json").read_text(
            encoding="utf-8"
        )
    )

    result = _gate().evaluate_release_gate(
        manifest=manifest,
        compatibility_reports=_compatibility("a" * 64),
        runtime_performance=performance,
        phase3_capability_evidence=_phase3_capability_evidence(),
        phase4_runtime_evidence=_phase4_runtime_evidence(),
        phase5_server_evidence=_phase5_evidence(),
    )

    assert result["status"] == "blocked"
    details = {row["detail"] for row in result["blockers"]}
    assert "generic persistent public-store performance gate remains failed" not in details
    readiness = {item["capability"]: item for item in result["capability_test_readiness"]}
    assert readiness["postgres-sequence-event-log"]["status"] == "passed"
    assert readiness["projections-snapshots-run-registry"]["status"] == "passed"
    assert readiness["queues-leases-lanes"]["status"] == "passed"
    assert readiness["graph-pgvector"]["status"] == "passed"
    assert readiness["workflow-runtime"]["status"] == "passed"
    assert readiness["server-rest-sse-mcp-cli"]["status"] == "passed"
    assert readiness["postgres-sequence-event-log"]["cutover_readiness"] == "ready"
    assert readiness["graph-pgvector"]["cutover_readiness"] == "blocked"
    assert readiness["workflow-runtime"]["cutover_readiness"] == "blocked"
    assert readiness["server-rest-sse-mcp-cli"]["cutover_readiness"] == "blocked"
    assert not any(row["gate"] == "capability-test-gate" for row in result["blockers"])
    assert any(
        row["gate"] == "capability-cutover-readiness" for row in result["blockers"]
    )
    assert not any(row["gate"] == "authority" for row in result["blockers"])
    assert result["library_uat"]["status"] == "passed"


def test_gate_does_not_treat_manifest_flags_as_test_evidence() -> None:
    manifest = json.loads((ROOT / "contracts" / "rust-port-v1.json").read_text(encoding="utf-8"))
    manifest["current_persistent_evidence"]["gate_status"] = "passed"

    result = _gate().evaluate_release_gate(
        manifest=manifest,
        compatibility_reports=_compatibility("a" * 64),
        runtime_performance=_runtime_performance(),
        phase3_capability_evidence=_phase3_capability_evidence(),
        phase4_runtime_evidence=_phase4_runtime_evidence(),
        phase5_server_evidence=_phase5_evidence(),
    )

    assert result["status"] == "blocked"
    assert result["server_operation_deferrals"] == {"status": "passed"}
    assert result["phase3_capability_evidence"] == {"status": "passed"}
    assert result["phase4_runtime_evidence"] == {"status": "passed"}
    assert result["phase5_server_evidence"] == {"status": "passed"}
    assert result["library_uat"]["status"] == "passed"
    readiness = {item["capability"]: item for item in result["capability_test_readiness"]}
    assert readiness["postgres-sequence-event-log"]["status"] == "passed"
    assert readiness["projections-snapshots-run-registry"]["status"] == "passed"
    assert readiness["queues-leases-lanes"]["status"] == "passed"
    assert readiness["graph-pgvector"]["status"] == "passed"
    assert readiness["workflow-runtime"]["status"] == "passed"
    assert readiness["server-rest-sse-mcp-cli"]["status"] == "passed"
    assert readiness["graph-pgvector"]["cutover_readiness"] == "blocked"
    assert readiness["workflow-runtime"]["cutover_readiness"] == "blocked"
    assert readiness["server-rest-sse-mcp-cli"]["cutover_readiness"] == "blocked"


def test_gate_accepts_one_candidate_report_containing_all_compatibility_layers() -> None:
    reports = _compatibility("a" * 64)
    combined = dict(reports[0])
    combined["layers"] = [layer for report in reports for layer in report["layers"]]

    result = _evaluate_compatibility([combined])

    assert result["library_uat"]["status"] == "passed"
    assert result["library_uat"]["reports"] == 1
    assert result["compatibility_layers"] == {
        "core": "passed",
        "parser": "passed",
        "sink": "passed",
        "application": "passed",
    }


def test_gate_requires_all_compatibility_layers_in_docker_uat() -> None:
    reports = _compatibility("a" * 64)
    reports[1]["candidate"]["backend"] = "local-python"
    reports.pop()

    result = _evaluate_compatibility(reports)

    details = {row["detail"] for row in result["blockers"]}
    assert result["library_uat"]["status"] == "blocked"
    assert "application layer lacks passing evidence" in details
    assert "candidate backend must be a supported isolated Docker UAT backend" in details


def test_gate_rejects_python_retirement_despite_full_rust_evidence(
    tmp_path: Path, monkeypatch,
) -> None:
    gate = _gate()
    manifest = json.loads((ROOT / "contracts" / "rust-port-v1.json").read_text(encoding="utf-8"))
    for item in manifest["capability_ownership"]:
        if item["capability"] in gate.REQUIRED_RUST_OWNERS:
            item["rust_cutover_ready"] = True
    manifest["current_persistent_evidence"]["gate_status"] = "passed"

    matrix = json.loads(
        (ROOT / "contracts" / "dual-language-support-matrix-v1.json").read_text(
            encoding="utf-8"
        )
    )
    matrix["policy"]["python_and_rust_are_maintained"] = False
    next(item for item in matrix["capabilities"] if item["id"] == "graph-pgvector")[
        "python"
    ]["status"] = "disabled"
    path = tmp_path / "dual-language-support-matrix-v1.json"
    path.write_text(json.dumps(matrix), encoding="utf-8")
    monkeypatch.setattr(gate, "SUPPORT_MATRIX_PATH", path)

    result = gate.evaluate_release_gate(
        manifest=manifest,
        compatibility_reports=_compatibility("a" * 64),
        runtime_performance=_runtime_performance(),
        phase3_capability_evidence=_phase3_capability_evidence(),
        phase4_runtime_evidence=_phase4_runtime_evidence(),
        phase5_server_evidence=_phase5_evidence(),
    )

    assert result["status"] == "blocked"
    assert result["dual_language_support"] == {"status": "blocked"}
    details = {row["detail"] for row in result["blockers"]}
    assert "dual-language policy must maintain both Python and Rust" in details
    assert "dual-language capability graph-pgvector does not retain Python" in details


def test_dual_language_support_rejects_missing_and_duplicate_capabilities() -> None:
    gate = _gate()
    matrix = json.loads(
        (ROOT / "contracts" / "dual-language-support-matrix-v1.json").read_text(
            encoding="utf-8"
        )
    )
    matrix["capabilities"] = [
        item for item in matrix["capabilities"] if item["id"] != "workflow-runtime"
    ]
    matrix["capabilities"].append(matrix["capabilities"][0])
    matrix["capabilities"][0]["python"]["status"] = []
    errors = gate.validate_dual_language_support(
        matrix, {"deterministic-contracts", "workflow-runtime"}
    )
    assert "dual-language capability workflow-runtime is missing" in errors
    assert "duplicate dual-language capability deterministic-contracts" in errors
    assert "dual-language capability deterministic-contracts does not retain Python" in errors


def test_gate_rejects_missing_or_duplicate_versioned_server_deferrals() -> None:
    manifest = json.loads((ROOT / "contracts" / "rust-port-v1.json").read_text(encoding="utf-8"))
    for item in manifest["capability_ownership"]:
        if item["capability"] in _gate().REQUIRED_RUST_OWNERS:
            item["rust_cutover_ready"] = True
    manifest["current_persistent_evidence"]["gate_status"] = "passed"
    manifest["server_operation_deferrals"]["syscall_group"]["operations"].append(
        manifest["server_operation_deferrals"]["syscall_group"]["operations"][0]
    )

    result = _gate().evaluate_release_gate(
        manifest=manifest,
        compatibility_reports=_compatibility("a" * 64),
        runtime_performance=_runtime_performance(),
        phase3_capability_evidence=_phase3_capability_evidence(),
        phase4_runtime_evidence=_phase4_runtime_evidence(),
        phase5_server_evidence=_phase5_evidence(),
    )

    assert result["status"] == "blocked"
    assert result["server_operation_deferrals"] == {"status": "blocked"}
    details = {row["detail"] for row in result["blockers"]}
    assert "syscall_group.operations must not contain duplicates" in details


def test_gate_rejects_historical_compatibility_report_after_source_changes() -> None:
    manifest = json.loads((ROOT / "contracts" / "rust-port-v1.json").read_text(encoding="utf-8"))
    result = _gate().evaluate_release_gate(
        manifest=manifest,
        compatibility_reports=_compatibility("a" * 64, "f" * 64),
        runtime_performance=_runtime_performance(),
        phase3_capability_evidence=_phase3_capability_evidence(),
        phase4_runtime_evidence=_phase4_runtime_evidence(),
        phase5_server_evidence=_phase5_evidence(),
    )

    details = {row["detail"] for row in result["blockers"]}
    assert "four layers lack current candidate source identity" in details


def test_gate_rejects_stale_or_skipped_phase5_async_sse_evidence() -> None:
    manifest = json.loads((ROOT / "contracts" / "rust-port-v1.json").read_text(encoding="utf-8"))
    evidence = _phase5_evidence("f" * 64)
    async_sse = next(group for group in evidence["groups"] if group["name"] == "async-sse")
    async_sse["counts"]["skipped"] = 1

    result = _gate().evaluate_release_gate(
        manifest=manifest,
        compatibility_reports=_compatibility("a" * 64),
        runtime_performance=_runtime_performance(),
        phase3_capability_evidence=_phase3_capability_evidence(),
        phase4_runtime_evidence=_phase4_runtime_evidence(),
        phase5_server_evidence=evidence,
    )

    details = {row["detail"] for row in result["blockers"]}
    assert "Phase-5 server evidence does not match current candidate source" in details
    assert "Phase-5 async-sse evidence must be live and unskipped" in details


def test_gate_rejects_rust_api_server_evidence_with_ignored_or_missing_tests() -> None:
    manifest = json.loads((ROOT / "contracts" / "rust-port-v1.json").read_text(encoding="utf-8"))
    evidence = _phase5_evidence()
    rust_api = next(group for group in evidence["groups"] if group["name"] == "rust-api")
    rust_api["counts"]["skipped"] = 1
    rust_api["required_services"] = ["PostgreSQL"]
    rust_api["test_profile"] = "release"

    result = _gate().evaluate_release_gate(
        manifest=manifest,
        compatibility_reports=_compatibility("a" * 64),
        runtime_performance=_runtime_performance(),
        phase3_capability_evidence=_phase3_capability_evidence(),
        phase4_runtime_evidence=_phase4_runtime_evidence(),
        phase5_server_evidence=evidence,
    )

    details = {row["detail"] for row in result["blockers"]}
    assert "Phase-5 Rust API group rust-api has failures or errors" in details
    assert "Phase-5 Rust API tests must use the CI debug profile" in details
    assert "Phase-5 Rust API evidence must require PostgreSQL and Chroma" in details
    assert (
        "Phase-5 Rust API evidence must pass all non-excluded tests with no ignored tests"
        in details
    )


def test_gate_rejects_stale_or_skipped_phase3_capability_evidence() -> None:
    manifest = json.loads((ROOT / "contracts" / "rust-port-v1.json").read_text(encoding="utf-8"))
    evidence = _phase3_capability_evidence("f" * 64)
    queue = next(group for group in evidence["groups"] if group["name"] == "queues-leases-lanes")
    queue["counts"]["skipped"] = 1

    result = _gate().evaluate_release_gate(
        manifest=manifest,
        compatibility_reports=_compatibility("a" * 64),
        runtime_performance=_runtime_performance(),
        phase3_capability_evidence=evidence,
        phase4_runtime_evidence=_phase4_runtime_evidence(),
        phase5_server_evidence=_phase5_evidence(),
    )

    details = {row["detail"] for row in result["blockers"]}
    assert "Phase-3 capability evidence does not match current candidate source" in details
    assert "Phase-3 capability group queues-leases-lanes has failures, errors, or skips" in details


def test_gate_rejects_stale_or_skipped_phase4_runtime_evidence() -> None:
    manifest = json.loads((ROOT / "contracts" / "rust-port-v1.json").read_text(encoding="utf-8"))
    evidence = _phase4_runtime_evidence("f" * 64)
    postgres = next(group for group in evidence["groups"] if group["name"] == "postgres")
    postgres["counts"]["skipped"] = 1

    result = _gate().evaluate_release_gate(
        manifest=manifest,
        compatibility_reports=_compatibility("a" * 64),
        runtime_performance=_runtime_performance(),
        phase3_capability_evidence=_phase3_capability_evidence(),
        phase4_runtime_evidence=evidence,
        phase5_server_evidence=_phase5_evidence(),
    )

    details = {row["detail"] for row in result["blockers"]}
    assert "Phase-4 runtime evidence does not match current candidate source" in details
    assert "Phase-4 runtime group postgres has failures, errors, or skips" in details
