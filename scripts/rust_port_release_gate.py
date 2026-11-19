"""Aggregate ADR-015 release evidence and report exact remaining blockers."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
from pathlib import Path
from typing import Any

try:
    # Script entry point imports its sibling without package qualification.
    from source_fingerprint import (
        candidate_source_fingerprint,  # type: ignore[import-not-found]
    )
except ModuleNotFoundError:  # imported as a repository module in unit tests
    from scripts.source_fingerprint import candidate_source_fingerprint


ROOT = Path(__file__).resolve().parents[1]
SUPPORT_MATRIX_PATH = ROOT / "contracts" / "dual-language-support-matrix-v1.json"
LAYERS = ("core", "parser", "sink", "application")
SERVER_DEFERRAL_ROUTE_COUNT_V1 = 0
SERVER_DEFERRAL_SYSCALL_COUNT_V1 = 4
REQUIRED_RUST_OWNERS = {
    "deterministic-contracts",
    "sqlite-meta",
    "postgres-sequence-event-log",
    "projections-snapshots-run-registry",
    "queues-leases-lanes",
    "graph-pgvector",
    "workflow-runtime",
    "server-rest-sse-mcp-cli",
}
REQUIRED_PHASE5_GROUPS = {
    "rest-sse",
    "auth-syscall",
    "mcp",
    "frozen-contracts",
    "async-sse",
    "rust-api",
}
CAPABILITY_WORKLIST_PATH = ROOT / "kogwistar" / "docs" / "ADR-015-capability-completion-worklist.md"
RUST_API_EXCLUDED_TESTS = {
    "runtime_serving_lookup_scale_probe",
    "postgres_queue_projects_to_chroma_without_cross_store_transaction",
    "python_mcp_sdk_receives_rust_server_initiated_notification",
}
REQUIRED_PHASE3_CAPABILITY_GROUPS = {
    "postgres-sequence-event-log",
    "projections-snapshots-run-registry",
    "queues-leases-lanes",
    "graph-pgvector",
}
REQUIRED_PHASE4_RUNTIME_GROUPS = {
    "durable-worker",
    "bijection",
    "bridge",
    "suspend-terminal",
    "postgres",
}
RUNTIME_PERFORMANCE_CHECKS = {
    "retry_median_large_over_small",
    "state_read_median_large_over_small",
}
LEGACY_RUNTIME_PERFORMANCE_CHECKS = {
    "retry_p95_large_over_small",
    "state_read_p95_large_over_small",
}
PERSISTENT_PERFORMANCE_CHECKS = {
    "p95_latency_no_worse_than_110_percent",
    "throughput_at_least_95_percent",
    "peak_rss_no_worse_than_110_percent",
}
PERSISTENT_PERFORMANCE_WORKLOADS = {
    "event_append",
    "event_replay",
    "projection_replace",
    "projection_list",
    "mixed_interleaved",
}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _validate_runtime_performance(report: dict[str, Any]) -> list[str]:
    gate = report.get("gate")
    if not isinstance(gate, dict):
        return ["runtime serving scale benchmark gate is missing"]
    if gate.get("status") != "passed":
        return ["runtime serving scale benchmark has not passed"]

    checks = gate.get("checks")
    threshold = gate.get("max_growth_ratio")
    if not isinstance(checks, dict):
        return ["runtime serving scale benchmark checks are incomplete"]
    ratios = gate.get("ratios")
    if isinstance(ratios, dict):
        expected_checks = RUNTIME_PERFORMANCE_CHECKS
    else:
        ratios = report.get("ratios")
        expected_checks = LEGACY_RUNTIME_PERFORMANCE_CHECKS
    if not isinstance(ratios, dict) or set(checks) != expected_checks or set(ratios) != expected_checks:
        return ["runtime serving scale benchmark ratios are incomplete"]
    if not _is_finite_number(threshold) or threshold <= 0:
        return ["runtime serving scale benchmark threshold is invalid"]

    errors = []
    for name in sorted(expected_checks):
        ratio = ratios[name]
        valid_ratio = _is_finite_number(ratio)
        if (
            checks[name] is not True
            or not valid_ratio
            or ratio <= 0
            or ratio > threshold
        ):
            errors.append(f"runtime serving scale benchmark check failed: {name}")
    return errors


def _validate_persistent_performance(report: dict[str, Any]) -> list[str]:
    version = report.get("benchmark_version", 1)
    if not isinstance(version, int) or isinstance(version, bool) or version not in {1, 3, 4}:
        return ["persistent-store benchmark version is unsupported"]
    gate = report.get("gate")
    if not isinstance(gate, dict) or gate.get("status") != "passed":
        return ["generic persistent public-store performance gate remains failed"]
    checks = gate.get("checks")
    if not isinstance(checks, dict) or set(checks) != PERSISTENT_PERFORMANCE_CHECKS:
        return ["persistent-store performance checks are incomplete"]

    ratios = report.get("ratios")
    rss_ratio = report.get("peak_rss_rust_over_python")
    if version >= 4:
        items = report.get("items")
        read_cycles = report.get("read_cycles")
        samples_per_mode = report.get("samples_per_mode")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in (items, read_cycles, samples_per_mode)
        ):
            return ["persistent-store workload dimensions are missing or invalid"]
        expected_operation_counts = {
            "event_append": items * samples_per_mode,
            "event_replay": read_cycles * samples_per_mode,
            "projection_replace": items * samples_per_mode,
            "projection_list": read_cycles * samples_per_mode,
            "mixed_interleaved": items * 3 * samples_per_mode,
        }
        modes = report.get("modes")
        if not isinstance(modes, dict):
            return ["persistent-store Python/Rust mode summaries are missing"]
        python_mode = modes.get("python")
        rust_mode = modes.get("rust")
        if not isinstance(python_mode, dict) or not isinstance(rust_mode, dict):
            return ["persistent-store Python/Rust mode summaries are missing"]
        python_metrics = python_mode.get("metrics")
        rust_metrics = rust_mode.get("metrics")
        if not isinstance(python_metrics, dict) or not isinstance(rust_metrics, dict):
            return ["persistent-store Python/Rust workload metrics are missing"]
        if set(python_metrics) != PERSISTENT_PERFORMANCE_WORKLOADS or set(
            rust_metrics
        ) != PERSISTENT_PERFORMANCE_WORKLOADS:
            return ["persistent-store Python/Rust workloads do not match"]
        derived_ratios: dict[str, dict[str, float]] = {}
        for workload in sorted(PERSISTENT_PERFORMANCE_WORKLOADS):
            py_values = python_metrics[workload]
            rust_values = rust_metrics[workload]
            py_latency = py_values.get("operation_latency_ms")
            rust_latency = rust_values.get("operation_latency_ms")
            if not isinstance(py_latency, dict) or not isinstance(rust_latency, dict):
                return [f"persistent-store operation latency metrics are missing: {workload}"]
            py_p95 = py_latency.get("p95_ms")
            rust_p95 = rust_latency.get("p95_ms")
            py_count = py_latency.get("operation_count")
            rust_count = rust_latency.get("operation_count")
            py_throughput = py_values.get("median_throughput_ops_per_sec")
            rust_throughput = rust_values.get("median_throughput_ops_per_sec")
            if (
                not _is_finite_number(py_p95)
                or py_p95 <= 0
                or not _is_finite_number(rust_p95)
                or rust_p95 <= 0
                or not isinstance(py_count, int)
                or isinstance(py_count, bool)
                or py_count <= 0
                or not isinstance(rust_count, int)
                or isinstance(rust_count, bool)
                or rust_count <= 0
                or py_count != rust_count
                or py_count != expected_operation_counts[workload]
                or not _is_finite_number(py_throughput)
                or py_throughput <= 0
                or not _is_finite_number(rust_throughput)
                or rust_throughput <= 0
            ):
                return [f"persistent-store operation measurements are invalid: {workload}"]
            derived_ratios[workload] = {
                "p95_latency_over_python": rust_p95 / py_p95,
                "throughput_over_python": rust_throughput / py_throughput,
            }
        recorded_rust_ratios = ratios.get("rust") if isinstance(ratios, dict) else None
        if not isinstance(recorded_rust_ratios, dict):
            return ["persistent-store Rust performance ratios are missing"]
        for workload, derived in derived_ratios.items():
            recorded = recorded_rust_ratios.get(workload)
            if not isinstance(recorded, dict) or any(
                not _is_finite_number(recorded.get(key))
                or not math.isclose(recorded[key], value, rel_tol=1e-9, abs_tol=1e-12)
                for key, value in derived.items()
            ):
                return [f"persistent-store performance ratios do not match measurements: {workload}"]
        rust_ratios = derived_ratios
    elif isinstance(ratios, dict) and isinstance(ratios.get("rust"), dict):
        rust_ratios = ratios["rust"]
    else:
        # Version-1 reports predate explicit ratio fields; derive them from the
        # recorded Python/Rust summaries instead of trusting their status flag.
        modes = report.get("modes")
        if not isinstance(modes, dict):
            return ["persistent-store Rust performance ratios are missing"]
        python_mode = modes.get("python")
        rust_mode = modes.get("rust")
        if not isinstance(python_mode, dict) or not isinstance(rust_mode, dict):
            return ["persistent-store Python/Rust mode summaries are missing"]
        if python_mode.get("state_digest") != rust_mode.get("state_digest"):
            return ["persistent-store Python/Rust state digests differ"]
        python_metrics = python_mode.get("metrics")
        rust_metrics = rust_mode.get("metrics")
        if not isinstance(python_metrics, dict) or not isinstance(rust_metrics, dict):
            return ["persistent-store Python/Rust workload metrics are missing"]
        workloads = set(python_metrics) & set(rust_metrics)
        if not workloads or workloads != set(python_metrics) or workloads != set(rust_metrics):
            return ["persistent-store Python/Rust workloads do not match"]
        rust_ratios = {}
        for workload in workloads:
            python_values = python_metrics[workload]
            rust_values = rust_metrics[workload]
            if not isinstance(python_values, dict) or not isinstance(rust_values, dict):
                return [f"persistent-store workload metrics are invalid: {workload}"]
            try:
                rust_ratios[workload] = {
                    "p95_latency_over_python": (
                        rust_values["p95_elapsed_ms"] / python_values["p95_elapsed_ms"]
                    ),
                    "throughput_over_python": (
                        rust_values["median_throughput_ops_per_sec"]
                        / python_values["median_throughput_ops_per_sec"]
                    ),
                }
            except (KeyError, TypeError, ZeroDivisionError):
                return [f"persistent-store workload metrics are incomplete: {workload}"]
        try:
            rss_ratio = rust_mode["p95_peak_rss_bytes"] / python_mode["p95_peak_rss_bytes"]
        except (KeyError, TypeError, ZeroDivisionError):
            return ["persistent-store peak RSS measurements are missing"]

    expected_workloads = (
        PERSISTENT_PERFORMANCE_WORKLOADS
        if report.get("benchmark_version", 1) >= 3
        else {"event_append", "event_replay", "projection_replace", "projection_list"}
    )
    if set(rust_ratios) != expected_workloads:
        return ["persistent-store performance workloads are incomplete"]

    errors = []
    for name in sorted(PERSISTENT_PERFORMANCE_CHECKS):
        if checks[name] is not True:
            errors.append(f"persistent-store performance check failed: {name}")
    for workload, values in rust_ratios.items():
        if not isinstance(values, dict):
            errors.append(f"persistent-store performance ratios are invalid: {workload}")
            continue
        latency = values.get("p95_latency_over_python")
        throughput = values.get("throughput_over_python")
        if not _is_finite_number(latency) or latency <= 0 or latency > 1.10:
            errors.append(f"persistent-store latency threshold failed: {workload}")
        if not _is_finite_number(throughput) or throughput < 0.95:
            errors.append(f"persistent-store throughput threshold failed: {workload}")
    if (
        not _is_finite_number(rss_ratio) or rss_ratio <= 0 or rss_ratio > 1.10
    ):
        errors.append("persistent-store peak RSS threshold failed")
    return errors


def _readiness_module():
    path = ROOT / "scripts" / "rust_port_readiness.py"
    spec = importlib.util.spec_from_file_location("rust_port_readiness", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_dual_language_support(
    matrix: dict[str, Any], required_capabilities: set[str]
) -> list[str]:
    """Keep Rust promotion from silently retiring the Python implementation."""
    errors: list[str] = []
    policy = matrix.get("policy")
    if not isinstance(policy, dict) or policy.get("python_and_rust_are_maintained") is not True:
        errors.append("dual-language policy must maintain both Python and Rust")
    if not isinstance(policy, dict) or policy.get("deprecation_requires_separate_adr") is not True:
        errors.append("dual-language deprecation requires a separate ADR")

    entries = matrix.get("capabilities")
    if not isinstance(entries, list):
        return [*errors, "dual-language capabilities must be an array"]
    by_id: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            errors.append("dual-language capability entries must have string IDs")
            continue
        capability = entry["id"]
        if capability in by_id:
            errors.append(f"duplicate dual-language capability {capability}")
        by_id[capability] = entry
    for capability, entry in by_id.items():
        python = entry.get("python")
        status = python.get("status") if isinstance(python, dict) else None
        if not isinstance(status, str) or status not in {"enabled", "opt-in"}:
            errors.append(f"dual-language capability {capability} does not retain Python")
    for capability in sorted(required_capabilities):
        if capability not in by_id:
            errors.append(f"dual-language capability {capability} is missing")
    return errors


def validate_server_operation_deferrals(manifest: dict[str, Any]) -> list[str]:
    """Validate Phase-5 Python rollback deferrals before release promotion.

    Rust's frozen inventories are checked against this manifest in the contract
    suite. This release-side check independently prevents a malformed or
    incomplete manifest from satisfying a readiness report merely because all
    capability flags and canaries were changed to green.
    """
    errors: list[str] = []
    deferrals = manifest.get("server_operation_deferrals")
    if not isinstance(deferrals, dict):
        return ["server_operation_deferrals is required"]
    if deferrals.get("contract_version") != 1:
        errors.append("contract_version must be 1")
    if deferrals.get("status") != "versioned-intentionally-deferred":
        errors.append("status must be versioned-intentionally-deferred")
    if deferrals.get("owner") != "python-rollback-deployment":
        errors.append("owner must be python-rollback-deployment")
    if not isinstance(deferrals.get("rule"), str) or not deferrals["rule"].strip():
        errors.append("rule must be non-empty")

    routes: set[tuple[str, str]] = set()
    route_groups = deferrals.get("route_groups")
    if not isinstance(route_groups, list):
        errors.append("route_groups must be an array")
        route_groups = []
    for index, group in enumerate(route_groups):
        prefix = f"route_groups[{index}]"
        if not isinstance(group, dict):
            errors.append(f"{prefix} must be an object")
            continue
        _validate_deferred_group(group, prefix, errors)
        operations = group.get("operations")
        if not isinstance(operations, list) or not operations:
            errors.append(f"{prefix}.operations must be a non-empty array")
            continue
        for operation_index, operation in enumerate(operations):
            operation_prefix = f"{prefix}.operations[{operation_index}]"
            if not isinstance(operation, dict):
                errors.append(f"{operation_prefix} must be an object")
                continue
            method = operation.get("method")
            path = operation.get("path")
            if not isinstance(method, str) or not method:
                errors.append(f"{operation_prefix}.method must be non-empty")
                continue
            if not isinstance(path, str) or not path.startswith("/"):
                errors.append(f"{operation_prefix}.path must start with /")
                continue
            key = (method, path)
            if key in routes:
                errors.append(f"duplicate deferred route {method} {path}")
            routes.add(key)
    if len(routes) != SERVER_DEFERRAL_ROUTE_COUNT_V1:
        errors.append(
            f"route_groups must contain {SERVER_DEFERRAL_ROUTE_COUNT_V1} unique operations"
        )

    syscall_group = deferrals.get("syscall_group")
    if not isinstance(syscall_group, dict):
        errors.append("syscall_group must be an object")
        return errors
    _validate_deferred_group(syscall_group, "syscall_group", errors)
    syscalls = syscall_group.get("operations")
    if not isinstance(syscalls, list) or not all(
        isinstance(operation, str) and operation for operation in syscalls
    ):
        errors.append("syscall_group.operations must be non-empty strings")
    elif len(set(syscalls)) != len(syscalls):
        errors.append("syscall_group.operations must not contain duplicates")
    elif len(syscalls) != SERVER_DEFERRAL_SYSCALL_COUNT_V1:
        errors.append(
            f"syscall_group must contain {SERVER_DEFERRAL_SYSCALL_COUNT_V1} operations"
        )
    return errors


def validate_phase5_server_evidence(
    evidence: dict[str, Any], *, current_source_sha256: str
) -> list[str]:
    """Require current-source server UAT evidence without implying deployment."""
    errors: list[str] = []
    if evidence.get("schema") != "adr015-phase5-server-gate/v3":
        errors.append("Phase-5 server evidence has an unsupported schema")
    if evidence.get("candidate_source_sha256") != current_source_sha256:
        errors.append("Phase-5 server evidence does not match current candidate source")
    if evidence.get("status") != "passed":
        errors.append("Phase-5 server evidence has not passed")
    if evidence.get("server_authority") != "python-baseline-before-rust-cutover":
        errors.append("Phase-5 server evidence does not declare Python baseline authority")
    if evidence.get("production_rolling_upgrade") != "not-proven-local":
        errors.append("Phase-5 local evidence must not claim production rolling upgrade")
    groups = evidence.get("groups")
    if not isinstance(groups, list):
        return [*errors, "Phase-5 server evidence groups must be an array"]
    by_name = {
        item.get("name"): item
        for item in groups
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    if set(by_name) != REQUIRED_PHASE5_GROUPS:
        errors.append("Phase-5 server evidence must cover exactly the required groups")
    for name in REQUIRED_PHASE5_GROUPS:
        group = by_name.get(name)
        if not isinstance(group, dict):
            continue
        counts = group.get("counts")
        if group.get("status") != "passed" or group.get("returncode") != 0:
            errors.append(f"Phase-5 group {name} has not passed")
        if not isinstance(counts, dict) or counts.get("tests", 0) <= 0:
            errors.append(f"Phase-5 group {name} lacks test evidence")
        elif counts.get("failures") != 0 or counts.get("errors") != 0:
            errors.append(f"Phase-5 group {name} has failures or errors")
        if isinstance(counts, dict):
            errors.extend(
                _named_test_result_errors(
                    group, prefix=f"Phase-5 group {name}", counts=counts
                )
            )
        if name == "async-sse" and (
            group.get("live_async_sse_required") is not True
            or not isinstance(counts, dict)
            or counts.get("skipped") != 0
        ):
            errors.append("Phase-5 async-sse evidence must be live and unskipped")
        if name == "rust-api":
            if (
                not isinstance(counts, dict)
                or counts.get("failures") != 0
                or counts.get("errors") != 0
                or counts.get("skipped") != 0
            ):
                errors.append(
                    "Phase-5 Rust API group rust-api has failures or errors"
                )
            if group.get("excluded_test_filters") != sorted(RUST_API_EXCLUDED_TESTS):
                errors.append("Phase-5 Rust API evidence has unexpected excluded tests")
            if group.get("required_services") != ["PostgreSQL", "Chroma"]:
                errors.append("Phase-5 Rust API evidence must require PostgreSQL and Chroma")
            if (
                not isinstance(counts, dict)
                or counts.get("passed", 0) <= 0
                or counts.get("skipped") != 0
                or counts.get("filtered_out") != 1
            ):
                errors.append(
                    "Phase-5 Rust API evidence must pass all non-excluded tests with no ignored tests"
                )
    return errors


def _named_test_result_errors(
    group: dict[str, Any], *, prefix: str, counts: dict[str, Any]
) -> list[str]:
    results = group.get("test_results")
    if not isinstance(results, list) or len(results) != counts.get("tests"):
        return [f"{prefix} named test results do not match test count"]
    if any(
        not isinstance(result, dict)
        or result.get("status") not in {"passed", "failed", "skipped"}
        or not isinstance(result.get("function"), str)
        or not result["function"]
        for result in results
    ):
        return [f"{prefix} has malformed named test results"]
    passed = sum(result["status"] == "passed" for result in results)
    failed = sum(result["status"] == "failed" for result in results)
    skipped = sum(result["status"] == "skipped" for result in results)
    if (passed, failed, skipped) != (
        counts.get("passed"),
        counts.get("failures", 0) + counts.get("errors", 0),
        counts.get("skipped"),
    ):
        return [f"{prefix} named test outcomes disagree with aggregate counts"]
    return []


def evaluate_capability_clause_tests(
    *,
    phase3: dict[str, Any],
    phase4: dict[str, Any],
    phase5: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Evaluate every pending profile clause against named test outcomes."""
    contract = _load_json(ROOT / "contracts" / "dual-language-capabilities-v1.json")
    relevant_capabilities = (
        REQUIRED_PHASE3_CAPABILITY_GROUPS
        | {"workflow-runtime", "server-rest-sse-mcp-cli"}
    )
    expected = {
        (capability, clause)
        for capability, clauses in contract.get("conformance_profiles", {}).items()
        if capability in relevant_capabilities
        for clause in clauses
    }
    worklist = CAPABILITY_WORKLIST_PATH.read_text(encoding="utf-8")
    table = worklist.partition("## Capability Gate Closure")[2].partition("## Tasklets")[0]
    rows = [
        line.strip().strip("|").split("|")
        for line in table.splitlines()
        if line.startswith("| `")
    ]
    row_keys = {
        (cells[0].strip().strip("`"), cells[1].strip().strip("`"))
        for cells in rows
        if len(cells) >= 4
    }
    errors: list[str] = []
    if row_keys != expected or len(rows) != len(expected):
        errors.append("capability worklist clauses do not exactly match the normative profiles")

    python_tests = {
        name
        for path in (ROOT / "tests").rglob("*.py")
        for name in re.findall(
            r"(?m)^\s*(?:async\s+)?def\s+(test_[A-Za-z0-9_]+)\s*\(",
            path.read_text(encoding="utf-8"),
        )
    }
    python_results: dict[str, list[dict[str, Any]]] = {}
    rust_results: dict[str, list[dict[str, Any]]] = {}
    for report in (phase3, phase4, phase5):
        for group in report.get("groups", []):
            for result in group.get("test_results", []):
                if not isinstance(result, dict):
                    continue
                index = python_results if result.get("source") == "pytest" else rust_results
                function = result.get("function")
                if isinstance(function, str):
                    index.setdefault(function, []).append(result)

    clause_results: list[dict[str, Any]] = []
    for cells in rows:
        if len(cells) < 4:
            continue
        capability, clause, _documented_status, evidence = (
            cell.strip().strip("`") for cell in cells[:4]
        )
        python_names = sorted(set(re.findall(r"\btest_[A-Za-z0-9_]+\b", evidence)) & python_tests)
        rust_names = set(re.findall(r"\btests::([A-Za-z_][A-Za-z0-9_]*)", evidence))
        rust_names.update(
            name
            for _, name in re.findall(
                r"`(rust/[^`]+?\.rs)::([A-Za-z_][A-Za-z0-9_]*)`", evidence
            )
        )
        references = [("pytest", name) for name in python_names] + [
            ("rust", name) for name in sorted(rust_names)
        ]
        checks: list[dict[str, str]] = []
        for source, name in references:
            outcomes = (python_results if source == "pytest" else rust_results).get(name, [])
            status = "missing" if not outcomes else (
                "passed" if all(item.get("status") == "passed" for item in outcomes) else "failed"
            )
            checks.append({"source": source, "function": name, "status": status})
        passed = bool(checks) and all(check["status"] == "passed" for check in checks)
        clause_results.append(
            {
                "capability": capability,
                "clause": clause,
                "status": "passed" if passed else "blocked",
                "tests": checks,
            }
        )
        if not checks:
            errors.append(f"{capability}/{clause} has no executable test references")
        elif not passed:
            missing = [
                f"{check['source']}::{check['function']}={check['status']}"
                for check in checks
                if check["status"] != "passed"
            ]
            errors.append(f"{capability}/{clause} test evidence incomplete: {', '.join(missing)}")

    return clause_results, errors


def validate_phase3_capability_evidence(
    evidence: dict[str, Any], *, current_source_sha256: str
) -> list[str]:
    """Require separate live-PostgreSQL UAT proof per pending store capability."""
    errors: list[str] = []
    if evidence.get("schema") != "adr015-phase3-capability-gate/v2":
        errors.append("Phase-3 capability evidence has an unsupported schema")
    if evidence.get("candidate_source_sha256") != current_source_sha256:
        errors.append("Phase-3 capability evidence does not match current candidate source")
    if evidence.get("status") != "passed":
        errors.append("Phase-3 capability evidence has not passed")
    if evidence.get("production_authority") != "not-promoted-local-evidence-only":
        errors.append("Phase-3 local evidence must not claim production authority")
    groups = evidence.get("groups")
    if not isinstance(groups, list):
        return [*errors, "Phase-3 capability evidence groups must be an array"]
    by_name = {
        item.get("name"): item
        for item in groups
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    if set(by_name) != REQUIRED_PHASE3_CAPABILITY_GROUPS:
        errors.append("Phase-3 capability evidence must cover exactly the required groups")
    for name in REQUIRED_PHASE3_CAPABILITY_GROUPS:
        group = by_name.get(name)
        if not isinstance(group, dict):
            continue
        counts = group.get("counts")
        if group.get("status") != "passed" or group.get("returncode") != 0:
            errors.append(f"Phase-3 capability group {name} has not passed")
        if group.get("live_postgres_required") is not True:
            errors.append(f"Phase-3 capability group {name} must require live PostgreSQL")
        if not isinstance(counts, dict) or counts.get("tests", 0) <= 0:
            errors.append(f"Phase-3 capability group {name} lacks test evidence")
        elif (
            counts.get("failures") != 0
            or counts.get("errors") != 0
            or counts.get("skipped") != 0
        ):
            errors.append(
                f"Phase-3 capability group {name} has failures, errors, or skips"
            )
        if isinstance(counts, dict):
            errors.extend(
                _named_test_result_errors(
                    group, prefix=f"Phase-3 capability group {name}", counts=counts
                )
            )
    return errors


def validate_phase4_runtime_evidence(
    evidence: dict[str, Any], *, current_source_sha256: str
) -> list[str]:
    """Require current durable runtime proof before a runtime promotion.

    Four-layer compatibility normally runs with the public Python runtime owner.
    It therefore cannot replace the dedicated Rust reducer/worker/restart/live
    PostgreSQL evidence captured by the Phase-4 gate.
    """
    errors: list[str] = []
    if evidence.get("schema") != "adr015-phase4-runtime-gate/v2":
        errors.append("Phase-4 runtime evidence has an unsupported schema")
    if evidence.get("candidate_source_sha256") != current_source_sha256:
        errors.append("Phase-4 runtime evidence does not match current candidate source")
    if evidence.get("status") != "passed":
        errors.append("Phase-4 runtime evidence has not passed")
    if evidence.get("async_rust_authority") != "async-v2-worker-protocol":
        errors.append("Phase-4 runtime evidence has an unexpected async authority boundary")
    groups = evidence.get("groups")
    if not isinstance(groups, list):
        return [*errors, "Phase-4 runtime evidence groups must be an array"]
    by_name = {
        item.get("name"): item
        for item in groups
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    if set(by_name) != REQUIRED_PHASE4_RUNTIME_GROUPS:
        errors.append("Phase-4 runtime evidence must cover exactly the required groups")
    for name in REQUIRED_PHASE4_RUNTIME_GROUPS:
        group = by_name.get(name)
        if not isinstance(group, dict):
            continue
        counts = group.get("counts")
        if group.get("status") != "passed" or group.get("returncode") != 0:
            errors.append(f"Phase-4 runtime group {name} has not passed")
        if not isinstance(counts, dict) or counts.get("tests", 0) <= 0:
            errors.append(f"Phase-4 runtime group {name} lacks test evidence")
        elif (
            counts.get("failures") != 0
            or counts.get("errors") != 0
            or counts.get("skipped") != 0
        ):
            errors.append(f"Phase-4 runtime group {name} has failures, errors, or skips")
        if isinstance(counts, dict):
            errors.extend(
                _named_test_result_errors(
                    group, prefix=f"Phase-4 runtime group {name}", counts=counts
                )
            )
        if name == "postgres" and group.get("live_postgres_required") is not True:
            errors.append("Phase-4 runtime postgres group must require live PostgreSQL")
    return errors


def _validate_deferred_group(group: dict[str, Any], prefix: str, errors: list[str]) -> None:
    if group.get("owner") != "python":
        errors.append(f"{prefix}.owner must be python")
    if group.get("status") != "intentionally-deferred":
        errors.append(f"{prefix}.status must be intentionally-deferred")
    authority = group.get("required_authority")
    if not isinstance(authority, list) or not all(
        isinstance(capability, str) and capability for capability in authority
    ):
        errors.append(f"{prefix}.required_authority must be non-empty strings")
    if not isinstance(group.get("exit_evidence"), str) or not group["exit_evidence"].strip():
        errors.append(f"{prefix}.exit_evidence must be non-empty")


def _compatibility_layer_errors(layer: dict[str, Any]) -> list[str]:
    """Require a passing layer summary to reconcile to its exact test-group runs."""
    name = layer.get("name")
    prefix = f"{name} layer"
    errors: list[str] = []
    count = layer.get("target_group_count")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        return [f"{prefix} has invalid target group count"]
    assignments = layer.get("group_assignments")
    if not isinstance(assignments, list) or len(assignments) != count or any(
        not isinstance(index, int) or isinstance(index, bool) or index < 0
        for index in assignments
    ):
        errors.append(f"{prefix} has invalid group assignments")
    selected = layer.get("selected_group_indexes")
    expected = list(range(count))
    if selected != expected:
        errors.append(f"{prefix} selected groups are missing or duplicated")

    runs = layer.get("runs")
    if not isinstance(runs, list) or any(not isinstance(run, dict) for run in runs):
        errors.append(f"{prefix} test-group runs are invalid")
        return errors
    run_indexes = [run.get("group_index") for run in runs]
    if (
        any(not isinstance(index, int) or isinstance(index, bool) for index in run_indexes)
        or len(run_indexes) != count
        or sorted(run_indexes) != expected
    ):
        errors.append(f"{prefix} test-group runs are missing or duplicated")
    group_keys = [run.get("group_key") for run in runs]
    if any(not isinstance(key, str) or not key for key in group_keys) or len(
        set(group_keys)
    ) != count:
        errors.append(f"{prefix} test-group keys are missing or duplicated")
    profile = layer.get("test_profile")
    allows_empty_selection = profile in {"milestone", "regression"} or name == "parser"
    if any(
        (
            run.get("returncode") != 0
            and not (allows_empty_selection and run.get("returncode") == 5)
        )
        or not isinstance(run.get("command"), list)
        for run in runs
    ):
        errors.append(f"{prefix} contains an unverified test-group run")
    commands = layer.get("commands")
    if not isinstance(commands, list) or commands != [run.get("command") for run in runs]:
        errors.append(f"{prefix} commands do not match test-group runs")
    return errors


def evaluate_release_gate(
    *,
    manifest: dict[str, Any],
    compatibility_reports: list[dict[str, Any]],
    runtime_performance: dict[str, Any],
    phase3_capability_evidence: dict[str, Any],
    phase4_runtime_evidence: dict[str, Any],
    phase5_server_evidence: dict[str, Any],
) -> dict[str, Any]:
    blockers: list[dict[str, str]] = []
    by_layer: dict[str, dict[str, Any]] = {}
    layer_counts: dict[str, int] = {}
    identities: set[str] = set()
    source_identities: set[str] = set()
    for report in compatibility_reports:
        candidate = report.get("candidate")
        if isinstance(candidate, dict) and isinstance(candidate.get("identity_sha256"), str):
            identities.add(candidate["identity_sha256"])
        if isinstance(candidate, dict) and isinstance(
            candidate.get("candidate_source_sha256"), str
        ):
            source_identities.add(candidate["candidate_source_sha256"])
        layers = report.get("layers")
        if not isinstance(layers, list):
            blockers.append({"gate": "compatibility", "detail": "compatibility report layers are invalid"})
            continue
        for layer in layers:
            if isinstance(layer, dict) and layer.get("name") in LAYERS:
                name = str(layer["name"])
                layer_counts[name] = layer_counts.get(name, 0) + 1
                by_layer.setdefault(name, layer)
    for layer in LAYERS:
        evidence = by_layer.get(layer)
        if layer_counts.get(layer, 0) > 1:
            blockers.append(
                {"gate": "compatibility", "detail": f"{layer} layer appears more than once"}
            )
        if evidence is None or evidence.get("status") != "passed" or evidence.get("returncode") != 0:
            blockers.append(
                {"gate": "compatibility", "detail": f"{layer} layer lacks passing evidence"}
            )
        elif evidence is not None:
            for detail in _compatibility_layer_errors(evidence):
                blockers.append({"gate": "compatibility", "detail": detail})
    if len(identities) != 1:
        blockers.append(
            {"gate": "compatibility", "detail": "four layers do not share one candidate identity"}
        )
    current_source_sha256, _ = candidate_source_fingerprint(ROOT)
    if source_identities != {current_source_sha256}:
        blockers.append(
            {
                "gate": "compatibility",
                "detail": "four layers lack current candidate source identity",
            }
        )

    for detail in validate_phase3_capability_evidence(
        phase3_capability_evidence, current_source_sha256=current_source_sha256
    ):
        blockers.append({"gate": "store-local", "detail": detail})

    for detail in validate_phase4_runtime_evidence(
        phase4_runtime_evidence, current_source_sha256=current_source_sha256
    ):
        blockers.append({"gate": "runtime-local", "detail": detail})

    for detail in validate_phase5_server_evidence(
        phase5_server_evidence, current_source_sha256=current_source_sha256
    ):
        blockers.append({"gate": "server-local", "detail": detail})

    capability_clause_results, clause_errors = evaluate_capability_clause_tests(
        phase3=phase3_capability_evidence,
        phase4=phase4_runtime_evidence,
        phase5=phase5_server_evidence,
    )
    for detail in clause_errors:
        blockers.append({"gate": "capability-clauses", "detail": detail})

    runtime_performance_errors = _validate_runtime_performance(runtime_performance)
    for detail in runtime_performance_errors:
        blockers.append({"gate": "performance", "detail": detail})
    persistent = manifest.get("current_persistent_evidence", {})
    persistent_report_path = persistent.get("benchmark_report")
    persistent_report = None
    if isinstance(persistent_report_path, str):
        candidate_path = (ROOT / persistent_report_path).resolve()
        try:
            candidate_path.relative_to(ROOT.resolve())
            persistent_report = _load_json(candidate_path)
        except (OSError, ValueError):
            persistent_report = None
    if persistent.get("gate_status") != "passed" or persistent_report is None:
        blockers.append(
            {"gate": "performance", "detail": "persistent-store benchmark evidence is missing or failed"}
        )
    else:
        for detail in _validate_persistent_performance(persistent_report):
            blockers.append({"gate": "performance", "detail": detail})

    for detail in validate_server_operation_deferrals(manifest):
        blockers.append({"gate": "server-deferral", "detail": detail})

    ownership = manifest.get("capability_ownership", [])
    required_support = REQUIRED_RUST_OWNERS | {
        item["capability"]
        for item in ownership
        if isinstance(item, dict) and isinstance(item.get("capability"), str)
    }
    try:
        support_errors = validate_dual_language_support(
            _load_json(SUPPORT_MATRIX_PATH), required_support
        )
    except (OSError, ValueError):
        support_errors = ["dual-language support matrix is missing or invalid"]
    for detail in support_errors:
        blockers.append({"gate": "dual-language-support", "detail": detail})
    active_rust = {
        item.get("capability")
        for item in ownership
        if isinstance(item, dict) and item.get("rust_cutover_ready") is True
    }
    for capability in sorted(REQUIRED_RUST_OWNERS - active_rust):
        blockers.append(
            {"gate": "authority", "detail": f"{capability} is not Rust-cutover-ready"}
        )

    expected_identity = next(iter(identities)) if len(identities) == 1 else None
    uat_candidates = []
    for report in compatibility_reports:
        candidate = report.get("candidate")
        if isinstance(candidate, dict):
            uat_candidates.append(
                _readiness_module().validate_library_uat_candidate(
                    candidate,
                    expected_candidate_identity_sha256=expected_identity,
                )
            )
    if not uat_candidates:
        blockers.append(
            {"gate": "library-uat", "detail": "containerized library UAT candidate evidence is missing"}
        )
    for result in uat_candidates:
        for detail in result["errors"]:
            blockers.append({"gate": "library-uat", "detail": detail})
    uat_backends = {result["backend"] for result in uat_candidates}
    if len(uat_backends) > 1:
        blockers.append(
            {"gate": "library-uat", "detail": "compatibility layers use different Docker UAT backends"}
        )
    compatibility_complete = all(
        layer_counts.get(layer) == 1
        and by_layer[layer].get("status") == "passed"
        and by_layer[layer].get("returncode") == 0
        and not _compatibility_layer_errors(by_layer[layer])
        for layer in LAYERS
    )
    library_uat = {
        "status": (
            "passed"
            if compatibility_complete
            and uat_candidates
            and all(result["library_uat_complete"] for result in uat_candidates)
            else "blocked"
        ),
        "candidate_identity_sha256": expected_identity,
        "backend": (
            uat_candidates[0]["backend"]
            if uat_candidates and all(
                result["backend"] == uat_candidates[0]["backend"]
                for result in uat_candidates
            )
            else None
        ),
        "reports": len(uat_candidates),
    }

    return {
        "status": "passed" if not blockers else "blocked",
        "candidate_identity_sha256": next(iter(identities)) if len(identities) == 1 else None,
        "candidate_source_sha256": (
            next(iter(source_identities)) if len(source_identities) == 1 else None
        ),
        "compatibility_layers": {
            layer: by_layer.get(layer, {}).get("status", "missing") for layer in LAYERS
        },
        "runtime_performance_status": (
            "passed" if not runtime_performance_errors else "failed"
        ),
        "dual_language_support": {"status": "passed" if not support_errors else "blocked"},
        "server_operation_deferrals": {
            "status": "passed" if not validate_server_operation_deferrals(manifest) else "blocked"
        },
        "phase5_server_evidence": {
            "status": (
                "passed"
                if not validate_phase5_server_evidence(
                    phase5_server_evidence, current_source_sha256=current_source_sha256
                )
                else "blocked"
            )
        },
        "phase3_capability_evidence": {
            "status": (
                "passed"
                if not validate_phase3_capability_evidence(
                    phase3_capability_evidence,
                    current_source_sha256=current_source_sha256,
                )
                else "blocked"
            )
        },
        "phase4_runtime_evidence": {
            "status": (
                "passed"
                if not validate_phase4_runtime_evidence(
                    phase4_runtime_evidence,
                    current_source_sha256=current_source_sha256,
                )
                else "blocked"
            )
        },
        "capability_clause_tests": {
            "total": len(capability_clause_results),
            "passed": sum(row["status"] == "passed" for row in capability_clause_results),
            "blocked": sum(row["status"] != "passed" for row in capability_clause_results),
            "results": capability_clause_results,
        },
        "library_uat": library_uat,
        "rust_cutover_ready_capabilities": sorted(value for value in active_rust if isinstance(value, str)),
        "blockers": blockers,
    }


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "contracts" / "rust-port-v1.json")
    parser.add_argument("--compat-report", type=Path, action="append", required=True)
    parser.add_argument(
        "--runtime-performance",
        type=Path,
        default=ROOT / "contracts" / "benchmarks" / "rust-runtime-serving-current-windows.json",
    )
    parser.add_argument(
        "--phase3-capability-evidence",
        type=Path,
        default=ROOT / ".codex" / "adr015-phase3-capability-gate.json",
    )
    parser.add_argument(
        "--phase4-runtime-evidence",
        type=Path,
        default=ROOT / ".codex" / "adr015-phase4-runtime-gate.json",
    )
    parser.add_argument(
        "--phase5-server-evidence",
        type=Path,
        default=ROOT / ".codex" / "adr015-phase5-server-gate.json",
    )
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _args()
    result = evaluate_release_gate(
        manifest=_load_json(args.manifest),
        compatibility_reports=[_load_json(path) for path in args.compat_report],
        runtime_performance=_load_json(args.runtime_performance),
        phase3_capability_evidence=_load_json(args.phase3_capability_evidence),
        phase4_runtime_evidence=_load_json(args.phase4_runtime_evidence),
        phase5_server_evidence=_load_json(args.phase5_server_evidence),
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(encoded, end="")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded, encoding="utf-8")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
