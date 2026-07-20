"""Aggregate ADR-015 release evidence and report exact remaining blockers."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

try:
    # Script entry point imports its sibling without package qualification.
    from adr015_source_identity import candidate_source_fingerprint  # type: ignore[import-not-found]
except ModuleNotFoundError:  # imported as a repository module in unit tests
    from scripts.adr015_source_identity import candidate_source_fingerprint


ROOT = Path(__file__).resolve().parents[1]
LAYERS = ("core", "parser", "sink", "application")
SERVER_DEFERRAL_ROUTE_COUNT_V1 = 16
SERVER_DEFERRAL_SYSCALL_COUNT_V1 = 5
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


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _readiness_module():
    path = ROOT / "scripts" / "rust_port_readiness.py"
    spec = importlib.util.spec_from_file_location("rust_port_readiness", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    if not isinstance(route_groups, list) or not route_groups:
        errors.append("route_groups must be a non-empty array")
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
    """Require current local serving evidence without mistaking it for rollout.

    This is deliberately independent of Rust authority flags and production
    canary evidence.  A later server promotion must have both the proven Python
    baseline and its real deployment evidence.
    """
    errors: list[str] = []
    if evidence.get("schema") != "adr015-phase5-server-gate/v1":
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
        if name == "async-sse" and (
            group.get("live_async_sse_required") is not True
            or not isinstance(counts, dict)
            or counts.get("skipped") != 0
        ):
            errors.append("Phase-5 async-sse evidence must be live and unskipped")
    return errors


def validate_phase3_capability_evidence(
    evidence: dict[str, Any], *, current_source_sha256: str
) -> list[str]:
    """Require separate live-PostgreSQL proof for each pending store capability.

    Local evidence proves only implementation readiness; authority remains
    Python-owned until the matching same-candidate production canary succeeds.
    """
    errors: list[str] = []
    if evidence.get("schema") != "adr015-phase3-capability-gate/v1":
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
    if evidence.get("schema") != "adr015-phase4-runtime-gate/v1":
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


def evaluate_release_gate(
    *,
    manifest: dict[str, Any],
    compatibility_reports: list[dict[str, Any]],
    runtime_performance: dict[str, Any],
    phase3_capability_evidence: dict[str, Any],
    phase4_runtime_evidence: dict[str, Any],
    phase5_server_evidence: dict[str, Any],
    canary_evidence: dict[str, Any],
) -> dict[str, Any]:
    blockers: list[dict[str, str]] = []
    by_layer: dict[str, dict[str, Any]] = {}
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
        for layer in report.get("layers", []):
            if isinstance(layer, dict) and layer.get("name") in LAYERS:
                by_layer[str(layer["name"])] = layer
    for layer in LAYERS:
        evidence = by_layer.get(layer)
        if evidence is None or evidence.get("status") != "passed" or evidence.get("returncode") != 0:
            blockers.append(
                {"gate": "compatibility", "detail": f"{layer} layer lacks passing evidence"}
            )
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

    performance_status = runtime_performance.get("gate", {}).get("status")
    if performance_status != "passed":
        blockers.append(
            {"gate": "performance", "detail": "runtime serving scale benchmark has not passed"}
        )
    persistent = manifest.get("current_persistent_evidence", {})
    if persistent.get("gate_status") != "passed":
        blockers.append(
            {
                "gate": "performance",
                "detail": "generic persistent public-store performance gate remains failed",
            }
        )

    for detail in validate_server_operation_deferrals(manifest):
        blockers.append({"gate": "server-deferral", "detail": detail})

    ownership = manifest.get("capability_ownership", [])
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
    canary = _readiness_module().validate_canary_evidence_set(
        canary_evidence,
        required_capabilities=REQUIRED_RUST_OWNERS,
        expected_candidate_identity_sha256=expected_identity,
    )
    if not canary["production_complete"]:
        blockers.append(
            {
                "gate": "canary",
                "detail": "per-capability production internal/test -> 1% -> 10% -> 50% -> 100% evidence is incomplete",
            }
        )

    return {
        "status": "passed" if not blockers else "blocked",
        "candidate_identity_sha256": next(iter(identities)) if len(identities) == 1 else None,
        "candidate_source_sha256": (
            next(iter(source_identities)) if len(source_identities) == 1 else None
        ),
        "compatibility_layers": {
            layer: by_layer.get(layer, {}).get("status", "missing") for layer in LAYERS
        },
        "runtime_performance_status": performance_status,
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
        "canary": canary,
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
        "--canary-evidence",
        type=Path,
        required=True,
        help="Schema-v2 same-candidate canary set, one five-stage record per Rust capability.",
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
        canary_evidence=_load_json(args.canary_evidence),
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(encoded, end="")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded, encoding="utf-8")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
