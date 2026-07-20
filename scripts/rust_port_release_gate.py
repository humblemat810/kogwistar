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

    canary = _readiness_module().validate_canary_evidence(canary_evidence)
    if not canary["production_complete"]:
        blockers.append(
            {
                "gate": "canary",
                "detail": "production internal/test -> 1% -> 10% -> 50% -> 100% evidence is incomplete",
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
    parser.add_argument("--canary-evidence", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _args()
    result = evaluate_release_gate(
        manifest=_load_json(args.manifest),
        compatibility_reports=[_load_json(path) for path in args.compat_report],
        runtime_performance=_load_json(args.runtime_performance),
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
