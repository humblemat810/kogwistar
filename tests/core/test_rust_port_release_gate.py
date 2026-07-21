from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


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
            },
            "layers": [{"name": layer, "status": "passed", "returncode": 0}],
        }
        for layer in ("core", "parser", "sink", "application")
    ]


def _production_canary_set() -> dict:
    seed = json.loads(
        (ROOT / "contracts" / "canary" / "adr015-rehearsal-v1.json").read_text(
            encoding="utf-8"
        )
    )
    canaries = []
    for capability in sorted(_gate().REQUIRED_RUST_OWNERS):
        payload = dict(seed)
        payload["capability"] = capability
        payload["evidence_kind"] = "production"
        payload["candidate_identity_sha256"] = "a" * 64
        payload["stages"] = [dict(stage) for stage in seed["stages"]]
        canaries.append(payload)
    return {
        "schema_version": 2,
        "evidence_kind": "production",
        "candidate_identity_sha256": "a" * 64,
        "canaries": canaries,
    }


def _phase5_evidence(source_sha256: str | None = None) -> dict:
    source_sha256 = source_sha256 or _gate().candidate_source_fingerprint(ROOT)[0]
    groups = []
    for name in sorted(_gate().REQUIRED_PHASE5_GROUPS):
        groups.append(
            {
                "name": name,
                "status": "passed",
                "returncode": 0,
                "live_async_sse_required": name == "async-sse",
                "counts": {"tests": 1, "failures": 0, "errors": 0, "skipped": 0},
            }
        )
    return {
        "schema": "adr015-phase5-server-gate/v1",
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
                "counts": {"tests": 1, "failures": 0, "errors": 0, "skipped": 0},
            }
        )
    return {
        "schema": "adr015-phase3-capability-gate/v1",
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
                "counts": {"tests": 1, "failures": 0, "errors": 0, "skipped": 0},
            }
        )
    return {
        "schema": "adr015-phase4-runtime-gate/v1",
        "candidate_source_sha256": source_sha256,
        "async_rust_authority": "async-v2-worker-protocol",
        "groups": groups,
        "status": "passed",
    }


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
        canary_evidence=json.loads(
            (ROOT / "contracts" / "canary" / "adr015-rehearsal-v1.json").read_text(
                encoding="utf-8"
            )
        ),
    )

    assert result["status"] == "blocked"
    details = {row["detail"] for row in result["blockers"]}
    assert "generic persistent public-store performance gate remains failed" not in details
    assert "sqlite-meta is not Rust-cutover-ready" not in details
    assert "postgres-sequence-event-log is not Rust-cutover-ready" in details
    assert "workflow-runtime is not Rust-cutover-ready" in details
    assert "server-rest-sse-mcp-cli is not Rust-cutover-ready" in details
    assert any("per-capability production internal/test" in detail for detail in details)


def test_gate_passes_only_with_all_layers_perf_authority_and_production_canary() -> None:
    manifest = json.loads((ROOT / "contracts" / "rust-port-v1.json").read_text(encoding="utf-8"))
    for item in manifest["capability_ownership"]:
        if item["capability"] in _gate().REQUIRED_RUST_OWNERS:
            item["rust_cutover_ready"] = True
    manifest["current_persistent_evidence"]["gate_status"] = "passed"

    result = _gate().evaluate_release_gate(
        manifest=manifest,
        compatibility_reports=_compatibility("a" * 64),
        runtime_performance={"gate": {"status": "passed"}},
        phase3_capability_evidence=_phase3_capability_evidence(),
        phase4_runtime_evidence=_phase4_runtime_evidence(),
        phase5_server_evidence=_phase5_evidence(),
        canary_evidence=_production_canary_set(),
    )

    assert result["status"] == "passed"
    assert result["blockers"] == []
    assert result["server_operation_deferrals"] == {"status": "passed"}
    assert result["phase3_capability_evidence"] == {"status": "passed"}
    assert result["phase4_runtime_evidence"] == {"status": "passed"}
    assert result["phase5_server_evidence"] == {"status": "passed"}


def test_gate_rejects_missing_or_duplicate_versioned_server_deferrals() -> None:
    manifest = json.loads((ROOT / "contracts" / "rust-port-v1.json").read_text(encoding="utf-8"))
    for item in manifest["capability_ownership"]:
        if item["capability"] in _gate().REQUIRED_RUST_OWNERS:
            item["rust_cutover_ready"] = True
    manifest["current_persistent_evidence"]["gate_status"] = "passed"
    manifest["server_operation_deferrals"]["route_groups"][0]["operations"].append(
        manifest["server_operation_deferrals"]["route_groups"][0]["operations"][0]
    )

    result = _gate().evaluate_release_gate(
        manifest=manifest,
        compatibility_reports=_compatibility("a" * 64),
        runtime_performance={"gate": {"status": "passed"}},
        phase3_capability_evidence=_phase3_capability_evidence(),
        phase4_runtime_evidence=_phase4_runtime_evidence(),
        phase5_server_evidence=_phase5_evidence(),
        canary_evidence=_production_canary_set(),
    )

    assert result["status"] == "blocked"
    assert result["server_operation_deferrals"] == {"status": "blocked"}
    details = {row["detail"] for row in result["blockers"]}
    assert "duplicate deferred route DELETE /admin/doc/{doc_id}" in details


def test_gate_rejects_historical_compatibility_report_after_source_changes() -> None:
    manifest = json.loads((ROOT / "contracts" / "rust-port-v1.json").read_text(encoding="utf-8"))
    result = _gate().evaluate_release_gate(
        manifest=manifest,
        compatibility_reports=_compatibility("a" * 64, "f" * 64),
        runtime_performance={"gate": {"status": "passed"}},
        phase3_capability_evidence=_phase3_capability_evidence(),
        phase4_runtime_evidence=_phase4_runtime_evidence(),
        phase5_server_evidence=_phase5_evidence(),
        canary_evidence=_production_canary_set(),
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
        runtime_performance={"gate": {"status": "passed"}},
        phase3_capability_evidence=_phase3_capability_evidence(),
        phase4_runtime_evidence=_phase4_runtime_evidence(),
        phase5_server_evidence=evidence,
        canary_evidence=_production_canary_set(),
    )

    details = {row["detail"] for row in result["blockers"]}
    assert "Phase-5 server evidence does not match current candidate source" in details
    assert "Phase-5 async-sse evidence must be live and unskipped" in details


def test_gate_rejects_stale_or_skipped_phase3_capability_evidence() -> None:
    manifest = json.loads((ROOT / "contracts" / "rust-port-v1.json").read_text(encoding="utf-8"))
    evidence = _phase3_capability_evidence("f" * 64)
    queue = next(group for group in evidence["groups"] if group["name"] == "queues-leases-lanes")
    queue["counts"]["skipped"] = 1

    result = _gate().evaluate_release_gate(
        manifest=manifest,
        compatibility_reports=_compatibility("a" * 64),
        runtime_performance={"gate": {"status": "passed"}},
        phase3_capability_evidence=evidence,
        phase4_runtime_evidence=_phase4_runtime_evidence(),
        phase5_server_evidence=_phase5_evidence(),
        canary_evidence=_production_canary_set(),
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
        runtime_performance={"gate": {"status": "passed"}},
        phase3_capability_evidence=_phase3_capability_evidence(),
        phase4_runtime_evidence=evidence,
        phase5_server_evidence=_phase5_evidence(),
        canary_evidence=_production_canary_set(),
    )

    details = {row["detail"] for row in result["blockers"]}
    assert "Phase-4 runtime evidence does not match current candidate source" in details
    assert "Phase-4 runtime group postgres has failures, errors, or skips" in details
