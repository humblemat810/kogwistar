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


def _compatibility(identity: str) -> list[dict]:
    return [
        {
            "candidate": {"identity_sha256": identity},
            "layers": [{"name": layer, "status": "passed", "returncode": 0}],
        }
        for layer in ("core", "parser", "sink", "application")
    ]


def _production_canary() -> dict:
    payload = json.loads(
        (ROOT / "contracts" / "canary" / "adr015-rehearsal-v1.json").read_text(
            encoding="utf-8"
        )
    )
    payload["evidence_kind"] = "production"
    payload["candidate_identity_sha256"] = "a" * 64
    return payload


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
        canary_evidence=json.loads(
            (ROOT / "contracts" / "canary" / "adr015-rehearsal-v1.json").read_text(
                encoding="utf-8"
            )
        ),
    )

    assert result["status"] == "blocked"
    details = {row["detail"] for row in result["blockers"]}
    assert "generic persistent public-store performance gate remains failed" not in details
    assert "sqlite-meta is not Rust-cutover-ready" in details
    assert "workflow-runtime is not Rust-cutover-ready" in details
    assert "server-rest-sse-mcp-cli is not Rust-cutover-ready" in details
    assert any("production internal/test" in detail for detail in details)


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
        canary_evidence=_production_canary(),
    )

    assert result["status"] == "passed"
    assert result["blockers"] == []
