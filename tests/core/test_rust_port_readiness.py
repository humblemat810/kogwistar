from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _readiness():
    path = ROOT / "scripts" / "rust_port_readiness.py"
    spec = importlib.util.spec_from_file_location("rust_port_readiness", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate() -> dict[str, object]:
    return {
        "identity_sha256": "a" * 64,
        "backend": "docker-clean-dual-venv-sharded-pgvector",
        "storage_root": "container-local-isolated",
    }


def test_isolated_docker_uat_candidate_is_accepted_for_library() -> None:
    result = _readiness().validate_library_uat_candidate(
        _candidate(), expected_candidate_identity_sha256="a" * 64
    )

    assert result["valid"] is True
    assert result["library_uat_complete"] is True
    assert result["environment"] == "isolated-docker-uat"


def test_readiness_rejects_local_or_customer_traffic_backend() -> None:
    candidate = _candidate()
    candidate["backend"] = "local-python"

    result = _readiness().validate_library_uat_candidate(candidate)

    assert result["valid"] is False
    assert result["library_uat_complete"] is False
    assert any("isolated Docker UAT" in error for error in result["errors"])


def test_readiness_fails_closed_on_mismatched_identity_or_storage() -> None:
    candidate = _candidate()
    candidate["storage_root"] = "shared-host-data"

    result = _readiness().validate_library_uat_candidate(
        candidate, expected_candidate_identity_sha256="b" * 64
    )

    assert result["valid"] is False
    assert any("does not match release candidate" in error for error in result["errors"])
    assert any("container-local-isolated" in error for error in result["errors"])
