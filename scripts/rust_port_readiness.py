"""Validate isolated Docker UAT candidate identity for library acceptance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DOCKER_UAT_BACKENDS = frozenset(
    {
        "docker-clean-dual-venv-sharded",
        "docker-clean-dual-venv-sharded-pgvector",
    }
)


def validate_library_uat_candidate(
    candidate: dict[str, Any],
    *,
    expected_candidate_identity_sha256: str | None = None,
) -> dict[str, Any]:
    """Require an isolated Docker UAT candidate, not customer traffic rollout."""
    errors: list[str] = []
    identity = candidate.get("identity_sha256")
    if not isinstance(identity, str) or len(identity) != 64:
        errors.append("candidate identity_sha256 must be a 64-character digest")
        identity = None
    if (
        expected_candidate_identity_sha256 is not None
        and identity != expected_candidate_identity_sha256
    ):
        errors.append("Docker UAT candidate identity does not match release candidate")

    backend = candidate.get("backend")
    if not isinstance(backend, str) or backend not in DOCKER_UAT_BACKENDS:
        errors.append("candidate backend must be a supported isolated Docker UAT backend")
    storage_root = candidate.get("storage_root")
    if storage_root != "container-local-isolated":
        errors.append("Docker UAT storage_root must be container-local-isolated")

    valid = not errors
    return {
        "valid": valid,
        "library_uat_complete": valid,
        "candidate_identity_sha256": identity,
        "environment": "isolated-docker-uat" if valid else None,
        "backend": backend if isinstance(backend, str) else None,
        "errors": errors,
    }


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path, help="Candidate object from a compatibility report")
    parser.add_argument("--expected-candidate-identity")
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _args()
    payload = json.loads(args.candidate.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("candidate evidence must be a JSON object")
    candidate = payload.get("candidate", payload)
    if not isinstance(candidate, dict):
        raise SystemExit("compatibility report candidate must be a JSON object")
    result = validate_library_uat_candidate(
        candidate,
        expected_candidate_identity_sha256=args.expected_candidate_identity,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(encoded, end="")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded, encoding="utf-8")
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
