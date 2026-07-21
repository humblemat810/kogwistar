"""Validate ADR-015 capability canary evidence without promoting ownership.

This checker is deliberately fail-closed. A local rehearsal can prove that the
evidence shape and rollback workflow work, but only production evidence covering
internal/test -> 1% -> 10% -> 50% -> 100% can satisfy the rollout gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


STAGES = (
    ("internal/test", 0),
    ("1%", 1),
    ("10%", 10),
    ("50%", 50),
    ("100%", 100),
)


def _validate_identity(value: object, *, field: str, errors: list[str]) -> str | None:
    if not isinstance(value, str) or len(value) != 64:
        errors.append(f"{field} must be a 64-character digest")
        return None
    return value


def validate_canary_evidence(payload: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if payload.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    evidence_kind = payload.get("evidence_kind")
    if evidence_kind not in {"rehearsal", "production"}:
        errors.append("evidence_kind must be rehearsal or production")
    capability = payload.get("capability")
    if not isinstance(capability, str) or not capability.strip():
        errors.append("capability must be non-empty")
    identity = _validate_identity(
        payload.get("candidate_identity_sha256"),
        field="candidate_identity_sha256",
        errors=errors,
    )

    stages = payload.get("stages")
    if not isinstance(stages, list):
        errors.append("stages must be an array")
        stages = []
    if len(stages) != len(STAGES):
        errors.append("stages must cover internal/test, 1%, 10%, 50%, and 100%")

    for index, (expected_name, expected_percent) in enumerate(STAGES):
        if index >= len(stages) or not isinstance(stages[index], dict):
            continue
        stage = stages[index]
        prefix = f"stages[{index}]"
        if stage.get("name") != expected_name:
            errors.append(f"{prefix}.name must be {expected_name}")
        if stage.get("traffic_percent") != expected_percent:
            errors.append(f"{prefix}.traffic_percent must be {expected_percent}")
        if not isinstance(stage.get("observed_seconds"), int) or stage["observed_seconds"] <= 0:
            errors.append(f"{prefix}.observed_seconds must be positive")
        if not isinstance(stage.get("requests"), int) or stage["requests"] <= 0:
            errors.append(f"{prefix}.requests must be positive")
        for field in (
            "parity_mismatches",
            "unexplained_correctness_mismatches",
            "duplicate_events",
            "corrupt_projections",
        ):
            if stage.get(field) != 0:
                errors.append(f"{prefix}.{field} must be zero")
        if stage.get("rollback_rehearsed") is not True:
            errors.append(f"{prefix}.rollback_rehearsed must be true")
        if stage.get("data_conversion_required") is not False:
            errors.append(f"{prefix}.data_conversion_required must be false")

    valid = not errors
    production_complete = valid and evidence_kind == "production"
    return {
        "valid": valid,
        "evidence_kind": evidence_kind,
        "production_complete": production_complete,
        "ownership_promotion_allowed": production_complete,
        "errors": errors,
    }


def validate_canary_evidence_set(
    payload: dict[str, Any],
    *,
    required_capabilities: set[str],
    expected_candidate_identity_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate one same-candidate production canary per capability.

    A valid five-stage canary for one store cannot authorize another store,
    runtime, or server capability.  The set is the full-release evidence; the
    individual v1 document remains useful for rehearsing one capability.
    """
    errors: list[str] = []
    if payload.get("schema_version") != 2:
        errors.append("canary evidence set schema_version must be 2")
    evidence_kind = payload.get("evidence_kind")
    if evidence_kind not in {"rehearsal", "production"}:
        errors.append("canary evidence set evidence_kind must be rehearsal or production")
    identity = _validate_identity(
        payload.get("candidate_identity_sha256"),
        field="canary evidence set candidate_identity_sha256",
        errors=errors,
    )
    if (
        expected_candidate_identity_sha256 is not None
        and identity is not None
        and identity != expected_candidate_identity_sha256
    ):
        errors.append("canary evidence set does not match release candidate identity")

    canaries = payload.get("canaries")
    if not isinstance(canaries, list):
        errors.append("canary evidence set canaries must be an array")
        canaries = []

    by_capability: dict[str, dict[str, Any]] = {}
    child_results: dict[str, dict[str, Any]] = {}
    for index, canary in enumerate(canaries):
        prefix = f"canaries[{index}]"
        if not isinstance(canary, dict):
            errors.append(f"{prefix} must be an object")
            continue
        capability = canary.get("capability")
        if not isinstance(capability, str) or not capability:
            errors.append(f"{prefix}.capability must be non-empty")
            continue
        if capability in by_capability:
            errors.append(f"duplicate capability canary {capability}")
            continue
        by_capability[capability] = canary
        result = validate_canary_evidence(canary)
        child_results[capability] = result
        if canary.get("evidence_kind") != evidence_kind:
            errors.append(f"{prefix}.evidence_kind must match evidence set")
        if identity is not None and canary.get("candidate_identity_sha256") != identity:
            errors.append(f"{prefix}.candidate_identity_sha256 must match evidence set")
        if not result["valid"]:
            errors.extend(f"{prefix}: {error}" for error in result["errors"])

    missing = sorted(required_capabilities - set(by_capability))
    extra = sorted(set(by_capability) - required_capabilities)
    if missing:
        errors.append("missing capability canaries: " + ", ".join(missing))
    if extra:
        errors.append("unexpected capability canaries: " + ", ".join(extra))

    valid = not errors
    production_complete = (
        valid
        and evidence_kind == "production"
        and all(
            child_results[capability]["production_complete"]
            for capability in required_capabilities
        )
    )
    return {
        "valid": valid,
        "evidence_kind": evidence_kind,
        "candidate_identity_sha256": identity,
        "required_capabilities": sorted(required_capabilities),
        "completed_capabilities": sorted(
            capability
            for capability, result in child_results.items()
            if result["production_complete"]
        ),
        "production_complete": production_complete,
        "ownership_promotion_allowed": production_complete,
        "errors": errors,
    }


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument(
        "--required-capability",
        action="append",
        help="Validate a schema-v2 canary set for this capability; repeat as needed.",
    )
    parser.add_argument(
        "--expected-candidate-identity",
        help="Require the schema-v2 canary set to match this release candidate.",
    )
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _args()
    payload = json.loads(args.evidence.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("canary evidence must be a JSON object")
    if args.required_capability:
        result = validate_canary_evidence_set(
            payload,
            required_capabilities=set(args.required_capability),
            expected_candidate_identity_sha256=args.expected_candidate_identity,
        )
    else:
        result = validate_canary_evidence(payload)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(encoded, end="")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded, encoding="utf-8")
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
