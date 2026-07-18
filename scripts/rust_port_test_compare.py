"""Compare equivalent serial and parallel ADR-015 compatibility reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("serial", type=Path)
    parser.add_argument("parallel", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--layer", action="append", help="Compare only named layers.")
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"report is not an object: {path}")
    return payload


def _layer_groups(report: dict[str, Any]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for layer in report.get("layers", []):
        name = layer.get("name")
        if not isinstance(name, str):
            raise ValueError("layer name is invalid")
        keys = [run.get("group_key") for run in layer.get("runs", [])]
        if not all(isinstance(key, str) for key in keys):
            raise ValueError(f"{name} has invalid group keys")
        if len(keys) != len(set(keys)):
            raise ValueError(f"{name} has duplicate group keys")
        result[name] = sorted(keys)
    return result


def compare(
    serial: dict[str, Any],
    parallel: dict[str, Any],
    *,
    selected_layers: list[str] | None = None,
) -> dict[str, Any]:
    if serial.get("status") != "passed" or parallel.get("status") != "passed":
        raise ValueError("both reports must have passed")
    serial_candidate = serial.get("candidate", {})
    parallel_candidate = parallel.get("candidate", {})
    for field in (
        "candidate_source_sha256",
        "verification_harness_sha256",
        "test_profile",
    ):
        if serial_candidate.get(field) != parallel_candidate.get(field):
            raise ValueError(f"candidate {field} differs")
    serial_groups = _layer_groups(serial)
    parallel_groups = _layer_groups(parallel)
    if selected_layers:
        missing = (set(selected_layers) - serial_groups.keys()) | (
            set(selected_layers) - parallel_groups.keys()
        )
        if missing:
            raise ValueError(f"selected layers are missing: {sorted(missing)}")
        serial_groups = {name: serial_groups[name] for name in selected_layers}
        parallel_groups = {name: parallel_groups[name] for name in selected_layers}
    if serial_groups != parallel_groups:
        raise ValueError("serial and parallel group coverage differs")
    serial_orchestrator = serial.get("execution", {}).get(
        "container_orchestrator_sha256"
    )
    parallel_orchestrator = parallel.get("execution", {}).get(
        "container_orchestrator_sha256"
    )
    if serial_orchestrator != parallel_orchestrator:
        raise ValueError("container orchestrator identity differs")

    if selected_layers:
        serial_seconds = sum(
            float(layer.get("duration_seconds", 0.0))
            for layer in serial["layers"]
            if layer.get("name") in selected_layers
        )
        parallel_seconds = sum(
            float(layer.get("duration_seconds", 0.0))
            for layer in parallel["layers"]
            if layer.get("name") in selected_layers
        )
    else:
        serial_seconds = float(
            serial.get("execution", {}).get("orchestrator_wall_seconds")
            or sum(
                float(layer.get("duration_seconds", 0.0)) for layer in serial["layers"]
            )
        )
        parallel_seconds = float(
            parallel.get("execution", {}).get("orchestrator_wall_seconds")
            or max(
                (
                    float(layer.get("duration_seconds", 0.0))
                    for layer in parallel["layers"]
                ),
                default=0.0,
            )
        )
    return {
        "comparison_version": 1,
        "candidate_source_sha256": serial_candidate["candidate_source_sha256"],
        "verification_harness_sha256": serial_candidate["verification_harness_sha256"],
        "test_profile": serial_candidate["test_profile"],
        "container_orchestrator_sha256": serial_orchestrator,
        "layers": {
            name: {"group_count": len(groups)} for name, groups in serial_groups.items()
        },
        "serial_duration_seconds": round(serial_seconds, 6),
        "parallel_duration_seconds": round(parallel_seconds, 6),
        "wall_speedup": (
            None
            if parallel_seconds <= 0
            else round(serial_seconds / parallel_seconds, 6)
        ),
        "coverage_equal": True,
    }


def main() -> int:
    args = _args()
    result = compare(
        _load(args.serial), _load(args.parallel), selected_layers=args.layer
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
