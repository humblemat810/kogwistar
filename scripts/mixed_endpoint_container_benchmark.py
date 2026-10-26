"""Compare Python, mixed-authority, and Rust-native MCP paths under one limit."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

try:
    from .mixed_mcp_benchmark import run_benchmark
    from .mixed_runtime_container_benchmark import (
        _cgroup_value,
        _installed_packages,
        _verify_cgroup_limits,
    )
except ImportError:
    from mixed_mcp_benchmark import run_benchmark
    from mixed_runtime_container_benchmark import (
        _cgroup_value,
        _installed_packages,
        _verify_cgroup_limits,
    )

ENDPOINT_MODES = ("http", "rust_authority_http", "rust_native")


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--items", type=int, action="append", required=True)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument(
        "--mode",
        action="append",
        choices=ENDPOINT_MODES,
        help="mode to include; defaults to Python HTTP, Python HTTP with Rust authority, and Rust-native HTTP",
    )
    parser.add_argument("--expected-cpus", type=float, required=True)
    parser.add_argument("--expected-memory-bytes", type=int, required=True)
    args = parser.parse_args()
    if args.samples < 2 or any(item < 1 for item in args.items):
        parser.error("provide positive --items values and at least two samples")
    modes = tuple(args.mode) if args.mode else (
        "http",
        "rust_authority_http",
        "rust_native",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    limits = {
        "cpu_max": _cgroup_value("cpu.max"),
        "memory_max": _cgroup_value("memory.max"),
        "expected_cpus": args.expected_cpus,
        "expected_memory_bytes": args.expected_memory_bytes,
    }
    reports: list[dict[str, Any]] = []
    for size in args.items:
        report = run_benchmark(
            args.output / str(size),
            items=size,
            samples=args.samples,
            modes=modes,
            include_concurrent=False,
        )
        reports.append(report)

    limits_verified = _verify_cgroup_limits(
        limits["cpu_max"],
        limits["memory_max"],
        args.expected_cpus,
        args.expected_memory_bytes,
    )
    result = {
        "schema": "kogwistar-resource-matched-endpoint-report/v1",
        "status": "complete" if limits_verified else "invalid_container_limits",
        "modes": list(modes),
        "items": args.items,
        "samples_per_mode": args.samples,
        "state_digest_equal": all(row["state_digest_equal"] for row in reports),
        "reports": reports,
        "runtime": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable": sys.executable,
            "cpu_count": os.cpu_count(),
            "packages": _installed_packages(),
            "limits": limits,
            "limits_verified": limits_verified,
            "native_binary": os.environ.get("KOGWISTAR_MIXED_NATIVE_BINARY"),
        },
    }
    if not result["state_digest_equal"]:
        result["status"] = "failed_digest_parity"
    _write_json(args.output / "endpoint-report.v1.json", result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
