"""Compare Python ACL decisions with the Rust graph ACL projection kernel."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

from kogwistar.acl.graph import ACLGraph

ROOT = Path(__file__).resolve().parents[1]
TRUTH_GRAPH = "knowledge"
PRINCIPAL = {"id": "user:reader", "groups": ["reviewers"], "security_scope": "team-a"}


def _metadata_for(index: int) -> dict[str, object]:
    mode = ("public", "shared", "group", "scope", "private")[index % 5]
    metadata: dict[str, object] = {
        "entity_type": "acl_record",
        "acl_truth_graph": TRUTH_GRAPH,
        "acl_target_entity_id": f"node-{index:06d}",
        "acl_target_grain": "node",
        "acl_version": 1,
        "acl_mode": mode,
        "tombstoned": index % 97 == 0,
    }
    if mode == "shared":
        metadata["shared_with_principals"] = [PRINCIPAL["id"]]
    elif mode == "group":
        metadata["shared_with_groups"] = ["reviewers"]
    elif mode == "scope":
        metadata["security_scope"] = "team-a"
    elif mode == "private":
        metadata["owner_id"] = "user:other"
    return metadata


def _workload(items: int) -> dict[str, object]:
    return {
        "schema": "kogwistar-acl-projection-workload/v1",
        "truth_graph": TRUTH_GRAPH,
        "principal": PRINCIPAL,
        "nodes": [
            {"id": f"node-{index:06d}", "metadata": {}}
            for index in range(items)
        ]
        + [
            {"id": f"acl-{index:06d}", "metadata": _metadata_for(index)}
            for index in range(items)
        ],
        "edges": [],
    }


def _digest(ids: list[str]) -> str:
    payload = json.dumps(ids, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _python_sample(workload: dict[str, object], iterations: int) -> dict[str, object]:
    nodes = workload["nodes"]
    principal = workload["principal"]
    assert isinstance(nodes, list) and isinstance(principal, dict)
    targets = nodes[: len(nodes) // 2]
    acl_records = nodes[len(nodes) // 2 :]
    acl = ACLGraph(cache_enabled=True, max_record_cache_size=len(targets) + 1)
    for record in acl_records:
        metadata = record["metadata"]
        acl.add_record(
            grain="node",
            truth_graph=TRUTH_GRAPH,
            entity_id=str(metadata["acl_target_entity_id"]),
            version=int(metadata["acl_version"]),
            mode=metadata["acl_mode"],
            owner_id=metadata.get("owner_id"),
            security_scope=metadata.get("security_scope"),
            shared_with_principals=metadata.get("shared_with_principals", ()),
            shared_with_groups=metadata.get("shared_with_groups", ()),
            tombstoned=bool(metadata["tombstoned"]),
        )
    started = time.perf_counter_ns()
    visible_ids: list[str] = []
    for _ in range(iterations):
        visible_ids = [
            str(record["id"])
            for record in targets
            if acl.decide(
                grain="node",
                truth_graph=TRUTH_GRAPH,
                entity_id=str(record["id"]),
                principal_id=str(principal["id"]),
                principal_groups=principal["groups"],
                security_scope=str(principal["security_scope"]),
            ).visible
        ]
    elapsed_ns = time.perf_counter_ns() - started
    return {
        "elapsed_ns": elapsed_ns,
        "iterations": iterations,
        "visible_count": len(visible_ids),
        "visible_digest": _digest(visible_ids),
    }


def _cgroup_value(name: str) -> str | None:
    path = Path("/sys/fs/cgroup") / name
    return path.read_text(encoding="ascii").strip() if path.is_file() else None


def _verify_limits(expected_cpus: float, expected_memory_bytes: int) -> dict[str, object]:
    cpu_max = _cgroup_value("cpu.max")
    memory_max = _cgroup_value("memory.max")
    actual_cpus: float | None = None
    if cpu_max and cpu_max != "max":
        quota, period = (int(value) for value in cpu_max.split())
        actual_cpus = quota / period
    actual_memory = int(memory_max) if memory_max and memory_max != "max" else None
    cpu_quota = _cgroup_value("cpu/cpu.cfs_quota_us")
    cpu_period = _cgroup_value("cpu/cpu.cfs_period_us")
    memory_limit = _cgroup_value("memory/memory.limit_in_bytes")
    if actual_cpus is None and cpu_quota and cpu_period:
        quota_value, period_value = int(cpu_quota), int(cpu_period)
        if quota_value > 0 and period_value > 0:
            actual_cpus = quota_value / period_value
    if actual_memory is None and memory_limit:
        memory_value = int(memory_limit)
        if memory_value > 0:
            actual_memory = memory_value
    return {
        "cpu_max": cpu_max,
        "memory_max": memory_max,
        "cpu_cfs_quota_us": cpu_quota,
        "cpu_cfs_period_us": cpu_period,
        "memory_limit_in_bytes": memory_limit,
        "actual_cpus": actual_cpus,
        "actual_memory_bytes": actual_memory,
        "expected_cpus": expected_cpus,
        "expected_memory_bytes": expected_memory_bytes,
        "verified": actual_cpus == expected_cpus and actual_memory == expected_memory_bytes,
    }


def _run_rust(workload_path: Path, samples: int, iterations: int) -> dict[str, Any]:
    command = [
        "cargo", "run", "--quiet", "--release", "--manifest-path",
        str(ROOT / "rust" / "Cargo.toml"), "-p", "kogwistar-store",
        "--example", "acl_projection_benchmark", "--",
        "--input", str(workload_path), "--samples", str(samples),
        "--iterations", str(iterations),
    ]
    completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--items", type=int, default=2000)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--expected-cpus", type=float, required=True)
    parser.add_argument("--expected-memory-bytes", type=int, required=True)
    args = parser.parse_args()
    if args.items < 1 or args.samples < 2 or args.iterations < 1:
        parser.error("items/iterations must be positive and samples must be at least 2")

    args.output.mkdir(parents=True, exist_ok=True)
    limits = _verify_limits(args.expected_cpus, args.expected_memory_bytes)
    workload = _workload(args.items)
    workload_path = args.output / "acl_projection_workload.v1.json"
    workload_path.write_text(
        json.dumps(workload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    python_warmup = _python_sample(workload, args.iterations)
    python_samples = [_python_sample(workload, args.iterations) for _ in range(args.samples)]
    rust = _run_rust(workload_path, args.samples, args.iterations)
    rust_results = rust["results"]
    rust_warmup = next(row for row in rust_results if row["kind"] == "warmup")
    rust_samples = [row for row in rust_results if row["kind"] == "sample"]
    expected_count = sum(
        not bool(_metadata_for(index)["tombstoned"]) and index % 5 != 4
        for index in range(args.items)
    )
    for row in [python_warmup, *python_samples]:
        if row["visible_count"] != expected_count:
            raise RuntimeError("Python ACL projection returned unexpected visible count")
    if len(rust_samples) != args.samples:
        raise RuntimeError("Rust benchmark returned an unexpected sample count")
    python_digest = python_samples[0]["visible_digest"]
    for row in [python_warmup, *python_samples]:
        if row["visible_digest"] != python_digest:
            raise RuntimeError("Python ACL result digest changed between samples")
    for row in [rust_warmup, *rust_samples]:
        if row["visible_count"] != expected_count or _digest(row["visible_ids"]) != python_digest:
            raise RuntimeError("Rust ACL projection differs from Python canonical result")
    if not limits["verified"]:
        raise RuntimeError(f"fixed resource limits not verified: {limits}")

    def summary(rows: list[dict[str, object]]) -> dict[str, object]:
        latencies = [int(row["elapsed_ns"]) / args.iterations / 1_000_000 for row in rows]
        mean = statistics.fmean(latencies)
        return {
            "median_ms_per_projection": statistics.median(latencies),
            "samples_ms_per_projection": latencies,
            "sample_cv_percent": statistics.pstdev(latencies) / mean * 100 if mean else 0.0,
            "visible_count": expected_count,
            "visible_digest": python_digest,
        }

    report = {
        "schema": "kogwistar-acl-projection-benchmark-report/v1",
        "status": "complete",
        "scope": "in-process ACL projection kernel; no HTTP, database, or authentication-token costs",
        "comparison_note": (
            "Python evaluates one target per ACLGraph.decide call; Rust filters a batch via "
            "GraphAclProjection.filter_records. The ratio includes this API/algorithm shape "
            "difference and is not a language-only speedup."
        ),
        "workload": {
            "items": args.items,
            "acl_records": args.items,
            "samples": args.samples,
            "warmup_iterations": args.iterations,
            "iterations_per_sample": args.iterations,
            "modes": ["python_acl_graph_decide_cached", "rust_graph_records_visible_by_acl"],
        },
        "limits": limits,
        "runtime": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "rustc_version": subprocess.run(
                ["rustc", "--version"], check=True, capture_output=True, text=True
            ).stdout.strip(),
        },
        "warmup": {"python": python_warmup, "rust": rust_warmup},
        "python": summary(python_samples),
        "rust": summary(rust_samples),
        "rust_python_latency_ratio": summary(rust_samples)["median_ms_per_projection"]
        / summary(python_samples)["median_ms_per_projection"],
    }
    output_path = args.output / "acl_projection_report.v1.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
