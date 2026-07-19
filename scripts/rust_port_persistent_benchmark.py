"""Benchmark current Python and Rust authoritative SQLite capability paths.

Unlike the Phase-2 snapshot benchmark, this uses persistent SQLite databases and
the production-compatible event/projection schema. Each mode runs in a fresh
process and performs the same durable capability calls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Callable

from kogwistar._rust_bridge import store_sqlite
from kogwistar.engine_core.engine_sqlite import EngineSQLite


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_VERSION = 1
DATASET = "rust-port-persistent-sqlite-v1"
WORKLOADS = (
    "event_append",
    "event_replay",
    "projection_replace",
    "projection_list",
)


def _peak_rss_bytes() -> int:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(ProcessMemoryCounters),
                wintypes.DWORD,
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            if psapi.GetProcessMemoryInfo(
                kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
            ):
                return int(counters.PeakWorkingSetSize)
        except (AttributeError, OSError, ValueError):
            pass
        try:
            import psutil  # type: ignore[import-untyped]

            return int(psutil.Process(os.getpid()).memory_info().rss)
        except (ImportError, OSError):
            raise RuntimeError("no supported Windows RSS collector is available") from None

    import resource

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * percentile) - 1)]


def _measure(operation: Callable[[], int]) -> dict[str, float | int]:
    started = time.perf_counter_ns()
    count = operation()
    elapsed_ns = time.perf_counter_ns() - started
    return {
        "elapsed_ms": elapsed_ns / 1_000_000,
        "operations": count,
        "throughput_ops_per_sec": count * 1_000_000_000 / elapsed_ns,
    }


def _projection(index: int) -> dict[str, Any]:
    return {
        "payload": {"index": index, "label": f"row-{index:05d}"},
        "last_authoritative_seq": index + 1,
        "last_materialized_seq": index + 1,
        "projection_schema_version": 1,
        "materialization_status": "ready",
    }


def _run_mode(mode: str, path: Path, *, items: int, read_cycles: int) -> dict[str, Any]:
    namespace = "benchmark"
    projection_namespace = "benchmark_projection"
    python_store = EngineSQLite(path.parent, filename=path.name)
    python_store.ensure_initialized()
    baseline_peak_rss_bytes = _peak_rss_bytes()

    if mode == "python":
        def append_events() -> int:
            for index in range(items):
                python_store.append_entity_event(
                    namespace=namespace,
                    event_id=f"event-{index:05d}",
                    entity_kind="node",
                    entity_id=f"node-{index:05d}",
                    op="UPSERT",
                    payload_json=_canonical({"replacement": {"id": f"node-{index:05d}"}}),
                )
            return items

        def replay_events() -> int:
            count = 0
            for _ in range(read_cycles):
                count += len(list(python_store.iter_entity_events(
                    namespace=namespace, from_seq=1, batch_size=items + 1
                )))
            return count

        def replace_projections() -> int:
            for index in range(items):
                python_store.replace_named_projection(
                    projection_namespace, f"key-{index:05d}", **_projection(index)
                )
            return items

        def list_projections() -> int:
            return sum(
                len(python_store.list_named_projections(projection_namespace))
                for _ in range(read_cycles)
            )
    else:
        def call(operation: dict[str, Any]) -> Any:
            return store_sqlite(path=path, operation=operation)

        def append_events() -> int:
            for index in range(items):
                call({
                    "kind": "raw_append",
                    "namespace": namespace,
                    "event_id": f"event-{index:05d}",
                    "entity_kind": "node",
                    "entity_id": f"node-{index:05d}",
                    "op": "UPSERT",
                    "payload_json": _canonical({"replacement": {"id": f"node-{index:05d}"}}),
                })
            return items

        def replay_events() -> int:
            return sum(len(call({
                "kind": "exclusive_raw_replay",
                "namespace": namespace,
                "after_seq": 0,
                "limit": items + 1,
            })) for _ in range(read_cycles))

        def replace_projections() -> int:
            for index in range(items):
                call({
                    "kind": "replace_named_projection",
                    "namespace": projection_namespace,
                    "key": f"key-{index:05d}",
                    **_projection(index),
                })
            return items

        def list_projections() -> int:
            return sum(len(call({
                "kind": "list_named_projections",
                "namespace": projection_namespace,
            })) for _ in range(read_cycles))

    metrics = {
        "event_append": _measure(append_events),
        "event_replay": _measure(replay_events),
        "projection_replace": _measure(replace_projections),
        "projection_list": _measure(list_projections),
    }
    if mode == "python":
        events = [
            list(row)
            for row in python_store.iter_entity_events(
                namespace=namespace, from_seq=1, batch_size=items + 1
            )
        ]
        projections = python_store.list_named_projections(projection_namespace)
    else:
        # Keep each authority benchmark on its own facade. Loading CPython's
        # `_sqlite3` and a bundled-rusqlite PyO3 extension against the same WAL
        # in one process is not a supported authority mode and can interpose
        # SQLite symbols on ELF platforms. Cross-implementation rollback is
        # verified separately, after the writer process exits.
        rust_events = call({
            "kind": "exclusive_raw_replay",
            "namespace": namespace,
            "after_seq": 0,
            "limit": items + 1,
        })
        events = [
            [
                row["seq"], row["entity_kind"], row["entity_id"],
                row["op"], row["payload_json"],
            ]
            for row in rust_events
        ]
        projections = call({
            "kind": "list_named_projections",
            "namespace": projection_namespace,
        })
    canonical_state = {
        "events": events,
        "projections": [
            {key: value for key, value in row.items() if key != "updated_at_ms"}
            for row in projections
        ],
    }
    return {
        "mode": mode,
        "metrics": metrics,
        "state_digest": _digest(canonical_state),
        "event_count": len(events),
        "projection_count": len(projections),
        "database_bytes": path.stat().st_size,
        "peak_rss_bytes": _peak_rss_bytes(),
        "workload_peak_rss_delta_bytes": max(
            0, _peak_rss_bytes() - baseline_peak_rss_bytes
        ),
    }


def _child(mode: str, path: Path, items: int, read_cycles: int) -> dict[str, Any]:
    command = [
        sys.executable, str(Path(__file__).resolve()), "--child", "--mode", mode,
        "--path", str(path), "--items", str(items), "--read-cycles", str(read_cycles),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(command, cwd=ROOT, env=env, text=True,
                               capture_output=True, check=True)
    return json.loads(completed.stdout)


def _summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    state_digests = {sample["state_digest"] for sample in samples}
    if len(state_digests) != 1:
        raise RuntimeError("persistent benchmark state drifted between samples")
    metrics: dict[str, Any] = {}
    for workload in WORKLOADS:
        elapsed = [float(sample["metrics"][workload]["elapsed_ms"]) for sample in samples]
        throughput = [float(sample["metrics"][workload]["throughput_ops_per_sec"])
                      for sample in samples]
        metrics[workload] = {
            "median_elapsed_ms": statistics.median(elapsed),
            "p95_elapsed_ms": _percentile(elapsed, 0.95),
            "median_throughput_ops_per_sec": statistics.median(throughput),
        }
    return {
        "sample_count": len(samples),
        "state_digest": state_digests.pop(),
        "metrics": metrics,
        "peak_rss_bytes": statistics.median(
            int(sample["peak_rss_bytes"]) for sample in samples
        ),
        "p95_peak_rss_bytes": _percentile(
            [float(sample["peak_rss_bytes"]) for sample in samples], 0.95
        ),
        "workload_peak_rss_delta_bytes": statistics.median(
            int(sample["workload_peak_rss_delta_bytes"]) for sample in samples
        ),
        "samples": samples,
    }


def run_benchmark(output_root: Path, *, items: int, read_cycles: int,
                  samples: int) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    modes: dict[str, Any] = {}
    for mode in ("python", "rust"):
        mode_samples = []
        for sample in range(samples):
            path = output_root / f"{mode}-{sample}.sqlite3"
            path.unlink(missing_ok=True)
            mode_samples.append(_child(mode, path, items, read_cycles))
        modes[mode] = _summary(mode_samples)
    if modes["python"]["state_digest"] != modes["rust"]["state_digest"]:
        raise RuntimeError("Python and Rust persistent states differ")
    ratios = {
        workload: {
            "throughput_rust_over_python": (
                modes["rust"]["metrics"][workload]["median_throughput_ops_per_sec"]
                / modes["python"]["metrics"][workload]["median_throughput_ops_per_sec"]
            ),
            "p95_latency_rust_over_python": (
                modes["rust"]["metrics"][workload]["p95_elapsed_ms"]
                / modes["python"]["metrics"][workload]["p95_elapsed_ms"]
            ),
        }
        for workload in WORKLOADS
    }
    rss_ratio = (
        modes["rust"]["p95_peak_rss_bytes"]
        / modes["python"]["p95_peak_rss_bytes"]
    )
    gate_checks = {
        "p95_latency_no_worse_than_110_percent": all(
            value["p95_latency_rust_over_python"] <= 1.10
            for value in ratios.values()
        ),
        "throughput_at_least_95_percent": all(
            value["throughput_rust_over_python"] >= 0.95
            for value in ratios.values()
        ),
        "peak_rss_no_worse_than_110_percent": rss_ratio <= 1.10,
    }
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "dataset": DATASET,
        "scope": "persistent authoritative SQLite event and named-projection capabilities",
        "items": items,
        "read_cycles": read_cycles,
        "samples_per_mode": samples,
        "python": sys.version,
        "modes": modes,
        "ratios": ratios,
        "peak_rss_rust_over_python": rss_ratio,
        "gate": {
            "checks": gate_checks,
            "status": "passed" if all(gate_checks.values()) else "failed",
        },
        "interpretation": (
            "Includes current Python-to-PyO3 JSON boundary and one SQLite connection per "
            "capability call; each mode verifies state through its own shipped facade. "
            "Cross-implementation rollback is a separate fresh-process compatibility gate."
        ),
    }


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--items", type=int, default=100)
    parser.add_argument("--read-cycles", type=int, default=20)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=("python", "rust"), help=argparse.SUPPRESS)
    parser.add_argument("--path", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = _args()
    if args.smoke:
        args.items, args.read_cycles, args.samples = 5, 2, 2
    if args.items < 1 or args.read_cycles < 1 or args.samples < 2:
        raise SystemExit("items/read-cycles must be positive and samples must be >= 2")
    if args.child:
        if args.mode is None or args.path is None:
            raise SystemExit("--child requires --mode and --path")
        print(_canonical(_run_mode(args.mode, args.path, items=args.items,
                                   read_cycles=args.read_cycles)))
        return 0
    if args.output_root is None:
        raise SystemExit("--output-root is required")
    report = run_benchmark(args.output_root.resolve(), items=args.items,
                           read_cycles=args.read_cycles, samples=args.samples)
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    print(payload, end="")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
