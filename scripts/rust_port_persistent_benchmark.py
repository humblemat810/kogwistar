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
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUST_ROOT = ROOT / "rust"
sys.path.insert(0, str(ROOT))

from kogwistar._rust_bridge import store_sqlite
from kogwistar.engine_core.engine_sqlite import EngineSQLite
from kogwistar.engine_core.rust_meta_sqlite import RustEngineSQLite
from kogwistar.engine_core.sqlite_context import sqlite_execution_context
from mixed_workload import (
    WorkloadOperation,
    read_trace,
    write_trace,
)

BENCHMARK_VERSION = 3
DATASET = "kogwistar-mixed-rewrite-workload-v2"
WORKLOADS = (
    "event_append",
    "event_replay",
    "projection_replace",
    "projection_list",
    "mixed_interleaved",
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


def _database_footprint_bytes(path: Path) -> int:
    return sum(
        candidate.stat().st_size
        for candidate in (
            path,
            Path(f"{path}-wal"),
            Path(f"{path}-shm"),
        )
        if candidate.is_file()
    )


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * percentile) - 1)]


def _sample_variation(values: list[float]) -> dict[str, float]:
    mean = statistics.mean(values)
    deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "mean": mean,
        "stdev": deviation,
        "coefficient_of_variation_pct": deviation * 100 / mean if mean else 0.0,
        "normal_approx_95pct_ci_low": mean - 1.96 * deviation / math.sqrt(len(values)),
        "normal_approx_95pct_ci_high": mean + 1.96 * deviation / math.sqrt(len(values)),
        "min": min(values),
        "max": max(values),
    }


def _measure(operation: Callable[[], int]) -> dict[str, float | int]:
    started = time.perf_counter_ns()
    cpu_started = time.process_time_ns()
    count = operation()
    cpu_elapsed_ns = time.process_time_ns() - cpu_started
    elapsed_ns = time.perf_counter_ns() - started
    return {
        "elapsed_ms": elapsed_ns / 1_000_000,
        "cpu_elapsed_ms": cpu_elapsed_ns / 1_000_000,
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


def _run_mode(
    mode: str,
    path: Path,
    *,
    items: int,
    read_cycles: int,
    trace: list[WorkloadOperation],
) -> dict[str, Any]:
    if mode == "rust_cached":
        EngineSQLite(path.parent, filename=path.name).ensure_initialized()
        with sqlite_execution_context("rust"):
            return _run_mode_bound(
                mode, path, items=items, read_cycles=read_cycles, trace=trace,
                initialize_python_store=False,
            )
    return _run_mode_bound(
        mode, path, items=items, read_cycles=read_cycles, trace=trace
    )


def _run_mode_bound(
    mode: str,
    path: Path,
    *,
    items: int,
    read_cycles: int,
    trace: list[WorkloadOperation],
    initialize_python_store: bool = True,
) -> dict[str, Any]:
    namespace = "benchmark"
    projection_namespace = "benchmark_projection"
    python_store = EngineSQLite(path.parent, filename=path.name)
    if initialize_python_store:
        python_store.ensure_initialized()
    baseline_peak_rss_bytes = _peak_rss_bytes()

    def call(operation: dict[str, Any]) -> Any:
        return store_sqlite(
            path=path,
            operation=operation,
            reuse_session=mode == "rust_cached",
        )

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

        def mixed_interleaved() -> int:
            operations = 0
            for step in trace:
                index = step["index"]
                entity_id = step["expected_entity_id"]
                projection_key = f"mixed-key-{index:05d}"
                if step["operation"] == "write_record":
                    with python_store.transaction():
                        python_store.append_entity_event(
                            namespace="mixed_interleaved",
                            event_id=f"mixed-event-{index:05d}",
                            entity_kind="node",
                            entity_id=entity_id,
                            op="UPSERT",
                            payload_json=_canonical({"id": entity_id, "revision": 1}),
                        )
                        python_store.replace_named_projection(
                            "mixed_interleaved_projection",
                            projection_key,
                            **_projection(index),
                        )
                elif step["operation"] == "read_event":
                    rows = list(python_store.iter_entity_events(
                        namespace="mixed_interleaved",
                        from_seq=index + 1,
                        batch_size=1,
                    ))
                    if not rows or rows[0][2] != entity_id:
                        raise RuntimeError("interleaved event read did not observe its write")
                elif step["operation"] == "read_projection":
                    projection = python_store.get_named_projection(
                        "mixed_interleaved_projection", projection_key
                    )
                    payload = projection.get("payload") if projection is not None else None
                    if not isinstance(payload, dict) or payload.get("index") != index:
                        raise RuntimeError("interleaved projection read did not observe its write")
                operations += 1
            return operations
    else:
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

        def mixed_interleaved() -> int:
            if mode == "rust_cached":
                # The cached operation session and the transactional facade
                # intentionally cannot own one SQLite execution context at the
                # same time. Close the raw session before opening UoWs; this
                # workload measures the supported atomic transaction path.
                call({"kind": "close"})
            rust_store = RustEngineSQLite(path.parent, filename=path.name)
            operations = 0
            for step in trace:
                index = step["index"]
                entity_id = step["expected_entity_id"]
                projection_key = f"mixed-key-{index:05d}"
                if step["operation"] == "write_record":
                    with rust_store.transaction():
                        rust_store.append_entity_event(
                            namespace="mixed_interleaved",
                            event_id=f"mixed-event-{index:05d}",
                            entity_kind="node",
                            entity_id=entity_id,
                            op="UPSERT",
                            payload_json=_canonical({"id": entity_id, "revision": 1}),
                        )
                        rust_store.replace_named_projection(
                            "mixed_interleaved_projection",
                            projection_key,
                            **_projection(index),
                        )
                elif step["operation"] == "read_event":
                    rows = list(rust_store.iter_entity_events(
                        namespace="mixed_interleaved",
                        from_seq=index + 1,
                        batch_size=1,
                    ))
                    if not rows or rows[0][2] != entity_id:
                        raise RuntimeError("interleaved Rust event read did not observe its write")
                elif step["operation"] == "read_projection":
                    projection = rust_store.get_named_projection(
                        "mixed_interleaved_projection",
                        projection_key,
                    )
                    payload = projection.get("payload") if projection is not None else None
                    if not isinstance(payload, dict) or payload.get("index") != index:
                        raise RuntimeError("interleaved Rust projection read did not observe its write")
                operations += 1
            return operations

    metrics = {
        "event_append": _measure(append_events),
        "event_replay": _measure(replay_events),
        "projection_replace": _measure(replace_projections),
        "projection_list": _measure(list_projections),
        "mixed_interleaved": _measure(mixed_interleaved),
    }
    if mode == "python":
        events = [
            list(row)
            for row in python_store.iter_entity_events(
                namespace=namespace, from_seq=1, batch_size=items + 1
            )
        ]
        projections = python_store.list_named_projections(projection_namespace)
        mixed_store = python_store
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
        if mode == "rust_cached":
            call({"kind": "close"})
        mixed_store = RustEngineSQLite(path.parent, filename=path.name)
    mixed_events = [
        list(row)
        for row in mixed_store.iter_entity_events(
            namespace="mixed_interleaved", from_seq=1, batch_size=items + 1
        )
    ]
    mixed_projections = mixed_store.list_named_projections(
        "mixed_interleaved_projection"
    )
    canonical_state = {
        "events": events,
        "projections": [
            {key: value for key, value in row.items() if key != "updated_at_ms"}
            for row in projections
        ],
        "mixed_events": mixed_events,
        "mixed_projections": [
            {key: value for key, value in row.items() if key != "updated_at_ms"}
            for row in mixed_projections
        ],
    }
    return {
        "mode": mode,
        "metrics": metrics,
        "state_digest": _digest(canonical_state),
        "event_count": len(events),
        "projection_count": len(projections),
        "database_bytes": _database_footprint_bytes(path),
        "peak_rss_bytes": _peak_rss_bytes(),
        "workload_peak_rss_delta_bytes": max(
            0, _peak_rss_bytes() - baseline_peak_rss_bytes
        ),
    }


def _child(
    mode: str, path: Path, items: int, read_cycles: int, trace_path: Path
) -> dict[str, Any]:
    if mode == "rust_native":
        executable = RUST_ROOT / "target" / "release" / "examples" / (
            "sqlite_native_authority_benchmark.exe"
            if sys.platform == "win32"
            else "sqlite_native_authority_benchmark"
        )
        completed = subprocess.run(
            [str(executable), str(path), str(items), str(read_cycles), str(trace_path)],
            cwd=RUST_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        result = json.loads(completed.stdout)
        native_state = result.pop("state")
        python_store = EngineSQLite(path.parent, filename=path.name)
        python_state = {
            "events": [
                list(row)
                for row in python_store.iter_entity_events(
                    namespace="benchmark", from_seq=1, batch_size=items + 1
                )
            ],
            "projections": [
                {key: value for key, value in row.items() if key != "updated_at_ms"}
                for row in python_store.list_named_projections("benchmark_projection")
            ],
            "mixed_events": [
                list(row)
                for row in python_store.iter_entity_events(
                    namespace="mixed_interleaved", from_seq=1, batch_size=items + 1
                )
            ],
            "mixed_projections": [
                {key: value for key, value in row.items() if key != "updated_at_ms"}
                for row in python_store.list_named_projections(
                    "mixed_interleaved_projection"
                )
            ],
        }
        if _digest(native_state) != _digest(python_state):
            raise RuntimeError("Rust-native state differs after Python reopens the database")
        return {
            "mode": mode,
            "metrics": result["metrics"],
            "state_digest": _digest(python_state),
            "event_count": len(python_state["events"]),
            "projection_count": len(python_state["projections"]),
            "database_bytes": result["database_bytes"],
            "peak_rss_bytes": None,
            "workload_peak_rss_delta_bytes": None,
        }
    command = [
        sys.executable, str(Path(__file__).resolve()), "--child", "--mode", mode,
        "--path", str(path), "--items", str(items), "--read-cycles", str(read_cycles),
        "--trace", str(trace_path),
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
        cpu_values = [sample["metrics"][workload].get("cpu_elapsed_ms") for sample in samples]
        cpu_elapsed = [float(value) for value in cpu_values if value is not None]
        throughput = [float(sample["metrics"][workload]["throughput_ops_per_sec"])
                      for sample in samples]
        metrics[workload] = {
            "median_elapsed_ms": statistics.median(elapsed),
            "elapsed_sample_variation": _sample_variation(elapsed),
            "p95_elapsed_ms": _percentile(elapsed, 0.95),
            "p99_elapsed_ms": _percentile(elapsed, 0.99),
            "median_cpu_elapsed_ms": (
                statistics.median(cpu_elapsed)
                if len(cpu_elapsed) == len(samples)
                else None
            ),
            "median_throughput_ops_per_sec": statistics.median(throughput),
            "throughput_sample_variation": _sample_variation(throughput),
        }
    peak_rss_values = [sample["peak_rss_bytes"] for sample in samples]
    workload_rss_values = [sample["workload_peak_rss_delta_bytes"] for sample in samples]
    has_rss = all(value is not None for value in peak_rss_values)
    return {
        "sample_count": len(samples),
        "state_digest": state_digests.pop(),
        "metrics": metrics,
        "peak_rss_bytes": (
            statistics.median(int(value) for value in peak_rss_values) if has_rss else None
        ),
        "p95_peak_rss_bytes": (
            _percentile([float(value) for value in peak_rss_values], 0.95)
            if has_rss else None
        ),
        "workload_peak_rss_delta_bytes": (
            statistics.median(int(value) for value in workload_rss_values)
            if all(value is not None for value in workload_rss_values) else None
        ),
        "samples": samples,
    }


def run_benchmark(output_root: Path, *, items: int, read_cycles: int,
                  samples: int) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    trace_path = output_root / "read_after_write_trace.v1.json"
    trace = write_trace(trace_path, items)
    subprocess.run(
        [
            "cargo", "build", "--release", "-p", "kogwistar-store-sqlite",
            "--example", "sqlite_native_authority_benchmark",
        ],
        cwd=RUST_ROOT,
        check=True,
    )
    modes: dict[str, Any] = {}
    for mode in ("python", "rust", "rust_cached", "rust_native"):
        warmup_path = output_root / f"{mode}-warmup.sqlite3"
        for suffix in ("", "-wal", "-shm"):
            Path(f"{warmup_path}{suffix}").unlink(missing_ok=True)
        warmup = _child(mode, warmup_path, items, read_cycles, trace_path)
        mode_samples = []
        for sample in range(samples):
            path = output_root / f"{mode}-{sample}.sqlite3"
            for suffix in ("", "-wal", "-shm"):
                Path(f"{path}{suffix}").unlink(missing_ok=True)
            mode_samples.append(_child(mode, path, items, read_cycles, trace_path))
        modes[mode] = _summary(mode_samples)
        if warmup["state_digest"] != modes[mode]["state_digest"]:
            raise RuntimeError(f"warm-up state differs for {mode}")
        modes[mode]["warmup"] = warmup
    expected_digest = modes["python"]["state_digest"]
    if any(mode["state_digest"] != expected_digest for mode in modes.values()):
        raise RuntimeError("Python and Rust persistent states differ")
    ratios = {
        mode_name: {
            workload: {
                "throughput_over_python": (
                    mode["metrics"][workload]["median_throughput_ops_per_sec"]
                    / modes["python"]["metrics"][workload]["median_throughput_ops_per_sec"]
                ),
                "p95_latency_over_python": (
                    mode["metrics"][workload]["p95_elapsed_ms"]
                    / modes["python"]["metrics"][workload]["p95_elapsed_ms"]
                ),
            }
            for workload in WORKLOADS
        }
        for mode_name, mode in modes.items()
        if mode_name != "python"
    }
    rss_ratio = (
        modes["rust"]["p95_peak_rss_bytes"]
        / modes["python"]["p95_peak_rss_bytes"]
    )
    gate_checks = {
        "p95_latency_no_worse_than_110_percent": all(
            value["p95_latency_over_python"] <= 1.10
            for value in ratios["rust"].values()
        ),
        "throughput_at_least_95_percent": all(
            value["throughput_over_python"] >= 0.95
            for value in ratios["rust"].values()
        ),
        "peak_rss_no_worse_than_110_percent": rss_ratio <= 1.10,
    }
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "dataset": DATASET,
        "scope": (
            "persistent authoritative SQLite event/projection operations plus a "
            "versioned sequential read-after-write trace"
        ),
        "items": items,
        "read_cycles": read_cycles,
        "samples_per_mode": samples,
        "warmup_samples_per_mode": 1,
        "workload_trace_sha256": _digest(trace),
        "workload_trace_path": str(trace_path),
        "python": sys.version,
        "modes": modes,
        "ratios": ratios,
        "peak_rss_rust_over_python": rss_ratio,
        "gate": {
            "checks": gate_checks,
            "status": "passed" if all(gate_checks.values()) else "failed",
        },
        "interpretation": (
            "rust_native calls SqliteStore directly in one Rust process; Python only "
            "launches the executable and verifies its final state after it exits. "
            "Each native write still uses its own store transaction/commit. rust_cached "
            "keeps the PyO3/JSON boundary per operation but reuses one scoped session. "
            "The regular rust gate compares the existing per-call Rust facade with Python."
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
    parser.add_argument(
        "--mode", choices=("python", "rust", "rust_cached"), help=argparse.SUPPRESS
    )
    parser.add_argument("--path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--trace", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = _args()
    if args.smoke:
        args.items, args.read_cycles, args.samples = 5, 2, 2
    if args.items < 1 or args.read_cycles < 1 or args.samples < 2:
        raise SystemExit("items/read-cycles must be positive and samples must be >= 2")
    if args.child:
        if args.mode is None or args.path is None or args.trace is None:
            raise SystemExit("--child requires --mode, --path, and --trace")
        print(_canonical(_run_mode(args.mode, args.path, items=args.items,
                                   read_cycles=args.read_cycles,
                                   trace=read_trace(args.trace))))
        return 0
    if args.output_root is None:
        raise SystemExit("--output-root is required")
    report = run_benchmark(args.output_root.resolve(), items=args.items,
                           read_cycles=args.read_cycles, samples=args.samples)
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    print(payload, end="")
    report_path = args.report or (args.output_root.resolve() / "report.v3.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
