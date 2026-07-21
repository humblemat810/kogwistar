"""Repeatable ADR-015 Phase-2 isolated-memory inspection benchmark.

This measures the current safety-first native boundary as it exists: each Rust
read reconstructs an isolated store from a JSON snapshot.  It is intentionally
not a benchmark of a future persistent Rust request path.
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any

from kogwistar.engine_core.in_memory_backend import build_in_memory_backend

try:
    import resource
except ImportError:  # pragma: no cover - unavailable on Windows
    resource = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_VERSION = 1
DATASET_VERSION = "rust-memory-phase2-v1"
OPERATIONS = (
    "graph_records",
    "vector_query",
    "replay_events",
    "named_projections",
)
THRESHOLDS = {
    "p95_latency_ratio_max": 1.10,
    "throughput_ratio_min": 0.95,
    "peak_rss_ratio_max": 1.10,
}


def _peak_rss_bytes() -> int:
    if os.name == "nt":
        class _ProcessMemoryCounters(ctypes.Structure):
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

        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        process = kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(
            process, ctypes.byref(counters), counters.cb
        )
        if ok:
            return int(counters.PeakWorkingSetSize)
    if resource is not None:
        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return peak * 1024 if sys.platform != "darwin" else peak
    return 0


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _projection_snapshot(meta: Any, namespaces: tuple[str, ...]) -> list[dict[str, Any]]:
    return [row for namespace in namespaces for row in meta.list_named_projections(namespace)]


def _state_snapshot(backend: Any) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for namespace, collection in (("phase2", backend.node),):
        rows = collection.get(include=["documents", "metadatas", "embeddings"])
        for row_id, document, metadata, embedding in zip(
            rows["ids"],
            rows["documents"],
            rows["metadatas"],
            rows["embeddings"],
            strict=True,
        ):
            records.append(
                {
                    "namespace": namespace,
                    "id": row_id,
                    "document": document,
                    "metadata": metadata,
                    "embedding": embedding,
                }
            )
    meta = backend._engine.meta_sqlite
    events = [
        {
            "namespace": namespace,
            "seq": row.seq,
            "event_id": row.event_id,
            "entity_kind": row.entity_kind,
            "entity_id": row.entity_id,
            "op": row.op,
            "payload": json.loads(row.payload_json),
        }
        for namespace, rows in meta._state.entity_events.items()
        for row in rows
    ]
    cursors = [
        {"namespace": namespace, "consumer": consumer, "last_seq": last_seq}
        for (namespace, consumer), last_seq in meta._state.replay_cursors.items()
    ]
    return {
        "records": records,
        "events": events,
        "cursors": cursors,
        "projections": _projection_snapshot(meta, ("bridge_governance",)),
    }


def _build_dataset(items: int) -> tuple[Any, dict[str, Any]]:
    backend = build_in_memory_backend(SimpleNamespace())
    ids = [f"node-{index:04d}" for index in range(items)]
    embeddings: list[list[float] | None] = []
    metadata: list[dict[str, Any]] = []
    documents: list[str] = []
    for index, row_id in enumerate(ids):
        documents.append(f"document-{row_id}")
        metadata.append({"team": "keep" if index % 4 != 3 else "skip", "rank": index})
        if index == 2:
            embeddings.append(None)
        elif index == 3:
            embeddings.append([1.0, 0.0, 0.0])
        elif index == 4:
            embeddings.append([0.0, 0.0])
        else:
            embeddings.append([1.0, 0.0] if index % 2 == 0 else [0.0, 1.0])
    backend.node.add(ids=ids, documents=documents, metadatas=metadata, embeddings=embeddings)
    meta = backend._engine.meta_sqlite
    for index, row_id in enumerate(ids):
        meta.append_entity_event(
            namespace="phase2",
            event_id=f"evt-{index:04d}",
            entity_kind="node",
            entity_id=row_id,
            op="UPSERT" if index % 2 == 0 else "TOMBSTONE",
            payload_json=_canonical(
                {"replacement": {"id": row_id}} if index % 2 == 0 else {"tombstone": True}
            ),
        )
    meta.cursor_set(namespace="phase2", consumer="benchmark", last_seq=max(0, items // 2))
    meta.replace_named_projection(
        "bridge_governance",
        "z-last",
        {"members": ["a", "b"], "dataset": DATASET_VERSION},
        last_authoritative_seq=items,
        last_materialized_seq=items - 1,
        projection_schema_version=1,
        materialization_status="ready",
    )
    meta.replace_named_projection(
        "bridge_governance",
        "a-first",
        {"active": True, "nested": {"items": [1, {"id": "two"}]}},
        last_authoritative_seq=items,
        last_materialized_seq=items,
        projection_schema_version=1,
        materialization_status="ready",
    )
    # Writers above build real InMemoryMetaStore rows. Freeze only their volatile
    # clock field so fresh-process canonical output digests remain comparable.
    with meta._lock:
        for (namespace, _key), row in meta._state.named_projections.items():
            if namespace == "bridge_governance":
                row["updated_at_ms"] = 1_700_000_000_000
    return backend, _state_snapshot(backend)


def _python_graph_records(backend: Any) -> list[dict[str, Any]]:
    rows = backend.node.get(where={"team": "keep"}, include=["documents", "metadatas", "embeddings"])
    return [
        {"id": row_id, "document": document, "metadata": metadata, "embedding": embedding}
        for row_id, document, metadata, embedding in zip(
            rows["ids"], rows["documents"], rows["metadatas"], rows["embeddings"], strict=True
        )
    ]


def _python_vector_query(backend: Any, limit: int) -> list[dict[str, Any]]:
    rows = backend.node.query(
        query_embeddings=[[1.0, 1.0]],
        where={"team": "keep"},
        n_results=limit,
        include=["documents", "metadatas", "embeddings", "distances"],
    )
    return [
        {
            "record": {"id": row_id, "document": document, "metadata": metadata, "embedding": embedding},
            "distance": distance,
        }
        for row_id, document, metadata, embedding, distance in zip(
            rows["ids"][0],
            rows["documents"][0],
            rows["metadatas"][0],
            rows["embeddings"][0],
            rows["distances"][0],
            strict=True,
        )
    ]


def _python_event_replay(backend: Any) -> list[dict[str, Any]]:
    return [
        {
            "namespace": "phase2",
            "seq": seq,
            "entity_kind": entity_kind,
            "entity_id": entity_id,
            "op": op,
            "payload": json.loads(payload_json),
        }
        for seq, entity_kind, entity_id, op, payload_json in backend._engine.meta_sqlite.iter_entity_events(
            namespace="phase2", from_seq=1
        )
    ]


def _python_cycle(backend: Any, limit: int) -> dict[str, Any]:
    return {
        "graph_records": _python_graph_records(backend),
        "vector_query": _python_vector_query(backend, limit),
        "replay_events": _python_event_replay(backend),
        "named_projections": backend._engine.meta_sqlite.list_named_projections("bridge_governance"),
    }


def _native_cycle(snapshot: dict[str, Any], limit: int) -> dict[str, Any]:
    from kogwistar import _rust

    def native(operation: dict[str, Any]) -> Any:
        return json.loads(_rust.store_memory_read_json(_canonical({"snapshot": snapshot, "operation": operation})))

    events = native({"kind": "replay_events", "namespace": "phase2", "after_seq": 0, "limit": limit})
    return {
        "graph_records": native(
            {"kind": "graph_records", "namespace": "phase2", "metadata": {"team": "keep"}}
        ),
        "vector_query": native(
            {
                "kind": "vector_query",
                "namespace": "phase2",
                "embedding": [1.0, 1.0],
                "limit": limit,
                "metadata": {"team": "keep"},
                "metric": "cosine",
            }
        ),
        "replay_events": [
            {key: value for key, value in event.items() if key != "event_id"}
            for event in events
        ],
        "named_projections": native({"kind": "named_projections", "namespace": "bridge_governance"}),
    }


def _run_child(mode: str, *, items: int, warmups: int, cycles: int) -> dict[str, Any]:
    backend, snapshot = _build_dataset(items)
    before = _digest(_state_snapshot(backend))
    run = (
        (lambda: _python_cycle(backend, items))
        if mode == "python"
        else (lambda: _native_cycle(snapshot, items))
    )
    expected = _python_cycle(backend, items)
    expected_digest = _digest(expected)
    for _ in range(warmups):
        if _digest(run()) != expected_digest:
            raise RuntimeError(f"{mode} warmup output mismatch")
    started = time.perf_counter_ns()
    output_count = 0
    for _ in range(cycles):
        result = run()
        if _digest(result) != expected_digest:
            raise RuntimeError(f"{mode} measured output mismatch")
        output_count += sum(len(result[operation]) for operation in OPERATIONS)
    elapsed_ns = time.perf_counter_ns() - started
    after = _digest(_state_snapshot(backend))
    if before != after:
        raise RuntimeError("benchmark read cycle mutated authoritative Python state")
    return {
        "mode": mode,
        "cycles": cycles,
        "operations_per_cycle": len(OPERATIONS),
        "output_count": output_count,
        "output_digest": expected_digest,
        "cycle_latency_ms": elapsed_ns / cycles / 1_000_000,
        "elapsed_ms": elapsed_ns / 1_000_000,
        "throughput_ops_per_sec": cycles * len(OPERATIONS) * 1_000_000_000 / elapsed_ns,
        "peak_rss_bytes": _peak_rss_bytes(),
        "external_db_queries": 0,
        "authoritative_writes": 0,
    }


def _p95(values: list[float]) -> float:
    if not values:
        raise ValueError("p95 requires samples")
    return sorted(values)[math.ceil(len(values) * 0.95) - 1]


def _summarize(mode: str, samples: list[dict[str, Any]]) -> dict[str, Any]:
    digests = {sample["output_digest"] for sample in samples}
    counts = {sample["output_count"] for sample in samples}
    if len(digests) != 1 or len(counts) != 1:
        raise RuntimeError(f"{mode} samples drifted in output")
    elapsed_seconds = sum(float(sample["elapsed_ms"]) for sample in samples) / 1000.0
    operations = sum(int(sample["cycles"]) * int(sample["operations_per_cycle"]) for sample in samples)
    return {
        "sample_count": len(samples),
        "p95_cycle_latency_ms": _p95([float(sample["cycle_latency_ms"]) for sample in samples]),
        "throughput_ops_per_sec": operations / elapsed_seconds,
        "peak_rss_bytes": max(int(sample["peak_rss_bytes"]) for sample in samples),
        "output_digest": digests.pop(),
        "output_count_per_sample": counts.pop(),
        "external_db_queries": 0,
        "authoritative_writes": 0,
        "samples": samples,
    }


def evaluate_gate(python: dict[str, Any], rust: dict[str, Any]) -> dict[str, Any]:
    ratios = {
        "p95_latency": float(rust["p95_cycle_latency_ms"]) / float(python["p95_cycle_latency_ms"]),
        "throughput": float(rust["throughput_ops_per_sec"]) / float(python["throughput_ops_per_sec"]),
        "peak_rss": float(rust["peak_rss_bytes"]) / float(python["peak_rss_bytes"]),
    }
    checks = {
        "p95_latency": ratios["p95_latency"] <= THRESHOLDS["p95_latency_ratio_max"],
        "throughput": ratios["throughput"] >= THRESHOLDS["throughput_ratio_min"],
        "peak_rss": ratios["peak_rss"] <= THRESHOLDS["peak_rss_ratio_max"],
    }
    gate: dict[str, Any] = {"thresholds": THRESHOLDS, "ratios": ratios, "checks": checks}
    if all(checks.values()):
        gate["status"] = "passed"
    else:
        gate["status"] = "recorded"
        gate["exception"] = {
            "reason": (
                "Current Rust path is read-only shadow-safety inspection that rebuilds an "
                "isolated snapshot per operation; it is not a cutover/default persistent request path."
            ),
            "failed_thresholds": [name for name, passed in checks.items() if not passed],
        }
    return gate


def _child_subprocess(
    mode: str, *, items: int, warmups: int, cycles: int, python_executable: str
) -> dict[str, Any]:
    command = [
        python_executable,
        str(Path(__file__).resolve()),
        "--child",
        "--mode",
        mode,
        "--items",
        str(items),
        "--warmups",
        str(warmups),
        "--cycles",
        str(cycles),
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    completed = subprocess.run(
        command, cwd=ROOT, env=environment, text=True, capture_output=True, check=True
    )
    return json.loads(completed.stdout)


def run_phase2_benchmark(
    *, items: int = 32, warmups: int = 2, samples: int = 5, cycles: int = 20,
    python_executable: str | None = None,
) -> dict[str, Any]:
    if items < 5 or warmups < 0 or samples < 2 or cycles < 1:
        raise ValueError("items >= 5, warmups >= 0, samples >= 2, and cycles >= 1 are required")
    executable = python_executable or sys.executable
    result = {
        mode: _summarize(
            mode,
            [
                _child_subprocess(
                    mode, items=items, warmups=warmups, cycles=cycles, python_executable=executable
                )
                for _ in range(samples)
            ],
        )
        for mode in ("python", "rust")
    }
    if result["python"]["output_digest"] != result["rust"]["output_digest"]:
        raise RuntimeError("Python and Rust benchmark output digests differ")
    if result["python"]["output_count_per_sample"] != result["rust"]["output_count_per_sample"]:
        raise RuntimeError("Python and Rust benchmark output counts differ")
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "dataset": DATASET_VERSION,
        "command": (
            f"{executable} scripts/rust_port_phase2_benchmark.py --items {items} "
            f"--warmups {warmups} --samples {samples} --cycles {cycles} --report <report.json>"
        ),
        "python": sys.version,
        "dataset_items": items,
        "warmups_per_fresh_process": warmups,
        "samples_per_mode": samples,
        "cycles_per_sample": cycles,
        "operations_per_cycle": list(OPERATIONS),
        "database_load": {"kind": "in-memory", "external_queries": 0},
        "authoritative_writes": 0,
        "modes": result,
        "gate": evaluate_gate(result["python"], result["rust"]),
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--cycles", type=int, default=20)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--smoke", action="store_true", help="Use small repeatable CI parameters.")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=("python", "rust"), help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if args.child:
        if args.mode is None:
            raise SystemExit("--child requires --mode")
        print(_canonical(_run_child(args.mode, items=args.items, warmups=args.warmups, cycles=args.cycles)))
        return 0
    if args.smoke:
        args.items, args.warmups, args.samples, args.cycles = 8, 1, 2, 2
    report = run_phase2_benchmark(
        items=args.items, warmups=args.warmups, samples=args.samples, cycles=args.cycles
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    print(payload, end="")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
