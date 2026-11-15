"""Benchmark SQLite metadata and real Chroma/pgvector vector retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kogwistar.engine_core.engine_sqlite import EngineSQLite


def _rss_bytes_from_statm(statm: str, page_size: int) -> int:
    fields = statm.split()
    if len(fields) < 2 or page_size <= 0:
        raise ValueError("invalid Linux /proc statm data")
    return int(fields[1]) * page_size


def process_rss_bytes() -> int:
    """Return current process RSS without requiring a native metrics package."""
    if sys.platform.startswith("linux"):
        statm = Path("/proc/self/statm").read_text(encoding="ascii")
        return _rss_bytes_from_statm(statm, os.sysconf("SC_PAGE_SIZE"))

    if sys.platform == "win32":
        import ctypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("NonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_bool

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(),
            ctypes.byref(counters),
            counters.cb,
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(counters.WorkingSetSize)

    try:
        import psutil
    except ImportError as exc:
        raise RuntimeError("RSS measurement is unsupported on this platform") from exc
    return int(psutil.Process().memory_info().rss)


try:
    from .mixed_workload import (
        VECTOR_DIMENSION,
        WorkloadOperation,
        ROOT_NODE_ID,
        build_graph_payloads,
        build_root_graph_payload,
        build_normalized_vector,
        graph_edge_id,
        graph_node_id,
        normalized_vector_digest,
        write_trace,
    )
except ImportError:
    from mixed_workload import (
        VECTOR_DIMENSION,
        WorkloadOperation,
        ROOT_NODE_ID,
        build_graph_payloads,
        build_root_graph_payload,
        build_normalized_vector,
        graph_edge_id,
        graph_node_id,
        normalized_vector_digest,
        write_trace,
    )


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, int(len(ordered) * percentile + 0.999999) - 1)]


def _sample_variation(values: list[float]) -> dict[str, float | None]:
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


def _sqlite_bytes(path: Path) -> int:
    return sum(
        candidate.stat().st_size
        for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm"))
        if candidate.is_file()
    )


def _persistent_bytes(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def _vector_query_indexes(items: int, maximum_queries: int) -> list[int]:
    if items < 1 or maximum_queries < 1:
        raise ValueError("items and maximum vector queries must be positive")
    if items <= maximum_queries:
        return list(range(items))
    if maximum_queries == 1:
        return [items - 1]
    return sorted(
        {round(index * (items - 1) / (maximum_queries - 1)) for index in range(maximum_queries)}
    )


def _make_vector_backend(
    backend_name: Literal["sqlite_event_projection", "chroma_sqlite_metadata", "postgres_pgvector"],
    run_root: Path,
    *,
    postgres_dsn: str | None,
    schema: str,
) -> tuple[Any | None, dict[str, Any], Any]:
    if backend_name == "sqlite_event_projection":
        return None, {"backend": "SQLite event/projection only", "metric": None}, None
    if backend_name == "chroma_sqlite_metadata":
        try:
            import chromadb  # type: ignore[reportMissingImports]
        except ImportError as exc:
            raise RuntimeError(
                "Chroma backend was requested but chromadb is unavailable; install kogwistar[chroma]"
            ) from exc
        os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
        persist_dir = run_root / "chroma"
        client = chromadb.PersistentClient(path=str(persist_dir))
        collection = client.get_or_create_collection(
            name="mixed_rewrite_nodes",
            metadata={"hnsw:space": "cosine"},
        )
        return collection, {
            "backend": "Chroma PersistentClient vector collection with Chroma SQLite metadata; app event/projection SQLite",
            "metric": "cosine",
            "vector_path": str(persist_dir),
        }, client
    if not postgres_dsn:
        raise RuntimeError(
            "PostgreSQL backend was requested but --postgres-dsn was not provided"
        )
    try:
        import sqlalchemy as sa  # type: ignore[reportMissingImports]
        from kogwistar.engine_core.postgres_backend import PgVectorBackend
    except ImportError as exc:
        raise RuntimeError(
            "PostgreSQL/pgvector backend was requested but sqlalchemy or pgvector is unavailable"
        ) from exc
    engine = sa.create_engine(postgres_dsn, pool_size=6, pool_pre_ping=True)
    backend = PgVectorBackend(engine=engine, embedding_dim=VECTOR_DIMENSION, schema=schema)
    backend.ensure_schema()
    return backend, {
        "backend": "Kogwistar PgVectorBackend; pgvector node table, app event/projection SQLite",
        "metric": "cosine",
        "schema": schema,
        "vector_dimension": VECTOR_DIMENSION,
    }, engine


def _vector_upsert(backend_name: str, backend: Any | None, *, index: int, vector: list[float]) -> None:
    entity_id = graph_node_id(index)
    document = f"Deterministic benchmark node {entity_id}"
    metadata = {"entity_id": entity_id, "revision": index + 1}
    if backend_name == "sqlite_event_projection":
        return
    if backend is None:
        raise RuntimeError(f"{backend_name} vector backend was not initialized")
    if backend_name == "chroma_sqlite_metadata":
        backend.add(
            ids=[entity_id],
            documents=[document],
            metadatas=[metadata],
            embeddings=[vector],
        )
        return
    backend.node_add(
        ids=[entity_id],
        documents=[document],
        metadatas=[metadata],
        embeddings=[vector],
    )


def _vector_query(
    backend_name: str,
    backend: Any | None,
    *,
    expected_entity_id: str,
    vector: list[float],
) -> dict[str, object]:
    if backend_name == "sqlite_event_projection":
        return {"semantic_available": False, "reason": "SQLite event/projection config has no vector index"}
    if backend is None:
        raise RuntimeError(f"{backend_name} vector backend was not initialized")
    if backend_name == "chroma_sqlite_metadata":
        result: Any = backend.query(query_embeddings=[vector], n_results=1)
        found = result["ids"][0][0] if result.get("ids") and result["ids"][0] else None
        distance = result["distances"][0][0] if result.get("distances") else None
    else:
        result = backend.node_query(
            query_embeddings=[vector], n_results=1, include=["distances", "metadatas"]
        )
        found = result["ids"][0][0] if result.get("ids") and result["ids"][0] else None
        distances = result.get("distances")
        distance = distances[0][0] if distances and distances[0] else None
    if found != expected_entity_id:
        raise RuntimeError(
            f"semantic nearest-neighbor mismatch: expected {expected_entity_id!r}, found {found!r}"
        )
    return {"semantic_available": True, "entity_id": found, "distance": distance}


def _write_graph_projection(
    store: EngineSQLite,
    *,
    entity_kind: Literal["node", "edge"],
    entity_id: str,
    payload: dict[str, object],
    seq: int,
) -> None:
    store.replace_named_projection(
        "backend_benchmark_projection",
        f"{entity_kind}:{entity_id}",
        {"entity_kind": entity_kind, "entity_id": entity_id, "entity": payload},
        last_authoritative_seq=seq,
        last_materialized_seq=seq,
        projection_schema_version=1,
        materialization_status="ready",
    )


def _seed_graph_root(store: EngineSQLite) -> None:
    root = build_root_graph_payload()
    with store.transaction():
        seq = store.append_entity_event(
            namespace="backend_benchmark",
            event_id="backend-root-node",
            entity_kind="node",
            entity_id=ROOT_NODE_ID,
            op="UPSERT",
            payload_json=_canonical(root),
        )
        _write_graph_projection(
            store, entity_kind="node", entity_id=ROOT_NODE_ID, payload=root, seq=seq
        )


def _final_state(store: EngineSQLite, items: int) -> dict[str, list[Any]]:
    events = [
        list(row)
        for row in store.iter_entity_events(
            namespace="backend_benchmark", from_seq=1, batch_size=1 + 2 * items
        )
    ]
    projections = [
        {key: value for key, value in row.items() if key != "updated_at_ms"}
        for row in store.list_named_projections("backend_benchmark_projection")
    ]
    return {"events": events, "projections": projections}


def _run_sample(
    run_root: Path,
    *,
    backend_name: Literal["sqlite_event_projection", "chroma_sqlite_metadata", "postgres_pgvector"],
    items: int,
    trace: list[WorkloadOperation],
    vector_query_indexes: set[int],
    postgres_dsn: str | None,
    sample_id: str,
) -> dict[str, Any]:
    process_rss_start = process_rss_bytes()
    run_root.mkdir(parents=True, exist_ok=True)
    if any(run_root.iterdir()):
        raise FileExistsError(f"backend sample directory must be empty: {run_root}")
    sqlite_path = run_root / "app-meta.sqlite3"
    meta = EngineSQLite(run_root, filename=sqlite_path.name)
    vector_prepare_started = time.perf_counter_ns()
    vectors = [build_normalized_vector(index) for index in range(items)]
    vector_digests = [normalized_vector_digest(vector) for vector in vectors]
    vector_prepare_ms = (time.perf_counter_ns() - vector_prepare_started) / 1_000_000
    graph_prepare_started = time.perf_counter_ns()
    graph_payloads = [
        build_graph_payloads(index, vector_sha256=vector_digests[index])
        for index in range(items)
    ]
    graph_prepare_ms = (time.perf_counter_ns() - graph_prepare_started) / 1_000_000
    startup_started = time.perf_counter_ns()
    meta.ensure_initialized()
    schema = f"bench_{uuid.uuid4().hex[:12]}"
    vector_backend, backend_info, backend_owner = _make_vector_backend(
        backend_name,
        run_root,
        postgres_dsn=postgres_dsn,
        schema=schema,
    )
    if backend_name != "sqlite_event_projection" and vector_backend is None:
        raise RuntimeError(f"{backend_name} vector backend initialization returned no backend")
    _seed_graph_root(meta)
    startup_ms = (time.perf_counter_ns() - startup_started) / 1_000_000

    operation_latencies: dict[str, list[float]] = {
        "write_record": [],
        "read_event": [],
        "read_projection": [],
        "vector_upsert": [],
        "vector_query": [],
    }
    operation_count = 0
    elapsed_started = time.perf_counter_ns()
    cpu_started = time.process_time_ns()
    for step in trace:
        index = step["index"]
        entity_id = step["expected_entity_id"]
        operation = step["operation"]
        started = time.perf_counter_ns()
        if operation == "write_record":
            node_payload, edge_payload = graph_payloads[index]
            edge_id = graph_edge_id(index)
            with meta.transaction():
                node_seq = meta.append_entity_event(
                    namespace="backend_benchmark",
                    event_id=f"backend-node-event-{index:05d}",
                    entity_kind="node",
                    entity_id=entity_id,
                    op="UPSERT",
                    payload_json=_canonical(node_payload),
                )
                edge_seq = meta.append_entity_event(
                    namespace="backend_benchmark",
                    event_id=f"backend-edge-event-{index:05d}",
                    entity_kind="edge",
                    entity_id=edge_id,
                    op="UPSERT",
                    payload_json=_canonical(edge_payload),
                )
                _write_graph_projection(
                    meta,
                    entity_kind="node",
                    entity_id=entity_id,
                    payload=node_payload,
                    seq=node_seq,
                )
                _write_graph_projection(
                    meta,
                    entity_kind="edge",
                    entity_id=edge_id,
                    payload=edge_payload,
                    seq=edge_seq,
                )
            operation_latencies[operation].append(
                (time.perf_counter_ns() - started) / 1_000_000
            )
            started = time.perf_counter_ns()
            _vector_upsert(backend_name, vector_backend, index=index, vector=vectors[index])
            operation_latencies["vector_upsert"].append(
                (time.perf_counter_ns() - started) / 1_000_000
            )
            operation_count += 1
        elif operation == "read_event":
            edge_id = graph_edge_id(index)
            node_seq = 2 + 2 * index
            rows = list(
                meta.iter_entity_events(
                    namespace="backend_benchmark",
                    from_seq=node_seq,
                    to_seq=node_seq + 1,
                    batch_size=2,
                )
            )
            if (
                len(rows) != 2
                or (rows[0][1], rows[0][2]) != ("node", entity_id)
                or (rows[1][1], rows[1][2]) != ("edge", edge_id)
            ):
                raise RuntimeError("backend graph event read violated read-after-write")
            node_payload = json.loads(rows[0][4])
            edge_payload = json.loads(rows[1][4])
            if (
                not isinstance(node_payload, dict)
                or node_payload.get("id") != entity_id
                or not isinstance(edge_payload, dict)
                or edge_payload.get("source_ids") != [step["expected_source_id"]]
                or edge_payload.get("target_ids") != [entity_id]
            ):
                raise RuntimeError("backend graph edge endpoints did not match the trace")
            operation_latencies[operation].append(
                (time.perf_counter_ns() - started) / 1_000_000
            )
            operation_count += 1
        else:
            edge_id = graph_edge_id(index)
            node_projection = meta.get_named_projection(
                "backend_benchmark_projection", f"node:{entity_id}"
            )
            edge_projection = meta.get_named_projection(
                "backend_benchmark_projection", f"edge:{edge_id}"
            )
            node_payload = node_projection.get("payload") if node_projection else None
            edge_payload = edge_projection.get("payload") if edge_projection else None
            if (
                not isinstance(node_payload, dict)
                or node_payload.get("entity_id") != entity_id
                or not isinstance(node_payload.get("entity"), dict)
                or node_payload["entity"].get("id") != entity_id
                or not isinstance(edge_payload, dict)
                or not isinstance(edge_payload.get("entity"), dict)
                or edge_payload["entity"].get("id") != edge_id
                or edge_payload["entity"].get("source_ids")
                != [step["expected_source_id"]]
                or edge_payload["entity"].get("target_ids") != [entity_id]
            ):
                raise RuntimeError("backend graph projection read violated graph contract")
            operation_latencies[operation].append(
                (time.perf_counter_ns() - started) / 1_000_000
            )
            if (
                backend_name != "sqlite_event_projection"
                and index in vector_query_indexes
            ):
                started = time.perf_counter_ns()
                _vector_query(
                    backend_name,
                    vector_backend,
                    expected_entity_id=entity_id,
                    vector=vectors[index],
                )
                operation_latencies["vector_query"].append(
                    (time.perf_counter_ns() - started) / 1_000_000
                )
            operation_count += 1
    elapsed_ns = time.perf_counter_ns() - elapsed_started
    cpu_elapsed_ns = time.process_time_ns() - cpu_started

    reopened_meta = EngineSQLite(run_root, filename=sqlite_path.name)
    reopened_meta.ensure_initialized()
    state = _final_state(reopened_meta, items)
    if len(state["events"]) != 1 + 2 * items or len(state["projections"]) != 1 + 2 * items:
        raise RuntimeError("backend benchmark lost authoritative event/projection data")
    vector_count = items if backend_name != "sqlite_event_projection" else 0
    backend_reopen_verified = True
    if backend_name == "chroma_sqlite_metadata":
        vector_backend, backend_info, backend_owner = _make_vector_backend(
            backend_name,
            run_root,
            postgres_dsn=postgres_dsn,
            schema=schema,
        )
        if vector_backend is None:
            raise RuntimeError("Chroma reopen returned no collection")
        vector_count = int(vector_backend.count())
        reopened = vector_backend.query(query_embeddings=[vectors[-1]], n_results=1)
        if reopened.get("ids", [[]])[0] != [graph_node_id(items - 1)]:
            raise RuntimeError("Chroma vector result did not survive PersistentClient reopen")
    elif backend_name == "postgres_pgvector":
        if vector_backend is None:
            raise RuntimeError("PostgreSQL vector backend disappeared before reopen")
        found = vector_backend.node_get(ids=[graph_node_id(items - 1)])
        if found.get("ids") != [graph_node_id(items - 1)]:
            raise RuntimeError("PostgreSQL backend did not reopen/read final vector node")
        vector_backend.close()
        vector_backend, backend_info, backend_owner = _make_vector_backend(
            backend_name,
            run_root,
            postgres_dsn=postgres_dsn,
            schema=schema,
        )
        if vector_backend is None:
            raise RuntimeError("PostgreSQL reopen returned no vector backend")
        reopened = vector_backend.node_query(
            query_embeddings=[vectors[-1]], n_results=1, include=["distances"]
        )
        if reopened.get("ids", [[]])[0] != [graph_node_id(items - 1)]:
            raise RuntimeError("PostgreSQL vector result did not survive backend reopen")
        vector_backend.close()
        vector_count = items
    if vector_count != items and backend_name != "sqlite_event_projection":
        raise RuntimeError(f"vector backend stored {vector_count} records, expected {items}")

    backend_disk_bytes = (
        _persistent_bytes(run_root / "chroma")
        if backend_name == "chroma_sqlite_metadata"
        else None
    )
    if backend_name == "postgres_pgvector":
        backend_disk_bytes = 0
        with backend_owner.connect() as connection:
            backend_disk_bytes = int(
                connection.exec_driver_sql(
                    "SELECT COALESCE(sum(pg_total_relation_size("
                    "(quote_ident(schemaname)||'.'||quote_ident(tablename))::regclass)),0) "
                    "FROM pg_tables WHERE schemaname = %s",
                    (schema,),
                ).scalar_one()
            )
        backend_owner.dispose()
    return {
        "backend": backend_info,
        "sample_id": sample_id,
        "items": items,
        "state_digest": _digest(state),
        "vector_digest": _digest(vector_digests),
        "vector_count": vector_count,
        "backend_reopen_verified": backend_reopen_verified,
        "startup_ms": startup_ms,
        "input_prepare_ms": vector_prepare_ms,
        "graph_payload_prepare_ms": graph_prepare_ms,
        "elapsed_ms": elapsed_ns / 1_000_000,
        "cpu_elapsed_ms": cpu_elapsed_ns / 1_000_000,
        "process_rss_start_bytes": process_rss_start,
        "process_rss_end_bytes": process_rss_bytes(),
        "process_rss_delta_bytes": process_rss_bytes() - process_rss_start,
        "throughput_ops_per_sec": operation_count * 1_000_000_000 / elapsed_ns,
        "operation_count": operation_count,
        "vector_query_count": len(operation_latencies["vector_query"]),
        "operations": {
            name: {
                "count": len(values),
                "p50_ms": _percentile(values, 0.50) if values else None,
                "p95_ms": _percentile(values, 0.95) if values else None,
                "p99_ms": _percentile(values, 0.99) if values else None,
            }
            for name, values in operation_latencies.items()
        },
        "app_sqlite_bytes": _sqlite_bytes(sqlite_path),
        "backend_disk_bytes": backend_disk_bytes,
    }


def run_benchmark(
    output: Path,
    *,
    items: int,
    samples: int,
    postgres_dsn: str | None,
    backends: tuple[Literal["sqlite_event_projection", "chroma_sqlite_metadata", "postgres_pgvector"], ...],
    vector_query_indexes: list[int],
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:12]
    run_root = output / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    trace_path = run_root / "backend_trace.v1.json"
    trace = write_trace(trace_path, items)
    trace_sha256 = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    results: dict[str, Any] = {}
    state_digests: set[str] = set()
    vector_digests: set[str] = set()
    warmups: dict[str, Any] = {}
    for backend in backends:
        print(f"[mixed-backend] warm-up: {backend} ({items} records)", file=sys.stderr, flush=True)
        warmup_root = run_root / f"warmup-{backend}"
        if any(warmup_root.iterdir()) if warmup_root.exists() else False:
            raise FileExistsError(f"warm-up sample directory must be empty: {warmup_root}")
        warmups[backend] = _run_sample(
            warmup_root,
            backend_name=backend,
            items=items,
            trace=trace,
            vector_query_indexes=set(vector_query_indexes),
            postgres_dsn=postgres_dsn,
            sample_id=f"warmup-{backend}",
        )
        runs = []
        for sample in range(samples):
            print(
                f"[mixed-backend] sample {sample + 1}/{samples}: {backend} ({items} records)",
                file=sys.stderr,
                flush=True,
            )
            sample_root = run_root / f"{backend}-{sample}"
            sample_root.mkdir(parents=True, exist_ok=False)
            runs.append(
                _run_sample(
                    sample_root,
                    backend_name=backend,
                    items=items,
                    trace=trace,
                    vector_query_indexes=set(vector_query_indexes),
                    postgres_dsn=postgres_dsn,
                    sample_id=f"{backend}-{sample}",
                )
            )
            state_digests.add(runs[-1]["state_digest"])
            vector_digests.add(runs[-1]["vector_digest"])
        print(f"[mixed-backend] complete: {backend}", file=sys.stderr, flush=True)
        results[backend] = {
            "state_digest": runs[0]["state_digest"],
            "vector_digest": runs[0]["vector_digest"],
            "median_elapsed_ms": statistics.median(run["elapsed_ms"] for run in runs),
            "p95_elapsed_ms": _percentile([run["elapsed_ms"] for run in runs], 0.95),
            "p99_elapsed_ms": _percentile([run["elapsed_ms"] for run in runs], 0.99),
            "median_startup_ms": statistics.median(run["startup_ms"] for run in runs),
            "median_cpu_elapsed_ms": statistics.median(run["cpu_elapsed_ms"] for run in runs),
            "median_throughput_ops_per_sec": statistics.median(
                run["throughput_ops_per_sec"] for run in runs
            ),
            "elapsed_sample_variation": _sample_variation(
                [run["elapsed_ms"] for run in runs]
            ),
            "throughput_sample_variation": _sample_variation(
                [run["throughput_ops_per_sec"] for run in runs]
            ),
            "vector_query_count_per_sample": runs[0]["vector_query_count"],
            "samples": runs,
        }
    if len(state_digests) != 1 or len(vector_digests) != 1:
        raise RuntimeError("requested backend configurations produced different seeded inputs/state")
    return {
        "schema": "kogwistar-mixed-backend-report/v3",
        "run_id": run_id,
        "items": items,
        "samples_per_backend": samples,
        "warmup_samples_per_backend": 1,
        "warmups": warmups,
        "trace_path": str(trace_path),
        "trace_sha256": trace_sha256,
        "vector_query_indexes": vector_query_indexes,
        "embedding": {
            "inference": "not run",
            "model": None,
            "dimension": VECTOR_DIMENSION,
            "seed": "kogwistar-mixed-rewrite-v1",
            "normalization": "L2-normalized float32",
        },
        "state_digest_equal": True,
        "vector_digest_equal": True,
        "backends": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--items", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--vector-query-samples", type=int, default=100)
    parser.add_argument("--postgres-dsn", default=os.getenv("GKE_PG_DSN"))
    parser.add_argument(
        "--backend",
        choices=("all", "sqlite_event_projection", "chroma_sqlite_metadata", "postgres_pgvector"),
        default="all",
    )
    args = parser.parse_args()
    if args.items < 1 or args.samples < 2:
        parser.error("--items must be positive and --samples must be at least 2")
    if args.vector_query_samples < 1:
        parser.error("--vector-query-samples must be positive")
    backends = (
        ("sqlite_event_projection", "chroma_sqlite_metadata", "postgres_pgvector")
        if args.backend == "all"
        else (args.backend,)
    )
    report = run_benchmark(
        args.output,
        items=args.items,
        samples=args.samples,
        postgres_dsn=args.postgres_dsn,
        backends=backends,
        vector_query_indexes=_vector_query_indexes(args.items, args.vector_query_samples),
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    (args.output / "report.v3.json").write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
