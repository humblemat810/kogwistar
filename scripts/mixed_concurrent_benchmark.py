"""Run deterministic one-writer/four-reader SQLite mixed workload traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kogwistar.engine_core.engine_sqlite import EngineSQLite
if TYPE_CHECKING:
    from kogwistar.engine_core.rust_meta_sqlite import RustEngineSQLite
try:
    from . import mixed_backend_benchmark as backend_bench
    from .mixed_workload import (
        ConcurrentPhase,
        ROOT_NODE_ID,
        build_graph_payloads,
        build_root_graph_payload,
        build_normalized_vector,
        graph_edge_id,
        graph_node_id,
        normalized_vector_digest,
        previous_graph_node_id,
        ConcurrentStep,
        write_concurrent_trace,
    )
except ImportError:
    import mixed_backend_benchmark as backend_bench
    from mixed_workload import (
        ConcurrentPhase,
        ROOT_NODE_ID,
        build_graph_payloads,
        build_root_graph_payload,
        build_normalized_vector,
        graph_edge_id,
        graph_node_id,
        normalized_vector_digest,
        previous_graph_node_id,
        ConcurrentStep,
        write_concurrent_trace,
    )


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        for attempt in range(6):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(0.01 * (2**attempt))
    finally:
        temporary.unlink(missing_ok=True)


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, int(len(ordered) * percentile + 0.999999) - 1)]


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


def _run_sample(
    db_path: Path,
    *,
    mode: Literal["python", "rust"],
    mix: Literal["balanced", "read_heavy"],
    phases: list[ConcurrentPhase],
    backend_name: Literal[
        "sqlite_event_projection", "chroma_sqlite_metadata", "postgres_pgvector"
    ] = "sqlite_event_projection",
    postgres_dsn: str | None = None,
    vector_query_indexes: set[int] | None = None,
    prepared_vectors: list[list[float]] | None = None,
    prepared_graph_payloads: dict[int, tuple[dict[str, object], dict[str, object]]] | None = None,
) -> dict[str, Any]:
    if mode == "rust":
        from kogwistar.engine_core.rust_meta_sqlite import RustEngineSQLite

    process_rss_start = backend_bench.process_rss_bytes()
    store_type = EngineSQLite if mode == "python" else RustEngineSQLite
    namespace = f"rewrite_concurrent_{mix}"
    projection_namespace = f"rewrite_concurrent_head_{mix}"
    step_start = threading.Barrier(5, timeout=60)
    pending_read = threading.Barrier(5, timeout=60)
    pre_commit_read_done = threading.Barrier(5, timeout=60)
    commit_done = threading.Barrier(5, timeout=60)
    step_done = threading.Barrier(5, timeout=60)
    samples: dict[str, list[float]] = {
        "write": [], "read": [], "vector_upsert": [], "vector_query": []
    }
    samples_lock = threading.Lock()
    startup_ns: list[int] = []
    startup_lock = threading.Lock()
    measurement_start: dict[str, int] = {}

    def begin_measurement() -> None:
        measurement_start["wall"] = time.perf_counter_ns()
        measurement_start["cpu"] = time.process_time_ns()

    start_gate = threading.Barrier(5, action=begin_measurement, timeout=30)
    vector_query_indexes = vector_query_indexes or set()
    max_index = max(
        (step["write_index"] for phase in phases for step in phase["steps"]),
        default=-1,
    )
    vectors = prepared_vectors or [
        build_normalized_vector(index) for index in range(max_index + 1)
    ]
    if len(vectors) != max_index + 1:
        raise ValueError("prepared vector count does not match concurrent trace")
    vector_digests = [normalized_vector_digest(vector) for vector in vectors]
    graph_payloads = prepared_graph_payloads or {
        index: build_graph_payloads(index, vector_sha256=vector_digests[index])
        for index in range(max_index + 1)
    }
    if set(graph_payloads) != set(range(max_index + 1)):
        raise ValueError("prepared graph payload indexes do not match concurrent trace")
    vector_backend: Any | None = None
    vector_owner: Any = None
    schema = f"bench_{uuid.uuid4().hex[:12]}"
    if backend_name != "sqlite_event_projection":
        backend_start = time.perf_counter_ns()
        vector_root = db_path.parent / f"{db_path.stem}-vectors"
        vector_backend, _backend_info, vector_owner = backend_bench._make_vector_backend(
            backend_name, vector_root, postgres_dsn=postgres_dsn, schema=schema
        )
        if vector_backend is None:
            raise RuntimeError("Chroma backend reopen returned no collection")
        if vector_backend is None:
            raise RuntimeError(f"{backend_name} vector backend initialization returned no backend")
        startup_ns.append(time.perf_counter_ns() - backend_start)

    @contextmanager
    def worker_session(store: EngineSQLite | RustEngineSQLite) -> Iterator[None]:
        if mode != "rust":
            with nullcontext():
                yield
            return
        if not isinstance(store, RustEngineSQLite):
            raise TypeError("Rust worker session requires RustEngineSQLite")
        started = time.perf_counter_ns()
        with store.session():
            with startup_lock:
                startup_ns.append(time.perf_counter_ns() - started)
            yield

    stores = []
    if mode == "rust":
        # Reuse the Python facade object. The native bridge still selects
        # SQLite sessions from each worker's execution context.
        store = store_type(db_path.parent, filename=db_path.name)
        store.ensure_initialized()
        stores = [store] * 5
    else:
        for _worker in range(5):
            store = store_type(db_path.parent, filename=db_path.name)
            started = time.perf_counter_ns()
            store.ensure_initialized()
            with startup_lock:
                startup_ns.append(time.perf_counter_ns() - started)
            stores.append(store)

    root_payload = build_root_graph_payload()
    root_store = stores[0]
    with root_store.transaction():
        root_seq = root_store.append_entity_event(
            namespace=namespace,
            event_id="mixed-root-node",
            entity_kind="node",
            entity_id=ROOT_NODE_ID,
            op="UPSERT",
            payload_json=_canonical(root_payload),
        )
        root_store.replace_named_projection(
            namespace,
            f"node:{ROOT_NODE_ID}",
            {"entity_kind": "node", "entity_id": ROOT_NODE_ID, "entity": root_payload},
            last_authoritative_seq=root_seq,
            last_materialized_seq=root_seq,
            projection_schema_version=1,
            materialization_status="ready",
        )
        root_store.replace_named_projection(
            projection_namespace,
            "head",
            {
                "revision": 0,
                "entity_id": ROOT_NODE_ID,
                "edge_id": None,
                "previous_entity_id": None,
                "node_seq": root_seq,
                "edge_seq": 0,
            },
            last_authoritative_seq=root_seq,
            last_materialized_seq=root_seq,
            projection_schema_version=1,
            materialization_status="ready",
        )

    def synchronize(barrier: threading.Barrier, label: str) -> None:
        try:
            barrier.wait()
        except threading.BrokenBarrierError as exc:
            raise RuntimeError(f"concurrent benchmark {label} barrier broke") from exc

    def start_workload() -> None:
        try:
            start_gate.wait()
        except threading.BrokenBarrierError as exc:
            raise RuntimeError("concurrent benchmark startup barrier broke") from exc

    def write_step(store: EngineSQLite | RustEngineSQLite, index: int) -> float:
        entity_id = graph_node_id(index)
        edge_id = graph_edge_id(index)
        node_payload, edge_payload = graph_payloads[index]
        if backend_name != "sqlite_event_projection":
            vector_started = time.perf_counter_ns()
            backend_bench._vector_upsert(
                backend_name,
                vector_backend,
                index=index,
                vector=vectors[index],
            )
            with samples_lock:
                samples["vector_upsert"].append(
                    (time.perf_counter_ns() - vector_started) / 1_000_000
                )
        transaction_started = time.perf_counter_ns()
        coordinator_wait_ns = 0
        with store.transaction():
            node_seq = store.append_entity_event(
                namespace=namespace,
                event_id=f"mixed-node-event-{index:05d}",
                entity_kind="node",
                entity_id=entity_id,
                op="UPSERT",
                payload_json=_canonical(node_payload),
            )
            edge_seq = store.append_entity_event(
                namespace=namespace,
                event_id=f"mixed-edge-event-{index:05d}",
                entity_kind="edge",
                entity_id=edge_id,
                op="UPSERT",
                payload_json=_canonical(edge_payload),
            )
            store.replace_named_projection(
                namespace,
                f"node:{entity_id}",
                {"entity_kind": "node", "entity_id": entity_id, "entity": node_payload},
                last_authoritative_seq=node_seq,
                last_materialized_seq=node_seq,
                projection_schema_version=1,
                materialization_status="ready",
            )
            store.replace_named_projection(
                namespace,
                f"edge:{edge_id}",
                {"entity_kind": "edge", "entity_id": edge_id, "entity": edge_payload},
                last_authoritative_seq=edge_seq,
                last_materialized_seq=edge_seq,
                projection_schema_version=1,
                materialization_status="ready",
            )
            store.replace_named_projection(
                projection_namespace,
                "head",
                {
                    "revision": index + 1,
                    "entity_id": entity_id,
                    "edge_id": edge_id,
                    "previous_entity_id": previous_graph_node_id(index),
                    "node_seq": node_seq,
                    "edge_seq": edge_seq,
                },
                last_authoritative_seq=edge_seq,
                last_materialized_seq=edge_seq,
                projection_schema_version=1,
                materialization_status="ready",
            )
            # Readers must observe the old committed snapshot while this write
            # remains uncommitted; the transaction boundary is part of the trace.
            gate_started = time.perf_counter_ns()
            synchronize(pending_read, "uncommitted-read")
            synchronize(pre_commit_read_done, "pre-commit-read completion")
            coordinator_wait_ns = time.perf_counter_ns() - gate_started
        return max(0, time.perf_counter_ns() - transaction_started - coordinator_wait_ns) / 1_000_000

    def observe_step(
        store: EngineSQLite | RustEngineSQLite, reader_id: int, step: ConcurrentStep,
        visibility: Literal["before_commit", "after_commit"],
    ) -> None:
        index = int(step["write_index"])
        read_contract = next(
            item
            for item in step["reads"]
            if item["reader_id"] == reader_id and item["visibility"] == visibility
        )
        minimum_revision = int(read_contract["minimum_revision"])
        maximum_revision = int(read_contract["maximum_revision"])
        if minimum_revision > maximum_revision:
            raise RuntimeError("concurrent trace contains an inverted revision window")
        started = time.perf_counter_ns()
        head = store.get_named_projection(projection_namespace, "head")
        if head is None:
            raise RuntimeError("graph head projection disappeared")
        payload = head["payload"]
        if not isinstance(payload, dict):
            raise RuntimeError("head projection payload is not an object")
        raw_revision = payload.get("revision")
        raw_entity_id = payload.get("entity_id")
        raw_edge_id = payload.get("edge_id")
        raw_previous_entity_id = payload.get("previous_entity_id")
        raw_node_seq = payload.get("node_seq")
        raw_edge_seq = payload.get("edge_seq")
        if (
            not isinstance(raw_revision, int)
            or not isinstance(raw_entity_id, str)
            or (raw_edge_id is not None and not isinstance(raw_edge_id, str))
            or (
                raw_previous_entity_id is not None
                and not isinstance(raw_previous_entity_id, str)
            )
            or not isinstance(raw_node_seq, int)
            or not isinstance(raw_edge_seq, int)
        ):
            raise RuntimeError("head projection payload has invalid graph fields")
        revision = raw_revision
        entity_id = raw_entity_id
        edge_id = raw_edge_id
        previous_entity_id = raw_previous_entity_id
        node_seq = raw_node_seq
        edge_seq = raw_edge_seq
        if not minimum_revision <= revision <= maximum_revision:
            raise RuntimeError(
                f"reader {reader_id} saw revision {revision} {visibility}; "
                f"expected [{minimum_revision}, {maximum_revision}]"
            )
        expected_entity_id = ROOT_NODE_ID if revision == 0 else graph_node_id(revision - 1)
        expected_edge_id = None if revision == 0 else graph_edge_id(revision - 1)
        expected_previous_id = None if revision == 0 else previous_graph_node_id(revision - 1)
        expected_node_seq = 1 if revision == 0 else 2 * revision
        expected_edge_seq = 0 if revision == 0 else expected_node_seq + 1
        if (
            entity_id != expected_entity_id
            or edge_id != expected_edge_id
            or previous_entity_id != expected_previous_id
            or node_seq != expected_node_seq
            or edge_seq != expected_edge_seq
        ):
            raise RuntimeError("head projection does not match expected graph revision")
        if revision == 0:
            events = list(
                store.iter_entity_events(
                    namespace=namespace,
                    from_seq=node_seq,
                    to_seq=node_seq,
                    batch_size=1,
                )
            )
            if len(events) != 1 or (events[0][1], events[0][2]) != ("node", entity_id):
                raise RuntimeError("graph root head does not match its committed node")
        else:
            events = list(
                store.iter_entity_events(
                    namespace=namespace,
                    from_seq=node_seq,
                    to_seq=edge_seq,
                    batch_size=2,
                )
            )
            if (
                len(events) != 2
                or (events[0][1], events[0][2]) != ("node", entity_id)
                or (events[1][1], events[1][2]) != ("edge", edge_id)
            ):
                raise RuntimeError("head node/edge events are missing or out of order")
            edge_payload = json.loads(events[1][4])
            if (
                not isinstance(edge_payload, dict)
                or edge_payload.get("source_ids") != [previous_entity_id]
                or edge_payload.get("target_ids") != [entity_id]
            ):
                raise RuntimeError("head edge endpoints do not match the committed graph")
        graph_read_elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        with samples_lock:
            samples["read"].append(graph_read_elapsed_ms)
        if (
            visibility == "after_commit"
            and backend_name != "sqlite_event_projection"
            and index in vector_query_indexes
        ):
            vector_started = time.perf_counter_ns()
            backend_bench._vector_query(
                backend_name,
                vector_backend,
                expected_entity_id=graph_node_id(index),
                vector=vectors[index],
            )
            with samples_lock:
                samples["vector_query"].append(
                    (time.perf_counter_ns() - vector_started) / 1_000_000
                )

    def writer() -> int:
        store = stores[0]
        count = 0
        try:
            with worker_session(store):
                start_workload()
                for phase in phases:
                    for step in phase["steps"]:
                        index = step["write_index"]
                        synchronize(step_start, "step-start")
                        write_elapsed_ms = write_step(store, index)
                        synchronize(commit_done, "commit")
                        synchronize(step_done, "step-completion")
                        with samples_lock:
                            samples["write"].append(write_elapsed_ms)
                        count += 1
            return count
        except BaseException:
            for gate in (
                step_start, pending_read, pre_commit_read_done, commit_done, step_done
            ):
                gate.abort()
            start_gate.abort()
            raise

    def reader(reader_id: int) -> int:
        store = stores[reader_id + 1]
        count = 0
        try:
            with worker_session(store):
                start_workload()
                for phase in phases:
                    for step in phase["steps"]:
                        synchronize(step_start, "step-start")
                        synchronize(pending_read, "uncommitted-read")
                        if any(
                            read["reader_id"] == reader_id
                            and read["visibility"] == "before_commit"
                            for read in step["reads"]
                        ):
                            observe_step(store, reader_id, step, "before_commit")
                        synchronize(pre_commit_read_done, "pre-commit-read completion")
                        synchronize(commit_done, "commit")
                        if any(
                            read["reader_id"] == reader_id
                            and read["visibility"] == "after_commit"
                            for read in step["reads"]
                        ):
                            observe_step(store, reader_id, step, "after_commit")
                        synchronize(step_done, "step-completion")
                        count += sum(
                            read["reader_id"] == reader_id for read in step["reads"]
                        )
            return count
        except BaseException:
            for gate in (
                step_start, pending_read, pre_commit_read_done, commit_done, step_done
            ):
                gate.abort()
            start_gate.abort()
            raise

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(writer)] + [pool.submit(reader, idx) for idx in range(4)]
        operation_count = 0
        failures: list[BaseException] = []
        for future in futures:
            try:
                operation_count += future.result()
            except BaseException as exc:
                failures.append(exc)
        if failures:
            details = "; ".join(repr(error) for error in failures)
            raise RuntimeError(f"concurrent workload worker failed: {details}") from failures[0]
    elapsed_ns = time.perf_counter_ns() - measurement_start["wall"]
    cpu_elapsed_ns = time.process_time_ns() - measurement_start["cpu"]

    final_store = store_type(db_path.parent, filename=db_path.name)
    events = [
        list(row)
        for row in final_store.iter_entity_events(
            namespace=namespace, from_seq=1, batch_size=2 * max_index + 2
        )
    ]
    graph_projections = [
        {key: value for key, value in row.items() if key != "updated_at_ms"}
        for row in final_store.list_named_projections(namespace)
    ]
    head = final_store.get_named_projection(projection_namespace, "head")
    expected_writes = sum(len(phase["steps"]) for phase in phases)
    if len(events) != 1 + 2 * expected_writes or len(graph_projections) != 1 + 2 * expected_writes or head is None:
        raise RuntimeError("concurrent run lost committed writes")
    final_payload = head["payload"]
    if (
        not isinstance(final_payload, dict)
        or final_payload.get("revision") != expected_writes
        or final_payload.get("entity_id") != graph_node_id(expected_writes - 1)
        or final_payload.get("edge_id") != graph_edge_id(expected_writes - 1)
        or final_payload.get("previous_entity_id")
        != previous_graph_node_id(expected_writes - 1)
    ):
        raise RuntimeError("concurrent run ended at an unexpected revision")
    for row in events[1:]:
        if row[1] != "edge":
            continue
        edge_payload = json.loads(row[4])
        if (
            not isinstance(edge_payload, dict)
            or not isinstance(edge_payload.get("source_ids"), list)
            or not isinstance(edge_payload.get("target_ids"), list)
            or len(edge_payload["source_ids"]) != 1
            or len(edge_payload["target_ids"]) != 1
        ):
            raise RuntimeError("concurrent final edge payload is malformed")
    backend_disk_bytes: int | None = None
    if backend_name == "chroma_sqlite_metadata":
        vector_root = db_path.parent / f"{db_path.stem}-vectors"
        vector_backend, _backend_info, vector_owner = backend_bench._make_vector_backend(
            backend_name, vector_root, postgres_dsn=postgres_dsn, schema=schema
        )
        if vector_backend is None:
            raise RuntimeError("Chroma backend reopen returned no collection")
        entity_id = final_payload.get("entity_id")
        if not isinstance(entity_id, str):
            raise RuntimeError("final head entity id is invalid")
        final_index = int(entity_id.rsplit("-", 1)[1])
        backend_bench._vector_query(
            backend_name,
            vector_backend,
            expected_entity_id=entity_id,
            vector=vectors[final_index],
        )
        reopened = vector_backend.query(
            query_embeddings=[vectors[final_index]], n_results=1
        )
        if reopened.get("ids", [[]])[0] != [entity_id]:
            raise RuntimeError("Chroma concurrent result did not survive reopen")
        backend_disk_bytes = backend_bench._persistent_bytes(vector_root / "chroma")
    elif backend_name == "postgres_pgvector":
        if vector_backend is None:
            raise RuntimeError("PostgreSQL vector backend disappeared before verification")
        entity_id = final_payload.get("entity_id")
        if not isinstance(entity_id, str):
            raise RuntimeError("final head entity id is invalid")
        final_index = int(entity_id.rsplit("-", 1)[1])
        backend_bench._vector_query(
            backend_name,
            vector_backend,
            expected_entity_id=entity_id,
            vector=vectors[final_index],
        )
        vector_backend.close()
        vector_backend, _backend_info, vector_owner = backend_bench._make_vector_backend(
            backend_name, db_path.parent, postgres_dsn=postgres_dsn, schema=schema
        )
        if vector_backend is None or vector_owner is None:
            raise RuntimeError("PostgreSQL backend reopen returned no backend or engine")
        reopened = vector_backend.node_query(
            query_embeddings=[vectors[final_index]], n_results=1,
            include=["distances"],
        )
        if reopened.get("ids", [[]])[0] != [entity_id]:
            raise RuntimeError("pgvector concurrent result did not survive reopen")
        vector_backend.close()
        if vector_owner is not None:
            with vector_owner.connect() as connection:
                backend_disk_bytes = int(
                    connection.exec_driver_sql(
                        "SELECT COALESCE(sum(pg_total_relation_size("
                        "(quote_ident(schemaname)||'.'||quote_ident(tablename))::regclass)),0) "
                        "FROM pg_tables WHERE schemaname = %s",
                        (schema,),
                    ).scalar_one()
                )
            with vector_owner.begin() as connection:
                connection.exec_driver_sql(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
            vector_owner.dispose()

    per_operation = {
        name: (
            {
                "count": len(values),
                "p50_elapsed_ms": _percentile(values, 0.50),
                "p95_elapsed_ms": _percentile(values, 0.95),
                "p99_elapsed_ms": _percentile(values, 0.99),
            }
            if values
            else {
                "count": 0,
                "p50_elapsed_ms": None,
                "p95_elapsed_ms": None,
                "p99_elapsed_ms": None,
            }
        )
        for name, values in samples.items()
    }
    state = {
        "events": events,
        "graph_projections": graph_projections,
        "head": {key: value for key, value in head.items() if key != "updated_at_ms"},
    }
    graph_operation_count = operation_count
    vector_operation_count = (
        len(samples["vector_upsert"]) + len(samples["vector_query"])
    )
    storage_operation_count = graph_operation_count + vector_operation_count
    return {
        "state_digest": _digest(state),
        "vector_digest": _digest(vector_digests),
        "backend": backend_name,
        "backend_reopen_verified": True,
        "operations": operation_count,
        "storage_operations": storage_operation_count,
        "elapsed_ms": elapsed_ns / 1_000_000,
        "cpu_elapsed_ms": cpu_elapsed_ns / 1_000_000,
        "throughput_ops_per_sec": storage_operation_count * 1_000_000_000 / elapsed_ns,
        "graph_throughput_ops_per_sec": graph_operation_count * 1_000_000_000 / elapsed_ns,
        "startup_median_ms": statistics.median(startup_ns) / 1_000_000,
        "process_rss_start_bytes": process_rss_start,
        "process_rss_end_bytes": backend_bench.process_rss_bytes(),
        "process_rss_delta_bytes": backend_bench.process_rss_bytes() - process_rss_start,
        "app_sqlite_bytes": backend_bench._sqlite_bytes(db_path),
        "backend_disk_bytes": backend_disk_bytes,
        "per_operation": per_operation,
    }


def run_benchmark(
    output: Path,
    *,
    items: int,
    samples: int,
    modes: tuple[Literal["python", "rust"], ...] = ("python", "rust"),
    backends: tuple[
        Literal["sqlite_event_projection", "chroma_sqlite_metadata", "postgres_pgvector"],
        ...,
    ] = ("sqlite_event_projection",),
    postgres_dsn: str | None = None,
    vector_query_indexes: set[int] | None = None,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema": "kogwistar-mixed-concurrent-report/v3",
        "status": "running",
        "items": items,
        "samples_per_mode": samples,
        "warmup_samples_per_mode": 1,
        "workers": {"writers": 1, "readers": 4},
        "backends": {},
        "active": None,
        "partial_samples": {},
    }
    vector_query_indexes = vector_query_indexes or set()
    run_id = uuid.uuid4().hex[:12]
    report["run_id"] = run_id
    _write_json_atomic(output / "progress.v3.json", report)
    for mix in ("balanced", "read_heavy"):
        trace_path = output / f"concurrent_{mix}.trace.v3.json"
        phases = write_concurrent_trace(trace_path, items, mix=mix)
        trace_sha256 = hashlib.sha256(trace_path.read_bytes()).hexdigest()
        max_index = max(
            (step["write_index"] for phase in phases for step in phase["steps"]),
            default=-1,
        )
        vectors = [build_normalized_vector(index) for index in range(max_index + 1)]
        vector_digests = [normalized_vector_digest(vector) for vector in vectors]
        graph_payloads = {
            index: build_graph_payloads(index, vector_sha256=vector_digests[index])
            for index in range(max_index + 1)
        }
        for backend_name in backends:
            backend_modes = modes if backend_name == "sqlite_event_projection" else ("python",)
            mode_reports: dict[str, Any] = {}
            digests: set[str] = set()
            for mode in backend_modes:
                base = output / backend_name / run_id
                base.mkdir(parents=True, exist_ok=True)
                warmup_path = base / f"{mix}-{mode}-warmup.sqlite3"
                report["active"] = {
                    "backend": backend_name,
                    "mix": mix,
                    "mode": mode,
                    "phase": "warmup",
                }
                _write_json_atomic(output / "progress.v3.json", report)
                print(
                    f"[mixed-concurrent] warmup: backend={backend_name} mix={mix} "
                    f"mode={mode} items={items}",
                    file=sys.stderr,
                    flush=True,
                )
                warmup = _run_sample(
                    warmup_path,
                    mode=mode,
                    mix=mix,
                    phases=phases,
                    backend_name=backend_name,
                    postgres_dsn=postgres_dsn,
                    vector_query_indexes=vector_query_indexes,
                    prepared_vectors=vectors,
                    prepared_graph_payloads=graph_payloads,
                )
                progress_key = f"{backend_name}/{mix}/{mode}"
                report["partial_samples"][progress_key] = {
                    "warmup": warmup,
                    "samples": [],
                }
                _write_json_atomic(output / "progress.v3.json", report)
                runs = []
                for sample in range(samples):
                    db_path = base / f"{mix}-{mode}-{sample}.sqlite3"
                    report["active"] = {
                        "backend": backend_name,
                        "mix": mix,
                        "mode": mode,
                        "phase": "sample",
                        "sample": sample + 1,
                        "samples": samples,
                    }
                    _write_json_atomic(output / "progress.v3.json", report)
                    print(
                        f"[mixed-concurrent] sample {sample + 1}/{samples}: "
                        f"backend={backend_name} mix={mix} mode={mode} items={items}",
                        file=sys.stderr,
                        flush=True,
                    )
                    result = _run_sample(
                        db_path,
                        mode=mode,
                        mix=mix,
                        phases=phases,
                        backend_name=backend_name,
                        postgres_dsn=postgres_dsn,
                        vector_query_indexes=vector_query_indexes,
                        prepared_vectors=vectors,
                        prepared_graph_payloads=graph_payloads,
                    )
                    runs.append(result)
                    digests.add(result["state_digest"])
                    report["partial_samples"][progress_key]["samples"] = runs
                    _write_json_atomic(output / "progress.v3.json", report)
                if warmup["state_digest"] != runs[0]["state_digest"]:
                    raise RuntimeError(
                        f"warm-up state mismatch for concurrent {backend_name}/{mix}/{mode}"
                    )
                mode_reports[mode] = {
                    "warmup": warmup,
                    "state_digest": runs[0]["state_digest"],
                    "vector_digest": runs[0]["vector_digest"],
                    "median_elapsed_ms": statistics.median(run["elapsed_ms"] for run in runs),
                    "elapsed_sample_variation": _sample_variation(
                        [run["elapsed_ms"] for run in runs]
                    ),
                    "p95_elapsed_ms": _percentile([run["elapsed_ms"] for run in runs], 0.95),
                    "p99_elapsed_ms": _percentile([run["elapsed_ms"] for run in runs], 0.99),
                    "median_cpu_elapsed_ms": statistics.median(
                        run["cpu_elapsed_ms"] for run in runs
                    ),
                    "median_throughput_ops_per_sec": statistics.median(
                        run["throughput_ops_per_sec"] for run in runs
                    ),
                    "throughput_sample_variation": _sample_variation(
                        [run["throughput_ops_per_sec"] for run in runs]
                    ),
                    "median_startup_ms": statistics.median(
                        run["startup_median_ms"] for run in runs
                    ),
                    "samples": runs,
                }
                report["partial_samples"].pop(progress_key, None)
                report["active"] = None
            if len(digests) != 1:
                raise RuntimeError(
                    f"backend state mismatch for concurrent {backend_name}/{mix} trace"
                )
            report["backends"].setdefault(backend_name, {"mixes": {}})["mixes"][mix] = {
                "trace_path": str(trace_path),
                "trace_sha256": trace_sha256,
                "mode_results": mode_reports,
                "state_digest_equal": True,
            }
            _write_json_atomic(output / "progress.v3.json", report)
    report["status"] = "complete"
    report["active"] = None
    _write_json_atomic(output / "progress.v3.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--items", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--mode", choices=("python", "rust", "both"), default="both")
    parser.add_argument(
        "--backend",
        choices=("sqlite_event_projection", "chroma_sqlite_metadata", "postgres_pgvector", "all"),
        default="sqlite_event_projection",
    )
    parser.add_argument("--postgres-dsn", default=os.getenv("GKE_PG_DSN"))
    parser.add_argument("--vector-query-samples", type=int, default=100)
    args = parser.parse_args()
    if args.items < 1 or args.samples < 2:
        parser.error("--items must be positive and --samples must be at least 2")
    modes = {
        "python": ("python",),
        "rust": ("rust",),
        "both": ("python", "rust"),
    }[args.mode]
    backends = (
        ("sqlite_event_projection", "chroma_sqlite_metadata", "postgres_pgvector")
        if args.backend == "all"
        else (args.backend,)
    )
    if "postgres_pgvector" in backends and not args.postgres_dsn:
        parser.error("--postgres-dsn is required when testing PostgreSQL/pgvector")
    if args.vector_query_samples < 1:
        parser.error("--vector-query-samples must be positive")
    report = run_benchmark(
        args.output,
        items=args.items,
        samples=args.samples,
        modes=modes,
        backends=backends,
        postgres_dsn=args.postgres_dsn,
        vector_query_indexes=set(
            backend_bench._vector_query_indexes(args.items, args.vector_query_samples)
        ),
    )
    payload = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    _write_json_atomic(args.output / "report.v3.json", report)
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
