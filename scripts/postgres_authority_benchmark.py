"""Compare the shipped Python and Rust PostgreSQL metadata paths.

This is a manual performance probe, not a CI test. It intentionally uses
precomputed JSON payloads and no embedding model so database/bridge costs are
measured separately from model inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
import uuid
from collections.abc import Callable
from typing import Literal, TypeAlias

import sqlalchemy as sa

from kogwistar.engine_core.engine_postgres_meta import EnginePostgresMetaStore
from kogwistar.engine_core.rust_postgres_session import RustEnginePostgresMetaStore

MetadataStore: TypeAlias = EnginePostgresMetaStore | RustEnginePostgresMetaStore
Workload: TypeAlias = Literal[
    "event_append_individual",
    "event_append_grouped_transaction",
    "conversation_record_atomic_each",
    "conversation_record_grouped_transaction",
    "event_replay_batch",
    "projection_list",
]
WORKLOADS: tuple[Workload, ...] = (
    "event_append_individual",
    "event_append_grouped_transaction",
    "conversation_record_atomic_each",
    "conversation_record_grouped_transaction",
    "event_replay_batch",
    "projection_list",
)


def _measure(operation: Callable[[], None], *, count: int) -> dict[str, float | int]:
    start = time.perf_counter_ns()
    operation()
    elapsed_ns = time.perf_counter_ns() - start
    elapsed_ms = elapsed_ns / 1_000_000
    return {
        "elapsed_ms": elapsed_ms,
        "operations": count,
        "ops_per_second": count * 1_000_000_000 / elapsed_ns,
    }


def _event(store: MetadataStore, *, namespace: str, prefix: str, index: int) -> None:
    store.append_entity_event(
        namespace=namespace,
        # Event IDs are globally unique within the metadata store, not merely
        # within a namespace. Include the workload namespace to keep samples
        # isolated from each other.
        event_id=f"{namespace}-{prefix}-{index:06d}",
        entity_kind="node",
        entity_id=f"node-{index:06d}",
        op="UPSERT",
        payload_json=json.dumps(
            {"id": f"node-{index:06d}", "value": index},
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def _projection(
    store: MetadataStore, *, namespace: str, prefix: str, index: int
) -> None:
    store.replace_named_projection(
        namespace,
        f"{prefix}-{index:06d}",
        {"id": f"node-{index:06d}", "value": index},
        last_authoritative_seq=index + 1,
        last_materialized_seq=index + 1,
        projection_schema_version=1,
        materialization_status="ready",
    )


def _append_many(
    store: MetadataStore, *, namespace: str, prefix: str, count: int, grouped: bool
) -> None:
    if grouped:
        with store.transaction():
            for index in range(count):
                _event(store, namespace=namespace, prefix=prefix, index=index)
        return
    for index in range(count):
        _event(store, namespace=namespace, prefix=prefix, index=index)


def _conversation_records(
    store: MetadataStore,
    *,
    namespace: str,
    prefix: str,
    count: int,
    grouped: bool,
) -> None:
    def write_one(index: int) -> None:
        _event(store, namespace=namespace, prefix=prefix, index=index)
        _projection(store, namespace=namespace, prefix=prefix, index=index)

    if grouped:
        # This intentionally changes the commit boundary; treat it as a
        # throughput sensitivity probe, not a recommendation for chat turns.
        with store.transaction():
            for index in range(count):
                write_one(index)
        return

    # Each logical record remains atomic. This models per-operation durability.
    for index in range(count):
        with store.transaction():
            write_one(index)


def _read_events(store: MetadataStore, *, namespace: str, count: int) -> None:
    rows = list(store.iter_entity_events(namespace=namespace, from_seq=1, batch_size=count + 1))
    if len(rows) != count:
        raise RuntimeError(f"expected {count} events, read {len(rows)}")


def _read_projections(store: MetadataStore, *, namespace: str, count: int) -> None:
    rows = store.list_named_projections(namespace)
    if len(rows) != count:
        raise RuntimeError(f"expected {count} projections, read {len(rows)}")


def _event_digest(store: MetadataStore, *, namespace: str, count: int) -> str:
    actual = list(store.iter_entity_events(namespace=namespace, from_seq=1, batch_size=count + 1))
    expected = [
        (
            index + 1,
            "node",
            f"node-{index:06d}",
            "UPSERT",
            json.dumps(
                {"id": f"node-{index:06d}", "value": index},
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        for index in range(count)
    ]
    if actual != expected:
        raise RuntimeError(f"PostgreSQL event payload/ordering mismatch in {namespace}")
    canonical = json.dumps(actual, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _projection_digest(
    store: MetadataStore, *, namespace: str, prefix: str, count: int
) -> str:
    rows = store.list_named_projections(namespace)
    expected = [
        {
            "key": f"{prefix}-{index:06d}",
            "payload": {"id": f"node-{index:06d}", "value": index},
            "last_authoritative_seq": index + 1,
            "last_materialized_seq": index + 1,
            "projection_schema_version": 1,
            "materialization_status": "ready",
        }
        for index in range(count)
    ]
    actual = [
        {
            key: row[key]
            for key in (
                "key",
                "payload",
                "last_authoritative_seq",
                "last_materialized_seq",
                "projection_schema_version",
                "materialization_status",
            )
        }
        for row in rows
    ]
    if actual != expected:
        raise RuntimeError(f"PostgreSQL projection payload/order mismatch in {namespace}")
    canonical = json.dumps(actual, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _run_workload(
    store: MetadataStore,
    *,
    workload: Workload,
    suffix: str,
    items: int,
    canonical_prefix: str | None = None,
) -> tuple[dict[str, float | int], str]:
    namespace = f"{suffix}-{workload}"
    projection_prefix = canonical_prefix or suffix
    if workload == "event_append_individual":
        metrics = _measure(
            lambda: _append_many(
                store, namespace=namespace, prefix=suffix, count=items, grouped=False
            ),
            count=items,
        )
        return metrics, _event_digest(store, namespace=namespace, count=items)

    if workload == "event_append_grouped_transaction":
        metrics = _measure(
            lambda: _append_many(
                store, namespace=namespace, prefix=suffix, count=items, grouped=True
            ),
            count=items,
        )
        return metrics, _event_digest(store, namespace=namespace, count=items)

    if workload in {
        "conversation_record_atomic_each",
        "conversation_record_grouped_transaction",
    }:
        metrics = _measure(
            lambda: _conversation_records(
                store,
                namespace=namespace,
                prefix=projection_prefix,
                count=items,
                grouped=workload == "conversation_record_grouped_transaction",
            ),
            count=items,
        )
        event_digest = _event_digest(store, namespace=namespace, count=items)
        projection_digest = _projection_digest(
            store, namespace=namespace, prefix=projection_prefix, count=items
        )
        combined = hashlib.sha256(
            f"{event_digest}:{projection_digest}".encode("ascii")
        ).hexdigest()
        return metrics, combined

    if workload == "event_replay_batch":
        _append_many(store, namespace=namespace, prefix=suffix, count=items, grouped=True)
        metrics = _measure(
            lambda: _read_events(store, namespace=namespace, count=items), count=items
        )
        return metrics, _event_digest(store, namespace=namespace, count=items)

    if workload == "projection_list":
        with store.transaction():
            for index in range(items):
                _projection(
                    store, namespace=namespace, prefix=projection_prefix, index=index
                )
        metrics = _measure(
            lambda: _read_projections(store, namespace=namespace, count=items), count=items
        )
        return metrics, _projection_digest(
            store, namespace=namespace, prefix=projection_prefix, count=items
        )

    raise ValueError(f"unsupported PostgreSQL benchmark workload: {workload}")


def _summarize(samples: list[dict[str, float | int]]) -> dict[str, float | int]:
    durations = [float(sample["elapsed_ms"]) for sample in samples]
    throughputs = [float(sample["ops_per_second"]) for sample in samples]
    ordered = sorted(durations)
    p95_index = max(0, (95 * len(ordered) + 99) // 100 - 1)
    return {
        "median_elapsed_ms": statistics.median(durations),
        "p95_elapsed_ms": ordered[p95_index],
        "median_ops_per_second": statistics.median(throughputs),
    }


def run_benchmark(
    *,
    dsn: str,
    items: int,
    samples: int,
    warmups: int = 1,
    workload: Workload | None = None,
) -> dict[str, object]:
    parsed_url = sa.engine.make_url(dsn)
    rust_dsn = parsed_url.set(drivername="postgresql").render_as_string(
        hide_password=False
    )
    selected_workloads = (workload,) if workload is not None else WORKLOADS
    results: dict[str, list[dict[str, float | int]]] = {
        f"{mode}.{workload_name}": []
        for mode in ("python", "rust")
        for workload_name in selected_workloads
    }
    parity_digests: dict[str, list[str]] = {
        workload_name: [] for workload_name in selected_workloads
    }

    for sample_index in range(warmups + samples):
        schema = f"gke_perf_{uuid.uuid4().hex[:20]}"
        admin_engine = sa.create_engine(dsn, pool_size=1, max_overflow=0)
        python_engine = sa.create_engine(
            dsn,
            pool_size=6,
            pool_pre_ping=True,
            pool_reset_on_return="rollback",
        )
        python = EnginePostgresMetaStore(engine=python_engine, schema=schema)
        rust = RustEnginePostgresMetaStore(dsn=rust_dsn, schema=schema)
        try:
            python.ensure_initialized()
            rust.ensure_initialized()
            # Prime Python's persistent SQLAlchemy pool. The Rust standalone
            # path remains per-call by design; transaction workloads reuse its
            # connection for the duration of the transaction.
            python.current_global_seq()
            rust.current_global_seq()

            modes = (("python", python), ("rust", rust))
            if sample_index % 2:
                modes = tuple(reversed(modes))
            sample_digests: dict[str, dict[str, str]] = {
                workload_name: {} for workload_name in selected_workloads
            }
            for mode, store in modes:
                suffix = f"s{sample_index}-{mode}"
                for workload_name in selected_workloads:
                    metrics, digest = _run_workload(
                        store,
                        workload=workload_name,
                        suffix=suffix,
                        items=items,
                        canonical_prefix=f"s{sample_index}",
                    )
                    results[f"{mode}.{workload_name}"].append(metrics)
                    sample_digests[workload_name][mode] = digest
            for workload_name in selected_workloads:
                python_digest = sample_digests[workload_name]["python"]
                rust_digest = sample_digests[workload_name]["rust"]
                if rust_digest != python_digest:
                    raise RuntimeError(
                        f"Python/Rust result digest mismatch for {workload_name} "
                        f"sample {sample_index}"
                    )
                parity_digests[workload_name].append(python_digest)
        finally:
            python_engine.dispose()
            try:
                rust.close()
            finally:
                with admin_engine.begin() as connection:
                    connection.exec_driver_sql(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
                admin_engine.dispose()

    warmup_results = {
        key: _summarize(value[:warmups])
        for key, value in results.items()
        if warmups
    }
    measured_results = {
        key: _summarize(value[warmups:]) for key, value in results.items()
    }
    return {
        "schema": "kogwistar-postgres-authority-benchmark/v3",
        "items_per_sample": items,
        "samples": samples,
        "warmups": warmups,
        "workloads": list(selected_workloads),
        "python_rust_result_digests": parity_digests,
        "database_cache": "warm/unspecified; no server or OS cache flush was attempted",
        "embedding": "excluded; precomputed JSON payloads only",
        "warmup_results": warmup_results,
        "results": measured_results,
        "comparison_rust_over_python_median_latency": {
            key.removeprefix("python."): (
                float(
                    measured_results[key.replace("python.", "rust.")][
                        "median_elapsed_ms"
                    ]
                )
                / float(measured_results[key]["median_elapsed_ms"])
            )
            for key in measured_results
            if key.startswith("python.")
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dsn",
        default=os.environ.get("GKE_PG_DSN") or os.environ.get("DATABASE_URL"),
        help="Disposable PostgreSQL DSN (or GKE_PG_DSN / DATABASE_URL).",
    )
    parser.add_argument("--items", type=int, default=100)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument(
        "--warmups", type=int, default=1,
        help="Unreported warm-up samples, each using a fresh schema (default: 1).",
    )
    parser.add_argument(
        "--workload",
        choices=("all", *WORKLOADS),
        default="all",
        help="Run one operation cell to bound runtime, or all cells (default).",
    )
    args = parser.parse_args()
    if not args.dsn:
        parser.error("provide --dsn or set GKE_PG_DSN; no database was contacted")
    if args.items < 1 or args.samples < 1 or args.warmups < 0:
        parser.error("--items and --samples must be positive; --warmups cannot be negative")
    print(
        json.dumps(
            run_benchmark(
                dsn=args.dsn,
                items=args.items,
                samples=args.samples,
                warmups=args.warmups,
                workload=None if args.workload == "all" else args.workload,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
