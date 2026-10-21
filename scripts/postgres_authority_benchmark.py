"""Compare the shipped Python and Rust PostgreSQL metadata paths.

This is a manual performance probe, not a CI test. It intentionally uses
precomputed JSON payloads and no embedding model so database/bridge costs are
measured separately from model inference.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import uuid
from collections.abc import Callable
from typing import TypeAlias

import sqlalchemy as sa

from kogwistar.engine_core.engine_postgres_meta import EnginePostgresMetaStore
from kogwistar.engine_core.rust_postgres_session import RustEnginePostgresMetaStore

MetadataStore: TypeAlias = EnginePostgresMetaStore | RustEnginePostgresMetaStore


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


def run_benchmark(*, dsn: str, items: int, samples: int) -> dict[str, object]:
    parsed_url = sa.engine.make_url(dsn)
    rust_dsn = parsed_url.set(drivername="postgresql").render_as_string(
        hide_password=False
    )
    results: dict[str, list[dict[str, float | int]]] = {
        f"{mode}.{workload}": []
        for mode in ("python", "rust")
        for workload in (
            "event_append_individual",
            "event_append_grouped_transaction",
            "conversation_record_atomic_each",
            "conversation_record_grouped_transaction",
            "event_replay_batch",
            "projection_list",
        )
    }

    for sample_index in range(samples):
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
            for mode, store in modes:
                suffix = f"s{sample_index}-{mode}"

                workload = "event_append_individual"
                namespace = f"{suffix}-append"
                results[f"{mode}.{workload}"].append(
                    _measure(
                        lambda: _append_many(
                            store,
                            namespace=namespace,
                            prefix=suffix,
                            count=items,
                            grouped=False,
                        ),
                        count=items,
                    )
                )

                workload = "event_append_grouped_transaction"
                namespace = f"{suffix}-append-grouped"
                results[f"{mode}.{workload}"].append(
                    _measure(
                        lambda: _append_many(
                            store,
                            namespace=namespace,
                            prefix=suffix,
                            count=items,
                            grouped=True,
                        ),
                        count=items,
                    )
                )

                workload = "conversation_record_atomic_each"
                namespace = f"{suffix}-conversation"
                results[f"{mode}.{workload}"].append(
                    _measure(
                        lambda: _conversation_records(
                            store,
                            namespace=namespace,
                            prefix=suffix,
                            count=items,
                            grouped=False,
                        ),
                        count=items,
                    )
                )

                workload = "conversation_record_grouped_transaction"
                namespace = f"{suffix}-conversation-grouped"
                results[f"{mode}.{workload}"].append(
                    _measure(
                        lambda: _conversation_records(
                            store,
                            namespace=namespace,
                            prefix=suffix,
                            count=items,
                            grouped=True,
                        ),
                        count=items,
                    )
                )

                workload = "event_replay_batch"
                namespace = f"{suffix}-read-events"
                _append_many(
                    store,
                    namespace=namespace,
                    prefix=suffix,
                    count=items,
                    grouped=True,
                )
                results[f"{mode}.{workload}"].append(
                    _measure(
                        lambda: _read_events(store, namespace=namespace, count=items),
                        count=items,
                    )
                )

                workload = "projection_list"
                namespace = f"{suffix}-read-projections"
                with store.transaction():
                    for index in range(items):
                        _projection(store, namespace=namespace, prefix=suffix, index=index)
                results[f"{mode}.{workload}"].append(
                    _measure(
                        lambda: _read_projections(
                            store, namespace=namespace, count=items
                        ),
                        count=items,
                    )
                )
        finally:
            python_engine.dispose()
            try:
                rust.close()
            finally:
                with admin_engine.begin() as connection:
                    connection.exec_driver_sql(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
                admin_engine.dispose()

    summarized = {key: _summarize(value) for key, value in results.items()}
    return {
        "schema": "kogwistar-postgres-authority-benchmark/v1",
        "items_per_sample": items,
        "samples": samples,
        "database_cache": "warm/unspecified; no server or OS cache flush was attempted",
        "embedding": "excluded; precomputed JSON payloads only",
        "results": summarized,
        "comparison_rust_over_python_median_latency": {
            key.removeprefix("python."): (
                float(summarized[key.replace("python.", "rust.")]["median_elapsed_ms"])
                / float(summarized[key]["median_elapsed_ms"])
            )
            for key in summarized
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
    args = parser.parse_args()
    if not args.dsn:
        parser.error("provide --dsn or set GKE_PG_DSN; no database was contacted")
    if args.items < 1 or args.samples < 1:
        parser.error("--items and --samples must be positive")
    print(json.dumps(run_benchmark(dsn=args.dsn, items=args.items, samples=args.samples), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
