from __future__ import annotations

"""
python -m kogwistar.workers.run_index_job_worker \
  --backend chroma \
  --persist-directory ./chroma_db \
  --namespace default        
"""

import argparse
import os
from typing import Any, Optional

from kogwistar.engine_core.engine import GraphKnowledgeEngine

PgVectorBackend: Any | None = None

from .index_job_worker import IndexJobWorker, run_forever


def _normalize_backend_name(raw_backend: str) -> str:
    backend = raw_backend.lower()
    aliases = {
        "sqlite": "chroma",
        "pgvector": "pg",
        "postgres": "pg",
    }
    return aliases.get(backend, backend)


def _build_engine(args: argparse.Namespace) -> GraphKnowledgeEngine:
    raw_backend = args.backend or os.getenv("GKE_BACKEND") or "chroma"
    backend = _normalize_backend_name(raw_backend)
    namespace = args.namespace or os.getenv("GKE_NAMESPACE") or "default"

    if backend == "chroma":
        persist = (
            args.persist_directory
            or os.getenv("GKE_PERSIST_DIRECTORY")
            or "./chroma_db"
        )
        eng = GraphKnowledgeEngine(persist_directory=str(persist))
        eng.namespace = namespace  # type: ignore[attr-defined]
        return eng

    if backend == "pg":
        import sqlalchemy as sa
        global PgVectorBackend

        if PgVectorBackend is None:
            from graph_knowledge_engine.engine_core.postgres_backend import (
                PgVectorBackend as _PgVectorBackend,
            )

            PgVectorBackend = _PgVectorBackend

        dsn = args.pg_url or os.getenv("GKE_PG_URL")
        if not dsn:
            raise SystemExit("pg backend requires --pg-url or GKE_PG_URL")
        schema = args.pg_schema or os.getenv("GKE_PG_SCHEMA") or "public"
        embedding_dim = int(
            args.embedding_dim or os.getenv("GKE_EMBEDDING_DIM") or 1536
        )

        sa_engine = sa.create_engine(dsn)
        backend_obj = PgVectorBackend(
            engine=sa_engine, embedding_dim=embedding_dim, schema=schema
        )
        eng = GraphKnowledgeEngine(backend=backend_obj)
        eng.namespace = namespace  # type: ignore[attr-defined]
        return eng

    raise SystemExit(
        f"Unknown backend: {raw_backend!r} "
        "(expected chroma/pg; accepted aliases: sqlite, pgvector, postgres)"
    )


def build_worker(eng: GraphKnowledgeEngine, args: argparse.Namespace) -> IndexJobWorker:
    return IndexJobWorker(
        engine=eng,
        batch_size=int(args.batch_size),
        lease_seconds=int(args.lease_seconds),
        max_jobs_per_tick=int(args.max_jobs_per_tick),
        namespace=args.namespace,
    )


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run the GKE index_jobs worker")
    ap.add_argument(
        "--backend",
        default=None,
        help="chroma/pg (accepted aliases: sqlite, pgvector, postgres; or env GKE_BACKEND)",
    )
    ap.add_argument(
        "--namespace", default=None, help="namespace to process (or env GKE_NAMESPACE)"
    )

    ap.add_argument(
        "--persist-directory",
        default=None,
        help="chroma persist dir (or env GKE_PERSIST_DIRECTORY)",
    )

    ap.add_argument("--pg-url", default=None, help="Postgres DSN (or env GKE_PG_URL)")
    ap.add_argument(
        "--pg-schema", default=None, help="Postgres schema (or env GKE_PG_SCHEMA)"
    )
    ap.add_argument(
        "--embedding-dim",
        dest="embedding_dim",
        default=None,
        help="embedding dimension (or env GKE_EMBEDDING_DIM)",
    )

    ap.add_argument(
        "--batch-size", type=int, default=int(os.getenv("GKE_WORKER_BATCH_SIZE", "50"))
    )
    ap.add_argument(
        "--lease-seconds",
        type=int,
        default=int(os.getenv("GKE_WORKER_LEASE_SECONDS", "60")),
    )
    ap.add_argument(
        "--max-jobs-per-tick",
        type=int,
        default=int(os.getenv("GKE_WORKER_MAX_JOBS_PER_TICK", "200")),
    )
    ap.add_argument(
        "--tick-interval",
        type=float,
        default=float(os.getenv("GKE_WORKER_TICK_INTERVAL", "0.5")),
    )

    ap.add_argument("--once", action="store_true", help="run a single tick then exit")

    args = ap.parse_args(argv)

    eng = _build_engine(args)
    worker = build_worker(eng, args)

    if args.once:
        m = worker.tick()
        print(
            f"claimed={m.claimed} done={m.done} retried={m.retried} failed={m.failed} "
            + (
                f"avg_s={m.avg_job_duration_s:.6f}"
                if m.avg_job_duration_s is not None
                else ""
            )
        )
        return 0

    def _log(m):
        print(
            f"claimed={m.claimed} done={m.done} retried={m.retried} failed={m.failed} "
            + (
                f"avg_s={m.avg_job_duration_s:.6f}"
                if m.avg_job_duration_s is not None
                else ""
            )
        )

    run_forever(worker=worker, tick_interval_s=args.tick_interval, on_tick=_log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
