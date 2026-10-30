from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest
import sqlalchemy as sa

ROOT = Path(__file__).resolve().parents[2]
SERVER = ROOT / "rust" / "target" / "debug" / (
    "kogwistar-server.exe" if os.name == "nt" else "kogwistar-server"
)
SERVER_SOURCES = tuple(
    path
    for path in (ROOT / "rust").rglob("*")
    if path.is_file()
    and "target" not in path.parts
    and "tests" not in path.parts
    and (path.suffix == ".rs" or path.name in {"Cargo.toml", "Cargo.lock"})
)


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _server_is_current() -> bool:
    return SERVER.exists() and not any(
        source.stat().st_mtime > SERVER.stat().st_mtime for source in SERVER_SOURCES
    )


def _wait_for_http_process(
    process: subprocess.Popen[bytes], endpoint: str, label: str
) -> None:
    for _ in range(120):
        if process.poll() is not None:
            raise AssertionError(f"{label} exited before readiness: {process.returncode}")
        try:
            if httpx.get(endpoint, timeout=0.5).status_code == 200:
                return
        except httpx.HTTPError:
            time.sleep(0.1)
    raise AssertionError(f"{label} did not become ready at {endpoint}")


@pytest.mark.manual
@pytest.mark.integration
@pytest.mark.e2e
def test_python_postgres_stage1_is_promoted_by_rust_native_worker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Prove Python Stage-1 -> durable Rust result -> Python recovery/readback."""
    dsn = os.environ.get("KOGWISTAR_TEST_PG_DSN")
    if not dsn:
        pytest.skip("set KOGWISTAR_TEST_PG_DSN to run PostgreSQL interoperability")
    if not _server_is_current():
        pytest.skip("build the current Rust kogwistar-server binary first")

    if dsn.startswith("postgres://"):
        sqlalchemy_dsn = "postgresql+psycopg://" + dsn.removeprefix("postgres://")
    elif dsn.startswith("postgresql://"):
        sqlalchemy_dsn = "postgresql+psycopg://" + dsn.removeprefix("postgresql://")
    else:
        sqlalchemy_dsn = dsn

    schema = f"duallang_interop_{uuid.uuid4().hex[:16]}"
    namespace = f"duallang-{uuid.uuid4().hex}"
    entity_id = f"{namespace}-node"
    sql_engine = sa.create_engine(sqlalchemy_dsn, pool_pre_ping=True)
    embedding_requests: list[list[str]] = []
    rust_process: subprocess.Popen[bytes] | None = None
    log_path = tmp_path / "rust-postgres-worker.log"
    log_file = log_path.open("wb")

    class EmbedHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path != "/api/embed":
                self.send_error(404)
                return
            size = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(size))
            texts = request.get("input")
            if not isinstance(texts, list):
                self.send_error(400)
                return
            embedding_requests.append([str(text) for text in texts])
            body = json.dumps({"embeddings": [[0.6, 0.8] for _ in texts]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    embedding_server = ThreadingHTTPServer(("127.0.0.1", 0), EmbedHandler)
    embedding_thread = threading.Thread(
        target=embedding_server.serve_forever, daemon=True
    )
    embedding_thread.start()
    embedding_endpoint = f"http://127.0.0.1:{embedding_server.server_port}"
    try:
        monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", "python")
        monkeypatch.setenv("KOGWISTAR_IMPL_GRAPH_STORE", "python")
        monkeypatch.setenv("KOGWISTAR_IMPL_POSTGRES_AUTHORITY", "python")

        from kogwistar.engine_core.engine import GraphKnowledgeEngine
        from kogwistar.engine_core.engine_postgres_meta import EnginePostgresMetaStore
        from kogwistar.engine_core.postgres_backend import PgVectorBackend
        from tests._helpers.graph_builders import build_entity_node

        backend = PgVectorBackend(engine=sql_engine, embedding_dim=2, schema=schema)
        writer = GraphKnowledgeEngine(
            backend=backend,
            namespace=namespace,
            persistence_mode="two_stage",
            embedding_profile_mode="inspect",
            embedding_function=lambda texts: [[0.01, 0.01] for _ in texts],
        )
        try:
            writer.write.add_node(
                build_entity_node(
                    node_id=entity_id,
                    doc_id="postgres-rust-worker-doc",
                    label="Python staged PostgreSQL node",
                )
            )
            staged = backend.stage1_projection_get(
                namespace=namespace,
                entity_kind="node",
                entity_id=entity_id,
            )
            assert staged is not None and staged["materialization_status"] == "pending"
            jobs = writer.metadata.list_index_jobs(
                namespace=namespace,
                entity_kind="node",
                entity_id=entity_id,
            )
            assert jobs and jobs[0].status == "PENDING", jobs
            assert backend.node_get(ids=[entity_id], include=["embeddings"])["ids"] == []
        finally:
            writer.close()

        # Fail only the vector projection. Rust should durably admit the
        # provider result, requeue the action, and avoid another provider call.
        with sql_engine.begin() as connection:
            connection.execute(
                sa.text(
                    f"CREATE FUNCTION \"{schema}\".fail_embedding_projection() "
                    "RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN "
                    f"IF NEW.id = '{entity_id}' AND NEW.embedding IS NOT NULL THEN "
                    "RAISE EXCEPTION 'injected projection failure'; END IF; "
                    "RETURN NEW; END $$"
                )
            )
            connection.execute(
                sa.text(
                    f"CREATE TRIGGER fail_embedding_projection BEFORE INSERT OR UPDATE "
                    f"ON \"{schema}\".gke_nodes FOR EACH ROW EXECUTE FUNCTION "
                    f"\"{schema}\".fail_embedding_projection()"
                )
            )

        port = _port()
        env = os.environ.copy()
        for name in (
            "JWT_SECRET",
            "JWT_ALG",
            "JWT_ISS",
            "JWT_AUD",
            "KOGWISTAR_PROJECTION_RECONCILE_INTERVAL_S",
            "KOGWISTAR_PROJECTION_RECONCILE_NAMESPACE",
        ):
            env.pop(name, None)
        env.update(
            {
                "AUTH_MODE": "jwt",
                "JWT_SECRET": uuid.uuid4().hex,
                "JWT_ALG": "HS256",
                "KOGWISTAR_BACKEND": "pg",
                "KOGWISTAR_PG_DSN": dsn,
                "KOGWISTAR_PG_SCHEMA_BASE": schema,
                "KOGWISTAR_EMBEDDING_WORKER_INLINE": "true",
                "KOGWISTAR_EMBEDDING_PROVIDER": "ollama",
                "KOGWISTAR_EMBEDDING_MODEL": "fixture-embedding-v1",
                "KOGWISTAR_EMBEDDING_DIMENSION": "2",
                "KOGWISTAR_EMBEDDING_ENDPOINT": embedding_endpoint,
                "KOGWISTAR_SERVER_HOST": "127.0.0.1",
                "KOGWISTAR_SERVER_PORT": str(port),
            }
        )
        rust_process = subprocess.Popen(
            [str(SERVER)],
            cwd=ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        base = f"http://127.0.0.1:{port}"
        _wait_for_http_process(rust_process, f"{base}/health", "Rust PostgreSQL API")
        health = httpx.get(f"{base}/health", timeout=2).json()
        assert health.get("embedding_inline_worker_enabled") is True, health

        metadata = EnginePostgresMetaStore(engine=sql_engine, schema=schema)
        index_jobs_table = f'"{schema}"."{metadata.index_jobs_table}"'
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            jobs = metadata.list_index_jobs(
                namespace=namespace,
                entity_kind="node",
                entity_id=entity_id,
            )
            staged = backend.stage1_projection_get(
                namespace=namespace,
                entity_kind="node",
                entity_id=entity_id,
            )
            if (
                jobs
                and jobs[0].status == "PENDING"
                and jobs[0].accepted_result_json is not None
                and jobs[0].accepted_result_sha256 is not None
                and staged is not None
            ):
                with sql_engine.begin() as connection:
                    parked = connection.execute(
                        sa.text(
                            f"UPDATE {index_jobs_table} "
                            "SET next_run_at=NOW() + INTERVAL '5 minutes' "
                            "WHERE job_id=:job_id AND status='PENDING' "
                            "AND accepted_result_json IS NOT NULL"
                        ),
                        {"job_id": jobs[0].job_id},
                    )
                if parked.rowcount == 1:
                    break
            if rust_process.poll() is not None:
                raise AssertionError(
                    f"Rust server exited before promotion: {rust_process.returncode}"
                )
            time.sleep(0.1)
        else:
            log_file.flush()
            details = log_path.read_text(encoding="utf-8", errors="replace")[-5000:]
            raise AssertionError(
                "Rust worker did not durably accept the PostgreSQL projection; "
                f"embedding_requests={embedding_requests!r}; jobs={jobs!r}; "
                f"stage1={staged!r}; server_log_tail={details!r}"
            )

        assert embedding_requests, "Rust worker did not call the configured provider"
        rust_process.terminate()
        rust_process.wait(timeout=10)
        rust_process = None

        with sql_engine.begin() as connection:
            connection.execute(
                sa.text(
                    f'DROP TRIGGER fail_embedding_projection ON "{schema}".gke_nodes'
                )
            )
            connection.execute(
                sa.text(
                    f'DROP FUNCTION "{schema}".fail_embedding_projection()'
                )
            )
            connection.execute(
                sa.text(
                    f"UPDATE {index_jobs_table} SET next_run_at=NOW() "
                    "WHERE namespace=:namespace AND entity_id=:entity_id "
                    "AND accepted_result_json IS NOT NULL AND status='PENDING'"
                ),
                {"namespace": namespace, "entity_id": entity_id},
            )

        def fail_if_embedded(_texts: list[str]) -> list[list[float]]:
            raise AssertionError("Python recovery must replay the accepted vector")

        reopened = GraphKnowledgeEngine(
            backend=PgVectorBackend(engine=sql_engine, embedding_dim=2, schema=schema),
            namespace=namespace,
            persistence_mode="two_stage",
            embedding_profile_mode="inspect",
            embedding_function=fail_if_embedded,
        )
        try:
            pending = reopened.backend.stage1_projection_get(
                namespace=namespace,
                entity_kind="node",
                entity_id=entity_id,
            )
            assert pending is not None
            accepted = reopened.metadata.list_index_jobs(
                namespace=namespace,
                entity_kind="node",
                entity_id=entity_id,
            )
            assert accepted and accepted[0].accepted_result_json is not None, accepted
            accepted_wire = json.loads(accepted[0].accepted_result_json)
            accepted_record = accepted_wire["record"]
            accepted_source_metadata = dict(accepted_record["metadata"])
            accepted_stage1_fingerprint = accepted_source_metadata.pop(
                "_kogwistar_stage1_source_fingerprint"
            )
            expected_fingerprint = accepted_source_metadata.pop(
                "_kogwistar_stage1_expected_fingerprint"
            )
            accepted_stage1_revision = accepted_source_metadata.pop(
                "_kogwistar_stage1_revision"
            )
            expected_revision = accepted_source_metadata.pop(
                "_kogwistar_stage1_expected_revision"
            )
            assert accepted_record["document"] == pending["document"]
            assert accepted_source_metadata == pending["metadata"], (
                accepted_source_metadata,
                pending["metadata"],
            )
            assert accepted_stage1_fingerprint == pending["source_fingerprint"]
            assert expected_fingerprint == pending["source_fingerprint"]
            assert accepted_stage1_revision == pending["revision"]
            assert type(expected_revision) is int
            worker = reopened.indexing.make_index_job_worker(max_jobs_per_tick=10)
            assert worker.projection_store is reopened.two_stage_projection_adapter
            metrics = worker.tick()
            attempted = reopened.metadata.list_index_jobs(
                namespace=namespace,
                entity_kind="node",
                entity_id=entity_id,
            )
            assert metrics.done >= 1 and metrics.retried == 0, (
                f"{metrics}; jobs={[(job.status, job.last_error) for job in attempted]!r}"
            )
            assert reopened.backend.stage1_projection_get(
                namespace=namespace,
                entity_kind="node",
                entity_id=entity_id,
            ) is None
            completed = reopened.metadata.list_index_jobs(
                namespace=namespace,
                entity_kind="node",
                entity_id=entity_id,
            )
            assert completed and completed[0].status == "DONE", completed
            assert reopened.backend.node_get(
                ids=[entity_id], include=["embeddings", "metadatas"]
            )["embeddings"][0] == pytest.approx([0.6, 0.8])
            assert embedding_requests == [[staged["document"]]]
        finally:
            reopened.close()
    finally:
        if rust_process is not None:
            rust_process.terminate()
            rust_process.wait(timeout=10)
        embedding_server.shutdown()
        embedding_server.server_close()
        embedding_thread.join(timeout=5)
        log_file.close()
        with sql_engine.begin() as connection:
            connection.execute(sa.text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        sql_engine.dispose()
