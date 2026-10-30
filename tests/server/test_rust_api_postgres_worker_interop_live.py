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
    """Prove Python -> Rust-native PostgreSQL worker -> Python visibility."""
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
            if jobs and jobs[0].status == "DONE" and staged is None:
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
                "Rust worker did not promote PostgreSQL Stage-1; "
                f"embedding_requests={embedding_requests!r}; jobs={jobs!r}; "
                f"stage1={staged!r}; server_log_tail={details!r}"
            )

        assert embedding_requests, "Rust worker did not call the configured provider"
        promoted = backend.node_get(ids=[entity_id], include=["embeddings", "metadatas"])
        assert promoted["ids"] == [entity_id], promoted
        assert promoted["embeddings"][0] == pytest.approx([0.6, 0.8]), promoted
        assert promoted["metadatas"][0].get("_kogwistar_stage2_ready") is True

        rust_process.terminate()
        rust_process.wait(timeout=10)
        rust_process = None

        reopened = GraphKnowledgeEngine(
            backend=PgVectorBackend(engine=sql_engine, embedding_dim=2, schema=schema),
            namespace=namespace,
            persistence_mode="two_stage",
            embedding_profile_mode="inspect",
            embedding_function=lambda texts: [[0.01, 0.01] for _ in texts],
        )
        try:
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
                ids=[entity_id], include=["embeddings"]
            )["embeddings"][0] == pytest.approx([0.6, 0.8])
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
