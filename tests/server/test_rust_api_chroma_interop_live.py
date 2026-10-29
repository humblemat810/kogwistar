from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import shutil
import socket
import subprocess
import sysconfig
import time
import uuid
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
RUST_ROOT = ROOT / "rust"
SERVER = RUST_ROOT / "target" / "debug" / (
    "kogwistar-server.exe" if os.name == "nt" else "kogwistar-server"
)
SERVER_SOURCES = tuple(
    path
    for path in RUST_ROOT.rglob("*")
    if path.is_file()
    and "target" not in path.parts
    and "tests" not in path.parts
    and (path.suffix == ".rs" or path.name in {"Cargo.toml", "Cargo.lock"})
)


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _token(secret: str, claims: dict[str, object]) -> str:
    def encode(value: object) -> str:
        raw = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    signing_input = f'{encode({"alg": "HS256", "typ": "JWT"})}.{encode(claims)}'
    signature = hmac.new(secret.encode(), signing_input.encode(), hashlib.sha256).digest()
    return f"{signing_input}.{base64.urlsafe_b64encode(signature).rstrip(b'=').decode()}"


def _server_is_current() -> bool:
    return SERVER.exists() and not any(
        source.stat().st_mtime > SERVER.stat().st_mtime for source in SERVER_SOURCES
    )


def _wait_for_http_process(
    process: subprocess.Popen[bytes], endpoint: str, label: str
) -> None:
    for _ in range(120):
        if process.poll() is not None:
            raise AssertionError(f"{label} exited before readiness with {process.returncode}")
        try:
            response = httpx.get(endpoint, timeout=0.5)
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            time.sleep(0.1)
    raise AssertionError(f"{label} did not become ready at {endpoint}")


def _rust_neighbors(
    *, base: str, secret: str, namespace: str, source_id: str
) -> dict[str, object]:
    now = int(time.time())
    claims = {
        "sub": "interop-reader",
        "ns": ["knowledge"],
        "storage_ns": namespace,
        "capabilities": ["read_graph"],
        "roles": ["reader"],
        "exp": now + 300,
    }
    response = httpx.post(
        f"{base}/internal/graph/neighbors",
        headers={"Authorization": f"Bearer {_token(secret, claims)}"},
        json={"rid": source_id, "truth_graph": "knowledge"},
        timeout=10,
    )
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.manual
@pytest.mark.integration
@pytest.mark.e2e
def test_python_chroma_projection_is_read_and_reconciled_by_rust_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise sequential Python-write/Rust-read recovery over shared stores."""
    script_name = "chroma.exe" if os.name == "nt" else "chroma"
    interpreter_chroma = Path(sysconfig.get_path("scripts")) / script_name
    chroma_command = (
        str(interpreter_chroma)
        if interpreter_chroma.is_file()
        else shutil.which("chroma")
    )
    if chroma_command is None:
        pytest.skip("install the Chroma extra and CLI to run this manual integration")
    if not _server_is_current():
        pytest.skip("build the current Rust kogwistar-server binary first")

    chroma_port = _port()
    chroma_endpoint = f"http://127.0.0.1:{chroma_port}"
    chroma_process = subprocess.Popen(
        [
            chroma_command,
            "run",
            "--path",
            str(tmp_path / "chroma-data"),
            "--host",
            "127.0.0.1",
            "--port",
            str(chroma_port),
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_http_process(
            chroma_process, f"{chroma_endpoint}/api/v2/heartbeat", "Chroma"
        )

        import chromadb

        import kogwistar.engine_core.engine as engine_module
        from kogwistar.engine_core.engine import GraphKnowledgeEngine
        from kogwistar.graph_query import GraphQuery
        from kogwistar.server.auth_middleware import claims_ctx
        from tests._helpers.graph_builders import build_entity_node, build_relationship_edge

        monkeypatch.setattr(
            engine_module,
            "_import_chroma_client",
            lambda: (
                lambda _settings: chromadb.HttpClient(
                    host="127.0.0.1", port=chroma_port
                ),
                chromadb.config.Settings,
            ),
        )

        namespace = f"interop-{uuid.uuid4().hex}"
        data_path = tmp_path / "shared-meta"
        source_id = f"{namespace}-source"
        target_id = f"{namespace}-target"
        edge_id = f"{namespace}-edge"
        owner = {
            "agent_id": "interop-owner",
            "sub": "interop-owner",
            "role": "rw",
            "ns": "knowledge",
            "storage_ns": namespace,
        }

        def open_engine() -> GraphKnowledgeEngine:
            return GraphKnowledgeEngine(
                persist_directory=str(data_path),
                namespace=namespace,
                kg_graph_type="knowledge",
                persistence_mode="two_stage",
                embedding_profile_mode="inspect",
                embedding_function=lambda texts: [[0.01, 0.01] for _ in texts],
                acl_enabled=True,
            )

        writer = open_engine()
        owner_token = claims_ctx.set(owner)
        try:
            writer.write.add_node(
                build_entity_node(
                    node_id=source_id,
                    doc_id="interop-doc",
                    metadata={"visibility": "public"},
                )
            )
            writer.write.add_node(
                build_entity_node(
                    node_id=target_id,
                    doc_id="interop-doc",
                    metadata={"visibility": "public"},
                )
            )
            writer.write.add_edge(
                build_relationship_edge(
                    edge_id=edge_id,
                    src=source_id,
                    tgt=target_id,
                    doc_id="interop-doc",
                    metadata={"visibility": "public"},
                )
            )
            worker = writer.indexing.make_index_job_worker(max_jobs_per_tick=20)
            metrics = worker.tick()
            jobs = writer.meta_sqlite.list_index_jobs(namespace=namespace)
            assert metrics.done >= 3, [
                (job.entity_kind, job.entity_id, job.index_kind, job.status, job.last_error)
                for job in jobs
            ]
            assert writer.meta_sqlite.get_stage1_node_projection(namespace, source_id) is None
            projected_nodes = writer.backend.node_get(
                ids=[source_id, target_id], include=["metadatas"]
            )
            assert set(projected_nodes["ids"]) == {source_id, target_id}, projected_nodes
            assert all(
                metadata.get("_kogwistar_stage2_ready") is True
                and metadata.get("namespace") == namespace
                for metadata in projected_nodes["metadatas"]
            ), projected_nodes
            projected_acl = writer.backend.node_get(
                where={"entity_type": "acl_record"},
                include=["metadatas"],
            )
            assert any(
                metadata.get("acl_truth_graph") == "knowledge"
                for metadata in projected_acl["metadatas"]
            ), projected_acl
            python_graph = GraphQuery(writer)
            python_node_neighbors = python_graph.neighbors(source_id)
            python_edge_neighbors = python_graph.neighbors(edge_id)
        finally:
            claims_ctx.reset(owner_token)
            writer.close()

        def start_rust_server() -> tuple[subprocess.Popen[bytes], str, str]:
            rust_port = _port()
            secret = uuid.uuid4().hex
            env = os.environ.copy()
            for name in ("JWT_SECRET", "JWT_ALG", "JWT_ISS", "JWT_AUD"):
                env.pop(name, None)
            env.update(
                {
                    "AUTH_MODE": "jwt",
                    "JWT_SECRET": secret,
                    "JWT_ALG": "HS256",
                    "KOGWISTAR_BACKEND": "sqlite",
                    "KOGWISTAR_META_SQLITE_PATH": str(data_path / "meta.sqlite"),
                    "KOGWISTAR_CHROMA_ENDPOINT": chroma_endpoint,
                    "KOGWISTAR_SERVER_HOST": "127.0.0.1",
                    "KOGWISTAR_SERVER_PORT": str(rust_port),
                    "KOGWISTAR_SERVER_REQUIRED_ROLES": "reader",
                }
            )
            process = subprocess.Popen(
                [str(SERVER)],
                cwd=ROOT,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            base = f"http://127.0.0.1:{rust_port}"
            _wait_for_http_process(process, f"{base}/health", "Rust API")
            return process, base, secret

        def stop(process: subprocess.Popen[bytes]) -> None:
            process.terminate()
            process.wait(timeout=10)

        rust_process, rust_base, rust_secret = start_rust_server()
        try:
            initial = _rust_neighbors(
                base=rust_base,
                secret=rust_secret,
                namespace=namespace,
                source_id=source_id,
            )
            assert initial["nodes"] == sorted(python_node_neighbors["nodes"]), initial
            assert initial["edges"] == sorted(python_node_neighbors["edges"]), initial
            assert initial["nodes"] == [target_id], initial
            assert initial["edges"] == [edge_id], initial
            rust_edge_neighbors = _rust_neighbors(
                base=rust_base,
                secret=rust_secret,
                namespace=namespace,
                source_id=edge_id,
            )
            assert rust_edge_neighbors["nodes"] == sorted(
                python_edge_neighbors["nodes"]
            ), rust_edge_neighbors
            assert rust_edge_neighbors["edges"] == sorted(
                python_edge_neighbors["edges"]
            ), rust_edge_neighbors
            assert rust_edge_neighbors["nodes"] == sorted([source_id, target_id])
        finally:
            stop(rust_process)

        writer = open_engine()
        owner_token = claims_ctx.set(owner)
        try:
            stale_projection = writer.backend.edge_get(
                ids=[edge_id], include=["documents", "metadatas", "embeddings"]
            )
            assert stale_projection["ids"] == [edge_id], stale_projection
            assert writer.lifecycle.tombstone_edge(edge_id, reason="interop-recovery")
            # Model an already-in-flight projection write completing after the
            # canonical tombstone. Chroma does not order these writes; this is
            # deliberately sequential test setup, not concurrent-writer support.
            writer.backend.edge_upsert(
                ids=[edge_id],
                documents=stale_projection["documents"],
                metadatas=stale_projection["metadatas"],
                embeddings=[
                    [float(value) for value in stale_projection["embeddings"][0]]
                ],
            )
            stale = writer.backend.edge_get(
                ids=[edge_id], include=["documents", "metadatas", "embeddings"]
            )
            assert stale["ids"] == [edge_id], stale
            assert stale["metadatas"][0]["_kogwistar_stage2_ready"] is True
        finally:
            claims_ctx.reset(owner_token)
            writer.close()

        rust_process, rust_base, rust_secret = start_rust_server()
        try:
            after_tombstone = _rust_neighbors(
                base=rust_base,
                secret=rust_secret,
                namespace=namespace,
                source_id=source_id,
            )
            assert after_tombstone["nodes"] == [], after_tombstone
            assert after_tombstone["edges"] == [], after_tombstone
        finally:
            stop(rust_process)

        writer = open_engine()
        try:
            # Re-admit the canonical DELETE intent to model retry/recovery of a
            # stale projection write after its earlier acknowledgement.
            writer.enqueue_index_jobs_for_edge(edge_id, op="DELETE")
            worker = writer.indexing.make_index_job_worker(max_jobs_per_tick=20)
            assert worker.tick().done >= 1
            assert writer.backend.edge_get(ids=[edge_id], include=["metadatas"])["ids"] == []
        finally:
            writer.close()
    finally:
        chroma_process.terminate()
        chroma_process.wait(timeout=10)
