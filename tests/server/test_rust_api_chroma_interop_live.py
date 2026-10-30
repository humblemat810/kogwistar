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
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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


def _rust_graph_tool(
    *,
    base: str,
    secret: str,
    namespace: str,
    tool_name: str,
    arguments: dict[str, object],
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
        f"{base}/mcp",
        headers={"Authorization": f"Bearer {_token(secret, claims)}"},
        json={
            "jsonrpc": "2.0",
            "id": f"interop-{tool_name}-{uuid.uuid4().hex}",
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        },
        timeout=10,
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    result = payload.get("result")
    assert isinstance(result, dict), payload
    assert result.get("implementation_authority") == "rust", result
    assert result.get("seam_crossings") == [], result
    return result


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
        private_target_id = f"{namespace}-private-target"
        edge_id = f"{namespace}-edge"
        private_endpoint_edge_id = f"{namespace}-private-endpoint-edge"
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
                    label="Source concept",
                    metadata={"visibility": "public"},
                )
            )
            writer.write.add_node(
                build_entity_node(
                    node_id=target_id,
                    doc_id="interop-doc",
                    label="Target concept",
                    metadata={"visibility": "public"},
                )
            )
            writer.write.add_node(
                build_entity_node(
                    node_id=private_target_id,
                    doc_id="interop-doc",
                    label="Target secret concept",
                    metadata={
                        "visibility": "private",
                        "owner_agent_id": owner["agent_id"],
                    },
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
            writer.write.add_edge(
                build_relationship_edge(
                    edge_id=private_endpoint_edge_id,
                    src=source_id,
                    tgt=private_target_id,
                    doc_id="interop-doc",
                    relation="related_to",
                    metadata={"visibility": "public"},
                )
            )
            worker = writer.indexing.make_index_job_worker(max_jobs_per_tick=20)
            metrics = worker.tick()
            jobs = writer.meta_sqlite.list_index_jobs(namespace=namespace)
            assert metrics.done >= 5, [
                (job.entity_kind, job.entity_id, job.index_kind, job.status, job.last_error)
                for job in jobs
            ]
            assert writer.meta_sqlite.get_stage1_node_projection(namespace, source_id) is None
            projected_nodes = writer.backend.node_get(
                ids=[source_id, target_id, private_target_id], include=["metadatas"]
            )
            assert set(projected_nodes["ids"]) == {
                source_id,
                target_id,
                private_target_id,
            }, projected_nodes
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
            reader_claims = {
                "sub": "interop-reader",
                "role": "ro",
                "roles": ["reader"],
                "ns": ["knowledge"],
                "storage_ns": namespace,
                "capabilities": ["read_graph"],
            }
            reader_token = claims_ctx.set(reader_claims)
            try:
                python_node_neighbors = python_graph.neighbors(source_id)
                python_edge_neighbors = python_graph.neighbors(edge_id)
                python_private_edge_neighbors = python_graph.neighbors(
                    private_endpoint_edge_id
                )
                python_find_edges = python_graph.find_edges(
                    relation="related_to",
                    src_label_contains="source",
                    tgt_label_contains="target",
                )
                python_path = python_graph.shortest_path(source_id, target_id)
                python_private_path = python_graph.shortest_path(
                    source_id, private_target_id
                )
                python_layers = python_graph.k_hop([source_id], k=1)
                assert not writer.read.get_nodes(ids=[private_target_id])
                assert not writer.read.get_edges(ids=[private_endpoint_edge_id])
            finally:
                claims_ctx.reset(reader_token)
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
            initial = _rust_graph_tool(
                base=rust_base,
                secret=rust_secret,
                namespace=namespace,
                tool_name="kg_neighbors",
                arguments={"rid": source_id},
            )
            assert initial["nodes"] == sorted(python_node_neighbors["nodes"]), initial
            assert initial["edges"] == sorted(python_node_neighbors["edges"]), initial
            assert initial["nodes"] == [target_id], initial
            assert initial["edges"] == [edge_id], initial
            assert private_endpoint_edge_id not in initial["edges"], initial
            rust_edge_neighbors = _rust_graph_tool(
                base=rust_base,
                secret=rust_secret,
                namespace=namespace,
                tool_name="kg_neighbors",
                arguments={"rid": edge_id},
            )
            assert rust_edge_neighbors["nodes"] == sorted(
                python_edge_neighbors["nodes"]
            ), rust_edge_neighbors
            assert rust_edge_neighbors["edges"] == sorted(
                python_edge_neighbors["edges"]
            ), rust_edge_neighbors
            assert rust_edge_neighbors["nodes"] == sorted([source_id, target_id])
            rust_private_edge_neighbors = _rust_graph_tool(
                base=rust_base,
                secret=rust_secret,
                namespace=namespace,
                tool_name="kg_neighbors",
                arguments={"rid": private_endpoint_edge_id},
            )
            assert rust_private_edge_neighbors["nodes"] == sorted(
                python_private_edge_neighbors["nodes"]
            ), rust_private_edge_neighbors
            assert rust_private_edge_neighbors["edges"] == sorted(
                python_private_edge_neighbors["edges"]
            ), rust_private_edge_neighbors
            assert rust_private_edge_neighbors["nodes"] == []
            assert rust_private_edge_neighbors["edges"] == []
            rust_find_edges = _rust_graph_tool(
                base=rust_base,
                secret=rust_secret,
                namespace=namespace,
                tool_name="kg_find_edges",
                arguments={
                    "relation": "related_to",
                    "src_label_contains": "source",
                    "tgt_label_contains": "target",
                },
            )
            assert rust_find_edges["edges"] == python_find_edges == [edge_id]
            rust_path = _rust_graph_tool(
                base=rust_base,
                secret=rust_secret,
                namespace=namespace,
                tool_name="kg_shortest_path",
                arguments={
                    "src_id": source_id,
                    "dst_id": target_id,
                    "max_depth": 8,
                },
            )
            assert rust_path["path"] == python_path == [source_id, edge_id, target_id]
            rust_private_path = _rust_graph_tool(
                base=rust_base,
                secret=rust_secret,
                namespace=namespace,
                tool_name="kg_shortest_path",
                arguments={
                    "src_id": source_id,
                    "dst_id": private_target_id,
                    "max_depth": 8,
                },
            )
            assert rust_private_path["path"] == python_private_path == []
            rust_layers = _rust_graph_tool(
                base=rust_base,
                secret=rust_secret,
                namespace=namespace,
                tool_name="kg_k_hop",
                arguments={"start_ids": [source_id], "k": 1},
            )
            assert rust_layers["layers"] == [
                {
                    "nodes": sorted(python_layers[0]["nodes"]),
                    "edges": sorted(python_layers[0]["edges"]),
                }
            ]
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
            after_tombstone = _rust_graph_tool(
                base=rust_base,
                secret=rust_secret,
                namespace=namespace,
                tool_name="kg_neighbors",
                arguments={"rid": source_id},
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


@pytest.mark.manual
@pytest.mark.integration
@pytest.mark.e2e
def test_python_stage1_is_promoted_by_rust_native_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Rust-only server resumes Python's closed SQLite Stage-1 work."""
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

    embedding_requests: list[list[str]] = []

    class OllamaEmbedHandler(BaseHTTPRequestHandler):
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
    embedding_server = ThreadingHTTPServer(("127.0.0.1", 0), OllamaEmbedHandler)
    embedding_thread = threading.Thread(target=embedding_server.serve_forever, daemon=True)
    embedding_thread.start()
    embedding_endpoint = f"http://127.0.0.1:{embedding_server.server_port}"
    rust_process: subprocess.Popen[bytes] | None = None
    rust_log_path = tmp_path / "rust-server.log"
    rust_log = rust_log_path.open("wb")
    try:
        _wait_for_http_process(
            chroma_process, f"{chroma_endpoint}/api/v2/heartbeat", "Chroma"
        )

        import chromadb

        import kogwistar.engine_core.engine as engine_module
        from kogwistar.engine_core.engine import GraphKnowledgeEngine
        from tests._helpers.graph_builders import build_entity_node

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
        monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", "python")
        namespace = f"python-rust-worker-{uuid.uuid4().hex}"
        entity_id = f"{namespace}-node"
        data_path = tmp_path / "shared-meta"
        writer = GraphKnowledgeEngine(
            persist_directory=str(data_path),
            namespace=namespace,
            persistence_mode="two_stage",
            embedding_profile_mode="inspect",
            embedding_function=lambda texts: [[0.01, 0.01] for _ in texts],
        )
        try:
            writer.write.add_node(
                build_entity_node(
                    node_id=entity_id,
                    doc_id="cross-language-worker-doc",
                    label="Python staged node",
                )
            )
            assert writer.meta_sqlite.get_stage1_node_projection(
                namespace, entity_id
            ) is not None
            jobs = writer.meta_sqlite.list_index_jobs(
                namespace=namespace, entity_kind="node", entity_id=entity_id
            )
            assert jobs and jobs[0].status == "PENDING"
        finally:
            writer.close()

        rust_port = _port()
        secret = uuid.uuid4().hex
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
                "JWT_SECRET": secret,
                "JWT_ALG": "HS256",
                "KOGWISTAR_BACKEND": "sqlite",
                "KOGWISTAR_META_SQLITE_PATH": str(data_path / "meta.sqlite"),
                "KOGWISTAR_CHROMA_ENDPOINT": chroma_endpoint,
                "KOGWISTAR_CHROMA_NODE_COLLECTION": "nodes",
                "KOGWISTAR_CHROMA_EDGE_COLLECTION": "edges",
                "KOGWISTAR_EMBEDDING_WORKER_INLINE": "true",
                "KOGWISTAR_EMBEDDING_PROVIDER": "ollama",
                "KOGWISTAR_EMBEDDING_MODEL": "fixture-embedding-v1",
                "KOGWISTAR_EMBEDDING_DIMENSION": "2",
                "KOGWISTAR_EMBEDDING_ENDPOINT": embedding_endpoint,
                "KOGWISTAR_SERVER_HOST": "127.0.0.1",
                "KOGWISTAR_SERVER_PORT": str(rust_port),
            }
        )
        rust_process = subprocess.Popen(
            [str(SERVER)],
            cwd=ROOT,
            env=env,
            stdout=rust_log,
            stderr=subprocess.STDOUT,
        )
        base = f"http://127.0.0.1:{rust_port}"
        _wait_for_http_process(rust_process, f"{base}/health", "Rust API")
        health = httpx.get(f"{base}/health", timeout=2).json()
        assert health.get("embedding_inline_worker_enabled") is True, health
        deadline = time.monotonic() + 30
        client = chromadb.HttpClient(host="127.0.0.1", port=chroma_port)
        collection = client.get_collection("nodes")
        projected = None
        while time.monotonic() < deadline:
            result = collection.get(
                ids=[entity_id], include=["embeddings", "metadatas"]
            )
            if result["ids"]:
                projected = result
                break
            if rust_process.poll() is not None:
                raise AssertionError(
                    f"Rust API exited before projection: {rust_process.returncode}"
                )
            time.sleep(0.1)
        if projected is None:
            rust_log.flush()
            details = rust_log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise AssertionError(
                "Rust inline worker did not promote Stage-1; "
                f"embedding_requests={embedding_requests!r}; server_log_tail={details!r}"
            )
        assert projected["metadatas"][0]["_kogwistar_stage2_ready"] is True
        assert projected["metadatas"][0]["namespace"] == namespace
        assert projected["embeddings"][0] == pytest.approx([0.6, 0.8])

        rust_process.terminate()
        rust_process.wait(timeout=10)
        rust_process = None

        verifier = GraphKnowledgeEngine(
            persist_directory=str(data_path),
            namespace=namespace,
            persistence_mode="two_stage",
            embedding_profile_mode="inspect",
            embedding_function=lambda texts: [[0.01, 0.01] for _ in texts],
        )
        try:
            assert verifier.meta_sqlite.get_stage1_node_projection(
                namespace, entity_id
            ) is None
            completed = verifier.meta_sqlite.list_index_jobs(
                namespace=namespace, entity_kind="node", entity_id=entity_id
            )
            assert completed and completed[0].status == "DONE"
        finally:
            verifier.close()
    finally:
        if rust_process is not None:
            rust_process.terminate()
            rust_process.wait(timeout=10)
        embedding_server.shutdown()
        embedding_server.server_close()
        embedding_thread.join(timeout=5)
        rust_log.close()
        chroma_process.terminate()
        chroma_process.wait(timeout=10)
