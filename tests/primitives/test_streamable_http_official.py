# tests/test_streamable_http_e2e.py
import asyncio
import json
import os
import pathlib
import socket
import subprocess
import sys
import tempfile
import time
import uuid

import pytest

pytestmark = pytest.mark.ci_full
import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

ROOT = pathlib.Path(__file__).resolve().parents[2]
CALL_TIMEOUT_S = 20
TEST_TIMEOUT_S = 80
STARTUP_TIMEOUT_S = 60
SERVER_LOG_TAIL_BYTES = 64 * 1024


def _free_port():
    s = socket.socket()
    s.bind(("", 0))
    p = s.getsockname()[1]
    s.close()
    return p


async def _wait(url: str, server: subprocess.Popen[str], timeout=15):
    async with httpx.AsyncClient() as c:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if server.poll() is not None:
                raise RuntimeError(f"MCP server exited during startup: {server.returncode}")
            try:
                r = await c.get(url, timeout=0.5)
                if r.status_code < 500:
                    return
            except Exception:
                pass
            await asyncio.sleep(0.1)
    raise TimeoutError("server not ready")


def _server_log_tail(log_file) -> str:
    """Return bounded server diagnostics without risking a PIPE backpressure hang."""
    log_file.flush()
    log_file.seek(0, os.SEEK_END)
    size = log_file.tell()
    log_file.seek(max(0, size - SERVER_LOG_TAIL_BYTES))
    return log_file.read() or "<no server output>"


# --- Preseed helper (same models/engine your server uses) ---
def _preseed_chroma_dir(persist_dir: str):
    from kogwistar.engine_core.engine import GraphKnowledgeEngine
    from kogwistar.engine_core.models import Edge, Node
    from tests._kg_factories import kg_document, kg_grounding

    eng = GraphKnowledgeEngine(persist_directory=persist_dir)
    doc = kg_document(
        doc_id="D1",
        content="Smoking causes lung cancer.",
        source="test_streamable_http_e2e",
    )
    eng.write.add_document(doc)

    n_smoke = Node(
        label="Smoking",
        type="entity",
        summary="habit",
        mentions=[kg_grounding(doc.id, excerpt="Smoking", end_char=7)],
        doc_id=doc.id,
    )
    n_cancer = Node(
        label="Lung cancer",
        type="entity",
        summary="disease",
        mentions=[kg_grounding(doc.id, excerpt="lung cancer", end_char=11)],
        doc_id=doc.id,
    )
    eng.write.add_node(n_smoke, doc_id=doc.id)
    eng.write.add_node(n_cancer, doc_id=doc.id)

    e_causes = Edge(
        label="Smoking→Cancer",
        type="relationship",
        relation="causes",
        source_ids=[n_smoke.id],
        target_ids=[n_cancer.id],
        summary="causal claim",
        source_edge_ids=[],
        target_edge_ids=[],
        mentions=[kg_grounding(doc.id, excerpt="causes", end_char=6)],
        doc_id=doc.id,
    )
    eng.write.add_edge(e_causes, doc_id=doc.id)
    seed_node_id = n_smoke.id
    eng.close()
    return seed_node_id


@pytest.mark.asyncio
async def test_streamable_http_e2e(monkeypatch, tmp_path):
    selectors = (
        "KOGWISTAR_IMPL_MODE",
        "KOGWISTAR_IMPL_CONTRACTS",
        "KOGWISTAR_IMPL_GRAPH_STORE",
        "KOGWISTAR_IMPL_META_STORE",
        "KOGWISTAR_IMPL_POSTGRES_AUTHORITY",
        "KOGWISTAR_IMPL_RUNTIME",
        "KOGWISTAR_IMPL_SERVER",
    )
    for selector in selectors:
        monkeypatch.setenv(selector, "python")

    from kogwistar._rust_bridge import RustExtensionUnavailableError

    def unavailable() -> None:
        raise RustExtensionUnavailableError("Python-only MCP smoke loaded Rust")

    monkeypatch.setattr("kogwistar._rust_bridge._load_extension", unavailable)
    chroma_dir = str(tmp_path / "chroma")
    seed_node_id = _preseed_chroma_dir(chroma_dir)  # seed BEFORE starting the server

    port = _free_port()
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "MCP_CHROMA_DIR": chroma_dir,
        **{selector: "python" for selector in selectors},
    }
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    python_only_bootstrap = f"""
import kogwistar._rust_bridge as bridge
def unavailable():
    raise RuntimeError("Python-only MCP server attempted to load Rust")
bridge._load_extension = unavailable
import uvicorn
uvicorn.run(
    "server_mcp:mcp.streamable_http_app",
    factory=True,
    host="127.0.0.1",
    port={port},
    log_level="warning",
)
"""
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as server_log:
        srv = subprocess.Popen(
            [
                sys.executable,
                "-c",
                python_only_bootstrap,
            ],
            cwd=str(ROOT),
            env=env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )
        try:
            await asyncio.wait_for(
                _wait(f"http://127.0.0.1:{port}/mcp", srv, timeout=STARTUP_TIMEOUT_S),
                timeout=STARTUP_TIMEOUT_S + 5,
            )

            async def _exercise() -> None:
                async with streamable_http_client(f"http://127.0.0.1:{port}/mcp") as (
                    read,
                    write,
                ):
                    async with ClientSession(read, write) as session:
                        await asyncio.wait_for(session.initialize(), timeout=CALL_TIMEOUT_S)

                        tools = await asyncio.wait_for(session.list_tools(), timeout=CALL_TIMEOUT_S)
                        names = {t.name for t in tools.tools}
                        assert {
                            "kg_shortest_path",
                            "kg_find_edges",
                            "kg_semantic_seed_then_expand_text",
                            "kg_k_hop",
                            "kg_neighbors",
                        } <= names

                        # Prefer a non-embedding call for CI determinism.
                        import json

                        res = await asyncio.wait_for(
                            session.call_tool(
                                "kg_find_edges",
                                arguments={"relation": "causes", "doc_id": "D1"},
                            ),
                            timeout=CALL_TIMEOUT_S,
                        )
                        assert res.content[0].type in {"json", "text"}
                        find_edges_payload = json.loads(res.content[0].text)
                        assert isinstance(find_edges_payload.get("edges"), list)

                        res = await asyncio.wait_for(
                            session.call_tool(
                                "kg_k_hop",
                                arguments={"start_ids": [seed_node_id], "k": 2, "doc_id": "D1"},
                            ),
                            timeout=CALL_TIMEOUT_S,
                        )
                        assert res.content[0].type in {"json", "text"}
                        hop_payload = json.loads(res.content[0].text)
                        assert isinstance(hop_payload.get("layers"), list)
                        assert any(
                            layer.get("nodes") or layer.get("edges")
                            for layer in hop_payload["layers"]
                        ), (
                            f"kg_k_hop returned empty layers for seed {seed_node_id!r}"
                        )

            await asyncio.wait_for(_exercise(), timeout=TEST_TIMEOUT_S)
        except (RuntimeError, TimeoutError) as exc:
            raise AssertionError(
                f"MCP protocol test failed: {exc}; server log tail:\n"
                f"{_server_log_tail(server_log)}"
            ) from exc
        finally:
            if srv.poll() is None:
                srv.terminate()
            try:
                srv.wait(timeout=2)
            except Exception:
                srv.kill()
                srv.wait(timeout=2)


@pytest.mark.asyncio
async def test_python_only_rest_upsert_to_authenticated_mcp_http(monkeypatch, tmp_path):
    selectors = (
        "KOGWISTAR_IMPL_MODE",
        "KOGWISTAR_IMPL_CONTRACTS",
        "KOGWISTAR_IMPL_GRAPH_STORE",
        "KOGWISTAR_IMPL_META_STORE",
        "KOGWISTAR_IMPL_POSTGRES_AUTHORITY",
        "KOGWISTAR_IMPL_RUNTIME",
        "KOGWISTAR_IMPL_SERVER",
    )
    for selector in selectors:
        monkeypatch.setenv(selector, "python")

    port = _free_port()
    data_dir = tmp_path / "authenticated-python-server"
    data_dir.mkdir(parents=True)
    auth_database = (data_dir / "auth.sqlite").as_posix()
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "GKE_BACKEND": "chroma",
        "GKE_PERSIST_DIRECTORY": str(data_dir),
        "AUTH_DB_URL": f"sqlite:///{auth_database}",
        "AUTH_MODE": "dev",
        "JWT_SECRET": "python-only-mcp-test-secret",
        "KOGWISTAR_TEST_EMBEDDING_FUNCTION_IMPORT": (
            "tests._helpers.embeddings:build_default_test_embedding_function"
        ),
        **{selector: "python" for selector in selectors},
    }
    python_only_bootstrap = f"""
import kogwistar._rust_bridge as bridge
def unavailable():
    raise RuntimeError("Python-only authenticated server attempted to load Rust")
bridge._load_extension = unavailable
import uvicorn
uvicorn.run(
    "kogwistar.server_mcp_with_admin:app",
    host="127.0.0.1",
    port={port},
    log_level="warning",
)
"""
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as server_log:
        srv = subprocess.Popen(
            [sys.executable, "-c", python_only_bootstrap],
            cwd=str(ROOT),
            env=env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )
        try:
            base_url = f"http://127.0.0.1:{port}"
            await asyncio.wait_for(
                _wait(f"{base_url}/health", srv, timeout=STARTUP_TIMEOUT_S),
                timeout=STARTUP_TIMEOUT_S + 5,
            )
            async with httpx.AsyncClient(timeout=CALL_TIMEOUT_S) as client:
                token_response = await client.post(
                    f"{base_url}/auth/dev-token",
                    json={"username": "python-e2e", "role": "rw", "ns": "docs"},
                )
                token_response.raise_for_status()
                headers = {"Authorization": f"Bearer {token_response.json()['token']}"}

                doc_id = f"python-mcp-{uuid.uuid4().hex}"
                content = "Alpha connects to Beta."
                page = {
                    "collection_page_url": f"document_collection/{doc_id}",
                    "document_page_url": f"document/{doc_id}",
                    "doc_id": doc_id,
                    "insertion_method": "python_only_http_smoke",
                    "start_page": 1,
                    "end_page": 1,
                }
                payload = {
                    "doc_id": doc_id,
                    "content": content,
                    "doc_type": "text",
                    "insertion_method": "python_only_http_smoke",
                    "nodes": [
                        {
                            "id": "nn:alpha",
                            "label": "Alpha",
                            "type": "entity",
                            "summary": "Source concept",
                            "references": [
                                {**page, "start_char": 0, "end_char": 5, "excerpt": "Alpha"}
                            ],
                        },
                        {
                            "id": "nn:beta",
                            "label": "Beta",
                            "type": "entity",
                            "summary": "Target concept",
                            "references": [
                                {**page, "start_char": 18, "end_char": 22, "excerpt": "Beta"}
                            ],
                        },
                    ],
                    "edges": [
                        {
                            "id": "ne:connects",
                            "label": "Alpha connects to Beta",
                            "type": "relationship",
                            "relation": "connects_to",
                            "summary": "Connects Alpha to Beta",
                            "source_ids": ["nn:alpha"],
                            "target_ids": ["nn:beta"],
                            "source_edge_ids": [],
                            "target_edge_ids": [],
                            "references": [
                                {
                                    **page,
                                    "start_char": 0,
                                    "end_char": len(content) - 1,
                                    "excerpt": content[:-1],
                                }
                            ],
                        }
                    ],
                }
                upsert = await client.post(
                    f"{base_url}/api/graph/upsert", json=payload, headers=headers
                )
                upsert.raise_for_status()
                persisted = upsert.json()
                assert persisted["nodes_added"] == 2
                assert persisted["edges_added"] == 1

            async with httpx.AsyncClient(
                headers=headers, timeout=CALL_TIMEOUT_S
            ) as mcp_http:
                async with streamable_http_client(
                    f"{base_url}/mcp", http_client=mcp_http
                ) as (read, write):
                    async with ClientSession(read, write) as session:
                        await asyncio.wait_for(session.initialize(), timeout=CALL_TIMEOUT_S)
                        tools = await asyncio.wait_for(
                            session.list_tools(), timeout=CALL_TIMEOUT_S
                        )
                        assert "kg_neighbors" in {tool.name for tool in tools.tools}
                        result = await asyncio.wait_for(
                            session.call_tool(
                                "kg_neighbors", {"rid": persisted["node_ids"][0]}
                            ),
                            timeout=CALL_TIMEOUT_S,
                        )
                        neighbors = json.loads(result.content[0].text)
                        assert persisted["node_ids"][1] in neighbors["nodes"]
                        assert persisted["edge_ids"][0] in neighbors["edges"]
        except (RuntimeError, TimeoutError) as exc:
            raise AssertionError(
                f"Authenticated Python-only REST-to-MCP smoke failed: {exc}; "
                f"server log tail:\n{_server_log_tail(server_log)}"
            ) from exc
        finally:
            if srv.poll() is None:
                srv.terminate()
            try:
                srv.wait(timeout=2)
            except Exception:
                srv.kill()
                srv.wait(timeout=2)
