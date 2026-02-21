import multiprocessing
import socket
import time
from typing import Any, Dict, List

import pytest
import json
import requests

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _run_uvicorn(app_import: str, host: str, port: int) -> None:
    """Run uvicorn in a child process.

    Import is by string so it works on Windows (spawn).
    """
    import uvicorn

    uvicorn.run(app_import, host=host, port=port, log_level="warning")


@pytest.fixture(scope="session")
def running_server() -> Dict[str, Any]:
    """Start the FastAPI+FastMCP server for these tests.

    The previous version of this test assumed a developer had a server already
    running on a fixed port (28110). That makes CI and local runs fragile.

    This fixture starts uvicorn on a free ephemeral port and waits for /health.
    """
    import importlib
    import traceback

    app_import_candidates = [
        "graph_knowledge_engine.server_mcp_with_admin:app",
    ]

    errors = []
    app_import = None

    for cand in app_import_candidates:
        mod, _, _ = cand.partition(":")
        try:
            importlib.import_module(mod)
            app_import = cand
            break
        except Exception as e:
            errors.append((cand, e, traceback.format_exc()))

    if not app_import:
        msg = "Could not import server app. Tried:\n"
        for cand, e, tb in errors:
            msg += f"\n- {cand}: {type(e).__name__}: {e}\n{tb}\n"
        raise RuntimeError(msg)

    host = "127.0.0.1"
    port = _pick_free_port()
    import subprocess, threading
    import sys
    from collections import deque

    log_buf = deque(maxlen=400)

    def _reader():
        if proc.stdout is None:
            raise Exception("stdout is None")
        for line in proc.stdout:
            log_buf.append(line)

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", app_import, "--host", "127.0.0.1", "--port", str(port), "--log-level", "debug"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    t = threading.Thread(target=_reader, daemon=True)
    t.start()

    base_http = f"http://{host}:{port}"
    deadline = time.time() + 65
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_http}/health", timeout=1)
            if r.ok:
                try:
                    yield {"proc": proc, "base_http": base_http}
                finally:
                    proc.terminate()
                    # proc.join(timeout=2)
                return
        except Exception as e:  # noqa: BLE001
            last_err = e
        time.sleep(0.2)

    proc.terminate()
    # proc.join(timeout=2)
    raise RuntimeError(f"Server failed to start on {base_http}: {last_err}")


@pytest.fixture()
def base_http(running_server: Dict[str, Any]) -> str:
    return str(running_server["base_http"])


@pytest.fixture()
def base_mcp(base_http: str) -> str:
    return f"{base_http}/mcp"


def _dev_token(base_http: str, role: str, ns: str = "docs") -> str:
    return requests.post(
        f"{base_http}/auth/dev-token",
        json={"username": "e2e", "role": role, "ns": ns},
        timeout=20,
    ).json()["token"]


def test_rollback_doc_node_edge_adjudicate(base_http: str, small_test_docs_nodes_edge_adjudcate):
    """Best-effort cleanup for local iteration."""
    rw_token = _dev_token(base_http, "rw", ns="docs")
    for doc_id in small_test_docs_nodes_edge_adjudcate["docs"]:
        requests.delete(
            f"{base_http}/admin/doc/{doc_id}",
            headers={"Authorization": f"Bearer {rw_token}"},
            timeout=20,
        )


@pytest.mark.asyncio
async def test_doc_node_edge_adjudicate(base_http: str, base_mcp: str, small_test_docs_nodes_edge_adjudcate):
    """E2E seam test:

    1) Upsert a small fixture graph via FastAPI.
    2) Verify RBAC seam on MCP tools/list (RO must not see RW-only tools).
    3) Call kg_crossdoc_adjudicate_anykind as RW (smoke).
    """

    bundle = small_test_docs_nodes_edge_adjudcate
    docs: Dict[str, str] = bundle["docs"]
    nodes: List[Dict[str, Any]] = bundle["nodes"]
    edges: List[Dict[str, Any]] = bundle["edges"]
    insertion_method = "fixture_sample"

    rw_token = _dev_token(base_http, "rw", ns="docs")
    ro_token = _dev_token(base_http, "ro", ns="docs")

    def _subset_for_doc(items: List[Dict[str, Any]], doc_id: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for it in items or []:
            refs = it.get("references") or []
            if not any(r.get("doc_id") == doc_id for r in refs):
                continue
            it2 = dict(it)
            it2["references"] = [
                {**r, "insertion_method": r.get("insertion_method") or insertion_method} for r in refs
            ]
            out.append(it2)
        return out

    # --- Upsert one payload per document ---
    for doc_id, content in docs.items():
        payload = {
            "doc_id": doc_id,
            "content": content,
            "doc_type": "plain",
            "insertion_method": insertion_method,
            "nodes": _subset_for_doc(nodes, doc_id),
            "edges": _subset_for_doc(edges, doc_id),
        }
        r = requests.post(
            f"{base_http}/api/graph/upsert",
            json=payload,
            headers={"Authorization": f"Bearer {rw_token}"},
            timeout=40,
        )
        assert r.ok, f"Upsert failed for {doc_id}: {r.status_code} {r.text}"

    # --- RBAC seam: RO must not see RW-only tools ---
    async with streamable_http_client(
        base_mcp,
        headers={"Authorization": f"Bearer {ro_token}"},
        sse_read_timeout=None,
        timeout=None,
    ) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = {t.name for t in tools.tools}
            forbidden = {"doc_parse", "store_document", "kg_extract"}
            assert forbidden.isdisjoint(names), f"Forbidden tools leaked into list_tools: {forbidden & names}"

    # --- RW smoke call ---
    async with streamable_http_client(
        base_mcp,
        headers={"Authorization": f"Bearer {rw_token}"},
        sse_read_timeout=None,
        timeout=None,
    ) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = {t.name for t in tools.tools}
            # keep your original assertion set; adjust if your server differs
            assert {"kg_extract", "doc_parse", "kg_crossdoc_adjudicate_anykind"} <= names

            # Optional cache/load tool if your server provides it
            if "kg_load_persisted" in names:
                _ = await session.call_tool(
                    "kg_load_persisted",
                    arguments={"inp": {"doc_ids": list(docs.keys()), "insertion_method": insertion_method}},
                )

            

            # convenience: allowed docs = the fixture doc ids
            allowed_docs = list(docs.keys())

            def _parse_mcp_payload(resp):
                c0 = resp.content[0]
                if getattr(c0, "type", "") == "json":
                    return c0.json
                # some clients return text; fall back to json.loads
                return json.loads(getattr(c0, "text", "{}"))

            def _collect_pairs(payload):
                """
                Normalize pairs from either:
                {"pairs": [{"left_id","right_id",...}, ...]}
                or
                {"pairs": [{"left": {"id", "kind"}, "right": {"id","kind"}}, ...]}
                into a set of unordered id-tuples {("A","B"), ...}.
                """
                out = set()
                for p in payload.get("pairs", []):
                    if "left_id" in p and "right_id" in p:
                        a, b = p["left_id"], p["right_id"]
                    else:
                        a, b = p.get("left", {}).get("id"), p.get("right", {}).get("id")
                    if a and b:
                        out.add(tuple(sorted((a, b))))
                return out

            # expected positives from the fixture
            must_have = {
                tuple(sorted(("N_CHLORO", "N_CHLORO_ALIAS"))),          # node↔node positive (alias)
                tuple(sorted(("E_PHOTO_LEAVES", "E_PHOTO_LEAVES_DUP"))),# edge↔edge positive (dup)
                tuple(sorted(("N_PHOTO_REIFIED", "E_PHOTO_LEAVES"))),   # cross-type positive (reified ↔ relation)
            }
            must_have_cross_doc = {
                tuple(sorted(('E_CHLORO_ABSORB','N_CHLORO_ALIAS'))),
                tuple(sorted(('E_PHOTO_LEAVES_DUP','N_CHLORO_ALIAS')))
            }
            must_have.update(must_have_cross_doc)
            # ----- Vector-based proposals -----
            for cross_doc_only in [True, False]:
                vec_res = await session.call_tool(
                    "propose_vector",
                    arguments={
                        "inp": {
                            "allowed_docs": allowed_docs,
                            "cross_doc_only": cross_doc_only,         # only cross-document candidates
                            "include_edges": True,          # allow node↔edge, edge↔edge too
                            "anchor_only": False,
                            "top_k": 8,
                            "where": {"insertion_method": {"$eq": insertion_method}},  # filter by our fixture provenance
                            # optional knobs your server already supports:
                            "score_mode": "distance",
                            "max_distance": 0.35,
                            # "min_similarity": 0.85,
                        }
                    },
                )
                vec_payload = _parse_mcp_payload(vec_res)
                vec_pairs = _collect_pairs(vec_payload)

                # ensure all required positives are present in vector proposals
                for pair in must_have if not cross_doc_only else must_have_cross_doc:
                    assert pair in vec_pairs, f"Vector proposer missing expected pair {pair}; got {sorted(vec_pairs)}"
                never_pair = tuple(sorted(("N_HEMO", "E_CHLORO_ABSORB")))
                assert never_pair not in vec_pairs, f"Vector proposer unexpectedly included negative {never_pair}"
                # vec_pairs adjudicate_pairs
                adj_result = await session.call_tool("adjudicate_pairs",arguments={
                    "inp": {
                        "pairs": vec_payload['pairs'],
                        "commit": False,
                    }
                })
                pass
                                                     
            # ----- Brute-force proposals -----
            bf_res = await session.call_tool(
                "kg_propose_bruteforce",
                arguments={
                    "inp": {
                        "allowed_docs": allowed_docs,
                        "cross_doc_only": False,
                        "include_edges": True,
                        "anchor_only": False,
                        "where": {"insertion_method": {"$eq": "fixture_sample"}},
                    }
                },
            )
            bf_payload = _parse_mcp_payload(bf_res)
            bf_pairs = _collect_pairs(bf_payload)

            for pair in must_have:
                assert tuple(sorted(pair)) in bf_pairs, f"Bruteforce proposer missing expected pair {pair}; got {sorted(bf_pairs)}"




            
            # Cross-document adjudication (node↔node, edge↔edge, node↔edge)
            res = await session.call_tool(
                "kg_crossdoc_adjudicate_anykind",
                arguments={"inp": {"doc_ids": list(docs.keys()), "insertion_method": insertion_method}},
            )

            assert res.content, "No MCP content returned"
            first = res.content[0]
            ok = (first.type == "json") or (first.type == "text" and json.loads(first.text))
            assert ok, f"Unexpected MCP response type: {first.type}"
            
            # # merge vector results:
            # from graph_knowledge_engine.server_mcp_with_admin import CrossDocAdjOut
            # adj_out = CrossDocAdjOut.model_validate(json.loads(adj_result.content[0].text))
            # for pairs in adj_out.results:
            #     lkind, rkind, same_entity = pairs.left_kind, pairs.right_kind, pairs.same_entity
            #     if same_entity:
            #         if lkind == rkind:
            #             canonical_id = engine.commit_merge(left, right, verdict)
            #         else:
            #             # Requires your engine to expose commit_any_kind(Node|Edge, Node|Edge, verdict)
            #             canonical_id = engine.commit_any_kind(left, right, verdict)
            #         if canonical_id:
            #             committed.append(str(canonical_id))
            
