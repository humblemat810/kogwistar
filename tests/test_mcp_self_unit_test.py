import pytest
import json
import pytest
import requests
from typing import Dict, Any, List

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


def test_rollback_doc_node_edge_adjudicate(small_test_docs_nodes_edge_adjudcate):
    graph_rag_port = 28110
    base_http = f"http://127.0.0.1:{graph_rag_port}"
    for doc_id in small_test_docs_nodes_edge_adjudcate["docs"]:
        r = requests.delete(f"{base_http}/admin/doc/{doc_id}", timeout=20)
        r
    pass
@pytest.mark.asyncio
async def test_doc_node_edge_adjudicate(small_test_docs_nodes_edge_adjudcate):
    """
    Uses the existing server at :28110:
      1) POST /api/graph/upsert for each document with only its nodes/edges
      2) Call MCP tool `kg_crossdoc_adjudicate_anykind`
    """
    graph_rag_port = 28110
    base_http = f"http://127.0.0.1:{graph_rag_port}"
    base_mcp = f"{base_http}/mcp"

    bundle = small_test_docs_nodes_edge_adjudcate
    docs: Dict[str, str] = bundle["docs"]
    nodes: List[Dict[str, Any]] = bundle["nodes"]
    edges: List[Dict[str, Any]] = bundle["edges"]
    insertion_method = "fixture_sample"

    def _subset_for_doc(items: List[Dict[str, Any]], doc_id: str) -> List[Dict[str, Any]]:
        """Keep only items that have at least one reference for doc_id.
        Also ensure each reference carries 'insertion_method' (server may filter by it)."""
        out: List[Dict[str, Any]] = []
        for it in items or []:
            refs = it.get("references") or []
            if not any(r.get("doc_id") == doc_id for r in refs):
                continue
            # normalize insertion_method on all refs (don’t change doc_id)
            it2 = dict(it)
            new_refs = []
            for r in refs:
                r2 = dict(r)
                if "insertion_method" not in r2 or r2["insertion_method"] is None:
                    r2["insertion_method"] = insertion_method
                new_refs.append(r2)
            it2["references"] = new_refs
            out.append(it2)
        return out

    # --- 1) Upsert one payload per document ---
    for doc_id, content in docs.items():
        n_for_doc = _subset_for_doc(nodes, doc_id)
        e_for_doc = _subset_for_doc(edges, doc_id)
        payload = {
            "doc_id": doc_id,
            "content": content,
            "doc_type": "plain",
            "insertion_method": insertion_method,
            "nodes": n_for_doc,
            "edges": e_for_doc,
        }
        r = requests.post(f"{base_http}/api/graph/upsert", json=payload, timeout=20)
        assert r.ok, f"Upsert failed for {doc_id}: {r.status_code} {r.text}"

    # --- 2) MCP: list tools and run cross-doc adjudication ---
    async with streamablehttp_client(base_mcp, sse_read_timeout=None, timeout=None) as (read, write, _):
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

            # Optional: a negative sanity check — this one should *not* be proposed by either
            
            # assert never_pair not in bf_pairs, f"Bruteforce proposer unexpectedly included negative {never_pair}"
            
            
            # Cross-document adjudication (node↔node, edge↔edge, node↔edge)
            res = await session.call_tool(
                "kg_crossdoc_adjudicate_anykind",
                arguments={"inp": {"doc_ids": list(docs.keys()), "insertion_method": insertion_method}},
            )

            assert res.content, "No MCP content returned"
            first = res.content[0]
            ok = (first.type == "json") or (first.type == "text" and json.loads(first.text))
            assert ok, f"Unexpected MCP response type: {first.type}"