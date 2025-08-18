# graph_knowledge_engine/graph_viz.py
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Set
import json
from graph_knowledge_engine.engine import GraphKnowledgeEngine
from ..models import Node, Edge

# def _nodes_by_doc(engine, doc_id: Optional[str]) -> List[str]:
#     """Return node ids scoped by doc (via node_docs index if present), else all."""
#     if not doc_id:
#         got = engine.node_collection.get(include=["ids"])
#         return list(got.get("ids") or [])
#     # Prefer node_docs_collection if available (handles multi-doc nodes)
#     try:
#         rows = engine.node_docs_collection.get(where={"doc_id": doc_id}, include=["documents"])
#         if rows.get("documents"):
#             return [json.loads(d)["node_id"] for d in rows["documents"]]
#     except Exception:
#         pass
#     # Fallback to flat metadata denorm
#     got = engine.node_collection.get(where={"doc_ids": {"$contains": doc_id}}, include=["ids"])
#     return list(got.get("ids") or [])

# def _edges_by_doc(engine, doc_id: Optional[str]) -> List[str]:
#     """Return edge ids scoped by doc (via edge_endpoints index), else all."""
#     if not doc_id:
#         got = engine.edge_collection.get(include=["ids"])
#         return list(got.get("ids") or [])
#     eps = engine.edge_endpoints_collection.get(where={"doc_id": doc_id}, include=["documents"])
#     if not eps.get("documents"):
#         return []
#     return list({json.loads(d)["edge_id"] for d in eps["documents"]})

# def _load_nodes(engine:GraphKnowledgeEngine, ids: List[str]) -> List[Node]:
#     if not ids: return []
#     got = engine.node_collection.get(ids=ids, include=["documents"])
#     out: List[Node] = []
#     for doc in got.get("documents") or []:
#         try:
#             out.append(Node.model_validate_json(doc))
#         except Exception:
#             pass
#     return out

# def _load_edges(engine:GraphKnowledgeEngine, ids: List[str]) -> List[Edge]:
#     if not ids: return []
#     got = engine.edge_collection.get(ids=ids, include=["documents"])
#     out: List[Edge] = []
#     for doc in got.get("documents") or []:
#         try:
#             out.append(Edge.model_validate_json(doc))
#         except Exception:
#             pass
#     return out

# -----------------------------
# Cytoscape format
# -----------------------------
def to_cytoscape(
    engine:GraphKnowledgeEngine,
    *,
    doc_id: Optional[str] = None,
    mode: str = "reify",           # "reify" | "explode"
    limit_nodes: int = 2000,
    limit_edges: int = 2000,
) -> Dict:
    """
    Produce Cytoscape.js-ready JSON elements (nodes + edges).
    - reify: represent each hyperedge as a node "E:<edge_id>" and connect with role-labeled links
    - explode: create simple edges for each (src x tgt); meta edge→edge endpoints are skipped
    """
    node_ids = engine._nodes_by_doc(doc_id)[:limit_nodes]
    edge_ids = engine._edge_ids_by_doc(doc_id)[:limit_edges]

    nodes = engine.get_nodes(node_ids)
    edges = engine.get_edges(edge_ids)

    elements: List[Dict] = []

    # Add real nodes first
    for n in nodes:
        elements.append({
            "data": {
                "id": n.id,
                "label": n.label,
                "type": n.type,
                "summary": n.summary,
            },
            "classes": n.type  # "entity" or "relationship"
        })

    added_node_ids: Set[str] = set(node_ids)

    def _ensure_node(nid: str, label: str, classes: str = "meta"):
        if nid in added_node_ids:
            return
        elements.append({"data": {"id": nid, "label": label}, "classes": classes})
        added_node_ids.add(nid)

    if mode == "reify":
        # Represent each edge as its own node, then connect sources/targets
        for e in edges:
            e_node_id = f"E:{e.id}"
            _ensure_node(e_node_id, e.label or e.relation or "edge", classes="edge-node")

            # node endpoints
            for s in (e.source_ids or []):
                _ensure_node(s, s)  # in case not in node scope but referenced
                elements.append({
                    "data": {
                        "id": f"{e.id}:src:{s}",
                        "source": s,
                        "target": e_node_id,
                        "label": e.relation or e.label,
                        "relation": e.relation or "src_of",
                        "role": "src",
                    },
                    "classes": "src"
                })
            for t in (e.target_ids or []):
                _ensure_node(t, t)
                elements.append({
                    "data": {
                        "id": f"{e.id}:tgt:{t}",
                        "source": e_node_id,
                        "target": t,
                        "label": e.relation or e.label,
                        "relation": e.relation or "tgt_of",
                        "role": "tgt",
                    },
                    "classes": "tgt"
                })

            # meta edge→edge endpoints (reify edge-nodes on both sides)
            for se in getattr(e, "source_edge_ids", []) or []:
                src_edge_node = f"E:{se}"
                _ensure_node(src_edge_node, se, classes="edge-node")
                elements.append({
                    "data": {
                        "id": f"{e.id}:srcE:{se}",
                        "source": src_edge_node,
                        "target": e_node_id,
                        "label": e.relation or "src_edge",
                        "relation": e.relation or "src_edge",
                        "role": "src_edge",
                    },
                    "classes": "src-edge"
                })
            for te in getattr(e, "target_edge_ids", []) or []:
                tgt_edge_node = f"E:{te}"
                _ensure_node(tgt_edge_node, te, classes="edge-node")
                elements.append({
                    "data": {
                        "id": f"{e.id}:tgtE:{te}",
                        "source": e_node_id,
                        "target": tgt_edge_node,
                        "label": e.relation or "tgt_edge",
                        "relation": e.relation or "tgt_edge",
                        "role": "tgt_edge",
                    },
                    "classes": "tgt-edge"
                })

    else:  # explode
        # Create simple edges for each (src x tgt). Skip edge→edge meta for simplicity.
        for e in edges:
            srcs = e.source_ids or []
            tgts = e.target_ids or []
            for s in srcs:
                _ensure_node(s, s)
                for t in tgts:
                    _ensure_node(t, t)
                    eid = f"{e.id}:{s}->{t}"
                    elements.append({
                        "data": {
                            "id": eid,
                            "source": s,
                            "target": t,
                            "label": e.relation or e.label,
                            "relation": e.relation,
                            "role": "edge",
                        },
                        "classes": "edge"
                    })

    return {"elements": elements, "mode": mode, "doc_id": doc_id}

# -----------------------------
# D3 (force) format
# -----------------------------
def to_d3_force(
    engine:GraphKnowledgeEngine,
    *,
    doc_id: Optional[str] = None,
    mode: str = "reify",
    limit_nodes: int = 2000,
    limit_edges: int = 2000,
) -> Dict:
    """
    Produce D3-force-friendly JSON: {nodes: [...], links: [...]}
    - reify: introduces edge-nodes "E:<edge_id>" and role-labeled links
    - explode: simple source→target links
    """
    node_ids = engine._nodes_by_doc(doc_id)[:limit_nodes]
    edge_ids = engine._edge_ids_by_doc(doc_id)[:limit_edges]

    nodes = engine.get_nodes(node_ids)
    edges = engine.get_edges(edge_ids)

    node_map: Dict[str, Dict] = {}
    for n in nodes:
        node_map[n.id] = {"id": n.id, "label": n.label, "type": n.type, "summary": n.summary}

    links: List[Dict] = []

    def _ensure_node(nid: str, label: str, typ: str = "entity"):
        if nid not in node_map:
            node_map[nid] = {"id": nid, "label": label, "type": typ}

    if mode == "reify":
        for e in edges:
            e_node = f"E:{e.id}"
            _ensure_node(e_node, e.label or e.relation or "edge", typ="edge-node")

            for s in (e.source_ids or []):
                _ensure_node(s, s, "entity")
                links.append({"id": f"{e.id}:src:{s}", "source": s, "target": e_node, "label": e.relation, "role": "src"})
            for t in (e.target_ids or []):
                _ensure_node(t, t, "entity")
                links.append({"id": f"{e.id}:tgt:{t}", "source": e_node, "target": t, "label": e.relation, "role": "tgt"})

            for se in getattr(e, "source_edge_ids", []) or []:
                src_e_node = f"E:{se}"
                _ensure_node(src_e_node, se, "edge-node")
                links.append({"id": f"{e.id}:srcE:{se}", "source": src_e_node, "target": e_node, "label": e.relation, "role": "src_edge"})
            for te in getattr(e, "target_edge_ids", []) or []:
                tgt_e_node = f"E:{te}"
                _ensure_node(tgt_e_node, te, "edge-node")
                links.append({"id": f"{e.id}:tgtE:{te}", "source": e_node, "target": tgt_e_node, "label": e.relation, "role": "tgt_edge"})

    else:
        for e in edges:
            for s in (e.source_ids or []):
                _ensure_node(s, s, "entity")
                for t in (e.target_ids or []):
                    _ensure_node(t, t, "entity")
                    links.append({"id": f"{e.id}:{s}->{t}", "source": s, "target": t, "label": e.relation, "role": "edge"})

    return {"nodes": list(node_map.values()), "links": links, "mode": mode, "doc_id": doc_id}
