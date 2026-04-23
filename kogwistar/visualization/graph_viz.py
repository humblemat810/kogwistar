# kogwistar/visualization/graph_viz.py
from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple, Type, TypeVar, Union, TYPE_CHECKING

from ..runtime.models import WorkflowEdge, WorkflowNode

from ..conversation.models import ConversationEdge, ConversationNode

if TYPE_CHECKING:
    from kogwistar.engine_core.engine import GraphKnowledgeEngine

    from ..engine_core.models import (
        Node,
        Edge,
    )  # , ConversationNode, ConversationEdge, WorkflowEdge, WorkflowNode

    T = TypeVar("T", bound=Union[Node, Edge])


def _safe_iter(x):
    return x if isinstance(x, list) and x else []


def _render_d3_from_raw(nodes, edges, mode: str = "reify") -> Dict:
    node_map = {getattr(n, "id", None): n for n in nodes if getattr(n, "id", None)}
    edge_map = {getattr(e, "id", None): e for e in edges if getattr(e, "id", None)}

    out_nodes: Dict[str, Dict] = {}
    links: List[Dict] = []

    for nid, n in node_map.items():
        out_nodes[nid] = {
            "id": nid,
            "label": getattr(n, "label", nid),
            "type": getattr(n, "type", "entity"),
            "summary": getattr(n, "summary", None),
            "properties": getattr(n, "properties", {}) or {},
        }

    if mode.lower() == "reify":
        for eid, e in edge_map.items():
            if eid not in out_nodes:
                out_nodes[eid] = {
                    "id": eid,
                    "label": getattr(e, "relation", None) or getattr(e, "label", None) or "edge",
                    "type": "edge-node",
                    "summary": getattr(e, "summary", None),
                    "properties": getattr(e, "properties", {}) or {},
                }
            for s in _safe_iter(getattr(e, "source_ids", None)):
                links.append(
                    {
                        "source": s,
                        "target": eid,
                        "relation": getattr(e, "relation", None) or getattr(e, "label", None),
                        "role": "src",
                        "properties": getattr(e, "properties", {}) or {},
                    }
                )
            for se in _safe_iter(getattr(e, "source_edge_ids", None)):
                links.append(
                    {
                        "source": se,
                        "target": eid,
                        "relation": getattr(e, "relation", None) or getattr(e, "label", None),
                        "role": "src",
                        "properties": getattr(e, "properties", {}) or {},
                    }
                )
            for t in _safe_iter(getattr(e, "target_ids", None)):
                links.append(
                    {
                        "source": eid,
                        "target": t,
                        "relation": getattr(e, "relation", None) or getattr(e, "label", None),
                        "role": "tgt",
                        "properties": getattr(e, "properties", {}) or {},
                    }
                )
            for te in _safe_iter(getattr(e, "target_edge_ids", None)):
                links.append(
                    {
                        "source": eid,
                        "target": te,
                        "relation": getattr(e, "relation", None) or getattr(e, "label", None),
                        "role": "tgt",
                        "properties": getattr(e, "properties", {}) or {},
                    }
                )
    else:
        for e in edge_map.values():
            for s in _safe_iter(getattr(e, "source_ids", None)):
                for t in _safe_iter(getattr(e, "target_ids", None)):
                    links.append(
                        {
                            "source": s,
                            "target": t,
                            "relation": getattr(e, "relation", None) or getattr(e, "label", None),
                            "properties": getattr(e, "properties", {}) or {},
                        }
                    )

    return {"nodes": list(out_nodes.values()), "links": links, "mode": mode, "doc_id": None}


def _load_node_map(
    engine: GraphKnowledgeEngine,
    ids: List[str],
    node_type: Type[Node] | None = None,
    include=["documents", "metadatas", "embeddings"],
) -> Dict[str, Node]:
    """Robustly load Node models by ids."""
    from ..engine_core.models import Node

    if node_type is None:
        node_type = Node
    if engine.kg_graph_type == "conversation":
        node_type = ConversationNode
    elif engine.kg_graph_type == "workflow":
        node_type = WorkflowNode
    else:
        node_type = Node
    if not ids:
        return {}
    try:
        return engine.read.load_node_map(ids, node_type=node_type)
    except Exception:
        nodes = engine.read.get_nodes(ids=ids, node_type=node_type, include=include)
        out = {n.id: n for n in nodes}
        # for rid, doc in zip(got.get("ids") or [], got.get("documents") or []):
        #     try:
        #         out[rid] = node_type.model_validate_json(doc)
        #     except Exception:
        #         pass
        return out


def _load_edge_map(
    engine: GraphKnowledgeEngine,
    ids: List[str],
    edge_type: Type[Edge] | None = None,
    include=["documents", "metadatas", "embeddings"],
) -> Dict[str, Edge]:
    """Robustly load Edge models by ids."""

    from ..engine_core.models import Edge

    if edge_type is None:
        edge_type = Edge
    if engine.kg_graph_type == "conversation":
        edge_type = ConversationEdge
    elif engine.kg_graph_type == "workflow":
        edge_type = WorkflowEdge
    else:
        edge_type = Edge
    if not ids:
        return {}
    try:
        return engine.read.load_edge_map(ids, edge_type=edge_type)
    except Exception:
        edges = engine.read.get_edges(ids=ids, edge_type=edge_type, include=include)
        out = {n.id: n for n in edges}
        return out


def _ids_by_doc(engine, doc_id: Optional[str]) -> Tuple[List[str], List[str]]:
    """Find ids scoped to a doc (fallback-safe)."""
    if not doc_id:
        # whole-graph fallback (cheap)
        n = engine.backend.node_get()
        e = engine.backend.edge_get()
        return (n.get("ids") or []), (e.get("ids") or [])
    # Prefer engine helpers if present
    try:
        node_ids = engine.read.node_ids_by_doc(doc_id)
    except Exception:
        # fallback: query node_docs table if present
        try:
            rows = engine.backend.node_docs_get(
                where={"doc_id": doc_id}, include=["metadatas"]
            )
            node_ids: list = list(
                {
                    (m or {}).get("node_id")
                    for m in (rows.get("metadatas") or [])
                    if m and m.get("node_id")
                }
            )
        except Exception:
            node_ids = []
    try:
        edge_ids = engine.read.edge_ids_by_doc(doc_id)
    except Exception:
        # fallback: query endpoints table if present
        try:
            eps = engine.backend.edge_endpoints_get(
                where={"doc_id": doc_id}, include=["metadatas"]
            )
            edge_ids: list = list(
                {
                    (m or {}).get("edge_id")
                    for m in (eps.get("metadatas") or [])
                    if m and m.get("edge_id")
                }
            )
        except Exception:
            edge_ids = []
    return node_ids, edge_ids


def _filter_by_insertion_method(
    engine,
    ids: List[str],
    kind: str,  # "node" | "edge"
    insertion_method: Optional[str],
    by_doc_id: Optional[str] = None,
) -> List[str]:
    """Filter ids to those that have at least one ReferenceSession with insertion_method (and optional doc)."""
    if not insertion_method or not ids:
        return ids

    # Fast path: use refs index if present
    coll_attr = "node_refs_collection" if kind == "node" else "edge_refs_collection"

    coll = getattr(engine, coll_attr, None)

    if coll:
        where = {"insertion_method": insertion_method}
        if by_doc_id:
            where = {"$and": [where, {"doc_id": by_doc_id}]}
            # where["doc_id"] = by_doc_id
        rows = coll.get(where=where, include=["metadatas"])
        idx_ids = set(
            (m.get("node_id") if kind == "node" else m.get("edge_id"))
            for m in (rows.get("metadatas") or [])
            if m
        )
        return [rid for rid in ids if rid in idx_ids]
    else:
        # Fallback: scan JSON documents
        store = engine.node_collection if kind == "node" else engine.edge_collection
        got = store.get(ids=ids, include=["documents", "metadatas"])
        keep = []
        for rid, doc in zip(got.get("ids") or [], got.get("documents") or []):
            obj = json.loads(doc)
            refs = obj.get("references") or []
            ok = False
            for r in refs:
                if r.get("insertion_method") != insertion_method:
                    continue
                if by_doc_id:
                    # match direct 'doc_id' or document_page_url that contains doc_id token
                    if r.get("doc_id") == by_doc_id:
                        ok = True
                        break
                    dp = r.get("document_page_url") or ""
                    if by_doc_id in dp:
                        ok = True
                        break
                else:
                    ok = True
                    break
            if ok:
                keep.append(rid)
        return keep


def _collect_ids(
    engine,
    doc_id: Optional[str],
    insertion_method: Optional[str],
) -> Tuple[List[str], List[str]]:
    """Base selection (doc filter) then optional insertion_method filter."""
    node_ids, edge_ids = _ids_by_doc(engine, doc_id)
    node_ids = _filter_by_insertion_method(
        engine, node_ids, "node", insertion_method, by_doc_id=doc_id
    )
    edge_ids = _filter_by_insertion_method(
        engine, edge_ids, "edge", insertion_method, by_doc_id=doc_id
    )
    return node_ids, edge_ids


def to_d3_force(
    engine,
    doc_id: Optional[str] = None,
    mode: str = "reify",  # "reify" | "classic"
    insertion_method: Optional[str] = None,
) -> Dict:
    """
    D3 payload.

    reify:
      - nodes: entity nodes + edge-nodes (type="edge-node")
      - links: entity_src -> edge-node (role=src), edge-node -> entity_tgt (role=tgt)

    classic:
      - nodes: entity nodes
      - links: entity_src -> entity_tgt
    """
    if not hasattr(engine, "kg_graph_type") and isinstance(engine, list) and isinstance(doc_id, list):
        return _render_d3_from_raw(engine, doc_id, mode=mode)

    node_ids, edge_ids = _collect_ids(engine, doc_id, insertion_method)
    node_map = _load_node_map(engine, node_ids)
    edge_map = _load_edge_map(engine, edge_ids)

    nodes: Dict[str, Dict] = {}
    links: List[Dict] = []

    # materialize entity nodes
    for nid, n in node_map.items():
        nodes[nid] = n.model_dump()
        nodes[nid].update(
            {
                "id": nid,
                "label": n.label,
                "type": "entity",
                "summary": n.summary,
                "properties": n.properties or {},
            }
        )

    if mode.lower() == "reify":
        for eid, e in edge_map.items():
            if eid not in nodes:
                nodes[eid] = e.model_dump()
                nodes[eid].update(
                    {
                        "id": eid,
                        "label": e.relation or e.label or "edge",
                        "type": "edge-node",
                        "summary": e.summary,
                        "properties": e.properties or {},
                    }
                )
            # node sources
            for s in _safe_iter(e.source_ids):
                if s in nodes:
                    links.append(
                        {
                            "source": s,
                            "target": eid,
                            "relation": e.relation or e.label,
                            "role": "src",
                            "properties": e.properties or {},
                        }
                    )
                else:
                    raise Exception(f"unexpected path: missing source node {s}")
            # edge sources (MISSING TODAY)
            for se in _safe_iter(e.source_edge_ids):
                if se in edge_map:  # link from another edge-node
                    links.append(
                        {
                            "source": se,
                            "target": eid,
                            "relation": e.relation or e.label,
                            "role": "src",
                            "properties": e.properties or {},
                        }
                    )
                else:
                    raise Exception(f"unexpected path: missing source edge {se}")
            # node targets
            for t in _safe_iter(e.target_ids):
                if t in nodes:
                    links.append(
                        {
                            "source": eid,
                            "target": t,
                            "relation": e.relation or e.label,
                            "role": "tgt",
                            "properties": e.properties or {},
                        }
                    )
                else:
                    raise Exception(f"unexpected path: missing target node {t}")
            # edge targets (MISSING TODAY)
            for te in _safe_iter(e.target_edge_ids):
                if te in edge_map:
                    links.append(
                        {
                            "source": eid,
                            "target": te,
                            "relation": e.relation or e.label,
                            "role": "tgt",
                            "properties": e.properties or {},
                        }
                    )
                else:
                    raise Exception(f"unexpected path: missing target edge {te}")

    else:
        # classic edges: direct src->tgt
        for _, e in edge_map.items():
            for s in _safe_iter(e.source_ids):
                for t in _safe_iter(e.target_ids):
                    if s in nodes and t in nodes:
                        links.append(
                            {
                                "source": s,
                                "target": t,
                                "relation": e.relation or e.label,
                                "properties": e.properties or {},
                            }
                        )

    return {
        "nodes": list(nodes.values()),
        "links": links,
        "mode": mode,
        "doc_id": doc_id,
    }


def to_cytoscape(
    engine,
    doc_id: Optional[str] = None,
    mode: str = "reify",  # "reify" | "classic"
    insertion_method: Optional[str] = None,
) -> Dict:
    """
    Cytoscape payload.

    reify:
      - elements: entity nodes, edge-nodes (class=edge-node)
      - edges: entity_src -> edge-node (class=src), edge-node -> entity_tgt (class=tgt)

    classic:
      - elements: entity nodes, edges: entity_src -> entity_tgt
    """
    if not hasattr(engine, "kg_graph_type") and isinstance(engine, list) and isinstance(doc_id, list):
        d3 = _render_d3_from_raw(engine, doc_id, mode=mode)
        elements = []
        for node in d3["nodes"]:
            elements.append({"data": node})
        for link in d3["links"]:
            elements.append({"data": link})
        return {"elements": elements, "mode": mode, "doc_id": None}

    node_ids, edge_ids = _collect_ids(engine, doc_id, insertion_method)
    node_map = _load_node_map(engine, node_ids)
    edge_map = _load_edge_map(engine, edge_ids)

    elements: List[Dict] = []

    # entity nodes
    for nid, n in node_map.items():
        elements.append(
            {
                "data": {"id": nid, "label": n.label, "type": "entity"},
                "classes": "",
            }
        )

    if mode.lower() == "reify":
        for eid, e in edge_map.items():
            elements.append(
                {
                    "data": {
                        "id": eid,
                        "label": e.relation or e.label or "edge",
                        "type": "edge-node",
                    },
                    "classes": "edge-node",
                }
            )
            # node sources
            for s in _safe_iter(e.source_ids):
                if s in node_map:
                    elements.append(
                        {
                            "data": {
                                "id": f"{eid}::src::{s}",
                                "source": s,
                                "target": eid,
                                "label": e.relation or e.label,
                            },
                            "classes": "src",
                        }
                    )
            # edge sources
            for se in _safe_iter(e.source_edge_ids):
                if se in edge_map:
                    elements.append(
                        {
                            "data": {
                                "id": f"{eid}::srcE::{se}",
                                "source": se,
                                "target": eid,
                                "label": e.relation or e.label,
                            },
                            "classes": "src",
                        }
                    )
            # node targets
            for t in _safe_iter(e.target_ids):
                if t in node_map:
                    elements.append(
                        {
                            "data": {
                                "id": f"{eid}::tgt::{t}",
                                "source": eid,
                                "target": t,
                                "label": e.relation or e.label,
                            },
                            "classes": "tgt",
                        }
                    )
            # edge targets
            for te in _safe_iter(e.target_edge_ids):
                if te in edge_map:
                    elements.append(
                        {
                            "data": {
                                "id": f"{eid}::tgtE::{te}",
                                "source": eid,
                                "target": te,
                                "label": e.relation or e.label,
                            },
                            "classes": "tgt",
                        }
                    )

    else:
        # classic: direct edge
        for eid, e in edge_map.items():
            for s in _safe_iter(e.source_ids):
                for t in _safe_iter(e.target_ids):
                    if s in node_map and t in node_map:
                        elements.append(
                            {
                                "data": {
                                    "id": f"{eid}::{s}->{t}",
                                    "source": s,
                                    "target": t,
                                    "label": e.relation or e.label,
                                },
                                "classes": "",
                            }
                        )

    return {"elements": elements, "mode": mode, "doc_id": doc_id}
