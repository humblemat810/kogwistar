# graph_knowledge_engine/graph_viz.py
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Set
import json
from graph_knowledge_engine.engine import GraphKnowledgeEngine
from ..models import Node, Edge

# -----------------------------
# Cytoscape format
# -----------------------------
# graph_knowledge_engine/visualization/graph_viz.py
from typing import Optional, Iterable, Tuple
from graph_knowledge_engine.models import Node, Edge

def _entity_has_insertion_method(engine, kind: str, rid: str, value: str) -> bool:
    """Slow-but-correct fallback: read JSON and scan references."""
    coll = engine.edge_collection if kind == "edge" else engine.node_collection
    got = coll.get(ids=[rid], include=["documents"])
    if not got.get("documents"):
        return False
    model_cls = Edge if kind == "edge" else Node
    ent = model_cls.model_validate_json(got["documents"][0])
    refs = getattr(ent, "references", None) or []
    for r in refs:
        # support both attribute and dict forms
        im = getattr(r, "insertion_method", None)
        if im is None and isinstance(r, dict):
            im = r.get("insertion_method")
        if im == value:
            return True
    return False

def _filter_ids_by_insertion_method(
    engine,
    node_ids: Iterable[str],
    edge_ids: Iterable[str],
    insertion_method: Optional[str],
    doc_id: Optional[str],
) -> Tuple[list[str], list[str]]:
    if not insertion_method:
        return list(node_ids), list(edge_ids)

    # Fast path if you added ref indexes:
    # try engine.ids_with_insertion_method(...). If not present, fall back:
    try:
        keep_nodes = set(engine.ids_with_insertion_method(
            kind="node", ids=list(node_ids), insertion_method=insertion_method, doc_id=doc_id
        ))
        keep_edges = set(engine.ids_with_insertion_method(
            kind="edge", ids=list(edge_ids), insertion_method=insertion_method, doc_id=doc_id
        ))
    except AttributeError:
        keep_nodes = {nid for nid in node_ids if _entity_has_insertion_method(engine, "node", nid, insertion_method)}
        keep_edges = {eid for eid in edge_ids if _entity_has_insertion_method(engine, "edge", eid, insertion_method)}

    # Optional: prune edges whose endpoints are gone (nicer drawings)
    if keep_edges:
        # load endpoints cheaply from edge_endpoints table if available
        eps = engine.edge_endpoints_collection.get(
            where={"edge_id": {"$in": list(keep_edges)}}, include=["metadatas"]
        )
        ok_nodes = keep_nodes
        pruned_edges = set()
        for meta in eps.get("metadatas") or []:
            eid = meta.get("edge_id")
            nid = meta.get("node_id")
            if eid and (nid is None or nid in ok_nodes):
                pruned_edges.add(eid)
        keep_edges = pruned_edges or keep_edges

    return sorted(keep_nodes), sorted(keep_edges)
from .basic_visualization import Visualizer

def to_cytoscape(engine, doc_id: Optional[str] = None, mode: str = "reify",
                 insertion_method: Optional[str] = None):
    visualiser = Visualizer(engine=engine)
    """
    Build Cytoscape JSON. If `insertion_method` is provided, only include
    nodes/edges that have at least one ReferenceSession with that value.
    """
    # 1) get the raw ids you would normally visualize (whatever you do today)
    ids = visualiser.resolve_readable(by_doc_id=doc_id)  # or your existing way
    node_ids = [n["id"] for n in ids["nodes"]]
    edge_ids = [e["id"] for e in ids["edges"]]

    # 2) filter (server-side)
    node_ids, edge_ids = _filter_ids_by_insertion_method(engine, node_ids, edge_ids, insertion_method, doc_id)

    # 3) build the actual elements using your existing logic, but only for filtered ids
    #    (pseudocode below — adapt to your current builder)
    elements = []
    node_map = visualiser._load_node_map(node_ids)
    for nid in node_ids:
        n = node_map[nid]
        elements.append({"data": {"id": nid, "label": n["label"]}, "classes": ""})

    edge_map = visualiser._load_edge_map(edge_ids)
    for eid in edge_ids:
        e = edge_map[eid]
        # normal edge
        for s in e["source_ids"] or []:
            for t in e["target_ids"] or []:
                if s in node_ids and t in node_ids:
                    elements.append({
                        "data": {"id": f"{eid}:{s}->{t}", "source": s, "target": t, "label": e["relation"]},
                    })
        # (if you draw reified “edge-nodes”, keep your existing logic here)

    return {"elements": elements, "mode": mode, "doc_id": doc_id}

def to_d3_force(engine, doc_id: Optional[str] = None, mode: str = "reify",
                insertion_method: Optional[str] = None):
    """
    Build D3-friendly JSON with optional server-side insertion_method filter.
    """
    visualiser = Visualizer(engine=engine)
    ids = visualiser.resolve_readable(by_doc_id=doc_id)
    node_ids = [n["id"] for n in ids["nodes"]]
    edge_ids = [e["id"] for e in ids["edges"]]

    node_ids, edge_ids = _filter_ids_by_insertion_method(engine, node_ids, edge_ids, insertion_method, doc_id)

    nodes = [{"id": nid, "label": visualiser._load_node_map([nid])[nid]["label"], "type": "node"} for nid in node_ids]
    links = []
    for eid in edge_ids:
        e = visualiser._load_edge_map([eid])[eid]
        for s in e["source_ids"] or []:
            for t in e["target_ids"] or []:
                if s in node_ids and t in node_ids:
                    links.append({"source": s, "target": t, "id": eid, "label": e["relation"]})
    return {"nodes": nodes, "links": links, "mode": mode, "doc_id": doc_id}
