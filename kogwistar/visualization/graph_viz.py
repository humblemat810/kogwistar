# kogwistar/visualization/graph_viz.py
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, TypeVar, cast

from ..conversation.models import ConversationEdge, ConversationNode
from ..runtime.models import WorkflowEdge, WorkflowNode

if TYPE_CHECKING:
    from kogwistar.engine_core.engine import GraphKnowledgeEngine

    from ..engine_core.models import (
        Edge,
        Node,
    )  # , ConversationNode, ConversationEdge, WorkflowEdge, WorkflowNode

    T = TypeVar("T", bound=Node | Edge)


def _safe_iter(x: object) -> list[object]:
    return x if isinstance(x, list) and x else []


def _object_mapping(value: object) -> Mapping[str, object]:
    return cast(Mapping[str, object], value) if isinstance(value, Mapping) else {}


def _string_list(value: object) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _object_id(value: object) -> str | None:
    identifier = getattr(value, "id", None)
    return str(identifier) if identifier else None


def _render_d3_from_raw(
    nodes: Sequence[object], edges: Sequence[object], mode: str = "reify"
) -> dict[str, object]:
    node_map = {
        identifier: n
        for n in nodes
        if (identifier := _object_id(n)) is not None
    }
    edge_map = {
        identifier: e
        for e in edges
        if (identifier := _object_id(e)) is not None
    }

    out_nodes: dict[str, dict] = {}
    links: list[dict] = []

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
    ids: list[str],
    node_type: type[Node] | None = None,
    include: Sequence[str] | None = None,
) -> dict[str, Node]:
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
    if include is None:
        include = ("documents", "metadatas", "embeddings")
    if not ids:
        return {}
    try:
        return engine.read.load_node_map(ids, node_type=node_type)
    except Exception:
        nodes = engine.read.get_nodes(
            ids=ids,
            node_type=node_type,
            include=None if include is None else list(include),
        )
        out = {identifier: n for n in nodes if (identifier := _object_id(n)) is not None}
        # for rid, doc in zip(got.get("ids") or [], got.get("documents") or []):
        #     try:
        #         out[rid] = node_type.model_validate_json(doc)
        #     except Exception:
        #         pass
        return out


def _load_edge_map(
    engine: GraphKnowledgeEngine,
    ids: list[str],
    edge_type: type[Edge] | None = None,
    include: Sequence[str] | None = None,
) -> dict[str, Edge]:
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
    if include is None:
        include = ("documents", "metadatas", "embeddings")
    if not ids:
        return {}
    try:
        return engine.read.load_edge_map(ids, edge_type=edge_type)
    except Exception:
        edges = engine.read.get_edges(
            ids=ids,
            edge_type=edge_type,
            include=None if include is None else list(include),
        )
        out = {identifier: n for n in edges if (identifier := _object_id(n)) is not None}
        return out


def _ids_by_doc(
    engine: GraphKnowledgeEngine, doc_id: str | None
) -> tuple[list[str], list[str]]:
    """Find ids scoped to a doc (fallback-safe)."""
    def _scan_all(
        kind: str,
    ) -> tuple[list[str], list[str], list[Mapping[str, object]]]:
        getter = getattr(engine.backend, f"{kind}_get", None)
        if not callable(getter):
            return [], [], []
        got = _object_mapping(getter(include=["documents", "metadatas"]))
        return (
            _string_list(got.get("ids")),
            [str(item) for item in _safe_iter(got.get("documents"))],
            [
                _object_mapping(item)
                for item in _safe_iter(got.get("metadatas"))
            ],
        )

    def _match_doc_id(
        doc: str | None, meta: Mapping[str, object] | None
    ) -> bool:
        if not doc_id:
            return True
        if meta and meta.get("doc_id") == doc_id:
            return True
        if meta:
            doc_ids = meta.get("doc_ids")
            if isinstance(doc_ids, list) and doc_id in doc_ids:
                return True
        if doc:
            try:
                obj = json.loads(doc)
            except Exception:
                return False
            obj_mapping = _object_mapping(obj)
            if obj_mapping.get("doc_id") == doc_id:
                return True
            md = obj_mapping.get("metadata")
            if isinstance(md, Mapping) and md.get("doc_id") == doc_id:
                return True
        return False

    if not doc_id:
        # whole-graph fallback (cheap)
        n = _object_mapping(engine.backend.node_get())
        e = _object_mapping(engine.backend.edge_get())
        return _string_list(n.get("ids")), _string_list(e.get("ids"))
    # Prefer engine helpers if present
    try:
        node_ids = engine.read.node_ids_by_doc(doc_id)
    except Exception:
        node_ids = []
    if not node_ids:
        try:
            rows = _object_mapping(engine.backend.node_docs_get(
                where={"doc_id": doc_id}, include=["metadatas"]
            ))
            node_ids = list(
                {
                    str(metadata["node_id"])
                    for metadata in (
                        _object_mapping(item)
                        for item in _safe_iter(rows.get("metadatas"))
                    )
                    if metadata.get("node_id")
                }
            )
        except Exception:
            node_ids = []
    if not node_ids:
        try:
            ids, docs, metas = _scan_all("node")
            node_ids = [
                rid
                for rid, doc, meta in zip(ids, docs, metas)
                if _match_doc_id(doc, meta)
            ]
        except Exception:
            node_ids = []
    try:
        edge_ids = engine.read.edge_ids_by_doc(doc_id)
    except Exception:
        edge_ids = []
    if not edge_ids:
        try:
            eps = _object_mapping(engine.backend.edge_endpoints_get(
                where={"doc_id": doc_id}, include=["metadatas"]
            ))
            edge_ids = list(
                {
                    str(metadata["edge_id"])
                    for metadata in (
                        _object_mapping(item)
                        for item in _safe_iter(eps.get("metadatas"))
                    )
                    if metadata.get("edge_id")
                }
            )
        except Exception:
            edge_ids = []
    if not edge_ids:
        try:
            ids, docs, metas = _scan_all("edge")
            edge_ids = [
                rid
                for rid, doc, meta in zip(ids, docs, metas)
                if _match_doc_id(doc, meta)
            ]
        except Exception:
            edge_ids = []
    return node_ids, edge_ids


def _filter_by_insertion_method(
    engine: GraphKnowledgeEngine,
    ids: list[str],
    kind: str,  # "node" | "edge"
    insertion_method: str | None,
    by_doc_id: str | None = None,
) -> list[str]:
    """Filter ids to those that have at least one ReferenceSession with insertion_method (and optional doc)."""
    if not insertion_method or not ids:
        return ids

    def _extract_refs(obj: dict) -> list[dict]:
        refs = obj.get("references")
        if not refs:
            refs = obj.get("mentions")
        return refs if isinstance(refs, list) else []

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
        kept = [rid for rid in ids if rid in idx_ids]
        return kept or ids
    else:
        # Fallback: scan JSON documents
        store_owner = getattr(engine, "backend", engine)
        getter = getattr(store_owner, f"{kind}_get", None)
        store = getattr(store_owner, f"{kind}_collection", None)
        if callable(getter):
            got = _object_mapping(
                getter(ids=ids, include=["documents", "metadatas"])
            )
            keep = []
            for rid, doc in zip(
                _string_list(got.get("ids")),
                [str(item) for item in _safe_iter(got.get("documents"))],
            ):
                obj = json.loads(doc)
                refs = _extract_refs(obj)
                ok = False
                for r in refs:
                    if r.get("insertion_method") != insertion_method:
                        continue
                    if by_doc_id:
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
            return keep or ids
        if store is None:
            return ids
        got = store.get(ids=ids, include=["documents", "metadatas"])
        keep = []
        for rid, doc in zip(got.get("ids") or [], got.get("documents") or []):
            obj = json.loads(doc)
            refs = _extract_refs(obj)
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
        return keep or ids


def _collect_ids(
    engine: GraphKnowledgeEngine,
    doc_id: str | None,
    insertion_method: str | None,
) -> tuple[list[str], list[str]]:
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
    engine: object,
    doc_id: str | None = None,
    mode: str = "reify",  # "reify" | "classic"
    insertion_method: str | None = None,
) -> dict:
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

    engine_obj = cast("GraphKnowledgeEngine", engine)
    node_ids, edge_ids = _collect_ids(engine_obj, doc_id, insertion_method)
    node_map = _load_node_map(engine_obj, node_ids)
    edge_map = _load_edge_map(engine_obj, edge_ids)

    nodes: dict[str, dict] = {}
    links: list[dict] = []

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


def to_sigma_hypergraph(
    engine: object,
    doc_id: str | None = None,
    insertion_method: str | None = None,
    ) -> dict[str, object]:
    """Return the lossless raw hypergraph contract used by the Sigma viewer.

    Unlike a D3 force projection, this payload keeps hyperedges as first-class
    records with all four endpoint collections. Render modes are browser-side
    projections of this raw model.
    """

    engine_obj = cast("GraphKnowledgeEngine", engine)
    node_ids, edge_ids = _collect_ids(engine_obj, doc_id, insertion_method)
    node_map = _load_node_map(engine_obj, node_ids)
    edge_map = _load_edge_map(engine_obj, edge_ids)

    def _dump(value: object) -> dict[str, object]:
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            dumped = model_dump(exclude={"embedding"})
            if isinstance(dumped, Mapping):
                return {str(key): item for key, item in dumped.items()}
        if isinstance(value, dict):
            return dict(value)
        raise TypeError(f"unsupported visualization value: {type(value)!r}")

    raw_nodes = []
    for node_id, node in node_map.items():
        item = _dump(node)
        item["id"] = node_id
        item.setdefault("label", getattr(node, "label", node_id))
        item.setdefault("properties", getattr(node, "properties", {}) or {})
        raw_nodes.append(item)

    raw_edges = []
    for edge_id, edge in edge_map.items():
        item = _dump(edge)
        item.update(
            {
                "id": edge_id,
                "relation": getattr(edge, "relation", None)
                or getattr(edge, "label", None)
                or "edge",
                "source_ids": list(getattr(edge, "source_ids", None) or []),
                "target_ids": list(getattr(edge, "target_ids", None) or []),
                "source_edge_ids": list(
                    getattr(edge, "source_edge_ids", None) or []
                ),
                "target_edge_ids": list(
                    getattr(edge, "target_edge_ids", None) or []
                ),
                "properties": getattr(edge, "properties", {}) or {},
            }
        )
        raw_edges.append(item)

    return {
        "raw_nodes": raw_nodes,
        "raw_edges": raw_edges,
        "mode": "raw-hypergraph",
        "doc_id": doc_id,
    }


def to_cytoscape(
    engine: object,
    doc_id: str | None = None,
    mode: str = "reify",  # "reify" | "classic"
    insertion_method: str | None = None,
) -> dict:
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
        for node in _safe_iter(d3.get("nodes")):
            elements.append({"data": node})
        for link in _safe_iter(d3.get("links")):
            elements.append({"data": link})
        return {"elements": elements, "mode": mode, "doc_id": None}

    engine_obj = cast("GraphKnowledgeEngine", engine)
    node_ids, edge_ids = _collect_ids(engine_obj, doc_id, insertion_method)
    node_map = _load_node_map(engine_obj, node_ids)
    edge_map = _load_edge_map(engine_obj, edge_ids)

    elements: list[dict] = []

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
