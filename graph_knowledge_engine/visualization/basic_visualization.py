from ..engine_core.engine import GraphKnowledgeEngine
from ..engine_core.models import Node, Edge
from typing import Iterable, Optional
import json


def _fmt_span_short(r: dict) -> str:
    # expects a dict (already model_dump()'d) ReferenceSession
    pg = ""
    if r.get("start_page") is not None and r.get("end_page") is not None:
        if r["start_page"] == r["end_page"]:
            pg = f"p{r['start_page']}"
        else:
            pg = f"p{r['start_page']}-{r['end_page']}"
    span = ""
    if r.get("start_char") is not None and r.get("end_char") is not None:
        span = f":{r['start_char']}-{r['end_char']}"
    url = r.get("document_page_url") or r.get("collection_page_url") or ""
    snip = r.get("excerpt") or ""
    snip = (snip[:60] + "…") if len(snip) > 60 else snip
    return f"{pg}{span} @{url}  “{snip}”".strip()


class Visualizer:
    def __init__(self, engine: GraphKnowledgeEngine):
        self.e = engine
        pass

    # ----------------------------
    # Visualization
    # ----------------------------
    def _load_node_map(self, ids: Iterable[str]) -> dict[str, dict]:
        """Return {id: {'label':..., 'type':..., 'summary':..., 'doc_ids': [...]}}, missing ids omitted."""
        ids = list(
            dict.fromkeys([i for i in ids if i])
        )  # dedupe/preserve order, drop falsy
        out: dict[str, dict] = {}
        if not ids:
            return out
        got = self.e.node_collection.get(ids=ids, include=["documents", "metadatas"])
        for nid, ndoc, meta in zip(
            got.get("ids") or [], got.get("documents") or [], got.get("metadatas") or []
        ):
            if not nid:
                continue
            try:
                n = Node.model_validate_json(ndoc)
                out[nid] = {
                    "label": n.label,
                    "type": n.type,
                    "summary": getattr(n, "summary", "") or "",
                    "doc_ids": json.loads((meta or {}).get("doc_ids") or "[]"),
                }
            except Exception:
                # fallback if pydantic fails
                out[nid] = {
                    "label": (meta or {}).get("label") or "(node)",
                    "type": (meta or {}).get("type") or "entity",
                    "summary": (meta or {}).get("summary") or "",
                    "doc_ids": json.loads((meta or {}).get("doc_ids") or "[]"),
                }
        return out

    def _load_edge_map(self, ids: Iterable[str]) -> dict[str, dict]:
        """Return {id: {'relation':..., 'source_ids':[...], 'target_ids':[...], 'label':..., 'summary':...}}."""
        ids = list(dict.fromkeys([i for i in ids if i]))
        out: dict[str, dict] = {}
        if not ids:
            return out
        got = self.e.edge_collection.get(ids=ids, include=["documents", "metadatas"])
        for eid, edoc, meta in zip(
            got.get("ids") or [], got.get("documents") or [], got.get("metadatas") or []
        ):
            if not eid:
                continue
            try:
                e = Edge.model_validate_json(edoc)
                out[eid] = {
                    "label": e.label,
                    "relation": e.relation,
                    "summary": getattr(e, "summary", "") or "",
                    "source_ids": e.source_ids or [],
                    "target_ids": e.target_ids or [],
                    "source_edge_ids": getattr(e, "source_edge_ids", []) or [],
                    "target_edge_ids": getattr(e, "target_edge_ids", []) or [],
                }
            except Exception:
                # metadata fallback (source/target ids may be stored as JSON strings)
                def parse_ids(k):
                    v = (meta or {}).get(k)
                    try:
                        return json.loads(v) if isinstance(v, str) else (v or [])
                    except Exception:
                        return v or []

                out[eid] = {
                    "label": (meta or {}).get("label") or "(edge)",
                    "relation": (meta or {}).get("relation") or "",
                    "summary": (meta or {}).get("summary") or "",
                    "source_ids": parse_ids("source_ids"),
                    "target_ids": parse_ids("target_ids"),
                    "source_edge_ids": parse_ids("source_edge_ids"),
                    "target_edge_ids": parse_ids("target_edge_ids"),
                }
        return out

    def resolve_readable(
        self,
        *,
        node_ids: Optional[Iterable[str]] = None,
        edge_ids: Optional[Iterable[str]] = None,
        by_doc_id: Optional[str] = None,
        include_refs: bool = False,
    ) -> dict:
        """
        Structured, human-readable snapshot.
        - If by_doc_id is given, it overrides explicit node_ids/edge_ids (it pulls all linked).
        - include_refs: adds compact reference strings (can be heavy).
        Returns:
        {
            "nodes": [{"id":..., "label":..., "type":..., "summary":..., "doc_ids":[...] , "refs":[...] }],
            "edges": [{"id":..., "relation":..., "summary":..., "sources":[{"id":..., "kind":"node|edge", "label":...}], "targets":[...], "refs":[...]}]
        }
        """
        # 1) Fetch ids from a doc if requested
        if by_doc_id:
            # Pull all node_ids linked to the given doc_id from the fast index
            n_links = self.e.node_docs_collection.get(
                where={"doc_id": by_doc_id}, include=["documents"]
            )
            node_ids = [
                json.loads(doc)["node_id"] for doc in (n_links.get("documents") or [])
            ]

            # Pull edge_ids by scanning edge_endpoints for that doc_id
            e_links = self.e.edge_endpoints_collection.get(
                where={"doc_id": by_doc_id}, include=["documents"]
            )
            edge_ids = list(
                {json.loads(doc)["edge_id"] for doc in (e_links.get("documents") or [])}
            )
        node_ids = list(dict.fromkeys(node_ids or []))
        edge_ids = list(dict.fromkeys(edge_ids or []))

        # 2) Build maps
        node_map = self._load_node_map(node_ids)
        edge_map = self._load_edge_map(edge_ids)

        # 3) If edges point to additional nodes/edges not explicitly requested, load them too
        extra_nodes, extra_edges = set(), set()
        for em in edge_map.values():
            extra_nodes.update(em.get("source_ids", []))
            extra_nodes.update(em.get("target_ids", []))
            extra_edges.update(em.get("source_edge_ids", []))
            extra_edges.update(em.get("target_edge_ids", []))
        # load missing
        missing_nodes = [i for i in extra_nodes if i not in node_map]
        missing_edges = [i for i in extra_edges if i not in edge_map]
        node_map.update(self._load_node_map(missing_nodes))
        edge_map.update(self._load_edge_map(missing_edges))

        # 4) Optionally load refs for nodes/edges
        node_out = []
        if node_map:
            got = self.e.node_collection.get(
                ids=list(node_map.keys()), include=["metadatas", "documents"]
            )
            for nid, meta, ndoc in zip(
                got.get("ids") or [],
                got.get("metadatas") or [],
                got.get("documents") or [],
            ):
                m = node_map.get(nid, {})
                entry = {"id": nid, **m}
                if include_refs:
                    try:
                        n = Node.model_validate_json(ndoc)
                        entry["refs"] = [
                            _fmt_span_short(r.model_dump()) for r in (n.mentions or [])
                        ]
                    except Exception:
                        # try metadata path
                        refs = []
                        raw = (meta or {}).get("references")
                        if isinstance(raw, str):
                            try:
                                for r in json.loads(raw) or []:
                                    refs.append(_fmt_span_short(r))
                            except Exception:
                                pass
                        entry["refs"] = refs
                node_out.append(entry)

        edge_out = []
        if edge_map:
            got = self.e.edge_collection.get(
                ids=list(edge_map.keys()), include=["metadatas", "documents"]
            )
            for eid, meta, edoc in zip(
                got.get("ids") or [],
                got.get("metadatas") or [],
                got.get("documents") or [],
            ):
                m = edge_map.get(eid, {})

                # resolve endpoint labels
                def resolve_list(ids, kind_hint: str):
                    items = []
                    for rid in ids:
                        if rid in node_map:
                            items.append(
                                {
                                    "id": rid,
                                    "kind": "node",
                                    "label": node_map[rid]["label"],
                                }
                            )
                        elif rid in edge_map:
                            items.append(
                                {
                                    "id": rid,
                                    "kind": "edge",
                                    "label": edge_map[rid]["label"],
                                }
                            )
                        else:
                            items.append(
                                {"id": rid, "kind": kind_hint, "label": "(missing)"}
                            )
                    return items

                entry = {
                    "id": eid,
                    "relation": m.get("relation", ""),
                    "label": m.get("label", ""),
                    "summary": m.get("summary", ""),
                    "sources": resolve_list(m.get("source_ids", []), "node"),
                    "targets": resolve_list(m.get("target_ids", []), "node"),
                }
                # include edge-endpoint edges if you use them
                se = m.get("source_edge_ids") or []
                te = m.get("target_edge_ids") or []
                if se or te:
                    entry["source_edges"] = resolve_list(se, "edge")
                    entry["target_edges"] = resolve_list(te, "edge")

                if include_refs:
                    try:
                        e = Edge.model_validate_json(edoc)
                        entry["refs"] = [
                            _fmt_span_short(r.model_dump()) for r in (e.mentions or [])
                        ]
                    except Exception:
                        refs = []
                        raw = (meta or {}).get("references")
                        if isinstance(raw, str):
                            try:
                                for r in json.loads(raw) or []:
                                    refs.append(_fmt_span_short(r))
                            except Exception:
                                pass
                        entry["refs"] = refs

                edge_out.append(entry)

        return {"nodes": node_out, "edges": edge_out}

    def pretty_print_graph(self, **kwargs) -> str:
        """
        Thin wrapper over resolve_readable() that renders a compact text block.
        kwargs are passed to resolve_readable (node_ids, edge_ids, by_doc_id, include_refs).
        """
        data = self.resolve_readable(**kwargs)
        lines = []
        if data["nodes"]:
            lines.append("Nodes:")
            for n in data["nodes"]:
                line = f"  • {n['id']}  [{n['type']}]  {n['label']}"
                if n.get("summary"):
                    line += f" — {n['summary']}"
                if n.get("doc_ids"):
                    line += f"  (docs: {', '.join(n['doc_ids'])})"
                lines.append(line)
                if kwargs.get("include_refs") and n.get("refs"):
                    for r in n["refs"]:
                        lines.append(f"     ↳ {r}")
        if data["edges"]:
            lines.append("Edges:")
            for e in data["edges"]:

                def fmt_endpoints(items):
                    return ", ".join([f"{i['label']}({i['id'][:8]})" for i in items])

                src = fmt_endpoints(e.get("sources", []))
                tgt = fmt_endpoints(e.get("targets", []))
                line = f"  → {e['id']}  [{e.get('relation', '')}]  {src}  ->  {tgt}"
                if e.get("summary"):
                    line += f" — {e['summary']}"
                lines.append(line)
                if e.get("source_edges") or e.get("target_edges"):
                    ss = fmt_endpoints(e.get("source_edges", []))
                    tt = fmt_endpoints(e.get("target_edges", []))
                    if ss:
                        lines.append(f"     (source-edges: {ss})")
                    if tt:
                        lines.append(f"     (target-edges: {tt})")
                if kwargs.get("include_refs") and e.get("refs"):
                    for r in e["refs"]:
                        lines.append(f"     ↳ {r}")
        return "\n".join(lines) or "(empty)"
