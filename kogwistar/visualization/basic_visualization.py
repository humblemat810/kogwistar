import json
from collections.abc import Iterable, Mapping
from typing import cast

from ..engine_core.engine import GraphKnowledgeEngine
from ..engine_core.models import Edge, Node


def _as_metadata(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _parse_json_list(value: object) -> list[object]:
    if isinstance(value, str):
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    return list(value) if isinstance(value, list) else []


def _as_list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _as_strings(value: object) -> list[str]:
    return [item for item in _as_list(value) if isinstance(item, str)]


def _as_metadata_list(value: object) -> list[Mapping[str, object]]:
    return [item for item in _as_list(value) if isinstance(item, Mapping)]


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
    def __init__(self, engine: GraphKnowledgeEngine) -> None:
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
            _as_strings(_as_metadata(got).get("ids")),
            _as_strings(_as_metadata(got).get("documents")),
            _as_metadata_list(_as_metadata(got).get("metadatas")),
        ):
            if not nid:
                continue
            metadata = _as_metadata(meta)
            try:
                n = Node.model_validate_json(ndoc)
                out[nid] = {
                    "label": n.label,
                    "type": n.type,
                    "summary": getattr(n, "summary", "") or "",
                    "doc_ids": _parse_json_list(metadata.get("doc_ids")),
                }
            except Exception:
                # fallback if pydantic fails
                out[nid] = {
                    "label": metadata.get("label") or "(node)",
                    "type": metadata.get("type") or "entity",
                    "summary": metadata.get("summary") or "",
                    "doc_ids": _parse_json_list(metadata.get("doc_ids")),
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
            _as_strings(_as_metadata(got).get("ids")),
            _as_strings(_as_metadata(got).get("documents")),
            _as_metadata_list(_as_metadata(got).get("metadatas")),
        ):
            if not eid:
                continue
            metadata = _as_metadata(meta)
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
                def parse_ids(k: str) -> list[str]:
                    v = metadata.get(k)
                    try:
                        raw = json.loads(v) if isinstance(v, str) else v
                        return _as_strings(raw)
                    except Exception:
                        return []

                out[eid] = {
                    "label": metadata.get("label") or "(edge)",
                    "relation": metadata.get("relation") or "",
                    "summary": metadata.get("summary") or "",
                    "source_ids": parse_ids("source_ids"),
                    "target_ids": parse_ids("target_ids"),
                    "source_edge_ids": parse_ids("source_edge_ids"),
                    "target_edge_ids": parse_ids("target_edge_ids"),
                }
        return out

    def resolve_readable(
        self,
        *,
        node_ids: Iterable[str] | None = None,
        edge_ids: Iterable[str] | None = None,
        by_doc_id: str | None = None,
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
                json.loads(doc)["node_id"]
                for doc in _as_strings(_as_metadata(n_links).get("documents"))
            ]

            # Pull edge_ids by scanning edge_endpoints for that doc_id
            e_links = self.e.edge_endpoints_collection.get(
                where={"doc_id": by_doc_id}, include=["documents"]
            )
            edge_ids = list(
                {
                    json.loads(doc)["edge_id"]
                    for doc in _as_strings(_as_metadata(e_links).get("documents"))
                }
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
                _as_strings(_as_metadata(got).get("ids")),
                _as_metadata_list(_as_metadata(got).get("metadatas")),
                _as_strings(_as_metadata(got).get("documents")),
            ):
                m = node_map.get(nid, {})
                metadata = _as_metadata(meta)
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
                        raw = metadata.get("references")
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
                _as_strings(_as_metadata(got).get("ids")),
                _as_metadata_list(_as_metadata(got).get("metadatas")),
                _as_strings(_as_metadata(got).get("documents")),
            ):
                m = edge_map.get(eid, {})
                metadata = _as_metadata(meta)

                # resolve endpoint labels
                def resolve_list(
                    ids: Iterable[str], kind_hint: str
                ) -> list[dict[str, str]]:
                    items: list[dict[str, str]] = []
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
                        raw = metadata.get("references")
                        if isinstance(raw, str):
                            try:
                                for r in json.loads(raw) or []:
                                    refs.append(_fmt_span_short(r))
                            except Exception:
                                pass
                        entry["refs"] = refs

                edge_out.append(entry)

        return {"nodes": node_out, "edges": edge_out}

    def pretty_print_graph(self, **kwargs: object) -> str:
        """
        Thin wrapper over resolve_readable() that renders a compact text block.
        kwargs are passed to resolve_readable (node_ids, edge_ids, by_doc_id, include_refs).
        """
        typed_kwargs = cast(dict[str, object], kwargs)
        data = self.resolve_readable(
            node_ids=cast(Iterable[str] | None, typed_kwargs.get("node_ids")),
            edge_ids=cast(Iterable[str] | None, typed_kwargs.get("edge_ids")),
            by_doc_id=cast(str | None, typed_kwargs.get("by_doc_id")),
            include_refs=bool(typed_kwargs.get("include_refs", False)),
        )
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

                def fmt_endpoints(items: Iterable[Mapping[str, object]]) -> str:
                    return ", ".join(
                        f"{i.get('label', '')}({str(i.get('id', ''))[:8]})" for i in items
                    )

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
