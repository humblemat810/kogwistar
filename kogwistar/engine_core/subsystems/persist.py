from __future__ import annotations

import json
import uuid
from graphlib import TopologicalSorter
from typing import Any, cast

from ..models import (
    Document,
    Edge,
    GraphExtractionWithIDs,
    Grounding,
    LLMGraphExtraction,
    Node,
    PureChromaEdge,
    PureChromaNode,
    PureGraph,
    Span,
)
from ..utils.aliasing import _is_alias, _is_new_edge, _is_new_node, _is_uuid
from ..utils.refs import merge_refs
from .base import NamespaceProxy


class PersistSubsystem(NamespaceProxy):
    def __init__(self, engine) -> None:
        super().__init__(engine)

    @staticmethod
    def _promote_llm_entity_payload(obj, *, insertion_method: str) -> dict[str, Any]:
        payload = obj.model_dump(field_mode="llm")
        payload["mentions"] = [
            grounding.model_dump(field_mode="backend")
            for grounding in (getattr(obj, "mentions", None) or [])
        ]
        for grounding in payload.get("mentions") or []:
            for span in grounding.get("spans") or []:
                if span.get("insertion_method") is None:
                    span["insertion_method"] = insertion_method
        return payload

    @staticmethod
    def _alloc_real_ids(parsed):
        nn2id, ne2id = {}, {}

        def map_id(x: str) -> str:
            if x.startswith("nn:"):
                nn2id.setdefault(x, str(uuid.uuid4()))
                return nn2id[x]
            if x.startswith("ne:"):
                ne2id.setdefault(x, str(uuid.uuid4()))
                return ne2id[x]
            return x

        for n in parsed.nodes or []:
            if n.id:
                n.id = map_id(n.id)
        for e in parsed.edges or []:
            if e.id:
                e.id = map_id(e.id)
            e.source_ids = [map_id(x) for x in (e.source_ids or [])]
            e.target_ids = [map_id(x) for x in (e.target_ids or [])]
            if hasattr(e, "source_edge_ids"):
                e.source_edge_ids = [map_id(x) for x in (e.source_edge_ids or [])]
            if hasattr(e, "target_edge_ids"):
                e.target_edge_ids = [map_id(x) for x in (e.target_edge_ids or [])]
        return nn2id, ne2id

    def preflight_validate(
        self,
        parsed: LLMGraphExtraction | PureGraph | GraphExtractionWithIDs,
        alias_key: str,
        alias_book=None,
    ):
        self.resolve_llm_ids(alias_key, parsed, alias_book=alias_book)

        batch_node_ids = {n.id for n in parsed.nodes}
        batch_edge_ids = {e.id for e in parsed.edges}

        need_nodes, need_edges = set(), set()
        for e in parsed.edges:
            need_nodes.update(e.source_ids or [])
            need_nodes.update(e.target_ids or [])
            if getattr(e, "source_edge_ids", None):
                need_edges.update(e.source_edge_ids or [])
            if getattr(e, "target_edge_ids", None):
                need_edges.update(e.target_edge_ids or [])

        need_nodes -= batch_node_ids
        need_edges -= batch_edge_ids

        missing_nodes, missing_edges = set(), set()
        if need_nodes:
            got = set(self._e.backend.node_get(ids=list(need_nodes)).get("ids") or [])
            missing_nodes = need_nodes - got
        if need_edges:
            got = set(self._e.backend.edge_get(ids=list(need_edges)).get("ids") or [])
            missing_edges = need_edges - got

        if missing_nodes or missing_edges:
            raise ValueError(
                f"Dangling references: nodes={sorted(missing_nodes)}, edges={sorted(missing_edges)}"
            )
        return batch_node_ids, batch_edge_ids

    def resolve_llm_ids(
        self,
        doc_id: str,
        parsed: LLMGraphExtraction | PureGraph | GraphExtractionWithIDs,
        alias_book=None,
    ) -> None:
        """Resolve graph-extraction identifiers into persisted backend IDs.

        This resolver is meant for newly parsed extraction payloads. It accepts
        batch-local `nn:<slug>` / `ne:<slug>` tokens, resolvable aliases,
        canonical UUIDs, and the limited node-label fallback already used by the
        extraction pipeline.

        Same-batch `nn:*` tokens are rewritten to a shared backend ID so that
        edges may point to nodes created in the same payload.
        """
        if alias_book is None:
            book = self._e._alias_book(doc_id)
        else:
            book = alias_book
        alias_to_real = book.alias_to_real

        def de_alias(x: str) -> str:
            if not x:
                return x
            if _is_uuid(x):
                return x
            return alias_to_real.get(x, x)

        nn2uuid: dict[str, str] = {}
        for n in parsed.nodes:
            token = cast(str | None, getattr(n, "local_id", None)) or n.id
            
            if token is None or token == "":
                n.id = str(uuid.uuid4())
                continue

            if _is_new_node(token):
                rid = nn2uuid.get(token)
                if rid is None:
                    rid = str(uuid.uuid4())
                    nn2uuid[token] = rid
                n.id = rid
            else:
                n.id = de_alias(token)

        ne2uuid: dict[str, str] = {}
        for e in parsed.edges:
            tok = getattr(e, "local_id", None) or e.id
            if (
                (not tok)
                or _is_new_edge(tok)
                or _is_new_edge(getattr(e, "local_id", None))
            ):
                rid = ne2uuid.get(tok or "") or str(uuid.uuid4())
                if tok:
                    ne2uuid[tok] = rid
                e.id = rid
            else:
                e.id = de_alias(tok)

        def _res(xs: list[str] | None, kind: str) -> list[str] | None:
            if not xs:
                return None
            out: list[str] = []
            for x in xs:
                if kind == "node":
                    if _is_new_node(x):
                        rid = nn2uuid.get(x)
                        if not rid:
                            raise ValueError(f"Unknown temp node id: {x}")
                        out.append(rid)
                    elif _is_alias(x) or _is_uuid(x):
                        out.append(de_alias(x))
                    else:
                        key = (x or "").strip().lower()
                        rid = next(
                            (
                                n.id
                                for n in parsed.nodes
                                if (n.label or "").strip().lower() == key
                            ),
                            None,
                        )
                        if not rid:
                            raise ValueError(f"Unresolvable node endpoint token: {x}")
                        out.append(rid)
                else:
                    if _is_new_edge(x):
                        rid = ne2uuid.get(x)
                        if not rid:
                            raise ValueError(f"Unknown temp edge id: {x}")
                        out.append(rid)
                    elif _is_alias(x) or _is_uuid(x):
                        out.append(de_alias(x))
                    else:
                        raise ValueError(f"Unresolvable edge endpoint token: {x}")
            return out

        for e in parsed.edges:
            e.source_ids = _res(e.source_ids, kind="node") or []
            e.target_ids = _res(e.target_ids, kind="node") or []
            e.source_edge_ids = _res(getattr(e, "source_edge_ids", None), kind="edge")
            e.target_edge_ids = _res(getattr(e, "target_edge_ids", None), kind="edge")

    def build_deps(self, parsed):
        ts = TopologicalSorter()
        id2kind, id2obj = {}, {}

        for n in parsed.nodes or []:
            rid = n.id or str(uuid.uuid4())
            id2kind[rid], id2obj[rid] = "node", n
            ts.add(rid)

        new_ids = set(id2obj.keys())

        def deps_for_edge(e):
            deps = set()
            for x in (e.source_ids or []) + (e.target_ids or []):
                if x in new_ids and not self.exists_any(x):
                    deps.add(x)
            for x in getattr(e, "source_edge_ids", []) or []:
                if x in new_ids and not self.exists_any(x):
                    deps.add(x)
            for x in getattr(e, "target_edge_ids", []) or []:
                if x in new_ids and not self.exists_any(x):
                    deps.add(x)
            return deps

        for e in parsed.edges or []:
            rid = e.id or str(uuid.uuid4())
            id2kind[rid], id2obj[rid] = "edge", e
            ts.add(rid, *deps_for_edge(e))

        order = list(ts.static_order())
        return order, id2kind, id2obj

    def assert_endpoints_exist(self, edge: Edge | PureChromaEdge):
        """Enforce the structural ingest contract for edge writes.

        All referenced node and edge endpoints must already exist in the backend
        before the base edge row is accepted. Derived-index retries do not rescue
        edge-before-node ingest ordering, so callers that receive out-of-order
        events must retry or stage those edges outside this write path.
        """
        need_nodes = set((edge.source_ids or []) + (edge.target_ids or []))
        if need_nodes:
            got = set(self._e.backend.node_get(ids=list(need_nodes)).get("ids") or [])
            if got != need_nodes:
                raise ValueError(f"Missing node endpoints: {sorted(need_nodes - got)}")

        for attr in ("source_edge_ids", "target_edge_ids"):
            ids = getattr(edge, attr, None) or []
            if ids:
                got = set(self._e.backend.edge_get(ids=ids).get("ids") or [])
                if got != set(ids):
                    raise ValueError(
                        f"Missing edge endpoints in {attr}: {sorted(set(ids) - got)}"
                    )

    def exists_node(self, rid: str) -> bool:
        g = self._e.backend.node_get(ids=[rid])
        return (g.get("ids") or [None])[0] == rid

    def exists_edge(self, rid: str) -> bool:
        g = self._e.backend.edge_get(ids=[rid])
        return (g.get("ids") or [None])[0] == rid

    def exists_any(self, rid: str) -> bool:
        return self.exists_node(rid) or self.exists_edge(rid)

    def dealias_span(self, *args, **kwargs):
        return self._e.extract.dealias_span(*args, **kwargs)

    def select_doc_context(
        self, doc_id: str, max_nodes: int = 200, max_edges: int = 400
    ):
        nodes = self._e.backend.node_get(
            where={"doc_id": doc_id}, include=["documents"]
        )
        edges = self._e.backend.edge_get(
            where={"doc_id": doc_id}, include=["documents"]
        )

        node_items = []
        for i, (nid, ndoc) in enumerate(
            zip(nodes.get("ids", []) or [], nodes.get("documents", []) or [])
        ):
            if i >= max_nodes:
                break
            n = Node.model_validate_json(ndoc)
            node_items.append(
                {"id": nid, "label": n.label, "type": n.type, "summary": n.summary}
            )

        edge_items = []
        for i, (eid, edoc) in enumerate(
            zip(edges.get("ids", []) or [], edges.get("documents", []) or [])
        ):
            if i >= max_edges:
                break
            e = Edge.model_validate_json(edoc)
            edge_items.append(
                {
                    "id": eid,
                    "relation": e.relation,
                    "source_ids": e.source_ids or [],
                    "target_ids": e.target_ids or [],
                }
            )

        return node_items, edge_items

    def persist_graph(
        self,
        *,
        parsed: PureGraph,
        session_id: str,
        mode=None,
    ):
        self.preflight_validate(parsed, alias_key=session_id)
        node_ids, edge_ids = [], []
        self._alloc_real_ids(parsed)
        order, id2kind, id2obj = self.build_deps(parsed)
        for rid in order:
            kind, obj = id2kind[rid], id2obj[rid]
            if kind == "node":
                ln: Node = obj
                if mode == "skip-if-exists":
                    got = self._e.backend.node_get(ids=[ln.id])
                    if got.get("ids"):
                        node_ids.append(ln.id)
                        continue

                n = PureChromaNode(
                    id=ln.id,
                    label=ln.label,
                    type=ln.type,
                    summary=ln.summary,
                    domain_id=ln.domain_id,
                    canonical_entity_id=ln.canonical_entity_id,
                    properties=ln.properties,
                    doc_id=None,
                    embedding=None,
                    metadata={},
                )
                emb_text = f"{n.label}: {n.summary}"
                n.embedding = self._e._ef([emb_text])[0]
                self._e.write.add_pure_node(n)
                node_ids.append(n.id)
            elif kind == "edge":
                le: Edge = obj
                if mode == "skip-if-exists":
                    got = self._e.backend.edge_get(ids=[le.id])
                    if got.get("ids"):
                        edge_ids.append(le.id)
                        continue
                e = PureChromaEdge(
                    id=le.id,
                    label=le.label,
                    type=le.type,
                    summary=le.summary,
                    domain_id=le.domain_id,
                    canonical_entity_id=le.canonical_entity_id,
                    properties=le.properties,
                    relation=le.relation,
                    source_ids=le.source_ids,
                    target_ids=le.target_ids,
                    source_edge_ids=getattr(le, "source_edge_ids", None),
                    target_edge_ids=getattr(le, "target_edge_ids", None),
                    doc_id=None,
                    embedding=None,
                    metadata={},
                )
                e.embedding = self._e._ef([f"{le.label}: {le.summary}"])[0]
                self._e.write.add_pure_edge(e)
                edge_ids.append(e.id)

        return {
            "node_ids": node_ids,
            "edge_ids": edge_ids,
            "nodes_added": len(node_ids),
            "edges_added": len(edge_ids),
        }

    def persist_graph_extraction(
        self,
        *,
        document: Document,
        parsed: LLMGraphExtraction,
        mode: str = "append",
        assign_real_id_in_place=True,
    ) -> dict:
        doc_id = document.id
        if not assign_real_id_in_place:
            parsed = parsed.model_copy(deep=True)
        if mode == "replace":
            self._e.rollback.rollback_document(doc_id)

        self._e.write.add_document(document)
        self.preflight_validate(parsed, doc_id)

        node_ids, edge_ids = [], []
        self._alloc_real_ids(parsed)
        order, id2kind, id2obj = self.build_deps(parsed)
        span_validator = self._e.get_span_validator_of_doc_type(document=document)
        nl = "\n"
        for rid in order:
            kind, obj = id2kind[rid], id2obj[rid]
            if kind == "node":
                ln: Node = Node.model_validate(
                    self._promote_llm_entity_payload(
                        obj,
                        insertion_method="llm_graph_extraction",
                    ),
                    context={"insertion_method": "llm_graph_extraction"},
                )
                ln.mentions = self.dealias_span(ln.mentions, document.id)
                if mode == "skip-if-exists":
                    got = self._e.backend.node_get(ids=[ln.id])
                    if got.get("ids"):
                        node_ids.append(ln.id)
                        continue

                for g in ln.mentions:
                    for sp in g.spans:
                        result = span_validator.validate_span(
                            doc_id=doc_id,
                            span=sp,
                            engine=self._e,
                            doc=document
                            if document.id == sp.doc_id
                            else self._e.get_document(sp.doc_id),
                        )
                        if result["correctness"] is not True:
                            raise Exception(
                                f"Incorrect span occur in grounding {str(g)} span {str(sp)}"
                            )
                n = ln.model_copy(deep=True)
                emb_text = f"{n.label}: {n.summary} : {nl.join(i['context'] for i in self._e.extract_reference_contexts(ln)[:1])}"
                n.embedding = self._e._ef([emb_text])[0]
                self._e.write.add_node(n, doc_id=doc_id)
                node_ids.append(n.id)
            elif kind == "edge":
                le: Edge = Edge.model_validate(
                    self._promote_llm_entity_payload(
                        obj,
                        insertion_method="llm_graph_extraction",
                    ),
                    context={"insertion_method": "llm_graph_extraction"},
                )
                le.mentions = self.dealias_span(le.mentions, document.id)
                if mode == "skip-if-exists":
                    got = self._e.backend.edge_get(ids=[le.id])
                    if got.get("ids"):
                        edge_ids.append(le.id)
                        continue
                for g in le.mentions:
                    for sp in g.spans:
                        result = span_validator.validate_span(
                            doc_id=doc_id,
                            span=sp,
                            engine=self._e,
                            doc=document
                            if document.id == sp.doc_id
                            else self._e.get_document(sp.doc_id),
                        )
                        if result["correctness"] is not True:
                            raise Exception(
                                f"Incorrect span occur in grounding {str(g)} span {str(sp)}"
                            )
                e = le.model_copy(deep=True)
                emb_text = f"{le.label}: {le.summary} : {nl.join(i['context'] for i in self._e.extract_reference_contexts(le)[:1])}"
                e.embedding = self._e._ef([emb_text])[0]
                self._e.write.add_edge(e, doc_id=doc_id)
                edge_ids.append(e.id)

        return {
            "document_id": doc_id,
            "node_ids": node_ids,
            "edge_ids": edge_ids,
            "nodes_added": len(node_ids),
            "edges_added": len(edge_ids),
        }

    def persist_document_graph_extraction(
        self,
        *,
        doc_id,
        parsed: GraphExtractionWithIDs | LLMGraphExtraction,
        mode: str = "append",
    ) -> dict:
        self.preflight_validate(parsed, doc_id)

        node_ids, edge_ids = [], []
        self._alloc_real_ids(parsed)
        order, id2kind, id2obj = self.build_deps(parsed)

        document = self._e.read.get_document(doc_id)
        span_validator = self._e.get_span_validator_of_doc_type(doc_id=doc_id)
        nl = "\n"
        for rid in order:
            kind, obj = id2kind[rid], id2obj[rid]
            if kind == "node":
                ln: Node = obj
                ln.mentions = self.dealias_span(ln.mentions, doc_id)
                for g in ln.mentions:
                    for sp in g.spans:
                        result = span_validator.validate_span(
                            doc_id=doc_id,
                            span=sp,
                            engine=self._e,
                            doc=document
                            if document.id == sp.doc_id
                            else self._e.get_document(sp.doc_id),
                        )
                        if result["correctness"] is not True:
                            raise Exception(
                                f"Incorrect span occur in grounding {str(g)} span {str(sp)}"
                            )
                if mode == "skip-if-exists":
                    got = self._e.backend.node_get(ids=[ln.id])
                    if got.get("ids"):
                        node_ids.append(ln.id)
                        continue
                n = ln
                emb_text = f"{n.label}: {n.summary} : {nl.join(i['context'] for i in self._e.extract_reference_contexts(ln)[:1])}"
                n.embedding = self._e._ef([emb_text])[0]
                self._e.write.add_node(n, doc_id=doc_id)
                node_ids.append(n.id)
            elif kind == "edge":
                le: Edge = obj
                le.mentions = self.dealias_span(le.mentions, doc_id)
                for g in le.mentions:
                    for sp in g.spans:
                        result = span_validator.validate_span(
                            doc_id=doc_id,
                            span=sp,
                            engine=self._e,
                            doc=document
                            if document.id == sp.doc_id
                            else self._e.get_document(sp.doc_id),
                        )
                        if result["correctness"] is not True:
                            raise Exception(
                                f"Incorrect span occur in grounding {str(g)} span {str(sp)}"
                            )
                if mode == "skip-if-exists":
                    got = self._e.backend.edge_get(ids=[le.id])
                    if got.get("ids"):
                        edge_ids.append(le.id)
                        continue
                e = le
                emb_text = f"{le.label}: {le.summary} : {nl.join(i['context'] for i in self._e.extract_reference_contexts(le)[:1])}"
                e.embedding = self._e._ef([emb_text])[0]
                self._e.write.add_edge(e, doc_id=doc_id)
                edge_ids.append(e.id)

        return {
            "document_id": doc_id,
            "node_ids": node_ids,
            "edge_ids": edge_ids,
            "nodes_added": len(node_ids),
            "edges_added": len(edge_ids),
        }

    def ingest_with_toposort(self, parsed, *, doc_id: str):
        self._alloc_real_ids(parsed)
        order, id2kind, id2obj = self.build_deps(parsed)
        node_ids_added = set()
        edge_ids_added = set()
        nodes_added = edges_added = 0
        span_validator = self._e.get_span_validator_of_doc_type(doc_id=doc_id)
        for rid in order:
            kind, obj = id2kind[rid], id2obj[rid]

            if kind == "node":
                ln: Node = obj
                if self.exists_node(rid):
                    if ln.mentions:
                        prior = self._e.backend.node_get(
                            ids=[rid], include=["documents", "metadatas"]
                        )
                        prior_meta = (prior.get("metadatas") or [None])[0] or {}
                        prior_mentions = cast(str, prior_meta.get("mentions"))
                        merged_list, merged_json = merge_refs(
                            prior_mentions, ln.mentions
                        )
                        doc = prior["documents"]
                        if not doc:
                            raise Exception("missing documents")
                        n = Node.model_validate_json(doc[0])
                        mentions = Grounding(
                            spans=[Span.model_validate(r) for r in merged_list]
                        )
                        for sp in mentions.spans:
                            span_validator.validate_span(span=sp)
                        n.mentions = [mentions]
                        self._e.backend.node_update(
                            ids=[rid],
                            documents=[n.model_dump_json(field_mode="backend")],
                            metadatas=[
                                {
                                    **{
                                        k: v
                                        for k, v in prior_meta.items()
                                        if v is not None
                                    },
                                    "references": merged_json,
                                }
                            ],
                        )
                        self._e.write.index_node_docs(n)
                    continue

                ln.mentions = self.dealias_span(ln.mentions, doc_id)
                node = ln
                self._e.write.add_node(node, doc_id=doc_id)
                nodes_added += 1
                node_ids_added.add(node.id)
            else:
                le: Edge = obj
                if self.exists_edge(rid):
                    if le.mentions:
                        prior = self._e.backend.edge_get(
                            ids=[rid], include=["documents", "metadatas"]
                        )
                        prior_meta = (prior.get("metadatas") or [None])[0] or {}
                        prior_mentions = cast(str, prior_meta.get("mentions"))
                        merged_list, merged_json = merge_refs(
                            prior_mentions, le.mentions
                        )
                        doc = prior["documents"]
                        if not doc:
                            raise Exception("missing documents")
                        e = Edge.model_validate_json(doc[0])
                        mentions = Grounding(
                            spans=[Span.model_validate(r) for r in merged_list]
                        )
                        for sp in mentions.spans:
                            span_validator.validate_span(span=sp)
                        e.mentions = [mentions]
                        self._e.backend.edge_update(
                            ids=[rid],
                            documents=[e.model_dump_json(field_mode="backend")],
                            metadatas=[
                                {
                                    **{
                                        k: v
                                        for k, v in prior_meta.items()
                                        if v is not None
                                    },
                                    "references": merged_json,
                                }
                            ],
                        )
                        self._e.write.maybe_reindex_edge_refs(e)
                    continue

                edge = le
                self._e.write.add_edge(edge, doc_id=doc_id)
                edge_ids_added.add(edge.id)
                edges_added += 1
        return {
            "document_id": doc_id,
            "node_ids": nodes_added,
            "edge_ids": edges_added,
            "nodes_added": len(node_ids_added),
            "edges_added": len(edge_ids_added),
        }

    def replay_namespace(
        self,
        *,
        namespace: str = "default",
        from_seq: int = 1,
        to_seq: int | None = None,
        apply_indexes: bool = False,
        repair_backend: bool = False,
    ) -> int:
        """Replay entity events back into one namespace's backend state.

        Replay suppresses event re-append to avoid recursive logging and can disable
        index-job enqueueing so base entities are restored first. If apply_indexes is
        enabled, a reconcile pass runs after replay; repair_backend additionally
        deletes current backend rows before reapplying ADD/REPLACE events.
        """
        iter_events = getattr(self._e.meta_sqlite, "iter_entity_events", None)
        if iter_events is None:
            return 0

        from .. import models

        last_seq = 0
        prev_log = getattr(self._e, "_disable_event_log", False)
        prev_idx = getattr(self._e, "_phase1_enable_index_jobs", False)

        self._e._disable_event_log = True
        if not apply_indexes:
            self._e._phase1_enable_index_jobs = False

        try:
            for seq, ek, eid, op, payload_json in iter_events(
                namespace=namespace,
                from_seq=int(from_seq),
                to_seq=(int(to_seq) if to_seq is not None else None),
            ):
                payload = {}
                try:
                    payload = json.loads(payload_json)
                except Exception:
                    payload = {}

                if repair_backend and op in ("ADD", "REPLACE"):
                    try:
                        if ek == "node":
                            self._e.backend.node_delete(ids=[str(eid)])
                        elif ek == "edge":
                            self._e.backend.edge_delete(ids=[str(eid)])
                    except Exception:
                        pass

                if ek == "node":
                    if op in ("ADD", "REPLACE"):
                        try:
                            node = models.Node.model_validate(payload)
                        except Exception:
                            node = models.Node.model_validate_json(json.dumps(payload))
                        self._e.write.add_node(node)
                    elif op in ("TOMBSTONE", "DELETE"):
                        self._e.lifecycle.tombstone_node(str(eid))

                elif ek == "edge":
                    if op in ("ADD", "REPLACE"):
                        try:
                            edge = models.Edge.model_validate(payload)
                        except Exception:
                            edge = models.Edge.model_validate_json(json.dumps(payload))
                        self._e.write.add_edge(edge)
                    elif op in ("TOMBSTONE", "DELETE"):
                        self._e.lifecycle.tombstone_edge(str(eid))

                elif ek == "search_index":
                    if op == "search_index.upsert":
                        from ..search_index.models import IndexingItem

                        try:
                            item = IndexingItem.model_validate(payload)
                        except Exception:
                            item = IndexingItem.model_validate_json(json.dumps(payload))
                        if hasattr(self._e, "search_index"):
                            self._e.search_index.upsert_entries([item])

                last_seq = int(seq)
        finally:
            self._e._disable_event_log = prev_log
            self._e._phase1_enable_index_jobs = prev_idx

        if apply_indexes and getattr(self._e, "_phase1_enable_index_jobs", False):
            try:
                self._e.reconcile_indexes(max_jobs=200)
            except Exception:
                pass

        return last_seq

    def replay_repair_namespace(
        self,
        *,
        namespace: str = "default",
        from_seq: int = 1,
        to_seq: int | None = None,
        apply_indexes: bool = False,
    ) -> int:
        """Replay a namespace in overwrite-first repair mode.

        This is the destructive variant of replay_namespace: current backend rows for
        replayed ADD/REPLACE events are cleared first so the event log can repair
        backend drift instead of layering on top of possibly corrupt state.
        """
        return self.replay_namespace(
            namespace=namespace,
            from_seq=from_seq,
            to_seq=to_seq,
            apply_indexes=apply_indexes,
            repair_backend=True,
        )
