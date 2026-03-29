from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence, cast

from ...cdc.change_event import EntityRefModel
from ..models import Document, Domain, Edge, Node, PureChromaEdge, PureChromaNode
from ..utils.metadata import json_or_none, strip_none
from ...utils.embedding_vectors import normalize_embedding_vector
from ..utils.refs import (
    edge_doc_and_meta as edge_doc_and_meta_util,
    extract_doc_ids_from_refs,
    node_doc_and_meta as node_doc_and_meta_util,
)
from .base import NamespaceProxy
from ...typing_interfaces import WriteLike


def _refs_fingerprint(refs) -> str:
    payload = [
        {
            "doc_id": getattr(r, "doc_id", None),
            "method": getattr(getattr(r, "verification", None), "method", None),
            "is_verified": getattr(
                getattr(r, "verification", None), "is_verified", None
            ),
            "score": getattr(getattr(r, "verification", None), "score", None),
            "sp": getattr(r, "start_page", None),
            "ep": getattr(r, "end_page", None),
            "sc": getattr(r, "start_char", None),
            "ec": getattr(r, "end_char", None),
            "snip": (getattr(r, "excerpt", None) or "")[:64],
        }
        for r in (refs or [])
    ]
    blob = json.dumps(payload, sort_keys=False, separators=(",", ":")).encode("utf-8")
    return hashlib.blake2b(blob, digest_size=16).hexdigest()


class WriteSubsystem(NamespaceProxy, WriteLike):
    def __init__(self, engine) -> None:
        super().__init__(engine)

    # Canonical write API
    def add_node(self, *args, **kwargs):
        return self._add_node_impl(*args, **kwargs)

    def add_edge(self, *args, **kwargs):
        return self._add_edge_impl(*args, **kwargs)

    def _run_pre_add_node_hooks(self, node: Node) -> None:
        for hook in list(getattr(self._e, "pre_add_node_hooks", []) or []):
            hook(node)

    def _run_pre_add_edge_hooks(self, edge: Edge, *, pure: bool) -> bool:
        hook_name = "pre_add_pure_edge_hooks" if pure else "pre_add_edge_hooks"
        for hook in list(getattr(self._e, hook_name, []) or []):
            if bool(hook(edge)):
                return True
        return False

    def _allow_missing_doc_id(self, edge: Edge) -> bool:
        for hook in list(
            getattr(self._e, "allow_missing_doc_id_on_endpoint_rows_hooks", []) or []
        ):
            if bool(hook(edge)):
                return True
        return False

    def enrich_edge_meta(self, edge: Edge):
        node_endpoint_count = len(edge.source_ids or []) + len(edge.target_ids or [])
        edge_endpoint_count = len(getattr(edge, "source_edge_ids", []) or []) + len(
            getattr(edge, "target_edge_ids", []) or []
        )
        total_endpoint_count = node_endpoint_count + edge_endpoint_count
        base_metadata: dict[str, Any] = strip_none(
            {
                "doc_id": edge.doc_id,
                "relation": edge.relation,
                "source_ids": json_or_none(edge.source_ids),
                "target_ids": json_or_none(edge.target_ids),
                "source_edge_ids": json_or_none(getattr(edge, "source_edge_ids", None)),
                "target_edge_ids": json_or_none(getattr(edge, "target_edge_ids", None)),
                "type": edge.type,
                "summary": edge.summary,
                "domain_id": edge.domain_id,
                "canonical_entity_id": edge.canonical_entity_id,
                "properties": json_or_none(edge.properties),
                "references": json_or_none(
                    [
                        r.model_dump(field_mode="backend")
                        for r in (getattr(edge, "mentions", None) or [])
                    ]
                ),
                "node_endpoint_count": node_endpoint_count,
                "edge_endpoint_count": edge_endpoint_count,
                "total_endpoint_count": total_endpoint_count,
            }
        )
        md = getattr(edge, "metadata", {}) or {}
        base_metadata.update(
            strip_none(
                {
                    "char_distance_from_last_summary": md.get(
                        "char_distance_from_last_summary"
                    ),
                    "turn_distance_from_last_summary": md.get(
                        "turn_distance_from_last_summary"
                    ),
                    **(
                        {"causal_type": md.get("causal_type")}
                        if md.get("causal_type")
                        else {}
                    ),
                }
            )
        )
        if self._e.kg_graph_type == "workflow":
            from ...runtime.models import WorkflowEdge

            edge = cast(WorkflowEdge, edge)
            edge_metadata = edge.metadata
            base_metadata.update(
                strip_none(
                    {
                        "entity_type": edge_metadata.get("entity_type"),
                        "workflow_id": edge_metadata.get("workflow_id"),
                        "wf_priority": edge_metadata.get("wf_priority")
                        or edge_metadata.get("priority"),
                        "wf_is_default": edge_metadata.get("wf_is_default")
                        or edge_metadata.get("is_default"),
                        "wf_predicate": edge_metadata.get("wf_predicate")
                        or edge_metadata.get("predicate"),
                        "wf_multiplicity": edge_metadata.get("wf_multiplicity")
                        or edge_metadata.get("multiplicity"),
                    }
                )
            )
        return base_metadata

    def _add_node_impl(self, node: Node, doc_id: str | None = None):
        """Persist the base node, then converge derived indexes around it.

        The backend node row is the source of truth. This method best-effort appends
        an entity event and then either enqueues index jobs plus an immediate drain
        attempt or updates join indexes inline when phase1 jobs are disabled.
        Callers should not assume derived rows are atomically present with the base
        node write.
        """
        if doc_id is not None:
            node.doc_id = doc_id
        self._run_pre_add_node_hooks(node)

        doc, meta = self.node_doc_and_meta(node)
        if node.embedding is None:
            node.embedding = self._e.embed.iterative_defensive_emb(doc)
        node.embedding = normalize_embedding_vector(node.embedding, allow_none=False)
        meta["_class_name"] = type(node).__name__

        self._e.backend.node_add(
            ids=[node.safe_get_id()],
            documents=[doc],
            embeddings=[node.embedding]
            if node.embedding is not None
            else [self._e.embed.iterative_defensive_emb(str(doc))],
            metadatas=[meta],
        )

        try:
            payload = node.model_dump(field_mode="backend", exclude=["embedding"])
            self._e._append_event_for_entity(
                namespace=getattr(self._e, "namespace", "default"),
                entity_kind="node",
                entity_id=node.safe_get_id(),
                op="ADD",
                payload=payload if isinstance(payload, dict) else {},
            )
        except Exception:
            pass

        if self._e._phase1_enable_index_jobs:
            self._e.enqueue_index_jobs_for_node(node.safe_get_id(), op="UPSERT")
            self._e.reconcile_indexes(max_jobs=50)
        else:
            self.index_node_docs(node)
            self.maybe_reindex_node_refs(node)
        self._e._emit_change(
            op="node.upsert",
            entity=EntityRefModel(
                kind="node",
                id=node.safe_get_id(),
                kg_graph_type=self._e.kg_graph_type,
                url=self._e.persist_directory,
            ),
            payload=node.to_jsonable()
            if hasattr(node, "to_jsonable")
            else node.model_dump(exclude=["embedding"]),
        )

    def _add_edge_impl(self, edge: Edge, doc_id: str | None = None):
        """Persist an edge only after all referenced endpoints already exist.

        Edge ingest is structurally strict: missing node or edge endpoints are
        rejected before the base edge row is written. After persistence, the method
        follows the same fast-path-versus-index-job split as nodes for refs and
        edge_endpoints fanout, so derived projections may converge after the base
        edge is durable.
        """
        if doc_id is not None:
            edge.doc_id = doc_id
        s_nodes, s_edges, t_nodes, t_edges = self._e.adjudicate.split_endpoints(
            edge.source_ids, edge.target_ids
        )
        edge.source_ids = s_nodes
        edge.source_edge_ids = (getattr(edge, "source_edge_ids", []) or []) + s_edges
        edge.target_ids = t_nodes
        edge.target_edge_ids = (getattr(edge, "target_edge_ids", []) or []) + t_edges
        self._e.persist.assert_endpoints_exist(edge)
        if self._run_pre_add_edge_hooks(edge, pure=False):
            return

        doc = edge.model_dump_json(field_mode="backend", exclude=["embedding"])
        if edge.embedding is None:
            edge.embedding = self._e.embed.iterative_defensive_emb(str(doc))
        edge.embedding = normalize_embedding_vector(edge.embedding, allow_none=False)

        doc = edge.model_dump_json(field_mode="backend", exclude=["embedding"])
        base_metadata = [self.enrich_edge_meta(edge)]
        self._e.backend.edge_add(
            ids=[edge.safe_get_id()],
            documents=[str(doc)],
            embeddings=[edge.embedding]
            if edge.embedding is not None
            else [self._e.embed.iterative_defensive_emb(str(doc))],
            metadatas=base_metadata,
        )

        try:
            payload = edge.model_dump(field_mode="backend", exclude=["embedding"])
            self._e._append_event_for_entity(
                namespace=getattr(self._e, "namespace", "default"),
                entity_kind="edge",
                entity_id=edge.safe_get_id(),
                op="ADD",
                payload=payload if isinstance(payload, dict) else {},
            )
        except Exception:
            pass

        if self._e._phase1_enable_index_jobs:
            self._e.enqueue_index_jobs_for_edge(edge.safe_get_id(), op="UPSERT")
            self._e.reconcile_indexes(max_jobs=50)
        else:
            self.maybe_reindex_edge_refs(edge)
            rows = self.fanout_endpoints_rows(edge, doc_id)
            if rows:
                ep_ids = [r["id"] for r in rows]
                ep_docs = [json.dumps(r) for r in rows]
                ep_metas: list[dict] = rows
                self._e.backend.edge_endpoints_add(
                    ids=ep_ids,
                    documents=ep_docs,
                    metadatas=ep_metas,
                    embeddings=[
                        self._e.embed.iterative_defensive_emb(str(d)) for d in ep_docs
                    ],
                )

        self._e._emit_change(
            op="edge.upsert",
            entity=EntityRefModel(
                kind="edge",
                id=edge.safe_get_id(),
                kg_graph_type=self._e.kg_graph_type,
                url=self._e.persist_directory,
            ),
            payload=edge.to_jsonable()
            if hasattr(edge, "to_jsonable")
            else edge.model_dump(exclude=["embedding"]),
        )

    def add_pure_node(self, node: PureChromaNode):
        doc, meta = node_doc_and_meta_util(node)
        if meta.get("doc_id"):
            meta.pop("doc_id")
        self._e.backend.node_add(
            ids=[node.id],
            documents=[doc],
            embeddings=[
                normalize_embedding_vector(node.embedding, allow_none=False)
                if node.embedding is not None
                else normalize_embedding_vector(
                    self._e.embed.iterative_defensive_emb(str(doc)), allow_none=False
                )
            ],
            metadatas=[meta],
        )

    def add_pure_edge(self, edge: PureChromaEdge):
        """Low-level edge add without endpoint fanout or duplicate checks."""
        s_nodes, s_edges, t_nodes, t_edges = self._e.adjudicate.split_endpoints(
            edge.source_ids,
            edge.target_ids,
        )
        edge.source_ids = s_nodes
        edge.source_edge_ids = (getattr(edge, "source_edge_ids", []) or []) + s_edges
        edge.target_ids = t_nodes
        edge.target_edge_ids = (getattr(edge, "target_edge_ids", []) or []) + t_edges
        self._e.persist.assert_endpoints_exist(edge)
        if self._run_pre_add_edge_hooks(edge, pure=True):
            return

        doc = edge.model_dump_json(field_mode="backend", exclude=["embedding"])
        base_metadata = [self.enrich_edge_meta(edge)]
        self._e.backend.edge_add(
            ids=[edge.id],
            documents=[str(doc)],
            embeddings=[
                normalize_embedding_vector(edge.embedding, allow_none=False)
                if edge.embedding is not None
                else normalize_embedding_vector(
                    self._e.embed.iterative_defensive_emb(str(doc)), allow_none=False
                )
            ],
            metadatas=base_metadata,
        )

    def add_document(self, document: Document):
        if document.embeddings is None:
            document.embeddings = self._e.embed.iterative_defensive_emb(
                str(document.content)
            )
        document.embeddings = normalize_embedding_vector(
            document.embeddings, allow_none=False
        )
        self._e.backend.document_add(
            ids=[document.id],
            documents=[str(document.content)],
            embeddings=[cast(Sequence[float], document.embeddings)]
            if document.embeddings is not None
            else [
                normalize_embedding_vector(
                    self._e.embed.iterative_defensive_emb(str(document.content)),
                    allow_none=False,
                )
            ],
            metadatas=[
                strip_none(
                    {
                        "doc_id": document.id,
                        "type": document.type,
                        "metadata": json_or_none(document.metadata),
                        "domain_id": document.domain_id,
                        "processed": document.processed,
                    }
                )
            ],
        )
        self._e._emit_change(
            op="doc.upsert",
            entity=EntityRefModel(
                kind="doc_node",
                id=document.id,
                kg_graph_type=self._e.kg_graph_type,
                url=self._e.persist_directory,
            ),
            payload=document.to_jsonable()
            if hasattr(document, "to_jsonable")
            else document.model_dump(exclude=["embeddings"]),
        )

    def add_domain(self, domain: Domain):
        self._e.backend.domain_add(
            ids=[domain.id],
            documents=[domain.model_dump_json()],
            metadatas=[
                self._e.chroma_sanitize_metadata(
                    {
                        "name": domain.name,
                        "description": domain.description,
                    }
                )
            ],
            embeddings=[
                normalize_embedding_vector(
                    self._e.embed.iterative_defensive_emb(str(domain.model_dump_json())),
                    allow_none=False,
                )
            ],
        )

    # Index/metadata helpers
    def index_node_docs(self, node: Node) -> list[str]:
        doc_ids = extract_doc_ids_from_refs(node.mentions)

        self._e.backend.node_docs_delete(where={"node_id": node.id})
        if doc_ids:
            ids: list[str] = []
            docs: list[str] = []
            metas: list[dict] = []
            for did in doc_ids:
                rid = f"{node.id}::{did}"
                row = {"id": rid, "node_id": node.id, "doc_id": did, "mention_count": 1}
                ids.append(rid)
                docs.append(json.dumps(row))
                metas.append(row)
            self._e.backend.node_docs_add(
                ids=ids,
                documents=docs,
                metadatas=metas,
                embeddings=[self._e.embed.iterative_defensive_emb(d) for d in docs],
            )

        current = self._e.backend.node_get(ids=[node.id], include=["metadatas"])
        cur_meta = (current.get("metadatas") or [None])[0] or {}
        new_doc_ids_json = json.dumps(doc_ids)
        if cur_meta.get("doc_ids") != new_doc_ids_json:
            self._e.backend.node_update(
                ids=[node.id],
                metadatas=[{"doc_ids": new_doc_ids_json}],
            )

        return doc_ids

    def index_node_refs(self, *args, **kwargs):
        node = args[0] if args else kwargs["node"]
        self.delete_node_ref_rows(node.id)

        ids, docs, metas = [], [], []
        for i, mention in enumerate(node.mentions or []):
            for j, span in enumerate(mention.spans):
                rid = f"{node.id}::mention::{i}::span::{j}"
                did = getattr(span, "doc_id", None) or node.doc_id
                ver = getattr(span, "verification", None)
                row = strip_none(
                    {
                        "id": rid,
                        "node_id": node.id,
                        "doc_id": did,
                        "insertion_method": getattr(span, "insertion_method", None),
                        "verification_method": getattr(ver, "method", None),
                        "is_verified": getattr(ver, "is_verified", None),
                        "verificication_score": getattr(ver, "score", None),
                        "page_number": getattr(span, "start_page", None),
                        "excerpt": getattr(span, "excerpt", None),
                        "start_char": getattr(span, "start_char", None),
                        "end_char": getattr(span, "end_char", None),
                    }
                )
                ids.append(rid)
                docs.append(json.dumps(row))
                metas.append(row)

        if ids:
            self._e.backend.node_refs_add(
                ids=ids,
                documents=docs,
                metadatas=metas,
                embeddings=[self._e._iterative_defensive_emb(str(d)) for d in docs],
            )
        return ids

    def index_edge_refs(self, *args, **kwargs):
        edge = args[0] if args else kwargs["edge"]
        self.delete_edge_ref_rows(edge.id)

        ids, docs, metas = [], [], []
        for i, ref in enumerate(edge.mentions or []):
            rid = f"{edge.id}::ref::{i}"
            did = getattr(ref, "doc_id", None) or edge.doc_id
            ver = getattr(ref, "verification", None)
            row = strip_none(
                {
                    "id": rid,
                    "edge_id": edge.id,
                    "doc_id": did,
                    "insertion_method": getattr(ref, "insertion_method", None),
                    "verification_method": getattr(ver, "method", None),
                    "is_verified": getattr(ver, "is_verified", None),
                    "verificication_score": getattr(ver, "score", None),
                    "start_page": getattr(ref, "start_page", None),
                    "end_page": getattr(ref, "end_page", None),
                    "start_char": getattr(ref, "start_char", None),
                    "end_char": getattr(ref, "end_char", None),
                }
            )
            ids.append(rid)
            docs.append(json.dumps(row))
            metas.append(row)

        if ids:
            self._e.backend.edge_refs_add(
                ids=ids,
                documents=docs,
                metadatas=metas,
                embeddings=[self._e._iterative_defensive_emb(str(d)) for d in docs],
            )
        return ids

    def fanout_endpoints_rows(self, edge: Edge, doc_id: str | None):
        """Build derived edge_endpoints rows from the current edge payload.

        Rows inherit doc_id from the explicit argument when available; otherwise the
        helper reads doc_id from the referenced node or edge metadata. Missing
        doc_ids are allowed only when _allow_missing_doc_id permits them. These rows
        are rebuildable projections and are safe to regenerate from the base edge.
        """

        def _maybe_doc_for_edge(eid: str) -> str | None:
            if doc_id is not None:
                return doc_id
            meta = self._e.backend.edge_get(ids=[eid], include=["metadatas"])
            metadata = meta.get("metadatas")
            if metadata and metadata[0]:
                if isinstance(metadata[0].get("doc_id"), str):
                    return str(metadata[0].get("doc_id"))
                if not self._allow_missing_doc_id(edge):
                    raise Exception("doc_id is not string")
            return None

        def _per_node_doc(nid: str) -> str | None:
            if doc_id is not None:
                return doc_id
            meta = self._e.backend.node_get(ids=[nid], include=["metadatas"])
            metadata = meta.get("metadatas")
            if metadata and metadata[0]:
                if isinstance(metadata[0].get("doc_id"), str):
                    return str(metadata[0].get("doc_id"))
                if not self._allow_missing_doc_id(edge):
                    raise Exception("doc_id is not string")
            return None

        rows: list[dict] = []
        for role, node_ids in (
            ("src", edge.source_ids or []),
            ("tgt", edge.target_ids or []),
        ):
            for nid in node_ids:
                r = {
                    "id": f"{edge.id}::{role}::node::{nid}",
                    "edge_id": edge.id,
                    "endpoint_id": nid,
                    "endpoint_type": "node",
                    "role": role,
                    "causal_type": (edge.metadata or {}).get("causal_type"),
                    "relation": edge.relation,
                }
                did = _per_node_doc(nid)
                if did is not None:
                    r["doc_id"] = did
                rows.append({k: v for k, v in r.items() if v is not None})

        for role, eids in (
            ("src", getattr(edge, "source_edge_ids", []) or []),
            ("tgt", getattr(edge, "target_edge_ids", []) or []),
        ):
            for mid in eids:
                r = {
                    "id": f"{edge.id}::{role}::edge::{mid}",
                    "edge_id": edge.id,
                    "endpoint_id": mid,
                    "endpoint_type": "edge",
                    "role": role,
                    "causal_type": (edge.metadata or {}).get("causal_type"),
                    "relation": edge.relation,
                }
                did = _maybe_doc_for_edge(mid)
                if did is not None:
                    r["doc_id"] = did
                rows.append({k: v for k, v in r.items() if v is not None})

        return [{k: v for k, v in r.items() if v is not None} for r in rows]

    def node_doc_and_meta(self, *args, **kwargs):
        return node_doc_and_meta_util(*args, **kwargs)

    def edge_doc_and_meta(self, *args, **kwargs):
        return edge_doc_and_meta_util(*args, **kwargs)

    def strip_none(self, *args, **kwargs):
        return strip_none(*args, **kwargs)

    def json_or_none(self, *args, **kwargs):
        return json_or_none(*args, **kwargs)

    def delete_edge_ref_rows(self, *args, **kwargs):
        edge_id = args[0] if args else kwargs["edge_id"]
        got = self._e.backend.edge_refs_get(where={"edge_id": edge_id}, include=[])
        ids = got.get("ids") or []
        if ids:
            self._e.backend.edge_refs_delete(ids=ids)
        return None

    def delete_node_ref_rows(self, *args, **kwargs):
        node_id = args[0] if args else kwargs["node_id"]
        got = self._e.backend.node_refs_get(where={"node_id": node_id}, include=[])
        ids = got.get("ids") or []
        if ids:
            self._e.backend.node_refs_delete(ids=ids)
        return None

    def maybe_reindex_edge_refs(self, edge: Edge, *, force: bool = False) -> None:
        """Repair edge_refs when mention fingerprints or observed rows drift.

        The fingerprint is stored on the base edge metadata, but the method also
        checks row count and doc_id membership so partial deletes or stale rows are
        repaired even when the stored fingerprint still matches. force=True bypasses
        the drift short-circuit.
        """
        new_fp = _refs_fingerprint(edge.mentions or [])
        meta = self._e.backend.edge_get(ids=[edge.safe_get_id()], include=["metadatas"])
        old_fp = None
        metadatas = meta.get("metadatas")
        if metadatas and metadatas[0]:
            old_fp = metadatas[0].get("edge_refs_fp")

        got = self._e.backend.edge_refs_get(
            where={"edge_id": edge.id}, include=["documents"]
        )
        current_rows = got.get("documents") or []
        current_doc_ids = {json.loads(d).get("doc_id") for d in current_rows}
        expect_doc_ids = {getattr(r, "doc_id", None) for r in (edge.mentions or [])}
        count_ok = len(current_rows) == len(edge.mentions or [])
        docset_ok = current_doc_ids == expect_doc_ids

        if force or (new_fp != old_fp) or (not count_ok) or (not docset_ok):
            self._e.backend.edge_update(
                ids=[edge.safe_get_id()], metadatas=[{"edge_refs_fp": new_fp}]
            )
            self.index_edge_refs(edge)

    def maybe_reindex_node_refs(self, node: Node, *, force: bool = False) -> None:
        """Repair node_refs when mention fingerprints or observed rows drift.

        The base node metadata stores the last reference fingerprint, but count and
        doc_id-set checks are also used so derived rows are rebuilt after partial
        corruption or manual deletes. force=True skips the fingerprint equality
        optimization and always reindexes.
        """
        new_fp = _refs_fingerprint(node.mentions or [])
        meta = self._e.backend.node_get(ids=[node.id], include=["metadatas"])
        old_fp = None
        metadatas = meta.get("metadatas")
        if metadatas and metadatas[0]:
            old_fp = metadatas[0].get("node_refs_fp")

        got = self._e.backend.node_refs_get(
            where={"node_id": node.id}, include=["documents"]
        )
        current_rows = got.get("documents") or []
        current_doc_ids = {json.loads(d).get("doc_id") for d in current_rows}
        expect_doc_ids = {getattr(r, "doc_id", None) for r in (node.mentions or [])}
        count_ok = len(current_rows) == len(node.mentions or [])
        docset_ok = current_doc_ids == expect_doc_ids

        if force or (new_fp != old_fp) or (not count_ok) or (not docset_ok):
            self._e.backend.node_update(
                ids=[node.id], metadatas=[{"node_refs_fp": new_fp}]
            )
            self.index_node_refs(node)

    def prune_node_refs_for_doc(self, node_id: str, doc_id: str) -> bool:
        """Remove references to doc_id from node; delete node_docs link; refresh denormalized meta."""
        got = self._e.backend.node_get(
            ids=[node_id], include=["documents", "metadatas"]
        )
        docs = got.get("documents")
        if not (docs and docs[0]):
            return False
        node = Node.model_validate_json(docs[0])
        before = len(node.mentions or [])
        for groundings in node.mentions:
            filtered_spans = [
                span for span in groundings.spans if span.doc_id != doc_id
            ]
            groundings.spans = filtered_spans

        changed = len(node.mentions or []) != before
        if changed:
            self._e.backend.node_update(
                ids=[node_id], documents=[node.model_dump_json(field_mode="backend")]
            )
            self._e.backend.node_docs_delete(
                where={"$and": [{"node_id": node_id}, {"doc_id": doc_id}]}
            )
            self.index_node_docs(node)
        return changed

    def rebuild_edge_refs_for_doc(self, doc_id: str) -> int:
        eps = self._e.backend.edge_endpoints_get(
            where={"doc_id": doc_id}, include=["documents"]
        )
        edge_ids = list(
            {json.loads(d)["edge_id"] for d in (eps.get("documents") or [])}
        )
        if not edge_ids:
            return 0
        got = self._e.backend.edge_get(ids=edge_ids, include=["documents"])
        cnt = 0
        for js in got.get("documents") or []:
            e = Edge.model_validate_json(js)
            self.index_edge_refs(e)
            cnt += 1
        return cnt

    def rebuild_all_edge_refs(self) -> int:
        got = self._e.backend.edge_get()
        total = 0
        for eid in got.get("ids") or []:
            edges = self._e.backend.edge_get(ids=[eid], include=["documents"])
            if edge_docs := edges.get("documents"):
                e = Edge.model_validate_json(edge_docs[0])
                self.index_edge_refs(e)
                total += 1
        return total

    def rebuild_node_refs_for_doc(self, doc_id: str) -> int:
        node_ids = []
        if hasattr(self._e, "node_docs_collection"):
            rows = self._e.backend.node_docs_get(
                where={"doc_id": doc_id}, include=["documents"]
            )
            node_ids = list(
                {json.loads(d)["node_id"] for d in (rows.get("documents") or [])}
            )
        else:
            got = self._e.backend.node_get(
                where={"doc_id": doc_id}, include=["documents"]
            )
            node_ids = list(got.get("ids") or [])

        if not node_ids:
            return 0

        got = self._e.backend.node_get(ids=node_ids, include=["documents"])
        cnt = 0
        for js in got.get("documents") or []:
            n = Node.model_validate_json(js)
            self.index_node_refs(n)
            cnt += 1
        return cnt

    def rebuild_all_node_refs(self) -> int:
        got = self._e.backend.node_get()
        total = 0
        for nid in got.get("ids") or []:
            doc = self._e.backend.node_get(ids=[nid], include=["documents"])
            if nod_docs := doc.get("documents"):
                n = Node.model_validate_json(nod_docs[0])
                self.index_node_refs(n)
                total += 1
        return total

    def delete_edges_by_ids(self, edge_ids: list[str]):
        if not edge_ids:
            return
        self._e.backend.edge_delete(ids=edge_ids)
        self._e.backend.edge_endpoints_delete(
            where=cast(dict[str, object], {"edge_id": {"$in": edge_ids}})
        )
