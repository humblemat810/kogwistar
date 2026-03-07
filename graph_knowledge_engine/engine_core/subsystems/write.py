from __future__ import annotations

import json
from typing import Sequence, cast

from ...cdc.change_event import EntityRefModel
from ..models import Document, Domain, Edge, Node
from ..utils.metadata import json_or_none, strip_none
from ..utils.refs import extract_doc_ids_from_refs
from .base import NamespaceProxy


class WriteSubsystem(NamespaceProxy):
    def __init__(self, engine) -> None:
        super().__init__(engine)

    # Canonical write API
    def add_node(self, *args, **kwargs):
        return self._e._impl_add_node(*args, **kwargs)

    def add_edge(self, *args, **kwargs):
        return self._e._impl_add_edge(*args, **kwargs)

    def _add_node_impl(self, node: Node, doc_id: str | None = None):
        if doc_id is not None:
            node.doc_id = doc_id
        if self._e.kg_graph_type == "conversation":
            from ...conversation.models import ConversationNode

            node_conv: ConversationNode = cast(ConversationNode, node)
            try:
                conv_id = node_conv.conversation_id
            except Exception:
                conv_id = node_conv.metadata["conversation_id"]

            if conv_id is None:
                raise Exception("conv id required")
            self._e._get_last_seq_node(conv_id)
            seq = self._e.meta_sqlite.next_user_seq(conv_id)
            node.metadata["seq"] = seq

        doc, meta = self.node_doc_and_meta(node)
        if node.embedding is None:
            node.embedding = self._e.embed.iterative_defensive_emb(doc)
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
            self._e._maybe_reindex_node_refs(node)
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
        if doc_id is not None:
            edge.doc_id = doc_id
        s_nodes, s_edges, t_nodes, t_edges = self._e._split_endpoints(edge.source_ids, edge.target_ids)
        edge.source_ids = s_nodes
        edge.source_edge_ids = (getattr(edge, "source_edge_ids", []) or []) + s_edges
        edge.target_ids = t_nodes
        edge.target_edge_ids = (getattr(edge, "target_edge_ids", []) or []) + t_edges
        self._e._assert_endpoints_exist(edge)

        from ...conversation.models import ConversationEdge

        if isinstance(edge, ConversationEdge):
            if self._e._is_duplicate_next_turn_noop(edge):
                return
            self._e._validate_conversation_edge_add(edge)

        doc = edge.model_dump_json(field_mode="backend", exclude=["embedding"])
        if edge.embedding is None:
            edge.embedding = self._e.embed.iterative_defensive_emb(str(doc))

        doc = edge.model_dump_json(field_mode="backend", exclude=["embedding"])
        base_metadata = [self._e.enrich_edge_meta(edge)]
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
            self._e._maybe_reindex_edge_refs(edge)
            rows = self.fanout_endpoints_rows(edge, doc_id)
            if rows:
                ep_ids = [r["id"] for r in rows]
                ep_docs = [json.dumps(r) for r in rows]
                ep_metas: list[dict] = rows
                self._e.backend.edge_endpoints_add(
                    ids=ep_ids,
                    documents=ep_docs,
                    metadatas=ep_metas,
                    embeddings=[self._e.embed.iterative_defensive_emb(str(d)) for d in ep_docs],
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

    def add_document(self, document: Document):
        if document.embeddings is None:
            document.embeddings = self._e.embed.iterative_defensive_emb(str(document.content))
        self._e.backend.document_add(
            ids=[document.id],
            documents=[str(document.content)],
            embeddings=[cast(Sequence[float], document.embeddings)]
            if document.embeddings is not None
            else [self._e.embed.iterative_defensive_emb(str(document.content))],
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
            embeddings=[self._e.embed.iterative_defensive_emb(str(domain.model_dump_json()))],
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
        return self._e._index_node_refs(*args, **kwargs)

    def index_edge_refs(self, *args, **kwargs):
        return self._e._index_edge_refs(*args, **kwargs)

    def fanout_endpoints_rows(self, edge: Edge, doc_id: str | None):
        def _maybe_doc_for_edge(eid: str) -> str | None:
            if doc_id is not None:
                return doc_id
            meta = self._e.backend.edge_get(ids=[eid], include=["metadatas"])
            metadata = meta.get("metadatas")
            if metadata and metadata[0]:
                if isinstance(metadata[0].get("doc_id"), str):
                    return str(metadata[0].get("doc_id"))
                if not self._e._entity_is_conversation(edge):
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
                if not self._e._entity_is_conversation(edge):
                    raise Exception("doc_id is not string")
            return None

        rows: list[dict] = []
        for role, node_ids in (("src", edge.source_ids or []), ("tgt", edge.target_ids or [])):
            for nid in node_ids:
                r = {
                    "id": f"{edge.id}::{role}::node::{nid}",
                    "edge_id": edge.id,
                    "endpoint_id": nid,
                    "endpoint_type": "node",
                    "role": role,
                    "causal_type": edge.metadata and edge.metadata.get("causal_type"),
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
                    "causal_type": edge.metadata and edge.metadata.get("causal_type"),
                    "relation": edge.relation,
                }
                did = _maybe_doc_for_edge(mid)
                if did is not None:
                    r["doc_id"] = did
                rows.append({k: v for k, v in r.items() if v is not None})

        return [{k: v for k, v in r.items() if v is not None} for r in rows]

    def node_doc_and_meta(self, *args, **kwargs):
        return self._e._node_doc_and_meta(*args, **kwargs)

    def edge_doc_and_meta(self, *args, **kwargs):
        return self._e._edge_doc_and_meta(*args, **kwargs)

    def strip_none(self, *args, **kwargs):
        return self._e._strip_none(*args, **kwargs)

    def json_or_none(self, *args, **kwargs):
        return self._e._json_or_none(*args, **kwargs)

    def delete_edge_ref_rows(self, *args, **kwargs):
        return self._e._delete_edge_ref_rows(*args, **kwargs)

    def delete_node_ref_rows(self, *args, **kwargs):
        return self._e._delete_node_ref_rows(*args, **kwargs)
