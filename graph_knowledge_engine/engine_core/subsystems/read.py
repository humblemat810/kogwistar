from __future__ import annotations

import ast
import json
from typing import Any, Dict, List, Literal, Optional, Sequence, Type, TypeVar, Union, cast

import numpy as np

from ...entity_registry import (
    default_edge_type_for_graph_kind,
    default_node_type_for_graph_kind,
    pick_edge_type,
    pick_node_type,
)
from ..models import Document, Edge, Node
from .base import NamespaceProxy

TNode = TypeVar("TNode", bound=Node)


class ReadSubsystem(NamespaceProxy):
    def __init__(self, engine) -> None:
        super().__init__(engine)

    # Canonical read API
    def get_nodes(
        self,
        ids: Sequence[str] | None = None,
        node_type: Type[Node] | None = None,
        include: list[str] | None = None,
        where=None,
        limit: int | None = 200,
        resolve_mode: Literal["active_only", "redirect", "include_tombstones"] = "active_only",
    ) -> list[Node]:
        if include is None:
            include = ["documents", "embeddings", "metadatas"]
        if not node_type:
            node_type = default_node_type_for_graph_kind(self._e.kg_graph_type)

        got = self._e.backend.node_get(
            ids=ids,
            include=include,
            where=where,
            limit=limit,
        )
        nodes = self.nodes_from_single_or_id_query_result(got, node_type=node_type)
        nodes = self._e._resolve_redirect_chain(
            initial_items=nodes,
            resolve_mode=resolve_mode,
            fetch_by_ids=lambda redirect_ids: self.get_nodes(
                redirect_ids,
                node_type=node_type,
                resolve_mode=resolve_mode,
            ),
        )
        return self._e._filter_items_by_resolve_mode(nodes, resolve_mode)

    def get_edges(
        self,
        ids: Sequence[str] | None = None,
        edge_type: Type[Edge] | None = None,
        where=None,
        limit: int | None = 400,
        include: list[str] | None = None,
        resolve_mode: Literal["active_only", "redirect", "include_tombstones"] = "active_only",
    ) -> list[Edge]:
        if include is None:
            include = ["documents", "embeddings", "metadatas"]
        if not edge_type:
            edge_type = default_edge_type_for_graph_kind(self._e.kg_graph_type)

        got = self._e.backend.edge_get(
            ids=ids,
            include=include,
            where=where,
            limit=limit,
        )
        edges = self.edges_from_single_or_id_query_result(got, edge_type=edge_type, include=include)
        edges = self._e._resolve_redirect_chain(
            initial_items=edges,
            resolve_mode=resolve_mode,
            fetch_by_ids=lambda redirect_ids: self.get_edges(
                redirect_ids,
                edge_type=edge_type,
                resolve_mode=resolve_mode,
            ),
        )
        return self._e._filter_items_by_resolve_mode(edges, resolve_mode)

    def get_document(self, doc_id: str) -> Document:
        doc_get_result = self._e.backend.document_get(ids=[doc_id])
        if len(doc_get_result["ids"]) == 0:
            raise ValueError(f"no document found for doc id = {doc_id}")
        metadatas = doc_get_result["metadatas"]
        docs = doc_get_result["documents"]

        if docs is None or metadatas is None:
            raise ValueError("Invalid documnet metadata")
        metadata: dict = cast(dict, metadatas[0])

        doc = Document(
            id=doc_get_result["ids"][0],
            content=docs[0],
            metadata=metadata,
            domain_id=(None if metadata.get("domain_id") is None else str(metadata.get("domain_id"))),
            type=metadata["type"],
            processed=metadata["processed"],
            embeddings=None,
            source_map=None,
        )
        return doc

    def query_nodes(
        self,
        *args,
        query=None,
        query_embeddings=None,
        include=["documents", "embeddings", "metadatas"],
        node_type: Type[Node] = Node,
        **kwargs,
    ):
        if query_embeddings is not None:
            if query is not None:
                raise Exception("either query or query embedding but not both specified.")
        else:
            if query is not None:
                query_embeddings = self._e._iterative_defensive_emb(query)
            else:
                raise ValueError("either query or query embeddings must be specified")

        got = self._e.backend.node_query(
            query_embeddings=query_embeddings,
            *args,
            include=include,
            **kwargs,
        )
        return self.nodes_from_query_result(got, node_type=node_type)

    def query_edges(
        self,
        *args,
        query=None,
        query_embeddings=None,
        include=["documents", "embeddings", "metadatas"],
        edge_type: Type[Edge] = Edge,
        **kwargs,
    ):
        if query_embeddings is None:
            if query is not None:
                query_embeddings = self._e._iterative_defensive_emb(query)
            else:
                raise ValueError("either query or query embeddings must be specified")

        got = self._e.backend.edge_query(
            query_embeddings=query_embeddings,
            *args,
            include=include,
            **kwargs,
        )
        return self.edges_from_query_result(got, edge_type=edge_type)

    def nodes_from_single_or_id_query_result(
        self,
        got,
        node_type: Type[TNode] = Node,
    ) -> list[TNode]:
        docs: list[str] = cast(list[str], got.get("documents"))
        if docs is None:
            raise Exception("Missing docs")

        embs = got.get("embeddings")
        if embs is None:
            raise Exception("Missing Embeddings")
        embs = cast(list[list[float]], embs)

        metadatas = cast(list[dict[str, Any]], got.get("metadatas"))
        if metadatas is None:
            raise Exception("Missing Metadatas")

        res: list[TNode] = []
        for d, emb, metadata in zip(docs, embs, metadatas):
            if isinstance(emb, np.ndarray):
                emb = emb.tolist()
            json_d = json.loads(d)
            json_d.update({"embedding": emb, "metadata": metadata})
            selected_type = pick_node_type(
                graph_kind=self._e.kg_graph_type,
                metadata=metadata,
                fallback=node_type,
            )
            res.append(selected_type.model_validate(json_d))
        return res

    def edges_from_single_or_id_query_result(self, got, edge_type: Type[Edge] = Edge, include=None):
        if include is None:
            include = ["documents", "metadatas", "embeddings"]
        docs: list[str] = cast(list[str], got.get("documents"))
        if docs is None and "documents" in include:
            raise Exception("Missing docs")

        embs = got.get("embeddings")
        if embs is None:
            raise Exception("Missing Embeddings")
        embs = cast(list[list[float]], embs)

        metadatas = cast(list[dict[str, Any]], got.get("metadatas"))
        if metadatas is None:
            raise Exception("Missing Metadatas")

        res = []
        for d, emb, metadata in zip(docs, embs, metadatas):
            if isinstance(emb, np.ndarray):
                emb = emb.tolist()
            json_d = json.loads(d)
            json_d.update({"embedding": emb, "metadata": metadata})
            selected_type = pick_edge_type(metadata=metadata, fallback=edge_type)
            res.append(selected_type.model_validate(json_d))
        return res

    def nodes_from_query_result(self, gots, node_type: Type[Node] = Node):
        res = []
        for i_q in range(len(gots["ids"])):
            n_doc = len(gots["ids"][i_q])
            for _ids, docs, embs, metadatas in zip(
                gots.get("ids"),
                gots.get("documents") if gots.get("documents") is not None else [[]] * n_doc,
                gots.get("embeddings") if gots.get("embeddings") is not None else [[]] * n_doc,
                gots.get("metadatas") if gots.get("metadatas") is not None else [[]] * n_doc,
            ):
                docs = cast(list[str], docs)
                got = {"documents": docs, "embeddings": embs, "metadatas": metadatas}
                res.append(self.nodes_from_single_or_id_query_result(got, node_type=node_type))
        return res

    def edges_from_query_result(self, gots, edge_type: Type[Edge] = Edge):
        res = []
        for i_q in range(len(gots["ids"])):
            n_doc = len(gots["ids"][i_q])
            for ids, docs, embs, metadatas in zip(
                gots.get("ids"),
                gots.get("documents") if gots.get("documents") is not None else [[]] * n_doc,
                gots.get("embeddings") if gots.get("embeddings") is not None else [[]] * n_doc,
                gots.get("metadatas") if gots.get("metadatas") is not None else [[]] * n_doc,
            ):
                docs = cast(list[str], docs)
                got = {"ids": ids, "documents": docs, "embeddings": embs, "metadatas": metadatas}
                res.append(self.edges_from_single_or_id_query_result(got, edge_type=edge_type))
        return res

    def where_update_from_resolve_mode(
        self,
        resolve_mode: Literal["active_only", "redirect", "include_tombstones"],
    ):
        if resolve_mode == "active_only":
            return {"lifecycle_status": "active"}
        return {}

    def _infer_doc_id_from_ref(self, ref) -> Optional[str]:
        did = getattr(ref, "doc_id", None)
        if did:
            return did
        url = getattr(ref, "document_page_url", None) or ""
        try:
            tail = url.strip("/").split("/")[-1]
            return tail or None
        except Exception:
            return None

    def extract_reference_contexts(
        self,
        node_or_id: Union[Node | Edge, str],
        *,
        window_chars: int = 120,
        max_contexts: Optional[int] = None,
        prefer_label_fallback: bool = True,
    ) -> List[Dict[str, Any]]:
        from ..models import GraphEntityRefBase

        if isinstance(node_or_id, GraphEntityRefBase):
            obj = node_or_id
        else:
            got = self._e.backend.node_get(ids=[node_or_id], include=["documents"])
            doc_list = got.get("documents") or []
            if doc_list:
                obj = Node.model_validate_json(doc_list[0])
            else:
                got = self._e.backend.edge_get(ids=[node_or_id], include=["documents"])
                edoc_list = got.get("documents") or []
                if not edoc_list:
                    raise ValueError(f"Unknown node/edge id: {node_or_id}")
                obj = Edge.model_validate_json(edoc_list[0])

        label = getattr(obj, "label", None)
        out: List[Dict[str, Any]] = []
        doc_cache = {}

        def _coerce_to_referencable_text(text_or_ast_str):
            try:
                return "\n".join((i["text"] for i in ast.literal_eval(text_or_ast_str)["OCR_text_clusters"]))
            except Exception:
                return text_or_ast_str

        mentions = getattr(obj, "mentions", None) or []
        for mention in mentions:
            for span in mention.spans:
                doc_id = self._infer_doc_id_from_ref(span)

                pages = doc_cache.get(doc_id)
                full_doc = None
                if not pages:
                    full_doc = self._e.extract.fetch_document_text(doc_id) if doc_id else None
                    pages = self._e.extract.coerce_pages(full_doc)
                    doc_cache[doc_id] = pages
                excerpt = getattr(span, "excerpt", None)
                ctx_text = excerpt or (label or "")
                span_start = None
                span_end = None

                if full_doc:
                    page_relevant = {p[0]: p[1] for p in pages if (p[0] >= span.page_number and p[0] <= span.page_number)}
                    if span.page_number and span.page_number:
                        if span.page_number == span.page_number:
                            ctx_text = ""
                            if span.start_char and span.end_char:
                                try:
                                    _coerce_to_referencable_text(page_relevant[span.page_number])
                                except Exception:
                                    raise
                                ctx_text = _coerce_to_referencable_text(page_relevant[span.page_number])[
                                    max(span.start_char - window_chars, 0) : span.end_char + window_chars
                                ]

                    if ctx_text is None:
                        idx = full_doc.find(excerpt) if excerpt else -1
                        if idx < 0 and label and prefer_label_fallback:
                            idx = full_doc.find(label)

                        if idx >= 0:
                            length = len(excerpt) if excerpt else (len(label) if label else 0)
                            span_start = idx
                            span_end = idx + length
                            left = max(0, span_start - window_chars)
                            right = min(len(full_doc), span_end + window_chars)
                            ctx_text = full_doc[left:right]
                out.append(
                    {
                        "doc_id": doc_id,
                        "collection_page_url": getattr(span, "collection_page_url", None),
                        "document_page_url": getattr(span, "document_page_url", None),
                        "start_page": getattr(span, "start_page", None),
                        "end_page": getattr(span, "end_page", None),
                        "start_char": getattr(span, "start_char", None),
                        "end_char": getattr(span, "end_char", None),
                        "insertion_method": getattr(span, "insertion_method", None),
                        "verification": (span.verification.model_dump() if getattr(span, "verification", None) else None),
                        "context": ctx_text,
                        "mention": mention,
                        "loc_found": (span_start is not None),
                        "loc_span": [span_start, span_end] if span_start is not None else None,
                        "ref": span.model_dump(field_mode="backend"),
                    }
                )

                if max_contexts and len(out) >= max_contexts:
                    break

        return out

    # Doc-index helpers
    def node_ids_by_doc(self, doc_id: str, insertion_method: str | None = None) -> list[str]:
        if insertion_method:
            return self._e.ids_with_insertion_method(
                kind="node",
                insertion_method=insertion_method,
                doc_id=doc_id,
            )
        if hasattr(self._e, "node_docs_collection"):
            rows = self._e.backend.node_docs_get(where={"doc_id": doc_id}, include=["metadatas"])
            result = set()
            for m in (rows.get("metadatas") or []):
                if m and m.get("node_id"):
                    result.add(m.get("node_id"))
            return sorted(result)
        got = self._e.backend.node_get(where={"doc_id": doc_id})
        return got.get("ids") or []

    def edge_ids_by_doc(self, doc_id: str, insertion_method: str | None = None) -> list[str]:
        if insertion_method:
            return self._e.ids_with_insertion_method(
                kind="edge",
                insertion_method=insertion_method,
                doc_id=doc_id,
            )
        eps = self._e.backend.edge_endpoints_get(where={"doc_id": doc_id}, include=["metadatas"])
        result = set()
        for m in (eps.get("metadatas") or []):
            if m and m.get("edge_id"):
                result.add(m.get("edge_id"))
        return sorted(result)

    def edges_by_doc(self, doc_id: str, where: dict | None = None) -> list[str]:
        where = (
            {"doc_id": doc_id}
            if not where
            else {"$and": [{"doc_id": doc_id}] + [{k: v} for k, v in where.items()]}
        )
        rows = self._e.backend.edge_refs_get(where=where, include=["documents"])
        return list({json.loads(d)["edge_id"] for d in (rows.get("documents") or [])})

    def list_edges_with_ref_filter(self, doc_id: str, where: dict | None = None) -> list[Edge]:
        ids = self.edges_by_doc(doc_id, where)
        if not ids:
            return []
        got = self._e.backend.edge_get(ids=ids, include=["documents"])
        return [Edge.model_validate_json(js) for js in (got.get("documents") or [])]

    def nodes_by_doc(self, doc_id: str, *, where: dict | None = None) -> list[str]:
        where = (
            {"doc_id": doc_id}
            if not where
            else {"$and": [{"doc_id": doc_id}] + [{k: v} for k, v in where.items()]}
        )
        rows = self._e.backend.node_refs_get(where=where, include=["documents"])
        return list({json.loads(d)["node_id"] for d in (rows.get("documents") or [])})

    def list_nodes_with_ref_filter(self, doc_id: str, *, where: dict | None = None) -> list[Node]:
        ids = self.nodes_by_doc(doc_id, where=where)
        if not ids:
            return []
        got = self._e.backend.node_get(ids=ids, include=["documents"])
        return [Node.model_validate_json(js) for js in (got.get("documents") or [])]

    # Legacy names retained during migration
    def nodes_by_doc_index(self, doc_id: str, insertion_method: str | None = None) -> list[str]:
        return self.node_ids_by_doc(doc_id, insertion_method=insertion_method)

    def edge_ids_by_doc_index(self, doc_id: str, insertion_method: str | None = None) -> list[str]:
        return self.edge_ids_by_doc(doc_id, insertion_method=insertion_method)

    def load_node_map(self, *args, **kwargs):
        ids = kwargs.pop("ids", None)
        if ids is None and args:
            ids = args[0]
            args = args[1:]
        node_type = kwargs.pop("node_type", None)
        include = kwargs.pop("include", None)
        if args or kwargs:
            raise TypeError("load_node_map accepts only ids, node_type, and include")
        if ids is None:
            return {}
        nodes = self.get_nodes(ids=list(ids), node_type=node_type, include=include or ["documents"])
        return {n.safe_get_id(): n for n in nodes}

    def load_edge_map(self, *args, **kwargs):
        ids = kwargs.pop("ids", None)
        if ids is None and args:
            ids = args[0]
            args = args[1:]
        edge_type = kwargs.pop("edge_type", None)
        include = kwargs.pop("include", None)
        if args or kwargs:
            raise TypeError("load_edge_map accepts only ids, edge_type, and include")
        if ids is None:
            return {}
        edges = self.get_edges(ids=list(ids), edge_type=edge_type, include=include or ["documents"])
        return {e.safe_get_id(): e for e in edges}
