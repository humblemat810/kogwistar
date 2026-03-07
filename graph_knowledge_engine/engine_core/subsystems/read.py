from __future__ import annotations

import json
from typing import Any, Literal, Sequence, Type, TypeVar, cast

import numpy as np

from ..models import Edge, Node
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
            from ...conversation.models import ConversationNode

            node_type = ConversationNode if self._e.kg_graph_type == "conversation" else Node

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
            from ...conversation.models import ConversationEdge

            edge_type = ConversationEdge if self._e.kg_graph_type == "conversation" else Edge

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

        from .. import models as core_models
        from ...conversation import models as conversation_models
        from ...runtime import models as runtime_models

        res: list[TNode] = []
        for d, emb, metadata in zip(docs, embs, metadatas):
            if isinstance(emb, np.ndarray):
                emb = emb.tolist()
            json_d = json.loads(d)
            override_node_type = None

            class_name = metadata.get("_class_name")
            if class_name:
                node_cls = (
                    getattr(core_models, class_name, None)
                    or getattr(conversation_models, class_name, None)
                    or getattr(runtime_models, class_name, None)
                )
                if node_cls:
                    override_node_type = node_cls

            if not override_node_type:
                entity_type = metadata.get("entity_type")
                if entity_type == "workflow_checkpoint" and self._e.kg_graph_type == "workflow":
                    from ...conversation.models import WorkflowCheckpointNode

                    override_node_type = WorkflowCheckpointNode

            json_d.update({"embedding": emb, "metadata": metadata})
            res.append((override_node_type or node_type).model_validate(json_d))
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

        from .. import models as core_models
        from ...conversation import models as conversation_models
        from ...runtime import models as runtime_models

        res = []
        for d, emb, metadata in zip(docs, embs, metadatas):
            if isinstance(emb, np.ndarray):
                emb = emb.tolist()
            json_d = json.loads(d)
            json_d.update({"embedding": emb, "metadata": metadata})
            override_edge_type = None
            class_name = metadata.get("_class_name")
            if class_name:
                edge_cls = (
                    getattr(core_models, class_name, None)
                    or getattr(conversation_models, class_name, None)
                    or getattr(runtime_models, class_name, None)
                )
                if edge_cls:
                    override_edge_type = edge_cls
            res.append((override_edge_type or edge_type).model_validate(json_d))
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
