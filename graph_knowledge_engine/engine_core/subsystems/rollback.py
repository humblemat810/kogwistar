from __future__ import annotations

import json
from typing import Any, cast

from ..models import Edge, Node
from ..utils.metadata import json_or_none, strip_none
from .base import NamespaceProxy


class RollbackSubsystem(NamespaceProxy):
    def __init__(self, engine) -> None:
        super().__init__(engine)

    def rollback_document(self, document_id: str):
        node_ids = self._e.read.node_ids_by_doc(document_id)

        deleted_edges: set[str] = set()
        updated_edges: set[str] = set()
        for nid in node_ids:
            res = self.prune_node_from_edges(nid)
            deleted_edges.update(res["deleted_edges"])
            updated_edges.update(res["updated_edges"])

        eps = self._e.backend.edge_endpoints_get(where={"doc_id": document_id})
        eps_doc = eps.get("documents", [])
        if eps_doc is None:
            raise Exception(f"edge endpoint collection lost for document id {document_id}")
        edge_ids = list({json.loads(doc)["edge_id"] for doc in eps_doc})
        if edge_ids:
            self._e.backend.edge_delete(ids=edge_ids)
            self._e.backend.edge_endpoints_delete(where={"doc_id": document_id})
            self._e.backend.edge_refs_delete(where={"node_id": {"$in": edge_ids}})
            deleted_edges.update(edge_ids)
        updated_edges = updated_edges - deleted_edges

        deleted_node_ids = []
        for nid in node_ids:
            self.prune_node_refs_for_doc(nid, document_id)
            got = self._e.backend.node_get(ids=[nid], include=["documents"])
            if docs := got.get("documents"):
                if docs[0]:
                    if not json.loads(docs[0]).get("references"):
                        self._e.backend.node_delete(ids=[nid])
                        deleted_node_ids.append(nid)
                    self._e.backend.node_refs_delete(
                        where=cast(dict[str, Any], {"node_id": {"$in": node_ids}})
                    )

        doc_ids = set(self._e.backend.document_get(where={"doc_id": document_id})["ids"])
        self._e.backend.document_delete(where={"doc_id": document_id})
        doc_ids_after = set(self._e.backend.document_get(where={"doc_id": document_id})["ids"])
        return {
            "rollrolled_back_doc_id": doc_ids - doc_ids_after,
            "updated_edge_ids": list(updated_edges),
            "deleted_edge_ids": list(deleted_edges),
            "deleted_docs": len(doc_ids - doc_ids_after),
            "deleted_node_ids": deleted_node_ids,
            "deleted_nodes": len(node_ids),
            "deleted_edges": len(deleted_edges),
            "updated_edges": len(updated_edges),
        }

    def prune_node_from_edges(self, node_id: str):
        eps = self._e.backend.edge_endpoints_get(
            where={"$and": [{"endpoint_id": node_id}, {"endpoint_type": "node"}]},
            include=["documents"],
        )
        if not eps["ids"]:
            return {"deleted_edges": set(), "updated_edges": set()}
        if eps_doc := eps["documents"]:
            pass
        else:
            raise Exception("Document loss")
        edge_ids = list({json.loads(doc)["edge_id"] for doc in eps_doc})
        edges = self._e.backend.edge_get(ids=edge_ids, include=["documents", "metadatas"])

        removed_edge_ids: set[str] = set()
        updated_edge_ids: set[str] = set()
        for eid, edoc, meta in zip(
            edges.get("ids") or [],
            edges.get("documents") or [],
            edges.get("metadatas") or [],
        ):
            e = Edge.model_validate_json(edoc)
            relation = (meta or {}).get("relation") or e.relation

            if relation == "same_as":
                edge_deleted, new_edge = self._e.adjudicate.rebalance_same_as_edge(
                    e,
                    removed_node_id=node_id,
                )
                if edge_deleted or (new_edge is None):
                    self._e.backend.edge_delete(ids=[eid])
                    self._e.backend.edge_endpoints_delete(where={"edge_id": eid})
                    removed_edge_ids.add(eid)
                else:
                    self._e.backend.edge_update(
                        ids=[eid],
                        documents=[new_edge.model_dump_json(field_mode="backend")],
                        metadatas=[
                            strip_none(
                                {
                                    "doc_id": (meta or {}).get("doc_id"),
                                    "relation": new_edge.relation,
                                    "source_ids": json_or_none(new_edge.source_ids),
                                    "target_ids": json_or_none(new_edge.target_ids),
                                    "type": new_edge.type,
                                    "summary": new_edge.summary,
                                    "domain_id": new_edge.domain_id,
                                    "canonical_entity_id": new_edge.canonical_entity_id,
                                    "properties": json_or_none(new_edge.properties),
                                    "references": json_or_none(
                                        [
                                            ref.model_dump(field_mode="backend")
                                            for ref in (new_edge.mentions or [])
                                        ]
                                    ),
                                }
                            )
                        ],
                    )
                    self._e.write.index_edge_refs(new_edge)
                    self._e.backend.edge_endpoints_delete(where={"edge_id": eid})
                    ep_ids, ep_docs, ep_metas = [], [], []
                    for role, node_ids in (
                        ("src", new_edge.source_ids or []),
                        ("tgt", new_edge.target_ids or []),
                    ):
                        for nid in node_ids:
                            ep_id = f"{eid}::{role}::{nid}"
                            node_doc = self._e.backend.node_get(ids=[nid], include=["documents"])
                            if node_doc is None:
                                raise Exception(f"node_doc for {nid} is lost")
                            per_doc_id = None
                            if node_doc_doc := node_doc.get("documents"):
                                try:
                                    n = Node.model_validate_json(node_doc_doc[0])
                                    per_doc_id = getattr(n, "doc_id", None)
                                except Exception:
                                    per_doc_id = None
                            meta_ep = strip_none(
                                {
                                    "id": ep_id,
                                    "edge_id": eid,
                                    "node_id": nid,
                                    "role": role,
                                    "relation": new_edge.relation,
                                    "doc_id": per_doc_id,
                                }
                            )
                            ep_ids.append(ep_id)
                            ep_docs.append(json.dumps(meta_ep))
                            ep_metas.append(meta_ep)
                    if ep_ids:
                        self._e.backend.edge_endpoints_add(
                            ids=ep_ids,
                            documents=ep_docs,
                            metadatas=ep_metas,
                            embeddings=[self._e._iterative_defensive_emb(d) for d in ep_docs],
                        )
                    updated_edge_ids.add(eid)
                continue

            new_src = [x for x in (e.source_ids or []) if x != node_id]
            new_tgt = [x for x in (e.target_ids or []) if x != node_id]
            if not new_src or not new_tgt:
                self._e.backend.edge_delete(ids=[eid])
                self._e.backend.edge_endpoints_delete(where={"edge_id": eid})
                removed_edge_ids.add(eid)
            else:
                e.source_ids, e.target_ids = new_src, new_tgt
                self._e.backend.edge_update(
                    ids=[eid],
                    documents=[e.model_dump_json(field_mode="backend")],
                    metadatas=[
                        strip_none(
                            {
                                "doc_id": (meta or {}).get("doc_id"),
                                "relation": e.relation,
                                "source_ids": json_or_none(e.source_ids),
                                "target_ids": json_or_none(e.target_ids),
                                "type": e.type,
                                "summary": e.summary,
                                "domain_id": e.domain_id,
                                "canonical_entity_id": e.canonical_entity_id,
                                "properties": json_or_none(e.properties),
                                "references": json_or_none(
                                    [
                                        ref.model_dump(field_mode="backend")
                                        for ref in (e.mentions or [])
                                    ]
                                ),
                            }
                        )
                    ],
                )
                self._e.backend.edge_endpoints_delete(
                    where={"$and": [{"edge_id": eid}, {"node_id": node_id}]}
                )
                updated_edge_ids.add(eid)
                self._e.write.index_edge_refs(e)

        return {
            "deleted_edges": removed_edge_ids,
            "updated_edges": updated_edge_ids - removed_edge_ids,
        }

    def rollback_many_documents(self, document_ids: list[str]):
        totals = {"deleted_nodes": 0, "deleted_edges": 0, "updated_edges": 0, "deleted_docs": 0}
        for did in document_ids:
            res = self.rollback_document(did)
            totals["deleted_docs"] += 1
            totals["deleted_nodes"] += len(res["deleted_node_ids"])
            totals["deleted_edges"] += len(res["deleted_edge_ids"])
            totals["updated_edges"] += res["updated_edges"]
        return totals

    def delete_edges_by_ids(self, *args, **kwargs):
        return self._e.write.delete_edges_by_ids(*args, **kwargs)

    def prune_node_refs_for_doc(self, *args, **kwargs):
        return self._e.write.prune_node_refs_for_doc(*args, **kwargs)
