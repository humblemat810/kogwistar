from __future__ import annotations

import json
from typing import Any, Literal

from ..async_compat import run_awaitable_blocking
from ..models import Edge, Node
from ..utils.metadata import json_or_none, strip_none
from ..utils.refs import extract_doc_ids_from_refs
from .base import NamespaceProxy


class RollbackSubsystem(NamespaceProxy):
    def __init__(self, engine) -> None:
        super().__init__(engine)

    def _filter_mentions_for_document(self, mentions, document_id: str):
        kept = []
        changed = False
        for grounding in mentions or []:
            copied = grounding.model_copy(deep=True)
            spans = []
            for span in copied.spans:
                span_doc_id = self._e._infer_doc_id_from_ref(span)
                if span_doc_id == document_id:
                    changed = True
                    continue
                spans.append(span)
            if spans:
                copied.spans = spans
                kept.append(copied)
            elif copied.spans:
                changed = True
        return kept, changed

    def _replacement_doc_id(self, mentions) -> str | None:
        doc_ids = extract_doc_ids_from_refs(mentions or [])
        if len(doc_ids) == 1:
            return doc_ids[0]
        return None

    def _clean_metadata_for_replacement(
        self, metadata: dict[str, Any] | None
    ) -> dict[str, Any]:
        cleaned = dict(metadata or {})
        for key in (
            "lifecycle_status",
            "redirect_to_id",
            "deleted_at",
            "delete_reason",
            "deleted_by",
        ):
            cleaned.pop(key, None)
        return cleaned

    def _delete_node_derived_rows(self, node_id: str) -> None:
        run_awaitable_blocking(self._e.backend.node_docs_delete(where={"node_id": node_id}))
        run_awaitable_blocking(self._e.backend.node_refs_delete(where={"node_id": node_id}))

    def _delete_edge_derived_rows(self, edge_id: str) -> None:
        run_awaitable_blocking(self._e.backend.edge_endpoints_delete(where={"edge_id": edge_id}))
        run_awaitable_blocking(self._e.backend.edge_refs_delete(where={"edge_id": edge_id}))

    def _replacement_node(self, node: Node, mentions) -> Node:
        payload = node.model_dump(field_mode="backend", exclude={"id", "embedding"})
        payload["mentions"] = list(mentions)
        payload["doc_id"] = self._replacement_doc_id(mentions)
        payload["metadata"] = self._clean_metadata_for_replacement(node.metadata)
        replacement = Node.model_validate(payload)
        replacement.embedding = None
        return replacement

    def _replacement_edge(
        self,
        edge: Edge,
        *,
        mentions,
        source_ids: list[str],
        target_ids: list[str],
        source_edge_ids: list[str],
        target_edge_ids: list[str],
    ) -> Edge:
        payload = edge.model_dump(field_mode="backend", exclude={"id", "embedding"})
        payload["mentions"] = list(mentions)
        payload["source_ids"] = list(source_ids)
        payload["target_ids"] = list(target_ids)
        payload["source_edge_ids"] = list(source_edge_ids)
        payload["target_edge_ids"] = list(target_edge_ids)
        payload["doc_id"] = self._replacement_doc_id(mentions)
        payload["metadata"] = self._clean_metadata_for_replacement(edge.metadata)
        replacement = Edge.model_validate(payload)
        replacement.embedding = None
        return replacement

    def _load_nodes(self, node_ids: list[str]) -> list[Node]:
        if not node_ids:
            return []
        got = run_awaitable_blocking(self._e.backend.node_get(ids=node_ids, include=["documents"]))
        out: list[Node] = []
        for doc in got.get("documents") or []:
            if doc:
                out.append(Node.model_validate_json(doc))
        return out

    def _load_edge(self, edge_id: str) -> Edge | None:
        got = run_awaitable_blocking(self._e.backend.edge_get(ids=[edge_id], include=["documents"]))
        docs = got.get("documents") or []
        if not docs or not docs[0]:
            return None
        return Edge.model_validate_json(docs[0])

    def _edge_ids_for_endpoint(
        self, endpoint_id: str, endpoint_type: Literal["node", "edge"]
    ) -> set[str]:
        rows = run_awaitable_blocking(self._e.backend.edge_endpoints_get(
            where={
                "$and": [{"endpoint_id": endpoint_id}, {"endpoint_type": endpoint_type}]
            },
            include=["metadatas"],
        ))
        edge_ids: set[str] = set()
        for metadata in rows.get("metadatas") or []:
            if metadata and metadata.get("edge_id"):
                edge_ids.add(str(metadata["edge_id"]))
        return edge_ids

    def rollback_document(self, document_id: str):
        """Remove one document's contribution while preserving surviving evidence.

        Nodes and edges that still have mentions or valid endpoints are rewritten as
        replacement records and the originals are redirected. Entities with no
        surviving support are tombstoned and their derived rows are deleted. Edge
        repairs cascade through downstream edge endpoints so rollback converges the
        graph instead of leaving broken references behind.
        """
        node_rows = run_awaitable_blocking(self._e.backend.node_docs_get(
            where={"doc_id": document_id}, include=["metadatas"]
        ))
        affected_node_ids = sorted(
            {
                str(metadata["node_id"])
                for metadata in (node_rows.get("metadatas") or [])
                if metadata and metadata.get("node_id")
            }
        )

        tombstoned_node_ids: list[str] = []
        deleted_node_ids: list[str] = []
        updated_node_ids: list[str] = []
        node_redirects: dict[str, str] = {}
        removed_node_ids: set[str] = set()

        for node in self._load_nodes(affected_node_ids):
            surviving_mentions, changed = self._filter_mentions_for_document(
                node.mentions, document_id
            )
            if not changed:
                continue
            if surviving_mentions:
                replacement = self._replacement_node(node, surviving_mentions)
                self._e.write.add_node(replacement)
                self._e.redirect_node(
                    node.safe_get_id(),
                    replacement.safe_get_id(),
                    reason=f"rollback_document:{document_id}",
                )
                node_redirects[node.safe_get_id()] = replacement.safe_get_id()
                updated_node_ids.append(replacement.safe_get_id())
            else:
                self._e.tombstone_node(
                    node.safe_get_id(), reason=f"rollback_document:{document_id}"
                )
                removed_node_ids.add(node.safe_get_id())
                deleted_node_ids.append(node.safe_get_id())
            self._delete_node_derived_rows(node.safe_get_id())
            tombstoned_node_ids.append(node.safe_get_id())

        edge_ids_from_refs = set(self._e.read.edges_by_doc(document_id))
        edge_ids_touching_nodes: set[str] = set()
        for node_id in affected_node_ids:
            edge_ids_touching_nodes.update(self._edge_ids_for_endpoint(node_id, "node"))

        edge_queue = list(sorted(edge_ids_from_refs | edge_ids_touching_nodes))
        processed_edge_revisions: dict[str, int] = {}
        edge_revision = 0

        deleted_edge_ids: list[str] = []
        updated_edge_ids: list[str] = []
        tombstoned_edge_ids: list[str] = []
        edge_redirects: dict[str, str] = {}
        removed_edge_ids: set[str] = set()

        while edge_queue:
            edge_id = edge_queue.pop(0)
            if processed_edge_revisions.get(edge_id) == edge_revision:
                continue
            processed_edge_revisions[edge_id] = edge_revision

            edge = self._load_edge(edge_id)
            if edge is None:
                continue

            surviving_mentions, mentions_changed = self._filter_mentions_for_document(
                edge.mentions,
                document_id,
            )

            mapped_source_ids = [
                node_redirects.get(endpoint_id, endpoint_id)
                for endpoint_id in (edge.source_ids or [])
                if endpoint_id not in removed_node_ids
            ]
            mapped_target_ids = [
                node_redirects.get(endpoint_id, endpoint_id)
                for endpoint_id in (edge.target_ids or [])
                if endpoint_id not in removed_node_ids
            ]
            mapped_source_edge_ids = [
                edge_redirects.get(endpoint_id, endpoint_id)
                for endpoint_id in (getattr(edge, "source_edge_ids", []) or [])
                if endpoint_id not in removed_edge_ids
            ]
            mapped_target_edge_ids = [
                edge_redirects.get(endpoint_id, endpoint_id)
                for endpoint_id in (getattr(edge, "target_edge_ids", []) or [])
                if endpoint_id not in removed_edge_ids
            ]

            endpoints_changed = (
                mapped_source_ids != (edge.source_ids or [])
                or mapped_target_ids != (edge.target_ids or [])
                or mapped_source_edge_ids
                != (getattr(edge, "source_edge_ids", []) or [])
                or mapped_target_edge_ids
                != (getattr(edge, "target_edge_ids", []) or [])
            )

            if edge.relation == "same_as" and (
                mapped_source_ids != (edge.source_ids or [])
                or mapped_target_ids != (edge.target_ids or [])
            ):
                remain = list(dict.fromkeys(mapped_source_ids + mapped_target_ids))
                if len(remain) >= 2:
                    anchor = self._e.adjudicate.choose_anchor(remain)
                    mapped_source_ids = [anchor]
                    mapped_target_ids = [nid for nid in remain if nid != anchor]
                else:
                    mapped_source_ids = remain[:1]
                    mapped_target_ids = remain[1:]

            has_source_endpoints = bool(mapped_source_ids or mapped_source_edge_ids)
            has_target_endpoints = bool(mapped_target_ids or mapped_target_edge_ids)

            if (
                not surviving_mentions
                or not has_source_endpoints
                or not has_target_endpoints
            ):
                self._e.tombstone_edge(
                    edge.safe_get_id(), reason=f"rollback_document:{document_id}"
                )
                self._delete_edge_derived_rows(edge.safe_get_id())
                tombstoned_edge_ids.append(edge.safe_get_id())
                deleted_edge_ids.append(edge.safe_get_id())
                removed_edge_ids.add(edge.safe_get_id())
                downstream = self._edge_ids_for_endpoint(edge.safe_get_id(), "edge")
                if downstream:
                    edge_revision += 1
                    edge_queue.extend(sorted(downstream))
                continue

            if not mentions_changed and not endpoints_changed:
                continue

            replacement_edge = self._replacement_edge(
                edge,
                mentions=surviving_mentions,
                source_ids=mapped_source_ids,
                target_ids=mapped_target_ids,
                source_edge_ids=mapped_source_edge_ids,
                target_edge_ids=mapped_target_edge_ids,
            )
            self._e.write.add_edge(replacement_edge)
            self._e.redirect_edge(
                edge.safe_get_id(),
                replacement_edge.safe_get_id(),
                reason=f"rollback_document:{document_id}",
            )
            self._delete_edge_derived_rows(edge.safe_get_id())
            tombstoned_edge_ids.append(edge.safe_get_id())
            updated_edge_ids.append(replacement_edge.safe_get_id())
            edge_redirects[edge.safe_get_id()] = replacement_edge.safe_get_id()

            downstream = self._edge_ids_for_endpoint(edge.safe_get_id(), "edge")
            if downstream:
                edge_revision += 1
                edge_queue.extend(sorted(downstream))

        doc_ids = set(
            run_awaitable_blocking(self._e.backend.document_get(where={"doc_id": document_id}))["ids"]
        )
        run_awaitable_blocking(self._e.backend.document_delete(where={"doc_id": document_id}))
        doc_ids_after = set(
            run_awaitable_blocking(self._e.backend.document_get(where={"doc_id": document_id}))["ids"]
        )
        return {
            "rolled_back_doc_id": document_id,
            "rolled_back_doc_ids": list(doc_ids - doc_ids_after),
            "node_redirects": node_redirects,
            "edge_redirects": edge_redirects,
            "tombstoned_node_ids": tombstoned_node_ids,
            "updated_node_ids": updated_node_ids,
            "deleted_node_ids": deleted_node_ids,
            "tombstoned_edge_ids": tombstoned_edge_ids,
            "updated_edge_ids": updated_edge_ids,
            "deleted_edge_ids": deleted_edge_ids,
            "deleted_docs": len(doc_ids - doc_ids_after),
            "updated_nodes": len(updated_node_ids),
            "deleted_nodes": len(deleted_node_ids),
            "deleted_edges": len(deleted_edge_ids),
            "updated_edges": len(updated_edge_ids),
        }

    def rollback_document_extraction(
        self,
        doc_id: str,
        extraction_method: Literal["llm_graph_extraction", "document_ingestion"],
    ) -> dict:
        """Undo one extraction method's refs without deleting the whole document.

        This pass edits raw node and edge reference payloads plus join indexes in
        place rather than building redirect chains. Matching refs are removed for the
        requested extraction method, surviving entities are updated in place, and
        entities are deleted only when no references remain after cleanup.
        """
        summary = {
            "doc_id": doc_id,
            "method": extraction_method,
            "updated_nodes": 0,
            "updated_edges": 0,
            "deleted_nodes": 0,
            "deleted_edges": 0,
            "deleted_node_refs": 0,
            "deleted_edge_refs": 0,
            "deleted_node_doc_rows": 0,
            "deleted_edge_endpoints": 0,
        }

        def _load_many(kind: str, ids):
            if not ids:
                return {}
            get_fn = getattr(self._e.backend, f"{kind}_get")
            got = get_fn(ids=list(ids), include=["documents"])
            docs = got.get("documents") or []
            ids_out = got.get("ids") or []
            out = {}
            for i, mj in enumerate(docs):
                try:
                    d = json.loads(mj)
                except Exception:
                    try:
                        d = (
                            (Node if kind == "node" else Edge)
                            .model_validate_json(mj)
                            .model_dump(field_mode="backend")
                        )
                    except Exception:
                        d = None
                if d is not None and i < len(ids_out):
                    out[ids_out[i]] = d
            return out

        def _save_node(d: dict):
            nid = d["id"]
            prior = run_awaitable_blocking(self._e.backend.node_get(ids=[nid], include=["metadatas"]))
            meta = (prior.get("metadatas") or [None])[0] or {}
            run_awaitable_blocking(self._e.backend.node_update(
                ids=[nid],
                documents=[json.dumps(d, ensure_ascii=False)],
                metadatas=[dict(meta)],
            ))
            try:
                self._e.write.index_node_docs(Node.model_validate(d))
            except Exception:
                pass

        def _save_edge(d: dict):
            eid = d["id"]
            prior = run_awaitable_blocking(self._e.backend.edge_get(ids=[eid], include=["metadatas"]))
            meta = (prior.get("metadatas") or [None])[0] or {}
            run_awaitable_blocking(self._e.backend.edge_update(
                ids=[eid],
                documents=[json.dumps(d, ensure_ascii=False)],
                metadatas=[dict(meta)],
            ))

        node_ids = set()
        try:
            nd = run_awaitable_blocking(self._e.backend.node_docs_get(
                where={"doc_id": doc_id}, include=["metadatas"]
            ))
            for m in nd.get("metadatas") or []:
                if m and m.get("node_id"):
                    node_ids.add(m["node_id"])
            summary["deleted_node_doc_rows"] = len(nd.get("ids") or [])
        except Exception:
            try:
                q = run_awaitable_blocking(self._e.backend.node_get(where={"doc_id": doc_id}))
                for nid in q.get("ids") or []:
                    node_ids.add(nid)
            except Exception:
                pass

        edge_ids = set()
        try:
            ee = run_awaitable_blocking(self._e.backend.edge_endpoints_get(
                where={"doc_id": doc_id}, include=["metadatas"]
            ))
            for m in ee.get("metadatas") or []:
                if m and m.get("edge_id"):
                    edge_ids.add(m["edge_id"])
            summary["deleted_edge_endpoints"] = len(ee.get("ids") or [])
        except Exception:
            try:
                q = run_awaitable_blocking(self._e.backend.edge_get(where={"doc_id": doc_id}))
                for eid in q.get("ids") or []:
                    edge_ids.add(eid)
            except Exception:
                pass

        nodes_map = _load_many("node", node_ids)
        for nid, d in nodes_map.items():
            refs = d.get("references") or []
            keep = []
            removed = 0
            for r in refs:
                if (
                    r
                    and (r.get("doc_id") == doc_id)
                    and (r.get("insertion_method") == extraction_method)
                ):
                    removed += 1
                else:
                    keep.append(r)
            if removed:
                summary["deleted_node_refs"] += removed
                if keep:
                    d["references"] = keep
                    _save_node(d)
                    summary["updated_nodes"] += 1
                else:
                    try:
                        run_awaitable_blocking(self._e.backend.node_delete(ids=[nid]))
                    except Exception:
                        pass
                    try:
                        run_awaitable_blocking(self._e.backend.node_docs_delete(
                            where={"node_id": nid, "doc_id": doc_id}
                        ))
                    except Exception:
                        pass
                    summary["deleted_nodes"] += 1
            else:
                try:
                    run_awaitable_blocking(self._e.backend.node_docs_delete(
                        where={"node_id": nid, "doc_id": doc_id}
                    ))
                except Exception:
                    pass

        edges_map = _load_many("edge", edge_ids)
        for eid, d in edges_map.items():
            refs = d.get("references") or []
            keep = []
            removed = 0
            for r in refs:
                if (
                    r
                    and (r.get("doc_id") == doc_id)
                    and (r.get("insertion_method") == extraction_method)
                ):
                    removed += 1
                else:
                    keep.append(r)

            try:
                run_awaitable_blocking(self._e.backend.edge_endpoints_delete(
                    where={"edge_id": eid, "doc_id": doc_id}
                ))
            except Exception:
                pass

            if removed:
                summary["deleted_edge_refs"] += removed
                if keep:
                    d["references"] = keep
                    _save_edge(d)
                    summary["updated_edges"] += 1
                else:
                    try:
                        run_awaitable_blocking(self._e.backend.edge_delete(ids=[eid]))
                    except Exception:
                        pass
                    try:
                        run_awaitable_blocking(self._e.backend.edge_endpoints_delete(where={"edge_id": eid}))
                    except Exception:
                        pass
                    summary["deleted_edges"] += 1

        try:
            run_awaitable_blocking(self._e.backend.node_docs_delete(where={"doc_id": doc_id}))
        except Exception:
            pass
        try:
            run_awaitable_blocking(self._e.backend.edge_endpoints_delete(where={"doc_id": doc_id}))
        except Exception:
            pass

        return summary

    def prune_node_from_edges(self, node_id: str):
        eps = run_awaitable_blocking(self._e.backend.edge_endpoints_get(
            where={"$and": [{"endpoint_id": node_id}, {"endpoint_type": "node"}]},
            include=["documents"],
        ))
        if not eps["ids"]:
            return {"deleted_edges": set(), "updated_edges": set()}
        if eps_doc := eps["documents"]:
            pass
        else:
            raise Exception("Document loss")
        edge_ids = list({json.loads(doc)["edge_id"] for doc in eps_doc})
        edges = run_awaitable_blocking(self._e.backend.edge_get(
            ids=edge_ids, include=["documents", "metadatas"]
        ))

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
                    run_awaitable_blocking(self._e.backend.edge_delete(ids=[eid]))
                    run_awaitable_blocking(self._e.backend.edge_endpoints_delete(where={"edge_id": eid}))
                    removed_edge_ids.add(eid)
                else:
                    run_awaitable_blocking(self._e.backend.edge_update(
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
                    ))
                    self._e.write.index_edge_refs(new_edge)
                    run_awaitable_blocking(self._e.backend.edge_endpoints_delete(where={"edge_id": eid}))
                    ep_ids, ep_docs, ep_metas = [], [], []
                    for role, node_ids in (
                        ("src", new_edge.source_ids or []),
                        ("tgt", new_edge.target_ids or []),
                    ):
                        for nid in node_ids:
                            ep_id = f"{eid}::{role}::{nid}"
                            node_doc = run_awaitable_blocking(self._e.backend.node_get(
                                ids=[nid], include=["documents"]
                            ))
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
                        run_awaitable_blocking(self._e.backend.edge_endpoints_add(
                            ids=ep_ids,
                            documents=ep_docs,
                            metadatas=ep_metas,
                            embeddings=[
                                self._e._iterative_defensive_emb(d) for d in ep_docs
                            ],
                        ))
                    updated_edge_ids.add(eid)
                continue

            new_src = [x for x in (e.source_ids or []) if x != node_id]
            new_tgt = [x for x in (e.target_ids or []) if x != node_id]
            if not new_src or not new_tgt:
                run_awaitable_blocking(self._e.backend.edge_delete(ids=[eid]))
                run_awaitable_blocking(self._e.backend.edge_endpoints_delete(where={"edge_id": eid}))
                removed_edge_ids.add(eid)
            else:
                e.source_ids, e.target_ids = new_src, new_tgt
                run_awaitable_blocking(self._e.backend.edge_update(
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
                ))
                run_awaitable_blocking(self._e.backend.edge_endpoints_delete(
                    where={"$and": [{"edge_id": eid}, {"node_id": node_id}]}
                ))
                updated_edge_ids.add(eid)
                self._e.write.index_edge_refs(e)

        return {
            "deleted_edges": removed_edge_ids,
            "updated_edges": updated_edge_ids - removed_edge_ids,
        }

    def rollback_many_documents(self, document_ids: list[str]):
        totals = {
            "deleted_nodes": 0,
            "deleted_edges": 0,
            "updated_edges": 0,
            "deleted_docs": 0,
        }
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
