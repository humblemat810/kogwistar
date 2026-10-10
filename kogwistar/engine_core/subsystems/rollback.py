from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, cast

from ..async_compat import run_awaitable_blocking
from ...json_types import JsonObject
from ..models import Edge, Grounding, Node
from ..utils.metadata import json_or_none, strip_none
from ..utils.refs import extract_doc_ids_from_refs
from .base import NamespaceProxy

if TYPE_CHECKING:
    from ..engine import GraphKnowledgeEngine


def _backend_object(value: object) -> JsonObject:
    """Narrow an untyped backend response before reading persisted fields."""

    if not isinstance(value, Mapping):
        return {}
    return cast(JsonObject, dict(value))


def _backend_list(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []


def _backend_strings(value: object) -> list[str]:
    return [item for item in _backend_list(value) if isinstance(item, str)]


def _backend_objects(value: object) -> list[JsonObject]:
    return [_backend_object(item) for item in _backend_list(value)]


def _redirected_ids(
    values: object, redirects: Mapping[str, str], removed: set[str]
) -> list[str]:
    result: list[str] = []
    for value in _backend_list(values):
        if not isinstance(value, str) or value in removed:
            continue
        result.append(redirects.get(value, value))
    return result


class RollbackSubsystem(NamespaceProxy["GraphKnowledgeEngine"]):
    def __init__(self, engine: GraphKnowledgeEngine) -> None:
        super().__init__(engine)

    def _filter_mentions_for_document(
        self, mentions: list[Grounding] | None, document_id: str
    ) -> tuple[list[Grounding], bool]:
        kept: list[Grounding] = []
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

    def _replacement_doc_id(self, mentions: list[Grounding] | None) -> str | None:
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

    def _replacement_node(
        self, node: Node, mentions: list[Grounding] | None
    ) -> Node:
        payload = node.model_dump(field_mode="backend", exclude={"id", "embedding"})
        payload["mentions"] = list(mentions or [])
        payload["doc_id"] = self._replacement_doc_id(mentions)
        payload["metadata"] = self._clean_metadata_for_replacement(node.metadata)
        replacement = Node.model_validate(payload)
        replacement.embedding = None
        return replacement

    def _replacement_edge(
        self,
        edge: Edge,
        *,
        mentions: list[Grounding] | None,
        source_ids: list[str],
        target_ids: list[str],
        source_edge_ids: list[str],
        target_edge_ids: list[str],
    ) -> Edge:
        payload = edge.model_dump(field_mode="backend", exclude={"id", "embedding"})
        payload["mentions"] = list(mentions or [])
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
        got = _backend_object(
            run_awaitable_blocking(
                self._e.backend.node_get(ids=node_ids, include=["documents"])
            )
        )
        out: list[Node] = []
        for doc in _backend_list(got.get("documents")):
            if doc:
                if isinstance(doc, str):
                    out.append(Node.model_validate_json(doc))
        return out

    def _load_edge(self, edge_id: str) -> Edge | None:
        got = _backend_object(
            run_awaitable_blocking(
                self._e.backend.edge_get(ids=[edge_id], include=["documents"])
            )
        )
        docs = _backend_list(got.get("documents"))
        if not docs or not docs[0]:
            return None
        return Edge.model_validate_json(cast(str, docs[0]))

    def _edge_ids_for_endpoint(
        self, endpoint_id: str, endpoint_type: Literal["node", "edge"]
    ) -> set[str]:
        rows = _backend_object(
            run_awaitable_blocking(
                self._e.backend.edge_endpoints_get(
                    where={
                        "$and": [
                            {"endpoint_id": endpoint_id},
                            {"endpoint_type": endpoint_type},
                        ]
                    },
                    include=["metadatas"],
                )
            )
        )
        edge_ids: set[str] = set()
        for metadata in _backend_list(rows.get("metadatas")):
            metadata_object = _backend_object(metadata)
            if metadata_object.get("edge_id"):
                edge_ids.add(str(metadata_object["edge_id"]))
        return edge_ids

    def rollback_document(self, document_id: str) -> dict[str, object]:
        """Remove one document's contribution while preserving surviving evidence.

        Nodes and edges that still have mentions or valid endpoints are rewritten as
        replacement records and the originals are redirected. Entities with no
        surviving support are tombstoned and their derived rows are deleted. Edge
        repairs cascade through downstream edge endpoints so rollback converges the
        graph instead of leaving broken references behind.
        """
        node_rows = _backend_object(
            run_awaitable_blocking(
                self._e.backend.node_docs_get(
                    where={"doc_id": document_id}, include=["metadatas"]
                )
            )
        )
        affected_node_ids = sorted(
            {
                str(metadata["node_id"])
                for metadata in _backend_objects(node_rows.get("metadatas"))
                if metadata.get("node_id")
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
                if endpoint_id is not None and endpoint_id not in removed_node_ids
            ]
            mapped_target_ids = [
                node_redirects.get(endpoint_id, endpoint_id)
                for endpoint_id in (edge.target_ids or [])
                if endpoint_id is not None and endpoint_id not in removed_node_ids
            ]
            mapped_source_edge_ids = _redirected_ids(
                getattr(edge, "source_edge_ids", []), edge_redirects, removed_edge_ids
            )
            mapped_target_edge_ids = _redirected_ids(
                getattr(edge, "target_edge_ids", []), edge_redirects, removed_edge_ids
            )

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
            _backend_strings(
                _backend_object(
                    run_awaitable_blocking(
                        self._e.backend.document_get(where={"doc_id": document_id})
                    )
                ).get("ids")
            )
        )
        if not self._e.write.rust_postgres_delete_existing(
            entity_kind="document", entity_ids=sorted(doc_ids)
        ):
            run_awaitable_blocking(
                self._e.backend.document_delete(where={"doc_id": document_id})
            )
        doc_ids_after = set(
            _backend_strings(
                _backend_object(
                    run_awaitable_blocking(
                        self._e.backend.document_get(where={"doc_id": document_id})
                    )
                ).get("ids")
            )
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
    ) -> dict[str, object]:
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

        def _load_many(kind: str, ids: set[str]) -> dict[str, dict[str, Any]]:
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

        def _filter_reference_payload(
            d: dict[str, Any],
        ) -> tuple[str, list[Any], int]:
            key = "mentions" if "mentions" in d else "references"
            refs = d.get(key) or []
            kept: list = []
            removed = 0
            for ref in refs:
                if not isinstance(ref, dict):
                    kept.append(ref)
                    continue
                spans = ref.get("spans")
                if isinstance(spans, list):
                    kept_spans = []
                    for span in spans:
                        if (
                            isinstance(span, dict)
                            and span.get("doc_id") == doc_id
                            and span.get("insertion_method") == extraction_method
                        ):
                            removed += 1
                        else:
                            kept_spans.append(span)
                    if kept_spans:
                        kept_grounding = dict(ref)
                        kept_grounding["spans"] = kept_spans
                        kept.append(kept_grounding)
                    continue
                if (
                    ref.get("doc_id") == doc_id
                    and ref.get("insertion_method") == extraction_method
                ):
                    removed += 1
                else:
                    kept.append(ref)
            return key, kept, removed

        def _save_node(d: dict[str, Any]) -> None:
            nid = d["id"]
            prior = _backend_object(
                run_awaitable_blocking(
                    self._e.backend.node_get(ids=[nid], include=["metadatas"])
                )
            )
            meta = (_backend_objects(prior.get("metadatas")) or [{}])[0]
            refs = d.get("mentions", d.get("references", []))
            meta = {**meta, "mentions": json.dumps(refs, ensure_ascii=False)}
            document = json.dumps(d, ensure_ascii=False)
            native_replaced = self._e.write.rust_postgres_replace_existing(
                entity_kind="node",
                entity_id=nid,
                document=document,
                metadata_patch=dict(meta),
                payload=d,
            )
            if not native_replaced:
                run_awaitable_blocking(self._e.backend.node_update(
                    ids=[nid],
                    documents=[document],
                    metadatas=[dict(meta)],
                ))
                try:
                    self._e.write.index_node_docs(Node.model_validate(d))
                except Exception:
                    pass

        def _save_edge(d: dict[str, Any]) -> None:
            eid = d["id"]
            prior = _backend_object(
                run_awaitable_blocking(
                    self._e.backend.edge_get(ids=[eid], include=["metadatas"])
                )
            )
            meta = (_backend_objects(prior.get("metadatas")) or [{}])[0]
            refs = d.get("mentions", d.get("references", []))
            meta = {**meta, "references": json.dumps(refs, ensure_ascii=False)}
            document = json.dumps(d, ensure_ascii=False)
            if not self._e.write.rust_postgres_replace_existing(
                entity_kind="edge",
                entity_id=eid,
                document=document,
                metadata_patch=dict(meta),
                payload=d,
            ):
                run_awaitable_blocking(self._e.backend.edge_update(
                    ids=[eid],
                    documents=[document],
                    metadatas=[dict(meta)],
                ))

        node_ids = set()
        try:
            nd = _backend_object(
                run_awaitable_blocking(
                    self._e.backend.node_docs_get(
                        where={"doc_id": doc_id}, include=["metadatas"]
                    )
                )
            )
            for m in _backend_list(nd.get("metadatas")):
                metadata = _backend_object(m)
                if metadata.get("node_id"):
                    node_ids.add(str(metadata["node_id"]))
            summary["deleted_node_doc_rows"] = len(_backend_list(nd.get("ids")))
        except Exception:
            pass
        if not node_ids:
            try:
                q = _backend_object(
                    run_awaitable_blocking(
                        self._e.backend.node_get(where={"doc_id": doc_id})
                    )
                )
                for nid in _backend_strings(q.get("ids")):
                    node_ids.add(nid)
            except Exception:
                pass

        edge_ids = set()
        try:
            ee = _backend_object(
                run_awaitable_blocking(
                    self._e.backend.edge_endpoints_get(
                        where={"doc_id": doc_id}, include=["metadatas"]
                    )
                )
            )
            for m in _backend_list(ee.get("metadatas")):
                metadata = _backend_object(m)
                if metadata.get("edge_id"):
                    edge_ids.add(str(metadata["edge_id"]))
            summary["deleted_edge_endpoints"] = len(_backend_list(ee.get("ids")))
        except Exception:
            pass
        if not edge_ids:
            try:
                q = _backend_object(
                    run_awaitable_blocking(
                        self._e.backend.edge_get(where={"doc_id": doc_id})
                    )
                )
                for eid in _backend_strings(q.get("ids")):
                    edge_ids.add(eid)
            except Exception:
                pass

        nodes_map = _load_many("node", node_ids)
        for nid, d in nodes_map.items():
            refs_key, keep, removed = _filter_reference_payload(d)
            if removed:
                summary["deleted_node_refs"] += removed
                if keep:
                    d[refs_key] = keep
                    _save_node(d)
                    summary["updated_nodes"] += 1
                else:
                    native_deleted = self._e.write.rust_postgres_delete_existing(
                        entity_kind="node", entity_ids=[nid]
                    )
                    if not native_deleted:
                        try:
                            run_awaitable_blocking(
                                self._e.backend.node_delete(ids=[nid])
                            )
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
            refs_key, keep, removed = _filter_reference_payload(d)

            try:
                run_awaitable_blocking(self._e.backend.edge_endpoints_delete(
                    where={"edge_id": eid, "doc_id": doc_id}
                ))
            except Exception:
                pass

            if removed:
                summary["deleted_edge_refs"] += removed
                if keep:
                    d[refs_key] = keep
                    _save_edge(d)
                    summary["updated_edges"] += 1
                else:
                    native_deleted = self._e.write.rust_postgres_delete_existing(
                        entity_kind="edge", entity_ids=[eid]
                    )
                    if not native_deleted:
                        try:
                            run_awaitable_blocking(
                                self._e.backend.edge_delete(ids=[eid])
                            )
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

    def prune_node_from_edges(self, node_id: str) -> dict[str, set[str]]:
        eps = _backend_object(
            run_awaitable_blocking(
                self._e.backend.edge_endpoints_get(
                    where={
                        "$and": [
                            {"endpoint_id": node_id},
                            {"endpoint_type": "node"},
                        ]
                    },
                    include=["documents"],
                )
            )
        )
        if not _backend_strings(eps.get("ids")):
            return {"deleted_edges": set(), "updated_edges": set()}
        if eps_doc := _backend_strings(eps.get("documents")):
            pass
        else:
            raise Exception("Document loss")
        edge_ids = list({json.loads(doc)["edge_id"] for doc in eps_doc})
        edges = _backend_object(
            run_awaitable_blocking(
                self._e.backend.edge_get(
                    ids=edge_ids, include=["documents", "metadatas"]
                )
            )
        )

        removed_edge_ids: set[str] = set()
        updated_edge_ids: set[str] = set()

        def _delete_base_edge(edge_id: str) -> None:
            if not self._e.write.rust_postgres_delete_existing(
                entity_kind="edge", entity_ids=[edge_id]
            ):
                run_awaitable_blocking(self._e.backend.edge_delete(ids=[edge_id]))

        def _replace_base_edge(edge: Edge, metadata: dict[str, Any] | None) -> None:
            replacement_metadata = strip_none(
                {
                    "doc_id": (metadata or {}).get("doc_id"),
                    "relation": edge.relation,
                    "source_ids": json_or_none(edge.source_ids),
                    "target_ids": json_or_none(edge.target_ids),
                    "type": edge.type,
                    "summary": edge.summary,
                    "domain_id": edge.domain_id,
                    "canonical_entity_id": edge.canonical_entity_id,
                    "properties": json_or_none(edge.properties),
                    "references": json_or_none(
                        [
                            ref.model_dump(field_mode="backend")
                            for ref in (edge.mentions or [])
                        ]
                    ),
                }
            )
            document = edge.model_dump_json(field_mode="backend")
            if not self._e.write.rust_postgres_replace_existing(
                entity_kind="edge",
                entity_id=edge.safe_get_id(),
                document=document,
                metadata_patch=replacement_metadata,
                payload=edge.model_dump(field_mode="backend", exclude=["embedding"]),
            ):
                run_awaitable_blocking(self._e.backend.edge_update(
                    ids=[edge.safe_get_id()],
                    documents=[document],
                    metadatas=[replacement_metadata],
                ))
                self._e.write.index_edge_refs(edge)

        for eid, edoc, meta in zip(
            _backend_strings(edges.get("ids")),
            _backend_strings(edges.get("documents")),
            _backend_list(edges.get("metadatas")),
        ):
            e = Edge.model_validate_json(edoc)
            metadata = _backend_object(meta)
            relation = metadata.get("relation") or e.relation

            if relation == "same_as":
                edge_deleted, new_edge = self._e.adjudicate.rebalance_same_as_edge(
                    e,
                    removed_node_id=node_id,
                )
                if edge_deleted or (new_edge is None):
                    _delete_base_edge(eid)
                    run_awaitable_blocking(self._e.backend.edge_endpoints_delete(where={"edge_id": eid}))
                    removed_edge_ids.add(eid)
                else:
                    _replace_base_edge(new_edge, metadata)
                    run_awaitable_blocking(self._e.backend.edge_endpoints_delete(where={"edge_id": eid}))
                    ep_ids, ep_docs, ep_metas = [], [], []
                    for role, node_ids in (
                        ("src", new_edge.source_ids or []),
                        ("tgt", new_edge.target_ids or []),
                    ):
                        for nid in node_ids:
                            ep_id = f"{eid}::{role}::{nid}"
                            node_doc = _backend_object(
                                run_awaitable_blocking(
                                    self._e.backend.node_get(
                                        ids=[nid], include=["documents"]
                                    )
                                )
                            )
                            per_doc_id = None
                            if node_doc_doc := _backend_strings(node_doc.get("documents")):
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
                _delete_base_edge(eid)
                run_awaitable_blocking(self._e.backend.edge_endpoints_delete(where={"edge_id": eid}))
                removed_edge_ids.add(eid)
            else:
                e.source_ids, e.target_ids = new_src, new_tgt
                _replace_base_edge(e, metadata)
                run_awaitable_blocking(self._e.backend.edge_endpoints_delete(
                    where={"$and": [{"edge_id": eid}, {"node_id": node_id}]}
                ))
                updated_edge_ids.add(eid)

        return {
            "deleted_edges": removed_edge_ids,
            "updated_edges": updated_edge_ids - removed_edge_ids,
        }

    def rollback_many_documents(self, document_ids: list[str]) -> dict[str, int]:
        totals = {
            "deleted_nodes": 0,
            "deleted_edges": 0,
            "updated_edges": 0,
            "deleted_docs": 0,
        }
        for did in document_ids:
            res = self.rollback_document(did)
            totals["deleted_docs"] += 1
            totals["deleted_nodes"] += len(cast(list[object], res["deleted_node_ids"]))
            totals["deleted_edges"] += len(cast(list[object], res["deleted_edge_ids"]))
            totals["updated_edges"] += cast(int, res["updated_edges"])
        return totals

    def delete_edges_by_ids(self, edge_ids: list[str]) -> None:
        self._e.write.delete_edges_by_ids(edge_ids)

    def prune_node_refs_for_doc(self, node_id: str, doc_id: str) -> bool:
        return self._e.write.prune_node_refs_for_doc(node_id, doc_id)
