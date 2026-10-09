from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, cast

from ...acl.graph import (
    ACLDecision,
    ACLGrain,
    ACLNodeReadDecision,
    ACLRecord,
    ACLTarget,
    ACLUsageDecision,
)
from ...acl.models import ACLEdge, ACLNode
from ...cdc.change_event import EntityRefModel
from ...engine_core.models import (
    Document,
    Domain,
    Edge,
    Grounding,
    Node,
    PureChromaEdge,
    PureChromaNode,
    Span,
)
from ...engine_core.vector_search import VectorSearchHit
from ...id_provider import stable_id
from ...json_types import JsonObject, JsonValue
from ...typing_interfaces import QueryEmbeddingInput, ReadLike, WriteLike
from .base import NamespaceProxy

if TYPE_CHECKING:
    from ..engine import GraphKnowledgeEngine


def _as_acl_grain(value: str | None) -> ACLGrain | None:
    if value is None:
        return None
    if value not in {"document", "grounding", "span", "node", "edge", "artifact"}:
        raise ValueError(f"unsupported ACL grain: {value}")
    return cast(ACLGrain, value)


class ACLSubsystem(NamespaceProxy["GraphKnowledgeEngine"]):
    def __init__(self, engine: GraphKnowledgeEngine) -> None:
        super().__init__(engine)
        # ACLGraph is a cacheable projection. Loaders point it back to canonical truth.
        self._e.acl_graph.bind_loaders(
            record_loader=lambda truth_graph, grain, entity_id, target_item_id: self._load_acl_records_from_truth(
                truth_graph=truth_graph,
                grain=grain,
                entity_id=entity_id,
                target_item_id=target_item_id,
            ),
            target_loader=lambda truth_graph, grain, target_item_id: self._load_acl_entity_ids_from_truth(
                truth_graph=truth_graph,
                grain=grain,
                target_item_id=target_item_id,
            ),
        )

    def current_principal_context(self) -> tuple[str, tuple[str, ...], str | None]:
        try:
            from ...server.auth_middleware import (
                get_current_agent_id,
                get_security_scope,
            )

            principal_id = str(get_current_agent_id() or "").strip().lower()
            security_scope = str(get_security_scope() or "").strip().lower() or None
        except Exception:
            principal_id = ""
            security_scope = None
        return principal_id or "system", (), security_scope

    def usage_ids_for_item(self, item: Node | Edge) -> tuple[tuple[str, ...], tuple[str, ...]]:
        groundings: list[str] = []
        spans: list[str] = []
        item_id = str(item.safe_get_id())
        for i, grounding in enumerate(getattr(item, "mentions", None) or []):
            groundings.append(f"{item_id}::mention::{i}")
            for j, _span in enumerate(getattr(grounding, "spans", None) or []):
                spans.append(f"{item_id}::mention::{i}::span::{j}")
        return tuple(groundings), tuple(spans)

    def _metadata_acl_defaults(
        self, item: Node | Edge
    ) -> tuple[str, str, str | None, tuple[str, ...], tuple[str, ...]]:
        metadata = getattr(item, "metadata", {}) or {}
        principal_id, _groups, current_scope = self.current_principal_context()
        raw_mode = str(metadata.get("visibility") or metadata.get("acl_mode") or "").strip().lower()
        mode = raw_mode if raw_mode in {"private", "shared", "scope", "group", "public"} else "private"
        owner_id = str(
            metadata.get("owner_agent_id")
            or metadata.get("agent_id")
            or metadata.get("owner_id")
            or metadata.get("user_id")
            or principal_id
        ).strip().lower()
        security_scope = str(
            metadata.get("security_scope")
            or metadata.get("owner_security_scope")
            or current_scope
            or ""
        ).strip().lower() or None
        shared_principals = tuple(
            str(v).strip().lower()
            for v in (metadata.get("shared_with_agents") or metadata.get("shared_with") or ())
            if str(v).strip()
        )
        shared_groups = tuple(
            str(v).strip().lower()
            for v in (metadata.get("shared_with_groups") or metadata.get("shared_groups") or ())
            if str(v).strip()
        )
        return mode, owner_id, security_scope, shared_principals, shared_groups

    def should_auto_acl_item(self, item: Node | Edge) -> bool:
        metadata = getattr(item, "metadata", {}) or {}
        entity_type = str(metadata.get("entity_type") or "")
        return not entity_type.startswith("acl_") and entity_type != "acl_record"

    def writes_can_share_backend_transaction(self) -> bool:
        """Return true when truth rows and ACL rows can share one backend txn."""
        return type(getattr(self._e, "_backend_uow", None)).__name__ != "NoopUnitOfWork"

    def _default_acl_context(
        self,
        item: Node | Edge,
        *,
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        mode, owner_id, security_scope, shared_principals, shared_groups = self._metadata_acl_defaults(item)
        principal_id, _groups, _scope = self.current_principal_context()
        if overrides:
            mode = str(overrides.get("mode") or mode)
            owner_id = str(overrides.get("owner_id") or owner_id)
            security_scope = overrides.get("security_scope", security_scope)
            created_by = str(overrides.get("created_by") or principal_id)
            shared_principals = tuple(overrides.get("shared_with_principals") or shared_principals)
            shared_groups = tuple(overrides.get("shared_with_groups") or shared_groups)
        else:
            created_by = principal_id
        return {
            "truth_graph": self._e.kg_graph_type,
            "entity_id": str(item.safe_get_id()),
            "mode": mode,
            "created_by": created_by,
            "owner_id": owner_id,
            "security_scope": security_scope,
            "shared_with_principals": tuple(shared_principals),
            "shared_with_groups": tuple(shared_groups),
        }

    def _default_acl_records_for_item(self, item: Node | Edge, *, grain: str) -> tuple[ACLRecord, ...]:
        ctx = self._default_acl_context(item)
        entity_id = str(ctx["entity_id"])

        def _record(grain_name: str, target_item_id: str | None = None) -> ACLRecord:
            return ACLRecord(
                target=ACLTarget(
                    truth_graph=str(ctx["truth_graph"]),
                    entity_id=entity_id,
                    grain=grain_name,  # type: ignore[arg-type]
                    target_item_id=target_item_id,
                ),
                version=self._next_version(
                    grain=grain_name,
                    truth_graph=str(ctx["truth_graph"]),
                    entity_id=entity_id,
                    target_item_id=target_item_id,
                ),
                mode=str(ctx["mode"]),  # type: ignore[arg-type]
                created_by=ctx["created_by"],
                owner_id=ctx["owner_id"],
                security_scope=ctx["security_scope"],
                shared_with_principals=tuple(ctx["shared_with_principals"]),
                shared_with_groups=tuple(ctx["shared_with_groups"]),
            )

        records = [_record(grain)]
        grounding_ids, span_ids = self.usage_ids_for_item(item)
        records.extend(_record("grounding", item_id) for item_id in grounding_ids)
        records.extend(_record("span", item_id) for item_id in span_ids)
        return tuple(records)

    def append_canonical_write_events_for_item(self, item: Node | Edge, *, grain: str) -> None:
        """Append canonical truth and ACL events before non-transactional projection."""
        if not self.should_auto_acl_item(item):
            return
        entity_kind = "edge" if grain == "edge" else "node"
        payload = item.model_dump(field_mode="backend", exclude=["embedding"])
        self._e._append_event_for_entity(
            namespace=self._e.namespace,
            entity_kind=entity_kind,
            entity_id=str(item.safe_get_id()),
            op="ADD",
            payload=payload if isinstance(payload, dict) else {},
        )
        for record in self._default_acl_records_for_item(item, grain=grain):
            acl_node = self._build_acl_record_node(record)
            acl_payload = acl_node.model_dump(field_mode="backend", exclude=["embedding"])
            self._e._append_event_for_entity(
                namespace=self._e.namespace,
                entity_kind="node",
                entity_id=str(acl_node.safe_get_id()),
                op="ADD",
                payload=acl_payload if isinstance(acl_payload, dict) else {},
            )

    def _next_version(
        self,
        *,
        grain: str,
        truth_graph: str,
        entity_id: str,
        target_item_id: str | None = None,
    ) -> int:
        previous = self._latest_persisted_acl_record(
            grain=grain,
            truth_graph=truth_graph,
            entity_id=entity_id,
            target_item_id=target_item_id,
        )
        return 1 if previous is None else previous.version + 1

    def record_default_acl_for_item(self, item: Node | Edge, *, grain: str) -> None:
        """Write default ACL rows for one truth item.

        This is the normal write-path helper. It writes the node/edge ACL record
        plus any grounding/span usage ACL rows derived from the item's mentions.
        Callers that need cross-backend best-effort atomicity should wrap the call
        in `engine.uow()`; the ACL subsystem itself keeps the write sequence
        together, but the backend decides whether that sequence is transactional.
        """
        if not self.should_auto_acl_item(item):
            return
        with self._e.uow():
            base_kwargs = self._default_acl_context(item)
            entity_id = str(base_kwargs["entity_id"])
            self.record_acl(
                grain=grain,
                version=self._next_version(
                    grain=grain,
                    truth_graph=self._e.kg_graph_type,
                    entity_id=entity_id,
                ),
                **base_kwargs,
            )
            grounding_ids, span_ids = self.usage_ids_for_item(item)
            for grounding_id in grounding_ids:
                self.record_acl(
                    grain="grounding",
                    target_item_id=grounding_id,
                    version=self._next_version(
                        grain="grounding",
                        truth_graph=self._e.kg_graph_type,
                        entity_id=entity_id,
                        target_item_id=grounding_id,
                    ),
                    **base_kwargs,
                )
            for span_id in span_ids:
                self.record_acl(
                    grain="span",
                    target_item_id=span_id,
                    version=self._next_version(
                        grain="span",
                        truth_graph=self._e.kg_graph_type,
                        entity_id=entity_id,
                        target_item_id=span_id,
                    ),
                    **base_kwargs,
                )

    def repair_default_acl_for_item(
        self,
        item: Node | Edge,
        *,
        grain: str,
        defaults: dict[str, Any] | None = None,
    ) -> int:
        """Repair missing default ACL rows for one truth item.

        Unlike `record_default_acl_for_item`, this helper is idempotent at the
        grain/usage level. It only writes rows that are absent, so it can be used
        after a crash or partial write on eventually consistent backends.
        """
        if not self.should_auto_acl_item(item):
            return 0
        base_kwargs = self._default_acl_context(item, overrides=defaults)
        entity_id = str(base_kwargs["entity_id"])
        repaired = 0

        def _repair_one(*, grain_name: str, target_item_id: str | None = None) -> None:
            nonlocal repaired
            existing = self._latest_persisted_acl_record(
                grain=grain_name,
                truth_graph=self._e.kg_graph_type,
                entity_id=entity_id,
                target_item_id=target_item_id,
            )
            if existing is not None:
                return
            self.record_acl(
                grain=grain_name,
                target_item_id=target_item_id,
                version=self._next_version(
                    grain=grain_name,
                    truth_graph=self._e.kg_graph_type,
                    entity_id=entity_id,
                    target_item_id=target_item_id,
                ),
                **base_kwargs,
            )
            repaired += 1

        _repair_one(grain_name=grain)
        grounding_ids, span_ids = self.usage_ids_for_item(item)
        for grounding_id in grounding_ids:
            _repair_one(grain_name="grounding", target_item_id=grounding_id)
        for span_id in span_ids:
            _repair_one(grain_name="span", target_item_id=span_id)
        return repaired

    def repair_acl_records_from_events(
        self,
        *,
        truth_graph: str | None = None,
        entity_id: str | None = None,
        limit: int = 256,
    ) -> dict[str, Any]:
        """Rebuild missing ACL record nodes from recent entity events only.

        This is the bounded event-source recovery path. It does not scan all
        truth nodes or edges. It only rehydrates missing ACL record rows from
        recent ACL record events, then lets normal ACL lookup see them again.
        """
        graph_name = truth_graph or self._e.kg_graph_type
        iter_events = getattr(self._e.meta_sqlite, "iter_entity_events", None)
        latest_seq_getter = getattr(self._e.meta_sqlite, "get_latest_entity_event_seq", None)
        if iter_events is None or latest_seq_getter is None:
            return {
                "truth_graph": graph_name,
                "scanned_events": 0,
                "repaired_acl_records": 0,
            }
        tail = max(0, int(limit))
        latest_seq = int(latest_seq_getter(namespace=self._e.namespace) or 0)
        from_seq = max(1, latest_seq - tail + 1) if latest_seq else 1
        scanned = 0
        repaired = 0
        seen_acl_ids: set[str] = set()
        prev_log = getattr(self._e, "_disable_event_log", False)
        self._e._disable_event_log = True
        try:
            for _seq, entity_kind, event_entity_id, op, payload_json in iter_events(
                namespace=self._e.namespace,
                from_seq=from_seq,
            ):
                scanned += 1
                if entity_kind != "node" or op not in {"ADD", "REPLACE"}:
                    continue
                try:
                    payload = json.loads(payload_json or "{}")
                except Exception:
                    payload = {}
                metadata = payload.get("metadata") if isinstance(payload, dict) else {}
                if not isinstance(metadata, dict):
                    continue
                if metadata.get("entity_type") != "acl_record":
                    continue
                if metadata.get("acl_truth_graph") != graph_name:
                    continue
                acl_target_entity_id = str(metadata.get("acl_target_entity_id") or "")
                if entity_id is not None and acl_target_entity_id != str(entity_id):
                    continue
                acl_node_id = str(event_entity_id)
                if acl_node_id in seen_acl_ids:
                    continue
                if self._e.raw_read.get_nodes(node_type=Node, ids=[acl_node_id]):
                    continue
                try:
                    self._e.raw_write.add_node(Node.model_validate(payload))
                    repaired += 1
                    seen_acl_ids.add(acl_node_id)
                except Exception:
                    continue
        finally:
            self._e._disable_event_log = prev_log
        return {
            "truth_graph": graph_name,
            "scanned_events": scanned,
            "repaired_acl_records": repaired,
        }

    def _acl_dummy_span(self, truth_graph: str) -> Span:
        return Span(
            collection_page_url=f"acl/{truth_graph}",
            document_page_url=f"acl/{truth_graph}",
            doc_id=f"acl:{truth_graph}",
            insertion_method="acl_record",
            page_number=1,
            start_char=0,
            end_char=1,
            excerpt="a",
            context_before="",
            context_after="",
            chunk_id=None,
            source_cluster_id=None,
            verification=None,
        )

    def _acl_record_node_id(
        self,
        *,
        grain: str,
        truth_graph: str,
        entity_id: str,
        target_item_id: str | None,
        version: int,
    ) -> str:
        return str(
            stable_id(
                "acl_record",
                truth_graph,
                entity_id,
                grain,
                target_item_id or "-",
                str(version),
            )
        )

    def _acl_supersedes_edge_id(self, *, current_id: str, previous_id: str) -> str:
        return str(stable_id("acl_supersedes", current_id, previous_id))

    def _build_acl_record_node(self, record: ACLRecord) -> ACLNode:
        target = record.target
        node_id = self._acl_record_node_id(
            grain=target.grain,
            truth_graph=target.truth_graph,
            entity_id=target.entity_id,
            target_item_id=target.target_item_id,
            version=record.version,
        )
        target_item = target.target_item_id or ""
        return ACLNode(
            id=node_id,
            label=f"acl:{target.truth_graph}:{target.entity_id}:{target.grain}",
            type="entity",
            summary=f"{record.mode} acl for {target.entity_id} {target.grain}",
            doc_id=f"acl:{target.truth_graph}",
            mentions=[Grounding(spans=[self._acl_dummy_span(target.truth_graph)])],
            metadata={
                "entity_type": "acl_record",
                "acl_truth_graph": target.truth_graph,
                "acl_target_entity_id": target.entity_id,
                "acl_target_grain": target.grain,
                "acl_target_item_id": target_item,
                "acl_version": record.version,
                "acl_mode": record.mode,
                "created_by": record.created_by,
                "owner_id": record.owner_id,
                "security_scope": record.security_scope,
                "tombstoned": record.tombstoned,
                "supersedes_version": record.supersedes_version,
                "level_from_root": 0,
            },
            properties={
                "shared_with_principals": list(record.shared_with_principals),
                "shared_with_groups": list(record.shared_with_groups),
                "source_ids": list(record.source_ids),
                "derivation_type": record.derivation_type,
                # Node properties use a scalar/list JSON contract. Keep the
                # structured audit lossless without widening that public model.
                "derivation_audit": json.dumps(
                    dict(record.derivation_audit or {}),
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                ),
                "target_item_id": target_item,
            },
            embedding=None,
            level_from_root=0,
            domain_id=None,
            canonical_entity_id=None,
        )

    def _build_acl_supersedes_edge(
        self, *, current: ACLRecord, previous: ACLRecord
    ) -> ACLEdge:
        current_id = self._acl_record_node_id(
            grain=current.target.grain,
            truth_graph=current.target.truth_graph,
            entity_id=current.target.entity_id,
            target_item_id=current.target.target_item_id,
            version=current.version,
        )
        previous_id = self._acl_record_node_id(
            grain=previous.target.grain,
            truth_graph=previous.target.truth_graph,
            entity_id=previous.target.entity_id,
            target_item_id=previous.target.target_item_id,
            version=previous.version,
        )
        return ACLEdge(
            id=self._acl_supersedes_edge_id(
                current_id=current_id, previous_id=previous_id
            ),
            label="acl_supersedes",
            type="relationship",
            summary="acl supersedes previous acl state",
            relation="acl_supersedes",
            source_ids=[current_id],
            target_ids=[previous_id],
            source_edge_ids=[],
            target_edge_ids=[],
            doc_id=f"acl:{current.target.truth_graph}",
            mentions=[Grounding(spans=[self._acl_dummy_span(current.target.truth_graph)])],
            metadata={
                "entity_type": "acl_supersedes",
                "acl_truth_graph": current.target.truth_graph,
                "acl_target_entity_id": current.target.entity_id,
                "acl_target_grain": current.target.grain,
                "acl_target_item_id": current.target.target_item_id or "",
                "level_from_root": 0,
            },
            properties=None,
            embedding=None,
            domain_id=None,
            canonical_entity_id=None,
        )

    def _record_from_acl_node(self, node: Node) -> ACLRecord | None:
        metadata = getattr(node, "metadata", {}) or {}
        if str(metadata.get("entity_type") or "") != "acl_record":
            return None
        properties = getattr(node, "properties", {}) or {}
        target_item_id = metadata.get("acl_target_item_id")
        if target_item_id == "":
            target_item_id = None
        raw_derivation_audit = properties.get("derivation_audit")
        if isinstance(raw_derivation_audit, str):
            try:
                derivation_audit = json.loads(raw_derivation_audit)
            except (TypeError, ValueError):
                derivation_audit = {}
        elif isinstance(raw_derivation_audit, dict):
            derivation_audit = dict(raw_derivation_audit)
        else:
            derivation_audit = {}
        return ACLRecord(
            target=ACLTarget(
                truth_graph=str(metadata.get("acl_truth_graph") or ""),
                entity_id=str(metadata.get("acl_target_entity_id") or ""),
                grain=str(metadata.get("acl_target_grain") or "node"),  # type: ignore[arg-type]
                target_item_id=target_item_id,
            ),
            version=int(metadata.get("acl_version") or 0),
            mode=str(metadata.get("acl_mode") or "private"),  # type: ignore[arg-type]
            created_by=metadata.get("created_by"),
            owner_id=metadata.get("owner_id"),
            security_scope=metadata.get("security_scope"),
            shared_with_principals=tuple(properties.get("shared_with_principals") or ()),
            shared_with_groups=tuple(properties.get("shared_with_groups") or ()),
            source_ids=tuple(properties.get("source_ids") or ()),
            derivation_type=properties.get("derivation_type"),
            derivation_audit=derivation_audit,
            tombstoned=bool(metadata.get("tombstoned")),
            supersedes_version=metadata.get("supersedes_version"),
        )

    def _load_acl_records_from_truth(
        self,
        *,
        grain: str | None = None,
        truth_graph: str,
        entity_id: str,
        target_item_id: str | None = None,
    ) -> tuple[ACLRecord, ...]:
        # Canonical ACL read path. Used by ACLGraph cache miss and by repair/rebuild.
        where_terms: list[dict[str, object]] = [
            {"entity_type": "acl_record"},
            {"acl_truth_graph": truth_graph},
        ]
        if entity_id:
            where_terms.append({"acl_target_entity_id": entity_id})
        if grain is not None:
            where_terms.append({"acl_target_grain": grain})
        nodes = self._e.raw_read.get_nodes(
            node_type=Node,
            where={"$and": where_terms},
            limit=400,
        )
        records: list[ACLRecord] = []
        for node in nodes:
            record = self._record_from_acl_node(node)
            if record is None:
                continue
            if target_item_id is not None and record.target.target_item_id != target_item_id:
                continue
            records.append(record)
        return tuple(records)

    def _latest_persisted_acl_record(
        self,
        *,
        grain: str | None = None,
        truth_graph: str,
        entity_id: str,
        target_item_id: str | None = None,
    ) -> ACLRecord | None:
        records = self._e.acl_graph._load_records(
            truth_graph=truth_graph,
            grain=_as_acl_grain(grain),
            entity_id=entity_id,
            target_item_id=target_item_id,
        )
        if not records:
            records = self._load_acl_records_from_truth(
                grain=grain,
                truth_graph=truth_graph,
                entity_id=entity_id,
                target_item_id=target_item_id,
            )
        if not records:
            return None
        active = [r for r in records if not r.tombstoned]
        pool = active or list(records)
        rank = {"public": 0, "group": 1, "shared": 2, "scope": 3, "private": 4}
        return max(pool, key=lambda r: (r.version, rank.get(r.mode, 99)))

    def _load_acl_entity_ids_from_truth(
        self,
        *,
        truth_graph: str,
        grain: str,
        target_item_id: str,
    ) -> tuple[str, ...]:
        # Reverse index path: find owning entities for one target item key.
        nodes = self._e.raw_read.get_nodes(
            node_type=Node,
            where={
                "$and": [
                    {"entity_type": "acl_record"},
                    {"acl_truth_graph": truth_graph},
                    {"acl_target_grain": grain},
                    {"acl_target_item_id": target_item_id},
                ]
            },
            limit=400,
        )
        entity_ids = {
            str((getattr(node, "metadata", {}) or {}).get("acl_target_entity_id") or "")
            for node in nodes
        }
        entity_ids.discard("")
        return tuple(sorted(entity_ids))

    def acl_entity_ids_for_target_item(
        self,
        *,
        truth_graph: str,
        grain: str,
        target_item_id: str,
    ) -> tuple[str, ...]:
        return self._e.acl_graph.entity_ids_for_target_item(
            truth_graph=truth_graph,
            grain=cast(ACLGrain, _as_acl_grain(grain)),
            target_item_id=target_item_id,
        )

    def record_acl(
        self,
        *,
        grain: str = "node",
        truth_graph: str,
        entity_id: str,
        target_item_id: str | None = None,
        version: int,
        mode: str,
        created_by: str | None = None,
        owner_id: str | None = None,
        security_scope: str | None = None,
        shared_with_principals: Sequence[str] = (),
        shared_with_groups: Sequence[str] = (),
        source_ids: Sequence[str] = (),
        derivation_type: str | None = None,
        derivation_audit: dict[str, JsonValue] | None = None,
        supersedes_version: int | None = None,
        tombstoned: bool = False,
    ) -> ACLRecord:
        with self._e.uow():
            previous = self._latest_persisted_acl_record(
                grain=grain,
                truth_graph=truth_graph,
                entity_id=entity_id,
                target_item_id=target_item_id,
            )
            record = ACLRecord(
                target=ACLTarget(
                    truth_graph=truth_graph,
                    entity_id=entity_id,
                    grain=grain,  # type: ignore[arg-type]
                    target_item_id=target_item_id,
                ),
                version=version,
                mode=mode,  # type: ignore[arg-type]
                created_by=created_by,
                owner_id=owner_id,
                security_scope=security_scope,
                shared_with_principals=tuple(shared_with_principals),
                shared_with_groups=tuple(shared_with_groups),
                source_ids=tuple(source_ids),
                derivation_type=derivation_type,
                derivation_audit=dict(derivation_audit or {}),
                supersedes_version=supersedes_version,
                tombstoned=tombstoned,
            )
            acl_node = self._build_acl_record_node(record)
            self._e.raw_write.add_node(acl_node)
            if previous is not None:
                self._e.raw_write.add_edge(
                    self._build_acl_supersedes_edge(current=record, previous=previous)
                )
            try:
                self._e._emit_change(
                    op="node.upsert",
                    entity=EntityRefModel(
                        kind="node",
                        id=entity_id,
                        kg_graph_type=truth_graph,
                        url=None,
                    ),
                    payload={
                        "acl_record": {
                            "target": {
                                "grain": grain,
                                "truth_graph": truth_graph,
                                "entity_id": entity_id,
                                "target_item_id": target_item_id,
                            },
                            "version": version,
                            "mode": mode,
                            "created_by": created_by,
                            "owner_id": owner_id,
                            "security_scope": security_scope,
                            "shared_with_principals": list(shared_with_principals),
                            "shared_with_groups": list(shared_with_groups),
                            "source_ids": list(source_ids),
                            "derivation_type": derivation_type,
                            "derivation_audit": dict(derivation_audit or {}),
                            "supersedes_version": supersedes_version,
                            "tombstoned": tombstoned,
                        }
                    },
                )
            except Exception:
                pass
            self._e.acl_graph.invalidate(
                truth_graph=truth_graph,
                grain=grain,  # type: ignore[arg-type]
                entity_id=entity_id,
                target_item_id=target_item_id,
            )
            return record

    def repair_missing_default_acls(
        self,
        *,
        truth_graph: str | None = None,
        limit: int = 10_000,
    ) -> dict[str, Any]:
        """Repair missing default ACL rows from persisted truth rows.

        This is the eventual-consistency recovery path for backends that cannot
        make truth writes and ACL writes atomic. It scans truth nodes and edges,
        skips ACL helper rows, and only materializes ACL rows that are absent.
        """
        graph_name = truth_graph or self._e.kg_graph_type
        repaired = 0
        scanned = 0
        iter_events = getattr(self._e.meta_sqlite, "iter_entity_events", None)
        if iter_events is not None:
            prev_log = getattr(self._e, "_disable_event_log", False)
            self._e._disable_event_log = True
            try:
                for _seq, entity_kind, entity_id, op, payload_json in iter_events(
                    namespace=self._e.namespace,
                    from_seq=1,
                ):
                    if entity_kind != "node" or op not in {"ADD", "REPLACE"}:
                        continue
                    try:
                        payload = json.loads(payload_json or "{}")
                    except Exception:
                        payload = {}
                    metadata = payload.get("metadata") if isinstance(payload, dict) else {}
                    if not isinstance(metadata, dict):
                        continue
                    if metadata.get("entity_type") != "acl_record":
                        continue
                    if metadata.get("acl_truth_graph") != graph_name:
                        continue
                    if self._e.raw_read.get_nodes(node_type=Node, ids=[str(entity_id)]):
                        continue
                    self._e.raw_write.add_node(Node.model_validate(payload))
                    repaired += 1
            finally:
                self._e._disable_event_log = prev_log
        nodes = self._e.raw_read.get_nodes(
            node_type=Node,
            limit=int(limit),
        )
        for node in nodes:
            metadata = getattr(node, "metadata", {}) or {}
            if str(metadata.get("entity_type") or "").startswith("acl_"):
                continue
            scanned += 1
            repaired += self.repair_default_acl_for_item(node, grain="node")
        edges = self._e.raw_read.get_edges(
            edge_type=Edge,
            limit=int(limit),
        )
        for edge in edges:
            metadata = getattr(edge, "metadata", {}) or {}
            if str(metadata.get("entity_type") or "").startswith("acl_"):
                continue
            scanned += 1
            repaired += self.repair_default_acl_for_item(edge, grain="edge")
        return {
            "truth_graph": graph_name,
            "scanned_truth_items": scanned,
            "repaired_acl_records": repaired,
        }

    def prefetch_acl_neighborhood(
        self,
        *,
        truth_graph: str,
        entity_id: str,
        item_grain: ACLGrain = "span",
        grounding_item_ids: Sequence[str] = (),
        target_item_ids: Sequence[str] = (),
        max_items: int = 32,
    ) -> dict[str, tuple[str, ...]]:
        acl_graph = self._e.acl_graph
        acl_graph.latest_record(
            grain="node",
            truth_graph=truth_graph,
            entity_id=entity_id,
        )
        warmed_items: list[str] = []
        warmed_neighbors: list[str] = []
        budget = max(0, int(max_items))
        for item_id in tuple(grounding_item_ids) + tuple(target_item_ids):
            if len(warmed_items) >= budget:
                break
            item_id = str(item_id)
            warmed_items.append(item_id)
            acl_graph.latest_record(
                grain=("grounding" if item_id in grounding_item_ids else item_grain),
                truth_graph=truth_graph,
                entity_id=entity_id,
                target_item_id=item_id,
            )
            neighbor_ids = acl_graph.entity_ids_for_target_item(
                truth_graph=truth_graph,
                grain=("grounding" if item_id in grounding_item_ids else item_grain),
                target_item_id=item_id,
            )
            for neighbor_id in neighbor_ids:
                if neighbor_id == entity_id or neighbor_id in warmed_neighbors:
                    continue
                if len(warmed_neighbors) >= budget:
                    break
                acl_graph.latest_record(
                    grain="node",
                    truth_graph=truth_graph,
                    entity_id=neighbor_id,
                )
                warmed_neighbors.append(neighbor_id)
        return {
            "warmed_items": tuple(warmed_items),
            "warmed_neighbor_entity_ids": tuple(warmed_neighbors),
        }

    def rebuild_acl_graph_from_truth(self, *, truth_graph: str | None = None) -> dict[str, Any]:
        acl_graph = self._e.acl_graph
        acl_graph.clear()
        nodes = self._e.raw_read.get_nodes(
            node_type=Node,
            where={"entity_type": "acl_record"},
            limit=10_000,
        )
        rebuilt = 0
        for node in nodes:
            record = self._record_from_acl_node(node)
            if record is None:
                continue
            if truth_graph is not None and record.target.truth_graph != str(truth_graph):
                continue
            acl_graph.add_record(
                grain=record.target.grain,
                truth_graph=record.target.truth_graph,
                entity_id=record.target.entity_id,
                target_item_id=record.target.target_item_id,
                version=record.version,
                mode=record.mode,
                created_by=record.created_by,
                owner_id=record.owner_id,
                security_scope=record.security_scope,
                shared_with_principals=record.shared_with_principals,
                shared_with_groups=record.shared_with_groups,
                source_ids=record.source_ids,
                derivation_type=record.derivation_type,
                derivation_audit=record.derivation_audit,
                supersedes_version=record.supersedes_version,
                tombstoned=record.tombstoned,
            )
            rebuilt += 1
        return {
            "rebuilt_record_count": rebuilt,
            "truth_graph": truth_graph,
        }

    def decide_acl(
        self,
        *,
        grain: str | None = None,
        truth_graph: str,
        entity_id: str,
        target_item_id: str | None = None,
        principal_id: str,
        principal_groups: Sequence[str] = (),
        security_scope: str | None = None,
    ) -> ACLDecision:
        record = self._latest_persisted_acl_record(
            grain=grain,
            truth_graph=truth_graph,
            entity_id=entity_id,
            target_item_id=target_item_id,
        )
        if record is None:
            return self._e.acl_graph.decide(
                grain=grain,  # type: ignore[arg-type]
                truth_graph=truth_graph,
                entity_id=entity_id,
                target_item_id=target_item_id,
                principal_id=principal_id,
                principal_groups=principal_groups,
                security_scope=security_scope,
            )
        if record.tombstoned:
            return ACLDecision(visible=False, record=record, reason="tombstoned")
        if record.mode == "public":
            return ACLDecision(visible=True, record=record, reason="public")
        if record.owner_id and principal_id == record.owner_id:
            return ACLDecision(visible=True, record=record, reason="owner")
        if record.mode == "private":
            return ACLDecision(visible=False, record=record, reason="private")
        if record.mode == "scope":
            ok = bool(security_scope) and security_scope == record.security_scope
            return ACLDecision(
                visible=ok,
                record=record,
                reason="scope_match" if ok else "scope_mismatch",
            )
        if record.mode == "shared":
            shared = principal_id in record.shared_with_principals
            return ACLDecision(
                visible=shared,
                record=record,
                reason="principal_share" if shared else "not_shared",
            )
        if record.mode == "group":
            groups = set(principal_groups)
            shared_groups = set(record.shared_with_groups)
            ok = bool(groups & shared_groups)
            return ACLDecision(
                visible=ok,
                record=record,
                reason="group_share" if ok else "not_shared",
            )
        return ACLDecision(visible=False, record=record, reason="unknown_mode")

    def decide_acl_usage(
        self,
        *,
        item_grain: str,
        truth_graph: str,
        entity_id: str,
        target_item_id: str,
        principal_id: str,
        principal_groups: Sequence[str] = (),
        security_scope: str | None = None,
    ) -> ACLUsageDecision:
        node_decision = self.decide_acl(
            grain="node",
            truth_graph=truth_graph,
            entity_id=entity_id,
            principal_id=principal_id,
            principal_groups=principal_groups,
            security_scope=security_scope,
        )
        item_decision = self.decide_acl(
            grain=item_grain,
            truth_graph=truth_graph,
            entity_id=entity_id,
            target_item_id=target_item_id,
            principal_id=principal_id,
            principal_groups=principal_groups,
            security_scope=security_scope,
        )
        if not node_decision.visible:
            return ACLUsageDecision(
                visible=False,
                node_decision=node_decision,
                item_decision=item_decision,
                reason=f"node_{node_decision.reason}",
            )
        if not item_decision.visible:
            return ACLUsageDecision(
                visible=False,
                node_decision=node_decision,
                item_decision=item_decision,
                reason=f"{item_grain}_{item_decision.reason}",
            )
        return ACLUsageDecision(
            visible=True,
            node_decision=node_decision,
            item_decision=item_decision,
            reason=f"node_and_{item_grain}_visible",
        )

    def decide_acl_node_read(
        self,
        *,
        item_grain: str,
        truth_graph: str,
        entity_id: str,
        target_item_ids: Sequence[str],
        principal_id: str,
        grounding_item_ids: Sequence[str] = (),
        principal_groups: Sequence[str] = (),
        security_scope: str | None = None,
    ) -> ACLNodeReadDecision:
        node_decision = self.decide_acl(
            grain="node",
            truth_graph=truth_graph,
            entity_id=entity_id,
            principal_id=principal_id,
            principal_groups=principal_groups,
            security_scope=security_scope,
        )
        grounding_decisions = tuple(
            self.decide_acl(
                grain="grounding",
                truth_graph=truth_graph,
                entity_id=entity_id,
                target_item_id=target_item_id,
                principal_id=principal_id,
                principal_groups=principal_groups,
                security_scope=security_scope,
            )
            for target_item_id in grounding_item_ids
        )
        item_decisions = grounding_decisions + tuple(
            self.decide_acl(
                grain=item_grain,
                truth_graph=truth_graph,
                entity_id=entity_id,
                target_item_id=target_item_id,
                principal_id=principal_id,
                principal_groups=principal_groups,
                security_scope=security_scope,
            )
            for target_item_id in target_item_ids
        )
        if not node_decision.visible:
            return ACLNodeReadDecision(
                visible=False,
                node_decision=node_decision,
                item_decisions=item_decisions,
                reason=f"node_{node_decision.reason}",
            )
        for item_decision in grounding_decisions:
            if not item_decision.visible:
                return ACLNodeReadDecision(
                    visible=False,
                    node_decision=node_decision,
                    item_decisions=item_decisions,
                    reason=f"grounding_{item_decision.reason}",
                )
        for item_decision in item_decisions[len(grounding_decisions) :]:
            if not item_decision.visible:
                return ACLNodeReadDecision(
                    visible=False,
                    node_decision=node_decision,
                    item_decisions=item_decisions,
                    reason=f"{item_grain}_{item_decision.reason}",
                )
        return ACLNodeReadDecision(
            visible=True,
            node_decision=node_decision,
            item_decisions=item_decisions,
            reason="node_and_all_items_visible",
        )

    def get_node_acl_checked(
        self,
        node_id: str,
        *,
        grounding_item_ids: Sequence[str] = (),
        target_item_ids: Sequence[str] = (),
        principal_id: str,
        principal_groups: Sequence[str] = (),
        security_scope: str | None = None,
        node_type=None,
        include: None | list[str] = None,
        resolve_mode: Literal["active_only", "redirect", "include_tombstones"] = "active_only",
    ) -> Node:
        nodes = self._e.raw_read.get_nodes(
            ids=[node_id],
            node_type=node_type,
            include=include,
            resolve_mode=resolve_mode,
        )
        if not nodes:
            raise ValueError(f"no node found for id = {node_id}")
        decision = self.decide_acl_node_read(
            item_grain="span",
            truth_graph=self._e.kg_graph_type,
            entity_id=node_id,
            grounding_item_ids=tuple(grounding_item_ids),
            target_item_ids=tuple(target_item_ids),
            principal_id=principal_id,
            principal_groups=tuple(principal_groups),
            security_scope=security_scope,
        )
        if not decision.visible:
            raise PermissionError(f"ACL denied node {node_id}: {decision.reason}")
        return nodes[0]

    def get_edge_acl_checked(
        self,
        edge_id: str,
        *,
        principal_id: str,
        principal_groups: Sequence[str] = (),
        security_scope: str | None = None,
        edge_type=None,
        include: None | list[str] = None,
        resolve_mode: Literal["active_only", "redirect", "include_tombstones"] = "active_only",
    ) -> Any:
        edges = self._e.raw_read.get_edges(
            ids=[edge_id],
            edge_type=edge_type,
            include=include,
            resolve_mode=resolve_mode,
        )
        if not edges:
            raise ValueError(f"no edge found for id = {edge_id}")
        decision = self.decide_acl(
            grain="edge",
            truth_graph=self._e.kg_graph_type,
            entity_id=edge_id,
            principal_id=principal_id,
            principal_groups=tuple(principal_groups),
            security_scope=security_scope,
        )
        if not decision.visible:
            raise PermissionError(f"ACL denied edge {edge_id}: {decision.reason}")
        return edges[0]


class ACLAwareReadSubsystem(NamespaceProxy["GraphKnowledgeEngine"], ReadLike):
    def __init__(self, engine: GraphKnowledgeEngine, raw_read: ReadLike) -> None:
        super().__init__(engine)
        self._raw = raw_read

    def __getattr__(self, name: str) -> object:
        return getattr(self._raw, name)

    def load_node_map(
        self,
        ids: Sequence[str],
        *,
        node_type: type[Node] | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Node]:
        """Load only ACL-visible nodes for the typed map contract."""

        nodes = self._raw.load_node_map(
            ids,
            node_type=node_type,
            include=include,
        )
        return {
            node_id: node
            for node_id, node in nodes.items()
            if self._node_visible(node)
        }

    def load_edge_map(
        self,
        ids: Sequence[str],
        *,
        edge_type: type[Edge] | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Edge]:
        """Load only ACL-visible edges for the typed map contract."""

        edges = self._raw.load_edge_map(
            ids,
            edge_type=edge_type,
            include=include,
        )
        return {
            edge_id: edge
            for edge_id, edge in edges.items()
            if self._edge_visible(edge)
        }

    def _node_visible(self, node: Node) -> bool:
        principal_id, groups, security_scope = self._e.acl.current_principal_context()
        grounding_ids, span_ids = self._e.acl.usage_ids_for_item(node)
        decision = self._e.acl.decide_acl_node_read(
            item_grain="span",
            truth_graph=self._e.kg_graph_type,
            entity_id=str(node.safe_get_id()),
            grounding_item_ids=grounding_ids,
            target_item_ids=span_ids,
            principal_id=principal_id,
            principal_groups=groups,
            security_scope=security_scope,
        )
        if decision.visible:
            return True
        if "no_acl_record" in str(decision.reason or ""):
            repaired = self._e.acl.repair_acl_records_from_events(
                truth_graph=self._e.kg_graph_type,
                entity_id=str(node.safe_get_id()),
                limit=64,
            )
            if repaired.get("repaired_acl_records", 0):
                retry = self._e.acl.decide_acl_node_read(
                    item_grain="span",
                    truth_graph=self._e.kg_graph_type,
                    entity_id=str(node.safe_get_id()),
                    grounding_item_ids=grounding_ids,
                    target_item_ids=span_ids,
                    principal_id=principal_id,
                    principal_groups=groups,
                    security_scope=security_scope,
                )
                return retry.visible
            repaired_truth = self._e.acl.repair_default_acl_for_item(node, grain="node")
            if repaired_truth:
                retry = self._e.acl.decide_acl_node_read(
                    item_grain="span",
                    truth_graph=self._e.kg_graph_type,
                    entity_id=str(node.safe_get_id()),
                    grounding_item_ids=grounding_ids,
                    target_item_ids=span_ids,
                    principal_id=principal_id,
                    principal_groups=groups,
                    security_scope=security_scope,
                )
                return retry.visible
        return False

    def _edge_visible(self, edge: Edge) -> bool:
        principal_id, groups, security_scope = self._e.acl.current_principal_context()
        edge_decision = self._e.acl.decide_acl(
            grain="edge",
            truth_graph=self._e.kg_graph_type,
            entity_id=str(edge.safe_get_id()),
            principal_id=principal_id,
            principal_groups=groups,
            security_scope=security_scope,
        )
        if not edge_decision.visible:
            if "no_acl_record" in str(edge_decision.reason or ""):
                repaired = self._e.acl.repair_acl_records_from_events(
                    truth_graph=self._e.kg_graph_type,
                    entity_id=str(edge.safe_get_id()),
                    limit=64,
                )
                if repaired.get("repaired_acl_records", 0):
                    edge_decision = self._e.acl.decide_acl(
                        grain="edge",
                        truth_graph=self._e.kg_graph_type,
                        entity_id=str(edge.safe_get_id()),
                        principal_id=principal_id,
                        principal_groups=groups,
                        security_scope=security_scope,
                    )
            if not edge_decision.visible:
                return False
        grounding_ids, span_ids = self._e.acl.usage_ids_for_item(edge)
        for grounding_id in grounding_ids:
            decision = self._e.acl.decide_acl(
                grain="grounding",
                truth_graph=self._e.kg_graph_type,
                entity_id=str(edge.safe_get_id()),
                target_item_id=grounding_id,
                principal_id=principal_id,
                principal_groups=groups,
                security_scope=security_scope,
            )
            if not decision.visible:
                return False
        for span_id in span_ids:
            decision = self._e.acl.decide_acl(
                grain="span",
                truth_graph=self._e.kg_graph_type,
                entity_id=str(edge.safe_get_id()),
                target_item_id=span_id,
                principal_id=principal_id,
                principal_groups=groups,
                security_scope=security_scope,
            )
            if not decision.visible:
                return False
        endpoint_ids = tuple(
            str(item)
            for item in tuple(getattr(edge, "source_ids", ()) or ())
            + tuple(getattr(edge, "target_ids", ()) or ())
            if str(item)
        )
        if endpoint_ids:
            endpoint_nodes = self._raw.get_nodes(node_type=Node, ids=list(endpoint_ids))
            for endpoint in endpoint_nodes:
                if not self._node_visible(endpoint):
                    return False
        return True

    def get_nodes(self, *args: Any, **kwargs: Any) -> list[Node]:
        return [node for node in self._raw.get_nodes(*args, **kwargs) if self._node_visible(node)]

    def get_edges(self, *args: Any, **kwargs: Any) -> list[Edge]:
        return [edge for edge in self._raw.get_edges(*args, **kwargs) if self._edge_visible(edge)]

    def get_document(self, doc_id: str) -> Any:
        return self._raw.get_document(doc_id)

    def node_exists(self, *args: Any, **kwargs: Any) -> bool:
        return bool(self._raw.node_exists(*args, **kwargs))

    def edge_exists(self, *args: Any, **kwargs: Any) -> bool:
        return bool(self._raw.edge_exists(*args, **kwargs))

    def get_node_metadatas(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return self._raw.get_node_metadatas(*args, **kwargs)

    def get_edge_metadatas(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return self._raw.get_edge_metadatas(*args, **kwargs)

    def node_ids_by_doc(self, doc_id: str, insertion_method: str | None = None) -> list[str]:
        return self._raw.node_ids_by_doc(doc_id, insertion_method=insertion_method)

    def edge_ids_by_doc(self, doc_id: str, insertion_method: str | None = None) -> list[str]:
        return self._raw.edge_ids_by_doc(doc_id, insertion_method=insertion_method)

    def edges_by_doc(self, doc_id: str, where: dict[str, JsonValue] | None = None) -> list[str]:
        return self._raw.edges_by_doc(doc_id, where=where)

    def list_edges_with_ref_filter(
        self, doc_id: str, where: dict[str, JsonValue] | None = None
    ) -> list[Edge]:
        return self._raw.list_edges_with_ref_filter(doc_id, where=where)

    def nodes_by_doc(
        self, doc_id: str, *, where: dict[str, JsonValue] | None = None
    ) -> list[str]:
        return self._raw.nodes_by_doc(doc_id, where=where)

    def list_nodes_with_ref_filter(
        self, doc_id: str, *, where: dict[str, JsonValue] | None = None
    ) -> list[Node]:
        return self._raw.list_nodes_with_ref_filter(doc_id, where=where)

    def ids_with_insertion_method(
        self,
        *,
        kind: str,
        insertion_method: str,
        ids: Sequence[str] | None = None,
        doc_id: str | None = None,
    ) -> list[str]:
        return self._raw.ids_with_insertion_method(
            kind=kind,
            insertion_method=insertion_method,
            ids=ids,
            doc_id=doc_id,
        )

    def extract_reference_contexts(
        self,
        node_or_id: Node | Edge | str,
        *,
        window_chars: int = 120,
        max_contexts: int | None = None,
        prefer_label_fallback: bool = True,
    ) -> list[dict[str, object]]:
        return self._raw.extract_reference_contexts(
            node_or_id,
            window_chars=window_chars,
            max_contexts=max_contexts,
            prefer_label_fallback=prefer_label_fallback,
        )

    def query_nodes(
        self,
        *args: object,
        query: str | None = None,
        query_embeddings: QueryEmbeddingInput | None = None,
        include: list[str] = ["documents", "embeddings", "metadatas"],
        node_type: type[Node] | None = None,
        **kwargs: object,
    ) -> list[list[Node]]:
        batches = self._raw.query_nodes(
            *args,
            query=query,
            query_embeddings=query_embeddings,
            include=include,
            node_type=node_type,
            **kwargs,
        )
        return [[node for node in batch if self._node_visible(node)] for batch in batches]

    def query_edges(
        self,
        *args: object,
        query: str | None = None,
        query_embeddings: QueryEmbeddingInput | None = None,
        include: list[str] = ["documents", "embeddings", "metadatas"],
        edge_type: type[Edge] | None = None,
        **kwargs: object,
    ) -> list[list[Edge]]:
        batches = self._raw.query_edges(
            *args,
            query=query,
            query_embeddings=query_embeddings,
            include=include,
            edge_type=edge_type,
            **kwargs,
        )
        return [[edge for edge in batch if self._edge_visible(edge)] for batch in batches]

    def search_nodes_as_of(
        self,
        *,
        query: str | None = None,
        query_embeddings: QueryEmbeddingInput | None = None,
        as_of_ts: datetime | str,
        where: dict[str, JsonValue] | None = None,
        n_results: int = 20,
        follow_redirects: bool = True,
        node_type: type[Node] = Node,
        include: list[str] | None = None,
        max_redirect_hops: int = 16,
        **kwargs: object,
    ) -> list[Node]:
        return [
            node
            for node in self._raw.search_nodes_as_of(
                query=query,
                query_embeddings=query_embeddings,
                as_of_ts=as_of_ts,
                where=where,
                n_results=n_results,
                follow_redirects=follow_redirects,
                node_type=node_type,
                include=include,
                max_redirect_hops=max_redirect_hops,
                **kwargs,
            )
            if self._node_visible(node)
        ]

    def search_nodes_as_of_scored(
        self,
        *,
        query: str | None = None,
        query_embeddings: QueryEmbeddingInput | None = None,
        as_of_ts: datetime | str,
        where: dict[str, JsonValue] | None = None,
        n_results: int = 20,
        follow_redirects: bool = True,
        node_type: type[Node] = Node,
        include: list[str] | None = None,
        max_redirect_hops: int = 16,
        similarity_threshold: float | None = None,
        **kwargs: object,
    ) -> list[VectorSearchHit[Node]]:
        return [
            hit
            for hit in self._raw.search_nodes_as_of_scored(
                query=query,
                query_embeddings=query_embeddings,
                as_of_ts=as_of_ts,
                where=where,
                n_results=n_results,
                follow_redirects=follow_redirects,
                node_type=node_type,
                include=include,
                max_redirect_hops=max_redirect_hops,
                similarity_threshold=similarity_threshold,
                **kwargs,
            )
            if self._node_visible(hit.node)
        ]

    def nodes_from_single_or_id_query_result(
        self,
        got: Mapping[str, JsonValue],
        node_type: type[Node] = Node,
    ) -> list[Node]:
        return self._raw.nodes_from_single_or_id_query_result(got, node_type=node_type)

    def edges_from_single_or_id_query_result(
        self,
        got: Mapping[str, JsonValue],
        edge_type: type[Edge] = Edge,
        include: Sequence[str] | None = None,
    ) -> list[Edge]:
        return self._raw.edges_from_single_or_id_query_result(
            got, edge_type=edge_type, include=include
        )

    def nodes_from_query_result(
        self,
        gots: Mapping[str, JsonValue],
        node_type: type[Node] = Node,
    ) -> list[list[Node]]:
        return self._raw.nodes_from_query_result(gots, node_type=node_type)

    def edges_from_query_result(
        self,
        gots: Mapping[str, JsonValue],
        edge_type: type[Edge] = Edge,
    ) -> list[list[Edge]]:
        return self._raw.edges_from_query_result(gots, edge_type=edge_type)

    def where_update_from_resolve_mode(
        self,
        resolve_mode: Literal["active_only", "redirect", "include_tombstones"],
    ) -> dict[str, str]:
        return self._raw.where_update_from_resolve_mode(resolve_mode)


class ACLAwareWriteSubsystem(NamespaceProxy["GraphKnowledgeEngine"], WriteLike):
    def __init__(self, engine: GraphKnowledgeEngine, raw_write: WriteLike) -> None:
        super().__init__(engine)
        self._raw = raw_write

    def __getattr__(self, name: str) -> object:
        return getattr(self._raw, name)

    def add_document(self, document: Document) -> None:
        self._raw.add_document(document)

    async def add_node_async(self, node: Node, doc_id: str | None = None) -> None:
        await self._raw.add_node_async(node, doc_id=doc_id)

    async def add_edge_async(self, edge: Edge, doc_id: str | None = None) -> None:
        await self._raw.add_edge_async(edge, doc_id=doc_id)

    def add_pure_node(self, node: PureChromaNode) -> None:
        self._raw.add_pure_node(node)

    def add_pure_edge(self, edge: PureChromaEdge) -> None:
        self._raw.add_pure_edge(edge)

    def add_domain(self, domain: Domain) -> None:
        self._raw.add_domain(domain)

    def enrich_edge_meta(self, edge: Edge) -> dict[str, object]:
        return self._raw.enrich_edge_meta(edge)

    def fanout_endpoints_rows(self, edge: Edge, doc_id: str | None) -> object:
        return self._raw.fanout_endpoints_rows(edge, doc_id)

    def maybe_reindex_edge_refs(self, edge: Edge, *, force: bool = False) -> None:
        self._raw.maybe_reindex_edge_refs(edge, force=force)

    def maybe_reindex_node_refs(self, node: Node, *, force: bool = False) -> None:
        self._raw.maybe_reindex_node_refs(node, force=force)

    def prune_node_refs_for_doc(self, node_id: str, doc_id: str) -> bool:
        return self._raw.prune_node_refs_for_doc(node_id, doc_id)

    def rebuild_edge_refs_for_doc(self, doc_id: str) -> int:
        return self._raw.rebuild_edge_refs_for_doc(doc_id)

    def rebuild_all_edge_refs(self) -> int:
        return self._raw.rebuild_all_edge_refs()

    def rebuild_node_refs_for_doc(self, doc_id: str) -> int:
        return self._raw.rebuild_node_refs_for_doc(doc_id)

    def rebuild_all_node_refs(self) -> int:
        return self._raw.rebuild_all_node_refs()

    def delete_edges_by_ids(self, edge_ids: list[str]) -> None:
        self._raw.delete_edges_by_ids(edge_ids)

    def rust_postgres_delete_existing(
        self, *, entity_kind: str, entity_ids: list[str]
    ) -> bool:
        return self._raw.rust_postgres_delete_existing(
            entity_kind=entity_kind, entity_ids=entity_ids
        )

    def rust_postgres_replace_existing(
        self,
        *,
        entity_kind: str,
        entity_id: str,
        document: str,
        metadata_patch: JsonObject,
        payload: JsonObject,
    ) -> bool:
        return self._raw.rust_postgres_replace_existing(
            entity_kind=entity_kind,
            entity_id=entity_id,
            document=document,
            metadata_patch=metadata_patch,
            payload=payload,
        )

    def uses_rust_postgres_authority(self) -> bool:
        return self._raw.uses_rust_postgres_authority()

    def add_node(self, node: Node, doc_id: str | None = None) -> None:
        if not self._e.acl.writes_can_share_backend_transaction():
            with self._e.uow():
                self._e.acl.append_canonical_write_events_for_item(node, grain="node")
            prev_log = getattr(self._e, "_disable_event_log", False)
            self._e._disable_event_log = True
            try:
                result = self._raw.add_node(node, doc_id=doc_id)
                self._e.acl.record_default_acl_for_item(node, grain="node")
                return result
            finally:
                self._e._disable_event_log = prev_log
        with self._e.uow():
            result = self._raw.add_node(node, doc_id=doc_id)
            self._e.acl.record_default_acl_for_item(node, grain="node")
            return result

    def add_edge(self, edge: Edge, doc_id: str | None = None) -> None:
        if not self._e.acl.writes_can_share_backend_transaction():
            with self._e.uow():
                self._e.acl.append_canonical_write_events_for_item(edge, grain="edge")
            prev_log = getattr(self._e, "_disable_event_log", False)
            self._e._disable_event_log = True
            try:
                result = self._raw.add_edge(edge, doc_id=doc_id)
                self._e.acl.record_default_acl_for_item(edge, grain="edge")
                return result
            finally:
                self._e._disable_event_log = prev_log
        with self._e.uow():
            result = self._raw.add_edge(edge, doc_id=doc_id)
            self._e.acl.record_default_acl_for_item(edge, grain="edge")
            return result

    def node_doc_and_meta(self, node: Node) -> tuple[str, dict[str, Any]]:
        return self._raw.node_doc_and_meta(node)

    def edge_doc_and_meta(self, edge: Edge) -> tuple[str, dict[str, Any]]:
        return self._raw.edge_doc_and_meta(edge)

    def strip_none(self, data: dict[str, Any]) -> dict[str, Any]:
        return self._raw.strip_none(data)

    def json_or_none(self, value: Any) -> str | None:
        return self._raw.json_or_none(value)

    def index_node_docs(self, node: Node) -> list[str]:
        return self._raw.index_node_docs(node)

    def index_node_refs(self, node: Node) -> list[str]:
        return self._raw.index_node_refs(node)

    def index_edge_refs(self, edge: Edge) -> list[str]:
        return self._raw.index_edge_refs(edge)

    def delete_edge_ref_rows(self, edge_id: str) -> None:
        self._raw.delete_edge_ref_rows(edge_id)

    def delete_node_ref_rows(self, node_id: str) -> None:
        self._raw.delete_node_ref_rows(node_id)
