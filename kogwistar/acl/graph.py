from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal


ACLMode = Literal["private", "shared", "scope", "group", "public"]
ACLGrain = Literal["document", "grounding", "span", "node", "edge", "artifact"]


@dataclass(frozen=True, slots=True)
class ACLTarget:
    truth_graph: str
    entity_id: str
    grain: ACLGrain = "node"
    target_item_id: str | None = None


@dataclass(frozen=True, slots=True)
class ACLRecord:
    target: ACLTarget
    version: int
    mode: ACLMode
    created_by: str | None = None
    owner_id: str | None = None
    security_scope: str | None = None
    shared_with_principals: tuple[str, ...] = ()
    shared_with_groups: tuple[str, ...] = ()
    tombstoned: bool = False
    supersedes_version: int | None = None


@dataclass(frozen=True, slots=True)
class ACLDecision:
    visible: bool
    record: ACLRecord | None = None
    reason: str = "no_acl"


@dataclass(frozen=True, slots=True)
class ACLUsageDecision:
    visible: bool
    node_decision: ACLDecision
    item_decision: ACLDecision
    reason: str = "no_acl"


@dataclass(frozen=True, slots=True)
class ACLNodeReadDecision:
    visible: bool
    node_decision: ACLDecision
    item_decisions: tuple[ACLDecision, ...]
    reason: str = "no_acl"


def _acl_rank(mode: ACLMode) -> int:
    return {"public": 0, "group": 1, "shared": 2, "scope": 3, "private": 4}.get(mode, 99)


class ACLGraph:
    """Versioned ACL overlay for any truth graph."""

    def __init__(self) -> None:
        self._records: list[ACLRecord] = []
        self._target_index: dict[tuple[str, ACLGrain, str], set[str]] = {}

    def add_record(
        self,
        *,
        grain: ACLGrain = "node",
        truth_graph: str,
        entity_id: str,
        target_item_id: str | None = None,
        version: int,
        mode: ACLMode,
        created_by: str | None = None,
        owner_id: str | None = None,
        security_scope: str | None = None,
        shared_with_principals: Iterable[str] = (),
        shared_with_groups: Iterable[str] = (),
        supersedes_version: int | None = None,
        tombstoned: bool = False,
    ) -> ACLRecord:
        record = ACLRecord(
            target=ACLTarget(
                grain=grain,
                truth_graph=truth_graph,
                entity_id=entity_id,
                target_item_id=target_item_id,
            ),
            version=version,
            mode=mode,
            created_by=created_by,
            owner_id=owner_id,
            security_scope=security_scope,
            shared_with_principals=tuple(shared_with_principals),
            shared_with_groups=tuple(shared_with_groups),
            supersedes_version=supersedes_version,
            tombstoned=tombstoned,
        )
        self._records.append(record)
        if target_item_id is not None:
            key = (truth_graph, grain, target_item_id)
            self._target_index.setdefault(key, set()).add(entity_id)
        return record

    def entity_ids_for_target_item(
        self,
        *,
        truth_graph: str,
        grain: ACLGrain,
        target_item_id: str,
    ) -> tuple[str, ...]:
        return tuple(
            sorted(
                self._target_index.get((truth_graph, grain, target_item_id), set())
            )
        )

    def record_targets_for_item(
        self,
        *,
        truth_graph: str,
        grain: ACLGrain,
        target_item_id: str,
    ) -> tuple[ACLRecord, ...]:
        return tuple(
            r
            for r in self._records
            if r.target.truth_graph == truth_graph
            and r.target.grain == grain
            and r.target.target_item_id == target_item_id
        )

    def latest_record(
        self,
        *,
        grain: ACLGrain | None = None,
        truth_graph: str,
        entity_id: str,
        target_item_id: str | None = None,
    ) -> ACLRecord | None:
        records = [
            r
            for r in self._records
            if r.target.truth_graph == truth_graph
            and r.target.entity_id == entity_id
            and (grain is None or r.target.grain == grain)
            and (target_item_id is None or r.target.target_item_id == target_item_id)
        ]
        if not records:
            return None
        active = [r for r in records if not r.tombstoned]
        pool = active or records
        return max(pool, key=lambda r: (r.version, _acl_rank(r.mode)))

    def decide(
        self,
        *,
        grain: ACLGrain | None = None,
        truth_graph: str,
        entity_id: str,
        target_item_id: str | None = None,
        principal_id: str,
        principal_groups: Iterable[str] = (),
        security_scope: str | None = None,
    ) -> ACLDecision:
        record = self.latest_record(
            grain=grain,
            truth_graph=truth_graph,
            entity_id=entity_id,
            target_item_id=target_item_id,
        )
        if record is None:
            return ACLDecision(visible=False, record=None, reason="no_acl_record")
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
            return ACLDecision(visible=ok, record=record, reason="scope_match" if ok else "scope_mismatch")
        if record.mode == "shared":
            shared = principal_id in record.shared_with_principals
            return ACLDecision(visible=shared, record=record, reason="principal_share" if shared else "not_shared")
        if record.mode == "group":
            groups = set(principal_groups)
            shared_groups = set(record.shared_with_groups)
            ok = bool(groups & shared_groups)
            return ACLDecision(visible=ok, record=record, reason="group_share" if ok else "not_shared")
        return ACLDecision(visible=False, record=record, reason="unknown_mode")

    def decide_node_item_usage(
        self,
        *,
        item_grain: ACLGrain,
        truth_graph: str,
        entity_id: str,
        target_item_id: str,
        principal_id: str,
        principal_groups: Iterable[str] = (),
        security_scope: str | None = None,
    ) -> ACLUsageDecision:
        node_decision = self.decide(
            grain="node",
            truth_graph=truth_graph,
            entity_id=entity_id,
            principal_id=principal_id,
            principal_groups=principal_groups,
            security_scope=security_scope,
        )
        item_decision = self.decide(
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

    def decide_node_read(
        self,
        *,
        item_grain: ACLGrain,
        truth_graph: str,
        entity_id: str,
        target_item_ids: Iterable[str],
        principal_id: str,
        grounding_item_ids: Iterable[str] = (),
        principal_groups: Iterable[str] = (),
        security_scope: str | None = None,
    ) -> ACLNodeReadDecision:
        node_decision = self.decide(
            grain="node",
            truth_graph=truth_graph,
            entity_id=entity_id,
            principal_id=principal_id,
            principal_groups=principal_groups,
            security_scope=security_scope,
        )
        grounding_decisions = tuple(
            self.decide(
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
            self.decide(
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
        for item_decision in item_decisions[len(grounding_decisions):]:
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

    def strictest_source_mode(
        self,
        records: Iterable[ACLRecord],
    ) -> ACLMode:
        pool = list(records)
        if not pool:
            return "private"
        return max(pool, key=lambda r: _acl_rank(r.mode)).mode
