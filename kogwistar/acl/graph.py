from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from threading import RLock
from typing import Callable, Iterable, Literal


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
    """Versioned ACL overlay for any truth graph.

    This object is a rebuildable in-memory view over persisted ACL truth.
    Canonical ACL state lives in engine-backed ACL record nodes and edges.
    """

    def __init__(
        self,
        *,
        max_record_cache_size: int = 1024,
        max_target_cache_size: int = 1024,
        cache_enabled: bool = True,
    ) -> None:
        """Create ACLGraph cache/projection with bounded record and target caches."""
        self._records: OrderedDict[
            tuple[str, ACLGrain | None, str, str | None],
            tuple[ACLRecord, ...],
        ] = OrderedDict()
        self._target_index: OrderedDict[
            tuple[str, ACLGrain, str],
            tuple[str, ...],
        ] = OrderedDict()
        self._record_loader: Callable[
            [str, ACLGrain | None, str, str | None],
            Iterable[ACLRecord],
        ] | None = None
        self._target_loader: Callable[[str, ACLGrain, str], Iterable[str]] | None = None
        self._max_record_cache_size = max(1, int(max_record_cache_size))
        self._max_target_cache_size = max(1, int(max_target_cache_size))
        self.cache_enabled = bool(cache_enabled)
        self._lock = RLock()

    def bind_loaders(
        self,
        *,
        record_loader: Callable[
            [str, ACLGrain | None, str, str | None],
            Iterable[ACLRecord],
        ] | None = None,
        target_loader: Callable[[str, ACLGrain, str], Iterable[str]] | None = None,
    ) -> None:
        """Bind canonical truth loaders for record lookup and reverse target lookup."""
        # record_loader:
        #   query ACL versions for one concrete ACL target.
        #   Example: (truth_graph="knowledge", grain="span", entity_id="node-1",
        #   target_item_id="sp:1") -> ACLRecord(version=1, mode="private", ...)
        #   This answers: "What ACL state belongs to node-1's span usage sp:1?"
        #
        # target_loader:
        #   reverse lookup owning entity ids for one target item key.
        #   Example: (truth_graph="knowledge", grain="span", target_item_id="sp:1")
        #   -> ("node-1", "node-7")
        #   This answers: "Which truth entities claim ACL over span usage sp:1?"
        self._record_loader = record_loader
        self._target_loader = target_loader

    def clear(self) -> None:
        """Drop all cached ACL records and target indexes."""
        with self._lock:
            self._records.clear()
            self._target_index.clear()

    def invalidate(
        self,
        *,
        truth_graph: str | None = None,
        grain: ACLGrain = "node",
        entity_id: str | None = None,
        target_item_id: str | None = None,
    ) -> None:
        """Invalidate cached entries for one entity or for the full ACL graph."""
        with self._lock:
            if truth_graph is None:
                self._records.clear()
                self._target_index.clear()
                return
            truth_graph = str(truth_graph)
            if entity_id is not None:
                entity_id = str(entity_id)
                record_keys = [
                    key
                    for key in self._records
                    if key[0] == truth_graph
                    and key[2] == entity_id
                ]
                for key in record_keys:
                    self._records.pop(key, None)
                target_keys = [
                    key for key in self._target_index if key[0] == truth_graph
                ]
                for key in target_keys:
                    self._target_index.pop(key, None)
            if target_item_id is not None:
                self._target_index.pop((truth_graph, grain, str(target_item_id)), None)

    def _store_record_cache(
        self, key: tuple[str, ACLGrain | None, str, str | None], records: Iterable[ACLRecord]
    ) -> tuple[ACLRecord, ...]:
        """Normalize record lookup, then cache it on hit and return the same value."""
        value = tuple(records)
        if not self.cache_enabled:
            return value
        with self._lock:
            self._records[key] = value
            self._records.move_to_end(key)
            while len(self._records) > self._max_record_cache_size:
                self._records.popitem(last=False)
        return value

    def _store_target_cache(
        self, key: tuple[str, ACLGrain, str], entity_ids: Iterable[str]
    ) -> tuple[str, ...]:
        """Normalize reverse lookup, then cache it on hit and return the same value."""
        value = tuple(sorted({str(item) for item in entity_ids if str(item)}))
        if not self.cache_enabled:
            return value
        with self._lock:
            self._target_index[key] = value
            self._target_index.move_to_end(key)
            while len(self._target_index) > self._max_target_cache_size:
                self._target_index.popitem(last=False)
        return value

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
        """Append one ACLRecord into in-memory projection cache."""
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
        key = (truth_graph, grain, entity_id, target_item_id)
        if not self.cache_enabled:
            return record
        with self._lock:
            cached = list(self._records.get(key, ()))
        cached.append(record)
        self._store_record_cache(key, cached)
        if target_item_id is not None:
            target_key = (truth_graph, grain, target_item_id)
            with self._lock:
                current = list(self._target_index.get(target_key, ()))
            current.append(entity_id)
            self._store_target_cache(target_key, current)
        return record

    def _load_records(
        self,
        *,
        truth_graph: str,
        grain: ACLGrain | None,
        entity_id: str,
        target_item_id: str | None,
    ) -> tuple[ACLRecord, ...]:
        """Load ACLRecord versions for one concrete ACL target, using cache on hit."""
        # Cache key is one ACL target lookup.
        key = (truth_graph, grain, entity_id, target_item_id)
        with self._lock:
            cached = self._records.get(key) if self.cache_enabled else None
            if cached is not None:
                self._records.move_to_end(key)
                return cached
            loader = self._record_loader
        if loader is None:
            return ()
        records = tuple(
            loader(
                truth_graph=str(truth_graph),
                grain=grain,
                entity_id=str(entity_id),
                target_item_id=target_item_id,
            )
        )
        with self._lock:
            cached = self._records.get(key) if self.cache_enabled else None
            if cached is not None:
                self._records.move_to_end(key)
                return cached
            return self._store_record_cache(key, records)

    def _load_entity_ids(
        self,
        *,
        truth_graph: str,
        grain: ACLGrain,
        target_item_id: str,
    ) -> tuple[str, ...]:
        """Load owning entity ids for one target item key, using cache on hit."""
        # Reverse lookup: same target item may map to many owning entities.
        key = (truth_graph, grain, target_item_id)
        with self._lock:
            cached = self._target_index.get(key) if self.cache_enabled else None
            if cached is not None:
                self._target_index.move_to_end(key)
                return cached
            loader = self._target_loader
        if loader is None:
            return ()
        ids = tuple(
            loader(
                truth_graph=str(truth_graph),
                grain=grain,
                target_item_id=str(target_item_id),
            )
        )
        with self._lock:
            cached = self._target_index.get(key) if self.cache_enabled else None
            if cached is not None:
                self._target_index.move_to_end(key)
                return cached
            return self._store_target_cache(key, ids)

    def entity_ids_for_target_item(
        self,
        *,
        truth_graph: str,
        grain: ACLGrain,
        target_item_id: str,
    ) -> tuple[str, ...]:
        """Return owning entity ids for one target item key."""
        return self._load_entity_ids(
            truth_graph=truth_graph, grain=grain, target_item_id=target_item_id
        )

    def record_targets_for_item(
        self,
        *,
        truth_graph: str,
        grain: ACLGrain,
        target_item_id: str,
    ) -> tuple[ACLRecord, ...]:
        """Return ACLRecords that claim one target item key."""
        loaded: list[ACLRecord] = []
        for entity_id in self.entity_ids_for_target_item(
            truth_graph=truth_graph,
            grain=grain,
            target_item_id=target_item_id,
        ):
            loaded.extend(
                self._load_records(
                    truth_graph=truth_graph,
                    grain=grain,
                    entity_id=entity_id,
                    target_item_id=target_item_id,
                )
            )
        return tuple(
            r
            for r in loaded
            if r.target.truth_graph == truth_graph
            and r.target.grain == grain
            and r.target.target_item_id == target_item_id
        )

    def prefetch_target_items(
        self,
        *,
        truth_graph: str,
        grain: ACLGrain,
        target_item_ids: Iterable[str],
    ) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """Warm reverse index for a bounded set of target item ids."""
        warmed: list[tuple[str, tuple[str, ...]]] = []
        for target_item_id in target_item_ids:
            ids = self.entity_ids_for_target_item(
                truth_graph=truth_graph,
                grain=grain,
                target_item_id=str(target_item_id),
            )
            warmed.append((str(target_item_id), ids))
        return tuple(warmed)

    def iter_records(self) -> tuple[ACLRecord, ...]:
        """Return all cached ACLRecords currently held in memory."""
        with self._lock:
            out: list[ACLRecord] = []
            for records in self._records.values():
                out.extend(records)
        return tuple(out)

    def latest_record(
        self,
        *,
        grain: ACLGrain | None = None,
        truth_graph: str,
        entity_id: str,
        target_item_id: str | None = None,
    ) -> ACLRecord | None:
        """Return newest non-tombstoned ACLRecord for one ACL target."""
        records = [
            r
            for r in self._load_records(
                truth_graph=truth_graph,
                grain=grain,
                entity_id=entity_id,
                target_item_id=target_item_id,
            )
            if r.target.truth_graph == truth_graph
            and r.target.entity_id == entity_id
            and (grain is None or r.target.grain == grain)
            and (target_item_id is None or r.target.target_item_id == target_item_id)
        ]
        if not records and grain is None:
            records = [
                r
                for r in self.iter_records()
                if r.target.truth_graph == truth_graph
                and r.target.entity_id == entity_id
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
        """Evaluate one ACL target against principal, groups, and scope."""
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
        """Evaluate node ACL plus one specific item usage ACL together."""
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
        """Require node ACL and every grounding/span usage ACL to pass."""
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
        """Pick most restrictive mode across source records."""
        pool = list(records)
        if not pool:
            return "private"
        return max(pool, key=lambda r: _acl_rank(r.mode)).mode
