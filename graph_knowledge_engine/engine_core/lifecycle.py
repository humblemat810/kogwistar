from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Literal, Sequence, TypeVar, TYPE_CHECKING
from .subsystems.base import NamespaceProxy
if TYPE_CHECKING:
    # Avoid runtime import cycles; we only need this for typing.
    from .engine import GraphKnowledgeEngine
T = TypeVar("T")  # Node/Edge-like


class LifecycleSubsystem(NamespaceProxy):
    """Lifecycle policy for Nodes/Edges.

    Owns:
    - soft-delete (tombstone) and redirect lifecycle patches
    - redirect-chain resolution for read paths
    - resolve_mode filtering

    Notes:
    - Keeps side-effects best-effort to match existing engine behavior.
    """

    def __init__(self, engine: GraphKnowledgeEngine) -> None:
        self._e = engine

    def _utc_now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # -----------------------
    # Read-side
    # -----------------------

    def filter_items(self, items: list[T], resolve_mode: str) -> list[T]:
        if resolve_mode == "include_tombstones":
            return items
        if resolve_mode in ("active_only", "redirect"):
            return [
                x
                for x in items
                if ((getattr(x, "metadata", {}) or {}).get("lifecycle_status") or "active") == "active"
            ]
        return items

    def resolve_redirect_chain(
        self,
        *,
        initial_items: list[T],
        fetch_by_ids: Callable[[Sequence[str]], list[T]],
        resolve_mode: Literal["active_only", "redirect", "include_tombstones"],
    ) -> list[T]:
        """Resolve redirect chains.

        Fix: do not enqueue str(None) into the frontier when tombstoned items have
        no redirect target.
        """
        if resolve_mode != "redirect":
            return initial_items

        resolved: dict[str, T] = {}
        visited: set[str] = set()
        frontier: list[T] = list(initial_items)

        while frontier:
            next_frontier_ids: set[str] = set()

            for item in frontier:
                item_id = str(getattr(item, "id"))
                if item_id in visited:
                    continue
                visited.add(item_id)

                meta = getattr(item, "metadata", {}) or {}
                status = meta.get("lifecycle_status")
                redirect_to = meta.get("redirect_to_id")

                if redirect_to:
                    next_frontier_ids.add(str(redirect_to))
                    continue

                # tombstoned without redirect is terminal -> drop
                if status == "tombstoned":
                    continue

                resolved[item_id] = item

            if not next_frontier_ids:
                break

            frontier = fetch_by_ids(list(next_frontier_ids))

        return self.filter_items(list(resolved.values()), "redirect")

    # -----------------------
    # Write-side
    # -----------------------

    def tombstone_node(self, node_id: str, **kw) -> bool:
        patch = {
            "lifecycle_status": "tombstoned",
            "redirect_to_id": None,
            "deleted_at": kw.get("deleted_at") or self._utc_now_iso(),
        }
        if kw.get("reason"):
            patch["delete_reason"] = kw["reason"]
        if kw.get("deleted_by"):
            patch["deleted_by"] = kw["deleted_by"]

        ok = self._e._backend_update_record_lifecycle(
            backend=self._e.backend, kind="node", record_id=node_id, lifecycle_patch=patch
        )

        if ok:
            self._best_effort_event(entity_kind="node", entity_id=node_id, op="TOMBSTONE", **kw)
            self._maybe_index_delete(entity_kind="node", entity_id=node_id)

        return ok

    def redirect_node(self, from_id: str, to_id: str, **kw) -> bool:
        if from_id == to_id:
            return False

        patch = {
            "lifecycle_status": "tombstoned",
            "redirect_to_id": str(to_id),
            "deleted_at": kw.get("deleted_at") or self._utc_now_iso(),
        }
        if kw.get("reason"):
            patch["delete_reason"] = kw["reason"]
        if kw.get("deleted_by"):
            patch["deleted_by"] = kw["deleted_by"]

        ok = self._e._backend_update_record_lifecycle(
            backend=self._e.backend, kind="node", record_id=from_id, lifecycle_patch=patch
        )
        if ok:
            self._maybe_index_delete(entity_kind="node", entity_id=from_id)
        return ok

    def tombstone_edge(self, edge_id: str, **kw) -> bool:
        patch = {
            "lifecycle_status": "tombstoned",
            "redirect_to_id": None,
            "deleted_at": kw.get("deleted_at") or self._utc_now_iso(),
        }
        if kw.get("reason"):
            patch["delete_reason"] = kw["reason"]
        if kw.get("deleted_by"):
            patch["deleted_by"] = kw["deleted_by"]

        ok = self._e._backend_update_record_lifecycle(
            backend=self._e.backend, kind="edge", record_id=edge_id, lifecycle_patch=patch
        )

        if ok:
            self._best_effort_event(entity_kind="edge", entity_id=edge_id, op="TOMBSTONE", **kw)
            self._maybe_index_delete(entity_kind="edge", entity_id=edge_id)

        return ok

    def redirect_edge(self, from_id: str, to_id: str, **kw) -> bool:
        if from_id == to_id:
            return False

        patch = {
            "lifecycle_status": "tombstoned",
            "redirect_to_id": str(to_id),
            "deleted_at": kw.get("deleted_at") or self._utc_now_iso(),
        }
        if kw.get("reason"):
            patch["delete_reason"] = kw["reason"]
        if kw.get("deleted_by"):
            patch["deleted_by"] = kw["deleted_by"]

        ok = self._e._backend_update_record_lifecycle(
            backend=self._e.backend, kind="edge", record_id=from_id, lifecycle_patch=patch
        )
        if ok:
            self._maybe_index_delete(entity_kind="edge", entity_id=from_id)
        return ok

    # -----------------------
    # Internals
    # -----------------------

    def _best_effort_event(self, *, entity_kind: str, entity_id: str, op: str, **kw) -> None:
        try:
            payload = {"entity_id": entity_id}
            if kw.get("reason") is not None:
                payload["reason"] = kw.get("reason")
            if kw.get("deleted_by") is not None:
                payload["deleted_by"] = kw.get("deleted_by")
            self._e._append_event_for_entity(
                namespace=getattr(self._e, "namespace", "default"),
                entity_kind=entity_kind,
                entity_id=entity_id,
                op=op,
                payload=payload,
            )
        except Exception:
            # never block primary write path
            pass

    def _maybe_index_delete(self, *, entity_kind: str, entity_id: str) -> None:
        if not getattr(self._e, "_phase1_enable_index_jobs", False):
            return
        try:
            if entity_kind == "node":
                self._e.enqueue_index_jobs_for_node(entity_id, op="DELETE")
            else:
                self._e.enqueue_index_jobs_for_edge(entity_id, op="DELETE")
            self._e.reconcile_indexes(max_jobs=50)
        except Exception:
            pass