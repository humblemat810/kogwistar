"""Read canonical entity revision from the existing append-only event stream."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class CanonicalEntityRevision:
    """Latest canonical event-derived state for one entity identity."""

    namespace: str
    entity_kind: str
    entity_id: str
    revision: int
    state: str
    payload: dict[str, Any]

    @property
    def is_deleted(self) -> bool:
        return self.state == "deleted"


def read_canonical_entity_revision(
    *,
    events: Iterable[tuple[int, str, str, str, str]],
    namespace: str,
    entity_kind: str,
    entity_id: str,
) -> CanonicalEntityRevision | None:
    """Fold one entity's events without treating backend rows as authority.

    This helper intentionally accepts the existing ``iter_entity_events`` row
    shape. It is a read seam for revision gates and repair planning, not a new
    materialized state store. Unknown operations or malformed payloads fail
    closed because accepting them could let stale derived work become visible.
    """

    latest: CanonicalEntityRevision | None = None
    for seq, row_kind, row_id, op, payload_json in events:
        if str(row_kind) != str(entity_kind) or str(row_id) != str(entity_id):
            continue
        try:
            payload = json.loads(payload_json)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid canonical event payload for {entity_kind}:{entity_id} at {seq}"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError(
                f"canonical event payload must be an object for {entity_kind}:{entity_id} at {seq}"
            )

        normalized_op = str(op).upper()
        if normalized_op in {"ADD", "REPLACE"}:
            patch = payload.get("lifecycle_patch")
            tombstoned = isinstance(patch, dict) and patch.get("lifecycle_status") == "tombstoned"
            state = "deleted" if tombstoned else "active"
        elif normalized_op in {"TOMBSTONE", "DELETE"}:
            state = "deleted"
        else:
            raise ValueError(
                f"unsupported canonical event op {op!r} for {entity_kind}:{entity_id}"
            )
        latest = CanonicalEntityRevision(
            namespace=str(namespace),
            entity_kind=str(entity_kind),
            entity_id=str(entity_id),
            revision=int(seq),
            state=state,
            payload=payload,
        )
    return latest
