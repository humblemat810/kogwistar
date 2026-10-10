"""Lossless event envelopes used by portable archive tooling."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass


def _as_int(value: object) -> int:
    if isinstance(value, (int, float, str)):
        return int(value)
    raise ValueError(f"expected an integer-compatible value, got {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class EntityEventEnvelope:
    """The immutable, portable embedding of one authoritative event."""

    namespace: str
    seq: int
    event_id: str
    entity_kind: str
    entity_id: str
    op: str
    payload_json: str
    created_at: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> EntityEventEnvelope:
        return cls(
            namespace=str(value["namespace"]),
            seq=_as_int(value["seq"]),
            event_id=str(value["event_id"]),
            entity_kind=str(value["entity_kind"]),
            entity_id=str(value["entity_id"]),
            op=str(value["op"]),
            payload_json=str(value["payload_json"]),
            created_at=_as_int(value["created_at"]),
        )
