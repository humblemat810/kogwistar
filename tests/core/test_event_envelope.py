from __future__ import annotations

import pytest

from kogwistar.engine_core.engine_sqlite import EngineSQLite
from kogwistar.engine_core.event_envelope import EntityEventEnvelope


pytestmark = [pytest.mark.ci, pytest.mark.core]


def test_sqlite_event_envelope_round_trip_preserves_identity_and_timestamp(tmp_path) -> None:
    meta = EngineSQLite(tmp_path)
    meta.ensure_initialized()
    event = EntityEventEnvelope(
        namespace="archive/source",
        seq=1,
        event_id="event-1",
        entity_kind="source",
        entity_id="source-1",
        op="UPSERT",
        payload_json='{"title":"雪"}',
        created_at=123456789,
    )

    assert meta.append_entity_event_envelope(event) == 1
    assert meta.append_entity_event_envelope(event) == 1
    assert list(meta.iter_entity_event_envelopes(namespace=event.namespace)) == [event]

    conflicting = EntityEventEnvelope(
        namespace=event.namespace,
        seq=1,
        event_id=event.event_id,
        entity_kind=event.entity_kind,
        entity_id=event.entity_id,
        op=event.op,
        payload_json='{"title":"different"}',
        created_at=event.created_at,
    )
    with pytest.raises(ValueError, match="conflicts"):
        meta.append_entity_event_envelope(conflicting)
