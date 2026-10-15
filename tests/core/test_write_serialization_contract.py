from __future__ import annotations

from typing import Any

import pytest

from kogwistar.engine_core.subsystems.write import _entity_payload

pytestmark = [pytest.mark.ci, pytest.mark.core]


def test_entity_payload_uses_backend_json_dump_mode() -> None:
    calls: list[dict[str, Any]] = []

    class _ModeAwareModel:
        def model_dump(self, **kwargs: Any) -> dict[str, object]:
            calls.append(kwargs)
            return {"id": "workflow-node", "mode": kwargs["field_mode"]}

    payload = _entity_payload(_ModeAwareModel())

    assert payload == {"id": "workflow-node", "mode": "backend"}
    assert calls == [
        {
            "field_mode": "backend",
            "dump_format": "json",
            "exclude": ["embedding"],
        }
    ]
