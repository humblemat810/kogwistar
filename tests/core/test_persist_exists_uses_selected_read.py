from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from kogwistar.engine_core.subsystems.persist import PersistSubsystem


@pytest.mark.parametrize(
    ("entity_kind", "method_name"),
    [("node", "exists_node"), ("edge", "exists_edge")],
)
def test_persist_exists_uses_selected_read_path_and_includes_tombstones(
    entity_kind: str, method_name: str
) -> None:
    calls: list[dict[str, Any]] = []

    class ReadStub:
        def get_nodes(self, **kwargs: Any) -> list[object]:
            calls.append(kwargs)
            return [object()]

        def get_edges(self, **kwargs: Any) -> list[object]:
            calls.append(kwargs)
            return [object()]

    def forbidden_backend_read(**_kwargs: Any) -> None:
        raise AssertionError("persistence existence check bypassed selected read path")

    backend = SimpleNamespace(
        node_get=forbidden_backend_read,
        edge_get=forbidden_backend_read,
    )
    subsystem = PersistSubsystem(SimpleNamespace(read=ReadStub(), backend=backend))  # type: ignore[arg-type]

    assert getattr(subsystem, method_name)("entity-1") is True
    assert calls == [
        {
            "ids": ["entity-1"],
            "limit": 1,
            "resolve_mode": "include_tombstones",
        }
    ]
