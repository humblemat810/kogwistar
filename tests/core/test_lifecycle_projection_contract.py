from __future__ import annotations

import pytest

from kogwistar.engine_core.lifecycle_projection import (
    backend_update_record_lifecycle,
    decode_lifecycle_get_result,
)


class _Backend:
    def __init__(self, response: object) -> None:
        self.response = response
        self.get_calls: list[tuple[str, dict[str, object]]] = []
        self.update_calls: list[tuple[str, dict[str, object]]] = []

    def node_get(self, **kwargs: object) -> object:
        self.get_calls.append(("node", kwargs))
        return self.response

    def node_update(self, **kwargs: object) -> object:
        self.update_calls.append(("node", kwargs))
        return None

    def edge_get(self, **kwargs: object) -> object:
        self.get_calls.append(("edge", kwargs))
        return self.response

    def edge_update(self, **kwargs: object) -> object:
        self.update_calls.append(("edge", kwargs))
        return None


@pytest.mark.ci
def test_lifecycle_projection_updates_only_metadata_and_preserves_existing_fields() -> None:
    backend = _Backend(
        {
            "ids": ["node-1"],
            "metadatas": [{"acl": "readers", "source_revision": 3}],
        }
    )

    assert backend_update_record_lifecycle(
        backend=backend,
        kind="node",
        record_id="node-1",
        lifecycle_patch={"lifecycle_status": "tombstoned"},
    )

    assert backend.get_calls == [
        ("node", {"ids": ["node-1"], "include": ["metadatas"]})
    ]
    assert backend.update_calls == [
        (
            "node",
            {
                "ids": ["node-1"],
                "metadatas": [
                    {
                        "acl": "readers",
                        "source_revision": 3,
                        "lifecycle_status": "tombstoned",
                    }
                ],
            },
        )
    ]
    assert "documents" not in backend.update_calls[0][1]
    assert "embeddings" not in backend.update_calls[0][1]


@pytest.mark.ci
def test_lifecycle_projection_selects_edge_methods_and_skips_absent_records() -> None:
    backend = _Backend({"ids": [], "metadatas": []})

    assert not backend_update_record_lifecycle(
        backend=backend,
        kind="edge",
        record_id="edge-1",
        lifecycle_patch={"lifecycle_status": "tombstoned"},
    )

    assert backend.get_calls == [
        ("edge", {"ids": ["edge-1"], "include": ["metadatas"]})
    ]
    assert backend.update_calls == []


@pytest.mark.ci
@pytest.mark.parametrize(
    "response",
    [
        None,
        {"ids": [1], "metadatas": [{}]},
        {"ids": ["node-1"], "metadatas": "not-a-list"},
        {"ids": ["node-1"], "metadatas": [{"unsafe": object()}]},
    ],
)
def test_lifecycle_projection_rejects_malformed_backend_results(
    response: object,
) -> None:
    with pytest.raises(TypeError):
        decode_lifecycle_get_result(response)


@pytest.mark.ci
def test_lifecycle_projection_rejects_non_json_patches_before_writing() -> None:
    backend = _Backend({"ids": ["node-1"], "metadatas": [{}]})

    with pytest.raises(TypeError, match="non-JSON"):
        backend_update_record_lifecycle(
            backend=backend,
            kind="node",
            record_id="node-1",
            lifecycle_patch={"unsafe": object()},
        )

    assert backend.update_calls == []
