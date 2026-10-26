from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from kogwistar.engine_core.projection_action import (
    ProjectionJobIdentity,
    decode_projection_action,
)


def _fixture() -> str:
    return (
        Path(__file__).resolve().parents[2]
        .joinpath("contracts/golden/graph-projection-action-v1.json")
        .read_text(encoding="utf-8")
        .strip()
    )


def _job(**changes: str) -> ProjectionJobIdentity:
    values = {
        "job_id": "projection-job",
        "namespace": "workspace",
        "entity_kind": "node",
        "entity_id": "node-1",
        "index_kind": "node_embedding",
        "op": "UPSERT",
    }
    values.update(changes)
    return ProjectionJobIdentity(**values)


def test_python_decodes_shared_rust_projection_action_fixture() -> None:
    action_json = _fixture()
    action = decode_projection_action(
        _job(), action_json, hashlib.sha256(action_json.encode("utf-8")).hexdigest()
    )

    assert action.version == 1
    assert action.operation == "UPSERT"
    assert action.record is not None
    assert action.record.id == "node-1"
    assert action.record.embedding == (0.25, 0.75)
    assert action.record.metadata == {"kind": "node"}


@pytest.mark.parametrize(
    ("job", "mutate"),
    [
        (_job(namespace="other"), lambda action: action),
        (_job(entity_id="other-node"), lambda action: action),
        (_job(op="DELETE"), lambda action: action),
        (_job(), lambda action: {**action, "version": 1.0}),
        (_job(), lambda action: {**action, "extra": True}),
        (
            _job(),
            lambda action: {
                **action,
                "record": {**action["record"], "metadata": {"_kogwistar_namespace": "other"}},
            },
        ),
    ],
)
def test_python_decoder_fails_closed_on_binding_or_schema_mismatch(
    job: ProjectionJobIdentity, mutate: object
) -> None:
    action = mutate(json.loads(_fixture()))  # type: ignore[operator]
    action_json = json.dumps(action, separators=(",", ":"), sort_keys=True)

    with pytest.raises(ValueError):
        decode_projection_action(
            job, action_json, hashlib.sha256(action_json.encode("utf-8")).hexdigest()
        )


def test_python_decoder_rejects_bad_digest_before_parsing_action() -> None:
    with pytest.raises(ValueError, match="invalid SHA-256"):
        decode_projection_action(_job(), _fixture(), "0" * 64)


def test_python_decoder_rejects_embedding_outside_f32_range() -> None:
    action = json.loads(_fixture())
    action["record"]["embedding"] = [3.5e38]
    action_json = json.dumps(action, separators=(",", ":"), sort_keys=True)

    with pytest.raises(ValueError, match="finite f32"):
        decode_projection_action(
            _job(), action_json, hashlib.sha256(action_json.encode("utf-8")).hexdigest()
        )
