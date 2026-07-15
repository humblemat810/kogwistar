from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from kogwistar.id_provider import stable_id
from kogwistar.provenance import evidence_pack_digest_hash
from kogwistar.runtime.serialize import stable_json_dumps
from kogwistar.engine_core.in_memory_backend import _matches_where, _matches_where_python
from kogwistar import shortids
from kogwistar.runtime.base_runtime import apply_state_update_inplace


pytestmark = [pytest.mark.ci, pytest.mark.core]

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = json.loads(
    (ROOT / "contracts" / "golden" / "deterministic-primitives.json").read_text(
        encoding="utf-8"
    )
)


@pytest.fixture(scope="module", autouse=True)
def _native_extension():
    return pytest.importorskip("kogwistar._rust")


@pytest.mark.parametrize("mode", ["shadow", "rust"])
def test_stable_id_matches_python_golden(monkeypatch, mode: str) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_CONTRACTS", mode)
    for case in FIXTURE["stable_ids"]:
        assert str(stable_id(case["kind"], *case["parts"])) == case["expected"]


@pytest.mark.parametrize("mode", ["shadow", "rust"])
def test_canonical_json_matches_python_golden(monkeypatch, mode: str) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_CONTRACTS", mode)
    for case in FIXTURE["canonical_json"]:
        assert stable_json_dumps(case["value"]) == case["expected"]


@pytest.mark.parametrize("mode", ["shadow", "rust"])
def test_evidence_hash_matches_python_golden(monkeypatch, mode: str) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_CONTRACTS", mode)
    for case in FIXTURE["evidence_hashes"]:
        assert evidence_pack_digest_hash(case["value"]) == case["expected"]


@pytest.mark.parametrize("mode", ["shadow", "rust"])
@pytest.mark.parametrize("case", FIXTURE["metadata_filters"], ids=lambda item: item["name"])
def test_metadata_filter_matches_python_golden(monkeypatch, mode: str, case: dict[str, Any]) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_CONTRACTS", mode)
    assert _matches_where(case["metadata"], case["where"]) is case["expected"]


@pytest.mark.parametrize("mode", ["shadow", "rust"])
@pytest.mark.parametrize(
    ("metadata", "where"),
    [
        ({"n": 1}, {"n": {"$in": 1}}),
        ({"n": 1}, {"n": {"$nin": 1}}),
        ({"tags": [1, 2]}, {"tags": {"$in": [2, 3]}}),
        ({"tags": [1, 2]}, {"tags": {"$nin": [2, 3]}}),
        ({"flag": True}, {"flag": 1}),
        ({"value": None}, {"value": {"$eq": None}}),
        ({"value": "1"}, {"value": {"$ne": 1}}),
        ({"tags": ("red", "blue")}, {"tags": {"$in": ("blue",)}}),
        ({"tags": {"red", "blue"}}, {"tags": {"$contains": "blue"}}),
        ({"n": 9007199254740993}, {"n": {"$gt": 9007199254740992}}),
        ({"value": [1, 2]}, {"value": {"$gt": [1, 1]}}),
        ({"name": "x"}, {"$and": []}),
        ({"name": "x"}, {"$or": []}),
    ],
)
def test_metadata_filter_property_table_matches_python(
    monkeypatch, mode: str, metadata: dict[str, Any], where: dict[str, Any]
) -> None:
    expected = _matches_where_python(metadata, where)
    monkeypatch.setenv("KOGWISTAR_IMPL_CONTRACTS", mode)
    assert _matches_where(metadata, where) is expected


@pytest.mark.parametrize(
    ("where", "exception", "code"),
    [
        ({"n": {"$bogus": 1}}, ValueError, "KOGWISTAR_CONTRACT_METADATA_FILTER_UNSUPPORTED_OPERATOR"),
        (["not", "a", "dict"], TypeError, "KOGWISTAR_CONTRACT_METADATA_FILTER_WHERE_TYPE"),
        ({"$and": 1}, TypeError, "KOGWISTAR_CONTRACT_METADATA_FILTER_LOGICAL_CLAUSES_TYPE"),
    ],
)
def test_metadata_filter_invalid_input_differential(
    monkeypatch, where: Any, exception: type[Exception], code: str
) -> None:
    metadata = {"n": 1}
    with pytest.raises(exception):
        _matches_where_python(metadata, where)

    monkeypatch.setenv("KOGWISTAR_IMPL_CONTRACTS", "rust")
    with pytest.raises(exception) as raised:
        _matches_where(metadata, where)
    assert getattr(raised.value, "code") == code


def test_metadata_filter_shadow_does_not_reenter_oracle(monkeypatch) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_CONTRACTS", "shadow")
    assert _matches_where({"n": 1}, {"n": {"$eq": 1}})


def test_metadata_filter_normalization_is_deterministic(_native_extension) -> None:
    payload = '{"where":{"z":{"$eq":1},"a":"x"},"metadata":{"z":1,"a":"x"}}'
    assert _native_extension.normalize_metadata_filter(payload) == (
        '{"metadata":{"a":"x","z":1},"where":{"a":"x","z":{"$eq":1}}}'
    )


@pytest.mark.parametrize("mode", ["shadow", "rust"])
def test_short_ids_round_trip_ordering_and_non_json_fallback(
    monkeypatch, tmp_path, mode: str
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("KOGWISTAR_IMPL_CONTRACTS", mode)
    shortids._MAPPERS.clear()
    shortids.set_current_token("rust-short-id")
    shortids.set_shortid_obj_depth(2)
    shortids.set_shortid_keys(scalars=("id",), lists=("source_ids",))
    value = {"id": "alpha", "source_ids": ["beta", "alpha"], "nested": {"id": "gamma"}}
    short = shortids.l2s_obj(value)
    assert short == {
        "id": "<sid>1",
        "source_ids": ["<sid>2", "<sid>1"],
        "nested": {"id": "<sid>3"},
    }
    assert shortids.s2l_obj(short) == value
    assert shortids.l2s_doc("not json") == "not json"
    with pytest.raises(ValueError, match="Only <sid>") as raised:
        shortids.s2l_id("alpha")
    assert type(raised.value) is ValueError
    with pytest.raises(ValueError, match="Unknown short id '<sid>999'") as raised:
        shortids.s2l_id("<sid>999")
    assert type(raised.value) is ValueError


@pytest.mark.parametrize("mode", ["shadow", "rust"])
def test_runtime_state_reducer_parity_idempotency_and_invalid_inputs(monkeypatch, mode: str) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", mode)
    state: dict[str, Any] = {"items": ["seed"], "keep": {"v": 1}}
    apply_state_update_inplace(
        state,
        [("u", {"answer": 1}), ("a", {"items": "a"}), ("e", {"items": ["b", "c"]})],
    )
    assert state == {"items": ["seed", "a", "b", "c"], "keep": {"v": 1}, "answer": 1}
    with pytest.raises(Exception) as raised:
        apply_state_update_inplace({}, [("u", {"x": 1})], {"x": 2})
    assert getattr(raised.value, "code") == "KOGWISTAR_CONTRACT_STATE_UPDATE_CONFLICT"
    with pytest.raises(TypeError):
        apply_state_update_inplace({}, [("e", {"items": 1})])
    with pytest.raises(AttributeError):
        apply_state_update_inplace({"items": 1}, [("a", {"items": "x"})])


@pytest.mark.parametrize("mode", ["shadow", "rust"])
def test_runtime_state_reducer_preserves_python_object_identity(
    monkeypatch, mode: str
) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", mode)
    untouched: dict[str, Any] = {"nested": []}
    assigned: list[str] = []
    appended: dict[str, str] = {"id": "a"}
    extended: dict[str, str] = {"id": "b"}
    state: dict[str, Any] = {"untouched": untouched, "items": []}

    apply_state_update_inplace(
        state,
        [
            ("u", {"assigned": assigned}),
            ("a", {"items": appended}),
            ("e", {"items": [extended]}),
        ],
    )

    assert state["untouched"] is untouched
    assert state["assigned"] is assigned
    assert state["items"][0] is appended
    assert state["items"][1] is extended


def test_entity_event_replay_golden_and_error_taxonomy(_native_extension) -> None:
    fixture = json.loads((ROOT / "contracts" / "golden" / "event-replay.json").read_text())
    result = json.loads(_native_extension.replay_entity_events(json.dumps({"events": fixture["events"]})))
    assert result == {
        "active_entities": fixture["expected_replay"]["active_entities"],
        "cursor": fixture["expected_replay"]["cursor"],
        "tombstoned_entities": fixture["expected_replay"]["tombstoned_entities"],
    }
    alias = dict(fixture["events"][-1])
    alias["type"] = "entity.delete"
    canonical = json.loads(_native_extension.canonical_entity_event(json.dumps(alias)))
    assert canonical["type"] == "entity.tombstone"
    with pytest.raises(ValueError) as raised:
        _native_extension.canonical_entity_event('{"type":"other","entity":{"id":"x"},"event_seq":1}')
    assert getattr(raised.value, "code") == "KOGWISTAR_CONTRACT_EVENT_TYPE_UNSUPPORTED"

    errors = json.loads((ROOT / "contracts" / "golden" / "errors.json").read_text())
    expected_codes = {item["code"] for item in errors["errors"]}
    invalid_short = {
        "state": {"next": 1, "l2s": {}, "s2l": {}},
        "input": "not-a-short-id",
        "direction": "s2l",
        "depth": 0,
        "scalar_keys": ["id"],
        "list_keys": [],
        "primitive": True,
    }
    with pytest.raises(ValueError) as raised:
        _native_extension.short_id_transform(json.dumps(invalid_short))
    assert getattr(raised.value, "code") in expected_codes
    with pytest.raises(ValueError) as raised:
        _native_extension.apply_state_update(
            '{"state":{},"state_update":[["u",{"x":1}]],"update":{"x":2}}'
        )
    assert getattr(raised.value, "code") in expected_codes
