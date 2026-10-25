from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from kogwistar import shortids
from kogwistar.acl.models import ACLEdge, ACLNode
from kogwistar.conversation.models import ConversationEdge, ConversationNode
from kogwistar._rust_bridge import RustParityError
from kogwistar.engine_core.in_memory_backend import (
    _matches_where,
    _matches_where_python,
)
from kogwistar.engine_core.graph_contracts import SUPPORTED_GRAPH_MODEL_CLASSES
from kogwistar.engine_core.models import (
    Edge,
    FlattenedLLMEdge,
    Grounding,
    LLMEdge,
    LLMNode,
    PureChromaNode,
    Node,
    Span,
    DocNode,
)
from kogwistar.id_provider import stable_id
from kogwistar.provenance import evidence_pack_digest_hash
from kogwistar.runtime.base_runtime import apply_state_update_inplace
from kogwistar.runtime.checkpointed_projection import ProjectionCheckpoint
from kogwistar.runtime.models import (
    WorkflowCancelledNode,
    WorkflowCheckpointNode,
    WorkflowCompletedNode,
    WorkflowEdge,
    WorkflowFailedNode,
    WorkflowRunNode,
    WorkflowRuntimeEdge,
    WorkflowStepExecNode,
    WorkflowNode,
)
from kogwistar.runtime.serialize import stable_json_dumps
from kogwistar.splitter import split_doc_deterministic

pytestmark = [pytest.mark.ci, pytest.mark.core]

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = json.loads(
    (ROOT / "contracts" / "golden" / "deterministic-primitives.json").read_text(
        encoding="utf-8"
    )
)
STATE_UPDATE_FIXTURE = json.loads(
    (ROOT / "contracts" / "golden" / "runtime-state-update-v1.json").read_text(
        encoding="utf-8"
    )
)
GRAPH_MODEL_FIXTURE = json.loads(
    (ROOT / "contracts" / "golden" / "python-graph-model-payloads-v1.json").read_text(
        encoding="utf-8"
    )
)
SOURCE_MAP_FIXTURE = json.loads(
    (ROOT / "contracts" / "golden" / "source-map-v1.json").read_text(
        encoding="utf-8"
    )
)
PROJECTION_CHECKPOINT_FIXTURE = json.loads(
    (ROOT / "contracts" / "golden" / "projection-checkpoint-v1.json").read_text(
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


def test_python_source_map_projection_matches_shared_chunk_contract() -> None:
    fixture = SOURCE_MAP_FIXTURE
    content = "alpha beta"
    chunks = split_doc_deterministic(
        doc_id=fixture["doc_id"],
        content=content,
        max_chars=5,
        overlap_chars=0,
        prefer_window=0,
    )
    projected = {
        "contract_version": 1,
        "doc_id": fixture["doc_id"],
        "chunks": [
            {
                "chunk_id": f"chunk-{index}",
                "doc_id": chunk.doc_id,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "text": chunk.text,
            }
            for index, chunk in enumerate(chunks)
        ],
    }
    assert projected == fixture


def test_python_projection_checkpoint_matches_shared_wire_contract() -> None:
    fixture = dict(PROJECTION_CHECKPOINT_FIXTURE)
    checkpoint = ProjectionCheckpoint.from_wire(fixture)
    assert checkpoint.to_wire() == fixture
    assert asdict(checkpoint) == {
        key: value
        for key, value in fixture.items()
        if key != "contract_version"
    } | {"processed_event_ids": tuple(fixture["processed_event_ids"])}


def test_python_projection_checkpoint_wire_decoder_fails_closed() -> None:
    fixture = dict(PROJECTION_CHECKPOINT_FIXTURE)
    with pytest.raises(ValueError, match="unknown"):
        ProjectionCheckpoint.from_wire({**fixture, "unexpected": True})

    with pytest.raises(ValueError, match="unique"):
        ProjectionCheckpoint.from_wire(
            {**fixture, "processed_event_ids": ["event-1", "event-1"]}
        )


@pytest.mark.requires_rust
def test_native_graph_model_registry_matches_python_seam() -> None:
    """Detect a stale native extension before it accepts partial graph parity."""

    from kogwistar import _rust

    assert tuple(_rust.supported_graph_model_classes()) == SUPPORTED_GRAPH_MODEL_CLASSES


@pytest.mark.parametrize("mode", ["shadow", "rust"])
def test_canonical_json_matches_python_golden(monkeypatch, mode: str) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_CONTRACTS", mode)
    for case in FIXTURE["canonical_json"]:
        assert stable_json_dumps(case["value"]) == case["expected"]


@pytest.mark.requires_rust
def test_python_node_edge_backend_payloads_round_trip_through_rust_json(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_CONTRACTS", "rust")

    def evidence(text: str) -> Span:
        return Span(
            collection_page_url="https://example.test/source",
            document_page_url="https://example.test/source#span",
            doc_id="doc:rust-python-parity",
            insertion_method="contract-test",
            page_number=1,
            start_char=0,
            end_char=len(text),
            excerpt=text,
            context_before="",
            context_after="",
        )

    node = Node(
        id="parity-node",
        label="Python node",
        type="entity",
        summary="A complete Pydantic node payload crosses the Rust persistence seam.",
        doc_id="doc:rust-python-parity",
        mentions=[Grounding(spans=[evidence("Node evidence")])],
        metadata={"workspace_id": "ws", "graph_space": "facts", "kind": "node"},
        embedding=[1.0, 0.0, 0.0],
        level_from_root=0,
        domain_id=None,
        canonical_entity_id="entity:parity-node",
        properties={"attributes": ["unicode-雪", "nested-payload"]},
    )
    edge = Edge(
        id="parity-edge",
        label="supports",
        type="relationship",
        summary="The typed edge preserves both endpoint and source-span data.",
        doc_id="doc:rust-python-parity",
        mentions=[Grounding(spans=[evidence("Edge evidence")])],
        metadata={"workspace_id": "ws", "graph_space": "facts", "kind": "edge"},
        embedding=[0.0, 1.0, 0.0],
        domain_id=None,
        canonical_entity_id=None,
        properties={"confidence_label": "high"},
        source_ids=[node.id],
        target_ids=["external:target"],
        relation="supports",
        source_edge_ids=[],
        target_edge_ids=[],
    )
    hyperedge = edge.model_copy(
        update={
            "id": "parity-hyperedge",
            "label": "qualifies",
            "summary": "An edge can relate to other edges through typed edge endpoint fields.",
            "relation": "qualifies",
            "source_edge_ids": ["parity-edge"],
            "target_edge_ids": ["parity-edge-target"],
        }
    )

    assert GRAPH_MODEL_FIXTURE["schema_version"] == 1
    for model, kind, case in zip(
        (node, edge, hyperedge),
        ("node", "edge", "edge_with_edge_endpoints"),
        GRAPH_MODEL_FIXTURE["cases"],
        strict=True,
    ):
        payload = model.model_dump(field_mode="backend", dump_format="json")
        wire_json = stable_json_dumps(payload)
        assert case["kind"] == kind
        assert payload == case["payload"]
        assert wire_json == case["canonical_json"]
        restored = type(model).model_validate(json.loads(wire_json))
        assert restored.model_dump(field_mode="backend", dump_format="json") == payload


@pytest.mark.requires_rust
def test_tombstoned_edge_lifecycle_is_shared_by_python_and_rust(monkeypatch) -> None:
    """An edge tombstone must cross the same typed seam as a node tombstone."""
    monkeypatch.setenv("KOGWISTAR_IMPL_CONTRACTS", "rust")
    from kogwistar import _rust

    payload = copy.deepcopy(GRAPH_MODEL_FIXTURE["cases"][1]["payload"])
    payload["metadata"]["lifecycle_status"] = "deleted"
    payload["metadata"]["redirect_to_id"] = "parity-edge-replacement"

    normalized = Edge.model_validate(payload).model_dump(
        field_mode="backend", dump_format="json"
    )
    assert normalized["metadata"]["lifecycle_status"] == "tombstoned"
    _rust.validate_graph_model_family(
        json.dumps(
            {"schema_version": 1, "model_class": "edge", "payload": normalized},
            ensure_ascii=False,
        )
    )

    invalid = copy.deepcopy(normalized)
    invalid["metadata"]["lifecycle_status"] = "purged"
    with pytest.raises(ValueError, match="lifecycle_status"):
        Edge.model_validate(invalid)
    with pytest.raises(Exception, match="lifecycle_status"):
        _rust.validate_graph_model_family(
            json.dumps(
                {"schema_version": 1, "model_class": "edge", "payload": invalid},
                ensure_ascii=False,
            )
        )


@pytest.mark.requires_rust
def test_python_family_payloads_validate_through_rust_graph_contract(monkeypatch) -> None:
    """Derived Python payloads use the same typed family boundary as Rust."""

    monkeypatch.setenv("KOGWISTAR_IMPL_CONTRACTS", "rust")
    from kogwistar import _rust

    base_node = copy.deepcopy(GRAPH_MODEL_FIXTURE["cases"][0]["payload"])

    def envelope(model_class: str, payload: dict[str, Any]) -> str:
        return json.dumps(
            {"schema_version": 1, "model_class": model_class, "payload": payload},
            ensure_ascii=False,
        )

    def project(
        model_type: type[BaseModel],
        payload: dict[str, Any],
        *,
        field_mode: str = "backend",
    ) -> dict[str, Any]:
        return model_type.model_validate(payload).model_dump(
            field_mode=field_mode, dump_format="json"
        )

    workflow_node = copy.deepcopy(base_node)
    workflow_node["metadata"] = {
        "entity_type": "workflow_node",
        "workflow_id": "wf:python-fixture",
        "wf_op": "parse",
    }
    workflow_node = project(WorkflowNode, workflow_node)
    _rust.validate_graph_model_family(envelope("workflow_node", workflow_node))

    workflow_edge = copy.deepcopy(GRAPH_MODEL_FIXTURE["cases"][1]["payload"])
    workflow_edge["metadata"] = {
        "entity_type": "workflow_edge",
        "workflow_id": "wf:python-fixture",
        "wf_predicate": "ok",
        "wf_priority": 1,
    }
    workflow_edge = project(WorkflowEdge, workflow_edge)
    _rust.validate_graph_model_family(envelope("workflow_edge", workflow_edge))

    runtime_nodes = [
        (
            WorkflowRunNode,
            "workflow_run_node",
            {
                "entity_type": "workflow_run",
                "workflow_id": "wf:python-fixture",
                "run_id": "run:python-fixture",
                "conversation_id": "conversation:python-fixture",
                "turn_node_id": "turn:python-fixture",
            },
        ),
        (
            WorkflowStepExecNode,
            "workflow_step_exec_node",
            {
                "entity_type": "workflow_step_exec",
                "run_id": "run:python-fixture",
                "workflow_id": "wf:python-fixture",
                "workflow_node_id": "wf-node:parse",
                "step_seq": 0,
                "op": "parse",
                "result_json": "{}",
            },
        ),
        (
            WorkflowCheckpointNode,
            "workflow_checkpoint_node",
            {
                "entity_type": "workflow_checkpoint",
                "run_id": "run:python-fixture",
                "workflow_id": "wf:python-fixture",
                "step_seq": 0,
                "state_json": "{}",
            },
        ),
        (
            WorkflowCompletedNode,
            "workflow_completed_node",
            {
                "entity_type": "workflow_completed",
                "workflow_id": "wf:python-fixture",
                "run_id": "run:python-fixture",
                "conversation_id": "conversation:python-fixture",
                "accepted_step_seq": 0,
            },
        ),
        (
            WorkflowCancelledNode,
            "workflow_cancelled_node",
            {
                "entity_type": "workflow_cancelled",
                "workflow_id": "wf:python-fixture",
                "run_id": "run:python-fixture",
                "conversation_id": "conversation:python-fixture",
                "accepted_step_seq": 0,
            },
        ),
        (
            WorkflowFailedNode,
            "workflow_failed_node",
            {
                "entity_type": "workflow_failed",
                "workflow_id": "wf:python-fixture",
                "run_id": "run:python-fixture",
                "conversation_id": "conversation:python-fixture",
                "accepted_step_seq": 0,
            },
        ),
    ]
    for model_type, model_class, metadata in runtime_nodes:
        runtime_node = copy.deepcopy(base_node)
        runtime_node["metadata"] = metadata
        runtime_node = project(model_type, runtime_node)
        _rust.validate_graph_model_family(envelope(model_class, runtime_node))

    runtime_edge = copy.deepcopy(GRAPH_MODEL_FIXTURE["cases"][1]["payload"])
    runtime_edge["metadata"] = {
        "entity_type": "conversation_edge",
        "relation": "next",
    }
    runtime_edge = project(WorkflowRuntimeEdge, runtime_edge)
    _rust.validate_graph_model_family(envelope("workflow_runtime_edge", runtime_edge))

    conversation_node = copy.deepcopy(base_node)
    conversation_node["metadata"] = {
        "entity_type": "conversation_node",
        "level_from_root": 0,
        "role": "user",
        "turn_index": 0,
    }
    conversation_node = project(ConversationNode, conversation_node)
    _rust.validate_graph_model_family(envelope("conversation_node", conversation_node))

    for entity_type, role, turn_index in (
        ("conversation_start", "system", -1),
        ("conversation_turn", "user", 0),
        ("conversation_summary", "system", 1),
        ("assistant_turn", "assistant", 2),
    ):
        conversation_subtype = copy.deepcopy(base_node)
        conversation_subtype["metadata"] = {
            "entity_type": entity_type,
            "level_from_root": 0,
            "role": role,
            "turn_index": turn_index,
            "conversation_id": "conversation:python-fixture",
            "user_id": "user:python-fixture",
        }
        conversation_subtype = project(ConversationNode, conversation_subtype)
        _rust.validate_graph_model_family(
            envelope("conversation_node", conversation_subtype)
        )

    conversation_edge = copy.deepcopy(GRAPH_MODEL_FIXTURE["cases"][1]["payload"])
    conversation_edge["metadata"] = {
        "causal_type": "chain",
        "char_distance_from_last_summary": 4,
        "turn_distance_from_last_summary": 1,
    }
    conversation_edge = project(ConversationEdge, conversation_edge)
    _rust.validate_graph_model_family(envelope("conversation_edge", conversation_edge))

    llm_node = copy.deepcopy(base_node)
    llm_node.pop("embedding")
    llm_node = project(LLMNode, llm_node, field_mode="llm")
    _rust.validate_graph_model_family(envelope("llm_node", llm_node))

    llm_edge = copy.deepcopy(GRAPH_MODEL_FIXTURE["cases"][1]["payload"])
    llm_edge.pop("embedding")
    llm_edge = project(LLMEdge, llm_edge, field_mode="llm")
    _rust.validate_graph_model_family(envelope("llm_edge", llm_edge))

    pure_chroma_node = copy.deepcopy(base_node)
    pure_chroma_node.pop("mentions")
    pure_chroma_node = project(PureChromaNode, pure_chroma_node)
    _rust.validate_graph_model_family(envelope("pure_chroma_node", pure_chroma_node))

    flattened_edge = copy.deepcopy(GRAPH_MODEL_FIXTURE["cases"][1]["payload"])
    flattened_edge.pop("embedding")
    flattened_edge["mentions"] = [{"span_ids": ["sp:edge-1"]}]
    flattened_edge = project(FlattenedLLMEdge, flattened_edge, field_mode="llm")
    _rust.validate_graph_model_family(
        envelope("flattened_llm_edge", flattened_edge)
    )
    _rust.validate_graph_model_family(
        envelope("assoc_flattened_llm_edge", flattened_edge)
    )

    doc_node = {
        "id": "doc-node:python",
        "label": "Page 1",
        "type": "page",
        "summary": "A document page",
        "doc_id": "document:python",
        "mentions": base_node["mentions"],
    }
    doc_node = project(DocNode, doc_node)
    _rust.validate_graph_model_family(envelope("doc_node", doc_node))

    acl_node = copy.deepcopy(base_node)
    acl_node["metadata"] = {
        "entity_type": "acl_record",
        "acl_truth_graph": "truth:python",
        "acl_target_entity_id": "parity-node",
        "acl_target_grain": "node",
        "acl_version": 1,
        "acl_mode": "private",
    }
    acl_node = project(ACLNode, acl_node)
    _rust.validate_graph_model_family(envelope("acl_node", acl_node))

    acl_edge = copy.deepcopy(GRAPH_MODEL_FIXTURE["cases"][1]["payload"])
    acl_edge["metadata"] = {
        "entity_type": "acl_targets_truth",
        "acl_truth_graph": "truth:python",
        "acl_target_entity_id": "parity-node",
        "acl_target_grain": "node",
        "level_from_root": 0,
    }
    acl_edge = project(ACLEdge, acl_edge)
    _rust.validate_graph_model_family(envelope("acl_edge", acl_edge))

    invalid_llm = copy.deepcopy(base_node)
    with pytest.raises(Exception):
        _rust.validate_graph_model_family(envelope("llm_node", invalid_llm))


@pytest.mark.parametrize("mode", ["shadow", "rust"])
def test_canonical_json_float_magnitude_sweep_matches_python(
    monkeypatch, mode: str
) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_CONTRACTS", mode)
    rng = random.Random(20261024)
    values = [
        math.ldexp((-1.0 if index % 2 else 1.0) * (0.5 + rng.random()), exponent)
        for exponent in range(-1074, 1024, 7)
        for index in range(3)
    ]
    values.extend(math.ldexp(1.0, exponent) for exponent in range(-1074, 1024))
    payload = {"values": values}
    expected = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )

    assert stable_json_dumps(payload) == expected


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
    assert raised.value.code == code


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


@pytest.mark.parametrize("as_document", [False, True])
def test_short_id_shadow_mismatch_rolls_back_entire_object_mapping(
    monkeypatch, tmp_path, as_document: bool
) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_CONTRACTS", "shadow")
    mapper = shortids.ShortIdMapper("shadow-rollback", root_dir=str(tmp_path))
    mapper.set_obj_max_depth(2)
    mapper.set_id_keys(scalars=("id",), lists=("source_ids",))
    original_state = copy.deepcopy(mapper.state)
    rust_transform = shortids._rust_short_id_transform
    calls = 0

    def mismatch_second_alias(*, payload, python_value):
        nonlocal calls
        result = rust_transform(payload=payload, python_value=python_value)
        calls += 1
        if calls == 2:
            return {**result, "value": "<sid>999"}
        return result

    monkeypatch.setattr(shortids, "_rust_short_id_transform", mismatch_second_alias)
    with pytest.raises(RustParityError, match="parity mismatch"):
        value = {"id": "node:one", "source_ids": ["edge:one"]}
        if as_document:
            mapper.l2s_doc(json.dumps(value))
        else:
            mapper.l2s_obj(value)

    assert calls == 2
    assert mapper.state == original_state
    assert json.loads(mapper._file().read_text(encoding="utf-8")) == original_state


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
    assert raised.value.code == "KOGWISTAR_CONTRACT_STATE_UPDATE_CONFLICT"
    with pytest.raises(TypeError):
        apply_state_update_inplace({}, [("e", {"items": 1})])
    with pytest.raises(AttributeError):
        apply_state_update_inplace({"items": 1}, [("a", {"items": "x"})])


@pytest.mark.parametrize("mode", ["shadow", "rust"])
def test_runtime_state_reducer_matches_shared_python_golden(
    monkeypatch, mode: str
) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", mode)
    assert STATE_UPDATE_FIXTURE["schema_version"] == 1
    for case in STATE_UPDATE_FIXTURE["cases"]:
        state = copy.deepcopy(case["state"])
        apply_state_update_inplace(
            state,
            copy.deepcopy(case["state_update"]),
            copy.deepcopy(case["update"]),
            state_schema=copy.deepcopy(case["state_schema"]),
        )
        assert state == case["expected"], f"case {case['name']} in {mode} mode"


def test_runtime_state_reducer_seeded_differential_sweep(monkeypatch) -> None:
    rng = random.Random(20261024)

    for sample in range(128):
        initial_state: dict[str, Any] = {
            "items": [sample, "seed"],
            "events": [{"sequence": sample}],
            "current": sample,
        }
        if sample % 2:
            state_update = [
                ("u", {"current": rng.randint(-10_000, 10_000)}),
                ("a", {"events": {"sequence": sample, "kind": "append"}}),
                ("e", {"items": [rng.randint(-100, 100) for _ in range(sample % 7)]}),
            ]
            update = None
            state_schema: dict[str, str] = {}
        else:
            state_update = []
            update = {
                "events": [rng.randint(-100, 100) for _ in range(sample % 5)],
                "current": rng.randint(-10_000, 10_000),
            }
            state_schema = {"events": "a"}

        python_state = copy.deepcopy(initial_state)
        monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "python")
        apply_state_update_inplace(
            python_state,
            copy.deepcopy(state_update),
            copy.deepcopy(update),
            state_schema=copy.deepcopy(state_schema),
        )
        rust_state = copy.deepcopy(initial_state)
        monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "rust")
        apply_state_update_inplace(
            rust_state,
            copy.deepcopy(state_update),
            copy.deepcopy(update),
            state_schema=copy.deepcopy(state_schema),
        )
        assert rust_state == python_state, f"sample {sample} differs between reducers"


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


def test_runtime_state_reducer_parity_mismatch_does_not_mutate_caller_state(
    monkeypatch,
) -> None:
    from kogwistar.runtime import base_runtime

    state = {"items": ["seed"], "stable": {"value": 1}}
    original = copy.deepcopy(state)
    monkeypatch.setattr(base_runtime, "runtime_implementation_mode", lambda: "rust")
    monkeypatch.setattr(
        base_runtime,
        "runtime_apply_state_update",
        lambda *, payload: {"items": ["seed", "incorrect"], "stable": {"value": 1}},
    )

    with pytest.raises(RustParityError, match="parity mismatch"):
        base_runtime.apply_state_update_inplace(
            state, [("a", {"items": "expected"})]
        )

    assert state == original


@pytest.mark.parametrize(
    ("native_name", "payload"),
    [
        ("runtime_select_route", {"edges": []}),
        (
            "runtime_plan_successors",
            {
                "token_id": "token",
                "step_seq": 1,
                "current_join_mask": 0,
                "join_outstanding": [],
                "successors": [],
            },
        ),
        (
            "runtime_apply_join_arrival",
            {
                "join_index": 0,
                "join_outstanding": [],
                "arrival": {"join_mask": 0, "token_id": "token"},
            },
        ),
        ("runtime_decide_retry", {"retry_budget": 1, "attempt_number": 0}),
        (
            "runtime_plan_nested_invocation",
            {
                "parent_run_id": "parent",
                "workflow_id": "workflow",
                "parent_conversation_id": "conversation",
                "parent_turn_node_id": "turn",
            },
        ),
        (
            "runtime_decide_dispatch",
            {"max_workers": 1, "inflight": 0, "pending": 0},
        ),
        (
            "runtime_decide_budget_suspend",
            {
                "token_budget": 0,
                "token_used": 0,
                "time_budget_ms": 0,
                "time_used_ms": 0,
                "rate_limit": 0,
                "rate_used": 0,
                "step_budget": 0,
                "step_used": 0,
                "call_budget": 0,
                "call_used": 0,
                "cost_budget": 0.0,
                "cost_used": 0.0,
            },
        ),
        (
            "runtime_scheduler_tick",
            {"pending": [], "inflight": 0, "max_workers": 1},
        ),
    ],
)
def test_native_runtime_json_boundary_rejects_unknown_dto_fields(
    _native_extension, native_name: str, payload: dict[str, Any]
) -> None:
    payload = {**payload, "unexpected": True}

    with pytest.raises(ValueError, match="unknown field"):
        getattr(_native_extension, native_name)(json.dumps(payload))


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
    assert raised.value.code == "KOGWISTAR_CONTRACT_EVENT_TYPE_UNSUPPORTED"

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
    assert raised.value.code in expected_codes
    with pytest.raises(ValueError) as raised:
        _native_extension.apply_state_update(
            '{"state":{},"state_update":[["u",{"x":1}]],"update":{"x":2}}'
        )
    assert raised.value.code in expected_codes
