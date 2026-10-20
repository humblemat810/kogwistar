from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.runtime.runtime import WorkflowRuntime, _json_int, _json_string_list


pytestmark = pytest.mark.ci


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, -1),
        (True, -1),
        ({"step_seq": 4}, -1),
        ("not-an-int", -1),
        (1.8, -1),
        (7.0, 7),
        ("7", 7),
        (7, 7),
    ],
)
def test_persisted_integer_boundary_rejects_structured_values(
    value: object, expected: int
) -> None:
    assert _json_int(value, default=-1) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, []),
        ({"errors": ["hidden"]}, []),
        ("not-a-list", []),
        (["first", 2, {"detail": "third"}], ["first", "2", "{'detail': 'third'}"]),
    ],
)
def test_persisted_error_list_boundary_is_total(
    value: object, expected: list[str]
) -> None:
    assert _json_string_list(value) == expected


def test_optional_metadata_adapter_is_absent_for_legacy_engine_doubles() -> None:
    runtime = object.__new__(WorkflowRuntime)
    runtime.conversation_engine = cast(GraphKnowledgeEngine, SimpleNamespace())

    assert runtime._runtime_metadata_store() is None


def _terminal_runtime(checkpoint_state: object) -> WorkflowRuntime:
    runtime = object.__new__(WorkflowRuntime)
    runtime.conversation_engine = cast(
        GraphKnowledgeEngine,
        SimpleNamespace(
            read=SimpleNamespace(
                get_nodes=lambda **_: [SimpleNamespace(metadata={"errors": ["boom"]})]
            )
        ),
    )
    runtime._latest_checkpoint_for_run = lambda **_: SimpleNamespace(
        metadata={"state_json": checkpoint_state}
    )
    return runtime


def test_terminal_run_rehydrates_checkpoint_state_as_typed_json() -> None:
    result = _terminal_runtime('{"answer": 42}')._terminal_run_result(
        conversation_id="conversation-1", run_id="run-1"
    )

    assert result is not None
    assert result.final_state == {"answer": 42}
    assert result.errors == ["boom"]


@pytest.mark.parametrize("state_json", ["[]", "null", "not-json"])
def test_terminal_run_rejects_malformed_checkpoint_state(state_json: object) -> None:
    with pytest.raises(ValueError, match="checkpoint state is not a JSON object"):
        _terminal_runtime(state_json)._terminal_run_result(
            conversation_id="conversation-1", run_id="run-1"
        )
