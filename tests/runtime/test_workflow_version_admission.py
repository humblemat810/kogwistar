from __future__ import annotations

import importlib
import json
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from kogwistar.runtime.design import validate_workflow_design
from kogwistar.runtime.models import WorkflowNode, WorkflowNodeMetadata
from kogwistar.runtime.step_worker import SyncStepWorker

pytestmark = [pytest.mark.ci, pytest.mark.runtime]

_OPERATION_VERSION_FIXTURE = json.loads(
    (
        Path(__file__).resolve().parents[2]
        / "contracts"
        / "golden"
        / "workflow-operation-version-v1.json"
    ).read_text(encoding="utf-8")
)
_EXECUTOR_ROUTING_FIXTURE = json.loads(
    (
        Path(__file__).resolve().parents[2]
        / "contracts"
        / "golden"
        / "runtime-executor-routing-v1.json"
    ).read_text(encoding="utf-8")
)


@dataclass
class _Node:
    id: str
    metadata: dict[str, object]


class _VersionResolver:
    ops = {"compute"}

    def __init__(
        self,
        supported: set[tuple[str, str]],
        environments: dict[tuple[str, str], str] | None = None,
    ) -> None:
        self.supported = supported
        self.environments = environments or {}

    def __call__(self, _op: str):
        return lambda _ctx: None

    def supports_version(self, op: str, version: str) -> bool:
        return (op, version) in self.supported

    def environment_for_version(self, op: str, version: str) -> str | None:
        return self.environments.get((op, version))


class _NonCallableVersionResolver:
    ops = {"compute"}

    def __call__(self, _op: str) -> object:
        return None

    def resolve_version(self, _op: str, _version: str) -> None:
        return None


def _design(monkeypatch, version: str, environment_id: str | None = None):
    metadata: dict[str, object] = {
        "wf_start": True,
        "wf_terminal": True,
        "wf_op": "compute",
        "wf_version": version,
    }
    if environment_id is not None:
        metadata["wf_environment_id"] = environment_id
    start = _Node(
        id="start",
        metadata=metadata,
    )
    design_module = importlib.import_module("kogwistar.runtime.design")
    monkeypatch.setattr(
        design_module,
        "load_workflow_design",
        lambda **_kwargs: (start, {start.id: start}, {start.id: []}, {start.id: []}),
    )
    return start


def test_workflow_admission_rejects_unregistered_operation_version(monkeypatch) -> None:
    _design(monkeypatch, "v2")

    with pytest.raises(ValueError, match="requires unsupported version 'v2'"):
        validate_workflow_design(
            workflow_engine=object(),
            workflow_id="wf-version-admission",
            predicate_registry={},
            resolver=_VersionResolver({("compute", "v1")}),
        )


def test_workflow_admission_rejects_non_callable_versioned_handler(monkeypatch) -> None:
    _design(monkeypatch, "v2")

    with pytest.raises(ValueError, match="requires unsupported version 'v2'"):
        validate_workflow_design(
            workflow_engine=object(),
            workflow_id="wf-version-admission",
            predicate_registry={},
            resolver=_NonCallableVersionResolver(),
        )


def test_step_worker_rejects_handler_changed_to_non_callable() -> None:
    class Resolver:
        def __call__(self, _op: str) -> object:
            raise AssertionError("legacy resolver must not be used for v2")

        def resolve_version(self, _op: str, _version: str) -> None:
            return None

    class Emitter:
        def step_started(self, _trace_context: object) -> None:
            pass

        def step_completed(
            self, _trace_context: object, *, status: str, duration_ms: int
        ) -> None:
            del status, duration_ms

    message_queue: queue.Queue[dict[str, object]] = queue.Queue()
    done_queue: queue.Queue[tuple[object, ...]] = queue.Queue()
    runtime = SimpleNamespace(
        step_resolver=Resolver(),
        lane_message_sender=None,
        lane_message_event_sink=None,
        emitter=Emitter(),
        conversation_engine=object(),
        _should_step_uow=lambda _op, _state: False,
    )
    worker = SyncStepWorker(
        runtime=runtime,
        workflow_id="wf",
        run_id="run",
        conversation_id=None,
        turn_node_id=None,
        nodes={"node": SimpleNamespace(op="compute", version="v2", fail_actiion="continue")},
        trace_context=None,
        message_queue=message_queue,
        done_queue=done_queue,
        work_condition=threading.Condition(),
        authority_context=None,
        cancel_requested=lambda: False,
        parent_lease_renew=None,
        renew_run_ownership=lambda: None,
        apply_child_result=lambda *_args: None,
        now_ms=iter((10, 20)).__next__,
    )

    worker("node", {}, "token", None, 1, 0)

    completed = done_queue.get_nowait()
    assert completed[5] == "error"
    result = completed[1]
    assert "versioned step resolver returned a non-callable handler" in result.errors[0]


def test_workflow_admission_accepts_registered_historical_version(monkeypatch) -> None:
    start = _design(monkeypatch, "v1")
    nodes = {start.id: start}
    adjacency = {start.id: []}
    design_module = importlib.import_module("kogwistar.runtime.design")
    monkeypatch.setattr(
        design_module,
        "load_workflow_design",
        lambda **_kwargs: (start, nodes, adjacency, {start.id: []}),
    )

    assert validate_workflow_design(
        workflow_engine=object(),
        workflow_id="wf-version-admission",
        predicate_registry={},
        resolver=_VersionResolver({("compute", "v1")}),
    ) == (start, nodes, adjacency)


def test_workflow_admission_leaves_rust_target_operation_to_rust_registry(
    monkeypatch,
) -> None:
    start = _design(monkeypatch, "v1")
    start.metadata["wf_op"] = "native-only-operation"
    start.metadata["wf_executor_target"] = "rust-native"

    assert validate_workflow_design(
        workflow_engine=object(),
        workflow_id="wf-native-admission",
        predicate_registry={},
        resolver=_VersionResolver(set()),
    )[0] is start


def test_workflow_admission_rejects_unregistered_python_target_operation(
    monkeypatch,
) -> None:
    start = _design(monkeypatch, "v1")
    start.metadata["wf_op"] = "native-only-operation"
    start.metadata["wf_executor_target"] = "python-worker"

    with pytest.raises(Exception, match="unregistered ops"):
        validate_workflow_design(
            workflow_engine=object(),
            workflow_id="wf-python-admission",
            predicate_registry={},
            resolver=_VersionResolver(set()),
        )


def test_legacy_callable_resolver_remains_valid_for_default_version(monkeypatch) -> None:
    start = _design(monkeypatch, "v1")

    assert validate_workflow_design(
        workflow_engine=object(),
        workflow_id="wf-legacy-version-admission",
        predicate_registry={},
        resolver=lambda _op: lambda _ctx: None,
    )[0] is start


def test_legacy_callable_resolver_cannot_claim_nondefault_version(monkeypatch) -> None:
    _design(monkeypatch, "v2")

    with pytest.raises(ValueError, match="requires unsupported version 'v2'"):
        validate_workflow_design(
            workflow_engine=object(),
            workflow_id="wf-legacy-version-admission",
            predicate_registry={},
            resolver=lambda _op: lambda _ctx: None,
        )


def test_workflow_admission_requires_matching_execution_environment(monkeypatch) -> None:
    _design(monkeypatch, "v1", environment_id="science-py311")
    resolver = _VersionResolver(
        {("compute", "v1")},
        environments={("compute", "v1"): "science-py311"},
    )

    validate_workflow_design(
        workflow_engine=object(),
        workflow_id="wf-environment-admission",
        predicate_registry={},
        resolver=resolver,
    )


def test_workflow_admission_rejects_wrong_execution_environment(monkeypatch) -> None:
    _design(monkeypatch, "v1", environment_id="science-py311")

    with pytest.raises(ValueError, match="requires environment 'science-py311'"):
        validate_workflow_design(
            workflow_engine=object(),
            workflow_id="wf-environment-admission",
            predicate_registry={},
            resolver=_VersionResolver({("compute", "v1")}),
        )


@pytest.mark.parametrize("environment_id", ["", " science-py311", "science-py311 "])
def test_workflow_metadata_rejects_noncanonical_execution_environment(
    environment_id: str,
) -> None:
    with pytest.raises(ValueError, match="wf_environment_id"):
        WorkflowNodeMetadata.model_validate(
            {
                "workflow_id": "wf",
                "wf_op": "compute",
                "wf_environment_id": environment_id,
            }
        )


def test_legacy_workflow_metadata_keeps_environment_field_absent() -> None:
    metadata = WorkflowNode.check_metadataWFN(
        {"workflow_id": "wf", "wf_op": "compute"}
    )

    assert "wf_environment_id" not in metadata
    assert "wf_executor_target" not in metadata
    assert metadata["wf_version"] == _OPERATION_VERSION_FIXTURE["legacy_default"]


@pytest.mark.parametrize("target", ["python-worker", "rust-native"])
def test_workflow_metadata_accepts_explicit_executor_target(target: str) -> None:
    metadata = WorkflowNodeMetadata.model_validate(
        {"workflow_id": "wf", "wf_op": "compute", "wf_executor_target": target}
    )

    assert metadata.wf_executor_target == target


def test_workflow_metadata_rejects_unknown_executor_target() -> None:
    with pytest.raises(ValueError, match="wf_executor_target"):
        WorkflowNodeMetadata.model_validate(
            {"workflow_id": "wf", "wf_op": "compute", "wf_executor_target": "gpu"}
        )


def test_workflow_metadata_rejects_legacy_native_operation_targeted_to_python() -> None:
    with pytest.raises(ValueError, match="cannot target the Python worker"):
        WorkflowNodeMetadata.model_validate(
            {
                "workflow_id": "wf",
                "wf_op": "rust-native/compute",
                "wf_executor_target": "python-worker",
            }
        )


@pytest.mark.parametrize("version", _OPERATION_VERSION_FIXTURE["valid"])
def test_workflow_operation_version_accepts_shared_canonical_values(
    version: str,
) -> None:
    metadata = WorkflowNodeMetadata.model_validate(
        {"workflow_id": "wf", "wf_op": "compute", "wf_version": version}
    )
    assert metadata.wf_version == version


def test_executor_routing_matches_shared_rust_contract_fixture() -> None:
    from kogwistar.runtime.rust_runtime_authority import _native_operation_for_target

    assert _EXECUTOR_ROUTING_FIXTURE["schema_version"] == 1
    for case in _EXECUTOR_ROUTING_FIXTURE["cases"]:
        operation = case["operation"]
        target = case["target"]
        assert _native_operation_for_target(operation, target) == case["native_operation"], (
            case["name"]
        )


@pytest.mark.parametrize(
    ("operation", "target", "message"),
    [
        ("rust-native/noop", "python-worker", "cannot route a rust-native operation"),
        ("rust-native/", None, "native operation name must be non-empty"),
        ("", "rust-native", "native operation name must be non-empty"),
        ("noop", "unknown", "unsupported executor target"),
    ],
)
def test_executor_routing_rejects_invalid_or_ambiguous_requests(
    operation: str, target: str | None, message: str
) -> None:
    from kogwistar.runtime.rust_runtime_authority import (
        RustRuntimeAuthorityError,
        _native_operation_for_target,
    )

    with pytest.raises(RustRuntimeAuthorityError, match=message):
        _native_operation_for_target(operation, target)


@pytest.mark.parametrize("version", _OPERATION_VERSION_FIXTURE["invalid"])
def test_workflow_operation_version_requires_canonical_string(version: object) -> None:
    with pytest.raises(ValueError, match="wf_version"):
        WorkflowNodeMetadata.model_validate(
            {"workflow_id": "wf", "wf_op": "compute", "wf_version": version}
        )


def test_workflow_metadata_normalization_preserves_declared_environment() -> None:
    metadata = WorkflowNode.check_metadataWFN(
        {
            "workflow_id": "wf",
            "wf_op": "compute",
            "wf_environment_id": "science-py311",
        }
    )

    assert metadata["wf_environment_id"] == "science-py311"
