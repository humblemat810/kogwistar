from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from kogwistar.runtime.design import validate_workflow_design
from kogwistar.runtime.models import WorkflowNode, WorkflowNodeMetadata

pytestmark = [pytest.mark.ci, pytest.mark.runtime]

_OPERATION_VERSION_FIXTURE = json.loads(
    (
        Path(__file__).resolve().parents[2]
        / "contracts"
        / "golden"
        / "workflow-operation-version-v1.json"
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
    assert metadata["wf_version"] == _OPERATION_VERSION_FIXTURE["legacy_default"]


@pytest.mark.parametrize("version", _OPERATION_VERSION_FIXTURE["valid"])
def test_workflow_operation_version_accepts_shared_canonical_values(
    version: str,
) -> None:
    metadata = WorkflowNodeMetadata.model_validate(
        {"workflow_id": "wf", "wf_op": "compute", "wf_version": version}
    )

    assert metadata.wf_version == version


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
