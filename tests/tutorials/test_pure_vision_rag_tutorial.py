from __future__ import annotations

import runpy
from pathlib import Path

import pytest

pytestmark = pytest.mark.ci

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "pure_vision_rag_tutorial.py"


def _load_module() -> dict[str, object]:
    return runpy.run_path(str(SCRIPT_PATH), run_name="pure_vision_rag_tutorial")


def test_mock_pure_vision_rag_retrieves_image_grounding() -> None:
    module = _load_module()
    result = module["run_mock"]("rabbit")

    assert result["hits"]
    assert result["hits"][0]["id"] == "image:rabbit"
    assert result["hits"][0]["multimodal_spans"][0]["modality"] == "image"
    assert result["hits"][0]["multimodal_spans"][0]["locator"]["kind"] == "spatial_region"
    assert result["profile"]["dimension"] == 4


def test_mock_queries_remain_in_one_declared_embedding_space() -> None:
    module = _load_module()
    rabbit = module["run_mock"]("rabbit")
    lecture = module["run_mock"]("lecture")

    assert rabbit["profile_fingerprint"] == lecture["profile_fingerprint"]
    assert rabbit["hits"][0]["id"] != lecture["hits"][0]["id"]
