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
    assert result["answer"]
    assert "Rabbit" in result["answer"]


def test_mock_queries_remain_in_one_declared_embedding_space() -> None:
    module = _load_module()
    rabbit = module["run_mock"]("rabbit")
    lecture = module["run_mock"]("lecture")

    assert rabbit["profile_fingerprint"] == lecture["profile_fingerprint"]
    assert rabbit["hits"][0]["id"] != lecture["hits"][0]["id"]


def test_tui_mock_explains_that_it_is_not_model_quality_evidence(monkeypatch, capsys) -> None:
    module = _load_module()
    answers = iter(["1", "rabbit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    module["run_tui"]()

    output = capsys.readouterr().out
    assert "deterministic answer" in output
    assert "says nothing about retrieval quality or speed" in output


def test_tui_curated_clip_uses_checked_in_fixture(monkeypatch, capsys) -> None:
    module = _load_module()
    answers = iter(["2", "", "diagram", "1", "", "1"])
    captured: dict[str, object] = {}

    def fake_run_real(**kwargs):
        captured.update(kwargs)
        return {"profile": {"model": "clip-ViT-B-32"}, "hits": [], "answer": None}

    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    monkeypatch.setitem(module["run_tui"].__globals__, "run_real", fake_run_real)
    monkeypatch.setitem(
        module["run_tui"].__globals__, "_wait_for_embedding_service", lambda _endpoint: None
    )

    module["run_tui"]()

    output = capsys.readouterr().out
    assert "Known CPU CLIP profile" in output
    assert "checked-in demo image" in output
    assert captured["embedding_api"] == "llm-wiki-v1"
    assert captured["image_paths"] == [("sample-diagram", module["DEMO_IMAGE_PATH"])]


def test_tui_curated_gpu_qwen_uses_repository_endpoint(monkeypatch, capsys) -> None:
    module = _load_module()
    answers = iter(["3", "", "diagram", "1", "", "1"])
    captured: dict[str, object] = {}

    def fake_run_real(**kwargs):
        captured.update(kwargs)
        return {"profile": {"model": "Qwen/Qwen3-VL-Embedding-2B"}, "hits": [], "answer": None}

    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    monkeypatch.setitem(module["run_tui"].__globals__, "run_real", fake_run_real)
    monkeypatch.setitem(
        module["run_tui"].__globals__, "_wait_for_embedding_service", lambda _endpoint: None
    )

    module["run_tui"]()

    output = capsys.readouterr().out
    assert "Known GPU Qwen3-VL profile" in output
    assert captured["embedding_endpoint"] == module["DEFAULT_QWEN_GPU_ENDPOINT"]
    assert captured["embedding_api"] == "llm-wiki-v1"
