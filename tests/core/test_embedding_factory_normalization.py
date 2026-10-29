import math
import sys
import tomllib
from types import SimpleNamespace
from pathlib import Path

import pytest

from kogwistar.engine_core.embedding_factory import OllamaEmbeddingFunction, _l2_normalize


@pytest.mark.ci
def test_numpy_is_not_a_base_dependency() -> None:
    metadata = tomllib.loads(
        (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )["project"]

    assert not any(str(requirement).lower().startswith("numpy") for requirement in metadata["dependencies"])
    assert any(str(requirement).lower().startswith("numpy") for requirement in metadata["optional-dependencies"]["test"])


@pytest.mark.ci
def test_l2_normalize_preserves_zero_and_unit_vectors() -> None:
    result = _l2_normalize([[0.0, 0.0], [3.0, 4.0]])

    assert result[0] == [0.0, 0.0]
    assert result[1] == pytest.approx([0.6, 0.8])


@pytest.mark.ci
def test_l2_normalize_scales_extreme_finite_values() -> None:
    result = _l2_normalize([[1.0e308, -1.0e308, 1.0e-308]])

    assert result[0] == pytest.approx(
        [math.sqrt(0.5), -math.sqrt(0.5), 0.0], abs=1.0e-12
    )


@pytest.mark.ci
def test_l2_normalize_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        _l2_normalize([[1.0, math.inf]])


@pytest.mark.ci
def test_ollama_embedding_batches_inputs_and_preserves_normalized_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, list[str]]] = []

    def embed(*, model: str, input: list[str]) -> SimpleNamespace:
        calls.append((model, input))
        return SimpleNamespace(embeddings=[[3.0, 4.0], [0.0, 0.0]])

    def legacy_embed(**kwargs: object) -> None:
        raise AssertionError(f"legacy endpoint unexpectedly used: {kwargs}")

    fake_ollama = SimpleNamespace(
        embed=embed,
        embeddings=legacy_embed,
        ResponseError=type("ResponseError", (Exception,), {}),
    )
    monkeypatch.setitem(sys.modules, "ollama", fake_ollama)

    result = OllamaEmbeddingFunction("example-model")(["first", "second"])

    assert calls == [("example-model", ["first", "second"])]
    assert result == [[0.6, 0.8], [0.0, 0.0]]


@pytest.mark.ci
def test_ollama_embedding_rejects_batch_result_count_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_ollama = SimpleNamespace(
        embed=lambda **kwargs: SimpleNamespace(embeddings=[[1.0, 0.0]]),
        embeddings=lambda **kwargs: pytest.fail("unexpected legacy fallback"),
        ResponseError=type("ResponseError", (Exception,), {}),
    )
    monkeypatch.setitem(sys.modules, "ollama", fake_ollama)

    with pytest.raises(ValueError, match="1 vectors for 2 inputs"):
        OllamaEmbeddingFunction("example-model")(["first", "second"])


@pytest.mark.ci
@pytest.mark.parametrize("status_code", [404, 405])
def test_ollama_embedding_uses_legacy_calls_only_when_batch_endpoint_is_missing(
    monkeypatch: pytest.MonkeyPatch, status_code: int
) -> None:
    legacy_prompts: list[str] = []

    class FakeResponseError(Exception):
        def __init__(self, error: str, status_code: int) -> None:
            super().__init__(error)
            self.status_code = status_code

    def embed(*, model: str, input: list[str]) -> None:
        assert model == "example-model"
        assert input == ["first", "second"]
        raise FakeResponseError("batch endpoint unavailable", status_code)

    def legacy_embed(*, model: str, prompt: str) -> SimpleNamespace:
        assert model == "example-model"
        legacy_prompts.append(prompt)
        return SimpleNamespace(embedding=[3.0, 4.0])

    fake_ollama = SimpleNamespace(
        embed=embed,
        embeddings=legacy_embed,
        ResponseError=FakeResponseError,
    )
    monkeypatch.setitem(sys.modules, "ollama", fake_ollama)

    result = OllamaEmbeddingFunction("example-model")(["first", "second"])

    assert legacy_prompts == ["first", "second"]
    assert result == [[0.6, 0.8], [0.6, 0.8]]


@pytest.mark.ci
def test_ollama_embedding_does_not_fallback_for_provider_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResponseError(Exception):
        status_code = 500

    def embed(**kwargs: object) -> None:
        raise FakeResponseError("provider unavailable")

    def legacy_embed(**kwargs: object) -> None:
        raise AssertionError(f"legacy endpoint unexpectedly used: {kwargs}")

    fake_ollama = SimpleNamespace(
        embed=embed,
        embeddings=legacy_embed,
        ResponseError=FakeResponseError,
    )
    monkeypatch.setitem(sys.modules, "ollama", fake_ollama)

    with pytest.raises(FakeResponseError, match="provider unavailable"):
        OllamaEmbeddingFunction("example-model")(["first"])
