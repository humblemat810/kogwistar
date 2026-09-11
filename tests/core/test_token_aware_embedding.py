from __future__ import annotations

from types import SimpleNamespace

import pytest

from kogwistar.engine_core.embedding_profile import EmbeddingProfile
from kogwistar.engine_core.subsystems.embed import EmbedSubsystem


class _TokenEmbedder:
    max_input_tokens = 8192

    def __init__(self, *, accepted_tokens: int = 8192) -> None:
        self.accepted_tokens = accepted_tokens
        self.calls: list[str] = []

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        return " ".join(text.split()[:max_tokens])

    def __call__(self, values: list[str]) -> list[list[float]]:
        text = values[0]
        self.calls.append(text)
        if self.count_tokens(text) > self.accepted_tokens:
            raise ValueError("context length exceeded")
        return [[3.0, 4.0]]


class _MarkerEmbedder(_TokenEmbedder):
    def __call__(self, values: list[str]) -> list[list[float]]:
        text = values[0]
        self.calls.append(text)
        if self.count_tokens(text) > self.accepted_tokens:
            raise ValueError("context length exceeded")
        return [[1.0, 0.0] if "relevant" in text else [0.0, 1.0]]


def _subsystem(embedder: object, *, budget: int) -> EmbedSubsystem:
    engine = SimpleNamespace(
        _ef=embedder,
        cached_embed=None,
        embedding_profile=EmbeddingProfile(
            provider="test",
            model="tokenizer",
            dimension=2,
            max_sequence_length=8192,
            crop_token_budget=budget,
            tokenizer_fingerprint="test-tokenizer",
            crop_policy="token_prefix",
        ),
        embedding_length_limit=512,
    )
    return EmbedSubsystem(engine)


@pytest.mark.parametrize("tokens", [7680, 8000, 8192])
def test_token_aware_crop_never_exceeds_configured_budget(tokens: int) -> None:
    embedder = _TokenEmbedder()
    result = _subsystem(embedder, budget=7680).iterative_defensive_emb_internal(
        " ".join(f"t{index}" for index in range(tokens))
    )

    assert result == [3.0, 4.0]
    assert embedder.count_tokens(embedder.calls[0]) <= 7680


def test_token_aware_retry_finds_largest_provider_accepted_prefix() -> None:
    embedder = _TokenEmbedder(accepted_tokens=4000)
    _subsystem(embedder, budget=7680).iterative_defensive_emb_internal(
        " ".join(f"t{index}" for index in range(7680))
    )

    assert embedder.count_tokens(embedder.calls[-1]) == 4000
    assert len(embedder.calls) > 2


def test_provider_without_tokenizer_keeps_character_fallback() -> None:
    calls: list[str] = []

    def embed(values: list[str]) -> list[list[float]]:
        calls.append(values[0])
        return [[1.0, 0.0]]

    result = _subsystem(embed, budget=7680).iterative_defensive_emb_internal("x" * 1000)

    assert result == [1.0, 0.0]
    assert len(calls[0]) == 515


def test_token_crop_preserves_relevant_content_inside_budget() -> None:
    embedder = _MarkerEmbedder()
    text = " ".join(f"t{index}" for index in range(7679)) + " relevant" + " after"

    result = _subsystem(embedder, budget=7680).iterative_defensive_emb_internal(text)

    assert result == [1.0, 0.0]
    assert embedder.count_tokens(embedder.calls[0]) == 7680
