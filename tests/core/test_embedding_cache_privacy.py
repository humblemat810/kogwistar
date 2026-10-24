from __future__ import annotations

from pathlib import Path

import pytest

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.in_memory_backend import build_in_memory_backend


class CountingEmbedding:
    def __init__(self) -> None:
        self.calls = 0

    @staticmethod
    def name() -> str:
        return "test-embedding-snapshot-v1"

    def __call__(self, values: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[1.0, 0.0] for _ in values]


def test_embedding_result_cache_does_not_log_or_persist_input_text(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KOGWISTAR_CACHE_BACKEND", "joblib")
    secret_input = "private-document-excerpt-should-not-be-cached-in-plain-text"
    cache_path = tmp_path / "embedding-cache"
    embedding = CountingEmbedding()
    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "graph"),
        embedding_cache_path=str(cache_path),
        embedding_function=embedding,
        backend_factory=build_in_memory_backend,
    )
    try:
        first = engine.embed.iterative_defensive_emb(secret_input)
        calls_after_first_embedding = embedding.calls
        second = engine.embed.iterative_defensive_emb(secret_input)

        assert first == second
        assert calls_after_first_embedding > 0
        assert embedding.calls == calls_after_first_embedding
        assert engine.cached_embed is not None
        assert engine.cached_embed.check_call_in_cache(secret_input) is True  # type: ignore[attr-defined]
        assert secret_input not in capsys.readouterr().out

        metadata_files = list(cache_path.rglob("metadata.json"))
        assert metadata_files
        assert all(secret_input not in path.read_text(encoding="utf-8") for path in metadata_files)
    finally:
        engine.close()
