from __future__ import annotations

import importlib

import pytest

pytestmark = pytest.mark.ci


def test_get_embedding_function_uses_test_override(monkeypatch):
    monkeypatch.setenv(
        "KOGWISTAR_TEST_EMBEDDING_FUNCTION_IMPORT",
        "tests.conftest:build_default_test_embedding_function",
    )
    monkeypatch.setenv("KOGWISTAR_TEST_EMBEDDING_DIM", "5")

    embedding_factory = importlib.import_module(
        "kogwistar.engine_core.embedding_factory"
    )
    embedding_factory = importlib.reload(embedding_factory)
    emb = embedding_factory.get_embedding_function()

    vecs = emb(["hello"])
    assert len(vecs) == 1
    assert len(vecs[0]) == 5

