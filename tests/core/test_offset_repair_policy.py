from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Sequence
import pytest

pytestmark = pytest.mark.ci

from kogwistar.engine_core.engine import GraphKnowledgeEngine

class EmbeddingFunction:  # type: ignore
    @staticmethod
    def name() -> str:
        return "default"

Embeddings = list[list[float]]  # type: ignore


class FakeEmbeddingFunction(EmbeddingFunction):
    @staticmethod
    def name() -> str:
        return "default"

    def __init__(self, dim: int = 384):
        self._dim = dim

    def __call__(self, input: Sequence[str]) -> Embeddings:
        return [[0.01] * self._dim for _ in input]
class FakeBackend:
    def __init__(self, engine):
        self._engine = engine

    def node_get(self, where=None, ids=None, include=None):
        nodes = self._engine.nodes
        if ids is not None:
            selected = [node for node in nodes if node.id in ids]
        else:
            selected = [node for node in nodes if _matches_where(node, where)]
        return {
            "ids": [node.id for node in selected],
            "documents": [getattr(node, "summary", None) for node in selected],
            "metadatas": [getattr(node, "metadata", None) for node in selected],
            "embeddings": [getattr(node, "embedding", None) for node in selected],
            "objects": selected,
        }

@pytest.fixture
def engine():
    run_id = uuid.uuid4().hex
    base = Path(".tmp_pytest_offset_policy")
    base.mkdir(parents=True, exist_ok=True)
    persist_dir = base / run_id
    persist_dir.mkdir(parents=True, exist_ok=True)
    eng = GraphKnowledgeEngine(
        persist_directory=str(persist_dir),
        embedding_cache_path=str(persist_dir / "emb_cache"),
        embedding_function=FakeEmbeddingFunction(),
        backend_factory = FakeBackend
    )
    try:
        yield eng
    finally:
        shutil.rmtree(persist_dir, ignore_errors=True)


def _lean_payload(*, excerpt: str, start_char: int, end_char: int) -> dict:
    return {
        "nodes": [
            {
                "local_id": "nn:a",
                "label": "A",
                "type": "entity",
                "summary": "Entity A",
                "mentions": [
                    {
                        "spans": [
                            {
                                "page_number": 1,
                                "start_char": start_char,
                                "end_char": end_char,
                                "excerpt": excerpt,
                            }
                        ]
                    }
                ],
            }
        ],
        "edges": [],
    }


def _flattened_lean_payload(*, excerpt: str, start_char: int, end_char: int) -> dict:
    return {
        "spans": [
            {
                "id": "sp:1",
                "page_number": 1,
                "start_char": start_char,
                "end_char": end_char,
                "excerpt": excerpt,
            }
        ],
        "nodes": [
            {
                "local_id": "nn:a",
                "label": "A",
                "type": "entity",
                "summary": "Entity A",
            }
        ],
        "edges": [],
        "groundings": [{"id": "gr:1"}],
        "node_groundings": [{"node_index": 0, "grounding_id": "gr:1"}],
        "edge_groundings": [],
        "grounding_spans": [{"grounding_id": "gr:1", "span_id": "sp:1"}],
    }


def _full_payload_with_mismatched_excerpt(
    *, excerpt: str, start_char: int, end_char: int
) -> dict:
    return {
        "nodes": [
            {
                "local_id": "nn:a",
                "label": "A",
                "type": "entity",
                "summary": "Entity A",
                "mentions": [
                    {
                        "spans": [
                            {
                                "collection_page_url": "document_collection/_DOC_ALIAS",
                                "document_page_url": "document/_DOC_ALIAS",
                                "doc_id": "_DOC_ALIAS",
                                "page_number": 1,
                                "start_char": start_char,
                                "end_char": end_char,
                                "excerpt": excerpt,
                                "context_before": "",
                                "context_after": "",
                            }
                        ]
                    }
                ],
            }
        ],
        "edges": [],
    }


def _flattened_full_payload_with_mismatched_excerpt(
    *, excerpt: str, start_char: int, end_char: int
) -> dict:
    return {
        "spans": [
            {
                "id": "sp:1",
                "collection_page_url": "document_collection/_DOC_ALIAS",
                "document_page_url": "document/_DOC_ALIAS",
                "doc_id": "_DOC_ALIAS",
                "page_number": 1,
                "start_char": start_char,
                "end_char": end_char,
                "excerpt": excerpt,
                "context_before": "",
                "context_after": "",
            }
        ],
        "nodes": [
            {
                "local_id": "nn:a",
                "label": "A",
                "type": "entity",
                "summary": "Entity A",
            }
        ],
        "edges": [],
        "groundings": [{"id": "gr:1"}],
        "node_groundings": [{"node_index": 0, "grounding_id": "gr:1"}],
        "edge_groundings": [],
        "grounding_spans": [{"grounding_id": "gr:1", "span_id": "sp:1"}],
    }


def test_offset_policy_strict_fails_for_lean_mismatch(engine):
    payload = _lean_payload(excerpt="world", start_char=0, end_char=5)
    content = "hello world"

    with pytest.raises(ValueError, match="mode=lean policy=strict"):
        engine._to_canonical_extraction_for_mode(
            mode="lean",
            parsed=payload,
            content=content,
            offset_mismatch_policy="strict",
        )


def test_offset_policy_exact_repairs_lean_offsets(engine):
    payload = _lean_payload(excerpt="world", start_char=0, end_char=5)
    content = "hello world"

    graph = engine._to_canonical_extraction_for_mode(
        mode="lean",
        parsed=payload,
        content=content,
        offset_mismatch_policy="exact",
    )
    span = graph.nodes[0].mentions[0].spans[0]
    assert span.start_char == 6
    assert span.end_char == 11
    assert span.excerpt == "world"


def test_offset_policy_exact_repairs_flattened_lean_offsets(engine):
    payload = _flattened_lean_payload(excerpt="world", start_char=0, end_char=5)
    content = "hello world"

    graph = engine._to_canonical_extraction_for_mode(
        mode="flattened_lean",
        parsed=payload,
        content=content,
        offset_mismatch_policy="exact",
    )
    span = graph.nodes[0].mentions[0].spans[0]
    assert span.start_char == 6
    assert span.end_char == 11
    assert span.excerpt == "world"


def test_offset_policy_strict_fails_for_flattened_lean_mismatch(engine):
    payload = _flattened_lean_payload(excerpt="world", start_char=0, end_char=5)
    content = "hello world"

    with pytest.raises(ValueError, match="mode=flattened_lean policy=strict"):
        engine._to_canonical_extraction_for_mode(
            mode="flattened_lean",
            parsed=payload,
            content=content,
            offset_mismatch_policy="strict",
        )


def test_offset_policy_exact_fuzzy_repairs_when_exact_missing(engine):
    payload = _lean_payload(excerpt="Proof-step one", start_char=5, end_char=19)
    content = "0123 Proof step one 7890"

    graph = engine._to_canonical_extraction_for_mode(
        mode="lean",
        parsed=payload,
        content=content,
        offset_mismatch_policy="exact_fuzzy",
    )
    span = graph.nodes[0].mentions[0].spans[0]
    assert span.start_char == 5
    assert span.end_char == 19
    assert span.excerpt == "Proof step one"
    assert content[span.start_char : span.end_char] == span.excerpt


def test_offset_policy_exact_fuzzy_repairs_flattened_lean_when_exact_missing(engine):
    payload = _flattened_lean_payload(
        excerpt="Proof-step one", start_char=5, end_char=19
    )
    content = "0123 Proof step one 7890"

    graph = engine._to_canonical_extraction_for_mode(
        mode="flattened_lean",
        parsed=payload,
        content=content,
        offset_mismatch_policy="exact_fuzzy",
    )
    span = graph.nodes[0].mentions[0].spans[0]
    assert span.start_char == 5
    assert span.end_char == 19
    assert span.excerpt == "Proof step one"
    assert content[span.start_char : span.end_char] == span.excerpt


def test_offset_policy_exact_fuzzy_prefers_per_call_scorer(engine):
    payload = _lean_payload(excerpt="Proof-step one", start_char=5, end_char=19)
    content = "0123 Proof step one 7890"

    def _engine_level_scorer(candidate: str, excerpt: str) -> float:
        raise AssertionError(
            "engine-level scorer should not be used when per-call scorer is provided"
        )

    calls: list[tuple[str, str]] = []

    def _per_call_scorer(candidate: str, excerpt: str) -> float:
        calls.append((candidate, excerpt))
        return 100.0 if candidate == "Proof step one" else 0.0

    engine.offset_repair_scorer = _engine_level_scorer

    graph = engine._to_canonical_extraction_for_mode(
        mode="lean",
        parsed=payload,
        content=content,
        offset_mismatch_policy="exact_fuzzy",
        offset_repair_scorer=_per_call_scorer,
    )
    assert calls, "per-call scorer was not invoked"
    span = graph.nodes[0].mentions[0].spans[0]
    assert span.excerpt == "Proof step one"


def test_offset_policy_exact_fuzzy_reports_actionable_failure(engine):
    payload = _lean_payload(
        excerpt="THIS STRING DOES NOT EXIST ANYWHERE",
        start_char=0,
        end_char=10,
    )
    content = "short content only"

    with pytest.raises(ValueError) as excinfo:
        engine._to_canonical_extraction_for_mode(
            mode="lean",
            parsed=payload,
            content=content,
            offset_mismatch_policy="exact_fuzzy",
        )
    message = str(excinfo.value)
    assert "mode=lean policy=exact_fuzzy" in message
    assert "nodes[0].mentions[0].spans[0]" in message
    assert "failed_spans=1" in message


def test_offset_policy_exact_fuzzy_reports_actionable_failure_for_flattened_lean(
    engine,
):
    payload = _flattened_lean_payload(
        excerpt="THIS STRING DOES NOT EXIST ANYWHERE",
        start_char=0,
        end_char=10,
    )
    content = "short content only"

    with pytest.raises(ValueError) as excinfo:
        engine._to_canonical_extraction_for_mode(
            mode="flattened_lean",
            parsed=payload,
            content=content,
            offset_mismatch_policy="exact_fuzzy",
        )
    message = str(excinfo.value)
    assert "mode=flattened_lean policy=exact_fuzzy" in message
    assert "spans[0]" in message
    assert "failed_spans=1" in message


def test_offset_policy_does_not_change_full_mode_behavior(engine):
    payload = _full_payload_with_mismatched_excerpt(
        excerpt="world",
        start_char=0,
        end_char=5,
    )
    content = "hello world"

    graph = engine._to_canonical_extraction_for_mode(
        mode="full",
        parsed=payload,
        content=content,
        offset_mismatch_policy="strict",
    )
    span = graph.nodes[0].mentions[0].spans[0]
    assert span.start_char == 0
    assert span.end_char == 5
    assert span.excerpt == "world"


def test_offset_policy_does_not_change_flattened_full_mode_behavior(engine):
    payload = _flattened_full_payload_with_mismatched_excerpt(
        excerpt="world",
        start_char=0,
        end_char=5,
    )
    content = "hello world"

    graph = engine._to_canonical_extraction_for_mode(
        mode="flattened_full",
        parsed=payload,
        content=content,
        offset_mismatch_policy="strict",
    )
    span = graph.nodes[0].mentions[0].spans[0]
    assert span.start_char == 0
    assert span.end_char == 5
    assert span.excerpt == "world"
