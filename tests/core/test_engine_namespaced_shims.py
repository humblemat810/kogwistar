from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest

from graph_knowledge_engine.engine_core.engine import GraphKnowledgeEngine, _SHIM_METHOD_MAP


_SHIM_CASES = [
    ("add_node", "write", "add_node", ("n1",), {}),
    ("add_edge", "write", "add_edge", ("e1",), {}),
    ("add_document", "write", "add_document", ("d1",), {}),
    ("add_domain", "write", "add_domain", ("dom1",), {}),
    ("get_nodes", "read", "get_nodes", (["n1"],), {}),
    ("get_edges", "read", "get_edges", (["e1"],), {}),
    ("query_nodes", "read", "query_nodes", (), {"query": "q"}),
    ("query_edges", "read", "query_edges", (), {"query": "q"}),
    ("tombstone_node", "lifecycle", "tombstone_node", ("n1",), {}),
    ("redirect_node", "lifecycle", "redirect_node", ("n1", "n2"), {}),
    ("tombstone_edge", "lifecycle", "tombstone_edge", ("e1",), {}),
    ("redirect_edge", "lifecycle", "redirect_edge", ("e1", "e2"), {}),
    ("persist_graph_extraction", "persist", "persist_graph_extraction", (object(), "doc1"), {}),
    ("ingest_document_with_llm", "ingest", "ingest_document_with_llm", ("doc1",), {}),
    ("rollback_document", "rollback", "rollback_document", ("doc1",), {}),
]


@pytest.mark.parametrize("method_name,ns_name,ns_method,args,kwargs", _SHIM_CASES)
def test_non_conversation_shims_forward_and_warn(
    monkeypatch,
    method_name: str,
    ns_name: str,
    ns_method: str,
    args: tuple[object, ...],
    kwargs: dict[str, object],
):
    test_db_dir = Path.cwd() / ".tmp_shim_tests" / str(uuid.uuid4())
    test_db_dir.mkdir(parents=True, exist_ok=True)
    engine = GraphKnowledgeEngine(persist_directory=str(test_db_dir))

    namespace_obj = getattr(engine, ns_name)
    sentinel = object()
    seen: dict[str, object] = {}

    def _stub(*a, **k):
        seen["args"] = a
        seen["kwargs"] = k
        return sentinel

    monkeypatch.setattr(namespace_obj, ns_method, _stub, raising=False)

    with pytest.warns(DeprecationWarning, match=fr"GraphKnowledgeEngine\.{method_name} is deprecated"):
        out = getattr(engine, method_name)(*args, **kwargs)

    assert out is sentinel
    assert seen["args"] == args
    assert seen["kwargs"] == kwargs
    shutil.rmtree(test_db_dir, ignore_errors=True)


def test_shim_map_does_not_include_conversation_methods():
    assert "_get_conversation_tail" not in _SHIM_METHOD_MAP
    assert "create_conversation_primitive" not in _SHIM_METHOD_MAP
    assert "respond_to_utterance" not in _SHIM_METHOD_MAP
