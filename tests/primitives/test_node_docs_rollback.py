# tests/test_node_docs_rollback.py
import pytest

from kogwistar.engine_core.models import Node

from tests._kg_factories import kg_document, kg_grounding
from tests.conftest import _make_engine_pair


@pytest.mark.parametrize("backend_kind", ["chroma", "pg"])
def test_node_docs_partial_then_full_rollback(
    backend_kind: str, tmp_path, sa_engine, pg_schema
):
    eng, _ = _make_engine_pair(
        backend_kind=backend_kind,
        tmp_path=tmp_path,
        sa_engine=sa_engine,
        pg_schema=pg_schema,
        dim=3,
        use_fake=True,
    )
    d1 = kg_document(
        doc_id="doc::test_node_docs_partial_then_full_rollback::1",
        content="D1",
        source="test_node_docs_partial_then_full_rollback",
    )
    d2 = kg_document(
        doc_id="doc::test_node_docs_partial_then_full_rollback::2",
        content="D2",
        source="test_node_docs_partial_then_full_rollback",
    )
    eng.write.add_document(d1)
    eng.write.add_document(d2)

    # One node with evidence in *both* documents (no single doc_id in node meta)
    n = Node(
        label="Shared",
        type="entity",
        summary="x",
        mentions=[kg_grounding(d1.id), kg_grounding(d2.id)],
    )
    eng.write.add_node(n)  # no doc_id passed; relies on references + node_docs

    # Sanity: node_docs has two rows
    rows = eng.backend.node_docs_get(where={"node_id": n.id}, include=["metadatas"])
    assert {m["doc_id"] for m in rows.get("metadatas") or []} == {d1.id, d2.id}

    # Rollback d1 only: old node is tombstoned and redirected to a new active node with only d2 evidence.
    res1 = eng.rollback_document(d1.id)
    assert isinstance(res1, dict)
    replacement_id = res1["node_redirects"][n.id]

    old_node = eng.backend.node_get(ids=[n.id], include=["metadatas"])
    assert old_node["ids"] == [n.id]
    assert old_node["metadatas"][0]["lifecycle_status"] == "tombstoned"
    assert old_node["metadatas"][0]["redirect_to_id"] == replacement_id
    redirected = eng.get_nodes(ids=[n.id], resolve_mode="redirect")
    assert [item.id for item in redirected] == [replacement_id]

    old_rows = eng.backend.node_docs_get(where={"node_id": n.id}, include=["metadatas"])
    assert not (old_rows.get("ids") or [])

    rows_after = eng.backend.node_docs_get(
        where={"node_id": replacement_id}, include=["metadatas"]
    )
    assert {m["doc_id"] for m in rows_after.get("metadatas") or []} == {d2.id}

    n_got = eng.backend.node_get(ids=[replacement_id], include=["documents"])
    node_json = n_got["documents"][0]
    node = Node.model_validate_json(node_json)
    assert all(sp.doc_id != d1.id for mention in node.mentions for sp in mention.spans)

    # Rollback d2: replacement node is tombstoned without a further replacement.
    res2 = eng.rollback_document(d2.id)
    assert isinstance(res2, dict)

    replacement_node = eng.backend.node_get(ids=[replacement_id], include=["metadatas"])
    assert replacement_node["ids"] == [replacement_id]
    assert replacement_node["metadatas"][0]["lifecycle_status"] == "tombstoned"
    assert replacement_node["metadatas"][0].get("redirect_to_id") is None

    final_rows = eng.backend.node_docs_get(
        where={"node_id": replacement_id}, include=["metadatas"]
    )
    assert not (final_rows.get("ids") or [])
