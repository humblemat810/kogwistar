import json
import importlib
import pytest
pytestmark = pytest.mark.ci

from kogwistar import shortids


@pytest.fixture(autouse=True)
def in_tmpdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    yield


@pytest.fixture
def fresh_run(monkeypatch):
    # fresh mapper cache & token
    shortids._MAPPERS.clear()
    shortids.set_current_token("jwt-A")
    shortids.set_shortid_obj_depth(1)
    shortids.set_shortid_keys(
        scalars=("id", "doc_id", "node_id", "node_ref_id", "edge_endpoint_id"),
        lists=("source_ids", "target_ids", "source_edge_ids", "target_edge_ids"),
    )


def test_id_roundtrip_and_reject_non_sid(fresh_run):
    LONG1 = "ANY-LONG-FORM-001"
    LONG2 = "42"  # even a number-like string counts as “long” (server→user)

    s1 = shortids.l2s_id(LONG1)
    s2 = shortids.l2s_id(LONG2)
    assert s1 == "<sid>1"
    assert s2 == "<sid>2"
    assert shortids.l2s_id(LONG1) == "<sid>1"  # stable

    assert shortids.s2l_id("<sid>1") == LONG1
    assert shortids.s2l_id("<sid>2") == LONG2

    with pytest.raises(ValueError):  # unknown short id
        shortids.s2l_id("<sid>999")

    with pytest.raises(ValueError):  # any non-<sid> in id field is rejected
        shortids.s2l_id(LONG1)


def test_obj_depth_and_lists(fresh_run):
    A = "NODE-A-LONG"
    B = "NODE-B-LONG"

    obj = {
        "id": A,
        "source_ids": [B],
        "meta": {"id": B},  # nested: ignored at default depth=1
    }

    out = shortids.l2s_obj(obj)
    assert out["id"] == "<sid>1"
    assert out["source_ids"] == ["<sid>2"]
    assert out["meta"]["id"] == B  # untouched at depth=1

    shortids.set_shortid_obj_depth(2)
    out2 = shortids.l2s_obj(obj)
    assert out2["meta"]["id"] == "<sid>2"

    # bad user input: non-<sid> in targeted field should raise
    with pytest.raises(ValueError):
        shortids.s2l_obj({"id": "NOT-SID"})

    back = shortids.s2l_obj({"id": "<sid>1", "source_ids": ["<sid>2"]})
    assert back["id"] == A
    assert back["source_ids"] == [B]


def test_pydantic_model(fresh_run):
    from pydantic import BaseModel

    # Define simple models that use targeted keys
    class Edge(BaseModel):
        id: str
        source_ids: list[str] = []
        target_edge_ids: list[str] = []

    class Node(BaseModel):
        node_id: str
        node_ref_id: str
        edge: Edge
        meta: dict = {}

    A = "NODE-LONG-A"
    B = "NODE-LONG-B"
    C = "EDGE-LONG-C"

    node = Node(
        node_id=A,
        node_ref_id=B,
        edge=Edge(id=C, source_ids=[A, B], target_edge_ids=[C]),
        meta={"id": B},  # nested scalar id (should not change at depth=1)
    )

    # Default depth = 1 (top-level only): nested edge/meta remain unchanged
    out = shortids.l2s_obj(node)
    assert out["node_id"] == "<sid>1"
    assert out["node_ref_id"] == "<sid>2"
    # Nested edge stays long at depth=1
    assert out["edge"]["id"] == C
    assert out["edge"]["source_ids"] == [A, B]
    assert out["edge"]["target_edge_ids"] == [C]
    # meta.id also remains long at depth=1
    assert out["meta"]["id"] == B

    # Increase depth so nested fields are processed
    shortids.set_shortid_obj_depth(2)
    out2 = shortids.l2s_obj(node)
    # Top-level still short
    assert out2["node_id"] == "<sid>1"
    assert out2["node_ref_id"] == "<sid>2"
    # Nested edge converted at depth=2
    assert out2["edge"]["id"] == "<sid>3"
    assert out2["edge"]["source_ids"] == ["<sid>1", "<sid>2"]
    assert out2["edge"]["target_edge_ids"] == ["<sid>3"]
    # meta.id converted at depth=2
    assert out2["meta"]["id"] == "<sid>2"

    # Now test user → server with a Pydantic instance carrying short IDs
    node_short = Node(
        node_id=out2["node_id"],
        node_ref_id=out2["node_ref_id"],
        edge=Edge(
            id=out2["edge"]["id"],
            source_ids=out2["edge"]["source_ids"],
            target_edge_ids=out2["edge"]["target_edge_ids"],
        ),
        meta={"id": out2["meta"]["id"]},
    )

    back = shortids.s2l_obj(node_short)
    assert back["node_id"] == A
    assert back["node_ref_id"] == B
    assert back["edge"]["id"] == C
    assert back["edge"]["source_ids"] == [A, B]
    assert back["edge"]["target_edge_ids"] == [C]
    assert back["meta"]["id"] == B


def test_doc_json_paths(fresh_run):
    A = "EDGE-AAA"
    B = "EDGE-BBB"

    doc = json.dumps(
        {
            "id": A,
            "node_ref_id": B,
            "source_ids": [A, B],
            "target_edge_ids": [B],
            "content": "untouched text; ids only change in targeted keys",
        }
    )

    doc_short = shortids.l2s_doc(doc)
    d = json.loads(doc_short)
    assert d["id"] == "<sid>1"
    assert d["node_ref_id"] == "<sid>2"
    assert d["source_ids"] == ["<sid>1", "<sid>2"]
    assert d["target_edge_ids"] == ["<sid>2"]
    assert d["content"].startswith("untouched")

    doc_long = shortids.s2l_doc(doc_short)
    d2 = json.loads(doc_long)
    assert d2["id"] == A
    assert d2["node_ref_id"] == B
    assert d2["source_ids"] == [A, B]
    assert d2["target_edge_ids"] == [B]

    # non-JSON doc is returned unchanged
    assert shortids.l2s_doc("not json") == "not json"
    assert shortids.s2l_doc("not json") == "not json"


def test_persistence_across_instances(in_tmpdir):
    token = "jwt-PERSIST"
    long_val = "WHATEVER-LONG"
    shortids._MAPPERS.clear()
    shortids.set_current_token(token)
    assert shortids.l2s_id(long_val) == "<sid>1"

    # simulate new instance
    shortids._MAPPERS.clear()
    importlib.reload(shortids)
    shortids.set_current_token(token)
    assert shortids.s2l_id("<sid>1") == long_val

    # ensure numbering continues
    assert shortids.l2s_id("SECOND") == "<sid>2"
