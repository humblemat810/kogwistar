import json
import sqlite3
from pathlib import Path

from kogwistar.engine_core.search_index.models import (
    AddIndexEntriesInput,
    build_embedding_text,
    make_index_key,
)
from kogwistar.engine_core.search_index.service import SearchIndexService


def test_search_index_entries_match_shared_rust_golden_fixture() -> None:
    root = Path(__file__).resolve().parents[2]
    fixture = json.loads(
        (root / "contracts" / "golden" / "search-index-entry-v1.json").read_text(
            encoding="utf-8"
        )
    )
    request = AddIndexEntriesInput.model_validate(fixture["request"])
    assert len(request.index) == len(fixture["expected"])

    for item, expected in zip(request.index, fixture["expected"], strict=True):
        assert make_index_key(item.node_id, item.canonical_title, item.provision) == (
            expected["index_key"]
        )
        assert build_embedding_text(item) == expected["embedding_text"]


def test_hybrid_search_fusion_matches_shared_rust_golden_fixture() -> None:
    root = Path(__file__).resolve().parents[2]
    fixture = json.loads(
        (
            root / "contracts" / "golden" / "search-index-hybrid-v1.json"
        ).read_text(encoding="utf-8")
    )
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute(
        "CREATE TABLE lexical_hits (index_key TEXT,node_id TEXT,canonical_title TEXT,"
        "provision TEXT,document_id TEXT,fts_score REAL)"
    )
    connection.executemany(
        "INSERT INTO lexical_hits VALUES (?,?,?,?,?,?)",
        [
            (
                hit["index_key"],
                hit["node_id"],
                hit["canonical_title"],
                hit["provision"],
                hit["document_id"],
                hit["bm25_score"],
            )
            for hit in fixture["lexical_hits"]
        ],
    )
    lexical_rows = connection.execute("SELECT * FROM lexical_hits").fetchall()
    vector_hits = fixture["vector_hits"]
    vector_results = {
        "ids": [[hit["id"] for hit in vector_hits]],
        "metadatas": [[hit["metadata"] for hit in vector_hits]],
        "distances": [[hit["distance"] for hit in vector_hits]],
    }
    service = SearchIndexService.__new__(SearchIndexService)
    actual = service._fuse_hybrid_rows(lexical_rows, vector_results)

    assert [
        {key: hit[key] for key in fixture["expected"][0]}
        for hit in actual
    ] == fixture["expected"]
    connection.close()
