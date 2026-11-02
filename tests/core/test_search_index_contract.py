import json
from pathlib import Path

from kogwistar.engine_core.search_index.models import (
    AddIndexEntriesInput,
    build_embedding_text,
    make_index_key,
)


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
