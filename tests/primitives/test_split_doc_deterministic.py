# test_split_doc_deterministic.py
#
# Pytest test cases for split_doc_deterministic() from your Pattern-1 implementation.
#
# Assumptions:
# - split_doc_deterministic(doc_id, content, max_chars, overlap_chars, prefer_window)
#   returns List[Chunk] where Chunk has fields: doc_id, start_char, end_char, text
# - end_char is exclusive
# - invariant: chunk.text == content[chunk.start_char:chunk.end_char]
#
# Adjust the import below to your module name.

import pytest
pytestmark = pytest.mark.ci
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from kogwistar.splitter import split_doc_deterministic


def _assert_basic_invariants(chunks, doc_id, content, max_chars, overlap_chars):
    assert chunks, "Should return at least one chunk"
    for c in chunks:
        assert c.doc_id == doc_id
        assert 0 <= c.start_char <= c.end_char <= len(content)
        assert c.text == content[c.start_char : c.end_char], (
            "Source-map invariant broken"
        )
        assert (c.end_char - c.start_char) <= max_chars or len(content) <= max_chars

    # Coverage: union of ranges should cover [0, len(content)] (allow overlaps)
    assert chunks[0].start_char == 0
    assert chunks[-1].end_char == len(content)

    # Monotonicity/progress (allow overlap so starts can go backwards, but ends should progress)
    ends = [c.end_char for c in chunks]
    assert all(e2 > e1 for e1, e2 in zip(ends, ends[1:])), (
        "Chunk ends must strictly increase"
    )

    # Overlap sanity (for adjacent chunks, overlap is at most overlap_chars, and >=0)
    for a, b in zip(chunks, chunks[1:]):
        overlap = max(0, a.end_char - b.start_char)
        assert overlap <= overlap_chars, (
            f"Overlap too large: {overlap} > {overlap_chars}"
        )


def test_empty_content_returns_single_empty_chunk():

    doc_id = "d0"
    chunks = split_doc_deterministic(
        doc_id=doc_id, content="", max_chars=10, overlap_chars=0
    )
    assert len(chunks) == 1
    c = chunks[0]
    assert c.doc_id == doc_id
    assert c.start_char == 0 and c.end_char == 0
    assert c.text == ""


def test_short_content_single_chunk():

    doc_id = "d1"
    content = "hello world"
    chunks = split_doc_deterministic(
        doc_id=doc_id, content=content, max_chars=50, overlap_chars=5, prefer_window=5
    )
    assert len(chunks) == 1
    chunks = split_doc_deterministic(
        doc_id=doc_id, content=content, max_chars=10, overlap_chars=4, prefer_window=5
    )
    assert len(chunks) == 2
    with pytest.raises(ValueError):
        split_doc_deterministic(
            doc_id=doc_id,
            content=content,
            max_chars=8,
            overlap_chars=4,
            prefer_window=4,
        )
    chunks = split_doc_deterministic(
        doc_id=doc_id, content=content, max_chars=9, overlap_chars=4, prefer_window=4
    )
    assert len(chunks) <= len(content)
    content = "hello world" * 20
    chunks = split_doc_deterministic(
        doc_id=doc_id, content=content, max_chars=22, overlap_chars=10, prefer_window=5
    )
    chunks = split_doc_deterministic(
        doc_id=doc_id, content=content, max_chars=22, overlap_chars=10, prefer_window=11
    )
    # assert chunks[0].text == content
    _assert_basic_invariants(chunks, doc_id, content, max_chars=22, overlap_chars=10)


def test_raises_when_overlap_ge_max_chars():

    with pytest.raises(ValueError):
        split_doc_deterministic(
            doc_id="d", content="abcdef", max_chars=10, overlap_chars=10
        )
    with pytest.raises(ValueError):
        split_doc_deterministic(
            doc_id="d", content="abcdef", max_chars=10, overlap_chars=11
        )


def test_deterministic_same_input_same_chunks():

    doc_id = "d2"
    content = (
        "para1 line1\npara1 line2\n\n"
        "para2 line1\npara2 line2\n\n"
        "para3 line1\npara3 line2\n"
    ) * 5

    kwargs = dict(max_chars=80, overlap_chars=10, prefer_window=30)
    c1 = split_doc_deterministic(doc_id=doc_id, content=content, **kwargs)
    c2 = split_doc_deterministic(doc_id=doc_id, content=content, **kwargs)

    assert [(c.start_char, c.end_char, c.text) for c in c1] == [
        (c.start_char, c.end_char, c.text) for c in c2
    ]


def test_prefers_double_newline_boundary_when_available():
    """
    Ensure it prefers '\\n\\n' break when the window contains it near hard limit.
    """

    doc_id = "d3"
    # Put a double newline within prefer_window before max_chars boundary.
    content = "A" * 40 + "\n\n" + "B" * 200
    chunks = split_doc_deterministic(
        doc_id=doc_id, content=content, max_chars=60, overlap_chars=0, prefer_window=50
    )

    # First chunk should end right after the "\n\n" (index 40..42)
    assert chunks[0].end_char == 42
    assert chunks[0].text.endswith("\n\n")
    _assert_basic_invariants(chunks, doc_id, content, max_chars=60, overlap_chars=0)


def test_prefers_single_newline_when_no_double_newline():

    doc_id = "d4"
    content = "A" * 40 + "\n" + "B" * 200
    chunks = split_doc_deterministic(
        doc_id=doc_id, content=content, max_chars=60, overlap_chars=0, prefer_window=50
    )

    # First chunk should end right after '\n' at index 40 (so end_char=41)
    assert chunks[0].end_char == 41
    assert chunks[0].text.endswith("\n")
    _assert_basic_invariants(chunks, doc_id, content, max_chars=60, overlap_chars=0)


def test_prefers_whitespace_boundary():

    doc_id = "d5"
    content = "word " * 50  # many spaces
    chunks = split_doc_deterministic(
        doc_id=doc_id, content=content, max_chars=60, overlap_chars=0, prefer_window=30
    )

    # Expect first chunk ends at a space boundary (end_char points after space).
    # So last char should be whitespace or the chunk ends exactly at limit if none.
    assert chunks[0].text[-1].isspace()
    _assert_basic_invariants(chunks, doc_id, content, max_chars=60, overlap_chars=0)


def test_falls_back_to_hard_cut_when_no_boundaries():

    doc_id = "d6"
    content = "X" * 200  # no whitespace, no punctuation, no newlines
    chunks = split_doc_deterministic(
        doc_id=doc_id, content=content, max_chars=50, overlap_chars=0, prefer_window=30
    )

    assert len(chunks) == 4
    assert chunks[0].end_char == 50
    assert chunks[1].start_char == 50
    assert chunks[1].end_char == 100
    _assert_basic_invariants(chunks, doc_id, content, max_chars=50, overlap_chars=0)


def test_overlap_behavior_basic():

    doc_id = "d7"
    content = "A" * 120
    chunks = split_doc_deterministic(
        doc_id=doc_id, content=content, max_chars=50, overlap_chars=10, prefer_window=10
    )

    _assert_basic_invariants(chunks, doc_id, content, max_chars=50, overlap_chars=10)

    # Adjacent chunks should overlap by exactly 10 in the hard-cut case
    # because there are no boundaries.
    for a, b in zip(chunks, chunks[1:]):
        assert a.end_char - b.start_char == 10


def test_multilingual_cjk_punctuation_preference():
    """
    Ensure the splitter can prefer CJK punctuation when present in the window.
    """

    doc_id = "d8"
    # Put a CJK sentence terminator near the boundary.
    content = "这是第一句。" + ("中" * 80) + "这是第二句！" + ("文" * 80)
    chunks = split_doc_deterministic(
        doc_id=doc_id, content=content, max_chars=90, overlap_chars=0, prefer_window=50
    )

    # First chunk should likely end shortly after the first CJK punctuation ("。")
    # because it's within the window before the max_chars point.
    assert "。" in chunks[0].text or "！" in chunks[0].text
    _assert_basic_invariants(chunks, doc_id, content, max_chars=90, overlap_chars=0)


def test_no_infinite_loop_when_overlap_large_but_valid():
    """
    Regression: ensure no infinite loops with big overlap (but still < max_chars).
    """

    doc_id = "d9"
    content = "X" * 500
    chunks = split_doc_deterministic(
        doc_id=doc_id, content=content, max_chars=100, overlap_chars=99, prefer_window=0
    )

    # Should still progress and terminate.
    assert len(chunks) > 1
    _assert_basic_invariants(chunks, doc_id, content, max_chars=100, overlap_chars=99)


def test_round_trip_reconstructs_original_when_stitching_nonoverlap_portions():
    """
    Reconstruct original by taking first chunk fully, then for each subsequent chunk,
    append only the non-overlapping suffix. Should match the original exactly.
    """

    doc_id = "d10"
    content = ("para1\n\n" + "A" * 80 + "\n\n" + "para2\n\n" + "B" * 80 + "\n\n") * 3
    overlap = 20
    chunks = split_doc_deterministic(
        doc_id=doc_id,
        content=content,
        max_chars=90,
        overlap_chars=overlap,
        prefer_window=40,
    )

    rebuilt = chunks[0].text
    for prev, cur in zip(chunks, chunks[1:]):
        ov = max(0, prev.end_char - cur.start_char)
        rebuilt += cur.text[ov:]  # drop overlap part
    assert rebuilt == content
