from __future__ import annotations

from kogwistar.engine_core.engine_postgres_meta import _BufferedResult


def test_buffered_result_is_iterable():
    rows = [(1, "node", "n-1", "ADD", "{}"), (2, "edge", "e-1", "ADD", "{}")]
    result = _BufferedResult(rows)

    assert list(result) == rows
    assert result.fetchone() == rows[0]
    assert result.fetchall() == rows
