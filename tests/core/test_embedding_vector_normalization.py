from __future__ import annotations

import pytest

pytestmark = pytest.mark.ci

from kogwistar.utils.embedding_vectors import (
    normalize_embedding_rows,
    normalize_embedding_vector,
)


class _ListLikeVector:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return list(self._values)


class _ListLikeMatrix:
    def __init__(self, rows):
        self._rows = rows

    def tolist(self):
        return [list(row) for row in self._rows]


def test_normalize_embedding_vector_accepts_plain_list():
    assert normalize_embedding_vector([1, 2.5, 3]) == [1.0, 2.5, 3.0]


def test_normalize_embedding_vector_accepts_tolist_provider_output():
    assert normalize_embedding_vector(_ListLikeVector([0.1, 0.2])) == [0.1, 0.2]


def test_normalize_embedding_rows_wraps_single_vector():
    assert normalize_embedding_rows([1, 2, 3], allow_empty=False) == [[1.0, 2.0, 3.0]]


def test_normalize_embedding_rows_accepts_tolist_matrix():
    assert normalize_embedding_rows(
        _ListLikeMatrix([[1, 2], [3, 4]]), allow_empty=False
    ) == [[1.0, 2.0], [3.0, 4.0]]
