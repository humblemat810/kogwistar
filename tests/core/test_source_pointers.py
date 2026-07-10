from __future__ import annotations

import pytest

from kogwistar.utils import SourcePointerValidationError, validate_source_pointer


def test_validate_source_pointer_accepts_exclusive_core_span() -> None:
    validated = validate_source_pointer(
        {
            "source_cluster_id": "cluster-1",
            "start_char": 0,
            "end_char": 5,
            "excerpt": "Alpha",
        },
        source_text_by_cluster={"cluster-1": "Alpha Beta"},
        end_mode="exclusive",
        require_source_cluster=True,
        require_source_text=True,
        require_text_match=True,
    )

    assert validated.start_char == 0
    assert validated.end_char == 5
    assert validated.slice_end_char == 5
    assert validated.text == "Alpha"


def test_validate_source_pointer_accepts_inclusive_parser_span_with_parent() -> None:
    validated = validate_source_pointer(
        {
            "source_cluster_id": "cluster-1",
            "start_char": 0,
            "end_char": 4,
            "verbatim_text": "Alpha",
        },
        source_text_by_cluster={"cluster-1": "Alpha Beta"},
        parent_pointers=[
            {
                "source_cluster_id": "cluster-1",
                "start_char": 0,
                "end_char": 9,
            }
        ],
        end_mode="inclusive",
        require_source_cluster=True,
        require_source_text=True,
        require_parent_containment=True,
        require_text_match=True,
    )

    assert validated.slice_end_char == 5


def test_validate_source_pointer_rejects_text_mismatch() -> None:
    with pytest.raises(SourcePointerValidationError) as exc_info:
        validate_source_pointer(
            {
                "source_cluster_id": "cluster-1",
                "start_char": 0,
                "end_char": 5,
                "excerpt": "Wrong",
            },
            source_text_by_cluster={"cluster-1": "Alpha Beta"},
            end_mode="exclusive",
            require_source_cluster=True,
            require_source_text=True,
            require_text_match=True,
        )

    assert exc_info.value.code == "text_mismatch"


def test_validate_source_pointer_rejects_outside_parent_span() -> None:
    with pytest.raises(SourcePointerValidationError) as exc_info:
        validate_source_pointer(
            {
                "source_cluster_id": "cluster-1",
                "start_char": 6,
                "end_char": 9,
                "verbatim_text": "Beta",
            },
            source_text_by_cluster={"cluster-1": "Alpha Beta"},
            parent_pointers=[
                {
                    "source_cluster_id": "cluster-1",
                    "start_char": 0,
                    "end_char": 4,
                }
            ],
            end_mode="inclusive",
            require_source_cluster=True,
            require_source_text=True,
            require_parent_containment=True,
            require_text_match=True,
        )

    assert exc_info.value.code == "outside_parent_span"
