from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

pytestmark = [pytest.mark.ci, pytest.mark.core]

ROOT = Path(__file__).resolve().parents[2]


def test_package_metadata_matches_the_supported_interpreter_matrix() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]

    assert metadata["requires-python"] == ">=3.11"
    classifiers = set(metadata["classifiers"])
    for minor in ("3.11", "3.12", "3.13", "3.14"):
        assert f"Programming Language :: Python :: {minor}" in classifiers


def test_release_candidate_version_matches_dl17_documentation() -> None:
    support_doc = (ROOT / "docs" / "python-runtime-support.md").read_text(
        encoding="utf-8"
    )
    candidate_match = re.search(
        r"The (\d+\.\d+\.\d+) release note describes an unpublished candidate",
        support_doc,
    )
    assert candidate_match is not None
    candidate_version = candidate_match.group(1)

    worklist = (
        ROOT
        / "kogwistar"
        / "docs"
        / "ADR-022-dual-language-capability-worklist.md"
    ).read_text(encoding="utf-8")
    item_ids = (
        "FW-17-03B",
        "FW-17-04B-DOCS",
        "FW-17-02B-METADATA",
        "FW-17-02B-DOCS",
    )
    for item_id in item_ids:
        row = next(
            line for line in worklist.splitlines() if line.startswith(f"| {item_id} |")
        )
        assert candidate_version in row, (
            f"{item_id} must reference support-doc candidate {candidate_version}"
        )
