from __future__ import annotations

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


def test_published_release_version_matches_dl17_documentation() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    published_version = metadata["version"]
    support_doc = (ROOT / "docs" / "python-runtime-support.md").read_text(
        encoding="utf-8"
    )
    assert f"PyPI `kogwistar=={published_version}`" in support_doc
    assert f"Kogwistar {published_version} was published" in support_doc

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
        assert published_version in row, (
            f"{item_id} must reference published version {published_version}"
        )
