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
