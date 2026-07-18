"""Shared source-identity hashing for ADR-015 wheel and compatibility evidence."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess


SOURCE_SUFFIXES = frozenset({".json", ".lock", ".py", ".rs", ".toml"})
SOURCE_PATHS = ("pyproject.toml", "rust", "kogwistar", "contracts")


def candidate_source_files(root: Path) -> list[Path]:
    if (root / ".git").exists():
        result = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
                "--",
                *SOURCE_PATHS,
            ],
            check=True,
            capture_output=True,
        )
        relative = sorted(
            value.decode("utf-8", errors="surrogateescape")
            for value in result.stdout.split(b"\0")
            if value
        )
        files = [
            root / value
            for value in relative
            if Path(value).suffix.lower() in SOURCE_SUFFIXES
        ]
    else:
        files = sorted(
            path
            for relative in SOURCE_PATHS
            for path in (
                [root / relative]
                if (root / relative).is_file()
                else (root / relative).rglob("*")
                if (root / relative).is_dir()
                else []
            )
            if path.is_file()
            and path.suffix.lower() in SOURCE_SUFFIXES
            and "__pycache__" not in path.parts
        )
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"candidate source input is missing: {missing[0]}")
    return files


def candidate_source_fingerprint(root: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    files = candidate_source_files(root)
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest(), len(files)


__all__ = [
    "SOURCE_PATHS",
    "SOURCE_SUFFIXES",
    "candidate_source_files",
    "candidate_source_fingerprint",
]
