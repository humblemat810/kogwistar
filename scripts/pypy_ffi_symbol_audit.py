"""Audit legacy PyO3 PyPy symbols against a PyPy installation.

This is diagnostic only. It does not patch PyO3 bindings or fail the job.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Iterable


_LINK_NAME = re.compile(r'link_name\s*=\s*"(PyPy[A-Za-z0-9_]+)"')


def _expected_symbols(root: Path) -> list[str]:
    symbols: set[str] = set()
    source_root = root / "src"
    if not source_root.is_dir():
        return []
    for path in source_root.rglob("*.rs"):
        symbols.update(_LINK_NAME.findall(path.read_text(encoding="utf-8")))
    return sorted(symbols)


def _library_candidates(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and (
            path.name == "pypy3"
            or path.name.startswith("libpypy")
            or path.name.startswith("pypy-c")
        )
    )


def _exported_symbols(paths: Iterable[Path]) -> tuple[list[str], str | None]:
    nm = shutil.which("nm")
    if nm is None:
        return [], "nm is not installed"
    exported: set[str] = set()
    errors: list[str] = []
    for path in paths:
        result = subprocess.run(
            [nm, "-D", "--defined-only", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            errors.append(f"{path}: {result.stderr.strip() or result.returncode}")
            continue
        for line in result.stdout.splitlines():
            fields = line.split()
            if fields:
                exported.add(fields[-1].split("@", 1)[0])
    return sorted(exported), "; ".join(errors) or None


def audit(*, pyo3_root: Path, pypy_root: Path) -> dict[str, object]:
    expected = _expected_symbols(pyo3_root)
    candidates = _library_candidates(pypy_root)
    exported, error = _exported_symbols(candidates)
    exported_set = set(exported)
    return {
        "pyo3_root": str(pyo3_root),
        "pypy_root": str(pypy_root),
        "expected_symbols": expected,
        "expected_count": len(expected),
        "library_candidates": [str(path) for path in candidates],
        "exported_count": len(exported),
        "missing_symbols": [symbol for symbol in expected if symbol not in exported_set],
        "nm_error": error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyo3-root", type=Path, default=Path(os.environ.get("PYO3_ROOT", "")))
    parser.add_argument("--pypy-root", type=Path, default=Path(os.environ.get("PYPY_HOME", "")))
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args()
    result = audit(pyo3_root=args.pyo3_root, pypy_root=args.pypy_root)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"PyO3 legacy PyPy symbols: {result['expected_count']}")
        print(f"PyPy library candidates: {len(result['library_candidates'])}")
        print(f"Exported dynamic symbols: {result['exported_count']}")
        print(f"Missing expected symbols: {len(result['missing_symbols'])}")
        if result["missing_symbols"]:
            print("missing:", ", ".join(result["missing_symbols"]))
        if result["nm_error"]:
            print("nm diagnostic:", result["nm_error"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
