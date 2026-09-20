"""Audit the Python ABI required by built PyPy native extensions.

The binary audit is the primary compatibility signal. The optional PyO3
source scan is retained only as supplementary context because source-level
bindings do not prove that a particular extension references those symbols.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Iterable, Sequence


_LINK_NAME = re.compile(r'link_name\s*=\s*"(PyPy[A-Za-z0-9_]+)"')
_PYTHON_ABI_SYMBOL = re.compile(r"^_?Py")


def _normalize_symbol(symbol: str) -> str:
    """Remove ELF symbol-version suffixes from an `nm` symbol."""

    return symbol.split("@", 1)[0]


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


def _nm_symbols(
    paths: Iterable[Path], *, mode: str
) -> tuple[set[str], list[str]]:
    nm = shutil.which("nm")
    if nm is None:
        return set(), ["nm is not installed"]
    symbols: set[str] = set()
    errors: list[str] = []
    for path in paths:
        result = subprocess.run(
            [nm, "-D", mode, str(path)],
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
                symbols.add(_normalize_symbol(fields[-1]))
    return symbols, errors


def _exported_symbols(paths: Iterable[Path]) -> tuple[list[str], str | None]:
    exported, errors = _nm_symbols(paths, mode="--defined-only")
    return sorted(exported), "; ".join(errors) or None


def _python_abi_symbols(symbols: Iterable[str]) -> set[str]:
    return {symbol for symbol in symbols if _PYTHON_ABI_SYMBOL.match(symbol)}


def _undefined_python_symbols(path: Path) -> tuple[set[str], list[str]]:
    undefined, errors = _nm_symbols([path], mode="--undefined-only")
    return _python_abi_symbols(undefined), errors


def audit_extension(path: Path, runtime_paths: Sequence[Path]) -> dict[str, object]:
    """Compare one finished extension's Python ABI requirements to PyPy."""

    required, required_errors = _undefined_python_symbols(path)
    exported, exported_errors = _nm_symbols(runtime_paths, mode="--defined-only")
    runtime_python = _python_abi_symbols(exported)
    errors = required_errors + exported_errors
    return {
        "path": str(path),
        "required_python_symbols": sorted(required),
        "required_python_symbol_count": len(required),
        "runtime_python_symbols": sorted(runtime_python),
        "runtime_python_symbol_count": len(runtime_python),
        "missing_python_symbols": sorted(required - runtime_python),
        "nm_errors": errors,
        "ok": not errors and not (required - runtime_python),
    }


def audit_source(pyo3_root: Path, runtime_paths: Sequence[Path]) -> dict[str, object]:
    """Return source-level context without treating it as binary evidence."""

    expected = _expected_symbols(pyo3_root)
    exported, error = _exported_symbols(runtime_paths)
    exported_set = set(exported)
    return {
        "pyo3_root": str(pyo3_root),
        "expected_symbols": expected,
        "expected_count": len(expected),
        "missing_symbols": [symbol for symbol in expected if symbol not in exported_set],
        "nm_error": error,
    }


def audit(
    *,
    extensions: Sequence[Path],
    pypy_root: Path,
    pyo3_root: Path | None = None,
) -> dict[str, object]:
    runtime_paths = _library_candidates(pypy_root)
    extension_results = [audit_extension(path, runtime_paths) for path in extensions]
    result: dict[str, object] = {
        "pypy_root": str(pypy_root),
        "runtime_paths": [str(path) for path in runtime_paths],
        "extensions": extension_results,
        "binary_abi_ok": bool(extension_results)
        and all(bool(item["ok"]) for item in extension_results),
    }
    if pyo3_root is not None:
        result["source_audit"] = audit_source(pyo3_root, runtime_paths)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extension",
        action="append",
        type=Path,
        default=[],
        help="finished native extension to audit; may be repeated",
    )
    parser.add_argument(
        "--pyo3-root",
        type=Path,
        default=None,
        help="optional PyO3 source root for supplementary diagnostics",
    )
    parser.add_argument(
        "--pypy-root",
        type=Path,
        default=Path(os.environ.get("PYPY_HOME", "")),
    )
    parser.add_argument(
        "--source-only",
        action="store_true",
        help="emit only supplementary PyO3 source diagnostics",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args()
    result = audit(
        extensions=[] if args.source_only else args.extension,
        pypy_root=args.pypy_root,
        pyo3_root=args.pyo3_root,
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"PyPy runtime libraries: {len(result['runtime_paths'])}")
        for extension in result["extensions"]:
            print(f"ABI audit: {extension['path']}")
            print(f"  required Python symbols: {extension['required_python_symbol_count']}")
            missing = extension["missing_python_symbols"]
            print(f"  missing from PyPy runtime: {len(missing)}")
            if missing:
                print("  missing:", ", ".join(missing))
            if extension["nm_errors"]:
                print("  nm diagnostic:", "; ".join(extension["nm_errors"]))
        source = result.get("source_audit")
        if source is not None:
            print(
                "Supplementary PyO3 source symbols missing from runtime:",
                len(source["missing_symbols"]),
            )
        print("Binary ABI audit:", "pass" if result["binary_abi_ok"] else "fail")
    return 0 if args.source_only or result["binary_abi_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
