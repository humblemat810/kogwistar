"""Load the installed Kogwistar extension without executing package __init__."""

from __future__ import annotations

import importlib.util
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
import sys
import sysconfig
import types


def _extension_candidates() -> list[Path]:
    roots = {
        Path(sysconfig.get_path(name))
        for name in ("purelib", "platlib")
        if sysconfig.get_path(name)
    }
    candidates: list[Path] = []
    for root in roots:
        package_root = root / "kogwistar"
        for suffix in EXTENSION_SUFFIXES:
            candidates.extend(package_root.glob(f"_rust*{suffix}"))
    return sorted(set(candidates))


def load_extension(path: Path) -> types.ModuleType:
    package = types.ModuleType("kogwistar")
    package.__path__ = [str(path.parent)]
    package.__package__ = "kogwistar"
    sys.modules["kogwistar"] = package
    spec = importlib.util.spec_from_file_location("kogwistar._rust", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot create extension spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    candidates = _extension_candidates()
    if not candidates:
        raise SystemExit("installed kogwistar._rust extension was not found")
    module = load_extension(candidates[0])
    print("implementation:", sys.implementation)
    print("executable:", sys.executable)
    print("native extension:", module.__file__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
