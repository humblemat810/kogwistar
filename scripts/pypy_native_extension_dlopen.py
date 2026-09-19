"""Preflight-load PyPy native extensions with the dynamic linker."""

from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path


def preflight(path: Path) -> str | None:
    """Resolve an extension with RTLD_NOW without executing its Python init."""

    try:
        ctypes.CDLL(str(path), mode=getattr(os, "RTLD_NOW", 2))
    except OSError as exc:
        return str(exc)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", action="append", type=Path, required=True)
    args = parser.parse_args()
    failed = False
    for path in args.extension:
        error = preflight(path)
        if error:
            failed = True
            print(f"FAIL {path}: {error}")
        else:
            print(f"PASS {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
