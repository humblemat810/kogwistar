"""Verify a pure-Python Kogwistar wheel install on PyPy 3.11."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import platform
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-version", required=True)
    args = parser.parse_args()

    if platform.python_implementation() != "PyPy" or sys.version_info[:2] != (3, 11):
        raise SystemExit("expected PyPy 3.11")

    import kogwistar
    from kogwistar.engine_core.models import Node

    module_path = Path(kogwistar.__file__).resolve()
    if "site-packages" not in module_path.parts:
        raise SystemExit(f"Kogwistar did not import from installed wheel: {module_path}")

    installed_version = importlib.metadata.version("kogwistar")
    if installed_version != args.expected_version or kogwistar.__version__ != installed_version:
        raise SystemExit(
            f"version mismatch: metadata={installed_version}, "
            f"package={kogwistar.__version__}, expected={args.expected_version}"
        )
    if importlib.util.find_spec("kogwistar._rust") is not None:
        raise SystemExit("pure-Python PyPy wheel unexpectedly exposes native _rust")

    print(
        f"implementation=PyPy {platform.python_version()} "
        f"version={installed_version} module={module_path} model={Node.__name__}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
