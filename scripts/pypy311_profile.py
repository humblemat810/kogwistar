"""Validate and report the hosted PyPy 3.11 Python-authority profile."""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--import-package",
        action="store_true",
        help="Import kogwistar after validating the interpreter.",
    )
    args = parser.parse_args()

    if sys.implementation.name != "pypy":
        raise SystemExit(f"expected PyPy, got {sys.implementation.name}")
    if sys.version_info[:2] != (3, 11):
        raise SystemExit(f"expected Python 3.11, got {sys.version_info[:2]}")

    print("implementation:", sys.implementation)
    print("version:", sys.version)
    print("executable:", sys.executable)
    if args.import_package:
        import kogwistar

        print("kogwistar:", kogwistar.__file__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
