"""Inspect the PyPy-only pydantic-core compatibility probe."""

from __future__ import annotations

import argparse
import importlib.metadata
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-version", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--expected-version")
    args = parser.parse_args()

    version = importlib.metadata.version("pydantic-core")
    if args.metadata_version:
        print(version)
        return 0

    if not args.verify:
        parser.error("choose --metadata-version or --verify")

    import pydantic_core

    assert pydantic_core.__version__ == version, (pydantic_core.__version__, version)
    if args.expected_version is not None:
        assert version == args.expected_version, (version, args.expected_version)
    print("pydantic_core:", pydantic_core.__version__)
    print("implementation:", sys.implementation.name)
    print("executable:", sys.executable)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
