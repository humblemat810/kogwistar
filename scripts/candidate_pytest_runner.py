"""Verify candidate import provenance, then run pytest with forwarded arguments."""

from __future__ import annotations

from pathlib import Path
import sys

import kogwistar
import pytest


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        raise SystemExit("expected candidate package path argument")
    expected = Path(args.pop(0)).resolve()
    if args[:1] == ["--"]:
        args.pop(0)
    resolved = Path(kogwistar.__file__).resolve()
    if resolved != expected:
        raise SystemExit(
            f"candidate import mismatch before pytest: expected {expected}, "
            f"resolved {resolved}"
        )
    return pytest.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
