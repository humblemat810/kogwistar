"""Print core package/import provenance for CI smoke verification."""

from __future__ import annotations

import kogwistar
import kogwistar.utils
import kogwistar.utils.log


def main() -> int:
    print(kogwistar.__file__)
    print(kogwistar.utils.__file__)
    print(kogwistar.utils.log.__file__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
