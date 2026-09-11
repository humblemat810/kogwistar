"""Print installed native-extension path for ADR-015 container staging."""

from __future__ import annotations

from pathlib import Path

import kogwistar._rust as native


def main() -> int:
    print(Path(native.__file__).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
