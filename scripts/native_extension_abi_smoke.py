"""Verify that Python loads one expected ADR-015 native extension ABI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import kogwistar._rust as native  # noqa: E402


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-extension", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _args()
    loaded = Path(native.__file__).resolve()
    if args.expected_extension is not None:
        expected = args.expected_extension.resolve()
        if loaded != expected:
            raise SystemExit(
                f"native extension mismatch: expected {expected}, loaded {loaded}"
            )
    result = json.loads(
        native.store_sqlite_json(
            json.dumps(
                {
                    "path": ":memory:",
                    "transaction_id": None,
                    "operation": {"kind": "open_init"},
                },
                separators=(",", ":"),
            )
        )
    )
    if result != {"initialized": True}:
        raise SystemExit(f"native SQLite ABI smoke returned {result!r}")
    print(
        json.dumps(
            {
                "extension": str(loaded),
                "extension_version": getattr(native, "__version__", None),
                "contract_version": getattr(native, "CONTRACT_VERSION", None),
                "sqlite_transaction_id_abi": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
