"""Install one built wheel, then verify its native contract in a clean process."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheelhouse", type=Path, default=Path("wheelhouse"))
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def _verify() -> int:
    import kogwistar
    from kogwistar import _rust

    assert _rust.CONTRACT_VERSION == "1.0.0"
    assert _rust.stable_id_json('["node","golden"]')
    print(kogwistar.__file__, _rust.__file__)
    return 0


def main() -> int:
    args = _args()
    if args.verify:
        return _verify()
    wheels = sorted(args.wheelhouse.glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected one wheel in {args.wheelhouse}, found {len(wheels)}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--force-reinstall", str(wheels[0])],
        check=True,
    )
    return subprocess.run(
        [sys.executable, "-P", str(Path(__file__).resolve()), "--verify"],
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
