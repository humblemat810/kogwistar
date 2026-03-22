from __future__ import annotations

import argparse

from . import list_submodules


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m kogwistar",
        description="List importable modules inside the kogwistar package.",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Show the full nested module tree instead of only immediate children.",
    )
    args = parser.parse_args()

    for module_name in list_submodules(recursive=args.recursive):
        print(module_name)


if __name__ == "__main__":
    main()
