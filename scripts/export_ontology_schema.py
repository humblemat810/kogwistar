"""Export the versioned ontology package JSON Schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kogwistar.ontology import ontology_package_json_schema


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("contracts/ontology/ontology_package.v1.schema.json"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            ontology_package_json_schema(),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
