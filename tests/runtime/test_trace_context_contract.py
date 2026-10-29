import json
from pathlib import Path

from kogwistar.runtime.telemetry import TraceContext


def test_trace_context_defaults_match_shared_rust_fixture() -> None:
    root = Path(__file__).resolve().parents[2]
    fixture = json.loads(
        (root / "contracts" / "golden" / "runtime-trace-context-v1.json").read_text(
            encoding="utf-8"
        )
    )
    fields = TraceContext(**fixture["context"]).as_fields()

    for field, expected in fixture["expected"].items():
        assert fields[field] == expected
