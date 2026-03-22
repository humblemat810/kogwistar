from __future__ import annotations

import argparse
import json
from pathlib import Path

from kogwistar.demo import run_provenance_quickstart
from kogwistar.server_mcp_with_admin import main as serve_main


def _print_quickstart_summary(summary: dict[str, object]) -> None:
    artifacts = dict(summary.get("artifacts") or {})
    print(f"Answer: {summary.get('answer_text', '')}")
    print(f"Replay: {'pass' if summary.get('replay_pass') else 'fail'}")
    print(f"Provenance Artifact: {artifacts.get('provenance_html', '')}")
    print(f"Graph Artifact: {artifacts.get('graph_html', '')}")
    print(f"Replay Report: {artifacts.get('replay_json', '')}")
    print(f"Next: {summary.get('next_command', '')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kogwistar",
        description="Provenance-first, replayable AI workflow tooling.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def _add_demo_args(target: argparse.ArgumentParser) -> None:
        target.add_argument("--data-dir", default=".gke-data/quickstart")
        target.add_argument(
            "--question",
            default="How does Kogwistar make AI workflows replayable and auditable?",
        )
        target.add_argument(
            "--open-browser",
            action=argparse.BooleanOptionalAction,
            default=False,
        )
        target.add_argument("--json", action="store_true")

    quickstart = sub.add_parser(
        "quickstart",
        help="Run the deterministic provenance-first demo.",
    )
    _add_demo_args(quickstart)

    serve = sub.add_parser(
        "serve", help="Start the existing MCP/server surface."
    )
    serve.add_argument(
        "--data-dir",
        default=None,
        help="Reserved for future use; server storage remains env-driven.",
    )

    demo = sub.add_parser("demo", help="Run named demos.")
    demo_sub = demo.add_subparsers(dest="demo_command", required=True)
    provenance = demo_sub.add_parser(
        "provenance", help="Run the provenance-first signature demo."
    )
    _add_demo_args(provenance)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "serve":
        serve_main()
        return 0

    if args.command == "quickstart" or (
        args.command == "demo" and args.demo_command == "provenance"
    ):
        summary = run_provenance_quickstart(
            data_dir=Path(args.data_dir),
            question=str(args.question),
            open_browser=bool(args.open_browser),
        )
        if args.json:
            print(json.dumps(summary, indent=2, ensure_ascii=False))
        else:
            _print_quickstart_summary(summary)
        return 0

    parser.error("Unknown command")
    return 2
