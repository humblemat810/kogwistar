from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "contracts" / "rust-port-v1.json"
IGNORED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
}


def _git_commit(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or None


def _application_root(raw: str | None, manifest: dict) -> Path:
    configured = raw or os.getenv(
        manifest["reference_application"]["root_environment_variable"]
    )
    root = (
        Path(configured).expanduser()
        if configured
        else ROOT / manifest["reference_application"]["default_relative_root"]
    ).resolve()
    if not root.is_dir():
        raise SystemExit(f"reference application root does not exist: {root}")
    return root


def _python_files(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            relative_parts = path.relative_to(root).parts
            if not any(
                part in IGNORED_PARTS or part.startswith(".")
                for part in relative_parts
            ):
                yield path


def _scan_file(path: Path) -> list[tuple[str, str, int]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"cannot inventory Python imports in {path}: {exc}") from exc

    imports: list[tuple[str, str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "kogwistar" or alias.name.startswith("kogwistar."):
                    imports.append((alias.name, "<module>", node.lineno))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "kogwistar" or module.startswith("kogwistar."):
                imports.extend((module, alias.name, node.lineno) for alias in node.names)
    return imports


def _classification_index(manifest: dict) -> dict[tuple[str, str], str]:
    index: dict[tuple[str, str], str] = {}
    for item in manifest["python_facade"]["modules"]:
        module = item["module"]
        status = item["status"]
        index[(module, "<module>")] = status
        if not item["symbols"]:
            index[(module, "*")] = status
        for symbol in item["symbols"]:
            index[(module, symbol)] = status
    for item in manifest.get("consumer_import_classifications", []):
        index[(item["module"], item["symbol"])] = item["status"]
    return index


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def build_inventory(application_root: Path, manifest: dict) -> dict[str, object]:
    repositories = {
        "application": {
            "root": application_root,
            "scan_roots": [application_root / "src", application_root / "tests"],
        },
        "parser": {
            "root": application_root / "kg-doc-parser",
            "scan_roots": [
                application_root / "kg-doc-parser" / "kg_doc_parser",
                application_root / "kg-doc-parser" / "tests",
            ],
        },
        "sink": {
            "root": application_root / "kogwistar-obsidian-sink",
            "scan_roots": [
                application_root
                / "kogwistar-obsidian-sink"
                / "kogwistar_obsidian_sink",
                application_root / "kogwistar-obsidian-sink" / "tests",
            ],
        },
    }
    classifications = _classification_index(manifest)
    found: dict[tuple[str, str, str, str], list[int]] = defaultdict(list)

    for repository, config in repositories.items():
        repo_root = config["root"]
        assert isinstance(repo_root, Path)
        scan_roots = config["scan_roots"]
        assert isinstance(scan_roots, list)
        for path in _python_files(scan_roots):
            scope = "test-only" if "tests" in path.parts else "runtime"
            relative = _relative(path, repo_root)
            for module, symbol, line in _scan_file(path):
                found[(repository, module, symbol, f"{scope}:{relative}")].append(line)

    records: list[dict[str, object]] = []
    for (repository, module, symbol, source), lines in sorted(found.items()):
        status = classifications.get(
            (module, symbol), classifications.get((module, "*"), "unclassified")
        )
        records.append(
            {
                "repository": repository,
                "module": module,
                "symbol": symbol,
                "scope": source.split(":", 1)[0],
                "source": source.split(":", 1)[1],
                "lines": sorted(set(lines)),
                "status": status,
            }
        )

    unclassified = [record for record in records if record["status"] == "unclassified"]
    return {
        "inventory_version": 1,
        "contract_version": manifest["contract_version"],
        "repositories": {
            name: {
                "root": (
                    "."
                    if name == "application"
                    else manifest["reference_application"]["nested_repositories"][name]
                ),
                "commit": _git_commit(config["root"]),
            }
            for name, config in repositories.items()
        },
        "records": records,
        "summary": {
            "records": len(records),
            "classified": len(records) - len(unclassified),
            "unclassified": len(unclassified),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventory consumer imports governed by ADR-015."
    )
    parser.add_argument("--application-root")
    parser.add_argument("--output")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero when any consumer import remains unclassified.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    application_root = _application_root(args.application_root, manifest)
    inventory = build_inventory(application_root, manifest)
    payload = json.dumps(inventory, indent=2, sort_keys=True) + "\n"
    print(payload, end="")
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    unclassified = inventory["summary"]["unclassified"]
    assert isinstance(unclassified, int)
    return 2 if args.check and unclassified else 0


if __name__ == "__main__":
    raise SystemExit(main())
