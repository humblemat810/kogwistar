from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.ci, pytest.mark.core]

ROOT = Path(__file__).resolve().parents[2]
DSN_ENV = "KOGWISTAR_TEST_PG_DSN"


def test_rust_postgres_tests_are_explicitly_gated_and_ci_runs_them() -> None:
    rust_root = ROOT / "rust" / "crates"
    postgres_references = 0

    for path in rust_root.rglob("*.rs"):
        source = path.read_text(encoding="utf-8")
        declarations = list(
            re.finditer(
                r"(?m)^(?P<indent>[ \t]*)(?:async[ \t]+)?fn[ \t]+([A-Za-z_]\w*)[ \t]*\(",
                source,
            )
        )
        for occurrence in re.finditer(
            rf'(?:std::)?env::var\(\s*"{DSN_ENV}"\s*\)', source
        ):
            postgres_references += 1
            preceding = [item for item in declarations if item.start() < occurrence.start()]
            declaration = preceding[-1] if preceding else None
            if declaration is not None:
                # A Rust test may contain local helper methods (for example a
                # Drop implementation). Attribute their DSN use to the outer
                # test rather than requiring the local method to be ignored.
                nested_indent = len(declaration.group("indent"))
                declaration = next(
                    (
                        item
                        for item in reversed(preceding[:-1])
                        if len(item.group("indent")) < nested_indent
                    ),
                    declaration,
                )
            assert declaration is not None, f"{path}: DSN reference is outside a function"

            preceding_lines = source[: declaration.start()].splitlines()
            attributes: list[str] = []
            for line in reversed(preceding_lines):
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("#["):
                    attributes.append(stripped)
                    continue
                break

            attribute_block = "\n".join(attributes)
            assert "#[ignore" in attribute_block, (
                f"{path}:{source.count(chr(10), 0, declaration.start()) + 1}: "
                "PostgreSQL tests must be opt-in, not successful no-ops"
            )
            assert DSN_ENV in attribute_block, (
                f"{path}:{source.count(chr(10), 0, declaration.start()) + 1}: "
                "the ignore reason must explain the PostgreSQL DSN requirement"
            )
            missing_dsn_handling = source[occurrence.end() : occurrence.end() + 120]
            assert any(token in missing_dsn_handling for token in (".expect(", ".unwrap(")), (
                f"{path}:{source.count(chr(10), 0, occurrence.start()) + 1}: "
                "an explicitly selected PostgreSQL test must fail if its DSN is missing"
            )

    assert postgres_references > 0

    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    rust_job = workflow.split("\n  rust:\n", maxsplit=1)[1]
    assert "image: pgvector/pgvector:pg17" in rust_job
    assert f"{DSN_ENV}: postgres://" in rust_job
    assert "--include-ignored" in rust_job
