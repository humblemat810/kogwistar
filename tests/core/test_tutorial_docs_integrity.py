from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT / "docs" / "tutorials"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def test_tutorial_doc_set_exists():
    expected = [
        DOCS_DIR / "README.md",
        DOCS_DIR / "level-0-simple-rag.md",
        DOCS_DIR / "level-1-retrieval-orchestration.md",
        DOCS_DIR / "level-2-provenance-pinning.md",
        DOCS_DIR / "level-3-event-loop-control.md",
    ]
    for path in expected:
        assert path.exists(), f"Missing tutorial doc: {path}"


def test_tutorial_readme_has_ordered_levels_and_matrix():
    content = _read(DOCS_DIR / "README.md")
    assert "Level 0" in content
    assert "Level 1" in content
    assert "Level 2" in content
    assert "Level 3" in content
    assert content.index("Level 0") < content.index("Level 1") < content.index("Level 2") < content.index("Level 3")
    assert "Pattern Matrix" in content
    assert "Problem | Pattern | Where It Is Implemented" in content


def test_each_level_doc_has_required_sections_and_powershell_commands():
    level_docs = [
        DOCS_DIR / "level-0-simple-rag.md",
        DOCS_DIR / "level-1-retrieval-orchestration.md",
        DOCS_DIR / "level-2-provenance-pinning.md",
        DOCS_DIR / "level-3-event-loop-control.md",
    ]
    for path in level_docs:
        content = _read(path)
        assert "## Quick Run" in content
        assert "## Inside The Engine" in content
        assert "## Checkpoint" in content
        assert "## Troubleshooting" in content
        assert "```powershell" in content


def test_tutorial_commands_reference_existing_scripts():
    texts = [
        _read(DOCS_DIR / "README.md"),
        _read(DOCS_DIR / "level-0-simple-rag.md"),
        _read(DOCS_DIR / "level-1-retrieval-orchestration.md"),
        _read(DOCS_DIR / "level-2-provenance-pinning.md"),
        _read(DOCS_DIR / "level-3-event-loop-control.md"),
    ]
    combined = "\n".join(texts)
    scripts = sorted(set(re.findall(r"python\s+(scripts/[A-Za-z0-9_.-]+\.py)", combined)))
    assert scripts, "No script commands found in tutorial docs."
    for rel in scripts:
        script_path = ROOT / rel
        assert script_path.exists(), f"Script referenced in docs is missing: {rel}"
