from __future__ import annotations

import pytest

pytestmark = pytest.mark.ci
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT / "docs"
TUTORIALS_DIR = DOCS_DIR / "tutorials"
SECTIONS_DIR = ROOT / "scripts" / "tutorial_sections"

LEARNING_PATH = DOCS_DIR / "LEARNING_PATH.md"
README = TUTORIALS_DIR / "README.md"

NUMBERED_DOCS = [
    TUTORIALS_DIR / "01_hello_graph_engine.md",
    TUTORIALS_DIR / "02_core_data_model.md",
    TUTORIALS_DIR / "03_build_a_small_knowledge_graph.md",
    TUTORIALS_DIR / "04_conversation_graph_basics.md",
    TUTORIALS_DIR / "05_context_snapshot_and_replay.md",
    TUTORIALS_DIR / "06_first_workflow.md",
    TUTORIALS_DIR / "07_branch_join_workflows.md",
    TUTORIALS_DIR / "08_storage_backends_and_parity.md",
    TUTORIALS_DIR / "09_indexing_pipeline.md",
    TUTORIALS_DIR / "10_event_log_replay_and_cdc.md",
    TUTORIALS_DIR / "11_build_a_mini_graphrag_app.md",
    TUTORIALS_DIR / "12_designer_api_integration.md",
    TUTORIALS_DIR / "13_how_to_test_this_repo.md",
    TUTORIALS_DIR / "14_architecture_deep_dive.md",
    TUTORIALS_DIR / "15_historical_search_tombstone_redirect.md",
    TUTORIALS_DIR / "16_leakage_prevention_with_model_slicing.md",
    TUTORIALS_DIR / "17_custom_llm_provider.md",
    TUTORIALS_DIR / "18_nested_workflow_invocation.md",
    TUTORIALS_DIR / "19_build_artifact_governance_workflow.md",
]

LEGACY_LEVEL_DOCS = [
    TUTORIALS_DIR / "level-0-simple-rag.md",
    TUTORIALS_DIR / "level-1-retrieval-orchestration.md",
    TUTORIALS_DIR / "level-2-provenance-pinning.md",
    TUTORIALS_DIR / "level-3-event-loop-control.md",
    TUTORIALS_DIR / "runtime-level-0-basics.md",
    TUTORIALS_DIR / "runtime-level-1-resolvers.md",
    TUTORIALS_DIR / "runtime-level-2-pause-resume.md",
    TUTORIALS_DIR / "runtime-level-3-observability-interop.md",
    TUTORIALS_DIR / "runtime-level-4-sandboxed-ops.md",
]

COMPANION_FILES = [
    SECTIONS_DIR / "01_hello_graph_engine.py",
    SECTIONS_DIR / "03_build_a_small_knowledge_graph.py",
    SECTIONS_DIR / "04_conversation_graph_basics.py",
    SECTIONS_DIR / "05_context_snapshot_and_replay.py",
    SECTIONS_DIR / "06_first_workflow.py",
    SECTIONS_DIR / "07_branch_join_workflows.py",
    SECTIONS_DIR / "10_event_log_replay_and_cdc.py",
    SECTIONS_DIR / "11_build_a_mini_graphrag_app.py",
    SECTIONS_DIR / "15_historical_search_tombstone_redirect.py",
    SECTIONS_DIR / "16_leakage_prevention_with_model_slicing.py",
    SECTIONS_DIR / "17_custom_llm_provider.py",
    SECTIONS_DIR / "18_nested_workflow_invocation.py",
    SECTIONS_DIR / "19_build_artifact_governance_workflow.py",
]

REQUIRED_TEMPLATE_HEADINGS = [
    "## What You Will Build",
    "## Why This Matters",
    "## Run or Inspect",
    "## Inspect The Result",
    "## Invariant Demonstrated",
    "## Next Tutorial",
]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_relative_links(path: Path) -> list[Path]:
    content = _read(path)
    out: list[Path] = []
    for raw in re.findall(r"\[[^\]]+\]\(([^)]+)\)", content):
        target = raw.strip()
        if not target or target.startswith("#"):
            continue
        if "://" in target or target.startswith("mailto:"):
            continue
        if target.startswith("/"):
            continue
        rel = target.split("#", 1)[0]
        out.append((path.parent / rel).resolve())
    return out


def test_learning_path_and_index_exist():
    assert LEARNING_PATH.exists()
    assert README.exists()


def test_numbered_tutorial_doc_set_exists():
    for path in NUMBERED_DOCS:
        assert path.exists(), f"Missing numbered tutorial doc: {path}"


def test_legacy_level_doc_set_exists():
    for path in LEGACY_LEVEL_DOCS:
        assert path.exists(), f"Missing legacy tutorial doc: {path}"


def test_learning_path_has_audience_bands_and_order():
    content = _read(LEARNING_PATH)
    assert "Beginner / evaluator" in content
    assert "Builder / integrator" in content
    assert "Advanced / contributor" in content
    assert (
        content.index("01 Hello Graph Engine")
        < content.index("07 Branch Join Workflows")
        < content.index("15 Historical Search With Tombstone and Redirect")
    )


def test_numbered_docs_follow_required_template():
    for path in NUMBERED_DOCS:
        content = _read(path)
        for heading in REQUIRED_TEMPLATE_HEADINGS:
            assert heading in content, f"Missing heading {heading!r} in {path}"


def test_legacy_level_docs_are_upgraded_and_still_runnable():
    for path in LEGACY_LEVEL_DOCS:
        content = _read(path)
        for heading in REQUIRED_TEMPLATE_HEADINGS:
            assert heading in content, f"Missing heading {heading!r} in {path}"
        assert "## Quick Run" in content, str(path) + " is missing '## Quick Run' "
        assert "## Checkpoint" in content, str(path) + " is missing '## Checkpoint' "
        assert "## Troubleshooting" in content, str(path) + " is missing '## Troubleshooting' "
        assert "```powershell" in content or "```bash" in content, str(path) + " is missing '## runnable scripts' "


def test_tutorial_index_mentions_learning_path_companions_and_pattern_matrix():
    content = _read(README)
    assert "Learning Path Docs" in content
    assert "VS Code Companion Files" in content
    assert "Pattern Matrix" in content
    assert "different depth/pathway" in content
    assert "Level 0" in content
    assert "Level 3" in content
    assert "Level 4" in content


def test_companion_files_exist_and_have_vscode_cells():
    helper = SECTIONS_DIR / "_helpers.py"
    assert helper.exists(), "Missing helper module for companion tutorial sections."
    for path in COMPANION_FILES:
        content = _read(path)
        assert path.exists(), f"Missing companion file: {path}"
        assert "# %% [markdown]" in content
        assert "# %%" in content


def test_relative_links_in_learning_docs_resolve():
    docs_to_check = [LEARNING_PATH, README, *NUMBERED_DOCS, *LEGACY_LEVEL_DOCS]
    missing: list[str] = []
    for path in docs_to_check:
        for target in _extract_relative_links(path):
            if not target.exists():
                missing.append(f"{path}: {target}")
    assert not missing, "Broken relative tutorial links:\n" + "\n".join(missing)


def test_tutorial_commands_reference_existing_scripts():
    texts = [
        _read(README),
        *(_read(path) for path in LEGACY_LEVEL_DOCS),
        *(_read(path) for path in NUMBERED_DOCS),
    ]
    combined = "\n".join(texts)
    scripts = sorted(
        set(re.findall(r"python\s+(scripts/[A-Za-z0-9_./-]+\.py)", combined))
    )
    assert scripts, "No script commands found in tutorial docs."
    for rel in scripts:
        script_path = ROOT / rel
        assert script_path.exists(), f"Script referenced in docs is missing: {rel}"


def test_historical_tutorial_cross_links_present():
    mini_app = _read(TUTORIALS_DIR / "11_build_a_mini_graphrag_app.md")
    rag_level2 = _read(TUTORIALS_DIR / "level-2-provenance-pinning.md")
    runtime_overview = _read(TUTORIALS_DIR / "runtime-ladder-overview.md")
    assert "./15_historical_search_tombstone_redirect.md" in mini_app
    assert "./15_historical_search_tombstone_redirect.md" in rag_level2
    assert "./runtime-level-4-sandboxed-ops.md" in runtime_overview
