from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
DATA_ROOT = ROOT / ".gke-data" / "tutorial-sections"

for candidate in (ROOT, SCRIPTS_DIR):
    text = str(candidate)
    if text not in sys.path:
        sys.path.insert(0, text)

from graph_knowledge_engine.engine_core.models import (
    Grounding,
    MentionVerification,
    Span,
)


def reset_data_dir(name: str) -> Path:
    path = DATA_ROOT / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def show(title: str, payload) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def tutorial_grounding(
    doc_id: str, excerpt: str, *, insertion_method: str = "tutorial_sections"
) -> Grounding:
    span = Span(
        collection_page_url=f"tutorial/{doc_id}",
        document_page_url=f"tutorial/{doc_id}",
        doc_id=doc_id,
        insertion_method=insertion_method,
        page_number=1,
        start_char=0,
        end_char=max(1, len(excerpt)),
        excerpt=excerpt[:256],
        context_before="",
        context_after="",
        chunk_id=None,
        source_cluster_id=None,
        verification=MentionVerification(
            method="system", is_verified=True, score=1.0, notes=insertion_method
        ),
    )
    return Grounding(spans=[span])


def banner(text: str) -> None:
    print(f"\n[text] {text}")
