# kge_debug_dump.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Literal

from graph_knowledge_engine.engine import GraphKnowledgeEngine
from graph_knowledge_engine.viz import to_d3_force  # adjust import if needed


def dump_d3_bundle(
    *,
    engine: GraphKnowledgeEngine,
    template_html: str,
    out_html: Path,
    doc_id: Optional[str] = None,
    mode: str = "reify",
    insertion_method: Optional[str] = None,
) -> Path:
    """
    Dump a self-contained D3 HTML bundle with embedded graph data.

    Can be called:
    - from CLI
    - from debugger (debugpy pause)
    """

    payload = to_d3_force(
        engine,
        doc_id=doc_id,
        mode=mode,
        insertion_method=insertion_method,
    )

    injected = template_html.replace(
        "/*__INJECT_DATA__*/",
        "window.__EMBEDDED_DATA__ = " + json.dumps(payload) + ";",
    )

    out_html.write_text(injected, encoding="utf-8")
    return out_html


def build_engine(
    *,
    persist_dir: Path,
    graph_type: Literal["knowledge", "conversation"],
) -> GraphKnowledgeEngine:
    return GraphKnowledgeEngine(
        persist_directory=str(persist_dir),
        kg_graph_type=graph_type,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Dump a self-contained D3 debug bundle for GraphKnowledgeEngine"
    )

    parser.add_argument(
        "--persist-dir",
        required=True,
        help="Chroma persist directory",
    )

    parser.add_argument(
        "--graph-type",
        choices=["knowledge", "conversation"],
        default="knowledge",
        help="Which graph to dump",
    )

    parser.add_argument(
        "--doc-id",
        help="Optional doc_id scope",
    )

    parser.add_argument(
        "--mode",
        default="reify",
        help="Visualization mode (default: reify)",
    )

    parser.add_argument(
        "--insertion-method",
        help="Optional insertion_method (e.g. document_ingestion)",
    )

    parser.add_argument(
        "--template",
        required=True,
        help="Path to d3 bundle template HTML",
    )

    parser.add_argument(
        "--out",
        default="d3.bundle.html",
        help="Output HTML file",
    )

    args = parser.parse_args()

    engine = build_engine(
        persist_dir=Path(args.persist_dir),
        graph_type=args.graph_type,
    )

    template_html = Path(args.template).read_text(encoding="utf-8")

    out = dump_d3_bundle(
        engine=engine,
        template_html=template_html,
        out_html=Path(args.out),
        doc_id=args.doc_id,
        mode=args.mode,
        insertion_method=args.insertion_method,
    )

    print(f"[OK] D3 bundle written to {out.absolute()}")


if __name__ == "__main__":
    main()
"""
python graph_knowledge_engine\utils\kge_debug_dump.py \
  --persist-dir ./chroma_kg \
  --graph-type knowledge \
  --doc-id DOC123 \
  --template ./templates/d3.bundle.html \
  --out ./debug_kg.html
  
  
python graph_knowledge_engine\utils\kge_debug_dump.py \
  --persist-dir ./chroma_conversation \
  --graph-type conversation \
  --template ./templates/d3.bundle.html \
  --out ./debug_conversation.html


# in debugger
from kge_debug_dump import dump_d3_bundle
from pathlib import Path

dump_d3_bundle(
    engine=conversation_engine,
    template_html=Path("templates/d3.bundle.html").read_text(),
    out_html=Path("/tmp/conv_snapshot.html"),
)

"""    