# graph_knowledge_engine/utils/kge_debug_dump.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Literal

from jinja2 import Environment
from markupsafe import Markup

from graph_knowledge_engine.engine import GraphKnowledgeEngine
from graph_knowledge_engine.visualization.graph_viz import to_d3_force

_GRAPH_TYPE = Literal["knowledge", "conversation"]


def build_engine(*, persist_dir: Path, graph_type: _GRAPH_TYPE) -> GraphKnowledgeEngine:
    return GraphKnowledgeEngine(persist_directory=str(persist_dir), kg_graph_type=graph_type)


def _tojson(value) -> Markup:
    # Deterministic JSON for embedding into <script>.
    return Markup(json.dumps(value, ensure_ascii=False))


def _render_template_html(template_html: str, *, context: dict) -> str:
    # Offline rendering: no server required.
    env = Environment(autoescape=False)
    env.filters["tojson"] = _tojson
    tmpl = env.from_string(template_html)
    html = tmpl.render(**context)

    # Guard against the exact regression: leaking Jinja into bundle artifacts.
    if "{{" in html or "{%" in html:
        raise RuntimeError("Rendered bundle still contains Jinja tokens ('{{' or '{%').")
    return html


def dump_d3_bundle(
    *,
    engine: GraphKnowledgeEngine,
    template_html: str,
    out_html: Path,
    doc_id: Optional[str] = None,
    mode: str = "reify",
    insertion_method: Optional[str] = None,
    bundle_meta: Optional[dict] = None,
) -> Path:
    payload = to_d3_force(engine, doc_id=doc_id, insertion_method=insertion_method, mode=mode)

    rendered = _render_template_html(
        template_html,
        context={
            "doc_id": json.dumps(doc_id),
            "mode": mode,
            "insertion_method": json.dumps(insertion_method),
            "is_bundle": json.dumps(bundle_meta is not None),
            "embedded_data": json.dumps(payload),
            "bundle_meta": json.dumps(bundle_meta) if bundle_meta is not None else None,
        },
    )
    
    out_html.write_text(rendered, encoding="utf-8")
    # if " None" in rendered or "\nNone" in rendered or "None" in rendered:
    #     raise RuntimeError("Invalid JS literal: Python None leaked into bundle")
    return out_html


def dump_paired_bundles(
    *,
    kg_engine: GraphKnowledgeEngine,
    conversation_engine: GraphKnowledgeEngine,
    workflow_engine: GraphKnowledgeEngine | None = None,
    template_html: str,
    out_dir: Path,
    kg_out: str = "kg.bundle.html",
    conversation_out: str = "conversation.bundle.html",
    work_flow_out: str = "workflow.bundle.html",
    kg_doc_id: Optional[str] = None,
    conversation_doc_id: Optional[str] = None,
    mode: str = "reify",
    insertion_method: Optional[str] = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle_meta = {
        "kg_bundle_href": f"./{kg_out}",
        "conversation_bundle_href": f"./{conversation_out}",
        "mode": mode,
        "insertion_method": insertion_method,
        "kg_doc_id": kg_doc_id,
        "conversation_doc_id": conversation_doc_id,
    }

    dump_d3_bundle(
        engine=kg_engine,
        template_html=template_html,
        out_html=out_dir / kg_out,
        doc_id=kg_doc_id,
        mode=mode,
        insertion_method=insertion_method,
        bundle_meta=bundle_meta,
    )

    dump_d3_bundle(
        engine=conversation_engine,
        template_html=template_html,
        out_html=out_dir / conversation_out,
        doc_id=conversation_doc_id,
        mode=mode,
        insertion_method=insertion_method,
        bundle_meta=bundle_meta,
    )
    if workflow_engine:
        wf_nodes = workflow_engine.get_nodes(where={"entity_type": "workflow_node"}, limit=20000)
        wf_edges = workflow_engine.get_edges(where={"entity_type": "workflow_edge"}, limit=50000)
        dump_d3_bundle(
            engine=workflow_engine,
            template_html=template_html,
            out_html=out_dir / work_flow_out,
            doc_id=conversation_doc_id,
            mode=mode,
            insertion_method=insertion_method,
            bundle_meta=bundle_meta,
        )
    (out_dir / "bundle.meta.json").write_text(json.dumps(bundle_meta, indent=2), encoding="utf-8")
    # os.startfile(str(out_dir))
    return bundle_meta


def _cmd_one(args: argparse.Namespace) -> None:
    engine = build_engine(persist_dir=Path(args.persist_dir), graph_type=args.graph_type)
    template_html = Path(args.template).read_text(encoding="utf-8")

    dump_d3_bundle(
        engine=engine,
        template_html=template_html,
        out_html=Path(args.out),
        doc_id=args.doc_id,
        mode=args.mode,
        insertion_method=args.insertion_method,
        bundle_meta=None,
    )
    print(f"[OK] D3 bundle written to {Path(args.out).absolute()}")


def _cmd_pair(args: argparse.Namespace) -> None:
    kg_engine = build_engine(persist_dir=Path(args.kg_persist_dir), graph_type="knowledge")
    conv_engine = build_engine(persist_dir=Path(args.conversation_persist_dir), graph_type="conversation")
    template_html = Path(args.template).read_text(encoding="utf-8")

    meta = dump_paired_bundles(
        kg_engine=kg_engine,
        conversation_engine=conv_engine,
        template_html=template_html,
        out_dir=Path(args.out_dir),
        kg_out=args.kg_out,
        conversation_out=args.conversation_out,
        kg_doc_id=args.kg_doc_id,
        conversation_doc_id=args.conversation_doc_id,
        mode=args.mode,
        insertion_method=args.insertion_method,
    )
    print(f"[OK] Paired bundle written to {Path(args.out_dir).absolute()}")
    print("[OK] bundle.meta.json:", json.dumps(meta, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump self-contained D3 debug bundles for GraphKnowledgeEngine")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("one", help="Dump one graph (knowledge or conversation)")
    p1.add_argument("--persist-dir", required=True, help="Chroma persist directory")
    p1.add_argument("--graph-type", choices=["knowledge", "conversation"], default="knowledge", help="Which graph")
    p1.add_argument("--doc-id", help="Optional doc_id scope")
    p1.add_argument("--mode", default="reify", help="Visualization mode (default: reify)")
    p1.add_argument("--insertion-method", help="Optional insertion_method (e.g. document_ingestion)")
    p1.add_argument("--template", required=True, help="Path to templates/d3.html")
    p1.add_argument("--out", default="d3.bundle.html", help="Output HTML file")
    p1.set_defaults(func=_cmd_one)

    p2 = sub.add_parser("pair", help="Dump paired bundles (KG + Conversation) into one folder")
    p2.add_argument("--kg-persist-dir", required=True, help="Chroma persist dir for knowledge graph")
    p2.add_argument("--conversation-persist-dir", required=True, help="Chroma persist dir for conversation graph")
    p2.add_argument("--kg-doc-id", help="Optional KG doc_id scope")
    p2.add_argument("--conversation-doc-id", help="Optional conversation doc_id scope")
    p2.add_argument("--mode", default="reify", help="Visualization mode (default: reify)")
    p2.add_argument("--insertion-method", help="Optional insertion_method (e.g. document_ingestion)")
    p2.add_argument("--template", required=True, help="Path to templates/d3.html")
    p2.add_argument("--out-dir", default="d3_bundle", help="Output folder")
    p2.add_argument("--kg-out", default="kg.bundle.html", help="KG bundle filename within out-dir")
    p2.add_argument("--conversation-out", default="conversation.bundle.html", help="Conversation bundle filename within out-dir")
    p2.set_defaults(func=_cmd_pair)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
