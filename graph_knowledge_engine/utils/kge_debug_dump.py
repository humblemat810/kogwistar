# graph_knowledge_engine/utils/kge_debug_dump.py
from __future__ import annotations

import sys, pathlib, os
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
import argparse
import json
from pathlib import Path
from typing import Optional, Literal, TYPE_CHECKING

from jinja2 import Environment
from jwt.algorithms import NoneAlgorithm
from markupsafe import Markup
if TYPE_CHECKING:
    from graph_knowledge_engine.engine import GraphKnowledgeEngine, EngineType
from graph_knowledge_engine.visualization.graph_viz import to_d3_force

_GRAPH_TYPE = Literal["knowledge", "conversation", "workflow"]


def build_engine(*, persist_dir: Path, graph_type: _GRAPH_TYPE) -> GraphKnowledgeEngine:
    from graph_knowledge_engine.engine import GraphKnowledgeEngine
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
    engine: GraphKnowledgeEngine | None,
    engine_type: EngineType | None, 
    template_html: str,
    out_html: Path,
    doc_id: Optional[str] = None,
    mode: str = "reify",
    insertion_method: Optional[str] = None,
    bundle_meta: Optional[dict] = None,
    # Live CDC (optional): when enabled, the bundle will connect to an external
    # FastAPI change-bridge (NOT hosted by the engine/debugging process).
    cdc_enabled: bool = False,
    cdc_ws_url: Optional[str] = None,
    # If true, embed an empty graph (useful for "listen-only" live CDC pages).
    embed_empty: bool = False,
) -> Path:
    if engine and engine_type != engine.kg_graph_type:
        raise Exception(f"argument engine_type {engine_type} disagree with engine kg_graph_type = {engine.kg_graph_type}")
    if not engine_type and not engine:
        raise ValueError("either engine type or engine required")
    if not engine and not embed_empty:
        raise ValueError("engine can only be empty when embed empty is set true")
    payload = {"nodes": [], "links": []} if embed_empty else to_d3_force(engine, doc_id=doc_id, insertion_method=insertion_method, mode=mode)
    # for i, n in enumerate(payload['nodes']):
    #     n.pop('embedding')
    rendered = _render_template_html(
        template_html,
        context={
            "doc_id": json.dumps(doc_id),
            "mode": mode,
            "insertion_method": json.dumps(insertion_method),
            "is_bundle": json.dumps(bundle_meta is not None),
            "embedded_data": json.dumps(payload),
            "bundle_meta": json.dumps(bundle_meta) if bundle_meta is not None else None,
            # Live CDC config injected into the bundle.
            "bundle_graph_type": json.dumps(engine_type) or json.dumps(getattr(engine, "kg_graph_type", None)),
            "cdc_enabled": json.dumps(bool(cdc_enabled)),
            "cdc_ws_url": json.dumps(cdc_ws_url) if cdc_ws_url is not None else "null",
        },
    )
    
    out_html.write_text(rendered, encoding="utf-8")
    # if " None" in rendered or "\nNone" in rendered or "None" in rendered:
    #     raise RuntimeError("Invalid JS literal: Python None leaked into bundle")
    return out_html


def dump_paired_bundles(
    *,
    kg_engine: GraphKnowledgeEngine | None = None,
    conversation_engine: GraphKnowledgeEngine | None = None,
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
    # Live CDC (optional)
    cdc_ws_url: Optional[str] = None,
    embed_empty = False,
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
        engine_type = "knowledge",
        template_html=template_html,
        out_html=out_dir / kg_out,
        doc_id=kg_doc_id,
        mode=mode,
        insertion_method=insertion_method,
        bundle_meta=bundle_meta,
        cdc_enabled=bool(cdc_ws_url),
        cdc_ws_url=cdc_ws_url,
        embed_empty = embed_empty,
    )

    dump_d3_bundle(
        engine_type = "conversation",
        engine=conversation_engine,
        template_html=template_html,
        out_html=out_dir / conversation_out,
        doc_id=conversation_doc_id,
        mode=mode,
        insertion_method=insertion_method,
        bundle_meta=bundle_meta,
        cdc_enabled=bool(cdc_ws_url),
        cdc_ws_url=cdc_ws_url,
        embed_empty = embed_empty,
    )
    if workflow_engine or (embed_empty and cdc_ws_url):
        dump_d3_bundle(
            engine_type = "workflow",
            engine=workflow_engine,
            template_html=template_html,
            out_html=out_dir / work_flow_out,
            doc_id=conversation_doc_id,
            mode=mode,
            insertion_method=insertion_method,
            bundle_meta=bundle_meta,
            cdc_enabled=bool(cdc_ws_url),
            cdc_ws_url=cdc_ws_url,
            embed_empty = embed_empty,
        )
    (out_dir / "bundle.meta.json").write_text(json.dumps(bundle_meta, indent=2), encoding="utf-8")
    # os.startfile(str(out_dir))
    return bundle_meta


def _cmd_one(args: argparse.Namespace) -> None:
    engine = build_engine(persist_dir=Path(args.persist_dir), graph_type=args.graph_type)
    template_html = Path(args.template).read_text(encoding="utf-8")
    os.makedirs(str(Path(args.out).parent), exist_ok = True)
    dump_d3_bundle(
        engine_type = engine.kg_graph_type,
        engine=engine,
        template_html=template_html,
        out_html=Path(args.out),
        doc_id=args.doc_id,
        mode=args.mode,
        insertion_method=args.insertion_method,
        bundle_meta=None,
        cdc_enabled=bool(args.cdc_ws_url),
        cdc_ws_url=args.cdc_ws_url,
        embed_empty=bool(args.empty),
    )
    print(f"[OK] D3 bundle written to {Path(args.out).absolute()}")


def _cmd_pair(args: argparse.Namespace) -> None:
    "name is still pair but it is bundle"
    if not args.empty:
        kg_engine = build_engine(persist_dir=Path(args.kg_persist_dir), graph_type="knowledge")
        conv_engine = build_engine(persist_dir=Path(args.conversation_persist_dir), graph_type="conversation")
        workflow_engine =build_engine(persist_dir=Path(args.workflow_persist_dir), graph_type="workflow")
    else:
        kg_engine = None
        conv_engine=None
        workflow_engine= None
    template_html = Path(args.template).read_text(encoding="utf-8")

    meta = dump_paired_bundles(
        kg_engine=kg_engine,
        conversation_engine=conv_engine,
        workflow_engine=workflow_engine,
        template_html=template_html,
        out_dir=Path(args.out_dir),
        kg_out=args.kg_out,
        conversation_out=args.conversation_out,
        kg_doc_id=args.kg_doc_id,
        conversation_doc_id=args.conversation_doc_id,
        mode=args.mode,
        insertion_method=args.insertion_method,
        cdc_ws_url=args.cdc_ws_url,
        embed_empty=args.empty,
    )
    print(f"[OK] Paired bundle written to {Path(args.out_dir).absolute()}")
    print("[OK] bundle.meta.json:", json.dumps(meta, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump self-contained D3 debug bundles for GraphKnowledgeEngine")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("one", help="Dump one graph (knowledge or conversation)")
    p1.add_argument("--persist-dir", required=True, help="Chroma persist directory")
    p1.add_argument("--graph-type", choices=["knowledge", "conversation", "workflow"], default="knowledge", help="Which graph")
    p1.add_argument("--doc-id", help="Optional doc_id scope")
    p1.add_argument("--mode", default="reify", help="Visualization mode (default: reify)")
    p1.add_argument("--insertion-method", help="Optional insertion_method (e.g. document_ingestion)")
    p1.add_argument("--template", required=True, help="Path to templates/d3.html")
    p1.add_argument("--out", default="d3.bundle.html", help="Output HTML file")
    p1.add_argument("--empty", action="store_true", help="Embed an empty graph (listen-only) instead of dumping current DB")
    p1.add_argument("--cdc-ws-url", help="Enable live CDC and connect to this WebSocket URL (e.g. ws://127.0.0.1:8787/changes/ws)")
    p1.set_defaults(func=_cmd_one)

    p2 = sub.add_parser("bundle", help="Dump paired bundles (KG + Conversation) into one folder")
    p2.add_argument("--empty", action="store_true", help="Embed an empty graph (listen-only) instead of dumping current DB")
    p2.add_argument("--kg-persist-dir", required=False, help="Chroma persist dir for knowledge graph, not required when dump empty CDC receiver")
    p2.add_argument("--conversation-persist-dir", required=False, help="Chroma persist dir for conversation graph, not required when dump empty CDC receiver")
    p2.add_argument("--workflow-persist-dir", required=False, help="Chroma persist dir for conversation graph, not required when dump empty CDC receiver")
    p2.add_argument("--kg-doc-id", help="Optional KG doc_id scope")
    p2.add_argument("--conversation-doc-id", help="Optional conversation doc_id scope")
    p2.add_argument("--mode", default="reify", help="Visualization mode (default: reify)")
    p2.add_argument("--insertion-method", help="Optional insertion_method (e.g. document_ingestion)")
    p2.add_argument("--template", required=True, help="Path to templates/d3.html")
    p2.add_argument("--out-dir", default="d3_bundle", help="Output folder")
    p2.add_argument("--kg-out", default="kg.bundle.html", help="KG bundle filename within out-dir")
    p2.add_argument("--conversation-out", default="conversation.bundle.html", help="Conversation bundle filename within out-dir")
    p2.add_argument("--cdc-ws-url", help="Enable live CDC in all bundles and connect to this WebSocket URL")
    p2.set_defaults(func=_cmd_pair)

    args = parser.parse_args()
    if not args.empty:
        if args.cmd=="bundle":
            missing = []
            if args.kg_persist_dir is None:
                missing.append("--kg-persist-dir")
            if args.conversation_persist_dir is None:
                missing.append("--conversation-persist-dir")
            if args.workflow_persist_dir is None:
                missing.append("--workflow-persist-dir")

            if missing:
                parser.error(
                    "the following arguments are required unless --empty is set: "
                    + ", ".join(missing)
                )    
    args.func(args)


if __name__ == "__main__":
    main()

# dump app data
# python graph_knowledge_engine\utils\kge_debug_dump.py bundle \
#    bundle
#    --kg-persist-dir C:\Users\chanh\AppData\Local\Temp\pytest-of-chanh\pytest-793\test_workflow_runtime_uses_def0\kg \
#    --conversation-persist-dir C:\Users\chanh\AppData\Local\Temp\pytest-of-chanh\pytest-793\test_workflow_runtime_uses_def0\conv \
#    --workflow-persist-dir C:\Users\chanh\AppData\Local\Temp\pytest-of-chanh\pytest-793\test_workflow_runtime_uses_def0\wf \
#    --template graph_knowledge_engine\templates\d3.html \
#    --out-dir C:\Users\chanh\AppData\Local\Temp\pytest-of-chanh\pytest-793\bundle \
#    --cdc-ws-url ws://127.0.0.1:8787/changes/ws \

# dump a CDC listener one graph
# python graph_knowledge_engine/utils/kge_debug_dump.py \
#   bundle 
#   --persist-dir ./chroma_db \
#   --graph-type knowledge \
#   --template ./d3.html \
#   --out-dir ./empty_cdc_streamer \
#   --empty \
#   --cdc-ws-url ws://127.0.0.1:8787/changes/ws


# python graph_knowledge_engine/utils/kge_debug_dump.py \
#   bundle
#   --template ./d3.html \
#   --out-dir ./empty_cdc_streamer \
#   --empty \
#   --cdc-ws-url ws://127.0.0.1:8787/changes/ws