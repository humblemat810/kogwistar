# mcp_endpoints.py — MCP-style wrapper exposing ingest & query as tools
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

# Robust imports (works both in a package and as flat files)
try:
    from .engine import GraphKnowledgeEngine
    from .graph_query import GraphQuery
    from .ingester import PagewiseSummaryIngestor
    from .models import Document, Edge, Node
except ImportError:
    from engine import GraphKnowledgeEngine
    from graph_query import GraphQuery
    from ingester import PagewiseSummaryIngestor
    from models import Document, Edge, Node


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    handler: Callable[[Dict[str, Any]], Dict[str, Any]]


class KnowledgeMCP:
    """
    Tiny, dependency-free MCP-style dispatcher.

    Tools exposed:
      1) kg.extract_graph              — LLM KG extraction (persist nodes/edges)
      2) doc.parse_summary_tree        — chunk → summarize → group → final summary (persist)
      3) kg.query                      — GraphQuery ops (neighbors/k_hop/…)
      4) doc.query                     — summary-tree helpers (final, per-level, children)
      5) kg.semantic_search            — seed-by-text or by-vector, then k-hop expand
    """

    def __init__(self, engine: GraphKnowledgeEngine, *, ingester_llm: Any | None = None):
        self.engine = engine
        self.query = GraphQuery(engine)
        self._ingester_llm = ingester_llm
        self._tools: Dict[str, ToolSpec] = {}
        self._register_all()

    # ---------- public ----------
    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
                "output_schema": t.output_schema,
            }
            for t in self._tools.values()
        ]

    def call(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}")
        return self._tools[name].handler(args)

    # ---------- registry ----------
    def _register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def _register_all(self) -> None:
        self._register(self._spec_extract_graph())
        self._register(self._spec_parse_summary_tree())
        self._register(self._spec_query_kg())
        self._register(self._spec_query_doc())
        self._register(self._spec_semantic_search())

    # 1) KG extraction (LLM)
    def _spec_extract_graph(self) -> ToolSpec:
        def _handle(args: Dict[str, Any]) -> Dict[str, Any]:
            doc_id: str = args["doc_id"]
            content: str = args["content"]
            page_number: int = int(args.get("page_number", 1))
            auto_adj: bool = bool(args.get("auto_adjudicate", True))

            # Ensure a document row exists, then ingest one "page" of text into KG
            self.engine.add_document(Document(id=doc_id, content=content, type="plain"))
            out = self.engine.add_page(
                document_id=doc_id,
                page_text=content,
                page_number=page_number,
                auto_adjudicate=auto_adj,
            )
            return {"ok": True, "doc_id": doc_id, **(out or {})}

        return ToolSpec(
            name="kg.extract_graph",
            description="Extract nodes/edges from raw text via LLM and persist them under doc_id.",
            input_schema={
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string"},
                    "content": {"type": "string"},
                    "page_number": {"type": "integer", "default": 1},
                    "auto_adjudicate": {"type": "boolean", "default": True},
                },
                "required": ["doc_id", "content"],
            },
            output_schema={"type": "object"},
            handler=_handle,
        )

    # 2) Summary tree (chunks → summaries → groups → final summary)
    def _spec_parse_summary_tree(self) -> ToolSpec:
        def _handle(args: Dict[str, Any]) -> Dict[str, Any]:
            if self._ingester_llm is None:
                raise RuntimeError("KnowledgeMCP was created without ingester_llm.")

            doc_id: str = args["doc_id"]
            content: str = args["content"]
            split_max_chars = int(args.get("split_max_chars", 1200))
            group_size = int(args.get("group_size", 5))
            max_levels = int(args.get("max_levels", 4))
            force_after = int(args.get("force_concat_after_levels", 3))

            ing = PagewiseSummaryIngestor(engine=self.engine, llm=self._ingester_llm)
            res = ing.ingest_document(
                document=Document(id=doc_id, content=content, type="plain"),
                split_max_chars=split_max_chars,
                group_size=group_size,
                max_levels=max_levels,
                force_concat_after_levels=force_after,
            )
            final_id = self.query.final_summary_node_id(doc_id)
            return {
                "ok": True,
                "doc_id": doc_id,
                "final_summary_node_id": final_id,
                **(res or {}),
            }

        return ToolSpec(
            name="doc.parse_summary_tree",
            description="Split → summarize → group into a tree; persist chunk nodes + edges for a document.",
            input_schema={
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string"},
                    "content": {"type": "string"},
                    "split_max_chars": {"type": "integer", "default": 1200},
                    "group_size": {"type": "integer", "default": 5},
                    "max_levels": {"type": "integer", "default": 4},
                    "force_concat_after_levels": {"type": "integer", "default": 3},
                },
                "required": ["doc_id", "content"],
            },
            output_schema={"type": "object"},
            handler=_handle,
        )

    # 3) GraphQuery ops
    def _spec_query_kg(self) -> ToolSpec:
        ALLOWED = {
            "neighbors",
            "k_hop",
            "shortest_path",
            "find_edges",
            "nodes_in_doc",
            "edges_in_doc",
            "document_subgraph",
            "final_summary_node_id",
            "final_summary_node",
            "search_nodes",
            "path_between_labels",
            "adjacency_list",
        }

        def _dispatch(payload: Dict[str, Any]) -> Dict[str, Any]:
            op = payload["op"]
            if op not in ALLOWED:
                raise ValueError(f"Unsupported kg.query op: {op}")
            f = getattr(self.query, op)
            # pass-through of named args (we allow callers to put them top-level)
            kw = {k: v for k, v in payload.items() if k not in {"op", "args"}}
            return {"ok": True, "op": op, "result": _jsonify(f(**kw))}

        return ToolSpec(
            name="kg.query",
            description="Run a GraphQuery operation (neighbors, k_hop, shortest_path, find_edges, …).",
            input_schema={
                "type": "object",
                "properties": {
                    "op": {"type": "string"},
                },
                "required": ["op"],
                "additionalProperties": True,
            },
            output_schema={"type": "object"},
            handler=lambda args: _dispatch({**args, **(args.get("args") or {})}),
        )

    # 4) Summary-tree helpers
    def _spec_query_doc(self) -> ToolSpec:
        def _handle(args: Dict[str, Any]) -> Dict[str, Any]:
            doc_id: str = args["doc_id"]
            what: str = args.get("what", "final_summary_node")

            if what == "final_summary_node":
                node = self.query.final_summary_node(doc_id)
                return {"ok": True, "doc_id": doc_id, "node": _jsonify(node)}

            if what == "level_nodes":
                level = int(args.get("level", 0))
                nodes = self.query.nodes_in_doc(doc_id)
                ids = [
                    n.id
                    for n in nodes
                    if (getattr(n, "properties", {}) or {}).get("level") == level
                ]
                return {"ok": True, "doc_id": doc_id, "level": level, "node_ids": ids}

            if what == "children":
                parent_id = args["parent_id"]
                eids = self.query.find_edges(relation="summarizes", doc_id=doc_id)
                out: List[str] = []
                if eids:
                    ed = self.engine.edge_collection.get(
                        ids=eids, include=["documents"]
                    ).get("documents") or []
                    for raw in ed:
                        e = Edge.model_validate_json(raw)
                        if parent_id in (e.source_ids or []):
                            out.extend(e.target_ids or [])
                return {
                    "ok": True,
                    "doc_id": doc_id,
                    "parent_id": parent_id,
                    "children": sorted(set(out)),
                }

            raise ValueError(f"Unsupported doc.query what={what!r}")

        return ToolSpec(
            name="doc.query",
            description="Helpers over the summary tree: final_summary_node | level_nodes | children",
            input_schema={
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string"},
                    "what": {
                        "type": "string",
                        "enum": ["final_summary_node", "level_nodes", "children"],
                    },
                    "level": {"type": "integer"},
                    "parent_id": {"type": "string"},
                },
                "required": ["doc_id"],
            },
            output_schema={"type": "object"},
            handler=_handle,
        )

    # 5) Semantic search (text or vector) → seed → k-hop
    def _spec_semantic_search(self) -> ToolSpec:
        def _handle(args: Dict[str, Any]) -> Dict[str, Any]:
            top_k = int(args.get("top_k", 5))
            hops = int(args.get("hops", 1))

            seeds: List[str]
            if "text" in args and args["text"]:
                # IMPORTANT: use the store’s default embedding function (no custom embedder)
                hits = self.engine.node_collection.query(
                    query_texts=[str(args["text"])], n_results=top_k
                )
                seeds = [nid for nid in (hits.get("ids") or [[]])[0] if nid]
            elif "embedding" in args and args["embedding"]:
                out = self.query.semantic_seed_then_expand(
                    args["embedding"], top_k=top_k, hops=hops
                )
                return {"ok": True, "seeds": out["seeds"], "layers": _jsonify(out["layers"])}
            else:
                raise ValueError("Provide either 'text' or 'embedding'.")

            layers = self.query.k_hop(seeds, k=hops)
            return {"ok": True, "seeds": seeds, "layers": _jsonify(layers)}

        return ToolSpec(
            name="kg.semantic_search",
            description="Seed by TEXT (default store embeddings) or by VECTOR, then k-hop expand.",
            input_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "embedding": {"type": "array", "items": {"type": "number"}},
                    "top_k": {"type": "integer", "default": 5},
                    "hops": {"type": "integer", "default": 1},
                },
                "oneOf": [
                    {"required": ["text"]},
                    {"required": ["embedding"]},
                ],
            },
            output_schema={"type": "object"},
            handler=_handle,
        )


# ---------- utils ----------
def _jsonify(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, list):
        return [_jsonify(i) for i in x]
    if isinstance(x, set):
        return sorted(list(x))
    if hasattr(x, "model_dump"):
        return x.model_dump()
    if isinstance(x, dict):
        return {k: _jsonify(v) for k, v in x.items()}
    return x